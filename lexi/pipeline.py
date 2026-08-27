"""The voice turn pipeline — the logic that deliberately lives on aragon.

    wake → VAD → STT → resolve identity → send text to brain
         → sentence-chunk the streamed answer → TTS → speaker

Only ``run_text_turn`` is exercised in tests (it needs no hardware): it drives
the brain client and the sentence chunker and hands each finished sentence to a
sink. ``run_voice_turn`` / ``listen_loop`` add the mic + engines on top and run
on the aragon device.
"""
from __future__ import annotations

import logging
from typing import Callable

from . import audio
from .config import LexiConfig
from .engines.registry import Engines
from .identity import IdentityResolver
from .obrenna_client import ObrennaClient
from .session import State, VoiceSession

logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        cfg: LexiConfig,
        client: ObrennaClient,
        identity: IdentityResolver,
        engines: Engines | None = None,
        session: VoiceSession | None = None,
    ):
        self._cfg = cfg
        self._client = client
        self._identity = identity
        self._engines = engines
        self.session = session or VoiceSession(account_id=identity.account_id())

    # ── core, hardware-free (tested) ─────────────────────────────────────────
    def run_text_turn(
        self,
        text: str,
        *,
        speak: bool = True,
        on_sentence: Callable[[str], None] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> str:
        """Send transcript text to the brain, sentence-chunk the streamed answer,
        and (optionally) speak each sentence as it completes. Returns full text."""
        self.session.set_state(State.THINKING)
        chunker = audio.SentenceChunker()

        def handle_sentence(sentence: str) -> None:
            if on_sentence:
                on_sentence(sentence)
            if speak:
                self.session.set_state(State.SPEAKING)
                self.speak_sentence(sentence)

        def on_token(tok: str) -> None:
            for sentence in chunker.feed(tok):
                handle_sentence(sentence)

        result = self._client.stream_turn(
            text,
            account_id=self.session.account_id,
            chat_id=self.session.chat_id,
            on_token=on_token,
            on_event=on_event,
        )
        tail = chunker.flush()
        if tail:
            handle_sentence(tail)

        self.session.chat_id = result.chat_id or self.session.chat_id
        self.session.set_state(State.IDLE)
        return result.text

    def speak_sentence(self, sentence: str) -> None:  # pragma: no cover - hardware
        if not self._engines:
            raise RuntimeError("No engines configured; cannot speak.")
        for pcm in self._engines.tts.synthesize(sentence):
            audio.play_pcm(pcm, self._engines.tts.sample_rate)

    # ── mic-driven (aragon device) ───────────────────────────────────────────
    def run_voice_turn(self, *, speak: bool = True) -> str:  # pragma: no cover - hardware
        if not self._engines:
            raise RuntimeError("No engines configured; cannot capture/transcribe.")
        self.session.set_state(State.LISTENING)
        pcm = audio.record_until_silence(self._cfg.engines.sample_rate, self._engines.vad)
        transcript = self._engines.stt.transcribe(pcm, self._cfg.engines.sample_rate)
        logger.info("Transcript: %s", transcript)
        if not transcript.strip():
            self.session.set_state(State.IDLE)
            return ""
        return self.run_text_turn(transcript, speak=speak)

    def listen_loop(self) -> None:  # pragma: no cover - hardware
        """Wake-word gated loop: wait for the wake word, then handle one turn."""
        if not self._engines:
            raise RuntimeError("No engines configured; cannot listen.")
        import sounddevice as sd  # noqa: PLC0415

        sr = self._cfg.engines.sample_rate
        wake = self._engines.wake
        frame_len = int(sr * 0.08)  # 80 ms, openWakeWord's expected frame size
        with sd.RawInputStream(
            samplerate=sr, channels=1, dtype="int16", blocksize=frame_len
        ) as stream:
            logger.info("Listening for wake word...")
            while True:
                data, _ = stream.read(frame_len)
                if wake is None or wake.detect(bytes(data)):
                    self.run_voice_turn(speak=True)
