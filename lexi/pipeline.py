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
from .history import ConversationLog
from .identity import IdentityResolver
from .intents import match_media_intent
from .media import MediaError, MediaPlayer
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
        media: MediaPlayer | None = None,
        history: ConversationLog | None = None,
    ):
        self._cfg = cfg
        self._client = client
        self._identity = identity
        self._engines = engines
        self._media = media
        self._history = history
        self.session = session or VoiceSession(account_id=identity.account_id())

    def _log(self, transcript: str, kind: str, detail: dict) -> None:
        if self._history is not None:
            self._history.append(
                transcript, kind, detail,
                chat_id=self.session.chat_id,
                account_id=self.session.account_id,
            )

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
        # Strip markdown here rather than in the chunker: the raw text still
        # reaches on_sentence/on_event callers (a UI may want the formatting),
        # but nothing with syntax in it ever reaches the speech engine.
        spoken = audio.strip_markdown(sentence)
        if not spoken:
            return
        for pcm in self._engines.tts.synthesize(spoken):
            audio.play_pcm(pcm, self._engines.tts.sample_rate)

    # ── mic-driven (aragon device) ───────────────────────────────────────────
    def run_voice_turn(self, *, speak: bool = True) -> str:  # pragma: no cover - hardware
        if not self._engines:
            raise RuntimeError("No engines configured; cannot capture/transcribe.")
        # The wake word just fired. Stop any music first: the 3.5mm jack is a
        # single-open device (freeing it lets us speak the reply) and it unmasks
        # the mic so the command is captured cleanly. (Duck-and-resume is a
        # future refinement.)
        if self._media is not None and self._media.is_playing:
            self._media.stop()
        self.session.set_state(State.LISTENING)
        pcm = audio.record_until_silence(self._cfg.engines.sample_rate, self._engines.vad)
        transcript = self._engines.stt.transcribe(pcm, self._cfg.engines.sample_rate)
        logger.info("Transcript: %s", transcript)
        if not transcript.strip():
            self.session.set_state(State.IDLE)
            return ""
        # Media commands ("play …", "stop") are handled locally — no brain, so
        # music works even when alison is offline.
        if self._media is not None and self._media.enabled:
            intent = match_media_intent(transcript)
            if intent is not None:
                label = self._handle_media_intent(intent, speak=speak)
                self._log(transcript, "media", {
                    "action": intent.get("action"),
                    "query": intent.get("query"),
                    "shuffle": intent.get("shuffle"),
                    "result": label,
                })
                return label
        reply = self.run_text_turn(transcript, speak=speak)
        self._log(transcript, "chat", {"reply": reply})
        return reply

    def _handle_media_intent(self, intent: dict, *, speak: bool) -> str:  # pragma: no cover - hardware
        """Play/stop music (single track or playlist, optionally shuffled) from
        Lexicon on the Pi. The 3.5mm output is single-open, so we speak the
        confirmation BEFORE starting mpv."""
        assert self._media is not None
        self._media.stop()
        action = intent.get("action")
        logger.info("media intent=%s query=%r shuffle=%s",
                    action, intent.get("query"), intent.get("shuffle"))

        def say(msg: str) -> None:
            if speak and msg:
                self.speak_sentence(msg)

        def done(ret: str = "") -> str:
            self.session.set_state(State.IDLE)
            return ret

        if action == "stop":
            say("Okay, stopped.")
            return done()

        if action == "next":
            say("I can't skip tracks yet.")
            return done()

        if action == "play_playlist":
            try:
                if intent.get("all"):
                    ids, label = self._media.all_music_ids(), "your music"
                else:
                    name = intent.get("query") or ""
                    pl = self._media.find_playlist(name)
                    if not pl:
                        say(f"I couldn't find a playlist called {name}.")
                        return done()
                    ids = self._media.playlist_media_ids(pl.get("id"))
                    label = pl.get("name") or name
            except MediaError as exc:
                logger.warning("playlist lookup failed: %s", exc)
                say("Sorry, I couldn't reach the music library.")
                return done()
            if not ids:
                say(f"There's nothing to play in {label}.")
                return done()
            shuffle = bool(intent.get("shuffle"))
            say(f"{'Shuffling' if shuffle else 'Playing'} {label}.")
            try:
                self._media.play_many(ids, shuffle=shuffle)
            except MediaError as exc:
                logger.warning("play_many failed: %s", exc)
                say("Sorry, I couldn't play that.")
            return done(label)

        # action == "play" (single track): exact search, then fuzzy fallback for
        # when Whisper mis-hears the title.
        query = intent.get("query", "")
        try:
            chosen = self._media.pick(self._media.search(query)) or self._media.fuzzy_find(query)
        except MediaError as exc:
            logger.warning("media search failed: %s", exc)
            say("Sorry, I couldn't reach the music library.")
            return done()
        if not chosen or chosen.get("id") is None:
            say(f"I couldn't find {query} in your library.")
            return done()
        title = chosen.get("title") or query
        say(f"Playing {title}.")
        try:
            self._media.play(int(chosen["id"]))
        except MediaError as exc:
            logger.warning("media play failed: %s", exc)
            say("Sorry, I couldn't play that.")
        return done(title)

    def listen_loop(self) -> None:  # pragma: no cover - hardware
        """Wake-word gated loop: wait for the wake word, then handle one turn.

        The wake-listen stream is opened and CLOSED around each detection so that
        ``run_voice_turn`` can open the mic itself. A single-capture USB mic (the
        M3) cannot be opened twice at once — doing so is PortAudio error -9985
        ("Device unavailable"). After the turn we reset the wake model so audio
        left in its buffer can't immediately re-trigger it.
        """
        if not self._engines:
            raise RuntimeError("No engines configured; cannot listen.")
        import sounddevice as sd  # noqa: PLC0415

        sr = self._cfg.engines.sample_rate
        wake = self._engines.wake
        frame_len = int(sr * 0.08)  # 80 ms, openWakeWord's expected frame size
        logger.info("Listening for wake word...")
        while True:
            detected = False
            with sd.RawInputStream(
                samplerate=sr, channels=1, dtype="int16", blocksize=frame_len
            ) as stream:
                while True:
                    data, _ = stream.read(frame_len)
                    if wake is None or wake.detect(bytes(data)):
                        detected = True
                        break
            # Stream closed here → the mic is free for run_voice_turn's capture.
            if detected:
                logger.info("Wake word detected.")
                self.run_voice_turn(speak=True)
                if wake is not None:
                    wake.reset()
                logger.info("Listening for wake word...")
