"""Audio helpers: streaming sentence segmentation (pure logic) and mic/playback.

The sentence chunker is the latency trick — it emits a chunk as soon as a
sentence boundary arrives in the token stream, so Piper can start speaking the
first sentence while the model is still generating the rest.

Mic capture / playback use ``sounddevice`` and are imported lazily so this
module (and the sentence chunker) import fine on a machine without an audio
stack or PortAudio.
"""
from __future__ import annotations

import re

# End-of-sentence punctuation followed by whitespace, or a newline. Kept simple
# and language-agnostic on purpose; refine per-language later if needed.
_BOUNDARY = re.compile(r"([.!?])(\s+)|(\n+)")
# Don't hold a partial sentence forever if the model never emits terminal
# punctuation — flush once the buffer exceeds this many characters at a space.
_SOFT_FLUSH_CHARS = 180


class SentenceChunker:
    """Feed streamed token text; get back complete sentences to synthesize."""

    def __init__(self, soft_flush_chars: int = _SOFT_FLUSH_CHARS):
        self._buf = ""
        self._soft = soft_flush_chars

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return zero or more ready-to-speak sentences."""
        self._buf += text
        out: list[str] = []
        while True:
            m = _BOUNDARY.search(self._buf)
            if m:
                end = m.end()
                chunk = self._buf[:end].strip()
                self._buf = self._buf[end:]
                if chunk:
                    out.append(chunk)
                continue
            # No hard boundary — consider a soft flush at the last space.
            if len(self._buf) >= self._soft:
                cut = self._buf.rfind(" ", 0, self._soft)
                if cut > 0:
                    chunk = self._buf[:cut].strip()
                    self._buf = self._buf[cut:].lstrip()
                    if chunk:
                        out.append(chunk)
                        continue
            break
        return out

    def flush(self) -> str | None:
        """Return whatever remains at end of stream (may lack punctuation)."""
        rest = self._buf.strip()
        self._buf = ""
        return rest or None


# ── Mic / playback (lazy — needs sounddevice + PortAudio) ────────────────────

def _require_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        return sd, np
    except Exception as exc:  # pragma: no cover - hardware/dep dependent
        raise RuntimeError(
            "Audio I/O needs 'sounddevice' + 'numpy' and a working PortAudio. "
            "Install Lexi's audio extra and run on the aragon device."
        ) from exc


def record_until_silence(
    sample_rate: int,
    vad,
    max_seconds: float = 15.0,
    frame_ms: int = 30,
    trailing_silence_ms: int = 800,
) -> bytes:
    """Capture mono PCM from the mic until the VAD reports trailing silence.

    webrtcvad needs 10/20/30 ms mono int16 frames — frame_ms must be one of
    those. Returns int16 PCM bytes (empty if nothing was ever detected as
    speech within max_seconds).
    """
    sd, np = _require_sounddevice()  # pragma: no cover - hardware dependent
    frame_len = int(sample_rate * frame_ms / 1000)  # samples per frame
    silence_frames_needed = trailing_silence_ms // frame_ms
    collected: list[bytes] = []
    started = False
    silence = 0
    with sd.RawInputStream(
        samplerate=sample_rate, channels=1, dtype="int16", blocksize=frame_len
    ) as stream:
        max_frames = int(max_seconds * 1000 / frame_ms)
        for _ in range(max_frames):
            data, _ = stream.read(frame_len)
            frame = bytes(data)
            speech = vad.is_speech(frame, sample_rate)
            if speech:
                started, silence = True, 0
                collected.append(frame)
            elif started:
                silence += 1
                collected.append(frame)
                if silence >= silence_frames_needed:
                    break
    return b"".join(collected)


def play_pcm(pcm: bytes, sample_rate: int) -> None:
    sd, np = _require_sounddevice()  # pragma: no cover - hardware dependent
    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, sample_rate)
    sd.wait()
