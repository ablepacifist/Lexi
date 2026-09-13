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


# Obrenna answers in markdown — it is the same brain the web UI talks to. Piper
# has no idea markdown exists and pronounces the syntax: measured on aragon,
# "**Tuesday at 7 PM**" takes 4.98 s to speak versus 2.11 s for the same words
# clean, because it says the asterisks out loud. Everything on the way to TTS
# gets stripped.
#
# Deliberately conservative — these only fire on *paired*, well-formed markup,
# so arithmetic ("5 * 3"), a lone underscore in a filename, or an unmatched
# asterisk pass through untouched rather than being silently mangled.


def _emph(delim: str) -> str:
    """Capture group for emphasis content delimited by ``delim``.

    Two rules, both from real markdown, and both load-bearing here:
    the content cannot contain the delimiter, so "**a** **b**" is two spans
    rather than one greedy match; and it cannot start or end with whitespace,
    so spaced-out arithmetic ("a * b * c") is not read as italics.
    """
    return rf"([^\s{delim}][^{delim}\n]*[^\s{delim}]|[^\s{delim}])"


_STAR = _emph("*")
_UNDER = _emph("_")
_TILDE = _emph("~")

_MD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"```.*?```", re.DOTALL), " "),        # fenced code
    (re.compile(r"`([^`\n]+)`"), r"\1"),               # inline code
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),        # images: drop entirely
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),     # links: keep the text
    # Emphasis, built from _emph() below.
    (re.compile(r"\*\*\*" + _STAR + r"\*\*\*"), r"\1"),
    (re.compile(r"\*\*" + _STAR + r"\*\*"), r"\1"),                   # bold
    (re.compile(r"(?<![\w*])\*" + _STAR + r"\*(?![\w*])"), r"\1"),    # italic
    (re.compile(r"(?<![\w_])__" + _UNDER + r"__(?![\w_])"), r"\1"),
    (re.compile(r"(?<![\w_])_" + _UNDER + r"_(?![\w_])"), r"\1"),
    (re.compile(r"~~" + _TILDE + r"~~"), r"\1"),                      # strikethrough
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE), ""),      # headings
    (re.compile(r"^\s{0,3}>\s?", re.MULTILINE), ""),           # blockquote
    (re.compile(r"^\s{0,3}([-*+]|\d+\.)\s+", re.MULTILINE), ""),  # list bullets
    (re.compile(r"^\s{0,3}([-*_])\s*(?:\1\s*){2,}$", re.MULTILINE), " "),  # hr
)


def strip_markdown(text: str) -> str:
    """Flatten markdown to plain prose for TTS.

    Not a parser and not trying to be — it removes the syntax a speech engine
    would otherwise read aloud, and leaves anything ambiguous alone.
    """
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


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
