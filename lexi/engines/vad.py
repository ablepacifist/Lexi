"""Voice-activity detection via webrtcvad (local, tiny). Lazy import.

webrtcvad needs 10/20/30 ms mono int16 frames at 8/16/32/48 kHz. The caller is
responsible for framing; ``is_speech`` classifies one frame.
"""
from __future__ import annotations

from ..config import EngineConfig


class WebrtcVad:
    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._vad = None

    def _ensure(self):
        if self._vad is not None:
            return
        try:
            import webrtcvad  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - dep dependent
            raise RuntimeError(
                "webrtcvad is not installed. Install Lexi's engine deps on the "
                "aragon device (see requirements.txt)."
            ) from exc
        self._vad = webrtcvad.Vad(self._cfg.vad_aggressiveness)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:  # pragma: no cover - dep dependent
        self._ensure()
        return self._vad.is_speech(frame, sample_rate)
