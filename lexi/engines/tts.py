"""Text-to-speech via Piper (local, CPU). Streams int16 PCM per call.

Lazy import so the package loads without piper installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from ..config import EngineConfig

logger = logging.getLogger(__name__)


class PiperTts:
    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._voice = None
        self._sample_rate = cfg.sample_rate

    def _ensure(self):
        if self._voice is not None:
            return
        try:
            from piper import PiperVoice  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - dep dependent
            raise RuntimeError(
                "piper-tts is not installed. Install Lexi's engine deps on the "
                "aragon device (see requirements.txt)."
            ) from exc
        model_path = self._resolve_voice_path()
        logger.info("Loading Piper voice=%s", model_path)
        self._voice = PiperVoice.load(str(model_path))
        self._sample_rate = getattr(self._voice.config, "sample_rate", self._cfg.sample_rate)

    def _resolve_voice_path(self) -> Path:
        v = self._cfg.tts_voice
        if v.endswith(".onnx"):
            return Path(v)
        return Path(self._cfg.models_dir) / "piper" / f"{v}.onnx"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[bytes]:  # pragma: no cover - model dependent
        self._ensure()
        # Piper yields audio chunks; expose raw int16 PCM bytes to the caller.
        for chunk in self._voice.synthesize_stream_raw(text):
            yield chunk
