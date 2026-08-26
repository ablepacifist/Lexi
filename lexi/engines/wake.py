"""Wake word via openWakeWord (local ONNX). Lazy import.

``detect`` takes one frame (int16 PCM) and returns True when the configured
wake word's score crosses the threshold.
"""
from __future__ import annotations

import logging

from ..config import EngineConfig

logger = logging.getLogger(__name__)


class OpenWakeWord:
    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from openwakeword.model import Model  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - dep dependent
            raise RuntimeError(
                "openwakeword is not installed. Install Lexi's engine deps on "
                "the aragon device (see requirements.txt)."
            ) from exc
        logger.info("Loading wake word model=%s", self._cfg.wake_model)
        self._model = Model(wakeword_models=[self._cfg.wake_model])

    def detect(self, frame: bytes) -> bool:  # pragma: no cover - model dependent
        self._ensure()
        import numpy as np  # noqa: PLC0415
        samples = np.frombuffer(frame, dtype=np.int16)
        scores = self._model.predict(samples)
        return any(score >= self._cfg.wake_threshold for score in scores.values())
