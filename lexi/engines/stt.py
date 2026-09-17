"""Speech-to-text via faster-whisper (local, CPU int8 by default).

Lazy import: the heavy dependency and model load only happen the first time
``transcribe`` runs, so importing Lexi on a machine without faster-whisper (e.g.
alison during brain-side dev) still works.
"""
from __future__ import annotations

import logging

from ..config import EngineConfig

logger = logging.getLogger(__name__)


class WhisperStt:
    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - dep dependent
            raise RuntimeError(
                "faster-whisper is not installed. Install Lexi's engine deps on "
                "the aragon device (see requirements.txt)."
            ) from exc
        logger.info("Loading Whisper model=%s device=%s compute=%s",
                    self._cfg.stt_model, self._cfg.stt_device, self._cfg.stt_compute_type)
        self._model = WhisperModel(
            self._cfg.stt_model,
            device=self._cfg.stt_device,
            compute_type=self._cfg.stt_compute_type,
            download_root=self._cfg.models_dir,
        )

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:  # pragma: no cover - model dependent
        self._ensure()
        import numpy as np  # noqa: PLC0415
        # faster-whisper wants float32 in [-1, 1].
        audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.stt_language,
            # Trim internal silence: even after our VAD endpointing, a clip can
            # carry trailing/leading quiet that Whisper "fills" with hallucinated
            # filler ("Hi there…", "I will let you know"). vad_filter drops it.
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_encoded(self, data: bytes) -> str:  # pragma: no cover - model dependent
        """Transcribe a compressed clip (WebM/Opus, MP4/AAC, WAV, …).

        Browsers hand back whatever their MediaRecorder produces — Opus in a
        WebM container on Chrome and Firefox, AAC in MP4 on Safari — so the
        container is not knowable in advance. faster-whisper already depends on
        PyAV, which decodes and resamples all of them, so this needs no ffmpeg
        binary on the host.
        """
        self._ensure()
        import io  # noqa: PLC0415
        from faster_whisper.audio import decode_audio  # noqa: PLC0415

        audio = decode_audio(io.BytesIO(data), sampling_rate=self._cfg.sample_rate)
        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.stt_language,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()
