"""Remote speech-to-text: POST the captured utterance to a `lexi-stt` service
(e.g. aragon on the LAN) and get the transcript back, so the heavy faster-whisper
work runs off the Pi.

Falls back to a local ``WhisperStt`` when the remote is unreachable — voice must
never hard-depend on another machine being up (aragon is often started by hand).
Duck-compatible with ``WhisperStt``: it exposes ``transcribe(pcm, sample_rate)``,
which is all ``pipeline.run_voice_turn`` calls.
"""
from __future__ import annotations

import logging

import httpx

from .. import config
from ..config import EngineConfig

logger = logging.getLogger(__name__)


class RemoteStt:
    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._url = cfg.stt_remote_url.rstrip("/")
        self._token = config.read_token("LEXI_TOOL_TOKEN", "./.tool_token")
        self._local = None            # lazy local WhisperStt (fallback)
        self._in_fallback = False     # so we log the transition once, not per turn

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/octet-stream"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        try:
            resp = httpx.post(
                f"{self._url}/stt",
                params={"sample_rate": sample_rate},
                content=pcm,
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            transcript = (resp.json().get("transcript") or "").strip()
        except Exception as exc:  # network / timeout / bad status / bad json
            if self._cfg.stt_remote_fallback_local:
                if not self._in_fallback:
                    logger.warning("remote STT %s failed (%s); using local Whisper",
                                   self._url, exc)
                    self._in_fallback = True
                return self._local_transcribe(pcm, sample_rate)
            logger.warning("remote STT %s failed and local fallback is off: %s",
                           self._url, exc)
            return ""
        if self._in_fallback:
            logger.info("remote STT %s recovered", self._url)
            self._in_fallback = False
        return transcript

    def _local_transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if self._local is None:
            from .stt import WhisperStt  # noqa: PLC0415 - lazy: only if we ever fall back
            self._local = WhisperStt(self._cfg)
        return self._local.transcribe(pcm, sample_rate)
