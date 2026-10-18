"""Build engine instances from config. Construction is cheap (no model load);
the heavy load is deferred to each engine's first use (lazy _ensure)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ..config import EngineConfig
from .stt import WhisperStt
from .stt_remote import RemoteStt
from .tts import PiperTts
from .vad import WebrtcVad
from .wake import OpenWakeWord


@dataclass
class Engines:
    stt: Union[WhisperStt, RemoteStt]  # remote when stt_remote_url is set, else local
    tts: PiperTts
    vad: WebrtcVad
    wake: OpenWakeWord | None


def build_engines(cfg: EngineConfig) -> Engines:
    stt = RemoteStt(cfg) if cfg.stt_remote_url else WhisperStt(cfg)
    return Engines(
        stt=stt,
        tts=PiperTts(cfg),
        vad=WebrtcVad(cfg),
        wake=OpenWakeWord(cfg) if cfg.wake_enabled else None,
    )
