"""Build engine instances from config. Construction is cheap (no model load);
the heavy load is deferred to each engine's first use (lazy _ensure)."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import EngineConfig
from .stt import WhisperStt
from .tts import PiperTts
from .vad import WebrtcVad
from .wake import OpenWakeWord


@dataclass
class Engines:
    stt: WhisperStt
    tts: PiperTts
    vad: WebrtcVad
    wake: OpenWakeWord | None


def build_engines(cfg: EngineConfig) -> Engines:
    return Engines(
        stt=WhisperStt(cfg),
        tts=PiperTts(cfg),
        vad=WebrtcVad(cfg),
        wake=OpenWakeWord(cfg) if cfg.wake_enabled else None,
    )
