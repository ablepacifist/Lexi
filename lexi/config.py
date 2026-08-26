"""Lexi configuration — loaded from a local TOML file, no secrets in code.

Resolution order for the config path:
  1. ``$LEXI_CONFIG`` if set,
  2. ``./config.toml`` in the current working directory,
  3. ``config.example.toml`` shipped with the repo (safe defaults; engines only).

The shared secret for the Obrenna gateway is read from the environment
(``$OBRENNA_AGENT_TOKEN``) or a token file, never from the committed config —
same discipline the gateway's auth service uses.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BrainConfig:
    """How to reach Obrenna's brain on alison, through its gateway."""
    # Tried in order: LAN first, then the public tunnel (decision: LAN primary,
    # tunnel fallback).
    lan_base_url: str = "http://alison.lan:9080"
    tunnel_base_url: str = "https://llm.alex-dyakin.com"
    # Per-request orchestrator slug — a small/fast model for voice. Sent as
    # ChatRequest.orchestrator so text chat can keep its bigger model.
    orchestrator: str = "qwen3.5-4b-claude-opus-reasoning-distilled-v2"
    workers_enabled: bool = False
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 3.0

    @property
    def bases(self) -> list[str]:
        return [b for b in (self.lan_base_url, self.tunnel_base_url) if b]


@dataclass
class IdentityConfig:
    """Per-Lexicon-user identity. Lexi resolves who is speaking to an account_id
    that the brain scopes memory by."""
    # Lexicon base URL (localhost on aragon). Used to resolve the current user.
    lexicon_base_url: str = "http://localhost:36568"
    # Fallback account_id when no Lexicon session is available (e.g. a shared
    # device). Real per-speaker identity (enrollment/diarization) is future work.
    default_account_id: str = "local-default"


@dataclass
class EngineConfig:
    stt_model: str = "base"          # faster-whisper size or local path
    stt_device: str = "cpu"          # cpu | cuda
    stt_compute_type: str = "int8"
    stt_language: str | None = None  # None = autodetect
    tts_voice: str = "en_US-lessac-medium"   # Piper voice name / .onnx path
    wake_enabled: bool = True
    wake_model: str = "hey_jarvis"   # openWakeWord model name / path
    wake_threshold: float = 0.5
    vad_aggressiveness: int = 2      # webrtcvad 0..3
    sample_rate: int = 16000
    models_dir: str = "./models"


@dataclass
class LexiConfig:
    brain: BrainConfig = field(default_factory=BrainConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    engines: EngineConfig = field(default_factory=EngineConfig)
    # Where the gateway shared-secret comes from (env wins over file).
    agent_token_env: str = "OBRENNA_AGENT_TOKEN"
    agent_token_file: str = "./.agent_token"

    def agent_token(self) -> str:
        env = os.getenv(self.agent_token_env, "").strip()
        if env:
            return env
        try:
            return Path(self.agent_token_file).read_text(encoding="utf-8").strip()
        except OSError:
            return ""


def _config_path() -> Path | None:
    env = os.getenv("LEXI_CONFIG", "").strip()
    if env:
        return Path(env)
    for candidate in (Path("config.toml"), Path(__file__).resolve().parent.parent / "config.example.toml"):
        if candidate.exists():
            return candidate
    return None


def _section(data: dict, name: str) -> dict:
    section = data.get(name, {})
    return section if isinstance(section, dict) else {}


def load_config(path: str | os.PathLike | None = None) -> LexiConfig:
    """Load config from TOML, falling back to dataclass defaults for anything
    the file omits."""
    p = Path(path) if path else _config_path()
    data: dict = {}
    if p and p.exists():
        with open(p, "rb") as fh:
            data = tomllib.load(fh)

    cfg = LexiConfig(
        brain=BrainConfig(**_section(data, "brain")),
        identity=IdentityConfig(**_section(data, "identity")),
        engines=EngineConfig(**_section(data, "engines")),
    )
    top = {k: v for k, v in data.items() if k in ("agent_token_env", "agent_token_file")}
    for k, v in top.items():
        setattr(cfg, k, v)
    return cfg
