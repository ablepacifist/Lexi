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
    # Tried in order: tunnel first (works off-site, no LAN dependency), then
    # LAN as fallback (decision: tunnel primary, LAN fallback).
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
        return [b for b in (self.tunnel_base_url, self.lan_base_url) if b]


@dataclass
class IdentityConfig:
    """Per-Lexicon-user identity. Lexi resolves who is speaking to an account_id
    that the brain scopes memory by."""
    # Lexicon base URL. On aragon's own box this is localhost; on the Pi/other
    # LAN devices, aragon's LAN IP (e.g. http://192.168.1.4:36568).
    lexicon_base_url: str = "http://localhost:36568"
    # Fallback Lexicon URL tried when the primary is unreachable — the Cloudflare
    # tunnel (https://api.alex-dyakin.com), so media still works off-LAN. Blank =
    # no fallback. Mirrors the brain client's LAN-primary/tunnel-fallback pattern.
    lexicon_fallback_url: str = ""
    # Fallback account_id when no Lexicon session is available (e.g. a shared
    # device). Real per-speaker identity (enrollment/diarization) is future work.
    default_account_id: str = "local-default"
    # Lexicon credentials Lexi logs in with for the media library (music
    # playback) and identity. Kept in config.toml (gitignored) or env, never
    # committed. Leave blank to disable media playback.
    lexicon_username: str = ""
    lexicon_password: str = ""


@dataclass
class EngineConfig:
    stt_model: str = "base"          # faster-whisper size or local path
    stt_device: str = "cpu"          # cpu | cuda
    stt_compute_type: str = "int8"
    stt_language: str | None = None  # None = autodetect
    # Offload STT to a remote lexi-stt service (e.g. aragon: "http://192.168.1.4:8765").
    # Empty = transcribe locally with faster-whisper (default). When set, the Pi POSTs
    # the utterance there; stt_model above is then the LOCAL FALLBACK model.
    stt_remote_url: str = ""
    stt_remote_fallback_local: bool = True  # on remote failure, transcribe locally
    tts_voice: str = "en_US-lessac-medium"   # Piper voice name / .onnx path
    wake_enabled: bool = True
    wake_model: str = "hey_jarvis"   # openWakeWord model name / path
    wake_threshold: float = 0.5
    vad_aggressiveness: int = 2      # webrtcvad 0..3
    sample_rate: int = 16000
    models_dir: str = "./models"
    # mpv --audio-device for music playback. Empty = let mpv choose. On the Pi,
    # route through PipeWire ("pipewire") so music and TTS mix; a raw ALSA device
    # like "alsa/plughw:CARD=Headphones" is single-open and cannot mix with TTS.
    mpv_audio_device: str = ""
    # True only when the OS mixes mpv + TTS on the same output (e.g. PipeWire on
    # the Pi). Then Lexi PAUSES music on wake and speaks over it, resuming after.
    # False (safe default) = music is STOPPED before a spoken chat reply, because
    # a paused mpv would otherwise hold an exclusive device and mute TTS.
    audio_shared: bool = False


@dataclass
class HistoryConfig:
    """Local transcript log of every voice turn, pruned by age. Private to the
    device (gitignored); a debug/transcript trail, not the brain's memory."""
    enabled: bool = True
    path: str = "./logs/turns.jsonl"
    ttl_days: int = 14


@dataclass
class LexiConfig:
    brain: BrainConfig = field(default_factory=BrainConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    engines: EngineConfig = field(default_factory=EngineConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    # Where the gateway shared-secret comes from (env wins over file).
    agent_token_env: str = "OBRENNA_AGENT_TOKEN"
    agent_token_file: str = "./.agent_token"
    # The Lexi tool token guards the lexi-stt / lexi-server HTTP hops (Pi ↔ aragon).
    # Same env-then-file discipline as the brain token, under its own name so the
    # two rotate apart.
    tool_token_env: str = "LEXI_TOOL_TOKEN"
    tool_token_file: str = "./.tool_token"

    def agent_token(self) -> str:
        return read_token(self.agent_token_env, self.agent_token_file)

    def tool_token(self) -> str:
        return read_token(self.tool_token_env, self.tool_token_file)


def read_token(env_name: str, file_path: str) -> str:
    """A shared secret
    Used for both the brain agent token and the lexi tool token."""
    env = os.getenv(env_name, "").strip()
    if env:
        return env
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
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
        history=HistoryConfig(**_section(data, "history")),
    )
    top = {k: v for k, v in data.items()
           if k in ("agent_token_env", "agent_token_file",
                    "tool_token_env", "tool_token_file")}
    for k, v in top.items():
        setattr(cfg, k, v)
    return cfg
