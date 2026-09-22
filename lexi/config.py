"""Lexi configuration — loaded from a local TOML file, no secrets in code.

Resolution order for the config path:
  1. ``$LEXI_CONFIG`` if set,
  2. ``./config.toml`` in the current working directory,
  3. ``config.example.toml`` shipped with the repo (safe defaults; engines only).

Destinations (brain URLs, the Lexicon base URL, the bind host/port, the tool
token) are a separate, higher-priority layer on top of that TOML: Lexi's own
``.env`` and the monorepo's root ``.env`` (the "one destination registry" —
see ``full-back-end-server/.env.example``). Lowest to highest priority:

  1. dataclass defaults below / ``config.toml`` (engine settings only — it
     carries no destinations any more)
  2. Lexi's own ``./.env`` (git-ignored) — so Lexi runs standalone
  3. the monorepo root ``.env`` — found via ``$MASTER_ENV_FILE``, else by
     walking up from this module's directory for a ``.env`` that contains a
     ``LAN_HOST=`` line; a lone Lexi checkout with no such file silently
     stays on layers 1-2
  4. real process environment variables (highest)

``_layered_env()`` builds a small dict from layers 2-3 only — it never
mutates ``os.environ``. Layer 4 (the real environment) is checked first at
every lookup (``env_value``), so it wins without that dict ever needing to
hold it: real env > master ``.env`` > Lexi's own ``.env``. The composed
destination fields (``brain.lan_base_url``, ``brain.tunnel_base_url``,
``identity.lexicon_base_url``, ``identity.lexicon_fallback_url``,
``engines.stt_remote_url``) are set from that layered env AFTER the TOML
load, and only when the corresponding key is actually present somewhere in
layers 2-4 — so a stale ``config.toml`` destination never survives once the
registry defines the key, but an absent key leaves the TOML/dataclass value
alone.

The shared secret for the Obrenna gateway is read from the environment
(``$OBRENNA_AGENT_TOKEN``) or a token file, never from the committed config —
same discipline the gateway's auth service uses. ``LEXI_TOOL_TOKEN`` (the
Lexi<->LexiconServer secret) additionally goes through the layered env above
before falling back to its token file, since it IS a root-registry key.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Lexi/ — this module's package is Lexi/lexi/, so its parent is the module root.
_MODULE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class BrainConfig:
    """How to reach Obrenna's brain on alison, through its gateway.

    Both URLs are destination-registry values: ``load_config()`` composes
    ``lan_base_url`` from ``ALISON_LAN_IP`` + ``BRAIN_LAN_PORT`` and sets
    ``tunnel_base_url`` from ``PUBLIC_BRAIN_URL`` whenever those env keys are
    present, overriding whatever config.toml/the dataclass default holds.
    """
    # Tried in order: tunnel first (works off-site, no LAN dependency), then
    # LAN as fallback (decision: tunnel primary, LAN fallback).
    # Blank = "no LAN fallback configured" (e.g. a standalone checkout with no
    # ALISON_LAN_IP/BRAIN_LAN_PORT set) — never an invented LAN address.
    lan_base_url: str = ""
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
    # Lexicon base URL. Composed as http://127.0.0.1:${LEXICON_PORT} whenever
    # LEXICON_PORT is set (loopback: Lexicon runs on this same box). The
    # default below is the same loopback shape for a standalone checkout.
    lexicon_base_url: str = "http://127.0.0.1:36568"
    # Fallback Lexicon URL tried when the primary is unreachable — set from
    # $PUBLIC_LEXICON_URL, the Cloudflare tunnel, so media still works off-LAN.
    # Blank = no fallback. Mirrors the brain client's LAN-primary/tunnel-fallback
    # pattern.
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
    # Set from $LEXI_STT_REMOTE_URL when present (used on the Pi). Empty =
    # transcribe locally with faster-whisper (default). When set, the Pi POSTs
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
    # The layered env dict (Lexi's own .env < the root master .env) this
    # config was built with, stashed by load_config(). A bare LexiConfig()
    # (e.g. in tests) has an empty dict here and tool_token() computes the
    # layered env fresh on first use instead.
    _env: dict = field(default_factory=dict, repr=False, compare=False)

    def agent_token(self) -> str:
        return read_token(self.agent_token_env, self.agent_token_file)

    def tool_token(self) -> str:
        return read_layered_token(self._env or _layered_env(), self.tool_token_env, self.tool_token_file)


def read_token(env_name: str, file_path: str) -> str:
    """A shared secret, real env var wins over a token file.
    Used for the brain agent token, which is NOT a root-registry key."""
    env = os.getenv(env_name, "").strip()
    if env:
        return env
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def read_layered_token(env: dict[str, str], env_name: str, file_path: str) -> str:
    """Like ``read_token``, but the env layer is Lexi's full layered registry
    (real process env > master ``.env`` > Lexi's own ``.env``) rather than only
    the real process environment — used for LEXI_TOOL_TOKEN, which the root
    registry also defines, so a value set only in the root ``.env`` now takes
    effect. Falls back to the token file when unset in every layer."""
    val = env_value(env, env_name)
    if val:
        return val
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def tool_token() -> str:
    """Convenience for callers that only have an ``EngineConfig``, not the
    full ``LexiConfig`` (e.g. ``RemoteStt``): resolves LEXI_TOOL_TOKEN through
    the same layered registry as ``LexiConfig.tool_token()``, using its
    default env name/file path."""
    return read_layered_token(_layered_env(), "LEXI_TOOL_TOKEN", "./.tool_token")


# ── layered .env registry (tiny stdlib-only parser, no new dependency) ──────

def _parse_dotenv(path: Path) -> dict[str, str]:
    """KEY=value lines; blanks and ``#`` comments are skipped, the value is
    split on the first ``=``, optional surrounding quotes are stripped. A
    missing or unreadable file yields an empty dict — never an error."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _find_master_env(module_dir: Path) -> Path | None:
    """Locate the monorepo root ``.env`` (the destination-config registry).

    ``$MASTER_ENV_FILE`` wins outright if set. Otherwise walk up from the
    module's parent directory looking for a ``.env`` that contains a
    ``LAN_HOST=`` line (the master file's fingerprint) — a lone Lexi checkout
    with no such ancestor silently runs on its own defaults."""
    env_override = os.getenv("MASTER_ENV_FILE", "").strip()
    if env_override:
        return Path(env_override)
    current = module_dir.parent
    for _ in range(12):  # bounded walk-up, never infinite
        candidate = current / ".env"
        if candidate.is_file():
            try:
                if "LAN_HOST=" in candidate.read_text(encoding="utf-8"):
                    return candidate
            except OSError:
                pass
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _layered_env(module_dir: Path | None = None) -> dict[str, str]:
    """The two file layers merged (Lexi's own ``.env`` lowest, the root
    master ``.env`` over it). Real process env is deliberately NOT folded in
    here — see ``env_value`` — so this dict is never used to mutate
    ``os.environ``.

    ``module_dir`` is resolved against the *module-level* ``_MODULE_DIR`` at
    call time (not baked in as a default), so tests can monkeypatch
    ``lexi.config._MODULE_DIR`` and have every caller (``load_config()``,
    ``bind_defaults()``, ``tool_token()``) pick it up."""
    module_dir = module_dir if module_dir is not None else _MODULE_DIR
    merged = _parse_dotenv(module_dir / ".env")
    master = _find_master_env(module_dir)
    if master:
        merged.update(_parse_dotenv(master))
    return merged


def env_value(env: dict[str, str], key: str) -> str | None:
    """Look up ``key`` with the full precedence: real process env > master
    ``.env`` > Lexi's own ``.env``. Returns ``None`` if unset (blank) in every
    layer, so the caller keeps whatever toml/dataclass value it already has."""
    real = os.environ.get(key)
    if real is not None and real.strip() != "":
        return real
    val = env.get(key)
    return val if val else None


def bind_defaults(module_dir: Path | None = None) -> tuple[str, int]:
    """Bind host/port for ``lexi/server.py``'s argparse defaults, layered:
    real env > master ``.env`` > Lexi's own ``.env`` > the standalone
    fallback 127.0.0.1:8765 (so a lone checkout never binds the LAN)."""
    env = _layered_env(module_dir)
    host = env_value(env, "LEXI_HOST") or "127.0.0.1"
    port_raw = env_value(env, "LEXI_PORT")
    try:
        port = int(port_raw) if port_raw else 8765
    except ValueError:
        port = 8765
    return host, port


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
    the file omits, then apply the env-derived destination overrides."""
    env = _layered_env()
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
    cfg._env = env

    # Destinations: env (real > master > Lexi's own .env) overrides the TOML
    # value, applied ONLY when the corresponding key is actually present
    # somewhere in that layered env — so an absent key leaves config.toml (or
    # the dataclass default) untouched, but a stale config.toml destination
    # never survives once the registry defines the key.
    public_brain = env_value(env, "PUBLIC_BRAIN_URL")
    if public_brain:
        cfg.brain.tunnel_base_url = public_brain
    alison_ip = env_value(env, "ALISON_LAN_IP")
    brain_lan_port = env_value(env, "BRAIN_LAN_PORT")
    if alison_ip and brain_lan_port:
        cfg.brain.lan_base_url = f"http://{alison_ip}:{brain_lan_port}"
    lexicon_port = env_value(env, "LEXICON_PORT")
    if lexicon_port:
        cfg.identity.lexicon_base_url = f"http://127.0.0.1:{lexicon_port}"
    public_lexicon = env_value(env, "PUBLIC_LEXICON_URL")
    if public_lexicon:
        cfg.identity.lexicon_fallback_url = public_lexicon
    stt_remote = env_value(env, "LEXI_STT_REMOTE_URL")
    if stt_remote:
        cfg.engines.stt_remote_url = stt_remote

    return cfg
