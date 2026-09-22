"""Layered destination-registry loading for Lexi.

Precedence proven here: real process env > master ``.env`` (walked up to, or
named by $MASTER_ENV_FILE) > Lexi's own ``.env`` > config.toml > dataclass
default. No network, no servers, no fixed ports — pure filesystem + dict
tests against tmp_path trees that mimic the real <root>/Lexi/ layout.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from lexi import config as cfgmod
from lexi.config import (
    IdentityConfig,
    _find_master_env,
    _layered_env,
    _parse_dotenv,
    bind_defaults,
    env_value,
    load_config,
    read_layered_token,
    tool_token,
)

# Keys the layered registry looks at — scrubbed from the real process env
# before every test so this machine's own shell can't leak into a result.
_REGISTRY_KEYS = (
    "LEXI_HOST", "LEXI_PORT", "LEXICON_PORT", "ALISON_LAN_IP",
    "BRAIN_LAN_PORT", "PUBLIC_BRAIN_URL", "PUBLIC_LEXICON_URL",
    "LEXI_TOOL_TOKEN", "LEXI_STT_REMOTE_URL", "MASTER_ENV_FILE", "LEXI_CONFIG",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _REGISTRY_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def module_tree(tmp_path):
    """A fake <root>/Lexi/ tree so the master-.env walk-up and config.toml
    resolution behave exactly like the real repo layout, without touching it."""
    root = tmp_path / "full-back-end-server"
    lexi_dir = root / "Lexi"
    lexi_dir.mkdir(parents=True)
    return root, lexi_dir


@pytest.fixture
def as_module_dir(monkeypatch, module_tree):
    """Point lexi.config's module-root global at the fake Lexi/ dir, and cwd
    into it (config.toml resolution is cwd-relative), for load_config()/
    bind_defaults()/tool_token()'s no-arg call sites."""
    root, lexi_dir = module_tree
    monkeypatch.setattr(cfgmod, "_MODULE_DIR", lexi_dir)
    monkeypatch.chdir(lexi_dir)
    return root, lexi_dir


# ── the stdlib-only .env parser ─────────────────────────────────────────────

def test_parse_dotenv_skips_blanks_comments_and_strips_quotes(tmp_path):
    p = tmp_path / ".env"
    _write(p, "\n# a comment\nFOO=bar\n\nQUOTED=\"hello world\"\nNOEQUALS\nKEY=a=b=c\n")
    data = _parse_dotenv(p)
    assert data == {"FOO": "bar", "QUOTED": "hello world", "KEY": "a=b=c"}


def test_parse_dotenv_missing_file_is_silently_empty(tmp_path):
    assert _parse_dotenv(tmp_path / "nope.env") == {}


# ── layering: missing files, master vs module, real env ────────────────────

def test_missing_module_and_master_env_are_silently_fine(module_tree):
    root, lexi_dir = module_tree
    assert _layered_env(lexi_dir) == {}
    assert _find_master_env(lexi_dir) is None


def test_master_env_beats_module_env(module_tree):
    root, lexi_dir = module_tree
    _write(lexi_dir / ".env", "LEXI_HOST=0.0.0.0\nLEXICON_PORT=11111\n")
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXICON_PORT=22222\n")
    env = _layered_env(lexi_dir)
    assert env["LEXICON_PORT"] == "22222"   # master overrides the module copy
    assert env["LEXI_HOST"] == "0.0.0.0"    # module-only key still comes through


def test_real_env_beats_master(module_tree, monkeypatch):
    root, lexi_dir = module_tree
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXICON_PORT=22222\n")
    monkeypatch.setenv("LEXICON_PORT", "33333")
    env = _layered_env(lexi_dir)
    assert env["LEXICON_PORT"] == "22222"          # the file layer, unaffected
    assert env_value(env, "LEXICON_PORT") == "33333"  # but env_value() resolves to real env


def test_master_env_file_var_wins_over_walk_up(module_tree, monkeypatch, tmp_path):
    root, lexi_dir = module_tree
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXICON_PORT=22222\n")
    alt = tmp_path / "elsewhere.env"
    _write(alt, "LAN_HOST=1.2.3.4\nLEXICON_PORT=99999\n")
    monkeypatch.setenv("MASTER_ENV_FILE", str(alt))
    assert _find_master_env(lexi_dir) == alt
    assert _layered_env(lexi_dir)["LEXICON_PORT"] == "99999"


def test_walk_up_requires_lan_host_fingerprint(module_tree):
    """A `.env` up the tree that lacks LAN_HOST= is not the master file."""
    root, lexi_dir = module_tree
    _write(root / ".env", "SOME_OTHER_KEY=1\n")
    assert _find_master_env(lexi_dir) is None


def test_layered_env_never_mutates_os_environ(module_tree):
    root, lexi_dir = module_tree
    _write(root / ".env", "LAN_HOST=192.168.1.4\nTOTALLY_UNIQUE_TEST_KEY=abc123\n")
    assert "TOTALLY_UNIQUE_TEST_KEY" not in os.environ
    _layered_env(lexi_dir)
    assert "TOTALLY_UNIQUE_TEST_KEY" not in os.environ  # a dict was built; os.environ untouched


# ── composed destination URLs, via load_config() ────────────────────────────

def test_lone_checkout_with_no_master_still_loads(as_module_dir):
    """No root .env, no Lexi/.env, no process env: dataclass defaults only,
    no invented LAN address, no network access."""
    root, lexi_dir = as_module_dir
    _write(lexi_dir / "config.toml", "")
    cfg = load_config()
    assert cfg.brain.tunnel_base_url == "https://llm.alex-dyakin.com"
    assert cfg.brain.lan_base_url == ""
    assert cfg.identity.lexicon_base_url == "http://127.0.0.1:36568"
    assert cfg.identity.lexicon_fallback_url == ""
    assert cfg.engines.stt_remote_url == ""


def test_load_config_composes_destination_urls_from_env(as_module_dir):
    root, lexi_dir = as_module_dir
    _write(lexi_dir / "config.toml", "")
    _write(
        root / ".env",
        "LAN_HOST=192.168.1.4\n"
        "ALISON_LAN_IP=192.168.1.2\n"
        "BRAIN_LAN_PORT=9080\n"
        "LEXICON_PORT=36568\n"
        "PUBLIC_BRAIN_URL=https://llm.example.com\n"
        "PUBLIC_LEXICON_URL=https://api.example.com\n"
        "LEXI_STT_REMOTE_URL=http://192.168.1.4:8765\n",
    )
    cfg = load_config()
    assert cfg.brain.lan_base_url == "http://192.168.1.2:9080"
    assert cfg.brain.tunnel_base_url == "https://llm.example.com"
    assert cfg.identity.lexicon_base_url == "http://127.0.0.1:36568"
    assert cfg.identity.lexicon_fallback_url == "https://api.example.com"
    assert cfg.engines.stt_remote_url == "http://192.168.1.4:8765"


def test_env_derived_value_beats_stale_toml_value(as_module_dir):
    """A config.toml still listing a dead/old destination does not defeat the
    registry once the master defines the corresponding key — but a key the
    master does NOT define leaves the toml value alone (partial override,
    applied per-key, not a blanket toml override)."""
    root, lexi_dir = as_module_dir
    _write(
        lexi_dir / "config.toml",
        '[brain]\n'
        'lan_base_url = "http://192.168.4.25:9080"\n'   # dead LAN, must lose
        'tunnel_base_url = "https://old-tunnel.example.com"\n'
        '[identity]\n'
        'lexicon_base_url = "http://localhost:36568"\n',
    )
    _write(
        root / ".env",
        "LAN_HOST=192.168.1.4\nPUBLIC_BRAIN_URL=https://llm.alex-dyakin.com\n",
    )
    cfg = load_config()
    assert cfg.brain.tunnel_base_url == "https://llm.alex-dyakin.com"  # env wins
    # ALISON_LAN_IP/BRAIN_LAN_PORT are NOT in the master here, so that
    # specific stale toml value is left exactly as the toml set it.
    assert cfg.brain.lan_base_url == "http://192.168.4.25:9080"
    # LEXICON_PORT likewise absent -> toml's identity.lexicon_base_url stands.
    assert cfg.identity.lexicon_base_url == "http://localhost:36568"


# ── tool token: layered env first, ./.tool_token file fallback ─────────────

def test_tool_token_prefers_layered_env_over_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / ".tool_token", "file-token-value\n")
    env = {"LEXI_TOOL_TOKEN": "env-token-value"}
    assert read_layered_token(env, "LEXI_TOOL_TOKEN", "./.tool_token") == "env-token-value"


def test_tool_token_falls_back_to_file_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / ".tool_token", "file-token-value\n")
    assert read_layered_token({}, "LEXI_TOOL_TOKEN", "./.tool_token") == "file-token-value"


def test_tool_token_missing_everywhere_is_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert read_layered_token({}, "LEXI_TOOL_TOKEN", "./.tool_token") == ""


def test_lexiconfig_tool_token_uses_the_stashed_layered_env(as_module_dir):
    root, lexi_dir = as_module_dir
    _write(lexi_dir / "config.toml", "")
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXI_TOOL_TOKEN=" + "t" * 40 + "\n")
    cfg = load_config()
    assert cfg.tool_token() == "t" * 40


def test_module_level_tool_token_matches_lexiconfig(as_module_dir):
    """lexi/engines/stt_remote.py calls config.tool_token() directly (no
    LexiConfig instance) — must resolve the same layered value."""
    root, lexi_dir = as_module_dir
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXI_TOOL_TOKEN=" + "s" * 40 + "\n")
    assert tool_token() == "s" * 40


def test_real_env_beats_master_for_tool_token(as_module_dir, monkeypatch):
    root, lexi_dir = as_module_dir
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXI_TOOL_TOKEN=" + "m" * 40 + "\n")
    monkeypatch.setenv("LEXI_TOOL_TOKEN", "e" * 40)
    assert tool_token() == "e" * 40


# ── bind host/port defaults (lexi/server.py argparse) ───────────────────────

def test_bind_defaults_standalone_fallback_is_loopback(module_tree):
    root, lexi_dir = module_tree
    assert bind_defaults(lexi_dir) == ("127.0.0.1", 8765)


def test_bind_defaults_from_master_env(module_tree):
    root, lexi_dir = module_tree
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXI_HOST=0.0.0.0\nLEXI_PORT=9999\n")
    assert bind_defaults(lexi_dir) == ("0.0.0.0", 9999)


def test_bind_defaults_real_env_beats_master(module_tree, monkeypatch):
    root, lexi_dir = module_tree
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXI_HOST=0.0.0.0\nLEXI_PORT=9999\n")
    monkeypatch.setenv("LEXI_PORT", "12345")
    assert bind_defaults(lexi_dir) == ("0.0.0.0", 12345)


def test_bind_defaults_picks_up_module_dir_global(as_module_dir):
    """The no-arg call site (as used by lexi/server.py's main()) resolves
    lexi.config._MODULE_DIR dynamically, not a stale import-time default."""
    root, lexi_dir = as_module_dir
    _write(root / ".env", "LAN_HOST=192.168.1.4\nLEXI_HOST=0.0.0.0\nLEXI_PORT=8111\n")
    assert bind_defaults() == ("0.0.0.0", 8111)


# ── media.py's IdentityConfig fallback (no third hardcoded literal) ────────

def test_identity_default_matches_media_fallback():
    """media.MediaPlayer falls back to IdentityConfig().lexicon_base_url
    instead of a hardcoded literal; this pins that default's shape."""
    assert IdentityConfig().lexicon_base_url == "http://127.0.0.1:36568"
