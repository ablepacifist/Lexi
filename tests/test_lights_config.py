"""Lights config: defaults, env-registry overrides, and the router factory."""
import pytest

from lexi import config as cfgmod
from lexi.config import LexiConfig, LightsConfig, load_config
from lexi.lights import build_light_router

_KEYS = ("LEXI_LIGHTS_FILE", "MQTT_HOST", "MQTT_PORT", "MQTT_USERNAME", "MQTT_PASSWORD",
         "TAPO_USERNAME", "TAPO_PASSWORD", "MASTER_ENV_FILE", "LEXI_CONFIG")

LIGHTS_TOML = '''
[[light]]
id = "lamp"
names = ["lamp"]
mac = "AA:BB:CC:DD:EE:01"
'''


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def lexi_dir(tmp_path, monkeypatch):
    d = tmp_path / "Lexi"
    d.mkdir()
    monkeypatch.setattr(cfgmod, "_MODULE_DIR", d)
    monkeypatch.chdir(d)
    return d


def test_defaults_are_safe_and_offline():
    lc = LightsConfig()
    assert lc.enabled and lc.file == "./lights.toml"
    assert lc.mqtt_host == "" and lc.tapo_username == "" and lc.tapo_password == ""
    assert lc.direct_fallback is True
    assert isinstance(LexiConfig().lights, LightsConfig)


def test_env_registry_populates_lights_settings(lexi_dir):
    (lexi_dir / ".env").write_text(
        "MQTT_HOST=192.168.1.11\nMQTT_PORT=1884\nMQTT_USERNAME=lexi\nMQTT_PASSWORD=s3cret\n"
        "TAPO_USERNAME=me@example.com\nTAPO_PASSWORD=pw\nLEXI_LIGHTS_FILE=/etc/lexi/lights.toml\n")
    lc = load_config().lights
    assert (lc.mqtt_host, lc.mqtt_port) == ("192.168.1.11", 1884)
    assert (lc.mqtt_username, lc.mqtt_password) == ("lexi", "s3cret")
    assert (lc.tapo_username, lc.tapo_password) == ("me@example.com", "pw")
    assert lc.file == "/etc/lexi/lights.toml"


def test_real_env_beats_the_dotenv_file(lexi_dir, monkeypatch):
    (lexi_dir / ".env").write_text("MQTT_HOST=from-file\n")
    monkeypatch.setenv("MQTT_HOST", "from-real-env")
    assert load_config().lights.mqtt_host == "from-real-env"


def test_a_bad_port_falls_back_instead_of_crashing_startup(lexi_dir):
    (lexi_dir / ".env").write_text("MQTT_PORT=not-a-number\n")
    assert load_config().lights.mqtt_port == 1883


def test_toml_section_is_honored_and_env_still_wins(lexi_dir):
    (lexi_dir / "config.toml").write_text('[lights]\nbrightness_step = 10\nmqtt_host = "toml-host"\n')
    (lexi_dir / ".env").write_text("MQTT_HOST=env-host\n")
    lc = load_config().lights
    assert lc.brightness_step == 10 and lc.mqtt_host == "env-host"


# ── build_light_router ───────────────────────────────────────────────────────

def make_cfg(tmp_path, **kw):
    cfg = LexiConfig()
    cfg.lights.file = str(tmp_path / "lights.toml")
    for k, v in kw.items():
        setattr(cfg.lights, k, v)
    return cfg


def test_no_lights_file_means_no_router_and_no_error(tmp_path):
    assert build_light_router(make_cfg(tmp_path)) is None


def test_a_malformed_lights_file_disables_lights_instead_of_crashing(tmp_path):
    (tmp_path / "lights.toml").write_text("this is not = [valid")
    assert build_light_router(make_cfg(tmp_path)) is None


def test_disabled_means_none(tmp_path):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    assert build_light_router(make_cfg(tmp_path, enabled=False)) is None


def test_direct_only_when_no_broker_is_configured(tmp_path):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    router = build_light_router(make_cfg(tmp_path))
    assert router is not None
    assert router._mqtt is None and router._direct is not None
    assert router.inventory.ids == ("lamp",)


def test_with_a_broker_both_paths_exist_and_direct_is_the_fallback(tmp_path, monkeypatch):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    started = []
    from lexi.lights import backends
    monkeypatch.setattr(backends.MqttBackend, "start", lambda self: started.append(self._host))
    router = build_light_router(make_cfg(tmp_path, mqtt_host="192.168.1.11"))
    assert started == ["192.168.1.11"]
    assert router._mqtt is not None and router._direct is not None


def test_bridge_only_mode_has_no_direct_backend(tmp_path, monkeypatch):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    from lexi.lights import backends
    monkeypatch.setattr(backends.MqttBackend, "start", lambda self: None)
    router = build_light_router(make_cfg(tmp_path, mqtt_host="h", direct_fallback=False))
    assert router._direct is None and router._mqtt is not None


def test_no_way_to_reach_any_light_disables_lights(tmp_path):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    assert build_light_router(make_cfg(tmp_path, direct_fallback=False)) is None


def test_missing_paho_falls_back_to_direct_instead_of_failing(tmp_path, monkeypatch):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    from lexi.lights import backends
    from lexi.lights.model import LightError

    def no_paho(self):
        raise LightError("paho-mqtt isn't installed")
    monkeypatch.setattr(backends.MqttBackend, "start", no_paho)
    router = build_light_router(make_cfg(tmp_path, mqtt_host="h"))
    assert router is not None and router._mqtt is None and router._direct is not None


def test_warm_up_is_opt_in(tmp_path, monkeypatch):
    (tmp_path / "lights.toml").write_text(LIGHTS_TOML)
    from lexi.lights.router import LightRouter
    calls = []
    monkeypatch.setattr(LightRouter, "warm_up", lambda self: calls.append(1))
    build_light_router(make_cfg(tmp_path))                  # the CLI: no warm-up
    assert calls == []
    build_light_router(make_cfg(tmp_path), warm_up=True)    # the voice service
    assert calls == [1]
