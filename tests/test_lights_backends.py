"""TapoBackend against a fake python-kasa world, MqttBackend against a fake client.

No bulb, no broker, no network — and neither python-kasa nor paho need be installed.
"""
import json
import time
import types

import pytest

from lexi.lights import backends
from lexi.lights.backends import MqttBackend, TapoBackend
from lexi.lights.model import LightCommand, LightInventory
from conftest import make_inventory


# ── a fake python-kasa ───────────────────────────────────────────────────────

class AuthenticationError(Exception):
    """Named like kasa's, which the backend recognises by class name."""


class FakeLightModule:
    def __init__(self, dev, hsv=True, color_temp=True, brightness=True):
        self._dev = dev
        self._features = {"hsv": hsv, "color_temp": color_temp, "brightness": brightness}
        self.brightness = 50
        self.hsv = types.SimpleNamespace(hue=10, saturation=20, value=50)
        self.color_temp = 3000

    def has_feature(self, name):
        return self._features[name]

    async def set_brightness(self, b):
        self._dev.calls.append(("brightness", b))
        self.brightness = b

    async def set_hsv(self, h, s, v=None):
        self._dev.calls.append(("hsv", h, s, v))

    async def set_color_temp(self, k, brightness=None):
        self._dev.calls.append(("ct", k, brightness))


class FakeDevice:
    def __init__(self, mac, on=False, update_error=None, **caps):
        self.mac = mac
        self.is_on = on
        self.alias = "Tapo bulb"
        self.model = "L530"
        self.calls = []
        self.updates = 0
        self.disconnected = False
        self.update_error = update_error
        self.modules = {"Light": FakeLightModule(self, **caps)}

    def _check_open(self):
        if self.disconnected:
            raise RuntimeError("session is closed")

    async def update(self):
        self._check_open()
        self.updates += 1
        if self.update_error is not None:
            raise self.update_error

    async def turn_on(self):
        self._check_open()
        self.calls.append("on")
        self.is_on = True

    async def turn_off(self):
        self._check_open()
        self.calls.append("off")
        self.is_on = False

    async def disconnect(self):
        self.disconnected = True


class FakeKasa:
    """Just enough of the ``kasa`` module: a LAN of devices keyed by IP."""

    def __init__(self, world):
        self.world = world            # host -> FakeDevice
        self.unsupported = []         # UnsupportedDeviceError-like objects to report
        self.discoveries = 0          # broadcast discoveries
        self.connects = []            # unicast discover_single targets, in order
        self.single_error = None      # raise this from discover_single (e.g. unsupported)
        outer = self

        class Credentials:
            def __init__(self, username="", password=""):
                self.username, self.password = username, password

        class Discover:
            @staticmethod
            async def discover(*, credentials=None, discovery_timeout=5, on_unsupported=None):
                outer.discoveries += 1
                return await outer._discover(on_unsupported)

            @staticmethod
            async def discover_single(host, *, credentials=None, discovery_timeout=5, **_):
                outer.connects.append(host)
                if outer.single_error is not None:
                    raise outer.single_error
                if host not in outer.world:
                    raise TimeoutError(f"Timed out getting discovery response for {host}")
                return outer.world[host]

        self.Credentials, self.Discover = Credentials, Discover

    async def _discover(self, on_unsupported):
        if on_unsupported is not None:
            for exc in self.unsupported:
                await on_unsupported(exc)
        return dict(self.world)


class FreshKasa(FakeKasa):
    """Like the real library: every discovery builds brand-new Device objects, and
    takes real time — so two concurrent discoveries genuinely overlap."""

    def __init__(self, macs_by_host):
        super().__init__({})
        self.macs_by_host = macs_by_host
        self.calls = {m: [] for m in macs_by_host.values()}   # shared per-bulb call log
        self.created = []

    async def _discover(self, on_unsupported):
        import asyncio
        await asyncio.sleep(0.05)                              # discovery takes time
        found = {}
        for host, mac in self.macs_by_host.items():
            dev = FakeDevice(mac)
            dev.calls = self.calls[mac]
            self.created.append(dev)
            found[host] = dev
        return found


LAMP_MAC = "AA:BB:CC:DD:EE:01"
BED_MAC = "AA-BB-CC-DD-EE-02"   # devices report MACs in either format


@pytest.fixture(autouse=True)
def _no_real_arp(monkeypatch):
    monkeypatch.setattr(TapoBackend, "_arp_table", classmethod(lambda cls: {}))
    monkeypatch.setattr(TapoBackend, "_prime_arp", staticmethod(lambda: None))


@pytest.fixture
def world():
    return {
        "192.168.1.50": FakeDevice(LAMP_MAC),
        "192.168.1.51": FakeDevice(BED_MAC),
    }


@pytest.fixture
def make(world):
    made = []

    def build(inv=None, kasa=None, **kw):
        backend = TapoBackend(inv or make_inventory(), "me@example.com", "pw",
                              kasa_module=kasa or FakeKasa(world), **kw)
        made.append(backend)
        return backend
    yield build
    for b in made:
        b.close()


# ── discovery by MAC ─────────────────────────────────────────────────────────

def test_finds_a_bulb_by_mac_with_no_ip_configured(make, world):
    kasa = FakeKasa(world)
    b = make(kasa=kasa)
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    assert world["192.168.1.50"].calls == ["on"]
    assert world["192.168.1.51"].calls == []


def test_one_broadcast_resolves_every_bulb(make, world):
    kasa = FakeKasa(world)
    b = make(kasa=kasa)
    b.execute(["lamp"], LightCommand(power=True))
    b.execute(["bedroom"], LightCommand(power=True))
    assert kasa.discoveries == 1  # the bedroom was cached by the same broadcast


def test_a_bulb_that_is_not_on_the_network_is_reported_not_found(make, world):
    del world["192.168.1.51"]
    b = make()
    assert b.execute(["bedroom"], LightCommand(power=True)) == {"bedroom": "not found on the network"}


def test_all_lights_are_commanded_together(make, world):
    assert make().execute(["lamp", "bedroom"], LightCommand(power=False)) == \
        {"lamp": None, "bedroom": None}
    assert world["192.168.1.50"].calls == ["off"] and world["192.168.1.51"].calls == ["off"]


def test_all_lights_on_a_cold_cache_run_ONE_discovery_and_no_session_is_torn_down(make):
    # Regression: both lights used to start their own discovery at once; each built new
    # Device objects and the second disconnected the first's while it was mid-command.
    kasa = FreshKasa({"192.168.1.50": LAMP_MAC, "192.168.1.51": BED_MAC})
    b = make(kasa=kasa)
    assert b.execute(["lamp", "bedroom"], LightCommand(power=True)) == {"lamp": None, "bedroom": None}
    assert kasa.discoveries == 1
    assert kasa.calls[LAMP_MAC] == ["on"] and kasa.calls[BED_MAC] == ["on"]
    assert not any(d.disconnected for d in kasa.created if d.calls)   # nothing in use was closed


def test_unknown_light_id(make):
    assert make().execute(["garage"], LightCommand(power=True)) == {"garage": "unknown light"}


# ── remembered addresses and lossy Wi-Fi discovery ──────────────────────────

class LossyKasa(FakeKasa):
    """Wi-Fi broadcast replies get lost: discovery only reports what ``script`` says."""

    def __init__(self, world, script):
        super().__init__(world)
        self.script = list(script)        # per-broadcast: list of hosts that "replied"

    async def _discover(self, on_unsupported):
        replied = self.script.pop(0) if self.script else list(self.world)
        return {h: self.world[h] for h in replied if h in self.world}


def _lossy(world, script):
    return LossyKasa(world, script)


def test_a_lost_broadcast_reply_is_retried_once(make, world):
    # First broadcast: the lamp's reply is lost. Second: it arrives.
    kasa = _lossy(world, [["192.168.1.51"], ["192.168.1.50", "192.168.1.51"]])
    assert make(kasa=kasa).execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    assert kasa.discoveries == 2
    assert world["192.168.1.50"].calls == ["on"]


def test_a_bulb_missing_from_both_broadcasts_is_reported_not_found(make, world):
    kasa = _lossy(world, [[], []])
    assert make(kasa=kasa).execute(["lamp"], LightCommand(power=True)) ==         {"lamp": "not found on the network"}
    assert kasa.discoveries == 2            # tried twice, then gave up (bounded)


def test_addresses_are_remembered_on_disk_so_a_fresh_process_never_broadcasts(world, tmp_path):
    cache = tmp_path / "logs" / "light_hosts.json"
    first = FakeKasa(world)
    b1 = TapoBackend(make_inventory(), "u", "p", kasa_module=first, host_cache=cache)
    try:
        assert b1.execute(["lamp", "bedroom"], LightCommand(power=True)) == {"lamp": None, "bedroom": None}
    finally:
        b1.close()
    assert first.discoveries == 1
    saved = json.loads(cache.read_text())
    assert saved == {"AABBCCDDEE01": "192.168.1.50", "AABBCCDDEE02": "192.168.1.51"}

    # A brand-new backend (a restart, the CLI, the bridge): connects by unicast, no broadcast.
    second = FakeKasa(world)
    for d in world.values():        # a new process gets fresh sessions, not the closed ones
        d.calls.clear()
        d.disconnected = False
    b2 = TapoBackend(make_inventory(), "u", "p", kasa_module=second, host_cache=cache)
    try:
        assert b2.execute(["lamp", "bedroom"], LightCommand(power=False)) == {"lamp": None, "bedroom": None}
    finally:
        b2.close()
    assert second.discoveries == 0
    assert sorted(second.connects) == ["192.168.1.50", "192.168.1.51"]


def test_a_stale_cached_address_falls_back_to_discovery_and_the_cache_heals(world, tmp_path):
    cache = tmp_path / "hosts.json"
    cache.write_text(json.dumps({"AABBCCDDEE01": "192.168.1.99"}))   # the lamp moved: .99 is gone
    kasa = FakeKasa(world)
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=kasa, host_cache=cache)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()
    assert kasa.connects[0] == "192.168.1.99" and kasa.discoveries == 1
    assert json.loads(cache.read_text())["AABBCCDDEE01"] == "192.168.1.50"   # healed


def test_an_address_that_now_belongs_to_another_bulb_is_not_trusted(world, tmp_path):
    cache = tmp_path / "hosts.json"
    cache.write_text(json.dumps({"AABBCCDDEE01": "192.168.1.51"}))   # .51 is the BEDROOM now
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=FakeKasa(world), host_cache=cache)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()
    assert world["192.168.1.50"].calls == ["on"] and world["192.168.1.51"].calls == []


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"AA": 5}', "", "null"])
def test_a_corrupt_cache_is_ignored_not_fatal(world, tmp_path, content):
    cache = tmp_path / "hosts.json"
    cache.write_text(content)
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=FakeKasa(world), host_cache=cache)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()


def test_an_unwritable_cache_location_does_not_break_control(world, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=FakeKasa(world),
                    host_cache=blocker / "sub" / "hosts.json")   # parent is a FILE: can't be created
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()


def test_a_power_cycled_bulb_is_retried_at_the_same_address_after_a_pause(world, tmp_path):
    # An address that just failed is skipped briefly, not forever.
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=FakeKasa(world))
    try:
        lamp = b._inv.get("lamp")
        b._known_hosts[lamp.mac] = "192.168.1.50"
        b._bad_hosts[("lamp", "192.168.1.50")] = time.monotonic()           # just failed
        assert b._candidate_hosts(lamp) == []
        b._bad_hosts[("lamp", "192.168.1.50")] = time.monotonic() - 31       # long enough ago
        assert b._candidate_hosts(lamp) == ["192.168.1.50"]
    finally:
        b.close()


# ── ARP: the reliable way from MAC to address ────────────────────────────────

ARP_SAMPLE = """IP address       HW type     Flags       HW address            Mask     Device
192.168.1.10     0x1         0x2         ec:75:0c:63:25:fe     *        wlan0
192.168.1.12     0x1         0x2         EC:75:0C:63:2B:DF     *        wlan0
192.168.1.99     0x1         0x0         00:00:00:00:00:00     *        wlan0
"""


def test_arp_table_parsing():
    parsed = TapoBackend._parse_arp(ARP_SAMPLE)
    assert parsed == {"EC750C6325FE": "192.168.1.10", "EC750C632BDF": "192.168.1.12"}
    assert TapoBackend._parse_arp("") == {}
    assert TapoBackend._parse_arp("only a header line\n") == {}


def test_arp_finds_a_bulb_without_any_broadcast(world, monkeypatch):
    monkeypatch.setattr(TapoBackend, "_arp_table",
                        classmethod(lambda cls: {"AABBCCDDEE01": "192.168.1.50"}))
    kasa = FakeKasa(world)
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()
    assert kasa.discoveries == 0 and kasa.connects == ["192.168.1.50"]


def test_arp_is_primed_only_when_the_table_lacks_the_bulb(world, monkeypatch):
    calls = []
    table = {}
    monkeypatch.setattr(TapoBackend, "_arp_table", classmethod(lambda cls: dict(table)))

    def prime():
        calls.append(1)
        table["AABBCCDDEE01"] = "192.168.1.50"    # the sweep populates the ARP table
    monkeypatch.setattr(TapoBackend, "_prime_arp", staticmethod(prime))
    kasa = FakeKasa(world)
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()
    assert calls == [1] and kasa.discoveries == 0


def test_when_arp_has_nothing_it_falls_through_to_discovery(world):
    kasa = FakeKasa(world)          # the autouse fixture makes ARP empty (like non-Linux)
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()
    assert kasa.discoveries == 1


def test_a_wrong_arp_entry_does_not_stick(world, monkeypatch):
    # ARP says .51, but that is the bedroom (MAC mismatch) -> ignored, discovery finds .50.
    monkeypatch.setattr(TapoBackend, "_arp_table",
                        classmethod(lambda cls: {"AABBCCDDEE01": "192.168.1.51"}))
    kasa = FakeKasa(world)
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    finally:
        b.close()
    assert world["192.168.1.50"].calls == ["on"] and world["192.168.1.51"].calls == []


def test_an_arp_host_running_the_unsupported_protocol_is_reported_as_such(world, monkeypatch):
    class Unsupported(Exception):
        pass
    Unsupported.__name__ = "UnsupportedDeviceError"
    kasa = FakeKasa(world)
    kasa.single_error = Unsupported("TPAP")
    monkeypatch.setattr(TapoBackend, "_arp_table",
                        classmethod(lambda cls: {"AABBCCDDEE01": "192.168.1.50"}))
    b = TapoBackend(make_inventory(), "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "isn't supported yet"}
    finally:
        b.close()


# ── ip hints ─────────────────────────────────────────────────────────────────

def test_a_host_hint_is_used_without_any_broadcast(world):
    inv = LightInventory.from_dict({"light": [
        {"id": "lamp", "names": ["lamp"], "mac": LAMP_MAC, "host": "192.168.1.50"}]})
    kasa = FakeKasa(world)
    b = TapoBackend(inv, "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
        assert kasa.discoveries == 0 and kasa.connects == ["192.168.1.50"]
    finally:
        b.close()


def test_a_hint_that_now_belongs_to_another_device_falls_back_to_discovery(world):
    # DHCP handed .50 to a different bulb: the MAC check must catch it.
    inv = LightInventory.from_dict({"light": [
        {"id": "lamp", "names": ["lamp"], "mac": LAMP_MAC, "host": "192.168.1.51"}]})
    kasa = FakeKasa(world)
    b = TapoBackend(inv, "u", "p", kasa_module=kasa)
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
        assert world["192.168.1.50"].calls == ["on"]     # the real lamp
        assert world["192.168.1.51"].calls == []         # the impostor, untouched
        assert kasa.discoveries == 1
    finally:
        b.close()


# ── stale connections ────────────────────────────────────────────────────────

def test_a_moved_bulb_is_rediscovered_after_a_failure(make, world):
    kasa = FakeKasa(world)
    b = make(kasa=kasa)
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": None}
    old = world.pop("192.168.1.50")
    old.update_error = OSError("connection reset")          # the cached session dies
    world["192.168.1.99"] = FakeDevice(LAMP_MAC)            # same bulb, new IP
    assert b.execute(["lamp"], LightCommand(power=False)) == {"lamp": None}
    assert world["192.168.1.99"].calls == ["off"]
    assert old.disconnected                                  # the dead session was released
    assert kasa.discoveries == 2


def test_wrong_credentials_are_not_retried(make, world):
    world["192.168.1.50"].update_error = AuthenticationError("bad creds")
    b = make()
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "authentication failed"}
    # Discovery updates the device once; the command path must NOT loop on a bad
    # password (that can lock the account) — so no further update attempts pile up.
    assert world["192.168.1.50"].updates <= 2


# ── newer-protocol (TPAP) bulbs ──────────────────────────────────────────────

class UnsupportedDeviceError(Exception):
    """Named like kasa's; carries the same discovery_result/host attributes."""

    def __init__(self, host, mac, protocol="TPAP", model="TL135", as_object=False):
        super().__init__("Unsupported device")
        self.host = host
        info = {"mac": mac, "device_model": model,
                "mgt_encrypt_schm": {"encrypt_type": protocol}}
        self.discovery_result = types.SimpleNamespace(
            mac=mac, device_model=model,
            mgt_encrypt_schm=types.SimpleNamespace(encrypt_type=protocol)) if as_object else info


def test_a_bulb_on_an_unsupported_protocol_is_reported_as_that_not_as_missing(make, world):
    del world["192.168.1.50"]                    # the lamp answers discovery only as "unsupported"
    kasa = FakeKasa(world)
    kasa.unsupported = [UnsupportedDeviceError("192.168.1.10", "EC-75-0C-63-25-FE")]
    inv = LightInventory.from_dict({"light": [
        {"id": "lamp", "names": ["lamp"], "mac": "EC:75:0C:63:25:FE"}]})
    b = make(inv, kasa)
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "isn't supported yet"}


def test_the_unsupported_bulb_does_not_stop_the_supported_one_working(make, world):
    kasa = FakeKasa(world)
    kasa.unsupported = [UnsupportedDeviceError("192.168.1.10", "EC:75:0C:63:25:FE")]
    inv = LightInventory.from_dict({"light": [
        {"id": "lamp", "names": ["lamp"], "mac": "EC:75:0C:63:25:FE"},
        {"id": "bedroom", "names": ["bedroom"], "mac": BED_MAC}]})
    b = make(inv, kasa)
    assert b.execute(["lamp", "bedroom"], LightCommand(power=True)) ==         {"lamp": "isn't supported yet", "bedroom": None}
    assert world["192.168.1.51"].calls == ["on"]


def test_discover_lists_unsupported_devices_with_their_protocol(make, world):
    kasa = FakeKasa(world)
    kasa.unsupported = [UnsupportedDeviceError("192.168.1.10", "EC-75-0C-63-25-FE", "TPAP")]
    found = make(kasa=kasa).discover()
    bad = [d for d in found if d.get("unsupported")]
    assert len(bad) == 1
    assert (bad[0]["host"], bad[0]["protocol"], bad[0]["model"]) == ("192.168.1.10", "TPAP", "TL135")
    assert len(found) == 3                        # the two supported bulbs are still listed


def test_unsupported_info_reads_dicts_objects_and_missing_data():
    f = TapoBackend._unsupported_info
    assert f(UnsupportedDeviceError("h", "AA", "TPAP"))["protocol"] == "TPAP"
    assert f(UnsupportedDeviceError("h", "AA", "TPAP", as_object=True))["mac"] == "AA"
    assert f(Exception("no details at all")) == {"host": "", "mac": "", "model": "", "protocol": ""}


def test_speech_for_an_unsupported_bulb_is_truthful(make, world):
    from lexi.lights.router import LightResult, summarize
    inv = make_inventory()
    res = LightResult(("lamp",), errors={"lamp": "isn't supported yet"})
    assert summarize(inv, LightCommand(power=True), res) == "The lamp isn't supported yet."


# ── the commands themselves ──────────────────────────────────────────────────

def test_setting_brightness_on_a_bulb_that_is_off_also_turns_it_on(make, world):
    b = make()
    assert b.execute(["lamp"], LightCommand(brightness=30)) == {"lamp": None}
    assert world["192.168.1.50"].calls == [("brightness", 30), "on"]


def test_brightness_on_a_bulb_that_is_already_on_does_not_toggle_power(make, world):
    world["192.168.1.50"].is_on = True
    make().execute(["lamp"], LightCommand(brightness=30))
    assert world["192.168.1.50"].calls == [("brightness", 30)]


def test_color_passes_hue_saturation_and_optional_brightness(make, world):
    world["192.168.1.50"].is_on = True
    b = make()
    b.execute(["lamp"], LightCommand(hue=240, saturation=100, brightness=30))
    assert world["192.168.1.50"].calls == [("hsv", 240, 100, 30)]


def test_color_temperature(make, world):
    world["192.168.1.50"].is_on = True
    make().execute(["lamp"], LightCommand(color_temp=2700))
    assert world["192.168.1.50"].calls == [("ct", 2700, None)]


@pytest.mark.parametrize("current,delta,expect", [
    (50, 25, 75), (90, 25, 100), (10, -25, 1), (50, -25, 25)])
def test_relative_brightness_is_clamped(make, world, current, delta, expect):
    dev = world["192.168.1.50"]
    dev.is_on = True
    dev.modules["Light"].brightness = current
    make().execute(["lamp"], LightCommand(brightness_delta=delta))
    assert dev.calls == [("brightness", expect)]


def test_toggle_flips_the_current_state(make, world):
    b = make()
    b.execute(["lamp"], LightCommand(toggle=True))
    b.execute(["lamp"], LightCommand(toggle=True))
    assert world["192.168.1.50"].calls == ["on", "off"]


def test_a_bulb_that_cannot_do_color_says_so_and_is_not_dropped(make, world):
    dev = world["192.168.1.50"]
    dev.modules["Light"] = FakeLightModule(dev, hsv=False)
    kasa = FakeKasa(world)
    b = make(kasa=kasa)
    assert b.execute(["lamp"], LightCommand(hue=0, saturation=100)) == {"lamp": "can't change color"}
    assert not dev.disconnected           # a capability limit isn't a connection fault
    assert kasa.discoveries == 1          # ...so no pointless re-discovery


def test_a_device_that_is_not_a_light(make, world):
    world["192.168.1.50"].modules = {}
    assert make().execute(["lamp"], LightCommand(power=True)) == {"lamp": "isn't a light"}


# ── state / discover / identify ──────────────────────────────────────────────

def test_state_reports_what_the_bulb_says(make, world):
    dev = world["192.168.1.50"]
    dev.is_on = True
    st = make().state(["lamp"])["lamp"]
    assert st == {"reachable": True, "on": True, "brightness": 50,
                  "hs": [10, 20], "color_temp": 3000}


def test_state_of_a_missing_bulb_is_unreachable_not_an_error(make, world):
    del world["192.168.1.51"]
    assert make().state(["bedroom"]) == {"bedroom": {"reachable": False}}


def test_discover_lists_devices_and_their_capabilities(make):
    found = make().discover()
    assert {d["host"] for d in found} == {"192.168.1.50", "192.168.1.51"}
    lamp = next(d for d in found if d["host"] == "192.168.1.50")
    assert lamp["is_light"] and lamp["model"] == "L530"
    assert lamp["features"] == {"brightness": True, "hsv": True, "color_temp": True}


def test_identify_blinks_and_restores_the_original_state(make, world, monkeypatch):
    async def instant(*_a, **_k):
        return None
    monkeypatch.setattr(backends.asyncio, "sleep", instant)
    dev = world["192.168.1.50"]                      # starts OFF
    assert make().identify("lamp") is None
    assert dev.calls == ["off", "on"] * 3 + ["off"]  # blinked 3x, then back to off


def test_missing_python_kasa_is_a_clear_error(world, monkeypatch):
    import builtins
    real = builtins.__import__

    def deny(name, *a, **k):
        if name == "kasa":
            raise ImportError("no kasa")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", deny)
    b = TapoBackend(make_inventory(), "u", "p")  # no injected module
    try:
        assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "python-kasa isn't installed"}
    finally:
        b.close()


# ── MQTT client ──────────────────────────────────────────────────────────────

class RC:
    def __init__(self, failure=False):
        self.is_failure = failure


class Msg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload if isinstance(payload, bytes) else payload.encode()


class FakeClient:
    """Records what the backend does; can auto-acknowledge like a live bridge."""

    def __init__(self):
        self.published, self.subscribed = [], []
        self.creds = self.addr = None
        self.loop_started = self.closed = False
        self.on_connect = self.on_disconnect = self.on_message = None
        self.responder = None    # callable(topic, payload_dict) -> ack dict | None
        self.backend = None

    def username_pw_set(self, u, p):
        self.creds = (u, p)

    def reconnect_delay_set(self, **_):
        pass

    def connect_async(self, host, port, keepalive=60):
        self.addr = (host, port)

    def loop_start(self):
        self.loop_started = True

    def loop_stop(self):
        pass

    def disconnect(self):
        self.closed = True

    def subscribe(self, topics, qos=0):
        self.subscribed.append(topics)

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        if self.responder and topic.endswith("/set"):
            data = json.loads(payload)
            ack = self.responder(topic, data)
            if ack is not None:
                light = topic.split("/")[2]
                self.backend._on_message(self, None, Msg(f"home/lights/{light}/ack", json.dumps(ack)))


@pytest.fixture
def mqtt():
    client = FakeClient()
    b = MqttBackend("bilbo", 1883, "lexi", "secret", ack_timeout=0.2, client_factory=lambda: client)
    client.backend = b
    b.start()
    return b, client


def online(b, client):
    b._on_connect(client, None, {}, RC(False), None)
    b._on_message(client, None, Msg(backends.STATUS_TOPIC, "online"))


def test_start_connects_with_credentials_in_the_background(mqtt):
    b, client = mqtt
    assert client.addr == ("bilbo", 1883) and client.creds == ("lexi", "secret")
    assert client.loop_started
    assert not b.bridge_online   # nothing heard yet


def test_bridge_is_online_only_when_connected_and_the_status_says_so(mqtt):
    b, client = mqtt
    b._on_connect(client, None, {}, RC(False), None)
    assert client.subscribed and not b.bridge_online       # connected, status unknown
    b._on_message(client, None, Msg(backends.STATUS_TOPIC, "online"))
    assert b.bridge_online
    b._on_message(client, None, Msg(backends.STATUS_TOPIC, "offline"))   # Last Will fired
    assert not b.bridge_online
    b._on_message(client, None, Msg(backends.STATUS_TOPIC, "online"))
    b._on_disconnect(client, None, None, None, None)                     # broker lost
    assert not b.bridge_online


def test_a_refused_connection_stays_offline(mqtt):
    b, client = mqtt
    b._on_connect(client, None, {}, RC(True), None)
    assert not client.subscribed and not b.bridge_online


def test_offline_bridge_does_not_publish_anything(mqtt):
    b, client = mqtt
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "the light bridge is offline"}
    assert client.published == []


def test_a_command_is_published_and_the_ack_returned(mqtt):
    b, client = mqtt
    online(b, client)
    client.responder = lambda topic, data: {"cid": data["cid"], "ok": True}
    assert b.execute(["lamp"], LightCommand(hue=240, saturation=100, color_name="blue")) == {"lamp": None}
    topic, payload, qos, retain = client.published[0]
    body = json.loads(payload)
    assert topic == "home/lights/lamp/set" and qos == 1 and not retain
    assert body["hue"] == 240 and body["color_name"] == "blue" and body["cid"]


def test_each_light_is_acknowledged_separately(mqtt):
    b, client = mqtt
    online(b, client)
    client.responder = lambda topic, data: (
        {"cid": data["cid"], "ok": True} if "lamp" in topic
        else {"cid": data["cid"], "ok": False, "error": "unreachable"})
    assert b.execute(["lamp", "bedroom"], LightCommand(power=True)) == \
        {"lamp": None, "bedroom": "unreachable"}


def test_a_missing_ack_times_out_so_the_router_can_fall_back(mqtt):
    b, client = mqtt
    online(b, client)
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "no reply from the bridge"}


def test_an_ack_for_someone_elses_command_is_ignored(mqtt):
    b, client = mqtt
    online(b, client)
    client.responder = lambda topic, data: {"cid": "not-ours", "ok": True}
    assert b.execute(["lamp"], LightCommand(power=True)) == {"lamp": "no reply from the bridge"}


def test_garbage_on_the_ack_topic_is_ignored(mqtt):
    b, client = mqtt
    online(b, client)
    b._on_message(client, None, Msg("home/lights/lamp/ack", "not json"))
    b._on_message(client, None, Msg("home/lights/lamp/ack", "[1,2]"))
    b._on_message(client, None, Msg("home/lights/lamp/state", "{}"))   # not an ack


def test_pending_waiters_are_cleaned_up(mqtt):
    b, client = mqtt
    online(b, client)
    b.execute(["lamp"], LightCommand(power=True))   # times out
    assert b._pending == {}


def test_close_stops_the_client(mqtt):
    b, client = mqtt
    b.close()
    assert client.closed and not b.bridge_online
