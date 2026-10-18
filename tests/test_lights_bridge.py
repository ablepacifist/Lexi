"""The Bilbo bridge: command handling, status/Last Will, state publishing."""
import json

import pytest

from lexi.lights import backends
from lexi.lights.bridge import LightsBridge
from conftest import make_inventory


class FakeTapo:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []
        self.closed = False

    def execute(self, ids, cmd):
        self.calls.append((list(ids), cmd))
        return {i: self.results.get(i) for i in ids}

    def state(self, ids):
        return {i: {"reachable": True, "on": True, "brightness": 80} for i in ids}

    def close(self):
        self.closed = True


class RC:
    def __init__(self, failure=False):
        self.is_failure = failure


class Msg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload if isinstance(payload, bytes) else payload.encode()


class Info:
    def wait_for_publish(self, timeout=None):
        return True


class FakeClient:
    def __init__(self):
        self.published, self.subscribed = [], []
        self.will = self.creds = self.addr = None
        self.on_connect = self.on_message = None
        self.stopped = self.disconnected = False

    def username_pw_set(self, u, p):
        self.creds = (u, p)

    def will_set(self, topic, payload, qos=0, retain=False):
        self.will = (topic, payload, qos, retain)

    def reconnect_delay_set(self, **_):
        pass

    def connect_async(self, host, port, keepalive=60):
        self.addr = (host, port)

    def loop_start(self):
        pass

    def loop_stop(self):
        self.stopped = True

    def disconnect(self):
        self.disconnected = True

    def subscribe(self, topic, qos=0):
        self.subscribed.append(topic)

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        return Info()


class ImmediatePool:
    """Runs submitted work inline so tests need no threads."""

    def submit(self, fn, *args):
        fn(*args)

    def shutdown(self, **_):
        pass


@pytest.fixture
def bridge():
    inv = make_inventory()
    tapo = FakeTapo()
    client = FakeClient()
    b = LightsBridge(tapo, inv, "127.0.0.1", 1883, "bridge", "pw", client_factory=lambda: client)
    b._pool = ImmediatePool()
    b.start()
    b._stop.set()   # keep the background poller from ticking during a test
    return b, tapo, client


def published(client, topic):
    return [(json.loads(p) if p and p[0] in "{[" else p, retain)
            for t, p, _q, retain in client.published if t == topic]


# ── handle_set (pure) ────────────────────────────────────────────────────────

def test_a_valid_command_runs_and_is_acknowledged(bridge):
    b, tapo, _ = bridge
    ack = b.handle_set("lamp", json.dumps({"cid": "c1", "power": True}))
    assert ack == {"cid": "c1", "ok": True, "error": None}
    ids, cmd = tapo.calls[0]
    assert ids == ["lamp"] and cmd.power is True


def test_the_correlation_id_never_reaches_the_command(bridge):
    b, tapo, _ = bridge
    b.handle_set("lamp", json.dumps({"cid": "c1", "brightness": 30}))
    assert tapo.calls[0][1].to_payload() == {"brightness": 30}


def test_a_backend_failure_is_reported_in_the_ack(bridge):
    b, tapo, _ = bridge
    tapo.results = {"lamp": "unreachable"}
    ack = b.handle_set("lamp", json.dumps({"cid": "c2", "power": True}))
    assert ack == {"cid": "c2", "ok": False, "error": "unreachable"}


def test_an_unknown_light_is_refused_without_touching_a_bulb(bridge):
    b, tapo, _ = bridge
    ack = b.handle_set("garage", json.dumps({"cid": "c3", "power": True}))
    assert ack == {"cid": "c3", "ok": False, "error": "unknown light"}
    assert tapo.calls == []


@pytest.mark.parametrize("raw", [b"not json", "", b"\xff\xfe", "[1, 2]", "42", "null"])
def test_garbage_is_ignored_with_no_ack_and_no_crash(bridge, raw):
    b, tapo, _ = bridge
    assert b.handle_set("lamp", raw) is None
    assert tapo.calls == []


@pytest.mark.parametrize("body", [
    {"cid": "x"},                                # empty command
    {"cid": "x", "brightness": "very"},          # wrong type
    {"cid": "x", "power": "yes"},
    {"cid": "x", "brightness": 5, "brightness_delta": 5},
])
def test_a_malformed_command_is_refused_and_never_reaches_a_bulb(bridge, body):
    b, tapo, _ = bridge
    ack = b.handle_set("lamp", json.dumps(body))
    assert ack["ok"] is False and ack["error"].startswith("bad command")
    assert tapo.calls == []


def test_out_of_range_values_are_clamped_not_passed_through(bridge):
    b, tapo, _ = bridge
    b.handle_set("lamp", json.dumps({"cid": "c", "brightness": 9999, "color_temp": 1}))
    cmd = tapo.calls[0][1]
    assert cmd.brightness == 100 and cmd.color_temp == 2500


# ── MQTT wiring ──────────────────────────────────────────────────────────────

def test_the_last_will_marks_the_bridge_offline_if_it_dies(bridge):
    _, _, client = bridge
    assert client.will == (backends.STATUS_TOPIC, "offline", 1, True)
    assert client.addr == ("127.0.0.1", 1883) and client.creds == ("bridge", "pw")


def test_on_connect_announces_online_and_subscribes_to_commands(bridge):
    b, _, client = bridge
    b._on_connect(client, None, {}, RC(False), None)
    assert "home/lights/+/set" in client.subscribed
    assert (backends.STATUS_TOPIC, "online", 1, True) in client.published


def test_a_refused_connect_does_not_announce_online(bridge):
    b, _, client = bridge
    b._on_connect(client, None, {}, RC(True), None)
    assert client.published == [] and client.subscribed == []


def test_connecting_publishes_the_initial_state_of_every_light(bridge):
    b, _, client = bridge
    b._on_connect(client, None, {}, RC(False), None)
    lamp = published(client, "home/lights/lamp/state")
    assert lamp and lamp[0][1] is True                        # retained
    assert lamp[0][0]["on"] is True and "ts" in lamp[0][0]
    assert published(client, "home/lights/bedroom/state")


def test_a_set_message_produces_an_ack_and_a_retained_state(bridge):
    b, tapo, client = bridge
    b._on_message(client, None, Msg("home/lights/lamp/set", json.dumps({"cid": "z", "power": False})))
    assert tapo.calls and tapo.calls[0][0] == ["lamp"]
    acks = published(client, "home/lights/lamp/ack")
    assert acks[0][0] == {"cid": "z", "ok": True, "error": None} and acks[0][1] is False
    assert published(client, "home/lights/lamp/state")[0][1] is True


def test_other_topics_are_ignored(bridge):
    b, tapo, client = bridge
    b._on_message(client, None, Msg("home/lights/lamp/state", "{}"))
    b._on_message(client, None, Msg("home/lights/bridge/status", "online"))
    b._on_message(client, None, Msg("home/other/x/set", "{}"))
    assert tapo.calls == []


def test_a_crash_in_one_command_does_not_kill_the_worker(bridge):
    b, tapo, client = bridge

    def boom(*_a, **_k):
        raise RuntimeError("bulb exploded")
    tapo.execute = boom
    b._on_message(client, None, Msg("home/lights/lamp/set", json.dumps({"cid": "z", "power": True})))
    # ...and the next command still works.
    tapo.execute = lambda ids, cmd: {i: None for i in ids}
    b._on_message(client, None, Msg("home/lights/lamp/set", json.dumps({"cid": "y", "power": True})))
    assert published(client, "home/lights/lamp/ack")[-1][0]["cid"] == "y"


def test_stop_announces_offline_and_releases_everything(bridge):
    b, tapo, client = bridge
    b.stop()
    assert (backends.STATUS_TOPIC, "offline", 1, True) in client.published
    assert client.stopped and client.disconnected and tapo.closed
