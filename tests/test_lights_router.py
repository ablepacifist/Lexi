"""Bridge-first / direct-fallback routing and the spoken summary."""
from lexi.lights.model import LightCommand, Targets
from lexi.lights.router import LightResult, LightRouter, summarize

ON = LightCommand(power=True)


class FakeBackend:
    """Stands in for both the MQTT bridge and the direct Tapo backend."""

    def __init__(self, results=None, online=True, raises=None):
        self._results = results or {}
        self._online = online
        self._raises = raises
        self.calls = []
        self.closed = False

    @property
    def bridge_online(self):
        return self._online

    def execute(self, ids, cmd):
        self.calls.append(list(ids))
        if self._raises:
            raise self._raises
        return {i: self._results.get(i) for i in ids}  # None = success

    def close(self):
        self.closed = True


def router(inv, mqtt=None, direct=None, **kw):
    return LightRouter(inv, mqtt=mqtt, direct=direct, **kw)


# ── routing ──────────────────────────────────────────────────────────────────

def test_bridge_up_is_used_and_direct_is_left_alone(inv):
    bridge, direct = FakeBackend(), FakeBackend()
    res = router(inv, bridge, direct).execute(["lamp", "bedroom"], ON)
    assert res.success and res.via == {"lamp": "bridge", "bedroom": "bridge"}
    assert bridge.calls == [["lamp", "bedroom"]] and direct.calls == []


def test_bridge_offline_goes_straight_to_direct(inv):
    bridge, direct = FakeBackend(online=False), FakeBackend()
    res = router(inv, bridge, direct).execute(["lamp"], ON)
    assert res.via == {"lamp": "direct"}
    assert bridge.calls == []  # a dead Bilbo costs nothing: it isn't even asked


def test_no_bridge_configured_uses_direct(inv):
    direct = FakeBackend()
    res = router(inv, None, direct).execute(["lamp"], ON)
    assert res.via == {"lamp": "direct"}


def test_only_the_lights_the_bridge_missed_fall_back(inv):
    bridge = FakeBackend({"bedroom": "no reply from the bridge"})
    direct = FakeBackend()
    res = router(inv, bridge, direct).execute(["lamp", "bedroom"], ON)
    assert res.success  # the fallback covered the miss
    assert res.via == {"lamp": "bridge", "bedroom": "direct"}
    assert direct.calls == [["bedroom"]]
    assert res.errors == {}


def test_a_bridge_exception_is_treated_as_a_miss_not_a_crash(inv):
    bridge = FakeBackend(raises=RuntimeError("broker exploded"))
    direct = FakeBackend()
    res = router(inv, bridge, direct).execute(["lamp"], ON)
    assert res.via == {"lamp": "direct"} and res.success


def test_a_direct_failure_is_reported_with_its_reason(inv):
    bridge = FakeBackend({"lamp": "no reply from the bridge"})
    direct = FakeBackend({"lamp": "unreachable"})
    res = router(inv, bridge, direct).execute(["lamp"], ON)
    assert not res.success and res.errors == {"lamp": "unreachable"}
    assert res.failed_ids == ("lamp",) and res.ok_ids == ()


def test_no_controller_at_all_is_an_error_not_silence(inv):
    res = router(inv, FakeBackend(online=False), None).execute(["lamp"], ON)
    assert res.errors == {"lamp": "no light controller is available"}


def test_duplicate_ids_are_collapsed(inv):
    direct = FakeBackend()
    router(inv, None, direct).execute(["lamp", "lamp"], ON)
    assert direct.calls == [["lamp"]]


def test_close_closes_backends_and_survives_a_failing_one(inv):
    class Boom(FakeBackend):
        def close(self):
            raise RuntimeError("nope")
    good = FakeBackend()
    router(inv, Boom(), good).close()
    assert good.closed


# ── warm-up ──────────────────────────────────────────────────────────────────

def test_warm_up_connects_to_every_bulb_in_the_background(inv):
    import threading
    done = threading.Event()
    seen = []

    class Direct(FakeBackend):
        def state(self, ids):
            seen.append(list(ids))
            done.set()
            return {}
    router(inv, None, Direct()).warm_up()
    assert done.wait(2.0)
    assert seen == [["lamp", "bedroom"]]


def test_warm_up_does_not_block_the_caller(inv):
    import threading
    release = threading.Event()

    class Slow(FakeBackend):
        def state(self, ids):
            release.wait(2.0)       # a 5 s discovery, in miniature
            return {}
    import time
    t0 = time.monotonic()
    router(inv, None, Slow()).warm_up()
    assert time.monotonic() - t0 < 0.5
    release.set()


def test_warm_up_is_a_noop_without_a_direct_backend_and_survives_a_failing_one(inv):
    import threading
    router(inv, FakeBackend(), None).warm_up()            # nothing to warm: must not raise
    done = threading.Event()

    class Boom(FakeBackend):
        def state(self, ids):
            done.set()
            raise RuntimeError("bulb exploded")
    router(inv, None, Boom()).warm_up()
    assert done.wait(2.0)                                 # ran, raised, and was swallowed


# ── "it" memory ──────────────────────────────────────────────────────────────

def test_recent_use_expires():
    from conftest import make_inventory
    now = [1000.0]
    r = LightRouter(make_inventory(), recent_s=120.0, clock=lambda: now[0])
    assert not r.has_recent() and r.last_targets is None
    r.remember(Targets(("lamp",), "lamp"))
    assert r.has_recent() and r.last_targets.ids == ("lamp",)
    now[0] += 119
    assert r.has_recent()
    now[0] += 5
    assert not r.has_recent()
    assert r.last_targets is not None  # remembered, just no longer "recent"


# ── spoken summaries ─────────────────────────────────────────────────────────

def res(targets, errors=None):
    return LightResult(tuple(targets), errors=errors or {})


def test_success_sentences(inv):
    assert summarize(inv, ON, res(["lamp"])) == "Lamp on."
    assert summarize(inv, LightCommand(power=False), res(["lamp", "bedroom"])) == "Both lights off."
    blue = LightCommand(hue=240, saturation=100, color_name="blue")
    assert summarize(inv, blue, res(["bedroom"])) == "Bedroom light set to blue."
    assert summarize(inv, LightCommand(brightness=30), res(["lamp"])) == "Lamp at 30 percent."
    assert summarize(inv, LightCommand(brightness_delta=-25), res(["lamp"])) == "Lamp dimmer."


def test_unreachable_light(inv):
    r = res(["lamp"], {"lamp": "unreachable"})
    assert summarize(inv, ON, r) == "I couldn't reach the lamp."


def test_partial_failure_says_what_worked_and_what_didnt(inv):
    r = res(["lamp", "bedroom"], {"bedroom": "timed out"})
    assert summarize(inv, ON, r) == "Lamp on, but I couldn't reach the bedroom light."


def test_both_unreachable(inv):
    r = res(["lamp", "bedroom"], {"lamp": "unreachable", "bedroom": "unreachable"})
    assert summarize(inv, ON, r) == "I couldn't reach the lamp and the bedroom light."


def test_a_capability_limit_is_not_reported_as_unreachable(inv):
    r = res(["bedroom"], {"bedroom": "can't change color"})
    assert summarize(inv, LightCommand(hue=0, saturation=100), r) == \
        "The bedroom light can't change color."


def test_capability_limit_beside_a_success(inv):
    r = res(["lamp", "bedroom"], {"bedroom": "can't change color"})
    blue = LightCommand(hue=240, saturation=100, color_name="blue")
    assert summarize(inv, blue, r) == "Lamp set to blue, but the bedroom light can't change color."
