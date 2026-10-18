"""The spoken light grammar — what does and, just as important, does NOT match."""
import pytest

from lexi.intents import match_media_intent
from lexi.lights.intents import match_light_intent
from lexi.lights.model import LightInventory
from conftest import make_inventory

BOTH = {"lamp", "bedroom"}


def go(inv, phrase, **kw):
    return match_light_intent(phrase, inv, **kw)


def payload(inv, phrase, **kw):
    intent = go(inv, phrase, **kw)
    assert intent is not None, f"expected a light command: {phrase!r}"
    return intent.command.to_payload()


def targets(inv, phrase, **kw):
    intent = go(inv, phrase, **kw)
    assert intent is not None, f"expected a light command: {phrase!r}"
    return None if intent.targets is None else set(intent.targets.ids)


# ── on / off ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,who,power", [
    ("turn on the lamp", {"lamp"}, True),
    ("Turn off the lamp.", {"lamp"}, False),
    ("turn the lamp off", {"lamp"}, False),
    ("turn the lamp on", {"lamp"}, True),
    ("lamp on please", {"lamp"}, True),
    ("switch on the bedroom light", {"bedroom"}, True),
    ("switch the bedroom light off", {"bedroom"}, False),
    ("put on the lamp", {"lamp"}, True),
    ("hey jarvis turn off all the lights", BOTH, False),
    ("okay jarvis, turn the lamp off", {"lamp"}, False),
    ("can you turn on the lamp", {"lamp"}, True),
    ("turn off both lights", BOTH, False),
    ("lights on", BOTH, True),
    ("turn on the lights", BOTH, True),
    ("turn off the lamp and the bedroom light", BOTH, False),
])
def test_power(inv, phrase, who, power):
    assert targets(inv, phrase) == who
    assert payload(inv, phrase) == {"power": power}


def test_no_light_named_means_the_default_target():
    inv = make_inventory(default_target="bedroom")
    intent = go(inv, "turn on the light")
    assert intent.defaulted is True
    assert intent.targets.ids == ("bedroom",)
    assert set(go(make_inventory(), "turn on the light").targets.ids) == BOTH


def test_on_as_a_location_is_not_a_power_word(inv):
    # "on" here is "on the desk", not "turn on": no power command → no light intent.
    assert go(inv, "the lamp on the desk") is None


# ── brightness ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,level", [
    ("set the lamp to 50 percent", 50),
    ("set the lamp to fifty percent", 50),
    ("set the lamp to 30%", 30),
    ("set the lamp to twenty five percent", 25),
    ("lamp to 80", 80),
    ("set the bedroom light to a hundred percent", 100),
    ("set the lamp to one hundred percent", 100),
    ("full brightness on the lamp", 100),
    ("set the lamp to minimum", 1),
    ("set the lamp to brightness 40", 40),
])
def test_absolute_brightness(inv, phrase, level):
    assert payload(inv, phrase) == {"brightness": level}


def test_zero_percent_means_off(inv):
    assert payload(inv, "set the lamp to zero percent") == {"power": False}


def test_all_lights_to_a_level(inv):
    assert targets(inv, "set both lights to 30 percent") == BOTH
    assert payload(inv, "set both lights to 30 percent") == {"brightness": 30}


@pytest.mark.parametrize("phrase,delta", [
    ("dim the lamp", -25),
    ("make the lamp brighter", 25),
    ("brighten the bedroom light", 25),
    ("dim the lamp a little", -12),
    ("make the lamp a lot brighter", 50),
    ("dim the lamp by 10 percent", -10),
    ("turn the lamp down", -25),
    ("turn the lamp up", 25),
])
def test_relative_brightness(inv, phrase, delta):
    assert payload(inv, phrase) == {"brightness_delta": delta}


def test_step_is_configurable(inv):
    assert payload(inv, "dim the lamp", step=10) == {"brightness_delta": -10}


def test_an_explicit_level_beats_a_relative_word(inv):
    assert payload(inv, "dim the lamp to 20 percent") == {"brightness": 20}


# ── color ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,hue,sat,name", [
    ("make the lamp blue", 240, 100, "blue"),
    ("set the lamp to red", 0, 100, "red"),
    ("turn the bedroom light green", 120, 100, "green"),
    ("make the lamp purple", 275, 100, "purple"),
    ("make the bedroom light pink", 330, 55, "pink"),
    ("make the lamp orange", 30, 100, "orange"),
])
def test_colors(inv, phrase, hue, sat, name):
    p = payload(inv, phrase)
    assert (p["hue"], p["saturation"], p["color_name"]) == (hue, sat, name)


def test_a_color_for_every_light(inv):
    assert targets(inv, "set both lights to red") == BOTH


@pytest.mark.parametrize("phrase,kelvin,name", [
    ("turn the lamp to warm white", 2700, "warm white"),
    ("set the lamp to soft white", 2700, "warm white"),
    ("set the lamp to cool white", 5000, "cool white"),
    ("set the lamp to daylight", 6000, "daylight"),
    ("turn the lamp white", 4000, "white"),
    ("make the lamp warm", 2700, "warm white"),
])
def test_white_temperatures(inv, phrase, kelvin, name):
    p = payload(inv, phrase)
    assert (p["color_temp"], p["color_name"]) == (kelvin, name)


def test_common_mishears_are_honored_only_after_a_trigger_word(inv):
    assert payload(inv, "make the lamp read")["hue"] == 0      # "read" → red
    assert payload(inv, "set the lamp to blew")["hue"] == 240  # "blew" → blue
    # Without a trigger the alias is just a word.
    assert go(inv, "the lamp read") is None


def test_color_and_brightness_combine(inv):
    p = payload(inv, "set the lamp to blue at 30 percent")
    assert (p["hue"], p["brightness"]) == (240, 30)


def test_a_light_named_after_a_color_is_not_read_as_that_color():
    inv = LightInventory.from_dict({"light": [
        {"id": "blueroom", "names": ["blue room"]}, {"id": "lamp", "names": ["lamp"]}]})
    # "blue" belongs to the NAME here, so this is only a power command...
    assert payload(inv, "turn on the blue room") == {"power": True}
    # ...and the actual color after the name is still found.
    assert payload(inv, "make the blue room red")["hue"] == 0


def test_off_wins_over_everything_else_spoken(inv):
    assert payload(inv, "turn off the lamp and make it blue") == {"power": False}


def test_toggle(inv):
    assert payload(inv, "toggle the lamp") == {"toggle": True}


# ── pronouns: only right after the lights were used ─────────────────────────

@pytest.mark.parametrize("phrase,expect", [
    ("make it blue", {"hue": 240, "saturation": 100, "color_name": "blue"}),
    ("turn it off", {"power": False}),
    ("brighter", {"brightness_delta": 25}),
    ("dimmer", {"brightness_delta": -25}),
    ("dim it", {"brightness_delta": -25}),
])
def test_it_refers_to_the_last_lights_when_they_were_just_used(inv, phrase, expect):
    intent = go(inv, phrase, has_recent=True)
    assert intent is not None and intent.targets is None  # router supplies the lights
    assert intent.command.to_payload() == expect


@pytest.mark.parametrize("phrase", ["make it blue", "turn it off", "brighter", "dim it"])
def test_it_means_nothing_without_recent_light_use(inv, phrase):
    assert go(inv, phrase, has_recent=False) is None


@pytest.mark.parametrize("phrase", ["turn it up", "turn it down", "turn that up"])
def test_bare_up_and_down_stay_with_the_volume(inv, phrase):
    # Even right after using the lights, "turn it up" is far more likely the music.
    assert go(inv, phrase, has_recent=True) is None


def test_a_bare_all_word_needs_recent_use(inv):
    assert go(inv, "turn off both") is None
    intent = go(inv, "turn off both", has_recent=True)
    assert set(intent.targets.ids) == BOTH


# ── must NOT be light commands ───────────────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    "turn on the radio",
    "put on the radio",
    "play some music",
    "shuffle all my music",
    "skip",
    "stop",
    "what's the weather",
    "what's on my calendar",
    "is the lamp on",            # a question — must not turn it on
    "are the lights on",
    "what color is the lamp",
    "set an alarm for seven",
    "turn everything off",       # may mean the music
    "turn up the volume",
    "tell me a joke",
    "",
    "   ",
])
def test_not_a_light_command(inv, phrase):
    assert go(inv, phrase) is None


def test_no_inventory_means_no_lights():
    assert match_light_intent("turn on the lamp", None) is None


# ── ordering against media ───────────────────────────────────────────────────

def test_lights_are_matched_before_media_because_put_on_is_also_a_media_phrase(inv):
    # Media alone would read this as "play a track called 'the lamp'"...
    assert match_media_intent("put on the lamp")["action"] == "play"
    # ...which is why the pipeline asks the lights first:
    assert go(inv, "put on the lamp") is not None
    # while a genuine media phrase is left alone by the lights.
    assert go(inv, "put on the radio") is None
    assert match_media_intent("put on the radio") is not None
