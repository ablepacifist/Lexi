"""Light inventory, name resolution and commands — pure logic, no hardware."""
import pytest

from lexi.lights.model import (
    Light,
    LightCommand,
    LightInventory,
    normalize_mac,
    tokenize,
)
from conftest import make_inventory


# ── inventory validation ─────────────────────────────────────────────────────

def test_mac_is_normalized_from_any_format(inv):
    assert inv.get("lamp").mac == "AABBCCDDEE01"
    assert inv.get("bedroom").mac == "AABBCCDDEE02"  # was given with dashes, lower-case
    assert normalize_mac("5c:62:8B-00-11-22") == "5C628B001122"
    assert normalize_mac("") == ""


def test_needs_at_least_one_light():
    with pytest.raises(ValueError, match="at least one"):
        LightInventory.from_dict({})


@pytest.mark.parametrize("bad_id", ["", "Has Space", "-lead", "UPPER!"])
def test_bad_ids_rejected(bad_id):
    with pytest.raises(ValueError, match="id must match"):
        LightInventory.from_dict({"light": [{"id": bad_id}]})


def test_duplicate_id_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        LightInventory.from_dict({"light": [{"id": "a"}, {"id": "a"}]})


def test_a_name_shared_by_two_lights_is_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        LightInventory.from_dict({"light": [
            {"id": "a", "names": ["lamp"]}, {"id": "b", "names": ["the lamp"]}]})


def test_group_with_unknown_member_rejected():
    with pytest.raises(ValueError, match="unknown light"):
        make_inventory(groups={"upstairs": ["lamp", "nope"]})


def test_default_target_must_exist():
    with pytest.raises(ValueError, match="default_target"):
        make_inventory(default_target="garage")


def test_from_file_missing_is_none_and_malformed_is_a_clear_error(tmp_path):
    assert LightInventory.from_file(tmp_path / "nope.toml") is None
    bad = tmp_path / "lights.toml"
    bad.write_text("this is = not [valid toml")
    with pytest.raises(ValueError, match="lights.toml"):
        LightInventory.from_file(bad)


def test_from_file_loads_the_example_shape(tmp_path):
    f = tmp_path / "lights.toml"
    f.write_text('''
default_target = "all"
[[light]]
id = "lamp"
names = ["lamp"]
mac = "AA:BB:CC:DD:EE:01"
[[light]]
id = "bedroom"
names = ["bedroom light", "bedroom"]
mac = "AA:BB:CC:DD:EE:02"
''')
    loaded = LightInventory.from_file(f)
    assert loaded.ids == ("lamp", "bedroom")


# ── labels (what Lexi says back) ─────────────────────────────────────────────

def test_labels(inv):
    assert inv.label_for(("lamp",)) == "lamp"
    assert inv.label_for(("bedroom",)) == "bedroom light"  # first configured name
    assert inv.label_for(("lamp", "bedroom")) == "both lights"
    three = LightInventory.from_dict({"light": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    assert three.label_for(("a", "b", "c")) == "all lights"
    assert three.label_for(("a", "b")) == "a and b"


def test_spoken_adds_an_article_only_when_needed():
    assert Light("l", ("lamp",)).spoken() == "the lamp"
    assert Light("l", ("the lamp",)).spoken() == "the lamp"


# ── name resolution ──────────────────────────────────────────────────────────

def ids(inv, phrase):
    targets, _consumed = inv.match(tokenize(phrase))
    return targets.ids if targets else None


def test_exact_names(inv):
    assert ids(inv, "turn on the lamp") == ("lamp",)
    assert ids(inv, "turn on the bedroom light") == ("bedroom",)
    assert ids(inv, "turn on the bedroom") == ("bedroom",)


def test_two_named_lights_union(inv):
    assert set(ids(inv, "turn off the lamp and the bedroom light")) == {"lamp", "bedroom"}
    assert set(ids(inv, "turn off lamp and bedroom")) == {"lamp", "bedroom"}


def test_adjacent_names_form_one_compound_and_the_longer_wins(inv):
    # "bedroom lamp" is inherently ambiguous between the two; the longer name wins.
    assert ids(inv, "turn on the bedroom lamp") == ("bedroom",)


def test_plural_noun_means_every_light(inv):
    assert set(ids(inv, "turn off the lights")) == {"lamp", "bedroom"}
    assert set(ids(inv, "turn off both lights")) == {"lamp", "bedroom"}
    assert set(ids(inv, "turn off all the lights")) == {"lamp", "bedroom"}


def test_all_word_without_a_light_noun_is_not_every_light(inv):
    # "turn everything off" may well mean the music; never assume it is the lights.
    assert ids(inv, "turn everything off") is None
    assert ids(inv, "turn off all") is None


def test_a_specific_name_beats_the_plural(inv):
    assert ids(inv, "turn on the bedroom lights") == ("bedroom",)


def test_fuzzy_handles_speech_recognition_splitting_a_name(inv):
    assert ids(inv, "turn on the bed room light") == ("bedroom",)


def test_fuzzy_ignores_unrelated_words(inv):
    assert ids(inv, "turn on the radio") is None
    assert ids(inv, "what is the weather") is None


def test_fuzzy_refuses_to_guess_between_two_equally_close_names():
    # "kitchan" is exactly as close to "kitchen" as to "kitchin" (0.857 each).
    inv = LightInventory.from_dict({"light": [
        {"id": "a", "names": ["kitchen"]}, {"id": "b", "names": ["kitchin"]}]})
    assert ids(inv, "turn on the kitchan") is None
    # ...but a clear winner is still taken.
    assert ids(inv, "turn on the kitchens") == ("a",)


def test_consumed_indices_cover_the_matched_name(inv):
    tokens = tokenize("make the bedroom light red")
    targets, consumed = inv.match(tokens)
    assert targets.ids == ("bedroom",)
    assert {tokens[i] for i in consumed} == {"bedroom", "light"}


def test_groups_resolve_to_their_members():
    inv = make_inventory(groups={"upstairs": ["lamp", "bedroom"]})
    assert set(ids(inv, "turn on upstairs")) == {"lamp", "bedroom"}


def test_default_target_all_or_specific():
    assert set(make_inventory().default_targets().ids) == {"lamp", "bedroom"}
    assert make_inventory(default_target="bedroom").default_targets().ids == ("bedroom",)


def test_resolve_name_for_the_cli(inv):
    assert inv.resolve_name("lamp").ids == ("lamp",)
    assert inv.resolve_name("bedroom light").ids == ("bedroom",)
    assert set(inv.resolve_name("all").ids) == {"lamp", "bedroom"}
    assert inv.resolve_name("garage") is None


# ── commands ─────────────────────────────────────────────────────────────────

def test_command_exclusivity():
    with pytest.raises(ValueError):
        LightCommand(brightness=50, brightness_delta=10)
    with pytest.raises(ValueError):
        LightCommand(hue=10, color_temp=3000)


def test_payload_round_trip():
    cmd = LightCommand(hue=240, saturation=100, brightness=30, color_name="blue")
    assert LightCommand.from_payload(cmd.to_payload()) == cmd


def test_to_payload_only_carries_set_fields():
    assert LightCommand(power=True).to_payload() == {"power": True}
    assert LightCommand(brightness_delta=-25).to_payload() == {"brightness_delta": -25}


def test_from_payload_clamps_out_of_range_values():
    cmd = LightCommand.from_payload({"brightness": 500, "hue": -5, "saturation": 999})
    assert (cmd.brightness, cmd.hue, cmd.saturation) == (100, 0, 100)
    assert LightCommand.from_payload({"color_temp": 100}).color_temp == 2500
    assert LightCommand.from_payload({"color_temp": 99999}).color_temp == 6500


@pytest.mark.parametrize("bad", [
    None, [], "on", {}, {"power": "yes"}, {"toggle": 1}, {"brightness": "high"},
    {"brightness": True}, {"brightness": 5, "brightness_delta": 5},
])
def test_from_payload_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        LightCommand.from_payload(bad)


@pytest.mark.parametrize("cmd,text", [
    (LightCommand(power=True), "on"),
    (LightCommand(power=False), "off"),
    (LightCommand(toggle=True), "toggled"),
    (LightCommand(brightness=30), "at 30 percent"),
    (LightCommand(brightness_delta=25), "brighter"),
    (LightCommand(brightness_delta=-25), "dimmer"),
    (LightCommand(hue=240, saturation=100, color_name="blue"), "set to blue"),
    (LightCommand(hue=240, saturation=100, brightness=30, color_name="blue"),
     "set to blue at 30 percent"),
    (LightCommand(color_temp=2700, color_name="warm white"), "set to warm white"),
])
def test_describe(cmd, text):
    assert cmd.describe() == text
