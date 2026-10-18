"""Spoken-command grammar for the lights — local, so it works with the brain off.

Like the media grammar this is deliberately literal: a phrase only becomes a light
command when it carries BOTH an action (on/off, a brightness, a color …) AND
something that says it is about the lights (a configured light name, a light noun,
or — right after you just used the lights — "it"). Anything else falls through to
media and then the brain, so "turn on the radio" or "what's on my calendar" are
never mistaken for lights.

Ordering in the pipeline is lights → media → brain, because ``put on …`` is also a
media phrase ("put on the desk light" must not try to play a song).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .model import (
    ALL_WORDS,
    PLURAL_NOUNS,
    SINGULAR_NOUNS,
    LightCommand,
    LightInventory,
    Targets,
    tokenize,
)

_LEAD = frozenset({"hey", "jarvis", "lexi", "okay", "ok", "please", "can", "could",
                   "would", "will", "you", "just", "yeah", "yes"})
# A question is never a command ("is the lamp on?" must not turn it on).
_QUESTION_START = frozenset({"is", "are", "was", "were", "does", "do", "did", "what",
                             "which", "how", "why", "when", "where", "who", "whether"})
_POWER_VERBS = frozenset({"turn", "switch", "put", "flip", "power"})
_TRAIL_FILLER = frozenset({"please", "now", "thanks", "thank", "you", "too", "for", "me"})
_PRONOUNS = frozenset({"it", "them"})

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}

# Relative brightness. "Strong" words are unambiguous about the lights; "weak" ones
# ("up", "down", "raise", "lower") also mean volume etc., so they only count when a
# light is named.
_DOWN_STRONG = frozenset({"dim", "dimmer", "darker"})
_DOWN_WEAK = frozenset({"down", "lower", "decrease", "reduce"})
_UP_STRONG = frozenset({"brighter", "brighten"})
_UP_WEAK = frozenset({"up", "raise", "increase"})
_SMALL = frozenset({"little", "bit", "slightly", "touch"})
_LARGE = frozenset({"lot", "much", "way"})
_FULL = frozenset({"full", "max", "maximum", "brightest"})
_MINIMUM = frozenset({"minimum", "lowest", "dimmest"})

# name -> (hue 0..360, saturation 0..100)
_COLORS: dict[str, tuple[int, int]] = {
    "red": (0, 100), "orange": (30, 100), "yellow": (60, 100), "green": (120, 100),
    "cyan": (180, 100), "teal": (180, 100), "aqua": (180, 100), "turquoise": (175, 100),
    "blue": (240, 100), "purple": (275, 100), "violet": (275, 100),
    "magenta": (300, 100), "pink": (330, 55),
}
# Whisper's common mishears of a color — honored ONLY right after a trigger word
# ("make it read"), never on their own.
_COLOR_ALIASES = {"read": "red", "blew": "blue", "bloo": "blue"}
_COLOR_TRIGGERS = frozenset({"make", "set", "change", "turn", "to", "color", "colour"})
# (tokens) -> (kelvin, spoken name). Tapo bulbs span roughly 2500-6500 K.
_WHITE_PAIRS: dict[tuple[str, str], tuple[int, str]] = {
    ("warm", "white"): (2700, "warm white"), ("soft", "white"): (2700, "warm white"),
    ("neutral", "white"): (4000, "neutral white"),
    ("cool", "white"): (5000, "cool white"), ("cold", "white"): (5000, "cool white"),
    ("day", "light"): (6000, "daylight"),
}
_WHITE_SINGLE: dict[str, tuple[int, str]] = {
    "white": (4000, "white"), "daylight": (6000, "daylight"),
}
_WHITE_NEEDS_TRIGGER: dict[str, tuple[int, str]] = {
    "warm": (2700, "warm white"), "cool": (5000, "cool white"),
}


@dataclass(frozen=True)
class LightIntent:
    command: LightCommand
    targets: Targets | None      # None = "it": the caller substitutes the last-used lights
    defaulted: bool = False      # no light was named; targets came from default_target


# ── numbers ──────────────────────────────────────────────────────────────────

def _numbers(tokens: list[str], masked: set[int]) -> list[tuple[int, int, int, bool]]:
    """Every number in the phrase as (start, end, value, had_percent_sign).
    Handles digits ("50", "50%") and words ("fifty", "twenty five", "a hundred")."""
    out: list[tuple[int, int, int, bool]] = []
    n = len(tokens)
    i = 0
    while i < n:
        if i in masked:
            i += 1
            continue
        t = tokens[i]
        m = re.fullmatch(r"(\d{1,3})(%?)", t)
        if m:
            out.append((i, i + 1, int(m.group(1)), bool(m.group(2))))
            i += 1
            continue
        if t in ("a", "one") and i + 1 < n and tokens[i + 1] == "hundred":
            out.append((i, i + 2, 100, False))
            i += 2
            continue
        if t == "hundred":
            out.append((i, i + 1, 100, False))
            i += 1
            continue
        if t in _TENS:
            value, j = _TENS[t], i + 1
            if j < n and j not in masked and 1 <= _UNITS.get(tokens[j], 0) <= 9:
                value += _UNITS[tokens[j]]
                j += 1
            out.append((i, j, value, False))
            i = j
            continue
        if t in _UNITS:
            out.append((i, i + 1, _UNITS[t], False))
        i += 1
    return out


def _percent_after(tokens: list[str], end: int, has_sign: bool) -> bool:
    return has_sign or (end < len(tokens) and tokens[end] == "percent")


# ── pieces of a command ──────────────────────────────────────────────────────

def _power(tokens: list[str]) -> bool | None:
    """on/off — but only where it is a power word: right after a verb ("turn on"),
    or trailing ("the lamp off"). "dim the light on the desk" is a location."""
    for i, t in enumerate(tokens):
        if t not in ("on", "off"):
            continue
        prev = tokens[i - 1] if i else ""
        trailing = all(x in _TRAIL_FILLER for x in tokens[i + 1:])
        if prev in _POWER_VERBS or trailing:
            return t == "on"
    return None


def _absolute_brightness(tokens: list[str], masked: set[int]) -> int | None:
    for start, end, value, sign in _numbers(tokens, masked):
        prev = tokens[start - 1] if start else ""
        if prev == "by":
            continue  # "dim by 20 percent" is relative, handled elsewhere
        if _percent_after(tokens, end, sign) or prev in ("to", "at", "brightness", "level"):
            if 0 <= value <= 100:
                return value
    for i, t in enumerate(tokens):
        if i in masked:
            continue
        if t in _FULL:
            return 100
        if t in _MINIMUM:
            return 1
    return None


def _relative_brightness(tokens: list[str], masked: set[int], step: int) -> tuple[int | None, bool]:
    """(signed delta or None, whether a *strong* light-specific word triggered it)."""
    direction = 0
    strong = False
    for i, t in enumerate(tokens):
        if i in masked:
            continue
        if t in _DOWN_STRONG:
            direction, strong = -1, True
            break
        if t in _UP_STRONG:
            direction, strong = 1, True
            break
        if t in _DOWN_WEAK:
            direction = -1
            break
        if t in _UP_WEAK:
            direction = 1
            break
    if direction == 0:
        return None, False
    amount = step
    for start, end, value, sign in _numbers(tokens, masked):  # "dim by 20 percent"
        if start and tokens[start - 1] == "by" and _percent_after(tokens, end, sign) and 0 < value <= 100:
            amount = value
            break
    else:
        if any(t in _SMALL for t in tokens):
            amount = max(5, step // 2)
        elif any(t in _LARGE for t in tokens):
            amount = min(100, step * 2)
    return direction * amount, strong


def _color(tokens: list[str], masked: set[int]) -> tuple[int | None, int | None, int | None, str]:
    """(hue, saturation, kelvin, spoken name) — the first color mentioned."""
    n = len(tokens)
    for i in range(n):
        if i in masked:
            continue
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < n and (i + 1) not in masked else ""
        if (t, nxt) in _WHITE_PAIRS:
            k, name = _WHITE_PAIRS[(t, nxt)]
            return None, None, k, name
        trigger_before = any(x in _COLOR_TRIGGERS for x in tokens[max(0, i - 4):i])
        if t in _COLORS:
            h, s = _COLORS[t]
            return h, s, None, t
        if t in _COLOR_ALIASES and trigger_before:
            name = _COLOR_ALIASES[t]
            h, s = _COLORS[name]
            return h, s, None, name
        if t in _WHITE_SINGLE:
            k, name = _WHITE_SINGLE[t]
            return None, None, k, name
        if t in _WHITE_NEEDS_TRIGGER and trigger_before:
            k, name = _WHITE_NEEDS_TRIGGER[t]
            return None, None, k, name
    return None, None, None, ""


def _parse(tokens: list[str], masked: set[int], step: int) -> tuple[LightCommand, bool, bool]:
    """Build the command from the tokens. Returns (command, strong_relative,
    weak_only) — ``weak_only`` is a command that is nothing but "up"/"down"-style
    words, which on a bare "it" more likely mean the volume than the lights."""
    power = _power(tokens)
    toggle = "toggle" in tokens
    absolute = _absolute_brightness(tokens, masked)
    delta, strong = _relative_brightness(tokens, masked, step)
    hue, sat, kelvin, cname = _color(tokens, masked)

    if absolute == 0:  # "set the lamp to zero percent" means off
        power, absolute = False, None
    if absolute is not None:
        delta = None  # an explicit level beats a relative nudge
    if power is False:
        return LightCommand(power=False), False, False  # "off" wins over anything else spoken
    cmd = LightCommand(
        power=power,
        toggle=toggle,
        brightness=absolute,
        brightness_delta=delta,
        hue=hue,
        saturation=sat,
        color_temp=kelvin,
        color_name=cname,
    )
    weak_only = (delta is not None and not strong and power is None and not toggle
                 and absolute is None and hue is None and kelvin is None)
    return cmd, strong, weak_only


def _strip_lead(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens) - 1 and tokens[i] in _LEAD:
        i += 1
    return tokens[i:]


# ── public entry ─────────────────────────────────────────────────────────────

def match_light_intent(
    text: str,
    inventory: LightInventory | None,
    *,
    has_recent: bool = False,
    step: int = 25,
) -> LightIntent | None:
    """Turn a transcript into a light command, or ``None`` if it isn't one.

    ``has_recent`` says the lights were used a moment ago; only then is a bare
    "it"/"them" ("make it blue", "turn it off", "brighter") understood as
    referring to them — otherwise those phrases are left to media / the brain.
    """
    if inventory is None or len(inventory) == 0:
        return None
    tokens = _strip_lead(tokenize(text))
    if not tokens or tokens[0] in _QUESTION_START:
        return None

    named, consumed = inventory.match(tokens)
    command, strong_relative, weak_only = _parse(tokens, consumed, step)
    if command.is_empty():
        return None

    if named is not None:
        return LightIntent(command, named)

    has_noun = any(t in SINGULAR_NOUNS or t in PLURAL_NOUNS for t in tokens)
    if has_noun:  # "turn on the light" — nothing specific spoken
        return LightIntent(command, inventory.default_targets(), defaulted=True)

    if has_recent:
        if any(t in ALL_WORDS for t in tokens):
            return LightIntent(command, inventory.all_targets())
        if strong_relative or (any(t in _PRONOUNS for t in tokens) and not weak_only):
            return LightIntent(command, None)
    return None
