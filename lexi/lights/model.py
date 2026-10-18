"""Light inventory, commands and spoken-name resolution.

Pure logic — no network, no hardware and no third-party imports — so it is
tested exhaustively without a bulb or a broker in the room.

A *light* is one physical bulb, known by a stable ``id`` (used in MQTT topics),
one or more spoken ``names``, and a ``mac`` (how it is found on the LAN — bulbs
get their IP from DHCP, so the MAC is the durable identity; ``host`` is only an
optional first-guess hint).
"""
from __future__ import annotations

import difflib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Words meaning "every light".
ALL_WORDS = frozenset({"all", "both", "every", "everything"})
# Plural light nouns also mean "every light" ("turn off the lights").
PLURAL_NOUNS = frozenset({"lights", "lamps", "bulbs"})
# Singular light nouns mean "a light, unspecified" (→ the default target).
SINGULAR_NOUNS = frozenset({"light", "lamp", "bulb", "lighting"})

_ARTICLES = frozenset({"the", "my", "a", "an", "our"})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Function words / verbs that must never be fuzzy-matched against a light name.
_FUZZY_STOP = frozenset({
    "turn", "switch", "put", "flip", "power", "set", "make", "change", "dim",
    "brighten", "brighter", "dimmer", "on", "off", "the", "a", "an", "my", "our",
    "to", "at", "and", "please", "percent", "light", "lights", "bulb", "bulbs",
    "it", "them", "up", "down", "all", "both", "every", "everything", "hey",
    "jarvis", "lexi", "okay", "ok", "can", "you", "could", "would", "of", "in",
})
_FUZZY_THRESHOLD = 0.84
_FUZZY_AMBIGUITY_GAP = 0.03


class LightError(RuntimeError):
    """A light could not be commanded. The message is short and speakable."""


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens. ``%`` is kept so "50%" survives as one token."""
    return re.findall(r"[a-z0-9%]+", text.lower())


def normalize_mac(mac: str) -> str:
    """Hex digits only, upper-case: 'aa-bb:cc' -> 'AABBCC'. Compare MACs in this
    form — devices report them with ':' or '-' depending on firmware."""
    return re.sub(r"[^0-9A-Fa-f]", "", mac or "").upper()


def _strip_articles(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens) - 1 and tokens[i] in _ARTICLES:
        i += 1
    return tokens[i:]


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Light:
    id: str
    names: tuple[str, ...]
    mac: str = ""   # normalized (hex only, upper-case); blank = find by ``host`` only
    host: str = ""  # optional IP hint; never required

    @property
    def display(self) -> str:
        """What Lexi says back — by convention the first configured name."""
        return self.names[0] if self.names else self.id

    def spoken(self) -> str:
        """``display`` with an article for use in a sentence ('the lamp')."""
        first = self.display.split()[0].lower() if self.display else ""
        return self.display if first in _ARTICLES else f"the {self.display}"


@dataclass(frozen=True)
class LightCommand:
    """What to do to a light. Every field is optional; several combine
    ("blue at 30 percent"). Unset fields are left as the bulb has them."""
    power: bool | None = None            # True on, False off
    toggle: bool = False
    brightness: int | None = None        # absolute, 1..100
    brightness_delta: int | None = None  # relative, -100..100
    hue: int | None = None               # 0..360
    saturation: int | None = None        # 0..100
    color_temp: int | None = None        # kelvin, 2500..6500
    color_name: str = ""                 # for speech only

    def __post_init__(self) -> None:
        if self.brightness is not None and self.brightness_delta is not None:
            raise ValueError("brightness and brightness_delta are mutually exclusive")
        if self.hue is not None and self.color_temp is not None:
            raise ValueError("a color (hue) and a color temperature are mutually exclusive")

    def is_empty(self) -> bool:
        return (self.power is None and not self.toggle and self.brightness is None
                and self.brightness_delta is None and self.hue is None
                and self.saturation is None and self.color_temp is None)

    def to_payload(self) -> dict:
        """JSON-safe dict of only the set fields (the MQTT command body)."""
        out: dict = {}
        if self.power is not None:
            out["power"] = self.power
        if self.toggle:
            out["toggle"] = True
        for key in ("brightness", "brightness_delta", "hue", "saturation", "color_temp"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.color_name:
            out["color_name"] = self.color_name
        return out

    @classmethod
    def from_payload(cls, data: object) -> "LightCommand":
        """Validate + clamp an untrusted payload (it arrives over MQTT). Raises
        ``ValueError`` on anything malformed or empty."""
        if not isinstance(data, dict):
            raise ValueError("payload must be a JSON object")

        def num(name: str, lo: int, hi: int) -> int | None:
            v = data.get(name)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a number")
            return max(lo, min(hi, int(round(v))))

        power = data.get("power")
        if power is not None and not isinstance(power, bool):
            raise ValueError("power must be true or false")
        toggle = data.get("toggle", False)
        if not isinstance(toggle, bool):
            raise ValueError("toggle must be true or false")
        cmd = cls(
            power=power,
            toggle=toggle,
            brightness=num("brightness", 1, 100),
            brightness_delta=num("brightness_delta", -100, 100),
            hue=num("hue", 0, 360),
            saturation=num("saturation", 0, 100),
            color_temp=num("color_temp", 2500, 6500),
            color_name=str(data.get("color_name", ""))[:40],
        )
        if cmd.is_empty():
            raise ValueError("empty command")
        return cmd

    def describe(self) -> str:
        """The tail of a confirmation sentence: 'on', 'set to blue at 30 percent'."""
        if self.hue is not None or self.color_temp is not None:
            what = self.color_name or (
                f"{self.color_temp} kelvin" if self.color_temp is not None else "that color")
            out = f"set to {what}"
            if self.brightness is not None:
                out += f" at {self.brightness} percent"
            return out
        if self.brightness is not None:
            return f"at {self.brightness} percent"
        if self.brightness_delta is not None:
            return "brighter" if self.brightness_delta > 0 else "dimmer"
        if self.power is not None:
            return "on" if self.power else "off"
        if self.toggle:
            return "toggled"
        return "updated"


@dataclass(frozen=True)
class Targets:
    """Which lights a command applies to, plus how to say it back."""
    ids: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class _Span:
    start: int
    end: int          # exclusive
    ids: tuple[str, ...]
    ntok: int
    chars: int

    def indices(self) -> range:
        return range(self.start, self.end)


# ── inventory ────────────────────────────────────────────────────────────────

class LightInventory:
    """The set of lights Lexi knows, plus spoken-name resolution."""

    def __init__(self, lights: list[Light], groups: dict[str, list[str]] | None = None,
                 default_target: str = "all"):
        if not lights:
            raise ValueError("no lights defined")
        self._lights: dict[str, Light] = {}
        for light in lights:
            if light.id in self._lights:
                raise ValueError(f"duplicate light id {light.id!r}")
            self._lights[light.id] = light
        self._groups: dict[str, tuple[str, ...]] = {}
        for name, members in (groups or {}).items():
            key = name.strip().lower()
            for m in members:
                if m not in self._lights:
                    raise ValueError(f"group {name!r} lists unknown light {m!r}")
            self._groups[key] = tuple(members)
        dt = default_target.strip().lower()
        if dt != "all" and dt not in self._lights and dt not in self._groups:
            raise ValueError(f"default_target {default_target!r} is not 'all', a light id or a group")
        self.default_target = dt
        self._index = self._build_index()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "LightInventory":
        raw = data.get("light")
        if not isinstance(raw, list) or not raw:
            raise ValueError("lights file needs at least one [[light]] table")
        lights: list[Light] = []
        for i, entry in enumerate(raw, 1):
            if not isinstance(entry, dict):
                raise ValueError(f"[[light]] #{i} is not a table")
            lid = str(entry.get("id", "")).strip().lower()
            if not _ID_RE.match(lid):
                raise ValueError(f"[[light]] #{i}: id must match [a-z0-9][a-z0-9_-]* (got {lid!r})")
            names = entry.get("names") or [lid]
            if (not isinstance(names, list)
                    or not all(isinstance(n, str) and n.strip() for n in names)):
                raise ValueError(f"light {lid!r}: names must be a list of non-empty strings")
            lights.append(Light(
                id=lid,
                names=tuple(n.strip() for n in names),
                mac=normalize_mac(str(entry.get("mac", ""))),
                host=str(entry.get("host", "")).strip(),
            ))
        groups = data.get("groups", {})
        if not isinstance(groups, dict):
            raise ValueError("[groups] must be a table")
        return cls(lights, {str(k): list(v) for k, v in groups.items()},
                   str(data.get("default_target", "all")))

    @classmethod
    def from_file(cls, path: str | Path) -> "LightInventory | None":
        """Load ``lights.toml``. ``None`` if the file does not exist (lights simply
        aren't configured); ``ValueError`` if it exists but is malformed."""
        p = Path(path)
        if not p.is_file():
            return None
        try:
            with open(p, "rb") as fh:
                data = tomllib.load(fh)
            return cls.from_dict(data)
        except (tomllib.TOMLDecodeError, ValueError) as exc:
            raise ValueError(f"{p}: {exc}") from exc

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._lights)

    def __iter__(self):
        return iter(self._lights.values())

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._lights)

    def get(self, light_id: str) -> Light | None:
        return self._lights.get(light_id)

    def resolve_name(self, phrase: str) -> Targets | None:
        """Resolve a bare name/id ('lamp', 'bedroom light') — used by the CLI."""
        tokens = tokenize(phrase)
        if not tokens:
            return None
        if len(tokens) == 1 and tokens[0] in self._lights:
            return Targets((tokens[0],), self.label_for((tokens[0],)))
        targets, _ = self.match(tokens)
        if targets is not None:
            return targets
        if tokens[0] in ALL_WORDS or tokens[0] in PLURAL_NOUNS:
            return self.all_targets()
        return None

    def all_targets(self) -> Targets:
        return Targets(self.ids, self.label_for(self.ids))

    def default_targets(self) -> Targets:
        """What an unqualified "turn on the light" means."""
        if self.default_target == "all":
            return self.all_targets()
        ids = (self.default_target,) if self.default_target in self._lights \
            else self._groups[self.default_target]
        return Targets(tuple(ids), self.label_for(tuple(ids)))

    def label_for(self, ids: tuple[str, ...] | list[str]) -> str:
        ids = tuple(ids)
        if not ids:
            return ""
        if len(self._lights) > 1 and set(ids) == set(self._lights):
            return "both lights" if len(self._lights) == 2 else "all lights"
        names = [self._lights[i].display for i in ids if i in self._lights]
        if len(names) <= 1:
            return names[0] if names else ""
        return ", ".join(names[:-1]) + " and " + names[-1]

    # -- name matching ------------------------------------------------------
    def _build_index(self) -> list[tuple[tuple[str, ...], tuple[str, ...], str]]:
        index: dict[tuple[str, ...], tuple[str, ...]] = {}

        def add(name: str, ids: tuple[str, ...], owner: str) -> None:
            tokens = tuple(_strip_articles(tokenize(name)))
            if not tokens:
                return
            if tokens in index and index[tokens] != ids:
                raise ValueError(f"the name {name!r} ({owner}) is ambiguous: it is already "
                                 f"used by another light or group")
            index[tokens] = ids

        for light in self._lights.values():
            for name in light.names:
                add(name, (light.id,), f"light {light.id}")
            add(light.id, (light.id,), f"light {light.id}")
        for gname, members in self._groups.items():
            add(gname, members, f"group {gname}")
        out = [(toks, ids, " ".join(toks)) for toks, ids in index.items()]
        out.sort(key=lambda e: (-len(e[0]), -len(e[2])))
        return out

    def _exact_spans(self, tokens: list[str]) -> list[_Span]:
        spans: list[_Span] = []
        for name_tokens, ids, joined in self._index:
            n = len(name_tokens)
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i:i + n]) == name_tokens:
                    spans.append(_Span(i, i + n, ids, n, len(joined)))
        return spans

    @staticmethod
    def _select(spans: list[_Span]) -> list[_Span]:
        """Overlapping/adjacent matches form one cluster describing ONE light
        ("bedroom light" also contains "bedroom"; "bedroom lamp" is a compound of
        two names) — keep the longest. Separate clusters ("the lamp AND the
        bedroom light") are all kept, so they union."""
        spans = sorted(spans, key=lambda s: (s.start, -s.ntok))
        clusters: list[list[_Span]] = []
        cur: list[_Span] = []
        cur_end = -1
        for s in spans:
            if cur and s.start <= cur_end:
                cur.append(s)
                cur_end = max(cur_end, s.end)
            else:
                if cur:
                    clusters.append(cur)
                cur = [s]
                cur_end = s.end
        if cur:
            clusters.append(cur)
        return [max(c, key=lambda s: (s.ntok, s.chars)) for c in clusters]

    def _fuzzy(self, tokens: list[str]) -> tuple[Targets, set[int]] | None:
        """Speech recognition splits/garbles names ("bed room", "lamps"): allow a
        close match. Refuses when two different lights are about equally close."""
        candidates: list[tuple[float, tuple[str, ...], range]] = []
        n = len(tokens)
        for size in (1, 2, 3):
            for i in range(n - size + 1):
                window = tokens[i:i + size]
                if any(t in _FUZZY_STOP or any(c.isdigit() for c in t) or "%" in t for t in window):
                    continue
                spaced, squashed = " ".join(window), "".join(window)
                for _toks, ids, joined in self._index:
                    r = max(difflib.SequenceMatcher(None, spaced, joined).ratio(),
                            difflib.SequenceMatcher(None, squashed, joined.replace(" ", "")).ratio())
                    if r >= _FUZZY_THRESHOLD:
                        candidates.append((r, ids, range(i, i + size)))
        if not candidates:
            return None
        candidates.sort(key=lambda c: -c[0])
        best_ratio, best_ids, best_range = candidates[0]
        for ratio, ids, _rng in candidates[1:]:
            if ids != best_ids and best_ratio - ratio < _FUZZY_AMBIGUITY_GAP:
                return None  # two different lights about equally close — don't guess
        return Targets(best_ids, self.label_for(best_ids)), set(best_range)

    def match(self, tokens: list[str]) -> tuple[Targets | None, set[int]]:
        """Which lights do these tokens *name*? Returns (targets, token indices the
        names consumed) — the intent parser masks those so a light called "blue
        room" is never read as the color blue. Order: exact names → "all"/plural
        nouns → fuzzy names."""
        spans = self._exact_spans(tokens)
        if spans:
            chosen = self._select(spans)
            ids: list[str] = []
            for s in chosen:
                for i in s.ids:
                    if i not in ids:
                        ids.append(i)
            consumed = {i for s in spans for i in s.indices()}
            return Targets(tuple(ids), self.label_for(tuple(ids))), consumed
        # "the lights" (plural noun) is every light; "all"/"both" only counts when a
        # light noun is also spoken — a bare "turn everything off" may mean the
        # music, so it is NOT taken as a light command here.
        if any(t in PLURAL_NOUNS for t in tokens) or (
                any(t in ALL_WORDS for t in tokens)
                and any(t in SINGULAR_NOUNS or t in PLURAL_NOUNS for t in tokens)):
            return self.all_targets(), set()
        fuzzy = self._fuzzy(tokens)
        if fuzzy is not None:
            return fuzzy
        return None, set()
