"""Run a light command: through Bilbo's MQTT bridge when it is up, otherwise —
or if the bridge fails to answer — directly from this device.

Bilbo is sometimes powered off, and the lights must still work then, so "direct"
is a genuine fallback rather than a debug mode. The MQTT client keeps a
background connection and learns the bridge's status from a retained message, so
a dead Bilbo costs nothing per command: ``bridge_online`` is simply false and we
go straight to the direct path.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .model import LightCommand, LightInventory, Targets

logger = logging.getLogger(__name__)


class LightBackend(Protocol):
    def execute(self, light_ids: list[str], cmd: LightCommand) -> dict[str, str | None]:
        """Apply ``cmd`` to each light. ``{id: None}`` on success, else a short
        reason. Never raises for a per-light failure."""


class BridgeBackend(LightBackend, Protocol):
    @property
    def bridge_online(self) -> bool: ...


@dataclass
class LightResult:
    targets: tuple[str, ...]
    via: dict[str, str] = field(default_factory=dict)      # id -> "bridge" | "direct"
    errors: dict[str, str] = field(default_factory=dict)   # id -> short reason

    @property
    def ok_ids(self) -> tuple[str, ...]:
        return tuple(i for i in self.targets if i not in self.errors)

    @property
    def failed_ids(self) -> tuple[str, ...]:
        return tuple(i for i in self.targets if i in self.errors)

    @property
    def success(self) -> bool:
        return not self.errors


class LightRouter:
    def __init__(
        self,
        inventory: LightInventory,
        *,
        mqtt: BridgeBackend | None = None,
        direct: LightBackend | None = None,
        recent_s: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._inv = inventory
        self._mqtt = mqtt
        self._direct = direct
        self._recent_s = recent_s
        self._clock = clock
        self._last: Targets | None = None
        self._last_at = 0.0

    @property
    def inventory(self) -> LightInventory:
        return self._inv

    # -- "it" support -------------------------------------------------------
    @property
    def last_targets(self) -> Targets | None:
        return self._last

    def has_recent(self) -> bool:
        """Were the lights used a moment ago? Only then is a bare "it"/"brighter"
        understood as referring to them."""
        return self._last is not None and (self._clock() - self._last_at) <= self._recent_s

    def remember(self, targets: Targets) -> None:
        self._last = targets
        self._last_at = self._clock()

    def warm_up(self) -> None:
        """Connect to the bulbs in the background so the FIRST spoken command is fast.
        Finding a bulb by broadcast takes ~5 s; doing it at startup hides that, and the
        connections stay cached for every later command. Never blocks, never raises."""
        state = getattr(self._direct, "state", None)
        if not callable(state):
            return
        ids = list(self._inv.ids)

        def run() -> None:
            try:
                state(ids)
            except Exception:  # noqa: BLE001 - warming is best-effort
                logger.debug("light warm-up failed", exc_info=True)

        threading.Thread(target=run, name="lexi-lights-warmup", daemon=True).start()

    # -- execution ----------------------------------------------------------
    def execute(self, light_ids: list[str] | tuple[str, ...], cmd: LightCommand) -> LightResult:
        ids = list(dict.fromkeys(light_ids))
        result = LightResult(tuple(ids))
        pending = ids

        if self._mqtt is not None and self._mqtt.bridge_online:
            try:
                outcome = self._mqtt.execute(pending, cmd)
            except Exception as exc:  # a broker glitch must never lose the command
                logger.warning("bridge call failed (%s); falling back to direct", exc)
                outcome = {i: f"bridge error: {exc}" for i in pending}
            still: list[str] = []
            for i in pending:
                err = outcome.get(i, "no reply from the bridge")
                if err is None:
                    result.via[i] = "bridge"
                else:
                    still.append(i)
                    result.errors[i] = err
            pending = still
            if pending:
                logger.info("bridge did not complete %s: %s", pending,
                            {i: result.errors[i] for i in pending})

        if pending and self._direct is not None:
            try:
                outcome = self._direct.execute(pending, cmd)
            except Exception as exc:
                logger.warning("direct call failed: %s", exc)
                outcome = {i: "unreachable" for i in pending}
            for i in pending:
                err = outcome.get(i, "unreachable")
                if err is None:
                    result.via[i] = "direct"
                    result.errors.pop(i, None)  # the fallback succeeded; drop the bridge error
                else:
                    result.errors[i] = err
        else:
            for i in pending:
                result.errors.setdefault(i, "no light controller is available")
        return result

    def close(self) -> None:
        for backend in (self._mqtt, self._direct):
            closer = getattr(backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - shutdown best-effort
                    logger.debug("backend close failed", exc_info=True)


# ── speech ───────────────────────────────────────────────────────────────────

def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def summarize(inventory: LightInventory, cmd: LightCommand, result: LightResult) -> str:
    """One short sentence to say back: 'Lamp set to blue.', or which light failed."""
    ok = result.ok_ids
    head = f"{inventory.label_for(ok)} {cmd.describe()}" if ok else ""
    if result.success:
        return _cap(head) + "."

    cant: list[str] = []
    unreachable: list[str] = []
    for lid in result.failed_ids:
        light = inventory.get(lid)
        spoken = light.spoken() if light else lid
        reason = result.errors[lid]
        if reason.startswith("can't") or reason.startswith("isn't"):
            cant.append(f"{spoken} {reason}")
        else:
            unreachable.append(spoken)
    rest = [_cap(c) if not head else c for c in cant]
    if unreachable:
        rest.append(f"I couldn't reach {_join(unreachable)}")
    if head:
        return f"{_cap(head)}, but {' and '.join(rest)}."
    return _cap(" and ".join(rest)) + "."
