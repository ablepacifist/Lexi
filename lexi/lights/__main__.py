"""Lights command line.

    python -m lexi.lights discover            find Tapo bulbs on the LAN (name your lights from this)
    python -m lexi.lights identify lamp       blink one bulb so you can see which one it is
    python -m lexi.lights say "turn the lamp blue"   the voice grammar, without the voice
    python -m lexi.lights state [lamp]        what the bulbs report right now
    python -m lexi.lights bridge              run the MQTT bridge (this is the Bilbo service)

``say`` runs the exact path a spoken command takes (grammar → router → bridge or
direct), so it is the quickest way to test the lights without a microphone. Add
``--direct`` to skip the bridge and talk to the bulbs from this machine.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from ..config import load_config
from . import build_light_router
from .backends import TapoBackend
from .bridge import LightsBridge
from .intents import match_light_intent
from .model import Light, LightError, LightInventory
from .router import summarize


def _inventory(cfg) -> LightInventory | None:
    try:
        return LightInventory.from_file(cfg.lights.file)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _backend(cfg, inv: LightInventory) -> TapoBackend:
    if not cfg.lights.tapo_username:
        print("note: TAPO_USERNAME / TAPO_PASSWORD are not set — trying without credentials "
              "(only works for bulbs never linked to a TP-Link account).", file=sys.stderr)
    return TapoBackend(inv, cfg.lights.tapo_username, cfg.lights.tapo_password,
                       host_cache=cfg.lights.host_cache or None)


def _colon(mac: str) -> str:
    digits = "".join(c for c in mac if c.isalnum()).upper()
    return ":".join(digits[i:i + 2] for i in range(0, len(digits), 2))


def _cmd_discover(cfg) -> int:
    inv = _inventory(cfg) if _has_file(cfg) else None
    # discover() never touches the inventory; a placeholder satisfies the type.
    backend = _backend(cfg, inv or LightInventory([Light("placeholder", ("placeholder",))]))
    try:
        found = backend.discover()
    except LightError as exc:
        print(f"discovery failed: {exc}", file=sys.stderr)
        return 2
    finally:
        backend.close()
    if not found:
        print("No TP-Link/Tapo devices answered. Check they're on the same network as this "
              "machine and that the router isn't isolating Wi-Fi clients.")
        return 1
    known = {l.mac: l for l in inv} if inv else {}
    print(f"Found {len(found)} device(s):\n")
    for d in found:
        norm = "".join(c for c in d["mac"] if c.isalnum()).upper()
        tag = f"  -> {known[norm].id}" if norm in known else ""
        if d.get("unsupported"):
            print(f"  {d['host'] or '?':<15} {_colon(d['mac']):<17} {d['model']:<8} "
                  f"UNSUPPORTED (protocol {d['protocol'] or 'unknown'}){tag}")
            continue
        feats = ",".join(k for k, v in d["features"].items() if v) or "-"
        print(f"  {d['host']:<15} {_colon(d['mac']):<17} {d['model']:<8} "
              f"alias={d['alias']!r:<20} light={d['is_light']} features={feats}{tag}")
    if any(d.get("unsupported") for d in found):
        print("\nUNSUPPORTED means the bulb answered but uses a newer TP-Link protocol that "
              "python-kasa can't talk to yet. In the Tapo app, turn on:\n"
              "    Me > Third-Party Services > Third-Party Compatibility\n"
              "then run discover again. (TP-Link disables it by default; see its FAQ for "
              "the security trade-off.)")
    unknown = [d for d in found if (d["is_light"] or d.get("unsupported"))
               and "".join(c for c in d["mac"] if c.isalnum()).upper() not in known]
    if unknown:
        print("\nAdd the ones you want to lights.toml (pick an id and what you'll say):\n")
        for d in unknown:
            print(f'[[light]]\nid = "CHOOSE_AN_ID"\nnames = ["WHAT YOU WILL SAY"]\n'
                  f'mac = "{_colon(d["mac"])}"\n')
        print("Then run `python -m lexi.lights identify <id>` to see which bulb is which.")
    return 0


def _has_file(cfg) -> bool:
    from pathlib import Path
    return Path(cfg.lights.file).is_file()


def _require_inventory(cfg) -> LightInventory | None:
    inv = _inventory(cfg)
    if inv is None:
        print(f"No lights file at {cfg.lights.file}. Copy lights.example.toml to lights.toml "
              "and fill it in (run `discover` first).", file=sys.stderr)
    return inv


def _cmd_identify(cfg, name: str) -> int:
    inv = _require_inventory(cfg)
    if inv is None:
        return 2
    targets = inv.resolve_name(name)
    if targets is None:
        print(f"No light called {name!r}. Known: {', '.join(inv.ids)}", file=sys.stderr)
        return 2
    backend = _backend(cfg, inv)
    rc = 0
    try:
        for lid in targets.ids:
            print(f"Blinking {lid} ...")
            err = backend.identify(lid)
            if err:
                print(f"  {lid}: {err}", file=sys.stderr)
                rc = 1
    finally:
        backend.close()
    return rc


def _cmd_say(cfg, phrase: str, direct: bool) -> int:
    router = build_light_router(cfg, use_mqtt=not direct)
    if router is None:
        print("Lights aren't configured (need lights.toml).", file=sys.stderr)
        return 2
    try:
        mqtt = getattr(router, "_mqtt", None)
        if mqtt is not None:  # the retained status arrives just after connecting
            for _ in range(30):
                if mqtt.bridge_online:
                    break
                time.sleep(0.1)
        intent = match_light_intent(phrase, router.inventory, step=cfg.lights.brightness_step)
        if intent is None or intent.targets is None:
            print("Not a light command.")
            return 1
        result = router.execute(intent.targets.ids, intent.command)
        print(summarize(router.inventory, intent.command, result))
        print("via:", result.via or "-", "errors:", result.errors or "-")
        return 0 if result.success else 1
    finally:
        router.close()


def _cmd_state(cfg, name: str | None) -> int:
    inv = _require_inventory(cfg)
    if inv is None:
        return 2
    ids = list(inv.ids)
    if name:
        t = inv.resolve_name(name)
        if t is None:
            print(f"No light called {name!r}.", file=sys.stderr)
            return 2
        ids = list(t.ids)
    backend = _backend(cfg, inv)
    try:
        for lid, st in backend.state(ids).items():
            print(f"{lid}: {st}")
    finally:
        backend.close()
    return 0


def _cmd_bridge(cfg) -> int:
    inv = _require_inventory(cfg)
    if inv is None:
        return 2
    lc = cfg.lights
    if not lc.mqtt_host:
        print("MQTT_HOST is not set — the bridge needs a broker to listen on.", file=sys.stderr)
        return 2
    backend = _backend(cfg, inv)
    bridge = LightsBridge(backend, inv, lc.mqtt_host, lc.mqtt_port, lc.mqtt_username,
                          lc.mqtt_password)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    bridge.start()
    print(f"Lights bridge running (broker {lc.mqtt_host}:{lc.mqtt_port}, "
          f"{len(inv)} light(s)). Ctrl-C to stop.")
    try:
        stop.wait()
    finally:
        bridge.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lexi.lights", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover", help="find Tapo bulbs on the LAN")
    p = sub.add_parser("identify", help="blink one bulb")
    p.add_argument("light")
    p = sub.add_parser("say", help="run a spoken command as text")
    p.add_argument("phrase", nargs="+")
    p.add_argument("--direct", action="store_true", help="skip the bridge; talk to the bulbs directly")
    p = sub.add_parser("state", help="show what the bulbs report")
    p.add_argument("light", nargs="?")
    sub.add_parser("bridge", help="run the MQTT bridge service")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if args.cmd == "discover":
        return _cmd_discover(cfg)
    if args.cmd == "identify":
        return _cmd_identify(cfg, args.light)
    if args.cmd == "say":
        return _cmd_say(cfg, " ".join(args.phrase), args.direct)
    if args.cmd == "state":
        return _cmd_state(cfg, args.light)
    return _cmd_bridge(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
