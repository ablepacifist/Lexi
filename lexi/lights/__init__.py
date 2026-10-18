"""Voice control of the Tapo bulbs — on/off, brightness, color.

Layout (pure logic first, hardware last, so nearly all of it is unit-tested):

  model.py     inventory, spoken-name resolution, ``LightCommand``
  intents.py   the spoken-command grammar (``match_light_intent``)
  router.py    bridge-first / direct-fallback execution + the spoken summary
  backends.py  ``TapoBackend`` (python-kasa) and ``MqttBackend`` (paho)
  bridge.py    the Bilbo service: MQTT in, Tapo out
  __main__.py  ``python -m lexi.lights <discover|identify|say|state|bridge>``
"""
from __future__ import annotations

import logging

from .backends import MqttBackend, TapoBackend
from .intents import LightIntent, match_light_intent
from .model import Light, LightCommand, LightError, LightInventory, Targets
from .router import LightResult, LightRouter, summarize

__all__ = [
    "Light", "LightCommand", "LightError", "LightInventory", "LightIntent",
    "LightResult", "LightRouter", "MqttBackend", "TapoBackend", "Targets",
    "build_light_router", "match_light_intent", "summarize",
]

logger = logging.getLogger(__name__)


def build_light_router(cfg, *, use_mqtt: bool = True, warm_up: bool = False) -> LightRouter | None:
    """Construct the router from a ``LexiConfig``. Returns ``None`` — never raises —
    when lights are disabled, unconfigured or their file is malformed, so a lights
    problem can never stop Lexi starting. ``warm_up`` connects to the bulbs in the
    background (the long-running voice service wants that; one-shot CLI runs don't)."""
    lc = cfg.lights
    if not lc.enabled:
        return None
    try:
        inventory = LightInventory.from_file(lc.file)
    except ValueError as exc:
        logger.error("light control disabled: %s", exc)
        return None
    if inventory is None:
        logger.info("no lights file at %s; light control is off", lc.file)
        return None

    direct = (TapoBackend(inventory, lc.tapo_username, lc.tapo_password,
                          host_cache=lc.host_cache or None)
              if lc.direct_fallback else None)
    mqtt = None
    if use_mqtt and lc.mqtt_host:
        mqtt = MqttBackend(lc.mqtt_host, lc.mqtt_port, lc.mqtt_username, lc.mqtt_password,
                           ack_timeout=lc.ack_timeout_s)
        try:
            mqtt.start()
        except LightError as exc:
            logger.warning("MQTT unavailable (%s); controlling lights directly", exc)
            mqtt = None
    if mqtt is None and direct is None:
        logger.error("light control disabled: no MQTT host and direct control is off")
        return None
    router = LightRouter(inventory, mqtt=mqtt, direct=direct, recent_s=lc.recent_s)
    if warm_up:
        router.warm_up()
    return router
