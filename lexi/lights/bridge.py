"""The Bilbo bridge: MQTT commands in, Tapo calls out.

Runs as a small always-on service next to Mosquitto (``python -m lexi.lights
bridge``). It listens on ``home/lights/<id>/set``, drives the bulbs with the same
``TapoBackend`` the Pi uses as its fallback, and answers with:

  * ``home/lights/<id>/ack``   — {cid, ok, error}  (so the Pi knows it ran)
  * ``home/lights/<id>/state`` — retained current state (bulbs are the source of
                                 truth, re-read after each command and periodically,
                                 so a Bilbo that was off while the Pi drove the bulbs
                                 directly is never stale when it comes back)
  * ``home/lights/bridge/status`` — retained "online"/"offline", with an MQTT Last
                                 Will so an unclean death still flips it to "offline"
                                 — that is how the Pi knows to skip the bridge.

Anything with write access to the ``set`` topics can control the lights; the broker's
ACL is the security boundary. Payloads are untrusted, so they are validated and
clamped (``LightCommand.from_payload``) before they reach a bulb.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import socket
import threading
import time
import uuid
from typing import Any, Callable

from .backends import (
    STATUS_TOPIC,
    TOPIC_ROOT,
    TapoBackend,
    ack_topic,
    state_topic,
)
from .model import LightCommand, LightInventory

logger = logging.getLogger(__name__)


class LightsBridge:
    def __init__(
        self,
        backend: TapoBackend,
        inventory: LightInventory,
        mqtt_host: str,
        mqtt_port: int = 1883,
        mqtt_username: str = "",
        mqtt_password: str = "",
        *,
        poll_interval: float = 60.0,
        workers: int = 4,
        client_factory: Callable[[], Any] | None = None,
    ):
        self._backend = backend
        self._inv = inventory
        self._host = mqtt_host
        self._port = mqtt_port
        self._user = mqtt_username
        self._pw = mqtt_password
        self._poll = poll_interval
        self._client_factory = client_factory
        self._client: Any = None
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="lexi-bridge")
        self._stop = threading.Event()
        self._poller: threading.Thread | None = None

    # -- the pure part (unit-tested without a broker) -----------------------
    def handle_set(self, light_id: str, raw: bytes | str) -> dict | None:
        """Execute one ``set`` message and return its ack (``None`` if the payload
        was so malformed there is no correlation id to answer)."""
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("ignoring non-JSON command for %s", light_id)
            return None
        if not isinstance(data, dict):
            return None
        cid = str(data.get("cid", ""))
        if self._inv.get(light_id) is None:
            return {"cid": cid, "ok": False, "error": "unknown light"}
        try:
            cmd = LightCommand.from_payload({k: v for k, v in data.items() if k != "cid"})
        except ValueError as exc:
            return {"cid": cid, "ok": False, "error": f"bad command: {exc}"}
        error = self._backend.execute([light_id], cmd).get(light_id)
        return {"cid": cid, "ok": error is None, "error": error}

    def state_message(self, light_id: str) -> dict:
        st = dict(self._backend.state([light_id]).get(light_id, {"reachable": False}))
        st["ts"] = int(time.time())
        return st

    # -- MQTT plumbing ------------------------------------------------------
    def _make_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        import paho.mqtt.client as mqtt  # noqa: PLC0415
        cid = f"lexi-lights-bridge-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)

    def start(self) -> None:
        client = self._make_client()
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        if self._user:
            client.username_pw_set(self._user, self._pw)
        # Last Will: if we die without saying goodbye the broker publishes this.
        client.will_set(STATUS_TOPIC, "offline", qos=1, retain=True)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client = client
        client.connect_async(self._host, self._port, keepalive=30)
        client.loop_start()
        self._poller = threading.Thread(target=self._poll_loop, name="lexi-bridge-poll",
                                        daemon=True)
        self._poller.start()

    def stop(self) -> None:
        self._stop.set()
        client, self._client = self._client, None
        if client is not None:
            try:
                client.publish(STATUS_TOPIC, "offline", qos=1, retain=True).wait_for_publish(2)
            except Exception:  # noqa: BLE001 - shutting down
                pass
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._backend.close()

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None) -> None:
        if getattr(reason_code, "is_failure", False):
            logger.warning("MQTT connect refused: %s", reason_code)
            return
        logger.info("bridge connected to %s:%s", self._host, self._port)
        client.subscribe(f"{TOPIC_ROOT}/+/set", qos=1)
        client.publish(STATUS_TOPIC, "online", qos=1, retain=True)
        self._pool.submit(self.publish_states)  # warm the bulbs + publish state now

    def _on_message(self, client, userdata, message) -> None:
        parts = message.topic.split("/")
        if len(parts) == 4 and parts[3] == "set":
            try:
                self._pool.submit(self._process, parts[2], message.payload)
            except RuntimeError:  # pool already shut down
                pass

    def _process(self, light_id: str, raw: bytes) -> None:
        try:
            ack = self.handle_set(light_id, raw)
            client = self._client
            if client is None:
                return
            if ack is not None:
                client.publish(ack_topic(light_id), json.dumps(ack), qos=1)
                logger.info("light %s: %s", light_id, "ok" if ack["ok"] else ack["error"])
            if self._inv.get(light_id) is not None:
                client.publish(state_topic(light_id), json.dumps(self.state_message(light_id)),
                               qos=1, retain=True)
        except Exception:  # noqa: BLE001 - one bad message must not kill the worker
            logger.exception("failed handling command for %s", light_id)

    def publish_states(self) -> None:
        client = self._client
        if client is None:
            return
        for light in self._inv:
            try:
                client.publish(state_topic(light.id), json.dumps(self.state_message(light.id)),
                               qos=1, retain=True)
            except Exception:  # noqa: BLE001
                logger.debug("state publish for %s failed", light.id, exc_info=True)

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll):
            self.publish_states()
