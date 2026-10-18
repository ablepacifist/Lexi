"""The two ways to reach a bulb.

``TapoBackend`` talks to Tapo bulbs directly over the LAN with python-kasa. It runs
in two places: on the Pi (the fallback) and inside the Bilbo bridge — one wrapper,
so both behave identically. ``MqttBackend`` is the Pi's client for the bridge.

python-kasa and paho-mqtt are imported lazily, so importing Lexi (and running its
test suite) needs neither.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .model import LightCommand, LightError, LightInventory, Light, normalize_mac

logger = logging.getLogger(__name__)

TOPIC_ROOT = "home/lights"


def set_topic(light_id: str) -> str:
    return f"{TOPIC_ROOT}/{light_id}/set"


def ack_topic(light_id: str) -> str:
    return f"{TOPIC_ROOT}/{light_id}/ack"


def state_topic(light_id: str) -> str:
    return f"{TOPIC_ROOT}/{light_id}/state"


STATUS_TOPIC = f"{TOPIC_ROOT}/bridge/status"


# ── direct: python-kasa ──────────────────────────────────────────────────────

class TapoBackend:
    """Control Tapo bulbs directly. Bulbs are found by MAC via UDP-broadcast
    discovery (their IPs come from DHCP and can change), cached, and re-discovered
    automatically when a cached connection stops working."""

    def __init__(
        self,
        inventory: LightInventory,
        username: str = "",
        password: str = "",
        *,
        discovery_timeout: int = 5,
        op_timeout: float = 25.0,
        kasa_module: Any = None,
        host_cache: str | os.PathLike | None = None,
    ):
        self._inv = inventory
        self._user = username
        self._pw = password
        self._disc_timeout = discovery_timeout
        self._op_timeout = op_timeout
        self._kasa = kasa_module
        self._devices: dict[str, Any] = {}
        self._hosts: dict[str, str] = {}
        # Last-known address per MAC, kept on disk so a fresh process (a restart, the
        # CLI, the bridge) can connect by unicast instead of relying on a Wi-Fi
        # broadcast whose replies are sometimes simply lost.
        self._host_cache_path = Path(host_cache) if host_cache else None
        self._known_hosts: dict[str, str] = self._load_host_cache()
        # (light id, host) -> monotonic time it last failed; skipped briefly, not forever
        # (a power-cycled bulb comes back at the SAME address).
        self._bad_hosts: dict[tuple[str, str], float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._disc_lock = asyncio.Lock()      # binds to the backend's loop on first use
        self._last_failures: dict[str, str] = {}
        self._discovered_at = float("-inf")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()

    # -- sync API (what the router, bridge and CLI call) --------------------
    def execute(self, light_ids: list[str], cmd: LightCommand) -> dict[str, str | None]:
        ids = list(light_ids)
        try:
            return self._call(self._execute_async(ids, cmd))
        except Exception as exc:  # noqa: BLE001 - never raise per-light failures
            reason = self._describe(exc)
            return {i: reason for i in ids}

    def state(self, light_ids: list[str]) -> dict[str, dict]:
        ids = list(light_ids)
        try:
            return self._call(self._state_async(ids))
        except Exception:  # noqa: BLE001
            return {i: {"reachable": False} for i in ids}

    def discover(self) -> list[dict]:
        return self._call(self._discover_async())

    def identify(self, light_id: str) -> str | None:
        """Blink one bulb so you can see which physical light an id is."""
        try:
            self._call(self._identify_async(light_id))
            return None
        except Exception as exc:  # noqa: BLE001
            return self._describe(exc)

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            self._call(self._close_async())
        except Exception:  # noqa: BLE001
            pass
        loop.call_soon_threadsafe(loop.stop)
        self._loop = None

    # -- event loop plumbing ------------------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=loop.run_forever, name="lexi-tapo", daemon=True)
                self._thread.start()
                self._loop = loop
            return self._loop

    def _call(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        try:
            return fut.result(timeout=self._op_timeout)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise LightError("timed out") from exc

    def _kasa_mod(self):
        if self._kasa is None:
            try:
                import kasa  # noqa: PLC0415
            except ImportError as exc:
                raise LightError("python-kasa isn't installed") from exc
            self._kasa = kasa
        return self._kasa

    def _credentials(self):
        return self._kasa_mod().Credentials(self._user, self._pw) if self._user else None

    def _lock(self, light_id: str) -> asyncio.Lock:
        if light_id not in self._locks:
            self._locks[light_id] = asyncio.Lock()
        return self._locks[light_id]

    UNSUPPORTED = "isn't supported yet"

    @staticmethod
    def _unsupported_info(exc: BaseException) -> dict:
        """Pull host/MAC/model/protocol out of python-kasa's UnsupportedDeviceError.
        Its ``discovery_result`` may be a dict or an object, so read it defensively."""
        dr = getattr(exc, "discovery_result", None)

        def pick(*keys):
            for k in keys:
                v = dr.get(k) if isinstance(dr, dict) else getattr(dr, k, None)
                if v:
                    return v
            return None

        scheme = pick("mgt_encrypt_schm")
        proto = (scheme.get("encrypt_type") if isinstance(scheme, dict)
                 else getattr(scheme, "encrypt_type", None)) or pick("encrypt_type") or ""
        return {
            "host": str(getattr(exc, "host", "") or ""),
            "mac": str(pick("mac") or ""),
            "model": str(pick("device_model", "device_type") or ""),
            "protocol": str(proto),
        }

    @staticmethod
    def _describe(exc: BaseException) -> str:
        if isinstance(exc, LightError):
            return str(exc)
        name = type(exc).__name__
        if name == "AuthenticationError":
            return "authentication failed"
        if name == "UnsupportedDeviceError":
            return TapoBackend.UNSUPPORTED
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "Timeout" in name:
            return "timed out"
        return "unreachable"

    # -- connection management ----------------------------------------------
    @staticmethod
    async def _safe_disconnect(dev: Any) -> None:
        try:
            await dev.disconnect()
        except Exception:  # noqa: BLE001
            pass

    async def _forget(self, light_id: str) -> None:
        dev = self._devices.pop(light_id, None)
        self._hosts.pop(light_id, None)
        if dev is not None:
            await self._safe_disconnect(dev)

    async def _connect(self, host: str) -> Any:
        """Connect to ONE known address. Deliberately ``discover_single`` and not
        ``Device.connect(host=...)``: the latter assumes the legacy protocol on port
        9999, which Tapo bulbs don't speak, whereas ``discover_single`` asks that one
        device (unicast — reliable, unlike a Wi-Fi broadcast) which protocol it uses and
        returns a correctly configured connection. Raises on no answer/unsupported."""
        kasa = self._kasa_mod()
        dev = await kasa.Discover.discover_single(
            host, credentials=self._credentials(), discovery_timeout=self._disc_timeout)
        if dev is None:
            raise OSError(f"no device answered at {host}")
        try:
            await dev.update()
        except Exception:
            await self._safe_disconnect(dev)
            raise
        return dev

    BAD_HOST_TTL = 30.0   # seconds an address that just failed is skipped

    # -- last-known-address cache (MAC -> host), persisted -------------------
    def _load_host_cache(self) -> dict[str, str]:
        if self._host_cache_path is None:
            return {}
        try:
            data = json.loads(self._host_cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {normalize_mac(k): v for k, v in data.items()
                if isinstance(k, str) and isinstance(v, str) and v}

    def _remember_host(self, mac: str, host: str) -> None:
        if not mac or not host or self._known_hosts.get(mac) == host:
            return
        self._known_hosts[mac] = host
        if self._host_cache_path is None:
            return
        try:  # atomic: a crash mid-write must never leave a half-written cache
            self._host_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._host_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._known_hosts, indent=1), encoding="utf-8")
            os.replace(tmp, self._host_cache_path)
        except OSError as exc:
            logger.debug("could not save the light address cache: %s", exc)

    def _candidate_hosts(self, light: Light) -> list[str]:
        """Addresses worth a direct connect, best first: this session's, the on-disk
        cache's, then the configured hint. Skips any that failed a moment ago."""
        now = time.monotonic()
        out: list[str] = []
        for host in (self._hosts.get(light.id),
                     self._known_hosts.get(light.mac) if light.mac else None,
                     light.host):
            if not host or host in out:
                continue
            if now - self._bad_hosts.get((light.id, host), float("-inf")) < self.BAD_HOST_TTL:
                continue
            out.append(host)
        return out

    async def _try_host(self, light: Light, host: str) -> Any | None:
        """Connect straight to a known address. ``None`` = not there any more (the
        caller falls back to discovery); wrong credentials / unsupported are raised."""
        try:
            dev = await self._connect(host)
            if light.mac and normalize_mac(getattr(dev, "mac", "")) not in ("", light.mac):
                # The address now belongs to a different device (DHCP moved things).
                await self._safe_disconnect(dev)
                raise LightError("address belongs to another device")
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if name == "AuthenticationError":
                raise  # wrong credentials: don't hammer the bulb by trying again
            if name == "UnsupportedDeviceError":
                raise LightError(self.UNSUPPORTED) from exc  # rediscovery can't fix this
            logger.info("%s at %s failed (%s); falling back to discovery", light.id, host, exc)
            self._bad_hosts[(light.id, host)] = time.monotonic()
            self._hosts.pop(light.id, None)
            return None
        self._devices[light.id] = dev
        self._hosts[light.id] = host
        self._remember_host(light.mac, host)
        return dev

    # -- ARP: MAC -> IP without relying on Wi-Fi broadcast --------------------
    @staticmethod
    def _parse_arp(text: str) -> dict[str, str]:
        """Parse Linux ``/proc/net/arp`` into {normalized MAC: ip}."""
        out: dict[str, str] = {}
        for line in text.splitlines()[1:]:
            f = line.split()
            if len(f) >= 4 and f[3] != "00:00:00:00:00:00":
                out[normalize_mac(f[3])] = f[0]
        return out

    @classmethod
    def _arp_table(cls) -> dict[str, str]:
        try:
            return cls._parse_arp(Path("/proc/net/arp").read_text(encoding="utf-8"))
        except OSError:
            return {}   # not Linux (or unreadable): the caller just skips this step

    @staticmethod
    def _prime_arp() -> None:
        """Nudge the OS into ARP-resolving every address on the local /24, so the ARP
        table holds any bulb that is awake. ARP is answered at the link layer, which
        is why it works when UDP broadcast discovery drops replies. Blocking (~1.5 s)."""
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("10.255.255.255", 9))   # selects our LAN address; sends nothing
                base = probe.getsockname()[0].rsplit(".", 1)[0]
            finally:
                probe.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            try:
                for i in range(1, 255):
                    try:
                        sock.sendto(b"\0", (f"{base}.{i}", 9))
                    except OSError:
                        pass
            finally:
                time.sleep(1.5)  # let the ARP replies arrive
                sock.close()
        except OSError:
            pass

    async def _resolve_by_arp(self, light: Light) -> str | None:
        if not light.mac:
            return None
        host = self._arp_table().get(light.mac)
        if host is None:
            await asyncio.get_running_loop().run_in_executor(None, self._prime_arp)
            host = self._arp_table().get(light.mac)
        return host

    async def _device(self, light: Light) -> Any:
        dev = self._devices.get(light.id)
        if dev is not None:
            return dev
        # 1) Known addresses first. A unicast connect is reliable; a Wi-Fi broadcast
        #    is not (replies get lost), so discovery is the backup, not the plan.
        tried = set(self._candidate_hosts(light))
        for host in list(tried):
            dev = await self._try_host(light, host)
            if dev is not None:
                return dev
        # 2) ARP: turn the MAC into an address at the link layer (Linux). Far more
        #    reliable than broadcast, and quick.
        arp_host = await self._resolve_by_arp(light)
        if arp_host and arp_host not in tried:
            tried.add(arp_host)
            dev = await self._try_host(light, arp_host)
            if dev is not None:
                return dev
        if light.mac:
            # 2) Broadcast. One broadcast finds every bulb, so lights must not each start
            #    their own: with the real library every discovery builds NEW Device
            #    objects, and a second, overlapping one would tear down sessions the
            #    first is still using (the "turn off all the lights" case). Serialize,
            #    then re-check the cache — another light's discovery may have found this
            #    one already.
            async with self._disc_lock:
                dev = self._devices.get(light.id)
                if dev is not None:
                    return dev
                recent = time.monotonic() - self._discovered_at < 3.0
                if recent and light.id in self._last_failures:
                    raise LightError(self._last_failures[light.id])  # just looked; same answer
                for attempt in (1, 2):
                    failures = await self._discover_into_cache()
                    self._last_failures = failures
                    self._discovered_at = time.monotonic()
                    dev = self._devices.get(light.id)
                    if dev is not None:
                        return dev
                    if light.id in failures:  # found on the network, but couldn't be used
                        raise LightError(failures[light.id])
                    if attempt == 1:  # a lost reply is likelier than a missing bulb
                        logger.info("%s not in the discovery replies; asking once more", light.id)
        raise LightError("not found on the network")

    async def _discover_into_cache(self) -> dict[str, str]:
        """One broadcast finds every bulb; cache each one we know by MAC. Returns
        {light id: reason} for known bulbs that answered but could not be used (most
        often wrong credentials), so that is reported as what it is rather than as
        "not found"."""
        kasa = self._kasa_mod()
        failures: dict[str, str] = {}
        unsupported: list[dict] = []

        async def on_unsupported(exc: BaseException) -> None:
            unsupported.append(self._unsupported_info(exc))

        found = await kasa.Discover.discover(
            credentials=self._credentials(), discovery_timeout=self._disc_timeout,
            on_unsupported=on_unsupported)
        # A bulb that answered but speaks a protocol python-kasa lacks (newer TP-Link
        # firmware, "TPAP") is not "missing" — say what it is.
        for info in unsupported:
            mac = normalize_mac(info["mac"])
            for light in self._inv:
                if light.mac and light.mac == mac:
                    failures[light.id] = self.UNSUPPORTED
                    logger.warning("%s is on the network (%s) but uses the %s protocol, "
                                   "which python-kasa does not support", light.id,
                                   info["host"] or "?", info["protocol"] or "unknown")
        for host, dev in found.items():
            lights: list[Light] = []
            try:
                mac = normalize_mac(getattr(dev, "mac", ""))
                if not mac:
                    await dev.update()
                    mac = normalize_mac(getattr(dev, "mac", ""))
                lights = [l for l in self._inv if l.mac and l.mac == mac]
                if not lights:
                    await self._safe_disconnect(dev)
                    continue
                await dev.update()
            except Exception as exc:  # noqa: BLE001
                logger.debug("discovered %s but could not use it: %s", host, exc)
                await self._safe_disconnect(dev)
                for light in lights:
                    failures[light.id] = self._describe(exc)
                continue
            for light in lights:
                old = self._devices.get(light.id)
                if old is not None and old is not dev:
                    await self._safe_disconnect(old)
                self._devices[light.id] = dev
                self._hosts[light.id] = host
                self._bad_hosts.pop((light.id, host), None)
                self._remember_host(light.mac, host)
        return failures

    # -- operations ---------------------------------------------------------
    async def _execute_async(self, ids: list[str], cmd: LightCommand) -> dict[str, str | None]:
        results = await asyncio.gather(*(self._one(i, cmd) for i in ids))
        return dict(zip(ids, results))

    async def _one(self, light_id: str, cmd: LightCommand) -> str | None:
        light = self._inv.get(light_id)
        if light is None:
            return "unknown light"
        async with self._lock(light_id):
            last: BaseException | None = None
            for _attempt in (1, 2):  # the second try re-discovers (stale IP / dropped session)
                try:
                    dev = await self._device(light)
                    await self._apply(dev, cmd)
                    return None
                except LightError as exc:
                    return str(exc)  # capability / not-found: retrying can't change it
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    await self._forget(light.id)
                    if type(exc).__name__ == "AuthenticationError":
                        break  # retrying wrong credentials only risks a lockout
                    logger.info("light %s failed (%s: %s); retrying once",
                                light_id, type(exc).__name__, exc)
            return self._describe(last) if last is not None else "unreachable"

    @staticmethod
    async def _apply(dev: Any, cmd: LightCommand) -> None:
        # python-kasa's Module.Light is the plain string "Light".
        light = dev.modules.get("Light")
        if light is None:
            raise LightError("isn't a light")
        await dev.update()
        was_on = bool(dev.is_on)
        if cmd.toggle:
            await (dev.turn_off() if was_on else dev.turn_on())
            return
        if cmd.power is False:
            await dev.turn_off()
            return
        has_attrs = any(v is not None for v in (
            cmd.brightness, cmd.brightness_delta, cmd.hue, cmd.saturation, cmd.color_temp))
        if not has_attrs:
            await dev.turn_on()
            return
        if cmd.hue is not None:
            if not light.has_feature("hsv"):
                raise LightError("can't change color")
            sat = cmd.saturation if cmd.saturation is not None else 100
            await light.set_hsv(cmd.hue, sat, cmd.brightness)
        elif cmd.color_temp is not None:
            if not light.has_feature("color_temp"):
                raise LightError("can't change color temperature")
            await light.set_color_temp(cmd.color_temp, brightness=cmd.brightness)
        elif cmd.brightness is not None:
            if not light.has_feature("brightness"):
                raise LightError("can't be dimmed")
            await light.set_brightness(cmd.brightness)
        elif cmd.brightness_delta is not None:
            if not light.has_feature("brightness"):
                raise LightError("can't be dimmed")
            current = int(light.brightness)
            await light.set_brightness(max(1, min(100, current + cmd.brightness_delta)))
        if not was_on:  # setting a color/level on a bulb that is off also turns it on
            await dev.turn_on()

    async def _state_async(self, ids: list[str]) -> dict[str, dict]:
        results = await asyncio.gather(*(self._state_one(i) for i in ids))
        return dict(zip(ids, results))

    async def _state_one(self, light_id: str) -> dict:
        light = self._inv.get(light_id)
        if light is None:
            return {"reachable": False}
        async with self._lock(light_id):
            try:
                dev = await self._device(light)
                await dev.update()
                out: dict = {"reachable": True, "on": bool(dev.is_on)}
                mod = dev.modules.get("Light")
                if mod is not None:
                    if mod.has_feature("brightness"):
                        out["brightness"] = int(mod.brightness)
                    if mod.has_feature("hsv"):
                        hsv = mod.hsv
                        out["hs"] = [int(hsv.hue), int(hsv.saturation)]
                    if mod.has_feature("color_temp"):
                        out["color_temp"] = int(mod.color_temp)
                return out
            except Exception as exc:  # noqa: BLE001
                logger.debug("state of %s failed: %s", light_id, exc)
                await self._forget(light_id)
                return {"reachable": False}

    async def _discover_async(self) -> list[dict]:
        kasa = self._kasa_mod()
        unsupported: list[dict] = []

        async def on_unsupported(exc: BaseException) -> None:
            unsupported.append(self._unsupported_info(exc))

        found = await kasa.Discover.discover(
            credentials=self._credentials(), discovery_timeout=self._disc_timeout,
            on_unsupported=on_unsupported)
        out: list[dict] = []
        for info in unsupported:  # devices that answered but can't be controlled yet
            out.append({
                "host": info["host"], "alias": "", "model": info["model"],
                "mac": info["mac"], "is_light": None, "on": False, "features": {},
                "unsupported": True, "protocol": info["protocol"],
            })
        for host, dev in found.items():
            try:
                await dev.update()
            except Exception:  # noqa: BLE001
                pass
            mod = None
            try:
                mod = dev.modules.get("Light")
            except Exception:  # noqa: BLE001
                pass
            features = {}
            if mod is not None:
                for name in ("brightness", "hsv", "color_temp"):
                    try:
                        features[name] = bool(mod.has_feature(name))
                    except Exception:  # noqa: BLE001
                        features[name] = False
            out.append({
                "host": host,
                "alias": getattr(dev, "alias", "") or "",
                "model": getattr(dev, "model", "") or "",
                "mac": getattr(dev, "mac", "") or "",
                "is_light": mod is not None,
                "on": bool(getattr(dev, "is_on", False)),
                "features": features,
            })
            await self._safe_disconnect(dev)
        return out

    async def _identify_async(self, light_id: str) -> None:
        light = self._inv.get(light_id)
        if light is None:
            raise LightError("unknown light")
        async with self._lock(light_id):
            dev = await self._device(light)
            await dev.update()
            was_on = bool(dev.is_on)
            for _ in range(3):
                await dev.turn_off()
                await asyncio.sleep(0.6)
                await dev.turn_on()
                await asyncio.sleep(0.6)
            if not was_on:
                await dev.turn_off()

    async def _close_async(self) -> None:
        for dev in list(self._devices.values()):
            await self._safe_disconnect(dev)
        self._devices.clear()
        self._hosts.clear()


# ── MQTT client (the Pi's side of the Bilbo bridge) ──────────────────────────

class _Waiter:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.ack: dict | None = None


class MqttBackend:
    """Publish commands to the bridge and wait for its acknowledgement.

    A persistent background connection (paho ``loop_start`` + auto-reconnect)
    tracks the bridge's retained ``status`` topic, so ``bridge_online`` is an
    instant, in-memory answer — a powered-off Bilbo costs no per-command wait.
    """

    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: str = "",
        password: str = "",
        *,
        ack_timeout: float = 4.0,
        client_factory: Callable[[], Any] | None = None,
    ):
        self._host = host
        self._port = port
        self._user = username
        self._pw = password
        self._ack_timeout = ack_timeout
        self._client_factory = client_factory
        self._client: Any = None
        self._connected = False
        self._bridge_status = ""
        self._pending: dict[tuple[str, str], _Waiter] = {}
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._client is not None:
            return
        client = self._make_client()
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        if self._user:
            client.username_pw_set(self._user, self._pw)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client = client
        client.connect_async(self._host, self._port, keepalive=30)
        client.loop_start()

    def _make_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        try:
            import paho.mqtt.client as mqtt  # noqa: PLC0415
        except ImportError as exc:
            raise LightError("paho-mqtt isn't installed") from exc
        cid = f"lexi-lights-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)

    def close(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._connected = False

    @property
    def bridge_online(self) -> bool:
        return self._connected and self._bridge_status == "online"

    # -- paho callbacks (API v2 signatures) ---------------------------------
    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None) -> None:
        failed = getattr(reason_code, "is_failure", False)
        if failed:
            logger.warning("MQTT connect to %s:%s refused: %s", self._host, self._port, reason_code)
            return
        self._connected = True
        client.subscribe([(STATUS_TOPIC, 1), (f"{TOPIC_ROOT}/+/ack", 1)])
        logger.info("MQTT connected to %s:%s", self._host, self._port)

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None,
                       properties=None) -> None:
        self._connected = False
        self._bridge_status = ""  # unknown while we can't hear the broker

    def _on_message(self, client, userdata, message) -> None:
        topic = message.topic
        try:
            payload = message.payload.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return
        if topic == STATUS_TOPIC:
            self._bridge_status = payload.strip().lower()
            return
        parts = topic.split("/")
        if len(parts) == 4 and parts[3] == "ack":
            try:
                data = json.loads(payload)
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            with self._lock:
                waiter = self._pending.get((str(data.get("cid", "")), parts[2]))
            if waiter is not None:
                waiter.ack = data
                waiter.event.set()

    # -- commands -----------------------------------------------------------
    def execute(self, light_ids: list[str], cmd: LightCommand) -> dict[str, str | None]:
        ids = list(light_ids)
        if self._client is None or not self.bridge_online:
            return {i: "the light bridge is offline" for i in ids}
        cid = uuid.uuid4().hex
        body = json.dumps({"cid": cid, **cmd.to_payload()})
        waiters = {i: _Waiter() for i in ids}
        with self._lock:
            for i, w in waiters.items():
                self._pending[(cid, i)] = w
        try:
            for i in ids:
                self._client.publish(set_topic(i), body, qos=1)
            deadline = time.monotonic() + self._ack_timeout
            out: dict[str, str | None] = {}
            for i in ids:
                w = waiters[i]
                if w.event.wait(max(0.0, deadline - time.monotonic())):
                    ack = w.ack or {}
                    out[i] = None if ack.get("ok") else str(ack.get("error") or "the bridge failed")
                else:
                    out[i] = "no reply from the bridge"
            return out
        finally:
            with self._lock:
                for i in ids:
                    self._pending.pop((cid, i), None)
