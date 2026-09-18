"""Play music from the Lexicon library on the Pi's own speaker.

This is Lexi-local logic (runs on the Pi): it logs into Lexicon, searches the
media library, and plays the track's stream through **mpv**. It works whether or
not alison's brain is online — "play music" is a device command Lexi handles
itself, straight to Lexicon (LAN or the api.alex-dyakin.com tunnel).

Audio note: mpv runs as a detached process and is driven live through its JSON
IPC socket (``--input-ipc-server``) — so we can pause / resume / skip without
killing it. ``stop`` kills the process. Whether TTS can speak *while* mpv is
paused depends on the OS audio setup: on the Pi both are routed through PipeWire
so they mix; on a single-open ALSA device the caller must stop before speaking.

Live stream: Lexicon also serves a synchronized communal stream
(``/api/livestream/*``) — everyone hears the same track at the same position.
``play_livestream`` joins it (seeking to the live offset) and follows the SSE
update feed so the Pi switches tracks when the communal stream advances.
"""
from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

from .config import IdentityConfig

logger = logging.getLogger(__name__)

# mpv JSON IPC socket, created in the Lexi working directory when playback starts.
_IPC_SOCKET = ".mpv-ipc.sock"


class MediaError(RuntimeError):
    """Login, search, or playback against Lexicon failed."""


class MediaPlayer:
    def __init__(self, cfg: IdentityConfig, audio_device: str = ""):
        self._cfg = cfg
        self._base = cfg.lexicon_base_url.rstrip("/")
        self._audio_device = audio_device  # mpv --audio-device (e.g. the 3.5mm jack)
        self._cookie: str | None = None   # JSESSIONID value
        self._user_id: int | None = None
        self._proc: subprocess.Popen | None = None
        self._ipc_path = str(Path(_IPC_SOCKET).resolve())
        self._paused = False
        # Live stream: a background thread follows the communal stream's SSE feed
        # and re-points mpv when the shared track changes.
        self._livestream_active = False
        self._ls_current_media: int | None = None
        self._ls_stop = threading.Event()
        self._ls_thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.lexicon_username)

    # ── Lexicon auth (lazy, cached) ──────────────────────────────────────────
    def _ensure_login(self) -> None:
        if self._cookie:
            return
        if not self._cfg.lexicon_username:
            raise MediaError("No Lexicon credentials configured for media playback.")
        try:
            resp = httpx.post(
                f"{self._base}/api/auth/login",
                json={
                    "username": self._cfg.lexicon_username,
                    "password": self._cfg.lexicon_password,
                },
                timeout=10.0,
            )
        except httpx.HTTPError as exc:
            raise MediaError(f"Could not reach Lexicon: {exc}") from exc
        if resp.status_code != 200:
            raise MediaError(f"Lexicon login failed ({resp.status_code}).")
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if not data.get("success"):
            raise MediaError("Lexicon login rejected (bad credentials?).")
        self._user_id = data.get("id") or data.get("playerId")
        jsid = resp.cookies.get("JSESSIONID") or _jsessionid_from_headers(resp)
        if not jsid:
            raise MediaError("Lexicon did not return a session cookie.")
        self._cookie = jsid
        logger.info("Lexicon media login ok (user_id=%s)", self._user_id)

    def _headers(self) -> dict[str, str]:
        return {"Cookie": f"JSESSIONID={self._cookie}"}

    # ── search ───────────────────────────────────────────────────────────────
    def search(self, query: str) -> list[dict]:
        self._ensure_login()
        try:
            resp = httpx.get(
                f"{self._base}/api/media/search",
                params={"q": query},
                headers=self._headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            items = resp.json()
        except httpx.HTTPError as exc:
            raise MediaError(f"Search failed: {exc}") from exc
        except ValueError:
            return []
        return items if isinstance(items, list) else []

    @staticmethod
    def pick(items: list[dict]) -> dict | None:
        """Choose the best match: prefer MUSIC, then AUDIOBOOK, else the first."""
        for want in ("MUSIC", "AUDIOBOOK"):
            for it in items:
                if it.get("mediaType") == want:
                    return it
        return items[0] if items else None

    # ── playback (detached mpv, driven over its IPC socket) ──────────────────
    def _url(self, media_id: int) -> str:
        return f"{self._base}/api/media/stream/{media_id}"

    def _spawn(self, urls: list[str], *, shuffle: bool = False,
               start: float | None = None) -> None:
        """(Re)launch mpv with an IPC socket for the given URL queue."""
        self._ensure_login()
        if not shutil.which("mpv"):
            raise MediaError("mpv is not installed on this device.")
        if not urls:
            raise MediaError("nothing to play")
        self.stop()
        try:
            Path(self._ipc_path).unlink()  # stale socket from a killed mpv
        except OSError:
            pass
        cmd = ["mpv", "--no-video", "--really-quiet",
               f"--input-ipc-server={self._ipc_path}",
               f"--http-header-fields=Cookie: JSESSIONID={self._cookie}"]
        if self._audio_device:
            cmd.append(f"--audio-device={self._audio_device}")  # force a sink (e.g. 3.5mm)
        if shuffle:
            cmd.append("--shuffle")
        if start is not None:
            cmd.append(f"--start=+{max(0, int(start))}")  # seek into a live track
        cmd += urls
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise MediaError(f"Could not start mpv: {exc}") from exc
        self._paused = False

    def play(self, media_id: int) -> None:
        # A one-item playlist so "next" degrades gracefully (no-op) instead of erroring.
        self._spawn([self._url(media_id)])
        logger.info("mpv playing media_id=%s", media_id)

    def stop(self) -> None:
        self._stop_livestream_follower()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._paused = False

    @property
    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_paused(self) -> bool:
        return self._paused and self.is_playing

    @property
    def livestream_active(self) -> bool:
        return self._livestream_active and self.is_playing

    # ── live control over mpv's JSON IPC socket (pause / resume / skip) ──────
    def _ipc(self, command: list) -> None:
        """Send one JSON command to the running mpv. Best-effort: a control
        hiccup must never crash a voice turn, so failures are logged, not raised."""
        if not self.is_playing:
            return
        payload = json.dumps({"command": command}).encode() + b"\n"
        last: Exception | None = None
        for _ in range(10):  # mpv creates the socket a beat after launch
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(1.0)
                try:
                    s.connect(self._ipc_path)
                    s.sendall(payload)
                    try:
                        s.recv(4096)  # drain the reply so mpv isn't left blocking
                    except OSError:
                        pass
                finally:
                    s.close()
                return
            except OSError as exc:
                last = exc
                time.sleep(0.1)
        logger.debug("mpv IPC command %s failed: %s", command, last)

    def pause(self) -> None:
        if self.is_playing and not self._paused:
            self._ipc(["set_property", "pause", True])
            self._paused = True

    def resume(self) -> None:
        if self.is_playing and self._paused:
            self._ipc(["set_property", "pause", False])
            self._paused = False

    def next(self) -> None:
        """Advance to the next queued track (and keep playing if we were paused)."""
        if self.is_playing:
            self._ipc(["playlist-next", "force"])
            if self._paused:
                self.resume()

    # ── library helpers: fuzzy track match, playlists, all-music ─────────────
    def _authed_json(self, path: str, params: dict | None = None):
        self._ensure_login()
        try:
            resp = httpx.get(
                f"{self._base}{path}", params=params or {},
                headers=self._headers(), timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise MediaError(f"Lexicon request failed ({path}): {exc}") from exc
        except ValueError:
            return None

    def fuzzy_find(self, query: str) -> dict | None:
        """Fallback when exact search misses (STT mis-hears a title): pick the
        closest-titled MUSIC/AUDIOBOOK track in the user's library."""
        import difflib

        self._ensure_login()  # resolve _user_id BEFORE it goes into the URL
        items = self._authed_json(f"/api/media/user/{self._user_id}") or []
        if not isinstance(items, list):
            return None
        cands = [it for it in items if it.get("mediaType") in ("MUSIC", "AUDIOBOOK")] or items
        q = query.lower()
        best, best_r = None, 0.0
        for it in cands:
            title = (it.get("title") or "").lower()
            if not title:
                continue
            r = difflib.SequenceMatcher(None, q, title).ratio()
            if q in title or title in q:  # reward substring hits
                r = max(r, 0.85)
            if r > best_r:
                best_r, best = r, it
        return best if best_r >= 0.6 else None

    def find_playlist(self, name: str) -> dict | None:
        import difflib

        self._ensure_login()  # resolve _user_id BEFORE it goes into the query
        pls = self._authed_json("/api/playlists", {"userId": self._user_id}) or []
        if not isinstance(pls, list):
            return None
        nl = name.lower()
        for p in pls:                         # exact name
            if (p.get("name") or "").lower() == nl:
                return p
        for p in pls:                         # substring
            if nl in (p.get("name") or "").lower():
                return p
        best, best_r = None, 0.0              # fuzzy
        for p in pls:
            r = difflib.SequenceMatcher(None, nl, (p.get("name") or "").lower()).ratio()
            if r > best_r:
                best_r, best = r, p
        return best if best_r >= 0.6 else None

    def playlist_media_ids(self, playlist_id) -> list[int]:
        pl = self._authed_json(f"/api/playlists/{playlist_id}") or {}
        items = pl.get("items") or []
        ids: list[int] = []
        for it in items:
            # Playlist items are wrappers, NOT MediaFile objects:
            #   {"playlistId":…, "mediaFileId": <track id>, "position":…, "mediaFile": {…}}
            # The track id is mediaFileId (fall back to the nested mediaFile.id).
            # (The server's itemCount/mediaFileIds fields are unreliable here.)
            mid = it.get("mediaFileId")
            if mid is None:
                mid = (it.get("mediaFile") or {}).get("id")
            if mid is not None:
                ids.append(int(mid))
        return ids

    def all_music_ids(self) -> list[int]:
        self._ensure_login()  # resolve _user_id BEFORE it goes into the URL
        items = self._authed_json(f"/api/media/user/{self._user_id}") or []
        if not isinstance(items, list):
            return []
        return [int(it["id"]) for it in items
                if it.get("mediaType") == "MUSIC" and it.get("id") is not None]

    def play_many(self, media_ids: list[int], shuffle: bool = False) -> None:
        """Play a list of tracks as an mpv queue (mpv auto-advances; --shuffle
        randomizes; "next" skips within it via IPC)."""
        if not media_ids:
            raise MediaError("nothing to play")
        self._spawn([self._url(m) for m in media_ids], shuffle=shuffle)
        logger.info("mpv playing %d tracks (shuffle=%s)", len(media_ids), shuffle)

    # ── live stream (synchronized communal stream) ───────────────────────────
    def livestream_state(self) -> dict:
        data = self._authed_json("/api/livestream/state") or {}
        return data.get("state") or {}

    def play_livestream(self) -> None:
        """Join Lexicon's communal stream at its live position and follow it."""
        st = self.livestream_state()
        mid = st.get("currentMediaId")
        if not mid:
            raise MediaError("The live stream has nothing playing right now.")
        self._spawn([self._url(int(mid))], start=self._livestream_offset(st))
        self._ls_current_media = int(mid)
        self._livestream_active = True
        self._start_livestream_follower()
        logger.info("joined live stream at media_id=%s", mid)

    def livestream_skip(self) -> None:
        """Cast a vote to skip the current communal track. The SSE feed then
        tells us to switch when the stream actually advances."""
        self._ensure_login()
        try:
            resp = httpx.post(
                f"{self._base}/api/livestream/skip",
                json={"userId": self._user_id},
                headers=self._headers(), timeout=10.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MediaError(f"Skip vote failed: {exc}") from exc

    @staticmethod
    def _livestream_offset(st: dict) -> float:
        """Seconds into the current track the communal stream is right now:
        its recorded position plus wall-clock elapsed since it started."""
        pos = (st.get("currentPositionMs") or 0) / 1000.0
        started = st.get("currentStartTime")
        if not started:
            return max(0.0, pos)
        try:
            t = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        except ValueError:
            return max(0.0, pos)
        # aware timestamp → compare in its zone; naive → assume same locale as the
        # Pi (both on-site), which is the common case for this homelab.
        now = datetime.now(t.tzinfo) if t.tzinfo else datetime.now()
        elapsed = (now - t).total_seconds()
        if elapsed < 0 or elapsed > 6 * 3600:  # clock skew / unparseable → ignore
            elapsed = 0.0
        return max(0.0, pos + elapsed)

    def _start_livestream_follower(self) -> None:
        self._stop_livestream_follower()
        self._ls_stop = threading.Event()
        self._ls_thread = threading.Thread(
            target=self._follow_livestream, name="lexi-livestream", daemon=True
        )
        self._ls_thread.start()

    def _stop_livestream_follower(self) -> None:
        self._livestream_active = False
        self._ls_stop.set()
        self._ls_thread = None
        self._ls_current_media = None

    def _follow_livestream(self) -> None:  # pragma: no cover - network thread
        url = f"{self._base}/api/livestream/updates"
        event: str | None = None
        try:
            with httpx.stream("GET", url, headers=self._headers(), timeout=None) as resp:
                for line in resp.iter_lines():
                    if self._ls_stop.is_set():
                        return
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event in ("init", "state-update"):
                        self._on_livestream_state(line.split(":", 1)[1].strip())
        except Exception as exc:  # network drop, shutdown, etc.
            logger.debug("live stream follower ended: %s", exc)

    def _on_livestream_state(self, data: str) -> None:  # pragma: no cover - network thread
        try:
            payload = json.loads(data)
        except ValueError:
            return
        st = payload.get("state") or payload.get("data") or payload
        mid = st.get("currentMediaId") if isinstance(st, dict) else None
        if not mid or int(mid) == self._ls_current_media:
            return
        self._ls_current_media = int(mid)
        logger.info("live stream advanced to media_id=%s", mid)
        self._ipc(["loadfile", self._url(int(mid)), "replace"])


def _jsessionid_from_headers(resp: httpx.Response) -> str | None:
    """Fallback: pull JSESSIONID out of a Set-Cookie header httpx didn't jar
    (the cookie's Path=/api can keep it out of resp.cookies in some versions)."""
    for raw in resp.headers.get_list("set-cookie"):
        if raw.startswith("JSESSIONID="):
            return raw.split(";", 1)[0][len("JSESSIONID="):]
    return None
