"""Play music from the Lexicon library on the Pi's own speaker.

This is Lexi-local logic (runs on the Pi): it logs into Lexicon, searches the
media library, and plays the track's stream through **mpv**. It works whether or
not alison's brain is online — "play music" is a device command Lexi handles
itself, straight to Lexicon (LAN or the api.alex-dyakin.com tunnel).

Audio note: the Pi's 3.5mm output is a single-open ALSA device, so the caller
speaks any TTS confirmation BEFORE starting playback, and stops playback before
speaking again. mpv runs as a detached process; ``stop`` kills it.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

import httpx

from .config import IdentityConfig

logger = logging.getLogger(__name__)


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

    # ── playback (detached mpv) ──────────────────────────────────────────────
    def play(self, media_id: int) -> None:
        self._ensure_login()
        if not shutil.which("mpv"):
            raise MediaError("mpv is not installed on this device.")
        self.stop()
        url = f"{self._base}/api/media/stream/{media_id}"
        cmd = ["mpv", "--no-video", "--really-quiet",
               f"--http-header-fields=Cookie: JSESSIONID={self._cookie}"]
        if self._audio_device:
            cmd.append(f"--audio-device={self._audio_device}")  # force the 3.5mm jack
        cmd.append(url)
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise MediaError(f"Could not start mpv: {exc}") from exc
        logger.info("mpv playing media_id=%s", media_id)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    @property
    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

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
        items = self._authed_json(f"/api/media/user/{self._user_id}") or []
        if not isinstance(items, list):
            return []
        return [int(it["id"]) for it in items
                if it.get("mediaType") == "MUSIC" and it.get("id") is not None]

    def play_many(self, media_ids: list[int], shuffle: bool = False) -> None:
        """Play a list of tracks as an mpv queue (mpv auto-advances; --shuffle
        randomizes)."""
        self._ensure_login()
        if not shutil.which("mpv"):
            raise MediaError("mpv is not installed on this device.")
        if not media_ids:
            raise MediaError("nothing to play")
        self.stop()
        cmd = ["mpv", "--no-video", "--really-quiet",
               f"--http-header-fields=Cookie: JSESSIONID={self._cookie}"]
        if self._audio_device:
            cmd.append(f"--audio-device={self._audio_device}")
        if shuffle:
            cmd.append("--shuffle")
        cmd += [f"{self._base}/api/media/stream/{mid}" for mid in media_ids]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise MediaError(f"Could not start mpv: {exc}") from exc
        logger.info("mpv playing %d tracks (shuffle=%s)", len(media_ids), shuffle)


def _jsessionid_from_headers(resp: httpx.Response) -> str | None:
    """Fallback: pull JSESSIONID out of a Set-Cookie header httpx didn't jar
    (the cookie's Path=/api can keep it out of resp.cookies in some versions)."""
    for raw in resp.headers.get_list("set-cookie"):
        if raw.startswith("JSESSIONID="):
            return raw.split(";", 1)[0][len("JSESSIONID="):]
    return None
