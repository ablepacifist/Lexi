"""Resolve which Lexicon user is speaking → an account_id the brain scopes by.

Decision: per-Lexicon-user identity. Lexicon owns identity (session cookie,
JSESSIONID). Lexi asks Lexicon "who am I" via ``GET /api/auth/me`` using a
session it holds, and passes the returned user id to the brain as account_id.

First cut: identity is the Lexicon user whose session Lexi holds (a per-device
login). Real per-speaker attribution on a shared device (voice enrollment /
diarization) is future work — until then a device resolves to one account, and
an unauthenticated device falls back to ``default_account_id``.
"""
from __future__ import annotations

import logging

import httpx

from .config import IdentityConfig

logger = logging.getLogger(__name__)


class IdentityResolver:
    def __init__(self, cfg: IdentityConfig, session_cookie: str | None = None):
        self._cfg = cfg
        # A Lexicon JSESSIONID for the device's user, if one is configured.
        self._session_cookie = session_cookie
        self._cached: str | None = None

    def account_id(self) -> str:
        """Return the account_id for the current speaker.

        Cached after first resolution. Falls back to the configured default when
        no Lexicon session is available or the lookup fails — Lexi must never
        block a turn on identity; it degrades to the shared bucket instead.
        """
        if self._cached:
            return self._cached
        self._cached = self._resolve() or self._cfg.default_account_id
        return self._cached

    def _resolve(self) -> str | None:
        if not self._session_cookie:
            return None
        url = self._cfg.lexicon_base_url.rstrip("/") + "/api/auth/me"
        try:
            resp = httpx.get(
                url,
                headers={"Cookie": f"JSESSIONID={self._session_cookie}"},
                timeout=5.0,
            )
            if resp.status_code != 200:
                logger.warning("Lexicon /api/auth/me returned %s", resp.status_code)
                return None
            data = resp.json()
            # Lexicon returns the user record; prefer a stable id field.
            uid = data.get("id") or data.get("playerId") or data.get("username")
            return str(uid) if uid is not None else None
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Lexicon identity lookup failed: %s", exc)
            return None
