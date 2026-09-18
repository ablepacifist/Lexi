"""Conversation transcript log with a TTL.

One JSON line per voice turn — what Whisper heard plus what Lexi did — so we can
see the transcriptions after the fact and debug without a live tail. It is
Lexi-local and private (gitignored), separate from the brain's per-user memory on
alison. Every write is fail-safe: logging must never break a voice turn.

Entry shape: {ts, transcript, kind: "chat"|"media"|"error", detail, chat_id,
account_id}. Old entries are pruned by age at startup.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationLog:
    def __init__(self, path: str, ttl_days: int = 14, enabled: bool = True):
        self._path = Path(path)
        self._ttl_days = max(0, int(ttl_days))
        self._enabled = bool(enabled and path)
        if self._enabled:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self.prune()
            except Exception as exc:  # pragma: no cover - fs dependent
                logger.warning("history init failed (%s); disabling logging", exc)
                self._enabled = False

    def append(
        self, transcript: str, kind: str, detail=None, *, chat_id=None, account_id=None
    ) -> None:
        if not self._enabled:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "transcript": transcript,
            "kind": kind,
            "detail": detail,
            "chat_id": chat_id,
            "account_id": account_id,
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - fs dependent
            logger.warning("history append failed: %s", exc)

    def prune(self) -> None:
        """Drop entries older than ttl_days. Called at startup; rewrites the file
        once. Unparseable lines are kept rather than risk losing data."""
        if not self._enabled or self._ttl_days <= 0 or not self._path.exists():
            return
        cutoff = datetime.now(timezone.utc).timestamp() - self._ttl_days * 86400
        try:
            kept: list[str] = []
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    ts = datetime.fromisoformat(json.loads(line)["ts"]).timestamp()
                except Exception:
                    kept.append(line)
                    continue
                if ts >= cutoff:
                    kept.append(line)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text("".join(f"{ln}\n" for ln in kept), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:  # pragma: no cover - fs dependent
            logger.warning("history prune failed: %s", exc)
