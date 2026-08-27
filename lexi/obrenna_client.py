"""Client for Obrenna's brain on alison.

Sends a transcript to the EXISTING ``POST /api/chat/stream`` (no new brain-side
route) and yields the answer's token text as it streams, so the pipeline can
sentence-chunk it into TTS. Authenticates to the Obrenna gateway with the same
headless shared secret the codebase-agent uses (``X-Obrenna-Agent-Token``).
Tries the LAN base first and fails over to the tunnel.

Per-user identity rides in the request body as ``account_id`` (resolved from
Lexicon); the shared secret only proves *this machine* may call the brain.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterator

import httpx

from .config import BrainConfig

logger = logging.getLogger(__name__)


class BrainUnreachable(RuntimeError):
    """No configured base URL (LAN or tunnel) accepted the request."""


class BrainAuthError(RuntimeError):
    """The gateway rejected the shared secret (401/403)."""


@dataclass
class TurnResult:
    chat_id: str
    text: str


class ObrennaClient:
    def __init__(self, cfg: BrainConfig, agent_token: str):
        self._cfg = cfg
        self._token = agent_token

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if self._token:
            # Match the gateway's verifier: header OR Bearer are both accepted.
            h["X-Obrenna-Agent-Token"] = self._token
        return h

    def stream_turn(
        self,
        message: str,
        *,
        account_id: str,
        chat_id: str | None = None,
        on_token=None,
        on_event=None,
    ) -> TurnResult:
        """Run one turn. Calls ``on_token(text)`` for each answer token (for live
        sentence-chunked TTS) and ``on_event(dict)`` for every agent event
        (phase/thinking/tool/…). Returns the final chat_id + full text.

        Fails over tunnel → LAN on a connection error *before* the response
        starts; once bytes are flowing we are committed to that base.
        """
        payload = {
            "message": message,
            "account_id": account_id,
            "orchestrator": self._cfg.orchestrator,
            "workers_enabled": self._cfg.workers_enabled,
        }
        if chat_id:
            payload["chat_id"] = chat_id

        last_err: Exception | None = None
        for base in self._cfg.bases:
            url = base.rstrip("/") + "/api/chat/stream"
            try:
                return self._stream_one(url, payload, on_token, on_event)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                logger.warning("Brain base %s unreachable (%s); trying next.", base, exc)
                last_err = exc
                continue
        raise BrainUnreachable(
            f"No Obrenna base reachable ({', '.join(self._cfg.bases)})"
        ) from last_err

    def _stream_one(self, url, payload, on_token, on_event) -> TurnResult:
        timeout = httpx.Timeout(self._cfg.request_timeout_s, connect=self._cfg.connect_timeout_s)
        chat_id = payload.get("chat_id", "")
        text_parts: list[str] = []
        final_text: str | None = None

        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
                if resp.status_code in (401, 403):
                    raise BrainAuthError(
                        f"Gateway rejected the shared secret ({resp.status_code}) at {url}"
                    )
                resp.raise_for_status()
                for event, data in _iter_sse(resp.iter_lines()):
                    if event == "agent_event":
                        env = _loads(data)
                        if env is None:
                            continue
                        if on_event:
                            on_event(env)
                        if env.get("chat_id"):
                            chat_id = env["chat_id"]
                        if env.get("type") == "token":
                            tok = (env.get("payload") or {}).get("text", "")
                            if tok:
                                text_parts.append(tok)
                                if on_token:
                                    on_token(tok)
                    elif event == "response":
                        env = _loads(data) or {}
                        chat_id = env.get("chat_id", chat_id)
                        msg = env.get("message") or {}
                        final_text = msg.get("text")
                    elif event == "error":
                        env = _loads(data) or {}
                        raise RuntimeError(f"Brain error: {env.get('message', 'unknown')}")

        return TurnResult(chat_id=chat_id, text=final_text if final_text is not None else "".join(text_parts))


def _loads(data: str) -> dict | None:
    try:
        obj = json.loads(data)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _iter_sse(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    """Minimal Server-Sent-Events parser.

    Yields (event, data) once per blank-line-terminated block. Defaults the event
    name to "message" per the SSE spec; comments (``: keepalive``) are ignored.
    """
    event = "message"
    data_lines: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue  # comment / keepalive
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)
