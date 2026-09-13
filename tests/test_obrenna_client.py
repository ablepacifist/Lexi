import json

import httpx
import pytest

from lexi import obrenna_client as oc
from lexi.config import BrainConfig
from lexi.obrenna_client import BrainUnreachable, ObrennaClient

SSE = (
    "event: agent_event\n"
    'data: {"channel":"agent_event","chat_id":"c1","message_id":"m1","type":"token","payload":{"text":"Hello"}}\n'
    "\n"
    "event: agent_event\n"
    'data: {"channel":"agent_event","chat_id":"c1","message_id":"m1","type":"token","payload":{"text":" world"}}\n'
    "\n"
    "event: response\n"
    'data: {"chat_id":"c1","message":{"id":"m1","role":"assistant","text":"Hello world","artifacts":[],"files":[],"tool_events":[],"created_at":"t"},"memory_events":[]}\n'
    "\n"
)


def _install(monkeypatch, handler):
    """Route the client's internal httpx.Client through a MockTransport."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(oc.httpx, "Client", factory)


def test_streams_tokens_and_returns_final_text(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("x-obrenna-agent-token")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE)

    _install(monkeypatch, handler)

    cfg = BrainConfig(lan_base_url="http://lan:9080", tunnel_base_url="https://tunnel")
    client = ObrennaClient(cfg, agent_token="s" * 40)

    tokens = []
    result = client.stream_turn(
        "hi", account_id="userA", chat_id=None, on_token=tokens.append
    )

    assert tokens == ["Hello", " world"]
    assert result.text == "Hello world"
    assert result.chat_id == "c1"
    # auth header + per-user account + per-request orchestrator all sent
    assert seen["token"] == "s" * 40
    assert seen["url"].endswith("/api/chat/stream")
    assert seen["body"]["account_id"] == "userA"
    assert seen["body"]["orchestrator"] == cfg.orchestrator


def test_tunnel_failure_falls_over_to_lan(monkeypatch):
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        if request.url.host == "tunnel":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE)

    _install(monkeypatch, handler)

    cfg = BrainConfig(lan_base_url="http://lan:9080", tunnel_base_url="https://tunnel")
    client = ObrennaClient(cfg, agent_token="s" * 40)
    result = client.stream_turn("hi", account_id="userA")

    assert hits == ["tunnel", "lan"]  # tried tunnel first, then LAN
    assert result.text == "Hello world"


def test_all_bases_down_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _install(monkeypatch, handler)

    cfg = BrainConfig(lan_base_url="http://lan:9080", tunnel_base_url="https://tunnel")
    client = ObrennaClient(cfg, agent_token="s" * 40)
    with pytest.raises(BrainUnreachable):
        client.stream_turn("hi", account_id="userA")


def test_tunnel_530_falls_over_to_lan(monkeypatch):
    """Cloudflare answers 530 when the tunnel has no origin behind it.

    Observed for real with alison powered off. The gateway is up, so this is an
    HTTP *status*, not a connection error — before this was handled it escaped
    as a raw HTTPStatusError, skipping the LAN base entirely.
    """
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        if request.url.host == "tunnel":
            return httpx.Response(530, text="")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE)

    _install(monkeypatch, handler)

    cfg = BrainConfig(lan_base_url="http://lan:9080", tunnel_base_url="https://tunnel")
    client = ObrennaClient(cfg, agent_token="s" * 40)
    result = client.stream_turn("hi", account_id="userA")

    assert hits == ["tunnel", "lan"]
    assert result.text == "Hello world"


@pytest.mark.parametrize("status", [502, 503, 504, 522, 530])
def test_origin_down_statuses_everywhere_raise_brain_unreachable(monkeypatch, status):
    """Every base down must surface as BrainUnreachable, not HTTPStatusError.

    This is the signal the offline fallback keys on, so the exception type is
    load-bearing rather than cosmetic.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="")

    _install(monkeypatch, handler)

    cfg = BrainConfig(lan_base_url="http://lan:9080", tunnel_base_url="https://tunnel")
    client = ObrennaClient(cfg, agent_token="s" * 40)
    with pytest.raises(BrainUnreachable):
        client.stream_turn("hi", account_id="userA")


def test_real_server_error_is_not_retried(monkeypatch):
    """A 500 from Obrenna itself is an answer, not an outage.

    Retrying it against the other base would double every genuine failure and
    hide the real error, so it must propagate as-is.
    """
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        return httpx.Response(500, text="boom")

    _install(monkeypatch, handler)

    cfg = BrainConfig(lan_base_url="http://lan:9080", tunnel_base_url="https://tunnel")
    client = ObrennaClient(cfg, agent_token="s" * 40)
    with pytest.raises(httpx.HTTPStatusError):
        client.stream_turn("hi", account_id="userA")

    assert hits == ["tunnel"]  # stopped at the first base
