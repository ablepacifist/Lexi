"""The bearer check and the upload size limit on both Lexi endpoints (/turn and the STT-only /stt)."""
import pytest
from fastapi.testclient import TestClient

from lexi.server import MAX_CLIP_BYTES, create_app, create_stt_app
from tests.test_server import TOKEN, _FakeState


class _FakeStt:
    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.seen = (len(pcm), sample_rate)
        return "hello"


class _FakeSttState:
    def __init__(self, token=TOKEN):
        self.stt = _FakeStt()
        self.token = token


def _turn(client, headers, body=b"fake-webm-bytes"):
    return client.post("/turn", content=body, headers={"Content-Type": "application/octet-stream", **headers})


def _stt(client, headers, body=b"\x01\x00" * 100):
    return client.post("/stt", content=body, headers={"Content-Type": "application/octet-stream", **headers})


@pytest.fixture
def turn_client():
    return TestClient(create_app(_FakeState()))


@pytest.fixture
def stt_client():
    return TestClient(create_stt_app(_FakeSttState()))


# ── the token must arrive as "Authorization: Bearer <token>" ──────────────────

@pytest.mark.parametrize("header", [
    {"Authorization": TOKEN},                      # a bare token, no scheme
    {"Authorization": f"Basic {TOKEN}"},           # wrong scheme
    {"Authorization": f"bearer {TOKEN}"},          # callers send exactly "Bearer"
    {"Authorization": f"Bearer {TOKEN}x"},         # right start, wrong value
    {"Authorization": f"Bearer {TOKEN[:-1]}"},     # a prefix of the right value
    {"Authorization": "Bearer "},
    {"Authorization": ""},
    {},
])
def test_only_a_proper_bearer_token_is_accepted(turn_client, stt_client, header):
    assert _turn(turn_client, header).status_code == 401
    assert _stt(stt_client, header).status_code == 401


def test_the_correct_bearer_token_still_works(stt_client):
    r = _stt(stt_client, {"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200, r.text
    assert r.json()["transcript"] == "hello"


def test_surrounding_whitespace_is_tolerated(stt_client):
    assert _stt(stt_client, {"Authorization": f"Bearer  {TOKEN} "}).status_code == 200


def test_the_comparison_is_constant_time():
    import inspect
    import lexi.server as server
    src = inspect.getsource(server._require_token)
    assert "hmac.compare_digest" in src
    assert "!= token" not in src and "== token" not in src


def test_an_unset_token_still_fails_closed(stt_client):
    client = TestClient(create_stt_app(_FakeSttState(token="")))
    assert _stt(client, {"Authorization": "Bearer anything"}).status_code == 503


# ── size limit is enforced before the body is read ───────────────────────────

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_a_declared_oversized_body_is_refused_without_reading_it(stt_client, monkeypatch):
    """A Content-Length over the limit is rejected up front: the body stream is never consumed."""
    from starlette.requests import Request
    consumed = []
    original = Request.stream

    async def spying_stream(self):
        async for chunk in original(self):
            consumed.append(len(chunk))
            yield chunk

    monkeypatch.setattr(Request, "stream", spying_stream)
    r = _stt(stt_client, {**AUTH, "Content-Length": str(MAX_CLIP_BYTES + 1)}, body=b"x" * 1000)
    assert r.status_code == 413
    assert consumed == []


def test_a_body_over_the_limit_is_refused_on_both_endpoints(turn_client, stt_client):
    big = b"x" * (MAX_CLIP_BYTES + 1)
    assert _turn(turn_client, AUTH, body=big).status_code == 413
    assert _stt(stt_client, AUTH, body=big).status_code == 413


def test_a_body_exactly_at_the_limit_is_allowed_through_to_the_engine(stt_client):
    assert _stt(stt_client, AUTH, body=b"\x01\x00" * (MAX_CLIP_BYTES // 2)).status_code == 200


def test_a_chunked_body_with_no_length_is_cut_off_at_the_limit(stt_client):
    def chunks():
        for _ in range(MAX_CLIP_BYTES // 65536 + 2):
            yield b"x" * 65536
    r = stt_client.post("/stt", content=chunks(), headers={**AUTH, "Content-Type": "application/octet-stream"})
    assert r.status_code == 413


def test_a_garbage_content_length_is_a_400(stt_client):
    r = _stt(stt_client, {**AUTH, "Content-Length": "abc"})
    assert r.status_code == 400
    assert "Content-Length" in r.json()["detail"]


def test_an_empty_body_is_still_a_400(stt_client, turn_client):
    assert _stt(stt_client, AUTH, body=b"").status_code == 400
    assert _turn(turn_client, AUTH, body=b"").status_code == 400
