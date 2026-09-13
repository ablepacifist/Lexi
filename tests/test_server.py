"""Lexi's /turn endpoint, driven with fake engines so no model is loaded.

The decode path itself (WebM/Opus -> text) is model- and codec-dependent and is
verified on the device; what is pinned here is everything around it: auth,
input validation, the shape of the response, and how outages surface.
"""
import base64
import wave
import io

import pytest
from fastapi.testclient import TestClient

from lexi.config import LexiConfig
from lexi.obrenna_client import BrainUnreachable
from lexi.server import create_app, pcm_to_wav

TOKEN = "t" * 40


class _FakeStt:
    def __init__(self, text="turn on the light"):
        self.text = text
        self.seen: bytes | None = None

    def transcribe_encoded(self, data: bytes) -> str:
        self.seen = data
        return self.text


class _FakeTts:
    sample_rate = 16000

    def synthesize(self, text):
        self.spoken = text
        yield b"\x01\x00" * 800


class _FakePipeline:
    def __init__(self, reply="Done.", raises=None):
        self.reply, self.raises, self.prompt = reply, raises, None

    def run_text_turn(self, text, speak=False, **_):
        self.prompt = text
        if self.raises:
            raise self.raises
        return self.reply


class _FakeState:
    """Duck-types server._State without loading anything."""

    def __init__(self, *, stt=None, pipeline=None, token=TOKEN):
        self.cfg = LexiConfig()
        self.engines = type("E", (), {"stt": stt or _FakeStt(), "tts": _FakeTts()})()
        self.pipeline = pipeline or _FakePipeline()
        self.token = token


def _client(state):
    return TestClient(create_app(state))


def _post(client, body=b"fake-webm-bytes", token=TOKEN):
    """The clip is the raw request body, not multipart — every hop just forwards
    bytes."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    headers["Content-Type"] = "application/octet-stream"
    return client.post("/turn", content=body, headers=headers)


# ── auth ─────────────────────────────────────────────────────────────────────

def test_health_needs_no_token():
    assert _client(_FakeState()).get("/health").json() == {"ok": True}


@pytest.mark.parametrize("token", [None, "wrong", ""])
def test_turn_rejects_missing_or_wrong_token(token):
    assert _post(_client(_FakeState()), token=token).status_code == 401


def test_unconfigured_token_fails_closed():
    """An empty token must never be read as 'no auth required'."""
    r = _post(_client(_FakeState(token="")), token="anything")
    assert r.status_code == 503


# ── input validation ─────────────────────────────────────────────────────────

def test_empty_upload_is_rejected():
    assert _post(_client(_FakeState()), body=b"").status_code == 400


def test_oversized_clip_is_rejected():
    from lexi.server import MAX_CLIP_BYTES
    r = _post(_client(_FakeState()), body=b"x" * (MAX_CLIP_BYTES + 1))
    assert r.status_code == 413


def test_undecodable_audio_is_a_400_not_a_500():
    class Boom(_FakeStt):
        def transcribe_encoded(self, data):
            raise RuntimeError("bad container")

    assert _post(_client(_FakeState(stt=Boom()))).status_code == 400


# ── behaviour ────────────────────────────────────────────────────────────────

def test_full_turn_returns_transcript_reply_and_playable_wav():
    state = _FakeState(stt=_FakeStt("what time is it"), pipeline=_FakePipeline("It is noon."))
    r = _post(_client(state))
    assert r.status_code == 200
    body = r.json()

    assert body["transcript"] == "what time is it"
    assert body["reply"] == "It is noon."
    assert state.pipeline.prompt == "what time is it"   # transcript drove the brain
    assert set(body["timings"]) == {"stt", "brain", "tts", "total"}

    wav = base64.b64decode(body["audio_wav_base64"])
    with wave.open(io.BytesIO(wav)) as w:      # parses => a browser can play it
        assert w.getnchannels() == 1 and w.getsampwidth() == 2


def test_markdown_is_stripped_before_synthesis():
    """The reply keeps its formatting; only what reaches TTS is flattened."""
    state = _FakeState(pipeline=_FakePipeline("It is **noon** now."))
    body = _post(_client(state)).json()
    assert body["reply"] == "It is **noon** now."
    assert body["spoken"] == "It is noon now."


def test_silence_is_reported_not_sent_to_the_brain():
    state = _FakeState(stt=_FakeStt("   "), pipeline=_FakePipeline())
    body = _post(_client(state)).json()
    assert body["note"] == "no speech detected"
    assert state.pipeline.prompt is None       # never bothered the brain


def test_brain_outage_surfaces_as_503():
    """Alison is often off; that is a service state, not a client error."""
    state = _FakeState(pipeline=_FakePipeline(raises=BrainUnreachable("no base")))
    r = _post(_client(state))
    assert r.status_code == 503
    assert "unreachable" in r.json()["detail"]


# ── wav helper ───────────────────────────────────────────────────────────────

def test_pcm_to_wav_roundtrips():
    pcm = b"\x01\x02" * 100
    with wave.open(io.BytesIO(pcm_to_wav(pcm, 22050))) as w:
        assert w.getframerate() == 22050
        assert w.readframes(w.getnframes()) == pcm
