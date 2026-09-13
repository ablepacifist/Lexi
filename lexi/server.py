"""Lexi's HTTP face — a whole voice turn from a clip of recorded audio.

Exists because aragon's own microphone is not always somewhere a person can
stand. A browser has a microphone; this lets that browser drive the same
pipeline the mic loop drives, so the chain (STT → brain → TTS) can be exercised
from anywhere without hardware in every room.

**Bound to localhost by default and never exposed directly.** The browser is on
HTTPS, so it cannot call a plain-HTTP port on aragon anyway — mixed content is
blocked. Requests arrive proxied through lexiconServer, which already terminates
TLS and owns the session cookie. The bearer token below guards that hop.

Deliberately push-to-talk rather than wake-word streaming: the caller sends one
finished clip, so there is no wake word, no VAD and no socket to keep alive.
That tests the expensive, interesting part of the pipeline and leaves the
streaming satellite protocol to its own phase.
"""
from __future__ import annotations

import base64
import io
import logging
import time
import wave

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import LexiConfig, load_config
from .engines.registry import Engines, build_engines
from .identity import IdentityResolver
from .obrenna_client import BrainAuthError, BrainUnreachable, ObrennaClient
from .pipeline import VoicePipeline

logger = logging.getLogger(__name__)

# Reject implausibly large uploads before reading them. A spoken command is a
# few seconds; anything past this is a mistake or an attack, not a request.
MAX_CLIP_BYTES = 8 * 1024 * 1024


class _State:
    """Built once at startup. Engines are lazy, so this is cheap; the first
    request pays the model load."""

    def __init__(self) -> None:
        self.cfg: LexiConfig = load_config()
        self.engines: Engines = build_engines(self.cfg.engines)
        self.client = ObrennaClient(self.cfg.brain, self.cfg.agent_token())
        self.identity = IdentityResolver(self.cfg.identity)
        self.pipeline = VoicePipeline(
            self.cfg, self.client, self.identity, engines=self.engines
        )
        self.token = _tool_token(self.cfg)


def _tool_token(cfg: LexiConfig) -> str:
    """The shared secret callers must present.

    Reuses the brain token's env-then-file discipline rather than inventing a
    second mechanism, but under its own name so the two can be rotated apart.
    """
    import os
    from pathlib import Path

    env = os.getenv("LEXI_TOOL_TOKEN", "").strip()
    if env:
        return env
    try:
        return Path("./.tool_token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw int16 mono PCM in a WAV header so a browser can play it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def create_app(state: _State | None = None) -> FastAPI:
    st = state or _State()
    app = FastAPI(title="Lexi", docs_url=None, redoc_url=None)

    # The browser talks to lexiconServer, not to us, so no browser origin ever
    # reaches this directly. CORS stays closed.
    app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=["POST"])

    def require_token(authorization: str = Header(default="")) -> None:
        if not st.token:
            # Fail closed. An unset token must not mean "no auth required".
            raise HTTPException(503, "Lexi tool token is not configured")
        presented = authorization.removeprefix("Bearer ").strip()
        if presented != st.token:
            raise HTTPException(401, "bad or missing tool token")

    @app.get("/health")
    def health() -> dict:
        """Liveness only — deliberately unauthenticated and says nothing about
        configuration, so it is safe for a proxy to poll."""
        return {"ok": True}

    @app.post("/turn", dependencies=[Depends(require_token)])
    async def turn(request: Request) -> dict:
        """One full voice turn: audio in, transcript + reply + spoken reply out.

        The body is the raw clip, not multipart. Every hop between the browser
        and here then just forwards bytes — no multipart assembly in Java, and
        the container is sniffed by the decoder rather than trusted from a
        filename that no caller reliably sets.

        Returns per-stage timings so the caller can see where a slow turn went,
        which is the whole point of being able to run this from a browser.
        """
        clip = await request.body()
        if not clip:
            raise HTTPException(400, "empty audio upload")
        if len(clip) > MAX_CLIP_BYTES:
            raise HTTPException(413, f"clip larger than {MAX_CLIP_BYTES} bytes")

        t0 = time.perf_counter()
        try:
            transcript = st.engines.stt.transcribe_encoded(clip)
        except Exception as exc:
            logger.warning("decode/transcribe failed: %s", exc)
            raise HTTPException(400, f"could not decode audio: {exc}") from exc
        t_stt = time.perf_counter() - t0

        if not transcript.strip():
            # Silence is a normal outcome, not an error: say so and stop rather
            # than sending an empty prompt to the brain.
            return {
                "transcript": "",
                "reply": "",
                "note": "no speech detected",
                "timings": {"stt": round(t_stt, 3)},
            }

        t1 = time.perf_counter()
        try:
            reply = st.pipeline.run_text_turn(transcript, speak=False)
        except BrainAuthError as exc:
            raise HTTPException(502, f"brain rejected our token: {exc}") from exc
        except BrainUnreachable as exc:
            raise HTTPException(503, f"brain unreachable: {exc}") from exc
        t_brain = time.perf_counter() - t1

        # Synthesize here rather than in the pipeline: this turn's audio goes
        # back over HTTP to whoever asked, not to aragon's own speakers.
        t2 = time.perf_counter()
        from . import audio as lexi_audio

        spoken = lexi_audio.strip_markdown(reply)
        pcm = b"".join(st.engines.tts.synthesize(spoken)) if spoken else b""
        wav = pcm_to_wav(pcm, st.engines.tts.sample_rate) if pcm else b""
        t_tts = time.perf_counter() - t2

        return {
            "transcript": transcript,
            "reply": reply,
            "spoken": spoken,
            "audio_wav_base64": base64.b64encode(wav).decode("ascii"),
            "timings": {
                "stt": round(t_stt, 3),
                "brain": round(t_brain, 3),
                "tts": round(t_tts, 3),
                "total": round(t_stt + t_brain + t_tts, 3),
            },
        }

    return app


def main() -> int:  # pragma: no cover - process entry
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="lexi-server")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (keep local)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
