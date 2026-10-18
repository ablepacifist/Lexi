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

from .config import LexiConfig, bind_defaults, load_config
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
        self.token = self.cfg.tool_token()


class _SttState:
    """State for the STT-only service (aragon): just the transcriber + token.
    Deliberately builds no tts/wake/vad/brain/identity, so the offload host needs
    only faster-whisper (numpy path → no PyAV)."""

    def __init__(self) -> None:
        from .engines.stt import WhisperStt

        self.cfg: LexiConfig = load_config()
        self.stt = WhisperStt(self.cfg.engines)
        self.token = self.cfg.tool_token()


def _require_token(token: str, authorization: str) -> None:
    """Shared bearer check. Fails closed: an unset token is 503, not open."""
    if not token:
        raise HTTPException(503, "Lexi tool token is not configured")
    if authorization.removeprefix("Bearer ").strip() != token:
        raise HTTPException(401, "bad or missing tool token")


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
        _require_token(st.token, authorization)

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


def create_stt_app(state: _SttState | None = None) -> FastAPI:
    """STT-only service for an offload host (aragon): POST raw PCM, get a
    transcript. No brain/tts/wake — the host needs only faster-whisper."""
    st = state or _SttState()
    app = FastAPI(title="Lexi STT", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=["POST"])

    def require_token(authorization: str = Header(default="")) -> None:
        _require_token(st.token, authorization)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/stt", dependencies=[Depends(require_token)])
    async def stt(request: Request, sample_rate: int = 16000) -> dict:
        """Transcribe raw int16 mono PCM. Raw PCM (not an encoded clip) keeps the
        host on faster-whisper's numpy path — no PyAV needed."""
        pcm = await request.body()
        if not pcm:
            raise HTTPException(400, "empty audio upload")
        if len(pcm) > MAX_CLIP_BYTES:
            raise HTTPException(413, f"clip larger than {MAX_CLIP_BYTES} bytes")
        t0 = time.perf_counter()
        try:
            transcript = st.stt.transcribe(pcm, sample_rate)
        except Exception as exc:
            logger.warning("transcribe failed: %s", exc)
            raise HTTPException(500, f"could not transcribe: {exc}") from exc
        return {"transcript": transcript,
                "timings": {"stt": round(time.perf_counter() - t0, 3)}}

    return app


def main() -> int:  # pragma: no cover - process entry
    import argparse

    import uvicorn

    default_host, default_port = bind_defaults()  # layered LEXI_HOST/LEXI_PORT; 127.0.0.1:8765 standalone

    ap = argparse.ArgumentParser(prog="lexi-server")
    ap.add_argument("--stt", action="store_true",
                    help="run the STT-only offload service (for hosts like aragon)")
    ap.add_argument("--host", default=default_host,
                    help="bind address (use 0.0.0.0 for --stt so the Pi can reach it)")
    ap.add_argument("--port", type=int, default=default_port)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = create_stt_app() if args.stt else create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
