"""Lexi entry point.

    python -m lexi                # console text mode (type; Lexi speaks/prints)
    python -m lexi --say "hi"     # one-shot
    python -m lexi --no-speak     # print the answer instead of TTS (no audio dev)
    python -m lexi --voice        # full mic loop (wake→listen→answer→speak)

Console text mode is the first thing to run: it exercises the whole aragon→alison
path (identity + gateway auth + brain streaming) without needing a mic.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import load_config
from .identity import IdentityResolver
from .obrenna_client import BrainAuthError, BrainUnreachable, ObrennaClient
from .pipeline import VoicePipeline


def _build(speak: bool) -> VoicePipeline:
    cfg = load_config()
    token = cfg.agent_token()
    if not token:
        print(
            "WARNING: no Obrenna gateway token found "
            f"(${cfg.agent_token_env} or {cfg.agent_token_file}). "
            "The gateway will reject the request.",
            file=sys.stderr,
        )
    client = ObrennaClient(cfg.brain, token)
    session_cookie = os.getenv("LEXICON_SESSION")  # optional per-device Lexicon session
    identity = IdentityResolver(cfg.identity, session_cookie=session_cookie)

    engines = None
    if speak:
        from .engines.registry import build_engines
        engines = build_engines(cfg.engines)
    return VoicePipeline(cfg, client, identity, engines=engines)


def _one(pipeline: VoicePipeline, text: str, speak: bool) -> None:
    printed: list[str] = []
    try:
        answer = pipeline.run_text_turn(
            text, speak=speak, on_sentence=(None if speak else printed.append)
        )
    except BrainAuthError as exc:
        print(f"[auth] {exc}", file=sys.stderr)
        return
    except BrainUnreachable as exc:
        print(f"[net] {exc}", file=sys.stderr)
        return
    if not speak:
        print(f"Lexi> {answer}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lexi")
    ap.add_argument("--say", help="one-shot: speak/print the answer to this text")
    ap.add_argument("--voice", action="store_true", help="full mic loop (needs audio)")
    ap.add_argument("--no-speak", action="store_true", help="print instead of TTS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    speak = not args.no_speak
    # Engines (and their heavy deps) are only built when we actually speak.
    pipeline = _build(speak=speak)

    if args.voice:  # pragma: no cover - hardware
        print("Voice mode. Ctrl-C to quit.")
        try:
            while True:
                pipeline.run_voice_turn(speak=True)
        except KeyboardInterrupt:
            return 0

    if args.say:
        _one(pipeline, args.say, speak=speak)
        return 0

    print("Text mode. Type a message; Ctrl-D to quit.")
    try:
        while True:
            line = input("you> ").strip()
            if line:
                _one(pipeline, line, speak=speak)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
