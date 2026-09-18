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

from .config import LexiConfig, load_config
from .identity import IdentityResolver
from .obrenna_client import BrainAuthError, BrainUnreachable, ObrennaClient
from .pipeline import VoicePipeline


def _build(speak: bool, *, need_engines: bool | None = None) -> tuple[VoicePipeline, LexiConfig]:
    """Build the pipeline and hand back the config that produced it.

    ``need_engines`` defaults to ``speak``: engines (and their heavy deps) are
    only constructed when something will actually use them. Voice mode needs
    them even with --no-speak, because the mic and STT live there too.
    """
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

    from .media import MediaPlayer
    media = MediaPlayer(cfg.identity, cfg.engines.mpv_audio_device)  # lazy login

    from .history import ConversationLog
    history = ConversationLog(cfg.history.path, cfg.history.ttl_days, cfg.history.enabled)

    engines = None
    if need_engines if need_engines is not None else speak:
        from .engines.registry import build_engines
        engines = build_engines(cfg.engines)
    return VoicePipeline(cfg, client, identity, engines=engines, media=media, history=history), cfg


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


def _claim_single_instance() -> None:
    """Ensure only one --voice instance holds the mic.

    An orphaned --voice process (SSH session closed without it exiting) keeps the
    single-capture USB mic busy, so the next run can't open the mic — or worse,
    the stale one keeps answering with old code. On start we kill a stale instance
    recorded in the pidfile, then record our own PID. Best-effort: a lock hiccup
    must never stop the assistant from starting.
    """
    import atexit
    import os
    import signal
    import time
    from pathlib import Path

    pidfile = Path(".voice.pid")  # cwd is the Lexi dir
    try:
        if pidfile.exists():
            old = int(pidfile.read_text().strip() or "0")
            if old and old != os.getpid():
                alive = True
                try:
                    os.kill(old, 0)
                except OSError:
                    alive = False
                cmd = ""
                try:
                    cmd = Path(f"/proc/{old}/cmdline").read_text(errors="replace")
                except OSError:
                    cmd = ""
                if alive and "lexi" in cmd:
                    print(f"Stopping stale voice instance (pid {old})...", file=sys.stderr)
                    try:
                        os.kill(old, signal.SIGTERM)
                        for _ in range(20):
                            time.sleep(0.1)
                            try:
                                os.kill(old, 0)
                            except OSError:
                                break
                        else:
                            os.kill(old, signal.SIGKILL)
                    except OSError:
                        pass
                    time.sleep(0.5)  # let ALSA release the mic
        pidfile.write_text(str(os.getpid()))
        atexit.register(lambda: pidfile.unlink(missing_ok=True))
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage (e.g. cp1252) that can't
    # print most of what an LLM reply might contain (emoji, smart quotes, …).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Audio (PipeWire, used by both mpv and TTS on the Pi) is reached through the
    # per-user socket under XDG_RUNTIME_DIR. Interactive shells set it, but a lean
    # launcher (cron/systemd) may not — default it so audio never silently fails.
    if os.name == "posix" and not os.environ.get("XDG_RUNTIME_DIR"):
        candidate = f"/run/user/{os.getuid()}"
        if os.path.isdir(candidate):
            os.environ["XDG_RUNTIME_DIR"] = candidate

    ap = argparse.ArgumentParser(prog="lexi")
    ap.add_argument("--say", help="one-shot: speak/print the answer to this text")
    ap.add_argument("--voice", action="store_true", help="full mic loop (needs audio)")
    ap.add_argument("--no-speak", action="store_true", help="print instead of TTS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    speak = not args.no_speak
    # Voice mode needs the mic and STT even when it is not speaking back.
    pipeline, cfg = _build(speak=speak, need_engines=speak or args.voice)

    if args.voice:  # pragma: no cover - hardware
        _claim_single_instance()  # kill any orphaned --voice holding the mic
        # Wake-gated when a wake word is configured, which is the normal case:
        # listen_loop waits for the wake word and only then records a turn.
        # Without this, --voice recorded and transcribed continuously — the
        # wake word was built but nothing ever called it.
        if cfg.engines.wake_enabled:
            print(f"Voice mode. Say the wake word ({cfg.engines.wake_model}). Ctrl-C to quit.")
        else:
            print("Voice mode, wake word DISABLED — recording every turn. Ctrl-C to quit.")
        try:
            if cfg.engines.wake_enabled:
                pipeline.listen_loop()
            else:
                while True:
                    pipeline.run_voice_turn(speak=speak)
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
