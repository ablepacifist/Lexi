"""One-time model fetch for Lexi (run at install, on the aragon device).

Downloads the STT / TTS / wake-word models into ``engines.models_dir`` so that
at runtime nothing is fetched over the network — the "fully offline" guarantee.
Run once with internet available; afterwards Lexi runs with the WAN blocked.

    python scripts/fetch_models.py

This is a scaffold: fill in the concrete downloads for the models you pin
(faster-whisper caches on first load; Piper voices + openWakeWord models are
plain files). Pin versions/hashes here so installs are reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lexi.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    models_dir = Path(cfg.engines.models_dir)
    (models_dir / "piper").mkdir(parents=True, exist_ok=True)
    print(f"models_dir: {models_dir.resolve()}")
    print("TODO: fetch and verify (hash-pinned):")
    print(f"  - faster-whisper '{cfg.engines.stt_model}' (int8) -> {models_dir}")
    print(f"  - Piper voice '{cfg.engines.tts_voice}' (.onnx + .json) -> {models_dir/'piper'}")
    print(f"  - openWakeWord '{cfg.engines.wake_model}' -> {models_dir}")
    print("Once fetched, Lexi runs with no runtime downloads (offline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
