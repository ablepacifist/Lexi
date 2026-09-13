"""One-time model fetch for Lexi (run at install, on the aragon device).

Downloads the STT / TTS / wake-word models so that at runtime nothing is fetched
over the network — the "fully offline" guarantee. Run once with internet
available; afterwards Lexi runs with the WAN blocked.

    python scripts/fetch_models.py            # fetch what config.toml asks for
    python scripts/fetch_models.py --force    # re-download even if present

Idempotent: everything already on disk is skipped, so it is safe to re-run.

Where each model lands, and why they are not all in one place:

  * faster-whisper  -> ``engines.models_dir``. Its loader takes a
    ``download_root``, so this one is fully ours.
  * Piper voice     -> ``engines.models_dir/piper``. Matches what
    ``PiperTts._resolve_voice_path`` looks for.
  * openWakeWord    -> the package's own ``resources/models`` directory inside
    the venv. ``Model(wakeword_models=["hey_jarvis"])`` resolves names against
    that directory, and the melspectrogram/embedding preprocessors must sit
    beside them. Relocating it would mean passing several explicit paths
    through ``wake.py`` for no real gain. **Consequence: recreating the venv
    means re-running this script.**
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lexi.config import load_config  # noqa: E402


def fetch_whisper(model: str, models_dir: Path, device: str, compute_type: str) -> None:
    """Pull the faster-whisper weights and prove they load.

    Constructing the model is the fetch: CTranslate2 downloads into
    ``download_root`` on first use and is a no-op afterwards. Loading it here
    rather than just checking for files means a corrupt or half-downloaded
    cache fails now, at install time, instead of mid-sentence later.
    """
    from faster_whisper import WhisperModel

    print(f"  faster-whisper '{model}' ({device}/{compute_type}) -> {models_dir}")
    WhisperModel(
        model,
        device=device,
        compute_type=compute_type,
        download_root=str(models_dir),
    )
    print("    ok (weights present and loadable)")


def fetch_piper(voice: str, piper_dir: Path, force: bool) -> None:
    """Pull the Piper voice: the .onnx **and** its .onnx.json.

    The JSON is not optional — it carries the sample rate and phoneme map, and
    ``PiperVoice.load`` looks for it at ``<model>.onnx.json``. A voice fetched
    without it fails at load with a confusing error.
    """
    from piper.download_voices import download_voice

    onnx = piper_dir / f"{voice}.onnx"
    meta = piper_dir / f"{voice}.onnx.json"
    if onnx.exists() and meta.exists() and not force:
        print(f"  Piper '{voice}' -> already present, skipped")
        return

    piper_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Piper '{voice}' -> {piper_dir}")
    download_voice(voice, piper_dir, force_redownload=force)

    missing = [p.name for p in (onnx, meta) if not p.exists()]
    if missing:
        raise RuntimeError(f"Piper voice '{voice}' incomplete, missing: {missing}")
    print(f"    ok ({onnx.name} + {meta.name})")


def fetch_wake(model: str) -> None:
    """Pull the wake-word model plus the shared preprocessors.

    ``download_models`` always fetches the melspectrogram, embedding and VAD
    models alongside the named wake word, and skips anything already on disk —
    so this needs no force flag of its own.
    """
    from openwakeword.utils import download_models

    print(f"  openWakeWord '{model}' -> package resources (inside the venv)")
    download_models(model_names=[model])
    print("    ok (wake word + melspectrogram/embedding/VAD preprocessors)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Lexi's local model files.")
    parser.add_argument(
        "--force", action="store_true", help="re-download even when already present"
    )
    args = parser.parse_args()

    cfg = load_config()
    models_dir = Path(cfg.engines.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"models_dir: {models_dir}")

    steps = (
        (
            "STT",
            lambda: fetch_whisper(
                cfg.engines.stt_model,
                models_dir,
                cfg.engines.stt_device,
                cfg.engines.stt_compute_type,
            ),
        ),
        ("TTS", lambda: fetch_piper(cfg.engines.tts_voice, models_dir / "piper", args.force)),
        ("wake", lambda: fetch_wake(cfg.engines.wake_model)),
    )

    failed: list[str] = []
    for label, step in steps:
        print(f"[{label}]")
        try:
            step()
        except Exception as exc:
            # Keep going: one unreachable host should not stop the other two,
            # and the summary below tells you exactly what to re-run.
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed.append(label)

    print()
    if failed:
        print(f"Incomplete — re-run after fixing: {', '.join(failed)}")
        return 1
    print("All models present. Lexi runs with no runtime downloads (offline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
