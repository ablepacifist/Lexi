"""Lexi — Obrenna's voice edge.

Runs on aragon (alongside Lexicon/Alchemy). Holds the mic/speaker and ALL the
voice logic + engines (wake word, VAD, STT, TTS). Only TEXT crosses the network:
Lexi transcribes locally, sends the transcript to Obrenna's brain on alison,
streams the answer tokens back, and speaks them locally.

Nothing here calls a third-party cloud service — every engine is local inference
on local model files. The only network calls are inside the homelab (the Obrenna
gateway on alison, and Lexicon auth on aragon).
"""

__version__ = "0.0.1"
