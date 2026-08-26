"""Engine interfaces.

Everything the pipeline needs is expressed here so a different local
implementation (or a mock in tests) can be dropped in without touching the
pipeline. No implementation in this module talks to the network.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class SttEngine(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Mono int16 PCM → text."""
        ...


@runtime_checkable
class TtsEngine(Protocol):
    def synthesize(self, text: str) -> Iterator[bytes]:
        """Text → mono int16 PCM chunks (streamed)."""
        ...

    @property
    def sample_rate(self) -> int:
        ...


@runtime_checkable
class WakeWord(Protocol):
    def detect(self, frame: bytes) -> bool:
        """One audio frame → True when the wake word fires."""
        ...


@runtime_checkable
class Vad(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """One audio frame → True when it contains speech."""
        ...
