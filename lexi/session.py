"""Per-conversation voice session state.

Holds the state machine and the chat_id that ties a spoken conversation to one
Obrenna chat (so memory/recency accrue across turns) and a cancel token used for
barge-in (a later refinement: new speech during SPEAKING aborts TTS + the turn).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class CancelToken:
    """Cooperative cancellation for barge-in."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class VoiceSession:
    account_id: str
    chat_id: str | None = None
    state: State = State.IDLE
    cancel: CancelToken = field(default_factory=CancelToken)

    def set_state(self, state: State) -> None:
        self.state = state
