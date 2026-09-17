"""Tiny local intent matcher for device commands Lexi handles itself.

Deliberately small and literal: only the phrasings that should NEVER round-trip
to the brain (media control) match here — everything else falls through to the
LLM. This is what lets music work even when alison's brain is offline.

Returns one of:
  {"action": "stop"}
  {"action": "next"}
  {"action": "play", "query": <track>}
  {"action": "play_playlist", "query": <name>|None, "shuffle": bool, "all": bool}
"""
from __future__ import annotations

import re

# Optional leading wake word ("hey jarvis ...") in case a bit of it lands in the
# transcript, plus a polite prefix ("can you", "please").
_LEAD = r"^\s*(?:hey\s+\w+[,.\s]+)?(?:can you\s+|could you\s+|please\s+)?"
_TRAIL = r"[.?!]*\s*$"

_STOP = re.compile(
    _LEAD + r"(?:stop|pause)(?:\s+(?:the\s+)?(?:music|song|playback|audio|it))?" + _TRAIL, re.I
)
_NEXT = re.compile(_LEAD + r"(?:next|skip)(?:\s+(?:song|track))?" + _TRAIL, re.I)

# "shuffle [the] [playlist] X"  |  "shuffle my music / everything / all"
_SHUFFLE = re.compile(_LEAD + r"shuffle\s+(?:the\s+)?(?:playlist\s+)?(.+?)" + _TRAIL, re.I)
# "play [the] playlist X [on shuffle|shuffled]"
_PLAYLIST = re.compile(
    _LEAD + r"(?:play|put on|start)\s+(?:the\s+)?playlist\s+(.+?)(\s+(?:on\s+shuffle|shuffled))?" + _TRAIL,
    re.I,
)
# "play X [on shuffle|shuffled]"  (single track, unless the shuffle suffix is present)
_PLAY = re.compile(
    _LEAD + r"(?:play|put on|start playing|play some|play me|play)\s+(.+?)(\s+(?:on\s+shuffle|shuffled))?" + _TRAIL,
    re.I,
)

_ALL_MUSIC = {"my music", "everything", "all", "music", "all music", "my library", "all my music"}


def match_media_intent(text: str) -> dict | None:
    if _STOP.match(text):
        return {"action": "stop"}
    if _NEXT.match(text):
        return {"action": "next"}

    m = _SHUFFLE.match(text)
    if m:
        target = m.group(1).strip()
        is_all = target.lower() in _ALL_MUSIC
        return {"action": "play_playlist", "query": None if is_all else target,
                "shuffle": True, "all": is_all}

    m = _PLAYLIST.match(text)
    if m:
        return {"action": "play_playlist", "query": m.group(1).strip(),
                "shuffle": bool(m.group(2)), "all": False}

    m = _PLAY.match(text)
    if m:
        query = m.group(1).strip()
        if not query:
            return None
        # "play X on shuffle" (no "playlist" keyword) → treat X as a collection.
        if m.group(2):
            return {"action": "play_playlist", "query": query, "shuffle": True, "all": False}
        return {"action": "play", "query": query}

    return None
