"""Shared fixtures."""
import pytest

from lexi.lights.model import LightInventory


def make_inventory(default_target: str = "all", **extra) -> LightInventory:
    """The user's real shape: two bulbs, 'lamp' and 'bedroom'. The first name is
    what Lexi says back, so the bedroom's is the friendlier 'bedroom light'."""
    return LightInventory.from_dict({
        "default_target": default_target,
        "light": [
            {"id": "lamp", "names": ["lamp", "the lamp"], "mac": "AA:BB:CC:DD:EE:01"},
            {"id": "bedroom", "names": ["bedroom light", "bedroom"],
             "mac": "aa-bb-cc-dd-ee-02"},
        ],
        **extra,
    })


@pytest.fixture
def inv() -> LightInventory:
    return make_inventory()
