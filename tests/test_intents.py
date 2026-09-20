"""Grammar tests for the local media intent matcher (no hardware, no network)."""
from lexi.intents import match_media_intent


def test_stop_and_skip():
    assert match_media_intent("stop")["action"] == "stop"
    assert match_media_intent("stop the music")["action"] == "stop"
    assert match_media_intent("skip")["action"] == "next"
    assert match_media_intent("next song")["action"] == "next"
    assert match_media_intent("hey jarvis, skip this track")["action"] == "next"


def test_filler_prefixes_still_match():
    # Filler words must not push a command to the brain (which is usually off).
    # "Just stop." previously fell through → brain call → crash when offline.
    assert match_media_intent("just stop")["action"] == "stop"
    assert match_media_intent("Just stop.")["action"] == "stop"
    assert match_media_intent("okay skip")["action"] == "next"
    assert match_media_intent("yeah next song")["action"] == "next"
    assert match_media_intent("just play the live stream") == {"action": "livestream"}


def test_live_stream_beats_play():
    # "play the live stream" must be the livestream action, NOT a track called
    # "the live stream" — the reason _LIVESTREAM is tested before _PLAY.
    for phrase in (
        "play the live stream",
        "play livestream",
        "put on the radio",
        "join the live stream",
        "tune in to the station",
    ):
        assert match_media_intent(phrase) == {"action": "livestream"}, phrase


def test_play_single_track_still_works():
    got = match_media_intent("play dragon rider")
    assert got == {"action": "play", "query": "dragon rider"}


def test_shuffle_and_playlists():
    assert match_media_intent("shuffle all my music") == {
        "action": "play_playlist", "query": None, "shuffle": True, "all": True,
    }
    got = match_media_intent("shuffle Stellaris")
    assert got["action"] == "play_playlist" and got["shuffle"] is True
    assert got["query"] == "Stellaris" and got["all"] is False
    pl = match_media_intent("play the playlist Bane on shuffle")
    assert pl["action"] == "play_playlist" and pl["shuffle"] is True
    assert pl["query"] == "Bane"


def test_non_media_falls_through():
    assert match_media_intent("what's the weather in Rosemount") is None
    assert match_media_intent("who won the game last night") is None
