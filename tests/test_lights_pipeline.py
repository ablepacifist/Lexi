"""Lights inside the voice turn: ordering against media, music pause/resume, and the
spoken confirmation — driven through the real ``run_voice_turn`` with a fake mic."""
from types import SimpleNamespace

import pytest

from lexi import pipeline as pipemod
from lexi.config import LexiConfig
from lexi.lights.router import LightRouter
from lexi.pipeline import VoicePipeline
from conftest import make_inventory


class FakeDirect:
    def __init__(self, results=None):
        self.results, self.calls = results or {}, []

    def execute(self, ids, cmd):
        self.calls.append((list(ids), cmd.to_payload()))
        return {i: self.results.get(i) for i in ids}


class FakeMedia:
    enabled = True

    def __init__(self, playing=True):
        self.is_playing = playing
        self.events = []

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def stop(self):
        self.events.append("stop")


class NoBrain:
    def stream_turn(self, *a, **k):
        raise AssertionError("a light command must never reach the brain")


class FakeHistory:
    def __init__(self):
        self.rows = []

    def append(self, transcript, kind, detail=None, **_):
        self.rows.append((transcript, kind, detail))


class Ident:
    def account_id(self):
        return "acct"


def build(monkeypatch, said, *, direct=None, media=None, shared=True, history=None):
    """A pipeline whose 'microphone' returns whatever ``said[0]`` currently holds."""
    cfg = LexiConfig()
    cfg.engines.audio_shared = shared
    inv = make_inventory()
    direct = direct or FakeDirect()
    router = LightRouter(inv, direct=direct)
    engines = SimpleNamespace(
        stt=SimpleNamespace(transcribe=lambda pcm, sr: said[0]), tts=None, vad=None, wake=None)
    pipe = VoicePipeline(cfg, NoBrain(), Ident(), engines=engines, media=media,
                         history=history, lights=router)
    monkeypatch.setattr(pipemod.audio, "record_until_silence", lambda sr, vad: b"pcm")
    spoken = []
    monkeypatch.setattr(pipe, "speak_sentence", lambda s: spoken.append(s))
    return pipe, direct, spoken


def test_a_light_command_runs_speaks_and_never_touches_the_brain(monkeypatch):
    said = ["turn on the lamp"]
    pipe, direct, spoken = build(monkeypatch, said)
    assert pipe.run_voice_turn() == "Lamp on."
    assert direct.calls == [(["lamp"], {"power": True})]
    assert spoken == ["Lamp on."]


def test_lights_are_asked_before_media(monkeypatch):
    # "put on X" is also a media phrase; with lights configured it must be the lamp.
    said = ["put on the lamp"]
    media = FakeMedia(playing=False)
    pipe, direct, spoken = build(monkeypatch, said, media=media)
    monkeypatch.setattr(pipe, "_handle_media_intent",
                        lambda *a, **k: pytest.fail("media must not see a light command"))
    pipe.run_voice_turn()
    assert direct.calls == [(["lamp"], {"power": True})]


def test_media_commands_are_left_to_media(monkeypatch):
    said = ["play the live stream"]
    pipe, direct, _ = build(monkeypatch, said, media=FakeMedia(playing=False))
    handled = []
    monkeypatch.setattr(pipe, "_handle_media_intent",
                        lambda intent, speak: handled.append(intent) or "live stream")
    assert pipe.run_voice_turn() == "live stream"
    assert handled and direct.calls == []


def test_music_is_paused_for_the_command_then_resumed(monkeypatch):
    said = ["make the lamp blue"]
    media = FakeMedia(playing=True)
    pipe, direct, spoken = build(monkeypatch, said, media=media)
    pipe.run_voice_turn()
    assert media.events == ["pause", "resume"]        # the music carries on afterwards
    assert "stop" not in media.events
    assert spoken == ["Lamp set to blue."]


def test_on_an_exclusive_audio_device_the_lights_change_silently_and_music_resumes(monkeypatch):
    # audio_shared=False: a paused mpv would hold the device and mute TTS, so the
    # confirmation is skipped rather than the music being killed.
    said = ["turn off the lamp"]
    media = FakeMedia(playing=True)
    pipe, direct, spoken = build(monkeypatch, said, media=media, shared=False)
    pipe.run_voice_turn()
    assert direct.calls == [(["lamp"], {"power": False})]
    assert spoken == []
    assert media.events == ["pause", "resume"]


def test_with_no_music_playing_the_confirmation_is_spoken_even_when_exclusive(monkeypatch):
    said = ["turn on the lamp"]
    pipe, _, spoken = build(monkeypatch, said, media=FakeMedia(playing=False), shared=False)
    pipe.run_voice_turn()
    assert spoken == ["Lamp on."]


def test_it_follows_the_lights_used_a_moment_ago(monkeypatch):
    said = ["make the bedroom light blue"]
    pipe, direct, spoken = build(monkeypatch, said)
    pipe.run_voice_turn()
    said[0] = "make it red"
    pipe.run_voice_turn()
    assert [c[0] for c in direct.calls] == [["bedroom"], ["bedroom"]]   # not the lamp
    assert direct.calls[1][1]["hue"] == 0 and spoken[-1] == "Bedroom light set to red."


def test_it_without_prior_use_is_not_a_light_command(monkeypatch):
    said = ["make it blue"]
    pipe, direct, _ = build(monkeypatch, said)
    monkeypatch.setattr(pipe, "run_text_turn", lambda text, speak: "brain answered")
    assert pipe.run_voice_turn() == "brain answered"
    assert direct.calls == []


def test_a_dead_bulb_is_spoken_and_the_music_still_resumes(monkeypatch):
    said = ["turn on the lamp"]
    media = FakeMedia(playing=True)
    pipe, direct, spoken = build(monkeypatch, said, media=media,
                                 direct=FakeDirect({"lamp": "unreachable"}))
    pipe.run_voice_turn()
    assert spoken == ["I couldn't reach the lamp."]
    assert media.events == ["pause", "resume"]


def test_a_tts_failure_does_not_undo_or_raise(monkeypatch):
    said = ["turn on the lamp"]
    pipe, direct, _ = build(monkeypatch, said)

    def boom(_s):
        raise RuntimeError("audio device busy")
    monkeypatch.setattr(pipe, "speak_sentence", boom)
    assert pipe.run_voice_turn() == "Lamp on."
    assert direct.calls


def test_the_turn_is_logged_with_how_it_was_carried_out(monkeypatch):
    said = ["set both lights to 30 percent"]
    hist = FakeHistory()
    pipe, direct, _ = build(monkeypatch, said, history=hist)
    pipe.run_voice_turn()
    transcript, kind, detail = hist.rows[-1]
    assert kind == "lights" and transcript == "set both lights to 30 percent"
    assert detail["targets"] == ["lamp", "bedroom"]
    assert detail["command"] == {"brightness": 30}
    assert detail["via"] == {"lamp": "direct", "bedroom": "direct"}


def test_a_pipeline_with_no_lights_behaves_exactly_as_before(monkeypatch):
    cfg = LexiConfig()
    engines = SimpleNamespace(stt=SimpleNamespace(transcribe=lambda p, s: "turn on the lamp"),
                              tts=None, vad=None, wake=None)
    pipe = VoicePipeline(cfg, NoBrain(), Ident(), engines=engines)   # lights=None
    monkeypatch.setattr(pipemod.audio, "record_until_silence", lambda sr, vad: b"pcm")
    monkeypatch.setattr(pipe, "run_text_turn", lambda text, speak: "went to the brain")
    assert pipe.run_voice_turn() == "went to the brain"
