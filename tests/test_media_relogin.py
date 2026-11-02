"""Lexi logs in to Lexicon once and reuses the session cookie. Lexicon forgets every session when it restarts,
and now requires a login on its API, so a 401 must trigger one fresh login + retry instead of failing until
Lexi itself is restarted. No network: httpx is replaced by a scripted fake."""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

try:  # use the real package where it is installed (the Pi); otherwise a minimal stand-in
    import httpx  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - developer PC without Lexi's dependencies
    fake = types.ModuleType("httpx")
    fake.HTTPError = type("HTTPError", (Exception,), {})
    fake.Response = object
    fake.post = fake.get = fake.request = fake.stream = lambda *a, **k: None
    sys.modules["httpx"] = fake

from lexi import media  # noqa: E402
from lexi.config import IdentityConfig  # noqa: E402


class FakeResponse:
    def __init__(self, status: int, body=None, cookies=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.cookies = cookies or {}
        self.headers = mock.Mock(get_list=lambda name: [])

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise media.httpx.HTTPError(f"HTTP {self.status_code}")


def make_player() -> media.MediaPlayer:
    cfg = IdentityConfig()
    cfg.lexicon_username = "lexi"
    cfg.lexicon_password = "pw"
    return media.MediaPlayer(cfg)


class ReloginOn401(unittest.TestCase):
    def test_expired_session_is_replaced_and_the_call_retried_once(self):
        player = make_player()
        logins = []

        def fake_post(url, **kw):  # the login call
            logins.append(url)
            sid = f"SESSION{len(logins)}"
            return FakeResponse(200, {"success": True, "id": 7}, cookies={"JSESSIONID": sid})

        seen_cookies = []

        def fake_request(method, url, headers=None, **kw):
            seen_cookies.append(headers["Cookie"])
            if headers["Cookie"] == "JSESSIONID=SESSION1":  # Lexicon restarted: the first session is gone
                return FakeResponse(401)
            return FakeResponse(200, [{"id": 1, "title": "song"}])

        with mock.patch.object(media.httpx, "post", fake_post), \
                mock.patch.object(media.httpx, "request", fake_request):
            result = player.search("song")

        self.assertEqual(result, [{"id": 1, "title": "song"}])
        self.assertEqual(len(logins), 2, "one login at the start, one after the 401")
        self.assertEqual(seen_cookies, ["JSESSIONID=SESSION1", "JSESSIONID=SESSION2"])

    def test_a_healthy_session_is_reused_without_logging_in_again(self):
        player = make_player()
        logins = []

        def fake_post(url, **kw):
            logins.append(url)
            return FakeResponse(200, {"success": True, "id": 7}, cookies={"JSESSIONID": "S"})

        with mock.patch.object(media.httpx, "post", fake_post), \
                mock.patch.object(media.httpx, "request", lambda *a, **k: FakeResponse(200, [])):
            player.search("a")
            player.search("b")

        self.assertEqual(len(logins), 1)

    def test_only_one_retry_so_a_rejected_login_cannot_loop(self):
        player = make_player()
        calls = []

        def fake_post(url, **kw):
            return FakeResponse(200, {"success": True, "id": 7}, cookies={"JSESSIONID": "S"})

        def fake_request(method, url, **kw):
            calls.append(url)
            return FakeResponse(401)

        with mock.patch.object(media.httpx, "post", fake_post), \
                mock.patch.object(media.httpx, "request", fake_request):
            with self.assertRaises(media.MediaError):
                player.search("x")

        self.assertEqual(len(calls), 2, "the original attempt plus exactly one retry")


if __name__ == "__main__":
    unittest.main()
