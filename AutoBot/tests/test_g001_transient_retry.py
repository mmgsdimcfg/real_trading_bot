# -*- coding: utf-8 -*-
"""g001 KIS HTTP proxy (2026-11-18): OPSQ0008 (HTTP 200, rt_cd=1) on GET is retried; a long run of exhausted retries stops the
collection (KIS outage) instead of failing every remaining symbol."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import g001_data_collect_symbols_daily as g001  # noqa: E402

ERR = b'{"rt_cd":"1","msg_cd":"OPSQ0008","msg1":"\xed\x98\xb8\xec\xb6\x9c \xed\x9b\x84\xec\xb2\x98\xeb\xa6\xac(MCI\xec\xa0\x84\xec\x86\xa1) \xec\x98\xa4\xeb\xa5\x98 \xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4."}'
OK = b'{"rt_cd":"0","msg_cd":"MCA00000","output2":[]}'


class Res:
    def __init__(self, body: bytes, status=200):
        self.content = body
        self.status_code = status
        self.text = body.decode("utf-8")


class FakeRequests:
    class exceptions:
        ConnectionError = ConnectionError
        Timeout = TimeoutError

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        outer = self

        class Session:
            def request(self, method, url, **kw):
                outer.calls += 1
                return outer.answers.pop(0) if outer.answers else Res(OK)

        self.Session = Session


class TransientRetryTest(unittest.TestCase):
    def setUp(self):
        g001.STOP_EVENT.clear()
        g001.KIS_UNAVAILABLE.clear()
        self.p = [mock.patch.object(g001, "KIS_TRANSIENT_BACKOFF", 0.0), mock.patch.object(g001, "KIS_MIN_REQUEST_INTERVAL", 0.0)]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        g001.STOP_EVENT.clear()
        g001.KIS_UNAVAILABLE.clear()

    def test_detects_only_small_200_bodies_with_the_code(self):
        self.assertEqual(g001._transient_kis_error(Res(ERR)), b"OPSQ0008")
        self.assertIsNone(g001._transient_kis_error(Res(OK)))
        self.assertIsNone(g001._transient_kis_error(Res(ERR, status=500)))
        self.assertIsNone(g001._transient_kis_error(Res(OK + b" " * 5000 + b"OPSQ0008")))

    def test_get_is_retried_until_success(self):
        fake = FakeRequests([Res(ERR), Res(ERR), Res(OK)])
        proxy = g001._KisRequestsProxy(fake)
        res = proxy.get("https://x/uapi/quotations")
        self.assertEqual(res.content, OK)
        self.assertEqual(fake.calls, 3)
        self.assertFalse(g001.KIS_UNAVAILABLE.is_set())

    def test_post_is_not_retried(self):
        fake = FakeRequests([Res(ERR)])
        proxy = g001._KisRequestsProxy(fake)
        self.assertEqual(proxy.post("https://x/order").content, ERR)
        self.assertEqual(fake.calls, 1)

    def test_retries_exhausted_returns_error_and_long_streak_stops_collection(self):
        n = g001.KIS_TRANSIENT_RETRIES + 1
        with mock.patch.object(g001, "KIS_TRANSIENT_ABORT_STREAK", 2):
            fake = FakeRequests([Res(ERR)] * n + [Res(OK)] + [Res(ERR)] * n * 2)
            proxy = g001._KisRequestsProxy(fake)
            self.assertEqual(proxy.get("u").content, ERR)        # streak 1
            self.assertEqual(proxy.get("u").content, OK)         # success resets the streak
            self.assertFalse(g001.KIS_UNAVAILABLE.is_set())
            proxy.get("u")                                       # streak 1
            self.assertFalse(g001.STOP_EVENT.is_set())
            proxy.get("u")                                       # streak 2 -> stop
            self.assertTrue(g001.KIS_UNAVAILABLE.is_set())
            self.assertTrue(g001.STOP_EVENT.is_set())
            with self.assertRaises(g001.CollectionStopped):
                proxy.get("u")


if __name__ == "__main__":
    unittest.main()
