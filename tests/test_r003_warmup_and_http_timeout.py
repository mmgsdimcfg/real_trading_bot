# -*- coding: utf-8 -*-
"""2026-10-28: r002 calculate_indicators_with_warmup (prior-day warmup) and r003 KIS HTTP timeout shim."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r002_strategy_core_shared as r002  # noqa: E402


def _bars(start: str, n: int, base: float, step: float, freq: str = "3min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    close = base + step * np.arange(n) + np.sin(np.arange(n)) * 20
    return pd.DataFrame({"open": close - 10, "high": close + 30, "low": close - 30, "close": close,
                         "volume": 1000.0 + np.arange(n) * 10}, index=idx)


class TestWarmup(unittest.TestCase):
    def setUp(self):
        self.warm = _bars("2026-10-27 12:00", 60, 26000.0, 5.0)
        self.today = _bars("2026-10-28 09:03", 6, 26800.0, 20.0)

    def test_no_warmup_equals_plain(self):
        pd.testing.assert_frame_equal(r002.calculate_indicators_with_warmup(self.today, None),
                                      r002.calculate_indicators(self.today))
        pd.testing.assert_frame_equal(r002.calculate_indicators_with_warmup(self.today, self.warm.iloc[:0]),
                                      r002.calculate_indicators(self.today))

    def test_rows_are_today_only_and_match_full_history(self):
        out = r002.calculate_indicators_with_warmup(self.today, self.warm)
        self.assertListEqual(list(out.index), list(self.today.index))
        full = r002.calculate_indicators(pd.concat([self.warm, self.today]))
        for col in ("RSI", "ADX", "DI_PLUS", "BB_MIDDLE", "STOCH_K", "MACD"):
            np.testing.assert_allclose(out[col].to_numpy(), full[col].iloc[-6:].to_numpy(), err_msg=col)

    def test_session_columns_are_today_only(self):
        out = r002.calculate_indicators_with_warmup(self.today, self.warm)
        plain = r002.calculate_indicators(self.today)
        for col in r002.SESSION_ANCHORED_INDICATOR_COLUMNS:
            np.testing.assert_allclose(out[col].to_numpy(), plain[col].to_numpy(), err_msg=col)

    def test_warmup_rows_at_or_after_today_ignored(self):
        overlap = pd.concat([self.warm, self.today.iloc[:2]])
        out = r002.calculate_indicators_with_warmup(self.today, overlap)
        ref = r002.calculate_indicators_with_warmup(self.today, self.warm)
        pd.testing.assert_frame_equal(out, ref)

    def test_extra_columns_kept(self):
        today = self.today.assign(market="J")
        out = r002.calculate_indicators_with_warmup(today, self.warm)
        self.assertListEqual(list(out["market"]), ["J"] * 6)


class TestHttpTimeoutShim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import r003_trade_live_execute as r003
        cls.r003 = r003

    def _shim(self):
        calls = []
        real = SimpleNamespace(get=lambda url, **kw: calls.append(("get", kw)),
                               post=lambda url, *a, **kw: calls.append(("post", kw)), exceptions="EXC")
        return self.r003._KisRequestsWithTimeout(real), calls

    def test_defaults(self):
        shim, calls = self._shim()
        with mock.patch.object(self.r003, "KIS_HTTP_CONNECT_TIMEOUT_SECONDS", 3.0), \
                mock.patch.object(self.r003, "KIS_HTTP_READ_TIMEOUT_SECONDS", 10.0):
            shim.get("u", params={})
            shim.post("u", data="{}")
        self.assertEqual(calls[0][1]["timeout"], (3.0, 10.0))
        self.assertEqual(calls[1][1]["timeout"], (3.0, None))  # order POST: no read timeout
        self.assertEqual(shim.exceptions, "EXC")

    def test_explicit_timeout_and_disabled(self):
        shim, calls = self._shim()
        shim.get("u", timeout=1)
        with mock.patch.object(self.r003, "KIS_HTTP_CONNECT_TIMEOUT_SECONDS", 0.0), \
                mock.patch.object(self.r003, "KIS_HTTP_READ_TIMEOUT_SECONDS", 0.0):
            shim.get("u")
        self.assertEqual(calls[0][1]["timeout"], 1)
        self.assertIsNone(calls[1][1]["timeout"])

    def test_installed_on_kis_auth_only(self):
        import requests
        self.assertIsInstance(self.r003.ka.requests, self.r003._KisRequestsWithTimeout)
        self.assertFalse(isinstance(requests, self.r003._KisRequestsWithTimeout))



class TestPriorDayLoader(unittest.TestCase):
    """r012 load_prior_day_warmup on a synthetic g001-style 10 s file."""

    def setUp(self):
        import tempfile
        import r012_prior_day_warmup as r012
        self.r012 = r012
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        day = self.root / "20261027"
        day.mkdir()
        idx = pd.date_range("2026-10-27 14:00:00", "2026-10-27 15:19:50", freq="10s")
        n = len(idx)
        close = 26000.0 + np.arange(n) * 0.5
        raw_bar = (idx.second == 0).astype(int)
        raw_bar[(idx.hour == 14) & (idx.minute == 31) & (idx.second == 0)] = 0  # no-trade minute 14:31
        df = pd.DataFrame({"datetime": idx, "open": close, "high": close + 5, "low": close - 5, "close": close,
                           "volume": 10.0, "raw_bar": raw_bar})
        last = pd.DataFrame({"datetime": [pd.Timestamp("2026-10-27 15:30:00")], "open": [26300.0], "high": [26300.0],
                             "low": [26300.0], "close": [26300.0], "volume": [600.0], "raw_bar": [1]})
        pd.concat([df, last]).to_csv(day / "091580_상신이디피_10s.txt", index=False)

    def _load(self, base=26300.0, minutes=180):
        return self.r012.load_prior_day_warmup(self.root, "20261028", "091580", minutes, 7, base, 0.3)

    def test_loads_and_keeps_bucket_with_no_trade_minute(self):
        m1, m3, why = self._load()
        self.assertIsNotNone(m1, why)
        self.assertIn("prior_day=20261027", why)
        self.assertEqual(m1.index[-1], pd.Timestamp("2026-10-27 15:31"))  # 15:30 single-price bar, END label
        self.assertNotIn(pd.Timestamp("2026-10-27 14:32"), m1.index)       # no-trade minute dropped from 1-min
        self.assertIn(pd.Timestamp("2026-10-27 14:33"), m3.index)          # but its 3-min bucket kept (slot count)
        self.assertNotIn(pd.Timestamp("2026-10-27 15:33"), m3.index)       # single-price bar alone is not a 3-min bar
        self.assertAlmostEqual(float(m1["volume"].iloc[0]), 60.0)          # 10 s volume x 6 restored

    def test_base_price_mismatch_or_missing_disables(self):
        m1, m3, why = self._load(base=13150.0)  # 2:1 split
        self.assertIsNone(m1)
        self.assertIn("price basis changed", why)
        m1, m3, why = self._load(base=None)
        self.assertIsNone(m1)
        self.assertIn("base_price_unavailable", why)

    def test_first_partial_bucket_dropped(self):
        m1, m3, _ = self._load(minutes=10)  # 1-min 15:11..15:31 (end labels) minus gaps
        self.assertGreaterEqual(m3.index[0] - pd.Timedelta(minutes=2), m1.index[0])

    def test_not_found(self):
        m1, m3, why = self.r012.load_prior_day_warmup(self.root, "20261028", "000001", 180, 7, 1000.0, 0.3)
        self.assertIsNone(m1)
        self.assertIn("not found", why)


class TestImportSideEffects(unittest.TestCase):
    def test_r003_does_not_reset_root_handlers(self):
        # fresh interpreter: an earlier test may already have imported r003 in this process
        import subprocess
        code = (
            "import logging, sys; sys.path.insert(0, %r);"
            "h = logging.NullHandler(); logging.getLogger().addHandler(h);"
            "import r003_trade_live_execute;"
            "assert h in logging.getLogger().handlers, 'root handlers replaced';"
            "assert 'g003_trade_simulate_by_date' not in sys.modules, 'g003 imported'"
        ) % str(ROOT / "live_trading")
        res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT / "live_trading"), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])


class TestR003LoaderCacheRetry(unittest.TestCase):
    def setUp(self):
        import r003_trade_live_execute as r003
        self.r003 = r003
        self.addCleanup(r003._PRIOR_DAY_WARMUP_CACHE.clear)
        self.addCleanup(r003._PRIOR_DAY_WARMUP_RETRY_AT.clear)
        r003._PRIOR_DAY_WARMUP_CACHE.clear()
        r003._PRIOR_DAY_WARMUP_RETRY_AT.clear()

    def test_base_missing_retries_then_caches(self):
        from datetime import datetime, timedelta
        r003 = self.r003
        m1 = pd.DataFrame({"close": [1.0]})
        loader = mock.Mock(return_value=(m1, None, "ok"))
        base = mock.Mock(side_effect=[None, 26250.0])
        t0 = datetime(2026, 10, 28, 9, 0, 5)
        with mock.patch.object(r003, "fetch_base_price", base), mock.patch.object(r003, "load_prior_day_warmup", loader), \
                mock.patch.object(r003, "log", lambda m: None), mock.patch.object(r003, "ENABLE_PRIOR_DAY_INDICATOR_WARMUP", True):
            self.assertEqual(r003._load_prior_day_warmup("091580", t0), (None, None))
            self.assertEqual(r003._load_prior_day_warmup("091580", t0 + timedelta(seconds=30)), (None, None))
            self.assertEqual(base.call_count, 1)  # waiting for retry window
            w = r003._load_prior_day_warmup("091580", t0 + timedelta(seconds=61))
            self.assertIs(w[0], m1)
            r003._load_prior_day_warmup("091580", t0 + timedelta(seconds=90))
            self.assertEqual(base.call_count, 2)  # cached after decision

    def test_fetch_base_price_ignores_prev_close_fields(self):
        r003 = self.r003
        df = pd.DataFrame([{"stck_sdpr": "", "bfdy_clpr": "26250", "prdy_clpr": "26250"}])
        with mock.patch.object(r003.dsf, "inquire_price", lambda **kw: df):
            self.assertIsNone(r003.fetch_base_price("091580", None))
        df2 = pd.DataFrame([{"stck_sdpr": "13125"}])
        with mock.patch.object(r003.dsf, "inquire_price", lambda **kw: df2):
            self.assertEqual(r003.fetch_base_price("091580", None), 13125.0)


class TestTokenPostTimeout(unittest.TestCase):
    def test_oauth_post_gets_read_timeout(self):
        import r003_trade_live_execute as r003
        calls = []
        real = SimpleNamespace(post=lambda url, *a, **kw: calls.append(kw))
        shim = r003._KisRequestsWithTimeout(real)
        with mock.patch.object(r003, "KIS_HTTP_CONNECT_TIMEOUT_SECONDS", 3.0), \
                mock.patch.object(r003, "KIS_HTTP_READ_TIMEOUT_SECONDS", 10.0):
            shim.post("https://x:9443/oauth2/tokenP", data="{}")
            shim.post("https://x:9443/uapi/domestic-stock/v1/trading/order-cash", data="{}")
        self.assertEqual(calls[0]["timeout"], (3.0, 10.0))
        self.assertEqual(calls[1]["timeout"], (3.0, None))


if __name__ == "__main__":
    unittest.main()
