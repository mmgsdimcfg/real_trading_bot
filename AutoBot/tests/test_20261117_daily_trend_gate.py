"""[2026-11-17] r005 _010 daily market trend gate (r001 ENABLE_MARKET_DAILY_TREND_GATE): index ETF daily close > MA60 and MA60 rising."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import r005_buy_conditions as r005  # noqa: E402


def _write(root: Path, code: str, closes: list[float], last_day: datetime) -> None:
    """daily_history/<code>.csv with one row per weekday ending at last_day (oldest first)."""
    days, d = [], last_day
    while len(days) < len(closes):
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    p = root / "daily_history"
    p.mkdir(parents=True, exist_ok=True)
    rows = ["date,open,high,low,close,volume,amount,source_date"]
    rows += [f"{x:%Y%m%d},{c},{c},{c},{c},1,1,{x:%Y%m%d}" for x, c in zip(days, closes)]
    (p / f"{code}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


UP = [100.0 + i for i in range(80)]                       # rising: close > MA60, MA60 rising
BELOW = [100.0 + i for i in range(70)] + [120.0] * 10     # last close under MA60
TODAY = datetime(2026, 11, 17, 9, 5)
YDAY = datetime(2026, 11, 16)


class DailyTrendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        r005._DAILY_TREND_CACHE.clear()
        self.p = mock.patch.object(r005, "RUNTIME_DATA_DIR", self.root)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()
        r005._DAILY_TREND_CACHE.clear()

    def _ctx(self):
        logs = []
        return SimpleNamespace(norm_code="159010", code="159010", symbol_label="159010_x", log=logs.append, current_dt=TODAY,
                               state=SimpleNamespace(buy_confirm_state={"159010": {"count": 1}}),
                               services=SimpleNamespace(market_regime=None)), logs

    def test_both_up_passes_and_one_down_blocks(self):
        _write(self.root, "069500", UP, YDAY)
        _write(self.root, "229200", UP, YDAY)
        ok, detail = r005.market_daily_trend(TODAY)
        self.assertTrue(ok, detail)
        r005._DAILY_TREND_CACHE.clear()
        _write(self.root, "229200", BELOW, YDAY)
        ok, detail = r005.market_daily_trend(TODAY)
        self.assertFalse(ok)
        self.assertIn("229200_CLOSE_BELOW", detail)
        self.assertNotIn("069500", detail)          # only the ETFs that failed are shown
        with mock.patch.object(r005, "MARKET_DAILY_TREND_SCOPE", "any"):
            r005._DAILY_TREND_CACHE.clear()
            self.assertTrue(r005.market_daily_trend(TODAY)[0])

    def test_ma_falling_today_row_ignored_and_unknown(self):
        falling = [200.0 - i for i in range(70)] + [140.0 + 2 * i for i in range(10)]   # close above MA60 but MA60 still falling
        _write(self.root, "069500", falling, YDAY)
        _write(self.root, "229200", UP, YDAY)
        ok, detail = r005.market_daily_trend(TODAY)
        self.assertFalse(ok)
        self.assertIn("069500_MA_FALLING", detail)
        # a row of today (provisional) is ignored: data ending today -> the last used close is yesterday's
        r005._DAILY_TREND_CACHE.clear()
        _write(self.root, "069500", UP, datetime(2026, 11, 17))
        self.assertIn("asof=20261116", r005.market_daily_trend(TODAY)[1])
        # missing file -> None; stale file -> None
        r005._DAILY_TREND_CACHE.clear()
        (self.root / "daily_history" / "069500.csv").unlink()
        ok, detail = r005.market_daily_trend(TODAY)
        self.assertIsNone(ok)
        self.assertIn("069500_NO_FILE", detail)
        r005._DAILY_TREND_CACHE.clear()
        _write(self.root, "069500", UP, datetime(2026, 11, 2))
        ok, detail = r005.market_daily_trend(TODAY)
        self.assertIsNone(ok)
        self.assertIn("069500_STALE_20261102", detail)
        # too few rows
        r005._DAILY_TREND_CACHE.clear()
        _write(self.root, "069500", UP[:50], YDAY)
        self.assertIn("FEW_ROWS_50", r005.market_daily_trend(TODAY)[1])

    def test_gate_in_010(self):
        _write(self.root, "069500", BELOW, YDAY)
        _write(self.root, "229200", UP, YDAY)
        with mock.patch.object(r005, "ENABLE_MARKET_DAILY_TREND_GATE", False):
            self.assertTrue(r005._010_market_regime(self._ctx()[0]))      # off: no change (1-minute gate has no service here)
        with mock.patch.object(r005, "ENABLE_MARKET_DAILY_TREND_GATE", True):
            ctx, logs = self._ctx()
            self.assertFalse(r005._010_market_regime(ctx))
            self.assertIn("MARKET_REGIME_DOWN_DAILY_MA60 | 069500_CLOSE_BELOW", logs[-1])
            self.assertNotIn("159010", ctx.state.buy_confirm_state)
            # unknown -> MARKET_DAILY_TREND_FAIL_OPEN
            (self.root / "daily_history" / "069500.csv").unlink()
            r005._DAILY_TREND_CACHE.clear()
            with mock.patch.object(r005, "MARKET_DAILY_TREND_FAIL_OPEN", True):
                self.assertTrue(r005._010_market_regime(self._ctx()[0]))
            with mock.patch.object(r005, "MARKET_DAILY_TREND_FAIL_OPEN", False):
                self.assertFalse(r005._010_market_regime(self._ctx()[0]))

    def test_cache_refresh(self):
        _write(self.root, "069500", BELOW, YDAY)
        _write(self.root, "229200", UP, YDAY)
        self.assertFalse(r005.market_daily_trend(TODAY)[0])
        _write(self.root, "069500", UP, YDAY)
        self.assertFalse(r005.market_daily_trend(TODAY + timedelta(minutes=10))[0])   # cached
        self.assertTrue(r005.market_daily_trend(TODAY + timedelta(minutes=31))[0])    # re-read after 30 minutes

    def test_shipped_off(self):
        import r001_define_config as r001
        self.assertIn("ENABLE_MARKET_DAILY_TREND_GATE", [c for c in r005._COND_010.related_defines])
        self.assertEqual(r001.MARKET_DAILY_TREND_MA, 60)


if __name__ == "__main__":
    unittest.main()
