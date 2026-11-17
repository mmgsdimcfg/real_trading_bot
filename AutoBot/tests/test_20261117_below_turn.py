# -*- coding: utf-8 -*-
"""[2026-11-17] triple trigger: 1-minute MA5 <= MA20 passes when MA5 turned up and MACD rises (r001 TRIPLE_1MIN_BELOW_TURN_PASS).

r002 evaluate_1min_below_turn / evaluate_triple_trigger(below_turn=...), r005 _008 wiring incl. the BB-mid live-guard exemption
(TRIPLE_BELOW_TURN_REQUIRE_ABOVE_MID), g003 parity of the same branch."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "tests"))

import r001_define_config as r001  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
from r013_tick_trigger import TickCross  # noqa: E402
import test_r005_triple_cross as base  # noqa: E402
from test_r005_triple_cross import _sk_frame, _tick  # noqa: E402

NOT_ABOVE = {"ok": False, "reason": "1MIN_MA5_MA20_NOT_ABOVE", "bars_since_cross": None, "cross_bar_time": None}


def _frame(ma5, macd):
    idx = pd.date_range("2026-11-17 10:00", periods=len(ma5), freq="1min")
    return pd.DataFrame({"close": [100.0] * len(ma5), "MA_5": ma5, "MACD": macd}, index=idx, dtype=float)


def _sk_with_macd(last_label: str, rising: bool = True) -> pd.DataFrame:
    df = _sk_frame(last_label).copy()
    step = 1.0 if rising else -1.0
    df["MACD"] = [-50.0 + step * i for i in range(len(df))]
    return df


class TestEvaluate1minBelowTurn(unittest.TestCase):
    def test_turned_up_and_macd_up(self):
        out = r002.evaluate_1min_below_turn(_frame([10, 9, 8, 8.5, 9], [-3, -2.5, -2, -1.5, -1]), 5, 3)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["reason"], "1MIN_BELOW_TURN_MA5_TURN_UP_MACD_UP")

    def test_not_rising(self):
        out = r002.evaluate_1min_below_turn(_frame([10, 9, 8, 8.5, 8.5], [-3, -2.5, -2, -1.5, -1]), 5, 3)
        self.assertEqual((out["ok"], out["reason"]), (False, "1MIN_BELOW_TURN_MA5_NOT_RISING"))

    def test_rising_for_long_is_not_a_turn(self):
        out = r002.evaluate_1min_below_turn(_frame([5, 6, 7, 8, 9], [-3, -2.5, -2, -1.5, -1]), 5, 3)
        self.assertEqual((out["ok"], out["reason"]), (False, "1MIN_BELOW_TURN_MA5_NOT_TURNED"))
        # a dip just before the window counts only inside the lookback
        out = r002.evaluate_1min_below_turn(_frame([7, 6, 7, 8, 9, 10], [-3, -2.5, -2, -1.5, -1, -0.5]), 5, 3)
        self.assertEqual(out["reason"], "1MIN_BELOW_TURN_MA5_NOT_TURNED")
        out = r002.evaluate_1min_below_turn(_frame([7, 6, 7, 8, 9, 10], [-3, -2.5, -2, -1.5, -1, -0.5]), 5, 4)
        self.assertTrue(out["ok"], out)

    def test_macd_not_rising(self):
        out = r002.evaluate_1min_below_turn(_frame([10, 9, 8, 8.5, 9], [-3, -2.5, -2, -1.5, -1.5]), 5, 3)
        self.assertEqual((out["ok"], out["reason"]), (False, "1MIN_BELOW_TURN_MA5_MACD_NOT_RISING"))

    def test_missing_and_short(self):
        f = _frame([10, 9, 8, 8.5, 9], [-3, -2.5, -2, -1.5, -1]).drop(columns="MACD")
        self.assertEqual(r002.evaluate_1min_below_turn(f, 5, 3)["reason"], "1MIN_BELOW_TURN_MA5_MACD_MISSING")
        self.assertEqual(r002.evaluate_1min_below_turn(_frame([8.5, 9], [-1.5, -1]), 5, 3)["reason"],
                         "1MIN_BELOW_TURN_MA5_INSUFFICIENT_BARS")
        out = r002.evaluate_1min_below_turn(_frame([float("nan"), 9, 8, 8.5, 9], [-3, -2.5, -2, -1.5, -1]), 5, 3)
        self.assertEqual((out["ok"], out["reason"]), (False, "1MIN_BELOW_TURN_MA5_MISSING"))
        self.assertFalse(r002.evaluate_1min_below_turn(None, 5, 3)["ok"])

    def test_rolling_fallback_without_ma_column(self):
        closes = [10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.5, 10.5]
        f = pd.DataFrame({"close": closes, "MACD": [0, 0, 0, 0, -1, -2, -1.8, -1.5, -1.0]},
                         index=pd.date_range("2026-11-17 10:00", periods=len(closes), freq="1min"))
        self.assertTrue(r002.evaluate_1min_below_turn(f, 5, 3)["ok"])


class TestEvaluateTripleTriggerBelowTurn(unittest.TestCase):
    TICKS = {60: TickCross(True, "x", 60)}

    def test_below_turn_ok_passes_above_mode(self):
        out = r002.evaluate_triple_trigger(NOT_ABOVE, float("nan"), self.TICKS, (60, 120), "above", "any",
                                           below_turn={"ok": True, "reason": "1MIN_BELOW_TURN_MA5_TURN_UP_MACD_UP"})
        self.assertTrue(out["ok"], out)
        self.assertTrue(out["below_turn"])
        self.assertEqual(out["bars_since_cross"], 0)
        self.assertTrue(out["label"].startswith("TRIPLE_CROSS_1MIN_MA5_BELOW_TURN_MA20+TICK60"), out["label"])

    def test_below_turn_failure_adds_its_reason(self):
        out = r002.evaluate_triple_trigger(NOT_ABOVE, float("nan"), self.TICKS, (60, 120), "above", "any",
                                           below_turn={"ok": False, "reason": "1MIN_BELOW_TURN_MA5_NOT_TURNED"})
        self.assertFalse(out["ok"])
        self.assertFalse(out["below_turn"])
        self.assertEqual(out["missing"], ["1MIN_MA5_MA20_NOT_ABOVE", "1MIN_BELOW_TURN_MA5_NOT_TURNED"])

    def test_without_below_turn_unchanged(self):
        out = r002.evaluate_triple_trigger(NOT_ABOVE, float("nan"), self.TICKS, (60, 120), "above", "any")
        self.assertEqual((out["ok"], out["missing"], out["below_turn"]), (False, ["1MIN_MA5_MA20_NOT_ABOVE"], False))

    def test_only_not_above_and_only_above_mode(self):
        ok_turn = {"ok": True, "reason": "x"}
        missing_ma = {"ok": False, "reason": "1MIN_MA5_MA20_MISSING", "bars_since_cross": None, "cross_bar_time": None}
        out = r002.evaluate_triple_trigger(missing_ma, float("nan"), self.TICKS, (60, 120), "above", "any", below_turn=ok_turn)
        self.assertFalse(out["ok"])
        out = r002.evaluate_triple_trigger(NOT_ABOVE, float("nan"), self.TICKS, (60, 120), "cross", "any", below_turn=ok_turn)
        self.assertFalse(out["ok"])
        self.assertFalse(out["below_turn"])

    def test_ticks_still_required(self):
        out = r002.evaluate_triple_trigger(NOT_ABOVE, float("nan"), {60: TickCross(False, "TICK60_MA5_NOT_ABOVE_MA20", 60)},
                                           (60,), "above", "any", below_turn={"ok": True, "reason": "x"})
        self.assertFalse(out["ok"])
        self.assertIn("TICK60_MA5_NOT_ABOVE_MA20", out["missing"])


class TestR005BelowTurn(unittest.TestCase):
    """326030 2026-10-30 09:18: 1-min MA5 76,360 < MA20 76,710, MA5 76,240 -> 76,200 -> 76,240 -> 76,260 -> 76,360 (turned up)."""

    # reuse the r005 harness of test_r005_triple_cross without re-running its tests
    _ctx = base.TestR005Triple._ctx
    setUp = base.TestR005Triple.setUp
    _run = base.TestR005Triple._run

    NOW = datetime(2026, 10, 30, 9, 18, 30)

    def _run_below(self, price, rising=True):
        logs, recs = [], []
        ctx = self._ctx(_sk_with_macd("2026-10-30 09:18", rising), self.NOW, price, lambda code, dt: _tick(self.NOW), logs, recs)
        return r005._009_hybrid_1min_trigger(ctx), ctx, logs, recs

    def test_defaults(self):
        self.assertTrue(r001.TRIPLE_1MIN_BELOW_TURN_PASS)
        self.assertEqual(r001.TRIPLE_1MIN_BELOW_TURN_LOOKBACK_BARS, 3)
        self.assertFalse(r001.TRIPLE_BELOW_TURN_REQUIRE_ABOVE_MID)

    def test_passes_below_mid_with_turn_and_macd(self):
        ok, ctx, logs, recs = self._run_below(76600.0)          # live price below the 1-min BB mid (76,710)
        self.assertTrue(ok, logs)
        self.assertTrue(ctx.trigger_reason.startswith("TRIPLE_CROSS_1MIN_MA5_BELOW_TURN_MA20+TICK60+TICK120"), ctx.trigger_reason)
        self.assertEqual(ctx.trigger_info["one_min"], "below_turn")
        self.assertEqual(recs[-1]["result"], "TRIPLE_CROSS_PASS")

    def test_macd_falling_rejects(self):
        ok, _, logs, _ = self._run_below(76600.0, rising=False)
        self.assertFalse(ok)
        self.assertIn("1MIN_MA5_MA20_NOT_ABOVE|1MIN_BELOW_TURN_MA5_MACD_NOT_RISING", logs[-1])

    def test_switch_off_rejects_as_before(self):
        with mock.patch.object(r005, "TRIPLE_1MIN_BELOW_TURN_PASS", False):
            ok, _, logs, _ = self._run_below(76600.0)
        self.assertFalse(ok)
        self.assertIn("1MIN_MA5_MA20_NOT_ABOVE", logs[-1])
        self.assertNotIn("BELOW_TURN", logs[-1])

    def test_mid_guard_kept_when_required(self):
        with mock.patch.object(r005, "TRIPLE_BELOW_TURN_REQUIRE_ABOVE_MID", True):
            ok, _, logs, _ = self._run_below(76600.0)
        self.assertFalse(ok)
        self.assertIn("TRIPLE_CROSS_1MIN_LIVE_BELOW_BB_MID", logs[-1])

    def test_live_drop_guard_still_applies(self):
        ok, _, logs, _ = self._run_below(76200.0)                # -0.52 % vs the 09:18 close 76,600
        self.assertFalse(ok)
        self.assertIn("TRIPLE_CROSS_1MIN_LIVE_DROP_", logs[-1])

    def test_above_path_label_unchanged(self):
        now = datetime(2026, 10, 30, 9, 22, 13)
        ok, ctx, logs, _ = self._run("2026-10-30 09:22", now, 77300.0, _tick(now))
        self.assertTrue(ok, logs)
        self.assertEqual(ctx.trigger_info["one_min"], "above")
        self.assertIn("1MIN_MA5_ABOVE_MA20", ctx.trigger_reason)


class TestLivePipelineIndicators(unittest.TestCase):
    def test_calculate_indicators_frame_has_the_columns_and_turns(self):
        # decline from 100 to 95, then three rising bars: MA5 < MA20, MA5 turned up, MACD rising
        closes = [100.0] * 30 + [99.5 - 0.5 * i for i in range(10)] + [95.5, 96.3, 97.2]
        df = pd.DataFrame({"open": closes, "high": [c + 0.2 for c in closes], "low": [c - 0.2 for c in closes],
                           "close": closes, "volume": [1000.0] * len(closes)},
                          index=pd.date_range("2026-11-17 09:00", periods=len(closes), freq="1min"))
        ind = r002.calculate_indicators(df)
        self.assertTrue({"MA_5", "MACD", "BB_MIDDLE"} <= set(ind.columns))
        self.assertEqual(r002.find_1min_ma_golden_cross(ind, 5, 20)["reason"], "1MIN_MA5_MA20_NOT_ABOVE")
        out = r002.evaluate_1min_below_turn(ind, 5, 3)
        self.assertTrue(out["ok"], out)

    def test_ma5_equal_ma20_counts_as_not_above(self):
        closes = [100.0] * 25
        df = pd.DataFrame({"close": closes}, index=pd.date_range("2026-11-17 09:00", periods=25, freq="1min"))
        self.assertEqual(r002.find_1min_ma_golden_cross(df, 5, 20)["reason"], "1MIN_MA5_MA20_NOT_ABOVE")


class TestG003Parity(unittest.TestCase):
    NOW = datetime(2026, 10, 30, 9, 18, 30)

    def setUp(self):
        try:
            sys.path.insert(0, str(ROOT / "data_simulation"))
            import g003_trade_simulate_by_date as g003
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"g003 import failed: {exc}")
        self.g003 = g003
        for p in (mock.patch.object(g003, "ENTRY_1MIN_FRAME_REQUIRE_LATEST", False),
                  mock.patch.object(g003, "run_3min_context_decision", lambda *a, **k: (True, "CTX"))):
            p.start()
        self.addCleanup(mock.patch.stopall)

    def _decide(self, price=76600.0, rising=True):
        feed = SimpleNamespace(evaluate=lambda code, dt: _tick(self.NOW))
        return self.g003._triple_entry_decision("326030", pd.Timestamp(self.NOW), price, pd.DataFrame(),
                                                 _sk_with_macd("2026-10-30 09:18", rising), {}, feed)

    def test_same_decisions_as_r005(self):
        ok, reason = self._decide()
        self.assertTrue(ok, reason)
        self.assertIn("1MIN_MA5_BELOW_TURN_MA20", reason)
        ok, reason = self._decide(rising=False)
        self.assertFalse(ok)
        self.assertIn("1MIN_MA5_MA20_NOT_ABOVE|1MIN_BELOW_TURN_MA5_MACD_NOT_RISING", reason)
        ok, reason = self._decide(price=76200.0)
        self.assertFalse(ok)
        self.assertIn("1MIN_LIVE_DROP_", reason)
        with mock.patch.object(self.g003, "TRIPLE_BELOW_TURN_REQUIRE_ABOVE_MID", True):
            ok, reason = self._decide()
        self.assertEqual((ok, reason), (False, "TRIPLE_CROSS_1MIN_LIVE_BELOW_BB_MID"))
        with mock.patch.object(self.g003, "TRIPLE_1MIN_BELOW_TURN_PASS", False):
            ok, reason = self._decide()
        self.assertFalse(ok)
        self.assertNotIn("BELOW_TURN", reason)


if __name__ == "__main__":
    unittest.main()
