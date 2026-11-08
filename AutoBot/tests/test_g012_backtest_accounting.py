# -*- coding: utf-8 -*-
"""g012 accounting / look-ahead tests (synthetic data, no runtime files)."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "tests"))

import g012_strategy_v2_backtest as g  # noqa: E402
from r008_strategy_v2 import DailyTrend, EntrySignal, V2Config  # noqa: E402
from test_r008_strategy_v2 import _m1_from_closes  # noqa: E402


class TestOpenFillNoLookAhead(unittest.TestCase):
    def test_same_bar_loss_elsewhere_does_not_block_open_fill(self):
        """X is stopped out inside bar ts with a loss beyond the daily limit; Y's signal from ts-1 must still
        fill at ts OPEN because that loss is only known after the open."""
        n = 80
        x_close = [10000.0] * n
        x_low = [9990.0] * n
        x_low[61] = 9000.0  # bar index 61: X crashes through its stop
        x = _m1_from_closes(x_close, lows=x_low)
        y = _m1_from_closes([20000.0] * n)
        bars = {"X": x, "Y": y}  # X iterated first on purpose
        universe = [DailyTrend("X", True, "OK", tier="mid"), DailyTrend("Y", True, "OK", tier="mid")]
        cfg = replace(V2Config(), daily_loss_limit_pct=0.3, slippage_ticks_entry=0, slippage_ticks_exit=0,
                      time_stop_min=0, trend_exit_on_5m_close=False, market_breadth_min_pct=0)
        t_x = x.index[30]
        t_y = x.index[60]

        def fake_step_entry(code, st, m1p, m5, cfg_, has_pos, market_ok=True, signal_gate=None):
            ts = m1p.index[-1]
            if code == "X" and ts == t_x and not has_pos:
                return EntrySignal("X", "PULLBACK", ts, 10000, 9900, 9950, 50, "t")
            if code == "Y" and ts == t_y and not has_pos:
                return EntrySignal("Y", "PULLBACK", ts, 20000, 19800, 19900, 100, "t")
            return None

        with mock.patch.object(g, "step_entry", side_effect=fake_step_entry):
            trades, diag = g.simulate_day("20261020", universe, bars, cfg, "A", None)
        codes = sorted(t.code for t in trades)
        self.assertEqual(codes, ["X", "Y"], diag)
        tx = next(t for t in trades if t.code == "X")
        self.assertIn("STOP_INITIAL", tx.exit_reasons)


class TestMetrics(unittest.TestCase):
    def test_mdd_counts_first_day_loss(self):
        tr = pd.DataFrame({"date": ["d1", "d2"], "net_pnl": [-100_000.0, 50_000.0], "r_net": [-1.0, 0.5],
                           "costs": [0.0, 0.0], "hold_min": [10.0, 10.0]})
        m = g.metrics(tr, ["d1", "d2"], 10_000_000)
        self.assertEqual(m["mdd_krw"], -100_000)
        self.assertEqual(m["profit_factor"], 0.5)

    def test_overrides(self):
        cfg = g.apply_overrides(V2Config(), ["time_stop_min=0", "trend_exit_on_5m_close=false", "trail_atr_mult=2"])
        self.assertEqual((cfg.time_stop_min, cfg.trend_exit_on_5m_close, cfg.trail_atr_mult), (0, False, 2.0))
        with self.assertRaises(SystemExit):
            g.apply_overrides(V2Config(), ["nope=1"])


if __name__ == "__main__":
    unittest.main()
