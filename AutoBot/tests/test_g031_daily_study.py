# -*- coding: utf-8 -*-
"""g031 daily study: day-level statistics, the equal-weight daily portfolio, and the fixed rules' masks."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data_simulation", "live_trading"):
    sys.path.insert(0, str(ROOT / sub))

import g031_daily_study as g031  # noqa: E402


class StatsTest(unittest.TestCase):
    def test_day_stats_use_days_as_the_unit(self):
        x = pd.Series([1.0, 3.0, -1.0, -1.0, 2.0])
        dates = pd.Series(["d1", "d1", "d2", "d2", "d3"])
        s = g031.day_stats(x, dates)
        self.assertEqual((s["n"], s["days"]), (5, 3))
        self.assertAlmostEqual(s["mean"], 0.8)
        self.assertAlmostEqual(s["day_mean"], 1.0)             # (2 + -1 + 2) / 3
        self.assertAlmostEqual(s["days_pos"], 2 / 3)

    def test_portfolio_is_the_equal_weight_mean_per_day(self):
        days = [f"d{i}" for i in range(6)]
        sel = pd.DataFrame({"date": days + days, "on": [1.0, -2.0, 1.0, 1.0, -1.0, 2.0] + [3.0, 0.0, 1.0, 1.0, -1.0, 0.0]})
        p = g031.portfolio(sel, "on")
        self.assertEqual((p["days"], p["n"]), (6, 12))
        self.assertAlmostEqual(p["total"], 2.0 - 1.0 + 1.0 + 1.0 - 1.0 + 1.0)
        self.assertAlmostEqual(p["worst"], -1.0)
        self.assertAlmostEqual(p["max_dd"], -1.0)              # 2, 1, 2, 3, 2, 3 -> deepest fall from a peak is 1
        self.assertEqual(g031.portfolio(sel.head(3), "on"), {"days": 3})

    def test_costs_and_rule_masks(self):
        self.assertAlmostEqual(g031.COST_PCT, 0.23)
        d = pd.DataFrame({"ret1": [4.0, 4.0, -4.0], "clv": [0.9, 0.5, 0.1], "vol_ratio": [2.5, 2.5, 1.0], "amt_rank": [10, 10, 10],
                          "ret20": [0.0, 0.0, 0.0], "hi20_dist": [0.0, 0.0, 0.0], "intra": [1.0, 1.0, -1.0], "rsi14": [50.0, 50.0, 20.0]})
        self.assertEqual(g031.RULES["strong_close_on"][2](d).tolist(), [True, False, False])
        self.assertEqual(g031.RULES["weak_close_on"][2](d).tolist(), [False, False, True])
        self.assertEqual(g031.RULES["any_on"][2](d).tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
