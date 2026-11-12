# -*- coding: utf-8 -*-
"""Edge AI "rise before the stop" label (g013 run_up / y_tp, 2026-11-12) and the trainer's target / absolute-expectancy
promotion rule (g014 --target, day_bootstrap_lcb, require_positive)."""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import g013_edge_ai_dataset as g013  # noqa: E402
import g014_edge_ai_train as g014  # noqa: E402

T = pd.Timestamp("10:00").time()


def run(o, h, l, times=None, tp=2.0, sl_pct=1.77, entry=10000.0):
    n = len(o)
    return g013.run_up(np.array(o, float), np.array(h, float), np.array(l, float), times or [T] * n, 0, entry,
                       entry * (1 - sl_pct / 100.0), tp)


class RunUpTest(unittest.TestCase):
    def test_target_reached_before_the_stop(self):
        out = run([10000, 10000, 10100, 10250], [10000, 10100, 10210, 10520], [10000, 9990, 10050, 10200])
        self.assertEqual((out["y_tp"], out["run_stopped"], out["run_tp_bars"]), (1, 0, 2))
        self.assertAlmostEqual(out["run_mfe_pct"], 5.2)            # keeps rising after the target: no take profit here

    def test_stop_first_then_a_rally_does_not_count(self):
        out = run([10000, 10000, 9900, 9900], [10000, 10050, 9950, 10600], [10000, 9990, 9800, 9890])
        self.assertEqual((out["y_tp"], out["run_stopped"], out["run_tp_bars"]), (0, 1, -1))
        self.assertAlmostEqual(out["run_mfe_pct"], 0.5)

    def test_the_bar_that_touches_the_stop_never_counts_its_high(self):
        # one bar spans both +2% and the stop: the order is unknown -> stopped, target not credited
        out = run([10000, 10000], [10000, 10300], [10000, 9800])
        self.assertEqual((out["y_tp"], out["run_stopped"]), (0, 1))
        self.assertEqual(out["run_mfe_pct"], 0.0)

    def test_gap_down_open_below_the_stop(self):
        out = run([10000, 10000, 9700], [10000, 10050, 10400], [10000, 9990, 9690])
        self.assertEqual((out["y_tp"], out["run_stopped"]), (0, 1))

    def test_stop_after_the_target_keeps_the_success(self):
        out = run([10000, 10000, 10150, 10000], [10000, 10100, 10220, 10010], [10000, 9990, 10100, 9700])
        self.assertEqual((out["y_tp"], out["run_stopped"], out["run_tp_bars"]), (1, 1, 2))

    def test_watching_ends_at_the_force_exit_time(self):
        late = pd.Timestamp("15:15").time()
        out = run([10000, 10000, 10050, 10050], [10000, 10050, 10100, 10900], [10000, 9990, 10000, 10040], times=[T, T, late, late])
        self.assertEqual(out["y_tp"], 0)                            # the +9% bar is after 15:15
        self.assertAlmostEqual(out["run_mfe_pct"], 1.0)

    def test_triple_barrier_carries_the_columns(self):
        args = argparse.Namespace(tp=2.0, sl=1.77, horizon=400, slip_ticks=0)
        o, h, l, c = [10000, 10000, 10100], [10000, 10100, 10350], [10000, 9990, 10050], [10000, 10100, 10300]
        out = g013.triple_barrier(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float), [T] * 3, 0, args)
        self.assertEqual((out["exit_reason"], out["y_tp"], out["run_stopped"]), ("tp", 1, 0))
        self.assertAlmostEqual(out["run_mfe_pct"], 3.5)


def oos(net_by_day: dict[str, list[float]], keep=True) -> pd.DataFrame:
    rows = [{"date": d, "net_ret": x, "keep": keep, "m3_ctx_pass": 0, "y": int(x > 0), "y_tp": int(x > 1), "p": 0.5 + x / 100,
             "run_mfe_pct": max(x, 0.0) + 0.3, "run_stopped": int(x < -1.5)} for d, xs in net_by_day.items() for x in xs]
    return pd.DataFrame(rows)


class TrainerTest(unittest.TestCase):
    def test_target_default_is_the_old_label_for_callers_without_the_option(self):
        self.assertEqual(g014.target_of(argparse.Namespace()), "y")
        self.assertEqual(g014.target_of(argparse.Namespace(target="y_tp")), "y_tp")

    def test_day_bootstrap_bound(self):
        good = oos({f"202611{d:02d}": [0.8, 1.2, 1.0] for d in range(1, 11)})
        self.assertGreater(g014.day_bootstrap_lcb(good), 0.7)
        mixed = oos({**{f"202611{d:02d}": [2.0, 2.0] for d in range(1, 6)}, **{f"202611{d:02d}": [-2.0, -2.0] for d in range(6, 11)}})
        self.assertLess(g014.day_bootstrap_lcb(mixed), 0)           # zero mean, wide between-day spread
        self.assertTrue(np.isnan(g014.day_bootstrap_lcb(oos({"20261101": [1.0]}))))   # too few days to say anything

    def test_run_stats(self):
        s = g014.run_stats(pd.DataFrame({"run_mfe_pct": [0.5, 2.0, 3.5, 6.0], "run_stopped": [1, 0, 0, 0]}))
        self.assertEqual((s["n"], s["tp2"], s["tp3"], s["tp5"], s["stopped"]), (4, 0.75, 0.5, 0.25, 0.25))
        self.assertIsNone(g014.run_stats(pd.DataFrame({"net_ret": [1.0]})))             # an old dataset

    def _promo(self, kept_days, require_positive):
        base = oos({f"202611{d:02d}": [-1.0] * 20 for d in range(1, 11)}, keep=False)
        df = pd.concat([base, oos(kept_days)], ignore_index=True)
        folds = [{"all_base": {"n": 10, "mean_net": -1.0}, "all_kept": {"n": 5, "mean_net": 0.0}}] * 3
        args = argparse.Namespace(target="y_tp", min_ctx_events=300, min_kept=10, min_uplift=0.05, min_fold_ratio=0.6,
                                  require_positive=require_positive)
        return g014.promotion(df, folds, args)

    def test_better_than_base_but_still_losing_is_not_promoted(self):
        losing = {f"202611{d:02d}": [-0.9, 0.2, -0.2] for d in range(1, 11)}        # +0.7%p uplift, some wins, still negative
        pm = self._promo(losing, require_positive=True)
        self.assertFalse(pm["promoted"])
        self.assertLess(pm["checks"]["kept_mean_net_lcb95"], 0)
        self.assertTrue(self._promo(losing, require_positive=False)["promoted"])     # the old relative rule alone

    def test_positive_expectancy_is_promoted_and_scope_is_all_events(self):
        pm = self._promo({f"202611{d:02d}": [0.6, 0.9, 0.7] for d in range(1, 11)}, require_positive=True)
        self.assertTrue(pm["promoted"])
        self.assertEqual((pm["scope"], pm["target"]), ("all", "y_tp"))
        self.assertGreater(pm["checks"]["kept_mean_net_lcb95"], 0)
        self.assertEqual(pm["run_kept"]["n"], 30)


if __name__ == "__main__":
    unittest.main()
