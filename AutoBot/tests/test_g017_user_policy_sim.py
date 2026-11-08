# -*- coding: utf-8 -*-
"""g017 user position-policy simulator + g018 sequencing (2026-10-31, incl. Codex review cases)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g017_user_policy_sim as g017  # noqa: E402
import g018_user_policy_tune as g018  # noqa: E402

# arm, t1, t2, stop, pyr, avgdn, tp1, tp2, tp3 (percent)
USER = np.array([0.8, 0.5, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0])
NO_ADD = np.array([0.8, 0.5, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 4.0])


def _bars(closes, obs=None, minute0=600):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c), np.minimum(o, c)
    ob = np.ones(len(c), bool) if obs is None else np.asarray(obs, bool)
    return o, h, l, c, ob, np.arange(len(c), dtype=np.int64) + minute0


def _sim(closes, prm=USER, **kw):
    o, h, l, c, ob, m = _bars(closes, **kw)
    return g017.simulate(o, h, l, c, ob, m, 0, prm, 300000.0, 200000.0)


class TestSimulate(unittest.TestCase):
    def test_quotes(self):
        self.assertEqual(g017._up(10060.05), 10070.0)
        self.assertEqual(g017._dn(9909.9), 9900.0)
        self.assertEqual(g017._up(4999.0), 5000.0)

    def test_rally_pyramids_then_ladder_exits(self):
        net, inv, reason, add, _ = _sim([10000, 10000] + list(np.linspace(10000, 10500, 30)))
        self.assertEqual((reason, add), (1, 1))  # +4% sells the rest, pyramid add happened
        self.assertGreater(net, 0)
        self.assertLessEqual(inv, 500000)

    def test_average_down_then_stop(self):
        net, inv, reason, add, _ = _sim([10000, 10000, 9950, 9900, 9850, 9800, 9750, 9700, 9650, 9600])
        self.assertEqual((reason, add), (3, 2))
        # entry 10,010 x29, add at 9,900 (limit, quote) x20 -> avg ~9,966; stop ~-2% -> loss ~-11k
        self.assertLess(net, -9000)
        self.assertGreater(net, -14000)

    def test_no_add_variant_stops_on_first_lot_only(self):
        net, inv, reason, add, _ = _sim([10000, 10000, 9950, 9900, 9850, 9800, 9750], prm=NO_ADD)
        self.assertEqual((reason, add), (3, 0))
        self.assertLess(inv, 300001)

    def test_trailing_after_rise(self):
        net, _, reason, add, _ = _sim([10000, 10000, 10050, 10100, 10150, 10200, 10150, 10100, 10050, 10000],
                                      prm=NO_ADD)
        self.assertEqual((reason, add), (2, 0))

    def test_synthetic_next_minute_is_not_a_fill(self):
        # bar 2 is synthetic: entry waits for bar 3 (open 10,000 -> close 10,300 path)
        r_obs = _sim([10000, 10000, 10000, 10300], obs=[1, 1, 0, 1])
        self.assertNotEqual(r_obs[2], 0)
        r_none = _sim([10000, 10000, 10000], obs=[1, 0, 0])
        self.assertEqual(r_none[2], 0)  # no traded minute after the signal -> no trade

    def test_eod_bar_runs_intrabar_first(self):
        o = np.array([10000, 10000, 10000.0])
        h = np.array([10000, 10000, 10600.0])
        c = l = o.copy()
        r = g017.simulate(o, h, l, c, np.ones(3, bool), np.array([913, 914, 915]), 0, USER, 300000.0, 200000.0)
        self.assertEqual(r[2], 1)  # take-profit ladder inside the 15:15 bar, not the EOD liquidation
        self.assertGreater(r[0], 0)

    def test_price_above_budget_is_skipped(self):
        p = np.array([400000.0, 400000.0])
        r = g017.simulate(p, p, p, p, np.ones(2, bool), np.array([600, 601]), 0, USER, 300000.0, 200000.0)
        self.assertEqual(r[2], 0)

    def test_cash_conservation_flat(self):
        # flat path to EOD: loss = entry slippage + exit slippage + fees/tax only
        net, inv, reason, _, _ = _sim([10000] * 6, prm=NO_ADD, minute0=910)
        self.assertEqual(reason, 4)
        qty = np.floor(300000 / 10010)
        expected = -10010 * qty * (1 + g017.FEE) + 9990 * qty * (1 - g017.FEE - g017.TAX)
        self.assertAlmostEqual(net, expected, places=6)


class TestSequence(unittest.TestCase):
    def test_combo_specific_exit_controls_reentry(self):
        df = pd.DataFrame({"date": ["d"] * 2, "code": ["A", "A"], "bar_i": [10, 20], "hold_ref": [5, 5],
                           "h0": [5, 5], "h1": [30, 30]})
        g018._ARR_CACHE.clear()
        mask = np.ones(2, bool)
        self.assertEqual(list(g018.sequence(df, mask, 10, 0)), [0, 1])   # exited at bar 15 -> re-entry at 20
        self.assertEqual(list(g018.sequence(df, mask, 10, 1)), [0])      # still holding until bar 40

    def test_daily_cap_and_new_day(self):
        df = pd.DataFrame({"date": ["d1", "d1", "d1", "d2"], "code": ["A", "B", "C", "A"],
                           "bar_i": [10, 11, 12, 10], "hold_ref": [1] * 4, "h0": [1] * 4})
        g018._ARR_CACHE.clear()
        self.assertEqual(list(g018.sequence(df, np.ones(4, bool), 2, 0)), [0, 1, 3])


if __name__ == "__main__":
    unittest.main()
