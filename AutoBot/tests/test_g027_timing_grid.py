# -*- coding: utf-8 -*-
"""g027 exit simulator on hand-made 1-min paths (2026-11-11)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g027_timing_grid as g027  # noqa: E402

ENTRY = 10000.0     # tick 10 (0.1%)
COST = (g027.FEE * 2 + g027.TAX) * 100.0


def path(bars, start_minute=600):
    """bars: (open, high, low, close) -> PATH_COLS rows (ma5 > ma20 = uptrend on every bar)."""
    rows = [(start_minute + j, o, h, l, c, c, c - 50.0, c) for j, (o, h, l, c) in enumerate(bars)]
    return np.array(rows, dtype=np.float64)


def run(bars, prm, atr=np.nan, start_minute=600, flip=False):
    P = path(bars, start_minute)
    prm = tuple(prm) + (0,) * (len(g027.PRM_COLS) - len(prm))      # cap = 0 unless given
    return g027._sim_one(P, 0, len(P), ENTRY, atr, np.array(prm, dtype=np.float64), flip)


def net_at(px):
    return (px * (1 - g027.FEE - g027.TAX) - ENTRY * (1 + g027.FEE)) / ENTRY * 100.0


FLAT = (10000, 10010, 9990, 10000)


class TestSimOne(unittest.TestCase):
    def test_stop_fills_one_tick_below_the_level(self):
        net, held, peak, kind = run([FLAT, (10000, 10000, 9800, 9850), FLAT], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual((kind, held), (1, 1))
        self.assertAlmostEqual(net, net_at(9820.0), places=6)      # stop 9,830 - 1 tick

    def test_stop_waits_for_the_hold_bars(self):
        bars = [FLAT, (10000, 10000, 9800, 9990), FLAT, FLAT, (9990, 9990, 9700, 9750), FLAT]
        _, held, _, kind = run(bars, (1.7, 0, 4, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual((kind, held), (1, 4))

    def test_gap_through_the_stop_fills_at_the_open(self):
        net, _, _, kind = run([FLAT, (9700, 9750, 9650, 9700), FLAT], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual(kind, 1)
        self.assertAlmostEqual(net, net_at(9690.0), places=6)

    def test_profit_lock_after_the_arm(self):
        # bullish bar to +1.5%, next bar falls 0.4% from the 10,150 peak (lock 10,109.4 -> fill 10,090)
        bars = [FLAT, (10000, 10150, 10000, 10150), (10150, 10150, 10050, 10060), FLAT]
        net, held, peak, kind = run(bars, (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual((kind, held), (2, 2))
        self.assertAlmostEqual(peak, 1.5, places=6)
        self.assertAlmostEqual(net, net_at(10090.0), places=6)

    def test_no_lock_below_the_arm(self):
        bars = [FLAT, (10000, 10080, 10000, 10080), (10080, 10080, 9950, 9960), FLAT]
        _, _, _, kind = run(bars, (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual(kind, 4)

    def test_bearish_bar_high_comes_first(self):
        # o->h->l->c: the high arms the lock, the same bar's fall takes it
        _, held, _, kind = run([FLAT, (10000, 10150, 9990, 10000), FLAT], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual((kind, held), (2, 1))

    def test_bullish_bar_low_comes_first(self):
        # o->l->h->c: the low is before the high, so the lock is not hit in this bar
        _, _, _, kind = run([FLAT, (10000, 10150, 9990, 10140), (10140, 10145, 10135, 10140)], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))
        self.assertEqual(kind, 4)

    def test_take_profit_is_a_limit_fill(self):
        net, _, _, kind = run([FLAT, (10000, 10300, 10000, 10250), FLAT], (1.7, 0, 0, 99.0, 0.4, 0, 0, 0, 2.0))
        self.assertEqual(kind, 3)
        self.assertAlmostEqual(net, net_at(10200.0), places=6)

    def test_ratio_mode_widens_the_giveback_with_the_profit(self):
        # peak +4%: fixed 0.4% lock sells at the 0.5% dip; ratio 0.5 (= 2% giveback) holds
        bars = [FLAT, (10000, 10400, 10000, 10400), (10400, 10400, 10340, 10350), (10350, 10360, 10340, 10350)]
        self.assertEqual(run(bars, (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))[3], 2)
        self.assertEqual(run(bars, (1.7, 0, 0, 1.0, 0.4, 2, 0, 0.5, 0))[3], 4)

    def test_trend_mode_uses_give2_while_ma5_above_ma20(self):
        bars = [FLAT, (10000, 10200, 10000, 10200), (10200, 10200, 10130, 10140), (10140, 10150, 10130, 10140)]
        self.assertEqual(run(bars, (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))[3], 2)
        self.assertEqual(run(bars, (1.7, 0, 0, 1.0, 0.4, 1, 1.2, 0, 0))[3], 4)

    def test_step_mode(self):
        bars = [FLAT, (10000, 10400, 10000, 10400), (10400, 10400, 10340, 10350), (10350, 10360, 10340, 10350)]
        self.assertEqual(run(bars, (1.7, 0, 0, 1.0, 0.4, 3, 1.2, 3.0, 0))[3], 4)     # peak >= +3% -> giveback 1.2%
        self.assertEqual(run(bars, (1.7, 0, 0, 1.0, 0.4, 3, 1.2, 5.0, 0))[3], 2)     # step not reached -> 0.4%

    def test_atr_stop_is_capped_by_the_fixed_stop(self):
        bars = [FLAT, (10000, 10000, 9890, 9900), FLAT]
        self.assertEqual(run(bars, (2.0, 3, 0, 1.0, 0.4, 0, 0, 0, 0), atr=0.3)[3], 1)   # 3 x 0.3 = 0.9% stop
        self.assertEqual(run(bars, (2.0, 3, 0, 1.0, 0.4, 0, 0, 0, 0), atr=1.0)[3], 4)   # 3.0 -> capped at 2.0%

    def test_eod_exit_one_tick_below_the_close(self):
        net, held, _, kind = run([FLAT, (10000, 10050, 9990, 10040)], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0), start_minute=g027.EOD_MIN - 1)
        self.assertEqual((kind, held), (4, 1))
        self.assertAlmostEqual(net, net_at(10030.0), places=6)

    def test_flat_round_trip_costs_fee_tax_and_one_exit_tick(self):
        net, _, _, _ = run([FLAT, (10000, 10000, 10000, 10000)], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0), start_minute=g027.EOD_MIN - 1)
        self.assertAlmostEqual(net, -(COST + 0.1 * (1 - g027.FEE - g027.TAX)), places=6)


    def test_stop_is_checked_on_the_eod_bar_before_the_close(self):
        # the 15:15 bar falls through the stop and recovers: stop fill, not the close
        net, _, _, kind = run([FLAT, (10000, 10000, 9800, 9990)], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0), start_minute=g027.EOD_MIN - 1)
        self.assertEqual(kind, 1)
        self.assertAlmostEqual(net, net_at(9820.0), places=6)

    def test_time_cap_sells_at_the_close_of_the_cap_bar(self):
        net, held, _, kind = run([FLAT, FLAT, (10000, 10030, 9990, 10020), FLAT, FLAT], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0, 3))
        self.assertEqual((kind, held), (4, 2))
        self.assertAlmostEqual(net, net_at(10010.0), places=6)

    def test_hold_and_cap_count_minutes_not_rows(self):
        # rows at minute 600, 630, 660 (missing minutes in between)
        rows = [(600 + 30 * j, o, h, l, c, c, c - 50.0, c) for j, (o, h, l, c) in enumerate(
            [FLAT, (10000, 10000, 9800, 9850), FLAT])]
        P = np.array(rows, dtype=np.float64)
        prm = np.array((1.7, 0, 4, 1.0, 0.4, 0, 0, 0, 0, 0), dtype=np.float64)
        self.assertEqual(g027._sim_one(P, 0, 3, ENTRY, np.nan, prm)[3], 1)        # 30 min later: the 4 min hold is over
        cap = np.array((9.0, 0, 0, 99.0, 0.4, 0, 0, 0, 0, 30), dtype=np.float64)
        net, held, _, kind = g027._sim_one(P, 0, 3, ENTRY, np.nan, cap)
        self.assertEqual((kind, held), (4, 1))                                    # 30 min cap -> the 630 row, not 660

    def test_flip_reverses_the_intrabar_order(self):
        bar = (10000, 10150, 9990, 10140)       # bullish: default low first (no lock), flipped high first (lock)
        self.assertEqual(run([FLAT, bar, (10140, 10145, 10135, 10140)], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0))[3], 4)
        self.assertEqual(run([FLAT, bar, (10140, 10145, 10135, 10140)], (1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0), flip=True)[3], 2)


class TestExcursions(unittest.TestCase):
    def test_touch_bar_high_counts_as_recovery_on_a_bullish_bar(self):
        P = path([FLAT, (10000, 10150, 9800, 10050), FLAT])
        levels = np.array([1.0])
        _, _, _, _, touched, later = g027.excursions(P, np.array([0]), np.array([len(P)]), np.array([ENTRY]), levels)
        self.assertTrue(touched[0, 0])
        self.assertAlmostEqual(later[0, 0], 1.5, places=6)

    def test_bearish_touch_bar_starts_from_its_close(self):
        P = path([FLAT, (10000, 10150, 9800, 9900), FLAT])
        _, _, _, _, touched, later = g027.excursions(P, np.array([0]), np.array([len(P)]), np.array([ENTRY]), np.array([1.0]))
        self.assertAlmostEqual(later[0, 0], 0.1, places=6)      # next bar high 10,010


class TestLoadOrder(unittest.TestCase):
    def test_interleaved_families_do_not_reset_the_busy_state(self):
        # ma_cross, random, ma_cross on one code: after the load sort the two ma_cross events are adjacent and the
        # second one (inside the first position) is skipped
        import pandas as pd
        ev = pd.DataFrame({"date": ["d"] * 3, "code": ["A"] * 3, "bar": [10, 11, 12], "family": ["ma_cross", "random", "ma_cross"]})
        ev = ev.sort_values(["date", "code", "family", "bar"], kind="stable").reset_index(drop=True)
        self.assertEqual(list(ev["family"]), ["ma_cross", "ma_cross", "random"])
        P = path([FLAT] * 12)
        starts = np.array([0, 2, 1], dtype=np.int64)
        lens = np.array([12, 10, 11], dtype=np.int64)
        group = pd.factorize(ev["date"] + ev["code"] + ev["family"])[0].astype(np.int64)
        prms = np.array([(1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0, 0)], dtype=np.float64)
        net, _, _ = g027.sim_grid(P, starts, lens, np.full(3, ENTRY), np.full(3, np.nan), group,
                                  ev["bar"].to_numpy(dtype=np.int64), prms, True)
        self.assertTrue(np.isfinite(net[0, 0]))
        self.assertTrue(np.isnan(net[0, 1]))
        self.assertTrue(np.isfinite(net[0, 2]))


class TestSequencing(unittest.TestCase):
    def test_second_event_inside_the_first_position_is_skipped(self):
        bars = [FLAT] * 5 + [(10000, 10000, 9800, 9850)] + [FLAT] * 5
        P = path(bars)
        starts = np.array([0, 2, 8], dtype=np.int64)
        lens = np.array([len(P), len(P) - 2, len(P) - 8], dtype=np.int64)
        entry = np.full(3, ENTRY)
        atr = np.full(3, np.nan)
        group = np.zeros(3, dtype=np.int64)
        bar = np.array([10, 12, 18], dtype=np.int64)        # trigger bars; entry bar = trigger + 1
        prms = np.array([(1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0, 0)], dtype=np.float64)
        net, kind, _ = g027.sim_grid(P, starts, lens, entry, atr, group, bar, prms, True)
        self.assertTrue(np.isfinite(net[0, 0]))
        self.assertTrue(np.isnan(net[0, 1]))                # first position is open until bar 16
        self.assertTrue(np.isfinite(net[0, 2]))
        free, _, _ = g027.sim_grid(P, starts, lens, entry, atr, group, bar, prms, False)
        self.assertTrue(np.isfinite(free[0]).all())

    def test_first_bar_after_the_exit_is_allowed(self):
        bars = [FLAT] * 3 + [(10000, 10000, 9800, 9850)] + [FLAT] * 6
        P = path(bars)
        starts = np.array([0, 4, 5], dtype=np.int64)
        lens = np.array([len(P), len(P) - 4, len(P) - 5], dtype=np.int64)
        prms = np.array([(1.7, 0, 0, 1.0, 0.4, 0, 0, 0, 0, 0)], dtype=np.float64)
        # trigger 10 -> entry bar 11 -> stopped on bar 14; a trigger on bar 14 is still busy, bar 15 is free
        net, _, _ = g027.sim_grid(P, starts, lens, np.full(3, ENTRY), np.full(3, np.nan), np.zeros(3, dtype=np.int64),
                                  np.array([10, 14, 15], dtype=np.int64), prms, True)
        self.assertTrue(np.isnan(net[0, 1]))
        self.assertTrue(np.isfinite(net[0, 2]))


if __name__ == "__main__":
    unittest.main()
