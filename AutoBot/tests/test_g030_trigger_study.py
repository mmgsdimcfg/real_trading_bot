# -*- coding: utf-8 -*-
"""g030 trigger study: tick-bar cross runs, the 1-minute state, and what each part of a variant gates."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data_simulation", "live_trading"):
    sys.path.insert(0, str(ROOT / sub))

import g030_trigger_study as g030  # noqa: E402


def minute_bars(closes, first_label=541):
    """g026 bar rows (label minute, o, h, l, c, v), flat candles."""
    c = np.asarray(closes, dtype=float)
    return np.column_stack([first_label + np.arange(len(c)), c, c, c, c, np.full(len(c), 100.0)])


class TickRunsTest(unittest.TestCase):
    def test_cross_time_and_close_follow_the_run(self):
        # 3-tick bars; closes fall, then rise: the cross is the bar where MA5 first exceeds MA20
        closes = np.r_[np.linspace(110, 100, 25), np.linspace(100.5, 120, 25)]
        price = np.repeat(closes, 3)
        sec = 9 * 3600 + np.arange(len(price)) * 2
        r = g030.tick_runs(sec, price, 3, np.array([]))
        self.assertEqual(len(r["t"]), 50)
        k = int(np.flatnonzero(r["above"])[0])
        self.assertFalse(r["above"][k - 1])
        self.assertEqual(r["cross_t"][k], r["t"][k])
        self.assertEqual(r["cross_c"][k], closes[k])
        self.assertEqual(r["cross_t"][k + 5], r["t"][k])                  # later bars of the run keep the cross of its start
        self.assertTrue(np.isnan(r["cross_t"][k - 1]))                    # no cross while below

    def test_a_run_that_began_before_today_has_no_cross_today(self):
        warm = np.linspace(100, 130, 40)                                  # already rising yesterday
        price = np.repeat(np.linspace(130.5, 140, 10), 3)
        r = g030.tick_runs(9 * 3600 + np.arange(len(price)), price, 3, warm)
        self.assertTrue(r["above"].all())
        self.assertTrue(np.isnan(r["cross_t"]).all())


class MinuteStateTest(unittest.TestCase):
    def test_run_start_is_the_label_of_the_cross_bar(self):
        closes = np.r_[np.linspace(110, 100, 25), np.linspace(100.5, 115, 25)]
        m1 = g030.minute_state(minute_bars(closes), None)
        k = int(np.flatnonzero(m1["above"])[0])
        self.assertEqual(m1["run_start"][k], (541 + k) * 60.0)
        self.assertEqual(m1["run_start"][k + 3], (541 + k) * 60.0)
        self.assertTrue(np.isnan(m1["run_start"][k - 1]))


class VariantGateTest(unittest.TestCase):
    """One symbol-day: the 1-minute series rises from 09:01 (state above from about 09:30 on), ticks cross once at 10:00."""

    @classmethod
    def setUpClass(cls):
        n = 360
        closes = np.r_[np.linspace(10100, 10000, 25), np.linspace(10002, 10300, n - 25)]
        cls.arr = minute_bars(closes)
        cls.m1 = g030.minute_state(cls.arr, None)
        # ticks: one trade per second from 09:00 at the minute series' price (so price > MA20 while it rises)
        cls.sec = 9 * 3600 + np.arange((n - 1) * 60)
        cls.price = np.interp(cls.sec, cls.arr[:, 0] * 60.0, closes)
        cross_t = 10 * 3600.0
        t = np.array([cross_t - 60, cross_t, cross_t + 3000.0, cross_t + 9000.0])
        cc = float(np.interp(cross_t, cls.sec, cls.price))
        cls.trun = {"t": t, "above": np.array([False, True, True, True]),
                    "cross_t": np.array([np.nan, cross_t, cross_t, cross_t]), "cross_c": np.array([np.nan, cc, cc, cc])}

    def first_entry(self, variant):
        tr = g030.simulate_code(self.sec, self.price, [self.trun], self.m1, self.arr, variant)
        return tr[0][0] if tr else None

    def test_tick_cross_is_required_and_expires(self):
        live = (True, 0.0, True, 240.0, None, "")
        t = self.first_entry(live)
        self.assertGreaterEqual(t, 10 * 3600)                             # not before the cross
        self.assertLessEqual(t, 10 * 3600 + 240)                          # inside its validity
        self.assertEqual(self.first_entry((True, 0.0, False, 0.0, None, "")) < 10 * 3600, True)   # without ticks: earlier

    def test_minute_state_age_limit_blocks_an_old_run(self):
        # the 1-minute run began around 09:30; at the 10:00 tick cross it is ~30 minutes old
        self.assertIsNone(self.first_entry((True, 900.0, True, 240.0, None, "")))
        self.assertIsNotNone(self.first_entry((True, 3600.0, True, 240.0, None, "")))

    def test_chase_guard_uses_the_cross_bar_close(self):
        # prices keep rising ~0.0085% per minute after the cross: within 240 s the chase stays far below 0.5%
        self.assertIsNotNone(self.first_entry((True, 0.0, True, 240.0, 0.5, "")))
        # a longer validity with a tiny allowance: the first check is still at the cross -> allowed there, so tighten to 0
        tr = g030.simulate_code(self.sec, self.price, [self.trun], self.m1, self.arr, (True, 0.0, True, 3000.0, 0.0001, ""))
        self.assertTrue(all(abs(x[0] - 10 * 3600) <= g030.CHECK_STEP_S for x in tr[:1]))

    def test_trigger_off_buys_at_the_first_check_and_one_position_at_a_time(self):
        tr = g030.simulate_code(self.sec, self.price, [self.trun], self.m1, self.arr, g030.VARIANTS["trigger_off"])
        self.assertEqual(tr[0][0], g030.ENTRY_START_S)
        for a, b in zip(tr, tr[1:]):
            self.assertGreater(b[0], a[0] + 60)                           # the next buy comes after the previous exit


if __name__ == "__main__":
    unittest.main()
