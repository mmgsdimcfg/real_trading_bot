# -*- coding: utf-8 -*-
"""g036 signal grid boundary cases (Codex review 2026-11-16): causal time exit, data ending early, re-entry on the exit bar,
stop-first take-profit / stop-loss, combo edge firing, 1-minute resampling labels."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g036_signal_grid as g  # noqa: E402


class ExitIndexTest(unittest.TestCase):
    def test_first_bar_at_or_after_target_capped_at_1515(self):
        end = np.array([600.0, 605.0, 610.0, 612.0, 625.0])
        seg = np.zeros(5, np.int64)
        x = g._exit_index(end, seg, 10.0)
        self.assertEqual(list(x[:2]), [2, 4])               # 600 -> first end >= 610; 605 -> first end >= 615 = 625
        self.assertEqual(x[4], -1)                           # last bar: no next bar

    def test_tick_bars_around_1515_never_pick_an_earlier_bar(self):
        end = np.array([900.0, 914.0, 916.0])                 # 15:14 and 15:16 bars
        x = g._exit_index(end, np.zeros(3, np.int64), 1e9)
        self.assertEqual(x[0], 2)                            # EOD exit = first bar completing after 15:15, not 15:14

    def test_data_ending_before_target_is_flagged(self):
        end = np.array([900.0, 905.0, 910.0])                 # file stops at 15:10
        x = g._exit_index(end, np.zeros(3, np.int64), 1e9)
        self.assertEqual(x[0], -2)

    def test_segments_are_independent(self):
        end = np.array([600.0, 700.0, 600.0, 700.0])
        seg = np.array([0, 0, 1, 1], np.int64)
        x = g._exit_index(end, seg, 30.0)
        self.assertEqual(list(x), [1, -1, 3, -1])


def _run(o, h, l, c, end, fire, exit_spec):
    n = len(o)
    seg = np.zeros(n, np.int64)
    E = np.zeros((len(g.EVENTS), n), bool)
    E[0] = fire
    R = np.vstack([g._recent(E[i], seg, g.WINDOW) for i in range(len(g.EVENTS))])
    S = np.zeros((len(g.STATES), n), bool)
    entry_ok = np.r_[np.ones(n - 1, bool), False]
    xk = np.array([s[0] for s in exit_spec], np.int64)
    xi = np.vstack([g._exit_index(np.asarray(end, float), seg, s[1]) for s in exit_spec])
    seg_last = np.full(n, n - 1, np.int64)
    kind = np.array([0], np.int64)
    comp = np.array([[0, -1, -1]], np.int64)
    return g._grid(kind, comp, E, R, S, entry_ok, seg, seg_last, xk, xi, np.array([s[2] for s in exit_spec]),
                   np.array([s[3] for s in exit_spec]), np.asarray(o, float), np.asarray(h, float), np.asarray(l, float),
                   np.asarray(c, float), np.zeros(n, np.int64), 1)


class GridTest(unittest.TestCase):
    def test_signal_on_the_exit_bar_re_enters_at_the_next_open(self):
        end = [600, 601, 602, 603, 604]
        px = [100.0] * 5
        tot, cnt, *_ = _run(px, px, px, px, end, np.array([1, 0, 1, 0, 0], bool), [(0, 2.0, 0, 0)])
        self.assertEqual(cnt[0, 0, 0], 2)                    # 600 -> exit bar 602; 602's signal enters at 603 open

    def test_stop_wins_when_one_bar_touches_both(self):
        end = [600, 601, 602, 700]          # last bar after the 60-minute limit
        o = [100, 100, 100, 100]
        h = [100, 103, 100, 100]
        l = [100, 98, 100, 100]
        c = [100, 100, 100, 100]
        tot, cnt, win, gro, _t = _run(o, h, l, c, end, np.array([1, 0, 0, 0], bool), [(1, 60.0, 0.015, 0.01)])
        self.assertAlmostEqual(gro[0, 0, 0], (100.5 * 0.99 / 100 - 1) * 100)   # stop 1 % under the slipped entry, not the target

    def test_gap_through_stop_fills_at_the_open(self):
        end = [600, 601, 602, 700]          # last bar after the 60-minute limit
        o = [100, 100, 95, 95]
        h = [100, 100, 95, 95]
        l = [100, 100, 95, 95]
        c = [100, 100, 95, 95]
        *_, gro, _t = _run(o, h, l, c, end, np.array([1, 0, 0, 0], bool), [(1, 60.0, 0.015, 0.01)])
        self.assertAlmostEqual(gro[0, 0, 0], -5.0)

    def test_truncated_trade_is_counted_apart(self):
        end = [900, 905, 910]
        px = [100.0] * 3
        tot, cnt, win, gro, trunc = _run(px, px, px, px, end, np.array([1, 0, 0], bool), [(0, 1e9, 0, 0)])
        self.assertEqual((cnt.sum(), trunc[0, 0]), (0, 1))

    def test_stop_before_the_data_ends_is_a_trade(self):
        """Codex re-review #1: the time target (60 min) is never observed, but the stop was hit first."""
        end = [600, 601, 602, 610]                          # file stops at 10:10, target 10:60
        o = [100, 100, 100, 100]
        h = [100, 100, 100, 100]
        l = [100, 100, 98, 100]
        c = [100, 100, 100, 100]
        tot, cnt, win, gro, trunc = _run(o, h, l, c, end, np.array([1, 0, 0, 0], bool), [(1, 60.0, 0.015, 0.01)])
        self.assertEqual((cnt.sum(), trunc[0, 0]), (1, 0))
        self.assertAlmostEqual(gro[0, 0, 0], (100.5 * 0.99 / 100 - 1) * 100, places=6)

    def test_take_profit_before_the_data_ends_is_a_trade(self):
        end = [600, 601, 602, 610]
        o = [100, 100, 100, 100]
        h = [100, 100, 103, 100]
        l = [100, 100, 100, 100]
        c = [100, 100, 100, 100]
        tot, cnt, win, gro, trunc = _run(o, h, l, c, end, np.array([1, 0, 0, 0], bool), [(1, 60.0, 0.015, 0.01)])
        self.assertEqual((cnt.sum(), trunc[0, 0]), (1, 0))
        self.assertAlmostEqual(gro[0, 0, 0], (100.5 * 1.015 / 100 - 1) * 100, places=6)

    def test_next_segment_trades_again_after_a_truncated_one(self):
        n = 6
        end = np.array([900.0, 901.0, 902.0, 600.0, 601.0, 640.0])
        seg = np.array([0, 0, 0, 1, 1, 1], np.int64)
        E = np.zeros((len(g.EVENTS), n), bool)
        E[0, 0] = E[0, 3] = True
        R = np.vstack([g._recent(E[i], seg, g.WINDOW) for i in range(len(g.EVENTS))])
        S = np.zeros((len(g.STATES), n), bool)
        entry_ok = np.array([True, True, False, True, True, False])
        xi = np.vstack([g._exit_index(end, seg, 30.0)])
        seg_last = np.array([2, 2, 2, 5, 5, 5], np.int64)
        px = np.full(n, 100.0)
        tot, cnt, win, gro, trunc = g._grid(np.array([0], np.int64), np.array([[0, -1, -1]], np.int64), E, R, S, entry_ok,
                                            seg, seg_last, np.array([0], np.int64), xi, np.zeros(1), np.zeros(1), px, px,
                                            px, px, np.zeros(n, np.int64), 1)
        self.assertEqual((trunc[0, 0], cnt.sum()), (1, 1))  # symbol-day 1 excluded, symbol-day 2 still trades

    def test_open_position_at_data_end_blocks_later_signals(self):
        end = [900, 901, 902, 903]                          # EOD target 15:15 never reached
        px = [100.0] * 4
        tot, cnt, win, gro, trunc = _run(px, px, px, px, end, np.array([1, 1, 1, 0], bool), [(0, 1e9, 0, 0)])
        self.assertEqual((cnt.sum(), trunc[0, 0]), (0, 1))  # one excluded position, not three

    def test_tick_size_from_the_actual_fill_prices(self):
        end = [600, 601, 602]
        o = [19990, 19990, 20500]
        h = [19990, 19990, 20500]
        l = [19990, 19990, 20500]
        c = [19990, 20500, 20500]                           # entry 19,990 (tick 10), exit 20,500 (tick 50)
        tot, cnt, *_ = _run(o, h, l, c, end, np.array([1, 0, 0], bool), [(0, 1.0, 0, 0)])
        epx, xpx = 19990 + 5, 20500 - 25
        exp = (xpx * (1 - 0.00015 - 0.0020) - epx * 1.00015) / epx * 100
        self.assertAlmostEqual(tot[0, 0, 0], exp, places=6)

    def test_combo_fires_once_when_the_set_completes(self):
        seg = np.zeros(6, np.int64)
        a = np.array([1, 0, 0, 0, 0, 0], bool)
        b = np.array([0, 0, 1, 0, 0, 0], bool)
        ra, rb = g._recent(a, seg, 3), g._recent(b, seg, 3)
        both = ra & rb
        fire = both & ~np.r_[False, both[:-1]]
        self.assertEqual(list(np.flatnonzero(fire)), [2])     # a at 0 still inside the 3-bar window at 2


class ResampleTest(unittest.TestCase):
    def test_end_labels(self):
        arr = np.array([[541 + i, 10 + i, 11 + i, 9 + i, 10 + i, 1.0] for i in range(6)])   # 09:01..09:06
        r = g.resample(arr, 3)
        self.assertEqual(list(r[:, 0]), [543.0, 546.0])
        self.assertEqual(list(r[0, 1:]), [10.0, 13.0, 9.0, 12.0, 3.0])


if __name__ == "__main__":
    unittest.main()
