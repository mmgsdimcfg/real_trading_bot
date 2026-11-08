# -*- coding: utf-8 -*-
"""g020 exit ladder simulator (2026-10-31)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g017_user_policy_sim as g017  # noqa: E402
import g020_exit_ladder_sim as g020  # noqa: E402

USER = np.array([2.0, 4.0, 0.5, 0.8, 1.2, 0.0])
SELL_K = 1 - g017.FEE - g017.TAX


def _bars(closes, minute0=600):
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c), np.minimum(o, c)
    return o, h, l, c, np.ones(len(c), bool), np.arange(len(c), dtype=np.int64) + minute0


class LadderTest(unittest.TestCase):
    def test_user_default_index(self):
        np.testing.assert_array_equal(g020.PRMS[g020.USER], USER)

    def test_stop_ladder_half_then_rest(self):
        # entry fill = 10000 + 10 tick = 10010; qty = floor(500000/10010) = 49
        o, h, l, c, obs, m = _bars([10000, 10000, 9930, 9880, 9850])
        net, reason, _ = g020.simulate_ladder(o, h, l, c, obs, m, 0, USER, 500_000.0)
        e1, q = 10010.0, 49.0
        sl1 = g017._dn(e1 * (1 - 0.008)) - 10   # 9930 -> 9920
        sl2 = g017._dn(e1 * (1 - 0.012)) - 10   # 9880 -> 9870
        q1 = np.floor(q * 0.5 + 0.5)
        exp = -e1 * q * (1 + g017.FEE) + sl1 * q1 * SELL_K + sl2 * (q - q1) * SELL_K
        self.assertEqual(reason, 3)
        self.assertAlmostEqual(net, exp, places=4)

    def test_tp1_then_trail_sells_rest(self):
        # rise to +2% (TP1 30%), peak +3%, then drop 0.5% from the peak -> trail sells the remaining 70%
        o, h, l, c, obs, m = _bars([10000, 10000, 10220, 10320, 10260])
        net, reason, _ = g020.simulate_ladder(o, h, l, c, obs, m, 0, USER, 500_000.0)
        e1, q = 10010.0, 49.0
        tp1 = g017._up(e1 * 1.02)                   # 10220 (limit fill)
        q1 = np.floor(q * 0.3 + 0.5)
        trail = g017._dn(10320 * 0.995) - 10        # 10260 -> 10250
        exp = -e1 * q * (1 + g017.FEE) + tp1 * q1 * SELL_K + trail * (q - q1) * SELL_K
        self.assertEqual(reason, 2)
        self.assertAlmostEqual(net, exp, places=4)

    def test_tp1_tp2_both_fill_on_surge(self):
        o, h, l, c, obs, m = _bars([10000, 10000, 10500, 10700, 10600])
        net, reason, _ = g020.simulate_ladder(o, h, l, c, obs, m, 0, USER, 500_000.0)
        e1, q = 10010.0, 49.0
        q1 = np.floor(q * 0.3 + 0.5)
        q2 = np.floor(q * 0.3 + 0.5)
        tp1, tp2 = g017._up(e1 * 1.02), g017._up(e1 * 1.04)
        trail = g017._dn(10700 * 0.995) - 10
        exp = (-e1 * q * (1 + g017.FEE) + tp1 * q1 * SELL_K + tp2 * q2 * SELL_K
               + trail * (q - q1 - q2) * SELL_K)
        self.assertEqual(reason, 2)
        self.assertAlmostEqual(net, exp, places=4)

    def test_no_trail_before_tp1(self):
        # +1.5% then back to +0.5% -> not armed (arm=0 waits for TP1) -> held to EOD
        o, h, l, c, obs, m = _bars([10000, 10000, 10160, 10060], minute0=g017.EOD_MINUTE - 3)
        _, reason, _ = g020.simulate_ladder(o, h, l, c, obs, m, 0, USER, 500_000.0)
        self.assertEqual(reason, 4)


if __name__ == "__main__":
    unittest.main()
