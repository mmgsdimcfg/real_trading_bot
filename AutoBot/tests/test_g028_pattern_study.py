# -*- coding: utf-8 -*-
"""g028 pattern study: helper semantics, no look-ahead in the features, label entry convention."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data_simulation", "live_trading"):
    sys.path.insert(0, str(ROOT / sub))

import g026_timing_study as g026  # noqa: E402
import g027_timing_grid as g027  # noqa: E402
import g028_pattern_study as g028  # noqa: E402


def _bars(n=200, seed=3):
    rng = np.random.default_rng(seed)
    c = 10000.0 + np.cumsum(rng.normal(0, 20, n)).round(-1)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 10.0
    l = np.minimum(o, c) - 10.0
    v = rng.integers(100, 5000, n).astype(float)
    minute = 9 * 60 + 1 + np.arange(n)
    return np.column_stack([minute, o, h, l, c, v]).astype(np.float64)


class HelperTest(unittest.TestCase):
    def test_run_len_counts_bars_since_the_flag_turned_true(self):
        flag = np.array([False, True, True, False, True])
        self.assertEqual(g028._run_len(flag).tolist(), [-1, 0, 1, -1, 0])

    def test_since_argmin_is_zero_on_a_new_low(self):
        x = np.array([5.0, 4.0, 6.0, 7.0, 3.0])
        self.assertEqual(g028._since_argmin(x, 3).tolist(), [0, 0, 1, 2, 0])

    def test_shift_pads_with_nan(self):
        out = g028._shift(np.array([1.0, 2.0, 3.0]), 1)
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1:].tolist(), [1.0, 2.0])


class FeatureTest(unittest.TestCase):
    def test_features_at_a_bar_do_not_change_when_later_bars_change(self):
        arr = _bars()
        ind, off = g026._indicator_frame(arr, None)
        full = g028.feature_arrays(ind, off, 10000.0)
        k = 150
        ind2, off2 = g026._indicator_frame(arr[:k + 1], None)
        cut = g028.feature_arrays(ind2, off2, 10000.0)
        for name in full:
            a, b = full[name][k], cut[name][k]
            self.assertTrue((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, name)

    def test_rule_masks_need_every_part_of_the_turn(self):
        g = {k: np.array([v]) for k, v in {
            "above_age": -1.0, "di_x_age": 0.0, "macd_x_age": 2.0, "rsi_x_age": 1.0, "ten_x_age": 0.0, "range20": 3.0,
            "bw_rel": 2.0, "brk20": -1.0, "vol_ratio": 1.0, "dd30": -4.5, "rsi_min10": 18.0, "adx_max10": 84.0,
            "d_close_ma20_pct": 10.0, "day_ret": -6.0, "tick_pct": 0.17}.items()}
        m = g028.rule_masks(g)
        self.assertTrue(m["V: dd<=-3, RSI low<25, ADX high>=60"][0])       # the 006360 13:56 bar of 2026-11-11
        self.assertTrue(m["turn early dd<=-2, tick>0.13%"][0])
        g["macd_x_age"] = np.array([-1.0])                                  # histogram still negative -> no turn
        self.assertFalse(g028.rule_masks(g)["turn"][0])


class LabelTest(unittest.TestCase):
    def test_entry_is_the_next_bar_open_plus_one_tick_and_tp3_pays_after_costs(self):
        rows = [(600 + j, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 9950.0, 10000.0) for j in range(6)]
        rows[3] = (603, 10000.0, 10400.0, 10000.0, 10400.0, 10000.0, 9950.0, 10000.0)        # +4% bar
        P = np.array(rows, dtype=np.float64)
        prms = np.array([g028.POLICIES["tp3"], g028.POLICIES["live"]], dtype=np.float64)
        entry, net, kind, held, mfe, mae = g028._labels(P, np.array([0], dtype=np.int64), 5, np.full(6, np.nan), prms)
        self.assertEqual(entry[0], 10010.0)
        self.assertEqual(kind[0, 0], 3.0)
        self.assertAlmostEqual(net[0, 0], (g027._dn(10010.0 * 1.03) * (1 - g027.FEE - g027.TAX) - 10010.0 * (1 + g027.FEE))
                               / 10010.0 * 100.0, places=6)
        self.assertGreater(mfe[0], 3.0)
        self.assertLess(net[0, 1], net[0, 0])       # the live profit lock gives part of the move back

    def test_a_trigger_on_the_last_bar_has_no_label(self):
        P = np.array([(600 + j, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 9950.0, 10000.0) for j in range(3)])
        prms = np.array([g028.POLICIES["tp3"]], dtype=np.float64)
        entry, net, *_ = g028._labels(P, np.array([1], dtype=np.int64), 2, np.full(3, np.nan), prms)
        self.assertTrue(np.isnan(entry[0]) and np.isnan(net[0, 0]))


class FlowTest(unittest.TestCase):
    def _write(self, rows, header="recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,"
                                  "shnu_cntg_csnu,askp1,bidp1,hour_cls_code"):
        import tempfile
        d = tempfile.mkdtemp()
        path = Path(d) / "000001.csv"
        path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_minute_end_labels_and_buy_sell_split(self):
        rows = []
        for k in range(150):        # 09:00 minute: 150 buys of 10 shares
            rows.append(f"x,000001,20261111,0900{k % 60:02d},10000,10,0,1,0,0,10010,10000,0")
        for k in range(100):        # 09:01 minute: 100 sells of 10 shares
            rows.append(f"x,000001,20261111,0901{k % 60:02d},10000,10,0,5,0,0,10010,10000,0")
        rows.append("x,000001,20261111,152500,10000,999,0,1,0,0,10010,10000,0")      # after the continuous session
        fm = g028.flow_minutes(self._write(rows)).set_index("minute")
        self.assertAlmostEqual(fm.loc[541, "flow1"], 1.0)                 # 09:00:xx ticks -> label 09:01
        self.assertAlmostEqual(fm.loc[542, "flow1"], -1.0)
        self.assertAlmostEqual(fm.loc[542, "flow5"], (1500 - 1000) / 2500)
        self.assertAlmostEqual(fm.loc[542, "strength_day"], 1.5)
        self.assertAlmostEqual(fm.loc[542, "spread_ticks"], 1.0)
        self.assertTrue(np.isnan(fm.loc[560, "flow1"]))                   # no ticks in that minute
        self.assertLess(fm.loc[560, "cov15"], 1.0)

    def test_other_venues_are_dropped_when_the_file_has_a_venue_column(self):
        head = ("recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,"
                "bidp1,hour_cls_code,venue,session")
        rows = [f"x,000001,20261111,0900{k % 60:02d},10000,10,0,1,0,0,10010,10000,0,KRX,regular" for k in range(250)]
        rows += [f"x,000001,20261111,0900{k % 60:02d},10000,10,0,5,0,0,10010,10000,0,NXT,regular" for k in range(250)]
        fm = g028.flow_minutes(self._write(rows, head)).set_index("minute")
        self.assertAlmostEqual(fm.loc[541, "flow1"], 1.0)


if __name__ == "__main__":
    unittest.main()
