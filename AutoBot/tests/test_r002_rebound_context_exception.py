# -*- coding: utf-8 -*-
"""Rebound context exception (2026-10-29, 003160 디아이 V-shaped rebound 10:42~11:28).

r002 _rebound_confirmed + _gate_bb_slope_rising / _gate_stochastic_buy_signal (r001 ENABLE_REBOUND_CONTEXT_EXCEPTION)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r002_strategy_core_shared as r002  # noqa: E402


def _ctx(close=43750.0, bb_mid=42538.0, live=43600.0, di_plus=33.6, di_minus=5.7, macd=150.0, macd_sig=100.0,
         slope=-2.9, k=98.0, d=95.0, k_prev=97.0, d_prev=93.0):
    """Values of 003160 at 11:06 (3-min close above mid, +DI dominant, MACD above signal, overbought stochastic)."""
    cur = pd.Series({"close": close, "DI_PLUS": di_plus, "DI_MINUS": di_minus, "MACD": macd, "MACD_SIGNAL": macd_sig,
                     "STOCH_K": k, "STOCH_D": d})
    prev = pd.Series({"STOCH_K": k_prev, "STOCH_D": d_prev})
    return SimpleNamespace(cur=cur, prev=prev, cur_bb=bb_mid, live_price=live, bb_slope_pct=slope,
                           config=SimpleNamespace(stoch_overbought=96.0))


class TestReboundException(unittest.TestCase):
    def _on(self):
        p = mock.patch.object(r002, "ENABLE_REBOUND_CONTEXT_EXCEPTION", True)
        p.start()
        self.addCleanup(p.stop)

    def test_off_keeps_existing_rejects(self):
        with mock.patch.object(r002, "ENABLE_REBOUND_CONTEXT_EXCEPTION", False):
            self.assertFalse(r002._rebound_confirmed(_ctx()))
            self.assertFalse(r002._gate_bb_slope_rising(_ctx())[0])
            ok, _, reason = r002._gate_stochastic_buy_signal(_ctx())
            self.assertFalse(ok)
            self.assertIn("STOCH", reason)

    def test_on_rebound_passes_both_gates(self):
        self._on()
        self.assertTrue(r002._rebound_confirmed(_ctx()))
        self.assertEqual(r002._gate_bb_slope_rising(_ctx())[0], True)
        self.assertEqual(r002._gate_stochastic_buy_signal(_ctx())[0], True)

    def test_on_each_condition_required(self):
        self._on()
        for kw in ({"close": 42400.0},            # 3-min close below mid
                   {"live": 42500.0},             # live below mid
                   {"di_plus": 18.2, "di_minus": 17.4},  # DI spread 0.8 < 10 (10:45)
                   {"macd": 90.0},                # MACD below signal
                   {"slope": -6.5}):              # crash: slope below floor
            ctx = _ctx(**kw)
            self.assertFalse(r002._rebound_confirmed(ctx), kw)
            self.assertFalse(r002._gate_bb_slope_rising(ctx)[0], kw)
            self.assertFalse(r002._gate_stochastic_buy_signal(ctx)[0], kw)

    def test_on_missing_values_not_confirmed(self):
        self._on()
        ctx = _ctx()
        ctx.cur = ctx.cur.drop("MACD")
        self.assertFalse(r002._rebound_confirmed(ctx))
        ctx = _ctx()
        ctx.live_price = None
        self.assertFalse(r002._rebound_confirmed(ctx))

    def test_on_stoch_data_missing_still_rejected(self):
        self._on()
        ctx = _ctx()
        ctx.prev = pd.Series({"STOCH_K": float("nan"), "STOCH_D": float("nan")})
        self.assertEqual(r002._gate_stochastic_buy_signal(ctx)[2], "STOCH_DATA_MISSING")

    def test_passing_gates_untouched(self):
        self._on()
        ctx = _ctx(slope=1.0, k=60.0, d=50.0, k_prev=45.0, d_prev=48.0, close=42000.0)  # not a rebound, gates pass
        self.assertTrue(r002._gate_bb_slope_rising(ctx)[0])
        self.assertTrue(r002._gate_stochastic_buy_signal(ctx)[0])


if __name__ == "__main__":
    unittest.main()
