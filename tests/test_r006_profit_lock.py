# -*- coding: utf-8 -*-
"""r006 _023_profit_lock (2026-11-02, 091580 상신이디피: entry 26,750 -> peak 27,050 (+1.12%) -> stopped at -1.12%)."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r006_sell_conditions as r006  # noqa: E402

ENTRY = 26750.0


class _Api:
    def __init__(self):
        self.orders = []
        self.meta = []
        self.accept = True

    def place_sell_order(self, code, qty, dt, reason, nxt, price=None, code_name=None, market_order=False):
        self.orders.append((qty, reason, market_order))
        return self.accept

    def _record_position_meta(self, code, pos):
        self.meta.append(dict(pos))

    def persist_live_state(self, date_str=None):
        pass


class TestProfitLock(unittest.TestCase):
    def setUp(self):
        self.api = _Api()
        self.pos = {"quantity": 18, "entry_quantity": 18}
        self.peak = ENTRY

    def _tick(self, price):
        self.peak = max(self.peak, price)
        ctx = SimpleNamespace(
            code="091580", name="상신이디피", symbol_label="091580_상신이디피", current_dt=datetime(2026, 11, 2, 10, 25),
            nxt_tradeable=False, pos=self.pos, price=price, api=self.api, date_str="20261102", log=lambda m: None,
            entry_price=ENTRY, pnl_pct=price / ENTRY - 1, highest_price=self.peak, peak_pnl_pct=self.peak / ENTRY - 1,
            atr_pct=0.008, atr_tp_pct=0.03, entry_qty=0, tp1_target_pct=float("nan"), tp2_target_pct=float("nan"),
            atr_tp1_dynamic_pct=float("nan"),
        )
        return r006._023_profit_lock(ctx)

    @mock.patch.multiple(r006, PROFIT_LOCK_GIVEBACK_PCT=0.005, PROFIT_LOCK_RATIO=0.50, PROFIT_LOCK_REST_BREAKEVEN=True)
    def test_091580_path_locks_half_then_breakeven(self):
        fired = [(p, self._tick(p)) for p in (26800, 26950, 27050, 27000, 26950)]
        self.assertFalse(any(f for _, f in fired))  # 26,950 > lock 26,915
        self.assertTrue(self._tick(26900))
        self.assertEqual(self.api.orders[-1], (9, "PROFIT_LOCK_50PCT_GIVEBACK_0.5PCT", True))
        self.assertTrue(self.pos["profit_lock_done"])
        self.assertTrue(self.api.meta[-1]["profit_lock_done"])
        self.pos["quantity"] = 9  # partial filled
        self.assertFalse(self._tick(26850))  # still in profit -> rest held
        self.assertTrue(self._tick(26750))   # breakeven
        self.assertEqual(self.api.orders[-1], (9, "PROFIT_LOCK_REST_BREAKEVEN", True))

    def test_live_config_sells_all_at_giveback_04(self):
        self.assertEqual((r006.PROFIT_LOCK_GIVEBACK_PCT, r006.PROFIT_LOCK_RATIO), (0.004, 1.0))  # 2026-11-03 user decision
        for p in (26950, 27050, 26950):  # 26,950 > lock 26,942
            self.assertFalse(self._tick(p))
        self.assertTrue(self._tick(26900))
        self.assertEqual(self.api.orders[-1], (18, "PROFIT_LOCK_100PCT_GIVEBACK_0.4PCT", True))
        self.pos["quantity"] = 0  # fully filled
        self.assertFalse(self._tick(26700))
        self.assertEqual(len(self.api.orders), 1)

    def test_not_armed_below_arm_pnl(self):
        for p in (26900, 27000, 26850, 26700):  # peak +0.93% < 1.0%
            self.assertFalse(self._tick(p))
        self.assertEqual(self.api.orders, [])

    def test_after_tp1_and_disabled(self):
        self.pos["tp1_done"] = True
        self._tick(27050)
        self.assertFalse(self._tick(26900))
        self.pos["tp1_done"] = False
        orig = r006.ENABLE_PROFIT_LOCK
        r006.ENABLE_PROFIT_LOCK = False
        try:
            self.assertFalse(self._tick(26900))
        finally:
            r006.ENABLE_PROFIT_LOCK = orig
        self.assertEqual(self.api.orders, [])

    def test_rejected_order_does_not_mark_done(self):
        self.api.accept = False
        self._tick(27050)
        self.assertTrue(self._tick(26900))
        self.assertFalse(self.pos.get("profit_lock_done", False))

    def test_condition_registered_before_012(self):
        names = [c.func_name for c in r006.SELL_CONDITIONS]
        self.assertLess(names.index("_023_profit_lock"), names.index("_012_peak_retracement_guard"))
        self.assertEqual(len({c.no for c in r006.SELL_CONDITIONS}), len(r006.SELL_CONDITIONS))


if __name__ == "__main__":
    unittest.main()
