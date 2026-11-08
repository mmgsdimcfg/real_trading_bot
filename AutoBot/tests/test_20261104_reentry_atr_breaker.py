# -*- coding: utf-8 -*-
"""2026-11-04 user decisions (12:55/13:44 tick golden crosses without buys):
(1) stop-loss circuit breaker removed (r005 _007, r006 arming), (2) re-entry limit 5 after a net-profitable exit
(r003 _reentry_limit / _confirm_pending_sell, g003 _last_exit_net_profit), (3) relaxed entry-ATR floor for the triple
path or after RELAXED_ENTRY_ATR_FROM (r002 min_entry_atr_threshold_pct)."""
from __future__ import annotations

import collections
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AUTO_TRADING_RUNTIME", tempfile.mkdtemp(prefix="t20261104_"))
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import r001_define_config as r001  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
import r006_sell_conditions as r006  # noqa: E402

try:
    import r003_trade_live_execute as r003  # noqa: E402
except ImportError:  # broker SDK missing
    r003 = None


class TestBreakerRemoved(unittest.TestCase):
    def test_no_breaker_condition_or_config(self):
        names = [c.func_name for c in r005.BUY_CONDITIONS]
        self.assertNotIn("_007_stoploss_circuit_breaker", names)
        self.assertFalse(hasattr(r001, "HARD_STOP_CIRCUIT_BREAKER_COUNT"))
        self.assertFalse(hasattr(r005.RiskState(), "circuit_breaker_until"))
        # basic mode list still builds (used to include _007 via range(1, 8))
        self.assertEqual([c.no for c in r005.BASIC_BUY_CONDITIONS][:6], [1, 2, 3, 4, 5, 6])

    def test_register_stop_counts_and_blocks_code_only(self):
        logs = []
        ctx = SimpleNamespace(code="326030", state=SimpleNamespace(hard_stop_today_codes=set()),
                              risk=r005.RiskState(), current_dt=datetime(2026, 11, 4, 11, 33), log=logs.append)
        for _ in range(5):
            r006._register_hard_stop(ctx, "STOP_LOSS(ATR/HARD)")
        self.assertEqual(ctx.risk.hard_stop_daily_count, 5)
        self.assertEqual(ctx.state.hard_stop_today_codes, {"326030"})
        self.assertFalse(any("CIRCUIT" in m for m in logs))


class TestRelaxedEntryAtr(unittest.TestCase):
    base = r001.STAGED_TP1_PCT * r001.MIN_ENTRY_ATR_TO_TP1_RATIO

    def test_threshold(self):
        f = r002.min_entry_atr_threshold_pct
        with mock.patch.object(r002, "ENABLE_RELAXED_ENTRY_ATR", True), \
                mock.patch.object(r002, "RELAXED_ENTRY_ATR_MIN_PCT", 0.004), \
                mock.patch.object(r002, "RELAXED_ENTRY_ATR_FROM", "12:00"):
            self.assertAlmostEqual(f(pd.Timestamp("2026-11-04 10:00"), False), self.base)
            self.assertAlmostEqual(f(pd.Timestamp("2026-11-04 10:00"), True), 0.004)
            self.assertAlmostEqual(f(pd.Timestamp("2026-11-04 11:59"), False), self.base)
            self.assertAlmostEqual(f(pd.Timestamp("2026-11-04 12:00"), False), 0.004)
            self.assertAlmostEqual(f(None, False), self.base)
            with mock.patch.object(r002, "RELAXED_ENTRY_ATR_MIN_PCT", 0.02):  # never raises the floor
                self.assertAlmostEqual(f(pd.Timestamp("2026-11-04 13:00"), True), self.base)
        with mock.patch.object(r002, "ENABLE_RELAXED_ENTRY_ATR", False):
            self.assertAlmostEqual(f(pd.Timestamp("2026-11-04 13:44"), True), self.base)

    def _gate(self, now, triple, atr_pct):
        cur = pd.Series({"ATR": 100000 * atr_pct})
        ctx = SimpleNamespace(cur=cur, live_price=100000.0, now=pd.Timestamp(now), triple_cross=triple)
        with mock.patch.object(r002, "ENABLE_MIN_ENTRY_ATR_FILTER", True), \
                mock.patch.object(r002, "ENABLE_RELAXED_ENTRY_ATR", True), \
                mock.patch.object(r002, "RELAXED_ENTRY_ATR_MIN_PCT", 0.004), \
                mock.patch.object(r002, "RELAXED_ENTRY_ATR_FROM", "12:00"):
            return r002._gate_min_entry_atr_volatility(ctx)[0]

    def test_gate_20261104_cases(self):
        self.assertTrue(self._gate("2026-11-04 13:45:17", False, 0.0047))   # SK 0.47% afternoon
        self.assertTrue(self._gate("2026-11-04 10:30", True, 0.0047))       # morning triple
        self.assertFalse(self._gate("2026-11-04 10:30", False, 0.0047))     # morning legacy keeps 0.60%
        self.assertFalse(self._gate("2026-11-04 13:54", True, 0.0032))      # BGF 0.32% still below 0.40%

    def test_steps_diagnostic_accepts_triple(self):
        import inspect
        self.assertIn("triple_cross", inspect.signature(r002.gate_steps_diagnostic).parameters)
        self.assertIn("triple_cross", inspect.signature(r005._steps_text).parameters)

    def test_eval_context_default_not_triple(self):
        self.assertIs(r002.BuyEvalContext.__dataclass_fields__["triple_cross"].default, False)


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class TestReentryLimitAfterProfit(unittest.TestCase):
    def test_limit(self):
        with mock.patch.object(r003, "REENTRY_MAX_PER_CODE", 1), mock.patch.object(r003, "REENTRY_MAX_PER_CODE_AFTER_PROFIT", 5):
            self.assertEqual(r003._reentry_limit(None), 1)
            self.assertEqual(r003._reentry_limit({"used": 1}), 1)
            self.assertEqual(r003._reentry_limit({"used": 1, "last_exit_profit": True}), 5)

    def _api(self, code, st):
        api = object.__new__(r003.TradingAPI)
        api.env_dv, api.dry_run = "real", True
        api.live_state = {"positions_meta": {code: {}}, "traded_today": {code}, "reentry": {code: st}}
        api.positions = {code: {"quantity": 2, "buy_price": 190600.0, "buy_time": datetime(2026, 11, 4, 13, 9, 40)}}
        api.pending_orders = {}
        api.buy_inflight_codes, api.startup_position_codes = {code}, set()
        api.trade_lock_until, api.last_sell_fill_price = {}, {}
        api.persist_live_state = lambda *a, **k: None
        return api

    def _sell(self, api, code, fill, qty, remaining):
        pending = {"side": "sell", "quantity": qty, "pre_submit_qty": 2, "reason": "X", "buy_price": 190600.0,
                   "requested_price": fill, "exchange": "KRX"}
        api._confirm_pending_sell(code, pending, {"filled_qty": qty, "avg_price": fill}, remaining)

    def test_profit_exit_sets_flag_and_serializes(self):
        code = "010060"
        st = {"used": 1, "graduated": True, "bot": True, "filled": True}
        api = self._api(code, st)
        self._sell(api, code, 193700.0, 2, 0)  # OCI 13:13 +1.63%
        self.assertTrue(st["last_exit_profit"])
        self.assertEqual(st["pos_net_krw"], 0.0)
        ser = r003._serialize_reentry_state(api.live_state["reentry"])
        self.assertTrue(ser[code]["last_exit_profit"])

    def test_tiny_gross_profit_is_net_loss(self):
        code = "010060"
        st = {"used": 1, "graduated": True}
        api = self._api(code, st)
        self._sell(api, code, 190700.0, 2, 0)  # +0.05% gross < costs
        self.assertFalse(st["last_exit_profit"])

    def test_partial_then_rest_accumulates(self):
        code = "010060"
        st = {"used": 0, "graduated": True}
        api = self._api(code, st)
        self._sell(api, code, 194600.0, 1, 1)   # +2.1% half
        self.assertNotIn("last_exit_profit", st)
        self._sell(api, code, 189900.0, 1, 0)   # -0.37% rest -> total still net positive
        self.assertTrue(st["last_exit_profit"])

    def test_rebalance_reenters_up_to_five_after_profit(self):
        code = "010060"
        api = SimpleNamespace(positions={}, has_buy_exposure=lambda c: False, live_state={"traded_today": set()})
        for used, profit, expect in ((1, False, False), (1, True, True), (4, True, True), (5, True, False)):
            st = {"used": used, "graduated": True, "bot": True, "filled": True, "last_exit_profit": profit}
            active: set = set()
            with mock.patch.object(r003, "REENTRY_MODE", "live"), mock.patch.object(r003, "REENTRY_MAX_PER_CODE", 1), \
                    mock.patch.object(r003, "REENTRY_MAX_PER_CODE_AFTER_PROFIT", 5), \
                    mock.patch.object(r003, "log", lambda *a, **k: None):
                r003._rebalance_active_watchlist(
                    current_dt=datetime(2026, 11, 4, 13, 40), api=api, watch_map={}, active_set=active,
                    backup_pool=collections.deque(), first_active_at={}, hard_stop_today_codes=set(),
                    gap_blocked_codes=set(), warn_state={}, frame_cache={}, reentry_state={code: st})
            self.assertEqual(code in active, expect, (used, profit))


class TestG003LastExitProfit(unittest.TestCase):
    def test_last_exit(self):
        try:
            import g003_trade_simulate_by_date as g003
        except BaseException as exc:  # heavy import / argparse guards
            self.skipTest(f"g003 import failed: {exc}")
        R = g003.TradeRecord
        def rec(action, price, qty, pnl=None, code="010060"):
            r = R(code, "n", action, pd.Timestamp("2026-11-04 13:00"), price, qty, "x", "KRX"); r.pnl_krw = pnl; return r
        log = [rec("BUY", 190600, 2), rec("SELL", 193700, 2, 6200)]
        self.assertTrue(g003._last_exit_net_profit(log, "010060"))
        log += [rec("BUY", 21752, 23, code="059090"), rec("SELL", 21486, 23, -6118, code="059090")]
        self.assertFalse(g003._last_exit_net_profit(log, "059090"))
        self.assertTrue(g003._last_exit_net_profit(log, "010060"))
        log += [rec("BUY", 191000, 2), rec("SELL", 191100, 2, 200)]  # gross +200 < costs
        self.assertFalse(g003._last_exit_net_profit(log, "010060"))
        self.assertFalse(g003._last_exit_net_profit([], "010060"))


if __name__ == "__main__":
    unittest.main()
