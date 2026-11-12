# -*- coding: utf-8 -*-
"""AUTO_SELL_MODE (2026-11-12: buys automatic, sells manual, only the net -2% stop automatic) and the golden-cross
volatility gate of buy condition _012 (1-minute ATR%)."""
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
import r006_sell_conditions as r006  # noqa: E402

NOW = datetime(2026, 11, 12, 10, 0, 5)
ENTRY = 10000.0


class _Api:
    def __init__(self, ok=True):
        self.ok, self.orders = ok, []

    def place_sell_order(self, code, qty, now, reason, nxt, price=None, code_name="", market_order=False, limit_price=None):
        self.orders.append((qty, reason, market_order))
        return self.ok


def sell_ctx(price, api=None):
    api = api or _Api()
    logs: list[str] = []
    c = SimpleNamespace(
        code="010060", name="OCI홀딩스", symbol_label="010060_OCI홀딩스", current_dt=NOW, nxt_tradeable=False,
        pos={"quantity": 7, "entry_quantity": 7, "buy_time": NOW - timedelta(minutes=5)}, price=price, api=api,
        date_str="20261112", log=logs.append, entry_price=ENTRY, pnl_pct=price / ENTRY - 1, bar_time=None,
        state=SimpleNamespace(signal_sell_bar={}, hard_stop_today_codes=set()), risk=SimpleNamespace(hard_stop_daily_count=0))
    return c, api, logs


class ModeListTest(unittest.TestCase):
    def test_default_is_stop_only(self):
        self.assertEqual(r001.AUTO_SELL_MODE, "stop_only")
        self.assertAlmostEqual(r001.AUTO_SELL_STOP_NET_PCT, 0.02)
        self.assertIs(r006.active_sell_conditions(), r006.STOP_ONLY_SELL_CONDITIONS)

    def test_mode_wins_over_every_strategy_mode(self):
        for strategy in ("normal", "basic", "edge"):
            with mock.patch.object(r006, "TRADING_STRATEGY_MODE", strategy):
                with mock.patch.object(r006, "AUTO_SELL_MODE", "stop_only"):
                    self.assertIs(r006.active_sell_conditions(), r006.STOP_ONLY_SELL_CONDITIONS)
                with mock.patch.object(r006, "AUTO_SELL_MODE", "off"):
                    self.assertIs(r006.active_sell_conditions(), r006.MANUAL_SELL_CONDITIONS)
        with mock.patch.object(r006, "AUTO_SELL_MODE", "all"):
            self.assertIs(r006.active_sell_conditions(), r006.SELL_CONDITIONS)
        with mock.patch.object(r006, "AUTO_SELL_MODE", "typo"):          # an unknown value keeps the full automatic sells
            self.assertIs(r006.active_sell_conditions(), r006.SELL_CONDITIONS)

    def test_lists_hold_no_other_exit(self):
        self.assertEqual([c.func_name for c in r006.STOP_ONLY_SELL_CONDITIONS],
                         ["_001_stale_live_price_guard", "_002_update_position_tracking", "_s04_net_stop_loss"])
        self.assertEqual([c.kind for c in r006.MANUAL_SELL_CONDITIONS], ["GUARD", "UPDATE"])     # no EXIT: the bot never sells


class NetStopTest(unittest.TestCase):
    def test_net_pnl_includes_both_fees_and_the_sell_tax(self):
        self.assertAlmostEqual(r006.net_pnl_pct(10000.0, 10000.0) * 100, -0.23, places=2)   # 0.015 x 2 + 0.20
        self.assertEqual(r006.net_pnl_pct(0.0, 10000.0), 0.0)

    def test_sells_everything_at_market_when_the_net_loss_reaches_two_percent(self):
        c, api, _ = sell_ctx(9823.0)                                     # price -1.77%, net -1.996%
        self.assertFalse(r006._s04_net_stop_loss(c))
        self.assertEqual(api.orders, [])
        c, api, logs = sell_ctx(9822.0)                                  # net -2.006%
        self.assertTrue(r006._s04_net_stop_loss(c))
        self.assertEqual(api.orders, [(7, "NET_STOP_LOSS_2.0PCT", True)])
        self.assertTrue(any("[SELL TRIGGER]" in m and "NET_STOP_LOSS_2.0PCT" in m for m in logs))
        self.assertIn("010060", c.state.hard_stop_today_codes)

    def test_profit_never_sells(self):
        for price in (10000.0, 10500.0, 13000.0):
            c, api, _ = sell_ctx(price)
            self.assertFalse(r006._s04_net_stop_loss(c))
            self.assertEqual(api.orders, [])

    def test_a_failed_order_is_retried_on_the_next_tick(self):
        c, api, _ = sell_ctx(9700.0, api=_Api(ok=False))
        self.assertTrue(r006._s04_net_stop_loss(c))
        self.assertTrue(r006._s04_net_stop_loss(c))
        self.assertEqual(len(api.orders), 2)
        self.assertEqual(c.state.hard_stop_today_codes, set())


class ScheduledLiquidationTest(unittest.TestCase):
    def _run(self, mode):
        import r003_trade_live_execute as r003
        api = mock.Mock()
        api.get_open_positions.return_value = {"010060": {"quantity": 7, "buy_price": 10000.0, "current_price": 10100.0}}
        api.has_pending_order.return_value = False
        logs: list[str] = []
        state = {"date": None, "done_1520": False, "done_1959": False}
        with mock.patch.object(r003, "AUTO_SELL_MODE", mode), mock.patch.object(r003, "log", logs.append), \
                mock.patch.object(r003, "ENABLE_NXT_SESSION", False), mock.patch.object(r003, "_is_today_buy_position", return_value=True):
            for _ in range(2):
                r003.run_scheduled_liquidations(datetime(2026, 11, 12, 15, 21, 0), api, {}, {"010060": "OCI홀딩스"}, state, "20261112", {"010060"})
            pending = r003._nxt_force_exit_pending(datetime(2026, 11, 12, 19, 59, 59), state)
        return api, logs, pending

    def test_no_scheduled_sell_unless_all(self):
        for mode in ("stop_only", "off"):
            api, logs, _ = self._run(mode)
            api.place_sell_order.assert_not_called()
            self.assertEqual(sum("[AUTO SELL   ]" in m for m in logs), 1)      # said once a day

    def test_all_keeps_the_scheduled_sell(self):
        api, logs, _ = self._run("all")
        self.assertEqual(api.place_sell_order.call_count, 1)

    def test_the_nxt_close_wait_never_holds_the_watch_loop_in_manual_modes(self):
        import r003_trade_live_execute as r003
        state = {"done_1959": False}
        late = datetime(2026, 11, 12, 19, 59, 59)
        with mock.patch.object(r003, "ENABLE_NXT_SESSION", True):
            with mock.patch.object(r003, "AUTO_SELL_MODE", "all"):
                self.assertTrue(r003._nxt_force_exit_pending(late, state))
            for mode in ("stop_only", "off"):
                with mock.patch.object(r003, "AUTO_SELL_MODE", mode):
                    self.assertFalse(r003._nxt_force_exit_pending(late, state))


def frame(atr, close=10000.0, with_atr=True):
    idx = pd.date_range("2026-11-12 09:01", periods=30, freq="1min")
    d = pd.DataFrame({"open": close, "high": close + atr / 2, "low": close - atr / 2, "close": close, "volume": 1000.0}, index=idx)
    if with_atr:
        d["ATR"] = atr
    return d


class VolatilityGateTest(unittest.TestCase):
    def test_atr_percent_of_the_last_bar(self):
        self.assertAlmostEqual(r005.atr_1min_pct(frame(50.0)), 0.5)
        self.assertIsNone(r005.atr_1min_pct(None))
        self.assertIsNone(r005.atr_1min_pct(frame(50.0).iloc[:1]))
        self.assertIsNone(r005.atr_1min_pct(frame(float("nan"))))
        got = r005.atr_1min_pct(frame(50.0, with_atr=False))              # no ATR column: computed from the bars
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, 0.5, places=1)

    def _gate(self, fr, enabled=True):
        rejects: list[str] = []
        ctx = SimpleNamespace(code="010060", current_dt=NOW, nxt_tradeable=False, services=SimpleNamespace(get_frame_1min=lambda *_: fr))
        with mock.patch.object(r005, "ENABLE_GC_VOLATILITY_GATE", enabled), mock.patch.object(r005, "GC_VOLATILITY_MIN_ATR_1MIN_PCT", 0.5), \
                mock.patch.object(r005, "_reject_hybrid", lambda _c, reason: rejects.append(reason)), \
                mock.patch.object(r005, "_context_gate", lambda _c, name: True):
            return r005._012_min_entry_atr_volatility(ctx), rejects

    def test_blocks_a_quiet_symbol_and_passes_a_volatile_one(self):
        ok, rejects = self._gate(frame(49.0))
        self.assertFalse(ok)
        self.assertEqual(rejects, ["GC_VOL_LOW_1MIN_ATR_0.490%_LT_0.500%"])
        self.assertEqual(self._gate(frame(50.0)), (True, []))             # exactly at the limit passes
        self.assertEqual(self._gate(frame(80.0)), (True, []))

    def test_unknown_atr_never_buys(self):
        ok, rejects = self._gate(None)
        self.assertFalse(ok)
        self.assertEqual(rejects, ["GC_VOL_1MIN_ATR_UNAVAILABLE"])

    def test_switched_off_falls_through_to_the_3min_gate(self):
        self.assertEqual(self._gate(frame(1.0), enabled=False), (True, []))

    def test_defaults(self):
        self.assertTrue(r001.ENABLE_GC_VOLATILITY_GATE)
        self.assertAlmostEqual(r001.GC_VOLATILITY_MIN_ATR_1MIN_PCT, 0.5)


if __name__ == "__main__":
    unittest.main()
