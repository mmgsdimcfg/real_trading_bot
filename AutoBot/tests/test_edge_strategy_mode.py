# -*- coding: utf-8 -*-
"""TRADING_STRATEGY_MODE="edge" (Edge-AI-only real trading, 2026-11-12): condition lists per mode, the buy signal
(_e09: regular session, fresh 1m frame, trigger, probability cut, fail-closed) and the sell rules (_e10 / _e11 / _e12)."""
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

import r005_buy_conditions as r005  # noqa: E402
import r006_sell_conditions as r006  # noqa: E402
import r011_edge_ai as r011  # noqa: E402

NOW = datetime(2026, 11, 12, 10, 0, 5)


def bars(n=60, start="2026-11-12 09:01", last_close=None) -> pd.DataFrame:
    """End-labeled 1m bars ending at the bar the bot expects at NOW (10:00), gently rising."""
    idx = pd.date_range(start, periods=n, freq="1min")
    c = 10000.0 + np.arange(n) * 10.0
    if last_close is not None:
        c[-1] = last_close
    return pd.DataFrame({"open": c - 5, "high": c + 10, "low": c - 10, "close": c, "volume": 1000.0}, index=idx)


class ModeListsTest(unittest.TestCase):
    def test_edge_lists_and_precedence(self):
        # AUTO_SELL_MODE="all": the strategy mode's own sell list (the manual-sell modes are covered by test_auto_sell_mode)
        with mock.patch.object(r006, "AUTO_SELL_MODE", "all"):
            with mock.patch.object(r005, "TRADING_STRATEGY_MODE", "edge"), mock.patch.object(r006, "TRADING_STRATEGY_MODE", "edge"):
                for trigger_mode in ("triple", "legacy"):                       # edge wins over the triple trigger mode
                    with mock.patch.object(r005, "ENTRY_TRIGGER_MODE", trigger_mode):
                        self.assertIs(r005.active_buy_conditions(), r005.EDGE_BUY_CONDITIONS)
                self.assertIs(r006.active_sell_conditions(), r006.EDGE_SELL_CONDITIONS)
            self.assertIsNot(r005.active_buy_conditions(), r005.EDGE_BUY_CONDITIONS)      # default stays normal
            self.assertIs(r006.active_sell_conditions(), r006.SELL_CONDITIONS)

    def test_edge_buy_keeps_the_safety_conditions_and_drops_the_strategy_filters(self):
        funcs = [c.func_name for c in r005.EDGE_BUY_CONDITIONS]
        self.assertEqual(funcs[:8], [c.func_name for c in r005.BUY_CONDITIONS[:8]])   # _001.._008 unchanged, same order
        self.assertEqual(funcs[8], "_e09_edge_ai_signal")
        self.assertEqual(funcs[9:], ["_023_affordable_buy_qty", "_024_fresh_live_price", "_025_orderbook_ask_not_thin"])
        nos = [c.no for c in r005.EDGE_BUY_CONDITIONS]
        self.assertEqual(nos, sorted(nos))

    def test_edge_sell_list(self):
        self.assertEqual([c.func_name for c in r006.EDGE_SELL_CONDITIONS],
                         ["_001_stale_live_price_guard", "_002_update_position_tracking", "_e10_edge_take_profit",
                          "_e11_edge_stop_loss", "_e12_edge_time_exit"])
        nos = [c.no for c in r006.EDGE_SELL_CONDITIONS]
        self.assertEqual(nos, sorted(nos))

    def test_defaults_match_the_model_label(self):
        self.assertEqual((r006.EDGE_TP_PCT, r006.EDGE_SL_PCT, r006.EDGE_MAX_HOLD_MINUTES), (0.020, 0.017, 60))

    def test_entry_time_shadow_is_off_in_edge_mode(self):
        with mock.patch.object(r005, "TRADING_STRATEGY_MODE", "edge"):
            self.assertFalse(r005.evaluate_entry_time_shadow(SimpleNamespace(), "x"))


class BuySignalTest(unittest.TestCase):
    def ctx(self, frame=None, session="regular", price=10600.0):
        logs: list[str] = []
        services = SimpleNamespace(
            classify_buy_session=lambda now: session,
            get_frame_1min=lambda code, now, nxt: bars() if frame is None else frame,
            fetch_prev_close=mock.Mock(return_value=9900.0),
            fetch_day_ref_prices=mock.Mock(return_value=(9950.0, 10000.0)),
        )
        c = SimpleNamespace(code="010060", symbol_label="010060_OCI홀딩스", current_dt=NOW, nxt_tradeable=False, price=price,
                            prev_close=None, day_open=None, day_vwap=None, config=None, services=services, log=logs.append,
                            trigger_flag=None, steps_text=None, trigger_reason=None, trigger_info=None, buy_reason=None)
        return c, logs

    def run_signal(self, c, prob=0.6, threshold=0.45, trigger=(True, "1MIN_BB_MID_GOLDEN_CROSS_LOOKBACK"), reason="pass"):
        gate = mock.Mock()
        gate.evaluate.return_value = r011.EdgeAIDecision(True, prob, threshold, True, reason)
        with mock.patch.object(r005, "_edge_mode_gate", return_value=gate), \
                mock.patch.object(r011, "hybrid_trigger_with_context", lambda *a, **k: trigger):
            return r005._e09_edge_ai_signal(c), gate

    def test_buys_when_trigger_and_probability_pass(self):
        c, _ = self.ctx()
        ok, gate = self.run_signal(c)
        self.assertTrue(ok)
        self.assertEqual((c.trigger_flag, c.trigger_reason), ("P", "EDGE_AI_SIGNAL"))
        self.assertIn("EDGE_AI_SIGNAL_p=0.600_thr=0.450_1MIN_BB_MID_GOLDEN_CROSS_LOOKBACK_bar=10:00", c.buy_reason)
        self.assertEqual((c.prev_close, c.day_open, c.day_vwap), (9900.0, 9950.0, 10000.0))     # fetched for the features
        self.assertEqual(gate.evaluate.call_args.kwargs["trigger_info"], c.trigger_info)

    def test_low_probability_is_rejected(self):
        c, logs = self.ctx()
        ok, _ = self.run_signal(c, prob=0.40)
        self.assertFalse(ok)
        self.assertIn("EDGE_SIGNAL_LOW_PROB_p=0.400_LT_0.450", logs[-1])
        self.assertEqual(c.trigger_flag, "-")

    def test_no_model_or_no_probability_never_buys(self):
        c, logs = self.ctx()
        gate = mock.Mock()
        gate.evaluate.return_value = r011.EdgeAIDecision(True, None, None, False, "fail_open:no_model")   # the normal filter would pass
        with mock.patch.object(r005, "_edge_mode_gate", return_value=gate), \
                mock.patch.object(r011, "hybrid_trigger_with_context", lambda *a, **k: (True, "T")):
            self.assertFalse(r005._e09_edge_ai_signal(c))
        self.assertIn("EDGE_SIGNAL_NO_PROB_fail_open:no_model", logs[-1])

    def test_exception_never_buys(self):
        c, logs = self.ctx()
        with mock.patch.object(r005, "_edge_mode_gate", side_effect=RuntimeError("boom")), \
                mock.patch.object(r011, "hybrid_trigger_with_context", lambda *a, **k: (True, "T")):
            self.assertFalse(r005._e09_edge_ai_signal(c))
        self.assertIn("EDGE_SIGNAL_ERROR_RuntimeError: boom", logs[-1])

    def test_rejects_without_a_trigger_outside_the_regular_session_and_on_a_stale_frame(self):
        c, logs = self.ctx()
        ok, gate = self.run_signal(c, trigger=(False, "NO_CROSS"))
        self.assertFalse(ok)
        self.assertIn("EDGE_SIGNAL_NO_TRIGGER | NO_CROSS", logs[-1])
        gate.evaluate.assert_not_called()
        c, logs = self.ctx(session="nxt_pre")
        self.assertFalse(self.run_signal(c)[0])
        self.assertIn("EDGE_SIGNAL_REGULAR_SESSION_ONLY", logs[-1])
        c, logs = self.ctx(frame=bars(55))                                   # last bar 09:55, expected 10:00
        self.assertFalse(self.run_signal(c)[0])
        self.assertIn("EDGE_SIGNAL_1MIN_FRAME_STALE", logs[-1])
        c, logs = self.ctx(frame=None)
        c.services.get_frame_1min = lambda *a: None
        self.assertFalse(self.run_signal(c)[0])
        self.assertIn("EDGE_SIGNAL_1MIN_FRAME_UNAVAILABLE", logs[-1])

    def test_rejects_when_the_bar_or_the_live_price_is_below_the_1m_mid(self):
        c, logs = self.ctx(frame=bars(last_close=9000.0), price=9000.0)
        self.assertFalse(self.run_signal(c)[0])
        self.assertIn("EDGE_SIGNAL_NO_TRIGGER_BELOW_1MIN_MID", logs[-1])
        c, logs = self.ctx(price=10000.0)                                    # bar closed above the mid, live price fell under it
        self.assertFalse(self.run_signal(c)[0])
        self.assertIn("EDGE_SIGNAL_LIVE_BELOW_1MIN_MID", logs[-1])

    def test_the_gate_of_edge_mode_is_always_enforced_with_the_edge_cut(self):
        with mock.patch.object(r005, "_EDGE_MODE_GATE", None), mock.patch.object(r005, "EDGE_ONLY_MIN_PROB", 0.5):
            gate = r005._edge_mode_gate(None)
            self.assertEqual((gate.mode, gate.min_prob_override), ("gate", 0.5))
            self.assertIs(r005._edge_mode_gate(None), gate)


class _Api:
    def __init__(self, ok=True):
        self.orders, self.ok = [], ok

    def place_sell_order(self, code, qty, now, reason, nxt, price=None, code_name=None):
        self.orders.append((qty, reason))
        return self.ok


class SellRulesTest(unittest.TestCase):
    ENTRY = 10000.0

    def ctx(self, price, held_min=5.0, api=None):
        api = api or _Api()
        logs: list[str] = []
        c = SimpleNamespace(
            code="010060", name="OCI홀딩스", symbol_label="010060_OCI홀딩스", current_dt=NOW, nxt_tradeable=False,
            pos={"quantity": 7, "entry_quantity": 7, "buy_time": NOW - timedelta(minutes=held_min)}, price=price, api=api,
            date_str="20261112", log=logs.append, entry_price=self.ENTRY, pnl_pct=price / self.ENTRY - 1,
            peak_pnl_pct=max(0.0, price / self.ENTRY - 1), bar_time=None, state=SimpleNamespace(signal_sell_bar={}))
        return c, api, logs

    def test_take_profit_sells_everything_at_two_percent(self):
        c, api, logs = self.ctx(10199.0)
        self.assertFalse(r006._e10_edge_take_profit(c))
        c, api, logs = self.ctx(10200.0)                                     # exactly +2.0% (float noise must not miss it)
        self.assertTrue(r006._e10_edge_take_profit(c))
        self.assertEqual(api.orders, [(7, "EDGE_TP_2.0PCT")])
        self.assertTrue(any("[SELL TRIGGER]" in m and "EDGE_TP_2.0PCT" in m for m in logs))

    def test_stop_loss_sells_everything_at_minus_one_point_seven(self):
        c, api, _ = self.ctx(9831.0)
        self.assertFalse(r006._e11_edge_stop_loss(c))
        c, api, _ = self.ctx(9830.0)
        self.assertTrue(r006._e11_edge_stop_loss(c))
        self.assertEqual(api.orders, [(7, "EDGE_SL_1.7PCT")])

    def test_time_exit_after_sixty_minutes(self):
        c, api, _ = self.ctx(10050.0, held_min=59.5)
        self.assertFalse(r006._e12_edge_time_exit(c))
        c, api, _ = self.ctx(10050.0, held_min=60.0)
        self.assertTrue(r006._e12_edge_time_exit(c))
        self.assertEqual(api.orders, [(7, "EDGE_TIME_60MIN")])

    def test_a_failed_order_still_stops_the_tick_and_is_retried_next_tick(self):
        c, api, _ = self.ctx(10300.0, api=_Api(ok=False))
        self.assertTrue(r006._e10_edge_take_profit(c))
        self.assertTrue(r006._e10_edge_take_profit(c))
        self.assertEqual(len(api.orders), 2)

    def test_order_of_the_rules_inside_the_list(self):
        # a position that is both past its time limit and in profit beyond the target takes the profit rule first
        c, api, _ = self.ctx(10300.0, held_min=120.0)
        for cond in r006.EDGE_SELL_CONDITIONS[2:]:
            if cond.run(c):
                break
        self.assertEqual(api.orders, [(7, "EDGE_TP_2.0PCT")])


class PaperSkipTest(unittest.TestCase):
    def test_paper_runner_is_not_started_while_the_real_mode_is_edge(self):
        import r003_trade_live_execute as r003
        logs: list[str] = []
        with mock.patch.object(r003, "EDGE_ONLY_MODE", "paper"), mock.patch.object(r003, "TRADING_STRATEGY_MODE", "edge"), \
                mock.patch.object(r003, "log", logs.append):
            self.assertIsNone(r003._make_edge_only_runner("20261112"))
        self.assertIn("paper mode skipped", logs[-1])
        with mock.patch.object(r003, "EDGE_ONLY_MODE", "off"):
            self.assertIsNone(r003._make_edge_only_runner("20261112"))


if __name__ == "__main__":
    unittest.main()
