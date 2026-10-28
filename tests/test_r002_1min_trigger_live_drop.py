# -*- coding: utf-8 -*-
"""1-min trigger live-price guard (2026-10-28, 091580 상신이디피 09:35 case).

r002 check_1min_trigger_live_guard + r005 _008 wiring (guard runs after the trigger-age timer, so a guard reject
must not reset the confirmed-bar trigger lifetime)."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r002_strategy_core_shared as r002  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402


def _frame_1min(close: float = 27850.0, bb_mid: float = 27800.0) -> pd.DataFrame:
    idx = pd.date_range("2026-10-28 09:33", periods=2, freq="1min")
    return pd.DataFrame(
        {"open": [27500.0, 27750.0], "high": [27600.0, 27900.0], "low": [27450.0, 27700.0],
         "close": [27550.0, close], "volume": [1000, 2000], "BB_MIDDLE": [27600.0, bb_mid]},
        index=idx,
    )


class TestLiveGuard(unittest.TestCase):
    def test_none_or_invalid_price_passes(self):
        for bad in (None, 0.0, -1.0, float("nan")):
            self.assertEqual(r002.check_1min_trigger_live_guard(_frame_1min(), bad), (True, ""), bad)

    def test_unchanged_or_rising_passes(self):
        self.assertTrue(r002.check_1min_trigger_live_guard(_frame_1min(), 27850.0)[0])
        self.assertTrue(r002.check_1min_trigger_live_guard(_frame_1min(), 27900.0)[0])

    def test_091580_drop_rejected(self):
        # 09:34:57: confirmed close 27,850 -> live 27,650 (-0.72%)
        ok, reason = r002.check_1min_trigger_live_guard(_frame_1min(), 27650.0)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("1MIN_LIVE_DROP_0.72%"), reason)

    def test_boundary_exactly_threshold_passes(self):
        # BB mid far below so only the drop rule applies: 10,000 -> 9,960 = exactly 0.4% (not "> 0.4%")
        frame = _frame_1min(close=10000.0, bb_mid=9000.0)
        self.assertTrue(r002.check_1min_trigger_live_guard(frame, 9960.0)[0])
        ok, reason = r002.check_1min_trigger_live_guard(frame, 9959.0)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("1MIN_LIVE_DROP_"), reason)

    def test_live_at_or_below_bb_mid_rejected(self):
        # -0.18% (within 0.4%) but live <= 1-min BB mid 27,800
        self.assertEqual(r002.check_1min_trigger_live_guard(_frame_1min(), 27800.0), (False, "1MIN_LIVE_BELOW_BB_MID"))

    def test_disabled_when_zero(self):
        with mock.patch.object(r002, "HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT", 0.0):
            self.assertEqual(r002.check_1min_trigger_live_guard(_frame_1min(), 20000.0), (True, ""))

    def test_missing_indicator_passes(self):
        frame = _frame_1min()
        frame["BB_MIDDLE"] = float("nan")
        self.assertTrue(r002.check_1min_trigger_live_guard(frame, 27000.0)[0])


class TestR005Wiring(unittest.TestCase):
    """_008: guard reject clears the confirm state but keeps the 1-min trigger age timer."""

    def _ctx(self, price: float, now: datetime, state: SimpleNamespace, logs: list):
        frame_1min = _frame_1min()
        services = SimpleNamespace(
            get_frame_1min=lambda code, dt, nxt: frame_1min,
            format_reject_detail=lambda reason, *a, **k: reason,
            log=logs.append, record_gate_decision=None,
        )
        frame3 = pd.DataFrame({"close": [1.0, 1.0], "BB_MIDDLE": [float("nan")] * 2},
                              index=pd.date_range("2026-10-28 09:30", periods=2, freq="3min"))
        ctx = SimpleNamespace(
            code="091580", norm_code="091580", symbol_label="091580_상신이디피", price=price, current_dt=now,
            nxt_tradeable=False, services=services, state=state, buy_frame=frame3, prev_bar=frame3.iloc[-2],
            cross_info={}, bar_time=pd.Timestamp("2026-10-28 09:33"), trigger_flag="", steps_text="",
            trigger_reason="", trigger_info=None,
        )
        ctx.log = logs.append
        return ctx

    def test_guard_reject_keeps_timer_and_clears_confirm(self):
        state = SimpleNamespace(buy_trigger_age_state={}, buy_confirm_state={"091580": {"count": 1}})
        logs: list = []
        patches = [
            mock.patch.object(r005, "ENTRY_1MIN_FRAME_REQUIRE_LATEST", False),
            mock.patch.object(r005, "_steps_text", lambda ctx: ""),
            mock.patch.object(r005, "check_buy_condition_1min_hybrid_trigger",
                              lambda *a, **k: (True, "1MIN_UPTREND_CONTINUATION")),
        ]
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)

        t0 = datetime(2026, 10, 28, 9, 34, 28)
        self.assertTrue(r005._008_hybrid_1min_trigger(self._ctx(27850.0, t0, state, logs)))
        start = state.buy_trigger_age_state["091580"]["start"]

        t1 = datetime(2026, 10, 28, 9, 34, 57)
        self.assertFalse(r005._008_hybrid_1min_trigger(self._ctx(27650.0, t1, state, logs)))
        self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_LIVE_DROP_0.72%", logs[-1])
        self.assertNotIn("091580", state.buy_confirm_state)
        self.assertEqual(state.buy_trigger_age_state["091580"]["start"], start)  # lifetime not reset


if __name__ == "__main__":
    unittest.main()
