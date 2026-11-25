# -*- coding: utf-8 -*-
"""[2026-11-25] the control tab's 청산 기준 as bounds for rule-strategy positions (r006 _c03_common_exit_bounds, user decision:
the existing rule exits stay, a set bound that is reached first sells all that is left)."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import r006_sell_conditions as r006  # noqa: E402

NOW = datetime(2026, 11, 25, 10, 0, 0)


def ctx(pnl, stale=False, qty=10, regular=True, accepted=True, marked=None):
    api = mock.Mock()
    api.place_sell_order.return_value = accepted
    return SimpleNamespace(code="005930", name="삼성전자", symbol_label="005930", current_dt=NOW, nxt_tradeable=False,
                           pos={"quantity": qty}, price=10_000.0, price_source="live", entry_price=10_000.0, pnl_pct=pnl,
                           peak_pnl_pct=max(pnl, 0.0), bar_time=NOW, api=api, log=lambda m: None,
                           state=SimpleNamespace(signal_sell_bar={} if marked is None else {"005930": marked}),
                           services=SimpleNamespace(is_stale_live_price_source=lambda src: stale,
                                                    is_regular_session=lambda now: regular))


class CommonExitTest(unittest.TestCase):
    def run_it(self, c, tp=0.0, sl=0.0, hold=0, held_min=5.0):
        with mock.patch.object(r006, "EDGE_ONLY_TP_PCT", tp), mock.patch.object(r006, "EDGE_ONLY_SL_PCT", sl), \
                mock.patch.object(r006, "EDGE_ONLY_HORIZON_MIN", hold), \
                mock.patch.object(r006, "_hold_info", return_value=(NOW, None, held_min * 60.0)):
            done = r006._c03_common_exit_bounds(c)
        reason = c.api.place_sell_order.call_args[0][3] if c.api.place_sell_order.called else None
        return done, reason

    def test_nothing_set_means_no_bound(self):
        self.assertEqual(self.run_it(ctx(-0.09)), (False, None))
        self.assertEqual(self.run_it(ctx(0.09), held_min=900), (False, None))

    def test_a_set_bound_that_is_reached_sells_all_that_is_left(self):
        self.assertEqual(self.run_it(ctx(0.0199), tp=2.0, sl=1.5, hold=60), (False, None))
        c = ctx(0.02, qty=7)
        self.assertEqual(self.run_it(c, tp=2.0, sl=1.5, hold=60), (True, "COMMON_TP_2.0PCT"))
        self.assertEqual(c.api.place_sell_order.call_args[0][1], 7)
        self.assertEqual(self.run_it(ctx(-0.015), tp=2.0, sl=1.5, hold=60), (True, "COMMON_SL_1.5PCT"))
        self.assertEqual(self.run_it(ctx(0.001), tp=2.0, sl=1.5, hold=60, held_min=61), (True, "COMMON_TIME_60MIN"))
        # only what is set: a stop of 1.77 alone leaves take profit and time to the rule exits
        self.assertEqual(self.run_it(ctx(0.08), sl=1.77, held_min=900), (False, None))
        self.assertEqual(self.run_it(ctx(-0.0177), sl=1.77), (True, "COMMON_SL_1.77PCT"))

    def test_not_judged_on_a_stale_price_or_outside_the_regular_session(self):
        self.assertEqual(self.run_it(ctx(-0.09, stale=True), tp=2.0, sl=1.5, hold=60), (False, None))
        # Codex review: in the NXT hours the quote can be the KRX close - no full sale on it
        self.assertEqual(self.run_it(ctx(-0.09, regular=False), tp=2.0, sl=1.5, hold=60, held_min=900), (False, None))

    def test_an_order_that_is_not_accepted_lets_the_rule_exits_go_on(self):
        # Codex review: cooldown / rejection - the conditions after this one are still judged, the bar is not marked as sold
        c = ctx(-0.02, accepted=False)
        self.assertEqual(self.run_it(c, sl=1.5), (False, "COMMON_SL_1.5PCT"))
        self.assertEqual(c.state.signal_sell_bar, {})
        earlier = datetime(2026, 11, 25, 9, 57)
        c = ctx(-0.02, accepted=False, marked=earlier)
        self.run_it(c, sl=1.5)
        self.assertEqual(c.state.signal_sell_bar, {"005930": earlier})
        c = ctx(-0.02)
        self.run_it(c, sl=1.5)
        self.assertEqual(c.state.signal_sell_bar, {"005930": NOW})                           # accepted: marked

    def test_the_bound_is_in_the_rule_lists_only_with_the_full_auto_sell(self):
        def names(mode, auto="all", engine="off"):
            with mock.patch.object(r006, "TRADING_STRATEGY_MODE", mode), mock.patch.object(r006, "auto_sell_mode", return_value=auto), \
                    mock.patch.object(r006, "exit_engine_mode", return_value=engine):
                return [c.func_name for c in r006.active_sell_conditions()]
        for mode in ("normal", "basic"):
            got = names(mode)
            self.assertEqual(got.count("_c03_common_exit_bounds"), 1)
            self.assertIn("track", got[got.index("_c03_common_exit_bounds") - 1], got)       # right after the position tracking
            got = names(mode, engine="live")
            self.assertEqual(got, ["_x02_track_if_fresh", "_x10_adaptive_exit", "_c03_common_exit_bounds"])   # after the engine
        self.assertNotIn("_c03_common_exit_bounds", names("edge"))                           # the edge strategy has its own exits
        self.assertNotIn("_c03_common_exit_bounds", names("normal", auto="stop_only"))
        self.assertNotIn("_c03_common_exit_bounds", names("normal", auto="off"))
        self.assertEqual(len({c.no for c in r006._with_common_exit(r006.SELL_CONDITIONS)}),
                         len(r006._with_common_exit(r006.SELL_CONDITIONS)))                  # its number is its own


if __name__ == "__main__":
    unittest.main()
