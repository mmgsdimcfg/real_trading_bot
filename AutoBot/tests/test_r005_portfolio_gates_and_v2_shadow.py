# -*- coding: utf-8 -*-
"""Tests for r005 portfolio guards (_025~_027, default OFF) and the r010 V2 shadow runner."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "tests"))

import r005_buy_conditions as r005  # noqa: E402
from r008_strategy_v2 import V2Config  # noqa: E402
from r010_v2_shadow import V2ShadowRunner  # noqa: E402


def _ctx(realized=0.0, traded=(), exposure=(), bid=10000, ask=10010, price=10000.0):
    logs = []
    api = SimpleNamespace(live_state={"realized_pnl_krw": realized}, has_buy_exposure=lambda c: c in exposure)
    services = SimpleNamespace(get_order_spec=lambda now, nxt: {"exchange": "KRX"},
                               fetch_bid_ask=lambda code, mkt: (bid, ask))
    return SimpleNamespace(api=api, services=services, log=logs.append, symbol_label="X", norm_code="000001", price=price,
                           current_dt=datetime(2026, 10, 23, 10), nxt_tradeable=False,
                           code="000001", state=SimpleNamespace(traded_today=set(traded), buy_confirm_state={})), logs


class TestPortfolioGates(unittest.TestCase):
    def test_flags_off_pass_everything(self):
        # [2026-11-04] 기본값이 ON으로 바뀜(사용자 결정) - OFF일 때 모두 통과하는지만 검증
        ctx, _ = _ctx(realized=-1e9, traded={"1", "2", "3", "4", "5", "6", "7"}, exposure={"1", "2", "3", "4", "5", "6", "7"},
                      bid=10000, ask=11000)
        with mock.patch.object(r005, "ENABLE_DAILY_LOSS_LIMIT_GATE", False), \
                mock.patch.object(r005, "ENABLE_MAX_OPEN_POSITIONS_GATE", False), \
                mock.patch.object(r005, "ENABLE_SPREAD_GATE", False), \
                mock.patch.object(r005, "ENABLE_TICK_COST_GATE", False):
            ctx.price = 2000.0      # tick 5 = 0.25%: passes only because the gate is off
            self.assertTrue(r005._007_daily_loss_limit(ctx))
            self.assertTrue(r005._008_max_open_positions(ctx))
            self.assertTrue(r005._021_entry_spread(ctx))

    def test_defaults_are_on(self):
        import r001_define_config as r001
        self.assertTrue(r001.ENABLE_DAILY_LOSS_LIMIT_GATE)
        self.assertTrue(r001.ENABLE_MAX_OPEN_POSITIONS_GATE)
        self.assertTrue(r001.ENABLE_SPREAD_GATE)
        self.assertTrue(r001.ENABLE_TICK_COST_GATE)
        self.assertEqual(r001.TICK_COST_MAX_PCT, 0.13)
        self.assertEqual(r001.EDGE_AI_MODE, "gate")

    def test_daily_loss_limit(self):
        with mock.patch.object(r005, "ENABLE_DAILY_LOSS_LIMIT_GATE", True), mock.patch.object(r005, "DAILY_LOSS_LIMIT_KRW", 100_000):
            self.assertTrue(r005._007_daily_loss_limit(_ctx(realized=-99_999)[0]))
            ctx, logs = _ctx(realized=-100_000)
            self.assertFalse(r005._007_daily_loss_limit(ctx))
            self.assertIn("DAILY_LOSS_LIMIT", logs[0])

    def test_max_open_positions_counts_only_bot_exposure(self):
        with mock.patch.object(r005, "ENABLE_MAX_OPEN_POSITIONS_GATE", True), mock.patch.object(r005, "MAX_OPEN_POSITIONS", 2):
            # 3 codes traded today, but only 1 still exposed -> allowed
            self.assertTrue(r005._008_max_open_positions(_ctx(traded={"a", "b", "c"}, exposure={"a"})[0]))
            self.assertFalse(r005._008_max_open_positions(_ctx(traded={"a", "b"}, exposure={"a", "b"})[0]))
            # the candidate itself is not counted
            self.assertTrue(r005._008_max_open_positions(_ctx(traded={"a", "000001"}, exposure={"a", "000001"})[0]))

    def test_spread_gate_fail_open_on_query_exception(self):
        # [2026-11-04 Codex 2차 #3] API 예외는 통과(fail-open), 정상 응답의 무호가는 차단
        def boom(code, mkt):
            raise RuntimeError("quote api down")
        with mock.patch.object(r005, "ENABLE_SPREAD_GATE", True), mock.patch.object(r005, "MAX_ENTRY_SPREAD_TICKS", 2):
            ctx, logs = _ctx()
            ctx.services.fetch_bid_ask = boom
            self.assertTrue(r005._021_entry_spread(ctx))
            self.assertIn("fail-open", logs[-1])
            ctx, logs = _ctx(bid=None, ask=None)
            self.assertFalse(r005._021_entry_spread(ctx))
            self.assertIn("SPREAD_NO_QUOTE", logs[-1])

    def test_spread_gate(self):
        with mock.patch.object(r005, "ENABLE_SPREAD_GATE", True), mock.patch.object(r005, "MAX_ENTRY_SPREAD_TICKS", 2):
            self.assertTrue(r005._021_entry_spread(_ctx(bid=10000, ask=10020)[0]))
            self.assertFalse(r005._021_entry_spread(_ctx(bid=10000, ask=10030)[0]))
            self.assertFalse(r005._021_entry_spread(_ctx(bid=None, ask=10030)[0]))

    def test_tick_cost_pct_at_every_tick_boundary(self):
        for price, tick in ((1999, 1), (2000, 5), (4995, 5), (5000, 10), (19990, 10), (20000, 50), (49950, 50),
                            (50000, 100), (199900, 100), (200000, 500), (499500, 500), (500000, 1000)):
            self.assertAlmostEqual(r005.tick_cost_pct(price), tick / price * 100.0, places=12, msg=str(price))
        for bad in (0, -1, float("nan"), float("inf"), None, "x"):
            self.assertIsNone(r005.tick_cost_pct(bad))

    def test_tick_cost_gate(self):
        with mock.patch.object(r005, "ENABLE_TICK_COST_GATE", True), mock.patch.object(r005, "TICK_COST_MAX_PCT", 0.13), \
                mock.patch.object(r005, "ENABLE_SPREAD_GATE", False):
            for price, ok in ((10000, True),        # tick 10 = 0.10%
                              (7700, True),         # 0.1299%
                              (7690, False),        # 0.1300..4% > 0.13
                              (5000, False),        # tick 10 = 0.20%
                              (4995, True),         # tick 5 = 0.1001%
                              (3840, False),        # tick 5 = 0.1302%
                              (20000, False),       # tick 50 = 0.25%
                              (19990, True),        # tick 10 = 0.05%
                              (50000, False),       # tick 100 = 0.20%
                              (77000, True),        # tick 100 = 0.1299%
                              ):
                ctx, logs = _ctx(price=price)
                self.assertEqual(r005._021_entry_spread(ctx), ok, msg=str(price))
                if not ok:
                    self.assertIn("TICK_COST_TOO_HIGH", logs[-1])
            # exactly at the limit passes: tick 10 / 0.13% -> price 7692.3..., use a synthetic limit instead
            with mock.patch.object(r005, "TICK_COST_MAX_PCT", 0.1):
                self.assertTrue(r005._021_entry_spread(_ctx(price=10000)[0]))     # 0.1000% == limit
                self.assertFalse(r005._021_entry_spread(_ctx(price=9990)[0]))

    def test_tick_cost_gate_passes_a_missing_price_and_runs_before_the_quote_query(self):
        calls = []

        def quote(code, mkt):
            calls.append(code)
            return 10000, 10010
        with mock.patch.object(r005, "ENABLE_TICK_COST_GATE", True), mock.patch.object(r005, "TICK_COST_MAX_PCT", 0.13), \
                mock.patch.object(r005, "ENABLE_SPREAD_GATE", True), mock.patch.object(r005, "MAX_ENTRY_SPREAD_TICKS", 2):
            for bad in (0.0, float("nan"), None):
                ctx, _ = _ctx(price=bad)
                ctx.services.fetch_bid_ask = quote
                self.assertTrue(r005._021_entry_spread(ctx))            # fail-open, the spread check still runs
            self.assertEqual(len(calls), 3)
            ctx, _ = _ctx(price=20000)
            ctx.services.fetch_bid_ask = quote
            self.assertFalse(r005._021_entry_spread(ctx))
            self.assertEqual(len(calls), 3)                             # blocked before the quote query

    def test_tick_cost_rejection_clears_the_poll_confirmation(self):
        # pass -> blocked poll -> pass must not count as two consecutive passes (_022)
        with mock.patch.object(r005, "ENABLE_TICK_COST_GATE", True), mock.patch.object(r005, "TICK_COST_MAX_PCT", 0.13), \
                mock.patch.object(r005, "ENABLE_SPREAD_GATE", False):
            ctx, _ = _ctx(price=7700)
            ctx.state.buy_confirm_state["000001"] = {"count": 1}
            ctx.state.buy_confirm_state["000002"] = {"count": 1}
            self.assertTrue(r005._021_entry_spread(ctx))
            self.assertIn("000001", ctx.state.buy_confirm_state)        # a pass keeps it
            ctx.price = 7690
            self.assertFalse(r005._021_entry_spread(ctx))
            self.assertNotIn("000001", ctx.state.buy_confirm_state)
            self.assertIn("000002", ctx.state.buy_confirm_state)        # other symbols untouched

    def test_tick_cost_constants_can_be_overridden_live(self):
        import r001_define_config as r001
        self.assertTrue(r001.live_override_key_allowed("ENABLE_TICK_COST_GATE", r001.ENABLE_TICK_COST_GATE))
        self.assertTrue(r001.live_override_key_allowed("TICK_COST_MAX_PCT", r001.TICK_COST_MAX_PCT))

    def test_tick_cost_defines_are_exported_with_the_condition(self):
        cond = next(c for c in r005.BUY_CONDITIONS if c.func_name == "_021_entry_spread")
        self.assertIn("ENABLE_TICK_COST_GATE", cond.related_defines)
        self.assertIn("TICK_COST_MAX_PCT", cond.related_defines)

    def test_registration_order(self):
        names = [c.func_name for c in r005.BUY_CONDITIONS]
        self.assertNotIn("_007_stoploss_circuit_breaker", names)  # removed 2026-11-04
        self.assertLess(names.index("_006_hard_stop_reentry_block"), names.index("_007_daily_loss_limit"))
        self.assertLess(names.index("_008_max_open_positions"), names.index("_009_hybrid_1min_trigger"))
        self.assertLess(names.index("_021_entry_spread"), names.index("_022_consecutive_poll_confirm"))
        self.assertEqual(len({c.no for c in r005.BUY_CONDITIONS}), len(r005.BUY_CONDITIONS))
        self.assertEqual([c.no for c in r005.SHADOW_ENTRY_CONDITIONS], [2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15])

    def test_numbers_follow_evaluation_order(self):
        # [2026-11-08] AutoBotH plots conditions bottom-up in evaluation order - the numbers must count up with it
        self.assertEqual([c.no for c in r005.BUY_CONDITIONS], list(range(1, len(r005.BUY_CONDITIONS) + 1)))
        for c in r005.BUY_CONDITIONS:
            self.assertTrue(c.func_name.startswith(f"_{c.no:03d}_"), c.func_name)
        basic = [c.no for c in r005.BASIC_BUY_CONDITIONS]
        self.assertEqual(basic, sorted(basic))
        self.assertEqual(r005.BASIC_BUY_CONDITIONS[-3:], r005.BUY_CONDITIONS[-3:])  # same pre-order safety tail


class TestV2Shadow(unittest.TestCase):
    def test_runner_never_raises_and_records(self):
        from test_r008_strategy_v2 import _m1_from_closes
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d) / "data"
            prev = data_dir / "20261019"
            prev.mkdir(parents=True)
            n = 120
            c = np.linspace(8000, 10000, n)
            pd.DataFrame({"date": pd.bdate_range("2026-04-01", periods=n).strftime("%Y%m%d").astype(int),
                          "open": c, "high": c * 1.02, "low": c * 0.98, "close": c, "volume": 1e6, "amount": c * 1e6}
                         ).to_csv(prev / "000001_TEST_daily.csv", index=False)
            logs = []
            cfg = V2Config(daily_min_avg_amount_krw=1e9, enable_breakout_retest=False, stop_min_pct=0.3, pb_max_depth_atr=5.0,
                           market_breadth_min_pct=0)
            r = V2ShadowRunner(cfg, data_dir, Path(d) / "out", "20261020", logs.append)
            up = list(np.linspace(10000, 10600, 60).round(-1))
            dip = list(np.linspace(10580, 10340, 12).round(-1))
            m1 = _m1_from_closes(up + dip + [10420] + [10450] * 10, [1000] * 72 + [4000] + [1000] * 10)
            for i in range(5, len(m1) + 1):
                r.on_symbol("000001", "TEST", m1.index[i - 1].to_pydatetime(), m1.iloc[:i], float(m1["close"].iloc[i - 1]))
            r.on_symbol("000001", "TEST", datetime(2026, 10, 20, 11), "not a frame", 100.0)  # garbage in -> logged, no raise
            events = [json.loads(x)["event"] for x in (Path(d) / "out" / "v2_shadow_20261020.jsonl").read_text().splitlines()]
            self.assertIn("DAILY", events)
            self.assertIn("ENTRY", events)
            self.assertTrue(any("error" in m for m in logs))


if __name__ == "__main__":
    unittest.main()
