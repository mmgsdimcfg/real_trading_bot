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


def _ctx(realized=0.0, traded=(), exposure=(), bid=10000, ask=10010):
    logs = []
    api = SimpleNamespace(live_state={"realized_pnl_krw": realized}, has_buy_exposure=lambda c: c in exposure)
    services = SimpleNamespace(get_order_spec=lambda now, nxt: {"exchange": "KRX"},
                               fetch_bid_ask=lambda code, mkt: (bid, ask))
    return SimpleNamespace(api=api, services=services, log=logs.append, symbol_label="X", norm_code="000001",
                           current_dt=datetime(2026, 10, 23, 10), nxt_tradeable=False,
                           state=SimpleNamespace(traded_today=set(traded))), logs


class TestPortfolioGates(unittest.TestCase):
    def test_default_off_pass_everything(self):
        ctx, _ = _ctx(realized=-1e9, traded={"1", "2", "3", "4", "5", "6", "7"}, exposure={"1", "2", "3", "4", "5", "6", "7"},
                      bid=10000, ask=11000)
        self.assertTrue(r005._025_daily_loss_limit(ctx))
        self.assertTrue(r005._026_max_open_positions(ctx))
        self.assertTrue(r005._027_entry_spread(ctx))

    def test_daily_loss_limit(self):
        with mock.patch.object(r005, "ENABLE_DAILY_LOSS_LIMIT_GATE", True), mock.patch.object(r005, "DAILY_LOSS_LIMIT_KRW", 100_000):
            self.assertTrue(r005._025_daily_loss_limit(_ctx(realized=-99_999)[0]))
            ctx, logs = _ctx(realized=-100_000)
            self.assertFalse(r005._025_daily_loss_limit(ctx))
            self.assertIn("DAILY_LOSS_LIMIT", logs[0])

    def test_max_open_positions_counts_only_bot_exposure(self):
        with mock.patch.object(r005, "ENABLE_MAX_OPEN_POSITIONS_GATE", True), mock.patch.object(r005, "MAX_OPEN_POSITIONS", 2):
            # 3 codes traded today, but only 1 still exposed -> allowed
            self.assertTrue(r005._026_max_open_positions(_ctx(traded={"a", "b", "c"}, exposure={"a"})[0]))
            self.assertFalse(r005._026_max_open_positions(_ctx(traded={"a", "b"}, exposure={"a", "b"})[0]))
            # the candidate itself is not counted
            self.assertTrue(r005._026_max_open_positions(_ctx(traded={"a", "000001"}, exposure={"a", "000001"})[0]))

    def test_spread_gate(self):
        with mock.patch.object(r005, "ENABLE_SPREAD_GATE", True), mock.patch.object(r005, "MAX_ENTRY_SPREAD_TICKS", 2):
            self.assertTrue(r005._027_entry_spread(_ctx(bid=10000, ask=10020)[0]))
            self.assertFalse(r005._027_entry_spread(_ctx(bid=10000, ask=10030)[0]))
            self.assertFalse(r005._027_entry_spread(_ctx(bid=None, ask=10030)[0]))

    def test_registration_order(self):
        names = [c.func_name for c in r005.BUY_CONDITIONS]
        self.assertLess(names.index("_007_stoploss_circuit_breaker"), names.index("_025_daily_loss_limit"))
        self.assertLess(names.index("_026_max_open_positions"), names.index("_008_hybrid_1min_trigger"))
        self.assertLess(names.index("_027_entry_spread"), names.index("_021_consecutive_poll_confirm"))
        self.assertEqual(len({c.no for c in r005.BUY_CONDITIONS}), len(r005.BUY_CONDITIONS))
        self.assertTrue(all(2 <= c.no <= 18 for c in r005.SHADOW_ENTRY_CONDITIONS))


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
