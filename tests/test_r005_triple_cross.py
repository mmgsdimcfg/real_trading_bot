# -*- coding: utf-8 -*-
"""Triple golden-cross entry (2026-10-30, 326030 SK바이오팜 case).

r002 find_1min_ma_golden_cross / HYBRID_1MIN_ALLOW_CONTINUATION_ENTRY, r005 _008 triple mode (1-min MA5>MA20 AND
60-tick AND 120-tick), soft context-gate bypass (_context_gate / _017), mode precedence, r003 g011 supervision."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r002_strategy_core_shared as r002  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
from r013_tick_trigger import TickCross, TickTriggerResult  # noqa: E402

# 326030 SK바이오팜 2026-10-30 KIS 1-min bars 09:01..09:41 (label = bar end), MA_5/BB_MIDDLE with prior-day warmup.
_OPEN = [78000, 77400, 76700, 76500, 76800, 77000, 77000, 76700, 77000, 76700, 76300, 76300, 76300, 76100, 76400,
         76200, 76300, 76300, 76700, 77000, 76600, 77100, 77300, 77500, 77300, 77500, 77600, 77600, 77700, 78100,
         77800, 78300, 78900, 78800, 78500, 78300, 78500, 78400, 78400, 78400, 77900]
_CLOSE = [77300, 76800, 76500, 76800, 77300, 77000, 76800, 76700, 76600, 76400, 76200, 76300, 76100, 76200, 76200,
          76400, 76400, 76600, 76700, 76800, 77100, 77300, 77600, 77300, 77700, 77600, 77500, 77600, 78100, 77800,
          78100, 78900, 78900, 78500, 78200, 78500, 78300, 78400, 78400, 77900, 77700]
_MA5 = [77680, 77480, 77240, 77040, 76940, 76880, 76880, 76920, 76880, 76700, 76540, 76440, 76320, 76240, 76200,
        76240, 76260, 76360, 76460, 76580, 76720, 76900, 77100, 77220, 77400, 77500, 77540, 77540, 77700, 77720,
        77820, 78100, 78360, 78440, 78520, 78600, 78480, 78380, 78360, 78300, 78140]
_BB = [77790, 77750, 77685, 77655, 77630, 77595, 77545, 77490, 77425, 77350, 77265, 77185, 77095, 77005, 76915,
       76835, 76765, 76710, 76655, 76605, 76595, 76620, 76675, 76700, 76720, 76750, 76785, 76830, 76905, 76975,
       77070, 77200, 77340, 77455, 77555, 77660, 77755, 77845, 77930, 77985, 78015]


def _sk_frame(last_label: str) -> pd.DataFrame:
    idx = pd.date_range("2026-10-30 09:01", periods=len(_CLOSE), freq="1min")
    df = pd.DataFrame({"open": _OPEN, "close": _CLOSE, "MA_5": _MA5, "BB_MIDDLE": _BB}, index=idx, dtype=float)
    return df.loc[:last_label]


def _tick(now: datetime, ok60=True, ok120=True, age_s=30.0) -> TickTriggerResult:
    crosses = {}
    for n, ok in ((60, ok60), (120, ok120)):
        crosses[n] = TickCross(ok, f"TICK{n}_MA5_MA20_GOLDEN_CROSS" if ok else f"TICK{n}_MA5_NOT_ABOVE_MA20", n,
                               cross_time=now - timedelta(seconds=age_s) if ok else None)
    first = next((n for n in (60, 120) if crosses[n].ok), None)
    return TickTriggerResult(first is not None, f"TICK{first}_MA5_MA20" if first else "",
                             "|".join(c.reason for c in crosses.values()),
                             crosses[first].cross_time if first else None, crosses)


class TestFind1minMaCross(unittest.TestCase):
    def test_sk_cross_at_0921(self):
        out = r002.find_1min_ma_golden_cross(_sk_frame("2026-10-30 09:22"), 5, 20)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["cross_bar_time"], pd.Timestamp("2026-10-30 09:21"))
        self.assertEqual(out["bars_since_cross"], 1)

    def test_not_above(self):
        out = r002.find_1min_ma_golden_cross(_sk_frame("2026-10-30 09:15"), 5, 20)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "1MIN_MA5_MA20_NOT_ABOVE")

    def test_above_since_frame_start_has_no_cross(self):
        frame = _sk_frame("2026-10-30 09:36").loc["2026-10-30 09:25":]
        out = r002.find_1min_ma_golden_cross(frame, 5, 20)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "1MIN_MA5_MA20_NO_CROSS_IN_FRAME")

    def test_rolling_fallback_without_columns(self):
        closes = [100.0] * 20 + [99.0, 98.0, 103.0, 105.0]
        frame = pd.DataFrame({"close": closes}, index=pd.date_range("2026-10-30 09:00", periods=len(closes), freq="1min"))
        out = r002.find_1min_ma_golden_cross(frame, 5, 20)
        self.assertTrue(out["ok"], out)
        # 09:22 MA5 == MA20 (100.0) is not "above", so the cross bar is the last one
        self.assertEqual((out["cross_bar_time"], out["bars_since_cross"]), (frame.index[-1], 0))


class TestContinuationFlag(unittest.TestCase):
    def _frame(self):
        # close above BB_MIDDLE for longer than the lookback -> no cross found; only continuation could pass
        n = 30
        idx = pd.date_range("2026-10-30 09:00", periods=n, freq="1min")
        return pd.DataFrame({"open": [100.0] * n, "close": [100.3] * n, "BB_MIDDLE": [100.0] * n}, index=idx)

    def test_continuation_blocked_by_default(self):
        self.assertFalse(r002.HYBRID_1MIN_ALLOW_CONTINUATION_ENTRY)
        ok, reason = r002.check_buy_condition_1min_hybrid_trigger(self._frame(), context_uptrend_continuation=True)
        self.assertEqual((ok, reason), (False, "1MIN_NO_BB_MID_GOLDEN_CROSS"))

    def test_continuation_allowed_when_flag_on(self):
        with mock.patch.object(r002, "HYBRID_1MIN_ALLOW_CONTINUATION_ENTRY", True):
            ok, reason = r002.check_buy_condition_1min_hybrid_trigger(self._frame(), context_uptrend_continuation=True)
        self.assertEqual((ok, reason), (True, "1MIN_UPTREND_CONTINUATION_3MIN_CTX"))

    def test_external_bars_since_cross_widens_gap(self):
        frame = self._frame()
        frame.iloc[-1, frame.columns.get_loc("close")] = 100.9  # gap 0.9% > BB_GAP_MAX(0.8) at 0 bars
        frame.iloc[-1, frame.columns.get_loc("open")] = 100.6
        ok0, r0 = r002.check_buy_condition_1min_hybrid_trigger(frame, tick_cross_label="X")
        ok1, r1 = r002.check_buy_condition_1min_hybrid_trigger(frame, tick_cross_label="X", external_bars_since_cross=1)
        self.assertFalse(ok0, r0)
        self.assertTrue(ok1, r1)


class TestR005Triple(unittest.TestCase):
    def _ctx(self, frame_1min, now, price, tick_fn, logs, records):
        services = SimpleNamespace(
            get_frame_1min=lambda code, dt, nxt: frame_1min,
            format_reject_detail=lambda reason, *a, **k: reason,
            log=logs.append, record_gate_decision=records.append, tick_trigger=tick_fn,
        )
        frame3 = pd.DataFrame({"close": [1.0, 1.0], "BB_MIDDLE": [float("nan")] * 2},
                              index=pd.date_range("2026-10-30 09:15", periods=2, freq="3min"))
        state = SimpleNamespace(buy_trigger_age_state={}, buy_confirm_state={"326030": {"count": 1}})
        ctx = SimpleNamespace(
            code="326030", norm_code="326030", symbol_label="326030_SK바이오팜", price=price, current_dt=now,
            nxt_tradeable=True, services=services, state=state, buy_frame=frame3, prev_bar=frame3.iloc[-2],
            cross_info={}, bar_time=pd.Timestamp("2026-10-30 09:18"), trigger_flag="", steps_text="",
            trigger_reason="", trigger_info=None, triple_cross=False, bypassed_gates=[],
        )
        ctx.log = logs.append
        return ctx

    def setUp(self):
        for p in (mock.patch.object(r005, "ENTRY_1MIN_FRAME_REQUIRE_LATEST", False),
                  mock.patch.object(r005, "_steps_text", lambda ctx: ""),
                  mock.patch.object(r005, "ENTRY_TRIGGER_MODE", "triple")):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r005._TRIGGER_AGE_SEEN.clear()

    def _run(self, last_label, now, price, tick):
        logs, recs = [], []
        tick_fn = None if tick is None else (lambda code, dt: tick)
        ctx = self._ctx(_sk_frame(last_label), now, price, tick_fn, logs, recs)
        return r005._008_hybrid_1min_trigger(ctx), ctx, logs, recs

    def test_0922_all_three_crosses_pass(self):
        now = datetime(2026, 10, 30, 9, 22, 13)
        ok, ctx, logs, recs = self._run("2026-10-30 09:22", now, 77300.0, _tick(now))
        self.assertTrue(ok, logs)
        self.assertTrue(ctx.triple_cross)
        self.assertEqual(ctx.trigger_reason, "TRIPLE_CROSS_1MIN_MA5_MA20+TICK60+TICK120_GOLDEN_CROSS")
        self.assertEqual(ctx.trigger_info["path"], "triple")
        self.assertEqual(ctx.trigger_info["bars_since_cross"], 1)
        self.assertEqual([r["result"] for r in recs], ["TRIPLE_CROSS_PASS"])

    def test_0936_no_fresh_1min_cross_rejected(self):
        now = datetime(2026, 10, 30, 9, 36, 39)
        ok, ctx, logs, _ = self._run("2026-10-30 09:36", now, 78300.0, _tick(now))
        self.assertFalse(ok)
        self.assertFalse(ctx.triple_cross)
        self.assertIn("TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_CROSS_EXPIRED_", logs[-1])
        self.assertNotIn("326030", ctx.state.buy_confirm_state)  # reject clears the 2-poll confirm state

    def test_tick_feed_missing_rejects(self):
        now = datetime(2026, 10, 30, 9, 22, 13)
        ok, _, logs, _ = self._run("2026-10-30 09:22", now, 77300.0, None)
        self.assertFalse(ok)
        self.assertIn("TICK_FEED_UNAVAILABLE", logs[-1])
        ok, _, logs, _ = self._run("2026-10-30 09:22", now, 77300.0,
                                   TickTriggerResult(False, reason="TICK_ABSTAIN_NO_FILE"))
        self.assertFalse(ok)
        self.assertIn("TICK_ABSTAIN_NO_FILE", logs[-1])

    def test_one_tick_size_missing_rejects(self):
        now = datetime(2026, 10, 30, 9, 22, 13)
        for tick, missing in ((_tick(now, ok120=False), "TICK120_MA5_NOT_ABOVE_MA20"),
                              (_tick(now, ok60=False), "TICK60_MA5_NOT_ABOVE_MA20")):
            ok, _, logs, _ = self._run("2026-10-30 09:22", now, 77300.0, tick)
            self.assertFalse(ok)
            self.assertIn(missing, logs[-1])

    def test_ticks_without_1min_cross_rejects(self):
        now = datetime(2026, 10, 30, 9, 18, 30)
        ok, _, logs, _ = self._run("2026-10-30 09:18", now, 76700.0, _tick(now))
        self.assertFalse(ok)
        self.assertIn("1MIN_MA5_MA20_NOT_ABOVE", logs[-1])

    def test_live_drop_rejects(self):
        now = datetime(2026, 10, 30, 9, 22, 13)
        ok, _, logs, _ = self._run("2026-10-30 09:22", now, 76900.0, _tick(now))  # -0.52% vs 77,300
        self.assertFalse(ok)
        self.assertIn("TRIPLE_CROSS_1MIN_LIVE_DROP_", logs[-1])

    def test_tick_service_exception_rejects(self):
        def boom(code, dt):
            raise RuntimeError("feed down")
        logs, recs = [], []
        now = datetime(2026, 10, 30, 9, 22, 13)
        ctx = self._ctx(_sk_frame("2026-10-30 09:22"), now, 77300.0, boom, logs, recs)
        self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertIn("TICK_FEED_UNAVAILABLE", logs[-1])


class TestSoftGateBypass(unittest.TestCase):
    def _ctx(self, triple):
        logs = []
        ctx = SimpleNamespace(triple_cross=triple, bypassed_gates=[], eval_ctx=None, trigger_reason="TRIPLE_X",
                              current_dt=datetime(2026, 10, 30, 9, 22), config=None, buy_reason="")
        return ctx, logs

    def _gate(self, name):
        return SimpleNamespace(eval_fn=lambda e: (False, e, f"{name.upper()}_FAIL"))

    def test_bypass_list_only(self):
        with mock.patch.dict(r005._GATES_BY_NAME, {"stochastic_buy_signal": self._gate("stoch"),
                                                   "min_liquidity_safety": self._gate("liq")}), \
                mock.patch.object(r005, "_reject_hybrid") as rej:
            ctx, _ = self._ctx(True)
            self.assertTrue(r005._context_gate(ctx, "stochastic_buy_signal"))
            self.assertEqual(ctx.bypassed_gates, ["stochastic_buy_signal"])
            self.assertFalse(r005._context_gate(ctx, "min_liquidity_safety"))
            rej.assert_called_once()
            ctx2, _ = self._ctx(False)
            self.assertFalse(r005._context_gate(ctx2, "stochastic_buy_signal"))

    def test_score_bypass_and_reason(self):
        with mock.patch.object(r005, "evaluate_context_score", lambda *a: (False, "HYBRID_3MIN_CTX_SCORE_5_LT_10")), \
                mock.patch.object(r005, "_reject_hybrid") as rej:
            ctx, _ = self._ctx(True)
            ctx.bypassed_gates = ["stochastic_buy_signal"]
            self.assertTrue(r005._017_context_score_threshold(ctx))
            self.assertEqual(ctx.buy_reason,
                             "TRIPLE_X+HYBRID_3MIN_CTX_SCORE_5_LT_10+BYPASS(stochastic_buy_signal,score)")
            with mock.patch.object(r005, "TRIPLE_CROSS_BYPASS_SCORE", False):
                ctx, _ = self._ctx(True)
                self.assertFalse(r005._017_context_score_threshold(ctx))
            ctx, _ = self._ctx(False)
            self.assertFalse(r005._017_context_score_threshold(ctx))
            self.assertEqual(rej.call_count, 2)

    def test_triple_overrides_basic(self):
        with mock.patch.object(r005, "TRADING_STRATEGY_MODE", "basic"):
            with mock.patch.object(r005, "ENTRY_TRIGGER_MODE", "triple"):
                self.assertIs(r005.active_buy_conditions(), r005.BUY_CONDITIONS)
            with mock.patch.object(r005, "ENTRY_TRIGGER_MODE", "legacy"):
                self.assertIs(r005.active_buy_conditions(), r005.BASIC_BUY_CONDITIONS)


class TestTickCollectorSupervisor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import r003_trade_live_execute as r003
        cls.r003 = r003

    def setUp(self):
        self.logs = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        r003 = self.r003
        r003._TICK_COLLECTOR_STATE.update({"proc": None, "last_start": None, "last_warn": None, "reused_pid": None,
                                           "log_fh": None, "warned_codes": None})
        for p in (mock.patch.object(r003, "log", self.logs.append),
                  mock.patch.object(r003, "TICK_COLLECTOR_AUTOSTART", True),
                  mock.patch.object(r003, "ENTRY_TRIGGER_MODE", "triple"),
                  mock.patch.object(r003, "RUNTIME_LOG_DIR", Path(self.tmp.name) / "log"),
                  mock.patch.object(r003, "RUNTIME_DATA_DIR", Path(self.tmp.name) / "data")):
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(lambda: r003._TICK_COLLECTOR_STATE["log_fh"] and r003._TICK_COLLECTOR_STATE["log_fh"].close())

    def test_start_once_then_rate_limited_and_warn(self):
        r003 = self.r003
        proc = mock.Mock(pid=4242)
        proc.poll.return_value = 1  # died right away
        proc.returncode = 1
        now = datetime(2026, 10, 30, 9, 10, 0)
        with mock.patch.object(r003, "_find_running_tick_collector", return_value=None), \
                mock.patch.object(r003.subprocess, "Popen", return_value=proc) as popen:
            r003._supervise_tick_collector(now, ["326030", "3160"])
            args = popen.call_args[0][0]
            self.assertIn("--codes", args)
            self.assertIn("003160", args)
            r003._supervise_tick_collector(now + timedelta(seconds=20), ["326030"])  # dead, but within 60 s
            self.assertEqual(popen.call_count, 1)
            r003._supervise_tick_collector(now + timedelta(seconds=61), ["326030"])
            self.assertEqual(popen.call_count, 2)
        self.assertTrue(any("[TICK FEED   ] WARN no tick files" in m for m in self.logs), self.logs)

    def test_reuse_running_collector_and_after_until(self):
        r003 = self.r003
        with mock.patch.object(r003, "_find_running_tick_collector", return_value=999), \
                mock.patch.object(r003.subprocess, "Popen") as popen:
            r003._supervise_tick_collector(datetime(2026, 10, 30, 8, 0), ["326030"])
            r003._supervise_tick_collector(datetime(2026, 10, 30, 15, 40), ["326030"])
            popen.assert_not_called()
        self.assertEqual(sum("reusing" in m for m in self.logs), 1)

    def test_reused_collector_missing_codes_and_cap_warn(self):
        r003 = self.r003
        with mock.patch.object(r003, "_find_running_tick_collector", return_value=999), \
                mock.patch.object(r003, "_tick_collector_codes_of", return_value={"326030"}):
            r003._supervise_tick_collector(datetime(2026, 10, 30, 8, 0), ["326030", "003160"])
        self.assertTrue(any("does not subscribe 1 watched codes ['003160']" in m for m in self.logs), self.logs)
        r003._TICK_COLLECTOR_STATE.update({"reused_pid": None, "last_start": None})
        proc = mock.Mock(pid=1)
        proc.poll.return_value = None
        codes = [f"{i:06d}" for i in range(45)]
        with mock.patch.object(r003, "_find_running_tick_collector", return_value=None), \
                mock.patch.object(r003.subprocess, "Popen", return_value=proc) as popen:
            r003._supervise_tick_collector(datetime(2026, 10, 30, 8, 0), codes)
        argv = popen.call_args[0][0]
        self.assertEqual(len(argv[argv.index("--codes") + 1:argv.index("--until")]), 40)
        self.assertTrue(any("45 watched codes > 40" in m for m in self.logs), self.logs)

    def test_per_code_missing_file_warn_once(self):
        r003 = self.r003
        tick_dir = Path(self.tmp.name) / "data" / "ticks" / "20261030"
        tick_dir.mkdir(parents=True)
        (tick_dir / "326030.csv").write_text("x\n")
        r003._TICK_COLLECTOR_STATE["proc"] = mock.Mock(**{"poll.return_value": None})
        now = datetime(2026, 10, 30, 9, 10)  # must match tick_dir's date (was "today" -> failed on other days)
        os_utime = __import__("os").utime
        os_utime(tick_dir / "326030.csv", (now.timestamp(), now.timestamp()))
        r003._supervise_tick_collector(now, ["326030", "003160"])
        r003._supervise_tick_collector(now + timedelta(seconds=30), ["326030", "003160"])
        self.assertEqual(sum("no tick file yet for 1 codes ['003160']" in m for m in self.logs), 1, self.logs)


if __name__ == "__main__":
    unittest.main()
