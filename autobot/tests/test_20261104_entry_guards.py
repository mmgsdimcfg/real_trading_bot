"""[2026-11-04] Entry guards from the log-pattern review: chase-from-cross guard (r002/r005/g003), tick density floor (r013)
and the pre-open TICK_ABSTAIN_NO_FILE log suppression (r005)."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import r001_define_config as r001  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
import r013_tick_trigger as r013  # noqa: E402
from r009_tick_bars import TickBar  # noqa: E402
from r013_tick_trigger import TickCross, TickTriggerResult  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_r005_triple_cross import _sk_frame  # noqa: E402


def _tick(now, cross_close=float("nan"), age_s=30.0):
    c = TickCross(True, "TICK60_MA5_MA20_GOLDEN_CROSS", 60, cross_time=now - timedelta(seconds=age_s),
                  cross_close=cross_close)
    c2 = TickCross(False, "TICK120_MA5_NOT_ABOVE_MA20", 120)
    return TickTriggerResult(True, "TICK60_MA5_MA20", "x", c.cross_time, {60: c, 120: c2})


class TestChaseFromCross(unittest.TestCase):
    def test_check_chase_from_cross(self):
        self.assertEqual(r002.check_chase_from_cross(100.5, 100.0, 0.5), (True, ""))
        ok, reason = r002.check_chase_from_cross(101.0, 100.0, 0.5)
        self.assertFalse(ok)
        self.assertEqual(reason, "CHASE_FROM_CROSS_1.00%_GT_0.50%")
        # disabled / no reference -> pass
        self.assertTrue(r002.check_chase_from_cross(101.0, 100.0, 0.0)[0])
        self.assertTrue(r002.check_chase_from_cross(101.0, float("nan"), 0.5)[0])
        self.assertTrue(r002.check_chase_from_cross(None, 100.0, 0.5)[0])

    def test_evaluate_triple_trigger_picks_freshest_cross_close(self):
        now = datetime(2026, 11, 4, 10, 0, 0)
        crosses = {
            60: TickCross(True, "ok", 60, cross_time=now - timedelta(seconds=20), cross_close=101.0),
            120: TickCross(True, "ok", 120, cross_time=now - timedelta(seconds=90), cross_close=99.0),
        }
        ma = {"ok": True, "bars_since_cross": 0, "cross_bar_time": None}
        out = r002.evaluate_triple_trigger(ma, 10.0, crosses, (60, 120), "above", "any", 240.0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["cross_ref_price"], 101.0)
        # a cross without close (old TickCross) is ignored
        crosses[60] = TickCross(True, "ok", 60, cross_time=now)
        out = r002.evaluate_triple_trigger(ma, 10.0, crosses, (60, 120), "above", "any", 240.0)
        self.assertEqual(out["cross_ref_price"], 99.0)


class TestDensity(unittest.TestCase):
    def _bars(self, start, count, step_s, n_ticks=60):
        out = []
        for i in range(count):
            ts = start + timedelta(seconds=step_s * (i + 1))
            out.append(TickBar("159010", ts - timedelta(seconds=step_s), ts, 1, 1, 1, 1, n_ticks=n_ticks, closed_at=ts))
        return out

    def test_prints_per_minute_and_floor(self):
        open_ = datetime(2026, 11, 4, 9, 0, 0)
        # 60-tick bar every 20 s -> 180 prints/min
        fast = self._bars(open_, 90, 20)
        now = open_ + timedelta(minutes=30)
        self.assertAlmostEqual(r013.prints_per_minute(fast, now, 30), 180.0, places=6)
        self.assertIsNone(r013.density_abstain({60: SimpleNamespace(bars=fast)}, now, 15.0, 30))
        # 60-tick bar every 5 min -> 12 prints/min < 15 -> abstain
        slow = self._bars(open_, 12, 300)
        now = open_ + timedelta(minutes=60)
        self.assertAlmostEqual(r013.prints_per_minute(slow, now, 30), 12.0, places=6)
        res = r013.density_abstain({60: SimpleNamespace(bars=slow), 120: SimpleNamespace(bars=[])}, now, 15.0, 30)
        self.assertIsNotNone(res)
        self.assertFalse(res.ok)
        self.assertTrue(res.reason.startswith("TICK_LOW_DENSITY_12/min_LT_15"))
        self.assertAlmostEqual(res.prints_per_min, 12.0, places=6)

    def test_exact_density_is_stable_across_bar_boundaries(self):
        # [Codex 2차 #2] 정확히 15체결/분(60틱봉 4분마다)이면 봉 완성 시점과 무관하게 항상 15
        open_ = datetime(2026, 11, 4, 9, 0, 0)
        bars = self._bars(open_, 30, 240)  # 2h of bars
        for minute in (30, 31, 32, 33, 45, 60, 61):
            d = r013.prints_per_minute(bars, open_ + timedelta(minutes=minute), 30)
            self.assertAlmostEqual(d, 15.0, places=6, msg=minute)

    def test_not_applied_right_after_open_or_when_disabled(self):
        open_ = datetime(2026, 11, 4, 9, 0, 0)
        slow = self._bars(open_, 1, 120)
        self.assertTrue(r013.prints_per_minute(slow, open_ + timedelta(minutes=3), 30) != r013.prints_per_minute(slow, open_ + timedelta(minutes=3), 30))  # nan
        self.assertIsNone(r013.density_abstain({60: SimpleNamespace(bars=slow)}, open_ + timedelta(minutes=3), 15.0, 30))
        self.assertIsNone(r013.density_abstain({60: SimpleNamespace(bars=slow)}, open_ + timedelta(minutes=60), 0.0, 30))
        # prior-day bars never count
        prior = self._bars(open_ - timedelta(days=1), 50, 20)
        self.assertTrue(r013.prints_per_minute(prior, open_ + timedelta(minutes=10), 30) != r013.prints_per_minute(prior, open_ + timedelta(minutes=10), 30))

    def test_feeds_accept_density_args(self):
        live = r013.TickTriggerFeed(Path("/nonexistent"), (60, 120), 5, 20, 240.0, 120.0, min_prints_per_min=15.0, density_window_min=30)
        self.assertEqual(live.min_prints_per_min, 15.0)
        rep = r013.TickReplayFeed(Path("/nonexistent"), (60, 120), 5, 20, 240.0, 120.0, min_prints_per_min=15.0)
        self.assertEqual(rep.density_window_min, 30.0)
        self.assertGreater(r001.TICK_MIN_PRINTS_PER_MIN, 0)


class TestR005TripleGuards(unittest.TestCase):
    def _ctx(self, frame_1min, now, price, tick_fn, logs):
        services = SimpleNamespace(
            get_frame_1min=lambda code, dt, nxt: frame_1min,
            format_reject_detail=lambda reason, *a, **k: reason,
            log=logs.append, record_gate_decision=lambda r: None, tick_trigger=tick_fn,
        )
        frame3 = pd.DataFrame({"close": [1.0, 1.0], "BB_MIDDLE": [float("nan")] * 2},
                              index=pd.date_range("2026-10-30 09:15", periods=2, freq="3min"))
        state = SimpleNamespace(buy_trigger_age_state={}, buy_confirm_state={"326030": {"count": 1}}, tick_nofile_logged=set())
        ctx = SimpleNamespace(
            code="326030", norm_code="326030", symbol_label="326030_SK바이오팜", price=price, current_dt=now,
            nxt_tradeable=True, services=services, state=state, buy_frame=frame3, prev_bar=frame3.iloc[-2],
            cross_info={}, bar_time=pd.Timestamp("2026-10-30 09:18"), trigger_flag="", steps_text="",
            trigger_reason="", trigger_info=None, triple_cross=False,
        )
        ctx.log = logs.append
        return ctx

    def setUp(self):
        for p in (mock.patch.object(r005, "ENTRY_1MIN_FRAME_REQUIRE_LATEST", False),
                  mock.patch.object(r005, "_steps_text", lambda ctx, **kw: ""),
                  mock.patch.object(r005, "ENTRY_TRIGGER_MODE", "triple"),
                  mock.patch.object(r005, "TRIPLE_1MIN_CONDITION", "above"),
                  mock.patch.object(r005, "TRIPLE_TICK_CONDITION", "any"),
                  mock.patch.object(r005, "ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD", True),
                  mock.patch.object(r005, "TRIPLE_CHASE_FROM_CROSS_MAX_PCT", 0.5)):
            p.start()
        self.addCleanup(mock.patch.stopall)
        r005._TRIGGER_AGE_SEEN.clear()

    def test_chase_from_cross_rejects_and_passes(self):
        now = datetime(2026, 10, 30, 9, 22, 13)
        frame = _sk_frame("2026-10-30 09:22")
        logs = []
        # 77300 vs cross close 76500 -> +1.05% > 0.5% -> reject
        ctx = self._ctx(frame, now, 77300.0, lambda c, d: _tick(now, cross_close=76500.0), logs)
        self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertTrue(any("TRIPLE_CROSS_CHASE_FROM_CROSS_1.05%_GT_0.50%" in m for m in logs), logs)
        # within 0.5% -> pass
        logs.clear()
        ctx = self._ctx(frame, now, 77300.0, lambda c, d: _tick(now, cross_close=77100.0), logs)
        self.assertTrue(r005._008_hybrid_1min_trigger(ctx), logs)
        # guard disabled -> pass even when chasing
        logs.clear()
        with mock.patch.object(r005, "ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD", False):
            ctx = self._ctx(frame, now, 77300.0, lambda c, d: _tick(now, cross_close=76500.0), logs)
            self.assertTrue(r005._008_hybrid_1min_trigger(ctx), logs)

    def test_pre_open_no_file_logged_once_and_silent(self):
        frame = _sk_frame("2026-10-30 09:22")
        nofile = TickTriggerResult(False, reason="TICK_ABSTAIN_NO_FILE")
        logs = []
        pre = datetime(2026, 10, 30, 8, 30, 0)
        ctx = self._ctx(frame, pre, 77300.0, lambda c, d: nofile, logs)
        for _ in range(3):
            self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertEqual(sum("no tick file before" in m for m in logs), 1)
        self.assertFalse(any("[REJECT" in m for m in logs), logs)
        self.assertNotIn("326030", ctx.state.buy_confirm_state)
        # after the open the usual REJECT line is logged
        logs.clear()
        ctx = self._ctx(frame, datetime(2026, 10, 30, 9, 22, 13), 77300.0, lambda c, d: nofile, logs)
        self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertTrue(any("TICK_ABSTAIN_NO_FILE" in m for m in logs), logs)


class TestG003Overrides(unittest.TestCase):
    def test_g003_exposes_env_overrides(self):
        try:
            import g003_trade_simulate_by_date as g003
        except BaseException as exc:
            self.skipTest(f"g003 import failed: {exc}")
        for nm in ("ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD", "TRIPLE_CHASE_FROM_CROSS_MAX_PCT", "TICK_MIN_PRINTS_PER_MIN"):
            self.assertTrue(hasattr(g003, nm), nm)


if __name__ == "__main__":
    unittest.main()
