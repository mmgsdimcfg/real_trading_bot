# -*- coding: utf-8 -*-
"""Tick-chart trigger (2026-10-29, 003160 디아이 10:42 case).

r013 tick_ma_golden_cross / TickTriggerFeed (incremental g011 file reader) + r002 tick_cross_label path +
r005 _008 OR wiring (tick cross replaces a missing/expired 1-min cross only)."""
from __future__ import annotations

import csv
import os
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
from r009_tick_bars import TICK_FILE_COLUMNS, TickBar  # noqa: E402
from r013_tick_trigger import TickTriggerFeed, TickTriggerResult, tick_ma_golden_cross  # noqa: E402

T0 = datetime(2026, 10, 29, 10, 30, 0)


def _bars(closes, step_s=10, invalid_at=()):
    out = []
    for i, c in enumerate(closes):
        ts = T0 + timedelta(seconds=step_s * i)
        out.append(TickBar("003160", ts, ts, c, c, c, c, n_ticks=60, closed_at=ts, invalid=i in invalid_at))
    return out


# 22 falling bars then 3 rising: MA5 crosses above MA20 on one of the last bars
FALL_RISE = [42000 - 10 * i for i in range(22)] + [41900, 42000, 42100]


class TestTickMaGoldenCross(unittest.TestCase):
    def _cross_index(self, closes):
        s = pd.Series(closes, dtype=float)
        above = s.rolling(5).mean() > s.rolling(20).mean()
        return next(i for i in range(len(s) - 1, 0, -1) if above[i] and not above[i - 1])

    def test_fresh_cross_ok(self):
        bars = _bars(FALL_RISE)
        now = bars[-1].available_at + timedelta(seconds=5)
        c = tick_ma_golden_cross(bars, now, 60, 5, 20, 240)
        self.assertTrue(c.ok, c.reason)
        self.assertEqual(c.reason, "TICK60_MA5_MA20_GOLDEN_CROSS")
        self.assertEqual(c.cross_time, bars[self._cross_index(FALL_RISE)].available_at)
        self.assertGreater(c.ma_fast, c.ma_slow)

    def test_cross_expired(self):
        bars = _bars(FALL_RISE)
        now = bars[self._cross_index(FALL_RISE)].available_at + timedelta(seconds=241)
        c = tick_ma_golden_cross(bars, now, 60, 5, 20, 240)
        self.assertFalse(c.ok)
        self.assertIn("CROSS_EXPIRED", c.reason)

    def test_below_not_ok(self):
        bars = _bars([42000 - 10 * i for i in range(25)])
        c = tick_ma_golden_cross(bars, bars[-1].available_at, 60, 5, 20, 240)
        self.assertEqual(c.reason, "TICK60_MA5_NOT_ABOVE_MA20")

    def test_few_bars_abstain(self):
        bars = _bars(FALL_RISE[:20])
        c = tick_ma_golden_cross(bars, bars[-1].available_at, 60, 5, 20, 240)
        self.assertFalse(c.ok)
        self.assertEqual(c.reason, "TICK60_ABSTAIN_FEW_BARS_20")

    def test_invalid_bar_in_window_abstains(self):
        i = self._cross_index(FALL_RISE)
        bars = _bars(FALL_RISE, invalid_at={i - 3})
        c = tick_ma_golden_cross(bars, bars[-1].available_at, 60, 5, 20, 240)
        self.assertEqual(c.reason, "TICK60_ABSTAIN_INVALID_BAR")
        # an invalid bar older than the MA window of the cross does not matter
        ok = tick_ma_golden_cross(_bars(FALL_RISE, invalid_at={0}), bars[-1].available_at, 60, 5, 20, 240)
        self.assertTrue(ok.ok, ok.reason)

    def test_no_look_ahead(self):
        bars = _bars(FALL_RISE)
        # at the time the last-but-two bar was available the rise had not happened yet
        now = bars[-3].available_at
        c = tick_ma_golden_cross(bars, now, 60, 5, 20, 240)
        self.assertFalse(c.ok)

    def test_yesterday_bars_ignored(self):
        bars = _bars(FALL_RISE)
        for b in bars[:10]:
            b.end_ts = b.end_ts - timedelta(days=1)
        c = tick_ma_golden_cross(bars, bars[-1].available_at, 60, 5, 20, 240)
        self.assertEqual(c.reason, "TICK60_ABSTAIN_FEW_BARS_15")


class TestTickTriggerFeed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "20261029" / "003160.csv"
        self.path.parent.mkdir(parents=True)
        self.k = 0
        self.acml = 0
        self.logs: list[str] = []
        # 3-tick bars, MA2 / MA4 so a handful of prints is enough
        self.feed = TickTriggerFeed(self.dir, (3,), 2, 4, 240, 120, self.logs.append)

    def _rows(self, prices, start):
        rows = []
        for j, p in enumerate(prices):
            self.k += 1
            self.acml += 10
            ts = start + timedelta(seconds=j)
            rows.append({
                "recv_ts": ts.isoformat(timespec="milliseconds"), "code": "003160", "bsop_date": "20261029",
                "trade_time": ts.strftime("%H%M%S"), "price": p, "cntg_vol": 10, "acml_vol": self.acml,
                "ccld_dvsn": 1, "seln_cntg_csnu": 0, "shnu_cntg_csnu": self.k, "askp1": p + 50, "bidp1": p,
                "hour_cls_code": 0,
            })
        return rows

    def _append(self, rows, header=False, partial_tail=""):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TICK_FILE_COLUMNS)
            if header:
                w.writeheader()
            w.writerows(rows)
            f.write(partial_tail)

    def test_no_file(self):
        self.path.unlink(missing_ok=True)
        r = self.feed.evaluate("3160", T0)
        self.assertEqual((r.ok, r.reason), (False, "TICK_ABSTAIN_NO_FILE"))

    def test_incremental_cross_and_partial_line(self):
        falling = [42000 - 10 * i for i in range(15)]  # 5 bars falling
        self._append(self._rows(falling, T0), header=True)
        now = T0 + timedelta(seconds=15)
        r = self.feed.evaluate("003160", now)
        self.assertFalse(r.ok)
        self.assertIn("TICK3_MA2_NOT_ABOVE_MA4", r.reason)
        self.assertEqual(len(self.feed._feeds["003160"].builders[3].bars), 5)

        rows = self._rows([42000, 42050, 42100, 42150, 42200, 42250], now)
        line = ",".join(str(rows[-1].get(c, "")) for c in TICK_FILE_COLUMNS)
        self._append(rows[:-1], partial_tail=line[:20])  # writer flushed mid-row
        r = self.feed.evaluate("003160", now + timedelta(seconds=6))
        bars = self.feed._feeds["003160"].builders[3].bars
        self.assertEqual(len(bars), 6)  # last bar still open: its 3rd print is the partial line
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line[20:] + "\r\n")
        r = self.feed.evaluate("003160", now + timedelta(seconds=7))
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.label, "TICK3_MA2_MA4")
        self.assertEqual(len(bars), 7)
        self.assertTrue(any("[TICK CROSS  ] 003160 | tick3" in m for m in self.logs))
        n_logs = len(self.logs)
        self.feed.evaluate("003160", now + timedelta(seconds=8))
        self.assertEqual(len(self.logs), n_logs)  # same cross logged once

    def test_feed_stale(self):
        self._append(self._rows([42000] * 6, T0), header=True)
        r = self.feed.evaluate("003160", T0 + timedelta(seconds=300))
        self.assertFalse(r.ok)
        self.assertTrue(r.reason.startswith("TICK_ABSTAIN_FEED_STALE_"), r.reason)

    def test_truncated_file_restarts(self):
        self._append(self._rows([42000] * 9, T0), header=True)
        self.feed.evaluate("003160", T0 + timedelta(seconds=9))
        self.path.write_text("")
        self.k = self.acml = 0
        self._append(self._rows([41000] * 3, T0 + timedelta(seconds=20)), header=True)
        self.feed.evaluate("003160", T0 + timedelta(seconds=23))
        bars = self.feed._feeds["003160"].builders[3].bars
        self.assertEqual([b.close for b in bars], [41000])

    def _cross_file(self):
        self._append(self._rows([42000 - 10 * i for i in range(15)], T0), header=True)
        self._append(self._rows([42000, 42050, 42100, 42150, 42200, 42250], T0 + timedelta(seconds=15)))
        now = T0 + timedelta(seconds=22)
        self.assertTrue(self.feed.evaluate("003160", now).ok)
        return now

    def _flat_bytes(self, size):
        """CSV of flat prices padded to exactly `size` bytes (same-size replacement)."""
        self.k = self.acml = 0
        rows = self._rows([42000] * 21, T0)
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=TICK_FILE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        data = buf.getvalue().encode()
        self.assertLessEqual(len(data), size)
        return data + b"\n" * (size - len(data))

    def test_same_size_replacement_new_inode_resets(self):
        now = self._cross_file()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(self._flat_bytes(self.path.stat().st_size))
        os.replace(tmp, self.path)
        r = self.feed.evaluate("003160", now + timedelta(seconds=1))
        self.assertFalse(r.ok, r.reason)
        self.assertIn("NOT_ABOVE", r.reason)

    def test_in_place_rewrite_same_inode_resets(self):
        now = self._cross_file()
        data = self._flat_bytes(self.path.stat().st_size)
        with open(self.path, "r+b") as f:
            f.write(data)
        r = self.feed.evaluate("003160", now + timedelta(seconds=1))
        self.assertFalse(r.ok, r.reason)
        self.assertIn("NOT_ABOVE", r.reason)

    def test_day_rollover_uses_new_file(self):
        now = self._cross_file()
        r = self.feed.evaluate("003160", now + timedelta(days=1))
        self.assertEqual((r.ok, r.reason), (False, "TICK_ABSTAIN_NO_FILE"))

    def test_exception_becomes_not_ok(self):
        with mock.patch.object(self.feed, "_evaluate", side_effect=RuntimeError("boom")):
            r = self.feed.evaluate("003160", T0)
        self.assertEqual((r.ok, r.reason), (False, "TICK_ERROR_RuntimeError"))


def _frame_1min(prev_close=41900.0, close=41950.0, bb_mid=42000.0, open_=41900.0):
    """1-min close still BELOW its BB mid (no 1-min cross)."""
    idx = pd.date_range("2026-10-29 10:40", periods=2, freq="1min")
    return pd.DataFrame(
        {"open": [41950.0, open_], "high": [42000.0, close], "low": [41850.0, open_],
         "close": [prev_close, close], "volume": [1000, 2000], "BB_MIDDLE": [42010.0, bb_mid]},
        index=idx,
    )


class TestR002TickPath(unittest.TestCase):
    def test_tick_label_replaces_missing_1min_cross(self):
        frame = _frame_1min()
        self.assertEqual(r002.check_buy_condition_1min_hybrid_trigger(frame)[1], "1MIN_NO_BB_MID_GOLDEN_CROSS")
        info: dict = {}
        ok, reason = r002.check_buy_condition_1min_hybrid_trigger(frame, info=info, tick_cross_label="TICK60_MA5_MA20")
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "TICK60_MA5_MA20_GOLDEN_CROSS")
        self.assertEqual((info["path"], info["bars_since_cross"]), ("tick", 0))

    def test_tick_path_keeps_candle_guard(self):
        frame = _frame_1min(close=41700.0, open_=41900.0)  # -0.48% confirmed candle
        ok, reason = r002.check_buy_condition_1min_hybrid_trigger(frame, tick_cross_label="TICK60_MA5_MA20")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("1MIN_CANDLE_NOT_BULLISH"), reason)

    def test_live_guard_above_mid_optional(self):
        frame = _frame_1min()
        self.assertEqual(r002.check_1min_trigger_live_guard(frame, 41950.0), (False, "1MIN_LIVE_BELOW_BB_MID"))
        self.assertEqual(r002.check_1min_trigger_live_guard(frame, 41950.0, require_above_mid=False), (True, ""))
        ok, reason = r002.check_1min_trigger_live_guard(frame, 41700.0, require_above_mid=False)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("1MIN_LIVE_DROP_"), reason)


class TestR005TickOr(unittest.TestCase):
    NOW = datetime(2026, 10, 29, 10, 41, 30)

    def _ctx(self, frame_1min, tick_result, logs, records):
        services = SimpleNamespace(
            get_frame_1min=lambda code, dt, nxt: frame_1min,
            format_reject_detail=lambda reason, *a, **k: reason,
            log=logs.append, record_gate_decision=records.append,
            tick_trigger=(lambda code, now: tick_result) if tick_result is not None else None,
        )
        frame3 = pd.DataFrame({"close": [1.0, 1.0], "BB_MIDDLE": [float("nan")] * 2},
                              index=pd.date_range("2026-10-29 10:36", periods=2, freq="3min"))
        state = SimpleNamespace(buy_trigger_age_state={}, buy_confirm_state={})
        ctx = SimpleNamespace(
            code="003160", norm_code="003160", symbol_label="003160_디아이", price=41950.0, current_dt=self.NOW,
            nxt_tradeable=False, services=services, state=state, buy_frame=frame3, prev_bar=frame3.iloc[-2],
            cross_info={}, bar_time=pd.Timestamp("2026-10-29 10:39"), trigger_flag="", steps_text="",
            trigger_reason="", trigger_info=None,
        )
        ctx.log = logs.append
        return ctx

    def setUp(self):
        for p in (mock.patch.object(r005, "ENTRY_1MIN_FRAME_REQUIRE_LATEST", False),
                  mock.patch.object(r005, "_steps_text", lambda ctx: ""),
                  mock.patch.object(r005, "TICK_TRIGGER_MODE", "live"),
                  mock.patch.object(r005, "ENTRY_TRIGGER_MODE", "legacy")):  # 1-min OR tick path
            p.start()
        self.addCleanup(mock.patch.stopall)
        r005._TRIGGER_AGE_SEEN.clear()

    def _tick_ok(self):
        return TickTriggerResult(True, "TICK120_MA5_MA20", "TICK60_X|TICK120_MA5_MA20_GOLDEN_CROSS",
                                 self.NOW - timedelta(seconds=20))

    def test_tick_cross_opens_trigger_when_1min_has_no_cross(self):
        logs, recs = [], []
        ctx = self._ctx(_frame_1min(), self._tick_ok(), logs, recs)
        self.assertTrue(r005._008_hybrid_1min_trigger(ctx), logs)
        self.assertEqual(ctx.trigger_reason, "TICK120_MA5_MA20_GOLDEN_CROSS")
        self.assertEqual(ctx.trigger_info["path"], "tick")
        rec = [r for r in recs if r.get("result") == "TICK_CROSS_PASS"]
        self.assertEqual(len(rec), 1)
        self.assertEqual(rec[0]["cross_age_s"], 20.0)

    def test_no_tick_result_keeps_1min_reject(self):
        for tick in (None, TickTriggerResult(False, reason="TICK_ABSTAIN_NO_FILE")):
            logs, recs = [], []
            self.assertFalse(r005._008_hybrid_1min_trigger(self._ctx(_frame_1min(), tick, logs, recs)))
            self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_NO_BB_MID_GOLDEN_CROSS", logs[-1])

    def test_mode_off_ignores_tick(self):
        logs, recs = [], []
        with mock.patch.object(r005, "TICK_TRIGGER_MODE", "off"):
            self.assertFalse(r005._008_hybrid_1min_trigger(self._ctx(_frame_1min(), self._tick_ok(), logs, recs)))

    def test_tick_path_still_rejected_by_live_drop(self):
        logs, recs = [], []
        ctx = self._ctx(_frame_1min(), self._tick_ok(), logs, recs)
        ctx.price = 41700.0  # -0.6% vs confirmed 41,950
        self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_LIVE_DROP_", logs[-1])

    def test_tick_guard_failure_reported_when_1min_failed(self):
        logs, recs = [], []
        frame = _frame_1min(close=41700.0, open_=41900.0)
        self.assertFalse(r005._008_hybrid_1min_trigger(self._ctx(frame, self._tick_ok(), logs, recs)))
        self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_CANDLE_NOT_BULLISH", logs[-1])

    def test_may_replace_rules(self):
        ctx = SimpleNamespace(current_dt=self.NOW)
        self.assertTrue(r005._tick_may_replace_1min(ctx, False, "1MIN_NO_BB_MID_GOLDEN_CROSS", {}))
        self.assertTrue(r005._tick_may_replace_1min(ctx, False, "1MIN_CONTINUATION_BELOW_BB_MID", {}))
        self.assertFalse(r005._tick_may_replace_1min(ctx, False, "1MIN_CHASE_BUY_BB_GAP_1.2%_GT_0.8%", {}))
        self.assertTrue(r005._tick_may_replace_1min(ctx, True, "1MIN_BB_MID_GOLDEN_CROSS_LOOKBACK", {"path": "cross"}))

    def _fresh_1min_cross(self):
        """Fresh 1-min cross: prev close below mid, confirmed close 42,050 above mid 42,000."""
        frame = _frame_1min(prev_close=41900.0, close=42050.0, bb_mid=42000.0, open_=41950.0)
        return frame

    def test_fresh_1min_cross_used_when_live_guard_passes(self):
        logs, recs = [], []
        ctx = self._ctx(self._fresh_1min_cross(), self._tick_ok(), logs, recs)
        ctx.price = 42050.0
        ctx.current_dt = datetime(2026, 10, 29, 10, 41, 20)
        self.assertTrue(r005._008_hybrid_1min_trigger(ctx), logs)
        self.assertEqual(ctx.trigger_info["path"], "cross")  # 1-min path keeps priority

    def test_tick_path_used_when_1min_fails_live_bb_mid(self):
        # Codex case: confirmed 42,050 > mid 42,000, live 41,950 (-0.24%) -> 1-min path 1MIN_LIVE_BELOW_BB_MID,
        # tick path (no BB-mid requirement, drop within 0.4%) passes
        for tick, expect in ((self._tick_ok(), True), (None, False)):
            logs, recs = [], []
            ctx = self._ctx(self._fresh_1min_cross(), tick, logs, recs)
            ctx.current_dt = datetime(2026, 10, 29, 10, 41, 20)
            self.assertEqual(r005._008_hybrid_1min_trigger(ctx), expect, logs)
            if expect:
                self.assertEqual(ctx.trigger_info["path"], "tick")
                rec = [r for r in recs if r.get("result") == "TICK_CROSS_PASS"][0]
                self.assertEqual(rec["one_min_reject"], "HYBRID_1MIN_TRIGGER_1MIN_LIVE_BELOW_BB_MID")
            else:
                self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_LIVE_BELOW_BB_MID", logs[-1])

    def test_both_paths_fail_reports_1min_reason(self):
        logs, recs = [], []
        ctx = self._ctx(self._fresh_1min_cross(), self._tick_ok(), logs, recs)
        ctx.price = 41800.0  # -0.59%: both live guards reject
        ctx.current_dt = datetime(2026, 10, 29, 10, 41, 20)
        self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_LIVE_DROP_", logs[-1])

    def test_tick_service_exception_falls_back_to_1min(self):
        logs, recs = [], []
        ctx = self._ctx(_frame_1min(), None, logs, recs)

        def boom(code, now):
            raise RuntimeError("feed down")
        ctx.services.tick_trigger = boom
        self.assertFalse(r005._008_hybrid_1min_trigger(ctx))
        self.assertTrue(any("evaluate failed: feed down" in m for m in logs))
        self.assertIn("HYBRID_1MIN_TRIGGER_1MIN_NO_BB_MID_GOLDEN_CROSS", logs[-1])


if __name__ == "__main__":
    unittest.main()


class TestSessionEdgeBars(unittest.TestCase):
    """[2026-11-02] prior-day warmup drops only invalid bars at a day edge."""

    def _bar(self, ts, invalid=False):
        return TickBar("326030", ts, ts, 1.0, 1.0, 1.0, 1.0, n_ticks=60, closed_at=ts, invalid=invalid)

    def test_edges_dropped_inner_gap_kept(self):
        from r013_tick_trigger import _drop_session_edge_bars
        d1, d2 = datetime(2026, 10, 30, 13, 51), datetime(2026, 11, 2, 9, 0, 8)
        bars = ([self._bar(d1, True)] + [self._bar(d1 + timedelta(minutes=i)) for i in range(1, 4)]
                + [self._bar(d1 + timedelta(minutes=5), True)]               # prior day's unfinished last bar
                + [self._bar(d2, True)]                                       # opening-auction print
                + [self._bar(d2 + timedelta(minutes=i)) for i in range(1, 3)]
                + [self._bar(d2 + timedelta(minutes=3), True)]               # gap inside today's session
                + [self._bar(d2 + timedelta(minutes=4))])
        out = _drop_session_edge_bars(bars, d2.date())
        self.assertEqual(len(out), len(bars) - 3)
        self.assertEqual(sum(b.invalid for b in out), 1)
        self.assertTrue(any(b.invalid and b.end_ts == d2 + timedelta(minutes=3) for b in out))
