# -*- coding: utf-8 -*-
"""VWAP trade (r029, 2026-11-27): the four-step buy signal and the paper runner built on r022."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("AUTO_TRADING_LIVE_OVERRIDES", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r029_vwap_trade as r029  # noqa: E402

P = r029.VwapParams()


def bars(closes, start="2026-11-27 09:01", vol=1000.0):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    c = np.array(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": np.full(len(c), vol)}, index=idx)


def tick(ok=True, cross_close=103.0, reason="TICK60_NO_CROSS"):
    c = SimpleNamespace(ok=ok, cross_close=cross_close, reason=reason)
    return SimpleNamespace(ok=ok, label="TICK60_MA5_MA15" if ok else "", reason=reason, crosses={60: c},
                           cross_time=datetime(2026, 11, 27, 9, 30, 20) if ok else None)


RISING = list(np.linspace(100, 103, 30))          # 09:01 .. 09:30, steadily up -> above VWAP, MA5 and SMA20 rising
NOW = datetime(2026, 11, 27, 9, 30, 30)


class GapAndTickOptionTest(unittest.TestCase):
    """[2026-11-28] VWAP_TRADE_MIN_OPEN_GAP_PCT (stage low_gap) and VWAP_TRADE_TICK_TRIGGER"""

    def test_running_defaults_gap_one_percent_and_no_tick_condition(self):
        p = r029.VwapParams.from_config(SimpleNamespace())
        self.assertEqual((p.tick_trigger, p.min_open_gap_pct), (False, 1.0))
        q = r029.VwapParams.from_config(r001)
        self.assertEqual((q.tick_trigger, q.min_open_gap_pct), (bool(r001.VWAP_TRADE_TICK_TRIGGER), float(r001.VWAP_TRADE_MIN_OPEN_GAP_PCT)))
        # the dataclass itself keeps the earlier behaviour
        self.assertEqual((P.tick_trigger, P.min_open_gap_pct), (True, 0.0))

    def test_gap_below_the_floor_or_unknown_is_no_buy(self):
        p = r029.VwapParams(min_open_gap_pct=1.0)
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, tick(), p)[1], "low_gap")                       # not known
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, tick(), p, open_gap_pct=0.99)[1], "low_gap")
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, tick(), p, open_gap_pct=float("nan"))[1], "low_gap")
        ok, stage, info = r029.vwap_signal(bars(RISING), 103.1, NOW, tick(), p, open_gap_pct=1.0)
        self.assertEqual((ok, stage, info["open_gap_pct"]), (True, "signal", 1.0))
        self.assertEqual(r029.trace_fields(stage, info)["open_gap"], 1.0)

    def test_tick_condition_off_needs_no_cross_and_keeps_the_other_guards(self):
        p = r029.VwapParams(tick_trigger=False)
        ok, stage, info = r029.vwap_signal(bars(RISING), 103.1, NOW, None, p)
        self.assertEqual((ok, stage, info["tick"]), (True, "signal", "TICK_OFF"))
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, tick(ok=False), p)[1], "signal")
        # the limit against the cross bar is part of the tick condition; the VWAP distance and the drop guard stay
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, tick(cross_close=100.0), p)[1], "signal")
        self.assertEqual(r029.vwap_signal(bars(RISING), 110.0, NOW, None, p)[1], "chase")
        self.assertEqual(r029.vwap_signal(bars(RISING), 102.0, NOW, None, p)[1], "price_drop")


class SignalTest(unittest.TestCase):
    def test_all_four_steps_pass(self):
        ok, stage, info = r029.vwap_signal(bars(RISING), 103.1, NOW, tick(), P)
        self.assertEqual((ok, stage), (True, "signal"))
        self.assertTrue(info["ma5_up"] and info["bb_mid_up"])
        self.assertGreater(info["vwap_gap_pct"], 0)

    def test_step1_below_vwap(self):
        falling = list(np.linspace(103, 100, 30))
        self.assertEqual(r029.vwap_signal(bars(falling), 100.0, NOW, tick(), P)[1], "below_vwap")
        # the bar closed above VWAP but the live price is already under it
        self.assertEqual(r029.vwap_signal(bars(RISING), 100.5, NOW, tick(), P)[1], "below_vwap")

    def test_step2_needs_a_rising_ma5_or_bb_middle(self):
        flat_top = list(np.linspace(100, 104, 8)) + [104.0] * 22        # above VWAP, MA5 flat, SMA20 flat at the end
        self.assertEqual(r029.vwap_signal(bars(flat_top), 104.0, NOW, tick(cross_close=104.0), P)[1], "no_trend")

    def test_step3_needs_the_tick_cross(self):
        ok, stage, info = r029.vwap_signal(bars(RISING), 103.1, NOW, tick(ok=False, reason="TICK60_CROSS_EXPIRED_75s"), P)
        self.assertEqual((ok, stage), (False, "no_tick_cross"))
        self.assertIn("EXPIRED", info["tick"])
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, None, P)[1], "no_tick_cross")

    def test_step4_chase_drop_and_freshness(self):
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.6, NOW, tick(cross_close=103.0), P)[1], "chase")      # +0.58 % over the cross
        far = r029.VwapParams(chase_max_pct=50.0, max_vwap_gap_pct=1.0)
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, tick(), far)[1], "chase")                     # too far above VWAP
        self.assertEqual(r029.vwap_signal(bars(RISING), 102.5, NOW, tick(cross_close=102.5), P)[1], "price_drop")  # -0.49 % under the bar
        late = datetime(2026, 11, 27, 9, 33, 0)
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, late, tick(), P)[1], "stale")

    def test_history_must_start_at_the_open(self):
        later = datetime(2026, 11, 27, 10, 30, 30)
        self.assertEqual(r029.vwap_signal(bars(RISING, start="2026-11-27 10:01"), 103.1, later, tick(), P)[1], "no_history")
        self.assertEqual(r029.vwap_signal(bars(RISING[:10]), 101.0, NOW, tick(), P)[1], "no_history")

    def test_session_vwap(self):
        df = bars([100.0, 110.0])
        df["volume"] = [1.0, 3.0]
        self.assertAlmostEqual(r029.session_vwap(df), 107.5)
        df["volume"] = 0.0
        self.assertTrue(np.isnan(r029.session_vwap(df)))


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.logs: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def _runner(self, tick_result, **kw):
        return r029.VwapPaperRunner(self.out, "20261127", self.logs.append, P, lambda code, now: tick_result, **kw)

    def _records(self):
        f = self.out / "vwap_paper_20261127.jsonl"
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []

    def test_buys_on_the_signal_and_sells_at_the_target(self):
        r = self._runner(tick())
        r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)
        recs = self._records()
        self.assertEqual([x["event"] for x in recs], ["ENTRY"])
        e = recs[0]
        self.assertEqual((e["code"], e["prob"], e["model_id"]), ("005930", None, r029.MODEL_ID))
        self.assertAlmostEqual(e["tp_px"], e["entry"] * 1.03)
        self.assertAlmostEqual(e["sl_px"], e["entry"] * (1 - 0.0127))
        self.assertIn("VWAP_gap", e["trigger"])
        self.assertTrue(any("[VWAP PAPER  ]" in m and "PAPER BUY" in m for m in self.logs))
        self.assertFalse(any("[EDGE ONLY" in m for m in self.logs))
        r.on_symbol("005930", "삼성전자", datetime(2026, 11, 27, 9, 40, 0), bars(RISING), e["tp_px"] + 1)
        self.assertEqual([x["event"] for x in self._records()], ["ENTRY", "EXIT"])
        self.assertEqual(self._records()[1]["reason"], "tp")

    def test_gap_floor_in_the_runner_uses_the_open_against_the_previous_close(self):
        import dataclasses
        r = self._runner(tick())
        r.params = dataclasses.replace(r.params, min_open_gap_pct=1.0, tick_trigger=False)
        r.tick_eval = lambda code, now: (_ for _ in ()).throw(AssertionError("the tick feed is not asked when the condition is off"))
        r._prev_close["005930"] = 99.5            # open 100 -> +0.50 %
        r._prev_close["000660"] = 98.0            # open 100 -> +2.04 %
        r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)
        r.on_symbol("000660", "SK하이닉스", NOW, bars(RISING), 103.1)
        r.on_symbol("035420", "NAVER", NOW, bars(RISING), 103.1)          # previous close not known
        self.assertEqual([x["code"] for x in self._records()], ["000660"])
        self.assertEqual((r.sym_state["005930"]["stage"], r.sym_state["035420"]["stage"]), ("low_gap", "low_gap"))

    def test_previous_close_comes_from_the_dated_data_folders_not_the_output_folder(self):
        """[Codex review] r003 gives the runner live_runtime as its folder; the daily files are in data/YYYYMMDD"""
        import dataclasses
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "20261126").mkdir()
            (root / "20261126" / "005930_삼성전자_daily.csv").write_text("date,open,high,low,close\n20261125,97,98,96,97\n20261126,98,99,97,98\n", encoding="utf-8-sig")
            (root / "20261126" / "000660_SK하이닉스_daily.csv").write_text("date,open,high,low,close\n20261126,99,100,99,99.6\n", encoding="utf-8-sig")
            p = dataclasses.replace(P, min_open_gap_pct=1.0, tick_trigger=False)
            r = r029.VwapPaperRunner(self.out, "20261127", self.logs.append, p, None, daily_root=root)
            (root / "20261128").mkdir()                                       # a later folder without daily files (a weekend)
            (root / "20261128" / "today_buys.txt").write_text("", encoding="utf-8")
            self.assertEqual(r._prev_close_of("005930"), 98.0)
            monday = r029.VwapPaperRunner(self.out, "20261130", self.logs.append, p, None, daily_root=root)
            self.assertEqual(monday._prev_close_of("005930"), 98.0)          # found in 20261126, past the empty folder
            # [Codex review] a file that stops before the reference day is not the previous close (005380: last row 11/25)
            (root / "20261126" / "005380_현대차_daily.csv").write_text("date,open,high,low,close\n20261125,97,98,96,97\n", encoding="utf-8-sig")
            self.assertTrue(np.isnan(monday._prev_close_of("005380")))
            # a read that failed is tried again later, not kept for the day
            with mock.patch.object(r029.time, "monotonic", return_value=r029.time.monotonic() + r029.PREV_RETRY_S + 1):
                (root / "20261126" / "005380_현대차_daily.csv").write_text("date,open,high,low,close\n20261126,97,98,96,97.5\n", encoding="utf-8-sig")
                self.assertEqual(monday._prev_close_of("005380"), 97.5)
            r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)          # open 100 / 98 = +2.04 %
            r.on_symbol("000660", "SK하이닉스", NOW, bars(RISING), 103.1)        # open 100 / 99.6 = +0.40 %
            r.on_symbol("035420", "NAVER", NOW, bars(RISING), 103.1)            # no daily file
            self.assertEqual([x["code"] for x in self._records()], ["005930"])
            self.assertEqual((r.sym_state["000660"]["stage"], r.sym_state["035420"]["stage"]), ("low_gap", "low_gap"))
        # without the folder (the earlier call) nothing is known: no buy, never a pass
        r2 = r029.VwapPaperRunner(self.out, "20261127", self.logs.append, p, None)
        r2.on_symbol("051910", "LG화학", NOW, bars(RISING), 103.1)
        self.assertEqual(r2.sym_state["051910"]["stage"], "low_gap")

    def test_the_gap_needs_the_opening_bar_and_an_exact_floor_passes(self):
        import dataclasses
        p = dataclasses.replace(P, min_open_gap_pct=0.5, tick_trigger=False)
        r = r029.VwapPaperRunner(self.out, "20261127", self.logs.append, p, None)
        r._prev_close["005930"] = 100.0 / 1.005      # open 100 is exactly +0.5 % (0.4999999... in floating point)
        r._prev_close["000660"] = 90.0
        r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)
        self.assertEqual([x["code"] for x in self._records()], ["005930"])
        late = bars(RISING, start="2026-11-27 09:02")                      # the 09:00 bar is missing: its open is not the day's open
        r.on_symbol("000660", "SK하이닉스", datetime(2026, 11, 27, 9, 31, 30), late, 103.1)
        self.assertEqual(r.sym_state["000660"]["stage"], "low_gap")
        self.assertEqual(r029.vwap_signal(bars(RISING), 103.1, NOW, None, p, open_gap_pct=float("inf"))[1], "low_gap")

    def test_no_buy_without_the_tick_cross_and_the_stage_says_why(self):
        r = self._runner(tick(ok=False))
        r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)
        self.assertEqual(self._records(), [])
        self.assertEqual(r.sym_state["005930"]["stage"], "no_tick_cross")

    def test_limits_and_funds(self):
        r = self._runner(tick(), max_positions=1)
        r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)
        r.on_symbol("000660", "SK하이닉스", NOW, bars(RISING), 103.1)
        r.on_symbol("069500", "KODEX 200", NOW, bars(RISING), 103.1)
        self.assertEqual([x["code"] for x in self._records()], ["005930"])
        self.assertEqual(r.sym_state["000660"]["stage"], "limit")

    def test_status_file_and_restore(self):
        r = self._runner(tick())
        r.on_symbol("005930", "삼성전자", NOW, bars(RISING), 103.1)
        self.assertTrue(r.write_status(NOW, force=True))
        st = json.loads((self.out / "vwap_paper_status_20261127.json").read_text(encoding="utf-8"))
        self.assertEqual((st["model_id"], st["threshold"], st["label"]["tp"]), (r029.MODEL_ID, None, 3.0))
        again = self._runner(tick())
        self.assertIn("005930", again.positions)             # restored from the day's file after a restart


class FactoryTest(unittest.TestCase):
    def _cfg(self, mode, ticks):
        return SimpleNamespace(VWAP_TRADE_MODE=mode, TICK_DATA_DIR=ticks)

    def test_off_makes_no_runner_and_paper_makes_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, ticks = Path(tmp) / "live", Path(tmp) / "ticks"
            out.mkdir()
            ticks.mkdir()
            logs: list[str] = []
            self.assertIsNone(r029.make_runner(self._cfg("off", ticks), out, "20261127", logs.append))
            run = r029.make_runner(self._cfg("paper", ticks), out, "20261127", logs.append)
            self.assertIsInstance(run, r029.VwapPaperRunner)
            self.assertFalse(run.close_only)
            self.assertTrue(any("paper mode enabled" in m for m in logs))

    def test_live_orders_on_means_no_new_paper_buys(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, ticks = Path(tmp) / "live", Path(tmp) / "ticks"
            out.mkdir()
            ticks.mkdir()
            cfg = SimpleNamespace(VWAP_TRADE_MODE="paper", VWAP_TRADE_LIVE_ORDERS=True, TICK_DATA_DIR=ticks)
            self.assertIsNone(r029.make_runner(cfg, out, "20261127", lambda m: None))

    def test_the_live_path_passes_the_days_vwap(self):
        late = bars(RISING, start="2026-11-27 12:31")                       # r003's cache no longer starts at the open
        now = datetime(2026, 11, 27, 13, 0, 30)
        self.assertEqual(r029.vwap_signal(late, 103.1, now, tick(), P)[1], "no_history")
        self.assertEqual(r029.vwap_signal(late, 103.1, now, tick(), P, vwap=102.0)[:2], (True, "signal"))
        self.assertEqual(r029.vwap_signal(late, 103.1, now, tick(), P, vwap=104.0)[1], "below_vwap")

    def test_defaults(self):
        self.assertEqual(r001.VWAP_TRADE_MODE, "off")
        self.assertEqual((r001.VWAP_TRADE_TP_PCT, r001.VWAP_TRADE_SL_PCT, r001.VWAP_TRADE_HORIZON_MIN), (3.0, 1.27, 300))
        self.assertEqual((r001.VWAP_TRADE_TICK_BAR_SIZE, r001.VWAP_TRADE_TICK_MA_FAST, r001.VWAP_TRADE_TICK_MA_SLOW), (60, 5, 15))
        self.assertTrue(r001.RULE_BUY_ENABLED)
        self.assertEqual(r001.EDGE_ONLY_MODE, "paper")


if __name__ == "__main__":
    unittest.main()
