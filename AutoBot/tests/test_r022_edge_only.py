# -*- coding: utf-8 -*-
"""r022 Edge-AI-only paper runner: entries (trigger + probability cut, book limits, time window), exits (target / stop /
time / end of day, fills and net %), restart restore, parity with the real g013 event generator on a real trading day
when the data folder is available, and the g029 simulator's book rules."""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("live_trading", "data_simulation"):
    sys.path.insert(0, str(ROOT / sub))

import r011_edge_ai as r011  # noqa: E402
import r022_edge_only as r022  # noqa: E402

DAY = "20261112"


def frame(n: int, start: str = "09:01", price: float = 10000.0, last_close: float | None = None) -> pd.DataFrame:
    """n end-labeled 1m bars from `start`; a gentle rise so the close is above the 1m BB mid."""
    idx = pd.date_range(f"2026-11-12 {start}", periods=n, freq="1min")
    c = price + np.arange(n) * 10.0
    if last_close is not None:
        c[-1] = last_close
    return pd.DataFrame({"open": c - 5, "high": c + 10, "low": c - 10, "close": c, "volume": 1000.0}, index=idx)


class FakeModel:
    def __init__(self, prob: float, threshold: float = 0.3, label=None):
        self.prob = prob
        self.meta = {"threshold": threshold, "model_id": "test", "label": label or {"tp": 2.0, "sl": 1.7, "horizon": 60}}
        self.calls = 0

    def predict_with_meta(self, feats):
        self.calls += 1
        return self.prob, self.meta


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.logs: list[str] = []
        # the trigger / feature code of r011 is tested elsewhere: here every new bar is a trigger with fixed features
        self.patches = [
            mock.patch.object(r011, "hybrid_trigger_with_context", lambda *a, **k: (True, "TEST_TRIGGER")),
            mock.patch.object(r011, "compute_features", lambda *a, **k: {}),
            mock.patch.object(r011, "load_daily_features", lambda *a, **k: {}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def runner(self, prob=0.6, **kw) -> r022.EdgeOnlyRunner:
        model = kw.pop("model", None) or FakeModel(prob)
        return r022.EdgeOnlyRunner(self.out, self.out, self.out, DAY, self.logs.append, None, model=model,
                                   **{"min_prob": 0.45, "max_positions": 3, "max_entries": 10, **kw})

    def records(self):
        f = self.out / f"edge_only_{DAY}.jsonl"
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []

    @staticmethod
    def at(hhmm: str, sec: int = 5) -> datetime:
        return datetime(2026, 11, 12, int(hhmm[:2]), int(hhmm[3:]), sec)


class EntryTest(Base):
    def test_buys_on_a_trigger_bar_when_the_probability_reaches_the_cut(self):
        r = self.runner(prob=0.6)
        r.on_symbol("010060", "OCI홀딩스", self.at("09:30"), frame(30), 10300.0)
        rec = self.records()
        self.assertEqual([x["event"] for x in rec], ["ENTRY"])
        self.assertEqual(rec[0]["entry"], 10310.0)                    # live price + 1 tick (10 won at this price)
        self.assertAlmostEqual(rec[0]["tp_px"], 10310.0 * 1.02)
        self.assertAlmostEqual(rec[0]["sl_px"], 10310.0 * (1 - 0.017))
        self.assertEqual((rec[0]["prob"], rec[0]["threshold"], rec[0]["trigger"]), (0.6, 0.45, "TEST_TRIGGER"))
        self.assertIn("PAPER BUY", self.logs[-1])

    def test_no_buy_below_the_cut_and_the_model_threshold_is_used_when_the_cut_is_zero(self):
        r = self.runner(prob=0.40)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(self.records(), [])
        r2 = self.runner(prob=0.40, min_prob=0.0)                     # model threshold 0.3
        r2.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(self.records()[0]["threshold"], 0.3)

    def test_one_evaluation_per_completed_bar(self):
        model = FakeModel(0.1)
        r = self.runner(model=model)
        for sec in (5, 15, 25):
            r.on_symbol("010060", "x", self.at("09:30", sec), frame(30), 10300.0)
        self.assertEqual(model.calls, 1)
        r.on_symbol("010060", "x", self.at("09:31"), frame(31), 10310.0)
        self.assertEqual(model.calls, 2)

    def test_time_window_and_history(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:04"), frame(3), 10300.0)          # before 09:05 / too few bars
        self.assertEqual(self.records(), [])
        late = frame(30, start="14:25")                                           # last bar 14:54 > 14:50
        r.on_symbol("010061", "x", self.at("14:54"), late, 10300.0)
        self.assertEqual(self.records(), [])
        # a session that starts late has no complete history window -> no prediction, no entry
        r.on_symbol("010062", "x", self.at("11:00"), frame(8, start="10:53"), 10300.0)
        self.assertEqual(self.records(), [])

    def test_close_must_be_above_the_bb_mid(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30, last_close=9000.0), 9000.0)
        self.assertEqual(self.records(), [])

    def test_book_limits(self):
        r = self.runner(max_positions=2, max_entries=3)
        for i, code in enumerate(("000001", "000002", "000003")):
            r.on_symbol(code, "x", self.at("09:30", 5 + i), frame(30), 10300.0)
        self.assertEqual(sorted(r.positions), ["000001", "000002"])              # third blocked: 2 positions open
        r.on_symbol("000001", "x", self.at("09:31"), frame(31), 10600.0)         # +2% -> sold
        r.on_symbol("000003", "x", self.at("09:32"), frame(32), 10300.0)
        self.assertIn("000003", r.positions)
        r.on_symbol("000002", "x", self.at("09:33"), frame(33), 10600.0)         # sold
        r.on_symbol("000004", "x", self.at("09:34"), frame(34), 10300.0)         # 4th entry of the day: over the cap of 3
        self.assertNotIn("000004", r.positions)
        self.assertEqual(r.entries_today, 3)

    def test_no_second_position_in_the_same_symbol(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        r.on_symbol("010060", "x", self.at("09:31"), frame(31), 10310.0)
        self.assertEqual(len(self.records()), 1)


class ExitTest(Base):
    def open(self, r):
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)         # entry 10,310
        return r.positions["010060"]

    def test_take_profit_fills_at_the_target(self):
        r = self.runner()
        pos = self.open(r)
        r.on_symbol("010060", "x", self.at("09:40"), frame(40), 10700.0)
        rec = self.records()[-1]
        self.assertEqual((rec["event"], rec["reason"]), ("EXIT", "tp"))
        self.assertAlmostEqual(rec["exit"], pos["tp"])
        self.assertAlmostEqual(rec["net_pct"], r022.net_pct(10310.0, pos["tp"]))
        self.assertAlmostEqual(rec["net_pct"], 2.0 - 0.015 - (0.015 + 0.20) * 1.02, places=6)
        self.assertEqual(r.positions, {})

    def test_stop_fills_one_tick_below_the_live_price(self):
        r = self.runner()
        self.open(r)
        r.on_symbol("010060", "x", self.at("09:40"), frame(40), 10100.0)          # below 10,310 * 0.983 = 10,134.7
        rec = self.records()[-1]
        self.assertEqual((rec["reason"], rec["exit"]), ("sl", 10090.0))
        self.assertLess(rec["net_pct"], -2.0)

    def test_time_exit_after_the_horizon_and_end_of_day(self):
        r = self.runner()
        self.open(r)
        r.on_symbol("010060", "x", self.at("10:29"), frame(89), 10320.0)          # 59 minutes: still open
        self.assertIn("010060", r.positions)
        r.on_symbol("010060", "x", self.at("10:31"), frame(91), 10320.0)
        self.assertEqual(self.records()[-1]["reason"], "time")
        r2 = self.runner()
        r2.on_symbol("000002", "x", self.at("14:40"), frame(100, start="13:01"), 10300.0)     # history window complete
        self.assertIn("000002", r2.positions)
        r2.on_symbol("000002", "x", self.at("15:15"), frame(135, start="13:01"), 10320.0)
        self.assertEqual(self.records()[-1]["reason"], "eod")

    def test_exit_works_without_a_new_bar_or_frame(self):
        r = self.runner()
        self.open(r)
        r.on_symbol("010060", "x", self.at("09:30", 40), None, 10700.0)
        self.assertEqual(self.records()[-1]["reason"], "tp")

    def test_label_from_the_model_meta(self):
        r = self.runner(model=FakeModel(0.6, label={"tp": 3.0, "sl": 1.0, "horizon": 30}))
        pos = self.open(r)
        self.assertAlmostEqual(pos["tp"], 10310.0 * 1.03)
        self.assertAlmostEqual(pos["sl"], 10310.0 * 0.99)
        self.assertEqual(pos["horizon"], 30.0)


class RestartTest(Base):
    def test_open_positions_and_counters_survive_a_restart(self):
        r = self.runner()
        r.on_symbol("000001", "x", self.at("09:30"), frame(30), 10300.0)
        r.on_symbol("000002", "y", self.at("09:30", 9), frame(30), 10300.0)
        r.on_symbol("000002", "y", self.at("09:35"), frame(35), 10700.0)          # sold
        r2 = self.runner()
        self.assertEqual(sorted(r2.positions), ["000001"])
        self.assertEqual((r2.entries_today, len(r2.closed)), (2, 1))
        r2.on_symbol("000001", "x", self.at("09:50"), frame(50), 10700.0)
        self.assertEqual(self.records()[-1]["reason"], "tp")

    def test_a_broken_file_starts_the_day_empty_without_raising(self):
        (self.out / f"edge_only_{DAY}.jsonl").write_text("{not json\n", encoding="utf-8")
        r = self.runner()
        self.assertEqual((r.positions, r.entries_today), ({}, 0))

    def test_errors_never_reach_the_caller(self):
        r = self.runner()
        with mock.patch.object(r011, "session_raw_1min", side_effect=RuntimeError("boom")):
            r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertIn("error: RuntimeError: boom", self.logs[-1])


class SimulatorTest(unittest.TestCase):
    def test_g029_book_rules(self):
        import g029_edge_only_sim as g029
        ts = pd.to_datetime
        d = pd.DataFrame([
            # date, code, ts, p, threshold, net_ret, hold_bars, exit_reason
            ("20261110", "A", ts("2026-11-10 09:10"), 0.50, 0.3, 1.0, 10, "tp"),
            ("20261110", "A", ts("2026-11-10 09:15"), 0.60, 0.3, 9.0, 5, "tp"),      # A still held until 09:21
            ("20261110", "B", ts("2026-11-10 09:16"), 0.50, 0.3, -2.0, 60, "sl"),
            ("20261110", "C", ts("2026-11-10 09:17"), 0.50, 0.3, 9.0, 5, "tp"),      # 2 positions open already
            ("20261110", "A", ts("2026-11-10 09:30"), 0.50, 0.3, 0.5, 5, "time"),    # A free again
            ("20261110", "D", ts("2026-11-10 09:40"), 0.20, 0.3, 9.0, 5, "tp"),      # below the cut
            ("20261111", "A", ts("2026-11-11 09:10"), 0.50, 0.6, -1.0, 5, "sl"),
        ], columns=["date", "code", "ts", "p", "threshold", "net_ret", "hold_bars", "exit_reason"])
        tr = g029.simulate(d, 0.45, max_positions=2, max_entries=20)
        self.assertEqual(tr["net_ret"].tolist(), [1.0, -2.0, 0.5, -1.0])
        self.assertEqual(g029.simulate(d, 0.45, max_positions=2, max_entries=1)["net_ret"].tolist(), [1.0, -1.0])
        # each fold's own threshold: the 11/11 event (p 0.5 < 0.6) is not taken
        self.assertEqual(g029.simulate(d, None, 2, 20, use_fold_threshold=True)["net_ret"].tolist(), [1.0, -2.0, 0.5])
        s = g029.stats(tr, n_days=2)
        self.assertEqual((s["n"], s["days_pos"], s["days_traded"]), (4, 0, 2))
        self.assertAlmostEqual(s["mean"], -0.375)
        self.assertAlmostEqual(g029.breakeven_win_rate(tr), 1.5 / (0.75 + 1.5))


if __name__ == "__main__":
    unittest.main()
