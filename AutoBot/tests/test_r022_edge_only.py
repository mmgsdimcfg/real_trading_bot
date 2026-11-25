# -*- coding: utf-8 -*-
"""r022 Edge-AI-only paper runner: entries (trigger + probability cut, book limits, time window), exits (target / stop /
time / end of day, fills and net %), restart restore, parity with the real g013 event generator on a real trading day
when the data folder is available, and the g029 simulator's book rules.
[2026-11-20] no ETF/ETN buys, no re-entry after a same-day stop (also after a restart), poll_orphans exits a paper position
r003 no longer evaluates (bought by hand / left the watchlist)."""
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
        # [2026-11-25] no 15:15 exit any more: the regular close is the 15:20 auction (session_close), filled at 15:30
        r2.on_symbol("000002", "x", self.at("15:15"), frame(135, start="13:01"), 10320.0)
        self.assertIn("000002", r2.positions)
        r2.on_symbol("000002", "x", self.at("15:20"), frame(140, start="13:01"), 10320.0)   # waits for the auction print
        self.assertIn("000002", r2.positions)
        self.assertEqual(r2.session_close(self.at("15:32")), 1)                             # no print by 15:32: estimated
        last = self.records()[-1]
        self.assertEqual((last["reason"], last["fill"], last["estimated"], last["exit"]), ("close_regular", "last", True, 10310.0))

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

    def test_user_exit_overrides_replace_the_model_label(self):
        # [2026-11-25] r001 EDGE_ONLY_TP_PCT / SL_PCT / HORIZON_MIN > 0 win over the model label; 0 / bad values keep the model's
        r = self.runner(model=FakeModel(0.6, label={"tp": 3.0, "sl": 1.0, "horizon": 30}), tp_pct=1.5, sl_pct=0, horizon_min=90)
        self.assertEqual(r._label(), {"tp": 1.5, "sl": 1.0, "horizon": 90.0})
        self.assertEqual(r._model_label(), {"tp": 3.0, "sl": 1.0, "horizon": 30.0})
        pos = self.open(r)
        self.assertAlmostEqual(pos["tp"], 10310.0 * 1.015)
        self.assertAlmostEqual(pos["sl"], 10310.0 * 0.99)
        self.assertEqual(pos["horizon"], 90.0)
        r2 = self.runner(model=FakeModel(0.6, label={"tp": 3.0, "sl": 1.0, "horizon": 30}), tp_pct=-1, sl_pct="x",
                         horizon_min=float("nan"))
        self.assertEqual(r2._label(), r2._model_label())


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



class PaperReviewRulesTest(Base):
    """[2026-11-20] user decisions after the 20261120 paper review"""

    def test_no_fund_products(self):
        r = self.runner()
        r.on_symbol("233740", "KODEX 코스닥150레버리지", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(self.records(), [])
        self.assertEqual(r.sym_state["233740"]["stage"], "fund")
        r2 = self.runner(exclude_funds=False)
        r2.on_symbol("233740", "KODEX 코스닥150레버리지", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual([x["event"] for x in self.records()], ["ENTRY"])

    def test_no_reentry_after_a_stop_also_after_a_restart(self):
        r = self.runner()
        r.on_symbol("058610", "에스피지", self.at("09:30"), frame(30), 10300.0)
        r.on_symbol("058610", "에스피지", self.at("09:31"), frame(31), 10000.0)      # -3% -> sl
        self.assertEqual(self.records()[-1]["reason"], "sl")
        r.on_symbol("058610", "에스피지", self.at("09:32"), frame(32), 10300.0)
        self.assertNotIn("058610", r.positions)
        self.assertEqual(r.sym_state["058610"]["stage"], "sl_today")
        r2 = self.runner()                                                           # r003 restart: restored from the file
        self.assertIn("058610", r2.sl_today)
        r2.on_symbol("058610", "에스피지", self.at("09:40"), frame(40), 10300.0)
        self.assertNotIn("058610", r2.positions)
        # a take-profit does not block
        r2.on_symbol("010060", "x", self.at("09:41"), frame(41), 10300.0)
        r2.on_symbol("010060", "x", self.at("09:42"), frame(42), 10600.0)
        r2.on_symbol("010060", "x", self.at("09:43"), frame(43), 10300.0)
        self.assertIn("010060", r2.positions)

    def test_orphan_position_is_closed_from_the_tick_feed(self):
        class Feed:
            def __init__(self):
                self.price = 10400.0

            def poll(self, code, now):
                return None, self.price

        r = self.runner()
        r.on_symbol("443060", "HD현대마린솔루션", self.at("09:30"), frame(30), 10300.0)
        self.assertIn("443060", r.positions)
        turn = datetime.now()
        feed = Feed()
        self.assertEqual(r.poll_orphans(feed, self.at("09:50"), turn), 1)             # not evaluated this turn -> checked
        self.assertIn("443060", r.positions)                                            # below the target: still held
        feed.price = 10600.0
        r.poll_orphans(feed, self.at("09:53"), datetime.now())
        self.assertNotIn("443060", r.positions)
        self.assertEqual(self.records()[-1]["reason"], "tp")

    def test_orphan_skips_positions_seen_this_turn_and_uses_only_a_fresh_last_price(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        turn = datetime(2000, 1, 1)                                                     # everything was seen after it
        self.assertEqual(r.poll_orphans(None, self.at("09:31"), turn), 0)
        self.assertIn("010060", r.positions)
        # [2026-11-25] no feed: nothing (the remembered price is not re-used - it would re-stamp itself as fresh)
        self.assertEqual(r.poll_orphans(None, self.at("09:31"), datetime.now()), 0)
        self.assertIn("010060", r.positions)
        # the session close still comes (r003 calls session_close every turn)
        self.assertEqual(r.session_close(self.at("15:33")), 1)
        self.assertEqual(self.records()[-1]["reason"], "close_regular")


class SessionCloseTest(Base):
    """[2026-11-25] user decision (Codex review): regular-only holdings close into the 15:20 auction (15:30 print), NXT holdings
    of an NXT runner at 19:59:50; positions are kept across the sessions"""

    def feed(self, auction=None):
        f = mock.Mock()
        f.auction_price.return_value = auction
        f.poll.return_value = (None, None)
        f.last_print.return_value = None
        return f

    def test_regular_holding_is_filled_at_the_auction_print(self):
        r = self.runner()
        r.tick_feed = self.feed()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(r.session_close(self.at("15:19")), 0)
        self.assertEqual(r.session_close(self.at("15:25")), 0)                      # before the 15:30 print
        self.assertIn("010060", r.positions)
        r.tick_feed.auction_price.return_value = 10450.0
        self.assertEqual(r.session_close(self.at("15:30")), 1)
        last = self.records()[-1]
        self.assertEqual((last["reason"], last["exit"], last["fill"], last.get("estimated"), last["nxt"]),
                         ("close_regular", 10450.0, "auction", None, False))

    def test_nxt_runner_keeps_an_nxt_symbol_until_1959_and_closes_a_regular_only_symbol_at_1520(self):
        r = self.runner(nxt=True, model=FakeModel(0.6, label={"tp": 2.0, "sl": 1.7, "horizon": 9999}))
        r.tick_feed = self.feed(auction=10200.0)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=True)
        r.on_symbol("010061", "y", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=False)
        self.assertEqual((r.positions["010060"]["nxt"], r.positions["010061"]["nxt"]), (True, False))
        self.assertEqual(r.session_close(self.at("15:31")), 1)                       # only the regular-only one
        self.assertEqual(self.records()[-1]["code"], "010061")
        self.assertIn("010060", r.positions)
        r.on_symbol("010060", "x", self.at("19:59", 40), None, 10300.0, nxt_tradeable=True)
        self.assertIn("010060", r.positions)
        r.on_symbol("010060", "x", self.at("19:59", 50), None, 10300.0, nxt_tradeable=True)
        last = self.records()[-1]
        self.assertEqual((last["code"], last["reason"], last["exit"], last.get("estimated")), ("010060", "close_nxt", 10290.0, None))

    def test_nxt_close_with_a_stale_price_is_estimated_and_final_closes_everything(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=True)
        self.assertEqual(r.session_close(self.at("19:59", 55)), 1)                    # last price from 09:30: estimated
        self.assertTrue(self.records()[-1]["estimated"])
        r2 = self.runner()
        r2.on_symbol("010061", "y", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(r2.session_close(self.at("12:00"), final=True), 1)           # r003 loop ended (e.g. stopped)
        self.assertEqual(self.records()[-1]["reason"], "close_regular")

    def test_final_close_waits_for_the_auction_print(self):
        r = self.runner()
        r.tick_feed = self.feed()
        calls = []

        def sleep(s):
            calls.append(s)
            r.tick_feed.auction_price.return_value = 10400.0     # the print arrives while waiting
        r.sleep = sleep
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(r.session_close(self.at("15:30", 0), final=True, wait_s=90.0), 1)
        self.assertEqual(calls, [5.0])
        self.assertEqual((self.records()[-1]["exit"], self.records()[-1]["fill"]), (10400.0, "auction"))

    def test_collector_symbols_and_runners_without_nxt_close_at_1520(self):
        r = self.runner(nxt=False)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=True)
        self.assertFalse(r.positions["010060"]["nxt"])
        r2 = self.runner(nxt=True, collector_max_positions=3, collector_max_entries=10)
        r2.on_symbol("010061", "y", self.at("09:30"), frame(30), 10300.0, group="collector", nxt_tradeable=True)
        self.assertFalse(r2.positions["010061"]["nxt"])

    def test_orphan_uses_the_tick_file_only_when_its_last_print_is_fresh(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        feed = mock.Mock()
        feed.poll.return_value = (None, 9000.0)                                  # far below the stop
        feed.last_print.return_value = (9000.0, self.at("09:40", 0))
        self.assertEqual(r.poll_orphans(feed, self.at("09:44"), datetime.now()), 0)   # 4 min old print: not used
        self.assertIn("010060", r.positions)
        self.assertEqual(r.poll_orphans(feed, self.at("09:42"), datetime.now()), 1)
        self.assertEqual(self.records()[-1]["reason"], "sl")
        self.assertEqual(r.poll_orphans(feed, self.at("15:25"), datetime.now()), 0)   # not after the regular close

    def test_restart_after_the_close_still_closes_with_the_buy_records_price(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        r2 = self.runner()                                                        # restarted: no price seen yet
        self.assertEqual(r2.session_close(self.at("19:00"), final=True), 1)
        last = self.records()[-1]
        self.assertEqual((last["reason"], last["exit"], last["estimated"]), ("close_regular", 10290.0, True))

    def test_regular_close_prefers_the_last_regular_print_of_the_tick_file(self):
        r = self.runner()
        r.tick_feed = self.feed()
        r.tick_feed.last_print.return_value = (10500.0, self.at("15:19", 58))
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(r.session_close(self.at("15:32")), 1)
        self.assertEqual((self.records()[-1]["exit"], self.records()[-1]["estimated"]), (10490.0, True))

    def test_auction_is_not_read_before_1530(self):
        r = self.runner()
        r.tick_feed = self.feed(auction=10450.0)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(r.session_close(self.at("15:20", 1)), 0)
        r.tick_feed.auction_price.assert_not_called()                           # no file read in front of the real 15:20 close
        self.assertEqual(r.session_close(self.at("15:30", 2)), 1)

    def test_a_failed_exit_write_keeps_the_position(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        with mock.patch.object(r, "_emit", return_value=False):
            self.assertEqual(r.session_close(self.at("15:33")), 0)
        self.assertIn("010060", r.positions)
        self.assertEqual(r.session_close(self.at("15:34")), 1)

    def test_one_tick_down_follows_the_tick_bands(self):
        self.assertEqual([r022.tick_down(p) for p in (2000.0, 5000.0, 10300.0, 50000.0)], [1999.0, 4995.0, 10290.0, 49950.0])

    def test_final_wait_never_passes_1532(self):
        r = self.runner()
        r.tick_feed = self.feed()
        calls = []
        r.sleep = calls.append
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(r.session_close(self.at("15:31", 58), final=True, wait_s=150.0), 1)
        self.assertEqual(calls, [2.0])                                           # 2 s left until 15:32
        self.assertTrue(self.records()[-1]["estimated"])

    def test_final_close_retries_a_failed_write(self):
        r = self.runner()
        r.sleep = lambda s: None
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        real = r._emit
        results = iter([False, True])
        with mock.patch.object(r, "_emit", side_effect=lambda rec: next(results) and real(rec)):
            self.assertEqual(r.session_close(self.at("16:00"), final=True), 1)
        self.assertNotIn("010060", r.positions)

    def test_nxt_orphan_is_priced_from_the_nxt_tick_file(self):
        r = self.runner(nxt=True, model=FakeModel(0.6, label={"tp": 2.0, "sl": 1.7, "horizon": 9999}))
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=True)
        r.nxt_feed = self.feed()
        r.nxt_feed.poll.return_value = (None, 9000.0)                           # below the stop
        r.nxt_feed.last_print.return_value = (9000.0, self.at("17:00", 0))
        krx = self.feed()
        self.assertEqual(r.poll_orphans(krx, self.at("17:01"), datetime.now()), 1)
        krx.poll.assert_not_called()
        self.assertEqual((self.records()[-1]["reason"], self.records()[-1]["exit"]), ("sl", 8990.0))

    def test_nxt_close_uses_a_fresh_print_of_the_nxt_tick_file(self):
        r = self.runner(nxt=True, model=FakeModel(0.6, label={"tp": 2.0, "sl": 1.7, "horizon": 9999}))
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=True)
        r.nxt_feed = self.feed()
        r.nxt_feed.last_print.return_value = (10500.0, self.at("19:59", 30))
        self.assertEqual(r.session_close(self.at("19:59", 50)), 1)
        last = self.records()[-1]
        self.assertEqual((last["reason"], last["exit"], last.get("estimated")), ("close_nxt", 10490.0, None))

    def test_future_or_stale_prints_are_not_used(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        feed = self.feed()
        feed.poll.return_value = (None, 9000.0)
        feed.last_print.return_value = (9000.0, self.at("09:45", 0))           # after `now`
        self.assertEqual(r.poll_orphans(feed, self.at("09:40"), datetime.now()), 0)
        r2 = self.runner(collector_max_positions=3, collector_max_entries=10)
        feed.codes.return_value = ["010061"]
        feed.last_print.return_value = (9000.0, self.at("09:00", 0))           # 40 min old
        self.assertEqual(r2.poll_collector(feed, self.at("09:40"), set()), 0)

    def test_close_only_runner_closes_but_never_buys(self):
        # [2026-11-25] paper trading switched off with an open position: exits only
        r = self.runner()
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        c = self.runner(close_only=True)
        self.assertIn("010060", c.positions)
        c.on_symbol("010061", "y", self.at("09:40"), frame(40), 10300.0)        # would be a buy
        self.assertNotIn("010061", c.positions)
        c.on_symbol("010060", "x", self.at("09:41"), frame(41), 10600.0)        # target reached
        self.assertEqual((self.records()[-1]["event"], self.records()[-1]["reason"]), ("EXIT", "tp"))
        self.assertTrue(c.write_status(self.at("09:41"), force=True))
        st = json.loads((self.out / f"edge_only_status_{DAY}.json").read_text(encoding="utf-8"))
        self.assertTrue(st["close_only"])

    def test_restart_restores_the_nxt_flag(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0, nxt_tradeable=True)
        self.assertTrue(self.runner(nxt=True).positions["010060"]["nxt"])
        self.assertTrue(self.records()[-1]["nxt"])


class LogOnlyTest(Base):
    """[2026-11-20] EDGE_ONLY_LOG_ONLY: no new paper buys, one CANDIDATE row per trigger bar, open positions still exit"""

    def cands(self):
        f = self.out / f"edge_only_candidates_{DAY}.jsonl"
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []

    def test_candidates_instead_of_buys(self):
        r = self.runner(prob=0.6, log_only=True)
        r.on_symbol("010060", "OCI홀딩스", self.at("09:30"), frame(30), 10300.0)
        r.on_symbol("010060", "OCI홀딩스", self.at("09:30", 20), frame(30), 10300.0)      # same bar: one row
        self.assertEqual(self.records(), [])                                              # no ENTRY
        self.assertEqual(r.entries["watch"], 0)
        c = self.cands()
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0]["reason"], c[0]["would_buy"], c[0]["prob"], c[0]["threshold"]), ("would_buy", True, 0.6, 0.45))
        self.assertEqual(c[0]["id"], f"{DAY}:010060:0930:test")
        self.assertEqual(r.sym_state["010060"]["stage"], "candidate")
        # below the cut is recorded too, and a fund / a stopped-out code is recorded with its block
        r2 = self.runner(prob=0.2, log_only=True)
        r2.on_symbol("000001", "x", self.at("09:31"), frame(31), 10300.0)
        r2.on_symbol("233740", "KODEX 코스닥150레버리지", self.at("09:31"), frame(31), 10300.0)
        reasons = {x["code"]: (x["reason"], x["block"]) for x in self.cands()}
        self.assertEqual(reasons["000001"], ("below_threshold", None))
        self.assertEqual(reasons["233740"], ("fund", "fund"))
        # restart: the same bar is not written again
        r3 = self.runner(prob=0.6, log_only=True)
        r3.on_symbol("010060", "OCI홀딩스", self.at("09:30", 40), frame(30), 10300.0)
        self.assertEqual(len([x for x in self.cands() if x["code"] == "010060"]), 1)

    def test_open_positions_still_exit(self):
        r = self.runner(prob=0.6)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)                  # bought before the switch
        r2 = self.runner(prob=0.6, log_only=True)                                         # restart in log-only mode
        self.assertIn("010060", r2.positions)
        r2.on_symbol("010060", "x", self.at("09:31"), frame(31), 10600.0)
        self.assertNotIn("010060", r2.positions)
        self.assertEqual(self.records()[-1]["reason"], "tp")


class FundNameTest(unittest.TestCase):
    def test_names(self):
        for n in ("KODEX 코스닥150레버리지", "TIGER 200", "RISE 2차전지TOP10", "신한 레버리지 WTI원유 선물 ETN", "MIDAS 코스닥액티브",
                  "KoAct 바이오헬스케어액티브"):
            self.assertTrue(r011.is_fund_product(n), n)
        for n in ("삼성전자", "HK이노엔", "파워로직스", "BNK금융지주", "IBK기업은행", "DS단석", "SK하이닉스", "", None):
            self.assertFalse(r011.is_fund_product(n), n)


if __name__ == "__main__":
    unittest.main()
