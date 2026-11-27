# -*- coding: utf-8 -*-
"""Timeline trace of the side strategies (r030, 2026-11-27): one row per judged bar (real orders and paper), a record only -
and the Buy block of the daily trade log for a side buy the fill ledger closed before the account row arrived."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

os.environ.setdefault("AUTO_TRADING_LIVE_OVERRIDES", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "tests"))

import r023_bot_holdings as r023  # noqa: E402
import r029_vwap_trade as r029  # noqa: E402
import r030_strategy_trace as r030  # noqa: E402
from test_edge_live import COSTS, DAY, NOW, _api, r003, row  # noqa: E402
from test_r029_vwap_trade import NOW as BAR_NOW, P, RISING, bars, tick  # noqa: E402


def rows(path: Path) -> list[dict]:
    r030.flush()
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


class WriterTest(unittest.TestCase):
    def test_rows_are_queued_and_written_and_bad_values_never_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = datetime(2026, 11, 27, 9, 31, 2)
            self.assertTrue(r030.trace(tmp, "vwap", "20261127", now, "5930", "live", "below_vwap", bar="09:30", price=100.0,
                                       vwap=float("nan"), gap=-0.3123456789, tick="", obj=object()))
            got = rows(r030.path_for(Path(tmp), "vwap", "20261127"))
            self.assertEqual(len(got), 1)
            r = got[0]
            self.assertEqual((r["t"], r["code"], r["mode"], r["stage"], r["bar"], r["gap"]),
                             ("2026-11-27T09:31:02", "005930", "live", "below_vwap", "09:30", -0.312346))
            self.assertNotIn("vwap", r)            # NaN is left out, an empty text too
            self.assertNotIn("tick", r)
            self.assertIn("obj", r)                # anything else as text
            # a directory that cannot be made: the writer counts it, nothing reaches the caller
            bad = Path(tmp) / "file.txt"
            bad.write_text("x", encoding="utf-8")
            before = r030.write_errors
            self.assertTrue(r030.trace(bad / "sub", "vwap", "20261127", now, "005930", "live", "x"))
            r030.flush()
            self.assertGreater(r030.write_errors, before)
            # a missing folder is never created (the bot's live_runtime exists; a test folder that is gone stays gone)
            gone = Path(tmp) / "gone"
            r030.trace(gone, "vwap", "20261127", now, "005930", "live", "x")
            r030.flush()
            self.assertFalse(gone.exists())

    def test_after_a_failed_write_the_next_rows_start_on_a_new_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = r030.path_for(Path(tmp), "vwap", "20261127")
            path.write_text('{"t": "2026-11-27T09:00:00", "code": "0059', encoding="utf-8")     # half a row left by a failed write
            r030._dirty.add(path)
            r030.trace(tmp, "vwap", "20261127", datetime(2026, 11, 27, 9, 1), "005930", "live", "below_vwap")
            r030.flush()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["stage"], "below_vwap")
            self.assertNotIn(path, r030._dirty)

    def test_a_full_queue_drops_and_counts(self):
        import queue
        # the running writer thread must not empty the test's queue: its drain does nothing while this runs
        with mock.patch.object(r030, "_q", queue.Queue(maxsize=1)), mock.patch.object(r030, "_ensure_thread"), \
                mock.patch.object(r030, "_drain"):
            d0 = r030.dropped
            self.assertTrue(r030.trace("x", "vwap", "20261127", NOW, "005930", "live", "a"))
            self.assertFalse(r030.trace("x", "vwap", "20261127", NOW, "005930", "live", "b"))
            self.assertEqual(r030.dropped, d0 + 1)

    def test_the_stage_of_a_refused_bar(self):
        f = r030.reject_stage
        self.assertEqual(f(8, "MAX_OPEN_POSITIONS open=6"), "cond_before")
        self.assertEqual(f(23, "INSUFFICIENT_BUYING_POWER_OR_BUDGET"), "cond_after")
        self.assertEqual(f(9, "VWAP_SIGNAL_BELOW_VWAP"), "below_vwap")
        self.assertEqual(f(9, "VWAP_SIGNAL_NO_TICK_CROSS | TICK60_NO_CROSS"), "no_tick_cross")
        self.assertEqual(f(9, "EDGE_SIGNAL_NO_TRIGGER_BELOW_1MIN_MID | close=1 mid=2"), "below_mid")
        self.assertEqual(f(9, "EDGE_SIGNAL_NO_TRIGGER | NO_CROSS"), "no_trigger")
        self.assertEqual(f(9, "EDGE_SIGNAL_LOW_PROB_p=0.330_LT_0.336 | CROSS"), "below_threshold")
        self.assertEqual(f(9, "EDGE_SIGNAL_MODEL_NOT_PROMOTED_p=0.4"), "not_promoted")
        self.assertEqual(f(9, "VWAP_SIGNAL_ERROR_ValueError: x"), "error")
        self.assertEqual(r029.trace_fields("chase", {"vwap": 100.0, "vwap_gap_pct": 2.5, "tick": "TICK60", "x": 1}),
                         {"stage": "chase", "vwap": 100.0, "gap": 2.5, "tick": "TICK60"})


class PaperTraceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        r030.flush()
        self.tmp.cleanup()

    def _runner(self, tick_result, **kw):
        r = r029.VwapPaperRunner(self.out, "20261127", lambda m: None, P, lambda code, now: tick_result, **kw)
        r.trace_enabled = True
        return r

    def test_a_runner_nobody_switched_on_writes_nothing(self):
        r = r029.VwapPaperRunner(self.out, "20261127", lambda m: None, P, lambda code, now: tick(ok=False))
        r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        self.assertEqual(self._rows(), [])

    def _rows(self):
        return rows(r030.path_for(self.out, "vwap", "20261127"))

    def test_one_row_per_new_bar_with_the_values_the_signal_used(self):
        r = self._runner(tick(ok=False))
        r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.2)          # the same bar again: no second row
        got = self._rows()
        self.assertEqual(len(got), 1)
        g = got[0]
        self.assertEqual((g["code"], g["mode"], g["stage"], g["bar"]), ("005930", "paper", "no_tick_cross", "09:30"))
        self.assertGreater(g["gap"], 0)
        self.assertGreater(g["vwap"], 0)
        self.assertEqual(g["price"], 103.1)

    def test_a_buy_is_one_row_and_the_holding_bars_follow(self):
        r = self._runner(tick())
        r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        self.assertEqual([x["stage"] for x in self._rows()], ["bought"])
        more = bars(RISING + [103.2])                                                # 09:31 bar
        r.on_symbol("005930", "삼성전자", datetime(2026, 11, 27, 9, 31, 30), more, 103.2)
        self.assertEqual([x["stage"] for x in self._rows()], ["bought", "held"])
        # the limit stop of a new bar is a row too
        r.max_positions = 1
        r.limits = {g: (1, v[1]) for g, v in r.limits.items()}
        r.on_symbol("000660", "SK하이닉스", BAR_NOW, bars(RISING), 103.1)
        self.assertEqual(self._rows()[-1]["stage"], "limit")

    def test_a_bar_that_comes_with_the_sale_still_has_its_row(self):
        """[Codex round 2] the poll that sells the position also brings a new bar: 'sold' was set before the bar was known"""
        r = self._runner(tick())
        r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        tp = r.positions["005930"]["tp"]
        r.on_symbol("005930", "삼성전자", datetime(2026, 11, 27, 9, 31, 30), bars(RISING + [103.2]), tp + 1)   # new bar + target hit
        self.assertNotIn("005930", r.positions)
        self.assertEqual([(x["stage"], x["bar"]) for x in self._rows()], [("bought", "09:30"), ("sold", "09:31")])

    def test_reading_the_trace_values_never_changes_the_paper_decision(self):
        """[Codex round 2] trace_fields raising must not stop the entry"""
        r = self._runner(tick())
        with mock.patch.object(r029, "trace_fields", side_effect=RuntimeError("x")):
            r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        self.assertIn("005930", r.positions)
        self.assertEqual([x["stage"] for x in self._rows()], ["bought"])

    def test_the_threshold_is_never_asked_again_for_the_trace(self):
        """[Codex round 2] _threshold() can reload the model: the trace only gets what the decision already had"""
        r = self._runner(tick(ok=False))
        with mock.patch.object(r, "_threshold", side_effect=AssertionError("asked again")):
            r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        self.assertEqual([x["stage"] for x in self._rows()], ["no_tick_cross"])

    def test_a_trace_error_never_reaches_the_runner(self):
        r = self._runner(tick(ok=False))
        with mock.patch.object(r030, "trace", side_effect=RuntimeError("disk")):
            r.on_symbol("005930", "삼성전자", BAR_NOW, bars(RISING), 103.1)
        self.assertEqual(r.sym_state["005930"]["stage"], "no_tick_cross")


class LiveTraceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.patches = [mock.patch.object(r003, "LIVE_RUNTIME_DIR", self.dir), mock.patch.object(r003, "_log_trade_block"),
                        mock.patch.dict(r003._LOG_CTX, {"trade_log": str(self.dir / "t.log")})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        r030.flush()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _rows(self, strategy="vwap"):
        return rows(r030.path_for(self.dir, strategy, NOW.strftime("%Y%m%d")))

    def test_a_refused_bar_and_an_ordered_bar(self):
        ctx = mock.Mock(side_eval={"stage": "below_vwap", "vwap": 10_100.0, "gap": -0.99}, price=10_000.0, buy_reason="VWAP_SIGNAL_x", qty=5)
        e09 = mock.Mock(no=9, func_name="_v09_vwap_signal")
        r003._side_trace(r023.VWAP, NOW, "005930", ctx, "09:59", failed=e09, msgs=["  [REJECT      ] 005930_삼성전자 | VWAP_SIGNAL_BELOW_VWAP"])
        # a shared condition before the signal: no VWAP values are claimed (the signal was not judged on this bar)
        r003._side_trace(r023.VWAP, NOW, "005930", mock.Mock(side_eval=None, price=10_000.0), "10:00",
                         failed=mock.Mock(no=8, func_name="_008_max_open_positions"), msgs=["  [BUY SKIP    ] 005930 | MAX_OPEN_POSITIONS open=6 max=6"])
        r003._side_trace(r023.VWAP, NOW, "005930", ctx, "10:01", stage="ordered")
        got = self._rows()
        self.assertEqual([(g["stage"], g["bar"]) for g in got], [("below_vwap", "09:59"), ("cond_before", "10:00"), ("ordered", "10:01")])
        self.assertEqual((got[0]["no"], got[0]["cond"], got[0]["detail"], got[0]["vwap"], got[0]["gap"], got[0]["mode"]),
                         (9, "_v09_vwap_signal", "VWAP_SIGNAL_BELOW_VWAP", 10_100.0, -0.99, "live"))
        self.assertEqual((got[1]["no"], got[1]["detail"]), (8, "MAX_OPEN_POSITIONS open=6 max=6"))
        self.assertNotIn("vwap", got[1])
        self.assertEqual((got[2]["reason"], got[2]["qty"]), ("VWAP_SIGNAL_x", 5))
        self.assertEqual(self._rows("edge"), [])

    def test_the_real_order_path_writes_one_row_per_judged_bar(self):
        """the rows come from _edge_live_try_buy itself: refused by a condition, approved but the order failed, ordered"""
        import dataclasses
        import pandas as pd
        api = _api(r023.new_fill_state())
        state = r003._edge_live_setup(api)
        state["enabled"] = True
        frames = {"bars": pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-11-25 09:58", "2026-11-25 09:59"]))}
        outcome = {}

        def check(ctx, conditions):
            ctx.qty, ctx.session, ctx.buy_reason, ctx.side_eval = 5, "regular", "VWAP_SIGNAL_x", outcome.get("eval")
            return mock.Mock(approved=outcome["approved"], failed=outcome.get("failed"))

        def run():
            return r003._edge_live_try_buy(api, state, "005930", "삼성전자", "005930", NOW, False, 10_000.0, "live",
                                           frames["bars"].index[-1], frames["bars"], {}, set(), set(), set(), set(), None, mock.Mock(), {}, {},
                                           strategy=r023.VWAP)

        def next_bar(label):
            frames["bars"] = pd.DataFrame({"close": [2.0, 3.0]}, index=pd.to_datetime(["2026-11-25 09:00", f"2026-11-25 {label}"]))

        with mock.patch.object(r003, "_get_or_refresh_1min_frame", side_effect=lambda *a: frames["bars"]), \
                mock.patch.object(r003, "check_buy_conditions", side_effect=check), \
                mock.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj), \
                mock.patch.object(r003, "BuyContext", side_effect=lambda **kw: mock.Mock(**kw)), \
                mock.patch.object(api, "edge_live_block_reason", return_value=""), \
                mock.patch.object(api, "place_buy_order") as order:
            outcome.update(approved=False, failed=mock.Mock(no=8, func_name="_008_max_open_positions"), eval=None)
            self.assertFalse(run())
            self.assertFalse(run())                                      # the same bar: not judged, no second row
            next_bar("10:00")
            outcome.update(approved=False, failed=mock.Mock(no=9, func_name="_v09_vwap_signal"), eval={"stage": "below_vwap", "gap": -0.5})
            self.assertFalse(run())
            next_bar("10:01")
            outcome.update(approved=True, failed=None, eval={"stage": "signal", "gap": 0.4})
            order.return_value = False
            self.assertFalse(run())
            next_bar("10:02")
            order.return_value = True
            self.assertTrue(run())
        got = self._rows()
        self.assertEqual([(g["stage"], g["bar"]) for g in got],
                         [("cond_before", "09:59"), ("below_vwap", "10:00"), ("order_refused", "10:01"), ("ordered", "10:02")])
        self.assertEqual((got[1]["gap"], got[1]["no"], got[3]["gap"], got[3]["reason"]), (-0.5, 9, 0.4, "VWAP_SIGNAL_x"))
        # the trace failing changes nothing of the outcome
        next_bar("10:03")
        with mock.patch.object(r003, "_get_or_refresh_1min_frame", side_effect=lambda *a: frames["bars"]), \
                mock.patch.object(r003, "check_buy_conditions", side_effect=check), \
                mock.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj), \
                mock.patch.object(r003, "BuyContext", side_effect=lambda **kw: mock.Mock(**kw)), \
                mock.patch.object(api, "edge_live_block_reason", return_value=""), \
                mock.patch.object(api, "place_buy_order", return_value=True), \
                mock.patch.object(r030, "trace", side_effect=RuntimeError("disk")):
            self.assertTrue(run())

    def test_the_vwap_signal_keeps_its_answer_when_the_trace_values_cannot_be_read(self):
        """[Codex round 2] r005 _v09_vwap_signal: trace_fields raising must not turn the signal into an ERROR reject"""
        import r005_buy_conditions as r005
        import r011_edge_ai as r011
        from types import SimpleNamespace
        from test_r029_vwap_trade import NOW as T, RISING, bars, tick
        logs = []

        def ctx():
            services = mock.Mock()
            services.classify_buy_session.return_value = "regular"
            services.get_frame_1min.return_value = bars(RISING)
            services.fetch_day_ref_prices.return_value = (100.0, 102.0)
            return SimpleNamespace(services=services, current_dt=T, code="005930", nxt_tradeable=False, price=103.1, symbol_label="005930",
                                   day_open=None, day_vwap=None, prev_close=98.0, log=logs.append, side_eval="unset", trigger_flag="", steps_text="",
                                   trigger_reason="", buy_reason="")
        with mock.patch.object(r011, "session_raw_1min", side_effect=lambda frame, day: frame), \
                mock.patch.object(r005, "_vwap_tick_eval", return_value=tick()):
            a = ctx()
            self.assertTrue(r005._v09_vwap_signal(a))
            self.assertEqual((a.side_eval["stage"], a.side_eval["vwap"]), ("signal", 102.0))
            b = ctx()
            with mock.patch.object(r029, "trace_fields", side_effect=RuntimeError("x")):
                self.assertTrue(r005._v09_vwap_signal(b))
            self.assertIsNone(b.side_eval)
            self.assertTrue(b.buy_reason.startswith("VWAP_SIGNAL_gap="))
            # [2026-11-28] VWAP_TRADE_MIN_OPEN_GAP_PCT: open 100 against a previous close of 99.9 is +0.10 % - no buy, stage low_gap
            c = ctx()
            c.prev_close = 99.9
            import r001_define_config as r001
            with mock.patch.object(r001, "VWAP_TRADE_MIN_OPEN_GAP_PCT", 1.0, create=True):
                self.assertFalse(r005._v09_vwap_signal(c))
            self.assertEqual(c.side_eval["stage"], "low_gap")
            self.assertTrue(any("VWAP_SIGNAL_LOW_GAP" in str(m) for m in logs))
            logs.clear()
        self.assertEqual(logs, [])

    def test_nothing_is_written_without_the_session_logging_and_errors_stay_inside(self):
        with mock.patch.dict(r003._LOG_CTX, {"trade_log": ""}):
            r003._side_trace(r023.VWAP, NOW, "005930", mock.Mock(side_eval=None, price=1.0), "09:59", stage="ordered")
        self.assertEqual(self._rows(), [])
        with mock.patch.object(r030, "trace", side_effect=RuntimeError("x")):
            r003._side_trace(r023.VWAP, NOW, "005930", mock.Mock(side_eval=None, price=1.0), "09:59", stage="ordered")
        r003._side_trace(r023.VWAP, NOW, "005930", object(), "09:59", failed=object(), msgs=None)   # odd inputs: no raise


class SideBuyBlockTest(unittest.TestCase):
    """the 2026-11-27 VWAP buy of 237690: closed by the fill ledger while the account had no row yet -> no Buy block"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(r003, "LIVE_RUNTIME_DIR", Path(self.tmp.name)), mock.patch.object(r003, "log")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _closed_buy(self, positions, filled=5):
        st = r023.new_fill_state()
        st["codes"]["237690"] = {"bot_qty": 0, "manual_qty": 0}
        r023.register_order(st, DAY, "14965900", "237690", "buy", 5, NOW.isoformat(timespec="seconds"), strategy=r023.VWAP)
        if filled:
            r023.apply_fill_rows(st, [row("14965900", "237690", "buy", 5, filled, 87_900)], NOW, COSTS)
        else:
            st["registry"][r023.fill_key(DAY, "14965900")]["terminal"] = True
        api = _api(st, positions)
        api.pending_orders["237690"] = {"side": "buy", "quantity": 5, "submitted_at": NOW, "order_no": "14965900", "strategy": r023.VWAP,
                                        "code_name": "에스티팜", "buy_detail": "VWAP_LIVE VWAP_SIGNAL_gap=+1.34%_MA5_UP_TICK60_MA5_MA15"}
        return api

    def test_the_block_is_written_when_the_account_row_is_not_there_yet(self):
        api = self._closed_buy({})
        with mock.patch.object(r003, "_log_trade_block") as block, mock.patch.object(api, "_confirm_pending_buy") as confirm:
            api.settle_edge_pending_by_ledger(NOW)
        self.assertEqual(confirm.call_count, 0)
        self.assertNotIn("237690", api.pending_orders)
        lines = block.call_args[0][0]
        self.assertEqual(lines[1], ">>> 237690_에스티팜 Buy - VWAP_LIVE VWAP_SIGNAL_gap=+1.34%_MA5_UP_TICK60_MA5_MA15")
        self.assertIn("매수가격=87,900 | 매수수량=5", lines[2])
        self.assertTrue(any("[VWAP LIVE   ]" in str(c.args[0]) and "종결" in str(c.args[0]) for c in r003.log.call_args_list))

    def test_a_partial_fill_then_cancel_is_written_with_what_that_order_filled(self):
        """[Codex round 2] quantity = the order's cumulative fills, price = the average of exactly that order (its lot)"""
        api = self._closed_buy({}, filled=3)
        api._fills["registry"][r023.fill_key(DAY, "14965900")]["terminal"] = True      # the rest was cancelled
        # another order of the same strategy in the book at another price must not move this order's price
        api._fills[r023.VWAP]["237690"]["lots"]["other"] = {"qty": 10, "avg": 80_000.0, "avg_at": 10}
        api._fills["applied"]["other"] = 10
        with mock.patch.object(r003, "_log_trade_block") as block:
            api.settle_edge_pending_by_ledger(NOW)
        self.assertIn("매수가격=87,900 | 매수수량=3", block.call_args[0][0][2])
        # the lot's price not known for exactly the applied quantity -> N/A, never another average
        self.assertEqual(r023.order_lot_price(api._fills, r023.VWAP, "237690", "nope"), 0.0)
        st = api._fills
        key = r023.fill_key(DAY, "14965900")
        st[r023.VWAP]["237690"]["lots"][key]["avg_at"] = 1
        self.assertEqual(r023.order_lot_price(st, r023.VWAP, "237690", key), 0.0)

    def test_with_the_account_row_the_normal_confirm_writes_it_once(self):
        api = self._closed_buy({"237690": {"quantity": 5, "buy_price": 87_900.0}})
        with mock.patch.object(r003, "_log_trade_block") as block, mock.patch.object(r003, "_entry_note"), \
                mock.patch.object(api, "_record_position_meta"), mock.patch.object(api, "persist_live_state"):
            api.settle_edge_pending_by_ledger(NOW)
        self.assertEqual(block.call_count, 1)
        self.assertIn("Buy - VWAP_LIVE", block.call_args[0][0][1])

    def test_no_fill_no_block_and_a_write_error_still_closes_the_order(self):
        api = self._closed_buy({}, filled=0)
        with mock.patch.object(r003, "_log_trade_block") as block, mock.patch.object(api, "_edge_entry_back"):
            api.settle_edge_pending_by_ledger(NOW)
        self.assertEqual(block.call_count, 0)
        api = self._closed_buy({})
        with mock.patch.object(r003, "_log_trade_block", side_effect=OSError("disk")):
            api.settle_edge_pending_by_ledger(NOW)
        self.assertNotIn("237690", api.pending_orders)


if __name__ == "__main__":
    unittest.main()
