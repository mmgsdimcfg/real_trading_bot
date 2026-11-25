# -*- coding: utf-8 -*-
"""[2026-11-23] Monitoring hours vs. trading hours (user request):
- r001 ENABLE_NXT_TRADING: with ENABLE_NXT_SESSION on and this off the bot watches the NXT hours but sends no order there -
  no new entry, no order spec (sells included), NXT-tradeable symbols are closed at 15:20 in the regular session, no 19:59 close
- NXT end-of-day close: a limit NXT_FORCE_EXIT_BID_TICKS steps below the best bid (r002.marketable_sell_limit), 0 = at the price
- r022 paper trading in the NXT sessions (EdgeOnlyRunner nxt=True): own entry windows / force exits, "session" on the records,
  the regular session unchanged
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("live_trading", "data_simulation"):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r001_define_config as r001  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
import r011_edge_ai as r011  # noqa: E402
import r022_edge_only as r022  # noqa: E402
from test_r022_edge_only import Base, FakeModel, frame  # noqa: E402

D = (2026, 11, 23)


def at(h, m, s=0):
    return datetime(*D, h, m, s)


class DefaultsTest(unittest.TestCase):
    def test_defaults_keep_the_full_day(self):
        self.assertTrue(r001.ENABLE_NXT_SESSION)
        self.assertTrue(r001.ENABLE_NXT_TRADING)
        self.assertTrue(r001.EDGE_ONLY_NXT)
        self.assertEqual(r001.AFTERNOON_NXT_FORCE_EXIT.strftime("%H:%M:%S"), "19:59:50")
        self.assertEqual(r001.NXT_FORCE_EXIT_BID_TICKS, 3)


class MarketableLimitTest(unittest.TestCase):
    def test_steps_below_the_bid_follow_the_tick_table(self):
        self.assertEqual(r002.marketable_sell_limit(10000, 10050, 3), 9970)      # 10-won ticks
        self.assertEqual(r002.marketable_sell_limit(5010, 5010, 3), 4990)        # 5010 -> 5000 -> 4995 -> 4990 (band change)
        self.assertEqual(r002.marketable_sell_limit(50000, 50000, 2), 49900)     # 50-won ticks below 50,000
        self.assertEqual(r002.marketable_sell_limit(200000, 0, 1), 199900)

    def test_fallback_and_zero_ticks(self):
        self.assertEqual(r002.marketable_sell_limit(None, 10050, 3), 10020)
        self.assertEqual(r002.marketable_sell_limit(0, 10050, 0), 10050)
        self.assertEqual(r002.marketable_sell_limit(10000, 1, 0), 10000)
        self.assertGreaterEqual(r002.marketable_sell_limit(2, 2, 5), 1)          # never to zero


class TradingHoursTest(unittest.TestCase):
    def setUp(self):
        import r003_trade_live_execute as r003
        self.r003 = r003

    def _flags(self, session, trading):
        return mock.patch.multiple(self.r003, ENABLE_NXT_SESSION=session, ENABLE_NXT_TRADING=trading)

    def test_watch_only_nxt_sends_no_order_in_the_nxt_sessions(self):
        r003 = self.r003
        with self._flags(True, False):
            self.assertFalse(r003.nxt_trading_enabled())
            for t in (at(8, 20), at(16, 0)):
                self.assertTrue(r003.is_nxt_session(t))                       # still watched
                self.assertTrue(r003.can_trade_code_now(t, True))             # the loop still looks at the symbol
                self.assertFalse(r003.is_new_entry_allowed(t, True))
                self.assertIsNone(r003.get_order_spec(t, True))
            self.assertEqual(r003.get_order_spec(at(10, 0), True)["exchange"], "KRX")
            self.assertTrue(r003.is_new_entry_allowed(at(10, 0), True))

    def test_full_trading_is_unchanged(self):
        r003 = self.r003
        with self._flags(True, True):
            self.assertTrue(r003.nxt_trading_enabled())
            self.assertTrue(r003.is_new_entry_allowed(at(16, 0), True))
            self.assertFalse(r003.is_new_entry_allowed(at(16, 0), False))
            self.assertEqual(r003.get_order_spec(at(16, 0), True)["exchange"], "NXT")
        with self._flags(False, True):                                        # monitoring off wins
            self.assertFalse(r003.nxt_trading_enabled())
            self.assertIsNone(r003.get_order_spec(at(16, 0), True))

    def _liquidate(self, session, trading, when, ticks=3, bid=10080.0):
        r003 = self.r003
        api = mock.Mock()
        api.get_open_positions.return_value = {"010060": {"quantity": 7, "buy_price": 10000.0, "current_price": 10100.0}}
        api.has_pending_order.return_value = False
        api.pending_orders = {}
        api.trade_lock_until = {}
        logs: list[str] = []
        state = {"date": None, "done_1520": False, "done_1959": False}
        with self._flags(session, trading), mock.patch.object(r003, "AUTO_SELL_MODE", "all"), mock.patch.object(r003, "log", logs.append), \
                mock.patch.object(r003, "_is_today_buy_position", return_value=True), \
                mock.patch.object(r003, "fetch_live_price", return_value=10100.0), \
                mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(bid, 10110.0)), \
                mock.patch.object(r003, "NXT_FORCE_EXIT_BID_TICKS", ticks), \
                mock.patch.object(r003, "datetime", mock.Mock(wraps=datetime, now=mock.Mock(return_value=when))):
            r003.run_scheduled_liquidations(when, api, {"010060": True}, {"010060": "OCI홀딩스"}, state, "20261123", {"010060"})
            pending = r003._nxt_force_exit_pending(at(19, 59, 45), state)
        return api, logs, state, pending

    def test_regular_only_trading_closes_nxt_symbols_at_1520(self):
        api, logs, state, pending = self._liquidate(True, False, at(15, 20, 1))
        self.assertEqual(api.place_sell_order.call_count, 1)
        self.assertNotIn("limit_price", api.place_sell_order.call_args.kwargs)   # regular session: market order path
        self.assertFalse(any("REG CLS DEFR" in m for m in logs))
        self.assertFalse(pending)                                                # nothing waits for 19:59
        api2, _logs, _state, _p = self._liquidate(True, False, at(19, 59, 51))
        self.assertEqual(api2.place_sell_order.call_count, 1)                    # only the 15:20 pass ran (late start), no NXT close
        self.assertNotIn("limit_price", api2.place_sell_order.call_args.kwargs)

    def test_full_trading_defers_to_the_nxt_close_below_the_bid(self):
        api, logs, _state, pending = self._liquidate(True, True, at(15, 20, 1))
        api.place_sell_order.assert_not_called()
        self.assertTrue(any("REG CLS DEFR" in m for m in logs))
        self.assertTrue(pending)
        api, logs, _state, _p = self._liquidate(True, True, at(19, 59, 51))
        self.assertEqual(api.place_sell_order.call_count, 1)
        self.assertEqual(api.place_sell_order.call_args.kwargs["limit_price"], 10050.0)   # bid 10,080 - 3 ticks of 10
        self.assertTrue(any("ONE_SHOT_LIMIT_BELOW_BID" in m for m in logs))

    def test_zero_ticks_is_the_old_limit_at_the_price_and_no_bid_falls_back(self):
        api, logs, _s, _p = self._liquidate(True, True, at(19, 59, 51), ticks=0)
        self.assertEqual(api.place_sell_order.call_args.kwargs["limit_price"], 10100.0)
        self.assertTrue(any("ONE_SHOT_LIMIT_AT_CURRENT" in m for m in logs))
        api, _l, _s, _p = self._liquidate(True, True, at(19, 59, 51), bid=None)
        self.assertEqual(api.place_sell_order.call_args.kwargs["limit_price"], 10070.0)   # price 10,100 - 3 ticks


class G003ParityTest(unittest.TestCase):
    def test_sim_follows_the_same_switch(self):
        import g003_trade_simulate_by_date as g003
        ts = pd.Timestamp(at(16, 0))
        with mock.patch.multiple(g003, ENABLE_NXT_SESSION=True, ENABLE_NXT_TRADING=False):
            self.assertFalse(g003.nxt_trading_enabled())
            self.assertFalse(g003.is_new_entry_allowed(ts, True))
            self.assertTrue(g003.is_nxt_session(ts))
        with mock.patch.multiple(g003, ENABLE_NXT_SESSION=True, ENABLE_NXT_TRADING=True):
            self.assertTrue(g003.is_new_entry_allowed(ts, True))


DAY = "20261112"


def nxt_frame(n, start, price=10000.0):
    return frame(n, start=start, price=price)


class PaperNxtTest(Base):
    def test_sessions_by_time(self):
        self.assertEqual([r022.nxt_session_of(datetime(2026, 1, 1, h, m).time()) for h, m in ((7, 59), (8, 0), (8, 50), (8, 51), (9, 30), (15, 30), (15, 31), (20, 0), (20, 1))],
                         [None, "nxt_pre", "nxt_pre", None, None, None, "nxt_after", "nxt_after", None])
        # [2026-11-25] closes: regular-only holdings 15:20 (auction), NXT holdings of an NXT runner 19:59:50
        self.assertEqual(r022.close_time_of({"nxt": False}, True).strftime("%H:%M:%S"), "15:20:00")
        self.assertEqual(r022.close_time_of({"nxt": True}, True).strftime("%H:%M:%S"), "19:59:50")
        self.assertEqual(r022.close_time_of({"nxt": True}, False).strftime("%H:%M:%S"), "15:20:00")

    def test_off_by_default_nothing_happens_in_the_nxt_hours(self):
        r = self.runner()
        r.on_symbol("010060", "x", self.at("16:30"), nxt_frame(60, "15:31"), 10300.0)
        self.assertEqual(self.records(), [])

    def test_after_session_buy_continues_the_regular_bars_and_is_tagged(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("09:10"), frame(10), 10100.0)             # regular store from 09:01 (also buys: regular)
        self.assertEqual([x.get("session") for x in self.records()], ["regular"])
        r.session_close(self.at("15:33"))                                             # regular-only symbol: 15:20 close
        r.on_symbol("010060", "x", self.at("16:00"), nxt_frame(30, "15:31"), 10300.0)
        rec = self.records()
        self.assertEqual([(x["event"], x.get("session")) for x in rec], [("ENTRY", "regular"), ("EXIT", "regular"), ("ENTRY", "nxt_after")])
        self.assertEqual(r.positions["010060"]["session"], "nxt_after")
        self.assertIn("[nxt_after]", self.logs[-1])

    def test_after_session_position_is_kept_until_the_nxt_close(self):
        # [2026-11-25] no 19:50 exit any more: an NXT holding is closed at 19:59:50
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("09:10"), frame(10), 10100.0)
        r.session_close(self.at("15:33"))
        r.on_symbol("010060", "x", self.at("19:20"), nxt_frame(229, "15:31"), 10300.0)
        self.assertIn("010060", r.positions)
        r.on_symbol("010060", "x", self.at("19:50"), nxt_frame(259, "15:31"), 10300.0)
        self.assertIn("010060", r.positions)
        r.on_symbol("010060", "x", self.at("19:59", 50), nxt_frame(268, "15:31"), 10300.0)
        self.assertNotIn("010060", r.positions)
        last = self.records()[-1]
        self.assertEqual((last["event"], last["reason"], last["session"], last["nxt"]), ("EXIT", "close_nxt", "nxt_after", True))

    def test_after_session_entry_window_and_history(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("15:33"), nxt_frame(2, "15:31"), 10300.0)   # before 15:35 / too few bars
        r.on_symbol("010061", "x", self.at("19:40"), nxt_frame(249, "15:31"), 10300.0)  # after 19:30
        # started after the close: no regular bars, the history window (90 bars back, not before 09:01) is not covered yet
        r.on_symbol("010062", "x", self.at("16:00"), nxt_frame(30, "15:31"), 10300.0)
        self.assertEqual(self.records(), [])
        self.assertEqual(r.sym_state["010062"]["stage"], "no_history")

    def test_pre_session_uses_its_own_bars(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("08:20"), nxt_frame(20, "08:01"), 10300.0)
        rec = self.records()
        self.assertEqual([(x["event"], x["session"]) for x in rec], [("ENTRY", "nxt_pre")])
        r.on_symbol("010060", "x", self.at("08:45"), nxt_frame(45, "08:01"), 10300.0)
        self.assertIn("010060", r.positions)                                           # [2026-11-25] kept past 08:45
        self.assertTrue(r.positions["010060"]["nxt"])
        r.on_symbol("010060", "x", self.at("09:10"), frame(10), 10300.0)               # into the regular session
        self.assertIn("010060", r.positions)
        self.assertEqual(r.session_close(self.at("15:33")), 0)                         # not a regular-only holding
        r2 = self.runner(nxt=True)
        r2.on_symbol("010061", "x", self.at("08:30"), nxt_frame(10, "08:21"), 10300.0)  # first bar after 08:10: no prediction
        self.assertEqual(r2.sym_state["010061"]["stage"], "no_history")
        r2.on_symbol("010062", "x", self.at("08:40"), nxt_frame(40, "08:01"), 10300.0)  # after 08:35
        self.assertEqual(r2.sym_state["010062"]["stage"], "outside_window")

    def test_pre_session_bars_do_not_reach_the_regular_store(self):
        r = self.runner(nxt=True, prob=0.1)
        r.on_symbol("010060", "x", self.at("08:20"), nxt_frame(20, "08:01"), 10300.0)
        both = pd.concat([nxt_frame(50, "08:01"), frame(10)])
        r.on_symbol("010060", "x", self.at("09:10"), both, 10100.0)
        self.assertEqual(r.bars["010060"].index[0].strftime("%H:%M"), "09:01")
        self.assertEqual(len(r.bars["010060"]), 10)

    def test_restart_restores_the_session_and_old_records_are_regular(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("09:10"), frame(10), 10100.0)
        r.session_close(self.at("15:33"))
        r.on_symbol("010060", "x", self.at("16:00"), nxt_frame(30, "15:31"), 10300.0)
        r2 = self.runner(nxt=True)
        self.assertEqual(r2.positions["010060"]["session"], "nxt_after")
        f = self.out / f"edge_only_{DAY}.jsonl"
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
        rows[-1].pop("session")
        f.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        self.assertEqual(self.runner(nxt=True).positions["010060"]["session"], "regular")

    def test_orphan_of_an_nxt_session_is_not_priced_from_the_regular_tick_file(self):
        r = self.runner(nxt=True)
        r.on_symbol("010060", "x", self.at("09:10"), frame(10), 10100.0)
        r.session_close(self.at("15:33"))
        r.on_symbol("010060", "x", self.at("16:00"), nxt_frame(30, "15:31"), 10300.0)
        feed = mock.Mock()
        feed.poll.return_value = (None, 5000.0)                                       # a stale regular price far below the stop
        # [2026-11-25] no orphan pricing outside the regular session (the NXT close is session_close's job)
        self.assertEqual(r.poll_orphans(feed, self.at("16:02"), datetime(2030, 1, 1)), 0)
        feed.poll.assert_not_called()
        self.assertIn("010060", r.positions)

    def test_log_only_candidate_carries_the_session_and_status_lists_the_windows(self):
        r = self.runner(nxt=True, log_only=True)
        r.on_symbol("010060", "x", self.at("08:20"), nxt_frame(20, "08:01"), 10300.0)
        rows = [json.loads(x) for x in (self.out / f"edge_only_candidates_{DAY}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([x["session"] for x in rows], ["nxt_pre"])
        self.assertEqual(self.records(), [])
        self.assertTrue(r.write_status(self.at("08:20"), force=True))
        st = json.loads((self.out / f"edge_only_status_{DAY}.json").read_text(encoding="utf-8"))
        self.assertTrue(st["nxt"])
        self.assertEqual(st["nxt_sessions"]["nxt_after"], {"entry_window": ["15:35", "19:30"], "force_exit": "19:59:50"})
        self.assertEqual((st["close_regular"], st["close_nxt"]), ("15:20", "19:59:50"))
        off = self.runner()
        off.write_status(self.at("09:20"), force=True)
        st = json.loads((self.out / f"edge_only_status_{DAY}.json").read_text(encoding="utf-8"))
        self.assertEqual((st["nxt"], st["nxt_sessions"]), (False, {}))


if __name__ == "__main__":
    unittest.main()
