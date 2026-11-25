# -*- coding: utf-8 -*-
"""[2026-11-25] task A (Codex review): A1 settle the bot's open orders on a normal stop, A2 hold the symbols of unresolved orders
after a restart until the broker shows them closed. No real broker: dsf calls are faked."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception:  # broker SDK missing on this machine
    r003 = None

NOW = datetime(2026, 11, 25, 10, 0, 0)
DAY = "20261125"


def _api(pending=None, holds=None, dry_run=False):
    api = object.__new__(r003.TradingAPI)
    api.env_dv = "real"
    api.dry_run = dry_run
    api.cano, api.acnt_prdt_cd = "00000000", "01"
    api.live_state = {"positions_meta": {}, "traded_today": set()}
    api.positions = {}
    api.pending_orders = pending or {}
    api.buy_inflight_codes = set()
    api.startup_position_codes = set()
    api.trade_lock_until = {}
    api.last_sell_fill_price = {}
    api._last_sync_at = None
    api._last_pending_poll_at = None
    api._last_live_state_save_at = None
    api._bot_order_window = {"buy": {}, "sell": {}}
    api.stopping = False
    api._holds_signature = None
    api._holds_next_check = None
    api._holds_warned_at = {}
    api.order_holds = holds or {}
    api.holds_all = False
    api._submit_intents = {}
    api._prev_hold_caps = {}
    api.bot_ledger = {}
    api.sync_positions_from_account = lambda force=False: None
    api.persist_live_state = lambda *a, **k: None
    return api


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class OrderHoldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.p = mock.patch.object(r003, "LIVE_RUNTIME_DIR", self.dir)
        self.p.start()
        self.p2 = mock.patch.object(r003, "_stop_requested", return_value=False)
        self.stop_flag = self.p2.start()

    def tearDown(self):
        self.p2.stop()
        self.p.stop()
        self.tmp.cleanup()

    def holds_file(self):
        return json.loads((self.dir / f"order_holds_{DAY}.json").read_text(encoding="utf-8"))

    def test_order_cash_is_blocked_while_stopping_or_for_a_held_symbol(self):
        api = _api(holds={"005930": {"side": "sell", "order_nos": ["1"]}})
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "9"}) as oc:
            self.assertEqual(api._order_cash(pdno="005930", ord_dv="buy", ord_qty="1")["rt_cd"], "1")   # held symbol
            self.assertFalse(r003._order_succeeded(api._order_cash(pdno="005930", ord_dv="sell", ord_qty="1")))
            self.assertEqual(api._order_cash(pdno="000660", ord_dv="buy", ord_qty="1")["rt_cd"], "0")   # other symbol
            api.stopping = True
            self.assertEqual(api._order_cash(pdno="000660", ord_dv="buy", ord_qty="1")["rt_cd"], "1")
            self.assertEqual(oc.call_count, 1)
        self.assertTrue(api.has_pending_order("005930"))
        self.assertFalse(api.has_pending_order("000660"))

    def test_holds_file_round_trip_other_day_and_broken_file(self):
        entries = r003._hold_entries_from_pending({
            "005930": {"side": "sell", "quantity": 3, "order_no": "11", "exchange": "KRX", "submitted_at": NOW},
            "000660": {"side": "buy", "quantity": 5, "legs": [{"order_no": "21", "done": False}, {"order_no": "22", "done": True}]},
        })
        self.assertEqual(entries["000660"]["order_nos"], ["21"])
        self.assertTrue(r003.save_order_holds(DAY, entries))
        self.assertEqual(sorted(r003.load_order_holds(DAY)), ["000660", "005930"])
        self.assertEqual(r003.load_order_holds("20261126"), {})                       # orders do not carry over
        (self.dir / f"order_holds_{DAY}.json").write_text("{broken", encoding="utf-8")
        logs = []
        with mock.patch.object(r003, "log", logs.append):
            self.assertIsNone(r003.load_order_holds(DAY))                                 # broken: hold everything
        self.assertTrue(any("unreadable" in m for m in logs))

    def test_holds_are_released_only_when_the_broker_shows_no_open_order(self):
        api = _api(holds={"005930": {"side": "sell", "order_nos": ["1"]}})
        with mock.patch.object(api, "_open_orders_at_broker", return_value=None):          # query failed: keep
            api.check_order_holds(NOW)
        self.assertIn("005930", api.order_holds)
        api._holds_next_check = None
        with mock.patch.object(api, "_open_orders_at_broker", return_value={("005930", "sell")}):   # still open: keep
            api.check_order_holds(NOW)
        self.assertIn("005930", api.order_holds)
        api._holds_next_check = None
        with mock.patch.object(api, "_open_orders_at_broker", return_value={("005930", "buy")}):    # other side only
            api.check_order_holds(NOW)
        # closed at the broker: still held until the next account sync has put a last fill into the r023 ledger
        self.assertTrue(api.order_holds["005930"]["release_after_sync"])
        self.assertTrue(api.has_pending_order("005930"))
        with mock.patch.object(r003.r023, "save_ledger"), mock.patch.object(r003.r023, "save_snapshot"):
            api._update_bot_ledger({"005930": {"quantity": 7}})
        self.assertNotIn("005930", api.order_holds)
        self.assertNotIn("005930", api._bot_order_window["sell"])                       # a later trade is not that order's
        self.assertEqual(self.holds_file()["orders"], {})

    def test_no_record_no_order(self):
        api = _api()
        with mock.patch.object(r003, "save_order_holds", return_value=False), \
                mock.patch.object(r003.dsf, "order_cash", create=True) as oc:
            self.assertEqual(api._order_cash(pdno="000660", ord_dv="buy", ord_qty="1")["rt_cd"], "1")
        oc.assert_not_called()

    def test_unknown_state_survives_a_restart_and_both_sides_must_close(self):
        api = _api()
        api.holds_all = True
        api.save_order_holds_if_changed(DAY, force=True)
        self.assertTrue(self.holds_file()["holds_all"])
        logs = []
        with mock.patch.object(r003, "log", logs.append):
            self.assertIsNone(r003.load_order_holds(DAY))                                 # the restart holds everything too
        with mock.patch.object(api, "_open_orders_at_broker", return_value={("005930", "buy"), ("005930", "sell")}):
            api.check_order_holds(NOW)
        self.assertEqual(api.order_holds["005930"]["side"], "both")
        api._holds_next_check = None
        with mock.patch.object(api, "_open_orders_at_broker", return_value={("005930", "sell")}):   # the buy closed only
            api.check_order_holds(NOW)
        self.assertFalse(api.order_holds["005930"].get("release_after_sync"))
        api._holds_next_check = None
        with mock.patch.object(api, "_open_orders_at_broker", return_value=set()):
            api.check_order_holds(NOW)
        self.assertTrue(api.order_holds["005930"]["release_after_sync"])

    def test_an_order_without_an_answer_becomes_a_hold_that_outlives_the_turn(self):
        api = _api()
        with mock.patch.object(r003.dsf, "order_cash", create=True, side_effect=TimeoutError("read timeout")):
            with self.assertRaises(TimeoutError):
                api._order_cash(pdno="000660", ord_dv="buy", ord_qty="5")
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value=pd.DataFrame()):
            api._order_cash(pdno="005930", ord_dv="sell", ord_qty="2")
        api.save_order_holds_if_changed(datetime.now().strftime("%Y%m%d"), turn_end=True)
        today = datetime.now().strftime("%Y%m%d")
        orders = json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]
        self.assertEqual((orders["000660"]["side"], orders["005930"]["side"]), ("buy", "sell"))
        self.assertTrue(api.has_pending_order("000660"))
        self.assertEqual(api._bot_order_window["buy"]["000660"], 2)

    def test_split_buy_intents_add_up(self):
        api = _api()
        seen = []

        def order_cash(**kw):
            today = datetime.now().strftime("%Y%m%d")
            seen.append(json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]["000660"]["quantity"])
            return {"rt_cd": "0", "odno": "9"}
        with mock.patch.object(r003.dsf, "order_cash", create=True, side_effect=order_cash):
            api._order_cash(pdno="000660", ord_dv="buy", ord_qty="3")
            api._order_cash(pdno="000660", ord_dv="buy", ord_qty="4")
        self.assertEqual(seen, [3, 7])

    def test_first_leg_accepted_second_leg_timed_out_keeps_both_quantities(self):
        api = _api()
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "31"}):
            api._order_cash(pdno="000660", ord_dv="buy", ord_qty="3")
        with mock.patch.object(r003.dsf, "order_cash", create=True, side_effect=TimeoutError("read timeout")):
            with self.assertRaises(TimeoutError):
                api._order_cash(pdno="000660", ord_dv="buy", ord_qty="4")
        api.pending_orders["000660"] = {"side": "buy", "quantity": 3, "legs": [{"order_no": "31", "done": False, "quantity": 3}]}
        api.save_order_holds_if_changed(datetime.now().strftime("%Y%m%d"), turn_end=True)
        today = datetime.now().strftime("%Y%m%d")
        e = json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]["000660"]
        self.assertEqual((e["side"], e["quantity"], e["order_nos"]), ("buy", 7, ["31"]))    # unknown 4 (hold) + tracked 3
        self.assertEqual(api._submit_intents, {})

    def test_a_reprice_while_the_old_order_is_pending_is_recorded(self):
        api = _api(pending={"000660": {"side": "sell", "quantity": 5, "order_no": "51"}})
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "52"}):
            api._order_cash(pdno="000660", ord_dv="sell", ord_qty="5")        # the resubmit, before pending gets 52
        api.save_order_holds_if_changed(datetime.now().strftime("%Y%m%d"), force=True)
        today = datetime.now().strftime("%Y%m%d")
        e = json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]["000660"]
        self.assertEqual((e["quantity"], sorted(e["order_nos"])), (10, ["51", "52"]))    # both on disk until pending switches
        api.pending_orders["000660"]["order_no"] = "52"
        api.save_order_holds_if_changed(today, turn_end=True)
        e = json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]["000660"]
        self.assertEqual((e["quantity"], e["order_nos"]), (5, ["52"]))

    def test_filled_part_is_not_a_hold_any_more(self):
        api = _api(pending={"000660": {"side": "buy", "quantity": 10, "order_no": "61", "last_filled_qty": 4},
                            "005930": {"side": "buy", "quantity": 7, "legs": [
                                {"order_no": "71", "done": False, "quantity": 3, "last_status": {"filled_qty": 1}},
                                {"order_no": "72", "done": True, "quantity": 4}]}})
        api.save_order_holds_if_changed(DAY, force=True)
        o = self.holds_file()["orders"]
        self.assertEqual((o["000660"]["quantity"], o["005930"]["quantity"]), (6, 2))

    def test_while_stopping_a_status_without_an_order_quantity_is_not_closed(self):
        pending = {"005930": {"side": "sell", "quantity": 3, "pre_submit_qty": 10, "order_no": "11", "reason": "x",
                              "buy_price": 1000.0, "requested_price": 1000.0, "submitted_at": NOW, "exchange": "KRX"}}
        api = _api(pending=pending)
        api.positions = {"005930": {"quantity": 7, "buy_price": 1000.0}}
        api.stopping = True
        api._fetch_today_order_status = lambda *a, **k: {"filled_qty": 0, "remaining_qty": 0, "rejected_qty": 0,
                                                         "order_qty": 0, "cancel_yn": "", "order_no": "11"}
        api.refresh_pending_orders(NOW)
        self.assertIn("005930", api.pending_orders)

    def test_an_accepted_order_is_counted_once(self):
        api = _api()
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "41"}):
            api._order_cash(pdno="000660", ord_dv="buy", ord_qty="5")
        api.pending_orders["000660"] = {"side": "buy", "quantity": 5, "order_no": "41"}   # registered in the same turn
        api.save_order_holds_if_changed(datetime.now().strftime("%Y%m%d"), force=True)   # a stop saves before the turn end
        today = datetime.now().strftime("%Y%m%d")
        e = json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]["000660"]
        self.assertEqual((e["quantity"], e["qty_by_side"]), (5, {"buy": 5}))

    def test_while_stopping_a_sell_still_open_at_the_broker_is_kept(self):
        pending = {"005930": {"side": "sell", "quantity": 3, "pre_submit_qty": 10, "order_no": "11", "reason": "x",
                              "buy_price": 1000.0, "requested_price": 1000.0, "submitted_at": NOW, "exchange": "KRX"}}
        api = _api(pending=pending)
        api.positions = {"005930": {"quantity": 5, "buy_price": 1000.0}}                  # 5 sold by hand meanwhile
        api.stopping = True
        api._fetch_today_order_status = lambda *a, **k: {"filled_qty": 0, "remaining_qty": 3, "rejected_qty": 0,
                                                         "order_qty": 3, "cancel_yn": "", "order_no": "11"}
        api.refresh_pending_orders(NOW)
        self.assertIn("005930", api.pending_orders)

    def test_while_held_an_increase_is_never_the_bots_and_a_decrease_comes_off_the_bot_part(self):
        # [2026-11-25] conservative attribution: bot_only auto-sell must never be able to sell shares bought by hand
        api = _api(holds={"005930": {"side": "buy", "quantity": 5}})
        api.bot_ledger = {"005930": {"acct_qty": 10, "bot_qty": 6}}                       # bot 6, by hand 4
        logs = []
        with mock.patch.object(r003.r023, "save_ledger"), mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "log", logs.append):
            api._update_bot_ledger({"005930": {"quantity": 15}})                           # +5 (maybe the held buy)
            self.assertEqual(api.bot_ledger["005930"], {"acct_qty": 15, "bot_qty": 6})    # not counted for the bot
            self.assertTrue(any("직접 보유로 분류" in m for m in logs))
            api._update_bot_ledger({"005930": {"quantity": 12}})                           # -3 (maybe by hand)
        self.assertEqual(api.bot_ledger["005930"], {"acct_qty": 12, "bot_qty": 3})        # taken from the bot part first

    def test_both_sides_reloaded_are_still_conservative(self):
        r003.save_order_holds(DAY, {"005930": {"side": "both", "quantity": 5, "qty_by_side": {"buy": 2, "sell": 3}}})
        api = _api(holds=r003.load_order_holds(DAY))
        api.bot_ledger = {"005930": {"acct_qty": 10, "bot_qty": 4}}
        with mock.patch.object(r003.r023, "save_ledger"), mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "log"):
            api._update_bot_ledger({"005930": {"quantity": 22}})
        self.assertEqual(api.bot_ledger["005930"], {"acct_qty": 22, "bot_qty": 4})

    def test_this_r003s_own_pending_order_still_attributes_normally(self):
        api = _api(holds={"005930": {"side": "buy", "quantity": 5}},
                   pending={"005930": {"side": "buy", "quantity": 5, "order_no": "81"}})
        api.bot_ledger = {"005930": {"acct_qty": 10, "bot_qty": 6}}
        with mock.patch.object(r003.r023, "save_ledger"), mock.patch.object(r003.r023, "save_snapshot"):
            api._update_bot_ledger({"005930": {"quantity": 15}})
        self.assertEqual(api.bot_ledger["005930"]["bot_qty"], 11)

    def test_previous_day_both_hold_attributes_its_sell_on_the_first_sync(self):
        r003.save_order_holds("20261124", {"005930": {"side": "both", "quantity": 5, "qty_by_side": {"buy": 2, "sell": 3}}})
        api = _api()
        api._load_startup_holds(DAY)
        self.assertEqual(api.order_holds, {})                                              # expired: no order hold today
        self.assertEqual((api._bot_order_window["buy"]["005930"], api._bot_order_window["sell"]["005930"]), (1, 1))
        api.bot_ledger = {"005930": {"acct_qty": 10, "bot_qty": 4}}                       # bot 4, by hand 6
        with mock.patch.object(r003.r023, "save_ledger"), mock.patch.object(r003.r023, "save_snapshot"):
            api._update_bot_ledger({"005930": {"quantity": 7}})                            # the bot's sell of 3 filled late
        self.assertEqual(api.bot_ledger["005930"], {"acct_qty": 7, "bot_qty": 1})
        self.assertEqual(api._prev_hold_caps, {})                                           # first sync only

    def test_a_hold_of_unknown_size_attributes_nothing(self):
        api = _api(holds={"005930": {"side": "buy", "quantity": 0, "from_broker": True}})
        api.bot_ledger = {"005930": {"acct_qty": 10, "bot_qty": 4}}
        with mock.patch.object(r003.r023, "save_ledger"), mock.patch.object(r003.r023, "save_snapshot"):
            api._update_bot_ledger({"005930": {"quantity": 15}})
        self.assertEqual(api.bot_ledger["005930"]["bot_qty"], 4)

    def test_holds_file_values_are_normalized_or_hold_everything(self):
        (self.dir / f"order_holds_{DAY}.json").write_text(json.dumps(
            {"date": DAY, "orders": {"5930": {"side": "sell", "order_nos": [123], "quantity": "4"}}}), encoding="utf-8")
        h = r003.load_order_holds(DAY)["005930"]
        self.assertEqual((h["order_nos"], h["quantity"]), (["123"], 4))
        api = _api()
        api._load_startup_holds(DAY)                                                        # the start-up log does not fail
        self.assertIn("005930", api.order_holds)
        (self.dir / f"order_holds_{DAY}.json").write_text(json.dumps(
            {"date": DAY, "orders": {"005930": {"side": "sell", "order_nos": "123"}}}), encoding="utf-8")
        with mock.patch.object(r003, "log"):
            self.assertIsNone(r003.load_order_holds(DAY))                                 # not restorable: hold everything
        for bad in ({"orders": []}, {"orders": None}, {}, {"orders": {"005930": {"side": "sll"}}},
                    {"orders": {"005930": {"side": "buy", "qty_by_side": {"bUy": 4}}}},
                    {"orders": {"005930": {"side": "buy", "qty_by_side": {"sell": 4}}}}):
            (self.dir / f"order_holds_{DAY}.json").write_text(json.dumps({"date": DAY, **bad}), encoding="utf-8")
            with mock.patch.object(r003, "log"):
                self.assertIsNone(r003.load_order_holds(DAY), bad)
        (self.dir / f"order_holds_{DAY}.json").write_text(json.dumps(
            {"date": DAY, "orders": {"005930": {"side": "both", "qty_by_side": {"buy": 1, "sell": 2}}}}), encoding="utf-8")
        self.assertEqual(r003.load_order_holds(DAY)["005930"]["qty_by_side"], {"buy": 1, "sell": 2})

    def test_both_sides_survive_a_reload(self):
        r003.save_order_holds(DAY, {"005930": {"side": "both", "order_nos": []}})
        self.assertEqual(r003.load_order_holds(DAY)["005930"]["side"], "both")

    def test_a_failed_ledger_write_keeps_the_hold_until_a_write_succeeds(self):
        api = _api(holds={"005930": {"side": "buy", "quantity": 5, "release_after_sync": True}})
        api.bot_ledger = {"005930": {"acct_qty": 5, "bot_qty": 0}}
        with mock.patch.object(r003.r023, "save_ledger", side_effect=OSError("disk full")), \
                mock.patch.object(r003.r023, "save_snapshot"):
            api._update_bot_ledger({"005930": {"quantity": 10}})
        self.assertIn("005930", api.order_holds)                                          # not released
        self.assertTrue(api._ledger_dirty)
        with mock.patch.object(r003.r023, "save_ledger") as sl, mock.patch.object(r003.r023, "save_snapshot"):
            api._update_bot_ledger({"005930": {"quantity": 10}})                         # same quantity: written again
        sl.assert_called_once()
        self.assertNotIn("005930", api.order_holds)

    def test_while_stopping_nothing_is_confirmed_without_an_exact_status(self):
        pending = {"005930": {"side": "sell", "quantity": 3, "pre_submit_qty": 3, "order_no": "11", "reason": "x",
                              "buy_price": 1000.0, "requested_price": 1000.0, "submitted_at": NOW, "exchange": "KRX"},
                   "000660": {"side": "sell", "quantity": 1, "submission_unknown": True, "unknown_since": NOW},
                   "035420": {"side": "buy", "quantity": 2, "legs": [{"order_no": "", "done": False, "quantity": 2}]}}
        api = _api(pending=pending)
        api.positions = {"005930": {"quantity": 0, "buy_price": 1000.0}}                 # sold by hand meanwhile
        api.stopping = True
        api._fetch_today_order_status = lambda *a, **k: None
        api.refresh_pending_orders(NOW)
        self.assertEqual(sorted(api.pending_orders), ["000660", "005930", "035420"])     # all kept -> A2 holds
        self.assertFalse(api.pending_orders["035420"]["legs"][0]["done"])

    def test_stop_settling_without_an_order_number_is_unknown(self):
        api = _api()
        api.stopping = True
        rows = pd.DataFrame([{"pdno": "005930", "odno": "11", "ord_qty": "3", "tot_ccld_qty": "3", "rmn_qty": "0"}])
        with mock.patch.object(r003.dsf, "inquire_daily_ccld", create=True, return_value=(rows, None)):
            self.assertIsNone(api._fetch_today_order_status("005930", "buy", NOW, ""))

    def test_open_order_list_without_its_columns_is_not_used(self):
        api = _api()
        with mock.patch.object(r003.dsf, "inquire_psbl_rvsecncl", create=True, return_value=pd.DataFrame([{"x": 1}])):
            self.assertIsNone(api._open_orders_at_broker())
        with mock.patch.object(r003.dsf, "inquire_psbl_rvsecncl", create=True, return_value=None):
            self.assertIsNone(api._open_orders_at_broker())
        with mock.patch.object(r003.dsf, "inquire_psbl_rvsecncl", create=True, return_value=pd.DataFrame()):
            self.assertEqual(api._open_orders_at_broker(), set())                       # a clean empty list

    def test_intent_is_on_disk_before_the_order_and_dropped_at_turn_end(self):
        api = _api()
        seen = []

        def order_cash(**kw):
            today = datetime.now().strftime("%Y%m%d")
            seen.append(json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"])
            return {"rt_cd": "0", "odno": "9"}
        with mock.patch.object(r003.dsf, "order_cash", create=True, side_effect=order_cash):
            api._order_cash(pdno="000660", ord_dv="buy", ord_qty="5", excg_id_dvsn_cd="KRX")
        self.assertEqual(seen[0]["000660"]["side"], "buy")
        self.assertTrue(seen[0]["000660"]["submission_unknown"])
        today = datetime.now().strftime("%Y%m%d")
        api.pending_orders["000660"] = {"side": "buy", "quantity": 5, "order_no": "0000000009"}   # tracked (zeros ignored)
        api.save_order_holds_if_changed(today, turn_end=True)
        self.assertEqual(json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))["orders"]["000660"]["quantity"], 5)
        self.assertEqual(api._submit_intents, {})
        self.assertEqual(api.order_holds, {})
        # accepted but never tracked (pending registration failed): becomes a hold at the turn end
        api.pending_orders.clear()
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "10"}):
            api._order_cash(pdno="005930", ord_dv="sell", ord_qty="2")
        api.save_order_holds_if_changed(today, turn_end=True)
        self.assertEqual(api.order_holds["005930"]["order_nos"], ["10"])
        self.assertTrue(api.has_pending_order("005930"))

    def test_a_stop_during_the_turn_blocks_the_next_order(self):
        api = _api()
        self.stop_flag.return_value = True
        with mock.patch.object(r003.dsf, "order_cash", create=True) as oc:
            self.assertEqual(api._order_cash(pdno="000660", ord_dv="buy", ord_qty="1")["rt_cd"], "1")
        oc.assert_not_called()
        self.assertTrue(api.stopping)

    def test_unreadable_holds_file_holds_everything_until_the_broker_list_is_read(self):
        api = _api()
        api.holds_all = True
        self.assertTrue(api.has_pending_order("123456"))
        with mock.patch.object(api, "_open_orders_at_broker", return_value=None):
            api.check_order_holds(NOW)
        self.assertTrue(api.holds_all)
        api._holds_next_check = None
        with mock.patch.object(api, "_open_orders_at_broker", return_value={("005930", "sell")}):
            api.check_order_holds(NOW)
        self.assertFalse(api.holds_all)
        self.assertEqual(list(api.order_holds), ["005930"])
        self.assertFalse(api.has_pending_order("123456"))

    def test_stop_settling_uses_only_the_row_of_this_order(self):
        api = _api()
        rows = pd.DataFrame([{"pdno": "005930", "odno": "0000000099", "ord_qty": "3", "tot_ccld_qty": "0", "rmn_qty": "0",
                              "cncl_yn": "Y", "ord_dt": DAY, "ord_tmd": "100000"}])
        with mock.patch.object(r003.dsf, "inquire_daily_ccld", create=True, return_value=(rows, None)):
            self.assertIsNotNone(api._fetch_today_order_status("005930", "buy", NOW, "11"))   # normal: unchanged behaviour
            api.stopping = True
            self.assertIsNone(api._fetch_today_order_status("005930", "buy", NOW, "11"))      # stopping: another order
            self.assertIsNotNone(api._fetch_today_order_status("005930", "buy", NOW, "99"))   # leading zeros ignored

    def test_ledger_caps_for_symbols_known_only_from_a_hold(self):
        led = {"005930": {"acct_qty": 10, "bot_qty": 0}}
        out = r003.r023.update_ledger(led, {"005930": 25}, {"005930"}, set(), buy_caps={"005930": 5})
        self.assertEqual(out["005930"], {"acct_qty": 25, "bot_qty": 5})                  # 10 more were bought by hand
        led = {"005930": {"acct_qty": 20, "bot_qty": 10}}
        out = r003.r023.update_ledger(led, {"005930": 12}, set(), {"005930"}, sell_caps={"005930": 3})
        self.assertEqual(out["005930"], {"acct_qty": 12, "bot_qty": 7})                  # bot 3, the other 5 from the manual part
        out = r003.r023.update_ledger(led, {"005930": 12}, set(), {"005930"})
        self.assertEqual(out["005930"]["bot_qty"], 2)                                     # no cap: as before

    def test_holds_are_checked_at_most_every_ten_seconds(self):
        api = _api(holds={"005930": {"side": "buy"}})
        with mock.patch.object(api, "_open_orders_at_broker", return_value={("005930", "buy")}) as q:
            api.check_order_holds(NOW)
            api.check_order_holds(NOW.replace(second=5))
            api.check_order_holds(NOW.replace(second=11))
        self.assertEqual(q.call_count, 2)

    def test_stop_cancels_and_settles_then_holds_what_is_left(self):
        pending = {
            "005930": {"side": "buy", "quantity": 3, "order_no": "11", "order_org_no": "06010", "exchange": "KRX", "submitted_at": NOW},
            "000660": {"side": "buy", "quantity": 2, "order_no": "21", "order_org_no": "06010", "exchange": "KRX", "submitted_at": NOW},
        }
        api = _api(pending=pending)
        status = {"005930": {"filled_qty": 0, "remaining_qty": 0, "rejected_qty": 0, "order_qty": 3, "cancel_yn": "Y"},
                  "000660": {"filled_qty": 0, "remaining_qty": 2, "rejected_qty": 0, "order_qty": 2, "cancel_yn": ""}}
        api._fetch_today_order_status = lambda code, side, now, order_no="": status[code]
        cancels = []

        def rvsecncl(**kw):
            cancels.append(kw["orgn_odno"])
            return {"rt_cd": "0"} if kw["orgn_odno"] == "11" else {"rt_cd": "1", "msg1": "already filled"}
        ticks = iter(range(1000))                                                        # one second per clock read
        with mock.patch.object(r003.dsf, "order_rvsecncl", create=True, side_effect=rvsecncl), \
                mock.patch("time.sleep"), mock.patch("time.monotonic", side_effect=lambda: float(next(ticks))):
            api.close_orders_for_stop(budget_s=6.0)
        self.assertTrue(api.stopping)
        self.assertEqual(sorted(cancels), ["11", "21"])
        self.assertNotIn("005930", api.pending_orders)                                   # cancel confirmed -> closed
        self.assertIn("000660", api.pending_orders)                                      # still open at the deadline
        today = datetime.now().strftime("%Y%m%d")                                        # the stop writes today's file
        held = json.loads((self.dir / f"order_holds_{today}.json").read_text(encoding="utf-8"))
        self.assertEqual(list(held["orders"]), ["000660"])

    def test_broker_query_failure_is_not_an_empty_list(self):
        api = _api()

        def failing(**kw):
            print("rt_cd : 1 / msg_cd : EGW00123 / msg1 : token expired")
            return pd.DataFrame()

        def ok(**kw):
            return pd.DataFrame([{"pdno": "005930", "sll_buy_dvsn_cd": "01", "psbl_qty": "3"},
                                 {"pdno": "000660", "sll_buy_dvsn_cd": "02", "psbl_qty": "0"}])
        with mock.patch.object(r003.dsf, "inquire_psbl_rvsecncl", create=True, side_effect=failing):
            self.assertIsNone(api._open_orders_at_broker())
        with mock.patch.object(r003.dsf, "inquire_psbl_rvsecncl", create=True, side_effect=ok):
            self.assertEqual(api._open_orders_at_broker(), {("005930", "sell")})


if __name__ == "__main__":
    unittest.main()
