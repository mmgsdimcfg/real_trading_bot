# -*- coding: utf-8 -*-
"""[2026-11-25] Edge AI live orders beside the rule strategy (r001 EDGE_LIVE_ORDERS, edge_live_parallel_plan_20261125.md):
the edge book of the fill ledger (r023), the order / exit / buy paths of r003. No broker call, no file outside a temp folder."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r023_bot_holdings as r023  # noqa: E402

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception:  # broker SDK missing on this machine
    r003 = None

DAY = "20261125"
NOW = datetime(2026, 11, 25, 10, 0, 0)
COSTS = (0.015, 0.20)


def row(odno, code, side, ord_qty, tot, px=0, orgn="", day=DAY):
    return {"ord_dt": day, "odno": odno, "orgn_odno": orgn, "pdno": code, "sll_buy_dvsn_cd": "01" if side == "sell" else "02",
            "ord_qty": str(ord_qty), "tot_ccld_qty": str(tot), "rmn_qty": str(ord_qty - tot), "cncl_yn": "", "rjct_qty": "0",
            "cncl_cfrm_qty": "0", "avg_prvs": str(px)}


def edge_state(code="005930", qty=10, px=10000, manual=0):
    """a ledger in which the edge strategy bought qty at px (order 1)"""
    st = r023.new_fill_state()
    st["codes"][code] = {"bot_qty": 0, "manual_qty": manual}
    r023.register_order(st, DAY, "1", code, "buy", qty, NOW.isoformat(timespec="seconds"), strategy=r023.EDGE)
    r023.apply_fill_rows(st, [row("1", code, "buy", qty, qty, px)], NOW, COSTS)
    return st


class EdgeBookTest(unittest.TestCase):
    def test_only_fills_of_edge_orders_move_the_edge_book(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "1", "005930", "buy", 10, "2026-11-25T09:30:00", strategy=r023.EDGE)
        r023.register_order(st, DAY, "2", "000660", "buy", 5)                                  # a rule order
        r023.apply_fill_rows(st, [row("1", "005930", "buy", 10, 4, 10000), row("2", "000660", "buy", 5, 5, 20000),
                                  row("9001", "005930", "buy", 3, 3, 9000)], NOW, COSTS)       # 9001: bought by hand
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (4, 10000.0))
        self.assertEqual(r023.edge_quantity(st, "000660"), 0)
        self.assertEqual(r023.edge_codes(st), {"005930"})
        self.assertEqual(r023.edge_open_buy_codes(st), {"005930"})
        self.assertEqual(r023.edge_entry_at(st, "005930"), "2026-11-25T09:30:00")
        r023.apply_fill_rows(st, [row("1", "005930", "buy", 10, 10, 10100)], NOW, COSTS)       # 10100 = the order's CUMULATIVE average
        self.assertEqual(r023.edge_quantity(st, "005930"), 10)
        self.assertAlmostEqual(r023.edge_avg_price(st, "005930"), 10100.0)                    # not (4 * 10000 + 6 * 10100) / 10
        # Codex code review: 4 at 9,000, then the order's cumulative average is 9,600 -> the cost is 9,600 (not 9,360);
        # the realized result of a sell in two parts uses each part's own price too
        st2 = r023.new_fill_state()
        r023.register_order(st2, DAY, "1", "000660", "buy", 10, strategy=r023.EDGE)
        r023.apply_fill_rows(st2, [row("1", "000660", "buy", 10, 4, 9000)], NOW, COSTS)
        r023.apply_fill_rows(st2, [row("1", "000660", "buy", 10, 10, 9600)], NOW, COSTS)
        self.assertAlmostEqual(r023.edge_avg_price(st2, "000660"), 9600.0)
        r023.register_order(st2, DAY, "2", "000660", "sell", 10, strategy=r023.EDGE)
        r023.apply_fill_rows(st2, [row("2", "000660", "sell", 10, 5, 9500)], NOW, (0.0, 0.0))
        r023.apply_fill_rows(st2, [row("2", "000660", "sell", 10, 10, 9400)], NOW, (0.0, 0.0))  # the second 5 at 9,300
        self.assertAlmostEqual(r023.edge_day(st2, DAY)["realized"], 5 * (9500 - 9600) + 5 * (9300 - 9600))
        # the status of one order applied at once; the complete listing later adds nothing
        st3 = r023.new_fill_state()
        r023.register_order(st3, DAY, "7", "005930", "buy", 10, strategy=r023.EDGE)
        r023.register_order(st3, "20261124", "8", "000660", "buy", 3, baseline=True)          # another day's baseline: untouched
        r023.apply_order_status(st3, DAY, "0000007", "005930", "buy", 10, 4, 6, 10000.0, False, NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st3, "005930"), r023.fill_bot_quantity(st3, "005930")), (4, 4))
        self.assertIn(r023.fill_key("20261124", "8"), st3["registry"])
        r023.apply_fill_rows(st3, [row("7", "005930", "buy", 10, 4, 10000)], NOW, COSTS)
        self.assertEqual(r023.edge_quantity(st3, "005930"), 4)
        self.assertEqual(r023.apply_order_status(st3, DAY, "7", "000660", "buy", 10, 9, 1, 10000.0), [])   # not this order's symbol
        self.assertEqual(r023.apply_order_status(st3, DAY, "99", "005930", "buy", 10, 9, 1, 10000.0), [])  # not registered
        self.assertEqual(st["codes"]["005930"]["bot_qty"], 10)                                # the hand buy is not the bot's
        self.assertEqual(r023.edge_open_buy_codes(st), set())

    def test_an_edge_sell_books_the_realized_result_with_costs(self):
        st = edge_state(qty=10, px=10000)
        r023.register_order(st, DAY, "3", "005930", "sell", 10, strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("3", "005930", "sell", 10, 10, 9800)], NOW, COSTS)
        self.assertEqual(r023.edge_quantity(st, "005930"), 0)
        self.assertEqual(r023.edge_codes(st), set())
        costs = (9800 * 10 * (0.015 + 0.20) + 10000 * 10 * 0.015) / 100.0
        self.assertAlmostEqual(r023.edge_day(st, DAY)["realized"], 10 * (9800 - 10000) - costs)
        self.assertEqual(r023.edge_day(st, "20261126")["realized"], 0.0)                      # a new day starts from zero

    def test_a_modified_edge_order_stays_an_edge_order_and_the_book_survives_a_restart(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "1", "005930", "buy", 5, strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("1", "005930", "buy", 5, 0), row("2", "005930", "buy", 5, 5, 10000, orgn="1")], NOW, COSTS)
        self.assertEqual(st["registry"][r023.fill_key(DAY, "2")]["strategy"], r023.EDGE)
        self.assertEqual(r023.edge_quantity(st, "005930"), 5)
        r023.edge_day(st, DAY).update(entries=1, sl_codes=["000660"])
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            st2 = r023.load_fill_state(Path(d))[0]
        self.assertEqual((r023.edge_quantity(st2, "005930"), r023.edge_avg_price(st2, "005930")), (5, 10000.0))
        self.assertEqual((st2["edge_day"]["entries"], st2["edge_day"]["sl_codes"]), (1, ["000660"]))
        self.assertEqual(st2["registry"][r023.fill_key(DAY, "1")]["strategy"], r023.EDGE)

    def test_the_edge_quantity_never_exceeds_the_bot_part(self):
        st = edge_state(qty=10, px=10000)
        r023.apply_fill_rows(st, [row("1", "005930", "buy", 10, 10, 10000), row("9002", "005930", "sell", 4, 4, 9900)], NOW, COSTS)
        self.assertEqual((st["codes"]["005930"]["bot_qty"], r023.edge_quantity(st, "005930")), (6, 6))   # sold by hand
        self.assertEqual(r023.edge_avg_price(st, "005930"), 10000.0)
        r023.reconcile_account(st, {"005930": 2}, {"005930"}, DAY)                             # the account: 2 left
        self.assertEqual(r023.edge_quantity(st, "005930"), 2)
        r023.reconcile_account(st, {}, {"005930"}, DAY)
        self.assertEqual(r023.edge_codes(st), set())

    def test_a_fill_without_a_price_makes_the_cost_unknown(self):
        st = edge_state(qty=10, px=0)
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (10, 0.0))
        r023.register_order(st, DAY, "3", "005930", "sell", 10, strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("3", "005930", "sell", 10, 10, 9800)], NOW, COSTS)
        self.assertEqual(r023.edge_day(st, DAY)["unpriced_sells"], 10)                        # the day's loss is not known


def _api(fills, positions=None):
    api = object.__new__(r003.TradingAPI)
    api.env_dv, api.dry_run = "real", False
    api.cano, api.acnt_prdt_cd = "00000000", "01"
    api.live_state = {"positions_meta": {}, "traded_today": set()}
    api.positions = positions or {}
    api.pending_orders = {}
    api.buy_inflight_codes = set()
    api.trade_lock_until = {}
    api.order_holds = {}
    api.holds_all = False
    api.stopping = False
    api._submit_intents = {}
    api._holds_signature = None
    api._bot_order_window = {"buy": {}, "sell": {}}
    api.bot_ledger = {}
    api._fills = fills
    api._fills_broken = False
    api._fills_dirty = False
    api._fills_next_query = None
    api.last_sell_fill_price = {}
    api.last_sell_confirm = {}
    api.startup_position_codes = set()
    return api


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class EdgeLiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.patches = [mock.patch.object(r003, "LIVE_RUNTIME_DIR", self.dir),
                        mock.patch.object(r003, "_stop_requested", return_value=False),
                        mock.patch.object(r003, "log_trade"), mock.patch.object(r003, "_entry_note"),
                        mock.patch.object(r003, "_log_trade_event_banner"),
                        mock.patch.object(r003, "EDGE_ONLY_MODE", "paper"),
                        mock.patch.object(r003, "TRADING_STRATEGY_MODE", "normal"),
                        mock.patch.object(r003, "EDGE_LIVE_ORDERS", True), mock.patch.object(r003, "EDGE_LIVE_ORDER_KRW", 100_000),
                        mock.patch.object(r003, "EDGE_LIVE_MAX_POSITIONS", 1),
                        mock.patch.object(r003, "EDGE_LIVE_MAX_ENTRIES_PER_DAY", 1),
                        mock.patch.object(r003, "EDGE_LIVE_DAILY_LOSS_KRW", 10_000),
                        mock.patch.object(r003, "_wall_now", return_value=NOW),
                        mock.patch.object(r003, "edge_exit_values", return_value=(0.02, 0.017, 60.0)),
                        mock.patch.object(r003, "auto_sell_mode", return_value="all")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    # ---- attribution -----------------------------------------------------------------------------------------------------
    def test_an_edge_buy_and_the_sells_of_an_edge_symbol_are_registered_as_edge_orders(self):
        api = _api(r023.new_fill_state())
        today = datetime.now().strftime("%Y%m%d")
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "0000101"}), \
                mock.patch.object(r003, "_extract_order_number", return_value="0000101"):
            api._order_strategy = r023.EDGE
            api._order_cash(pdno="005930", ord_dv="buy", ord_qty="9")
            api._order_strategy = ""
        e = api._fills["registry"][r023.fill_key(today, "101")]
        self.assertEqual((e["strategy"], e["side"]), (r023.EDGE, "buy"))
        self.assertEqual(r003._intent_detail(next(iter(api._submit_intents.values())))["0000101"]["strategy"], r023.EDGE)
        r023.apply_fill_rows(api._fills, [row("101", "005930", "buy", 9, 9, 10000, day=today)], datetime.now(), COSTS)
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "0000102"}), \
                mock.patch.object(r003, "_extract_order_number", return_value="0000102"):
            api._order_cash(pdno="005930", ord_dv="sell", ord_qty="9")                       # no strategy given: an edge symbol
            api._order_cash(pdno="000660", ord_dv="sell", ord_qty="1")                       # not an edge symbol
        self.assertEqual(api._fills["registry"][r023.fill_key(today, "102")].get("strategy"), r023.EDGE)
        # an edge order kept as a hold (restart) is registered as an edge order again
        api2 = _api(r023.new_fill_state())
        api2._register_hold_orders("005930", {"side": "buy", "order_detail": {
            "77": {"date": DAY, "side": "buy", "qty": 5, "strategy": r023.EDGE}}}, DAY)
        self.assertEqual(api2._fills["registry"][r023.fill_key(DAY, "77")]["strategy"], r023.EDGE)
        self.assertEqual(r003._hold_entries_from_pending({"005930": {"side": "buy", "quantity": 5, "order_no": "77",
                                                                      "submitted_at": NOW, "strategy": r023.EDGE}}
                                                         )["005930"]["order_detail"]["77"]["strategy"], r023.EDGE)

    # ---- limits ----------------------------------------------------------------------------------------------------------
    def test_the_days_limits_and_every_uncertain_state_stop_new_edge_buys(self):
        api = _api(r023.new_fill_state())
        self.assertEqual(api.edge_live_block_reason("005930", NOW), "")
        day = r023.edge_day(api._fills, DAY)
        day["entries"] = 1
        self.assertTrue(api.edge_live_block_reason("005930", NOW).startswith("MAX_ENTRIES_PER_DAY"))
        day["entries"] = 0
        day["realized"] = -10_000.0
        self.assertTrue(api.edge_live_block_reason("005930", NOW).startswith("DAILY_LOSS"))
        day["realized"] = 0.0
        day["unpriced_sells"] = 3
        self.assertEqual(api.edge_live_block_reason("005930", NOW), "REALIZED_PNL_UNKNOWN")
        day.pop("unpriced_sells")
        day["sl_codes"] = ["005930"]
        self.assertEqual(api.edge_live_block_reason("005930", NOW), "STOPPED_OUT_TODAY")
        self.assertEqual(api.edge_live_block_reason("000660", NOW), "")
        api.order_holds = {"035420": {"side": "buy"}}
        self.assertEqual(api.edge_live_block_reason("000660", NOW), "UNRESOLVED_ORDER_HOLD")
        api.order_holds = {}
        api._submit_intents = {"000660:1": {"code": "000660", "side": "buy", "state": "sent", "strategy": r023.EDGE}}
        self.assertTrue(api.edge_live_block_reason("005380", NOW).startswith("MAX_POSITIONS"))   # an answer still open
        api._submit_intents = {}
        with mock.patch.object(r003, "EDGE_LIVE_MAX_ENTRIES_PER_DAY", 5):
            held = _api(edge_state("005930", 10, 10000))
            self.assertTrue(held.edge_live_block_reason("000660", NOW).startswith("MAX_POSITIONS"))
            with mock.patch.object(r003, "EDGE_LIVE_MAX_POSITIONS", 3):
                self.assertEqual(held.edge_live_block_reason("000660", NOW), "")
                held._fills["codes"]["005930"]["manual_qty"] = 2                              # bought by hand on top
                self.assertEqual(held.edge_live_block_reason("000660", NOW), "MIXED_HOLDING_005930")
        broken = _api(r023.new_fill_state())
        broken._fills_broken = True
        self.assertEqual(broken.edge_live_block_reason("005930", NOW), "NO_FILL_LEDGER")
        with mock.patch.object(r003, "EDGE_LIVE_DAILY_LOSS_KRW", 0):
            self.assertTrue(_api(r023.new_fill_state()).edge_live_block_reason("005930", NOW).startswith("DAILY_LOSS"))

    def test_the_settings_that_allow_new_edge_buys(self):
        self.assertEqual(r003._edge_live_config_ok(), (True, ""))
        with mock.patch.object(r003, "EDGE_LIVE_ORDERS", False):
            self.assertFalse(r003._edge_live_config_ok()[0])
        for other in ("edge", "basic"):
            with mock.patch.object(r003, "TRADING_STRATEGY_MODE", other):
                self.assertFalse(r003._edge_live_config_ok()[0])
        with mock.patch.object(r003, "auto_sell_mode", return_value="stop_only"):
            self.assertFalse(r003._edge_live_config_ok()[0])
        with mock.patch.object(r003, "EDGE_ONLY_MODE", "off"):                               # "Edge AI 매매" unchecked
            self.assertFalse(r003._edge_live_config_ok()[0])
        with mock.patch.object(r003, "EDGE_LIVE_MAX_ENTRIES_PER_DAY", 0):
            self.assertFalse(r003._edge_live_config_ok()[0])
        dry = _api(r023.new_fill_state())
        dry.dry_run = True
        self.assertFalse(r003._edge_live_setup(dry)["enabled"])
        self.assertTrue(r003._edge_live_setup(_api(r023.new_fill_state()))["enabled"])
        self.assertFalse(r003._edge_live_setup(_api(None))["enabled"])

    # ---- the buy order ---------------------------------------------------------------------------------------------------
    def _buy(self, api, result, qty=50, ask=10_050.0):
        sent = []

        def order(**kw):
            sent.append((kw, getattr(api, "_order_strategy", "")))
            if isinstance(result, Exception):
                raise result
            return result
        with mock.patch.object(api, "get_affordable_buy_qty", return_value=qty), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(10_000.0, ask)), \
                mock.patch.object(api, "_order_cash", side_effect=order), mock.patch.object(api, "persist_live_state"):
            ok = api.place_buy_order("005930", 10_020.0, qty, NOW, False, "regular", buy_detail="EDGE_LIVE x",
                                     code_name="삼성전자", strategy=r023.EDGE)
        return ok, sent

    def test_the_edge_buy_is_one_limit_order_at_the_ask_within_the_amount(self):
        api = _api(r023.new_fill_state())
        ok, sent = self._buy(api, {"rt_cd": "0", "odno": "0000201", "krx_fwdg_ord_orgno": "91"})
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)                                                       # no split
        kw, strategy = sent[0]
        self.assertEqual((kw["ord_dvsn"], kw["ord_unpr"], kw["ord_qty"], strategy), ("00", "10050", "9", r023.EDGE))  # 100,000 // 10,050
        p = api.pending_orders["005930"]
        self.assertEqual((p["strategy"], p["quantity"], p["legs"]), (r023.EDGE, 9, None))
        self.assertEqual(r023.edge_day(api._fills, DAY)["entries"], 1)
        self.assertEqual(r023.load_fill_state(self.dir)[0]["edge_day"]["entries"], 1)         # on disk before the order
        self.assertEqual(getattr(api, "_order_strategy", ""), "")
        # the day's one entry is used: the next signal does not order
        api2 = _api(api._fills)
        ok2, sent2 = self._buy(api2, {"rt_cd": "0", "odno": "0000202"})
        self.assertEqual((ok2, sent2), (False, []))

    def test_a_rejected_edge_buy_gives_the_entry_back_an_unknown_answer_keeps_it(self):
        api = _api(r023.new_fill_state())
        ok, sent = self._buy(api, {"rt_cd": "1", "msg1": "rejected"})
        self.assertEqual((ok, len(sent), r023.edge_day(api._fills, DAY)["entries"]), (False, 1, 0))
        self.assertNotIn("005930", api.buy_inflight_codes)
        api = _api(r023.new_fill_state())
        ok, sent = self._buy(api, RuntimeError("timeout"))
        self.assertEqual((ok, r023.edge_day(api._fills, DAY)["entries"]), (False, 1))        # may be at the broker
        # Codex code review: an empty answer that the wrapper turned into a failure, but that became an A2 hold - unknown
        api = _api(r023.new_fill_state())

        def lost(**kw):
            api.order_holds["005930"] = {"side": "buy", "submission_unknown": True}
            return {"rt_cd": "1", "msg1": "no answer"}
        with mock.patch.object(api, "get_affordable_buy_qty", return_value=50), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(10_000.0, 10_050.0)), \
                mock.patch.object(api, "_order_cash", side_effect=lost), mock.patch.object(api, "persist_live_state"):
            self.assertFalse(api.place_buy_order("005930", 10_020.0, 50, NOW, False, "regular", strategy=r023.EDGE))
        self.assertEqual(r023.edge_day(api._fills, DAY)["entries"], 1)
        self.assertEqual(api.edge_live_block_reason("000660", NOW), "UNRESOLVED_ORDER_HOLD")
        # too expensive for the amount / no ask / not KRX / dry-run: no order at all
        api = _api(r023.new_fill_state())
        ok, sent = self._buy(api, {"rt_cd": "0", "odno": "1"}, ask=150_000.0)
        self.assertEqual((ok, sent, r023.edge_day(api._fills, DAY)["entries"]), (False, [], 0))
        api = _api(r023.new_fill_state())
        ok, sent = self._buy(api, {"rt_cd": "0", "odno": "1"}, ask=None)
        self.assertEqual((ok, sent, r023.edge_day(api._fills, DAY)["entries"]), (False, [], 0))
        api = _api(r023.new_fill_state())
        api.dry_run = True
        self.assertEqual(self._buy(api, {"rt_cd": "0", "odno": "1"}), (False, []))

    def test_an_unfilled_edge_buy_is_cancelled_not_repriced_and_gives_the_entry_back(self):
        api = _api(r023.new_fill_state())
        r023.edge_day(api._fills, DAY)["entries"] = 1
        sub = datetime.now() - timedelta(seconds=float(r003.EDGE_LIVE_UNFILLED_CANCEL_SECONDS) + 1)
        api.pending_orders["005930"] = {"side": "buy", "quantity": 9, "submitted_at": sub, "order_no": "201", "order_org_no": "91",
                                        "exchange": "KRX", "strategy": r023.EDGE, "reprice_attempt": 0, "cancel_inflight": False,
                                        "legs": None}
        api._last_pending_poll_at = None
        api._last_sync_at = datetime.now()
        status = {"filled_qty": 0, "remaining_qty": 9, "order_qty": 9, "cancel_yn": "", "rejected_qty": 0, "order_no": "201"}
        with mock.patch.object(api, "_fetch_today_order_status", return_value=status), \
                mock.patch.object(api, "_request_buy_reprice") as reprice, mock.patch.object(api, "_mark_bot_order_windows"), \
                mock.patch.object(api, "sync_positions_from_account"), \
                mock.patch.object(r003.dsf, "order_rvsecncl", create=True, return_value={"rt_cd": "0"}) as cancel:
            api.refresh_pending_orders(datetime.now())
            self.assertEqual((reprice.call_count, cancel.call_count), (0, 1))
            self.assertTrue(api.pending_orders["005930"]["cancel_inflight"])
            api._last_pending_poll_at = None
            status.update(remaining_qty=0, cancel_yn="Y")
            api.refresh_pending_orders(datetime.now() + timedelta(seconds=10))
        self.assertNotIn("005930", api.pending_orders)
        self.assertEqual(r023.edge_day(api._fills, datetime.now().strftime("%Y%m%d"))["entries"], 0)

    # ---- exits -----------------------------------------------------------------------------------------------------------
    def _manage(self, api, price, now=NOW, mode="all"):
        with mock.patch.object(r003, "fetch_live_price", return_value=price), \
                mock.patch.object(r003, "auto_sell_mode", return_value=mode), \
                mock.patch.object(api, "_refresh_bot_fills", return_value=True), \
                mock.patch.object(api, "place_sell_order", return_value=True) as sell:
            r003._edge_live_manage(api, "005930", "삼성전자", now)
        return sell

    def test_edge_exits_use_the_edge_books_quantity_and_cost(self):
        pos = {"005930": {"quantity": 13, "buy_price": 9_500.0, "current_price": 10_000.0}}   # 3 more bought by hand, lower average
        for price, now, reason in ((10_100.0, NOW, None), (9_830.0, NOW, "EDGE_SL_1.70PCT"), (10_200.0, NOW, "EDGE_TP_2.00PCT"),
                                   (10_050.0, NOW + timedelta(minutes=61), "EDGE_TIME_60MIN"),
                                   (10_050.0, datetime.combine(NOW.date(), dt_time(15, 20, 5)), "EDGE_CLOSE_1520")):
            api = _api(edge_state("005930", 10, 10_000, manual=3), dict(pos))
            api.trade_lock_until["005930"] = now + timedelta(minutes=5)                       # the buy's cooldown
            sell = self._manage(api, price, now)
            if reason is None:
                self.assertEqual(sell.call_count, 0)
                continue
            args, kw = sell.call_args
            self.assertEqual((args[0], args[1], args[3], kw["strategy"]), ("005930", 10, reason, r023.EDGE), reason)   # 10, not 13
            self.assertNotIn("005930", api.trade_lock_until)
            self.assertEqual(r023.edge_day(api._fills, now.strftime("%Y%m%d"))["sl_codes"],
                             ["005930"] if reason.startswith("EDGE_SL") else [])

    def test_edge_exits_wait_or_narrow_with_the_order_state_and_the_auto_sell_mode(self):
        pos = {"005930": {"quantity": 10, "buy_price": 10_000.0, "current_price": 9_000.0}}
        api = _api(edge_state(), dict(pos))
        api.pending_orders["005930"] = {"side": "sell", "quantity": 10}
        self.assertEqual(self._manage(api, 9_000.0).call_count, 0)                            # its sell is out already
        api = _api(edge_state(), dict(pos))
        api.order_holds = {"005930": {"side": "sell"}}
        self.assertEqual(self._manage(api, 9_000.0).call_count, 0)                            # order state unknown: no order
        api = _api(edge_state(), dict(pos))
        # stage 4 review: what the edge strategy bought is closed by edge rules whatever AUTO_SELL_MODE says (set, unset, reset)
        self.assertEqual(self._manage(api, 9_000.0, mode="off").call_args[0][3], "EDGE_SL_1.70PCT")
        self.assertEqual(self._manage(api, 10_500.0, mode="stop_only").call_args[0][3], "EDGE_TP_2.00PCT")
        self.assertEqual(self._manage(api, 10_050.0, datetime.combine(NOW.date(), dt_time(15, 20, 5)), mode="off").call_args[0][3],
                         "EDGE_CLOSE_1520")
        api = _api(edge_state(), dict(pos))
        self.assertEqual(self._manage(api, None).call_args[0][3], "EDGE_SL_1.70PCT")          # no live price: the account's (9,000)
        self.assertEqual(self._manage(api, 9_000.0, now=datetime.combine(NOW.date(), dt_time(15, 40))).call_count, 0)  # closed
        api = _api(r023.new_fill_state(), dict(pos))                                          # not an edge symbol: nothing
        self.assertEqual(self._manage(api, 9_000.0).call_count, 0)

    def test_the_1520_close_cancels_the_rest_of_an_open_edge_buy_first(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "1", "005930", "buy", 10, NOW.isoformat(timespec="seconds"), strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("1", "005930", "buy", 10, 4, 10_000)], NOW, COSTS)      # 4 of 10 filled
        api = _api(st, {"005930": {"quantity": 4, "buy_price": 10_000.0, "current_price": 10_000.0}})
        api.pending_orders["005930"] = {"side": "buy", "quantity": 10, "last_filled_qty": 4, "order_no": "1", "order_org_no": "9",
                                        "exchange": "KRX", "strategy": r023.EDGE}
        at = datetime.combine(NOW.date(), dt_time(15, 20, 1))

        def sent_cancel(code, pending, filled, remaining):                                   # what an accepted cancel does
            pending["cancel_inflight"] = True
        with mock.patch.object(api, "_cancel_buy_remainder", side_effect=sent_cancel) as cancel:
            sell = self._manage(api, 10_000.0, at)
            self.assertEqual(cancel.call_args[0][2:], (4, 0))                                # 0 = all that is left of the order
            self.assertEqual(sell.call_count, 0)                                             # not while the buy is open
            self.assertEqual(self._manage(api, 10_000.0, at).call_count, 0)
            self.assertEqual(cancel.call_count, 1)                                           # the cancel is not sent twice
            api.pending_orders.pop("005930")                                                 # the buy is closed
            sell = self._manage(api, 10_000.0, at)
        self.assertEqual((sell.call_args[0][1], sell.call_args[0][3]), (4, "EDGE_CLOSE_1520"))
        # a stop loss during the regular hours waits for the open buy the same way
        api.pending_orders["005930"] = {"side": "buy", "quantity": 10, "last_filled_qty": 4, "order_no": "1", "order_org_no": "9",
                                        "exchange": "KRX", "strategy": r023.EDGE}
        with mock.patch.object(api, "_cancel_buy_remainder", side_effect=sent_cancel) as cancel:
            self.assertEqual(self._manage(api, 9_000.0).call_count, 0)
            self.assertEqual(cancel.call_count, 1)
            # the order status cannot be read, but the complete listing shows the buy order finished: closed, the stop goes out
            r023.apply_fill_rows(st, [dict(row("1", "005930", "buy", 10, 4, 10_000), rmn_qty="0", cncl_yn="Y")], NOW, COSTS)
            sell = self._manage(api, 9_000.0)
        self.assertNotIn("005930", api.pending_orders)
        self.assertEqual((sell.call_args[0][1], sell.call_args[0][3]), (4, "EDGE_SL_1.70PCT"))

    def test_an_order_status_reaches_the_edge_book_at_once_and_only_for_this_order(self):
        # Codex code review: the order status shows 4 filled; the complete listing has not run - the stop loss must see them
        api = _api(r023.new_fill_state(), {"005930": {"quantity": 4, "buy_price": 10_000.0, "current_price": 9_000.0}})
        sub = datetime.now()
        r023.register_order(api._fills, sub.strftime("%Y%m%d"), "201", "005930", "buy", 10, sub.isoformat(timespec="seconds"),
                            strategy=r023.EDGE)
        api.pending_orders["005930"] = {"side": "buy", "quantity": 10, "submitted_at": sub, "order_no": "0000201", "order_org_no": "91",
                                        "exchange": "KRX", "strategy": r023.EDGE, "reprice_attempt": 0, "cancel_inflight": False,
                                        "legs": None}
        api._last_pending_poll_at = None
        other = {"filled_qty": 10, "remaining_qty": 0, "order_qty": 10, "cancel_yn": "", "rejected_qty": 0, "order_no": "0000999",
                 "avg_price": 10_000.0}
        mine = {**other, "filled_qty": 4, "remaining_qty": 6, "order_no": "0000201"}
        with mock.patch.object(api, "_mark_bot_order_windows"), mock.patch.object(api, "sync_positions_from_account"), \
                mock.patch.object(api, "_confirm_pending_buy") as confirm, mock.patch.object(api, "_cancel_buy_remainder"):
            with mock.patch.object(api, "_fetch_today_order_status", return_value=other):    # another order's row: not used
                api.refresh_pending_orders(datetime.now())
            self.assertEqual((api.edge_quantity("005930"), confirm.call_count), (0, 0))
            api._last_pending_poll_at = None
            api.pending_orders["005930"].pop("next_status_poll_at", None)
            with mock.patch.object(api, "_fetch_today_order_status", return_value=mine):
                api.refresh_pending_orders(datetime.now() + timedelta(seconds=60))
        self.assertEqual((api.edge_quantity("005930"), api.edge_avg_price("005930")), (4, 10_000.0))
        self.assertEqual(r023.edge_quantity(r023.load_fill_state(self.dir)[0], "005930"), 4)

    def test_an_edge_sell_is_closed_by_its_order_or_the_ledger_never_by_the_account_alone(self):
        # Codex 2nd review: edge 5 + 5 by hand, the edge sell of 5 is open, its status cannot be read; the user sells their 5
        sub = datetime.now()
        day = sub.strftime("%Y%m%d")
        st = edge_state("005930", 5, 10_000, manual=5)
        r023.register_order(st, day, "301", "005930", "sell", 5, strategy=r023.EDGE)
        api = _api(st, {"005930": {"quantity": 5, "buy_price": 10_000.0}})                    # the account dropped to 5
        api.pending_orders["005930"] = {"side": "sell", "quantity": 5, "submitted_at": sub, "order_no": "301", "pre_submit_qty": 10,
                                        "strategy": r023.EDGE, "reason": "EDGE_SL_1.70PCT", "requested_price": 9_800.0,
                                        "buy_price": 10_000.0}
        api._last_pending_poll_at = None
        with mock.patch.object(api, "_mark_bot_order_windows"), mock.patch.object(api, "sync_positions_from_account"), \
                mock.patch.object(api, "_fetch_today_order_status", return_value=None), \
                mock.patch.object(api, "_confirm_pending_sell") as confirm:
            api.refresh_pending_orders(datetime.now())
            self.assertEqual(confirm.call_count, 0)
            self.assertIn("005930", api.pending_orders)                                      # still tracked: no second sell
            self.assertEqual(self._manage(api, 9_000.0, datetime.combine(sub.date(), dt_time(10, 0))).call_count, 0)
            api.settle_edge_pending_by_ledger(datetime.now())                                # not finished in the ledger yet
            self.assertEqual(confirm.call_count, 0)
            r023.apply_fill_rows(st, [row("301", "005930", "sell", 5, 5, 9_800, day=day)], datetime.now(), COSTS)
            api.settle_edge_pending_by_ledger(datetime.now())
        args = confirm.call_args[0]
        self.assertEqual((args[0], args[2]["filled_qty"], args[2]["avg_price"], args[3]), ("005930", 5, 9_800.0, 5))
        # a rule sell keeps the account-quantity confirmation it had
        api2 = _api(r023.new_fill_state(), {})
        api2.pending_orders["000660"] = {"side": "sell", "quantity": 3, "submitted_at": sub, "order_no": "9", "pre_submit_qty": 3}
        api2._last_pending_poll_at = None
        with mock.patch.object(api2, "_mark_bot_order_windows"), mock.patch.object(api2, "sync_positions_from_account"), \
                mock.patch.object(api2, "_fetch_today_order_status", return_value=None), \
                mock.patch.object(api2, "_confirm_pending_sell") as confirm2:
            api2.refresh_pending_orders(datetime.now())
        self.assertEqual(confirm2.call_count, 1)

    def test_an_edge_buy_whose_status_cannot_be_read_is_still_cancelled_in_time_and_closed_by_the_ledger(self):
        sub = datetime.now() - timedelta(seconds=120)
        day = sub.strftime("%Y%m%d")
        st = r023.new_fill_state()
        r023.register_order(st, day, "201", "005930", "buy", 9, strategy=r023.EDGE)
        r023.edge_day(st, day)["entries"] = 1
        api = _api(st)
        api.buy_inflight_codes.add("005930")
        api.pending_orders["005930"] = {"side": "buy", "quantity": 9, "submitted_at": sub, "order_no": "201", "order_org_no": "91",
                                        "exchange": "KRX", "strategy": r023.EDGE, "cancel_inflight": False, "legs": None}
        api._last_pending_poll_at = None
        with mock.patch.object(api, "_mark_bot_order_windows"), mock.patch.object(api, "sync_positions_from_account"), \
                mock.patch.object(api, "_fetch_today_order_status", return_value=None), \
                mock.patch.object(r003.dsf, "order_rvsecncl", create=True, return_value={"rt_cd": "0"}) as cancel:
            api.refresh_pending_orders(datetime.now())
        self.assertEqual(cancel.call_count, 1)
        self.assertIn("005930", api.pending_orders)                                          # kept until its end is known
        r023.apply_fill_rows(st, [dict(row("201", "005930", "buy", 9, 0, day=day), rmn_qty="0", cncl_yn="Y")], datetime.now(), COSTS)
        self.assertEqual(api.edge_codes(), set())                                            # not an edge symbol any more
        api.settle_edge_pending_by_ledger(datetime.now())                                    # the main tick still closes it
        self.assertNotIn("005930", api.pending_orders)
        self.assertNotIn("005930", api.buy_inflight_codes)
        self.assertEqual(r023.edge_day(st, day)["entries"], 0)

    def test_a_fill_booked_without_a_price_gets_its_cost_from_a_later_row(self):
        # Codex 2nd review: the order status has no price; the complete listing shows the same 4 with their amount
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "7", "005930", "buy", 10, strategy=r023.EDGE)
        r023.apply_order_status(st, DAY, "7", "005930", "buy", 10, 4, 6, 0.0, False, NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (4, 0.0))
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            st = r023.load_fill_state(Path(d))[0]
        r023.apply_fill_rows(st, [dict(row("7", "005930", "buy", 10, 4, 10_000), tot_ccld_amt="40000")], NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (4, 10_000.0))
        r023.apply_fill_rows(st, [dict(row("7", "005930", "buy", 10, 10, 10_060), tot_ccld_amt="100600")], NOW, COSTS)
        self.assertAlmostEqual(r023.edge_avg_price(st, "005930"), 10_060.0)                  # the other 6 at 10,100
        # the status carries the amount when the broker gives it
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "8", "005930", "buy", 10, strategy=r023.EDGE)
        r023.apply_order_status(st, DAY, "8", "005930", "buy", 10, 4, 6, 0.0, False, NOW, COSTS, amount=36_000.0)
        self.assertEqual(r023.edge_avg_price(st, "005930"), 9_000.0)

    def test_the_cost_of_a_lot_is_its_orders_cumulative_average_whatever_came_first(self):
        # Codex 3rd review: 1 share booked without a price, then the row shows 2 shares for 21,000 -> 10,500 (not 15,750)
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "7", "005930", "buy", 5, strategy=r023.EDGE)
        r023.apply_order_status(st, DAY, "7", "005930", "buy", 5, 1, 4, 0.0, False, NOW, COSTS)
        r023.apply_fill_rows(st, [dict(row("7", "005930", "buy", 5, 2, 10_500), tot_ccld_amt="21000")], NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (2, 10_500.0))
        # one part priced, the next without a price, then the cumulative row
        r023.apply_order_status(st, DAY, "7", "005930", "buy", 5, 3, 2, 0.0, False, NOW, COSTS)
        # Codex 5th review: 10,500 is the average of 2 shares, 3 are applied now - the cost is not known until a row shows it
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (3, 0.0))
        r023.apply_fill_rows(st, [dict(row("7", "005930", "buy", 5, 5, 10_400), tot_ccld_amt="52000")], NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (5, 10_400.0))
        r023.register_order(st, DAY, "8", "005930", "sell", 5, strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("8", "005930", "sell", 5, 5, 10_000)], NOW, (0.0, 0.0))
        self.assertAlmostEqual(r023.edge_day(st, DAY)["realized"], 5 * (10_000 - 10_400))

    def test_an_edge_buy_left_open_by_an_earlier_run_is_cancelled_at_the_start(self):
        api = _api(r023.new_fill_state())
        api.order_holds = {
            "005930": {"side": "buy", "order_detail": {"201": {"date": DAY, "side": "buy", "qty": 9, "strategy": r023.EDGE,
                                                                 "order_org_no": "91", "exchange": "KRX"}}},
            "000660": {"side": "buy", "order_detail": {"301": {"date": DAY, "side": "buy", "qty": 5}}},          # a rule order
            "035420": {"side": "buy", "order_detail": {"401": {"date": "20261124", "side": "buy", "qty": 5, "strategy": r023.EDGE,
                                                                 "order_org_no": "91"}}},                         # another day
            "005380": {"side": "buy", "order_detail": {"501": {"date": DAY, "side": "buy", "qty": 5, "strategy": r023.EDGE}}}}
        with mock.patch.object(r003.dsf, "order_rvsecncl", create=True, return_value={"rt_cd": "0"}) as cancel:
            api._cancel_recovered_edge_buys(DAY)
        self.assertEqual([(c.kwargs["orgn_odno"], c.kwargs["krx_fwdg_ord_orgno"], c.kwargs["qty_all_ord_yn"])
                          for c in cancel.call_args_list], [("201", "91", "Y")])
        self.assertEqual(sorted(api.order_holds), ["000660", "005380", "005930", "035420"])   # released by the broker's list only
        # the identifiers come from the order's own records
        it = {"order_no": "201", "side": "buy", "quantity": 9, "submitted_at": NOW.isoformat(), "strategy": r023.EDGE,
              "order_org_no": "91", "exchange": "KRX"}
        self.assertEqual(r003._intent_detail(it)["201"]["order_org_no"], "91")
        held = r003._hold_entries_from_pending({"005930": {"side": "buy", "quantity": 9, "order_no": "201", "order_org_no": "91",
                                                           "exchange": "KRX", "submitted_at": NOW, "strategy": r023.EDGE}})
        self.assertEqual(held["005930"]["order_detail"]["201"]["order_org_no"], "91")

    def test_the_time_limit_of_an_edge_buy_does_not_wait_for_the_status_backoff(self):
        api = _api(r023.new_fill_state())
        now = datetime.now()
        api.pending_orders["005930"] = {"side": "buy", "quantity": 9, "submitted_at": now - timedelta(seconds=25), "order_no": "201",
                                        "order_org_no": "91", "exchange": "KRX", "strategy": r023.EDGE, "cancel_inflight": False,
                                        "legs": None, "next_status_poll_at": now + timedelta(seconds=30)}   # the status backs off
        api._last_pending_poll_at = None
        with mock.patch.object(api, "_mark_bot_order_windows"), mock.patch.object(api, "sync_positions_from_account"), \
                mock.patch.object(api, "_fetch_today_order_status") as status, \
                mock.patch.object(r003.dsf, "order_rvsecncl", create=True, return_value={"rt_cd": "1", "msg1": "x"}) as cancel:
            api.refresh_pending_orders(now)
            self.assertEqual((status.call_count, cancel.call_count), (0, 1))
            api._last_pending_poll_at = None
            api.refresh_pending_orders(now + timedelta(seconds=5))                             # a rejected cancel: not every poll
            self.assertEqual(cancel.call_count, 1)
            api._last_pending_poll_at = None
            api.refresh_pending_orders(now + timedelta(seconds=11))
            self.assertEqual(cancel.call_count, 2)

    def test_the_1520_close_goes_out_without_a_price_or_a_cost(self):
        api = _api(edge_state("005930", 10, 0), {"005930": {"quantity": 10, "buy_price": 0.0, "current_price": 0.0}})
        self.assertEqual(self._manage(api, None).call_count, 0)                               # the regular hours: nothing to judge
        sell = self._manage(api, None, datetime.combine(NOW.date(), dt_time(15, 21)))
        self.assertEqual((sell.call_args[0][1], sell.call_args[0][3], sell.call_args.kwargs["price"]), (10, "EDGE_CLOSE_1520", None))
        # the result of an edge sell is booked against the edge cost - on the pending record from the moment it is made
        # (the sell can be confirmed inside place_sell_order); and no edge sell on an account read that is not fresh
        api = _api(edge_state("005930", 10, 10_000, manual=3), {"005930": {"quantity": 13, "buy_price": 9_500.0}})
        seen = []
        with mock.patch.object(api, "sync_positions_from_account"), mock.patch.object(api, "_in_cooldown", return_value=False), \
                mock.patch.object(api, "_refresh_bot_fills", return_value=True), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(api, "_order_cash", return_value={"rt_cd": "0", "odno": "0000401"}), \
                mock.patch.object(api, "_mark_trade_lock"), \
                mock.patch.object(api, "refresh_pending_orders",
                                  side_effect=lambda now: seen.append(api.pending_orders["005930"]["buy_price"])):
            api._acct_stale_since = NOW - timedelta(minutes=5)                                # the account read keeps failing
            self.assertFalse(api.place_sell_order("005930", 10, NOW, "EDGE_SL_1.70PCT", False, price=9_800.0, strategy=r023.EDGE))
            def fresh(force=False):                                                          # a real answer: with the counts
                api._acct_seq = getattr(api, "_acct_seq", 0) + 1
                api._acct_today = {"005930": (10, 0)}
            api.sync_positions_from_account = fresh
            self.assertTrue(api.place_sell_order("005930", 10, NOW, "EDGE_SL_1.70PCT", False, price=9_800.0, strategy=r023.EDGE))
        self.assertEqual(seen, [10_000.0])

    def test_a_listing_behind_the_order_status_does_not_take_edge_shares_away(self):
        # Codex 5th review: the order status applied 10; the complete listing still shows the order with 4, later with 10
        st = r023.new_fill_state()
        st["codes"]["005930"] = {"bot_qty": 0, "manual_qty": 0}
        r023.set_day_anchor(st, DAY, set())
        r023.register_order(st, DAY, "7", "005930", "buy", 10, strategy=r023.EDGE)
        r023.apply_order_status(st, DAY, "7", "005930", "buy", 10, 10, 0, 10_000.0, False, NOW, COSTS)
        for tot in (4, 10):
            rows = [row("7", "005930", "buy", 10, tot, 10_000)]
            r023.apply_fill_rows(st, rows, NOW, COSTS)
            self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [], tot)
            self.assertEqual((st["applied"][r023.fill_key(DAY, "7")], r023.fill_bot_quantity(st, "005930"),
                              r023.edge_quantity(st, "005930")), (10, 10, 10), tot)
        self.assertEqual(r023.clamp_to_day_bound(st, [], DAY), [])                           # the row missing altogether
        self.assertEqual(r023.edge_quantity(st, "005930"), 10)

    def test_no_take_profit_or_stop_loss_on_a_cost_that_is_not_known(self):
        # Codex 5th review: 2 at 10,500 are priced, 3 more came without a price - 10,500 is not the cost of the 5
        st = r023.new_fill_state()
        st["codes"]["005930"] = {"bot_qty": 0, "manual_qty": 0}
        r023.register_order(st, DAY, "7", "005930", "buy", 5, NOW.isoformat(timespec="seconds"), strategy=r023.EDGE)
        r023.apply_fill_rows(st, [dict(row("7", "005930", "buy", 5, 2, 10_500), tot_ccld_amt="21000")], NOW, COSTS)
        r023.apply_order_status(st, DAY, "7", "005930", "buy", 5, 5, 0, 0.0, False, NOW, COSTS)
        api = _api(st, {"005930": {"quantity": 8, "buy_price": 10_300.0, "current_price": 10_170.0}})   # 3 by hand mixed in
        self.assertEqual(api.edge_cost("005930"), 0.0)
        self.assertEqual(self._manage(api, 10_170.0).call_count, 0)                           # -3.1 % of 10,500: no stop loss
        self.assertEqual(self._manage(api, 9_000.0).call_count, 0)
        self.assertEqual(self._manage(api, 10_170.0, NOW + timedelta(minutes=61)).call_args[0][3], "EDGE_TIME_60MIN")
        api.positions["005930"]["quantity"] = 5                                              # nothing by hand: the account's average
        self.assertEqual(api.edge_cost("005930"), 10_300.0)
        self.assertEqual(self._manage(api, 10_000.0).call_args[0][3], "EDGE_SL_1.70PCT")
        r023.apply_fill_rows(st, [dict(row("7", "005930", "buy", 5, 5, 10_200), tot_ccld_amt="51000")], NOW, COSTS)
        api.positions["005930"]["quantity"] = 8
        self.assertEqual(api.edge_cost("005930"), 10_200.0)                                  # the row with the price: judged again
        self.assertEqual(self._manage(api, 10_170.0).call_count, 0)
        self.assertEqual(self._manage(api, 10_000.0).call_args[0][3], "EDGE_SL_1.70PCT")

    def test_an_edge_sell_needs_every_account_read_of_the_last_look(self):
        # Codex 5th review: the first read works, the listing is ahead of it, the second read fails -> no edge sell
        api = _api(edge_state("005930", 10, 10_000), {"005930": {"quantity": 10, "buy_price": 10_000.0}})
        reads = []

        def sync(force=False):
            reads.append(force)
            if len(reads) == 1:
                api._acct_seq = getattr(api, "_acct_seq", 0) + 1
                api._acct_today = {"005930": (10, 0)}
        api.sync_positions_from_account = sync
        api._listed_today = {"005930": (10, 3)}                                              # the listing counts a sale more
        with mock.patch.object(api, "_refresh_bot_fills", return_value=True):
            self.assertIsNone(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x", need_account=True))
            reads.clear()
            api.sync_positions_from_account = lambda force=False: setattr(api, "_acct_seq", getattr(api, "_acct_seq", 0) + 1)
            self.assertEqual(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x", need_account=True), 10)

    def test_an_edge_sell_needs_the_accounts_counts_of_today_for_the_symbol(self):
        # Codex 6th review: the account read works but has no bought / sold counts for the symbol - a hand sale and rebuy the
        # listing does not show yet could not be seen
        api = _api(edge_state("005930", 10, 10_000), {"005930": {"quantity": 10, "buy_price": 10_000.0}})

        def sync(force=False):
            api._acct_seq = getattr(api, "_acct_seq", 0) + 1
        api.sync_positions_from_account = sync
        with mock.patch.object(api, "_refresh_bot_fills", return_value=True):
            api._acct_today, api._acct_fields_ok = {"000660": (0, 0)}, True                   # other rows have them
            self.assertIsNone(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x", need_account=True))
            self.assertEqual(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x"), 10)   # the rule path: as before
            api._acct_today = {"005930": (10, 0)}
            api._listed_today = {"005930": (10, 0)}
            self.assertEqual(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x", need_account=True), 10)
            api._acct_today, api._acct_fields_ok = {}, False                                  # the answer never has the fields
            self.assertEqual(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x", need_account=True), 10)
        # Codex 7th review, through the real account read: the account has the two columns; one answer whose value cannot be
        # read is a bad answer (no edge sell), not "this account has no such fields"
        import pandas as pd

        def answer(buy, sell, with_cols=True):
            r = {"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "10000", "prpr": "10000"}
            if with_cols:
                r.update(thdt_buyqty=buy, thdt_sll_qty=sell)
            return pd.DataFrame([r]), None
        empty = (pd.DataFrame(columns=["pdno", "hldg_qty", "thdt_buyqty", "thdt_sll_qty"]), None)   # no rows, with the columns
        one_col = (pd.DataFrame([{"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "10000", "prpr": "10000",
                                  "thdt_buyqty": "10"}]), None)
        for frames, expect in (([answer("10", "0"), answer("10", "")], None),        # a good answer, then an unreadable value
                               ([answer("10", "0"), answer("10", "inf")], None),
                               ([answer("10", "0"), answer("10", "0")], 10),
                               ([empty, answer("", "", with_cols=False)], None),      # Codex 8th: the columns were seen once
                               ([one_col, one_col], None),                            # one column only: an incomplete answer
                               ([answer("", "", with_cols=False)] * 2, 10)):          # an account without the fields
            api = _api(edge_state("005930", 10, 10_000), {"005930": {"quantity": 10, "buy_price": 10_000.0}})
            api._last_sync_at = None
            api._listed_today = {"005930": (10, 0)}
            with mock.patch.object(r003.dsf, "inquire_balance_rlz_pl", mock.Mock(side_effect=list(frames)), create=True), \
                    mock.patch.object(api, "_update_bot_ledger"), mock.patch.object(api, "_reconcile_pending_positions"), \
                    mock.patch.object(api, "_apply_persisted_position_meta"), mock.patch.object(api, "_record_position_meta"), \
                    mock.patch.object(api, "_refresh_bot_fills", return_value=True):
                api.sync_positions_from_account(force=True)
                self.assertEqual(api._late_bot_sell_qty("005930", 10, "SELL HOLD   ", "x", need_account=True), expect)
        # the lots the edge strategy holds keep their order counts through the ledger clean-up
        st = edge_state("005930", 10, 10_000)
        st["registry"][r023.fill_key(DAY, "1")]["terminal"] = True
        r023.prune_fill_state(st, "20261231")
        self.assertEqual(r023.edge_avg_price(st, "005930"), 10_000.0)
        self.assertIn(r023.fill_key(DAY, "1"), st["applied"])

    def test_the_cost_of_a_lot_ignores_an_older_row_that_comes_late(self):
        # Codex 4th review: 2 shares at an average of 10,000 are applied, then a row of the same order with 1 share at 12,000
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "7", "005930", "buy", 2, strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("7", "005930", "buy", 2, 2, 10_000)], NOW, COSTS)
        r023.apply_fill_rows(st, [row("7", "005930", "buy", 2, 1, 12_000)], NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st, "005930"), r023.edge_avg_price(st, "005930")), (2, 10_000.0))
        r023.register_order(st, DAY, "8", "005930", "sell", 1, strategy=r023.EDGE)
        r023.apply_fill_rows(st, [row("7", "005930", "buy", 2, 2, 10_000), row("8", "005930", "sell", 1, 1, 9_000)], NOW, (0.0, 0.0))
        self.assertAlmostEqual(r023.edge_day(st, DAY)["realized"], -1_000.0)
        # an order price is not a fill price
        self.assertEqual(r023._row_price({"ord_unpr": "10000", "tot_ccld_qty": "1"}), 0.0)

    def test_the_cancel_of_a_recovered_edge_buy_is_tried_again_and_its_identifiers_are_saved_at_once(self):
        api = _api(r023.new_fill_state())
        api.order_holds = {"005930": {"side": "buy", "order_detail": {"201": {"date": DAY, "side": "buy", "qty": 9, "strategy": r023.EDGE,
                                                                                 "order_org_no": "91", "exchange": "KRX"}}}}
        with mock.patch.object(r003.dsf, "order_rvsecncl", create=True, side_effect=RuntimeError("down")) as cancel:
            api._cancel_recovered_edge_buys(DAY)
            api._cancel_recovered_edge_buys(DAY)                                              # within 15 s: not again
            self.assertEqual(cancel.call_count, 1)
            api._edge_recover_next = datetime.now() - timedelta(seconds=1)
            api._cancel_recovered_edge_buys(DAY)
            self.assertEqual(cancel.call_count, 2)
            api._edge_recover_next = datetime.now() - timedelta(seconds=1)
            api.pending_orders["005930"] = {"side": "buy", "order_no": "0000201"}             # this run tracks it itself
            api._cancel_recovered_edge_buys(DAY)
            self.assertEqual(cancel.call_count, 2)
        # an accepted edge buy: order number, organisation number and exchange are in the holds file before anything else
        api = _api(r023.new_fill_state())
        today = datetime.now().strftime("%Y%m%d")
        with mock.patch.object(r003.dsf, "order_cash", create=True,
                               return_value={"rt_cd": "0", "odno": "0000501", "krx_fwdg_ord_orgno": "91"}), \
                mock.patch.object(r003, "_extract_order_number", return_value="0000501"), \
                mock.patch.object(r003, "_extract_order_org_no", return_value="91"):
            api._order_strategy = r023.EDGE
            api._order_cash(pdno="005930", ord_dv="buy", ord_qty="9", excg_id_dvsn_cd="KRX")
        saved = r003.load_order_holds(today)["005930"]["order_detail"]["0000501"]
        self.assertEqual((saved["strategy"], saved["order_org_no"], saved["exchange"]), (r023.EDGE, "91", "KRX"))
        # every cancel of an edge buy's rest goes through one place: not twice within 10 seconds, not while one is in flight
        pending = {"side": "buy", "quantity": 9, "order_no": "501", "order_org_no": "91", "exchange": "KRX", "strategy": r023.EDGE}
        with mock.patch.object(api, "_cancel_buy_remainder") as cancel:
            api._edge_cancel_buy("005930", pending, NOW)
            api._edge_cancel_buy("005930", pending, NOW + timedelta(seconds=5))
            api._edge_cancel_buy("005930", pending, NOW + timedelta(seconds=11))
            pending["cancel_inflight"] = True
            api._edge_cancel_buy("005930", pending, NOW + timedelta(seconds=30))
        self.assertEqual(cancel.call_count, 2)

    def test_no_edge_buy_after_the_entry_window_by_the_wall_clock(self):
        # Codex 9th review: the turn started before the window's end, the wall clock is past it
        end = datetime.combine(NOW.date(), r003.EDGE_LIVE_ENTRY_END)
        api = _api(r023.new_fill_state())
        self.assertEqual(api.edge_live_block_reason("005930", end - timedelta(seconds=1)), "")
        self.assertTrue(api.edge_live_block_reason("005930", end).startswith("ENTRY_WINDOW_CLOSED"))
        with mock.patch.object(r003, "_wall_now", return_value=end + timedelta(seconds=1)):
            self.assertTrue(api.edge_live_block_reason("005930", end - timedelta(seconds=1)).startswith("ENTRY_WINDOW_CLOSED"))
            self.assertEqual(self._buy(api, {"rt_cd": "0", "odno": "1"}), (False, []))
        # the lookups and the save take the order past the window: nothing is sent, the day's entry is free again
        api = _api(r023.new_fill_state())
        clock = iter([NOW, end + timedelta(seconds=1)])                                      # the limit check, then the last look
        with mock.patch.object(r003, "_wall_now", side_effect=lambda: next(clock)):
            ok, sent = self._buy(api, {"rt_cd": "0", "odno": "1"})
        self.assertEqual((ok, sent, r023.edge_day(api._fills, DAY)["entries"]), (False, [], 0))
        self.assertEqual(r023.load_fill_state(self.dir)[0]["edge_day"]["entries"], 0)

    def test_the_edge_buy_amount_is_within_both_limits_at_the_order_price(self):
        api = _api(r023.new_fill_state())
        with mock.patch.object(r003, "EDGE_LIVE_ORDER_KRW", 200_000), mock.patch.object(r003, "MAX_ORDER_AMOUNT_KRW", 100_000):
            ok, sent = self._buy(api, {"rt_cd": "0", "odno": "0000201", "krx_fwdg_ord_orgno": "91"}, qty=10, ask=10_100.0)
        self.assertEqual((ok, sent[0][0]["ord_qty"]), (True, "9"))                           # 10 x 10,100 would be 101,000

    def test_the_edge_sell_is_limited_to_the_edge_book_right_before_the_order(self):
        # Codex code review: edge 10 + 5 by hand; a hand sale of 8 is read at the last look -> the edge strategy holds 7.
        # AUTO_SELL_SCOPE=all (no bot_only limit): still not more than the edge book
        for scope in ("all", "bot_only"):
            api = _api(edge_state("005930", 10, 10_000, manual=5), {"005930": {"quantity": 15, "buy_price": 9_800.0}})
            sent = []

            def late(code, qty, tag, reason="", need_account=False):
                r023.apply_fill_rows(api._fills, [row("1", "005930", "buy", 10, 10, 10_000), row("9005", "005930", "sell", 8, 8, 9_900)],
                                     NOW, COSTS)
                api.positions["005930"]["quantity"] = 7
                return min(qty, 7)
            with mock.patch.object(r003, "AUTO_SELL_SCOPE", scope), mock.patch.object(api, "_in_cooldown", return_value=False), \
                    mock.patch.object(api, "sync_positions_from_account"), mock.patch.object(api, "_refresh_bot_fills", return_value=True), \
                    mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                    mock.patch.object(api, "_late_bot_sell_qty", side_effect=late) as look, \
                    mock.patch.object(api, "_order_cash", side_effect=lambda **kw: sent.append(
                        (kw["ord_qty"], getattr(api, "_order_strategy", ""))) or {"rt_cd": "1"}):
                api.place_sell_order("005930", 10, NOW, "EDGE_SL_1.70PCT", False, price=9_800.0, strategy=r023.EDGE)
            self.assertEqual((look.call_count, look.call_args.kwargs["need_account"]), (1, True), scope)
            self.assertEqual(sent, [("7", r023.EDGE)], scope)
        # the caller's strategy registers the order, and is used once
        api = _api(r023.new_fill_state())
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "odno": "0000301"}), \
                mock.patch.object(r003, "_extract_order_number", return_value="0000301"):
            api._order_strategy = r023.EDGE
            api._order_cash(pdno="005930", ord_dv="sell", ord_qty="3")
            self.assertNotIn("_order_strategy", api.__dict__)
        self.assertEqual(api._fills["registry"][r023.fill_key(datetime.now().strftime("%Y%m%d"), "301")]["strategy"], r023.EDGE)

    def test_the_rule_paths_leave_edge_symbols_alone(self):
        api = _api(edge_state(), {"005930": {"quantity": 10, "buy_price": 10_000.0, "current_price": 10_500.0},
                                  "000660": {"quantity": 5, "buy_price": 20_000.0, "current_price": 21_000.0}})
        at = datetime.combine(NOW.date(), dt_time(15, 20, 1))
        with mock.patch.object(r003, "AUTO_SELL_MODE", "all"), mock.patch.object(r003, "nxt_trading_enabled", return_value=False), \
                mock.patch.object(r003, "_is_today_buy_position", return_value=True), \
                mock.patch.object(r003, "_session_exit_plan", return_value=("sell", "REGULAR_CLOSE")), \
                mock.patch.object(api, "place_sell_order", return_value=True) as sell:
            r003.run_scheduled_liquidations(at, api, {}, {"005930": "삼성전자", "000660": "SK하이닉스"}, {}, DAY, set())
        self.assertEqual([c.args[0] for c in sell.call_args_list], ["000660"])                # the edge symbol is not in it
        # an edge sell does not move the rule strategy's re-entry accounting
        api.live_state["reentry"] = {"005930": {"pos_net_krw": 0.0, "used": 0}}
        pending = {"side": "sell", "quantity": 10, "requested_price": 9_800.0, "buy_price": 10_000.0, "reason": "EDGE_SL_1.70PCT",
                   "strategy": r023.EDGE, "pre_submit_qty": 10}
        with mock.patch.object(r003, "_log_trade_block"), mock.patch.object(api, "persist_live_state"):
            api._confirm_pending_sell("005930", pending, {"filled_qty": 10, "avg_price": 9_800.0}, 0)
        self.assertEqual(api.live_state["reentry"]["005930"], {"pos_net_krw": 0.0, "used": 0})
        self.assertLess(api.live_state["realized_pnl_krw"], 0)                                # the day's total still counts it

    # ---- the buy decision ------------------------------------------------------------------------------------------------
    def test_the_edge_signal_is_judged_once_per_bar_on_its_own_state(self):
        import pandas as pd
        api = _api(r023.new_fill_state())
        edge_live = r003._edge_live_setup(api)
        bars = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-11-25 09:58", "2026-11-25 09:59"]))
        traded = set()
        seen = []

        def check(ctx, conditions):
            seen.append((conditions is r003.EDGE_BUY_CONDITIONS, ctx.state.buy_confirm_state is edge_live["confirm"],
                         ctx.state.signal_buy_bar is edge_live["signal_bar"], ctx.state.traded_today is traded))
            ctx.qty, ctx.session, ctx.buy_reason = 50, "regular", "EDGE_AI_SIGNAL_p=0.6"
            ctx.state.traded_today.add(ctx.code)                                             # what _025 reserves
            return r003.BuyDecision(approved=approved) if hasattr(r003, "BuyDecision") else mock.Mock(approved=approved)

        def run(price_source="live", now=NOW):
            return r003._edge_live_try_buy(api, edge_live, "005930", "삼성전자", "005930", now, False, 10_000.0, price_source,
                                           bars.index[-1], bars, {}, traded, set(), set(), set(), None, services, {}, {})
        import dataclasses
        services = mock.Mock()
        with mock.patch.object(r003, "_get_or_refresh_1min_frame", side_effect=lambda *a: bars), \
                mock.patch.object(r003, "check_buy_conditions", side_effect=check), \
                mock.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj), \
                mock.patch.object(r003, "BuyContext", side_effect=lambda **kw: mock.Mock(**kw)), \
                mock.patch.object(api, "place_buy_order", return_value=False) as order:
            approved = False
            self.assertFalse(run("bar_close(stale_live=40s)"))                               # not a live price: not judged
            self.assertFalse(run(now=datetime.combine(NOW.date(), dt_time(16, 0))))           # not the regular session
            self.assertEqual(seen, [])
            self.assertFalse(run())
            self.assertEqual(seen, [(True, True, True, True)])
            self.assertFalse(run())                                                          # the same bar: not judged again
            self.assertEqual(len(seen), 1)
            bars = pd.DataFrame({"close": [2.0, 3.0]}, index=pd.to_datetime(["2026-11-25 09:59", "2026-11-25 10:00"]))
            approved = True
            self.assertFalse(run())                                                          # approved, the order failed
            self.assertEqual(order.call_args.kwargs["strategy"], r023.EDGE)
            self.assertEqual(traded, set())                                                  # the reservation is released
            edge_live["enabled"] = False
            bars = pd.DataFrame({"close": [3.0, 4.0]}, index=pd.to_datetime(["2026-11-25 10:00", "2026-11-25 10:01"]))
            self.assertFalse(run())
            self.assertEqual(len(seen), 2)                                                   # switched off: nothing judged


if __name__ == "__main__":
    unittest.main()
