# -*- coding: utf-8 -*-
"""[2026-11-27] VWAP trade live orders (r001 VWAP_TRADE_LIVE_ORDERS): the side-strategy book "vwap" of the fill ledger (r023) beside
the edge book, and the order / exit / limit paths of r003 with strategy "vwap". No broker call, no file outside a temp folder."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from unittest import mock

os.environ.setdefault("AUTO_TRADING_LIVE_OVERRIDES", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "tests"))

import r001_define_config as r001  # noqa: E402
import r023_bot_holdings as r023  # noqa: E402
from test_edge_live import COSTS, DAY, NOW, _api, r003, row  # noqa: E402


def side_state(strategy, code="005930", qty=10, px=10000, odno="1"):
    st = r023.new_fill_state()
    st["codes"][code] = {"bot_qty": 0, "manual_qty": 0}
    r023.register_order(st, DAY, odno, code, "buy", qty, NOW.isoformat(timespec="seconds"), strategy=strategy)
    r023.apply_fill_rows(st, [row(odno, code, "buy", qty, qty, px)], NOW, COSTS)
    return st


class SideBookTest(unittest.TestCase):
    def test_each_side_strategy_has_its_own_book_and_day(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "1", "005930", "buy", 10, "2026-11-25T09:30:00", strategy=r023.EDGE)
        r023.register_order(st, DAY, "2", "000660", "buy", 5, "2026-11-25T09:40:00", strategy=r023.VWAP)
        r023.register_order(st, DAY, "3", "035420", "buy", 4)                                   # a rule order
        r023.apply_fill_rows(st, [row("1", "005930", "buy", 10, 10, 10000), row("2", "000660", "buy", 5, 5, 20000),
                                  row("3", "035420", "buy", 4, 4, 30000)], NOW, COSTS)
        self.assertEqual((r023.edge_quantity(st, "005930", r023.EDGE), r023.edge_quantity(st, "005930", r023.VWAP)), (10, 0))
        self.assertEqual((r023.edge_quantity(st, "000660", r023.VWAP), r023.edge_quantity(st, "000660", r023.EDGE)), (5, 0))
        self.assertEqual((r023.edge_quantity(st, "000660"), r023.edge_avg_price(st, "000660")), (5, 20000.0))   # whichever book
        self.assertEqual((r023.side_of(st, "005930"), r023.side_of(st, "000660"), r023.side_of(st, "035420")),
                         (r023.EDGE, r023.VWAP, ""))
        self.assertEqual((r023.edge_codes(st, r023.EDGE), r023.edge_codes(st, r023.VWAP), r023.edge_codes(st)),
                         ({"005930"}, {"000660"}, {"005930", "000660"}))
        # a VWAP sell books its result in the VWAP day record only
        r023.register_order(st, DAY, "4", "000660", "sell", 5, strategy=r023.VWAP)
        r023.apply_fill_rows(st, [row("4", "000660", "sell", 5, 5, 20600)], NOW, (0.0, 0.0))
        self.assertAlmostEqual(r023.edge_day(st, DAY, r023.VWAP)["realized"], 5 * 600.0)
        self.assertEqual(r023.edge_day(st, DAY, r023.EDGE)["realized"], 0.0)
        self.assertEqual(r023.side_of(st, "000660"), "")

    def test_the_books_survive_a_save_and_a_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = side_state(r023.VWAP)
            r023.edge_day(st, DAY, r023.VWAP)["entries"] = 2
            r023.save_fill_state(Path(tmp), st, NOW)
            back, status = r023.load_fill_state(Path(tmp))
            self.assertEqual(status, "ok")
            self.assertEqual((r023.edge_quantity(back, "005930", r023.VWAP), back["vwap_day"]["entries"]), (10, 2))
            self.assertEqual(back["edge"], {})

    def test_an_open_buy_names_its_strategy_before_any_fill(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "9", "005930", "buy", 3, strategy=r023.VWAP)
        self.assertEqual(r023.side_of(st, "005930"), r023.VWAP)
        self.assertEqual(r023.edge_open_buy_codes(st, r023.EDGE), set())


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class VwapLiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.patches = [mock.patch.object(r003, "LIVE_RUNTIME_DIR", self.dir),
                        mock.patch.object(r003, "_stop_requested", return_value=False),
                        mock.patch.object(r003, "log_trade"), mock.patch.object(r003, "_entry_note"),
                        mock.patch.object(r003, "_log_trade_event_banner"),
                        mock.patch.object(r003, "TRADING_STRATEGY_MODE", "normal"),
                        mock.patch.object(r001, "VWAP_TRADE_MODE", "paper"), mock.patch.object(r001, "VWAP_TRADE_LIVE_ORDERS", True),
                        mock.patch.object(r001, "VWAP_LIVE_ORDER_KRW", 200_000), mock.patch.object(r001, "VWAP_LIVE_MAX_POSITIONS", 2),
                        mock.patch.object(r001, "VWAP_LIVE_MAX_ENTRIES_PER_DAY", 3),
                        mock.patch.object(r001, "VWAP_LIVE_DAILY_LOSS_KRW", 20_000),
                        mock.patch.object(r001, "VWAP_TRADE_TP_PCT", 3.0), mock.patch.object(r001, "VWAP_TRADE_SL_PCT", 1.27),
                        mock.patch.object(r001, "VWAP_TRADE_HORIZON_MIN", 300),
                        # the edge limits are tighter: a VWAP order must not be judged by them
                        mock.patch.object(r003, "EDGE_LIVE_ORDER_KRW", 50_000), mock.patch.object(r003, "EDGE_LIVE_MAX_POSITIONS", 1),
                        mock.patch.object(r003, "EDGE_LIVE_MAX_ENTRIES_PER_DAY", 1),
                        mock.patch.object(r003, "_wall_now", return_value=NOW),
                        mock.patch.object(r003, "auto_sell_mode", return_value="all")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_the_settings_that_allow_new_vwap_buys(self):
        self.assertEqual(r003._vwap_live_config_ok(), (True, ""))
        for name, value in (("VWAP_TRADE_LIVE_ORDERS", False), ("VWAP_TRADE_MODE", "off"), ("VWAP_LIVE_ORDER_KRW", 0),
                            ("VWAP_TRADE_SL_PCT", 0.0)):
            with mock.patch.object(r001, name, value):
                self.assertFalse(r003._vwap_live_config_ok()[0], name)
        with mock.patch.object(r003, "auto_sell_mode", return_value="stop_only"):
            self.assertFalse(r003._vwap_live_config_ok()[0])
        with mock.patch.object(r003, "TRADING_STRATEGY_MODE", "edge"):
            self.assertFalse(r003._vwap_live_config_ok()[0])
        self.assertTrue(r003._vwap_live_setup(_api(r023.new_fill_state()))["enabled"])
        self.assertFalse(r003._vwap_live_setup(_api(None))["enabled"])
        self.assertEqual(r001.VWAP_TRADE_LIVE_ORDERS, True)       # patched here; the shipped default is checked below

    def test_the_vwap_limits_are_its_own(self):
        api = _api(r023.new_fill_state())
        self.assertEqual(api.edge_live_block_reason("005930", NOW, r023.VWAP), "")
        day = r023.edge_day(api._fills, DAY, r023.VWAP)
        day["entries"] = 3
        self.assertTrue(api.edge_live_block_reason("005930", NOW, r023.VWAP).startswith("MAX_ENTRIES_PER_DAY_3/3"))
        self.assertEqual(api.edge_live_block_reason("005930", NOW, r023.EDGE), "")              # the edge day is untouched
        day["entries"] = 0
        day["realized"] = -20_000.0
        self.assertTrue(api.edge_live_block_reason("005930", NOW, r023.VWAP).startswith("DAILY_LOSS"))
        day["realized"] = 0.0
        day["sl_codes"] = ["005930"]
        self.assertEqual(api.edge_live_block_reason("005930", NOW, r023.VWAP), "STOPPED_OUT_TODAY")
        # positions are counted per strategy: an edge holding does not use a VWAP slot
        held = _api(side_state(r023.EDGE, "000660"))
        self.assertEqual(held.edge_live_block_reason("005930", NOW, r023.VWAP), "")
        self.assertTrue(held.edge_live_block_reason("005930", NOW, r023.EDGE).startswith("MAX_POSITIONS"))

    def _buy(self, api, result, strategy, qty=50, ask=10_050.0):
        sent = []

        def order(**kw):
            sent.append((kw, getattr(api, "_order_strategy", "")))
            return result
        with mock.patch.object(api, "get_affordable_buy_qty", return_value=qty), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(10_000.0, ask)), \
                mock.patch.object(api, "_order_cash", side_effect=order), mock.patch.object(api, "persist_live_state"):
            ok = api.place_buy_order("005930", 10_020.0, qty, NOW, False, "regular", buy_detail="x", code_name="삼성전자",
                                     strategy=strategy)
        return ok, sent

    def test_the_vwap_buy_is_one_limit_order_at_the_ask_within_its_own_amount(self):
        api = _api(r023.new_fill_state())
        ok, sent = self._buy(api, {"rt_cd": "0", "odno": "0000301", "krx_fwdg_ord_orgno": "91"}, r023.VWAP)
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)
        kw, strategy = sent[0]
        self.assertEqual((kw["ord_dvsn"], kw["ord_unpr"], kw["ord_qty"], strategy), ("00", "10050", "19", r023.VWAP))   # 200,000 // 10,050
        self.assertEqual(api.pending_orders["005930"]["strategy"], r023.VWAP)
        self.assertEqual((r023.edge_day(api._fills, DAY, r023.VWAP)["entries"], r023.edge_day(api._fills, DAY, r023.EDGE)["entries"]), (1, 0))
        self.assertEqual(r023.load_fill_state(self.dir)[0]["vwap_day"]["entries"], 1)            # on disk before the order

    def _manage(self, api, price, now=NOW):
        with mock.patch.object(r003, "fetch_live_price", return_value=price), \
                mock.patch.object(api, "_refresh_bot_fills", return_value=True), \
                mock.patch.object(api, "place_sell_order", return_value=True) as sell:
            r003._edge_live_manage(api, "005930", "삼성전자", now)
        return sell

    def test_vwap_exits_use_the_vwap_values_reasons_and_book(self):
        pos = {"005930": {"quantity": 10, "buy_price": 10_000.0, "current_price": 10_000.0}}
        with mock.patch.object(r003, "edge_exit_values", return_value=(0.005, 0.005, 1.0)):      # edge exits must not be used
            for price, now, reason in ((10_100.0, NOW, None), (9_870.0, NOW, "VWAP_SL_1.27PCT"), (10_300.0, NOW, "VWAP_TP_3.00PCT"),
                                       (10_050.0, NOW + timedelta(minutes=301), "VWAP_TIME_300MIN"),
                                       (10_050.0, datetime.combine(NOW.date(), dt_time(15, 20, 5)), "VWAP_CLOSE_1520")):
                api = _api(side_state(r023.VWAP), dict(pos))
                sell = self._manage(api, price, now)
                if reason is None:
                    self.assertEqual(sell.call_count, 0)
                    continue
                args, kw = sell.call_args
                self.assertEqual((args[0], args[1], args[3], kw["strategy"]), ("005930", 10, reason, r023.VWAP), reason)
                self.assertEqual(r023.edge_day(api._fills, now.strftime("%Y%m%d"), r023.VWAP)["sl_codes"],
                                 ["005930"] if reason.startswith("VWAP_SL") else [])
                self.assertEqual(r023.edge_day(api._fills, now.strftime("%Y%m%d"), r023.EDGE)["sl_codes"], [])

    def test_a_side_symbol_is_left_to_its_strategy(self):
        api = _api(side_state(r023.VWAP))
        self.assertTrue(api.is_edge_symbol("005930"))            # "held by a side strategy": the rule paths skip it
        self.assertEqual(api.edge_quantity("005930"), 10)
        state = {"enabled": True, "bar_seen": {}, "confirm": {}, "age": {}, "signal_bar": {}}
        # neither strategy buys a symbol a side strategy already holds
        for strategy in (r023.EDGE, r023.VWAP):
            self.assertFalse(r003._edge_live_try_buy(api, state, "005930", "삼성전자", "005930_삼성전자", NOW, False, 10_000.0, "live",
                                                     None, None, None, set(), set(), set(), set(), None, None, {}, {},
                                                     strategy=strategy))


class ControlRulesTest(unittest.TestCase):
    def test_r016_checks_the_ranges_and_what_switching_on_needs(self):
        import r016_control as r016
        base = {"VWAP_TRADE_MODE": "paper", "AUTO_SELL_MODE": "all", "TRADING_STRATEGY_MODE": "normal"}
        self.assertEqual(r016.validate_vwap_live({"VWAP_TRADE_LIVE_ORDERS": True}, dict(base)), {})
        self.assertIn("VWAP_TRADE_MODE=paper", r016.validate_vwap_live({"VWAP_TRADE_LIVE_ORDERS": True}, {**base, "VWAP_TRADE_MODE": "off"})["VWAP_TRADE_LIVE_ORDERS"])
        self.assertIn("AUTO_SELL_MODE=all", r016.validate_vwap_live({"VWAP_TRADE_LIVE_ORDERS": True}, {**base, "AUTO_SELL_MODE": "stop_only"})["VWAP_TRADE_LIVE_ORDERS"])
        # with the live orders stored on, the trade cannot be switched off / the sell mode narrowed alone
        on = {**base, "VWAP_TRADE_LIVE_ORDERS": True}
        self.assertIn("VWAP_TRADE_MODE", r016.validate_vwap_live({"VWAP_TRADE_MODE": "off"}, dict(on)))
        self.assertEqual(r016.validate_vwap_live({"VWAP_TRADE_MODE": "off", "VWAP_TRADE_LIVE_ORDERS": False}, dict(on)), {})
        self.assertIn("AUTO_SELL_MODE", r016.validate_vwap_live({"AUTO_SELL_MODE": "off"}, dict(on)))
        # ranges
        bad = r016.validate_vwap_live({"VWAP_LIVE_ORDER_KRW": 5, "VWAP_LIVE_MAX_POSITIONS": 9, "VWAP_TRADE_SL_PCT": 0.0, "VWAP_TRADE_TP_PCT": 3.0}, {})
        self.assertEqual(sorted(bad), ["VWAP_LIVE_MAX_POSITIONS", "VWAP_LIVE_ORDER_KRW", "VWAP_TRADE_SL_PCT"])
        ok, rej = r016.validate_assignments({"VWAP_TRADE_MODE": "live"})
        self.assertIn("off / paper", rej["VWAP_TRADE_MODE"])

    def test_the_status_summary_is_per_strategy(self):
        import r016_control as r016
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            live = Path(tmp) / "data" / "live_runtime"
            live.mkdir(parents=True)
            st = side_state(r023.VWAP, "000660", 5, 20000)
            r023.save_fill_state(live, st, NOW)
            paths = SimpleNamespace(runtime_root=Path(tmp))
            v, e = r016.edge_live_summary(paths, "vwap"), r016.edge_live_summary(paths)
            self.assertEqual(([h["code"] for h in v["held"]], v["held"][0]["avg_price"]), (["000660"], 20000.0))
            self.assertEqual(e["held"], [])


class DefaultsTest(unittest.TestCase):
    def test_shipped_defaults_are_off_and_small(self):
        import importlib
        src = (ROOT / "live_trading" / "r001_define_config.py").read_text(encoding="utf-8-sig")
        self.assertIn("VWAP_TRADE_LIVE_ORDERS = False", src)
        self.assertIn("VWAP_LIVE_MAX_POSITIONS = 1 ", src)
        self.assertIn("VWAP_LIVE_ORDER_KRW = 100_000", src)
        self.assertTrue(importlib.util.find_spec("r029_vwap_trade") is not None)

    def test_the_vwap_conditions_are_the_shared_guards_around_the_vwap_signal(self):
        import r005_buy_conditions as r005
        self.assertEqual([c.no for c in r005.VWAP_BUY_CONDITIONS], [1, 2, 3, 4, 5, 6, 7, 8, 9, 23, 24, 25])
        self.assertEqual(r005.VWAP_BUY_CONDITIONS[8].func_name, "_v09_vwap_signal")
        self.assertEqual([c.func_name for c in r005.VWAP_BUY_CONDITIONS if c.no != 9],
                         [c.func_name for c in r005.EDGE_BUY_CONDITIONS if c.no != 9])


if __name__ == "__main__":
    unittest.main()
