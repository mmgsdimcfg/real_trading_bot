# -*- coding: utf-8 -*-
"""[2026-11-13] r023 bot holdings ledger (bot-bought vs hand-bought quantity), r003 AUTO_SELL_SCOPE wiring and r016 holdings."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r016_control as r016  # noqa: E402
import r023_bot_holdings as r023  # noqa: E402

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception:  # broker SDK missing
    r003 = None


class TestLedger(unittest.TestCase):
    def test_bot_buy_then_manual_buy_then_sells(self):
        led = r023.update_ledger({}, {"222800": 3}, bot_buy_codes={"222800"})
        self.assertEqual(led, {"222800": {"acct_qty": 3, "bot_qty": 3}})
        # the user adds 5 by hand (no bot order working) -> the bot part does not grow
        led = r023.update_ledger(led, {"222800": 8})
        self.assertEqual(led["222800"], {"acct_qty": 8, "bot_qty": 3})
        # the user sells 4 by hand -> the manual part shrinks first
        led = r023.update_ledger(led, {"222800": 4})
        self.assertEqual(led["222800"], {"acct_qty": 4, "bot_qty": 3})
        # the user sells 2 more by hand: 1 manual left, so 1 comes out of the bot part
        led = r023.update_ledger(led, {"222800": 2})
        self.assertEqual(led["222800"], {"acct_qty": 2, "bot_qty": 2})
        # the bot sells 1 -> the bot part shrinks first
        led = r023.update_ledger(led, {"222800": 1}, bot_sell_codes={"222800"})
        self.assertEqual(led["222800"], {"acct_qty": 1, "bot_qty": 1})
        # gone from the account -> gone from the ledger
        self.assertEqual(r023.update_ledger(led, {}), {})

    def test_bot_sell_leaves_manual_part(self):
        led = {"010060": {"acct_qty": 8, "bot_qty": 3}}
        led = r023.update_ledger(led, {"010060": 5}, bot_sell_codes={"010060"})
        self.assertEqual(led["010060"], {"acct_qty": 5, "bot_qty": 0})
        self.assertEqual(r023.bot_quantity(led, "10060"), 0)   # codes are normalised to 6 digits

    def test_manual_only_and_clamps(self):
        led = r023.update_ledger({}, {"033780": 4, "000001": 0, "bad": "x"})
        self.assertEqual(led, {"033780": {"acct_qty": 4, "bot_qty": 0}})
        # a corrupt row (bot > account) is clamped, never negative
        self.assertEqual(r023.normalize_ledger({"1": {"acct_qty": 2, "bot_qty": 9}, "2": {"acct_qty": 0, "bot_qty": 1},
                                                "3": "x", "4": {"acct_qty": 3, "bot_qty": -5}}),
                         {"000001": {"acct_qty": 2, "bot_qty": 2}, "000004": {"acct_qty": 3, "bot_qty": 0}})
        # a bot buy while the user also sold more than the bot bought: net down, treated as a manual sell
        led = r023.update_ledger({"5": {"acct_qty": 10, "bot_qty": 2}}, {"000005": 7}, bot_buy_codes={"000005"})
        self.assertEqual(led["000005"], {"acct_qty": 7, "bot_qty": 2})

    def test_seed_and_files(self):
        d = Path(tempfile.mkdtemp())
        (d / "20261112.json").write_text(json.dumps({"traded_today": ["018260"], "positions_meta": {
            "011200": {"buy_session": "synced"}, "403870": {"buy_session": "regular"}}}), encoding="utf-8")
        (d / "20261113.json").write_text(json.dumps({"traded_today": ["033780"]}), encoding="utf-8")  # today's file is not history
        positions = {"018260": {"quantity": 2}, "033780": {"quantity": 4}, "011200": {"quantity": 1},
                     "222800": {"quantity": 3}, "122870": {"quantity": 8}, "403870": {"quantity": 5}, "000000": {"quantity": 0}}
        live_state = {"traded_today": {"122870"}, "positions_meta": {"222800": {"buy_session": "regular"},
                                                                     "033780": {"buy_session": "synced"}}}
        led = r023.seed_ledger(positions, live_state, d, "20261113")
        self.assertEqual({c: r["bot_qty"] for c, r in led.items()},
                         {"018260": 2, "033780": 0, "011200": 0, "222800": 3, "122870": 8, "403870": 5})
        self.assertIsNone(r023.load_ledger(d))
        r023.save_ledger(d, led)
        self.assertEqual(r023.load_ledger(d), led)
        (d / r023.LEDGER_FILENAME).write_text("{broken", encoding="utf-8")
        self.assertIsNone(r023.load_ledger(d))

    def test_snapshot(self):
        d = Path(tempfile.mkdtemp())
        positions = {"222800": {"quantity": 8, "buy_price": 130000.0, "current_price": 131000.0,
                                "buy_time": datetime(2026, 11, 13, 9, 48, 30)},
                     "033780": {"quantity": 4, "buy_price": 170000.0, "current_price": 176100.0, "buy_time": None}}
        snap = r023.build_snapshot(positions, {"222800": {"acct_qty": 8, "bot_qty": 3}}, {"222800": "심텍"}, "20261113",
                                   "bot_only", datetime(2026, 11, 13, 10, 0, 0))
        self.assertEqual(snap["schema"], 1)
        self.assertEqual(snap["auto_sell_scope"], "bot_only")
        self.assertEqual(snap["holdings"][1], {"code": "222800", "name": "심텍", "qty": 8, "bot_qty": 3, "manual_qty": 5,
                                               "avg_price": 130000.0, "current_price": 131000.0,
                                               "buy_time": "2026-11-13T09:48:30", "bought_today": True})
        self.assertEqual((snap["holdings"][0]["bot_qty"], snap["holdings"][0]["manual_qty"], snap["holdings"][0]["bought_today"]),
                         (0, 4, False))
        self.assertIsNone(r023.load_snapshot(d))
        r023.save_snapshot(d, snap)
        self.assertEqual(r023.load_snapshot(d), snap)


class TestR016Holdings(unittest.TestCase):
    def test_holdings_command(self):
        root = Path(tempfile.mkdtemp())
        paths = SimpleNamespace(runtime_root=root, r003_lock=root / "data" / "live_runtime" / ".r003.lock")
        with mock.patch.object(r016.r019, "is_locked", return_value=False):
            self.assertEqual(r016.holdings(paths), {"ok": True, "available": False, "running": False, "holdings": []})
            snap = r023.build_snapshot({"222800": {"quantity": 3, "buy_price": 1.0}}, {"222800": {"acct_qty": 3, "bot_qty": 3}},
                                       {}, "20261113", "bot_only")
            r023.save_snapshot(root / "data" / "live_runtime", snap)
            out = r016.holdings(paths)
        self.assertTrue(out["available"])
        self.assertEqual(out["holdings"][0]["bot_qty"], 3)
        self.assertLess(out["age_seconds"], 60)
        self.assertEqual(r001.AUTO_SELL_SCOPE, "bot_only")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = r016.main(["holdings"], paths=r016.default_paths())
        self.assertEqual(rc, 0)
        self.assertTrue(json.loads(buf.getvalue())["ok"])


@unittest.skipIf(r003 is None, "broker SDK missing")
class TestR003Wiring(unittest.TestCase):
    def _api(self, positions, ledger, pending=None, dry_run=False):
        api = object.__new__(r003.TradingAPI)
        api.positions = positions
        api.pending_orders = pending or {}
        api.bot_ledger = ledger
        api._bot_order_window = {"buy": {}, "sell": {}}
        api._bot_ledger_empty_syncs = 0
        api.live_state = {"positions_meta": {}, "traded_today": set()}
        api.dry_run = dry_run
        return api

    def test_update_bot_ledger_windows(self):
        d = Path(tempfile.mkdtemp())
        logs = []
        with mock.patch.object(r003, "LIVE_RUNTIME_DIR", d), mock.patch.object(r003, "log", logs.append):
            # a bot buy order is working -> the new quantity is the bot's; the window survives 2 syncs after the order is gone
            api = self._api({"222800": {"quantity": 3, "buy_price": 1.0}}, {}, pending={"222800": {"side": "buy"}})
            api._update_bot_ledger(api.positions)
            self.assertEqual(api.bot_quantity("222800"), 3)
            api.pending_orders = {}
            api.positions = {"222800": {"quantity": 5, "buy_price": 1.0}}   # late partial fill seen one sync later
            api._update_bot_ledger(api.positions)
            self.assertEqual(api.bot_quantity("222800"), 5)
            api._update_bot_ledger(api.positions)
            self.assertEqual(api._bot_order_window["buy"], {})
            api.positions = {"222800": {"quantity": 9, "buy_price": 1.0}}   # the user buys 4 by hand afterwards
            api._update_bot_ledger(api.positions)
            self.assertEqual(api.bot_quantity("222800"), 5)
            self.assertEqual(r023.load_ledger(d)["222800"], {"acct_qty": 9, "bot_qty": 5})
            self.assertEqual(r023.load_snapshot(d)["holdings"][0]["manual_qty"], 4)
            # no ledger file yet -> seeded from the live state
            api = self._api({"122870": {"quantity": 8}, "033780": {"quantity": 4}}, None)
            api.live_state = {"traded_today": {"122870"}, "positions_meta": {}}
            api._update_bot_ledger(api.positions)
            self.assertEqual((api.bot_quantity("122870"), api.bot_quantity("033780")), (8, 0))
            self.assertTrue(any("[BOT LEDGER  ] seeded" in line for line in logs))

    def test_scope(self):
        api = self._api({}, {"222800": {"acct_qty": 8, "bot_qty": 3}})
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"):
            self.assertTrue(api.sells_bot_quantity_only())
            self.assertFalse(self._api({}, {}, dry_run=True).sells_bot_quantity_only())
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "all"):
            self.assertFalse(api.sells_bot_quantity_only())
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "typo"):     # unknown value -> the side that sells less
            self.assertTrue(api.sells_bot_quantity_only())

    def _sell(self, api, qty, scope):
        """run place_sell_order up to the order call; returns the quantity sent to the broker (None = no order)"""
        sent = {}

        def order_cash(**kw):
            sent["qty"] = int(kw["ord_qty"])
            return {"rt_cd": "0", "odno": "1"}

        api.env_dv, api.cano, api.acnt_prdt_cd = "real", "0", "01"
        api.trade_lock_until = {}
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", scope), mock.patch.object(r003, "log", lambda *_: None), \
                mock.patch.object(r003.TradingAPI, "sync_positions_from_account", lambda self, force=False: None), \
                mock.patch.object(r003.TradingAPI, "refresh_pending_orders", lambda self, now: None), \
                mock.patch.object(r003, "get_order_spec", lambda now, nxt: {"exchange": "KRX"}), \
                mock.patch.object(r003.dsf, "order_cash", order_cash, create=True), \
                mock.patch.object(r003, "_order_succeeded", lambda r: True):
            ok = api.place_sell_order("222800", qty, datetime(2026, 11, 13, 10, 0), "TEST", False, price=100.0)
        return ok, sent.get("qty")

    def test_place_sell_order_sells_only_the_bot_part(self):
        pos = lambda: {"222800": {"quantity": 8, "buy_price": 100.0, "current_price": 100.0}}  # noqa: E731
        ledger = {"222800": {"acct_qty": 8, "bot_qty": 3}}
        self.assertEqual(self._sell(self._api(pos(), dict(ledger)), 8, "bot_only"), (True, 3))
        self.assertEqual(self._sell(self._api(pos(), dict(ledger)), 2, "bot_only"), (True, 2))
        self.assertEqual(self._sell(self._api(pos(), dict(ledger)), 8, "all"), (True, 8))
        # nothing of this holding is the bot's -> no order at all
        self.assertEqual(self._sell(self._api(pos(), {"222800": {"acct_qty": 8, "bot_qty": 0}}), 8, "bot_only"), (False, None))
        self.assertEqual(self._sell(self._api(pos(), {}), 8, "bot_only"), (False, None))


if __name__ == "__main__":
    unittest.main()
