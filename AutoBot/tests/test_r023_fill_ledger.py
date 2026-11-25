# -*- coding: utf-8 -*-
"""[2026-11-25] r023 fill-based ledger (r023_fill_ledger_plan_20261125.md, Codex review): the bot's part changes only with the fills
of its own orders; the account is an upper bound (hand sales take the manual part first); bot_only sells use the sellable part."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
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


def row(odno, code, side, ord_qty, tot, rmn=None, orgn="", day=DAY, cncl=""):
    return {"ord_dt": day, "odno": odno, "orgn_odno": orgn, "pdno": code, "sll_buy_dvsn_cd": "01" if side == "sell" else "02",
            "ord_qty": str(ord_qty), "tot_ccld_qty": str(tot), "rmn_qty": str(ord_qty - tot if rmn is None else rmn),
            "cncl_yn": cncl, "rjct_qty": "0", "cncl_cfrm_qty": "0"}


class PureTest(unittest.TestCase):
    def test_partial_then_full_buy_and_no_double_count_after_a_reload(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "0000006053100", "225190", "buy", 2)
        self.assertEqual(r023.apply_fill_rows(st, [row("0006053100", "225190", "buy", 2, 1)], NOW)[0][2], 1)
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            st2, status = r023.load_fill_state(Path(d))
        self.assertEqual(status, "ok")
        r023.apply_fill_rows(st2, [row("0006053100", "225190", "buy", 2, 2)], NOW)
        r023.apply_fill_rows(st2, [row("0006053100", "225190", "buy", 2, 2)], NOW)          # same listing again
        self.assertEqual(r023.fill_bot_quantity(st2, "225190"), 2)
        self.assertTrue(st2["registry"][r023.fill_key(DAY, "6053100")]["terminal"])

    # ---- v3 (Codex 6th): the manual part is tracked from the listing's orders by hand; the account only at stable points ----
    def known(self, bot, manual):
        st = r023.new_fill_state({"005930": bot} if bot else {})
        st["codes"]["005930"] = {"bot_qty": bot, "manual_qty": manual}
        return st

    def test_a_bot_sell_and_a_hand_buy_in_the_same_interval_do_not_cancel_out(self):
        st = self.known(5, 5)
        r023.register_order(st, DAY, "7001", "005930", "sell", 5)
        r023.apply_fill_rows(st, [row("7001", "005930", "sell", 5, 5), row("8001", "005930", "buy", 5, 5)], NOW)
        r023.reconcile_account(st, {"005930": 10}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 0, "manual_qty": 10})

    def test_hand_orders_take_the_manual_part_first(self):
        st = self.known(3, 4)
        r023.apply_fill_rows(st, [row("9999", "005930", "buy", 4, 4), row("9998", "005930", "sell", 2, 2)], NOW)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 3, "manual_qty": 6})
        r023.apply_fill_rows(st, [row("9997", "005930", "sell", 8, 8)], NOW)              # more than the manual part
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 1, "manual_qty": 0})

    def test_the_account_is_compared_only_at_stable_points_manual_first(self):
        st = self.known(4, 6)
        r023.reconcile_account(st, {"005930": 7}, set())                                    # moving: nothing
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 4, "manual_qty": 6})
        r023.reconcile_account(st, {"005930": 7}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 4, "manual_qty": 3})
        r023.reconcile_account(st, {"005930": 2}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 2, "manual_qty": 0})
        r023.reconcile_account(st, {"005930": 9}, {"005930"})                               # more: manual, never the bot
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 2, "manual_qty": 7})

    def test_an_unknown_manual_part_starts_from_the_account(self):
        st = r023.new_fill_state({"005930": 10, "000660": 3})
        r023.reconcile_account(st, {"005930": 14, "000660": 1, "035420": 5}, {"000660"} - {"000660"})   # moving: stays unknown
        self.assertIsNone(st["codes"]["005930"]["manual_qty"])
        r023.reconcile_account(st, {"005930": 14, "000660": 1, "035420": 5}, {"005930", "000660", "035420"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 4})
        self.assertEqual(st["codes"]["000660"], {"bot_qty": 1, "manual_qty": 0})
        self.assertEqual(st["codes"]["035420"], {"bot_qty": 0, "manual_qty": 5})

    def test_codex_5th_and_6th_cases(self):
        # 5th (a): bot 10, bot sell 4 open, 2 sold by hand, the bot sell fills, 4 bought by hand -> bot 4
        st = self.known(10, 0)
        r023.register_order(st, DAY, "72", "005930", "sell", 4)
        r023.apply_fill_rows(st, [row("72", "005930", "sell", 4, 0), row("8101", "005930", "sell", 2, 2)], NOW)
        r023.apply_fill_rows(st, [row("72", "005930", "sell", 4, 4), row("8101", "005930", "sell", 2, 2),
                                  row("8102", "005930", "buy", 4, 4)], NOW)
        r023.reconcile_account(st, {"005930": 8}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 4, "manual_qty": 4})
        # 6th (1): bot 10, 4 sold by hand and a bot buy of 4 before the account read -> bot 10, then 4 bought by hand -> manual 4
        st = self.known(10, 0)
        r023.register_order(st, DAY, "74", "005930", "buy", 4)
        r023.apply_fill_rows(st, [row("8201", "005930", "sell", 4, 4), row("74", "005930", "buy", 4, 4)], NOW)
        r023.apply_fill_rows(st, [row("8201", "005930", "sell", 4, 4), row("74", "005930", "buy", 4, 4),
                                  row("8202", "005930", "buy", 4, 4)], NOW)
        r023.reconcile_account(st, {"005930": 14}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 4})
        # 5th (b): a bot buy of 4 (10 -> 14), 4 sold by hand, bought back by hand -> bot 14 - 4 = 10 (manual part was 0)
        st = self.known(10, 0)
        r023.register_order(st, DAY, "73", "005930", "buy", 4)
        r023.apply_fill_rows(st, [row("73", "005930", "buy", 4, 4), row("8301", "005930", "sell", 4, 4),
                                  row("8302", "005930", "buy", 4, 4)], NOW)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 4})

    def test_reversed_rows_cannot_make_the_bot_part_larger(self):
        # Codex 7th: a hand sale of 4 and a hand buy of 4 in one listing, newest first - sells are applied first
        st = self.known(10, 0)
        r023.apply_fill_rows(st, [row("8402", "005930", "buy", 4, 4), row("8401", "005930", "sell", 4, 4)], NOW)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 6, "manual_qty": 4})

    def test_a_sell_beyond_both_parts_is_paid_from_the_same_listings_buys_bot_first(self):
        # Codex 8th: 0 shares, a bot buy of 4 -> 4 sold by hand -> 4 bought back by hand, all in one listing -> bot 0, manual 4
        st = self.known(0, 0)
        r023.register_order(st, DAY, "76", "005930", "buy", 4)
        r023.apply_fill_rows(st, [row("8602", "005930", "buy", 4, 4), row("8601", "005930", "sell", 4, 4),
                                  row("76", "005930", "buy", 4, 4)], NOW)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 0, "manual_qty": 4})
        r023.reconcile_account(st, {"005930": 4}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 0, "manual_qty": 4})
        # the debt is per symbol: another symbol's bot buy is untouched
        st = self.known(0, 0)
        r023.register_order(st, DAY, "77", "000660", "buy", 3)
        r023.apply_fill_rows(st, [row("8701", "005930", "sell", 2, 2), row("77", "000660", "buy", 3, 3)], NOW)
        self.assertEqual(r023.fill_bot_quantity(st, "000660"), 3)
        # Codex 17th: it is kept between listings (also across a restart) - the bot buy that was sold shows up one listing later
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            st = r023.load_fill_state(Path(d))[0]
        self.assertEqual(st["debt"], {"005930": 2})
        r023.register_order(st, DAY, "78", "005930", "buy", 2)
        r023.apply_fill_rows(st, [row("8701", "005930", "sell", 2, 2), row("78", "005930", "buy", 2, 2)], NOW)
        self.assertEqual((r023.fill_bot_quantity(st, "005930"), st["debt"]), (0, {}))
        # ... until the symbol was matched with the account at a stable point
        st = self.known(0, 0)
        r023.apply_fill_rows(st, [row("8702", "005930", "sell", 2, 2)], NOW)
        r023.reconcile_account(st, {}, {"005930"}, DAY)
        self.assertEqual(st["debt"], {})

    def test_a_sale_listed_before_the_bot_buy_it_sold_with_an_unknown_manual_part(self):
        # Codex 17th: the day's first trade - the bot buys 5, the user sells those 5 and buys 5 by hand; the account already
        # held the 5 at the anchor (manual part unknown: no day bound), the first listing shows only the sale
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "79", "005930", "buy", 5)
        r023.set_day_anchor(st, DAY, {"005930"})
        r023.apply_fill_rows(st, [row("8801", "005930", "sell", 5, 5)], NOW)
        rows = [row("79", "005930", "buy", 5, 5), row("8801", "005930", "sell", 5, 5), row("8802", "005930", "buy", 5, 5)]
        r023.apply_fill_rows(st, rows, NOW)
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [])
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 0)
        r023.reconcile_account(st, {"005930": 5}, {"005930"}, DAY)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 0, "manual_qty": 5})

    def test_prune_keeps_applied_counts_of_dates_still_listed(self):
        # Codex 8th: an old unfinished bot order keeps the listing start back - those days' counts must stay
        st = r023.new_fill_state()
        r023.register_order(st, "20261110", "1", "005930", "sell", 3)                          # unfinished
        r023.register_order(st, "20261112", "2", "005930", "buy", 1)
        st["registry"][r023.fill_key("20261112", "2")]["terminal"] = True
        st["applied"][r023.fill_key("20261112", "2")] = 1
        st["applied"][r023.fill_key("20261112", "9001")] = 5                                   # by hand
        st["applied"][r023.fill_key("20261105", "9002")] = 5                                   # before the listing start
        st["deducted"]["20261110:005930::3"] = {"code": "005930", "qty": 3, "from": 3}          # on the cutoff day itself
        r023.prune_fill_state(st, DAY)
        self.assertIn(r023.fill_key("20261112", "2"), st["registry"])
        self.assertIn(r023.fill_key("20261112", "9001"), st["applied"])
        self.assertNotIn(r023.fill_key("20261105", "9002"), st["applied"])
        self.assertIn("20261110:005930::3", st["deducted"])
        st["registry"][r023.fill_key("20261110", "1")]["terminal"] = True                       # finished: the normal 7 days
        r023.prune_fill_state(st, DAY)
        self.assertNotIn(r023.fill_key("20261112", "9001"), st["applied"])
        self.assertNotIn("20261110:005930::3", st["deducted"])
        st["applied"][r023.fill_key("20261112", "9003")] = 2                                   # the last listed day keeps too
        st["listed_through"] = "20261112"
        r023.prune_fill_state(st, DAY)
        self.assertIn(r023.fill_key("20261112", "9003"), st["applied"])

    def test_a_buy_the_account_showed_first_is_not_added_twice(self):
        # Codex 9th: the account read shows a hand buy of 4 the listing does not show yet; it comes in the next listing
        st = self.known(10, 0)
        r023.reconcile_account(st, {"005930": 14}, {"005930"}, DAY)
        self.assertEqual((st["codes"]["005930"], st["pending_up"]),
                         ({"bot_qty": 10, "manual_qty": 4}, {"005930": {"qty": 4, "date": DAY}}))
        r023.apply_fill_rows(st, [row("8801", "005930", "buy", 4, 4)], NOW)
        self.assertEqual((st["codes"]["005930"], st["pending_up"]), ({"bot_qty": 10, "manual_qty": 4}, {}))
        r023.apply_fill_rows(st, [row("8801", "005930", "buy", 4, 4), row("8802", "005930", "sell", 8, 8),
                                  row("8803", "005930", "buy", 4, 4)], NOW)
        r023.reconcile_account(st, {"005930": 10}, {"005930"})
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 6, "manual_qty": 4})
        with tempfile.TemporaryDirectory() as d:                                             # both new parts survive a restart
            st["listed_through"], st["listed_prev"] = DAY, "20261124"
            st["pending_up"] = {"005930": {"qty": 2, "date": DAY}}
            r023.save_fill_state(Path(d), st, NOW)
            st2 = r023.load_fill_state(Path(d))[0]
        self.assertEqual((st2["listed_through"], st2["listed_prev"], st2["pending_up"]),
                         (DAY, "20261124", {"005930": {"qty": 2, "date": DAY}}))
        # the first value of an unknown manual part is not pending (a hand sale of old shares must not cut the bot part);
        # a buy counted twice there is taken off again at the next stable comparison
        st = r023.new_fill_state({"005930": 10})
        r023.reconcile_account(st, {"005930": 14}, {"005930"}, DAY)
        self.assertEqual(st["pending_up"], {})
        r023.apply_fill_rows(st, [row("8901", "005930", "sell", 3, 3)], NOW)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 1})
        r023.apply_fill_rows(st, [row("8901", "005930", "sell", 3, 3), row("8902", "005930", "buy", 4, 4)], NOW)
        r023.reconcile_account(st, {"005930": 11}, {"005930"}, DAY)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 1})

    # ---- Codex 12th: a bound that does not depend on the order the fills show up in ----
    def test_the_day_bound_holds_whatever_order_the_fills_show_up_in(self):
        # really: 4 sold by hand, then 6 bought by hand (bot 6 / manual 6). The buy shows up one listing earlier than the sale
        final = [row("9401", "005930", "sell", 4, 4), row("9402", "005930", "buy", 6, 6)]
        for early_account in (False, True):
            st = self.known(10, 0)
            r023.set_day_anchor(st, DAY, {"005930"})
            first = [row("9401", "005930", "sell", 4, 0), row("9402", "005930", "buy", 6, 6)]
            if early_account:
                r023.reconcile_account(st, {"005930": 12}, {"005930"}, DAY)
            r023.apply_fill_rows(st, first, NOW)
            r023.clamp_to_day_bound(st, first, DAY)
            r023.apply_fill_rows(st, final, NOW)
            self.assertEqual(r023.clamp_to_day_bound(st, final, DAY), [("005930", 10, 6)])
            r023.reconcile_account(st, {"005930": 12}, {"005930"}, DAY)
            self.assertEqual(st["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
            self.assertEqual(r023.clamp_to_day_bound(st, final, DAY), [])                    # the same listing again: nothing

    def test_the_day_bound_leaves_plain_bot_trading_alone(self):
        # the bot sells its 5 and buys 5 again, then 5 more; the user sells 5 of their own 10
        st = self.known(5, 10)
        r023.set_day_anchor(st, DAY, {"005930"})
        for no, side, qty in (("1", "sell", 5), ("2", "buy", 5), ("3", "buy", 5)):
            r023.register_order(st, DAY, no, "005930", side, qty)
        rows = [row("1", "005930", "sell", 5, 5), row("2", "005930", "buy", 5, 5), row("3", "005930", "buy", 5, 5),
                row("9501", "005930", "sell", 5, 5)]
        r023.apply_fill_rows(st, rows, NOW)
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [])
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 5})
        # rows of another day and another day's anchor are not used
        self.assertEqual(r023.clamp_to_day_bound(st, [row("9502", "005930", "sell", 9, 9, day="20261124")], DAY), [])
        self.assertEqual(r023.clamp_to_day_bound(st, [row("9503", "005930", "sell", 9, 9)], "20261126"), [])

    def test_the_day_bound_is_not_used_where_the_start_is_unknown(self):
        st = r023.new_fill_state({"005930": 10})                                              # manual part unknown
        r023.set_day_anchor(st, DAY, {"005930", "000660"})                                    # 000660: held, not in the ledger
        self.assertIsNone(st["day_anchor"]["codes"]["000660"]["manual_qty"])
        r023.register_order(st, DAY, "4", "000660", "buy", 4)
        rows = [row("9601", "005930", "sell", 3, 3), row("4", "000660", "buy", 4, 4), row("9602", "000660", "sell", 5, 5)]
        st["codes"]["005930"]["manual_qty"] = 5
        st["codes"]["000660"] = {"bot_qty": 4, "manual_qty": 15}
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [])
        # a baselined order: the fills it had at the baseline are in the inherited quantity, later ones count
        st = self.known(10, 0)
        r023.set_day_anchor(st, DAY, {"005930"})
        r023.register_order(st, DAY, "5", "005930", "buy", 6, baseline=True)
        r023.apply_fill_rows(st, [row("5", "005930", "buy", 6, 2)], NOW)                      # baseline at 2
        rows = [row("5", "005930", "buy", 6, 6)]
        r023.apply_fill_rows(st, rows, NOW)
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [])
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 14)
        with tempfile.TemporaryDirectory() as d:                                             # the anchor survives a restart
            r023.save_fill_state(Path(d), st, NOW)
            st2 = r023.load_fill_state(Path(d))[0]
        self.assertEqual(st2["day_anchor"], st["day_anchor"])
        self.assertEqual(st2["registry"][r023.fill_key(DAY, "5")]["base"], 2)

    def test_an_unknown_manual_part_is_set_after_a_second_look(self):
        # Codex 13th: bot 10, manual unknown; a hand buy of 6 is in the account before the listing shows it
        st = r023.new_fill_state({"005930": 10})
        t = NOW
        r023.reconcile_account(st, {"005930": 16}, {"005930"}, DAY, t, 60)
        self.assertIsNone(st["codes"]["005930"]["manual_qty"])                               # first look only
        r023.apply_fill_rows(st, [row("9801", "005930", "buy", 6, 6)], t)                     # its row: counted as applied
        r023.reconcile_account(st, {"005930": 16}, set(), DAY, t + timedelta(seconds=20), 60)  # moving: the wait starts again
        r023.reconcile_account(st, {"005930": 16}, {"005930"}, DAY, t + timedelta(seconds=40), 60)
        r023.reconcile_account(st, {"005930": 16}, {"005930"}, DAY, t + timedelta(seconds=70), 60)   # only 30 s apart
        self.assertIsNone(st["codes"]["005930"]["manual_qty"])
        r023.reconcile_account(st, {"005930": 16}, {"005930"}, DAY, t + timedelta(seconds=101), 60)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 10, "manual_qty": 6})
        rows = [row("9801", "005930", "buy", 6, 6), row("9802", "005930", "sell", 8, 8)]
        r023.apply_fill_rows(st, rows, t)
        r023.apply_fill_rows(st, rows + [row("9803", "005930", "buy", 4, 4)], t)
        r023.reconcile_account(st, {"005930": 12}, {"005930"}, DAY, t + timedelta(seconds=200), 60)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 8, "manual_qty": 4})
        # a changed account quantity starts the wait again; the bot part is cut to the account at once
        st = r023.new_fill_state({"005930": 10})
        r023.reconcile_account(st, {"005930": 7}, {"005930"}, DAY, t, 60)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 7, "manual_qty": None})
        r023.reconcile_account(st, {"005930": 9}, {"005930"}, DAY, t + timedelta(seconds=61), 60)
        self.assertIsNone(st["codes"]["005930"]["manual_qty"])
        r023.reconcile_account(st, {"005930": 9}, {"005930"}, DAY, t + timedelta(seconds=122), 60)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 7, "manual_qty": 2})

    def test_the_day_bound_counts_a_baseline_once(self):
        # Codex 13th: the anchor was made after the baseline (2) - anchor.applied and base are the same fills
        st = self.known(10, 0)
        r023.register_order(st, DAY, "6", "005930", "buy", 6, baseline=True)
        r023.apply_fill_rows(st, [row("6", "005930", "buy", 6, 2)], NOW)
        r023.set_day_anchor(st, DAY, {"005930"})
        self.assertEqual(st["day_anchor"]["applied"], {r023.fill_key(DAY, "6"): 2})
        rows = [row("6", "005930", "buy", 6, 6)]
        r023.apply_fill_rows(st, rows, NOW)
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [])
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 14)

    def test_the_day_bound_starts_with_the_unpaid_sells_of_the_anchor(self):
        # Codex 18th: 2 sold yesterday that nothing could pay; today's anchor keeps them, a bot buy of 2 today pays them
        st = self.known(0, 0)
        r023.apply_fill_rows(st, [row("8703", "005930", "sell", 2, 2, day="20261124")], NOW)
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            st = r023.load_fill_state(Path(d))[0]
        r023.set_day_anchor(st, DAY, set())
        self.assertEqual(st["day_anchor"]["debt"], {"005930": 2})
        st["debt"] = {}                                                                      # as if the ledger had lost it
        r023.register_order(st, DAY, "80", "005930", "buy", 2)
        rows = [row("80", "005930", "buy", 2, 2)]
        r023.apply_fill_rows(st, rows, NOW)
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [("005930", 2, 0)])

    def test_a_hand_round_trip_on_a_bot_symbol_moves_that_much_to_the_manual_part(self):
        # no fill times: bought 5 by hand and sold 5 by hand on a symbol the bot holds - the bound cannot tell it from
        # "sold the bot's 5, then bought 5": the smaller bot part (kept on purpose, logged by r003)
        st = self.known(5, 0)
        r023.set_day_anchor(st, DAY, {"005930"})
        r023.apply_fill_rows(st, [row("9701", "005930", "buy", 5, 5)], NOW)
        rows = [row("9701", "005930", "buy", 5, 5), row("9702", "005930", "sell", 5, 5)]
        r023.apply_fill_rows(st, rows, NOW)
        self.assertEqual(r023.clamp_to_day_bound(st, rows, DAY), [("005930", 5, 0)])
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 0, "manual_qty": 5})

    def test_the_pending_buy_is_kept_apart_from_the_manual_part(self):
        # Codex 10th: account shows a hand buy of 5 first -> its hand sale row comes before the buy row -> the buy row ->
        # a hand sale and rebuy of 5 in one listing: account 10, the bot part must end at 5
        st = self.known(10, 0)
        r023.reconcile_account(st, {"005930": 15}, {"005930"}, DAY)
        r023.apply_fill_rows(st, [row("9102", "005930", "sell", 5, 5)], NOW)
        r023.reconcile_account(st, {"005930": 10}, set(), DAY)                                # moving: nothing
        r023.apply_fill_rows(st, [row("9101", "005930", "buy", 5, 5), row("9102", "005930", "sell", 5, 5)], NOW)
        # Codex 11th: the sale came before its buy was confirmed - it may not use the provisional 5: the smaller bot part
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 5, "manual_qty": 5})
        r023.apply_fill_rows(st, [row("9103", "005930", "sell", 5, 5), row("9104", "005930", "buy", 5, 5)], NOW)
        r023.reconcile_account(st, {"005930": 10}, {"005930"}, DAY)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 5, "manual_qty": 5})
        # Codex 11th: 4 sold and 6 bought by hand, the account (12) seen before both rows -> bot 6 / manual 6, not 8 / 4
        st = self.known(10, 0)
        r023.reconcile_account(st, {"005930": 12}, {"005930"}, DAY)
        r023.apply_fill_rows(st, [], NOW)                                                     # the rows are not listed yet
        r023.apply_fill_rows(st, [row("9302", "005930", "buy", 6, 6), row("9301", "005930", "sell", 4, 4)], NOW)
        r023.reconcile_account(st, {"005930": 12}, {"005930"}, DAY)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
        self.assertEqual(st["pending_up"], {})
        # it ends when its day leaves the listing range
        st["pending_up"] ={"005930": {"qty": 3, "date": "20261123"}, "000660": {"qty": 1, "date": "20261124"}}
        r023.expire_pending(st, "20261124")
        self.assertEqual(list(st["pending_up"]), ["000660"])

    def test_a_hand_sale_the_account_showed_first_is_counted_twice_bot_smaller(self):
        # kept on purpose (Codex 10th agrees): no pending_down - the bot part only gets smaller (5 shares leave auto exit)
        st = self.known(10, 5)
        r023.reconcile_account(st, {"005930": 10}, {"005930"}, DAY)                         # the hand sale seen in the account
        r023.apply_fill_rows(st, [row("9201", "005930", "sell", 5, 5)], NOW)                 # its row later
        r023.reconcile_account(st, {"005930": 10}, {"005930"}, DAY)
        self.assertEqual(st["codes"]["005930"], {"bot_qty": 5, "manual_qty": 5})
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            self.assertEqual(r023.fill_bot_quantity(r023.load_fill_state(Path(d))[0], "005930"), 5)

    def test_a_new_symbols_first_bot_buy_is_not_lost(self):
        # Codex 7th: the account read shows 0, the listing right after shows the bot buy of 4
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "75", "000660", "buy", 4)
        changes = r023.apply_fill_rows(st, [row("75", "000660", "buy", 4, 4)], NOW)
        r023.reconcile_account(st, {}, set() if changes else {"000660"})                    # changed in this listing: moving
        self.assertEqual(r023.fill_bot_quantity(st, "000660"), 4)
        r023.reconcile_account(st, {"000660": 4}, {"000660"})                                # the next round: stable
        self.assertEqual(st["codes"]["000660"], {"bot_qty": 4, "manual_qty": 0})

    def test_open_orders_mark_the_symbol_busy(self):
        self.assertEqual(r023.busy_codes([row("1", "005930", "buy", 4, 1), row("2", "000660", "sell", 2, 2),
                                          row("3", "035420", "buy", 3, 0, rmn=3, cncl="Y")]), {"005930"})

    def test_a_modified_bot_order_keeps_its_fills_the_bots(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "0000197400", "005930", "buy", 1)
        r023.apply_fill_rows(st, [row("0000197400", "005930", "buy", 1, 0, rmn=0),
                                  row("0000257700", "005930", "buy", 1, 1, orgn="0000197400")], NOW)
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 1)
        self.assertEqual(st["registry"][r023.fill_key(DAY, "257700")]["child_of"], r023.fill_key(DAY, "197400"))

    def test_a_row_that_does_not_match_the_registry_is_not_applied(self):
        st = r023.new_fill_state()
        r023.register_order(st, DAY, "5", "005930", "buy", 3)
        out = r023.apply_fill_rows(st, [row("5", "000660", "buy", 3, 3)], NOW)
        self.assertTrue(out[0][3].endswith(":mismatch"))
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 0)

    def test_reserved_sell_and_open_dates(self):
        st = r023.new_fill_state({"005930": 10})
        r023.register_order(st, "20261124", "1", "005930", "sell", 4)
        r023.register_order(st, DAY, "2", "005930", "sell", 3)
        r023.apply_fill_rows(st, [row("2", "005930", "sell", 3, 1)], NOW)
        self.assertEqual(r023.reserved_sell(st, "005930"), 4 + 2)
        self.assertEqual(r023.open_order_dates(st), ["20261124", DAY])
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 9)

    def test_a_modified_order_inherits_the_baseline(self):
        st = r023.new_fill_state({"005930": 10})
        r023.register_order(st, DAY, "81", "005930", "buy", 5, baseline=True)
        r023.apply_fill_rows(st, [row("81", "005930", "buy", 5, 0, rmn=0), row("82", "005930", "buy", 5, 5, orgn="81")], NOW)
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 10)                          # not 15

    def test_a_baseline_waits_for_its_row_and_a_missing_past_order_is_dropped(self):
        st = r023.new_fill_state({"005930": 10})
        r023.register_order(st, DAY, "97", "005930", "buy", 4, baseline=True)
        r023.register_order(st, "20261124", "98", "005930", "buy", 2, baseline=True)
        r023.apply_fill_rows(st, [], NOW)                                                   # neither listed yet
        self.assertIn("005930", r023.baseline_codes(st))
        self.assertIn(r023.fill_key(DAY, "97"), st["registry"])
        self.assertNotIn(r023.fill_key("20261124", "98"), st["registry"])                  # a past day's: not an order
        r023.apply_fill_rows(st, [row("97", "005930", "buy", 4, 4)], NOW)                   # listed: baseline set
        r023.apply_fill_rows(st, [row("97", "005930", "buy", 4, 4)], NOW)
        self.assertEqual(r023.fill_bot_quantity(st, "005930"), 10)                          # not 14

    def test_deduct_once_per_key(self):
        st = r023.new_fill_state({"005930": 10})
        self.assertEqual(r023.deduct_once(st, "20261124:005930::4", "005930", 4), (10, 6))
        self.assertIsNone(r023.deduct_once(st, "20261124:005930::4", "005930", 4))
        with tempfile.TemporaryDirectory() as d:
            r023.save_fill_state(Path(d), st, NOW)
            st2 = r023.load_fill_state(Path(d))[0]
        self.assertIsNone(r023.deduct_once(st2, "20261124:005930::4", "005930", 4))        # also after a restart
        self.assertEqual(r023.fill_bot_quantity(st2, "005930"), 6)

    def test_prune_and_load_states(self):
        st = r023.new_fill_state()
        r023.register_order(st, "20261101", "1", "005930", "buy", 1)
        st["registry"][r023.fill_key("20261101", "1")]["terminal"] = True
        st["applied"][r023.fill_key("20261101", "1")] = 1
        r023.register_order(st, "20261102", "2", "005930", "buy", 1)                          # unfinished: kept
        r023.prune_fill_state(st, DAY)
        self.assertEqual(list(st["registry"]), [r023.fill_key("20261102", "2")])
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(r023.load_fill_state(Path(d)), (None, "missing"))
            (Path(d) / r023.FILL_LEDGER_FILENAME).write_text('{"schema": 2, "codes": []}', encoding="utf-8")
            self.assertEqual(r023.load_fill_state(Path(d)), (None, "corrupt"))


def _fresh_account(api):
    """the account read of a test: nothing changes, but it counts as a fresh read (r003 _acct_seq)"""
    def sync(force=False):
        api._acct_seq = getattr(api, "_acct_seq", 0) + 1
    api.sync_positions_from_account = sync


def _api(tmp: Path, fills=None, positions=None):
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
    api._fills = fills if fills is not None else r023.new_fill_state()
    api._fills_broken = False
    api._fills_dirty = False
    api._fills_next_query = None
    return api


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class WiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.p = mock.patch.object(r003, "LIVE_RUNTIME_DIR", self.dir)
        self.p.start()
        self.p2 = mock.patch.object(r003, "_stop_requested", return_value=False)
        self.p2.start()

    def tearDown(self):
        self.p2.stop()
        self.p.stop()
        self.tmp.cleanup()

    def test_sellable_quantity(self):
        st = r023.new_fill_state({"005930": 8})
        r023.register_order(st, DAY, "1", "005930", "sell", 3)
        api = _api(self.dir, st, {"005930": {"quantity": 10}})
        api._submit_intents = {"005930:9": {"code": "005930", "side": "sell", "quantity": 2, "state": "sent"}}
        self.assertEqual(api.sellable_bot_quantity("005930"), 8 - 3 - 2)
        api.positions["005930"]["quantity"] = 4                                             # the account below the bot part
        self.assertEqual(api.sellable_bot_quantity("005930"), 0)
        api._fills_broken = True
        api.positions["005930"]["quantity"] = 10
        self.assertEqual(api.sellable_bot_quantity("005930"), 0)

    def test_an_accepted_order_is_registered_and_its_fills_applied(self):
        api = _api(self.dir)
        with mock.patch.object(r003.dsf, "order_cash", create=True, return_value={"rt_cd": "0", "ODNO": "0000123", "odno": "0000123"}), \
                mock.patch.object(r003, "_extract_order_number", return_value="0000123"):
            api._order_cash(pdno="000660", ord_dv="buy", ord_qty="5")
        self.assertIn(r023.fill_key(datetime.now().strftime("%Y%m%d"), "123"), api._fills["registry"])
        today = datetime.now().strftime("%Y%m%d")
        with mock.patch.object(r003, "_fetch_ccld_rows", return_value=None):
            self.assertFalse(api._refresh_bot_fills(datetime.now()))                        # incomplete listing: nothing
        self.assertEqual(r023.fill_bot_quantity(api._fills, "000660"), 0)
        api._fills_next_query = None
        with mock.patch.object(r003, "_fetch_ccld_rows", return_value=[row("0000123", "000660", "buy", 5, 5, day=today)]):
            self.assertTrue(api._refresh_bot_fills(datetime.now()))
        self.assertEqual(r023.fill_bot_quantity(api._fills, "000660"), 5)
        saved, status = r023.load_fill_state(self.dir)
        self.assertEqual((status, r023.fill_bot_quantity(saved, "000660")), ("ok", 5))

    def test_ledger_update_clamps_builds_the_view_and_releases_holds(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 6, "000660": 2}))
        api.order_holds = {"005930": {"side": "sell", "release_after_sync": True}}
        with mock.patch.object(r003.r023, "save_snapshot"), mock.patch.object(r003, "_fetch_ccld_rows", return_value=[]):
            api._update_bot_ledger({"005930": {"quantity": 4}, "035420": {"quantity": 3}})
        self.assertEqual(api.bot_ledger, {"005930": {"acct_qty": 4, "bot_qty": 4}, "035420": {"acct_qty": 3, "bot_qty": 0}})
        self.assertNotIn("000660", api._fills["codes"])                                     # left the account
        self.assertEqual(api.order_holds, {})

    def test_a_hold_is_released_only_after_its_orders_finished_and_an_unknown_sell_is_taken_off(self):
        st = r023.new_fill_state({"005930": 10, "000660": 10})
        r023.register_order(st, DAY, "41", "000660", "sell", 4)
        api = _api(self.dir, st)
        api.order_holds = {
            "005930": {"side": "sell", "release_after_sync": True, "submission_unknown": True, "quantity": 10,
                       "qty_by_side": {"sell": 10}, "order_nos": [], "order_detail": {}},
            "000660": {"side": "sell", "release_after_sync": True, "quantity": 4, "qty_by_side": {"sell": 4},
                       "order_nos": ["41"], "order_detail": {"41": {"date": DAY, "side": "sell", "qty": 4}}}}
        pos = {"005930": {"quantity": 20}, "000660": {"quantity": 14}}
        with mock.patch.object(r003.r023, "save_snapshot"), mock.patch.object(r003, "_fetch_ccld_rows", return_value=None):
            api._update_bot_ledger(pos)                                                        # fills unknown: nothing released
        self.assertEqual(sorted(api.order_holds), ["000660", "005930"])
        with mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "_fetch_ccld_rows", return_value=[row("41", "000660", "sell", 4, 2)]):
            api._update_bot_ledger(pos)                                                        # 41 still open: kept
        self.assertIn("000660", api.order_holds)
        self.assertNotIn("005930", api.order_holds)
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 0)                    # the unknown sell of 10 assumed
        with mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "_fetch_ccld_rows", return_value=[row("41", "000660", "sell", 4, 4)]):
            api._update_bot_ledger(pos)
        self.assertEqual(api.order_holds, {})
        self.assertEqual(r023.fill_bot_quantity(api._fills, "000660"), 6)

    def test_an_unknown_sell_deduction_is_saved_before_the_release(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 10}))
        hold = {"side": "sell", "release_after_sync": True, "submission_unknown": True, "quantity": 4, "qty_by_side": {"sell": 4},
                "order_nos": [], "order_detail": {}}
        api.order_holds = {"005930": dict(hold)}
        with mock.patch.object(r003.r023, "save_snapshot"), mock.patch.object(r003, "_fetch_ccld_rows", return_value=[]), \
                mock.patch.object(r003.r023, "save_fill_state", side_effect=OSError("disk full")):
            api._update_bot_ledger({"005930": {"quantity": 20}})
        self.assertIn("005930", api.order_holds)                                             # kept
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 10)                   # undone
        with mock.patch.object(r003.r023, "save_snapshot"), mock.patch.object(r003, "_fetch_ccld_rows", return_value=[]):
            api._update_bot_ledger({"005930": {"quantity": 20}})
        self.assertNotIn("005930", api.order_holds)
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 6)                    # taken off once

    def test_the_switch_baseline_is_for_buys_only(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 10}})
        api._fills_migrated = True
        api._register_hold_orders("005930", {"side": "both", "order_detail": {
            "95": {"date": DAY, "side": "sell", "qty": 4}, "96": {"date": DAY, "side": "buy", "qty": 3}}}, DAY)
        with mock.patch.object(r003, "_fetch_ccld_rows", return_value=[row("95", "005930", "sell", 4, 4),
                                                                      row("96", "005930", "buy", 3, 3)]):
            api._refresh_bot_fills(NOW, force=True)
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 6)                   # the sell counts, the buy not

    def test_an_earlier_days_unknown_sell_is_taken_off_once_at_start(self):
        r003.save_order_holds("20261124", {"005930": {"side": "sell", "submission_unknown": True, "quantity": 4,
                                                       "qty_by_side": {"sell": 4}, "order_nos": [], "submitted_at": "x"}})
        for _ in range(2):                                                                   # two starts
            api = _api(self.dir, r023.load_fill_state(self.dir)[0] or r023.new_fill_state({"005930": 10}))
            api._load_startup_holds(DAY)
            api._save_fills(NOW)
        self.assertEqual(r023.fill_bot_quantity(r023.load_fill_state(self.dir)[0], "005930"), 6)

    def test_an_interrupted_switch_repeats_the_baselines(self):
        st = r023.migrate_from_ledger({"005930": {"acct_qty": 10, "bot_qty": 10}})
        st["migrating"] = True
        r023.save_fill_state(self.dir, st, NOW)                                               # stopped before the holds step
        api = _api(self.dir)
        api._load_fill_ledger(DAY)
        self.assertTrue(api._fills_migrated)
        api._register_hold_orders("005930", {"side": "buy", "order_detail": {"91": {"date": DAY, "side": "buy", "qty": 5}}}, DAY)
        self.assertTrue(api._fills["registry"][r023.fill_key(DAY, "91")]["baseline"])
        api._finish_fill_switch()
        self.assertNotIn("migrating", r023.load_fill_state(self.dir)[0])

    def test_recovered_orders_after_the_switch_start_from_a_baseline(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 10}})
        api._fills_migrated = True
        api._register_hold_orders("005930", {"side": "buy", "order_nos": ["51"],
                                             "order_detail": {"51": {"date": DAY, "side": "buy", "qty": 10}}}, DAY)
        self.assertEqual(api.sellable_bot_quantity("005930"), 0)                            # not known until the baseline
        with mock.patch.object(r003, "_fetch_ccld_rows", return_value=[row("51", "005930", "buy", 10, 10)]):
            api._refresh_bot_fills(NOW, force=True)
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 10)                   # not 20
        self.assertEqual(api.sellable_bot_quantity("005930"), 10)

    def test_a_lost_registry_write_holds_the_symbol_with_its_order_number(self):
        api = _api(self.dir)
        with mock.patch.object(r003.r023, "save_fill_state", side_effect=OSError("disk full")):
            api._register_bot_order("000660", "buy", 5, "0000777")
        self.assertEqual(api.order_holds["000660"]["order_nos"], ["0000777"])
        self.assertTrue(api.has_pending_order("000660"))

    def test_a_bot_only_sell_needs_a_fresh_listing(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 10, "buy_price": 1000.0}})
        _fresh_account(api)
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), mock.patch.object(r003, "_fetch_ccld_rows", return_value=None):
            self.assertFalse(api.place_sell_order("005930", 10, NOW, "test", False))
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"),                 mock.patch.object(r003, "_fetch_ccld_rows", return_value=[row("8501", "005930", "sell", 4, 4, day=NOW.strftime("%Y%m%d")),
                                                                         row("8502", "005930", "buy", 4, 4, day=NOW.strftime("%Y%m%d"))]):
            api._refresh_bot_fills(NOW, force=True)
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 6)                   # the hand sale was found first

    def test_the_bot_part_is_read_again_right_before_the_sell_goes_out(self):
        # Codex 8th: a hand sale between the first listing and the order (NXT price lookup) - the order takes the newer part
        def fresh(now, force=False, want=False):
            calls.append(now)
            if len(calls) == 2:
                api._fills["codes"]["005930"]["bot_qty"] = results.pop(0)
            return oks.pop(0)
        for oks, results, expect in (([True, False], [10], False), ([True, True], [0], False), ([True, True], [6], True)):
            calls = []
            api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 10, "buy_price": 1000.0}})
            _fresh_account(api)
            sent = []

            def order(**kw):                                                                 # no real order: record and stop here
                sent.append(kw)
                raise RuntimeError("stop")
            with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), mock.patch.object(api, "_refresh_bot_fills", side_effect=fresh), \
                    mock.patch.object(api, "_in_cooldown", return_value=False), \
                    mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                    mock.patch.object(api, "_order_cash", side_effect=order):
                try:
                    got = api.place_sell_order("005930", 10, NOW, "test", False)
                except RuntimeError:
                    got = None
            self.assertEqual(len(calls), 2)
            if expect:
                self.assertEqual([str(k.get("ord_qty")) for k in sent], ["6"])
            else:
                self.assertFalse(got)
                self.assertEqual(sent, [])

    def test_a_new_day_reads_from_the_last_listed_day(self):
        # Codex 9th: 4 sold and 4 bought back by hand after the last listing yesterday (account total unchanged)
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 0
        st["listed_through"] = "20261124"
        api = _api(self.dir, st, {"005930": {"quantity": 10}})
        seen = []

        def fetch(env, cano, prdt, start, end):
            seen.append((start, end))
            return [row("8501", "005930", "sell", 4, 4, day="20261124"), row("8502", "005930", "buy", 4, 4, day="20261124")]
        with mock.patch.object(r003, "_fetch_ccld_rows", side_effect=fetch):
            self.assertTrue(api._refresh_bot_fills(NOW))                                     # runs without force: the day changed
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(minutes=5)))               # same day, nothing open: no listing
        self.assertEqual(seen, [("20261124", DAY)])
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 4})
        self.assertEqual((api._fills["listed_through"], api._fills["listed_prev"]), (DAY, "20261124"))

    def test_yesterdays_late_rows_are_still_read_today(self):
        # Codex 10th: the first listing of the day does not show yesterday's hand sale/rebuy yet, a later forced one does
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 0
        st["listed_through"] = "20261124"
        api = _api(self.dir, st, {"005930": {"quantity": 10}})
        answers = [[], [row("8501", "005930", "sell", 4, 4, day="20261124"), row("8502", "005930", "buy", 4, 4, day="20261124")]]
        seen = []

        def fetch(env, cano, prdt, start, end):
            seen.append(start)
            return answers.pop(0)
        with mock.patch.object(r003, "_fetch_ccld_rows", side_effect=fetch):
            self.assertTrue(api._refresh_bot_fills(NOW))
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(hours=1), force=True))
        self.assertEqual(seen, ["20261124", "20261124"])
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 4})
        r023.prune_fill_state(api._fills, DAY)
        self.assertIn(r023.fill_key("20261124", "8501"), api._fills["applied"])

    def test_nxt_sell_reprice_reads_the_bot_part_again_after_the_quote(self):
        def fresh(now, force=False, want=False):
            calls.append(now)
            if len(calls) == 2:
                api._fills["codes"]["005930"]["bot_qty"] = after
            return oks.pop(0)
        for oks, after, expect in (([True, False], 10, None), ([True, True], 0, None), ([True, True], 6, "6")):
            calls, sent = [], []
            api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 10}})
            api.pending_orders["005930"] = {"side": "sell", "quantity": 10, "order_no": "1", "reprice_attempt": 0}
            _fresh_account(api)

            def order(**kw):
                sent.append(kw)
                raise RuntimeError("stop")
            with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), mock.patch.object(api, "_refresh_bot_fills", side_effect=fresh), \
                    mock.patch.object(r003, "get_order_spec", return_value={"exchange": "NXT"}), \
                    mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(1000.0, 1010.0)), \
                    mock.patch.object(api, "_order_cash", side_effect=order):
                try:
                    api._resubmit_repriced_sell_order("005930", api.pending_orders["005930"], NOW)
                except RuntimeError:
                    pass
            self.assertEqual(len(calls), 2)
            if expect:
                self.assertEqual([str(k.get("ord_qty")) for k in sent], [expect])
            else:
                self.assertEqual(sent, [])
                self.assertNotIn("005930", api.pending_orders)

    def test_split_listings_cannot_leave_hand_shares_sellable(self):
        # Codex 12th through r003: the buy of 6 by hand is listed before the earlier hand sale of 4 - sellable ends at 6
        today = NOW.strftime("%Y%m%d")
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 0
        api = _api(self.dir, st, {"005930": {"quantity": 10}})
        answers = [[row("9401", "005930", "sell", 4, 0, day=today), row("9402", "005930", "buy", 6, 6, day=today)],
                   [row("9401", "005930", "sell", 4, 4, day=today), row("9402", "005930", "buy", 6, 6, day=today)]]
        with mock.patch.object(r003, "_fetch_ccld_rows", side_effect=lambda *a: answers.pop(0)):
            self.assertTrue(api._refresh_bot_fills(NOW, force=True))
            self.assertEqual(api._fills["day_anchor"]["codes"]["005930"], {"bot_qty": 10, "manual_qty": 0})
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(seconds=5), force=True))
        api.positions["005930"]["quantity"] = 12
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
        self.assertEqual(api.sellable_bot_quantity("005930"), 6)
        self.assertEqual(r023.load_fill_state(self.dir)[0]["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})

    def test_the_anchor_is_made_after_yesterdays_fills(self):
        # Codex 13th: bot 10 / manual 5; yesterday's hand sale of 5 arrives with today's first listing; today 4 sold by hand,
        # then 6 bought by hand, the buy listed first -> sellable 6
        today = NOW.strftime("%Y%m%d")
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 5
        st["listed_through"] = "20261124"
        api = _api(self.dir, st, {"005930": {"quantity": 12}})
        old = row("9001", "005930", "sell", 5, 5, day="20261124")
        answers = [[old, row("9401", "005930", "sell", 4, 0, day=today), row("9402", "005930", "buy", 6, 6, day=today)],
                   [old, row("9401", "005930", "sell", 4, 4, day=today), row("9402", "005930", "buy", 6, 6, day=today)]]
        with mock.patch.object(r003, "_fetch_ccld_rows", side_effect=lambda *a: answers.pop(0)):
            self.assertTrue(api._refresh_bot_fills(NOW, force=True))
            self.assertEqual(api._fills["day_anchor"]["codes"]["005930"], {"bot_qty": 10, "manual_qty": 0})
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(seconds=5), force=True))
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
        self.assertEqual(api.sellable_bot_quantity("005930"), 6)

    def test_an_unknown_manual_part_asks_for_listings_until_it_is_set(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 10}))
        pos = {"005930": {"quantity": 16}}
        calls = []
        with mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "_fetch_ccld_rows", side_effect=lambda *a: calls.append(a) or []):
            api._update_fill_ledger(pos, NOW, DAY)
            self.assertIsNone(api._fills["codes"]["005930"]["manual_qty"])
            api._update_fill_ledger(pos, NOW + timedelta(seconds=r003.FILL_INIT_CONFIRM_SECONDS + r003.ORDER_HOLD_CHECK_SECONDS + 1), DAY)
        self.assertEqual(len(calls), 2)
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 10, "manual_qty": 6})

    def test_a_manual_part_set_today_gets_its_own_anchor(self):
        # Codex 14th: bot 10, manual unknown at the day's first listing; set to 0 after two looks; then 4 sold and 6 bought by
        # hand, the buy listed 5 s before the sale -> sellable 6
        today = NOW.strftime("%Y%m%d")
        api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 10}})
        early = row("9000", "005930", "buy", 3, 3, day=today)                                # a bot buy already counted today
        r023.register_order(api._fills, today, "9000", "005930", "buy", 3)
        api._fills["codes"]["005930"]["bot_qty"] = 7
        buy, sale0, sale = (row("9402", "005930", "buy", 6, 6, day=today), row("9401", "005930", "sell", 4, 0, rmn=0, day=today),
                            row("9401", "005930", "sell", 4, 4, day=today))
        answers = [[early], [early], [early], [early, sale0, buy], [early, sale, buy], [early, sale, buy]]
        step = r003.FILL_INIT_CONFIRM_SECONDS + r003.ORDER_HOLD_CHECK_SECONDS + 1
        with mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "_fetch_ccld_rows", side_effect=lambda *a: answers.pop(0)):
            api._update_fill_ledger({"005930": {"quantity": 10}}, NOW, today)                 # the bot buy applied: moving
            self.assertIsNone(api._fills["day_anchor"]["codes"]["005930"]["manual_qty"])
            api._update_fill_ledger({"005930": {"quantity": 10}}, NOW + timedelta(seconds=step), today)       # first look
            self.assertIsNone(api._fills["codes"]["005930"]["manual_qty"])
            api._update_fill_ledger({"005930": {"quantity": 10}}, NOW + timedelta(seconds=2 * step), today)   # second: set
            self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 10, "manual_qty": 0})
            self.assertEqual(api._fills["day_anchor"]["codes"]["005930"], {"bot_qty": 10, "manual_qty": 0})
            self.assertEqual(api._fills["day_anchor"]["applied"], {r023.fill_key(today, "9000"): 3})
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(seconds=3 * step), force=True))
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(seconds=3 * step + 5), force=True))
            api.positions = {"005930": {"quantity": 12}}
            api._update_fill_ledger(api.positions, NOW + timedelta(seconds=4 * step), today)
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
        self.assertEqual(api.sellable_bot_quantity("005930"), 6)

    def test_yesterdays_bound_is_applied_once_more_before_todays_anchor(self):
        # Codex 15th: yesterday 4 sold then 6 bought by hand; yesterday's last listing showed only the buy
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 0
        r023.set_day_anchor(st, "20261124", {"005930"})
        r023.apply_fill_rows(st, [row("9402", "005930", "buy", 6, 6, day="20261124")], NOW - timedelta(days=1))
        st["listed_through"] = "20261124"
        api = _api(self.dir, st, {"005930": {"quantity": 12}})
        rows = [row("9401", "005930", "sell", 4, 4, day="20261124"), row("9402", "005930", "buy", 6, 6, day="20261124")]
        with mock.patch.object(r003, "_fetch_ccld_rows", return_value=rows):
            self.assertTrue(api._refresh_bot_fills(NOW, force=True))
            self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
            self.assertEqual(api._fills["day_anchor"]["date"], DAY)
            self.assertEqual(api._fills["day_anchor"]["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
            self.assertTrue(api._refresh_bot_fills(NOW + timedelta(seconds=5), force=True))
        self.assertEqual(api.sellable_bot_quantity("005930"), 6)

    def test_a_sell_waits_while_the_listing_is_behind_the_account(self):
        # Codex 15th: the account already counts 6 sold and 6 bought by hand today (total unchanged), the listing shows nothing
        self.assertEqual(r023.day_fill_totals([row("1", "005930", "buy", 6, 6), row("2", "005930", "sell", 9, 4),
                                               row("3", "005930", "buy", 2, 1), row("4", "000660", "sell", 1, 1, day="20261124")],
                                              DAY), {"005930": (7, 4)})
        today = datetime.now().strftime("%Y%m%d")
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 4
        done = [row("9901", "005930", "sell", 6, 6, day=today), row("9902", "005930", "buy", 6, 6, day=today)]
        for listing, expect in (([], "4"), (done, "8")):                                     # 10 - 6 unlisted sells / bot 8
            for path in ("new", "reprice"):
                api = _api(self.dir, json.loads(json.dumps(st)), {"005930": {"quantity": 14, "buy_price": 1000.0}})
                _fresh_account(api)
                api._acct_today = {"005930": (6, 6)}
                sent = []
                with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), mock.patch.object(api, "_in_cooldown", return_value=False), \
                        mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                        mock.patch.object(r003, "_fetch_ccld_rows", return_value=listing), \
                        mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(1000.0, 1010.0)), \
                        mock.patch.object(api, "_order_cash", side_effect=lambda **kw: sent.append(kw) or None):
                    try:
                        if path == "new":
                            api.place_sell_order("005930", 10, datetime.now(), "test", False)
                        else:
                            api.pending_orders["005930"] = {"side": "sell", "quantity": 10, "order_no": "1", "reprice_attempt": 0}
                            api._resubmit_repriced_sell_order("005930", api.pending_orders["005930"], datetime.now())
                    except Exception:
                        pass                                                                 # past the order call
                self.assertEqual([str(k.get("ord_qty")) for k in sent], [expect], path)
        # Codex 16th: the account could not be read again - no order on old counts; after the limit it goes on with them
        api = _api(self.dir, json.loads(json.dumps(st)), {"005930": {"quantity": 14, "buy_price": 1000.0}})
        api.sync_positions_from_account = lambda force=False: None                           # a failed read: no new counts
        api._acct_today = {"005930": (6, 6)}
        sent = []
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), mock.patch.object(api, "_in_cooldown", return_value=False), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(r003, "_fetch_ccld_rows", return_value=[]), \
                mock.patch.object(api, "_order_cash", side_effect=lambda **kw: sent.append(kw) or None):
            self.assertFalse(api.place_sell_order("005930", 10, datetime.now(), "test", False))
            self.assertEqual(sent, [])
            api._acct_stale_since = datetime.now() - timedelta(seconds=r003.FILL_GAP_MAX_HOLD_SECONDS + 1)
            try:
                api.place_sell_order("005930", 10, datetime.now(), "test", False)
            except Exception:
                pass
        self.assertEqual([str(k.get("ord_qty")) for k in sent], ["4"])
        self.assertEqual(api._unlisted_sell_qty("000660"), 0)                                # no account counts: nothing taken off

    def test_equal_sums_of_different_orders_do_not_pass(self):
        # Codex 18th: bot 10 / manual 6. The account read counts hand sale A (6): 10 held, sold 6. Then hand sale B (6) and a hand
        # buy of 6 fill; the listing shows B and the buy, not A. Really bot 4 / manual 6 - the order may carry 4 at most
        today = datetime.now().strftime("%Y%m%d")
        st = r023.new_fill_state({"005930": 10})
        st["codes"]["005930"]["manual_qty"] = 6
        listing = [row("9912", "005930", "sell", 6, 6, day=today), row("9913", "005930", "buy", 6, 6, day=today)]
        for path in ("new", "reprice"):
            api = _api(self.dir, json.loads(json.dumps(st)), {"005930": {"quantity": 10, "buy_price": 1000.0}})
            _fresh_account(api)
            api._acct_today = {"005930": (0, 6)}
            sent = []
            with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), mock.patch.object(api, "_in_cooldown", return_value=False), \
                    mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                    mock.patch.object(r003, "_fetch_ccld_rows", return_value=listing), \
                    mock.patch.object(r003, "_fetch_bid_ask_price", return_value=(1000.0, 1010.0)), \
                    mock.patch.object(api, "_order_cash", side_effect=lambda **kw: sent.append(kw) or None):
                try:
                    if path == "new":
                        api.place_sell_order("005930", 10, datetime.now(), "test", False)
                    else:
                        api.pending_orders["005930"] = {"side": "sell", "quantity": 10, "order_no": "1", "reprice_attempt": 0}
                        api._resubmit_repriced_sell_order("005930", api.pending_orders["005930"], datetime.now())
                except Exception:
                    pass
            self.assertEqual([str(k.get("ord_qty")) for k in sent], ["4"], path)
            self.assertGreaterEqual(api._acct_seq, 2, path)                                  # the account was read once more

    def test_only_a_real_account_answer_counts_as_a_fresh_read(self):
        # Codex 17th: a failed / empty answer keeps _acct_seq; a real one raises it and ends the "no account" wait
        import pandas as pd
        api = _api(self.dir, r023.new_fill_state(), {"005930": {"quantity": 3, "buy_price": 1000.0}})
        api._last_sync_at = None
        api._acct_stale_since = NOW
        good = pd.DataFrame([{"pdno": "005930", "hldg_qty": "3", "pchs_avg_pric": "1000", "prpr": "1010",
                              "thdt_buyqty": "6", "thdt_sll_qty": "4"}])
        for answer, seq, stale in ((RuntimeError("down"), 0, True), ((pd.DataFrame(), None), 0, True), ((good, None), 1, False)):
            fn = mock.Mock(side_effect=answer) if isinstance(answer, Exception) else mock.Mock(return_value=answer)
            with mock.patch.object(r003.dsf, "inquire_balance_rlz_pl", fn, create=True), \
                    mock.patch.object(api, "_update_bot_ledger"), mock.patch.object(api, "_reconcile_pending_positions"), \
                    mock.patch.object(api, "_apply_persisted_position_meta"), mock.patch.object(api, "_record_position_meta"):
                api.sync_positions_from_account(force=True)
            self.assertEqual((getattr(api, "_acct_seq", 0), hasattr(api, "_acct_stale_since")), (seq, stale))
        self.assertEqual(api._acct_today, {"005930": (6, 4)})

    def test_no_account_comparison_while_account_and_listing_disagree_on_today(self):
        # Codex 16th: bot 10, manual unknown, account 12 with 6 bought / 4 sold today, the listing shows only the buy
        today = NOW.strftime("%Y%m%d")
        api = _api(self.dir, r023.new_fill_state({"005930": 10}), {"005930": {"quantity": 12}})
        api._acct_today = {"005930": (6, 4)}
        buy, sale = row("9402", "005930", "buy", 6, 6, day=today), row("9401", "005930", "sell", 4, 4, day=today)
        step = r003.FILL_INIT_CONFIRM_SECONDS + r003.ORDER_HOLD_CHECK_SECONDS + 1
        answers = [[buy], [buy], [buy], [sale, buy], [sale, buy], [sale, buy]]
        with mock.patch.object(r003.r023, "save_snapshot"), \
                mock.patch.object(r003, "_fetch_ccld_rows", side_effect=lambda *a: answers.pop(0)):
            for i in range(3):
                api._update_fill_ledger(api.positions, NOW + timedelta(seconds=i * step), today)
            self.assertIsNone(api._fills["codes"]["005930"]["manual_qty"])                   # not set while they disagree
            for i in range(3, 6):
                api._update_fill_ledger(api.positions, NOW + timedelta(seconds=i * step), today)
        self.assertEqual(api._fills["codes"]["005930"], {"bot_qty": 6, "manual_qty": 6})
        self.assertEqual(api.sellable_bot_quantity("005930"), 6)

    def test_sell_reprice_needs_the_latest_fills(self):
        api = _api(self.dir, r023.new_fill_state({"005930": 5}), {"005930": {"quantity": 5}})
        api.pending_orders["005930"] = {"side": "sell", "quantity": 5, "order_no": "1", "reprice_attempt": 0}
        _fresh_account(api)
        with mock.patch.object(r003, "AUTO_SELL_SCOPE", "bot_only"), \
                mock.patch.object(api, "_refresh_bot_fills", return_value=False):
            api._resubmit_repriced_sell_order("005930", api.pending_orders["005930"], NOW)
        self.assertNotIn("005930", api.pending_orders)                                      # no resubmit without them

    def test_own_pager_reads_every_page_or_nothing(self):
        def resp(ok, rows, cont):
            body = SimpleNamespace(output1=rows, ctx_area_fk100="f", ctx_area_nk100="n")
            return SimpleNamespace(isOK=lambda: ok, getBody=lambda: body, getHeader=lambda: SimpleNamespace(tr_cont=cont))
        pages = iter([resp(True, [{"odno": "1"}], "M"), resp(True, [{"odno": "2"}], "D")])
        with mock.patch.object(r003.ka, "_url_fetch", side_effect=lambda *a, **k: next(pages)), \
                mock.patch.object(r003.ka, "smart_sleep", create=True):
            self.assertEqual([r["odno"] for r in r003._fetch_ccld_rows("real", "x", "01", DAY, DAY)], ["1", "2"])
        pages = iter([resp(True, [{"odno": "1"}], "M"), resp(False, [], "")])
        with mock.patch.object(r003.ka, "_url_fetch", side_effect=lambda *a, **k: next(pages)), \
                mock.patch.object(r003.ka, "smart_sleep", create=True):
            self.assertIsNone(r003._fetch_ccld_rows("real", "x", "01", DAY, DAY))
        with mock.patch.object(r003.ka, "_url_fetch", return_value=resp(True, [{"odno": "1"}], "M")), \
                mock.patch.object(r003.ka, "smart_sleep", create=True), mock.patch.object(r003, "CCLD_MAX_PAGES", 3):
            self.assertIsNone(r003._fetch_ccld_rows("real", "x", "01", DAY, DAY))

    def test_first_switch_inherits_the_old_ledger_and_a_corrupt_file_sells_nothing(self):
        (self.dir / r023.LEDGER_FILENAME).write_text(json.dumps({"codes": {"005930": {"acct_qty": 10, "bot_qty": 4}}}), encoding="utf-8")
        api = _api(self.dir)
        api.bot_ledger = r023.load_ledger(self.dir)
        api._load_fill_ledger(DAY)
        self.assertEqual(r023.fill_bot_quantity(api._fills, "005930"), 4)
        self.assertTrue((self.dir / f"bot_holdings.pre_fill_{DAY}.json").is_file())
        self.assertEqual(r023.load_fill_state(self.dir)[1], "ok")
        (self.dir / r023.FILL_LEDGER_FILENAME).write_text("{broken", encoding="utf-8")
        api2 = _api(self.dir)
        api2._load_fill_ledger(DAY)
        self.assertTrue(api2._fills_broken)


if __name__ == "__main__":
    unittest.main()
