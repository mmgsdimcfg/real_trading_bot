# -*- coding: utf-8 -*-
"""r027 adaptive exit LIVE manager + r006 _x10 wiring: decide / submitted / reconcile (exact fill record, holding
difference fallback), failed submit retry, unknown submission, manual sell, late buy fill rebase, stale account rows,
restart restore, new entry replaces the old one, re-manage a holding left after DONE, r006 fallbacks."""
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "tests"))

from r024_adaptive_exit import ExitConfig  # noqa: E402
from r027_exit_live import HALT_VALID_S, RESTORE_WAIT_S, UP_CONFIRM_S, LiveExitManager, RETRY_S  # noqa: E402
import r006_sell_conditions as r006  # noqa: E402
from test_r026_exit_shadow import DATE, T0, TickWriter, at  # noqa: E402

GROSS = ExitConfig(pnl_basis="gross")
CODE = "000001"


def pos(qty=100, entry=10000.0, entry_at=T0):
    return {"buy_price": entry, "quantity": qty, "entry_buy_time": entry_at}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ticks = self.tmp / "ticks"
        self.data = self.tmp / "data"
        self.out = self.tmp / "live_runtime"
        self.data.mkdir()
        self.logs: list[str] = []
        self.writer = TickWriter(self.ticks / DATE / f"{CODE}.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mgr(self, cfg=GROSS):
        return LiveExitManager(self.ticks, self.data, self.out, DATE, self.logs.append, cfg, async_profile=False)

    def decide(self, m, sec, px, qty=100, fill=None, in_flight=False, p=None, add_tick=True, rest_px="same", qs=None,
               buy_in_flight=False):
        if add_tick:
            self.writer.add(sec, px)
        rp = float(px) if rest_px == "same" else rest_px
        return m.decide(CODE, "TEST", at(sec), p if p is not None else pos(qty), None, rp, qs or {},
                        sell_fill=fill, sell_in_flight=in_flight, buy_in_flight=buy_in_flight)

    def trades(self):
        p = self.out / f"exit_live_trades_{DATE}.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


class DecideTests(Base):
    def test_open_then_stop_then_fill_record_finishes(self):
        m = self.mgr()
        self.assertIsNone(self.decide(m, 0, 10000))
        self.assertIn(CODE, m.managed_codes())
        o = self.decide(m, 1, 9700)                                  # -3%: below every initial stop
        self.assertIsNotNone(o)
        self.assertEqual((o.kind, o.qty), ("STOP", 100))
        self.assertTrue(o.reason.startswith("ADAPTIVE_"))
        m.submitted(o, ok=True, in_flight=False, now=at(1))
        self.assertIsNone(self.decide(m, 2, 9700, in_flight=True))   # r003 still tracks the order
        fill = {"reason": o.reason, "filled_qty": 100, "at": at(2)}
        self.assertIsNone(self.decide(m, 3, 9700, qty=100, fill=fill))   # account not refreshed yet: record wins
        self.assertNotIn(CODE, m.managed_codes())
        self.assertEqual(self.trades()[-1]["reason"], "SOLD_OUT")
        self.assertEqual(self.trades()[-1]["sold"], 100)

    def test_ladder_steps_follow_exact_fills(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)                                 # +2% -> TP1 (40 % of 100)
        self.assertEqual((o.kind, o.qty), ("LADDER", 40))
        m.submitted(o, True, False, at(1))
        fill = {"reason": o.reason, "filled_qty": 40, "at": at(2)}
        self.assertIsNone(self.decide(m, 4, 10200, qty=60, fill=fill))
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.sold_qty, st.ladder_step, st.pending_qty), (60, 40, 1, 0))
        o2 = self.decide(m, 6, 10300, qty=60, fill=fill)             # +3% -> TP2 (cumulative 70 -> 30 more)
        self.assertEqual((o2.kind, o2.qty), ("LADDER", 30))

    def test_unfilled_order_reverts_stage_and_waits(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        self.assertIsNone(self.decide(m, 2, 10200))                  # no record, holding unchanged -> 0 filled
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.ladder_step, st.pending_qty, st.held_qty), (0, 0, 100))
        self.assertIsNone(self.decide(m, 3, 10200))                  # retry wait
        o2 = self.decide(m, 2 + RETRY_S + 1, 10200)
        self.assertEqual((o2.kind, o2.qty), ("LADDER", 40))

    def test_holding_difference_used_without_record(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        self.decide(m, 3, 10200, qty=60)
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.sold_qty, st.ladder_step), (60, 40, 1))

    def test_failed_submit_retries_after_wait(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 9700)
        m.submitted(o, ok=False, in_flight=False, now=at(1))
        self.assertIsNone(self.decide(m, 2, 9700))
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual(st.pending_qty, 0)
        o2 = self.decide(m, 1 + RETRY_S + 0.5, 9700)
        self.assertEqual(o2.kind, "STOP")
        self.assertTrue(any("submit failed" in x for x in self.logs))

    def test_unknown_submission_counts_as_sent(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 9700)
        m.submitted(o, ok=False, in_flight=True, now=at(1))
        self.assertEqual(m.positions[m.by_code[CODE]].state.pending_qty, 100)
        self.decide(m, 5, 9700, qty=0 + 100, fill={"reason": o.reason, "filled_qty": 100, "at": at(4)})
        self.assertNotIn(CODE, m.managed_codes())

    def test_old_fill_record_with_same_reason_is_ignored(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 10, 9700)
        m.submitted(o, True, False, at(10))
        stale = {"reason": o.reason, "filled_qty": 100, "at": at(1)}   # older than the order
        self.decide(m, 12, 9700, qty=100, fill=stale)
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.pending_qty), (100, 0))     # fell back to the holding: nothing filled


class HoldingTests(Base):
    def test_manual_sell_syncs_down(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        self.decide(m, 1, 10050, qty=30)
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.sold_qty), (30, 70))
        self.assertTrue(any("without an engine order" in x for x in self.logs))

    def test_late_buy_fill_rebases_before_any_sell(self):
        m = self.mgr()
        self.decide(m, 0, 10000, qty=40)
        self.decide(m, 1, 10000, p=pos(100, 10010.0))
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.q0, st.entry_price), (100, 100, 10010.0))

    def test_increase_after_a_sell_waits_then_rebases(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        self.decide(m, 2, 10200, qty=100, fill={"reason": o.reason, "filled_qty": 40, "at": at(2)})
        self.decide(m, 3, 10200, qty=100)                            # account still shows 100 (stale row)
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.q0), (60, 100))
        self.decide(m, 4, 10200, qty=60)                              # account caught up: the wait resets
        self.decide(m, 10, 10200, qty=144)                            # late buy fill after the sell (Codex live #4)
        self.decide(m, 10 + UP_CONFIRM_S - 1, 10200, qty=144)
        self.assertEqual(m.positions[m.by_code[CODE]].state.held_qty, 60)
        self.decide(m, 10 + UP_CONFIRM_S + 1, 10200, qty=144)
        self.assertEqual(m.positions[m.by_code[CODE]].state.held_qty, 144)

    def test_increase_with_a_buy_in_flight_rebases_at_once(self):
        m = self.mgr()
        self.decide(m, 0, 10000, qty=40)
        o = self.decide(m, 1, 10200, qty=40)
        m.submitted(o, True, False, at(1))
        self.decide(m, 2, 10200, qty=24, fill={"reason": o.reason, "filled_qty": 16, "at": at(2)})
        self.decide(m, 3, 10150, qty=84, buy_in_flight=True)
        st = m.positions[m.by_code[CODE]].state
        self.assertEqual(st.held_qty, 84)

    def test_new_entry_replaces_old_and_restart_restores(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        self.decide(m, 2, 10200, qty=60, fill={"reason": o.reason, "filled_qty": 40, "at": at(2)})
        m2 = self.mgr()                                              # restart
        st = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.ladder_step), (60, 1))
        self.decide(m2, 5, 10100, p=pos(50, 10100.0, at(4)))         # re-buy: new entry time
        self.assertEqual(self.trades()[-1]["reason"], "REPLACED_BY_NEW_ENTRY")
        st2 = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st2.q0, st2.entry_price), (50, 10100.0))

    def test_restart_with_order_in_flight_waits_for_the_holding(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)                                  # TP1 40 sh
        m.submitted(o, True, False, at(1))
        m2 = self.mgr()                                               # restart: r003 forgot the pending order
        self.assertIsNone(self.decide(m2, 3, 10200, qty=100))          # holding unchanged: wait, no resend
        self.assertIsNone(self.decide(m2, 20, 10200, qty=100))
        st = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st.pending_qty, st.ladder_step), (40, 1))
        self.decide(m2, 25, 10200, qty=60)                             # the fill shows up
        st = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.sold_qty, st.pending_qty, st.ladder_step), (60, 40, 0, 1))

    def test_restart_wait_allows_only_a_stop_for_uncovered_shares(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        m2 = self.mgr()
        self.decide(m2, 3, 10200, qty=100)
        o2 = self.decide(m2, 5, 9700, qty=100)                         # crash below the stop while waiting
        self.assertEqual((o2.kind, o2.qty), ("STOP", 60))

    def test_restart_partial_fill_keeps_the_rest_in_flight(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)                                  # TP1 40 sh
        m.submitted(o, True, False, at(1))
        m2 = self.mgr()
        self.decide(m2, 3, 10200, qty=100)
        self.decide(m2, 4, 10200, qty=90)                              # 10 of 40 filled so far
        st = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.pending_qty, st.sold_qty), (90, 30, 10))
        o2 = self.decide(m2, 5, 9700, qty=90)                          # stop: only the 60 not covered by the 30 in flight
        self.assertEqual((o2.kind, o2.qty), ("STOP", 60))

    def test_first_sight_with_stale_rest_opens_on_ticks_and_force_exits(self):
        m = self.mgr()
        self.writer.add(0, 10000)
        force_at = (datetime(2026, 11, 16, 15, 16, 0) - T0).total_seconds()
        o = self.decide(m, force_at, 10050, rest_px=None)               # no checkpoint, REST stale, ticks fresh
        self.assertIn(CODE, m.managed_codes())
        self.assertIsNotNone(o)
        self.assertEqual((o.kind, o.qty), ("FORCE", 100))

    def test_restart_partial_fill_then_deadline_keeps_the_stage(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        m2 = self.mgr()
        self.decide(m2, 3, 10200, qty=100)
        self.decide(m2, 4, 10200, qty=90)                              # 10 filled
        self.decide(m2, 4 + RESTORE_WAIT_S + 1, 10200, qty=90)         # the rest never filled
        st = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st.held_qty, st.sold_qty, st.pending_qty, st.ladder_step), (90, 10, 0, 1))

    def test_first_sight_with_stale_rest_stops_on_ticks(self):
        m = self.mgr()
        self.writer.add(0, 9700)
        o = self.decide(m, 1, 9700, rest_px=None)                      # first look: REST stale, ticks below the stop
        self.assertIn(CODE, m.managed_codes())
        if o is None:                                                  # opened on this call; the next one decides
            o = self.decide(m, 2, 9700, rest_px=None)
        self.assertEqual((o.kind, o.qty), ("STOP", 100))

    def test_first_sight_without_any_price_waits(self):
        m = LiveExitManager(self.ticks, self.data, self.out, "20261117", self.logs.append, GROSS, async_profile=False)
        self.assertIsNone(m.decide(CODE, "TEST", at(0), pos(), None, None, {}))   # no tick file for that day, REST stale
        self.assertNotIn(CODE, m.managed_codes())

    def test_restart_wait_times_out_as_unfilled(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 10200)
        m.submitted(o, True, False, at(1))
        m2 = self.mgr()
        self.decide(m2, 3, 10200, qty=100)
        self.decide(m2, 3 + RESTORE_WAIT_S + 1, 10200, qty=100)
        st = m2.positions[m2.by_code[CODE]].state
        self.assertEqual((st.pending_qty, st.ladder_step, st.held_qty), (0, 0, 100))   # reverted, may fire again

    def test_crash_between_decide_and_submitted_is_treated_as_in_flight(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 9700)                                    # decide persisted a "submitting" marker
        self.assertIsNotNone(o)
        m2 = self.mgr()                                               # crash before submitted()
        self.assertIsNone(self.decide(m2, 2, 9700, qty=100))           # wait (the order may be at the broker)
        self.assertEqual(m2.positions[m2.by_code[CODE]].state.pending_qty, 100)

    def test_stale_rest_price_still_force_exits_on_ticks(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        force_at = (datetime(2026, 11, 16, 15, 15, 1) - T0).total_seconds()
        o = self.decide(m, force_at, 10050, rest_px=None)               # REST stale: price=None, ticks fresh
        self.assertIsNotNone(o)
        self.assertEqual(o.kind, "FORCE")

    def test_halt_flag_expires_by_its_own_quote_time(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        qs = {"temp_stop_yn": "Y", "at": at(1)}
        self.assertIsNone(self.decide(m, 1, 9700, qs=qs))               # fresh halt: frozen, no stop
        o = self.decide(m, 1 + HALT_VALID_S + 5, 9700, qs=qs)           # same cached flag, quote not refreshed
        self.assertEqual(o.kind, "STOP")

    def test_holding_left_after_done_is_managed_again_later(self):
        m = self.mgr()
        self.decide(m, 0, 10000)
        o = self.decide(m, 1, 9700)
        m.submitted(o, True, False, at(1))
        self.decide(m, 2, 9700, qty=100, fill={"reason": o.reason, "filled_qty": 100, "at": at(2)})
        self.assertIsNone(self.decide(m, 10, 9700, qty=100))        # within REOPEN_AFTER_CLOSE_S
        self.assertNotIn(CODE, m.managed_codes())
        self.decide(m, 40, 9700, qty=100)
        self.assertIn(CODE, m.managed_codes())


class _Api:
    def __init__(self, ok=True, bot_only=False):
        self.ok, self.orders, self.pending, self.bot_only = ok, [], None, bot_only
        self.last_sell_confirm = {}
        self.startup_position_codes = set()

    def place_sell_order(self, code, qty, now, reason, nxt, price=None, code_name="", market_order=False, limit_price=None):
        self.orders.append((qty, reason, market_order))
        return self.ok

    def get_pending_order(self, code):
        return self.pending

    def sells_bot_quantity_only(self):
        return self.bot_only

    def bot_quantity(self, code):
        return 0

    def persist_live_state(self, date_str=None):
        pass


def sell_ctx(price, api, mgr, regular=True, now=None, source="live"):
    now = now or at(1)
    logs: list[str] = []
    svc = SimpleNamespace(exit_live=mgr, is_regular_session=lambda _d: regular, get_quote_status=lambda _c: {},
                          log=logs.append, is_stale_live_price_source=lambda s: s.startswith("bar_close"))
    c = SimpleNamespace(code=CODE, name="TEST", symbol_label="000001_TEST", current_dt=now, nxt_tradeable=False,
                        pos=pos(), price=price, price_source=source, api=api, date_str=DATE, log=logs.append,
                        entry_price=10000.0,
                        pnl_pct=price / 10000.0 - 1, bar_time=None, services=svc,
                        state=SimpleNamespace(signal_sell_bar={}, hard_stop_today_codes=set()),
                        risk=SimpleNamespace(hard_stop_daily_count=0))
    return c, logs


class R006Tests(Base):
    def test_mode_selection(self):
        with mock.patch.object(r006, "AUTO_SELL_MODE", "all"):
            with mock.patch.object(r006, "EXIT_ENGINE_MODE", "live"):
                self.assertIs(r006.active_sell_conditions(), r006.ADAPTIVE_SELL_CONDITIONS)
            with mock.patch.object(r006, "EXIT_ENGINE_MODE", "shadow"):
                self.assertIs(r006.active_sell_conditions(), r006.SELL_CONDITIONS)
        with mock.patch.object(r006, "EXIT_ENGINE_MODE", "live"):
            with mock.patch.object(r006, "AUTO_SELL_MODE", "stop_only"):
                self.assertIs(r006.active_sell_conditions(), r006.STOP_ONLY_SELL_CONDITIONS)
            with mock.patch.object(r006, "AUTO_SELL_MODE", "off"):
                self.assertIs(r006.active_sell_conditions(), r006.MANUAL_SELL_CONDITIONS)
        with mock.patch.object(r006, "AUTO_SELL_MODE", "all"), mock.patch.object(r006, "EXIT_ENGINE_MODE", "live"):
            for strategy, expected in (("normal", r006.ADAPTIVE_SELL_CONDITIONS), ("basic", r006.ADAPTIVE_SELL_CONDITIONS),
                                       ("edge", r006.EDGE_SELL_CONDITIONS)):   # edge keeps its model-label exits
                with mock.patch.object(r006, "TRADING_STRATEGY_MODE", strategy):
                    self.assertIs(r006.active_sell_conditions(), expected)
        self.assertEqual([c.func_name for c in r006.ADAPTIVE_SELL_CONDITIONS],
                         ["_x02_track_if_fresh", "_x10_adaptive_exit"])

    def test_stop_sends_market_order_and_registers_hard_stop(self):
        m, api = self.mgr(), _Api()
        self.writer.add(0, 10000)
        c, _ = sell_ctx(10000.0, api, m, now=at(0))
        self.assertFalse(r006._x10_adaptive_exit(c))
        self.writer.add(1, 9700)
        c, _ = sell_ctx(9700.0, api, m)
        self.assertTrue(r006._x10_adaptive_exit(c))
        self.assertEqual(len(api.orders), 1)
        qty, reason, market = api.orders[0]
        self.assertEqual((qty, market), (100, True))
        self.assertTrue(reason.startswith("ADAPTIVE_"))
        self.assertIn(CODE, c.state.hard_stop_today_codes)
        self.assertEqual(m.positions[m.by_code[CODE]].state.pending_qty, 100)

    def test_fallbacks_to_net_stop(self):
        api = _Api()
        c, _ = sell_ctx(9700.0, api, None)                           # no manager
        self.assertTrue(r006._x10_adaptive_exit(c))
        self.assertEqual(api.orders[-1][1], "NET_STOP_LOSS_2.0PCT")
        m = self.mgr()
        c, _ = sell_ctx(9700.0, api, m, regular=False)               # outside the regular session
        self.assertTrue(r006._x10_adaptive_exit(c))
        self.assertEqual(api.orders[-1][1], "NET_STOP_LOSS_2.0PCT")
        with mock.patch.object(m, "_decide", side_effect=ValueError("boom")):
            c, logs = sell_ctx(9700.0, api, m)
            self.assertTrue(r006._x10_adaptive_exit(c))
            self.assertEqual(api.orders[-1][1], "NET_STOP_LOSS_2.0PCT")
            self.assertTrue(any("fallback" in x for x in logs))

    def test_stale_price_fallback_skips_and_engine_gets_none(self):
        api = _Api()
        c, _ = sell_ctx(9700.0, api, None, source="bar_close(stale_live=99s)")   # no manager + stale: like _001
        self.assertTrue(r006._x10_adaptive_exit(c))
        self.assertEqual(api.orders, [])
        m = self.mgr()
        self.writer.add(0, 10000)
        c, _ = sell_ctx(10000.0, api, m, now=at(0))
        r006._x10_adaptive_exit(c)
        with mock.patch.object(m, "decide", wraps=m.decide) as spy:
            self.writer.add(1, 10000)
            c, _ = sell_ctx(9000.0, api, m, source="bar_close(stale_live=99s)")
            r006._x10_adaptive_exit(c)
            self.assertIsNone(spy.call_args.args[5])                     # stale REST price is not fed to the engine
        self.assertEqual(api.orders, [])                                # ticks at 10,000: no stop from a stale 9,000

    def test_pending_sell_means_no_decision(self):
        m, api = self.mgr(), _Api()
        api.pending = {"side": "sell"}
        self.writer.add(0, 9700)
        c, _ = sell_ctx(9700.0, api, m, now=at(0))
        self.assertFalse(r006._x10_adaptive_exit(c))
        self.assertEqual(api.orders, [])


if __name__ == "__main__":
    unittest.main()
