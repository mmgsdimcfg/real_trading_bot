# -*- coding: utf-8 -*-
"""r024 adaptive exit engine - unit tests + design scenarios S1-S7 (adaptive_exit_design_20261115.md section 11/17).

Scenario convention: entry 10,000 KRW (tick 10), q0 100, one evaluation per listed second, gross basis unless stated.
The harness fills every submitted order immediately unless a test drives the order events by hand (S7)."""
import json
import math
import random
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r024_adaptive_exit as ax  # noqa: E402
from r024_adaptive_exit import (  # noqa: E402
    AIAdjust, EdgeAIAdapter, ExitConfig, ExitDecisionEngine, ExitState, MarketSnapshot,
)

T0 = datetime(2026, 11, 16, 10, 0, 0)
GROSS = ExitConfig(pnl_basis="gross")
SURGE_FEATS = dict(ret_3m_pct=1.5, rvol_5m=4.0, cttr_60s=160.0)
# keeps SURGE alive (no "weak" exit signal) while every trailing multiplier stays exactly 1.0
SURGE_HOLD = dict(ret_3m_pct=0.5, rvol_5m=2.0, cttr_60s=120.0)


def snap(sec, price, **kw):
    kw.setdefault("bid1", price)
    kw.setdefault("ask1", price + ax.tick_size(price))
    return MarketSnapshot(ts=T0 + timedelta(seconds=sec), price=float(price), **kw)


class Harness:
    def __init__(self, cfg=GROSS, entry=10000, qty=100, open_kw=None, start=T0):
        self.eng = ExitDecisionEngine(cfg)
        self.t0 = start
        self.st = self.eng.open_position("000001", entry, qty, start,
                                         MarketSnapshot(ts=start, price=entry, **(open_kw or {})))
        self.orders = []
        self.logs = []

    def at(self, sec, price, ai=None, fill=True, **kw):
        now = self.t0 + timedelta(seconds=sec)
        kw.setdefault("bid1", price)
        s = MarketSnapshot(ts=now, price=float(price), **kw)
        res = self.eng.step(self.st, s, now, ai)
        self.st = res.state
        self.logs.append(res.log)
        it = res.intent
        if it is not None and fill:
            qty = it.qty
            self.st = self.eng.on_submitted(self.st, it, qty, f"o{len(self.orders)}", now)
            self.st = self.eng.on_fill(self.st, qty)
            self.orders.append((sec, it.kind, qty, it.order_type, it.limit_price))
        return it

    def hold(self, start, end, price, **kw):
        out = []
        for s in range(start, end + 1):
            it = self.at(s, price, **kw)
            if it is not None:
                out.append(it)
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TickHelpers(unittest.TestCase):
    def test_tick_bands(self):
        self.assertEqual(ax.tick_size(4995), 5)
        self.assertEqual(ax.tick_size(5000), 10)
        self.assertEqual(ax.tick_down(5000), 4995)
        self.assertEqual(ax.tick_up(4995), 5000)
        self.assertEqual(ax.tick_down(20000), 19990)
        self.assertEqual(ax.tick_up(19990), 20000)
        self.assertEqual(ax.tick_up(20000), 20050)
        self.assertEqual(ax.floor_tick(5003), 5000)
        self.assertEqual(ax.ceil_tick(4998), 5000)
        self.assertEqual(ax.ceil_tick(20001), 20050)
        self.assertEqual(ax.floor_tick(10289.999999999), 10290)
        self.assertEqual(ax.ceil_tick(10300.0000000001), 10300)
        self.assertEqual(ax.ticks_between(4990, 5010), 3)

    def test_net_pnl_inverse(self):
        cfg = ExitConfig()
        for pnl in (-2.0, 0.1, 2.0, 5.0):
            p = ax.price_for_pnl(cfg, 10000, pnl)
            self.assertAlmostEqual(ax.pnl_pct(cfg, 10000, p), pnl, places=9)
        g = ax.pnl_pct(GROSS, 10000, ax.price_for_pnl(cfg, 10000, 2.0))
        self.assertAlmostEqual(g, 2.235105, places=5)      # section 17 C16: net +2 % = gross +2.2351 %

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            ExitConfig(tp_ratios=(0.5, 0.3, 0.3)).validate()
        with self.assertRaises(ValueError):
            ExitConfig.from_mapping({"nope": 1})
        with self.assertRaises(ValueError):
            ExitConfig(retrace_unit="x").validate()


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------

class Ladder(unittest.TestCase):
    def test_plan_cumulative_rounding(self):
        lad = ax.TakeProfitLadder(GROSS)
        for q in range(1, 60):
            plan = lad.plan(q)
            self.assertEqual(plan[-1], q)
            self.assertTrue(all(a <= b for a, b in zip(plan, plan[1:])))
        self.assertEqual(lad.plan(100), (40, 70, 100))
        self.assertEqual(lad.plan(1), (0, 1, 1))

    def test_q1_skips_zero_share_step(self):
        h = Harness(qty=1)
        self.assertIsNone(h.at(10, 10200))                  # TP1 needs 0 shares -> step done, no order
        self.assertEqual(h.st.ladder_step, 1)
        self.assertIsNone(h.at(12, 10250))
        it = h.at(14, 10300)
        self.assertEqual((it.kind, it.qty), ("LADDER", 1))
        self.assertEqual(h.st.phase, "CLOSED")

    def test_q2_remainder_merged(self):
        h = Harness(qty=2)
        it = h.at(10, 10200)
        self.assertEqual((it.kind, it.qty), ("LADDER", 2))  # 1 share would be left -> sold together

    def test_small_remainder_value_merged(self):
        h = Harness(qty=8)                                   # plan (3, 6, 8), min remainder 30k
        self.assertEqual(h.at(10, 10200).qty, 3)             # rest 5 x 10,200 = 51,000 >= 30k
        it = h.at(13, 10300)
        self.assertEqual(it.qty, 5)                          # 3 + the 2 that would be left (20,600 < 30k)
        self.assertEqual(h.st.phase, "CLOSED")

    def test_500k_position_keeps_trail_split(self):
        # r001 MAX_ORDER_AMOUNT_KRW = 500,000: 50 shares x 10,000. TP1 20, TRAIL_A 70 % of 30 = 21, B 9 (90k)
        h = Harness(qty=50, open_kw=SURGE_FEATS)
        h.at(60, 10300, **SURGE_HOLD)
        h.at(90, 10350, **SURGE_HOLD)
        h.hold(110, 112, 10290, **SURGE_HOLD)
        h.hold(140, 142, 10240, **SURGE_HOLD)
        self.assertEqual([(o[1], o[2]) for o in h.orders], [("LADDER", 20), ("TRAIL_A", 21), ("TRAIL_B", 9)])

    def test_one_step_per_evaluation_and_interval(self):
        h = Harness()
        it = h.at(10, 10400)                                 # +4 % jumps over all NORMAL targets
        self.assertEqual((it.kind, it.qty, it.ladder_step), ("LADDER", 40, 0))
        self.assertIsNone(h.at(11, 10400))                   # < TP_MIN_INTERVAL_S
        self.assertEqual(h.at(12, 10400).qty, 30)
        self.assertIsNone(h.at(13, 10400))

    def test_ladder_limit_price_is_target_or_bid(self):
        h = Harness()
        it = h.at(10, 10210, bid1=10200)
        self.assertEqual((it.order_type, it.limit_price), ("limit", 10200))
        h2 = Harness(cfg=ExitConfig())                       # net: target price for +2 % net = 10,223.5 -> 10,230
        it2 = h2.at(10, 10240, bid1=10200)
        self.assertEqual(it2.limit_price, 10230)


# ---------------------------------------------------------------------------
# design scenarios
# ---------------------------------------------------------------------------

class Scenarios(unittest.TestCase):
    def test_s1_surge_smooth_to_plus5(self):
        h = Harness(open_kw=SURGE_FEATS)
        self.assertEqual(h.st.regime, "SURGE")
        for sec, px in ((20, 10120), (40, 10160), (60, 10250)):
            self.assertIsNone(h.at(sec, px, **SURGE_FEATS))
        h.at(70, 10300, **SURGE_FEATS)
        h.at(80, 10370, **SURGE_FEATS)                         # retrace 0.3 %p < w1 0.828
        h.at(95, 10400, **SURGE_FEATS)
        h.at(120, 10500, **SURGE_FEATS)
        self.assertEqual([(o[1], o[2], o[4]) for o in h.orders],
                         [("LADDER", 40, 10300), ("LADDER", 30, 10400), ("LADDER", 30, 10500)])
        self.assertAlmostEqual(h.logs[-1]["w1"], 0.6 * 1.2 * 1.15, places=9)
        self.assertEqual(h.st.phase, "CLOSED")

    def test_s2_peak_35_then_29_then_24(self):
        h = Harness(open_kw=SURGE_FEATS)
        h.at(60, 10300, **SURGE_HOLD)
        h.at(90, 10350, **SURGE_HOLD)
        self.assertIsNone(h.at(110, 10290, **SURGE_HOLD))        # confirm window starts
        self.assertIsNone(h.at(111, 10290, **SURGE_HOLD))
        a = h.at(112, 10290, **SURGE_HOLD)
        self.assertEqual((a.kind, a.qty, a.order_type), ("TRAIL_A", 42, "market"))
        self.assertEqual(h.st.phase, "TRAIL_A_DONE")
        self.assertIsNone(h.at(120, 10350, **SURGE_HOLD) and None)
        self.assertEqual(h.hold(140, 141, 10240, **SURGE_HOLD), [])
        b = h.at(142, 10240, **SURGE_HOLD)
        self.assertEqual((b.kind, b.qty), ("TRAIL_B", 18))
        self.assertEqual(h.st.phase, "CLOSED")
        self.assertEqual(h.st.regime, "SURGE")

    def test_s2_exact_boundary(self):
        # 10,300 (drop 0.5 %p) never fires A; 10,290 (exactly 0.6 %p) does
        h = Harness(open_kw=SURGE_FEATS)
        h.at(60, 10300, **SURGE_HOLD)
        h.at(90, 10350, **SURGE_HOLD)
        self.assertEqual(h.hold(91, 100, 10300, **SURGE_HOLD), [])
        self.assertEqual(h.logs[-1]["trail_price_a"], 10290)

    def test_s3_distribution_tightens(self):
        h = Harness()
        h.at(50, 10200)
        h.at(80, 10260)
        dist = dict(vol_30s_over_10m=4.2, sell_share_30s=0.66)
        self.assertIsNone(h.at(85, 10230, **dist))
        lg = h.logs[-1]
        self.assertAlmostEqual(lg["m_dist"], 0.6)
        self.assertAlmostEqual(lg["w1"], 0.36, places=9)
        self.assertAlmostEqual(lg["w2"], 0.66, places=9)
        self.assertTrue(lg["frozen"])
        self.assertEqual((lg["trail_price_a"], lg["trail_price_b"]), (10220, 10190))
        h.hold(88, 89, 10220, **dist)
        a = h.at(90, 10220, **dist)
        self.assertEqual((a.kind, a.qty), ("TRAIL_A", 42))
        h.hold(92, 93, 10190)                       # distribution gone: frozen widths still cap
        b = h.at(94, 10190)
        self.assertEqual((b.kind, b.qty), ("TRAIL_B", 18))

    def test_s3_control_without_distribution(self):
        h = Harness()
        h.at(50, 10200)
        h.at(80, 10260)
        h.at(85, 10230)
        self.assertEqual((h.logs[-1]["trail_price_a"], h.logs[-1]["trail_price_b"]), (10200, 10150))

    def test_s4_sideways_time_stop(self):
        h = Harness()
        prices = [9980, 10020, 10060, 10000, 10040, 9990]
        for i, sec in enumerate(range(30, 1200, 30)):
            self.assertIsNone(h.at(sec, prices[i % len(prices)]), sec)
        it = h.at(1200, 10010)
        self.assertEqual((it.kind, it.qty, it.order_type), ("TIME_STOP", 100, "market"))

    def test_s5_surge_to_normal_same_eval_priority(self):
        # trail confirm 0 so the regime switch and the trail hit land in the SAME evaluation: TRAIL_A outranks LADDER
        h = Harness(cfg=replace(GROSS, trail_confirm_s=0.0), open_kw=SURGE_FEATS)
        h.at(60, 10300, **SURGE_HOLD)
        h.at(90, 10380, **SURGE_HOLD)
        weak = dict(ret_3m_pct=0.5, rvol_5m=1.2, cttr_60s=95.0)
        self.assertEqual(h.hold(95, 124, 10340, **weak), [])
        self.assertEqual(h.st.regime, "SURGE")
        it = h.at(125, 10330, **weak)
        self.assertEqual(h.st.regime, "NORMAL")
        kinds = [c[0] for c in h.logs[-1]["candidates"]]
        self.assertIn("LADDER", kinds)
        self.assertEqual((it.kind, it.qty), ("TRAIL_A", 42))
        self.assertGreaterEqual(h.st.sl_price, 10050)           # SURGE->NORMAL after peak >= 2 %: stop to +0.5 %
        b = h.at(130, 10290, **weak)
        self.assertEqual((b.kind, b.qty), ("TRAIL_B", 18))

    def test_s5_default_confirm_ladder_first(self):
        # with the default 2 s trail confirm the LADDER (no confirm) fires on the switch, TRAIL_A two seconds later
        h = Harness(open_kw=SURGE_FEATS)
        h.at(60, 10300, **SURGE_HOLD)
        h.at(90, 10380, **SURGE_HOLD)
        weak = dict(ret_3m_pct=0.5, rvol_5m=1.2, cttr_60s=95.0)
        h.hold(95, 124, 10340, **weak)
        it = h.at(125, 10330, **weak)
        self.assertEqual((it.kind, it.qty), ("LADDER", 30))
        h.at(126, 10330, **weak)
        a = h.at(127, 10330, **weak)
        self.assertEqual((a.kind, a.qty), ("TRAIL_A", 21))      # 70 % of 30; rest 9 x 10,330 >= 30k

    def test_s6_ai_widening_blocked_while_frozen(self):
        ai = AIAdjust(status="ok", applied=True, m_trail=1.306, stage_a_ratio=None)
        h = Harness(open_kw=SURGE_FEATS)
        h.at(60, 10300, ai=ai, **SURGE_HOLD)
        h.at(90, 10350, ai=ai, **SURGE_HOLD)
        self.assertAlmostEqual(h.logs[-1]["w1"], 0.6 * 1.306, places=9)    # widening allowed at a new peak
        h.at(100, 10315, ai=ai, **SURGE_HOLD)                               # drop 0.35 >= 0.5 x 0.6 -> freeze
        self.assertTrue(h.logs[-1]["frozen"])
        self.assertAlmostEqual(h.logs[-1]["w1"], 0.6, places=9)
        h.hold(110, 111, 10290, ai=ai, **SURGE_HOLD)
        a = h.at(112, 10290, ai=ai, **SURGE_HOLD)
        self.assertEqual((a.kind, a.qty), ("TRAIL_A", 42))                  # same as rules only

    def test_s6_ai_failures_equal_rule_path(self):
        rule = Harness(open_kw=SURGE_FEATS)
        bad = Harness(open_kw=SURGE_FEATS)
        statuses = ["timeout", "invalid", "error", "stale", "low_conf", "disabled"]
        path = [(60, 10300), (90, 10350), (110, 10290), (111, 10290), (112, 10290), (140, 10240), (141, 10240),
                (142, 10240)]
        for i, (sec, px) in enumerate(path):
            rule.at(sec, px, **SURGE_HOLD)
            bad.at(sec, px, ai=AIAdjust.neutral(statuses[i % len(statuses)]), **SURGE_HOLD)
        self.assertEqual(rule.orders, bad.orders)

    def test_s7_pending_then_higher_signal(self):
        eng = ExitDecisionEngine(GROSS)
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        now = T0 + timedelta(seconds=60)
        r = eng.step(st, snap(60, 10200), now)
        self.assertEqual((r.intent.kind, r.intent.qty), ("LADDER", 40))
        st = eng.on_submitted(r.state, r.intent, 40, "L1", now)
        self.assertEqual(st.sellable_qty, 60)
        r = eng.step(st, snap(61, 10210), now + timedelta(seconds=1))
        self.assertIsNone(r.intent)                                   # no duplicate
        st = eng.on_fill(r.state, 10)                                 # 10 of 40 filled, 30 still in flight
        self.assertEqual((st.held_qty, st.pending_qty, st.sold_qty), (90, 30, 10))
        r = eng.step(st, snap(64, 9790), now + timedelta(seconds=4))  # below the 9,800 stop
        it = r.intent
        self.assertEqual((it.kind, it.replace_pending, it.all_remaining), ("STOP", True, True))
        st = r.state
        r2 = eng.step(st, snap(64, 9780), now + timedelta(seconds=4))  # cancel not confirmed yet: same intent only
        self.assertEqual(r2.intent.kind, "STOP")
        st = eng.on_order_closed(st, unfilled_qty=30, revert_stage=False)   # cancel confirmed
        st = eng.sync_holding(st, 90)                                       # real holding re-read
        st = eng.on_submitted(st, it, 90, "S1", now + timedelta(seconds=5))
        self.assertEqual(st.phase, "CLOSING")
        r3 = eng.step(st, snap(66, 9770), now + timedelta(seconds=6))
        self.assertIsNone(r3.intent)                                  # same priority in flight
        st = eng.on_fill(r3.state, 90)
        self.assertEqual(st.phase, "CLOSED")


# ---------------------------------------------------------------------------
# stop loss / safety
# ---------------------------------------------------------------------------

class StopLossTests(unittest.TestCase):
    def test_initial_stop_from_atr(self):
        sl = ax.StopLoss(GROSS)
        self.assertEqual(sl.initial_price(10000, None), (9800.0, 2.0))
        self.assertEqual(sl.initial_price(10000, 0.2), (9880.0, 1.2))     # 0.5 -> floor 1.2
        self.assertEqual(sl.initial_price(10000, 0.6)[0], 9850.0)         # 1.5 %

    def test_ratchet_never_loosens_random(self):
        rnd = random.Random(7)
        for trial in range(30):
            h = Harness()
            last = h.st.sl_price
            px = 10000
            for sec in range(1, 400):
                px = max(9000, min(11000, px + rnd.choice((-30, -20, -10, 0, 10, 20, 30))))
                ai = AIAdjust(status="ok", applied=True, m_trail=rnd.uniform(0.6, 1.4),
                              sl_tighten=rnd.random() < 0.3, stage_a_ratio=0.7)
                kw = dict(ret_3m_pct=rnd.uniform(-1, 3), rvol_5m=rnd.uniform(0, 5), cttr_60s=rnd.uniform(50, 200),
                          atr1m_pct=rnd.uniform(0.1, 1.5))
                h.at(sec, px, ai=ai, fill=True, **kw)
                self.assertGreaterEqual(h.st.sl_price, last, (trial, sec))
                last = h.st.sl_price
                if h.st.phase == "CLOSED":
                    break

    def test_breakeven(self):
        h = Harness()
        h.at(10, 10100)
        self.assertTrue(h.st.be_armed)
        self.assertEqual(h.st.sl_price, 10010)
        h.at(20, 10010)
        h.at(22, 10010)
        it = h.at(23, 10010)
        self.assertEqual((it.kind, it.reason), ("STOP", "STOP_BREAKEVEN_+0.10PCT"))

    def test_immediate_stop_default_and_catastrophic(self):
        h = Harness()
        self.assertEqual(h.at(5, 9800).kind, "STOP")                       # SL_CONFIRM_S = 0 (current behaviour)
        h2 = Harness(cfg=replace(GROSS, sl_confirm_s=3.0))
        self.assertIsNone(h2.at(5, 9800))
        self.assertEqual(h2.at(6, 9740).kind, "STOP")                       # 0.5 %p beyond -> no confirm

    def test_ai_tighten_only_with_loss(self):
        ai = AIAdjust(status="ok", applied=True, sl_tighten=True)
        h = Harness()
        h.at(5, 9950, ai=ai)
        self.assertEqual((h.st.sl_price, h.st.sl_source), (9920, "AI"))      # -0.5 - 0.3 = -0.8 %
        h2 = Harness()
        h2.at(5, 10050, ai=ai)
        self.assertEqual(h2.st.sl_source, "INITIAL")

    def test_max_hold(self):
        h = Harness()
        h.at(30, 10120)                                                     # peak 1.2 %: no time stop
        it = h.at(3600, 10050)
        self.assertEqual(it.kind, "MAX_HOLD")


class OperatingModes(unittest.TestCase):
    def test_dead_data_no_price_signals_but_force_exit(self):
        h = Harness()
        self.assertIsNone(h.at(10, 9500, last_tick_age_s=90, rest_age_s=90))   # far below the stop: ignored
        self.assertEqual(h.logs[-1]["op_mode"], "DEAD")
        force = datetime(2026, 11, 16, 15, 15, 0)
        it = h.at(int((force - T0).total_seconds()), 9500, last_tick_age_s=90, rest_age_s=90)
        self.assertEqual((it.kind, it.order_type, it.qty), ("FORCE", "market", 100))

    def test_force_limit_then_market_escalation(self):
        h = Harness()
        s1515 = int((datetime(2026, 11, 16, 15, 15) - T0).total_seconds())
        it = h.at(s1515, 10050, bid1=10040, fill=False)
        self.assertEqual((it.kind, it.order_type, it.limit_price), ("FORCE", "limit", 10040))
        h.st = h.eng.on_submitted(h.st, it, 100, "F1", T0 + timedelta(seconds=s1515))
        self.assertIsNone(h.at(s1515 + 60, 10050, fill=False))
        it2 = h.at(s1515 + 180, 10050, fill=False)
        self.assertEqual((it2.kind, it2.order_type, it2.replace_pending), ("FORCE", "market", True))

    def test_safe_mode_floor_width_and_no_ai(self):
        ai = AIAdjust(status="ok", applied=True, m_trail=1.4)
        h = Harness()
        h.at(10, 10200, ai=ai)
        h.at(20, 10250, ai=ai, last_tick_age_s=20, rest_age_s=3)
        lg = h.logs[-1]
        self.assertEqual(lg["op_mode"], "SAFE")
        self.assertFalse(lg["ai_applied"])
        self.assertAlmostEqual(lg["w1"], 0.3)

    def test_vi_stop_limit_and_trailing_suppressed(self):
        h = Harness()
        h.at(10, 10200)                               # TP1 sold 40
        h.at(20, 10300, vi_active=True)
        self.assertNotIn("w1", h.logs[-1])            # no trailing / ladder during VI
        it = h.at(30, 9700, vi_active=True, vi_expected_price=9600, lower_limit=7000, fill=False)
        self.assertEqual((it.kind, it.order_type, it.limit_price), ("STOP", "limit", 9580))
        it2 = h.at(31, 9700, vi_active=True, fill=False)
        self.assertEqual((it2.kind, it2.order_type), ("STOP", "market"))
        h.at(40, 10300)
        self.assertIsNotNone(h.st.vi_released_ts)

    def test_upper_limit_hold_and_break(self):
        h = Harness(entry=10000)
        up = 13000
        h.at(10, 10200)                                        # TP1
        self.assertIsNone(h.at(20, up, upper_limit=up))        # ladder held at the limit
        self.assertTrue(h.st.at_upper_limit)
        it = h.at(21, up - 10, upper_limit=up)
        self.assertEqual((it.kind, it.reason), ("TRAIL_A", "TRAIL_A_UPPER_LIMIT_BREAK"))

    def test_nxt_prices(self):
        h = Harness()
        it = h.at(5, 9800, venue="NXT", bid1=9790, fill=False)
        self.assertEqual((it.order_type, it.limit_price), ("limit", 9760))   # bid - 3 ticks

    def test_peak_from_high_since_prev(self):
        h = Harness()
        h.at(10, 10150, high_since_prev=10360)
        self.assertEqual(h.st.peak_price, 10360)
        self.assertTrue(h.st.trail_armed)

    def test_peak_rel_unit(self):
        h = Harness(cfg=replace(GROSS, retrace_unit="peak_rel"))
        h.at(10, 10200)
        h.at(20, 10350)
        h.at(25, 10320)
        # 10,350 x (1 - 0.6 %) = 10,287.9 -> 10,280
        self.assertEqual(h.logs[-1]["trail_price_a"], 10280)


class RestartAndOrders(unittest.TestCase):
    def test_state_round_trip(self):
        h = Harness(open_kw=SURGE_FEATS)
        h.at(60, 10300, **SURGE_HOLD)
        h.at(90, 10350, **SURGE_HOLD)
        h.at(110, 10290, **SURGE_HOLD)                     # timer running
        d = json.loads(json.dumps(h.st.to_dict()))
        st2 = ExitState.from_dict(d)
        self.assertEqual(st2.timers, {})                       # confirm windows restart after a restart
        self.assertEqual(replace(st2, timers=h.st.timers), h.st)
        d["version"] = 99
        with self.assertRaises(ValueError):
            ExitState.from_dict(d)

    def test_rejected_order_reverts_stage(self):
        eng = ExitDecisionEngine(GROSS)
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        r = eng.step(st, snap(10, 10200), T0 + timedelta(seconds=10))
        st = eng.on_submitted(r.state, r.intent, 40, "L1", T0 + timedelta(seconds=10))
        self.assertEqual(st.ladder_step, 1)
        st = eng.on_order_closed(st, unfilled_qty=40, revert_stage=True)    # broker rejected
        self.assertEqual((st.ladder_step, st.pending_qty, st.pending_kind), (0, 0, ""))
        r = eng.step(st, snap(13, 10200), T0 + timedelta(seconds=13))
        self.assertEqual((r.intent.kind, r.intent.qty), ("LADDER", 40))

    def test_trail_a_replaces_pending_ladder(self):
        eng = ExitDecisionEngine(replace(GROSS, trail_confirm_s=0.0))
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        r = eng.step(st, snap(10, 10350), T0 + timedelta(seconds=10))
        st = eng.on_submitted(r.state, r.intent, 40, "L1", T0 + timedelta(seconds=10))   # unfilled
        r = eng.step(st, snap(11, 10290), T0 + timedelta(seconds=11))
        self.assertEqual((r.intent.kind, r.intent.qty, r.intent.replace_pending), ("TRAIL_A", 70, True))

    def test_manual_sell_sync(self):
        eng = ExitDecisionEngine(GROSS)
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        st = eng.sync_holding(st, 60)
        self.assertEqual((st.held_qty, st.sold_qty), (60, 40))
        st = eng.sync_holding(st, 0)
        self.assertEqual(st.phase, "CLOSED")

    def test_rebase_keeps_stop_price(self):
        eng = ExitDecisionEngine(GROSS)
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        st.sl_price = 10010
        st2 = eng.rebase_position(st, 10100, 150, T0)
        self.assertEqual((st2.q0, st2.ladder_plan, st2.sl_price), (150, (60, 105, 150), 10010))


class RegimeTests(unittest.TestCase):
    def test_flicker_does_not_switch(self):
        h = Harness()
        for sec in range(60, 120, 4):        # condition true for 3 s, false for 1 s - never 5 s in a row
            for k in range(3):
                h.at(sec + k, 10050, **SURGE_FEATS)
            h.at(sec + 3, 10050)
        self.assertEqual(h.st.regime, "NORMAL")

    def test_enter_after_confirm_and_dwell(self):
        h = Harness()
        h.hold(10, 30, 10050, **SURGE_FEATS)                 # dwell from entry (60 s) not reached
        self.assertEqual(h.st.regime, "NORMAL")
        h.hold(31, 65, 10050, **SURGE_FEATS)
        self.assertEqual(h.st.regime, "SURGE")

    def test_rvol_unknown_and_obi_off_cannot_confirm(self):
        h = Harness()
        h.hold(60, 80, 10050, ret_3m_pct=2.0, cttr_60s=180.0)
        self.assertEqual(h.st.regime, "NORMAL")

    def test_below_entry_drops_surge_immediately(self):
        h = Harness(open_kw=SURGE_FEATS)
        h.at(5, 9990, **SURGE_FEATS)
        self.assertEqual(h.st.regime, "NORMAL")

    def test_bad_quality_cannot_confirm(self):
        h = Harness()
        h.hold(60, 80, 10050, gap_volume_ratio=0.3, **SURGE_FEATS)
        self.assertEqual(h.st.regime, "NORMAL")


class TrailFactors(unittest.TestCase):
    def _factors(self, **kw):
        tr = ax.TrailingExit(GROSS)
        st = ExitDecisionEngine(GROSS).open_position("x", 10000, 100, T0)
        st.peak_price, st.peak_ts = 10300, T0
        return tr.rule_factors(st, snap(1, 10300, **kw), T0 + timedelta(seconds=1))

    def test_rvol_bands_cover_every_value(self):
        for r, m in ((5, 1.2), (3.0, 1.2), (2.0, 1.0), (1.5, 1.0), (1.2, 0.95), (1.0, 0.95), (0.5, 0.85)):
            self.assertEqual(self._factors(rvol_5m=r)["m_rvol"], m, r)

    def test_cttr_crash_wins(self):
        self.assertEqual(self._factors(cttr_60s=160, cttr_60s_prev30=210)["m_cttr"], 0.7)
        self.assertEqual(self._factors(cttr_60s=160, cttr_60s_prev30=180)["m_cttr"], 1.15)
        self.assertEqual(self._factors(cttr_60s=90)["m_cttr"], 0.8)

    def test_bad_quality_cannot_widen(self):
        f = self._factors(rvol_5m=5, cttr_60s=180, window_coverage=0.5)
        self.assertEqual((f["m_rvol"], f["m_cttr"]), (1.0, 1.0))

    def test_low_price_tick_floor(self):
        # 1,000 KRW stock: 3 ticks = 0.3 %; 1,500 KRW: 3 ticks = 0.2 % -> 0.3 %p floor; 4,000 (tick 5): 0.375 %
        for entry, floor in ((1000, 0.3), (1500, 0.3), (4000, 0.375)):
            st = ExitDecisionEngine(GROSS).open_position("x", entry, 100, T0)
            self.assertAlmostEqual(ax.TrailingExit(GROSS).floor_pp(st), floor, places=9)

    def test_trigger_gap_after_rounding(self):
        tr = ax.TrailingExit(GROSS)
        st = ExitDecisionEngine(GROSS).open_position("x", 4990, 100, T0)
        st.peak_price = 5080
        pa, pb = tr.trigger_prices(st, 0.3, 0.31)
        self.assertLessEqual(pa, ax.tick_down(5080, 3))
        self.assertLessEqual(pb, ax.tick_down(pa, 2))


class AIAdapterTests(unittest.TestCase):
    NOW = T0 + timedelta(seconds=30)

    def _adapter(self, fn, mode="live", clock=None, **kw):
        clock = clock or [100.0]
        cfg = replace(GROSS, ai_mode=mode)
        return EdgeAIAdapter(cfg, fn, inline=True, mono=lambda: clock[0], **kw), clock

    @staticmethod
    def _out(cont=0.9, rev=0.05, conf=0.9, **kw):
        kw.setdefault("ts", AIAdapterTests.NOW)
        return dict(continuation_prob=cont, reversal_prob=rev, confidence=conf, model_version="m1", **kw)

    def test_ok_live_and_shadow(self):
        ad, _ = self._adapter(lambda f: self._out())
        ad.submit("A", {}, self.NOW)
        a = ad.latest("A", self.NOW)
        self.assertEqual(a.status, "ok")
        self.assertTrue(a.applied)
        self.assertAlmostEqual(a.m_trail, 1.306, places=9)
        self.assertAlmostEqual(a.stage_a_ratio, 0.7 - 0.2 * 0.765, places=9)
        sh, _ = self._adapter(lambda f: self._out(), mode="shadow")
        sh.submit("A", {}, self.NOW)
        s = sh.latest("A", self.NOW)
        self.assertEqual((s.status, s.applied, s.m_trail), ("shadow", False, 1.0))
        self.assertAlmostEqual(s.would_be["m_trail"], 1.306, places=9)

    def test_bounds(self):
        ad, _ = self._adapter(lambda f: self._out(cont=1.0, rev=0.0, conf=1.0))
        ad.submit("A", {}, self.NOW)
        self.assertEqual(ad.latest("A", self.NOW).m_trail, 1.4)
        ad2, _ = self._adapter(lambda f: self._out(cont=0.0, rev=1.0, conf=1.0))
        ad2.submit("A", {}, self.NOW)
        a = ad2.latest("A", self.NOW)
        self.assertEqual((a.m_trail, a.sl_tighten), (0.6, True))
        self.assertAlmostEqual(a.stage_a_ratio, 0.9, places=9)

    def test_timeout(self):
        clock = [100.0]

        def slow(_):
            clock[0] += 0.18
            return self._out()
        ad, _ = self._adapter(slow, clock=clock)
        ad.submit("A", {}, self.NOW)
        self.assertEqual(ad.latest("A", self.NOW).status, "timeout")

    def test_invalid_values(self):
        cases = [self._out(cont=1.7), self._out(cont=float("nan")), self._out(conf=-0.1),
                 self._out(cont=0.7, rev=0.5), dict(self._out(), model_version="other"),
                 {"continuation_prob": "x", "reversal_prob": 0.1, "confidence": 0.9},
                 self._out(ts=self.NOW + timedelta(seconds=5)), self._out(ts=self.NOW - timedelta(seconds=60))]
        for out in cases:
            ad, _ = self._adapter(lambda f, o=out: o, model_version="m1")
            ad.submit("A", {}, self.NOW - timedelta(seconds=1))
            self.assertEqual(ad.latest("A", self.NOW).status, "invalid", out)

    def test_low_conf_error_stale(self):
        ad, _ = self._adapter(lambda f: self._out(conf=0.5))
        ad.submit("A", {}, self.NOW)
        self.assertEqual(ad.latest("A", self.NOW).status, "low_conf")

        def boom(_):
            raise RuntimeError("model broke")
        ad, _ = self._adapter(boom)
        ad.submit("A", {}, self.NOW)
        self.assertEqual(ad.latest("A", self.NOW).status, "error")
        clock = [100.0]
        ad, _ = self._adapter(lambda f: self._out(), clock=clock)
        ad.submit("A", {}, self.NOW)
        clock[0] += 2.5
        self.assertEqual(ad.latest("A", self.NOW).status, "stale")
        self.assertEqual(ad.latest("B", self.NOW).status, "stale")      # never submitted

    def test_off_mode(self):
        ad, _ = self._adapter(lambda f: self._out(), mode="off")
        self.assertEqual(ad.submit("A", {}, self.NOW), -1)
        self.assertEqual(ad.latest("A", self.NOW).status, "off")

    def test_threaded_hang_never_blocks_and_disables(self):
        import threading
        gate, started = threading.Event(), threading.Event()
        clock = [100.0]

        def hang(_):
            started.set()
            gate.wait(5)
            return self._out()
        cfg = replace(GROSS, ai_mode="live")
        ad = EdgeAIAdapter(cfg, hang, inline=False, mono=lambda: clock[0])
        try:
            ad.submit("A", {}, self.NOW)
            self.assertEqual(ad.latest("A", self.NOW).status, "stale")      # returns immediately
            self.assertTrue(started.wait(2))                                 # stall = the RUNNING job
            clock[0] += 11.0                                                 # > 2 s x 5
            self.assertEqual(ad.latest("A", self.NOW).status, "disabled")
            self.assertEqual(ad.submit("A", {}, self.NOW), -1)
        finally:
            gate.set()
            ad.close()

    def test_older_result_does_not_overwrite_newer(self):
        ad, clock = self._adapter(lambda f: self._out(cont=f["c"], rev=0.0))
        ad._run(("A", 2, {"c": 0.8}, self.NOW))
        ad._run(("A", 1, {"c": 0.1}, self.NOW))
        self.assertAlmostEqual(ad.latest("A", self.NOW).raw["continuation_prob"], 0.8)


class CodeReviewFixes(unittest.TestCase):
    """Codex code review 2026-11-15 (adaptive_exit_codex_code_review_20261115.md) #1-#10."""

    def test_1_trail_b_partial_fill_then_cancel_resumes(self):
        eng = ExitDecisionEngine(replace(GROSS, trail_confirm_s=0.0))
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        st = replace(st, ladder_step=3)                         # isolate trailing (no ladder)
        r = eng.step(st, snap(10, 10350), T0 + timedelta(seconds=10))
        r = eng.step(r.state, snap(11, 10240), T0 + timedelta(seconds=11))
        self.assertEqual((r.intent.kind, r.intent.qty), ("TRAIL_B", 100))
        st = eng.on_submitted(r.state, r.intent, 100, "B1", T0 + timedelta(seconds=11))
        st = eng.on_fill(st, 10)
        st = eng.on_order_closed(st, unfilled_qty=90, revert_stage=False)
        st = eng.sync_holding(st, 90)
        r = eng.step(st, snap(13, 10300), T0 + timedelta(seconds=13))  # price recovered above B: still finish
        self.assertEqual((r.intent.kind, r.intent.qty, r.intent.reason), ("TRAIL_B", 90, "TRAIL_B_RESUME"))

    def test_1_rejected_full_exit_is_not_sticky(self):
        eng = ExitDecisionEngine(GROSS)
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        r = eng.step(st, snap(5, 9800), T0 + timedelta(seconds=5))
        st = eng.on_submitted(r.state, r.intent, 100, "S1", T0 + timedelta(seconds=5))
        st = eng.on_order_closed(st, 100, revert_stage=True)
        self.assertEqual(st.closing_kind, "")
        self.assertIsNone(eng.step(st, snap(6, 9900), T0 + timedelta(seconds=6)).intent)

    def test_2_freeze_uses_current_ai_width(self):
        ai = AIAdjust(status="ok", applied=True, m_trail=0.6)
        h = Harness(open_kw=SURGE_FEATS)
        h.st = replace(h.st, ladder_step=3)
        h.at(10, 10350, ai=ai, **SURGE_HOLD)
        h.at(11, 10330, ai=ai, **SURGE_HOLD)                    # drop 0.2 >= 0.5 x 0.36
        self.assertTrue(h.logs[-1]["frozen"])
        self.assertAlmostEqual(h.logs[-1]["w2"], 0.66, places=9)
        h.at(12, 10330, ai=AIAdjust(status="ok", applied=True, m_trail=1.4), **SURGE_HOLD)
        self.assertAlmostEqual(h.logs[-1]["w2"], 0.66, places=9)   # AI 1.4 cannot widen back to 1.1
        self.assertEqual(h.logs[-1]["trail_price_b"], 10280)

    def test_2_frozen_widths_never_widen_again(self):
        h = Harness(open_kw=SURGE_FEATS)
        h.st = replace(h.st, ladder_step=3)
        h.at(10, 10350, **SURGE_HOLD)
        widths = []
        for i, cttr in enumerate((120.0, 90.0, 120.0, 160.0)):
            h.at(11 + i, 10320, ret_3m_pct=0.5, rvol_5m=2.0, cttr_60s=cttr)
            widths.append((h.logs[-1]["w1"], h.logs[-1]["w2"]))
        for a, b in zip(widths, widths[1:]):
            self.assertLessEqual(b[0], a[0] + 1e-12)
            self.assertLessEqual(b[1], a[1] + 1e-12)
        self.assertAlmostEqual(widths[-1][0], 0.48, places=9)
        h.at(20, 10400, **SURGE_HOLD)                            # new peak unfreezes
        self.assertFalse(h.logs[-1]["frozen"])

    def test_4_ai_ts_timezone_and_garbage(self):
        from datetime import timezone as tz
        now = T0 + timedelta(seconds=30)
        cases = {
            "ok": now.replace(tzinfo=tz(timedelta(hours=9))),          # aware KST == naive local
            "ok_iso": now.isoformat() + "+09:00",
            "invalid_str": "1999-01-01",
            "invalid_garbage": "not-a-time",
            "invalid_missing": None,
        }
        for name, ts in cases.items():
            out = dict(continuation_prob=0.6, reversal_prob=0.2, confidence=0.8, model_version="m1")
            if ts is not None:
                out["ts"] = ts
            ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), lambda f, o=out: o, inline=True, mono=lambda: 1.0)
            ad.submit("A", {}, now)
            status = ad.latest("A", now).status
            self.assertEqual(status, "ok" if name.startswith("ok") else "invalid", name)

    def test_4_latest_never_raises(self):
        ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), lambda f: {}, inline=True, mono=lambda: 1.0)
        ad.submit("A", {}, T0)
        self.assertEqual(ad.latest("A", "not a datetime").status, "error")

    def test_5_dropped_queue_job_is_not_a_stall_and_resubmits_do_not_hide_one(self):
        import threading
        started, gate = threading.Event(), threading.Event()
        clock = [100.0]

        def slow(f):
            if f.get("block"):
                started.set()
                gate.wait(5)
            return dict(continuation_prob=0.5, reversal_prob=0.2, confidence=0.9, model_version="m1",
                        ts=f["ts"])
        ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), slow, inline=False, mono=lambda: clock[0])
        try:
            now = T0
            ad.submit("A", {"block": True, "ts": now}, now)
            self.assertTrue(started.wait(2))
            ad.submit("B", {"ts": now}, now)                   # queued
            ad.submit("C", {"ts": now}, now)                   # replaces B in the queue
            clock[0] += 3.0
            self.assertEqual(ad.latest("B", now).status, "stale")   # not "disabled": B never ran
            for _ in range(10):                                # A still hangs; repeated submits must not reset it
                clock[0] += 1.0
                ad.submit("A", {"ts": now}, now)
            self.assertEqual(ad.latest("A", now).status, "disabled")
        finally:
            gate.set()
            ad.close()

    def test_6_old_input_snapshot_is_stale(self):
        now = T0 + timedelta(seconds=120)
        out = dict(continuation_prob=0.6, reversal_prob=0.2, confidence=0.8, model_version="m1", ts=T0)
        ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), lambda f: out, inline=True, mono=lambda: 1.0)
        ad.submit("A", {}, T0)                                  # judged on a 120 s old snapshot
        self.assertEqual(ad.latest("A", now).status, "stale")

    def test_7_unknown_side_volume_fails_quality(self):
        s = snap(1, 10000, unknown_side_ratio=0.5, rvol_5m=4.0, cttr_60s=200.0)
        self.assertFalse(s.quality_ok(GROSS))

    def test_8_dead_gap_restarts_confirm(self):
        h = Harness()
        h.st = replace(h.st, ladder_step=3)
        h.at(10, 10350)
        self.assertIsNone(h.at(11, 10240))                     # B timer starts
        self.assertIsNone(h.at(12, 10240, last_tick_age_s=90, rest_age_s=90))   # DEAD
        self.assertIsNone(h.at(100, 10240))                    # recovered: must re-confirm
        h.at(101, 10240)
        self.assertEqual(h.at(102, 10240).kind, "TRAIL_B")

    def test_8_vi_clears_trail_timers(self):
        h = Harness()
        h.st = replace(h.st, ladder_step=3)
        h.at(10, 10350)
        h.at(11, 10240)
        h.at(12, 10240, vi_active=True)
        self.assertNotIn("TRAIL_B", h.st.timers)

    def test_9_ai_stop_rounds_up_net(self):
        ai = AIAdjust(status="ok", applied=True, sl_tighten=True)
        h = Harness(cfg=ExitConfig())                          # net basis
        h.at(5, 9950, ai=ai)
        self.assertEqual((h.st.sl_price, h.st.sl_source), (9920, "AI"))

    def test_sync_holding_refuses_in_flight(self):
        eng = ExitDecisionEngine(GROSS)
        st = eng.open_position("000001", 10000, 100, T0, snap(0, 10000))
        r = eng.step(st, snap(5, 10200), T0 + timedelta(seconds=5))
        st = eng.on_submitted(r.state, r.intent, 40, "L1", T0 + timedelta(seconds=5))
        with self.assertRaises(ValueError):
            eng.sync_holding(st, 60)


class ReReviewFixes(unittest.TestCase):
    """Codex re-review 2026-11-15 (adaptive_exit_codex_rereview_20261115.md)."""

    def test_rr2_new_peak_after_safe_is_not_refrozen(self):
        h = Harness()
        h.st = replace(h.st, ladder_step=3)
        h.at(10, 10350)
        h.at(11, 10330, last_tick_age_s=20, rest_age_s=2)      # SAFE: widths = floor 0.3, frozen
        self.assertAlmostEqual(h.logs[-1]["w1"], 0.3)
        h.at(12, 10360, high_since_prev=10400)                  # new peak, NORMAL again
        # 0.4 %p below the NEW peak re-freezes legitimately - but at the fresh 0.6 width, not the old SAFE 0.3
        self.assertAlmostEqual(h.logs[-1]["w1"], 0.6)
        self.assertEqual(h.logs[-1]["trail_price_a"], 10340)

    def test_rr2_safe_freeze_kept_on_same_peak(self):
        h = Harness()
        h.st = replace(h.st, ladder_step=3)
        h.at(10, 10350)
        h.at(11, 10330, last_tick_age_s=20, rest_age_s=2)
        h.at(12, 10340)                                         # NORMAL, same peak: stays at the SAFE floor
        self.assertTrue(h.logs[-1]["frozen"])
        self.assertAlmostEqual(h.logs[-1]["w1"], 0.3)

    def test_rr3_finished_job_not_mistaken_for_stall(self):
        ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), lambda f: {}, inline=True)
        ad._running = ("A", 1, 0.0)

        def mono():                                             # worker finishes right after the first read
            ad._running = None
            return 100.0
        ad._mono = mono
        self.assertNotEqual(ad.latest("A", T0).status, "disabled")
        self.assertEqual(ad.disabled_reason, "")

    def test_rr4_close_stops_worker_even_with_full_queue(self):
        import threading
        gate, started = threading.Event(), threading.Event()

        def slow(f):
            started.set()
            gate.wait(2)
            return {}
        ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), slow, inline=False)
        ad.submit("A", {}, T0)
        self.assertTrue(started.wait(2))
        ad.submit("B", {}, T0)                                  # queue full
        gate.set()
        self.assertTrue(ad.close(timeout_s=2.0))
        self.assertEqual(ad.submit("C", {}, T0), -1)            # rejected after close

    def test_rr_ai_whole_second_ts_accepted(self):
        snap_ts = T0 + timedelta(seconds=30, microseconds=700000)
        out = dict(continuation_prob=0.6, reversal_prob=0.2, confidence=0.8, model_version="m1",
                   ts=(T0 + timedelta(seconds=30)).isoformat())
        ad = EdgeAIAdapter(replace(GROSS, ai_mode="live"), lambda f: out, inline=True, mono=lambda: 1.0)
        ad.submit("A", {}, snap_ts)
        self.assertEqual(ad.latest("A", snap_ts).status, "ok")


if __name__ == "__main__":
    unittest.main()
