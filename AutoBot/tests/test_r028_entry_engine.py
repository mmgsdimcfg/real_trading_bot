# -*- coding: utf-8 -*-
"""r028 entry engine: parameters, bars, indicator parity with the g034 study, SEQ / WIN state machines, filters,
Edge AI gate, order executor, latency logger, shadow runner on a tick file, and the stage-4 scenario table
(entry_engine_analysis_20261115.md section 8)."""
import json
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import r028_entry_engine as r028  # noqa: E402
from r028_entry_engine import (Bar, BarSeries, EdgeAIGate, EntryEngine, EntryFilter, EntryParams, LatencyLogger,  # noqa: E402
                               OrderExecutor, Print, SeriesSignal, SignalDetector)

DAY = datetime(2026, 11, 16)
CODE = "000001"
# small tick bars and a 1-minute regime so a synthetic day stays short; filters relaxed per test
TEST_OVR = {"bars": {"regime_minutes": 1, "trigger_tick_sizes": [10], "warmup_prior_day": False},
            "signal": {"combo": "MA"}}


def P(sec: float, price: float, qty: int = 10, side: int = 1, acml: int = 0) -> Print:
    t = DAY + timedelta(seconds=sec)
    return Print(ts=t.replace(microsecond=0), recv=t, price=float(price), qty=qty, side=side, acml_vol=acml)


def params(**over) -> EntryParams:
    o = json.loads(json.dumps(TEST_OVR))
    for k, v in over.items():
        o.setdefault(k, {}).update(v)
    return EntryParams.load(overrides=o)


def uptrend_then_dip(start_s=9 * 3600, base=10000.0, minutes=30, dip_ticks=3, tick=10.0):
    """1 print / s. Gentle uptrend (1 tick / 2 min) -> regime up; then a short dip and a recovery so the 10-tick MA5
    crosses MA20 again. Returns (prints, seconds of day after the last print, last price)."""
    out, acml, s, px = [], 0, float(start_s), base
    for i in range(minutes * 60):
        px = base + tick * (i // 120)
        acml += 10
        out.append(P(s, px, side=1 if i % 2 else 5, acml=acml))
        s += 1
    top = px
    for k in range(1, dip_ticks + 1):                     # dip: 40 prints per tick down
        for _ in range(40):
            acml += 10
            out.append(P(s, top - tick * k, side=5, acml=acml))
            s += 1
    for k in range(dip_ticks - 1, -1, -1):                # recovery
        for _ in range(40):
            acml += 10
            out.append(P(s, top - tick * k, side=1, acml=acml))
            s += 1
    return out, s, top, acml


class ParamsTest(unittest.TestCase):
    def test_yaml_loads_and_validates(self):
        p = EntryParams.load()
        self.assertEqual(p["signal"]["combo"], "SEQ")
        self.assertEqual(p["bars"]["regime_minutes"], 3)
        self.assertEqual(p["filter"]["end"], "14:00:00")

    def test_override_merges_nested(self):
        p = EntryParams.load(overrides={"filter": {"chase_max_pct": 0.5}})
        self.assertEqual(p["filter"]["chase_max_pct"], 0.5)
        self.assertEqual(p["filter"]["vwap_gap_max_pct"], 1.5)        # untouched sibling kept

    def test_invalid_values_rejected(self):
        for bad in ({"signal": {"combo": "XYZ"}}, {"filter": {"start": "15:00:00"}}, {"order": {"style": "iceberg"}},
                    {"edge_ai": {"mode": "approve"}}, {"bars": {"trigger_tick_sizes": []}}, {"signal": {"macd": [26, 12, 9]}},
                    {"bars": {"regime_minutes": 1, "time_bar_delay_s": 60}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                EntryParams.load(overrides=bad)


class BarTest(unittest.TestCase):
    def test_tick_bar_closes_on_nth_print_at_its_receive_time(self):
        s = BarSeries("tick", 3)
        out = [s.add(p, p.recv) for p in (P(32400, 100), P(32401, 102), P(32402, 101))]
        self.assertEqual([len(x) for x in out], [0, 0, 1])
        b = out[-1][0]
        self.assertEqual((b.open, b.high, b.low, b.close, b.n), (100, 102, 100, 101, 3))
        self.assertEqual(b.avail, DAY + timedelta(seconds=32402))

    def test_time_bar_closes_by_next_print_or_clock_and_drops_late_prints(self):
        s = BarSeries("time", 1, delay_s=5)
        s.add(P(32400, 100), DAY + timedelta(seconds=32400))
        self.assertEqual(s.advance(DAY + timedelta(seconds=32464)), [])            # 09:01:04 < end + 5 s
        closed = s.advance(DAY + timedelta(seconds=32465))
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].avail, DAY + timedelta(seconds=32465))
        self.assertEqual(s.add(P(32459, 90), DAY + timedelta(seconds=32466)), [])  # late print of the closed minute
        self.assertIsNone(s.forming)

    def test_time_bar_waits_for_the_confirmation_delay_and_takes_late_prints(self):
        """Codex review #2: a print of the next minute does not confirm the bar; end + delay does, and a late print of the
        ended minute that arrives inside the delay still counts."""
        s = BarSeries("time", 1, delay_s=5)
        s.add(P(32470, 101), DAY + timedelta(seconds=32470))                       # 09:01:10
        self.assertEqual(s.add(P(32520, 103), DAY + timedelta(seconds=32520)), [])  # 09:02:00 next minute: pending only
        self.assertEqual(s.add(P(32519, 99), DAY + timedelta(seconds=32522)), [])   # late 09:01:59 print received 09:02:02
        self.assertEqual(s.advance(DAY + timedelta(seconds=32524)), [])
        closed = s.advance(DAY + timedelta(seconds=32525))                         # 09:02:05 = end + 5 s
        self.assertEqual((closed[0].close, closed[0].low, closed[0].n), (99, 99, 2))
        self.assertEqual(closed[0].avail, DAY + timedelta(seconds=32525))
        self.assertEqual(s.forming.close, 103)

    def test_warm_up_never_completes_a_prior_session_tick_bar(self):
        """Codex review #6."""
        eng = EntryEngine(params())
        prior = [Print(ts=DAY - timedelta(days=1) + timedelta(seconds=50000 + i),
                       recv=DAY - timedelta(days=1) + timedelta(seconds=50000 + i), price=100.0, qty=1, acml_vol=i + 1)
                 for i in range(25)]                                              # 2 full 10-tick bars + 5 prints
        eng.warm(CODE, prior)
        st = eng.codes[CODE]
        self.assertIsNone(st.builder.series["t10"].forming)
        self.assertEqual(len(st.builder.series["t10"].bars), 2)
        self.assertEqual(st.det["regime"].ind.n, 1)                                 # the prior day's last minute is final

    def test_builder_drops_duplicates_and_out_of_session_prints(self):
        b = r028.BarBuilder(params())
        self.assertEqual(b.ingest(P(8 * 3600 + 59 * 60, 100, acml=5)), {})       # 08:59 pre-market
        self.assertTrue(b.ingest(P(32400, 100, acml=10)))
        self.assertEqual(b.ingest(P(32400, 100, acml=10)), {})                    # same cumulative volume = duplicate
        self.assertEqual(len(b._px), 1)

    def test_trim_keeps_features_identical(self):
        a, b = r028.BarBuilder(params()), r028.BarBuilder(params())
        b.TRIM_AT, b.KEEP_S, b._next_trim = 500, 2400.0, 500
        rng = np.random.default_rng(1)
        for i in range(6000):
            p = P(32400 + i, 10000 + 10 * int(rng.integers(-3, 4)), qty=int(rng.integers(1, 50)), side=int(rng.choice([1, 5])),
                  acml=i + 1)
            a.ingest(p)
            b.ingest(p)
        self.assertLess(len(b._rt), 3000)
        now = DAY + timedelta(seconds=32400 + 5999)
        fa, fb = a.features(now), b.features(now)
        for k in fa:
            self.assertAlmostEqual(fa[k], fb[k], places=9, msg=k)

    def test_features_use_only_visible_prints(self):
        b = r028.BarBuilder(params())
        for i in range(600):
            b.ingest(P(32400 + i, 100 + (i // 60), side=1, acml=i + 1))
        f = b.features(DAY + timedelta(seconds=32400 + 300))
        self.assertEqual(f["price"], 105)                                         # print at +300 s, not the last one
        self.assertGreater(f["vwap_gap"], 0)


class IndicatorParityTest(unittest.TestCase):
    """SignalDetector (incremental) must equal the g034 pandas study on the same closes."""

    def test_values_and_combos_match_g034(self):
        import pandas as pd
        import g034_entry_bar_study as g034
        rng = np.random.default_rng(7)
        closes = 10000 + np.cumsum(rng.choice([-10, 0, 10], size=600, p=[0.33, 0.32, 0.35]))
        ref = g034.indicators(closes)
        for combo in ("SEQ", "WIN", "MA"):
            p = EntryParams.load(overrides={"signal": {"combo": combo, "window_bars": 10}})
            det = SignalDetector("t60", p)
            fires = []
            for k, c in enumerate(closes):
                s = det.update(Bar(None, c, c, c, c, avail=DAY + timedelta(seconds=k)))
                fires.append(s.fired)
                if combo == "SEQ":
                    for key in ("mid", "rsi", "hist"):
                        a, b = ref[key][k], s.__dict__[key]
                        if a == a:
                            self.assertAlmostEqual(a, b, places=6, msg=f"{key}@{k}")
                        else:
                            self.assertNotEqual(b, b, msg=f"{key}@{k} should be nan")
            bars = pd.DataFrame({"avail": np.arange(len(closes), dtype=float), "close": closes})
            sig = g034.series_signals(bars, np.array([]), 10)
            self.assertEqual(list(np.flatnonzero(fires)), list(np.flatnonzero(sig[combo])), combo)
            self.assertGreater(sum(fires), 0, combo)

    def test_regime_matches_g034(self):
        import pandas as pd
        import g034_entry_bar_study as g034
        rng = np.random.default_rng(3)
        closes = 10000 + np.cumsum(rng.choice([-10, 10], size=300))
        det = SignalDetector("regime", EntryParams.load())
        got = []
        for k, c in enumerate(closes):
            det.update(Bar(None, c, c, c, c, avail=DAY + timedelta(seconds=k)))
            got.append(det.regime_ok())
        sig = g034.series_signals(pd.DataFrame({"avail": np.arange(300.0), "close": closes}), np.array([]), 10)
        self.assertEqual(got, list(sig["regime"]))


class SeqStateMachineTest(unittest.TestCase):
    F0 = {"bbx": False, "macdx": False, "rsix": False, "max": False, "above": True, "hpos": True, "rpos": True}

    def run_flags(self, seq, w=3):
        det = SignalDetector("t60", EntryParams.load(overrides={"signal": {"window_bars": w}}))
        return [det._combo({**self.F0, **f}, k, commit=True) for k, f in enumerate(seq)]

    def test_in_order_within_window_fires(self):
        self.assertEqual(self.run_flags([{"bbx": True}, {"macdx": True}, {}, {"rsix": True}]), [False, False, False, True])

    def test_same_bar_all_three_fires(self):
        self.assertEqual(self.run_flags([{"bbx": True, "macdx": True, "rsix": True}]), [True])

    def test_wrong_order_does_not_fire(self):
        self.assertFalse(any(self.run_flags([{"rsix": True}, {"bbx": True}, {"macdx": True}, {}])))

    def test_step_too_late_resets(self):
        self.assertFalse(any(self.run_flags([{"bbx": True}, {}, {}, {}, {"macdx": True}, {"rsix": True}], w=3)))

    def test_fake_breakout_falls_back_below_mid_resets(self):
        """Scenario 2 (거래량 없는 가짜 돌파 후 하락): back under the middle before MACD / RSI confirm -> no signal."""
        flags = [{"bbx": True}, {"above": False}, {"macdx": True}, {"rsix": True}]
        self.assertFalse(any(self.run_flags(flags)))


class FilterTest(unittest.TestCase):
    def setUp(self):
        self.f = EntryFilter(EntryParams.load())
        self.sig = SeriesSignal("t60", True, DAY, 10000.0, rsi=60.0, upper=10300.0)
        self.now = DAY + timedelta(hours=10)

    def test_pass(self):
        self.assertEqual(self.f.check(self.now, 10010, 10000, self.sig, {"vwap_gap": 0.5}), (True, "OK"))

    def test_reasons(self):
        cases = [((DAY + timedelta(hours=14, minutes=1), 10010, 10000, {"vwap_gap": 0.5}), "TIME_"),
                 ((self.now, 10040, 10000, {"vwap_gap": 0.5}), "CHASE_"),
                 ((self.now, 10010, 10000, {"vwap_gap": 2.0}), "VWAP_GAP_"),
                 ((self.now, 10010, 10000, {"vwap_gap": -0.2}), "VWAP_GAP_")]
        for (now, px, ref, feat), prefix in cases:
            ok, why = self.f.check(now, px, ref, self.sig, feat)
            self.assertFalse(ok)
            self.assertTrue(why.startswith(prefix), why)
        hot = SeriesSignal("t60", True, DAY, 10000.0, rsi=80.0)
        self.assertTrue(self.f.check(self.now, 10010, 10000, hot, {"vwap_gap": 0.5})[1].startswith("OVERHEAT_RSI"))

    def test_nan_features_do_not_block(self):
        self.assertTrue(self.f.check(self.now, 10010, 10000, self.sig, {"vwap_gap": float("nan")})[0])


class EdgeAIGateTest(unittest.TestCase):
    def gate(self, fn, **a):
        return EdgeAIGate(EntryParams.load(overrides={"edge_ai": {"mode": "veto", **a}}), fn)

    def test_off_always_allows(self):
        self.assertEqual(EdgeAIGate(EntryParams.load(), lambda f: 0.0).check({}), (True, 1.0, "AI_OFF"))

    def test_veto_shrink_ok(self):
        self.assertFalse(self.gate(lambda f: 0.3).check({})[0])
        self.assertEqual(self.gate(lambda f: 0.55).check({})[:2], (True, 0.5))
        self.assertEqual(self.gate(lambda f: 0.9).check({})[:2], (True, 1.0))

    def test_timeout_and_error_fall_back_to_rules(self):
        """Scenario 7 (Edge AI 타임아웃): rules only, never a veto."""
        slow = self.gate(lambda f: (time.sleep(0.5), 0.0)[1], timeout_s=0.05)
        t = time.perf_counter()
        self.assertEqual(slow.check({}), (True, 1.0, "AI_TIMEOUT_RULES_ONLY"))
        self.assertLess(time.perf_counter() - t, 0.4)
        self.assertTrue(self.gate(lambda f: 1 / 0).check({})[2].startswith("AI_ERROR_RULES_ONLY"))

    def test_hung_model_is_not_stacked_and_trips_the_breaker(self):
        """Codex review #7: one inference in flight at most; repeated timeouts open the breaker."""
        ev = __import__("threading").Event()
        clock = [0.0]
        g = self.gate(lambda f: (ev.wait(5), 0.9)[1], timeout_s=0.01)
        g.clock = lambda: clock[0]
        self.assertEqual(g.check({})[2], "AI_TIMEOUT_RULES_ONLY")
        self.assertEqual(g.check({})[2], "AI_BUSY_RULES_ONLY")                      # still running: not submitted again
        ev.set()
        time.sleep(0.05)
        slow = self.gate(lambda f: (time.sleep(0.05), 0.9)[1], timeout_s=0.01)
        slow.clock = lambda: clock[0]
        for _ in range(3):
            self.assertEqual(slow.check({})[2], "AI_TIMEOUT_RULES_ONLY")
            time.sleep(0.1)                                                        # the previous inference ends
        self.assertEqual(slow.check({})[2], "AI_BREAKER_RULES_ONLY")
        clock[0] += slow.BREAKER_S + 1
        time.sleep(0.1)
        self.assertNotEqual(slow.check({})[2], "AI_BREAKER_RULES_ONLY")


class OrderExecutorTest(unittest.TestCase):
    def test_duplicate_guard_reprice_cap_and_cancel(self):
        """Scenario 8 (지정가 미체결 후 가격 이탈): reprice up to the chase cap, then cancel - never a market chase."""
        ex = OrderExecutor(EntryParams.load())
        t0 = DAY + timedelta(hours=10)
        plan = ex.submit(CODE, 10, t0, 10000, ref_price=10000)
        self.assertIsNotNone(plan)
        self.assertIsNone(ex.submit(CODE, 10, t0, 10000, 10000))                  # one active order per symbol
        self.assertEqual(ex.step(CODE, t0 + timedelta(seconds=2), 10010)[0], "NONE")
        self.assertEqual(ex.step(CODE, t0 + timedelta(seconds=3), 10010), ("REPRICE", 10010))
        self.assertEqual(ex.step(CODE, t0 + timedelta(seconds=6), 10050)[0], "NONE")   # above cap 10030
        self.assertEqual(ex.step(CODE, t0 + timedelta(seconds=10), 10050)[0], "CANCEL")
        self.assertIsNotNone(ex.submit(CODE, 10, t0 + timedelta(seconds=11), 10000, 10000))

    def test_fill_completes(self):
        ex = OrderExecutor(EntryParams.load())
        ex.submit(CODE, 10, DAY, 100, 100)
        ex.on_fill(CODE, 4)
        self.assertEqual(ex.active[CODE].state, "ACTIVE")
        ex.on_fill(CODE, 6)
        self.assertEqual(ex.active[CODE].state, "FILLED")


class LatencyLoggerTest(unittest.TestCase):
    def test_engine_record_then_real_order_carries_t0_to_t5(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            lg = LatencyLogger(tmp / "lat.jsonl")
            t = DAY + timedelta(hours=10)
            lg.engine_start(CODE, t, 100.0, t + timedelta(seconds=30), 100.5, series="t60")
            lg.engine_decision(CODE, t + timedelta(seconds=31), 100.5, "T60_SEQ")
            lg.order_submit(CODE, t + timedelta(seconds=32), 100.5, "LIVE")
            lg.engine_start(CODE, None, float("nan"), t + timedelta(seconds=33), 101.0)   # new candidate while pending
            lg.order_fill(CODE, t + timedelta(seconds=35), 101.0)
            rows = [json.loads(x) for x in (tmp / "lat.jsonl").read_text(encoding="utf-8").splitlines()]
            order = [r for r in rows if r["kind"] == "order"][0]
            self.assertEqual(order["outcome"], "FILLED")
            self.assertEqual((order["t0->t1_s"], order["t1->t_decision_s"], order["t_decision->t_submit_s"],
                              order["t_submit->t_fill_s"]), (30.0, 1.0, 1.0, 3.0))
            self.assertEqual(order["first_to_fill_pct"], 1.0)
            self.assertEqual([r["outcome"] for r in rows if r["kind"] == "engine"], ["SHADOW_BUY"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_attach_only_in_time_order_and_window(self):
        lg = LatencyLogger(None)
        t = DAY + timedelta(hours=10)
        lg.engine_start(CODE, None, float("nan"), t, 100.0)
        lg.order_submit(CODE, t + timedelta(seconds=LatencyLogger.ATTACH_S + 1), 100.0)   # too old
        self.assertNotIn("t1", lg.order_open[CODE])
        lg.engine_start(CODE, None, float("nan"), t + timedelta(hours=1), 100.0)
        lg.order_submit(CODE, t + timedelta(minutes=59), 100.0)                           # T1 after the submit
        self.assertNotIn("t1", lg.order_open[CODE])
        self.assertEqual(lg.rows[-1]["outcome"], "NO_FILL")                               # the first order superseded

    def test_decision_after_the_submit_is_not_copied(self):
        lg = LatencyLogger(None)
        t = DAY + timedelta(hours=10)
        lg.engine_start(CODE, None, float("nan"), t, 100.0)
        lg.engine_decision(CODE, t + timedelta(seconds=2), 100.0, "X")
        lg.order_submit(CODE, t + timedelta(seconds=1), 100.0)
        rec = lg.order_open[CODE]
        self.assertIn("t1", rec)
        self.assertNotIn("t_decision", rec)

    def test_close_writes_open_records(self):
        lg = LatencyLogger(None)
        t = DAY + timedelta(hours=10)
        lg.engine_start(CODE, None, float("nan"), t, 100.0)
        lg.order_submit("000002", t, 50.0)
        lg.close("DATE_CHANGE")
        self.assertEqual(sorted(r["outcome"] for r in lg.rows), ["DATE_CHANGE", "DATE_CHANGE"])
        self.assertEqual((lg.engine_open, lg.order_open), ({}, {}))


class ScenarioTest(unittest.TestCase):
    """Stage-4 scenario table. Synthetic day: 1 print / s, 10-tick trigger bars, 1-minute regime, combo MA."""

    def run_day(self, prints, p, until_s=None, blocked=None, ai=None):
        eng = EntryEngine(p, ai)
        out, k = [], 0
        end = until_s or int(prints[-1].recv.timestamp() - DAY.timestamp()) + 1
        start = int(prints[0].recv.timestamp() - DAY.timestamp())
        for sec in range(start, end + 1):
            now = DAY + timedelta(seconds=sec)
            while k < len(prints) and prints[k].recv <= now:
                eng.on_print(CODE, prints[k])
                k += 1
            out += eng.evaluate(now, CODE, blocked=blocked)
        return eng, out

    def base(self, **f):
        return params(filter={"start": "09:05:00", "end": "14:00:00", "chase_max_pct": 0.3, "vwap_gap_min_pct": 0.0,
                              "vwap_gap_max_pct": 1.5, "rsi_max": 101.0, **f})

    def test_1_real_breakout_buys(self):
        prints, _s, top, _a = uptrend_then_dip()
        _eng, out = self.run_day(prints, self.base())
        buys = [d for d in out if d.action == "BUY"]
        self.assertEqual(len(buys), 1, [(d.action, d.reason) for d in out])
        d = buys[0]
        self.assertEqual(d.series, "t10")
        self.assertLessEqual(d.price, top)
        self.assertIsNotNone(d.t0)                                                # T0 recorded from the forming bar
        self.assertLessEqual(d.t0, d.t1)
        self.assertEqual(d.exit_hints["regime"], "up")

    def run_with_tail(self, tail_fn, n_tail):
        """Feed the synthetic day until the candidate appears, then replace the rest of the day by tail_fn(i) prices."""
        prints, *_ = uptrend_then_dip()
        eng = EntryEngine(self.base())
        k = 0
        while eng.codes.get(CODE) is None or eng.codes[CODE].cand is None:
            eng.on_print(CODE, prints[k])
            k += 1
        c = eng.codes[CODE].cand
        s0 = prints[k - 1].recv.timestamp() - DAY.timestamp()
        acml = prints[k - 1].acml_vol
        decisions = []
        for i in range(1, n_tail + 1):
            acml += 10
            eng.on_print(CODE, P(s0 + i, tail_fn(i, c.p1), side=1, acml=acml))
            decisions += eng.evaluate(DAY + timedelta(seconds=s0 + i), CODE)
        return c, s0, decisions

    def test_3_late_signal_after_rise_is_given_up(self):
        """+0.8 % jump right after the signal: the chase limit (0.3 % over T0 / T1) refuses until the signal expires."""
        _c, _s0, decisions = self.run_with_tail(lambda i, p1: p1 * 1.008, 120)
        self.assertFalse([d for d in decisions if d.action == "BUY"])
        exp = [d for d in decisions if d.action == "EXPIRED"]
        self.assertTrue(exp and exp[0].reason.startswith("CHASE_"), [(d.action, d.reason) for d in decisions])

    def test_4_pullback_then_rebound_buys_within_validity(self):
        """Spike +0.8 % (refused), pull back to the signal price within the validity window -> buy then."""
        _c, s0, decisions = self.run_with_tail(lambda i, p1: p1 * 1.008 if i <= 10 else p1, 30)
        buys = [d for d in decisions if d.action == "BUY"]
        self.assertEqual(len(buys), 1, [(d.action, d.reason) for d in decisions])
        self.assertGreaterEqual(buys[0].at, DAY + timedelta(seconds=s0 + 11))     # only after the pullback

    def test_5_before_0900_nothing(self):
        prints = [P(8 * 3600 + 30 * 60 + i, 10000 + i, acml=i + 1) for i in range(1500)]
        eng, out = self.run_day(prints, self.base())
        self.assertEqual(out, [])
        self.assertEqual(eng.codes[CODE].builder.series["t10"].bars, type(eng.codes[CODE].builder.series["t10"].bars)())

    def test_6_thin_symbol_no_tick_bars_no_signal(self):
        prints = [P(9 * 3600 + 120 * i, 10000 + 10 * (i % 3), acml=i + 1) for i in range(60)]   # 1 print / 2 min
        eng, out = self.run_day(prints, self.base())
        self.assertEqual([d for d in out if d.action == "BUY"], [])
        self.assertLess(len(eng.codes[CODE].builder.series["t10"].bars), 10)

    def test_7_edge_ai_timeout_buys_on_rules(self):
        prints, *_ = uptrend_then_dip()
        p = params(filter={"chase_max_pct": 0.3, "rsi_max": 101.0}, edge_ai={"mode": "veto", "timeout_s": 0.01})
        ai = EdgeAIGate(p, lambda f: (time.sleep(0.2), 0.0)[1])
        _eng, out = self.run_day(prints, p, ai=ai)
        buys = [d for d in out if d.action == "BUY"]
        self.assertEqual(len(buys), 1)
        self.assertIn("AI_TIMEOUT_RULES_ONLY", buys[0].reason)

    def test_edge_ai_veto_blocks(self):
        prints, *_ = uptrend_then_dip()
        p = params(filter={"chase_max_pct": 0.3, "rsi_max": 101.0}, edge_ai={"mode": "veto"})
        _eng, out = self.run_day(prints, p, ai=EdgeAIGate(p, lambda f: 0.1))
        self.assertFalse([d for d in out if d.action == "BUY"])

    def test_position_blocks_and_candidate_expires(self):
        prints, *_ = uptrend_then_dip()
        _eng, out = self.run_day(prints, self.base(), blocked=lambda c: "POSITION")
        self.assertFalse([d for d in out if d.action == "BUY"])
        self.assertTrue(any(d.action == "EXPIRED" and d.reason == "POSITION" for d in out))

    def test_regime_down_blocks(self):
        prints, s, top, acml = uptrend_then_dip()
        down = [P(p.recv.timestamp() - DAY.timestamp(), 2 * 10000 - p.price, acml=p.acml_vol) for p in prints]
        _eng, out = self.run_day(down, self.base())
        self.assertFalse([d for d in out if d.action == "BUY"])


class ShadowRunnerTest(unittest.TestCase):
    def test_runner_reads_tick_file_and_writes_decisions(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            day = DAY.strftime("%Y%m%d")
            f = tmp / "ticks" / day / f"{CODE}.csv"
            f.parent.mkdir(parents=True)
            prints, s, _top, _a = uptrend_then_dip()
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write("recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,"
                         "askp1,bidp1,hour_cls_code,venue,session\n")
                for p in prints:
                    fh.write(f"{p.recv.isoformat(timespec='milliseconds')},{CODE},{day},{p.ts:%H%M%S},{p.price:.0f},{p.qty},"
                             f"{p.acml_vol},{p.side},0,0,{p.price + 10:.0f},{p.price:.0f},0,KRX,regular\n")
            logs = []
            runner = r028.EntryShadowRunner(tmp / "ticks", tmp / "out", day,
                                            params(filter={"rsi_max": 101.0}, entry={"valid_s": 100000.0}), logs.append)
            runner.set_watch([CODE])
            out = runner.step(DAY + timedelta(seconds=s + 1))       # everything visible at once: candidate appears
            out += runner.step(DAY + timedelta(seconds=s + 2))
            buys = [d for d in out if d.action == "BUY"]
            self.assertEqual(len(buys), 1, logs)
            rows = (tmp / "out" / f"entry_engine_{day}.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(rows[0])["action"], "BUY")
            runner.note_submit(CODE, DAY + timedelta(seconds=s + 3), buys[0].price, "TEST")
            runner.note_fill(CODE, DAY + timedelta(seconds=s + 5), buys[0].price)
            lat = json.loads((tmp / "out" / f"entry_latency_{day}.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual((lat["kind"], lat["outcome"]), ("order", "FILLED"))
            self.assertIn("t_decision->t_submit_s", lat)
            self.assertTrue(any("SHADOW BUY" in m for m in logs))
            # Codex review #4: a symbol that leaves the watchlist is dropped, not evaluated on stale prices
            runner.set_watch([])
            self.assertNotIn(CODE, runner.engine.codes)
            self.assertEqual(runner.step(DAY + timedelta(seconds=s + 6)), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_service_stop_processes_every_note_and_writes_open_records(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            day = DAY.strftime("%Y%m%d")
            holder = {}

            def factory():
                r = r028.EntryShadowRunner(tmp / "ticks", tmp / "out", day, params(), lambda m: None)
                orig = r.note_submit

                def flaky(code, at, price, reason):
                    if code == "BAD":
                        raise RuntimeError("bad note")
                    orig(code, at, price, reason)
                r.note_submit = flaky
                holder["r"] = r
                return r
            svc = r028.EntryShadowService(factory, interval_s=10.0)
            self.assertTrue(svc.ready.wait(5))
            svc.note_submit("BAD", DAY, 1.0)
            for i in range(600):
                svc.note_submit(f"{i:06d}", DAY + timedelta(hours=10), 100.0)
            self.assertTrue(svc.close(timeout_s=10.0))
            rows = [json.loads(x) for x in (tmp / "out" / f"entry_latency_{day}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 600)                                    # every note after the bad one
            self.assertEqual({r["outcome"] for r in rows}, {"STOP"})
            self.assertEqual(holder["r"].errors, 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_service_never_raises_on_the_trading_thread(self):
        svc = r028.EntryShadowService(lambda: (_ for _ in ()).throw(RuntimeError("boom")), log=lambda m: None)
        svc.set_watch(["1"])
        svc.note_submit("1", DAY, 1.0)
        svc.note_fill("1", DAY, 1.0)
        svc.close(1.0)
        self.assertFalse(svc.ready.is_set())


if __name__ == "__main__":
    unittest.main()
