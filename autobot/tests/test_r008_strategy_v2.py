# -*- coding: utf-8 -*-
"""Unit tests for live_trading/r008_strategy_v2.py (run: python -m unittest discover -s tests)."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

from r008_strategy_v2 import (  # noqa: E402
    DayRisk, EntryState, V2Config, V2Position, build_5m, completed_5m, daily_features, evaluate_daily_trend,
    judge_daily_row, krx_tick_size, market_breadth_ok, prepare_1m, round_to_tick, size_position, spread_ok,
    step_entry, step_exit,
)

DAY = "2026-10-20"


def _m1_from_closes(closes, vols=None, start="09:00", highs=None, lows=None, opens=None) -> pd.DataFrame:
    """1m bars labelled by END time; open = previous close unless given."""
    n = len(closes)
    idx = pd.date_range(f"{DAY} {start}", periods=n, freq="1min") + pd.Timedelta(minutes=1)
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else np.r_[c[0], c[:-1]]
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, c) + 10
    lo = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, c) - 10
    v = np.asarray(vols, dtype=float) if vols is not None else np.full(n, 1000.0)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v, "amount": c * v}, index=idx)


def _random_session(seed=1, n=200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 10000 + np.cumsum(rng.normal(3, 25, n)).round(-1)
    vols = rng.integers(200, 3000, n).astype(float)
    return _m1_from_closes(closes, vols)


class TestTickSize(unittest.TestCase):
    def test_table(self):
        self.assertEqual(krx_tick_size(1999), 1)
        self.assertEqual(krx_tick_size(2000), 5)
        self.assertEqual(krx_tick_size(19990), 10)
        self.assertEqual(krx_tick_size(20000), 50)
        self.assertEqual(krx_tick_size(199900), 100)
        self.assertEqual(krx_tick_size(200000), 500)
        self.assertEqual(krx_tick_size(500000), 1000)
        self.assertEqual(round_to_tick(10234, "down"), 10230)
        self.assertEqual(round_to_tick(10231, "up"), 10240)


class TestDailyFilter(unittest.TestCase):
    def _daily(self, closes):
        n = len(closes)
        c = np.asarray(closes, dtype=float)
        return pd.DataFrame({"open": c, "high": c * 1.02, "low": c * 0.98, "close": c,
                             "volume": np.full(n, 1e6), "amount": c * 1e6})

    def test_uptrend_passes_downtrend_fails(self):
        cfg = V2Config(daily_min_avg_amount_krw=1e9)
        up = self._daily(np.linspace(10000, 13000, 120))
        self.assertTrue(evaluate_daily_trend("X", up, cfg).ok)
        down = self._daily(np.linspace(13000, 10000, 120))
        r = evaluate_daily_trend("Y", down, cfg)
        self.assertFalse(r.ok)
        self.assertIn("NOT_STACKED", r.reason)

    def test_short_history_and_liquidity(self):
        cfg = V2Config()
        r = evaluate_daily_trend("X", self._daily(np.linspace(10000, 11000, 40)), cfg)
        self.assertFalse(r.ok)
        self.assertTrue(r.reason.startswith("DAILY_SHORT_HISTORY"))
        r = evaluate_daily_trend("X", self._daily(np.linspace(10000, 13000, 120)), replace(cfg, daily_min_avg_amount_krw=1e15))
        self.assertEqual(r.reason, "DAILY_ILLIQUID")

    def test_features_are_causal(self):
        """Row i of daily_features must not change when later rows are appended."""
        cfg = V2Config()
        d = self._daily(10000 + np.cumsum(np.random.default_rng(3).normal(10, 80, 150)))
        full = daily_features(d, cfg)
        part = daily_features(d.iloc[:100], cfg)
        pd.testing.assert_frame_equal(full.iloc[:100], part)
        self.assertEqual(judge_daily_row("X", full.iloc[99], cfg).reason, judge_daily_row("X", part.iloc[-1], cfg).reason)


class TestNoLookAhead(unittest.TestCase):
    def test_indicator_slicing_equals_recompute(self):
        cfg = V2Config()
        m1 = _random_session(7)
        p_full = prepare_1m(m1, cfg)
        m5_full = build_5m(p_full, cfg)
        for k in (12, 33, 57, 101, 150, 199):
            p_part = prepare_1m(m1.iloc[:k], cfg)
            pd.testing.assert_frame_equal(p_full.iloc[:k], p_part)
            last_end = m1.index[k - 1]
            m5_part = completed_5m(build_5m(p_part, cfg), last_end)
            m5_cut = completed_5m(m5_full, last_end)
            pd.testing.assert_frame_equal(m5_cut, m5_part)
            # every used 5m bucket must be complete at decision time
            self.assertTrue((m5_cut.index <= last_end).all())

    def test_entry_engine_same_on_full_vs_recomputed(self):
        """Signals produced from a precomputed full-day frame sliced at t equal those from recomputing at t."""
        cfg = replace(V2Config(), m1_trigger_vol_ratio=0.8, pb_impulse_min_atr=0.3, pb_min_depth_atr=0.1, entry_start=pd.Timestamp("09:00").time())
        m1 = _random_session(11, 240)
        p_full = prepare_1m(m1, cfg)
        m5_full = build_5m(p_full, cfg)
        st_a, st_b = EntryState(), EntryState()
        sig_a, sig_b = [], []
        for i in range(2, len(m1) + 1):
            a = step_entry("X", st_a, p_full.iloc[:i], m5_full, cfg, False)
            p_part = prepare_1m(m1.iloc[:i], cfg)
            b = step_entry("X", st_b, p_part, build_5m(p_part, cfg), cfg, False)
            sig_a.append(None if a is None else (a.ts, a.setup, a.stop))
            sig_b.append(None if b is None else (b.ts, b.setup, b.stop))
        self.assertEqual(sig_a, sig_b)


class TestSetups(unittest.TestCase):
    def _run(self, m1, cfg):
        p = prepare_1m(m1, cfg)
        m5 = build_5m(p, cfg)
        st = EntryState()
        sigs = []
        for i in range(2, len(p) + 1):
            s = step_entry("X", st, p.iloc[:i], m5, cfg, False)
            if s:
                sigs.append(s)
        return sigs, st

    def test_pullback_signal(self):
        cfg = replace(V2Config(), enable_breakout_retest=False, stop_min_pct=0.3, pb_max_depth_atr=5.0)
        # 60 min steady climb 10000 -> 10600, 12 min dip back toward EMA/VWAP, then a strong volume rebound bar
        up = list(np.linspace(10000, 10600, 60).round(-1))
        dip = list(np.linspace(10580, 10340, 12).round(-1))
        closes = up + dip + [10420]
        vols = [1000] * 72 + [4000]
        sigs, _st = self._run(_m1_from_closes(closes, vols), cfg)
        self.assertTrue(sigs, "expected a pullback signal")
        s = sigs[0]
        self.assertEqual(s.setup, "PULLBACK")
        self.assertLess(s.stop, s.ref_price)

    def test_no_signal_without_volume(self):
        cfg = replace(V2Config(), enable_breakout_retest=False, stop_min_pct=0.3, pb_max_depth_atr=5.0)
        up = list(np.linspace(10000, 10600, 60).round(-1))
        dip = list(np.linspace(10580, 10340, 12).round(-1))
        sigs, st = self._run(_m1_from_closes(up + dip + [10420], [1000] * 73), cfg)
        self.assertFalse(sigs)
        self.assertEqual(st.last_reject, "PB_TRIG_VOLUME_WEAK")

    def test_too_deep_pullback_is_not_armed(self):
        cfg = replace(V2Config(), enable_breakout_retest=False, stop_min_pct=0.3, pb_max_depth_atr=3.0)
        up = list(np.linspace(10000, 10600, 60).round(-1))
        dip = list(np.linspace(10580, 10340, 12).round(-1))
        sigs, st = self._run(_m1_from_closes(up + dip + [10420], [1000] * 72 + [4000]), cfg)
        self.assertFalse(sigs)

    def test_breakout_retest_signal(self):
        cfg = replace(V2Config(), enable_pullback=False, stop_min_pct=0.3)
        # opening range 09:00-09:30 capped at ~10100, then a 5m volume breakout to 10200, retest 10100, rebound
        rng_closes = [10000 + (i % 5) * 20 for i in range(30)]
        highs = [c + 10 for c in rng_closes]
        highs[10] = 10100
        brk = [10120, 10150, 10180, 10200, 10220]
        retest = [10180, 10150, 10120, 10105, 10110]
        closes = rng_closes + [10050] * 5 + brk + retest + [10170]
        vols = [1000] * 35 + [6000] * 5 + [800] * 5 + [3000]
        h = highs + [c + 10 for c in closes[30:]]
        lows = [c - 10 for c in closes]
        sigs, st = self._run(_m1_from_closes(closes, vols, highs=h, lows=lows), cfg)
        self.assertTrue(sigs, f"expected breakout-retest signal, state={st.bo_state} level={st.bo_level} reject={st.last_reject}")
        self.assertEqual(sigs[0].setup, "BREAKOUT_RETEST")

    def test_market_gate_withholds_signal_but_keeps_state(self):
        cfg = replace(V2Config(), enable_breakout_retest=False, stop_min_pct=0.3, pb_max_depth_atr=5.0)
        up = list(np.linspace(10000, 10600, 60).round(-1))
        dip = list(np.linspace(10580, 10340, 12).round(-1))
        m1 = _m1_from_closes(up + dip + [10420], [1000] * 72 + [4000])
        p = prepare_1m(m1, cfg)
        m5 = build_5m(p, cfg)
        st = EntryState()
        out = [step_entry("X", st, p.iloc[:i], m5, cfg, False, market_ok=False) for i in range(2, len(p) + 1)]
        self.assertFalse(any(out))
        self.assertEqual(st.trades_taken, 0)


class TestExit(unittest.TestCase):
    def _m5(self, ts, close=10000, ema=9900, vwap=9900, atr=50):
        return pd.DataFrame({"close": [close], "EMA": [ema], "VWAP": [vwap], "ATR": [atr]}, index=[ts])

    def test_stop_gap_fills_at_open(self):
        cfg = V2Config()
        t0 = pd.Timestamp(f"{DAY} 10:00")
        p = V2Position("X", "PULLBACK", t0, 10000, 100, 9900, 100)
        ts = t0 + pd.Timedelta(minutes=2)
        acts = step_exit(p, {"open": 9850, "high": 9870, "low": 9800, "close": 9820}, ts, self._m5(ts), cfg)
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0].price, 9850)
        self.assertEqual(acts[0].reason, "STOP_INITIAL")
        self.assertEqual(p.remaining, 0)

    def test_tp1_then_breakeven(self):
        cfg = replace(V2Config(), trail_start_r=99)
        t0 = pd.Timestamp(f"{DAY} 10:00")
        p = V2Position("X", "PULLBACK", t0, 10000, 100, 9900, 100)
        ts = t0 + pd.Timedelta(minutes=1, seconds=30)
        acts = step_exit(p, {"open": 10050, "high": 10160, "low": 10040, "close": 10150}, ts, self._m5(t0), cfg)
        self.assertEqual([a.reason for a in acts], ["TP1"])
        self.assertEqual(acts[0].qty, 50)
        self.assertEqual(p.remaining, 50)
        self.assertTrue(p.be_moved)
        self.assertGreater(p.stop, p.entry_price)  # breakeven covers round-trip costs

    def test_stop_checked_before_target_in_same_bar(self):
        cfg = V2Config()
        t0 = pd.Timestamp(f"{DAY} 10:00")
        p = V2Position("X", "PULLBACK", t0, 10000, 100, 9900, 100)
        ts = t0 + pd.Timedelta(minutes=1, seconds=30)
        acts = step_exit(p, {"open": 10000, "high": 10300, "low": 9890, "close": 10200}, ts, self._m5(t0), cfg)
        self.assertEqual(acts[0].reason, "STOP_INITIAL")

    def test_trend_break_and_flatten(self):
        cfg = V2Config()
        t0 = pd.Timestamp(f"{DAY} 10:00")
        p = V2Position("X", "PULLBACK", t0, 10000, 100, 9800, 200)
        ts = pd.Timestamp(f"{DAY} 10:05")
        acts = step_exit(p, {"open": 9990, "high": 10000, "low": 9950, "close": 9960}, ts,
                         self._m5(ts, close=9960, ema=9980, vwap=9970), cfg)
        self.assertEqual(acts[0].reason, "TREND_BREAK_5M")
        p2 = V2Position("X", "PULLBACK", t0, 10000, 100, 9800, 200)
        ts2 = pd.Timestamp(f"{DAY} 15:15")
        acts = step_exit(p2, {"open": 10010, "high": 10020, "low": 10000, "close": 10010}, ts2, self._m5(t0), replace(cfg, time_stop_min=0))
        self.assertEqual(acts[0].reason, "EOD_FLATTEN")


class TestRisk(unittest.TestCase):
    def test_size_by_risk_and_cap(self):
        cfg = V2Config(account_equity_krw=10_000_000, risk_per_trade_pct=0.5, max_position_pct=20)
        q = size_position(10000, 9900, 10_000_000, cfg)
        # 50,000 KRW risk / (100 + ~23 cost) ~= 406 shares, but notional cap 2,000,000 / 10,000 = 200
        self.assertEqual(q, 200)
        q2 = size_position(10000, 9700, 10_000_000, cfg)
        self.assertLessEqual(q2 * (300 + 10000 * 0.0023), 50_000 + 1)
        self.assertEqual(size_position(10000, 10000, 1e7, cfg), 0)

    def test_daily_loss_limit_latches(self):
        cfg = V2Config(daily_loss_limit_pct=1.5, max_open_positions=3)
        r = DayRisk(equity=10_000_000)
        self.assertTrue(r.can_open(0, 0, cfg)[0])
        r.realized_pnl = -100_000
        self.assertFalse(r.can_open(0, -60_000, cfg)[0])
        r.realized_pnl = 0  # latched for the rest of the day
        self.assertEqual(r.can_open(0, 0, cfg), (False, "DAILY_LOSS_LIMIT"))
        self.assertEqual(DayRisk(equity=1e7).can_open(3, 0, cfg), (False, "MAX_OPEN_POSITIONS"))

    def test_spread_and_breadth(self):
        cfg = V2Config(max_spread_ticks=2)
        self.assertTrue(spread_ok(10000, 10020, cfg)[0])
        self.assertFalse(spread_ok(10000, 10030, cfg)[0])
        self.assertFalse(spread_ok(None, 10030, cfg)[0])
        self.assertEqual(market_breadth_ok(3, 10, V2Config(market_breadth_min_pct=50))[0], False)
        self.assertEqual(market_breadth_ok(6, 10, V2Config(market_breadth_min_pct=50))[0], True)
        self.assertEqual(market_breadth_ok(0, 10, V2Config(market_breadth_min_pct=0))[0], True)


if __name__ == "__main__":
    unittest.main()


class TestBreakoutRetestUndercut(unittest.TestCase):
    def test_first_retest_bar_undercut_fails_setup(self):
        cfg = replace(V2Config(), enable_pullback=False, stop_min_pct=0.3)
        rng_closes = [10000 + (i % 5) * 20 for i in range(30)]
        highs = [c + 10 for c in rng_closes]
        highs[10] = 10100
        brk = [10120, 10150, 10180, 10200, 10220]
        closes = rng_closes + [10050] * 5 + brk + [10170]
        vols = [1000] * 35 + [6000] * 5 + [5000]
        h = highs + [c + 10 for c in closes[30:]]
        lows = [c - 10 for c in closes]
        lows[-1] = 10000  # level 10100, undercut limit 0.5% -> 10049.5; low 10000 breaks it, close recovers
        p = prepare_1m(_m1_from_closes(closes, vols, highs=h, lows=lows), cfg)
        m5 = build_5m(p, cfg)
        st = EntryState()
        sigs = [s for i in range(2, len(p) + 1) if (s := step_entry("X", st, p.iloc[:i], m5, cfg, False))]
        self.assertFalse(sigs)
        self.assertEqual(st.bo_state, "FAILED")


class TestSignalGate(unittest.TestCase):
    def test_rejected_candidate_keeps_setup_armed(self):
        cfg = replace(V2Config(), enable_breakout_retest=False, stop_min_pct=0.3, pb_max_depth_atr=5.0)
        up = list(np.linspace(10000, 10600, 60).round(-1))
        dip = list(np.linspace(10580, 10340, 12).round(-1))
        closes = up + dip + [10420, 10380, 10470]
        vols = [1000] * 72 + [4000, 1000, 5000]
        p = prepare_1m(_m1_from_closes(closes, vols), cfg)
        m5 = build_5m(p, cfg)
        st = EntryState()
        calls = []

        def gate(sig):
            calls.append(sig.ts)
            return (len(calls) > 1), "TICK_TEST_REJECT"

        sigs = [s for i in range(2, len(p) + 1) if (s := step_entry("X", st, p.iloc[:i], m5, cfg, False, signal_gate=gate))]
        self.assertGreaterEqual(len(calls), 2, "a second candidate must be possible after the first rejection")
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].ts, calls[1])
        self.assertEqual(st.trades_taken, 1)
