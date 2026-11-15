# -*- coding: utf-8 -*-
"""r025 exit tick features: integrity (dup / out-of-order / gaps), rolling windows, RVOL profile, price source."""
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

from r009_tick_bars import Tick  # noqa: E402
from r025_exit_tick_feed import ExitTickWindow, RestQuote, RvolProfile, minute_volumes_from_bars  # noqa: E402

T0 = datetime(2026, 11, 16, 10, 0, 0)


class Feed:
    def __init__(self):
        self.w = ExitTickWindow("000001")
        self.acml = 1000

    def add(self, sec, price, qty, side=1, gap=0, recv_delay=0.0):
        self.acml += qty + gap
        ts = T0 + timedelta(seconds=sec)
        t = Tick("000001", ts, float(price), qty, self.acml, side, None, price + 10, price,
                 ts + timedelta(seconds=recv_delay))
        return self.w.ingest(t)


class Integrity(unittest.TestCase):
    def test_duplicate_and_late_dropped(self):
        f = Feed()
        f.add(0, 10000, 10)
        ts = T0 + timedelta(seconds=1)
        self.assertFalse(f.w.ingest(Tick("000001", ts, 10000.0, 10, f.acml, 1, None, 0, 0, ts)))      # dup
        self.assertFalse(f.w.ingest(Tick("000001", ts, 10000.0, 5, f.acml - 5, 1, None, 0, 0, ts)))   # late
        self.assertEqual(f.w.dropped, 2)

    def test_gap_volume_unknown_side(self):
        f = Feed()
        for s in range(60):
            f.add(s, 10000, 10, side=1)
        f.add(60, 10000, 10, side=1, gap=1000)            # 1,000 missed shares (side unknown)
        snap = f.w.snapshot(T0 + timedelta(seconds=60))
        self.assertGreater(snap.gap_volume_ratio, 0.6)
        self.assertEqual(snap.cttr_60s, 200.0)             # only buyer prints observed; gap not assigned

    def test_cttr_needs_five_prints_and_is_capped(self):
        f = Feed()
        for s in range(4):
            f.add(s, 10000, 10, side=5)
        self.assertIsNone(f.w.snapshot(T0 + timedelta(seconds=4)).cttr_60s)
        f.add(4, 10000, 10, side=1)
        self.assertAlmostEqual(f.w.snapshot(T0 + timedelta(seconds=4)).cttr_60s, 25.0)

    def test_not_visible_before_receive(self):
        f = Feed()
        f.add(0, 10000, 10)
        f.add(5, 10100, 10, recv_delay=3.0)
        s = f.w.snapshot(T0 + timedelta(seconds=6))
        self.assertEqual(s.price, 10000.0)                 # received at 8 s


class Windows(unittest.TestCase):
    def test_ret_3m_and_coverage(self):
        f = Feed()
        for s in range(0, 241, 5):
            f.add(s, 10000 + s, 10)
        snap = f.w.snapshot(T0 + timedelta(seconds=240))
        self.assertAlmostEqual(snap.ret_3m_pct, (10240 / 10060 - 1) * 100, places=9)
        self.assertEqual(snap.window_coverage, 1.0)
        f2 = Feed()
        f2.add(0, 10000, 10)
        s2 = f2.w.snapshot(T0 + timedelta(seconds=30))
        self.assertIsNone(s2.ret_3m_pct)
        self.assertAlmostEqual(s2.window_coverage, 0.5)

    def test_volume_burst_and_sell_share(self):
        f = Feed()
        for s in range(0, 570):
            f.add(s, 10000, 10, side=1)
        for s in range(570, 600):
            f.add(s, 9990, 50, side=5)
        snap = f.w.snapshot(T0 + timedelta(seconds=599))
        self.assertGreater(snap.vol_30s_over_10m, 3.0)
        self.assertAlmostEqual(snap.sell_share_30s, 1.0)
        self.assertIsNotNone(snap.cttr_60s_prev30)
        self.assertGreater(snap.cttr_60s_prev30, snap.cttr_60s)

    def test_high_since_prev_resets(self):
        f = Feed()
        f.add(0, 10000, 10)
        f.add(1, 10300, 10)
        f.add(2, 10100, 10)
        s = f.w.snapshot(T0 + timedelta(seconds=2))
        self.assertEqual((s.price, s.high_since_prev), (10100.0, 10300.0))
        self.assertIsNone(f.w.snapshot(T0 + timedelta(seconds=3)).high_since_prev)

    def test_atr_from_completed_bars(self):
        f = Feed()
        for m in range(8):
            base = 10000 + m * 10
            f.add(m * 60 + 1, base, 10)
            f.add(m * 60 + 30, base + 50, 10)
            f.add(m * 60 + 59, base, 10)
        snap = f.w.snapshot(T0 + timedelta(seconds=8 * 60 + 5))
        self.assertIsNotNone(snap.atr1m_pct)
        # TR = max(h-l 50, |h - prev close| 60, |l - prev close| 10) = 60 (closes step +10 per bar)
        self.assertAlmostEqual(snap.atr1m_pct, 60 / 10070 * 100, places=6)


class PriceSource(unittest.TestCase):
    def test_rest_used_when_ticks_stale(self):
        f = Feed()
        f.add(0, 10000, 10)
        now = T0 + timedelta(seconds=30)
        rest = RestQuote(ts=now - timedelta(seconds=2), price=10050, bid1=10040, ask1=10050, vi_active=True,
                         vi_expected_price=10000, upper_limit=13000, lower_limit=7000)
        s = f.w.snapshot(now, rest)
        self.assertEqual((s.price, s.bid1, s.last_tick_age_s, s.rest_age_s), (10050, 10040, 30.0, 2.0))
        self.assertTrue(s.vi_active)
        self.assertEqual((s.upper_limit, s.lower_limit), (13000, 7000))

    def test_no_data_at_all(self):
        s = ExitTickWindow("x").snapshot(T0)
        self.assertGreaterEqual(min(s.last_tick_age_s, s.rest_age_s), 60)


class Rvol(unittest.TestCase):
    def test_minute_label_right(self):
        rows = [(datetime(2026, 11, 13, 9, 1), 100), (datetime(2026, 11, 13, 9, 2), 50)]
        self.assertEqual(minute_volumes_from_bars(rows), {540: 100.0, 541: 50.0})

    def test_profile_sources(self):
        day = {k: 100.0 for k in range(540, 930)}
        self.assertEqual(RvolProfile.build([day] * 12).source, "slot")
        shape = {k: (3.0 if k < 570 else 1.0) for k in range(540, 930)}
        p = RvolProfile.build([day] * 6, market_shape=shape)
        self.assertEqual(p.source, "market_shape")
        self.assertGreater(p.per_minute[545], p.per_minute[700])         # U shape kept
        self.assertAlmostEqual(sum(p.per_minute.values()), 39000.0)
        self.assertIsNone(RvolProfile.build([day] * 4, market_shape=shape))
        self.assertIsNone(RvolProfile.build([{}] * 12))                  # halted / empty days excluded

    def test_missing_minute_counts_as_zero(self):
        days = [{600: 100.0}] * 5 + [{601: 100.0}] * 5
        p = RvolProfile.build(days)
        self.assertEqual((p.per_minute[600], p.per_minute[601]), (50.0, 50.0))

    def test_expected_overlap_and_rvol(self):
        p = RvolProfile({k: 60.0 for k in range(540, 930)}, "slot", 20)
        self.assertAlmostEqual(p.expected(datetime(2026, 11, 16, 10, 0, 30)), 300.0)
        f = Feed()
        for s in range(0, 301):
            f.add(s, 10000, 3)
        snap = f.w.snapshot(T0 + timedelta(seconds=300), profile=p)
        self.assertAlmostEqual(snap.rvol_5m, 900 / 300.0, places=6)     # prints in (0, 300]: 300 x 3


class CodeReviewFixes(unittest.TestCase):
    def test_3_unreceived_print_not_in_peak_age_or_bars(self):
        f = Feed()
        f.add(0, 10000, 10)
        f.add(5, 11000, 10, recv_delay=3.0)                 # executed at 5 s, received at 8 s
        s6 = f.w.snapshot(T0 + timedelta(seconds=6))
        self.assertEqual((s6.price, s6.high_since_prev), (10000.0, 10000.0))
        self.assertGreaterEqual(s6.last_tick_age_s, 0.0)
        self.assertEqual(f.w._bars[T0][1], 10000.0)          # bar high not yet touched by the late print
        s8 = f.w.snapshot(T0 + timedelta(seconds=8))
        self.assertEqual((s8.price, s8.high_since_prev), (11000.0, 11000.0))   # delivered once, when received

    def test_7_unknown_side_tracked_and_not_counted_as_strength(self):
        f = Feed()
        for s in range(60):
            f.add(s, 10000, 5, side=1)
        f.add(60, 10000, 3000, side=0)                       # big print with unknown side
        snap = f.w.snapshot(T0 + timedelta(seconds=60))
        self.assertGreater(snap.unknown_side_ratio, 0.9)
        self.assertEqual(snap.gap_volume_ratio, 0.0)

    def test_10_rvol_uses_most_recent_20_days(self):
        old = [{600: 100.0}] * 80
        recent = [{600: 1000.0}] * 20
        p = RvolProfile.build(old + recent)
        self.assertEqual((p.per_minute[600], p.days), (1000.0, 20))


class ReReviewFixes(unittest.TestCase):
    def test_rr1_receive_order_reversal_keeps_latest_execution(self):
        f = Feed()
        f.add(0, 10000, 10)
        f.add(5, 10100, 10, recv_delay=5.0)                  # executed 5 s, received 10 s
        f.add(6, 10200, 10)                                  # executed 6 s, received 6 s
        self.assertEqual(f.w.snapshot(T0 + timedelta(seconds=6)).price, 10200.0)
        s10 = f.w.snapshot(T0 + timedelta(seconds=10))
        self.assertEqual(s10.price, 10200.0)                 # the late 5 s print must not become "current"
        self.assertEqual([p.ts.second for p in f.w._prints], [0, 5, 6])
        self.assertEqual(f.w._bars[T0][3], 10200.0)          # bar close = latest execution

    def test_rr5_staging_is_bounded(self):
        f = Feed()
        f.w.MAX_STAGED = 10
        for s in range(30):
            f.add(s, 10000, 1, recv_delay=3600.0)            # all far in the future
        self.assertLessEqual(len(f.w._staged), 10)
        self.assertEqual(f.w.staged_overflow, 20)

    def test_rr_coverage_is_observation_time(self):
        f = Feed()
        f.add(0, 10000, 10, recv_delay=300.0)                # a 5-minute-old print received now
        s = f.w.snapshot(T0 + timedelta(seconds=300))
        self.assertEqual(s.window_coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
