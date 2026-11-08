# -*- coding: utf-8 -*-
"""Unit tests for live_trading/r009_tick_bars.py and the g011 frame parser."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

from r009_tick_bars import (  # noqa: E402
    TICK_FILE_COLUMNS, Tick, TickBarBuilder, build_tick_bars, iter_tick_file, tick_momentum,
)

T0 = datetime(2026, 10, 23, 9, 0, 0)


def ticks(n, start_px=10000, step=10, qty=10, side=1, t0=T0, acml0=0, cnt0=0):
    out = []
    acml, cnt = acml0, cnt0
    for i in range(n):
        acml += qty
        cnt += 1
        out.append(Tick("000001", t0 + timedelta(seconds=i), start_px + step * i, qty, acml, side, cnt))
    return out


class TestTickBarBuilder(unittest.TestCase):
    def test_bar_boundaries_and_ohlcv(self):
        b = TickBarBuilder("000001", 5)
        done = []
        for t in ticks(12):
            done += b.ingest(t)
        self.assertEqual(len(done), 2)
        self.assertEqual((done[0].open, done[0].close, done[0].n_ticks, done[0].volume), (10000, 10040, 5, 50))
        self.assertEqual(done[1].open, 10050)
        self.assertEqual(done[0].buy_vol, 50)

    def test_duplicate_dropped(self):
        b = TickBarBuilder("000001", 3)
        ts = ticks(3)
        for t in [ts[0], ts[0], ts[1], ts[1], ts[2]]:
            b.ingest(t)
        self.assertEqual(b.stats.duplicates, 2)
        self.assertEqual(len(b.bars), 1)
        self.assertEqual(b.bars[0].volume, 30)

    def test_out_of_order_dropped_and_gap_counted(self):
        b = TickBarBuilder("000001", 100)
        ts = ticks(4)
        for t in [ts[0], ts[2], ts[1], ts[3]]:  # ts[1] arrives late
            b.ingest(t)
        self.assertEqual(b.stats.out_of_order, 1)
        # ts[2] arrived before ts[1]: one print missing at that moment (count delta 2), its 10 shares are gap volume
        self.assertEqual(b.stats.gap_ticks, 1)
        self.assertEqual(b.stats.gap_volume, 10)
        self.assertEqual(b._cur.n_ticks, 4)  # missed print still counted -> aligned with exchange tape
        self.assertEqual(b._cur.volume, 40)

    def test_gap_without_counts_uses_volume(self):
        b = TickBarBuilder("000001", 100)
        ts = ticks(3)
        for t in ts:
            t.cum_count = None
        b.ingest(ts[0])
        b.ingest(ts[2])
        self.assertEqual(b.stats.gap_events, 1)
        self.assertEqual(b.stats.gap_volume, 10)
        self.assertEqual(b.stats.gap_ticks, 0)  # number of missed prints is unknown without counts
        self.assertTrue(b._cur.invalid)         # alignment unknown from here on

    def test_unreliable_counts_fall_back(self):
        b = TickBarBuilder("000001", 100)
        ts = ticks(3)
        ts[1].cum_count = ts[0].cum_count  # count did not advance while volume did
        ts[2].cum_count = ts[0].cum_count
        for t in ts:
            b.ingest(t)
        self.assertFalse(b.count_reliable)
        self.assertEqual(b.stats.gap_ticks, 0)

    def test_session_reset(self):
        b = TickBarBuilder("000001", 3)
        for t in ticks(2):
            b.ingest(t)
        nxt = ticks(3, t0=T0 + timedelta(days=1))
        for t in nxt:
            b.ingest(t)
        self.assertEqual(len(b.bars), 1)
        self.assertEqual(b.bars[0].start_ts.date(), (T0 + timedelta(days=1)).date())

    def test_bars_until_is_causal(self):
        b = build_tick_bars(ticks(30), "000001", 10)
        self.assertEqual(len(b.bars_until(T0 + timedelta(seconds=9))), 1)
        self.assertEqual(len(b.bars_until(T0 + timedelta(seconds=8))), 0)


class TestMomentum(unittest.TestCase):
    def test_ok_and_rejects(self):
        b = build_tick_bars(ticks(60), "000001", 10)
        now = T0 + timedelta(seconds=60)
        m = tick_momentum(b.bars, now, 5, 3, 0.55, 0.05, 120)
        self.assertTrue(m.ok, m.reason)
        down = build_tick_bars(ticks(60, step=-10, side=5), "000001", 10)
        m2 = tick_momentum(down.bars, now, 5, 3, 0.55, 0.05, 120)
        self.assertFalse(m2.ok)
        self.assertTrue(m2.reason.startswith("TICK_UP_BARS"))

    def test_abstain_reasons(self):
        b = build_tick_bars(ticks(20), "000001", 10)
        self.assertEqual(tick_momentum(b.bars, T0 + timedelta(seconds=30), 5, 3, 0.5, 0.05, 120).reason, "TICK_ABSTAIN_FEW_BARS_2")
        b2 = build_tick_bars(ticks(60), "000001", 10)
        self.assertEqual(tick_momentum(b2.bars, T0 + timedelta(seconds=600), 5, 3, 0.5, 0.05, 120).reason, "TICK_ABSTAIN_STALE")
        gappy = [t for i, t in enumerate(ticks(80)) if i % 4 != 0]
        b3 = build_tick_bars(gappy, "000001", 10)
        # every 4th print missing: some gaps cross boundaries (invalid bars) - either way the filter abstains
        self.assertIn(tick_momentum(b3.bars, T0 + timedelta(seconds=80), 5, 3, 0.5, 0.05, 120).reason,
                      ("TICK_ABSTAIN_GAPS", "TICK_ABSTAIN_INVALID_BAR"))
        b4 = build_tick_bars([t for i, t in enumerate(ticks(80)) if i % 10 != 5], "000001", 10)  # gaps inside bars only
        self.assertEqual(tick_momentum(b4.bars, T0 + timedelta(seconds=80), 5, 3, 0.5, 0.05, 120).reason, "TICK_ABSTAIN_GAPS")


class TestFileRoundTrip(unittest.TestCase):
    def test_collector_parse_and_file_reader(self):
        import g011_tick_collector as g
        fields = []
        for k, (px, q, ac, s, b) in enumerate([(100, 5, 5, 0, 1), (101, 3, 8, 0, 2), (100, 2, 10, 1, 2)]):
            f = [""] * g.NCOL
            f[g._IDX["MKSC_SHRN_ISCD"]] = "005930"
            f[g._IDX["STCK_CNTG_HOUR"]] = f"09000{k}"
            f[g._IDX["STCK_PRPR"]] = str(px)
            f[g._IDX["CNTG_VOL"]] = str(q)
            f[g._IDX["ACML_VOL"]] = str(ac)
            f[g._IDX["CCLD_DVSN"]] = "1" if s == 0 else "5"
            f[g._IDX["SELN_CNTG_CSNU"]] = str(s)
            f[g._IDX["SHNU_CNTG_CSNU"]] = str(b)
            f[g._IDX["BSOP_DATE"]] = "20261023"
            fields += f
        recs = g.parse_frame(f"0|H0STCNT0|003|" + "^".join(fields))
        self.assertEqual(len(recs), 3)
        self.assertEqual(g.parse_frame("0|H0STASP0|001|x"), [])  # other TR ignored
        with tempfile.TemporaryDirectory() as d:
            w = g.TickWriter(Path(d))
            for r in recs + [recs[1]]:  # duplicate delivery
                w.write("2026-10-23T09:00:00.000", r)
            w.close()
            path = Path(d) / "005930.csv"
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(next(csv.reader(fh)), TICK_FILE_COLUMNS)
            b = build_tick_bars(iter_tick_file(path), "005930", 2)
            # an old record re-delivered after a newer one is classified as late (out_of_order); either way it is dropped
            self.assertEqual(b.stats.duplicates + b.stats.out_of_order, 1)
            self.assertEqual(b.stats.accepted, 3)
            self.assertEqual(len(b.bars), 1)


if __name__ == "__main__":
    unittest.main()


class TestBoundaryCrossingGaps(unittest.TestCase):
    def test_missed_prints_cross_boundary_keep_alignment(self):
        b = TickBarBuilder("000001", 10)
        ts = ticks(30)
        for t in ts[:9]:
            b.ingest(t)                       # 9 prints in the current bar
        done = b.ingest(ts[11])               # prints #10 and #11 missed, #12 received
        self.assertEqual(len(done), 1)
        self.assertTrue(done[0].invalid)      # bar closed by a missed print -> close unknown
        self.assertEqual(done[0].n_ticks, 10)
        self.assertEqual(b._cur.n_ticks, 2)   # #11 (missed) + #12 -> next boundary still at exchange print #20
        self.assertTrue(b._cur.invalid)
        for t in ts[12:20]:
            b.ingest(t)
        self.assertEqual(len(b.bars), 2)
        self.assertEqual(b.bars[1].close, ts[19].price)  # print #20 closes bar 2 exactly

    def test_multi_bar_gap_emits_placeholders(self):
        b = TickBarBuilder("000001", 10)
        ts = ticks(40)
        b.ingest(ts[0])
        done = b.ingest(ts[34])  # 33 prints missed
        self.assertEqual(len(done), 3)
        self.assertTrue(all(x.invalid for x in done))
        self.assertEqual(b._cur.n_ticks, 5)

    def test_momentum_abstains_on_invalid_and_uses_recv_time(self):
        ts = ticks(60)
        for t in ts:
            t.recv = t.ts + timedelta(seconds=3)  # every print arrives 3s late
        b = build_tick_bars(ts, "000001", 10)
        last_end = b.bars[-1].end_ts
        # at the exchange time of the last print it has not been received yet
        self.assertEqual(len(b.bars_until(last_end)), len(b.bars) - 1)
        self.assertEqual(len(b.bars_until(last_end + timedelta(seconds=3))), len(b.bars))
        b.bars[-2].invalid = True
        m = tick_momentum(b.bars, last_end + timedelta(seconds=3), 5, 3, 0.5, 0.05, 120)
        self.assertEqual(m.reason, "TICK_ABSTAIN_INVALID_BAR")


class TestRound2TickFixes(unittest.TestCase):
    def test_missing_count_field_does_not_create_false_gap(self):
        ts = ticks(3)
        ts[1].cum_count = None  # 1 -> None -> 3, nothing actually missed
        b = TickBarBuilder("000001", 3)
        for t in ts:
            b.ingest(t)
        self.assertEqual(b.stats.gap_ticks, 0)
        self.assertEqual(len(b.bars), 1)
        self.assertEqual(b.bars[0].n_ticks, 3)

    def test_previous_session_bars_do_not_vote(self):
        b = TickBarBuilder("000001", 10)
        for t in ticks(40, t0=T0 - timedelta(days=1) + timedelta(hours=6)):  # yesterday 15:00, 4 up bars
            b.ingest(t)
        for t in ticks(10, t0=T0):  # today: 1 bar only
            b.ingest(t)
        m = tick_momentum(b.bars, T0 + timedelta(seconds=20), 5, 3, 0.5, 0.05, 120)
        self.assertEqual(m.reason, "TICK_ABSTAIN_FEW_BARS_1")

    def test_count_reliability_resets_each_session(self):
        b = TickBarBuilder("000001", 100)
        ts = ticks(3)
        ts[1].cum_count = ts[0].cum_count
        for t in ts:
            b.ingest(t)
        self.assertFalse(b.count_reliable)
        b.ingest(ticks(1, t0=T0 + timedelta(days=1))[0])
        self.assertTrue(b.count_reliable)

    def test_gap_volume_attribution_at_boundary(self):
        ts = ticks(4)  # 10 shares each
        b = TickBarBuilder("000001", 3)
        b.ingest(ts[0])
        b.ingest(ts[1])
        b.ingest(ts[3])  # #3 missed and it is exactly the boundary print
        self.assertEqual(b.bars[0].volume, 30)
        self.assertTrue(b.bars[0].invalid)
        self.assertEqual(b._cur.volume, 10)
        self.assertFalse(b._cur.invalid)
        b2 = TickBarBuilder("000001", 3)
        for t in [ts[0], ts[1]]:
            b2.ingest(t)
        t5 = ticks(5)
        b2.ingest(t5[4])  # #3,#4 missed: #3 closes bar 1, #4 starts bar 2 -> 20 missed shares split unknown
        self.assertEqual(b2.bars[0].volume, 20)
        self.assertEqual(b2._cur.unattributed_volume, 20)  # spans two bars -> kept out of both volumes
        self.assertEqual(b2._cur.volume, 10)
        self.assertTrue(b2._cur.invalid)


class TestRound3Alignment(unittest.TestCase):
    def test_unknown_gap_invalidates_until_resync(self):
        """Codex r3: #1 recv, #2-#4 missed, #5 recv WITHOUT counts, #6.. recv with counts (n=5)."""
        ts = ticks(30)
        b = TickBarBuilder("000001", 5)
        b.ingest(ts[0])
        ts[4].cum_count = None
        b.ingest(ts[4])           # volume gap, count unknown -> unaligned
        for t in ts[5:30]:
            b.ingest(t)
        # absolute boundaries: bars end at prints #5, #10, #15, ... ; resync happened at print #6
        valid = [x for x in b.bars if not x.invalid]
        self.assertTrue(all(x.close == ts[(i + 2) * 5 - 1].price for i, x in enumerate(valid)), [x.close for x in valid])
        self.assertTrue(all(x.invalid for x in b.bars[: len(b.bars) - len(valid)]))
        m = tick_momentum(b.bars, ts[29].ts, 5, 3, 0.5, 0.05, 120)
        self.assertTrue(m.ok, m.reason)  # five clean bars after resync

    def test_unknown_gap_bars_do_not_vote(self):
        ts = ticks(12)
        for t in ts:
            t.cum_count = None
        b = TickBarBuilder("000001", 3)
        b.ingest(ts[0])
        for t in ts[2:]:
            b.ingest(t)           # never resyncs (no counts at all)
        self.assertTrue(all(x.invalid for x in b.bars))
        self.assertTrue(tick_momentum(b.bars, ts[-1].ts, 2, 1, 0.5, 0.05, 120).reason.startswith("TICK_ABSTAIN"))

    def test_first_print_mid_bar_is_invalid_start(self):
        ts = ticks(12, cnt0=7)    # collector joins at exchange print #8
        b = build_tick_bars(ts, "000001", 5)
        self.assertTrue(b.bars[0].invalid)           # bar #6..#10 joined at #8
        self.assertEqual(b.bars[0].close, ts[2].price)
        self.assertFalse(b.bars[1].invalid)          # #11..#15


class TestRound4(unittest.TestCase):
    def test_whole_bars_missing_after_closed_bar(self):
        ts = ticks(12)
        b = TickBarBuilder("000001", 3)
        for t in ts[:3] + ts[9:12]:   # #1-3, then #10-12 (bars #4-6, #7-9 entirely missed)
            b.ingest(t)
        self.assertEqual(len(b.bars), 4)
        self.assertEqual([x.invalid for x in b.bars], [False, True, True, False])
        self.assertEqual(sum(x.unattributed_volume for x in b.bars), 60)  # 6 missed prints over 2 bars
        self.assertEqual(tick_momentum(b.bars, ts[11].ts, 2, 1, 0.5, 0.05, 120).reason, "TICK_ABSTAIN_INVALID_BAR")

    def test_bar_closed_by_later_print_is_available_only_then(self):
        ts = ticks(5)
        ts[4].recv = ts[4].ts + timedelta(seconds=30)
        b = TickBarBuilder("000001", 3)
        for t in (ts[0], ts[1], ts[4]):  # #3 missed -> bar 1 is closed only when #5 is received
            b.ingest(t)
        self.assertEqual(len(b.bars), 1)
        self.assertEqual(b.bars[0].available_at, ts[4].recv)
        self.assertEqual(b.bars_until(ts[1].ts + timedelta(seconds=1)), [])

    def test_inference_requires_exact_volume(self):
        ts = ticks(3)
        ts[1].cum_count = None
        ts[1].acml_vol = ts[0].acml_vol + 5   # delta 5 < qty 10: inconsistent, must not be trusted
        ts[2].acml_vol = ts[1].acml_vol + 10
        b = TickBarBuilder("000001", 3)
        for t in ts:
            b.ingest(t)
        self.assertEqual(b.stats.inconsistent, 1)
        self.assertTrue(all(x.invalid for x in b.bars))

    def test_late_previous_session_print_dropped(self):
        b = TickBarBuilder("000001", 3)
        today = ticks(2)
        b.ingest(today[0])
        late = ticks(1, t0=T0 - timedelta(days=1))[0]
        b.ingest(late)
        b.ingest(today[1])
        self.assertEqual(b.stats.out_of_order, 1)
        self.assertEqual(b._cur.n_ticks, 2)


class TestRound5(unittest.TestCase):
    def test_volume_count_contradiction_invalidates(self):
        ts = ticks(12)
        for i, t in enumerate(ts):
            if i >= 1:
                t.acml_vol += 100   # volume jumps by 100 at print #2 while the count advances by exactly 1
        b = build_tick_bars(ts, "000001", 3)
        self.assertEqual(b.stats.inconsistent, 1)
        self.assertTrue(b.bars[0].invalid)
        self.assertEqual(tick_momentum(b.bars, ts[-1].ts, 4, 1, 0.5, 0.05, 120).reason, "TICK_ABSTAIN_INVALID_BAR")
