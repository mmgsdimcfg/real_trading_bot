# -*- coding: utf-8 -*-
"""g003 triple-trigger replay (2026-11-02).

r013 TickReplayFeed must answer like the live TickTriggerFeed would have at the same moment, and g003's triple path must
pick the mode per date and apply the r005 BYPASS rules for the 3-min context."""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

from r009_tick_bars import TICK_FILE_COLUMNS  # noqa: E402
from r013_tick_trigger import TickReplayFeed, TickTriggerFeed  # noqa: E402

import g003_trade_simulate_by_date as g003  # noqa: E402

T0 = datetime(2026, 11, 2, 10, 0, 0)


def _rows(prices, start, k0=0, step_s=1):
    rows = []
    for j, p in enumerate(prices):
        k = k0 + j + 1
        ts = start + timedelta(seconds=step_s * j)
        rows.append({
            "recv_ts": ts.isoformat(timespec="milliseconds"), "code": "003160", "bsop_date": ts.strftime("%Y%m%d"),
            "trade_time": ts.strftime("%H%M%S"), "price": p, "cntg_vol": 10, "acml_vol": 10 * k,
            "ccld_dvsn": 1, "seln_cntg_csnu": 0, "shnu_cntg_csnu": k, "askp1": p + 50, "bidp1": p,
            "hour_cls_code": 0,
        })
    return rows


def _write(path: Path, rows, header: bool):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TICK_FILE_COLUMNS)
        if header:
            w.writeheader()
        w.writerows(rows)


# falling, rising (cross), flat, falling again, gap (stale), rising again
PRICES = ([42000 - 10 * i for i in range(30)] + [41720 + 30 * i for i in range(30)] + [42600] * 12
          + [42600 - 20 * i for i in range(30)])


class TestTickReplayParity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.live_dir = Path(self.tmp.name) / "live"
        self.full_dir = Path(self.tmp.name) / "full"
        for d in (self.live_dir, self.full_dir):
            (d / "20261102").mkdir(parents=True)
        rows = _rows(PRICES, T0)
        # a 200 s hole then more prints: exercises the stale abstain and the recovery
        rows += _rows([42000 + 40 * i for i in range(30)], T0 + timedelta(seconds=len(PRICES) + 200), k0=len(PRICES))
        self.rows = rows
        _write(self.full_dir / "20261102" / "003160.csv", rows, header=True)

    def test_replay_matches_live_feed_every_second(self):
        args = ((3,), 2, 4, 240, 120)
        live = TickTriggerFeed(self.live_dir, *args)
        replay = TickReplayFeed(self.full_dir, *args)
        live_path = self.live_dir / "20261102" / "003160.csv"
        written = 0
        end = datetime.fromisoformat(self.rows[-1]["recv_ts"]) + timedelta(seconds=5)
        now = T0  # before the first print the live file does not exist yet (NO_FILE vs NO_TICKS - both abstain)
        seen_ok = seen_stale = False
        while now <= end:
            new = [r for r in self.rows[written:] if datetime.fromisoformat(r["recv_ts"]) <= now]
            if new:
                _write(live_path, new, header=written == 0)
                written += len(new)
            a, b = live.evaluate("003160", now), replay.evaluate("003160", now)
            self.assertEqual((a.ok, a.reason, a.cross_time), (b.ok, b.reason, b.cross_time), now)
            seen_ok |= a.ok
            seen_stale |= a.reason.startswith("TICK_ABSTAIN_FEED_STALE_")
            now += timedelta(seconds=1)
        self.assertTrue(seen_ok)
        self.assertTrue(seen_stale)

    def test_no_file_and_before_first_print(self):
        replay = TickReplayFeed(self.full_dir, (3,), 2, 4, 240, 120)
        self.assertEqual(replay.evaluate("000001", T0).reason, "TICK_ABSTAIN_NO_FILE")
        self.assertEqual(replay.evaluate("003160", T0 - timedelta(seconds=1)).reason, "TICK_ABSTAIN_NO_TICKS")


class TestPriorDayWarmup(unittest.TestCase):
    """[2026-11-02] prior-day tick file seeds the bars so the MAs work from the first prints of the day (HTS-like)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        for d in ("20261030", "20261102"):
            (self.dir / d).mkdir()
        prev = datetime(2026, 10, 30, 15, 0, 0)
        _write(self.dir / "20261030" / "003160.csv", _rows([42000 - 10 * i for i in range(30)], prev), header=True)
        # today: 3 falling-then-rising bars only - not enough for MA4 on its own
        self.today_rows = _rows([41700, 41700, 41700, 41800, 41900, 42000, 42100, 42200, 42300], T0)
        _write(self.dir / "20261102" / "003160.csv", self.today_rows, header=True)
        self.now = T0 + timedelta(seconds=9)

    def test_without_warmup_few_bars(self):
        r = TickTriggerFeed(self.dir, (3,), 2, 4, 240, 120).evaluate("003160", self.now)
        self.assertIn("TICK3_ABSTAIN_FEW_BARS_3", r.reason)

    def test_warmup_live_and_replay_agree_and_cross(self):
        live = TickTriggerFeed(self.dir, (3,), 2, 4, 240, 120, prior_day_warmup=True)
        replay = TickReplayFeed(self.dir, (3,), 2, 4, 240, 120, prior_day_warmup=True)
        a, b = live.evaluate("003160", self.now), replay.evaluate("003160", self.now)
        self.assertEqual((a.ok, a.reason, a.cross_time), (b.ok, b.reason, b.cross_time))
        self.assertTrue(a.ok, a.reason)
        self.assertEqual(a.cross_time.date(), T0.date())  # the cross itself is today's bar

    def test_warmup_survives_file_replacement(self):
        live = TickTriggerFeed(self.dir, (3,), 2, 4, 240, 120, prior_day_warmup=True)
        before = live.evaluate("003160", self.now)
        path = self.dir / "20261102" / "003160.csv"
        tmp = path.with_suffix(".new")
        _write(tmp, self.today_rows, header=True)
        os.replace(tmp, path)  # new inode -> reader resets
        after = live.evaluate("003160", self.now)
        self.assertEqual((before.ok, before.reason, before.cross_time), (after.ok, after.reason, after.cross_time))
        self.assertGreater(len(live._feeds["003160"].builders[3].bars), 5)

    def test_prior_tail_is_bounded(self):
        from r013_tick_trigger import iter_prior_tail
        ticks = list(iter_prior_tail(self.dir / "20261030" / "003160.csv", 7))
        self.assertEqual(len(ticks), 7)
        self.assertEqual(ticks[-1].price, 42000 - 10 * 29)

    def test_prior_file_only_from_latest_earlier_day(self):
        from r013_tick_trigger import find_prior_tick_file
        self.assertEqual(find_prior_tick_file(self.dir, "20261102", "003160"), self.dir / "20261030" / "003160.csv")
        self.assertIsNone(find_prior_tick_file(self.dir, "20261102", "000001"))
        self.assertIsNone(find_prior_tick_file(self.dir, "20261030", "003160"))
        (self.dir / "20261031").mkdir()  # latest earlier day without the code -> no warmup (no older fallback)
        self.assertIsNone(find_prior_tick_file(self.dir, "20261102", "003160"))


class TestEntryTriggerMode(unittest.TestCase):
    def test_auto_uses_triple_only_when_tick_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20261030").mkdir()
            (root / "20261030" / "003160.csv").write_text("x")
            (root / "20261029").mkdir()  # folder without csv
            with mock.patch.object(g003, "ENTRY_TRIGGER_MODE_SIM", "auto"), \
                    mock.patch.object(g003, "ENTRY_TRIGGER_MODE", "triple"):
                self.assertEqual(g003.resolve_entry_trigger_mode("20261030", root), "triple")
                self.assertEqual(g003.resolve_entry_trigger_mode("20261029", root), "legacy")
                self.assertEqual(g003.resolve_entry_trigger_mode("20261007", root), "legacy")
            with mock.patch.object(g003, "ENTRY_TRIGGER_MODE_SIM", "auto"), \
                    mock.patch.object(g003, "ENTRY_TRIGGER_MODE", "legacy"):
                self.assertEqual(g003.resolve_entry_trigger_mode("20261030", root), "legacy")
            with mock.patch.object(g003, "ENTRY_TRIGGER_MODE_SIM", "triple"):
                self.assertEqual(g003.resolve_entry_trigger_mode("20261007", root), "triple")


def _gate(name, ok):
    return SimpleNamespace(name=name, eval_fn=lambda ctx: (ok, ctx, None if ok else f"{name.upper()}_FAIL"))


class TestTripleContextBypass(unittest.TestCase):
    def _run(self, gates, score_ok, bypass_score=True):
        with mock.patch.object(g003, "build_context_eval", return_value=(object(), "")), \
                mock.patch.object(g003, "HYBRID_3MIN_CONTEXT_GATES", gates), \
                mock.patch.object(g003, "TRIPLE_CROSS_BYPASS_CONTEXT_GATES", ("stochastic_buy_signal", "di_spread_min")), \
                mock.patch.object(g003, "TRIPLE_CROSS_BYPASS_SCORE", bypass_score), \
                mock.patch.object(g003, "evaluate_context_score",
                                  return_value=(True, "HYBRID_3MIN_CTX_SCORE_5") if score_ok
                                  else (False, "HYBRID_3MIN_CTX_LOW_SCORE_2_LT_4")):
            return g003.run_3min_context_pipeline_triple(None, None, 0.0, {}, None)

    def test_soft_gate_failures_and_low_score_bypassed(self):
        ok, reason, bypassed = self._run(
            [_gate("stochastic_buy_signal", False), _gate("min_liquidity_safety", True), _gate("di_spread_min", False)],
            score_ok=False,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "HYBRID_3MIN_CTX_LOW_SCORE_2_LT_4")
        self.assertEqual(bypassed, ["stochastic_buy_signal", "di_spread_min", "score"])

    def test_hard_gate_failure_rejects(self):
        ok, reason, _ = self._run(
            [_gate("stochastic_buy_signal", False), _gate("min_entry_atr_volatility", False)], score_ok=True,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "HYBRID_3MIN_CTX_MIN_ENTRY_ATR_VOLATILITY_FAIL")

    def test_low_score_rejects_without_score_bypass(self):
        ok, reason, _ = self._run([_gate("stochastic_buy_signal", True)], score_ok=False, bypass_score=False)
        self.assertFalse(ok)
        self.assertIn("LOW_SCORE", reason)


if __name__ == "__main__":
    unittest.main()
