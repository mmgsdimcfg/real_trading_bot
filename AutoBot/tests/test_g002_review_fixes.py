# -*- coding: utf-8 -*-
"""g002 / g005 / g043 fixes after the 2026-11-28 review: no look-ahead in the picks file of a date, the next TRADING day,
the stale picks file of a zero-pick rerun, daily rows after the target date."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g002_data_scan_trade_candidates as g002  # noqa: E402
import g005_watchlist_bridge as g005  # noqa: E402
import g008_selection_reason_report as g008  # noqa: E402
import g043_selection_replay as g043  # noqa: E402


def _daily(dir_: Path, code: str, dates: list[str]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    rows = ["date,open,high,low,close,volume"] + [f"{d},100,110,90,{100 + i},1000" for i, d in enumerate(dates)]
    (dir_ / f"{code}_테스트_daily.csv").write_text("\n".join(rows), encoding="utf-8")


class TestDailyCsvTargetDate(unittest.TestCase):
    def test_rows_after_the_target_date_are_dropped(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _daily(root / "20261126", "000001", ["20261124", "20261125", "20261126", "20261127"])
            df = g002._load_daily_csv("000001", root, "20261126")
            self.assertIsNotNone(df)
            self.assertEqual(df["date"].iloc[-1].strftime("%Y%m%d"), "20261126")
            self.assertEqual(len(df), 3)

    def test_no_row_of_the_target_date_means_no_history(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _daily(root / "20261126", "000002", ["20261123", "20261124", "20261125"])
            self.assertIsNone(g002._load_daily_csv("000002", root, "20261126"))

    def test_normal_file_is_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _daily(root / "20261126", "000003", ["20261124", "20261125", "20261126"])
            df = g002._load_daily_csv("000003", root, "20261126")
            self.assertEqual(len(df), 3)


class TestRetireStalePicks(unittest.TestCase):
    def test_old_picks_file_is_moved_not_deleted(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "_20261126_picks.txt"
            f.write_text("005930,삼성전자", encoding="utf-8")
            dest = g002._retire_stale_picks(f)
            self.assertFalse(f.exists())
            self.assertEqual(dest.name, "_20261126_picks.txt.stale")
            self.assertEqual(dest.read_text(encoding="utf-8"), "005930,삼성전자")

    def test_nothing_to_move(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(g002._retire_stale_picks(Path(t) / "_20261126_picks.txt"))


class TestWatchlistOfADate(unittest.TestCase):
    def test_next_trading_day_skips_the_weekend(self):
        self.assertEqual(g005.next_trading_day("20261127"), "20261130")  # Friday -> Monday
        self.assertEqual(g005.next_trading_day("20261125"), "20261126")
        self.assertIsNone(g005.next_trading_day("bad"))

    def test_r004_header_names_the_next_trading_day(self):
        text = g005.render_r004_content(["005930,삼성전자"], scan_date="20261127", for_next_trading_day=True)
        self.assertIn("# for_trading_day=20261130", text)
        self.assertNotIn("20261128", text)

    def test_picks_of_a_date_come_from_the_scan_before_it(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for d in ("20261125", "20261126", "20261127"):
                (root / d).mkdir()
            (root / "20261125" / "_20261125_picks.txt").write_text("000001,A", encoding="utf-8")
            (root / "20261127" / "_20261127_picks.txt").write_text("000002,B", encoding="utf-8")  # made after the close of the 27th
            got = g005.prior_scan_picks_path(root, "20261127")
            self.assertEqual(got, root / "20261125" / "_20261125_picks.txt")       # the 26th had no scan
            self.assertIsNone(g005.prior_scan_picks_path(root, "20261125"))
            auto = root / "live"
            auto.mkdir()
            with mock.patch.object(g005, "FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS", True):
                self.assertEqual(g005.resolve_watchlist_path(auto, "20261127", "auto", root), got)
                # no earlier scan: the live watchlist file, never the date's own (later) picks
                self.assertEqual(g005.resolve_watchlist_path(auto, "20261125", "auto", root), auto / g005.R004_WATCHLIST_FILENAME)
            self.assertEqual(g005.resolve_watchlist_path(auto, None, "auto", root), auto / g005.R004_WATCHLIST_FILENAME)


class TestReplayNextTradingDay(unittest.TestCase):
    def test_trading_day_after(self):
        self.assertEqual(g043.trading_day_after("20261127", 1), "20261130")
        self.assertEqual(g043.trading_day_after("20261127", 3), "20261202")

    def test_a_halted_next_day_gives_no_label(self):
        # the symbol has no row on Monday 11/30 (halt): the old code took Tuesday as "the next day"
        hist = {"dates": ["20261127", "20261201", "20261202"], "open": [100.0, 120.0, 121.0], "close": [100.0, 130.0, 110.0]}
        hist["idx"] = {d: i for i, d in enumerate(hist["dates"])}
        with mock.patch.dict(g043._HIST, {"999999": hist}):
            out = g043.labels("999999", "20261127")
        self.assertNotIn("d1", out)
        self.assertNotIn("d1_oc", out)
        self.assertAlmostEqual(out["d3"], 110.0 / 100.0 - 1.0)   # 12/02 is the third trading day

    def test_next_folder_needs_the_trading_day_itself(self):
        with mock.patch.object(g043, "_FOLDERS", ["20261127", "20261201"]):
            self.assertIsNone(g043.next_folder("20261127"))     # 11/30 was not collected
        with mock.patch.object(g043, "_FOLDERS", ["20261127", "20261130"]):
            self.assertEqual(g043.next_folder("20261127"), "20261130")


def _cand(code, price=10000, amount=10e9, gap=0.05, rsi=55.0, vol_ratio=1.0, fails=(), skip=None):
    return {"code": code, "name": code, "price": price, "amount_ma20": amount, "close_ma20_ratio": gap, "rsi14": rsi,
            "ind": {"vol_ratio": vol_ratio}, "fail_reasons": list(fails), "soft_flags": [], "skip_reason": skip, "pattern_tier_rank": 2}


class TestSimpleProfileS1(unittest.TestCase):
    def test_pct_rank(self):
        self.assertEqual(g002._pct_rank([3.0, 1.0, 2.0]), [1.0, 1 / 3, 2 / 3])
        self.assertEqual(g002._pct_rank([5.0, 5.0]), [0.75, 0.75])
        self.assertEqual(g002._pct_rank([]), [])

    def test_tick_ratio_bands(self):
        self.assertAlmostEqual(g002._s1_tick_ratio(1999), 1 / 1999)
        self.assertAlmostEqual(g002._s1_tick_ratio(2000), 5 / 2000)
        self.assertAlmostEqual(g002._s1_tick_ratio(50000), 100 / 50000)
        self.assertIsNone(g002._s1_tick_ratio(None))

    def test_gates(self):
        rows = [
            _cand("ok"),
            _cand("thin", amount=4.9e9),
            _cand("below", gap=-0.01),
            _cand("hot_gap", gap=0.16),
            _cand("hot_rsi", rsi=71.0),
            # reasons that only the v0 selection has are dropped
            _cand("old_only", fails=("down_trend", "pattern_overheated", "low_score", "liquidity_below_market_dual")),
            _cand("spac", fails=("preferred_or_spac_excluded", "down_trend")),
            _cand("nodata", vol_ratio=None),
            _cand("skipped", skip="no_data"),
        ]
        g002.apply_simple_profile(rows)
        by = {r["code"]: r for r in rows}
        self.assertTrue(by["ok"]["eligible"])
        self.assertTrue(by["old_only"]["eligible"])
        self.assertEqual(by["thin"]["fail_reasons"], ["s1_low_turnover"])
        self.assertEqual(by["below"]["fail_reasons"], ["s1_below_ma20"])
        self.assertEqual(by["hot_gap"]["fail_reasons"], ["s1_overheated"])
        self.assertEqual(by["hot_rsi"]["fail_reasons"], ["s1_overheated"])
        self.assertEqual(by["spac"]["fail_reasons"], ["preferred_or_spac_excluded"])
        self.assertEqual(by["nodata"]["fail_reasons"], ["pattern_unknown"])
        self.assertFalse(by["skipped"]["eligible"])
        self.assertTrue(all(r["pattern_tier_rank"] == 0 for r in rows))

    def test_rank_is_volume_ratio_and_low_tick_cost(self):
        rows = [
            _cand("busy_cheap_tick", price=19990, vol_ratio=3.0),   # tick 10 / 19,990 = 0.05 %
            _cand("busy_wide_tick", price=2000, vol_ratio=3.0),     # tick 5 / 2,000 = 0.25 %
            _cand("quiet_cheap_tick", price=19990, vol_ratio=0.5),
            _cand("quiet_wide_tick", price=2000, vol_ratio=0.5),
        ]
        g002.apply_simple_profile(rows)
        order = [r["code"] for r in sorted(rows, key=g002._selection_rank_key, reverse=True)]
        self.assertEqual(order[0], "busy_cheap_tick")
        self.assertEqual(order[-1], "quiet_wide_tick")
        top = rows[0]
        self.assertEqual({g["key"] for g in top["score_breakdown"]["gains"]}, {"s1_volume_ratio", "s1_low_tick_cost"})
        self.assertAlmostEqual(top["score"], sum(g["value"] for g in top["score_breakdown"]["gains"]))

    def test_the_default_profile_is_s1_and_v0_is_selectable(self):
        """[2026-11-28] the live default became s1 (user decision); G002_FILTER_PROFILE=v0 is the earlier selection."""
        import os
        self.assertIn("s1", g002.FILTER_PROFILES)
        with mock.patch.dict(os.environ, {"G002_FILTER_PROFILE": "s1"}):
            self.assertEqual(g002._filter_profile(), "s1")
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("G002_FILTER_PROFILE", None)
            self.assertEqual(g002._filter_profile(), "s1")
        with mock.patch.dict(os.environ, {"G002_FILTER_PROFILE": "v0"}):
            self.assertEqual(g002._filter_profile(), "v0")

    def test_every_etf_and_etn_is_excluded(self):
        """[2026-11-28] s1 leaves out every fund product, not only the derivative ones; a share whose name starts like a brand stays."""
        rows = [{"code": c, "name": n, "fail_reasons": list(f), "skip_reason": None, "price": 10000.0, "amount_ma20": 9e9,
                 "close_ma20_ratio": 0.03, "rsi14": 55.0, "ind": {"vol_ratio": 1.5}}
                for c, n, f in (("etf", "KODEX 증권", ()), ("etn", "신한 WTI원유 ETN", ()), ("lev", "KODEX 레버리지", ("derivative_etf_excluded",)),
                                ("share", "HK이노엔", ()), ("share2", "삼성전자", ()))]
        g002.apply_simple_profile(rows)
        by = {r["code"]: r for r in rows}
        self.assertEqual(by["etf"]["fail_reasons"], ["s1_fund_excluded"])
        self.assertEqual(by["etn"]["fail_reasons"], ["s1_fund_excluded"])
        self.assertEqual(by["lev"]["fail_reasons"], ["derivative_etf_excluded"])
        self.assertTrue(by["share"]["eligible"] and by["share2"]["eligible"])
        self.assertFalse(by["etf"]["eligible"] or by["etn"]["eligible"] or by["lev"]["eligible"])

    def test_reason_texts_exist(self):
        for k in ("s1_low_turnover", "s1_below_ma20", "s1_overheated", "s1_fund_excluded"):
            self.assertIn(k, g008.FAIL_REASON_DESC)


if __name__ == "__main__":
    unittest.main()
