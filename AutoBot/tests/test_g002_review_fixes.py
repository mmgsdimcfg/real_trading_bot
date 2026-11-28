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


if __name__ == "__main__":
    unittest.main()
