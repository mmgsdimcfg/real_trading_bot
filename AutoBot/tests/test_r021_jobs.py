# -*- coding: utf-8 -*-
"""r021 data jobs: the scan step honours --keep-watchlist (r004 untouched for past-date backfills / Edge AI)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import r021_jobs as r021  # noqa: E402


class TestScanStepArgv(unittest.TestCase):
    def test_default_exports_watchlist(self):
        argv = r021.STEPS["scan"].build(r021.JobArgs(date="20261110"), {})
        self.assertIn("--export-watchlist", argv)
        self.assertIn("--for-next-trading-day", argv)
        self.assertNotIn("--no-export-watchlist", argv)

    def test_keep_watchlist_disables_export(self):
        argv = r021.STEPS["scan"].build(r021.JobArgs(date="20261110", keep_watchlist=True), {})
        self.assertIn("--no-export-watchlist", argv)
        self.assertNotIn("--export-watchlist", argv)
        self.assertNotIn("--for-next-trading-day", argv)
        self.assertEqual(argv[1:3], ["--date", "20261110"])

    def test_postclose_steps_keep_order_with_edge_ai(self):
        names = [s.name for s in r021.job_steps("postclose", r021.JobArgs(date="20261110", edge_ai=True, keep_watchlist=True))]
        self.assertEqual(names, ["collect", "history", "scan", "edge_dataset", "edge_train", "edge_check"])


class TestStartArgv(unittest.TestCase):
    def test_keep_watchlist_forwarded_to_runner(self):
        import argparse
        import tempfile

        captured: list[list[str]] = []

        def fake_spawn(argv, **_kw):
            captured.append(list(argv))
            return 4242

        with tempfile.TemporaryDirectory() as tmp:
            paths = r021.Paths(Path(tmp))
            old = r021.START_CONFIRM_SEC
            r021.START_CONFIRM_SEC = 0.0  # no runner is really started
            try:
                r021.start_job(paths, "postclose", r021.JobArgs(date="20261110", edge_ai=True, keep_watchlist=True), "test",
                               spawn=fake_spawn)
            finally:
                r021.START_CONFIRM_SEC = old
        self.assertEqual(len(captured), 1)
        self.assertIn("--keep-watchlist", captured[0])
        self.assertIn("--edge-ai", captured[0])

        sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
        r021.add_job_parser(sub)
        ns = sub.choices["job"].parse_args(["start", "postclose", "--date", "20261110", "--keep-watchlist"])
        self.assertTrue(ns.keep_watchlist)
        ns = sub.choices["job"].parse_args(["start", "postclose"])
        self.assertFalse(ns.keep_watchlist)


if __name__ == "__main__":
    unittest.main()


class TestDateDataInfo(unittest.TestCase):
    """[2026-11-11] job data --date D: what the date folder holds (g001 files, collected list, g002 picks)."""

    def test_missing_folder(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            info = r021.date_data_info(r021.Paths(Path(tmp)), "20261110")
            self.assertTrue(info["ok"])
            self.assertFalse(info["exists"])
            self.assertFalse(info["scannable"])
            self.assertIsNone(info["picks"])

    def test_counts_marker_and_picks(self):
        import argparse
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "data" / "20261110"
            d.mkdir(parents=True)
            for code in ("005930", "000660", "035720"):
                (d / f"{code}_x_10s.txt").write_text("t\n", encoding="utf-8")
            (d / "005930_x_daily.csv").write_text("d\n", encoding="utf-8")
            (d / "_20261110_collected.txt").write_text("\ufeff005930,삼성전자\n000660,SK하이닉스\n\n", encoding="utf-8")
            (d / "_20261110_picks.txt").write_text("005930,삼성전자\n# c\n660,SK하이닉스\n005930,dup\nabc,x\n", encoding="utf-8")
            paths = r021.Paths(Path(tmp))
            info = r021.date_data_info(paths, "20261110")
            self.assertTrue(info["exists"] and info["scannable"] and info["collected"])
            self.assertEqual((info["files_10s"], info["files_daily"], info["collected_count"]), (3, 1, 2))
            self.assertIsNotNone(info["data_mtime"])
            self.assertEqual(info["picks"], [{"code": "005930", "name": "삼성전자"}, {"code": "000660", "name": "SK하이닉스"}])
            self.assertIsNotNone(info["picks_mtime"])
            # through the r016 `job` parser: no job name needed, bad date -> rc 2
            out, rc = r021.run_job_command(argparse.Namespace(action="data", name=None, date="20261110", n=10), "t", paths)
            self.assertEqual((rc, out["files_10s"]), (0, 3))
            out, rc = r021.run_job_command(argparse.Namespace(action="data", name=None, date="2026-11-10", n=10), "t", paths)
            self.assertEqual(rc, 2)
