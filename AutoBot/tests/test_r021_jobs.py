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
