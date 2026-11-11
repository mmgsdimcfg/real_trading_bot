# -*- coding: utf-8 -*-
"""r021 data jobs: step switches (collect / scan / apply), the apply step, start refusals, job data."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import r021_jobs as r021  # noqa: E402


class TestStepsAndArgv(unittest.TestCase):
    """[2026-11-11] postclose = collect -> history -> scan -> apply; --no-collect / --no-scan / --no-apply pick 1+2+3, 2+3, 3."""

    def test_scan_step_never_exports_itself(self):
        argv = r021.STEPS["scan"].build(r021.JobArgs(date="20261110"), {})
        self.assertEqual(argv[1:], ["--date", "20261110", "--no-export-watchlist"])
        self.assertNotIn("--export-watchlist", argv)

    def test_apply_step_calls_r016_watchlist_apply(self):
        argv = r021.STEPS["apply"].build(r021.JobArgs(date="20261110", by="autoboth"), {})
        self.assertTrue(argv[0].endswith("r016_control.py"))
        self.assertEqual(argv[1:], ["--by", "autoboth", "watchlist", "apply", "--date", "20261110", "--restart-collector"])

    def test_step_combinations(self):
        names = lambda **kw: [s.name for s in r021.job_steps("postclose", r021.JobArgs(date="20261110", **kw))]  # noqa: E731
        self.assertEqual(names(), ["collect", "history", "scan", "apply"])
        self.assertEqual(names(collect=False), ["scan", "apply"])
        self.assertEqual(names(collect=False, scan=False), ["apply"])
        self.assertEqual(names(apply=False), ["collect", "history", "scan"])
        self.assertEqual(names(edge_ai=True, apply=False), ["collect", "history", "scan", "edge_dataset", "edge_train", "edge_check"])
        self.assertEqual([s.name for s in r021.job_steps("scan", r021.JobArgs(date="20261110", apply=False))], ["scan"])
        self.assertEqual([s.name for s in r021.job_steps("collect", r021.JobArgs(date="20261110", scan=False, apply=False))], ["collect"])

    def test_keep_watchlist_is_no_apply(self):
        import argparse

        sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
        r021.add_job_parser(sub)
        ns = sub.choices["job"].parse_args(["start", "postclose", "--date", "20261110", "--keep-watchlist"])
        a = r021._job_args(ns, "20261110")
        self.assertTrue(a.keep_watchlist)
        self.assertFalse(a.apply)
        self.assertTrue(a.collect and a.scan)
        ns = sub.choices["job"].parse_args(["start", "postclose", "--date", "20261110", "--no-collect", "--no-scan"])
        a = r021._job_args(ns, "20261110")
        self.assertEqual((a.collect, a.scan, a.apply), (False, False, True))

    def test_start_refuses_missing_inputs_of_skipped_steps(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paths = r021.Paths(Path(tmp))
            out = r021.start_job(paths, "postclose", r021.JobArgs(date="20261110", collect=False), "t", spawn=lambda *a, **k: 1)
            self.assertFalse(out["ok"])
            self.assertIn("no collected data", out["error"])
            d = Path(tmp) / "data" / "20261110"
            d.mkdir(parents=True)
            (d / "005930_x_10s.txt").write_text("t\n", encoding="utf-8")
            out = r021.start_job(paths, "postclose", r021.JobArgs(date="20261110", collect=False, scan=False), "t",
                                 spawn=lambda *a, **k: 1)
            self.assertFalse(out["ok"])
            self.assertIn("no picks file", out["error"])
            out = r021.start_job(paths, "postclose", r021.JobArgs(date="20261110", collect=False, scan=False, apply=False), "t",
                                 spawn=lambda *a, **k: 1)
            self.assertEqual(out["error"], "no steps selected")


class TestStartArgv(unittest.TestCase):
    def test_step_switches_forwarded_to_runner(self):
        import tempfile

        captured: list[list[str]] = []

        def fake_spawn(argv, **_kw):
            captured.append(list(argv))
            return 4242

        with tempfile.TemporaryDirectory() as tmp:
            paths = r021.Paths(Path(tmp))
            d = Path(tmp) / "data" / "20261110"
            d.mkdir(parents=True)
            (d / "005930_x_10s.txt").write_text("t\n", encoding="utf-8")
            old = r021.START_CONFIRM_SEC
            r021.START_CONFIRM_SEC = 0.0  # no runner is really started
            try:
                r021.start_job(paths, "postclose", r021.JobArgs(date="20261110", edge_ai=True, collect=False, apply=False), "test",
                               spawn=fake_spawn)
            finally:
                r021.START_CONFIRM_SEC = old
        self.assertEqual(len(captured), 1)
        self.assertIn("--no-collect", captured[0])
        self.assertIn("--no-apply", captured[0])
        self.assertNotIn("--no-scan", captured[0])
        self.assertIn("--edge-ai", captured[0])


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



class TestApplyGuard(unittest.TestCase):
    """[2026-11-11] the apply step runs only on picks written by this run's scan (or an existing file when the scan was skipped)."""

    def _run(self, tmp: Path, scan_code: str, args: "r021.JobArgs") -> dict:
        from unittest import mock

        steps = {"scan": r021.Step("scan", lambda a, rc: ["-c", scan_code]),
                 "apply": r021.Step("apply", lambda a, rc: ["-c", "print('apply out')"])}
        with mock.patch.dict(r021.STEPS, steps):
            self.assertEqual(r021.run_detached_entry("scan", args, "t", r021.Paths(tmp)), 0)
        return r021.read_state(r021.Paths(tmp), "scan")

    def test_stale_or_missing_picks_skip_apply(self):
        import os
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "data" / "20261110"
            d.mkdir(parents=True)
            (d / "005930_x_10s.txt").write_text("t\n", encoding="utf-8")
            # no picks file at all: the scan found nothing -> apply skipped, job still done
            st = self._run(root, "print('scan out')", r021.JobArgs(date="20261110", collect=False))
            self.assertEqual(st["state"], "done")
            self.assertEqual([(s["name"], s["state"]) for s in st["steps"]], [("scan", "done"), ("apply", "skipped")])
            self.assertIn("no picks", st["steps"][1]["note"])
            self.assertIn("apply skipped", st["note"])
            # an old picks file (older than this run's scan) is not applied either
            pf = d / "_20261110_picks.txt"
            pf.write_text("005930,x\n", encoding="utf-8")
            old = time.time() - 600
            os.utime(pf, (old, old))
            st = self._run(root, "print('scan out')", r021.JobArgs(date="20261110", collect=False))
            self.assertEqual(st["steps"][1]["state"], "skipped")
            self.assertIn("predates", st["steps"][1]["note"])
            # the scan of this run writes the file -> apply runs
            code = f"open(r'{pf}', 'w').write('005930,x\\n'); print('scan out')"
            st = self._run(root, code, r021.JobArgs(date="20261110", collect=False))
            self.assertEqual([(s["name"], s["state"]) for s in st["steps"]], [("scan", "done"), ("apply", "done")])
            self.assertIsNone(st["note"])
            # scan skipped (step 3 alone): the existing file is applied as it is
            st = self._run(root, "raise SystemExit(9)", r021.JobArgs(date="20261110", collect=False, scan=False))
            self.assertEqual([(s["name"], s["state"]) for s in st["steps"]], [("apply", "done")])
