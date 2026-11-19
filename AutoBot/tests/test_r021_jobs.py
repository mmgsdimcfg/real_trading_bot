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

    def test_email_adds_notify_email_to_the_scan_step(self):
        """[2026-11-11] --email -> g002 --notify-email (scan step only); off by default."""
        import argparse

        argv = r021.STEPS["scan"].build(r021.JobArgs(date="20261110", email=True), {})
        self.assertEqual(argv[1:], ["--date", "20261110", "--no-export-watchlist", "--notify-email"])
        for name in ("collect", "history", "apply"):
            self.assertNotIn("--notify-email", r021.STEPS[name].build(r021.JobArgs(date="20261110", email=True), {}))
        sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
        r021.add_job_parser(sub)
        ns = sub.choices["job"].parse_args(["start", "postclose", "--date", "20261110", "--email"])
        self.assertTrue(r021._job_args(ns, "20261110").email)
        ns = sub.choices["job"].parse_args(["start", "postclose", "--date", "20261110"])
        self.assertFalse(r021._job_args(ns, "20261110").email)

    def test_apply_step_calls_r016_watchlist_apply(self):
        argv = r021.STEPS["apply"].build(r021.JobArgs(date="20261110", by="autoboth"), {})
        self.assertTrue(argv[0].endswith("r016_control.py"))
        self.assertEqual(argv[1:], ["--by", "autoboth", "watchlist", "apply", "--date", "20261110", "--restart-collector"])

    def test_step_combinations(self):
        names = lambda **kw: [s.name for s in r021.job_steps("postclose", r021.JobArgs(date="20261110", **kw))]  # noqa: E731
        self.assertEqual(names(), ["collect", "history", "hi52", "scan", "picklog", "apply"])
        self.assertEqual(names(collect=False), ["scan", "picklog", "apply"])
        self.assertEqual(names(collect=False, scan=False), ["apply"])
        self.assertEqual(names(apply=False), ["collect", "history", "hi52", "scan", "picklog"])
        self.assertEqual(names(edge_ai=True, apply=False),
                         ["collect", "history", "hi52", "scan", "picklog", "edge_dataset", "edge_train", "edge_check"])
        self.assertEqual([s.name for s in r021.job_steps("scan", r021.JobArgs(date="20261110", apply=False))], ["scan"])
        self.assertEqual([s.name for s in r021.job_steps("collect", r021.JobArgs(date="20261110", scan=False, apply=False))], ["collect"])

    def test_picklog_step_logs_the_job_date_and_follows_scan(self):
        """[2026-11-19] picklog = g047 --date D --evaluate, not required, after scan / before apply, on and off with scan."""
        st = r021.STEPS["picklog"]
        self.assertFalse(st.required)
        argv = st.build(r021.JobArgs(date="20261119"), {})
        self.assertTrue(argv[0].endswith("g047_pick_forward_log.py"))
        self.assertEqual(argv[1:], ["--date", "20261119", "--evaluate"])
        no_scan = [s.name for s in r021.job_steps("postclose", r021.JobArgs(date="20261119", scan=False))]
        self.assertNotIn("picklog", no_scan)
        self.assertNotIn("picklog", [s.name for s in r021.job_steps("scan", r021.JobArgs(date="20261119"))])

    def test_hi52_step_logs_the_job_date_and_is_optional(self):
        """[2026-11-18] hi52 = g042 --date D --evaluate, not required, only in postclose (follows the collect switch)."""
        st = r021.STEPS["hi52"]
        argv = st.build(r021.JobArgs(date="20261118"), {})
        self.assertTrue(argv[0].endswith("g042_hi52_signal_log.py"))
        self.assertEqual(argv[1:], ["--date", "20261118", "--evaluate"])
        self.assertFalse(st.required)
        for job in ("collect", "history", "scan"):
            self.assertNotIn("hi52", [s.name for s in r021.job_steps(job, r021.JobArgs(date="20261118"))])

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
                r021.start_job(paths, "postclose", r021.JobArgs(date="20261110", edge_ai=True, collect=False, apply=False, email=True,
                                                                  scheduled=True), "test",
                               spawn=fake_spawn)
            finally:
                r021.START_CONFIRM_SEC = old
        self.assertEqual(len(captured), 1)
        self.assertIn("--no-collect", captured[0])
        self.assertIn("--no-apply", captured[0])
        self.assertNotIn("--no-scan", captured[0])
        self.assertIn("--edge-ai", captured[0])
        self.assertIn("--email", captured[0])
        self.assertIn("--scheduled", captured[0])


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


class TestListDataDates(unittest.TestCase):
    """[2026-11-17] job dates: every YYYYMMDD folder of <runtime>/data, newest first, with collected / scannable / picks."""

    def test_flags_order_and_non_date_folders(self):
        import argparse
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            full, partial, empty = data / "20261113", data / "20261116", data / "20261117"
            for d in (full, partial, empty, data / "ticks", data / "sim", data / "2026111"):
                d.mkdir(parents=True)
            (full / "005930_x_10s.txt").write_text("t\n", encoding="utf-8")
            (full / "_20261113_collected.txt").write_text("005930\n", encoding="utf-8")
            (full / "_20261113_picks.txt").write_text("005930,x\n", encoding="utf-8")
            (partial / "000660_x_10s.txt").write_text("t\n", encoding="utf-8")       # collect stopped before the marker
            (empty / "_20261117_collected.txt.tmp").write_text("", encoding="utf-8")
            (data / "20261112.txt").write_text("", encoding="utf-8")                  # a file, not a folder
            out = r021.list_data_dates(r021.Paths(Path(tmp)))
            self.assertTrue(out["ok"])
            self.assertEqual(out["dates"], [
                {"date": "20261117", "collected": False, "scannable": False, "picks": False},
                {"date": "20261116", "collected": False, "scannable": True, "picks": False},
                {"date": "20261113", "collected": True, "scannable": True, "picks": True},
            ])
            ns = argparse.Namespace(action="dates", name=None, date=None, n=10)
            out2, rc = r021.run_job_command(ns, "t", r021.Paths(Path(tmp)))
            self.assertEqual((rc, len(out2["dates"])), (0, 3))
            sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
            r021.add_job_parser(sub)
            self.assertEqual(sub.choices["job"].parse_args(["dates"]).action, "dates")

    def test_missing_data_root(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = r021.list_data_dates(r021.Paths(Path(tmp)))
            self.assertFalse(out["ok"])
            self.assertIn("cannot list", out["error"])



class TestHistoryAfterFailedCollect(unittest.TestCase):
    """[2026-11-19] a failed collect still runs history (after_failure); the other later steps are skipped."""

    def test_history_runs_after_collect_failure(self):
        import tempfile
        from unittest import mock

        self.assertTrue(r021.STEPS["history"].after_failure)
        self.assertFalse(any(r021.STEPS[n].after_failure for n in ("collect", "hi52", "scan", "apply")))
        with tempfile.TemporaryDirectory() as tmp:
            steps = {n: r021.Step(n, (lambda a, rc, n=n: ["-c", "raise SystemExit(1)" if n == "collect" else f"print('{n}')"]),
                                  required=r021.STEPS[n].required, after_failure=r021.STEPS[n].after_failure)
                     for n in ("collect", "history", "hi52", "scan", "picklog", "apply")}
            with mock.patch.dict(r021.STEPS, steps):
                rc = r021.run_detached_entry("postclose", r021.JobArgs(date="20261118"), "t", r021.Paths(Path(tmp)))
            st = r021.read_state(r021.Paths(Path(tmp)), "postclose")
        self.assertEqual(rc, 1)
        self.assertEqual(st["state"], "failed")
        self.assertIn("collect failed", st["error"])
        self.assertEqual([(s["name"], s["state"]) for s in st["steps"]],
                         [("collect", "failed"), ("history", "done"), ("hi52", "skipped"), ("scan", "skipped"), ("picklog", "skipped"), ("apply", "skipped")])
        self.assertEqual(st["steps"][2]["note"], "collect failed")


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
            self.assertIs(st["args"]["scheduled"], False)  # [2026-11-11] manual run; --scheduled -> True
