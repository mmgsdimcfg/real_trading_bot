# -*- coding: utf-8 -*-
"""r019 locks / r020 supervisor / r016 process backend / r021 jobs (2026-11-10, Windows notebook = AutoBotH local mode).

The integration tests start real processes: r016 start spawns the r020 supervisor, which runs a fake r003 (R020_R003_SCRIPT)
that holds the r003 lock like the real one and ends on the stop-request file or with a chosen exit code."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r016_control as r016  # noqa: E402
import r019_platform as r019  # noqa: E402
import r020_supervisor as r020  # noqa: E402
import r021_jobs as r021  # noqa: E402

FAKE_R003 = textwrap.dedent('''
    import os, sys, time
    from pathlib import Path
    sys.path.insert(0, os.environ["FAKE_R003_LIB"])
    import r019_platform as r019
    live = Path(os.environ["AUTO_TRADING_RUNTIME"]) / "data" / "live_runtime"
    live.mkdir(parents=True, exist_ok=True)
    mode = (live.parent.parent / "fake_mode.txt").read_text().strip() if (live.parent.parent / "fake_mode.txt").exists() else "loop"
    (live.parent.parent / "fake_args.txt").write_text(" ".join(sys.argv[1:]) + "|" + os.environ.get("AUTO_TRADING_RISK_PROFILE", ""))
    fh = r019.acquire_instance_lock(live / ".r003.lock")
    if fh is None:
        sys.exit(3)
    if mode.startswith("exit"):
        sys.exit(int(mode[4:]))
    if mode == "ignore":  # never looks at the stop file (hung loop / blocked order call)
        while True:
            time.sleep(0.1)
    stop = live / "r003.stop"
    while not stop.exists():
        time.sleep(0.1)
    stop.unlink()
    sys.exit(0)
''')


def wait_for(pred, timeout=20.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class LockTests(unittest.TestCase):
    def test_instance_lock_is_exclusive_and_content_readable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".x.lock"
            fh = r019.acquire_instance_lock(p)
            self.assertIsNotNone(fh)
            self.assertIsNone(r019.acquire_instance_lock(p))
            self.assertTrue(r019.is_locked(p))
            self.assertEqual(r019.running_owner(p)["pid"], os.getpid())  # readable while locked (Windows: far-offset lock)
            fh.close()
            self.assertFalse(r019.is_locked(p))
            self.assertIsNone(r019.running_owner(p))
            self.assertFalse(r019.is_locked(Path(td) / "missing.lock"))
            self.assertFalse((Path(td) / "missing.lock").exists())

    def test_locked_file_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".w.lock"
            with r019.locked_file(p):
                with self.assertRaises(TimeoutError):
                    with r019.locked_file(p, timeout=0.2):
                        pass
            with r019.locked_file(p, timeout=0.2):
                pass

    def test_host_role_default_and_write(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(r019.read_host_role(td)["role"], "primary")
            self.assertFalse(r019.read_host_role(td)["set"])
            r019.write_host_role(td, "standby", "test", "laptop trades")
            self.assertEqual(r019.read_host_role(td)["role"], "standby")
            with self.assertRaises(ValueError):
                r019.write_host_role(td, "boss", "test")
            r019.host_role_path(td).write_text("{bad", encoding="utf-8")
            self.assertEqual(r019.read_host_role(td)["role"], "primary")

    def test_pid_alive(self):
        self.assertTrue(r019.pid_alive(os.getpid()))
        self.assertFalse(r019.pid_alive(0))


class EnvParityTests(unittest.TestCase):
    """r020 builds the same r003 command line as r003_service.sh from r003.env."""

    def test_build_command_from_env_file(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "r003.env"
            env.write_text('# c\nR003_DRY_RUN="1"\nAUTO_TRADING_RISK_PROFILE="aggressive"\nR003_DATE="20261110"\n'
                           'R003_EXTRA_ARGS="--a 1  --b"\nR003_WATCHLIST_SOURCE="scan"\nEVIL="x"\nR003_DATE2=1\n'
                           'R003_EXTRA_ARGS2="$(rm)"\n', encoding="utf-8")
            vals = r020.read_env(env)
            argv, extra = r020.build_command(vals, python="py", script=Path("r003.py"))
            self.assertEqual(argv, ["py", "r003.py", "--dry-run", "--date", "20261110", "--watchlist-source", "scan", "--a", "1", "--b"])
            self.assertEqual(extra, {"AUTO_TRADING_RISK_PROFILE": "aggressive"})

    def test_unsafe_value_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "r003.env"
            env.write_text('R003_EXTRA_ARGS="--x; rm -rf"\nR003_DRY_RUN=0\n', encoding="utf-8")
            vals = r020.read_env(env)
            self.assertNotIn("R003_EXTRA_ARGS", vals)
            self.assertEqual(r020.build_command(vals, python="py", script=Path("s"))[0], ["py", "s"])


class ProcessBackendTests(unittest.TestCase):
    """r016 start/stop/status with SERVICE_BACKEND=process against a real r020 + fake r003."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "live_trading").mkdir()
        self.root = root
        self.paths = r016.ControlPaths(runtime_root=root / "runtime", live_trading_dir=root / "live_trading")
        self.paths.watchlist.write_text("005930,삼성전자\n", encoding="utf-8")
        fake = root / "fake_r003.py"
        fake.write_text(FAKE_R003, encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {"R020_R003_SCRIPT": str(fake), "R020_RESTART_SEC": "0.5",
                                                "R020_STOP_TIMEOUT": "10", "AUTO_TRADING_RUNTIME": str(root / "runtime"),
                                                "FAKE_R003_LIB": str(ROOT / "live_trading")})
        self.env.start()
        # the fake gets the live_trading dir as argv[1] (to import r019): pass it through R003_EXTRA_ARGS
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.paths.env_file.write_text(f'R003_EXTRA_ARGS="{(ROOT / "live_trading").as_posix()}"\n', encoding="utf-8")
        self.backend = mock.patch.object(r016, "SERVICE_BACKEND", "process")
        self.backend.start()
        self.rp = r020.Paths(self.paths.runtime_root)

    def tearDown(self):
        # never leave a supervisor behind
        if r019.is_locked(self.rp.lock):
            self.rp.stop_flag.touch()
            self.rp.r003_stop.touch()
            wait_for(lambda: not r019.is_locked(self.rp.lock), 20)
        self.backend.stop()
        self.env.stop()
        for _ in range(20):  # Windows: files of just-ended processes can stay busy for a moment
            try:
                self.td.cleanup()
                break
            except OSError:
                time.sleep(0.3)

    def cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = r016.main(list(argv), paths=self.paths)
        return rc, json.loads(buf.getvalue())

    def mode(self, m: str):
        (self.paths.runtime_root / "fake_mode.txt").write_text(m, encoding="utf-8")

    def test_start_status_stop(self):
        rc, out = self.cli("start", "--dry-run", "--profile", "conservative")
        self.assertEqual(rc, 0, out)
        self.assertTrue(wait_for(lambda: r019.is_locked(self.paths.r003_lock)))
        rc, st = self.cli("status")
        self.assertEqual(st["service"]["active_state"], "active")
        self.assertTrue(st["running"])
        self.assertEqual(st["external_r003"], [])  # the supervisor's child is not "external"
        self.assertEqual(st["host"]["backend"], "process")
        args = (self.paths.runtime_root / "fake_args.txt").read_text(encoding="utf-8")
        self.assertIn("--dry-run", args)
        self.assertTrue(args.endswith("|conservative"))
        rc, out = self.cli("start")
        self.assertEqual(rc, 3)  # already active
        rc, out = self.cli("stop")
        self.assertEqual(rc, 0, out)
        self.assertFalse(r019.is_locked(self.rp.lock))
        self.assertFalse(r019.is_locked(self.paths.r003_lock))
        rc, st = self.cli("status")
        self.assertEqual(st["service"]["active_state"], "inactive")
        self.assertFalse(st["running"])

    def test_crash_restarts_then_start_limit(self):
        self.mode("exit1")
        rc, out = self.cli("start")
        self.assertEqual(rc, 0, out)
        self.assertTrue(wait_for(lambda: not r019.is_locked(self.rp.lock), 30))
        st = r020.service_state(self.rp)
        self.assertEqual(st["active_state"], "failed")
        self.assertIn("start-limit-hit", st["result"])
        self.assertEqual(st["restarts"], 2)

    def test_refused_rc3_is_not_restarted(self):
        self.mode("exit3")
        self.cli("start")
        self.assertTrue(wait_for(lambda: not r019.is_locked(self.rp.lock), 20))
        st = r020.service_state(self.rp)
        self.assertEqual(st["active_state"], "failed")
        self.assertEqual(st["restarts"], 0)

    def test_external_r003_blocks_start(self):
        fh = r019.acquire_instance_lock(self.paths.r003_lock)  # an r003 started by hand in a console
        try:
            rc, out = self.cli("start")
            self.assertEqual(rc, 3)
            self.assertTrue(out["external_r003"])
            rc, st = self.cli("status")
            self.assertEqual([p["pid"] for p in st["external_r003"]], [os.getpid()])
            rc, out = self.cli("role", "set", "standby")
            self.assertEqual(rc, 3)  # r003 runs here
        finally:
            fh.close()

    def test_standby_refuses_start_and_collector(self):
        rc, out = self.cli("role", "set", "standby", "--note", "laptop is primary")
        self.assertEqual(rc, 0, out)
        rc, out = self.cli("start")
        self.assertEqual(rc, 3)
        self.assertIn("standby", out["error"])
        rc, out = self.cli("collector", "start", "--force")
        self.assertEqual(rc, 3)
        rc, out = self.cli("role", "get")
        self.assertEqual(out["role"], "standby")
        rc, out = self.cli("history", "-n", "5")
        self.assertEqual(out["rows"][-1]["action"], "role")

    def test_stop_ends_an_r003_without_supervisor(self):
        # supervisor died / r003 started by hand: stop must still end it, and only report success once it is gone
        p = r019.spawn_detached([sys.executable, os.environ["R020_R003_SCRIPT"]], env=r019.child_env())
        self.assertTrue(wait_for(lambda: r019.is_locked(self.paths.r003_lock)))
        rc, out = self.cli("stop")
        self.assertEqual(rc, 0, out)
        self.assertFalse(r019.is_locked(self.paths.r003_lock))
        self.assertFalse(self.rp.r003_stop.exists())

    def test_stop_reports_failure_when_r003_ignores_it(self):
        self.mode("ignore")
        p = r019.spawn_detached([sys.executable, os.environ["R020_R003_SCRIPT"]], env=r019.child_env())
        self.assertTrue(wait_for(lambda: r019.is_locked(self.paths.r003_lock)))
        try:
            with mock.patch.object(r020, "STOP_TIMEOUT", 1.0):
                rc, out = self.cli("stop")
            self.assertEqual(rc, 3)
            self.assertIn("did not stop", out["stderr"])
        finally:
            owner = r019.running_owner(self.paths.r003_lock)
            if owner:
                r019.kill_tree(owner["ppid"] or owner["pid"])
                r019.terminate(owner["pid"])
            wait_for(lambda: not r019.is_locked(self.paths.r003_lock), 10)

    def test_supervisor_kills_the_real_r003_after_the_stop_timeout(self):
        self.mode("ignore")
        with mock.patch.dict(os.environ, {"R020_STOP_TIMEOUT": "1"}):
            rc, out = self.cli("start")
        self.assertEqual(rc, 0, out)
        self.assertTrue(wait_for(lambda: r019.is_locked(self.paths.r003_lock)))
        real = r019.running_owner(self.paths.r003_lock)["pid"]
        rc, out = self.cli("stop")
        self.assertEqual(rc, 0, out)
        self.assertFalse(r019.pid_alive(real))  # not only the venv launcher
        st = r020.service_state(self.rp)
        self.assertEqual((st["active_state"], st["result"]), ("inactive", "timeout"))

    def test_stop_requested_right_after_start_is_honoured(self):
        self.rp.live.mkdir(parents=True, exist_ok=True)
        self.rp.stop_flag.touch()  # a stop that arrived before the supervisor's first look
        sup = r020.Supervisor(self.rp, by="t", restart_sec=0.1, stop_timeout=2)
        self.assertEqual(sup.run(), 0)
        self.assertEqual(r020.read_state(self.rp)["result"], "stopped-before-start")
        self.assertIsNone(r020.read_state(self.rp)["child_pid"])

    def test_start_clears_a_stale_stop_file(self):
        self.rp.live.mkdir(parents=True, exist_ok=True)
        self.rp.stop_flag.touch()
        self.rp.r003_stop.touch()
        rc, out = self.cli("start")
        self.assertEqual(rc, 0, out)
        self.assertTrue(wait_for(lambda: r019.is_locked(self.paths.r003_lock)))
        self.assertEqual(r020.service_state(self.rp)["active_state"], "active")

    def test_stale_active_state_without_supervisor_is_failed(self):
        r020.write_state(self.rp, {"pid": 1, "active_state": "active", "child_pid": 2})
        st = r020.service_state(self.rp)
        self.assertEqual(st["active_state"], "failed")
        self.assertEqual(st["result"], "supervisor-died")


class CollectorStartTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        self.paths = r016.ControlPaths(runtime_root=root / "runtime", live_trading_dir=root)

    def tearDown(self):
        self.td.cleanup()

    def test_window_and_spawn(self):
        from datetime import datetime
        calls = []

        def spawn(argv, **kw):
            calls.append(argv)
            return 777

        with mock.patch.object(r016, "_has_proc", return_value=False):
            out = r016.collector_start(self.paths, "t", now=datetime(2026, 11, 8, 10, 0), spawn=spawn)  # Sunday
            self.assertFalse(out["ok"])
            out = r016.collector_start(self.paths, "t", now=datetime(2026, 11, 10, 8, 40), spawn=spawn)  # before 08:45
            self.assertFalse(out["ok"])
            out = r016.collector_start(self.paths, "t", now=datetime(2026, 11, 10, 8, 46), spawn=spawn)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["pid"], 777)
            self.assertIn("--fill-from-scan", calls[-1])
            out = r016.collector_start(self.paths, "t", now=datetime(2026, 11, 10, 16, 0), spawn=spawn, force=True)
            self.assertTrue(out["ok"])
            fh = r019.acquire_instance_lock(self.paths.g011_lock)
            try:
                out = r016.collector_start(self.paths, "t", now=datetime(2026, 11, 10, 9, 30), spawn=spawn)
                self.assertTrue(out["already_running"])
                st = r016.collector_status(self.paths, [])
                self.assertTrue(st["running"])
                self.assertEqual(st["pid"], os.getpid())
            finally:
                fh.close()


class JobTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.paths = r021.Paths(Path(self.td.name) / "runtime")

    def tearDown(self):
        for _ in range(20):
            try:
                self.td.cleanup()
                break
            except OSError:
                time.sleep(0.3)

    def fake_steps(self, rcs: dict, sleep: float = 0.0):
        steps = {}
        for name in ("collect", "history", "scan"):
            code = f"import time; time.sleep({sleep}); print('{name} out'); raise SystemExit({rcs.get(name, 0)})"
            steps[name] = r021.Step(name, (lambda c: lambda a, rc: ["-c", c])(code), required=(name != "history"))
        return mock.patch.dict(r021.STEPS, steps)

    def run_job(self, name="postclose"):
        return r021.run_detached_entry(name, r021.JobArgs(date="20261110"), "t", self.paths)

    def test_postclose_all_steps(self):
        with self.fake_steps({}):
            self.assertEqual(self.run_job(), 0)
        st = r021.read_state(self.paths, "postclose")
        self.assertEqual(st["state"], "done")
        self.assertEqual([s["name"] for s in st["steps"]], ["collect", "history", "scan"])
        log = r021.tail_log(self.paths, "postclose")
        self.assertTrue(any("scan out" in ln for ln in log["lines"]))

    def test_required_failure_skips_rest_optional_failure_continues(self):
        with self.fake_steps({"collect": 2}):
            self.assertEqual(self.run_job(), 1)
        st = r021.read_state(self.paths, "postclose")
        self.assertEqual((st["state"], [s["name"] for s in st["steps"]]), ("failed", ["collect"]))
        with self.fake_steps({"history": 1}):
            self.run_job()
        st = r021.read_state(self.paths, "postclose")
        self.assertEqual(st["state"], "done_with_errors")
        self.assertEqual([s["name"] for s in st["steps"]], ["collect", "history", "scan"])

    def test_busy_runner_refuses_and_dead_runner_is_failed(self):
        fh = r019.acquire_instance_lock(self.paths.lock)
        try:
            fh.write("collect\n")
            fh.flush()
            out = r021.start_job(self.paths, "scan", r021.JobArgs(date="20261110"), "t", spawn=lambda *a, **k: 1)
            self.assertFalse(out["ok"])
            self.assertEqual(r021.list_jobs(self.paths)["running"]["name"], "collect")
        finally:
            fh.close()
        r021._write_json(self.paths.state("scan"), {"name": "scan", "state": "running"})
        row = [j for j in r021.list_jobs(self.paths)["jobs"] if j["name"] == "scan"][0]
        self.assertEqual(row["last"]["state"], "failed")

    def test_start_reports_a_runner_that_never_came_up(self):
        with mock.patch.object(r021, "START_CONFIRM_SEC", 0.5):
            out = r021.start_job(self.paths, "scan", r021.JobArgs(date="20261110"), "t", spawn=lambda *a, **k: 4242)
        self.assertFalse(out["ok"])
        self.assertIn("did not start", out["error"])

    def test_trading_day_only_skips_weekend(self):
        import argparse
        args = argparse.Namespace(action="start", name="postclose", date="20261108", workers=None, edge_ai=False,
                                  trading_day_only=True, n=10)
        out, rc = r021.run_job_command(args, "t", self.paths)
        self.assertEqual((rc, out.get("skipped")), (0, True))

    def test_stop_ends_grandchildren_too(self):
        import threading
        td = Path(self.td.name)
        lock = td / "grandchild.lock"
        (td / "worker.py").write_text(textwrap.dedent('''
            import sys, time
            from pathlib import Path
            sys.path.insert(0, sys.argv[1])
            import r019_platform as r
            fh = r.acquire_instance_lock(Path(sys.argv[2]))
            time.sleep(60)
        '''), encoding="utf-8")
        (td / "parent.py").write_text(textwrap.dedent('''
            import subprocess, sys, time
            subprocess.Popen([sys.executable] + sys.argv[1:])
            time.sleep(60)
        '''), encoding="utf-8")
        argv = [str(td / "parent.py"), str(td / "worker.py"), str(ROOT / "live_trading"), str(lock)]
        steps = {"collect": r021.Step("collect", lambda a, rc: argv)}
        with mock.patch.dict(r021.STEPS, steps):
            t = threading.Thread(target=self.run_job, args=("collect",))
            t.start()
            self.assertTrue(wait_for(lambda: r019.is_locked(lock), 20))
            self.paths.stop("collect").touch()
            t.join(60)
        self.assertTrue(wait_for(lambda: not r019.is_locked(lock), 10))  # the step's worker is gone too
        self.assertEqual(r021.read_state(self.paths, "collect")["state"], "stopped")

    def test_bad_input(self):
        self.assertFalse(r021.start_job(self.paths, "nope", r021.JobArgs(date="20261110"), "t")["ok"])
        self.assertFalse(r021.start_job(self.paths, "scan", r021.JobArgs(date="2026-11-10"), "t")["ok"])

    def test_stop_request_ends_current_step(self):
        import threading
        with self.fake_steps({}, sleep=30):
            t = threading.Thread(target=self.run_job)
            t.start()
            self.assertTrue(wait_for(lambda: (r021.read_state(self.paths, "postclose") or {}).get("steps")))
            self.paths.stop("postclose").touch()
            t.join(40)
        st = r021.read_state(self.paths, "postclose")
        self.assertEqual(st["state"], "stopped")
        self.assertEqual(len(st["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
