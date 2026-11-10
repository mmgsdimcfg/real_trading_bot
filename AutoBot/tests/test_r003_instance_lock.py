# -*- coding: utf-8 -*-
"""r003 single-instance flock + running snapshot (2026-11-05, remote control safety)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r003_trade_live_execute as r003  # noqa: E402


class InstanceLockTests(unittest.TestCase):
    def test_second_holder_is_refused_until_released(self):
        import r019_platform as r019  # [2026-11-10] flock (Linux) / msvcrt (Windows)
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / ".r003.lock"
            with mock.patch.object(r003, "R003_INSTANCE_LOCK_PATH", lock_path), mock.patch.object(r003, "_R003_LOCK_FH", None):
                self.assertTrue(r003._acquire_instance_lock())
                self.assertIn(str(os.getpid()), lock_path.read_text(encoding="utf-8"))
                other = open(lock_path, "a+")
                self.assertFalse(r019.try_lock(other))
                r003._R003_LOCK_FH.close()   # release
                self.assertTrue(r019.try_lock(other))
                with mock.patch.object(r003, "_R003_LOCK_FH", None):
                    self.assertFalse(r003._acquire_instance_lock())
                other.close()


class SnapshotTests(unittest.TestCase):
    def test_snapshot_written_and_removed_only_by_owner(self):
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "r003_running.json"
            wl = Path(td) / "r004.txt"
            wl.write_text("005930,x\n", encoding="utf-8")
            with mock.patch.object(r003, "R003_RUNNING_SNAPSHOT_PATH", snap):
                r003._write_running_snapshot(dry_run=True, env_dv="real", watch_file=wl, code_count=1)
                data = json.loads(snap.read_text(encoding="utf-8"))
                self.assertEqual(data["pid"], os.getpid())
                self.assertTrue(data["dry_run"])
                self.assertEqual(data["watchlist_codes"], 1)
                self.assertAlmostEqual(data["watchlist_mtime"], wl.stat().st_mtime)
                self.assertIsInstance(data["overrides_applied"], dict)
                self.assertIn("risk_profile", data)
                # another pid's file is left alone
                snap.write_text(json.dumps({"pid": 1}), encoding="utf-8")
                r003._remove_running_snapshot()
                self.assertTrue(snap.exists())
                snap.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
                r003._remove_running_snapshot()
                self.assertFalse(snap.exists())


class StopRequestAndRoleTests(unittest.TestCase):
    """[2026-11-10] stop-request file (r016 stop / r020 on Windows) and the standby host refusal."""

    def test_stop_request_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r003.stop"
            with mock.patch.object(r003, "R003_STOP_REQUEST_PATH", p):
                self.assertFalse(r003._stop_requested())
                p.touch()
                self.assertTrue(r003._stop_requested())
                r003._clear_stop_request()
                self.assertFalse(p.exists())
                r003._clear_stop_request()  # missing file is fine

    def test_stop_written_during_startup_is_kept_stale_one_removed(self):
        import time as _t
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r003.stop"
            now = _t.time()
            with mock.patch.object(r003, "R003_STOP_REQUEST_PATH", p), mock.patch.object(r003, "_R003_PROCESS_STARTED", now - 30):
                p.touch()
                os.utime(p, (now - 120, now - 120))  # left by an earlier run
                r003._clear_stale_stop_request()
                self.assertFalse(p.exists())
                p.touch()
                os.utime(p, (now - 10, now - 10))  # written while this process was importing (after it was spawned)
                r003._clear_stale_stop_request()
                self.assertTrue(p.exists())

    def test_spawn_baseline_from_supervisor(self):
        base = r003._R003_IMPORT_STARTED
        with mock.patch.dict(os.environ, {"R003_SPAWNED_AT": f"{base - 5:.3f}"}):
            self.assertAlmostEqual(r003._process_start_baseline(), base - 5, places=2)
        with mock.patch.dict(os.environ, {"R003_SPAWNED_AT": f"{base + 100:.3f}"}):  # implausible (future)
            self.assertEqual(r003._process_start_baseline(), base)
        with mock.patch.dict(os.environ, {"R003_SPAWNED_AT": "x"}):
            self.assertEqual(r003._process_start_baseline(), base)

    def test_standby_role_refuses_unless_ignored(self):
        import r019_platform as r019
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            data.mkdir()
            with mock.patch.object(r003, "RUNTIME_DATA_DIR", data), mock.patch.dict(os.environ, {"R003_IGNORE_ROLE": ""}):
                self.assertIsNone(r003._host_role_refusal())  # no file = primary
                r019.write_host_role(td, "standby", "autoboth@x")
                self.assertIn("standby", r003._host_role_refusal())
                with mock.patch.dict(os.environ, {"R003_IGNORE_ROLE": "1"}):
                    self.assertIsNone(r003._host_role_refusal())
                r019.write_host_role(td, "primary", "autoboth@x")
                self.assertIsNone(r003._host_role_refusal())


if __name__ == "__main__":
    unittest.main()
