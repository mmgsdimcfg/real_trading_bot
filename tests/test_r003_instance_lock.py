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
        import fcntl
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / ".r003.lock"
            with mock.patch.object(r003, "R003_INSTANCE_LOCK_PATH", lock_path), mock.patch.object(r003, "_R003_LOCK_FH", None):
                self.assertTrue(r003._acquire_instance_lock())
                self.assertIn(str(os.getpid()), lock_path.read_text(encoding="utf-8"))
                other = open(lock_path, "a+")
                with self.assertRaises(OSError):
                    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                r003._R003_LOCK_FH.close()   # release
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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


if __name__ == "__main__":
    unittest.main()
