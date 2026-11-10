# -*- coding: utf-8 -*-
"""r016_control (2026-11-05): remote control CLI for AutoBotH - overrides set/unset/validation, watchlist edit,
service actions through a fake systemctl runner, pending-restart detection."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402
    import r016_control as r016  # noqa: E402


class FakeSystemctl:
    def __init__(self, active=False, pid=0, started_mono_us=0):
        self.active, self.pid, self.started = active, pid, started_mono_us
        self.calls: list[list[str]] = []
        self.fail_verbs: set[str] = set()

    def __call__(self, cmd):
        self.calls.append(cmd)
        if cmd[:3] == ["systemctl", "--user", "show"]:
            state = "active" if self.active else "inactive"
            out = (f"ActiveState={state}\nSubState={'running' if self.active else 'dead'}\nMainPID={self.pid}\n"
                   f"ActiveEnterTimestampMonotonic={self.started}\nNRestarts=0\nResult=success\nLoadState=loaded\n")
            return subprocess.CompletedProcess(cmd, 0, out, "")
        verb = cmd[2]
        if verb in self.fail_verbs:
            return subprocess.CompletedProcess(cmd, 1, "", f"Failed to {verb}")
        if verb in ("start", "restart"):
            self.active, self.pid = True, 4242
        elif verb == "stop":
            self.active, self.pid = False, 0
        return subprocess.CompletedProcess(cmd, 0, "", "")


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "live_trading").mkdir()
        self.paths = r016.ControlPaths(runtime_root=root / "runtime", live_trading_dir=root / "live_trading")
        self.paths.watchlist.write_text("# header\n# Format: code,name\n\n005930,삼성전자\n000660,SK하이닉스\n", encoding="utf-8")
        self.sysctl = FakeSystemctl()
        self.sleep = mock.patch.object(r016.time, "sleep", lambda *_: None)
        self.sleep.start()
        # [2026-11-10] these tests cover the systemd backend + /proc process discovery on any OS (the Windows process
        # backend has its own tests in test_r020_supervisor.py)
        self.backend = mock.patch.object(r016, "SERVICE_BACKEND", "systemd")
        self.backend.start()
        self.hasproc = mock.patch.object(r016, "_has_proc", lambda proc_root=None: True)
        self.hasproc.start()

    def tearDown(self):
        self.hasproc.stop()
        self.backend.stop()
        self.sleep.stop()
        self.td.cleanup()

    def write_snapshot(self, **kw):
        snap = {"pid": os.getpid(), "started_at": "2026-11-05T08:30:00", "dry_run": False, "risk_profile": "neutral",
                "overrides_applied": {}, "watchlist_mtime": self.paths.watchlist.stat().st_mtime, "watchlist_codes": 2}
        snap.update(kw)
        self.paths.running_snapshot.parent.mkdir(parents=True, exist_ok=True)
        self.paths.running_snapshot.write_text(json.dumps(snap), encoding="utf-8")
        return snap

    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = r016.main(list(argv), paths=self.paths, runner=self.sysctl)
        return rc, json.loads(buf.getvalue())


class OverrideTests(Base):
    def test_set_valid_keys_writes_file_and_history(self):
        rc, out = self.run_cli("--by", "autoboth", "set", "ENABLE_PROFIT_LOCK=false", "PROFIT_LOCK_GIVEBACK_PCT=0.006", "EDGE_AI_MODE=shadow")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out["accepted"], {"ENABLE_PROFIT_LOCK": False, "PROFIT_LOCK_GIVEBACK_PCT": 0.006, "EDGE_AI_MODE": "shadow"})
        self.assertTrue(out["restart_required"])
        data = json.loads(self.paths.overrides.read_text(encoding="utf-8"))
        self.assertEqual(data["updated_by"], "autoboth")
        self.assertEqual(data["overrides"]["EDGE_AI_MODE"], "shadow")
        hist = [json.loads(l) for l in self.paths.history.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(hist[-1]["action"], "set")
        self.assertEqual(hist[-1]["changes"]["ENABLE_PROFIT_LOCK"], {"from": None, "to": False})
        # r001 accepts exactly what r016 wrote
        ns = {"ENABLE_PROFIT_LOCK": True, "PROFIT_LOCK_GIVEBACK_PCT": 0.004, "EDGE_AI_MODE": "gate"}
        with contextlib.redirect_stdout(io.StringIO()):
            applied = r001._apply_live_overrides(path=self.paths.overrides, namespace=ns)
        self.assertEqual(set(applied), set(out["accepted"]))
        self.assertEqual(ns["LIVE_OVERRIDES_ERRORS"], [])

    def test_set_rejects_bad_keys_and_keeps_good_ones(self):
        rc, out = self.run_cli("set", "NOPE=1", "RUNTIME_ROOT=/x", "ENABLE_PROFIT_LOCK=1", "MAX_OPEN_POSITIONS=abc", "MAX_OPEN_POSITIONS=2")
        self.assertEqual(rc, 2)
        self.assertEqual(out["accepted"], {"MAX_OPEN_POSITIONS": 2})
        self.assertIn("unknown", out["rejected"]["NOPE"])
        self.assertIn("not overridable", out["rejected"]["RUNTIME_ROOT"])
        self.assertIn("type mismatch", out["rejected"]["ENABLE_PROFIT_LOCK"])
        self.assertEqual(r016.read_overrides(self.paths), {"MAX_OPEN_POSITIONS": 2})

    def test_set_without_equals_is_an_error(self):
        rc, out = self.run_cli("set", "ENABLE_PROFIT_LOCK")
        self.assertEqual(rc, 2)
        self.assertFalse(self.paths.overrides.exists())

    def test_get_unset_reset(self):
        self.run_cli("set", "MAX_OPEN_POSITIONS=2", "ENABLE_SPREAD_GATE=false")
        rc, out = self.run_cli("get", "MAX_OPEN_POSITIONS", "POLL_INTERVAL_SECONDS")
        self.assertEqual(rc, 0)
        self.assertEqual(out["values"]["MAX_OPEN_POSITIONS"]["override"], 2)
        self.assertTrue(out["values"]["MAX_OPEN_POSITIONS"]["has_override"])
        self.assertFalse(out["values"]["MAX_OPEN_POSITIONS"]["pending"])  # nothing running -> applies at next start
        self.assertFalse(out["applied_state_known"])
        self.assertFalse(out["values"]["POLL_INTERVAL_SECONDS"]["has_override"])
        rc, out = self.run_cli("get", "NOPE")
        self.assertEqual(rc, 2)
        self.assertEqual(out["unknown"], ["NOPE"])
        rc, out = self.run_cli("unset", "MAX_OPEN_POSITIONS", "NEVER_SET")
        self.assertEqual(out["removed"], {"MAX_OPEN_POSITIONS": 2})
        self.assertEqual(out["not_set"], ["NEVER_SET"])
        self.assertEqual(r016.read_overrides(self.paths), {"ENABLE_SPREAD_GATE": False})
        rc, out = self.run_cli("reset")
        self.assertEqual(r016.read_overrides(self.paths), {})
        rc, out = self.run_cli("history", "-n", "10")
        self.assertEqual([r["action"] for r in out["rows"]], ["set", "unset", "reset"])

    def test_schema_default_is_enable_keys_plus_overrides(self):
        self.run_cli("set", "MAX_OPEN_POSITIONS=2")
        rc, out = self.run_cli("schema")
        keys = {r["key"]: r for r in out["keys"]}
        self.assertIn("ENABLE_PROFIT_LOCK", keys)
        self.assertIn("MAX_OPEN_POSITIONS", keys)
        self.assertNotIn("POLL_INTERVAL_SECONDS", keys)
        self.assertNotIn("RUNTIME_ROOT", keys)
        self.assertEqual(keys["MAX_OPEN_POSITIONS"]["override"], 2)
        rc, out = self.run_cli("schema", "--all")
        all_keys = {r["key"] for r in out["keys"]}
        self.assertIn("POLL_INTERVAL_SECONDS", all_keys)
        self.assertNotIn("RUNTIME_ROOT", all_keys)
        self.assertNotIn("TICK_DATA_DIR", all_keys)
        self.assertNotIn("LIVE_OVERRIDES_PATH", all_keys)

    def test_conditions_annotates_defines(self):
        self.run_cli("set", "ENABLE_MARKET_REGIME_GATE=false")
        self.write_snapshot(overrides_applied={})
        with mock.patch.object(r016, "_pid_is_r003", return_value=True):
            rc, out = self.run_cli("conditions")
        self.assertEqual(rc, 0)
        row = next(r for r in out["buy"] if r["func"] == "_010_market_regime")
        meta = row["values_meta"]["ENABLE_MARKET_REGIME_GATE"]
        self.assertTrue(meta["editable"])
        self.assertEqual(meta["override"], False)
        self.assertEqual(meta["running"], True)
        self.assertTrue(meta["pending"])
        self.assertTrue(out["applied_state_known"])
        self.assertEqual(out["overrides"], {"ENABLE_MARKET_REGIME_GATE": False})
        self.assertEqual(out["pending_keys"], ["ENABLE_MARKET_REGIME_GATE"])


class WatchlistTests(Base):
    def test_get(self):
        rc, out = self.run_cli("watchlist", "get")
        self.assertEqual([c["code"] for c in out["codes"]], ["005930", "000660"])
        self.assertEqual(out["codes"][1]["name"], "SK하이닉스")

    def test_set_backs_up_and_rewrites(self):
        rc, out = self.run_cli("--by", "autoboth", "watchlist", "set", "--codes", "035720,카카오;5930;035720,dup")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out["count"], 2)
        self.assertTrue(Path(out["backup"]).is_file())
        text = self.paths.watchlist.read_text(encoding="utf-8")
        self.assertIn("# Format: code,name", text)
        self.assertIn("# [r016] edited by autoboth", text)
        self.assertEqual([l for l in text.splitlines() if l and not l.startswith("#")], ["035720,카카오", "005930,005930"])

    def test_set_rejects_bad_code_and_empty(self):
        rc, out = self.run_cli("watchlist", "set", "--codes", "ABCDEF,x")
        self.assertEqual(rc, 2)
        rc, out = self.run_cli("watchlist", "set", "--codes", "")
        self.assertEqual(rc, 2)
        self.assertIn("005930", self.paths.watchlist.read_text(encoding="utf-8"))

    def test_set_from_stdin(self):
        with mock.patch.object(sys, "stdin", io.StringIO("# c\n000660,SK하이닉스\n\n005930,삼성전자\n")):
            rc, out = self.run_cli("watchlist", "set", "--stdin")
        self.assertEqual(out["count"], 2)


class ServiceTests(Base):
    def test_start_writes_env_and_calls_systemctl(self):
        with mock.patch.object(r016, "find_r003_processes", return_value=[]):
            rc, out = self.run_cli("start", "--dry-run", "--profile", "conservative")
        self.assertEqual(rc, 0, out)
        self.assertTrue(out["ok"])
        self.assertIn(["systemctl", "--user", "start", r016.SERVICE_NAME], self.sysctl.calls)
        env = r016.read_env_file(self.paths)
        self.assertEqual(env, {"R003_DRY_RUN": "1", "AUTO_TRADING_RISK_PROFILE": "conservative"})
        self.assertEqual(out["service"]["active_state"], "active")

    def test_start_refused_when_external_r003_runs(self):
        with mock.patch.object(r016, "find_r003_processes", return_value=[{"pid": 777, "cmdline": "python r003_trade_live_execute.py"}]):
            rc, out = self.run_cli("start")
        self.assertEqual(rc, 3)
        self.assertFalse(out["ok"])
        self.assertNotIn(["systemctl", "--user", "start", r016.SERVICE_NAME], self.sysctl.calls)

    def test_restart_live_flag_clears_dry_run_and_stop_failure_rc3(self):
        r016.write_env_file(self.paths, {"R003_DRY_RUN": "1"})
        with mock.patch.object(r016, "find_r003_processes", return_value=[]):
            rc, out = self.run_cli("restart", "--live")
        self.assertEqual(rc, 0)
        self.assertEqual(r016.read_env_file(self.paths)["R003_DRY_RUN"], "0")
        self.sysctl.fail_verbs.add("stop")
        rc, out = self.run_cli("stop")
        self.assertEqual(rc, 3)
        self.assertIn("Failed", out["stderr"])

    def test_env_file_rejects_unsafe_values(self):
        with self.assertRaises(ValueError):
            r016.write_env_file(self.paths, {"R003_EXTRA_ARGS": "--x; rm -rf /"})

    def test_status_pending_from_snapshot(self):
        self.sysctl.active, self.sysctl.pid = True, 999
        self.write_snapshot(overrides_applied={"MAX_OPEN_POSITIONS": 1, "ENABLE_SPREAD_GATE": False})
        self.run_cli("set", "MAX_OPEN_POSITIONS=1", "ENABLE_SPREAD_GATE=false")
        procs = [{"pid": os.getpid(), "cmdline": "python r003_trade_live_execute.py"}]
        with mock.patch.object(r016, "find_r003_processes", return_value=procs), \
             mock.patch.object(r016, "_pid_is_r003", return_value=True), \
             mock.patch.object(r016, "_service_tree_pids", return_value={str(os.getpid())}):
            rc, out = self.run_cli("status")
            self.assertTrue(out["running"])
            self.assertTrue(out["applied_state_known"])
            self.assertEqual(out["external_r003"], [])
            self.assertFalse(out["pending_restart"], out["pending"])
            self.assertIs(out["dry_run"], False)
            self.assertEqual(out["running_overrides"], {"MAX_OPEN_POSITIONS": 1, "ENABLE_SPREAD_GATE": False})
            # change one, add one, remove one, flip dry-run desire, touch the watchlist
            self.run_cli("set", "MAX_OPEN_POSITIONS=2", "ENABLE_PROFIT_LOCK=false")
            self.run_cli("unset", "ENABLE_SPREAD_GATE")
            r016.write_env_file(self.paths, {"R003_DRY_RUN": "1"})
            os.utime(self.paths.watchlist, (time.time() + 5, time.time() + 5))
            rc, out = self.run_cli("status")
            pend = out["pending"]
            self.assertEqual(pend["overrides"], {"added": ["ENABLE_PROFIT_LOCK"], "changed": ["MAX_OPEN_POSITIONS"], "removed": ["ENABLE_SPREAD_GATE"]})
            self.assertEqual(pend["pending_keys"], ["ENABLE_PROFIT_LOCK", "ENABLE_SPREAD_GATE", "MAX_OPEN_POSITIONS"])
            self.assertTrue(pend["env"]["dry_run"])
            self.assertTrue(pend["watchlist"])
            self.assertTrue(out["pending_restart"])
            self.assertTrue(out["desired"]["dry_run"])
            self.assertIs(out["dry_run"], False)     # running value, not the desired one
            rc, out = self.run_cli("get", "ENABLE_SPREAD_GATE")
            self.assertTrue(out["values"]["ENABLE_SPREAD_GATE"]["pending"])
            self.assertEqual(out["values"]["ENABLE_SPREAD_GATE"]["running"], False)
            self.assertFalse(out["values"]["ENABLE_SPREAD_GATE"]["has_override"])

    def test_status_stale_snapshot_is_ignored_and_legacy_fallback(self):
        self.write_snapshot(pid=1)   # pid 1 is not r003
        boot = r016._boot_epoch()
        self.sysctl.active, self.sysctl.pid = True, 999
        # +1 s: on Windows the boot epoch comes from time.monotonic() (coarse), so "now" could land a few ms before the watchlist
        # mtime written in setUp and count it as changed after the start
        self.sysctl.started = int((time.time() - boot + 1) * 1e6)
        procs = [{"pid": 999, "cmdline": "x"}, {"pid": 1000, "cmdline": "screen r003"}]
        with mock.patch.object(r016, "find_r003_processes", return_value=procs), \
             mock.patch.object(r016, "_service_tree_pids", return_value={"999"}):
            rc, out = self.run_cli("status")
            self.assertFalse(out["applied_state_known"])
            self.assertIsNone(out["dry_run"])
            self.assertEqual([p["pid"] for p in out["external_r003"]], [1000])
            self.assertFalse(out["pending_restart"])
            self.run_cli("set", "MAX_OPEN_POSITIONS=1")
            os.utime(self.paths.overrides, (time.time() + 5, time.time() + 5))
            rc, out = self.run_cli("status")
            self.assertEqual(out["pending"]["overrides"]["changed"], ["MAX_OPEN_POSITIONS"])
            self.assertTrue(out["pending_restart"])

    def test_restart_refused_with_external_r003_and_start_refused_when_active(self):
        ext = [{"pid": 777, "cmdline": "python r003_trade_live_execute.py"}]
        with mock.patch.object(r016, "find_r003_processes", return_value=ext):
            rc, out = self.run_cli("restart")
        self.assertEqual(rc, 3)
        self.assertIn("outside the service", out["error"])
        self.assertNotIn(["systemctl", "--user", "restart", r016.SERVICE_NAME], self.sysctl.calls)
        self.sysctl.active, self.sysctl.pid = True, 999
        with mock.patch.object(r016, "find_r003_processes", return_value=[]):
            rc, out = self.run_cli("start", "--dry-run")
        self.assertEqual(rc, 3)
        self.assertIn("already active", out["error"])
        self.assertFalse(self.paths.env_file.exists())   # desired mode NOT changed by a refused start

    def test_set_rejects_nan_inf_and_element_type_mismatch(self):
        rc, out = self.run_cli("set", "ACCOUNT_SYNC_MAX_STALE_SECONDS=NaN")
        self.assertEqual(rc, 2)
        self.assertIn("non-finite", out["error"])
        rc, out = self.run_cli("set", "ACCOUNT_SYNC_MAX_STALE_SECONDS=1e309", "QUOTE_STATUS_BLOCK_WARN_CODES=[2,3]",
                               'MARKET_REGIME_ETF={"KOSPI":null}', "TICK_TRIGGER_BAR_SIZES=[30]",
                               "BB_UPPER_GAP_MIN_PCT_BY_TICK=[[0.1,0.3]]", 'QUOTE_STATUS_BLOCK_WARN_CODES=["04"]')
        self.assertEqual(rc, 2)
        self.assertEqual(set(out["rejected"]), {"ACCOUNT_SYNC_MAX_STALE_SECONDS", "MARKET_REGIME_ETF"})
        self.assertEqual(out["accepted"], {"TICK_TRIGGER_BAR_SIZES": [30], "BB_UPPER_GAP_MIN_PCT_BY_TICK": [[0.1, 0.3]],
                                           "QUOTE_STATUS_BLOCK_WARN_CODES": ["04"]})
        self.assertNotIn("NaN", self.paths.overrides.read_text(encoding="utf-8"))

    def test_collector_status_and_restart(self):
        day = self.paths.ticks_dir / datetime.now().strftime("%Y%m%d")
        day.mkdir(parents=True)
        (day / "005930.csv").write_text("h\n", encoding="utf-8")
        procs = [{"pid": 4321, "cmdline": "python g011_tick_collector.py --codes 005930 --until 15:31",
                  "argv": ["python", "g011_tick_collector.py", "--codes", "005930", "--until", "15:31"]}]
        with mock.patch.object(r016, "_find_processes", return_value=procs):
            rc, out = self.run_cli("collector", "status")
            self.assertTrue(out["running"])
            self.assertEqual(out["subscribed"], ["005930"])
            self.assertEqual(out["missing"], ["000660"])
            self.assertIsNotNone(out["last_write"])
            with mock.patch.object(r016.os, "kill") as kill:
                rc, out = self.run_cli("collector", "restart")
            self.assertEqual(rc, 0)
            self.assertEqual(out["signalled"], [4321])
            kill.assert_called_once()
            rc, out = self.run_cli("watchlist", "set", "--codes", "000660,SK하이닉스")
            self.assertEqual(out["collector_missing"], ["000660"])
        with mock.patch.object(r016, "_find_processes", return_value=[]):
            rc, out = self.run_cli("collector", "restart")
            self.assertEqual(rc, 3)

    def test_concurrent_set_calls_do_not_lose_keys(self):
        import threading
        errors = []

        def worker(i):
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    r016.main(["set", f"MAX_OPEN_POSITIONS={i}", "ENABLE_PROFIT_LOCK=false"] if i % 2 else
                              ["set", "ENABLE_SPREAD_GATE=false", f"MAX_ENTRY_SPREAD_TICKS={i}"], paths=self.paths, runner=self.sysctl)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 9)]
        with contextlib.redirect_stdout(io.StringIO()):
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self.assertEqual(set(r016.read_overrides(self.paths)), {"MAX_OPEN_POSITIONS", "ENABLE_PROFIT_LOCK", "ENABLE_SPREAD_GATE", "MAX_ENTRY_SPREAD_TICKS"})
        self.assertEqual(len(self.paths.history.read_text(encoding="utf-8").splitlines()), 8)


class ChartDispatchTests(Base):
    def test_bars_and_indicators_dispatch_to_r017(self):
        import r017_chart_data as r017
        with mock.patch.object(r017, "bars", return_value={"ok": True, "bars": []}) as b:
            rc, out = self.run_cli("bars", "--date", "20261105", "--code", "5930", "--kind", "tick", "--ticks", "120", "--since-ts", "2026-11-05T09:00:00")
            self.assertEqual(rc, 0)
            b.assert_called_once_with("20261105", "5930", "tick", None, 1, 120, 0, "2026-11-05T09:00:00", False)
        with mock.patch.object(r017, "indicators", return_value={"ok": False, "error": "x"}) as i:
            rc, out = self.run_cli("indicators", "--date", "20261105", "--code", "005930", "--frame", "1m", "--last", "5")
            self.assertEqual(rc, 2)
            i.assert_called_once_with("20261105", "005930", "1m", 5)


class ProcScanTests(unittest.TestCase):
    def test_find_r003_processes_from_fake_proc(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td)
            (proc / "12").mkdir()
            (proc / "12" / "cmdline").write_bytes(b"/home/x/venv/bin/python\0r003_trade_live_execute.py\0--dry-run\0")
            (proc / "13").mkdir()
            (proc / "13" / "cmdline").write_bytes(b"python\0g011_tick_collector.py\0")
            (proc / "self").mkdir()
            found = r016.find_r003_processes(proc)
        self.assertEqual([f["pid"] for f in found], [12])
        self.assertIn("--dry-run", found[0]["cmdline"])


class LazyChartImportTest(unittest.TestCase):
    """status 폴링(AutoBotH 30 s)이 r017/pandas import 비용(~1 s, ~50 MB)을 내지 않는지."""

    def test_non_chart_command_does_not_import_r017_or_pandas(self):
        code = (
            "import contextlib, io, sys, tempfile\n"
            f"sys.path.insert(0, {str(ROOT / 'live_trading')!r})\n"
            "import r016_control as m\n"
            "with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):\n"
            "    from pathlib import Path\n"
            "    m.main(['history', '-n', '1'], paths=m.ControlPaths(runtime_root=Path(d), live_trading_dir=Path(d)))\n"
            "print('r017_chart_data' in sys.modules, 'pandas' in sys.modules)\n"
        )
        env = dict(os.environ, AUTO_TRADING_LIVE_OVERRIDES="0")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False False")

    def test_chart_command_args_are_parsed_by_r017(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as cm:
            r016.main(["bars", "--kind", "day"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--date", json.loads(out.getvalue())["error"])

    def test_unknown_option_on_other_command_is_rejected_as_json(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as cm:
            r016.main(["status", "--bogus"])
        self.assertEqual(cm.exception.code, 2)
        self.assertEqual(json.loads(out.getvalue())["ok"], False)

    def test_unknown_command_keeps_invalid_choice_text(self):  # AutoBotH detects old servers by "invalid choice"
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            r016.main(["nosuch"])
        self.assertIn("invalid choice", json.loads(out.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
