# -*- coding: utf-8 -*-
"""R020 - r003 감시 프로세스 (systemd --user autobot-r003.service 를 대신한다. Windows 노트북 = AutoBotH 로컬 모드).

r016_control.py start 가 이 스크립트를 분리 실행(r019.spawn_detached)한다. AutoBotH 나 r016 이 끝나도 계속 돈다.

  python live_trading/r020_supervisor.py run [--by NAME]

하는 일 (autobot-r003.service + r003_service.sh 와 같은 규칙):
- <runtime>/config/r003.env 를 데이터로 읽어(셸 코드 실행 없음) r003 인자/환경을 만든다:
  R003_DRY_RUN=1 -> --dry-run, AUTO_TRADING_RISK_PROFILE -> 환경변수, R003_DATE -> --date,
  R003_WATCHLIST_SOURCE -> --watchlist-source, R003_EXTRA_ARGS -> 공백으로 나눈 추가 인자. 재시작할 때마다 다시 읽는다.
- r003 을 자식으로 실행(콘솔 창 없음, stdout/stderr -> <runtime>/log/<YYYYMMDD>/r003_service_stdout.log = journal 대신).
- 정상 종료(rc 0, 장 마감 등)면 감시도 끝낸다(Restart=on-failure). 비정상 종료면 RESTART_SEC 뒤 다시 실행하되
  START_LIMIT_INTERVAL 안에 START_LIMIT_BURST 번을 넘으면 failed 로 멈춘다. rc 3(이중 실행/standby 거부)은 다시 실행해도
  같으므로 바로 failed.
- 정지: r016 stop 이 <live_runtime>/r020.stop(감시 종료 표시)과 r003.stop(r003 정상 종료 요청)을 쓴다. r003 이 상태를 저장하고
  끝나기를 STOP_TIMEOUT 초 기다리고, 넘으면 강제 종료(TerminateProcess = systemd TimeoutStopSec 후 SIGKILL).
- 상태: <live_runtime>/r020_supervisor.json (r016 status 가 읽는다). 감시자 자신은 <live_runtime>/.r020.lock 을 잡는다
  (잠금이 풀려 있으면 감시자는 없는 것 - 상태 파일이 active 라도 무시).

g011 틱 수집기는 r003 이 띄우고(TICK_COLLECTOR_AUTOSTART), r003 이 끝나도 남는다(KillMode=process 와 같음).

Update log:
- [2026-11-10] type=feat owner=claude
    summary: 신규 - 윈도우에서 r003 을 앱과 독립적으로 실행/재시작/정상 정지 (systemd user 서비스 대체).
    impact: tooling (r003 매매 로직 변경 없음)
    compatibility: backward-compatible (신규 파일, Linux 는 기존 systemd 그대로)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r019_platform as r019  # noqa: E402

RESTART_SEC = float(os.environ.get("R020_RESTART_SEC") or 60)  # systemd RestartSec (env: tests)
START_LIMIT_INTERVAL = 600
START_LIMIT_BURST = 3
STOP_TIMEOUT = float(os.environ.get("R020_STOP_TIMEOUT") or 90)  # systemd TimeoutStopSec (env: tests)
POLL_SEC = 1.0
REFUSED_RC = 3  # r003: another instance holds the lock / host role standby
ENV_KEYS = ("R003_DRY_RUN", "AUTO_TRADING_RISK_PROFILE", "R003_DATE", "R003_WATCHLIST_SOURCE", "R003_EXTRA_ARGS")
UNSAFE_CHARS = set("`$;|&<>()")
UNIT_NAME = "r020-supervisor"


def runtime_root() -> Path:
    raw = os.environ.get("AUTO_TRADING_RUNTIME")
    return Path(raw) if raw else HERE.parents[2] / "runtime"


class Paths:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else runtime_root()
        self.live = self.root / "data" / "live_runtime"
        self.config = self.root / "config"

    @property
    def lock(self) -> Path:
        return self.live / ".r020.lock"

    @property
    def state(self) -> Path:
        return self.live / "r020_supervisor.json"

    @property
    def stop_flag(self) -> Path:
        return self.live / "r020.stop"

    @property
    def r003_stop(self) -> Path:
        return self.live / "r003.stop"

    @property
    def r003_lock(self) -> Path:
        return self.live / ".r003.lock"

    @property
    def env_file(self) -> Path:
        return self.config / "r003.env"

    def stdout_log(self, now: datetime | None = None) -> Path:
        return self.root / "log" / (now or datetime.now()).strftime("%Y%m%d") / "r003_service_stdout.log"


def r003_script() -> Path:
    """r003 to run (R020_R003_SCRIPT overrides it - tests use a fake r003)."""
    raw = os.environ.get("R020_R003_SCRIPT")
    return Path(raw) if raw else HERE / "r003_trade_live_execute.py"


def read_env(path: Path) -> dict:
    """Same parsing as r003_service.sh: KEY=VALUE lines, quotes stripped, unknown keys and unsafe values ignored."""
    out: dict = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] == '"':
            val = val[1:-1]
        if key not in ENV_KEYS or any(c in UNSAFE_CHARS for c in val):
            continue
        out[key] = val
    return out


def build_command(env_values: dict, python: str | None = None, script: Path | None = None) -> tuple[list[str], dict]:
    """(argv, extra environment) for one r003 run."""
    argv = [python or r019.python_exe(), str(script or r003_script())]
    if env_values.get("R003_DRY_RUN") == "1":
        argv.append("--dry-run")
    if env_values.get("R003_DATE"):
        argv += ["--date", env_values["R003_DATE"]]
    if env_values.get("R003_WATCHLIST_SOURCE"):
        argv += ["--watchlist-source", env_values["R003_WATCHLIST_SOURCE"]]
    if env_values.get("R003_EXTRA_ARGS"):
        argv += shlex.split(env_values["R003_EXTRA_ARGS"], posix=True)
    extra = {}
    if env_values.get("AUTO_TRADING_RISK_PROFILE"):
        extra["AUTO_TRADING_RISK_PROFILE"] = env_values["AUTO_TRADING_RISK_PROFILE"]
    return argv, extra


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_state(paths: Paths, state: dict) -> None:
    state = dict(state, updated_at=_now())
    paths.live.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".r020_supervisor.", suffix=".tmp", dir=str(paths.live))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        for i in range(40):  # Windows: a reader holding the file open makes os.replace fail for a moment
            try:
                os.replace(tmp, paths.state)
                break
            except PermissionError:
                if i == 39:
                    raise
                time.sleep(0.05)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_state(paths: Paths) -> dict | None:
    try:
        data = json.loads(paths.state.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def service_state(paths: Paths) -> dict:
    """systemd-like ServiceState for r016 status (same keys as r016.service_state)."""
    owner = r019.running_owner(paths.lock)
    st = read_state(paths) or {}
    base = {"available": True, "unit": UNIT_NAME, "loaded": True, "backend": "process"}
    if owner is None:
        # no supervisor: whatever the file says is history. A file left 'active' = the supervisor died (crash / reboot)
        last = st.get("active_state")
        failed = last == "failed" or last in ("active", "activating", "deactivating")
        return {**base, "active_state": "failed" if failed else "inactive", "sub_state": "failed" if failed else "dead",
                "main_pid": None, "started_at": None, "started_epoch": None, "restarts": int(st.get("restarts") or 0),
                "result": st.get("result") or ("supervisor-died" if last in ("active", "activating", "deactivating") else "success"),
                "supervisor_pid": None}
    started = st.get("child_started_at") if st.get("active_state") == "active" else None
    epoch = None
    if started:
        with contextlib.suppress(ValueError):
            epoch = datetime.fromisoformat(started).timestamp()
    return {**base, "active_state": st.get("active_state") or "activating", "sub_state": st.get("sub_state") or "start",
            "main_pid": st.get("child_pid"), "started_at": started, "started_epoch": epoch,
            "restarts": int(st.get("restarts") or 0), "result": st.get("result") or "success",
            "supervisor_pid": owner.get("pid")}


def tree_pids(paths: Paths) -> set[str]:
    st = service_state(paths)
    return {str(p) for p in (st.get("supervisor_pid"), st.get("main_pid")) if p}


# ---------------------------------------------------------------------------
# supervisor loop
# ---------------------------------------------------------------------------
class Supervisor:
    def __init__(self, paths: Paths, by: str = "r016", restart_sec: float = RESTART_SEC, stop_timeout: float = STOP_TIMEOUT):
        self.paths = paths
        self.by = by
        self.restart_sec = restart_sec
        self.stop_timeout = stop_timeout
        self.state: dict = {"pid": os.getpid(), "by": by, "started_at": _now(), "active_state": "activating",
                            "sub_state": "start", "restarts": 0, "result": "success", "child_pid": None,
                            "child_started_at": None, "last_rc": None, "argv": None}
        self.starts: list[float] = []

    def _set(self, **kw) -> None:
        self.state.update(kw)
        write_state(self.paths, self.state)

    def stop_requested(self) -> bool:
        return self.paths.stop_flag.exists()

    def _spawn(self) -> subprocess.Popen:
        argv, extra = build_command(read_env(self.paths.env_file))
        extra["R003_SPAWNED_AT"] = f"{time.time():.3f}"  # r003 keeps a stop file written after this (stop during start-up)
        log_path = self.paths.stdout_log()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab")
        try:
            fh.write(f"\n===== {_now()} r020 start {' '.join(argv[1:])}\n".encode("utf-8"))
            fh.flush()
            kw: dict = {"cwd": str(r003_script().parent), "stdin": subprocess.DEVNULL, "stdout": fh, "stderr": subprocess.STDOUT,
                        "env": r019.child_env(extra)}
            if r019.IS_WINDOWS:
                kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW (its g011 child gets no window either)
            proc = subprocess.Popen(argv, **kw)
        finally:
            fh.close()
        self.starts.append(time.time())
        self._set(active_state="active", sub_state="running", child_pid=proc.pid, child_started_at=_now(), argv=argv[1:])
        return proc

    def _wait_child(self, proc: subprocess.Popen) -> tuple[int | None, bool]:
        """Wait for the child; returns (rc, stopped_by_request)."""
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc, self.stop_requested()
            if self.stop_requested():
                self._set(active_state="deactivating", sub_state="stop-sigterm")
                with contextlib.suppress(OSError):
                    self.paths.r003_stop.parent.mkdir(parents=True, exist_ok=True)
                    self.paths.r003_stop.touch()
                deadline = time.time() + self.stop_timeout
                while proc.poll() is None and time.time() < deadline:
                    time.sleep(POLL_SEC)
                if proc.poll() is None:
                    self._set(sub_state="stop-sigkill")
                    self._kill_child(proc)
                    self.state["result"] = "timeout"
                return proc.poll(), True
            time.sleep(POLL_SEC)

    def _our_r003(self, proc: subprocess.Popen) -> int | None:
        """pid of the r003 holding the instance lock when it is our child (Windows venv: the launcher's child)."""
        owner = r019.running_owner(self.paths.r003_lock)
        if owner and (owner.get("pid") == proc.pid or owner.get("ppid") == proc.pid):
            return owner.get("pid")
        return None

    def _kill_child(self, proc: subprocess.Popen) -> None:
        """systemd SIGKILL after TimeoutStopSec: end the real r003 first (Windows: the venv launcher's child - ending only the
        launcher must not leave it trading), then the process we started. Its g011 child is left alone (KillMode=process)."""
        real = self._our_r003(proc)
        if real and real != proc.pid:
            r019.terminate(real)
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        if not r019.wait_unlocked(self.paths.r003_lock, 10):
            still = r019.running_owner(self.paths.r003_lock) or {}
            self.state["kill_failed_pid"] = still.get("pid")

    def _sleep_or_stop(self, seconds: float) -> bool:
        """True when a stop was requested during the wait."""
        end = time.time() + seconds
        while time.time() < end:
            if self.stop_requested():
                return True
            time.sleep(min(POLL_SEC, max(0.0, end - time.time())))
        return self.stop_requested()

    def run(self) -> int:
        # stale stop files are removed by r016 start BEFORE it spawns us: one that exists now was written by a stop that came
        # right after the start - honour it instead of deleting it
        self._set()
        if self.stop_requested():
            self._set(active_state="inactive", sub_state="dead", result="stopped-before-start")
            return 0
        while True:
            proc = self._spawn()
            rc, stopped = self._wait_child(proc)
            self._set(last_rc=rc, child_pid=None)
            if self.state.get("kill_failed_pid"):
                self._set(active_state="failed", sub_state="failed",
                          result=f"stop-kill-failed (r003 pid {self.state['kill_failed_pid']} still holds the lock)")
                return 1
            if stopped:
                self._set(active_state="inactive", sub_state="dead", result=self.state.get("result") or "success")
                return 0
            if rc == 0:
                self._set(active_state="inactive", sub_state="dead", result="success")
                return 0
            if rc == REFUSED_RC:
                self._set(active_state="failed", sub_state="failed", result=f"exit-code {rc} (refused: lock/role)")
                return 1
            recent = [t for t in self.starts if time.time() - t < START_LIMIT_INTERVAL]
            if len(recent) >= START_LIMIT_BURST:
                self._set(active_state="failed", sub_state="failed", result=f"start-limit-hit (last exit-code {rc})")
                return 1
            self._set(active_state="activating", sub_state="auto-restart", result=f"exit-code {rc}",
                      restarts=int(self.state.get("restarts") or 0) + 1)
            if self._sleep_or_stop(self.restart_sec):
                self._set(active_state="inactive", sub_state="dead")
                return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="r003 supervisor (Windows replacement of the systemd user service)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--by", default=os.environ.get("R016_BY") or "r016")
    p.add_argument("--runtime", default=None, help="runtime root (passed by r016 - a WMI-started process has no inherited env)")
    sub.add_parser("state")
    args = ap.parse_args(argv)
    if getattr(args, "runtime", None):
        os.environ["AUTO_TRADING_RUNTIME"] = args.runtime  # also for the r003 child
    paths = Paths()
    if args.cmd == "state":
        print(json.dumps(service_state(paths), ensure_ascii=False, indent=2))
        return 0
    fh = r019.acquire_instance_lock(paths.lock)
    if fh is None:
        print(f"[r020] another supervisor holds {paths.lock}", flush=True)
        return 3
    try:
        sup = Supervisor(paths, by=args.by)
        try:
            return sup.run()
        except Exception as exc:  # never leave the state file saying 'active'
            sup._set(active_state="failed", sub_state="failed", result=f"supervisor-error {type(exc).__name__}: {exc}")
            raise
        finally:
            with contextlib.suppress(OSError):
                paths.stop_flag.unlink()
    finally:
        fh.close()


if __name__ == "__main__":
    sys.exit(main())
