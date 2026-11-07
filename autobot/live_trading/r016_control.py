# -*- coding: utf-8 -*-
"""R016 - r003 원격 제어 CLI (AutoBotH 윈도우 앱이 SSH exec으로 호출). 출력은 항상 JSON 한 덩어리.

실행: cd real_trading_bot/autobot && ~/venv/bin/python live_trading/r016_control.py <command> [...]

  status                         서비스 상태(systemd --user autobot-r003), 실행 스냅샷(r003이 실제로 쓰는 값), 외부(screen 등)
                                 r003 PID, 틱 수집기(g011) 상태, 미적용 변경(desired vs running)
  start [--dry-run|--live] [--profile P]   systemctl --user start. 이미 실행 중(서비스/외부 r003)이면 거부(이중 주문 방지)
  stop | restart [--dry-run|--live] [--profile P]   restart 도 외부 r003 이 있으면 거부
  schema [--all]                 편집 가능한 r001 상수 목록 {key,type,base,override,effective,running,pending}
                                 (기본: ENABLE_* + 오버라이드/미적용 키, --all: 전체 편집 가능 키)
  get [KEY ...]                  현재 값
  set KEY=JSON ... [--by NAME]   오버라이드 저장(검증 통과분만). 값은 JSON (true/1.5/"gate"/[1,2]); JSON이 아니면 문자열
  unset KEY ... | reset          오버라이드 제거 / 전부 제거
  history [-n N]                 최근 변경 이력
  conditions                     r015 조건 JSON + 각 상수의 editable/base/override/running/pending
  watchlist get | watchlist set (--codes "005930,삼성전자;000660" | --stdin)   r004 감시목록 읽기/쓰기(백업 후 교체)
  collector status | collector restart      g011 틱 수집기 상태 / SIGTERM (g022 cron 또는 r003 감시자가 새 감시목록으로 재기동)
  indicators ... | bars ...        차트 데이터 (live_trading/r017_chart_data.py 로 위임 - 그 파일의 docstring 참조)

파일(저장소 밖): <runtime>/config/live_overrides.json, live_overrides_history.jsonl, r003.env, watchlist_backup/, .r016.lock
실행 스냅샷: <runtime>/data/live_runtime/r003_running.json (r003 이 시작 시 기록) - "지금 돌고 있는 값"의 근거.
적용 시점: r001은 import 시점에 live_overrides.json 을 읽으므로 설정/감시목록 변경은 r003 재시작 후 반영된다
(status.pending_restart). 핫리로드 없음. 이 스크립트는 주문을 내지 않고 브로커 모듈을 import 하지 않는다.
모든 쓰기(set/unset/reset/env/watchlist)는 <runtime>/config/.r016.lock flock 아래에서 읽기-수정-쓰기 하고 고유 임시 파일 + os.replace.

종료 코드: 0 성공, 2 검증 실패/거부, 3 systemctl 실패/실행 거부.

Update log:
- [2026-11-07] type=perf owner=claude
    summary: r017(pandas) 지연 import - indicators/bars 일 때만 불러와 r017 인자 정의로 다시 파싱. status 2.1 s/70 MB -> 0.4 s/17 MB
      (AutoBotH 30초 폴링, 라즈베리파이 부하). 그 밖의 명령에 모르는 옵션이 오면 기존처럼 rc 2.
    impact: tooling
    compatibility: backward-compatible (명령/출력 동일)
- [2026-11-06] type=feat owner=claude
    summary: AutoBotH 차트 요청 - indicators/bars 하위 명령(r017 위임), collector 블록에 started_at(/proc starttime) 추가.
    impact: tooling
    compatibility: backward-compatible
- [2026-11-05] type=fix owner=claude
    summary: Codex 검토 반영 - (1) restart 도 외부 r003 검사, 서비스 active 면 start 거부 (2) 쓰기 전부 flock + 고유 임시
      파일 (3) NaN/Infinity JSON 거부, allow_nan=False (4) status/pending 을 r003_running.json 스냅샷(실제 dry_run/profile/
      적용 오버라이드/감시목록 mtime) 과 desired 파일 비교로 판단 - 제거된 키도 pending, 스냅샷 없는 옛 r003 은
      applied_state_known=false (5) 틱 수집기 상태 + collector restart (감시목록 변경 후 구독 갱신) (6) 쓰기 결과에 수집기
      미구독 종목 경고.
    impact: tooling
    compatibility: backward-compatible (JSON 에 필드 추가)
- [2026-11-05] type=feat owner=claude
    summary: AutoBotH 원격 제어용 서버측 CLI 신설 - systemd user 서비스 start/stop/restart, r001 라이브 오버라이드
      편집(r001.live_override_* 검증 공유), r004 감시목록 편집, 미적용 변경 감지, 변경 이력.
    impact: tooling (r003 코드 변경 없음)
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402

SERVICE_NAME = "autobot-r003.service"
R003_SCRIPT_NAME = "r003_trade_live_execute.py"
G011_SCRIPT_NAME = "g011_tick_collector.py"
WATCHLIST_MAX_CODES = 100
ENV_KEYS = ("R003_DRY_RUN", "AUTO_TRADING_RISK_PROFILE", "R003_DATE", "R003_WATCHLIST_SOURCE", "R003_EXTRA_ARGS")
COLLECTOR_STALE_SECONDS = 60
MARKET_OPEN, MARKET_CLOSE = dt_time(9, 0), dt_time(15, 30)

Runner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass
class ControlPaths:
    runtime_root: Path
    live_trading_dir: Path

    @property
    def config_dir(self) -> Path:
        return self.runtime_root / "config"

    @property
    def lock(self) -> Path:
        return self.config_dir / ".r016.lock"

    @property
    def overrides(self) -> Path:
        return self.config_dir / "live_overrides.json"

    @property
    def history(self) -> Path:
        return self.config_dir / "live_overrides_history.jsonl"

    @property
    def env_file(self) -> Path:
        return self.config_dir / "r003.env"

    @property
    def watchlist(self) -> Path:
        return self.live_trading_dir / "r004_trade_watchlist_today.txt"

    @property
    def watchlist_backup_dir(self) -> Path:
        return self.config_dir / "watchlist_backup"

    @property
    def log_dir(self) -> Path:
        return self.runtime_root / "log"

    @property
    def running_snapshot(self) -> Path:
        return self.runtime_root / "data" / "live_runtime" / "r003_running.json"

    @property
    def ticks_dir(self) -> Path:
        return self.runtime_root / "data" / "ticks"


def default_paths() -> ControlPaths:
    return ControlPaths(runtime_root=Path(r001.RUNTIME_ROOT), live_trading_dir=Path(__file__).resolve().parent)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# atomic files + lock
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def write_lock(paths: ControlPaths) -> Iterator[None]:
    """Serialize every read-modify-write of the editable files (two SSH calls can overlap)."""
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    fh = open(paths.lock, "a+", encoding="utf-8")
    try:
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        fh.close()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name} is not allowed")


def _json_loads(text: str):
    return json.loads(text, parse_constant=_reject_constant)


# ---------------------------------------------------------------------------
# overrides file
# ---------------------------------------------------------------------------
def read_overrides(paths: ControlPaths) -> dict:
    if not paths.overrides.is_file():
        return {}
    try:
        data = _json_loads(paths.overrides.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict) and isinstance(data.get("overrides"), dict):
        return dict(data["overrides"])
    return dict(data) if isinstance(data, dict) else {}


def write_overrides(paths: ControlPaths, overrides: dict, by: str) -> None:
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), "updated_by": by,
               "overrides": {k: overrides[k] for k in sorted(overrides)}}
    _atomic_write(paths.overrides, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_history(paths: ControlPaths, action: str, changes: dict, by: str) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "by": by, "action": action, "changes": changes}
    with paths.history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    return "list"


def _jsonable(value):
    if isinstance(value, (tuple, set, frozenset, list)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def base_value(key: str):
    """Value before the live override (risk profile applied) - r001 remembers it when it applied the file."""
    if key in r001.LIVE_OVERRIDES_BASE:
        return r001.LIVE_OVERRIDES_BASE[key]
    return getattr(r001, key)


def editable_keys() -> list[str]:
    out = []
    for key in dir(r001):
        if key != key.upper() or key.startswith("_"):
            continue
        if r001.live_override_key_allowed(key, getattr(r001, key)):
            out.append(key)
    return sorted(out)


def validate_assignments(assignments: dict) -> tuple[dict, dict]:
    """Returns (accepted {key: value}, rejected {key: reason}) using the same rules as r001."""
    accepted: dict = {}
    rejected: dict = {}
    for key, value in assignments.items():
        if not hasattr(r001, key):
            rejected[key] = "unknown constant"
            continue
        current = base_value(key)
        if not r001.live_override_key_allowed(key, current):
            rejected[key] = "not overridable (path/dir/secret or non-scalar)"
            continue
        if not r001.live_override_value_compatible(current, value):
            rejected[key] = f"type mismatch (expected {_type_name(current)}, got {_type_name(value)}; elements must match, no NaN/inf)"
            continue
        accepted[key] = value
    return accepted, rejected


def parse_assignment(text: str) -> tuple[str, object]:
    if "=" not in text:
        raise ValueError(f"expected KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        value = _json_loads(raw)
    except ValueError as exc:
        if "non-finite" in str(exc):
            raise
        value = raw  # bare string such as gate / shadow
    return key, value


# ---------------------------------------------------------------------------
# running snapshot (written by r003) / desired vs running diff
# ---------------------------------------------------------------------------
def _pid_cmdline(pid: int, proc_root: Path = Path("/proc")) -> list[str] | None:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except Exception:
        return None
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def _pid_is_r003(pid: int) -> bool:
    argv = _pid_cmdline(pid)
    return bool(argv) and any(a.endswith(R003_SCRIPT_NAME) for a in argv)


def read_running_snapshot(paths: ControlPaths) -> dict | None:
    """r003_running.json if its pid is still an r003 process, else None (stale file from a crash is ignored)."""
    try:
        data = _json_loads(paths.running_snapshot.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    if not _pid_is_r003(int(data["pid"])):
        return None
    return data


def desired_overrides(paths: ControlPaths) -> tuple[dict, dict]:
    """What r003 WOULD apply on the next start: the file contents minus what r001 would reject."""
    return validate_assignments(read_overrides(paths))


def _same(a, b) -> bool:
    return _jsonable(a) == _jsonable(b)


def pending_diff(paths: ControlPaths, snapshot: dict | None, started_epoch: float | None) -> dict:
    """Desired (editable files) vs running (snapshot). Without a snapshot: file mtimes vs service start, state unknown."""
    desired, rejected = desired_overrides(paths)
    env = read_env_file(paths)
    desired_dry = env.get("R003_DRY_RUN", "0") == "1"
    desired_profile = env.get("AUTO_TRADING_RISK_PROFILE") or "neutral"
    wl_m = _mtime(paths.watchlist)
    out = {"applied_state_known": snapshot is not None, "overrides": {"added": [], "changed": [], "removed": []},
           "env": {"dry_run": False, "risk_profile": False}, "watchlist": False, "rejected_in_file": rejected}
    if snapshot is not None:
        applied = snapshot.get("overrides_applied") or {}
        for k, v in desired.items():
            if k not in applied:
                out["overrides"]["added"].append(k)
            elif not _same(applied[k], v):
                out["overrides"]["changed"].append(k)
        out["overrides"]["removed"] = sorted(k for k in applied if k not in desired)
        out["env"]["dry_run"] = bool(snapshot.get("dry_run")) != desired_dry
        out["env"]["risk_profile"] = str(snapshot.get("risk_profile") or "neutral") != desired_profile
        snap_wl = snapshot.get("watchlist_mtime")
        out["watchlist"] = bool(wl_m and snap_wl and abs(float(snap_wl) - wl_m) > 1e-6) or bool(wl_m and not snap_wl)
    else:
        # old r003 without a snapshot (or not running): fall back to "file changed after service start"
        for label, path in (("overrides", paths.overrides), ("env", paths.env_file), ("watchlist", paths.watchlist)):
            m = _mtime(path)
            changed = bool(m and started_epoch and m > started_epoch)
            if label == "overrides" and changed:
                out["overrides"]["changed"] = sorted(desired)
            elif label == "env" and changed:
                out["env"] = {"dry_run": True, "risk_profile": True}
            elif label == "watchlist":
                out["watchlist"] = changed
    keys = set(out["overrides"]["added"]) | set(out["overrides"]["changed"]) | set(out["overrides"]["removed"])
    out["pending_keys"] = sorted(keys)
    out["pending_restart"] = bool(keys or out["env"]["dry_run"] or out["env"]["risk_profile"] or out["watchlist"])
    return out


def schema_rows(paths: ControlPaths, diff: dict, snapshot: dict | None, include_all: bool = False) -> list[dict]:
    overrides = read_overrides(paths)
    applied = (snapshot or {}).get("overrides_applied") or {}
    pending = set(diff["pending_keys"])
    rows = []
    for key in editable_keys():
        base = base_value(key)
        running = applied[key] if key in applied else (base if snapshot is not None else None)
        row = {"key": key, "type": _type_name(base), "base": _jsonable(base),
               "override": _jsonable(overrides.get(key)) if key in overrides else None,
               "has_override": key in overrides, "effective": _jsonable(getattr(r001, key)),
               "running": _jsonable(running), "pending": key in pending}
        if include_all or key in overrides or key in pending or key.startswith("ENABLE_"):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# service / process state
# ---------------------------------------------------------------------------
def _boot_epoch() -> float:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except Exception:
        pass
    return time.time() - time.monotonic()


def service_state(runner: Runner = _run) -> dict:
    props = ["ActiveState", "SubState", "MainPID", "ActiveEnterTimestampMonotonic", "NRestarts", "Result", "LoadState"]
    try:
        cp = runner(["systemctl", "--user", "show", SERVICE_NAME, "-p", ",".join(props)])
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if cp.returncode != 0:
        return {"available": False, "error": (cp.stderr or cp.stdout).strip()}
    kv = {}
    for line in cp.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    started = None
    mono = kv.get("ActiveEnterTimestampMonotonic", "0")
    if mono.isdigit() and int(mono) > 0 and kv.get("ActiveState") == "active":
        started = _boot_epoch() + int(mono) / 1e6
    pid = int(kv["MainPID"]) if kv.get("MainPID", "").isdigit() else 0
    return {"available": True, "unit": SERVICE_NAME, "loaded": kv.get("LoadState") == "loaded",
            "active_state": kv.get("ActiveState"), "sub_state": kv.get("SubState"), "main_pid": pid or None,
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds") if started else None,
            "started_epoch": started, "restarts": int(kv.get("NRestarts") or 0), "result": kv.get("Result")}


def _find_processes(script_name: str, proc_root: Path = Path("/proc")) -> list[dict]:
    found = []
    try:
        names = os.listdir(proc_root)
    except Exception:
        return found
    for name in names:
        if not name.isdigit():
            continue
        parts = _pid_cmdline(int(name), proc_root)
        if parts and any(p.endswith(script_name) for p in parts):
            found.append({"pid": int(name), "cmdline": " ".join(parts), "argv": parts})
    return found


def find_r003_processes(proc_root: Path = Path("/proc")) -> list[dict]:
    """Every running r003 (systemd or screen/terminal) - used to refuse a second instance."""
    return [{"pid": p["pid"], "cmdline": p["cmdline"]} for p in _find_processes(R003_SCRIPT_NAME, proc_root)]


def _service_tree_pids(main_pid: int | None, proc_root: Path = Path("/proc")) -> set[str]:
    """main pid + its direct children (the bash wrapper's python child is the real r003)."""
    if not main_pid:
        return set()
    out = {str(main_pid)}
    try:
        for name in os.listdir(proc_root):
            if not name.isdigit():
                continue
            stat = (proc_root / name / "stat").read_text()
            ppid = stat.rsplit(")", 1)[1].split()[1]
            if ppid == str(main_pid):
                out.add(name)
    except Exception:
        pass
    return out


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def latest_main_log(paths: ControlPaths, date_str: str | None = None) -> str | None:
    date_str = date_str or datetime.now().strftime("%Y%m%d")
    day = paths.log_dir / date_str
    try:
        logs = [p for p in day.glob("r003_trade_live_execute_*.log") if not p.name.endswith("_buy_sell.log")]
    except Exception:
        return None
    if not logs:
        return None
    return str(max(logs, key=lambda p: p.stat().st_mtime))


def read_env_file(paths: ControlPaths) -> dict:
    out = {}
    if not paths.env_file.is_file():
        return out
    for line in paths.env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def write_env_file(paths: ControlPaths, values: dict) -> None:
    lines = ["# r003 service environment (written by r016_control / AutoBotH). Parsed as data by r003_service.sh (not sourced)."]
    for k in ENV_KEYS:
        if k in values and values[k] not in (None, ""):
            v = str(values[k])
            if not re.fullmatch(r"[A-Za-z0-9_./:+=, -]*", v):
                raise ValueError(f"{k}: unsafe characters")
            lines.append(f'{k}="{v}"')
    _atomic_write(paths.env_file, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# tick collector (g011)
# ---------------------------------------------------------------------------
def _collector_codes_of(argv: list[str], ticks_day_dir: Path) -> set[str] | None:
    if "--codes" in argv:
        out = set()
        for a in argv[argv.index("--codes") + 1:]:
            if a.startswith("--"):
                break
            out.add(a.zfill(6))
        return out
    try:  # no --codes: last "CODES a,b,..." line of today's collector log
        text = (ticks_day_dir / "_collector.log").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    codes = None
    for ln in text.splitlines():
        if " CODES " in ln:
            codes = {c.strip().zfill(6) for c in ln.split(" CODES ", 1)[1].split(",") if c.strip()}
    return codes


def _pid_started_at(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
        start_ticks = int(stat.rsplit(")", 1)[1].split()[19])
        hz = os.sysconf("SC_CLK_TCK")
        return datetime.fromtimestamp(_boot_epoch() + start_ticks / hz).isoformat(timespec="seconds")
    except Exception:
        return None


def collector_status(paths: ControlPaths, watch_codes: list[str], now: datetime | None = None) -> dict:
    now = now or datetime.now()
    day = paths.ticks_dir / now.strftime("%Y%m%d")
    procs = _find_processes(G011_SCRIPT_NAME)
    out = {"running": bool(procs), "pid": procs[0]["pid"] if procs else None, "pids": [p["pid"] for p in procs],
           "started_at": _pid_started_at(procs[0]["pid"]) if procs else None,
           "subscribed": None, "symbols": 0, "last_write": None, "stale": [], "missing": [], "ticks_dir": str(day)}
    subscribed = _collector_codes_of(procs[0]["argv"], day) if procs else None
    if subscribed is not None:
        out["subscribed"] = sorted(subscribed)
        out["symbols"] = len(subscribed)
        out["missing"] = sorted(c for c in watch_codes if c not in subscribed)
    try:
        files = {p.stem: p.stat().st_mtime for p in day.glob("*.csv") if re.fullmatch(r"\d{6}", p.stem)}
    except Exception:
        files = {}
    if files:
        out["last_write"] = datetime.fromtimestamp(max(files.values())).isoformat(timespec="seconds")
        if not out["symbols"]:
            out["symbols"] = len(files)
    in_session = MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5
    if in_session and out["running"]:
        limit = now.timestamp() - COLLECTOR_STALE_SECONDS
        for code in watch_codes:
            m = files.get(code)
            if m is None or m < limit:
                out["stale"].append({"code": code, "last_write": datetime.fromtimestamp(m).isoformat(timespec="seconds") if m else None})
    return out


def collector_restart(paths: ControlPaths, by: str) -> dict:
    procs = _find_processes(G011_SCRIPT_NAME)
    if not procs:
        return {"ok": False, "error": "g011 tick collector is not running"}
    killed = []
    for p in procs:
        with contextlib.suppress(OSError):
            os.kill(p["pid"], signal.SIGTERM)
            killed.append(p["pid"])
    append_history(paths, "collector.restart", {"pids": killed}, by)
    return {"ok": bool(killed), "signalled": killed,
            "note": "g022 cron (*/5 09-16) or the r003 supervisor restarts g011 with the current r004 list within ~5 min"}


# ---------------------------------------------------------------------------
# status / service actions
# ---------------------------------------------------------------------------
def status(paths: ControlPaths, runner: Runner = _run) -> dict:
    svc = service_state(runner)
    procs = find_r003_processes()
    tree = _service_tree_pids(svc.get("main_pid"))
    external = [p for p in procs if str(p["pid"]) not in tree]
    started = svc.get("started_epoch")
    snapshot = read_running_snapshot(paths)
    diff = pending_diff(paths, snapshot, started)
    env = read_env_file(paths)
    overrides = read_overrides(paths)
    wl = read_watchlist(paths)
    running = bool(svc.get("active_state") == "active" or procs)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "service": {k: v for k, v in svc.items() if k != "started_epoch"},
        "running": running,
        "external_r003": external,
        "snapshot": None if snapshot is None else {k: v for k, v in snapshot.items() if k != "overrides_applied"},
        "running_overrides": (snapshot or {}).get("overrides_applied") if snapshot else None,
        "applied_state_known": snapshot is not None,
        "desired": {"dry_run": env.get("R003_DRY_RUN", "0") == "1", "risk_profile": env.get("AUTO_TRADING_RISK_PROFILE") or "neutral", "env": env},
        "dry_run": (bool(snapshot.get("dry_run")) if snapshot else None),
        "risk_profile": (snapshot.get("risk_profile") if snapshot else None),
        "override_count": len(overrides),
        "override_errors": list(r001.LIVE_OVERRIDES_ERRORS) + [f"{k}: {v}" for k, v in diff["rejected_in_file"].items()],
        "pending": diff,
        "pending_restart": diff["pending_restart"],
        "latest_main_log": latest_main_log(paths),
        "watchlist_codes": len(wl["codes"]),
        "collector": collector_status(paths, [c["code"] for c in wl["codes"]]),
    }


def service_action(paths: ControlPaths, verb: str, dry_run: bool | None, profile: str | None,
                   runner: Runner = _run, by: str = "cli") -> dict:
    if verb not in ("start", "stop", "restart"):
        raise ValueError(verb)
    svc = service_state(runner)
    if verb in ("start", "restart"):
        tree = _service_tree_pids(svc.get("main_pid"))
        external = [p for p in find_r003_processes() if str(p["pid"]) not in tree]
        if external:
            return {"ok": False, "error": "r003 already running outside the service (screen/terminal?) - stop it first",
                    "external_r003": external}
        if verb == "start" and svc.get("active_state") == "active":
            return {"ok": False, "error": "service already active - use restart to apply changes",
                    "service": {k: v for k, v in svc.items() if k != "started_epoch"}}
        with write_lock(paths):
            env = read_env_file(paths)
            changed = {}
            if dry_run is not None:
                env["R003_DRY_RUN"] = "1" if dry_run else "0"
                changed["R003_DRY_RUN"] = env["R003_DRY_RUN"]
            if profile:
                env["AUTO_TRADING_RISK_PROFILE"] = profile
                changed["AUTO_TRADING_RISK_PROFILE"] = profile
            if changed or not paths.env_file.is_file():
                write_env_file(paths, env)
                if changed:
                    append_history(paths, "env", changed, by)
    cp = runner(["systemctl", "--user", verb, SERVICE_NAME])
    ok = cp.returncode == 0
    append_history(paths, f"service.{verb}", {"ok": ok}, by)
    out = {"ok": ok, "verb": verb, "stderr": (cp.stderr or "").strip()}
    if ok:
        time.sleep(1.0)
        out["service"] = {k: v for k, v in service_state(runner).items() if k != "started_epoch"}
    return out


# ---------------------------------------------------------------------------
# watchlist (r004)
# ---------------------------------------------------------------------------
def read_watchlist(paths: ControlPaths) -> dict:
    header: list[str] = []
    codes: list[dict] = []
    if paths.watchlist.is_file():
        for line in paths.watchlist.read_text(encoding="utf-8-sig").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header.append(s)
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 8:
                parts = parts[1:]
            code = parts[0].zfill(6)
            name = parts[1] if len(parts) >= 2 and parts[1] else code
            codes.append({"code": code, "name": name})
    m = _mtime(paths.watchlist)
    return {"path": str(paths.watchlist), "mtime": datetime.fromtimestamp(m).isoformat(timespec="seconds") if m else None,
            "header": header, "codes": codes}


def parse_watchlist_entries(text: str) -> list[dict]:
    """'005930,삼성전자;000660' or one entry per line -> [{code, name}]; duplicates keep the first; bad codes raise."""
    entries: list[dict] = []
    seen: set[str] = set()
    for chunk in re.split(r"[;\n]", text):
        s = chunk.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(",")]
        code = parts[0]
        if not re.fullmatch(r"\d{1,6}", code):
            raise ValueError(f"invalid code {code!r} (6 digits expected)")
        code = code.zfill(6)  # same leniency as r003.load_today_codes
        name = parts[1] if len(parts) >= 2 and parts[1] else code
        if "," in name or "\t" in name:
            raise ValueError(f"invalid name for {code}")
        if code in seen:
            continue
        seen.add(code)
        entries.append({"code": code, "name": name})
    if len(entries) > WATCHLIST_MAX_CODES:
        raise ValueError(f"too many codes ({len(entries)} > {WATCHLIST_MAX_CODES})")
    return entries


def write_watchlist(paths: ControlPaths, entries: list[dict], by: str) -> dict:
    with write_lock(paths):
        current = read_watchlist(paths)
        header = current["header"] or ["# Daily live trading watchlist.", "# Format: code,name"]
        header = [h for h in header if not h.startswith("# [r016]")]
        header.append(f"# [r016] edited by {by} at {datetime.now().isoformat(timespec='seconds')}")
        backup = None
        if paths.watchlist.is_file():
            paths.watchlist_backup_dir.mkdir(parents=True, exist_ok=True)
            backup = paths.watchlist_backup_dir / f"r004_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
            shutil.copy2(paths.watchlist, backup)
        body = "\n".join(header) + "\n\n" + "\n".join(f"{e['code']},{e['name']}" for e in entries) + "\n"
        _atomic_write(paths.watchlist, body)
        append_history(paths, "watchlist", {"codes": [e["code"] for e in entries], "backup": str(backup) if backup else None}, by)
    coll = collector_status(paths, [e["code"] for e in entries])
    return {"ok": True, "count": len(entries), "backup": str(backup) if backup else None, "path": str(paths.watchlist),
            "restart_required": True, "collector_missing": coll["missing"],
            "collector_note": ("running g011 does not subscribe these codes - run 'collector restart' (or wait for r003 restart) "
                               "so the tick/triple trigger can see them" if coll["missing"] else "")}


# ---------------------------------------------------------------------------
# conditions (r015 + override annotations)
# ---------------------------------------------------------------------------
def conditions(paths: ControlPaths, diff: dict, snapshot: dict | None) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        import r015_export_conditions as r015  # noqa: E402
    data = r015.build()
    overrides = read_overrides(paths)
    applied = (snapshot or {}).get("overrides_applied") or {}
    pending = set(diff["pending_keys"])
    for side in ("buy", "sell"):
        for row in data.get(side, []):
            meta = {}
            for key in row.get("defines", []):
                if not hasattr(r001, key):
                    continue
                base = base_value(key)
                running = applied[key] if key in applied else (base if snapshot is not None else None)
                meta[key] = {"editable": r001.live_override_key_allowed(key, base), "type": _type_name(base),
                             "base": _jsonable(base), "override": _jsonable(overrides.get(key)) if key in overrides else None,
                             "has_override": key in overrides, "running": _jsonable(running), "pending": key in pending}
            row["values_meta"] = meta
    data["overrides"] = {k: _jsonable(v) for k, v in overrides.items()}
    data["override_errors"] = list(r001.LIVE_OVERRIDES_ERRORS) + [f"{k}: {v}" for k, v in diff["rejected_in_file"].items()]
    data["overrides_file"] = str(paths.overrides)
    data["applied_state_known"] = snapshot is not None
    data["pending_keys"] = sorted(pending)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _emit(obj, code: int = 0) -> int:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable, allow_nan=False))
    return code


def main(argv: list[str] | None = None, paths: ControlPaths | None = None, runner: Runner = _run) -> int:
    paths = paths or default_paths()
    ap = argparse.ArgumentParser(description="r003 remote control (JSON out)")
    ap.add_argument("--by", default=os.environ.get("R016_BY") or "cli", help="who made the change (history)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for verb in ("start", "restart"):
        p = sub.add_parser(verb)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
        g.add_argument("--live", dest="dry_run", action="store_false")
        p.add_argument("--profile", choices=["conservative", "neutral", "aggressive"], default=None)
    sub.add_parser("stop")
    p = sub.add_parser("schema")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("get")
    p.add_argument("keys", nargs="*")
    p = sub.add_parser("set")
    p.add_argument("assignments", nargs="+", help="KEY=JSON")
    p = sub.add_parser("unset")
    p.add_argument("keys", nargs="+")
    sub.add_parser("reset")
    p = sub.add_parser("history")
    p.add_argument("-n", type=int, default=50)
    sub.add_parser("conditions")
    p = sub.add_parser("watchlist")
    p.add_argument("action", choices=["get", "set"])
    p.add_argument("--codes", default=None, help='"005930,삼성전자;000660,SK하이닉스"')
    p.add_argument("--stdin", action="store_true", help="read entries (one per line) from stdin")
    p = sub.add_parser("collector")
    p.add_argument("action", choices=["status", "restart"])
    # 차트 명령(indicators/bars)은 r017(pandas, r012/r002)을 불러와야 해서 기동이 ~1 s 느리다. 나머지 명령(status 폴링 등)은
    # 그 비용을 내지 않도록 자리만 등록하고, 이 두 명령일 때만 r017을 import해 r017의 인자 정의로 다시 파싱한다.
    for chart_cmd in ("indicators", "bars"):
        sub.add_parser(chart_cmd, add_help=False)
    args, extra = ap.parse_known_args(argv)

    if args.cmd in ("indicators", "bars"):
        import r017_chart_data as r017  # noqa: E402
        cp = argparse.ArgumentParser(prog=f"{ap.prog}")
        r017.add_subparsers(cp.add_subparsers(dest="cmd", required=True))
        out = r017.run_command(cp.parse_args([args.cmd, *extra]))
        return _emit(out, 0 if out.get("ok") else 2)
    if extra:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.cmd == "status":
        return _emit(status(paths, runner))

    if args.cmd in ("start", "stop", "restart"):
        try:
            out = service_action(paths, args.cmd, getattr(args, "dry_run", None), getattr(args, "profile", None), runner, args.by)
        except ValueError as exc:
            return _emit({"ok": False, "error": str(exc)}, 2)
        return _emit(out, 0 if out.get("ok") else 3)

    if args.cmd == "collector":
        if args.action == "status":
            return _emit(collector_status(paths, [c["code"] for c in read_watchlist(paths)["codes"]]))
        out = collector_restart(paths, args.by)
        return _emit(out, 0 if out.get("ok") else 3)

    snapshot = read_running_snapshot(paths)
    started = service_state(runner).get("started_epoch")
    diff = pending_diff(paths, snapshot, started)
    pending_keys = set(diff["pending_keys"])

    if args.cmd == "schema":
        return _emit({"generated_at": datetime.now().isoformat(timespec="seconds"), "overrides_file": str(paths.overrides),
                      "applied_state_known": snapshot is not None, "keys": schema_rows(paths, diff, snapshot, args.all),
                      "errors": list(r001.LIVE_OVERRIDES_ERRORS) + [f"{k}: {v}" for k, v in diff["rejected_in_file"].items()]})

    if args.cmd == "get":
        overrides = read_overrides(paths)
        applied = (snapshot or {}).get("overrides_applied") or {}
        keys = args.keys or sorted(overrides)
        rows, missing = {}, []
        for key in keys:
            if not hasattr(r001, key):
                missing.append(key)
                continue
            base = base_value(key)
            running = applied[key] if key in applied else (base if snapshot is not None else None)
            rows[key] = {"base": _jsonable(base), "override": _jsonable(overrides.get(key)) if key in overrides else None,
                         "has_override": key in overrides, "effective": _jsonable(getattr(r001, key)),
                         "running": _jsonable(running), "pending": key in pending_keys}
        return _emit({"values": rows, "unknown": missing, "applied_state_known": snapshot is not None}, 2 if missing else 0)

    if args.cmd == "set":
        try:
            assignments = dict(parse_assignment(a) for a in args.assignments)
        except ValueError as exc:
            return _emit({"ok": False, "error": str(exc)}, 2)
        accepted, rejected = validate_assignments(assignments)
        if accepted:
            with write_lock(paths):
                overrides = read_overrides(paths)
                before = {k: overrides.get(k) for k in accepted}
                overrides.update(accepted)
                write_overrides(paths, overrides, args.by)
                append_history(paths, "set", {k: {"from": _jsonable(before[k]), "to": _jsonable(v)} for k, v in accepted.items()}, args.by)
        return _emit({"ok": not rejected, "accepted": {k: _jsonable(v) for k, v in accepted.items()}, "rejected": rejected,
                      "restart_required": bool(accepted)}, 2 if rejected else 0)

    if args.cmd in ("unset", "reset"):
        with write_lock(paths):
            overrides = read_overrides(paths)
            keys = list(overrides) if args.cmd == "reset" else args.keys
            removed = {k: _jsonable(overrides.pop(k)) for k in keys if k in overrides}
            missing = [k for k in keys if k not in removed]
            if removed:
                write_overrides(paths, overrides, args.by)
                append_history(paths, args.cmd, removed, args.by)
        return _emit({"ok": True, "removed": removed, "not_set": missing, "restart_required": bool(removed)})

    if args.cmd == "history":
        rows = []
        if paths.history.is_file():
            lines = paths.history.read_text(encoding="utf-8").splitlines()
            for line in lines[-max(1, args.n):]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return _emit({"rows": rows})

    if args.cmd == "conditions":
        return _emit(conditions(paths, diff, snapshot))

    if args.cmd == "watchlist":
        if args.action == "get":
            return _emit(read_watchlist(paths))
        text = sys.stdin.read() if args.stdin else (args.codes or "")
        try:
            entries = parse_watchlist_entries(text)
        except ValueError as exc:
            return _emit({"ok": False, "error": str(exc)}, 2)
        if not entries:
            return _emit({"ok": False, "error": "no codes given"}, 2)
        return _emit(write_watchlist(paths, entries, args.by))

    return 2


if __name__ == "__main__":
    sys.exit(main())
