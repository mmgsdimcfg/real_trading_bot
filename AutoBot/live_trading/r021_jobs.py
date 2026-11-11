# -*- coding: utf-8 -*-
"""R021 - 데이터 작업 실행기 (장 마감 후 수집 / 일봉 이력 / 감시 종목 선별 / Edge AI 재학습 / 유니버스 갱신).

AutoBotH 가 r016_control.py job ... 으로 부른다 (Linux 서버, Windows 노트북 공통). 작업은 앱과 독립된 백그라운드 프로세스
(r019.spawn_detached)로 돌고, 앱을 꺼도 계속된다. 한 번에 한 작업만 (<runtime>/data/jobs/.r021.lock) - 모두 KIS 호출 제한 /
CPU 를 많이 쓰는 일이다.

  r016 job list                                 작업 목록 + 각 작업의 마지막 실행 상태 + 지금 도는 작업
  r016 job start NAME [--date YYYYMMDD] [--workers N] [--edge-ai] [--no-collect] [--no-scan] [--no-apply]
  r016 job stop NAME                            지금 단계(자식 프로세스)를 끝내고 stopped 로 기록
  r016 job log NAME [-n 200]                    마지막 실행 로그 끝부분
  r016 job data --date YYYYMMDD                 그 날짜 폴더(<runtime>/data/D)의 수집 데이터 유무(10초봉/일봉 파일 수, 수집 목록)와
                                                g002 선별 결과(_D_picks.txt) - 앱의 "감시 종목 선별" 카드가 기준일마다 읽는다

작업 (단계는 순서대로, 각 단계 = data_simulation 스크립트 한 번 실행):
  collect    g001 --date D                                   분봉/일봉 수집 (KIS, 오래 걸림)
  history    g025 --append D                                 종목별 일봉 이력 갱신 (AutoBotH 일/주/월봉)
  scan       g002 --date D --no-export-watchlist             그 날짜의 선별 파일(_D_picks.txt, 리포트)만 만든다 (r004 는 안 건드림)
             -> apply                                         r016 watchlist apply --date D: 선별 파일을 r004 에 반영(백업/이력)
  edge_ai    g013 --live, g014, g024 --edge-ai                Edge AI 데이터셋/학습/점검 (g015_edge_ai_daily.sh 와 같음)
  universe   g010                                             수집 대상 종목 목록(g004) 갱신
  postclose  collect -> history -> scan -> apply [-> edge_ai] 장 마감 후 한 번에. collect 가 실패하면 뒤 단계는 하지 않는다
             (오래된 데이터로 감시목록을 만들지 않게). history 실패는 건너뛰고 계속.

  [2026-11-11] 단계 선택 (postclose / scan): --no-collect (수집+이력 생략), --no-scan (선별 생략), --no-apply (r004 반영 생략).
    1+2+3 = 기본, 2+3 = --no-collect (그 날짜 폴더에 수집 데이터가 있어야 함), 3 = --no-collect --no-scan (선별 파일이 있어야 함).
    start 는 생략된 앞 단계의 산출물이 없으면 거부한다. --keep-watchlist 는 --no-apply 의 옛 이름.
    apply 는 선별 파일이 없거나 비어 있거나, 이 실행의 scan 보다 오래된 파일(g002 가 후보 0 이면 파일을 안 만들고 옛 파일이
    남는다)이면 건너뛴다(단계 state=skipped, note) - 옛 선별이 r004 에 들어가지 않게. 작업 자체는 done.

파일: <runtime>/data/jobs/<NAME>.json (상태), <NAME>.stop (정지 요청), <runtime>/log/jobs/<YYYYMMDD>/<NAME>_<HHMMSS>.log

Update log:
- [2026-11-11] type=feat owner=claude
    summary: 파이프라인 단계 선택 - postclose = collect -> history -> scan -> apply. scan 단계는 항상 g002 --no-export-watchlist
      (선별 파일만), r004 반영은 새 apply 단계(r016 watchlist apply --date D: 백업/이력/다음 거래일 헤더)가 한다. --no-collect /
      --no-scan / --no-apply 로 1+2+3, 2+3, 3 조합을 고르고, start 는 앞 단계 산출물(10초봉 파일 / 선별 파일)이 없으면 거부.
      --keep-watchlist 는 --no-apply 의 별칭으로 유지. 상태 파일 args 에 collect/scan/apply 기록 (keep_watchlist 대신).
    impact: tooling (r003 영향 없음. g002 의 r004 내보내기 대신 r016 경로로 쓰므로 r004 헤더에 scanner_config 줄이 빠지고
      for_trading_day 가 r014 다음 거래일로 바뀜 - r003 은 헤더를 읽지 않음)
    compatibility: backward-compatible (앱 스케줄러의 `job start postclose` 기본 = 네 단계 전부)
- [2026-11-11] type=feat owner=claude
    summary: job data --date D (date_data_info) - 그 날짜 폴더의 g001 산출물(*_10s.txt / *_daily.csv 수, _D_collected.txt 목록 수,
      마지막 파일 시각)과 g002 선별 결과(_D_picks.txt 의 code,name 목록, 시각)를 돌려준다. 앱이 "수집된 날짜면 g002 선별만 실행"
      과 "선별 후 r004 적용" 흐름에 쓴다(r016 가 r004 비교 블록을 덧붙임 - watchlist apply).
    impact: tooling (읽기 전용 명령 추가)
    compatibility: backward-compatible
- [2026-11-10] type=feat owner=claude
    summary: 신규 - AutoBotH(노트북 메인)에서 장 마감 후 데이터 취득과 감시 종목 선별을 실행/정지/진행 확인.
    impact: tooling (스크립트 자체는 그대로 실행)
    compatibility: backward-compatible (신규 파일)
- [2026-11-11] type=feat owner=claude
    summary: --keep-watchlist (start/run) - scan 단계를 g002 --no-export-watchlist 로 실행해 r004 감시목록을 바꾸지 않는다.
      과거 날짜 데이터를 보충 수집하거나 Edge AI 재학습용으로 postclose 를 돌릴 때 지금 감시 중인 종목이 바뀌는 문제.
      상태 파일 args 에 keep_watchlist 기록 (앱이 "감시목록은 바꾸지 않음" 표시).
    impact: tooling (r016 job start 옵션 추가, 기본 동작 불변)
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "data_simulation"
sys.path.insert(0, str(HERE))

import r019_platform as r019  # noqa: E402

POLL_SEC = 1.0
STOP_GRACE_SEC = 15
START_CONFIRM_SEC = 10.0


def runtime_root() -> Path:
    raw = os.environ.get("AUTO_TRADING_RUNTIME")
    return Path(raw) if raw else HERE.parents[2] / "runtime"


@dataclass
class JobArgs:
    date: str
    workers: int | None = None
    edge_ai: bool = False
    # [2026-11-11] pipeline steps (postclose / scan): 1 collect (+history), 2 scan, 3 apply (r004 <- picks)
    collect: bool = True
    scan: bool = True
    apply: bool = True
    by: str = "cli"  # who started it - recorded in the r004 header / history by the apply step

    @property
    def keep_watchlist(self) -> bool:  # old name (2026-11-11 morning): r004 untouched
        return not self.apply


@dataclass
class Step:
    name: str
    build: Callable[[JobArgs, dict], list[str]]  # (args, rcs of earlier steps) -> argv after python
    required: bool = True  # False: a failure is recorded and the job goes on


def _py(script: str, *a: str) -> list[str]:
    return [str(SIM / script), *a]


def _workers(a: JobArgs, default: int) -> str:
    return str(a.workers if a.workers and a.workers > 0 else default)


STEPS: dict[str, Step] = {
    "collect": Step("collect", lambda a, rc: _py("g001_data_collect_symbols_daily.py", "--date", a.date,
                                                 "--workers", _workers(a, 4))),
    "history": Step("history", lambda a, rc: _py("g025_daily_history.py", "--append", a.date), required=False),
    "scan": Step("scan", lambda a, rc: _py("g002_data_scan_trade_candidates.py", "--date", a.date, "--no-export-watchlist")),
    # r004 <- <runtime>/data/D/_D_picks.txt through r016 (write lock, backup, history, for_trading_day header, collector note)
    # --restart-collector: a running g011 that misses new codes is restarted (it subscribes r004 only at its start)
    "apply": Step("apply", lambda a, rc: [str(HERE / "r016_control.py"), "--by", a.by, "watchlist", "apply", "--date", a.date,
                                          "--restart-collector"]),
    "edge_dataset": Step("edge_dataset", lambda a, rc: _py("g013_edge_ai_dataset.py", "--workers", _workers(a, 2), "--live"),
                         required=False),
    "edge_train": Step("edge_train", lambda a, rc: _py("g014_edge_ai_train.py", "--threads", _workers(a, 2)), required=False),
    "edge_check": Step("edge_check", lambda a, rc: _py("g024_ops_check.py", "--edge-ai", "--rc-dataset",
                                                       str(rc.get("edge_dataset", 0)), "--rc-train",
                                                       str(rc.get("edge_train", 0))), required=False),
    "universe": Step("universe", lambda a, rc: _py("g010_universe_update.py")),
}

JOBS: dict[str, dict] = {
    "collect": {"title": "데이터 수집 (g001)", "steps": ["collect"]},
    "history": {"title": "일봉 이력 (g025)", "steps": ["history"]},
    "scan": {"title": "감시 종목 선별 (g002 -> r004)", "steps": ["scan", "apply"]},
    "edge_ai": {"title": "Edge AI 재학습 (g013/g014/g024)", "steps": ["edge_dataset", "edge_train", "edge_check"]},
    "universe": {"title": "유니버스 갱신 (g010)", "steps": ["universe"]},
    "postclose": {"title": "장 마감 파이프라인 (수집 -> 이력 -> 선별 -> r004)", "steps": ["collect", "history", "scan", "apply"]},
}

STEP_SWITCH = {"collect": "collect", "history": "collect", "scan": "scan", "apply": "apply"}  # step -> JobArgs flag


def job_steps(name: str, args: JobArgs) -> list[Step]:
    names = [n for n in JOBS[name]["steps"] if getattr(args, STEP_SWITCH.get(n, ""), True)]
    if name == "postclose" and args.edge_ai:
        names += JOBS["edge_ai"]["steps"]
    return [STEPS[n] for n in names]


def check_steps(paths: "Paths", name: str, args: JobArgs) -> str | None:
    """[2026-11-11] a skipped earlier step must have left its output: scan without collect needs the date's 10-second files,
    apply without scan needs the date's picks file. Returns the refusal text (None = fine)."""
    names = [s.name for s in job_steps(name, args)]
    if not names:
        return "no steps selected"
    info = date_data_info(paths, args.date) if ("scan" in names or "apply" in names) and "collect" not in names else None
    if info is not None and "scan" in names and not info["scannable"]:
        return f"no collected data for {args.date} ({info['dir']}) - run the collect step (g001) first"
    if info is not None and "apply" in names and "scan" not in names and not info["picks"]:
        return f"no picks file for {args.date} ({info['picks_file']}) - run the scan step (g002) first"
    return None


class Paths:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else runtime_root()
        self.dir = self.root / "data" / "jobs"

    @property
    def lock(self) -> Path:
        return self.dir / ".r021.lock"

    def state(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def stop(self, name: str) -> Path:
        return self.dir / f"{name}.stop"

    def log(self, name: str, now: datetime) -> Path:
        return self.root / "log" / "jobs" / now.strftime("%Y%m%d") / f"{name}_{now:%H%M%S}.log"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        for i in range(40):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # Windows: a reader has the file open
                if i == 39:
                    raise
                time.sleep(0.05)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_state(paths: Paths, name: str) -> dict | None:
    try:
        d = json.loads(paths.state(name).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def current(paths: Paths) -> dict | None:
    """{pid, started_at, name, run_id} of the job holding the runner lock (None = no job running)."""
    owner = r019.running_owner(paths.lock)
    if owner is None:
        return None
    name = run_id = None
    with contextlib.suppress(Exception):
        parts = paths.lock.read_text(encoding="utf-8").splitlines()[1].split()
        name = parts[0] if parts else None
        run_id = parts[1] if len(parts) > 1 else None
    return {**owner, "name": name, "run_id": run_id}


def _effective(paths: Paths, name: str, st: dict | None, cur: dict | None) -> dict | None:
    """A state file that says running while no runner holds the lock = the runner died (crash / reboot)."""
    if st and st.get("state") == "running" and not (cur and cur.get("name") == name):
        st = dict(st, state="failed", error=st.get("error") or "runner ended without finishing (crash / reboot)")
    return st


def list_jobs(paths: Paths) -> dict:
    cur = current(paths)
    rows = []
    for name, spec in JOBS.items():
        rows.append({"name": name, "title": spec["title"], "steps": spec["steps"],
                     "last": _effective(paths, name, read_state(paths, name), cur)})
    return {"ok": True, "running": cur, "jobs": rows, "generated_at": _now()}


def start_job(paths: Paths, name: str, args: JobArgs, by: str, spawn=r019.spawn_detached) -> dict:
    if name not in JOBS:
        return {"ok": False, "error": f"unknown job {name!r} (one of {', '.join(JOBS)})"}
    if not (len(args.date) == 8 and args.date.isdigit()):
        return {"ok": False, "error": f"bad date {args.date!r} (YYYYMMDD)"}
    cur = current(paths)
    if cur:
        return {"ok": False, "error": f"another job is running: {cur.get('name') or '?'} (pid {cur.get('pid')})", "running": cur}
    problem = check_steps(paths, name, args)
    if problem:
        return {"ok": False, "error": problem}
    paths.dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        paths.stop(name).unlink()
    import uuid

    run_id = uuid.uuid4().hex[:12]
    argv = [r019.python_exe(), str(Path(__file__).resolve()), "run", name, "--date", args.date, "--by", by,
            "--run-id", run_id, "--runtime", str(paths.root)]
    if args.workers:
        argv += ["--workers", str(args.workers)]
    if args.edge_ai:
        argv.append("--edge-ai")
    argv += [f"--no-{k}" for k in ("collect", "scan", "apply") if not getattr(args, k)]
    now = datetime.now()
    pid = spawn(argv, cwd=str(HERE), log_path=paths.root / "log" / "jobs" / "r021_runner.log",
                env=r019.child_env({"AUTO_TRADING_RUNTIME": str(paths.root)}))
    # success = THIS run (run_id) holds the runner lock, or already wrote its state (a very short job). Another start that won
    # the race, or a runner that died at start, is reported as a failure.
    deadline = time.time() + START_CONFIRM_SEC
    while time.time() < deadline:
        cur = current(paths)
        if cur and cur.get("run_id") == run_id:
            return {"ok": True, "name": name, "pid": pid, "run_id": run_id, "started_at": now.isoformat(timespec="seconds"),
                    "running": cur}
        if (read_state(paths, name) or {}).get("run_id") == run_id:
            return {"ok": True, "name": name, "pid": pid, "run_id": run_id, "started_at": now.isoformat(timespec="seconds"),
                    "running": cur, "finished": True}
        time.sleep(0.1)
    return {"ok": False, "error": f"job runner did not start (see {paths.root / 'log' / 'jobs' / 'r021_runner.log'})",
            "running": current(paths), "run_id": run_id}


def stop_job(paths: Paths, name: str) -> dict:
    cur = current(paths)
    if not cur or cur.get("name") != name:
        return {"ok": False, "error": f"job {name!r} is not running", "running": cur}
    paths.dir.mkdir(parents=True, exist_ok=True)
    paths.stop(name).touch()
    deadline = time.time() + STOP_GRACE_SEC + 10
    while time.time() < deadline and current(paths):
        time.sleep(0.2)
    still = current(paths)
    return {"ok": not still, "stopped": not still, "running": still,
            **({} if not still else {"error": "runner did not end in time (it will stop after the current step ends)"})}


def _file_iso(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def read_picks(paths: Paths, date: str) -> tuple[Path, list[dict] | None]:
    """g002 picks of a date (<runtime>/data/D/_D_picks.txt, "code,name" per line) -> (path, [{code, name}] or None)."""
    p = paths.root / "data" / date / f"_{date}_picks.txt"
    if not p.is_file():
        return p, None
    rows: list[dict] = []
    seen: set[str] = set()
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.split(",")]
        code = parts[0].zfill(6)
        if not code.isdigit() or len(code) != 6 or code in seen:
            continue
        seen.add(code)
        rows.append({"code": code, "name": parts[1] if len(parts) > 1 and parts[1] else code})
    return p, rows


def date_data_info(paths: Paths, date: str) -> dict:
    """[2026-11-11] What <runtime>/data/D holds: g001 output (10-second / daily files, the collected list) and the g002 picks.

    scannable = at least one *_10s.txt (g002 reads the 10-second files of the date; daily files add the trend filters)."""
    d = paths.root / "data" / date
    out: dict = {"ok": True, "date": date, "dir": str(d), "exists": d.is_dir(), "scannable": False, "collected": False,
                 "collected_count": 0, "files_10s": 0, "files_daily": 0, "data_mtime": None,
                 "picks": None, "picks_file": str(d / f"_{date}_picks.txt"), "picks_mtime": None, "scan_log_mtime": None,
                 "generated_at": _now()}
    if not d.is_dir():
        return out
    latest = 0.0
    try:
        for p in d.iterdir():
            name = p.name
            if name.startswith("_"):
                continue
            if name.endswith("_10s.txt"):
                out["files_10s"] += 1
            elif name.endswith("_daily.csv"):
                out["files_daily"] += 1
            else:
                continue
            with contextlib.suppress(OSError):
                latest = max(latest, p.stat().st_mtime)
    except OSError as exc:
        out["error"] = f"cannot list {d} ({type(exc).__name__})"
    if latest:
        out["data_mtime"] = datetime.fromtimestamp(latest).isoformat(timespec="seconds")
    out["scannable"] = out["files_10s"] > 0
    marker = d / f"_{date}_collected.txt"
    if marker.is_file():
        out["collected"] = True
        with contextlib.suppress(OSError):
            out["collected_count"] = sum(1 for ln in marker.read_text(encoding="utf-8-sig").splitlines()
                                         if ln.strip() and not ln.strip().startswith("#"))
    pf, picks = read_picks(paths, date)
    if picks is not None:
        out["picks"] = picks
        out["picks_mtime"] = _file_iso(pf)
    log = d / f"_{date}_scan.log"
    if log.is_file():
        out["scan_log_mtime"] = _file_iso(log)
    return out


def tail_log(paths: Paths, name: str, n: int = 200) -> dict:
    st = read_state(paths, name)
    if not st or not st.get("log"):
        return {"ok": True, "name": name, "log": None, "lines": []}
    p = Path(st["log"])
    try:
        with open(p, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256 * 1024))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": f"cannot read log ({type(exc).__name__})", "log": str(p)}
    lines = text.splitlines()[-max(1, n):]
    return {"ok": True, "name": name, "log": str(p), "size": size, "lines": lines, "state": st.get("state")}


# ---------------------------------------------------------------------------
# runner (the detached process)
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, paths: Paths, name: str, args: JobArgs, by: str, run_id: str | None = None):
        self.paths, self.name, self.args, self.by = paths, name, args, by
        self.args.by = by
        now = datetime.now()
        self.log_path = paths.log(name, now)
        self.state = {"name": name, "title": JOBS[name]["title"], "state": "running", "pid": os.getpid(), "by": by,
                      "run_id": run_id,
                      "started_at": now.isoformat(timespec="seconds"), "finished_at": None, "rc": None,
                      "args": {"date": args.date, "workers": args.workers, "edge_ai": args.edge_ai,
                               "collect": args.collect, "scan": args.scan, "apply": args.apply},
                      "log": str(self.log_path), "steps": [], "error": None, "note": None}

    def _save(self) -> None:
        _write_json(self.paths.state(self.name), self.state)

    def _stop_requested(self) -> bool:
        return self.paths.stop(self.name).exists()

    def _apply_skip_reason(self) -> str | None:
        """[2026-11-11] the apply step only publishes picks this run can vouch for: a picks file must exist, be non-empty and - when
        the scan ran in this job - be written by that scan (g002 leaves no file when it finds no candidate, so an older file would
        otherwise be applied)."""
        pf, picks = read_picks(self.paths, self.args.date)
        if not picks:
            return f"no picks for {self.args.date} (the scan found no candidate) - r004 unchanged"
        scans = [r for r in self.state["steps"] if r["name"] == "scan" and r.get("started_at")]
        if scans:
            started = datetime.fromisoformat(scans[-1]["started_at"]).timestamp()
            try:
                written = pf.stat().st_mtime
            except OSError:
                written = 0.0
            if written < started - 1.0:
                return f"picks file predates this run's scan (the scan found no candidate) - r004 unchanged"
        return None

    def _skip_step(self, step: Step, reason: str, fh) -> None:
        now = _now()
        self.state["steps"].append({"name": step.name, "argv": [], "state": "skipped", "rc": None, "started_at": now,
                                    "finished_at": now, "note": reason})
        self.state["note"] = f"{step.name} skipped: {reason}"
        self._save()
        self._log(fh, f"step {step.name} skipped: {reason}")

    def _log(self, fh, text: str) -> None:
        fh.write(f"[{_now()}] [r021] {text}\n".encode("utf-8"))
        fh.flush()

    def _run_step(self, step: Step, rcs: dict, fh) -> tuple[int | None, bool]:
        argv = [r019.python_exe(), *step.build(self.args, rcs)]
        row = {"name": step.name, "argv": argv[1:], "state": "running", "rc": None, "started_at": _now(), "finished_at": None}
        self.state["steps"].append(row)
        self._save()
        self._log(fh, f"step {step.name}: {' '.join(argv[1:])}")
        kw: dict = {"cwd": str(SIM), "stdin": subprocess.DEVNULL, "stdout": fh, "stderr": subprocess.STDOUT,
                    "env": r019.child_env({"AUTO_TRADING_RUNTIME": str(self.paths.root)})}
        if r019.IS_WINDOWS:
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = True  # own process group: a stop ends its workers too (killpg)
        proc = subprocess.Popen(argv, **kw)
        stopped = False
        while proc.poll() is None:
            if self._stop_requested():
                stopped = True
                self._log(fh, f"stop requested - ending {step.name} and its child processes (pid {proc.pid})")
                if r019.IS_WINDOWS:
                    # taskkill /T /F: the venv launcher, the real interpreter and its multiprocessing workers
                    r019.kill_tree(proc.pid)
                else:
                    import signal

                    pgid = proc.pid  # start_new_session=True: the step leads its own group (kept even after it is reaped)
                    with contextlib.suppress(OSError):
                        os.killpg(pgid, signal.SIGTERM)  # g001/g013 workers get a chance to finish a file
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=STOP_GRACE_SEC)
                    with contextlib.suppress(OSError):
                        os.killpg(pgid, signal.SIGKILL)  # workers that ignored SIGTERM / outlived the parent
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=STOP_GRACE_SEC)
                break
            time.sleep(POLL_SEC)
        rc = proc.wait()
        row.update(state="stopped" if stopped else ("done" if rc == 0 else "failed"), rc=rc, finished_at=_now())
        self._save()
        self._log(fh, f"step {step.name} rc={rc}{' (stopped)' if stopped else ''}")
        return rc, stopped

    def run(self) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._save()
        rcs: dict = {}
        final_rc = 0
        with open(self.log_path, "ab") as fh:
            self._log(fh, f"job {self.name} start date={self.args.date} by={self.by}")
            for step in job_steps(self.name, self.args):
                if self._stop_requested():
                    self.state.update(state="stopped")
                    break
                if step.name == "apply":
                    reason = self._apply_skip_reason()
                    if reason:
                        self._skip_step(step, reason, fh)
                        rcs[step.name] = 0
                        continue
                rc, stopped = self._run_step(step, rcs, fh)
                rcs[step.name] = rc
                if stopped:
                    self.state.update(state="stopped")
                    break
                if rc != 0:
                    final_rc = rc or 1
                    if step.required:
                        self.state.update(state="failed", error=f"{step.name} failed (rc={rc}) - later steps skipped")
                        break
                    self.state["error"] = f"{step.name} failed (rc={rc}) - continued"
            if self.state["state"] == "running":
                self.state["state"] = "done_with_errors" if self.state.get("error") else "done"
            self.state.update(rc=final_rc, finished_at=_now())
            self._save()
            self._log(fh, f"job {self.name} {self.state['state']} rc={final_rc}")
        with contextlib.suppress(OSError):
            self.paths.stop(self.name).unlink()
        return 0 if self.state["state"] in ("done", "stopped") else 1


def run_detached_entry(name: str, args: JobArgs, by: str, paths: Paths | None = None, run_id: str | None = None) -> int:
    paths = paths or Paths()
    fh = r019.acquire_instance_lock(paths.lock)
    if fh is None:
        print(f"[r021] another job holds {paths.lock}", flush=True)
        return 3
    try:
        fh.write(f"{name} {run_id or '-'}\n")  # second line = job name + run id (current())
        fh.flush()
        return Runner(paths, name, args, by, run_id).run()
    finally:
        fh.close()


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def add_job_parser(sub) -> None:
    """r016 `job` sub-commands (shared with this file's own CLI)."""
    p = sub.add_parser("job", help="data jobs: list | start NAME | stop NAME | log NAME | data --date D")
    p.add_argument("action", choices=["list", "start", "stop", "log", "data"])
    p.add_argument("name", nargs="?", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--edge-ai", action="store_true")
    _add_step_switches(p)
    p.add_argument("--trading-day-only", action="store_true", help="start: skip (rc 0, skipped) on a weekend / r014 closed day")
    p.add_argument("-n", type=int, default=200)


def _add_step_switches(p) -> None:
    p.add_argument("--no-collect", action="store_true", help="postclose: skip collect (g001) + history (g025) - the date must be collected")
    p.add_argument("--no-scan", action="store_true", help="postclose/scan: skip the scan (g002) - the date must have a picks file")
    p.add_argument("--no-apply", action="store_true", help="postclose/scan: do not write r004 (picks files only)")
    p.add_argument("--keep-watchlist", action="store_true", help="old name of --no-apply")


def _job_args(args, date: str) -> JobArgs:
    return JobArgs(date=date, workers=args.workers, edge_ai=args.edge_ai,
                   collect=not getattr(args, "no_collect", False), scan=not getattr(args, "no_scan", False),
                   apply=not (getattr(args, "no_apply", False) or getattr(args, "keep_watchlist", False)))


def is_trading_day(ymd: str) -> bool:
    d = datetime.strptime(ymd, "%Y%m%d")
    if d.weekday() >= 5:
        return False
    try:
        from r014_market_calendar import session_hours  # noqa: E402
        return session_hours(d) is not None
    except Exception:
        return True


def run_job_command(args, by: str, paths: Paths | None = None) -> tuple[dict, int]:
    paths = paths or Paths()
    if args.action == "list":
        return list_jobs(paths), 0
    if args.action == "data":  # [2026-11-11] the date folder's data / picks (no job name)
        date = args.date or args.name or _today()
        if not (len(date) == 8 and date.isdigit()):
            return {"ok": False, "error": f"bad date {date!r} (YYYYMMDD)"}, 2
        return date_data_info(paths, date), 0
    if not args.name:
        return {"ok": False, "error": "job name required"}, 2
    if args.name not in JOBS:
        return {"ok": False, "error": f"unknown job {args.name!r} (one of {', '.join(JOBS)})"}, 2
    if args.action == "start":
        date = args.date or _today()
        if getattr(args, "trading_day_only", False) and len(date) == 8 and date.isdigit() and not is_trading_day(date):
            return {"ok": True, "skipped": True, "reason": f"{date} is not a trading day"}, 0
        out = start_job(paths, args.name, _job_args(args, date), by)
        return out, 0 if out.get("ok") else 3
    if args.action == "stop":
        out = stop_job(paths, args.name)
        return out, 0 if out.get("ok") else 3
    return tail_log(paths, args.name, args.n), 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="r021 data job runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("name", choices=list(JOBS))
    p.add_argument("--date", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--edge-ai", action="store_true")
    _add_step_switches(p)
    p.add_argument("--by", default="cli")
    p.add_argument("--run-id", default=None)
    p.add_argument("--runtime", default=None, help="runtime root (a WMI-started process has no inherited env)")
    args = ap.parse_args(argv)
    if args.runtime:
        os.environ["AUTO_TRADING_RUNTIME"] = args.runtime
    return run_detached_entry(args.name, _job_args(args, args.date or _today()), args.by, run_id=args.run_id)


if __name__ == "__main__":
    sys.exit(main())
