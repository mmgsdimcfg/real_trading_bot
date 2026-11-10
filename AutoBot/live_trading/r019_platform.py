# -*- coding: utf-8 -*-
"""R019 - Linux(라즈베리파이)와 Windows(노트북)에서 같은 코드가 돌도록 플랫폼 차이를 한곳에 모은 모듈.

- 파일 잠금: POSIX fcntl.flock / Windows msvcrt.locking. 잠금은 열린 파일 핸들 단위이고, 핸들을 닫거나 프로세스가 끝나면 풀린다.
  Windows는 바이트 범위 잠금이라 파일 내용(pid 등)을 다른 프로세스가 읽을 수 있도록 내용과 겹치지 않는 먼 오프셋
  (_WIN_LOCK_OFFSET) 1바이트를 잠근다. 공유 잠금(LOCK_SH)은 Windows에 없으므로 "잠겨 있는지 확인"만 is_locked()로 제공한다.
- "실행 중인가": /proc 을 못 쓰는 Windows에서는 각 프로그램이 들고 있는 단일 실행 잠금 파일(r003 .r003.lock,
  g011 .g011.lock, r020 .r020.lock)이 잠겨 있는지로 판단한다. 잠금 파일 첫 줄은 "<pid> <YYYY-MM-DD HH:MM:SS>".
- 분리 실행(spawn_detached): 부모(AutoBotH, r016)가 끝나도 계속 도는 백그라운드 프로세스. Windows는 콘솔 창 없이
  (CREATE_NO_WINDOW - 자식이 또 띄우는 python 도 창이 뜨지 않는다) 새 프로세스 그룹으로, 가능하면 부모의 Job 밖으로
  (CREATE_BREAKAWAY_FROM_JOB) 띄운다. POSIX는 새 세션(setsid).
- pid_alive / terminate: Windows는 OpenProcess + GetExitCodeProcess / TerminateProcess(os.kill).

이 모듈은 표준 라이브러리만 쓴다(r003, r016, g001, g011, g013, r018 이 import).

Update log:
- [2026-11-10] type=feat owner=claude
    summary: 신규 - 윈도우 노트북에서 r003/g0xx 실행(AutoBotH 로컬 모드). fcntl 직접 사용을 이 모듈로 대체.
    impact: live/scanner/tooling
    compatibility: backward-compatible (Linux 동작 동일: flock)
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Iterator

IS_WINDOWS = os.name == "nt"

# Windows: 잠그는 바이트 위치(내용과 겹치지 않게 1 GiB 지점). 파일 끝 너머 잠금은 허용된다.
_WIN_LOCK_OFFSET = 1 << 30


# ---------------------------------------------------------------------------
# file locks
# ---------------------------------------------------------------------------
def _fd(fh) -> int:
    return fh if isinstance(fh, int) else fh.fileno()


def _win_lock_region(fd: int, mode: int) -> None:
    import msvcrt

    pos = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
    try:
        msvcrt.locking(fd, mode, 1)
    finally:
        os.lseek(fd, pos, os.SEEK_SET)


def try_lock(fh) -> bool:
    """Exclusive, non-blocking. True = this handle now holds the lock, False = another handle holds it."""
    with contextlib.suppress(Exception):
        if hasattr(fh, "flush"):
            fh.flush()
    fd = _fd(fh)
    if IS_WINDOWS:
        import msvcrt

        try:
            _win_lock_region(fd, msvcrt.LK_NBLCK)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def lock(fh, timeout: float | None = None, poll: float = 0.05) -> bool:
    """Exclusive, blocking (timeout None = wait forever). False only when the timeout passed."""
    if not IS_WINDOWS and timeout is None:
        import fcntl

        fcntl.flock(_fd(fh), fcntl.LOCK_EX)
        return True
    deadline = None if timeout is None else time.time() + timeout
    while True:
        if try_lock(fh):
            return True
        if deadline is not None and time.time() >= deadline:
            return False
        time.sleep(poll)


def unlock(fh) -> None:
    fd = _fd(fh)
    with contextlib.suppress(OSError):
        if IS_WINDOWS:
            import msvcrt

            _win_lock_region(fd, msvcrt.LK_UNLCK)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def locked_file(path: Path, timeout: float | None = None) -> Iterator[IO[str]]:
    """Open (create) `path` and hold the exclusive lock for the with-block. TimeoutError when it could not be taken."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        if not lock(fh, timeout):
            raise TimeoutError(f"lock busy: {path}")
        try:
            yield fh
        finally:
            unlock(fh)
    finally:
        fh.close()


def acquire_instance_lock(path: Path) -> IO[str] | None:
    """Single-instance lock: open + try_lock + write "<pid> <now>". Returns the open handle (keep it for the process lifetime;
    closing it releases the lock) or None when another process holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    if not try_lock(fh):
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    # ppid: on Windows a venv's python.exe is a launcher that runs the real interpreter as its child, so the pid a parent got
    # from Popen is this process's parent - r016 matches either to tell "our child" from an external process
    fh.write(f"{os.getpid()} {datetime.now():%Y-%m-%d %H:%M:%S} ppid={os.getppid()}\n")
    fh.flush()
    return fh


def is_locked(path: Path) -> bool:
    """True while some process holds the lock of `path` (missing file = not locked). Never creates the file."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        fh = open(path, "a+", encoding="utf-8")
    except OSError:
        return False
    try:
        if try_lock(fh):
            unlock(fh)
            return False
        return True
    finally:
        fh.close()


def lock_owner(path: Path) -> dict:
    """{pid, started_at 'YYYY-MM-DDTHH:MM:SS', ppid} from the first line of an instance lock file
    ("<pid> <date> <time> [ppid=<n>]"); missing parts are None."""
    out: dict = {"pid": None, "started_at": None, "ppid": None}
    try:
        first = Path(path).read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
    except Exception:
        return out
    parts = first.split()
    if parts and parts[0].isdigit():
        out["pid"] = int(parts[0])
    if len(parts) >= 3:
        with contextlib.suppress(ValueError):
            out["started_at"] = datetime.strptime(f"{parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S").isoformat(timespec="seconds")
    for p in parts[3:]:
        if p.startswith("ppid=") and p[5:].isdigit():
            out["ppid"] = int(p[5:])
    return out


def running_owner(path: Path) -> dict | None:
    """{pid, started_at, ppid} of the process holding the instance lock `path`, None when nobody holds it."""
    if not is_locked(path):
        return None
    return lock_owner(path)


# ---------------------------------------------------------------------------
# host role (two hosts, one KIS account)
# ---------------------------------------------------------------------------
# <runtime>/config/host_role.json {"role": "primary"|"standby", "by": ..., "at": ...}. 계좌(appkey)는 하나인데 r003 을 돌릴 수 있는
# 곳이 둘(라즈베리파이, 노트북)이라, 지금 매매하지 않는 쪽을 standby 로 표시한다. standby 호스트에서는 r003 이 시작을
# 거부하고(R003_IGNORE_ROLE=1 이면 무시), g022 cron 이 g011(KIS 웹소켓 - appkey 당 세션 1개)을 띄우지 않는다.
# 파일이 없거나 읽을 수 없으면 primary (기존 단일 서버 동작 그대로).
HOST_ROLES = ("primary", "standby")


def host_role_path(runtime_root: Path | str) -> Path:
    return Path(runtime_root) / "config" / "host_role.json"


def read_host_role(runtime_root: Path | str) -> dict:
    import json

    p = host_role_path(runtime_root)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"role": "primary", "set": False}
    role = data.get("role") if isinstance(data, dict) else None
    if role not in HOST_ROLES:
        return {"role": "primary", "set": False}
    return {"role": role, "set": True, "by": data.get("by"), "at": data.get("at"), "note": data.get("note")}


ROLE_LOCK_TIMEOUT = 10.0


@contextlib.contextmanager
def role_lock(runtime_root: Path | str, timeout: float = ROLE_LOCK_TIMEOUT) -> Iterator[None]:
    """Serializes "is r003 running here? -> write standby" (r016 role set) with "take the r003 instance lock -> read the role"
    (r003 start), so a host can never be switched to standby while an r003 that already read 'primary' is starting.
    TimeoutError when it cannot be taken."""
    with locked_file(Path(runtime_root) / "config" / ".host_role.lock", timeout=timeout):
        yield


def write_host_role(runtime_root: Path | str, role: str, by: str, note: str = "") -> dict:
    import json
    import tempfile

    if role not in HOST_ROLES:
        raise ValueError(f"role must be one of {HOST_ROLES}")
    p = host_role_path(runtime_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"role": role, "by": by, "at": datetime.now().isoformat(timespec="seconds"), "note": note}
    fd, tmp = tempfile.mkstemp(prefix=".host_role.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return read_host_role(runtime_root)


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------
_K32 = None


def _kernel32():
    """kernel32 with full prototypes (64-bit HANDLEs must not be truncated to c_int)."""
    global _K32
    if _K32 is None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        _K32 = k32
    return _K32


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if IS_WINDOWS:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = _kernel32()
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            from ctypes import wintypes

            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def terminate(pid: int) -> bool:
    """POSIX SIGTERM (graceful: r003/g011 handlers run) / Windows TerminateProcess (immediate). False when it failed."""
    import signal

    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except OSError:
        return False


def kill_tree(pid: int) -> None:
    """Force-end `pid` and every process it started (data jobs: g001 workers, multiprocessing pools). Windows: taskkill /T /F
    (also covers the venv launcher -> real interpreter pair). POSIX: SIGKILL to its process group when it leads one (children
    started with start_new_session=True), else to the pid."""
    import signal

    if IS_WINDOWS:
        with contextlib.suppress(Exception):
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True, timeout=20,
                           creationflags=_CREATE_NO_WINDOW)
        return
    with contextlib.suppress(OSError):
        if os.getpgid(int(pid)) == int(pid):
            os.killpg(int(pid), signal.SIGKILL)
            return
    with contextlib.suppress(OSError):
        os.kill(int(pid), signal.SIGKILL)


def wait_unlocked(path: Path, timeout: float, poll: float = 0.25) -> bool:
    """True once nobody holds the lock of `path` (within timeout)."""
    end = time.time() + timeout
    while is_locked(path):
        if time.time() >= end:
            return False
        time.sleep(poll)
    return True


# Windows creation flags
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def spawn_detached(argv: list[str], *, cwd: Path | str | None = None, log_path: Path | None = None,
                   env: dict | None = None) -> int:
    """Start `argv` in the background, independent of this process (it keeps running when the caller exits). stdout/stderr
    are appended to log_path (or discarded). Returns the pid."""
    out = subprocess.DEVNULL
    fh = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab")
        out = fh
    kw: dict = {"cwd": str(cwd) if cwd else None, "stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT,
                "env": env, "close_fds": True}
    try:
        if IS_WINDOWS:
            base = _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
            try:
                p = subprocess.Popen(argv, creationflags=base | _CREATE_BREAKAWAY_FROM_JOB, **kw)
            except OSError:
                # the caller runs in a job that does not allow breakaway: staying in it could kill the bot when the caller's
                # job closes (KILL_ON_JOB_CLOSE). Let WMI create it instead - its parent is the WMI host, outside any job of ours.
                return _spawn_via_wmi(argv, cwd=cwd, log_path=log_path)
        else:
            p = subprocess.Popen(argv, start_new_session=True, **kw)
        return p.pid
    finally:
        if fh is not None:
            fh.close()


def _spawn_via_wmi(argv: list[str], *, cwd, log_path: Path | None) -> int:
    """Win32_Process.Create (hidden window). The environment is NOT inherited - callers pass what the child needs as arguments
    (r020/r021 take --runtime)."""
    argv = [str(a) for a in argv]
    if Path(argv[0]).name.lower().startswith("python"):
        argv = [argv[0], "-X", "utf8", "-u", *argv[1:]]  # no PYTHONUTF8 / PYTHONUNBUFFERED without an inherited environment
    cmdline = subprocess.list2cmdline(argv)
    if log_path is not None:
        cmdline = f'cmd.exe /d /s /c "{cmdline} >> "{log_path}" 2>&1"'
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
          "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          "@{CommandLine=$env:R019_CMD; CurrentDirectory=$env:R019_CWD; ProcessStartupInformation=$si}; "
          "if ($r.ReturnValue -ne 0) { exit [int]$r.ReturnValue }; [Console]::Out.Write($r.ProcessId)")
    env = dict(os.environ, R019_CMD=cmdline, R019_CWD=str(cwd or os.getcwd()))
    cp = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True,
                        timeout=60, env=env, creationflags=_CREATE_NO_WINDOW)
    if cp.returncode != 0 or not cp.stdout.strip().isdigit():
        raise OSError(f"WMI process create failed (rc={cp.returncode}): {(cp.stderr or cp.stdout).strip()[:300]}")
    return int(cp.stdout.strip())


def child_env(extra: dict | None = None) -> dict:
    """Environment for Python children: UTF-8 stdio/files on Windows (cp949 console otherwise) and unbuffered output."""
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def python_exe() -> str:
    """The interpreter running this code (venv python on both hosts)."""
    return sys.executable or "python"
