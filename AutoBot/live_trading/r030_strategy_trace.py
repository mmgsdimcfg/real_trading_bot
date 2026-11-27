#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-bar evaluation trace of the side strategies (Edge AI / VWAP; real orders and paper) for the app's timeline.

Update log
    - [2026-11-27] type=feat - 사용자 요청(타임라인을 실행 중인 매매 방식에 맞게): 보조 전략은 1분봉마다 한 번 판정하지만 어느
      단계에서 멈췄는지는 남지 않았다(실주문 경로는 신호 조건 실패를 일부러 조용히 넘기고, 가상 매매 상태 파일은 종목별 "마지막"
      단계만 가진다). 판정이 끝난 자리에서 한 줄씩 runtime/data/live_runtime/strategy_eval_<strategy>_<date>.jsonl 에 남긴다.
      판정에는 쓰이지 않는 기록 전용이다: 호출은 큐에 넣기만 하고(가득 차면 버리고 센다) 파일 쓰기는 별도 데몬 스레드가 한다 -
      디스크가 느려도 매매 루프는 기다리지 않는다. 어떤 오류도 호출한 쪽으로 올라가지 않는다. Codex 계획 검토 반영(별도 파일,
      실제 판정 경계에서 기록, 비동기 쓰기 + 누락 카운터).

Row: {"t": 판정 시각(ISO, 초), "bar": 판정한 봉 "HH:MM", "code", "mode": "live" | "paper", "stage", ...}
    stage  멈춘 단계 / 결과. VWAP: below_vwap / no_trend / no_tick_cross / chase / price_drop / stale / no_history,
           Edge: below_mid / no_trigger / below_threshold / not_promoted / no_model, 공통: cond_before / cond_after (공통 안전 조건,
           no = 조건 번호) / signal / ordered / order_refused / bought / candidate / held / wait_bars / outside_window / limit ...
    그 밖의 키는 봇이 그 판정에 쓴 값 그대로(vwap, gap, tick, prob, thr, price, no, detail) - 앱은 다시 계산하지 않는다.
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

QUEUE_MAX = 4000
_q: "queue.Queue[tuple[Path, str]]" = queue.Queue(maxsize=QUEUE_MAX)
_lock = threading.Lock()
_write_lock = threading.Lock()   # one drain at a time (writer thread / flush): rows reach the file in the order they were queued
_wake = threading.Event()
_thread: threading.Thread | None = None
dropped = 0        # rows not written because the queue was full or a row could not be encoded
write_errors = 0   # batches the writer could not append


def path_for(out_dir: Path, strategy: str, date_str: str) -> Path:
    return Path(out_dir) / f"strategy_eval_{strategy}_{date_str}.jsonl"


def _clean(v):
    if isinstance(v, float):
        return None if v != v or v in (float("inf"), float("-inf")) else round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)[:200]


_dirty: set[Path] = set()         # files whose last write failed: the next batch starts on a new line


def _drain(wait: float = -1) -> None:
    """write everything queued now, grouped per file (one open per file and batch). wait: seconds to wait for another drain
    (-1 = until it is done)."""
    global write_errors
    if not _write_lock.acquire(timeout=wait):
        return
    try:
        batch: dict[Path, list[str]] = {}
        while True:
            try:
                item = _q.get_nowait()
            except queue.Empty:
                break
            batch.setdefault(item[0], []).append(item[1])
        for path, lines in batch.items():
            try:                                  # the folder is the bot's live_runtime (it exists): never created here
                with open(path, "a", encoding="utf-8") as f:
                    # after a failed write the file may end in half a row: a line break first, so the rows that follow are
                    # whole (the reader skips the broken one)
                    f.write(("\n" if path in _dirty else "") + "".join(lines))
                _dirty.discard(path)
            except Exception:
                write_errors += 1
                _dirty.add(path)
    finally:
        _write_lock.release()


def _run() -> None:
    while True:
        try:
            _wake.wait(5.0)
            _wake.clear()
            _drain()
        except Exception:
            pass


def _ensure_thread() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_run, name="strategy-trace", daemon=True)
            _thread.start()


def trace(out_dir, strategy: str, date_str: str, now: datetime, code: str, mode: str, stage: str, **fields) -> bool:
    """queue one row (never blocks, never raises). False = the row was dropped."""
    global dropped
    try:
        row = {"t": now.isoformat(timespec="seconds"), "code": str(code).zfill(6), "mode": "paper" if mode == "paper" else "live",
               "stage": str(stage)}
        for k, v in fields.items():
            v = _clean(v)
            if v is not None and v != "":
                row[k] = v
        line = json.dumps(row, ensure_ascii=False) + "\n"
        _q.put_nowait((path_for(Path(out_dir), str(strategy), str(date_str)), line))
        _ensure_thread()
        _wake.set()
        return True
    except Exception:
        dropped += 1
        return False


def flush(wait: float = -1) -> None:
    """write what is queued in the calling thread (tests; interpreter exit with a bounded wait)"""
    try:
        _drain(wait)
    except Exception:
        pass


def _flush_at_exit(limit: float = 2.0) -> None:
    """[Codex round 3] at interpreter exit: the drain (lock AND file write) runs in a daemon thread that is waited for at most
    `limit` seconds - a disk that hangs cannot hold the exit; what is not written by then is lost (a record only)"""
    try:
        t = threading.Thread(target=flush, args=(limit,), name="strategy-trace-exit", daemon=True)
        t.start()
        t.join(limit)
    except Exception:
        pass


atexit.register(_flush_at_exit)


# ---- the stage of a refused real-order bar, from the condition that refused it and what that condition wrote -------------
_SIGNAL_STAGES = (
    # VWAP signal (r005 _v09_vwap_signal -> r029.vwap_signal stage names, upper-cased in its reject line)
    ("VWAP_SIGNAL_BELOW_VWAP", "below_vwap"), ("VWAP_SIGNAL_NO_TREND", "no_trend"), ("VWAP_SIGNAL_NO_TICK_CROSS", "no_tick_cross"),
    ("VWAP_SIGNAL_CHASE", "chase"), ("VWAP_SIGNAL_PRICE_DROP", "price_drop"), ("VWAP_SIGNAL_STALE", "stale"),
    ("VWAP_SIGNAL_NO_HISTORY", "no_history"), ("VWAP_SIGNAL_INSUFFICIENT_BARS", "wait_bars"),
    ("VWAP_SIGNAL_NO_DAY_VWAP", "no_history"), ("VWAP_SIGNAL_1MIN_FRAME_UNAVAILABLE", "wait_bars"),
    ("VWAP_SIGNAL_REGULAR_SESSION_ONLY", "outside_window"),
    # Edge AI signal (r005 _e09_edge_ai_signal)
    ("EDGE_SIGNAL_NO_TRIGGER_BELOW_1MIN_MID", "below_mid"), ("EDGE_SIGNAL_LIVE_BELOW_1MIN_MID", "below_mid"),
    ("EDGE_SIGNAL_NO_TRIGGER", "no_trigger"), ("EDGE_SIGNAL_LOW_PROB", "below_threshold"),
    ("EDGE_SIGNAL_MODEL_NOT_PROMOTED", "not_promoted"), ("EDGE_SIGNAL_NO_PROB", "no_model"),
    ("EDGE_SIGNAL_1MIN_FRAME_STALE", "stale"), ("EDGE_SIGNAL_1MIN_FRAME_UNAVAILABLE", "wait_bars"),
    ("EDGE_SIGNAL_INSUFFICIENT_BARS", "wait_bars"), ("EDGE_SIGNAL_REGULAR_SESSION_ONLY", "outside_window"),
)


def reject_stage(no: int, detail: str) -> str:
    """stage name of a bar the buy conditions refused: no = number of the refusing condition (9 = the strategy's signal),
    detail = what it wrote. A text this table does not know is 'error' (shown with the detail as written)."""
    if no < 9:
        return "cond_before"
    if no > 9:
        return "cond_after"
    d = str(detail or "")
    for prefix, stage in _SIGNAL_STAGES:
        if d.startswith(prefix):
            return stage
    return "error"
