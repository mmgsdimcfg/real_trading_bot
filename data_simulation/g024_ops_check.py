# -*- coding: utf-8 -*-
"""g024_ops_check.py - after-close health checks for the unattended jobs; emails (g016) only when something is wrong.

  --ticks    (cron after the close, e.g. 16:45): today's g011 tick files - no files / no rows, collector stopped
             early, frequent reconnects, failed subscriptions. Skipped on a holiday (r014 calendar).
  --edge-ai  (called by g015 at the end): g013/g014 exit codes and whether current_meta.json exists.

Exit code is 0 even when a problem is found (the mail is the alert); stdout is the cron log line.

Run:
  python g024_ops_check.py --ticks [--date 20261102] [--mail-always] [--no-mail]
  python g024_ops_check.py --edge-ai --rc-dataset 0 --rc-train 0 --log <daily log>

Update log:
- [2026-11-02] type=feat owner=claude
    summary: 신규 - Codex cron 검토(알림 부재) 반영. 틱 수집 장 마감 후 점검 + AI 일일 학습 실패 알림(g016 메일).
    impact: ops (주문/매매 무관, 읽기 전용 + 메일)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))

from r001_define_config import RUNTIME_DATA_DIR, EDGE_AI_DIR  # noqa: E402
from r014_market_calendar import session_hours  # noqa: E402

MAX_DISCONNECTS = 10            # more reconnects than this in a day -> warn
STOP_EARLY_MINUTES = 10         # newest tick file older than close - this -> collector died / stalled
MIN_ROWS_PER_CODE = 1           # a subscribed code with fewer data rows is listed (halted codes legitimately 0)
MAX_ZERO_ROW_CODE_RATIO = 0.25  # more than this share of files without rows -> problem


def _data_rows(path: Path) -> int:
    with open(path, "rb") as f:
        n = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    return max(0, n - 1)  # header


def check_ticks(day: datetime) -> tuple[list[str], list[str]]:
    """(problems, info lines) for one trading day's tick collection."""
    hours = session_hours(day)
    if hours is None:
        return [], [f"{day:%Y%m%d} market closed - skipped"]
    problems: list[str] = []
    info: list[str] = []
    tick_dir = RUNTIME_DATA_DIR / "ticks" / day.strftime("%Y%m%d")
    files = sorted(tick_dir.glob("*.csv")) if tick_dir.is_dir() else []
    if not files:
        return [f"틱 파일 없음: {tick_dir} (g011이 오늘 실행되지 않음)"], info
    log_path = tick_dir / "_collector.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    # every code g011 was asked to subscribe today (its "CODES a,b,..." lines); files appear only on the first print
    requested = {c for ln in text.splitlines() if " CODES " in ln
                 for c in ln.split(" CODES ", 1)[1].strip().split(",") if c}
    rows = {f.stem: _data_rows(f) for f in files}
    for code in requested - rows.keys():
        rows[code] = 0
    total = sum(rows.values())
    empty = sorted(c for c, n in rows.items() if n < MIN_ROWS_PER_CODE)
    newest = max(f.stat().st_mtime for f in files)
    close_dt = datetime.combine(day.date(), hours[1])
    info.append(f"codes={len(rows)} (requested={len(requested)}, files={len(files)}) rows={total:,} "
                f"empty_codes={len(empty)} newest_write={datetime.fromtimestamp(newest):%H:%M:%S} "
                f"close={hours[1]:%H:%M}")
    if total == 0:
        problems.append("모든 틱 파일에 체결 0건")
    elif len(empty) > len(rows) * MAX_ZERO_ROW_CODE_RATIO:
        problems.append(f"체결 0건(파일 없음 포함) 종목 {len(empty)}/{len(rows)}: {empty[:20]}")
    if datetime.fromtimestamp(newest) < close_dt - timedelta(minutes=STOP_EARLY_MINUTES):
        problems.append(f"마지막 기록 {datetime.fromtimestamp(newest):%H:%M:%S} - 마감({hours[1]:%H:%M}) 전에 수집 중단")
    starts = text.count(" START ")
    disconnects = [ln for ln in text.splitlines() if " DISCONNECTED " in ln]
    idle = sum(1 for ln in disconnects if "IDLE" in ln or "subscribe ACK" in ln)
    sub_fail = re.findall(r"SUBSCRIPTIONS ok=(\d+) failed=(\d+)", text)
    torn = sum(int(x) for x in re.findall(r"torn_rows_closed=(\d+)", text))
    info.append(f"starts={starts} disconnects={len(disconnects)} (idle/ack={idle}) torn_rows={torn}")
    if len(disconnects) > MAX_DISCONNECTS:
        problems.append(f"재접속 {len(disconnects)}회(무수신/ACK {idle}회) - 마지막: {disconnects[-1][:160]}")
    if sub_fail and int(sub_fail[-1][1]) > 0:
        problems.append(f"구독 실패 {sub_fail[-1][1]}종목 (ok={sub_fail[-1][0]})")
    if "WARN still no subscribe ACK" in text:
        problems.append("구독 ACK 누락이 재접속 후에도 계속됨(_collector.log의 WARN 참고)")
    return problems, info


def check_edge_ai(rc_dataset: int, rc_train: int, log_path: Path | None) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    if rc_dataset != 0:
        problems.append(f"g013(학습 데이터 생성) 실패 rc={rc_dataset}")
    if rc_train != 0:
        problems.append(f"g014(학습/배포) 실패 rc={rc_train}")
    if not (EDGE_AI_DIR / "current_meta.json").exists():
        problems.append("배포된 모델 없음(current_meta.json) - r011은 fail-open(no_model)으로 기록만 함")
    info = [f"rc_dataset={rc_dataset} rc_train={rc_train}"]
    if problems and log_path is not None and log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        info.append("--- log tail ---")
        info.extend(tail)
    return problems, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", action="store_true")
    ap.add_argument("--edge-ai", action="store_true")
    ap.add_argument("--date", default=None, help="YYYYMMDD for --ticks (default today)")
    ap.add_argument("--rc-dataset", type=int, default=0)
    ap.add_argument("--rc-train", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--mail-always", action="store_true", help="send the summary even when everything is fine")
    ap.add_argument("--no-mail", action="store_true")
    args = ap.parse_args()
    if args.ticks == args.edge_ai:
        ap.error("choose exactly one of --ticks / --edge-ai")

    day = datetime.strptime(args.date, "%Y%m%d") if args.date else datetime.now()
    if args.ticks:
        title = f"틱 수집 {day:%Y%m%d}"
        problems, info = check_ticks(day)
    else:
        title = f"AI 일일 학습 {day:%Y%m%d}"
        problems, info = check_edge_ai(args.rc_dataset, args.rc_train, Path(args.log) if args.log else None)

    status = "PROBLEM" if problems else "OK"
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} g024 {title}: {status} | " + " | ".join(info[:2]), flush=True)
    for p in problems:
        print(f"  - {p}", flush=True)
    if args.no_mail or not (problems or args.mail_always):
        return
    from g016_email_notify import send_text

    body = "\n".join([f"[{status}] {title}", ""] + [f"- {p}" for p in problems] + [""] + info)
    if not send_text(f"[autobot] {title} {status}", body):
        print("  [EMAIL] 알림 전송 실패", flush=True)


if __name__ == "__main__":
    main()
