# -*- coding: utf-8 -*-
"""r014_market_calendar.py - KRX regular-session hours per day (holidays / late-open days).

Reads r014_market_calendar.txt next to this file:
  YYYYMMDD closed            weekday holiday
  YYYYMMDD HH:MM HH:MM       non-standard regular session (e.g. CSAT day 10:00~16:30)
Any other weekday is 09:00~15:30, weekends are closed. A malformed line is ignored (never raises),
so a broken file degrades to the normal schedule instead of stopping a collector.

Used by g011 (default --until, idle watchdog window) and g024 (skip checks on holidays).
g022_tick_collect_daily.sh parses the same file with awk.

Update log:
- [2026-11-02] type=feat owner=claude
    summary: 신규 - 특수일(휴장/수능일 등) 정규장 시각표. g011 기본 종료 시각과 무수신 감시 구간, g022 기동 구간이 사용.
      같은 날짜 줄이 여러 개면 첫 줄 사용, closed 대소문자 무시(g022 awk와 같은 규칙).
    impact: collector (r003 매매 시각은 기존 REGULAR_START/END 그대로)
    compatibility: backward-compatible (파일 없으면 평일 09:00~15:30)
"""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

CALENDAR_PATH = Path(__file__).resolve().parent / "r014_market_calendar.txt"
DEFAULT_OPEN = time(9, 0)
DEFAULT_CLOSE = time(15, 30)


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def load_calendar(path: Path = CALENDAR_PATH) -> dict[str, tuple[time, time] | None]:
    """{YYYYMMDD: (open, close) or None (closed)}."""
    out: dict[str, tuple[time, time] | None] = {}
    seen: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts or len(parts[0]) != 8 or not parts[0].isdigit() or parts[0] in seen:
            continue
        seen.add(parts[0])  # the first line for a date wins even when malformed (g022's awk does the same)
        try:
            if len(parts) >= 2 and parts[1].lower() == "closed":
                out[parts[0]] = None
            elif len(parts) >= 3:
                out[parts[0]] = (_parse_hhmm(parts[1]), _parse_hhmm(parts[2]))
        except ValueError:
            continue
    return out


def session_hours(day: date | datetime | str, path: Path = CALENDAR_PATH) -> tuple[time, time] | None:
    """(open, close) of the regular session, or None when the market is closed that day."""
    if isinstance(day, str):
        d = datetime.strptime(day, "%Y%m%d").date()
    elif isinstance(day, datetime):
        d = day.date()
    else:
        d = day
    key = d.strftime("%Y%m%d")
    cal = load_calendar(path)
    if key in cal:
        return cal[key]
    if d.weekday() >= 5:
        return None
    return DEFAULT_OPEN, DEFAULT_CLOSE
