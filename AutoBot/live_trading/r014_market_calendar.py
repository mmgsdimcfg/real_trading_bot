# -*- coding: utf-8 -*-
"""r014_market_calendar.py - KRX regular-session hours per day (holidays / late-open days).

Reads r014_market_calendar.txt next to this file:
  YYYYMMDD closed            weekday holiday
  YYYYMMDD HH:MM HH:MM       non-standard regular session (e.g. CSAT day 10:00~16:30)
  YYYYMMDD HH:MM HH:MM key=HH:MM-HH:MM|none ...   the same plus explicit extended sessions (keys below)
Any other weekday is 09:00~15:30, weekends are closed. A malformed line is ignored (never raises),
so a broken file degrades to the normal schedule instead of stopping a collector.

trading_hours(day) adds the sessions outside the KRX regular one (2026-11-19, observed on KIS minute bars):
  nxt_pre    NXT pre-market         normal 08:00~08:50; a late-open day: none (CSAT day 20261119 had no NXT pre-market)
  nxt_after  NXT after-market       KRX close + 10 min ~ 20:00 (normal 15:40~20:00, CSAT day 16:40~20:00)
  krx_after  KRX after-hours market KRX close + 30 min ~ 20:00 (normal 16:00~20:00, CSAT day 17:00~20:00) - continuous 1-minute
             prints of every KRX symbol (also those that do not trade on NXT, e.g. 403870 HPSP); KRX has no pre-market
A key on the calendar line overrides the derived value ("none" = not held that day).

Used by g011 (default --until, idle watchdog window) and g024 (skip checks on holidays); trading_hours by r017/r018 (chart
minute bars outside the regular session, AutoBotH). g022_tick_collect_daily.sh parses the same file with awk ($2 $3 only).

Update log:
- [2026-11-19] type=feat owner=claude
    summary: 사용자 요청(HPSP 처럼 NXT 비대상이지만 KRX 장후에 거래되는 종목 구분) - trading_hours(day) 신설: 정규장 + NXT 장전/장후 +
      KRX 장후(16:00~20:00, 연속 체결) 시각. 특수일은 마감 시각에서 파생(수능일: NXT 장전 없음, NXT 장후 16:40, KRX 장후 17:00),
      달력 줄의 nxt_pre= / nxt_after= / krx_after= 로 덮어쓸 수 있다.
    impact: 차트 데이터(r017/r018). r003 매매 시각은 기존 r001 고정값 그대로(변경은 별도 승인 사항)
    compatibility: backward-compatible (load_calendar / session_hours 동작 동일, 달력 줄 뒤 키는 g022 awk 가 무시)
- [2026-11-02] type=feat owner=claude
    summary: 신규 - 특수일(휴장/수능일 등) 정규장 시각표. g011 기본 종료 시각과 무수신 감시 구간, g022 기동 구간이 사용.
      같은 날짜 줄이 여러 개면 첫 줄 사용, closed 대소문자 무시(g022 awk와 같은 규칙).
    impact: collector (r003 매매 시각은 기존 REGULAR_START/END 그대로)
    compatibility: backward-compatible (파일 없으면 평일 09:00~15:30)
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

CALENDAR_PATH = Path(__file__).resolve().parent / "r014_market_calendar.txt"
DEFAULT_OPEN = time(9, 0)
DEFAULT_CLOSE = time(15, 30)
NXT_PRE = (time(8, 0), time(8, 50))
EXT_END = time(20, 0)                        # NXT after-market and KRX after-hours both end at 20:00
NXT_AFTER_FROM_CLOSE = timedelta(minutes=10)
KRX_AFTER_FROM_CLOSE = timedelta(minutes=30)
EXT_KEYS = ("nxt_pre", "nxt_after", "krx_after")


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


def _ext_overrides(path: Path, key: str) -> dict[str, tuple[time, time] | None]:
    """{nxt_pre|nxt_after|krx_after: (start, end) or None} written on the first calendar line of `key` (bad tokens ignored)."""
    out: dict[str, tuple[time, time] | None] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts or parts[0] != key:
            continue
        for tok in parts[3:]:
            k, _, v = tok.partition("=")
            if k not in EXT_KEYS or not v:
                continue
            if v.lower() == "none":
                out[k] = None
                continue
            a, _, b = v.partition("-")
            try:
                out[k] = (_parse_hhmm(a), _parse_hhmm(b))
            except ValueError:
                continue
        break  # the first line of the date wins (same as load_calendar)
    return out


def _plus(t: time, delta: timedelta) -> time:
    return (datetime.combine(date(2000, 1, 1), t) + delta).time()


def trading_hours(day: date | datetime | str, path: Path = CALENDAR_PATH) -> dict | None:
    """The day's sessions, or None when the market is closed:
    {"regular": (open, close), "nxt_pre": (s, e) | None, "nxt_after": (s, e) | None, "krx_after": (s, e) | None}."""
    reg = session_hours(day, path)
    if reg is None:
        return None
    if isinstance(day, str):
        key = day
    else:
        key = (day.date() if isinstance(day, datetime) else day).strftime("%Y%m%d")
    open_t, close_t = reg
    out = {
        "regular": reg,
        "nxt_pre": NXT_PRE if open_t == DEFAULT_OPEN else None,
        "nxt_after": (_plus(close_t, NXT_AFTER_FROM_CLOSE), EXT_END),
        "krx_after": (_plus(close_t, KRX_AFTER_FROM_CLOSE), EXT_END),
    }
    for k, v in _ext_overrides(path, key).items():
        out[k] = v
    for k in EXT_KEYS:  # a session that would start at/after 20:00 is not held
        if out[k] is not None and out[k][0] >= out[k][1]:
            out[k] = None
    return out


def trading_hours_json(day: date | datetime | str, path: Path = CALENDAR_PATH) -> dict | None:
    """trading_hours with "HH:MM" strings (r017 output / AutoBotH)."""
    h = trading_hours(day, path)
    if h is None:
        return None
    return {k: ([v[0].strftime("%H:%M"), v[1].strftime("%H:%M")] if v else None) for k, v in h.items()}
