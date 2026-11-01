# -*- coding: utf-8 -*-
"""R013 tick-chart trigger - 60/120-tick bar MA5 > MA20 golden cross, OR'ed with the 1-min BB-mid cross (r005 _008).

Data path
- g011_tick_collector (separate process, KIS websocket H0STCNT0) appends every execution print to
  <TICK_DATA_DIR>/YYYYMMDD/<code>.csv and flushes about once per second.
- TickTriggerFeed tails those files incrementally (byte offset per code, partial last line kept for the next read) and
  feeds r009 TickBarBuilder per bar size, so each poll only parses the prints added since the previous poll.
- Contract: the file is append-only (g011 TickWriter opens it with mode "a"). As a safety net the reader starts over
  when the inode changes, the file shrinks, or the last bytes it already read differ; a same-inode rewrite that keeps
  those last bytes identical is NOT detected - writers must append or replace the file atomically.

Signal (per bar size n, completed bars of today only, no look-ahead via TickBar.available_at)
- MA_fast / MA_slow of tick-bar closes. Cross bar i: MA_fast[i-1] <= MA_slow[i-1] and MA_fast[i] > MA_slow[i], and
  MA_fast stayed above MA_slow on every bar after i (latest cross of the current "above" run).
- Valid while now - available_at(cross bar) <= max_age_s.
- Fail-closed (not ok, TICK<n>_ABSTAIN_*): no file, feed stale (last print received > stale_s ago), fewer than
  slow+1 bars, or an invalid bar (missed prints across a boundary) among the bars the MAs at i-1..last are built from.

This module never raises into the caller: TickTriggerFeed.evaluate() turns any exception into a not-ok result, so the
1-min trigger keeps working when the tick feed is broken.

Update log:
- [2026-11-02] type=feat owner=claude
    summary: TickReplayFeed 신규 - g003 3중 트리거 재현용. 하루치 g011 파일을 한 번에 읽어 과거 시각 now 기준으로 판정
      (봉은 tick_ma_golden_cross가 available_at <= now로 거르고, stale은 now 이전 마지막 수신 시각 기준). 라이브 미사용.
    impact: sim (g003)
    compatibility: backward-compatible (신규 클래스)
- [2026-10-29] type=feat owner=claude
    summary: 신규 - 60/120틱봉 MA5/MA20 골든크로스 판정 + g011 수집 파일 증분 리더(사용자 요청, 003160 디아이 10:42 사례).
    impact: live (r003 -> r005 _008, r001 TICK_TRIGGER_MODE="live")
    compatibility: backward-compatible (신규 모듈, TICK_TRIGGER_MODE="off"면 미사용)
"""
from __future__ import annotations

import bisect
import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from r009_tick_bars import TickBar, TickBarBuilder, iter_tick_file, parse_tick_row

_TAIL_CHECK_BYTES = 256  # > one CSV row, so a rewrite of the last read row is always noticed


@dataclass
class TickCross:
    ok: bool
    reason: str
    n: int
    cross_time: datetime | None = None  # available_at of the cross bar
    ma_fast: float = float("nan")
    ma_slow: float = float("nan")
    bars: int = 0


@dataclass
class TickTriggerResult:
    ok: bool
    label: str = ""                     # e.g. "TICK60_MA5_MA20" of the passing cross ("" if none)
    reason: str = ""                    # per-size reasons joined by "|"
    cross_time: datetime | None = None
    crosses: dict[int, TickCross] = field(default_factory=dict)


def tick_ma_golden_cross(bars: list[TickBar], now: datetime, n: int, fast: int, slow: int,
                         max_age_s: float) -> TickCross:
    tag = f"TICK{n}"
    recent = [b for b in bars if b.available_at <= now and b.end_ts.date() == now.date()]
    if len(recent) < slow + 1:
        return TickCross(False, f"{tag}_ABSTAIN_FEW_BARS_{len(recent)}", n, bars=len(recent))
    closes = [b.close for b in recent]
    prefix = [0.0]
    for c in closes:
        prefix.append(prefix[-1] + c)

    def ma(i: int, w: int) -> float:
        return (prefix[i + 1] - prefix[i + 1 - w]) / w

    def above(i: int) -> bool:
        return ma(i, fast) > ma(i, slow)

    last = len(recent) - 1
    first = slow - 1  # first index with both MAs defined
    f_last, s_last = ma(last, fast), ma(last, slow)
    if not above(last):
        return TickCross(False, f"{tag}_MA{fast}_NOT_ABOVE_MA{slow}", n, ma_fast=f_last, ma_slow=s_last, bars=len(recent))
    i = last
    while i - 1 >= first and above(i - 1):
        if (now - recent[i - 1].available_at).total_seconds() > max_age_s:
            # the run started before the age window: whatever the cross bar is, it is already expired
            return TickCross(False, f"{tag}_CROSS_EXPIRED", n, ma_fast=f_last, ma_slow=s_last, bars=len(recent))
        i -= 1
    if i - 1 < first:
        return TickCross(False, f"{tag}_NO_CROSS_IN_SESSION", n, ma_fast=f_last, ma_slow=s_last, bars=len(recent))
    cross_bar = recent[i]
    age = (now - cross_bar.available_at).total_seconds()
    if age > max_age_s:
        return TickCross(False, f"{tag}_CROSS_EXPIRED_{age:.0f}s", n, cross_bar.available_at, f_last, s_last, len(recent))
    # MA_slow at i-1 uses bars i-slow .. i-1, so every bar from i-slow on decides this cross
    if any(b.invalid for b in recent[i - slow:]):
        return TickCross(False, f"{tag}_ABSTAIN_INVALID_BAR", n, cross_bar.available_at, f_last, s_last, len(recent))
    return TickCross(True, f"{tag}_MA{fast}_MA{slow}_GOLDEN_CROSS", n, cross_bar.available_at, f_last, s_last, len(recent))


class _CodeFeed:
    """Incremental reader of one g011 tick file + one TickBarBuilder per bar size."""

    def __init__(self, code: str, path: Path, sizes: tuple[int, ...]):
        self.code = code
        self.path = path
        self.offset = 0
        self.pending = b""
        self.header: list[str] | None = None
        self.builders = {n: TickBarBuilder(code=code, n=n) for n in sizes}
        self.last_recv: datetime | None = None
        self.file_id: tuple[int, int] | None = None  # (st_dev, st_ino) of the file the state was built from
        self.tail = b""  # last bytes before offset - detects an in-place rewrite of already-read content

    def _reset(self) -> None:
        self.__init__(self.code, self.path, tuple(self.builders))

    def poll(self) -> None:
        try:
            f = open(self.path, "rb")
        except FileNotFoundError:
            return
        with f:
            st = os.fstat(f.fileno())
            file_id = (st.st_dev, st.st_ino)
            if self.file_id is not None and file_id != self.file_id:
                self._reset()  # file replaced (new inode): bars from the old file must not vote
            elif st.st_size < self.offset:
                self._reset()  # truncated
            elif self.tail:
                f.seek(self.offset - len(self.tail))
                if f.read(len(self.tail)) != self.tail:
                    self._reset()  # rewritten in place (same or larger size)
            self.file_id = file_id
            if st.st_size == self.offset:
                return
            f.seek(self.offset)
            chunk = f.read(st.st_size - self.offset)
        self.offset += len(chunk)
        self.tail = (self.tail + chunk)[-_TAIL_CHECK_BYTES:]
        data = self.pending + chunk
        lines = data.split(b"\n")
        self.pending = lines.pop()  # incomplete last line (writer flushed mid-row) waits for the next poll
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            if not line:
                continue
            fields = next(csv.reader([line]))
            if self.header is None:
                self.header = fields
                continue
            if len(fields) != len(self.header):
                continue
            tick = parse_tick_row(dict(zip(self.header, fields)))
            if tick is None:
                continue
            if tick.recv is not None and (self.last_recv is None or tick.recv > self.last_recv):
                self.last_recv = tick.recv
            for b in self.builders.values():
                b.ingest(tick)


class TickTriggerFeed:
    def __init__(self, data_dir: Path, sizes: tuple[int, ...], fast: int, slow: int, max_age_s: float,
                 stale_s: float, log: Callable[[str], None] | None = None):
        self.data_dir = Path(data_dir)
        self.sizes = tuple(int(n) for n in sizes)
        self.fast = int(fast)
        self.slow = int(slow)
        self.max_age_s = float(max_age_s)
        self.stale_s = float(stale_s)
        self.log = log
        self._feeds: dict[str, _CodeFeed] = {}
        self._day: str | None = None
        self._logged: set[tuple] = set()

    def _feed(self, code: str, now: datetime) -> _CodeFeed:
        day = now.strftime("%Y%m%d")
        if day != self._day:
            self._day = day
            self._feeds.clear()
            self._logged.clear()
        feed = self._feeds.get(code)
        if feed is None:
            feed = _CodeFeed(code, self.data_dir / day / f"{code}.csv", self.sizes)
            self._feeds[code] = feed
        return feed

    def evaluate(self, code: str, now: datetime) -> TickTriggerResult:
        try:
            return self._evaluate(str(code).zfill(6), now)
        except Exception as exc:  # never break the buy loop because of the tick feed
            return TickTriggerResult(False, reason=f"TICK_ERROR_{type(exc).__name__}")

    def _evaluate(self, code: str, now: datetime) -> TickTriggerResult:
        feed = self._feed(code, now)
        if not feed.path.exists():
            return TickTriggerResult(False, reason="TICK_ABSTAIN_NO_FILE")
        feed.poll()
        if feed.last_recv is None:
            return TickTriggerResult(False, reason="TICK_ABSTAIN_NO_TICKS")
        idle = (now - feed.last_recv).total_seconds()
        if idle > self.stale_s:
            return TickTriggerResult(False, reason=f"TICK_ABSTAIN_FEED_STALE_{idle:.0f}s")
        result = TickTriggerResult(False)
        for n in self.sizes:
            c = tick_ma_golden_cross(feed.builders[n].bars, now, n, self.fast, self.slow, self.max_age_s)
            result.crosses[n] = c
            if c.ok:
                self._log_cross(code, c)
                if not result.ok:
                    result.ok, result.label, result.cross_time = True, f"TICK{n}_MA{self.fast}_MA{self.slow}", c.cross_time
        result.reason = "|".join(c.reason for c in result.crosses.values())
        return result

    def _log_cross(self, code: str, c: TickCross) -> None:
        key = (code, c.n, c.cross_time)
        if self.log is None or key in self._logged:
            return
        self._logged.add(key)
        fmt = lambda v: "nan" if math.isnan(v) else f"{v:,.1f}"  # noqa: E731
        self.log(f"  [TICK CROSS  ] {code} | tick{c.n} MA{self.fast}>MA{self.slow} at {c.cross_time:%H:%M:%S} "
                 f"ma{self.fast}={fmt(c.ma_fast)} ma{self.slow}={fmt(c.ma_slow)} bars={c.bars}")


class TickReplayFeed:
    """Simulation counterpart of TickTriggerFeed (g003): each code's whole-day g011 file is read once in file (= receive)
    order, then evaluate(code, now) answers as the live feed would have at `now` - same cross rule and abstain reasons.
    Completed bars are filtered by available_at <= now inside tick_ma_golden_cross; the stale check uses the last print
    received at or before `now`."""

    def __init__(self, data_dir: Path, sizes: tuple[int, ...], fast: int, slow: int, max_age_s: float,
                 stale_s: float):
        self.data_dir = Path(data_dir)
        self.sizes = tuple(int(n) for n in sizes)
        self.fast = int(fast)
        self.slow = int(slow)
        self.max_age_s = float(max_age_s)
        self.stale_s = float(stale_s)
        self._loaded: dict[tuple[str, str], tuple[dict[int, TickBarBuilder], list[datetime]] | None] = {}

    def _load(self, code: str, day: str):
        key = (code, day)
        if key not in self._loaded:
            path = self.data_dir / day / f"{code}.csv"
            if not path.exists():
                self._loaded[key] = None
            else:
                builders = {n: TickBarBuilder(code=code, n=n) for n in self.sizes}
                recvs: list[datetime] = []
                for tick in iter_tick_file(path):
                    if tick.recv is not None:
                        recvs.append(tick.recv)
                    for b in builders.values():
                        b.ingest(tick)
                recvs.sort()
                self._loaded[key] = (builders, recvs)
        return self._loaded[key]

    def evaluate(self, code: str, now: datetime) -> TickTriggerResult:
        try:
            return self._evaluate(str(code).zfill(6), now)
        except Exception as exc:
            return TickTriggerResult(False, reason=f"TICK_ERROR_{type(exc).__name__}")

    def _evaluate(self, code: str, now: datetime) -> TickTriggerResult:
        loaded = self._load(code, now.strftime("%Y%m%d"))
        if loaded is None:
            return TickTriggerResult(False, reason="TICK_ABSTAIN_NO_FILE")
        builders, recvs = loaded
        i = bisect.bisect_right(recvs, now)
        if i == 0:
            return TickTriggerResult(False, reason="TICK_ABSTAIN_NO_TICKS")
        idle = (now - recvs[i - 1]).total_seconds()
        if idle > self.stale_s:
            return TickTriggerResult(False, reason=f"TICK_ABSTAIN_FEED_STALE_{idle:.0f}s")
        result = TickTriggerResult(False)
        for n in self.sizes:
            c = tick_ma_golden_cross(builders[n].bars, now, n, self.fast, self.slow, self.max_age_s)
            result.crosses[n] = c
            if c.ok and not result.ok:
                result.ok, result.label, result.cross_time = True, f"TICK{n}_MA{self.fast}_MA{self.slow}", c.cross_time
        result.reason = "|".join(c.reason for c in result.crosses.values())
        return result
