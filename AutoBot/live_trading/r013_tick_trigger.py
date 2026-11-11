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
- [2026-11-11] type=feat owner=claude
    summary: tick_ma_golden_cross 워밍업 가드 - 시계열 첫 (slow-1)+CROSS_WARMUP_EXTRA_BARS 봉 안의 크로스는 ABSTAIN_WARMUP_CROSS
      (r001 TICK_CROSS_WARMUP_EXTRA_BARS, r003/g003 가 피드 생성자 warmup_extra_bars 로 넘김). 086450 20261111 09:49 수집기 재시작 뒤 21번째 봉 크로스 매수.
    impact: common
    compatibility: backward-compatible (기본 0 = 종전; r001 값 5 적용 시 매수 감소 방향)
- [2026-11-04] type=feat owner=claude
    summary: (1) TickCross.cross_close(크로스 봉 종가) - r002 evaluate_triple_trigger가 가장 최근 통과 크로스의 종가를
      cross_ref_price로 넘겨 r005/g003 추격폭 가드에 쓴다. (2) 틱 밀도 하한 - prints_per_minute/density_abstain: 최근
      N분(TICK_MIN_PRINTS_WINDOW_MINUTES) 체결/분 < TICK_MIN_PRINTS_PER_MIN이면 TICK_LOW_DENSITY 기권(live/replay 공용,
      개장 5분 미만 미적용, 전일 워밍업 봉 제외). TickTriggerResult.prints_per_min 기록.
    impact: live + sim
    compatibility: backward-compatible (min_prints_per_min 기본 0 = 비활성; r003/g003은 r001 값 전달)
- [2026-11-02] type=feat owner=claude
    summary: 사용자 요청(20261102 장 초반 60틱 크로스 FEW_BARS 미인식) - prior_day_warmup: 직전 수집일 파일(find_prior_tick_file)로
      틱봉을 먼저 채워 HTS처럼 전일 봉에 이어 MA 계산(tick_ma_golden_cross same_day_only=False). r001 TICK_TRIGGER_PRIOR_DAY_WARMUP.
      일자 경계의 무효 봉(전일 마지막 미완성 봉 / 당일 첫 봉 = 동시호가 체결)은 _drop_session_edge_bars로 제외.
    impact: live (r003) + sim (g003)
    compatibility: breaking (장 초반 틱 크로스 판정 가능 -> 매수 증가 방향. 롤백: TICK_TRIGGER_PRIOR_DAY_WARMUP=False)
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
from datetime import datetime, timedelta
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
    cross_close: float = float("nan")  # [2026-11-04] close of the cross bar (chase-from-cross guard reference)


@dataclass
class TickTriggerResult:
    ok: bool
    label: str = ""                     # e.g. "TICK60_MA5_MA20" of the passing cross ("" if none)
    reason: str = ""                    # per-size reasons joined by "|"
    cross_time: datetime | None = None
    crosses: dict[int, TickCross] = field(default_factory=dict)
    prints_per_min: float = float("nan")  # [2026-11-04] recent print density (nan = not enough session elapsed)


def _drop_session_edge_bars(bars: list[TickBar], today) -> list[TickBar]:
    """Prior-day warmup: drop INVALID bars that exist only because of a session edge - the first bar of each day
    (collection started mid-bar / opening-auction print without counts) and the last bar of an earlier day (unfinished
    at the close / closing-auction print). Without this every cross in the first ~slow bars of the day abstains on
    INVALID_BAR. Invalid bars inside a session are kept (a real gap still abstains)."""
    out: list[TickBar] = []
    for i, b in enumerate(bars):
        if b.invalid:
            d = b.end_ts.date()
            first_of_day = i == 0 or bars[i - 1].end_ts.date() != d
            last_of_prior_day = d < today and (i == len(bars) - 1 or bars[i + 1].end_ts.date() != d)
            if first_of_day or last_of_prior_day:
                continue
        out.append(b)
    return out


# [2026-11-11] crosses inside the first (slow-1)+CROSS_WARMUP_EXTRA_BARS bars of a bar series are ignored (r001
# TICK_CROSS_WARMUP_EXTRA_BARS, passed by r003/g003 to the feed constructors): right after a collector (re)start MA_slow comes out of its very
# first window and the first "cross" there is an artifact (086450 2026-11-11: cross on bar 21 at 09:49 -> buy 09:50 -> -1.09%)
CROSS_WARMUP_EXTRA_BARS = 0


def tick_ma_golden_cross(bars: list[TickBar], now: datetime, n: int, fast: int, slow: int,
                         max_age_s: float, same_day_only: bool = True, warmup_extra_bars: int | None = None) -> TickCross:
    tag = f"TICK{n}"
    extra = int(CROSS_WARMUP_EXTRA_BARS if warmup_extra_bars is None else warmup_extra_bars)
    # same_day_only=False: prior-day bars loaded by the prior-day warmup also feed the MAs (HTS-like continuity)
    recent = [b for b in bars if b.available_at <= now and (not same_day_only or b.end_ts.date() == now.date())]
    if not same_day_only:
        recent = _drop_session_edge_bars(recent, now.date())
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
    if extra > 0 and i < first + extra:
        return TickCross(False, f"{tag}_ABSTAIN_WARMUP_CROSS_{i}_LT_{first + extra}", n, cross_bar.available_at, f_last, s_last,
                         len(recent))
    age = (now - cross_bar.available_at).total_seconds()
    if age > max_age_s:
        return TickCross(False, f"{tag}_CROSS_EXPIRED_{age:.0f}s", n, cross_bar.available_at, f_last, s_last, len(recent))
    # MA_slow at i-1 uses bars i-slow .. i-1, so every bar from i-slow on decides this cross
    if any(b.invalid for b in recent[i - slow:]):
        return TickCross(False, f"{tag}_ABSTAIN_INVALID_BAR", n, cross_bar.available_at, f_last, s_last, len(recent))
    return TickCross(True, f"{tag}_MA{fast}_MA{slow}_GOLDEN_CROSS", n, cross_bar.available_at, f_last, s_last, len(recent),
                     cross_close=float(cross_bar.close))


_DENSITY_MIN_ELAPSED_MIN = 5.0


def prints_per_minute(bars: list[TickBar], now: datetime, window_min: float) -> float:
    """[2026-11-04] Print density over the last `window_min` minutes of today's session (prints counted from completed
    bars, incl. missed prints). nan when less than _DENSITY_MIN_ELAPSED_MIN minutes of today's bars exist (session just
    opened) so the floor never blocks the first minutes. Shared by the live feed and the g003 replay feed."""
    today = now.date()
    todays = [b for b in bars if b.end_ts.date() == today and b.available_at <= now]
    if not todays:
        return float("nan")
    first = min(b.start_ts for b in todays)
    # [2026-11-04 Codex] 창의 끝은 now가 아니라 마지막 완성 봉의 종료시각 - 진행 중인 봉 구간을 분모에서 빼야 봉 완성 시점에
    # 따라 밀도가 출렁이지 않는다(정확히 15체결/분 입력이면 항상 15).
    last_end = max(b.end_ts for b in todays)
    elapsed = min(float(window_min), (last_end - first).total_seconds() / 60.0)
    if elapsed < _DENSITY_MIN_ELAPSED_MIN:
        return float("nan")
    cutoff = last_end - timedelta(minutes=elapsed)
    # [2026-11-04 Codex] 창 경계에 걸친 봉은 겹치는 시간 비율만큼만 더한다 - 봉 완성 시점에 따라 14/16체결/분으로 흔들리던
    # 양자화 제거(정확히 15체결/분 입력이면 항상 15).
    prints = 0.0
    for b in todays:
        if b.end_ts <= cutoff:
            continue
        span = (b.end_ts - b.start_ts).total_seconds()
        if span <= 0 or b.start_ts >= cutoff:
            prints += float(b.n_ticks)
        else:
            prints += float(b.n_ticks) * (b.end_ts - cutoff).total_seconds() / span
    return prints / elapsed if elapsed > 0 else float("nan")


def density_abstain(builders: dict, now: datetime, floor: float, window_min: float) -> TickTriggerResult | None:
    """TickTriggerResult(False, TICK_LOW_DENSITY_...) when the recent print density is below `floor`, else None."""
    if floor <= 0 or not builders:
        return None
    d = prints_per_minute(builders[min(builders)].bars, now, window_min)
    if d != d:  # nan - not enough session elapsed
        return None
    if d < floor:
        return TickTriggerResult(False, reason=f"TICK_LOW_DENSITY_{d:.0f}/min_LT_{floor:.0f}", prints_per_min=d)
    return None


def find_prior_tick_file(data_dir: Path, day: str, code: str) -> Path | None:
    """<data_dir>/<D>/<code>.csv of the latest collection day D < day (None if that day has no file for the code)."""
    try:
        days = sorted((p.name for p in Path(data_dir).iterdir()
                       if p.is_dir() and len(p.name) == 8 and p.name.isdigit() and p.name < day), reverse=True)
    except OSError:
        return None
    if not days:
        return None
    path = Path(data_dir) / days[0] / f"{code}.csv"
    return path if path.exists() else None


def iter_prior_tail(path: Path, max_ticks: int):
    """Last max_ticks prints of a prior-day g011 file (file order). Only the tail is csv-parsed so a large file does not
    block the trading loop; a bar cut by the tail start is invalid + first of its day -> dropped by the edge rule."""
    from collections import deque
    with open(path, "r", newline="", encoding="utf-8") as f:
        header_line = f.readline()
        tail = deque(f, maxlen=max(1, int(max_ticks)))
    header = next(csv.reader([header_line]))
    for line in tail:
        fields = next(csv.reader([line]))
        if len(fields) != len(header):
            continue
        tick = parse_tick_row(dict(zip(header, fields)))
        if tick is not None:
            yield tick


def prior_tail_ticks(sizes: tuple[int, ...], slow: int) -> int:
    # enough prints for slow+1 bars of the largest size, twice over (missed prints / edge bar)
    return 2 * (int(slow) + 1) * max(int(n) for n in sizes)


class _CodeFeed:
    """Incremental reader of one g011 tick file + one TickBarBuilder per bar size."""

    def __init__(self, code: str, path: Path, sizes: tuple[int, ...], prior_path: Path | None = None,
                 prior_ticks: int = 0):
        self.code = code
        self.path = path
        self.prior_path = prior_path
        self.prior_ticks = prior_ticks  # prior-day file ingested once before today's prints (bars only, no last_recv)
        self.prior_loaded = False
        self.offset = 0
        self.pending = b""
        self.header: list[str] | None = None
        self.builders = {n: TickBarBuilder(code=code, n=n) for n in sizes}
        self.last_recv: datetime | None = None
        self.file_id: tuple[int, int] | None = None  # (st_dev, st_ino) of the file the state was built from
        self.tail = b""  # last bytes before offset - detects an in-place rewrite of already-read content

    def _reset(self) -> None:
        self.__init__(self.code, self.path, tuple(self.builders), self.prior_path, self.prior_ticks)

    def _load_prior(self) -> None:
        self.prior_loaded = True
        if self.prior_path is None:
            return
        for tick in iter_prior_tail(self.prior_path, self.prior_ticks):
            for b in self.builders.values():
                b.ingest(tick)

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
            if not self.prior_loaded:
                self._load_prior()  # also right after a _reset above: prior-day bars must precede today's prints
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
                 stale_s: float, log: Callable[[str], None] | None = None, prior_day_warmup: bool = False,
                 min_prints_per_min: float = 0.0, density_window_min: float = 30.0, warmup_extra_bars: int = 0):
        self.data_dir = Path(data_dir)
        self.prior_day_warmup = bool(prior_day_warmup)
        self.warmup_extra_bars = int(warmup_extra_bars or 0)  # [2026-11-11] r001 TICK_CROSS_WARMUP_EXTRA_BARS
        self.min_prints_per_min = float(min_prints_per_min)  # [2026-11-04] 0 = no density floor
        self.density_window_min = float(density_window_min)
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
            prior = find_prior_tick_file(self.data_dir, day, code) if self.prior_day_warmup else None
            feed = _CodeFeed(code, self.data_dir / day / f"{code}.csv", self.sizes, prior,
                             prior_tail_ticks(self.sizes, self.slow))
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
        low = density_abstain(feed.builders, now, self.min_prints_per_min, self.density_window_min)
        if low is not None:
            return low
        result = TickTriggerResult(False)
        result.prints_per_min = prints_per_minute(feed.builders[min(feed.builders)].bars, now, self.density_window_min)
        for n in self.sizes:
            c = tick_ma_golden_cross(feed.builders[n].bars, now, n, self.fast, self.slow, self.max_age_s,
                                     same_day_only=not self.prior_day_warmup, warmup_extra_bars=self.warmup_extra_bars)
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
                 stale_s: float, prior_day_warmup: bool = False, min_prints_per_min: float = 0.0, warmup_extra_bars: int = 0,
                 density_window_min: float = 30.0):
        self.data_dir = Path(data_dir)
        self.prior_day_warmup = bool(prior_day_warmup)
        self.warmup_extra_bars = int(warmup_extra_bars or 0)  # [2026-11-11] r001 TICK_CROSS_WARMUP_EXTRA_BARS
        self.min_prints_per_min = float(min_prints_per_min)
        self.density_window_min = float(density_window_min)
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
                prior = find_prior_tick_file(self.data_dir, day, code) if self.prior_day_warmup else None
                if prior is not None:  # same order as the live feed: prior-day bars first, no receive times
                    for tick in iter_prior_tail(prior, prior_tail_ticks(self.sizes, self.slow)):
                        for b in builders.values():
                            b.ingest(tick)
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
        low = density_abstain(builders, now, self.min_prints_per_min, self.density_window_min)
        if low is not None:
            return low
        result = TickTriggerResult(False)
        result.prints_per_min = prints_per_minute(builders[min(builders)].bars, now, self.density_window_min)
        for n in self.sizes:
            c = tick_ma_golden_cross(builders[n].bars, now, n, self.fast, self.slow, self.max_age_s,
                                     same_day_only=not self.prior_day_warmup, warmup_extra_bars=self.warmup_extra_bars)
            result.crosses[n] = c
            if c.ok and not result.ok:
                result.ok, result.label, result.cross_time = True, f"TICK{n}_MA{self.fast}_MA{self.slow}", c.cross_time
        result.reason = "|".join(c.reason for c in result.crosses.values())
        return result
