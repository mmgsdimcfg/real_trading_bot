# -*- coding: utf-8 -*-
"""R025 exit tick features - rolling per-position windows over KIS H0STCNT0 prints -> r024 MarketSnapshot.

Pure logic: ticks come in as r009 `Tick` objects (live: the r013-style incremental file reader; replay: r009
iter_tick_file), REST fields come in as `RestQuote`, the clock is an argument. Same code live and in g032 replay.

Availability: a print is STAGED by ingest() and becomes visible only when snapshot(now) is called with
now >= its receive time (Tick.recv, else its exchange time). Everything - price, peak (high_since_prev), quotes,
last receive time, 1-minute bars / ATR, windows - is built from visible prints only (code review 2026-11-15 #3).

Integrity (same rule as r009): the cumulative volume (acml_vol) orders the prints.
- acml_vol <= last accepted        -> duplicate / late print, dropped (its volume is already inside a later delta)
- acml_vol delta > this print's qty -> missed prints; the difference is GAP volume with UNKNOWN side
- side not in (1 buy, 5 sell)       -> observed volume with UNKNOWN side
Neither kind is ever assigned to buyers or sellers; both lower the window quality (section 17 C11, review #7).

Features (section 2.1 / 4.3 / 17 C10-C12)
- ret_3m_pct           last price vs the last price at or before now-180 s (None until 180 s of coverage)
- atr1m_pct            simple ATR over the last <=14 COMPLETED 1-minute bars built from the prints (None below 6 bars)
- cttr_60s / _prev30   buyer-initiated qty / seller-initiated qty x 100 over (now-60, now] and (now-90, now-30];
                       None with fewer than 5 side-known prints; capped at 200 (seller qty 0 -> 200)
- sell_share_30s       seller-initiated share of side-known qty over 30 s
- vol_30s_over_10m     30 s volume / average 30 s volume over the covered part of the last 10 min (>= 120 s needed)
- gap_volume_ratio / unknown_side_ratio over 60 s; window_coverage = covered seconds of the 60 s window / 60
- rvol_5m              5 min volume / RvolProfile expected volume for the same clock window

Update log (append only):
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 적응형 청산 엔진(r024) 입력 특징. 아직 어떤 실행 경로도 import하지 않음.
    impact: none
    compatibility: backward-compatible
- [2026-11-15] type=fix owner=claude
    summary: Codex 코드 검토 반영 - 수신 전 틱 미노출(staging), 방향 불명 거래량 분리, RVOL 최근 20일 제한.
    impact: none (미연결)
    compatibility: backward-compatible
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from r009_tick_bars import Tick
from r024_adaptive_exit import MarketSnapshot

KEEP_SECONDS = 690.0          # 10 min window + 90 s for the prev30 trade-strength window
SIDE_BUY, SIDE_SELL = 1, 5
MIN_SIDE_PRINTS = 5


@dataclass(frozen=True)
class RestQuote:
    ts: datetime                      # when the quote was received
    price: float
    bid1: float = 0.0
    ask1: float = 0.0
    vi_active: bool = False
    vi_expected_price: float | None = None
    halted: bool = False
    upper_limit: float | None = None
    lower_limit: float | None = None
    obi: float | None = None          # (bid_total - ask_total) / (bid_total + ask_total)


@dataclass
class _Print:
    ts: datetime
    avail: datetime
    price: float
    qty: int
    side: int
    gap: int
    bid1: float
    ask1: float
    acml: int = 0

    @property
    def key(self) -> tuple[datetime, int]:
        """Exchange order: execution second, then cumulative volume (strictly increasing for accepted prints)."""
        return (self.ts, self.acml)


class RvolProfile:
    """Expected volume per minute of day (key = minute index of the minute in which trades happened, 0..1439).

    build(days): `days` is ordered OLDEST -> NEWEST (one {minute: volume} dict per trading day). Empty days (halted /
    no file) are dropped first, then only the most recent `max_days` remain (section 17 C12: 20 trading days).
             >= min_days days -> this stock's own average (source "slot");
             >= min_days_fallback days and a market shape -> market shape x this stock's average daily volume
             (source "market_shape", keeps the intraday U shape); otherwise None (no RVOL -> SURGE cannot be
             confirmed without OBI - conservative)."""

    def __init__(self, per_minute: dict[int, float], source: str, days: int):
        self.per_minute = per_minute
        self.source = source
        self.days = days

    @classmethod
    def build(cls, days: list[dict[int, float]], market_shape: dict[int, float] | None = None,
              min_days: int = 10, min_days_fallback: int = 5, max_days: int = 20) -> "RvolProfile | None":
        days = [d for d in days if d and sum(d.values()) > 0][-max_days:]
        n = len(days)
        if n >= min_days:
            keys = set().union(*days)
            per = {k: sum(d.get(k, 0.0) for d in days) / n for k in keys}   # a missing minute on a trading day = 0
            return cls(per, "slot", n)
        if n >= min_days_fallback and market_shape:
            total = sum(market_shape.values())
            if total <= 0:
                return None
            avg_daily = sum(sum(d.values()) for d in days) / n
            return cls({k: v / total * avg_daily for k, v in market_shape.items()}, "market_shape", n)
        return None

    def expected(self, now: datetime, seconds: float = 300.0) -> float | None:
        """Expected volume in (now - seconds, now], minutes weighted by overlap."""
        end = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
        start = end - seconds
        total = 0.0
        m = int(start // 60)
        while m * 60 < end:
            lo, hi = max(start, m * 60), min(end, m * 60 + 60)
            if hi > lo:
                total += self.per_minute.get(m, 0.0) * (hi - lo) / 60.0
            m += 1
        return total if total > 0 else None


def minute_volumes_from_bars(rows: Iterable[tuple[datetime, float]], label: str = "right") -> dict[int, float]:
    """(bar_time, volume) of 1-minute bars -> {minute index: volume}. label="right": bar_time is the END of the
    minute (KIS / r003 convention, 09:01 = trades of 09:00:00-09:00:59)."""
    out: dict[int, float] = {}
    for t, v in rows:
        if v is None or not math.isfinite(float(v)):
            continue
        k = t.hour * 60 + t.minute - (1 if label == "right" else 0)
        out[k] = out.get(k, 0.0) + float(v)
    return out


def _insort(lst: list, item, keyfn) -> None:
    """Insert keeping lst sorted by keyfn (stable, after equal keys). Python-version independent."""
    k, lo, hi = keyfn(item), 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if keyfn(lst[mid]) <= k:
            lo = mid + 1
        else:
            hi = mid
    lst.insert(lo, item)


class ExitTickWindow:
    """Staging is ordered by receive time (binary insert), visible prints by exchange order (execution second, acml).

    Input contract: feed prints roughly as they are received (live: file tail; replay: interleave ingest with
    snapshot in time). MAX_STAGED bounds the staging list; beyond it the EARLIEST-available prints are promoted
    immediately (counted in `staged_overflow`) so memory stays bounded even for a mis-fed replay."""

    MAX_STAGED = 50000

    def __init__(self, code: str, tick_stale_s: float = 15.0):
        self.code = code
        self.tick_stale_s = tick_stale_s
        self._staged: list[_Print] = []                  # sorted by (avail, acml)
        self._prints: list[_Print] = []                  # visible, sorted by key (exchange order)
        self._last_acml: int | None = None
        self.first_ts: datetime | None = None            # earliest visible execution time (trade-time history)
        self.first_avail: datetime | None = None         # first receive time (observation start -> coverage)
        self.last_avail: datetime | None = None
        self._bars: dict[datetime, list] = {}            # minute -> [o, h, l, c, first_key, last_key]
        self._high_since_snap: float | None = None
        # peak tracking only counts executions at/after this time (the position's entry): the day file is read from
        # the start, and pre-entry highs must never become the position's peak (g032 replay finding 2026-11-15)
        self.peak_from: datetime | None = None
        self.dropped = 0
        self.staged_overflow = 0
        self.last_bid1 = 0.0
        self.last_ask1 = 0.0

    # -- ingest ------------------------------------------------------------------------------------
    def ingest(self, tick: Tick) -> bool:
        acml = int(tick.acml_vol)
        if self._last_acml is not None and acml <= self._last_acml:
            self.dropped += 1
            return False
        qty = max(0, int(tick.qty))
        gap = 0 if self._last_acml is None else max(0, (acml - self._last_acml) - qty)
        self._last_acml = acml
        p = _Print(tick.ts, tick.recv or tick.ts, float(tick.price), qty, int(tick.side), gap,
                   float(tick.bid1 or 0.0), float(tick.ask1 or 0.0), acml)
        _insort(self._staged, p, lambda x: (x.avail, x.acml))
        if len(self._staged) > self.MAX_STAGED:
            self.staged_overflow += 1
            self._make_visible([self._staged.pop(0)])
        return True

    def _promote(self, now: datetime) -> None:
        n = 0
        while n < len(self._staged) and self._staged[n].avail <= now:
            n += 1
        if n:
            ready, self._staged = self._staged[:n], self._staged[n:]
            self._make_visible(ready)

    def _make_visible(self, ready: list[_Print]) -> None:
        for p in ready:
            newest = not self._prints or p.key > self._prints[-1].key
            if newest:
                self._prints.append(p)
            else:                                # received late: slot it into exchange order (re-review #1)
                _insort(self._prints, p, lambda x: x.key)
            if self.first_ts is None or p.ts < self.first_ts:
                self.first_ts = p.ts
            if self.first_avail is None:
                self.first_avail = p.avail
            self.last_avail = p.avail if self.last_avail is None else max(self.last_avail, p.avail)
            if newest:                           # quotes / last price only from the newest execution
                if p.bid1:
                    self.last_bid1 = p.bid1
                if p.ask1:
                    self.last_ask1 = p.ask1
            if self.peak_from is None or p.ts >= self.peak_from:
                self._high_since_snap = p.price if self._high_since_snap is None else max(self._high_since_snap, p.price)
            mk = p.ts.replace(second=0, microsecond=0)
            bar = self._bars.get(mk)
            if bar is None:
                self._bars[mk] = [p.price, p.price, p.price, p.price, p.key, p.key]
                for old in sorted(self._bars)[:-20]:
                    del self._bars[old]
            else:
                bar[1], bar[2] = max(bar[1], p.price), min(bar[2], p.price)
                if p.key < bar[4]:
                    bar[0], bar[4] = p.price, p.key
                if p.key > bar[5]:
                    bar[3], bar[5] = p.price, p.key
        if self._prints:
            cutoff = self._prints[-1].ts - timedelta(seconds=KEEP_SECONDS)
            i = 0
            while i < len(self._prints) and self._prints[i].ts < cutoff:
                i += 1
            if i:
                del self._prints[:i]

    # -- features ----------------------------------------------------------------------------------
    def _window(self, now: datetime, start_s: float, end_s: float = 0.0) -> list[_Print]:
        lo, hi = now - timedelta(seconds=start_s), now - timedelta(seconds=end_s)
        return [p for p in self._prints if lo < p.ts <= hi]

    def _covered(self, now: datetime) -> float:
        """Trade-time history available (seconds since the earliest visible execution)."""
        return 0.0 if self.first_ts is None else max(0.0, (now - self.first_ts).total_seconds())

    def trades_after(self, ts: datetime) -> list[tuple[datetime, float]]:
        """Visible executions strictly AFTER ts (exchange time), in exchange order - virtual order fills must only use
        prints that happened after the order was placed (r026, wiring review #5/#6)."""
        return [(p.ts, p.price) for p in self._prints if p.ts > ts]

    def _observed(self, now: datetime) -> float:
        """Seconds this window has actually been receiving prints (coverage of the 60 s quality window)."""
        return 0.0 if self.first_avail is None else max(0.0, (now - self.first_avail).total_seconds())

    @staticmethod
    def _cttr(ps: list[_Print]) -> float | None:
        known = [p for p in ps if p.side in (SIDE_BUY, SIDE_SELL)]
        if len(known) < MIN_SIDE_PRINTS:
            return None
        buy = sum(p.qty for p in known if p.side == SIDE_BUY)
        sell = sum(p.qty for p in known if p.side == SIDE_SELL)
        if sell <= 0:
            return 200.0 if buy > 0 else None
        return min(200.0, buy / sell * 100.0)

    def atr1m_pct(self, now: datetime) -> float | None:
        cur = now.replace(second=0, microsecond=0)
        keys = [k for k in sorted(self._bars) if k < cur][-15:]
        if len(keys) < 6:
            return None
        trs, prev_c = [], None
        for k in keys:
            o, h, l, c = self._bars[k][:4]
            if prev_c is not None:
                trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
            prev_c = c
        last_c = self._bars[keys[-1]][3]
        trs = trs[-14:]
        return sum(trs) / len(trs) / last_c * 100.0 if trs and last_c > 0 else None

    def snapshot(self, now: datetime, rest: RestQuote | None = None, profile: RvolProfile | None = None,
                 venue: str = "KRX") -> MarketSnapshot:
        self._promote(now)
        last = self._prints[-1] if self._prints else None
        tick_age = (now - self.last_avail).total_seconds() if self.last_avail is not None else 1e9
        rest_age = (now - rest.ts).total_seconds() if rest is not None else 1e9
        if last is not None and (tick_age < self.tick_stale_s or rest is None or tick_age <= rest_age):
            price = last.price
        elif rest is not None:
            price = float(rest.price)
        else:
            price = 0.0
        covered = self._covered(now)
        ret_3m = None
        if last is not None and covered >= 180.0:
            ref = None
            cutoff = now - timedelta(seconds=180)
            for p in self._prints:          # exchange order
                if p.ts <= cutoff:
                    ref = p.price
                else:
                    break
            if ref:
                ret_3m = (last.price / ref - 1.0) * 100.0
        w60, w30 = self._window(now, 60), self._window(now, 30)
        observed60 = sum(p.qty for p in w60)
        gap60 = sum(p.gap for p in w60)
        unknown60 = sum(p.qty for p in w60 if p.side not in (SIDE_BUY, SIDE_SELL))
        total60 = observed60 + gap60
        buy30 = sum(p.qty for p in w30 if p.side == SIDE_BUY)
        sell30 = sum(p.qty for p in w30 if p.side == SIDE_SELL)
        span10 = min(600.0, covered)
        vol_ratio = None
        if span10 >= 120.0:
            vol_span = sum(p.qty + p.gap for p in self._window(now, span10))
            avg30 = vol_span / (span10 / 30.0)
            vol30 = sum(p.qty + p.gap for p in w30)
            vol_ratio = vol30 / avg30 if avg30 > 0 else None
        rvol = None
        if profile is not None and covered >= 300.0:
            exp = profile.expected(now, 300.0)
            if exp:
                rvol = sum(p.qty + p.gap for p in self._window(now, 300)) / exp
        high = self._high_since_snap
        self._high_since_snap = None
        bid = self.last_bid1 or (rest.bid1 if rest else 0.0)
        ask = self.last_ask1 or (rest.ask1 if rest else 0.0)
        if rest is not None and rest_age < tick_age:
            bid, ask = rest.bid1 or bid, rest.ask1 or ask
        return MarketSnapshot(
            ts=now, price=price, bid1=bid, ask1=ask, high_since_prev=high, venue=venue,
            last_tick_age_s=tick_age, rest_age_s=rest_age, ret_3m_pct=ret_3m, atr1m_pct=self.atr1m_pct(now),
            rvol_5m=rvol, cttr_60s=self._cttr(w60), cttr_60s_prev30=self._cttr(self._window(now, 90, 30)),
            sell_share_30s=(sell30 / (buy30 + sell30)) if (buy30 + sell30) > 0 else None,
            vol_30s_over_10m=vol_ratio,
            gap_volume_ratio=gap60 / total60 if total60 > 0 else 0.0,
            unknown_side_ratio=unknown60 / total60 if total60 > 0 else 0.0,
            window_coverage=min(1.0, self._observed(now) / 60.0),
            obi=rest.obi if rest else None,
            vi_active=bool(rest and rest.vi_active),
            vi_expected_price=rest.vi_expected_price if rest else None,
            halted=bool(rest and rest.halted),
            upper_limit=rest.upper_limit if rest else None,
            lower_limit=rest.lower_limit if rest else None,
        )
