# -*- coding: utf-8 -*-
"""N-tick bars (60/120 tick) from KIS real-time executions + tick momentum filter (pure logic, no I/O to broker).

Source: KIS websocket H0STCNT0 (KRX real-time execution). One record = one execution print with
  STCK_CNTG_HOUR (HHMMSS, 1-second resolution), STCK_PRPR (price), CNTG_VOL (this print's quantity),
  ACML_VOL (session cumulative volume), CCLD_DVSN (1=buyer-initiated, 5=seller-initiated, 3=pre-open),
  SELN_CNTG_CSNU / SHNU_CNTG_CSNU (session cumulative seller/buyer-initiated execution COUNTS).
There is no per-print sequence number, so integrity is derived from the cumulative fields:
  - ACML_VOL equal to the last accepted value        -> duplicate (dropped)
  - ACML_VOL lower than the last accepted value      -> late / out-of-order (dropped; its volume is already
                                                         inside the next accepted delta)
  - ACML_VOL delta > CNTG_VOL                        -> prints were missed (volume gap = delta - CNTG_VOL)
  - (SELN+SHNU) count delta > 1                      -> number of missed prints (exact when counts present)
Tick count vs volume: a "tick" is one execution print (count), CNTG_VOL is shares. 120-tick bars close every
120 prints regardless of size. Missed prints are still counted (count_missing=True) so bar boundaries stay
aligned with the exchange tape; such bars carry gap_ticks > 0 and the momentum filter abstains on them.

Update log:
- [2026-10-23] type=feat owner=claude
    summary: 신규 - 60/120틱 봉 생성기(누락/중복/순서역전 처리) + 틱 모멘텀 보조 필터 + 수집 파일 로더.
    impact: sim(g012 strategy B/C) / collector(g011)
    compatibility: backward-compatible (신규 모듈)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

TICK_FILE_COLUMNS = [
    "recv_ts", "code", "bsop_date", "trade_time", "price", "cntg_vol", "acml_vol", "ccld_dvsn",
    "seln_cntg_csnu", "shnu_cntg_csnu", "askp1", "bidp1", "hour_cls_code",
]


@dataclass
class Tick:
    code: str
    ts: datetime           # exchange execution time (second resolution)
    price: float
    qty: int
    acml_vol: int
    side: int              # 1 buy-initiated, 5 sell-initiated, other = unknown/auction
    cum_count: int | None  # seln_cntg_csnu + shnu_cntg_csnu (None if not provided)
    ask1: float = 0.0
    bid1: float = 0.0
    recv: datetime | None = None  # local receive time; availability for decisions (late prints are not usable earlier)


@dataclass
class TickBar:
    code: str
    start_ts: datetime
    end_ts: datetime
    open: float
    high: float
    low: float
    close: float
    n_ticks: int = 0        # prints counted toward the bar (incl. missed ones)
    gap_ticks: int = 0      # missed prints inside this bar
    volume: int = 0         # shares (incl. gap volume)
    buy_vol: int = 0        # buyer-initiated shares (received prints only)
    sell_vol: int = 0
    invalid: bool = False   # boundary crossed by missed prints: OHLC / boundary position not reliable
    unattributed_volume: int = 0  # missed volume that spans several bars (cannot be assigned to one)
    end_recv: datetime | None = None  # receive time of the bar's last print
    closed_at: datetime | None = None  # receive (else exchange) time of the print that completed the bar

    @property
    def available_at(self) -> datetime:
        return self.closed_at or self.end_recv or self.end_ts


@dataclass
class TickStats:
    accepted: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    gap_events: int = 0
    gap_ticks: int = 0
    gap_volume: int = 0
    inconsistent: int = 0   # prints whose cumulative-volume delta was smaller than their own qty


@dataclass
class TickBarBuilder:
    """N-tick bar builder.

    Alignment
    - ABSOLUTE mode (count fields present and behaving as session-cumulative counters): print number
      k = SELN_CNTG_CSNU + SHNU_CNTG_CSNU, bar id = (k-1)//n. Boundaries sit on exchange print numbers (like an
      HTS tick chart counted from the session start), so missed prints never shift later boundaries.
    - RELATIVE mode (no usable counts): bars count received prints from the first print. A volume gap of
      unknown print count makes alignment UNKNOWN; every bar touched while unknown is invalid until the next
      counted print re-synchronises to absolute mode.
    Invalid bars (unknown start/close/boundary) are kept for bookkeeping but never vote in tick_momentum.
    """

    code: str
    n: int
    count_missing: bool = True   # kept for API compatibility; absolute mode always accounts for missed prints
    count_reliable: bool = True
    bars: list[TickBar] = field(default_factory=list)
    stats: TickStats = field(default_factory=TickStats)
    _cur: TickBar | None = None
    _cur_id: int | None = None   # absolute bar id of _cur (None in relative mode)
    _last_acml: int | None = None
    _last_k: int | None = None   # last absolute print number
    _unaligned: bool = False
    _session: str | None = None

    def ingest(self, t: Tick) -> list[TickBar]:
        """Feed one print. Returns bars completed by this print (0 or more)."""
        session = t.ts.strftime("%Y%m%d")
        if self._session is not None and session < self._session:
            self.stats.out_of_order += 1  # late print from a previous session: never reopen it
            return []
        if session != self._session:
            self._session = session
            self._last_acml = None
            self._last_k = None
            self._cur = None
            self._cur_id = None
            self._unaligned = False
            self.count_reliable = True  # reliability is judged per session
        if t.price <= 0:
            return []
        qty = max(t.qty, 0)
        gap_vol = 0
        exact = False        # cumulative volume advanced by exactly this print's qty
        inconsistent = False  # cumulative volume advanced by LESS than this print's qty (bad data)
        if self._last_acml is not None:
            if t.acml_vol == self._last_acml:
                self.stats.duplicates += 1
                return []
            if t.acml_vol < self._last_acml:
                self.stats.out_of_order += 1
                return []
            delta = t.acml_vol - self._last_acml
            exact = delta == qty
            inconsistent = delta < qty
            gap_vol = max(0, delta - qty)
            if inconsistent:
                self.stats.inconsistent += 1
        self._last_acml = t.acml_vol
        self.stats.accepted += 1

        k = t.cum_count
        if k is None and self.count_reliable and self._last_k is not None and exact:
            # Count fields missing on this print, but cumulative volume proves nothing was missed: it is print k+1.
            k = self._last_k + 1
        if k is not None and self._last_k is not None and k <= self._last_k:
            # Volume advanced but the cumulative count did not: counts are not session-cumulative for this feed.
            self.count_reliable = False
        if self.count_reliable and k is not None and k > 0:
            return self._ingest_absolute(t, k, qty, gap_vol, inconsistent)
        return self._ingest_relative(t, qty, gap_vol, inconsistent)

    # ---- absolute (exchange print number) alignment ----------------------------------------
    def _ingest_absolute(self, t: Tick, k: int, qty: int, gap_vol: int, inconsistent: bool) -> list[TickBar]:
        done: list[TickBar] = []
        n = self.n
        bar_id = (k - 1) // n
        pos = (k - 1) % n + 1  # 1..n
        missed = k - self._last_k - 1 if self._last_k is not None else None
        # Count and volume must agree: count says nothing missed but volume jumped, or count says prints were
        # missed but no volume is missing -> contradictory data, the receiving bar cannot be trusted.
        if missed is not None and ((missed == 0 and gap_vol > 0) or (missed > 0 and gap_vol == 0)):
            self.stats.inconsistent += 1
            inconsistent = True
        if missed or gap_vol:
            self.stats.gap_events += 1
            self.stats.gap_ticks += missed or 0
            self.stats.gap_volume += gap_vol
        same_bar = self._cur is not None and self._cur_id == bar_id and not self._unaligned and missed is not None
        if same_bar:
            self._add_print(self._cur, t, qty + gap_vol, missed, inconsistent)
            self._cur.n_ticks = pos
        else:
            prev = self._cur
            prev_id = self._cur_id
            # Id of the bar that held the last counted print (known even if that bar already closed).
            base_id = prev_id if prev is not None else ((self._last_k - 1) // n if self._last_k is not None else None)
            if prev is not None:
                tail_missing = prev.n_ticks < n
                prev.invalid = prev.invalid or tail_missing or prev_id is None or self._unaligned
                if tail_missing and prev_id is not None:
                    prev.gap_ticks += n - prev.n_ticks
                    prev.n_ticks = n
                done.append(self._close(prev, t))
            placeholders: dict[int, TickBar] = {}
            if base_id is not None and not self._unaligned:
                ref = prev.close if prev is not None else t.price
                for pid in range(base_id + 1, bar_id):  # whole bars made only of missed prints
                    ph = TickBar(self.code, t.ts, t.ts, ref, ref, ref, ref, n_ticks=n, gap_ticks=n, invalid=True)
                    placeholders[pid] = ph
            nb = TickBar(self.code, t.ts, t.ts, t.price, t.price, t.price, t.price,
                         invalid=pos > 1 or (base_id is None and self._last_k is None and pos > 1))
            self._add_print(nb, t, qty, pos - 1, inconsistent)
            nb.n_ticks = pos
            # Attribute missed volume only when all missed prints [last_k+1, k-1] fall inside ONE bar.
            if gap_vol:
                target = None
                if missed and self._last_k is not None:
                    first_id, last_id = self._last_k // n, (k - 2) // n
                    if first_id == last_id:
                        target = prev if (prev is not None and first_id == prev_id) else (
                            nb if first_id == bar_id else placeholders.get(first_id))
                if target is not None:
                    target.volume += gap_vol
                else:
                    nb.unattributed_volume += gap_vol
            for pid in sorted(placeholders):
                done.append(self._close(placeholders[pid], t))
            self._cur = nb
            self._cur_id = bar_id
            self._unaligned = False
        self._last_k = k
        if self._cur is not None and self._cur.n_ticks >= n:
            done.append(self._close(self._cur, t))
        return done

    # ---- relative alignment (no usable counts) ------------------------------------------------
    def _ingest_relative(self, t: Tick, qty: int, gap_vol: int, inconsistent: bool) -> list[TickBar]:
        done: list[TickBar] = []
        self._last_k = None  # released: the next counted print re-synchronises to absolute alignment
        if gap_vol > 0:
            # Prints were missed but how many is unknown: boundary position is unknown from here on.
            self._unaligned = True
            self.stats.gap_events += 1
            self.stats.gap_volume += gap_vol
        if self._cur is not None and self._cur_id is not None:
            # leaving absolute mode mid-bar: that bar can no longer be closed on the right print
            self._cur.invalid = True
            done.append(self._close(self._cur, t))
        if self._cur is None:
            self._cur = TickBar(self.code, t.ts, t.ts, t.price, t.price, t.price, t.price)
            self._cur_id = None
        b = self._cur
        self._add_print(b, t, qty + gap_vol, 0, inconsistent)
        b.n_ticks += 1
        if self._unaligned:
            b.invalid = True
        if b.n_ticks >= self.n:
            done.append(self._close(b, t))
        return done

    def _add_print(self, b: TickBar, t: Tick, vol: int, missed: int, inconsistent: bool = False) -> None:
        b.high = max(b.high, t.price)
        b.low = min(b.low, t.price)
        b.close = t.price
        b.end_ts = t.ts
        b.end_recv = t.recv
        b.gap_ticks += missed
        b.volume += vol
        if inconsistent:
            b.invalid = True
        if t.side == 1:
            b.buy_vol += max(t.qty, 0)
        elif t.side == 5:
            b.sell_vol += max(t.qty, 0)

    def _close(self, b: TickBar, closer: Tick) -> TickBar:
        # A bar is known to be complete only when the closing print is RECEIVED (it may be a later print).
        b.closed_at = closer.recv or closer.ts
        self.bars.append(b)
        if self._cur is b:
            self._cur = None
            self._cur_id = None
        return b

    def bars_until(self, ts: datetime) -> list[TickBar]:
        """Completed bars that were AVAILABLE (received) at or before ts (no look-ahead)."""
        return [b for b in self.bars if b.available_at <= ts]


@dataclass
class TickMomentum:
    ok: bool
    reason: str
    up_bars: int = 0
    buy_ratio: float = float("nan")
    gap_ratio: float = float("nan")
    secs_per_bar: float = float("nan")


def tick_momentum(bars: list[TickBar], now: datetime, lookback: int, min_up_bars: int,
                  min_buy_ratio: float, max_gap_ratio: float, stale_seconds: int) -> TickMomentum:
    """Auxiliary momentum filter over the last `lookback` completed tick bars. Fail-closed for entries:
    insufficient / stale / gappy data returns ok=False with a TICK_ABSTAIN_* reason."""
    # Current session only: yesterday's bars must never vote on today's entry.
    recent = [b for b in bars if b.available_at <= now and b.end_ts.date() == now.date()][-lookback:]
    if len(recent) < lookback:
        return TickMomentum(False, f"TICK_ABSTAIN_FEW_BARS_{len(recent)}")
    if (now - recent[-1].end_ts).total_seconds() > stale_seconds:
        return TickMomentum(False, "TICK_ABSTAIN_STALE")
    if any(b.invalid for b in recent):
        return TickMomentum(False, "TICK_ABSTAIN_INVALID_BAR")
    total_ticks = sum(b.n_ticks for b in recent)
    gap_ratio = sum(b.gap_ticks for b in recent) / total_ticks if total_ticks else 1.0
    if gap_ratio > max_gap_ratio:
        return TickMomentum(False, "TICK_ABSTAIN_GAPS", gap_ratio=gap_ratio)
    up = sum(1 for b in recent if b.close > b.open)
    classified = sum(b.buy_vol + b.sell_vol for b in recent)
    buy_ratio = sum(b.buy_vol for b in recent) / classified if classified else float("nan")
    span = (recent[-1].end_ts - recent[0].start_ts).total_seconds()
    spb = span / len(recent) if recent else float("nan")
    m = TickMomentum(False, "", up, buy_ratio, gap_ratio, spb)
    if up < min_up_bars:
        m.reason = f"TICK_UP_BARS_{up}_LT_{min_up_bars}"
        return m
    if not (buy_ratio >= min_buy_ratio):
        m.reason = f"TICK_BUY_RATIO_{buy_ratio:.2f}_LT_{min_buy_ratio:.2f}"
        return m
    if not (recent[-1].close >= recent[0].open):
        m.reason = "TICK_NET_DOWN"
        return m
    m.ok = True
    m.reason = "OK"
    return m


# ---------------------------------------------------------------------------
# File I/O (collector output: runtime/data/ticks/YYYYMMDD/<code>.csv)
# ---------------------------------------------------------------------------

def _to_int(v: str) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_tick_row(row: dict) -> Tick | None:
    try:
        d = str(row["bsop_date"]).strip()
        hhmmss = str(row["trade_time"]).strip().zfill(6)
        ts = datetime.strptime(d + hhmmss, "%Y%m%d%H%M%S")
    except (KeyError, ValueError):
        return None
    seln, shnu = row.get("seln_cntg_csnu", ""), row.get("shnu_cntg_csnu", "")
    cum = _to_int(seln) + _to_int(shnu) if str(seln).strip() and str(shnu).strip() else None
    return Tick(
        code=str(row.get("code", "")).zfill(6), ts=ts, price=_to_float(row.get("price")),
        qty=_to_int(row.get("cntg_vol")), acml_vol=_to_int(row.get("acml_vol")),
        side=_to_int(row.get("ccld_dvsn")), cum_count=cum,
        ask1=_to_float(row.get("askp1")), bid1=_to_float(row.get("bidp1")),
        recv=_parse_recv(row.get("recv_ts")),
    )


def _parse_recv(v) -> datetime | None:
    try:
        return datetime.fromisoformat(str(v).strip()) if v and str(v).strip() else None
    except ValueError:
        return None


def iter_tick_file(path: Path) -> Iterator[Tick]:
    """Yields ticks in RECEIVE order (file order). The builder handles duplicates / reordering."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = parse_tick_row(row)
            if t is not None:
                yield t


def build_tick_bars(ticks: Iterable[Tick], code: str, n: int) -> TickBarBuilder:
    b = TickBarBuilder(code=code, n=n)
    for t in ticks:
        b.ingest(t)
    return b
