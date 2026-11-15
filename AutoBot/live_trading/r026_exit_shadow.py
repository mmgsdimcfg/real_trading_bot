# -*- coding: utf-8 -*-
"""R026 adaptive exit SHADOW runner - runs the r024 engine on every bot position with VIRTUAL fills (NO orders).

Enabled only when r001 EXIT_ENGINE_MODE == "shadow" (design: adaptive_exit_design_20261115.md sections 17 C3, 20, 21).
Real selling stays with r003 / r006 exactly as before (AUTO_SELL_MODE). The runner answers "what would the adaptive
engine have done with this position", so it keeps managing its virtual copy after the real position is sold.

Two layers
- ShadowExitRunner  synchronous core (tests, g032 replay): observe() / tick(now) / finalize() / persist().
- ShadowExitService what r003 uses: a daemon worker thread owns the core. r003 only puts observations into a BOUNDED
                    queue (put_nowait; dropped and counted when full), so file I/O, tick parsing and engine work never run
                    on the trading thread (wiring review #1). The worker evaluates every SHADOW_EVAL_INTERVAL_S.

Virtual copies are keyed by (code, real entry time): a re-buy after the real position was sold opens a NEW copy while
the old one keeps running (review #4). A closed copy is never reopened (closed_keys, rebuilt from the trades journal on
restore - review #2). When the bot ADDS shares, only the increase is added to the virtual remainder (review #3).

Virtual fills (design section 17 C3, review #5/#6) - an order fills only on data that arrived AFTER it was placed:
- market: first print after the order - 1 tick, else a REST quote received after the order - 1 tick
- limit:  only when a print after the order traded ABOVE the limit; after SHADOW_LIMIT_TO_MARKET_S unfilled it is
          treated as two reprices then market (filled like a market order, on fresh data only)
- halted / no fresh data -> stays unfilled; still open at the session end -> the copy ends as INCOMPLETE (never a fake
  fill at a stale price)

Files (runtime/data/live_runtime)
- exit_engine_YYYYMMDD.jsonl        decision log: events + changes + a 60 s heartbeat per copy; capped per day
                                    (MAX_DECISION_BYTES, events still written), kept DECISION_RETENTION_DAYS days
- exit_shadow_trades_YYYYMMDD.jsonl one record per copy: CLOSED (all sold virtually) or INCOMPLETE (with the reason)
- exit_shadow_YYYYMMDD.json         restart checkpoint, written right after every open / order / fill / close / rebase

Update log:
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 적응형 청산 엔진(r024) shadow 러너. r003에서 EXIT_ENGINE_MODE="shadow"일 때만 호출, 주문 없음.
    impact: live (기록만 - 실제 매도 판정·주문 무변경)
    compatibility: backward-compatible (EXIT_ENGINE_MODE="off"면 import도 하지 않음)
- [2026-11-15] type=fix owner=claude
    summary: Codex 연결 검토(adaptive_exit_codex_wiring_review_20261115.md) 반영 - 작업 스레드+제한 큐(r003 스레드 I/O 제거),
      (종목, 진입 시각) 단위 사본, 추가 매수 증가분만 반영, 주문 이후 데이터로만 가상 체결, 신선한 데이터 없으면 미체결/INCOMPLETE,
      이벤트 직후 체크포인트 + 거래 저널로 복구 중복 방지, 장 마감/날짜 전환 시 미완료 확정, 첫 읽기·기록 용량·보존 기간 상한.
    impact: live (기록만)
    compatibility: backward-compatible
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
from dataclasses import asdict, replace
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Callable

from r009_tick_bars import Tick, parse_tick_row
from r024_adaptive_exit import (
    AIAdjust, ExitConfig, ExitDecisionEngine, ExitIntent, ExitState, MarketSnapshot, pnl_pct, tick_down,
)
from r025_exit_tick_feed import ExitTickWindow, RestQuote, RvolProfile, minute_volumes_from_bars

TAG = "  [EXIT SHADOW ]"
SHADOW_EVAL_INTERVAL_S = 1.0
SHADOW_LIMIT_TO_MARKET_S = 9.0      # 3 s x (1 + 2 reprices), section 7.3
FRESH_REST_S = 15.0                 # a REST quote younger than this (and newer than the order) can fill a market order
HEARTBEAT_S = 60.0
PERSIST_EVERY_S = 30.0
SESSION_END = dt_time(15, 30)
MAX_INITIAL_BYTES = 4 * 1024 * 1024  # first read of a tick file: at most the last 4 MB (+ header)
MAX_CHUNK_BYTES = 4 * 1024 * 1024    # per poll
MAX_DECISION_BYTES = 30 * 1024 * 1024
DECISION_RETENTION_DAYS = 30
QUEUE_MAX = 2000
HALT_VALID_S = 60.0                 # a halt flag older than this (REST no longer refreshed) no longer blocks fills
DROP_LOG_EVERY_S = 600.0
_TAIL_CHECK_BYTES = 256
_VI_OFF = {"", "N", "0", "00", "NONE"}


# ---------------------------------------------------------------------------
# incremental tick file reader (same integrity rules as r013 _CodeFeed.poll, but yields Ticks; bounded reads)
# ---------------------------------------------------------------------------

class TickFileTail:
    def __init__(self, path: Path, max_initial_bytes: int = MAX_INITIAL_BYTES, max_chunk_bytes: int = MAX_CHUNK_BYTES):
        self.path = path
        self.max_initial_bytes = max_initial_bytes
        self.max_chunk_bytes = max_chunk_bytes
        self.resets = -1
        self._reset()

    def _reset(self) -> None:
        self.offset = 0
        self.pending = b""
        self.header: list[str] | None = None
        self.file_id: tuple[int, int] | None = None
        self.tail = b""
        self.skip_partial = False
        self.resets += 1

    def poll(self) -> list[Tick]:
        """New complete rows since the last poll. A replaced / truncated / rewritten file restarts from byte 0
        (the window's acml_vol check drops prints it already has)."""
        try:
            f = open(self.path, "rb")
        except (FileNotFoundError, PermissionError):
            return []
        with f:
            st = os.fstat(f.fileno())
            file_id = (st.st_dev, st.st_ino)
            if self.file_id is not None and file_id != self.file_id:
                self._reset()
            elif st.st_size < self.offset:
                self._reset()
            elif self.tail:
                f.seek(self.offset - len(self.tail))
                if f.read(len(self.tail)) != self.tail:
                    self._reset()
            self.file_id = file_id
            if self.header is None and self.offset == 0:
                f.seek(0)
                first = f.readline()
                if not first.endswith(b"\n"):
                    return []                                   # header not complete yet
                self.header = next(csv.reader([first.decode("utf-8", errors="replace").rstrip("\r\n")]))
                self.offset = len(first)
                if st.st_size - self.offset > self.max_initial_bytes:
                    self.offset = st.st_size - self.max_initial_bytes   # bounded first read (review #1)
                    self.skip_partial = True
                self.tail = b""
            if st.st_size == self.offset:
                return []
            f.seek(self.offset)
            chunk = f.read(min(st.st_size - self.offset, self.max_chunk_bytes))
        self.offset += len(chunk)
        self.tail = (self.tail + chunk)[-_TAIL_CHECK_BYTES:]
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        if self.skip_partial and lines:
            lines.pop(0)                                       # first line after a mid-file seek is partial
            self.skip_partial = False
        out: list[Tick] = []
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            if not line:
                continue
            fields = next(csv.reader([line]))
            if len(fields) != len(self.header or []):
                continue
            t = parse_tick_row(dict(zip(self.header, fields)))
            if t is not None:
                out.append(t)
        return out


# ---------------------------------------------------------------------------
# RVOL profile from g001 10 s files (raw_bar == 1 rows = real KIS minute bars, volume / 6 per 10 s row)
# ---------------------------------------------------------------------------

def _minute_rows_from_10s(path: Path) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("raw_bar", "")).strip() not in ("1", "1.0", "True"):
                continue
            try:
                rows.append((datetime.fromisoformat(str(r["datetime"]).strip()), float(r["volume"])))
            except (KeyError, ValueError, TypeError):
                continue
    if len(rows) >= 2:
        gaps = sorted((b[0] - a[0]).total_seconds() for a, b in zip(rows, rows[1:]))
        med = gaps[len(gaps) // 2]
        expansion = max(1.0, round(med / 10.0)) if med > 0 else 6.0
    else:
        expansion = 6.0
    return [(t, v * expansion) for t, v in rows]


def load_rvol_profile(data_dir: Path, code: str, date_str: str, max_days: int = 20,
                      scan_dirs: int = 40) -> RvolProfile | None:
    """Most recent `max_days` collected days strictly before date_str (section 17 C12). Never raises."""
    try:
        dirs = sorted((p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8
                       and p.name < date_str), reverse=True)[:scan_dirs]
        days: list[dict[int, float]] = []
        for d in dirs:
            files = sorted(d.glob(f"{code}_*_10s.txt"))
            if not files:
                continue
            mv = minute_volumes_from_bars(_minute_rows_from_10s(files[0]), label="right")
            if mv:
                days.append(mv)
            if len(days) >= max_days:
                break
        days.reverse()                                     # oldest -> newest (RvolProfile.build contract)
        return RvolProfile.build(days)
    except Exception:  # noqa: BLE001
        return None


def _entry_key(code: str, entry_ts: datetime) -> str:
    return f"{code}@{entry_ts.isoformat(timespec='seconds')}"


def _parse_ts(value: Any, default: datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# synchronous core
# ---------------------------------------------------------------------------

class _VPos:
    def __init__(self, key: str, state: ExitState, name: str, tick_path: Path, venue: str = "KRX"):
        self.key = key
        self.state = state
        self.name = name
        self.venue = venue
        self.window = ExitTickWindow(state.code)
        self.window.peak_from = state.entry_ts
        self.tail = TickFileTail(tick_path)
        self.rest: RestQuote | None = None
        self.profile: RvolProfile | None = None
        self.profile_loaded = False
        self.order: dict | None = None          # virtual order in flight
        self.fills: list[dict] = []
        self.bought_qty = state.q0
        self.bought_value = state.q0 * state.entry_price
        self.last_bot_qty = state.q0
        self.last_real_qty: int | None = None
        self.last_real_avg: float | None = None
        self.pending_add = 0
        self.pending_add_value = 0.0
        self.last_log_at: datetime | None = None
        self.last_price = 0.0

    def to_dict(self) -> dict:
        return {"key": self.key, "state": self.state.to_dict(), "name": self.name, "venue": self.venue,
                "order": self.order, "fills": self.fills, "bought_qty": self.bought_qty,
                "bought_value": self.bought_value, "last_bot_qty": self.last_bot_qty,
                "last_real_qty": self.last_real_qty, "last_real_avg": self.last_real_avg,
                "pending_add": self.pending_add, "pending_add_value": self.pending_add_value}


class ShadowExitRunner:
    def __init__(self, tick_dir: Path, data_dir: Path, out_dir: Path, date_str: str, log: Callable[[str], None],
                 cfg: ExitConfig | None = None, cleanup: bool = True):
        self.tick_dir = Path(tick_dir)
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.date_str = date_str
        self.log = log
        base = cfg or ExitConfig()
        # no exit model exists yet (design section 6 / Q10): AI stays off in shadow until g034 promotes one
        self.cfg = ExitConfig.from_mapping({**asdict(base), "ai_mode": "off"})
        self.engine = ExitDecisionEngine(self.cfg)
        # results are always reported NET, with the run's own fee / tax (re-review #7)
        self._net_cfg = ExitConfig(pnl_basis="net", fee_pct=self.cfg.fee_pct, tax_pct=self.cfg.tax_pct)
        self.positions: dict[str, _VPos] = {}
        self.closed_keys: set[str] = set()
        # code -> key of the copy opened for a real position WITHOUT an entry time (account-restored): reused until that
        # copy ends, so repeated observations never open duplicates (re-review #1)
        self.noentry_keys: dict[str, str] = {}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.out_dir / f"exit_engine_{date_str}.jsonl"
        self.trades_path = self.out_dir / f"exit_shadow_trades_{date_str}.jsonl"
        self.state_path = self.out_dir / f"exit_shadow_{date_str}.json"
        self._decision_bytes = self.decisions_path.stat().st_size if self.decisions_path.exists() else 0
        self._decision_cap_logged = False
        self._last_persist: datetime | None = None
        self._dirty = False
        self.errors = 0
        if cleanup:
            self._cleanup_old()
        self._restore()

    # -- public ----------------------------------------------------------------------------------------
    def observe(self, code: str, name: str, now: datetime, pos: dict | None, bot_qty: int | None, price: float,
                quote_status: dict | None = None, venue: str = "KRX", price_ts: datetime | None = None) -> None:
        try:
            self._observe(code, name, now, pos, bot_qty, price, quote_status, venue, price_ts or now)
        except Exception as exc:  # noqa: BLE001
            self._error(f"observe {code}", exc)

    def tick(self, now: datetime) -> None:
        try:
            for key in list(self.positions):
                try:
                    self._step(key, now)
                except Exception as exc:  # noqa: BLE001
                    self._error(f"step {key}", exc)
            if now.time() >= SESSION_END and self.positions:
                self.finalize(now, "SESSION_END")
            if self._dirty and (self._last_persist is None
                                or (now - self._last_persist).total_seconds() >= PERSIST_EVERY_S):
                self.persist(now)
        except Exception as exc:  # noqa: BLE001
            self._error("tick", exc)

    def finalize(self, now: datetime, reason: str) -> None:
        """Ends every open copy as INCOMPLETE (no fake fill). Used at the session end and on a date change."""
        for key in list(self.positions):
            try:
                self._close(self.positions[key], now, status="INCOMPLETE", reason=reason)
            except Exception as exc:  # noqa: BLE001
                self._error(f"finalize {key}", exc)
        self.persist(now)

    # -- internals -------------------------------------------------------------------------------------
    def _error(self, where: str, exc: Exception) -> None:
        self.errors += 1
        if self.errors <= 20 or self.errors % 100 == 0:
            self.log(f"{TAG} error ({self.errors}) {where}: {type(exc).__name__}: {exc}")

    def _observe(self, code, name, now, pos, bot_qty, price, quote_status, venue, price_ts) -> None:
        code = str(code).zfill(6)
        qs = quote_status or {}
        vi = str(qs.get("vi_cls_code", "") or "").strip().upper() not in _VI_OFF
        halted = str(qs.get("temp_stop_yn", "") or "").strip().upper() == "Y"
        rest = RestQuote(ts=price_ts, price=float(price), vi_active=vi, halted=halted)
        for vp in self.positions.values():                     # REST quote belongs to the symbol: every copy
            if vp.state.code == code:
                vp.rest = rest
                vp.venue = venue
        if not pos:
            return
        real_qty = int(pos.get("quantity", 0) or 0)
        qty = int(bot_qty) if bot_qty is not None else real_qty
        qty = min(qty, real_qty) if real_qty > 0 else qty
        real_avg = float(pos.get("buy_price") or 0)
        if qty <= 0 or real_avg <= 0:
            return
        raw_entry = pos.get("entry_buy_time") or pos.get("buy_time")
        entry_ts = _parse_ts(raw_entry, None) if raw_entry else None
        if entry_ts is not None:
            key = _entry_key(code, entry_ts)
        else:
            key = self.noentry_keys.get(code)
            if key is None or key in self.closed_keys:
                if key is not None:
                    return                                     # that copy ended; same unidentified position: no reopen
                entry_ts = now
                key = f"{code}@noentry@{now.isoformat(timespec='seconds')}"
                self.noentry_keys[code] = key
                self.log(f"{TAG} {name}({code}) | entry time unknown (account-restored position) - copy keyed by "
                         f"first sight {now:%H:%M:%S}")
        if key in self.closed_keys:
            return
        vp = self.positions.get(key)
        if vp is None:
            st = self.engine.open_position(code, real_avg, qty, entry_ts, MarketSnapshot(ts=now, price=float(price)))
            vp = _VPos(key, st, name, self.tick_dir / self.date_str / f"{code}.csv", venue)
            vp.rest = rest
            vp.last_real_qty, vp.last_real_avg = real_qty, real_avg
            self.positions[key] = vp
            self.log(f"{TAG} {name}({code}) | OPEN virtual | entry={real_avg:,.0f} qty={qty} "
                     f"sl={st.sl_price:,.0f} regime={st.regime}")
            self._decision({"ts": now.isoformat(), "key": key, "code": code, "event": "OPEN", "entry": real_avg,
                            "qty": qty, "entry_ts": entry_ts.isoformat(), "sl_price": st.sl_price, "shadow": True},
                           event=True)
            self.persist(now)
            return
        # bot added shares (review #3): only the INCREASE goes to the virtual remainder, at the added shares' price
        inc = qty - vp.last_bot_qty
        if inc > 0:
            add_px = real_avg
            if vp.last_real_qty and vp.last_real_avg and real_qty > vp.last_real_qty:
                add_px = (real_avg * real_qty - vp.last_real_avg * vp.last_real_qty) / (real_qty - vp.last_real_qty)
                if add_px <= 0:
                    add_px = real_avg
            vp.pending_add += inc
            vp.pending_add_value += inc * add_px
        # the baseline follows the real bot quantity in BOTH directions: a real (partial) sell lowers it, so a later
        # re-buy is seen as an increase (re-review #2). Real sells never change the virtual remainder.
        vp.last_bot_qty = qty
        vp.last_real_qty, vp.last_real_avg = real_qty, real_avg
        self._apply_pending_add(vp, now)

    def _apply_pending_add(self, vp: _VPos, now: datetime) -> None:
        if vp.pending_add <= 0 or vp.order is not None:
            return                                             # while a virtual order is in flight: keep it pending
        st = vp.state
        if st.closed or st.held_qty <= 0:
            # the copy was sold out virtually while the bot bought more (re-review #3): the added shares are a new
            # entry into a flat virtual position - fresh engine state, same copy / key / accounting
            add_px = vp.pending_add_value / vp.pending_add
            vp.state = self.engine.open_position(st.code, add_px, vp.pending_add, now,
                                                 MarketSnapshot(ts=now, price=add_px))
            held, avg = vp.pending_add, add_px
        else:
            held = st.held_qty + vp.pending_add
            avg = (st.entry_price * st.held_qty + vp.pending_add_value) / held
            vp.state = self.engine.rebase_position(st, avg, held, now)
        vp.bought_qty += vp.pending_add
        vp.bought_value += vp.pending_add_value
        self._decision({"ts": now.isoformat(), "key": vp.key, "code": st.code, "event": "REBASE", "added": vp.pending_add,
                        "entry": avg, "held": held, "shadow": True}, event=True)
        vp.pending_add, vp.pending_add_value = 0, 0.0
        self.persist(now)

    def _step(self, key: str, now: datetime) -> None:
        vp = self.positions[key]
        if not vp.profile_loaded:
            vp.profile_loaded = True
            vp.profile = load_rvol_profile(self.data_dir, vp.state.code, self.date_str)
        for t in vp.tail.poll():
            vp.window.ingest(t)
        snap = vp.window.snapshot(now, self._effective_rest(vp, now), vp.profile, vp.venue)
        if snap.price <= 0:
            return                                              # nothing known about the price yet
        vp.last_price = snap.price
        if vp.order is not None and self._try_fill(vp, snap, now):
            self._apply_pending_add(vp, now)                   # before the "finished?" check (re-review #3)
            if vp.state.closed and vp.pending_add <= 0:
                self._close(vp, now)
                return
        res = self.engine.step(vp.state, snap, now, AIAdjust.neutral("off"))
        prev_phase = vp.state.phase
        vp.state = res.state
        if res.intent is not None:
            self._submit_virtual(vp, res.intent, snap, now)
        changed = (res.intent is not None or res.log.get("sl_changes")
                   or (res.log.get("regime_detail") or {}).get("changed") or res.state.phase != prev_phase)
        if changed or vp.last_log_at is None or (now - vp.last_log_at).total_seconds() >= HEARTBEAT_S:
            self._decision(dict(res.log, key=key, shadow=True, snapshot=_snap_fields(snap)), event=bool(changed))
            vp.last_log_at = now
        if changed or res.log.get("new_peak"):
            self._dirty = True

    def _submit_virtual(self, vp: _VPos, intent: ExitIntent, snap: MarketSnapshot, now: datetime) -> None:
        st = vp.state
        if intent.replace_pending and vp.order is not None:
            # virtual cancel is confirmed at once; nothing of the cancelled part filled
            st = self.engine.on_order_closed(st, vp.order["qty"], revert_stage=False)
            vp.order = None
        qty = st.held_qty if intent.all_remaining else min(intent.qty, st.sellable_qty)
        if qty <= 0:
            vp.state = st
            return
        oid = f"V{len(vp.fills) + 1}-{now:%H%M%S}"
        st = self.engine.on_submitted(st, intent, qty, oid, now)
        vp.order = {"id": oid, "kind": intent.kind, "qty": qty, "type": intent.order_type,
                    "limit": intent.limit_price, "reason": intent.reason, "at": now.isoformat()}
        vp.state = st
        self.log(f"{TAG} {vp.name}({st.code}) | {intent.kind} {qty}sh {intent.order_type}"
                 f"{'' if intent.limit_price is None else f'@{intent.limit_price:,.0f}'} | {intent.reason} | "
                 f"price={snap.price:,.0f} pnl={pnl_pct(self.cfg, st.entry_price, snap.price):+.2f}% (virtual)")
        self._decision({"ts": now.isoformat(), "key": vp.key, "code": st.code, "event": "ORDER", **vp.order,
                        "shadow": True}, event=True)
        self.persist(now)

    def _fresh_after(self, vp: _VPos, snap: MarketSnapshot, at: datetime, now: datetime) -> float | None:
        """First price that arrived AFTER `at`: a print after it, else a REST quote received after it (and fresh)."""
        if self._halted(vp, now):
            return None
        trades = vp.window.trades_after(at)
        if trades:
            return trades[0][1]
        r = vp.rest
        if r is not None and r.ts > at and (now - r.ts).total_seconds() <= FRESH_REST_S and r.price > 0:
            return float(r.price)
        return None

    @staticmethod
    def _effective_rest(vp: _VPos, now: datetime) -> RestQuote | None:
        """The REST quote as the ENGINE may use it: halt / VI flags expire HALT_VALID_S after they were received (the
        quote stops refreshing once the real position is sold), so an old flag can never freeze the engine while
        prints keep coming (final gate review). The price / age stay as received."""
        r = vp.rest
        if r is None or not (r.halted or r.vi_active) or (now - r.ts).total_seconds() <= HALT_VALID_S:
            return r
        return replace(r, halted=False, vi_active=False)

    @staticmethod
    def _halted(vp: _VPos, now: datetime) -> bool:
        r = vp.rest
        return bool(r is not None and r.halted and (now - r.ts).total_seconds() <= HALT_VALID_S)

    def _try_fill(self, vp: _VPos, snap: MarketSnapshot, now: datetime) -> bool:
        o = vp.order
        at = datetime.fromisoformat(o["at"])
        if o["type"] == "limit":
            if self._halted(vp, now):
                return False
            lim = float(o["limit"])
            if any(px > lim + 1e-9 for _, px in vp.window.trades_after(at)):
                self._fill(vp, lim, now)
                return True
            if (now - at).total_seconds() < SHADOW_LIMIT_TO_MARKET_S:
                return False
            px = self._fresh_after(vp, snap, at + timedelta(seconds=SHADOW_LIMIT_TO_MARKET_S), now)
            if px is None:
                return False
            self._fill(vp, tick_down(px), now, note="limit->market")
            return True
        px = self._fresh_after(vp, snap, at, now)
        if px is None:
            return False                                       # no data after the order yet / halted: unfilled
        self._fill(vp, tick_down(px), now)
        return True

    def _fill(self, vp: _VPos, price: float, now: datetime, note: str = "") -> None:
        o = vp.order
        qty = int(o["qty"])
        entry_then = vp.state.entry_price
        vp.state = self.engine.on_fill(vp.state, qty)
        fill = {"ts": now.isoformat(), "kind": o["kind"], "qty": qty, "price": price, "reason": o["reason"],
                "order_at": o["at"], "cost": entry_then, "net_pct": round(pnl_pct(self._net_cfg, entry_then, price), 4),
                "note": note}
        vp.fills.append(fill)
        vp.order = None
        self._decision({"ts": now.isoformat(), "key": vp.key, "code": vp.state.code, "event": "FILL", **fill,
                        "held_after": vp.state.held_qty, "shadow": True}, event=True)
        self.persist(now)

    def _close(self, vp: _VPos, now: datetime, status: str = "CLOSED", reason: str = "") -> None:
        st = vp.state
        if status == "CLOSED" and vp.pending_add > 0:
            raise RuntimeError("CLOSED with pending bot shares")   # invariant (re-review #3)
        sold = sum(f["qty"] for f in vp.fills)
        proceeds = sum(f["qty"] * f["price"] for f in vp.fills)
        cfg = self._net_cfg
        fee, tax = cfg.fee_pct / 100.0, cfg.tax_pct / 100.0
        # each fill's cost = the virtual average cost AT that fill: later adds never re-price earlier sales (#6)
        cost_sold = sum(f["qty"] * float(f.get("cost", st.entry_price)) for f in vp.fills)
        net = ((proceeds * (1 - fee - tax) - cost_sold * (1 + fee)) / (cost_sold * (1 + fee)) * 100.0
               if cost_sold > 0 else None)
        rec = {"ts": now.isoformat(), "key": vp.key, "code": st.code, "name": vp.name, "status": status,
               "reason": reason, "entry": round(vp.bought_value / vp.bought_qty, 2) if vp.bought_qty else st.entry_price,
               "bought_qty": vp.bought_qty, "sold_qty": sold, "remaining_qty": st.held_qty,
               "entry_ts": st.entry_ts.isoformat(), "peak": st.peak_price,
               "peak_net_pct": round(pnl_pct(cfg, st.entry_price, st.peak_price), 4),
               "avg_exit": round(proceeds / sold, 2) if sold else None,
               "net_pct_sold": None if net is None else round(net, 4), "last_price": vp.last_price,
               "open_order": vp.order, "fills": vp.fills, "regime_switches": st.regime_switches, "shadow": True}
        self._write(self.trades_path, rec)
        self.closed_keys.add(vp.key)
        self.log(f"{TAG} {vp.name}({st.code}) | {status} virtual{(' ' + reason) if reason else ''} | "
                 f"sold={sold}/{vp.bought_qty} net={rec['net_pct_sold']} peak={rec['peak_net_pct']:+.2f}%")
        del self.positions[vp.key]
        self.persist(now)

    # -- persistence / files ---------------------------------------------------------------------------
    def persist(self, now: datetime | None = None) -> None:
        try:
            data = {"date": self.date_str, "positions": {k: vp.to_dict() for k, vp in self.positions.items()},
                    "closed_keys": sorted(self.closed_keys), "noentry_keys": self.noentry_keys}
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, default=_json_default), encoding="utf-8")
            os.replace(tmp, self.state_path)
            self._last_persist = now
            self._dirty = False
        except Exception as exc:  # noqa: BLE001
            self._error("persist", exc)

    def _restore(self) -> None:
        try:
            # the trades journal is the source of truth for finished copies (a crash may have lost the checkpoint)
            if self.trades_path.exists():
                for line in self.trades_path.read_text(encoding="utf-8").splitlines():
                    try:
                        k = json.loads(line).get("key")
                    except ValueError:
                        continue
                    if k:
                        self.closed_keys.add(k)
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.closed_keys |= set(data.get("closed_keys") or [])
            self.noentry_keys = dict(data.get("noentry_keys") or {})
            for key, d in (data.get("positions") or {}).items():
                if key in self.closed_keys:
                    continue
                st = ExitState.from_dict(d["state"])
                vp = _VPos(key, st, d.get("name", st.code), self.tick_dir / self.date_str / f"{st.code}.csv",
                           d.get("venue", "KRX"))
                vp.order, vp.fills = d.get("order"), d.get("fills") or []
                vp.bought_qty = int(d.get("bought_qty", st.q0))
                vp.bought_value = float(d.get("bought_value", st.q0 * st.entry_price))
                vp.last_bot_qty = int(d.get("last_bot_qty", st.q0))
                vp.last_real_qty, vp.last_real_avg = d.get("last_real_qty"), d.get("last_real_avg")
                vp.pending_add = int(d.get("pending_add", 0))
                vp.pending_add_value = float(d.get("pending_add_value", 0.0))
                self.positions[key] = vp
            if self.positions:
                self.log(f"{TAG} restored {len(self.positions)} virtual position(s) from {self.state_path.name}")
        except Exception as exc:  # noqa: BLE001
            self._error("restore", exc)

    def _cleanup_old(self) -> None:
        try:
            cutoff = (datetime.strptime(self.date_str, "%Y%m%d") - timedelta(days=DECISION_RETENTION_DAYS)).strftime("%Y%m%d")
            for p in self.out_dir.glob("exit_engine_*.jsonl"):
                d = p.stem.rsplit("_", 1)[-1]
                if d.isdigit() and len(d) == 8 and d < cutoff:
                    p.unlink()
        except Exception as exc:  # noqa: BLE001
            self._error("cleanup", exc)

    def _decision(self, rec: dict, event: bool) -> None:
        if not event and self._decision_bytes >= MAX_DECISION_BYTES:
            if not self._decision_cap_logged:
                self._decision_cap_logged = True
                self.log(f"{TAG} decision log reached {MAX_DECISION_BYTES // (1024 * 1024)} MB - heartbeats dropped "
                         f"for today, events still written")
            return
        self._decision_bytes += self._write(self.decisions_path, rec)

    def _write(self, path: Path, rec: dict) -> int:
        line = json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return len(line.encode("utf-8"))


# ---------------------------------------------------------------------------
# threaded service for r003 (the trading thread only enqueues)
# ---------------------------------------------------------------------------

class ShadowExitService:
    """runner: a ShadowExitRunner, or a zero-argument factory that builds one ON THE WORKER THREAD (its restore /
    cleanup file I/O then never runs on the trading thread - re-review #5)."""

    def __init__(self, runner, interval_s: float = SHADOW_EVAL_INTERVAL_S,
                 queue_max: int = QUEUE_MAX, clock: Callable[[], datetime] = datetime.now,
                 log: Callable[[str], None] | None = None):
        self._factory = None if isinstance(runner, ShadowExitRunner) else runner
        self.runner: ShadowExitRunner | None = runner if isinstance(runner, ShadowExitRunner) else None
        self.interval_s = interval_s
        self.clock = clock
        self.log = log or (self.runner.log if self.runner else (lambda _m: None))
        self.dropped = 0
        self._dropped_logged = 0
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._finalize_reason: str | None = None
        self._thread = threading.Thread(target=self._worker, name="exit-shadow", daemon=True)
        self._thread.start()

    def observe(self, code: str, name: str, now: datetime, pos: dict | None, bot_qty: int | None, price: float,
                quote_status: dict | None = None, venue: str = "KRX") -> None:
        """Called on the trading thread: copies the inputs and enqueues them. Never blocks, never raises."""
        try:
            item = (code, name, now, dict(pos) if pos else None, bot_qty, float(price),
                    dict(quote_status) if quote_status else None, venue)
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1
        except Exception:  # noqa: BLE001
            self.dropped += 1

    def tick(self, now: datetime) -> None:
        """Kept for interface compatibility: evaluation runs on the worker's own schedule."""

    def close(self, finalize_reason: str | None = None, timeout_s: float = 5.0) -> bool:
        """Stops the worker. finalize_reason (e.g. "DATE_CHANGE") ends every open copy as INCOMPLETE; None keeps
        them for the restart checkpoint - unless the stop happens at/after the session end, then the session-end
        evaluation + SESSION_END finalize still run (re-review #4). timeout_s=0: do not wait (date change keeps the
        trading loop free; the worker finishes on its own). Returns True when the worker exited within timeout_s."""
        self._finalize_reason = finalize_reason
        self._stop.set()
        if timeout_s > 0:
            self._thread.join(timeout_s)
        alive = self._thread.is_alive()
        if alive and timeout_s > 0:
            self.log(f"{TAG} worker did not stop within {timeout_s:.0f}s - checkpoint may be incomplete")
        return not alive

    def _drain(self, limit: int = 500) -> None:
        for _ in range(limit):
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            self.runner.observe(*item)

    def _log_drops(self, now: datetime, force: bool = False) -> None:
        if self.dropped > self._dropped_logged and (force or self._last_drop_log is None
                                                    or (now - self._last_drop_log).total_seconds() >= DROP_LOG_EVERY_S):
            self.log(f"{TAG} {self.dropped} observation(s) dropped so far (queue full) | queue={self._q.qsize()}")
            self._dropped_logged = self.dropped
            self._last_drop_log = now

    def _worker(self) -> None:
        self._last_drop_log: datetime | None = None
        try:
            if self.runner is None:
                self.runner = self._factory()
                self.log = self.runner.log
        except Exception as exc:  # noqa: BLE001
            self.log(f"{TAG} init failed on worker, shadow disabled: {type(exc).__name__}: {exc}")
            self._stop.set()
            return
        self.ready.set()
        last_eval: datetime | None = None
        while not self._stop.is_set():
            try:
                self._drain()
                now = self.clock()
                if last_eval is None or (now - last_eval).total_seconds() >= self.interval_s:
                    self.runner.tick(now)
                    last_eval = now
                self._log_drops(now)
            except Exception as exc:  # noqa: BLE001
                self.runner._error("worker", exc)
            self._stop.wait(0.2)
        try:
            self._drain(limit=QUEUE_MAX)
            now = self.clock()
            if self._finalize_reason:
                self.runner.finalize(now, self._finalize_reason)
            elif now.time() >= SESSION_END:
                self.runner.tick(now)                          # session-end evaluation + SESSION_END finalize
            else:
                self.runner.persist(now)
            self._log_drops(now, force=True)
        except Exception as exc:  # noqa: BLE001
            self.runner._error("worker stop", exc)


def _snap_fields(snap: MarketSnapshot) -> dict:
    d = asdict(snap)
    d["ts"] = snap.ts.isoformat()
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}


def _json_default(o: Any):
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
