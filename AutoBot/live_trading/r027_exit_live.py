# -*- coding: utf-8 -*-
"""R027 adaptive exit LIVE manager - the r024 engine decides, r003 place_sell_order executes (real orders).

Enabled only when r001 EXIT_ENGINE_MODE == "live" AND AUTO_SELL_MODE == "all" (r006 ADAPTIVE_SELL_CONDITIONS).
"stop_only" / "off" keep priority: the user chose manual selling, the engine never sends an order then.
Design: adaptive_exit_design_20261115.md section 24 (live wiring, user decision 2026-11-15 "바로 적용").

Division of work
- r027 (this file, no broker import): one engine state per bot position keyed by (code, entry time), tick features
  from the g011 tick file (r025 window, same as shadow), restart checkpoint, decision log. decide() returns at most one
  LiveExitOrder; submitted() reports what r003 did with it.
- r006 _x10_adaptive_exit: calls decide(), sends the order through api.place_sell_order (KRX regular session = market
  order, the existing tested path incl. pending tracking / unknown-submission reconcile / bot_only quantity cap).
- r003: calls decide() only while NO sell order of the symbol is in flight (pending -> `continue`). The next call after
  the order ended reconciles: fills come from api.last_sell_confirm (exact filled qty of OUR order, matched by reason
  and time), otherwise from the holding difference.

Execution rules (assumptions, section 24)
- Every engine order goes out as a KRX market order: the ladder's marketable limit (max(target, bid)) and the VI limit
  (※1) are not used live - place_sell_order has no KRX limit + reprice path. During VI the engine only emits STOP / FORCE
  and the market order joins the VI single-price auction.
- A failed submit (rejected / cooldown / no order spec) reverts the stage and retries after RETRY_S.
- Holding increase: rebase at once while a bot BUY is in flight or nothing was sold yet (late buy fill); after a sell it
  must persist UP_CONFIRM_S (a stale account row right after our own sell resolves within one sync) - Codex live #4.
  Holding decrease without an engine order -> manual sell, sync_holding.
- Restart (Codex live #1): r003 forgets its pending orders on restart, so an engine order restored as in flight (or a
  submit that crashed before submitted()) is NOT assumed finished: the position waits up to RESTORE_WAIT_S for the
  holding to show the fill (market orders end within seconds; a VI auction within ~2 min). Meanwhile only STOP / FORCE
  may sell, and only the shares not covered by the in-flight order.
- Stale REST price (Codex live #2): r006 passes price=None; the engine still runs on the tick file (SAFE / DEAD modes,
  FORCE by time). Halt / VI flags expire HALT_VALID_S after the quote that carried them (quote_status["at"], #3).
- Outside the regular session r006 falls back to the net stop (_s04); FORCE starts at 15:15 (limit -> market 15:18 in
  the engine; live: market), r003's 15:20 scheduled liquidation stays as a backstop.

Files (runtime/data/live_runtime)
- exit_live_engine_YYYYMMDD.jsonl  decision log (events + 60 s heartbeat, capped per day like r026)
- exit_live_trades_YYYYMMDD.jsonl   one record per finished position (engine view: sold qty, reasons)
- exit_live_YYYYMMDD.json           restart checkpoint (engine states, closed keys)

Update log:
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 적응형 청산 엔진(r024) 실주문 관리자. 사용자 요청 "오늘 만든 매도 엔진을 바로 적용"(2주 shadow 생략).
    impact: live (EXIT_ENGINE_MODE="live" + AUTO_SELL_MODE="all" 일 때 매도 판정을 r024 엔진이 한다)
    compatibility: backward-compatible (EXIT_ENGINE_MODE="shadow"|"off" 면 기존 매도 조건 그대로)
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from r024_adaptive_exit import (
    PRIORITY, AIAdjust, ExitConfig, ExitDecisionEngine, ExitIntent, ExitState, MarketSnapshot, pnl_pct,
)
from r025_exit_tick_feed import ExitTickWindow, RestQuote, RvolProfile
from r026_exit_shadow import (
    DECISION_RETENTION_DAYS, HALT_VALID_S, HEARTBEAT_S, MAX_DECISION_BYTES, TickFileTail, _entry_key, _json_default,
    _parse_ts, _snap_fields, _VI_OFF, load_rvol_profile,
)

TAG = "  [EXIT ENGINE ]"
RETRY_S = 5.0                 # after a failed submit
RESTORE_WAIT_S = 180.0        # restored in-flight order: wait this long for the holding to show the fill
UP_CONFIRM_S = 30.0           # holding increase after a sell must persist this long (else stale account row)
FAIL_WARN_EVERY = 3           # log a warning every N consecutive failures
REOPEN_AFTER_CLOSE_S = 30.0   # a closed key whose holding is still > 0 after this is reopened (missed fill / stale)
PERSIST_EVERY_S = 30.0


@dataclass
class LiveExitOrder:
    key: str
    code: str
    kind: str
    qty: int
    reason: str           # r003 pending "reason" - also used to match the fill record
    intent: ExitIntent
    pnl: float
    price: float


class _LPos:
    def __init__(self, key: str, state: ExitState, name: str, tick_path: Path):
        self.key = key
        self.state = state
        self.name = name
        self.window = ExitTickWindow(state.code)
        self.window.peak_from = state.entry_ts
        self.tail = TickFileTail(tick_path)
        self.rest: RestQuote | None = None
        self.profile: RvolProfile | None = None
        self.profile_started = False
        self.order: dict | None = None        # last submitted order awaiting reconcile
        self.retry_after: datetime | None = None
        self.fail_count = 0
        self.reasons: list[str] = []
        self.last_log_at: datetime | None = None
        self.last_price = 0.0
        self.flag_at: datetime | None = None     # receive time of the quote that carried the halt / VI flags
        self.restored_inflight = False           # restored with an order whose end r003 no longer tracks
        self.restore_deadline: datetime | None = None
        self.up_since: datetime | None = None    # holding above the engine's since
        self.restore_filled = 0                  # shares of the restored in-flight order seen filled so far

    def to_dict(self) -> dict:
        return {"key": self.key, "state": self.state.to_dict(), "name": self.name, "order": self.order,
                "reasons": self.reasons}


class LiveExitManager:
    def __init__(self, tick_dir: Path, data_dir: Path, out_dir: Path, date_str: str, log: Callable[[str], None],
                 cfg: ExitConfig | None = None, cleanup: bool = True, async_profile: bool = True):
        self.tick_dir = Path(tick_dir)
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.date_str = date_str
        self.log = log
        base = cfg or ExitConfig()
        # no exit model exists yet (design section 6 / Q10): AI stays off until g034 promotes one
        self.cfg = ExitConfig.from_mapping({**asdict(base), "ai_mode": "off"})
        self.engine = ExitDecisionEngine(self.cfg)
        self.async_profile = async_profile
        self.positions: dict[str, _LPos] = {}       # key -> position
        self.by_code: dict[str, str] = {}           # code -> key of the managed position
        self.closed: dict[str, datetime] = {}       # key -> close time
        self.noentry_keys: dict[str, str] = {}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.out_dir / f"exit_live_engine_{date_str}.jsonl"
        self.trades_path = self.out_dir / f"exit_live_trades_{date_str}.jsonl"
        self.state_path = self.out_dir / f"exit_live_{date_str}.json"
        self._decision_bytes = self.decisions_path.stat().st_size if self.decisions_path.exists() else 0
        self._decision_cap_logged = False
        self._last_persist: datetime | None = None
        self.errors = 0
        if cleanup:
            self._cleanup_old()
        self._restore()

    # -- public ----------------------------------------------------------------------------------------
    def decide(self, code: str, name: str, now: datetime, pos: dict | None, bot_qty: int | None, price: float | None,
               quote_status: dict | None, sell_fill: dict | None = None, sell_in_flight: bool = False,
               buy_in_flight: bool = False, venue: str = "KRX") -> LiveExitOrder | None:
        """One evaluation of one symbol. price=None: the REST price is stale (the engine runs on ticks only).
        sell_fill = api.last_sell_confirm.get(code). An internal error is logged and re-raised as RuntimeError so the
        caller can fall back to its safety stop."""
        try:
            px = float(price) if price is not None and float(price) > 0 else None
            return self._decide(str(code).zfill(6), name, now, pos, bot_qty, px, quote_status or {},
                                sell_fill, sell_in_flight, buy_in_flight, venue)
        except Exception as exc:  # noqa: BLE001
            self._error(f"decide {code}", exc)
            raise RuntimeError(f"adaptive exit decide failed: {type(exc).__name__}: {exc}") from exc

    def submitted(self, order: LiveExitOrder, ok: bool, in_flight: bool, now: datetime) -> None:
        """ok = place_sell_order returned True; in_flight = a sell pending exists anyway (submission unknown -
        the broker may have it). Either way the engine treats the order as sent."""
        lp = self.positions.get(order.key)
        if lp is None:
            return
        try:
            if (lp.order or {}).get("submitting"):
                lp.order = None
            if ok or in_flight:
                lp.state = self.engine.on_submitted(lp.state, order.intent, order.qty, order.reason, now)
                lp.order = {"kind": order.kind, "qty": order.qty, "reason": order.reason, "at": now.isoformat(),
                            "held_before": lp.state.held_qty, "unknown": bool(in_flight and not ok)}
                lp.reasons.append(order.reason)
                lp.fail_count = 0
                lp.retry_after = None
                self._decision({"ts": now.isoformat(), "key": lp.key, "code": order.code, "event": "ORDER",
                                **lp.order, "price": order.price, "pnl": round(order.pnl, 4)}, event=True)
            else:
                lp.fail_count += 1
                lp.retry_after = now + timedelta(seconds=RETRY_S)
                if lp.fail_count == 1 or lp.fail_count % FAIL_WARN_EVERY == 0:
                    self.log(f"{TAG} {lp.name}({order.code}) | {order.kind} submit failed ({lp.fail_count}) - "
                             f"retry in {RETRY_S:.0f}s | {order.reason}")
                self._decision({"ts": now.isoformat(), "key": lp.key, "code": order.code, "event": "SUBMIT_FAIL",
                                "kind": order.kind, "qty": order.qty, "fails": lp.fail_count}, event=True)
            self.persist(now)
        except Exception as exc:  # noqa: BLE001
            self._error(f"submitted {order.code}", exc)

    def managed_codes(self) -> set[str]:
        return set(self.by_code)

    # -- core ------------------------------------------------------------------------------------------
    def _decide(self, code, name, now, pos, bot_qty, price, qs, sell_fill, sell_in_flight, buy_in_flight, venue):
        real_qty = int((pos or {}).get("quantity", 0) or 0)
        qty = real_qty if bot_qty is None else min(int(bot_qty), real_qty)
        avg = float((pos or {}).get("buy_price") or 0)
        key = self._key_for(code, pos, now)
        lp = self.positions.get(self.by_code.get(code, "")) if code in self.by_code else None
        if lp is not None and key is not None and lp.key != key:
            # a different entry of the same symbol: the old one is over (sold out and re-bought)
            self._finish(lp, now, "REPLACED_BY_NEW_ENTRY")
            lp = None
        if lp is not None:
            self._set_rest(lp, now, price, qs)
        if sell_in_flight:
            return None                                          # r003 tracks the order; reconcile afterwards
        if lp is None:
            if qty <= 0 or avg <= 0 or key is None:
                return None
            closed_at = self.closed.get(key)
            if closed_at is not None:
                if (now - closed_at).total_seconds() < REOPEN_AFTER_CLOSE_S:
                    return None
                self.log(f"{TAG} {name}({code}) | holding {qty} still present {REOPEN_AFTER_CLOSE_S:.0f}s after the "
                         f"engine finished it - managing it again")
                key = f"{key}#r{now:%H%M%S}"
            lp = self._open(code, name, key, pos, qty, avg, now, price, qs)
            if lp is None:
                return None                                      # no price at all yet (stale REST, no prints)
        st = lp.state
        waiting = False
        if lp.restored_inflight:
            waiting = self._restore_wait(lp, qty, now)
        elif lp.order is not None or st.pending_qty > 0:
            self._reconcile(lp, qty, sell_fill, now)
        else:
            self._sync(lp, qty, avg, now, buy_in_flight)
        if lp.state.closed or lp.state.held_qty <= 0:
            self._finish(lp, now, "SOLD_OUT")
            return None
        if lp.retry_after is not None and now < lp.retry_after and not waiting:
            return None
        if not lp.profile_started:
            self._start_profile(lp)
        for t in lp.tail.poll():
            lp.window.ingest(t)
        snap = lp.window.snapshot(now, self._effective_rest(lp, now), lp.profile, venue)
        if snap.price <= 0:
            return None
        lp.last_price = snap.price
        res = self.engine.step(lp.state, snap, now, AIAdjust.neutral("off"))
        prev_phase = lp.state.phase
        lp.state = res.state
        changed = (res.intent is not None or res.log.get("sl_changes")
                   or (res.log.get("regime_detail") or {}).get("changed") or res.state.phase != prev_phase)
        if changed or lp.last_log_at is None or (now - lp.last_log_at).total_seconds() >= HEARTBEAT_S:
            self._decision(dict(res.log, key=lp.key, live=True, snapshot=_snap_fields(snap)), event=bool(changed))
            lp.last_log_at = now
        if changed or res.log.get("new_peak"):
            if self._last_persist is None or changed or (now - self._last_persist).total_seconds() >= PERSIST_EVERY_S:
                self.persist(now)
        intent = res.intent
        if intent is None:
            return None
        st = lp.state
        if waiting:
            # restored in-flight order not resolved yet: only a stop / force exit, only the uncovered shares
            if intent.priority < PRIORITY["STOP"]:
                return None
            n = st.sellable_qty
        elif intent.all_remaining or intent.replace_pending:
            n = st.held_qty
        else:
            n = min(int(intent.qty), st.sellable_qty)
        if n <= 0:
            return None
        pnl = pnl_pct(self.cfg, st.entry_price, snap.price)
        reason = f"ADAPTIVE_{intent.reason}"
        self.log(f"{TAG} {lp.name}({code}) | {intent.kind} {n}/{st.held_qty}sh | {intent.reason} | price={snap.price:,.0f} "
                 f"pnl={pnl:+.2f}% peak={pnl_pct(self.cfg, st.entry_price, st.peak_price):+.2f}% "
                 f"sl={st.sl_price:,.0f} regime={st.regime} mode={res.log.get('op_mode')}")
        # crash between the order and submitted(): the restore treats this marker as an order in flight
        if not waiting:
            lp.order = {"submitting": True, "kind": intent.kind, "qty": n, "reason": reason, "at": now.isoformat()}
        self.persist(now)
        return LiveExitOrder(lp.key, code, intent.kind, n, reason, intent, pnl, snap.price)

    def _key_for(self, code: str, pos: dict | None, now: datetime) -> str | None:
        if not pos:
            return self.by_code.get(code)
        raw = pos.get("entry_buy_time") or pos.get("buy_time")
        ts = _parse_ts(raw, None) if raw else None
        if ts is not None:
            return _entry_key(code, ts)
        key = self.noentry_keys.get(code)
        if key is None:
            key = f"{code}@noentry@{now.isoformat(timespec='seconds')}"
            self.noentry_keys[code] = key
        return key

    def _open(self, code, name, key, pos, qty, avg, now, price, qs) -> _LPos | None:
        """price=None (stale REST price) still opens on the tick file's price, so a stop / FORCE is never skipped just
        because the first look came with a stale quote (Codex live re-review #2)."""
        raw = (pos or {}).get("entry_buy_time") or (pos or {}).get("buy_time")
        entry_ts = _parse_ts(raw, None) or now
        st = self.engine.open_position(code, avg, qty, entry_ts, MarketSnapshot(ts=now, price=float(price or avg)))
        lp = _LPos(key, st, name, self.tick_dir / self.date_str / f"{code}.csv")
        self._set_rest(lp, now, price, qs)
        for t in lp.tail.poll():                                 # history since entry -> peak / ATR before 1st step
            lp.window.ingest(t)
        snap = lp.window.snapshot(now, lp.rest, None, "KRX")
        if price is None and snap.price <= 0:
            return None
        if snap.atr1m_pct:
            lp.state = self.engine.open_position(code, avg, qty, entry_ts, snap)
        self.positions[key] = lp
        self.by_code[code] = key
        self.log(f"{TAG} {name}({code}) | MANAGE | entry={avg:,.0f} qty={qty} sl={lp.state.sl_price:,.0f} "
                 f"regime={lp.state.regime} entry_ts={entry_ts:%H:%M:%S}")
        self._decision({"ts": now.isoformat(), "key": key, "code": code, "event": "OPEN", "entry": avg, "qty": qty,
                        "entry_ts": entry_ts.isoformat(), "sl_price": lp.state.sl_price, "live": True}, event=True)
        self.persist(now)
        return lp

    def _set_rest(self, lp: _LPos, now: datetime, price: float | None, qs: dict) -> None:
        """price=None (stale REST price): keep the previous quote - it ages and the engine falls back to ticks."""
        vi = str(qs.get("vi_cls_code", "") or "").strip().upper() not in _VI_OFF
        halted = str(qs.get("temp_stop_yn", "") or "").strip().upper() == "Y"
        flag_at = qs.get("at") if isinstance(qs.get("at"), datetime) else now
        if lp.flag_at is None or flag_at >= lp.flag_at:
            lp.flag_at = flag_at
        if price is not None:
            lp.rest = RestQuote(ts=now, price=float(price), vi_active=vi, halted=halted)
        elif lp.rest is not None:
            lp.rest = replace(lp.rest, vi_active=vi, halted=halted)

    @staticmethod
    def _effective_rest(lp: _LPos, now: datetime) -> RestQuote | None:
        """Halt / VI flags count only HALT_VALID_S after the quote that carried them was received (Codex live #3)."""
        r = lp.rest
        if r is None or not (r.halted or r.vi_active):
            return r
        at = lp.flag_at or r.ts
        if (now - at).total_seconds() <= HALT_VALID_S:
            return r
        return replace(r, halted=False, vi_active=False)

    def _restore_wait(self, lp: _LPos, held_now: int, now: datetime) -> bool:
        """Restored with an order in flight. True while still waiting (the caller allows STOP / FORCE only)."""
        st = lp.state
        if lp.restore_deadline is None:
            lp.restore_deadline = now + timedelta(seconds=RESTORE_WAIT_S)
            self.log(f"{TAG} {lp.name}({st.code}) | restored with an order in flight ({(lp.order or {}).get('kind')} "
                     f"{(lp.order or {}).get('qty')}sh) - waiting up to {RESTORE_WAIT_S:.0f}s for the holding to show it")
        if st.pending_qty <= 0 and lp.order is not None and not lp.order.get("counted"):
            # a "submitting" marker: the engine never saw on_submitted - count those shares as in flight
            lp.state = st = replace(st, pending_qty=min(int(lp.order.get("qty", 0) or 0), st.held_qty))
            lp.order = dict(lp.order, counted=True)
        if held_now < st.held_qty and st.pending_qty > 0:
            # only the DECREASE is a fill; an unfilled rest stays in flight (Codex live re-review #1)
            filled = min(st.held_qty - held_now, st.pending_qty)
            lp.state = st = self.engine.on_fill(st, filled)
            lp.restore_filled += filled
            self._decision({"ts": now.isoformat(), "key": lp.key, "code": st.code, "event": "RESTORED_FILL",
                            "filled": filled, "pending_left": st.pending_qty, "held_engine": st.held_qty}, event=True)
            self.persist(now)
        if st.pending_qty <= 0 or now >= lp.restore_deadline:
            if st.pending_qty > 0:     # deadline: the rest never filled -> over (a KRX market order cannot be open this long)
                lp.state = st = self.engine.on_order_closed(st, st.pending_qty, revert_stage=(lp.restore_filled == 0))
                if lp.restore_filled == 0:
                    lp.retry_after = now + timedelta(seconds=RETRY_S)
            lp.restored_inflight = False
            lp.restore_deadline = None
            lp.restore_filled = 0
            lp.order = None
            self._sync(lp, held_now, None, now)
            self.persist(now)
            return False
        return True

    def _reconcile(self, lp: _LPos, held_now: int, sell_fill: dict | None, now: datetime) -> None:
        """The engine's order is over (r003 has no pending sell for the symbol any more)."""
        st, order = lp.state, lp.order or {}
        if order.get("submitting") and st.pending_qty <= 0:
            lp.order = None                       # decide() returned an order but submitted() never came: none sent
            return
        filled = None
        sub_at = _parse_ts(order.get("at"), None)
        if sell_fill and order and str(sell_fill.get("reason", "")) == order.get("reason"):
            at = _parse_ts(sell_fill.get("at"), None)
            if at is None or sub_at is None or at >= sub_at - timedelta(seconds=1):
                filled = int(sell_fill.get("filled_qty", 0) or 0)
        if filled is None:
            filled = max(0, st.held_qty - held_now)
        filled = min(filled, st.pending_qty) if st.pending_qty > 0 else 0
        if filled > 0:
            st = self.engine.on_fill(st, filled)
        if st.pending_qty > 0:
            st = self.engine.on_order_closed(st, st.pending_qty, revert_stage=(filled == 0))
        self._decision({"ts": now.isoformat(), "key": lp.key, "code": st.code, "event": "ORDER_DONE",
                        "kind": order.get("kind"), "ordered": order.get("qty"), "filled": filled,
                        "held_engine": st.held_qty, "held_account": held_now, "unknown": order.get("unknown")},
                       event=True)
        if filled == 0:
            lp.retry_after = now + timedelta(seconds=RETRY_S)     # unfilled / cancelled: do not resend at once
        lp.order = None
        lp.state = st
        self._sync(lp, held_now, None, now)
        self.persist(now)

    def _sync(self, lp: _LPos, held_now: int, avg: float | None, now: datetime, buy_in_flight: bool = False) -> None:
        st = lp.state
        if held_now <= st.held_qty:
            lp.up_since = None
        if held_now == st.held_qty:
            return
        if held_now < st.held_qty:
            lp.state = self.engine.sync_holding(st, held_now)
            self._decision({"ts": now.isoformat(), "key": lp.key, "code": st.code, "event": "HOLDING_DOWN",
                            "engine": st.held_qty, "account": held_now}, event=True)
            if st.held_qty - held_now > 0:
                self.log(f"{TAG} {lp.name}({st.code}) | holding {st.held_qty} -> {held_now} without an engine order "
                         f"(manual sell or late account sync)")
        else:
            if lp.up_since is None:
                lp.up_since = now
            confirmed = (st.sold_qty == 0 or buy_in_flight
                         or (now - lp.up_since).total_seconds() >= UP_CONFIRM_S)
            if not confirmed or not avg:
                return                       # maybe a stale account row right after our own sell: wait
            # late buy fill (or a bot buy) into the managed position -> rebase on the real holding (design C15)
            lp.state = self.engine.rebase_position(st, avg, held_now, now)
            lp.up_since = None
            self._decision({"ts": now.isoformat(), "key": lp.key, "code": st.code, "event": "REBASE",
                            "held": held_now, "entry": avg, "engine_held": st.held_qty, "sold": st.sold_qty},
                           event=True)
            if st.sold_qty > 0 or buy_in_flight:
                self.log(f"{TAG} {lp.name}({st.code}) | holding {st.held_qty} -> {held_now} (bot buy fill) - rebased")
        self.persist(now)

    def _finish(self, lp: _LPos, now: datetime, reason: str) -> None:
        st = lp.state
        rec = {"ts": now.isoformat(), "key": lp.key, "code": st.code, "name": lp.name, "reason": reason,
               "entry": st.entry_price, "q0": st.q0, "sold": st.sold_qty, "held": st.held_qty,
               "orders": lp.reasons, "peak_price": st.peak_price,
               "peak_pct": round(pnl_pct(self.cfg, st.entry_price, st.peak_price), 4), "sl_price": st.sl_price,
               "regime": st.regime}
        self._write(self.trades_path, rec)
        self.closed[lp.key] = now
        self.positions.pop(lp.key, None)
        if self.by_code.get(st.code) == lp.key:
            self.by_code.pop(st.code, None)
        if self.noentry_keys.get(st.code) == lp.key:
            self.noentry_keys.pop(st.code, None)
        self.log(f"{TAG} {lp.name}({st.code}) | DONE {reason} | sold={st.sold_qty}/{st.q0} orders={len(lp.reasons)}")
        self.persist(now)

    def _start_profile(self, lp: _LPos) -> None:
        lp.profile_started = True

        def _load() -> None:
            lp.profile = load_rvol_profile(self.data_dir, lp.state.code, self.date_str)

        if self.async_profile:
            threading.Thread(target=_load, name=f"exit-rvol-{lp.state.code}", daemon=True).start()
        else:
            _load()

    # -- persistence / files ---------------------------------------------------------------------------
    def persist(self, now: datetime | None = None) -> None:
        try:
            data = {"date": self.date_str, "positions": {k: lp.to_dict() for k, lp in self.positions.items()},
                    "closed": {k: v.isoformat() for k, v in self.closed.items()}, "noentry_keys": self.noentry_keys}
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, default=_json_default), encoding="utf-8")
            os.replace(tmp, self.state_path)
            self._last_persist = now
        except Exception as exc:  # noqa: BLE001
            self._error("persist", exc)

    def _restore(self) -> None:
        try:
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            for k, v in (data.get("closed") or {}).items():
                ts = _parse_ts(v, None)
                if ts is not None:
                    self.closed[k] = ts
            self.noentry_keys = dict(data.get("noentry_keys") or {})
            for key, d in (data.get("positions") or {}).items():
                st = ExitState.from_dict(d["state"])
                lp = _LPos(key, st, d.get("name", st.code), self.tick_dir / self.date_str / f"{st.code}.csv")
                lp.order = d.get("order")
                lp.reasons = list(d.get("reasons") or [])
                # r003 lost its pending orders on restart: an order still in flight here is not known to be over
                lp.restored_inflight = bool(lp.order) or st.pending_qty > 0
                self.positions[key] = lp
                self.by_code[st.code] = key
            if self.positions:
                self.log(f"{TAG} restored {len(self.positions)} position(s) from {self.state_path.name}")
        except Exception as exc:  # noqa: BLE001
            self._error("restore", exc)

    def _cleanup_old(self) -> None:
        try:
            cutoff = (datetime.strptime(self.date_str, "%Y%m%d") - timedelta(days=DECISION_RETENTION_DAYS)).strftime("%Y%m%d")
            for p in self.out_dir.glob("exit_live_engine_*.jsonl"):
                d = p.stem.rsplit("_", 1)[-1]
                if d.isdigit() and len(d) == 8 and d < cutoff:
                    p.unlink()
        except Exception as exc:  # noqa: BLE001
            self._error("cleanup", exc)

    def _decision(self, rec: dict, event: bool) -> None:
        if not event and self._decision_bytes >= MAX_DECISION_BYTES:
            if not self._decision_cap_logged:
                self._decision_cap_logged = True
                self.log(f"{TAG} decision log reached {MAX_DECISION_BYTES // (1024 * 1024)} MB - heartbeats dropped")
            return
        try:
            self._decision_bytes += self._write(self.decisions_path, rec)
        except Exception as exc:  # noqa: BLE001
            self._error("decision log", exc)

    def _write(self, path: Path, rec: dict) -> int:
        line = json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return len(line.encode("utf-8"))

    def _error(self, where: str, exc: Exception) -> None:
        self.errors += 1
        if self.errors <= 20 or self.errors % 100 == 0:
            self.log(f"{TAG} error ({self.errors}) {where}: {type(exc).__name__}: {exc}")
