# -*- coding: utf-8 -*-
"""R024 adaptive exit engine - regime-aware take-profit ladder, dynamic peak-retrace trailing, ratchet stop loss and an
Edge AI adapter whose influence is bounded. Pure logic: no broker, no file I/O, no wall clock (every method takes `now`).

Design: AutoBot/adaptive_exit_design_20261115.md (section 17 overrides earlier sections). Codex review:
AutoBot/adaptive_exit_codex_review_20261115.md.

Units
- Config percentages are PERCENT numbers (2.0 = +2 %). Prices are KRW. All trailing / stop widths are converted to a
  trigger PRICE before comparison (section 17 C9): trailing triggers are rounded DOWN to a valid tick (fires only after
  at least the configured retrace), stop triggers are rounded UP (fires no later than the stop).
- PnL basis (`pnl_basis`): "net" = after fee on both sides and the sell tax, "gross" = price / entry - 1.

Modules (each unit-testable on its own)
- RegimeDetector      NORMAL / SURGE with hysteresis, confirm windows and a minimum dwell time.
- TakeProfitLadder    cumulative-plan partial take profit (40/70/100 % of the entry qty by default).
- TrailingExit        dynamic retrace widths (volatility, RVOL, trade strength, distribution, AI) -> trigger prices.
- StopLoss            ratchet-only stop level (initial ATR stop, breakeven, regime tighten, AI tighten), time stops.
- EdgeAIAdapter       latest-result cache with a worker thread, validation, timeouts and fallback; bounded output.
- ExitDecisionEngine  operating mode, priority arbitration, one intent per evaluation, order event bookkeeping.

The engine never places orders. `step()` returns at most one ExitIntent; the caller submits it and reports back with
on_submitted / on_fill / on_order_closed. Stage flags (ladder step, trail stage) only advance in on_submitted, so a
rejected submission is simply re-evaluated on the next step.

Update log (append only):
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 적응형 익절·손절 엔진 순수 로직(설계 §1~§17). r003/r006 연결 전 단계(EXIT_ENGINE_MODE 미연결).
    impact: none (아직 어떤 실행 경로도 import하지 않음)
    compatibility: backward-compatible
- [2026-11-15] type=fix owner=claude
    summary: Codex 코드 검토(adaptive_exit_codex_code_review_20261115.md) 반영 - 전량 청산 결정 유지(closing_kind, 부분
      체결 후 취소 잔량 재청산), 동결 폭 단조 감소(현재 AI 축소 폭 포함 동결 판정), AI ts 필수·시간대 정규화·입력 나이
      제한·latest() 예외 차단, 정지 감시는 실행 중 작업 기준, 관측 불가 모드 진입 시 확인 타이머 초기화(재시작 포함),
      AI 손절 올림(ceil), 방향 불명 거래량 품질 반영, sync_holding은 미체결 없을 때만.
    impact: none (미연결)
    compatibility: backward-compatible
"""
from __future__ import annotations

import math
import queue
import threading
import time as _time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Callable

STATE_VERSION = 1

# ---------------------------------------------------------------------------
# Tick size helpers (unified KRX table since 2023-01-25; same table as r008.krx_tick_size, duplicated here so this
# module stays import-light for the Pi and for tests)
# ---------------------------------------------------------------------------

_TICK_TABLE = ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500))


def tick_size(price: float) -> int:
    p = float(price)
    for upper, size in _TICK_TABLE:
        if p < upper:
            return size
    return 1000


def floor_tick(price: float) -> float:
    """Largest valid price <= price."""
    p = float(price)
    t = tick_size(p)
    v = math.floor(p / t + 1e-6) * t       # tolerance: 10289.9999999 (float) is 10,290
    return float(v)


def ceil_tick(price: float) -> float:
    """Smallest valid price >= price."""
    p = float(price)
    t = tick_size(p)
    v = float(math.ceil(p / t - 1e-6) * t)
    # crossing a band boundary upwards: 4,998 -> 5,000 is valid (multiple of both 5 and 10)
    return v


def tick_down(price: float, n: int = 1) -> float:
    p = float(price)
    for _ in range(max(0, int(n))):
        p = p - tick_size(p - 1e-6)
    return p


def tick_up(price: float, n: int = 1) -> float:
    p = float(price)
    for _ in range(max(0, int(n))):
        p = p + tick_size(p)
    return p


def ticks_between(low: float, high: float) -> int:
    """Number of tick steps from low up to high (both valid prices, low <= high)."""
    n, p = 0, float(low)
    while p < float(high) - 1e-9 and n < 10000:
        p = tick_up(p)
        n += 1
    return n


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _parse_hhmm(s: str) -> dt_time:
    hh, mm = str(s).split(":")
    return dt_time(int(hh), int(mm))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitConfig:
    # basis / costs (percent)
    pnl_basis: str = "net"
    fee_pct: float = 0.015
    tax_pct: float = 0.20
    # regime
    surge_ret_min_pct: float = 1.0
    surge_ret_atr_mult: float = 2.0
    surge_rvol_min: float = 3.0
    surge_cttr_min: float = 130.0
    surge_min_confirms: int = 2
    surge_enter_confirm_s: float = 5.0
    surge_exit_ret_below_pct: float = 0.3
    surge_exit_rvol_below: float = 1.5
    surge_exit_cttr_below: float = 105.0
    surge_exit_need: int = 2
    surge_exit_confirm_s: float = 30.0
    regime_min_dwell_s: float = 60.0
    use_obi: bool = False
    obi_min: float = 0.0
    # take profit ladder
    tp_normal_pct: tuple[float, ...] = (2.0, 3.0, 4.0)
    tp_surge_pct: tuple[float, ...] = (3.0, 4.0, 5.0)
    tp_ratios: tuple[float, ...] = (0.4, 0.3, 0.3)
    tp_min_interval_s: float = 2.0
    # trailing
    retrace_unit: str = "pnl_pp"          # "pnl_pp" | "peak_rel"
    trail_arm_pct: float = 1.5
    w1_base: float = 0.6
    w2_base: float = 1.1
    stage_a_ratio: float = 0.7
    w_min_pp: float = 0.3
    w_min_ticks: int = 3
    w1_max: float = 1.5
    w2_max: float = 2.5
    ab_gap_ticks: int = 2
    rule_mult_min: float = 0.5
    rule_mult_max: float = 1.5
    freeze_at: float = 0.5
    trail_confirm_s: float = 2.0
    atr_ref_pct: float = 0.45
    m_vol_min: float = 0.7
    m_vol_max: float = 1.5
    rvol_bands: tuple[tuple[float, float], ...] = ((3.0, 1.2), (1.5, 1.0), (1.0, 0.95))  # (>= threshold, mult)
    rvol_below_mult: float = 0.85
    cttr_strong: float = 150.0
    cttr_strong_mult: float = 1.15
    cttr_weak: float = 100.0
    cttr_weak_mult: float = 0.8
    cttr_crash_drop: float = 40.0
    cttr_crash_mult: float = 0.7
    dist_vol_mult: float = 3.0
    dist_sell_share: float = 0.6
    dist_near_peak_s: float = 120.0
    dist_min_ticks_below: int = 2
    dist_mult: float = 0.6
    quality_max_gap: float = 0.10
    quality_min_coverage: float = 0.8
    # stop loss
    sl_atr_mult: float = 2.5
    sl_min_pct: float = 1.2
    sl_max_pct: float = 2.0
    sl_confirm_s: float = 0.0
    sl_catastrophic_pp: float = 0.5
    be_arm_pct: float = 1.0
    be_floor_pct: float = 0.1
    be_confirm_s: float = 3.0
    regime_tighten_peak_min_pct: float = 2.0
    regime_tighten_floor_pct: float = 0.5
    time_stop_min: float = 20.0
    time_stop_peak_max_pct: float = 1.0
    time_stop_pnl_max_pct: float = 0.3
    max_hold_min: float = 60.0
    # edge ai
    ai_mode: str = "shadow"               # "off" | "shadow" | "live"
    ai_min_conf: float = 0.55
    ai_max_latency_ms: float = 50.0
    ai_max_age_s: float = 2.0
    ai_mult_min: float = 0.6
    ai_mult_max: float = 1.4
    ai_mult_gain: float = 0.4
    ai_ratio_min: float = 0.5
    ai_ratio_max: float = 0.9
    ai_ratio_gain: float = 0.2
    ai_ladder_delta: float = 0.1
    ai_sl_rev_min: float = 0.65
    ai_sl_offset_pp: float = 0.3
    ai_stall_factor: float = 5.0
    # execution
    nxt_slip_ticks_trail: int = 2
    nxt_slip_ticks_stop: int = 3
    vi_slip_ticks: int = 2
    min_remainder_krw: float = 30000.0    # 100k (design draft) merged every TRAIL_A on 500k positions (r001 MAX_ORDER_AMOUNT_KRW)
    force_start: str = "15:15"
    force_market: str = "15:18"
    # safety
    tick_stale_s: float = 15.0
    data_dead_s: float = 60.0
    vi_grace_s: float = 10.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ExitConfig":
        if not values:
            return cls()
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown ExitConfig keys: {sorted(unknown)}")
        return cls(**values)

    def validate(self) -> None:
        if self.pnl_basis not in ("net", "gross"):
            raise ValueError("pnl_basis must be net|gross")
        if self.retrace_unit not in ("pnl_pp", "peak_rel"):
            raise ValueError("retrace_unit must be pnl_pp|peak_rel")
        if self.ai_mode not in ("off", "shadow", "live"):
            raise ValueError("ai_mode must be off|shadow|live")
        if not (len(self.tp_normal_pct) == len(self.tp_surge_pct) == len(self.tp_ratios)):
            raise ValueError("tp tables and ratios must have the same length")
        if abs(sum(self.tp_ratios) - 1.0) > 1e-6:
            raise ValueError("tp_ratios must sum to 1")
        if not (0 < self.w1_base <= self.w2_base):
            raise ValueError("0 < w1_base <= w2_base")


# ---------------------------------------------------------------------------
# PnL <-> price
# ---------------------------------------------------------------------------

def pnl_pct(cfg: ExitConfig, entry: float, price: float) -> float:
    e, p = float(entry), float(price)
    if e <= 0:
        return 0.0
    if cfg.pnl_basis == "gross":
        return (p / e - 1.0) * 100.0
    f, t = cfg.fee_pct / 100.0, cfg.tax_pct / 100.0
    return (p * (1.0 - f - t) / (e * (1.0 + f)) - 1.0) * 100.0


def price_for_pnl(cfg: ExitConfig, entry: float, pnl: float) -> float:
    e, r = float(entry), float(pnl) / 100.0
    if cfg.pnl_basis == "gross":
        return e * (1.0 + r)
    f, t = cfg.fee_pct / 100.0, cfg.tax_pct / 100.0
    return (1.0 + r) * e * (1.0 + f) / (1.0 - f - t)


# ---------------------------------------------------------------------------
# Inputs / state / outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSnapshot:
    """One evaluation input (built by r025 live, or by the replay simulator - same code)."""
    ts: datetime
    price: float
    bid1: float = 0.0
    ask1: float = 0.0
    high_since_prev: float | None = None   # highest print since the previous snapshot (peak tracking)
    venue: str = "KRX"                     # "KRX" | "NXT"
    last_tick_age_s: float = 0.0
    rest_age_s: float = 0.0
    ret_3m_pct: float | None = None
    atr1m_pct: float | None = None
    rvol_5m: float | None = None
    cttr_60s: float | None = None
    cttr_60s_prev30: float | None = None   # 60 s trade strength measured 30 s ago (crash detection)
    sell_share_30s: float | None = None
    vol_30s_over_10m: float | None = None
    gap_volume_ratio: float = 0.0          # missed volume / (observed + missed), 60 s
    unknown_side_ratio: float = 0.0        # observed volume without buy/sell side / (observed + missed), 60 s
    window_coverage: float = 1.0
    obi: float | None = None
    vi_active: bool = False
    vi_expected_price: float | None = None
    halted: bool = False
    upper_limit: float | None = None
    lower_limit: float | None = None

    def quality_ok(self, cfg: ExitConfig) -> bool:
        return (self.gap_volume_ratio + self.unknown_side_ratio <= cfg.quality_max_gap
                and self.window_coverage >= cfg.quality_min_coverage)


@dataclass
class ExitState:
    code: str
    entry_price: float
    q0: int
    entry_ts: datetime
    held_qty: int
    sl_price: float
    sl_source: str = "INITIAL"
    sold_qty: int = 0
    pending_qty: int = 0
    pending_kind: str = ""
    pending_priority: int = 0
    pending_order_id: str = ""
    pending_order_type: str = ""
    pending_revert: dict = field(default_factory=dict)
    peak_price: float = 0.0
    peak_ts: datetime | None = None
    be_armed: bool = False
    trail_armed: bool = False
    ladder_step: int = 0
    ladder_plan: tuple[int, ...] = ()
    last_ladder_ts: datetime | None = None
    trail_stage: int = 0                   # 0 none, 1 A done, 2 B done
    frozen_w1: float | None = None
    frozen_w2: float | None = None
    last_w1: float | None = None
    last_w2: float | None = None
    closing_kind: str = ""                 # sticky full-exit decision (STOP / TRAIL_B / ...) until held_qty == 0
    regime: str = "NORMAL"
    regime_since: datetime | None = None
    regime_enter_since: datetime | None = None
    regime_exit_since: datetime | None = None
    regime_switches: int = 0
    regime_tighten_done: bool = False
    timers: dict = field(default_factory=dict)
    at_upper_limit: bool = False
    vi_was_active: bool = False
    vi_released_ts: datetime | None = None
    closed: bool = False
    version: int = STATE_VERSION

    @property
    def sellable_qty(self) -> int:
        return max(0, int(self.held_qty) - int(self.pending_qty))

    @property
    def phase(self) -> str:
        if self.closed or (self.held_qty <= 0 and self.pending_qty <= 0):
            return "CLOSED"
        if self.pending_kind in ("STOP", "FORCE", "TIME_STOP", "MAX_HOLD", "TRAIL_B"):
            return "CLOSING"
        if self.trail_stage >= 1:
            return "TRAIL_A_DONE"
        if self.trail_armed:
            return "TRAIL_ARMED"
        if self.be_armed:
            return "BE_ARMED"
        return "OPEN"

    # --- persistence (section 17 C2) -------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, tuple):
                d[k] = list(v)
        d["timers"] = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.timers.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExitState":
        if int(d.get("version", -1)) != STATE_VERSION:
            raise ValueError(f"ExitState version {d.get('version')} != {STATE_VERSION}")
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        dt_fields = ("entry_ts", "peak_ts", "last_ladder_ts", "regime_since", "regime_enter_since",
                     "regime_exit_since", "vi_released_ts")
        for k in dt_fields:
            if kw.get(k):
                kw[k] = datetime.fromisoformat(kw[k])
        kw["ladder_plan"] = tuple(int(x) for x in kw.get("ladder_plan") or ())
        # confirm windows restart after a restart: the condition was not observed while the bot was down
        kw["timers"] = {}
        return cls(**kw)


@dataclass(frozen=True)
class ExitIntent:
    kind: str              # LADDER | TRAIL_A | TRAIL_B | STOP | TIME_STOP | MAX_HOLD | FORCE
    qty: int               # shares to sell (for replace: computed on held_qty incl. the pending order's shares)
    order_type: str        # "market" | "limit"
    limit_price: float | None
    priority: int
    reason: str
    replace_pending: bool = False
    all_remaining: bool = False   # caller re-reads the real holding after a cancel and sells all of it
    ladder_step: int | None = None


PRIORITY = {"FORCE": 100, "STOP": 90, "TIME_STOP": 80, "MAX_HOLD": 80, "TRAIL_B": 70, "TRAIL_A": 60, "LADDER": 50}
_ALL_KINDS = ("FORCE", "STOP", "TIME_STOP", "MAX_HOLD", "TRAIL_B")


@dataclass(frozen=True)
class AIAdjust:
    status: str                    # ok | off | shadow | timeout | low_conf | invalid | stale | error | disabled | safe
    applied: bool                  # True only when ai_mode == live and status == ok and op mode NORMAL
    m_trail: float = 1.0
    stage_a_ratio: float | None = None
    ladder_delta: float = 0.0      # fraction of q0 added to the NEXT ladder cumulative target
    sl_tighten: bool = False
    raw: dict = field(default_factory=dict)
    would_be: dict = field(default_factory=dict)   # what live mode would have applied (shadow logging)

    @staticmethod
    def neutral(status: str, raw: dict | None = None, would_be: dict | None = None) -> "AIAdjust":
        return AIAdjust(status=status, applied=False, raw=raw or {}, would_be=would_be or {})


@dataclass(frozen=True)
class TrailWidths:
    w1: float
    w2: float
    w1_pre_ai: float
    w2_pre_ai: float
    floor_pp: float
    factors: dict
    frozen: bool
    price_a: float
    price_b: float


@dataclass
class StepResult:
    state: ExitState
    intent: ExitIntent | None
    log: dict


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    def __init__(self, cfg: ExitConfig):
        self.cfg = cfg

    def _surge_condition(self, snap: MarketSnapshot) -> tuple[bool, dict]:
        c = self.cfg
        ok_q = snap.quality_ok(c)
        ret = snap.ret_3m_pct
        need_ret = c.surge_ret_min_pct
        if _finite(snap.atr1m_pct):
            need_ret = max(need_ret, c.surge_ret_atr_mult * float(snap.atr1m_pct))
        speed = _finite(ret) and float(ret) >= need_ret
        confirms = {
            "rvol": ok_q and _finite(snap.rvol_5m) and float(snap.rvol_5m) >= c.surge_rvol_min,
            "cttr": ok_q and _finite(snap.cttr_60s) and float(snap.cttr_60s) >= c.surge_cttr_min,
        }
        if c.use_obi:
            confirms["obi"] = _finite(snap.obi) and float(snap.obi) >= c.obi_min
        n = sum(1 for v in confirms.values() if v)
        return bool(speed and n >= c.surge_min_confirms), {"speed": speed, "need_ret": need_ret, **confirms}

    def _exit_condition(self, snap: MarketSnapshot) -> tuple[bool, dict]:
        c = self.cfg
        # unknown values count as "weak" -> toward NORMAL (conservative)
        weak = {
            "ret": not _finite(snap.ret_3m_pct) or float(snap.ret_3m_pct) < c.surge_exit_ret_below_pct,
            "rvol": not _finite(snap.rvol_5m) or float(snap.rvol_5m) < c.surge_exit_rvol_below,
            "cttr": not _finite(snap.cttr_60s) or float(snap.cttr_60s) < c.surge_exit_cttr_below,
        }
        return sum(1 for v in weak.values() if v) >= c.surge_exit_need, weak

    def initial(self, snap: MarketSnapshot) -> str:
        cond, _ = self._surge_condition(snap)
        return "SURGE" if cond else "NORMAL"

    def update(self, st: ExitState, snap: MarketSnapshot, now: datetime, force_normal: bool = False) -> dict:
        """Mutates st (regime fields). Returns detail for the log."""
        c = self.cfg
        prev = st.regime
        since = st.regime_since or st.entry_ts
        dwell_ok = (now - since).total_seconds() >= c.regime_min_dwell_s
        detail: dict = {}
        if force_normal:
            if st.regime != "NORMAL":
                self._switch(st, "NORMAL", now)
            st.regime_enter_since = st.regime_exit_since = None
            detail["forced"] = True
        elif st.regime == "NORMAL":
            cond, detail = self._surge_condition(snap)
            if cond:
                st.regime_enter_since = st.regime_enter_since or now
                held = (now - st.regime_enter_since).total_seconds()
                if held >= c.surge_enter_confirm_s and dwell_ok:
                    self._switch(st, "SURGE", now)
            else:
                st.regime_enter_since = None
        else:  # SURGE
            if float(snap.price) < float(st.entry_price):
                self._switch(st, "NORMAL", now)      # surge premise broken: immediate (dwell exception)
                detail["below_entry"] = True
            else:
                cond, detail = self._exit_condition(snap)
                if cond:
                    st.regime_exit_since = st.regime_exit_since or now
                    held = (now - st.regime_exit_since).total_seconds()
                    if held >= c.surge_exit_confirm_s and dwell_ok:
                        self._switch(st, "NORMAL", now)
                else:
                    st.regime_exit_since = None
        detail["changed"] = st.regime != prev
        detail["from"] = prev
        return detail

    @staticmethod
    def _switch(st: ExitState, regime: str, now: datetime) -> None:
        st.regime = regime
        st.regime_since = now
        st.regime_enter_since = None
        st.regime_exit_since = None
        st.regime_switches += 1


# ---------------------------------------------------------------------------
# TakeProfitLadder
# ---------------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


class TakeProfitLadder:
    def __init__(self, cfg: ExitConfig):
        self.cfg = cfg

    def plan(self, q0: int) -> tuple[int, ...]:
        """Cumulative target shares per step, fixed at entry (section 17 C13). Last step = q0."""
        cum, out = 0.0, []
        for i, r in enumerate(self.cfg.tp_ratios):
            cum += r
            out.append(int(q0) if i == len(self.cfg.tp_ratios) - 1 else min(int(q0), _round_half_up(q0 * cum)))
        return tuple(out)

    def targets(self, regime: str) -> tuple[float, ...]:
        return self.cfg.tp_surge_pct if regime == "SURGE" else self.cfg.tp_normal_pct

    def evaluate(self, st: ExitState, snap: MarketSnapshot, pnl: float, now: datetime,
                 ai: AIAdjust) -> tuple[ExitIntent | None, bool, dict]:
        """Returns (intent, skip_step, detail). skip_step=True -> the step needs 0 shares and is marked done."""
        c = self.cfg
        n = len(c.tp_ratios)
        if st.trail_stage >= 1 or st.ladder_step >= n:
            return None, False, {"ladder": "inactive"}
        if st.at_upper_limit:
            return None, False, {"ladder": "upper_limit_hold"}
        if st.last_ladder_ts is not None and (now - st.last_ladder_ts).total_seconds() < c.tp_min_interval_s:
            return None, False, {"ladder": "interval"}
        k = st.ladder_step
        target = self.targets(st.regime)[k]
        if pnl < target - 1e-9:
            return None, False, {"ladder": "below", "target": target}
        plan = st.ladder_plan or self.plan(st.q0)
        committed = st.sold_qty + st.pending_qty
        cum = plan[k]
        if ai.applied and k < n - 1 and ai.ladder_delta:
            upper = plan[k + 1]
            cum = int(_clamp(_round_half_up(cum + ai.ladder_delta * st.q0), committed, upper))
        detail = {"ladder": "hit", "step": k + 1, "target": target, "cum": cum}
        if k < n - 1 and cum - committed <= 0:
            return None, True, detail          # plan already covered by sold + in-flight shares: step done
        qty = st.sellable_qty if k == n - 1 else min(cum - committed, st.sellable_qty)
        if qty <= 0:
            return None, False, dict(detail, ladder="no_sellable")   # everything in flight: wait, do not skip
        qty = _merge_remainder(c, qty, st.sellable_qty, snap.price)
        price = _limit_price_ladder(c, st, snap, target)
        return ExitIntent("LADDER", qty, "limit", price, PRIORITY["LADDER"],
                          f"TP{k + 1}_{st.regime}_{target:.1f}PCT", ladder_step=k), False, detail


def _merge_remainder(cfg: ExitConfig, qty: int, sellable: int, price: float) -> int:
    rest = sellable - qty
    if rest > 0 and (rest <= 1 or rest * float(price) < cfg.min_remainder_krw):
        return sellable
    return qty


def _limit_price_ladder(cfg: ExitConfig, st: ExitState, snap: MarketSnapshot, target_pct: float) -> float:
    bid = float(snap.bid1 or 0) or float(snap.price)
    if snap.venue == "NXT":
        return floor_tick(bid)
    tgt = ceil_tick(price_for_pnl(cfg, st.entry_price, target_pct))
    return max(tgt, floor_tick(bid))


# ---------------------------------------------------------------------------
# TrailingExit
# ---------------------------------------------------------------------------

class TrailingExit:
    def __init__(self, cfg: ExitConfig):
        self.cfg = cfg

    def floor_pp(self, st: ExitState) -> float:
        c = self.cfg
        t = tick_size(st.entry_price)
        return max(c.w_min_pp, c.w_min_ticks * t / float(st.entry_price) * 100.0)

    def rule_factors(self, st: ExitState, snap: MarketSnapshot, now: datetime) -> dict:
        c = self.cfg
        ok_q = snap.quality_ok(c)
        m_vol = 1.0
        if _finite(snap.atr1m_pct) and c.atr_ref_pct > 0:
            m_vol = _clamp(float(snap.atr1m_pct) / c.atr_ref_pct, c.m_vol_min, c.m_vol_max)
        # RVOL: full piecewise table (section 17 C10)
        m_rvol = 1.0
        if _finite(snap.rvol_5m):
            r = float(snap.rvol_5m)
            m_rvol = c.rvol_below_mult
            for thr, mult in c.rvol_bands:
                if r >= thr:
                    m_rvol = mult
                    break
        # trade strength: crash rule wins (min of the applicable values)
        m_cttr = 1.0
        if _finite(snap.cttr_60s):
            v = float(snap.cttr_60s)
            m_cttr = c.cttr_strong_mult if v >= c.cttr_strong else (1.0 if v >= c.cttr_weak else c.cttr_weak_mult)
            if _finite(snap.cttr_60s_prev30) and float(snap.cttr_60s_prev30) - v >= c.cttr_crash_drop:
                m_cttr = min(m_cttr, c.cttr_crash_mult)
        # distribution: volume burst near the peak + seller-initiated + price below the peak
        m_dist = 1.0
        near_peak = st.peak_ts is not None and (now - st.peak_ts).total_seconds() <= c.dist_near_peak_s
        below = st.peak_price > 0 and float(snap.price) <= tick_down(st.peak_price, c.dist_min_ticks_below)
        if (near_peak and below and _finite(snap.vol_30s_over_10m) and float(snap.vol_30s_over_10m) >= c.dist_vol_mult
                and _finite(snap.sell_share_30s) and float(snap.sell_share_30s) >= c.dist_sell_share):
            m_dist = c.dist_mult
        if not ok_q:
            # incomplete windows may only tighten (section 17 C11)
            m_rvol, m_cttr = min(m_rvol, 1.0), min(m_cttr, 1.0)
        prod = m_vol * m_rvol * m_cttr * m_dist
        return {"m_vol": m_vol, "m_rvol": m_rvol, "m_cttr": m_cttr, "m_dist": m_dist, "quality_ok": ok_q,
                "m_rule": _clamp(prod, c.rule_mult_min, c.rule_mult_max)}

    def drop(self, st: ExitState, price: float) -> float:
        if st.peak_price <= 0:
            return 0.0
        if self.cfg.retrace_unit == "peak_rel":
            return (st.peak_price - float(price)) / st.peak_price * 100.0
        return pnl_pct(self.cfg, st.entry_price, st.peak_price) - pnl_pct(self.cfg, st.entry_price, price)

    def level_price(self, st: ExitState, w: float) -> float:
        """Exact (unrounded) price at which the retrace equals w."""
        if self.cfg.retrace_unit == "peak_rel":
            return st.peak_price * (1.0 - w / 100.0)
        peak_pnl = pnl_pct(self.cfg, st.entry_price, st.peak_price)
        return price_for_pnl(self.cfg, st.entry_price, peak_pnl - w)

    def _final(self, w1_pre: float, w2_pre: float, m_ai: float, floor_pp: float, safe: bool) -> tuple[float, float]:
        c = self.cfg
        if safe:
            return floor_pp, floor_pp
        w1 = _clamp(w1_pre * m_ai, floor_pp, c.w1_max)
        return w1, _clamp(max(w2_pre * m_ai, w1), floor_pp, c.w2_max)

    def widths(self, st: ExitState, snap: MarketSnapshot, now: datetime, ai: AIAdjust, safe: bool) -> TrailWidths:
        """Order fixed by section 17 C10 (+ code review 2026-11-15 #2): (1) peak already updated by the engine
        (new peak -> unfreeze) (2) rule widths before AI (3) AI (4) freeze test against the SMALLEST of the previous
        final, the pre-AI and the current final w1 (5) while frozen both widths only ever shrink until a new peak:
        every evaluation's narrower result is folded into the frozen caps. Mutates st.frozen_* / last_w*."""
        c = self.cfg
        floor_pp = self.floor_pp(st)
        if safe:
            factors = {"m_vol": 1.0, "m_rvol": 1.0, "m_cttr": 1.0, "m_dist": 1.0, "m_rule": 0.0, "safe": True}
            w1_pre = w2_pre = floor_pp
        else:
            factors = self.rule_factors(st, snap, now)
            m = factors["m_rule"]
            w1_pre = _clamp(c.w1_base * m, floor_pp, c.w1_max)
            w2_pre = _clamp(max(c.w2_base * m, w1_pre), floor_pp, c.w2_max)
        drop = self.drop(st, snap.price)
        m_ai = ai.m_trail if ai.applied else 1.0
        if st.frozen_w1 is not None:
            m_ai = min(m_ai, 1.0)
        w1, w2 = self._final(w1_pre, w2_pre, m_ai, floor_pp, safe)
        if st.frozen_w1 is None:
            ref = min(x for x in (w1, w1_pre, st.last_w1) if x is not None)
            if drop >= c.freeze_at * ref:
                if m_ai > 1.0:                                  # freezing now: AI may not widen this evaluation
                    m_ai = 1.0
                    w1, w2 = self._final(w1_pre, w2_pre, m_ai, floor_pp, safe)
                st.frozen_w1 = w1 if st.last_w1 is None else min(w1, st.last_w1)
                st.frozen_w2 = w2 if st.last_w2 is None else min(w2, st.last_w2)
        if st.frozen_w1 is not None:
            st.frozen_w1 = min(st.frozen_w1, w1)
            st.frozen_w2 = min(st.frozen_w2 if st.frozen_w2 is not None else w2, w2)
            w1, w2 = st.frozen_w1, st.frozen_w2
        w2 = max(w2, w1)
        st.last_w1, st.last_w2 = w1, w2
        pa, pb = self.trigger_prices(st, w1, w2)
        factors = dict(factors, m_ai=m_ai, drop=drop)
        return TrailWidths(w1, w2, w1_pre, w2_pre, floor_pp, factors, st.frozen_w1 is not None, pa, pb)

    def trigger_prices(self, st: ExitState, w1: float, w2: float) -> tuple[float, float]:
        c = self.cfg
        peak = floor_tick(st.peak_price)
        pa = floor_tick(self.level_price(st, w1))
        min_a = tick_down(peak, c.w_min_ticks)
        if pa > min_a:
            pa = min_a
        pb = floor_tick(self.level_price(st, w2))
        max_b = tick_down(pa, c.ab_gap_ticks)
        if pb > max_b:
            pb = max_b
        return pa, pb


# ---------------------------------------------------------------------------
# StopLoss
# ---------------------------------------------------------------------------

class StopLoss:
    def __init__(self, cfg: ExitConfig):
        self.cfg = cfg

    def initial_price(self, entry: float, atr1m_pct: float | None) -> tuple[float, float]:
        c = self.cfg
        sl = c.sl_max_pct if not _finite(atr1m_pct) else _clamp(c.sl_atr_mult * float(atr1m_pct), c.sl_min_pct, c.sl_max_pct)
        return ceil_tick(price_for_pnl(c, entry, -sl)), sl

    @staticmethod
    def _raise(st: ExitState, price: float, source: str) -> bool:
        """The ONLY place the stop level changes: ratchet up, never down."""
        if price > st.sl_price + 1e-9:
            st.sl_price = price
            st.sl_source = source
            return True
        return False

    def update_level(self, st: ExitState, pnl: float, peak_pnl: float, regime_detail: dict, ai: AIAdjust) -> list[str]:
        c = self.cfg
        changed = []
        if not st.be_armed and peak_pnl >= c.be_arm_pct:
            st.be_armed = True
        if st.be_armed and self._raise(st, ceil_tick(price_for_pnl(c, st.entry_price, c.be_floor_pct)), "BREAKEVEN"):
            changed.append("BREAKEVEN")
        if (regime_detail.get("changed") and regime_detail.get("from") == "SURGE" and st.regime == "NORMAL"
                and peak_pnl >= c.regime_tighten_peak_min_pct and not st.regime_tighten_done):
            st.regime_tighten_done = True
            if self._raise(st, ceil_tick(price_for_pnl(c, st.entry_price, c.regime_tighten_floor_pct)), "REGIME"):
                changed.append("REGIME")
        if ai.applied and ai.sl_tighten and pnl < 0:
            lvl = ceil_tick(price_for_pnl(c, st.entry_price, pnl - c.ai_sl_offset_pp))   # stops round UP (C9)
            if self._raise(st, lvl, "AI"):
                changed.append("AI")
        return changed

    def evaluate(self, st: ExitState, snap: MarketSnapshot, now: datetime, pnl: float, peak_pnl: float,
                 confirm_mult: float = 1.0) -> tuple[str | None, str]:
        """Returns (kind, reason) with kind in STOP / TIME_STOP / MAX_HOLD or None. Uses st.timers."""
        c = self.cfg
        price = float(snap.price)
        sl_pnl = pnl_pct(c, st.entry_price, st.sl_price)
        catastrophic = price <= floor_tick(price_for_pnl(c, st.entry_price, sl_pnl - c.sl_catastrophic_pp))
        confirm = (c.be_confirm_s if st.sl_source == "BREAKEVEN" else c.sl_confirm_s) * confirm_mult
        if price <= st.sl_price + 1e-9:
            if catastrophic or _timer_hit(st, "STOP", now, confirm):
                return "STOP", f"STOP_{st.sl_source}_{sl_pnl:+.2f}PCT"
        else:
            st.timers.pop("STOP", None)
        held_min = (now - st.entry_ts).total_seconds() / 60.0
        if held_min >= c.time_stop_min and peak_pnl < c.time_stop_peak_max_pct and pnl < c.time_stop_pnl_max_pct:
            return "TIME_STOP", f"TIME_STOP_{c.time_stop_min:.0f}MIN"
        if held_min >= c.max_hold_min and not st.trail_armed:
            return "MAX_HOLD", f"MAX_HOLD_{c.max_hold_min:.0f}MIN"
        return None, ""


def _timer_hit(st: ExitState, key: str, now: datetime, confirm_s: float) -> bool:
    if confirm_s <= 0:
        st.timers.pop(key, None)
        return True
    since = st.timers.get(key)
    if since is None:
        st.timers[key] = now
        return False
    return (now - since).total_seconds() >= confirm_s


# ---------------------------------------------------------------------------
# EdgeAIAdapter
# ---------------------------------------------------------------------------

@dataclass
class _AIResult:
    seq: int
    snapshot_ts: datetime
    raw: dict
    latency_ms: float
    received_mono: float
    error: str = ""


class EdgeAIAdapter:
    """Latest-result cache in front of an exit model (section 6 + section 17 C14).

    predict_fn(features: dict) -> dict with continuation_prob, reversal_prob, confidence, model_version, ts.
    - One worker thread, queue size 1 (only the newest request is kept), per-code request sequence numbers; a result
      older than the newest accepted one for that code is dropped.
    - latest() never raises and never blocks; anything unusual -> neutral AIAdjust (rule defaults).
    - inline=True runs predict_fn synchronously inside submit() (tests / replay)."""

    def __init__(self, cfg: ExitConfig, predict_fn: Callable[[dict], dict] | None,
                 model_version: str | None = None, inline: bool = False,
                 mono: Callable[[], float] = _time.monotonic):
        self.cfg = cfg
        self.predict_fn = predict_fn
        self.model_version = model_version
        self.inline = inline
        self._mono = mono
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._results: dict[str, _AIResult] = {}
        self._running: tuple[str, int, float] | None = None   # (code, seq, mono start) of the job being executed
        self.disabled_reason = ""
        self.closed = False
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        if predict_fn is not None and not inline and cfg.ai_mode != "off":
            self._thread = threading.Thread(target=self._worker, name="exit-ai", daemon=True)
            self._thread.start()

    # -- producer ----------------------------------------------------------------------------------
    def submit(self, code: str, features: dict, snapshot_ts: datetime) -> int:
        if self.predict_fn is None or self.cfg.ai_mode == "off" or self.disabled_reason or self.closed:
            return -1
        with self._lock:
            seq = self._seq.get(code, 0) + 1
            self._seq[code] = seq
        job = (code, seq, dict(features), snapshot_ts)
        if self.inline:
            self._run(job)
            return seq
        try:
            self._q.put_nowait(job)
        except queue.Full:
            try:
                self._q.get_nowait()          # drop the older pending request
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(job)
            except queue.Full:
                pass
        return seq

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop.is_set():
                return
            self._run(job)

    def _run(self, job) -> None:
        code, seq, feats, snap_ts = job
        t0 = self._mono()
        with self._lock:
            self._running = (code, seq, t0)
        raw, err = {}, ""
        try:
            raw = dict(self.predict_fn(feats) or {})
        except Exception as exc:  # noqa: BLE001 - model errors must never reach the engine
            err = f"{type(exc).__name__}: {exc}"
        t1 = self._mono()
        res = _AIResult(seq, snap_ts, raw, (t1 - t0) * 1000.0, t1, err)
        with self._lock:
            self._running = None
            cur = self._results.get(code)
            if cur is None or seq > cur.seq:
                self._results[code] = res

    def close(self, timeout_s: float = 1.0) -> bool:
        """Stops accepting work and stops the worker. Returns True when the worker has exited within timeout_s
        (a model call that never returns keeps the daemon thread alive; it is then abandoned, never joined)."""
        self.closed = True
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    # -- consumer ----------------------------------------------------------------------------------
    def latest(self, code: str, now: datetime) -> AIAdjust:
        """Never raises: any unexpected failure -> neutral 'error' (rule defaults)."""
        try:
            return self._latest(code, now)
        except Exception as exc:  # noqa: BLE001
            return AIAdjust.neutral("error", {"error": f"{type(exc).__name__}: {exc}"})

    def _latest(self, code: str, now: datetime) -> AIAdjust:
        c = self.cfg
        if c.ai_mode == "off" or self.predict_fn is None:
            return AIAdjust.neutral("off")
        if self.disabled_reason:
            return AIAdjust.neutral("disabled", {"reason": self.disabled_reason})
        with self._lock:
            res = self._results.get(code)
            running = self._running
        mono_now = self._mono()
        # stall = the job actually executing has run too long (queued / dropped requests do not count).
        # Re-checked under the lock: the worker may have finished between the first read and now (re-review #3).
        if running is not None and (mono_now - running[2]) > c.ai_max_age_s * c.ai_stall_factor:
            with self._lock:
                if self._running is running:
                    self.disabled_reason = f"worker stalled {mono_now - running[2]:.1f}s on {running[0]}#{running[1]}"
            if self.disabled_reason:
                return AIAdjust.neutral("disabled", {"reason": self.disabled_reason})
            with self._lock:
                res = self._results.get(code)
        if res is None:
            return AIAdjust.neutral("stale", {"reason": "no result"})
        raw = dict(res.raw)
        if res.error:
            return AIAdjust.neutral("error", {"error": res.error})
        if not _finite(res.latency_ms) or res.latency_ms < 0 or res.latency_ms > c.ai_max_latency_ms:
            return AIAdjust.neutral("timeout", dict(raw, latency_ms=res.latency_ms))
        if (mono_now - res.received_mono) > c.ai_max_age_s:
            return AIAdjust.neutral("stale", dict(raw, age_s=mono_now - res.received_mono))
        input_age = (_to_local_naive(now) - _to_local_naive(res.snapshot_ts)).total_seconds()
        if input_age > c.ai_max_age_s or input_age < -1.0:
            return AIAdjust.neutral("stale", dict(raw, input_age_s=input_age))   # judged on an old snapshot
        bad = self._invalid(raw, res.snapshot_ts, now)
        if bad:
            return AIAdjust.neutral("invalid", dict(raw, why=bad))
        cont, rev, conf = float(raw["continuation_prob"]), float(raw["reversal_prob"]), float(raw["confidence"])
        raw = dict(raw, latency_ms=res.latency_ms)
        if conf < c.ai_min_conf:
            return AIAdjust.neutral("low_conf", raw)
        d = (cont - rev) * conf
        values = {
            "m_trail": _clamp(1.0 + c.ai_mult_gain * d, c.ai_mult_min, c.ai_mult_max),
            "stage_a_ratio": _clamp(c.stage_a_ratio - c.ai_ratio_gain * d, c.ai_ratio_min, c.ai_ratio_max),
            "ladder_delta": _clamp(c.ai_ladder_delta * (rev - cont) * conf, -c.ai_ladder_delta, c.ai_ladder_delta),
            "sl_tighten": rev >= c.ai_sl_rev_min,
        }
        if c.ai_mode != "live":
            return AIAdjust.neutral("shadow", raw, values)
        return AIAdjust(status="ok", applied=True, raw=raw, would_be=values, **values)

    def _invalid(self, raw: dict, snap_ts: datetime, now: datetime) -> str:
        for k in ("continuation_prob", "reversal_prob", "confidence"):
            v = raw.get(k)
            if not _finite(v):
                return f"{k} not finite"
            if not (0.0 <= float(v) <= 1.0):
                return f"{k} out of range"
        if float(raw["continuation_prob"]) + float(raw["reversal_prob"]) > 1.0 + 1e-6:
            return "cont+rev > 1"
        if self.model_version and raw.get("model_version") != self.model_version:
            return "model_version mismatch"
        ts = _to_local_naive(raw.get("ts"))
        if ts is None:
            return "ts missing or unparsable"
        # 1 s tolerance: models may stamp whole seconds while snapshots carry microseconds
        if (_to_local_naive(snap_ts) - ts).total_seconds() > 1.0 or (ts - _to_local_naive(now)).total_seconds() > 1.0:
            return "ts out of order"
        if (_to_local_naive(now) - ts).total_seconds() > self.cfg.ai_max_age_s:
            return "ts too old"
        return ""


_KST = timezone(timedelta(hours=9))


def _to_local_naive(value: Any) -> datetime | None:
    """datetime / ISO string -> naive KST datetime (the repo's clock convention). None if missing / unparsable."""
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is not None:
            value = value.astimezone(_KST).replace(tzinfo=None)
        return value
    except (ValueError, TypeError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# ExitDecisionEngine
# ---------------------------------------------------------------------------

class ExitDecisionEngine:
    def __init__(self, cfg: ExitConfig | None = None):
        self.cfg = cfg or ExitConfig()
        self.cfg.validate()
        self.regime = RegimeDetector(self.cfg)
        self.ladder = TakeProfitLadder(self.cfg)
        self.trail = TrailingExit(self.cfg)
        self.stop = StopLoss(self.cfg)
        self._force_start = _parse_hhmm(self.cfg.force_start)
        self._force_market = _parse_hhmm(self.cfg.force_market)

    # -- lifecycle ---------------------------------------------------------------------------------
    def open_position(self, code: str, entry_price: float, qty: int, now: datetime,
                      snap: MarketSnapshot | None = None) -> ExitState:
        sl_price, _ = self.stop.initial_price(entry_price, snap.atr1m_pct if snap else None)
        st = ExitState(code=code, entry_price=float(entry_price), q0=int(qty), entry_ts=now, held_qty=int(qty),
                       sl_price=sl_price, peak_price=float(entry_price), peak_ts=now,
                       ladder_plan=self.ladder.plan(int(qty)), regime_since=now)
        if snap is not None:
            st.regime = self.regime.initial(snap)
        return st

    def rebase_position(self, st: ExitState, entry_price: float, held_qty: int, now: datetime,
                        snap: MarketSnapshot | None = None) -> ExitState:
        """Additional bot buy into a managed position (section 17 C15): new q0/avg price/ladder plan for the
        remaining quantity; stop PRICE ratchets (never below the previous level)."""
        st = replace(st, timers=dict(st.timers))
        new_sl, _ = self.stop.initial_price(entry_price, snap.atr1m_pct if snap else None)
        st.entry_price, st.q0, st.held_qty = float(entry_price), int(held_qty), int(held_qty)
        st.sold_qty, st.ladder_step, st.ladder_plan = 0, 0, self.ladder.plan(int(held_qty))
        StopLoss._raise(st, new_sl, st.sl_source)
        return st

    def on_submitted(self, st: ExitState, intent: ExitIntent, order_qty: int, order_id: str, now: datetime) -> ExitState:
        st = replace(st, timers=dict(st.timers), pending_revert=dict(st.pending_revert))
        st.pending_revert = {"ladder_step": st.ladder_step, "trail_stage": st.trail_stage,
                             "closing_kind": st.closing_kind,
                             "last_ladder_ts": st.last_ladder_ts.isoformat() if st.last_ladder_ts else None}
        if intent.kind in _ALL_KINDS and PRIORITY[intent.kind] >= PRIORITY.get(st.closing_kind, 0):
            st.closing_kind = intent.kind
        st.pending_qty += int(order_qty)
        st.pending_kind, st.pending_priority, st.pending_order_id = intent.kind, intent.priority, order_id
        st.pending_order_type = intent.order_type
        if intent.kind == "LADDER" and intent.ladder_step is not None:
            st.ladder_step = intent.ladder_step + 1
            st.last_ladder_ts = now
        elif intent.kind == "TRAIL_A":
            st.trail_stage = 1
        elif intent.kind in _ALL_KINDS:
            st.trail_stage = max(st.trail_stage, 2) if intent.kind == "TRAIL_B" else st.trail_stage
        st.timers.pop(intent.kind, None)
        return st

    def on_fill(self, st: ExitState, filled_increment: int) -> ExitState:
        """Report only the INCREASE of the order's cumulative filled quantity (section 17 C5)."""
        n = max(0, int(filled_increment))
        st = replace(st, timers=dict(st.timers))
        st.sold_qty += n
        st.held_qty = max(0, st.held_qty - n)
        st.pending_qty = max(0, st.pending_qty - n)
        if st.pending_qty == 0:   # order fully done
            st.pending_kind, st.pending_priority, st.pending_order_id, st.pending_revert = "", 0, "", {}
            st.pending_order_type = ""
        if st.held_qty == 0 and st.pending_qty == 0:
            st.closed = True
        return st

    def on_order_closed(self, st: ExitState, unfilled_qty: int, revert_stage: bool) -> ExitState:
        """Order reached a terminal state (cancel confirmed / rejected / done). revert_stage=True when nothing of it
        filled and the caller is NOT replacing it - the stage it advanced is undone so the signal can fire again."""
        st = replace(st, timers=dict(st.timers))
        st.pending_qty = max(0, st.pending_qty - max(0, int(unfilled_qty)))
        if revert_stage and st.pending_revert:
            st.ladder_step = int(st.pending_revert.get("ladder_step", st.ladder_step))
            st.trail_stage = int(st.pending_revert.get("trail_stage", st.trail_stage))
            st.closing_kind = str(st.pending_revert.get("closing_kind", st.closing_kind) or "")
            ts = st.pending_revert.get("last_ladder_ts")
            st.last_ladder_ts = datetime.fromisoformat(ts) if ts else None
        if st.pending_qty == 0:
            st.pending_kind, st.pending_priority, st.pending_order_id, st.pending_revert = "", 0, "", {}
            st.pending_order_type = ""
        if st.held_qty == 0 and st.pending_qty == 0:
            st.closed = True
        return st

    def sync_holding(self, st: ExitState, held_qty: int) -> ExitState:
        """Real (bot) holding re-read after a cancel or on restart reconciliation. Manual sells lower it.

        Contract (code review 2026-11-15, regression risk 2): call it only when NO engine order is in flight and every
        fill of the closed orders has already been reported with on_fill - otherwise a late fill report is counted
        twice. The order manager (r026) enforces this."""
        if st.pending_qty > 0:
            raise ValueError("sync_holding while an order is in flight")
        st = replace(st, timers=dict(st.timers))
        h = max(0, int(held_qty))
        if h < st.held_qty:
            st.sold_qty += st.held_qty - h
        st.held_qty = h
        st.pending_qty = min(st.pending_qty, h)
        st.closed = h == 0 and st.pending_qty == 0
        return st

    # -- evaluation --------------------------------------------------------------------------------
    def op_mode(self, snap: MarketSnapshot) -> str:
        c = self.cfg
        if snap.halted:
            return "FROZEN"
        if min(snap.last_tick_age_s, snap.rest_age_s) >= c.data_dead_s:
            return "DEAD"
        if snap.vi_active:
            return "VI"
        if snap.last_tick_age_s >= c.tick_stale_s:
            return "SAFE"
        return "NORMAL"

    def step(self, st: ExitState, snap: MarketSnapshot, now: datetime, ai: AIAdjust | None = None) -> StepResult:
        c = self.cfg
        st = replace(st, timers=dict(st.timers), pending_revert=dict(st.pending_revert))
        mode = self.op_mode(snap)
        log: dict = {"ts": now.isoformat(), "code": st.code, "op_mode": mode, "price": snap.price,
                     "state_prev": st.phase, "held_qty": st.held_qty, "pending_qty": st.pending_qty}
        if st.phase == "CLOSED":
            log["skip"] = "closed"
            return StepResult(st, None, log)
        ai_in = ai or AIAdjust.neutral("off")
        ai_eff = ai_in if (mode == "NORMAL" and ai_in.applied) else AIAdjust.neutral(
            ai_in.status if mode == "NORMAL" else "safe", ai_in.raw, ai_in.would_be)

        # VI transitions
        if st.vi_was_active and not snap.vi_active:
            st.vi_released_ts = now
        st.vi_was_active = snap.vi_active
        in_grace = st.vi_released_ts is not None and (now - st.vi_released_ts).total_seconds() < c.vi_grace_s
        confirm_mult = 2.0 if in_grace else 1.0

        price_usable = mode in ("NORMAL", "SAFE", "VI")
        # confirm windows require the condition to be OBSERVED continuously (code review #8)
        if not price_usable:
            st.timers.clear()
        elif mode == "VI":
            st.timers.pop("TRAIL_A", None)
            st.timers.pop("TRAIL_B", None)
        # (1) peak
        new_peak = False
        if price_usable:
            hi = max(float(snap.price), float(snap.high_since_prev or 0.0))
            if hi > st.peak_price + 1e-9:
                st.peak_price, st.peak_ts, new_peak = hi, now, True
                # widths belonged to the previous peak: unfreeze AND forget them (re-review #2)
                st.frozen_w1 = st.frozen_w2 = None
                st.last_w1 = st.last_w2 = None
        pnl = pnl_pct(c, st.entry_price, snap.price)
        peak_pnl = pnl_pct(c, st.entry_price, st.peak_price)
        # upper limit tracking
        was_upper = st.at_upper_limit
        st.at_upper_limit = bool(snap.upper_limit) and float(snap.price) >= float(snap.upper_limit) - 1e-9
        left_upper = was_upper and not st.at_upper_limit and price_usable
        # (2) regime
        regime_detail = self.regime.update(st, snap, now, force_normal=(mode != "NORMAL")) if price_usable else {}
        # (3) stop ratchet
        sl_changes = self.stop.update_level(st, pnl, peak_pnl, regime_detail, ai_eff) if price_usable else []
        if peak_pnl >= c.trail_arm_pct:
            st.trail_armed = True
        log.update({"pnl": round(pnl, 4), "peak_price": st.peak_price, "peak_pnl": round(peak_pnl, 4),
                    "new_peak": new_peak, "regime": st.regime, "regime_detail": regime_detail,
                    "sl_price": st.sl_price, "sl_source": st.sl_source, "sl_changes": sl_changes,
                    "ai_status": ai_in.status, "ai_applied": ai_eff.applied, "ai_raw": ai_in.raw,
                    "ai_would_be": ai_in.would_be})

        cands: list[ExitIntent] = []
        # FORCE (time based, runs even with dead data - section 17 C7 / ※2)
        t = now.time()
        if t >= self._force_start and st.held_qty > 0:
            market = t >= self._force_market or mode in ("DEAD", "SAFE")
            cands.append(self._make(st, snap, "FORCE", st.held_qty, "FORCE_EXIT", market_ok=market, mode=mode))

        widths = None
        if price_usable and st.closing_kind and st.held_qty > 0 and st.pending_qty == 0:
            # a full-exit decision whose order ended with shares left (partial fill + cancel): finish it (review #1)
            ck = st.closing_kind
            if ck != "FORCE" and (mode != "VI" or ck == "STOP"):
                cands.append(self._make(st, snap, ck, st.held_qty, f"{ck}_RESUME", mode=mode))
        if price_usable:
            kind, reason = self.stop.evaluate(st, snap, now, pnl, peak_pnl, confirm_mult)
            if kind == "STOP" or (kind and mode != "VI"):
                cands.append(self._make(st, snap, kind, st.held_qty, reason, mode=mode))
            if mode != "VI" and st.trail_armed:
                widths = self.trail.widths(st, snap, now, ai_eff, safe=(mode == "SAFE"))
                log.update({"w1": widths.w1, "w2": widths.w2, "w1_pre_ai": widths.w1_pre_ai,
                            "w2_pre_ai": widths.w2_pre_ai, "floor_pp": widths.floor_pp, "frozen": widths.frozen,
                            "trail_price_a": widths.price_a, "trail_price_b": widths.price_b, **widths.factors})
                px = float(snap.price)
                trail_confirm = c.trail_confirm_s * confirm_mult
                if st.trail_stage < 2 and px <= widths.price_b + 1e-9:
                    if _timer_hit(st, "TRAIL_B", now, trail_confirm):
                        cands.append(self._make(st, snap, "TRAIL_B", st.held_qty, "TRAIL_B", mode=mode))
                else:
                    st.timers.pop("TRAIL_B", None)
                a_hit = px <= widths.price_a + 1e-9 or left_upper
                if st.trail_stage == 0 and a_hit:
                    if left_upper or _timer_hit(st, "TRAIL_A", now, trail_confirm):
                        ratio = ai_eff.stage_a_ratio if (ai_eff.applied and ai_eff.stage_a_ratio) else c.stage_a_ratio
                        base = st.held_qty  # pending (if any) is cancelled when this outranks it
                        qty = _round_half_up(base * ratio)
                        if base - qty <= 0 or qty <= 0 or base <= 1:
                            qty = base
                        qty = _merge_remainder(c, qty, base, px)
                        reason = "TRAIL_A_UPPER_LIMIT_BREAK" if left_upper else "TRAIL_A"
                        cands.append(self._make(st, snap, "TRAIL_A", qty, reason, mode=mode,
                                                all_remaining=(qty >= base)))
                elif st.trail_stage == 0:
                    st.timers.pop("TRAIL_A", None)
            if mode != "VI":
                lad, skip, ldetail = self.ladder.evaluate(st, snap, pnl, now, ai_eff)
                log["ladder_detail"] = ldetail
                if skip:
                    st.ladder_step += 1
                    st.last_ladder_ts = now
                    log["ladder_skipped_step"] = st.ladder_step
                elif lad is not None:
                    cands.append(lad)
        else:
            log["price_signals"] = "suppressed"

        intent = self._arbitrate(st, cands)
        log.update({"candidates": [(x.kind, x.qty, x.priority) for x in cands],
                    "chosen": intent.kind if intent else None,
                    "intent": asdict(intent) if intent else None,
                    "ladder_step": st.ladder_step, "trail_stage": st.trail_stage, "state": st.phase})
        return StepResult(st, intent, log)

    def _arbitrate(self, st: ExitState, cands: list[ExitIntent]) -> ExitIntent | None:
        if not cands:
            return None
        best = max(cands, key=lambda x: x.priority)
        if st.pending_qty > 0:
            escalate = (best.kind == "FORCE" and st.pending_kind == "FORCE" and best.order_type == "market"
                        and st.pending_order_type == "limit")      # 15:18: limit force exit -> market
            if best.priority <= st.pending_priority and not escalate:
                return None                   # same/lower priority while an order is in flight: wait
            return replace(best, replace_pending=True,
                           all_remaining=best.all_remaining or best.kind in _ALL_KINDS)
        if best.qty > st.sellable_qty:
            best = replace(best, qty=st.sellable_qty)
        if best.qty <= 0:
            return None
        return best

    def _make(self, st: ExitState, snap: MarketSnapshot, kind: str, qty: int, reason: str, mode: str,
              market_ok: bool = True, all_remaining: bool | None = None) -> ExitIntent:
        c = self.cfg
        bid = float(snap.bid1 or 0) or float(snap.price)
        order_type, limit = "market", None
        if mode == "VI":
            # section 17 C6 / ※1: limit min(stop, expected - 2 ticks) >= lower limit; unknown expected -> market
            if _finite(snap.vi_expected_price) and float(snap.vi_expected_price) > 0:
                lvl = min(st.sl_price if kind == "STOP" else float("inf"),
                          tick_down(floor_tick(float(snap.vi_expected_price)), c.vi_slip_ticks))
                if snap.lower_limit:
                    lvl = max(lvl, float(snap.lower_limit))
                order_type, limit = "limit", floor_tick(lvl)
        elif snap.venue == "NXT":
            slip = c.nxt_slip_ticks_stop if kind in ("STOP", "FORCE") else c.nxt_slip_ticks_trail
            order_type, limit = "limit", tick_down(floor_tick(bid), slip)
        elif kind == "FORCE" and not market_ok:
            order_type, limit = "limit", floor_tick(bid)
        if all_remaining is None:
            all_remaining = kind in _ALL_KINDS
        return ExitIntent(kind, int(qty), order_type, limit, PRIORITY[kind], reason, all_remaining=all_remaining)
