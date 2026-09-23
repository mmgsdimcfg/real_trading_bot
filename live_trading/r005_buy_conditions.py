"""R005 live buy conditions - numbered checklist for a NEW entry (used by the r003 live executor).

Overview
- r003 used to decide a new entry with ~200 inline lines inside run(). Those checks now live here, one object per
  condition, named `_001_...`, `_002_...`, ... in the exact order they are evaluated.
- `check_buy_conditions(ctx)` is the main function: it walks BUY_CONDITIONS in order and stops at the first condition
  that does not pass. A condition returns True (passed) or False (stop). A failing condition logs / mutates state at
  the same point, with the same text, as the old inline code did.
- The numeric logic of the 1-min trigger and of the 3-min context gates stays in r002 (shared with the g003
  backtester); the `_008`..`_017` conditions only delegate to it. The refactor itself changed no threshold.
- [2026-09-22] `_020` now also requires the same 3-min bar and a bounded price drift between the two consecutive
  passes, and its gap window is 45 s (r001 BUY_CONFIRM_MAX_GAP_SECONDS / _REQUIRE_SAME_BAR / _MAX_PRICE_DRIFT_PCT).
  That is the only behavior change in this file since the refactor.
- No broker imports: everything that talks to the broker / clock is injected through `BuyServices` and `ctx.api`, so the
  whole checklist can be unit-tested without authenticating.

Order (stage)                                     Old r003 inline block
  STATE     _001 entry time window                is_new_entry_allowed
            _002 session open warm-up             STARTUP_WARMUP_SECONDS
            _003 trade cool-down                  api._in_cooldown
            _004 same-bar signal dedup            signal_buy_bar
            _005 no open buy exposure             api.has_buy_exposure
            _006 hard-stop re-entry block         hard_stop_today_codes
            _007 stop-loss circuit breaker        RiskState.circuit_breaker_until
  TRIGGER   _008 hybrid 1-min trigger             check_buy_condition_1min_hybrid_trigger
  CONTEXT   _009 3-min context frame ready        run_3min_context_pipeline (prefix)
            _010 bb_slope_rising                  HYBRID_3MIN_CONTEXT_GATES
            _011 bb_mid_downtrend_block           (skipped when ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK=False)
            _012 bb_upper_gap_min
            _013 stochastic_buy_signal
            _014 min_entry_atr_volatility
            _015 min_liquidity_safety
            _016 di_spread_min
            _017 context score threshold          run_3min_context_pipeline (suffix)
  PRE_ORDER _018 excessive rise from prev close   MAX_BUY_RISE_PCT_FROM_PREV_CLOSE
            _019 opening gap / volume gate        passes_opening_gap_volume_gate
            _020 consecutive poll confirm         BUY_CONSECUTIVE_CONFIRM_COUNT
            _021 affordable buy qty               api.get_affordable_buy_qty
            _022 fresh live price                 _is_stale_live_price_source
            _023 order-book ask not thin          _fetch_orderbook_totals (+ reservation, see docstring)

Behavior notes kept from the old inline code (do not "fix" them here - they are strategy decisions):
- _001~_004 rejects are silent and do NOT clear the confirm state; _008~_019 rejects clear it; _021/_022/_023 rejects and
  a failed order do NOT clear it.
- _020 rewrites the confirm state on every pass, also once the count is >= BUY_CONSECUTIVE_CONFIRM_COUNT.
- The 1-min frame fetch (_008), prev-close fetch (_018), buying-power query (_021) and order-book query (_023) are
  API calls: they run only when reached, in this order.

Update log (append only):
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("매수/매도 컨셉 재검토", 204620 글로벌텍스프리 09:17 매수 지연 사례) + Codex
      설계검토 - _008_hybrid_1min_trigger에 진입 이벤트 유효기한 추가. update_timed_condition_state()로
      1분봉 트리거가 "계속 유효" 상태로 지속된 시간을 재고, HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS(r001,
      기본 300초) 초과 시 HYBRID_1MIN_TRIGGER_EXPIRED_*로 반려한다. 신규 상태 BuyState.buy_trigger_age_state
      추가(r003에서 주입/정리). r001 Update log 2026-09-23 참조.
    impact: live (r003), g003도 동일 로직 적용(백테스트 정합성)
    compatibility: breaking (HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS<=0이면 기존과 동일)
- [2026-09-21] type=refactor owner=claude
    summary: r003 run()의 신규 매수 판정 인라인 블록을 번호 붙은 조건 객체(_001~_023)와 메인 함수
      check_buy_conditions()로 분리. 판정 순서/로그 문구/상태 변경/API 호출 시점은 기존과 동일(동작 불변).
      Codex 사전 검토 반영: STEPS 진단은 기존과 같은 지점(_008, 트리거 판정 직후)에서 계산, 연속 손절 상태는
      run() 수명 동안 하나인 RiskState 인스턴스로 공유, session/buy_detail은 예약 직전에 생성.
    impact: live (r003)
    compatibility: backward-compatible
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from r001_define_config import (
    BUY_CONFIRM_MAX_GAP_SECONDS,
    BUY_CONFIRM_MAX_PRICE_DRIFT_PCT,
    BUY_CONFIRM_REQUIRE_SAME_BAR,
    BUY_CONSECUTIVE_CONFIRM_COUNT,
    ENABLE_OPENING_GAP_VOLUME_GATE,
    HARD_STOP_BLOCK_REENTRY_TODAY,
    HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS,
    LIVE_PRICE_STALE_TTL_SECONDS,
    MAX_BUY_RISE_PCT_FROM_PREV_CLOSE,
    STARTUP_WARMUP_SECONDS,
)
from r002_strategy_core_shared import (
    HYBRID_3MIN_CONTEXT_GATES,
    _compute_bb_slope_pct,
    _evaluate_bb_mid_cross,
    _num,
    build_context_eval,
    check_buy_condition_1min_hybrid_trigger,
    evaluate_context_score,
    gate_steps_diagnostic,
    passes_opening_gap_volume_gate,
    update_timed_condition_state,
)


# ---------------------------------------------------------------------------
# Context / services / decision
# ---------------------------------------------------------------------------

@dataclass
class RiskState:
    """당일 손절 누적 상태. r003 run() 수명 동안 인스턴스 하나를 매수(_007 읽기)/매도(손절 조건 갱신)가 공유한다.

    기존에는 run()의 재바인딩되는 지역변수 2개(hard_stop_daily_count, hard_stop_circuit_breaker_until)였다.
    날짜가 바뀌면 reset()으로 초기화한다(새 객체로 갈아끼우지 않는다 - 양쪽이 같은 인스턴스를 봐야 함)."""

    hard_stop_daily_count: int = 0
    circuit_breaker_until: datetime | None = None

    def reset(self) -> None:
        self.hard_stop_daily_count = 0
        self.circuit_breaker_until = None


@dataclass
class BuyState:
    """r003 run()의 가변 상태 컨테이너들(참조 전달 - 조건이 직접 변경). 틱마다 새로 재바인딩되는 traded_today를
    쓰므로 종목 순회마다 BuyState를 새로 만든다."""

    buy_confirm_state: dict[str, dict[str, object]]
    buy_trigger_age_state: dict[str, dict[str, object]]
    signal_buy_bar: dict[str, object]
    hard_stop_today_codes: set[str]
    gap_blocked_codes: set[str]
    traded_today: set[str]


@dataclass
class BuyServices:
    """브로커/시간/로그에 의존하는 함수들(r003이 주입). 이 모듈은 브로커를 import하지 않는다."""

    log: Callable[[str], None]
    is_new_entry_allowed: Callable[[datetime, bool], bool]
    get_session_open_datetime: Callable[[datetime, bool], datetime | None]
    get_frame_1min: Callable[[str, datetime, bool], pd.DataFrame | None]
    fetch_prev_close: Callable[[str, datetime, bool], float | None]
    rise_from_prev_close: Callable[[float, float], float | None]
    is_stale_live_price_source: Callable[[str], bool]
    classify_buy_session: Callable[[datetime], str]
    get_order_spec: Callable[[datetime, bool], dict | None]
    fetch_orderbook_totals: Callable[[str, str], tuple[float | None, float | None]]
    format_reject_detail: Callable[..., str]


@dataclass
class BuyContext:
    """종목 1개 x 틱 1회의 신규 매수 판정 입력 + 조건들이 채우는 파생값."""

    code: str
    symbol_label: str
    current_dt: datetime
    nxt_tradeable: bool
    price: float
    price_source: str
    bar_time: Any
    buy_frame: pd.DataFrame
    cross_info: dict[str, object]
    config: Any  # r002.R76StrategyConfig
    api: Any  # TradingAPI (duck-typed: _in_cooldown / has_buy_exposure / get_affordable_buy_qty / live_state)
    state: BuyState
    risk: RiskState
    services: BuyServices

    # 조건이 채우는 파생값
    session_open_dt: datetime | None = None
    trigger_flag: str = "-"       # STEPS 로그의 1min_trigger(P|F|-)
    trigger_reason: str = ""
    steps_text: str = ""          # "1min_trigger(P) bb_slope_rising(P) ..." (_008에서 계산)
    eval_ctx: Any = None          # r002.BuyEvalContext
    buy_reason: str = ""
    prev_close: float | None = None
    rise_ratio: float | None = None
    qty: int = 0
    session: str = ""
    buy_detail: str = ""

    @property
    def norm_code(self) -> str:
        return str(self.code).zfill(6)

    @property
    def prev_bar(self) -> pd.Series:
        frame = self.buy_frame
        return frame.iloc[-2] if len(frame) >= 2 else frame.iloc[-1]

    def log(self, msg: str) -> None:
        self.services.log(msg)


@dataclass(frozen=True)
class BuyCondition:
    no: int
    func_name: str
    stage: str
    title: str
    description: str
    related_defines: tuple[str, ...]
    check: Callable[[BuyContext], bool]


@dataclass(frozen=True)
class BuyDecision:
    """approved=True면 모든 조건 통과(이때 _023의 예약이 남아 있음 - r003이 주문 후 실패 시 해제).
    approved=False면 failed가 처음 통과하지 못한 조건."""

    approved: bool
    failed: BuyCondition | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _steps_text(ctx: BuyContext) -> str:
    ctx_steps = gate_steps_diagnostic(
        ctx.buy_frame, ctx.current_dt, ctx.price, ctx.cross_info, ctx.config,
        gates=HYBRID_3MIN_CONTEXT_GATES,
    )
    return f"1min_trigger({ctx.trigger_flag}) {ctx_steps}"


def _reject_hybrid(ctx: BuyContext, buy_reason: str) -> None:
    """트리거/3분 컨텍스트(_008~_017) 공통 리젝 로그: 확인 상태 초기화 + 지표 스냅샷 + STEPS 진단."""
    ctx.state.buy_confirm_state.pop(ctx.code, None)
    detail = ctx.services.format_reject_detail(
        buy_reason,
        ctx.buy_frame.iloc[-1],
        ctx.prev_bar,
        live_price=ctx.price,
        cross_info=ctx.cross_info,
        frame=ctx.buy_frame,
    )
    if ctx.steps_text:
        detail = f"{detail} | STEPS {ctx.steps_text}"
    ctx.log(f"  [REJECT  ] {ctx.symbol_label} | {detail}")


def _context_gate(ctx: BuyContext, gate_name: str) -> bool:
    gate = _GATES_BY_NAME.get(gate_name)
    if gate is None:
        return True  # r002가 이 게이트를 HYBRID_3MIN_CONTEXT_GATES에서 뺀 경우(플래그 OFF) - 기존과 동일하게 건너뜀
    passed, ctx.eval_ctx, reason = gate.eval_fn(ctx.eval_ctx)
    if not passed:
        _reject_hybrid(ctx, f"HYBRID_3MIN_CTX_{reason}")
        return False
    return True


# ---------------------------------------------------------------------------
# STATE stage
# ---------------------------------------------------------------------------

def _001_entry_time_window(ctx: BuyContext) -> bool:
    return bool(ctx.services.is_new_entry_allowed(ctx.current_dt, ctx.nxt_tradeable))


def _002_session_open_warmup(ctx: BuyContext) -> bool:
    ctx.session_open_dt = ctx.services.get_session_open_datetime(ctx.current_dt, ctx.nxt_tradeable)
    if ctx.session_open_dt is not None:
        elapsed = (ctx.current_dt - ctx.session_open_dt).total_seconds()
        if elapsed < STARTUP_WARMUP_SECONDS:
            ctx.log(
                f"  [REJECT  ] {ctx.symbol_label} | SESSION_OPEN_WARMUP | "
                f"elapsed={elapsed:.0f}s / {STARTUP_WARMUP_SECONDS}s | "
                f"session_open={ctx.session_open_dt:%H:%M:%S}"
            )
            return False
    return True


def _003_trade_cooldown(ctx: BuyContext) -> bool:
    return not ctx.api._in_cooldown(ctx.code, ctx.current_dt)


def _004_same_bar_signal_dedup(ctx: BuyContext) -> bool:
    return ctx.state.signal_buy_bar.get(ctx.code) != ctx.bar_time


def _005_no_open_buy_exposure(ctx: BuyContext) -> bool:
    if ctx.api.has_buy_exposure(ctx.norm_code):
        ctx.log(f"  {ctx.symbol_label} [BUY SKIP] | ALREADY_TRADED_TODAY_UNTIL_SELL")
        return False
    return True


def _006_hard_stop_reentry_block(ctx: BuyContext) -> bool:
    # 당일 HARD_STOP 발생 종목 재진입 차단
    if HARD_STOP_BLOCK_REENTRY_TODAY and ctx.norm_code in ctx.state.hard_stop_today_codes:
        ctx.log(f"  {ctx.symbol_label} [BUY SKIP] | HARD_STOP_REENTRY_BLOCKED_TODAY")
        return False
    return True


def _007_stoploss_circuit_breaker(ctx: BuyContext) -> bool:
    # 연속 HARD_STOP 서킷브레이커 활성 시 신규 매수 차단
    until = ctx.risk.circuit_breaker_until
    if until is not None and ctx.current_dt < until:
        ctx.log(
            f"  {ctx.symbol_label} [BUY SKIP] | CIRCUIT_BREAKER_ACTIVE until={until:%H:%M:%S} "
            f"count={ctx.risk.hard_stop_daily_count}"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# TRIGGER stage
# ---------------------------------------------------------------------------

def _008_hybrid_1min_trigger(ctx: BuyContext) -> bool:
    frame_1min = ctx.services.get_frame_1min(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
    if frame_1min is None or frame_1min.empty or len(frame_1min) < 2:
        ctx.trigger_flag = "-"
        ctx.steps_text = _steps_text(ctx)
        _reject_hybrid(ctx, "HYBRID_1MIN_FRAME_UNAVAILABLE")
        return False

    # [2026-09-09] 3분 컨텍스트가 이미 uptrend_continuation으로 판정한 상태면 그 신호를 1분 트리거에도
    # 그대로 전달한다 - 452190 한빛레이저 사례(1분 트리거가 룩백 밖의 오래/강하게 지속된 랠리를 계속 놓침)
    # 대책 중 하나.
    frame = ctx.buy_frame
    cur3 = frame.iloc[-1]
    prev3 = frame.iloc[-2] if len(frame) >= 2 else cur3
    cur3_bb = _num(cur3, "BB_MIDDLE")
    prev3_bb = _num(prev3, "BB_MIDDLE")
    context_uptrend_continuation = False
    if not any(pd.isna(v) for v in (cur3_bb, prev3_bb)):
        bb_slope = _compute_bb_slope_pct(frame)
        cross_eval = _evaluate_bb_mid_cross(
            frame, cur3, prev3, cur3_bb, prev3_bb, ctx.price, bb_slope, ctx.cross_info,
        )
        context_uptrend_continuation = bool(cross_eval.get("uptrend_continuation"))

    trigger_ok, trigger_reason = check_buy_condition_1min_hybrid_trigger(
        frame_1min, context_uptrend_continuation=context_uptrend_continuation,
    )
    ctx.trigger_flag = "P" if trigger_ok else "F"
    ctx.steps_text = _steps_text(ctx)  # 기존과 동일하게 트리거 판정 직후, 통과 여부와 무관하게 계산
    # [2026-09-23] 1분봉 트리거가 "계속 유효" 상태로 지속된 시간을 추적한다 - 크로스 자체는 유효한데
    # 3분봉 컨텍스트 게이트가 못 따라와 대기만 길어지다가, 뒤늦게 게이트가 맞아떨어진 시점엔 이미
    # 추격매수 구간인 사례(204620 글로벌텍스프리 2026-09-23 09:04 트리거 유효 -> 09:17 매수, 13분
    # 지연) 대응. r002 check_buy_condition_1min_hybrid_trigger 자체는 무변경 - 이 함수가 반환한
    # trigger_ok를 그대로 사용해 별도로 체류시간만 잰다(g003도 동일 패턴 사용, r001 Update log 참조).
    trigger_age_seconds = update_timed_condition_state(
        ctx.state.buy_trigger_age_state, ctx.code, "1min_trigger", ctx.current_dt, trigger_ok,
    )
    if not trigger_ok:
        _reject_hybrid(ctx, f"HYBRID_1MIN_TRIGGER_{trigger_reason}")
        return False
    if HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS > 0 and trigger_age_seconds > HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS:
        _reject_hybrid(
            ctx,
            f"HYBRID_1MIN_TRIGGER_EXPIRED_{trigger_age_seconds:.0f}s_GT_{HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS:.0f}s",
        )
        return False
    ctx.trigger_reason = trigger_reason
    return True


# ---------------------------------------------------------------------------
# CONTEXT stage (3-min) - numeric logic lives in r002 gates
# ---------------------------------------------------------------------------

def _009_context_frame_ready(ctx: BuyContext) -> bool:
    eval_ctx, reason = build_context_eval(ctx.buy_frame, ctx.current_dt, ctx.price, ctx.cross_info, ctx.config)
    if eval_ctx is None:
        _reject_hybrid(ctx, reason)
        return False
    ctx.eval_ctx = eval_ctx
    return True


def _010_bb_slope_rising(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "bb_slope_rising")


def _011_bb_mid_downtrend_block(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "bb_mid_downtrend_block")


def _012_bb_upper_gap_min(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "bb_upper_gap_min")


def _013_stochastic_buy_signal(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "stochastic_buy_signal")


def _014_min_entry_atr_volatility(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "min_entry_atr_volatility")


def _015_min_liquidity_safety(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "min_liquidity_safety")


def _016_di_spread_min(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "di_spread_min")


def _017_context_score_threshold(ctx: BuyContext) -> bool:
    ok, reason = evaluate_context_score(ctx.eval_ctx, ctx.current_dt, ctx.config)
    if not ok:
        _reject_hybrid(ctx, reason)
        return False
    ctx.buy_reason = f"HYBRID_1MIN_TRIGGER_{ctx.trigger_reason}+{reason}"
    return True


# ---------------------------------------------------------------------------
# PRE_ORDER stage
# ---------------------------------------------------------------------------

def _018_excessive_rise_from_prev_close(ctx: BuyContext) -> bool:
    # 상승률 제한이 꺼져 있어도 전일 종가 조회(API)는 항상 수행한다(기존 동작 유지).
    prev_close = ctx.services.fetch_prev_close(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
    rise_ratio = ctx.services.rise_from_prev_close(ctx.price, float(prev_close or 0.0))
    ctx.prev_close = prev_close
    ctx.rise_ratio = rise_ratio
    if (
        MAX_BUY_RISE_PCT_FROM_PREV_CLOSE > 0
        and rise_ratio is not None
        and rise_ratio >= MAX_BUY_RISE_PCT_FROM_PREV_CLOSE
    ):
        ctx.state.buy_confirm_state.pop(ctx.code, None)
        ctx.log(
            f"  [REJECT  ] {ctx.symbol_label} | EXCESSIVE_RISE_FROM_PREV_CLOSE_"
            f"{rise_ratio*100:.2f}%_GE_{MAX_BUY_RISE_PCT_FROM_PREV_CLOSE*100:.2f}% | "
            f"prev_close={float(prev_close):,.0f} live={ctx.price:,.0f}"
        )
        return False
    return True


def _019_opening_gap_volume_gate(ctx: BuyContext) -> bool:
    if not ENABLE_OPENING_GAP_VOLUME_GATE:
        return True
    # passes_opening_gap_volume_gate는 심한 갭하락 종목을 gap_blocked_codes에 추가할 수 있다.
    gap_ok, gap_reason = passes_opening_gap_volume_gate(
        ctx.code, ctx.current_dt, ctx.session_open_dt, ctx.rise_ratio, ctx.buy_frame,
        ctx.state.gap_blocked_codes,
    )
    if not gap_ok:
        ctx.state.buy_confirm_state.pop(ctx.code, None)
        gap_txt = f"{ctx.rise_ratio*100:.2f}%" if ctx.rise_ratio is not None else "nan"
        ctx.log(f"  [REJECT  ] {ctx.symbol_label} | {gap_reason} | gap={gap_txt} live={ctx.price:,.0f}")
        return False
    return True


def _020_consecutive_poll_confirm(ctx: BuyContext) -> bool:
    """연속 BUY_CONSECUTIVE_CONFIRM_COUNT회 통과 확인. 이 지점에 도달할 때마다(결과와 무관) 상태를 갱신한다.

    직전 통과와 아래 중 하나라도 어긋나면 횟수를 1로 되돌리고 이번 통과를 새 기준으로 삼는다:
      (a) 간격 > BUY_CONFIRM_MAX_GAP_SECONDS
      (b) BUY_CONFIRM_REQUIRE_SAME_BAR이면 3분봉(bar_time)이 다름
      (c) BUY_CONFIRM_MAX_PRICE_DRIFT_PCT > 0이면 직전 통과 시점 실시간가 대비 이탈(절대값)이 이를 초과
    [2026-09-22] 정규장 활성 50종목의 평가주기(중앙 약 22초)가 예전 20초 창보다 길어 2회 연속 확인이 거의
    성립하지 않아 창을 45초로 넓히고, (b)(c)로 낡은 신호를 확인하지 않도록 했다(r001 BUY_CONFIRM_* 참조)."""
    confirm_state = ctx.state.buy_confirm_state.get(ctx.code)
    last_confirmed = confirm_state.get("confirmed_at") if confirm_state else None
    elapsed = (
        (ctx.current_dt - last_confirmed).total_seconds()
        if isinstance(last_confirmed, datetime) else None
    )
    reset_reason = ""
    if elapsed is None:
        confirm_count = 1
    elif elapsed > BUY_CONFIRM_MAX_GAP_SECONDS:
        confirm_count = 1
        reset_reason = f"GAP_{elapsed:.0f}s_GT_{BUY_CONFIRM_MAX_GAP_SECONDS:.0f}s"
    elif BUY_CONFIRM_REQUIRE_SAME_BAR and confirm_state.get("bar_time") != ctx.bar_time:
        confirm_count = 1
        reset_reason = "BAR_CHANGED"
    else:
        prev_price = confirm_state.get("price")
        drift_pct = (
            abs(ctx.price / prev_price - 1.0) * 100.0
            if isinstance(prev_price, (int, float)) and prev_price > 0 else 0.0
        )
        if BUY_CONFIRM_MAX_PRICE_DRIFT_PCT > 0 and drift_pct > BUY_CONFIRM_MAX_PRICE_DRIFT_PCT:
            confirm_count = 1
            reset_reason = f"PRICE_DRIFT_{drift_pct:.2f}%_GT_{BUY_CONFIRM_MAX_PRICE_DRIFT_PCT:.2f}%"
        else:
            confirm_count = int(confirm_state.get("count", 0)) + 1

    ctx.state.buy_confirm_state[ctx.code] = {
        "confirmed_at": ctx.current_dt, "count": confirm_count, "bar_time": ctx.bar_time, "price": ctx.price,
    }
    if confirm_count < BUY_CONSECUTIVE_CONFIRM_COUNT:
        reset_txt = f" | confirm_reset={reset_reason}" if reset_reason else ""
        ctx.log(
            f"  [BUY HOLD] {ctx.symbol_label} | reason=WAIT_NEXT_POLL_CONFIRM | "
            f"count={confirm_count}/{BUY_CONSECUTIVE_CONFIRM_COUNT} | "
            f"live={ctx.price:,.0f} bb_mid={_num(ctx.buy_frame.iloc[-1], 'BB_MIDDLE'):.1f} bar={ctx.bar_time:%H:%M:%S}"
            f"{reset_txt}"
        )
        return False
    return True


def _021_affordable_buy_qty(ctx: BuyContext) -> bool:
    ctx.qty = ctx.api.get_affordable_buy_qty(ctx.code, ctx.price, ctx.current_dt, ctx.nxt_tradeable)
    if ctx.qty <= 0:
        ctx.log(f"  [REJECT  ] {ctx.symbol_label} | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price={ctx.price:,.0f}")
        return False
    return True


def _022_fresh_live_price(ctx: BuyContext) -> bool:
    if ctx.services.is_stale_live_price_source(ctx.price_source):
        ctx.log(
            f"  [REJECT  ] {ctx.symbol_label} | STALE_LIVE_PRICE | "
            f"source={ctx.price_source} ttl={LIVE_PRICE_STALE_TTL_SECONDS}s"
        )
        return False
    return True


def _023_orderbook_ask_not_thin(ctx: BuyContext) -> bool:
    """세션/매수 상세(buy_detail)를 만들고 신규 매수를 '예약'(traded_today/signal_buy_bar)한 뒤 호가를 조회한다.
    통과하면 예약이 남는다 - r003이 이어서 place_buy_order를 호출하고 실패하면 예약을 해제한다. 매도잔량이
    얇아 거절할 때는 여기서 예약을 해제한다(확인 상태는 지우지 않음 - 기존 동작)."""
    ctx.session = ctx.services.classify_buy_session(ctx.current_dt)
    cur = ctx.buy_frame.iloc[-1]
    latest_vol = _num(cur, "volume")
    latest_vol_ma = _num(cur, "VOL_MA20")
    latest_vol_ratio = (
        latest_vol / latest_vol_ma
        if not any(pd.isna(v) for v in (latest_vol, latest_vol_ma)) and latest_vol_ma > 0
        else float("nan")
    )
    ctx.buy_detail = (
        f"reason={ctx.buy_reason} signal={ctx.cross_info.get('signal')} "
        f"live={ctx.price:,.0f} bb_mid={_num(cur, 'BB_MIDDLE'):.1f} "
        f"bar_close={_num(cur, 'close'):,.0f} ma5={_num(cur, 'MA_5'):.1f} "
        f"VOL={latest_vol:,.0f} VOLMA={latest_vol_ma:,.0f} vol_ratio={latest_vol_ratio:.4f}"
    )
    ctx.state.traded_today.add(ctx.norm_code)
    ctx.api.live_state["traded_today"] = ctx.state.traded_today
    ctx.state.signal_buy_bar[ctx.code] = ctx.bar_time

    spec = ctx.services.get_order_spec(ctx.current_dt, ctx.nxt_tradeable)
    mkt = "NX" if (spec and spec.get("exchange") == "NXT") else "J"
    ask_total, bid_total = ctx.services.fetch_orderbook_totals(ctx.norm_code, mkt)
    if ask_total is not None and bid_total is not None and bid_total > 0:
        if ask_total < bid_total * 0.5:
            ctx.log(
                f"  [REJECT  ] {ctx.symbol_label} | ORDERBOOK_ASK_THIN | "
                f"ask={ask_total:,.0f} bid={bid_total:,.0f} ratio={ask_total / bid_total:.2f}"
            )
            ctx.state.traded_today.discard(ctx.norm_code)
            ctx.api.live_state["traded_today"] = ctx.state.traded_today
            ctx.state.signal_buy_bar.pop(ctx.code, None)
            return False
    return True


# ---------------------------------------------------------------------------
# Registry + main function
# ---------------------------------------------------------------------------

_GATES_BY_NAME = {gate.name: gate for gate in HYBRID_3MIN_CONTEXT_GATES}

BUY_CONDITIONS: tuple[BuyCondition, ...] = (
    BuyCondition(1, "_001_entry_time_window", "STATE", "신규 진입 허용 시간대",
                 "정규장/NXT 세션별 신규 진입 마감 시각 이전인가 (마감 후에는 청산만)", (), _001_entry_time_window),
    BuyCondition(2, "_002_session_open_warmup", "STATE", "세션 개장 워밍업",
                 "세션(NXT 프리/정규/NXT 애프터) 시작 후 STARTUP_WARMUP_SECONDS 경과", ("STARTUP_WARMUP_SECONDS",), _002_session_open_warmup),
    BuyCondition(3, "_003_trade_cooldown", "STATE", "종목 거래 쿨다운",
                 "직전 주문 후 TRADE_COOLDOWN_MINUTES 경과 (api._in_cooldown)", ("TRADE_COOLDOWN_MINUTES",), _003_trade_cooldown),
    BuyCondition(4, "_004_same_bar_signal_dedup", "STATE", "같은 봉 재신호 차단",
                 "이 3분봉에서 이미 매수 신호를 낸 종목은 건너뜀 (signal_buy_bar)", (), _004_same_bar_signal_dedup),
    BuyCondition(5, "_005_no_open_buy_exposure", "STATE", "미청산 매수 노출 없음",
                 "당일 매수 후 아직 매도되지 않았거나 미체결 매수가 있으면 차단 (has_buy_exposure)", (), _005_no_open_buy_exposure),
    BuyCondition(6, "_006_hard_stop_reentry_block", "STATE", "손절 종목 당일 재진입 차단",
                 "HARD_STOP/ATR 손절이 난 종목은 당일 재진입 금지", ("HARD_STOP_BLOCK_REENTRY_TODAY",), _006_hard_stop_reentry_block),
    BuyCondition(7, "_007_stoploss_circuit_breaker", "STATE", "손절 누적 서킷브레이커",
                 "손절이 HARD_STOP_CIRCUIT_BREAKER_COUNT회 누적되면 COOLDOWN_MIN분간 신규 매수 중단",
                 ("HARD_STOP_CIRCUIT_BREAKER_COUNT", "HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN"), _007_stoploss_circuit_breaker),
    BuyCondition(8, "_008_hybrid_1min_trigger", "TRIGGER", "1분봉 트리거",
                 "1분봉 BB중심선 골든크로스(룩백)/우상향 지속 + 1분 캔들·BB갭 추격 가드 (r002 check_buy_condition_1min_hybrid_trigger) "
                 "+ 트리거 유효 지속시간이 HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS 초과 시 만료 반려",
                 ("HYBRID_1MIN_TRIGGER_LOOKBACK_BARS", "HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT", "HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT",
                  "HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT", "HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR",
                  "HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT", "HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT",
                  "HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS"), _008_hybrid_1min_trigger),
    BuyCondition(9, "_009_context_frame_ready", "CONTEXT", "3분봉 컨텍스트 준비",
                 "3분봉 2개 이상 + BB 지표 산출 완료", (), _009_context_frame_ready),
    BuyCondition(10, "_010_bb_slope_rising", "CONTEXT", "BB 중심선 상승 기울기",
                 "BB_MIDDLE 최근 lookback봉 기울기 > BB_SLOPE_MIN_PCT", ("BB_SLOPE_LOOKBACK_BARS", "BB_SLOPE_MIN_PCT"), _010_bb_slope_rising),
    BuyCondition(11, "_011_bb_mid_downtrend_block", "CONTEXT", "BB 중심선 연속 하락 차단",
                 "BB 중심선 최근 N봉 연속 하락이면 차단(가격이 이미 돌파했으면 해제). ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK=False면 건너뜀",
                 ("BB_MID_DOWNTREND_BARS", "ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK"), _011_bb_mid_downtrend_block),
    BuyCondition(12, "_012_bb_upper_gap_min", "CONTEXT", "BB 상단 여유",
                 "BB 상단까지 남은 폭 >= BB_UPPER_GAP_MIN_PCT", ("BB_UPPER_GAP_MIN_PCT",), _012_bb_upper_gap_min),
    BuyCondition(13, "_013_stochastic_buy_signal", "CONTEXT", "스토캐스틱(+윌리엄스) 매수신호",
                 "%K>%D(골든크로스 포함) + %D>=STOCH_D_BUY_MIN + %K 30~90 밴드",
                 ("STOCH_BUY_MIN", "STOCH_D_BUY_MIN", "WILLIAMS_BUY_FLOOR", "WILLIAMS_OVERBOUGHT_CEIL"), _013_stochastic_buy_signal),
    BuyCondition(14, "_014_min_entry_atr_volatility", "CONTEXT", "최소 변동성(ATR%)",
                 "ATR% >= STAGED_TP1_PCT x MIN_ENTRY_ATR_TO_TP1_RATIO", ("ENABLE_MIN_ENTRY_ATR_FILTER", "MIN_ENTRY_ATR_TO_TP1_RATIO"), _014_min_entry_atr_volatility),
    BuyCondition(15, "_015_min_liquidity_safety", "CONTEXT", "최소 유동성",
                 "거래량/거래량MA/비율/거래대금 최소치", ("MIN_ENTRY_VOL_MA", "MIN_ENTRY_VOLUME", "MIN_ENTRY_TURNOVER_KRW"), _015_min_liquidity_safety),
    BuyCondition(16, "_016_di_spread_min", "CONTEXT", "DI 스프레드",
                 "+DI - -DI >= DI_SPREAD_MIN_REQUIRED", ("DI_SPREAD_MIN_REQUIRED",), _016_di_spread_min),
    BuyCondition(17, "_017_context_score_threshold", "CONTEXT", "가점 합계 임계값",
                 "가점 12항목 합계 >= BB_BUY_SCORE_THRESHOLD (개장 15분은 OPENING_GUARD_SCORE_THRESHOLD)",
                 ("BB_BUY_SCORE_THRESHOLD", "OPENING_GUARD_MINUTES", "OPENING_GUARD_SCORE_THRESHOLD"), _017_context_score_threshold),
    BuyCondition(18, "_018_excessive_rise_from_prev_close", "PRE_ORDER", "전일 종가 대비 과도 상승",
                 "전일 종가 대비 상승률 < MAX_BUY_RISE_PCT_FROM_PREV_CLOSE (API: 전일 종가 조회)",
                 ("MAX_BUY_RISE_PCT_FROM_PREV_CLOSE",), _018_excessive_rise_from_prev_close),
    BuyCondition(19, "_019_opening_gap_volume_gate", "PRE_ORDER", "개장 초반 갭/거래량 게이트",
                 "개장 후 5분 이내: 갭 범위 + 거래량 폭발 확인, 큰 갭하락은 당일 영구 차단",
                 ("ENABLE_OPENING_GAP_VOLUME_GATE", "OPENING_GAP_GATE_WINDOW_MINUTES", "OPENING_GAP_MIN_PCT", "OPENING_GAP_MAX_PCT",
                  "OPENING_GAP_HARD_FLOOR_PCT", "OPENING_MIN_EARLY_VOLUME_RATIO"), _019_opening_gap_volume_gate),
    BuyCondition(20, "_020_consecutive_poll_confirm", "PRE_ORDER", "연속 폴링 확인",
                 "모든 조건이 연속 BUY_CONSECUTIVE_CONFIRM_COUNT회 통과해야 매수 - 직전 통과와 간격 BUY_CONFIRM_MAX_GAP_SECONDS 이내, "
                 "같은 3분봉(BUY_CONFIRM_REQUIRE_SAME_BAR), 실시간가 이탈 BUY_CONFIRM_MAX_PRICE_DRIFT_PCT 이내여야 연속으로 인정",
                 ("BUY_CONSECUTIVE_CONFIRM_COUNT", "BUY_CONFIRM_MAX_GAP_SECONDS", "BUY_CONFIRM_REQUIRE_SAME_BAR",
                  "BUY_CONFIRM_MAX_PRICE_DRIFT_PCT", "POLL_INTERVAL_SECONDS"), _020_consecutive_poll_confirm),
    BuyCondition(21, "_021_affordable_buy_qty", "PRE_ORDER", "매수 가능 수량",
                 "예수금/1회 주문 한도로 1주 이상 살 수 있는가 (API: 계좌 조회)", ("MAX_ORDER_AMOUNT_KRW",), _021_affordable_buy_qty),
    BuyCondition(22, "_022_fresh_live_price", "PRE_ORDER", "신선한 현재가",
                 "현재가가 stale 캐시/봉 종가 대체값이 아닌가", ("LIVE_PRICE_STALE_TTL_SECONDS",), _022_fresh_live_price),
    BuyCondition(23, "_023_orderbook_ask_not_thin", "PRE_ORDER", "호가 잔량 확인",
                 "매도호가 총잔량 >= 매수호가 총잔량 x 0.5 (API: 호가 조회). 통과 시 매수 예약 상태로 남음", (), _023_orderbook_ask_not_thin),
)

# r002에 3분 컨텍스트 게이트가 추가/삭제되면 위 번호 목록도 함께 고치도록 import 시점에 바로 실패시킨다.
_CONTEXT_GATE_FUNC_TO_NAME = {
    "_010_bb_slope_rising": "bb_slope_rising",
    "_011_bb_mid_downtrend_block": "bb_mid_downtrend_block",
    "_012_bb_upper_gap_min": "bb_upper_gap_min",
    "_013_stochastic_buy_signal": "stochastic_buy_signal",
    "_014_min_entry_atr_volatility": "min_entry_atr_volatility",
    "_015_min_liquidity_safety": "min_liquidity_safety",
    "_016_di_spread_min": "di_spread_min",
}
_registered_gate_names = {
    _CONTEXT_GATE_FUNC_TO_NAME[c.func_name] for c in BUY_CONDITIONS if c.func_name in _CONTEXT_GATE_FUNC_TO_NAME
}
# r002가 플래그로 HYBRID_3MIN_CONTEXT_GATES에서 뺄 수 있는(=r005에서 건너뛰는) 의도된 선택 게이트.
_OPTIONAL_CONTEXT_GATES = {"bb_mid_downtrend_block"}
if not set(_GATES_BY_NAME) <= _registered_gate_names:
    raise RuntimeError(
        "r005 BUY_CONDITIONS에 없는 3분 컨텍스트 게이트가 r002 HYBRID_3MIN_CONTEXT_GATES에 있음: "
        f"{sorted(set(_GATES_BY_NAME) - _registered_gate_names)}"
    )
# 반대 방향(Codex 사후 검토): r002에서 삭제/이름 변경된 게이트를 r005가 계속 등록해 두면 번호 목록에는 있는데
# 실제로는 조용히 건너뛰게 된다 - 선택 게이트를 뺀 나머지는 정확히 일치해야 한다.
if not _registered_gate_names - set(_GATES_BY_NAME) <= _OPTIONAL_CONTEXT_GATES:
    raise RuntimeError(
        "r005 BUY_CONDITIONS에 등록됐지만 r002 HYBRID_3MIN_CONTEXT_GATES에 없는 게이트: "
        f"{sorted(_registered_gate_names - set(_GATES_BY_NAME) - _OPTIONAL_CONTEXT_GATES)}"
    )


def check_buy_conditions(ctx: BuyContext, conditions: tuple[BuyCondition, ...] | None = None) -> BuyDecision:
    """신규 매수 메인 판정: 조건을 번호 순서대로 평가하고 처음 통과하지 못한 조건에서 멈춘다."""
    for cond in (BUY_CONDITIONS if conditions is None else conditions):
        if not cond.check(ctx):
            return BuyDecision(approved=False, failed=cond)
    return BuyDecision(approved=True)


def describe_buy_conditions() -> list[dict[str, object]]:
    """문서/발표자료용 조건 목록(번호, 함수명, 단계, 제목, 설명, 관련 상수)."""
    return [
        {"no": c.no, "func": c.func_name, "stage": c.stage, "title": c.title,
         "description": c.description, "defines": list(c.related_defines)}
        for c in BUY_CONDITIONS
    ]
