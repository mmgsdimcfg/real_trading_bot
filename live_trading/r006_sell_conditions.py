"""R006 live sell conditions - numbered checklist for an OPEN position (used by the r003 live executor).

Overview
- r003 used to manage an open position with ~680 inline lines inside run(). Those checks now live here, one object per
  condition, named `_001_...`, `_002_...`, ... in the exact order they are evaluated.
- `evaluate_sell_conditions(ctx)` is the main function: it walks SELL_CONDITIONS in order. Every condition returns
  True  -> STOP_SYMBOL   (this symbol is finished for this tick: an order was attempted, or a hold/dedup decision was
                           taken - the old inline code did `continue`)
  False -> FALL_THROUGH  (nothing to do here; keep evaluating the next condition; log-only / state-only branches too)
  A condition performs its own order call, logging and state changes at the same point, with the same text, as the old
  inline code. Nothing in this file changes a threshold or a strategy decision.
- No broker imports: everything that talks to the broker is injected through `SellServices` and `ctx.api`.

Order                                        Kind        Old r003 inline block (baseline lines)
  _001 stale live price guard                GUARD       4431-4436  ([SELL REJECT] STALE_LIVE_PRICE)
  _002 update position tracking              UPDATE      4438-4471  (current_price, price samples, highest/peak)
  _003 peak next-bar bearish exit            EXIT        4473-4508
  _004 hard stop loss                        EXIT        4544-4567 (2026-09-23: added confirm window)
  _005 pyramid add entry                     ADD_ENTRY   4569-4615  (add-on BUY, not a sell)
  _006 staged TP1 partial                    EXIT        4617-4690  (ENABLE_STAGED_TAKE_PROFIT)
  _007 surge ladder TP2/TP3                  EXIT        4692-4732
  _008 staged TP2 partial                    EXIT        4734-4751
  _009 TP3 trail arm                         STATE       4753-4766  (flag + log only)
  _010 legacy TP full / partial              EXIT        4767-4796  (only when ENABLE_STAGED_TAKE_PROFIT=False)
  _011 hybrid 1min dead-cross exit           EXIT        (2026-09-23 신규, "매수/매도 컨셉 재검토" + Codex
                                                            설계검토 - _008_hybrid_1min_trigger의 대칭 매도판)
  _012 peak retracement guard                EXIT        (2026-09-23 신규 - TP1/ATR익절선 도달 전 구간의
                                                            고점대비 되돌림 익절가드)
  _013 signal exit: stochastic K<D           EXIT        4799-4816
  _014 signal exit: MACD hist down 2 bars    EXIT        4817-4831
  _015 ATR take profit / TP extension        EXIT/LOG    4833-4850
  _016 post-buy entry drop guard             EXIT        4852-4886
  _017 breakeven fail guard                  EXIT        4888-4920
  _018 no-trend time exit                    EXIT        4922-4957
  _019 ATR stop loss                         EXIT        4959-4994
  _020 trailing stop                         EXIT        4996-5053
  _021 same-bar sell dedup                   GUARD       5055-5056
  _022 shared reversal sell                  EXIT        5058-5099  (r002 check_sell_condition)

Update log (append only):
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("매수/매도 컨셉 재검토" 3~4단계 적용) + Codex 설계검토 - 신규 _011_hybrid_1min_dead_cross_exit
      추가(매수측 _008_hybrid_1min_trigger의 대칭 매도판, r002 check_1min_dead_cross 사용). 기존 _021
      shared_reversal_sell(AUX_REVERSAL_SCORE)이 구조적으로 손실 포지션을 절대 청산 못 하는 문제
      (2026-09-23 컨셉 재검토 메모)를 "수익보호 전용으로 남기고 손실은 별도 조건이 담당"하는 방식으로
      해결. 기존 _011~_020이 _012~_021로 한 칸씩 밀림(로직 무변경, 이름/번호만 이동) - r001 Update log
      2026-09-23 참조.
    impact: common (r003 실전/g003 백테스트 공용)
    compatibility: breaking (ENABLE_HYBRID_1MIN_DEADCROSS_EXIT=False로 롤백 가능)
- [2026-09-23] type=fix owner=claude
    summary: 사용자 요청(119850 지엔씨에너지 2026-09-23 09:02 하드손절 직후 반등 사례 + Codex 검토) - _004
      hard_stop_loss가 20개 조건 중 유일하게 확인창 없이 단일 폴링 틱에서 즉시 발동되던 것을 수정. _017
      atr_stop_loss와 동일한 update_timed_condition_state() 헬퍼로 신규 HARD_STOP_CONFIRM_SECONDS(r001,
      기본 10초) 동안 조건이 연속 유지되어야 발동하도록 변경. r001 Update log 2026-09-23 참조(트레이드오프
      설명 포함 - 손실 상한 보장 아님).
    impact: live (r003)
    compatibility: breaking (HARD_STOP_CONFIRM_SECONDS<=0이면 기존과 동일. 기본값 10초에서는 하드손절이
      최소 10초 확인 후 발동 - 지속 하락 시 이전보다 다소 늦게/낮은 가격에 체결될 수 있음)
- [2026-09-21] type=refactor owner=claude
    summary: r003 run()의 보유 종목 관리(매도) 인라인 블록을 번호 붙은 조건 객체(_001~_020)와 메인 함수
      evaluate_sell_conditions()로 분리. 평가 순서/주문 시점/로그 문구/상태 변경은 기존과 동일(동작 불변).
      Codex 사전 검토 반영: 포지션 추적 갱신(_002)은 stale 가드(_001) 뒤에서만 실행, 손절 누적 상태는
      run() 수명 동안 하나인 RiskState(r005)로 매수/매도가 공유, 반환값 의미를 STOP_SYMBOL/FALL_THROUGH로 명시.
    impact: live (r003)
    compatibility: backward-compatible
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from r001_define_config import (
    ATR_STOP_CONFIRM_SECONDS,
    ATR_STOP_MULTIPLIER,
    ATR_TAKE_PROFIT_MULTIPLIER,
    AUX_SELL_MIN_REALIZED_TARGET_PCT,
    AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT,
    BREAKEVEN_FAIL_ARM_PNL,
    BREAKEVEN_FAIL_CONFIRM_SECONDS,
    BREAKEVEN_FAIL_GIVEBACK_PCT,
    ENABLE_HYBRID_1MIN_DEADCROSS_EXIT,
    ENABLE_PEAK_NEXT_BAR_BEARISH_EXIT,
    ENABLE_PYRAMIDING,
    ENABLE_STAGED_TAKE_PROFIT,
    ENABLE_SURGE_LADDER_TP,
    ENABLE_TP1_CAP_AT_ATR_TP,
    ENABLE_TP_EXTENSION_TRAILING,
    HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN,
    HARD_STOP_CIRCUIT_BREAKER_COUNT,
    HARD_STOP_CONFIRM_SECONDS,
    HARD_STOP_LOSS_PCT,
    HARD_STOP_MIN_HOLD_SECONDS,
    HYBRID_1MIN_DEADCROSS_CONFIRM_SECONDS,
    HYBRID_1MIN_DEADCROSS_LOOKBACK_BARS,
    HYBRID_1MIN_DEADCROSS_LOSS_EXIT_PNL_MAX,
    ENABLE_PEAK_RETRACE_GUARD,
    PEAK_RETRACE_GUARD_ARM_PNL,
    PEAK_RETRACE_GUARD_MIN_PCT,
    PEAK_RETRACE_GUARD_ATR_MULT,
    PEAK_RETRACE_GUARD_CONFIRM_SECONDS,
    HYBRID_1MIN_DEADCROSS_MIN_HOLD_SECONDS,
    LIVE_PRICE_STALE_TTL_SECONDS,
    NO_TREND_EXIT_ARM_SECONDS,
    NO_TREND_EXIT_CONFIRM_SECONDS,
    NO_TREND_EXIT_MAX_PEAK_PNL,
    NO_TREND_EXIT_MIN_PNL,
    PEAK_NEXT_BAR_DROP_PCT,
    POST_BUY_BB_DROP_ARMED_SECONDS,
    POST_BUY_BB_DROP_PCT,
    POST_BUY_DROP_CONFIRM_SECONDS,
    PYRAMID_TRIGGER_PNL_PCT,
    SIGNAL_EXIT_MACD_PNL_MAX,
    SIGNAL_EXIT_MIN_HOLD_SECONDS,
    SIGNAL_EXIT_STOCH_SUPPRESS_PNL_MIN,
    SIGNAL_EXIT_STRONG_TREND_ADX_MIN,
    STAGED_TP1_PCT,
    STAGED_TP1_RATIO,
    STAGED_TP2_PCT,
    STAGED_TP2_RATIO,
    SURGE_LOOKBACK_SECONDS,
    SURGE_MIN_CONFIRMS,
    SURGE_SPEED_ATR_MULT,
    SURGE_SPEED_MIN_PCT,
    SURGE_TP2_PCT,
    SURGE_TP2_RATIO,
    SURGE_TP3_PCT,
    SURGE_VOLUME_RATIO_MIN,
    TP1_ATR_MULTIPLIER,
    TP_EXTENSION_TRAIL_FROM_PEAK,
    TRAILING_STOP_FROM_PEAK,
)
from r002_strategy_core_shared import (
    _compute_bb_slope_pct,
    _num,
    check_1min_dead_cross,
    compute_staged_tp1_target_pct,
    detect_price_surge,
    next_surge_ladder_action,
    update_recent_price_samples,
    update_timed_condition_state,
)
from r005_buy_conditions import RiskState

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Context / services / decision
# ---------------------------------------------------------------------------

@dataclass
class SellState:
    """r003 run()의 가변 상태 컨테이너들(참조 전달 - 조건이 직접 변경)."""

    recent_price_samples: dict[str, list[tuple[datetime, float]]]
    trailing_sell_confirm_state: dict[str, dict[str, object]]
    post_buy_bb_drop_state: dict[str, dict]
    breakeven_fail_state: dict[str, dict]
    no_trend_exit_state: dict[str, dict]
    atr_stop_confirm_state: dict[str, dict]
    hard_stop_confirm_state: dict[str, dict]
    hybrid_1min_dead_cross_state: dict[str, dict]
    peak_retrace_guard_state: dict[str, dict]
    signal_sell_bar: dict[str, object]
    hard_stop_today_codes: set[str]


@dataclass
class SellServices:
    """r003 내부 함수(브로커/로그 의존)를 주입한다. 이 모듈은 브로커를 import하지 않는다."""

    log: Callable[[str], None]
    is_stale_live_price_source: Callable[[str], bool]
    sanitize_surge_ladder: Callable[[object], dict | None]
    extract_aux_score_from_reason: Callable[[str], int | None]
    aux_min_pnl_for_score: Callable[[int], float | None]
    check_sell_condition: Callable[[pd.DataFrame, float, float, dict], tuple[bool, str]]
    classify_buy_session: Callable[[datetime], str]
    get_frame_1min: Callable[[str, datetime, bool], pd.DataFrame | None]


@dataclass
class SellContext:
    """보유 종목 1개 x 틱 1회의 매도/포지션 관리 입력 + 조건들이 채우는 파생값.

    entry_price/pnl_pct/is_startup_position은 기존 코드에서 stale 가드보다 앞서 계산되던 값이라 생성 시 계산한다.
    포지션을 변경하는 추적 갱신(_002)은 stale 가드(_001) 뒤에서만 실행된다."""

    code: str
    name: str
    symbol_label: str
    current_dt: datetime
    nxt_tradeable: bool
    pos: dict
    price: float
    price_source: str
    frame: pd.DataFrame
    cur: pd.Series
    bar_time: Any
    cross_info: dict[str, object]
    date_str: str
    api: Any  # TradingAPI (duck-typed)
    state: SellState
    risk: RiskState
    services: SellServices

    # 생성 시 계산 (기존 4428-4430)
    entry_price: float = 0.0
    pnl_pct: float = 0.0
    is_startup_position: bool = False

    # _002가 채우는 값 (기존 4438-4471)
    bar_low: float = _NAN
    atr_val: float = _NAN
    atr_pct: float = _NAN
    atr_tp_price: float = _NAN
    atr_sl_price: float = _NAN
    atr_tp_pct: float = _NAN
    atr_sl_pct: float = _NAN
    pnl_sl: float = 0.0
    highest_price: float = 0.0
    peak_pnl_pct: float = 0.0
    profit_giveback: float = 0.0

    # 스테이지드 익절 블록(_006~_009) 공유값
    entry_qty: int = 0
    tp1_target_pct: float = _NAN
    tp2_target_pct: float = _NAN
    atr_tp1_dynamic_pct: float = _NAN
    surge_ladder: dict | None = None

    def __post_init__(self) -> None:
        self.entry_price = float(self.pos["buy_price"])
        self.pnl_pct = (self.price / self.entry_price) - 1.0
        self.is_startup_position = str(self.code).zfill(6) in self.api.startup_position_codes

    def log(self, msg: str) -> None:
        self.services.log(msg)


@dataclass(frozen=True)
class SellCondition:
    no: int
    func_name: str
    kind: str   # GUARD | UPDATE | EXIT | ADD_ENTRY | STATE
    title: str
    description: str
    related_defines: tuple[str, ...]
    run: Callable[[SellContext], bool]   # True = STOP_SYMBOL, False = FALL_THROUGH


@dataclass(frozen=True)
class SellDecision:
    """stopped_by가 있으면 그 조건에서 이 종목의 이번 틱 처리가 끝났다(STOP_SYMBOL), None이면 끝까지 통과."""

    stopped_by: SellCondition | None = None


# ---------------------------------------------------------------------------
# helpers (pure - values are computed at use time from pos/price, exactly as the old inline code did)
# ---------------------------------------------------------------------------

def _hold_info(ctx: SellContext) -> tuple[datetime, object, float]:
    """(_buy_time, _buy_token, _held_for_guard) - 기존 4853-4858."""
    pos, code, current_dt = ctx.pos, ctx.code, ctx.current_dt
    buy_time_raw = pos.get("buy_time")
    buy_time = buy_time_raw if isinstance(buy_time_raw, datetime) else current_dt
    buy_token = buy_time_raw if isinstance(buy_time_raw, datetime) else (
        f"unknown_buy_time:{code}:{pos.get('entry_price', 0)}:{int(pos.get('quantity', 0))}"
    )
    held = (current_dt - buy_time).total_seconds()
    return buy_time, buy_token, held


def _signal_trend_inputs(ctx: SellContext) -> dict[str, float | bool]:
    """시그널 청산(_011/_012)이 공유하는 입력 - 기존 4510-4542."""
    frame, cur, pos, price, current_dt = ctx.frame, ctx.cur, ctx.pos, ctx.price, ctx.current_dt
    k_now = _num(cur, "STOCH_K")
    d_now = _num(cur, "STOCH_D")
    hist_now = _num(cur, "MACD_HIST")
    hist_prev = _num(frame.iloc[-2], "MACD_HIST") if len(frame) >= 2 else float("nan")
    hist_prev2 = _num(frame.iloc[-3], "MACD_HIST") if len(frame) >= 3 else float("nan")
    adx_now = _num(cur, "ADX")
    di_plus_now = _num(cur, "DI_PLUS")
    di_minus_now = _num(cur, "DI_MINUS")
    # 강한 상승 추세: ADX > 28 이고 +DI > -DI 이면 스토캐스틱 K<D 매도 신호 무시
    sig_buy_time_raw = pos.get("buy_time")
    sig_held_seconds = (current_dt - sig_buy_time_raw).total_seconds() if isinstance(sig_buy_time_raw, datetime) else 0.0
    adx_uptrend = (
        not pd.isna(adx_now) and adx_now > SIGNAL_EXIT_STRONG_TREND_ADX_MIN
        and not pd.isna(di_plus_now) and not pd.isna(di_minus_now)
        and di_plus_now > di_minus_now
    )
    # ADX 보완: ADX>28 도달 전(추세 초입)에도 가격이 이미 BB_MID 위에서 MA5가 상승 중이면 상승추세로 간주해
    # 시그널 매도를 억제한다(000500 가온전선 2026-08-27 09:06 사례).
    ma5_now = _num(cur, "MA_5")
    ma5_prev_trend = _num(frame.iloc[-2], "MA_5") if len(frame) >= 2 else float("nan")
    bb_mid_now = _num(cur, "BB_MIDDLE")
    price_uptrend = (
        not any(pd.isna(v) for v in (ma5_now, ma5_prev_trend, bb_mid_now))
        and ma5_now > ma5_prev_trend
        and price > bb_mid_now
    )
    return {
        "k_now": k_now, "d_now": d_now,
        "hist_now": hist_now, "hist_prev": hist_prev, "hist_prev2": hist_prev2,
        "sig_held_seconds": sig_held_seconds,
        "adx_uptrend": adx_uptrend, "price_uptrend": price_uptrend,
        "strong_uptrend": adx_uptrend or price_uptrend,
    }


def _staged_targets(ctx: SellContext) -> None:
    """스테이지드 익절 공유값(entry_qty/tp1_target_pct/tp2_target_pct)을 채운다 - 기존 4618-4634.
    같은 틱에서 _006이 먼저 호출한다(트리거 시 pos를 바꾸고 STOP_SYMBOL이므로 이후 조건은 같은 값을 본다)."""
    pos = ctx.pos
    entry_qty = int(pos.get("entry_quantity", 0) or 0)
    if entry_qty <= 0:
        entry_qty = int(pos["quantity"])
    # 1차 익절 목표를 종목 변동성(ATR)에 연동해 동적으로 산출한다. 목표 = max(고정 STAGED_TP1_PCT,
    # ATR*TP1_ATR_MULTIPLIER/진입가)를 ATR 익절선(=TP_EXTENSION 트레일 무장선)으로 상한 처리 - 저변동성 종목에서
    # 트레일이 TP1보다 먼저 무장되어 1차 분할 없이 전량 매도되던 문제 수정(r001 ENABLE_TP1_CAP_AT_ATR_TP 참조).
    # 2차는 기존 1차-2차 간격(STAGED_TP2_PCT - STAGED_TP1_PCT)만큼 그 위에 얹는다.
    tp1_target_pct, atr_tp1_dynamic_pct = compute_staged_tp1_target_pct(
        ctx.atr_pct,
        ctx.atr_tp_pct,
        staged_tp1_pct=STAGED_TP1_PCT,
        tp1_atr_multiplier=TP1_ATR_MULTIPLIER,
        cap_at_atr_tp=ENABLE_TP1_CAP_AT_ATR_TP,
    )
    ctx.entry_qty = entry_qty
    ctx.tp1_target_pct = tp1_target_pct
    ctx.tp2_target_pct = tp1_target_pct + (STAGED_TP2_PCT - STAGED_TP1_PCT)
    ctx.atr_tp1_dynamic_pct = atr_tp1_dynamic_pct


def _register_hard_stop(ctx: SellContext, label: str) -> None:
    """손절 매도 성공 후: 당일 재진입 차단 등록 + 누적 카운트 + 서킷브레이커(기존 4561-4565/4988-4992)."""
    ctx.state.hard_stop_today_codes.add(str(ctx.code).zfill(6))
    ctx.risk.hard_stop_daily_count += 1
    if ctx.risk.hard_stop_daily_count >= HARD_STOP_CIRCUIT_BREAKER_COUNT:
        ctx.risk.circuit_breaker_until = ctx.current_dt + timedelta(minutes=HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN)
        ctx.log(
            f"  [CIRCUIT_BREAKER] {label} {ctx.risk.hard_stop_daily_count}회 발생 → 신규 매수 "
            f"{HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN}분 차단 until {ctx.risk.circuit_breaker_until:%H:%M:%S}"
        )


# ---------------------------------------------------------------------------
# GUARD / UPDATE
# ---------------------------------------------------------------------------

def _001_stale_live_price_guard(ctx: SellContext) -> bool:
    if ctx.services.is_stale_live_price_source(ctx.price_source):
        ctx.log(
            f"  {ctx.symbol_label} [SELL REJECT] | STALE_LIVE_PRICE | "
            f"source={ctx.price_source} ttl={LIVE_PRICE_STALE_TTL_SECONDS}s"
        )
        return True
    return False


def _002_update_position_tracking(ctx: SellContext) -> bool:
    """포지션 추적 갱신(항상 FALL_THROUGH): 현재가/최근 가격 표본/최고가/고점봉/ATR 익절·손절선/최고 손익."""
    cur, price, pos, entry_price = ctx.cur, ctx.price, ctx.pos, ctx.entry_price
    # 저가(bar low)가 현재가보다 낮으면 손절 판정에 우선 반영 - 확정봉 low를 직접 섞으면 현재가가 회복된
    # 상태에서도 오손절이 날 수 있어 실시간 손절은 현재가 기준으로 판정한다.
    ctx.bar_low = float(cur["low"]) if "low" in cur.index and not pd.isna(cur["low"]) else float("nan")
    atr_val = _num(cur, "ATR")
    # ATR은 현재가 스케일의 절대값(원)이므로 반드시 ATR%(=atr_val/현재가)로 정규화한 뒤 entry_price에 상대
    # 비율로 적용한다.
    atr_pct = float("nan")
    atr_tp_price = float("nan")
    atr_sl_price = float("nan")
    atr_tp_pct = float("nan")
    atr_sl_pct = float("nan")
    if not pd.isna(atr_val) and atr_val > 0 and price > 0 and entry_price > 0:
        atr_pct = float(atr_val) / price
        atr_tp_pct = atr_pct * ATR_TAKE_PROFIT_MULTIPLIER
        atr_sl_pct = -atr_pct * ATR_STOP_MULTIPLIER
        atr_tp_price = entry_price * (1.0 + atr_tp_pct)
        atr_sl_price = entry_price * (1.0 + atr_sl_pct)
    ctx.atr_val, ctx.atr_pct = atr_val, atr_pct
    ctx.atr_tp_price, ctx.atr_sl_price = atr_tp_price, atr_sl_price
    ctx.atr_tp_pct, ctx.atr_sl_pct = atr_tp_pct, atr_sl_pct
    sl_price = price
    ctx.pnl_sl = (sl_price / entry_price) - 1.0

    pos["current_price"] = price
    if ENABLE_SURGE_LADDER_TP:
        update_recent_price_samples(ctx.state.recent_price_samples, ctx.code, ctx.current_dt, price, SURGE_LOOKBACK_SECONDS * 2.0)
    prev_highest_price = float(pos.get("highest_price", 0.0) or 0.0)
    pos["highest_price"] = max(prev_highest_price, price)
    highest_price = float(pos["highest_price"])  # TP/트레일링 로직 계산 전에 갱신값 반영
    if highest_price > prev_highest_price:
        # 이 틱에서 신고점을 찍었다 - 그 신고점이 속한 확정봉을 "고점봉"으로 기록해둔다
        # (PEAK_NEXT_BAR_BEARISH_EXIT에서 "고점봉 바로 다음 확정봉" 판정에 사용).
        pos["peak_bar_time"] = ctx.bar_time
    ctx.highest_price = highest_price
    ctx.peak_pnl_pct = (highest_price / entry_price) - 1.0 if highest_price > 0 and entry_price > 0 else 0.0
    ctx.profit_giveback = ctx.peak_pnl_pct - ctx.pnl_pct
    return False


# ---------------------------------------------------------------------------
# EXIT / ADD_ENTRY
# ---------------------------------------------------------------------------

def _003_peak_next_bar_bearish_exit(ctx: SellContext) -> bool:
    # 고점봉 바로 다음 확정봉이 음봉이면서 고점 대비 PEAK_NEXT_BAR_DROP_PCT 이상 하락하면 TP_EXTENSION_TRAIL
    # (고점 대비 -1.0%) 도달을 기다리지 않고 즉시 매도(017670 SK텔레콤 2026-08-24 12:40 사례). ATR_TP 도달
    # 이후(peak_pnl_pct>=atr_tp_pct)에만 작동.
    code, pos, price, cur, bar_time = ctx.code, ctx.pos, ctx.price, ctx.cur, ctx.bar_time
    api = ctx.api
    if (
        ENABLE_PEAK_NEXT_BAR_BEARISH_EXIT
        and not pd.isna(ctx.atr_tp_pct)
        and ctx.peak_pnl_pct >= ctx.atr_tp_pct
        and pos.get("peak_bar_time") is not None
        and bar_time > pos["peak_bar_time"]
        and pos.get("peak_next_bar_checked") != pos["peak_bar_time"]
    ):
        pos["peak_next_bar_checked"] = pos["peak_bar_time"]
        cur_open_pb = _num(cur, "open")
        cur_close_pb = _num(cur, "close")
        if not any(pd.isna(v) for v in (cur_open_pb, cur_close_pb)) and ctx.highest_price > 0:
            drop_from_peak_pct = (ctx.highest_price - cur_close_pb) / ctx.highest_price
            if cur_close_pb < cur_open_pb and drop_from_peak_pct >= PEAK_NEXT_BAR_DROP_PCT:
                reason_pb = f"PEAK_NEXT_BAR_BEARISH_{PEAK_NEXT_BAR_DROP_PCT*100:.1f}pct"
                ctx.log(
                    f"  [SELL TRIGGER] {code} | {reason_pb} | "
                    f"peak={ctx.highest_price:,.0f}({pos['peak_bar_time']:%H:%M:%S}) "
                    f"next_bar={bar_time:%H:%M:%S} open={cur_open_pb:,.0f} close={cur_close_pb:,.0f} "
                    f"drop={drop_from_peak_pct*100:.2f}% pnl={ctx.pnl_pct*100:.2f}%"
                )
                ctx.state.trailing_sell_confirm_state.pop(code, None)
                if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_pb, ctx.nxt_tradeable, price=price, code_name=ctx.name, market_order=True):
                    ctx.log(f"  [SELL EXECUTED] {code} | {reason_pb} | qty={pos['quantity']} price={price:,.0f}")
                ctx.state.signal_sell_bar[code] = bar_time
                return True
    return False


def _004_hard_stop_loss(ctx: SellContext) -> bool:
    # 매수 후 HARD_STOP_MIN_HOLD_SECONDS 경과 후 활성화(초기 구간은 POST_BUY guard가 담당)
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    held_for_hard_sl = (
        (ctx.current_dt - pos["buy_time"]).total_seconds()
        if isinstance(pos.get("buy_time"), datetime) else 9999.0
    )
    _, buy_token, _ = _hold_info(ctx)  # position_token만 재사용 - held_for_hard_sl은 위 9999.0 폴백을 그대로 유지
    hard_sl_condition = ctx.pnl_pct <= -HARD_STOP_LOSS_PCT and held_for_hard_sl >= HARD_STOP_MIN_HOLD_SECONDS
    # [2026-09-23] 단일 폴링 틱 노이즈로 즉시 손절되는 걸 막기 위해 HARD_STOP_CONFIRM_SECONDS 동안 조건이
    # 연속 유지될 때만 실행 (119850 지엔씨에너지 2026-09-23 09:02 사례 - 21초 폴링 간격에 -1.27%->-2.00%로
    # 임계값을 그냥 통과해 확인 없이 즉시 발동됨). _019_atr_stop_loss와 동일한 헬퍼/취지(033790 피노
    # 2026-08-31 사례) - 이 조건만 유일하게 확인창이 없었다. HARD_STOP_CONFIRM_SECONDS<=0이면 첫 통과 틱에서
    # 바로 발동해 기존과 동일하게 동작한다(update_timed_condition_state는 첫 통과 시 0.0을 반환).
    # 주의: 이 확인창은 손실 상한을 보장하지 않는다 - 하락이 계속되면 확인 시간만큼 더 나쁜 가격에 체결될 수
    # 있다(일시적 휩쏘 방지 <-> 지속 하락 시 손실 확대의 트레이드오프, Codex 검토로 확인).
    hard_sl_hold_seconds = update_timed_condition_state(
        ctx.state.hard_stop_confirm_state,
        code,
        buy_token,
        ctx.current_dt,
        hard_sl_condition,
    )
    # hard_sl_condition도 함께 확인: update_timed_condition_state는 조건이 False일 때와 막 True가 된 첫
    # 틱 모두 0.0을 반환하므로, HARD_STOP_CONFIRM_SECONDS<=0(롤백값)일 때 hold_seconds>=0.0만 보면 조건이
    # False인 틱에서도 발동해버리는 버그가 됨(2026-09-23 g003 검증 중 발견 - old/new 비교 백테스트에서
    # pnl 0% 근처 매도가 즉시 발생해 발견).
    if hard_sl_condition and hard_sl_hold_seconds >= HARD_STOP_CONFIRM_SECONDS:
        reason_hard_sl = f"HARD_STOP_LOSS_{HARD_STOP_LOSS_PCT*100:.1f}PCT"
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_hard_sl} | "
            f"price={price:,.0f} entry={ctx.entry_price:,.0f} pnl={ctx.pnl_pct*100:.2f}% "
            f"held={held_for_hard_sl:.0f}s confirm={hard_sl_hold_seconds:.0f}s"
        )
        ctx.state.trailing_sell_confirm_state.pop(code, None)
        ctx.state.hard_stop_confirm_state.pop(code, None)
        if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_hard_sl, ctx.nxt_tradeable, price=price, code_name=ctx.name, market_order=True):
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_hard_sl} | qty={pos['quantity']} price={price:,.0f}")
            # 서킷브레이커: HARD_STOP 누적 및 당일 재진입 차단 등록
            _register_hard_stop(ctx, "HARD_STOP")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _005_pyramid_add_entry(ctx: SellContext) -> bool:
    """불타기: 추세 지속 시 1회 추가 진입(ADD_ENTRY - 매도가 아니라 추가 매수 주문)."""
    code, pos, price, cur, frame, api = ctx.code, ctx.pos, ctx.price, ctx.cur, ctx.frame, ctx.api
    if (
        ENABLE_PYRAMIDING
        and not bool(pos.get("pyramid_done", False))
        and not api.has_pending_order(str(code).zfill(6))
        and ctx.pnl_pct >= PYRAMID_TRIGGER_PNL_PCT
    ):
        prev_pyr = frame.iloc[-2] if len(frame) >= 2 else cur
        ma5_cur_pyr = _num(cur, "MA_5")
        ma5_prev_pyr = _num(prev_pyr, "MA_5")
        bb_mid_cur_pyr = _num(cur, "BB_MIDDLE")
        bb_mid_prev_pyr = _num(prev_pyr, "BB_MIDDLE")
        adx_cur_pyr = _num(cur, "ADX")
        adx_prev_pyr = _num(prev_pyr, "ADX")
        pyramid_ok = (
            not any(
                pd.isna(v) for v in (
                    ma5_cur_pyr, ma5_prev_pyr, bb_mid_cur_pyr,
                    bb_mid_prev_pyr, adx_cur_pyr, adx_prev_pyr,
                )
            )
            and ma5_cur_pyr > ma5_prev_pyr
            and bb_mid_cur_pyr > bb_mid_prev_pyr
            and adx_cur_pyr > adx_prev_pyr
        )
        if pyramid_ok:
            pyr_qty = api.get_affordable_buy_qty(code, price, ctx.current_dt, ctx.nxt_tradeable)
            if pyr_qty > 0:
                pyr_session = ctx.services.classify_buy_session(ctx.current_dt)
                reason_pyr = f"PYRAMID_2ND_ENTRY_PNL_{ctx.pnl_pct*100:.2f}PCT"
                ctx.log(
                    f"  [BUY TRIGGER] {code} | {reason_pyr} | "
                    f"add_qty={pyr_qty} price={price:,.0f} avg_entry={ctx.entry_price:,.0f} pnl={ctx.pnl_pct*100:.2f}% | "
                    f"MA5={ma5_prev_pyr:.1f}->{ma5_cur_pyr:.1f} BB_MID={bb_mid_prev_pyr:.1f}->{bb_mid_cur_pyr:.1f} "
                    f"ADX={adx_prev_pyr:.1f}->{adx_cur_pyr:.1f}"
                )
                pos["pyramid_done"] = True
                if api.place_buy_order(
                    code, price, pyr_qty, ctx.current_dt, ctx.nxt_tradeable, pyr_session,
                    buy_detail=reason_pyr, code_name=ctx.name, pyramid=True,
                ):
                    api._record_position_meta(code, pos)
                    api.persist_live_state(date_str=ctx.date_str)
                else:
                    pos["pyramid_done"] = False
                return True
    return False


def _006_staged_tp1_partial(ctx: SellContext) -> bool:
    if not ENABLE_STAGED_TAKE_PROFIT:
        return False
    code, pos, price, cur, api = ctx.code, ctx.pos, ctx.price, ctx.cur, ctx.api
    _staged_targets(ctx)
    entry_qty, tp1_target_pct = ctx.entry_qty, ctx.tp1_target_pct
    atr_tp1_dynamic_pct = ctx.atr_tp1_dynamic_pct

    # 1차 익절: entry_qty의 STAGED_TP1_RATIO(40%), tp1_target_pct(동적) 도달 시
    if (not bool(pos.get("tp1_done", False))) and ctx.pnl_pct >= tp1_target_pct:
        tp1_qty = max(1, int(round(entry_qty * STAGED_TP1_RATIO)))
        tp1_qty = min(tp1_qty, int(pos["quantity"]))
        # 급등 여부는 1차 익절 시점에 판정한다. 급등이면 잔량을 TP1 체결가 기준 +2%/+4% 사다리로 추가 익절
        # (_007), 아니면 기존대로 잔량은 트레일 위임.
        surge_now = False
        surge_detail = ""
        if ENABLE_SURGE_LADDER_TP:
            surge_now, surge_detail = detect_price_surge(
                ctx.state.recent_price_samples.get(code, []),
                ctx.current_dt,
                price,
                ctx.atr_pct,
                _num(cur, "BB_UPPER"),
                _num(cur, "volume"),
                _num(cur, "VOL_MA20"),
                lookback_seconds=SURGE_LOOKBACK_SECONDS,
                speed_min_pct=SURGE_SPEED_MIN_PCT,
                speed_atr_mult=SURGE_SPEED_ATR_MULT,
                volume_ratio_min=SURGE_VOLUME_RATIO_MIN,
                min_confirms=SURGE_MIN_CONFIRMS,
            )
        reason_tp1 = (
            f"TP1_PARTIAL_{STAGED_TP1_RATIO*100:.0f}PCT_{tp1_target_pct*100:.2f}PCT"
            + ("_SURGE" if surge_now else "")
        )
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_tp1} | "
            f"qty={tp1_qty}/{int(pos['quantity'])} price={price:,.0f} pnl={ctx.pnl_pct*100:.2f}% "
            f"target={tp1_target_pct*100:.2f}% (atr_based={atr_tp1_dynamic_pct*100:.2f}%, atr_tp={ctx.atr_tp_pct*100:.2f}%)"
            + (f" | surge={surge_now} [{surge_detail}]" if ENABLE_SURGE_LADDER_TP else "")
        )
        api.last_sell_fill_price.pop(str(code).zfill(6), None)
        if api.place_sell_order(code, tp1_qty, ctx.current_dt, reason_tp1, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            pos["tp1_done"] = True
            pos["entry_quantity"] = entry_qty
            if STAGED_TP2_RATIO <= 0:
                pos["tp2_done"] = True
            if surge_now:
                # 사다리 기준가 = 1차 익절 실제 체결가(체결 확인 전이면 트리거 시점 현재가로 대체).
                tp1_fill_price = float(api.last_sell_fill_price.get(str(code).zfill(6), 0.0) or 0.0)
                ladder_base = tp1_fill_price if tp1_fill_price > 0 else float(price)
                pos["surge_ladder"] = {"base": ladder_base, "tp2_done": False, "tp3_done": False}
                ctx.log(
                    f"  [SURGE_LADDER_ARMED] {code} | 급등 중 1차 익절 -> 잔량 사다리 익절 | "
                    f"base(TP1 체결가)={ladder_base:,.0f} "
                    f"TP2={ladder_base*(1.0+SURGE_TP2_PCT):,.0f}(+{SURGE_TP2_PCT*100:.1f}%) "
                    f"TP3={ladder_base*(1.0+SURGE_TP3_PCT):,.0f}(+{SURGE_TP3_PCT*100:.1f}%)"
                )
            api._record_position_meta(code, pos)
            api.persist_live_state(date_str=ctx.date_str)
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_tp1} | qty={tp1_qty} price={price:,.0f}")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _007_surge_ladder_tp(ctx: SellContext) -> bool:
    """급등 중 1차 익절을 한 포지션의 잔량 사다리 익절: TP1 체결가(base) 기준 +SURGE_TP2_PCT에서 진입수량의
    SURGE_TP2_RATIO, +SURGE_TP3_PCT에서 잔량 전량. 트레일링 스탑/손절은 뒤 조건에서 그대로 병행된다. 주문 성공 시에만
    단계 플래그를 세우므로 쿨다운으로 주문이 막히면 다음 틱에 같은 단계를 재시도한다."""
    if not ENABLE_STAGED_TAKE_PROFIT:
        return False
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    surge_ladder = ctx.services.sanitize_surge_ladder(pos.get("surge_ladder")) if ENABLE_SURGE_LADDER_TP else None
    ctx.surge_ladder = surge_ladder
    if surge_ladder is not None and bool(pos.get("tp1_done", False)):
        ladder_step = next_surge_ladder_action(
            price,
            surge_ladder["base"],
            int(pos["quantity"]),
            ctx.entry_qty,
            surge_ladder["tp2_done"],
            surge_ladder["tp3_done"],
            tp2_pct=SURGE_TP2_PCT,
            tp3_pct=SURGE_TP3_PCT,
            tp2_ratio=SURGE_TP2_RATIO,
        )
        if ladder_step is not None:
            rung, rung_qty, rung_target = ladder_step
            rung_pct = SURGE_TP2_PCT if rung == "TP2" else SURGE_TP3_PCT
            reason_ladder = f"{rung}_SURGE_LADDER_{rung_pct*100:.1f}PCT_OF_TP1"
            ctx.log(
                f"  [SELL TRIGGER] {code} | {reason_ladder} | "
                f"qty={rung_qty}/{int(pos['quantity'])} price={price:,.0f} target={rung_target:,.0f} "
                f"(tp1_base={surge_ladder['base']:,.0f}) pnl={ctx.pnl_pct*100:.2f}%"
                + (" | TP2 미실행 상태에서 TP3 도달 - TP2+TP3 합산 잔량 전량 청산" if rung == "TP3" and not surge_ladder["tp2_done"] else "")
            )
            if api.place_sell_order(code, rung_qty, ctx.current_dt, reason_ladder, ctx.nxt_tradeable, price=price, code_name=ctx.name):
                surge_ladder["tp2_done"] = True
                if rung == "TP3":
                    surge_ladder["tp3_done"] = True
                pos["surge_ladder"] = surge_ladder
                if rung != "TP3":
                    # TP3는 잔량 전량이라 포지션이 닫힌다 - 닫힌 포지션의 메타를 되살리지 않도록 기록하지 않는다.
                    api._record_position_meta(code, pos)
                    api.persist_live_state(date_str=ctx.date_str)
                ctx.log(f"  [SELL EXECUTED] {code} | {reason_ladder} | qty={rung_qty} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = ctx.bar_time
            return True
    return False


def _008_staged_tp2_partial(ctx: SellContext) -> bool:
    # 2차 익절: entry_qty의 STAGED_TP2_RATIO(30%), tp2_target_pct 도달 시 (1차 완료 후). STAGED_TP2_RATIO<=0이면
    # 2차 단계 자체를 건너뛴다. 급등 사다리가 가동 중이면 사다리가 2차 이후를 담당하므로 건너뛴다.
    if not ENABLE_STAGED_TAKE_PROFIT:
        return False
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    if (
        STAGED_TP2_RATIO > 0
        and ctx.surge_ladder is None
        and bool(pos.get("tp1_done", False))
        and (not bool(pos.get("tp2_done", False)))
        and ctx.pnl_pct >= ctx.tp2_target_pct
    ):
        tp2_qty = max(1, int(round(ctx.entry_qty * STAGED_TP2_RATIO)))
        tp2_qty = min(tp2_qty, int(pos["quantity"]))
        reason_tp2 = f"TP2_PARTIAL_{STAGED_TP2_RATIO*100:.0f}PCT_{ctx.tp2_target_pct*100:.2f}PCT"
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_tp2} | "
            f"qty={tp2_qty}/{int(pos['quantity'])} price={price:,.0f} pnl={ctx.pnl_pct*100:.2f}%"
        )
        if api.place_sell_order(code, tp2_qty, ctx.current_dt, reason_tp2, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            pos["tp2_done"] = True
            api._record_position_meta(code, pos)
            api.persist_live_state(date_str=ctx.date_str)
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_tp2} | qty={tp2_qty} price={price:,.0f}")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _009_tp3_trail_arm(ctx: SellContext) -> bool:
    # 3차: 고정 목표가 전량청산 대신, 1/2차 완료 후 잔량(30%)을 트레일링 스탑에 위임한다(Trail 30%). 여기서는 sell
    # 주문을 내지 않고 플래그와 로그만 1회 남긴 뒤 아래 시그널 청산/트레일링 스탑 로직으로 이어지도록 둔다.
    if not ENABLE_STAGED_TAKE_PROFIT:
        return False
    code, pos, price = ctx.code, ctx.pos, ctx.price
    if (
        bool(pos.get("tp1_done", False))
        and bool(pos.get("tp2_done", False))
        and ctx.surge_ladder is None
        and not bool(pos.get("tp3_trail_armed", False))
    ):
        pos["tp3_trail_armed"] = True
        ctx.log(
            f"  [TP3_TRAIL_ARMED] {code} | 1,2차 익절 완료 - 잔량 {int(pos['quantity'])}주 "
            f"고정청산 없이 트레일링 스탑에 위임 | price={price:,.0f} pnl={ctx.pnl_pct*100:.2f}%"
        )
    return False


def _010_legacy_tp_full_and_partial(ctx: SellContext) -> bool:
    """ENABLE_STAGED_TAKE_PROFIT=False일 때만 쓰는 구형 익절(+2.0% 전량 / +1.0% 50% 1회)."""
    if ENABLE_STAGED_TAKE_PROFIT:
        return False
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    # +2.0% full take profit
    if ctx.pnl_pct >= 0.020:
        reason_tp2 = "TP2_FULL_2.0PCT"
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_tp2} | "
            f"price={price:,.0f} entry={ctx.entry_price:,.0f} pnl={ctx.pnl_pct*100:.2f}%"
        )
        ctx.state.trailing_sell_confirm_state.pop(code, None)
        if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_tp2, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_tp2} | qty={pos['quantity']} price={price:,.0f}")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True

    # +1.0% one-time 50% partial take profit
    if (not bool(pos.get("tp1_done", False))) and ctx.pnl_pct >= 0.010:
        partial_qty = max(1, int(int(pos["quantity"]) * 0.5))
        partial_qty = min(partial_qty, int(pos["quantity"]))
        reason_tp1 = "TP1_PARTIAL_50PCT_1.0PCT"
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_tp1} | "
            f"qty={partial_qty}/{int(pos['quantity'])} price={price:,.0f} pnl={ctx.pnl_pct*100:.2f}%"
        )
        if api.place_sell_order(code, partial_qty, ctx.current_dt, reason_tp1, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            pos["tp1_done"] = True
            api._record_position_meta(code, pos)
            api.persist_live_state(date_str=ctx.date_str)
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_tp1} | qty={partial_qty} price={price:,.0f}")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _011_hybrid_1min_dead_cross_exit(ctx: SellContext) -> bool:
    """[2026-09-23] 매수측 _008_hybrid_1min_trigger(1분봉 BB중심선 골든크로스)의 대칭 매도판. 사용자 요청
    ("매수/매도 컨셉 재검토", 204620 글로벌텍스프리 사례) + Codex 설계검토 반영.

    _022_shared_reversal_sell(r002 check_sell_condition, AUX_REVERSAL_SCORE)은 구조적으로 min_pnl_req가
    항상 양수라 손실 포지션을 절대 청산할 수 없다(2026-09-23 컨셉 재검토 메모 참조) - 그 결함을 고치는
    대신(Codex 권고: "수익보호 전용으로 남기고, 손실을 자르는 건 별도 조건으로") 여기서 새로 담당한다.

    손실 구간(pnl<=LOSS_EXIT_PNL_MAX)은 수익 요건 없이 즉시 청산 - Codex 권고 (3). 그 외(수익 중이거나
    얕은 손실)는 1분봉 BB중심선 기울기가 이미 꺾였는지 추가로 확인 - Codex 권고 (4), 정상 상승추세
    눌림목에서의 휩쏘 매도 방지. 확인창(CONFIRM_SECONDS)으로 노이즈도 걸러낸다 - Codex 권고 (5)."""
    if not ENABLE_HYBRID_1MIN_DEADCROSS_EXIT:
        return False
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    _, buy_token, held = _hold_info(ctx)
    if held < HYBRID_1MIN_DEADCROSS_MIN_HOLD_SECONDS:
        return False

    frame_1min = ctx.services.get_frame_1min(code, ctx.current_dt, ctx.nxt_tradeable)
    found, cross_reason, bars_since_cross = check_1min_dead_cross(frame_1min, HYBRID_1MIN_DEADCROSS_LOOKBACK_BARS)
    if not found:
        ctx.state.hybrid_1min_dead_cross_state.pop(code, None)
        return False

    is_loss_exit = ctx.pnl_pct <= HYBRID_1MIN_DEADCROSS_LOSS_EXIT_PNL_MAX
    bb_slope_1min = float("nan")
    if not is_loss_exit:
        # 수익/얕은 손실 구간: BB중심선(1분봉) 자체가 이미 꺾였을 때만 인정 - 상승추세 정상 눌림목 보호.
        bb_slope_1min = _compute_bb_slope_pct(frame_1min)
        if pd.isna(bb_slope_1min) or bb_slope_1min > 0:
            ctx.state.hybrid_1min_dead_cross_state.pop(code, None)
            return False

    dead_cross_hold_seconds = update_timed_condition_state(
        ctx.state.hybrid_1min_dead_cross_state, code, buy_token, ctx.current_dt, True,
    )
    if dead_cross_hold_seconds < HYBRID_1MIN_DEADCROSS_CONFIRM_SECONDS:
        return False

    kind = "LOSS" if is_loss_exit else "TREND_FLIP"
    reason = f"HYBRID_1MIN_DEAD_CROSS_{kind}_{cross_reason}"
    ctx.log(
        f"  [SELL TRIGGER] {code} | {reason} | price={price:,.0f} entry={ctx.entry_price:,.0f} "
        f"pnl={ctx.pnl_pct*100:.2f}% held={held:.0f}s bars_since_cross={bars_since_cross} "
        f"bb_slope_1min={bb_slope_1min:.3f}% confirm={dead_cross_hold_seconds:.0f}s"
    )
    ctx.state.trailing_sell_confirm_state.pop(code, None)
    ctx.state.hybrid_1min_dead_cross_state.pop(code, None)
    if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason, ctx.nxt_tradeable, price=price, code_name=ctx.name, market_order=True):
        ctx.log(f"  [SELL EXECUTED] {code} | {reason} | qty={pos['quantity']} price={price:,.0f}")
    ctx.state.signal_sell_bar[code] = ctx.bar_time
    return True


def _012_peak_retracement_guard(ctx: SellContext) -> bool:
    """[2026-09-23] 사용자 요청("급등 후 꺾이면 고점 대비 -0.8%서 익절") + Codex 설계검토. TP1(+3.0%)/
    ATR익절선 도달 전 구간은 현재 아무 보호장치가 없다 - 그 밑에서 고점을 찍고 반납해도 하드손절이나
    시그널청산이 걸릴 때까지 아무 것도 안 잡는다. 이 공백을 메운다.

    주의(탐색적 구현): 이 파일의 _017_breakeven_fail_guard 이력(BREAKEVEN_FAIL_GIVEBACK_PCT가
    0.8%->2.0%->2.8%로 점진 완화됨 - 백테스트 결과 92.9%가 본전 이상 회복)은 고정 0.8% 되돌림처럼 좁은
    문턱이 정상적인 되돌림도 조기청산할 위험을 보여준다. Codex 권고대로 ATR% 기반으로 문턱을 정해(고정
    MIN_PCT는 하한선일 뿐) 변동성 큰 종목은 자동으로 문턱이 넓어지게 한다."""
    if not ENABLE_PEAK_RETRACE_GUARD:
        return False
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    # [2026-09-23, Codex 재검토] 아래 4개 조기 return은 모두 "이 조건 자체가 이번 틱엔 적용 대상이 아님"을
    # 뜻하므로 진행 중이던 확인창(peak_retrace_guard_state)도 함께 지워야 한다 - 안 지우면 자격을 잃었다
    # 되찾는 사이(예: pnl이 잠깐 0% 이하로 빠졌다가 다시 양전) 이전 타이머가 그대로 이어져, 실제로는
    # 연속되지 않은 되돌림 구간인데도 CONFIRM_SECONDS를 즉시 만족해버릴 수 있다(단일 틱 노이즈 방지라는
    # 확인창의 취지를 무력화).
    if bool(pos.get("tp1_done", False)):
        ctx.state.peak_retrace_guard_state.pop(code, None)
        return False  # TP1 이후는 사다리/TP2/트레일링 스탑이 담당
    if not pd.isna(ctx.atr_tp_pct) and ctx.peak_pnl_pct >= ctx.atr_tp_pct:
        ctx.state.peak_retrace_guard_state.pop(code, None)
        return False  # 이미 ATR 익절선 도달 - 확장 트레일링 영역, 그쪽(_015/_020)에 위임
    if ctx.pnl_pct <= 0:
        ctx.state.peak_retrace_guard_state.pop(code, None)
        return False  # 손실/본전 이하는 _017_breakeven_fail_guard 담당
    if ctx.peak_pnl_pct < PEAK_RETRACE_GUARD_ARM_PNL:
        ctx.state.peak_retrace_guard_state.pop(code, None)
        return False

    _, buy_token, held = _hold_info(ctx)
    retrace_threshold = max(
        PEAK_RETRACE_GUARD_MIN_PCT,
        (ctx.atr_pct * PEAK_RETRACE_GUARD_ATR_MULT) if not pd.isna(ctx.atr_pct) else 0.0,
    )
    condition = ctx.profit_giveback >= retrace_threshold
    hold_seconds = update_timed_condition_state(
        ctx.state.peak_retrace_guard_state, code, buy_token, ctx.current_dt, condition,
    )
    # condition도 함께 확인 - update_timed_condition_state는 조건 False/막 True 첫 틱 모두 0.0을 반환하므로
    # PEAK_RETRACE_GUARD_CONFIRM_SECONDS<=0(롤백값)이면 조건 False에서도 발동하는 잠재 버그가 된다
    # (_004_hard_stop_loss/_019_atr_stop_loss와 동일 패턴, 2026-09-23 발견 당시 이 함수엔 미적용).
    if not condition or hold_seconds < PEAK_RETRACE_GUARD_CONFIRM_SECONDS:
        return False

    reason = f"PEAK_RETRACE_GUARD_{retrace_threshold*100:.2f}pct"
    ctx.log(
        f"  [SELL TRIGGER] {code} | {reason} | price={price:,.0f} entry={ctx.entry_price:,.0f} "
        f"peak={ctx.highest_price:,.0f} pnl={ctx.pnl_pct*100:.2f}% peak_pnl={ctx.peak_pnl_pct*100:.2f}% "
        f"giveback={ctx.profit_giveback*100:.2f}% atr_pct={ctx.atr_pct*100:.2f}% held={held:.0f}s "
        f"confirm={hold_seconds:.0f}s"
    )
    ctx.state.trailing_sell_confirm_state.pop(code, None)
    ctx.state.peak_retrace_guard_state.pop(code, None)
    if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason, ctx.nxt_tradeable, price=price, code_name=ctx.name, market_order=True):
        ctx.log(f"  [SELL EXECUTED] {code} | {reason} | qty={pos['quantity']} price={price:,.0f}")
    ctx.state.signal_sell_bar[code] = ctx.bar_time
    return True


def _013_signal_exit_stoch_k_lt_d(ctx: SellContext) -> bool:
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    s = _signal_trend_inputs(ctx)
    k_now, d_now = s["k_now"], s["d_now"]
    if not any(pd.isna(v) for v in (k_now, d_now)) and k_now < d_now:
        if s["strong_uptrend"] or s["sig_held_seconds"] < SIGNAL_EXIT_MIN_HOLD_SECONDS or ctx.pnl_pct > SIGNAL_EXIT_STOCH_SUPPRESS_PNL_MIN:
            ctx.log(
                f"  [SELL SKIP] {code} | STOCH_K_LT_D suppressed | "
                f"K={k_now:.1f} D={d_now:.1f} pnl={ctx.pnl_pct*100:.2f}% held={s['sig_held_seconds']:.0f}s "
                f"uptrend(adx={s['adx_uptrend']},price={s['price_uptrend']})"
            )
        else:
            reason_sig_kd = "SIGNAL_EXIT_STOCH_K_LT_D"
            ctx.log(
                f"  [SELL TRIGGER] {code} | {reason_sig_kd} | "
                f"K={k_now:.1f} D={d_now:.1f} pnl={ctx.pnl_pct*100:.2f}%"
            )
            ctx.state.trailing_sell_confirm_state.pop(code, None)
            if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_sig_kd, ctx.nxt_tradeable, price=price, code_name=ctx.name):
                ctx.log(f"  [SELL EXECUTED] {code} | {reason_sig_kd} | qty={pos['quantity']} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = ctx.bar_time
            return True
    return False


def _014_signal_exit_macd_hist_down(ctx: SellContext) -> bool:
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    s = _signal_trend_inputs(ctx)
    hist_now, hist_prev, hist_prev2 = s["hist_now"], s["hist_prev"], s["hist_prev2"]
    if (not any(pd.isna(v) for v in (hist_now, hist_prev, hist_prev2))
            and hist_now < hist_prev < hist_prev2
            and not s["strong_uptrend"]
            and s["sig_held_seconds"] >= SIGNAL_EXIT_MIN_HOLD_SECONDS
            and ctx.pnl_pct <= SIGNAL_EXIT_MACD_PNL_MAX):
        reason_sig_macd = "SIGNAL_EXIT_MACD_HIST_DOWN_2BARS"
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_sig_macd} | "
            f"HIST={hist_prev2:.3f}->{hist_prev:.3f}->{hist_now:.3f} pnl={ctx.pnl_pct*100:.2f}%"
        )
        ctx.state.trailing_sell_confirm_state.pop(code, None)
        if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_sig_macd, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_sig_macd} | qty={pos['quantity']} price={price:,.0f}")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _015_atr_take_profit(ctx: SellContext) -> bool:
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    if not pd.isna(ctx.atr_tp_pct) and ctx.pnl_pct >= ctx.atr_tp_pct:
        if ENABLE_TP_EXTENSION_TRAILING:
            # TP 도달 시 즉시 익절 대신 고점 트레일링 모드로 전환 (주문 없이 로그만 - FALL_THROUGH)
            ctx.log(
                f"  [TP_EXTENSION] {code} | pnl={ctx.pnl_pct*100:.2f}% >= ATR_TP {ctx.atr_tp_pct*100:.2f}% | "
                f"고점 트레일링 모드 전환 (trail={TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%) | "
                f"price={price:,.0f} peak={ctx.highest_price:,.0f} atr={float(ctx.atr_val):.2f}"
            )
        else:
            reason_tp = f"ATR_TAKE_PROFIT_{ATR_TAKE_PROFIT_MULTIPLIER:.1f}x"
            ctx.log(
                f"  [SELL TRIGGER] {code} | {reason_tp} | price={price:,.0f} entry={ctx.entry_price:,.0f} "
                f"pnl={ctx.pnl_pct*100:.2f}% atr={float(ctx.atr_val):.2f} tp={ctx.atr_tp_price:,.0f}"
            )
            if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_tp, ctx.nxt_tradeable, price=price, code_name=ctx.name):
                ctx.log(f"  [SELL EXECUTED] {code} | {reason_tp} | qty={pos['quantity']} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = ctx.bar_time
            return True
    return False


def _clear_timed_guard_states(ctx: SellContext) -> None:
    """가드 매도 직전 공통 정리(기존 4875-4879/4909-4913/4948-4952)."""
    code, st = ctx.code, ctx.state
    st.post_buy_bb_drop_state.pop(code, None)
    st.breakeven_fail_state.pop(code, None)
    st.no_trend_exit_state.pop(code, None)
    st.trailing_sell_confirm_state.pop(code, None)
    st.atr_stop_confirm_state.pop(code, None)
    st.hybrid_1min_dead_cross_state.pop(code, None)
    st.peak_retrace_guard_state.pop(code, None)
    st.hard_stop_confirm_state.pop(code, None)


def _016_post_buy_entry_drop_guard(ctx: SellContext) -> bool:
    code, pos, price, api, entry_price = ctx.code, ctx.pos, ctx.price, ctx.api, ctx.entry_price
    _buy_time, buy_token, held_for_guard = _hold_info(ctx)
    if held_for_guard <= POST_BUY_BB_DROP_ARMED_SECONDS:
        drop_condition = price < entry_price * (1.0 - POST_BUY_BB_DROP_PCT)
        drop_hold_seconds = update_timed_condition_state(
            ctx.state.post_buy_bb_drop_state,
            code,
            buy_token,
            ctx.current_dt,
            drop_condition,
        )
        # drop_condition도 함께 확인 - POST_BUY_DROP_CONFIRM_SECONDS<=0에서 조건 False에도 발동하는
        # 잠재 버그 방지 (_004_hard_stop_loss와 동일 패턴).
        if drop_condition and drop_hold_seconds >= POST_BUY_DROP_CONFIRM_SECONDS:
            drop_pct_guard = (price / entry_price - 1.0) * 100.0
            reason_bbdrop = f"POST_BUY_ENTRY_DROP_{POST_BUY_BB_DROP_PCT*100:.1f}pct_{POST_BUY_DROP_CONFIRM_SECONDS:.0f}s"
            ctx.log(
                f"  [SELL TRIGGER] {code} | {reason_bbdrop} | "
                f"held={held_for_guard:.0f}s price={price:,.0f} entry={entry_price:,.0f} "
                f"drop={drop_pct_guard:.2f}% hold={drop_hold_seconds:.0f}s pnl={ctx.pnl_pct*100:.2f}%"
            )
            _clear_timed_guard_states(ctx)
            if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_bbdrop, ctx.nxt_tradeable, price=price, code_name=ctx.name):
                ctx.log(f"  [SELL EXECUTED] {code} | {reason_bbdrop} | qty={pos['quantity']} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = ctx.bar_time
            return True
    else:
        ctx.state.post_buy_bb_drop_state.pop(code, None)
    return False


def _017_breakeven_fail_guard(ctx: SellContext) -> bool:
    code, pos, price, api, entry_price = ctx.code, ctx.pos, ctx.price, ctx.api, ctx.entry_price
    _buy_time, buy_token, held_for_guard = _hold_info(ctx)
    if not ctx.is_startup_position:
        breakeven_condition = (
            ctx.peak_pnl_pct >= BREAKEVEN_FAIL_ARM_PNL
            and ctx.pnl_pct < -0.005
            and ctx.profit_giveback >= BREAKEVEN_FAIL_GIVEBACK_PCT
        )
        breakeven_hold_seconds = update_timed_condition_state(
            ctx.state.breakeven_fail_state,
            code,
            buy_token,
            ctx.current_dt,
            breakeven_condition,
        )
        # breakeven_condition도 함께 확인 - BREAKEVEN_FAIL_CONFIRM_SECONDS<=0에서 조건 False에도 발동하는
        # 잠재 버그 방지 (_004_hard_stop_loss와 동일 패턴).
        if breakeven_condition and breakeven_hold_seconds >= BREAKEVEN_FAIL_CONFIRM_SECONDS:
            reason_breakeven = (
                f"BREAKEVEN_FAIL_peak{BREAKEVEN_FAIL_ARM_PNL*100:.1f}_"
                f"giveback{BREAKEVEN_FAIL_GIVEBACK_PCT*100:.2f}_{BREAKEVEN_FAIL_CONFIRM_SECONDS:.0f}s"
            )
            ctx.log(
                f"  [SELL TRIGGER] {code} | {reason_breakeven} | held={held_for_guard:.0f}s "
                f"price={price:,.0f} entry={entry_price:,.0f} peak={ctx.highest_price:,.0f} "
                f"peak_pnl={ctx.peak_pnl_pct*100:.2f}% giveback={ctx.profit_giveback*100:.2f}%"
            )
            _clear_timed_guard_states(ctx)
            if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_breakeven, ctx.nxt_tradeable, price=price, code_name=ctx.name):
                ctx.log(f"  [SELL EXECUTED] {code} | {reason_breakeven} | qty={pos['quantity']} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = ctx.bar_time
            return True
    else:
        ctx.state.breakeven_fail_state.pop(code, None)
    return False


def _018_no_trend_time_exit(ctx: SellContext) -> bool:
    code, pos, price, api, cur = ctx.code, ctx.pos, ctx.price, ctx.api, ctx.cur
    _buy_time, buy_token, held_for_guard = _hold_info(ctx)
    bb_mid_guard = _num(cur, "BB_MIDDLE")
    no_trend_condition = (
        held_for_guard >= NO_TREND_EXIT_ARM_SECONDS
        and ctx.peak_pnl_pct <= NO_TREND_EXIT_MAX_PEAK_PNL
        and ctx.pnl_pct <= NO_TREND_EXIT_MIN_PNL
        and bb_mid_guard > 0
        and price < bb_mid_guard
    )
    no_trend_hold_seconds = update_timed_condition_state(
        ctx.state.no_trend_exit_state,
        code,
        buy_token,
        ctx.current_dt,
        no_trend_condition,
    )
    # no_trend_condition도 함께 확인 - NO_TREND_EXIT_CONFIRM_SECONDS<=0에서 조건 False에도 발동하는 잠재
    # 버그 방지 (_004_hard_stop_loss와 동일 패턴).
    if no_trend_condition and no_trend_hold_seconds >= NO_TREND_EXIT_CONFIRM_SECONDS:
        reason_no_trend = (
            f"NO_TREND_EXIT_{NO_TREND_EXIT_ARM_SECONDS/60:.0f}m_"
            f"peakLT{NO_TREND_EXIT_MAX_PEAK_PNL*100:.1f}_{NO_TREND_EXIT_CONFIRM_SECONDS:.0f}s"
        )
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_no_trend} | held={held_for_guard:.0f}s "
            f"price={price:,.0f} bb_mid={bb_mid_guard:,.1f} pnl={ctx.pnl_pct*100:.2f}% "
            f"peak_pnl={ctx.peak_pnl_pct*100:.2f}% hold={no_trend_hold_seconds:.0f}s"
        )
        _clear_timed_guard_states(ctx)
        if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_no_trend, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_no_trend} | qty={pos['quantity']} price={price:,.0f}")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _019_atr_stop_loss(ctx: SellContext) -> bool:
    code, pos, price, api = ctx.code, ctx.pos, ctx.price, ctx.api
    buy_time, buy_token, _held_for_guard = _hold_info(ctx)
    held_sl = (ctx.current_dt - buy_time).total_seconds()
    atr_sl_condition = (
        not pd.isna(ctx.atr_sl_pct) and ctx.pnl_sl <= ctx.atr_sl_pct and held_sl >= HARD_STOP_MIN_HOLD_SECONDS
    )
    # 단일 폴링 틱 노이즈로 즉시 손절되는 걸 막기 위해 ATR_STOP_CONFIRM_SECONDS 동안 조건이 연속 유지될 때만
    # 실행 (033790 피노 2026-08-31 13:35 사례 참조).
    atr_sl_hold_seconds = update_timed_condition_state(
        ctx.state.atr_stop_confirm_state,
        code,
        buy_token,
        ctx.current_dt,
        atr_sl_condition,
    )
    # atr_sl_condition도 함께 확인 - update_timed_condition_state는 조건 False/막 True 첫 틱 모두 0.0을
    # 반환하므로 ATR_STOP_CONFIRM_SECONDS<=0으로 설정하면 조건 False에서도 발동하는 잠재 버그가 있다
    # (평소 기본값 20.0이라 드러나지 않았음 - 2026-09-23 하드스탑 확인창 추가 중 동일 패턴에서 발견해 방어적으로 수정).
    if atr_sl_condition and atr_sl_hold_seconds >= ATR_STOP_CONFIRM_SECONDS:
        reason_sl = f"ATR_STOP_LOSS_{ATR_STOP_MULTIPLIER:.1f}x"
        ctx.log(
            f"  [SELL TRIGGER] {code} | {reason_sl} | held={held_sl:.0f}s price={price:,.0f} "
            f"bar_low={ctx.bar_low:,.0f} entry={ctx.entry_price:,.0f} pnl={ctx.pnl_pct*100:.2f}% "
            f"sl_pnl={ctx.pnl_sl*100:.2f}% atr={float(ctx.atr_val):.2f} sl={ctx.atr_sl_price:,.0f} "
            f"confirm={atr_sl_hold_seconds:.0f}s"
        )
        ctx.state.trailing_sell_confirm_state.pop(code, None)
        ctx.state.atr_stop_confirm_state.pop(code, None)
        if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_sl, ctx.nxt_tradeable, price=price, code_name=ctx.name, market_order=True):
            ctx.log(f"  [SELL EXECUTED] {code} | {reason_sl} | qty={pos['quantity']} price={price:,.0f}")
            # ATR_STOP_LOSS도 HARD_STOP_LOSS와 동일하게 당일 재진입 차단/서킷브레이커에 반영한다(2026-07-20 로그
            # 분석: 당일 손실의 87%가 ATR_STOP_LOSS였는데 이 카운터는 HARD_STOP_LOSS만 추적했음).
            _register_hard_stop(ctx, "STOP_LOSS(ATR/HARD)")
        ctx.state.signal_sell_bar[code] = ctx.bar_time
        return True
    return False


def _020_trailing_stop(ctx: SellContext) -> bool:
    code, pos, price, api, bar_time = ctx.code, ctx.pos, ctx.price, ctx.api, ctx.bar_time
    highest_price, entry_price, peak_pnl_pct = ctx.highest_price, ctx.entry_price, ctx.peak_pnl_pct
    trailing_sell_confirm_state = ctx.state.trailing_sell_confirm_state
    if highest_price > 0 and entry_price > 0:
        # Entry-anchored trailing stop: 1) peak가 먼저 수익 구간으로 이동, 2) 그 뒤 수익이 트레일 폭만큼 되돌려질
        # 때만 매도, 3) 현재 손익이 0 이하일 때는 트레일링 스탑을 발동하지 않는다.
        # TP 도달 구간(peak >= TP)에서 이익이 1% 이내로 줄면 트레일링 적용
        current_pnl_pct = (price / entry_price) - 1.0
        profit_giveback = peak_pnl_pct - current_pnl_pct
        ctx.profit_giveback = profit_giveback
        if ENABLE_TP_EXTENSION_TRAILING and (
            bool(pos.get("tp1_done", False))
            or (not pd.isna(ctx.atr_tp_pct) and peak_pnl_pct >= ctx.atr_tp_pct)
        ):
            trail_threshold = TP_EXTENSION_TRAIL_FROM_PEAK
            reason_ts = f"TP_EXTENSION_TRAIL_{TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%"
        else:
            trail_threshold = TRAILING_STOP_FROM_PEAK
            reason_ts = f"TRAILING_STOP_GIVEBACK_{TRAILING_STOP_FROM_PEAK*100:.1f}%"
        trailing_condition = peak_pnl_pct > 0 and current_pnl_pct > 0 and profit_giveback >= trail_threshold

        pending_state = trailing_sell_confirm_state.get(code)
        if reason_ts.startswith("TRAILING_STOP_GIVEBACK_"):
            # Clear pending if retrace condition has recovered.
            if not trailing_condition and pending_state is not None:
                trailing_sell_confirm_state.pop(code, None)
                ctx.log(
                    f"  [SELL HOLD CANCEL] {code} | trailing recovered before confirm | "
                    f"pnl={current_pnl_pct*100:.2f}% peak_pnl={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                )

        if trailing_condition:
            if reason_ts.startswith("TRAILING_STOP_GIVEBACK_"):
                # First hit: defer to next 3-minute bar confirmation.
                if pending_state is None:
                    trailing_sell_confirm_state[code] = {
                        "trigger_bar_time": bar_time,
                        "triggered_at": ctx.current_dt,
                        "reason": reason_ts,
                    }
                    ctx.log(
                        f"  [SELL HOLD] {code} | {reason_ts} first hit, wait next 3m bar confirm | "
                        f"bar={bar_time:%H:%M:%S} pnl={current_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                    )
                    return True

                # Still same bar: keep waiting.
                if pending_state.get("trigger_bar_time") == bar_time:
                    return True

            ctx.log(
                f"  [SELL TRIGGER] {code} | {reason_ts} | "
                f"price={price:,.0f} entry={entry_price:,.0f} peak={highest_price:,.0f} | "
                f"pnl={current_pnl_pct*100:.2f}% peak_pnl={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
            )
            trailing_sell_confirm_state.pop(code, None)
            if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, reason_ts, ctx.nxt_tradeable, price=price, code_name=ctx.name):
                ctx.log(f"  [SELL EXECUTED] {code} | {reason_ts} | qty={pos['quantity']} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = bar_time
            return True
    return False


def _021_same_bar_sell_dedup(ctx: SellContext) -> bool:
    return ctx.state.signal_sell_bar.get(ctx.code) == ctx.bar_time


def _022_shared_reversal_sell(ctx: SellContext) -> bool:
    """r002 check_sell_condition(BB 중심선 하향돌파 / 보조 반전 점수) 결과를 실행한다. 마지막 조건이라 어느 경로든
    이 종목의 이번 틱 처리는 여기서 끝난다(기존에는 명시적 continue 없이 블록 끝으로 흘렀음)."""
    code, pos, price, api, cur, frame = ctx.code, ctx.pos, ctx.price, ctx.api, ctx.cur, ctx.frame
    # [2026-09-16] frame이 세션 오픈 직후 1개 봉만 가진 경우 iloc[-2]가 out-of-bounds로 터지는 문제 -
    # 직전 봉이 없으면 현재 봉으로 대체.
    prev_bar = frame.iloc[-2] if len(frame) >= 2 else cur
    sell_ok, sell_reason = ctx.services.check_sell_condition(frame, ctx.pnl_pct, price, ctx.cross_info)
    if sell_ok:
        aux_score = ctx.services.extract_aux_score_from_reason(sell_reason)
        if aux_score is not None:
            aux_base_min = ctx.services.aux_min_pnl_for_score(aux_score)
            if aux_base_min is not None:
                aux_required_pnl = max(
                    AUX_SELL_MIN_REALIZED_TARGET_PCT,
                    aux_base_min + AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT,
                )
                if ctx.pnl_pct < aux_required_pnl:
                    ctx.log(
                        f"  [SELL HOLD] {code} | AUX_TRIGGER_BUFFER_BLOCK "
                        f"score={aux_score} pnl={ctx.pnl_pct*100:.2f}% "
                        f"required>={aux_required_pnl*100:.2f}% "
                        f"(base={aux_base_min*100:.2f}%+buffer={AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT*100:.2f}%)"
                    )
                    return True
        if api.place_sell_order(code, int(pos["quantity"]), ctx.current_dt, sell_reason, ctx.nxt_tradeable, price=price, code_name=ctx.name):
            ctx.log(
                f"  [SELL EVAL] {code} | OK {sell_reason} | {ctx.current_dt:%H:%M:%S} | "
                f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}->{_num(cur, 'BB_MIDDLE'):.1f} | "
                f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                f"K={_num(prev_bar, 'STOCH_K'):.1f}->{_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}->{_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                f"MACD {_num(prev_bar, 'MACD'):.2f}->{_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} | "
                f"ADX={_num(cur, 'ADX'):.1f} | pnl={ctx.pnl_pct*100:.2f}% peak={ctx.highest_price:,.0f}"
            )
            ctx.log(f"  [SELL EXECUTED] {code} | {sell_reason} | qty={pos['quantity']} price={price:,.0f}")
            ctx.state.signal_sell_bar[code] = ctx.bar_time
    elif (
        sell_reason.startswith("AUX_BLOCKED")
        or sell_reason.startswith("LIVE_PRICE_BB_DOWN_CROSS_WEAK_SCORE")
        or sell_reason.startswith("LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_SCORE")
        or sell_reason.startswith("BOX_RANGE_HOLD")
    ):
        ctx.log(f"  [SELL HOLD] {code} | {sell_reason}")
    return True


# ---------------------------------------------------------------------------
# Registry + main function
# ---------------------------------------------------------------------------

SELL_CONDITIONS: tuple[SellCondition, ...] = (
    SellCondition(1, "_001_stale_live_price_guard", "GUARD", "stale 현재가 보호",
                  "현재가가 stale 캐시/봉 종가 대체값이면 이번 틱 매도 판단을 건너뜀", ("LIVE_PRICE_STALE_TTL_SECONDS",), _001_stale_live_price_guard),
    SellCondition(2, "_002_update_position_tracking", "UPDATE", "포지션 추적 갱신",
                  "현재가/최근 가격 표본/최고가/고점봉/ATR 익절·손절선/최고 손익 갱신 (항상 통과)",
                  ("ATR_TAKE_PROFIT_MULTIPLIER", "ATR_STOP_MULTIPLIER", "SURGE_LOOKBACK_SECONDS"), _002_update_position_tracking),
    SellCondition(3, "_003_peak_next_bar_bearish_exit", "EXIT", "고점 다음봉 음봉 즉시 청산",
                  "ATR 익절선 도달 후 고점봉 바로 다음 확정봉이 음봉이고 고점 대비 PEAK_NEXT_BAR_DROP_PCT 이상 하락하면 시장가 전량 청산",
                  ("ENABLE_PEAK_NEXT_BAR_BEARISH_EXIT", "PEAK_NEXT_BAR_DROP_PCT"), _003_peak_next_bar_bearish_exit),
    SellCondition(4, "_004_hard_stop_loss", "EXIT", "하드 손절",
                  "손익 <= -HARD_STOP_LOSS_PCT 이고 보유 HARD_STOP_MIN_HOLD_SECONDS 경과가 HARD_STOP_CONFIRM_SECONDS 지속되면 시장가 전량 청산 + 당일 재진입 차단/서킷브레이커 집계",
                  ("HARD_STOP_LOSS_PCT", "HARD_STOP_MIN_HOLD_SECONDS", "HARD_STOP_CONFIRM_SECONDS", "HARD_STOP_CIRCUIT_BREAKER_COUNT", "HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN"), _004_hard_stop_loss),
    SellCondition(5, "_005_pyramid_add_entry", "ADD_ENTRY", "불타기(추가 매수)",
                  "수익 PYRAMID_TRIGGER_PNL_PCT 이상 + MA5/BB중심선/ADX 상승 시 1회 추가 매수 (매도 아님)",
                  ("ENABLE_PYRAMIDING", "PYRAMID_TRIGGER_PNL_PCT"), _005_pyramid_add_entry),
    SellCondition(6, "_006_staged_tp1_partial", "EXIT", "1차 분할익절",
                  "손익 >= min(max(STAGED_TP1_PCT, ATR%xTP1_ATR_MULTIPLIER), ATR 익절선) 도달 시 진입수량의 STAGED_TP1_RATIO 매도. 급등이면 사다리 무장",
                  ("ENABLE_STAGED_TAKE_PROFIT", "STAGED_TP1_PCT", "STAGED_TP1_RATIO", "TP1_ATR_MULTIPLIER", "ENABLE_TP1_CAP_AT_ATR_TP",
                   "ENABLE_SURGE_LADDER_TP", "SURGE_SPEED_MIN_PCT", "SURGE_SPEED_ATR_MULT", "SURGE_VOLUME_RATIO_MIN", "SURGE_MIN_CONFIRMS"), _006_staged_tp1_partial),
    SellCondition(7, "_007_surge_ladder_tp", "EXIT", "급등 사다리 익절(TP2/TP3)",
                  "급등 중 1차 익절한 포지션: TP1 체결가 +SURGE_TP2_PCT에서 진입수량 일부, +SURGE_TP3_PCT에서 잔량 전량",
                  ("SURGE_TP2_PCT", "SURGE_TP3_PCT", "SURGE_TP2_RATIO"), _007_surge_ladder_tp),
    SellCondition(8, "_008_staged_tp2_partial", "EXIT", "2차 분할익절",
                  "1차 완료 후 손익 >= TP1 목표 + (STAGED_TP2_PCT-STAGED_TP1_PCT) 도달 시 진입수량의 STAGED_TP2_RATIO 매도 (사다리 가동 중이면 건너뜀)",
                  ("STAGED_TP2_PCT", "STAGED_TP2_RATIO"), _008_staged_tp2_partial),
    SellCondition(9, "_009_tp3_trail_arm", "STATE", "3차 트레일 무장",
                  "1·2차 완료 후 잔량을 고정 청산 없이 트레일링 스탑에 위임 (플래그+로그만, 통과)", (), _009_tp3_trail_arm),
    SellCondition(10, "_010_legacy_tp_full_and_partial", "EXIT", "구형 익절(비활성 시에만)",
                  "ENABLE_STAGED_TAKE_PROFIT=False일 때만: +2.0% 전량 / +1.0% 50% 1회", ("ENABLE_STAGED_TAKE_PROFIT",), _010_legacy_tp_full_and_partial),
    SellCondition(11, "_011_hybrid_1min_dead_cross_exit", "EXIT", "1분봉 BB중심선 데드크로스 청산",
                  "1분봉 데드크로스(신선/룩백) + 확인창 지속 시 전량 시장가 청산 - 손실구간은 즉시, "
                  "수익구간은 1분봉 BB기울기 하향 전환까지 확인 (r002 check_1min_dead_cross)",
                  ("ENABLE_HYBRID_1MIN_DEADCROSS_EXIT", "HYBRID_1MIN_DEADCROSS_LOOKBACK_BARS",
                   "HYBRID_1MIN_DEADCROSS_MIN_HOLD_SECONDS", "HYBRID_1MIN_DEADCROSS_CONFIRM_SECONDS",
                   "HYBRID_1MIN_DEADCROSS_LOSS_EXIT_PNL_MAX"), _011_hybrid_1min_dead_cross_exit),
    SellCondition(12, "_012_peak_retracement_guard", "EXIT", "고점 대비 되돌림 익절가드",
                  "TP1/ATR익절선 도달 전, 고점(peak_pnl>=ARM_PNL) 대비 되돌림이 max(MIN_PCT, ATR%xATR_MULT) "
                  "이상 지속되면 전량 시장가 익절 청산",
                  ("ENABLE_PEAK_RETRACE_GUARD", "PEAK_RETRACE_GUARD_ARM_PNL", "PEAK_RETRACE_GUARD_MIN_PCT",
                   "PEAK_RETRACE_GUARD_ATR_MULT", "PEAK_RETRACE_GUARD_CONFIRM_SECONDS"), _012_peak_retracement_guard),
    SellCondition(13, "_013_signal_exit_stoch_k_lt_d", "EXIT", "시그널 청산: 스토캐스틱 K<D",
                  "%K<%D이면 전량 청산. 강한 상승추세/최소 보유시간 미달/수익 충분 시에는 억제(로그만)",
                  ("SIGNAL_EXIT_MIN_HOLD_SECONDS", "SIGNAL_EXIT_STRONG_TREND_ADX_MIN", "SIGNAL_EXIT_STOCH_SUPPRESS_PNL_MIN"), _013_signal_exit_stoch_k_lt_d),
    SellCondition(14, "_014_signal_exit_macd_hist_down", "EXIT", "시그널 청산: MACD 히스토그램 2봉 하락",
                  "MACD_HIST 2봉 연속 하락 + 상승추세 아님 + 최소 보유시간 + 손익 <= SIGNAL_EXIT_MACD_PNL_MAX 이면 전량 청산",
                  ("SIGNAL_EXIT_MACD_PNL_MAX", "SIGNAL_EXIT_MIN_HOLD_SECONDS"), _014_signal_exit_macd_hist_down),
    SellCondition(15, "_015_atr_take_profit", "EXIT", "ATR 익절선 도달",
                  "손익 >= ATR 익절선: 트레일 확장 모드면 로그만(트레일 무장), 아니면 전량 익절",
                  ("ATR_TAKE_PROFIT_MULTIPLIER", "ENABLE_TP_EXTENSION_TRAILING", "TP_EXTENSION_TRAIL_FROM_PEAK"), _015_atr_take_profit),
    SellCondition(16, "_016_post_buy_entry_drop_guard", "EXIT", "매수 직후 진입가 이탈 가드",
                  "매수 후 POST_BUY_BB_DROP_ARMED_SECONDS 이내 진입가 대비 POST_BUY_BB_DROP_PCT 이상 하락이 POST_BUY_DROP_CONFIRM_SECONDS 지속되면 청산",
                  ("POST_BUY_BB_DROP_ARMED_SECONDS", "POST_BUY_BB_DROP_PCT", "POST_BUY_DROP_CONFIRM_SECONDS"), _016_post_buy_entry_drop_guard),
    SellCondition(17, "_017_breakeven_fail_guard", "EXIT", "본전 이탈 가드",
                  "고점 수익이 났다가 손실로 반납(peak>=ARM, pnl<-0.5%, giveback>=GIVEBACK)이 CONFIRM초 지속되면 청산",
                  ("BREAKEVEN_FAIL_ARM_PNL", "BREAKEVEN_FAIL_GIVEBACK_PCT", "BREAKEVEN_FAIL_CONFIRM_SECONDS"), _017_breakeven_fail_guard),
    SellCondition(18, "_018_no_trend_time_exit", "EXIT", "무추세 시간 청산",
                  "보유 NO_TREND_EXIT_ARM_SECONDS 이상, 고점 수익 미미, 손실 + BB중심선 아래가 CONFIRM초 지속되면 청산",
                  ("NO_TREND_EXIT_ARM_SECONDS", "NO_TREND_EXIT_MAX_PEAK_PNL", "NO_TREND_EXIT_MIN_PNL", "NO_TREND_EXIT_CONFIRM_SECONDS"), _018_no_trend_time_exit),
    SellCondition(19, "_019_atr_stop_loss", "EXIT", "ATR 손절",
                  "손익 <= ATR 손절선이 ATR_STOP_CONFIRM_SECONDS 지속되면 시장가 전량 청산 + 재진입 차단/서킷브레이커 집계",
                  ("ATR_STOP_MULTIPLIER", "ATR_STOP_CONFIRM_SECONDS", "HARD_STOP_MIN_HOLD_SECONDS"), _019_atr_stop_loss),
    SellCondition(20, "_020_trailing_stop", "EXIT", "트레일링 스탑",
                  "고점 대비 되돌림이 트레일 폭 이상이면 청산. 확장 트레일은 즉시, 일반 트레일은 다음 3분봉 확인 후",
                  ("TP_EXTENSION_TRAIL_FROM_PEAK", "TRAILING_STOP_FROM_PEAK"), _020_trailing_stop),
    SellCondition(21, "_021_same_bar_sell_dedup", "GUARD", "같은 봉 재매도 차단",
                  "이 3분봉에서 이미 매도 신호를 낸 종목은 건너뜀 (signal_sell_bar)", (), _021_same_bar_sell_dedup),
    SellCondition(22, "_022_shared_reversal_sell", "EXIT", "반전 신호 매도(r002)",
                  "BB 중심선 하향돌파/보조 반전 점수 (r002 check_sell_condition) + 점수별 최소 수익 요건/슬리피지 버퍼",
                  ("AUX_SELL_MIN_PNL_SCORE2", "AUX_SELL_MIN_PNL_SCORE3", "AUX_SELL_MIN_PNL_SCORE4", "AUX_SELL_MIN_REALIZED_TARGET_PCT",
                   "AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT"), _022_shared_reversal_sell),
)


def evaluate_sell_conditions(ctx: SellContext, conditions: tuple[SellCondition, ...] | None = None) -> SellDecision:
    """보유 종목 메인 판정: 조건을 번호 순서대로 평가하고 STOP_SYMBOL(True)을 반환한 첫 조건에서 멈춘다."""
    for cond in (SELL_CONDITIONS if conditions is None else conditions):
        if cond.run(ctx):
            return SellDecision(stopped_by=cond)
    return SellDecision(stopped_by=None)


def describe_sell_conditions() -> list[dict[str, object]]:
    """문서/발표자료용 조건 목록(번호, 함수명, 종류, 제목, 설명, 관련 상수)."""
    return [
        {"no": c.no, "func": c.func_name, "kind": c.kind, "title": c.title,
         "description": c.description, "defines": list(c.related_defines)}
        for c in SELL_CONDITIONS
    ]
