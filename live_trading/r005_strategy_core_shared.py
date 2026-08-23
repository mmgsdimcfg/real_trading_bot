"""R76 shared strategy core.

Overview:
- Single source of truth for live/sim buy-sell decision logic.
- Keep this file as the only place for core rule changes.

Used by:
- r76_trade_live_execute.py
- r76_trade_simulate_by_date.py

Quick usage:
- from r76_strategy_core_shared import R76StrategyConfig, check_buy_condition, check_sell_condition, update_live_price_cross_state

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-08-23] type=fix owner=copilot
    summary: run_buy_condition_pipeline_comment(현재 활성 3분봉 다중필터 경로, r006에서
      ENABLE_1MIN_GOLDEN_CROSS_BUY=False라 실제로 매매를 결정하는 그 함수)의 최소 봉 수
      요건이 len(frame)<2(사실상 3분봉 기준 장 시작 후 약 6분부터 판정 시작)로만 걸려있어,
      calculate_indicators()가 BB_PERIOD(20) 미만 구간에서 rolling(min_periods=1)으로
      만든 불안정한 BB_MIDDLE/BB_STD(사실상 2~3봉짜리 초단기 평균)를 그대로 판정에 썼음 -
      사용자가 HTS 차트에서 볼린저밴드가 3분봉은 10시부터, 1분봉은 09:20부터만 그려지는
      것을 보고 발견. 완전히 동일한 원인이 check_buy_condition_1min(1분봉 골든크로스
      대체경로, 현재 비활성)에서는 2026-08-17에 이미 len(frame)<BB_PERIOD 가드로 수정된
      바 있었는데, 실제 매매를 결정하는 이 3분봉 경로에는 그 수정이 적용되지 않고 있었음.
      동일한 가드로 통일. OPENING_GUARD_MINUTES(15분, 점수 상향 방식)는 그대로 두되, 이제
      BB_PERIOD 미만 구간(3분봉 기준 최초 약 60분)에는 그보다 먼저 이 가드가 걸려 판정
      자체가 발생하지 않는다.
    impact: common
    compatibility: breaking (장 시작 후 약 60분간 3분봉 신규 매수가 전면 차단됨 - 그 구간에서
      나오던 신호는 통계적으로 신뢰하기 어려운 "가짜 돌파"였을 가능성이 높음)
- [2026-08-23] type=feat owner=copilot
    summary: _buy_support_score()에 장기 추세 정합성 가점 신규 추가 - EMA20 > EMA60이면
      +EMA_TREND_ALIGN_SCORE(기본 2, r003)점. 사용자가 제안한 3분봉 Signal V1 스코어
      구성안(EMA20>EMA60/BB Middle상승/RSI/ADX/VolumeRatio/StochK>D) 중 기존 코드에
      전혀 없던 유일한 신호였음 - 나머지(BB돌파/RSI/ADX/VolumeRatio/Stochastic)는 이미
      필수조건 또는 가점으로 반영돼 있었고, 제안안의 V2 항목(윌리엄스%R/BB폭/추격매수
      방지/개장시간필터/거래대금/시장상대강도)도 대부분 이미 구현돼 있었음(r002 RS15,
      turnover 필터 등). calculate_indicators()에 EMA_60 컬럼 추가(EMA_20은 1분봉
      게이트용으로 기존에 이미 계산되던 컬럼을 재사용, 3분봉 프레임에도 동일하게 존재).
    impact: common
    compatibility: backward-compatible (기존 가점 항목에 최대 +2점 추가되는 것뿐, 새 필수조건
      아님 - 매수 문턱을 낮추는 방향이라 빈도가 소폭 늘어날 수 있음)
- [2026-08-23] type=feat owner=copilot
    summary: check_entry_condition_1min() 신규 추가 - 3분봉 다중필터(run_buy_condition_pipeline_
      comment) 통과 이후 1분봉에서 진입 타이밍을 한 번 더 점수제로 검증하는 2단계 게이트.
      필수조건 없이 EMA9>EMA20(+2)/Close>EMA9(+1)/직전고점돌파(+2)/거래량>VOL_MA20(+2) 4개를
      점수화해 ENTRY_SCORE_THRESHOLD(기본 5, r003) 이상이면 통과. calculate_indicators()에
      EMA_9/EMA_20 컬럼 신규 추가(EMA 계열 지표가 기존에 전혀 없었음, MA_5 SMA만 존재).
      r006/r007의 실제 배선은 ENABLE_1MIN_ENTRY_SCORE_GATE(r003) 플래그로 제어되며, 기존
      ENABLE_1MIN_GOLDEN_CROSS_BUY(단독 대체 경로, 섹션12)와는 완전히 별개.
    impact: common
    compatibility: breaking (신규 2단계 게이트로 매수 빈도가 줄어들 수 있음, False로 즉시 롤백 가능)
- [2026-08-21] type=fix owner=copilot
    summary: 필수조건 2-b(직전 매집봉 확인)의 거래량 증가 판정을 "직전봉 대비"에서 "VOL_MA20(20봉
      평균) 대비"로 변경, 룩백도 3->5봉으로 확장(PRE_CROSS_ACCUM_LOOKBACK_BARS, r003). 어제
      추가된 이 조건이 257720 실리콘투 2026-08-21 09:57 BB중앙선 상향 돌파를 막은 사례를 사용자가
      발견 - 돌파 직전 3봉 중 09:48봉이 직전 09:45봉(장중 거래량 스파이크)보다 거래량이 낮게
      나와 "거래량 증가" 조건에서 탈락했는데, VOL_MA20 대비로는 낮지 않았음. 봉 대 봉 비교는
      매집 국면에서도 노이즈로 인해 오탈락하기 쉬워, 더 안정적인 평균 대비 기준으로 교체.
    impact: common
    compatibility: breaking (필수조건 2-b 통과 기준 완화로 매수 빈도가 다시 늘어날 수 있음)
- [2026-08-20] type=feat owner=copilot
    summary: run_buy_condition_pipeline_comment에 필수조건 2-b(직전 매집봉 확인) 신규 추가 -
      BB 중앙선 상향 돌파 직전 PRE_CROSS_ACCUM_LOOKBACK_BARS(3, r003)개 봉 중 하나라도
      (1) 그 봉의 저가~고가가 그 봉 시점 BB중간값을 포함, (2) 양봉, (3) 거래량이 바로 이전
      봉보다 큼을 모두 만족해야 통과. 직전봉 1개만 보면 매수 빈도가 크게 줄어들어 최근 3봉
      룩백으로 완화. ENABLE_PRE_CROSS_ACCUM_BAR_CHECK(r003) False 시 건너뜀(즉시 롤백 가능).
    impact: common
    compatibility: breaking (신규 필수조건 추가로 매수 빈도가 줄어들 수 있음)
- [2026-08-17] type=fix owner=copilot
    summary: 사용자가 캡처한 실제 차트(2026-08-14 현대차/삼성전기 3분봉)에서 육안상 매수가
      나와야 할 지점이 실제로는 걸러지는 것을 발견해 원인 2건 수정. (1) 필수조건 5(스토캐스틱)의
      상한을 STOCH_BUY_MAX(50)에서 config.stoch_overbought(96, 진짜 과열 기준)로 완화 -
      %K는 10봉 오실레이터라 BB중앙선 돌파 순간엔 이미 50을 넘는 게 정상적인 움직임인데,
      50을 상한으로 걸어놓아 정상적인 돌파 대부분을 걸러내고 있었음(현대차 12:12 사례로 확인).
      (2) 필수조건 1(BB 기울기)의 하드코딩 임계값 -0.7%를 신규 상수 BB_SLOPE_MIN_PCT(-2.0%,
      r003)로 분리 및 완화 - 60분 lookback 기울기가 급등 후 되돌림 국면에서 가격/스토캐스틱/
      윌리엄스%R이 이미 반전된 뒤에도 한참 뒤늦게 양전환되어 진입을 막고 있었음(삼성전기 10:42
      사례로 확인). 두 수정 모두 2026-08-14 실데이터 재생으로 검증 완료.
    impact: common
    compatibility: breaking (매수 필수조건 통과 기준이 완화되어 매수 빈도/타이밍이 달라짐)
- [2026-08-17] type=feat owner=copilot
    summary: run_buy_condition_pipeline_comment(3분봉 다중필터 매수)에 필수조건 5/6으로
      스토캐스틱 패스트/윌리엄스 %R 매수신호를 신규 추가. 사용자가 원래 의도한 설계는
      "3분봉 BB 중앙선 골든크로스 시점에 스토캐스틱 패스트와 윌리엄스 %R의 매수신호가
      함께 나올 때 매수"였는데, 실제 코드는 2026-06-28 리팩토링(불필요 로직 제거 명목)
      이후 스토캐스틱은 매도 판단에만, 윌리엄스 %R은 어디에도 쓰이지 않는 상태였음
      (STOCH_BUY_MIN/MAX, WILLIAMS_BUY_FLOOR/OVERBOUGHT_CEIL 등 관련 config 필드도
      R76StrategyConfig에 정의만 되고 아무 곳에서도 읽히지 않는 죽은 필드였음). 필수조건 5는
      스토캐스틱 %K가 %D를 상향 돌파하거나(골든크로스) %K>%D이면서 STOCH_BUY_MIN~MAX
      구간(과열 아님)일 때 통과, 필수조건 6은 윌리엄스 %R이 직전봉 대비 상승 중이면서
      WILLIAMS_BUY_FLOOR~WILLIAMS_OVERBOUGHT_CEIL 구간(바닥권 탈출~과열 직전)일 때
      통과. r003의 ENABLE_1MIN_GOLDEN_CROSS_BUY도 False로 되돌려 이 3분봉 경로가 다시
      기본 매수 경로가 되도록 함(r003 Update log 참조).
    impact: common
    compatibility: breaking (매수 필수조건이 2개 늘어나 매수 빈도가 줄어들 수 있음)
- [2026-08-17] type=refactor owner=copilot
    summary: 죽은 코드 제거 - buy_1st_live_price_above_bb_mid_within_gap_comment ~
      buy_10th_prev_close_and_volume_soft_guard_comment (10개 함수, "가짜" 10단계 매수 파이프라인)가
      r006/r007 어디에서도 호출되지 않는 것으로 확인됨(run_buy_condition_pipeline_comment는 자체
      로직을 재구현하고 있어 이 함수들과 무관). 실제 사용되지 않는 죽은 코드라 전체 삭제.
    impact: common
    compatibility: backward-compatible (미사용 함수 삭제만, 동작 변화 없음)
- [2026-07-18] type=fix owner=copilot
    summary: BB_MID_DOWNTREND 조건 완화 - 가격이 이미 BB_MID 위에 있을 경우 BB_MID 하락추세 차단 해제. 후행 지표 아티팩트로 인한 오진입 차단 방지.
    impact: common
    compatibility: backward-compatible (BB_MID 하락 시 가격이 이미 돌파한 경우 추가 허용)
- [2026-07-02] type=fix owner=copilot
    summary: 거래량비율 점수에 0.7~1.2배 구간(+1점) 추가. 093370 사례처럼 vol_ratio 0.7~1.2 사이에서 0점 처리되어 유효 반전신호가 과도하게 반려되던 문제 완화.
    impact: common
    compatibility: backward-compatible (more entries pass score gate)
- [2026-06-28] type=refactor owner=copilot
    summary: 불필요 로직 제거(Williams/RSI_SIGNAL/DI이중/EMA미사용) + VWAP·거래량방향·BB폭확장·MA5방향 신호 추가; BB_BUY_SCORE_THRESHOLD 8->9
    impact: common
    compatibility: backward-compatible
- [2026-06-28] type=fix owner=copilot
    summary: buy_9th hard_floor max(0.30,req*0.75)->max(0.65,req*0.90) 저거래량 진입 완전 차단
    impact: common
    compatibility: backward-compatible
- [2026-06-26] type=fix owner=copilot
    summary: buy_9th hard_floor 공식 강화 max(0.15,req*0.65)->max(0.30,req*0.75) 저거래량 종목 매수 차단
- [2026-06-25] type=feat owner=copilot
    summary: (1) BB_MID_CHASE_MAX_GAP_PCT 1.0%→0.7% 강화; (2) BB_UPPER_GAP_MIN_PCT 0.25%→0.5% 상향; (3) 개장 직후(09:00~09:08, 첫 3봉) CLOSE/UPTREND_CONT 진입 시 score threshold를 10으로 상향(OPENING_GUARD).
    impact: common
    compatibility: backward-compatible
- [2026-06-17] type=feat owner=copilot
    summary: BB 중간선 최근 4봉(12분) 연속 우하향 시 매수 차단 조건 추가 (BB_MID_DOWNTREND_4BARS); _bb_middle_is_downtrend 함수 신규.
    impact: common
    compatibility: backward-compatible
- [2026-06-17] type=fix owner=copilot
    summary: (1) 크로스 룩백을 3봉→5봉으로 확장하여 최근 15분 내 크로스 인정, (2) UPTREND_CONT 진입경로 추가: live>BB중간선+종가상승+ADX30이상++DI>-DI+BB위3봉이상이면 크로스 없이도 매수 허용.
    impact: common
    compatibility: backward-compatible
- [2026-06-05] type=fix owner=copilot
    summary: (한글) BB 중간값 진입 보조조건을 최근 3봉 유지(2/3)에서 "현재가 > 직전 3분봉 종가 && 현재가 > BB 중간값"으로 변경.
    impact: common
    compatibility: backward-compatible
- [2026-05-10] type=refactor owner=copilot
    summary: created r76 shared core module and centralized core strategy logic.
    impact: common
    compatibility: backward-compatible
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from r003_define_config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD_MULTIPLIER,
    BB_SLOPE_LOOKBACK_BARS,
    BB_SLOPE_MIN_PCT,
    BB_MID_DOWNTREND_BARS,
    BB_MID_CHASE_MAX_GAP_PCT,
    BB_UPPER_GAP_MIN_PCT,
    CANDLE_GAIN_MAX_PCT,
    CANDLE_GAIN_MIN_PCT,
    EARLY_NEAR_CROSS_MIN_TURNOVER_KRW,
    EARLY_NEAR_CROSS_MIN_VOL_MA,
    EARLY_NEAR_CROSS_MIN_VOLUME,
    MIN_ENTRY_VOL_MA,
    MIN_ENTRY_VOLUME,
    MACD_FAST,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW,
    MA_PERIOD,
    MFI_PERIOD,
    NEAR_CROSS_ARM_GAP_MAX,
    NEAR_CROSS_ARM_MA_RISE_MIN,
    NEAR_CROSS_EARLY_GAP_MAX,
    NEAR_CROSS_EARLY_MA_RISE_MIN,
    OBV_MA_PERIOD,
    RSI_PERIOD,
    RSI_SIGNAL_PERIOD,
    STOCH_BUY_MIN,
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    VOLUME_MA_PERIOD,
    WILLIAMS_BUY_FLOOR,
    WILLIAMS_D_PERIOD,
    WILLIAMS_OVERBOUGHT_CEIL,
    WILLIAMS_R_PERIOD,
    OPENING_GUARD_MINUTES,
    OPENING_GUARD_SCORE_THRESHOLD,
    ENABLE_PRE_CROSS_ACCUM_BAR_CHECK,
    PRE_CROSS_ACCUM_LOOKBACK_BARS,
    EMA_9_PERIOD,
    EMA_20_PERIOD,
    ENTRY_PREV_HIGH_LOOKBACK_BARS,
    ENTRY_EMA_CROSS_SCORE,
    ENTRY_CLOSE_ABOVE_EMA9_SCORE,
    ENTRY_PREV_HIGH_BREAKOUT_SCORE,
    ENTRY_VOLUME_ABOVE_MA_SCORE,
    ENTRY_SCORE_THRESHOLD,
    EMA_60_PERIOD,
    EMA_TREND_ALIGN_SCORE,
)


@dataclass(frozen=True)
class R76StrategyConfig:
    live_price_bb_buffer_pct: float
    live_price_cross_confirm_polls: int
    live_price_cross_confirm_seconds: float
    live_price_down_cross_confirm_polls: int
    live_price_down_cross_confirm_seconds: float

    require_strict_buy_golden_cross: bool
    stoch_overbought: float
    williams_overbought_ceil: float
    bb_upper_proximity_max: float
    bb_squeeze_min_width_pct: float
    adx_min_trend: float

    stop_loss_percent: float
    take_profit_percent: float
    enable_box_range_hold_tech_sell: bool
    box_range_hold_lookback_bars: int
    box_range_hold_max_range_pct: float
    box_range_hold_max_bb_width_pct: float

    ma5_bb_down_cross_min_pnl: float
    ma5_bb_down_cross_immediate_pnl: float
    ma5_bb_down_cross_immediate_score: int
    aux_sell_min_pnl_score2: float
    aux_sell_min_pnl_score3: float
    aux_sell_min_pnl_score4: float

    stoch_buy_min: float
    stoch_buy_max: float
    rsi_buy_min: float
    rsi_buy_max: float
    williams_buy_floor: float
    obv_breakout_lookback_bars: int
    enable_price_lead_bb_breakout: bool
    price_lead_breakout_min_score: int
    price_lead_breakout_min_adx: float
    price_lead_breakout_allow_overbought: bool
    enable_strong_trend_overbought_bypass: bool = False
    strong_trend_overbought_min_score: int = 5
    strong_trend_overbought_min_vol_ratio: float = 1.5
    strong_trend_overbought_min_adx: float = 30.0
    ma5_bb_follow_chase_max_gap_pct: float = 0.01
    bb_buy_score_threshold: int = 8


def _num(candle: pd.Series, key: str) -> float:
    value = candle.get(key)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def update_timed_condition_state(
    state_by_code: dict[str, dict[str, object]],
    code: str,
    position_token: object,
    ts: object,
    condition: bool,
) -> float:
    state = state_by_code.get(code)
    if not condition:
        if state is not None and state.get("position_token") == position_token:
            state_by_code.pop(code, None)
        return 0.0

    if state is None or state.get("position_token") != position_token:
        state_by_code[code] = {"position_token": position_token, "start": ts}
        return 0.0

    start = state.get("start")
    if start is None:
        state["start"] = ts
        return 0.0

    try:
        return max((ts - start).total_seconds(), 0.0)
    except Exception:
        state["start"] = ts
        return 0.0


def _compute_bb_slope_pct(frame: pd.DataFrame, lookback: int = BB_SLOPE_LOOKBACK_BARS) -> float:
    if "BB_MIDDLE" not in frame.columns or len(frame) < 2:
        return float("nan")
    n = min(lookback, len(frame) - 1)
    bb_now = _num(frame.iloc[-1], "BB_MIDDLE")
    bb_ago = _num(frame.iloc[-(n + 1)], "BB_MIDDLE")
    if pd.isna(bb_now) or pd.isna(bb_ago) or bb_ago <= 0:
        return float("nan")
    return (bb_now - bb_ago) / bb_ago * 100.0



def _bb_middle_is_downtrend(frame: pd.DataFrame, n: int = BB_MID_DOWNTREND_BARS) -> bool:
    """최근 n봉(=n*3분) 동안 BB 중간선이 연속으로 하락 중이면 True."""
    if "BB_MIDDLE" not in frame.columns or len(frame) < n + 1:
        return False
    vals = [_num(frame.iloc[-(i + 1)], "BB_MIDDLE") for i in range(n)]
    if any(pd.isna(v) for v in vals):
        return False
    return all(vals[i] < vals[i + 1] for i in range(n - 1))

def _buy_support_score(
    cur: pd.Series,
    prev: pd.Series,
    frame: pd.DataFrame,
    config: R76StrategyConfig,
) -> int:
    score = 0

    # RSI 구간 점수 (최대 2점)
    rsi_c = _num(cur, "RSI")
    if not pd.isna(rsi_c):
        if 50.0 <= rsi_c <= 65.0:
            score += 2
        elif 45.0 <= rsi_c < 50.0 or 65.0 < rsi_c <= 70.0:
            score += 1

    # 장기 추세 정합성 (최대 EMA_TREND_ALIGN_SCORE점): EMA20 > EMA60 → 상위 추세 우상향
    ema20_c = _num(cur, "EMA_20")
    ema60_c = _num(cur, "EMA_60")
    if not any(pd.isna(v) for v in (ema20_c, ema60_c)) and ema20_c > ema60_c:
        score += EMA_TREND_ALIGN_SCORE

    # 거래량 비율 점수 (최대 3점)
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        vol_ratio = vol / vol_ma
        if vol_ratio >= 2.0:
            score += 3
        elif vol_ratio >= 1.5:
            score += 2
        elif vol_ratio >= 1.2:
            score += 1
        elif vol_ratio >= 0.7:
            score += 1

    # ADX 추세 강도 점수 (최대 3점)
    adx_c = _num(cur, "ADX")
    if not pd.isna(adx_c):
        if adx_c >= 35.0:
            score += 3
        elif adx_c >= 30.0:
            score += 2
        elif adx_c >= 25.0:
            score += 1

    # VWAP 대비 현재가 위치 (최대 2점): 현재가 > VWAP → 기관 평균 매수가 상회 (DI이중반영 제거 후 대체)
    vwap_v = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not any(pd.isna(v) for v in (vwap_v, close_v)) and vwap_v > 0:
        if close_v > vwap_v * 1.002:
            score += 2
        elif close_v > vwap_v:
            score += 1

    # 거래량 증가 방향 (최대 1점): 현봉 > 전봉 → 관심 유입 중
    vol_prev = _num(prev, "volume")
    if not any(pd.isna(v) for v in (vol, vol_prev)) and vol_prev > 0:
        if vol > vol_prev:
            score += 1

    # BB 폭 확장 (최대 1점): 스퀴즈 해소 → 추세 발생 초기 신호
    bb_upper_c = _num(cur, "BB_UPPER")
    bb_lower_c = _num(cur, "BB_LOWER")
    bb_upper_p = _num(prev, "BB_UPPER")
    bb_lower_p = _num(prev, "BB_LOWER")
    if not any(pd.isna(v) for v in (bb_upper_c, bb_lower_c, bb_upper_p, bb_lower_p)):
        if (bb_upper_c - bb_lower_c) > (bb_upper_p - bb_lower_p):
            score += 1

    # MA5 단기 상승 (최대 1점): MA5[t] > MA5[t-1] → 단기 추세 유지
    ma5_c = _num(cur, "MA_5")
    ma5_p = _num(prev, "MA_5")
    if not any(pd.isna(v) for v in (ma5_c, ma5_p)) and ma5_p > 0:
        if ma5_c > ma5_p:
            score += 1

    # MACD 골든크로스 점수 (최대 2점)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c)) and macd_c > msig_c:
        score += 2

    # BB 중앙선 기울기 강도 점수 (최대 3점)
    bb_slope_pct = _compute_bb_slope_pct(frame)
    if not pd.isna(bb_slope_pct):
        if bb_slope_pct >= 1.5:
            score += 3
        elif bb_slope_pct >= 1.0:
            score += 2
        elif bb_slope_pct >= 0.5:
            score += 1

    return score


def _sell_support_score(cur: pd.Series, prev: pd.Series, config: R76StrategyConfig) -> int:
    score = 0

    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if k_c < d_c:
            score += 1

    # RSI 하락 방향 (빠른 모멘텀 약화 감지): RSI[t] < RSI[t-1] — RSI_SIGNAL 크로스보다 빠름
    rsi_c = _num(cur, "RSI")
    rsi_p = _num(prev, "RSI")
    if not any(pd.isna(v) for v in (rsi_c, rsi_p)):
        if rsi_c < rsi_p:
            score += 1

    # VWAP 이탈 (기관 매도 압력): 현재가 < VWAP — Williams %R/%D 제거 후 대체
    vwap_v = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not any(pd.isna(v) for v in (vwap_v, close_v)) and vwap_v > 0:
        if close_v < vwap_v:
            score += 1

    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if macd_p >= msig_p and macd_c < msig_c:
            score += 1

    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_c < obv_ma_c and obv_c < obv_p:
            score += 1

    return score


def _is_box_range_hold_zone(frame: pd.DataFrame, config: R76StrategyConfig) -> tuple[bool, str]:
    if len(frame) < config.box_range_hold_lookback_bars:
        return False, "INSUFFICIENT_BOX_BARS"

    recent = frame.tail(config.box_range_hold_lookback_bars)
    high_v = pd.to_numeric(recent["high"], errors="coerce").max()
    low_v = pd.to_numeric(recent["low"], errors="coerce").min()
    close_v = _num(recent.iloc[-1], "close")
    bb_up = _num(recent.iloc[-1], "BB_UPPER")
    bb_low = _num(recent.iloc[-1], "BB_LOWER")

    if any(pd.isna(v) for v in (high_v, low_v, close_v, bb_up, bb_low)) or close_v <= 0:
        return False, "BOX_DATA_NAN"

    range_pct = (float(high_v) - float(low_v)) / float(close_v)
    bb_width_pct = (float(bb_up) - float(bb_low)) / float(close_v)
    is_box = range_pct <= config.box_range_hold_max_range_pct and bb_width_pct <= config.box_range_hold_max_bb_width_pct

    return is_box, f"RANGE_{range_pct*100:.2f}%_BBW_{bb_width_pct*100:.2f}%"


def update_live_price_cross_state(
    cross_state: dict[str, dict],
    code: str,
    now: pd.Timestamp,
    live_price: float,
    bb_middle: float,
    config: R76StrategyConfig,
) -> dict[str, object]:
    relation = "on"
    upper_trigger = bb_middle * (1.0 + config.live_price_bb_buffer_pct)
    lower_trigger = bb_middle * (1.0 - config.live_price_bb_buffer_pct)

    if live_price >= upper_trigger:
        relation = "above"
    elif live_price <= lower_trigger:
        relation = "below"

    tracker = cross_state.get(code)
    if tracker is None:
        tracker = {
            "confirmed_relation": relation if relation in {"above", "below"} else None,
            "pending": None,
        }
        cross_state[code] = tracker
        return {
            "relation": relation,
            "confirmed_relation": tracker["confirmed_relation"],
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    confirmed_relation = tracker.get("confirmed_relation")
    pending = tracker.get("pending")

    if relation not in {"above", "below"}:
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    if relation == confirmed_relation:
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    if pending is None or pending.get("side") != relation:
        tracker["pending"] = {"side": relation, "started_at": now, "count": 1}
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": relation,
            "pending_count": 1,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    pending["count"] = int(pending.get("count", 0)) + 1
    pending_seconds = max(0.0, (now - pending["started_at"]).total_seconds())

    if relation == "below":
        req_polls = config.live_price_down_cross_confirm_polls
        req_seconds = config.live_price_down_cross_confirm_seconds
    else:
        req_polls = config.live_price_cross_confirm_polls
        req_seconds = config.live_price_cross_confirm_seconds

    if pending["count"] >= req_polls and pending_seconds >= req_seconds:
        tracker["confirmed_relation"] = relation
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": pending_seconds,
            "signal": "cross_up" if relation == "above" else "cross_down",
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    return {
        "relation": relation,
        "confirmed_relation": confirmed_relation,
        "pending_side": relation,
        "pending_count": int(pending["count"]),
        "pending_seconds": pending_seconds,
        "signal": None,
        "upper_trigger": upper_trigger,
        "lower_trigger": lower_trigger,
    }


def check_buy_condition(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
    volume_ratio_threshold_fn: Callable[[pd.Timestamp, float], float],
) -> tuple[bool, str]:
    return run_buy_condition_pipeline_comment(
        frame=frame,
        now=now,
        live_price=live_price,
        cross_info=cross_info,
        config=config,
        volume_ratio_threshold_fn=volume_ratio_threshold_fn,
    )


def run_buy_condition_pipeline_comment(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
    volume_ratio_threshold_fn,
) -> tuple[bool, str]:
    """BB 중앙선 상승 돌파 전략: 6개 필수조건(BB돌파+스토캐스틱+윌리엄스%R 포함) + 가점 임계값 이상.

    BB_MIDDLE/BB_STD는 calculate_indicators()에서 rolling(window=BB_PERIOD,
    min_periods=1)로 계산되어, 실제 봉 수가 BB_PERIOD(20) 미만인 장 시작 직후에는
    "BB 중간값"이 사실상 그 짧은 구간의 평균처럼 움직인다 - 가격이 잠깐만 올라도
    이 불안정한 평균을 위로 뚫는 "가짜 돌파"가 쉽게 발생한다(check_buy_condition_1min과
    동일한 근본 원인, r005/r006 Update log 2026-08-17 참조 - 그쪽은 이미 이 최소 봉 수
    가드가 걸려 있었으나 이 3분봉 경로에는 누락돼 있었음). BB_MIDDLE이 실제로 20봉
    평균이 되는 시점(3분봉 기준 장 시작 후 약 60분) 이전에는 판정 자체를 하지 않는다.
    """
    if len(frame) < BB_PERIOD:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")
    cur_bb_upper = _num(cur, "BB_UPPER")
    prev_bb = _num(prev, "BB_MIDDLE")

    if any(pd.isna(v) for v in (cur_bb, cur_bb_upper, prev_bb)):
        return False, "MISSING_INDICATOR"

    # 필수조건 1: BB 중앙선 상승 추세 (최근 lookback봉 대비 기울기 > BB_SLOPE_MIN_PCT)
    bb_slope_pct = _compute_bb_slope_pct(frame)
    if pd.isna(bb_slope_pct) or bb_slope_pct <= BB_SLOPE_MIN_PCT:
        slope_str = f"{bb_slope_pct:.3f}" if not pd.isna(bb_slope_pct) else "nan"
        return False, f"BB_SLOPE_NOT_RISING_{slope_str}%"

    # 필수조건 1-b: BB 중간선 최근 12분(4봉) 연속 우하향 시 매수 차단
    # 단, 가격이 이미 BB_MID를 상향 돌파한 경우(후행 지표 아티팩트) 차단 해제
    if _bb_middle_is_downtrend(frame):
        _price_above_bb_pct = (live_price - cur_bb) / cur_bb * 100.0 if cur_bb > 0 else 0.0
        if _price_above_bb_pct <= 0.0:
            bb_vals = [_num(frame.iloc[-(i+1)], "BB_MIDDLE") for i in range(4)] if len(frame) >= 4 else []
            vals_str = " > ".join(f"{v:.1f}" for v in reversed(bb_vals))
            return False, f"BB_MID_DOWNTREND_4BARS_{vals_str}"

    # 필수조건 2: BB 중앙선 상향 돌파 (live cross signal OR close 기준 크로스)
    live_cross_up = cross_info.get("signal") == "cross_up"
    prev_close = _num(prev, "close")
    cur_close = _num(cur, "close")
    close_cross = (
        not any(pd.isna(v) for v in (prev_close, cur_close, prev_bb, cur_bb))
        and prev_close <= prev_bb
        and cur_close > cur_bb
    )
    # N봉 이전 크로스 체크 (5봉으로 확장): 전환봉 이후 현재까지 BB 위 연속 유지 시 크로스 인정
    if not close_cross:
        for _lb in range(3, min(6, len(frame)) + 1):
            _bar_n_close = _num(frame.iloc[-_lb], "close")
            _bar_n_bb = _num(frame.iloc[-_lb], "BB_MIDDLE")
            if any(pd.isna(v) for v in (_bar_n_close, _bar_n_bb)) or _bar_n_close > _bar_n_bb:
                continue  # 이 봉도 BB 위이면 더 이전 탐색 or NaN
            # 전환봉 이후 현재까지 모든 봉이 연속으로 BB 위인지 확인
            _all_above = all(
                not any(pd.isna(v) for v in (_num(frame.iloc[-_k], "close"), _num(frame.iloc[-_k], "BB_MIDDLE")))
                and _num(frame.iloc[-_k], "close") > _num(frame.iloc[-_k], "BB_MIDDLE")
                for _k in range(1, _lb)
            )
            if _all_above:
                close_cross = True
                break

    # 우상향 추세 지속 진입: 크로스 이벤트 없이도 강한 추세 + BB 위 지속이면 매수 허용
    # 크로스가 5봉 이전에 발생했지만 추세가 계속 이어지는 구간 포착
    _uptrend_continuation = False
    if not live_cross_up and not close_cross:
        _adx = _num(cur, "ADX")
        _di_plus = _num(cur, "DI_PLUS")
        _di_minus = _num(cur, "DI_MINUS")
        _ma5_cur = _num(cur, "MA_5")
        _ma5_prev = _num(prev, "MA_5")
        # 최근 최대 5봉 중 종가가 BB 위에 있는 봉 수 카운트
        _n_above = 0
        for _i in range(1, min(6, len(frame)) + 1):
            _bc = _num(frame.iloc[-_i], "close")
            _bbb = _num(frame.iloc[-_i], "BB_MIDDLE")
            if not any(pd.isna(v) for v in (_bc, _bbb)) and _bc > _bbb:
                _n_above += 1
        if (
            not any(pd.isna(v) for v in (cur_close, cur_bb, prev_close, _adx, _di_plus, _di_minus))
            and live_price > cur_bb        # 현재가 BB 중간선 위
            and cur_close > cur_bb         # 현재봉 종가 BB 위
            and cur_close > prev_close     # 현재봉 종가 > 전봉 종가 (상승 모멘텀)
            and bb_slope_pct > 0.0         # BB 중간선 상승 추세
            and _adx >= 30.0               # ADX 30 이상 (뚜렷한 추세)
            and _di_plus > _di_minus       # +DI > -DI (상승 방향성 우세)
            and _n_above >= 3              # 최근 5봉 중 3봉 이상 BB 위 유지
            and not any(pd.isna(v) for v in (_ma5_cur, _ma5_prev))
            and _ma5_cur > _ma5_prev       # MA5 단기 상승 중 (이미 꺾인 추세 추격 차단)
        ):
            _uptrend_continuation = True

    if not live_cross_up and not close_cross and not _uptrend_continuation:
        return False, "NO_BB_MID_CROSS_UP"

    # 필수조건 2-b: 직전 매집봉 확인 (BB중간선 포함 + 양봉 + 거래량 증가)
    # 돌파 직전에 매집(거래량 증가) 양봉이 있었는지 확인. 직전봉(-2) 1개만 보면 매수 빈도가
    # 크게 줄어들므로, close_cross 판정과 동일하게 최근 PRE_CROSS_ACCUM_LOOKBACK_BARS개 봉
    # 중 하나라도 만족하면 통과시켜 빈도 감소를 완화한다.
    # 거래량 증가 판정은 "직전봉 대비"가 아니라 "VOL_MA20(20봉 평균) 대비"로 본다 - 직전봉 대비는
    # 봉 하나하나의 잡음에 취약해서(연속 매집 구간에서도 바로 전 봉보다만 작으면 탈락), 실제로는
    # 평균 대비 거래량이 살아있는 매집봉을 놓치는 사례가 확인됨(257720 실리콘투 2026-08-21 09:57
    # 사례: 09:48봉 거래량이 직전 09:45봉보다는 줄었지만 VOL_MA20 대비로는 낮지 않았음).
    if ENABLE_PRE_CROSS_ACCUM_BAR_CHECK:
        _accum_ok = False
        _accum_bars_checked = 0
        for _ofs in range(2, 2 + PRE_CROSS_ACCUM_LOOKBACK_BARS):
            if _ofs + 1 > len(frame):
                break
            _accum_bars_checked += 1
            _cand = frame.iloc[-_ofs]
            _cand_open = _num(_cand, "open")
            _cand_close = _num(_cand, "close")
            _cand_low = _num(_cand, "low")
            _cand_high = _num(_cand, "high")
            _cand_bb = _num(_cand, "BB_MIDDLE")
            _cand_vol = _num(_cand, "volume")
            _cand_vol_ma = _num(_cand, "VOL_MA20")
            if any(pd.isna(v) for v in (
                _cand_open, _cand_close, _cand_low, _cand_high, _cand_bb, _cand_vol, _cand_vol_ma,
            )):
                continue
            _contains_bb_mid = _cand_low <= _cand_bb <= _cand_high
            _is_bullish = _cand_close > _cand_open
            _vol_rising = _cand_vol > _cand_vol_ma
            if _contains_bb_mid and _is_bullish and _vol_rising:
                _accum_ok = True
                break
        if not _accum_ok:
            return False, f"NO_PRE_CROSS_ACCUM_BAR_LOOKBACK_{_accum_bars_checked}"

    # 필수조건 3: 현재 진행중인 3분봉 양봉 (+CANDLE_GAIN_MIN_PCT% 이상)
    cur_open = _num(cur, "open")
    if pd.isna(cur_open) or cur_open <= 0:
        return False, "CANDLE_OPEN_MISSING"
    candle_gain_pct = (live_price - cur_open) / cur_open * 100.0
    if candle_gain_pct < CANDLE_GAIN_MIN_PCT:
        return False, f"CANDLE_NOT_BULLISH_{candle_gain_pct:.2f}%_LT_{CANDLE_GAIN_MIN_PCT:.1f}%"
    # 추격 매수 방지 1: 현재봉 과도 상승 차단 (급등봉 추격)
    if candle_gain_pct > CANDLE_GAIN_MAX_PCT:
        return False, f"CHASE_BUY_INTRABAR_{candle_gain_pct:.2f}%_GT_{CANDLE_GAIN_MAX_PCT:.1f}%"
    # 추격 매수 방지 2: BB 중간선 대비 현재가 갭 과도 차단 (스파이크 추격)
    _bb_mid_gap_pct = (live_price - cur_bb) / cur_bb * 100.0 if cur_bb > 0 else 0.0
    if _bb_mid_gap_pct > BB_MID_CHASE_MAX_GAP_PCT:
        return False, f"CHASE_BUY_BB_GAP_{_bb_mid_gap_pct:.2f}%_GT_{BB_MID_CHASE_MAX_GAP_PCT:.1f}%"

    # 필수조건 4: BB 상단까지 충분한 공간 (>= BB_UPPER_GAP_MIN_PCT%)
    if live_price <= 0:
        return False, "LIVE_PRICE_INVALID"
    bb_upper_gap_pct = (cur_bb_upper - live_price) / live_price * 100.0
    if bb_upper_gap_pct < BB_UPPER_GAP_MIN_PCT:
        return False, f"BB_UPPER_GAP_TOO_SMALL_{bb_upper_gap_pct:.2f}%_LT_{BB_UPPER_GAP_MIN_PCT:.1f}%"

    # 필수조건 5: 스토캐스틱 패스트 매수신호 (%K가 %D를 상향 돌파, 또는 %K>%D이면서 진짜 과열(stoch_overbought) 아닌 구간)
    # 주의: %K는 10봉 오실레이터라 BB중앙선 돌파 순간엔 이미 50을 넘어서는 게 정상적인 움직임이라
    # STOCH_BUY_MAX(50)를 상한으로 쓰면 정상적인 돌파 대부분을 걸러내 버린다. 상한은 config.stoch_overbought
    # (진짜 과열 기준, 기본 96)로 완화한다.
    stoch_k = _num(cur, "STOCH_K")
    stoch_d = _num(cur, "STOCH_D")
    stoch_k_prev = _num(prev, "STOCH_K")
    stoch_d_prev = _num(prev, "STOCH_D")
    if any(pd.isna(v) for v in (stoch_k, stoch_d, stoch_k_prev, stoch_d_prev)):
        return False, "STOCH_DATA_MISSING"
    stoch_golden_cross = stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d
    stoch_buy_signal = stoch_golden_cross or (
        stoch_k > stoch_d and STOCH_BUY_MIN <= stoch_k < config.stoch_overbought
    )
    if not stoch_buy_signal:
        return False, f"NO_STOCH_BUY_SIGNAL_K_{stoch_k:.1f}_D_{stoch_d:.1f}"

    # 필수조건 6: 윌리엄스 %R 매수신호 (상승 중이면서 과열 상한 밑, 바닥권 탈출 하한 이상)
    williams_r = _num(cur, "WILLIAMS_R")
    williams_r_prev = _num(prev, "WILLIAMS_R")
    if any(pd.isna(v) for v in (williams_r, williams_r_prev)):
        return False, "WILLIAMS_DATA_MISSING"
    williams_buy_signal = (
        williams_r > williams_r_prev
        and WILLIAMS_BUY_FLOOR <= williams_r <= WILLIAMS_OVERBOUGHT_CEIL
    )
    if not williams_buy_signal:
        return False, f"NO_WILLIAMS_BUY_SIGNAL_R_{williams_r:.1f}"

    # 안전장치: 거래량 절대/상대 최소치 (저유동성 종목 차단)
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)):
        if vol_ma < MIN_ENTRY_VOL_MA:
            return False, f"LOW_VOL_MA_ABS_{vol_ma:.0f}_LT_{MIN_ENTRY_VOL_MA}"
        if vol < MIN_ENTRY_VOLUME:
            return False, f"LOW_ABS_VOLUME_{vol:.0f}_LT_{MIN_ENTRY_VOLUME}"
        if vol_ma > 0:
            vol_ratio = vol / vol_ma
            if vol_ratio < 0.10:
                return False, f"LOW_VOLUME_RATIO_{vol_ratio:.4f}_LT_0.10"

    # 가점 조건 합산 (RSI/거래량/ADX/DI/MACD/BB기울기, 최대 15점)
    score = _buy_support_score(cur, prev, frame, config)

    # 개장 직후 보호: 09:00~09:08 (첫 3봉) 동안 추격매수 차단 강화
    # live_cross_up(실시간 돌파)은 제외, close/uptrend 기반 진입만 제한
    if not live_cross_up and now.hour == 9 and now.minute < OPENING_GUARD_MINUTES:
        if score < OPENING_GUARD_SCORE_THRESHOLD:
            return False, f"OPENING_GUARD_{now.strftime('%H:%M')}_{score}_LT_{OPENING_GUARD_SCORE_THRESHOLD}"

    if score < config.bb_buy_score_threshold:
        return False, f"LOW_SCORE_{score}_LT_{config.bb_buy_score_threshold}"

    if live_cross_up:
        trigger = "LIVE_PRICE_BB_UP_CROSS"
    elif close_cross:
        trigger = "CLOSE_BB_UP_CROSS"
    else:
        trigger = "UPTREND_CONT"
    return True, f"{trigger}_SCORE_{score}"


def check_sell_condition(
    frame: pd.DataFrame,
    pnl_pct: float,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
) -> tuple[bool, str]:
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    if config.enable_box_range_hold_tech_sell and config.stop_loss_percent < pnl_pct < config.take_profit_percent:
        is_box, box_info = _is_box_range_hold_zone(frame, config)
        if is_box:
            return False, f"BOX_RANGE_HOLD_{box_info}"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")

    price_cross_down = (
        cross_info.get("signal") == "cross_down"
        and not pd.isna(cur_bb)
        and live_price < cur_bb
    )
    if price_cross_down:
        if pnl_pct < config.ma5_bb_down_cross_min_pnl:
            return False, (
                f"LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_PNL_{pnl_pct * 100:.2f}%"
                f"_LT_{config.ma5_bb_down_cross_min_pnl * 100:.2f}%"
            )

        score = _sell_support_score(cur, prev, config)
        if pnl_pct <= config.ma5_bb_down_cross_immediate_pnl or score >= config.ma5_bb_down_cross_immediate_score:
            if score >= 1:
                return True, f"LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_{score}"
            return True, "LIVE_PRICE_BB_DOWN_CROSS"
        return False, f"LIVE_PRICE_BB_DOWN_CROSS_WEAK_SCORE_{score}"

    score = _sell_support_score(cur, prev, config)
    if score >= 4:
        min_pnl_req: float | None = config.aux_sell_min_pnl_score4
    elif score == 3:
        min_pnl_req = config.aux_sell_min_pnl_score3
    elif score == 2:
        min_pnl_req = config.aux_sell_min_pnl_score2
    else:
        return False, "NO_SELL_SIGNAL"

    if pnl_pct >= min_pnl_req:
        return True, f"AUX_REVERSAL_SCORE_{score}"
    return False, f"AUX_BLOCKED_SCORE_{score}_PNL_{pnl_pct * 100:.2f}%_LT_{min_pnl_req * 100:.2f}%"


def _entry_score_1min(cur: pd.Series, frame_1min: pd.DataFrame) -> tuple[int, dict[str, bool]]:
    score = 0
    detail: dict[str, bool] = {}

    ema9 = _num(cur, "EMA_9")
    ema20 = _num(cur, "EMA_20")
    if not any(pd.isna(v) for v in (ema9, ema20)) and ema9 > ema20:
        score += ENTRY_EMA_CROSS_SCORE
        detail["ema_cross"] = True

    close_v = _num(cur, "close")
    if not any(pd.isna(v) for v in (close_v, ema9)) and close_v > ema9:
        score += ENTRY_CLOSE_ABOVE_EMA9_SCORE
        detail["close_above_ema9"] = True

    if len(frame_1min) > ENTRY_PREV_HIGH_LOOKBACK_BARS:
        prior = frame_1min.iloc[-(ENTRY_PREV_HIGH_LOOKBACK_BARS + 1):-1]
        prior_high = pd.to_numeric(prior["high"], errors="coerce").max()
        if not pd.isna(prior_high) and not pd.isna(close_v) and close_v > prior_high:
            score += ENTRY_PREV_HIGH_BREAKOUT_SCORE
            detail["prev_high_breakout"] = True

    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol > vol_ma:
        score += ENTRY_VOLUME_ABOVE_MA_SCORE
        detail["volume_above_ma"] = True

    return score, detail


def check_entry_condition_1min(frame_1min: pd.DataFrame) -> tuple[bool, str]:
    """3분봉 신호(check_buy_condition) 통과 후 1분봉에서 진입 타이밍을 재검증하는 점수제 게이트.

    필수조건 없이 EMA9>EMA20 / Close>EMA9 / 직전고점돌파 / 거래량>VOL_MA20 4개를 점수화해
    ENTRY_SCORE_THRESHOLD 이상이면 통과. r003.ENABLE_1MIN_ENTRY_SCORE_GATE로 켜고 끈다.
    """
    if frame_1min is None or len(frame_1min) < ENTRY_PREV_HIGH_LOOKBACK_BARS + 1:
        return False, "1MIN_ENTRY_INSUFFICIENT_BARS"

    cur = frame_1min.iloc[-1]
    score, _detail = _entry_score_1min(cur, frame_1min)
    if score < ENTRY_SCORE_THRESHOLD:
        return False, f"1MIN_ENTRY_LOW_SCORE_{score}_LT_{ENTRY_SCORE_THRESHOLD}"
    return True, f"1MIN_ENTRY_SCORE_{score}"


# ---------------------------------------------------------------------------
# Shared indicator calculation (used by r006 and r007)
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
    out["VOL_MA20"] = out["volume"].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()

    out["EMA_9"] = out["close"].ewm(span=EMA_9_PERIOD, adjust=False).mean()
    out["EMA_20"] = out["close"].ewm(span=EMA_20_PERIOD, adjust=False).mean()
    out["EMA_60"] = out["close"].ewm(span=EMA_60_PERIOD, adjust=False).mean()

    out["BB_MIDDLE"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).mean()
    out["BB_STD"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).std()
    out["BB_UPPER"] = out["BB_MIDDLE"] + out["BB_STD"] * BB_STD_MULTIPLIER
    out["BB_LOWER"] = out["BB_MIDDLE"] - out["BB_STD"] * BB_STD_MULTIPLIER

    delta = out["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out["RSI"] = 100 - (100 / (1 + rs))
    out.loc[avg_loss == 0, "RSI"] = 100.0
    out["RSI_SIGNAL"] = out["RSI"].rolling(window=RSI_SIGNAL_PERIOD, min_periods=1).mean()

    low_n = out["low"].rolling(window=STOCH_K_PERIOD, min_periods=1).min()
    high_n = out["high"].rolling(window=STOCH_K_PERIOD, min_periods=1).max()
    denom = (high_n - low_n).replace(0, float("nan"))
    out["STOCH_K"] = 100.0 * (out["close"] - low_n) / denom
    out["STOCH_D"] = out["STOCH_K"].rolling(window=STOCH_D_PERIOD, min_periods=1).mean()

    high_w = out["high"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).max()
    low_w = out["low"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).min()
    wr_denom = (high_w - low_w).replace(0, float("nan"))
    out["WILLIAMS_R"] = -100.0 * (high_w - out["close"]) / wr_denom
    out["WILLIAMS_D"] = out["WILLIAMS_R"].rolling(window=WILLIAMS_D_PERIOD, min_periods=1).mean()

    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    raw_money_flow = typical_price * out["volume"]
    price_delta = typical_price.diff()
    positive_flow = raw_money_flow.where(price_delta > 0, 0.0)
    negative_flow = raw_money_flow.where(price_delta < 0, 0.0)
    positive_sum = positive_flow.rolling(window=MFI_PERIOD, min_periods=1).sum()
    negative_sum = negative_flow.rolling(window=MFI_PERIOD, min_periods=1).sum()
    money_ratio = positive_sum / negative_sum.replace(0, float("nan"))
    out["MFI"] = 100.0 - (100.0 / (1.0 + money_ratio))
    out.loc[(positive_sum == 0) & (negative_sum == 0), "MFI"] = 50.0
    out.loc[(negative_sum == 0) & (positive_sum > 0), "MFI"] = 100.0
    out.loc[(positive_sum == 0) & (negative_sum > 0), "MFI"] = 0.0

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    high_diff = out["high"] - out["high"].shift(1)
    low_diff = out["low"].shift(1) - out["low"]
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    ema_tr = tr.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    ema_plus = plus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    ema_minus = minus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    out["DI_PLUS"] = 100.0 * ema_plus / ema_tr.replace(0, float("nan"))
    out["DI_MINUS"] = 100.0 * ema_minus / ema_tr.replace(0, float("nan"))
    di_sum = (out["DI_PLUS"] + out["DI_MINUS"]).replace(0, float("nan"))
    dx = 100.0 * (out["DI_PLUS"] - out["DI_MINUS"]).abs() / di_sum
    out["ADX"] = dx.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()

    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))

    # ATR - 변동성 기반 손익비 판단용 평균 진폭
    true_range = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = true_range.ewm(alpha=1.0 / ATR_PERIOD, min_periods=1, adjust=False).mean()

    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


def _near_cross_momentum_flags(cur: pd.Series, prev: pd.Series) -> dict[str, float | bool]:
    """Builds near-cross diagnostics used by both live and sim entry logic."""
    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")

    if any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)):
        return {"can_arm": False, "can_early": False, "gap_ratio": float("nan"), "ma_rise_ratio": float("nan")}

    gap = cur_bb - cur_ma5
    gap_ratio = gap / max(abs(cur_bb), 1.0)
    ma_rise_ratio = (cur_ma5 - prev_ma5) / max(abs(prev_ma5), 1.0)

    below_or_equal = cur_ma5 <= cur_bb
    arm_shape_ok = (prev_ma5 <= prev_bb) and below_or_equal

    can_arm = arm_shape_ok and (gap_ratio >= 0) and (gap_ratio <= NEAR_CROSS_ARM_GAP_MAX) and (ma_rise_ratio >= NEAR_CROSS_ARM_MA_RISE_MIN)
    can_early = arm_shape_ok and (gap_ratio >= 0) and (gap_ratio <= NEAR_CROSS_EARLY_GAP_MAX) and (ma_rise_ratio >= NEAR_CROSS_EARLY_MA_RISE_MIN)

    return {
        "can_arm": can_arm,
        "can_early": can_early,
        "gap_ratio": float(gap_ratio),
        "ma_rise_ratio": float(ma_rise_ratio),
    }


def _passes_early_near_cross_liquidity(cur: pd.Series) -> tuple[bool, str]:
    """Liquidity guard for ARM/EARLY near-cross entry."""
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    close_v = _num(cur, "close")
    if any(pd.isna(v) for v in (vol, vol_ma, close_v)):
        return False, "LIQUIDITY_DATA_NAN"

    turnover = close_v * vol
    if vol < EARLY_NEAR_CROSS_MIN_VOLUME:
        return False, f"LOW_ABS_VOLUME_{vol:.0f}_LT_{EARLY_NEAR_CROSS_MIN_VOLUME}"
    if vol_ma < EARLY_NEAR_CROSS_MIN_VOL_MA:
        return False, f"LOW_VOL_MA_{vol_ma:.0f}_LT_{EARLY_NEAR_CROSS_MIN_VOL_MA}"
    if turnover < EARLY_NEAR_CROSS_MIN_TURNOVER_KRW:
        return False, f"LOW_TURNOVER_{turnover:,.0f}_LT_{EARLY_NEAR_CROSS_MIN_TURNOVER_KRW:,}"
    return True, "OK"


