# -*- coding: utf-8 -*-

# Update log
# - [2026-07-21] type=fix owner=copilot
#     summary: BB_BUY_SCORE_THRESHOLD 8->10 - logs/20260720 r006 실매매 로그 분석(16건) 결과 score=8
#       매수 6건 전부 손실(-18,975원, 당일 총손실의 87%), score=9 4건도 순손실(-2,295원). score>=10
#       매수만 순이익 포함(순 -550원). 2026-07-01에 9->8로 완화했던 조치가 이번 표본에서는 저품질
#       진입을 다수 통과시켜 역효과. 표본 1일뿐이라 추가 로그로 지속 검증 필요.
#     impact: live
#     compatibility: breaking (매수 빈도 감소 예상)
# - [2026-07-20] type=fix owner=copilot
#     summary: LIVE_PRICE_POLL_INTERVAL_SECONDS 20->10 원복 - 직전 커밋에서 사용자가 실제로 알고
#       있던 기존 값(10초)을 확인 없이 20초로 임의 변경했던 것을 사용자 확인 후 되돌림. r006의
#       틱 정렬 로직(_next_aligned_tick/_sleep_until_next_tick)은 이 상수를 그대로 읽어 매분
#       00/10/20/30/40/50초(6틱/분)에 정렬되도록 자동 적용되므로 이 값 변경 외 다른 수정은 불필요.
#     impact: live
#     compatibility: backward-compatible
# - [2026-07-20] type=fix owner=copilot
#     summary: LIVE_PRICE_POLL_INTERVAL_SECONDS 10->20 - r006 메인 루프를 매분 00/20/40초
#       벽시계 정렬 틱(3틱/분)으로 고정 스케줄링하도록 바꾸면서, 20초 배수가 되도록 조정.
#       (r006 Update log 참조)
#     impact: live
#     compatibility: backward-compatible
# - [2026-07-18] type=fix owner=copilot
#     summary: BB_UPPER_GAP_MIN_PCT 0.5% → 0.25% 하향 조정 - 저변동성 종목 오후 구간에서 BB 폭 좁아져 매수 차단 완화.
#     impact: common
#     compatibility: backward-compatible
# - [2026-07-16] type=fix owner=copilot
#     summary: 매도/손절 조건 완화 - r007로 D일 picks를 D+1일 실데이터에 매매 시뮬레이션(look-ahead 없는 교차일 백테스트)한
#       결과, 급락 후 반등을 놓치는 조기청산 비중이 높게 나타나 값 조정.
#       (1) HARD_STOP_LOSS_PCT 1.2%->1.7%, HARD_STOP_MIN_HOLD_SECONDS 180->240 (58건 중 손절 후 34.5%가 +2%까지 회복)
#       (2) BREAKEVEN_FAIL_GIVEBACK_PCT 2.0%->2.8%, CONFIRM_SECONDS 120->150 (14건 중 92.9%가 본전 이상 회복)
#       (3) POST_BUY_BB_DROP_PCT 1.0%->1.4%, CONFIRM_SECONDS 60->90 (12건 중 58.3%가 +2%까지 회복, 평균 최대회복 +4.17%)
#       (4) TP_EXTENSION_TRAIL_FROM_PEAK 0.4%->0.6% (익절 후 트레일링이 과도하게 타이트하여 수익 조기 제한)
#       동일 근거로 r006/r007의 시그널 매도(STOCH K<D, MACD_HIST 2봉하락) pnl 임계값도 함께 완화함
#       (-0.8%->-1.2%, -0.5%->-0.8%, 해당 파일 Update log 참조).
#     impact: live
#     compatibility: breaking (매도/손절 발동이 이전보다 늦어짐 - 손실 트레이드당 손실폭은 커질 수 있으나 조기청산으로
#       인한 기회손실 감소가 목표)
# - [2026-07-12] type=refactor owner=copilot
#     summary: 매매(손절/익절/트레일링스탑/진입필터 등) 판단에 쓰이는 모든 설정값을
#       파일 상단 "LIVE TRADING TUNABLE PARAMETERS" 섹션 하나로 모아 재배치.
#       값은 전혀 변경하지 않음(재배치만 수행) - 지표 계산 주기/세션 시간/폴링
#       간격/시뮬레이션 전용 값은 하단 별도 섹션으로 분리.
#     impact: common
#     compatibility: backward-compatible (값 변경 없음, 위치만 이동)
# - [2026-07-01] type=fix owner=copilot summary=BB_BUY_SCORE_THRESHOLD 9->8 (실매매 중 유효 반전신호 과잉반려 완화, 093370 14:51 크로스 사례)
# - [2026-06-28] type=refactor owner=copilot summary=(3) BB_BUY_SCORE_THRESHOLD 8->9 (VWAP/거래량방향/BB폭/MA5 신호 추가 점수 체계 반영)
# - [2026-06-28] type=fix owner=copilot summary=(2) NO_TREND_EXIT peak 0.3%->0.6% + 연속HARD_STOP 서킷브레이커 + 당일HARD_STOP 종목 재진입 차단 상수 추가
# - [2026-06-28] type=fix owner=copilot summary=저거래량 추격매수 차단 강화 + 거래량 hard_floor 상향 + 개장 직후 보호 강화 + BB 갭 추격 기준 엄격화
# - [2026-06-26] type=fix owner=copilot summary=BREAKEVEN_FAIL 완화(giveback 2.0%, confirm 120s) + VOLUME_RATIO_MIDDAY 강화(0.50)

"""R76 shared configuration for live trading and simulation.

Risk profile presets:
- Set AUTO_TRADING_RISK_PROFILE to one of conservative|neutral|aggressive.
- Korean aliases are supported: 보수|중립|공격.
- If no profile is specified, neutral is applied by default.
- Preset JSON files live under xgraph/auto_trading/risk_profiles/.

튜닝 가이드:
- 실전 매매(r006_trade_live_execute.py)의 손절/익절/트레일링스탑/진입필터 등
  "매매 판단"에 직접 영향을 주는 값은 전부 아래 "LIVE TRADING TUNABLE
  PARAMETERS" 섹션에 모여 있다. 실전매매 결과를 보고 값을 튜닝할 때는 이
  섹션만 확인하면 된다.
- 지표 계산 주기(봉 개수 등), 세션/시간대, 폴링/백오프 같은 시스템 운영값,
  r007 시뮬레이션 전용 값은 그 아래 별도 섹션에 분리되어 있다.
"""

import json
import os
from datetime import time as dt_time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path / file constants
# ---------------------------------------------------------------------------
DEFINE_TODAY_CODE_PATH = "r008_trade_watchlist_today.txt"
DATA_DIR_NAME = "data"
# ---------------------------------------------------------------------------
# Feature: R008 watchlist pipeline (scanner -> r008 -> r006 live)
# ---------------------------------------------------------------------------
FEATURE_R008_WATCHLIST = True
FEATURE_SCAN_EXPORT_TO_R008 = True
FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS = True

R008_WATCHLIST_FILENAME = DEFINE_TODAY_CODE_PATH
SCAN_PICKS_LEGACY_FILENAME = "picks.txt"
SCAN_PICKS_PREFIX_TEMPLATE = "_{date}_picks.txt"


# =============================================================================
# LIVE TRADING TUNABLE PARAMETERS
#   r006_trade_live_execute.py(실전 매매)가 매수/매도/리스크 판단에 직접
#   사용하는 값 모음. 실전매매 결과를 보고 튜닝할 때는 이 섹션만 수정하면 된다.
#   (r007_trade_simulate_by_date.py도 동일 로직 재현을 위해 이 값들을 그대로 사용)
# =============================================================================

# --- 1. 손절 (Stop Loss) ----------------------------------------------------
TAKE_PROFIT_PERCENT = 0.025  # 고정 익절 비율(+2.5%) - 섹션 2와 함께 관리
STOP_LOSS_PERCENT = -0.021  # 기본 손절 기준(-2.1%)
STOP_LOSS_EARLY_PERCENT = -0.020  # 보유 초기 구간에서 사용하는 완화 손절 기준
STOP_LOSS_MIN_HOLD_SECONDS = 600  # 손절 로직이 본격 적용되기 전 최소 보유 시간(초)
HARD_STOP_LOSS_PCT = 0.017  # 하드스탑 손절 기준 (0.8%->1.2%->1.7%: 백테스트 결과 하드스탑 58건 중 63.8%가 손절 후 본전 이상 회복, 34.5%는 +2%까지 회복 - 급락 후 반등을 놓치지 않도록 완화)
HARD_STOP_MIN_HOLD_SECONDS = 240.0  # 하드스탑 활성화 최소 보유 시간(초) (180->240: 초기 변동성 노이즈에 덜 민감하도록 연장)
ATR_STOP_MULTIPLIER = 1.5  # ATR 기반 손절 배수

# --- 2. 익절 / 트레일링 스탑 (Take Profit / Trailing Stop) ------------------
TRAILING_STOP_FROM_PEAK = 0.02  # 트레일링 스탑: 고점 대비 되돌림 허용폭 (2%)
ENABLE_TP_EXTENSION_TRAILING = True  # 익절 후 연장 트레일링 기능 사용 여부
TP_EXTENSION_TRAIL_FROM_PEAK = 0.006  # 익절 연장 구간 트레일링 폭 (0.4%->0.6%: 너무 타이트하게 익절을 잠가 수익을 조기에 제한하던 문제 완화)
ATR_TAKE_PROFIT_MULTIPLIER = 3.0  # ATR 기반 익절 배수

# --- 3. 매수 후 보호 가드 (Post-buy protective exits) -----------------------
# 매수 직후 급락 방지: 매수 후 일정 시간 동안 현재가가 매수가 대비
# POST_BUY_BB_DROP_PCT 이상 낮은 상태가 POST_BUY_DROP_CONFIRM_SECONDS 동안
# 유지되면 손절 이전에 조기 매도한다.
POST_BUY_BB_DROP_POLLS = 6  # 레거시 로그/호환용 값
POST_BUY_BB_DROP_PCT = 0.014  # 매수가 대비 이탈 임계치 (-1.0%->-1.4%: 백테스트 결과 12건 중 83.3%가 본전 이상 회복, 58.3%는 +2%까지 회복(평균 최대회복 +4.17%) - 매수직후 정상 노이즈 조기청산 완화)
POST_BUY_BB_DROP_ARMED_SECONDS = 180.0  # 매수 후 가드 활성 구간 (3분, 하드스탑 활성 전까지)
POST_BUY_DROP_CONFIRM_SECONDS = 90.0  # 급락 지속 확인 시간 (60초->90초)

# 초기 수익 반납 실패 보호 (본전 부근에서 반납 시 조기 청산)
BREAKEVEN_FAIL_ARM_PNL = 0.010  # 한 번이라도 +1.0% 이익 도달 시 활성화 (0.8%->1.0%)
BREAKEVEN_FAIL_GIVEBACK_PCT = 0.028  # 고점 대비 2.8% 이상 반납 (0.8%->2.0%->2.8%: 백테스트 결과 14건 중 92.9%가 본전 이상 회복, 42.9%는 +2%까지 회복 - 조기청산 완화)
BREAKEVEN_FAIL_CONFIRM_SECONDS = 150.0  # 실패 지속 확인 시간 (60초->120초->150초, 일시 pullback 무시 강화)

# 무추세 시간 손절
NO_TREND_EXIT_ARM_SECONDS = 1200.0  # 20분 동안 추세 미발생 시 점검 시작
NO_TREND_EXIT_MAX_PEAK_PNL = 0.006  # 0.3%->0.6%: 24건 전패(-58,653원) 개선 - 단기 노이즈 무추세 오판 방지
NO_TREND_EXIT_MIN_PNL = -0.005  # 현재 손익이 -0.5% 이하일 때만 적용
NO_TREND_EXIT_CONFIRM_SECONDS = 90.0  # BB 하단 약세 지속 확인 시간 (초)

# --- 4. 매도 보조 시그널 / 박스권 홀드 / MA5-BB 하향크로스 --------------------
# 보조 매도 시그널 score별 최소 요구 수익률
AUX_SELL_MIN_PNL_SCORE2 = 0.015
AUX_SELL_MIN_PNL_SCORE3 = 0.008
AUX_SELL_MIN_PNL_SCORE4 = 0.003
# AUX 매도 슬리피지 버퍼
AUX_SELL_MIN_REALIZED_TARGET_PCT = 0.010
AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT = 0.005

# MA5-BB 하향 크로스 시 즉시 매도 허용 손익 기준
MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007
# MA5-BB 하향 크로스 즉시 매도 최소 점수
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2
# MA5-BB 하향 크로스 확정 최소 점수
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1
# MA5-BB 하향 크로스 매도 허용 최소 손익
MA5_BB_DOWN_CROSS_MIN_PNL = 0.000

# 박스권(횡보) 구간에서는 기술적 매도 홀드 기능 사용
ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
# 박스권 판정 시 참조할 최근 봉 개수
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
# 박스권 판정용 가격 범위 상한
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065
# 박스권 판정용 BB 폭 상한
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080

# --- 5. 매수 진입 - 실시간가/BB 크로스 확인 & 추격매수 방지 -------------------
# BB 크로스 판정용 버퍼(노이즈 필터)
LIVE_PRICE_BB_BUFFER_PCT = 0.0005  # 0.0008
# 상향 크로스 확정에 필요한 연속 관측 횟수 / 최소 유지 시간(초)
LIVE_PRICE_CROSS_CONFIRM_POLLS = 2  # 3 -> 2: 크로스 확인 속도 향상 (30s -> ~10s)
LIVE_PRICE_CROSS_CONFIRM_SECONDS = 10  # 20 -> 10 (보다 빠른 진입 확인)
# 하향 크로스 확정에 필요한 연속 관측 횟수 / 최소 유지 시간(초)
LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS = 1
LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS = 0
# 효과: 라이브 가격 크로스만으로도 진입 가능 (False -> 완화)
ENABLE_STRICT_MA5_BB_GOLDEN_CROSS = False  # True -> False
# 추격매수 방지: 전일 종가 대비 현재가 상승률이 임계치 이상이면 매수 차단
MAX_BUY_RISE_PCT_FROM_PREV_CLOSE = 0.23  # 23%
# 추격매수 방지: 실시간 BB 상향 크로스 없이(신호 없음) MA5/BB 후행 진입 시
# 현재가가 BB 중심선 대비 과도하게 이격되면 매수 차단
MA5_BB_FOLLOW_CHASE_MAX_GAP_PCT = 0.002  # 0.20% -- tightened: buy only when price is within 0.2% of BB middle
# 매수 연속 확인 횟수 / 진입 전 필요한 최소 확정 봉 개수
BUY_CONSECUTIVE_CONFIRM_COUNT = 2
MIN_BARS_REQUIRED = 3

# --- 6. 매수 진입 - 근접교차(Near-cross) / 조기진입 / 가격선행돌파 ------------
# Near-cross ARM 모드: BB 중단과 MA5 간 최대 허용 갭 / MA5 최소 상승률
ENABLE_NEAR_CROSS_ARM = True
NEAR_CROSS_ARM_GAP_MAX = 0.0045
NEAR_CROSS_ARM_MA_RISE_MIN = 0.0006
# ARM 상태가 유효한 최대 봉 수
NEAR_CROSS_ARM_EXPIRE_BARS = 2

# Early near-cross 모드: BB 중단과 MA5 간 최대 허용 갭 / MA5 최소 상승률
ENABLE_EARLY_NEAR_CROSS_ENTRY = True
NEAR_CROSS_EARLY_GAP_MAX = 0.0045
NEAR_CROSS_EARLY_MA_RISE_MIN = 0.0010
# early near-cross 진입 허용 시작/종료 시각, NXT 세션 허용 여부
EARLY_NEAR_CROSS_ALLOWED_START = dt_time(9, 0)
EARLY_NEAR_CROSS_ALLOWED_END = dt_time(11, 30)
EARLY_NEAR_CROSS_ALLOW_NXT = False
# early near-cross 절대 거래량 최소치 / 거래량 MA 최소치 / 최소 거래대금(KRW)
EARLY_NEAR_CROSS_MIN_VOLUME = 800
EARLY_NEAR_CROSS_MIN_VOL_MA = 500
EARLY_NEAR_CROSS_MIN_TURNOVER_KRW = 5_000_000

# 매수 진입 거래량 MA20 최소치 / 현재봉 거래량 최소치 (저유동성 차단, 공통)
MIN_ENTRY_VOL_MA = 1000
MIN_ENTRY_VOLUME = 1500

# 가격 선행 돌파(price-lead breakout) 진입 로직 사용 여부 및 조건
ENABLE_PRICE_LEAD_BB_BREAKOUT = True
PRICE_LEAD_BREAKOUT_MIN_SCORE = 3  # 최소 보조지표 점수
PRICE_LEAD_BREAKOUT_MIN_ADX = 25.0  # 최소 ADX
PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT = True  # 과열 상태에서도 허용 여부

# BB 중앙선 상승 돌파 전략 파라미터 (BB slope break cross strategy)
BB_SLOPE_LOOKBACK_BARS = 20      # BB 기울기 측정 봉 수 (3분봉 기준 약 1시간)
BB_MID_DOWNTREND_BARS = 5        # BB 중간선 우하향 감지 봉 수 (3분봉 기준 약 15분): 연속 하락 시 매수 차단
BB_UPPER_GAP_MIN_PCT = 0.25      # BB 상단 여유 최소치 (%) - 상단까지 여유 없으면 매수 차단 (0.5->0.25->0.5->0.25)
CANDLE_GAIN_MIN_PCT = 0.0        # 현재봉 양봉 최소 상승률 (%) - 음봉만 차단, 0.00%는 허용 (0.1->0.0)
CANDLE_GAIN_MAX_PCT = 0.8        # 현재봉 최대 허용 상승률 (%) - 초과 시 추격 매수 차단
BB_MID_CHASE_MAX_GAP_PCT = 0.35  # BB 중간선 대비 현재가 최대 허용 갭 (%) - 초과 시 추격 매수 차단 (1.0->0.7)
BB_BUY_SCORE_THRESHOLD = 10  # 8->10: 2026-07-20 실매매 로그 분석 결과 score=8(구 임계값) 매수 6건이
  # 전부 손실(합계 -18,975원, 당일 총손실 -21,820원의 87%). score=9 매수 4건도 순손실(-2,295원).
  # score>=10 매수만 순이익 포함(GS +1,400원 익절 등, 순 -550원 vs 조정 전 -21,820원 전체손익).
  # 표본 1일(16건)이라 지속 모니터링 필요. 매수 최소 점수 (공격형=6, 중립형=8, 보수형=10)
# 개장 직후 보호: 장 시작(09:00) 후 이 분 수 이내에는 score threshold를 높여 추격매수 방지
OPENING_GUARD_MINUTES = 15       # 개장 후 15분(첫 5봉) 동안 강화된 필터 적용
OPENING_GUARD_SCORE_THRESHOLD = 12  # 개장 직후 요구 최소 점수 (일반=8, 개장보호=12)

# --- 7. 매수 진입 - 보조지표 임계값 (과열/모멘텀 필터) ------------------------
STOCH_OVERBOUGHT = 96.0  # 85.0 -> 92.0 -> 96.0 (과열 차단 기준 완화)
STOCH_BUY_MIN = 20.0  # 매수 스토캐스틱 K 하한
STOCH_BUY_MAX = 50.0  # 매수 스토캐스틱 K 상한
RSI_BUY_MIN = 50.0  # 매수 RSI 하한
RSI_BUY_MAX = 70.0  # 매수 RSI 상한
RSI_BUY_MOMENTUM_MAX = 60.0  # RSI 모멘텀 허용 상한(기본: 50~60 구간 유지)
WILLIAMS_BUY_FLOOR = -70.0  # 매수 허용 Williams %R 하한
WILLIAMS_OVERBOUGHT_CEIL = -10  # -20 -> -10 (Williams R 완화)
BB_UPPER_PROXIMITY_MAX = 1.05  # 0.85 -> 1.05 (BB 상단에서 충분히 진입 허용)
BB_SQUEEZE_MIN_WIDTH_PCT = 0.0  # BB 폭 최소치(너무 좁은 횡보 구간 배제)

ADX_MIN_TREND = 15.0  # 20.0 -> 15.0 (ADX 최소값 완화)
ADX_STRONG_TREND = 40.0
ADX_BUY_MIN = 25.0  # 매수 진입용 ADX 최소값
REQUIRE_ADX_RISING = True  # 매수 시 ADX가 직전봉 대비 우상향이어야 하는지 여부
REQUIRE_DI_PLUS_DOMINANT = True  # 매수 시 +DI가 -DI보다 커야 하는지 여부

MFI_BUY_MIN = 50.0  # 매수 진입용 MFI 최소값
MFI_OVERBOUGHT_MAX = 80.0  # MFI 과열 상한(이상일 경우 추격 매수 금지)

REQUIRE_OBV_SIGNAL_CROSS = True  # OBV 시그널 골든크로스+돌파 확인 필수 여부
OBV_BREAKOUT_LOOKBACK_BARS = 5  # OBV 돌파 판정에 사용할 과거 봉 개수

# 강추세일 때 과열 필터 일부 우회 허용
ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS = True
STRONG_TREND_OVERBOUGHT_MIN_SCORE = 2  # 과열 우회 허용 최소 보조지표 점수
STRONG_TREND_OVERBOUGHT_MIN_VOL_RATIO = 1.00  # 과열 우회 허용 최소 거래량 비율
STRONG_TREND_OVERBOUGHT_MIN_ADX = 15.0  # 과열 우회 허용 최소 ADX

# --- 8. 거래량 필터 ----------------------------------------------------------
VOLUME_RATIO_OPEN = 0.75  # 0.40 -> 0.75 (개장 초반 저거래량 종목 진입 차단 강화)
VOLUME_RATIO_MIDDAY = 0.75  # 0.50 -> 0.75 (저거래량 추격매수 차단)
VOLUME_RATIO_CLOSE = 0.80  # 0.60 -> 0.80 (장마감 저거래량 차단 강화)
VOLUME_RATIO_NXT = 0.30  # 0.40 -> 0.30 (NXT 세션 거래량 필터 더 완화)
VOLUME_RATIO_STRONG_RELAX = 0.15  # 0.10 -> 0.15 (강한 추세 보너스 증가)
VOLUME_RATIO_FLOOR = 0.70  # 0.55 -> 0.70 (최소 하한 강화)

# --- 9. 인트라바 실시간 진입 필터 --------------------------------------------
ENABLE_INTRABAR_LIVE_ENTRY_FILTER = True
INTRABAR_MIN_ELAPSED_SECONDS = 90.0
INTRABAR_MFI_MIN = 50.0
INTRABAR_MFI_MAX = 75.0
INTRABAR_RSI_MIN = 50.0
INTRABAR_RSI_MAX = 70.0
INTRABAR_ADX_MIN = 20.0

# --- 10. 리스크 서킷브레이커 / 재진입 & 안전장치 -----------------------------
# HARD_STOP이 N번 연속 발생하면 N분간 신규 매수 차단 (6/22, 6/23 전손절 방지)
HARD_STOP_CIRCUIT_BREAKER_COUNT = 3      # 연속 HARD_STOP 발생 횟수 임계치
HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN = 60  # 서킷브레이커 발동 시 신규 매수 차단 시간(분)
# 당일 HARD_STOP_LOSS 발생 종목 재진입 차단 여부 (같은 날 같은 종목 손절 후 재매수 금지)
HARD_STOP_BLOCK_REENTRY_TODAY = True
# 당일 같은 종목 재매수 허용 여부 / 동일 종목 재진입 쿨다운(분)
ALLOW_REBUY_SAME_CODE = False
TRADE_COOLDOWN_MINUTES = 3
# 시장일 확인 실패 시 보수적으로 비거래 처리
MARKET_DAY_FAIL_CLOSED = True

# --- 11. 주문/자금 관리 -------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000  # 1회 매수 주문 최대 금액(KRW)


# ---------------------------------------------------------------------------
# Indicator periods (지표 계산 주기) - 매매 판단 임계값이 아닌 지표 자체의
# 계산 기간/배수. 값을 바꾸면 지표 모양 자체가 달라지므로 위 튜닝 섹션과는
# 분리해서 관리한다.
# ---------------------------------------------------------------------------
# 볼린저밴드 기준 기간(봉 개수)
BB_PERIOD = 20
# 볼린저밴드 표준편차 배수(상/하단 폭)
BB_STD_MULTIPLIER = 2.0
# 기본 이동평균 기간(공용)
MA_PERIOD = 5
# 스토캐스틱 %K 계산 기간
STOCH_K_PERIOD = 10
# 스토캐스틱 %D 평활 기간
STOCH_D_PERIOD = 5
# RSI 계산 기간
RSI_PERIOD = 14
# RSI 시그널선 평활 기간
RSI_SIGNAL_PERIOD = 6
# Williams %R 계산 기간
WILLIAMS_R_PERIOD = 10
# Williams %D(평활) 기간
WILLIAMS_D_PERIOD = 9
# MFI 계산 기간
MFI_PERIOD = 14
# 거래량 이동평균 기간
VOLUME_MA_PERIOD = 20
# OBV 이동평균 기간
OBV_MA_PERIOD = 10
# MACD 단기 EMA 기간
MACD_FAST = 5
# MACD 장기 EMA 기간
MACD_SLOW = 12
# MACD 시그널 EMA 기간
MACD_SIGNAL_PERIOD = 4
# ADX 계산 기간
ADX_PERIOD = 7
# ATR 계산 기간
ATR_PERIOD = 14

# ---------------------------------------------------------------------------
# Session / time constants
# ---------------------------------------------------------------------------
# NXT 세션 활성화 여부 및 시간 설정
ENABLE_NXT_SESSION = True  # NXT 세션 포함 운용 여부
MORNING_NXT_START = dt_time(8, 0)
MORNING_NXT_END = dt_time(8, 50)
REGULAR_START = dt_time(9, 0)
REGULAR_END = dt_time(15, 30)
# 신규 진입 허용 종료 시각(정규장)
REGULAR_NEW_ENTRY_CUTOFF = dt_time(15, 20)
# 강제 청산 시작 시각(정규장)
REGULAR_FORCE_EXIT = dt_time(15, 20)
AFTERNOON_NXT_START = dt_time(15, 30)
AFTERNOON_NXT_END = dt_time(20, 0)
# 신규 진입 허용 종료 시각(오후 NXT)
AFTERNOON_NXT_NEW_ENTRY_CUTOFF = dt_time(19, 59)
# 강제 청산 시작 시각(오후 NXT)
AFTERNOON_NXT_FORCE_EXIT = dt_time(19, 59)
# 세션 종료 컷오프에서 전량 청산 강제 여부
SESSION_FORCE_CLOSE_ALL_AT_CUTOFF = True
# 세션 종료 홀드 예외 허용 여부
ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP = False
# 오전 NXT 신규 진입 종료 시각(별도 변수)
MORNING_NXT_NEW_ENTRY_CUTOFF = MORNING_NXT_END

# ---------------------------------------------------------------------------
# Live execution operational parameters (r006_trade_live_execute) - 매매
# 판단값이 아닌 시스템 운영(폴링 주기/백오프/재시도 등) 파라미터.
# ---------------------------------------------------------------------------
STARTUP_WARMUP_SECONDS = 90
# 메인 루프 폴링 간격(초)
POLL_INTERVAL_SECONDS = 10  # 15 -> 10 (더 빠른 대응)
# 실시간 현재가 재조회 간격(초) - 매분 00/10/20/30/40/50초 등 이 값의 배수 시각에 맞춰 폴링(r006 참고)
LIVE_PRICE_POLL_INTERVAL_SECONDS = 10  # 20->10 원복: 사용자 확인 결과 기존 10초 유지가 맞음
# 계좌/체결 상태 동기화 주기(초)
ACCOUNT_SYNC_INTERVAL_SECONDS = 90
# 3분봉 프레임 갱신 주기(초)
FRAME_POLL_INTERVAL_SECONDS = 20
# 시작 시 과거 바 백필 동기화 범위(초)
FRAME_BACKFILL_SYNC_SECONDS = 600
# 주문 상태 폴링 간격(초)
ORDER_STATUS_POLL_INTERVAL_SECONDS = 15
# 현재가 조회 backoff 초기값(초) / 최대값(초)
LIVE_PRICE_BACKOFF_BASE_SECONDS = 5
LIVE_PRICE_BACKOFF_MAX_SECONDS = 60
# 현재가 stale TTL(초)
LIVE_PRICE_STALE_TTL_SECONDS = 20
# 미결 주문 상태 backoff 최대값(초)
PENDING_STATUS_BACKOFF_MAX_SECONDS = 120
# 메인 루프 연속 오류 허용 최대 횟수
MAIN_LOOP_MAX_CONSECUTIVE_ERRORS = 20
# 라이브 상태 저장 주기(초)
LIVE_STATE_SAVE_INTERVAL_SECONDS = 60
# 미결 매수 주문 대기 여유 시간(초)
PENDING_BUY_GRACE_SECONDS = 90
# 매수 미체결 경고 기준 시간(초)
BUY_ORDER_STALE_WARN_SECONDS = 60
# 계좌-감시종목 불일치 로그 출력 최소 간격(초)
WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS = 300

# ---------------------------------------------------------------------------
# Simulation parameters (r007_trade_simulate_by_date) - r006 실전 매매에는
# 적용되지 않는 시뮬레이션 전용 값.
# ---------------------------------------------------------------------------
# 초기 시뮬레이션 자본금(KRW)
SIM_INITIAL_CAPITAL = 5_000_000
# 매도 완료 후 재진입 허용 여부
SIM_ALLOW_REENTRY_AFTER_COMPLETED_SELL = True
# 웜업용 tail 봉 개수
SIM_WARMUP_TAIL_BARS = 160
# 웜업 이전 데이터 최대 조회일수
SIM_WARMUP_PRIOR_MAX_DAYS = 20
# 기술적 매도 최소 보유 시간(초)
TECH_SELL_MIN_HOLD_SECONDS = 300
# 당일 최소 처리 봉 수
SAME_DAY_MIN_BARS = 10
# 시뮬레이션 내부 체크 간격(초)
SIM_CHECK_INTERVAL_SECONDS = 10
# 10초 그리드 시뮬레이션 기본값
SIMULATE_10S_GRID_DEFAULT = True
# 인트라바 볼륨 폴백 활성화
ENABLE_INTRABAR_VOLUME_FALLBACK = True
# 인트라바 볼륨 폴백 최소 진행률
INTRABAR_VOLUME_FALLBACK_MIN_PROGRESS = 0.30

# 시뮬레이션 완화 게이트 (공유 진입 게이트가 거래를 0건으로 막는 것을 방지)
SIM_RELAXED_SHARED_GATES = True
SIM_RELAXED_MIN_SUPPORT_SCORE = 0
SIM_RELAXED_VWAP_MAX_UNDER_PCT = 0.0025
SIM_RELAXED_REQUIRE_CONFIRMED_ABOVE = False
SIM_RELAXED_REQUIRE_MA5_BIAS = False
SIM_RELAXED_ALLOW_BELOW_BB = True
SIM_RELAXED_ALLOW_FALLING_TREND = True


def _apply_risk_profile_overrides() -> None:
	raw_profile = (os.environ.get("AUTO_TRADING_RISK_PROFILE") or os.environ.get("RISK_PROFILE") or "").strip()
	if not raw_profile:
		raw_profile = "neutral"

	alias = {
		"safe": "conservative",
		"conservative": "conservative",
		"보수": "conservative",
		"balanced": "neutral",
		"neutral": "neutral",
		"중립": "neutral",
		"aggressive": "aggressive",
		"attack": "aggressive",
		"공격": "aggressive",
	}
	profile = alias.get(raw_profile.lower())
	if profile is None:
		print(
			"[WARN] Unknown AUTO_TRADING_RISK_PROFILE "
			f"'{raw_profile}'. Expected conservative|neutral|aggressive (or 보수|중립|공격)."
		)
		return

	profile_path = Path(__file__).resolve().parent / "risk_profiles" / f"{profile}.json"
	if not profile_path.is_file():
		print(f"[WARN] Risk profile file not found: {profile_path}")
		return

	try:
		overrides = json.loads(profile_path.read_text(encoding="utf-8"))
	except Exception as exc:
		print(f"[WARN] Failed to load risk profile '{profile}': {exc}")
		return

	if not isinstance(overrides, dict):
		print(f"[WARN] Invalid risk profile format (expected object): {profile_path}")
		return

	applied_keys: list[str] = []
	for key, value in overrides.items():
		if key in globals():
			globals()[key] = value
			applied_keys.append(key)

	print(f"[INFO] Applied risk profile '{profile}' ({len(applied_keys)} overrides)")


_apply_risk_profile_overrides()