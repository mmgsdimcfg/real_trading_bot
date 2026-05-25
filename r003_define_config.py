# -*- coding: utf-8 -*-

"""R76 shared configuration for live trading and simulation."""

from datetime import time as dt_time

# ---------------------------------------------------------------------------
# Path / file constants
# ---------------------------------------------------------------------------
DEFINE_TODAY_CODE_PATH = "r008_trade_watchlist_today.txt"
DATA_DIR_NAME = "data"

# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------
BB_PERIOD = 20
BB_STD_MULTIPLIER = 2.0
MA_PERIOD = 5
STOCH_K_PERIOD = 10
STOCH_D_PERIOD = 5
RSI_PERIOD = 14
RSI_SIGNAL_PERIOD = 6
WILLIAMS_R_PERIOD = 10
WILLIAMS_D_PERIOD = 9
VOLUME_MA_PERIOD = 20
OBV_MA_PERIOD = 10
MACD_FAST = 5
MACD_SLOW = 12
MACD_SIGNAL_PERIOD = 4
ADX_PERIOD = 7

# ADX 최소값 완화
ADX_MIN_TREND = 15.0  # 20.0 -> 15.0
ADX_STRONG_TREND = 40.0

# ---------------------------------------------------------------------------
# Trading parameters
# ---------------------------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000
TAKE_PROFIT_PERCENT = 0.035

# STOP LOSS
STOP_LOSS_PERCENT = -0.025  # -2.5%
STOP_LOSS_EARLY_PERCENT = -0.020
STOP_LOSS_MIN_HOLD_SECONDS = 600

# Post-buy drop early-exit guard (entry-price anchored)
# 매수 직후 급락 방지: 매수 후 일정 시간 동안 현재가가
# 매수가 대비 POST_BUY_BB_DROP_PCT 이상 낮은 상태가
# POST_BUY_DROP_CONFIRM_SECONDS 동안 유지되면 손절 이전에 조기 매도한다.
POST_BUY_BB_DROP_POLLS = 6  # 레거시 로그/호환용 값
POST_BUY_BB_DROP_PCT = 0.007  # 매수가 대비 이탈 임계치 (-0.7%)
POST_BUY_BB_DROP_ARMED_SECONDS = 300.0  # 매수 후 가드 활성 구간 (초, 기본 5분)
POST_BUY_DROP_CONFIRM_SECONDS = 40.0  # 급락 지속 확인 시간 (초)

# 초기 수익 반납 실패 보호
BREAKEVEN_FAIL_ARM_PNL = 0.008  # 한 번이라도 +0.8% 이익 도달 시 활성화
BREAKEVEN_FAIL_GIVEBACK_PCT = 0.0075  # 고점 대비 0.75% 이상 반납
BREAKEVEN_FAIL_CONFIRM_SECONDS = 30.0  # 실패 지속 확인 시간 (초)

# 무추세 시간 손절
NO_TREND_EXIT_ARM_SECONDS = 900.0  # 15분 동안 추세 미발생 시 점검 시작
NO_TREND_EXIT_MAX_PEAK_PNL = 0.005  # 최고 수익이 +0.5% 미만이면 무추세로 간주
NO_TREND_EXIT_MIN_PNL = -0.003  # 현재 손익이 -0.3% 이하일 때만 적용
NO_TREND_EXIT_CONFIRM_SECONDS = 60.0  # BB 하단 약세 지속 확인 시간 (초)

STARTUP_WARMUP_SECONDS = 90

# 1. 트레일링 임계값 상향
TRAILING_STOP_FROM_PEAK = 0.02  # 2%

AUX_SELL_MIN_PNL_SCORE2 = 0.015
AUX_SELL_MIN_PNL_SCORE3 = 0.008
AUX_SELL_MIN_PNL_SCORE4 = 0.003

MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1
MA5_BB_DOWN_CROSS_MIN_PNL = 0.000

ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
ENABLE_TP_EXTENSION_TRAILING = True
TP_EXTENSION_TRAIL_FROM_PEAK = 0.01  # 1%
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080

NEAR_CROSS_ARM_GAP_MAX = 0.0045
NEAR_CROSS_ARM_MA_RISE_MIN = 0.0006
NEAR_CROSS_EARLY_GAP_MAX = 0.0045
NEAR_CROSS_EARLY_MA_RISE_MIN = 0.0010
NEAR_CROSS_ARM_EXPIRE_BARS = 2

ENABLE_NEAR_CROSS_ARM = True
ENABLE_EARLY_NEAR_CROSS_ENTRY = True
ENABLE_PRICE_LEAD_BB_BREAKOUT = True
PRICE_LEAD_BREAKOUT_MIN_SCORE = 3
PRICE_LEAD_BREAKOUT_MIN_ADX = 25.0
PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT = True

EARLY_NEAR_CROSS_ALLOWED_START = dt_time(9, 0)
EARLY_NEAR_CROSS_ALLOWED_END = dt_time(11, 30)
EARLY_NEAR_CROSS_ALLOW_NXT = False
EARLY_NEAR_CROSS_MIN_VOLUME = 800
EARLY_NEAR_CROSS_MIN_VOL_MA = 500
EARLY_NEAR_CROSS_MIN_TURNOVER_KRW = 5_000_000

# 폴링 주기
POLL_INTERVAL_SECONDS = 10  # 15 -> 10 (더 빠른 대응)
LIVE_PRICE_POLL_INTERVAL_SECONDS = 5
LIVE_PRICE_BB_BUFFER_PCT = 0.0005  # 0.0008
LIVE_PRICE_CROSS_CONFIRM_POLLS = 3
LIVE_PRICE_CROSS_CONFIRM_SECONDS = 10  # 20 -> 10 (보다 빠른 진입 확인)
LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS = 1
LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS = 0
ACCOUNT_SYNC_INTERVAL_SECONDS = 90

MIN_BARS_REQUIRED = 3
ALLOW_REBUY_SAME_CODE = False
TRADE_COOLDOWN_MINUTES = 3

# ---------------------------------------------------------------------------
# Session / time constants
# ---------------------------------------------------------------------------
# NXT 세션 활성화 여부 및 시간 설정
ENABLE_NXT_SESSION = True  # False
MORNING_NXT_START = dt_time(8, 0)
MORNING_NXT_END = dt_time(8, 50)
REGULAR_START = dt_time(9, 0)
REGULAR_END = dt_time(15, 30)
REGULAR_NEW_ENTRY_CUTOFF = dt_time(15, 20)
REGULAR_FORCE_EXIT = dt_time(15, 20)
AFTERNOON_NXT_START = dt_time(15, 30)
AFTERNOON_NXT_END = dt_time(20, 0)
AFTERNOON_NXT_NEW_ENTRY_CUTOFF = dt_time(19, 59)
AFTERNOON_NXT_FORCE_EXIT = dt_time(19, 59)

# ---------------------------------------------------------------------------
# Filter coefficients
# ---------------------------------------------------------------------------
STOCH_OVERBOUGHT = 92.0  # 85.0 -> 92.0 (3분봉 단타에서 80~90 이상도 급등신호)
STOCH_BUY_MIN = 20.0
STOCH_BUY_MAX = 50.0
RSI_BUY_MIN = 50.0
RSI_BUY_MAX = 70.0
WILLIAMS_BUY_FLOOR = -70.0

# 2. Williams R 완화
WILLIAMS_OVERBOUGHT_CEIL = -10  # -20 -> -10
BB_UPPER_PROXIMITY_MAX = 1.05  # 0.85 -> 1.05 (BB 상단에서 충분히 진입 허용)
BB_SQUEEZE_MIN_WIDTH_PCT = 0.0
OBV_BREAKOUT_LOOKBACK_BARS = 5

# 효과: 라이브 가격 크로스만으로도 진입 가능
ENABLE_STRICT_MA5_BB_GOLDEN_CROSS = False  # True -> False
ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS = True
STRONG_TREND_OVERBOUGHT_MIN_SCORE = 2
STRONG_TREND_OVERBOUGHT_MIN_VOL_RATIO = 1.00
STRONG_TREND_OVERBOUGHT_MIN_ADX = 15.0

# 3. 거래량 완화
VOLUME_RATIO_OPEN = 0.40  # 0.60 -> 0.40 (아침 변동성 높은 시간 더 완화)
VOLUME_RATIO_MIDDAY = 0.35  # 0.45 -> 0.35 (한낮 변동성 낮은 시간 더 완화)
VOLUME_RATIO_CLOSE = 0.50  # 0.70 -> 0.50 (마감 거래량 필터 완화)
VOLUME_RATIO_NXT = 0.30  # 0.40 -> 0.30 (NXT 세션 거래량 필터 더 완화)
VOLUME_RATIO_STRONG_RELAX = 0.15  # 0.10 -> 0.15 (강한 추세 보너스 증가)
VOLUME_RATIO_FLOOR = 0.50

MARKET_DAY_FAIL_CLOSED = True
SESSION_FORCE_CLOSE_ALL_AT_CUTOFF = True
ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP = False
WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS = 300

MORNING_NXT_NEW_ENTRY_CUTOFF = MORNING_NXT_END
