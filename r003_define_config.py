# -*- coding: utf-8 -*-
"""Central strategy settings for simulation, optimization, and live trading."""

from datetime import time as dt_time

DEFINE_TODAY_CODE_PATH = "r008_trade_watchlist_today.txt"

BB_PERIOD = 20
BB_STD_DEV_MULTIPLIER = 2.0
MA_PERIOD = 5

MAX_ORDER_AMOUNT_KRW = 500_000
INTRADAY_TAKE_PROFIT_PERCENT = 0.02
# 손절 폭 완화: -1.5% -> -2.2% (전략 필터와 함께 조정)
INTRADAY_STOP_LOSS_PERCENT = -0.022
USE_INTRADAY_TP_SL = True

MORNING_NXT_START = dt_time(8, 0)
MORNING_NXT_END = dt_time(8, 50)
REGULAR_START = dt_time(9, 0)
REGULAR_END = dt_time(15, 30)
CLOSING_AUCTION_START = dt_time(15, 20)
AFTERNOON_NXT_START = dt_time(15, 30)
AFTERNOON_NXT_END = dt_time(20, 0)
AFTERNOON_NXT_FORCE_LIQUIDATION = dt_time(19, 59)

NXT_LIST_REFRESH_AT = (dt_time(8, 0), dt_time(15, 50))
HISTORICAL_CANDLE_TAIL = BB_PERIOD + MA_PERIOD + 10

# --- R41 / S41: MA5×볼밴 중심 골든/데드 + RSI·Stochastic Fast 동시 신호 ---
RSI_PERIOD = 14
# RSI 매수/매도 신호: 기준값을 직전 봉에서 상향/하향 돌파
RSI_SIGNAL_LEVEL = 50.0
# 직전봉 RSI가 일정 값 이하일 때만 매수(과매수 구간에서 50 상향 돌파 배제). None이면 비활성
RSI_BUY_PREV_MAX = 46.0
# 직전봉 RSI가 일정 값 이상일 때만 매도(과매도 구간에서 50 하향 이탈 배제). None이면 비활성
RSI_SELL_PREV_MIN = 54.0
# Stochastic Fast: %K 기간, %D(%K 이동평균) 기간 (일반적인 빠른 스토캐스틱 5,3)
STOCH_FAST_K = 5
STOCH_FAST_D = 3
# 매수: 직전봉 max(%K,%D)가 밴드 이하(과매수 근처 배제, K가 D 상향). None이면 비활성
STOCH_BUY_PREV_BAND_MAX = 42.0
# 매도: 직전봉 min(%K,%D)가 밴드 이상(과매도 근처 배제, K가 D 하향). None이면 비활성
STOCH_SELL_PREV_BAND_MIN = 58.0

# 정규장 매수 시작 시각: 장 시작 직후 구간 스킵(변동성/갭 구간). REGULAR_START와 같으면 비활성
REGULAR_FIRST_BUY_TIME = dt_time(9, 30)

# 15:29 정규장 마감 직전·19:59 야간 NXT: 손실이 -1% 이내면 조건부 청산
EOD_CLOSE_PROFIT_LOSS_THRESHOLD = -0.01

# 정규장 매매 가능 마감 시각 (15:20 이후 정규장 진입 없음)
REGULAR_TRADE_END = dt_time(15, 20)
# 정규장 마감 직전 조건부 청산 시각
MAIN_EOD_CONDITIONAL_TIME = dt_time(15, 29)

DATA_DIR_NAME = "data"
PICKS_FILENAME = "picks.txt"
INITIAL_CAPITAL = 1_000_000

# 장애·레이트리밋 시 API 대신 사용할 NXT 매매 가능 종목 (실전에서는 API로 갱신)
NXT_ELIGIBLE_CODES_FALLBACK = frozenset(
    {"005930", "000660", "035420", "068270", "018880", "103590", "003490"}
)

# 워밍업: 당일 계산 전 직전 거래일 CSV 꼬리 구간 병합
SIM_WARMUP_TAIL_BARS = 160
SIM_WARMUP_PRIOR_MAX_DAYS = 20

# --- R51 / S51: 실전형 3분봉 패턴 기반 (MA5 vs BB중심) ---
R51_DEFINE_TODAY_CODE_PATH = "r008_trade_watchlist_today.txt"
R51_BB_PERIOD = 20
R51_BB_STD_DEV_MULTIPLIER = 2.0
R51_MA_SHORT_PERIOD = 5
R51_MAX_ORDER_AMOUNT_KRW = 500_000
R51_TAKE_PROFIT_PERCENT = 0.04
R51_STOP_LOSS_PERCENT = -0.015

# NXT 가능 종목 체크 적용 모드
# True: NXT 시간대에만 NXT 가능 종목 필터 적용(정규장은 전체 허용)
# False: 정규장과 NXT 모두 NXT 가능 종목만 거래
R51_NXT_FILTER_ONLY_DURING_NXT = True

# 매수: MA5가 BB중심선을 상향 돌파 + 양봉
R51_REQUIRE_BULLISH_BUY = True
# 매도: MA5가 BB중심선을 하향 돌파 + 음봉
R51_REQUIRE_BEARISH_SELL = True

# R51 RSI/Stochastic Fast 필터
R51_RSI_PERIOD = 14
R51_RSI_SIGNAL_LEVEL = 50.0
# 매수: RSI 상향 돌파 + 직전 RSI 상한 조건. None이면 비활성
R51_RSI_BUY_PREV_MAX = 60.0
# 매도: RSI 하향 돌파 + 직전 RSI 하한 조건. None이면 비활성
R51_RSI_SELL_PREV_MIN = 45.0

R51_STOCH_FAST_K = 5
R51_STOCH_FAST_D = 3
# 매수: 직전봉 max(%K,%D) <= 상한값. None이면 비활성
R51_STOCH_BUY_PREV_BAND_MAX = 70.0
# 매도: 직전봉 min(%K,%D) >= 하한값. None이면 비활성
R51_STOCH_SELL_PREV_BAND_MIN = 30.0

# 매수(엄격): RSI/스토캐스틱 모두 교차 신호가 동시에 필요
# 매수(완화): 교차가 아니어도 RSI>시그널, K>D 상태면 허용
R51_BUY_REQUIRE_CROSS_SIGNAL_ONLY = False

# 완화 모드에서 교차 신호 최소 1개 강제 여부
# False: 기존 동작(상태 기반 매수 허용) / True: 교차 신호가 있어야만 매수
R51_BUY_REQUIRE_AT_LEAST_ONE_CROSS_SIGNAL = False

# 매도(완화): RSI<시그널 또는 K<D 상태면 교차 없이도 청산 허용
R51_SELL_ALLOW_STATE_EXIT = True

# R51 정규장 매매 가능 마감 시각 (해당 시각 이후 정규장 진입 없음, 15:20 대비 버퍼)
R51_REGULAR_TRADE_END = dt_time(14, 30)

# Basic settings
INVEST_AMOUNT = 100000
ORDER_SIZE = 10

# Buy conditions
RSI_BUY = 65       # 55에서 65로 완화 (골든크로스 시점 매수 확률 증가)
VOLUME_THRESHOLD = 0.0  # 0.0으로 설정하여 골든크로스에서 거래량 필터 무시
MA_LONG_PERIOD = 60
BB_PULLBACK_TOLERANCE = 0.03
MIN_BB_WIDTH = 0.005
USE_RSI_BUY_FILTER = False
ALLOW_REBUY_SAME_CODE = False

# Sell conditions
RSI_SELL = 78
STOP_LOSS = -3.0
TAKE_PROFIT = 1.0

# Partial take profit settings
TAKE_PROFIT_PARTIAL_1 = 2.0
TAKE_PROFIT_PARTIAL_1_QTY_PCT = 0.4
TAKE_PROFIT_PARTIAL_2 = 3.5
TAKE_PROFIT_PARTIAL_2_QTY_PCT = 0.5
BREAK_EVEN_BUFFER = 0.2

# Time-based liquidation
HOLD_TIME_MINUTES = 6
FORCE_SELL_TIME_MINUTES = 60

# Trailing stop
TRAILING_STOP = -1.6

# Optimizer defaults
OPTIMIZER_PARAM_GRID = {
    "rsi_buy": [42, 48, 55, 60],
    "rsi_sell": [72, 76, 78],
    "stop_loss": [-2.0, -2.5, -3.0],
    "take_profit": [3.5, 5.0, 6.0],
    "volume_threshold": [0.3, 0.4, 0.6],
    "trailing_stop": [-1.0, -1.6, -2.0],
    "hold_time_minutes": [6, 9, 12],
    "force_sell_time_minutes": [30, 45, 60],
}


