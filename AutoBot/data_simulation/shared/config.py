# -*- coding: utf-8 -*-
"""R73 MA5/BB Cross Strategy - Unified Configuration."""

from datetime import time as dt_time

# ---------------------------------------------------------------------------
# Indicator Parameters
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

MACD_FAST = 5
MACD_SLOW = 12
MACD_SIGNAL_PERIOD = 4

ADX_PERIOD = 7
ADX_MIN_TREND = 20.0
ADX_STRONG_TREND = 40.0

# ---------------------------------------------------------------------------
# Trading Parameters
# ---------------------------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000
TAKE_PROFIT_PERCENT = 0.035
STOP_LOSS_PERCENT = -0.015
TRAILING_STOP_FROM_PEAK = 0.012

POLL_INTERVAL_SECONDS = 20
ACCOUNT_SYNC_INTERVAL_SECONDS = 90
MIN_BARS_REQUIRED = 3
ALLOW_REBUY_SAME_CODE = False
TRADE_COOLDOWN_MINUTES = 3

# ---------------------------------------------------------------------------
# Session / Time Constants
# ---------------------------------------------------------------------------
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
# Indicator Thresholds
# ---------------------------------------------------------------------------
STOCH_OVERBOUGHT = 80.0
STOCH_BUY_MAX = 72.0
RSI_BUY_MIN = 45.0
RSI_BUY_MAX = 72.0
WILLIAMS_BUY_FLOOR = -70.0
WILLIAMS_OVERBOUGHT_CEIL = -20.0
BB_UPPER_PROXIMITY_MAX = 0.85

# ---------------------------------------------------------------------------
# Volume Ratio Thresholds (by time session)
# ---------------------------------------------------------------------------
VOLUME_RATIO_OPEN = 0.80          # 09:00~10:00
VOLUME_RATIO_MIDDAY = 0.60        # 10:00~14:30
VOLUME_RATIO_CLOSE = 0.70         # 14:30~15:30
VOLUME_RATIO_NXT = 0.55           # NXT session
VOLUME_RATIO_STRONG_RELAX = 0.10  # relaxation when ADX strong trend
VOLUME_RATIO_FLOOR = 0.50         # absolute floor
