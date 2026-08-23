# -*- coding: utf-8 -*-
"""R73 Buy/Sell Signal Logic."""

import pandas as pd
from datetime import datetime, time as dt_time

from .config import (
    ADX_MIN_TREND,
    ADX_STRONG_TREND,
    AFTERNOON_NXT_END,
    AFTERNOON_NXT_START,
    BB_UPPER_PROXIMITY_MAX,
    MA_PERIOD,
    MORNING_NXT_END,
    MORNING_NXT_START,
    REGULAR_END,
    REGULAR_START,
    RSI_BUY_MAX,
    RSI_BUY_MIN,
    STOCH_BUY_MAX,
    STOCH_OVERBOUGHT,
    VOLUME_RATIO_CLOSE,
    VOLUME_RATIO_FLOOR,
    VOLUME_RATIO_MIDDAY,
    VOLUME_RATIO_NXT,
    VOLUME_RATIO_OPEN,
    VOLUME_RATIO_STRONG_RELAX,
    WILLIAMS_BUY_FLOOR,
    WILLIAMS_OVERBOUGHT_CEIL,
)


def _num(candle: pd.Series, key: str) -> float:
    """Safely extract numeric value from candle."""
    value = candle.get(key)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def is_regular_session(ts: pd.Timestamp | datetime) -> bool:
    """Check if timestamp is in regular session."""
    if isinstance(ts, pd.Timestamp):
        ts_time = ts.time()
    else:
        ts_time = ts.time()
    return REGULAR_START <= ts_time <= REGULAR_END


def is_nxt_session(ts: pd.Timestamp | datetime) -> bool:
    """Check if timestamp is in NXT session."""
    if isinstance(ts, pd.Timestamp):
        ts_time = ts.time()
    else:
        ts_time = ts.time()
    return (MORNING_NXT_START <= ts_time <= MORNING_NXT_END) or (AFTERNOON_NXT_START <= ts_time <= AFTERNOON_NXT_END)


def get_volume_ratio_threshold(ts: pd.Timestamp | datetime, adx_val: float) -> float:
    """Get dynamic volume ratio threshold based on session and trend strength."""
    if isinstance(ts, pd.Timestamp):
        current_time = ts.time()
    else:
        current_time = ts.time()

    if is_nxt_session(ts):
        ratio = VOLUME_RATIO_NXT
    elif current_time < dt_time(10, 0):
        ratio = VOLUME_RATIO_OPEN
    elif current_time < dt_time(14, 30):
        ratio = VOLUME_RATIO_MIDDAY
    else:
        ratio = VOLUME_RATIO_CLOSE

    # Relax volume filter on strong trends
    if not pd.isna(adx_val) and adx_val >= ADX_STRONG_TREND:
        ratio = max(VOLUME_RATIO_FLOOR, ratio - VOLUME_RATIO_STRONG_RELAX)

    return ratio


def _buy_support_score(cur: pd.Series, prev: pd.Series) -> int:
    """Calculate buy support indicator score (0~4).
    
    Score >= 2 is required for buy signal.
    """
    score = 0

    # 1) Stochastic K(10)/D(5): golden cross or K>D in non-overbought zone
    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if (k_p <= d_p and k_c > d_c) or (k_c > d_c and k_c <= STOCH_BUY_MAX):
            score += 1

    # 2) RSI(14)/Signal(6): signal upward cross or RSI in buy zone
    rsi_c = _num(cur, "RSI")
    sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI")
    sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        in_buy_zone = RSI_BUY_MIN <= rsi_c <= RSI_BUY_MAX
        if (rsi_p <= sig_p and rsi_c > sig_c) or (rsi_c > sig_c and in_buy_zone):
            score += 1

    # 3) Williams %R: rising and oversold recovery
    wr_c = _num(cur, "WILLIAMS_R")
    wr_p = _num(prev, "WILLIAMS_R")
    if not pd.isna(wr_c) and not pd.isna(wr_p):
        if wr_c > wr_p and wr_c >= WILLIAMS_BUY_FLOOR:
            score += 1

    # 4) MACD: golden cross or MACD > Signal with positive momentum
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if (macd_p <= msig_p and macd_c > msig_c) or (macd_c > msig_c and macd_c > 0):
            score += 1

    return score


def _sell_support_score(cur: pd.Series, prev: pd.Series) -> int:
    """Calculate sell support indicator score (0~4).
    
    Score >= 1 with MA5 dead cross confirms sell.
    Score >= 2 triggers auxiliary reversal sell.
    """
    score = 0

    # 1) Stochastic: dead cross from overbought
    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if k_p >= d_p and k_c < d_c and k_p >= STOCH_OVERBOUGHT:
            score += 1

    # 2) RSI: signal downward cross
    rsi_c = _num(cur, "RSI")
    sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI")
    sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        if rsi_p >= sig_p and rsi_c < sig_c:
            score += 1

    # 3) Williams %R: downward cross from %D
    wr_c = _num(cur, "WILLIAMS_R")
    wd_c = _num(cur, "WILLIAMS_D")
    wr_p = _num(prev, "WILLIAMS_R")
    wd_p = _num(prev, "WILLIAMS_D")
    if not any(pd.isna(v) for v in (wr_c, wd_c, wr_p, wd_p)):
        if wr_p >= wd_p and wr_c < wd_c:
            score += 1

    # 4) MACD: dead cross (downward)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if macd_p >= msig_p and macd_c < msig_c:
            score += 1

    return score


def check_buy_condition(frame: pd.DataFrame, now: pd.Timestamp | datetime) -> tuple[bool, str]:
    """Check if buy condition is met.
    
    Args:
        frame: DataFrame with all indicators
        now: Current timestamp
    
    Returns:
        (buy_signal: bool, reason: str)
    """
    if len(frame) < 3:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    # Core signal: MA5 crosses above BB middle
    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")

    if any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)):
        return False, "MISSING_INDICATOR"

    ma5_cross = prev_ma5 <= prev_bb and cur_ma5 > cur_bb
    if not ma5_cross:
        return False, "NO_MA5_BB_CROSS_UP"

    # Direction confirmation
    if cur_bb < prev_bb:
        return False, "BB_MIDDLE_FALLING"
    if cur_ma5 < prev_ma5:
        return False, "MA5_FALLING"

    # Bullish candle
    if float(cur["close"]) <= float(cur["open"]):
        return False, "NOT_BULLISH"

    # Overbought prevention 1: Stochastic
    stoch_k = _num(cur, "STOCH_K")
    if not pd.isna(stoch_k) and stoch_k >= STOCH_OVERBOUGHT:
        return False, f"OVERBOUGHT_STOCH_{stoch_k:.1f}"

    # Overbought prevention 2: Williams %R
    wr_val = _num(cur, "WILLIAMS_R")
    if not pd.isna(wr_val) and wr_val >= WILLIAMS_OVERBOUGHT_CEIL:
        return False, f"OVERBOUGHT_WR_{wr_val:.1f}"

    # Overbought prevention 3: BB proximity
    bb_up = _num(cur, "BB_UPPER")
    bb_low = _num(cur, "BB_LOWER")
    close_val = _num(cur, "close")
    if not any(pd.isna(v) for v in (bb_up, bb_low, close_val)) and bb_up > bb_low:
        bb_pos = (close_val - bb_low) / (bb_up - bb_low)
        if bb_pos >= BB_UPPER_PROXIMITY_MAX:
            return False, f"NEAR_BB_UPPER_{bb_pos:.2f}"

    adx_val = _num(cur, "ADX")

    # Volume filter
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        ratio = get_volume_ratio_threshold(now, adx_val)
        if vol < (vol_ma * ratio):
            return False, f"LOW_VOLUME_{(vol / vol_ma):.2f}_LT_{ratio:.2f}"

    # Trend strength filter
    if not pd.isna(adx_val) and adx_val < ADX_MIN_TREND:
        return False, f"WEAK_TREND_ADX_{adx_val:.1f}"

    # Support score
    support_score = _buy_support_score(cur, prev)
    if support_score < 2:
        return False, f"LOW_SCORE_{support_score}"

    return True, f"MA5_BB_UP_CROSS_SCORE_{support_score}"


def check_sell_condition(frame: pd.DataFrame) -> tuple[bool, str]:
    """Check if sell condition is met.
    
    Args:
        frame: DataFrame with all indicators
    
    Returns:
        (sell_signal: bool, reason: str)
    """
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")

    # MA5 dead cross (highest priority)
    ma5_dead = (
        not any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb))
        and prev_ma5 >= prev_bb
        and cur_ma5 < cur_bb
    )

    if ma5_dead:
        score = _sell_support_score(cur, prev)
        if score >= 1:
            return True, f"MA5_BB_DOWN_CROSS_CONFIRMED_{score}"
        return True, "MA5_BB_DOWN_CROSS"

    # Auxiliary reversal sell (supporting indicator reversals)
    score = _sell_support_score(cur, prev)
    if score >= 2:
        return True, f"AUX_REVERSAL_SCORE_{score}"

    return False, "NO_SELL_SIGNAL"
