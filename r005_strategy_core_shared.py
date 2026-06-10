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


def _buy_support_score(
    cur: pd.Series,
    prev: pd.Series,
    frame: pd.DataFrame,
    config: R76StrategyConfig,
) -> int:
    score = 0

    rsi_c = _num(cur, "RSI")
    if not pd.isna(rsi_c) and rsi_c > 50:
        score += 1

    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c)) and macd_c > msig_c:
        score += 1

    adx_c = _num(cur, "ADX")
    if not pd.isna(adx_c) and adx_c >= config.adx_min_trend:
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

    rsi_c = _num(cur, "RSI")
    sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI")
    sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        if rsi_p >= sig_p and rsi_c < sig_c:
            score += 1

    wr_c = _num(cur, "WILLIAMS_R")
    wd_c = _num(cur, "WILLIAMS_D")
    wr_p = _num(prev, "WILLIAMS_R")
    wd_p = _num(prev, "WILLIAMS_D")
    if not any(pd.isna(v) for v in (wr_c, wd_c, wr_p, wd_p)):
        if wr_p >= wd_p and wr_c < wd_c:
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
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_ma5 = _num(cur, "MA_5")
    prev_ma5 = _num(prev, "MA_5")

    if any(pd.isna(v) for v in (cur_bb, prev_bb, cur_ma5, prev_ma5)):
        return False, "MISSING_INDICATOR"

    live_cross_up_signal = cross_info.get("signal") == "cross_up"

    # 1차 필터: 가격/거래량/추격매수
    prev_close = _num(prev, "close")
    bb_buffer = max(config.live_price_bb_buffer_pct, 0.0)
    bb_gate = cur_bb * (1.0 + bb_buffer)
    prev_close_gate = prev_close * (1.0 + bb_buffer) if not pd.isna(prev_close) else float("nan")

    if live_price <= bb_gate:
        return False, "LIVE_PRICE_NOT_ABOVE_BB_MIDDLE"
    if pd.isna(prev_close):
        return False, "PREV_CLOSE_MISSING"
    if live_price <= prev_close_gate:
        return False, "LIVE_NOT_ABOVE_PREV_CLOSE"

    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    adx_val = _num(cur, "ADX")
    if any(pd.isna(v) for v in (vol, vol_ma)) or vol_ma <= 0:
        return False, "MISSING_VOLUME_DATA"
    vol_ratio_req = volume_ratio_threshold_fn(now, adx_val)
    vol_ratio = vol / vol_ma
    if vol_ratio < vol_ratio_req:
        return False, f"LOW_VOLUME_RATIO_{vol_ratio:.2f}_LT_{vol_ratio_req:.2f}"

    if cur_bb > 0:
        bb_gap_pct = (live_price - cur_bb) / cur_bb
        if bb_gap_pct > config.ma5_bb_follow_chase_max_gap_pct:
            return False, (
                f"CHASE_BUY_BLOCK_BB_GAP_{bb_gap_pct*100:.2f}%"
                f"_GT_{config.ma5_bb_follow_chase_max_gap_pct*100:.2f}%"
            )

    # 2차 필터: 추세/정렬
    if cur_ma5 < prev_ma5:
        return False, "MA5_FALLING"
    if cur_bb < prev_bb:
        return False, "BB_MIDDLE_FALLING"

    ema5 = _num(cur, "EMA_5")
    ema20 = _num(cur, "EMA_20")
    if any(pd.isna(v) for v in (ema5, ema20)) or ema5 < ema20 or live_price <= ema20:
        return False, "EMA_ALIGNMENT_FAIL"

    # 3차 필터: 모멘텀/추세 보조
    if pd.isna(adx_val) or adx_val < config.adx_min_trend:
        return False, f"WEAK_TREND_ADX_{adx_val:.1f}"

    rsi_c = _num(cur, "RSI")
    if pd.isna(rsi_c):
        return False, "MISSING_RSI"
    if rsi_c >= config.rsi_buy_max:
        return False, f"RSI_OVERBOUGHT_{rsi_c:.1f}"

    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_confirm = not any(pd.isna(v) for v in (macd_c, msig_c)) and macd_c > msig_c
    rsi_confirm = rsi_c > 50
    if not (rsi_confirm or macd_confirm):
        return False, "MOMENTUM_CONFIRM_MISSING"

    trigger = "LIVE_PRICE_BB_UP_CROSS" if live_cross_up_signal else "LIVE_ABOVE_BB_PREV_CLOSE"
    score = _buy_support_score(cur, prev, frame, config)
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
