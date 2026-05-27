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

    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if k_c > d_c:
            score += 1

    rsi_c = _num(cur, "RSI")
    rsi_p = _num(prev, "RSI")
    if not any(pd.isna(v) for v in (rsi_c, rsi_p)):
        if config.rsi_buy_min <= rsi_c < config.rsi_buy_max and rsi_c > rsi_p:
            score += 1

    wr_c = _num(cur, "WILLIAMS_R")
    wr_p = _num(prev, "WILLIAMS_R")
    if not pd.isna(wr_c) and not pd.isna(wr_p):
        if wr_c > wr_p and wr_c >= config.williams_buy_floor:
            score += 1

    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    hist_c = _num(cur, "MACD_HIST")
    hist_p = _num(prev, "MACD_HIST")
    if not any(pd.isna(v) for v in (macd_c, msig_c, hist_c)):
        hist_accel = (hist_c > 0) or (not pd.isna(hist_p) and hist_c < 0 and hist_c > hist_p)
        if macd_c > msig_c and hist_accel:
            score += 1

    vwap = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not pd.isna(vwap) and not pd.isna(close_v) and close_v > vwap:
        score += 1

    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    obv_breakout = False
    if "OBV" in frame.columns and len(frame) >= config.obv_breakout_lookback_bars + 1:
        obv_series = pd.to_numeric(frame["OBV"], errors="coerce")
        recent_obv_high = obv_series.iloc[-(config.obv_breakout_lookback_bars + 1):-1].max()
        if not pd.isna(obv_c) and not pd.isna(recent_obv_high):
            obv_breakout = obv_c > float(recent_obv_high)
    obv_uptrend = not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)) and (obv_c > obv_ma_c and obv_c > obv_p)
    if obv_breakout or obv_uptrend:
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
    confirmed_above = cross_info.get("confirmed_relation") == "above"
    ma5_golden_cross_now = prev_ma5 <= prev_bb and cur_ma5 > cur_bb
    ma5_bias_ok = cur_ma5 >= cur_bb and cur_ma5 >= prev_ma5
    upper_trigger = float(cross_info.get("upper_trigger", cur_bb))
    support_score = _buy_support_score(cur, prev, frame, config)
    cur_open = _num(cur, "open")
    cur_close = _num(cur, "close")

    # Require recent BB-middle hold when entering without a fresh live cross-up.
    recent_window = frame.tail(3)
    recent_close = pd.to_numeric(recent_window.get("close"), errors="coerce")
    recent_bb_mid = pd.to_numeric(recent_window.get("BB_MIDDLE"), errors="coerce")
    recent_above_count = int((recent_close > recent_bb_mid).sum())
    cur_close_above_bb = (not pd.isna(cur_close)) and cur_close > cur_bb

    # Late-chase guard: if the live cross has not fired, only allow entries
    # while price is still very close to BB middle.
    if not live_cross_up_signal and cur_bb > 0:
        bb_gap_pct = (live_price - cur_bb) / cur_bb
        if bb_gap_pct > min(config.ma5_bb_follow_chase_max_gap_pct, 0.005):
            return False, f"CHASE_BUY_BLOCK_BB_GAP_{bb_gap_pct*100:.2f}%_GT_0.50%"

    if not live_cross_up_signal:
        if recent_above_count < 2 or not cur_close_above_bb:
            return False, f"BB_MIDDLE_NOT_HELD_ABOVE_{recent_above_count}/3"

        # Allow early entry when broad support signals are already strong
        # even if explicit live-price cross confirmation has not fired yet.
        strong_signal_fallback = (
            support_score >= config.strong_trend_overbought_min_score
            and live_price > cur_bb
            and not pd.isna(cur_open)
            and not pd.isna(cur_close)
            and cur_close > cur_open
            and cur_bb >= prev_bb
            and ma5_bias_ok
        )
        fallback_ok = (confirmed_above and ma5_bias_ok and live_price >= upper_trigger) or strong_signal_fallback
        if not fallback_ok:
            return False, "NO_LIVE_PRICE_BB_CROSS_UP"

    if config.require_strict_buy_golden_cross and not ma5_golden_cross_now:
        return False, "NO_MA5_BB_GOLDEN_CROSS"

    if live_price <= cur_bb:
        return False, "LIVE_PRICE_NOT_ABOVE_BB_MIDDLE"
    if cur_bb < prev_bb:
        return False, "BB_MIDDLE_FALLING"
    if cur_ma5 < prev_ma5:
        return False, "MA5_FALLING"

    if not pd.isna(cur_open) and cur_open > 0 and live_price < (cur_open * 0.995):
        return False, "NOT_BULLISH"

    # Prevent late chase entries after MA5/BB cross when no fresh live-price cross-up signal exists.
    if (not live_cross_up_signal) and (not ma5_golden_cross_now) and cur_bb > 0:
        bb_gap_pct = (live_price - cur_bb) / cur_bb
        if bb_gap_pct > config.ma5_bb_follow_chase_max_gap_pct:
            return False, (
                f"CHASE_BUY_BLOCK_BB_GAP_{bb_gap_pct*100:.2f}%"
                f"_GT_{config.ma5_bb_follow_chase_max_gap_pct*100:.2f}%"
            )

    stoch_k = _num(cur, "STOCH_K")
    is_stoch_overbought = not pd.isna(stoch_k) and stoch_k >= config.stoch_overbought

    adx_val = _num(cur, "ADX")
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    score = _buy_support_score(cur, prev, frame, config)
    strong_entry_ok = (
        config.enable_strong_trend_overbought_bypass
        and score >= config.strong_trend_overbought_min_score
        and live_price > cur_bb
        and cur_bb >= prev_bb
        and cur_ma5 >= prev_ma5
        and (pd.isna(adx_val) or adx_val >= config.strong_trend_overbought_min_adx)
    )

    wr_val = _num(cur, "WILLIAMS_R")
    if not strong_entry_ok and not pd.isna(wr_val) and wr_val >= config.williams_overbought_ceil:
        return False, f"OVERBOUGHT_WR_{wr_val:.1f}"

    bb_up = _num(cur, "BB_UPPER")
    bb_low_v = _num(cur, "BB_LOWER")
    if not any(pd.isna(v) for v in (bb_up, bb_low_v)) and bb_up > bb_low_v:
        bb_pos = (live_price - bb_low_v) / (bb_up - bb_low_v)
        if bb_pos >= config.bb_upper_proximity_max:
            return False, f"NEAR_BB_UPPER_{bb_pos:.2f}"

    cur_close = _num(cur, "close")
    if not any(pd.isna(v) for v in (bb_up, bb_low_v, cur_close)) and cur_close > 0:
        bb_width_pct = (bb_up - bb_low_v) / cur_close
        if bb_width_pct < config.bb_squeeze_min_width_pct:
            return False, f"BB_SQUEEZE_{bb_width_pct*100:.2f}%_LT_{config.bb_squeeze_min_width_pct*100:.2f}%"

    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        ratio = volume_ratio_threshold_fn(now, adx_val)
        if (not strong_entry_ok) and vol < (vol_ma * ratio):
            return False, f"LOW_VOLUME_{(vol / vol_ma):.2f}_LT_{ratio:.2f}"

    if not pd.isna(adx_val) and adx_val < config.adx_min_trend:
        return False, f"WEAK_TREND_ADX_{adx_val:.1f}"

    if is_stoch_overbought:
        bypass_overbought = (
            config.enable_strong_trend_overbought_bypass
            and score >= config.strong_trend_overbought_min_score
        )
        if bypass_overbought:
            trigger = "LIVE_PRICE_BB_UP_CROSS" if live_cross_up_signal else "MA5_BB_GOLDEN_CROSS_ABOVE_BB"
            return True, f"{trigger}_OVERBOUGHT_BYPASS_SCORE_{score}"

    # --- 추가 필수 조건 ---
    vwap = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if (not strong_entry_ok) and (pd.isna(vwap) or pd.isna(close_v) or not (close_v > vwap)):
        return False, "NO_VWAP_BREAK"

    # 거래량 > 평균 거래량 1.5배
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        if (not strong_entry_ok) and vol < (vol_ma * 1.5):
            return False, f"LOW_VOLUME_1.5X_{(vol / vol_ma):.2f}"

    # RSI > 55
    rsi_c = _num(cur, "RSI")
    if pd.isna(rsi_c) or rsi_c <= 55:
        return False, f"RSI_TOO_LOW_{rsi_c:.2f}"

    # MACD Histogram 증가
    hist_c = _num(cur, "MACD_HIST")
    hist_p = _num(prev, "MACD_HIST")
    if pd.isna(hist_c) or pd.isna(hist_p) or not (hist_c > hist_p):
        return False, f"MACD_HIST_NOT_INCREASING_{hist_p:.2f}_TO_{hist_c:.2f}"

    # 체결강도 > 120 (값이 있으면 체크)
    exec_strength = _num(cur, "EXECUTION_STRENGTH") if "EXECUTION_STRENGTH" in cur else float('nan')
    if not pd.isna(exec_strength) and exec_strength <= 120:
        return False, f"EXEC_STRENGTH_TOO_LOW_{exec_strength:.2f}"

    score = support_score

    if is_stoch_overbought:
        bypass_overbought = (
            config.enable_strong_trend_overbought_bypass
            and score >= config.strong_trend_overbought_min_score
        )
        if not bypass_overbought:
            return False, f"OVERBOUGHT_STOCH_{stoch_k:.1f}"

    if config.enable_price_lead_bb_breakout:
        breakout_window = frame.iloc[:-1].tail(3)
        if breakout_window.empty:
            return False, "INSUFFICIENT_BREAKOUT_HISTORY"

        recent_high = pd.to_numeric(breakout_window["high"], errors="coerce").max()
        if pd.isna(recent_high) or recent_high <= 0:
            return False, "MISSING_BREAKOUT_HIGH"

        breakout_buffer = 1.0 + max(config.live_price_bb_buffer_pct, 0.0005)
        if live_price <= float(recent_high) * breakout_buffer:
            return False, f"NO_PRICE_LEAD_BREAKOUT_{live_price:.0f}_LE_{float(recent_high):.0f}"

        if score < config.price_lead_breakout_min_score:
            return False, f"LOW_BREAKOUT_SCORE_{score}"

        if not pd.isna(adx_val) and adx_val < config.price_lead_breakout_min_adx:
            return False, f"WEAK_BREAKOUT_ADX_{adx_val:.1f}"

        if not config.price_lead_breakout_allow_overbought and is_stoch_overbought:
            return False, f"OVERBOUGHT_STOCH_{stoch_k:.1f}"

    if score < 2:  # 3 → 2로 완화
        return False, f"LOW_SCORE_{score}"

    trigger = "LIVE_PRICE_BB_UP_CROSS" if live_cross_up_signal else "MA5_BB_GOLDEN_CROSS_ABOVE_BB"
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
