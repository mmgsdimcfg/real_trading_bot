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
from r003_define_config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD_MULTIPLIER,
    EARLY_NEAR_CROSS_MIN_TURNOVER_KRW,
    EARLY_NEAR_CROSS_MIN_VOL_MA,
    EARLY_NEAR_CROSS_MIN_VOLUME,
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
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    VOLUME_MA_PERIOD,
    WILLIAMS_D_PERIOD,
    WILLIAMS_R_PERIOD,
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
    return run_buy_condition_pipeline_comment(
        frame=frame,
        now=now,
        live_price=live_price,
        cross_info=cross_info,
        config=config,
        volume_ratio_threshold_fn=volume_ratio_threshold_fn,
    )


def buy_1st_live_price_above_bb_mid_within_gap_comment(
    live_price: float,
    cur_bb: float,
    bb_buffer: float,
    max_gap_pct: float,
) -> tuple[bool, str]:
    bb_gate = cur_bb * (1.0 + max(bb_buffer, 0.0))
    if live_price <= bb_gate:
        return False, "LIVE_PRICE_NOT_ABOVE_BB_MIDDLE"

    if cur_bb > 0:
        bb_gap_pct = (live_price - cur_bb) / cur_bb
        if bb_gap_pct > max_gap_pct:
            return False, (
                f"CHASE_BUY_BLOCK_BB_GAP_{bb_gap_pct*100:.2f}%"
                f"_GT_{max_gap_pct*100:.2f}%"
            )

    return True, "BUY_1ST_PASS"


def buy_2nd_prev_bar_low_below_current_bb_mid_comment(prev: pd.Series, cur_bb: float) -> tuple[bool, str]:
    prev_low = _num(prev, "low")
    if pd.isna(prev_low) or pd.isna(cur_bb):
        return False, "PREV_LOW_OR_BB_MISSING"
    if prev_low >= cur_bb:
        return False, f"PREV_LOW_NOT_BELOW_BB_MIDDLE_{prev_low:.1f}_GTE_{cur_bb:.1f}"
    return True, "BUY_2ND_PASS"


def buy_3rd_primary_gate_confirm_comment(first_ok: bool, second_ok: bool) -> tuple[bool, str]:
    if first_ok and second_ok:
        return True, "BUY_3RD_PRIMARY_GATE_PASS"
    return False, "BUY_3RD_PRIMARY_GATE_FAIL"


def buy_4th_macd_hist_positive_comment(cur: pd.Series) -> tuple[bool, str]:
    macd_hist = _num(cur, "MACD_HIST")
    if pd.isna(macd_hist) or macd_hist <= 0:
        return False, "MACD_HIST_NOT_POSITIVE"
    return True, "BUY_4TH_PASS"


def buy_5th_stoch_k_over_d_comment(cur: pd.Series) -> tuple[bool, str]:
    stoch_k = _num(cur, "STOCH_K")
    stoch_d = _num(cur, "STOCH_D")
    if any(pd.isna(v) for v in (stoch_k, stoch_d)) or stoch_k < stoch_d - 0.5:
        return False, "STOCH_K_NOT_ABOVE_D"
    return True, "BUY_5TH_PASS"


def buy_6th_stoch_overheat_guard_comment(
    cur: pd.Series,
    config: R76StrategyConfig,
    confirmed_cross: bool,
) -> tuple[bool, str]:
    stoch_k = _num(cur, "STOCH_K")
    if pd.isna(stoch_k):
        return False, "STOCH_K_NOT_ABOVE_D"
    if stoch_k >= config.stoch_overbought:
        adx_val = _num(cur, "ADX")
        bypass = (
            config.enable_strong_trend_overbought_bypass
            and confirmed_cross
            and not pd.isna(adx_val)
            and adx_val >= config.strong_trend_overbought_min_adx
        )
        if not bypass:
            return False, f"STOCH_K_OVERHEATED_{stoch_k:.1f}"
    return True, "BUY_6TH_PASS"


def buy_7th_rsi_floor_comment(cur: pd.Series) -> tuple[bool, str]:
    rsi_c = _num(cur, "RSI")
    if pd.isna(rsi_c) or rsi_c <= 25:
        return False, f"RSI_TOO_LOW_{rsi_c:.1f}"
    return True, "BUY_7TH_PASS"


def buy_8th_di_bullish_comment(cur: pd.Series) -> tuple[bool, str]:
    di_plus = _num(cur, "DI_PLUS")
    di_minus = _num(cur, "DI_MINUS")
    if any(pd.isna(v) for v in (di_plus, di_minus)) or di_plus <= di_minus:
        return False, f"DI_NOT_BULLISH_+DI_{di_plus:.1f}_-DI_{di_minus:.1f}"
    return True, "BUY_8TH_PASS"


def buy_9th_volume_ratio_hard_floor_comment(
    cur: pd.Series,
    now: pd.Timestamp,
    volume_ratio_threshold_fn: Callable[[pd.Timestamp, float], float],
) -> tuple[bool, str, bool]:
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    adx_val = _num(cur, "ADX")

    volume_soft_fail = False
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        vol_ratio = vol / vol_ma
        vol_ratio_req = volume_ratio_threshold_fn(now, adx_val)
        volume_soft_fail = vol_ratio < vol_ratio_req

        hard_floor = max(0.10, vol_ratio_req * 0.40)
        if vol_ratio < hard_floor:
            return False, f"LOW_VOLUME_RATIO_{vol_ratio:.4f}_LT_{hard_floor:.4f}", volume_soft_fail

    return True, "BUY_9TH_PASS", volume_soft_fail


def buy_10th_prev_close_and_volume_soft_guard_comment(
    prev_close_soft_fail: bool,
    volume_soft_fail: bool,
) -> tuple[bool, str]:
    if prev_close_soft_fail and volume_soft_fail:
        return False, "LIVE_NOT_ABOVE_PREV_CLOSE_AND_LOW_VOLUME"
    return True, "BUY_10TH_PASS"


def run_buy_condition_pipeline_comment(
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
    bb_buffer = max(config.live_price_bb_buffer_pct, 0.0)

    first_ok, first_reason = buy_1st_live_price_above_bb_mid_within_gap_comment(
        live_price=live_price,
        cur_bb=cur_bb,
        bb_buffer=bb_buffer,
        max_gap_pct=config.ma5_bb_follow_chase_max_gap_pct,
    )
    if not first_ok:
        return False, first_reason

    second_ok, second_reason = buy_2nd_prev_bar_low_below_current_bb_mid_comment(prev=prev, cur_bb=cur_bb)
    if not second_ok:
        return False, second_reason

    third_ok, third_reason = buy_3rd_primary_gate_confirm_comment(first_ok=first_ok, second_ok=second_ok)
    if not third_ok:
        return False, third_reason

    close_val_cur = _num(cur, "close")
    prev_close = _num(prev, "close")
    confirmed_cross = (
        (not any(pd.isna(v) for v in (prev_ma5, cur_ma5)) and prev_ma5 <= prev_bb and cur_ma5 > cur_bb)
        or (not any(pd.isna(v) for v in (prev_close, close_val_cur, prev_bb, cur_bb))
            and prev_close <= prev_bb and close_val_cur > cur_bb)
    )

    # Require a fresh cross: either the live price polling detected an up-cross,
    # or the most recent bar shows MA5/close crossing above BB middle from below.
    # Without this, stocks floating above BB since market open trigger chase entries.
    if not live_cross_up_signal and not confirmed_cross:
        return False, "NO_FRESH_CROSS_GATE"

    indicator_rules: list[Callable[[], tuple[bool, str]]] = [
        lambda: buy_4th_macd_hist_positive_comment(cur),
        lambda: buy_5th_stoch_k_over_d_comment(cur),
        lambda: buy_6th_stoch_overheat_guard_comment(cur, config, confirmed_cross),
        lambda: buy_7th_rsi_floor_comment(cur),
        lambda: buy_8th_di_bullish_comment(cur),
    ]
    for rule in indicator_rules:
        ok, reason = rule()
        if not ok:
            return False, reason

    # Volume and previous-close soft guard
    prev_close_soft_fail = False
    if not pd.isna(prev_close):
        prev_close_gate = prev_close * (1.0 + bb_buffer)
        prev_close_soft_fail = live_price <= prev_close_gate

    ninth_ok, ninth_reason, volume_soft_fail = buy_9th_volume_ratio_hard_floor_comment(
        cur=cur,
        now=now,
        volume_ratio_threshold_fn=volume_ratio_threshold_fn,
    )
    if not ninth_ok:
        return False, ninth_reason

    tenth_ok, tenth_reason = buy_10th_prev_close_and_volume_soft_guard_comment(
        prev_close_soft_fail=prev_close_soft_fail,
        volume_soft_fail=volume_soft_fail,
    )
    if not tenth_ok:
        return False, tenth_reason

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


# ---------------------------------------------------------------------------
# Shared indicator calculation (used by r006 and r007)
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
    out["EMA_5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["EMA_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["VOL_MA20"] = out["volume"].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()

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

