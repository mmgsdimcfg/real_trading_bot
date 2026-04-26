# -*- coding: utf-8 -*-
"""S73 simulation script based on the R73 MA5/BB cross strategy.

Replays 3-minute CSV data from data/YYYYMMDD and applies the same strategy as
r73_real_trade_with_today_code.py (MA5/BB middle cross + multi-indicator filter).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas. "
        "On Ubuntu, run 'python3 -m pip install -r requirements.txt' from repository root."
    ) from exc

from define_01_config import (
    DATA_DIR_NAME,
    INITIAL_CAPITAL,
    NXT_ELIGIBLE_CODES_FALLBACK,
    PICKS_FILENAME,
    SIM_WARMUP_PRIOR_MAX_DAYS,
    SIM_WARMUP_TAIL_BARS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TODAY_CODE_FILE = SCRIPT_DIR / "define_today_code.txt"

# ---------------------------------------------------------------------------
# 지표 파라미터 (r73 최신 기준)
# ---------------------------------------------------------------------------
BB_PERIOD = 20
BB_STD_MULTIPLIER = 2.0
MA_PERIOD = 5
RSI_PERIOD = 14
RSI_SIGNAL_PERIOD = 6
STOCH_K_PERIOD = 10
STOCH_D_PERIOD = 5
WILLIAMS_R_PERIOD = 10
WILLIAMS_D_PERIOD = 9
VOLUME_MA_PERIOD = 20
OBV_MA_PERIOD = 10
MACD_FAST = 5
MACD_SLOW = 12
MACD_SIGNAL_PERIOD = 4
ADX_PERIOD = 7
ADX_MIN_TREND = 20.0
ADX_STRONG_TREND = 40.0

# ---------------------------------------------------------------------------
# 매매 파라미터 (r73 최신 기준)
# ---------------------------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000
TAKE_PROFIT_PERCENT = 0.035
STOP_LOSS_PERCENT = -0.015
TRAILING_STOP_FROM_PEAK = 0.012
AUX_SELL_MIN_PNL_SCORE2 = 0.010
AUX_SELL_MIN_PNL_SCORE3 = 0.005
AUX_SELL_MIN_PNL_SCORE4 = 0.000
MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1
MA5_BB_DOWN_CROSS_MIN_PNL = 0.010
TECH_SELL_MIN_HOLD_SECONDS = 300
ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
ENABLE_TP_EXTENSION_TRAILING = True
TP_EXTENSION_TRAIL_FROM_PEAK = 0.010
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080
NEAR_CROSS_ARM_GAP_MAX = 0.0015
NEAR_CROSS_ARM_MA_RISE_MIN = 0.0006
NEAR_CROSS_EARLY_GAP_MAX = 0.0006
NEAR_CROSS_EARLY_MA_RISE_MIN = 0.0010
NEAR_CROSS_ARM_EXPIRE_BARS = 2
REQUIRE_STRICT_BUY_GOLDEN_CROSS = True
ENABLE_NEAR_CROSS_ARM = True
ENABLE_EARLY_NEAR_CROSS_ENTRY = True
EARLY_NEAR_CROSS_ALLOWED_START = dt_time(9, 0)
EARLY_NEAR_CROSS_ALLOWED_END = dt_time(11, 30)
EARLY_NEAR_CROSS_ALLOW_NXT = False
EARLY_NEAR_CROSS_MIN_VOLUME = 800
EARLY_NEAR_CROSS_MIN_VOL_MA = 500
EARLY_NEAR_CROSS_MIN_TURNOVER_KRW = 5_000_000

MIN_BARS_REQUIRED = 3
SAME_DAY_MIN_BARS = 10
ALLOW_REBUY_SAME_CODE = False
TRADE_COOLDOWN_MINUTES = 3

# ---------------------------------------------------------------------------
# 세션 / 시간 상수
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

# 보조지표 임계값
STOCH_OVERBOUGHT = 80.0
STOCH_BUY_MAX = 72.0
RSI_BUY_MIN = 45.0
RSI_BUY_MAX = 72.0
WILLIAMS_BUY_FLOOR = -70.0
WILLIAMS_OVERBOUGHT_CEIL = -20.0
BB_UPPER_PROXIMITY_MAX = 0.85

VOLUME_RATIO_OPEN = 0.80
VOLUME_RATIO_MIDDAY = 0.60
VOLUME_RATIO_CLOSE = 0.70
VOLUME_RATIO_NXT = 0.55
VOLUME_RATIO_STRONG_RELAX = 0.10
VOLUME_RATIO_FLOOR = 0.50

SAMPLE_CODE_MAP = {
    "001440": "대한전선",
    "450080": "에코프로머티",
    "032500": "케이엠더블유",
}

log_dir = SCRIPT_DIR / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

LAST_SIM_STATS: dict[str, float | int | str] = {}


def log(msg: str) -> None:
    logger.info(msg)


def raw(msg: str) -> None:
    print(msg)


def _load_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=encoding) as file_obj:
                return [line.strip() for line in file_obj if line.strip()]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("define_today_code", b"", 0, 1, "Unable to decode file")


def load_code_name_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for line in _load_text_lines(path):
        if line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        code = parts[0].zfill(6)
        name = parts[1] if len(parts) >= 2 and parts[1] else code
        result[code] = name
    return result


def load_picks(picks_file: Path) -> dict[str, str] | None:
    if not picks_file.exists():
        return None
    return load_code_name_map(picks_file)


def read_ohlc_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        log(f"CSV load failed: {csv_path} | {exc}")
        return None

    time_col = None
    for col in ("timestamp", "datetime", "Time", "time", "date", "Date"):
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.iloc[:, 1:]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return None

    df = df[~df.index.isna()].sort_index()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df if not df.empty else None


def find_prior_trading_day_csv(data_root: Path, as_of: date, code: str) -> Path | None:
    cursor = as_of - timedelta(days=1)
    for _ in range(SIM_WARMUP_PRIOR_MAX_DAYS):
        candidate = data_root / cursor.strftime("%Y%m%d") / f"{code}.csv"
        if candidate.is_file():
            return candidate
        cursor -= timedelta(days=1)
    return None


def build_simulation_frame(data_root: Path, simulation_date_str: str, code: str) -> pd.DataFrame | None:
    target_date = datetime.strptime(simulation_date_str, "%Y%m%d").date()
    today_path = data_root / simulation_date_str / f"{code}.csv"
    if not today_path.is_file():
        return None

    today = read_ohlc_csv(today_path)
    if today is None or today.empty:
        return None

    merged = today
    prior_path = find_prior_trading_day_csv(data_root, target_date, code)
    if prior_path is not None:
        prior = read_ohlc_csv(prior_path)
        if prior is not None and not prior.empty:
            merged = pd.concat([prior.tail(SIM_WARMUP_TAIL_BARS), today])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    return calculate_indicators(merged)


def is_regular_session(ts: pd.Timestamp) -> bool:
    return REGULAR_START <= ts.time() <= REGULAR_END


def is_nxt_session(ts: pd.Timestamp) -> bool:
    current_time = ts.time()
    return (MORNING_NXT_START <= current_time <= MORNING_NXT_END) or (AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END)


def classify_buy_session(ts: pd.Timestamp) -> str:
    current_time = ts.time()
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return "morning_nxt"
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return "afternoon_nxt"
    return "regular"


def can_trade_code_now(ts: pd.Timestamp, nxt_tradeable: bool) -> bool:
    current_time = ts.time()
    if REGULAR_START <= current_time <= REGULAR_END:
        return True
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return nxt_tradeable
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return nxt_tradeable
    return False


def is_new_entry_allowed(ts: pd.Timestamp, nxt_tradeable: bool) -> bool:
    """NXT 가능 여부에 따라 매수 허용 시간 결정.
    - 정규장(09:00-15:20): 모든 종목
    - NXT 시간(08:00-08:50, 15:30-19:59): NXT 가능 종목만
    """
    if is_regular_session(ts):
        return ts.time() < REGULAR_NEW_ENTRY_CUTOFF
    if is_nxt_session(ts) and nxt_tradeable:
        return ts.time() < AFTERNOON_NXT_NEW_ENTRY_CUTOFF or ts.time() <= MORNING_NXT_END
    return False


def get_volume_ratio_threshold(ts: pd.Timestamp, adx_val: float) -> float:
    current_time = ts.time()

    if is_nxt_session(ts):
        ratio = VOLUME_RATIO_NXT
    elif current_time < dt_time(10, 0):
        ratio = VOLUME_RATIO_OPEN
    elif current_time < dt_time(14, 30):
        ratio = VOLUME_RATIO_MIDDAY
    else:
        ratio = VOLUME_RATIO_CLOSE

    if not pd.isna(adx_val) and adx_val >= ADX_STRONG_TREND:
        ratio = max(VOLUME_RATIO_FLOOR, ratio - VOLUME_RATIO_STRONG_RELAX)

    return ratio


def is_early_near_cross_allowed(ts: pd.Timestamp, nxt_tradeable: bool) -> bool:
    current_time = ts.time()
    if is_regular_session(ts):
        return EARLY_NEAR_CROSS_ALLOWED_START <= current_time <= EARLY_NEAR_CROSS_ALLOWED_END
    if EARLY_NEAR_CROSS_ALLOW_NXT and is_nxt_session(ts) and nxt_tradeable:
        return True
    return False


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    # 이동평균 & 볼린저밴드
    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
    out["VOL_MA20"] = out["volume"].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()
    out["BB_MIDDLE"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).mean()
    out["BB_STD"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).std()
    out["BB_UPPER"] = out["BB_MIDDLE"] + out["BB_STD"] * BB_STD_MULTIPLIER
    out["BB_LOWER"] = out["BB_MIDDLE"] - out["BB_STD"] * BB_STD_MULTIPLIER

    # RSI - Wilder's smoothing (align with real trading script)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out["RSI"] = 100 - (100 / (1 + rs))
    out.loc[loss == 0, "RSI"] = 100.0
    out["RSI_SIGNAL"] = out["RSI"].rolling(window=RSI_SIGNAL_PERIOD, min_periods=1).mean()

    # Stochastic Fast K(10)/D(5)
    low_n = out["low"].rolling(window=STOCH_K_PERIOD, min_periods=1).min()
    high_n = out["high"].rolling(window=STOCH_K_PERIOD, min_periods=1).max()
    denom = (high_n - low_n).replace(0, float("nan"))
    out["STOCH_K"] = 100.0 * (out["close"] - low_n) / denom
    out["STOCH_D"] = out["STOCH_K"].rolling(window=STOCH_D_PERIOD, min_periods=1).mean()

    # Williams %R(10) & %D(9)
    high_w = out["high"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).max()
    low_w = out["low"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).min()
    wr_denom = (high_w - low_w).replace(0, float("nan"))
    out["WILLIAMS_R"] = -100.0 * (high_w - out["close"]) / wr_denom
    out["WILLIAMS_D"] = out["WILLIAMS_R"].rolling(window=WILLIAMS_D_PERIOD, min_periods=1).mean()

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

    # VWAP / OBV (r73 매수 스코어 반영)
    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))
    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


def _num(candle: pd.Series, key: str) -> float:
    value = candle.get(key)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def _buy_support_score(cur: pd.Series, prev: pd.Series) -> int:
    """매수 보조지표 스코어 (0~6)."""
    score = 0

    # ① Stochastic K(10)/D(5): 상향 돌파 또는 과열 아닌 구간에서 K>D
    k_c = _num(cur, "STOCH_K"); d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K"); d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if (k_p <= d_p and k_c > d_c) or (k_c > d_c and k_c <= STOCH_BUY_MAX):
            score += 1

    # ② RSI(14)/Signal(6): RSI가 Signal 상향 돌파 또는 매수 구간 유지
    rsi_c = _num(cur, "RSI"); sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI"); sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        in_buy_zone = RSI_BUY_MIN <= rsi_c <= RSI_BUY_MAX
        if (rsi_p <= sig_p and rsi_c > sig_c) or (rsi_c > sig_c and in_buy_zone):
            score += 1

    # ③ Williams %R(10): 상승 중이며 과매도 회복
    wr_c = _num(cur, "WILLIAMS_R"); wr_p = _num(prev, "WILLIAMS_R")
    if not pd.isna(wr_c) and not pd.isna(wr_p):
        if wr_c > wr_p and wr_c >= WILLIAMS_BUY_FLOOR:
            score += 1

    # ④ MACD
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if (macd_p <= msig_p and macd_c > msig_c) or (macd_c > msig_c and macd_c > 0):
            score += 1

    # ⑤ VWAP
    vwap = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not pd.isna(vwap) and not pd.isna(close_v) and close_v > vwap:
        score += 1

    # ⑥ OBV
    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_c > obv_ma_c or obv_c > obv_p:
            score += 1

    return score


def _sell_support_score(cur: pd.Series, prev: pd.Series) -> int:
    """매도 보조지표 스코어 (0~5)."""
    score = 0

    k_c = _num(cur, "STOCH_K"); d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K"); d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if k_p >= d_p and k_c < d_c and k_p >= STOCH_OVERBOUGHT:
            score += 1

    rsi_c = _num(cur, "RSI"); sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI"); sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        if rsi_p >= sig_p and rsi_c < sig_c:
            score += 1

    wr_c = _num(cur, "WILLIAMS_R"); wd_c = _num(cur, "WILLIAMS_D")
    wr_p = _num(prev, "WILLIAMS_R"); wd_p = _num(prev, "WILLIAMS_D")
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


def _near_cross_momentum_flags(cur: pd.Series, prev: pd.Series) -> dict[str, float | bool]:
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


def _is_box_range_hold_zone(frame: pd.DataFrame) -> tuple[bool, str]:
    if len(frame) < BOX_RANGE_HOLD_LOOKBACK_BARS:
        return False, "INSUFFICIENT_BOX_BARS"

    recent = frame.tail(BOX_RANGE_HOLD_LOOKBACK_BARS)
    high_v = pd.to_numeric(recent["high"], errors="coerce").max()
    low_v = pd.to_numeric(recent["low"], errors="coerce").min()
    close_v = _num(recent.iloc[-1], "close")
    bb_up = _num(recent.iloc[-1], "BB_UPPER")
    bb_low = _num(recent.iloc[-1], "BB_LOWER")

    if any(pd.isna(v) for v in (high_v, low_v, close_v, bb_up, bb_low)) or close_v <= 0:
        return False, "BOX_DATA_NAN"

    range_pct = (float(high_v) - float(low_v)) / float(close_v)
    bb_width_pct = (float(bb_up) - float(bb_low)) / float(close_v)
    is_box = range_pct <= BOX_RANGE_HOLD_MAX_RANGE_PCT and bb_width_pct <= BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT

    return is_box, f"RANGE_{range_pct*100:.2f}%_BBW_{bb_width_pct*100:.2f}%"


def check_buy_condition(frame: pd.DataFrame, now: pd.Timestamp) -> tuple[bool, str]:
    """MA5가 BB 중심선을 상향 돌파하는 시점의 진입 조건."""
    if len(frame) < 3:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    prev_ma5 = _num(prev, "MA_5"); cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE"); cur_bb = _num(cur, "BB_MIDDLE")

    if any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)):
        return False, "MISSING_INDICATOR"

    # ① MA5가 BB 중심선을 상향 돌파
    ma5_cross = prev_ma5 <= prev_bb and cur_ma5 > cur_bb
    if not ma5_cross:
        return False, "NO_MA5_BB_CROSS_UP"

    # ② MA5/BB 중심선 방향 확인
    if cur_bb < prev_bb:
        return False, "BB_MID_FALLING"
    if cur_ma5 < prev_ma5:
        return False, "MA5_FALLING"

    # ③ 양봉
    if float(cur["close"]) <= float(cur["open"]):
        return False, "NOT_BULLISH"

    stoch_k = _num(cur, "STOCH_K")
    if not pd.isna(stoch_k) and stoch_k >= STOCH_OVERBOUGHT:
        return False, f"OVERBOUGHT_STOCH_{stoch_k:.1f}"

    wr_val = _num(cur, "WILLIAMS_R")
    if not pd.isna(wr_val) and wr_val >= WILLIAMS_OVERBOUGHT_CEIL:
        return False, f"OVERBOUGHT_WR_{wr_val:.1f}"

    bb_up = _num(cur, "BB_UPPER")
    bb_low = _num(cur, "BB_LOWER")
    close_val = _num(cur, "close")
    if not any(pd.isna(v) for v in (bb_up, bb_low, close_val)) and bb_up > bb_low:
        bb_pos = (close_val - bb_low) / (bb_up - bb_low)
        if bb_pos >= BB_UPPER_PROXIMITY_MAX:
            return False, f"NEAR_BB_UPPER_{bb_pos:.2f}"

    adx_val = _num(cur, "ADX")

    # ④ 거래량 필터
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        ratio = get_volume_ratio_threshold(now, adx_val)
        if vol < (vol_ma * ratio):
            return False, f"LOW_VOLUME_{(vol / vol_ma):.2f}_LT_{ratio:.2f}"

    if not pd.isna(adx_val) and adx_val < ADX_MIN_TREND:
        return False, f"WEAK_TREND_ADX_{adx_val:.1f}"

    # ⑤ 보조지표 스코어 ≥ 3
    support_score = _buy_support_score(cur, prev)
    if support_score < 3:
        return False, f"LOW_SCORE_{support_score}"

    return True, f"MA5_BB_UP_CROSS_SCORE_{support_score}"


def check_sell_condition(frame: pd.DataFrame, pnl_pct: float, held_seconds: float | None = None) -> tuple[bool, str]:
    """r73 매도 조건(데드크로스+수익률 게이트+보조지표+박스권 홀드)."""
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    if held_seconds is not None and held_seconds < TECH_SELL_MIN_HOLD_SECONDS:
        return False, f"TECH_SELL_MIN_HOLD_{held_seconds:.0f}s_LT_{TECH_SELL_MIN_HOLD_SECONDS}s"

    if ENABLE_BOX_RANGE_HOLD_TECH_SELL and STOP_LOSS_PERCENT < pnl_pct < TAKE_PROFIT_PERCENT:
        is_box, box_info = _is_box_range_hold_zone(frame)
        if is_box:
            return False, f"BOX_RANGE_HOLD_{box_info}"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    prev_ma5 = _num(prev, "MA_5"); cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE"); cur_bb = _num(cur, "BB_MIDDLE")

    # ① MA5 dead cross
    ma5_dead = (
        not any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb))
        and prev_ma5 >= prev_bb
        and cur_ma5 < cur_bb
    )

    if ma5_dead:
        if pnl_pct < MA5_BB_DOWN_CROSS_MIN_PNL:
            return False, (
                f"MA5_BB_DOWN_CROSS_BLOCKED_PNL_{pnl_pct*100:.2f}%"
                f"_LT_{MA5_BB_DOWN_CROSS_MIN_PNL*100:.2f}%"
            )

        score = _sell_support_score(cur, prev)
        if pnl_pct <= MA5_BB_DOWN_CROSS_IMMEDIATE_PNL or score >= MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE:
            if score >= 1:
                return True, f"MA5_BB_DOWN_CROSS_CONFIRMED_{score}"
            return True, "MA5_BB_DOWN_CROSS"

        return False, f"MA5_BB_DOWN_CROSS_ARMED_{score}"

    if len(frame) >= 3:
        prev2 = frame.iloc[-3]
        prev2_ma5 = _num(prev2, "MA_5")
        prev2_bb = _num(prev2, "BB_MIDDLE")
        prev_cross = (
            not any(pd.isna(v) for v in (prev2_ma5, prev_ma5, prev2_bb, prev_bb))
            and prev2_ma5 >= prev2_bb
            and prev_ma5 < prev_bb
        )
        cur_below = not any(pd.isna(v) for v in (cur_ma5, cur_bb)) and cur_ma5 < cur_bb
        if prev_cross and cur_below:
            if pnl_pct < MA5_BB_DOWN_CROSS_MIN_PNL:
                return False, (
                    f"MA5_BB_DOWN_CROSS_NEXT_BAR_BLOCKED_PNL_{pnl_pct*100:.2f}%"
                    f"_LT_{MA5_BB_DOWN_CROSS_MIN_PNL*100:.2f}%"
                )

            score = _sell_support_score(cur, prev)
            if score >= MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE:
                return True, f"MA5_BB_DOWN_CROSS_NEXT_BAR_{score}"
            return True, "MA5_BB_DOWN_CROSS_NEXT_BAR"

    score = _sell_support_score(cur, prev)
    min_pnl_req = None
    if score >= 4:
        min_pnl_req = AUX_SELL_MIN_PNL_SCORE4
    elif score == 3:
        min_pnl_req = AUX_SELL_MIN_PNL_SCORE3
    elif score == 2:
        min_pnl_req = AUX_SELL_MIN_PNL_SCORE2

    if min_pnl_req is not None:
        if pnl_pct >= min_pnl_req:
            return True, f"AUX_REVERSAL_SCORE_{score}"
        return False, f"AUX_BLOCKED_SCORE_{score}_PNL_{pnl_pct*100:.2f}%_LT_{min_pnl_req*100:.2f}%"

    return False, "NO_SELL_SIGNAL"


class TradeRecord:
    def __init__(
        self,
        code: str,
        name: str,
        action: str,
        bar_time: pd.Timestamp,
        price: float,
        qty: int,
        reason: str,
        session: str,
    ):
        self.code = code
        self.name = name
        self.action = action
        self.bar_time = bar_time
        self.price = price
        self.qty = qty
        self.reason = reason
        self.session = session
        self.pnl_pct: float | None = None
        self.pnl_krw: float | None = None

    def __repr__(self) -> str:
        pnl_str = f" | pnl={self.pnl_pct:.2f}%" if self.pnl_pct is not None else ""
        return (
            f"{self.bar_time:%H:%M} | {self.action:4s} | {self.code}({self.name}) | "
            f"qty={self.qty} | price={self.price:,.0f}{pnl_str} | {self.reason}"
        )


class SimPosition:
    def __init__(self, code: str, name: str, buy_price: float, quantity: int, buy_time: pd.Timestamp, buy_session: str):
        self.code = code
        self.name = name
        self.buy_price = buy_price
        self.quantity = quantity
        self.buy_time = buy_time
        self.buy_session = buy_session
        self.highest_price = buy_price


class Simulator:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, SimPosition] = {}
        self.trade_log: list[TradeRecord] = []
        self.cooldown_until: dict[str, pd.Timestamp] = {}
        self.completed_codes: set[str] = set()

    def in_cooldown(self, code: str, now: pd.Timestamp) -> bool:
        until = self.cooldown_until.get(code)
        return until is not None and now < until

    def set_cooldown(self, code: str, now: pd.Timestamp) -> None:
        self.cooldown_until[code] = now + pd.Timedelta(minutes=TRADE_COOLDOWN_MINUTES)

    def buy(self, code: str, name: str, price: float, now: pd.Timestamp, session: str, reason: str) -> bool:
        if self.in_cooldown(code, now) or price <= 0 or code in self.positions:
            return False
        if (not ALLOW_REBUY_SAME_CODE) and code in self.completed_codes:
            return False

        qty = int(MAX_ORDER_AMOUNT_KRW / price)
        if qty <= 0:
            return False

        affordable_qty = int(self.cash / price)
        qty = min(qty, affordable_qty)
        if qty <= 0:
            return False

        self.cash -= price * qty
        self.positions[code] = SimPosition(code, name, price, qty, now, session)
        self.set_cooldown(code, now)

        rec = TradeRecord(code, name, "BUY", now, price, qty, reason, session)
        self.trade_log.append(rec)
        log(f"  {rec}")
        return True

    def sell(self, code: str, price: float, now: pd.Timestamp, reason: str, session: str) -> bool:
        pos = self.positions.get(code)
        if pos is None or pos.quantity <= 0 or price <= 0:
            return False

        proceeds = price * pos.quantity
        self.cash += proceeds

        rec = TradeRecord(code, pos.name, "SELL", now, price, pos.quantity, reason, session)
        rec.pnl_pct = (price / pos.buy_price - 1.0) * 100.0
        rec.pnl_krw = (price - pos.buy_price) * pos.quantity
        self.trade_log.append(rec)
        log(f"  {rec}")

        del self.positions[code]
        self.completed_codes.add(code)
        self.set_cooldown(code, now)
        return True

    def portfolio_value(self, last_prices: dict[str, float]) -> float:
        total = self.cash
        for code, pos in self.positions.items():
            total += last_prices.get(code, pos.buy_price) * pos.quantity
        return total


def get_latest_price_up_to(frame: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    available = frame[frame.index <= ts]
    if available.empty:
        return None
    return float(available.iloc[-1]["close"])


def run_scheduled_liquidations(
    sim: Simulator,
    frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    ts: pd.Timestamp,
    state: dict,
) -> None:
    current_date = ts.date()
    if state.get("date") != current_date:
        state["date"] = current_date
        state["done_regular"] = False
        state["done_nxt"] = False

    # 정규장 15:20 강제 청산 (당일 전량청산)
    if not state["done_regular"] and ts.time() >= REGULAR_FORCE_EXIT:
        state["done_regular"] = True
        for code in list(sim.positions.keys()):
            pos = sim.positions.get(code)
            if pos is None:
                continue
            price = get_latest_price_up_to(frames[code], ts)
            if price is None:
                continue
            sim.sell(code, price, ts, "EOD_FLAT_1520_ALL", "regular")

    # NXT 19:59 강제 청산 (잔여 전량청산)
    if not state["done_nxt"] and ts.time() >= AFTERNOON_NXT_FORCE_EXIT:
        state["done_nxt"] = True
        for code in list(sim.positions.keys()):
            pos = sim.positions.get(code)
            if pos is None:
                continue
            price = get_latest_price_up_to(frames[code], ts)
            if price is None:
                continue
            sim.sell(code, price, ts, "EOD_FLAT_1959_ALL", "afternoon_nxt")


def simulate_date(
    date_str: str,
    data_root: Path,
    codes: list[str] | None = None,
    names: dict[str, str] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> int:
    global LAST_SIM_STATS
    data_dir = data_root / date_str
    log(f"\n{'=' * 60}")
    log(f"Simulation date : {date_str} [R73 MA5-BB Multi-indicator]")
    log(f"Data directory  : {data_dir}")
    log(f"Initial capital : {initial_capital:,.0f} KRW")
    log(f"Entry window    : 09:00-15:20 (정규) / NXT 세션 가능")
    log(f"Force exit      : 15:20 / 19:59 (당일 전량청산)")
    log(f"TP / SL / Trail : +3.5% / -1.5% / -1.2%(from peak)")
    log(f"{'=' * 60}\n")

    if not data_dir.exists():
        log(f"ERROR: Data directory not found: {data_dir}")
        return 1

    csv_files = {path.stem.zfill(6): path for path in data_dir.glob("*.csv")}
    if not csv_files:
        log(f"ERROR: No CSV files found in {data_dir}")
        return 1

    selected_names = dict(names or {})
    picks = None if codes else load_picks(data_dir / PICKS_FILENAME)
    if codes:
        target_codes = {code.zfill(6) for code in codes}
        csv_files = {code: path for code, path in csv_files.items() if code in target_codes}
        log(f"Filtered by --codes: {sorted(target_codes)}")
    elif picks:
        csv_files = {code: path for code, path in csv_files.items() if code in picks}
        selected_names.update({code: name for code, name in picks.items() if code not in selected_names})
        log(f"Loaded picks.txt ({len(picks)}): {sorted(picks.keys())}")
    else:
        sample_codes = set(SAMPLE_CODE_MAP.keys())
        matched = {code: path for code, path in csv_files.items() if code in sample_codes}
        if matched:
            csv_files = matched
            selected_names.update({code: name for code, name in SAMPLE_CODE_MAP.items() if code in matched})
            log(f"picks.txt not found; using sample codes: {sorted(matched.keys())}")
        else:
            log("picks.txt not found; using all CSV files in date folder")

    if not csv_files:
        log("ERROR: No matching CSV files after filtering")
        return 1

    frames: dict[str, pd.DataFrame] = {}
    for code in sorted(csv_files):
        frame = build_simulation_frame(data_root, date_str, code)
        if frame is None or frame.empty:
            log(f"Skipped {code}: failed to build simulation frame")
            continue
        frames[code] = frame

    if not frames:
        log("ERROR: No valid chart data loaded")
        return 1

    target_date = datetime.strptime(date_str, "%Y%m%d").date()
    all_times = sorted({ts for df in frames.values() for ts in df.index if ts.date() == target_date})
    if not all_times:
        log("ERROR: No bars found for target date")
        return 1

    sim = Simulator(initial_capital=initial_capital)
    liq_state: dict[str, object] = {}
    signal_buy_bar: dict[str, pd.Timestamp] = {}
    signal_sell_bar: dict[str, pd.Timestamp] = {}
    near_cross_armed_at: dict[str, pd.Timestamp] = {}

    for ts in all_times:
        run_scheduled_liquidations(sim, frames, selected_names, ts, liq_state)

        for code, frame in frames.items():
            if ts not in frame.index:
                continue

            nxt_tradeable = code in NXT_ELIGIBLE_CODES_FALLBACK
            if not can_trade_code_now(ts, nxt_tradeable):
                continue

            available = frame[frame.index <= ts]
            if len(available) < MIN_BARS_REQUIRED:
                continue

            cur = available.iloc[-1]
            price = float(cur["close"])
            session = classify_buy_session(ts)
            pos = sim.positions.get(code)

            if pos is not None:
                profit_pct = price / pos.buy_price - 1.0
                pos.highest_price = max(pos.highest_price, price)
                if profit_pct >= TAKE_PROFIT_PERCENT and not ENABLE_TP_EXTENSION_TRAILING:
                    sim.sell(code, price, ts, f"TAKE_PROFIT_+{TAKE_PROFIT_PERCENT*100:.1f}%", session)
                    signal_sell_bar[code] = ts
                    continue
                if profit_pct <= STOP_LOSS_PERCENT:
                    sim.sell(code, price, ts, f"STOP_LOSS_{STOP_LOSS_PERCENT*100:.1f}%", session)
                    signal_sell_bar[code] = ts
                    continue
                highest_price = float(pos.highest_price)
                if highest_price > 0 and pos.buy_price > 0:
                    peak_pnl_pct = (highest_price / pos.buy_price) - 1.0
                    current_pnl_pct = profit_pct
                    giveback = peak_pnl_pct - current_pnl_pct
                    if ENABLE_TP_EXTENSION_TRAILING and peak_pnl_pct >= TAKE_PROFIT_PERCENT:
                        trail_threshold = TP_EXTENSION_TRAIL_FROM_PEAK
                        reason_ts = f"TP_EXTENSION_TRAIL_{TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%"
                    else:
                        trail_threshold = TRAILING_STOP_FROM_PEAK
                        reason_ts = f"TRAILING_STOP_GIVEBACK_{TRAILING_STOP_FROM_PEAK*100:.1f}%"
                    if peak_pnl_pct > 0 and current_pnl_pct > 0 and giveback >= trail_threshold:
                        sim.sell(code, price, ts, reason_ts, session)
                        signal_sell_bar[code] = ts
                        continue

                if signal_sell_bar.get(code) == ts:
                    continue

                held_seconds = (ts - pos.buy_time).total_seconds() if isinstance(pos.buy_time, pd.Timestamp) else None
                should_sell, reason = check_sell_condition(available, profit_pct, held_seconds=held_seconds)
                if should_sell:
                    sim.sell(code, price, ts, reason, session)
                    signal_sell_bar[code] = ts
                continue

            if not is_new_entry_allowed(ts, nxt_tradeable):
                continue
            if sim.in_cooldown(code, ts):
                continue
            if (not ALLOW_REBUY_SAME_CODE) and code in sim.completed_codes:
                continue
            if signal_buy_bar.get(code) == ts:
                continue

            # 당일 봉이 최소 SAME_DAY_MIN_BARS 이상
            same_day_bars = available[available.index.date == target_date]
            if len(same_day_bars) < SAME_DAY_MIN_BARS:
                continue

            cur2 = available.iloc[-1]
            prev2 = available.iloc[-2]
            armed_at = near_cross_armed_at.get(code)
            if armed_at is not None:
                arm_expire_at = armed_at + pd.Timedelta(minutes=3 * NEAR_CROSS_ARM_EXPIRE_BARS)
                if ts > arm_expire_at:
                    near_cross_armed_at.pop(code, None)
                    armed_at = None

            should_buy, reason = check_buy_condition(available, ts)
            near_flags = _near_cross_momentum_flags(cur2, prev2)
            early_liq_ok, early_liq_reason = _passes_early_near_cross_liquidity(cur2)
            early_time_ok = is_early_near_cross_allowed(ts, nxt_tradeable)

            if (
                not should_buy
                and (not REQUIRE_STRICT_BUY_GOLDEN_CROSS)
                and ENABLE_EARLY_NEAR_CROSS_ENTRY
                and early_time_ok
                and armed_at is not None
            ):
                if reason.startswith("NO_MA5_BB_CROSS_UP") and bool(near_flags["can_early"]) and early_liq_ok:
                    should_buy = True
                    reason = "EARLY_NEAR_CROSS_MOMENTUM"

            if (
                not should_buy
                and (not REQUIRE_STRICT_BUY_GOLDEN_CROSS)
                and ENABLE_NEAR_CROSS_ARM
                and early_time_ok
                and reason.startswith("NO_MA5_BB_CROSS_UP")
                and bool(near_flags["can_arm"])
                and early_liq_ok
            ):
                if code not in near_cross_armed_at:
                    near_cross_armed_at[code] = ts

            if should_buy:
                log(
                    f"  [BUY SIGNAL] {code} | {ts:%H:%M} | {reason} | "
                    f"MA5={_num(cur2,'MA_5'):.1f} BB_MID={_num(cur2,'BB_MIDDLE'):.1f} | "
                    f"RSI={_num(cur2,'RSI'):.1f} SIG={_num(cur2,'RSI_SIGNAL'):.1f} | "
                    f"K={_num(cur2,'STOCH_K'):.1f} D={_num(cur2,'STOCH_D'):.1f} | "
                    f"WR={_num(cur2,'WILLIAMS_R'):.1f} WD={_num(cur2,'WILLIAMS_D'):.1f}"
                )
                if sim.buy(code, selected_names.get(code, code), price, ts, session, reason):
                    near_cross_armed_at.pop(code, None)
                    signal_buy_bar[code] = ts

            if reason.startswith("NO_MA5_BB_CROSS_UP") and not early_time_ok:
                log(f"  [BUY HOLD] {code} | EARLY_NEAR_CROSS_BLOCKED_TIME_WINDOW")
            elif reason.startswith("NO_MA5_BB_CROSS_UP") and not early_liq_ok:
                log(f"  [BUY HOLD] {code} | EARLY_NEAR_CROSS_BLOCKED_{early_liq_reason}")

    final_close_time = pd.Timestamp(datetime.combine(target_date, AFTERNOON_NXT_END))
    for code in list(sim.positions.keys()):
        price = get_latest_price_up_to(frames[code], final_close_time)
        if price is None:
            continue
        session = sim.positions[code].buy_session
        sim.sell(code, price, final_close_time, "EOD_CLOSE", session)

    last_prices = {
        code: float(frame[frame.index.date == target_date].iloc[-1]["close"])
        for code, frame in frames.items()
        if not frame[frame.index.date == target_date].empty
    }
    final_value = sim.portfolio_value(last_prices)
    total_pnl = final_value - initial_capital
    total_pnl_pct = total_pnl / initial_capital * 100.0 if initial_capital else 0.0

    sell_trades = [record for record in sim.trade_log if record.action == "SELL" and record.pnl_krw is not None]
    realized_pnl = sum(record.pnl_krw for record in sell_trades)
    wins = [record for record in sell_trades if record.pnl_krw > 0]
    losses = [record for record in sell_trades if record.pnl_krw <= 0]

    traded_codes: list[str] = []
    seen: set[str] = set()
    for record in sim.trade_log:
        if record.code not in seen:
            traded_codes.append(record.code)
            seen.add(record.code)

    log(f"\n{'=' * 60}")
    log(f"SIMULATION RESULTS [{date_str}] - R73 MA5-BB Cross")
    log(f"{'=' * 60}")

    for code in traded_codes:
        name = selected_names.get(code, code)
        buys = [record for record in sim.trade_log if record.code == code and record.action == "BUY"]
        sells = [record for record in sim.trade_log if record.code == code and record.action == "SELL"]
        code_pnl = sum(record.pnl_krw or 0.0 for record in sells)

        raw(f"\n  [{code}] {name}")
        raw(f"  {'─' * 50}")
        for record in buys:
            raw(f"    {record.bar_time:%H:%M}  매수  qty={record.qty:>4d}  price={record.price:>9,.0f}  [{record.reason}]")
        for record in sells:
            pnl_pct = record.pnl_pct if record.pnl_pct is not None else 0.0
            pnl_krw = record.pnl_krw if record.pnl_krw is not None else 0.0
            raw(
                f"    {record.bar_time:%H:%M}  매도  qty={record.qty:>4d}  price={record.price:>9,.0f}"
                f"  {pnl_pct:+.2f}%  {pnl_krw:+,.0f} KRW  [{record.reason}]"
            )
        raw(f"    {'─' * 46}")
        raw(f"    종목 손익: {code_pnl:+,.0f} KRW  (매수 {len(buys)}회 / 매도 {len(sells)}회)")

    total_sell_count = len(sell_trades)
    win_rate = len(wins) / total_sell_count * 100.0 if total_sell_count else 0.0

    raw(f"\n{'=' * 60}")
    raw("전체 결과 (R73 MA5-BB Cross)")
    raw(f"{'─' * 60}")
    raw(f"  초기 자본    : {initial_capital:>15,.0f} KRW")
    raw(f"  최종 평가액  : {final_value:>15,.0f} KRW")
    raw(f"  실현 손익    : {realized_pnl:>+15,.0f} KRW")
    raw(f"  총 손익      : {total_pnl:>+15,.0f} KRW  ({total_pnl_pct:+.2f}%)")
    raw(f"  잔여 현금    : {sim.cash:>15,.0f} KRW")
    raw(f"{'─' * 60}")
    raw(f"  매도 횟수    : {total_sell_count}회  (익절 {len(wins)}회 / 손절·기타 {len(losses)}회)")
    raw(f"  승률         : {win_rate:.1f}%")
    raw(f"{'=' * 60}\n")

    LAST_SIM_STATS = {
        "date": date_str,
        "initial_capital": float(initial_capital),
        "final_value": float(final_value),
        "realized_pnl": float(realized_pnl),
        "total_pnl": float(total_pnl),
        "total_pnl_pct": float(total_pnl_pct),
        "sell_count": int(total_sell_count),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": float(win_rate),
        "traded_code_count": int(len(traded_codes)),
    }

    return 0


def main() -> None:
    global TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT, TRAILING_STOP_FROM_PEAK

    parser = argparse.ArgumentParser(description="Simulate R73 MA5-BB cross strategy on historical 3-minute bar data")
    parser.add_argument("--date", required=True, help="Simulation date in YYYYMMDD format")
    parser.add_argument("--codes", nargs="*", help="Optional stock codes to simulate")
    parser.add_argument(
        "--data-root",
        default=str(SCRIPT_DIR / DATA_DIR_NAME),
        help="Root directory containing date subfolders",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=INITIAL_CAPITAL,
        help="Initial capital in KRW",
    )
    parser.add_argument(
        "--today-code-file",
        default=str(TODAY_CODE_FILE),
        help="Path to define_today_code.txt for code names",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=None,
        help="Take-profit ratio (e.g. 0.035 for 3.5%%)",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=None,
        help="Stop-loss ratio as positive number (e.g. 0.015 for 1.5%%)",
    )
    parser.add_argument(
        "--trail",
        type=float,
        default=None,
        help="Trailing stop from peak ratio (e.g. 0.012 for 1.2%%)",
    )
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        print(f"ERROR: Invalid date format: {args.date} (expected YYYYMMDD)")
        sys.exit(1)

    if args.tp is not None:
        TAKE_PROFIT_PERCENT = float(args.tp)
    if args.sl is not None:
        STOP_LOSS_PERCENT = -abs(float(args.sl))
    if args.trail is not None:
        TRAILING_STOP_FROM_PEAK = abs(float(args.trail))

    names = load_code_name_map(Path(args.today_code_file))
    exit_code = simulate_date(
        date_str=args.date,
        data_root=Path(args.data_root),
        codes=args.codes,
        names=names,
        initial_capital=args.capital,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
