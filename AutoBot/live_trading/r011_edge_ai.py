# -*- coding: utf-8 -*-
"""Edge AI meta-label filter for the primary buy path (LightGBM, runs on the Raspberry Pi CPU).

What it does
- The rule-based strategy (r005 _001~_020) stays the *primary* signal. When it is about to buy, this module estimates
  P(win) = probability that the entry ends with a positive net return (after fees/tax/slippage) under a
  triple-barrier exit (take-profit / stop-loss / time limit). This is "meta-labeling" (Lopez de Prado,
  Advances in Financial Machine Learning): the model never creates trades, it only filters/sizes the
  primary model's trades.
- The same compute_features() is used by the offline dataset builder (g013) and live (r005 _028), so
  train/serve features are produced by one code path (no train/serve skew by construction).

Feature parity rules (live <-> offline)
- 1m bars: KRX regular session only (end-time label > 09:00), indicators recomputed here with
  r002.calculate_indicators -> NXT pre-market bars in r003's cache do not leak into live features.
- Fixed history window (v2): indicators are computed on the regular-session bars with label >=
  max(09:01, now - (FEATURE_HISTORY_BARS-1) min) only - offline AND live. EMA/ADX/3m values therefore do not
  depend on how much history r003's cache (max 200 bars, starts at bot start) happens to hold. When the live
  cache does not reach back to that start (e.g. bot restarted mid-day) the gate fails open
  ("insufficient_history") instead of predicting on different inputs.
- 3m bars: built HERE from the window's 1m bars, completed buckets only (label="right", closed="right"); a
  leading bucket the window only partly covers is dropped. r003's intrabar 3m bar is NOT used.
- Trigger features (trig_path / trig_bars_since_cross): live passes r005 _008's real trigger_info; offline
  (g013) runs the same r002 trigger with the 3m uptrend_continuation context like _008.
- Day refs: prev close / day open / day VWAP are passed in (live: KIS quote, offline: reconstructed).
- Daily: latest collected <code>_*_daily.csv in a data folder strictly BEFORE the trade date.
- Rolling windows never exceed 60 1m bars.

Modes (r001 EDGE_AI_MODE)
- "off": nothing is computed.
- "shadow": predict + log only, never blocks (default).
- "auto": blocks entries with P(win) < threshold ONLY if the current model was promoted by g014's
  walk-forward check; otherwise behaves like "shadow".
- "gate": always blocks below threshold when a model is loaded (manual override, not recommended).
Any error / missing model / missing lightgbm -> fail-open (entry allowed, reason logged).

Update log:
- [2026-10-26] type=feat owner=claude
    summary: 신규 - 경량 Edge AI(LightGBM 메타 라벨링) 피처/모델/게이트. r005 _028이 호출, g013/g014가 학습.
    impact: live (기본 EDGE_AI_MODE="shadow" - 기록만, 주문 영향 없음)
    compatibility: backward-compatible
- [2026-10-27] type=fix owner=claude
    summary: Codex 1차 리뷰 반영 - 고정 히스토리 창(FEATURE_HISTORY_BARS, FEATURE_VERSION 2), _008 실제 trigger_info
      사용, 버전별 모델 파일 + meta 단일 원자 교체/스냅샷 예측, 비정상 확률·임계값 fail-open.
    impact: live (shadow 기본 - 주문 영향 없음). v1 모델은 feature_version 불일치로 로드 거부(fail-open).
    compatibility: backward-compatible (EdgeAIGate.evaluate에 trigger_info 선택 인자 추가)
"""
from __future__ import annotations

import json
import math
import threading
import time as _time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import r002_strategy_core_shared as _r002
from r002_strategy_core_shared import (
    BUY_SCORE_RULES,
    HYBRID_3MIN_CONTEXT_GATES,
    StrategyConfig,
    _compute_bb_slope_pct,
    _evaluate_bb_mid_cross,
    _num,
    build_context_eval,
    calculate_indicators,
    check_buy_condition_1min_hybrid_trigger,
)
from r008_strategy_v2 import krx_tick_size

FEATURE_VERSION = 2
SESSION_OPEN = dt_time(9, 0)
SESSION_LAST_BAR = dt_time(15, 20)   # end label of the last continuous-session 1m bar
MAX_WINDOW_BARS = 60
FEATURE_HISTORY_BARS = 90            # indicator history (1m bars) - must fit r003's 200-bar cache incl. NXT bars
MIN_SESSION_BARS = 5                 # fewer confirmed regular-session bars -> no prediction (fail-open)

_TRIGGER_PATH_CODE = {"cross": 0, "uptrend_1min": 1, "uptrend_3min": 2}

FEATURE_NAMES: tuple[str, ...] = (
    # 1m price action
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_30",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "range_pct",
    "vol_ratio_20", "vol_ratio_5", "turnover_30_log",
    "bb_gap_pct", "bb_width_pct", "bb_pctb", "bars_above_mid",
    "rsi", "stoch_k", "stoch_d", "williams_r", "mfi",
    "macd_hist_pct", "adx", "di_spread", "atr_pct", "ema20_gap_pct", "ema20_60_pct", "obv_above_ma",
    "high60_dist_pct", "low60_dist_pct",
    "trig_path", "trig_bars_since_cross",
    # 3m context (completed buckets)
    "m3_score", "m3_gates_passed", "m3_ctx_pass", "m3_rsi", "m3_adx", "m3_bb_slope_pct",
    "m3_vol_ratio", "m3_bb_gap_pct", "m3_ema20_60_pct",
    # day refs
    "rise_prev_close_pct", "rise_open_pct", "gap_open_pct", "vwap_gap_pct",
    "minutes_since_open", "log_price", "tick_pct",
    # daily (D-1 and earlier)
    "d_ret_5", "d_ret_20", "d_close_ma20_pct", "d_ma20_ma60_pct", "d_atr_pct", "d_turnover_log20",
    "d_high60_dist_pct",
)


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _pct(a: float, b: float) -> float:
    a, b = _f(a), _f(b)
    if not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return float("nan")
    return (a / b - 1.0) * 100.0


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def session_raw_1min(frame_1min: pd.DataFrame | None, trade_date) -> pd.DataFrame | None:
    """End-labeled OHLCV 1m bars of the KRX regular session on trade_date (no indicators)."""
    if frame_1min is None or len(frame_1min) == 0:
        return None
    idx = pd.DatetimeIndex(frame_1min.index)
    day = pd.Timestamp(trade_date).normalize()
    t = idx.time
    mask = (idx.normalize() == day) & (t > SESSION_OPEN) & (t <= SESSION_LAST_BAR)
    raw = frame_1min.loc[mask, ["open", "high", "low", "close", "volume"]]
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return None if raw.empty else raw


def prepare_session_1min(frame_1min: pd.DataFrame | None, trade_date) -> pd.DataFrame | None:
    """session_raw_1min() with indicators over the whole session (used for offline trigger detection)."""
    raw = session_raw_1min(frame_1min, trade_date)
    return None if raw is None else calculate_indicators(raw)


# ---------------------------------------------------------------------------
# indicators (NumPy port of r002.calculate_indicators - same formulas/columns, ~50x faster on the Pi)
# ---------------------------------------------------------------------------

def _ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    """pandas Series.ewm(alpha=alpha, adjust=False, ignore_na=False, min_periods<=1).mean()."""
    out = np.full(len(x), np.nan)
    weighted, old_wt = np.nan, 1.0
    for i, cur in enumerate(x):
        obs = cur == cur
        if weighted == weighted:
            old_wt *= 1.0 - alpha
            if obs:
                weighted = (old_wt * weighted + alpha * cur) / (old_wt + alpha)
                old_wt = 1.0
        elif obs:
            weighted = cur
        out[i] = weighted
    return out


def _windows(x: np.ndarray, w: int) -> np.ndarray:
    padded = np.concatenate([np.full(w - 1, np.nan), x])
    return np.lib.stride_tricks.sliding_window_view(padded, w)


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """rolling(w, min_periods=1).mean() - NaNs skipped, all-NaN window -> NaN."""
    v = _windows(x, w)
    ok = ~np.isnan(v)
    n = ok.sum(axis=1)
    s = np.where(ok, v, 0.0).sum(axis=1)
    return np.divide(s, n, out=np.full(len(x), np.nan), where=n > 0)


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    v = _windows(x, w)
    ok = ~np.isnan(v)
    n = ok.sum(axis=1)
    return np.where(n > 0, np.where(ok, v, 0.0).sum(axis=1), np.nan)


def _roll_extreme(x: np.ndarray, w: int, fn) -> np.ndarray:
    v = _windows(x, w)
    n = (~np.isnan(v)).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = fn(v, axis=1)
    return np.where(n > 0, r, np.nan)


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    """rolling(w, min_periods=1).std(ddof=1); a constant window is exactly 0 (pandas does the same)."""
    v = _windows(x, w)
    ok = ~np.isnan(v)
    n = ok.sum(axis=1)
    mean = np.divide(np.where(ok, v, 0.0).sum(axis=1), n, out=np.zeros(len(x)), where=n > 0)
    dev = np.where(ok, v - mean[:, None], 0.0)
    var = np.divide((dev * dev).sum(axis=1), n - 1, out=np.full(len(x), np.nan), where=n > 1)
    const = _roll_extreme(x, w, np.nanmax) == _roll_extreme(x, w, np.nanmin)
    var = np.where(const & (n > 1), 0.0, var)
    return np.sqrt(var)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b with b == 0 -> NaN (pandas .replace(0, nan) idiom)."""
    with np.errstate(all="ignore"):
        return np.where(b == 0, np.nan, a / np.where(b == 0, 1.0, b))


def fast_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Same columns/values as r002.calculate_indicators (tests assert equality to 1e-9)."""
    data: dict[str, Any] = {k: df[k] for k in df.columns}
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
    o, h, l, c, v = (data[k] for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    span = lambda p: 2.0 / (p + 1.0)  # noqa: E731
    shift = lambda x: np.r_[np.nan, x[:-1]] if n else x  # noqa: E731
    cols: dict[str, np.ndarray] = {}
    cols["MA_5"] = _roll_mean(c, _r002.MA_PERIOD)
    cols["VOL_MA20"] = _roll_mean(v, _r002.VOLUME_MA_PERIOD)
    cols["EMA_20"] = _ewm(c, span(_r002.EMA_20_PERIOD))
    cols["EMA_60"] = _ewm(c, span(_r002.EMA_60_PERIOD))
    mid = _roll_mean(c, _r002.BB_PERIOD)
    std = _roll_std(c, _r002.BB_PERIOD)
    cols["BB_MIDDLE"], cols["BB_STD"] = mid, std
    cols["BB_UPPER"] = mid + std * _r002.BB_STD_MULTIPLIER
    cols["BB_LOWER"] = mid - std * _r002.BB_STD_MULTIPLIER

    delta = c - shift(c)
    with np.errstate(invalid="ignore"):
        gain = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
        loss = np.where(np.isnan(delta), np.nan, -np.minimum(delta, 0.0))
    avg_gain = _ewm(gain, 1.0 / _r002.RSI_PERIOD)
    avg_loss = _ewm(loss, 1.0 / _r002.RSI_PERIOD)
    rsi = 100 - (100 / (1 + _safe_div(avg_gain, avg_loss)))
    rsi = np.where(avg_loss == 0, 100.0, rsi)
    cols["RSI"] = rsi
    cols["RSI_SIGNAL"] = _roll_mean(rsi, _r002.RSI_SIGNAL_PERIOD)

    low_n = _roll_extreme(l, _r002.STOCH_K_PERIOD, np.nanmin)
    high_n = _roll_extreme(h, _r002.STOCH_K_PERIOD, np.nanmax)
    stoch_k = 100.0 * _safe_div(c - low_n, high_n - low_n)
    cols["STOCH_K"] = stoch_k
    cols["STOCH_D"] = _roll_mean(stoch_k, _r002.STOCH_D_PERIOD)
    high_w = _roll_extreme(h, _r002.WILLIAMS_R_PERIOD, np.nanmax)
    low_w = _roll_extreme(l, _r002.WILLIAMS_R_PERIOD, np.nanmin)
    wr = -100.0 * _safe_div(high_w - c, high_w - low_w)
    cols["WILLIAMS_R"] = wr
    cols["WILLIAMS_D"] = _roll_mean(wr, _r002.WILLIAMS_D_PERIOD)

    tp = (h + l + c) / 3.0
    flow = tp * v
    tp_delta = tp - shift(tp)
    with np.errstate(invalid="ignore"):
        pos_sum = _roll_sum(np.where(tp_delta > 0, flow, 0.0), _r002.MFI_PERIOD)
        neg_sum = _roll_sum(np.where(tp_delta < 0, flow, 0.0), _r002.MFI_PERIOD)
    mfi = 100.0 - (100.0 / (1.0 + _safe_div(pos_sum, neg_sum)))
    mfi = np.where((pos_sum == 0) & (neg_sum == 0), 50.0, mfi)
    mfi = np.where((neg_sum == 0) & (pos_sum > 0), 100.0, mfi)
    mfi = np.where((pos_sum == 0) & (neg_sum > 0), 0.0, mfi)
    cols["MFI"] = mfi

    macd = _ewm(c, span(_r002.MACD_FAST)) - _ewm(c, span(_r002.MACD_SLOW))
    signal = _ewm(macd, span(_r002.MACD_SIGNAL_PERIOD))
    cols["MACD"], cols["MACD_SIGNAL"], cols["MACD_HIST"] = macd, signal, macd - signal

    prev_c = shift(c)
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0) if n else c
    high_diff = h - shift(h)
    low_diff = shift(l) - l
    with np.errstate(invalid="ignore"):
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    a_adx = 1.0 / _r002.ADX_PERIOD
    ema_tr = _ewm(tr, a_adx)
    di_p = 100.0 * _safe_div(_ewm(plus_dm, a_adx), ema_tr)
    di_m = 100.0 * _safe_div(_ewm(minus_dm, a_adx), ema_tr)
    cols["DI_PLUS"], cols["DI_MINUS"] = di_p, di_m
    cols["ADX"] = _ewm(100.0 * _safe_div(np.abs(di_p - di_m), di_p + di_m), a_adx)

    cols["VWAP"] = _safe_div(np.cumsum(c * v), np.cumsum(v))
    cols["ATR"] = _ewm(tr, 1.0 / _r002.ATR_PERIOD)
    with np.errstate(invalid="ignore"):
        cdiff = c - prev_c
        obv = np.cumsum(v * (cdiff > 0) - v * (cdiff < 0))
    cols["OBV"] = obv
    cols["OBV_MA"] = _roll_mean(obv, _r002.OBV_MA_PERIOD)
    data.update(cols)  # one DataFrame build - inserting ~30 columns one by one costs ~40 ms on the Pi
    return pd.DataFrame(data, index=df.index)


def history_window_start(last_label) -> pd.Timestamp:
    last = pd.Timestamp(last_label)
    first = last.normalize() + pd.Timedelta(hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute + 1)
    return max(first, last - pd.Timedelta(minutes=FEATURE_HISTORY_BARS - 1))


def feature_frames(raw: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None] | None:
    """(m1, m3) for compute_features from session OHLCV ending at the current bar, indicators computed on the
    fixed history window only. None when raw does not reach back to the window start (incomplete history)."""
    if raw is None or raw.empty:
        return None
    start = history_window_start(raw.index[-1])
    if raw.index[0] > start:
        return None
    m1 = fast_indicators(raw[raw.index >= start])
    return m1, completed_3min(m1)


def completed_3min(m1: pd.DataFrame) -> pd.DataFrame | None:
    """3m bars (label=right/closed=right like r003) from the session 1m bars, completed buckets only.
    A leading bucket that started before m1's first bar is dropped (it would be a partial bar)."""
    if m1 is None or m1.empty:
        return None
    bars = m1[["open", "high", "low", "close", "volume"]].resample("3min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    bars = bars[bars.index <= m1.index[-1]]
    first_bucket_start = pd.Timestamp(m1.index[0]) - pd.Timedelta(minutes=1)
    if not bars.empty and first_bucket_start != first_bucket_start.floor("3min"):
        bars = bars.iloc[1:]
    if bars.empty:
        return None
    return fast_indicators(bars)


def build_strategy_config(cfg_module) -> StrategyConfig:
    """Same field mapping as r003 SHARED_STRATEGY_CONFIG (only score/stoch fields matter for features)."""
    g = lambda name, default=None: getattr(cfg_module, name, default)  # noqa: E731
    return StrategyConfig(
        live_price_bb_buffer_pct=g("LIVE_PRICE_BB_BUFFER_PCT", 0.0),
        live_price_cross_confirm_polls=g("LIVE_PRICE_CROSS_CONFIRM_POLLS", 1),
        live_price_cross_confirm_seconds=g("LIVE_PRICE_CROSS_CONFIRM_SECONDS", 0.0),
        live_price_down_cross_confirm_polls=g("LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS", 1),
        live_price_down_cross_confirm_seconds=g("LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS", 0.0),
        stoch_overbought=g("STOCH_OVERBOUGHT", 96.0),
        stop_loss_percent=g("STOP_LOSS_PERCENT", -0.02),
        take_profit_percent=g("STAGED_TP1_PCT", 0.03),
        enable_box_range_hold_tech_sell=g("ENABLE_BOX_RANGE_HOLD_TECH_SELL", False),
        box_range_hold_lookback_bars=g("BOX_RANGE_HOLD_LOOKBACK_BARS", 8),
        box_range_hold_max_range_pct=g("BOX_RANGE_HOLD_MAX_RANGE_PCT", 0.65),
        box_range_hold_max_bb_width_pct=g("BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT", 0.8),
        ma5_bb_down_cross_min_pnl=g("MA5_BB_DOWN_CROSS_MIN_PNL", 0.0),
        ma5_bb_down_cross_immediate_pnl=g("MA5_BB_DOWN_CROSS_IMMEDIATE_PNL", -0.7),
        ma5_bb_down_cross_immediate_score=g("MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE", 2),
        aux_sell_min_pnl_score2=g("AUX_SELL_MIN_PNL_SCORE2", 1.5),
        aux_sell_min_pnl_score3=g("AUX_SELL_MIN_PNL_SCORE3", 0.8),
        aux_sell_min_pnl_score4=g("AUX_SELL_MIN_PNL_SCORE4", 0.3),
        bb_buy_score_threshold=g("BB_BUY_SCORE_THRESHOLD", 10),
    )


def hybrid_trigger_with_context(frame_1min: pd.DataFrame, frame_3min: pd.DataFrame | None, price: float,
                                info: dict) -> tuple[bool, str]:
    """Offline replica of r005 _008 / g003: the 3m context's uptrend_continuation is passed to the r002 1m trigger.
    (No live-price cross state offline -> cross_info={}, same as g003 without tick data.)"""
    context_uptrend_continuation = False
    if frame_3min is not None and len(frame_3min) >= 2:
        cur3, prev3 = frame_3min.iloc[-1], frame_3min.iloc[-2]
        cur3_bb, prev3_bb = _num(cur3, "BB_MIDDLE"), _num(prev3, "BB_MIDDLE")
        if not any(pd.isna(v) for v in (cur3_bb, prev3_bb)):
            cross_eval = _evaluate_bb_mid_cross(frame_3min, cur3, prev3, cur3_bb, prev3_bb, float(price),
                                                _compute_bb_slope_pct(frame_3min), {})
            context_uptrend_continuation = bool(cross_eval.get("uptrend_continuation"))
    return check_buy_condition_1min_hybrid_trigger(
        frame_1min, context_uptrend_continuation=context_uptrend_continuation, info=info)


# ---------------------------------------------------------------------------
# daily context
# ---------------------------------------------------------------------------

def find_prev_data_dir(data_root: Path, trade_date: str) -> Path | None:
    """Latest data/YYYYMMDD folder strictly before trade_date (that day's daily.csv ends at D-1)."""
    try:
        names = sorted(p.name for p in Path(data_root).iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8)
    except OSError:
        return None
    prev = [n for n in names if n < trade_date]
    return Path(data_root) / prev[-1] if prev else None


def daily_features_from_frame(daily: pd.DataFrame | None, trade_date: str) -> dict[str, float]:
    out = {k: float("nan") for k in FEATURE_NAMES if k.startswith("d_")}
    if daily is None or daily.empty or "close" not in daily.columns:
        return out
    d = daily.copy()
    d["date"] = d["date"].astype(str).str.slice(0, 8)
    d = d[d["date"] < trade_date].sort_values("date")
    if len(d) < 2:
        return out
    close = pd.to_numeric(d["close"], errors="coerce")
    high = pd.to_numeric(d["high"], errors="coerce")
    low = pd.to_numeric(d["low"], errors="coerce")
    amount = pd.to_numeric(d.get("amount", close * pd.to_numeric(d["volume"], errors="coerce")), errors="coerce")
    c = _f(close.iloc[-1])
    ma20 = _f(close.tail(20).mean()) if len(close) >= 20 else float("nan")
    ma60 = _f(close.tail(60).mean()) if len(close) >= 60 else float("nan")
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["d_ret_5"] = _pct(c, close.iloc[-6]) if len(close) >= 6 else float("nan")
    out["d_ret_20"] = _pct(c, close.iloc[-21]) if len(close) >= 21 else float("nan")
    out["d_close_ma20_pct"] = _pct(c, ma20)
    out["d_ma20_ma60_pct"] = _pct(ma20, ma60)
    out["d_atr_pct"] = _f(tr.tail(14).mean()) / c * 100.0 if c > 0 else float("nan")
    amt20 = _f(amount.tail(20).mean())
    out["d_turnover_log20"] = math.log10(amt20 + 1.0) if math.isfinite(amt20) and amt20 >= 0 else float("nan")
    out["d_high60_dist_pct"] = _pct(c, high.tail(60).max())
    return out


def load_daily_features(data_root: Path, code: str, trade_date: str) -> dict[str, float]:
    prev_dir = find_prev_data_dir(data_root, trade_date)
    daily = None
    if prev_dir is not None:
        files = sorted(prev_dir.glob(f"{str(code).zfill(6)}_*_daily.csv"))
        if files:
            try:
                daily = pd.read_csv(files[0], encoding="utf-8-sig", dtype={"date": str})
            except Exception:
                daily = None
    return daily_features_from_frame(daily, trade_date)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def _ret(close: np.ndarray, k: int) -> float:
    if len(close) <= k:
        return float("nan")
    return _pct(close[-1], close[-1 - k])


def context_3min_features(m3: pd.DataFrame | None, now: pd.Timestamp, price: float, config: StrategyConfig) -> dict[str, float]:
    out = {"m3_score": np.nan, "m3_gates_passed": np.nan, "m3_ctx_pass": np.nan, "m3_rsi": np.nan, "m3_adx": np.nan,
           "m3_bb_slope_pct": np.nan, "m3_vol_ratio": np.nan, "m3_bb_gap_pct": np.nan, "m3_ema20_60_pct": np.nan}
    if m3 is None or len(m3) < 2:
        return out
    cur = m3.iloc[-1]
    out["m3_rsi"] = _f(cur.get("RSI"))
    out["m3_adx"] = _f(cur.get("ADX"))
    vma = _f(cur.get("VOL_MA20"))
    out["m3_vol_ratio"] = _f(cur.get("volume")) / vma if vma and vma > 0 else np.nan
    out["m3_bb_gap_pct"] = _pct(price, cur.get("BB_MIDDLE"))
    out["m3_ema20_60_pct"] = _pct(cur.get("EMA_20"), cur.get("EMA_60"))
    try:
        ctx, _reason = build_context_eval(m3, now, float(price), {}, config)
        if ctx is None:
            return out
        out["m3_bb_slope_pct"] = _f(ctx.bb_slope_pct)
        passed_all = True
        n_pass = 0
        for gate in HYBRID_3MIN_CONTEXT_GATES:
            ok, ctx, _r = gate.eval_fn(ctx)
            if ok:
                n_pass += 1
            else:
                passed_all = False
                break  # gates mutate ctx sequentially - same order/semantics as r005
        score = sum(rule.eval_fn(ctx) for rule in BUY_SCORE_RULES)
        out["m3_gates_passed"] = float(n_pass)
        out["m3_score"] = float(score)
        out["m3_ctx_pass"] = float(passed_all and score >= config.bb_buy_score_threshold)
    except Exception:
        pass
    return out


def compute_features(
    m1: pd.DataFrame,
    m3: pd.DataFrame | None,
    now: pd.Timestamp,
    price: float,
    prev_close: float | None,
    day_open: float | None,
    day_vwap: float | None,
    daily_feats: dict[str, float],
    config: StrategyConfig,
    trigger_info: dict | None = None,
) -> dict[str, float]:
    """m1: prepare_session_1min() output ending at the latest confirmed bar; m3: completed_3min() of the same.
    trigger_info: info dict already filled by check_buy_condition_1min_hybrid_trigger on the same
    m1.iloc[-MAX_WINDOW_BARS:] window (optional - recomputed here when None)."""
    w = m1.iloc[-MAX_WINDOW_BARS:]
    cur = w.iloc[-1]
    close = w["close"].to_numpy(dtype=float)
    vol = w["volume"].to_numpy(dtype=float)
    o, h, l, c = (_f(cur[k]) for k in ("open", "high", "low", "close"))
    feats: dict[str, float] = {}
    for k in (1, 3, 5, 10, 30):
        feats[f"ret_{k}"] = _ret(close, k)
    feats["body_pct"] = _pct(c, o)
    feats["upper_wick_pct"] = (h - max(o, c)) / c * 100.0 if c > 0 else np.nan
    feats["lower_wick_pct"] = (min(o, c) - l) / c * 100.0 if c > 0 else np.nan
    feats["range_pct"] = (h - l) / c * 100.0 if c > 0 else np.nan
    vma = _f(cur.get("VOL_MA20"))
    feats["vol_ratio_20"] = vol[-1] / vma if vma and vma > 0 else np.nan
    prev5 = vol[-6:-1]
    feats["vol_ratio_5"] = vol[-1] / prev5.mean() if len(prev5) and prev5.mean() > 0 else np.nan
    turnover = float(np.nansum(close[-30:] * vol[-30:]))
    feats["turnover_30_log"] = math.log10(turnover + 1.0)
    mid, up, lo = _f(cur.get("BB_MIDDLE")), _f(cur.get("BB_UPPER")), _f(cur.get("BB_LOWER"))
    feats["bb_gap_pct"] = _pct(c, mid)
    feats["bb_width_pct"] = (up - lo) / mid * 100.0 if mid and mid > 0 else np.nan
    feats["bb_pctb"] = (c - lo) / (up - lo) if math.isfinite(up - lo) and (up - lo) > 0 else np.nan
    above = (w["close"] > w["BB_MIDDLE"]).to_numpy()
    run = 0
    for v in above[::-1]:
        if not v:
            break
        run += 1
    feats["bars_above_mid"] = float(run)
    feats["rsi"] = _f(cur.get("RSI"))
    feats["stoch_k"] = _f(cur.get("STOCH_K"))
    feats["stoch_d"] = _f(cur.get("STOCH_D"))
    feats["williams_r"] = _f(cur.get("WILLIAMS_R"))
    feats["mfi"] = _f(cur.get("MFI"))
    feats["macd_hist_pct"] = _f(cur.get("MACD_HIST")) / c * 100.0 if c > 0 else np.nan
    feats["adx"] = _f(cur.get("ADX"))
    feats["di_spread"] = _f(cur.get("DI_PLUS")) - _f(cur.get("DI_MINUS"))
    feats["atr_pct"] = _f(cur.get("ATR")) / c * 100.0 if c > 0 else np.nan
    feats["ema20_gap_pct"] = _pct(c, cur.get("EMA_20"))
    feats["ema20_60_pct"] = _pct(cur.get("EMA_20"), cur.get("EMA_60"))
    feats["obv_above_ma"] = float(_f(cur.get("OBV")) > _f(cur.get("OBV_MA")))
    feats["high60_dist_pct"] = _pct(c, w["high"].max())
    feats["low60_dist_pct"] = _pct(c, w["low"].min())
    info: dict = trigger_info if trigger_info is not None else {}
    if trigger_info is None:
        try:
            check_buy_condition_1min_hybrid_trigger(m1.iloc[-MAX_WINDOW_BARS:], False, info)
        except Exception:
            info = {}
    feats["trig_path"] = float(_TRIGGER_PATH_CODE.get(info.get("path"), -1))
    bsc = info.get("bars_since_cross")
    feats["trig_bars_since_cross"] = float(bsc) if bsc is not None else np.nan

    feats.update(context_3min_features(m3, now, price, config))

    feats["rise_prev_close_pct"] = _pct(price, prev_close)
    feats["rise_open_pct"] = _pct(price, day_open)
    feats["gap_open_pct"] = _pct(day_open, prev_close)
    feats["vwap_gap_pct"] = _pct(price, day_vwap)
    now_ts = pd.Timestamp(now)
    feats["minutes_since_open"] = (now_ts - now_ts.normalize() - pd.Timedelta(hours=9)).total_seconds() / 60.0
    feats["log_price"] = math.log10(price) if price and price > 0 else np.nan
    feats["tick_pct"] = krx_tick_size(price) / price * 100.0 if price and price > 0 else np.nan
    feats.update(daily_feats)
    return {k: _f(feats.get(k)) for k in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def valid_prob(value: Any) -> bool:
    """Finite and within [0, 1] (probabilities / thresholds)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0.0 <= value <= 1.0


class EdgeAIModel:
    """LightGBM booster + meta, hot-reloaded when <model_dir>/current_meta.json changes.

    g014 writes each model to an immutable models/model_<id>.txt and then atomically replaces current_meta.json,
    whose "model_file" points at it -> the meta file alone selects the (booster, threshold, promoted) generation.
    The loaded pair is held in ONE attribute so a concurrent reader never mixes generations."""

    def __init__(self, model_dir: Path, reload_check_seconds: float = 60.0):
        self.model_dir = Path(model_dir)
        self.reload_check_seconds = reload_check_seconds
        self._loaded: tuple[Any, dict] | None = None   # (booster, meta) - replaced as a whole
        self.error = "not_loaded"
        self._mtime: float | None = None
        self._last_check = 0.0
        self._lock = threading.Lock()

    @property
    def booster(self):
        loaded = self._loaded
        return loaded[0] if loaded else None

    @property
    def meta(self) -> dict:
        loaded = self._loaded
        return loaded[1] if loaded else {}

    @property
    def meta_path(self) -> Path:
        return self.model_dir / "current_meta.json"

    def _load(self, meta_text: str) -> tuple[Any, dict]:
        import lightgbm as lgb  # lazy: r003 must still start without lightgbm installed
        meta = json.loads(meta_text)
        if int(meta.get("feature_version", -1)) != FEATURE_VERSION:
            raise ValueError(f"feature_version {meta.get('feature_version')} != {FEATURE_VERSION}")
        if list(meta.get("features", [])) != list(FEATURE_NAMES):
            raise ValueError("feature list mismatch")
        threshold = meta.get("threshold")
        if threshold is not None and not valid_prob(threshold):
            raise ValueError(f"invalid threshold {threshold!r}")
        model_file = (self.model_dir / str(meta.get("model_file", ""))).resolve()
        if not model_file.is_file() or self.model_dir.resolve() not in model_file.parents:
            raise ValueError(f"model file missing: {meta.get('model_file')}")
        return lgb.Booster(model_file=str(model_file)), meta

    def maybe_reload(self) -> None:
        now = _time.monotonic()
        if self._loaded is not None and now - self._last_check < self.reload_check_seconds:
            return
        with self._lock:
            self._last_check = now
            try:
                mtime = self.meta_path.stat().st_mtime
            except OSError:
                self._loaded, self._mtime, self.error = None, None, "no_model"
                return
            if self._mtime == mtime and self._loaded is not None:
                return
            try:
                self._loaded = self._load(self.meta_path.read_text(encoding="utf-8"))
                self._mtime, self.error = mtime, ""
            except Exception as exc:
                self._loaded, self._mtime, self.error = None, mtime, f"load_failed:{exc}"

    def predict_with_meta(self, feats: dict[str, float]) -> tuple[float | None, dict]:
        """(P(win), meta of the SAME generation). P is None when no model is loaded."""
        self.maybe_reload()
        loaded = self._loaded
        if loaded is None:
            return None, {}
        booster, meta = loaded
        x = np.array([[feats.get(k, np.nan) for k in FEATURE_NAMES]], dtype=float)
        return float(booster.predict(x)[0]), meta

    def predict(self, feats: dict[str, float]) -> float | None:
        return self.predict_with_meta(feats)[0]


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

@dataclass
class EdgeAIDecision:
    allow: bool
    prob: float | None
    threshold: float | None
    enforced: bool
    reason: str
    features: dict[str, float] = field(default_factory=dict)


class EdgeAIGate:
    """r005 _028 entry point. One instance per process; caches per (code, last 1m bar) so the
    consecutive-confirm polls of _021 within the same minute get the same decision."""

    def __init__(self, mode: str, model_dir: Path, data_root: Path, log_dir: Path, config: StrategyConfig,
                 min_prob_override: float = 0.0):
        self.mode = str(mode).lower()
        self.model = EdgeAIModel(model_dir)
        self.data_root = Path(data_root)
        self.log_dir = Path(log_dir)
        self.config = config
        self.min_prob_override = float(min_prob_override or 0.0)
        self._daily_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._decision_cache: dict[str, tuple[Any, EdgeAIDecision]] = {}

    def _threshold(self, meta: dict) -> float | None:
        if self.min_prob_override > 0:
            return self.min_prob_override
        t = meta.get("threshold")
        return float(t) if valid_prob(t) else None

    def _enforce(self, meta: dict) -> bool:
        if self.mode == "gate":
            return True
        if self.mode == "auto":
            return meta.get("promoted") is True
        return False

    def evaluate(self, code: str, now: datetime, price: float, frame_1min: pd.DataFrame | None,
                 prev_close: float | None, day_open: float | None, day_vwap: float | None,
                 trigger_info: dict | None = None) -> EdgeAIDecision:
        """trigger_info: r005 _008's info dict of the trigger that actually passed (path/bars_since_cross)."""
        if self.mode == "off":
            return EdgeAIDecision(True, None, None, False, "off")
        trade_date = now.strftime("%Y%m%d")
        raw = session_raw_1min(frame_1min, now.date())
        if raw is None or len(raw) < MIN_SESSION_BARS:
            return EdgeAIDecision(True, None, None, False, "insufficient_bars")
        key = raw.index[-1]
        cached = self._decision_cache.get(code)
        if cached is not None and cached[0] == key:
            return cached[1]
        frames = feature_frames(raw)
        if frames is None:
            decision = EdgeAIDecision(True, None, None, False, "fail_open:insufficient_history")
            self._decision_cache[code] = (key, decision)
            self._write_log(code, now, price, key, decision, {})
            return decision
        m1, m3 = frames
        dkey = (str(code).zfill(6), trade_date)
        if dkey not in self._daily_cache:
            self._daily_cache = {k: v for k, v in self._daily_cache.items() if k[1] == trade_date}
            self._daily_cache[dkey] = load_daily_features(self.data_root, dkey[0], trade_date)
        feats = compute_features(m1, m3, pd.Timestamp(now), float(price), prev_close, day_open, day_vwap,
                                 self._daily_cache[dkey], self.config, trigger_info=trigger_info)
        prob, meta = self.model.predict_with_meta(feats)
        threshold = self._threshold(meta)
        enforced = self._enforce(meta)
        if prob is None:
            decision = EdgeAIDecision(True, None, threshold, False, f"fail_open:{self.model.error}", feats)
        elif not valid_prob(prob):
            decision = EdgeAIDecision(True, None, threshold, False, "fail_open:bad_prob", feats)
        elif threshold is None:
            decision = EdgeAIDecision(True, prob, None, False, "fail_open:no_threshold", feats)
        elif prob >= threshold:
            decision = EdgeAIDecision(True, prob, threshold, enforced, "pass", feats)
        else:
            decision = EdgeAIDecision(not enforced, prob, threshold, enforced,
                                      "block" if enforced else "would_block", feats)
        self._decision_cache[code] = (key, decision)
        self._write_log(code, now, price, key, decision, meta)
        return decision

    def _write_log(self, code: str, now: datetime, price: float, bar_label, d: EdgeAIDecision, meta: dict) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": now.isoformat(timespec="seconds"), "code": str(code).zfill(6), "price": price,
                "bar": str(bar_label), "mode": self.mode, "prob": d.prob, "threshold": d.threshold,
                "enforced": d.enforced, "allow": d.allow, "reason": d.reason,
                "model_id": meta.get("model_id"), "promoted": meta.get("promoted"),
                "features": d.features,
            }
            path = self.log_dir / f"edge_ai_{now:%Y%m%d}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


_GATE: EdgeAIGate | None = None
_GATE_LOCK = threading.Lock()


def get_gate(factory: Callable[[], EdgeAIGate]) -> EdgeAIGate:
    global _GATE
    if _GATE is None:
        with _GATE_LOCK:
            if _GATE is None:
                _GATE = factory()
    return _GATE
