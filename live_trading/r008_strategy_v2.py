# -*- coding: utf-8 -*-
"""Strategy V2 - daily uptrend universe + 5m trend context + 1m trigger (pure logic, no broker I/O).

Shared by the backtester (data_simulation/g012_strategy_v2_backtest.py) and the r003 shadow hook,
so both evaluate exactly the same rules.

Layers
1) Daily filter   : D-1 completed daily bars only (MA20 > MA60, slopes, extension, liquidity, ATR band).
2) 5m context     : completed 5m bars only (EMA20 rising, session VWAP, ATR5m).
3) Entry setups   : (a) pullback to EMA20/VWAP after an impulse, (b) opening-range breakout -> retest.
4) 1m trigger     : completed 1m bar reclaims support, closes above the previous 1m high on volume.
5) Exits          : per-position state machine (structure stop, breakeven, partial TP, chandelier trail,
                    5m trend break, time stop, flatten time). Entry state and position state are independent.

Look-ahead rules
- Every function takes only bars that are COMPLETE at decision time. 1m bars are labelled by their END
  time (same convention as r003 _normalize_intraday_frame), 5m buckets are (t-5m, t] and a 5m bucket is
  used only when the last 1m bar end >= bucket end.
- Indicators are causal (ewm/rolling/cumsum), so slicing a full-day precomputed frame at t is identical to
  recomputing on bars <= t. tests/test_r008_strategy_v2.py checks this equivalence.

Update log:
- [2026-10-23] type=feat owner=claude
    summary: 신규 - 전략 V2(일봉 MA20/MA60 추세 필터 + 5분 EMA20/VWAP + 1분 거래량 반등 트리거, 눌림목/돌파 재진입
      셋업 분리, 진입/청산 상태 독립). r001 V2_* 설정 사용, 기본 OFF.
    impact: sim(g012) / live shadow only (V2_LIVE_MODE="shadow"일 때 r003 기록만)
    compatibility: backward-compatible (신규 모듈, 기존 경로 미변경)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def _cfg_get(src: Any, name: str):
    if isinstance(src, dict):
        return src[name]
    return getattr(src, name)


@dataclass(frozen=True)
class V2Config:
    """Frozen copy of the r001 V2_* values (so a backtest grid can override fields with dataclasses.replace)."""

    daily_ma_fast: int = 20
    daily_ma_slow: int = 60
    daily_ma_fast_slope_days: int = 5
    daily_ma_fast_slope_min_pct: float = 0.0
    daily_ma_slow_slope_days: int = 20
    daily_ma_slow_slope_min_pct: float = -1.0
    daily_max_ext_above_ma_fast_pct: float = 15.0
    daily_min_price: float = 1000
    daily_min_avg_amount_krw: float = 3e9
    daily_atr_pct_min: float = 1.5
    daily_atr_pct_max: float = 12.0
    universe_max_symbols: int = 40
    liq_tier_large_krw: float = 5e10
    liq_tier_mid_krw: float = 1e10

    enable_pullback: bool = True
    enable_breakout_retest: bool = True

    m5_ema_period: int = 20
    m5_ema_rising_bars: int = 3
    m5_min_bars: int = 6
    m5_atr_period: int = 14

    pb_impulse_min_atr: float = 1.0
    pb_zone_band_pct: float = 0.25
    pb_max_close_below_support_pct: float = 0.30
    pb_min_depth_atr: float = 0.5
    pb_max_depth_atr: float = 3.0
    pb_arm_expiry_min: int = 30

    bo_opening_range_min: int = 30
    bo_break_vol_ratio: float = 1.5
    bo_break_close_above_pct: float = 0.10
    bo_retest_band_pct: float = 0.40
    bo_retest_max_undercut_pct: float = 0.50
    bo_arm_expiry_min: int = 45

    m1_vol_ma_period: int = 20
    m1_trigger_vol_ratio: float = 1.3
    m1_trigger_break_prev_high: bool = True
    m1_trigger_max_bar_gain_pct: float = 1.5
    m1_trigger_max_above_support_pct: float = 1.5

    stop_buffer_ticks: int = 1
    stop_min_pct: float = 1.0
    stop_max_pct: float = 3.0
    breakeven_at_r: float = 1.0
    tp1_at_r: float = 1.5
    tp1_fraction: float = 0.5
    trail_start_r: float = 1.0
    trail_atr_mult: float = 1.2
    trend_exit_on_5m_close: bool = True
    time_stop_min: int = 45
    time_stop_min_r: float = 0.5
    market_breadth_min_pct: float = 50.0
    entry_start: dt_time = dt_time(9, 5)
    entry_end: dt_time = dt_time(14, 30)
    flatten_at: dt_time = dt_time(15, 15)

    account_equity_krw: float = 10_000_000
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 20.0
    max_open_positions: int = 4
    max_trades_per_symbol: int = 2
    daily_loss_limit_pct: float = 1.5
    max_spread_ticks: int = 2
    max_tick_pct: float = 0.30

    commission_pct: float = 0.015
    sell_tax_pct: float = 0.20
    slippage_ticks_entry: int = 1
    slippage_ticks_exit: int = 1
    slippage_extra_bps_small: float = 5.0

    @classmethod
    def from_module(cls, mod: Any) -> "V2Config":
        """Build from r001 (module) or a dict holding the V2_* names."""
        g = lambda n: _cfg_get(mod, n)  # noqa: E731
        return cls(
            daily_ma_fast=g("V2_DAILY_MA_FAST"), daily_ma_slow=g("V2_DAILY_MA_SLOW"),
            daily_ma_fast_slope_days=g("V2_DAILY_MA_FAST_SLOPE_DAYS"),
            daily_ma_fast_slope_min_pct=g("V2_DAILY_MA_FAST_SLOPE_MIN_PCT"),
            daily_ma_slow_slope_days=g("V2_DAILY_MA_SLOW_SLOPE_DAYS"),
            daily_ma_slow_slope_min_pct=g("V2_DAILY_MA_SLOW_SLOPE_MIN_PCT"),
            daily_max_ext_above_ma_fast_pct=g("V2_DAILY_MAX_EXT_ABOVE_MA_FAST_PCT"),
            daily_min_price=g("V2_DAILY_MIN_PRICE"), daily_min_avg_amount_krw=g("V2_DAILY_MIN_AVG_AMOUNT_KRW"),
            daily_atr_pct_min=g("V2_DAILY_ATR_PCT_MIN"), daily_atr_pct_max=g("V2_DAILY_ATR_PCT_MAX"),
            universe_max_symbols=g("V2_UNIVERSE_MAX_SYMBOLS"),
            liq_tier_large_krw=g("V2_LIQ_TIER_LARGE_KRW"), liq_tier_mid_krw=g("V2_LIQ_TIER_MID_KRW"),
            enable_pullback=g("V2_ENABLE_PULLBACK_SETUP"), enable_breakout_retest=g("V2_ENABLE_BREAKOUT_RETEST_SETUP"),
            m5_ema_period=g("V2_5M_EMA_PERIOD"), m5_ema_rising_bars=g("V2_5M_EMA_RISING_BARS"),
            m5_min_bars=g("V2_5M_MIN_BARS"), m5_atr_period=g("V2_5M_ATR_PERIOD"),
            pb_impulse_min_atr=g("V2_PB_IMPULSE_MIN_ATR"), pb_zone_band_pct=g("V2_PB_ZONE_BAND_PCT"),
            pb_max_close_below_support_pct=g("V2_PB_MAX_CLOSE_BELOW_SUPPORT_PCT"),
            pb_min_depth_atr=g("V2_PB_MIN_DEPTH_ATR"), pb_max_depth_atr=g("V2_PB_MAX_DEPTH_ATR"),
            pb_arm_expiry_min=g("V2_PB_ARM_EXPIRY_MIN"),
            bo_opening_range_min=g("V2_BO_OPENING_RANGE_MIN"), bo_break_vol_ratio=g("V2_BO_BREAK_VOL_RATIO"),
            bo_break_close_above_pct=g("V2_BO_BREAK_CLOSE_ABOVE_PCT"), bo_retest_band_pct=g("V2_BO_RETEST_BAND_PCT"),
            bo_retest_max_undercut_pct=g("V2_BO_RETEST_MAX_UNDERCUT_PCT"), bo_arm_expiry_min=g("V2_BO_ARM_EXPIRY_MIN"),
            m1_vol_ma_period=g("V2_1M_VOL_MA_PERIOD"), m1_trigger_vol_ratio=g("V2_1M_TRIGGER_VOL_RATIO"),
            m1_trigger_break_prev_high=g("V2_1M_TRIGGER_BREAK_PREV_HIGH"),
            m1_trigger_max_bar_gain_pct=g("V2_1M_TRIGGER_MAX_BAR_GAIN_PCT"),
            m1_trigger_max_above_support_pct=g("V2_1M_TRIGGER_MAX_ABOVE_SUPPORT_PCT"),
            stop_buffer_ticks=g("V2_STOP_BUFFER_TICKS"), stop_min_pct=g("V2_STOP_MIN_PCT"), stop_max_pct=g("V2_STOP_MAX_PCT"),
            breakeven_at_r=g("V2_BREAKEVEN_AT_R"), tp1_at_r=g("V2_TP1_AT_R"), tp1_fraction=g("V2_TP1_FRACTION"),
            trail_start_r=g("V2_TRAIL_START_R"), trail_atr_mult=g("V2_TRAIL_ATR_MULT"),
            trend_exit_on_5m_close=g("V2_TREND_EXIT_ON_5M_CLOSE"),
            time_stop_min=g("V2_TIME_STOP_MIN"), time_stop_min_r=g("V2_TIME_STOP_MIN_R"),
            market_breadth_min_pct=g("V2_MARKET_BREADTH_MIN_PCT"),
            entry_start=g("V2_ENTRY_START"), entry_end=g("V2_ENTRY_END"), flatten_at=g("V2_FLATTEN_AT"),
            account_equity_krw=g("V2_ACCOUNT_EQUITY_KRW"), risk_per_trade_pct=g("V2_RISK_PER_TRADE_PCT"),
            max_position_pct=g("V2_MAX_POSITION_PCT"), max_open_positions=g("V2_MAX_OPEN_POSITIONS"),
            max_trades_per_symbol=g("V2_MAX_TRADES_PER_SYMBOL"), daily_loss_limit_pct=g("V2_DAILY_LOSS_LIMIT_PCT"),
            max_spread_ticks=g("V2_MAX_SPREAD_TICKS"), max_tick_pct=g("V2_MAX_TICK_PCT"),
            commission_pct=g("V2_COMMISSION_PCT"), sell_tax_pct=g("V2_SELL_TAX_PCT"),
            slippage_ticks_entry=g("V2_SLIPPAGE_TICKS_ENTRY"), slippage_ticks_exit=g("V2_SLIPPAGE_TICKS_EXIT"),
            slippage_extra_bps_small=g("V2_SLIPPAGE_EXTRA_BPS_SMALL"),
        )


# ---------------------------------------------------------------------------
# KRX tick size (unified KOSPI/KOSDAQ table since 2023-01-25)
# ---------------------------------------------------------------------------

def krx_tick_size(price: float) -> int:
    p = float(price)
    if p < 2000:
        return 1
    if p < 5000:
        return 5
    if p < 20000:
        return 10
    if p < 50000:
        return 50
    if p < 200000:
        return 100
    if p < 500000:
        return 500
    return 1000


def round_to_tick(price: float, direction: str = "down") -> float:
    t = krx_tick_size(price)
    if direction == "up":
        return float(math.ceil(price / t) * t)
    return float(math.floor(price / t) * t)


# ---------------------------------------------------------------------------
# 1) Daily trend filter
# ---------------------------------------------------------------------------

@dataclass
class DailyTrend:
    code: str
    ok: bool
    reason: str
    close: float = float("nan")
    ma_fast: float = float("nan")
    ma_slow: float = float("nan")
    ma_fast_slope_pct: float = float("nan")
    ma_slow_slope_pct: float = float("nan")
    ext_pct: float = float("nan")
    atr_pct: float = float("nan")
    avg_amount: float = float("nan")
    tier: str = ""
    score: float = 0.0


def liquidity_tier(avg_amount: float, cfg: V2Config) -> str:
    if avg_amount >= cfg.liq_tier_large_krw:
        return "large"
    if avg_amount >= cfg.liq_tier_mid_krw:
        return "mid"
    return "small"


def daily_features(daily: pd.DataFrame, cfg: V2Config) -> pd.DataFrame:
    """Per-row causal daily features (row i uses rows <= i only). Columns: close, MA_F, MA_S, MA_F_SLOPE,
    MA_S_SLOPE, EXT, ATR_PCT, AMT20, N (row count so far)."""
    close = daily["close"].astype(float)
    high = daily["high"].astype(float)
    low = daily["low"].astype(float)
    if "amount" in daily.columns and daily["amount"].notna().any():
        amount = daily["amount"].astype(float).fillna(close * daily["volume"].astype(float))
    else:
        amount = close * daily["volume"].astype(float)
    ma_f = close.rolling(cfg.daily_ma_fast).mean()
    ma_s = close.rolling(cfg.daily_ma_slow).mean()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out = pd.DataFrame(index=daily.index)
    out["close"] = close
    out["MA_F"] = ma_f
    out["MA_S"] = ma_s
    out["MA_F_SLOPE"] = (ma_f / ma_f.shift(cfg.daily_ma_fast_slope_days) - 1.0) * 100.0
    out["MA_S_SLOPE"] = (ma_s / ma_s.shift(cfg.daily_ma_slow_slope_days) - 1.0) * 100.0
    out["EXT"] = (close / ma_f - 1.0) * 100.0
    out["ATR_PCT"] = atr / close * 100.0
    out["AMT20"] = amount.rolling(20, min_periods=1).mean()
    out["N"] = np.arange(1, len(daily) + 1)
    return out


def evaluate_daily_trend(code: str, daily: pd.DataFrame, cfg: V2Config) -> DailyTrend:
    """daily: rows strictly BEFORE the trading day (caller must slice), columns open/high/low/close/volume[/amount]."""
    if daily is None or len(daily) == 0:
        return DailyTrend(code, False, "DAILY_SHORT_HISTORY_0")
    return judge_daily_row(code, daily_features(daily, cfg).iloc[-1], cfg)


def judge_daily_row(code: str, row: pd.Series, cfg: V2Config) -> DailyTrend:
    """Judge one daily_features() row (the last completed day before the trading day)."""
    need = max(cfg.daily_ma_slow + cfg.daily_ma_slow_slope_days, cfg.daily_ma_fast + cfg.daily_ma_fast_slope_days)
    if int(row["N"]) < need:
        return DailyTrend(code, False, f"DAILY_SHORT_HISTORY_{int(row['N'])}")
    c = float(row["close"])
    mf = float(row["MA_F"])
    ms = float(row["MA_S"])
    mf_slope = float(row["MA_F_SLOPE"])
    ms_slope = float(row["MA_S_SLOPE"])
    ext = float(row["EXT"])
    atr_pct = float(row["ATR_PCT"])
    avg_amt = float(row["AMT20"])
    tier = liquidity_tier(avg_amt, cfg)
    base = dict(close=c, ma_fast=mf, ma_slow=ms, ma_fast_slope_pct=mf_slope, ma_slow_slope_pct=ms_slope,
                ext_pct=ext, atr_pct=atr_pct, avg_amount=avg_amt, tier=tier)

    def fail(reason: str) -> DailyTrend:
        return DailyTrend(code, False, reason, **base)

    if c < cfg.daily_min_price:
        return fail("DAILY_PRICE_TOO_LOW")
    if not (avg_amt >= cfg.daily_min_avg_amount_krw):
        return fail("DAILY_ILLIQUID")
    if not (c > mf > ms):
        return fail("DAILY_NOT_STACKED_CLOSE_MA20_MA60")
    if not (mf_slope >= cfg.daily_ma_fast_slope_min_pct):
        return fail("DAILY_MA20_FALLING")
    if not (ms_slope >= cfg.daily_ma_slow_slope_min_pct):
        return fail("DAILY_MA60_FALLING")
    if not (ext <= cfg.daily_max_ext_above_ma_fast_pct):
        return fail("DAILY_OVEREXTENDED")
    if not (cfg.daily_atr_pct_min <= atr_pct <= cfg.daily_atr_pct_max):
        return fail("DAILY_ATR_OUT_OF_BAND")
    if krx_tick_size(c) / c * 100.0 > cfg.max_tick_pct:
        return fail("DAILY_TICK_TOO_COARSE")
    # Rank: steady trend (MA20 slope), room (not extended), liquidity. Log-scaled liquidity keeps
    # large caps and small caps comparable instead of large caps always winning.
    score = mf_slope * 2.0 + min(ms_slope, 5.0) - max(ext - 8.0, 0.0) * 0.5 + math.log10(max(avg_amt, 1.0)) - 9.0
    return DailyTrend(code, True, "OK", score=score, **base)


def select_universe(trends: list[DailyTrend], cfg: V2Config) -> list[DailyTrend]:
    ok = [t for t in trends if t.ok]
    ok.sort(key=lambda t: (-t.score, t.code))
    return ok[: cfg.universe_max_symbols]


# ---------------------------------------------------------------------------
# 2) Intraday frames (1m -> indicators, 5m completed buckets)
# ---------------------------------------------------------------------------

def prepare_1m(m1: pd.DataFrame, cfg: V2Config) -> pd.DataFrame:
    """m1 index = bar END timestamps of one session, columns open/high/low/close/volume[/amount].
    Adds VOL_MA (prior bars only), VWAP (session cumulative)."""
    df = m1.sort_index().copy()
    vol = df["volume"].astype(float).clip(lower=0)
    if "amount" in df.columns and df["amount"].notna().all() and float(df["amount"].sum()) > 0:
        amt = df["amount"].astype(float).clip(lower=0)
    else:
        amt = (df["high"] + df["low"] + df["close"]).astype(float) / 3.0 * vol
    cum_v = vol.cumsum()
    df["VWAP"] = (amt.cumsum() / cum_v.replace(0, np.nan)).ffill()
    # Volume MA of PRIOR bars, so a spike bar is compared with what came before it.
    df["VOL_MA"] = vol.shift(1).rolling(cfg.m1_vol_ma_period, min_periods=5).mean()
    return df


def build_5m(m1p: pd.DataFrame, cfg: V2Config) -> pd.DataFrame:
    """Resample prepared 1m bars to 5m buckets (t-5m, t]. Returns all buckets; caller keeps completed ones."""
    agg = m1p.resample("5min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "VWAP": "last"}
    ).dropna(subset=["close"])
    c = agg["close"].astype(float)
    agg["EMA"] = c.ewm(span=cfg.m5_ema_period, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat([agg["high"] - agg["low"], (agg["high"] - prev_c).abs(), (agg["low"] - prev_c).abs()], axis=1).max(axis=1)
    agg["ATR"] = tr.ewm(alpha=1.0 / cfg.m5_atr_period, adjust=False).mean()
    # Volume ratio vs the mean of PRIOR 5m bars of the session (20-bar MA is not available early in the day).
    v = agg["volume"].astype(float)
    agg["VOL_REF"] = v.shift(1).rolling(20, min_periods=3).mean()
    return agg


def completed_5m(m5: pd.DataFrame, last_1m_end: pd.Timestamp) -> pd.DataFrame:
    # Positional slice (index is sorted) == m5[m5.index <= last_1m_end], without a boolean take per call.
    return m5.iloc[: int(m5.index.searchsorted(last_1m_end, side="right"))]


def _tail_records(df: pd.DataFrame, n: int) -> list[dict]:
    """Last n rows as plain dicts (all-float frames -> numpy view, much cheaper than df.iloc[k] per row)."""
    arr = df.iloc[-n:].to_numpy(dtype=float)
    cols = list(df.columns)
    return [dict(zip(cols, row)) for row in arr]


def market_breadth_ok(above_vwap: int, total: int, cfg: V2Config) -> tuple[bool, float]:
    """Breadth gate: share of the V2 universe trading above its session VWAP. Intraday longs fail together
    when the tape turns, so new entries pause when fewer than market_breadth_min_pct are above VWAP."""
    if cfg.market_breadth_min_pct <= 0:
        return True, float("nan")
    if total < 5:
        return True, float("nan")  # too few names to measure; do not block on missing breadth
    pct = above_vwap / total * 100.0
    return pct >= cfg.market_breadth_min_pct, pct


# ---------------------------------------------------------------------------
# 3+4) Entry engine (per symbol, per day)
# ---------------------------------------------------------------------------

@dataclass
class EntrySignal:
    code: str
    setup: str              # "PULLBACK" | "BREAKOUT_RETEST"
    ts: pd.Timestamp        # 1m bar end that fired
    ref_price: float        # trigger bar close
    stop: float
    support: float
    atr5: float
    reason: str


@dataclass
class EntryState:
    """Entry-side state only. Never touched by the exit logic."""

    pb_state: str = "IDLE"          # IDLE | ARMED
    pb_armed_at: pd.Timestamp | None = None
    pb_low: float = float("inf")
    pb_swing_high: float = 0.0
    pb_last_traded_swing_high: float = 0.0
    bo_state: str = "WAIT_LEVEL"    # WAIT_LEVEL | WAIT_BREAK | BROKEN | ARMED | DONE | FAILED
    bo_level: float = float("nan")
    bo_armed_at: pd.Timestamp | None = None
    bo_low: float = float("inf")
    session_high: float = 0.0
    trades_taken: int = 0
    last_reject: str = ""


def _ctx_5m(m5c: pd.DataFrame, cfg: V2Config) -> dict | None:
    if len(m5c) < max(cfg.m5_min_bars, cfg.m5_ema_rising_bars + 1):
        return None
    last = _tail_records(m5c, 1)[0]
    ema_now = float(last["EMA"])
    ema_prev = float(m5c["EMA"].to_numpy()[-1 - cfg.m5_ema_rising_bars])
    return {
        "close": float(last["close"]), "ema": ema_now, "vwap": float(last["VWAP"]), "atr": float(last["ATR"]),
        "ema_rising": ema_now > ema_prev, "ts": m5c.index[-1],
        "vol": float(last["volume"]), "vol_ref": float(last["VOL_REF"]) if pd.notna(last["VOL_REF"]) else float("nan"),
    }


def _trigger_ok(bar, prev, support: float, cfg: V2Config) -> tuple[bool, str]:
    o, h, c = float(bar["open"]), float(bar["high"]), float(bar["close"])
    v = float(bar["volume"])
    vma = float(bar["VOL_MA"]) if pd.notna(bar["VOL_MA"]) else float("nan")
    if not (c > o):
        return False, "TRIG_NOT_BULLISH"
    if not (c > support):
        return False, "TRIG_BELOW_SUPPORT"
    if cfg.m1_trigger_break_prev_high and not (c > float(prev["high"])):
        return False, "TRIG_NO_PREV_HIGH_BREAK"
    if not (vma > 0) or v < cfg.m1_trigger_vol_ratio * vma:
        return False, "TRIG_VOLUME_WEAK"
    if (c / o - 1.0) * 100.0 > cfg.m1_trigger_max_bar_gain_pct:
        return False, "TRIG_BAR_TOO_LONG"
    if (c / support - 1.0) * 100.0 > cfg.m1_trigger_max_above_support_pct:
        return False, "TRIG_CHASE_ABOVE_SUPPORT"
    return True, "OK"


def _stop_from_structure(low: float, ref: float, cfg: V2Config) -> tuple[float, str]:
    stop = low - cfg.stop_buffer_ticks * krx_tick_size(low)
    dist_pct = (ref / stop - 1.0) * 100.0 if stop > 0 else float("inf")
    if dist_pct < cfg.stop_min_pct:
        # Too tight: widen to the minimum instead of rejecting (a 0.2% stop is just noise on KRX ticks).
        stop = round_to_tick(ref * (1.0 - cfg.stop_min_pct / 100.0), "down")
        return stop, "STOP_WIDENED_TO_MIN"
    if dist_pct > cfg.stop_max_pct:
        return stop, "STOP_TOO_WIDE"
    return stop, "OK"


def in_entry_window(ts: pd.Timestamp, cfg: V2Config) -> bool:
    t = ts.time()
    return cfg.entry_start <= t <= cfg.entry_end


def step_entry(
    code: str,
    st: EntryState,
    m1p_upto: pd.DataFrame,
    m5_all: pd.DataFrame,
    cfg: V2Config,
    has_position: bool,
    market_ok: bool = True,
    signal_gate=None,
) -> EntrySignal | None:
    """Advance the entry state with the newest COMPLETED 1m bar (last row of m1p_upto). Call once per new bar.

    m1p_upto: prepare_1m() output sliced to bars <= now (completed only).
    m5_all  : build_5m() output; only buckets ending <= last 1m end are used.
    signal_gate: optional callable(EntrySignal) -> (ok, reason) evaluated BEFORE the signal is committed
              (e.g. tick momentum). A rejected candidate leaves the setup armed and does not count as a trade.
    """
    if len(m1p_upto) < 2:
        return None
    prev, bar = _tail_records(m1p_upto, 2)
    ts = m1p_upto.index[-1]
    st.last_reject = ""  # diagnostic for THIS bar only
    st.session_high = max(st.session_high, float(bar["high"]))
    m5c = completed_5m(m5_all, ts)
    ctx = _ctx_5m(m5c, cfg)

    # ---- opening-range level (breakout setup) ----
    if st.bo_state == "WAIT_LEVEL":
        session_open = ts.normalize() + pd.Timedelta(hours=9)
        or_end = session_open + pd.Timedelta(minutes=cfg.bo_opening_range_min)
        if ts >= or_end:
            or_bars = m1p_upto[(m1p_upto.index > session_open) & (m1p_upto.index <= or_end)]
            if len(or_bars) >= max(3, cfg.bo_opening_range_min // 3):
                st.bo_level = float(or_bars["high"].max())
                st.bo_state = "WAIT_BREAK"
            else:
                st.bo_state = "FAILED"  # too few prints in the opening range: illiquid, skip this setup

    if ctx is None or not (ctx["atr"] > 0):
        st.last_reject = "CTX_NOT_READY"
        return None

    signal: EntrySignal | None = None
    # Setups keep tracking structure while blocked; only the signal is withheld (market_ok = breadth gate).
    allow_new = (not has_position) and market_ok and in_entry_window(ts, cfg) and st.trades_taken < cfg.max_trades_per_symbol
    low = float(bar["low"])
    close = float(bar["close"])

    # ---- Setup 1: pullback ----
    if cfg.enable_pullback:
        support = max(ctx["ema"], ctx["vwap"])
        floor_level = min(ctx["ema"], ctx["vwap"])
        trend_ok = ctx["ema_rising"] and ctx["close"] >= floor_level * (1.0 - cfg.pb_max_close_below_support_pct / 100.0)
        if st.pb_state == "ARMED":
            st.pb_low = min(st.pb_low, low)
            expired = st.pb_armed_at is not None and ts - st.pb_armed_at > pd.Timedelta(minutes=cfg.pb_arm_expiry_min)
            broken = ctx["close"] < floor_level * (1.0 - cfg.pb_max_close_below_support_pct / 100.0)
            too_deep = (st.pb_swing_high - st.pb_low) / ctx["atr"] > cfg.pb_max_depth_atr
            if expired or broken or too_deep or not ctx["ema_rising"]:
                st.pb_state = "IDLE"
                st.pb_low = float("inf")
                st.last_reject = "PB_DISARM_" + ("EXPIRED" if expired else "BROKEN" if broken else "TOO_DEEP" if too_deep else "EMA_FLAT")
        if st.pb_state == "IDLE" and trend_ok:
            swing_high = st.session_high
            impulse_ok = (swing_high - ctx["vwap"]) >= cfg.pb_impulse_min_atr * ctx["atr"]
            fresh_swing = swing_high > st.pb_last_traded_swing_high
            touched = low <= support * (1.0 + cfg.pb_zone_band_pct / 100.0)
            depth_atr = (swing_high - low) / ctx["atr"]
            if impulse_ok and fresh_swing and touched and cfg.pb_min_depth_atr <= depth_atr <= cfg.pb_max_depth_atr:
                st.pb_state = "ARMED"
                st.pb_armed_at = ts
                st.pb_low = low
                st.pb_swing_high = swing_high
            elif touched:
                st.last_reject = ("PB_NO_IMPULSE" if not impulse_ok else "PB_SWING_ALREADY_TRADED" if not fresh_swing
                                  else f"PB_DEPTH_{depth_atr:.1f}ATR_OUT_OF_BAND")
        if st.pb_state == "ARMED" and allow_new:
            ok, why = _trigger_ok(bar, prev, support, cfg)
            if ok:
                stop, stop_note = _stop_from_structure(st.pb_low, close, cfg)
                if stop_note == "STOP_TOO_WIDE":
                    st.last_reject = "PB_STOP_TOO_WIDE"
                else:
                    cand = EntrySignal(code, "PULLBACK", ts, close, stop, support, ctx["atr"],
                                       f"PB swing={st.pb_swing_high:,.0f} low={st.pb_low:,.0f} sup={support:,.1f} {stop_note}")
                    g_ok, g_why = signal_gate(cand) if signal_gate is not None else (True, "")
                    if g_ok:
                        signal = cand
                        st.pb_last_traded_swing_high = st.pb_swing_high
                        st.pb_state = "IDLE"
                        st.pb_low = float("inf")
                    else:
                        st.last_reject = "PB_GATE_" + g_why
            else:
                st.last_reject = "PB_" + why

    # ---- Setup 2: breakout -> retest ----
    if cfg.enable_breakout_retest and signal is None and st.bo_state not in ("WAIT_LEVEL", "DONE", "FAILED"):
        lvl = st.bo_level
        if st.bo_state == "WAIT_BREAK":
            if m5c.index[-1] == ts:
                last5 = _tail_records(m5c, 1)[0]  # evaluate the break only when a new 5m bucket just completed
                vol_ref = float(last5["VOL_REF"]) if pd.notna(last5["VOL_REF"]) else float("nan")
                if (float(last5["close"]) > lvl * (1.0 + cfg.bo_break_close_above_pct / 100.0)
                        and vol_ref > 0 and float(last5["volume"]) >= cfg.bo_break_vol_ratio * vol_ref):
                    st.bo_state = "BROKEN"
        elif st.bo_state == "BROKEN":
            floor_px = lvl * (1.0 - cfg.bo_retest_max_undercut_pct / 100.0)
            # The first retest bar must respect the undercut limit too (low AND close), otherwise it could arm
            # and fire on the same bar after already invalidating the level.
            if close < floor_px or low < floor_px:
                st.bo_state = "FAILED"
                st.last_reject = "BO_FAILED_BREAKOUT"
            elif low <= lvl * (1.0 + cfg.bo_retest_band_pct / 100.0):
                st.bo_state = "ARMED"
                st.bo_armed_at = ts
                st.bo_low = low
        elif st.bo_state == "ARMED":
            st.bo_low = min(st.bo_low, low)
            expired = st.bo_armed_at is not None and ts - st.bo_armed_at > pd.Timedelta(minutes=cfg.bo_arm_expiry_min)
            if close < lvl * (1.0 - cfg.bo_retest_max_undercut_pct / 100.0) or st.bo_low < lvl * (1.0 - cfg.bo_retest_max_undercut_pct / 100.0):
                st.bo_state = "FAILED"
                st.last_reject = "BO_RETEST_LOST_LEVEL"
            elif expired:
                st.bo_state = "BROKEN"  # retest took too long; wait for a fresh touch
                st.bo_low = float("inf")
        if st.bo_state == "ARMED" and allow_new and ctx["ema_rising"]:
            ok, why = _trigger_ok(bar, prev, lvl, cfg)
            if ok:
                stop, stop_note = _stop_from_structure(min(st.bo_low, lvl), close, cfg)
                if stop_note == "STOP_TOO_WIDE":
                    st.last_reject = "BO_STOP_TOO_WIDE"
                else:
                    cand = EntrySignal(code, "BREAKOUT_RETEST", ts, close, stop, lvl, ctx["atr"],
                                       f"BO level={lvl:,.0f} retest_low={st.bo_low:,.0f} {stop_note}")
                    g_ok, g_why = signal_gate(cand) if signal_gate is not None else (True, "")
                    if g_ok:
                        signal = cand
                        st.bo_state = "DONE"
                    else:
                        st.last_reject = "BO_GATE_" + g_why
            else:
                st.last_reject = "BO_" + why

    if signal is not None:
        st.trades_taken += 1
    return signal


# ---------------------------------------------------------------------------
# 5) Exit engine (per position)
# ---------------------------------------------------------------------------

@dataclass
class ExitAction:
    qty: int
    price: float
    reason: str


@dataclass
class V2Position:
    code: str
    setup: str
    entry_ts: pd.Timestamp
    entry_price: float           # actual fill (incl. slippage)
    qty: int
    stop: float
    initial_risk: float          # entry - initial stop (per share, > 0)
    remaining: int = 0
    highest_close: float = 0.0
    be_moved: bool = False
    tp1_done: bool = False
    trail_active: bool = False
    exits: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.remaining <= 0:
            self.remaining = self.qty
        self.highest_close = max(self.highest_close, self.entry_price)

    def r_multiple(self, price: float) -> float:
        return (price - self.entry_price) / self.initial_risk if self.initial_risk > 0 else 0.0


def round_trip_cost_pct(cfg: V2Config) -> float:
    return 2.0 * cfg.commission_pct + cfg.sell_tax_pct


def step_exit(pos: V2Position, bar, ts: pd.Timestamp, m5c: pd.DataFrame, cfg: V2Config) -> list[ExitAction]:
    """Evaluate one COMPLETED 1m bar for an open position. Order inside the bar is pessimistic:
    stop first (using low; gap-through fills at open), then TP1 (using high), then close-based rules.
    Returned prices are raw trigger prices; the caller applies slippage/costs."""
    acts: list[ExitAction] = []
    if pos.remaining <= 0:
        return acts
    o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

    # (1) protective stop
    if l <= pos.stop:
        px = o if o < pos.stop else pos.stop
        reason = "STOP_BE" if pos.be_moved and pos.stop >= pos.entry_price else ("STOP_TRAIL" if pos.trail_active else "STOP_INITIAL")
        acts.append(ExitAction(pos.remaining, px, reason))
        pos.remaining = 0
        return acts

    # (2) partial take profit
    if not pos.tp1_done and cfg.tp1_fraction > 0:
        tp1 = pos.entry_price + cfg.tp1_at_r * pos.initial_risk
        if h >= tp1:
            q = int(math.floor(pos.qty * cfg.tp1_fraction))
            if 0 < q < pos.remaining:
                acts.append(ExitAction(q, max(tp1, o), "TP1"))
                pos.remaining -= q
            pos.tp1_done = True

    # (3) close-based state updates
    pos.highest_close = max(pos.highest_close, c)
    r_now = pos.r_multiple(c)
    if not pos.be_moved and pos.r_multiple(pos.highest_close) >= cfg.breakeven_at_r:
        be = round_to_tick(pos.entry_price * (1.0 + round_trip_cost_pct(cfg) / 100.0), "up")
        pos.stop = max(pos.stop, be)
        pos.be_moved = True
    m5_last = _tail_records(m5c, 1)[0] if len(m5c) else None
    if pos.r_multiple(pos.highest_close) >= cfg.trail_start_r and m5_last is not None and float(m5_last["ATR"]) > 0:
        trail = round_to_tick(pos.highest_close - cfg.trail_atr_mult * float(m5_last["ATR"]), "down")
        if trail > pos.stop:
            pos.stop = trail
            pos.trail_active = True

    # (4) trend break on a just-completed 5m bar
    if cfg.trend_exit_on_5m_close and m5_last is not None and m5c.index[-1] == ts and ts > pos.entry_ts:
        if float(m5_last["close"]) < float(m5_last["EMA"]) and float(m5_last["close"]) < float(m5_last["VWAP"]):
            acts.append(ExitAction(pos.remaining, c, "TREND_BREAK_5M"))
            pos.remaining = 0
            return acts

    # (5) time stop: no progress
    held_min = (ts - pos.entry_ts).total_seconds() / 60.0
    if cfg.time_stop_min > 0 and held_min >= cfg.time_stop_min and pos.r_multiple(pos.highest_close) < cfg.time_stop_min_r and not pos.tp1_done:
        acts.append(ExitAction(pos.remaining, c, "TIME_STOP"))
        pos.remaining = 0
        return acts

    # (6) flatten before the close
    if ts.time() >= cfg.flatten_at:
        acts.append(ExitAction(pos.remaining, c, "EOD_FLATTEN"))
        pos.remaining = 0
        return acts
    _ = r_now
    return acts


# ---------------------------------------------------------------------------
# Position sizing / portfolio guards
# ---------------------------------------------------------------------------

def size_position(entry_price: float, stop: float, equity: float, cfg: V2Config) -> int:
    """Risk-based size: loss at stop (+ round-trip costs) <= risk_per_trade_pct of equity, capped by notional."""
    if not (entry_price > 0 and stop > 0 and entry_price > stop and equity > 0):
        return 0
    per_share_risk = (entry_price - stop) + entry_price * round_trip_cost_pct(cfg) / 100.0
    by_risk = math.floor(equity * cfg.risk_per_trade_pct / 100.0 / per_share_risk)
    by_notional = math.floor(equity * cfg.max_position_pct / 100.0 / entry_price)
    return int(max(0, min(by_risk, by_notional)))


def spread_ok(bid: float | None, ask: float | None, cfg: V2Config) -> tuple[bool, str]:
    """Live-only filter. Missing quotes fail closed for a new entry."""
    if not (bid and ask and bid > 0 and ask >= bid):
        return False, "SPREAD_NO_QUOTE"
    ticks = (ask - bid) / krx_tick_size(bid)
    if ticks > cfg.max_spread_ticks + 1e-9:
        return False, f"SPREAD_{ticks:.0f}T_GT_{cfg.max_spread_ticks}T"
    return True, "OK"


def slippage_price(price: float, side: str, ticks: int, extra_bps: float = 0.0) -> float:
    t = krx_tick_size(price)
    adj = ticks * t + price * extra_bps / 10000.0
    return price + adj if side == "buy" else max(t, price - adj)


@dataclass
class DayRisk:
    """Portfolio-level day state shared by all symbols (entry side reads, exit side writes)."""

    equity: float
    realized_pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""

    def can_open(self, open_positions: int, unrealized_pnl: float, cfg: V2Config) -> tuple[bool, str]:
        if self.halted:
            return False, self.halt_reason
        if (self.realized_pnl + min(unrealized_pnl, 0.0)) <= -self.equity * cfg.daily_loss_limit_pct / 100.0:
            self.halted = True
            self.halt_reason = "DAILY_LOSS_LIMIT"
            return False, self.halt_reason
        if open_positions >= cfg.max_open_positions:
            return False, "MAX_OPEN_POSITIONS"
        return True, "OK"


def fill_cost_krw(side: str, price: float, qty: int, cfg: V2Config) -> float:
    gross = price * qty
    fee = gross * cfg.commission_pct / 100.0
    tax = gross * cfg.sell_tax_pct / 100.0 if side == "sell" else 0.0
    return fee + tax
