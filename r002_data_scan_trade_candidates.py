"""
r002_data_scan_trade_candidates.py

Purpose:
- Read multi-day intraday data, aggregate to daily bars, and rank symbols
  for trading watchlists based on liquidity and stock health.
- Export picks used by live trading and simulation.

Run examples:
- python xgraph/auto_trading/r002_data_scan_trade_candidates.py --date 20260508
- python xgraph/auto_trading/r002_data_scan_trade_candidates.py --date 20260508 --max-picks 20 --config balanced

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <scanner/live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-07-21] type=fix owner=copilot
    summary: scan()의 fallback 채우기 로직이 trend_state==up + score>=cutoff만 확인하고
      down_trend/daily_5d_downtrend/bb_lower_not_uptrend/linreg_long_term_downtrend/
      bearish_2in3d_*/big_bearish_candle 등 다른 모든 하드필터 fail_reason을 무시하던 문제 수정.
      089030(테크윙)이 evaluate_candidate에서 linreg_long_term_downtrend로 확실히 탈락했음에도
      MA5/MA20 단기 반등으로 trend_state=up이 되어 fallback으로 최종 picks에 재유입되는 것을
      2026-07-20 데이터 재스캔에서 확인. FALLBACK_RELAXABLE_FAIL_REASONS(low_score,
      liquidity_below_market_dual, volume_declining)만 fallback이 override 가능하도록 제한.
    impact: scanner
    compatibility: breaking (fallback 유입 후보 감소 - eligible pool이 작을 때 최종 picks 수가
      max_picks보다 적어질 수 있음)
- [2026-07-21] type=feat owner=copilot
    summary: 20일 선형회귀 기울기 기반 장기추세 하드 필터 추가 (_linreg_slope_pct_per_day/_is_long_term_downtrend).
      기존 3~5일 연속하락/연속음봉 체크는 짧은 반등 캔들 1개만 있어도 체인이 끊겨 무력화되는 허점이 있어
      (033160/089030/035720/377300 사례: 반등 하루/이틀로 down_trend, daily_5d_downtrend, 3d_close_downtrend,
      bb_lower_not_uptrend, bearish_2in3d를 모두 통과), window 전체 종가의 OLS 기울기로 짧은 반등에
      흔들리지 않는 장기 하락추세를 별도 판정. 반전 신호 예외 없음(의도적).
    impact: scanner
    compatibility: breaking (기존에 통과하던 장기 하락추세 종목이 새로 배제될 수 있음)
- [2026-07-16] type=feat owner=copilot
    summary: 신규 하드 필터 추가 - 일봉 기준 볼린저밴드 하한선(20일, 2.0 std)이 최근 3거래일
      이상 순상승하지 않는(횡보/우하향 포함) 종목 배제. calc_bb_lower()/_bb_lower_is_uptrend()
      신규 추가, evaluate_candidate()의 3일 연속 하락 필터 다음 단계에 배치.
    impact: scanner
    compatibility: breaking (BB 하한선이 우상향이 아닌 종목은 이제 무조건 배제됨 - 후보 감소 가능)
- [2026-07-01] type=fix owner=copilot
    summary: trend_state==down 하드 필터에 반전 신호 예외 추가 (_has_reversal_signal 또는 신규 _has_volume_thrust_reversal). MA5/MA20 후행성으로 막 반전한 종목(6/29~6/30 유형)이 무조건 배제되던 문제 수정.
    impact: scanner
    compatibility: breaking (more candidates pass when a genuine reversal pattern is detected)
- [2026-07-01] type=feat owner=copilot
    summary: 기본 선별 종목 수 확대 - INTRADAY_CONFIG.max_picks(25->50), BALANCED_CONFIG.max_picks(30->50)
    impact: scanner
    compatibility: backward-compatible (more candidates output)
- [2026-06-28] type=fix owner=copilot
    summary: INTRADAY_CONFIG 강화 - volume_ma20_min(50k->100k), amount_ma20_min(20억->50억), min_up_days_in_5(2->3), volume_trend(0.9->1.0), price_min(3k->4k)
    impact: scanner
    compatibility: breaking (fewer candidates)
- [2026-06-28] type=fix owner=copilot
    summary: INTRADAY_CONFIG 강화 - volume_ma20_min(50k->100k), amount_ma20_min(20억->50억), min_up_days_in_5(2->3), volume_trend(0.9->1.0), price_min(3k->4k), max_picks(30->25)
    impact: scanner
    compatibility: breaking (fewer candidates)
- [2026-06-26] type=fix owner=copilot
    summary: 신규 하드 필터 4개 추가 - (1)연속 상한가 후 급락, (2)대상일 장대 음봉(>=2.0%), (3)3일 연속 종가 하락, (4)3일 중 음봉 2개 이상(bearish_2in3d, 단 최근 종가 > 3일전 종가면 예외)
    impact: scanner
    compatibility: breaking (fewer candidates)
- [2026-06-25] type=fix owner=copilot
    summary: 일봉 5일 연속 하락 하드 필터 추가 (_is_5d_close_downtrend) + 반전 예외 처리 (_has_reversal_signal) + r001 daily CSV 로드 (_load_daily_csv); MA 지연으로 통과되던 우하향 종목 차단
    impact: scanner
    compatibility: breaking (fewer candidates: continuously-declining stocks now hard-fail)
- [2026-06-17] type=fix owner=copilot
    summary: restored min_listing_days as a hard filter; stocks with fewer data days than config.min_listing_days are now disqualified before scoring to prevent unreliable ATR14/MA20/vol_rel_strength estimates from inflating scores.
    impact: scanner
    compatibility: breaking (fewer candidates may pass when data history is short)
- [2026-06-01] type=feat owner=copilot
    summary: replaced static liquidity hard-thresholds with market-relative filters using KOSPI/KOSDAQ average amount and price-adjusted volume benchmark.
    impact: scanner
    compatibility: breaking (candidate eligibility behavior changed)
- [2026-05-27] type=feat owner=copilot
    summary: tuned balanced preset for bullish regime; added recent-pick penalty and diversified top-pool sampling to reduce repeated symbols.
    impact: scanner
    compatibility: breaking (ranking/selection behavior updated)
- [2026-05-25] type=refactor owner=copilot
    summary: removed listing_days scoring/flags; changed near_52w_high and prev_day_gap_risk from hard-fail to score penalties; aligned fallback eligibility.
    impact: scanner
    compatibility: breaking (ranking behavior updated)
- [2026-05-10] type=docs owner=copilot
    summary: added standardized file header and expandable update-log format.
    impact: scanner
    compatibility: backward-compatible
- [2026-05-13] type=refactor owner=copilot
    summary: removed intraday real-time signal conditions; replaced with daily-bar
             health/liquidity screening (listing age, consecutive up days, volume
             trend, 52-week high position, gap risk).
    impact: scanner
    compatibility: breaking (ScannerConfig fields changed, scan() takes data_root)
"""

import argparse
import json
import math
import random
import re
from dataclasses import dataclass, replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScannerConfig:
    name: str
    # Price range
    price_min: int
    price_max: int
    # Volatility: daily ATR / close (e.g. 0.010 = 1% daily range minimum)
    atr_ratio_min: float
    # Liquidity
    volume_ma20_min: int            # daily volume MA20 (shares/day)
    amount_ma20_min: int            # daily trading amount MA20 (KRW/day)
    # Stock health
    min_listing_days: int           # minimum trading days on record
    min_up_days_in_5: int           # minimum bullish candles (close > open) in last 5 days
    max_52w_high_ratio: float       # risk-zone threshold if price >= ratio * 52-week high
    max_prev_day_change: float      # exclude if previous day abs return >= this
    volume_trend_min_ratio: float   # recent 5d avg vol / prior 5d avg vol minimum
    recent_pick_penalty_per_day: float  # score penalty per recent-day repeat
    recent_pick_penalty_lookback_days: int  # number of prior trading days to check
    diversified_pick_pool_mult: int  # top-pool multiplier for diversified sampling
    # Output
    max_picks: int | None


STRICT_CONFIG = ScannerConfig(
    name="strict",
    price_min=2_000,
    price_max=1_000_000,
    atr_ratio_min=0.015,            # 1.5% daily ATR minimum
    volume_ma20_min=200_000,        # 20만주/일
    amount_ma20_min=10_000_000_000,  # 100억원/일
    min_listing_days=15,
    min_up_days_in_5=3,
    max_52w_high_ratio=0.90,        # 52주 고가 근접 리스크 구간 시작
    max_prev_day_change=0.07,       # 전일 7% 이상 급등락 제외
    volume_trend_min_ratio=1.0,     # 거래량 감소 종목 제외
    recent_pick_penalty_per_day=3.2,
    recent_pick_penalty_lookback_days=4,
    diversified_pick_pool_mult=2,
    max_picks=None,
)

BALANCED_CONFIG = ScannerConfig(
    name="balanced",
    price_min=2_000,
    price_max=1_000_000,
    atr_ratio_min=0.008,
    volume_ma20_min=50_000,
    amount_ma20_min=5_000_000_000,    # 50억원/일
    min_listing_days=10,
    min_up_days_in_5=2,
    max_52w_high_ratio=0.98,        # 강세장에서는 신고가 근접 허용폭 확대
    max_prev_day_change=0.12,
    volume_trend_min_ratio=1.0,
    recent_pick_penalty_per_day=3.5,
    recent_pick_penalty_lookback_days=4,
    diversified_pick_pool_mult=3,
    max_picks=50,
)

RELAXED_CONFIG = ScannerConfig(
    name="relaxed",
    price_min=2_000,
    price_max=1_000_000,
    atr_ratio_min=0.007,
    volume_ma20_min=20_000,
    amount_ma20_min=1_000_000_000,    # 10억원/일
    min_listing_days=5,
    min_up_days_in_5=1,
    max_52w_high_ratio=0.97,        # 52주 고가 근접 리스크 구간 시작
    max_prev_day_change=0.10,
    volume_trend_min_ratio=0.8,
    recent_pick_penalty_per_day=2.0,
    recent_pick_penalty_lookback_days=3,
    diversified_pick_pool_mult=4,
    max_picks=20,
)

INTRADAY_CONFIG = ScannerConfig(
    name="intraday",
    price_min=4_000,              # 저가주 제외 (3000->4000, 유동성 부족 차단 강화)
    price_max=300_000,
    atr_ratio_min=0.010,          # 일중 1% 이상 변동 필수
    volume_ma20_min=100_000,      # 50,000->100,000주/일 (유동성 기준 강화)
    amount_ma20_min=5_000_000_000, # 20억->50억원/일 (거래대금 기준 강화)
    min_listing_days=10,          # 5->10거래일 (신뢰할 수 있는 기술적 지표 최소 기간)
    min_up_days_in_5=3,           # 2->3: 5일 중 3일 이상 양봉 (우상향 필수)
    max_52w_high_ratio=0.99,      # 52주 신고가 직전까지 허용
    max_prev_day_change=0.09,     # 전일 12%->9% 이상 급등락 제외
    volume_trend_min_ratio=1.0,   # 0.9->1.0: 거래량 증가 추세 필수 (감소 종목 배제)
    recent_pick_penalty_per_day=3.0,
    recent_pick_penalty_lookback_days=3,
    diversified_pick_pool_mult=3,
    max_picks=50,                 # 25->50 (반전 신호 종목 등 후보군 확대)
)

CONFIG_MAP = {
    "strict": STRICT_CONFIG,
    "balanced": BALANCED_CONFIG,
    "relaxed": RELAXED_CONFIG,
    "intraday": INTRADAY_CONFIG,
}

DEFAULT_CONFIG = INTRADAY_CONFIG
DEFAULT_HISTORY_WINDOW = 0
DAILY_LOOKBACK = 260  # trading days of history to load per stock
MIN_REQUIRED_BARS = 1
MAX_PICKS_LIMIT = 50
SCORE_CUTOFF = 30.0
LIQUIDITY_RELAX_FACTOR = 0.70
LOW_UP_DAYS_TOLERANCE = 1
FALLBACK_RELAXABLE_FAIL_REASONS = {
    "low_score", "liquidity_below_market_dual", "volume_declining",
}  # fallback may override only these; trend/candle-pattern fail_reasons block rescue


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def ensure_datetime_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    for col in ["date", "Date", "datetime", "Datetime", "time", "Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.set_index(col)
            return df

    return df


def _extract_code_from_stem(stem):
    """Extract 6-digit symbol code from file stem.

    Supported examples:
    - 005930
    - 005930_삼성전자
    - 005930_삼성전자_1m
    - 005930(삼성전자)
    """
    match = re.match(r"^(\d{6})(?:[_(].*)?$", str(stem))
    return match.group(1) if match else None


def _symbol_file_priority(path):
    """Prefer 1m files over others, and avoid legacy 20s when possible."""
    stem = path.stem.lower()
    if stem.endswith("_1m"):
        return 0
    if stem.endswith("_20s"):
        return 2
    return 1


def resolve_symbol_file(data_dir, code):
    """Return best matching symbol file path among supported naming formats."""
    # 1) Legacy exact names first
    txt_path = data_dir / f"{code}.txt"
    if txt_path.exists():
        return txt_path

    csv_path = data_dir / f"{code}.csv"
    if csv_path.exists():
        return csv_path

    # 2) New names: {code}_{name}.txt, {code}_{name}_1m.txt, etc.
    candidates = []
    candidates.extend(data_dir.glob(f"{code}*.txt"))
    candidates.extend(data_dir.glob(f"{code}*.csv"))
    candidates = [p for p in candidates if _extract_code_from_stem(p.stem) == code]
    if not candidates:
        return None

    return sorted(candidates, key=_symbol_file_priority)[0]


def load_data(code, data_dir, warn=True):
    file_path = resolve_symbol_file(data_dir, code)

    if file_path is None:
        if warn:
            print(f"[WARN] {code} 파일 없음 (.txt/.csv)")
        return None

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        return df
    except Exception as exc:
        if warn:
            print(f"[WARN] {code} 데이터 로드 실패: {exc}")
        return None


def build_daily_bars(data_root, code, target_date_str=None, lookback=DAILY_LOOKBACK, single_date_only=False):
    """Aggregate per-date intraday files into a daily OHLCV DataFrame.

    Each date directory under data_root is expected to contain intraday
    minute-bar CSV files named {code}.txt.  This function reads each file,
    aggregates to a single daily bar (open=first, high=max, low=min,
    close=last, volume=sum), and returns a sorted daily DataFrame.
    """
    if single_date_only and target_date_str:
        target_dir = data_root / target_date_str
        date_dirs = [target_dir] if target_dir.is_dir() else []
    else:
        date_dirs = sorted(
            d for d in data_root.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        )
        if target_date_str:
            date_dirs = [d for d in date_dirs if d.name <= target_date_str]
    date_dirs = date_dirs[-lookback:]

    rows = []
    for date_dir in date_dirs:
        df = load_data(code, date_dir, warn=False)
        if df is None or df.empty:
            continue
        df = ensure_datetime_index(df)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index.year != 1900]
        if df.empty:
            continue
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        if df.empty:
            continue
        rows.append({
            "date": pd.to_datetime(date_dir.name, format="%Y%m%d"),
            "open": df["open"].iloc[0] if "open" in df.columns else df["close"].iloc[0],
            "high": df["high"].max() if "high" in df.columns else df["close"].max(),
            "low": df["low"].min() if "low" in df.columns else df["close"].min(),
            "close": df["close"].iloc[-1],
            "volume": df["volume"].sum() if "volume" in df.columns else 0.0,
        })

    if not rows:
        return None

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    daily["amount"] = daily["close"] * daily["volume"]
    return daily


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def calc_atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def calc_bb_lower(daily_df, period=20, std_mult=2.0):
    """Daily-bar Bollinger Band lower line (MA - std_mult * rolling STD)."""
    ma = daily_df["close"].rolling(period, min_periods=1).mean()
    std = daily_df["close"].rolling(period, min_periods=1).std()
    return ma - std_mult * std


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def classify_trend(ma5, ma20, ma5_slope_3d=None):
    if ma5 is None or ma20 is None:
        return "unknown"
    # Require both MA alignment and short-term MA5 slope direction.
    if ma5 > ma20 and (ma5_slope_3d is None or ma5_slope_3d > 0):
        return "up"
    if ma5 < ma20 and (ma5_slope_3d is None or ma5_slope_3d < 0):
        return "down"
    return "flat"


def calc_volume_trend_ratio(daily_df, window=10):
    """Ratio of recent half-window avg volume vs prior half-window avg volume.

    > 1.0  volume is increasing (bullish signal).
    < 1.0  volume is declining.
    Falls back to available data when fewer than window days exist (min 2).
    """
    vols = daily_df["volume"].tail(window)
    n = len(vols)
    if n < 2:
        return None
    if n < window:
        half = max(1, n // 2)
        recent_avg = float(vols.tail(half).mean())
        prior_avg = float(vols.head(n - half).mean())
    else:
        half = window // 2
        recent_avg = float(vols.tail(half).mean())
        prior_avg = float(vols.head(half).mean())
    if prior_avg == 0:
        return None
    return recent_avg / prior_avg


def calc_volume_relative_strength(daily_df, recent_window=5, base_window=20):
    """Recent volume strength vs prior base window.

    ratio = mean(volume, recent_window) / mean(volume, prior_base_window)
    where prior_base_window excludes recent_window bars.
    """
    need = recent_window + base_window
    vols = daily_df["volume"].tail(need)
    if len(vols) < need:
        return None
    recent_avg = float(vols.tail(recent_window).mean())
    prior_avg = float(vols.head(base_window).mean())
    if prior_avg == 0:
        return None
    return recent_avg / prior_avg


# ---------------------------------------------------------------------------
# Daily trend filter helpers
# ---------------------------------------------------------------------------

def _is_5d_close_downtrend(df: pd.DataFrame) -> bool:
    """True when the last 5 daily closes form an unbroken consecutive decline.
    Catches downtrends that MA5/MA20 comparison misses due to lag.
    """
    closes = df["close"].dropna()
    if len(closes) < 5:
        return False
    tail = closes.tail(5).values
    return all(tail[i] > tail[i + 1] for i in range(4))


def _has_reversal_signal(df: pd.DataFrame) -> bool:
    """True when bottom reversal detected:
    - 2 consecutive bullish candles (close > open), OR
    - 2 consecutive rising closes.
    """
    if len(df) < 2:
        return False
    last2 = df.tail(2)
    if "open" in df.columns and all(last2["close"].values > last2["open"].values):
        return True
    closes = df["close"].dropna().tail(3).values
    return len(closes) >= 3 and float(closes[-1]) > float(closes[-2]) > float(closes[-3])


def _has_volume_thrust_reversal(df: pd.DataFrame) -> bool:
    """True when the last bar is a single strong V-bottom reversal day:
    - Bullish body >= 3% of open
    - Volume >= 1.5x the prior 5-day average (money-flow thrust)
    - Closes above the prior day's high (breaks immediate overhead resistance)
    - Closes above the 5-day MA (recovers above short-term average)
    Catches sharp one-day breakouts that _has_reversal_signal's 2-day
    pattern misses (e.g. down day followed by a single large up day).
    """
    if len(df) < 6:
        return False
    last = df.iloc[-1]
    prior5 = df.iloc[-6:-1]
    open_p = float(last.get("open") or 0)
    close_p = float(last.get("close") or 0)
    if open_p <= 0 or close_p <= open_p:
        return False
    body_pct = (close_p - open_p) / open_p
    avg_vol5 = float(prior5["volume"].mean())
    ma5_close = float(df["close"].tail(5).mean())
    last_volume = float(last.get("volume") or 0)
    prior_day_high = float(prior5["high"].iloc[-1])
    return (
        body_pct >= 0.03
        and avg_vol5 > 0
        and last_volume >= avg_vol5 * 1.5
        and close_p > prior_day_high
        and close_p > ma5_close
    )

def _load_daily_csv(code: str, data_root: Path, target_date_str) -> "pd.DataFrame | None":
    """Load {code}_*_daily.csv saved by r001 from the target date directory.
    Returns None if file not found or unreadable.
    """
    if not target_date_str:
        return None
    date_dir = data_root / str(target_date_str)
    if not date_dir.is_dir():
        return None
    matches = sorted(date_dir.glob(f"{code}_*_daily.csv")) + sorted(date_dir.glob(f"{code}_daily.csv"))
    if not matches:
        return None
    try:
        df = pd.read_csv(matches[0])
        if "date" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df if not df.empty else None
    except Exception:
        return None


def _is_3d_close_downtrend(df):
    """True when the last 3 daily closes form an unbroken consecutive decline."""
    closes = df["close"].dropna()
    if len(closes) < 4:
        return False
    tail = closes.tail(4).values
    return all(tail[i] > tail[i + 1] for i in range(3))


def _bb_lower_is_uptrend(df, lookback_days=3):
    """True when the daily BB-lower line has risen net over the last
    `lookback_days` trading days (BB_LOWER[-1] > BB_LOWER[-1-lookback_days]).
    Requires at least lookback_days+1 valid bars; treated as not-uptrend
    (conservative exclusion) when there isn't enough history yet.
    """
    bb_lower = calc_bb_lower(df).dropna()
    if len(bb_lower) < lookback_days + 1:
        return False
    return float(bb_lower.iloc[-1]) > float(bb_lower.iloc[-1 - lookback_days])


def _linreg_slope_pct_per_day(df, window=20, min_bars=15):
    """OLS(최소제곱법) slope of the last `window` daily closes, expressed as
    %/day relative to the window's mean close. Unlike the short (3~5-day)
    consecutive-decline checks, this fits every bar in the window so a
    1~2 day bounce inside a longer decline cannot reset/hide the trend.
    """
    closes = df["close"].dropna().tail(window)
    n = len(closes)
    if n < min_bars:
        return None
    y = closes.to_numpy(dtype=float)
    x = np.arange(n, dtype=float)
    mean_close = float(y.mean())
    if mean_close <= 0:
        return None
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / mean_close


def _is_long_term_downtrend(df, window=20, min_bars=15, slope_pct_per_day_threshold=-0.003):
    """True when the `window`-day linear-regression slope of daily closes is
    persistently negative (default: -0.3%/day, roughly -6% net over 20
    trading days). Intended to catch multi-week downtrends that a brief
    1~2 day bounce would otherwise hide from the short-window (3~5 day)
    filters and from the MA5/MA20 trend_state classification. No reversal-
    signal exception: that exception is exactly what let a short bounce
    mask a longer downtrend in the 2026-07-20 scan (033160/089030/035720/
    377300).
    """
    slope_pct = _linreg_slope_pct_per_day(df, window=window, min_bars=min_bars)
    if slope_pct is None:
        return False
    return slope_pct <= slope_pct_per_day_threshold


def _has_upperlimit_streak_then_crash(df, lookback=15, upperlimit_min_pct=0.20, crash_pct=0.10):
    """연속 상한가(2일 이상, +20% 기준) 이후 급락(-10% 이상) 패턴 감지. pump & dump / 테마 급등락 종목 차단."""
    if len(df) < 4:
        return False
    closes = df["close"].dropna().tail(lookback)
    if len(closes) < 4:
        return False
    pct = closes.pct_change().fillna(0).values
    for i in range(len(pct) - 2):
        if pct[i] >= upperlimit_min_pct and pct[i + 1] >= upperlimit_min_pct:
            for j in range(i + 2, len(pct)):
                if pct[j] <= -crash_pct:
                    return True
    return False


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(code, name, daily_df, config, recent_pick_count=0, daily_hist=None, w52_info=None):

    candidate = {
        "code": code,
        "name": name,
        "eligible": False,
        "skip_reason": None,
        "fail_reasons": [],
        "soft_flags": [],
        "price": None,
        "atr": None,
        "atr_ratio": None,
        "ma5": None,
        "ma20": None,
        "ma_gap": None,
        "trend_state": "unknown",
        "vol_ma20": None,
        "amount_ma20": None,
        "listing_days": None,
        "up_days_in_5": None,
        "vol_trend_ratio": None,
        "vol_rel_strength": None,
        "high_52w_ratio": None,
        "near_52w_high_override": False,
        "prev_day_change": None,
        "is_last_bearish": None,
        "close_3d_return": None,
        "market": "unknown",
        "liquidity_amount_benchmark": None,
        "liquidity_volume_benchmark": None,
        "repeat_recent_days": int(recent_pick_count or 0),
        "score": 0.0,
    }

    # --- Basic metrics (must be defined before use) ---
    if daily_df is None or len(daily_df) < MIN_REQUIRED_BARS:
        candidate["skip_reason"] = "insufficient_daily_bars"
        return candidate

    price = safe_float(daily_df["close"].iloc[-1])
    if price is None or price <= 0:
        candidate["skip_reason"] = "invalid_price"
        return candidate

    # r001이 KIS inquire_daily_itemchartprice API로 받아온 실제 서버 일봉(daily_hist)이
    # 있으면 이를 우선 사용한다. 로컬 인트라데이 파일을 date-dir별로 집계한 daily_df는
    # data_root에 실제로 존재하는 날짜 폴더 수만큼만 히스토리가 쌓이므로(현재 약 13~14일)
    # MA20/ATR14/추세 판정이 짧은 창에서 왜곡될 수 있다. price(당일 실거래가)만은 실시간
    # 인트라데이 집계인 daily_df를 그대로 쓰고, 그 외 이력 기반 지표는 _check_df를 쓴다.
    _check_df = daily_hist if daily_hist is not None else daily_df

    # Daily ATR (14-day); fall back to single-bar range when history is short
    atr_series = calc_atr(_check_df)
    atr = safe_float(atr_series.iloc[-1]) if not atr_series.empty else None
    if atr is None:
        # Use intraday range of the most recent bar as a volatility proxy
        last_high = safe_float(_check_df["high"].iloc[-1])
        last_low = safe_float(_check_df["low"].iloc[-1])
        if last_high is not None and last_low is not None and last_high > last_low:
            atr = last_high - last_low
    atr_ratio = (atr / price) if (atr is not None and price > 0) else None

    # Daily moving averages
    ma5 = safe_float(_check_df["close"].rolling(5, min_periods=1).mean().iloc[-1])
    ma20 = safe_float(_check_df["close"].rolling(20, min_periods=1).mean().iloc[-1])
    ma_gap = (ma5 - ma20) if (ma5 is not None and ma20 is not None) else None

    # Liquidity
    vol_ma20 = safe_float(_check_df["volume"].rolling(20, min_periods=1).mean().iloc[-1])
    amount_ma20 = safe_float(_check_df["amount"].rolling(20, min_periods=1).mean().iloc[-1])

    # Listing age (number of trading days with data)
    listing_days = len(_check_df)

    # Consecutive up days: close > open in last 5 trading days
    last5 = _check_df.tail(5)
    up_days_in_5 = int((last5["close"] > last5["open"]).sum())

    # Volume trend ratio (recent 5d vs prior 5d)
    vol_trend_ratio = calc_volume_trend_ratio(_check_df, window=10)
    # Relative strength: recent 5d avg volume vs prior 20d avg volume
    vol_rel_strength = calc_volume_relative_strength(_check_df, recent_window=5, base_window=20)

    # 52-week high position: prefer KIS inquire_price server value (w52_hgpr, real
    # 52-week high) over the local daily-bar max, which only spans a few weeks and
    # understates how far a stock has fallen from its real 52-week high.
    week52_high = None
    if w52_info and w52_info.get("w52_high"):
        week52_high = safe_float(w52_info.get("w52_high"))
    if week52_high is None:
        lookback_52w = min(252, len(_check_df))
        week52_high = safe_float(_check_df["high"].tail(lookback_52w).max())
    high_52w_ratio = (price / week52_high) if (week52_high is not None and week52_high > 0) else None

    # Previous day absolute return (gap / surge risk)
    prev_day_change = None
    if len(_check_df) >= 2:
        prev_close = safe_float(_check_df["close"].iloc[-2])
        if prev_close is not None and prev_close > 0:
            prev_day_change = abs(price - prev_close) / prev_close

    ma5_series = _check_df["close"].rolling(5, min_periods=1).mean()
    ma5_slope_3d = None
    if len(ma5_series) >= 4:
        ma5_slope_3d = safe_float(ma5_series.iloc[-1] - ma5_series.iloc[-4])

    trend_state = classify_trend(ma5, ma20, ma5_slope_3d)

    # Near 52w-high override: keep strong momentum/liquidity names in play.
    near_52w_overextended = (
        high_52w_ratio is not None and high_52w_ratio >= config.max_52w_high_ratio
    )
    support_signals = 0
    if trend_state == "up":
        support_signals += 1
    if ma_gap is not None and ma_gap > 0:
        support_signals += 1
    if up_days_in_5 >= max(config.min_up_days_in_5, 3):
        support_signals += 1
    if vol_rel_strength is not None and vol_rel_strength >= 1.2:
        support_signals += 1
    if vol_trend_ratio is not None and vol_trend_ratio >= 1.0:
        support_signals += 1
    near_52w_high_override = near_52w_overextended and support_signals >= 3

    is_last_bearish = False
    if len(_check_df) >= 1:
        last = _check_df.iloc[-1]
        is_last_bearish = bool(last["close"] < last["open"])

    close_3d_return = None
    if len(_check_df) >= 3:
        c0 = safe_float(_check_df["close"].iloc[-3])
        c2 = safe_float(_check_df["close"].iloc[-1])
        if c0 is not None and c0 > 0 and c2 is not None:
            close_3d_return = (c2 / c0) - 1.0

    candidate.update({
        "price": price,
        "atr": atr,
        "atr_ratio": atr_ratio,
        "ma5": ma5,
        "ma20": ma20,
        "ma_gap": ma_gap,
        "trend_state": trend_state,
        "vol_ma20": vol_ma20,
        "amount_ma20": amount_ma20,
        "listing_days": listing_days,
        "up_days_in_5": up_days_in_5,
        "vol_trend_ratio": vol_trend_ratio,
        "vol_rel_strength": vol_rel_strength,
        "high_52w_ratio": high_52w_ratio,
        "near_52w_high_override": near_52w_high_override,
        "prev_day_change": prev_day_change,
        "is_last_bearish": is_last_bearish,
        "close_3d_return": close_3d_return,
    })

    # --- Hard filters (fail = disqualified) ---
    # [최적화] 절대적인 거부 조건만 하드 필터로 남깁니다.
    if price < config.price_min:
        candidate["fail_reasons"].append("price_floor")
    if price > config.price_max:
        candidate["fail_reasons"].append("price_ceiling")

    # 데이터 일수 최소 기준: 부족하면 ATR14/MA20/vol_rel_strength 등 핵심 지표가
    # 단축 계산(min_periods=1/fallback)되어 신뢰할 수 없음.
    if listing_days < config.min_listing_days:
        candidate["fail_reasons"].append("insufficient_listing_days")

    # [추가] 모멘텀 스캐너이므로 역배열(하락추세) 종목은 스코어와 상관없이 확실히 배제합니다.
    # 단, MA5/MA20는 후행지표라 6/29~6/30 사례처럼 막 반전한 종목을 놓칠 수 있어
    # 일봉 반전 신호(2일 연속 양봉/종가상승, 또는 거래량 동반 단일 V자 반등)가 있으면 예외 허용.
    trend_reversal_detected = _has_reversal_signal(_check_df) or _has_volume_thrust_reversal(_check_df)
    if trend_state == "down":
        if trend_reversal_detected:
            candidate["soft_flags"].append("down_trend_reversal_override")
        else:
            candidate["fail_reasons"].append("down_trend")

    # 일봉 5일 연속 하락 하드 필터 (MA 지연으로 통과되는 우하향 종목 차단)
    # 단, 최근 2일 연속 양봉 또는 2일 연속 종가 상승이면 바닥 반전으로 간주하여 통과
    if _is_5d_close_downtrend(_check_df) and not _has_reversal_signal(_check_df):
        candidate["fail_reasons"].append("daily_5d_downtrend")

    # 20일 선형회귀 기울기 기반 장기추세 하드 필터 (반전 신호 예외 없음)
    # 짧은 반등 캔들 1~2개로는 무너지지 않는 window 전체 OLS 기울기 기준이라,
    # 3~5일 연속하락 체크만으로는 잡히지 않는 다주(多週) 하락추세를 별도로 차단한다.
    if _is_long_term_downtrend(_check_df):
        candidate["fail_reasons"].append("linreg_long_term_downtrend")

    # 연속 상한가(2일 이상) 이후 급락 하드 필터 - pump & dump / 테마 급등락 차단
    if _has_upperlimit_streak_then_crash(_check_df):
        candidate["fail_reasons"].append("upperlimit_streak_crash")

    # 대상일 일봉 장대 음봉 하드 필터 (close < open, body >= 2.0%)
    if len(daily_df) >= 1:
        _last = daily_df.iloc[-1]
        _open_p = float(_last.get("open") or 0)
        _close_p = float(_last.get("close") or 0)
        if _open_p > 0 and _close_p < _open_p and (_open_p - _close_p) / _open_p >= 0.020:
            candidate["fail_reasons"].append("big_bearish_candle")

    # 이전 3 거래일 연속 종가 하락 하드 필터 (반전 신호 없을 때)
    if _is_3d_close_downtrend(_check_df) and not _has_reversal_signal(_check_df):
        candidate["fail_reasons"].append("3d_close_downtrend")

    # 볼린저밴드 하한선 우상향 필터: 최근 3거래일 이상 일봉 기준 BB 하한선이
    # 순상승하지 않으면(횡보/우하향 포함) 배제
    if not _bb_lower_is_uptrend(_check_df, lookback_days=3):
        candidate["fail_reasons"].append("bb_lower_not_uptrend")

    # 이전 3 거래일 중 음봉 2개 이상 하드 필터
    # 예외: 1거래일(최근) 종가 > 3거래일 종가이면 전반적 상승 중 조정으로 간주하여 통과
    if len(_check_df) >= 3 and "open" in _check_df.columns:
        _last3 = _check_df.tail(3)
        _bearish_count = int((_last3["close"] < _last3["open"]).sum())
        if _bearish_count >= 2:
            _c1 = float(_last3["close"].iloc[-1])   # 최근(1거래일) 종가
            _c3 = float(_last3["close"].iloc[0])    # 3거래일 전 종가
            if not (_c3 > 0 and _c1 > _c3):
                candidate["fail_reasons"].append(f"bearish_2in3d_{_bearish_count}")

    # --- [최적화] 하드 필터에서 -> 스코어/소프트 플래그 처리로 이관된 항목들 ---
    # 기존에 'atr_ratio'와 'low_up_days'로 즉시 탈락시키던 것을 제거하고 soft_flags로 전환하여 
    # 종합 점수(SCORE_CUTOFF)에 의해 유연하게 필터링되도록 합니다. (Or 조건화)
    if atr_ratio is None or atr_ratio < config.atr_ratio_min:
        candidate["soft_flags"].append("low_volatility_atr")
        
    min_up_days_required = max(0, config.min_up_days_in_5 - LOW_UP_DAYS_TOLERANCE)
    if len(daily_df) >= 5 and up_days_in_5 < min_up_days_required:
        candidate["soft_flags"].append("low_up_days_warning")

    # volume_trend_min_ratio 미달 종목: INTRADAY_CONFIG(=1.0)에서는 하드 필터로 차단
    # 거래량 감소 추세면 단기 모멘텀 매매 대상 제외
    if vol_trend_ratio is not None and vol_trend_ratio < config.volume_trend_min_ratio:
        candidate["fail_reasons"].append("volume_declining")

    # --- Soft flags (warning only, not disqualified) ---
    if trend_state == "flat":
        candidate["soft_flags"].append("flat_trend")
    if high_52w_ratio is not None and high_52w_ratio < 0.4:
        candidate["soft_flags"].append("far_from_52w_high")
    if high_52w_ratio is not None and high_52w_ratio >= config.max_52w_high_ratio:
        if near_52w_high_override:
            candidate["soft_flags"].append("near_52w_high_override")
        else:
            candidate["soft_flags"].append("near_52w_high")
    if prev_day_change is not None and prev_day_change >= config.max_prev_day_change:
        candidate["soft_flags"].append("prev_day_gap_risk")

    # 시장상대 유동성 필터(apply_market_relative_liquidity_filters)는 
    # 함수 외부(scan)에서 실행되며 'liquidity_below_market_dual'을 fail_reasons에 추가하므로 그대로 유지됩니다.

    # 1차 스코어 계산 후 cutoff 검증
    candidate["score"] = calculate_candidate_score(candidate, config)
    if candidate["score"] < SCORE_CUTOFF:
        candidate["fail_reasons"].append("low_score")
        
    candidate["eligible"] = not candidate["fail_reasons"]
    return candidate


def calculate_candidate_score(candidate, config):
    price = candidate.get("price")
    atr_ratio = candidate.get("atr_ratio")
    vol_ma20 = candidate.get("vol_ma20")
    amount_ma20 = candidate.get("amount_ma20")
    ma5 = candidate.get("ma5")
    ma20 = candidate.get("ma20")
    up_days = candidate.get("up_days_in_5") or 0
    vol_trend = candidate.get("vol_trend_ratio")
    vol_rel_strength = candidate.get("vol_rel_strength")
    high_52w_ratio = candidate.get("high_52w_ratio")
    near_52w_high_override = bool(candidate.get("near_52w_high_override"))
    prev_day_change = candidate.get("prev_day_change")
    repeat_recent_days = int(candidate.get("repeat_recent_days") or 0)
    is_last_bearish = bool(candidate.get("is_last_bearish"))

    if None in (price, atr_ratio, vol_ma20, amount_ma20):
        return 0.0

    score = 0.0

    # Volatility (max 22): reward above threshold with diminishing returns.
    atr_ratio_norm = max(0.0, atr_ratio / config.atr_ratio_min)
    score += min(22.0, 10.0 * math.log1p(atr_ratio_norm * 1.8))

    # Liquidity volume (max 18): use relative strength vs prior 20-day baseline.
    # 1.0 means neutral; below 0.8 gives little/no contribution, >=1.6 saturates.
    if vol_rel_strength is not None:
        score += max(0.0, min(18.0, (vol_rel_strength - 0.8) * 22.5))

    # Liquidity amount (max 18): smooth by log scale to reduce score clustering.
    amount_benchmark = candidate.get("liquidity_amount_benchmark")
    if amount_benchmark in (None, 0):
        amount_benchmark = config.amount_ma20_min
    amt_ratio = max(0.0, amount_ma20 / max(1.0, float(amount_benchmark)))
    score += min(18.0, 9.0 * math.log1p(amt_ratio * 1.6))

    # Momentum: consecutive up days in last 5 (max 14)
    score += min(14.0, up_days * 2.8)

    # Momentum: volume trend ratio (max 8)
    if vol_trend is not None:
        score += max(0.0, min(8.0, (vol_trend - 0.9) * 10.0))

    # Penalize latest bearish close (단기매매에서는 전일 조정이 진입 기회일 수 있으므로 -4로 완화).
    if is_last_bearish:
        score -= 4.0

    # Trend alignment: MA5 / MA20 ratio (±10)
    if ma5 is not None and ma20 not in (None, 0):
        trend_ratio = (ma5 / ma20) - 1.0
        score += max(-10.0, min(10.0, trend_ratio * 420.0))

    # 52-week positioning (max 5): 단기매매는 신고가 모멘텀 중시, 70% 구간 선호.
    if high_52w_ratio is not None and 0.20 <= high_52w_ratio <= config.max_52w_high_ratio:
        center = 0.70
        width = 0.40
        proximity = max(0.0, 1.0 - (abs(high_52w_ratio - center) / width))
        score += 5.0 * proximity

    # Overextended near 52w high: keep candidate but apply risk penalty.
    if high_52w_ratio is not None and high_52w_ratio >= config.max_52w_high_ratio:
        overflow = min(0.20, high_52w_ratio - config.max_52w_high_ratio)
        near_high_penalty = min(12.0, 6.0 + (overflow / 0.20) * 6.0)
        if near_52w_high_override:
            # Strong supporting signals can offset most overextension risk.
            near_high_penalty *= 0.25
        score -= near_high_penalty

    # Previous-day surge/gap risk: keep candidate but apply risk penalty.
    if prev_day_change is not None and prev_day_change >= config.max_prev_day_change:
        overflow = min(0.15, prev_day_change - config.max_prev_day_change)
        score -= min(10.0, 4.0 + (overflow / 0.15) * 6.0)

    # 3일 수익률 모멘텀 가산점 (±5): 단기 상승 탄력 평가.
    close_3d_return = candidate.get("close_3d_return")
    if close_3d_return is not None:
        score += max(-5.0, min(5.0, close_3d_return * 60))

    # Penalize symbols repeatedly selected in recent days to reduce over-concentration.
    if repeat_recent_days > 0:
        score -= min(12.0, repeat_recent_days * config.recent_pick_penalty_per_day)

    # Price range preference (max 5): soft bias to tradable middle range.
    if price <= config.price_max:
        pref = 1.0 - min(1.0, max(0.0, (price - config.price_min) / max(1.0, (200_000 - config.price_min))))
        score += 2.0 + (3.0 * pref)

    # Soft-flag penalty to separate near-identical candidates.
    score -= 0.7 * len(candidate.get("soft_flags", []))

    return round(score, 2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def format_metric(value, digits=0):
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def summarize_candidates(candidates, selected_rows, skipped):
    fail_counts = {}
    soft_counts = {}
    for candidate in candidates:
        for reason in candidate["fail_reasons"]:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
        for reason in candidate["soft_flags"]:
            soft_counts[reason] = soft_counts.get(reason, 0) + 1
    return {
        "selected_count": len(selected_rows),
        "eligible_pool_count": sum(1 for row in candidates if row["eligible"]),
        "skipped": skipped,
        "fail_counts": dict(sorted(fail_counts.items(), key=lambda item: (-item[1], item[0]))),
        "soft_counts": dict(sorted(soft_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def is_scorable_candidate(row):
    return (
        row.get("skip_reason") is None
        and row.get("price") is not None
        and row.get("atr_ratio") is not None
        and row.get("vol_ma20") is not None
        and row.get("amount_ma20") is not None
    )


def selection_weight(row):
    """Weight for tie randomization based on liquidity and volatility quality."""
    amt = max(1.0, float(row.get("amount_ma20") or 0.0))
    vol = max(1.0, float(row.get("vol_ma20") or 0.0))
    atr = max(1e-6, float(row.get("atr_ratio") or 0.0))
    trend_bonus = 1.05 if row.get("trend_state") == "up" else 1.0
    return (amt ** 0.22) * (vol ** 0.18) * (atr ** 0.35) * trend_bonus


def weighted_sample_without_replacement(rows, k, rng):
    if k <= 0 or not rows:
        return []

    pool = list(rows)
    selected = []
    while pool and len(selected) < k:
        weights = [selection_weight(row) for row in pool]
        chosen = rng.choices(pool, weights=weights, k=1)[0]
        selected.append(chosen)
        pool.remove(chosen)
    return selected


def pick_diversified_top_pool(sorted_rows, max_picks, rng, pool_mult=3):
    """Diversified selection from a top-ranked pool, not only exact-score ties."""
    if max_picks is None:
        return list(sorted_rows)
    if max_picks <= 0 or not sorted_rows:
        return []

    pool_size = min(len(sorted_rows), max(max_picks, max_picks * max(1, int(pool_mult))))
    top_pool = list(sorted_rows[:pool_size])
    if len(top_pool) <= max_picks:
        return top_pool

    top_score = float(top_pool[0].get("score") or 0.0)
    base_score = float(top_pool[-1].get("score") or 0.0)
    span = max(1.0, top_score - base_score)

    pool = list(top_pool)
    selected = []
    while pool and len(selected) < max_picks:
        weights = []
        for row in pool:
            score = float(row.get("score") or 0.0)
            score_boost = 1.0 + ((score - base_score) / span)
            weights.append(selection_weight(row) * (score_boost ** 1.35))
        chosen = rng.choices(pool, weights=weights, k=1)[0]
        selected.append(chosen)
        pool.remove(chosen)

    selected.sort(
        key=lambda row: (row["score"], row.get("amount_ma20") or 0.0, row.get("atr_ratio") or 0.0),
        reverse=True,
    )
    return selected


def pick_with_tie_randomization(sorted_rows, max_picks, rng):
    """Pick by score rank, but randomize inside same-score buckets."""
    if max_picks is None:
        return list(sorted_rows)

    picked = []
    idx = 0
    n = len(sorted_rows)

    while idx < n and len(picked) < max_picks:
        score = sorted_rows[idx]["score"]
        bucket = []
        while idx < n and sorted_rows[idx]["score"] == score:
            bucket.append(sorted_rows[idx])
            idx += 1

        remaining = max_picks - len(picked)
        if len(bucket) <= remaining:
            picked.extend(bucket)
            continue

        picked.extend(weighted_sample_without_replacement(bucket, remaining, rng))
        break

    picked.sort(
        key=lambda row: (row["score"], row["amount_ma20"] or 0.0, row["atr_ratio"] or 0.0),
        reverse=True,
    )
    return picked



def render_ranked_csv(selected_rows):
    header = (
        "rank,code,name,score,price,atr_ratio,vol_ma20,amount_ma20,"
        "trend_state,ma_gap,up_days_in_5,vol_trend_ratio,high_52w_ratio,"
        "listing_days,prev_day_change,repeat_recent_days"
    )
    lines = [header]
    for rank, row in enumerate(selected_rows, start=1):
        lines.append(",".join([
            str(rank),
            row["code"],
            row["name"].replace(",", " "),
            f"{row['score']:.2f}",
            format_metric(row["price"], 0).replace(",", ""),
            f"{row['atr_ratio']:.6f}" if row["atr_ratio"] is not None else "",
            format_metric(row["vol_ma20"], 0).replace(",", ""),
            format_metric(row["amount_ma20"], 0).replace(",", ""),
            row["trend_state"],
            f"{row['ma_gap']:.2f}" if row["ma_gap"] is not None else "",
            str(row.get("up_days_in_5", 0)),
            f"{row['vol_trend_ratio']:.3f}" if row.get("vol_trend_ratio") is not None else "",
            f"{row['high_52w_ratio']:.3f}" if row.get("high_52w_ratio") is not None else "",
            str(row.get("listing_days", 0)),
            f"{row['prev_day_change']:.4f}" if row.get("prev_day_change") is not None else "",
            str(row.get("repeat_recent_days", 0)),
        ]))
    return "\n".join(lines)


def render_all_scan_csv(all_rows):
    header = (
        "rank,code,name,eligible,skip_reason,fail_reasons,soft_flags,score,price,atr_ratio,vol_ma20,amount_ma20,"
        "trend_state,ma_gap,up_days_in_5,vol_trend_ratio,high_52w_ratio,listing_days,prev_day_change,repeat_recent_days"
    )
    lines = [header]
    for rank, row in enumerate(all_rows, start=1):
        fail_reasons = "|".join(row.get("fail_reasons", []))
        soft_flags = "|".join(row.get("soft_flags", []))
        lines.append(",".join([
            str(rank),
            row.get("code", ""),
            str(row.get("name", "")).replace(",", " "),
            "Y" if row.get("eligible") else "N",
            str(row.get("skip_reason") or ""),
            fail_reasons,
            soft_flags,
            f"{float(row.get('score') or 0.0):.2f}",
            format_metric(row.get("price"), 0).replace(",", "") if row.get("price") is not None else "",
            f"{row.get('atr_ratio'):.6f}" if row.get("atr_ratio") is not None else "",
            format_metric(row.get("vol_ma20"), 0).replace(",", "") if row.get("vol_ma20") is not None else "",
            format_metric(row.get("amount_ma20"), 0).replace(",", "") if row.get("amount_ma20") is not None else "",
            str(row.get("trend_state") or ""),
            f"{row.get('ma_gap'):.2f}" if row.get("ma_gap") is not None else "",
            str(row.get("up_days_in_5") if row.get("up_days_in_5") is not None else ""),
            f"{row.get('vol_trend_ratio'):.3f}" if row.get("vol_trend_ratio") is not None else "",
            f"{row.get('high_52w_ratio'):.3f}" if row.get("high_52w_ratio") is not None else "",
            str(row.get("listing_days") if row.get("listing_days") is not None else ""),
            f"{row.get('prev_day_change'):.4f}" if row.get("prev_day_change") is not None else "",
            str(row.get("repeat_recent_days") if row.get("repeat_recent_days") is not None else ""),
        ]))
    return "\n".join(lines)


def render_all_scan_markdown(all_rows):
    headers = [
        "rank", "code", "name", "score", "price", "atr_ratio", "vol_ma20", "amount_ma20",
        "trend_state", "ma_gap", "up_days_in_5", "high_52w_ratio", "listing_days", "prev_day_change", "repeat_recent_days",
        "soft_flags", "fail_reasons", "eligible", "skip_reason", "vol_trend_ratio",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for rank, row in enumerate(all_rows, start=1):
        fail_reasons = "|".join(row.get("fail_reasons", []))
        soft_flags = "|".join(row.get("soft_flags", []))
        values = [
            str(rank),
            row.get("code", ""),
            str(row.get("name", "")).replace(",", " "),
            f"{float(row.get('score') or 0.0):.2f}",
            format_metric(row.get("price"), 0).replace(",", "") if row.get("price") is not None else "",
            f"{row.get('atr_ratio'):.6f}" if row.get("atr_ratio") is not None else "",
            format_metric(row.get("vol_ma20"), 0).replace(",", "") if row.get("vol_ma20") is not None else "",
            format_metric(row.get("amount_ma20"), 0).replace(",", "") if row.get("amount_ma20") is not None else "",
            str(row.get("trend_state") or ""),
            f"{row.get('ma_gap'):.2f}" if row.get("ma_gap") is not None else "",
            str(row.get("up_days_in_5") if row.get("up_days_in_5") is not None else ""),
            f"{row.get('high_52w_ratio'):.3f}" if row.get("high_52w_ratio") is not None else "",
            str(row.get("listing_days") if row.get("listing_days") is not None else ""),
            f"{row.get('prev_day_change'):.4f}" if row.get("prev_day_change") is not None else "",
            str(row.get("repeat_recent_days") if row.get("repeat_recent_days") is not None else ""),
            soft_flags,
            fail_reasons,
            "Y" if row.get("eligible") else "N",
            str(row.get("skip_reason") or ""),
            f"{row.get('vol_trend_ratio'):.3f}" if row.get("vol_trend_ratio") is not None else "",
        ]
        sanitized = [str(value).replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(sanitized) + " |")

    return "\n".join(lines)


def build_history_comparison(data_root, stock_list, history_window=DEFAULT_HISTORY_WINDOW):
    date_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    compare_dirs = date_dirs[-history_window:]
    comparison_rows = []
    total = len(compare_dirs)
    print(f"\n[INFO] 히스토리 비교 리포트 생성 중 ({total}일치)...")

    for i, date_dir in enumerate(compare_dirs, start=1):
        try:
            target_date = datetime.strptime(date_dir.name, "%Y%m%d")
        except ValueError:
            continue
        print(f"  [{i}/{total}] {date_dir.name} 스캔 중...", end="\r", flush=True)
        result = scan(
            data_root, target_date=target_date, config=DEFAULT_CONFIG,
            return_details=True, verbose=False, stock_list=stock_list,
        )
        comparison_rows.append({
            "date": date_dir.name,
            "eligible": result["summary"]["eligible_pool_count"],
            "selected": result["summary"]["selected_count"],
        })

    print(f"  히스토리 비교 완료 ({total}일치)          ")
    return comparison_rows


def render_report(data_root, target_date, config, scan_result, comparison_rows):
    target_label = target_date.strftime("%Y%m%d") if target_date else str(data_root)
    summary = scan_result["summary"]
    lines = [
        f"# Scanner Report - {target_label}",
        "",
        "## Active Scanner Config",
        "",
        f"- config: {config.name}",
        f"- price range: {config.price_min:,} ~ {config.price_max:,}",
        f"- ATR/price min (daily): {config.atr_ratio_min:.4f}",
        "- liquidity filter: market-relative (KOSPI/KOSDAQ average amount + price-adjusted volume)",
        f"- amount baseline(for score only): {config.amount_ma20_min:,}",
        f"- min listing days: {config.min_listing_days}",
        f"- min up days in 5: {config.min_up_days_in_5}",
        f"- max 52w high ratio: {config.max_52w_high_ratio}",
        f"- max prev day change: {config.max_prev_day_change:.1%}",
        f"- volume trend min ratio: {config.volume_trend_min_ratio}",
        f"- recent pick penalty/day: {config.recent_pick_penalty_per_day}",
        f"- recent pick lookback days: {config.recent_pick_penalty_lookback_days}",
        f"- diversified pick pool x: {config.diversified_pick_pool_mult}",
        f"- max picks: {config.max_picks}",
        "",
        "## Summary",
        "",
        f"- selection mode: {scan_result.get('selection_mode', 'strict_eligible')}",
        f"- eligible pool: {summary['eligible_pool_count']}",
        f"- selected picks: {summary['selected_count']}",
        f"- skipped: {summary['skipped']}",
        "",
        "## Ranked Picks",
        "",
        "| rank | code | name | score | price | ATR/price | vol MA20 | amount MA20 | trend | up/5 | vol trend | 52w pos | listing | repeat(d) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for rank, row in enumerate(scan_result["selected_rows"], start=1):
        lines.append(
            "| {rank} | {code} | {name} | {score:.2f} | {price} | {atr} | {vol} | {amt} | {trend} | {up} | {vtd} | {h52w} | {listing} | {repeat} |".format(
                rank=rank,
                code=row["code"],
                name=row["name"],
                score=row["score"],
                price=format_metric(row["price"], 0),
                atr=f"{row['atr_ratio']:.2%}" if row["atr_ratio"] is not None else "-",
                vol=format_metric(row["vol_ma20"], 0),
                amt=format_metric(row["amount_ma20"], 0),
                trend=row["trend_state"],
                up=row.get("up_days_in_5", "-"),
                vtd=f"{row['vol_trend_ratio']:.2f}" if row.get("vol_trend_ratio") is not None else "-",
                h52w=f"{row['high_52w_ratio']:.1%}" if row.get("high_52w_ratio") is not None else "-",
                listing=row.get("listing_days", "-"),
                repeat=row.get("repeat_recent_days", 0),
            )
        )

    lines.extend([
        "",
        "## Filter Counts",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ])
    for reason, count in summary["fail_counts"].items():
        lines.append(f"| {reason} | {count} |")

    if summary["soft_counts"]:
        lines.extend([
            "",
            "## Soft Flags",
            "",
            "| flag | count |",
            "| --- | ---: |",
        ])
        for reason, count in summary["soft_counts"].items():
            lines.append(f"| {reason} | {count} |")

    if comparison_rows:
        lines.extend([
            "",
            f"## Recent {len(comparison_rows)}-Day Comparison",
            "",
            "| date | eligible | selected |",
            "| --- | ---: | ---: |",
        ])
        for row in comparison_rows:
            lines.append("| {date} | {eligible} | {selected} |".format(**row))

    return "\n".join(lines) + "\n"


def load_symbols():
    symbols_path = Path("r009_universe_symbols_master.txt")

    if not symbols_path.exists():
        raise SystemExit("r009_universe_symbols_master.txt 파일이 없습니다.")

    df = pd.read_csv(symbols_path)

    if "code" not in df.columns:
        raise SystemExit("r009_universe_symbols_master.txt에 'code' 컬럼이 필요합니다.")

    name_col = "name" if "name" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    if name_col:
        return list(zip(df["code"].astype(str).str.zfill(6), df[name_col].astype(str)))
    return [(c, "") for c in df["code"].astype(str).str.zfill(6)]


def filter_stock_list_by_existing_data(stock_list, data_root, verbose=True):
    """Pre-filter to codes that have at least one data file anywhere under data_root."""
    all_codes: set[str] = set()
    for date_dir in data_root.iterdir():
        if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
            for p in list(date_dir.glob("*.txt")) + list(date_dir.glob("*.csv")):
                code = _extract_code_from_stem(p.stem)
                if code is not None:
                    all_codes.add(code)

    filtered = [(code, name) for code, name in stock_list if code in all_codes]
    if verbose:
        missing = len(stock_list) - len(filtered)
        print(f"[INFO] 데이터 존재 종목 필터링: {len(stock_list)} -> {len(filtered)} (제외 {missing})")
    return filtered


def find_nearest_trading_date(data_root, target_date_str):
    """
    If target_date_str has no data (holiday), find the most recent trading date <= target_date_str.
    Returns target_date_str if it exists, or the nearest earlier trading date.
    """
    date_dirs = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    
    # Filter to dates <= target_date_str
    available_dates = [d for d in date_dirs if d <= target_date_str]
    
    if not available_dates:
        # No data before target_date; use the earliest available date
        return date_dirs[0] if date_dirs else target_date_str
    
    # Return the most recent available date
    return available_dates[-1]


def filter_stock_list_by_min_bars(stock_list, data_root, target_date_str=None, min_bars=5, verbose=True, single_date_only=False):
    """Pre-filter to codes that have at least min_bars daily files up to target_date."""
    date_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    if target_date_str:
        if single_date_only:
            date_dirs = [d for d in date_dirs if d.name == target_date_str]
        else:
            date_dirs = [d for d in date_dirs if d.name <= target_date_str]

    qualified = []
    for code, name in stock_list:
        bars = 0
        for date_dir in date_dirs:
            if resolve_symbol_file(date_dir, code) is not None:
                bars += 1
                if bars >= min_bars:
                    qualified.append((code, name))
                    break

    if verbose:
        excluded = len(stock_list) - len(qualified)
        label = target_date_str or "latest"
        print(
            f"[INFO] 최소 {min_bars}일 데이터 필터링({label} 기준): "
            f"{len(stock_list)} -> {len(qualified)} (제외 {excluded})"
        )
    return qualified


def load_recent_pick_counts(data_root: Path, target_date_str: str | None, lookback_days: int) -> dict[str, int]:
    """Count how many prior recent days each symbol was picked."""
    if not target_date_str or lookback_days <= 0:
        return {}

    date_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8 and d.name < target_date_str
    )
    if not date_dirs:
        return {}

    recent_dirs = date_dirs[-lookback_days:]
    counts: dict[str, int] = {}
    for d in recent_dirs:
        picks_file = d / f"_{d.name}_picks.txt"
        if not picks_file.exists():
            continue
        try:
            lines = [ln.strip() for ln in picks_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            continue
        seen_today: set[str] = set()
        for ln in lines:
            code = str(ln.split(",", 1)[0]).strip().zfill(6)
            if not re.fullmatch(r"\d{6}", code):
                continue
            if code in seen_today:
                continue
            seen_today.add(code)
            counts[code] = counts.get(code, 0) + 1
    return counts


def load_w52_high_low(data_root: Path, target_date_str: str | None) -> dict[str, dict]:
    """Load code -> {w52_high, w52_low} saved by r001 from KIS inquire_price
    (real server 52-week high/low, field w52_hgpr/w52_lwpr). Returns {} if the
    snapshot file is missing (falls back to local daily-bar max in that case).
    """
    if not target_date_str:
        return {}
    snapshot_path = data_root / str(target_date_str) / "_52w_high_low.json"
    if not snapshot_path.exists():
        return {}
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _pick_market_reference_date(data_root: Path, target_date_str: str | None) -> str | None:
    if target_date_str:
        return find_nearest_trading_date(data_root, target_date_str)

    date_dirs = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    return date_dirs[-1] if date_dirs else None


def load_market_map(data_root: Path, target_date_str: str | None, verbose: bool = True) -> tuple[dict[str, str], str | None]:
    """Load code->market map (kospi/kosdaq) from pykrx ticker snapshots."""
    ref_date = _pick_market_reference_date(data_root, target_date_str)
    if not ref_date:
        return {}, None

    try:
        from pykrx import stock  # type: ignore
    except Exception as exc:
        if verbose:
            print(f"[WARN] pykrx import 실패로 시장 구분 매핑을 사용하지 않습니다: {exc}")
        return {}, ref_date

    parsed = datetime.strptime(ref_date, "%Y%m%d")
    for offset in range(0, 8):
        date_str = (parsed - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            kospi = set(stock.get_market_ticker_list(date_str, market="KOSPI"))
            kosdaq = set(stock.get_market_ticker_list(date_str, market="KOSDAQ"))
        except Exception:
            continue

        if not kospi and not kosdaq:
            continue

        market_map: dict[str, str] = {}
        for code in kospi:
            market_map[str(code).zfill(6)] = "kospi"
        for code in kosdaq:
            market_map[str(code).zfill(6)] = "kosdaq"

        if verbose:
            print(
                f"[INFO] 시장 매핑 로드 완료 ({date_str}) | "
                f"KOSPI={len(kospi)} KOSDAQ={len(kosdaq)}"
            )
        return market_map, date_str

    if verbose:
        print(f"[WARN] 시장 매핑 조회 실패: ref_date={ref_date} (최근 7일 내 유효 데이터 없음)")
    return {}, ref_date


def apply_market_relative_liquidity_filters(candidates: list[dict], market_map: dict[str, str]) -> dict[str, object]:
    """Apply market-relative liquidity thresholds.

    - amount threshold: market avg amount_ma20 (KOSPI/KOSDAQ)
    - volume threshold: market avg amount converted by each stock price
      (price-adjusted equivalent shares)
    """
    for row in candidates:
        code = str(row.get("code") or "").zfill(6)
        row["market"] = market_map.get(code, "unknown")

    scorable = [
        row for row in candidates
        if row.get("skip_reason") is None
        and row.get("price") is not None
        and row.get("amount_ma20") is not None
        and row.get("vol_ma20") is not None
    ]

    if not scorable:
        return {
            "enabled": False,
            "global_amount_avg": None,
            "market_amount_avg": {},
            "market_counts": {},
        }

    global_amount_avg = float(pd.Series([float(r["amount_ma20"]) for r in scorable]).mean())
    market_amount_avg: dict[str, float] = {}
    market_counts: dict[str, int] = {}

    for market in ("kospi", "kosdaq", "unknown"):
        rows = [r for r in scorable if r.get("market") == market]
        market_counts[market] = len(rows)
        if rows:
            market_amount_avg[market] = float(pd.Series([float(r["amount_ma20"]) for r in rows]).mean())

    min_sample = 25
    for row in candidates:
        if row.get("skip_reason") is not None:
            continue

        price = row.get("price")
        amount_ma20 = row.get("amount_ma20")
        vol_ma20 = row.get("vol_ma20")
        if price is None or amount_ma20 is None or vol_ma20 is None:
            continue

        market = str(row.get("market") or "unknown")
        market_count = int(market_counts.get(market, 0))
        market_avg_amt = market_amount_avg.get(market)
        if market_avg_amt is None or market_count < min_sample:
            market_avg_amt = global_amount_avg

        required_amt = float(market_avg_amt) * LIQUIDITY_RELAX_FACTOR
        required_vol = float(required_amt) / max(1.0, float(price))

        row["liquidity_amount_benchmark"] = required_amt
        row["liquidity_volume_benchmark"] = required_vol

        vol_below = float(vol_ma20) < required_vol
        amt_below = float(amount_ma20) < required_amt

        # Relaxation: disqualify only when both liquidity checks fail.
        if vol_below and amt_below:
            row["fail_reasons"].append("liquidity_below_market_dual")
        else:
            if vol_below:
                row["soft_flags"].append("volume_below_market_price_adjusted_avg")
            if amt_below:
                row["soft_flags"].append("amount_below_market_avg")

    return {
        "enabled": True,
        "global_amount_avg": global_amount_avg,
        "market_amount_avg": market_amount_avg,
        "market_counts": market_counts,
    }


# ---------------------------------------------------------------------------
# Scan orchestration
# ---------------------------------------------------------------------------

def scan(
    data_root,
    target_date=None,
    config=DEFAULT_CONFIG,
    return_details=False,
    verbose=True,
    stock_list=None,
    single_date_only=False,
):
    target_date_str = target_date.strftime("%Y%m%d") if target_date else None
    rng = random.Random(int(target_date_str) if target_date_str and target_date_str.isdigit() else 42)
    
    # Auto-adjust target_date if it's a holiday (no data directory exists)
    if target_date_str:
        adjusted_target_date_str = find_nearest_trading_date(data_root, target_date_str)
        if adjusted_target_date_str != target_date_str:
            if verbose:
                print(f"[INFO] {target_date_str}은(는) 거래 없는 날(공휴일)입니다. {adjusted_target_date_str}로 조정합니다.")
            target_date_str = adjusted_target_date_str
    
    stock_list = stock_list or load_symbols()
    total = len(stock_list)

    if verbose:
        label = target_date_str or "latest"
        print(f"\n[INFO] 총 {total} 종목 스캔 시작 (기준일: {label}, config: {config.name})\n")

    if total == 0:
        print("[ERROR] stock_list is empty! check load_symbols() or filter_stock_list_by_existing_data()")
        return {"selected": [], "selected_rows": [], "eligible_rows": [], "summary": {"selected_count": 0, "eligible_pool_count": 0, "skipped": 0, "fail_counts": {}, "soft_counts": {}}, "config": config, "selection_mode": "error_empty_list"} if return_details else []

    recent_pick_counts = load_recent_pick_counts(
        data_root,
        target_date_str,
        lookback_days=config.recent_pick_penalty_lookback_days,
    )
    market_map, market_ref_date = load_market_map(data_root, target_date_str, verbose=verbose)
    w52_map = load_w52_high_low(data_root, target_date_str)

    candidates = []
    skipped = 0
    insufficient_bars_count = 0

    for idx, (code, name) in enumerate(stock_list, start=1):
        if verbose:
            print(f"[{idx}/{total}] {code} 검증중...", end="\r", flush=True)

        daily_df = build_daily_bars(data_root, code, target_date_str, single_date_only=single_date_only)
        if daily_df is None or len(daily_df) < MIN_REQUIRED_BARS:
            skipped += 1
            if daily_df is None or len(daily_df) == 0:
                insufficient_bars_count += 1
            if verbose and idx % 50 == 0:
                print(f"[{idx}/{total}] 진행중 - 현재 스킵 {skipped}종목")
            continue

        recent_repeat_days = int(recent_pick_counts.get(code, 0))
        daily_hist = _load_daily_csv(code, data_root, target_date_str)
        candidate = evaluate_candidate(
            code, name, daily_df, config,
            recent_pick_count=recent_repeat_days, daily_hist=daily_hist,
            w52_info=w52_map.get(code),
        )
        if recent_repeat_days > 0:
            candidate["soft_flags"].append(f"recent_pick_repeat_{recent_repeat_days}d")
        candidates.append(candidate)

        if verbose:
            if candidate["eligible"]:
                # score 기반으로 🟢 개수 계산 (10점당 1개, 반올림, 최소 0개)
                green_dots_count = max(0, round(candidate["score"] / 10.0))
                green_dots = "🟢" * green_dots_count
                
                # 🟢 개수가 0개일 경우 가독성을 위해 공백 한 칸 유지
                dots_str = f" {green_dots}" if green_dots else ""
                
                print(f"[{idx}/{total}] {code}_{name}{dots_str} [후보] (score={candidate['score']:.2f})          ")
            else:
                reasons = candidate["fail_reasons"] or [candidate["skip_reason"] or "unknown"]
                print(f"[{idx}/{total}] {code}_{name} [탈락] ({', '.join(reasons)})          ")

    liquidity_filter_info = apply_market_relative_liquidity_filters(candidates, market_map)

    for row in candidates:
        score = calculate_candidate_score(row, config)
        row["score"] = score
        row["fail_reasons"] = [reason for reason in row.get("fail_reasons", []) if reason != "low_score"]
        if score < SCORE_CUTOFF:
            row["fail_reasons"].append("low_score")
        row["eligible"] = not row["fail_reasons"]

    eligible_rows = [row for row in candidates if row["eligible"]]
    eligible_rows.sort(
        key=lambda row: (row["score"], row["amount_ma20"] or 0.0, row["atr_ratio"] or 0.0),
        reverse=True,
    )

    selected_rows = eligible_rows
    selection_mode = "strict_eligible"

    if config.max_picks is not None:
        selected_rows = pick_diversified_top_pool(
            eligible_rows,
            config.max_picks,
            rng,
            pool_mult=config.diversified_pick_pool_mult,
        )

    # Fallback: fill up to max_picks with scorable non-down-trend candidates.
    # NOTE: fallback previously only checked trend_state/is_last_bearish/score and
    # explicitly excluded "low_score", but silently ignored every OTHER hard-filter
    # fail_reason (down_trend, daily_5d_downtrend, bb_lower_not_uptrend,
    # linreg_long_term_downtrend, bearish_2in3d_*, big_bearish_candle,
    # upperlimit_streak_crash). A stock whose short-term MA5/MA20 flipped to "up"
    # off a bounce (e.g. 089030 on 2026-07-20) could fail every downtrend hard
    # filter and still get rescued back into the picks via this path. Liquidity-
    # only reasons stay relaxable (that was the original intent); trend/candle
    # pattern failures are not.
    if config.max_picks is not None and len(selected_rows) < config.max_picks:
        selected_codes = {row["code"] for row in selected_rows}
        fallback_rows = [
            row for row in candidates
            if is_scorable_candidate(row)
            # Keep fallback defensive: only clear up-trend names.
            and row.get("trend_state") == "up"
            and not bool(row.get("is_last_bearish"))
            and row.get("score", 0.0) >= SCORE_CUTOFF
            and not (set(row.get("fail_reasons", [])) - FALLBACK_RELAXABLE_FAIL_REASONS)
            and row["code"] not in selected_codes
        ]
        fallback_rows.sort(
            key=lambda row: (row["score"], row["amount_ma20"] or 0.0),
            reverse=True,
        )
        needed = config.max_picks - len(selected_rows)
        supplement = pick_with_tie_randomization(fallback_rows, needed, rng)
        if supplement:
            selected_rows = [*selected_rows, *supplement]
            selection_mode = "strict_plus_fallback"
            for row in supplement:
                if "fallback_selected" not in row["soft_flags"]:
                    row["soft_flags"].append("fallback_selected")

    # Final output order: always keep selected rows sorted by score desc.
    selected_rows.sort(
        key=lambda row: (row["score"], row["amount_ma20"] or 0.0, row["atr_ratio"] or 0.0),
        reverse=True,
    )

    selected = [f"{row['code']},{row['name']}" for row in selected_rows]
    summary = summarize_candidates(candidates, selected_rows, skipped)

    if verbose:
        print("\n========== 스캔 완료 ==========")
        print(f"총종목: {total}")
        print(f"데이터 불가(None/empty): {insufficient_bars_count}")
        print(f"데이터 부족(<{MIN_REQUIRED_BARS}일): {skipped - insufficient_bars_count}")
        print(f"총 스킵: {skipped}")
        print(f"평가된 종목: {len(candidates)}")
        print(f"적격 수: {summary['eligible_pool_count']}")
        print(f"최종 선정: {summary['selected_count']}")
        if liquidity_filter_info.get("enabled"):
            avg_amt = liquidity_filter_info.get("global_amount_avg")
            mkt_counts = liquidity_filter_info.get("market_counts") or {}
            print(
                "시장상대 유동성 필터: "
                f"ref={market_ref_date or target_date_str or 'latest'} "
                f"global_avg_amt={avg_amt:,.0f} "
                f"(kospi={mkt_counts.get('kospi', 0)}, "
                f"kosdaq={mkt_counts.get('kosdaq', 0)}, "
                f"unknown={mkt_counts.get('unknown', 0)})"
            )

    if return_details:
        all_rows = sorted(
            candidates,
            key=lambda row: (row.get("score", 0.0), row.get("amount_ma20") or 0.0, row.get("atr_ratio") or 0.0),
            reverse=True,
        )
        return {
            "selected": selected,
            "selected_rows": selected_rows,
            "eligible_rows": eligible_rows,
            "all_rows": all_rows,
            "summary": summary,
            "config": config,
            "selection_mode": selection_mode,
            "market_ref_date": market_ref_date,
            "liquidity_filter": liquidity_filter_info,
        }
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Trade candidate scanner (daily-bar mode)")
    parser.add_argument("--date", type=str, default=None, help="Base date (YYYYMMDD)")
    parser.add_argument("--data-dir", type=str, default=None, help="Root data folder (default: data/)")
    parser.add_argument(
        "--config", type=str, default="intraday",
        choices=list(CONFIG_MAP.keys()),
        help="Scanner config preset",
    )
    parser.add_argument("--max-picks", type=int, default=None, help="Override config max_picks")
    parser.add_argument(
        "--history-window", type=int, default=DEFAULT_HISTORY_WINDOW,
        help="Number of recent trading days to include in comparison report",
    )
    parser.add_argument(
        "--export-watchlist",
        action="store_true",
        help="[FEATURE_SCAN_EXPORT_TO_R008] Export picks to r008_trade_watchlist_today.txt",
    )
    parser.add_argument(
        "--no-export-watchlist",
        action="store_true",
        help="Disable r008 export even when FEATURE_SCAN_EXPORT_TO_R008 is True",
    )
    parser.add_argument(
        "--export-watchlist-path",
        type=str,
        default=None,
        help="Override r008 export path (default: auto_trading/r008_trade_watchlist_today.txt)",
    )
    parser.add_argument(
        "--for-next-trading-day",
        action="store_true",
        help="Add metadata comment for next calendar day in r008 export header",
    )
    parser.add_argument(
        "--write-picks-alias",
        action="store_true",
        help="Also write data/YYYYMMDD/picks.txt (legacy r006 --date compatibility)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y%m%d")
        except ValueError:
            raise SystemExit("날짜 형식은 YYYYMMDD 여야 합니다.")

    data_root = Path(args.data_dir) if args.data_dir else Path("data")
    if not data_root.exists():
        raise SystemExit(f"데이터 폴더가 없습니다: {data_root}")

    config = CONFIG_MAP[args.config]
    if args.max_picks is not None:
        config = dc_replace(config, max_picks=min(args.max_picks, MAX_PICKS_LIMIT))
    elif config.max_picks is not None:
        config = dc_replace(config, max_picks=min(config.max_picks, MAX_PICKS_LIMIT))

    effective_target_date = target_date
    single_date_only = False
    if target_date:
        target_date_str = target_date.strftime("%Y%m%d")
        adjusted = find_nearest_trading_date(data_root, target_date_str)
        if adjusted != target_date_str:
            print(f"[INFO] {target_date_str}은(는) 거래 없는 날(공휴일)입니다. {adjusted}로 조정합니다.")
            effective_target_date = datetime.strptime(adjusted, "%Y%m%d")

    stock_list = load_symbols()
    stock_list = filter_stock_list_by_existing_data(stock_list, data_root)
    stock_list = filter_stock_list_by_min_bars(
        stock_list,
        data_root,
        target_date_str=effective_target_date.strftime("%Y%m%d") if effective_target_date else None,
        min_bars=MIN_REQUIRED_BARS,
        single_date_only=single_date_only,
    )
    if not stock_list:
        raise SystemExit(f"{data_root} 에 스캔 가능한 데이터가 없습니다.")

    scan_result = scan(
        data_root,
        effective_target_date,
        config=config,
        return_details=True,
        stock_list=stock_list,
        single_date_only=single_date_only,
    )
    picks = scan_result["selected"]

    comparison_rows = []
    if args.history_window > 0:
        comparison_rows = build_history_comparison(data_root, stock_list, history_window=args.history_window)

    # Determine output directory
    if effective_target_date:
        out_dir = data_root / effective_target_date.strftime("%Y%m%d")
        out_dir.mkdir(exist_ok=True)
    else:
        date_dirs = sorted(
            d for d in data_root.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        )
        out_dir = date_dirs[-1] if date_dirs else data_root

    output_prefix = ""
    if effective_target_date:
        output_prefix = effective_target_date.strftime("%Y%m%d")
    elif out_dir.name.isdigit() and len(out_dir.name) == 8:
        output_prefix = out_dir.name

    picks_filename = f"_{output_prefix}_picks.txt" if output_prefix else "_picks.txt"
    ranked_filename = f"_{output_prefix}_ranked.txt" if output_prefix else "_ranked.txt"
    report_filename = f"_{output_prefix}_scanner_report.md" if output_prefix else "_scanner_report.md"
    all_scan_filename = f"_{output_prefix}_scan_all.md" if output_prefix else "_scan_all.md"

    if picks:
        picks_file = out_dir / picks_filename
        picks_file.write_text("\n".join(picks), encoding="utf-8")
        print(f"추천 종목 리스트를 저장했습니다: {picks_file}")

    try:
        from r003_define_config import FEATURE_SCAN_EXPORT_TO_R008
    except Exception:
        FEATURE_SCAN_EXPORT_TO_R008 = True

    export_enabled = FEATURE_SCAN_EXPORT_TO_R008 and not args.no_export_watchlist
    if args.export_watchlist:
        export_enabled = True

    if export_enabled and picks:
        from r010_watchlist_bridge import export_picks_to_r008, write_legacy_picks_alias

        auto_dir = Path(__file__).resolve().parent
        export_path = Path(args.export_watchlist_path) if args.export_watchlist_path else None
        scan_date_str = output_prefix or None
        r008_out = export_picks_to_r008(
            picks,
            auto_dir,
            scan_date=scan_date_str,
            config_name=config.name,
            for_next_trading_day=bool(args.for_next_trading_day),
            out_path=export_path,
        )
        if r008_out:
            print(f"[FEATURE_SCAN_EXPORT_TO_R008] r008 watchlist 저장: {r008_out}")

        if args.write_picks_alias and scan_date_str:
            alias = write_legacy_picks_alias(picks, data_root, scan_date_str)
            if alias:
                print(f"[FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS] legacy picks alias: {alias}")

    ranked_file = out_dir / ranked_filename
    ranked_file.write_text(render_ranked_csv(scan_result["selected_rows"]), encoding="utf-8")
    print(f"점수 기반 랭킹 파일을 저장했습니다: {ranked_file}")

    report_file = out_dir / report_filename
    report_file.write_text(
        render_report(data_root, effective_target_date, config, scan_result, comparison_rows),
        encoding="utf-8",
    )
    print(f"스캐너 리포트를 저장했습니다: {report_file}")

    all_scan_file = out_dir / all_scan_filename
    all_scan_file.write_text(render_all_scan_markdown(scan_result["all_rows"]), encoding="utf-8")
    print(f"전체 스캔 결과를 저장했습니다: {all_scan_file}")

    label = effective_target_date.strftime("%Y%m%d") if effective_target_date else "최신 데이터"
    print(f"\n[{label}] 기준 추천 종목:")
    if picks:
        for pick in picks:
            print(pick)
    else:
        print("(없음)")






