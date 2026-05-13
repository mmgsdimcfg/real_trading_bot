"""R76 trade-candidate scanner.

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
from dataclasses import dataclass, replace as dc_replace
from datetime import datetime
from pathlib import Path

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
    max_52w_high_ratio: float       # exclude if price >= ratio * 52-week high
    max_prev_day_change: float      # exclude if previous day abs return >= this
    volume_trend_min_ratio: float   # recent 5d avg vol / prior 5d avg vol minimum
    # Output
    max_picks: int | None


STRICT_CONFIG = ScannerConfig(
    name="strict",
    price_min=2_000,
    price_max=1_000_000,
    atr_ratio_min=0.015,            # 1.5% daily ATR minimum
    volume_ma20_min=200_000,        # 20만주/일
    amount_ma20_min=1_000_000_000,  # 10억원/일
    min_listing_days=120,
    min_up_days_in_5=3,
    max_52w_high_ratio=0.90,        # 52주 고가 90% 이상이면 제외
    max_prev_day_change=0.07,       # 전일 7% 이상 급등락 제외
    volume_trend_min_ratio=1.0,     # 거래량 감소 종목 제외
    max_picks=None,
)

BALANCED_CONFIG = ScannerConfig(
    name="balanced",
    price_min=2_000,
    price_max=1_000_000,
    atr_ratio_min=0.010,
    volume_ma20_min=50_000,
    amount_ma20_min=300_000_000,    # 3억원/일
    min_listing_days=60,
    min_up_days_in_5=2,
    max_52w_high_ratio=0.95,
    max_prev_day_change=0.08,
    volume_trend_min_ratio=0.9,
    max_picks=10,
)

RELAXED_CONFIG = ScannerConfig(
    name="relaxed",
    price_min=2_000,
    price_max=1_000_000,
    atr_ratio_min=0.007,
    volume_ma20_min=20_000,
    amount_ma20_min=100_000_000,    # 1억원/일
    min_listing_days=30,
    min_up_days_in_5=1,
    max_52w_high_ratio=0.97,
    max_prev_day_change=0.10,
    volume_trend_min_ratio=0.8,
    max_picks=20,
)

CONFIG_MAP = {
    "strict": STRICT_CONFIG,
    "balanced": BALANCED_CONFIG,
    "relaxed": RELAXED_CONFIG,
}

DEFAULT_CONFIG = BALANCED_CONFIG
DEFAULT_HISTORY_WINDOW = 0
DAILY_LOOKBACK = 260  # trading days of history to load per stock


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


def load_data(code, data_dir, warn=True):
    file_path = data_dir / f"{code}.txt"

    if not file_path.exists():
        if warn:
            print(f"[WARN] {code} 파일 없음")
        return None

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        return df
    except Exception as exc:
        if warn:
            print(f"[WARN] {code} 데이터 로드 실패: {exc}")
        return None


def build_daily_bars(data_root, code, target_date_str=None, lookback=DAILY_LOOKBACK):
    """Aggregate per-date intraday files into a daily OHLCV DataFrame.

    Each date directory under data_root is expected to contain intraday
    minute-bar CSV files named {code}.txt.  This function reads each file,
    aggregates to a single daily bar (open=first, high=max, low=min,
    close=last, volume=sum), and returns a sorted daily DataFrame.
    """
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


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def classify_trend(ma5, ma20):
    if ma5 is None or ma20 is None:
        return "unknown"
    if ma5 > ma20:
        return "up"
    if ma5 < ma20:
        return "down"
    return "flat"


def calc_volume_trend_ratio(daily_df, window=10):
    """Ratio of recent half-window avg volume vs prior half-window avg volume.

    > 1.0  volume is increasing (bullish signal).
    < 1.0  volume is declining.
    """
    vols = daily_df["volume"].tail(window)
    if len(vols) < window:
        return None
    half = window // 2
    recent_avg = float(vols.tail(half).mean())
    prior_avg = float(vols.head(half).mean())
    if prior_avg == 0:
        return None
    return recent_avg / prior_avg


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(code, name, daily_df, config):
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
        "high_52w_ratio": None,
        "prev_day_change": None,
        "score": 0.0,
    }

    if daily_df is None or len(daily_df) < 5:
        candidate["skip_reason"] = "insufficient_daily_bars"
        return candidate

    price = safe_float(daily_df["close"].iloc[-1])
    if price is None or price <= 0:
        candidate["skip_reason"] = "invalid_price"
        return candidate

    # Daily ATR (14-day); fall back to single-bar range when history is short
    atr_series = calc_atr(daily_df)
    atr = safe_float(atr_series.iloc[-1]) if not atr_series.empty else None
    if atr is None:
        # Use intraday range of the most recent bar as a volatility proxy
        last_high = safe_float(daily_df["high"].iloc[-1])
        last_low = safe_float(daily_df["low"].iloc[-1])
        if last_high is not None and last_low is not None and last_high > last_low:
            atr = last_high - last_low
    atr_ratio = (atr / price) if (atr is not None and price > 0) else None

    # Daily moving averages
    ma5 = safe_float(daily_df["close"].rolling(5, min_periods=1).mean().iloc[-1])
    ma20 = safe_float(daily_df["close"].rolling(20, min_periods=1).mean().iloc[-1])
    ma_gap = (ma5 - ma20) if (ma5 is not None and ma20 is not None) else None

    # Liquidity
    vol_ma20 = safe_float(daily_df["volume"].rolling(20, min_periods=1).mean().iloc[-1])
    amount_ma20 = safe_float(daily_df["amount"].rolling(20, min_periods=1).mean().iloc[-1])

    # Listing age (number of trading days with data)
    listing_days = len(daily_df)

    # Consecutive up days: close > open in last 5 trading days
    last5 = daily_df.tail(5)
    up_days_in_5 = int((last5["close"] > last5["open"]).sum())

    # Volume trend ratio (recent 5d vs prior 5d)
    vol_trend_ratio = calc_volume_trend_ratio(daily_df, window=10)

    # 52-week high position
    lookback_52w = min(252, len(daily_df))
    week52_high = safe_float(daily_df["high"].tail(lookback_52w).max())
    high_52w_ratio = (price / week52_high) if (week52_high is not None and week52_high > 0) else None

    # Previous day absolute return (gap / surge risk)
    prev_day_change = None
    if len(daily_df) >= 2:
        prev_close = safe_float(daily_df["close"].iloc[-2])
        if prev_close is not None and prev_close > 0:
            prev_day_change = abs(price - prev_close) / prev_close

    trend_state = classify_trend(ma5, ma20)

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
        "high_52w_ratio": high_52w_ratio,
        "prev_day_change": prev_day_change,
    })

    # --- Hard filters (fail = disqualified) ---
    if price < config.price_min:
        candidate["fail_reasons"].append("price_floor")
    if price > config.price_max:
        candidate["fail_reasons"].append("price_ceiling")
    if atr_ratio is None or atr_ratio < config.atr_ratio_min:
        candidate["fail_reasons"].append("atr_ratio")
    if vol_ma20 is None or vol_ma20 < config.volume_ma20_min:
        candidate["fail_reasons"].append("volume_ma20")
    if amount_ma20 is None or amount_ma20 < config.amount_ma20_min:
        candidate["fail_reasons"].append("amount_ma20")
    # new_listing: listing_days reflects local data count, not actual IPO age.
    # Universe stocks are already established; skip this as a hard filter.
    # (Noted as soft flag below when data is scarce.)
    # low_up_days: meaningful only when we have at least 5 bars
    if len(daily_df) >= 5 and up_days_in_5 < config.min_up_days_in_5:
        candidate["fail_reasons"].append("low_up_days")
    if high_52w_ratio is not None and high_52w_ratio >= config.max_52w_high_ratio:
        candidate["fail_reasons"].append("near_52w_high")
    if prev_day_change is not None and prev_day_change >= config.max_prev_day_change:
        candidate["fail_reasons"].append("prev_day_gap_risk")
    if trend_state == "down":
        candidate["fail_reasons"].append("down_trend")

    # --- Soft flags (warning only, not disqualified) ---
    if listing_days < config.min_listing_days:
        candidate["soft_flags"].append("limited_history")
    if vol_trend_ratio is not None and vol_trend_ratio < config.volume_trend_min_ratio:
        candidate["soft_flags"].append("volume_declining")
    if trend_state == "flat":
        candidate["soft_flags"].append("flat_trend")
    if high_52w_ratio is not None and high_52w_ratio < 0.4:
        candidate["soft_flags"].append("far_from_52w_high")

    candidate["score"] = calculate_candidate_score(candidate, config)
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
    high_52w_ratio = candidate.get("high_52w_ratio")
    listing_days = candidate.get("listing_days") or 0

    if None in (price, atr_ratio, vol_ma20, amount_ma20):
        return 0.0

    score = 0.0

    # Volatility (max 25)
    score += min(25.0, (atr_ratio / config.atr_ratio_min) * 12.5)

    # Liquidity: volume MA20 (max 20)
    score += min(20.0, (vol_ma20 / config.volume_ma20_min) * 7.0)

    # Liquidity: amount MA20 (max 20)
    score += min(20.0, (amount_ma20 / config.amount_ma20_min) * 7.0)

    # Momentum: consecutive up days in last 5 (max 15)
    score += up_days * 3.0

    # Momentum: volume trend ratio (max 10)
    if vol_trend is not None and vol_trend > 1.0:
        score += min(10.0, (vol_trend - 1.0) * 20.0)

    # Trend alignment: MA5 / MA20 ratio (±10)
    if ma5 is not None and ma20 not in (None, 0):
        trend_ratio = (ma5 / ma20) - 1.0
        score += max(-10.0, min(10.0, trend_ratio * 500.0))

    # 52-week high position: optimal zone 40~85% of 52-week high (max 10)
    if high_52w_ratio is not None:
        if 0.40 <= high_52w_ratio < 0.85:
            score += 10.0
        elif 0.85 <= high_52w_ratio < config.max_52w_high_ratio:
            score += 5.0

    # Listing stability bonus (max 5)
    if listing_days >= 250:
        score += 5.0
    elif listing_days >= 120:
        score += 3.0

    # Price range preference (max 5)
    if price <= 100_000:
        score += 5.0
    elif price <= config.price_max:
        score += 2.0

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



def render_ranked_csv(selected_rows):
    header = (
        "rank,code,name,score,price,atr_ratio,vol_ma20,amount_ma20,"
        "trend_state,ma_gap,up_days_in_5,vol_trend_ratio,high_52w_ratio,"
        "listing_days,prev_day_change"
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
        ]))
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
        f"- volume MA20 min: {config.volume_ma20_min:,}",
        f"- amount MA20 min: {config.amount_ma20_min:,}",
        f"- min listing days: {config.min_listing_days}",
        f"- min up days in 5: {config.min_up_days_in_5}",
        f"- max 52w high ratio: {config.max_52w_high_ratio}",
        f"- max prev day change: {config.max_prev_day_change:.1%}",
        f"- volume trend min ratio: {config.volume_trend_min_ratio}",
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
        "| rank | code | name | score | price | ATR/price | vol MA20 | amount MA20 | trend | up/5 | vol trend | 52w pos | listing |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    for rank, row in enumerate(scan_result["selected_rows"], start=1):
        lines.append(
            "| {rank} | {code} | {name} | {score:.2f} | {price} | {atr} | {vol} | {amt} | {trend} | {up} | {vtd} | {h52w} | {listing} |".format(
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
            for p in date_dir.glob("*.txt"):
                if p.stem.isdigit():
                    all_codes.add(p.stem.zfill(6))

    filtered = [(code, name) for code, name in stock_list if code in all_codes]
    if verbose:
        missing = len(stock_list) - len(filtered)
        print(f"[INFO] 데이터 존재 종목 필터링: {len(stock_list)} -> {len(filtered)} (제외 {missing})")
    return filtered


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
):
    target_date_str = target_date.strftime("%Y%m%d") if target_date else None
    stock_list = stock_list or load_symbols()
    total = len(stock_list)

    if verbose:
        label = target_date_str or "latest"
        print(f"\n[INFO] 총 {total} 종목 스캔 시작 (기준일: {label}, config: {config.name})\n")

    candidates = []
    skipped = 0

    for idx, (code, name) in enumerate(stock_list, start=1):
        if verbose:
            print(f"[{idx}/{total}] {code} 검증중...", end="\r")

        daily_df = build_daily_bars(data_root, code, target_date_str)
        if daily_df is None or len(daily_df) < 5:
            skipped += 1
            continue

        candidate = evaluate_candidate(code, name, daily_df, config)
        candidates.append(candidate)

        if verbose:
            if candidate["eligible"]:
                print(f"[{idx}/{total}] {code} 후보 (score={candidate['score']:.2f})          ")
            else:
                reasons = candidate["fail_reasons"] or [candidate["skip_reason"] or "unknown"]
                print(f"[{idx}/{total}] {code} 탈락 ({', '.join(reasons)})          ")

    eligible_rows = [row for row in candidates if row["eligible"]]
    eligible_rows.sort(
        key=lambda row: (row["score"], row["amount_ma20"] or 0.0, row["atr_ratio"] or 0.0),
        reverse=True,
    )

    selected_rows = eligible_rows
    selection_mode = "strict_eligible"

    if config.max_picks is not None:
        selected_rows = eligible_rows[:config.max_picks]

    # Fallback: fill up to max_picks with scorable non-down-trend candidates.
    if config.max_picks is not None and len(selected_rows) < config.max_picks:
        selected_codes = {row["code"] for row in selected_rows}
        fallback_rows = [
            row for row in candidates
            if is_scorable_candidate(row)
            and row.get("trend_state") != "down"
            and row["code"] not in selected_codes
        ]
        fallback_rows.sort(
            key=lambda row: (row["score"], row["amount_ma20"] or 0.0),
            reverse=True,
        )
        needed = config.max_picks - len(selected_rows)
        supplement = fallback_rows[:needed]
        if supplement:
            selected_rows = [*selected_rows, *supplement]
            selection_mode = "strict_plus_fallback"
            for row in supplement:
                if "fallback_selected" not in row["soft_flags"]:
                    row["soft_flags"].append("fallback_selected")

    selected = [f"{row['code']},{row['name']}" for row in selected_rows]
    summary = summarize_candidates(candidates, selected_rows, skipped)

    if verbose:
        print("\n========== 스캔 완료 ==========")
        print(f"총종목: {total}")
        print(f"적격 수: {summary['eligible_pool_count']}")
        print(f"최종 선정: {summary['selected_count']}")
        print(f"스킵: {skipped}")

    if return_details:
        return {
            "selected": selected,
            "selected_rows": selected_rows,
            "eligible_rows": eligible_rows,
            "summary": summary,
            "config": config,
            "selection_mode": selection_mode,
        }
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Trade candidate scanner (daily-bar mode)")
    parser.add_argument("--date", type=str, default=None, help="Base date (YYYYMMDD)")
    parser.add_argument("--data-dir", type=str, default=None, help="Root data folder (default: data/)")
    parser.add_argument(
        "--config", type=str, default="balanced",
        choices=list(CONFIG_MAP.keys()),
        help="Scanner config preset",
    )
    parser.add_argument("--max-picks", type=int, default=None, help="Override config max_picks")
    parser.add_argument(
        "--history-window", type=int, default=DEFAULT_HISTORY_WINDOW,
        help="Number of recent trading days to include in comparison report",
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
        config = dc_replace(config, max_picks=args.max_picks)

    stock_list = load_symbols()
    stock_list = filter_stock_list_by_existing_data(stock_list, data_root)
    if not stock_list:
        raise SystemExit(f"{data_root} 에 스캔 가능한 데이터가 없습니다.")

    scan_result = scan(data_root, target_date, config=config, return_details=True, stock_list=stock_list)
    picks = scan_result["selected"]

    comparison_rows = []
    if args.history_window > 0:
        comparison_rows = build_history_comparison(data_root, stock_list, history_window=args.history_window)

    # Determine output directory
    if target_date:
        out_dir = data_root / target_date.strftime("%Y%m%d")
        out_dir.mkdir(exist_ok=True)
    else:
        date_dirs = sorted(
            d for d in data_root.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        )
        out_dir = date_dirs[-1] if date_dirs else data_root

    if picks:
        picks_file = out_dir / "picks.txt"
        picks_file.write_text("\n".join(picks), encoding="utf-8")
        print(f"추천 종목 리스트를 저장했습니다: {picks_file}")

    ranked_file = out_dir / "picks_ranked.txt"
    ranked_file.write_text(render_ranked_csv(scan_result["selected_rows"]), encoding="utf-8")
    print(f"점수 기반 랭킹 파일을 저장했습니다: {ranked_file}")

    report_file = out_dir / "scanner_report.md"
    report_file.write_text(
        render_report(data_root, target_date, config, scan_result, comparison_rows),
        encoding="utf-8",
    )
    print(f"스캐너 리포트를 저장했습니다: {report_file}")

    label = target_date.strftime("%Y%m%d") if target_date else "최신 데이터"
    print(f"\n[{label}] 기준 추천 종목:", picks)

    print(f"스캐너 리포트를 저장했습니다: {report_file}")

    if target_date:
        print(f"\n[{target_date.strftime('%Y%m%d')}] 기준 추천 종목:", picks)
    else:
        print("\n[최신 데이터] 기준 추천 종목:", picks)


