# -*- coding: utf-8 -*-
"""Collect date-based minute data in the same schema as 20260401 files.

Output schema:
- datetime, open, high, low, close, volume, market
- R73 indicator columns (MA_5, VOL_MA20, BB_*, RSI*, STOCH*, WILLIAMS*, MACD*, DI/ADX, VWAP, OBV*)

Key behavior:
- Collects by date from KIS minute API (`inquire_time_dailychartprice`).
- Uses market priority NX > J > UN for overlapping timestamps.
- Supports NXT pre/after sessions (08:00~19:59) when symbol is NXT-tradeable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_user" / "domestic_stock"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_time_dailychartprice"))

import kis_auth as ka
import domestic_stock_functions as dsf
from inquire_time_dailychartprice import inquire_time_dailychartprice


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# R73 indicator parameters
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
OBV_MA_PERIOD = 10
MACD_FAST = 5
MACD_SLOW = 12
MACD_SIGNAL_PERIOD = 4
ADX_PERIOD = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Date-based KIS minute collector (20260401-compatible schema)")
    parser.add_argument("--env", type=str, default="real", choices=["real", "demo"], help="API environment")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="Target date YYYYMMDD")
    parser.add_argument("--symbols-file", type=str, default=str(SCRIPT_DIR / "symbols.csv"), help="Path to symbols.csv")
    parser.add_argument("--data-root", type=str, default=str(SCRIPT_DIR / "data"), help="Output data root")
    parser.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between symbols")
    return parser.parse_args()


def _is_truthy_flag(value) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if text in {"Y", "1", "TRUE", "T", "O", "YES"}:
        return True
    if text in {"N", "0", "FALSE", "F", "X", "NO"}:
        return False
    return None


def probe_nxt_tradeable(code: str) -> bool:
    stock_info_fn = getattr(dsf, "search_stock_info", None)
    if not callable(stock_info_fn):
        return False

    try:
        result = stock_info_fn(prdt_type_cd="300", pdno=code)
    except Exception as exc:
        logger.warning("NXT probe failed for %s: %s", code, exc)
        return False

    if result is None or getattr(result, "empty", True):
        return False

    row = result.iloc[-1]
    for key in ("cptt_trad_tr_psbl_yn", "nxt_tr_stop_yn", "tr_stop_yn"):
        if key not in result.columns:
            continue
        flag = _is_truthy_flag(row.get(key))
        if flag is None:
            continue
        return flag if key == "cptt_trad_tr_psbl_yn" else (not flag)

    return False


def _parse_one_market(code: str, market_div: str, target_date: str, max_pages: int = 15) -> pd.DataFrame | None:
    # Regular market: until 15:30, NXT: until 20:00.
    current_hour = "200000" if market_div == "NX" else "153000"
    all_df: list[pd.DataFrame] = []
    last_earliest_time: int | None = None

    for _ in range(max_pages):
        try:
            _, df = inquire_time_dailychartprice(
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
                fid_input_hour_1=current_hour,
                fid_input_date_1=target_date,
                fid_pw_data_incu_yn="Y",
                fid_fake_tick_incu_yn="",
            )
        except Exception as exc:
            logger.warning("chart fetch failed %s (%s): %s", code, market_div, exc)
            break

        if df is None or df.empty:
            break

        if "stck_bsop_date" in df.columns:
            df = df[df["stck_bsop_date"].astype(str) == target_date].copy()
        if df.empty:
            break

        all_df.append(df)

        times = pd.to_numeric(df.get("stck_cntg_hour"), errors="coerce").dropna().astype("int64")
        if times.empty:
            break

        earliest = int(times.min())
        if last_earliest_time is not None and earliest >= last_earliest_time:
            break
        last_earliest_time = earliest

        if earliest <= 80000:
            break

        next_dt = datetime.strptime(f"{earliest:06d}", "%H%M%S") - timedelta(minutes=1)
        current_hour = next_dt.strftime("%H%M%S")
        time.sleep(0.1)

    if not all_df:
        return None

    merged = pd.concat(all_df, ignore_index=True)
    merged["stck_cntg_hour"] = merged["stck_cntg_hour"].astype(str).str.zfill(6)

    for col in ("stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr", "cntg_vol"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["datetime"] = pd.to_datetime(
        target_date + merged["stck_cntg_hour"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    merged = merged.dropna(subset=["datetime"]).copy()

    merged["_has_trade"] = (merged["cntg_vol"].fillna(0) > 0).astype("int64")
    merged = merged.sort_values(
        by=["datetime", "_has_trade", "cntg_vol"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["datetime"], keep="first")

    valid_price = merged[["stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr"]].max(axis=1) > 0
    merged = merged[valid_price].copy()
    if merged.empty:
        return None

    out = merged.rename(
        columns={
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_prpr": "close",
            "cntg_vol": "volume",
        }
    )[["datetime", "open", "high", "low", "close", "volume"]]
    out["market"] = market_div
    return out.sort_values("datetime").reset_index(drop=True)


def fetch_symbol_data(code: str, target_date: str, include_nxt: bool) -> pd.DataFrame | None:
    market_candidates = ["J", "UN"]
    if include_nxt:
        market_candidates = ["NX", "J", "UN"]

    collected: list[pd.DataFrame] = []
    for market in market_candidates:
        data = _parse_one_market(code, market, target_date)
        if data is not None and not data.empty:
            collected.append(data)

    if not collected:
        return None

    merged = pd.concat(collected, ignore_index=True)

    # One row per minute. Priority: NX > J > UN.
    priority = {"NX": 0, "J": 1, "UN": 2}
    merged["_priority"] = merged["market"].map(priority).fillna(9).astype(int)
    merged = merged.sort_values(["datetime", "_priority"]).drop_duplicates(subset=["datetime"], keep="first")
    merged = merged.drop(columns=["_priority"]).sort_values("datetime").reset_index(drop=True)

    return merged


def calculate_r73_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
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

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
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

    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


def enrich_with_strategy_indicators(minute_df: pd.DataFrame) -> pd.DataFrame:
    base = minute_df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base = base.dropna(subset=["datetime"]).sort_values("datetime")
    if base.empty:
        return base

    idx = base.set_index("datetime")
    bars_3m = idx.resample("3min", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    bars_3m = bars_3m.dropna(subset=["open", "high", "low", "close"])
    if bars_3m.empty:
        return base

    bars_3m = calculate_r73_indicators(bars_3m)
    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    bars_for_merge = bars_3m[indicator_cols].reset_index().rename(columns={"index": "datetime"})

    enriched = pd.merge_asof(
        base.sort_values("datetime"),
        bars_for_merge.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    return enriched


def load_symbols(symbols_file: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(symbols_file)
    if "code" not in df.columns:
        raise ValueError(f"'code' column not found in {symbols_file}")

    names = df["name"].astype(str).tolist() if "name" in df.columns else [""] * len(df)
    pairs: list[tuple[str, str]] = []
    for code, name in zip(df["code"].astype(str), names):
        code6 = code.strip().zfill(6)
        if code6:
            pairs.append((code6, str(name).strip() or code6))

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for code, name in pairs:
        if code in seen:
            continue
        seen.add(code)
        deduped.append((code, name))
    return deduped


def main() -> None:
    args = parse_args()

    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        raise SystemExit(f"Invalid --date '{args.date}'. expected YYYYMMDD")

    symbols_file = Path(args.symbols_file)
    if not symbols_file.is_file():
        raise SystemExit(f"symbols file not found: {symbols_file}")

    output_dir = Path(args.data_root) / args.date
    output_dir.mkdir(parents=True, exist_ok=True)

    ka.auth(svr="prod" if args.env == "real" else "vps")

    symbols = load_symbols(symbols_file)
    logger.info("collect start: date=%s, symbols=%d", args.date, len(symbols))

    saved_count = 0
    empty_count = 0

    for idx, (code, name) in enumerate(symbols, start=1):
        try:
            nxt_tradeable = probe_nxt_tradeable(code)
            df = fetch_symbol_data(code=code, target_date=args.date, include_nxt=nxt_tradeable)
        except Exception as exc:
            logger.error("[%d/%d] %s(%s) | fetch error: %s", idx, len(symbols), code, name, exc)
            continue

        if df is None or df.empty:
            empty_count += 1
            logger.info("[%d/%d] %s(%s) | NXT=%s | no data", idx, len(symbols), code, name, nxt_tradeable)
        else:
            df = enrich_with_strategy_indicators(df)
            file_path = output_dir / f"{code}.csv"
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            saved_count += 1
            logger.info(
                "[%d/%d] %s(%s) | NXT=%s | saved=%d rows -> %s",
                idx,
                len(symbols),
                code,
                name,
                nxt_tradeable,
                len(df),
                file_path,
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    logger.info("done: saved=%d, empty=%d, out=%s", saved_count, empty_count, output_dir)


if __name__ == "__main__":
    main()

