# -*- coding: utf-8 -*-
"""
KIS 1-Minute OHLCV Data Collector
- Fetches same-day 1-minute intraday data from KIS.
- Supports handling empty (no-trade) minutes.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 📁 Set project and internal module paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "examples_llm"))
sys.path.insert(0, str(project_root / "examples_llm" / "domestic_stock" / "inquire_time_itemchartprice"))

import kis_auth as ka
from inquire_time_itemchartprice import inquire_time_itemchartprice

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='KIS 1-min intraday data collector')
    parser.add_argument('--env', type=str, default='real', choices=['real', 'demo'],
                        help='API env for quotation call (real or demo). auth maps to prod/vps')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y%m%d'),
                        help='Target date to fetch data (format: YYYYMMDD)')
    parser.add_argument('--empty-bar-mode', type=str, default='nan', choices=['drop', 'nan', 'ffill'],
                        help='How to handle no-trade 1-minute bars: drop, nan, or ffill (default: nan)')
    parser.add_argument('--include-empty-bars', action='store_true',
                        help='Deprecated: same as --empty-bar-mode ffill')
    args = parser.parse_args()
    if args.include_empty_bars:
        args.empty_bar_mode = 'ffill'
    return args


def fetch_1min_data(stock_code, target_date, env_dv='real', empty_bar_mode='nan'):
    """Fetch same-day 1-min raw data and optionally fill empty bars."""

    all_df = []
    today_yyyymmdd = datetime.now().strftime("%Y%m%d")
    # For same-day collection, never query a future time because KIS can return repeated current values.
    if target_date == today_yyyymmdd:
        current_hour = min(datetime.now().strftime("%H%M%S"), "153000")
    else:
        current_hour = "153000"

    last_earliest_time = None

    page_count = 0
    max_pages = 20  # API returns up to ~30 rows per call; 20 pages cover full session with overlap.
    consecutive_errors = 0
    max_api_errors = 5

    # --- [1] This loop fetches 1-minute raw data from KIS server ---
    while True:
        page_count += 1
        if page_count > max_pages:
            logger.warning("%s reached max pages (%s). stop pagination.", stock_code, max_pages)
            break

        try:
            _, df = inquire_time_itemchartprice(
                env_dv=env_dv,
                fid_cond_mrkt_div_code="J",
                fid_input_iscd=stock_code,
                fid_input_hour_1=current_hour,
                fid_pw_data_incu_yn="Y",
                fid_etc_cls_code=""
            )
        except Exception as e:
            consecutive_errors += 1
            logger.warning(f"{stock_code} API error: {e}")
            if consecutive_errors >= max_api_errors:
                logger.warning("%s too many API errors (%s). skip symbol.", stock_code, max_api_errors)
                break
            time.sleep(1.0)
            continue

        consecutive_errors = 0

        if df is None or df.empty:
            break

        all_df.append(df)

        # Determine oldest row by value instead of relying on response row order.
        times = pd.to_numeric(df["stck_cntg_hour"], errors="coerce").dropna().astype("int64")
        if times.empty:
            break
        earliest_time_int = int(times.min())
        earliest_time = f"{earliest_time_int:06d}"

        # Stop if reached market open (09:00)
        if earliest_time_int <= 90000:
            break

        if last_earliest_time is not None and earliest_time_int >= last_earliest_time:
            logger.warning(
                "%s pagination stalled (current=%s, earliest=%s). stop to prevent duplicated bars.",
                stock_code,
                current_hour,
                earliest_time,
            )
            break
        last_earliest_time = earliest_time_int

        # Subtract 1 minute to move back in time for the next API call
        current_dt = datetime.strptime(earliest_time, "%H%M%S")
        next_dt = current_dt - timedelta(minutes=1)
        current_hour = next_dt.strftime("%H%M%S")

        time.sleep(0.15) 

    if not all_df:
        return None

    # --- [2] Data Post-Processing (1-minute data) ---
    final_df = pd.concat(all_df)

    # Normalize time strings before dedup/datetime parsing.
    final_df["stck_cntg_hour"] = final_df["stck_cntg_hour"].astype(str).str.zfill(6)
    final_df = final_df.drop_duplicates(subset=["stck_cntg_hour"], keep="last")
    
    # Numeric conversion
    cols = {"stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low", "stck_prpr": "close", "cntg_vol": "volume"}
    for old_col in cols.keys():
        final_df[old_col] = pd.to_numeric(final_df[old_col], errors="coerce")

    # Remove invalid source rows where all price fields are 0 or missing.
    price_cols = ["stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr"]
    valid_price_mask = final_df[price_cols].max(axis=1) > 0
    final_df = final_df[valid_price_mask].copy()

    if final_df.empty:
        return None

    # Create Datetime Index
    final_df["datetime"] = pd.to_datetime(
        target_date + final_df["stck_cntg_hour"],
        format="%Y%m%d%H%M%S"
    )
    final_df = final_df.set_index("datetime").sort_index()

    # Filter market hours (09:00 ~ 15:30)
    final_df = final_df.between_time("09:00", "15:30")

    # Keep 1-minute bars only.
    minute_df = final_df.rename(columns={
        'stck_oprc': 'open',
        'stck_hgpr': 'high',
        'stck_lwpr': 'low',
        'stck_prpr': 'close',
        'cntg_vol': 'volume',
    })[["open", "high", "low", "close", "volume"]]

    if empty_bar_mode == 'drop':
        # Keep only actual traded 1-minute bars.
        minute_df = minute_df[(minute_df["close"] > 0) & (minute_df["volume"] > 0)].dropna()
    else:
        # Keep a continuous 1-minute timeline (09:00 ~ 15:30) even when there are no trades.
        full_index = pd.date_range(
            start=pd.to_datetime(target_date + "090000", format="%Y%m%d%H%M%S"),
            end=pd.to_datetime(target_date + "153000", format="%Y%m%d%H%M%S"),
            freq="1min"
        )
        minute_df = minute_df.reindex(full_index)
        minute_df.index.name = "datetime"

        # Treat non-positive OHLC as invalid in no-trade intervals.
        minute_df.loc[minute_df["close"] <= 0, ["open", "high", "low", "close"]] = pd.NA
        minute_df["volume"] = minute_df["volume"].fillna(0).astype("int64")

        # Trim pre-open empty rows before first traded bar.
        first_valid_idx = minute_df["close"].first_valid_index()
        if first_valid_idx is None:
            return None
        minute_df = minute_df.loc[first_valid_idx:]

        if empty_bar_mode == 'ffill':
            # For no-trade bars, use previous close for OHLC.
            minute_df["close"] = minute_df["close"].ffill()
            minute_df["open"] = minute_df["open"].fillna(minute_df["close"])
            minute_df["high"] = minute_df["high"].fillna(minute_df["close"])
            minute_df["low"] = minute_df["low"].fillna(minute_df["close"])

    return minute_df


if __name__ == "__main__":
    args = get_args()
    target_date = args.date
    ka.auth(svr="prod" if args.env == "real" else "vps")

    # Directory for 1-minute data
    base_dir = Path(__file__).resolve().parent / "data" / target_date
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load symbol list
    symbols_path = Path(__file__).resolve().parent / "symbols.csv"
    if not symbols_path.exists():
        logger.error("❌ symbols.csv missing")
        sys.exit(1)

    stock_df = pd.read_csv(symbols_path)
    stock_list = stock_df["code"].astype(str).str.zfill(6).tolist()

    logger.info(f"🚀 Collecting 1-min data for {target_date} (env={args.env})")

    for idx, code in enumerate(stock_list, start=1):
        try:
            df = fetch_1min_data(code, target_date, env_dv=args.env, empty_bar_mode=args.empty_bar_mode)

            if df is not None and not df.empty:
                df.to_csv(base_dir / f"{code}.csv")
                print(f"[{idx}/{len(stock_list)}] {code} ✅ saved ({len(df)} 1-min bars)")
            else:
                print(f"[{idx}/{len(stock_list)}] {code} ⚠️ no active trade data")

        except Exception as e:
            print(f"[{idx}/{len(stock_list)}] {code} ❌ error: {e}")

        time.sleep(0.2)

    print(f"\n✨ Process complete. Files saved in: {base_dir}")

