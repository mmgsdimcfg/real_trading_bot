# -*- coding: utf-8 -*-
"""
r001_data_collect_symbols_daily.py

Purpose:
- Collect minute bars for a target date from KIS APIs.
- Build normalized intraday outputs and indicator columns used by live/sim flows.

Output schema:
- datetime, open, high, low, close, volume, market
- R76 indicator columns (MA_5, VOL_MA20, BB_*, RSI*, STOCH*, WILLIAMS*, MACD*, DI/ADX, VWAP, OBV*)

Key behavior:
- Collects by date from KIS minute API (`inquire_time_dailychartprice`).
- Uses market priority NX > J > UN for overlapping timestamps.
- Supports NXT pre/after sessions (08:00~19:59) when symbol is NXT-tradeable.
- Exports 10-second interpolated bars (`_10s.txt`) for r007 simulation input.
- Optionally saves legacy `_1m/_3m/_20s` files for backward compatibility.

Usage examples:
- Single code on specific date (regular market only):
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --code 067310
- Single code on specific date (include NXT market):
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --code 067310 --nxt
- Multiple codes on specific date:
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --code 067310,005930 --nxt
- Full list from symbols file:
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --symbols-file xgraph/auto_trading/r009_universe_symbols_master.txt

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <collector/live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-07-21] type=feat owner=copilot
    summary: KIS inquire_price API로 실서버 52주 고가/저가(w52_hgpr/w52_lwpr) 스냅샷 추가 수집
      (fetch_52w_high_low), 종목별 누적하여 _52w_high_low.json으로 저장. r002가 이 값을 직접 읽어
      high_52w_ratio 계산에 쓰면 로컬 일봉(약 13~20일치) 최고값으로 진짜 52주 고가를 대체하던
      왜곡을 제거함. 종목당 API 1회 추가 호출(기존 일봉 fetch와 동일한 루프/env_dv/rate-limit 패턴 재사용).
    impact: collector
    compatibility: backward-compatible (API 실패 시 None 반환, r002는 기존 로컬 계산으로 폴백)
- [2026-06-28] type=fix owner=copilot
    summary: 일봉 lookback 10->20일, 10초봉 보간 이상값 클리핑(±15%초과), 3분봉 기본 저장
    impact: collector
    compatibility: backward-compatible
- [2026-06-25] type=feat owner=copilot
    summary: 일봉 데이터 취득 추가 (fetch_and_save_daily_ohlcv); inquire_daily_itemchartprice API로 이전 10 영업일 일봉 OHLCV 저장 ({code}_daily.csv) - r002 우하향 종목 필터에 사용
    impact: collector
    compatibility: backward-compatible
- [2026-05-10] type=docs owner=copilot
    summary: added standardized file header and expandable update-log format.
    impact: collector
    compatibility: backward-compatible
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("OPEN_TRADING_API_ROOT", str(Path.home() / "git" / "open-trading-api")))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_user" / "domestic_stock"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_time_dailychartprice"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_daily_itemchartprice"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_llm" / "domestic_stock" / "inquire_price"))

import kis_auth as ka
import domestic_stock_functions as dsf
from inquire_time_dailychartprice import inquire_time_dailychartprice
from r003_define_config import (
    ADX_PERIOD,
    BB_PERIOD,
    BB_STD_MULTIPLIER,
    MA_PERIOD,
    MACD_FAST,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW,
    OBV_MA_PERIOD,
    RSI_PERIOD,
    RSI_SIGNAL_PERIOD,
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    VOLUME_MA_PERIOD,
    WILLIAMS_D_PERIOD,
    WILLIAMS_R_PERIOD,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# Indicator parameters are imported from r003_define_config so collector, live,
# and simulation stay in sync when strategy settings change.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Date-based KIS minute collector (20260401-compatible schema)")
    parser.add_argument("--env", type=str, default="real", choices=["real", "demo"], help="API environment")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYYMMDD, or comma-separated list (e.g. 20260508,20260509,20260510)")
    parser.add_argument("--code", type=str, default="", help="Collect only this code (6-digit). Comma-separated supported")
    parser.add_argument("--symbols-file", type=str, default=str(SCRIPT_DIR / "r009_universe_symbols_master.txt"), help="Path to r009_universe_symbols_master.txt")
    parser.add_argument("--watchlist-file", type=str, default="", help="r008-style watchlist file(s) with code,name per line. Comma-separated multiple paths.")
    parser.add_argument("--watchlist-only", action="store_true", help="Use --watchlist-file as the only symbol source, ignoring --symbols-file.")
    parser.add_argument("--data-root", type=str, default=str(SCRIPT_DIR / "data"), help="Output data root")
    parser.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between symbols")
    parser.add_argument("--nxt", action="store_true", help="Include NXT market data (08:00~20:00)")
    parser.add_argument(
        "--save-legacy-files",
        action="store_true",
        help="Also save legacy _1m/_3m/_20s files in addition to default _10s output",
    )
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

    last_exc = None
    for attempt in range(2):
        try:
            result = stock_info_fn(prdt_type_cd="300", pdno=code)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.debug("NXT probe attempt 1 failed for %s, retrying: %s", code, exc)
                time.sleep(0.3)

    if last_exc is not None:
        logger.warning("NXT probe failed for %s (both attempts): %s", code, last_exc)
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

    # Filter data based on market hours
    if not include_nxt:
        merged = merged[merged["datetime"].dt.time >= datetime.strptime("09:00:00", "%H:%M:%S").time()]

    # One row per minute. Priority: NX > J > UN.
    priority = {"NX": 0, "J": 1, "UN": 2}
    merged["_priority"] = merged["market"].map(priority).fillna(9).astype(int)
    merged = merged.sort_values(["datetime", "_priority"]).drop_duplicates(subset=["datetime"], keep="first")
    merged = merged.drop(columns=["_priority"]).sort_values("datetime").reset_index(drop=True)

    return merged


def calculate_r76_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def interpolate_to_20sec(minute_df: pd.DataFrame) -> pd.DataFrame:
    """1遺꾨큺 ?곗씠?곕? 20珥?媛꾧꺽?쇰줈 ?좏삎 蹂닿컙???뺤옣?쒕떎.

    Note:
    - ???곗씠?곕뒗 ?덇굅???명솚 紐⑹쟻??蹂닿컙 寃곌낵?대ŉ,
      ?쒕쾭?먯꽌 20珥?二쇨린濡?吏곸젒 ?섏쭛??泥닿껐/?멸? ?곗씠?곌? ?꾨땲??
    """
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    
    if df.empty:
        return df
    
    # datetime???몃뜳?ㅻ줈 吏??    df_indexed = df.set_index("datetime")
    
    # 20珥?媛꾧꺽???쒓컙 ?앹꽦
    time_range_20sec = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq="20s"
    )
    
    # datetime ?몃뜳?ㅻ줈 由ъ씤?깆떛
    df_reindexed = df_indexed.reindex(time_range_20sec.union(df_indexed.index)).sort_index()
    
    # 媛寃??곗씠???좏삎 蹂닿컙
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df_reindexed[col] = df_reindexed[col].interpolate(method="linear", limit_direction="both")
    
    # 嫄곕옒?됱? 20珥?遊??⑷퀎媛 ?먮낯 遺꾨큺 ?⑷퀎? 理쒕????쇱튂?섎룄濡?遺꾪븷?쒕떎.
    step_seconds = 20.0
    if len(df.index) >= 2:
        src_seconds = df["datetime"].diff().dropna().dt.total_seconds().median()
        if pd.notna(src_seconds) and src_seconds > 0:
            expansion = max(1.0, round(float(src_seconds) / step_seconds))
        else:
            expansion = 3.0
    else:
        expansion = 3.0

    df_reindexed["volume"] = df_reindexed["volume"].ffill()
    df_reindexed["volume"] = pd.to_numeric(df_reindexed["volume"], errors="coerce") / expansion
    df_reindexed["volume"] = df_reindexed["volume"].fillna(0)
    
    # Forward-fill market column if present.
    if "market" in df_reindexed.columns:
        df_reindexed["market"] = df_reindexed["market"].ffill()
    
    # 吏??而щ읆? ?댁쟾 媛믪쑝濡?梨꾩?(1遊??댁긽?대㈃ 洹몃?濡?蹂듭궗)
    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    for col in indicator_cols:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill()
    
    # Keep only 20-second rows.
    df_20sec = df_reindexed.loc[time_range_20sec].reset_index()
    df_20sec.rename(columns={"index": "datetime"}, inplace=True)
    
    return df_20sec


def interpolate_to_10sec(minute_df: pd.DataFrame) -> pd.DataFrame:
    """1遺꾨큺 ?곗씠?곕? 10珥?媛꾧꺽?쇰줈 ?좏삎 蹂닿컙???뺤옣?쒕떎.

    Note:
    - ???곗씠?곕뒗 ?덇굅???명솚/?쒕??덉씠???뺣????μ긽 紐⑹쟻??蹂닿컙 寃곌낵?대ŉ,
      ?쒕쾭?먯꽌 10珥?二쇨린濡?吏곸젒 ?섏쭛??泥닿껐 ?곗씠?곌? ?꾨땲??
    """
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if df.empty:
        return df

    df_indexed = df.set_index("datetime")
    time_range_10sec = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq="10s"
    )

    df_reindexed = df_indexed.reindex(time_range_10sec.union(df_indexed.index)).sort_index()

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df_reindexed[col] = df_reindexed[col].interpolate(method="linear", limit_direction="both")

    # 10초 선형 보간 아티팩트 제거: 연속 봉 간 ±15% 초과 변화는 이전 값으로 대체
    for col in price_cols:
        pct_chg = df_reindexed[col].pct_change().abs()
        mask = pct_chg > 0.15
        df_reindexed[col] = df_reindexed[col].where(~mask, df_reindexed[col].ffill())

    step_seconds = 10.0
    if len(df.index) >= 2:
        src_seconds = df["datetime"].diff().dropna().dt.total_seconds().median()
        if pd.notna(src_seconds) and src_seconds > 0:
            expansion = max(1.0, round(float(src_seconds) / step_seconds))
        else:
            expansion = 6.0
    else:
        expansion = 6.0

    df_reindexed["volume"] = df_reindexed["volume"].ffill()
    df_reindexed["volume"] = pd.to_numeric(df_reindexed["volume"], errors="coerce") / expansion
    df_reindexed["volume"] = df_reindexed["volume"].fillna(0)

    if "market" in df_reindexed.columns:
        df_reindexed["market"] = df_reindexed["market"].ffill()

    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    for col in indicator_cols:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill()

    df_10sec = df_reindexed.loc[time_range_10sec].reset_index()
    df_10sec.rename(columns={"index": "datetime"}, inplace=True)
    return df_10sec


def build_3min_indicator_frame(minute_df: pd.DataFrame) -> pd.DataFrame:
    base = minute_df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base = base.dropna(subset=["datetime"]).sort_values("datetime")
    if base.empty:
        return pd.DataFrame(columns=base.columns)

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
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    bars_3m = calculate_r76_indicators(bars_3m)
    return bars_3m.reset_index().rename(columns={"index": "datetime"})


def enrich_with_strategy_indicators(minute_df: pd.DataFrame) -> pd.DataFrame:
    base = minute_df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base = base.dropna(subset=["datetime"]).sort_values("datetime")
    if base.empty:
        return base

    bars_3m = build_3min_indicator_frame(base)
    if bars_3m.empty:
        return base

    indicator_cols = [
        "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_STD", "BB_UPPER", "BB_LOWER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
        "VWAP", "OBV", "OBV_MA",
    ]
    bars_for_merge = bars_3m[["datetime", *indicator_cols]].copy()

    enriched = pd.merge_asof(
        base.sort_values("datetime"),
        bars_for_merge.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    return enriched


def fetch_and_save_daily_ohlcv(
    code: str,
    name: str,
    env_dv: str,
    target_date: str,
    output_dir: Path,
    lookback_days: int = 20,
) -> bool:
    """Fetch actual daily (일봉) OHLCV for the past lookback_days business days.
    Saves {code}_{name}_daily.csv to output_dir for r002 downtrend filter.
    Uses KIS inquire_daily_itemchartprice API.  Returns True on success.
    """
    try:
        from inquire_daily_itemchartprice import inquire_daily_itemchartprice as _daily_api
    except ImportError:
        logger.debug("inquire_daily_itemchartprice not importable; skipping daily OHLCV for %s", code)
        return False

    target_dt = datetime.strptime(target_date, "%Y%m%d")
    date_from = (target_dt - timedelta(days=lookback_days * 2 + 5)).strftime("%Y%m%d")

    try:
        _, df2 = _daily_api(
            env_dv=env_dv,
            fid_cond_mrkt_div_code="J",
            fid_input_iscd=code,
            fid_input_date_1=date_from,
            fid_input_date_2=target_date,
            fid_period_div_code="D",
            fid_org_adj_prc="0",
        )
    except Exception as exc:
        logger.debug("daily OHLCV fetch failed %s: %s", code, exc)
        return False

    if df2 is None or df2.empty:
        return False

    col_map = {
        "stck_bsop_date": "date",
        "stck_oprc": "open",
        "stck_hgpr": "high",
        "stck_lwpr": "low",
        "stck_clpr": "close",
        "acml_vol": "volume",
        "acml_tr_pbmn": "amount",
    }
    df2 = df2.rename(columns={k: v for k, v in col_map.items() if k in df2.columns})
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in df2.columns]
    if "date" not in keep or "close" not in keep:
        return False

    df2 = df2[keep].copy()
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=["close"]).sort_values("date").tail(lookback_days)
    if df2.empty:
        return False

    safe_name = str(name).replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"{code}_{safe_name}_daily.csv"
    df2.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.debug("[daily] %s(%s) | %d rows -> %s", code, name, len(df2), out_path)
    return True


def fetch_52w_high_low(code: str, env_dv: str) -> dict | None:
    """Fetch the real 52-week high/low from KIS inquire_price (fields
    w52_hgpr/w52_lwpr). Used by r002 to compute high_52w_ratio against the
    true 52-week high instead of the max of a short local daily-bar window.
    Returns None on any failure (r002 falls back to its own local calc).
    """
    try:
        from inquire_price import inquire_price as _price_api
    except ImportError:
        logger.debug("inquire_price not importable; skipping 52w snapshot for %s", code)
        return None

    try:
        df = _price_api(env_dv=env_dv, fid_cond_mrkt_div_code="J", fid_input_iscd=code)
    except Exception as exc:
        logger.debug("52w price snapshot fetch failed %s: %s", code, exc)
        return None

    if df is None or df.empty:
        return None

    row = df.iloc[0]
    try:
        w52_high = float(row.get("w52_hgpr"))
        w52_low = float(row.get("w52_lwpr"))
    except (TypeError, ValueError):
        return None
    if w52_high <= 0:
        return None

    return {"w52_high": w52_high, "w52_low": w52_low}


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


def _load_watchlist_file(path: Path) -> list[tuple[str, str]]:
    """Load code,name pairs from an r008-style watchlist (# comment lines skipped)."""
    pairs: list[tuple[str, str]] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=encoding) as _f:
                for line in _f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",", 1)
                    code = parts[0].strip().zfill(6)
                    if not code.isdigit() or len(code) != 6:
                        continue
                    name = parts[1].strip() if len(parts) > 1 else code
                    pairs.append((code, name))
            break
        except UnicodeDecodeError:
            continue
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for code, name in pairs:
        if code not in seen:
            seen.add(code)
            deduped.append((code, name))
    return deduped


def _compute_prev_close_from_data(code: str, target_date: str, data_root: Path) -> float | None:
    """Return last-bar close from the most recent prior trading day already collected.

    Replicates r006 fetch_prev_close() without extra API calls, so r007 can apply
    MAX_BUY_RISE_PCT_FROM_PREV_CLOSE on stored data.
    """
    import json as _json_pc
    target = datetime.strptime(target_date, "%Y%m%d").date()
    for back in range(1, 15):
        prior = (target - timedelta(days=back)).strftime("%Y%m%d")
        prior_dir = data_root / prior
        if not prior_dir.is_dir():
            continue
        daily_close_path = prior_dir / "_daily_close.json"
        if daily_close_path.exists():
            try:
                with open(daily_close_path, encoding="utf-8") as _f:
                    dc = _json_pc.load(_f)
                val = dc.get(str(code).zfill(6))
                if val is not None:
                    return float(val)
            except Exception:
                pass
        for p in sorted(prior_dir.glob(f"{code}_*_10s.txt")):
            try:
                import pandas as _pd_pc
                _df = _pd_pc.read_csv(p, usecols=["close"])
                _col = _df["close"].dropna()
                if len(_col) > 0:
                    return float(_col.iloc[-1])
            except Exception:
                pass
    return None


def _parse_code_filter(raw: str) -> set[str]:
    if not raw:
        return set()
    tokens = [part.strip() for part in raw.split(",")]
    return {token.zfill(6) for token in tokens if token}


def resolve_target_date(date_arg: str | None, symbols: list[tuple[str, str]]) -> str:
    if date_arg:
        datetime.strptime(date_arg, "%Y%m%d")
        return date_arg

    # Auto-select latest tradeable date when --date is omitted (weekend/holiday safe).
    probe_symbols = symbols[:8] if symbols else []
    today = datetime.now().date()
    for back in range(0, 14):
        target = (today - timedelta(days=back)).strftime("%Y%m%d")
        for code, _ in probe_symbols:
            try:
                probe_df = fetch_symbol_data(code=code, target_date=target, include_nxt=True)
            except Exception:
                probe_df = None
            if probe_df is not None and not probe_df.empty:
                if back > 0:
                    logger.info("--date omitted. auto-selected latest tradeable date: %s", target)
                return target

    fallback = today.strftime("%Y%m%d")
    logger.warning("could not detect recent tradeable date automatically. fallback to today: %s", fallback)
    return fallback


def main() -> None:
    args = parse_args()

    include_nxt = args.nxt

    symbols_file = Path(args.symbols_file)
    if not symbols_file.is_file():
        raise SystemExit(f"symbols file not found: {symbols_file}")

    if args.watchlist_only:
        symbols = []
    else:
        symbols = load_symbols(symbols_file)

    # Merge extra symbols from --watchlist-file (r008 watchlist)
    if args.watchlist_file:
        wl_paths = [p.strip() for p in args.watchlist_file.split(",") if p.strip()]
        for wl_path_str in wl_paths:
            wl_path = Path(wl_path_str)
            if not wl_path.is_file():
                logger.warning("watchlist-file not found (skipped): %s", wl_path)
                continue
            wl_syms = _load_watchlist_file(wl_path)
            existing_codes = {c for c, _ in symbols}
            added = 0
            for c, n in wl_syms:
                if c not in existing_codes:
                    symbols.append((c, n))
                    existing_codes.add(c)
                    added += 1
            logger.info(
                "watchlist-file %s: %d symbols added (%d total after merge)",
                wl_path.name, added, len(symbols),
            )

    # Parse --date: single value or comma-separated list (e.g. 20260508,20260509,20260510)
    raw_dates = [d.strip() for d in (args.date or "").split(",") if d.strip()]
    if raw_dates:
        for d in raw_dates:
            try:
                datetime.strptime(d, "%Y%m%d")
            except ValueError:
                raise SystemExit(f"Invalid --date '{d}'. expected YYYYMMDD")
        target_dates = raw_dates
    else:
        try:
            target_dates = [resolve_target_date(None, symbols)]
        except ValueError:
            raise SystemExit("Could not resolve a target date automatically.")

    ka.auth(svr="prod" if args.env == "real" else "vps")
    selected_codes = _parse_code_filter(args.code)
    if selected_codes:
        symbols = [(code, name) for code, name in symbols if code in selected_codes]
        if not symbols:
            raise SystemExit(f"No matching symbols from --code: {','.join(sorted(selected_codes))}")

    for target_date in target_dates:
        output_dir = Path(args.data_root) / target_date
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("collect start: date=%s, symbols=%d, include_nxt=%s", target_date, len(symbols), include_nxt)

        saved_count = 0
        empty_count = 0
        nxt_flags: dict[str, bool] = {}
        saved_symbols: list[tuple[str, str]] = []
        w52_map: dict[str, dict] = {}

        for idx, (code, name) in enumerate(symbols, start=1):

            nxt_tradeable = False
            df = None
            last_error = None

            # 理쒕? 2???쒕룄
            for attempt in range(2):
                try:
                    nxt_tradeable = probe_nxt_tradeable(code) if include_nxt else False
                    nxt_flags[code] = nxt_tradeable
                    df = fetch_symbol_data(code=code, target_date=target_date, include_nxt=nxt_tradeable)
                    last_error = None
                    break  # ?깃났?섎㈃ 猷⑦봽 ?덉텧
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        logger.warning("[%d/%d] %s(%s) | fetch error (retry): %s", idx, len(symbols), code, name, exc)
                        time.sleep(0.5)  # ?ъ떆?????좉퉸 ?湲?
            # 2???쒕룄 紐⑤몢 ?ㅽ뙣??寃쎌슦
            if last_error is not None:
                logger.error("[%d/%d] %s(%s) | fetch error (final): %s", idx, len(symbols), code, name, last_error)
                nxt_flags[code] = False
                empty_count += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)

                continue

            if df is None or df.empty:

                empty_count += 1
                logger.info("[%d/%d] %s(%s) | NXT=%s | no data", idx, len(symbols), code, name, nxt_tradeable)
            else:
                df_10s = interpolate_to_10sec(df)

                df_10s = calculate_r76_indicators(df_10s)
                
                safe_name = str(name).replace("/", "_").replace("\\", "_")
                file_10s_path = output_dir / f"{code}_{safe_name}_10s.txt"
                df_10s.to_csv(file_10s_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")

                # 3분봉 파일 기본 저장 (r006 전략 로직의 메인 신호 소스)
                df_3m = build_3min_indicator_frame(df)
                file_3m_path = output_dir / f"{code}_{safe_name}_3m.txt"
                df_3m.to_csv(file_3m_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")
                legacy_log = f" | 3m={len(df_3m)} -> {file_3m_path}"
                if args.save_legacy_files:
                    df_1m = enrich_with_strategy_indicators(df)
                    df_20s = interpolate_to_20sec(df_1m)

                    file_1m_path = output_dir / f"{code}_{safe_name}_1m.txt"
                    legacy_20s_path = output_dir / f"{code}_{safe_name}_20s.txt"

                    df_1m.to_csv(file_1m_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")
                    df_20s.to_csv(legacy_20s_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")
                    legacy_log += (
                        f" | 1m={len(df_1m)} -> {file_1m_path}"
                        f" | 20s(interpolated)={len(df_20s)} -> {legacy_20s_path}"
                    )

                # 일봉 데이터 취득 및 저장 (r002 우하향 종목 필터용)
                fetch_and_save_daily_ohlcv(
                    code=code, name=name, env_dv=args.env,
                    target_date=target_date, output_dir=output_dir,
                )

                # Real 52-week high/low snapshot (KIS inquire_price) for r002's
                # high_52w_ratio, replacing its short local daily-bar max.
                w52 = fetch_52w_high_low(code=code, env_dv=args.env)
                if w52:
                    w52_map[code] = w52

                saved_count += 1
                saved_symbols.append((code, name))
                logger.info(
                    "[%d/%d] %s(%s) | NXT=%s | 10s(interpolated)=%d -> %s%s",
                    idx,
                    len(symbols),
                    code,
                    name,
                    nxt_tradeable,
                    len(df_10s),
                    file_10s_path,
                    legacy_log,
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

        # Save NXT tradeable flags for live/sim scripts.
        if nxt_flags:
            nxt_flags_path = output_dir / "nxt_flags.json"
            with open(nxt_flags_path, "w", encoding="utf-8") as _f:
                json.dump(nxt_flags, _f, ensure_ascii=False, indent=2)
            logger.info("saved NXT flags (%d codes): %s", len(nxt_flags), nxt_flags_path)

        # Save daily close (last bar close) for each symbol used as next-day prev_close in r007.
        daily_close_map: dict[str, float] = {}
        for _dc_code, _dc_name in saved_symbols:
            _safe = str(_dc_name).replace("/", "_").replace("\\", "_")
            _f10 = output_dir / f"{_dc_code}_{_safe}_10s.txt"
            if _f10.exists():
                try:
                    _dc_df = pd.read_csv(_f10, usecols=["close"])
                    _dc_col = _dc_df["close"].dropna()
                    if len(_dc_col) > 0:
                        daily_close_map[_dc_code] = float(_dc_col.iloc[-1])
                except Exception:
                    pass
        if daily_close_map:
            daily_close_path = output_dir / "_daily_close.json"
            with open(daily_close_path, "w", encoding="utf-8") as _f:
                json.dump(daily_close_map, _f, ensure_ascii=False, indent=2)
            logger.info("saved daily_close.json (%d codes): %s", len(daily_close_map), daily_close_path)

        if w52_map:
            w52_path = output_dir / "_52w_high_low.json"
            with open(w52_path, "w", encoding="utf-8") as _f:
                json.dump(w52_map, _f, ensure_ascii=False, indent=2)
            logger.info("saved 52w_high_low.json (%d codes): %s", len(w52_map), w52_path)

        # Save prev_close from prior trading day data (mirrors r006 fetch_prev_close).
        prev_close_map: dict[str, float] = {}
        for _pc_code, _ in saved_symbols:
            _pc = _compute_prev_close_from_data(_pc_code, target_date, output_dir.parent)
            if _pc is not None:
                prev_close_map[_pc_code] = _pc
        if prev_close_map:
            prev_close_path = output_dir / "_prev_close.json"
            with open(prev_close_path, "w", encoding="utf-8") as _f:
                json.dump(prev_close_map, _f, ensure_ascii=False, indent=2)
            logger.info("saved prev_close.json (%d codes): %s", len(prev_close_map), prev_close_path)

        # Save date-scoped picks file for r007 simulation input.
        if saved_symbols:
            picks_lines = [f"{code},{name}" for code, name in saved_symbols]
            picks_payload = "\n".join(picks_lines) + "\n"

            underscored_dated_picks = output_dir / f"_{target_date}_picks.txt"

            underscored_dated_picks.write_text(picks_payload, encoding="utf-8-sig")
            logger.info(
                "saved picks file (%d codes): %s",
                len(saved_symbols),
                underscored_dated_picks,
            )

        logger.info("done: date=%s saved=%d, empty=%d, out=%s", target_date, saved_count, empty_count, output_dir)


if __name__ == "__main__":
    main()


