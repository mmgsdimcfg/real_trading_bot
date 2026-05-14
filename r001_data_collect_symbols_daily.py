# -*- coding: utf-8 -*-
"""R76 date-based market data collector.

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

Usage examples:
- Single code on specific date:
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --code 067310
- Multiple codes on specific date:
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --code 067310,005930
- Multiple dates (comma-separated):
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508,20260509,20260510
- Full list from symbols file:
    python xgraph/auto_trading/r001_data_collect_symbols_daily.py --date 20260508 --symbols-file xgraph/auto_trading/r009_universe_symbols_master.txt

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <collector/live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-05-10] type=docs owner=copilot
    summary: added standardized file header and expandable update-log format.
    impact: collector
    compatibility: backward-compatible
"""

from __future__ import annotations

import argparse
import json
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
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# R76 indicator parameters
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
    parser.add_argument("--date", type=str, default=None, help="Target date YYYYMMDD, or comma-separated list (e.g. 20260508,20260509,20260510)")
    parser.add_argument("--code", type=str, default="", help="Collect only this code (6-digit). Comma-separated supported")
    parser.add_argument("--symbols-file", type=str, default=str(SCRIPT_DIR / "r009_universe_symbols_master.txt"), help="Path to r009_universe_symbols_master.txt")
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
    """1�??�이?��? 20�?간격?�로 ?�형 보간?�여 ?�??"""
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    
    if df.empty:
        return df
    
    # datetime을 인덱스로 지정
    df_indexed = df.set_index("datetime")
    
    # 20초 간격의 시간 생성
    time_range_20sec = pd.date_range(
        start=df_indexed.index.min(),
        end=df_indexed.index.max(),
        freq="20s"
    )
    
    # datetime 인덱스로 리인덱싱
    df_reindexed = df_indexed.reindex(time_range_20sec.union(df_indexed.index)).sort_index()
    
    # 가격 데이터 선형 보간
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df_reindexed[col] = df_reindexed[col].interpolate(method="linear", limit_direction="both")
    
    # 거래량은 이전 값으로 채움(0이 아닌 경우만)
    df_reindexed["volume"] = df_reindexed["volume"].ffill()
    df_reindexed["volume"] = df_reindexed["volume"].fillna(0)
    
    # Forward-fill market column if present.
    if "market" in df_reindexed.columns:
        df_reindexed["market"] = df_reindexed["market"].ffill()
    
    # 지표 컬럼은 이전 값으로 채움(1봉 이상이면 그대로 복사)
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

    bars_3m = calculate_r76_indicators(bars_3m)
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

    symbols_file = Path(args.symbols_file)
    if not symbols_file.is_file():
        raise SystemExit(f"symbols file not found: {symbols_file}")

    symbols = load_symbols(symbols_file)

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

        logger.info("collect start: date=%s, symbols=%d", target_date, len(symbols))

        saved_count = 0
        empty_count = 0
        nxt_flags: dict[str, bool] = {}

        for idx, (code, name) in enumerate(symbols, start=1):

            nxt_tradeable = False
            df = None
            last_error = None

            # 최대 2회 시도
            for attempt in range(2):
                try:
                    nxt_tradeable = probe_nxt_tradeable(code)
                    nxt_flags[code] = nxt_tradeable
                    df = fetch_symbol_data(code=code, target_date=target_date, include_nxt=nxt_tradeable)
                    last_error = None
                    break  # 성공하면 루프 탈출
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        logger.warning("[%d/%d] %s(%s) | fetch error (retry): %s", idx, len(symbols), code, name, exc)
                        time.sleep(0.5)  # 재시도 전 잠깐 대기

            # 2회 시도 모두 실패한 경우
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
                df_1m = enrich_with_strategy_indicators(df)
                df_20s = interpolate_to_20sec(df_1m)
                # 파일명을 종목코드(종목명).txt로 저장
                safe_name = str(name).replace("/", "_").replace("\\", "_")
                file_path = output_dir / f"{code}({safe_name}).txt"
                df_20s.to_csv(file_path, index=False, encoding="utf-8-sig", sep=",", float_format="%.2f")

                saved_count += 1
                logger.info(
                    "[%d/%d] %s(%s) | NXT=%s | saved 20s=%d rows -> %s",
                    idx,
                    len(symbols),
                    code,
                    name,
                    nxt_tradeable,
                    len(df_20s),
                    file_path,
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

        # Save NXT tradeable flags for live/sim scripts.
        if nxt_flags:
            nxt_flags_path = output_dir / "nxt_flags.json"
            with open(nxt_flags_path, "w", encoding="utf-8") as _f:
                json.dump(nxt_flags, _f, ensure_ascii=False, indent=2)
            logger.info("saved NXT flags (%d codes): %s", len(nxt_flags), nxt_flags_path)

        logger.info("done: date=%s saved=%d, empty=%d, out=%s", target_date, saved_count, empty_count, output_dir)


if __name__ == "__main__":
    main()


