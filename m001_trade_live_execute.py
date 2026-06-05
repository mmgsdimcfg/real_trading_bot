# -*- coding: utf-8 -*-

"""m001 live trading executor.

Strategy summary:
- Fetch intraday bars from server API and calculate indicators on the fly.
- Poll live price every 10 seconds.
- Entry uses multi-indicator score from live 1m/3m snapshots.
- Exit rules:
  1) stop-loss at -1.4%
  2) base take-profit at +2.0%
    3) profit-lock exit on momentum reversal (r006 style AUX_REVERSAL)
    4) breakeven-fail guard (peak gain giveback protection)
    5) profit giveback trailing from post-entry peak
    6) if strong surge mode is detected, hold for >= +5.0%
    7) in strong mode, sell when price drops by configured pct from post-entry 3m peak

Usage examples:
- python xgraph/auto_trading/m001_trade_live_execute.py --date 20260529
- python xgraph/auto_trading/m001_trade_live_execute.py --dry-run
- python xgraph/auto_trading/m001_trade_live_execute.py --single-pass
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "examples_llm"))
sys.path.insert(0, str(PROJECT_ROOT / "examples_user" / "domestic_stock"))

import kis_auth as ka
import domestic_stock_functions as dsf
from shared.indicators import calculate_indicators as shared_calculate_indicators


DEFAULT_CONFIG = {
    "POLL_SECONDS": 10,
    "DATA_FETCH_SECONDS": 20,
    "ACCOUNT_SYNC_SECONDS": 30,
    "STARTUP_WARMUP_SECONDS": 60,
    "LIVE_PRICE_STALE_SECONDS": 20,
    "COOLDOWN_SECONDS": 180,
    "MAX_SYMBOLS": 40,
    "ORDER_AMOUNT_KRW": 500_000,
    "ENV_DV": "real",
    "DRY_RUN": True,
    "ORDER_EXCHANGE": "KRX",
    "ORDER_TYPE_MARKET": "01",
    "SESSION_START": "09:00:00",
    "SESSION_END": "15:20:00",
    "ENTRY_MIN_SCORE": 5,
    "ENTRY_BB_BUFFER_PCT": 0.0005,
    "ENTRY_MIN_RSI": 52.0,
    "ENTRY_MAX_RSI": 74.0,
    "ENTRY_MIN_ADX": 18.0,
    "ENTRY_MIN_VOL_RATIO_1M": 1.20,
    "ENTRY_MIN_VOL_TREND_3M": 1.00,
    "STOP_LOSS_PCT": -0.014,
    "TAKE_PROFIT_PCT": 0.020,
    "AUX_SELL_MIN_PNL": 0.003,
    "AUX_SELL_MIN_SCORE": 2,
    "AUX_SELL_CONFIRM_SECONDS": 20,
    "BREAKEVEN_FAIL_ARM_PNL": 0.008,
    "BREAKEVEN_FAIL_GIVEBACK_PCT": 0.0075,
    "BREAKEVEN_FAIL_MAX_PNL": 0.003,
    "BREAKEVEN_FAIL_CONFIRM_SECONDS": 30,
    "PROFIT_GIVEBACK_ARM_PNL": 0.010,
    "PROFIT_GIVEBACK_TRAIL_PCT": 0.0075,
    "STRONG_MODE_MIN_PNL": 0.020,
    "STRONG_MODE_MIN_ADX": 24.0,
    "STRONG_MODE_MIN_RSI": 62.0,
    "STRONG_MODE_MIN_VOL_RATIO_1M": 1.45,
    "STRONG_MODE_MIN_GAIN_PCT": 0.050,
    "TRAIL_DROP_FROM_3M_PEAK_PCT": 0.10,
    "REENTRY_BLOCK_SECONDS": 300,
}


@dataclass
class Position:
    code: str
    name: str
    qty: int
    entry_price: float
    entry_time: str
    peak_price: float
    strong_mode: bool
    strong_mode_armed: bool


@dataclass
class RuntimeState:
    positions: dict[str, Position]
    last_trade_at: dict[str, str]
    cooldown_until: dict[str, str]


@dataclass
class RunLogFiles:
    decision_path: Path
    trade_path: Path


RUN_LOG_FILES: RunLogFiles | None = None


def _next_log_sequence(log_dir: Path, date_str: str, stem: str) -> int:
    pattern = re.compile(rf"^{re.escape(date_str)}_{re.escape(stem)}_(\d+)(?:_buy_sell)?\.txt$")
    max_seq = 0
    for path in log_dir.glob(f"{date_str}_{stem}_*.txt"):
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            max_seq = max(max_seq, int(match.group(1)))
        except Exception:
            continue
    return max_seq + 1


def init_run_logs(date_str: str, stem: str) -> RunLogFiles:
    log_dir = CURRENT_DIR / "mlog"
    log_dir.mkdir(parents=True, exist_ok=True)
    seq = _next_log_sequence(log_dir, date_str, stem)
    decision_path = log_dir / f"{date_str}_{stem}_{seq:03d}.txt"
    trade_path = log_dir / f"{date_str}_{stem}_{seq:03d}_buy_sell.txt"
    decision_path.touch(exist_ok=True)
    trade_path.touch(exist_ok=True)
    return RunLogFiles(decision_path=decision_path, trade_path=trade_path)


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(text + "\n")


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    if RUN_LOG_FILES is not None:
        _append_text(RUN_LOG_FILES.decision_path, line)


def log_trade(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    if RUN_LOG_FILES is not None:
        _append_text(RUN_LOG_FILES.decision_path, line)
        _append_text(RUN_LOG_FILES.trade_path, line)


def parse_time(text: str, fallback: dt_time) -> dt_time:
    try:
        return datetime.strptime(text.strip(), "%H:%M:%S").time()
    except Exception:
        return fallback


def parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"

    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except Exception:
            return text

    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", text):
        try:
            return float(text)
        except Exception:
            return text

    return text


def load_config(config_path: Path) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if not config_path.exists():
        log(f"WARN config file not found: {config_path}. defaults are used.")
        return cfg

    for line in config_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            continue
        cfg[key] = parse_scalar(value)

    return cfg


def parse_watchlist(watch_file: Path) -> dict[str, str]:
    if not watch_file.exists():
        raise FileNotFoundError(f"watchlist file not found: {watch_file}")

    out: dict[str, str] = {}
    for line in watch_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [p.strip() for p in text.split(",")]
        code = parts[0].zfill(6)
        name = parts[1] if len(parts) > 1 else ""
        if re.fullmatch(r"\d{6}", code):
            out[code] = name

    return out


def resolve_watch_file(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        return path if path.is_absolute() else (CURRENT_DIR / path)

    # Default source for same-day trade watchlist.
    return CURRENT_DIR / "m003_trade_watchlist_today.txt"


def _latest_date_dir(data_root: Path) -> Path | None:
    if not data_root.exists() or not data_root.is_dir():
        return None
    date_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    return date_dirs[-1] if date_dirs else None


def resolve_runtime_date_dir(date_str: str) -> tuple[Path, str]:
    resolved = date_str.strip() or datetime.now().strftime("%Y%m%d")
    date_dir = CURRENT_DIR / "data" / resolved
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir, resolved


def resolve_data_file(date_dir: Path, code: str, suffix: str) -> Path | None:
    # suffix examples: "", "_1m", "_3m"
    candidates = list(date_dir.glob(f"{code}*{suffix}.txt"))
    if not candidates:
        return None

    def key(path: Path) -> tuple[int, str]:
        stem = path.stem.lower()
        return (0 if suffix in stem else 1, stem)

    return sorted(candidates, key=key)[0]


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def read_frame_tail(path: Path | None, rows: int = 240) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log(f"WARN read failed: {path.name} ({exc})")
        return None

    if df.empty:
        return None

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")

    for col in (
        "open", "high", "low", "close", "volume", "MA_5", "VOL_MA20", "BB_MIDDLE", "BB_UPPER",
        "RSI", "RSI_SIGNAL", "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX", "OBV", "OBV_MA",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.tail(rows).copy()


def extract_live_price_row(row: dict[str, Any]) -> float | None:
    for key in ("stck_prpr", "prpr", "stck_clpr", "cur_prc"):
        value = to_float(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def fetch_live_price(code: str, env_dv: str) -> float | None:
    for market_div in ("J", "UN", "NX"):
        try:
            quote_df = dsf.inquire_price(
                env_dv=env_dv,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
            )
        except Exception:
            continue

        if quote_df is None or quote_df.empty:
            continue

        row = quote_df.iloc[-1].to_dict()
        price = extract_live_price_row(row)
        if price is not None:
            return price

    return None


def _normalize_intraday_output(output2: pd.DataFrame) -> pd.DataFrame:
    if output2 is None or output2.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = output2.copy()
    for col in ("stck_bsop_date", "stck_cntg_hour", "stck_oprc", "stck_hgpr", "stck_lwpr", "stck_prpr", "cntg_vol"):
        if col not in df.columns:
            df[col] = None

    date_txt = df["stck_bsop_date"].astype(str).str.replace(r"\D", "", regex=True)
    time_txt = df["stck_cntg_hour"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6).str[:6]
    ymd_default = datetime.now().strftime("%Y%m%d")
    date_txt = date_txt.where(date_txt.str.len() == 8, ymd_default)
    dt = pd.to_datetime(date_txt + time_txt, format="%Y%m%d%H%M%S", errors="coerce")

    out = pd.DataFrame(
        {
            "datetime": dt,
            "open": pd.to_numeric(df["stck_oprc"], errors="coerce"),
            "high": pd.to_numeric(df["stck_hgpr"], errors="coerce"),
            "low": pd.to_numeric(df["stck_lwpr"], errors="coerce"),
            "close": pd.to_numeric(df["stck_prpr"], errors="coerce"),
            "volume": pd.to_numeric(df["cntg_vol"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    out["volume"] = out["volume"].fillna(0.0)
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return out


def _aggregate_bars(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    temp = frame.copy().set_index("datetime")
    rule = f"{int(minutes)}min"
    agg = temp.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg


def _append_obv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=20, min_periods=1).mean()
    return out


def _build_live_frames(code: str, env_dv: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    now_hms = datetime.now().strftime("%H%M%S")
    for market_div in ("J", "UN", "NX"):
        try:
            _out1, out2 = dsf.inquire_time_itemchartprice(
                env_dv=env_dv,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
                fid_input_hour_1=now_hms,
                fid_pw_data_incu_yn="Y",
            )
        except Exception:
            continue

        base_1m = _normalize_intraday_output(out2)
        if base_1m.empty or len(base_1m) < 12:
            continue

        bars_1m = _aggregate_bars(base_1m, 1)
        bars_3m = _aggregate_bars(base_1m, 3)
        if len(bars_1m) < 12 or len(bars_3m) < 6:
            continue

        frame_1m = _append_obv_columns(shared_calculate_indicators(bars_1m.tail(240)))
        frame_3m = _append_obv_columns(shared_calculate_indicators(bars_3m.tail(240)))
        return frame_3m, frame_1m

    return None, None


def order_succeeded(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, pd.DataFrame):
        return not result.empty
    if isinstance(result, dict):
        rt_cd = str(result.get("rt_cd", ""))
        return rt_cd == "0"
    return False


def place_market_order(
    ord_dv: str,
    code: str,
    qty: int,
    env_dv: str,
    cano: str,
    acnt_prdt_cd: str,
    exchange: str,
    order_type_market: str,
    dry_run: bool,
) -> bool:
    if qty <= 0:
        return False

    if dry_run:
        log(f"DRY_RUN {ord_dv.upper()} | code={code} qty={qty}")
        return True

    try:
        result = dsf.order_cash(
            env_dv=env_dv,
            ord_dv=ord_dv,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
            pdno=code,
            ord_dvsn=order_type_market,
            ord_qty=str(qty),
            ord_unpr="0",
            excg_id_dvsn_cd=exchange,
        )
    except Exception as exc:
        log(f"ORDER ERROR | side={ord_dv} code={code} qty={qty} err={exc}")
        return False

    ok = order_succeeded(result)
    if not ok:
        log(f"ORDER FAIL | side={ord_dv} code={code} qty={qty}")
    return ok


def build_entry_score(live_price: float, row3: pd.Series, row1: pd.Series, cfg: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    bb_mid = to_float(row3.get("BB_MIDDLE"))
    ma5 = to_float(row3.get("MA_5"))
    rsi = to_float(row3.get("RSI"))
    rsi_sig = to_float(row3.get("RSI_SIGNAL"))
    stoch_k = to_float(row3.get("STOCH_K"))
    stoch_d = to_float(row3.get("STOCH_D"))
    macd_hist = to_float(row3.get("MACD_HIST"))
    adx = to_float(row3.get("ADX"))
    di_plus = to_float(row3.get("DI_PLUS"))
    di_minus = to_float(row3.get("DI_MINUS"))
    vol1 = to_float(row1.get("volume"))
    volma1 = to_float(row1.get("VOL_MA20"))
    vol3 = to_float(row3.get("volume"))
    volma3 = to_float(row3.get("VOL_MA20"))

    if bb_mid and live_price > bb_mid * (1.0 + float(cfg["ENTRY_BB_BUFFER_PCT"])):
        score += 1
        reasons.append("price_above_bb_mid")

    if ma5 and bb_mid and ma5 > bb_mid:
        score += 1
        reasons.append("ma5_above_bb_mid")

    if rsi and float(cfg["ENTRY_MIN_RSI"]) <= rsi <= float(cfg["ENTRY_MAX_RSI"]):
        score += 1
        reasons.append("rsi_in_range")

    if rsi and rsi_sig and rsi >= rsi_sig:
        score += 1
        reasons.append("rsi_above_signal")

    if stoch_k and stoch_d and stoch_k >= stoch_d and 20.0 <= stoch_k <= 90.0:
        score += 1
        reasons.append("stoch_support")

    if macd_hist is not None and macd_hist > 0:
        score += 1
        reasons.append("macd_hist_positive")

    if adx is not None and adx >= float(cfg["ENTRY_MIN_ADX"]):
        score += 1
        reasons.append("adx_trending")

    if di_plus is not None and di_minus is not None and di_plus >= di_minus:
        score += 1
        reasons.append("di_plus_dominant")

    if vol1 and volma1 and vol1 >= volma1 * float(cfg["ENTRY_MIN_VOL_RATIO_1M"]):
        score += 1
        reasons.append("volume_spike_1m")

    if vol3 and volma3 and vol3 >= volma3 * float(cfg["ENTRY_MIN_VOL_TREND_3M"]):
        score += 1
        reasons.append("volume_support_3m")

    return score, reasons


def build_aux_sell_score(
    live_price: float,
    row3: pd.Series,
    row3_prev: pd.Series,
    row1: pd.Series,
    row1_prev: pd.Series,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    bb_mid = to_float(row3.get("BB_MIDDLE"))
    if bb_mid and live_price <= bb_mid:
        score += 1
        reasons.append("live_below_bb_mid")

    rsi = to_float(row3.get("RSI"))
    rsi_sig = to_float(row3.get("RSI_SIGNAL"))
    rsi_prev = to_float(row3_prev.get("RSI"))
    rsi_sig_prev = to_float(row3_prev.get("RSI_SIGNAL"))
    if (
        rsi is not None and rsi_sig is not None and rsi_prev is not None and rsi_sig_prev is not None
        and rsi_prev >= rsi_sig_prev and rsi < rsi_sig
    ):
        score += 1
        reasons.append("rsi_dead_cross")

    stoch_k = to_float(row3.get("STOCH_K"))
    stoch_d = to_float(row3.get("STOCH_D"))
    stoch_k_prev = to_float(row3_prev.get("STOCH_K"))
    stoch_d_prev = to_float(row3_prev.get("STOCH_D"))
    if (
        stoch_k is not None and stoch_d is not None and stoch_k_prev is not None and stoch_d_prev is not None
        and stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d
    ):
        score += 1
        reasons.append("stoch_dead_cross")

    macd = to_float(row3.get("MACD"))
    macd_sig = to_float(row3.get("MACD_SIGNAL"))
    macd_prev = to_float(row3_prev.get("MACD"))
    macd_sig_prev = to_float(row3_prev.get("MACD_SIGNAL"))
    if (
        macd is not None and macd_sig is not None and macd_prev is not None and macd_sig_prev is not None
        and macd_prev >= macd_sig_prev and macd < macd_sig
    ):
        score += 1
        reasons.append("macd_dead_cross")

    obv = to_float(row3.get("OBV"))
    obv_ma = to_float(row3.get("OBV_MA"))
    obv_prev = to_float(row3_prev.get("OBV"))
    if obv is not None and obv_ma is not None and obv_prev is not None and obv < obv_ma and obv < obv_prev:
        score += 1
        reasons.append("obv_weak")

    close1 = to_float(row1.get("close"))
    close1_prev = to_float(row1_prev.get("close"))
    if close1 is not None and close1_prev is not None and close1 < close1_prev:
        score += 1
        reasons.append("price_1m_down")

    return score, reasons


def _elapsed_seconds_since(entry_time_iso: str, now: datetime) -> float:
    try:
        dt = datetime.fromisoformat(str(entry_time_iso))
    except Exception:
        return 0.0
    return max(0.0, (now - dt).total_seconds())


def update_timed_condition_state(
    state: dict[str, datetime],
    code: str,
    now: datetime,
    condition: bool,
) -> float:
    if condition:
        first_seen = state.get(code)
        if first_seen is None:
            state[code] = now
            return 0.0
        return max(0.0, (now - first_seen).total_seconds())

    state.pop(code, None)
    return 0.0


def is_strong_surge(live_price: float, pnl_pct: float, row3: pd.Series, row1: pd.Series, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    checks: list[str] = []
    strong_points = 0

    bb_upper = to_float(row3.get("BB_UPPER"))
    rsi = to_float(row3.get("RSI"))
    adx = to_float(row3.get("ADX"))
    macd_hist = to_float(row3.get("MACD_HIST"))
    di_plus = to_float(row3.get("DI_PLUS"))
    di_minus = to_float(row3.get("DI_MINUS"))
    vol1 = to_float(row1.get("volume"))
    volma1 = to_float(row1.get("VOL_MA20"))

    if pnl_pct >= float(cfg["STRONG_MODE_MIN_PNL"]):
        strong_points += 1
        checks.append("pnl_min")

    if bb_upper and live_price >= bb_upper * 0.999:
        strong_points += 1
        checks.append("price_near_bb_upper")

    if rsi and rsi >= float(cfg["STRONG_MODE_MIN_RSI"]):
        strong_points += 1
        checks.append("rsi_strong")

    if adx and adx >= float(cfg["STRONG_MODE_MIN_ADX"]):
        strong_points += 1
        checks.append("adx_strong")

    if macd_hist is not None and macd_hist > 0:
        strong_points += 1
        checks.append("macd_hist_pos")

    if di_plus and di_minus and di_plus > di_minus:
        strong_points += 1
        checks.append("di_plus_gt_di_minus")

    if vol1 and volma1 and vol1 >= volma1 * float(cfg["STRONG_MODE_MIN_VOL_RATIO_1M"]):
        strong_points += 1
        checks.append("volume_strong_1m")

    return strong_points >= 4, checks


def state_file_path(date_dir: Path) -> Path:
    return date_dir / "m001_runtime_state.json"


def load_state(date_dir: Path) -> RuntimeState:
    path = state_file_path(date_dir)
    if not path.exists():
        return RuntimeState(positions={}, last_trade_at={}, cooldown_until={})

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return RuntimeState(positions={}, last_trade_at={}, cooldown_until={})

    positions: dict[str, Position] = {}
    for code, raw in (payload.get("positions") or {}).items():
        try:
            positions[code] = Position(
                code=str(raw.get("code", code)).zfill(6),
                name=str(raw.get("name", "")),
                qty=int(raw.get("qty", 0)),
                entry_price=float(raw.get("entry_price", 0.0)),
                entry_time=str(raw.get("entry_time", datetime.now().isoformat())),
                peak_price=float(raw.get("peak_price", 0.0)),
                strong_mode=bool(raw.get("strong_mode", False)),
                strong_mode_armed=bool(raw.get("strong_mode_armed", False)),
            )
        except Exception:
            continue

    return RuntimeState(
        positions=positions,
        last_trade_at={str(k).zfill(6): str(v) for k, v in (payload.get("last_trade_at") or {}).items()},
        cooldown_until={str(k).zfill(6): str(v) for k, v in (payload.get("cooldown_until") or {}).items()},
    )


def save_state(date_dir: Path, state: RuntimeState) -> None:
    payload = {
        "positions": {code: asdict(pos) for code, pos in state.positions.items()},
        "last_trade_at": state.last_trade_at,
        "cooldown_until": state.cooldown_until,
        "saved_at": datetime.now().isoformat(),
    }
    path = state_file_path(date_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_first_value(row: pd.Series, candidates: list[str]) -> Any:
    for key in candidates:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _parse_positive_int(value: Any) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        return 0
    return parsed if parsed > 0 else 0


def sync_state_positions_from_account(
    state: RuntimeState,
    env_dv: str,
    cano: str,
    acnt_prdt_cd: str,
) -> list[str] | None:
    """Return removed position codes, or None if sync failed."""
    try:
        holdings_df, _ = dsf.inquire_balance(
            env_dv=env_dv,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
            afhr_flpr_yn="N",
            inqr_dvsn="02",
            unpr_dvsn="01",
            fund_sttl_icld_yn="N",
            fncg_amt_auto_rdpt_yn="N",
            prcs_dvsn="00",
        )
    except Exception as exc:
        log(f"WARN holdings sync failed: {exc}")
        return None

    account_positions: dict[str, dict[str, Any]] = {}
    if holdings_df is not None and not holdings_df.empty:
        for _, row in holdings_df.iterrows():
            code_raw = _pick_first_value(row, ["pdno", "mksc_shrn_iscd", "stck_shrn_iscd"])
            if code_raw is None:
                continue
            code = str(code_raw).strip().zfill(6)
            if not re.fullmatch(r"\d{6}", code):
                continue

            qty_raw = _pick_first_value(row, ["hldg_qty", "hold_qty", "bal_qty"])
            qty = _parse_positive_int(qty_raw)
            if qty <= 0:
                continue

            avg_price = to_float(_pick_first_value(row, ["pchs_avg_pric", "avg_prvs"]))
            account_positions[code] = {
                "qty": qty,
                "avg_price": float(avg_price or 0.0),
            }

    removed_codes: list[str] = []
    for code in list(state.positions.keys()):
        if code not in account_positions:
            removed_codes.append(code)
            state.positions.pop(code, None)
            state.cooldown_until.pop(code, None)

    for code, acc in account_positions.items():
        pos = state.positions.get(code)
        if pos is None:
            continue
        pos.qty = int(acc["qty"])
        if float(acc["avg_price"]) > 0:
            pos.entry_price = float(acc["avg_price"])
            pos.peak_price = max(float(pos.peak_price), float(acc["avg_price"]))

    return removed_codes


def in_cooldown(code: str, state: RuntimeState, now: datetime) -> bool:
    until = state.cooldown_until.get(code)
    if not until:
        return False
    try:
        return now < datetime.fromisoformat(until)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="m001 live trading executor")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="data folder date (YYYYMMDD)")
    parser.add_argument("--watch-file", type=str, default=None, help="watchlist file path")
    parser.add_argument("--config-file", type=str, default=str(CURRENT_DIR / "m002_config_value.txt"), help="config file path")
    parser.add_argument("--env", type=str, default=None, help="override env_dv (real/demo)")
    parser.add_argument("--dry-run", action="store_true", help="force dry run")
    parser.add_argument("--single-pass", action="store_true", help="run one loop and exit")
    args = parser.parse_args()

    date_str = args.date.strip()
    date_dir, resolved_date = resolve_runtime_date_dir(date_str)
    date_str = resolved_date

    global RUN_LOG_FILES
    RUN_LOG_FILES = init_run_logs(date_str, Path(__file__).stem)
    log(f"run log file: {RUN_LOG_FILES.decision_path}")
    log(f"trade log file: {RUN_LOG_FILES.trade_path}")

    cfg = load_config(Path(args.config_file))
    env_dv = str(args.env or cfg["ENV_DV"]).strip()
    dry_run = bool(cfg.get("DRY_RUN", True) or args.dry_run)

    poll_seconds = max(1, int(cfg["POLL_SECONDS"]))
    data_fetch_seconds = max(1, int(cfg.get("DATA_FETCH_SECONDS", 20)))
    account_sync_seconds = max(5, int(cfg.get("ACCOUNT_SYNC_SECONDS", 30)))
    max_symbols = max(1, int(cfg["MAX_SYMBOLS"]))
    order_amount = max(50_000, int(cfg["ORDER_AMOUNT_KRW"]))

    session_start = parse_time(str(cfg["SESSION_START"]), dt_time(9, 0, 0))
    session_end = parse_time(str(cfg["SESSION_END"]), dt_time(15, 20, 0))

    watch_file = resolve_watch_file(args.watch_file)
    watch_map = parse_watchlist(watch_file)
    watch_items = list(watch_map.items())[:max_symbols]
    if not watch_items:
        raise SystemExit(f"watchlist is empty: {watch_file}")

    log(
        f"config: env={env_dv} dry_run={dry_run} poll={poll_seconds}s "
        f"data_fetch={data_fetch_seconds}s max_symbols={len(watch_items)}"
    )
    log(f"watch file: {watch_file}")
    print("WATCHLIST (code,name)")
    for code, name in watch_items:
        print(f"{code},{name}")

    warmup_seconds = max(0, int(cfg.get("STARTUP_WARMUP_SECONDS", 60)))
    startup_warmup_until: datetime | None = None
    now0 = datetime.now()
    if session_start <= now0.time() <= session_end and warmup_seconds > 0:
        startup_warmup_until = datetime.fromtimestamp(now0.timestamp() + warmup_seconds)
        log(f"startup in-session warmup enabled: collect-only for {warmup_seconds}s")

    ka.auth()
    trenv = ka.getTREnv()
    cano = trenv.my_acct
    acnt_prdt_cd = trenv.my_prod

    state = load_state(date_dir)
    aux_sell_state: dict[str, datetime] = {}
    breakeven_fail_state: dict[str, datetime] = {}
    live_frame_cache: dict[str, dict[str, Any]] = {}
    last_account_sync_at: datetime | None = None

    if not dry_run:
        removed = sync_state_positions_from_account(state, env_dv=env_dv, cano=cano, acnt_prdt_cd=acnt_prdt_cd)
        if removed is not None:
            if removed:
                log(f"state/account reconcile removed stale positions: {','.join(sorted(removed))}")
            save_state(date_dir, state)
        last_account_sync_at = datetime.now()

    while True:
        now = datetime.now()

        if not dry_run and (
            last_account_sync_at is None
            or (now - last_account_sync_at).total_seconds() >= account_sync_seconds
        ):
            removed = sync_state_positions_from_account(state, env_dv=env_dv, cano=cano, acnt_prdt_cd=acnt_prdt_cd)
            if removed is not None:
                if removed:
                    log(f"state/account reconcile removed stale positions: {','.join(sorted(removed))}")
                    for code in removed:
                        aux_sell_state.pop(code, None)
                        breakeven_fail_state.pop(code, None)
                save_state(date_dir, state)
            last_account_sync_at = now

        if now.time() < session_start or now.time() > session_end:
            if now.time() > session_end:
                log("session end reached. stop loop.")
                break
            time.sleep(poll_seconds)
            continue

        is_warmup_phase = startup_warmup_until is not None and now < startup_warmup_until
        warmup_ready_count = 0
        warmup_total_count = len(watch_items)

        for code, name in watch_items:
            cached = live_frame_cache.get(code)
            use_cache = False
            if cached is not None:
                fetched_at = cached.get("fetched_at")
                if isinstance(fetched_at, datetime):
                    elapsed = (now - fetched_at).total_seconds()
                    if elapsed < data_fetch_seconds:
                        use_cache = True

            if use_cache:
                frame_3m = cached.get("frame_3m")
                frame_1m = cached.get("frame_1m")
            else:
                frame_3m, frame_1m = _build_live_frames(code, env_dv=env_dv)
                if frame_3m is not None and frame_1m is not None and (not frame_3m.empty) and (not frame_1m.empty):
                    live_frame_cache[code] = {
                        "fetched_at": now,
                        "frame_3m": frame_3m,
                        "frame_1m": frame_1m,
                    }
                elif cached is not None:
                    frame_3m = cached.get("frame_3m")
                    frame_1m = cached.get("frame_1m")
                else:
                    log(f"SKIP {code} {name} | live 1m/3m frame unavailable")
                    continue

            if frame_3m is None or frame_1m is None or frame_3m.empty or frame_1m.empty:
                log(f"SKIP {code} {name} | live 1m/3m frame unavailable")
                continue

            row3 = frame_3m.iloc[-1]
            row1 = frame_1m.iloc[-1]

            live_price = fetch_live_price(code, env_dv=env_dv)
            if live_price is None or live_price <= 0:
                log(f"SKIP {code} {name} | live price unavailable")
                continue

            if is_warmup_phase:
                warmup_ready_count += 1
                continue

            pos = state.positions.get(code)
            if pos is None:
                aux_sell_state.pop(code, None)
                breakeven_fail_state.pop(code, None)
                if in_cooldown(code, state, now):
                    log(f"HOLD {code} {name} | reason=COOLDOWN")
                    continue

                score, reasons = build_entry_score(live_price, row3, row1, cfg)
                log(
                    f"CHECK {code} {name} | side=BUY | score={score}/{int(cfg['ENTRY_MIN_SCORE'])} "
                    f"price={live_price:,.0f} reasons={','.join(reasons) if reasons else '-'}"
                )
                if score < int(cfg["ENTRY_MIN_SCORE"]):
                    log(f"REJECT {code} {name} | side=BUY | reason=LOW_SCORE")
                    continue

                qty = int(order_amount / max(1.0, live_price))
                if qty <= 0:
                    continue

                if place_market_order(
                    ord_dv="buy",
                    code=code,
                    qty=qty,
                    env_dv=env_dv,
                    cano=cano,
                    acnt_prdt_cd=acnt_prdt_cd,
                    exchange=str(cfg["ORDER_EXCHANGE"]),
                    order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                    dry_run=dry_run,
                ):
                    state.positions[code] = Position(
                        code=code,
                        name=name,
                        qty=qty,
                        entry_price=float(live_price),
                        entry_time=now.isoformat(),
                        peak_price=float(max(live_price, to_float(row3.get("high"), live_price) or live_price)),
                        strong_mode=False,
                        strong_mode_armed=False,
                    )
                    state.last_trade_at[code] = now.isoformat()
                    state.cooldown_until[code] = datetime.fromtimestamp(
                        now.timestamp() + int(cfg["COOLDOWN_SECONDS"])
                    ).isoformat()
                    save_state(date_dir, state)
                    log_trade(
                        f"BUY {code} {name} | qty={qty} price={live_price:,.0f} "
                        f"score={score} reasons={','.join(reasons)}"
                    )
                continue

            # manage open position
            pos.peak_price = max(
                float(pos.peak_price),
                float(live_price),
                float(to_float(row3.get("high"), live_price) or live_price),
            )
            pnl_pct = (live_price / max(1e-9, pos.entry_price)) - 1.0
            peak_pnl_pct = (pos.peak_price / max(1e-9, pos.entry_price)) - 1.0
            profit_giveback = peak_pnl_pct - pnl_pct
            held_seconds = _elapsed_seconds_since(pos.entry_time, now)

            row3_prev = frame_3m.iloc[-2] if len(frame_3m) >= 2 else row3
            row1_prev = frame_1m.iloc[-2] if len(frame_1m) >= 2 else row1
            log(
                f"CHECK {code} {name} | side=SELL | pnl={pnl_pct*100:.2f}% "
                f"entry={pos.entry_price:,.0f} live={live_price:,.0f} peak={pos.peak_price:,.0f} "
                f"strong={pos.strong_mode} armed={pos.strong_mode_armed}"
            )

            # r006-style: profitable reversal exit by multi-signal sell score.
            aux_score, aux_reasons = build_aux_sell_score(live_price, row3, row3_prev, row1, row1_prev)
            aux_hold = update_timed_condition_state(
                aux_sell_state,
                code,
                now,
                pnl_pct >= float(cfg["AUX_SELL_MIN_PNL"]) and aux_score >= int(cfg["AUX_SELL_MIN_SCORE"]),
            )
            if aux_hold >= float(cfg["AUX_SELL_CONFIRM_SECONDS"]):
                reason = f"AUX_REVERSAL_SCORE_{aux_score}"
                if place_market_order(
                    ord_dv="sell",
                    code=code,
                    qty=pos.qty,
                    env_dv=env_dv,
                    cano=cano,
                    acnt_prdt_cd=acnt_prdt_cd,
                    exchange=str(cfg["ORDER_EXCHANGE"]),
                    order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                    dry_run=dry_run,
                ):
                    state.positions.pop(code, None)
                    state.last_trade_at[code] = now.isoformat()
                    state.cooldown_until[code] = datetime.fromtimestamp(
                        now.timestamp() + int(cfg["COOLDOWN_SECONDS"])
                    ).isoformat()
                    aux_sell_state.pop(code, None)
                    breakeven_fail_state.pop(code, None)
                    save_state(date_dir, state)
                    log_trade(
                        f"SELL {code} {name} | {reason} pnl={pnl_pct*100:.2f}% "
                        f"hold={aux_hold:.0f}s reasons={','.join(aux_reasons) if aux_reasons else '-'}"
                    )
                continue

            # r006-style: if peak profit is given back for a while, protect capital.
            breakeven_hold = update_timed_condition_state(
                breakeven_fail_state,
                code,
                now,
                (
                    peak_pnl_pct >= float(cfg["BREAKEVEN_FAIL_ARM_PNL"])
                    and profit_giveback >= float(cfg["BREAKEVEN_FAIL_GIVEBACK_PCT"])
                    and pnl_pct <= float(cfg["BREAKEVEN_FAIL_MAX_PNL"])
                ),
            )
            if breakeven_hold >= float(cfg["BREAKEVEN_FAIL_CONFIRM_SECONDS"]):
                reason = (
                    f"BREAKEVEN_FAIL_peak{float(cfg['BREAKEVEN_FAIL_ARM_PNL'])*100:.1f}_"
                    f"giveback{float(cfg['BREAKEVEN_FAIL_GIVEBACK_PCT'])*100:.2f}_{int(float(cfg['BREAKEVEN_FAIL_CONFIRM_SECONDS']))}s"
                )
                if place_market_order(
                    ord_dv="sell",
                    code=code,
                    qty=pos.qty,
                    env_dv=env_dv,
                    cano=cano,
                    acnt_prdt_cd=acnt_prdt_cd,
                    exchange=str(cfg["ORDER_EXCHANGE"]),
                    order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                    dry_run=dry_run,
                ):
                    state.positions.pop(code, None)
                    state.last_trade_at[code] = now.isoformat()
                    state.cooldown_until[code] = datetime.fromtimestamp(
                        now.timestamp() + int(cfg["COOLDOWN_SECONDS"])
                    ).isoformat()
                    aux_sell_state.pop(code, None)
                    breakeven_fail_state.pop(code, None)
                    save_state(date_dir, state)
                    log_trade(
                        f"SELL {code} {name} | {reason} pnl={pnl_pct*100:.2f}% "
                        f"peak={peak_pnl_pct*100:.2f}% held={held_seconds:.0f}s"
                    )
                continue

            # Peak-profit giveback trailing (works even when strong_mode is not active).
            if (
                peak_pnl_pct >= float(cfg["PROFIT_GIVEBACK_ARM_PNL"])
                and pnl_pct > 0
                and profit_giveback >= float(cfg["PROFIT_GIVEBACK_TRAIL_PCT"])
            ):
                reason = f"PROFIT_GIVEBACK_TRAIL_{float(cfg['PROFIT_GIVEBACK_TRAIL_PCT'])*100:.2f}%"
                if place_market_order(
                    ord_dv="sell",
                    code=code,
                    qty=pos.qty,
                    env_dv=env_dv,
                    cano=cano,
                    acnt_prdt_cd=acnt_prdt_cd,
                    exchange=str(cfg["ORDER_EXCHANGE"]),
                    order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                    dry_run=dry_run,
                ):
                    state.positions.pop(code, None)
                    state.last_trade_at[code] = now.isoformat()
                    state.cooldown_until[code] = datetime.fromtimestamp(
                        now.timestamp() + int(cfg["COOLDOWN_SECONDS"])
                    ).isoformat()
                    aux_sell_state.pop(code, None)
                    breakeven_fail_state.pop(code, None)
                    save_state(date_dir, state)
                    log_trade(
                        f"SELL {code} {name} | {reason} pnl={pnl_pct*100:.2f}% "
                        f"peak={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                    )
                continue

            # stop-loss: hard risk cap
            if pnl_pct <= float(cfg["STOP_LOSS_PCT"]):
                if place_market_order(
                    ord_dv="sell",
                    code=code,
                    qty=pos.qty,
                    env_dv=env_dv,
                    cano=cano,
                    acnt_prdt_cd=acnt_prdt_cd,
                    exchange=str(cfg["ORDER_EXCHANGE"]),
                    order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                    dry_run=dry_run,
                ):
                    state.positions.pop(code, None)
                    state.last_trade_at[code] = now.isoformat()
                    state.cooldown_until[code] = datetime.fromtimestamp(
                        now.timestamp() + int(cfg["REENTRY_BLOCK_SECONDS"])
                    ).isoformat()
                    aux_sell_state.pop(code, None)
                    breakeven_fail_state.pop(code, None)
                    save_state(date_dir, state)
                    log_trade(f"SELL {code} {name} | STOP_LOSS pnl={pnl_pct*100:.2f}%")
                continue

            # base take-profit gate
            if not pos.strong_mode and pnl_pct >= float(cfg["TAKE_PROFIT_PCT"]):
                strong, checks = is_strong_surge(live_price, pnl_pct, row3, row1, cfg)
                if strong:
                    pos.strong_mode = True
                    save_state(date_dir, state)
                    log(f"HOLD {code} {name} | STRONG_MODE ON checks={','.join(checks)}")
                else:
                    if place_market_order(
                        ord_dv="sell",
                        code=code,
                        qty=pos.qty,
                        env_dv=env_dv,
                        cano=cano,
                        acnt_prdt_cd=acnt_prdt_cd,
                        exchange=str(cfg["ORDER_EXCHANGE"]),
                        order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                        dry_run=dry_run,
                    ):
                        state.positions.pop(code, None)
                        state.last_trade_at[code] = now.isoformat()
                        state.cooldown_until[code] = datetime.fromtimestamp(
                            now.timestamp() + int(cfg["COOLDOWN_SECONDS"])
                        ).isoformat()
                        aux_sell_state.pop(code, None)
                        breakeven_fail_state.pop(code, None)
                        save_state(date_dir, state)
                        log_trade(f"SELL {code} {name} | TAKE_PROFIT_2pct pnl={pnl_pct*100:.2f}%")
                    continue

            # strong-mode trailing logic
            if pos.strong_mode:
                if pnl_pct >= float(cfg["STRONG_MODE_MIN_GAIN_PCT"]):
                    pos.strong_mode_armed = True

                if pos.strong_mode_armed:
                    peak_drop = 1.0 - (live_price / max(1e-9, pos.peak_price))
                    if peak_drop >= float(cfg["TRAIL_DROP_FROM_3M_PEAK_PCT"]) and pnl_pct > 0:
                        if place_market_order(
                            ord_dv="sell",
                            code=code,
                            qty=pos.qty,
                            env_dv=env_dv,
                            cano=cano,
                            acnt_prdt_cd=acnt_prdt_cd,
                            exchange=str(cfg["ORDER_EXCHANGE"]),
                            order_type_market=str(cfg["ORDER_TYPE_MARKET"]),
                            dry_run=dry_run,
                        ):
                            state.positions.pop(code, None)
                            state.last_trade_at[code] = now.isoformat()
                            state.cooldown_until[code] = datetime.fromtimestamp(
                                now.timestamp() + int(cfg["COOLDOWN_SECONDS"])
                            ).isoformat()
                            aux_sell_state.pop(code, None)
                            breakeven_fail_state.pop(code, None)
                            save_state(date_dir, state)
                            log_trade(
                                f"SELL {code} {name} | STRONG_TRAIL peak_drop={peak_drop*100:.2f}% "
                                f"pnl={pnl_pct*100:.2f}%"
                            )
                        continue

            save_state(date_dir, state)

        if is_warmup_phase:
            remaining = int((startup_warmup_until - now).total_seconds()) if startup_warmup_until else 0
            log(
                f"WARMUP collecting only | ready={warmup_ready_count}/{warmup_total_count} "
                f"remaining={max(0, remaining)}s"
            )
        elif startup_warmup_until is not None:
            log("WARMUP completed. trading enabled.")
            startup_warmup_until = None

        if args.single_pass:
            break

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
