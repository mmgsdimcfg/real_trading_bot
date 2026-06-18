# -*- coding: utf-8 -*-

"""R76 live trading executor - BB middle cross strategy with multi indicators.

Core idea:
1) Buy when live price crosses above BB middle and stays there long enough.
2) Sell when live price crosses below BB middle and stays there long enough.
3) Use Stochastic Fast, RSI, Williams %R as confirmation filters.
4) Use take-profit, stop-loss, and trailing-stop for risk control.

Run examples:
- python xgraph/auto_trading/r006_trade_live_execute.py
- python xgraph/auto_trading/r006_trade_live_execute.py --date 20260508
- python xgraph/auto_trading/r006_trade_live_execute.py --fake

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-06-18] type=feat owner=copilot
    summary: 손절(HARD_STOP_LOSS_0.8PCT, ATR_STOP_LOSS) 시 시장가(ord_dvsn=01) 즉시 매도; place_sell_order에 market_order 파라미터 추가. NXT 세션은 지정가 유지.
    impact: live
    compatibility: backward-compatible
- [2026-06-17] type=feat owner=copilot
    summary: (r005 연동) BB 중간선 최근 4봉(12분) 연속 우하향 시 매수 차단 - BB_MID_DOWNTREND_4BARS 거부 사유 추가.
    impact: live
    compatibility: backward-compatible
- [2026-06-17] type=fix owner=copilot
    summary: (r005 연동) UPTREND_CONT 진입 경로 추가 - 크로스 없이도 ADX30+/+DI우세/BB위3봉이상이면 매수; 크로스 룩백 5봉 확장.
    impact: live
    compatibility: backward-compatible
- [2026-06-07] type=feat owner=copilot
    summary: strengthened buy gates with ADX rising+DI dominance, MFI overheat guard, RSI 50-break/50-60 zone, and OBV signal-cross confirmation.
    impact: live
    compatibility: backward-compatible
- [2026-06-06] type=feat owner=copilot
    summary: added ADX+MFI entry gate and ATR-based variable TP/SL risk model.
    impact: live
    compatibility: backward-compatible
- [2026-06-05] type=fix owner=copilot
    summary: (한글) BB 중간값 진입 보조조건 변경에 맞춰 리젝트 사유 로그를 PREV_CLOSE_MISSING / LIVE_NOT_ABOVE_PREV_CLOSE_AND_BB_MIDDLE로 상세화.
    impact: live
    compatibility: backward-compatible
- [2026-05-24] type=feat owner=copilot
    summary: exclude pre-held watchlist positions; trade only same-day buys with data/YYYYMMDD/today_buys.txt persistence.
    impact: live
    compatibility: backward-compatible
- [2026-05-22] type=fix owner=copilot
    summary: traded_today reserve-before-submit; cooldown keys zfill(6); persist traded_today on buy submit; discard on buy fail.
    impact: live
    compatibility: backward-compatible
- [2026-05-22] type=fix owner=copilot
    summary: buy_inflight exposure guard + stable trade_events append log (flush trade_logger, main-loop BUY EXECUTED log_trade).
    impact: live
    compatibility: backward-compatible
- [2026-05-22] type=fix owner=copilot
    summary: live-trading safety fixes (fail-closed market day, strict orders, live state, dry-run, session force close).
    impact: live
    compatibility: backward-compatible
- [2026-05-10] type=docs owner=copilot
    summary: added standardized file header and expandable update-log format.
    impact: live
    compatibility: backward-compatible

Note: This script cannot guarantee profit. Always paper-test before live trading.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import json
import os
import signal
import inspect
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional

import pandas as pd
from r003_define_config import (
    ACCOUNT_SYNC_INTERVAL_SECONDS,
    ADX_BUY_MIN,
    ADX_MIN_TREND,
    ADX_PERIOD,
    ADX_STRONG_TREND,
    AFTERNOON_NXT_END,
    AFTERNOON_NXT_FORCE_EXIT,
    AFTERNOON_NXT_NEW_ENTRY_CUTOFF,
    AFTERNOON_NXT_START,
    ALLOW_REBUY_SAME_CODE,
    AUX_SELL_MIN_PNL_SCORE2,
    AUX_SELL_MIN_PNL_SCORE3,
    AUX_SELL_MIN_PNL_SCORE4,
    BB_PERIOD,
    BB_SQUEEZE_MIN_WIDTH_PCT,
    BB_STD_MULTIPLIER,
    BB_UPPER_PROXIMITY_MAX,
    BOX_RANGE_HOLD_LOOKBACK_BARS,
    BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT,
    BOX_RANGE_HOLD_MAX_RANGE_PCT,
    DATA_DIR_NAME,
    DEFINE_TODAY_CODE_PATH,
    EARLY_NEAR_CROSS_ALLOWED_END,
    EARLY_NEAR_CROSS_ALLOWED_START,
    EARLY_NEAR_CROSS_ALLOW_NXT,
    EARLY_NEAR_CROSS_MIN_TURNOVER_KRW,
    EARLY_NEAR_CROSS_MIN_VOL_MA,
    EARLY_NEAR_CROSS_MIN_VOLUME,
    ENABLE_BOX_RANGE_HOLD_TECH_SELL,
    ENABLE_EARLY_NEAR_CROSS_ENTRY,
    ENABLE_NEAR_CROSS_ARM,
    ENABLE_PRICE_LEAD_BB_BREAKOUT,
    ENABLE_STRICT_MA5_BB_GOLDEN_CROSS,
    ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS,
    ENABLE_NXT_SESSION,
    ENABLE_TP_EXTENSION_TRAILING,
    LIVE_PRICE_BB_BUFFER_PCT,
    LIVE_PRICE_CROSS_CONFIRM_POLLS,
    LIVE_PRICE_CROSS_CONFIRM_SECONDS,
    LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS,
    LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS,
    MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE,
    MA5_BB_DOWN_CROSS_IMMEDIATE_PNL,
    MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE,
    MA5_BB_DOWN_CROSS_MIN_PNL,
    MA5_BB_FOLLOW_CHASE_MAX_GAP_PCT,
    MA_PERIOD,
    MACD_FAST,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW,
    MAX_ORDER_AMOUNT_KRW,
    MIN_BARS_REQUIRED,
    MFI_BUY_MIN,
    MFI_PERIOD,
    MORNING_NXT_END,
    MORNING_NXT_START,
    NEAR_CROSS_ARM_EXPIRE_BARS,
    NEAR_CROSS_ARM_GAP_MAX,
    NEAR_CROSS_ARM_MA_RISE_MIN,
    NEAR_CROSS_EARLY_GAP_MAX,
    NEAR_CROSS_EARLY_MA_RISE_MIN,
    OBV_BREAKOUT_LOOKBACK_BARS,
    BREAKEVEN_FAIL_ARM_PNL,
    BREAKEVEN_FAIL_CONFIRM_SECONDS,
    BREAKEVEN_FAIL_GIVEBACK_PCT,
    NO_TREND_EXIT_ARM_SECONDS,
    NO_TREND_EXIT_CONFIRM_SECONDS,
    NO_TREND_EXIT_MAX_PEAK_PNL,
    NO_TREND_EXIT_MIN_PNL,
    ATR_PERIOD,
    ATR_STOP_MULTIPLIER,
    ATR_TAKE_PROFIT_MULTIPLIER,
    OBV_MA_PERIOD,
    POLL_INTERVAL_SECONDS,
    PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT,
    PRICE_LEAD_BREAKOUT_MIN_ADX,
    PRICE_LEAD_BREAKOUT_MIN_SCORE,
    POST_BUY_DROP_CONFIRM_SECONDS,
    REGULAR_END,
    REGULAR_FORCE_EXIT,
    REGULAR_NEW_ENTRY_CUTOFF,
    REGULAR_START,
    RSI_BUY_MAX,
    RSI_BUY_MIN,
    RSI_PERIOD,
    RSI_SIGNAL_PERIOD,
    STARTUP_WARMUP_SECONDS,
    POST_BUY_BB_DROP_ARMED_SECONDS,
    POST_BUY_BB_DROP_PCT,
    POST_BUY_BB_DROP_POLLS,
    STOP_LOSS_EARLY_PERCENT,
    STOP_LOSS_MIN_HOLD_SECONDS,
    STOP_LOSS_PERCENT,
    MAX_BUY_RISE_PCT_FROM_PREV_CLOSE,
    STOCH_BUY_MAX,
    STOCH_BUY_MIN,
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    STOCH_OVERBOUGHT,
    STRONG_TREND_OVERBOUGHT_MIN_ADX,
    STRONG_TREND_OVERBOUGHT_MIN_SCORE,
    STRONG_TREND_OVERBOUGHT_MIN_VOL_RATIO,
    TAKE_PROFIT_PERCENT,
    TP_EXTENSION_TRAIL_FROM_PEAK,
    TRADE_COOLDOWN_MINUTES,
    TRAILING_STOP_FROM_PEAK,
    VOLUME_MA_PERIOD,
    VOLUME_RATIO_CLOSE,
    VOLUME_RATIO_FLOOR,
    VOLUME_RATIO_MIDDAY,
    VOLUME_RATIO_NXT,
    VOLUME_RATIO_OPEN,
    VOLUME_RATIO_STRONG_RELAX,
    WILLIAMS_BUY_FLOOR,
    WILLIAMS_D_PERIOD,
    WILLIAMS_OVERBOUGHT_CEIL,
    WILLIAMS_R_PERIOD,
    AUX_SELL_MIN_REALIZED_TARGET_PCT,
    AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT,
    BB_BUY_SCORE_THRESHOLD,
    BUY_CONSECUTIVE_CONFIRM_COUNT,
    ENABLE_INTRABAR_LIVE_ENTRY_FILTER,
    ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP,
    FRAME_BACKFILL_SYNC_SECONDS,
    FRAME_POLL_INTERVAL_SECONDS,
    INTRABAR_ADX_MIN,
    INTRABAR_MFI_MAX,
    INTRABAR_MFI_MIN,
    INTRABAR_MIN_ELAPSED_SECONDS,
    INTRABAR_RSI_MAX,
    INTRABAR_RSI_MIN,
    LIVE_PRICE_BACKOFF_BASE_SECONDS,
    LIVE_PRICE_BACKOFF_MAX_SECONDS,
    LIVE_PRICE_POLL_INTERVAL_SECONDS,
    LIVE_PRICE_STALE_TTL_SECONDS,
    LIVE_STATE_SAVE_INTERVAL_SECONDS,
    MAIN_LOOP_MAX_CONSECUTIVE_ERRORS,
    MARKET_DAY_FAIL_CLOSED,
    MFI_OVERBOUGHT_MAX,
    MORNING_NXT_NEW_ENTRY_CUTOFF,
    ORDER_STATUS_POLL_INTERVAL_SECONDS,
    PENDING_BUY_GRACE_SECONDS,
    PENDING_STATUS_BACKOFF_MAX_SECONDS,
    REQUIRE_ADX_RISING,
    REQUIRE_DI_PLUS_DOMINANT,
    REQUIRE_OBV_SIGNAL_CROSS,
    RSI_BUY_MOMENTUM_MAX,
    SESSION_FORCE_CLOSE_ALL_AT_CUTOFF,
    WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS,
)
from r005_strategy_core_shared import (
    R76StrategyConfig,
    calculate_indicators,
    check_buy_condition as shared_check_buy_condition,
    check_sell_condition as shared_check_sell_condition,
    update_timed_condition_state,
    update_live_price_cross_state as shared_update_live_price_cross_state,
    _compute_bb_slope_pct,
    _near_cross_momentum_flags,
    _passes_early_near_cross_liquidity,
)
from r010_watchlist_bridge import resolve_watchlist_path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.insert(0, str(project_root / "examples_llm"))
sys.path.insert(0, str(project_root / "examples_user" / "domestic_stock"))
sys.path.insert(0, str(project_root / "examples_llm" / "domestic_stock" / "inquire_time_itemchartprice"))
sys.path.insert(0, str(current_dir))

import kis_auth as ka
import domestic_stock_functions as dsf

try:
    from inquire_time_itemchartprice import inquire_time_itemchartprice
except Exception:
    inquire_time_itemchartprice = None

TODAY_CODE_FILE = current_dir / DEFINE_TODAY_CODE_PATH
DATA_DIR = current_dir / DATA_DIR_NAME
TODAY_BUYS_FILENAME = "today_buys.txt"

LIVE_RUNTIME_DIR = DATA_DIR / "live_runtime"
LIVE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
POSITION_META_PATH = LIVE_RUNTIME_DIR / "position_meta.json"
# Legacy alias (daily traded_today + optional combined state)
LIVE_STATE_DIR = LIVE_RUNTIME_DIR

KIS_ENV_DV = (os.environ.get("KIS_ENV_DV", "real") or "real").strip()
_LIVE_DRY_RUN_RAW = (os.environ.get("LIVE_DRY_RUN", "") or "").strip().lower()
LIVE_DRY_RUN = _LIVE_DRY_RUN_RAW in ("1", "true", "yes", "on", "y")

ENABLE_LIVE_DRY_RUN = LIVE_DRY_RUN

SHARED_R76_CONFIG = R76StrategyConfig(
    live_price_bb_buffer_pct=LIVE_PRICE_BB_BUFFER_PCT,
    live_price_cross_confirm_polls=LIVE_PRICE_CROSS_CONFIRM_POLLS,
    live_price_cross_confirm_seconds=LIVE_PRICE_CROSS_CONFIRM_SECONDS,
    live_price_down_cross_confirm_polls=LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS,
    live_price_down_cross_confirm_seconds=LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS,
    require_strict_buy_golden_cross=ENABLE_STRICT_MA5_BB_GOLDEN_CROSS,
    stoch_overbought=STOCH_OVERBOUGHT,
    williams_overbought_ceil=WILLIAMS_OVERBOUGHT_CEIL,
    bb_upper_proximity_max=BB_UPPER_PROXIMITY_MAX,
    bb_squeeze_min_width_pct=BB_SQUEEZE_MIN_WIDTH_PCT,
    adx_min_trend=ADX_MIN_TREND,
    stop_loss_percent=STOP_LOSS_PERCENT,
    take_profit_percent=TAKE_PROFIT_PERCENT,
    enable_box_range_hold_tech_sell=ENABLE_BOX_RANGE_HOLD_TECH_SELL,
    box_range_hold_lookback_bars=BOX_RANGE_HOLD_LOOKBACK_BARS,
    box_range_hold_max_range_pct=BOX_RANGE_HOLD_MAX_RANGE_PCT,
    box_range_hold_max_bb_width_pct=BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT,
    ma5_bb_down_cross_min_pnl=MA5_BB_DOWN_CROSS_MIN_PNL,
    ma5_bb_down_cross_immediate_pnl=MA5_BB_DOWN_CROSS_IMMEDIATE_PNL,
    ma5_bb_down_cross_immediate_score=MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE,
    aux_sell_min_pnl_score2=AUX_SELL_MIN_PNL_SCORE2,
    aux_sell_min_pnl_score3=AUX_SELL_MIN_PNL_SCORE3,
    aux_sell_min_pnl_score4=AUX_SELL_MIN_PNL_SCORE4,
    stoch_buy_min=STOCH_BUY_MIN,
    stoch_buy_max=STOCH_BUY_MAX,
    rsi_buy_min=RSI_BUY_MIN,
    rsi_buy_max=RSI_BUY_MAX,
    williams_buy_floor=WILLIAMS_BUY_FLOOR,
    obv_breakout_lookback_bars=OBV_BREAKOUT_LOOKBACK_BARS,
    enable_price_lead_bb_breakout=ENABLE_PRICE_LEAD_BB_BREAKOUT,
    price_lead_breakout_min_score=PRICE_LEAD_BREAKOUT_MIN_SCORE,
    price_lead_breakout_min_adx=PRICE_LEAD_BREAKOUT_MIN_ADX,
    price_lead_breakout_allow_overbought=PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT,
    enable_strong_trend_overbought_bypass=ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS,
    strong_trend_overbought_min_score=STRONG_TREND_OVERBOUGHT_MIN_SCORE,
    strong_trend_overbought_min_vol_ratio=STRONG_TREND_OVERBOUGHT_MIN_VOL_RATIO,
    strong_trend_overbought_min_adx=STRONG_TREND_OVERBOUGHT_MIN_ADX,
    ma5_bb_follow_chase_max_gap_pct=MA5_BB_FOLLOW_CHASE_MAX_GAP_PCT,
    bb_buy_score_threshold=BB_BUY_SCORE_THRESHOLD,
)

INDICATOR_WARMUP_BARS = 1  # backtest uses MIN_BARS_REQUIRED=3; live needs only 1 closed bar (indicators use min_periods=1)

# Market-day status cache (holiday API call reduction).
_MARKET_DAY_STATUS_CACHE: dict[str, tuple[bool, str]] = {}

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
class _SuppressLibLogs(logging.Filter):
    _SUPPRESS = frozenset(["Data fetch complete.", "Call Next page...", "Max recursive depth reached."])
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in self._SUPPRESS

_suppress_filter = _SuppressLibLogs()

_LOG_CTX: dict[str, object] = {"date_str": datetime.now().strftime("%Y%m%d")}

import threading

_TRADE_LOG_WRITE_LOCK = threading.Lock()

_SYMBOL_CODE_PATTERN = re.compile(r"\b(\d{6})\b")
_INVALID_FILENAME_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
_SYMBOL_NAME_MAP: dict[str, str] = {}


def _sanitize_log_filename(name: str) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(name).strip())
    cleaned = cleaned.rstrip(" .")
    return cleaned or "UNKNOWN"


def register_symbol_names(symbol_name_map: dict[str, str]) -> None:
    _SYMBOL_NAME_MAP.clear()
    for code, name in symbol_name_map.items():
        normalized_code = str(code).zfill(6)
        normalized_name = str(name).strip() if name else ""
        _SYMBOL_NAME_MAP[normalized_code] = normalized_name


class _PerSymbolFileHandler(logging.Handler):
    """Write log lines to per-symbol files based on 6-digit code in message."""

    def __init__(self, base_dir: Path, buy_sell: bool = False):
        super().__init__(level=logging.INFO)
        self.base_dir = base_dir
        self.buy_sell = buy_sell
        self._streams: dict[Path, object] = {}

    def _resolve_path(self, code: str) -> Path:
        symbol_name = _SYMBOL_NAME_MAP.get(code, "")
        date_str = str(_LOG_CTX.get("date_str") or datetime.now().strftime("%Y%m%d"))
        label = f"{date_str}_{code}_{symbol_name}" if symbol_name else f"{date_str}_{code}"
        stem = _sanitize_log_filename(label)
        suffix = "_buy_sell.txt" if self.buy_sell else ".txt"
        return self.base_dir / f"{stem}{suffix}"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            match = _SYMBOL_CODE_PATTERN.search(msg)
            if not match:
                return
            code = match.group(1)
            if code not in _SYMBOL_NAME_MAP:
                return
            path = self._resolve_path(code)
            stream = self._streams.get(path)
            if stream is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = open(path, "a", encoding="utf-8")
                self._streams[path] = stream
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception:
                pass
        self._streams.clear()
        super().close()

def _rotate_logging_for_date(date_str: str) -> None:
    _LOG_CTX["date_str"] = date_str
    log_dir = current_dir / "logs"
    log_date_dir = log_dir / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    log_date_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_stem = Path(__file__).stem
    flat_log_filename = log_dir / f"{timestamp}_{script_stem}.log"
    flat_trade_log_filename = log_dir / f"{timestamp}_{script_stem}_buy_sell.log"
    log_filename = log_date_dir / f"{timestamp}_{script_stem}.log"
    trade_log_filename = log_date_dir / f"{timestamp}_{script_stem}_buy_sell.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(flat_log_filename, encoding="utf-8"),
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    global logger, trade_logger, _trade_handler, _symbol_general_handler, _symbol_trade_handler
    logger = logging.getLogger(__name__)
    logging.getLogger("domestic_stock_functions").setLevel(logging.WARNING)
    logging.getLogger("inquire_time_itemchartprice").setLevel(logging.WARNING)
    for _handler in logging.getLogger().handlers:
        if not any(isinstance(f, _SuppressLibLogs) for f in getattr(_handler, "filters", [])):
            _handler.addFilter(_suppress_filter)

    trade_logger = logging.getLogger("trade_events")
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False
    for h in list(trade_logger.handlers):
        trade_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    _trade_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _trade_handler = logging.FileHandler(flat_trade_log_filename, encoding="utf-8")
    _trade_handler.setFormatter(_trade_formatter)
    trade_logger.addHandler(_trade_handler)
    _trade_date_handler = logging.FileHandler(trade_log_filename, encoding="utf-8")
    _trade_date_handler.setFormatter(_trade_formatter)
    trade_logger.addHandler(_trade_date_handler)

    for h in list(logger.handlers):
        if isinstance(h, _PerSymbolFileHandler):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    _symbol_general_handler = _PerSymbolFileHandler(log_date_dir, buy_sell=False)
    _symbol_general_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_symbol_general_handler)

    for h in list(trade_logger.handlers):
        if isinstance(h, _PerSymbolFileHandler):
            trade_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    _symbol_trade_handler = _PerSymbolFileHandler(log_date_dir, buy_sell=True)
    _symbol_trade_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    trade_logger.addHandler(_symbol_trade_handler)

    for handler in trade_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    _LOG_CTX["flat_trade_log"] = flat_trade_log_filename
    _LOG_CTX["trade_log"] = trade_log_filename

_rotate_logging_for_date(str(_LOG_CTX["date_str"]))

def log(msg: str) -> None:
    logger.info(msg)





def _trade_log_target_paths() -> list[Path]:
    date_str = str(_LOG_CTX.get("date_str") or datetime.now().strftime("%Y%m%d"))
    candidates: list[Path] = [
        current_dir / "logs" / f"buy_sell_{date_str}.log",
        current_dir / "logs" / f"trade_events_{date_str}.log",
    ]
    for key in ("flat_trade_log", "trade_log", "session_buy_sell_log"):
        value = _LOG_CTX.get(key)
        if value:
            candidates.append(Path(value))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _bind_session_trade_log() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = current_dir / "logs" / f"{timestamp}_r006_trade_live_execute_buy_sell.log"
    _LOG_CTX["session_buy_sell_log"] = session_path
    log_trade(f"SESSION trade log | path={session_path}")


def _trade_logger_file_paths() -> set[str]:
    paths: set[str] = set()
    for handler in getattr(trade_logger, "handlers", []):
        file_name = getattr(handler, "baseFilename", None)
        if not file_name:
            continue
        try:
            paths.add(str(Path(file_name).resolve()))
        except Exception:
            paths.add(str(file_name))
    return paths


def log_trade(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [INFO] {msg}\n"
    logger_managed_paths = _trade_logger_file_paths()
    with _TRADE_LOG_WRITE_LOCK:
        for target_path in _trade_log_target_paths():
            try:
                target_key = str(target_path.resolve())
                if target_key in logger_managed_paths:
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "a", encoding="utf-8") as trade_f:
                    trade_f.write(line)
                    trade_f.flush()
                    os.fsync(trade_f.fileno())
            except Exception as exc:
                logger.warning(f"trade log append failed ({target_path}): {exc}")
    trade_logger.info(msg)
    for handler in trade_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    log(f"[TRADE] {msg}")


def _log_trade_block(lines: list[str], event_time: datetime | None = None, mirror_main_log: bool = False) -> None:
    ts = event_time or datetime.now()
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    payload = [f"{stamp} [INFO] {line}\n" for line in lines]

    with _TRADE_LOG_WRITE_LOCK:
        for target_path in _trade_log_target_paths():
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "a", encoding="utf-8") as trade_f:
                    trade_f.writelines(payload)
                    trade_f.flush()
                    os.fsync(trade_f.fileno())
            except Exception as exc:
                logger.warning(f"trade log append failed ({target_path}): {exc}")

    if mirror_main_log:
        for line in lines:
            log(line)


# ---------------------------------------------------------------------------
# 파일 유틸
# ---------------------------------------------------------------------------

def _load_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=encoding) as file_obj:
                return [line.strip() for line in file_obj if line.strip()]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("define_today_code", b"", 0, 1, "Unable to decode")


def load_today_codes(code_file: Path | None = None) -> dict[str, str]:
    source_file = code_file or TODAY_CODE_FILE

    if not source_file.exists():
        return {}

    result: dict[str, str] = {}
    for line in _load_text_lines(source_file):
        if line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        # Backward compatible formats:
        # - code,name
        # - YYYYMMDD,code,name
        if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 8:
            code = parts[1].zfill(6)
            name = parts[2] if parts[2] else code
        else:
            code = parts[0].zfill(6)
            name = parts[1] if len(parts) >= 2 and parts[1] else code
        result[code] = name

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R76 real trading runner")
    parser.add_argument("--date", type=str, help="Watchlist date (YYYYMMDD). Use data/YYYYMMDD/picks.txt first")
    parser.add_argument(
        "--watchlist-source",
        type=str,
        default="auto",
        choices=["auto", "r008", "scan-picks", "picks"],
        help="Watchlist resolution: auto, r008, scan-picks, or legacy picks.txt",
    )
    parser.add_argument("--dry-run", "--fake", dest="dry_run", action="store_true", help="Log orders without sending to broker")
    parser.add_argument("--env-dv", type=str, default=None, help="KIS env_dv override (default: env KIS_ENV_DV or real)")
    return parser.parse_args()


def _resolve_watchlist_file(target_date: str | None, watchlist_source: str = "auto") -> Path:
    return resolve_watchlist_path(current_dir, target_date, watchlist_source, DATA_DIR)


def _today_buys_file_path(date_str: str) -> Path:
    return DATA_DIR / date_str / TODAY_BUYS_FILENAME


def load_today_buy_codes(date_str: str) -> set[str]:
    path = _today_buys_file_path(date_str)
    if not path.exists():
        return set()

    codes: set[str] = set()
    try:
        for line in _load_text_lines(path):
            if line.startswith("#"):
                continue
            code = line.split(",", 1)[0].strip()
            if code:
                codes.add(str(code).zfill(6))
    except Exception as exc:
        log(f"WARNING: today-buy file load failed ({path}): {exc}")
        return set()
    return codes


def save_today_buy_codes(date_str: str, codes: set[str]) -> None:
    path = _today_buys_file_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(sorted({str(code).zfill(6) for code in (codes or set())}))
    if payload:
        payload += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _is_today_buy_position(code: str, pos: dict, date_str: str, today_buy_codes: set[str]) -> bool:
    norm = str(code).zfill(6)
    if norm in today_buy_codes:
        return True

    buy_time = pos.get("buy_time")
    if isinstance(buy_time, datetime):
        return buy_time.strftime("%Y%m%d") == date_str

    if isinstance(buy_time, str) and buy_time.strip():
        try:
            return datetime.fromisoformat(buy_time.strip()).strftime("%Y%m%d") == date_str
        except ValueError:
            return False

    return False


# ---------------------------------------------------------------------------
# 주문 결과 파싱
# ---------------------------------------------------------------------------

def _order_succeeded(result) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        rt_cd = result.get("rt_cd")
        if rt_cd is not None:
            return str(rt_cd).strip() == "0"
        # Some wrappers return output-only dict without rt_cd on success.
        return bool(result)
    try:
        if hasattr(result, "empty") and bool(result.empty):
            return False
        if hasattr(result, "columns"):
            lowered = {str(col).lower() for col in result.columns}
            if "rt_cd" in lowered:
                col_name = next(col for col in result.columns if str(col).lower() == "rt_cd")
                return str(result.iloc[0][col_name]).strip() == "0"
            # domestic_stock_functions.order_cash() returns a non-empty DataFrame
            # containing output fields (for example odno/ord_tmd) when successful,
            # and an empty DataFrame when failed.
            return len(result) > 0
    except Exception:
        pass
    return False


def _extract_order_value(result, candidates: tuple[str, ...]):
    if result is None:
        return None

    def _pick(mapping):
        if not isinstance(mapping, dict):
            return None

        lowered = {str(key).lower(): value for key, value in mapping.items()}
        for key in candidates:
            value = lowered.get(str(key).lower())
            if value not in (None, ""):
                return value

        for nested in ("output", "output1", "output2", "data"):
            inner = mapping.get(nested)
            if isinstance(inner, dict):
                picked = _pick(inner)
                if picked is not None:
                    return picked
            elif isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        picked = _pick(item)
                        if picked is not None:
                            return picked
        return None

    if isinstance(result, dict):
        return _pick(result)

    try:
        if hasattr(result, "to_dict"):
            records = result.to_dict(orient="records")
            if records:
                return _pick(records[0])
    except Exception:
        pass

    return None


def _extract_order_price(result):
    value = _extract_order_value(
        result,
        ("avg_pric", "avg_price", "avg_prvs", "stck_prpr", "ord_unpr", "fill_pric", "ccld_pric", "prpr"),
    )
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _extract_order_number(result) -> str:
    value = _extract_order_value(result, ("odno", "ord_no", "order_no"))
    text = str(value).strip() if value is not None else ""
    return text
    
def _extract_aux_score_from_reason(reason: str) -> int | None:
    m = re.search(r"AUX_REVERSAL_SCORE_(\d+)", str(reason or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _aux_min_pnl_for_score(score: int) -> float | None:
    if score >= 4:
        return float(AUX_SELL_MIN_PNL_SCORE4)
    if score == 3:
        return float(AUX_SELL_MIN_PNL_SCORE3)
    if score == 2:
        return float(AUX_SELL_MIN_PNL_SCORE2)
    return None


def _extract_order_time(result) -> str:
    value = _extract_order_value(result, ("ord_tmd", "ord_time", "order_time"))
    text = str(value).strip() if value is not None else ""
    return text


def _format_pending_pnl(reason_prefix: str, pnl_pct: float) -> str:
    return f"{reason_prefix}_{pnl_pct * 100:.2f}%"


def _session_exit_plan(reason_prefix: str, pnl_pct: float) -> tuple[str, str]:
    if SESSION_FORCE_CLOSE_ALL_AT_CUTOFF:
        return "sell", _format_pending_pnl(f"{reason_prefix}_SESSION_FORCE_CLOSE_ALL", pnl_pct)
    if pnl_pct > 0:
        return "sell", _format_pending_pnl(f"{reason_prefix}_PROFIT_CLOSE", pnl_pct)
    if ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP and pnl_pct >= STOP_LOSS_PERCENT:
        return "hold", _format_pending_pnl(f"{reason_prefix}_OVERNIGHT_HOLD_WITHIN_STOP", pnl_pct)
    return "sell", _format_pending_pnl(f"{reason_prefix}_STOP_LOSS_BREACH", pnl_pct)


def _extract_order_error_detail(result) -> str:
    """Best-effort extraction of broker error code/message from order response."""
    if result is None:
        return "NO_RESULT"

    key_candidates = ("rt_cd", "msg_cd", "msg1", "msg", "error_code", "error_message")

    def _pull(mapping) -> dict[str, str]:
        found: dict[str, str] = {}
        if not isinstance(mapping, dict):
            return found

        for key in key_candidates:
            value = mapping.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                found[key] = text

        for nested in ("output", "output1", "output2", "data"):
            inner = mapping.get(nested)
            if isinstance(inner, dict):
                found.update({k: v for k, v in _pull(inner).items() if k not in found})
            elif isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        found.update({k: v for k, v in _pull(item).items() if k not in found})

        return found

    extracted: dict[str, str] = {}
    if isinstance(result, dict):
        extracted = _pull(result)
    else:
        try:
            if hasattr(result, "to_dict"):
                records = result.to_dict(orient="records")
                if records:
                    extracted = _pull(records[0])
        except Exception:
            extracted = {}

    if not extracted:
        return "UNKNOWN_ORDER_ERROR"

    parts = []
    for key in ("rt_cd", "msg_cd", "msg1", "msg", "error_code", "error_message"):
        value = extracted.get(key)
        if value:
            parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else "UNKNOWN_ORDER_ERROR"


def _format_code_label(code: str, code_name: str = "") -> str:
    return f"{code}({code_name})" if code_name else code


def _format_trade_time_label(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "N/A"
        # Broker order time may arrive as HHMMSS.
        if text.isdigit() and len(text) == 6:
            return f"{text[0:2]}:{text[2:4]}:{text[4:6]}"
        try:
            return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return text
    return "N/A"


def _log_trade_event_banner(event: str, code: str, qty: int, price: float, detail: str = "", code_name: str = "") -> None:
    """Prints a high-visibility trade event block to both console and file logs."""
    if str(event).strip().upper() == "BUY SUBMITTED":
        return
    line = "=" * 110
    title = f"*** {event} | {_format_code_label(code, code_name)} | qty={qty} | price={price:,.0f} ***"
    log(line)
    log(title)
    log_trade(line)
    log_trade(title)
    if detail:
        log(f"*** DETAIL: {detail}")
        log_trade(f"*** DETAIL: {detail}")
    log(line)
    log_trade(line)


# ---------------------------------------------------------------------------
# NXT 거래 가능 종목 판단
# ---------------------------------------------------------------------------

def _is_truthy_flag(value) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if text in {"Y", "1", "TRUE", "T", "O", "YES"}:
        return True
    if text in {"N", "0", "FALSE", "F", "X", "NO"}:
        return False
    return None


NXT_TRADABLE_CACHE: dict[str, bool] = {}
_NXT_PROBE_FAILED_LOGGED = False


def _probe_nxt_tradeable_via_stock_info(code: str) -> bool | None:
    stock_info_fn = getattr(dsf, "search_stock_info", None)
    if not callable(stock_info_fn):
        return None

    try:
        result = stock_info_fn(prdt_type_cd="300", pdno=code)
    except Exception as exc:
        log(f"WARNING: search_stock_info failed for {code}: {exc}")
        return None

    if result is None or getattr(result, "empty", True):
        return None

    row = result.iloc[-1]
    for key in ("cptt_trad_tr_psbl_yn", "nxt_tr_stop_yn", "tr_stop_yn"):
        if key not in result.columns:
            continue
        flag = _is_truthy_flag(row.get(key))
        if flag is None:
            continue
        return flag if key == "cptt_trad_tr_psbl_yn" else not flag

    return None


def is_nxt_tradeable(code: str) -> bool:
    global _NXT_PROBE_FAILED_LOGGED

    if code in NXT_TRADABLE_CACHE:
        return NXT_TRADABLE_CACHE[code]

    tradeable = _probe_nxt_tradeable_via_stock_info(code)
    if tradeable is not None:
        NXT_TRADABLE_CACHE[code] = tradeable
        return tradeable

    if not _NXT_PROBE_FAILED_LOGGED:
        log("WARNING: NXT probe failed; defaulting to False for unknown codes")
        _NXT_PROBE_FAILED_LOGGED = True

    NXT_TRADABLE_CACHE[code] = False
    return False


# ---------------------------------------------------------------------------
# 세션 헬퍼
# ---------------------------------------------------------------------------

def is_weekday_market_day(now: datetime) -> bool:
    return now.weekday() < 5


def _market_day__market_day_fail_closed_result(date_label: str, reason: str) -> tuple[bool, str]:
    return (
        False,
        f"MARKET DAY CHECK | {date_label} | CLOSED | source=fail_closed | reason={reason}",
    )


def get_market_day_status(now: datetime) -> tuple[bool, str]:
    cache_key = now.strftime("%Y%m%d")
    cached = _MARKET_DAY_STATUS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    date_label = now.strftime("%Y-%m-%d")
    if not is_weekday_market_day(now):
        result = (False, f"MARKET DAY CHECK | {date_label} | CLOSED | source=weekday | reason=weekend")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    holiday_fn = getattr(dsf, "chk_holiday", None)
    if not callable(holiday_fn):
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, "chk_holiday_unavailable")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday | reason=chk_holiday_unavailable")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    try:
        df = holiday_fn(bass_dt=now.strftime("%Y%m%d"))
    except Exception as exc:
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, f"chk_holiday_failed:{exc}")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=chk_holiday_failed:{exc}")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    if df is None or df.empty:
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, "empty_holiday_response")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=empty_holiday_response")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    date_str = now.strftime("%Y%m%d")
    if "bass_dt" in df.columns:
        row = df[df["bass_dt"].astype(str) == date_str]
        if row.empty:
            row = df.iloc[[0]]
    else:
        row = df.iloc[[0]]

    flag = _is_truthy_flag(row.iloc[-1].get("opnd_yn"))
    if flag is None:
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, "unknown_opnd_yn")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=unknown_opnd_yn")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    status_text = "OPEN" if flag else "CLOSED"
    bass_dt = str(row.iloc[-1].get("bass_dt", date_str)).strip() or date_str
    opnd_yn = str(row.iloc[-1].get("opnd_yn", "")).strip() or "UNKNOWN"
    result = (flag, f"MARKET DAY CHECK | {date_label} | {status_text} | source=chk_holiday | bass_dt={bass_dt} | opnd_yn={opnd_yn}")
    _MARKET_DAY_STATUS_CACHE[cache_key] = result
    return result


def is_open_trading_day(now: datetime) -> bool:
    is_open, _ = get_market_day_status(now)
    return is_open


def is_regular_session(now: datetime) -> bool:
    return REGULAR_START <= now.time() <= REGULAR_END


def is_regular_call_auction(now: datetime) -> bool:
    current_time = now.time()
    return REGULAR_NEW_ENTRY_CUTOFF <= current_time < REGULAR_END


def is_nxt_session(now: datetime) -> bool:
    if not ENABLE_NXT_SESSION:
        return False
    current_time = now.time()
    return (MORNING_NXT_START <= current_time <= MORNING_NXT_END) or (AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END)


def classify_buy_session(now: datetime) -> str:
    if not ENABLE_NXT_SESSION:
        return "regular"
    current_time = now.time()
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return "morning_nxt"
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return "afternoon_nxt"
    return "regular"


def can_trade_code_now(now: datetime, nxt_tradeable: bool) -> bool:
    current_time = now.time()
    if REGULAR_START <= current_time <= REGULAR_END:
        return True
    if not ENABLE_NXT_SESSION:
        return False
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return nxt_tradeable
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return nxt_tradeable
    return False


def is_new_entry_allowed(now: datetime, nxt_tradeable: bool) -> bool:
    if is_regular_session(now):
        return now.time() < REGULAR_NEW_ENTRY_CUTOFF
    if not ENABLE_NXT_SESSION or not nxt_tradeable or not is_nxt_session(now):
        return False
    current_time = now.time()
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return current_time < MORNING_NXT_NEW_ENTRY_CUTOFF
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return current_time < AFTERNOON_NXT_NEW_ENTRY_CUTOFF
    return False


def _fetch_bid_ask_price(code: str, market_div: str) -> tuple[float | None, float | None]:
    """매수1호가(bid)와 매도1호가(ask) 조회. 실패 시 (None, None) 반환."""
    try:
        df1, _ = dsf.inquire_asking_price_exp_ccn(
            env_dv=KIS_ENV_DV,
            fid_cond_mrkt_div_code=market_div,
            fid_input_iscd=str(code).zfill(6),
        )
        if df1.empty:
            return None, None
        row = df1.iloc[0]
        bid = float(row.get("bidp1") or 0) or None
        ask = float(row.get("askp1") or 0) or None
        return bid, ask
    except Exception as exc:
        log(f"WARNING: 호가 조회 실패 {code}: {exc}")
        return None, None


def get_order_spec(now: datetime, nxt_tradeable: bool) -> dict | None:
    if is_regular_session(now):
        return {"exchange": "KRX", "ord_dvsn": "01", "ord_unpr": "0"}
    if not ENABLE_NXT_SESSION:
        return None
    if is_nxt_session(now) and nxt_tradeable:
        return {"exchange": "NXT", "ord_dvsn": "00", "ord_unpr": None}
    return None


def get_session_open_datetime(now: datetime, nxt_tradeable: bool) -> datetime | None:
    current_time = now.time()
    if REGULAR_START <= current_time <= REGULAR_END:
        return datetime.combine(now.date(), REGULAR_START)
    if not ENABLE_NXT_SESSION:
        return None
    if nxt_tradeable and MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return datetime.combine(now.date(), MORNING_NXT_START)
    if nxt_tradeable and AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return datetime.combine(now.date(), AFTERNOON_NXT_START)
    return None


def get_volume_ratio_threshold(now: datetime, adx_val: float) -> float:
    current_time = now.time()

    if is_nxt_session(now):
        ratio = VOLUME_RATIO_NXT
    elif current_time < dt_time(10, 0):
        ratio = VOLUME_RATIO_OPEN
    elif current_time < dt_time(14, 30):
        ratio = VOLUME_RATIO_MIDDAY
    else:
        ratio = VOLUME_RATIO_CLOSE

    # 추세가 매우 강하면 거래량 필터 완화
    if not pd.isna(adx_val) and adx_val >= ADX_STRONG_TREND:
        ratio = max(VOLUME_RATIO_FLOOR, ratio - VOLUME_RATIO_STRONG_RELAX)

    return ratio


def is_early_near_cross_allowed(now: datetime, nxt_tradeable: bool) -> bool:
    current_time = now.time()
    if is_regular_session(now):
        return EARLY_NEAR_CROSS_ALLOWED_START <= current_time <= EARLY_NEAR_CROSS_ALLOWED_END
    if EARLY_NEAR_CROSS_ALLOW_NXT and is_nxt_session(now) and nxt_tradeable:
        return True
    return False


# ---------------------------------------------------------------------------
# 이익 실현
# ---------------------------------------------------------------------------

def _is_allowed_intraday_time(ts: pd.Timestamp, nxt_tradeable: bool) -> bool:
    t = ts.time()
    if REGULAR_START <= t <= REGULAR_END:
        return True
    if not ENABLE_NXT_SESSION:
        return False
    if nxt_tradeable and (
        (MORNING_NXT_START <= t <= MORNING_NXT_END)
        or (AFTERNOON_NXT_START <= t <= AFTERNOON_NXT_END)
    ):
        return True
    return False


def _normalize_intraday_frame(df: pd.DataFrame, target_date: str, nxt_tradeable: bool) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    out = df.copy()
    rename_map = {
        "stck_cntg_hour": "time",
        "stck_oprc": "open",
        "stck_hgpr": "high",
        "stck_lwpr": "low",
        "stck_prpr": "close",
        "cntg_vol": "volume",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    if not {"time", "open", "high", "low", "close", "volume"}.issubset(set(out.columns)):
        return None

    out["time"] = out["time"].astype(str).str.zfill(6)
    out["datetime"] = pd.to_datetime(target_date + out["time"], format="%Y%m%d%H%M%S", errors="coerce")
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    out = out[~out.index.duplicated(keep="last")]

    allowed_mask = pd.Series([_is_allowed_intraday_time(ts, nxt_tradeable) for ts in out.index], index=out.index)
    out = out[allowed_mask]
    if out.empty:
        return None

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[(out[["open", "high", "low", "close"]].max(axis=1) > 0)].copy()
    if out.empty:
        return None

    out = out.resample("3min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])

    return out if not out.empty else None


def fetch_3min_frame(code: str, now: datetime, nxt_tradeable: bool) -> pd.DataFrame | None:
    if inquire_time_itemchartprice is None:
        return None

    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]
    today_str = now.strftime("%Y%m%d")

    for market_div in candidates:
        try:
            _, raw_df = inquire_time_itemchartprice(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
                fid_input_hour_1=now.strftime("%H%M%S"),
                fid_pw_data_incu_yn="Y",
                fid_etc_cls_code="",
            )
        except Exception as exc:
            log(f"WARNING: chart fetch failed for {code} ({market_div}): {exc}")
            continue

        frame = _normalize_intraday_frame(raw_df, today_str, nxt_tradeable)
        if frame is not None and not frame.empty:
            # fetch 시점 기준으로 확정 완료된 3분봉까지만 사용
            last_closed_bar = pd.Timestamp(now).floor("3min")
            frame = frame[frame.index <= last_closed_bar]
            if frame.empty:
                continue
            return calculate_indicators(frame)

    return None


def fetch_live_price(code: str, now: datetime, nxt_tradeable: bool) -> float | None:
    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]

    for market_div in candidates:
        try:
            quote_df = dsf.inquire_price(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
            )
        except Exception as exc:
            log(f"WARNING: live price fetch failed for {code} ({market_div}): {exc}")
            continue

        if quote_df is None or quote_df.empty:
            continue

        row = quote_df.iloc[-1]
        for key in ("stck_prpr", "prpr"):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return None


def fetch_prev_close(code: str, now: datetime, nxt_tradeable: bool) -> float | None:
    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]

    for market_div in candidates:
        try:
            quote_df = dsf.inquire_price(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
            )
        except Exception:
            continue

        if quote_df is None or quote_df.empty:
            continue

        row = quote_df.iloc[-1]
        for key in ("stck_sdpr", "bfdy_clpr", "prdy_clpr", "stck_prdy_clpr"):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return None


def should_refresh_3min_frame(
    now: datetime,
    cached_frame: pd.DataFrame | None,
    last_refresh_at: datetime | None,
) -> bool:
    if cached_frame is None or cached_frame.empty:
        return True

    if last_refresh_at is None:
        return True

    # Poll frame data every 20 seconds even if no new 3-minute bar closed yet.
    # This keeps server-side data updates in sync with live checks.
    elapsed_seconds = (now - last_refresh_at).total_seconds()
    if elapsed_seconds >= FRAME_POLL_INTERVAL_SECONDS:
        return True

    # Extra safety backfill in case refresh timestamps drift unexpectedly.
    if elapsed_seconds >= FRAME_BACKFILL_SYNC_SECONDS:
        return True

    return False


def _live_price_backoff_seconds(fail_count: int) -> int:
    # Exponential backoff: 5s, 10s, 20s, 40s, 60s cap.
    seconds = LIVE_PRICE_BACKOFF_BASE_SECONDS * (2 ** max(0, fail_count - 1))
    return int(min(LIVE_PRICE_BACKOFF_MAX_SECONDS, seconds))


def _pending_status_backoff_seconds(fail_count: int) -> int:
    # Exponential backoff with wider cap for repeated status-query failures.
    seconds = ORDER_STATUS_POLL_INTERVAL_SECONDS * (2 ** max(0, fail_count - 1))
    return int(min(PENDING_STATUS_BACKOFF_MAX_SECONDS, seconds))


def update_live_price_cross_state(
    cross_state: dict[str, dict],
    code: str,
    now: datetime,
    live_price: float,
    bb_middle: float,
) -> dict[str, object]:
    return shared_update_live_price_cross_state(
        cross_state=cross_state,
        code=code,
        now=pd.Timestamp(now),
        live_price=live_price,
        bb_middle=bb_middle,
        config=SHARED_R76_CONFIG,
    )
def _build_realtime_entry_frame(
    frame: pd.DataFrame,
    code: str,
    now: datetime,
    live_price: float,
    realtime_bar_state: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, float]:
    now_ts = pd.Timestamp(now)
    bar_start = now_ts.floor("3min")
    bar_end = bar_start + pd.Timedelta(minutes=3)
    elapsed_seconds = max(0.0, (now_ts - bar_start).total_seconds())

    if frame is None or frame.empty or pd.isna(live_price) or live_price <= 0:
        return frame, elapsed_seconds

    state = realtime_bar_state.get(code)
    if state is None or state.get("bar_end") != bar_end:
        state = {
            "bar_end": bar_end,
            "open": float(live_price),
            "high": float(live_price),
            "low": float(live_price),
            "close": float(live_price),
        }
    else:
        state["high"] = max(float(state.get("high", live_price)), float(live_price))
        state["low"] = min(float(state.get("low", live_price)), float(live_price))
        state["close"] = float(live_price)

    realtime_bar_state[code] = state

    working = frame.copy()
    # 현재 미완성봉 거래량: API가 10초 누적을 신뢰성 있게 제공하지 않으므로
    # 직전 확정봉의 거래량을 대리값으로 사용해 volume=0 으로 인한 리젝을 방지한다.
    prev_bar_volume = float(frame.iloc[-1]["volume"]) if not frame.empty else 0.0
    realtime_row = {
        "open": float(state["open"]),
        "high": float(state["high"]),
        "low": float(state["low"]),
        "close": float(state["close"]),
        "volume": prev_bar_volume,
    }
    if len(working) > 0 and working.index[-1] == bar_end:
        for key, val in realtime_row.items():
            working.at[bar_end, key] = val
    else:
        working.loc[bar_end, ["open", "high", "low", "close", "volume"]] = [
            realtime_row["open"],
            realtime_row["high"],
            realtime_row["low"],
            realtime_row["close"],
            realtime_row["volume"],
        ]
    working = working.sort_index()
    return calculate_indicators(working), elapsed_seconds


# ---------------------------------------------------------------------------
# 전략 로직
# ---------------------------------------------------------------------------

def _num(candle: pd.Series, key: str) -> float:
    value = candle.get(key)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


SYMBOL_LOG_WIDTH = 25


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _symbol_log_label(code: str, name: str, width: int = SYMBOL_LOG_WIDTH) -> str:
    label = f"{code}_{name}" if name else code
    pad = max(0, width - _display_width(label))
    return label + " " * pad


def _buy_reject_detail(
    buy_reason: str,
    cur: pd.Series,
    prev: pd.Series,
    live_price: float | None = None,
    cross_info: dict | None = None,
    frame: pd.DataFrame | None = None,
) -> str:
    """Returns reject reason + indicator context for every reject type."""
    snapshot = _buy_condition_snapshot(cur, prev, live_price=live_price, cross_info=cross_info, frame=frame)

    # 새 전략 필수조건 거부 사유 (이유 문자열에 이미 값 포함)
    if buy_reason.startswith("BB_SLOPE_NOT_RISING"):
        bb_slope = _compute_bb_slope_pct(frame) if frame is not None else float("nan")
        cur_bb = _num(cur, "BB_MIDDLE")
        return f"{buy_reason} | bb_mid={cur_bb:.1f} bb_slope={bb_slope:.3f}% | {snapshot}"

    if buy_reason == "NO_BB_MID_CROSS_UP":
        prev_bb = _num(prev, "BB_MIDDLE")
        cur_bb = _num(cur, "BB_MIDDLE")
        prev_close = _num(prev, "close")
        cur_close = _num(cur, "close")
        live_txt = f"{live_price:,.0f}" if live_price is not None and pd.notna(live_price) else "nan"
        return (
            f"{buy_reason} | live={live_txt} close={cur_close:.0f} bb_mid={cur_bb:.1f} | "
            f"prev_close={prev_close:.0f} prev_bb_mid={prev_bb:.1f} | signal={cross_info.get('signal') if cross_info else None} | {snapshot}"
        )

    if buy_reason.startswith("CANDLE_NOT_BULLISH"):
        cur_open = _num(cur, "open")
        live_txt = f"{live_price:,.0f}" if live_price is not None and pd.notna(live_price) else "nan"
        return f"{buy_reason} | live={live_txt} open={cur_open:.0f} | {snapshot}"

    if buy_reason.startswith("BB_UPPER_GAP_TOO_SMALL"):
        bb_upper = _num(cur, "BB_UPPER")
        live_txt = f"{live_price:,.0f}" if live_price is not None and pd.notna(live_price) else "nan"
        return f"{buy_reason} | live={live_txt} bb_upper={bb_upper:.1f} | {snapshot}"

    if buy_reason.startswith("LOW_VOLUME_RATIO"):
        return f"{buy_reason} | {snapshot}"

    if buy_reason.startswith("LOW_SCORE"):
        # 새 15점 채점 방식 상세 분석
        rsi_c = _num(cur, "RSI")
        vol = _num(cur, "volume"); vol_ma = _num(cur, "VOL_MA20")
        vol_ratio = vol / vol_ma if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0 else float("nan")
        adx_c = _num(cur, "ADX")
        di_plus = _num(cur, "DI_PLUS"); di_minus = _num(cur, "DI_MINUS")
        macd_c = _num(cur, "MACD"); msig_c = _num(cur, "MACD_SIGNAL")
        bb_slope = _compute_bb_slope_pct(frame) if frame is not None else float("nan")
        score = _buy_support_score(cur, prev, frame)

        rsi_str = f"{rsi_c:.1f}" if not pd.isna(rsi_c) else "nan"
        vol_str = f"{vol_ratio:.2f}x" if not pd.isna(vol_ratio) else "nan"
        adx_str = f"{adx_c:.1f}" if not pd.isna(adx_c) else "nan"
        di_str = f"+DI={di_plus:.1f}>-DI={di_minus:.1f}" if not any(pd.isna(v) for v in (di_plus, di_minus)) else "nan"
        macd_str = f"MACD>{msig_c:.3f}" if not any(pd.isna(v) for v in (macd_c, msig_c)) and macd_c > msig_c else f"MACD<={msig_c:.3f}" if not any(pd.isna(v) for v in (macd_c, msig_c)) else "nan"
        slope_str = f"{bb_slope:.2f}%" if not pd.isna(bb_slope) else "nan"
        return (
            f"{buy_reason} | RSI={rsi_str} VOL={vol_str} ADX={adx_str} {di_str} {macd_str} BB_SLOPE={slope_str} total={score}/15 | {snapshot}"
        )

    # 기타 / 하위 호환
    return f"{buy_reason} | {snapshot}"



def _buy_support_score(cur: pd.Series, prev: pd.Series, frame: pd.DataFrame | None = None) -> int:
    """로컬 표시용 점수 계산 (r005 _buy_support_score와 동일 로직)."""
    score = 0

    # RSI 구간 점수 (최대 2점)
    rsi_c = _num(cur, "RSI")
    if not pd.isna(rsi_c):
        if 50.0 <= rsi_c <= 65.0:
            score += 2
        elif 45.0 <= rsi_c < 50.0 or 65.0 < rsi_c <= 70.0:
            score += 1

    # 거래량 비율 점수 (최대 3점)
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        vol_ratio = vol / vol_ma
        if vol_ratio >= 2.0:
            score += 3
        elif vol_ratio >= 1.5:
            score += 2
        elif vol_ratio >= 1.2:
            score += 1

    # ADX 추세 강도 점수 (최대 3점)
    adx_c = _num(cur, "ADX")
    if not pd.isna(adx_c):
        if adx_c >= 35.0:
            score += 3
        elif adx_c >= 30.0:
            score += 2
        elif adx_c >= 25.0:
            score += 1

    # DI+/DI- 방향성 점수 (최대 2점)
    di_plus = _num(cur, "DI_PLUS")
    di_minus = _num(cur, "DI_MINUS")
    if not any(pd.isna(v) for v in (di_plus, di_minus)) and di_plus > di_minus:
        score += 2

    # MACD 골든크로스 점수 (최대 2점)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c)) and macd_c > msig_c:
        score += 2

    # BB 중앙선 기울기 강도 점수 (최대 3점)
    if frame is not None:
        bb_slope_pct = _compute_bb_slope_pct(frame)
        if not pd.isna(bb_slope_pct):
            if bb_slope_pct >= 1.5:
                score += 3
            elif bb_slope_pct >= 1.0:
                score += 2
            elif bb_slope_pct >= 0.5:
                score += 1

    return score
def _sell_support_score(cur: pd.Series, prev: pd.Series) -> int:
    score = 0

    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if k_p >= d_p and k_c < d_c and k_p >= STOCH_OVERBOUGHT:
            score += 1

    rsi_c = _num(cur, "RSI")
    sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI")
    sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        if rsi_p >= sig_p and rsi_c < sig_c:
            score += 1

    wr_c = _num(cur, "WILLIAMS_R")
    wd_c = _num(cur, "WILLIAMS_D")
    wr_p = _num(prev, "WILLIAMS_R")
    wd_p = _num(prev, "WILLIAMS_D")
    if not any(pd.isna(v) for v in (wr_c, wd_c, wr_p, wd_p)):
        if wr_p >= wd_p and wr_c < wd_c:
            score += 1

    # 4) MACD: 데드크로스(하향 전환)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if macd_p >= msig_p and macd_c < msig_c:
            score += 1

    # 5) OBV: OBV < OBV_MA 이고 하락 중 (거래량 방향 매도 우위)
    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_c < obv_ma_c and obv_c < obv_p:
            score += 1

    return score


def _price_lead_breakout_context(
    frame: pd.DataFrame,
    now: datetime,
    live_price: float,
    cross_info: dict[str, object],
) -> dict[str, object]:
    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    prev_close = _num(prev, "close")
    close_val = _num(cur, "close")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")
    adx_val = _num(cur, "ADX")
    macd_val = _num(cur, "MACD")
    macd_sig = _num(cur, "MACD_SIGNAL")
    hist_val = _num(cur, "MACD_HIST")
    support_score = _buy_support_score(cur, prev, frame=frame)
    near_flags = _near_cross_momentum_flags(cur, prev)
    liquidity_ok, liquidity_reason = _passes_early_near_cross_liquidity(cur)

    current_time = now.time()
    time_window_ok = (
        (is_regular_session(now) and EARLY_NEAR_CROSS_ALLOWED_START <= current_time <= EARLY_NEAR_CROSS_ALLOWED_END)
        or (EARLY_NEAR_CROSS_ALLOW_NXT and is_nxt_session(now))
    )
    price_breakout = not any(pd.isna(v) for v in (prev_close, close_val, prev_bb, cur_bb)) and (prev_close <= prev_bb) and (close_val > cur_bb)
    live_cross_up = cross_info.get("signal") == "cross_up"
    macd_momentum_ok = not any(pd.isna(v) for v in (macd_val, macd_sig, hist_val)) and macd_val > macd_sig and hist_val > 0

    near_entry_mode = None
    if time_window_ok and ENABLE_EARLY_NEAR_CROSS_ENTRY and bool(near_flags["can_early"]):
        near_entry_mode = "EARLY_NEAR_CROSS"
    elif time_window_ok and ENABLE_NEAR_CROSS_ARM and bool(near_flags["can_arm"]):
        near_entry_mode = "ARMED_NEAR_CROSS"

    can_enter = (
        ENABLE_PRICE_LEAD_BB_BREAKOUT
        and price_breakout
        and live_cross_up
        and near_entry_mode is not None
        and liquidity_ok
        and support_score >= PRICE_LEAD_BREAKOUT_MIN_SCORE
        and not pd.isna(adx_val)
        and adx_val >= PRICE_LEAD_BREAKOUT_MIN_ADX
        and macd_momentum_ok
        and live_price > cur_bb
    )

    return {
        "can_enter": can_enter,
        "entry_mode": f"PRICE_LEAD_{near_entry_mode}" if can_enter and near_entry_mode is not None else near_entry_mode,
        "support_score": support_score,
        "liquidity_ok": liquidity_ok,
        "liquidity_reason": liquidity_reason,
        "price_breakout": price_breakout,
        "live_cross_up": live_cross_up,
        "time_window_ok": time_window_ok,
        "allow_overbought": can_enter and PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT,
        "near_flags": near_flags,
        "adx": adx_val,
    }


def _buy_condition_snapshot(
    cur: pd.Series,
    prev: pd.Series,
    live_price: float | None = None,
    cross_info: dict | None = None,
    frame: pd.DataFrame | None = None,
) -> str:
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")
    cur_bb_upper = _num(cur, "BB_UPPER")
    prev_close = _num(prev, "close")
    cur_close = _num(cur, "close")
    cur_open = _num(cur, "open")
    support_score = _buy_support_score(cur, prev, frame=frame) if frame is not None else -1
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    vol_ratio = vol / vol_ma if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0 else float("nan")
    adx_c = _num(cur, "ADX")
    di_plus = _num(cur, "DI_PLUS"); di_minus = _num(cur, "DI_MINUS")
    rsi_c = _num(cur, "RSI")
    macd_c = _num(cur, "MACD"); msig_c = _num(cur, "MACD_SIGNAL")
    bb_slope_pct = _compute_bb_slope_pct(frame) if frame is not None else float("nan")
    close_cross = (
        not any(pd.isna(v) for v in (prev_close, cur_close, prev_bb, cur_bb))
        and prev_close <= prev_bb and cur_close > cur_bb
    )
    candle_gain_pct = (live_price - cur_open) / cur_open * 100.0 if (live_price and not pd.isna(cur_open) and cur_open > 0) else float("nan")
    bb_upper_gap_pct = (cur_bb_upper - live_price) / live_price * 100.0 if (live_price and live_price > 0 and not pd.isna(cur_bb_upper)) else float("nan")
    live_signal = cross_info.get("signal") if cross_info else None
    live_part = f"live={live_price:,.0f}" if live_price is not None and pd.notna(live_price) else "live=nan"
    return (
        f"GATES close_cross={close_cross} signal={live_signal} {live_part} "
        f"bb_mid={cur_bb:.1f} bb_upper={cur_bb_upper:.1f} "
        f"bb_slope={bb_slope_pct:.3f}% bb_upper_gap={bb_upper_gap_pct:.2f}% candle_gain={candle_gain_pct:.2f}% "
        f"RSI={rsi_c:.1f} ADX={adx_c:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f} MACD={macd_c:.3f} SIG={msig_c:.3f} "
        f"vol={vol:,.0f} vol_ma={vol_ma:,.0f} vol_ratio={vol_ratio:.4f} score={support_score}/15"
    )


def _rise_from_prev_close(live_price: float, prev_close: float) -> float | None:
    if live_price <= 0 or prev_close <= 0:
        return None
    return (float(live_price) / float(prev_close)) - 1.0


def _extract_score_from_buy_reason(buy_reason: str) -> int | None:
    m = re.search(r"SCORE_(\d+)", str(buy_reason or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _passes_loss_pattern_buy_filter(frame: pd.DataFrame, buy_reason: str, live_price: float) -> tuple[bool, str]:
    """Blocks entry patterns that showed repeated losses in today's live logs.

    Focus: overbought-bypass entries with weak support/momentum.
    """
    reason_text = str(buy_reason or "")
    if "OVERBOUGHT_BYPASS" not in reason_text:
        return True, "OK"

    score = _extract_score_from_buy_reason(reason_text)
    if score is not None and score <= 3:
        return False, f"LOSS_PATTERN_BLOCK_OVERBOUGHT_BYPASS_SCORE_{score}"

    cur = frame.iloc[-1]
    ma5 = _num(cur, "MA_5")
    bar_close = _num(cur, "close")
    bb_mid = _num(cur, "BB_MIDDLE")

    weak_live_vs_ma5 = not pd.isna(ma5) and live_price <= ma5
    bar_above_or_equal_ma5 = not pd.isna(bar_close) and not pd.isna(ma5) and bar_close >= ma5
    near_bb_mid = (
        not pd.isna(bb_mid)
        and bb_mid > 0
        and ((live_price / bb_mid) - 1.0) <= 0.005
    )

    if score is not None and score <= 4 and weak_live_vs_ma5 and bar_above_or_equal_ma5:
        return False, "LOSS_PATTERN_BLOCK_WEAK_OVERBOUGHT_BYPASS_MA5"

    if score is not None and score <= 4 and weak_live_vs_ma5 and near_bb_mid:
        return False, "LOSS_PATTERN_BLOCK_WEAK_OVERBOUGHT_BYPASS_BBMID"

    return True, "OK"


def check_buy_condition(
    frame: pd.DataFrame,
    now: datetime,
    live_price: float,
    cross_info: dict[str, object],
    intrabar_elapsed_seconds: float | None = None,
) -> tuple[bool, str]:
    return shared_check_buy_condition(
        frame=frame,
        now=pd.Timestamp(now),
        live_price=live_price,
        cross_info=cross_info,
        config=SHARED_R76_CONFIG,
        volume_ratio_threshold_fn=lambda ts, adx_val: get_volume_ratio_threshold(ts.to_pydatetime(), adx_val),
    )


def check_sell_condition(frame: pd.DataFrame, pnl_pct: float, live_price: float, cross_info: dict[str, object]) -> tuple[bool, str]:
    return shared_check_sell_condition(
        frame=frame,
        pnl_pct=pnl_pct,
        live_price=live_price,
        cross_info=cross_info,
        config=SHARED_R76_CONFIG,
    )



# ---------------------------------------------------------------------------
# Live state persistence (DATA_DIR/live_state/YYYYMMDD.json)
# ---------------------------------------------------------------------------

def _live_state_path(date_str: str) -> Path:
    return LIVE_STATE_DIR / f"{date_str}.json"


def _serialize_live_state(live_state: dict) -> dict:
    def _json_safe(value):
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat") and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value.isoformat()
            except Exception:
                pass
        if isinstance(value, set):
            return sorted(_json_safe(item) for item in value)
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return value

    positions_meta = {}
    for code, meta in (live_state.get("positions_meta") or {}).items():
        positions_meta[str(code).zfill(6)] = {
            "buy_time": _json_safe(meta.get("buy_time")),
            "entry_buy_time": _json_safe(meta.get("entry_buy_time") or meta.get("buy_time")),
            "buy_session": _json_safe(meta.get("buy_session")),
            "highest_price": _json_safe(meta.get("highest_price")),
            "tp1_done": bool(meta.get("tp1_done", False)),
        }
    traded = sorted({str(c).zfill(6) for c in (live_state.get("traded_today") or set())})
    return {"positions_meta": positions_meta, "traded_today": traded}


def load_live_state(date_str: str) -> dict:
    path = _live_state_path(date_str)
    if not path.exists():
        return {"date": date_str, "positions_meta": {}, "traded_today": set()}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"WARNING: live state load failed ({path}): {exc}")
        return {"date": date_str, "positions_meta": {}, "traded_today": set()}

    positions_meta: dict[str, dict] = {}
    for code, meta in (raw.get("positions_meta") or {}).items():
        norm = str(code).zfill(6)
        buy_time_raw = (meta or {}).get("buy_time")
        entry_buy_time_raw = (meta or {}).get("entry_buy_time")
        buy_time = None
        if buy_time_raw:
            try:
                buy_time = datetime.fromisoformat(str(buy_time_raw))
            except ValueError:
                buy_time = _parse_buy_time_from_holding_fields(meta)
        entry_buy_time = None
        if entry_buy_time_raw:
            try:
                entry_buy_time = datetime.fromisoformat(str(entry_buy_time_raw))
            except ValueError:
                entry_buy_time = buy_time
        if entry_buy_time is None:
            entry_buy_time = buy_time
        positions_meta[norm] = {
            "buy_time": buy_time,
            "entry_buy_time": entry_buy_time,
            "buy_session": (meta or {}).get("buy_session"),
            "highest_price": (meta or {}).get("highest_price"),
            "tp1_done": bool((meta or {}).get("tp1_done", False)),
        }
    traded = {str(c).zfill(6) for c in (raw.get("traded_today") or [])}
    return {"date": date_str, "positions_meta": positions_meta, "traded_today": traded}


def save_live_state(live_state: dict, date_str: str | None = None) -> None:
    date_key = date_str or str(live_state.get("date") or _LOG_CTX.get("date_str") or datetime.now().strftime("%Y%m%d"))
    live_state["date"] = date_key
    path = _live_state_path(date_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_live_state(live_state)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    save_today_buy_codes(date_key, {str(c).zfill(6) for c in (payload.get("traded_today") or [])})


def _parse_buy_time_from_holding_fields(row_or_meta: dict) -> datetime | None:
    if not isinstance(row_or_meta, dict):
        return None
    for key in ("pchs_dt", "buy_dt", "ord_dt", "pchs_date"):
        raw = row_or_meta.get(key)
        if raw in (None, ""):
            continue
        text = str(raw).strip()
        try:
            if len(text) >= 14 and text[:14].isdigit():
                return datetime.strptime(text[:14], "%Y%m%d%H%M%S")
            if len(text) >= 8 and text[:8].isdigit():
                return datetime.strptime(text[:8], "%Y%m%d")
        except ValueError:
            continue
    return None


def _is_stale_live_price_source(price_source: str) -> bool:
    src = str(price_source or "")
    return "stale_live=" in src or src.startswith("bar_close(stale")


def _log_account_watchlist_mismatch(api, watch_map: dict[str, str]) -> None:
    open_codes = {str(c).zfill(6) for c, p in api.get_open_positions().items() if int(p.get("quantity", 0) or 0) > 0}
    watch_codes = {str(c).zfill(6) for c in watch_map}
    extra = sorted(open_codes - watch_codes)
    if extra:
        log(f"WARNING: account holdings not in watchlist: {', '.join(extra)}")


# ---------------------------------------------------------------------------
# Trading API
# ---------------------------------------------------------------------------

class TradingAPI:
    def __init__(self, env_dv: str | None = None, dry_run: bool = False, live_state: dict | None = None):
        self.env_dv = (env_dv or KIS_ENV_DV).strip() or "real"
        self.dry_run = bool(dry_run)
        self.live_state = live_state if live_state is not None else {"positions_meta": {}, "traded_today": set()}
        ka.auth()

        trenv = ka.getTREnv()
        self.cano = trenv.my_acct
        self.acnt_prdt_cd = trenv.my_prod
        self.positions: dict[str, dict] = {}
        self.pending_orders: dict[str, dict] = {}
        self.buy_inflight_codes: set[str] = set()
        self.startup_position_codes: set[str] = set()
        self.trade_lock_until: dict[str, datetime] = {}
        self._last_sync_at: datetime | None = None
        self._last_pending_poll_at: datetime | None = None
        self._last_live_state_save_at: datetime | None = None
        self.sync_positions_from_account(force=True)
        self._apply_persisted_position_meta()
        for code, pos in self.positions.items():
            if int(pos.get("quantity", 0) or 0) > 0:
                norm_code = str(code).zfill(6)
                self.buy_inflight_codes.add(norm_code)
                self.startup_position_codes.add(norm_code)
        log(
            f"TradingAPI (r76 MA5-BB multi-indicator) initialized | env_dv={self.env_dv} | dry_run={self.dry_run}"
        )

    def _apply_persisted_position_meta(self) -> None:
        meta_map = self.live_state.get("positions_meta") or {}
        for code, pos in self.positions.items():
            meta = meta_map.get(code) or {}
            if meta.get("buy_time") is not None:
                pos["buy_time"] = meta["buy_time"]
            if meta.get("entry_buy_time") is not None:
                pos["entry_buy_time"] = meta["entry_buy_time"]
            if meta.get("buy_session"):
                pos["buy_session"] = meta["buy_session"]
            if meta.get("highest_price") is not None:
                pos["highest_price"] = max(
                    float(pos.get("highest_price", 0.0)),
                    float(meta.get("highest_price", 0.0)),
                    float(pos.get("current_price", 0.0)),
                )
            pos["tp1_done"] = bool(meta.get("tp1_done", pos.get("tp1_done", False)))

    def _record_position_meta(self, code: str, pos: dict) -> None:
        meta_map = self.live_state.setdefault("positions_meta", {})
        meta_map[str(code).zfill(6)] = {
            "buy_time": pos.get("entry_buy_time") or pos.get("buy_time"),
            "entry_buy_time": pos.get("entry_buy_time") or pos.get("buy_time"),
            "buy_session": pos.get("buy_session"),
            "highest_price": pos.get("highest_price"),
            "tp1_done": bool(pos.get("tp1_done", False)),
        }

    def _sync_live_state_from_positions(self) -> None:
        for code, pos in self.positions.items():
            if int(pos.get("quantity", 0) or 0) > 0:
                self._record_position_meta(code, pos)

    def persist_live_state(self, date_str: str | None = None) -> None:
        self._sync_live_state_from_positions()
        save_live_state(self.live_state, date_str=date_str)
        self._last_live_state_save_at = datetime.now()

    def maybe_persist_live_state_interval(self, now: datetime, date_str: str) -> None:
        if self._last_live_state_save_at is None:
            self.persist_live_state(date_str=date_str)
            return
        elapsed = (now - self._last_live_state_save_at).total_seconds()
        if elapsed >= LIVE_STATE_SAVE_INTERVAL_SECONDS:
            self.persist_live_state(date_str=date_str)

    def sync_positions_from_account(self, force: bool = False) -> None:
        if not force and self._last_sync_at is not None:
            elapsed = (datetime.now() - self._last_sync_at).total_seconds()
            if elapsed < ACCOUNT_SYNC_INTERVAL_SECONDS:
                return

        balance_fn = getattr(dsf, "inquire_balance_rlz_pl", None)
        if balance_fn is None:
            return

        try:
            df_holdings, _ = balance_fn(
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                afhr_flpr_yn="N",
                inqr_dvsn="02",
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00",
            )
        except Exception as exc:
            log(f"WARNING: holdings sync failed: {exc}")
            return

        self._last_sync_at = datetime.now()

        if df_holdings is None or df_holdings.empty:
            if self.positions:
                log(
                    "WARNING: holdings API empty but local positions non-empty; keeping previous positions"
                )
            else:
                self.positions.clear()
            self._reconcile_pending_positions()
            return

        updated: dict[str, dict] = {}
        for _, row in df_holdings.iterrows():
            code = str(row.get("pdno", "")).zfill(6)
            try:
                qty = int(float(row.get("hldg_qty", 0)))
                avg_price = float(row.get("pchs_avg_pric", 0))
                current_price = float(row.get("prpr", avg_price))
            except (TypeError, ValueError):
                continue
            if qty <= 0 or avg_price <= 0:
                continue

            prev = self.positions.get(code, {})
            persisted = (self.live_state.get("positions_meta") or {}).get(code) or {}
            buy_time = prev.get("buy_time") or persisted.get("buy_time")
            entry_buy_time = prev.get("entry_buy_time") or persisted.get("entry_buy_time") or buy_time
            if buy_time is None:
                buy_time = _parse_buy_time_from_holding_fields(row.to_dict() if hasattr(row, "to_dict") else dict(row))
            if entry_buy_time is None:
                entry_buy_time = buy_time
            buy_session = prev.get("buy_session") or persisted.get("buy_session") or "synced"
            highest = max(
                float(prev.get("highest_price", current_price)),
                float(persisted.get("highest_price", current_price) or current_price),
                current_price,
            )
            updated[code] = {
                "buy_price": avg_price,
                "quantity": qty,
                "buy_time": buy_time,
                "entry_buy_time": entry_buy_time,
                "buy_session": buy_session,
                "current_price": current_price,
                "highest_price": highest,
                "tp1_done": bool(prev.get("tp1_done", persisted.get("tp1_done", False))),
            }
            self._record_position_meta(code, updated[code])

        self.positions = updated
        self._apply_persisted_position_meta()
        self._reconcile_pending_positions()

    def get_open_positions(self) -> dict[str, dict]:
        return self.positions

    def has_pending_order(self, code: str) -> bool:
        return str(code).zfill(6) in self.pending_orders

    def has_buy_exposure(self, code: str) -> bool:
        norm = str(code).zfill(6)
        if norm in self.buy_inflight_codes:
            return True
        if self.has_pending_order(norm):
            return True
        pos = self.positions.get(norm)
        if pos is not None and int(pos.get("quantity", 0) or 0) > 0:
            return True
        return False

    def get_pending_order(self, code: str) -> dict | None:
        return self.pending_orders.get(str(code).zfill(6))

    def _in_cooldown(self, code: str, now: datetime) -> bool:
        key = str(code).zfill(6)
        until = self.trade_lock_until.get(key)
        return until is not None and now < until
        entry_time = pos.get("entry_buy_time") or pos.get("buy_time")
        buy_time = entry_time
    def _mark_trade_lock(self, code: str, now: datetime) -> None:
        key = str(code).zfill(6)
        self.trade_lock_until[key] = now + timedelta(minutes=TRADE_COOLDOWN_MINUTES)

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            text = str(value).replace(",", "").strip()
            if text == "":
                return default
            return int(float(text))
        except Exception:
            return default

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            text = str(value).replace(",", "").strip()
            if text == "":
                return default
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _pending_side_to_ccld_code(side: str) -> str:
        return "02" if side == "buy" else "01"

    def _reconcile_pending_positions(self) -> None:
        for code, pending in self.pending_orders.items():
            if pending.get("side") != "buy":
                continue
            pos = self.positions.get(code)
            if pos is None:
                continue
            pos["buy_time"] = pending.get("submitted_at", pos.get("buy_time", datetime.now()))
            pos["buy_session"] = pending.get("session", pos.get("buy_session", "synced"))
            pos["highest_price"] = max(
                float(pos.get("highest_price", 0.0)),
                float(pos.get("current_price", 0.0)),
                float(pos.get("buy_price", 0.0)),
            )

    def _fetch_today_order_status(self, code: str, side: str, now: datetime, order_no: str = "") -> dict[str, object] | None:
        ccld_fn = getattr(dsf, "inquire_daily_ccld", None)
        if not callable(ccld_fn):
            return None

        date_str = now.strftime("%Y%m%d")
        try:
            df_orders, _ = ccld_fn(
                env_dv=self.env_dv,
                pd_dv="inner",
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                inqr_strt_dt=date_str,
                inqr_end_dt=date_str,
                sll_buy_dvsn_cd=self._pending_side_to_ccld_code(side),
                ccld_dvsn="00",
                inqr_dvsn="00",
                inqr_dvsn_3="00",
                pdno=code,
                odno=order_no,
                excg_id_dvsn_cd="ALL",
            )
        except Exception as exc:
            log(f"WARNING: order status query failed for {code}({side}): {exc}")
            return None

        if df_orders is None or df_orders.empty:
            return None

        orders = df_orders.copy()
        if "pdno" in orders.columns:
            orders = orders[orders["pdno"].astype(str).str.zfill(6) == code]
        if order_no and "odno" in orders.columns:
            matched = orders[orders["odno"].astype(str).str.strip() == order_no]
            if not matched.empty:
                orders = matched
        if orders.empty:
            return None

        sort_columns = [col for col in ("ord_dt", "ord_tmd", "odno") if col in orders.columns]
        if sort_columns:
            orders = orders.sort_values(sort_columns)

        row = orders.iloc[-1].to_dict()
        order_qty = self._to_int(row.get("ord_qty"), 0)
        filled_qty = self._to_int(row.get("tot_ccld_qty"), 0)
        remaining_qty = self._to_int(row.get("rmn_qty"), max(0, order_qty - filled_qty))
        rejected_qty = self._to_int(row.get("rjct_qty"), 0)
        avg_price = self._to_float(row.get("avg_prvs"), 0.0)
        cancel_yn = str(row.get("cncl_yn", "")).strip().upper()

        return {
            "order_no": str(row.get("odno", "")).strip(),
            "order_time": str(row.get("ord_tmd", "")).strip(),
            "order_qty": order_qty,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "rejected_qty": rejected_qty,
            "avg_price": avg_price,
            "cancel_yn": cancel_yn,
        }

    def _maybe_log_pending_progress(self, pending: dict, message: str, signature: str) -> None:
        if pending.get("last_status_signature") == signature:
            return
        pending["last_status_signature"] = signature
        log(message)

    def _confirm_pending_buy(self, code: str, pending: dict, pos: dict, status: dict | None) -> None:
        requested_qty = int(pending.get("quantity", 0))
        filled_qty = int(status.get("filled_qty", 0)) if status else int(pos.get("quantity", 0))
        avg_price = float(status.get("avg_price", 0.0)) if status else 0.0
        fill_price = avg_price if avg_price > 0 else float(pos.get("buy_price") or pos.get("current_price") or pending.get("requested_price", 0.0))

        entry_time = pending.get("submitted_at", pos.get("entry_buy_time", pos.get("buy_time", datetime.now())))
        pos["buy_time"] = entry_time
        pos["entry_buy_time"] = entry_time
        pos["buy_session"] = pending.get("session", pos.get("buy_session", "synced"))
        pos["tp1_done"] = False
        if fill_price > 0:
            pos["buy_price"] = fill_price
            pos["current_price"] = float(pos.get("current_price") or fill_price)
            pos["highest_price"] = max(float(pos.get("highest_price", fill_price)), float(pos["current_price"]), fill_price)

        code_name = str(pending.get("code_name", ""))
        code_label = _format_code_label(code, code_name)
        compact_code_label = f"{code}_{code_name}" if code_name else code
        event_time = datetime.now()
        buy_time_label = _format_trade_time_label(pending.get("submitted_at"))
        buy_price_label = f"{fill_price:,.0f}" if fill_price > 0 else "N/A"
        buy_detail_text = str(pending.get("buy_detail", "")).strip()
        summary_line = f">>> {compact_code_label} Buy - {buy_detail_text}" if buy_detail_text else f">>> {compact_code_label} Buy"
        _log_trade_block(
            [
                "=" * 110,
                summary_line,
                f"*** 매수시각={buy_time_label} | 매수가격={buy_price_label} | 매수수량={filled_qty}",
            ],
            event_time=event_time,
            mirror_main_log=False,
        )
        self._record_position_meta(code, pos)
        self.startup_position_codes.discard(str(code).zfill(6))
        self.persist_live_state()
        self.pending_orders.pop(str(code).zfill(6), None)

    def _confirm_pending_sell(self, code: str, pending: dict, status: dict | None, remaining_qty: int) -> None:
        requested_qty = int(pending.get("quantity", 0))
        filled_qty = int(status.get("filled_qty", requested_qty)) if status else requested_qty
        avg_price = float(status.get("avg_price", 0.0)) if status else 0.0
        fill_price = avg_price if avg_price > 0 else float(pending.get("requested_price", 0.0))
        buy_price = float(pending.get("buy_price", 0.0))
        pnl_pct = ((fill_price / buy_price) - 1.0) * 100.0 if buy_price > 0 and fill_price > 0 else float("nan")
        profit_amount = (fill_price - buy_price) * filled_qty if buy_price > 0 and fill_price > 0 and filled_qty > 0 else float("nan")
        event_time = datetime.now()
        buy_time_label = _format_trade_time_label(pending.get("buy_time"))
        sell_time_label = _format_trade_time_label(event_time)
        buy_price_label = f"{buy_price:,.0f}" if buy_price > 0 else "N/A"
        fill_price_label = f"{fill_price:,.0f}" if fill_price > 0 else "N/A"
        profit_amount_label = f"{profit_amount:,.0f}원" if pd.notna(profit_amount) else "N/A"
        pnl_pct_label = f"{pnl_pct:.2f}%" if pd.notna(pnl_pct) else "N/A"
        code_name = str(pending.get("code_name", "")).strip() or _SYMBOL_NAME_MAP.get(str(code).zfill(6), "")
        code_label = _format_code_label(code, code_name)
        compact_code_label = f"{code}_{code_name}" if code_name else code
        buy_qty = int(pending.get("pre_submit_qty", filled_qty))
        reason_text = str(pending.get("reason", "UNKNOWN"))
        exch_text = str(pending.get("exchange", "UNKNOWN"))
        log(
            f"{code_label} SELL executed  | filled_qty={filled_qty}/{requested_qty} | "
            f"price={fill_price:,.0f} | remaining={remaining_qty} | pnl={pnl_pct:.2f}% | "
            f"reason={reason_text} | exch={exch_text}"
        )
        _log_trade_block(
            [
                "=" * 110,
                f">>> {compact_code_label} Sell - 수익금액={profit_amount_label} | 수익율={pnl_pct_label} | reason={reason_text} exch={exch_text}",
                f"*** 매수시각={buy_time_label} | 매수가격={buy_price_label} | 매수수량={buy_qty}",
                f"*** 매도시각={sell_time_label} | 매도가격={fill_price_label} | 매도수량={filled_qty}",
            ],
            event_time=event_time,
            mirror_main_log=False,
        )
        if remaining_qty <= 0:
            self.buy_inflight_codes.discard(str(code).zfill(6))
            self.startup_position_codes.discard(str(code).zfill(6))
            self.positions.pop(code, None)
            (self.live_state.get("positions_meta") or {}).pop(str(code).zfill(6), None)
            traded = self.live_state.setdefault("traded_today", set())
            traded.discard(str(code).zfill(6))
        self.persist_live_state()
        self.pending_orders.pop(str(code).zfill(6), None)

    def refresh_pending_orders(self, now: datetime) -> None:
        if not self.pending_orders:
            return
        if self._last_pending_poll_at is not None:
            elapsed = (now - self._last_pending_poll_at).total_seconds()
            if elapsed < ORDER_STATUS_POLL_INTERVAL_SECONDS:
                return

        self._last_pending_poll_at = now
        self.sync_positions_from_account(force=False)

        for code, pending in list(self.pending_orders.items()):
            next_poll_at = pending.get("next_status_poll_at")
            if isinstance(next_poll_at, datetime) and now < next_poll_at:
                continue

            side = str(pending.get("side", ""))
            pos = self.positions.get(code)
            status = self._fetch_today_order_status(code, side, now, str(pending.get("order_no", "")))

            if status is None:
                fail_count = int(pending.get("status_fail_count", 0)) + 1
                pending["status_fail_count"] = fail_count
                pending["next_status_poll_at"] = now + timedelta(seconds=_pending_status_backoff_seconds(fail_count))
            else:
                pending["status_fail_count"] = 0
                pending.pop("next_status_poll_at", None)

            if side == "buy":
                if pos is not None and pos.get("quantity", 0) > 0:
                    if status and int(status.get("remaining_qty", 0)) > 0:
                        self._maybe_log_pending_progress(
                            pending,
                            f"BUY partial | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                            f"| remaining={int(status.get('remaining_qty', 0))} | avg={float(status.get('avg_price', 0.0)):,.0f}",
                            f"buy_partial:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}:{float(status.get('avg_price', 0.0))}",
                        )
                        continue
                    self._confirm_pending_buy(code, pending, pos, status)
                    continue

                if status is not None:
                    if int(status.get("filled_qty", 0)) <= 0 and (
                        int(status.get("remaining_qty", 0)) <= 0
                        or str(status.get("cancel_yn", "")) == "Y"
                        or int(status.get("rejected_qty", 0)) >= int(status.get("order_qty", pending.get("quantity", 0)))
                    ):
                        submitted_at = pending.get("submitted_at")
                        if isinstance(submitted_at, datetime):
                            if (now - submitted_at).total_seconds() < PENDING_BUY_GRACE_SECONDS:
                                continue
                        self._maybe_log_pending_progress(
                            pending,
                            f"BUY closed without fill | {code} | order_no={pending.get('order_no', '')} | exch={pending.get('exchange', 'UNKNOWN')}",
                            "buy_closed_without_fill",
                        )
                        self.buy_inflight_codes.discard(str(code).zfill(6))
                        self.pending_orders.pop(str(code).zfill(6), None)
                        continue

                    self._maybe_log_pending_progress(
                        pending,
                        f"BUY pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                        f"| remaining={int(status.get('remaining_qty', 0))} | order_no={status.get('order_no', '')}",
                        f"buy_pending:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}",
                    )
                    log_trade(
                        f"BUY pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                        f"| remaining={int(status.get('remaining_qty', 0))} | order_no={status.get('order_no', '')}"
                    )
                continue

            pre_submit_qty = int(pending.get("pre_submit_qty", pending.get("quantity", 0)))
            current_qty = int(pos.get("quantity", 0)) if pos is not None else 0
            expected_remaining_qty = max(0, pre_submit_qty - int(pending.get("quantity", 0)))

            if status is not None:
                filled_qty = int(status.get("filled_qty", 0))
                remaining_qty = int(status.get("remaining_qty", 0))
                rejected_qty = int(status.get("rejected_qty", 0))
                order_qty = int(status.get("order_qty", pending.get("quantity", 0)))
                cancel_yn = str(status.get("cancel_yn", ""))
                terminal = remaining_qty <= 0 or cancel_yn == "Y" or rejected_qty >= order_qty
                if filled_qty > 0 and terminal:
                    self._confirm_pending_sell(code, pending, status, current_qty)
                    continue

            if current_qty <= expected_remaining_qty:
                self._confirm_pending_sell(code, pending, status, current_qty)
                continue

            if status is not None:
                if int(status.get("filled_qty", 0)) <= 0 and (
                    int(status.get("remaining_qty", 0)) <= 0
                    or str(status.get("cancel_yn", "")) == "Y"
                    or int(status.get("rejected_qty", 0)) >= int(status.get("order_qty", pending.get("quantity", 0)))
                ):
                    self._maybe_log_pending_progress(
                        pending,
                        f"SELL closed without fill | {code} | reason={pending.get('reason', 'UNKNOWN')} | order_no={pending.get('order_no', '')}",
                        "sell_closed_without_fill",
                    )
                    self.pending_orders.pop(code, None)
                    continue

                self._maybe_log_pending_progress(
                    pending,
                    f"SELL pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                    f"| remaining={int(status.get('remaining_qty', 0))} | current_qty={current_qty} | reason={pending.get('reason', 'UNKNOWN')}",
                    f"sell_pending:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}:{current_qty}",
                )

    def get_affordable_buy_qty(self, code: str, price: float, now: datetime, nxt_tradeable: bool) -> int:
        price_krw = max(1, int(round(price)))
        qty_by_budget = int(MAX_ORDER_AMOUNT_KRW / price_krw)
        if qty_by_budget <= 0:
            return 0

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return 0

        psbl_fn = getattr(dsf, "inquire_psbl_order", None)
        if not callable(psbl_fn):
            return qty_by_budget

        try:
            psbl_df = psbl_fn(
                env_dv=self.env_dv,
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                pdno=code,
                ord_unpr=str(price_krw),
                ord_dvsn=order_spec["ord_dvsn"],
                cma_evlu_amt_icld_yn="N",
                ovrs_icld_yn="N",
            )
        except Exception as exc:
            log(f"WARNING: inquire_psbl_order failed for {code}: {exc}")
            return qty_by_budget

        if psbl_df is None or psbl_df.empty:
            return qty_by_budget

        row = psbl_df.iloc[-1].to_dict()
        qty_candidates = ("nrcvb_buy_qty", "max_buy_qty", "ord_psbl_qty")
        qty_by_psbl = 0
        for key in qty_candidates:
            qty_by_psbl = max(qty_by_psbl, self._to_int(row.get(key), 0))

        # 응답 포맷 차이를 대비해 유사 컬럼명도 허용
        if qty_by_psbl <= 0:
            for key, value in row.items():
                key_text = str(key).lower()
                if "qty" in key_text and ("buy" in key_text or "psbl" in key_text or "ord" in key_text):
                    qty_by_psbl = max(qty_by_psbl, self._to_int(value, 0))

        if qty_by_psbl <= 0:
            amt_candidates = ("nrcvb_buy_amt", "max_buy_amt", "ord_psbl_cash", "ord_psbl_amt")
            psbl_amt = 0
            for key in amt_candidates:
                psbl_amt = max(psbl_amt, self._to_int(row.get(key), 0))

            if psbl_amt <= 0:
                for key, value in row.items():
                    key_text = str(key).lower()
                    if "amt" in key_text or "cash" in key_text:
                        if "buy" in key_text or "psbl" in key_text or "ord" in key_text:
                            psbl_amt = max(psbl_amt, self._to_int(value, 0))

            if psbl_amt > 0:
                qty_by_psbl = int(psbl_amt / price_krw)

        if qty_by_psbl <= 0:
            # 주문가 응답이 예상과 다르거나 수량/금액 해석이 안되면 과주문 방지 위해 보수적으로 0 처리
            log(f"WARNING: psbl-order parse failed for {code}; force qty=0 to avoid over-order")
            return 0

        return max(0, min(qty_by_budget, qty_by_psbl))

    def place_buy_order(self, code: str, price: float, qty: int, now: datetime, nxt_tradeable: bool, session: str, buy_detail: str = "", code_name: str = "") -> bool:
        norm_code = str(code).zfill(6)
        if self.has_buy_exposure(norm_code):
            log(f"BUY skipped | {code} | reason=BUY_EXPOSURE_ACTIVE")
            return False
        if qty <= 0 or self._in_cooldown(norm_code, now) or self.has_pending_order(norm_code):
            return False

        affordable_qty = self.get_affordable_buy_qty(code, price, now, nxt_tradeable)
        qty = min(int(qty), int(affordable_qty))
        if qty <= 0:
            log(f"BUY skipped | {code} | reason=INSUFFICIENT_BUYING_POWER_AT_ORDER_TIME")
            return False

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return False

        market_div = "NX" if order_spec["exchange"] == "NXT" else "J"
        bid_price, _ = _fetch_bid_ask_price(norm_code, market_div)
        limit_price = int(round(bid_price)) if (bid_price and bid_price > 0) else int(round(price))
        ord_dvsn = "00"  # 매수1호가 지정가 주문
        ord_unpr = str(limit_price)
        self.buy_inflight_codes.add(norm_code)
        if self.dry_run:
            log(f"DRY_RUN BUY | {code} | qty={qty} | price={price:,.0f} | session={session} | exch={order_spec['exchange']}")
            order_result = {"rt_cd": "0", "odno": "DRYRUN", "avg_pric": str(int(round(price)))}
        else:
            try:
                order_result = dsf.order_cash(
                    env_dv=self.env_dv,
                    ord_dv="buy",
                    cano=self.cano,
                    acnt_prdt_cd=self.acnt_prdt_cd,
                    pdno=code,
                    ord_dvsn=ord_dvsn,
                    ord_qty=str(qty),
                    ord_unpr=ord_unpr,
                    excg_id_dvsn_cd=order_spec["exchange"],
                )
            except Exception as exc:
                self.buy_inflight_codes.discard(norm_code)
                log(f"BUY error | {code} | {exc}")
                return False
        if not _order_succeeded(order_result):
            self.buy_inflight_codes.discard(norm_code)
            error_detail = _extract_order_error_detail(order_result)
            log(f"BUY failed | {code} | qty={qty} | {error_detail}")
            return False

        requested_price = _extract_order_price(order_result) or price
        order_no = _extract_order_number(order_result)
        order_time = _extract_order_time(order_result)
        self.pending_orders[norm_code] = {
            "side": "buy",
            "quantity": int(qty),
            "submitted_at": now,
            "session": session,
            "requested_price": float(requested_price),
            "exchange": order_spec["exchange"],
            "order_no": order_no,
            "order_time": order_time,
            "buy_detail": buy_detail,
            "code_name": code_name,
        }
        self._mark_trade_lock(norm_code, now)
        detail_suffix = f" | {buy_detail}" if buy_detail else ""
        code_label = _format_code_label(code, code_name)
        log(
            f"BUY submitted | {code_label} | qty={qty} | requested={requested_price:,.0f} | "
            f"session={session} | exch={order_spec['exchange']} | order_no={order_no or 'UNKNOWN'}{detail_suffix}"
        )
        log_trade(
            f"BUY submitted | {code_label} | qty={qty} | requested={requested_price:,.0f} | "
            f"session={session} | exch={order_spec['exchange']} | order_no={order_no or 'UNKNOWN'}{detail_suffix}"
        )
        _log_trade_event_banner(
            event="BUY SUBMITTED",
            code=code,
            qty=int(qty),
            price=float(requested_price),
            detail=buy_detail,
            code_name=code_name,
        )
        traded = self.live_state.setdefault("traded_today", set())
        traded.add(norm_code)
        self.persist_live_state()
        return True

    def place_sell_order(self, code: str, qty: int, now: datetime, reason: str, nxt_tradeable: bool, price: float | None = None, code_name: str = "", market_order: bool = False) -> bool:
        self.sync_positions_from_account(force=False)
        pos = self.positions.get(code)
        if not pos or pos.get("quantity", 0) <= 0:
            return False

        norm_code = str(code).zfill(6)
        qty = min(int(qty), int(pos["quantity"]))
        if qty <= 0 or self._in_cooldown(norm_code, now) or self.has_pending_order(norm_code):
            return False

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return False

        current_price = float(price or pos.get("current_price") or pos["buy_price"])
        market_div = "NX" if order_spec["exchange"] == "NXT" else "J"
        # 손절 시장가: 정규장(KRX)은 시장가(01), NXT는 지정가(ask) 유지
        use_market = market_order and order_spec["exchange"] != "NXT"
        if use_market:
            ord_dvsn = "01"  # 시장가
            ord_unpr = "0"
        else:
            _, ask_price = _fetch_bid_ask_price(norm_code, market_div)
            limit_price = int(round(ask_price)) if (ask_price and ask_price > 0) else int(round(current_price))
            ord_dvsn = "00"  # 지정가
            ord_unpr = str(limit_price)

        if self.dry_run:
            log(f"DRY_RUN SELL | {code} | qty={qty} | reason={reason} | exch={order_spec['exchange']}")
            order_result = {"rt_cd": "0", "odno": "DRYRUN", "avg_pric": str(int(round(current_price)))}
        else:
            try:
                order_result = dsf.order_cash(
                    env_dv=self.env_dv,
                    ord_dv="sell",
                    cano=self.cano,
                    acnt_prdt_cd=self.acnt_prdt_cd,
                    pdno=code,
                    ord_dvsn=ord_dvsn,
                    ord_qty=str(qty),
                    ord_unpr=ord_unpr,
                    excg_id_dvsn_cd=order_spec["exchange"],
                )
            except Exception as exc:
                log(f"SELL error | {code} | {exc}")
                return False

        if not _order_succeeded(order_result):
            error_detail = _extract_order_error_detail(order_result)
            log(f"SELL failed | {code} | qty={qty} | reason={reason} | {error_detail}")
            return False

        requested_price = _extract_order_price(order_result) or current_price
        order_no = _extract_order_number(order_result)
        order_time = _extract_order_time(order_result)
        self.pending_orders[norm_code] = {
            "side": "sell",
            "quantity": int(qty),
            "submitted_at": now,
            "requested_price": float(requested_price),
            "exchange": order_spec["exchange"],
            "order_no": order_no,
            "order_time": order_time,
            "reason": reason,
            "code_name": code_name,
            "buy_price": float(pos.get("buy_price", 0.0)),
            "buy_time": pos.get("buy_time"),
            "pre_submit_qty": int(pos.get("quantity", qty)),
        }
        self._mark_trade_lock(code, now)
        code_label = _format_code_label(code, code_name)
        log(
            f"SELL submitted | {code_label} | qty={qty} | requested={requested_price:,.0f} | "
            f"reason={reason} | exch={order_spec['exchange']} | order_no={order_no or 'UNKNOWN'}"
        )
        self.refresh_pending_orders(now)
        return True


# ---------------------------------------------------------------------------
# 예약 청산
# ---------------------------------------------------------------------------

def run_scheduled_liquidations(
    current_dt: datetime,
    api: TradingAPI,
    nxt_map: dict[str, bool],
    watch_map: dict[str, str],
    state: dict,
    date_str: str,
    today_buy_codes: set[str],
) -> None:
    trade_date = current_dt.date()
    current_time = current_dt.time()

    if state.get("date") != trade_date:
        state["date"] = trade_date
        state["done_1520"] = False
        state["done_1959"] = False

    if not state["done_1520"] and current_time >= REGULAR_FORCE_EXIT:
        state["done_1520"] = True
        for code, pos in list(api.get_open_positions().items()):
            if code not in watch_map:
                continue
            if not _is_today_buy_position(code, pos, date_str, today_buy_codes):
                log(f"  [REGULAR CLOSE SKIP] {code} | NOT_TODAY_BUY_POSITION")
                continue
            if api.has_pending_order(code):
                log(f"  [REGULAR CLOSE SKIP] {code} | pending_order_active")
                continue
            price = float(pos.get("current_price") or pos["buy_price"])
            buy_price = float(pos.get("buy_price") or 0)
            if buy_price <= 0 or price <= 0:
                log(f"  [REGULAR CLOSE HOLD] {code} | INVALID_PRICE | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            pnl_pct = (price / buy_price) - 1.0
            action, reason = _session_exit_plan("REGULAR_CLOSE", pnl_pct)
            if action == "hold":
                log(f"  [REGULAR CLOSE HOLD] {code} | {reason} | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            api.trade_lock_until.pop(code, None)
            api.place_sell_order(code, int(pos["quantity"]), current_dt, reason, nxt_map.get(code, False), price=price, code_name=watch_map.get(code, ""))

    if ENABLE_NXT_SESSION and (not state["done_1959"]) and current_time >= AFTERNOON_NXT_FORCE_EXIT:
        state["done_1959"] = True
        for code, pos in list(api.get_open_positions().items()):
            if code not in watch_map:
                continue
            if not _is_today_buy_position(code, pos, date_str, today_buy_codes):
                log(f"  [NXT CLOSE SKIP] {code} | NOT_TODAY_BUY_POSITION")
                continue
            if api.has_pending_order(code):
                log(f"  [NXT CLOSE SKIP] {code} | pending_order_active")
                continue
            if not nxt_map.get(code, False):
                log(f"  [NXT CLOSE HOLD] {code} | NXT_NOT_TRADABLE")
                continue

            price = float(pos.get("current_price") or pos["buy_price"])
            buy_price = float(pos.get("buy_price") or 0)
            if buy_price <= 0 or price <= 0:
                log(f"  [NXT CLOSE HOLD] {code} | INVALID_PRICE | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            pnl_pct = (price / buy_price) - 1.0
            action, reason = _session_exit_plan("NXT_CLOSE", pnl_pct)
            if action == "hold":
                log(f"  [NXT CLOSE HOLD] {code} | {reason} | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            api.trade_lock_until.pop(code, None)
            api.place_sell_order(code, int(pos["quantity"]), current_dt, reason, nxt_map.get(code, False), price=price, code_name=watch_map.get(code, ""))



_SHUTDOWN_API: TradingAPI | None = None


def _ensure_log_date_for(now: datetime | None = None) -> None:
    dt = now or datetime.now()
    date_str = dt.strftime("%Y%m%d")
    if str(_LOG_CTX.get("date_str")) != date_str:
        _rotate_logging_for_date(date_str)


def _log_pending_orders_on_shutdown(api: TradingAPI | None) -> None:
    if api is None or not getattr(api, "pending_orders", None):
        log("SHUTDOWN | no pending orders")
        return
    if not api.pending_orders:
        log("SHUTDOWN | no pending orders")
        return
    log(f"SHUTDOWN | pending_orders={len(api.pending_orders)}")
    for code, pending in api.pending_orders.items():
        log(
            f"  PENDING | {code} | side={pending.get('side')} qty={pending.get('quantity')} "
            f"order_no={pending.get('order_no', '')} submitted={pending.get('submitted_at')}"
        )


def _install_shutdown_handlers(api: TradingAPI) -> None:
    global _SHUTDOWN_API
    _SHUTDOWN_API = api

    def _atexit_shutdown() -> None:
        _log_pending_orders_on_shutdown(_SHUTDOWN_API)
        _shutdown_save_live_state(_SHUTDOWN_API)

    atexit.register(_atexit_shutdown)

    def _handler(signum, frame):  # noqa: ARG001
        _log_pending_orders_on_shutdown(api)
        _shutdown_save_live_state(api)
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def _shutdown_save_live_state(api: TradingAPI | None) -> None:
    if api is None:
        return
    try:
        api.persist_live_state()
    except Exception as exc:
        log(f"WARNING: live state save on shutdown failed: {exc}")


def run(target_date: str | None = None, env_dv: str | None = None, dry_run: bool | None = None, watchlist_source: str | None = None) -> None:
    now = datetime.now()
    _ensure_log_date_for(now)
    print(f"[R006 START] {now:%Y-%m-%d %H:%M:%S} | initializing live executor", flush=True)
    log("R006 START | initializing live executor")

    try:
        print("[R006 AUTH] starting ka.auth()", flush=True)
        ka.auth()
        print("[R006 AUTH] success", flush=True)
    except Exception as exc:
        print(f"[R006 AUTH ERROR] {exc}", flush=True)
        log(f"R006 AUTH ERROR | {exc}")
        return

    _bind_session_trade_log()
    is_open_day, market_day_log = get_market_day_status(now)
    print(f"[R006 MARKET] {market_day_log}", flush=True)
    log(market_day_log)

    if not is_open_day:
        print("[R006 STOP] market closed day", flush=True)
        return

    watch_file = _resolve_watchlist_file(target_date, watchlist_source=watchlist_source or "auto")
    print(f"[R006 WATCHLIST FILE] {watch_file}", flush=True)
    try:
        watch_map = load_today_codes(watch_file)
    except Exception as exc:
        print(f"[R006 WATCHLIST ERROR] {exc}", flush=True)
        log(f"Failed to load code list: {exc}")
        watch_map = {}

    if not watch_map:
        print("[R006 WATCHLIST] No codes loaded", flush=True)
        log("No codes loaded")
        return

    print(f"[R006 WATCHLIST] loaded {len(watch_map)} codes", flush=True)

    register_symbol_names(watch_map)

    log(f"[FEATURE_R008_WATCHLIST] Watchlist source: {watch_file}")

    if ENABLE_NXT_SESSION:
        log("MODE BANNER: REGULAR+NXT_MODE")
    else:
        log("MODE BANNER: REGULAR_ONLY_MODE")

    nxt_map = {code: is_nxt_tradeable(code) for code in watch_map}
    for code, name in watch_map.items():
#        print(f"[R006 WATCH] {code} | {name} | NXT={nxt_map[code]}", flush=True)
        log(f"WATCH | {code} | {name} | NXT={nxt_map[code]}")

    log("Strategy: live price cross over buffered BB middle + Stoch/RSI/Williams confirmation")
    log(
        f"Live-cross filter: BB buffer={LIVE_PRICE_BB_BUFFER_PCT*100:.3f}% | "
        f"confirm polls={LIVE_PRICE_CROSS_CONFIRM_POLLS} | confirm seconds={LIVE_PRICE_CROSS_CONFIRM_SECONDS}"
    )
    log(
        f"Polling: live={LIVE_PRICE_POLL_INTERVAL_SECONDS}s | "
        f"frame refresh every {FRAME_POLL_INTERVAL_SECONDS}s (3min bars) + backfill {FRAME_BACKFILL_SYNC_SECONDS}s | "
        f"buy consecutive confirms={BUY_CONSECUTIVE_CONFIRM_COUNT}"
    )
    log(
        f"ATR model: stop={ATR_STOP_MULTIPLIER:.1f}x | tp={ATR_TAKE_PROFIT_MULTIPLIER:.1f}x | "
        f"trail={TRAILING_STOP_FROM_PEAK*100:.1f}%"
    )
    log(f"Indicator warmup: require bars >= {INDICATOR_WARMUP_BARS} before new entries")
    date_str = now.strftime("%Y%m%d")
    live_state = load_live_state(date_str)
    traded_today: set[str] = set(live_state.get("traded_today") or [])
    effective_env = (env_dv or KIS_ENV_DV).strip() or "real"
    effective_dry = LIVE_DRY_RUN if dry_run is None else bool(dry_run)
    if ENABLE_LIVE_DRY_RUN:
        effective_dry = True
    log(f"CONFIG BANNER | env_dv={effective_env} | dry_run={effective_dry}")
    log(f"CONFIG BANNER | MARKET_DAY_FAIL_CLOSED={MARKET_DAY_FAIL_CLOSED} | SESSION_FORCE_CLOSE_ALL_AT_CUTOFF={SESSION_FORCE_CLOSE_ALL_AT_CUTOFF}")
    log(
        "CONFIG BANNER | aux-trigger guard | "
        f"min_target={AUX_SELL_MIN_REALIZED_TARGET_PCT*100:.2f}% "
        f"slippage_buffer={AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT*100:.2f}%"
    )
    log(
        "CONFIG BANNER | price-lead integrated | "
        f"ENABLE_PRICE_LEAD_BB_BREAKOUT={ENABLE_PRICE_LEAD_BB_BREAKOUT}"
    )
    api = TradingAPI(env_dv=effective_env, dry_run=effective_dry, live_state=live_state)
    _install_shutdown_handlers(api)
    api.live_state["traded_today"] = traded_today
    signal_buy_bar: dict[str, object] = {}
    signal_sell_bar: dict[str, object] = {}
    trailing_sell_confirm_state: dict[str, dict[str, object]] = {}
    buy_confirm_state: dict[str, dict[str, object]] = {}
    post_buy_bb_drop_state: dict[str, dict] = {}
    breakeven_fail_state: dict[str, dict] = {}
    no_trend_exit_state: dict[str, dict] = {}
    live_price_cross_state: dict[str, dict] = {}
    live_price_cache: dict[str, float] = {}
    live_price_cache_at: dict[str, datetime] = {}
    live_price_fail_count: dict[str, int] = {}
    live_price_backoff_until: dict[str, datetime] = {}
    frame_cache: dict[str, pd.DataFrame] = {}
    frame_last_refresh_at: dict[str, datetime] = {}
    realtime_entry_bar_state: dict[str, dict[str, object]] = {}
    liquidation_state: dict = {}
    current_trade_date = now.date()
    loop_consecutive_errors = 0
    last_watchlist_mismatch_log_at: datetime | None = None
    if ENABLE_NXT_SESSION:
        log(
            f"Session-open warmup: new entries blocked for {STARTUP_WARMUP_SECONDS}s "
            f"after each session start ({MORNING_NXT_START:%H:%M}, {REGULAR_START:%H:%M}, {AFTERNOON_NXT_START:%H:%M})"
        )
    else:
        log(
            f"Session-open warmup: new entries blocked for {STARTUP_WARMUP_SECONDS}s "
            f"after regular session start ({REGULAR_START:%H:%M})"
        )

    while True:
        current_dt = datetime.now()
        _ensure_log_date_for(current_dt)

        if current_dt.date() != current_trade_date:
            current_trade_date = current_dt.date()
            date_str = current_dt.strftime("%Y%m%d")
            live_state = load_live_state(date_str)
            _rotate_logging_for_date(date_str)
            live_state = load_live_state(date_str)
            traded_today = set(live_state.get("traded_today") or [])
            api.live_state = live_state
            api.live_state["traded_today"] = traded_today
            api._apply_persisted_position_meta()
            post_buy_bb_drop_state.clear()
            breakeven_fail_state.clear()
            no_trend_exit_state.clear()
            signal_buy_bar.clear()
            signal_sell_bar.clear()
            trailing_sell_confirm_state.clear()
            buy_confirm_state.clear()
            live_price_cross_state.clear()
            live_price_cache.clear()
            live_price_cache_at.clear()
            live_price_fail_count.clear()
            live_price_backoff_until.clear()
            frame_cache.clear()
            frame_last_refresh_at.clear()
            realtime_entry_bar_state.clear()

        is_open_day, market_day_log = get_market_day_status(current_dt)
        if not is_open_day:
            log(market_day_log)
            log("MARKET LOOP STOP | reason=market_closed_day")
            break

        market_end_time = AFTERNOON_NXT_END if ENABLE_NXT_SESSION else REGULAR_END
        if current_dt.time() >= market_end_time:
            run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state, date_str, traded_today)
            log(f"{market_end_time:%H:%M} reached. Stopping.")
            break

        if not is_regular_session(current_dt) and not is_nxt_session(current_dt):
            time.sleep(LIVE_PRICE_POLL_INTERVAL_SECONDS)
            continue

        try:
            # MAIN_LOOP_TICK_START
            if (
                last_watchlist_mismatch_log_at is None
                or (current_dt - last_watchlist_mismatch_log_at).total_seconds() >= WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS
            ):
                _log_account_watchlist_mismatch(api, watch_map)
                last_watchlist_mismatch_log_at = current_dt
            api.sync_positions_from_account(force=False)
            api.refresh_pending_orders(current_dt)
            traded_today = {str(c).zfill(6) for c in (api.live_state.get("traded_today") or [])}
            api.live_state["traded_today"] = traded_today
            run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state, date_str, traded_today)

            # 15:20~15:30 정규장 마감 구간은 종목별 매도 체크 건너뛰고
            # 동시호가 예약 청산 로직(당일 매수 + 수익 구간)만 실행한다.
            if is_regular_call_auction(current_dt):
                time.sleep(LIVE_PRICE_POLL_INTERVAL_SECONDS)
                continue

            for code, name in watch_map.items():
                nxt_tradeable = nxt_map.get(code, False)
                symbol_label = _symbol_log_label(code, name)
                if not can_trade_code_now(current_dt, nxt_tradeable):
                    log(f"  [SKIP] {symbol_label} | can_trade_code_now=False | time={current_dt:%H:%M:%S} nxt={nxt_tradeable}")
                    continue

                cached_frame = frame_cache.get(code)
                last_frame_refresh = frame_last_refresh_at.get(code)
                frame: pd.DataFrame | None = cached_frame

                if should_refresh_3min_frame(current_dt, cached_frame, last_frame_refresh):
                    try:
                        refreshed_frame = fetch_3min_frame(code, current_dt, nxt_tradeable)
                    except Exception as exc:
                        log(f"{code} frame error: {exc}")
                        refreshed_frame = None

                    if refreshed_frame is not None and not refreshed_frame.empty:
                        frame_cache[code] = refreshed_frame
                        frame_last_refresh_at[code] = current_dt
                        frame = refreshed_frame

                if frame is None:
                    log(f"  [SKIP] {symbol_label} | frame=None (fetch failed)")
                    continue
                if len(frame) < INDICATOR_WARMUP_BARS:
                    log(f"  [SKIP] {symbol_label} | bars={len(frame)} < INDICATOR_WARMUP_BARS={INDICATOR_WARMUP_BARS}")
                    continue

                bar_time = frame.index[-1]
                last_closed_bar = pd.Timestamp(current_dt).floor("3min")
                bar_age_sec = max(0.0, (pd.Timestamp(current_dt) - bar_time).total_seconds())
                cur = frame.iloc[-1]
                live_backoff_until = live_price_backoff_until.get(code)
                can_fetch_live = live_backoff_until is None or current_dt >= live_backoff_until
                price_source = "live"
                price = None

                if can_fetch_live:
                    price = fetch_live_price(code, current_dt, nxt_tradeable)
                    if price is None or price <= 0:
                        fail_count = int(live_price_fail_count.get(code, 0)) + 1
                        live_price_fail_count[code] = fail_count
                        backoff_seconds = _live_price_backoff_seconds(fail_count)
                        live_price_backoff_until[code] = current_dt + timedelta(seconds=backoff_seconds)
                        price_source = f"fallback_backoff_{backoff_seconds}s"
                    else:
                        live_price_cache[code] = float(price)
                        live_price_cache_at[code] = current_dt
                        live_price_fail_count[code] = 0
                        live_price_backoff_until.pop(code, None)
                else:
                    price_source = "cached_live(backoff_active)"

                if price is None or price <= 0:
                    cached_live = live_price_cache.get(code)
                    cached_at = live_price_cache_at.get(code)
                    cache_age_sec = (current_dt - cached_at).total_seconds() if isinstance(cached_at, datetime) else None
                    cache_fresh = cache_age_sec is not None and cache_age_sec <= LIVE_PRICE_STALE_TTL_SECONDS

                    if cached_live is not None and cached_live > 0 and cache_fresh:
                        price = float(cached_live)
                        if price_source == "live":
                            price_source = "cached_live"
                    else:
                        price = float(cur["close"])
                        age_text = f"{cache_age_sec:.0f}s" if cache_age_sec is not None else "unknown"
                        price_source = f"bar_close(stale_live={age_text})"

                buy_frame = frame
                intrabar_elapsed_seconds = max(0.0, (pd.Timestamp(current_dt) - pd.Timestamp(current_dt).floor("3min")).total_seconds())
                if ENABLE_INTRABAR_LIVE_ENTRY_FILTER and price is not None and price > 0:
                    try:
                        buy_frame, intrabar_elapsed_seconds = _build_realtime_entry_frame(
                            frame,
                            code,
                            current_dt,
                            float(price),
                            realtime_entry_bar_state,
                        )
                    except Exception as exc:
                        log(f"  [WARN] {symbol_label} | realtime entry-frame build failed: {exc}")
                        buy_frame = frame

                buy_cur = buy_frame.iloc[-1] if buy_frame is not None and not buy_frame.empty else cur

                cross_info = update_live_price_cross_state(
                    live_price_cross_state,
                    code,
                    current_dt,
                    float(price),
                    _num(buy_cur, "BB_MIDDLE"),
                )
                pos = api.get_open_positions().get(code)
                pending = api.get_pending_order(code)

                if pending is not None:
                    pending_side = str(pending.get("side", "")).upper() or "UNKNOWN"
                    pending_qty = int(pending.get("quantity", 0))
                    pending_time = pending.get("submitted_at", current_dt)
                    log(
                        f"  [PENDING] {symbol_label} | side={pending_side} qty={pending_qty} "
                        f"submitted={pending_time:%H:%M:%S} | order_no={pending.get('order_no', '') or 'UNKNOWN'}"
                    )
                    continue

                log(
                    f"  {symbol_label} [CHECK] | bars={len(frame)} live={price:,.0f}  bar_close={float(cur['close']):,.0f} | "
                    f"confirmed_bar={bar_time:%H:%M:%S} cutoff={last_closed_bar:%H:%M:%S} bar_age={bar_age_sec:.0f}s | "
                    f"MA5={_num(cur, 'MA_5'):.1f} BB_MID={_num(cur, 'BB_MIDDLE'):.1f} BB_UP={_num(cur, 'BB_UPPER'):.1f} BB_LW={_num(cur, 'BB_LOWER'):.1f} | "
                    f"CROSS relation={cross_info.get('relation')} upper={float(cross_info.get('upper_trigger', 0.0)):.1f} lower={float(cross_info.get('lower_trigger', 0.0)):.1f} pending={cross_info.get('pending_side')} cnt={cross_info.get('pending_count')} sec={float(cross_info.get('pending_seconds', 0.0)):.0f} signal={cross_info.get('signal')} | "
                    f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                    f"K={_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                    f"WR={_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                    f"MACD={_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} HIST={_num(cur, 'MACD_HIST'):.2f} | "
                    f"ADX={_num(cur, 'ADX'):.1f} +DI={_num(cur, 'DI_PLUS'):.1f} -DI={_num(cur, 'DI_MINUS'):.1f} | "
                    f"VOL={_num(cur, 'volume'):,.0f} VOLMA={_num(cur, 'VOL_MA20'):,.0f} | "
                    f"VWAP={_num(cur, 'VWAP'):,.0f} OBV={_num(cur, 'OBV'):,.0f} OBVMA={_num(cur, 'OBV_MA'):,.0f}"
                )

                if pos is None or pos.get("quantity", 0) <= 0:
                    post_buy_bb_drop_state.pop(code, None)
                    breakeven_fail_state.pop(code, None)
                    no_trend_exit_state.pop(code, None)

                if pos is not None and pos.get("quantity", 0) > 0:
                    if not _is_today_buy_position(code, pos, date_str, traded_today):
                        log(f"  {symbol_label} [HOLD SKIP] | NOT_TODAY_BUY_POSITION")
                        continue
                    buy_confirm_state.pop(code, None)
                    entry_price = float(pos["buy_price"])
                    pnl_pct = (price / entry_price) - 1.0
                    is_startup_position = str(code).zfill(6) in api.startup_position_codes
                    if _is_stale_live_price_source(price_source):
                        log(
                            f"  {symbol_label} [SELL REJECT] | STALE_LIVE_PRICE | "
                            f"source={price_source} ttl={LIVE_PRICE_STALE_TTL_SECONDS}s"
                        )
                        continue

                    # 저가(bar low)가 현재가보다 낮으면 손절 판정에 우선 반영
                    # 확정봉 low를 직접 섞으면 현재가가 회복된 상태에서도 오손절이 날 수 있어
                    # 실시간 손절은 현재가 기준으로 판정한다.
                    _bar_low = float(cur["low"]) if "low" in cur.index and not pd.isna(cur["low"]) else float("nan")
                    atr_val = _num(cur, "ATR")
                    atr_tp_price = float("nan")
                    atr_sl_price = float("nan")
                    atr_tp_pct = float("nan")
                    atr_sl_pct = float("nan")
                    if not pd.isna(atr_val) and atr_val > 0 and entry_price > 0:
                        atr_tp_price = entry_price + float(atr_val) * ATR_TAKE_PROFIT_MULTIPLIER
                        atr_sl_price = entry_price - float(atr_val) * ATR_STOP_MULTIPLIER
                        atr_tp_pct = (atr_tp_price / entry_price) - 1.0
                        atr_sl_pct = (atr_sl_price / entry_price) - 1.0
                    _sl_price = price
                    _pnl_sl = (_sl_price / entry_price) - 1.0

                    pos["current_price"] = price
                    pos["highest_price"] = max(float(pos.get("highest_price", price)), price)
                    highest_price = float(pos["highest_price"])  # TP/트레일링 로직 계산 전에 갱신값 반영
                    peak_pnl_pct = (highest_price / entry_price) - 1.0 if highest_price > 0 and entry_price > 0 else 0.0
                    profit_giveback = peak_pnl_pct - pnl_pct

                    # ── Aggressive Exit Plan (+1/+2, signal exits, -0.8% SL) ───────
                    k_now = _num(cur, "STOCH_K")
                    d_now = _num(cur, "STOCH_D")
                    hist_now = _num(cur, "MACD_HIST")
                    hist_prev = _num(frame.iloc[-2], "MACD_HIST")
                    hist_prev2 = _num(frame.iloc[-3], "MACD_HIST") if len(frame) >= 3 else float("nan")
                    adx_now = _num(cur, "ADX")
                    di_plus_now = _num(cur, "DI_PLUS")
                    di_minus_now = _num(cur, "DI_MINUS")
                    # 강한 상승 추세: ADX > 28 이고 +DI > -DI 이면 스토캐스틱 K<D 매도 신호 무시
                    _sig_buy_time_raw = pos.get("buy_time")
                    _sig_held_seconds = (current_dt - _sig_buy_time_raw).total_seconds() if isinstance(_sig_buy_time_raw, datetime) else 0.0
                    _signal_min_hold_seconds = 300.0  # signal exit min hold: 5 min
                    _strong_uptrend = (
                        not pd.isna(adx_now) and adx_now > 28
                        and not pd.isna(di_plus_now) and not pd.isna(di_minus_now)
                        and di_plus_now > di_minus_now
                    )

                    # Hard stop-loss first
                    if pnl_pct <= -0.008:
                        reason_hard_sl = "HARD_STOP_LOSS_0.8PCT"
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_hard_sl} | "
                            f"price={price:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}%"
                        )
                        trailing_sell_confirm_state.pop(code, None)
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_hard_sl, nxt_tradeable, price=price, code_name=name, market_order=True):
                            log(f"  [SELL EXECUTED] {code} | {reason_hard_sl} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                    # +2.0% full take profit
                    if pnl_pct >= 0.020:
                        reason_tp2 = "TP2_FULL_2.0PCT"
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_tp2} | "
                            f"price={price:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}%"
                        )
                        trailing_sell_confirm_state.pop(code, None)
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_tp2, nxt_tradeable, price=price, code_name=name):
                            log(f"  [SELL EXECUTED] {code} | {reason_tp2} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                    # +1.0% one-time 50% partial take profit
                    if (not bool(pos.get("tp1_done", False))) and pnl_pct >= 0.010:
                        partial_qty = max(1, int(int(pos["quantity"]) * 0.5))
                        partial_qty = min(partial_qty, int(pos["quantity"]))
                        reason_tp1 = "TP1_PARTIAL_50PCT_1.0PCT"
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_tp1} | "
                            f"qty={partial_qty}/{int(pos['quantity'])} price={price:,.0f} pnl={pnl_pct*100:.2f}%"
                        )
                        if api.place_sell_order(code, partial_qty, current_dt, reason_tp1, nxt_tradeable, price=price, code_name=name):
                            pos["tp1_done"] = True
                            api._record_position_meta(code, pos)
                            api.persist_live_state(date_str=date_str)
                            log(f"  [SELL EXECUTED] {code} | {reason_tp1} | qty={partial_qty} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                    # Signal-based full exits
                    if not any(pd.isna(v) for v in (k_now, d_now)) and k_now < d_now:
                        if _strong_uptrend or _sig_held_seconds < _signal_min_hold_seconds:
                            log(
                                f"  [SELL SKIP] {code} | STOCH_K_LT_D suppressed by strong uptrend | "
                                f"K={k_now:.1f} D={d_now:.1f} ADX={adx_now:.1f} +DI={di_plus_now:.1f} -DI={di_minus_now:.1f}"
                            )
                        else:
                            reason_sig_kd = "SIGNAL_EXIT_STOCH_K_LT_D"
                            log(
                                f"  [SELL TRIGGER] {code} | {reason_sig_kd} | "
                                f"K={k_now:.1f} D={d_now:.1f} pnl={pnl_pct*100:.2f}%"
                            )
                            trailing_sell_confirm_state.pop(code, None)
                            if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_sig_kd, nxt_tradeable, price=price, code_name=name):
                                log(f"  [SELL EXECUTED] {code} | {reason_sig_kd} | qty={pos['quantity']} price={price:,.0f}")
                            signal_sell_bar[code] = bar_time
                            continue
                    if (not any(pd.isna(v) for v in (hist_now, hist_prev, hist_prev2))
                            and hist_now < hist_prev < hist_prev2
                            and not _strong_uptrend
                            and _sig_held_seconds >= _signal_min_hold_seconds):
                        reason_sig_macd = "SIGNAL_EXIT_MACD_HIST_DOWN_2BARS"
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_sig_macd} | "
                            f"HIST={hist_prev2:.3f}->{hist_prev:.3f}->{hist_now:.3f} pnl={pnl_pct*100:.2f}%"
                        )
                        trailing_sell_confirm_state.pop(code, None)
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_sig_macd, nxt_tradeable, price=price, code_name=name):
                            log(f"  [SELL EXECUTED] {code} | {reason_sig_macd} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                    if not pd.isna(atr_tp_pct) and pnl_pct >= atr_tp_pct:
                        if ENABLE_TP_EXTENSION_TRAILING:
                            # TP 도달 시 즉시 익절 대신 고점 트레일링 모드로 전환
                            log(
                                f"  [TP_EXTENSION] {code} | pnl={pnl_pct*100:.2f}% >= ATR_TP {atr_tp_pct*100:.2f}% | "
                                f"고점 트레일링 모드 전환 (trail={TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%) | "
                                f"price={price:,.0f} peak={highest_price:,.0f} atr={float(atr_val):.2f}"
                            )
                        else:
                            reason_tp = f"ATR_TAKE_PROFIT_{ATR_TAKE_PROFIT_MULTIPLIER:.1f}x"
                            log(
                                f"  [SELL TRIGGER] {code} | {reason_tp} | price={price:,.0f} entry={entry_price:,.0f} "
                                f"pnl={pnl_pct*100:.2f}% atr={float(atr_val):.2f} tp={atr_tp_price:,.0f}"
                            )
                            if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_tp, nxt_tradeable, price=price, code_name=name):
                                log(f"  [SELL EXECUTED] {code} | {reason_tp} | qty={pos['quantity']} price={price:,.0f}")
                            signal_sell_bar[code] = bar_time
                            continue

                    # ── POST-BUY ENTRY DROP GUARD ─────────────────────────────────────
                    _buy_time_raw = pos.get("buy_time")
                    _buy_time = _buy_time_raw if isinstance(_buy_time_raw, datetime) else current_dt
                    _buy_token = _buy_time_raw if isinstance(_buy_time_raw, datetime) else (
                        f"unknown_buy_time:{code}:{pos.get('entry_price', 0)}:{int(pos.get('quantity', 0))}"
                    )
                    _held_for_guard = (current_dt - _buy_time).total_seconds()
                    if _held_for_guard <= POST_BUY_BB_DROP_ARMED_SECONDS:
                        _drop_hold_seconds = update_timed_condition_state(
                            post_buy_bb_drop_state,
                            code,
                            _buy_token,
                            current_dt,
                            price < entry_price * (1.0 - POST_BUY_BB_DROP_PCT),
                        )
                        if _drop_hold_seconds >= POST_BUY_DROP_CONFIRM_SECONDS:
                            _drop_pct_guard = (price / entry_price - 1.0) * 100.0
                            reason_bbdrop = f"POST_BUY_ENTRY_DROP_{POST_BUY_BB_DROP_PCT*100:.1f}pct_{POST_BUY_DROP_CONFIRM_SECONDS:.0f}s"
                            log(
                                f"  [SELL TRIGGER] {code} | {reason_bbdrop} | "
                                f"held={_held_for_guard:.0f}s price={price:,.0f} entry={entry_price:,.0f} "
                                f"drop={_drop_pct_guard:.2f}% hold={_drop_hold_seconds:.0f}s pnl={pnl_pct*100:.2f}%"
                            )
                            post_buy_bb_drop_state.pop(code, None)
                            breakeven_fail_state.pop(code, None)
                            no_trend_exit_state.pop(code, None)
                            trailing_sell_confirm_state.pop(code, None)
                            if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_bbdrop, nxt_tradeable, price=price, code_name=name):
                                log(f"  [SELL EXECUTED] {code} | {reason_bbdrop} | qty={pos['quantity']} price={price:,.0f}")
                            signal_sell_bar[code] = bar_time
                            continue
                    else:
                        post_buy_bb_drop_state.pop(code, None)
                    # ── END POST-BUY ENTRY DROP GUARD ─────────────────────────────────

                    # ── BREAKEVEN FAILURE GUARD ────────────────────────────────────────
                    if not is_startup_position:
                        _breakeven_hold_seconds = update_timed_condition_state(
                            breakeven_fail_state,
                            code,
                            _buy_token,
                            current_dt,
                            peak_pnl_pct >= BREAKEVEN_FAIL_ARM_PNL
                            and pnl_pct < 0
                            and profit_giveback >= BREAKEVEN_FAIL_GIVEBACK_PCT,
                        )
                        if _breakeven_hold_seconds >= BREAKEVEN_FAIL_CONFIRM_SECONDS:
                            reason_breakeven = (
                                f"BREAKEVEN_FAIL_peak{BREAKEVEN_FAIL_ARM_PNL*100:.1f}_"
                                f"giveback{BREAKEVEN_FAIL_GIVEBACK_PCT*100:.2f}_{BREAKEVEN_FAIL_CONFIRM_SECONDS:.0f}s"
                            )
                            log(
                                f"  [SELL TRIGGER] {code} | {reason_breakeven} | held={_held_for_guard:.0f}s "
                                f"price={price:,.0f} entry={entry_price:,.0f} peak={highest_price:,.0f} "
                                f"peak_pnl={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                            )
                            post_buy_bb_drop_state.pop(code, None)
                            breakeven_fail_state.pop(code, None)
                            no_trend_exit_state.pop(code, None)
                            trailing_sell_confirm_state.pop(code, None)
                            if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_breakeven, nxt_tradeable, price=price, code_name=name):
                                log(f"  [SELL EXECUTED] {code} | {reason_breakeven} | qty={pos['quantity']} price={price:,.0f}")
                            signal_sell_bar[code] = bar_time
                            continue
                    else:
                        breakeven_fail_state.pop(code, None)
                    # ── END BREAKEVEN FAILURE GUARD ───────────────────────────────────

                    # ── NO-TREND TIME EXIT ────────────────────────────────────────────
                    _bb_mid_guard = _num(cur, "BB_MIDDLE")
                    _no_trend_condition = (
                        _held_for_guard >= NO_TREND_EXIT_ARM_SECONDS
                        and peak_pnl_pct <= NO_TREND_EXIT_MAX_PEAK_PNL
                        and pnl_pct <= NO_TREND_EXIT_MIN_PNL
                        and _bb_mid_guard > 0
                        and price < _bb_mid_guard
                    )
                    _no_trend_hold_seconds = update_timed_condition_state(
                        no_trend_exit_state,
                        code,
                        _buy_token,
                        current_dt,
                        _no_trend_condition,
                    )
                    if _no_trend_hold_seconds >= NO_TREND_EXIT_CONFIRM_SECONDS:
                        reason_no_trend = (
                            f"NO_TREND_EXIT_{NO_TREND_EXIT_ARM_SECONDS/60:.0f}m_"
                            f"peakLT{NO_TREND_EXIT_MAX_PEAK_PNL*100:.1f}_{NO_TREND_EXIT_CONFIRM_SECONDS:.0f}s"
                        )
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_no_trend} | held={_held_for_guard:.0f}s "
                            f"price={price:,.0f} bb_mid={_bb_mid_guard:,.1f} pnl={pnl_pct*100:.2f}% "
                            f"peak_pnl={peak_pnl_pct*100:.2f}% hold={_no_trend_hold_seconds:.0f}s"
                        )
                        post_buy_bb_drop_state.pop(code, None)
                        breakeven_fail_state.pop(code, None)
                        no_trend_exit_state.pop(code, None)
                        trailing_sell_confirm_state.pop(code, None)
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_no_trend, nxt_tradeable, price=price, code_name=name):
                            log(f"  [SELL EXECUTED] {code} | {reason_no_trend} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue
                    # ── END NO-TREND TIME EXIT ────────────────────────────────────────

                    _held_sl = (current_dt - _buy_time).total_seconds()
                    if not pd.isna(atr_sl_pct) and _pnl_sl <= atr_sl_pct:
                        reason_sl = f"ATR_STOP_LOSS_{ATR_STOP_MULTIPLIER:.1f}x"
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_sl} | held={_held_sl:.0f}s price={price:,.0f} "
                            f"bar_low={_bar_low:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}% "
                            f"sl_pnl={_pnl_sl*100:.2f}% atr={float(atr_val):.2f} sl={atr_sl_price:,.0f}"
                        )
                        trailing_sell_confirm_state.pop(code, None)
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_sl, nxt_tradeable, price=price, code_name=name, market_order=True):
                            log(f"  [SELL EXECUTED] {code} | {reason_sl} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                    if highest_price > 0 and entry_price > 0:
                        # Entry-anchored trailing stop:
                        # 1) First, peak must move into profit zone.
                        # 2) Then, sell only if profit retraces by trailing width.
                        # 3) Never trigger trailing stop while current pnl is non-positive.
                        # TP 도달 구간(peak >= TP)에서 이익이 1% 이내로 줄면 트레일링 적용
                        current_pnl_pct = (price / entry_price) - 1.0
                        profit_giveback = peak_pnl_pct - current_pnl_pct
                        if ENABLE_TP_EXTENSION_TRAILING and not pd.isna(atr_tp_pct) and peak_pnl_pct >= atr_tp_pct:
                            trail_threshold = TP_EXTENSION_TRAIL_FROM_PEAK
                            reason_ts = f"TP_EXTENSION_TRAIL_{TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%"
                        else:
                            trail_threshold = TRAILING_STOP_FROM_PEAK
                            reason_ts = f"TRAILING_STOP_GIVEBACK_{TRAILING_STOP_FROM_PEAK*100:.1f}%"
                        trailing_condition = peak_pnl_pct > 0 and current_pnl_pct > 0 and profit_giveback >= trail_threshold

                        pending_state = trailing_sell_confirm_state.get(code)
                        if reason_ts.startswith("TRAILING_STOP_GIVEBACK_"):
                            # Clear pending if retrace condition has recovered.
                            if not trailing_condition and pending_state is not None:
                                trailing_sell_confirm_state.pop(code, None)
                                log(
                                    f"  [SELL HOLD CANCEL] {code} | trailing recovered before confirm | "
                                    f"pnl={current_pnl_pct*100:.2f}% peak_pnl={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                                )

                        if trailing_condition:
                            if reason_ts.startswith("TRAILING_STOP_GIVEBACK_"):
                                # First hit: defer to next 3-minute bar confirmation.
                                if pending_state is None:
                                    trailing_sell_confirm_state[code] = {
                                        "trigger_bar_time": bar_time,
                                        "triggered_at": current_dt,
                                        "reason": reason_ts,
                                    }
                                    log(
                                        f"  [SELL HOLD] {code} | {reason_ts} first hit, wait next 3m bar confirm | "
                                        f"bar={bar_time:%H:%M:%S} pnl={current_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                                    )
                                    continue

                                # Still same bar: keep waiting.
                                if pending_state.get("trigger_bar_time") == bar_time:
                                    continue

                            log(
                                f"  [SELL TRIGGER] {code} | {reason_ts} | "
                                f"price={price:,.0f} entry={entry_price:,.0f} peak={highest_price:,.0f} | "
                                f"pnl={current_pnl_pct*100:.2f}% peak_pnl={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                            )
                            trailing_sell_confirm_state.pop(code, None)
                            if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_ts, nxt_tradeable, price=price, code_name=name):
                                log(f"  [SELL EXECUTED] {code} | {reason_ts} | qty={pos['quantity']} price={price:,.0f}")
                            signal_sell_bar[code] = bar_time
                            continue

                    if signal_sell_bar.get(code) == bar_time:
                        continue

                    prev_bar = frame.iloc[-2]
                    sell_ok, sell_reason = check_sell_condition(frame, pnl_pct, price, cross_info)
                    if sell_ok:
                        aux_score = _extract_aux_score_from_reason(sell_reason)
                        if aux_score is not None:
                            aux_base_min = _aux_min_pnl_for_score(aux_score)
                            if aux_base_min is not None:
                                aux_required_pnl = max(
                                    AUX_SELL_MIN_REALIZED_TARGET_PCT,
                                    aux_base_min + AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT,
                                )
                                if pnl_pct < aux_required_pnl:
                                    log(
                                        f"  [SELL HOLD] {code} | AUX_TRIGGER_BUFFER_BLOCK "
                                        f"score={aux_score} pnl={pnl_pct*100:.2f}% "
                                        f"required>={aux_required_pnl*100:.2f}% "
                                        f"(base={aux_base_min*100:.2f}%+buffer={AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT*100:.2f}%)"
                                    )
                                    continue
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, sell_reason, nxt_tradeable, price=price, code_name=name):
                            log(
                                f"  [SELL EVAL] {code} | OK {sell_reason} | {current_dt:%H:%M:%S} | "
                                f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}->{_num(cur, 'BB_MIDDLE'):.1f} | "
                                f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                                f"K={_num(prev_bar, 'STOCH_K'):.1f}->{_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                                f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}->{_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                                f"MACD {_num(prev_bar, 'MACD'):.2f}->{_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} | "
                                f"ADX={_num(cur, 'ADX'):.1f} | pnl={pnl_pct*100:.2f}% peak={highest_price:,.0f}"
                            )
                            log(f"  [SELL EXECUTED] {code} | {sell_reason} | qty={pos['quantity']} price={price:,.0f}")
                            signal_sell_bar[code] = bar_time
                    elif (
                        sell_reason.startswith("AUX_BLOCKED")
                        or sell_reason.startswith("LIVE_PRICE_BB_DOWN_CROSS_WEAK_SCORE")
                        or sell_reason.startswith("BOX_RANGE_HOLD")
                    ):
                        log(f"  [SELL HOLD] {code} | {sell_reason}")

                else:
                    trailing_sell_confirm_state.pop(code, None)
                    if not is_new_entry_allowed(current_dt, nxt_tradeable):
                        continue
                    session_open_dt = get_session_open_datetime(current_dt, nxt_tradeable)
                    if session_open_dt is not None:
                        _warmup_elapsed = (current_dt - session_open_dt).total_seconds()
                        if _warmup_elapsed < STARTUP_WARMUP_SECONDS:
                            log(
                                f"  {symbol_label} [BUY REJECT] | SESSION_OPEN_WARMUP | "
                                f"elapsed={_warmup_elapsed:.0f}s / {STARTUP_WARMUP_SECONDS}s | "
                                f"session_open={session_open_dt:%H:%M:%S}"
                            )
                            continue
                    if api._in_cooldown(code, current_dt):
                        continue
                    if signal_buy_bar.get(code) == bar_time:
                        continue

                    prev_bar = buy_frame.iloc[-2]
                    norm_code = str(code).zfill(6)
                    if api.has_buy_exposure(norm_code):
                        log(f"  {symbol_label} [BUY SKIP] | ALREADY_TRADED_TODAY_UNTIL_SELL")
                        continue

                    buy_ok, buy_reason = check_buy_condition(
                        buy_frame,
                        current_dt,
                        price,
                        cross_info,
                        intrabar_elapsed_seconds=intrabar_elapsed_seconds,
                    )

                    if not buy_ok:
                        buy_confirm_state.pop(code, None)
                        detail = _buy_reject_detail(
                            buy_reason,
                            buy_frame.iloc[-1],
                            prev_bar,
                            live_price=price,
                            cross_info=cross_info,
                            frame=buy_frame,
                        )
                        log(f"  {symbol_label} [BUY REJECT] | {detail}")
                        continue

                    pattern_ok, pattern_reason = _passes_loss_pattern_buy_filter(buy_frame, buy_reason, price)
                    if not pattern_ok:
                        buy_confirm_state.pop(code, None)
                        log(
                            f"  {symbol_label} [BUY REJECT] | {pattern_reason} | "
                            f"reason={buy_reason} live={price:,.0f} "
                            f"bb_mid={_num(buy_frame.iloc[-1], 'BB_MIDDLE'):.1f} bar_close={_num(buy_frame.iloc[-1], 'close'):,.0f} ma5={_num(buy_frame.iloc[-1], 'MA_5'):.1f}"
                        )
                        continue

                    prev_close = fetch_prev_close(code, current_dt, nxt_tradeable)
                    rise_ratio = _rise_from_prev_close(price, float(prev_close or 0.0))
                    if (
                        MAX_BUY_RISE_PCT_FROM_PREV_CLOSE > 0
                        and rise_ratio is not None
                        and rise_ratio >= MAX_BUY_RISE_PCT_FROM_PREV_CLOSE
                    ):
                        buy_confirm_state.pop(code, None)
                        log(
                            f"  {symbol_label} [BUY REJECT] | EXCESSIVE_RISE_FROM_PREV_CLOSE_"
                            f"{rise_ratio*100:.2f}%_GE_{MAX_BUY_RISE_PCT_FROM_PREV_CLOSE*100:.2f}% | "
                            f"prev_close={float(prev_close):,.0f} live={price:,.0f}"
                        )
                        continue

                    confirm_state = buy_confirm_state.get(code)
                    _last_confirmed = confirm_state.get("confirmed_at") if confirm_state else None
                    _elapsed = (
                        (current_dt - _last_confirmed).total_seconds()
                        if isinstance(_last_confirmed, datetime) else None
                    )
                    if _elapsed is None or _elapsed > POLL_INTERVAL_SECONDS * 2:
                        confirm_count = 1
                    else:
                        confirm_count = int(confirm_state.get("count", 0)) + 1

                    buy_confirm_state[code] = {"confirmed_at": current_dt, "count": confirm_count, "bar_time": bar_time}
                    if confirm_count < BUY_CONSECUTIVE_CONFIRM_COUNT:
                        log(
                                f"  {symbol_label} [BUY HOLD] | reason=WAIT_NEXT_POLL_CONFIRM | "
                                f"count={confirm_count}/{BUY_CONSECUTIVE_CONFIRM_COUNT} | "
                                f"live={price:,.0f} bb_mid={_num(cur, 'BB_MIDDLE'):.1f} bar={bar_time:%H:%M:%S}"
                        )
                        continue

                    qty = api.get_affordable_buy_qty(code, price, current_dt, nxt_tradeable)
                    if qty <= 0:
                        log(f"  {symbol_label} [BUY REJECT] | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price={price:,.0f}")
                        continue

                    if _is_stale_live_price_source(price_source):
                        log(
                            f"  {symbol_label} [BUY REJECT] | STALE_LIVE_PRICE | "
                            f"source={price_source} ttl={LIVE_PRICE_STALE_TTL_SECONDS}s"
                        )
                        continue

                    session = classify_buy_session(current_dt)
                    latest_vol = _num(buy_frame.iloc[-1], "volume")
                    latest_vol_ma = _num(buy_frame.iloc[-1], "VOL_MA20")
                    latest_vol_ratio = (
                        latest_vol / latest_vol_ma
                        if not any(pd.isna(v) for v in (latest_vol, latest_vol_ma)) and latest_vol_ma > 0
                        else float("nan")
                    )
                    buy_detail = (
                        f"reason={buy_reason} signal={cross_info.get('signal')} "
                        f"live={price:,.0f} bb_mid={_num(buy_frame.iloc[-1], 'BB_MIDDLE'):.1f} "
                        f"bar_close={_num(buy_frame.iloc[-1], 'close'):,.0f} ma5={_num(buy_frame.iloc[-1], 'MA_5'):.1f} "
                        f"VOL={latest_vol:,.0f} VOLMA={latest_vol_ma:,.0f} vol_ratio={latest_vol_ratio:.4f}"
                    )
                    traded_today.add(norm_code)
                    api.live_state["traded_today"] = traded_today
                    signal_buy_bar[code] = bar_time

                    cur_bar_open = _num(buy_frame.iloc[-1], "open")
                    cur_bar_close = _num(buy_frame.iloc[-1], "close")
                    if not any(pd.isna(v) for v in (cur_bar_open, cur_bar_close)) and cur_bar_open > 0:
                        if cur_bar_close <= cur_bar_open:
                            log(f"  {symbol_label} [BUY REJECT] | BEARISH_BAR | open={cur_bar_open:,.0f} close={cur_bar_close:,.0f}")
                            traded_today.discard(norm_code)
                            api.live_state["traded_today"] = traded_today
                            signal_buy_bar.pop(code, None)
                            continue
                    if api.place_buy_order(code, price, qty, current_dt, nxt_tradeable, session, buy_detail=buy_detail, code_name=name):
                        log(
                            f"  {symbol_label} [BUY EVAL] | OK {buy_reason} | {current_dt:%H:%M:%S} | "
                            f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}->{_num(cur, 'BB_MIDDLE'):.1f} | "
                            f"RSI={_num(buy_frame.iloc[-1], 'RSI'):.1f} SIG={_num(buy_frame.iloc[-1], 'RSI_SIGNAL'):.1f} | "
                            f"K={_num(prev_bar, 'STOCH_K'):.1f}->{_num(buy_frame.iloc[-1], 'STOCH_K'):.1f} D={_num(buy_frame.iloc[-1], 'STOCH_D'):.1f} | "
                            f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}->{_num(buy_frame.iloc[-1], 'WILLIAMS_R'):.1f} WD={_num(buy_frame.iloc[-1], 'WILLIAMS_D'):.1f} | "
                            f"MACD {_num(prev_bar, 'MACD'):.2f}->{_num(buy_frame.iloc[-1], 'MACD'):.2f} SIG={_num(buy_frame.iloc[-1], 'MACD_SIGNAL'):.2f} | "
                            f"ADX={_num(buy_frame.iloc[-1], 'ADX'):.1f} +DI={_num(buy_frame.iloc[-1], 'DI_PLUS'):.1f} -DI={_num(buy_frame.iloc[-1], 'DI_MINUS'):.1f} | "
                            f"VOL={_num(buy_frame.iloc[-1], 'volume'):,.0f} VOLMA={_num(buy_frame.iloc[-1], 'VOL_MA20'):,.0f} | "
                            f"VWAP={_num(buy_frame.iloc[-1], 'VWAP'):,.0f} OBV={_num(buy_frame.iloc[-1], 'OBV'):,.0f} OBVMA={_num(buy_frame.iloc[-1], 'OBV_MA'):,.0f}"
                        )
                        log(f"  {symbol_label} [BUY EXECUTED] | {buy_reason} | qty={qty} price={price:,.0f} session={session}")
                        buy_confirm_state.pop(code, None)
                        log("=" * 110)
                    else:
                        traded_today.discard(norm_code)
                        api.live_state["traded_today"] = traded_today
                        signal_buy_bar.pop(code, None)

            api.maybe_persist_live_state_interval(current_dt, current_dt.strftime("%Y%m%d"))
            loop_consecutive_errors = 0
        except KeyboardInterrupt:
            _log_pending_orders_on_shutdown(api)
            try:
                api.persist_live_state(date_str=current_dt.strftime("%Y%m%d"))
            except Exception:
                pass
            log("MAIN LOOP STOP | reason=keyboard_interrupt")
            break
        except Exception as exc:
            loop_consecutive_errors += 1
            log(f"MAIN LOOP ERROR ({loop_consecutive_errors}/{MAIN_LOOP_MAX_CONSECUTIVE_ERRORS}): {exc}")
            if loop_consecutive_errors >= MAIN_LOOP_MAX_CONSECUTIVE_ERRORS:
                _log_pending_orders_on_shutdown(api)
                try:
                    api.persist_live_state(date_str=current_dt.strftime("%Y%m%d"))
                except Exception:
                    pass
                log("MAIN LOOP STOP | reason=max_consecutive_errors")
                break

        time.sleep(LIVE_PRICE_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    args = _parse_args()
    run(target_date=args.date, env_dv=args.env_dv, dry_run=args.dry_run, watchlist_source=args.watchlist_source)


