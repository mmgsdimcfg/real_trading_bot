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

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-05-10] type=docs owner=copilot
    summary: added standardized file header and expandable update-log format.
    impact: live
    compatibility: backward-compatible

Note: This script cannot guarantee profit. Always paper-test before live trading.
"""

from __future__ import annotations

import argparse
import collections
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
    MA_PERIOD,
    MACD_FAST,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW,
    MAX_ORDER_AMOUNT_KRW,
    MIN_BARS_REQUIRED,
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
)
from r005_strategy_core_shared import (
    R76StrategyConfig,
    check_buy_condition as shared_check_buy_condition,
    check_sell_condition as shared_check_sell_condition,
    update_timed_condition_state,
    update_live_price_cross_state as shared_update_live_price_cross_state,
)

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
)

# 3-minute frame refresh controls:
# - Keep polling live price every loop.
# - Refresh OHLCV frame only when a new 3-minute bar can be finalized,
#   and occasionally force a backfill sync for safety.
FRAME_REFRESH_SETTLE_DELAY_SECONDS = 3
FRAME_BACKFILL_SYNC_SECONDS = 600

# Live-price polling / order trigger controls.
LIVE_PRICE_POLL_INTERVAL_SECONDS = 10
BUY_CONSECUTIVE_CONFIRM_COUNT = 2
ORDER_STATUS_POLL_INTERVAL_SECONDS = 15

# Live warmup:
# Use only a minimal closed-bar requirement and let the 10-second live-price
# checks drive actual entries/exits.
INDICATOR_WARMUP_BARS = MIN_BARS_REQUIRED

# Live-price fetch backoff controls.
LIVE_PRICE_BACKOFF_BASE_SECONDS = 5
LIVE_PRICE_BACKOFF_MAX_SECONDS = 60
LIVE_PRICE_STALE_TTL_SECONDS = 20

# Pending-order status query backoff controls.
PENDING_STATUS_BACKOFF_MAX_SECONDS = 120

# Market-day status cache (holiday API call reduction).
_MARKET_DAY_STATUS_CACHE: dict[str, tuple[bool, str]] = {}

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
log_dir = current_dir / "logs"
log_dir.mkdir(exist_ok=True)
log_date_str = datetime.now().strftime("%Y%m%d")
log_date_dir = log_dir / log_date_str
log_date_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_stem = Path(__file__).stem
log_filename = f"{timestamp}_{script_stem}.log"
trade_log_filename = f"{timestamp}_{script_stem}_buy_sell.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# domestic_stock_functions / inquire_time_itemchartprice 관련 라이브러리 로그 억제
# 출력되는 "Data fetch complete.", "Call Next page...", "Max recursive depth reached." 메시지를 필터링한다.
# 루트 로거의 INFO 노이즈를 줄이고, 모듈 로그는 WARNING 이상만 남긴다.
logging.getLogger("domestic_stock_functions").setLevel(logging.WARNING)
logging.getLogger("inquire_time_itemchartprice").setLevel(logging.WARNING)
# 일부 모듈이 module-level logging.info/warning 으로 루트 로거를 직접 사용하는 경우
# 루트 로거 자체에서 메시지들을 레벨로 필터링할 필요가 있으므로 아래 필터 추가
class _SuppressLibLogs(logging.Filter):
    _SUPPRESS = frozenset(["Data fetch complete.", "Call Next page...", "Max recursive depth reached."])
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in self._SUPPRESS

_suppress_filter = _SuppressLibLogs()
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_suppress_filter)


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
        label = f"{log_date_str}_{code}({symbol_name})" if symbol_name else f"{log_date_str}_{code}"
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


def log(msg: str) -> None:
    logger.info(msg)


trade_logger = logging.getLogger("trade_events")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False

_trade_handler = logging.FileHandler(log_dir / trade_log_filename, encoding="utf-8")
_trade_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
trade_logger.handlers.clear()
trade_logger.addHandler(_trade_handler)

_symbol_general_handler = _PerSymbolFileHandler(log_date_dir, buy_sell=False)
_symbol_general_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_symbol_general_handler)

_symbol_trade_handler = _PerSymbolFileHandler(log_date_dir, buy_sell=True)
_symbol_trade_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
trade_logger.addHandler(_symbol_trade_handler)


def log_trade(msg: str) -> None:
    trade_logger.info(msg)


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
        code = parts[0].zfill(6)
        name = parts[1] if len(parts) >= 2 and parts[1] else code
        result[code] = name

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R76 real trading runner")
    parser.add_argument("--date", type=str, help="Watchlist date (YYYYMMDD). Use data/YYYYMMDD/picks.txt first")
    return parser.parse_args()


def _resolve_watchlist_file(target_date: str | None) -> Path:
    if not target_date:
        return TODAY_CODE_FILE

    picks_file = DATA_DIR / target_date / "picks.txt"
    if picks_file.exists():
        return picks_file

    log(f"WARNING: picks file not found for --date {target_date}: {picks_file}. Fallback to {TODAY_CODE_FILE}")
    return TODAY_CODE_FILE


# ---------------------------------------------------------------------------
# 주문 결과 파싱
# ---------------------------------------------------------------------------

def _order_succeeded(result) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        rt_cd = result.get("rt_cd")
        return str(rt_cd).strip() == "0" if rt_cd is not None else True
    try:
        if hasattr(result, "columns") and "rt_cd" in result.columns:
            return str(result.iloc[0]["rt_cd"]).strip() == "0"
        empty = getattr(result, "empty", None)
        if empty is not None:
            return not bool(empty)
    except Exception:
        pass
    return True


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


def _extract_order_time(result) -> str:
    value = _extract_order_value(result, ("ord_tmd", "ord_time", "order_time"))
    text = str(value).strip() if value is not None else ""
    return text


def _format_pending_pnl(reason_prefix: str, pnl_pct: float) -> str:
    return f"{reason_prefix}_{pnl_pct * 100:.2f}%"


def _session_exit_plan(reason_prefix: str, pnl_pct: float) -> tuple[str, str]:
    if pnl_pct > 0:
        return "sell", _format_pending_pnl(f"{reason_prefix}_PROFIT_CLOSE", pnl_pct)
    if pnl_pct >= STOP_LOSS_PERCENT:
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


def _log_trade_event_banner(event: str, code: str, qty: int, price: float, detail: str = "", code_name: str = "") -> None:
    """Prints a high-visibility trade event block to both console and file logs."""
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
        result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday | reason=chk_holiday_unavailable")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    try:
        df = holiday_fn(bass_dt=now.strftime("%Y%m%d"))
    except Exception as exc:
        result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=chk_holiday_failed:{exc}")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    if df is None or df.empty:
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
    if not ENABLE_NXT_SESSION:
        return False
    if is_nxt_session(now) and nxt_tradeable:
        return now.time() < AFTERNOON_NXT_NEW_ENTRY_CUTOFF or now.time() <= MORNING_NXT_END
    return False


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
                env_dv="real",
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
                env_dv="real",
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


def should_refresh_3min_frame(
    now: datetime,
    cached_frame: pd.DataFrame | None,
    last_refresh_at: datetime | None,
) -> bool:
    if cached_frame is None or cached_frame.empty:
        return True

    now_ts = pd.Timestamp(now)
    current_closed_bar = now_ts.floor("3min")
    cached_last_bar = pd.Timestamp(cached_frame.index[-1])

    # Refresh when a new closed 3-minute bar should be available.
    if cached_last_bar < current_closed_bar and now.second >= FRAME_REFRESH_SETTLE_DELAY_SECONDS:
        return True

    # Periodic backfill refresh to recover from missed updates or API glitches.
    if last_refresh_at is None:
        return True
    if (now - last_refresh_at).total_seconds() >= FRAME_BACKFILL_SYNC_SECONDS:
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


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
    out["VOL_MA20"] = out["volume"].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()

    out["BB_MIDDLE"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).mean()
    out["BB_STD"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).std()
    out["BB_UPPER"] = out["BB_MIDDLE"] + out["BB_STD"] * BB_STD_MULTIPLIER
    out["BB_LOWER"] = out["BB_MIDDLE"] - out["BB_STD"] * BB_STD_MULTIPLIER

    # RSI - Wilder's smoothing (EWM alpha=1/period, 단순이동평균보다 정확)
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

    # MACD - 단기 모멘텀 방향성(MA 돌파 전략 핵심 확인 지표)
    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    # ADX / DI - 추세 강도 (Wilder's smoothing, 보조지표 신뢰도)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
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

    # VWAP - 당일 누적 거래량 가중 평균가 (세션 시작 시 누적값 리셋)
    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))

    # OBV - 거래량 방향성(급등 방향, 가격 추세 확인)
    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


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
    label = f"{code}({name})"
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
    prev_ma5 = _num(prev, "MA_5");  cur_ma5  = _num(cur,  "MA_5")
    prev_bb  = _num(prev, "BB_MIDDLE"); cur_bb = _num(cur, "BB_MIDDLE")
    snapshot = _buy_condition_snapshot(cur, prev, live_price=live_price, cross_info=cross_info, frame=frame)

    if buy_reason == "NO_LIVE_PRICE_BB_CROSS_UP":
        live_part = f"live={live_price:,.0f} " if live_price is not None and pd.notna(live_price) else ""
        if cross_info:
            pending_side = cross_info.get("pending_side")
            pending_count = cross_info.get("pending_count", 0)
            pending_seconds = cross_info.get("pending_seconds", 0.0)
            return (
                f"{buy_reason} | {live_part}BB_MID {cur_bb:.1f} upper={float(cross_info.get('upper_trigger', cur_bb)):.1f} | "
                f"relation={cross_info.get('relation')} pending={pending_side} "
                f"count={pending_count}/{LIVE_PRICE_CROSS_CONFIRM_POLLS} "
                f"seconds={pending_seconds:.0f}/{LIVE_PRICE_CROSS_CONFIRM_SECONDS} | {snapshot}"
            )
        return (
            f"{buy_reason} | "
            f"{live_part}BB_MID {prev_bb:.1f}->{cur_bb:.1f} "
            f"(need live price to stay above BB middle long enough) | {snapshot}"
        )

    if buy_reason == "NO_MA5_BB_GOLDEN_CROSS":
        return f"{buy_reason} | MA5 {prev_ma5:.1f}->{cur_ma5:.1f} | BB_MID {prev_bb:.1f}->{cur_bb:.1f} | {snapshot}"

    if buy_reason == "MA5_AT_OR_BELOW_BB_MIDDLE":
        gap_pct = ((cur_bb - cur_ma5) / max(cur_bb, 1.0)) * 100
        return f"{buy_reason} | MA5={cur_ma5:.1f} BB_MID={cur_bb:.1f} gap={gap_pct:.3f}% | {snapshot}"

    if buy_reason == "BB_MIDDLE_FALLING":
        return f"{buy_reason} | BB_MID {prev_bb:.1f}->{cur_bb:.1f} | {snapshot}"

    if buy_reason == "MA5_FALLING":
        return f"{buy_reason} | MA5 {prev_ma5:.1f}->{cur_ma5:.1f} | {snapshot}"

    if buy_reason == "NOT_BULLISH":
        close_v = _num(cur, "close"); open_v = _num(cur, "open")
        return f"{buy_reason} | close={close_v:.0f} open={open_v:.0f} | {snapshot}"

    if buy_reason == "MISSING_INDICATOR":
        missing = [k for k in ("MA_5", "BB_MIDDLE") if pd.isna(_num(cur, k)) or pd.isna(_num(prev, k))]
        return f"{buy_reason} | NaN={','.join(missing)}"

    if not buy_reason.startswith("LOW_SCORE"):
        # OVERBOUGHT_*, NEAR_BB_UPPER_*, LOW_VOLUME_*, WEAK_TREND_ADX_*
        # reason string already contains the offending value; no extra context needed
        return f"{buy_reason} | {snapshot}"

    # LOW_SCORE: show which of the 6 sub-indicators failed
    failed = []

    k_c = _num(cur, "STOCH_K"); d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K"); d_p = _num(prev, "STOCH_D")
    if any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)) or not (
        (k_p <= d_p and k_c > d_c)
        and (STOCH_BUY_MIN <= k_c <= STOCH_BUY_MAX)
        and (k_c > k_p)
    ):
        failed.append("STOCH")

    rsi_c = _num(cur, "RSI")
    rsi_p = _num(prev, "RSI")
    if any(pd.isna(v) for v in (rsi_c, rsi_p)) or not (RSI_BUY_MIN <= rsi_c < RSI_BUY_MAX and rsi_c > rsi_p):
        failed.append("RSI")

    wr_c = _num(cur, "WILLIAMS_R"); wr_p = _num(prev, "WILLIAMS_R")
    if pd.isna(wr_c) or pd.isna(wr_p) or not (wr_c > wr_p and wr_c >= WILLIAMS_BUY_FLOOR):
        failed.append("WILLIAMS_R")

    macd_c = _num(cur, "MACD"); msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD"); msig_p = _num(prev, "MACD_SIGNAL")
    hist_c = _num(cur, "MACD_HIST")
    hist_p = _num(prev, "MACD_HIST")
    macd_accel_ok = (not pd.isna(hist_c) and hist_c > 0) or (
        not any(pd.isna(v) for v in (hist_c, hist_p)) and hist_c < 0 and hist_c > hist_p
    )
    if any(pd.isna(v) for v in (macd_c, msig_c)) or not (macd_c > msig_c and macd_accel_ok):
        failed.append("MACD")

    vwap = _num(cur, "VWAP"); close_v = _num(cur, "close")
    if pd.isna(vwap) or pd.isna(close_v) or not (close_v > vwap):
        failed.append("VWAP")

    obv_c = _num(cur, "OBV"); obv_ma_c = _num(cur, "OBV_MA"); obv_p = _num(prev, "OBV")
    obv_breakout = False
    if frame is not None and "OBV" in frame.columns and len(frame) >= OBV_BREAKOUT_LOOKBACK_BARS + 1:
        obv_series = pd.to_numeric(frame["OBV"], errors="coerce")
        recent_obv_high = obv_series.iloc[-(OBV_BREAKOUT_LOOKBACK_BARS + 1):-1].max()
        if not pd.isna(obv_c) and not pd.isna(recent_obv_high):
            obv_breakout = obv_c > float(recent_obv_high)
    obv_uptrend = not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)) and (obv_c > obv_ma_c and obv_c > obv_p)
    if not (obv_breakout or obv_uptrend):
        failed.append("OBV")

    suffix = f" | FAILED={','.join(failed)}" if failed else ""
    return f"{buy_reason}{suffix} | {snapshot}"


def _buy_support_score(cur: pd.Series, prev: pd.Series, frame: pd.DataFrame | None = None) -> int:
    score = 0

    # 1) Stoch: K가 D 상향 돌파 + 20~50 구간 + 상승 중
    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if (k_p <= d_p and k_c > d_c) and (STOCH_BUY_MIN <= k_c <= STOCH_BUY_MAX) and (k_c > k_p):
            score += 1

    # 2) RSI: 50 이상 70 미만 + 상승 중
    rsi_c = _num(cur, "RSI")
    rsi_p = _num(prev, "RSI")
    if not any(pd.isna(v) for v in (rsi_c, rsi_p)):
        if RSI_BUY_MIN <= rsi_c < RSI_BUY_MAX and rsi_c > rsi_p:
            score += 1

    # 3) Williams %R: 상승 전환 및 과매수/과매도 복귀
    wr_c = _num(cur, "WILLIAMS_R")
    wr_p = _num(prev, "WILLIAMS_R")
    if not pd.isna(wr_c) and not pd.isna(wr_p):
        if wr_c > wr_p and wr_c >= WILLIAMS_BUY_FLOOR:
            score += 1

    # 4) MACD: MACD > Signal + 히스토그램이 양수 또는 수축
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    hist_c = _num(cur, "MACD_HIST")
    hist_p = _num(prev, "MACD_HIST")
    if not any(pd.isna(v) for v in (macd_c, msig_c, hist_c)):
        hist_accel = (hist_c > 0) or (not pd.isna(hist_p) and hist_c < 0 and hist_c > hist_p)
        if macd_c > msig_c and hist_accel:
            score += 1

    # 5) VWAP: 현재가 > VWAP (당일 수급 우위, 평균매수가 상회)
    vwap = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not pd.isna(vwap) and not pd.isna(close_v) and close_v > vwap:
        score += 1

    # 6) OBV: 최근 5봉 고점 돌파 또는 OBV 상승세
    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    obv_breakout = False
    if frame is not None and "OBV" in frame.columns and len(frame) >= OBV_BREAKOUT_LOOKBACK_BARS + 1:
        obv_series = pd.to_numeric(frame["OBV"], errors="coerce")
        recent_obv_high = obv_series.iloc[-(OBV_BREAKOUT_LOOKBACK_BARS + 1):-1].max()
        if not pd.isna(obv_c) and not pd.isna(recent_obv_high):
            obv_breakout = obv_c > float(recent_obv_high)
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_breakout or (obv_c > obv_ma_c and obv_c > obv_p):
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


def _near_cross_momentum_flags(cur: pd.Series, prev: pd.Series) -> dict[str, float | bool]:
    """Builds near-cross diagnostics for reject logging.

    Live entry decisions are owned by the shared core strategy.
    """
    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")

    if any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)):
        return {"can_arm": False, "can_early": False, "gap_ratio": float("nan"), "ma_rise_ratio": float("nan")}

    gap = cur_bb - cur_ma5
    gap_ratio = gap / max(abs(cur_bb), 1.0)
    ma_rise_ratio = (cur_ma5 - prev_ma5) / max(abs(prev_ma5), 1.0)

    below_or_equal = cur_ma5 <= cur_bb
    arm_shape_ok = (prev_ma5 <= prev_bb) and below_or_equal

    can_arm = arm_shape_ok and (gap_ratio >= 0) and (gap_ratio <= NEAR_CROSS_ARM_GAP_MAX) and (ma_rise_ratio >= NEAR_CROSS_ARM_MA_RISE_MIN)
    can_early = arm_shape_ok and (gap_ratio >= 0) and (gap_ratio <= NEAR_CROSS_EARLY_GAP_MAX) and (ma_rise_ratio >= NEAR_CROSS_EARLY_MA_RISE_MIN)

    return {
        "can_arm": can_arm,
        "can_early": can_early,
        "gap_ratio": float(gap_ratio),
        "ma_rise_ratio": float(ma_rise_ratio),
    }


def _passes_early_near_cross_liquidity(cur: pd.Series) -> tuple[bool, str]:
    """Liquidity guard for ARM/EARLY near-cross entry."""
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    close_v = _num(cur, "close")
    if any(pd.isna(v) for v in (vol, vol_ma, close_v)):
        return False, "LIQUIDITY_DATA_NAN"

    turnover = close_v * vol
    if vol < EARLY_NEAR_CROSS_MIN_VOLUME:
        return False, f"LOW_ABS_VOLUME_{vol:.0f}_LT_{EARLY_NEAR_CROSS_MIN_VOLUME}"
    if vol_ma < EARLY_NEAR_CROSS_MIN_VOL_MA:
        return False, f"LOW_VOL_MA_{vol_ma:.0f}_LT_{EARLY_NEAR_CROSS_MIN_VOL_MA}"
    if turnover < EARLY_NEAR_CROSS_MIN_TURNOVER_KRW:
        return False, f"LOW_TURNOVER_{turnover:,.0f}_LT_{EARLY_NEAR_CROSS_MIN_TURNOVER_KRW:,}"
    return True, "OK"


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
    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")
    prev_close = _num(prev, "close")
    close_val = _num(cur, "close")
    support_score = _buy_support_score(cur, prev, frame=frame) if frame is not None else -1
    near_flags = _near_cross_momentum_flags(cur, prev)
    liquidity_ok, liquidity_reason = _passes_early_near_cross_liquidity(cur)
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    vol_ratio = vol / vol_ma if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0 else float("nan")
    ma_cross = not any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)) and (prev_ma5 <= prev_bb) and (cur_ma5 > cur_bb)
    close_cross = not any(pd.isna(v) for v in (prev_close, close_val, prev_bb, cur_bb)) and (prev_close <= prev_bb) and (close_val > cur_bb)
    gap_ratio = float(near_flags["gap_ratio"]) * 100 if not pd.isna(float(near_flags["gap_ratio"])) else float("nan")
    ma_rise_ratio = float(near_flags["ma_rise_ratio"]) * 100 if not pd.isna(float(near_flags["ma_rise_ratio"])) else float("nan")
    live_signal = cross_info.get("signal") if cross_info else None
    live_part = f"live={live_price:,.0f}" if live_price is not None and pd.notna(live_price) else "live=nan"
    return (
        f"GATES ma_cross={ma_cross} close_cross={close_cross} signal={live_signal} {live_part} "
        f"arm={bool(near_flags['can_arm'])} early={bool(near_flags['can_early'])} "
        f"gap={gap_ratio:.3f}% rise={ma_rise_ratio:.3f}% liq={liquidity_ok}:{liquidity_reason} "
        f"score={support_score} vol_ratio={vol_ratio:.2f}"
    )


def _is_box_range_hold_zone(frame: pd.DataFrame) -> tuple[bool, str]:
    """Detects narrow-range consolidation zone for technical-sell hold."""
    if len(frame) < BOX_RANGE_HOLD_LOOKBACK_BARS:
        return False, "INSUFFICIENT_BOX_BARS"

    recent = frame.tail(BOX_RANGE_HOLD_LOOKBACK_BARS)
    high_v = pd.to_numeric(recent["high"], errors="coerce").max()
    low_v = pd.to_numeric(recent["low"], errors="coerce").min()
    close_v = _num(recent.iloc[-1], "close")
    bb_up = _num(recent.iloc[-1], "BB_UPPER")
    bb_low = _num(recent.iloc[-1], "BB_LOWER")

    if any(pd.isna(v) for v in (high_v, low_v, close_v, bb_up, bb_low)) or close_v <= 0:
        return False, "BOX_DATA_NAN"

    range_pct = (float(high_v) - float(low_v)) / float(close_v)
    bb_width_pct = (float(bb_up) - float(bb_low)) / float(close_v)
    is_box = range_pct <= BOX_RANGE_HOLD_MAX_RANGE_PCT and bb_width_pct <= BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT

    return is_box, f"RANGE_{range_pct*100:.2f}%_BBW_{bb_width_pct*100:.2f}%"


def check_buy_condition(frame: pd.DataFrame, now: datetime, live_price: float, cross_info: dict[str, object]) -> tuple[bool, str]:
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
# Trading API
# ---------------------------------------------------------------------------

class TradingAPI:
    def __init__(self):
        ka.auth()
        trenv = ka.getTREnv()
        self.cano = trenv.my_acct
        self.acnt_prdt_cd = trenv.my_prod
        self.positions: dict[str, dict] = {}
        self.pending_orders: dict[str, dict] = {}
        self.trade_lock_until: dict[str, datetime] = {}
        self._last_sync_at: datetime | None = None
        self._last_pending_poll_at: datetime | None = None
        self.sync_positions_from_account(force=True)
        log("TradingAPI (r76 MA5-BB multi-indicator) initialized")

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
            updated[code] = {
                "buy_price": avg_price,
                "quantity": qty,
                "buy_time": prev.get("buy_time", datetime.now()),
                "buy_session": prev.get("buy_session", "synced"),
                "current_price": current_price,
                "highest_price": max(float(prev.get("highest_price", current_price)), current_price),
            }

        self.positions = updated
        self._reconcile_pending_positions()

    def get_open_positions(self) -> dict[str, dict]:
        return self.positions

    def has_pending_order(self, code: str) -> bool:
        return code in self.pending_orders

    def get_pending_order(self, code: str) -> dict | None:
        return self.pending_orders.get(code)

    def _in_cooldown(self, code: str, now: datetime) -> bool:
        until = self.trade_lock_until.get(code)
        return until is not None and now < until

    def _mark_trade_lock(self, code: str, now: datetime) -> None:
        self.trade_lock_until[code] = now + timedelta(minutes=TRADE_COOLDOWN_MINUTES)

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
                env_dv="real",
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

        pos["buy_time"] = pending.get("submitted_at", pos.get("buy_time", datetime.now()))
        pos["buy_session"] = pending.get("session", pos.get("buy_session", "synced"))
        if fill_price > 0:
            pos["buy_price"] = fill_price
            pos["current_price"] = float(pos.get("current_price") or fill_price)
            pos["highest_price"] = max(float(pos.get("highest_price", fill_price)), float(pos["current_price"]), fill_price)

        code_name = str(pending.get("code_name", ""))
        detail_suffix = f" | {pending['buy_detail']}" if pending.get("buy_detail") else ""
        code_label = _format_code_label(code, code_name)
        log(
            f"BUY executed  | {code_label} | filled_qty={filled_qty}/{requested_qty} | "
            f"price={fill_price:,.0f} | session={pending.get('session', 'unknown')} | exch={pending.get('exchange', 'UNKNOWN')}{detail_suffix}"
        )
        log_trade(
            f"BUY executed  | {code_label} | filled_qty={filled_qty}/{requested_qty} | "
            f"price={fill_price:,.0f} | session={pending.get('session', 'unknown')} | exch={pending.get('exchange', 'UNKNOWN')}{detail_suffix}"
        )
        _log_trade_event_banner(
            event="BUY EXECUTED",
            code=code,
            qty=filled_qty,
            price=fill_price,
            detail=f"session={pending.get('session', 'unknown')} exch={pending.get('exchange', 'UNKNOWN')}" + detail_suffix,
            code_name=code_name,
        )
        self.pending_orders.pop(code, None)

    def _confirm_pending_sell(self, code: str, pending: dict, status: dict | None, remaining_qty: int) -> None:
        requested_qty = int(pending.get("quantity", 0))
        filled_qty = int(status.get("filled_qty", requested_qty)) if status else requested_qty
        avg_price = float(status.get("avg_price", 0.0)) if status else 0.0
        fill_price = avg_price if avg_price > 0 else float(pending.get("requested_price", 0.0))
        buy_price = float(pending.get("buy_price", 0.0))
        pnl_pct = ((fill_price / buy_price) - 1.0) * 100.0 if buy_price > 0 and fill_price > 0 else float("nan")
        code_name = str(pending.get("code_name", ""))
        code_label = _format_code_label(code, code_name)
        log(
            f"SELL executed  | {code_label} | filled_qty={filled_qty}/{requested_qty} | "
            f"price={fill_price:,.0f} | remaining={remaining_qty} | pnl={pnl_pct:.2f}% | "
            f"reason={pending.get('reason', 'UNKNOWN')} | exch={pending.get('exchange', 'UNKNOWN')}"
        )
        log_trade(
            f"SELL executed  | {code_label} | filled_qty={filled_qty}/{requested_qty} | "
            f"price={fill_price:,.0f} | remaining={remaining_qty} | pnl={pnl_pct:.2f}% | "
            f"reason={pending.get('reason', 'UNKNOWN')} | exch={pending.get('exchange', 'UNKNOWN')}"
        )
        _log_trade_event_banner(
            event="SELL EXECUTED",
            code=code,
            qty=filled_qty,
            price=fill_price,
            detail=f"pnl={pnl_pct:.2f}% reason={pending.get('reason', 'UNKNOWN')} exch={pending.get('exchange', 'UNKNOWN')}",
            code_name=code_name,
        )
        self.pending_orders.pop(code, None)

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
                        self._maybe_log_pending_progress(
                            pending,
                            f"BUY closed without fill | {code} | order_no={pending.get('order_no', '')} | exch={pending.get('exchange', 'UNKNOWN')}",
                            "buy_closed_without_fill",
                        )
                        self.pending_orders.pop(code, None)
                        continue

                    self._maybe_log_pending_progress(
                        pending,
                        f"BUY pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                        f"| remaining={int(status.get('remaining_qty', 0))} | order_no={status.get('order_no', '')}",
                        f"buy_pending:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}",
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
                env_dv="real",
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
        if qty <= 0 or self._in_cooldown(code, now) or self.has_pending_order(code):
            return False

        # 주문 직전 시점에 주문가능수량을 다시 확인(50만원 한도 즉시 반영)
        affordable_qty = self.get_affordable_buy_qty(code, price, now, nxt_tradeable)
        qty = min(int(qty), int(affordable_qty))
        if qty <= 0:
            log(f"BUY skipped | {code} | reason=INSUFFICIENT_BUYING_POWER_AT_ORDER_TIME")
            return False

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return False

        ord_unpr = order_spec["ord_unpr"] if order_spec["ord_unpr"] is not None else str(int(round(price)))
        try:
            order_result = dsf.order_cash(
                env_dv="real",
                ord_dv="buy",
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                pdno=code,
                ord_dvsn=order_spec["ord_dvsn"],
                ord_qty=str(qty),
                ord_unpr=ord_unpr,
                excg_id_dvsn_cd=order_spec["exchange"],
            )
        except Exception as exc:
            log(f"BUY error | {code} | {exc}")
            return False

        if not _order_succeeded(order_result):
            error_detail = _extract_order_error_detail(order_result)
            log(f"BUY failed | {code} | qty={qty} | {error_detail}")
            return False

        requested_price = _extract_order_price(order_result) or price
        order_no = _extract_order_number(order_result)
        order_time = _extract_order_time(order_result)
        self.pending_orders[code] = {
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
        self._mark_trade_lock(code, now)
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
            detail=f"session={session} exch={order_spec['exchange']} order_no={order_no or 'UNKNOWN'}" + (f" | {buy_detail}" if buy_detail else ""),
            code_name=code_name,
        )
        self.refresh_pending_orders(now)
        return True

    def place_sell_order(self, code: str, qty: int, now: datetime, reason: str, nxt_tradeable: bool, price: float | None = None, code_name: str = "") -> bool:
        self.sync_positions_from_account(force=False)
        pos = self.positions.get(code)
        if not pos or pos.get("quantity", 0) <= 0:
            return False

        qty = min(int(qty), int(pos["quantity"]))
        if qty <= 0 or self._in_cooldown(code, now) or self.has_pending_order(code):
            return False

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return False

        current_price = float(price or pos.get("current_price") or pos["buy_price"])
        ord_unpr = order_spec["ord_unpr"] if order_spec["ord_unpr"] is not None else str(int(round(current_price)))

        try:
            order_result = dsf.order_cash(
                env_dv="real",
                ord_dv="sell",
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                pdno=code,
                ord_dvsn=order_spec["ord_dvsn"],
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
        self.pending_orders[code] = {
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
            "pre_submit_qty": int(pos.get("quantity", qty)),
        }
        self._mark_trade_lock(code, now)
        code_label = _format_code_label(code, code_name)
        log(
            f"SELL submitted | {code_label} | qty={qty} | requested={requested_price:,.0f} | "
            f"reason={reason} | exch={order_spec['exchange']} | order_no={order_no or 'UNKNOWN'}"
        )
        log_trade(
            f"SELL submitted | {code_label} | qty={qty} | requested={requested_price:,.0f} | "
            f"reason={reason} | exch={order_spec['exchange']} | order_no={order_no or 'UNKNOWN'}"
        )
        _log_trade_event_banner(
            event="SELL SUBMITTED",
            code=code,
            qty=int(qty),
            price=float(requested_price),
            detail=f"reason={reason} exch={order_spec['exchange']} order_no={order_no or 'UNKNOWN'}",
            code_name=code_name,
        )
        self.refresh_pending_orders(now)
        return True


# ---------------------------------------------------------------------------
# 예약 청산
# ---------------------------------------------------------------------------

def run_scheduled_liquidations(current_dt: datetime, api: TradingAPI, nxt_map: dict[str, bool], watch_map: dict[str, str], state: dict) -> None:
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


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run(target_date: str | None = None) -> None:
    ka.auth()
    now = datetime.now()
    is_open_day, market_day_log = get_market_day_status(now)
    log(market_day_log)

    if not is_open_day:
        return

    watch_file = _resolve_watchlist_file(target_date)
    try:
        watch_map = load_today_codes(watch_file)
    except Exception as exc:
        log(f"Failed to load code list: {exc}")
        watch_map = {}

    if not watch_map:
        log("No codes loaded")
        return

    register_symbol_names(watch_map)

    log(f"Watchlist source: {watch_file}")

    if ENABLE_NXT_SESSION:
        log("MODE BANNER: REGULAR+NXT_MODE")
    else:
        log("MODE BANNER: REGULAR_ONLY_MODE")

    nxt_map = {code: is_nxt_tradeable(code) for code in watch_map}
    for code, name in watch_map.items():
        log(f"WATCH | {code} | {name} | NXT={nxt_map[code]}")

    log("Strategy: live price cross over buffered BB middle + Stoch/RSI/Williams confirmation")
    log(
        f"Live-cross filter: BB buffer={LIVE_PRICE_BB_BUFFER_PCT*100:.3f}% | "
        f"confirm polls={LIVE_PRICE_CROSS_CONFIRM_POLLS} | confirm seconds={LIVE_PRICE_CROSS_CONFIRM_SECONDS}"
    )
    log(
        f"Polling: live={LIVE_PRICE_POLL_INTERVAL_SECONDS}s | "
        f"frame refresh=3min boundary + backfill {FRAME_BACKFILL_SYNC_SECONDS}s | "
        f"buy consecutive confirms={BUY_CONSECUTIVE_CONFIRM_COUNT}"
    )
    log(f"TP={TAKE_PROFIT_PERCENT*100:.1f}% | SL={STOP_LOSS_PERCENT*100:.1f}% | Trail={TRAILING_STOP_FROM_PEAK*100:.1f}%")
    log(f"Indicator warmup: require bars >= {INDICATOR_WARMUP_BARS} before new entries")
    if ENABLE_NEAR_CROSS_ARM or ENABLE_EARLY_NEAR_CROSS_ENTRY or ENABLE_PRICE_LEAD_BB_BREAKOUT:
        log("NOTICE: near-cross / price-lead breakout flags are currently diagnostic-only in live executor; shared core buy logic remains the source of truth.")

    api = TradingAPI()
    traded_today: set[str] = set()
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
    liquidation_state: dict = {}
    current_trade_date = now.date()
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

        if current_dt.date() != current_trade_date:
            current_trade_date = current_dt.date()
            traded_today.clear()
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

        is_open_day, market_day_log = get_market_day_status(current_dt)
        if not is_open_day:
            log(market_day_log)
            log("MARKET LOOP STOP | reason=market_closed_day")
            break

        market_end_time = AFTERNOON_NXT_END if ENABLE_NXT_SESSION else REGULAR_END
        if current_dt.time() >= market_end_time:
            run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state)
            log(f"{market_end_time:%H:%M} reached. Stopping.")
            break

        if not is_regular_session(current_dt) and not is_nxt_session(current_dt):
            time.sleep(LIVE_PRICE_POLL_INTERVAL_SECONDS)
            continue

        api.sync_positions_from_account(force=False)
        api.refresh_pending_orders(current_dt)
        run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state)

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

            cross_info = update_live_price_cross_state(
                live_price_cross_state,
                code,
                current_dt,
                float(price),
                _num(cur, "BB_MIDDLE"),
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
                f"  [CHECK] {symbol_label} | {current_dt:%H:%M:%S} | bars={len(frame)} live={price:,.0f} ({price_source}) bar_close={float(cur['close']):,.0f} | "
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
                buy_confirm_state.pop(code, None)
                entry_price = float(pos["buy_price"])
                pnl_pct = (price / entry_price) - 1.0

                # 저가(bar low)가 현재가보다 낮으면 손절 판정에 우선 반영
                # 확정봉 low를 직접 섞으면 현재가가 회복된 상태에서도 오손절이 날 수 있어
                # 실시간 손절은 현재가 기준으로 판정한다.
                _bar_low = float(cur["low"]) if "low" in cur.index and not pd.isna(cur["low"]) else float("nan")
                _sl_price = price
                _pnl_sl = (_sl_price / entry_price) - 1.0

                pos["current_price"] = price
                pos["highest_price"] = max(float(pos.get("highest_price", price)), price)
                highest_price = float(pos["highest_price"])  # TP/트레일링 로직 계산 전에 갱신값 반영
                peak_pnl_pct = (highest_price / entry_price) - 1.0 if highest_price > 0 and entry_price > 0 else 0.0
                profit_giveback = peak_pnl_pct - pnl_pct

                if pnl_pct >= TAKE_PROFIT_PERCENT:
                    if ENABLE_TP_EXTENSION_TRAILING:
                        # TP 도달 시 즉시 익절 대신 고점 트레일링 모드로 전환
                        log(
                            f"  [TP_EXTENSION] {code} | pnl={pnl_pct*100:.2f}% >= TP {TAKE_PROFIT_PERCENT*100:.1f}% | "
                            f"고점 트레일링 모드 전환 (trail={TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%) | "
                            f"price={price:,.0f} peak={highest_price:,.0f}"
                        )
                    else:
                        reason_tp = f"TAKE_PROFIT_+{TAKE_PROFIT_PERCENT*100:.1f}%"
                        log(f"  [SELL TRIGGER] {code} | {reason_tp} | price={price:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}%")
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_tp, nxt_tradeable, price=price, code_name=name):
                            log(f"  [SELL EXECUTED] {code} | {reason_tp} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                # ── POST-BUY ENTRY DROP GUARD ─────────────────────────────────────
                _held_for_guard = (current_dt - pos.get("buy_time", current_dt)).total_seconds()
                if _held_for_guard <= POST_BUY_BB_DROP_ARMED_SECONDS:
                    _drop_hold_seconds = update_timed_condition_state(
                        post_buy_bb_drop_state,
                        code,
                        pos.get("buy_time"),
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
                _breakeven_hold_seconds = update_timed_condition_state(
                    breakeven_fail_state,
                    code,
                    pos.get("buy_time"),
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
                    pos.get("buy_time"),
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

                _held_sl = (current_dt - pos.get("buy_time", current_dt)).total_seconds()
                _sl_threshold = STOP_LOSS_EARLY_PERCENT if _held_sl < STOP_LOSS_MIN_HOLD_SECONDS else STOP_LOSS_PERCENT
                if _pnl_sl <= _sl_threshold:
                    reason_sl = f"STOP_LOSS_EARLY_{_sl_threshold*100:.1f}%" if _held_sl < STOP_LOSS_MIN_HOLD_SECONDS else f"STOP_LOSS_{_sl_threshold*100:.1f}%"
                    log(f"  [SELL TRIGGER] {code} | {reason_sl} | held={_held_sl:.0f}s price={price:,.0f} bar_low={_bar_low:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}% sl_pnl={_pnl_sl*100:.2f}%")
                    trailing_sell_confirm_state.pop(code, None)
                    if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_sl, nxt_tradeable, price=price, code_name=name):
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
                    if ENABLE_TP_EXTENSION_TRAILING and peak_pnl_pct >= TAKE_PROFIT_PERCENT:
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
                            f"  [BUY REJECT] {symbol_label} | SESSION_OPEN_WARMUP | "
                            f"elapsed={_warmup_elapsed:.0f}s / {STARTUP_WARMUP_SECONDS}s | "
                            f"session_open={session_open_dt:%H:%M:%S}"
                        )
                        continue
                if not ALLOW_REBUY_SAME_CODE and code in traded_today:
                    continue
                if api._in_cooldown(code, current_dt):
                    continue
                if signal_buy_bar.get(code) == bar_time:
                    continue

                prev_bar = frame.iloc[-2]

                buy_ok, buy_reason = check_buy_condition(frame, current_dt, price, cross_info)

                if not buy_ok:
                    buy_confirm_state.pop(code, None)
                    detail = _buy_reject_detail(
                        buy_reason,
                        cur,
                        prev_bar,
                        live_price=price,
                        cross_info=cross_info,
                        frame=frame,
                    )
                    log(f"  [BUY REJECT] {symbol_label} | {detail}")
                    continue

                confirm_state = buy_confirm_state.get(code)
                if (
                    confirm_state is None
                    or confirm_state.get("bar_time") != bar_time
                ):
                    confirm_count = 1
                else:
                    confirm_count = int(confirm_state.get("count", 0)) + 1

                buy_confirm_state[code] = {"bar_time": bar_time, "count": confirm_count}
                if confirm_count < BUY_CONSECUTIVE_CONFIRM_COUNT:
                    log(
                        f"  [BUY HOLD] {symbol_label} | reason=WAIT_CONSECUTIVE_CONFIRM | "
                        f"count={confirm_count}/{BUY_CONSECUTIVE_CONFIRM_COUNT} | "
                        f"bar={bar_time:%H:%M:%S}"
                    )
                    continue

                qty = api.get_affordable_buy_qty(code, price, current_dt, nxt_tradeable)
                if qty <= 0:
                    log(f"  [BUY REJECT] {symbol_label} | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price={price:,.0f}")
                    continue

                if "bar_close(stale_live=" in price_source:
                    log(
                        f"  [BUY REJECT] {symbol_label} | STALE_LIVE_PRICE | "
                        f"source={price_source} ttl={LIVE_PRICE_STALE_TTL_SECONDS}s"
                    )
                    continue

                session = classify_buy_session(current_dt)
                buy_detail = (
                    f"reason={buy_reason} signal={cross_info.get('signal')} "
                    f"live={price:,.0f} bb_mid={_num(cur, 'BB_MIDDLE'):.1f} "
                    f"bar_close={_num(cur, 'close'):,.0f} ma5={_num(cur, 'MA_5'):.1f}"
                )
                if api.place_buy_order(code, price, qty, current_dt, nxt_tradeable, session, buy_detail=buy_detail, code_name=name):
                    log(
                        f"  [BUY EVAL] {symbol_label} | OK {buy_reason} | {current_dt:%H:%M:%S} | "
                        f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}->{_num(cur, 'BB_MIDDLE'):.1f} | "
                        f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                        f"K={_num(prev_bar, 'STOCH_K'):.1f}->{_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                        f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}->{_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                        f"MACD {_num(prev_bar, 'MACD'):.2f}->{_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} | "
                        f"ADX={_num(cur, 'ADX'):.1f} +DI={_num(cur, 'DI_PLUS'):.1f} -DI={_num(cur, 'DI_MINUS'):.1f} | "
                        f"VOL={_num(cur, 'volume'):,.0f} VOLMA={_num(cur, 'VOL_MA20'):,.0f} | "
                        f"VWAP={_num(cur, 'VWAP'):,.0f} OBV={_num(cur, 'OBV'):,.0f} OBVMA={_num(cur, 'OBV_MA'):,.0f}"
                    )
                    log(f"  [BUY EXECUTED] {symbol_label} | {buy_reason} | qty={qty} price={price:,.0f} session={session}")
                    traded_today.add(code)
                    signal_buy_bar[code] = bar_time
                    buy_confirm_state.pop(code, None)

                log("=" * 110)

        time.sleep(LIVE_PRICE_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    args = _parse_args()
    run(target_date=args.date)

