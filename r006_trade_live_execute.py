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
import inspect
import logging
import sys
import time
import unicodedata
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional

import pandas as pd
from r005_strategy_core_shared import (
    R76StrategyConfig,
    check_buy_condition as shared_check_buy_condition,
    check_sell_condition as shared_check_sell_condition,
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


TODAY_CODE_FILE = current_dir / "r008_trade_watchlist_today.txt"
DATA_DIR = current_dir / "data"

# ---------------------------------------------------------------------------
# 지표 파라미터
# ---------------------------------------------------------------------------
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
OBV_MA_PERIOD = 10      # OBV 이동평균 기간 (방향성 보조)
MACD_FAST = 5           # 빠른 EMA 기간 (API 30�??�약 고려)
MACD_SLOW = 12          # ?�린 EMA 기간
MACD_SIGNAL_PERIOD = 4  # MACD ?�그??기간
ADX_PERIOD = 7          # ADX/DI 계산 기간 (?�기 추세 강도)
ADX_MIN_TREND = 20.0    # ADX 20 미만 = 약보 구간, 신규 매수 보류
ADX_STRONG_TREND = 40.0  # 강한 추세 구간?�서??거래??기�? ?�화

# ---------------------------------------------------------------------------
# 매매 ?�라미터
# ---------------------------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000
TAKE_PROFIT_PERCENT = 0.035
STOP_LOSS_PERCENT = -0.012
STOP_LOSS_EARLY_PERCENT = -0.020     # 진입 초기(STOP_LOSS_MIN_HOLD_SECONDS 이내) 손실 발생시 조기 손절
STOP_LOSS_MIN_HOLD_SECONDS = 600    # 보유시간(10분 미만이면 EARLY 손절 기준 적용, 기본 10분)
STARTUP_WARMUP_SECONDS = 180        # 스크립트 시작 후 초기 시간(3분 이내 신규 매수 차단, 지표 안정+모니터링 여유 보장)
TRAILING_STOP_FROM_PEAK = 0.005
# 보조지??기반 매도(AUX_REVERSAL) 최소 ?�익�?게이??
# 보조지표 조건 만족 시 이익률일 때만 매도 허용
AUX_SELL_MIN_PNL_SCORE2 = 0.010  # score=2  -> +1.0% ?�상
AUX_SELL_MIN_PNL_SCORE3 = 0.005  # score=3  -> +0.5% ?�상
AUX_SELL_MIN_PNL_SCORE4 = 0.000  # score>=4 -> ?�익분기 ?�상
# MA5 ?�드?�로??매도 ?�바?�스: ?�쏘 방�?�??�해 ?�반 구간?� 1�?�??�인 ??매도
# ?? ?�실/?�세 강도가 ?�면 즉시 매도??급락 리스?�는 방어
MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007     # -0.7% ?�하 ?�실?�면 즉시 매도 ?�용
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2         # 보조 ?�세 ?�수 ?�으�?즉시 매도 ?�용
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1       # ?�음 �??�인 매도 최소 보조 ?�수
MA5_BB_DOWN_CROSS_MIN_PNL = 0.000             # ?�드?�로??계열 매도??최소 ?�익(기본 0%=본전) ?�상?�서�??�용
# 박스�?구간?�서??기술??매도 ?�호�?보류?�고, ?�절/?�절(�??�레?�링)�??�용
ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
# ?�절 기�? ?�달 ??즉시 매도 ?�??고점 ?�레?�링 모드�??�환
# (강모멘�? 급등 ??추�? ?�익 구간 ?�보)
ENABLE_TP_EXTENSION_TRAILING = True
TP_EXTENSION_TRAIL_FROM_PEAK = 0.010     # TP ?�장 구간?�서 고점 ?��?1.0% ?�락 ??매도
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065         # 최근 N봉 고저폭이 현재가의 0.65% 이내
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080      # 볼린?� 밴드 ??�� ?�재가 ?��?0.80% ?�내
# 매수 2?�계(근접 ARM -> ?�정/강모멘�? 진입) ?�라미터
NEAR_CROSS_ARM_GAP_MAX = 0.0045        # MA5가 BB_MIDDLE 아래여도 gap 0.45% 이내면 ARM 후보
NEAR_CROSS_ARM_MA_RISE_MIN = 0.0006    # ARM 최소 MA5 ?�승�?(0.06%)
NEAR_CROSS_EARLY_GAP_MAX = 0.0045      # ARM ?�후 조기진입 ?�용 gap 0.45% ?�내
NEAR_CROSS_EARLY_MA_RISE_MIN = 0.0010  # ARM ?�후 조기진입 최소 MA5 ?�승�?(0.10%)
NEAR_CROSS_ARM_EXPIRE_BARS = 2         # ARM ?�효 기간(3분봉 기�? �?개수)
# 기본 원칙은 MA5 상향 돌파 우선. 예외 경로는 지정 시간창에서만 제한적으로 허용.
ENABLE_NEAR_CROSS_ARM = True
ENABLE_EARLY_NEAR_CROSS_ENTRY = True
ENABLE_PRICE_LEAD_BB_BREAKOUT = True
PRICE_LEAD_BREAKOUT_MIN_SCORE = 3
PRICE_LEAD_BREAKOUT_MIN_ADX = 25.0
PRICE_LEAD_BREAKOUT_ALLOW_OVERBOUGHT = True
EARLY_NEAR_CROSS_ALLOWED_START = dt_time(9, 0)
EARLY_NEAR_CROSS_ALLOWED_END = dt_time(11, 30)
EARLY_NEAR_CROSS_ALLOW_NXT = False
# 조기진입(ARM/EARLY) 유동성 필터: 거래량 체결 급등으로 인한 허위 진입 방지
EARLY_NEAR_CROSS_MIN_VOLUME = 800
EARLY_NEAR_CROSS_MIN_VOL_MA = 500
EARLY_NEAR_CROSS_MIN_TURNOVER_KRW = 5_000_000
POLL_INTERVAL_SECONDS = 20
LIVE_PRICE_BB_BUFFER_PCT = 0.0008      # 현재가가 BB 중간선 위로 0.08% 이상 넘어야 유효 돌파로 인정
LIVE_PRICE_CROSS_CONFIRM_POLLS = 3      # 20초 폴링 기준 3회 연속 확인 (상향 돌파/매수)
LIVE_PRICE_CROSS_CONFIRM_SECONDS = 60   # �?감�? ??최소 60�??��? (?�향 ?�로??/ 매수)
# 하향 돌파(매도)는 즉시 반응, 지연되면 급락 구간에서 매도 기회 놓침
LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS = 1     # ?�향 ?�로?? 1??감�? 즉시 ?�호 발동
LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS = 0   # 하향 돌파는 지표가 즉시 확인
ACCOUNT_SYNC_INTERVAL_SECONDS = 90
MIN_BARS_REQUIRED = 3  # ?�전봉·현?�봉 비교??최소 3�??�요 (지?�는 min_periods=1�??�체 보완)
ALLOW_REBUY_SAME_CODE = False
TRADE_COOLDOWN_MINUTES = 3

# ---------------------------------------------------------------------------
# ?�션 / ?�간 ?�수
# ---------------------------------------------------------------------------
MORNING_NXT_START = dt_time(8, 0)
MORNING_NXT_END = dt_time(8, 50)
REGULAR_START = dt_time(9, 0)
REGULAR_END = dt_time(15, 30)
REGULAR_NEW_ENTRY_CUTOFF = dt_time(15, 20)
REGULAR_FORCE_EXIT = dt_time(15, 20)
AFTERNOON_NXT_START = dt_time(15, 30)
AFTERNOON_NXT_END = dt_time(20, 0)
AFTERNOON_NXT_NEW_ENTRY_CUTOFF = dt_time(19, 59)
AFTERNOON_NXT_FORCE_EXIT = dt_time(19, 59)

# 보조지표 계수
STOCH_OVERBOUGHT = 80.0
STOCH_BUY_MIN = 20.0
STOCH_BUY_MAX = 50.0
RSI_BUY_MIN = 50.0
RSI_BUY_MAX = 70.0
WILLIAMS_BUY_FLOOR = -70.0
WILLIAMS_OVERBOUGHT_CEIL = -20.0
BB_UPPER_PROXIMITY_MAX = 0.85  # (close-bb_lower)/(bb_upper-bb_lower) 값이 ?�면 ?�단 추격 구간
OBV_BREAKOUT_LOOKBACK_BARS = 5  # OBV가 최근 N봉 고점 돌파하는지 확인
ENABLE_STRICT_MA5_BB_GOLDEN_CROSS = True  # �?기�? MA5 ?�향 ?�파�??�규 매수 ?�수 조건?�로 ?�용
ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS = True
STRONG_TREND_OVERBOUGHT_MIN_SCORE = 5
STRONG_TREND_OVERBOUGHT_MIN_VOL_RATIO = 1.50
STRONG_TREND_OVERBOUGHT_MIN_ADX = 30.0

# 거래량 필터(시간대/세션 가변)
VOLUME_RATIO_OPEN = 0.80       # ?�초�?09:00~10:00)
VOLUME_RATIO_MIDDAY = 0.60     # ?�중(10:00~14:30)
VOLUME_RATIO_CLOSE = 0.70      # ?�후�?14:30~15:30)
VOLUME_RATIO_NXT = 0.55        # NXT ?�션
VOLUME_RATIO_STRONG_RELAX = 0.10  # ADX 강추세 시 완화 비율
VOLUME_RATIO_FLOOR = 0.50      # ?�화 ?�한

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
    bb_squeeze_min_width_pct=0.0,
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

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
log_dir = current_dir / "logs"
log_dir.mkdir(exist_ok=True)
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

# domestic_stock_functions / inquire_time_itemchartprice 관련 라이브러리에 있음
# 출력?�는 "Data fetch complete.", "Call Next page...", "Max recursive depth reached."
# 로그�??�제 (루트 로거??INFO지�??��? 모듈?� WARNING ?�상�??�시)
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


def log(msg: str) -> None:
    logger.info(msg)


trade_logger = logging.getLogger("trade_events")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False

_trade_handler = logging.FileHandler(log_dir / trade_log_filename, encoding="utf-8")
_trade_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
trade_logger.handlers.clear()
trade_logger.addHandler(_trade_handler)


def log_trade(msg: str) -> None:
    trade_logger.info(msg)


# ---------------------------------------------------------------------------
# ?�일 ?�틸
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
# 주문 결과 ?�싱
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


def _extract_order_price(result):
    if result is None:
        return None

    candidates = ("avg_pric", "avg_price", "stck_prpr", "ord_unpr", "fill_pric", "ccld_pric", "prpr")

    def _pick(mapping):
        if not isinstance(mapping, dict):
            return None
        for key in candidates:
            value = mapping.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        for nested in ("output", "output1", "data"):
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
# NXT 가??종목 ?��?
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
# ?�션 ?�퍼
# ---------------------------------------------------------------------------

def is_weekday_market_day(now: datetime) -> bool:
    return now.weekday() < 5


def is_open_trading_day(now: datetime) -> bool:
    if not is_weekday_market_day(now):
        return False

    holiday_fn = getattr(dsf, "chk_holiday", None)
    if not callable(holiday_fn):
        return True

    try:
        df = holiday_fn(bass_dt=now.strftime("%Y%m%d"))
    except Exception as exc:
        log(f"WARNING: chk_holiday failed, fallback to weekday: {exc}")
        return True

    if df is None or df.empty:
        return True

    date_str = now.strftime("%Y%m%d")
    if "bass_dt" in df.columns:
        row = df[df["bass_dt"].astype(str) == date_str]
        if row.empty:
            row = df.iloc[[0]]
    else:
        row = df.iloc[[0]]

    flag = _is_truthy_flag(row.iloc[-1].get("opnd_yn"))
    return True if flag is None else flag


def is_regular_session(now: datetime) -> bool:
    return REGULAR_START <= now.time() <= REGULAR_END


def is_regular_call_auction(now: datetime) -> bool:
    current_time = now.time()
    return REGULAR_NEW_ENTRY_CUTOFF <= current_time < REGULAR_END


def is_nxt_session(now: datetime) -> bool:
    current_time = now.time()
    return (MORNING_NXT_START <= current_time <= MORNING_NXT_END) or (AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END)


def classify_buy_session(now: datetime) -> str:
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
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return nxt_tradeable
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return nxt_tradeable
    return False


def is_new_entry_allowed(now: datetime, nxt_tradeable: bool) -> bool:
    if is_regular_session(now):
        return now.time() < REGULAR_NEW_ENTRY_CUTOFF
    if is_nxt_session(now) and nxt_tradeable:
        return now.time() < AFTERNOON_NXT_NEW_ENTRY_CUTOFF or now.time() <= MORNING_NXT_END
    return False


def get_order_spec(now: datetime, nxt_tradeable: bool) -> dict | None:
    if is_regular_session(now):
        return {"exchange": "KRX", "ord_dvsn": "01", "ord_unpr": "0"}
    if is_nxt_session(now) and nxt_tradeable:
        return {"exchange": "NXT", "ord_dvsn": "00", "ord_unpr": None}
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
            # fetch ?�점 기�??�로 ?�정 ?�료??3분봉까�?�??�용
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
# 지??계산
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

    # RSI - Wilder's smoothing (EWM alpha=1/period, ?�순?�동?�균보다 ?�확)
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

    # VWAP - ?�일 ?�적 거래??가�??�균가 (기�? ?�균매수가 ?��?지?? ?�트?�데??리셋)
    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))

    # OBV - 거래량 방향성(급등 방향, 가격 추세 확인)
    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


# ---------------------------------------------------------------------------
# ?�략 로직
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
        # OVERBOUGHT_*, NEAR_BB_UPPER_*, LOW_VOLUME_*, WEAK_TREND_ADX_* ??
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

    # 1) Stoch: K가 D�??�향 ?�파 + 20~50 구간 + ?�승 �?
    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if (k_p <= d_p and k_c > d_c) and (STOCH_BUY_MIN <= k_c <= STOCH_BUY_MAX) and (k_c > k_p):
            score += 1

    # 2) RSI: 50 ?�상 70 미만 + ?�승 �?
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

    # 5) VWAP: ?�재가 > VWAP (?�일 ?�급 ?�위, 기�? ?�균매수가 ??
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

    # 4) MACD: ?�드?�로??(?�향 ?�환)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if macd_p >= msig_p and macd_c < msig_c:
            score += 1

    # 5) OBV: OBV < OBV_MA �??�락 �?(거래??방향??매도 ?�위)
    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_c < obv_ma_c and obv_c < obv_p:
            score += 1

    return score


def _near_cross_momentum_flags(cur: pd.Series, prev: pd.Series) -> dict[str, float | bool]:
    """Builds 2-stage near-cross flags: ARM candidate and early-entry candidate."""
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
        self.trade_lock_until: dict[str, datetime] = {}
        self._last_sync_at: datetime | None = None
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
                "buy_price": float(prev.get("buy_price", avg_price)),
                "quantity": qty,
                "buy_time": prev.get("buy_time", datetime.now()),
                "buy_session": prev.get("buy_session", "synced"),
                "current_price": current_price,
                "highest_price": float(prev.get("highest_price", current_price)),
            }

        self.positions = updated

    def get_open_positions(self) -> dict[str, dict]:
        return self.positions

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

        # ?�답 ?�맷 차이�??�비해 ?�사 컬럼명도 ?�용
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
        if qty <= 0 or self._in_cooldown(code, now):
            return False

        # 주문 직전 ?�점??주문가?�수?�으�??�보??(50만원 ?�도?� ?�시 ?�용)
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

        fill_price = _extract_order_price(order_result) or price
        self.positions[code] = {
            "buy_price": float(fill_price),
            "quantity": int(qty),
            "buy_time": now,
            "buy_session": session,
            "current_price": float(fill_price),
            "highest_price": float(fill_price),
        }
        self._mark_trade_lock(code, now)
        detail_suffix = f" | {buy_detail}" if buy_detail else ""
        code_label = _format_code_label(code, code_name)
        log(f"BUY success | {code_label} | qty={qty} | price={fill_price:,.0f} | session={session} | exch={order_spec['exchange']}{detail_suffix}")
        log_trade(f"BUY success | {code_label} | qty={qty} | price={fill_price:,.0f} | session={session} | exch={order_spec['exchange']}{detail_suffix}")
        _log_trade_event_banner(
            event="BUY EXECUTED",
            code=code,
            qty=int(qty),
            price=float(fill_price),
            detail=f"session={session} exch={order_spec['exchange']}" + (f" | {buy_detail}" if buy_detail else ""),
            code_name=code_name,
        )
        return True

    def place_sell_order(self, code: str, qty: int, now: datetime, reason: str, nxt_tradeable: bool, price: float | None = None, code_name: str = "") -> bool:
        self.sync_positions_from_account(force=True)
        pos = self.positions.get(code)
        if not pos or pos.get("quantity", 0) <= 0:
            return False

        qty = min(int(qty), int(pos["quantity"]))
        if qty <= 0 or self._in_cooldown(code, now):
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

        fill_price = _extract_order_price(order_result) or current_price
        remaining = int(pos["quantity"]) - qty
        if remaining <= 0:
            self.positions.pop(code, None)
        else:
            pos["quantity"] = remaining
            pos["current_price"] = fill_price
            pos["highest_price"] = max(float(pos.get("highest_price", fill_price)), float(fill_price))

        self._mark_trade_lock(code, now)
        pnl_pct = ((fill_price / float(pos["buy_price"])) - 1.0) * 100.0
        code_label = _format_code_label(code, code_name)
        log(f"SELL success | {code_label} | qty={qty} | price={fill_price:,.0f} | pnl={pnl_pct:.2f}% | reason={reason} | exch={order_spec['exchange']}")
        log_trade(f"SELL success | {code_label} | qty={qty} | price={fill_price:,.0f} | pnl={pnl_pct:.2f}% | reason={reason} | exch={order_spec['exchange']}")
        _log_trade_event_banner(
            event="SELL EXECUTED",
            code=code,
            qty=int(qty),
            price=float(fill_price),
            detail=f"pnl={pnl_pct:.2f}% reason={reason} exch={order_spec['exchange']}",
            code_name=code_name,
        )
        return True


# ---------------------------------------------------------------------------
# ?�약 �?��
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
            if pos.get("buy_time", current_dt).date() != trade_date:
                continue
            if pos.get("buy_session") not in ("regular", "morning_nxt"):
                continue
            price = float(pos.get("current_price") or pos["buy_price"])
            buy_price = float(pos.get("buy_price") or 0)
            if buy_price <= 0 or price <= buy_price:
                log(f"  [CALL_AUCTION HOLD] {code} | NOT_IN_PROFIT | price={price:,.0f} buy={buy_price:,.0f}")
                continue
            api.trade_lock_until.pop(code, None)
            api.place_sell_order(code, int(pos["quantity"]), current_dt, "CALL_AUCTION_TAKE_PROFIT_1520", nxt_map.get(code, False), price=price, code_name=watch_map.get(code, ""))

    if not state["done_1959"] and current_time >= AFTERNOON_NXT_FORCE_EXIT:
        state["done_1959"] = True
        for code, pos in list(api.get_open_positions().items()):
            if pos.get("buy_time", current_dt).date() != trade_date:
                continue
            if pos.get("buy_session") != "afternoon_nxt":
                continue
            price = float(pos.get("current_price") or pos["buy_price"])
            api.trade_lock_until.pop(code, None)
            api.place_sell_order(code, int(pos["quantity"]), current_dt, "EOD_NXT_1959", nxt_map.get(code, False), price=price, code_name=watch_map.get(code, ""))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run(target_date: str | None = None) -> None:
    ka.auth()
    now = datetime.now()

    if not is_open_trading_day(now):
        log(f"Today is not an open trading day: {now:%Y-%m-%d %A}")
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

    log(f"Watchlist source: {watch_file}")

    nxt_map = {code: is_nxt_tradeable(code) for code in watch_map}
    for code, name in watch_map.items():
        log(f"WATCH | {code} | {name} | NXT={nxt_map[code]}")

    log("Strategy: live price cross over buffered BB middle + Stoch/RSI/Williams confirmation")
    log(
        f"Live-cross filter: BB buffer={LIVE_PRICE_BB_BUFFER_PCT*100:.3f}% | "
        f"confirm polls={LIVE_PRICE_CROSS_CONFIRM_POLLS} | confirm seconds={LIVE_PRICE_CROSS_CONFIRM_SECONDS}"
    )
    log(f"TP={TAKE_PROFIT_PERCENT*100:.1f}% | SL={STOP_LOSS_PERCENT*100:.1f}% | Trail={TRAILING_STOP_FROM_PEAK*100:.1f}%")

    api = TradingAPI()
    traded_today: set[str] = set()
    signal_buy_bar: dict[str, object] = {}
    signal_sell_bar: dict[str, object] = {}
    live_price_cross_state: dict[str, dict] = {}
    liquidation_state: dict = {}
    current_trade_date = now.date()
    startup_time = datetime.now()
    log(f"Startup warmup: new entries blocked for {STARTUP_WARMUP_SECONDS}s until {(startup_time + timedelta(seconds=STARTUP_WARMUP_SECONDS)):%H:%M:%S}")

    while True:
        current_dt = datetime.now()

        if current_dt.date() != current_trade_date:
            current_trade_date = current_dt.date()
            traded_today.clear()
            signal_buy_bar.clear()
            signal_sell_bar.clear()
            live_price_cross_state.clear()

        if not is_open_trading_day(current_dt):
            log("Market closed day. Stopping.")
            break

        if current_dt.time() >= AFTERNOON_NXT_END:
            run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state)
            log("20:00 reached. Stopping.")
            break

        if not is_regular_session(current_dt) and not is_nxt_session(current_dt):
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        api.sync_positions_from_account(force=False)
        run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state)

        # 15:20~15:30 정규장 마감 구간은 종목별 매도 체크 건너뛰고
        # ?�시?��? �?�� 로직(?�일 매수 + ?�익 구간)�??�행?�다.
        if is_regular_call_auction(current_dt):
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        for code, name in watch_map.items():
            nxt_tradeable = nxt_map.get(code, False)
            symbol_label = _symbol_log_label(code, name)
            if not can_trade_code_now(current_dt, nxt_tradeable):
                log(f"  [SKIP] {symbol_label} | can_trade_code_now=False | time={current_dt:%H:%M:%S} nxt={nxt_tradeable}")
                continue

            try:
                frame = fetch_3min_frame(code, current_dt, nxt_tradeable)
            except Exception as exc:
                log(f"{code} frame error: {exc}")
                continue

            if frame is None:
                log(f"  [SKIP] {symbol_label} | frame=None (fetch failed)")
                continue
            if len(frame) < MIN_BARS_REQUIRED:
                log(f"  [SKIP] {symbol_label} | bars={len(frame)} < MIN_BARS_REQUIRED={MIN_BARS_REQUIRED}")
                continue

            bar_time = frame.index[-1]
            last_closed_bar = pd.Timestamp(current_dt).floor("3min")
            bar_age_sec = max(0.0, (pd.Timestamp(current_dt) - bar_time).total_seconds())
            cur = frame.iloc[-1]
            price = fetch_live_price(code, current_dt, nxt_tradeable)
            if price is None or price <= 0:
                price = float(cur["close"])

            cross_info = update_live_price_cross_state(
                live_price_cross_state,
                code,
                current_dt,
                float(price),
                _num(cur, "BB_MIDDLE"),
            )
            pos = api.get_open_positions().get(code)

            log(
                f"  [CHECK] {symbol_label} | {current_dt:%H:%M:%S} | bars={len(frame)} live={price:,.0f} bar_close={float(cur['close']):,.0f} | "
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

            if pos is not None and pos.get("quantity", 0) > 0:
                entry_price = float(pos["buy_price"])
                pnl_pct = (price / entry_price) - 1.0

                # 저가(bar low)가 현재가 보다 낮은 값이면 손절 처리 우선
                # ??20�??�링 간격?�서 ?�?�을 ?�쳐 ?�절??지?�되??문제 방�?
                _bar_low = float(cur["low"]) if "low" in cur.index and not pd.isna(cur["low"]) else price
                _sl_price = min(price, _bar_low)
                _pnl_sl = (_sl_price / entry_price) - 1.0

                pos["current_price"] = price
                pos["highest_price"] = max(float(pos.get("highest_price", price)), price)
                highest_price = float(pos["highest_price"])  # Fix: TP/?�레?�링 로직보다 먼�? ?�의

                if pnl_pct >= TAKE_PROFIT_PERCENT:
                    if ENABLE_TP_EXTENSION_TRAILING:
                        # TP ?�달 ??즉시 ?�절 ?�??고점 ?�레?�링 모드�??�환
                        log(
                            f"  [TP_EXTENSION] {code} | pnl={pnl_pct*100:.2f}% >= TP {TAKE_PROFIT_PERCENT*100:.1f}% | "
                            f"고점 ?�레?�링 모드 ?�환 (trail={TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%) | "
                            f"price={price:,.0f} peak={highest_price:,.0f}"
                        )
                    else:
                        reason_tp = f"TAKE_PROFIT_+{TAKE_PROFIT_PERCENT*100:.1f}%"
                        log(f"  [SELL TRIGGER] {code} | {reason_tp} | price={price:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}%")
                        if api.place_sell_order(code, int(pos["quantity"]), current_dt, reason_tp, nxt_tradeable, price=price, code_name=name):
                            log(f"  [SELL EXECUTED] {code} | {reason_tp} | qty={pos['quantity']} price={price:,.0f}")
                        signal_sell_bar[code] = bar_time
                        continue

                _held_sl = (current_dt - pos.get("buy_time", current_dt)).total_seconds()
                _sl_threshold = STOP_LOSS_EARLY_PERCENT if _held_sl < STOP_LOSS_MIN_HOLD_SECONDS else STOP_LOSS_PERCENT
                if _pnl_sl <= _sl_threshold:
                    reason_sl = f"STOP_LOSS_EARLY_{_sl_threshold*100:.1f}%" if _held_sl < STOP_LOSS_MIN_HOLD_SECONDS else f"STOP_LOSS_{_sl_threshold*100:.1f}%"
                    log(f"  [SELL TRIGGER] {code} | {reason_sl} | held={_held_sl:.0f}s price={price:,.0f} bar_low={_bar_low:,.0f} entry={entry_price:,.0f} pnl={pnl_pct*100:.2f}% sl_pnl={_pnl_sl*100:.2f}%")
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
                    peak_pnl_pct = (highest_price / entry_price) - 1.0
                    current_pnl_pct = (price / entry_price) - 1.0
                    profit_giveback = peak_pnl_pct - current_pnl_pct
                    if ENABLE_TP_EXTENSION_TRAILING and peak_pnl_pct >= TAKE_PROFIT_PERCENT:
                        trail_threshold = TP_EXTENSION_TRAIL_FROM_PEAK
                        reason_ts = f"TP_EXTENSION_TRAIL_{TP_EXTENSION_TRAIL_FROM_PEAK*100:.1f}%"
                    else:
                        trail_threshold = TRAILING_STOP_FROM_PEAK
                        reason_ts = f"TRAILING_STOP_GIVEBACK_{TRAILING_STOP_FROM_PEAK*100:.1f}%"
                    if peak_pnl_pct > 0 and current_pnl_pct > 0 and profit_giveback >= trail_threshold:
                        log(
                            f"  [SELL TRIGGER] {code} | {reason_ts} | "
                            f"price={price:,.0f} entry={entry_price:,.0f} peak={highest_price:,.0f} | "
                            f"pnl={current_pnl_pct*100:.2f}% peak_pnl={peak_pnl_pct*100:.2f}% giveback={profit_giveback*100:.2f}%"
                        )
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
                if not is_new_entry_allowed(current_dt, nxt_tradeable):
                    continue
                _warmup_elapsed = (current_dt - startup_time).total_seconds()
                if _warmup_elapsed < STARTUP_WARMUP_SECONDS:
                    log(f"  [BUY REJECT] {symbol_label} | STARTUP_WARMUP | elapsed={_warmup_elapsed:.0f}s / {STARTUP_WARMUP_SECONDS}s")
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

                qty = api.get_affordable_buy_qty(code, price, current_dt, nxt_tradeable)
                if qty <= 0:
                    log(f"  [BUY REJECT] {symbol_label} | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price={price:,.0f}")
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

                log("=" * 110)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    args = _parse_args()
    run(target_date=args.date)

