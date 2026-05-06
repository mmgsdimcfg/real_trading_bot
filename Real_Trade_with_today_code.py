# -*- coding: utf-8 -*-
"""R73 real trading script - live price / BB middle cross strategy with multi indicators.

Core idea:
1) Buy when live price crosses above BB middle and stays there long enough.
2) Sell when live price crosses below BB middle and stays there long enough.
3) Use Stochastic Fast, RSI, Williams %R as confirmation filters.
4) Use take-profit, stop-loss, and trailing-stop for risk control.

Note: This script cannot guarantee profit. Always paper-test before live trading.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional

import pandas as pd

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


TODAY_CODE_FILE = current_dir / "define_today_code.txt"
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
OBV_MA_PERIOD = 10      # OBV 이동평균 기간 (방향성 신호선)
MACD_FAST = 5           # 빠른 EMA 기간 (API 30봉 제약 고려)
MACD_SLOW = 12          # 느린 EMA 기간
MACD_SIGNAL_PERIOD = 4  # MACD 시그널 기간
ADX_PERIOD = 7          # ADX/DI 계산 기간 (단기 추세 강도)
ADX_MIN_TREND = 20.0    # ADX 이 값 미만 = 횡보 구간 → 신규 매수 보류
ADX_STRONG_TREND = 40.0  # 강한 추세 구간에서는 거래량 기준 완화

# ---------------------------------------------------------------------------
# 매매 파라미터
# ---------------------------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000
TAKE_PROFIT_PERCENT = 0.035
STOP_LOSS_PERCENT = -0.012
STOP_LOSS_EARLY_PERCENT = -0.020     # 진입 초기(STOP_LOSS_MIN_HOLD_SECONDS 이내) 노이즈 방지용 넓은 손절
STOP_LOSS_MIN_HOLD_SECONDS = 600    # 이 시간(초) 미만이면 EARLY 손절 기준 적용 (기본 10분)
STARTUP_WARMUP_SECONDS = 180        # 스크립트 시작 후 이 시간(초) 동안 신규 매수 없음 (지표 안정화 + 모니터링 여유 보장)
TRAILING_STOP_FROM_PEAK = 0.005
# 보조지표 기반 매도(AUX_REVERSAL) 최소 수익률 게이트
# 점수가 낮을수록 더 높은 수익률일 때만 매도 허용
AUX_SELL_MIN_PNL_SCORE2 = 0.010  # score=2  -> +1.0% 이상
AUX_SELL_MIN_PNL_SCORE3 = 0.005  # score=3  -> +0.5% 이상
AUX_SELL_MIN_PNL_SCORE4 = 0.000  # score>=4 -> 손익분기 이상
# MA5 데드크로스 매도 디바운스: 휩쏘 방지를 위해 일반 구간은 1개 바 확인 후 매도
# 단, 손실/약세 강도가 크면 즉시 매도해 급락 리스크는 방어
MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007     # -0.7% 이하 손실이면 즉시 매도 허용
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2         # 보조 약세 점수 높으면 즉시 매도 허용
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1       # 다음 바 확인 매도 최소 보조 점수
MA5_BB_DOWN_CROSS_MIN_PNL = 0.000             # 데드크로스 계열 매도는 최소 손익(기본 0%=본전) 이상에서만 허용
# 박스권 구간에서는 기술적 매도 신호를 보류하고, 손절/익절(및 트레일링)만 허용
ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
# 익절 기준 도달 후 즉시 매도 대신 고점 트레일링 모드로 전환
# (강모멘텀 급등 시 추가 수익 구간 확보)
ENABLE_TP_EXTENSION_TRAILING = True
TP_EXTENSION_TRAIL_FROM_PEAK = 0.010     # TP 연장 구간에서 고점 대비 1.0% 하락 시 매도
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065         # 최근 N개 봉 고저폭이 현재가 대비 0.65% 이내
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080      # 볼린저 밴드 폭이 현재가 대비 0.80% 이내
# 매수 2단계(근접 ARM -> 확정/강모멘텀 진입) 파라미터
NEAR_CROSS_ARM_GAP_MAX = 0.0015        # MA5가 BB_MIDDLE 아래에 있더라도 gap 0.15% 이내면 ARM 후보
NEAR_CROSS_ARM_MA_RISE_MIN = 0.0006    # ARM 최소 MA5 상승률 (0.06%)
NEAR_CROSS_EARLY_GAP_MAX = 0.0006      # ARM 이후 조기진입 허용 gap 0.06% 이내
NEAR_CROSS_EARLY_MA_RISE_MIN = 0.0010  # ARM 이후 조기진입 최소 MA5 상승률 (0.10%)
NEAR_CROSS_ARM_EXPIRE_BARS = 2         # ARM 유효 기간(3분봉 기준 바 개수)
# 기본 원칙은 MA5 상향돌파 우선. 예외 경로는 특정 시간창에서만 제한적으로 허용.
ENABLE_NEAR_CROSS_ARM = True
ENABLE_EARLY_NEAR_CROSS_ENTRY = True
EARLY_NEAR_CROSS_ALLOWED_START = dt_time(9, 0)
EARLY_NEAR_CROSS_ALLOWED_END = dt_time(11, 30)
EARLY_NEAR_CROSS_ALLOW_NXT = False
# 조기진입(ARM/EARLY) 유동성 필터: 소량 체결 급등으로 인한 오진입 방지
EARLY_NEAR_CROSS_MIN_VOLUME = 800
EARLY_NEAR_CROSS_MIN_VOL_MA = 500
EARLY_NEAR_CROSS_MIN_TURNOVER_KRW = 5_000_000
POLL_INTERVAL_SECONDS = 20
LIVE_PRICE_BB_BUFFER_PCT = 0.0008      # 현재가가 BB 중간선 대비 0.08% 이상 넘어야 유효 돌파로 인정
LIVE_PRICE_CROSS_CONFIRM_POLLS = 3      # 20초 폴링 기준 3회 연속 확인 (상향 크로스 / 매수)
LIVE_PRICE_CROSS_CONFIRM_SECONDS = 60   # 첫 감지 후 최소 60초 유지 (상향 크로스 / 매수)
# 하향 크로스(매도)는 즉시 반응 — 대기 시간을 두면 급락 구간에서 매도 기회를 놓침
LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS = 1     # 하향 크로스: 1회 감지 즉시 신호 발동
LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS = 0   # 하향 크로스: 지연 없이 즉시 확인
ACCOUNT_SYNC_INTERVAL_SECONDS = 90
MIN_BARS_REQUIRED = 3  # 이전봉·현재봉 비교에 최소 3개 필요 (지표는 min_periods=1로 자체 보완)
ALLOW_REBUY_SAME_CODE = False
TRADE_COOLDOWN_MINUTES = 3

# ---------------------------------------------------------------------------
# 세션 / 시간 상수
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

# 보조지표 임계값
STOCH_OVERBOUGHT = 80.0
STOCH_BUY_MAX = 72.0
RSI_BUY_MIN = 45.0
RSI_BUY_MAX = 72.0
WILLIAMS_BUY_FLOOR = -70.0
WILLIAMS_OVERBOUGHT_CEIL = -20.0
BB_UPPER_PROXIMITY_MAX = 0.85  # (close-bb_lower)/(bb_upper-bb_lower) 값이 크면 상단 추격 구간

# 거래량 필터(시간대/세션 가변)
VOLUME_RATIO_OPEN = 0.80       # 장초반(09:00~10:00)
VOLUME_RATIO_MIDDAY = 0.60     # 장중(10:00~14:30)
VOLUME_RATIO_CLOSE = 0.70      # 장후반(14:30~15:30)
VOLUME_RATIO_NXT = 0.55        # NXT 세션
VOLUME_RATIO_STRONG_RELAX = 0.10  # ADX 강추세 시 완화 폭
VOLUME_RATIO_FLOOR = 0.50      # 완화 하한

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

# domestic_stock_functions / inquire_time_itemchartprice 등 라이브러리에서
# 출력하는 "Data fetch complete.", "Call Next page...", "Max recursive depth reached."
# 로그를 억제 (루트 로거는 INFO지만 외부 모듈은 WARNING 이상만 표시)
logging.getLogger("domestic_stock_functions").setLevel(logging.WARNING)
logging.getLogger("inquire_time_itemchartprice").setLevel(logging.WARNING)
# 위 두 모듈이 module-level logging.info/warning 으로 루트 로거를 직접 사용하는 경우
# 루트 로거 자체는 낮추되 핸들러 레벨로 필터링할 수 없으므로, 대신 아래 필터를 추가
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
    parser = argparse.ArgumentParser(description="R73 real trading runner")
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
# NXT 가능 종목 탐지
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

    # 추세가 매우 강하면 거래량 필터를 소폭 완화
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
# 데이터 수신
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


def update_live_price_cross_state(
    cross_state: dict[str, dict],
    code: str,
    now: datetime,
    live_price: float,
    bb_middle: float,
) -> dict[str, object]:
    relation = "on"
    upper_trigger = bb_middle * (1.0 + LIVE_PRICE_BB_BUFFER_PCT)
    lower_trigger = bb_middle * (1.0 - LIVE_PRICE_BB_BUFFER_PCT)

    if live_price >= upper_trigger:
        relation = "above"
    elif live_price <= lower_trigger:
        relation = "below"

    tracker = cross_state.get(code)
    if tracker is None:
        tracker = {
            "confirmed_relation": relation if relation in {"above", "below"} else None,
            "pending": None,
        }
        cross_state[code] = tracker
        return {
            "relation": relation,
            "confirmed_relation": tracker["confirmed_relation"],
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    confirmed_relation = tracker.get("confirmed_relation")
    pending = tracker.get("pending")

    if relation not in {"above", "below"}:
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    if relation == confirmed_relation:
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    if pending is None or pending.get("side") != relation:
        tracker["pending"] = {"side": relation, "started_at": now, "count": 1}
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": relation,
            "pending_count": 1,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    pending["count"] = int(pending.get("count", 0)) + 1
    pending_seconds = max(0.0, (now - pending["started_at"]).total_seconds())
    signal = None

    # 방향별 확인 임계값 선택: 하향(매도)은 즉시, 상향(매수)은 충분히 확인
    if relation == "below":
        req_polls = LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS
        req_seconds = LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS
    else:
        req_polls = LIVE_PRICE_CROSS_CONFIRM_POLLS
        req_seconds = LIVE_PRICE_CROSS_CONFIRM_SECONDS

    if pending["count"] >= req_polls and pending_seconds >= req_seconds:
        tracker["confirmed_relation"] = relation
        tracker["pending"] = None
        signal = "cross_up" if relation == "above" else "cross_down"
        return {
            "relation": relation,
            "confirmed_relation": relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": pending_seconds,
            "signal": signal,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    return {
        "relation": relation,
        "confirmed_relation": confirmed_relation,
        "pending_side": relation,
        "pending_count": int(pending["count"]),
        "pending_seconds": pending_seconds,
        "signal": None,
        "upper_trigger": upper_trigger,
        "lower_trigger": lower_trigger,
    }


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

    # MACD - 단기 모멘텀 방향성 (MA 크로스 전략 핵심 확인 지표)
    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    # ADX / DI - 추세 강도 (Wilder's smoothing, 횡보장 오신호 필터)
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

    # VWAP - 당일 누적 거래량 가중 평균가 (기관 평균매수가 대리 지표, 인트라데이 리셋)
    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))

    # OBV - 거래량 방향성 누적 (수급 이동 방향, 가격 선행 지표)
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


def _buy_reject_detail(buy_reason: str, cur: pd.Series, prev: pd.Series, live_price: float | None = None, cross_info: dict | None = None) -> str:
    """Returns reject reason + indicator context for every reject type."""
    prev_ma5 = _num(prev, "MA_5");  cur_ma5  = _num(cur,  "MA_5")
    prev_bb  = _num(prev, "BB_MIDDLE"); cur_bb = _num(cur, "BB_MIDDLE")

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
                f"seconds={pending_seconds:.0f}/{LIVE_PRICE_CROSS_CONFIRM_SECONDS}"
            )
        return (
            f"{buy_reason} | "
            f"{live_part}BB_MID {prev_bb:.1f}→{cur_bb:.1f} "
            f"(need live price to stay above BB middle long enough)"
        )

    if buy_reason == "BB_MIDDLE_FALLING":
        return f"{buy_reason} | BB_MID {prev_bb:.1f}→{cur_bb:.1f}"

    if buy_reason == "MA5_FALLING":
        return f"{buy_reason} | MA5 {prev_ma5:.1f}→{cur_ma5:.1f}"

    if buy_reason == "NOT_BULLISH":
        close_v = _num(cur, "close"); open_v = _num(cur, "open")
        return f"{buy_reason} | close={close_v:.0f} open={open_v:.0f}"

    if buy_reason == "MISSING_INDICATOR":
        missing = [k for k in ("MA_5", "BB_MIDDLE") if pd.isna(_num(cur, k)) or pd.isna(_num(prev, k))]
        return f"{buy_reason} | NaN={','.join(missing)}"

    if not buy_reason.startswith("LOW_SCORE"):
        # OVERBOUGHT_*, NEAR_BB_UPPER_*, LOW_VOLUME_*, WEAK_TREND_ADX_* —
        # reason string already contains the offending value; no extra context needed
        return buy_reason

    # LOW_SCORE: show which of the 6 sub-indicators failed
    failed = []

    k_c = _num(cur, "STOCH_K"); d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K"); d_p = _num(prev, "STOCH_D")
    if any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)) or not (
        (k_p <= d_p and k_c > d_c) or (k_c > d_c and k_c <= STOCH_BUY_MAX)
    ):
        failed.append("STOCH")

    rsi_c = _num(cur, "RSI"); sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI"); sig_p = _num(prev, "RSI_SIGNAL")
    if any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)) or not (
        (rsi_p <= sig_p and rsi_c > sig_c) or (rsi_c > sig_c and RSI_BUY_MIN <= rsi_c <= RSI_BUY_MAX)
    ):
        failed.append("RSI")

    wr_c = _num(cur, "WILLIAMS_R"); wr_p = _num(prev, "WILLIAMS_R")
    if pd.isna(wr_c) or pd.isna(wr_p) or not (wr_c > wr_p and wr_c >= WILLIAMS_BUY_FLOOR):
        failed.append("WILLIAMS_R")

    macd_c = _num(cur, "MACD"); msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD"); msig_p = _num(prev, "MACD_SIGNAL")
    if any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)) or not (
        (macd_p <= msig_p and macd_c > msig_c) or (macd_c > msig_c and macd_c > 0)
    ):
        failed.append("MACD")

    vwap = _num(cur, "VWAP"); close_v = _num(cur, "close")
    if pd.isna(vwap) or pd.isna(close_v) or not (close_v > vwap):
        failed.append("VWAP")

    obv_c = _num(cur, "OBV"); obv_ma_c = _num(cur, "OBV_MA"); obv_p = _num(prev, "OBV")
    if any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)) or not (obv_c > obv_ma_c or obv_c > obv_p):
        failed.append("OBV")

    suffix = f" | FAILED={','.join(failed)}" if failed else ""
    return f"{buy_reason}{suffix}"


def _buy_support_score(cur: pd.Series, prev: pd.Series) -> int:
    score = 0

    # 1) Stoch: 골든크로스 또는 K>D with 과열 아닌 구간
    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if (k_p <= d_p and k_c > d_c) or (k_c > d_c and k_c <= STOCH_BUY_MAX):
            score += 1

    # 2) RSI: Signal 상향 또는 RSI 중심 이상
    rsi_c = _num(cur, "RSI")
    sig_c = _num(cur, "RSI_SIGNAL")
    rsi_p = _num(prev, "RSI")
    sig_p = _num(prev, "RSI_SIGNAL")
    if not any(pd.isna(v) for v in (rsi_c, sig_c, rsi_p, sig_p)):
        in_buy_zone = RSI_BUY_MIN <= rsi_c <= RSI_BUY_MAX
        if (rsi_p <= sig_p and rsi_c > sig_c) or (rsi_c > sig_c and in_buy_zone):
            score += 1

    # 3) Williams %R: 상승 전환 and 과매도 회복
    wr_c = _num(cur, "WILLIAMS_R")
    wr_p = _num(prev, "WILLIAMS_R")
    if not pd.isna(wr_c) and not pd.isna(wr_p):
        if wr_c > wr_p and wr_c >= WILLIAMS_BUY_FLOOR:
            score += 1

    # 4) MACD: 골든크로스 또는 MACD > Signal (양의 모멘텀)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if (macd_p <= msig_p and macd_c > msig_c) or (macd_c > msig_c and macd_c > 0):
            score += 1

    # 5) VWAP: 현재가 > VWAP (당일 수급 우위, 기관 평균매수가 위)
    vwap = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not pd.isna(vwap) and not pd.isna(close_v) and close_v > vwap:
        score += 1

    # 6) OBV: OBV > OBV_MA 또는 상승 중 (거래량 방향성 매수 우위)
    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_c > obv_ma_c or obv_c > obv_p:
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

    # 4) MACD: 데드크로스 (하향 전환)
    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if macd_p >= msig_p and macd_c < msig_c:
            score += 1

    # 5) OBV: OBV < OBV_MA 및 하락 중 (거래량 방향성 매도 우위)
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
    if len(frame) < 3:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")

    if any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)):
        return False, "MISSING_INDICATOR"

    if cross_info.get("signal") != "cross_up":
        return False, "NO_LIVE_PRICE_BB_CROSS_UP"

    if live_price <= cur_bb:
        return False, "LIVE_PRICE_NOT_ABOVE_BB_MIDDLE"

    # 실시간가뿐 아니라 봉 종가도 BB_MIDDLE 위여야 신규 매수 허용
    close_val = _num(cur, "close")
    if pd.isna(close_val) or close_val <= cur_bb:
        return False, "BAR_CLOSE_NOT_ABOVE_BB_MIDDLE"

    # MA5가 BB_MIDDLE 아래에 있으면 이미 데드크로스 상태 → 매수 차단
    if cur_ma5 <= cur_bb:
        return False, "MA5_AT_OR_BELOW_BB_MIDDLE"

    # MA5가 BB_MIDDLE 위에 있어도 gap이 극히 작고 MA5가 하락 중이면 데드크로스 임박 → 매수 차단
    _ma5_bb_gap_pct = (cur_ma5 - cur_bb) / max(cur_bb, 1.0)
    if _ma5_bb_gap_pct < NEAR_CROSS_ARM_GAP_MAX and cur_ma5 <= prev_ma5:
        return False, f"MA5_DEAD_CROSS_IMMINENT_{_ma5_bb_gap_pct*100:.3f}%"

    # 모멘텀 방향 확인
    if cur_bb < prev_bb:
        return False, "BB_MIDDLE_FALLING"
    if cur_ma5 < prev_ma5:
        return False, "MA5_FALLING"

    # 현재가 기준으로도 시가 위에 있어야 추격 매수 위험이 낮음
    if live_price <= float(cur["open"]):
        return False, "NOT_BULLISH"

    # 과열 추격 매수 방지 1) Stochastic 과열 구간
    stoch_k = _num(cur, "STOCH_K")
    if not pd.isna(stoch_k) and stoch_k >= STOCH_OVERBOUGHT:
        return False, f"OVERBOUGHT_STOCH_{stoch_k:.1f}"

    # 과열 추격 매수 방지 2) Williams %R 과열(0에 가까울수록 과열)
    wr_val = _num(cur, "WILLIAMS_R")
    if not pd.isna(wr_val) and wr_val >= WILLIAMS_OVERBOUGHT_CEIL:
        return False, f"OVERBOUGHT_WR_{wr_val:.1f}"

    # 과열 추격 매수 방지 3) 볼린저 상단 과근접
    bb_up = _num(cur, "BB_UPPER")
    bb_low = _num(cur, "BB_LOWER")
    if not any(pd.isna(v) for v in (bb_up, bb_low, close_val)) and bb_up > bb_low:
        bb_pos = (close_val - bb_low) / (bb_up - bb_low)
        if bb_pos >= BB_UPPER_PROXIMITY_MAX:
            return False, f"NEAR_BB_UPPER_{bb_pos:.2f}"

    adx_val = _num(cur, "ADX")

    # 거래량 필터: 급감 구간 진입 회피
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        ratio = get_volume_ratio_threshold(now, adx_val)
        if vol < (vol_ma * ratio):
            return False, f"LOW_VOLUME_{(vol / vol_ma):.2f}_LT_{ratio:.2f}"

    # ADX 추세 강도 필터: 횡보장에서 MA크로스는 오신호 빈발
    if not pd.isna(adx_val) and adx_val < ADX_MIN_TREND:
        return False, f"WEAK_TREND_ADX_{adx_val:.1f}"

    support_score = _buy_support_score(cur, prev)
    if support_score < 3:
        return False, f"LOW_SCORE_{support_score}"

    return True, f"LIVE_PRICE_BB_UP_CROSS_SCORE_{support_score}"


def check_sell_condition(frame: pd.DataFrame, pnl_pct: float, live_price: float, cross_info: dict[str, object]) -> tuple[bool, str]:
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]

    prev_ma5 = _num(prev, "MA_5")
    cur_ma5 = _num(cur, "MA_5")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")

    # ── Fix 2 & 4: MA5 데드크로스 (bar 레벨 판단) ──────────────────────────────
    # BOX_RANGE_HOLD 체크보다 먼저 수행 → 박스권에서도 데드크로스는 즉시 매도 판단
    if not any(pd.isna(v) for v in (prev_ma5, cur_ma5, prev_bb, cur_bb)):
        ma5_dead_cross = (prev_ma5 >= prev_bb) and (cur_ma5 < cur_bb)
        if ma5_dead_cross:
            if pnl_pct < MA5_BB_DOWN_CROSS_MIN_PNL:
                return False, (
                    f"MA5_BB_DEAD_CROSS_BLOCKED_PNL_{pnl_pct*100:.2f}%"
                    f"_LT_{MA5_BB_DOWN_CROSS_MIN_PNL*100:.2f}%"
                )
            score = _sell_support_score(cur, prev)
            # 즉시 매도: 손실 크거나 보조 약세 점수 충분
            if pnl_pct <= MA5_BB_DOWN_CROSS_IMMEDIATE_PNL or score >= MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE:
                return True, f"MA5_BB_DEAD_CROSS_SCORE_{score}"
            # 최소 점수 이상이면 확인 후 매도
            if score >= MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE:
                return True, f"MA5_BB_DEAD_CROSS_CONFIRM_SCORE_{score}"
            return False, f"MA5_BB_DEAD_CROSS_WEAK_SCORE_{score}"

    # ── Fix 4: BOX_RANGE_HOLD — MA5 데드크로스 이외의 기술적 매도만 보류 ──────
    if ENABLE_BOX_RANGE_HOLD_TECH_SELL and STOP_LOSS_PERCENT < pnl_pct < TAKE_PROFIT_PERCENT:
        is_box, box_info = _is_box_range_hold_zone(frame)
        if is_box:
            return False, f"BOX_RANGE_HOLD_{box_info}"

    price_cross_down = cross_info.get("signal") == "cross_down" and live_price < cur_bb

    if price_cross_down:
        if pnl_pct < MA5_BB_DOWN_CROSS_MIN_PNL:
            return False, (
                f"LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_PNL_{pnl_pct*100:.2f}%"
                f"_LT_{MA5_BB_DOWN_CROSS_MIN_PNL*100:.2f}%"
            )

        score = _sell_support_score(cur, prev)
        if pnl_pct <= MA5_BB_DOWN_CROSS_IMMEDIATE_PNL or score >= MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE:
            if score >= 1:
                return True, f"LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_{score}"
            return True, "LIVE_PRICE_BB_DOWN_CROSS"

        return False, f"LIVE_PRICE_BB_DOWN_CROSS_WEAK_SCORE_{score}"

    # 보조지표 하락 전환 + 점수별 최소 수익률 게이트
    score = _sell_support_score(cur, prev)
    min_pnl_req = None
    if score >= 4:
        min_pnl_req = AUX_SELL_MIN_PNL_SCORE4
    elif score == 3:
        min_pnl_req = AUX_SELL_MIN_PNL_SCORE3
    elif score == 2:
        min_pnl_req = AUX_SELL_MIN_PNL_SCORE2

    if min_pnl_req is not None:
        if pnl_pct >= min_pnl_req:
            return True, f"AUX_REVERSAL_SCORE_{score}"
        return False, f"AUX_BLOCKED_SCORE_{score}_PNL_{pnl_pct*100:.2f}%_LT_{min_pnl_req*100:.2f}%"

    return False, "NO_SELL_SIGNAL"


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
        log("TradingAPI (r73 MA5-BB multi-indicator) initialized")

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
            # 주문가능 응답은 왔지만 수량/금액 해석이 안 되면 과주문 방지를 위해 보수적으로 0주 처리
            log(f"WARNING: psbl-order parse failed for {code}; force qty=0 to avoid over-order")
            return 0

        return max(0, min(qty_by_budget, qty_by_psbl))

    def place_buy_order(self, code: str, price: float, qty: int, now: datetime, nxt_tradeable: bool, session: str, buy_detail: str = "", code_name: str = "") -> bool:
        if qty <= 0 or self._in_cooldown(code, now):
            return False

        # 주문 직전 시점의 주문가능수량으로 재보정 (50만원 한도와 동시 적용)
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

        # 15:20~15:30 정규장 동시호가 구간은 종목별 시그널 체크를 건너뛰고
        # 동시호가 청산 로직(당일 매수 + 수익 구간)만 수행한다.
        if is_regular_call_auction(current_dt):
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        for code, name in watch_map.items():
            nxt_tradeable = nxt_map.get(code, False)
            if not can_trade_code_now(current_dt, nxt_tradeable):
                log(f"  [SKIP] {code}({name}) | can_trade_code_now=False | time={current_dt:%H:%M:%S} nxt={nxt_tradeable}")
                continue

            try:
                frame = fetch_3min_frame(code, current_dt, nxt_tradeable)
            except Exception as exc:
                log(f"{code} frame error: {exc}")
                continue

            if frame is None:
                log(f"  [SKIP] {code}({name}) | frame=None (fetch failed)")
                continue
            if len(frame) < MIN_BARS_REQUIRED:
                log(f"  [SKIP] {code}({name}) | bars={len(frame)} < MIN_BARS_REQUIRED={MIN_BARS_REQUIRED}")
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
                f"  [CHECK] {code}({name}) | {current_dt:%H:%M:%S} | bars={len(frame)} live={price:,.0f} bar_close={float(cur['close']):,.0f} | "
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

                # 봉 내 저가(bar low)와 현재가 중 낮은 값으로 손절 트리거 판단
                # → 20초 폴링 간격에서 저점을 놓쳐 손절이 지연되는 문제 방지
                _bar_low = float(cur["low"]) if "low" in cur.index and not pd.isna(cur["low"]) else price
                _sl_price = min(price, _bar_low)
                _pnl_sl = (_sl_price / entry_price) - 1.0

                pos["current_price"] = price
                pos["highest_price"] = max(float(pos.get("highest_price", price)), price)
                highest_price = float(pos["highest_price"])  # Fix: TP/트레일링 로직보다 먼저 정의

                if pnl_pct >= TAKE_PROFIT_PERCENT:
                    if ENABLE_TP_EXTENSION_TRAILING:
                        # TP 도달 → 즉시 익절 대신 고점 트레일링 모드로 전환
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
                    # TP 연장 구간(peak >= TP)에서는 더 타이트한 1% 트레일 적용
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
                            f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}→{_num(cur, 'BB_MIDDLE'):.1f} | "
                            f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                            f"K={_num(prev_bar, 'STOCH_K'):.1f}→{_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                            f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}→{_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                            f"MACD {_num(prev_bar, 'MACD'):.2f}→{_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} | "
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
                    log(f"  [BUY REJECT] {code}({name}) | STARTUP_WARMUP | elapsed={_warmup_elapsed:.0f}s / {STARTUP_WARMUP_SECONDS}s")
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
                    detail = _buy_reject_detail(buy_reason, cur, prev_bar, live_price=price, cross_info=cross_info)
                    log(f"  [BUY REJECT] {code}({name}) | {detail}")
                    continue

                qty = api.get_affordable_buy_qty(code, price, current_dt, nxt_tradeable)
                if qty <= 0:
                    log(f"  [BUY REJECT] {code}({name}) | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price={price:,.0f}")
                    continue

                session = classify_buy_session(current_dt)
                buy_detail = (
                    f"reason={buy_reason} signal={cross_info.get('signal')} "
                    f"live={price:,.0f} bb_mid={_num(cur, 'BB_MIDDLE'):.1f} "
                    f"bar_close={_num(cur, 'close'):,.0f} ma5={_num(cur, 'MA_5'):.1f}"
                )
                if api.place_buy_order(code, price, qty, current_dt, nxt_tradeable, session, buy_detail=buy_detail, code_name=name):
                    log(
                        f"  [BUY EVAL] {code}({name}) | OK {buy_reason} | {current_dt:%H:%M:%S} | "
                        f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}→{_num(cur, 'BB_MIDDLE'):.1f} | "
                        f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                        f"K={_num(prev_bar, 'STOCH_K'):.1f}→{_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                        f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}→{_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                        f"MACD {_num(prev_bar, 'MACD'):.2f}→{_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} | "
                        f"ADX={_num(cur, 'ADX'):.1f} +DI={_num(cur, 'DI_PLUS'):.1f} -DI={_num(cur, 'DI_MINUS'):.1f} | "
                        f"VOL={_num(cur, 'volume'):,.0f} VOLMA={_num(cur, 'VOL_MA20'):,.0f} | "
                        f"VWAP={_num(cur, 'VWAP'):,.0f} OBV={_num(cur, 'OBV'):,.0f} OBVMA={_num(cur, 'OBV_MA'):,.0f}"
                    )
                    log(f"  [BUY EXECUTED] {code}({name}) | {buy_reason} | qty={qty} price={price:,.0f} session={session}")
                    traded_today.add(code)
                    signal_buy_bar[code] = bar_time

                log("=" * 110)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    args = _parse_args()
    run(target_date=args.date)
