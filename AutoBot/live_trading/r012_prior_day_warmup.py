# -*- coding: utf-8 -*-
"""R012 prior-day bar restore / indicator warmup source - shared by r003 (live) and g003 (sim).

Overview
- g001 saves each trading day as runtime/data/<YYYYMMDD>/<code>_<name>_10s.txt: KIS 1-min bars stamped with their minute
  START label and linearly interpolated to 10 s (raw_bar=1 marks the original minute rows).
- causal_minute_frames() restores the real 1-min bars from such a file; load_prior_day_warmup() returns the previous trading
  day's last N 1-min bars and their 3-min aggregation (END labels, like r003 _normalize_intraday_frame) so that r002
  calculate_indicators_with_warmup can continue the indicators from yesterday instead of starting cold at 09:00.
- This module has no import side effects (no logging setup, no broker imports). The restore helpers used to live in g003,
  whose module-level logging.basicConfig(force=True) replaced the live r003 log handlers when r003 imported it (Codex review).

Update log (append only):
- [2026-10-28] type=feat owner=claude
    summary: 신규 - g003의 read_ohlc_csv/_find_code_txt_path_finest/_session_gap_mask/_causal_3min_slot_counts/
      _causal_minute_frames를 그대로 옮기고(g003은 이 모듈을 import), 전일 warmup 로더 load_prior_day_warmup() 추가.
      3분 완성 판정은 슬롯 수 기준(무체결 분이 낀 정상 봉 유지), 오늘 기준가(액면분할/권리락 반영)와 전일 종가가
      어긋나면 warmup 미사용.
    impact: common (r003 실전 warmup, g003 동작 불변)
    compatibility: backward-compatible
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from r001_define_config import MORNING_NXT_END, REGULAR_END, REGULAR_NEW_ENTRY_CUTOFF, REGULAR_START

logger = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def read_ohlc_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.info(f"CSV load failed: {csv_path} | {exc}")
        return None

    time_col = None
    for col in ("timestamp", "datetime", "Time", "time", "date", "Date"):
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.iloc[:, 1:]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return None

    df = df[~df.index.isna()].sort_index()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df if not df.empty else None


def _extract_code_from_stem(stem: str) -> str | None:
    match = re.search(r"(\d{6})", stem)
    if not match:
        return None
    return match.group(1)


def find_code_txt_path_finest(day_dir: Path, code: str, require_10s: bool = True) -> Path | None:
    """Return finest-granularity file for live-price simulation."""
    target = str(code).zfill(6)
    candidates: list[Path] = []
    for path in sorted(day_dir.glob("*.txt")):
        parsed = _extract_code_from_stem(path.stem)
        if parsed == target:
            candidates.append(path)

    if not candidates:
        return None

    if require_10s:
        ten_sec = [path for path in candidates if "_10s" in path.stem.lower()]
        if not ten_sec:
            return None
        ten_sec.sort(key=lambda path: (len(path.name), path.name))
        return ten_sec[0]

    def _fineness(path: Path) -> tuple[int, int, str]:
        stem = path.stem.lower()
        if "_10s" in stem:
            tier = 0  # finest granularity
        elif stem.endswith("_20s"):
            tier = 1
        elif stem.endswith("_1m"):
            tier = 3  # coarsest
        else:
            tier = 2
        return (tier, len(path.name), path.name)

    candidates.sort(key=_fineness)
    return candidates[0]


def session_gap_mask(index: pd.DatetimeIndex):
    """세션 공백(08:50~09:00, 15:20~15:30 단일가 구간) 여부 - 이 구간의 :00 행은 보간으로 생긴 가짜 행."""
    _t = index.time
    return ((_t >= MORNING_NXT_END) & (_t < REGULAR_START)) | ((_t >= REGULAR_NEW_ENTRY_CUTOFF) & (_t < REGULAR_END))


def causal_3min_slot_counts(raw_df: pd.DataFrame) -> pd.Series:
    """3분 구간별 1분 슬롯 수(분 종료 라벨 기준, 세션 공백 제외 :00 행 전체 - 원본/보간 구분 없음).

    [2026-10-10] raw_bar 필터 이후에는 무체결 분이 빠져 원본 분봉 개수로 세면 정상 3분봉까지 "부분봉"으로
    버려진다. r003 _normalize_intraday_frame은 첫 부분봉만 버리므로, "3개가 다 찬" 판정은 raw_bar 도입 전과
    같은 슬롯 기준으로 유지하고(첫 부분봉/15:30 단일가 단독봉만 제외) 봉의 OHLCV만 원본 분봉으로 만든다."""
    slots = raw_df.index[(raw_df.index.second == 0)]
    slots = slots[~session_gap_mask(slots)] + pd.Timedelta(minutes=1)
    return pd.Series(1, index=slots).resample("3min", label="right", closed="right").sum()


def causal_minute_frames(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """[2026-10-06] g001 _10s 파일(KIS 1분봉을 "분 시작" 라벨 그대로 찍고 10초 선형보간)에서 미래참조를 없앤다.

    Codex 1차 검토: 보간 10초 행은 다음 분 종가까지 섞여 있어 +60초 이동만으로는 미래정보가 남고, 보간 OHLC를 다시
    집계하면 실제 분봉과 다른 봉이 된다. 그래서 원본 앵커(:00 행)만 뽑아 실제 1분봉을 복원한다.
    반환: (minute_start - 분 시작 라벨 원본 1분봉, 체결 근사용 / minute_end - 분 종료 라벨 = 공개 시각, 봉·가격용).
    거래량은 g001 interpolate_to_10sec가 나눈 배수(원본 분 간격/10초 간격)를 곱해 복원한다.
    10초 보간 파일이 아니면(간격 중앙값이 10초가 아니면) None -> 호출측이 legacy 경로 사용."""
    if raw_df is None or raw_df.empty or len(raw_df.index) < 3:
        return None
    diffs = raw_df.index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty or abs(float(diffs.median()) - 10.0) > 0.5:
        return None
    m = raw_df[raw_df.index.second == 0]
    # [2026-10-10] g001이 raw_bar 열(1=원본 분봉)을 쓰면 그것만 사용 - 장중 무체결 분의 :00 행도 보간 합성
    # (직전 분 거래량 복사 + 가격 직선)이라 가짜 분봉이 섞였음(20261008 표본 308종목 중 168종목 거래량 +5% 초과).
    # 구버전 파일(열 없음)은 기존처럼 :00 행 전체 사용.
    if "raw_bar" in m.columns:
        m = m[pd.to_numeric(m["raw_bar"], errors="coerce") == 1]
    # 세션 공백(08:50~09:00, 15:20~15:30 단일가 구간)의 :00 행은 보간으로 생긴 가짜 행이라 제외한다.
    m = m[~session_gap_mask(m.index)]
    if len(m.index) < 2:
        return None
    m = m[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    m = m.dropna(subset=["open", "high", "low", "close"])
    m = m[m[["open", "high", "low", "close"]].max(axis=1) > 0]
    if m.empty:
        return None
    m_diffs = m.index.to_series().diff().dropna().dt.total_seconds()
    expansion = max(1.0, round(float(m_diffs.median()) / 10.0)) if not m_diffs.empty else 6.0
    m = m.copy()
    m["volume"] = m["volume"].fillna(0.0) * expansion
    m_end = m.copy()
    m_end.index = m_end.index + pd.Timedelta(minutes=1)
    return m, m_end


def load_prior_day_warmup(
    data_root: Path,
    date_str: str,
    code: str,
    minutes: int,
    max_lookback_days: int,
    base_price: float | None,
    max_base_diff_pct: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """[2026-10-28] 이전 거래일 원본 1분봉 마지막 `minutes`개와 그 3분 집계(종료 라벨, 1분 슬롯 3개가 다 찬 봉만).

    base_price: 오늘의 기준가(KIS stck_sdpr / g003 _prev_close.json). 전일 마지막 종가와 max_base_diff_pct 넘게 다르면
    액면분할/권리락 등으로 가격 기준이 바뀐 것이므로 warmup을 쓰지 않는다(이어 붙이면 급락으로 계산됨 - Codex 검토).
    None이면 검사 불가로 보고 warmup을 쓰지 않는다. 반환 (1분봉 | None, 3분봉 | None, 사유 문자열).
    """
    if minutes <= 0:
        return None, None, "disabled"
    target = str(code).zfill(6)
    base = datetime.strptime(date_str, "%Y%m%d")
    for back in range(1, int(max_lookback_days) + 1):
        day = (base - timedelta(days=back)).strftime("%Y%m%d")
        path = find_code_txt_path_finest(Path(data_root) / day, target, require_10s=True)
        if path is None:
            continue
        raw = read_ohlc_csv(path)
        if raw is None or raw.empty:
            continue
        raw = raw[raw.index.strftime("%Y%m%d") == day]
        causal = causal_minute_frames(raw) if not raw.empty else None
        if causal is None:
            continue
        m1 = causal[1][OHLCV].astype("float64")
        if m1.empty:
            continue
        last_close = float(m1["close"].iloc[-1])
        if base_price is None or not base_price > 0:
            return None, None, f"prior_day={day} base_price_unavailable"
        diff_pct = abs(last_close - float(base_price)) / float(base_price) * 100.0
        if diff_pct > max_base_diff_pct:
            return None, None, (
                f"prior_day={day} close={last_close:,.0f} != base={float(base_price):,.0f} ({diff_pct:.2f}%) - price basis changed"
            )
        m1 = m1.tail(int(minutes))
        m3 = m1.resample("3min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])
        slots = causal_3min_slot_counts(raw).reindex(m3.index).fillna(0)
        m3 = m3[slots >= 3]
        # tail()로 잘린 첫 3분 구간은 1분봉 일부만 담긴 부분봉일 수 있다 - 1분봉 시작 이전 슬롯이 있으면 버린다.
        if not m3.empty and m3.index[0] - pd.Timedelta(minutes=3) < m1.index[0] - pd.Timedelta(minutes=1):
            m3 = m3.iloc[1:]
        return m1, (m3 if not m3.empty else None), f"prior_day={day} 1min={len(m1)} 3min={len(m3)}"
    return None, None, f"prior-day 10s file not found within {max_lookback_days}d"
