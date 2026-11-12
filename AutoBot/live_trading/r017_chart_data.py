# -*- coding: utf-8 -*-
"""R017 - 차트/지표 데이터 조회 (AutoBotH 타임라인 차트용, 읽기 전용, stdout JSON 하나).

실행: ~/venv/bin/python live_trading/r017_chart_data.py <command> ...   (r016_control.py indicators|bars 로도 호출됨)

  indicators --date YYYYMMDD --code CODE --frame 1m|3m|daily [--last N]
      봇이 쓰는 것과 같은 입력(g001 _10s 파일의 raw_bar=1 원본 1분봉, 전일 1분봉 PRIOR_DAY_WARMUP_MINUTES개 예열, 기준가 =
      그 날짜 폴더 일봉의 직전 종가)과 같은 함수(r012.causal_minute_frames / load_prior_day_warmup,
      r002.calculate_indicators_with_warmup)로 다시 계산한다. r003 메모리의 프레임은 건드리지 않는다.
      1m/3m 행의 ts = 봉 시작, ts_end = 봉 종료(=봇이 그 봉을 볼 수 있는 시각). daily 는 g001 _daily.csv 의 지표 열 그대로
      (일봉 지표는 HTS 기본 기간: MACD 12/26/9, ADX 14 - g001 참조).
  bars --date YYYYMMDD --code CODE --kind tick|min|day|week|month [--n N] [--interval M] [--ticks T] [--warmup K]
       [--since-ts ISO] [--all-sessions]
      tick: g011 틱 CSV 에서 T틱(기본 60) 봉. min: 틱 CSV 가 있으면 틱으로 M분봉(기본 1), 없으면 _10s 원본 분봉.
      day/week/month: runtime/data/daily_history/<code>.csv (g025) 가 있으면 그것, 없으면 날짜 폴더 _daily.csv.
      --since-ts 는 그 시각 이후 봉만(마지막 봉 포함 - 갱신될 수 있음). 정규장+동시호가 체결만 쓰고 --all-sessions 면 전부.
      지난 날짜의 tick/min 집계는 runtime/cache/bars/ 에 캐시(당일은 캐시 안 함).

종료 코드: 0 성공, 2 입력/파일 없음.

Update log:
- [2026-11-12] type=feat owner=claude
    summary: bars --kind min: 파일이 없고 KRX 분봉도 없으면 KRX+NXT 통합 분봉(r018 minutes market=UN)으로 폴백, --source kis-un
      으로 통합 분봉을 직접 요청. 09:00 이전(NXT 프리마켓)에 AutoBotH 차트가 비던 문제.
    impact: tooling (AutoBotH 차트). 출력 형식 동일, source 가 "kis:<tr>:UN" 으로 표시됨
    compatibility: backward-compatible (--source 선택지 추가)
- [2026-11-11] type=feat owner=claude
    summary: bars --source kis (min 전용) - 틱/_10s 파일이 있어도 KIS 1분봉(r018.minutes)으로 응답. AutoBotH 가 틱 수집기가
      끊긴 구간(g011 중단 - 봇도 그 구간의 틱을 못 봄)을 KIS 분봉으로 채워 그리는 데 쓴다. 기본값 auto 는 기존과 같다.
    impact: tooling
    compatibility: backward-compatible (옵션 추가)
- [2026-11-07] type=feat owner=claude
    summary: bars KIS 폴백(r018, 읽기 전용) - 서버 파일이 없을 때만: day/week/month = KIS 일봉 3년, min = KIS 1분봉(당일
      inquire_time_itemchartprice / 지난 날짜 inquire_time_dailychartprice, 직전일 예열 최대 3일 시도), tick 은 오류 문구 고정.
      KIS 오류는 kis_code + rc 3(_rc). KIS 응답이면 name 추가.
    impact: tooling
    compatibility: backward-compatible (파일이 있으면 기존과 동일)
- [2026-11-06] type=fix owner=claude
    summary: Codex 검토 반영 - (1) 캐시에 (st_dev, st_ino) + 오프셋 직전 바이트 지문을 저장해 교체/축소-재증가 파일이면 재구성
      (2) 예열 기준가를 그 날짜 폴더 _prev_close.json(g003 parity)에서 먼저 읽고 warmup_basis 로 알림(실전 r003 은 KIS stck_sdpr -
      권리락/분할일엔 다를 수 있음) (3) 1분 프레임은 r003 FRAME_CACHE_MAX_BARS=200 롤링 창과 같은 값이 되도록 200봉 초과 행은
      창별 재계산 (4) 당일 3분봉은 r003 _normalize_intraday_frame 처럼 첫 부분봉만 버리고 15:30 동시호가 봉은 유지(예열은 r012 규칙 유지)
      (5) 틱 행은 csv.reader 로 헤더 폭 검사(끊긴 행 제외, r009/r013 과 동일) (6) _10s 파일이 없는 당일은 틱으로 만든 1분봉으로
      지표 계산(source 에 ticks 표시, 장중 근사) (7) 캐시 임시 파일 mkstemp.
    impact: tooling
    compatibility: backward-compatible (JSON 필드 추가: warmup_basis)
- [2026-11-06] type=feat owner=claude
    summary: AutoBotH 차트 요청(P2-2 indicators, P3-1 bars) 신규. 틱 집계는 pandas 벡터 연산(acml_vol 기준 중복 제거)이라
      r009 TickBarBuilder 의 갭/역순 규칙과 미세하게 다를 수 있음(차트용). 5MB 틱 파일 기준 Pi 에서 약 0.6s.
    impact: tooling
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402
    import r002_strategy_core_shared as r002  # noqa: E402
    import r012_prior_day_warmup as r012  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume"]
INDICATOR_COLUMNS = [
    "MA_5", "EMA_20", "EMA_60", "VOL_MA20", "BB_MIDDLE", "BB_UPPER", "BB_LOWER", "BB_STD", "RSI", "RSI_SIGNAL",
    "STOCH_K", "STOCH_D", "WILLIAMS_R", "WILLIAMS_D", "MACD", "MACD_SIGNAL", "MACD_HIST", "DI_PLUS", "DI_MINUS", "ADX",
    "ATR", "MFI", "VWAP", "OBV", "OBV_MA",
]
TRADING_SESSIONS = ("regular", "close_auction")


def _runtime_root() -> Path:
    return Path(r001.RUNTIME_ROOT)


def _data_root() -> Path:
    return _runtime_root() / "data"


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else round(f, 6)


def _ts(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def daily_csv_path(date_str: str, code: str) -> Path | None:
    day = _data_root() / date_str
    hits = sorted(day.glob(f"{code}_*_daily.csv")) if day.is_dir() else []
    return hits[0] if hits else None


def latest_daily_csv(date_str: str, code: str, lookback_days: int = 10) -> tuple[Path | None, str | None]:
    base = datetime.strptime(date_str, "%Y%m%d")
    for back in range(0, lookback_days + 1):
        d = (base - timedelta(days=back)).strftime("%Y%m%d")
        p = daily_csv_path(d, code)
        if p is not None:
            return p, d
    return None, None


def read_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    for c in OHLCV + ["amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def daily_history_path(code: str) -> Path | None:
    p = _data_root() / "daily_history" / f"{code}.csv"
    return p if p.is_file() else None


def base_price_for(date_str: str, code: str) -> tuple[float | None, str]:
    """Today's price basis for the warmup check: g003 parity = that folder's _prev_close.json, else the daily csv's
    previous row. r003 live uses KIS stck_sdpr instead (differs on ex-rights/split days -> r003 may skip warmup)."""
    pc = _data_root() / date_str / "_prev_close.json"
    with contextlib.suppress(Exception):
        v = json.loads(pc.read_text(encoding="utf-8")).get(code)
        if v is not None and float(v) > 0:
            return float(v), "_prev_close.json"
    v = prev_close_from_daily(date_str, code)
    return v, ("daily_csv_prev_row" if v else "none")


def prev_close_from_daily(date_str: str, code: str) -> float | None:
    path, _ = latest_daily_csv(date_str, code)
    if path is None:
        return None
    df = read_daily(path)
    before = df[df.index < pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))]
    return float(before["close"].iloc[-1]) if not before.empty else None


def _day_3min(m1_end: pd.DataFrame) -> pd.DataFrame:
    """Same as r003 _normalize_intraday_frame: 3-min bins on END-labelled minutes, drop only a partial FIRST bin
    (the trailing 15:30 closing-auction bin stays, unlike the stricter prior-day warmup rule in r012)."""
    tmp = m1_end[OHLCV].copy()
    tmp["n"] = 1
    m3 = tmp.resample("3min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "n": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    if not m3.empty and int(m3["n"].iloc[0]) < 3:
        m3 = m3.iloc[1:]
    return m3.drop(columns=["n"])


def minute_frames(date_str: str, code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str] | None:
    """(m1_start_label, m1_end_label, m3_end_label, source) from the g001 _10s file (r012 causal frames) or, when that
    file does not exist yet (today, during the session), from the tick file's regular-session prints (approximation)."""
    path = r012.find_code_txt_path_finest(_data_root() / date_str, code, require_10s=True)
    if path is not None:
        raw = r012.read_ohlc_csv(path)
        if raw is None or raw.empty:
            return None
        raw = raw[raw.index.strftime("%Y%m%d") == date_str]
        causal = r012.causal_minute_frames(raw) if not raw.empty else None
        if causal is None:
            return None
        m1, m1_end = causal
        m1 = m1[OHLCV].astype("float64")
        m1_end = m1_end[OHLCV].astype("float64")
        return m1, m1_end, _day_3min(m1_end), str(path)
    tf = tick_file(date_str, code)
    if tf is None:
        return None
    ticks = load_ticks(tf)
    if ticks.empty:
        return None
    mb = minute_bars(ticks, 1)
    m1 = pd.DataFrame({"open": mb["o"].to_numpy(), "high": mb["h"].to_numpy(), "low": mb["l"].to_numpy(),
                       "close": mb["c"].to_numpy(), "volume": mb["v"].to_numpy()}, index=pd.DatetimeIndex(mb["t"])).astype("float64")
    m1_end = m1.copy()
    m1_end.index = m1_end.index + pd.Timedelta(1, "min")
    return m1, m1_end, _day_3min(m1_end), f"{tf} (ticks -> 1m, live approximation)"


def warmup_frames(date_str: str, code: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    if not bool(getattr(r001, "ENABLE_PRIOR_DAY_INDICATOR_WARMUP", False)):
        return None, None, "disabled"
    base_price, basis = base_price_for(date_str, code)
    m1, m3, why = r012.load_prior_day_warmup(
        _data_root(), date_str, code, int(r001.PRIOR_DAY_WARMUP_MINUTES), int(r001.PRIOR_DAY_WARMUP_MAX_LOOKBACK_DAYS),
        base_price, float(r001.PRIOR_DAY_WARMUP_MAX_BASE_DIFF_PCT),
    )
    return m1, m3, f"{why} (base_price from {basis}; r003 live uses KIS stck_sdpr)"


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def _params() -> dict:
    names = ["MA_PERIOD", "VOLUME_MA_PERIOD", "EMA_20_PERIOD", "EMA_60_PERIOD", "BB_PERIOD", "BB_STD_MULTIPLIER", "RSI_PERIOD",
             "RSI_SIGNAL_PERIOD", "STOCH_K_PERIOD", "STOCH_D_PERIOD", "WILLIAMS_R_PERIOD", "WILLIAMS_D_PERIOD", "MFI_PERIOD",
             "OBV_MA_PERIOD", "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL_PERIOD", "ADX_PERIOD", "ATR_PERIOD",
             "PRIOR_DAY_WARMUP_MINUTES", "ENABLE_PRIOR_DAY_INDICATOR_WARMUP"]
    return {n: getattr(r001, n) for n in names if hasattr(r001, n)}


FRAME_CACHE_MAX_BARS = 200  # r003 _merge_bar_frame keeps only the last 200 bars of the day before computing


PARITY_MAX_WINDOWS = 40  # each trailing-window recomputation costs ~0.16 s on the Pi; beyond this fall back to one pass


def _indicators_rolling(day: pd.DataFrame, warmup: pd.DataFrame | None, max_bars: int, need_last: int | None = None) -> tuple[pd.DataFrame, str]:
    """r003 parity: the bot computes on at most the last `max_bars` day bars (+ warmup). calculate_indicators is causal,
    so rows < max_bars equal the plain prefix computation; later rows are recomputed from their own trailing window.
    Only the rows that will be returned (need_last) are recomputed; if that is more than PARITY_MAX_WINDOWS the whole
    day is computed in one pass (tiny recursive-indicator drift vs the bot after 200 bars) and parity says so."""
    n = len(day)
    if n <= max_bars:
        return r002.calculate_indicators_with_warmup(day, warmup), "exact"
    first_needed = max(max_bars, n - int(need_last)) if need_last else max_bars
    if n - first_needed > PARITY_MAX_WINDOWS:
        return r002.calculate_indicators_with_warmup(day, warmup), f"single_pass (>{max_bars} bars; r003 keeps a {max_bars}-bar window - small drift in EMA/RSI/ADX)"
    full = r002.calculate_indicators_with_warmup(day, warmup)   # rows < max_bars are exact here (causal)
    pieces = [full.iloc[:first_needed]]
    for k in range(first_needed, n):
        win = day.iloc[k - max_bars + 1:k + 1]
        pieces.append(r002.calculate_indicators_with_warmup(win, warmup).iloc[-1:])
    return pd.concat(pieces), "exact"


def _frame_rows(df: pd.DataFrame, columns: list[str], ts_end_offset: pd.Timedelta | None) -> list[list]:
    rows = []
    for ts, r in df.iterrows():
        row = [_ts(ts)]
        if ts_end_offset is not None:
            row.append(_ts(ts + ts_end_offset))
        row += [_num(r.get(c)) for c in columns]
        rows.append(row)
    return rows


def indicators(date_str: str, code: str, frame: str, last: int | None = None) -> dict:
    t0 = time.time()
    code = str(code).zfill(6)
    if frame == "daily":
        path, src_date = latest_daily_csv(date_str, code)
        if path is None:
            return {"ok": False, "error": f"no _daily.csv for {code} around {date_str}"}
        df = read_daily(path)
        cols = [c for c in OHLCV + ["amount"] + INDICATOR_COLUMNS + ["MA_20", "MA_60", "MA_120"] if c in df.columns]
        if last:
            df = df.tail(int(last))
        return {"ok": True, "frame": "daily", "code": code, "date": date_str, "source": str(path), "source_date": src_date,
                "params": {"note": "daily indicator columns are g001's (HTS default periods), not r001 intraday periods"},
                "columns": ["ts"] + cols, "rows": [[_ts(ts)] + [_num(r.get(c)) for c in cols] for ts, r in df.iterrows()],
                "elapsed_ms": int((time.time() - t0) * 1000)}
    if frame not in ("1m", "3m"):
        return {"ok": False, "error": "frame must be 1m|3m|daily"}
    mf = minute_frames(date_str, code)
    if mf is None:
        return {"ok": False, "error": f"no _10s file (raw_bar rows) for {code} on {date_str}"}
    m1, m1_end, m3, path = mf
    w1, w3, why = warmup_frames(date_str, code)
    if frame == "1m":
        df, parity = _indicators_rolling(m1_end, w1, FRAME_CACHE_MAX_BARS, last)
        offset = pd.Timedelta(1, "min")
    else:
        df, parity = _indicators_rolling(m3, w3, FRAME_CACHE_MAX_BARS, last)
        offset = pd.Timedelta(3, "min")
    cols = [c for c in OHLCV + INDICATOR_COLUMNS if c in df.columns]
    if last:
        df = df.tail(int(last))
    # bot frames carry END labels; show start = end - size and end explicitly
    out_rows = []
    for ts_end, r in df.iterrows():
        out_rows.append([_ts(ts_end - offset), _ts(ts_end)] + [_num(r.get(c)) for c in cols])
    return {"ok": True, "frame": frame, "code": code, "date": date_str, "source": str(path), "warmup": why, "parity": parity,
            "params": _params(), "columns": ["ts", "ts_end"] + cols, "rows": out_rows,
            "elapsed_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------
def tick_file(date_str: str, code: str) -> Path | None:
    p = _data_root() / "ticks" / date_str / f"{code}.csv"
    return p if p.is_file() else None


TICK_FIELDS = ("bsop_date", "trade_time", "price", "cntg_vol", "acml_vol", "session")


def _rows_from_lines(header: list[str], lines) -> list[list]:
    """Complete rows only: width must equal the header width (torn rows are dropped exactly like r009/r013)."""
    idx = [header.index(c) if c in header else None for c in TICK_FIELDS]
    width = len(header)
    out = []
    for fields in csv.reader(lines):
        if len(fields) != width:
            continue
        out.append([fields[i] if i is not None else None for i in idx])
    return out


def _ticks_frame(rows: list[list], all_sessions: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(TICK_FIELDS))
    if df.empty:
        return pd.DataFrame(columns=["ts", "price", "qty", "acml", "session"])
    df["ts"] = pd.to_datetime(df["bsop_date"].str.strip() + df["trade_time"].str.strip().str.zfill(6), format="%Y%m%d%H%M%S", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["cntg_vol"], errors="coerce").fillna(0)
    df["acml"] = pd.to_numeric(df["acml_vol"], errors="coerce")
    df = df.dropna(subset=["ts", "price", "acml"])
    if df["session"].isna().all():
        hhmm = df["ts"].dt.strftime("%H%M%S")
        df["session"] = "regular"
        df.loc[hhmm < "090000", "session"] = "pre"
        df.loc[(hhmm >= "153000") & (hhmm < "154000"), "session"] = "close_auction"
        df.loc[hhmm >= "154000", "session"] = "post"
    df["session"] = df["session"].fillna("")
    # exchange order = cumulative volume order; duplicates (reconnect replays) share acml + ts
    df = df.sort_values(["acml", "ts"], kind="stable").drop_duplicates(subset=["ts", "acml", "price", "qty"], keep="first")
    if not all_sessions:
        df = df[df["session"].isin(TRADING_SESSIONS)]
    return df[["ts", "price", "qty", "acml", "session"]].reset_index(drop=True)


def load_ticks(path: Path, all_sessions: bool = False) -> pd.DataFrame:
    """Deduplicated prints in exchange order: ts, price, qty, acml_vol, session."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader([fh.readline()]))
        rows = _rows_from_lines(header, fh)
    return _ticks_frame(rows, all_sessions)


def _agg(df: pd.DataFrame, key) -> pd.DataFrame:
    g = df.groupby(key, sort=True)
    out = pd.DataFrame({
        "t": g["ts"].first(), "o": g["price"].first(), "h": g["price"].max(), "l": g["price"].min(),
        "c": g["price"].last(), "v": g["qty"].sum(), "n": g["price"].size(), "t_last": g["ts"].last(),
    })
    return out.reset_index(drop=True)


def tick_bars(df: pd.DataFrame, ticks: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "n", "t_last"])
    key = (pd.RangeIndex(len(df)) // max(1, int(ticks)))
    return _agg(df, key)


def minute_bars(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "n", "t_last"])
    key = df["ts"].dt.floor(f"{max(1, int(interval))}min")
    out = _agg(df, key)
    out["t"] = df.groupby(key, sort=True)["ts"].first().dt.floor(f"{max(1, int(interval))}min").to_numpy()
    return out


def minute_bars_from_10s(date_str: str, code: str, interval: int) -> tuple[pd.DataFrame, str] | None:
    mf = minute_frames(date_str, code)
    if mf is None:
        return None
    m1, _m1_end, _m3, path = mf
    if interval > 1:
        m1 = m1.resample(f"{interval}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    out = pd.DataFrame({"t": m1.index, "o": m1["open"].to_numpy(), "h": m1["high"].to_numpy(), "l": m1["low"].to_numpy(),
                        "c": m1["close"].to_numpy(), "v": m1["volume"].to_numpy(), "n": None, "t_last": m1.index})
    return out.reset_index(drop=True), str(path)


def day_bars(date_str: str, code: str, kind: str) -> tuple[pd.DataFrame, str] | None:
    hist = daily_history_path(code)
    if hist is not None:
        df = read_daily(hist)
        src = str(hist)
    else:
        path, _ = latest_daily_csv(date_str, code)
        if path is None:
            return None
        df = read_daily(path)
        src = str(path)
    return _shape_daily(df, date_str, kind), src


def _shape_daily(df: pd.DataFrame, date_str: str, kind: str) -> pd.DataFrame:
    """daily frame (index date, open..volume) -> day/week/month bars t,o,h,l,c,v,n,t_last up to date_str."""
    df = df[df.index <= pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))]
    if kind == "week":
        df = df.resample("W-FRI", label="left", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
        df.index = df.index + pd.Timedelta(3, "D")  # label left = previous Friday -> Monday of that week
    elif kind == "month":
        df = df.resample("MS").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    out = pd.DataFrame({"t": df.index, "o": df["open"].to_numpy(), "h": df["high"].to_numpy(), "l": df["low"].to_numpy(),
                        "c": df["close"].to_numpy(), "v": df["volume"].to_numpy(), "n": None, "t_last": df.index})
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KIS fallback (r018, read-only) - only when the server has no file for the code/day
# ---------------------------------------------------------------------------
KIS_WARMUP_MAX_DAYS = 3  # 거래일 기준(r014 달력). 모든 KIS 조회는 r018.shared_client() 의 명령당 마감 시각을 나눠 쓴다


def _kis():
    import r018_kis_market as r018  # noqa: E402  (requests/yaml - only when a fallback is needed)
    return r018


def kis_day_bars(date_str: str, code: str, kind: str) -> tuple[pd.DataFrame, str, str | None]:
    d = _kis().daily(code)
    df = pd.DataFrame([r[1:] for r in d["rows"]], columns=["open", "high", "low", "close", "volume"],
                      index=pd.to_datetime([r[0] for r in d["rows"]], format="%Y%m%d"))
    return _shape_daily(df, date_str, kind), d["source"], d.get("name")


def kis_minute_bars(date_str: str, code: str, interval: int, market: str = "J") -> tuple[pd.DataFrame, str, str | None] | None:
    """market "UN" = KRX+NXT combined minutes (08:00~20:00), for the hours KRX has no bars."""
    m = _kis().minutes(code, date_str) if market == "J" else _kis().minutes(code, date_str, market=market)
    if not m or not m["rows"]:
        return None
    idx = pd.to_datetime([date_str + r[0] for r in m["rows"]], format="%Y%m%d%H%M%S")
    m1 = pd.DataFrame([r[1:] for r in m["rows"]], columns=["open", "high", "low", "close", "volume"], index=idx)
    if interval > 1:
        m1 = m1.resample(f"{interval}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    out = pd.DataFrame({"t": m1.index, "o": m1["open"].to_numpy(), "h": m1["high"].to_numpy(), "l": m1["low"].to_numpy(),
                        "c": m1["close"].to_numpy(), "v": m1["volume"].to_numpy(), "n": None, "t_last": m1.index})
    return out.reset_index(drop=True), m["source"], m.get("name")


def _kis_error(exc) -> dict:
    return {"ok": False, "error": f"KIS: {exc.message}", "kis_code": exc.code, "_rc": 3}


def _bars_json(df: pd.DataFrame) -> list[list]:
    return [[_ts(r.t), _num(r.o), _num(r.h), _num(r.l), _num(r.c), _num(r.v)] for r in df.itertuples()]


def _cache_path(date_str: str, code: str, tag: str) -> Path:
    return _runtime_root() / "cache" / "bars" / date_str / f"{code}_{tag}.json"


def _parse_tick_text(text: str, header: str, all_sessions: bool) -> pd.DataFrame:
    cols = next(csv.reader([header]))
    return _ticks_frame(_rows_from_lines(cols, io.StringIO(text)), all_sessions)


def _tail_fingerprint(fh, offset: int, n: int = 256) -> str:
    start = max(0, offset - n)
    fh.seek(start)
    return hashlib.sha1(fh.read(offset - start)).hexdigest()


def _fold_bars(prev: pd.DataFrame | None, new_ticks: pd.DataFrame, kind: str, ticks: int, interval: int,
               carry: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append new prints to existing bars. carry = prints of the last (still open) bar; returns (bars, new carry)."""
    work = pd.concat([carry, new_ticks], ignore_index=True) if carry is not None and not carry.empty else new_ticks
    if work.empty:
        return (prev if prev is not None else tick_bars(work, ticks)), work
    base = prev.iloc[:-1] if (prev is not None and not prev.empty and carry is not None and not carry.empty) else prev
    if kind == "tick":
        agg = tick_bars(work, ticks)
        n_full = len(work) // max(1, int(ticks))
        remainder = work.iloc[n_full * max(1, int(ticks)):]
        new_carry = remainder if not remainder.empty else work.tail(max(1, int(ticks)))  # last bar full -> carry it whole
        if remainder.empty:
            new_carry = work.iloc[(n_full - 1) * max(1, int(ticks)):] if n_full > 0 else work
    else:
        agg = minute_bars(work, interval)
        last_key = work["ts"].dt.floor(f"{max(1, int(interval))}min").iloc[-1]
        new_carry = work[work["ts"].dt.floor(f"{max(1, int(interval))}min") == last_key]
    out = pd.concat([base, agg], ignore_index=True) if base is not None and not base.empty else agg
    return out.reset_index(drop=True), new_carry.reset_index(drop=True)


def _bars_from_json(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["t"] = pd.to_datetime(df["t"])
    df["n"] = None
    df["t_last"] = df["t"]
    return df


def _ticks_to_json(df: pd.DataFrame) -> list:
    return [[_ts(r.ts), _num(r.price), _num(r.qty), _num(r.acml), r.session] for r in df.itertuples()]


def _ticks_from_json(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "price", "qty", "acml", "session"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def _intraday_bars(date_str: str, code: str, kind: str, ticks: int, interval: int, all_sessions: bool) -> tuple[pd.DataFrame, str] | None:
    """Bars from the tick file with a byte-offset cache: a later call parses only the appended prints (10 s polling
    during the session must not re-read 5 MB). Past dates are cached whole; a cache that does not match the file
    (shrunk / header changed) is rebuilt from scratch."""
    tf = tick_file(date_str, code)
    if tf is None:
        return minute_bars_from_10s(date_str, code, interval) if kind == "min" else None
    tag = (f"tick{ticks}" if kind == "tick" else f"min{interval}") + ("_all" if all_sessions else "")
    cache = _cache_path(date_str, code, tag)
    st = tf.stat()
    size = st.st_size
    file_id = [st.st_dev, st.st_ino]
    with open(tf, "rb") as fb:
        header = fb.readline().decode("utf-8", "replace")
        state = None
        if cache.is_file():
            with contextlib.suppress(Exception):
                state = json.loads(cache.read_text(encoding="utf-8"))
                ok = (state.get("header") == header and state.get("offset", 0) <= size and state.get("file_id") == file_id
                      and state.get("fingerprint") == _tail_fingerprint(fb, int(state.get("offset", 0))))
                if not ok:
                    state = None   # replaced / rewritten / shrunk file: rebuild from scratch
        if state is not None and state.get("offset") == size:
            return _bars_from_json(state["bars"]), f"{tf} (cache)"
        offset = state["offset"] if state else len(header.encode("utf-8"))
        fb.seek(offset)
        text = fb.read().decode("utf-8", "replace")
    # keep an incomplete trailing line for the next call
    cut = text.rfind("\n")
    if cut < 0:
        text, consumed = "", 0
    else:
        consumed = len(text[:cut + 1].encode("utf-8"))
        text = text[:cut + 1]
    new_ticks = _parse_tick_text(text, header, all_sessions) if text else pd.DataFrame(columns=["ts", "price", "qty", "acml", "session"])
    prev = _bars_from_json(state["bars"]) if state else None
    carry = _ticks_from_json(state["carry"]) if state and state.get("carry") else None
    if state and not new_ticks.empty:
        last_acml = state.get("last_acml")
        if last_acml is not None:
            new_ticks = new_ticks[new_ticks["acml"] > float(last_acml)]  # replayed prints after a reconnect
    bars_df, new_carry = _fold_bars(prev, new_ticks, kind, ticks, interval, carry)
    with contextlib.suppress(Exception):
        cache.parent.mkdir(parents=True, exist_ok=True)
        last_acml = float(new_carry["acml"].max()) if not new_carry.empty else (state.get("last_acml") if state else None)
        with open(tf, "rb") as fb:
            fingerprint = _tail_fingerprint(fb, offset + consumed)
        payload = {"header": header, "offset": offset + consumed, "file_id": file_id, "fingerprint": fingerprint,
                   "bars": _bars_json(bars_df), "carry": _ticks_to_json(new_carry), "last_acml": last_acml, "source": str(tf)}
        fd, tmp = tempfile.mkstemp(prefix=f".{cache.name}.", suffix=".tmp", dir=str(cache.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(payload))
        os.replace(tmp, cache)
    return bars_df, str(tf)


def _prev_days(date_str: str, max_back: int = 10):
    base = datetime.strptime(date_str, "%Y%m%d")
    for back in range(1, max_back + 1):
        yield (base - timedelta(days=back)).strftime("%Y%m%d")


_KIS_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")


def bars(date_str: str, code: str, kind: str, n: int | None = None, interval: int = 1, ticks: int = 60, warmup: int = 0,
         since_ts: str | None = None, all_sessions: bool = False, source: str = "auto") -> dict:
    t0 = time.time()
    code = str(code).zfill(6)
    name = None
    if kind in ("day", "week", "month"):
        res = day_bars(date_str, code, kind)
        if res is None:
            if not _KIS_CODE_RE.fullmatch(code):
                return {"ok": False, "error": f"no daily data for {code} around {date_str}"}
            r018 = _kis()
            try:
                df, src, name = kis_day_bars(date_str, code, kind)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            except r018.KisError as exc:
                return _kis_error(exc)
            if df.empty:
                return {"ok": False, "error": f"no daily data for {code} on or before {date_str} (KIS)"}
        else:
            df, src = res
        warm_rows, warm_date = [], None
    elif kind in ("tick", "min"):
        # [2026-11-11] --source kis: minute bars from KIS even when a tick / _10s file exists (AutoBotH fills the minutes the
        # tick collector missed - a g011 outage leaves holes in the tick file that the bot could not see either)
        # [2026-11-12] --source kis-un: KRX+NXT combined minute bars (08:00~20:00). With source auto and no file, KRX minutes
        # are asked first and the combined ones when KRX has none (before 09:00 the chart of an NXT symbol was empty).
        from_kis = source in ("kis", "kis-un") and kind == "min"
        res = None if from_kis else _intraday_bars(date_str, code, kind, ticks, interval, all_sessions)
        if res is not None and kind == "min" and res[0].empty:
            res = None   # a tick file with pre-open prints only (08:30~09:00): no bar yet -> KIS minutes
        kis_used = False
        if res is None:
            if kind == "tick":
                return {"ok": False, "error": "tick bars need the tick collector (not subscribed): use min"}
            r018 = _kis()
            try:
                kres = kis_minute_bars(date_str, code, interval, "UN" if source == "kis-un" else "J")
                if kres is None and source == "auto":
                    kres = kis_minute_bars(date_str, code, interval, "UN")
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            except r018.KisError as exc:
                return _kis_error(exc)
            if kres is None:
                return {"ok": False, "error": f"no tick file / _10s file for {code} on {date_str} (KIS has no minute bars either)"}
            df, src, name = kres
            kis_used = True
        else:
            df, src = res
        warm_rows, warm_date = [], None
        if warmup and warmup > 0:
            kis_tries = 0
            for d in _prev_days(date_str):
                prev = _intraday_bars(d, code, kind, ticks, interval, all_sessions)
                if ((prev is None or prev[0].empty) and kind == "min" and kis_used and kis_tries < KIS_WARMUP_MAX_DAYS
                        and _kis()._session_hours(d) is not None):  # 휴장일은 시도 횟수에 넣지 않는다
                    kis_tries += 1
                    try:
                        kp = kis_minute_bars(d, code, interval)
                    except (ValueError, _kis().KisError):
                        kp = None
                    prev = (kp[0], kp[1]) if kp else None
                if prev is not None and not prev[0].empty:
                    warm_rows, warm_date = _bars_json(prev[0].tail(int(warmup))), d
                    break
    else:
        return {"ok": False, "error": "kind must be tick|min|day|week|month"}
    if since_ts:
        try:
            cut = pd.Timestamp(since_ts)
        except Exception:
            return {"ok": False, "error": f"bad --since-ts {since_ts!r}"}
        df = df[df["t"] >= cut] if not df.empty else df
        warm_rows = []
    if n and n > 0:
        df = df.tail(int(n))
    next_since = _ts(df["t"].iloc[-1]) if not df.empty else (since_ts or None)
    out_name = {"name": name} if name else {}
    return {"ok": True, "kind": kind, "code": code, "date": date_str, "source": src, **out_name,
            "ticks": ticks if kind == "tick" else None, "interval": interval if kind == "min" else None,
            "sessions": "all" if all_sessions else "+".join(TRADING_SESSIONS),
            "columns": ["t", "o", "h", "l", "c", "v"], "bars": _bars_json(df), "count": int(len(df)),
            "warm": warm_rows, "warm_date": warm_date, "next_since": next_since,
            "elapsed_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_subparsers(sub) -> None:
    p = sub.add_parser("indicators")
    p.add_argument("--date", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--frame", choices=["1m", "3m", "daily"], default="3m")
    p.add_argument("--last", type=int, default=None)
    p = sub.add_parser("bars")
    p.add_argument("--date", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--kind", choices=["tick", "min", "day", "week", "month"], default="min")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--interval", type=int, default=1, help="minutes per bar for kind=min")
    p.add_argument("--ticks", type=int, default=60, help="prints per bar for kind=tick")
    p.add_argument("--warmup", type=int, default=0, help="also return the last K bars of the previous day")
    p.add_argument("--since-ts", dest="since_ts", default=None)
    p.add_argument("--all-sessions", dest="all_sessions", action="store_true")
    p.add_argument("--source", choices=["auto", "kis", "kis-un"], default="auto",
                   help="min: 'kis' = KIS minute bars even when the tick / _10s file exists (fill tick-collector gaps); "
                        "'kis-un' = KRX+NXT combined minute bars (08:00~20:00)")


def run_command(args) -> dict:
    if args.cmd == "indicators":
        return indicators(args.date, args.code, args.frame, args.last)
    return bars(args.date, args.code, args.kind, args.n, args.interval, args.ticks, args.warmup, args.since_ts, args.all_sessions,
                getattr(args, "source", "auto"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chart data (JSON out)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_subparsers(sub)
    args = ap.parse_args(argv)
    out = run_command(args)
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))
    return 0 if out.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
