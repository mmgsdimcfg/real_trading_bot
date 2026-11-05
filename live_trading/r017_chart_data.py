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
- [2026-11-06] type=feat owner=claude
    summary: AutoBotH 차트 요청(P2-2 indicators, P3-1 bars) 신규. 틱 집계는 pandas 벡터 연산(acml_vol 기준 중복 제거)이라
      r009 TickBarBuilder 의 갭/역순 규칙과 미세하게 다를 수 있음(차트용). 5MB 틱 파일 기준 Pi 에서 약 0.6s.
    impact: tooling
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
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


def prev_close_from_daily(date_str: str, code: str) -> float | None:
    path, _ = latest_daily_csv(date_str, code)
    if path is None:
        return None
    df = read_daily(path)
    before = df[df.index < pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))]
    return float(before["close"].iloc[-1]) if not before.empty else None


def minute_frames(date_str: str, code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path] | None:
    """(m1_start_label, m1_end_label, m3_end_label, path) from the g001 _10s file, exactly like r012/g003."""
    path = r012.find_code_txt_path_finest(_data_root() / date_str, code, require_10s=True)
    if path is None:
        return None
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
    m3 = m1_end.resample("3min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    slots = r012.causal_3min_slot_counts(raw).reindex(m3.index).fillna(0)
    m3 = m3[slots >= 3]
    return m1, m1_end, m3, path


def warmup_frames(date_str: str, code: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    if not bool(getattr(r001, "ENABLE_PRIOR_DAY_INDICATOR_WARMUP", False)):
        return None, None, "disabled"
    base_price = prev_close_from_daily(date_str, code)
    return r012.load_prior_day_warmup(
        _data_root(), date_str, code, int(r001.PRIOR_DAY_WARMUP_MINUTES), int(r001.PRIOR_DAY_WARMUP_MAX_LOOKBACK_DAYS),
        base_price, float(r001.PRIOR_DAY_WARMUP_MAX_BASE_DIFF_PCT),
    )


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def _params() -> dict:
    names = ["MA_PERIOD", "VOLUME_MA_PERIOD", "EMA_20_PERIOD", "EMA_60_PERIOD", "BB_PERIOD", "BB_STD_MULTIPLIER", "RSI_PERIOD",
             "RSI_SIGNAL_PERIOD", "STOCH_K_PERIOD", "STOCH_D_PERIOD", "WILLIAMS_R_PERIOD", "WILLIAMS_D_PERIOD", "MFI_PERIOD",
             "OBV_MA_PERIOD", "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL_PERIOD", "ADX_PERIOD", "ATR_PERIOD",
             "PRIOR_DAY_WARMUP_MINUTES", "ENABLE_PRIOR_DAY_INDICATOR_WARMUP"]
    return {n: getattr(r001, n) for n in names if hasattr(r001, n)}


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
        df = r002.calculate_indicators_with_warmup(m1_end, w1)
        offset = pd.Timedelta(minutes=1)
    else:
        df = r002.calculate_indicators_with_warmup(m3, w3)
        offset = pd.Timedelta(minutes=3)
    cols = [c for c in OHLCV + INDICATOR_COLUMNS if c in df.columns]
    if last:
        df = df.tail(int(last))
    # bot frames carry END labels; show start = end - size and end explicitly
    out_rows = []
    for ts_end, r in df.iterrows():
        out_rows.append([_ts(ts_end - offset), _ts(ts_end)] + [_num(r.get(c)) for c in cols])
    return {"ok": True, "frame": frame, "code": code, "date": date_str, "source": str(path), "warmup": why,
            "params": _params(), "columns": ["ts", "ts_end"] + cols, "rows": out_rows,
            "elapsed_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------
def tick_file(date_str: str, code: str) -> Path | None:
    p = _data_root() / "ticks" / date_str / f"{code}.csv"
    return p if p.is_file() else None


def load_ticks(path: Path, all_sessions: bool = False) -> pd.DataFrame:
    """Deduplicated prints in exchange order: ts, price, qty, acml_vol, session."""
    df = pd.read_csv(path, dtype=str, usecols=lambda c: c in ("bsop_date", "trade_time", "price", "cntg_vol", "acml_vol", "session"),
                     encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["bsop_date", "trade_time", "price"])
    df["ts"] = pd.to_datetime(df["bsop_date"].str.strip() + df["trade_time"].str.strip().str.zfill(6), format="%Y%m%d%H%M%S", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["cntg_vol"], errors="coerce").fillna(0)
    df["acml"] = pd.to_numeric(df["acml_vol"], errors="coerce")
    df = df.dropna(subset=["ts", "price", "acml"])
    if "session" not in df.columns:
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
    df = df[df.index <= pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))]
    if kind == "week":
        df = df.resample("W-FRI", label="left", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
        df.index = df.index + pd.Timedelta(days=3)  # label left = previous Friday -> Monday of that week
    elif kind == "month":
        df = df.resample("MS").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    out = pd.DataFrame({"t": df.index, "o": df["open"].to_numpy(), "h": df["high"].to_numpy(), "l": df["low"].to_numpy(),
                        "c": df["close"].to_numpy(), "v": df["volume"].to_numpy(), "n": None, "t_last": df.index})
    return out.reset_index(drop=True), src


def _bars_json(df: pd.DataFrame) -> list[list]:
    return [[_ts(r.t), _num(r.o), _num(r.h), _num(r.l), _num(r.c), _num(r.v)] for r in df.itertuples()]


def _cache_path(date_str: str, code: str, tag: str) -> Path:
    return _runtime_root() / "cache" / "bars" / date_str / f"{code}_{tag}.json"


def _parse_tick_text(text: str, header: str, all_sessions: bool) -> pd.DataFrame:
    return load_ticks(io.StringIO(header + text), all_sessions)


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
    size = tf.stat().st_size
    with open(tf, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline()
        state = None
        if cache.is_file():
            with contextlib.suppress(Exception):
                state = json.loads(cache.read_text(encoding="utf-8"))
                if state.get("header") != header or state.get("offset", 0) > size:
                    state = None
        if state is not None and state.get("offset") == size:
            return _bars_from_json(state["bars"]), f"{tf} (cache)"
        offset = state["offset"] if state else len(header.encode("utf-8"))
        fh.seek(offset)
        text = fh.read()
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
        payload = {"header": header, "offset": offset + consumed, "bars": _bars_json(bars_df), "carry": _ticks_to_json(new_carry),
                   "last_acml": last_acml, "source": str(tf)}
        tmp = cache.with_suffix(f".{id(payload)}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(cache)
    return bars_df, str(tf)


def _prev_days(date_str: str, max_back: int = 10):
    base = datetime.strptime(date_str, "%Y%m%d")
    for back in range(1, max_back + 1):
        yield (base - timedelta(days=back)).strftime("%Y%m%d")


def bars(date_str: str, code: str, kind: str, n: int | None = None, interval: int = 1, ticks: int = 60, warmup: int = 0,
         since_ts: str | None = None, all_sessions: bool = False) -> dict:
    t0 = time.time()
    code = str(code).zfill(6)
    if kind in ("day", "week", "month"):
        res = day_bars(date_str, code, kind)
        if res is None:
            return {"ok": False, "error": f"no daily data for {code} around {date_str}"}
        df, src = res
        warm_rows, warm_date = [], None
    elif kind in ("tick", "min"):
        res = _intraday_bars(date_str, code, kind, ticks, interval, all_sessions)
        if res is None:
            return {"ok": False, "error": f"no tick file / _10s file for {code} on {date_str}"}
        df, src = res
        warm_rows, warm_date = [], None
        if warmup and warmup > 0:
            for d in _prev_days(date_str):
                prev = _intraday_bars(d, code, kind, ticks, interval, all_sessions)
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
    return {"ok": True, "kind": kind, "code": code, "date": date_str, "source": src,
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


def run_command(args) -> dict:
    if args.cmd == "indicators":
        return indicators(args.date, args.code, args.frame, args.last)
    return bars(args.date, args.code, args.kind, args.n, args.interval, args.ticks, args.warmup, args.since_ts, args.all_sessions)


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
