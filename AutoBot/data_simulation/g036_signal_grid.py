# -*- coding: utf-8 -*-
"""G036 indicator signal grid (2026-11-16 사용자 요청: 1분/3분/5분/60틱/120틱 봉에서 보조지표 매수 신호 하나, 또는 2~3개를
종합한 단순 조건 중 비용 후 수익이 나는 조건이 있는가).

Data
  1m / 3m / 5m  g026 minute-bar cache (runtime/edge_ai/timing_study/bars/<day>.pkl, liquid top-250 universe + traded codes,
                every collected day); 3m / 5m are resampled from 1m on the exchange clock from 09:00. Previous day's bars warm
                up the indicators.
  t60 / t120    g011 tick files (watchlist symbols only, since 2026-10-30), bars of n prints in receive order, prior-day
                tick file as warm-up.
Signals (computed on CLOSED bars only)
  events  ma_cross (MA5 x MA20 up), bb_mid (close x BB20 middle up), bb_low (close back above the lower band), bb_up (close
          above the upper band), macd (MACD x signal up), macd0 (MACD x 0 up), rsi30 / rsi50 (RSI14 x 30 / 50 up), stoch (%K x %D
          up from below 30), vol_spike (volume > 2 x MA20 of volume and an up bar), donchian (close > highest high of the 20
          bars before), vwap (close x session VWAP up)
  states  trend (close > MA60 and MA20 above its value 3 bars ago), above_vwap, rsi_lt70, day_up (close > the day's first open)
Rules
  single event (12) / two events (66) or three events (220) inside the last 3 bars (fires on the bar that completes the set)
  / event + state (48) / "always" baseline (every bar)
Trade model
  decision at the signal bar close, bar end 09:05..14:50; buy at the NEXT bar's open + half a tick, sell at the exit bar's
  close - half a tick (one tick round trip), fee 0.015 % per side + tax 0.20 %; one position per symbol (greedy, no overlap)
  exits: +10 / +30 / +60 minutes after the signal bar end, or the last bar at or before 15:15 (EOD); every exit is capped at
  15:15
Statistics
  per rule x exit: trades, mean / median net %, win %, day-clustered t (mean of the daily means / its standard error), excess
  over the "always" baseline of the same exit; train = first 2/3 of the days, test = the rest. Selection is done on TRAIN
  only (top rules by train t), and those rules are then read on TEST - the grid has ~7,000 cells, so a full-sample "best"
  is mostly noise.

  python g036_signal_grid.py [--workers 12] [--tfs 1m,3m,5m,t60,t120]
Output: <runtime>/data/sim/g036/<run>/grid.csv + report.md

Update log
- [2026-11-16] type=fix owner=claude
    summary: Codex 재검토 반영 - (1) 시간 목표 전에 데이터가 끝나도 그 전에 닿은 손절·익절은 거래로 인정, 끝까지 미청산이면
      제외 (2) 호가단위는 실제 진입가 / 청산가에서 계산 (3) 데이터 끊김 제외 뒤 같은 종목-일의 후속 신호는 보지 않음(포지션이
      남아 있는 것으로 처리) - truncated 는 이제 제외된 거래 수.
    impact: sim only
    compatibility: backward-compatible
- [2026-11-16] type=fix owner=claude
    summary: Codex 검토 반영 - (1) 시간 청산은 목표 시각(최대 15:15) 이후 첫 봉으로(소급 선택 제거) (2) 목표 시각 전에
      데이터가 끝난 거래는 통계에서 빼고 따로 집계 (3) 청산 봉의 신호로 다음 봉 진입 허용 (4) 거래 가중 평균에 맞는
      날짜 클러스터 t + 날짜 동일 가중 평균 병기 (5) 후보 선택은 학습 표본 수만 사용 (6) VWAP 은 틱 체결금액 / 분봉
      (H+L+C)/3 x V (7) 틱 종목-일 사후 필터 제거. 추가: 손절·익절 청산 2종, 상태 vol_up, 신호 + 상태 2개 조합, 비용 전 손익.
    impact: sim only
    compatibility: backward-compatible
- [2026-11-16] type=feat owner=claude
    summary: 신규 - 봉 종류 5개 x 보조지표 신호 단일/2개/3개 조합/상태 필터 346 규칙 x 청산 4개 전수 그리드(학습 선택 -> 검증 평가).
    impact: sim only
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))

import g026_timing_study as g026  # noqa: E402

FEE, TAX = 0.00015, 0.0020
ENTRY_FIRST_MIN, ENTRY_LAST_MIN, EOD_MIN = 9 * 60 + 5, 14 * 60 + 50, 15 * 60 + 15
EXITS = ("10m", "30m", "60m", "EOD", "TP1.5/SL1/60m", "TP2/SL1.5/EOD")
# kind 0 = time exit at the first bar ending at / after min(signal end + minutes, 15:15); kind 1 = take profit / stop loss on
# the bar path (stop first when one bar touches both - conservative; a gap through a level fills at that bar's open), else
# the time exit
EXIT_SPEC = ((0, 10.0, 0.0, 0.0), (0, 30.0, 0.0, 0.0), (0, 60.0, 0.0, 0.0), (0, 1e9, 0.0, 0.0),
             (1, 60.0, 0.015, 0.010), (1, 1e9, 0.020, 0.015))
WINDOW = 3
EVENTS = ("ma_cross", "bb_mid", "bb_low", "bb_up", "macd", "macd0", "rsi30", "rsi50", "stoch", "vol_spike", "donchian", "vwap")
STATES = ("trend", "above_vwap", "rsi_lt70", "day_up", "vol_up")
TICK_ROOT = g026.g013.DATA_ROOT / "ticks"


# ---------------------------------------------------------------------------------------------------------------------
# bars + signals per symbol-day
# ---------------------------------------------------------------------------------------------------------------------
def _krx_tick(p: np.ndarray) -> np.ndarray:
    t = np.full(p.shape, 1000.0)
    for lim, s in ((500_000, 500.0), (200_000, 100.0), (50_000, 50.0), (20_000, 10.0), (5_000, 5.0), (2_000, 1.0)):
        t = np.where(p < lim, s, t)
    return t


def resample(arr: np.ndarray, m: int) -> np.ndarray:
    """1-minute rows (end-label minute, o, h, l, c, v) -> m-minute rows labelled by their end minute (09:00 anchored)."""
    if m == 1 or len(arr) == 0:
        return arr
    lab = 540 + np.ceil((arr[:, 0] - 540) / m) * m
    df = pd.DataFrame(arr[:, 1:], columns=["o", "h", "l", "c", "v"])
    df["lab"] = lab
    g = df.groupby("lab", sort=True)
    out = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(), "v": g["v"].sum()})
    return np.column_stack([out.index.to_numpy(dtype=float), out.to_numpy(dtype=float)])


def signals(o, h, l, c, v, today: np.ndarray, pv: np.ndarray | None = None) -> dict:
    """Event / state flags for every bar (warm-up rows included; callers slice). today = bool mask of today's rows."""
    s = pd.Series(c)
    ma5, ma20, ma60 = s.rolling(5).mean(), s.rolling(20).mean(), s.rolling(60).mean()
    sd = s.rolling(20).std(ddof=0)
    up, lo = ma20 + 2 * sd, ma20 - 2 * sd
    ema12, ema26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100.0)
    rsi[s.index < 14] = np.nan
    hh, ll = pd.Series(h).rolling(14).max(), pd.Series(l).rolling(14).min()
    k = ((s - ll) / (hh - ll).replace(0, np.nan) * 100).rolling(3).mean()
    dd = k.rolling(3).mean()
    vma = pd.Series(v).rolling(20).mean()
    donch = pd.Series(h).rolling(20).max().shift(1)
    # session VWAP (today only) and the day's first open
    if pv is None:
        pv = (h + l + c) / 3.0 * v                  # minute bars: typical-price approximation (no traded value in the cache)
    cv = np.where(today, pv, 0.0).cumsum()
    vv = np.where(today, v, 0.0).cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(today & (vv > 0), cv / vv, np.nan)
    first = int(np.argmax(today)) if today.any() else len(c)
    day_open = o[first] if first < len(c) else np.nan

    def up_x(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pa, pb = np.r_[np.nan, a[:-1]], np.r_[np.nan, b[:-1]]
        with np.errstate(invalid="ignore"):
            return (a > b) & (pa <= pb)

    cc = s.to_numpy()
    prev_vwap_today = np.r_[False, today[:-1]]
    with np.errstate(invalid="ignore"):
        ev = {
            "ma_cross": up_x(ma5, ma20), "bb_mid": up_x(cc, ma20), "bb_low": up_x(cc, lo), "bb_up": up_x(cc, up),
            "macd": up_x(macd, sig), "macd0": up_x(macd, np.zeros(len(cc))), "rsi30": up_x(rsi, np.full(len(cc), 30.0)),
            "rsi50": up_x(rsi, np.full(len(cc), 50.0)),
            "stoch": up_x(k, dd) & (np.r_[np.nan, k.to_numpy()[:-1]] < 30),
            "vol_spike": (v > 2 * np.r_[np.nan, vma.to_numpy()[:-1]]) & (c > o),
            "donchian": cc > donch.to_numpy(),
            "vwap": up_x(cc, vwap) & prev_vwap_today,
        }
        st = {
            "trend": (cc > ma60.to_numpy()) & (ma20.to_numpy() > ma20.shift(3).to_numpy()),
            "above_vwap": cc > vwap, "rsi_lt70": rsi.to_numpy() < 70, "day_up": cc > day_open,
            "vol_up": v > 1.5 * np.r_[np.nan, vma.to_numpy()[:-1]],
        }
    return {**{f"e_{k_}": np.nan_to_num(v_, nan=0).astype(bool) for k_, v_ in ev.items()},
            **{f"s_{k_}": np.nan_to_num(v_, nan=0).astype(bool) for k_, v_ in st.items()}}


def _pack(rows: np.ndarray, warm: np.ndarray, end_today: np.ndarray, pv: np.ndarray | None = None) -> dict | None:
    """rows/warm: (end, o, h, l, c, v); pv = traded value per row (warm + today) or None. Returns today's arrays + flags."""
    if len(rows) < 2:
        return None
    allr = np.vstack([warm, rows]) if len(warm) else rows
    today = np.r_[np.zeros(len(warm), bool), np.ones(len(rows), bool)]
    f = signals(allr[:, 1], allr[:, 2], allr[:, 3], allr[:, 4], allr[:, 5], today, pv)
    sl = slice(len(warm), None)
    return {"end": end_today, "o": rows[:, 1], "h": rows[:, 2], "l": rows[:, 3], "c": rows[:, 4],
            **{k: a[sl] for k, a in f.items()}}


def minute_day(task) -> dict:
    date, prev_date, tfs = task
    day = g026.load_day(date)
    prev = g026.load_day(prev_date) if prev_date else None
    out = {tf: [] for tf in tfs}
    if not day:
        return {"date": date, "tfs": out}
    for code, arr in day["bars"].items():
        parr = prev["bars"].get(code) if prev else None
        for tf in tfs:
            m = int(tf[:-1])
            rows = resample(arr, m)
            warm = resample(parr, m)[-120:] if parr is not None and len(parr) else np.empty((0, 6))
            p = _pack(rows, warm, rows[:, 0] if len(rows) else rows)
            if p:
                out[tf].append(p)
    return {"date": date, "tfs": out}


def tick_day(task) -> dict:
    date, tfs = task
    import g034_entry_bar_study as g034
    out = {tf: [] for tf in tfs}
    for f in sorted((TICK_ROOT / date).glob("*.csv")):
        t = g034.load_ticks(f)
        if t.empty:
            continue
        prior = None
        for d in sorted((p.name for p in TICK_ROOT.iterdir() if p.is_dir() and p.name.isdigit() and p.name < date),
                        reverse=True)[:3]:
            pf = TICK_ROOT / d / f.name
            if pf.exists():
                prior = g034.load_ticks(pf)
                break
        for tf in tfs:
            n = int(tf[1:])
            b = g034.tick_bars(t, n)
            if len(b) < 2:
                continue
            m_ = (len(t) // n) * n
            q = t["qty"].to_numpy()[:m_].reshape(-1, n).sum(1).astype(float)
            pv = (t["qty"].to_numpy()[:m_] * t["price"].to_numpy()[:m_]).reshape(-1, n).sum(1).astype(float)
            rows = np.column_stack([b["avail"].to_numpy() / 60.0, b["open"], b["high"], b["low"], b["close"], q])
            warm, wpv = np.empty((0, 6)), np.empty(0)
            if prior is not None and len(prior) >= n:
                pb = g034.tick_bars(prior, n)
                pm = (len(prior) // n) * n
                pq = prior["qty"].to_numpy()[:pm].reshape(-1, n).sum(1).astype(float)
                warm = np.column_stack([np.zeros(len(pb)), pb["open"], pb["high"], pb["low"], pb["close"], pq])[-120:]
                wpv = np.zeros(len(warm))                    # warm rows never enter the session VWAP
            p = _pack(rows, warm, rows[:, 0], np.r_[wpv, pv])
            if p:
                out[tf].append(p)
    return {"date": date, "tfs": out}


# ---------------------------------------------------------------------------------------------------------------------
# grid evaluation (numba)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _recent(ev, seg, w):
    n = ev.shape[0]
    out = np.zeros(n, np.bool_)
    last = -10 ** 9
    for i in range(n):
        if i == 0 or seg[i] != seg[i - 1]:
            last = -10 ** 9
        if ev[i]:
            last = i
        out[i] = i - last < w
    return out


@njit(cache=True)
def _exit_index(end, seg, h):
    """For each bar i: index j > i in the same segment of the FIRST bar whose end >= min(end[i] + h, 15:15) (causal: the
    exit is decided at the target time and executed on the first bar that completes after it). -2 when the segment ends
    before that (data stopped - reported apart, never counted as a normal exit); -1 when i is the last bar."""
    n = end.shape[0]
    out = np.full(n, -1, np.int64)
    i = 0
    while i < n:
        s = i
        while i < n and seg[i] == seg[s]:
            i += 1
        e = i
        j = s
        for k in range(s, e):
            if j <= k:
                j = k + 1
            target = end[k] + h
            if target > 915.0:
                target = 915.0
            while j < e and end[j] < target:
                j += 1
            if k + 1 >= e:
                out[k] = -1
            elif j < e:
                out[k] = j
            else:
                out[k] = -2
    return out


@njit(cache=True)
def _tick(p):
    """KRX tick size of one price (same table as _krx_tick)."""
    if p < 2000.0:
        return 1.0
    if p < 5000.0:
        return 5.0
    if p < 20000.0:
        return 10.0
    if p < 50000.0:
        return 50.0
    if p < 200000.0:
        return 100.0
    if p < 500000.0:
        return 500.0
    return 1000.0


@njit(parallel=True, cache=True)
def _grid(kind, comp, E, R, S, entry_ok, seg, seg_last, xkind, xidx, xtp, xsl, o, h, l, c, day, n_days):
    """kind: 0 single event, 1 event combo (comp = event ids, -1 padded), 2 event + state, 4 event + two states (comp[1..2]),
    3 always. Exit x: xkind 0 time / 1 tp-sl, xidx = time-exit bar (-2 = the data ends before the target). A tp / sl hit
    before the data ends is a normal trade; a position still open when the data ends is excluded and counted in `trunc`,
    and the symbol-day takes no further entry (the position would still be held). Tick sizes come from the actual entry /
    exit prices. Returns per (rule, exit, day): net sum, trades, wins, gross sum (same fills, no slippage / fee / tax)."""
    n_rules = kind.shape[0]
    n_exit = xidx.shape[0]
    n = seg.shape[0]
    tot = np.zeros((n_rules, n_exit, n_days))
    cnt = np.zeros((n_rules, n_exit, n_days))
    win = np.zeros((n_rules, n_exit, n_days))
    gro = np.zeros((n_rules, n_exit, n_days))
    trunc = np.zeros((n_rules, n_exit))
    for r in prange(n_rules):
        for x in range(n_exit):
            busy = -1
            prev_all = False
            for i in range(n):
                if i > 0 and seg[i] != seg[i - 1]:
                    prev_all = False
                k = kind[r]
                if k == 0:
                    fire = E[comp[r, 0], i]
                elif k == 1:
                    allr = True
                    for cc in range(3):
                        if comp[r, cc] >= 0 and not R[comp[r, cc], i]:
                            allr = False
                    fire = allr and not prev_all
                    prev_all = allr
                elif k == 2:
                    fire = E[comp[r, 0], i] and S[comp[r, 1], i]
                elif k == 4:
                    fire = E[comp[r, 0], i] and S[comp[r, 1], i] and S[comp[r, 2], i]
                else:
                    fire = True
                if not fire or not entry_ok[i] or i < busy:     # the exit bar's own signal may enter at the next open
                    continue
                jt = xidx[x, i]
                if jt == -1:
                    continue
                e0 = o[i + 1]
                epx = e0 + 0.5 * _tick(e0)
                j = -1
                raw = 0.0
                if xkind[x] == 1:
                    tp = epx * (1.0 + xtp[x])
                    sl = epx * (1.0 - xsl[x])
                    last = jt if jt >= 0 else seg_last[i]
                    for q in range(i + 1, last + 1):
                        if q > i + 1 and o[q] <= sl:
                            j, raw = q, o[q]
                            break
                        if q > i + 1 and o[q] >= tp:
                            j, raw = q, o[q]
                            break
                        if l[q] <= sl:
                            j, raw = q, sl
                            break
                        if h[q] >= tp:
                            j, raw = q, tp
                            break
                if j < 0:
                    if jt == -2:                                # still open when the data ends: excluded, symbol-day done
                        trunc[r, x] += 1.0
                        busy = seg_last[i] + 1
                        continue
                    j, raw = jt, c[jt]
                xpx = raw - 0.5 * _tick(raw)
                net = (xpx * (1.0 - 0.00015 - 0.0020) - epx * 1.00015) / epx * 100.0
                d = day[i]
                tot[r, x, d] += net
                gro[r, x, d] += (raw / e0 - 1.0) * 100.0
                cnt[r, x, d] += 1.0
                if net > 0:
                    win[r, x, d] += 1.0
                busy = j
    return tot, cnt, win, gro, trunc


def rules() -> list[tuple[str, int, tuple]]:
    out = [(e, 0, (i, -1, -1)) for i, e in enumerate(EVENTS)]
    for a, b in itertools.combinations(range(len(EVENTS)), 2):
        out.append((f"{EVENTS[a]}+{EVENTS[b]}", 1, (a, b, -1)))
    for a, b, c in itertools.combinations(range(len(EVENTS)), 3):
        out.append((f"{EVENTS[a]}+{EVENTS[b]}+{EVENTS[c]}", 1, (a, b, c)))
    for i, e in enumerate(EVENTS):
        for j, s in enumerate(STATES):
            out.append((f"{e}|{s}", 2, (i, j, -1)))
    for i, e in enumerate(EVENTS):
        for a, b in itertools.combinations(range(len(STATES)), 2):
            out.append((f"{e}|{STATES[a]}&{STATES[b]}", 4, (i, a, b)))
    out.append(("ALWAYS", 3, (-1, -1, -1)))
    return out


def evaluate(parts: list[tuple[int, dict]], n_days: int):
    """parts: (day index, symbol-day pack). Concatenates and runs the grid."""
    seg = np.concatenate([np.full(len(p["end"]), k, np.int64) for k, (_d, p) in enumerate(parts)])
    day = np.concatenate([np.full(len(p["end"]), d, np.int64) for d, p in parts])
    end = np.concatenate([p["end"] for _d, p in parts]).astype(float)
    o = np.concatenate([p["o"] for _d, p in parts]).astype(float)
    hh = np.concatenate([p["h"] for _d, p in parts]).astype(float)
    ll = np.concatenate([p["l"] for _d, p in parts]).astype(float)
    c = np.concatenate([p["c"] for _d, p in parts]).astype(float)
    E = np.vstack([np.concatenate([p[f"e_{e}"] for _d, p in parts]) for e in EVENTS])
    S = np.vstack([np.concatenate([p[f"s_{s}"] for _d, p in parts]) for s in STATES])
    R = np.vstack([_recent(E[i], seg, WINDOW) for i in range(len(EVENTS))])
    nxt_same = np.r_[seg[1:] == seg[:-1], False]
    entry_ok = (end >= ENTRY_FIRST_MIN) & (end <= ENTRY_LAST_MIN) & nxt_same
    nxt_open = np.r_[o[1:], np.nan]
    entry_ok &= np.isfinite(nxt_open) & (nxt_open > 0)
    seg_last = np.empty(len(seg), np.int64)                      # last index of each bar's segment
    brk = np.r_[np.flatnonzero(seg[1:] != seg[:-1]), len(seg) - 1]
    seg_last[:] = np.repeat(brk, np.diff(np.r_[-1, brk]))
    xkind = np.array([k for k, _h, _tp, _sl in EXIT_SPEC], np.int64)
    xtp = np.array([tp for _k, _h, tp, _sl in EXIT_SPEC])
    xsl = np.array([sl for _k, _h, _tp, sl in EXIT_SPEC])
    xidx = np.vstack([_exit_index(end, seg, hm) for _k, hm, _tp, _sl in EXIT_SPEC])
    rl = rules()
    kind = np.array([k for _n, k, _c in rl], np.int64)
    comp = np.array([c_ for _n, _k, c_ in rl], np.int64)
    return _grid(kind, comp, E, R, S, entry_ok, seg, seg_last, xkind, xidx, xtp, xsl, o, hh, ll, c, day, n_days)


def _cluster_t(t: np.ndarray, n: np.ndarray) -> float:
    """t of the trade-weighted mean with day-clustered standard errors (Codex review #4)."""
    m = n > 0
    D = int(m.sum())
    N = n[m].sum()
    if D < 3 or N <= 0:
        return np.nan
    mean = t[m].sum() / N
    resid = t[m] - mean * n[m]
    var = (resid ** 2).sum() * D / (D - 1) / N ** 2
    return mean / np.sqrt(var) if var > 0 else np.nan


def stats(tot, cnt, win, gro, trunc, days: list[str], tf: str) -> pd.DataFrame:
    n_days = len(days)
    split = int(round(n_days * 2 / 3))
    rl = rules()
    rows = []
    for r, (name, _k, _c) in enumerate(rl):
        for x, ex in enumerate(EXITS):
            t, n, w, g = tot[r, x], cnt[r, x], win[r, x], gro[r, x]
            rec = {"tf": tf, "rule": name, "exit": ex, "n": int(n.sum()), "truncated": int(trunc[r, x])}
            for part, sl in (("all", slice(None)), ("train", slice(0, split)), ("test", slice(split, None))):
                tt, nn, ww, gg = t[sl], n[sl], w[sl], g[sl]
                N = nn.sum()
                rec[f"{part}_n"] = int(N)
                rec[f"{part}_mean"] = tt.sum() / N if N else np.nan
                rec[f"{part}_gross"] = gg.sum() / N if N else np.nan
                rec[f"{part}_win"] = ww.sum() / N * 100 if N else np.nan
                dm = tt[nn > 0] / nn[nn > 0]
                rec[f"{part}_days"] = len(dm)
                rec[f"{part}_daymean"] = dm.mean() if len(dm) else np.nan
                rec[f"{part}_t"] = _cluster_t(tt, nn)
                rec[f"{part}_pos_days"] = float((dm > 0).mean()) if len(dm) else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    b = df[df["rule"] == "ALWAYS"].set_index("exit")
    for part in ("all", "train", "test"):
        df[f"{part}_excess"] = df[f"{part}_mean"] - df["exit"].map(b[f"{part}_mean"])
    return df


# ---------------------------------------------------------------------------------------------------------------------
def _q(v, fmt="+.3f"):
    return "-" if v is None or v != v else format(v, fmt)


def report(df: pd.DataFrame, meta: dict) -> str:
    L = [f"# g036 보조지표 신호 그리드 ({datetime.now():%Y-%m-%d})", "",
         "다음 봉 시가 + 반 틱 매수, 청산 봉 종가 - 반 틱 매도, 수수료 0.015% x 2 + 세금 0.20% 포함 순손익 %. 종목당 동시 보유 1개.",
         "excess = 같은 청산의 무조건 매수(ALWAYS) 대비. t = 날짜별 평균의 t 값. 학습에서 고르고 검증에서 읽는다.", ""]
    for tf, m in meta.items():
        L.append(f"- {tf}: {m}")
    L += ["", "## 1. 기준선: 무조건 매수(ALWAYS)", "",
          "| 봉 | 청산 | 거래 | 순손익 평균 | 비용 전 | 승률 | 학습 | 검증 | 데이터 끊김으로 제외된 거래 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df[df["rule"] == "ALWAYS"].iterrows():
        L.append(f"| {r.tf} | {r.exit} | {r.n:,} | {_q(r.all_mean)} | {_q(r.all_gross)} | {_q(r.all_win, '.0f')}% | "
                 f"{_q(r.train_mean)} | {_q(r.test_mean)} | {r.truncated:,} |")
    min_n = {"1m": 300, "3m": 200, "5m": 150, "t60": 60, "t120": 40}
    L += ["", "## 2. 전체 기간 평균이 양수인 규칙 수 (표본 하한 이상)", "",
          "| 봉 | 규칙 x 청산 | 표본 충분 | 평균 > 0 | 학습 > 0 이고 검증 > 0 | 그중 학습 t >= 2 |", "|---|---|---|---|---|---|"]
    for tf, g in df[df["rule"] != "ALWAYS"].groupby("tf", sort=False):
        ok = g[g["n"] >= min_n.get(tf, 100)]
        both = ok[(ok["train_mean"] > 0) & (ok["test_mean"] > 0)]
        L.append(f"| {tf} | {len(g)} | {len(ok)} | {(ok['all_mean'] > 0).sum()} | {len(both)} | {(both['train_t'] >= 2).sum()} |")
    L += ["", "## 3. 학습 구간 t 상위 15개 규칙 -> 검증 구간 결과 (봉별)", ""]
    for tf, g in df[df["rule"] != "ALWAYS"].groupby("tf", sort=False):
        ok = g[g["train_n"] >= min_n.get(tf, 100) * 2 // 3].sort_values("train_t", ascending=False).head(15)   # train only
        L += [f"### {tf}", "", "| 규칙 | 청산 | 학습 거래 | 학습 평균 | 학습 t | 검증 거래 | 검증 평균 | 검증 t | 검증 excess | 검증 승률 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in ok.iterrows():
            L.append(f"| {r.rule} | {r.exit} | {r.train_n} | {_q(r.train_mean)} | {_q(r.train_t, '+.2f')} | {r.test_n} | "
                     f"{_q(r.test_mean)} | {_q(r.test_t, '+.2f')} | {_q(r.test_excess)} | {_q(r.test_win, '.0f')}% |")
        if len(ok):
            L.append(f"- 상위 15개 검증 평균의 평균 {ok['test_mean'].mean():+.3f}%, 검증 양수 {int((ok['test_mean'] > 0).sum())}/{len(ok)}")
        L.append("")
    L += ["## 4. 전체 기간 평균 상위 20개 (참고 - 다중 비교라 대부분 우연일 수 있음)", "",
          "| 봉 | 규칙 | 청산 | 거래 | 평균 | 승률 | t | 학습 | 검증 |", "|---|---|---|---|---|---|---|---|---|"]
    ok = df[(df["rule"] != "ALWAYS") & (df["n"] >= df["tf"].map(min_n).fillna(100))]
    for _, r in ok.sort_values("all_mean", ascending=False).head(20).iterrows():
        L.append(f"| {r.tf} | {r.rule} | {r.exit} | {r.n} | {_q(r.all_mean)} | {_q(r.all_win, '.0f')}% | {_q(r.all_t, '+.2f')} | "
                 f"{_q(r.train_mean)} | {_q(r.test_mean)} |")
    L += ["", "## 4-1. 비용 전 평균이 비용(약 0.23% + 1틱)을 넘는 규칙 (학습·검증 모두 비용 전 > 0.4%)", "",
          "| 봉 | 규칙 | 청산 | 거래 | 비용 전 학습 | 비용 전 검증 | 순손익 |", "|---|---|---|---|---|---|---|"]
    gx = ok[(ok["train_gross"] > 0.4) & (ok["test_gross"] > 0.4)].sort_values("all_gross", ascending=False).head(20)
    for _, r in gx.iterrows():
        L.append(f"| {r.tf} | {r.rule} | {r.exit} | {r.n} | {_q(r.train_gross)} | {_q(r.test_gross)} | {_q(r.all_mean)} |")
    if gx.empty:
        L.append("| (없음) | | | | | | |")
    L += ["", "## 5. 신호별 평균 (단일 신호, 청산별)", "", "| 봉 | 신호 | " + " | ".join(EXITS) + " |",
          "|---|---|" + "---|" * len(EXITS)]
    single = df[df["rule"].isin(EVENTS)]
    for (tf, rule), g in single.groupby(["tf", "rule"], sort=False):
        cells = [f"{_q(g[g['exit'] == e]['all_mean'].iloc[0])} ({int(g[g['exit'] == e]['n'].iloc[0])})" for e in EXITS]
        L.append(f"| {tf} | {rule} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tfs", default="1m,3m,5m,t60,t120")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tfs = args.tfs.split(",")
    out = Path(args.out) if args.out else g026.g013.DATA_ROOT / "sim" / "g036" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    frames, meta = [], {}
    mtfs = [t for t in tfs if t.endswith("m")]
    if mtfs:
        dates = sorted(p.stem for p in g026.BARS_DIR.glob("*.pkl"))
        tasks = [(d, dates[i - 1] if i else None, mtfs) for i, d in enumerate(dates)][1:]   # first day = warm-up only
        days = [t[0] for t in tasks]
        parts = {tf: [] for tf in mtfs}
        with Pool(args.workers) as pool:
            for k, res in enumerate(pool.imap(minute_day, tasks), 1):
                di = days.index(res["date"])
                for tf in mtfs:
                    parts[tf] += [(di, p) for p in res["tfs"][tf]]
                if k % 10 == 0:
                    print(f"minute days {k}/{len(tasks)}", flush=True)
        for tf in mtfs:
            print(f"grid {tf}: symbol-days {len(parts[tf])}", flush=True)
            tot, cnt, win, gro, trunc = evaluate(parts[tf], len(days))
            frames.append(stats(tot, cnt, win, gro, trunc, days, tf))
            meta[tf] = f"{days[0]}~{days[-1]} {len(days)}거래일, 종목-일 {len(parts[tf]):,}, 봉 {sum(len(p['end']) for _d, p in parts[tf]):,}"
            parts[tf] = None
    ttfs = [t for t in tfs if t.startswith("t")]
    if ttfs:
        tdays = sorted(p.name for p in TICK_ROOT.iterdir() if p.is_dir() and p.name.isdigit())
        parts = {tf: [] for tf in ttfs}
        with Pool(min(args.workers, len(tdays))) as pool:
            for res in pool.imap(tick_day, [(d, ttfs) for d in tdays]):
                di = tdays.index(res["date"])
                for tf in ttfs:
                    parts[tf] += [(di, p) for p in res["tfs"][tf]]
        for tf in ttfs:
            tot, cnt, win, gro, trunc = evaluate(parts[tf], len(tdays))
            frames.append(stats(tot, cnt, win, gro, trunc, tdays, tf))
            meta[tf] = f"{tdays[0]}~{tdays[-1]} {len(tdays)}거래일(감시 종목만), 종목-일 {len(parts[tf]):,}"
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "grid.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text(report(df, meta), encoding="utf-8")
    with open(out / "meta.pkl", "wb") as fh:
        pickle.dump(meta, fh)
    print(f"out: {out}")


if __name__ == "__main__":
    main()
