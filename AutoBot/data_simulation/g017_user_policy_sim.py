# -*- coding: utf-8 -*-
"""User position policy simulator + dataset (2026-10-31 사용자 요구사항 1~14).

Stage 1 (default): for every trading day folder in runtime/data, every liquid common stock whose DAILY chart is
rising (D-1 close > MA20 and MA20 rising), find 1-min MA5>MA20 golden-cross candidate bars and simulate the user
position policy on 1-min OHLC for a grid of parameters. Writes one events_<date>.csv.gz per day:
  candidate filter columns (daily trend, 1m/3m/BB state)  -> tuned offline (g018)
  per grid combo k: net_krw_k (round-trip net PnL in KRW after fee/tax)  -> win = net_krw_k > 0
Stage 2 (--features <rule.json>): recompute r011 compute_features() for the events that pass the tuned entry rule
and attach the label of the chosen policy combo -> Edge AI training set (g014 --tag).

Position policy (r001 USER_POLICY_* live parity; prices on 1-min bars, path o->l->h->c on bullish bars and
o->h->l->c on bearish bars, gaps fill at the open):
  entry   : next bar open + 1 tick, 1st buy = BUY1_KRW (300k)
  add     : once - pyramid when price >= first_entry*(1+pyr) / average down when price <= first_entry*(1-avgdn),
            ADD_KRW (200k). Peak is reset to the add price.
  trailing: armed when peak >= avg*(1+arm); price <= peak*(1-t1) -> sell 50% of current, <= peak*(1-t2) -> sell all
  take    : avg+tp1 sell 50% of current, avg+tp2 sell 50% of current, avg+tp3 sell all
  stop    : price <= avg*(1-stop) -> sell all
  eod     : first bar >= 15:15 -> sell all at close
Costs: fee 0.015% each side + sell tax 0.20%; market-type fills (stops, trailing, entry, pyramid) 1 tick slippage,
take-profit/average-down are limit-type (no slippage).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))

import g013_edge_ai_dataset as g013  # noqa: E402
from r011_edge_ai import (  # noqa: E402
    build_strategy_config, compute_features, daily_features_from_frame, fast_indicators, session_raw_1min,
)

OUT_ROOT = g013.EDGE_AI_DIR / "policy"
ENTRY_START = pd.Timestamp("09:05").time()
ENTRY_END = pd.Timestamp("14:30").time()
EOD_MINUTE = 15 * 60 + 15
BUY1_KRW = 300_000.0
ADD_KRW = 200_000.0
FEE = 0.00015
TAX = 0.0020
MIN_GAP_BARS = 5          # min bars between two candidate events of the same code
CROSS_LOOKBACK = 3        # 1m MA5>MA20 cross at most this many bars ago

# grid: (arm, t1, t2, stop, pyr, avgdn, tp1, tp2, tp3) in percent. avgdn/pyr 0 = disabled.
GRID_AXES = {
    "arm": (0.5, 0.8, 1.2),
    "t1": (0.5,),
    "t2": (0.8, 1.0),
    "stop": (1.0, 1.5, 2.0, 3.0),
    "pyr": (0.0, 0.5, 1.0),
    "avgdn": (0.0, 1.0),
    "tp1": (2.0,),
    "tp2": (3.0,),
    "tp3": (4.0,),
}


def grid() -> np.ndarray:
    return np.array(list(itertools.product(*GRID_AXES.values())), dtype=np.float64)


# ---------------------------------------------------------------------------
# policy simulation (numba)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _tick(p):
    if p < 2000:
        return 1.0
    if p < 5000:
        return 5.0
    if p < 20000:
        return 10.0
    if p < 50000:
        return 50.0
    if p < 200000:
        return 100.0
    if p < 500000:
        return 500.0
    return 1000.0


@njit(cache=True)
def _up(p):
    """Round a buy price up to a valid KRX quote."""
    t = _tick(p)
    return np.ceil(p / t - 1e-9) * t


@njit(cache=True)
def _dn(p):
    """Round a sell price down to a valid KRX quote."""
    t = _tick(p)
    return np.floor(p / t + 1e-9) * t


@njit(cache=True)
def simulate(o, h, l, c, obs, minute, i, prm, buy1, add_krw):
    """Returns (net_krw, invested_max, reason, add_type, hold_bars). reason: 0 no-trade 1 tp3 2 trail 3 stop 4 eod
    5 end. add_type: 0 none 1 pyramid 2 average-down. obs[j] False = synthetic (no-trade) minute: no fills there.
    Fills: market-type (entry, pyramid, stops, trailing) = trigger rounded to a quote +-1 tick; limit-type
    (average-down, take-profit) = the trigger rounded to a quote (touch fill). A gap through a trigger fills at
    the open (+-1 tick for market-type)."""
    arm, t1, t2, stop, pyr, avgdn, tp1, tp2, tp3 = (prm[0] / 100, prm[1] / 100, prm[2] / 100, prm[3] / 100,
                                                    prm[4] / 100, prm[5] / 100, prm[6] / 100, prm[7] / 100,
                                                    prm[8] / 100)
    n = len(c)
    e = i + 1
    while e < n and not obs[e]:
        e += 1
    if e >= n or minute[e] >= EOD_MINUTE:
        return 0.0, 0.0, 0, 0, 0
    e1 = _up(o[e]) + _tick(o[e])
    qty = np.floor(buy1 / e1)
    if qty < 1:
        return 0.0, 0.0, 0, 0, 0   # budget below one share: the live bot would not buy
    cash = -e1 * qty * (1 + FEE)
    cost = e1 * qty                # cost basis of the shares still held (avg price = cost / qty)
    invested = cost
    peak = e1
    armed = False
    add_done = False
    add_type = 0
    tp_stage = 0
    trail_stage = 0
    prev = e1
    pts = np.empty(4)
    lvl = np.empty(3)
    lvl[0], lvl[1], lvl[2] = tp1, tp2, tp3
    for j in range(e, n):
        if obs[j]:
            if c[j] >= o[j]:
                pts[0], pts[1], pts[2], pts[3] = o[j], l[j], h[j], c[j]
            else:
                pts[0], pts[1], pts[2], pts[3] = o[j], h[j], l[j], c[j]
            start_k = 0 if j > e else 1  # the entry bar starts at its open (the fill)
            for k in range(start_k, 4):
                p0 = prev
                p1 = pts[k]
                gap = k == 0
                if p1 > p0:
                    # ascending triggers: pyramid, tp1, tp2, tp3 (re-evaluated after each fill)
                    progressed = True
                    while progressed and qty > 0:
                        progressed = False
                        avg = cost / qty
                        if not add_done and pyr > 0 and p1 >= e1 * (1 + pyr):
                            trig = e1 * (1 + pyr)
                            fill = (p1 + _tick(p1)) if gap else (_up(trig) + _tick(trig))
                            q2 = np.floor(add_krw / fill)
                            if q2 >= 1:
                                cash -= fill * q2 * (1 + FEE)
                                cost += fill * q2
                                qty += q2
                                invested = max(invested, cost)
                                peak = fill
                                armed = False
                                add_type = 1
                            add_done = True
                            progressed = True
                            continue
                        if tp_stage < 3 and p1 >= _up(avg * (1 + lvl[tp_stage])):
                            trig = _up(avg * (1 + lvl[tp_stage]))
                            fill = p1 if gap else trig
                            if tp_stage == 2:
                                sq = qty
                            else:
                                sq = np.floor(qty * 0.5)
                                if sq < 1:
                                    sq = qty
                            cash += fill * sq * (1 - FEE - TAX)
                            cost -= avg * sq
                            qty -= sq
                            tp_stage += 1
                            progressed = True
                    if qty <= 0:
                        return cash, invested, 1, add_type, j - i
                    if p1 > peak:
                        peak = p1
                    if qty > 0 and peak >= (cost / qty) * (1 + arm):
                        armed = True   # sticky once the rise was seen
                elif p1 < p0:
                    progressed = True
                    while progressed and qty > 0:
                        progressed = False
                        avg = cost / qty
                        # descending order of trigger prices
                        cands_px = np.full(4, -1.0)
                        if armed and trail_stage == 0:
                            cands_px[0] = peak * (1 - t1)
                        if armed:
                            cands_px[1] = peak * (1 - t2)
                        if not add_done and avgdn > 0:
                            cands_px[2] = _dn(e1 * (1 - avgdn))
                        cands_px[3] = avg * (1 - stop)
                        best = -1
                        for q in range(4):
                            if cands_px[q] > 0 and p1 <= cands_px[q] and (best < 0 or cands_px[q] > cands_px[best]):
                                best = q
                        if best < 0:
                            break
                        trig = cands_px[best]
                        if best == 2:  # average down (resting limit at the trigger)
                            fill = p1 if gap else trig
                            q2 = np.floor(add_krw / fill)
                            if q2 >= 1:
                                cash -= fill * q2 * (1 + FEE)
                                cost += fill * q2
                                qty += q2
                                invested = max(invested, cost)
                                add_type = 2
                            add_done = True
                            peak = fill
                            armed = False
                            progressed = True
                            continue
                        fill = (p1 - _tick(p1)) if gap else (_dn(trig) - _tick(trig))
                        if best == 0:
                            sq = np.floor(qty * 0.5)
                            if sq < 1:
                                sq = qty
                            trail_stage = 1
                        else:
                            sq = qty
                        cash += fill * sq * (1 - FEE - TAX)
                        cost -= avg * sq
                        qty -= sq
                        progressed = True
                        if qty <= 0:
                            return cash, invested, 3 if best == 3 else 2, add_type, j - i
                prev = p1
        if minute[j] >= EOD_MINUTE:   # end-labelled bar: its intrabar path ran first, then liquidate at its close
            px = prev - _tick(prev)
            cash += px * qty * (1 - FEE - TAX)
            return cash, invested, 4, add_type, j - i
    px = prev - _tick(prev)
    cash += px * qty * (1 - FEE - TAX)
    return cash, invested, 5, add_type, n - 1 - i


@njit(cache=True)
def simulate_grid(o, h, l, c, obs, minute, i, prms, buy1, add_krw):
    m = prms.shape[0]
    net = np.empty(m)
    reason = np.empty(m, dtype=np.int64)
    addt = np.empty(m, dtype=np.int64)
    hold = np.empty(m, dtype=np.int64)
    inv = np.empty(m)
    for k in range(m):
        net[k], inv[k], reason[k], addt[k], hold[k] = simulate(o, h, l, c, obs, minute, i, prms[k], buy1, add_krw)
    return net, inv, reason, addt, hold


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def daily_trend(daily: pd.DataFrame | None, trade_date: str) -> dict | None:
    """D-1 daily trend metrics (percent). None when history is too short."""
    if daily is None or daily.empty:
        return None
    d = daily[daily["date"].astype(str).str.slice(0, 8) < trade_date]
    if len(d) < 65:
        return None
    close = pd.to_numeric(d["close"], errors="coerce").to_numpy(float)
    ma5 = pd.Series(close).rolling(5).mean().to_numpy()
    ma20 = pd.Series(close).rolling(20).mean().to_numpy()
    ma60 = pd.Series(close).rolling(60).mean().to_numpy()
    amount = pd.to_numeric(d["amount"], errors="coerce").to_numpy(float)
    return {
        "prev_close": close[-1],
        "d_close_ma20": (close[-1] / ma20[-1] - 1) * 100,
        "d_ma5_ma20": (ma5[-1] / ma20[-1] - 1) * 100,
        "d_ma20_ma60": (ma20[-1] / ma60[-1] - 1) * 100,
        "d_ma20_slope5": (ma20[-1] / ma20[-6] - 1) * 100,
        "d_ma60_slope5": (ma60[-1] / ma60[-6] - 1) * 100,
        "d_ret5": (close[-1] / close[-6] - 1) * 100,
        "d_ret20": (close[-1] / close[-21] - 1) * 100,
        "d_turnover_log": float(np.log10(max(np.nanmean(amount[-20:]), 1.0))),
    }


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()


def symbol_events(code: str, trade_date: str, minute_file: Path, daily_file: Path | None, prms: np.ndarray) -> list[dict]:
    daily = g013._read_daily(daily_file)
    dt = daily_trend(daily, trade_date)
    if dt is None or not (dt["d_close_ma20"] > 0 and dt["d_ma20_slope5"] > 0):
        return []
    raw = session_raw_1min(g013.load_minute_bars(minute_file), pd.Timestamp(trade_date))
    if raw is None or len(raw) < 40:
        return []
    # full minute grid (missing minutes = no trade: flat bar at the previous close, volume 0)
    idx = pd.date_range(raw.index[0], raw.index[-1], freq="1min")
    raw = raw.reindex(idx)
    obs = raw["close"].notna().to_numpy()   # real (traded) minutes - fills only happen there
    raw["close"] = raw["close"].ffill()
    for k in ("open", "high", "low"):
        raw[k] = raw[k].fillna(raw["close"])
    raw["volume"] = raw["volume"].fillna(0.0)
    m1 = fast_indicators(raw)
    o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = m1["volume"].to_numpy(float)
    ma5, ma20 = _sma(c, 5), _sma(c, 20)
    mid, up, lo = (m1[k].to_numpy(float) for k in ("BB_MIDDLE", "BB_UPPER", "BB_LOWER"))
    rsi = m1["RSI"].to_numpy(float) if "RSI" in m1.columns else np.full(len(c), np.nan)
    times = m1.index
    minute = (times.hour * 60 + times.minute).to_numpy(np.int64)
    # 3m completed bars, MA5/MA20, mapped to each 1m bar (last completed 3m bucket <= now)
    m3 = raw[["open", "high", "low", "close", "volume"]].resample("3min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    c3 = m3["close"].to_numpy(float)
    m3_ma5, m3_ma20 = _sma(c3, 5), _sma(c3, 20)
    m3_mid_fast = _sma(c3, 10)
    pos3 = np.searchsorted(m3.index.to_numpy(), times.to_numpy(), side="right") - 1
    above3 = m3_ma5 > m3_ma20
    # bars since the current "above" run of 3m MA5>MA20 began (-1 when below / unknown)
    age3 = np.full(len(c3), -1, dtype=np.int64)
    for k in range(1, len(c3)):
        if above3[k] and not np.isnan(m3_ma20[k]):
            age3[k] = age3[k - 1] + 1 if (above3[k - 1] and age3[k - 1] >= 0) else (0 if not np.isnan(m3_ma20[k - 1]) else -1)
    day_open = o[0]
    cum_pv, cum_v = np.cumsum(c * v), np.cumsum(v)
    vol_ma20 = _sma(v, 20)
    events = []
    last_ev = -10**9
    for i in range(20, len(c) - 1):
        t = times[i].time()
        if t < ENTRY_START or t > ENTRY_END or i - last_ev < MIN_GAP_BARS:
            continue
        if not obs[i] or not (ma5[i] > ma20[i]):
            continue
        cross_age = -1
        for b in range(0, CROSS_LOOKBACK + 1):
            if i - b - 1 >= 0 and ma5[i - b] > ma20[i - b] and ma5[i - b - 1] <= ma20[i - b - 1]:
                cross_age = b
                break
        if cross_age < 0:
            continue
        # bars since close crossed above the BB mid (current above run)
        bb_age = -1
        if c[i] > mid[i]:
            b = 0
            while i - b - 1 >= 0 and c[i - b - 1] > mid[i - b - 1]:
                b += 1
            bb_age = b
        k3 = pos3[i]
        net, inv, reason, addt, hold = g_sim(o, h, l, c, obs, minute, i, prms)
        if reason[REF] == 0:
            continue   # no fill possible (no later traded minute / budget below one share)
        rec = {"date": trade_date, "code": code, "ts": times[i].strftime("%H:%M"), "bar_i": i, "price": c[i],
               "cross_age": cross_age, "bb_age": bb_age,
               "bb_gap": (c[i] / mid[i] - 1) * 100, "bb_pctb": (c[i] - lo[i]) / (up[i] - lo[i]) if up[i] > lo[i] else np.nan,
               "bb_width": (up[i] - lo[i]) / mid[i] * 100, "bb_mid_slope3": (mid[i] / mid[i - 3] - 1) * 100,
               "ma20_slope5": (ma20[i] / ma20[i - 5] - 1) * 100, "ma5_ma20": (ma5[i] / ma20[i] - 1) * 100,
               "candle_bull": int(c[i] > o[i]), "body_pct": (c[i] / o[i] - 1) * 100,
               "rsi": rsi[i], "vol_ratio": v[i] / vol_ma20[i] if vol_ma20[i] > 0 else np.nan,
               "day_ret": (c[i] / dt["prev_close"] - 1) * 100, "from_open": (c[i] / day_open - 1) * 100,
               "vwap_gap": (c[i] / (cum_pv[i] / cum_v[i]) - 1) * 100 if cum_v[i] > 0 else np.nan,
               "m3_above": int(k3 >= 0 and bool(above3[k3])), "m3_age": int(age3[k3]) if k3 >= 0 else -1,
               "m3_ma5_ma20": (m3_ma5[k3] / m3_ma20[k3] - 1) * 100 if k3 >= 0 else np.nan,
               "m3_close_mid": (c3[k3] / m3_mid_fast[k3] - 1) * 100 if k3 >= 0 else np.nan,
               "m3_ma20_slope": (m3_ma20[k3] / m3_ma20[k3 - 3] - 1) * 100 if k3 >= 3 else np.nan,
               "mfe60": (np.max(h[i + 1:i + 61]) / (o[i + 1]) - 1) * 100,
               "mae60": (np.min(l[i + 1:i + 61]) / (o[i + 1]) - 1) * 100}
        rec.update(dt)
        for k in range(len(net)):
            rec[f"n{k}"] = round(float(net[k]), 1)
            rec[f"h{k}"] = int(hold[k])
        # detail for the reference combo (k = REF) is cheap to keep
        rec["inv_ref"], rec["reason_ref"], rec["add_ref"], rec["hold_ref"] = inv[REF], reason[REF], addt[REF], hold[REF]
        events.append(rec)
        last_ev = i
    return events


PRMS = grid()
REF = int(np.flatnonzero((PRMS[:, 0] == 0.8) & (PRMS[:, 2] == 1.0) & (PRMS[:, 3] == 2.0) & (PRMS[:, 4] == 1.0)
                         & (PRMS[:, 5] == 1.0))[0])


def g_sim(o, h, l, c, obs, minute, i, prms):
    return simulate_grid(o, h, l, c, obs, minute, i, prms, BUY1_KRW, ADD_KRW)


def build_day(task) -> tuple[str, int, float]:
    trade_date, out_dir, args = task
    t0 = time.time()
    out = out_dir / f"events_{trade_date}.csv.gz"
    if out.exists() and not args.force:
        return trade_date, -1, 0.0
    panel = g013.load_panel(g013.EDGE_AI_DIR / "cache" / "turnover_panel.csv")
    codes = g013.universe_for(trade_date, panel, args)
    day_dir = g013.DATA_ROOT / trade_date
    minute_idx = g013._index_dir(day_dir, "_10s.txt")
    daily_idx = g013._index_dir(day_dir, "_daily.csv")
    rows = []
    for code in codes:
        mf = minute_idx.get(code)
        if mf is None:
            continue
        try:
            rows.extend(symbol_events(code, trade_date, mf, daily_idx.get(code), PRMS))
        except Exception as exc:  # one bad file must not kill the day
            print(f"  {trade_date} {code} skipped: {exc}", flush=True)
    df = pd.DataFrame(rows)
    tmp = out.with_suffix(".tmp.gz")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)
    return trade_date, len(df), time.time() - t0


# ---------------------------------------------------------------------------
# stage 2: r011 features for the tuned rule (edge AI training set)
# ---------------------------------------------------------------------------

def features_for_day(task) -> tuple[str, int]:
    trade_date, events, combo, feat_dir = task
    config = build_strategy_config(g013.cfg)
    day_dir = g013.DATA_ROOT / trade_date
    minute_idx = g013._index_dir(day_dir, "_10s.txt")
    daily_idx = g013._index_dir(day_dir, "_daily.csv")
    out_rows = []
    for code, grp in events.groupby("code"):
        code = str(code).zfill(6)
        mf = minute_idx.get(code)
        if mf is None:
            continue
        raw = session_raw_1min(g013.load_minute_bars(mf), pd.Timestamp(trade_date))
        if raw is None:
            continue
        m1_full = fast_indicators(raw)
        m3_full = g013.completed_3min(m1_full)
        daily = g013._read_daily(daily_idx.get(code))
        dfeat = daily_features_from_frame(daily, trade_date)
        prev_close = float(grp["prev_close"].iloc[0])
        o0 = float(raw["open"].iloc[0])
        cv = (raw["close"] * raw["volume"]).cumsum()
        vv = raw["volume"].cumsum()
        for _, ev in grp.iterrows():
            now = pd.Timestamp(f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {ev['ts']}")
            pos = raw.index.searchsorted(now, side="right") - 1
            if pos < 0 or raw.index[pos] != now:
                continue  # event bar was a synthetic (no-trade) minute
            frames = g013.offline_feature_frames(raw, m1_full, m3_full, pos)
            if frames is None:
                continue
            vwap = float(cv.iloc[pos] / vv.iloc[pos]) if vv.iloc[pos] > 0 else float("nan")
            feats = compute_features(frames[0], frames[1], now, float(raw["close"].iloc[pos]), prev_close, o0, vwap,
                                     dfeat, config, trigger_info=None)
            net = float(ev[f"n{combo}"])
            inv = BUY1_KRW
            rec = {"date": trade_date, "code": code, "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
                   "price": float(raw["close"].iloc[pos]), "trigger": "USER_POLICY", "m3_ctx_pass": 1.0}
            rec.update(feats)
            rec["m3_ctx_pass"] = 1.0  # g014 promotion subset = every event here already passed the tuned rule
            rec.update({"net_krw": net, "net_ret": net / inv * 100.0, "y": int(net > 0)})
            rec.update({k: float(ev[k]) for k in ev.index if k[:1] in "nh" and k[1:].isdigit()})  # every grid combo
            out_rows.append(rec)
    df = pd.DataFrame(out_rows)
    out = feat_dir / f"events_{trade_date}.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    return trade_date, len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", help="YYYYMMDD list (default: every data folder)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-turnover", type=float, default=3e9)
    ap.add_argument("--min-price", type=float, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--features", help="stage 2: rule json from g018 (entry filter + combo index)")
    ap.add_argument("--feat-tag", default="userpolicy_v1")
    args = ap.parse_args()
    args.max_codes = 0
    dates = args.dates or sorted(p.name for p in g013.DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit())
    out_dir = OUT_ROOT / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "grid.json").write_text(json.dumps({"axes": GRID_AXES, "rows": PRMS.tolist(), "ref": REF,
                                                    "buy1": BUY1_KRW, "add": ADD_KRW}, indent=1))
    if args.features:
        sys.path.insert(0, str(HERE))
        from g018_user_policy_tune import apply_rule  # noqa: E402
        rule = json.loads(Path(args.features).read_text())
        feat_dir = g013.EDGE_AI_DIR / "dataset" / args.feat_tag
        feat_dir.mkdir(parents=True, exist_ok=True)
        (feat_dir / "params.json").write_text(json.dumps({"source": "g017", "rule": rule}, indent=1))
        tasks = []
        for d in dates:
            f = out_dir / f"events_{d}.csv.gz"
            if not f.exists():
                continue
            ev = pd.read_csv(f, dtype={"code": str})
            ev = ev[apply_rule(ev, rule)] if not ev.empty else ev
            if not ev.empty:
                tasks.append((d, ev, int(rule["combo"]), feat_dir))
        with Pool(args.workers) as pool:
            for d, n in pool.imap_unordered(features_for_day, tasks):
                print(f"features {d}: {n}", flush=True)
        return
    simulate_grid(np.ones(3), np.ones(3), np.ones(3), np.ones(3), np.ones(3, dtype=np.bool_), np.zeros(3, dtype=np.int64), 0,
                  PRMS[:1], 1.0, 1.0)
    tasks = [(d, out_dir, args) for d in dates]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for d, n, sec in pool.imap_unordered(build_day, tasks):
            print(f"{d}: events={n} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
