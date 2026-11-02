# -*- coding: utf-8 -*-
"""Profit-lock exit simulation (2026-11-02 사용자 요청, 091580 상신이디피 10:24 고점 +1.12% -> -1.12% 손절 사례).

Question: once a position has been up >= arm%, sell a fraction when price gives back g% from the peak (even below the
TP), optionally moving the rest's stop to breakeven. Does that beat the current exits on the g019/g020 entry events?

Base exits (simplified live r006, no adds):
  TP1    price >= entry*(1+3.0%)                    -> sell 40% (STAGED_TP1_PCT / _RATIO)
  trail  before TP1: peak >= entry*(1+1.2%) and price <= peak*(1-1.2%) -> rest (TRAILING_STOP_FROM_PEAK, pnl>0)
         after TP1:  price <= peak*(1-1.0%)          -> rest (TP_EXTENSION_TRAIL)
  stop   price <= entry*(1-s%)                       -> rest (ATR stop proxy; s = 1.2 by default)
  EOD    first bar >= 15:15                          -> rest at the close
Lock (added on top, combo k > 0):
  lock   peak >= entry*(1+arm%) and price <= peak*(1-g%) -> sell frac of the ENTRY qty (market), once
  be     after the lock, rest stops at entry (breakeven) if be=1
Fill/cost model = g017/g020 (fee 0.015%/side, tax 0.20%, market fills 1 tick slippage, 1-min path o->l->h->c on bullish
bars, o->h->l->c on bearish bars). Entry events = g020 signals (ma_cross = live 1-min MA5>MA20 proxy, random control).

  build : python g023_profit_lock_sim.py [--dates ...] [--workers 1]
  report: python g023_profit_lock_sim.py --report
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

import g013_edge_ai_dataset as g013  # noqa: E402
import g017_user_policy_sim as g017  # noqa: E402
import g019_entry_edge_scan as g019  # noqa: E402
import g020_exit_ladder_sim as g020  # noqa: E402

OUT_DIR = g013.EDGE_AI_DIR / "profit_lock"
FAMILIES = ("ma_cross", "random")
BUY_KRW = 500_000.0
TP1, F_TP1, TRAIL_PRE, TRAIL_POST = 0.030, 0.40, 0.012, 0.010
GRID_COLS = ("arm", "g", "frac", "be", "stop")
_tick, _up, _dn = g017._tick, g017._up, g017._dn


def grid() -> np.ndarray:
    rows = []
    for stop in (1.2, 1.7):
        rows.append((0.0, 0.0, 0.0, 0.0, stop))  # base (no lock)
        rows += [(a, g, f, be, stop) for a, g, f, be in
                 itertools.product((0.6, 0.8, 1.0, 1.5), (0.4, 0.5, 0.7), (0.5, 1.0), (0.0, 1.0)) if not (f == 1.0 and be)]
    return np.array(rows, dtype=np.float64)


PRMS = grid()


@njit(cache=True)
def simulate(o, h, l, c, obs, minute, i, prm, buy_krw):
    """-> (net_krw, reason, hold_bars). reason 0 no-trade 1 lock(all) 2 trail 3 stop 4 eod 5 end 6 be."""
    arm, g, frac, be, stop = prm[0] / 100, prm[1] / 100, prm[2], prm[3], prm[4] / 100
    n = len(c)
    e = i + 1
    while e < n and not obs[e]:
        e += 1
    if e >= n or minute[e] >= g017.EOD_MINUTE:
        return 0.0, 0, 0
    e1 = _up(o[e]) + _tick(o[e])
    q0 = np.floor(buy_krw / e1)
    if q0 < 1:
        return 0.0, 0, 0
    qty = q0
    cash = -e1 * q0 * (1 + g017.FEE)
    sell_k = 1 - g017.FEE - g017.TAX
    peak = e1
    tp1_done = False
    locked = arm <= 0
    prev = e1
    pts = np.empty(4)
    for j in range(e, n):
        if obs[j]:
            if c[j] >= o[j]:
                pts[0], pts[1], pts[2], pts[3] = o[j], l[j], h[j], c[j]
            else:
                pts[0], pts[1], pts[2], pts[3] = o[j], h[j], l[j], c[j]
            start_k = 0 if j > e else 1
            for k in range(start_k, 4):
                p0 = prev
                p1 = pts[k]
                gap = k == 0
                if p1 > p0:
                    if not tp1_done:
                        trig = _up(e1 * (1 + TP1))
                        if p1 >= trig:
                            fill = p1 if gap else trig
                            sq = g020._part(q0, qty, F_TP1)
                            cash += fill * sq * sell_k
                            qty -= sq
                            tp1_done = True
                            if qty <= 0:
                                return cash, 2, j - i
                    if p1 > peak:
                        peak = p1
                elif p1 < p0:
                    while qty > 0:
                        best_px = -1.0
                        best = -1
                        if not locked and peak >= e1 * (1 + arm) and p1 <= peak * (1 - g):
                            best_px, best = peak * (1 - g), 1
                        if locked and arm > 0 and be > 0 and p1 <= e1 and e1 > best_px:
                            best_px, best = e1, 6
                        if tp1_done:
                            lvl = peak * (1 - TRAIL_POST)
                            if p1 <= lvl and lvl > best_px:
                                best_px, best = lvl, 2
                        else:
                            lvl = peak * (1 - TRAIL_PRE)
                            if lvl > e1 and p1 <= lvl and lvl > best_px:
                                best_px, best = lvl, 2
                        if p1 <= e1 * (1 - stop) and e1 * (1 - stop) > best_px:
                            best_px, best = e1 * (1 - stop), 3
                        if best < 0:
                            break
                        fill = (p1 - _tick(p1)) if gap else (_dn(best_px) - _tick(best_px))
                        if best == 1:
                            sq = g020._part(q0, qty, frac)
                            locked = True
                        else:
                            sq = qty
                        cash += fill * sq * sell_k
                        qty -= sq
                        if qty <= 0:
                            return cash, best, j - i
                prev = p1
        if minute[j] >= g017.EOD_MINUTE:
            cash += (prev - _tick(prev)) * qty * sell_k
            return cash, 4, j - i
    cash += (prev - _tick(prev)) * qty * sell_k
    return cash, 5, n - 1 - i


@njit(cache=True)
def sim_grid(o, h, l, c, obs, minute, i, prms, buy_krw):
    m = prms.shape[0]
    net = np.empty(m)
    reason = np.empty(m, dtype=np.int64)
    for k in range(m):
        net[k], reason[k], _ = simulate(o, h, l, c, obs, minute, i, prms[k], buy_krw)
    return net, reason


def symbol_events(code: str, trade_date: str, minute_file: Path, daily_file: Path | None, seed: int) -> list[dict]:
    daily = g013._read_daily(daily_file)
    dt = g017.daily_trend(daily, trade_date)
    if dt is None or not (dt["d_close_ma20"] > 0 and dt["d_ma20_slope5"] > 0):
        return []
    s = g019.prepare_session(minute_file, trade_date)
    if s is None:
        return []
    o, h, l, c, obs, minute, eod_i = (s[k] for k in ("o", "h", "l", "c", "obs", "minute", "eod_i"))
    events = []
    for fam, i in g020.signals(s, np.random.default_rng(seed)):
        if fam not in FAMILIES:
            continue
        e = i + 1
        while e < len(c) and not obs[e]:
            e += 1
        if e >= eod_i:
            continue
        net, reason = sim_grid(o, h, l, c, obs, minute, i, PRMS, BUY_KRW)
        if reason[0] == 0:
            continue
        hi = float(np.max(h[e:eod_i + 1]))
        rec = {"date": trade_date, "code": code, "fam": fam, "ts": s["times"][i].strftime("%H:%M"),
               "mfe": round((hi / o[e] - 1) * 100, 3)}
        for k in range(len(net)):
            rec[f"n{k}"] = round(float(net[k]), 1)
            rec[f"r{k}"] = int(reason[k])
        events.append(rec)
    return events


def build_day(task) -> tuple[str, int, float]:
    trade_date, args = task
    t0 = time.time()
    out = OUT_DIR / f"events_{trade_date}.csv.gz"
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
            rows.extend(symbol_events(code, trade_date, mf, daily_idx.get(code),
                                      seed=int(trade_date) * 1000 + int(code[-3:], 36) % 1000))
        except Exception as exc:  # one bad file must not kill the day
            print(f"  {trade_date} {code} skipped: {exc}", flush=True)
    df = pd.DataFrame(rows)
    tmp = out.with_suffix(".tmp.gz")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)
    return trade_date, len(df), time.time() - t0


def _stats(net: np.ndarray) -> dict:
    if len(net) == 0:
        return {"n": 0, "mean": float("nan"), "pct": float("nan"), "win": float("nan")}
    return {"n": int(len(net)), "mean": float(net.mean()), "pct": float(net.mean() / BUY_KRW * 100),
            "win": float((net > 0).mean() * 100)}


def report(args) -> Path:
    files = sorted(OUT_DIR.glob("events_*.csv.gz"))
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files if f.stat().st_size > 30], ignore_index=True)
    dates = sorted(df["date"].astype(str).unique())
    cut = dates[int(len(dates) * args.train_frac)] if len(dates) > 1 else dates[-1]
    df["split"] = np.where(df["date"].astype(str) < cut, "train", "hold")
    lines = [f"# g023 profit-lock report {time.strftime('%Y%m%d_%H%M%S')}", "",
             f"days={len(dates)} ({dates[0]}~{dates[-1]}), holdout from {cut}, events={len(df)}", "",
             "net = KRW per 500k trade after fee/tax/slippage; pct = net/500k. base = k with arm 0.", ""]
    for fam in FAMILIES:
        d = df[df["fam"] == fam]
        if d.empty:
            continue
        lines += [f"## {fam} (n={len(d)})", "",
                  "| k | arm | g | frac | be | stop | train pct | train win | hold pct | hold win | Δhold vs base |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for k, prm in enumerate(PRMS):
            tr, ho = _stats(d.loc[d.split == "train", f"n{k}"].to_numpy()), _stats(d.loc[d.split == "hold", f"n{k}"].to_numpy())
            rows.append((k, prm, tr, ho))
        base_hold = {prm[4]: ho["pct"] for k, prm, tr, ho in rows if prm[0] == 0}
        rows.sort(key=lambda r: -(r[2]["pct"] if not np.isnan(r[2]["pct"]) else -9e9))
        for k, prm, tr, ho in rows[:args.top]:
            lines.append(f"| {k} | {prm[0]:.1f} | {prm[1]:.1f} | {prm[2]:.1f} | {int(prm[3])} | {prm[4]:.1f} | "
                         f"{tr['pct']:+.3f}% | {tr['win']:.1f}% | {ho['pct']:+.3f}% | {ho['win']:.1f}% | "
                         f"{ho['pct'] - base_hold[prm[4]]:+.3f}%p |")
        for k, prm, tr, ho in rows:
            if prm[0] == 0:
                lines.append(f"| base k{k} | - | - | - | - | {prm[4]:.1f} | {tr['pct']:+.3f}% | {tr['win']:.1f}% | "
                             f"{ho['pct']:+.3f}% | {ho['win']:.1f}% | 0 |")
        lines.append("")
    out = OUT_DIR / f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", help="YYYYMMDD list (default: every data folder)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-turnover", type=float, default=3e9)
    ap.add_argument("--min-price", type=float, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    args.max_codes = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args)
        return
    (OUT_DIR / "grid.json").write_text(json.dumps({"cols": GRID_COLS, "rows": PRMS.tolist()}, indent=1))
    dates = args.dates or sorted(p.name for p in g013.DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit())
    t0 = time.time()
    if args.workers <= 1:
        it = map(build_day, [(d, args) for d in dates])
        for d, n, sec in it:
            print(f"{d}: events={n} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
        return
    with Pool(args.workers) as pool:
        for d, n, sec in pool.imap_unordered(build_day, [(d, args) for d in dates]):
            print(f"{d}: events={n} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
