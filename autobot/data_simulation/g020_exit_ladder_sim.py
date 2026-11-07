# -*- coding: utf-8 -*-
"""Exit ladder + RSI-cross entry simulation (2026-10-31 사용자 요청: 손실 축소 재설계, 물타기 없음).

Exit ladder (fractions of the ENTRY quantity, prices relative to the entry fill, no adds):
  TP1    price >= entry*(1+tp1)                 -> sell 30% (limit, touch fill)
  TP2    price >= entry*(1+tp2) (surge)         -> sell 30% (limit)
  trail  armed after TP1 (arm=0) or once peak >= entry*(1+arm%); price <= peak*(1-trail) -> sell the rest
  SL1    price <= entry*(1-s1)                  -> sell 50% of the current quantity (market)
  SL2    price <= entry*(1-s2)                  -> sell the rest (market)
  EOD    first bar >= 15:15                     -> sell the rest at the close
User default: tp1 2, tp2 4, trail 0.5 (after TP1), s1 0.8, s2 1.2. Fill/cost model = g017 (fee 0.015%/side, tax 0.20%,
market-type fills 1 tick slippage, 1-min path o->l->h->c on bullish bars, o->h->l->c on bearish bars).

Entry families on the g019 universe (liquid, daily uptrend):
  ma_cross   1m MA5 crosses above MA20 (live trigger baseline)
  rsi_gc     1m RSI(14) crosses above RSI_SIGNAL(6) - stand-in for the 60/120-tick RSI cross (ticks only exist
             since 2026-10-30, so the tick version cannot be backtested)
  rsi_gc_ma  rsi_gc while 1m MA5 > MA20
  random     g019 causal random control
Feature columns are kept so "no overheat" filters can be evaluated in --report.

  build : python g020_exit_ladder_sim.py [--dates ...] [--workers 2]
  report: python g020_exit_ladder_sim.py --report
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

OUT_DIR = g013.EDGE_AI_DIR / "exit_ladder"
FAMILIES = ("ma_cross", "rsi_gc", "rsi_gc_ma", "random")
BUY_KRW = 500_000.0
MIN_GAP_BARS = 10
F_TP1, F_TP2, F_SL1 = 0.30, 0.30, 0.50
GRID_COLS = ("tp1", "tp2", "trail", "s1", "s2", "arm")
_tick, _up, _dn = g017._tick, g017._up, g017._dn


def grid() -> np.ndarray:
    rows = list(itertools.product((1.5, 2.0, 2.5), (3.0, 4.0, 5.0), (0.5, 0.8, 1.0), (0.6, 0.8, 1.0),
                                  (1.2, 1.5, 2.0), (0.0, 1.0)))
    return np.array(rows, dtype=np.float64)


PRMS = grid()
USER = int(np.flatnonzero((PRMS == np.array([2.0, 4.0, 0.5, 0.8, 1.2, 0.0])).all(axis=1))[0])


@njit(cache=True)
def _part(qty_entry, qty, frac):
    sq = np.floor(qty_entry * frac + 0.5)
    if sq < 1:
        sq = 1.0
    if sq > qty:
        sq = qty
    return sq


@njit(cache=True)
def simulate_ladder(o, h, l, c, obs, minute, i, prm, buy_krw):
    """-> (net_krw, reason, hold_bars). reason 0 no-trade 1 all sold by TP 2 trail 3 SL2/SL1 4 eod 5 end."""
    tp1, tp2, trail, s1, s2, arm = prm[0] / 100, prm[1] / 100, prm[2] / 100, prm[3] / 100, prm[4] / 100, prm[5] / 100
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
    tp_stage = 0
    sl1_done = False
    armed = False
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
                    while qty > 0 and tp_stage < 2:
                        lvl = tp1 if tp_stage == 0 else tp2
                        trig = _up(e1 * (1 + lvl))
                        if p1 < trig:
                            break
                        fill = p1 if gap else trig
                        sq = _part(q0, qty, F_TP1 if tp_stage == 0 else F_TP2)
                        cash += fill * sq * sell_k
                        qty -= sq
                        tp_stage += 1
                    if qty <= 0:
                        return cash, 1, j - i
                    if p1 > peak:
                        peak = p1
                    if (arm == 0 and tp_stage >= 1) or (arm > 0 and peak >= e1 * (1 + arm)):
                        armed = True
                elif p1 < p0:
                    while qty > 0:
                        best_px = -1.0
                        best = -1
                        if armed and p1 <= peak * (1 - trail) and peak * (1 - trail) > best_px:
                            best_px, best = peak * (1 - trail), 0
                        if not sl1_done and p1 <= e1 * (1 - s1) and e1 * (1 - s1) > best_px:
                            best_px, best = e1 * (1 - s1), 1
                        if p1 <= e1 * (1 - s2) and e1 * (1 - s2) > best_px:
                            best_px, best = e1 * (1 - s2), 2
                        if best < 0:
                            break
                        fill = (p1 - _tick(p1)) if gap else (_dn(best_px) - _tick(best_px))
                        if best == 1:
                            sq = _part(qty, qty, F_SL1)
                            sl1_done = True
                        else:
                            sq = qty
                        cash += fill * sq * sell_k
                        qty -= sq
                        if qty <= 0:
                            return cash, 2 if best == 0 else 3, j - i
                prev = p1
        if minute[j] >= g017.EOD_MINUTE:
            cash += (prev - _tick(prev)) * qty * sell_k
            return cash, 4, j - i
    cash += (prev - _tick(prev)) * qty * sell_k
    return cash, 5, n - 1 - i


@njit(cache=True)
def ladder_grid(o, h, l, c, obs, minute, i, prms, buy_krw):
    m = prms.shape[0]
    net = np.empty(m)
    reason = np.empty(m, dtype=np.int64)
    hold = np.empty(m, dtype=np.int64)
    for k in range(m):
        net[k], reason[k], hold[k] = simulate_ladder(o, h, l, c, obs, minute, i, prms[k], buy_krw)
    return net, reason, hold


def signals(s: dict, rng) -> list[tuple[str, int]]:
    c, obs, minute, ma5, ma20, rsi, sig = (s[k] for k in ("c", "obs", "minute", "ma5", "ma20", "rsi", "rsi_sig"))
    lo_m = g017.ENTRY_START.hour * 60 + g017.ENTRY_START.minute
    hi_m = g017.ENTRY_END.hour * 60 + g017.ENTRY_END.minute
    valid = [i for i in range(1, len(c) - 1) if obs[i] and lo_m <= minute[i] <= hi_m]
    out = []
    target = int(rng.integers(lo_m, hi_m + 1))
    for i in valid:
        if minute[i] >= target:
            out.append(("random", i))
            break
    last = {f: -10**9 for f in FAMILIES}
    for i in valid:
        hits = []
        if ma5[i] > ma20[i] and ma5[i - 1] <= ma20[i - 1]:
            hits.append("ma_cross")
        if rsi[i] > sig[i] and rsi[i - 1] <= sig[i - 1]:
            hits.append("rsi_gc")
            if ma5[i] > ma20[i]:
                hits.append("rsi_gc_ma")
        for f in hits:
            if i - last[f] >= MIN_GAP_BARS:
                last[f] = i
                out.append((f, i))
    return out


def symbol_events(code: str, trade_date: str, minute_file: Path, daily_file: Path | None, seed: int) -> list[dict]:
    daily = g013._read_daily(daily_file)
    dt = g017.daily_trend(daily, trade_date)
    if dt is None or not (dt["d_close_ma20"] > 0 and dt["d_ma20_slope5"] > 0):
        return []
    s = g019.prepare_session(minute_file, trade_date)
    if s is None:
        return []
    o, h, l, c, obs, minute, vwap, eod_i = (s[k] for k in ("o", "h", "l", "c", "obs", "minute", "vwap", "eod_i"))
    events = []
    for fam, i in signals(s, np.random.default_rng(seed)):
        e = i + 1
        while e < len(c) and not obs[e]:
            e += 1
        if e >= eod_i:
            continue
        net, reason, hold = ladder_grid(o, h, l, c, obs, minute, i, PRMS, BUY_KRW)
        if reason[0] == 0:
            continue
        rec = {"date": trade_date, "code": code, "fam": fam, "ts": s["times"][i].strftime("%H:%M"), "bar_i": i,
               "price": c[i], "rsi": s["rsi"][i], "rsi_sig": s["rsi_sig"][i],
               "day_ret": (c[i] / dt["prev_close"] - 1) * 100, "from_open": (c[i] / o[0] - 1) * 100,
               "vwap_gap": (c[i] / vwap[i] - 1) * 100, "g30": (c[min(e + 29, eod_i)] / o[e] - 1) * 100,
               "d_ret5": dt["d_ret5"], "d_close_ma20": dt["d_close_ma20"], "reason_user": int(reason[USER])}
        for k in range(len(net)):
            rec[f"n{k}"] = round(float(net[k]), 1)
            rec[f"h{k}"] = int(hold[k])
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


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

# "no overheat" entry filter: buckets that were worse than average in g019 on BOTH splits (report 20261031_195425
# quintiles; thresholds = quintile edges). The holdout was looked at when picking them -> treat as indicative.
COOL = {"rsi": 55.0, "day_ret": 3.4, "from_open": 3.2, "vwap_gap": 1.15, "d_ret5": 14.0}
MIN_PRICE = 7380.0


def cool_mask(df: pd.DataFrame) -> np.ndarray:
    m = df["price"].to_numpy(float) >= MIN_PRICE
    for k, thr in COOL.items():
        m &= df[k].to_numpy(float) <= thr
    return m


def _seq(df: pd.DataFrame, k: int, max_per_day: int) -> np.ndarray:
    import g018_user_policy_tune as g018
    d = df.sort_values(["date", "minute", "code"]).reset_index(drop=True)
    d = d.assign(hold_ref=d[f"h{k}"])
    sel = g018.sequence(d, np.ones(len(d), dtype=bool), max_per_day, k)
    return d[f"n{k}"].to_numpy(float)[sel]


def _fmt(st: dict) -> str:
    return (f"{st['n']:,} | {st['win']:.1%} | {st['avg']:+,.0f} | {st['avg_pct']:+.3f}% | {st['total']:+,.0f}"
            if st["n"] else "0 | - | - | - | -")


def _stats(net: np.ndarray) -> dict:
    if not len(net):
        return {"n": 0}
    return {"n": int(len(net)), "win": float((net > 0).mean()), "avg": float(net.mean()),
            "avg_pct": float(net.mean() / BUY_KRW * 100), "total": float(net.sum())}


META_COLS = ("date", "code", "fam", "ts", "bar_i", "price", "rsi", "day_ret", "from_open", "vwap_gap", "d_ret5",
             "d_close_ma20", "g30", "reason_user")


def _read(f: Path, ks) -> pd.DataFrame | None:
    """One day file with the meta columns + n/h columns of combos ks only (the full grid is ~1000 columns, ~550k
    rows over 82 days: loading it at once does not fit the Pi's memory)."""
    cols = list(META_COLS) + [f"n{k}" for k in ks] + [f"h{k}" for k in ks]
    try:
        d = pd.read_csv(f, usecols=cols, dtype={"code": str, "date": str, **{f"n{k}": np.float32 for k in ks}})
    except pd.errors.EmptyDataError:   # a day without any event is written header-less
        return None
    return d if not d.empty else None


def report(args) -> None:
    files = sorted(OUT_DIR.glob("events_*.csv.gz"))
    file_dates = sorted(f.name[7:15] for f in files)
    if not 0 < args.train_frac < 1 or len(file_dates) < 2:
        raise SystemExit(f"need 0 < --train-frac < 1 and >= 2 event days (got {args.train_frac}, {len(file_dates)} days)")
    cut0 = file_dates[min(max(int(len(file_dates) * args.train_frac), 1), len(file_dates) - 1)]
    # pass 1 (streamed): per family x filter, sum of every combo's net over TRAIN events -> best combo
    all_k = range(len(PRMS))
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    for f in files:
        if f.name[7:15] >= cut0:
            continue
        d = _read(f, all_k)
        if d is None:
            continue
        n = d[[f"n{k}" for k in all_k]].to_numpy(np.float64)
        cool = cool_mask(d)
        for fam in FAMILIES:
            fm = (d["fam"] == fam).to_numpy()
            for flt, m in (("all", fm), ("cool", fm & cool)):
                if m.any():
                    sums[(fam, flt)] = sums.get((fam, flt), 0.0) + n[m].sum(axis=0)
                    counts[(fam, flt)] = counts.get((fam, flt), 0) + int(m.sum())
        del d, n
    best_by = {key: int(np.argmax(v)) for key, v in sums.items()}
    # pass 2: only the user default + selected combos
    ks = sorted({USER, *best_by.values()})
    parts = [p for p in (_read(f, ks) for f in files) if p is not None]
    if not parts:
        raise SystemExit(f"no events in {OUT_DIR}")
    df = pd.concat(parts, ignore_index=True)
    df = df.assign(minute=df["ts"].str.slice(0, 2).astype(int) * 60 + df["ts"].str.slice(3, 5).astype(int))
    dates = sorted(df["date"].unique())
    cut = cut0   # same split as pass 1 (the best combos were chosen on dates < cut0)
    pstr = lambda k: "/".join(f"{x:g}" for x in PRMS[k])  # noqa: E731
    lines = [f"# Exit ladder sim {time.strftime('%Y%m%d_%H%M%S')}", "",
             f"- events {len(df):,}, days {len(dates)} ({dates[0]}~{dates[-1]}), train < {cut}, holdout >= {cut}",
             f"- per-trade budget {BUY_KRW:,.0f} KRW, no adds. combo = tp1/tp2/trail/s1/s2/arm (arm 0 = trail after TP1)",
             f"- user default = k{USER} ({pstr(USER)}); best = highest TRAIN per-event mean; seq = live-like "
             f"(1 position per code, max {args.max_per_day}/day)",
             "- filter `cool` (no-overheat) thresholds were read off g019 quintiles of BOTH splits -> its holdout rows "
             "are EXPLORATORY, not out-of-sample", "",
             "| family | filter | combo | split | events | win | avg KRW | avg % | seq n | seq win | seq avg | seq total |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for fam in FAMILIES:
        for flt in ("all", "cool"):
            f = df[df["fam"] == fam]
            if flt == "cool":
                f = f[cool_mask(f)]
            tr, te = f[f["date"] < cut], f[f["date"] >= cut]
            if tr.empty or (fam, flt) not in best_by:
                continue
            best = best_by[(fam, flt)]
            for label, k in (("user", USER), ("best", best)):
                for split, s in (("train", tr), ("hold", te)):
                    ev = _stats(s[f"n{k}"].to_numpy(float))
                    sq = _stats(_seq(s, k, args.max_per_day)) if len(s) else {"n": 0}
                    summary[f"{fam}|{flt}|{label}|{split}"] = {"combo": k, "params": PRMS[k].tolist(), "events": ev, "seq": sq}
                    seq_txt = (f"{sq['n']:,} | {sq['win']:.1%} | {sq['avg']:+,.0f} | {sq['total']:+,.0f}" if sq["n"] else "0 | - | - | -")
                    ev_txt = (f"{ev['n']:,} | {ev['win']:.1%} | {ev['avg']:+,.0f} | {ev['avg_pct']:+.3f}%" if ev["n"] else "0 | - | - | -")
                    lines.append(f"| {fam} | {flt}{'*' if flt == 'cool' else ''} | {label} k{k} {pstr(k)} | {split} | {ev_txt} | {seq_txt} |")
    # exit-reason mix of the user default on ma_cross (all)
    f = df[df["fam"] == "ma_cross"]
    mix = f.groupby("reason_user")[f"n{USER}"].agg(["size", "mean"]).round(0)
    lines += ["", "## user default exit mix (ma_cross, all days)", "", "```", mix.to_string(), "```"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"report_{stamp}.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / f"report_{stamp}.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    print(f"saved {OUT_DIR / f'report_{stamp}.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", help="YYYYMMDD list (default: every data folder)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-turnover", type=float, default=3e9)
    ap.add_argument("--min-price", type=float, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--max-per-day", type=int, default=10)
    args = ap.parse_args()
    args.max_codes = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args)
        return
    (OUT_DIR / "grid.json").write_text(json.dumps({"cols": GRID_COLS, "rows": PRMS.tolist(), "user": USER}, indent=1))
    dates = args.dates or sorted(p.name for p in g013.DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit())
    t0 = time.time()
    with Pool(args.workers) as pool:
        for d, n, sec in pool.imap_unordered(build_day, [(d, args) for d in dates]):
            print(f"{d}: events={n} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
