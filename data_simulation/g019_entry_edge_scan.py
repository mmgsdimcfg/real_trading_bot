# -*- coding: utf-8 -*-
"""Entry-family edge scan (2026-10-31, profit-first redesign: no averaging down).

g017/g018 showed the 1-min MA5>MA20 golden cross has ~0 pre-cost edge, so no exit rule can make it profitable.
This scan measures the PRE-COST forward return of several entry families on the same universe (liquid common
stocks whose daily chart is rising, g017 daily_trend) and the after-cost result of a no-average-down exit grid:

  ma_cross   1m MA5 crosses above MA20 (current live trigger, baseline)
  orb15/30   first close above the 09:00~09:15 / 09:00~09:30 opening-range high
  pdh        first close above the previous day's high (day opened below it)
  vwap_rcl   close back above VWAP after >= 10 bars below it, day still up
  pullback   after a >= +2% run from the open, a dip of 1~4% from the day high that touches VWAP/MA20 and
             closes bullish back above MA5 and VWAP
  hod_brk    new high of day after a tight (<1.2%) 15-bar base, >= 09:30
  bb_sqz     close above the upper BB right after a BB-width squeeze (60-bar low)
  dip        close below the lower BB with RSI < 30 (contrarian, daily uptrend)
  random     one random observed bar per code-day (drift control)

Entry = next traded bar open (+1 tick for the net grid). Gross forward returns g5/g15/g30/g60/geod (percent,
from the raw next open, no costs) measure the signal; n<k> = net KRW of grid combo k (g017.simulate with
avgdn = 0, BUY1 300k, pyramid ADD 200k) measures the tradeable result.

  build : python g019_entry_edge_scan.py [--dates ...] [--workers 2]
  report: python g019_entry_edge_scan.py --report [--train-frac 0.7]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g013_edge_ai_dataset as g013  # noqa: E402
import g017_user_policy_sim as g017  # noqa: E402
from r011_edge_ai import fast_indicators, session_raw_1min  # noqa: E402

OUT_DIR = g013.EDGE_AI_DIR / "entry_scan"
FAMILIES = ("ma_cross", "orb15", "orb30", "pdh", "vwap_rcl", "pullback", "hod_brk", "bb_sqz", "dip", "random")
HORIZONS = (5, 15, 30, 60)
MIN_GAP_BARS = 10   # per family and code


def grid() -> np.ndarray:
    """(arm, t1, t2, stop, pyr, avgdn, tp1, tp2, tp3) percent, avgdn always 0 (no averaging down)."""
    rows = []
    for arm, (t1, t2), stop, pyr, tps in itertools.product(
            (0.5, 1.0), ((0.5, 0.8), (1.0, 1.5)), (0.7, 1.0, 1.5, 2.0, 3.0), (0.0, 1.0), ((2.0, 3.0, 4.0), (3.0, 5.0, 8.0))):
        rows.append((arm, t1, t2, stop, pyr, 0.0) + tps)
    return np.array(rows, dtype=np.float64)


PRMS = grid()
GRID_COLS = ("arm", "t1", "t2", "stop", "pyr", "avgdn", "tp1", "tp2", "tp3")


def _prev_high(daily: pd.DataFrame | None, trade_date: str) -> float:
    if daily is None or daily.empty:
        return float("nan")
    d = daily[daily["date"].astype(str).str.slice(0, 8) < trade_date]
    return float(pd.to_numeric(d["high"], errors="coerce").iloc[-1]) if len(d) else float("nan")


def signals(o, h, l, c, v, obs, minute, ma5, ma20, mid, up, lo, rsi, vwap, prev_high, rng) -> list[tuple[str, int]]:
    """Causal (bar-close) signals -> [(family, bar index)]. Index 0 is the first session bar."""
    n = len(c)
    out: list[tuple[str, int]] = []
    last = {f: -10**9 for f in FAMILIES}
    fired_once = set()
    day_open = o[0]
    hod = np.maximum.accumulate(h)
    hod_idx = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        hod_idx[i] = i if h[i] >= hod[i - 1] else hod_idx[i - 1]
    orh = {}
    for fam, end in (("orb15", 9 * 60 + 15), ("orb30", 9 * 60 + 30)):
        m = minute <= end
        orh[fam] = (h[m].max() if m.any() else np.nan, end)
    width = (up - lo) / mid
    lo_start, hi_end = g017.ENTRY_START.hour * 60 + g017.ENTRY_START.minute, g017.ENTRY_END.hour * 60 + g017.ENTRY_END.minute
    # per-family warmups are applied below (NaN indicators compare False); ORB must see 09:16 onwards
    valid = [i for i in range(1, n - 1) if obs[i] and lo_start <= minute[i] <= hi_end]
    # random control: target minute drawn before the session, fired at the first eligible bar at/after it (causal)
    target = int(rng.integers(lo_start, hi_end + 1))
    for i in valid:
        if minute[i] >= target:
            out.append(("random", i))
            break

    def emit(fam, i, once=False):
        if once and fam in fired_once:
            return
        if i - last[fam] < MIN_GAP_BARS:
            return
        last[fam] = i
        fired_once.add(fam)
        out.append((fam, i))

    for i in valid:
        t = minute[i]
        if ma5[i] > ma20[i] and ma5[i - 1] <= ma20[i - 1]:
            emit("ma_cross", i)
        for fam, (rh, end) in orh.items():
            lim = 11 * 60 if fam == "orb15" else 12 * 60
            if end < t <= lim and c[i] > rh and c[i - 1] <= rh:
                emit(fam, i, once=True)
        if prev_high == prev_high and o[0] < prev_high and c[i] > prev_high and c[i - 1] <= prev_high:
            emit("pdh", i, once=True)
        if i >= 10 and c[i] > vwap[i] and np.all(c[i - 10:i] < vwap[i - 10:i]) and c[i] > day_open:
            emit("vwap_rcl", i)
        dd = (c[i] / hod[i - 1] - 1) * 100
        if (hod[i - 1] >= day_open * 1.02 and i - hod_idx[i - 1] >= 5 and -4.0 <= dd <= -1.0
                and (l[i] <= vwap[i] * 1.003 or l[i] <= ma20[i] * 1.002)
                and c[i] > o[i] and c[i] > ma5[i] and c[i] > vwap[i]):
            emit("pullback", i)
        if i >= 15 and t >= 9 * 60 + 30 and c[i] > hod[i - 1]:
            base_hi, base_lo = h[i - 15:i].max(), l[i - 15:i].min()
            if (base_hi / base_lo - 1) * 100 < 1.2:
                emit("hod_brk", i)
        if i >= 61 and c[i] > up[i] and width[i - 1] <= np.nanmin(width[i - 61:i - 1]) * 1.1:
            emit("bb_sqz", i)
        if c[i] < lo[i] and rsi[i] < 30:
            emit("dip", i)
    return out


def prepare_session(minute_file: Path, trade_date: str) -> dict | None:
    """Full 1-min grid of one session (missing minutes = flat no-trade bars, obs False) + indicator arrays."""
    raw = session_raw_1min(g013.load_minute_bars(minute_file), pd.Timestamp(trade_date))
    if raw is None or len(raw) < 40:
        return None
    idx = pd.date_range(raw.index[0], raw.index[-1], freq="1min")
    raw = raw.reindex(idx)
    obs = raw["close"].notna().to_numpy()
    raw["close"] = raw["close"].ffill()
    for k in ("open", "high", "low"):
        raw[k] = raw[k].fillna(raw["close"])
    raw["volume"] = raw["volume"].fillna(0.0)
    m1 = fast_indicators(raw)
    o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = m1["volume"].to_numpy(float)
    ma5, ma20 = g017._sma(c, 5), g017._sma(c, 20)
    mid, up, lo = (m1[k].to_numpy(float) for k in ("BB_MIDDLE", "BB_UPPER", "BB_LOWER"))
    rsi = m1["RSI"].to_numpy(float) if "RSI" in m1.columns else np.full(len(c), np.nan)
    times = m1.index
    minute = (times.hour * 60 + times.minute).to_numpy(np.int64)
    cum_v = np.cumsum(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cum_v > 0, np.cumsum(c * v) / cum_v, c)
    eod = np.flatnonzero(minute >= g017.EOD_MINUTE)
    eod_i = int(eod[0]) if len(eod) else len(c) - 1
    rsi_sig = m1["RSI_SIGNAL"].to_numpy(float) if "RSI_SIGNAL" in m1.columns else np.full(len(c), np.nan)
    return {"o": o, "h": h, "l": l, "c": c, "v": v, "obs": obs, "minute": minute, "times": times, "ma5": ma5,
            "ma20": ma20, "mid": mid, "up": up, "lo": lo, "rsi": rsi, "rsi_sig": rsi_sig, "vwap": vwap, "eod_i": eod_i}


def symbol_events(code: str, trade_date: str, minute_file: Path, daily_file: Path | None, seed: int) -> list[dict]:
    daily = g013._read_daily(daily_file)
    dt = g017.daily_trend(daily, trade_date)
    if dt is None or not (dt["d_close_ma20"] > 0 and dt["d_ma20_slope5"] > 0):
        return []
    s = prepare_session(minute_file, trade_date)
    if s is None:
        return []
    o, h, l, c, v, obs, minute, times = (s[k] for k in ("o", "h", "l", "c", "v", "obs", "minute", "times"))
    ma5, ma20, mid, up, lo, rsi, vwap, eod_i = (s[k] for k in ("ma5", "ma20", "mid", "up", "lo", "rsi", "vwap", "eod_i"))
    rng = np.random.default_rng(seed)
    events = []
    for fam, i in signals(o, h, l, c, v, obs, minute, ma5, ma20, mid, up, lo, rsi, vwap,
                          _prev_high(daily, trade_date), rng):
        e = i + 1
        while e < len(c) and not obs[e]:
            e += 1
        if e >= eod_i:
            continue
        net, inv, reason, addt, hold = g017.simulate_grid(o, h, l, c, obs, minute, i, PRMS, g017.BUY1_KRW, g017.ADD_KRW)
        if reason[0] == 0:
            continue
        px = o[e]
        x_end = min(e + 60, eod_i + 1)   # excursions stop at the EOD liquidation bar like g60 / the simulator
        rec = {"date": trade_date, "code": code, "fam": fam, "ts": times[i].strftime("%H:%M"), "bar_i": i,
               "price": c[i], "day_ret": (c[i] / dt["prev_close"] - 1) * 100, "from_open": (c[i] / o[0] - 1) * 100,
               "vwap_gap": (c[i] / vwap[i] - 1) * 100, "rsi": rsi[i],
               "vol_ratio": v[i] / np.nanmean(v[max(0, i - 20):i]) if i > 0 and np.nanmean(v[max(0, i - 20):i]) > 0 else np.nan,
               "geod": (c[eod_i] / px - 1) * 100,
               "mfe60": (h[e:x_end].max() / px - 1) * 100, "mae60": (l[e:x_end].min() / px - 1) * 100}
        for hz in HORIZONS:
            rec[f"g{hz}"] = (c[min(e + hz - 1, eod_i)] / px - 1) * 100
        rec.update(dt)
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
            rows.extend(symbol_events(code, trade_date, mf, daily_idx.get(code), seed=int(trade_date) * 1000 + int(code[-3:], 36) % 1000))
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

def _seq_net(df: pd.DataFrame, k: int, max_per_day: int) -> np.ndarray:
    """Live-like sequencing (one open position per code, max entries per day) via g018."""
    import g018_user_policy_tune as g018
    d = df.sort_values(["date", "minute", "code"]).reset_index(drop=True)
    d = d.assign(hold_ref=d[f"h{k}"])
    sel = g018.sequence(d, np.ones(len(d), dtype=bool), max_per_day, k)
    return d[f"n{k}"].to_numpy(float)[sel]


def report(args) -> None:
    parts = []
    for f in sorted(OUT_DIR.glob("events_*.csv.gz")):
        try:
            parts.append(pd.read_csv(f, dtype={"code": str, "date": str}))
        except pd.errors.EmptyDataError:   # a day without any event is written header-less
            continue
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit(f"no events in {OUT_DIR}")
    df = pd.concat(parts, ignore_index=True)
    df = df.assign(minute=df["ts"].str.slice(0, 2).astype(int) * 60 + df["ts"].str.slice(3, 5).astype(int))
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * args.train_frac)]
    ncols = [f"n{k}" for k in range(len(PRMS))]
    lines = [f"# Entry edge scan {time.strftime('%Y%m%d_%H%M%S')}", "",
             f"- events {len(df):,}, days {len(dates)} ({dates[0]}~{dates[-1]}), train < {cut}, holdout >= {cut}",
             f"- gross = pre-cost % from next open; round-trip cost ~0.23% fee+tax + slippage. net = KRW per 300k entry,"
             f" best combo picked on TRAIN (mean net, sequenced, max {args.max_per_day}/day), reported on HOLDOUT", "",
             "| fam | split | n | g5 | g15 | g30 | g60 | geod | win60 | best combo (train) | seq n | win | avg net KRW | total |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for fam in FAMILIES:
        f = df[df["fam"] == fam]
        if f.empty:
            continue
        tr, te = f[f["date"] < cut], f[f["date"] >= cut]
        best_k, best_avg = 0, -np.inf   # train objective = sequenced mean net (what the report shows)
        for k in range(len(ncols) if len(tr) else 0):
            seq = _seq_net(tr, k, args.max_per_day)
            if len(seq) and seq.mean() > best_avg:
                best_k, best_avg = k, float(seq.mean())
        p = dict(zip(GRID_COLS, PRMS[best_k].tolist()))
        for name, s in (("train", tr), ("hold", te)):
            if s.empty:
                continue
            seq = _seq_net(s, best_k, args.max_per_day)
            g = {c: s[c].mean() for c in ("g5", "g15", "g30", "g60", "geod")}
            row = {"n": len(s), **g, "win60": float((s["g60"] > 0).mean()), "combo": best_k, "params": p,
                   "seq_n": len(seq), "seq_win": float((seq > 0).mean()) if len(seq) else 0.0,
                   "seq_avg": float(seq.mean()) if len(seq) else 0.0, "seq_total": float(seq.sum())}
            summary.setdefault(fam, {})[name] = row
            pstr = f"k{best_k} stop{p['stop']} arm{p['arm']} t{p['t1']}/{p['t2']} pyr{p['pyr']} tp{p['tp1']:.0f}/{p['tp2']:.0f}/{p['tp3']:.0f}"
            lines.append(f"| {fam} | {name} | {len(s):,} | {g['g5']:+.3f} | {g['g15']:+.3f} | {g['g30']:+.3f} | {g['g60']:+.3f} "
                         f"| {g['geod']:+.3f} | {row['win60']:.1%} | {pstr} | {len(seq):,} | {row['seq_win']:.1%} "
                         f"| {row['seq_avg']:+,.0f} | {row['seq_total']:+,.0f} |")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"report_{stamp}.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / f"report_{stamp}.json").write_text(json.dumps(summary, indent=1, default=float))
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
    (OUT_DIR / "grid.json").write_text(json.dumps({"cols": GRID_COLS, "rows": PRMS.tolist()}, indent=1))
    dates = args.dates or sorted(p.name for p in g013.DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit())
    t0 = time.time()
    with Pool(args.workers) as pool:
        for d, n, sec in pool.imap_unordered(build_day, [(d, args) for d in dates]):
            print(f"{d}: events={n} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
