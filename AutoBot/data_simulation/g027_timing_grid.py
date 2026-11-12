# -*- coding: utf-8 -*-
"""Exit grid + dip/recovery statistics on the g026 events (2026-11-11 사용자 요청: 손절폭 / 익절 구조 최적화).

Exit policy (one position, no adds, all prices relative to the entry fill; 1-min path o->l->h->c on bullish bars,
o->h->l->c on bearish bars; a level that a bar gaps through fills at the open):
  stop    price <= entry*(1-S)      S = stop% (fixed), or min(stop%, atr_k * ATR%(1m, trigger bar)) when atr_k > 0;
                                    not before `hold` minutes after the entry bar (live HARD_STOP_MIN_HOLD_SECONDS)
  lock    once peak >= entry*(1+arm): price <= peak*(1-G) -> sell all (market)       (live _007 profit lock)
          G by mode:
            0 fixed     G = give
            1 trend     G = give2 while the last completed bar has MA5 > MA20 and close > MA20, else give
            2 ratio     G = max(give, p1 * peak_pnl)            (giveback grows with the profit already made)
            3 step      G = give2 once peak >= entry*(1+p1%), else give
  tp      price >= entry*(1+tp)     -> sell all (limit, no slippage); 0 = off
  eod     first bar >= 15:15, or `cap` minutes held (0 = off) -> sell all at that bar's close (after its own stop/lock/tp)
Costs: fee 0.015% per side + sell tax 0.20%, market fills (entry, stop, lock, eod) 1 tick slippage.
Sequencing: per code and day a new entry only after the previous position is closed (like the live bot).

  python g027_timing_grid.py --report dip        MAE / recovery table (stop-width question)
  python g027_timing_grid.py --report grid       exit grid, train (first 70% of days) vs holdout
  python g027_timing_grid.py --report entry      feature bucket scan at the trigger bar
  python g027_timing_grid.py --report cost       net by tick size / price (needs g026 --stage panel)
  python g027_timing_grid.py --report model      walk-forward LightGBM on the panel
Outputs: <runtime>/edge_ai/timing_study/report_*.md|csv. Nothing here is imported by r003.

Update log
- [2026-11-11] type=feat - first version
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g026_timing_study as g026  # noqa: E402
from g017_user_policy_sim import _dn, _tick, _up  # noqa: E402

FEE = 0.00015
TAX = 0.0020
EOD_MIN = g026.EOD_MIN
PRM_COLS = ("stop", "atr_k", "hold", "arm", "give", "mode", "give2", "p1", "tp", "cap")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_events(real: bool = False):
    """-> (events frame sorted by date/code/bar, flat path array, start offsets, lengths)."""
    files = [g026.OUT_DIR / "events" / "real.pkl"] if real else sorted((g026.OUT_DIR / "events").glob("events_*.pkl"))
    frames, paths = [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        ev = d["events"]
        if ev.empty:
            continue
        ev = ev.copy()
        ev["path"] = ev["path"] + len(paths)
        frames.append(ev)
        paths.extend(d["paths"])
    ev = pd.concat(frames, ignore_index=True)
    # family before bar: sim_grid sequencing needs every date/code/family group contiguous (Codex review 2026-11-11)
    ev = ev.sort_values(["date", "code", "family", "bar"], kind="stable").reset_index(drop=True)
    order = ev["path"].to_numpy()
    lens = np.array([len(paths[k]) for k in order], dtype=np.int64)
    starts = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    flat = np.vstack([paths[k] for k in order])
    return ev, flat, starts, lens


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------
@njit(cache=True)
def _sim_one(P, s, n, entry, atr_pct, prm, flip=False):
    """One position on path rows P[s:s+n]. Returns (net %, bars held, peak %, exit kind 1 stop/2 lock/3 tp/4 eod or time cap).
    flip=True reverses the assumed intrabar order (bullish o->h->l->c, bearish o->l->h->c) - sensitivity check."""
    stop, atr_k, hold, arm, give, mode, give2, p1, tp = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8]
    cap = prm[9]
    S = stop
    if atr_k > 0.0 and atr_pct == atr_pct:
        a = atr_k * atr_pct
        if a < S:
            S = a
        if S < 0.3:
            S = 0.3
    stop_px = entry * (1.0 - S / 100.0)
    arm_px = entry * (1.0 + arm / 100.0)
    tp_px = entry * (1.0 + tp / 100.0) if tp > 0.0 else 1e18
    peak = entry
    exit_px = 0.0
    kind = 0
    held = n - 1
    m0 = P[s, 0]
    for j in range(n):
        r = s + j
        minute, o, h, l, c = P[r, 0], P[r, 1], P[r, 2], P[r, 3], P[r, 4]
        elapsed = minute - m0          # minutes since the entry bar (a missing minute still counts as time)
        trend = False
        if mode == 1.0 and j > 0:
            pm5, pm20, pc = P[r - 1, 5], P[r - 1, 6], P[r - 1, 4]
            trend = pm5 > pm20 and pc > pm20
        # leg sequence inside the bar
        if (c >= o) != flip:
            a1, a2, a3 = l, h, c
        else:
            a1, a2, a3 = h, l, c
        prev = o
        for leg in range(4):
            if leg == 0:
                tgt = o          # the open itself (gap check)
            elif leg == 1:
                tgt = a1
            elif leg == 2:
                tgt = a2
            else:
                tgt = a3
            lo = tgt if tgt < prev else prev
            hi = tgt if tgt > prev else prev
            if tgt >= prev:      # rising leg (or the open): tp first, then peak
                if hi >= tp_px:
                    exit_px = tp_px if prev < tp_px else prev
                    exit_px = _dn(exit_px)
                    kind = 3
                    break
                if hi > peak:
                    peak = hi
            if tgt <= prev or leg == 0:      # falling leg (or the open)
                # lock level from the peak known so far
                lock_px = -1.0
                if peak >= arm_px:
                    G = give
                    if mode == 1.0:
                        if trend:
                            G = give2
                    elif mode == 2.0:
                        g2 = p1 * (peak / entry - 1.0) * 100.0
                        if g2 > G:
                            G = g2
                    elif mode == 3.0:
                        if peak >= entry * (1.0 + p1 / 100.0):
                            G = give2
                    lock_px = peak * (1.0 - G / 100.0)
                stop_on = elapsed >= hold
                lvl = -1.0
                k2 = 0
                if lock_px > 0.0 and lo <= lock_px:
                    lvl = lock_px
                    k2 = 2
                if stop_on and lo <= stop_px and stop_px > lvl:
                    lvl = stop_px
                    k2 = 1
                if k2 > 0:
                    fill = lvl if prev > lvl else prev      # gapped through -> the price we are at
                    exit_px = _dn(fill - _tick(fill))
                    kind = k2
                    break
            prev = tgt
        if kind > 0:
            held = j
            break
        # liquidation at the close of the EOD bar / the last bar / the time cap - after this bar's own stop, lock and tp
        if minute >= EOD_MIN or j == n - 1 or (cap > 0.0 and elapsed + 1.0 >= cap):
            exit_px = _dn(c - _tick(c))
            kind = 4
            held = j
            break
    net = (exit_px * (1.0 - FEE - TAX) - entry * (1.0 + FEE)) / entry * 100.0
    return net, held, (peak / entry - 1.0) * 100.0, kind


@njit(cache=True, parallel=True)
def sim_grid(P, starts, lens, entry, atr_pct, group, bar, prms, sequence, flip=False):
    """net[k, e] (nan = skipped by sequencing), kind[k, e], held[k, e]."""
    K, E = prms.shape[0], starts.shape[0]
    net = np.full((K, E), np.nan, dtype=np.float32)
    kind = np.zeros((K, E), dtype=np.int8)
    held = np.zeros((K, E), dtype=np.int16)
    for k in prange(K):
        last_group = -1
        busy_until = -1
        for e in range(E):
            if sequence:
                if group[e] != last_group:
                    last_group = group[e]
                    busy_until = -1
                if bar[e] < busy_until:
                    continue
            r, hbars, _, kd = _sim_one(P, starts[e], lens[e], entry[e], atr_pct[e], prms[k], flip)
            net[k, e] = r
            kind[k, e] = kd
            held[k, e] = hbars
            busy_until = bar[e] + 1 + hbars + 1
    return net, kind, held


@njit(cache=True)
def excursions(P, starts, lens, entry, levels):
    """Per event: MFE/MAE to EOD, and for each stop level x: touched?, later max % (after the touch), close-at-EOD %.
    Also the deepest dip before the first touch of +1% (nan when +1% is never reached)."""
    E, L = starts.shape[0], levels.shape[0]
    mfe = np.zeros(E)
    mae = np.zeros(E)
    eod = np.zeros(E)
    dip_before_1 = np.full(E, np.nan)
    touched = np.zeros((E, L), dtype=np.bool_)
    later_max = np.full((E, L), np.nan)
    for e in range(E):
        s, n, en = starts[e], lens[e], entry[e]
        hi, lo = en, en
        reached = False
        for j in range(n):
            r = s + j
            if P[r, 0] > EOD_MIN:
                break
            h, l, c = P[r, 2], P[r, 3], P[r, 4]
            bull = c >= P[r, 1]
            # bullish bar: low first; bearish: high first
            if bull:
                if l < lo:
                    lo = l
            for x in range(L):
                if touched[e, x]:
                    v = (h / en - 1.0) * 100.0
                    if not (later_max[e, x] >= v):
                        later_max[e, x] = v
            if not reached and h >= en * 1.01:
                reached = True
                dip_before_1[e] = (lo / en - 1.0) * 100.0
            if h > hi:
                hi = h
            if l < lo:
                lo = l
            for x in range(L):
                if not touched[e, x] and l <= en * (1.0 - levels[x] / 100.0):
                    touched[e, x] = True
                    # bullish bar = low before high: the bar's own high comes after the touch
                    later_max[e, x] = ((h if bull else c) / en - 1.0) * 100.0
            eod[e] = (c / en - 1.0) * 100.0
        mfe[e] = (hi / en - 1.0) * 100.0
        mae[e] = (lo / en - 1.0) * 100.0
    return mfe, mae, eod, dip_before_1, touched, later_max


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def entry_prices(ev: pd.DataFrame, flat: np.ndarray, starts: np.ndarray) -> np.ndarray:
    if "buy_price" in ev.columns:
        return ev["buy_price"].to_numpy(dtype=np.float64)
    o = flat[starts, 1]
    return np.array([_up(x) + _tick(x) for x in o])


def split_dates(ev: pd.DataFrame, frac: float) -> tuple[np.ndarray, np.ndarray, str]:
    dates = sorted(ev["date"].unique())
    cut = dates[int(len(dates) * frac)]
    tr = (ev["date"] < cut).to_numpy()
    return tr, ~tr, cut


def day_stats(net: np.ndarray, dates: np.ndarray, mask: np.ndarray) -> dict:
    """Trade mean and a by-day t-stat (days are the independent unit, trades inside a day are correlated)."""
    ok = mask & np.isfinite(net)
    if ok.sum() == 0:
        return {"n": 0, "mean": np.nan, "win": np.nan, "t_day": np.nan, "sum": 0.0, "days_pos": np.nan}
    s = pd.Series(net[ok]).groupby(dates[ok]).sum()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std(ddof=1) > 0 else np.nan
    return {"n": int(ok.sum()), "mean": float(net[ok].mean()), "win": float((net[ok] > 0).mean()),
            "t_day": float(t), "sum": float(net[ok].sum()), "days_pos": float((s > 0).mean())}


def grid() -> np.ndarray:
    rows = []
    stops = [0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
    holds = [0, 4]
    arms = [0.6, 1.0, 1.5, 2.0]
    gives = [0.3, 0.4, 0.6, 0.8, 1.2]
    for stop, hold, arm, give in itertools.product(stops, holds, arms, gives):
        rows.append((stop, 0, hold, arm, give, 0, 0, 0, 0, 0))
    # ATR-scaled stop (cap = the fixed stop)
    for stop, k, arm, give in itertools.product([2.0, 3.0], [3, 4, 6, 8], [1.0], [0.4, 0.8]):
        rows.append((stop, k, 4, arm, give, 0, 0, 0, 0, 0))
    # let-it-run variants on a few stops
    for stop, arm, give in itertools.product([1.2, 1.7, 2.5], [1.0, 1.5], [0.4, 0.6]):
        for give2 in (0.8, 1.2, 2.0):
            rows.append((stop, 0, 4, arm, give, 1, give2, 0, 0, 0))            # trend hold
        for ratio in (0.25, 0.35, 0.5):
            rows.append((stop, 0, 4, arm, give, 2, 0, ratio, 0, 0))            # giveback = ratio of the peak profit
        for p1, give2 in itertools.product((2.0, 3.0), (0.8, 1.2, 2.0)):
            rows.append((stop, 0, 4, arm, give, 3, give2, p1, 0, 0))           # step at +p1%
    # fixed take profit instead of / on top of the lock
    for stop, tp in itertools.product([1.2, 1.7, 2.5], [1.0, 1.5, 2.0, 3.0, 5.0]):
        rows.append((stop, 0, 4, 1.0, 0.4, 0, 0, 0, tp, 0))
        rows.append((stop, 0, 4, 99.0, 0.4, 0, 0, 0, tp, 0))                   # tp only (no lock)
    # no lock, no tp: stop + EOD only
    for stop in stops:
        rows.append((stop, 0, 4, 99.0, 0.4, 0, 0, 0, 0, 0))
    # pre-declared exits for the setup families (Codex review): lock + 60 min cap, wide lock + cap, stop + 45 min cap
    rows.append((1.5, 0, 0, 1.0, 0.4, 0, 0, 0, 0, 60))
    rows.append((2.0, 0, 0, 2.0, 0.8, 0, 0, 0, 0, 60))
    rows.append((2.0, 0, 0, 99.0, 0.4, 0, 0, 0, 0, 45))
    # time cap (minutes held) on the live-like lock
    for stop, cap in itertools.product([1.2, 1.7, 2.5], [30, 60, 120]):
        rows.append((stop, 0, 4, 1.0, 0.4, 0, 0, 0, 0, cap))
    return np.array(rows, dtype=np.float64)


FAMILIES = ["ma_cross", "random_up", "random", "orb_rs", "pb_vwap", "dip_rev"]
LIVE = (1.7, 0, 4, 1.0, 0.4, 0, 0, 0, 0, 0)      # live proxy: hard stop 1.7% after 4 min, lock +1.0% / 0.4%


def _prep(real: bool):
    ev, flat, starts, lens = load_events(real)
    entry = entry_prices(ev, flat, starts)
    atr = ev["atr_pct"].to_numpy(dtype=np.float64)
    group = pd.factorize(ev["date"] + ev["code"] + ev["family"])[0].astype(np.int64)
    bar = ev["bar"].to_numpy(dtype=np.int64)
    return ev, flat, starts, lens, entry, atr, group, bar


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def report_dip(args) -> None:
    out = []
    levels = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0])
    for real in (False, True):
        ev, flat, starts, lens, entry, *_ = _prep(real)
        mfe, mae, eod, dip1, touched, later = excursions(flat, starts, lens, entry, levels)
        fams = ["real"] if real else FAMILIES
        for fam in fams:
            m = (ev["family"] == fam).to_numpy()
            if not m.any():
                continue
            out.append(f"\n## {fam}  n={m.sum()}  (gross %, from the entry fill to 15:15)")
            out.append(f"MFE median {np.median(mfe[m]):.2f} / mean {mfe[m].mean():.2f}   MAE median {np.median(mae[m]):.2f} / mean {mae[m].mean():.2f}"
                       f"   EOD mean {eod[m].mean():.3f}")
            r1 = m & np.isfinite(dip1)
            q = np.percentile(dip1[r1], [50, 75, 90, 95, 99]) if r1.sum() else [np.nan] * 5
            out.append(f"reach +1%: {r1.mean() / max(m.mean(), 1e-9) * 100:.1f}% of events; their deepest dip BEFORE +1%: "
                       f"median {q[0]:.2f}, 75% within {np.percentile(dip1[r1], 25):.2f}, 90% within {np.percentile(dip1[r1], 10):.2f}, "
                       f"95% within {np.percentile(dip1[r1], 5):.2f}")
            out.append("| stop x% | touched | of touched: back to entry | back to +1% | EOD mean if held | EOD mean if never touched |")
            out.append("|---|---|---|---|---|---|")
            for x, lv in enumerate(levels):
                t = m & touched[:, x]
                if t.sum() == 0:
                    continue
                out.append(f"| -{lv:.1f} | {t.sum() / m.sum() * 100:.1f}% | {(later[t, x] >= 0).mean() * 100:.1f}% | "
                           f"{(later[t, x] >= 1.0).mean() * 100:.1f}% | {eod[t].mean():+.2f} | {eod[m & ~touched[:, x]].mean():+.2f} |")
    text = "\n".join(out)
    (g026.OUT_DIR / "report_dip.md").write_text(text, encoding="utf-8")
    print(text)


def report_grid(args) -> None:
    prms = np.vstack([np.array([LIVE]), grid()])
    rows = []
    for real in (False, True):
        ev, flat, starts, lens, entry, atr, group, bar = _prep(real)
        t0 = time.time()
        net, kind, held = sim_grid(flat, starts, lens, entry, atr, group, bar, prms, True)
        net_f, _, _ = sim_grid(flat, starts, lens, entry, atr, group, bar, prms, True, True)   # reversed intrabar order
        print(f"sim real={real} events={len(ev)} combos={len(prms)} {time.time() - t0:.0f}s", flush=True)
        dates = ev["date"].to_numpy()
        tr, ho, cut = split_dates(ev, args.train_frac)
        fams = ["real"] if real else FAMILIES
        for fam in fams:
            fm = (ev["family"] == fam).to_numpy()
            for k in range(len(prms)):
                row = {"family": fam, "k": k, **dict(zip(PRM_COLS, prms[k]))}
                for name, mask in (("tr", tr & fm), ("ho", ho & fm), ("all", fm)):
                    st = day_stats(net[k], dates, mask)
                    row.update({f"{name}_{a}": b for a, b in st.items()})
                row["all_mean_flip"] = day_stats(net_f[k], dates, fm)["mean"]
                kd = kind[k][fm & np.isfinite(net[k])]
                row.update({"p_stop": float((kd == 1).mean()), "p_lock": float((kd == 2).mean()), "p_tp": float((kd == 3).mean()),
                            "p_eod": float((kd == 4).mean()), "held_med": float(np.median(held[k][fm & np.isfinite(net[k])]))})
                rows.append(row)
        if not real:
            print(f"train < {cut} <= holdout")
    df = pd.DataFrame(rows)
    df.to_csv(g026.OUT_DIR / "report_grid.csv", index=False)
    print(f"-> {g026.OUT_DIR / 'report_grid.csv'}")


def report_entry(args) -> None:
    ev, flat, starts, lens, entry, atr, group, bar = _prep(False)
    prm = np.array([tuple(float(x) for x in args.policy.split(","))]) if args.policy else np.array([LIVE])
    net, _, _ = sim_grid(flat, starts, lens, entry, atr, group, bar, prm, False)
    ev["net"] = net[0]
    tr, ho, cut = split_dates(ev, args.train_frac)
    feats = [c for c in ev.columns if c not in ("date", "code", "family", "bar", "trigger_close", "entry_minute", "path", "net")]
    out = []
    for fam in ("ma_cross", "random_up"):
        fm = (ev["family"] == fam).to_numpy()
        base_tr, base_ho = ev.loc[fm & tr, "net"].mean(), ev.loc[fm & ho, "net"].mean()
        out.append(f"\n## {fam}: baseline net% train {base_tr:+.3f} (n={int((fm & tr).sum())}) holdout {base_ho:+.3f} (n={int((fm & ho).sum())})")
        out.append("| feature | bucket (train quintile) | range | train net% | holdout net% | n_tr | n_ho |")
        out.append("|---|---|---|---|---|---|---|")
        for f in feats:
            x = pd.to_numeric(ev[f], errors="coerce")
            xt = x[fm & tr].dropna()
            if xt.nunique() < 5:
                continue
            edges = np.unique(np.quantile(xt, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            b = pd.cut(x, edges, labels=False)
            for q in range(len(edges) - 1):
                mt, mh = fm & tr & (b == q), fm & ho & (b == q)
                out.append(f"| {f} | {q + 1} | {edges[q]:.3g} ~ {edges[q + 1]:.3g} | {ev.loc[mt, 'net'].mean():+.3f} | "
                           f"{ev.loc[mh, 'net'].mean():+.3f} | {int(mt.sum())} | {int(mh.sum())} |")
    text = "\n".join(out)
    (g026.OUT_DIR / "report_entry.md").write_text(text, encoding="utf-8")
    print(f"-> {g026.OUT_DIR / 'report_entry.md'} ({len(out)} lines)")


def load_panel() -> pd.DataFrame:
    files = sorted((g026.OUT_DIR / "panel").glob("panel_*.pkl"))
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)


PANEL_LABELS = ["entry", "f15", "f30", "f60", "feod", "mfe60", "mae60", "mfe_eod", "net_live", "net_tight", "net_run",
                "held_live", "held_tight", "held_run"]


def krx_tick(p: np.ndarray) -> np.ndarray:
    return np.select([p < 2000, p < 5000, p < 20000, p < 50000, p < 200000, p < 500000], [1, 5, 10, 50, 100, 500], 1000)


def report_cost(args) -> None:
    """Net / gross by tick size as a share of the price (the mechanical part of the round-trip cost)."""
    d = load_panel()
    d["tick_pct"] = krx_tick(d["price"].to_numpy()) / d["price"] * 100.0
    cut = sorted(d["date"].unique())[int(d["date"].nunique() * args.train_frac)]
    d["split"] = np.where(d["date"] < cut, "train", "holdout")
    b = pd.cut(d["tick_pct"], [0, 0.06, 0.08, 0.10, 0.13, 0.17, 0.21, 0.30])
    out = ["## panel (every 3rd bar): by tick/price %",
           "| tick % | n | net_live train | net_live holdout | gross f60 | gross fEOD |", "|---|---|---|---|---|---|"]
    for key, g in d.groupby(b, observed=True):
        out.append(f"| {key} | {len(g)} | {g.loc[g.split == 'train', 'net_live'].mean():+.3f} | "
                   f"{g.loc[g.split == 'holdout', 'net_live'].mean():+.3f} | {g['f60'].mean():+.3f} | {g['feod'].mean():+.3f} |")
    for thr in (0.08, 0.10, 0.13, 0.17):
        g = d[d["tick_pct"] <= thr]
        out.append(f"tick <= {thr:.2f}%: {len(g) / len(d) * 100:.0f}% of bars, net_live {g['net_live'].mean():+.3f} "
                   f"(all {d['net_live'].mean():+.3f})")
    ev, *_ = _prep(True)
    ev["tick_pct"] = krx_tick(ev["buy_price"].to_numpy()) / ev["buy_price"] * 100.0
    out += ["", "## real trades (actual result, gross of fee/tax)", "| tick % | n | mean % | median % | sum KRW |", "|---|---|---|---|---|"]
    for key, g in ev.groupby(pd.cut(ev["tick_pct"], [0, 0.08, 0.10, 0.13, 0.17, 0.30]), observed=True):
        out.append(f"| {key} | {len(g)} | {g['real_ret_pct'].mean():+.3f} | {g['real_ret_pct'].median():+.3f} | "
                   f"{g['real_pnl_krw'].sum():+,.0f} |")
    text = "\n".join(out)
    (g026.OUT_DIR / "report_cost.md").write_text(text, encoding="utf-8")
    print(text)


def report_model(args) -> None:
    """Walk-forward LightGBM on the panel: can any feature combination predict the after-cost result?
    Fold = train days / next days for early stopping / later days for the test; ranks are taken inside each test fold."""
    import lightgbm as lgb
    d = load_panel()
    feats = [c for c in d.columns if c not in PANEL_LABELS + ["code", "bar", "date", "price"]]
    dates = sorted(d["date"].unique())
    n = len(dates)
    cuts = [(0.55, 0.65, 0.82), (0.72, 0.82, 1.0)]
    out = [f"features ({len(feats)}): {', '.join(feats)}"]
    for target in ("net_live", "net_run", "f60"):
        res = []
        for a, b, c in cuts:
            tr, va, te = dates[:int(n * a)], dates[int(n * a):int(n * b)], dates[int(n * b):int(n * c)]
            A, B = d[d["date"].isin(tr) & d[target].notna()], d[d["date"].isin(va) & d[target].notna()]
            C = d[d["date"].isin(te) & d[target].notna()].copy()
            prm = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, bagging_fraction=0.7,
                       bagging_freq=1, feature_fraction=0.7, lambda_l2=5, verbose=-1, num_threads=12, seed=7)
            m = lgb.train(prm, lgb.Dataset(A[feats], A[target].clip(-5, 5)), 600,
                          valid_sets=[lgb.Dataset(B[feats], B[target].clip(-5, 5))],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            C["p"] = m.predict(C[feats], num_iteration=m.best_iteration)
            C["q"] = C["p"].rank(pct=True)
            res.append(C)
            out.append(f"{target}: train {tr[0]}~{tr[-1]} valid {va[0]}~{va[-1]} test {te[0]}~{te[-1]} trees {m.best_iteration} "
                       f"corr {np.corrcoef(C['p'], C[target])[0, 1]:+.4f}")
        C = pd.concat(res)
        out += ["| predicted top | n | net_live | net_tight | net_run | gross f60 | gross fEOD | days |", "|---|---|---|---|---|---|---|---|"]
        for lo in (0.0, 0.5, 0.9, 0.95, 0.98, 0.99):
            x = C[C["q"] >= lo]
            out.append(f"| {100 * (1 - lo):.0f}% | {len(x)} | {x['net_live'].mean():+.3f} | {x['net_tight'].mean():+.3f} | "
                       f"{x['net_run'].mean():+.3f} | {x['f60'].mean():+.3f} | {x['feod'].mean():+.3f} | {x['date'].nunique()} |")
    text = "\n".join(out)
    (g026.OUT_DIR / "report_model.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", choices=("dip", "grid", "entry", "cost", "model"), required=True)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--policy", help="entry report: exit policy as 10 comma separated values " + ",".join(PRM_COLS))
    args = ap.parse_args()
    {"dip": report_dip, "grid": report_grid, "entry": report_entry, "cost": report_cost, "model": report_model}[args.report](args)


if __name__ == "__main__":
    main()
