# -*- coding: utf-8 -*-
"""Tune the entry rule + exit/add parameters of the user position policy on g017 events (2026-10-31).

Split by date: the first --train-frac of days choose the rule (random search over entry filters x every g017 grid
combo); the remaining days are a holdout that is only reported, never used for selection.
Trades are sequenced like the live bot: one position per code, a new entry only after the previous one exited
(exit bar = entry bar + hold_ref), at most --max-per-day entries per day across all codes (earliest first).
Objective (train): maximise total net KRW among rules with win rate >= --min-win and >= --min-trades trades.
Writes runtime/edge_ai/policy/rule_<ts>.json (+ the best one as rule_best.json) and prints a report.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

POLICY_DIR = Path(os.environ.get("AUTO_TRADING_RUNTIME") or Path(__file__).resolve().parents[3] / "runtime") / "edge_ai" / "policy"

# filter space: column -> (op, candidate thresholds); None = filter off
SPACE = {
    "d_ma5_ma20": (">", [None, 0.0, 1.0, 2.0, 4.0]),
    "d_ma20_slope5": (">", [0.0, 1.0, 2.0, 4.0]),
    "d_ma20_ma60": (">", [None, 0.0, 3.0]),
    "d_ret5": ("<", [None, 10.0, 20.0]),
    "m3_above": (">=", [None, 1]),
    "m3_age": ("<=", [None, 3, 6, 12]),
    "m3_ma20_slope": (">", [None, 0.0, 0.1]),
    "bb_gap": ("<=", [None, 0.5, 1.0, 1.5]),
    "bb_mid_slope3": (">", [None, 0.0, 0.05, 0.1]),
    "bb_pctb": ("<=", [None, 0.8, 0.9, 1.0]),
    "bb_width": (">=", [None, 1.0, 2.0]),
    "cross_age": ("<=", [None, 0, 1]),
    "candle_bull": (">=", [None, 1]),
    "vol_ratio": (">=", [None, 1.0, 1.5, 2.5]),
    "rsi": ("<=", [None, 70.0, 80.0]),
    "day_ret": ("<=", [None, 5.0, 10.0, 15.0]),
    "day_ret_min": (">=", [None, -1.0, 0.0, 1.0]),
    "vwap_gap": (">=", [None, 0.0]),
    "minute": ("<=", [None, 600, 660, 780]),
}
COLMAP = {"day_ret_min": "day_ret"}


def apply_rule(df: pd.DataFrame, rule: dict) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for k, spec in rule["filters"].items():
        op, thr = spec
        if thr is None:
            continue
        if k == "minute" and "minute" not in df.columns:
            col = df["ts"].str.slice(0, 2).astype(int) * 60 + df["ts"].str.slice(3, 5).astype(int)
        else:
            col = df[COLMAP.get(k, k)]
        x = col.to_numpy(float)
        with np.errstate(invalid="ignore"):
            if op == ">":
                m &= x > thr
            elif op == ">=":
                m &= x >= thr
            elif op == "<":
                m &= x < thr
            else:
                m &= x <= thr
    return m


def load_events(src: Path) -> pd.DataFrame:
    parts = [pd.read_csv(f, dtype={"code": str, "date": str}) for f in sorted(src.glob("events_*.csv.gz"))]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df.assign(minute=df["ts"].str.slice(0, 2).astype(int) * 60 + df["ts"].str.slice(3, 5).astype(int))
    df = df.sort_values(["date", "minute", "code"]).reset_index(drop=True)
    return df


@njit(cache=True)
def _sequence(sel, day, code, bar, hold, n_codes, max_per_day):
    busy = np.full(n_codes, -1, dtype=np.int64)
    busy_day = np.full(n_codes, -1, dtype=np.int64)
    out = np.empty(len(sel), dtype=np.int64)
    m = 0
    cur_day = -1
    count = 0
    for j in range(len(sel)):
        r = sel[j]
        if day[r] != cur_day:
            cur_day = day[r]
            count = 0
        cd = code[r]
        if (busy_day[cd] == day[r] and busy[cd] >= bar[r]) or count >= max_per_day:
            continue
        busy[cd] = bar[r] + hold[r]
        busy_day[cd] = day[r]
        count += 1
        out[m] = r
        m += 1
    return out[:m]


def sequence(df: pd.DataFrame, mask: np.ndarray, max_per_day: int, combo: int | None = None) -> np.ndarray:
    """Indices of the events the live bot would actually take (one open position per code, entry after that
    combo's exit, at most max_per_day entries per day). df must be sorted by date, minute."""
    arr = _arrays(df)
    hold = arr["hold"][combo] if combo is not None else arr["hold_ref"]
    return _sequence(np.flatnonzero(mask), arr["day"], arr["code"], arr["bar"], hold, arr["n_codes"], max_per_day)


_ARR_CACHE: dict = {}


def _arrays(df: pd.DataFrame) -> dict:
    key = id(df)
    hit = _ARR_CACHE.get(key)
    if hit is None or hit["df"] is not df:  # ids are reused after a frame is freed: check identity
        _ARR_CACHE.clear()
        hcols = sorted((c for c in df.columns if c[:1] == "h" and c[1:].isdigit()), key=lambda c: int(c[1:]))
        _ARR_CACHE[key] = {
            "day": pd.factorize(df["date"])[0].astype(np.int64),
            "code": pd.factorize(df["code"])[0].astype(np.int64),
            "n_codes": int(df["code"].nunique()),
            "bar": df["bar_i"].to_numpy(np.int64),
            "hold_ref": df["hold_ref"].to_numpy(np.int64),
            "hold": {int(c[1:]): df[c].to_numpy(np.int64) for c in hcols},
            "df": df,
        }
    return _ARR_CACHE[key]


def stats(net: np.ndarray, days: int) -> dict:
    if len(net) == 0:
        return {"trades": 0, "win": 0.0, "net_krw": 0.0, "avg_krw": 0.0, "per_day": 0.0, "pf": 0.0}
    gain, loss = net[net > 0].sum(), -net[net <= 0].sum()
    return {"trades": int(len(net)), "win": float((net > 0).mean()), "net_krw": float(net.sum()),
            "avg_krw": float(net.mean()), "per_day": len(net) / max(days, 1),
            "pf": float(gain / loss) if loss > 0 else float("inf")}


def random_rule(rng) -> dict:
    return {"filters": {k: (op, vals[rng.integers(len(vals))]) for k, (op, vals) in SPACE.items()}}


def evaluate(df, rule, ncols, max_per_day, days, combos=None):
    """-> {combo: net array of the trades taken under that combo's own exits}"""
    mask = apply_rule(df, rule)
    ks = range(ncols.shape[1]) if combos is None else combos
    return {k: ncols[sequence(df, mask, max_per_day, k), k] for k in ks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(POLICY_DIR / "events"))
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--min-win", type=float, default=0.60)
    ap.add_argument("--min-trades-per-day", type=float, default=1.0)
    ap.add_argument("--max-per-day", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    df = load_events(Path(args.src))
    grid = json.loads((POLICY_DIR / "grid.json").read_text())
    ncol_names = [f"n{k}" for k in range(len(grid["rows"]))]
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * args.train_frac)]
    tr = df[df["date"] < cut].reset_index(drop=True)
    te = df[df["date"] >= cut].reset_index(drop=True)
    tr_days, te_days = tr["date"].nunique(), te["date"].nunique()
    n_tr, n_te = tr[ncol_names].to_numpy(float), te[ncol_names].to_numpy(float)
    print(f"events={len(df)} days={len(dates)} train<{cut}: {len(tr)} ev/{tr_days}d  holdout: {len(te)} ev/{te_days}d")

    base = {"filters": {k: (op, None) for k, (op, _) in SPACE.items()}}
    base["filters"]["d_ma20_slope5"] = (">", 0.0)
    rng = np.random.default_rng(args.seed)
    results = []
    t0 = time.time()
    min_trades = args.min_trades_per_day * tr_days
    for s in range(args.samples):
        rule = base if s == 0 else random_rule(rng)
        if apply_rule(tr, rule).sum() < min_trades:
            continue
        for k, nets in evaluate(tr, rule, n_tr, args.max_per_day, tr_days).items():
            if len(nets) >= min_trades:
                results.append((rule, k, len(nets), float((nets > 0).mean()), float(nets.sum())))
    print(f"searched {args.samples} rules x {len(ncol_names)} combos in {time.time() - t0:.0f}s, {len(results)} feasible")
    ok = [r for r in results if r[3] >= args.min_win]
    by_net = bool(ok)  # list.sort empties the list while sorting - do not reference `ok` inside the key
    pool = ok if ok else results
    pool.sort(key=lambda r: (r[4] if by_net else r[3]), reverse=True)
    if not ok:
        print(f"!! no rule reached win >= {args.min_win:.0%} on train - ranking by win rate instead")

    def describe(rule, k):
        f = {kk: v for kk, v in rule["filters"].items() if v[1] is not None}
        return f, dict(zip(grid["axes"].keys(), grid["rows"][k]))

    out = []
    seen = set()
    for rule, k, n, w, tot in pool:
        key = (json.dumps(rule, sort_keys=True), k)
        if key in seen:
            continue
        seen.add(key)
        st_tr = stats(evaluate(tr, rule, n_tr, args.max_per_day, tr_days, [k])[k], tr_days)
        st_te = stats(evaluate(te, rule, n_te, args.max_per_day, te_days, [k])[k], te_days)
        f, p = describe(rule, k)
        out.append({"rule": rule, "combo": k, "params": p, "filters": f, "train": st_tr, "holdout": st_te})
        if len(out) >= args.top:
            break
    for r in out:
        a, b = r["train"], r["holdout"]
        print(f"combo={r['combo']:2d} {r['params']}\n  filters={r['filters']}\n"
              f"  TRAIN n={a['trades']} ({a['per_day']:.1f}/d) win={a['win']:.1%} net={a['net_krw']:,.0f} avg={a['avg_krw']:,.0f} pf={a['pf']:.2f}\n"
              f"  HOLD  n={b['trades']} ({b['per_day']:.1f}/d) win={b['win']:.1%} net={b['net_krw']:,.0f} avg={b['avg_krw']:,.0f} pf={b['pf']:.2f}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = [{"filters": r["rule"]["filters"], "combo": r["combo"], "params": r["params"], "train": r["train"],
                "holdout": r["holdout"], "cut": cut} for r in out]
    (POLICY_DIR / f"rules_{stamp}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    if payload:
        (POLICY_DIR / "rule_best.json").write_text(json.dumps(payload[0], indent=1, ensure_ascii=False))
    print(f"saved {POLICY_DIR / f'rules_{stamp}.json'}")


if __name__ == "__main__":
    main()
