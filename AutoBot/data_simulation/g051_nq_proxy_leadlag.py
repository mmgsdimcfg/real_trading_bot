# -*- coding: utf-8 -*-
"""Offline falsification with local 1-minute bars only (exact time alignment, all days in runtime/data):
does the Nasdaq-100 proxy (133690) say anything about the NEXT minutes of the KRX index ETFs / watchlist stocks beyond what the
index ETF itself already says? Day-clustered statistics. Read-only."""
import glob, json, os, sys
import numpy as np
import pandas as pd

DATA = r"D:\make_utils\Trading\runtime\data"
days = sorted(d for d in os.listdir(DATA) if d.isdigit() and len(d) == 8)


def load(code, d):
    fs = glob.glob(os.path.join(DATA, d, f"{code}_*_10s.txt"))
    if not fs:
        return None
    try:
        df = pd.read_csv(fs[0], usecols=["datetime", "close", "raw_bar"], encoding="utf-8-sig")
    except Exception:
        return None
    df = df[df["raw_bar"] == 1]
    s = pd.Series(df["close"].values, index=pd.to_datetime(df["datetime"]))
    s = s[~s.index.duplicated()]
    idx = pd.date_range(f"{d} 09:00", f"{d} 15:19", freq="1min")
    if len(s.reindex(idx).dropna()) < 300:
        return None
    return s.reindex(idx).ffill()


# stocks: every code that appears in the paper-trade logs / candidates (the symbols the bot actually watched)
codes = set()
for f in glob.glob(os.path.join(DATA, "live_runtime", "edge_only_2026*.jsonl")):
    for line in open(f, encoding="utf-8"):
        try:
            codes.add(json.loads(line)["code"])
        except Exception:
            pass
codes = sorted(c for c in codes if c not in ("133690", "069500", "229200"))
print("days", len(days), days[0], days[-1], "| stocks", len(codes))

H = (5, 15, 30)
rows = []
for d in days:
    p, k, q = load("133690", d), load("069500", d), load("229200", d)
    if p is None or k is None or q is None:
        continue
    base = pd.DataFrame({"p": p, "k": k, "q": q})
    feat = {}
    for h in H:
        for c in ("p", "k", "q"):
            feat[f"{c}_past{h}"] = base[c].pct_change(h) * 100
            feat[f"{c}_fwd{h}"] = (base[c].shift(-h) / base[c] - 1) * 100
    f = pd.DataFrame(feat)
    f["day"] = d
    f["kind"] = "idx"
    f = f.iloc[30:-30:5]                      # every 5 minutes, away from the edges
    rows.append(f)
    for c in codes:
        s = load(c, d)
        if s is None:
            continue
        g = f[[x for x in f.columns if x.startswith(("p_past", "k_past", "q_past"))]].copy()
        for h in H:
            g[f"s_past{h}"] = (s.pct_change(h) * 100).reindex(g.index)
            g[f"s_fwd{h}"] = ((s.shift(-h) / s - 1) * 100).reindex(g.index)
        g["day"], g["kind"], g["code"] = d, "stk", c
        rows.append(g)
df = pd.concat(rows)
idx = df[df["kind"] == "idx"]
stk = df[df["kind"] == "stk"]
print("index rows", len(idx), "days", idx["day"].nunique(), "| stock rows", len(stk), "days", stk["day"].nunique())


def day_boot(data, fn, n=400, seed=1):
    """value and a 95% interval from resampling whole days"""
    ds = data["day"].unique()
    groups = {d: g for d, g in data.groupby("day")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(ds, len(ds), replace=True)
        vals.append(fn(pd.concat([groups[d] for d in pick])))
    return fn(data), np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def partial(data, x, y, ctrl):
    a = data[[x, y] + ctrl].dropna()
    if len(a) < 50:
        return np.nan
    X = np.column_stack([np.ones(len(a))] + [a[c].values for c in ctrl])
    rx = a[x].values - X @ np.linalg.lstsq(X, a[x].values, rcond=None)[0]
    ry = a[y].values - X @ np.linalg.lstsq(X, a[y].values, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


print("\n[1] same-time corr proxy vs index ETF (1-minute aligned)")
for h in H:
    for c, nm in (("k", "KODEX200"), ("q", "KOSDAQ150")):
        v, lo, hi = day_boot(idx, lambda g: g[f"p_past{h}"].corr(g[f"{c}_past{h}"]))
        print(f"  {h:>2}m {nm:<9} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\n[2] lead: proxy past h -> index ETF next h, raw and partial (controlling the ETF's own past h)")
for h in H:
    for c, nm in (("k", "KODEX200"), ("q", "KOSDAQ150")):
        v, lo, hi = day_boot(idx, lambda g: g[f"p_past{h}"].corr(g[f"{c}_fwd{h}"]))
        pv, plo, phi = day_boot(idx, lambda g: partial(g, f"p_past{h}", f"{c}_fwd{h}", [f"{c}_past{h}"]))
        print(f"  {h:>2}m {nm:<9} raw {v:+.3f} [{lo:+.3f}, {hi:+.3f}] | partial {pv:+.3f} [{plo:+.3f}, {phi:+.3f}]")

print("\n[3] stocks: proxy past h -> stock next h, partial (controlling stock's own past h and KODEX200 past h)")
for h in H:
    v, lo, hi = day_boot(stk, lambda g: g[f"p_past{h}"].corr(g[f"s_fwd{h}"]), n=200)
    pv, plo, phi = day_boot(stk, lambda g: partial(g, f"p_past{h}", f"s_fwd{h}", [f"s_past{h}", f"k_past{h}"]), n=200)
    print(f"  {h:>2}m raw {v:+.3f} [{lo:+.3f}, {hi:+.3f}] | partial {pv:+.3f} [{plo:+.3f}, {phi:+.3f}]")

print("\n[4] rule check on stocks: next 30m mean return when proxy 30m <= thr vs otherwise (difference, day-bootstrap)")
for thr in (-0.2, -0.3, -0.5):
    def diff(g, thr=thr):
        a = g[g["p_past30"] <= thr]["s_fwd30"].mean()
        b = g[g["p_past30"] > thr]["s_fwd30"].mean()
        return a - b
    v, lo, hi = day_boot(stk, diff, n=200)
    m = stk["p_past30"] <= thr
    print(f"  thr {thr:+.1f}%: share blocked {100*m.mean():.1f}% (days with any: {stk[m]['day'].nunique()}) | blocked-minus-kept next30m {v:+.3f}%p [{lo:+.3f}, {hi:+.3f}]")
    # and the same rule on KODEX200's own 30m return (what the bot can already see)
    def diffk(g, thr=thr):
        return g[g["k_past30"] <= thr]["s_fwd30"].mean() - g[g["k_past30"] > thr]["s_fwd30"].mean()
    v, lo, hi = day_boot(stk, diffk, n=200)
    print(f"           same rule on KODEX200 30m: {v:+.3f}%p [{lo:+.3f}, {hi:+.3f}]")
