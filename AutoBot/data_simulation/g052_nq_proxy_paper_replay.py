# -*- coding: utf-8 -*-
"""Codex minimum experiment: the Edge AI paper trades as they happened, with the candidate rule "no buy when 133690's 30-minute
return <= thr" judged only from 1-minute bars that had CLOSED before the entry time. Read-only."""
import glob, json, os
import numpy as np
import pandas as pd

DATA = r"D:\make_utils\Trading\runtime\data"
tr = []
for f in sorted(glob.glob(os.path.join(DATA, "live_runtime", "edge_only_2026*.jsonl"))):
    ent = {}
    for line in open(f, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "ENTRY":
            ent[r["code"]] = r
        elif r.get("event") == "EXIT" and r.get("net_pct") is not None:
            e = ent.pop(r["code"], None)
            if e:
                tr.append((pd.Timestamp(e["ts"]), r["code"], float(r["net_pct"])))
t = pd.DataFrame(tr, columns=["ts", "code", "net"])
t = t[(t["ts"].dt.time >= pd.Timestamp("09:00").time()) & (t["ts"].dt.time < pd.Timestamp("15:20").time())].copy()
print("paper trades (regular session):", len(t), "days", t["ts"].dt.date.nunique())
cache = {}


def bars(code, d):
    if (code, d) not in cache:
        fs = glob.glob(os.path.join(DATA, d, f"{code}_*_10s.txt"))
        s = None
        if fs:
            df = pd.read_csv(fs[0], usecols=["datetime", "close", "raw_bar"], encoding="utf-8-sig")
            df = df[df["raw_bar"] == 1]
            s = pd.Series(df["close"].values, index=pd.to_datetime(df["datetime"]) + pd.Timedelta(minutes=1))   # index = bar END
        cache[(code, d)] = s
    return cache[(code, d)]


def ret30(code, ts):
    s = bars(code, ts.strftime("%Y%m%d"))
    if s is None:
        return np.nan
    a = s[:ts]                                # bars that ended at or before the entry
    if len(a) < 31 or (ts - a.index[-1]) > pd.Timedelta(minutes=3):
        return np.nan                         # fewer than 30 minutes of today's bars / stale: missing, never yesterday's bars
    return (a.iloc[-1] / a.iloc[-31] - 1) * 100


for code in ("133690", "069500"):
    t[code] = [ret30(code, x) for x in t["ts"]]
print("missing proxy value:", int(t["133690"].isna().sum()), "of", len(t), "(entries before 09:31 have no 30-minute return today)")
u = t.dropna(subset=["133690"])
print(f"usable {len(u)} trades / {u['ts'].dt.date.nunique()} days | all: sum {u['net'].sum():+.2f}%p mean {u['net'].mean():+.3f}% win {100*(u['net']>0).mean():.0f}%")
for code, nm in (("133690", "Nasdaq-100 ETF"), ("069500", "KODEX200")):
    for thr in (-0.3, -0.2, -0.1):
        b = u[u[code] <= thr]
        k = u[u[code] > thr]
        byday = u.assign(blocked=u[code] <= thr).groupby(u["ts"].dt.date).apply(lambda g: -g[g["blocked"]]["net"].sum())
        loo = [byday.drop(d).sum() for d in byday.index]
        print(f" {nm:<14} thr {thr:+.1f}%: blocked {len(b):>2} (winners lost {int((b['net']>0).sum())}, mean {b['net'].mean() if len(b) else float('nan'):+.3f}%) "
              f"| day P&L change {byday.sum():+.2f}%p, per day {[round(x,2) for x in byday.values]} | leave-one-day-out {min(loo):+.2f}..{max(loo):+.2f}")
