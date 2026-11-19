# -*- coding: utf-8 -*-
"""g046_gated_hold.py - V0 picks + market gate + tick filter + holding period (selection redesign follow-up, research only).

Why: g045 (2026-11-19) - the picks beat the tradable universe more the longer they are held (10 days: -0.75% vs -3.11%), and were
positive on the rising train days (strict picks 10 days +0.56%), but every long trade lost heavily in the Oct-Nov 2026 fall
(KODEX 200 -29%, KOSDAQ150 -25%). Picks whose tick is >= 0.20% of the price were the worst in train and holdout alike.
This checks the combination with a market gate fixed in advance - the same gate as g042 (HI52 log), not tuned here:
  gate UP on day D = both index ETFs (069500, 229200; r001 MARKET_REGIME_ETF): close_D > MA60_D and MA60_D > MA60 ten rows earlier
  (daily_history closes; known at the close of D, the trade starts at the open of D+1 - no look-ahead).

Pre-declared grid (not tuned afterwards):
  selection  picked (strict + fallback) / picked strict
  tick       all / tick(close_D)/close_D < 0.20%
  gate       none / UP
  hold       intraday (g044 base rule, 09:05 entry) and k = 1, 3, 5, 10, 20 trading days (g045: D+1 open +1 tick -> close of the
             k-th day -1 tick, fee 0.015% x 2 + tax 0.20%)
Benchmarks on the same days: the tradable universe and the g043 random 20/day sample (with the same gate).
A combination counts only when it is positive (mean and day-average) and above the gated universe in BOTH train and holdout.

Output: <runtime>/research/10_selection/gated_hold_v0.txt
Run:    python g046_gated_hold.py [--rows rows_v0.jsonl.gz]

Update log:
- [2026-11-19] type=feat owner=claude
    summary: 신규 - V0 선정 + 시장 필터(g042 와 같은 지수 ETF 게이트) + 호가 단위 필터 + 보유 기간 조합 검증. 연구용 출력만.
    impact: research tooling (runtime/research/10_selection)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import random
from pathlib import Path

import g043_selection_replay as g043
import g044_exit_rule_compare as g044
import g045_tick_cost_and_hold as g045

ETFS = ("069500", "229200")
HOLDS = [1, 3, 5, 10, 20]
TICK_MAX = 0.0020


def gate_days() -> dict[str, str]:
    """{date: 'UP' | 'DOWN' | 'NA'} from the daily_history closes of both index ETFs."""
    import pandas as pd
    closes = {}
    for e in ETFS:
        df = pd.read_csv(g043.HISTORY_DIR / f"{e}.csv", usecols=["date", "close"], dtype={"date": str})
        closes[e] = df[df["close"] > 0].set_index("date")["close"].astype(float)
    C = pd.DataFrame(closes).dropna().sort_index()
    m60 = C.rolling(60, min_periods=60).mean()
    m60p = m60.shift(10)
    up = ((C > m60) & (m60 > m60p)).all(axis=1)
    known = m60.notna().all(axis=1) & m60p.notna().all(axis=1)
    return {d: ("NA" if not known[d] else "UP" if up[d] else "DOWN") for d in C.index}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="rows_v0.jsonl.gz")
    args = ap.parse_args()
    tag = Path(args.rows).name.replace("rows_", "").replace(".jsonl.gz", "")
    rows = g043.load_rows(g043.RESEARCH_DIR / args.rows)
    dates = sorted({r["date"] for r in rows})
    n_train = max(1, int(round(len(dates) * g043.TRAIN_FRAC)))
    split = {d: ("train" if i < n_train else "holdout") for i, d in enumerate(dates)}
    gate = gate_days()
    pk = [r for r in rows if r["picked"]]
    tr = [r for r in rows if g043.tradable(r)]
    rng = random.Random(20261119)
    rand = []
    for d in dates:
        pool = sorted((r for r in tr if r["date"] == d), key=lambda r: r["code"])
        rand.extend(rng.sample(pool, min(g043.RANDOM_PER_DAY, len(pool))))
    with gzip.open(g043.RESEARCH_DIR / f"exit_bars_{tag}.pkl.gz", "rb") as fh:
        bars = pickle.load(fh)
    nd = {d: g043.next_folder(d) for d in dates}

    def intraday(r):
        b = bars.get((r["code"], nd[r["date"]]))
        t = g044.simulate(b, g044.RULES[0]) if b else None
        return t["net"] if t else None

    parts = ("train", "holdout", "all")
    out = [f"# V0 picks + market gate + tick filter + hold - {args.rows}",
           f"dates {len(dates)}: train {dates[0]}~{dates[n_train - 1]} / holdout {dates[n_train]}~{dates[-1]}",
           "gate UP = 069500 and 229200 both: close > MA60 and MA60 > MA60 10 rows earlier (same as g042); pre-declared grid", ""]
    for part in parts:
        ds = [d for d in dates if part == "all" or split[d] == part]
        cnt = {k: sum(gate.get(d, "NA") == k for d in ds) for k in ("UP", "DOWN", "NA")}
        out.append(f"gate days {part:<8}: {len(ds)} days  UP {cnt['UP']}  DOWN {cnt['DOWN']}  NA {cnt['NA']}")
    out.append("")

    sels = {"picked": pk, "picked strict": [r for r in pk if not r["fallback"]]}
    benches = {"universe": tr, "random 20/day": rand}
    cache: dict = {}

    def value(r, k):
        key = (r["code"], r["date"], k)
        if key not in cache:
            cache[key] = intraday(r) if k == 0 else g045.hold_trade(r["code"], r["date"], k, None)
        return cache[key]

    def series(grp, k, part, gated, tickf):
        xs, ds = [], []
        for r in grp:
            if part != "all" and split[r["date"]] != part:
                continue
            if gated and gate.get(r["date"]) != "UP":
                continue
            if tickf and not ((g045.tick_pct(r) or 1) < TICK_MAX):
                continue
            v = value(r, k)
            if v is not None:
                xs.append(v)
                ds.append(r["date"])
        return xs, ds

    def mean_day(xs, ds):
        by: dict[str, list[float]] = {}
        for x, d in zip(xs, ds):
            by.setdefault(d, []).append(x)
        return g043._mean([g043._mean(v) for v in by.values()]) if by else float("nan")

    verdict = []
    for k in [0] + HOLDS:
        hold = "intraday (09:05, base exit)" if k == 0 else f"hold {k}d (D+1 open -> close of day {k})"
        out.append(f"## {hold}")
        for gated in (False, True):
            for part in parts:
                bx = {b: series(g, k, part, gated, False) for b, g in benches.items()}
                out.append(f"[gate {'UP' if gated else 'none'} / {part}]  universe {g045.st(*bx['universe'])}")
                out.append(f"{'':>28}random   {g045.st(*bx['random 20/day'])}")
                for sname, sg in sels.items():
                    for tickf in (False, True):
                        xs, ds = series(sg, k, part, gated, tickf)
                        label = f"{sname}{' tick<0.20%' if tickf else ''}"
                        uni = g043._mean(bx["universe"][0]) if bx["universe"][0] else float("nan")
                        ex = (g043._mean(xs) - uni) * 100 if xs else float("nan")
                        out.append(f"   {label:<26} {g045.st(xs, ds)}  vs universe {ex:+.2f}%p")
            out.append("")
        # verdict per combination (gated only): positive and above the gated universe in train AND holdout
        for sname, sg in sels.items():
            for tickf in (False, True):
                ok = True
                cells = []
                for part in ("train", "holdout"):
                    xs, ds = series(sg, k, part, True, tickf)
                    ux, _ = series(tr, k, part, True, False)
                    if len(xs) < 20:
                        ok = False
                        cells.append(f"{part} n{len(xs)} (too few)")
                        continue
                    m, md, um = g043._mean(xs), mean_day(xs, ds), g043._mean(ux)
                    ok = ok and m > 0 and md > 0 and m > um
                    cells.append(f"{part} n{len(xs)} {m * 100:+.2f}% day-avg {md * 100:+.2f}% (univ {um * 100:+.2f}%)")
                verdict.append(f"{'PASS' if ok else 'fail'}  {hold:<34} {sname}{' tick<0.20%' if tickf else '':<12} | " + " | ".join(cells))
    out.insert(len(out), "## verdict (gate UP): positive mean + day-average and above the gated universe in BOTH train and holdout")
    out.extend(verdict)
    text = "\n".join(out)
    path = g043.RESEARCH_DIR / f"gated_hold_{tag}.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[OUT] {path}")


if __name__ == "__main__":
    main()
