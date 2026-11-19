# -*- coding: utf-8 -*-
"""g045_tick_cost_and_hold.py - (가) tick-size cost of the picks and (다) multi-day holding (selection redesign follow-up, research).

Why: g044 (2026-11-19) showed the picks' next-day move after a 09:05 entry is about +0.06% before costs, while one round trip costs
0.23% (fee 0.015% x 2 + tax 0.20%) plus one price tick each way. No exit rule can turn that positive. Two questions follow:
  (가) how much of the loss is the tick size (1 tick = 0.05% ~ 0.25% of the price depending on the KRX price band), and does a
       point-in-time filter "tick / close(D) below X" help?
  (다) do the same picks earn more when held several days, where the fixed round-trip cost is spread over a larger move?

Selection fixed: V0 picks of rows_v0.jsonl.gz (g043). Benchmarks: the tradable universe of the same day (20d turnover >= 5bn,
no structural exclusion) and the same random 20/day sample as g043/g044. First 70% of days = train, the rest = holdout.
Everything below is pre-declared (thresholds / hold lengths written before the results, not tuned afterwards).

(가) intraday, the g044 "base" rule on the next day's 1-minute bars (exit_bars_v0.pkl.gz from g044): picks bucketed by
     tick(close_D)/close_D (known on day D - no look-ahead): <0.10% / 0.10-0.15% / 0.15-0.20% / >=0.20%, plus the filters
     "tick% < 0.15%" and "tick% < 0.10%". Gross = net + the trade's own cost (fees/tax + 2 ticks), to see the move before costs.
(다) daily bars (runtime/data/daily_history): buy the open of D+1 + 1 tick, sell the close of the k-th trading day counted from
     the entry day (k = 1: same day close, 2, 3, 5, 10) - 1 tick, fee/tax as above. Variants with a close-based stop (-5% / -8%:
     sold at the first close at or below the level, inside the window). Only trades whose whole window is in the data count.
     Overlapping holds: the numbers are per trade (mean / median / win / PF), plus the mean of the per-day averages (each pick
     day weighs the same) - no portfolio equity curve.

Output: <runtime>/research/10_selection/cost_hold_v0.txt
Run:    python g045_tick_cost_and_hold.py [--rows rows_v0.jsonl.gz]

Update log:
- [2026-11-19] type=feat owner=claude
    summary: 신규 - V0 선정 종목의 호가 단위 비용(가)과 며칠 보유(다) 비교. 사전 고정 기준, 학습/검증 분리. 연구용 출력만.
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

FEE, TAX = g043.FEE, g043.TAX
tick = g043.krx_tick
TICK_BUCKETS = [(0.0, 0.0010, "<0.10%"), (0.0010, 0.0015, "0.10-0.15%"), (0.0015, 0.0020, "0.15-0.20%"), (0.0020, 1.0, ">=0.20%")]
TICK_FILTERS = [0.0015, 0.0010]
HOLDS = [1, 2, 3, 5, 10]
STOPS = [None, 0.05, 0.08]


def tick_pct(r: dict) -> float | None:
    p = r.get("price")
    return tick(p) / p if p and p > 0 else None


def net_of(buy_px: float, sell_px: float) -> float:
    return (sell_px * (1 - FEE - TAX)) / (buy_px * (1 + FEE)) - 1.0


def hold_trade(code: str, d: str, k: int, stop: float | None) -> float | None:
    """Buy D+1 open + 1 tick, sell the close of the k-th trading day from the entry day - 1 tick (or the first close <= stop)."""
    h = g043._history(code)
    if not h or d not in h["idx"]:
        return None
    i = h["idx"][d] + 1  # entry day
    j = i + k - 1        # exit day
    if j >= len(h["dates"]):
        return None
    o = h["open"][i]
    buy = o + tick(o)
    for x in range(i, j + 1):
        c = h["close"][x]
        if stop is not None and c <= buy * (1 - stop) and x < j:
            return net_of(buy, c - tick(c))
    c = h["close"][j]
    return net_of(buy, c - tick(c))


def st(xs: list[float], days: list[str] | None = None) -> str:
    if not xs:
        return "n0"
    wins = [x for x in xs if x > 0]
    loss = sum(x for x in xs if x <= 0)
    pf = sum(wins) / -loss if loss < 0 else float("inf")
    s = f"n{len(xs):<5} mean {g043._mean(xs) * 100:+.2f}% med {g043._med(xs) * 100:+.2f}% win {len(wins) / len(xs) * 100:3.0f}% PF {pf:.2f}"
    if days:
        by: dict[str, list[float]] = {}
        for x, d in zip(xs, days):
            by.setdefault(d, []).append(x)
        s += f"  day-avg {g043._mean([g043._mean(v) for v in by.values()]) * 100:+.2f}% ({len(by)}d)"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="rows_v0.jsonl.gz")
    args = ap.parse_args()
    tag = Path(args.rows).name.replace("rows_", "").replace(".jsonl.gz", "")
    rows = g043.load_rows(g043.RESEARCH_DIR / args.rows)
    dates = sorted({r["date"] for r in rows})
    n_train = max(1, int(round(len(dates) * g043.TRAIN_FRAC)))
    split = {d: ("train" if i < n_train else "holdout") for i, d in enumerate(dates)}
    pk = [r for r in rows if r["picked"]]
    tr = [r for r in rows if g043.tradable(r)]
    rng = random.Random(20261119)
    rand = []
    for d in dates:
        pool = sorted((r for r in tr if r["date"] == d), key=lambda r: r["code"])
        rand.extend(rng.sample(pool, min(g043.RANDOM_PER_DAY, len(pool))))
    parts = ("train", "holdout", "all")
    inpart = lambda r, p: p == "all" or split[r["date"]] == p  # noqa: E731

    out = [f"# (가) tick cost / (다) multi-day hold - selection fixed: {args.rows}",
           f"dates {len(dates)}: train {dates[0]}~{dates[n_train - 1]} / holdout {dates[n_train]}~{dates[-1]}",
           "costs: fee 0.015% x2 + tax 0.20% + 1 tick each way; thresholds / holds pre-declared (not tuned)", ""]

    # ---------------- (가) intraday base rule by tick bucket ----------------
    bars_path = g043.RESEARCH_DIR / f"exit_bars_{tag}.pkl.gz"
    with gzip.open(bars_path, "rb") as fh:
        bars = pickle.load(fh)
    nd = {d: g043.next_folder(d) for d in dates}
    base = g044.RULES[0]

    def intraday(r):
        b = bars.get((r["code"], nd[r["date"]]))
        t = g044.simulate(b, base) if b else None
        if not t:
            return None
        i0 = next(i for i, x in enumerate(b) if x[0] >= g044.ENTRY_HHMM)
        o0 = b[i0][1]
        entry = o0 + tick(o0)
        cost = FEE * 2 + TAX + 2 * tick(o0) / entry
        return t["net"], t["net"] + cost

    out.append("## (가) intraday g044 base rule (entry 09:05, stop -3.5% / lock +1.0%/-0.4%), picks by tick/close(D)")
    out.append("   net = after costs ; gross = net + the trade's own cost (fees/tax + 2 ticks) ; share = part of the picks")
    for grp_name, grp in (("picked", pk), ("random", rand)):
        sims = {(r["code"], r["date"]): intraday(r) for r in grp}
        for part in parts:
            sel = [r for r in grp if inpart(r, part) and sims[(r["code"], r["date"])] and tick_pct(r) is not None]
            out.append(f"[{grp_name} {part}]  all: net {st([sims[(r['code'], r['date'])][0] for r in sel])}")
            for lo, hi, lab in TICK_BUCKETS:
                b = [r for r in sel if lo <= tick_pct(r) < hi]
                if not b:
                    continue
                net = [sims[(r["code"], r["date"])][0] for r in b]
                gross = [sims[(r["code"], r["date"])][1] for r in b]
                out.append(f"   tick {lab:<11} share {len(b) / len(sel) * 100:3.0f}%  net {st(net)}  | gross mean {g043._mean(gross) * 100:+.2f}%")
            for th in TICK_FILTERS:
                b = [r for r in sel if tick_pct(r) < th]
                out.append(f"   filter tick<{th:.2%}  share {len(b) / len(sel) * 100:3.0f}%  net {st([sims[(r['code'], r['date'])][0] for r in b], [r['date'] for r in b])}")
        out.append("")

    # ---------------- (다) multi-day hold ----------------
    out.append("## (다) hold k trading days: buy D+1 open +1 tick, sell close of the k-th day (k=1 same day) -1 tick")
    groups = {"picked": pk, "picked strict": [r for r in pk if not r["fallback"]], "picked fallback": [r for r in pk if r["fallback"]],
              "random 20/day": rand, "tradable universe": tr}
    for stop in STOPS:
        out.append(f"### close-based stop: {'none' if stop is None else f'-{stop:.0%}'}")
        for k in HOLDS:
            out.append(f"[hold {k}d]")
            for part in parts:
                for gname, grp in groups.items():
                    xs, ds = [], []
                    for r in grp:
                        if not inpart(r, part):
                            continue
                        v = hold_trade(r["code"], r["date"], k, stop)
                        if v is not None:
                            xs.append(v)
                            ds.append(r["date"])
                    out.append(f"   {part:<8} {gname:<18} {st(xs, ds)}")
            out.append("")
    # (가) x (다): tick filter on the multi-day hold of the picks
    out.append("### picks with tick/close(D) < 0.15% (point-in-time), no stop")
    for k in HOLDS:
        for part in parts:
            xs, ds = [], []
            for r in pk:
                if inpart(r, part) and (tick_pct(r) or 1) < 0.0015:
                    v = hold_trade(r["code"], r["date"], k, None)
                    if v is not None:
                        xs.append(v)
                        ds.append(r["date"])
            out.append(f"   hold {k:>2}d {part:<8} {st(xs, ds)}")
    text = "\n".join(out)
    path = g043.RESEARCH_DIR / f"cost_hold_{tag}.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[OUT] {path}")


if __name__ == "__main__":
    main()
