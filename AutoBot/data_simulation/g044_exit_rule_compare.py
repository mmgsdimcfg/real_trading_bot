# -*- coding: utf-8 -*-
"""g044_exit_rule_compare.py - same picks, different exit rules (selection redesign follow-up, research only).

Why: the g043 replays (2026-11-19) showed every selection variant (V0, V1, V2, single filter releases) loses -0.27~-0.40% per trade
next day with a 70% win rate and PF ~0.65: the exit structure (hard stop -3.5% vs +1.0% profit lock with 0.4% giveback) decides
more than the selection. This keeps the selection fixed (V0 picks of rows_v0.jsonl.gz, plus the same random 20/day tradable
baseline as g043) and replays a FIXED, pre-declared list of exit rules on the next day's raw 1-minute bars.

Not an optimiser: the rule list below is written before looking at the results and is not tuned afterwards. Rules are compared on
the train days (first 70%) and checked on the holdout days (last 30%); a rule only counts as better when it is better in both.
Live r003 exits (EXIT_ENGINE_MODE live: regime, staged take-profit, dynamic trailing, flexible stop) are NOT modelled - the
baseline "base" is the same simplified rule as g043 (hard stop -3.5% after 4 min, lock +1.0% / -0.4%, eod 15:15).

Fill model (same as g043): entry = 09:05 bar open + 1 tick. Path inside a bar: bullish o->l->h->c, bearish o->h->l->c (the
conservative order). Market exits (stop / trailing / eod) get 1 tick slippage, a level gapped through at the open fills at the
open. Take-profit is a resting limit: filled at the level only when the bar trades ABOVE it (high > level), no slippage.
Costs: fee 0.015% per side, sell tax 0.20%.

Output: <runtime>/research/10_selection/exit_compare_v0.txt ; bars cache exit_bars_v0.pkl.gz (built once, --rebuild to refresh)
Run:    python g044_exit_rule_compare.py [--rows rows_v0.jsonl.gz] [--workers 14] [--rebuild]

Update log:
- [2026-11-19] type=feat owner=claude
    summary: 신규 - 선정 고정(V0) + 청산 규칙 비교. 고정 규칙 목록, 학습/검증 분리. 연구용 출력만, 실거래 코드 미변경.
    impact: research tooling (runtime/research/10_selection)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import g043_selection_replay as g043

ENTRY_HHMM = g043.ENTRY_HHMM
FEE, TAX = g043.FEE, g043.TAX
tick = g043.krx_tick


@dataclass(frozen=True)
class Rule:
    name: str
    stop: float | None = 0.035          # hard stop below entry (None = no stop)
    stop_delay: int = 4                  # minutes after entry before the stop is active
    arm: float | None = 0.010            # trailing (profit lock) arms when the peak reaches entry*(1+arm); None = no trailing
    giveback: float = 0.004              # trailing exit at peak*(1-giveback)
    tp: float | None = None              # take-profit limit at entry*(1+tp) for tp_frac of the position
    tp_frac: float = 1.0
    breakeven_after_tp: bool = False     # after a partial take-profit the rest's stop moves to the entry price
    eod: str = "15:15"


# Pre-declared list (written before the results; not tuned afterwards)
RULES: list[Rule] = [
    Rule("base: stop3.5 lock1.0/0.4"),
    # stop width, same lock
    Rule("stop2.0 lock1.0/0.4", stop=0.020),
    Rule("stop2.5 lock1.0/0.4", stop=0.025),
    Rule("stop5.0 lock1.0/0.4", stop=0.050),
    Rule("nostop lock1.0/0.4", stop=None),
    # lock / trailing, same stop
    Rule("stop3.5 lock1.0/0.8", giveback=0.008),
    Rule("stop3.5 lock1.5/0.5", arm=0.015, giveback=0.005),
    Rule("stop3.5 lock1.5/1.0", arm=0.015, giveback=0.010),
    Rule("stop3.5 lock2.0/1.0", arm=0.020, giveback=0.010),
    Rule("stop3.5 lock2.0/1.2", arm=0.020, giveback=0.012),
    Rule("stop3.5 lock3.0/1.5", arm=0.030, giveback=0.015),
    # fixed take-profit, no trailing
    Rule("stop3.5 tp1.5", arm=None, tp=0.015),
    Rule("stop3.5 tp2.0", arm=None, tp=0.020),
    Rule("stop3.5 tp2.5", arm=None, tp=0.025),
    Rule("stop3.5 tp3.0", arm=None, tp=0.030),
    Rule("stop2.0 tp2.0", stop=0.020, arm=None, tp=0.020),
    Rule("stop2.0 tp3.0", stop=0.020, arm=None, tp=0.030),
    # half at a take-profit, the rest trailing (stop moved to entry)
    Rule("stop3.5 half tp1.0 + rest BE lock2.0/1.0", arm=0.020, giveback=0.010, tp=0.010, tp_frac=0.5, breakeven_after_tp=True),
    Rule("stop3.5 half tp1.5 + rest BE lock3.0/1.5", arm=0.030, giveback=0.015, tp=0.015, tp_frac=0.5, breakeven_after_tp=True),
    Rule("stop2.0 half tp1.5 + rest BE lock3.0/1.5", stop=0.020, arm=0.030, giveback=0.015, tp=0.015, tp_frac=0.5, breakeven_after_tp=True),
    # references
    Rule("stop3.5 only -> eod", arm=None),
    Rule("hold -> eod (no stop, no lock)", stop=None, arm=None),
    Rule("base, exit 14:00", eod="14:00"),
]


def simulate(bars, rule: Rule) -> dict | None:
    """Net return of one trade (position-weighted when part is sold at the take-profit)."""
    i0 = next((i for i, b in enumerate(bars) if b[0] >= ENTRY_HHMM), None)
    if i0 is None:
        return None
    o0 = bars[i0][1]
    entry = o0 + tick(o0)
    stop = entry * (1 - rule.stop) if rule.stop is not None else -1.0
    tp_px = entry * (1 + rule.tp) if rule.tp is not None else None
    left = 1.0
    proceeds = 0.0  # sum of frac * net-of-cost exit price
    peak, lo = entry, entry
    why = None
    be_on = False  # breakeven stop after a partial take-profit: active at once (no stop_delay)

    def sell(frac: float, px: float) -> None:
        nonlocal left, proceeds
        proceeds += frac * px * (1 - FEE - TAX)
        left -= frac

    for k in range(i0, len(bars)):
        t, o, h, l, c = bars[k]
        if t >= rule.eod:
            sell(left, o - tick(o))
            why = why or "eod"
            break
        stop_on = be_on or k - i0 >= rule.stop_delay
        path = (o, l, h, c) if c >= o else (o, h, l, c)
        prev = o
        done = False
        for j, p in enumerate(path):
            if j == 0 or p < prev:  # moving down (or the open): stop / trailing
                armed = rule.arm is not None and peak >= entry * (1 + rule.arm)
                lock = peak * (1 - rule.giveback) if armed else -1.0
                lvl = max(stop if stop_on else -1.0, lock)
                if lvl > 0 and p <= lvl:
                    fill = p if j == 0 else lvl
                    sell(left, fill - tick(fill))
                    why = (why + "+" if why else "") + ("lock" if armed and lvl == lock else ("be" if stop >= entry else "stop"))
                    done = True
                    break
            if tp_px is not None and p > tp_px and (j > 0 and p > prev or j == 0):  # moving up through the limit
                fill = tp_px if j > 0 else max(tp_px, p)  # an open above the limit fills at the open
                sell(min(rule.tp_frac, left), fill)
                tp_px = None
                why = "tp"
                if left <= 1e-9:
                    done = True
                    break
                if rule.breakeven_after_tp:
                    stop = max(stop, entry)
                    be_on = stop_on = True
            peak = max(peak, p)
            lo = min(lo, p)
            prev = p
        if done:
            break
    else:
        c = bars[-1][4]
        sell(left, c - tick(c))
        why = (why + "+" if why else "") + "eod"
    net = proceeds / (entry * (1 + FEE)) - 1.0
    return {"net": net, "why": why, "mfe": peak / entry - 1.0, "mae": lo / entry - 1.0}


def _load(key):
    code, nd = key
    return key, g043.minute_bars(code, nd)


def load_bars(keys: list[tuple[str, str]], path: Path, workers: int, rebuild: bool) -> dict:
    have: dict = {}
    if path.exists() and not rebuild:
        with gzip.open(path, "rb") as fh:
            have = pickle.load(fh)
    need = [k for k in keys if k not in have]
    if need:
        print(f"[bars] reading {len(need)} next-day minute files ({workers} workers)", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (k, b) in enumerate(ex.map(_load, need, chunksize=32), 1):
                have[k] = b
                if i % 500 == 0:
                    print(f"[bars] {i}/{len(need)}", flush=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(have, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return have


def stats(trades: list[dict], days: list[str]) -> dict:
    nets = [t["net"] for t in trades]
    if not nets:
        return {"n": 0}
    wins = [x for x in nets if x > 0]
    loss = sum(x for x in nets if x <= 0)
    by_day: dict[str, list[float]] = {}
    for t, d in zip(trades, days):
        by_day.setdefault(d, []).append(t["net"])
    eq = peak = mdd = 0.0
    for d in sorted(by_day):
        eq += g043._mean(by_day[d])
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    return {"n": len(nets), "net": g043._mean(nets) * 100, "med": g043._med(nets) * 100, "win": len(wins) / len(nets) * 100,
            "pf": (sum(wins) / -loss) if loss < 0 else float("inf"), "sum": eq * 100, "mdd": mdd * 100,
            "why": Counter(t["why"] for t in trades)}


def fmt(s: dict) -> str:
    if not s.get("n"):
        return "n0"
    return f"n{s['n']:<5} net {s['net']:+.3f}% win {s['win']:3.0f}% PF {s['pf']:.2f} daily-sum {s['sum']:+6.1f}% MDD {s['mdd']:6.1f}%p"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="rows_v0.jsonl.gz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    rows = g043.load_rows(g043.RESEARCH_DIR / args.rows)
    dates = sorted({r["date"] for r in rows})
    n_train = max(1, int(round(len(dates) * g043.TRAIN_FRAC)))
    split = {d: ("train" if i < n_train else "holdout") for i, d in enumerate(dates)}
    pk = [r for r in rows if r["picked"]]
    tr = [r for r in rows if g043.tradable(r)]
    rng = random.Random(20261119)  # the same random baseline as g043
    rand = []
    for d in dates:
        pool = sorted((r for r in tr if r["date"] == d), key=lambda r: r["code"])
        rand.extend(rng.sample(pool, min(g043.RANDOM_PER_DAY, len(pool))))
    groups = {"picked": pk, "random": rand}

    nd_of = {d: g043.next_folder(d) for d in dates}
    keys = sorted({(r["code"], nd_of[r["date"]]) for g in groups.values() for r in g if nd_of[r["date"]]})
    tag = Path(args.rows).name.replace("rows_", "").replace(".jsonl.gz", "")
    bars = load_bars(keys, g043.RESEARCH_DIR / f"exit_bars_{tag}.pkl.gz", args.workers, args.rebuild)

    res: dict[str, dict[str, dict[str, dict]]] = {}
    for rule in RULES:
        res[rule.name] = {}
        for gname, grp in groups.items():
            per = {"train": ([], []), "holdout": ([], []), "all": ([], [])}
            for r in grp:
                b = bars.get((r["code"], nd_of[r["date"]]))
                t = simulate(b, rule) if b else None
                if not t:
                    continue
                for part in (split[r["date"]], "all"):
                    per[part][0].append(t)
                    per[part][1].append(r["date"])
            res[rule.name][gname] = {p: stats(*v) for p, v in per.items()}

    base = res[RULES[0].name]["picked"]
    out = [f"# exit rule comparison - selection fixed: {args.rows} (picked) + random 20/day tradable",
           f"dates {len(dates)}: train {dates[0]}~{dates[n_train - 1]} / holdout {dates[n_train]}~{dates[-1]}",
           "entry 09:05 open +1 tick; costs fee 0.015%x2 + tax 0.20%, market exits 1 tick slippage; rules pre-declared (not tuned)",
           "better = net per trade above base in BOTH train and holdout", ""]
    out.append(f"{'rule':<44} {'train net/PF':>16} {'holdout net/PF':>16} {'all net / PF / MDD':>26}  both-better")
    for rule in RULES:
        s = res[rule.name]["picked"]
        both = s["train"]["net"] > base["train"]["net"] and s["holdout"]["net"] > base["holdout"]["net"]
        out.append(f"{rule.name:<44} {s['train']['net']:+.3f}%/{s['train']['pf']:.2f}   {s['holdout']['net']:+.3f}%/{s['holdout']['pf']:.2f}"
                   f"   {s['all']['net']:+.3f}% / {s['all']['pf']:.2f} / {s['all']['mdd']:6.1f}%p  {'YES' if both else ''}")
    out.append("")
    for rule in RULES:
        out.append(f"## {rule.name}")
        for gname in groups:
            for part in ("train", "holdout", "all"):
                s = res[rule.name][gname][part]
                ex = " exits " + "/".join(f"{k}{v}" for k, v in sorted(s.get("why", {}).items())) if part == "all" else ""
                out.append(f"  {gname:<7} {part:<8} {fmt(s)}{ex}")
        out.append("")
    text = "\n".join(out)
    path = g043.RESEARCH_DIR / f"exit_compare_{tag}.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[OUT] {path}")


if __name__ == "__main__":
    main()
