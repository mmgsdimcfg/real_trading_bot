# -*- coding: utf-8 -*-
"""Buy-gate value scan (2026-10-31 사용자 요청: 좋은 후보는 통과, 나쁜 후보는 차단하도록 게이트 재설계).

For every g019 `ma_cross` event (1-min MA5>MA20 golden cross, liquid daily-uptrend stocks, 82 days) evaluate each
live 3-min context gate (r002 HYBRID_3MIN_CONTEXT_GATES, the same eval functions r005 _010~_016 call) INDEPENDENTLY
- no short-circuit, like r002.gate_steps_diagnostic - plus the context score (BUY_SCORE_RULES). 3-min bars are the
completed session buckets up to the event bar (r011 completed_3min; no prior-day warmup, so very early bars differ
from live). Outcomes come from the g019 row: gross g30/g60/geod from the next open and net n<k> of the g019 grid.

  build : python g021_gate_value_scan.py [--dates ...] [--workers 1]
  report: python g021_gate_value_scan.py --report   (per gate: pass vs fail, train vs holdout)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))

import g013_edge_ai_dataset as g013  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
from r011_edge_ai import build_strategy_config, completed_3min, fast_indicators, session_raw_1min  # noqa: E402

SRC_DIR = g013.EDGE_AI_DIR / "entry_scan"
OUT_DIR = g013.EDGE_AI_DIR / "gate_scan"
NET_COMBO = 53        # g019 grid k53 (stop 2 / arm 1 / trail 0.5-0.8 / tp 3-5-8): best ma_cross combo on train
GATES = [g.name for g in r002.HYBRID_3MIN_CONTEXT_GATES]


def gate_flags(m3: pd.DataFrame, now: pd.Timestamp, price: float, config) -> dict | None:
    ctx, _ = r002.build_context_eval(m3, now, price, {}, config)
    if ctx is None:
        return None
    out = {}
    for gate in r002.HYBRID_3MIN_CONTEXT_GATES:
        try:
            passed, ctx, _ = gate.eval_fn(ctx)
            out[f"p_{gate.name}"] = int(bool(passed))
        except Exception:
            out[f"p_{gate.name}"] = -1
    out["score"] = int(sum(rule.eval_fn(ctx) for rule in r002.BUY_SCORE_RULES))
    out["score_ok"] = int(out["score"] >= config.bb_buy_score_threshold)
    cur = m3.iloc[-1]
    for col in ("STOCH_K", "STOCH_D", "ATR", "ADX", "RSI"):
        out[f"m3_{col.lower()}"] = float(cur[col]) if col in m3.columns else np.nan
    out["m3_atr_pct"] = out["m3_atr"] / price * 100 if price else np.nan
    return out


def build_day(task) -> tuple[str, int, float]:
    trade_date, force = task
    t0 = time.time()
    out = OUT_DIR / f"gates_{trade_date}.csv.gz"
    src = SRC_DIR / f"events_{trade_date}.csv.gz"
    if (out.exists() and not force) or not src.exists():
        return trade_date, -1, 0.0
    keep = ["date", "code", "fam", "ts", "bar_i", "price", "g5", "g15", "g30", "g60", "geod", "mfe60", "mae60",
            f"n{NET_COMBO}", "day_ret", "vwap_gap"]
    try:
        ev = pd.read_csv(src, dtype={"code": str, "date": str}, usecols=keep)
    except pd.errors.EmptyDataError:
        ev = pd.DataFrame()
    ev = ev[ev["fam"] == "ma_cross"] if not ev.empty else ev
    config = build_strategy_config(g013.cfg)
    minute_idx = g013._index_dir(g013.DATA_ROOT / trade_date, "_10s.txt")
    rows = []
    for code, grp in (ev.groupby("code") if not ev.empty else []):
        mf = minute_idx.get(code)
        if mf is None:
            continue
        try:
            raw = session_raw_1min(g013.load_minute_bars(mf), pd.Timestamp(trade_date))
            if raw is None:
                continue
            # same no-trade minute fill as g019.prepare_session, so bar_i indexes the same grid
            idx = pd.date_range(raw.index[0], raw.index[-1], freq="1min")
            raw = raw.reindex(idx)
            raw["close"] = raw["close"].ffill()
            for k in ("open", "high", "low"):
                raw[k] = raw[k].fillna(raw["close"])
            raw["volume"] = raw["volume"].fillna(0.0)
            m3_all = completed_3min(fast_indicators(raw))
            if m3_all is None:
                continue
            for ev_row in grp.itertuples(index=False):
                now = raw.index[int(ev_row.bar_i)]
                m3 = m3_all[m3_all.index <= now]
                flags = gate_flags(m3, now, float(ev_row.price), config)
                if flags is None:
                    continue
                rec = ev_row._asdict()
                rec.update(flags)
                rows.append(rec)
        except Exception as exc:  # one bad file must not kill the day
            print(f"  {trade_date} {code} skipped: {exc}", flush=True)
    df = pd.DataFrame(rows)
    tmp = out.with_suffix(".tmp.gz")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)
    return trade_date, len(df), time.time() - t0


def report(args) -> None:
    parts = []
    for f in sorted(OUT_DIR.glob("gates_*.csv.gz")):
        try:
            p = pd.read_csv(f, dtype={"code": str, "date": str})
        except pd.errors.EmptyDataError:
            continue
        if not p.empty:
            parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    dates = sorted(df["date"].unique())
    cut = dates[min(max(int(len(dates) * args.train_frac), 1), len(dates) - 1)]
    df["split"] = np.where(df["date"] < cut, "train", "hold")
    net = f"n{NET_COMBO}"
    cols = [f"p_{g}" for g in GATES] + ["score_ok"]
    lines = [f"# Gate value scan {time.strftime('%Y%m%d_%H%M%S')}", "",
             f"- ma_cross events {len(df):,}, days {len(dates)}, train < {cut}. gross % from next open; "
             f"net = g019 combo k{NET_COMBO} KRW per 300k", "",
             "| gate | split | pass n | fail n | pass g30 | fail g30 | pass geod | fail geod | pass net | fail net | "
             "verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for col in cols:
        verdicts = []
        for split in ("train", "hold"):
            s = df[(df["split"] == split) & (df[col] >= 0)]
            p, f = s[s[col] == 1], s[s[col] == 0]
            d = {k: (p[k].mean(), f[k].mean()) for k in ("g30", "geod", net)}
            # gate "helps" when passing events beat failing ones on net
            verdicts.append(np.sign(d[net][0] - d[net][1]))
            lines.append(f"| {col[2:] if col.startswith('p_') else col} | {split} | {len(p):,} | {len(f):,} "
                         f"| {d['g30'][0]:+.3f} | {d['g30'][1]:+.3f} | {d['geod'][0]:+.3f} | {d['geod'][1]:+.3f} "
                         f"| {d[net][0]:+,.0f} | {d[net][1]:+,.0f} | |")
        tag = ("KEEP (blocks worse)" if all(v > 0 for v in verdicts) else
               "INVERTED (blocks better)" if all(v < 0 for v in verdicts) else "MIXED")
        lines[-1] = lines[-1][:-3] + f" {tag} |"
    # all gates passed vs rest
    allp = (df[[f"p_{g}" for g in GATES]] == 1).all(axis=1) & (df["score_ok"] == 1)
    for split in ("train", "hold"):
        s = df["split"] == split
        lines.append(f"| ALL (gates+score) | {split} | {int((allp & s).sum()):,} | {int((~allp & s).sum()):,} "
                     f"| {df.loc[allp & s, 'g30'].mean():+.3f} | {df.loc[~allp & s, 'g30'].mean():+.3f} "
                     f"| {df.loc[allp & s, 'geod'].mean():+.3f} | {df.loc[~allp & s, 'geod'].mean():+.3f} "
                     f"| {df.loc[allp & s, net].mean():+,.0f} | {df.loc[~allp & s, net].mean():+,.0f} | |")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"report_{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {OUT_DIR / f'report_{stamp}.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args)
        return
    dates = args.dates or sorted(f.name[7:15] for f in SRC_DIR.glob("events_*.csv.gz"))
    t0 = time.time()
    with Pool(args.workers) as pool:
        for d, n, sec in pool.imap_unordered(build_day, [(d, args.force) for d in dates]):
            print(f"{d}: events={n} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
