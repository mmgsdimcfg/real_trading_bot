# -*- coding: utf-8 -*-
"""Edge-AI-only trading simulation (2026-11-12 사용자 요청: Edge AI 메타 모델만으로 매수/매도).

Policy "edge only" (what r001 STRATEGY_MODE="edge_only" does live):
  buy   a 1-minute hybrid trigger event (the events the model was trained on) whose model probability >= cut.
        No other rule gate (3-minute context, RSI/MFI/ATR gates, market regime ...) is applied.
  sell  the model's own label definition: +tp% target, -sl% stop, `horizon` minutes, or 15:15 - whichever comes first.
  book  one position per symbol; at most --max-positions at a time; at most --max-entries new entries per day.

Probabilities are the walk-forward OUT-OF-SAMPLE ones written by g014 (runtime/edge_ai/reports/oos_<id>.csv.gz): every
event is scored by a model trained only on earlier days. Outcomes (net %, exit reason, minutes held) come from the g013
dataset of the same tag - costs and 1 tick of slippage per side are already inside net_ret.

  python g029_edge_only_sim.py                    newest OOS file, cuts 0.30..0.60 + the fold thresholds
  python g029_edge_only_sim.py --cuts 0.45 0.5 --max-positions 3 --max-entries 10
Output: runtime/edge_ai/reports/edge_only_<id>.md (+ printed). Nothing here is imported by r003.

Update log
- [2026-11-12] type=feat - first version
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))

import r001_define_config as cfg  # noqa: E402

EDGE_AI_DIR = Path(getattr(cfg, "EDGE_AI_DIR", Path(cfg.RUNTIME_ROOT) / "edge_ai"))
EVENT_COLS = ["date", "code", "ts", "exit_reason", "hold_bars", "mfe_pct", "mae_pct", "tick_pct"]


def load(oos_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """OOS predictions joined with the dataset rows of the same events. -> (frame sorted by time, report meta)."""
    reports = EDGE_AI_DIR / "reports"
    if oos_path is None:
        files = sorted(reports.glob("oos_*.csv.gz"))
        if not files:
            raise SystemExit(f"no OOS file under {reports} - run g014_edge_ai_train.py first")
        oos_path = files[-1]
    oos = pd.read_csv(oos_path, dtype={"code": str, "date": str})
    model_id = oos_path.name[len("oos_"):-len(".csv.gz")]
    meta = {}
    rep = reports / f"report_{model_id}.json"
    if rep.is_file():
        meta = json.loads(rep.read_text(encoding="utf-8"))
    tag = meta.get("dataset")
    ds = EDGE_AI_DIR / "dataset" / str(tag) if tag else None
    if ds is None or not ds.is_dir():
        dirs = sorted(p for p in (EDGE_AI_DIR / "dataset").iterdir() if p.is_dir())
        if not dirs:
            raise SystemExit("no dataset folder - run g013_edge_ai_dataset.py first")
        ds = dirs[-1]
    days = set(oos["date"].unique())
    parts = []
    for f in sorted(ds.glob("events_*.csv.gz")):
        if f.name[len("events_"):len("events_") + 8] in days:
            parts.append(pd.read_csv(f, dtype={"code": str, "date": str}, usecols=lambda c: c in EVENT_COLS))
    ev = pd.concat(parts, ignore_index=True)
    d = oos.merge(ev, on=["date", "code", "ts"], how="left")
    d["ts"] = pd.to_datetime(d["ts"])
    d = d.sort_values(["ts", "code"], kind="stable").reset_index(drop=True)
    if not meta.get("label"):
        # the report of older trainers has no label block: the model's own meta file has it
        for cand in (EDGE_AI_DIR / "models" / f"meta_{model_id}.json", EDGE_AI_DIR / "current_meta.json"):
            if cand.is_file():
                lab = json.loads(cand.read_text(encoding="utf-8")).get("label")
                if lab:
                    meta["label"] = lab
                    break
    meta = dict(meta, oos_file=str(oos_path), dataset_dir=str(ds), model_id=model_id)
    return d, meta


def simulate(d: pd.DataFrame, cut: float | None, max_positions: int, max_entries: int, use_fold_threshold: bool = False) -> pd.DataFrame:
    """Trades the policy would have taken, in time order. cut=None with use_fold_threshold -> each fold's own threshold
    (chosen on its inner validation days, i.e. without looking at the test days)."""
    sel = d[d["p"] >= d["threshold"]] if use_fold_threshold else d[d["p"] >= float(cut)]
    taken = []
    for _date, day in sel.groupby("date", sort=True):
        open_until: dict[str, pd.Timestamp] = {}      # code -> time its position closes
        entries = 0
        for row in day.itertuples(index=False):
            now = row.ts
            # positions whose exit time has passed are free again
            for code in [c for c, t in open_until.items() if t <= now]:
                del open_until[code]
            if row.code in open_until or len(open_until) >= max_positions or entries >= max_entries:
                continue
            hold = float(row.hold_bars) if row.hold_bars == row.hold_bars else 1.0
            open_until[row.code] = pd.Timestamp(now) + pd.Timedelta(float(hold) + 1.0, "min")   # entry is the next bar's open
            entries += 1
            taken.append(row)
    return pd.DataFrame(taken, columns=d.columns) if taken else d.iloc[0:0]


def stats(tr: pd.DataFrame, n_days: int) -> dict:
    if tr.empty:
        return {"n": 0}
    by = tr.groupby("date")["net_ret"].sum()
    reason = tr["exit_reason"].value_counts(normalize=True)
    wins, losses = tr.loc[tr["net_ret"] > 0, "net_ret"], tr.loc[tr["net_ret"] <= 0, "net_ret"]
    return {
        "n": int(len(tr)), "per_day": len(tr) / n_days, "win": float((tr["net_ret"] > 0).mean()),
        "mean": float(tr["net_ret"].mean()), "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "tp": float(reason.get("tp", 0.0)), "sl": float(reason.get("sl", 0.0)),
        "day_sum": float(by.sum() / n_days), "days_pos": int((by > 0).sum()), "days_traded": int(len(by)),
        "worst_day": float(by.min()), "best_day": float(by.max()),
    }


def breakeven_win_rate(tr: pd.DataFrame) -> float:
    """Win rate at which the mean is zero with this sample's average win and loss."""
    wins, losses = tr.loc[tr["net_ret"] > 0, "net_ret"], tr.loc[tr["net_ret"] <= 0, "net_ret"]
    if not len(wins) or not len(losses):
        return float("nan")
    return float(-losses.mean() / (wins.mean() - losses.mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oos", type=Path, default=None)
    ap.add_argument("--cuts", type=float, nargs="*", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    ap.add_argument("--max-positions", type=int, default=int(getattr(cfg, "MAX_CONCURRENT_POSITIONS", 3) or 3))
    ap.add_argument("--max-entries", type=int, default=20)
    args = ap.parse_args()
    d, meta = load(args.oos)
    n_days = d["date"].nunique()
    label = meta.get("label") or {}
    lines = [
        f"# Edge AI only - model {meta.get('model_id')} ({d['date'].min()}~{d['date'].max()}, {n_days} out-of-sample days)", "",
        f"- events: {len(d):,} trigger events of the liquid universe ({len(d) / n_days:,.0f}/day), every one scored by a model "
        "trained on earlier days only",
        f"- sell rule = the label: +{label.get('tp', '?')}% / -{label.get('sl', '?')}% / {label.get('horizon', '?')} min / 15:15, "
        "costs and 1 tick slippage per side included",
        f"- book: one position per symbol, at most {args.max_positions} at a time, at most {args.max_entries} entries a day",
        f"- all events without any cut: win {(d['net_ret'] > 0).mean() * 100:.1f}%, mean {d['net_ret'].mean():+.3f}%", "",
        "| cut | trades | /day | win | mean net | avg win | avg loss | break-even win | TP | SL | day sum | days + | worst day |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [(f"p >= {c:.2f}", simulate(d, c, args.max_positions, args.max_entries)) for c in args.cuts]
    rows.append(("fold threshold", simulate(d, None, args.max_positions, args.max_entries, use_fold_threshold=True)))
    for name, tr in rows:
        s = stats(tr, n_days)
        if s["n"] < 20:
            lines.append(f"| {name} | {s['n']} | | | | | | | | | | | |")
            continue
        lines.append(
            f"| {name} | {s['n']:,} | {s['per_day']:.1f} | {s['win'] * 100:.1f}% | {s['mean']:+.3f}% | {s['avg_win']:+.2f}% | "
            f"{s['avg_loss']:+.2f}% | {breakeven_win_rate(tr) * 100:.1f}% | {s['tp'] * 100:.0f}% | {s['sl'] * 100:.0f}% | "
            f"{s['day_sum']:+.2f}% | {s['days_pos']}/{s['days_traded']} | {s['worst_day']:+.1f}% |")
    lines += ["", "day sum = sum of the trades' net % per day, averaged over all days (each trade = one equal slot).", ""]
    # probability deciles without the book limits: is the ranking real?
    lines += ["## Ranking check (all events, no book limits)", "", "| probability decile | events | p mean | win | mean net |", "|---|---|---|---|---|"]
    dec = pd.qcut(d["p"], 10, duplicates="drop")
    for k, v in d.groupby(dec, observed=True):
        lines.append(f"| {k} | {len(v):,} | {v['p'].mean():.3f} | {(v['net_ret'] > 0).mean() * 100:.1f}% | {v['net_ret'].mean():+.3f}% |")
    text = "\n".join(lines) + "\n"
    out = EDGE_AI_DIR / "reports" / f"edge_only_{meta.get('model_id')}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("->", out)


if __name__ == "__main__":
    main()
