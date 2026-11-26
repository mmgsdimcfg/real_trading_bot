# -*- coding: utf-8 -*-
"""Study (read-only, no model is published): does a domestic index condition help the Edge AI entries?

Question (nq_futures_indicator_plan_20261126.md 6-3, user 2026-11-26): on 4 days of paper trades "no buy when KODEX 200's
30-minute return <= -0.3%" removed mostly losers. Here the same condition is judged on the whole Edge AI dataset (every
1-minute trigger event with its simulated exit, g013) with the model's out-of-sample probabilities (g014 walk-forward), and
the index returns are also offered to the model as features.

Inputs
- runtime/edge_ai/dataset/<tag>/events_*.csv.gz           (g013; ts = start label of the confirmed trigger bar)
- runtime/cache/kis/min/<date>/069500.json, 229200.json   (r018.minutes; rows [HHMMSS, o, h, l, c, v], start labels)

Time alignment: the decision is made when the trigger bar has closed, so the index bar with the SAME label has closed too.
index return N = close(label ts) / close(label ts - N minutes) - 1, same day only (missing before 09:00 + N).

Usage:  python g053_edge_market_condition.py [--threads 2] [--skip-model]

Update log:
- [2026-11-26] type=feat owner=claude
    summary: 신규 - Edge AI 진입에 국내 지수(069500 / 229200) 30분 수익률 조건을 걸었을 때의 효과(전체 이벤트 / 모델이 고른 이벤트,
      거래일 단위 부트스트랩, 전후반 분할)와 지수 피처를 넣은 모델의 walk-forward 비교.
    impact: sim (읽기 전용, 모델 배포 없음)
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))
import g014_edge_ai_train as g014  # noqa: E402

RUNTIME = Path(r"D:\make_utils\Trading\runtime")
MIN_DIR = RUNTIME / "cache" / "kis" / "min"
INDEX = {"k": "069500", "q": "229200"}        # KODEX 200 / KODEX 코스닥150
HORIZONS = (5, 15, 30)
PRE_REGISTERED = ("k_ret30", -0.3)            # the rule seen on the paper trades - everything else here is exploration


def index_frame(date: str) -> pd.DataFrame | None:
    cols = {}
    for key, code in INDEX.items():
        path = MIN_DIR / date / f"{code}.json"
        if not path.is_file():
            return None
        rows = (json.loads(path.read_text(encoding="utf-8")) or {}).get("rows") or []
        if len(rows) < 300:
            return None
        s = pd.Series({pd.Timestamp(f"{date} {r[0][:2]}:{r[0][2:4]}:00"): float(r[4]) for r in rows if "090000" <= r[0] <= "152000"})
        idx = pd.date_range(f"{date} 09:00", f"{date} 15:20", freq="1min")
        s = s.reindex(idx).ffill()
        first_open = float(next(r[1] for r in rows if r[0] >= "090000"))
        for h in HORIZONS:
            cols[f"{key}_ret{h}"] = (s / s.shift(h) - 1) * 100
        cols[f"{key}_open"] = (s / first_open - 1) * 100
    out = pd.DataFrame(cols)
    out.index.name = "ts"
    return out.reset_index()


def day_boot(df: pd.DataFrame, fn, n: int = 500, seed: int = 7):
    groups = {d: g for d, g in df.groupby("date")}
    days = np.array(list(groups))
    rng = np.random.default_rng(seed)
    vals = [fn(pd.concat([groups[d] for d in rng.choice(days, len(days), replace=True)])) for _ in range(n)]
    return fn(df), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def rule_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n--- {title}: {len(df):,} events / {df['date'].nunique()} days, mean net {df['net_ret'].mean():+.3f}% win {100 * (df['net_ret'] > 0).mean():.1f}%")
    days = sorted(df["date"].unique())
    half = days[len(days) // 2]
    for col in ("k_ret30", "q_ret30", "k_ret15", "k_open"):
        for thr in (-0.3, -0.2, -0.1) if col != "k_open" else (-1.0, -0.5):
            tag = " [pre-registered]" if (col, thr) == PRE_REGISTERED else ""
            m = df[col] <= thr

            def uplift(g, col=col, thr=thr):
                kept = g[~(g[col] <= thr)]
                return kept["net_ret"].mean() - g["net_ret"].mean() if len(kept) else np.nan
            v, lo, hi = day_boot(df, uplift)
            per_day = df.assign(b=m).groupby("date").apply(lambda g: g[~g["b"]]["net_ret"].mean() - g["net_ret"].mean() if (~g["b"]).any() else np.nan)
            better = int((per_day > 1e-9).sum())
            worse = int((per_day < -1e-9).sum())
            h1, h2 = uplift(df[df["date"] < half]), uplift(df[df["date"] >= half])
            print(f" {col} <= {thr:+.1f}%{tag}: blocked {100 * m.mean():5.1f}% ({int(m.sum()):,}; mean {df[m]['net_ret'].mean():+.3f}% win {100 * (df[m]['net_ret'] > 0).mean():.1f}%)"
                  f" | mean net of the rest {v:+.3f}%p vs all [{lo:+.3f}, {hi:+.3f}] | days better/worse {better}/{worse} | halves {h1:+.3f} / {h2:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-model", action="store_true", help="rule tables on all events only (no walk-forward)")
    cli = ap.parse_args()
    try:                                          # below-normal priority: the live bot runs on this machine
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

    meta = json.loads((RUNTIME / "edge_ai" / "current_meta.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(**{**meta["args"], "threads": cli.threads, "no_publish": True})
    ds_dir = RUNTIME / "edge_ai" / "dataset" / meta["dataset"]
    frames = [pd.read_csv(f, dtype={"date": str, "code": str}) for f in sorted(ds_dir.glob("events_*.csv.gz"))]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    target = g014.target_of(args)
    df = df.dropna(subset=[target, "net_ret"]).sort_values(["date", "ts"]).reset_index(drop=True)
    funds = g014.fund_product_codes()
    df = df[~df["code"].astype(str).str.zfill(6).isin(funds)].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    print(f"dataset {ds_dir.name}: {len(df):,} events / {df['date'].nunique()} days")

    idx = [f for f in (index_frame(d) for d in sorted(df["date"].unique())) if f is not None]
    mk = pd.concat(idx, ignore_index=True)
    df = df.merge(mk, on="ts", how="left")
    have = df["k_ret30"].notna()
    print(f"index bars: {len(idx)} of {df['date'].nunique()} days | events with a 30-minute index return: {100 * have.mean():.1f}% "
          f"(the rest: before 09:30 or no index file)")
    base = df[have].copy()
    print(f"same-time corr of the trigger bar: stock ret_30 vs k_ret30 = {base['ret_30'].corr(base['k_ret30']):+.3f}")

    rule_table(base, "ALL trigger events (no model)")
    if cli.skip_model:
        return

    feats = list(g014.FEATURE_NAMES)
    df[feats] = df[feats].astype(np.float32)
    print("\n=== walk-forward, current features (v2)", flush=True)
    oos, folds = g014.walk_forward(df, args)
    pm = g014.promotion(oos, folds, args)
    print(f"OOS auc={pm['auc_oos']:.3f} base={pm['base']['mean_net']:+.3f}% kept={pm['kept']['mean_net']:+.3f}% (n={pm['kept']['n']:,}) win {100 * pm['kept']['win_rate']:.1f}%")
    kept = oos[oos["keep"] & oos["k_ret30"].notna()]
    rule_table(kept, "events the model KEPT out of sample (p >= threshold)")
    top = oos[oos["k_ret30"].notna()].sort_values("p", ascending=False).groupby("date").head(20)
    rule_table(top, "the 20 highest-probability events of each day (out of sample)")

    extra = [c for c in mk.columns if c != "ts"]
    df[extra] = df[extra].astype(np.float32)
    g014.FEATURE_NAMES = tuple(feats + extra)
    print(f"\n=== walk-forward, v2 + index features {extra}", flush=True)
    oos2, folds2 = g014.walk_forward(df, args)
    pm2 = g014.promotion(oos2, folds2, args)
    print(f"OOS auc={pm2['auc_oos']:.3f} base={pm2['base']['mean_net']:+.3f}% kept={pm2['kept']['mean_net']:+.3f}% (n={pm2['kept']['n']:,}) win {100 * pm2['kept']['win_rate']:.1f}%")
    print("\nfold by fold: kept mean net  v2 -> v2+index   (auc v2 -> v2+index)")
    wins = 0
    for a, b in zip(folds, folds2):
        ka, kb = a["all_kept"], b["all_kept"]
        wins += int(kb["mean_net"] > ka["mean_net"])
        print(f"  {a['test']}: {ka['mean_net']:+.3f}% (n={ka['n']:,}) -> {kb['mean_net']:+.3f}% (n={kb['n']:,})   auc {a['auc']:.3f} -> {b['auc']:.3f}")
    print(f"folds where v2+index kept better entries: {wins} / {len(folds)}")
    for name, o in (("v2", oos), ("v2+index", oos2)):
        k = o[o["keep"]]
        v, lo, hi = day_boot(k, lambda g: g["net_ret"].mean())
        t20 = o.sort_values("p", ascending=False).groupby("date").head(20)
        v2, lo2, hi2 = day_boot(t20, lambda g: g["net_ret"].mean())
        print(f"  {name:<9} kept mean net {v:+.3f}% [{lo:+.3f}, {hi:+.3f}] | top-20/day mean net {v2:+.3f}% [{lo2:+.3f}, {hi2:+.3f}] win {100 * (t20['net_ret'] > 0).mean():.1f}%")
    g014.FEATURE_NAMES = tuple(feats)


if __name__ == "__main__":
    main()
