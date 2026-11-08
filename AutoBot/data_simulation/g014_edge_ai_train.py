# -*- coding: utf-8 -*-
"""Edge AI trainer: walk-forward validation, threshold selection, promotion, atomic model publish.

Reads g013 events (runtime/edge_ai/dataset/<tag>/events_*.csv.gz) and trains a small LightGBM binary
classifier P(net_ret > 0 | features at the primary trigger).

Validation (no look-ahead)
- Folds are split by trade DATE: train on dates before the test block, skip --embargo-days, test the next
  --test-days, roll forward. All labels end inside the same session (15:15 time barrier), so whole-day
  purging removes every train/test label overlap.
- Inside each training window the last --val-frac of dates is an inner validation split used only for
  early stopping and for choosing the probability threshold (maximise mean net return per trade with at
  least --min-keep of the events kept). The test block never influences the threshold.

Promotion (r011 "auto" mode enforces the gate only for promoted models)
- Measured on the pooled out-of-sample predictions of the primary-signal-like subset (m3_ctx_pass == 1; falls back to
  all events when that subset is smaller than --min-ctx-events):
  mean net return uplift >= --min-uplift (%p), win-rate uplift > 0, kept trades >= --min-kept, and the
  filter improved mean net return in >= --min-fold-ratio of the folds.

Final model = refit on the most recent training window with the inner-validation iteration count and
threshold; written to an immutable runtime/edge_ai/models/model_<id>.txt, then published by atomically replacing
current_meta.json whose "model_file" points at it. The meta file alone selects the generation, so a reader (or a
publish interrupted at any point) never pairs a new booster with an old threshold/promoted flag.

Update log:
- [2026-10-26] type=feat owner=claude
    summary: 신규 - Edge AI(LightGBM) walk-forward 학습/임계값 선정/승격/모델 배포.
    impact: sim (live는 r011이 current_* 파일을 읽을 때만 반영)
    compatibility: backward-compatible
- [2026-10-27] type=fix owner=claude
    summary: Codex 1차 리뷰 ③ 반영 - current_model.txt 복사 제거, meta가 버전별 모델 파일을 가리키는 단일 원자 교체.
    impact: sim / live 모델 로드 경로(r011 v2와 함께 배포)
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))

import r001_define_config as cfg  # noqa: E402
from r011_edge_ai import FEATURE_NAMES, FEATURE_VERSION  # noqa: E402

EDGE_AI_DIR = Path(getattr(cfg, "EDGE_AI_DIR", Path(cfg.RUNTIME_ROOT) / "edge_ai"))

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def trade_stats(net: np.ndarray) -> dict:
    net = np.asarray(net, dtype=float)
    if len(net) == 0:
        return {"n": 0, "win_rate": float("nan"), "mean_net": float("nan"), "sum_net": 0.0, "pf": float("nan")}
    gains, losses = net[net > 0].sum(), -net[net <= 0].sum()
    return {"n": int(len(net)), "win_rate": float((net > 0).mean()), "mean_net": float(net.mean()),
            "sum_net": float(net.sum()), "pf": float(gains / losses) if losses > 0 else float("inf")}


def choose_threshold(p: np.ndarray, net: np.ndarray, min_keep: float, min_n: int) -> float:
    """Probability cut maximising mean net return/trade on the inner validation set (0.0 = keep all)."""
    best_t, best_mean = 0.0, float(np.mean(net)) if len(net) else 0.0
    for q in np.arange(0.05, 1.0 - min_keep + 1e-9, 0.05):
        t = float(np.quantile(p, q))
        kept = net[p >= t]
        if len(kept) < max(min_n, int(min_keep * len(net))):
            continue
        m = float(kept.mean())
        if m > best_mean:
            best_t, best_mean = t, m
    return best_t


def recency_weights(dates: pd.Series, halflife_days: float) -> np.ndarray | None:
    if not halflife_days or halflife_days <= 0:
        return None
    order = {d: i for i, d in enumerate(sorted(dates.unique()))}
    age = (len(order) - 1) - dates.map(order).to_numpy()
    return np.power(0.5, age / float(halflife_days))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def fit_with_inner_val(df: pd.DataFrame, args, num_threads: int):
    """Early-stopped fit on the first (1-val_frac) dates, threshold from the last val_frac dates,
    then refit on all dates of df with the found iteration count. Returns (booster, best_iter, threshold)."""
    import lightgbm as lgb

    dates = sorted(df["date"].unique())
    n_val = max(1, int(round(len(dates) * args.val_frac)))
    tr = df[df["date"].isin(dates[:-n_val])]
    va = df[df["date"].isin(dates[-n_val:])]
    params = {**LGB_PARAMS, "num_threads": num_threads}
    dtr = lgb.Dataset(tr[list(FEATURE_NAMES)].to_numpy(np.float32), tr["y"].to_numpy(),
                      weight=recency_weights(tr["date"], args.halflife), free_raw_data=True)
    dva = lgb.Dataset(va[list(FEATURE_NAMES)].to_numpy(np.float32), va["y"].to_numpy(), reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=args.max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(args.early_stop, verbose=False)])
    best_iter = max(20, int(booster.best_iteration or args.max_rounds))
    p_va = booster.predict(va[list(FEATURE_NAMES)].to_numpy(np.float32), num_iteration=best_iter)
    threshold = choose_threshold(p_va, va["net_ret"].to_numpy(), args.min_keep, args.min_kept)
    dall = lgb.Dataset(df[list(FEATURE_NAMES)].to_numpy(np.float32), df["y"].to_numpy(),
                       weight=recency_weights(df["date"], args.halflife))
    final = lgb.train(params, dall, num_boost_round=best_iter)
    return final, best_iter, threshold


def walk_forward(df: pd.DataFrame, args) -> tuple[pd.DataFrame, list[dict]]:
    dates = sorted(df["date"].unique())
    preds, folds = [], []
    start = args.min_train_days
    while start + args.embargo_days < len(dates):
        test_dates = dates[start + args.embargo_days: start + args.embargo_days + args.test_days]
        if not test_dates:
            break
        train_dates = dates[:start]
        if args.train_window > 0:
            train_dates = train_dates[-args.train_window:]
        train = df[df["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)].copy()
        booster, best_iter, threshold = fit_with_inner_val(train, args, args.threads)
        test["p"] = booster.predict(test[list(FEATURE_NAMES)].to_numpy(np.float32))
        test["threshold"] = threshold
        test["keep"] = test["p"] >= threshold
        preds.append(test)
        fold = {"train": f"{train_dates[0]}~{train_dates[-1]}", "test": f"{test_dates[0]}~{test_dates[-1]}",
                "best_iter": best_iter, "threshold": threshold, "auc": auc(test["y"], test["p"])}
        for name, sub in (("all", test), ("ctx", test[test["m3_ctx_pass"] == 1])):
            fold[f"{name}_base"] = trade_stats(sub["net_ret"])
            fold[f"{name}_kept"] = trade_stats(sub.loc[sub["keep"], "net_ret"])
        folds.append(fold)
        print(f"  fold test {fold['test']}: auc={fold['auc']:.3f} thr={threshold:.3f} "
              f"all {fold['all_base']['mean_net']:+.3f}%->{fold['all_kept']['mean_net']:+.3f}% "
              f"(kept {fold['all_kept']['n']}/{fold['all_base']['n']})", flush=True)
        start += args.test_days
    return (pd.concat(preds) if preds else pd.DataFrame()), folds


def promotion(oos: pd.DataFrame, folds: list[dict], args) -> dict:
    ctx = oos[oos["m3_ctx_pass"] == 1]
    scope = "ctx" if len(ctx) >= args.min_ctx_events else "all"
    sub = ctx if scope == "ctx" else oos
    base = trade_stats(sub["net_ret"])
    kept = trade_stats(sub.loc[sub["keep"], "net_ret"])
    improved = [f for f in folds if f[f"{scope}_kept"]["n"] > 0 and
                f[f"{scope}_kept"]["mean_net"] > f[f"{scope}_base"]["mean_net"]]
    fold_ratio = len(improved) / len(folds) if folds else 0.0
    checks = {
        "uplift_mean_net": (kept["mean_net"] - base["mean_net"]) if kept["n"] else float("nan"),
        "uplift_win_rate": (kept["win_rate"] - base["win_rate"]) if kept["n"] else float("nan"),
        "kept_n": kept["n"],
        "fold_improved_ratio": fold_ratio,
    }
    ok = (kept["n"] >= args.min_kept
          and checks["uplift_mean_net"] >= args.min_uplift
          and checks["uplift_win_rate"] > 0
          and fold_ratio >= args.min_fold_ratio)
    return {"promoted": bool(ok), "scope": scope, "base": base, "kept": kept, "checks": checks,
            "auc_oos": auc(oos["y"], oos["p"]), "n_folds": len(folds)}


def reliability(oos: pd.DataFrame, bins: int = 10) -> list[dict]:
    q = pd.qcut(oos["p"], bins, duplicates="drop")
    g = oos.groupby(q, observed=True)
    return [{"p_range": str(k), "n": int(len(v)), "p_mean": float(v["p"].mean()), "win_rate": float(v["y"].mean()),
             "mean_net": float(v["net_ret"].mean())} for k, v in g]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def publish(booster, meta: dict, model_dir: Path) -> Path:
    """Immutable models/model_<id>.txt first, then ONE atomic replace of current_meta.json pointing at it."""
    models = model_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    model_file = models / f"model_{meta['model_id']}.txt"
    tmp_model = model_file.with_name(model_file.name + ".tmp")
    booster.save_model(str(tmp_model))
    os.replace(tmp_model, model_file)
    full = {**meta, "model_file": f"models/{model_file.name}"}
    text = json.dumps(full, indent=2, default=str)
    _write_atomic(models / f"meta_{meta['model_id']}.json", text)
    _write_atomic(model_dir / "current_meta.json", text)
    return model_file


def live_feedback(tag: str) -> dict | None:
    """Summary of g013 --live output: real buy signals scored live (shadow) and their labeled outcomes,
    plus live-vs-offline feature differences (train/serve skew)."""
    files = sorted((EDGE_AI_DIR / "live_labeled" / tag).glob("live_*.csv.gz"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, dtype={"date": str, "code": str}) for f in files], ignore_index=True)
    lab = df.dropna(subset=["net_ret", "prob"]) if "net_ret" in df.columns else df.iloc[0:0]
    out: dict = {"n_signals": int(len(df)), "n_labeled": int(len(lab)), "days": int(df["date"].nunique())}
    if len(lab):
        keep = lab["prob"] >= lab["threshold"].fillna(0.0)
        out["all"] = trade_stats(lab["net_ret"])
        out["model_pass"] = trade_stats(lab.loc[keep, "net_ret"])
        out["model_block"] = trade_stats(lab.loc[~keep, "net_ret"])
        out["auc"] = auc(lab["y"].astype(int), lab["prob"]) if lab["y"].nunique() == 2 else float("nan")
    drift = []
    for k in FEATURE_NAMES:
        a, b = f"live_{k}", f"off_{k}"
        if a in df.columns and b in df.columns:
            both = df[[a, b]].dropna()
            if len(both):
                scale = float(np.nanstd(both[b])) or 1.0
                drift.append((k, float((both[a] - both[b]).abs().mean() / scale), int(len(both))))
    out["drift"] = sorted(drift, key=lambda x: -x[1])[:15]
    return out


def write_report(path: Path, meta: dict, folds: list[dict], rel: list[dict]) -> None:
    pm = meta["promotion"]
    lines = [f"# Edge AI training report {meta['model_id']}", "",
             f"- dataset: `{meta['dataset']}` ({meta['n_events']:,} events, {meta['n_days']} days {meta['date_range']})",
             f"- label: {meta['label']}",
             f"- OOS AUC: {pm['auc_oos']:.3f} over {pm['n_folds']} walk-forward folds",
             f"- promoted: **{pm['promoted']}** (scope={pm['scope']}) checks={json.dumps(pm['checks'], default=str)}",
             f"- final threshold: {meta['threshold']:.4f}, iterations: {meta['best_iter']}", "",
             "| scope | n | win rate | mean net % | PF |", "|---|---|---|---|---|"]
    for name in ("base", "kept"):
        s = pm[name]
        lines.append(f"| {pm['scope']} {name} | {s['n']} | {s['win_rate']:.3f} | {s['mean_net']:+.4f} | {s['pf']:.3f} |")
    lines += ["", "## Folds", "", "| test | AUC | thr | all base -> kept (mean net %) | ctx base -> kept |", "|---|---|---|---|---|"]
    for f in folds:
        lines.append(
            f"| {f['test']} | {f['auc']:.3f} | {f['threshold']:.3f} | "
            f"{f['all_base']['mean_net']:+.3f} ({f['all_base']['n']}) -> {f['all_kept']['mean_net']:+.3f} ({f['all_kept']['n']}) | "
            f"{f['ctx_base']['mean_net']:+.3f} ({f['ctx_base']['n']}) -> {f['ctx_kept']['mean_net']:+.3f} ({f['ctx_kept']['n']}) |")
    lines += ["", "## Reliability (OOS)", "", "| p range | n | p mean | win rate | mean net % |", "|---|---|---|---|---|"]
    for r in rel:
        lines.append(f"| {r['p_range']} | {r['n']} | {r['p_mean']:.3f} | {r['win_rate']:.3f} | {r['mean_net']:+.3f} |")
    fb = meta.get("live_feedback")
    if fb:
        lines += ["", "## Live shadow feedback (real buy signals)", "",
                  f"- signals {fb['n_signals']} / labeled {fb['n_labeled']} over {fb['days']} day(s), AUC {fb.get('auc', float('nan')):.3f}",
                  "", "| group | n | win rate | mean net % | PF |", "|---|---|---|---|---|"]
        for name in ("all", "model_pass", "model_block"):
            s_ = fb.get(name)
            if s_:
                lines.append(f"| {name} | {s_['n']} | {s_['win_rate']:.3f} | {s_['mean_net']:+.4f} | {s_['pf']:.3f} |")
        lines += ["", "Feature drift live vs offline (mean |diff| / offline std, top):", ""]
        lines += [f"- {k}: {d:.3f} (n={n})" for k, d, n in fb["drift"]]
    lines += ["", "## Top features (gain)", ""]
    for name, gain in meta["feature_importance"][:20]:
        lines.append(f"- {name}: {gain:,.0f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward train + publish the Edge AI meta-label model")
    p.add_argument("--tag", default=None, help="dataset tag (default: newest folder under edge_ai/dataset)")
    p.add_argument("--min-train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=5)
    p.add_argument("--embargo-days", type=int, default=1)
    p.add_argument("--train-window", type=int, default=60, help="rolling train window in days (0 = expanding)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--halflife", type=float, default=0.0, help="recency weight half-life in days (0 = off)")
    p.add_argument("--max-rounds", type=int, default=600)
    p.add_argument("--early-stop", type=int, default=50)
    p.add_argument("--min-keep", type=float, default=0.2, help="threshold must keep >= this fraction")
    p.add_argument("--min-kept", type=int, default=50)
    p.add_argument("--min-ctx-events", type=int, default=300)
    p.add_argument("--min-uplift", type=float, default=0.05, help="required mean net uplift in %%p")
    p.add_argument("--min-fold-ratio", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--no-publish", action="store_true", help="evaluate only, keep current model")
    args = p.parse_args()

    root = EDGE_AI_DIR / "dataset"
    tags = sorted((d for d in root.iterdir() if d.is_dir() and d.name.startswith(f"v{FEATURE_VERSION}_")),
                  key=lambda d: d.stat().st_mtime)
    if args.tag:
        tags = [root / args.tag]
    if not tags:
        raise SystemExit(f"no dataset for feature v{FEATURE_VERSION} under {root}")
    ds_dir = tags[-1]
    files = sorted(ds_dir.glob("events_*.csv.gz"))
    frames = [pd.read_csv(f, dtype={"date": str, "code": str}) for f in files]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.dropna(subset=["y", "net_ret"]).sort_values(["date", "ts"]).reset_index(drop=True)
    df[list(FEATURE_NAMES)] = df[list(FEATURE_NAMES)].astype(np.float32)
    n_days = df["date"].nunique()
    print(f"dataset {ds_dir.name}: {len(df):,} events / {n_days} days, base win={df['y'].mean():.3f} "
          f"mean_net={df['net_ret'].mean():+.3f}%", flush=True)
    if n_days < args.min_train_days + args.embargo_days + 1:
        raise SystemExit("not enough days for walk-forward")

    oos, folds = walk_forward(df, args)
    pm = promotion(oos, folds, args)
    rel = reliability(oos)
    print(f"OOS auc={pm['auc_oos']:.3f} scope={pm['scope']} base={pm['base']} kept={pm['kept']} "
          f"promoted={pm['promoted']}", flush=True)

    dates = sorted(df["date"].unique())
    final_dates = dates[-args.train_window:] if args.train_window > 0 else dates
    final_df = df[df["date"].isin(final_dates)]
    booster, best_iter, threshold = fit_with_inner_val(final_df, args, args.threads)
    gains = booster.feature_importance(importance_type="gain")
    importance = sorted(zip(FEATURE_NAMES, map(float, gains)), key=lambda x: -x[1])
    params_path = ds_dir / "params.json"
    label = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "model_id": model_id, "created": datetime.now().isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION, "features": list(FEATURE_NAMES),
        "threshold": threshold, "best_iter": best_iter, "promoted": pm["promoted"], "promotion": pm,
        "dataset": ds_dir.name, "n_events": int(len(df)), "n_days": int(n_days),
        "date_range": f"{dates[0]}~{dates[-1]}", "train_range": f"{final_dates[0]}~{final_dates[-1]}",
        "label": {k: label.get(k) for k in ("tp", "sl", "horizon", "slip_ticks")},
        "lgb_params": LGB_PARAMS, "args": vars(args), "feature_importance": importance,
        "live_feedback": live_feedback(ds_dir.name),
    }
    reports = EDGE_AI_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"report_{model_id}.json").write_text(json.dumps({"meta": meta, "folds": folds, "reliability": rel},
                                                                indent=2, default=str), encoding="utf-8")
    write_report(reports / f"report_{model_id}.md", meta, folds, rel)
    oos[["date", "code", "ts", "p", "threshold", "keep", "y", "net_ret", "m3_ctx_pass"]].to_csv(
        reports / f"oos_{model_id}.csv.gz", index=False, compression="gzip")
    if args.no_publish:
        print(f"evaluated only (no publish): report_{model_id}.md")
        return
    path = publish(booster, meta, EDGE_AI_DIR)
    print(f"published {path.name} threshold={threshold:.4f} promoted={pm['promoted']} -> {EDGE_AI_DIR}/current_*")


if __name__ == "__main__":
    main()
