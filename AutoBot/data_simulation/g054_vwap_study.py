# -*- coding: utf-8 -*-
"""g054 - VWAP position study (read-only; no orders, no bot state, nothing written under runtime/).

Question (user 2026-11-26): "bars that stay above the session VWAP go on rising, bars under it go on falling" - is the position
against VWAP usable as a condition of the rule strategy / the rebound alert, beyond what today's rise already says?

Parts
  A  every liquid code-day of the Edge AI dataset, one sample each 10 minutes (09:30 - 14:10): VWAP state -> next 30 / 60 minutes
     and to 15:15, raw and as the excess over the same-minute mean of all codes (market move removed); also inside quintiles of
     "rise since the open" (momentum control). Day level: close vs the day's VWAP -> next day's open / close.
  B  the rule strategy's trigger events (g013 dataset rows): label net return by VWAP state at the trigger.
  C  LightGBM time split with / without the extra VWAP persistence features (the model already has vwap_gap_pct).

VWAP = cumsum(close x volume) / cumsum(volume) of the regular session's 1m bars (same as g013 / the app's chart).
Day-clustered bootstrap for every difference (days are the independent unit).

usage: g054_vwap_study.py --out <dir> [--dataset <dir>] [--workers 3] [--days N]

Update log:
- [2026-11-26] type=feat owner=claude
    summary: 신규 - VWAP 위치/유지 상태가 이후 수익률을 예측하는지 검증(무조건부, 규칙 매매 트리거, 모델 추가 피처). 읽기 전용.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))
sys.path.insert(0, str(HERE))
import r001_define_config as cfg  # noqa: E402
from g013_edge_ai_dataset import _index_dir, load_minute_bars  # noqa: E402
from r011_edge_ai import FEATURE_NAMES, session_raw_1min  # noqa: E402

DATA_ROOT = Path(cfg.RUNTIME_DATA_DIR)
EXTRA = ["vw_frac30", "vw_frac_sess", "vw_slope30", "vw_gap", "vw_cross30"]


def code_day(path: Path, trade_date: str) -> pd.DataFrame | None:
    raw = session_raw_1min(load_minute_bars(path), pd.Timestamp(trade_date))
    if raw is None or len(raw) < 60:
        return None
    c, v = raw["close"].astype(float), raw["volume"].astype(float)
    cv = v.cumsum()
    vwap = ((c * v).cumsum() / cv.where(cv > 0)).ffill()
    above = (c > vwap).astype(float)
    out = pd.DataFrame(index=raw.index)
    out["close"] = c
    out["vw_gap"] = (c / vwap - 1) * 100
    out["vw_frac30"] = above.rolling(30, min_periods=30).mean()
    out["vw_frac_sess"] = above.expanding(min_periods=10).mean()
    out["vw_slope30"] = (vwap / vwap.shift(30) - 1) * 100
    out["vw_cross30"] = above.diff().abs().rolling(30, min_periods=30).sum()
    out["rise_open"] = (c / float(raw["open"].iloc[0]) - 1) * 100
    out["fwd30"] = (c.shift(-30) / c - 1) * 100
    out["fwd60"] = (c.shift(-60) / c - 1) * 100
    eod = c[c.index.time <= pd.Timestamp("15:15").time()]
    out["fwd_eod"] = (float(eod.iloc[-1]) / c - 1) * 100 if len(eod) else np.nan
    out["day_open"] = float(raw["open"].iloc[0])
    out["day_close"] = float(c.iloc[-1])
    out["day_vwap"] = float(vwap.iloc[-1])
    return out


def build_day(task) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_date, codes, ev_ts = task
    files = _index_dir(DATA_ROOT / trade_date, "_10s.txt")
    samples, events, days = [], [], []
    for code in codes:
        f = files.get(code)
        d = code_day(f, trade_date) if f else None
        if d is None:
            continue
        t = d.index
        m = (t.minute % 10 == 0) & (t.time >= pd.Timestamp("09:30").time()) & (t.time <= pd.Timestamp("14:10").time())
        s = d.loc[m, EXTRA + ["rise_open", "fwd30", "fwd60", "fwd_eod"]].copy()
        s.insert(0, "hm", s.index.strftime("%H:%M"))
        s.insert(0, "code", code)
        samples.append(s.reset_index(drop=True))
        ts = pd.to_datetime(ev_ts.get(code, []))
        e = d.reindex(ts)[EXTRA].copy()
        e.insert(0, "ts", ts.strftime("%Y-%m-%d %H:%M:%S"))
        e.insert(0, "code", code)
        events.append(e.reset_index(drop=True))
        last = d.iloc[-1]
        days.append({"code": code, "day_open": last["day_open"], "day_close": last["day_close"], "day_vwap": last["day_vwap"]})
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    a, b, c = cat(samples), cat(events), pd.DataFrame(days)
    for x in (a, b, c):
        x.insert(0, "date", trade_date)
    return trade_date, a, b, c


def boot_diff(df: pd.DataFrame, col: str, hi: pd.Series, lo: pd.Series, n: int = 400, seed: int = 7) -> tuple[float, float, float]:
    """mean(col | hi) - mean(col | lo), day-clustered bootstrap 95% interval"""
    g = pd.DataFrame({"d": df["date"], "x": df[col], "hi": hi, "lo": lo}).dropna(subset=["x"])
    a = g[g.hi].groupby("d")["x"].agg(["sum", "count"])
    b = g[g.lo].groupby("d")["x"].agg(["sum", "count"])
    days = sorted(set(a.index) | set(b.index))
    a, b = a.reindex(days).fillna(0).to_numpy(), b.reindex(days).fillna(0).to_numpy()
    point = a[:, 0].sum() / max(1, a[:, 1].sum()) - b[:, 0].sum() / max(1, b[:, 1].sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(n, len(days)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    d = sa[:, 0] / np.maximum(1, sa[:, 1]) - sb[:, 0] / np.maximum(1, sb[:, 1])
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def auc(y, p) -> float:
    """rank AUC (no sklearn in the venv)"""
    y = np.asarray(y, dtype=float)
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1.0, n1 * n0))


def state_of(df: pd.DataFrame) -> pd.Series:
    s = pd.Series("mixed", index=df.index)
    s[(df["vw_frac30"] >= 0.999) & (df["vw_gap"] > 0)] = "above_30m"
    s[(df["vw_frac30"] <= 0.001) & (df["vw_gap"] < 0)] = "below_30m"
    s[df["vw_frac30"].isna()] = "na"
    return s


def table(df: pd.DataFrame, by: pd.Series, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = g[cols].mean().round(3)
    out.insert(0, "n", g.size())
    for c in cols[:2]:
        out[f"{c}_pos%"] = (g[c].apply(lambda x: (x > 0).mean()) * 100).round(1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default=None, help="g013 dataset folder (default: the newest under runtime/edge_ai/dataset)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--days", type=int, default=0, help="only the newest N days (0 = all)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Path(args.dataset) if args.dataset else max((d for d in (Path(cfg.RUNTIME_ROOT) / "edge_ai" / "dataset").iterdir() if d.is_dir()),
                                                      key=lambda d: d.stat().st_mtime)
    files = sorted(ds.glob("events_*.csv.gz"))
    if args.days:
        files = files[-args.days:]
    ev = pd.concat((pd.read_csv(f, dtype={"date": str, "code": str}) for f in files), ignore_index=True)
    print(f"dataset {ds.name}: {len(ev)} events, {ev['date'].nunique()} days", flush=True)

    cache = out / "samples.pkl"
    if cache.exists():
        A, B, D = pd.read_pickle(cache)
    else:
        tasks = []
        for d, g in ev.groupby("date"):
            tasks.append((d, sorted(g["code"].unique()), g.groupby("code")["ts"].apply(list).to_dict()))
        As, Bs, Ds = [], [], []
        with Pool(max(1, args.workers)) as pool:
            for d, a, b, c in pool.imap_unordered(build_day, tasks):
                As.append(a), Bs.append(b), Ds.append(c)
                print(f"  {d}: {len(a)} samples", flush=True)
        A, B, D = (pd.concat(x, ignore_index=True) for x in (As, Bs, Ds))
        pd.to_pickle((A, B, D), cache)
    rep: list[str] = []

    def emit(title: str, body) -> None:
        text = f"\n## {title}\n{body if isinstance(body, str) else body.to_string()}"
        print(text, flush=True)
        rep.append(text)

    # ---- A: unconditional ----
    A = A.sort_values(["date", "hm"]).reset_index(drop=True)
    fw = ["fwd30", "fwd60", "fwd_eod"]
    for c in fw:
        A[f"x_{c}"] = A[c] - A.groupby(["date", "hm"])[c].transform("mean")
    A["state"] = state_of(A)
    xs = [f"x_{c}" for c in fw]
    emit(f"A1 state -> forward return % (n={len(A)}, {A['date'].nunique()} days; x_ = excess over same-minute mean)", table(A, A["state"], xs + fw))
    hi, lo = A["state"] == "above_30m", A["state"] == "below_30m"
    emit("A2 above_30m - below_30m, day bootstrap 95%",
         "\n".join(f"{c}: %+.3f [%+.3f, %+.3f]" % boot_diff(A, c, hi, lo) for c in xs + fw))
    A["rise_q"] = A.groupby(["date", "hm"])["rise_open"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x) >= 20 else np.nan)
    rows = []
    for q, g in A.dropna(subset=["rise_q"]).groupby("rise_q"):
        h, l_ = g["state"] == "above_30m", g["state"] == "below_30m"
        r = {"rise_q": int(q), "rise_open_mean": round(g["rise_open"].mean(), 2), "n_above": int(h.sum()), "n_below": int(l_.sum())}
        for c in ("x_fwd30", "x_fwd60", "x_fwd_eod"):
            d0, a0, b0 = boot_diff(g, c, h, l_, n=200)
            r[c] = f"{d0:+.3f} [{a0:+.3f},{b0:+.3f}]"
        rows.append(r)
    emit("A3 above - below inside quintiles of rise since the open (momentum control)", pd.DataFrame(rows).set_index("rise_q"))
    A["gap_b"] = pd.cut(A["vw_gap"], [-99, -2, -1, -0.3, 0, 0.3, 1, 2, 99])
    emit("A4 close vs VWAP gap % bucket", table(A, A["gap_b"], xs))
    A["sess_b"] = pd.cut(A["vw_frac_sess"], [-0.01, 0.1, 0.3, 0.7, 0.9, 1.0])
    emit("A5 share of the session's closes above VWAP", table(A, A["sess_b"], xs))
    A["slope_b"] = pd.cut(A["vw_slope30"], [-99, -0.5, -0.1, 0.1, 0.5, 99])
    emit("A6 VWAP 30-minute slope %", table(A, A["slope_b"], xs))
    # reclaim / loss: the cross itself
    A["cross_b"] = pd.cut(A["vw_cross30"], [-1, 0, 2, 5, 99])
    emit("A7 VWAP crossings in the last 30 minutes (0 = one side only)", table(A, [A["cross_b"], np.sign(A["vw_gap"]).map({1: "above", -1: "below", 0: "at"})], xs))

    # ---- day level ----
    D = D.sort_values(["code", "date"]).reset_index(drop=True)
    dates = sorted(D["date"].unique())
    nxt = dict(zip(dates[:-1], dates[1:]))
    D["next"] = D["date"].map(nxt)
    N = D[["date", "code", "day_open", "day_close"]].rename(columns={"date": "next", "day_open": "n_open", "day_close": "n_close"})
    D = D.merge(N, on=["next", "code"], how="inner")
    D["cv"] = (D["day_close"] / D["day_vwap"] - 1) * 100
    D["day_ret"] = (D["day_close"] / D["day_open"] - 1) * 100
    D["gap_next"] = (D["n_open"] / D["day_close"] - 1) * 100
    D["oc_next"] = (D["n_close"] / D["n_open"] - 1) * 100
    D["cc_next"] = (D["n_close"] / D["day_close"] - 1) * 100
    for c in ("gap_next", "oc_next", "cc_next"):
        D[f"x_{c}"] = D[c] - D.groupby("date")[c].transform("mean")
    D["cv_b"] = pd.cut(D["cv"], [-99, -2, -0.5, 0, 0.5, 2, 99])
    dx = ["x_gap_next", "x_oc_next", "x_cc_next"]
    emit(f"D1 day close vs day VWAP % -> next day (n={len(D)}; excess over the day's mean)", table(D, D["cv_b"], dx + ["day_ret"]))
    emit("D2 close above VWAP - close below VWAP, day bootstrap 95%",
         "\n".join(f"{c}: %+.3f [%+.3f, %+.3f]" % boot_diff(D, c, D["cv"] > 0, D["cv"] < 0) for c in dx))

    # ---- B: rule trigger events ----
    E = ev.merge(B, on=["date", "code", "ts"], how="left")
    E["state"] = state_of(E)
    lab = [c for c in ("net_ret", "y_tp", "y", "fwd_30", "fwd_60", "run_mfe_pct", "mae_pct") if c in E.columns]
    emit(f"B1 rule trigger events by VWAP state (label of {ds.name})", table(E, E["state"], lab))
    emit("B2 above_30m - below_30m at the trigger, day bootstrap 95%",
         "\n".join(f"{c}: %+.3f [%+.3f, %+.3f]" % boot_diff(E, c, E["state"] == "above_30m", E["state"] == "below_30m") for c in lab[:5]))
    E["gap_b"] = pd.cut(E["vwap_gap_pct"], [-99, -1, -0.3, 0, 0.3, 1, 2, 3, 99])
    emit("B3 by vwap_gap_pct at the trigger", table(E, E["gap_b"], lab))
    E["sess_b"] = pd.cut(E["vw_frac_sess"], [-0.01, 0.1, 0.3, 0.7, 0.9, 1.0])
    emit("B4 by share of the session above VWAP", table(E, E["sess_b"], lab))
    rule = (E["vw_gap"] > 0) & (E["vw_frac30"] >= 0.8) & (E["vw_slope30"] > 0)
    emit("B5 candidate rule: above VWAP, >= 80% of the last 30 closes above, VWAP rising",
         table(E, rule.map({True: "pass", False: "fail"}), lab) .to_string() + "\npass - fail: " +
         "; ".join(f"{c} %+.3f [%+.3f, %+.3f]" % boot_diff(E, c, rule, ~rule) for c in lab[:2]))

    # ---- C: model with / without the persistence features ----
    try:
        import lightgbm as lgb
        target = "y_tp" if "y_tp" in E.columns else "y"
        days = sorted(E["date"].unique())
        cut = days[int(len(days) * 0.7)]
        tr, te = E[E["date"] < cut], E[E["date"] >= cut]
        base = [f for f in FEATURE_NAMES if f in E.columns]
        prm = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 200,
               "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 42,
               "num_threads": args.workers}
        lines = [f"target {target}; train {len(tr)} (< {cut}), test {len(te)}; base test mean net {te['net_ret'].mean():+.3f}"]
        for name, feats in (("base", base), ("base+vwap_persist", base + [f for f in EXTRA if f != "vw_gap"])):
            m = lgb.train(prm, lgb.Dataset(tr[feats], tr[target]), 150)
            pr = m.predict(te[feats])
            top = te[pr >= np.quantile(pr, 0.8)]
            imp = dict(zip(feats, m.feature_importance("gain")))
            rank = sorted(imp, key=imp.get, reverse=True)
            extra = ", ".join(f"{f}#{rank.index(f) + 1}" for f in feats if f.startswith("vw") and f in rank)
            lines.append(f"{name}: AUC {auc(te[target], pr):.4f}, top 20% mean net {top['net_ret'].mean():+.3f} "
                         f"win {top[target].mean() * 100:.1f}% | rank of {extra}")
        emit("C model time split (70 / 30 by day)", "\n".join(lines))
    except Exception as exc:  # study only
        emit("C model", f"skipped: {exc!r}")

    (out / "report.txt").write_text("\n".join(rep), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"dataset": ds.name, "events": len(ev), "samples": len(A)}), encoding="utf-8")


if __name__ == "__main__":
    main()
