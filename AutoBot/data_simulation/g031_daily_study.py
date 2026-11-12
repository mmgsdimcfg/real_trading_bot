# -*- coding: utf-8 -*-
"""Daily-bar study: is there a condition that made money after costs? (2026-11-12 사용자 요청)

Every intraday entry tested so far loses about the round-trip cost (g026~g030). This looks at other holding periods on the
daily history (runtime/data/daily_history, about 11 months x 3,000 symbols):
  on    buy the close of day D, sell the open of D+1      (overnight)
  id    buy the open of D+1, sell the close of D+1        (the intraday leg - what the bot trades today)
  cc1   buy the close of D, sell the close of D+1         cc3 / cc5: hold 3 / 5 days
Costs: commission 0.015% per side + sell tax 0.20% (0.23% per round trip) and `--slip` % per side on top (default 0.05).
Closing / opening auction prices are taken as fills. Everything a condition uses is known at the close of day D.

Universe per day (known at D): common stocks, close >= 2,000 won, top `--top` by 20-day average traded value, not at the
upper limit on D (cannot be bought at the close) and with trades on D+1.

  python g031_daily_study.py --report base      holding periods, by month
  python g031_daily_study.py --report scan      one condition at a time (deciles), first part of the history vs from July
  python g031_daily_study.py --report model     walk-forward LightGBM, top-K per day portfolios
  python g031_daily_study.py --report rules     fixed rules chosen before July, checked from July
Output: runtime/edge_ai/timing_study/report_daily_<name>.md. Nothing here is imported by r003.

Update log
- [2026-11-12] type=feat - first version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g013_edge_ai_dataset as g013  # noqa: E402
import g026_timing_study as g026  # noqa: E402

HIST_DIR = g013.DATA_ROOT / "daily_history"
COST_PCT = 0.015 * 2 + 0.20          # % per round trip
SPLIT = "20260701"                   # conditions are chosen before this day and checked from it
TARGETS = ("on", "id", "cc1", "cc3", "cc5")
FEATURES = ["ret1", "gap", "intra", "clv", "range", "vol_ratio", "ret5", "ret20", "ret60", "ma20_gap", "ma60_gap",
            "hi20_dist", "lo20_dist", "atr_pct", "rsi14", "amt_rank", "mkt_ret1", "rel_ret1", "rel_ret5", "upper_wick", "log_price"]


def load_panel() -> dict[str, pd.DataFrame]:
    """date x code frames: open, high, low, close, volume, amount (common stocks only)."""
    cols = {k: {} for k in ("open", "high", "low", "close", "volume", "amount")}
    for f in sorted(HIST_DIR.glob("*.csv")):
        code = f.stem
        if not (code.isdigit() and code.endswith("0")):
            continue
        try:
            d = pd.read_csv(f, usecols=["date", "open", "high", "low", "close", "volume", "amount"], dtype={"date": str})
        except Exception:
            continue
        if len(d) < 80:
            continue
        d = d.drop_duplicates("date").set_index("date")
        for k in cols:
            cols[k][code] = pd.to_numeric(d[k], errors="coerce")
    return {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}


def build(top: int, slip: float) -> pd.DataFrame:
    """Long frame: one row per (date D, code in the universe) with features at D and net returns after D."""
    p = load_panel()
    o, h, l, c, v, a = (p[k] for k in ("open", "high", "low", "close", "volume", "amount"))
    c = c.where(c > 0)
    prev = c.shift(1)
    ret1 = c / prev - 1
    # a daily move beyond the +-30% price limit is a split / data error: nothing around it is used
    bad = (ret1.abs() > 0.305)
    bad = bad | bad.shift(-1, fill_value=False) | bad.shift(1, fill_value=False)
    amt20 = a.rolling(20, min_periods=10).mean()
    rank = amt20.rank(axis=1, ascending=False)
    cost = COST_PCT + 2 * slip
    nxt_o, nxt_c, nxt_v = o.shift(-1), c.shift(-1), v.shift(-1)
    tgt = {
        "on": (nxt_o / c - 1) * 100 - cost,
        "id": (nxt_c / nxt_o - 1) * 100 - cost,
        "cc1": (nxt_c / c - 1) * 100 - cost,
        "cc3": (c.shift(-3) / c - 1) * 100 - cost,
        "cc5": (c.shift(-5) / c - 1) * 100 - cost,
    }
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()]).groupby(level=0).max()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    mkt_mask = (rank <= top) & (c >= 2000)
    mkt = ret1.where(mkt_mask).mean(axis=1)
    rng = (h - l).where(h > l)
    feats = {
        "ret1": ret1 * 100, "gap": (o / prev - 1) * 100, "intra": (c / o - 1) * 100, "clv": (c - l) / rng,
        "range": rng / prev * 100, "vol_ratio": v / v.rolling(20, min_periods=10).mean().shift(1),
        "ret5": (c / c.shift(5) - 1) * 100, "ret20": (c / c.shift(20) - 1) * 100, "ret60": (c / c.shift(60) - 1) * 100,
        "ma20_gap": (c / c.rolling(20).mean() - 1) * 100, "ma60_gap": (c / c.rolling(60).mean() - 1) * 100,
        "hi20_dist": (c / h.rolling(20).max() - 1) * 100, "lo20_dist": (c / l.rolling(20).min() - 1) * 100,
        "atr_pct": tr.rolling(14).mean() / c * 100, "rsi14": 100 - 100 / (1 + up / dn), "amt_rank": rank,
        "mkt_ret1": pd.DataFrame(np.repeat(mkt.to_numpy()[:, None] * 100, c.shape[1], axis=1), index=c.index, columns=c.columns),
        "rel_ret1": ret1.sub(mkt, axis=0) * 100, "rel_ret5": (c / c.shift(5) - 1).sub(c.where(mkt_mask).pct_change(5, fill_method=None).mean(axis=1), axis=0) * 100,
        "upper_wick": (h - np.maximum(o, c)) / rng, "log_price": np.log10(c),
    }
    ok = mkt_mask & ~bad & (ret1 < 0.29) & (nxt_v > 0) & nxt_o.notna() & (v > 0)
    idx = ok.stack()
    idx = idx[idx].index
    out = pd.DataFrame(index=idx)
    for k, f in {**feats, **tgt}.items():
        out[k] = f.stack(future_stack=True).reindex(idx).to_numpy()
    out = out.reset_index()
    out.columns = ["date", "code"] + list(out.columns[2:])
    return out.dropna(subset=["on", "cc1", "ret20", "atr_pct"]).reset_index(drop=True)


def day_stats(x: pd.Series, dates: pd.Series) -> dict:
    """Trade mean and the day-level picture (days are the independent unit)."""
    by = x.groupby(dates.to_numpy()).mean()
    t = by.mean() / (by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 2 and by.std(ddof=1) > 0 else float("nan")
    return {"n": int(len(x)), "days": int(len(by)), "mean": float(x.mean()), "win": float((x > 0).mean()),
            "day_mean": float(by.mean()), "days_pos": float((by > 0).mean()), "t": float(t)}


def fmt(s: dict) -> str:
    return f"{s['mean']:+.3f}% (n {s['n']:,}, win {s['win'] * 100:.0f}%, day avg {s['day_mean']:+.3f}%, days+ {s['days_pos'] * 100:.0f}%, t {s['t']:+.1f})"


def report_base(d: pd.DataFrame, args) -> list[str]:
    lines = ["## Holding periods - every universe stock every day (equal weight)", "",
             "| period | all | before July | from July |", "|---|---|---|---|"]
    a, b = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    for k in TARGETS:
        x = d.dropna(subset=[k])
        lines.append(f"| {k} | {fmt(day_stats(x[k], x['date']))} | {fmt(day_stats(a[k].dropna(), a.loc[a[k].notna(), 'date']))} | "
                     f"{fmt(day_stats(b[k].dropna(), b.loc[b[k].notna(), 'date']))} |")
    lines += ["", "## By month (mean net % per position)", "", "| month | " + " | ".join(TARGETS) + " | stocks/day |", "|---|" + "---|" * (len(TARGETS) + 1)]
    for m, g in d.groupby(d["date"].str.slice(0, 6)):
        lines.append(f"| {m} | " + " | ".join(f"{g[k].mean():+.2f}" for k in TARGETS) + f" | {len(g) / g['date'].nunique():.0f} |")
    return lines


def report_scan(d: pd.DataFrame, args) -> list[str]:
    lines = ["## One condition at a time - deciles of the value at the close of day D", "",
             "Each cell: mean net % before July / from July. A condition is only interesting when both halves agree.", ""]
    a, b = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    for target in ("on", "cc1", "cc5"):
        lines += [f"### target {target}", "", "| feature | lowest 10% | 10-30% | 30-70% | 70-90% | highest 10% |", "|---|---|---|---|---|---|"]
        for f in FEATURES:
            qs = d[f].quantile([0.1, 0.3, 0.7, 0.9]).to_numpy()
            edges = [-np.inf, *qs, np.inf]
            cells = []
            for k in range(5):
                ma = a.loc[(a[f] > edges[k]) & (a[f] <= edges[k + 1]), target].mean()
                mb = b.loc[(b[f] > edges[k]) & (b[f] <= edges[k + 1]), target].mean()
                mark = " **" if ma > 0 and mb > 0 else ""
                cells.append(f"{ma:+.2f} / {mb:+.2f}{mark}")
            lines.append(f"| {f} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def portfolio(sel: pd.DataFrame, target: str) -> dict:
    """Equal-weight daily portfolio of the selected rows -> day series statistics."""
    by = sel.groupby("date")[target].mean()
    if len(by) < 5:
        return {"days": int(len(by))}
    cum = by.cumsum()
    dd = float((cum - cum.cummax()).min())
    t = by.mean() / (by.std(ddof=1) / np.sqrt(len(by)))
    return {"days": int(len(by)), "n": int(len(sel)), "per_day": len(sel) / len(by), "day_mean": float(by.mean()),
            "days_pos": float((by > 0).mean()), "t": float(t), "total": float(by.sum()), "max_dd": dd,
            "win": float((sel[target] > 0).mean()), "worst": float(by.min())}


def pfmt(s: dict) -> str:
    if s.get("days", 0) < 5:
        return f"days {s.get('days', 0)}"
    return (f"{s['day_mean']:+.3f}%/day, total {s['total']:+.1f}% over {s['days']} days, days+ {s['days_pos'] * 100:.0f}%, "
            f"t {s['t']:+.1f}, worst day {s['worst']:+.1f}%, max drawdown {s['max_dd']:+.1f}%, {s['per_day']:.1f} names/day, win {s['win'] * 100:.0f}%")


def report_model(d: pd.DataFrame, args) -> list[str]:
    """Walk-forward: every month from July is scored by a model trained on all earlier days (5-day embargo)."""
    import lightgbm as lgb
    lines = ["## Walk-forward LightGBM (monthly refit, trained only on earlier days)", ""]
    dates = sorted(d["date"].unique())
    months = sorted({x[:6] for x in dates if x >= SPLIT})
    for target in ("on", "cc1", "cc5"):
        parts = []
        for m in months:
            te = d[d["date"].str.slice(0, 6) == m]
            first = te["date"].min()
            cut = dates[max(0, dates.index(first) - 6)]
            tr = d[(d["date"] < cut)].dropna(subset=[target])
            if len(tr) < 5000 or te.empty:
                continue
            y = tr[target].clip(-10, 10)
            b = lgb.train({"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 300,
                           "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0,
                           "verbose": -1, "seed": 7, "num_threads": args.threads}, lgb.Dataset(tr[FEATURES], y), num_boost_round=200)
            te = te.dropna(subset=[target]).copy()
            te["score"] = b.predict(te[FEATURES])
            parts.append(te)
        if not parts:
            continue
        oos = pd.concat(parts)
        lines += [f"### target {target} ({oos['date'].min()}~{oos['date'].max()})", "",
                  f"- all universe stocks: {pfmt(portfolio(oos, target))}"]
        for k in (5, 10, 20):
            top = oos.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(k)
            bot = oos.sort_values(["date", "score"]).groupby("date").head(k)
            lines.append(f"- top {k} by score each day: {pfmt(portfolio(top, target))}")
            lines.append(f"- bottom {k} (check): {pfmt(portfolio(bot, target))}")
        lines.append("")
        if target == "on":
            imp = pd.Series(b.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
            lines += ["- last model, gain share: " + ", ".join(f"{k} {v / imp.sum() * 100:.0f}%" for k, v in imp.head(8).items()), ""]
    return lines


RULES = {
    # name: (target, description, mask builder) - written from the before-July half of the scan, then checked from July
    "strong_close_on": ("on", "당일 +3% 이상 상승, 종가가 당일 고가 근처(clv>=0.8), 거래량 2배 이상 -> 종가 매수, 익일 시가 매도",
                        lambda d: (d["ret1"] >= 3) & (d["clv"] >= 0.8) & (d["vol_ratio"] >= 2)),
    "strong_close_on_top50": ("on", "위 조건 + 거래대금 상위 50",
                              lambda d: (d["ret1"] >= 3) & (d["clv"] >= 0.8) & (d["vol_ratio"] >= 2) & (d["amt_rank"] <= 50)),
    "big_up_on": ("on", "당일 +7% 이상 상승, 종가가 고가 근처(clv>=0.7) -> 종가 매수, 익일 시가 매도",
                  lambda d: (d["ret1"] >= 7) & (d["clv"] >= 0.7)),
    "weak_close_on": ("on", "당일 -3% 이하 하락, 종가가 저가 근처(clv<=0.2) -> 종가 매수, 익일 시가 매도 (반등)",
                      lambda d: (d["ret1"] <= -3) & (d["clv"] <= 0.2)),
    "momentum_cc5": ("cc5", "20일 +20% 이상, 20일 고가에서 3% 이내, 당일 양봉 -> 종가 매수, 5일 보유",
                     lambda d: (d["ret20"] >= 20) & (d["hi20_dist"] >= -3) & (d["intra"] > 0)),
    "dip_cc5": ("cc5", "20일 -15% 이하, RSI14 < 30 -> 종가 매수, 5일 보유 (낙폭 과대)",
                lambda d: (d["ret20"] <= -15) & (d["rsi14"] < 30)),
    "any_on": ("on", "조건 없음: 종목군 전체 종가 매수, 익일 시가 매도", lambda d: d["ret1"].notna()),
}


def report_rules(d: pd.DataFrame, args) -> list[str]:
    lines = ["## Fixed rules", "", "| rule | 설명 | before July | from July |", "|---|---|---|---|"]
    a, b = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    for name, (target, desc, mask) in RULES.items():
        sa, sb = a[mask(a)].dropna(subset=[target]), b[mask(b)].dropna(subset=[target])
        lines.append(f"| {name} | {desc} | {pfmt(portfolio(sa, target))} | {pfmt(portfolio(sb, target))} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", choices=("base", "scan", "model", "rules", "all"), default="all")
    ap.add_argument("--top", type=int, default=300, help="universe size by 20-day traded value")
    ap.add_argument("--slip", type=float, default=0.05, help="slippage %% per side on top of commission and tax")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    d = build(args.top, args.slip)
    head = [f"# g031 daily study - {d['date'].min()}~{d['date'].max()}, {d['date'].nunique()} days, {d['code'].nunique():,} symbols, "
            f"{len(d):,} stock-days", "",
            f"Universe: top {args.top} by 20-day traded value. Net of {COST_PCT:.2f}% costs + {args.slip:.2f}% slippage per side. "
            f"'before July' = {d['date'].min()}~20260630 (conditions chosen here), 'from July' = check.", ""]
    names = ("base", "scan", "model", "rules") if args.report == "all" else (args.report,)
    for name in names:
        lines = head + {"base": report_base, "scan": report_scan, "model": report_model, "rules": report_rules}[name](d, args)
        text = "\n".join(lines) + "\n"
        out = g026.OUT_DIR / f"report_daily_{name}.md"
        out.write_text(text, encoding="utf-8")
        print(text)
        print("->", out)


if __name__ == "__main__":
    main()
