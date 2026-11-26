# -*- coding: utf-8 -*-
"""g056_candidate_study.py - event study of entry-candidate definitions for the Edge AI redesign (read-only, no orders).

Why: AutoBot/vwap_indicator_review_20261126.md 2-4 / 6-(4). The pool the Edge AI model filters (the hybrid 1-minute trigger events
of g013) averages about -0.57 % net per trade under the live exits and no model target finds a profitable part of it. A filter
cannot help a pool without an edge, so candidate definitions are measured FIRST, each with the next tradable price and all costs.

Candidates (fixed before the run; the first event per code-day, bars labelled CAND_FROM .. CAND_TO unless noted)
  orb         opening-range breakout: first close above the highest high of the bars labelled <= 09:30
  volsurge    first bar with volume >= 5 x the mean of the 20 bars before it, close > open, close >= highest high of those 20 bars
  consol      first close above the session high when that high had stood for >= 30 bars (from 10:00)
  pullback    daily uptrend (D-1 close > MA20 > MA60), the session had been >= +2 % above the open, pulled back >= 1.5 % from the
              session high, then the first close above the highest high of the 10 bars before it while above VWAP
  gaphold     gap up >= +2 % over the D-1 close and, at the bar labelled 09:30, close > day open and close > VWAP (entry after it)
  strongclose at the bar labelled 14:30: close >= +3 % over the open and within 1 % of the session high

Measured per event
  entry = open of the next bar + 1 tick.
  net_live = g013.triple_barrier with the live exits (--tp / --sl / --horizon, default = the label the nightly dataset uses,
             g013.live_label()), net_eod = close of the last bar labelled <= 15:15 - 1 tick, net_60 = close 60 bars later - 1 tick.
  All net of commission 0.015 % x 2 and sell tax 0.20 %.
Split: the first 70 % of the days are "discovery", the last 30 % "holdout". A candidate is worth a model only if the day-bootstrap
95 % lower bound of a net mean is above 0 in BOTH parts (the report says so per row). Nothing here changes the bot.

Run:
  python g056_candidate_study.py --out DIR [--workers 4] [--days N]
Output: DIR/candidates.csv, DIR/candidate_report.md

Update log:
- [2026-11-27] type=feat owner=claude
    summary: 신규 - Edge AI 후보군 재정의용 사건 연구(후보 6종, 실전 청산 라벨 + 비용, 발견/검증 구간 분리). 주문 없음.
    impact: tooling (매매 판정 불변)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))
sys.path.insert(0, str(HERE))
import r001_define_config as cfg  # noqa: E402
import g013_edge_ai_dataset as g013  # noqa: E402
import g055_vwap_reclaim_log as g055  # noqa: E402
from r011_edge_ai import find_prev_data_dir, is_fund_product, session_raw_1min  # noqa: E402

DATA_ROOT = Path(cfg.RUNTIME_DATA_DIR)
UNIVERSE = SimpleNamespace(min_turnover=3e9, min_price=1000, max_codes=0)
CAND_FROM = pd.Timestamp("09:10").time()
CAND_TO = pd.Timestamp("14:30").time()
T0930 = pd.Timestamp("09:30").time()
T1000 = pd.Timestamp("10:00").time()
T1430 = pd.Timestamp("14:30").time()
EOD = pd.Timestamp("15:15").time()
KINDS = ("orb", "volsurge", "consol", "pullback", "gaphold", "strongclose")
COLS = ["date", "code", "kind", "ts", "tick_pct", "entry", "net_live", "exit_reason", "net_eod", "net_60"]


def first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else -1


def daily_trend(daily: pd.DataFrame | None, trade_date: str) -> tuple[float, bool]:
    """(D-1 close, D-1 close > MA20 > MA60) from a <code>_daily.csv frame; (nan, False) when it cannot be told"""
    if daily is None or daily.empty or "close" not in daily.columns:
        return float("nan"), False
    d = daily[daily["date"].astype(str).str.slice(0, 8) < trade_date]
    close = pd.to_numeric(d["close"], errors="coerce").dropna()
    if close.empty:
        return float("nan"), False
    prev = float(close.iloc[-1])
    if len(close) < 60:
        return prev, False
    ma20, ma60 = float(close.tail(20).mean()), float(close.tail(60).mean())
    return prev, bool(prev > ma20 > ma60)


def candidates(raw: pd.DataFrame, prev_close: float, uptrend: bool) -> dict[str, int]:
    """kind -> bar position of the first event of the session (missing = none). Every rule reads bars <= the event bar only."""
    o, h, l, c, v = (raw[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    times = raw.index.time
    in_win = np.array([CAND_FROM <= t <= CAND_TO for t in times])
    cv = np.cumsum(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cv > 0, np.cumsum(c * v) / cv, np.nan)
    hs = pd.Series(h)
    out: dict[str, int] = {}

    opening = np.array([t <= T0930 for t in times])
    if opening.any() and not opening.all():
        rng_high = h[opening].max()
        out["orb"] = first_true(~opening & in_win & (c > rng_high))

    prev_vol = pd.Series(v).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    prev_high20 = hs.shift(1).rolling(20, min_periods=20).max().to_numpy()
    with np.errstate(invalid="ignore"):
        out["volsurge"] = first_true(in_win & (v >= 5 * prev_vol) & (prev_vol > 0) & (c > o) & (c >= prev_high20))

    prev_sess_high = np.r_[np.nan, np.maximum.accumulate(h)[:-1]]
    # bars since the session high (before this bar) was set
    age = np.zeros(n)
    last_set = 0
    run_max = -np.inf
    for i in range(n):
        age[i] = i - last_set
        if h[i] > run_max:
            run_max, last_set = h[i], i
    after10 = np.array([t >= T1000 for t in times])
    with np.errstate(invalid="ignore"):
        out["consol"] = first_true(in_win & after10 & (c > prev_sess_high) & (age >= 30))

    if uptrend:
        sess_high = np.maximum.accumulate(h)
        was_up = np.maximum.accumulate(h) >= o[0] * 1.02
        # the deepest pullback from the session high seen so far, reset when a new session high is made
        pulled = np.zeros(n, dtype=bool)
        deep = False
        for i in range(n):
            if i and sess_high[i] > sess_high[i - 1]:
                deep = False
            if l[i] <= sess_high[i] * (1 - 0.015):
                deep = True
            pulled[i] = deep
        prev_pulled = np.r_[False, pulled[:-1]]
        prev_high10 = hs.shift(1).rolling(10, min_periods=10).max().to_numpy()
        with np.errstate(invalid="ignore"):
            out["pullback"] = first_true(in_win & was_up & prev_pulled & (c > prev_high10) & (c > vwap))

    if prev_close == prev_close and prev_close > 0 and o[0] >= prev_close * 1.02:
        at = first_true(np.array([t == T0930 for t in times]))
        if at >= 0 and c[at] > o[0] and c[at] > vwap[at]:
            out["gaphold"] = at

    at = first_true(np.array([t == T1430 for t in times]))
    if at >= 0 and c[at] >= o[0] * 1.03 and c[at] >= np.max(h[: at + 1]) * 0.99:
        out["strongclose"] = at
    return {k: i for k, i in out.items() if 0 <= i < n - 1}


def code_events(code: str, trade_date: str, minute_file: Path, daily_file: Path | None, label) -> list[dict]:
    raw = session_raw_1min(g013.load_minute_bars(minute_file), pd.Timestamp(trade_date))
    if raw is None or len(raw) < 60:
        return []
    prev_close, uptrend = daily_trend(g013._read_daily(daily_file), trade_date)
    found = candidates(raw, prev_close, uptrend)
    if not found:
        return []
    o, h, l, c = (raw[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    times = raw.index.time
    n = len(c)
    eod_idx = np.flatnonzero(np.array([t <= EOD for t in times]))
    rows = []
    for kind, i in found.items():
        lab = g013.triple_barrier(o, h, l, c, times, i, label)
        if lab is None or not len(eod_idx) or i + 1 > eod_idx[-1]:
            continue
        entry = lab["entry_px"]
        net_eod = float(g055.net_pct(np.array([entry]), np.array([c[eod_idx[-1]]]))[0])
        net_60 = float(g055.net_pct(np.array([entry]), np.array([c[i + 61]]))[0]) if i + 61 < n else float("nan")
        rows.append({"date": trade_date, "code": code, "kind": kind, "ts": raw.index[i].strftime("%H:%M"),
                     "tick_pct": float(g055.ticks(np.array([c[i]]))[0] / c[i] * 100.0), "entry": entry,
                     "net_live": lab["net_ret"], "exit_reason": lab["exit_reason"], "net_eod": net_eod, "net_60": net_60})
    return rows


def build_day(task) -> tuple[str, list[dict]]:
    trade_date, label_dict = task
    label = SimpleNamespace(**label_dict)
    files = g013._index_dir(DATA_ROOT / trade_date, "_10s.txt")
    latest = next((d for d in sorted((d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8),
                                     reverse=True) if any(d.glob("*_daily.csv"))), None)
    if not files or latest is None:
        return trade_date, []
    panel = g013.turnover_panel(latest, g013.EDGE_AI_DIR / "cache" / "turnover_panel.csv")
    daily_files = g013._index_dir(find_prev_data_dir(DATA_ROOT, trade_date), "_daily.csv")
    rows: list[dict] = []
    for code in g013.universe_for(trade_date, panel, UNIVERSE):
        f = files.get(code)
        if f is None or (f.name.count("_") >= 2 and is_fund_product(f.name.split("_")[1])):
            continue
        try:
            rows.extend(code_events(code, trade_date, f, daily_files.get(code), label))
        except Exception as exc:
            print(f"[WARN] {trade_date} {code}: {exc}", flush=True)
    return trade_date, rows


def split_days(days: list[str], holdout_frac: float = 0.3) -> tuple[set[str], set[str]]:
    n_hold = max(1, int(round(len(days) * holdout_frac))) if len(days) > 1 else 0
    return set(days[: len(days) - n_hold]), set(days[len(days) - n_hold:])


def report(ev: pd.DataFrame, days: list[str], label) -> str:
    disc, hold = split_days(days)
    lines = [f"# Entry-candidate event study ({days[0]} ~ {days[-1]}, {len(days)} days)", "",
             f"- live exits: +{label.tp:g}% / -{label.sl:g}% / {label.horizon} bars / 15:15, entry = next open + 1 tick, all costs in",
             f"- discovery = first {len(disc)} days, holdout = last {len(hold)} days; [..] = day-bootstrap 95 % interval of the mean",
             "- usable = the lower bound of a net mean is above 0 in both parts", ""]
    for col, title in (("net_live", "live exits"), ("net_eod", "held to 15:15"), ("net_60", "60 bars")):
        lines += [f"## {title} ({col})", "", "| candidate | part | events | per day | mean net % | 95 % | win | usable |",
                  "|---|---|---|---|---|---|---|---|"]
        for kind in KINDS:
            sub = ev[ev["kind"] == kind]
            lows = []
            rows = []
            for part, dset in (("discovery", disc), ("holdout", hold)):
                s = sub[sub["date"].isin(dset)]
                m, lo, hi, n = g055.boot_mean(s, col)
                lows.append(lo)
                win = float((s[col].dropna() > 0).mean()) if n else float("nan")
                rows.append(f"| {kind} | {part} | {n} | {n / max(1, len(dset)):.1f} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {win:.3f} |")
            usable = "yes" if all(x == x and x > 0 for x in lows) else "no"
            lines += [r + f" {usable} |" for r in rows]
        lines.append("")
    lines += ["## exit reasons under the live exits (all days)", "", "| candidate | tp | sl | time | eod |", "|---|---|---|---|---|"]
    for kind in KINDS:
        s = ev[ev["kind"] == kind]
        if len(s):
            r = s["exit_reason"].value_counts(normalize=True)
            lines.append(f"| {kind} | " + " | ".join(f"{r.get(k, 0.0):.3f}" for k in ("tp", "sl", "time", "eod")) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Event study of entry-candidate definitions (read-only)")
    lab = g013.live_label()
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--days", type=int, default=0, help="only the newest N days (0 = all)")
    p.add_argument("--tp", type=float, default=lab["tp"])
    p.add_argument("--sl", type=float, default=lab["sl"])
    p.add_argument("--horizon", type=int, default=int(lab["horizon"]))
    args = p.parse_args(argv)
    label = {"tp": args.tp, "sl": args.sl, "horizon": args.horizon, "slip_ticks": 1}
    days = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8
                  and g013._day_settled(d, 30))
    if args.days > 0:
        days = days[-args.days:]
    if not days:
        print("no settled data folder")
        return 2
    rows: list[dict] = []
    with Pool(max(1, args.workers)) as pool:
        for trade_date, part in pool.imap_unordered(build_day, [(d, label) for d in days]):
            rows.extend(part)
            print(f"  {trade_date}: {len(part)} events", flush=True)
    ev = pd.DataFrame(rows, columns=COLS).sort_values(["date", "kind", "code"]).reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "candidates.csv", index=False)
    text = report(ev, days, SimpleNamespace(**label))
    (out / "candidate_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
