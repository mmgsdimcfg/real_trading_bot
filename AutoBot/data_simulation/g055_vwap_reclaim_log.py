# -*- coding: utf-8 -*-
"""g055_vwap_reclaim_log.py - forward log of "plunge, then VWAP reclaimed and held" (no orders, no KIS call, no bot state).

Why: AutoBot/vwap_indicator_review_20261126.md 3-1 / 5. The only VWAP effect of the study was a hypothesis found after looking at
many cuts: among the stocks that fell most from the open, those holding above VWAP did better to 15:15 than those holding below
(+0.39 %p, in sample). Codex: test it on new days with a rule fixed in advance, one event per code-day, a matched control, the next
tradable price and costs. This file is that rule. It must not be tuned on the days it scores - a change makes a new RULE_VERSION and
starts the forward count again.

RULE_VERSION 1 (fixed 2026-11-27; days from FORWARD_FROM on are the test, earlier days are in-sample backfill)
  universe   g013.universe_for(D): 20-day mean turnover >= 3 bn KRW and close >= 1,000 as of D-1, every code (no sampling),
             ETF / ETN left out by name (r011.is_fund_product)
  bars       regular-session 1-minute bars with minute-END labels (g013.load_minute_bars + r011.session_raw_1min)
  VWAP       cumsum(close x volume) / cumsum(volume) from the first bar of the session
  plunge     the lowest low so far <= the day's open x (1 - 3 %)
  reclaim    a bar labelled 09:30 .. 13:30, plunge true, close > VWAP, the bar before it close <= VWAP
  confirm    the 30 bars after a reclaim bar all close > VWAP -> the event is the 30th bar (the first such per code-day)
  control    at the confirm bar's minute: every other universe code with plunge true whose last 30 closes are all < VWAP
  entry      open of the next bar + 1 tick.  exit: close of the last bar labelled <= 15:15 - 1 tick ("eod"), and the close 60 bars
             after the entry bar - 1 tick ("60").  net % after commission 0.015 % x 2 and sell tax 0.20 %
  also logged (not the test): the first reclaim bar of each code-day (kind "reclaim"), entry right after it

Primary measure: mean net_eod of the "confirm" events on forward days, and its excess over the same-minute control.
Decision (written before any forward day): with >= 40 forward days and >= 200 confirm events, the pattern becomes a watch-only
alert kind only if the day-bootstrap 95 % lower bound of BOTH mean net_eod and mean excess is above 0. Otherwise it is dropped.

Files (<runtime>/data/vwap_reclaim/): events.csv (one row per event, a date is replaced as a whole), report.md
Run:
  python g055_vwap_reclaim_log.py --date 20261127      one date (r021 postclose step "vwaplog") + the report
  python g055_vwap_reclaim_log.py --backfill           every data folder not logged yet (in-sample rows) + the report
  python g055_vwap_reclaim_log.py --report             report only
  [--out DIR] writes somewhere else (tests / trials)
Exit codes: 0 ok (also: no data folder for --date) / 2 no usable input.

Update log:
- [2026-11-27] type=feat owner=claude
    summary: 신규 - "급락 후 VWAP 재돌파·30분 유지" 전진 기록(사전에 고정한 규칙, 대조군, 다음 봉 체결 + 비용). 주문 없음.
    impact: tooling (r021 postclose 의 vwaplog 단계, 매매 판정 불변)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import os
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
from r008_strategy_v2 import krx_tick_size  # noqa: E402
from r011_edge_ai import is_fund_product, session_raw_1min  # noqa: E402

RULE_VERSION = 1
FORWARD_FROM = "20261127"
PLUNGE_PCT = 3.0
HOLD_BARS = 30
RECLAIM_FROM, RECLAIM_TO = pd.Timestamp("09:30").time(), pd.Timestamp("13:30").time()
EOD = pd.Timestamp("15:15").time()
FEE, TAX = 0.00015, 0.0020
MIN_FORWARD_DAYS, MIN_FORWARD_EVENTS = 40, 200
UNIVERSE = SimpleNamespace(min_turnover=3e9, min_price=1000, max_codes=0)
DATA_ROOT = Path(cfg.RUNTIME_DATA_DIR)
OUT_DIR = DATA_ROOT / "vwap_reclaim"
COLS = ["date", "code", "name", "kind", "ts", "open_pct", "low_pct", "vwap_gap_pct", "entry", "net_eod", "net_60", "ctl_n",
        "ctl_net_eod", "ctl_net_60", "forward", "rule"]


def ticks(px: np.ndarray) -> np.ndarray:
    """KRX tick size per price (NaN stays NaN)"""
    return np.array([float(krx_tick_size(x)) if x == x and x > 0 else np.nan for x in np.asarray(px, dtype=float)], dtype=float)


def net_pct(entry: np.ndarray, exit_px: np.ndarray) -> np.ndarray:
    """net % of buying at `entry` (slip included) and selling at `exit_px` - 1 tick, after commission and tax"""
    ex = exit_px - ticks(exit_px)
    return ((ex - entry) - (entry * FEE + ex * (FEE + TAX))) / entry * 100.0


def code_frame(path: Path, trade_date: str) -> pd.DataFrame | None:
    """per bar: the state the rule reads and what an entry right after the bar would have made"""
    raw = session_raw_1min(g013.load_minute_bars(path), pd.Timestamp(trade_date))
    if raw is None or len(raw) < HOLD_BARS * 2:
        return None
    o, lo, c, v = (raw[k].to_numpy(float) for k in ("open", "low", "close", "volume"))
    cv = np.cumsum(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cv > 0, np.cumsum(c * v) / cv, np.nan)
    above = c > vwap
    below = c < vwap
    n = len(c)
    run_above = pd.Series(above.astype(float)).rolling(HOLD_BARS, min_periods=HOLD_BARS).sum().to_numpy() == HOLD_BARS
    run_below = pd.Series(below.astype(float)).rolling(HOLD_BARS, min_periods=HOLD_BARS).sum().to_numpy() == HOLD_BARS
    plunge = np.minimum.accumulate(lo) <= o[0] * (1 - PLUNGE_PCT / 100.0)
    times = raw.index.time
    in_win = np.array([RECLAIM_FROM <= t <= RECLAIM_TO for t in times])
    prev_not_above = np.r_[False, ~above[:-1]]
    reclaim = in_win & plunge & above & prev_not_above
    # entry after bar i = open of bar i+1 + 1 tick
    nxt = np.r_[o[1:], np.nan]
    entry = nxt + ticks(nxt)
    eod_idx = np.flatnonzero(np.array([t <= EOD for t in times]))
    eod_close = c[eod_idx[-1]] if len(eod_idx) else np.nan
    last = eod_idx[-1] if len(eod_idx) else -1
    net_eod = net_pct(entry, np.full(n, eod_close))
    net_eod[np.arange(n) + 1 > last] = np.nan            # the entry bar must come before the 15:15 bar closes
    c60 = np.r_[c[61:], np.full(min(61, n), np.nan)][:n]  # close 60 bars after the entry bar (i+1)
    net_60 = net_pct(entry, c60)
    return pd.DataFrame({"open_pct": (c / o[0] - 1) * 100, "low_pct": (np.minimum.accumulate(lo) / o[0] - 1) * 100,
                         "vwap_gap_pct": (c / vwap - 1) * 100, "plunge": plunge, "reclaim": reclaim, "run_above": run_above,
                         "run_below": run_below, "entry": entry, "net_eod": net_eod, "net_60": net_60}, index=raw.index)


def events_of(frame: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """[(kind, bar)] of one code-day: the first reclaim, and the first reclaim that held HOLD_BARS bars (its confirm bar)"""
    idx = np.flatnonzero(frame["reclaim"].to_numpy())
    if not len(idx):
        return []
    out = [("reclaim", frame.index[idx[0]])]
    run = frame["run_above"].to_numpy()
    for i in idx:
        j = i + HOLD_BARS
        if j < len(frame) and run[j]:        # bars i+1 .. i+HOLD_BARS all above VWAP
            out.append(("confirm", frame.index[j]))
            break
    return out


def build_day(trade_date: str) -> tuple[str, pd.DataFrame | None]:
    day_dir = DATA_ROOT / trade_date
    files = g013._index_dir(day_dir, "_10s.txt")
    if not files:
        return trade_date, None
    latest = next((d for d in sorted((d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8),
                                     reverse=True) if any(d.glob("*_daily.csv"))), None)
    if latest is None:
        return trade_date, None
    panel = g013.turnover_panel(latest, g013.EDGE_AI_DIR / "cache" / "turnover_panel.csv")
    frames: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    for code in g013.universe_for(trade_date, panel, UNIVERSE):
        f = files.get(code)
        if f is None:
            continue
        name = f.name.split("_")[1] if f.name.count("_") >= 2 else ""
        if is_fund_product(name):
            continue
        fr = code_frame(f, trade_date)
        if fr is not None:
            frames[code], names[code] = fr, name
    rows = []
    for code, fr in frames.items():
        for kind, ts in events_of(fr):
            r = fr.loc[ts]
            ctl_e, ctl_6 = [], []
            if kind == "confirm":
                for other, of in frames.items():
                    if other == code or ts not in of.index:
                        continue
                    q = of.loc[ts]
                    if q["plunge"] and q["run_below"] and q["entry"] == q["entry"]:
                        ctl_e.append(q["net_eod"])
                        ctl_6.append(q["net_60"])
            rows.append({"date": trade_date, "code": code, "name": names[code], "kind": kind, "ts": ts.strftime("%H:%M"),
                         "open_pct": round(float(r["open_pct"]), 3), "low_pct": round(float(r["low_pct"]), 3),
                         "vwap_gap_pct": round(float(r["vwap_gap_pct"]), 3), "entry": r["entry"],
                         "net_eod": round(float(r["net_eod"]), 4), "net_60": round(float(r["net_60"]), 4), "ctl_n": len(ctl_e),
                         "ctl_net_eod": round(float(np.nanmean(ctl_e)), 4) if ctl_e else np.nan,
                         "ctl_net_60": round(float(np.nanmean(ctl_6)), 4) if ctl_6 and not np.all(np.isnan(ctl_6)) else np.nan,
                         "forward": int(trade_date >= FORWARD_FROM), "rule": RULE_VERSION})
    return trade_date, pd.DataFrame(rows, columns=COLS)


def load_events(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "events.csv"
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(p, dtype={"date": str, "code": str, "ts": str}, encoding="utf-8-sig")


def save_events(out_dir: Path, ev: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "events.csv.tmp"
    ev.sort_values(["date", "ts", "code", "kind"]).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, out_dir / "events.csv")


def boot_mean(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 11) -> tuple[float, float, float, int]:
    """(mean, day-bootstrap 2.5 %, 97.5 %, rows used)"""
    d = df.dropna(subset=[col])
    if d.empty:
        return float("nan"), float("nan"), float("nan"), 0
    g = d.groupby("date")[col].agg(["sum", "count"]).to_numpy()
    if len(g) < 2:
        return float(d[col].mean()), float("nan"), float("nan"), len(d)
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    s = g[idx].sum(axis=1)
    m = s[:, 0] / np.maximum(1, s[:, 1])
    return float(d[col].mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)


def report(ev: pd.DataFrame, logged_days: dict[str, int]) -> str:
    lines = [f"# VWAP reclaim forward log (rule {RULE_VERSION}, forward from {FORWARD_FROM})", "",
             f"Rule: plunge {PLUNGE_PCT:g}% from the open, close back above VWAP (09:30-13:30), {HOLD_BARS} bars held above. "
             "Entry next open + 1 tick, exit 15:15 close - 1 tick, net of costs. Control: plunged codes holding below VWAP at "
             "the same minute.", ""]
    ev = ev[ev["rule"] == RULE_VERSION].copy()
    ev["excess_eod"] = ev["net_eod"] - ev["ctl_net_eod"]
    ev["excess_60"] = ev["net_60"] - ev["ctl_net_60"]
    for fwd, title in ((1, "Forward days (the test)"), (0, "Backfill (in sample - not evidence)")):
        days = sum(1 for d, f in logged_days.items() if f == fwd)
        lines += [f"## {title}: {days} day(s) logged", "",
                  "| kind | events | measure | n | mean % | 95% (day bootstrap) | median % | > 0 % |", "|---|---|---|---|---|---|---|---|"]
        for kind in ("confirm", "reclaim"):
            d = ev[(ev["forward"] == fwd) & (ev["kind"] == kind)]
            for col in (("net_eod", "net_60", "excess_eod", "excess_60") if kind == "confirm" else ("net_eod", "net_60")):
                m, lo, hi, n = boot_mean(d, col)
                x = d[col].dropna()
                f3 = lambda v: "-" if v != v else f"{v:+.3f}"  # noqa: E731
                lines.append(f"| {kind} | {len(d)} | {col} | {n} | {f3(m)} | [{f3(lo)}, {f3(hi)}] | "
                             f"{f3(float(x.median()) if n else float('nan'))} | {'-' if not n else format((x > 0).mean() * 100, '.1f')} |")
        lines.append("")
    f = ev[(ev["forward"] == 1) & (ev["kind"] == "confirm")]
    fdays = sum(1 for v in logged_days.values() if v == 1)
    lines.append("## Decision")
    if fdays < MIN_FORWARD_DAYS or len(f) < MIN_FORWARD_EVENTS:
        lines.append(f"Not yet: {fdays} / {MIN_FORWARD_DAYS} forward days, {len(f)} / {MIN_FORWARD_EVENTS} confirm events.")
    else:
        a, b = boot_mean(f, "net_eod"), boot_mean(f, "excess_eod")
        ok = a[1] > 0 and b[1] > 0
        lines.append(f"{'ADOPT as a watch-only alert kind' if ok else 'DROP'}: net_eod lower bound {a[1]:+.3f}, excess lower bound {b[1]:+.3f} "
                     "(both must be above 0).")
    return "\n".join(lines) + "\n"


def days_file(out_dir: Path) -> Path:
    return out_dir / "days.csv"


def load_days(out_dir: Path) -> dict[str, int]:
    p = days_file(out_dir)
    if not p.exists():
        return {}
    d = pd.read_csv(p, dtype={"date": str})
    d = d[d["rule"] == RULE_VERSION]
    return dict(zip(d["date"], d["forward"].astype(int)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out_dir = Path(args.out) if args.out else OUT_DIR
    logged = load_days(out_dir)
    todo: list[str] = []
    if args.date:
        if not (DATA_ROOT / args.date).is_dir():
            print(f"no data folder for {args.date} - nothing logged")
            return 0
        todo = [args.date]
    elif args.backfill:
        todo = [d.name for d in sorted(DATA_ROOT.iterdir()) if d.is_dir() and d.name.isdigit() and len(d.name) == 8
                and d.name not in logged]
    elif not args.report:
        ap.error("--date, --backfill or --report")
    ev = load_events(out_dir)
    if todo:
        done: list[pd.DataFrame] = []
        if len(todo) == 1:
            results = [build_day(todo[0])]
        else:
            with Pool(max(1, args.workers)) as pool:
                results = []
                for r in pool.imap_unordered(build_day, todo):
                    print(f"  {r[0]}: {'no data' if r[1] is None else len(r[1])} event(s)", flush=True)
                    results.append(r)
        for d, rows in results:
            if rows is None:
                continue
            logged[d] = int(d >= FORWARD_FROM)
            done.append(rows)
        if not done and args.date:
            print(f"no usable 1-minute files / daily files for {args.date}")
            return 2
        dates = {d for d, r in results if r is not None}
        ev = pd.concat([ev[~(ev["date"].isin(dates) & (ev["rule"] == RULE_VERSION))]] + [x for x in done if not x.empty],
                       ignore_index=True) if done else ev
        save_events(out_dir, ev)
        pd.DataFrame({"date": sorted(logged), "forward": [logged[d] for d in sorted(logged)], "rule": RULE_VERSION}).to_csv(
            days_file(out_dir), index=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = report(ev, logged)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
