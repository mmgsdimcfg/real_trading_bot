# -*- coding: utf-8 -*-
"""Entry-trigger relaxation study on the tick files (2026-11-12 사용자 요청: _009 트리거를 풀면 매수 횟수와 손익이 어떻게 되나).

Replays only the trigger of r005 _009 (not the gates after it) on the days that have g011 tick files:
  1m state   1-minute MA5 > MA20 on the last confirmed bar, run not older than `m1_age` seconds (0 = no limit),
             and the live price above the 1-minute MA20
  tick cross a 60- or 120-tick bar series whose MA5 crossed above MA20 at most `tick_age` seconds ago and is still above
  chase      live price at most `chase`% above the close of the earliest valid tick cross bar (None = off)
A variant switches parts off or widens them. Signals are checked every 10 seconds between 09:05 and 14:50; a buy is the
first check that passes while the symbol has no position, at the last trade price + 1 tick. The position is then closed by
the live-like exit of g027 (hard stop 1.7% after 4 min, profit lock +1.0% / 0.4%, 15:15) on the following 1-minute bars,
costs included - the same exit for every variant, so only the entry differs.

Not the live bot: the 3-minute context, market-regime, ATR, liquidity, Edge AI, spread, confirmation and order gates after
the trigger are not applied, tick bars start at 09:00 of each day plus the previous day's ticks as MA warm-up, and the
first minute after a buy is not in the exit path. Use the numbers to compare variants, not as a profit forecast.

  python g030_trigger_study.py [--workers 6]     -> runtime/edge_ai/timing_study/report_trigger.md

Update log
- [2026-11-12] type=feat - first version
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g013_edge_ai_dataset as g013  # noqa: E402
import g026_timing_study as g026  # noqa: E402
import g027_timing_grid as g027  # noqa: E402

TICK_ROOT = g013.DATA_ROOT / "ticks"
ENTRY_START_S = 9 * 3600 + 5 * 60
ENTRY_END_S = 14 * 3600 + 50 * 60
CHECK_STEP_S = 10
CONFIRM_DELAY_S = 5          # a 1-minute bar is used a few seconds after its minute ends (r003 CANDLE_CONFIRM_DELAY_SECONDS)

# name -> (use_1m, m1_age_s, use_tick, tick_age_s, chase_pct or None, description)
VARIANTS = {
    "live": (True, 900.0, True, 240.0, 0.5, "현행: 1분 상태(15분 이내) + 틱 교차(4분 이내) + 추격 0.5%"),
    "m1_age_off": (True, 0.0, True, 240.0, 0.5, "1분 상태 나이 제한 해제"),
    "tick_age_10m": (True, 900.0, True, 600.0, 0.5, "틱 교차 유효 4분 -> 10분"),
    "chase_1pct": (True, 900.0, True, 240.0, 1.0, "추격 허용 0.5% -> 1.0%"),
    "chase_off": (True, 900.0, True, 240.0, None, "추격 가드 끔"),
    "all_relaxed": (True, 0.0, True, 600.0, None, "위 세 가지 모두 완화"),
    "tick_only": (False, 0.0, True, 240.0, 0.5, "1분 조건 없이 틱 교차만"),
    "m1_only": (True, 900.0, False, 0.0, None, "틱 조건 없이 1분 상태만"),
    "trigger_off": (False, 0.0, False, 0.0, None, "트리거 끔 (보유가 없으면 바로 매수)"),
}


def _sec(hhmmss: pd.Series) -> np.ndarray:
    s = hhmmss.astype(str).str.zfill(6)
    return (s.str.slice(0, 2).astype(int) * 3600 + s.str.slice(2, 4).astype(int) * 60 + s.str.slice(4, 6).astype(int)).to_numpy()


def load_ticks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(second of day, price) of the continuous KRX session, in file order."""
    want = ("trade_time", "price", "venue")
    try:
        t = pd.read_csv(path, usecols=lambda c: c in want, dtype={"trade_time": str})
    except Exception:
        return np.array([]), np.array([])
    if "venue" in t.columns:
        t = t[t["venue"] == "KRX"]
    t = t[(t["trade_time"] >= "090000") & (t["trade_time"] < "152000") & (t["price"] > 0)]
    return _sec(t["trade_time"]), t["price"].to_numpy(dtype=np.float64)


def tick_runs(sec: np.ndarray, price: np.ndarray, size: int, warm: np.ndarray) -> dict:
    """Bars of `size` ticks (previous-day closes `warm` only feed the moving averages).
    -> bar close time (s), above flag (MA5 > MA20), and for every bar the time / close of the cross that started its run."""
    n = len(price) // size
    close = price[size - 1: n * size: size]
    t = sec[size - 1: n * size: size]
    allc = np.r_[warm, close]
    s = pd.Series(allc)
    ma5, ma20 = s.rolling(5).mean().to_numpy(), s.rolling(20).mean().to_numpy()
    above = (ma5 > ma20)[len(warm):]
    prev_above = np.r_[(ma5 > ma20)[len(warm) - 1] if len(warm) else False, above[:-1]]
    cross_t = np.full(n, np.nan)
    cross_c = np.full(n, np.nan)
    ct, cc = np.nan, np.nan
    for k in range(n):
        if above[k] and not prev_above[k]:
            ct, cc = t[k], close[k]
        elif not above[k]:
            ct, cc = np.nan, np.nan
        cross_t[k], cross_c[k] = ct, cc      # nan while below, or while a run began before today's first bar
    return {"t": t, "above": above, "cross_t": cross_t, "cross_c": cross_c}


def minute_state(arr: np.ndarray, prev: np.ndarray | None) -> dict:
    """1-minute MA5 / MA20 (previous-day bars as warm-up). arr = g026 bar rows (label minute, o, h, l, c, v)."""
    warm = prev[-g026.WARM_BARS:, 4] if prev is not None and len(prev) else np.array([])
    s = pd.Series(np.r_[warm, arr[:, 4]])
    ma5, ma20 = s.rolling(5).mean().to_numpy()[len(warm):], s.rolling(20).mean().to_numpy()[len(warm):]
    above = ma5 > ma20
    run_start = np.full(len(arr), np.nan)       # label second of the bar where the current "above" run began
    cur = np.nan
    for k in range(len(arr)):
        if above[k]:
            if not (k and above[k - 1]):
                cur = arr[k, 0] * 60.0
        else:
            cur = np.nan
        run_start[k] = cur
    return {"label_s": arr[:, 0] * 60.0, "above": above, "ma20": ma20, "ma5": ma5, "run_start": run_start}


def simulate_code(sec, price, runs, m1, arr, variant) -> list[tuple]:
    """Trades of one symbol-day for one variant: (entry second, entry price, net %, held bars, exit kind)."""
    use_1m, m1_age, use_tick, tick_age, chase, _ = variant
    P = np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], m1["ma5"], m1["ma20"], arr[:, 4]]).astype(np.float64)
    eod = np.flatnonzero(arr[:, 0] <= g026.EOD_MIN)
    last = int(eod[-1]) if len(eod) else len(arr) - 1
    prm = np.array(g027.LIVE, dtype=np.float64)
    out = []
    busy_until = -1.0
    if not len(sec):
        return out
    for T in range(max(ENTRY_START_S, int(sec[0]) + 1), ENTRY_END_S + 1, CHECK_STEP_S):
        if T < busy_until:
            continue
        i = int(np.searchsorted(sec, T, side="right")) - 1          # last trade at or before T
        if i < 0 or T - sec[i] > 120:                                # no fresh price
            continue
        px = price[i]
        k = int(np.searchsorted(m1["label_s"], T - CONFIRM_DELAY_S, side="right")) - 1   # last confirmed 1m bar
        if k < 0:
            continue
        if use_1m:
            if not m1["above"][k] or not (px > m1["ma20"][k]):
                continue
            if m1_age > 0 and not (T - m1["run_start"][k] <= m1_age):
                continue
        ref = np.nan
        if use_tick:
            ok = False
            for r in runs:
                j = int(np.searchsorted(r["t"], T, side="right")) - 1
                if j < 0 or not r["above"][j]:
                    continue
                ct = r["cross_t"][j]
                if ct == ct and T - ct <= tick_age:
                    ok = True
                    if not (ref == ref) or ct < ref_t:               # earliest valid cross is the chase reference
                        ref, ref_t = r["cross_c"][j], ct
            if not ok:
                continue
            if chase is not None and ref == ref and (px / ref - 1.0) * 100.0 > chase:
                continue
        s = int(np.searchsorted(m1["label_s"], T + 60, side="left"))  # first bar that starts after the buy
        if s >= last:
            break
        entry = px + g027._tick(px)
        net, held, _peak, kind = g027._sim_one(P, s, last - s + 1, entry, np.nan, prm)
        out.append((T, entry, float(net), int(held), int(kind)))
        busy_until = (arr[s + int(held), 0] + 1) * 60.0
    return out


def run_day(date: str) -> pd.DataFrame:
    day = g026.load_day(date)
    dates = sorted(p.stem for p in g026.BARS_DIR.glob("*.pkl"))
    prev_date = dates[dates.index(date) - 1] if date in dates and dates.index(date) > 0 else None
    prev = g026.load_day(prev_date) if prev_date else None
    rows = []
    for f in sorted((TICK_ROOT / date).glob("*.csv")):
        code = f.stem
        arr = day["bars"].get(code) if day else None
        if arr is None or len(arr) < 60:
            continue
        sec, price = load_ticks(f)
        if len(price) < 2000:                                         # too thin for 60/120-tick bars to mean anything
            continue
        pf = TICK_ROOT / prev_date / f.name if prev_date else None
        _ps, pprice = load_ticks(pf) if pf is not None and pf.is_file() else (np.array([]), np.array([]))
        runs = []
        for size in (60, 120):
            warm = pprice[size - 1: (len(pprice) // size) * size: size][-40:] if len(pprice) else np.array([])
            runs.append(tick_runs(sec, price, size, warm))
        m1 = minute_state(arr, prev["bars"].get(code) if prev else None)
        for name, variant in VARIANTS.items():
            for T, entry, net, held, kind in simulate_code(sec, price, runs, m1, arr, variant):
                rows.append((date, code, name, T, entry, net, held, kind))
    return pd.DataFrame(rows, columns=["date", "code", "variant", "sec", "entry", "net", "held", "kind"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dates", nargs="*")
    args = ap.parse_args()
    cached = {p.stem for p in g026.BARS_DIR.glob("*.pkl")}
    dates = sorted(p.name for p in TICK_ROOT.iterdir() if p.is_dir() and p.name in cached and (not args.dates or p.name in args.dates))
    with Pool(min(args.workers, max(1, len(dates)))) as pool:
        d = pd.concat(pool.map(run_day, dates), ignore_index=True)
    n_days = len(dates)
    code_days = d.groupby(["date", "code"]).ngroups
    base = d[d["variant"] == "live"]
    lines = [f"# g030 trigger study - {n_days} tick days ({dates[0]}~{dates[-1]}), {code_days} symbol-days", "",
             "Only the trigger differs between rows; the gates after it are not applied and the exit is the same live-like rule. "
             "net = mean % per trade after costs.", "",
             "| variant | 설명 | trades | /day | vs live | win | mean net | sum/day | days + | stop | lock | EOD |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in VARIANTS.items():
        t = d[d["variant"] == name]
        if t.empty:
            lines.append(f"| {name} | {v[5]} | 0 | | | | | | | | | |")
            continue
        by = t.groupby("date")["net"].sum()
        kinds = t["kind"].value_counts(normalize=True)
        lines.append(
            f"| {name} | {v[5]} | {len(t):,} | {len(t) / n_days:.0f} | x{len(t) / max(1, len(base)):.1f} | {(t['net'] > 0).mean() * 100:.0f}% | "
            f"{t['net'].mean():+.3f}% | {by.sum() / n_days:+.1f}% | {(by > 0).sum()}/{n_days} | {kinds.get(1, 0) * 100:.0f}% | "
            f"{kinds.get(2, 0) * 100:.0f}% | {kinds.get(4, 0) * 100:.0f}% |")
    lines += ["", "sum/day = sum of the trades' net % per day (each trade one equal slot), averaged over the days.", ""]
    # the extra trades a relaxed variant adds: entries that are not within 3 minutes of a live entry of the same symbol
    lines += ["## 완화로 새로 생기는 거래만 (같은 종목의 현행 진입 3분 이내가 아닌 진입)", "",
              "| variant | extra trades | win | mean net |", "|---|---|---|---|"]
    live_by = {k: g["sec"].to_numpy() for k, g in base.groupby(["date", "code"])}
    for name in VARIANTS:
        if name == "live":
            continue
        t = d[d["variant"] == name]
        if t.empty:
            continue
        keep = []
        for row in t.itertuples():
            ls = live_by.get((row.date, row.code))
            keep.append(ls is None or not np.any(np.abs(ls - row.sec) <= 180))
        x = t[np.array(keep)]
        if len(x):
            lines.append(f"| {name} | {len(x):,} | {(x['net'] > 0).mean() * 100:.0f}% | {x['net'].mean():+.3f}% |")
    text = "\n".join(lines) + "\n"
    out = g026.OUT_DIR / "report_trigger.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("->", out)


if __name__ == "__main__":
    main()
