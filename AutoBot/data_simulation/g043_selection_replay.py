# -*- coding: utf-8 -*-
"""g043_selection_replay.py - point-in-time replay of the g002 selection over a date range (filter redesign U1).

Why: AutoBot/selection_filter_redesign_20261119.md - compare the current selection (V0) with redesigned versions (V1~V4) on the
same data, and measure what each hard filter really removes ("release one filter and rescan").

What: for every date folder in runtime/data (--start..--end) whose collection is complete (daily.csv >= --min-complete x 10s files),
run g002 scan() exactly as the live scan does (same code, same folder = data known on that day; daily.csv in each folder ends on
that date). One process per date (--workers), g002 itself runs sequentially inside (G002_WORKERS=1).
  --release REASON  : the same scan with one hard filter switched off (see RELEASE below). Patches are applied inside each worker
                      process only - g002 is not modified.
  --variant v0      : current g002 (later variants are g002 profiles, U3~U6).
Labels (outcomes, not inputs): runtime/data/daily_history/<code>.csv - D+1 open->close, D+1/D+3/D+5 close/close(D) returns.
The universe benchmark of a day = mean over the tradable universe (20d avg turnover >= 5bn KRW, no structural exclusion).

Output: <runtime>/research/10_selection/rows_<tag>.jsonl.gz  one row per (date, code) with fail_reasons / soft_flags / tier / picked
        <runtime>/research/10_selection/summary_<tag>.txt    counts per day, filter totals / solo, returns of picks vs universe,
                                                             first 70% of days (train) and the rest (holdout) separately
Run:
  python g043_selection_replay.py --start 20260629 --end 20261117                 # V0
  python g043_selection_replay.py --start 20260629 --end 20261117 --release down_trend
  python g043_selection_replay.py --report rows_v0.jsonl.gz rows_v0__no_down_trend.jsonl.gz   # side-by-side summary

Update log:
- [2026-11-28] type=fix owner=claude
    summary: 다음 거래일을 봇의 거래일 달력(r021.is_trading_day)으로 잡는다. 이전에는 종목 이력의 다음 행 / 다음 데이터 폴더라
      거래정지 / 수집 누락일이 있으면 그 다음 날 결과가 '다음 날'로 들어갔다(이제 결측으로 빠진다).
      --full-rows: 후보의 모든 단순 필드를 rows 파일에 남긴다(규칙 연구용).
    impact: research tooling
    compatibility: backward-compatible (기본 출력 형식 동일, 결측 처리만 달라짐)
- [2026-11-19] type=feat owner=claude
    summary: 신규 - 선정 필터 구조 개선 U1(재현 하네스). g002 미변경, 연구용 출력만.
    impact: research tooling (runtime/research/10_selection)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))
os.environ.setdefault("G002_WORKERS", "1")  # one process per date; no nested pools

import g002_data_scan_trade_candidates as g002  # noqa: E402

RESEARCH_DIR = g002.RUNTIME_DATA_DIR.parent / "research" / "10_selection"
HISTORY_DIR = g002.RUNTIME_DATA_DIR / "daily_history"
TRADABLE_MIN_AMOUNT = 5_000_000_000
STRUCTURAL = {"price_floor", "price_ceiling", "preferred_or_spac_excluded", "derivative_etf_excluded",
              "admin_issue_or_halt_excluded", "pattern_unknown"}
TRAIN_FRAC = 0.7
KEEP = ("code", "name", "eligible", "picked", "skip_reason", "fail_reasons", "soft_flags", "pattern_tier", "trend_state", "score",
        "amount_ma20", "price", "close_ma20_ratio", "buy_signal_count", "intraday_atr3_med_pct", "intraday_atr3_status",
        "is_last_bearish", "market", "sector")

# ---------------------------------------------------------------------------------------------------------------------------
# Releasing one hard filter (worker-local patches). Non-pattern reasons are stripped from the candidate after the check that
# adds them; pattern reasons are switched off inside _pattern_assessment so the tier logic after them still runs.
# pattern_below_ma20 is the definition of the trend pattern and is not releasable.
# ---------------------------------------------------------------------------------------------------------------------------
STRIP_AFTER_EVAL = {
    "price_floor", "price_ceiling", "preferred_or_spac_excluded", "derivative_etf_excluded", "admin_issue_or_halt_excluded",
    "down_trend", "flat_trend", "box_range_no_progress", "bearish_candle_dominant", "linreg_long_term_downtrend",
    "upperlimit_streak_crash", "big_bearish_candle",
}
PATTERN_CONST = {
    "pattern_ma120_falling": {"PATTERN_MA120_MIN_SLOPE": -math.inf},
    "pattern_no_pullback_base": {"PATTERN_BASE_MAX_VS_MA20": math.inf},
    "pattern_overheated": {"PATTERN_MAX_VS_MA20": math.inf, "PATTERN_MAX_RSI": math.inf,
                           "PATTERN_MAX_MA60_SLOPE20": None, "PATTERN_MAX_RET_20D": None},
    "pattern_few_buy_signals": {"PATTERN_MIN_SIGNALS": 0},
    "pattern_bearish_candles": {"PATTERN_MAX_BEAR5": 99},
    "pattern_b_extended": {"PATTERN_B_MAX_VS_MA20": math.inf},
}
PATTERN_FEATURE = {  # feature overrides handed to _pattern_assessment only
    "pattern_below_long_ma": {"MA_60": None, "MA_120": None},
    "pattern_ma20_falling": {"ma20_slope5": None},
    "pattern_no_bullish_close": {"weak_close": False},
}
RELEASABLE = sorted(STRIP_AFTER_EVAL | set(PATTERN_CONST) | set(PATTERN_FEATURE)
                    | {"liquidity_below_market_dual", "low_intraday_atr", "intraday_atr_unknown", "low_score"})


def _strip(row: dict, reason: str) -> None:
    row["fail_reasons"] = [r for r in row.get("fail_reasons", []) if r != reason]


def apply_release(reason: str | None) -> None:
    if not reason:
        return
    if reason not in RELEASABLE:
        raise SystemExit(f"--release {reason}: not releasable ({', '.join(RELEASABLE)})")
    if reason in STRIP_AFTER_EVAL:
        orig_eval = g002.evaluate_candidate

        def evaluate_candidate(*a, **kw):
            cand = orig_eval(*a, **kw)
            _strip(cand, reason)
            return cand
        g002.evaluate_candidate = evaluate_candidate
    elif reason in PATTERN_CONST:
        for k, v in PATTERN_CONST[reason].items():
            setattr(g002, k, v)
    elif reason in PATTERN_FEATURE:
        orig_pat = g002._pattern_assessment

        def _pattern_assessment(price, f):
            return orig_pat(price, {**f, **PATTERN_FEATURE[reason]} if f else f)
        g002._pattern_assessment = _pattern_assessment
    elif reason == "liquidity_below_market_dual":
        orig_liq = g002.apply_market_relative_liquidity_filters

        def apply_market_relative_liquidity_filters(candidates, market_map):
            info = orig_liq(candidates, market_map)
            for row in candidates:
                _strip(row, reason)
            return info
        g002.apply_market_relative_liquidity_filters = apply_market_relative_liquidity_filters
    elif reason in ("low_intraday_atr", "intraday_atr_unknown"):
        orig_apply = g002._apply_intraday_metrics

        def _apply_intraday_metrics(candidate, metrics):
            orig_apply(candidate, metrics)
            _strip(candidate, reason)
        g002._apply_intraday_metrics = _apply_intraday_metrics
    elif reason == "low_score":
        g002.SCORE_CUTOFF = -1e9
        g002.SCORE_CUTOFF_V2 = -1e9


# ---------------------------------------------------------------------------------------------------------------------------
# Dates and scanning
# ---------------------------------------------------------------------------------------------------------------------------
def list_dates(start: str, end: str, min_complete: float) -> tuple[list[str], list[tuple[str, str]]]:
    root = g002.RUNTIME_DATA_DIR
    ok, skipped = [], []
    for d in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8):
        if not (start <= d <= end):
            continue
        names = os.listdir(root / d)
        n_daily = sum(n.endswith("_daily.csv") for n in names)
        n_10s = sum(n.endswith(".txt") and not n.startswith("_") for n in names)
        if n_10s == 0 or n_daily < min_complete * n_10s:
            skipped.append((d, f"daily.csv {n_daily} / 10s {n_10s}"))
            continue
        ok.append(d)
    return ok, skipped


_BASE_LIST = None


# ---------------------------------------------------------------------------------------------------------------------------
# [2026-11-19] --cache: per-date pickle of what scan() reads per symbol (daily.csv frame, the target-day daily bar from the 10s file,
# the intraday metrics), built once with the original g002 loaders, so later variants skip ~3,000 CSV parses per date.
# Off by default - a run without --cache reads the files exactly as before.
# ---------------------------------------------------------------------------------------------------------------------------
CACHE_DIR = RESEARCH_DIR / "cache"
_ORIG = {"daily": g002._load_daily_csv, "bars": g002.build_daily_bars, "intraday": g002._load_intraday_metrics}
_C: dict = {}
_C_DATE: str | None = None


def _stock_list(d: str):
    global _BASE_LIST
    root = g002.RUNTIME_DATA_DIR
    if _BASE_LIST is None:
        _BASE_LIST = g002.filter_stock_list_by_existing_data(g002.load_symbols(), root, verbose=False)
    return g002.filter_stock_list_by_min_bars(_BASE_LIST, root, target_date_str=d, min_bars=g002.MIN_REQUIRED_BARS, verbose=False)


def build_cache_date(d: str) -> str:
    import pickle
    path = CACHE_DIR / f"{d}.pkl"
    if path.exists():
        return f"{d} cached"
    root = g002.RUNTIME_DATA_DIR
    data = {}
    for code, _name in _stock_list(d):
        hist = _ORIG["daily"](code, root, d)
        bars = _ORIG["bars"](root, code, d, single_date_only=True)
        met = _ORIG["intraday"](root, d, code) if bars is not None else None
        data[code] = (hist, bars, met)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return f"{d} built {len(data)}"


def _c_load_daily_csv(code, data_root, target_date_str):
    if target_date_str != _C_DATE or code not in _C:
        return _ORIG["daily"](code, data_root, target_date_str)
    h = _C[code][0]
    return None if h is None else h.copy()


def _c_build_daily_bars(data_root, code, target_date_str=None, lookback=g002.DAILY_LOOKBACK, single_date_only=False):
    if not single_date_only or target_date_str != _C_DATE or code not in _C:
        return _ORIG["bars"](data_root, code, target_date_str, lookback=lookback, single_date_only=single_date_only)
    b = _C[code][1]
    return None if b is None else b.copy()


def _c_load_intraday_metrics(data_root, target_date_str, code):
    if target_date_str != _C_DATE or code not in _C or _C[code][2] is None:
        return _ORIG["intraday"](data_root, target_date_str, code)
    return dict(_C[code][2])


def _use_cache(d: str) -> None:
    global _C, _C_DATE
    import pickle
    build_cache_date(d)
    with open(CACHE_DIR / f"{d}.pkl", "rb") as fh:
        _C = pickle.load(fh)
    _C_DATE = d


_CACHE_ON = False


def _worker_init(release: str | None, cache: bool = False) -> None:
    global _CACHE_ON
    apply_release(release)
    if cache:
        _CACHE_ON = True
        g002._load_daily_csv = _c_load_daily_csv
        g002.build_daily_bars = _c_build_daily_bars
        g002._load_intraday_metrics = _c_load_intraday_metrics


def scan_date(d: str) -> list[dict]:
    root = g002.RUNTIME_DATA_DIR
    if _CACHE_ON:
        _use_cache(d)
    sl = _stock_list(d)
    with contextlib.redirect_stdout(io.StringIO()):
        res = g002.scan(root, datetime.strptime(d, "%Y%m%d"), config=g002.DEFAULT_CONFIG, return_details=True,
                        verbose=False, stock_list=sl)
    out = []
    for row in res["all_rows"]:
        r = {"date": d, **{k: row.get(k) for k in KEEP}}
        if os.environ.get("G043_FULL_ROWS") == "1":   # [2026-11-28] --full-rows: every plain field of the candidate (rule studies)
            for k, v in row.items():
                if k not in r and (v is None or isinstance(v, (bool, int, float, str))):
                    r[k] = v
            for k, v in (row.get("ind") or {}).items():
                if v is None or isinstance(v, (bool, int, float, str)):
                    r[f"ind_{k}"] = v
            r["score_gains"] = {g["key"]: g["value"] for g in (row.get("score_breakdown") or {}).get("gains", [])}
            r["score_penalties"] = {g["key"]: g["value"] for g in (row.get("score_breakdown") or {}).get("penalties", [])}
        r["picked"] = bool(r["picked"])
        r["fallback"] = "fallback_selected" in (row.get("soft_flags") or [])
        out.append(r)
    return out


def run_scan(args) -> Path:
    dates, skipped = list_dates(args.start, args.end, args.min_complete)
    tag = args.variant + (f"__no_{args.release}" if args.release else "") + (f"__{args.suffix}" if args.suffix else "")
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out = RESEARCH_DIR / f"rows_{tag}.jsonl.gz"
    print(f"[g043] {tag}: {len(dates)} dates ({dates[0] if dates else '-'} ~ {dates[-1] if dates else '-'}), "
          f"skipped {len(skipped)}: {', '.join(d for d, _ in skipped)}", flush=True)
    t0 = time.time()
    results: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(args.release, args.cache)) as pool:
        for d, rows in zip(dates, pool.map(scan_date, dates)):
            results[d] = rows
            if not args.quiet:
                print(f"  {d} rows {len(rows)} picked {sum(r['picked'] for r in rows)} ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for d in dates:
            for r in results[d]:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[g043] wrote {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Labels and summary
# ---------------------------------------------------------------------------------------------------------------------------
_HIST: dict[str, dict] = {}


def _history(code: str) -> dict | None:
    """{date: (open, close)} and the sorted date list of a code (daily_history)."""
    if code in _HIST:
        return _HIST[code]
    p = HISTORY_DIR / f"{code}.csv"
    h = None
    if p.exists():
        import pandas as pd
        df = pd.read_csv(p, usecols=["date", "open", "close"], dtype={"date": str})
        df = df[(df["close"] > 0) & (df["open"] > 0)]
        h = {"dates": df["date"].tolist(), "open": df["open"].astype(float).tolist(), "close": df["close"].astype(float).tolist()}
        h["idx"] = {d: i for i, d in enumerate(h["dates"])}
    _HIST[code] = h
    return h


_TDAY: dict[tuple[str, int], str | None] = {}


def trading_day_after(d: str, k: int = 1) -> str | None:
    """[2026-11-28] the k-th trading day after d by the bot's calendar (r021.is_trading_day). Was: the symbol's own next row /
    the next data folder - a halted or uncollected day made the day after it count as 'the next day'."""
    key = (d, k)
    if key not in _TDAY:
        from datetime import timedelta
        from r021_jobs import is_trading_day
        cur, left = datetime.strptime(d, "%Y%m%d"), k
        for _ in range(40):
            cur += timedelta(days=1)
            if is_trading_day(cur.strftime("%Y%m%d")):
                left -= 1
                if left == 0:
                    break
        _TDAY[key] = cur.strftime("%Y%m%d") if left == 0 else None
    return _TDAY[key]


def labels(code: str, d: str) -> dict:
    """forward results of a pick of date d; a label whose trading day has no row for the symbol (halt, no data) is left out"""
    h = _history(code)
    if not h or d not in h["idx"]:
        return {}
    c0 = h["close"][h["idx"][d]]
    out = {}
    for k in (1, 3, 5):
        j = h["idx"].get(trading_day_after(d, k) or "")
        if j is None:
            continue
        if k == 1:
            out["d1_oc"] = h["close"][j] / h["open"][j] - 1.0
        out[f"d{k}"] = h["close"][j] / c0 - 1.0
    return out


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def tradable(r: dict) -> bool:
    return (r.get("amount_ma20") or 0) >= TRADABLE_MIN_AMOUNT and not (set(r.get("fail_reasons") or []) & STRUCTURAL) \
        and not r.get("skip_reason")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _med(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


def summarize(path: Path) -> str:
    rows = load_rows(path)
    dates = sorted({r["date"] for r in rows})
    n_train = max(1, int(round(len(dates) * TRAIN_FRAC)))
    split = {d: ("train" if i < n_train else "holdout") for i, d in enumerate(dates)}
    for r in rows:
        r.update(labels(r["code"], r["date"]))
    lines = [f"# {path.name}", f"dates {len(dates)}: {dates[0]} ~ {dates[-1]}  train {dates[0]}~{dates[n_train - 1]} "
             f"/ holdout {dates[n_train] if n_train < len(dates) else '-'}~{dates[-1]}", ""]
    D = len(dates)
    ev = [r for r in rows if not r.get("skip_reason")]
    tr = [r for r in rows if tradable(r)]
    pk = [r for r in rows if r["picked"]]
    lines.append("## counts per day")
    lines.append(f"rows {len(rows) / D:.0f}  evaluated {len(ev) / D:.0f}  tradable {len(tr) / D:.1f}  "
                 f"tier A {sum(r['pattern_tier'] == 'A' for r in ev) / D:.1f}  tier B {sum(r['pattern_tier'] == 'B' for r in ev) / D:.1f}  "
                 f"(tradable A/B {sum(bool(r['pattern_tier']) for r in tr) / D:.1f})")
    lines.append(f"eligible {sum(bool(r['eligible']) for r in ev) / D:.1f}  picked {len(pk) / D:.1f} "
                 f"(strict {sum(not r['fallback'] for r in pk) / D:.1f} + fallback {sum(r['fallback'] for r in pk) / D:.1f})")
    lines.append("")
    tot, solo = Counter(), Counter()
    for r in ev:
        f = set(r["fail_reasons"] or [])
        tot.update(f)
        if len(f) == 1:
            solo.update(f)
    lines.append("## hard filters per day (all evaluated rows): total / solo")
    lines.extend(f"  {k:<34} {v / D:8.1f} {solo[k] / D:7.2f}" for k, v in tot.most_common())
    lines.append("")
    lines.append("## returns (%) - mean / median / win% / n ; excess = minus the tradable-universe mean of the same day")
    for part in ("train", "holdout", "all"):
        sel = (lambda r: True) if part == "all" else (lambda r, p=part: split[r["date"]] == p)
        uni = {}
        for d in dates:
            if part != "all" and split[d] != part:
                continue
            for key in ("d1_oc", "d1", "d3", "d5"):
                xs = [r[key] for r in tr if r["date"] == d and key in r]
                uni[(d, key)] = _mean(xs)
        lines.append(f"[{part}]")
        for name, grp in (("picked", pk), ("picked strict", [r for r in pk if not r["fallback"]]),
                          ("picked fallback", [r for r in pk if r["fallback"]]),
                          ("tradable tier A/B", [r for r in tr if r["pattern_tier"]]), ("tradable universe", tr)):
            cells = []
            for key in ("d1_oc", "d1", "d3", "d5"):
                xs = [r[key] for r in grp if sel(r) and key in r]
                ex = [r[key] - uni[(r["date"], key)] for r in grp if sel(r) and key in r and not math.isnan(uni.get((r["date"], key), float("nan")))]
                cells.append(f"{key} {_mean(xs) * 100:+.2f}/{_med(xs) * 100:+.2f}/{(sum(x > 0 for x in xs) / len(xs) * 100 if xs else float('nan')):.0f}%"
                             f"/n{len(xs)} ex{_mean(ex) * 100:+.2f}")
            lines.append(f"  {name:<18} " + "  ".join(cells))
    lines.append("")
    lines.extend(trade_report(rows, dates, split, tr, pk))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------------
# U2 - live-like trade simulation of the picks on the next trading day (1-minute raw bars of the g001 10s file).
# Entry is a fixed time (the live intraday trigger was no better than random in runtime/research 02/05/07), so the comparison
# measures the selection only. Exit = r001 live settings (2026-11-19): hard stop -3.5% after 240 s, profit lock armed at
# +1.0% peak with 0.4% giveback (full), end of day 15:15. Not modelled: staged TP, adaptive exit, adds, the entry gates.
# Path inside a bar: bullish o->l->h->c, bearish o->h->l->c (conservative); a level a bar gaps through fills at the open.
# Costs: fee 0.015% per side, sell tax 0.20%, market fills (entry / stop / lock / eod) 1 tick slippage.
# ---------------------------------------------------------------------------------------------------------------------------
ENTRY_HHMM = "09:05"
EOD_HHMM = "15:15"
HARD_STOP = 0.035
HARD_STOP_HOLD_MIN = 4
LOCK_ARM = 0.010
LOCK_GIVEBACK = 0.004
FEE = 0.00015
TAX = 0.0020
RANDOM_PER_DAY = 20


def krx_tick(p: float) -> float:
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return t
    return 1000


_FOLDERS: list[str] | None = None


def next_folder(d: str) -> str | None:
    global _FOLDERS
    if _FOLDERS is None:
        root = g002.RUNTIME_DATA_DIR
        _FOLDERS = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8)
    # [2026-11-28] the next TRADING day, and only when its folder was collected (was: the next folder, whatever day it was)
    nd = trading_day_after(d, 1)
    return nd if nd in _FOLDERS else None


def minute_bars(code: str, d: str):
    """Regular-session raw 1-minute bars [(hhmm, o, h, l, c)] of a code on date d, or None."""
    import pandas as pd
    path = g002.resolve_symbol_file(g002.RUNTIME_DATA_DIR / d, code)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c in {"timestamp", "datetime", "time", "open", "high", "low", "close", "raw_bar"})
    except Exception:
        return None
    tcol = next((c for c in ("timestamp", "datetime", "time") if c in df.columns), None)
    if tcol is None or "raw_bar" not in df.columns:
        return None
    df = df[pd.to_numeric(df["raw_bar"], errors="coerce") == 1]
    hhmm = pd.to_datetime(df[tcol], errors="coerce").dt.strftime("%H:%M")
    m = (hhmm >= "09:00") & (hhmm < "15:30")
    df, hhmm = df[m], hhmm[m]
    vals = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    out = [(t, *v) for t, v in zip(hhmm.tolist(), vals.tolist()) if all(x == x and x > 0 for x in v)]
    return out or None


def simulate(bars) -> dict | None:
    i0 = next((i for i, b in enumerate(bars) if b[0] >= ENTRY_HHMM), None)
    if i0 is None:
        return None
    o0 = bars[i0][1]
    entry = o0 + krx_tick(o0)
    stop = entry * (1 - HARD_STOP)
    peak, lo = entry, entry
    exit_px, why = None, None
    for k in range(i0, len(bars)):
        t, o, h, l, c = bars[k]
        if t >= EOD_HHMM:
            exit_px, why = o - krx_tick(o), "eod"
            break
        path = (o, l, h, c) if c >= o else (o, h, l, c)
        stop_on = k - i0 >= HARD_STOP_HOLD_MIN
        prev = o
        for j, p in enumerate(path):
            if j == 0 or p < prev:  # moving down (or the open): stop / lock checks
                armed = peak >= entry * (1 + LOCK_ARM)
                lock = peak * (1 - LOCK_GIVEBACK)
                lvl = max(stop if stop_on else -1, lock if armed else -1)
                if lvl > 0 and p <= lvl:
                    fill = p if j == 0 else lvl  # gap through at the open fills at the open
                    exit_px, why = fill - krx_tick(fill), ("lock" if armed and lvl == lock else "stop")
                    break
            peak = max(peak, p)
            lo = min(lo, p)
            prev = p
        if exit_px is not None:
            break
    if exit_px is None:
        c = bars[-1][4]
        exit_px, why = c - krx_tick(c), "eod"
    net = (exit_px * (1 - FEE - TAX)) / (entry * (1 + FEE)) - 1.0
    return {"net": net, "why": why, "mfe": peak / entry - 1.0, "mae": lo / entry - 1.0}


def _trade_stats(trades: list[dict], dates_of: list[str]) -> str:
    if not trades:
        return "n0"
    nets = [t["net"] for t in trades]
    wins = [x for x in nets if x > 0]
    loss = [x for x in nets if x <= 0]
    pf = sum(wins) / abs(sum(loss)) if loss and sum(loss) != 0 else float("inf")
    by_day: dict[str, list[float]] = {}
    for t, d in zip(trades, dates_of):
        by_day.setdefault(d, []).append(t["net"])
    eq, peak, mdd = 0.0, 0.0, 0.0
    for d in sorted(by_day):  # equal weight per pick, one unit of capital per day
        eq += _mean(by_day[d])
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    why = Counter(t["why"] for t in trades)
    return (f"n{len(nets)} net {_mean(nets) * 100:+.3f}% med {_med(nets) * 100:+.2f}% win {len(wins) / len(nets) * 100:.0f}% "
            f"PF {pf:.2f} daily-sum {eq * 100:+.1f}% MDD {mdd * 100:.1f}%p MFE {_mean([t['mfe'] for t in trades]) * 100:+.2f}% "
            f"MAE {_mean([t['mae'] for t in trades]) * 100:+.2f}% exits " + "/".join(f"{k}{v}" for k, v in sorted(why.items())))


def trade_report(rows, dates, split, tr, pk) -> list[str]:
    import random
    rng = random.Random(20261119)
    rand = []
    for d in dates:
        pool = sorted((r for r in tr if r["date"] == d), key=lambda r: r["code"])
        rand.extend(rng.sample(pool, min(RANDOM_PER_DAY, len(pool))))
    groups = {"picked": pk, "picked strict": [r for r in pk if not r["fallback"]], "picked fallback": [r for r in pk if r["fallback"]],
              "tradable tier A/B": [r for r in tr if r["pattern_tier"]], f"random {RANDOM_PER_DAY}/day tradable": rand}
    cache: dict[tuple[str, str], dict | None] = {}
    for grp in groups.values():
        for r in grp:
            key = (r["code"], r["date"])
            if key not in cache:
                nd = next_folder(r["date"])
                bars = minute_bars(r["code"], nd) if nd else None
                cache[key] = simulate(bars) if bars else None
    lines = [f"## trade simulation (next day, entry {ENTRY_HHMM} open +1 tick, hard stop -{HARD_STOP:.1%} after {HARD_STOP_HOLD_MIN} min, "
             f"lock +{LOCK_ARM:.1%}/-{LOCK_GIVEBACK:.1%}, eod {EOD_HHMM}, fee {FEE:.3%}x2 + tax {TAX:.2%}, 1 tick slippage)"]
    for part in ("train", "holdout", "all"):
        lines.append(f"[{part}]")
        for name, grp in groups.items():
            sel = [r for r in grp if (part == "all" or split[r["date"]] == part) and cache.get((r["code"], r["date"]))]
            lines.append(f"  {name:<24} " + _trade_stats([cache[(r['code'], r['date'])] for r in sel], [r["date"] for r in sel]))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="20260629")
    ap.add_argument("--end", default="20261117")
    ap.add_argument("--variant", default="v0", choices=list(g002.FILTER_PROFILES))
    ap.add_argument("--release", default=None, help=f"one of: {', '.join(RELEASABLE)}")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--min-complete", type=float, default=0.9, help="skip a date when daily.csv < this x 10s files")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--suffix", default="", help="extra tag in the output file names (e.g. a test run)")
    ap.add_argument("--cache", action="store_true", help="read per-date pickles in research/10_selection/cache (built on first use)")
    ap.add_argument("--build-cache", action="store_true", help="only build the per-date cache for --start..--end")
    ap.add_argument("--full-rows", action="store_true", help="keep every plain candidate field in the rows file (larger; for rule studies)")
    ap.add_argument("--report", nargs="*", help="summarize existing rows_*.jsonl.gz files (names in research/10_selection)")
    args = ap.parse_args()
    if args.report:
        for name in args.report:
            p = Path(name) if Path(name).exists() else RESEARCH_DIR / name
            text = summarize(p)
            (RESEARCH_DIR / p.name.replace("rows_", "summary_").replace(".jsonl.gz", ".txt")).write_text(text, encoding="utf-8")
            print(text)
        return
    if args.build_cache:
        dates, _ = list_dates(args.start, args.end, args.min_complete)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for msg in pool.map(build_cache_date, dates):
                print(f"  {msg} ({time.time() - t0:.0f}s)", flush=True)
        return
    if args.full_rows:
        os.environ["G043_FULL_ROWS"] = "1"
    os.environ["G002_FILTER_PROFILE"] = args.variant  # inherited by the worker processes (g002 reads it per call)
    out = run_scan(args)
    text = summarize(out)
    (RESEARCH_DIR / out.name.replace("rows_", "summary_").replace(".jsonl.gz", ".txt")).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
