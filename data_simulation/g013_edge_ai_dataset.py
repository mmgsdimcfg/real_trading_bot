# -*- coding: utf-8 -*-
"""Edge AI dataset builder: 1m-trigger events + triple-barrier labels from collected 1m data.

For every trade date D in runtime/data and every code in D's universe:
  1. Rebuild the real 1m bars from g001's <code>_*_10s.txt (raw_bar==1 anchors, minute-START label in the
     file -> converted to minute-END label like r003; volume x interpolation factor, same as g003).
  2. Event = bar where the primary trigger passes, evaluated like r005 _008 (r011.hybrid_trigger_with_context:
     r002 1m trigger on the last <= 200 bars + the 3m uptrend_continuation context), 09:05 <= t <= 14:50, at least
     --min-gap bars after the previous event of the same code.
  3. Features = r011.feature_frames() + compute_features() (the SAME functions r005 _028 calls live), i.e. indicators
     on the fixed FEATURE_HISTORY_BARS window; events whose window is not fully covered are skipped (live fails open).
  4. Label = triple barrier on the following bars: entry = next 1m open + slippage, TP/SL in %, time limit.
     A later bar that OPENS beyond a barrier exits at that open (gap); otherwise same-bar TP/SL ambiguity is
     resolved as SL (conservative). Net of fees/tax/slippage. y = net_ret > 0.

Universe of D (no look-ahead): codes whose 20-day average turnover over dates < D is >= --min-turnover and
price >= --min-price, common shares only; deterministic random sample of --max-codes per day, plus the
scanner picks produced the evening before (data/<prev>/_<prev>_picks.txt = what r003 trades on D).

Output: runtime/edge_ai/dataset/<tag>/events_YYYYMMDD.csv.gz (one file per day, incremental).
<tag> = v<FEATURE_VERSION>_tp_sl_h_<hash of every other generation setting>; params.json must match (no mixing).

Live feedback (--live): r005 _028 writes every live buy signal it scored to runtime/edge_ai/live/edge_ai_D.jsonl
(features as computed live + P(win)). For each such signal (first record per code and 1m bar) this labels the
outcome with the same triple barrier and recomputes the offline features at that bar, so g014 can report
(a) how the model did on REAL buy signals and (b) live-vs-offline feature drift (train/serve skew check).
Output: runtime/edge_ai/live_labeled/<tag>/live_YYYYMMDD.csv.gz.

Run:
  ~/venv/bin/python data_simulation/g013_edge_ai_dataset.py                # all days not built yet
  ~/venv/bin/python data_simulation/g013_edge_ai_dataset.py --dates 20261022 --rebuild

Update log:
- [2026-10-26] type=feat owner=claude
    summary: 신규 - Edge AI(LightGBM 메타 라벨링) 학습 데이터 생성기.
    impact: sim
    compatibility: backward-compatible
- [2026-10-27] type=fix owner=claude
    summary: Codex 1차 리뷰 반영 - 고정 히스토리 창 피처(r011 v2), _008과 같은 3분 continuation 트리거, 시가 갭의
      TP/SL 우선 처리, 데이터셋 태그에 생성 설정 해시 + params.json 불일치 시 증분 생성 거부.
    impact: sim (v1 데이터셋과 별도 폴더)
    compatibility: backward-compatible
- [2026-10-27] type=fix owner=claude
    summary: 장중 실행 시 오늘 폴더(daily.csv 없음)를 최신 폴더로 잡아 빈 거래대금 패널을 캐시 -> KeyError 'amount'.
      daily.csv가 있는 가장 최근 폴더 사용 + 빈 패널은 캐시하지 않고 오류.
    impact: sim
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))

import r001_define_config as cfg  # noqa: E402
from r008_strategy_v2 import krx_tick_size  # noqa: E402
from r011_edge_ai import (  # noqa: E402
    FEATURE_HISTORY_BARS, FEATURE_NAMES, FEATURE_VERSION, MIN_SESSION_BARS, build_strategy_config, completed_3min,
    compute_features, daily_features_from_frame, fast_indicators, feature_frames, find_prev_data_dir,
    history_window_start,
    hybrid_trigger_with_context, session_raw_1min,
)

DATA_ROOT = Path(cfg.RUNTIME_DATA_DIR)
EDGE_AI_DIR = Path(getattr(cfg, "EDGE_AI_DIR", Path(cfg.RUNTIME_ROOT) / "edge_ai"))
FEE_RATE = 0.00015
SELL_TAX_RATE = 0.0020
ENTRY_START = pd.Timestamp("09:05").time()
ENTRY_END = pd.Timestamp("14:50").time()
FORCE_EXIT = pd.Timestamp("15:15").time()
TRIGGER_CACHE_BARS = 200   # r003 1m cache cap (_merge_bar_frame) - the live trigger never sees more history
GEN_PARAM_KEYS = ("tp", "sl", "horizon", "slip_ticks", "min_gap", "min_turnover", "min_price", "max_codes")


def generation_params(args) -> dict:
    """Every setting that changes event selection or labels (worker/date/run options excluded)."""
    out = {k: getattr(args, k) for k in GEN_PARAM_KEYS}
    out.update({"feature_version": FEATURE_VERSION, "feature_history_bars": FEATURE_HISTORY_BARS,
                "trigger_cache_bars": TRIGGER_CACHE_BARS})
    return out


def dataset_tag(args) -> str:
    digest = hashlib.md5(json.dumps(generation_params(args), sort_keys=True).encode()).hexdigest()[:6]
    return f"v{FEATURE_VERSION}_tp{args.tp:g}_sl{args.sl:g}_h{args.horizon}_{digest}"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_minute_bars(path: Path) -> pd.DataFrame | None:
    """g001 10s file -> real 1m bars with minute-END labels (g003 _causal_minute_frames equivalent)."""
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in
                          ("datetime", "open", "high", "low", "close", "volume", "raw_bar"))
    except Exception:
        return None
    if raw.empty:
        return None
    raw.index = pd.to_datetime(raw.pop("datetime"), errors="coerce")
    raw = raw[raw.index.notna()]
    m = raw[raw.index.second == 0]
    if "raw_bar" in m.columns:
        m = m[pd.to_numeric(m["raw_bar"], errors="coerce") == 1]
    m = m[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna(
        subset=["open", "high", "low", "close"])
    m = m[m[["open", "high", "low", "close"]].min(axis=1) > 0]
    if len(m) < 2:
        return None
    gaps = raw.index.to_series().diff().dropna().dt.total_seconds()
    step = float(gaps.median()) if not gaps.empty else 60.0
    m_gaps = m.index.to_series().diff().dropna().dt.total_seconds()
    expansion = max(1.0, round(float(m_gaps.median()) / step)) if step > 0 and not m_gaps.empty else 1.0
    m = m.copy()
    m["volume"] = m["volume"].fillna(0.0) * expansion
    m.index = m.index + pd.Timedelta(minutes=1)
    return m


def turnover_panel(latest_dir: Path, cache_path: Path) -> pd.DataFrame:
    """date x code daily turnover (amount) panel from the newest folder's daily.csv files (cached)."""
    if cache_path.exists() and cache_path.stat().st_mtime > latest_dir.stat().st_mtime:
        return load_panel(cache_path)
    rows = {}
    closes = {}
    for f in latest_dir.glob("*_daily.csv"):
        code = f.name.split("_", 1)[0]
        try:
            d = pd.read_csv(f, encoding="utf-8-sig", usecols=["date", "close", "amount"], dtype={"date": str})
        except Exception:
            continue
        d["date"] = d["date"].str.slice(0, 8)
        rows[code] = pd.to_numeric(d.set_index("date")["amount"], errors="coerce")
        closes[code] = pd.to_numeric(d.set_index("date")["close"], errors="coerce")
    if not rows:
        raise RuntimeError(f"no *_daily.csv in {latest_dir} - refusing to cache an empty turnover panel")
    amount = pd.DataFrame(rows).sort_index()
    close = pd.DataFrame(closes).sort_index()
    panel = pd.concat({"amount": amount, "close": close}, axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache_path)
    return panel


def load_panel(cache_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(cache_path, header=[0, 1], index_col=0)
    panel.index = panel.index.astype(str)
    return panel


def universe_for(trade_date: str, panel: pd.DataFrame, args) -> list[str]:
    amount = panel["amount"]
    close = panel["close"]
    hist = amount[amount.index.astype(str) < trade_date].tail(20)
    if len(hist) < 10:
        return []
    avg = hist.mean()
    last_close = close[close.index.astype(str) < trade_date].iloc[-1]
    ok = (avg >= args.min_turnover) & (last_close >= args.min_price)
    codes = sorted(c for c in avg.index[ok.fillna(False)] if str(c).isdigit() and str(c).endswith("0"))
    if args.max_codes and len(codes) > args.max_codes:
        seed = int(hashlib.md5(trade_date.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        codes = sorted(rng.choice(codes, size=args.max_codes, replace=False).tolist())
    prev_dir = find_prev_data_dir(DATA_ROOT, trade_date)
    if prev_dir is not None:
        picks = prev_dir / f"_{prev_dir.name}_picks.txt"
        if picks.exists():
            for line in picks.read_text(encoding="utf-8-sig").splitlines():
                code = line.split(",", 1)[0].strip()
                if code.isdigit() and code not in codes:
                    codes.append(code)
    return codes


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def triple_barrier(o, h, l, c, times, i: int, args) -> dict | None:
    """Entry at open of bar i+1 (+slip ticks). Returns label dict or None if no next bar."""
    n = len(c)
    if i + 1 >= n:
        return None
    raw_entry = o[i + 1]
    tick = krx_tick_size(raw_entry)
    entry = raw_entry + args.slip_ticks * tick
    tp = entry * (1 + args.tp / 100.0)
    sl = entry * (1 - args.sl / 100.0)
    exit_px, reason, j_exit = None, "time", None
    last = min(n - 1, i + args.horizon)
    mfe, mae = 0.0, 0.0
    for j in range(i + 1, last + 1):
        bar_open = entry if j == i + 1 else o[j]
        mfe = max(mfe, h[j] / entry - 1)
        mae = min(mae, l[j] / entry - 1)
        # the open is the first print of the bar: a gap beyond a barrier exits there, before any intrabar path
        if bar_open >= tp:
            exit_px, reason, j_exit = bar_open, "tp", j
            break
        if bar_open <= sl:
            exit_px, reason, j_exit = bar_open, "sl", j
            break
        if l[j] <= sl:  # open between the barriers: stop first on ambiguous bars
            exit_px, reason, j_exit = sl, "sl", j
            break
        if h[j] >= tp:
            exit_px, reason, j_exit = tp, "tp", j
            break
        if times[j] >= FORCE_EXIT:
            exit_px, reason, j_exit = c[j], "eod", j
            break
    if exit_px is None:
        j_exit = last
        exit_px = c[last]
    exit_px = exit_px - args.slip_ticks * krx_tick_size(exit_px)
    cost = entry * FEE_RATE + exit_px * (FEE_RATE + SELL_TAX_RATE)
    net = (exit_px - entry - cost) / entry
    out = {"entry_px": entry, "exit_px": exit_px, "exit_reason": reason, "hold_bars": j_exit - i,
           "net_ret": net * 100.0, "y": int(net > 0), "mfe_pct": mfe * 100.0, "mae_pct": mae * 100.0}
    for k in (10, 30, 60):
        out[f"fwd_{k}"] = (c[min(n - 1, i + k)] / entry - 1) * 100.0
    return out


# ---------------------------------------------------------------------------
# per day
# ---------------------------------------------------------------------------

def _read_daily(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"date": str})
    except Exception:
        return None


def _index_dir(path: Path | None, suffix: str) -> dict[str, Path]:
    """code -> file for one data folder (one listdir instead of a glob per code)."""
    if path is None or not path.is_dir():
        return {}
    out: dict[str, Path] = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(suffix) and not name.startswith("_"):
            out.setdefault(name.split("_", 1)[0], path / name)
    return out


def offline_feature_frames(raw: pd.DataFrame, m1_full: pd.DataFrame, m3_full: pd.DataFrame | None, i: int):
    """r011.feature_frames(raw.iloc[:i+1]) - reuses the whole-session frames while the history window still
    starts at the first bar (indicators are causal, verified by tests), recomputes once it slides."""
    now = raw.index[i]
    start = history_window_start(now)
    if start < raw.index[0]:
        return None
    if start == raw.index[0]:
        return m1_full.iloc[: i + 1], (m3_full[m3_full.index <= now] if m3_full is not None else None)
    return feature_frames(raw.iloc[: i + 1])


def build_symbol_day(code: str, trade_date: str, minute_file: Path | None, daily_file: Path | None,
                     config, args) -> list[dict]:
    if minute_file is None:
        return []
    bars = load_minute_bars(minute_file)
    raw = session_raw_1min(bars, pd.Timestamp(trade_date))
    if raw is None or len(raw) < MIN_SESSION_BARS + 2:
        return []
    m1 = fast_indicators(raw)  # whole-session indicators (== r002.calculate_indicators): trigger detection
    m3_all = completed_3min(m1)
    daily = _read_daily(daily_file)
    daily_feats = daily_features_from_frame(daily, trade_date)
    prev_close = float("nan")
    if daily is not None and not daily.empty:
        d = daily[daily["date"].astype(str).str.slice(0, 8) < trade_date]
        if not d.empty:
            prev_close = float(pd.to_numeric(d["close"], errors="coerce").iloc[-1])
    o = m1["open"].to_numpy(float)
    h = m1["high"].to_numpy(float)
    l = m1["low"].to_numpy(float)
    c = m1["close"].to_numpy(float)
    v = m1["volume"].to_numpy(float)
    mid = m1["BB_MIDDLE"].to_numpy(float)
    times = m1.index.time
    day_open = o[0]
    cum_pv = np.cumsum(c * v)
    cum_v = np.cumsum(v)
    events = []
    last_event = -10**9
    for i in range(MIN_SESSION_BARS - 1, len(m1) - 1):
        t = times[i]
        if t < ENTRY_START or t > ENTRY_END or i - last_event < args.min_gap:
            continue
        # every trigger path needs close above the 1m BB mid (r001 HYBRID_1MIN_CONTINUATION_REQUIRE_ABOVE_MID)
        if not c[i] > mid[i]:
            continue
        now = m1.index[i]
        win = m1.iloc[max(0, i + 1 - TRIGGER_CACHE_BARS): i + 1]
        info: dict = {}
        ok, reason = hybrid_trigger_with_context(
            win, m3_all[m3_all.index <= now] if m3_all is not None else None, float(c[i]), info)
        if not ok:
            continue
        label = triple_barrier(o, h, l, c, times, i, args)
        if label is None:
            continue
        frames = offline_feature_frames(raw, m1, m3_all, i)
        if frames is None:
            continue
        vwap = cum_pv[i] / cum_v[i] if cum_v[i] > 0 else float("nan")
        feats = compute_features(frames[0], frames[1], now, float(c[i]), prev_close, day_open, vwap, daily_feats,
                                 config, trigger_info=info)
        rec = {"date": trade_date, "code": code, "ts": now.strftime("%Y-%m-%d %H:%M:%S"), "price": c[i],
               "trigger": reason}
        rec.update(feats)
        rec.update(label)
        events.append(rec)
        last_event = i
    return events


def build_live_day(trade_date: str, args) -> int:
    src = EDGE_AI_DIR / "live" / f"edge_ai_{trade_date}.jsonl"
    if not src.exists():
        return 0
    recs = []
    for line in src.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            continue
    seen: set[tuple[str, str]] = set()
    first = []
    for r in recs:
        key = (str(r.get("code", "")).zfill(6), str(r.get("bar")))
        if r.get("prob") is None or key in seen:
            continue
        seen.add(key)
        first.append(r)
    if not first:
        return 0
    config = build_strategy_config(cfg)
    minute_files = _index_dir(DATA_ROOT / trade_date, "_10s.txt")
    daily_files = _index_dir(find_prev_data_dir(DATA_ROOT, trade_date), "_daily.csv")
    rows = []
    cache: dict[str, tuple] = {}
    for r in first:
        code = str(r["code"]).zfill(6)
        if code not in cache:
            bars = load_minute_bars(minute_files[code]) if code in minute_files else None
            raw = session_raw_1min(bars, pd.Timestamp(trade_date))
            m1 = fast_indicators(raw) if raw is not None else None
            daily = _read_daily(daily_files.get(code))
            prev_close = float("nan")
            if daily is not None and not daily.empty:
                d = daily[daily["date"].astype(str).str.slice(0, 8) < trade_date]
                if not d.empty:
                    prev_close = float(pd.to_numeric(d["close"], errors="coerce").iloc[-1])
            cache[code] = (raw, m1, completed_3min(m1) if m1 is not None else None,
                           daily_features_from_frame(daily, trade_date), prev_close)
        raw, m1, m3_all, daily_feats, prev_close = cache[code]
        out = {"date": trade_date, "code": code, "ts": r.get("ts"), "bar": r.get("bar"), "price": r.get("price"),
               "prob": r.get("prob"), "threshold": r.get("threshold"), "allow": r.get("allow"),
               "enforced": r.get("enforced"), "model_id": r.get("model_id")}
        out.update({f"live_{k}": v for k, v in (r.get("features") or {}).items()})
        if m1 is not None:
            pos = m1.index.get_indexer([pd.Timestamp(r.get("bar"))])[0]
            if pos >= 0:
                o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
                v = m1["volume"].to_numpy(float)
                label = triple_barrier(o, h, l, c, m1.index.time, pos, args)
                if label is not None:
                    out.update(label)
                now = m1.index[pos]
                price = float(r.get("price") or c[pos])
                vwap = float(np.sum(c[: pos + 1] * v[: pos + 1]) / max(np.sum(v[: pos + 1]), 1e-9))
                info: dict = {}
                hybrid_trigger_with_context(m1.iloc[max(0, pos + 1 - TRIGGER_CACHE_BARS): pos + 1],
                                            m3_all[m3_all.index <= now] if m3_all is not None else None, price, info)
                frames = offline_feature_frames(raw, m1, m3_all, pos)
                if frames is not None:
                    off = compute_features(frames[0], frames[1], now, price, prev_close, o[0], vwap, daily_feats,
                                           config, trigger_info=info)
                    out.update({f"off_{k}": val for k, val in off.items()})
        rows.append(out)
    out_dir = EDGE_AI_DIR / "live_labeled" / dataset_tag(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"live_{trade_date}.csv.gz", index=False, compression="gzip")
    return len(rows)


def build_day(task) -> tuple[str, int, float]:
    trade_date, args_dict = task
    args = argparse.Namespace(**args_dict)
    started = time.time()
    out_dir = EDGE_AI_DIR / "dataset" / dataset_tag(args)
    out_path = out_dir / f"events_{trade_date}.csv.gz"
    config = build_strategy_config(cfg)
    panel = load_panel(EDGE_AI_DIR / "cache" / "turnover_panel.csv")
    codes = universe_for(trade_date, panel, args)
    day_dir = DATA_ROOT / trade_date
    prev_dir = find_prev_data_dir(DATA_ROOT, trade_date)
    minute_files = _index_dir(day_dir, "_10s.txt")
    daily_files = _index_dir(prev_dir, "_daily.csv")
    rows: list[dict] = []
    for code in codes:
        try:
            rows.extend(build_symbol_day(code, trade_date, minute_files.get(code), daily_files.get(code), config, args))
        except Exception as exc:  # one bad file must not kill the day
            print(f"[WARN] {trade_date} {code}: {exc}", flush=True)
    cols = ["date", "code", "ts", "price", "trigger", *FEATURE_NAMES, "entry_px", "exit_px", "exit_reason",
            "hold_bars", "net_ret", "y", "mfe_pct", "mae_pct", "fwd_10", "fwd_30", "fwd_60"]
    df = pd.DataFrame(rows, columns=cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.gz")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out_path)
    return trade_date, len(df), time.time() - started


def _day_settled(day_dir: Path, settle_minutes: int) -> bool:
    """True when the folder has 10s files and none was modified in the last settle_minutes."""
    try:
        newest = max((e.stat().st_mtime for e in os.scandir(day_dir) if e.name.endswith("_10s.txt")), default=None)
    except OSError:
        return False
    return newest is not None and time.time() - newest >= settle_minutes * 60


def main() -> None:
    p = argparse.ArgumentParser(description="Build Edge AI meta-label dataset from runtime/data")
    p.add_argument("--dates", nargs="*", help="YYYYMMDD list (default: every data folder)")
    p.add_argument("--rebuild", action="store_true", help="rebuild days that already have a file")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--tp", type=float, default=2.0, help="take-profit barrier %%")
    p.add_argument("--sl", type=float, default=1.7, help="stop-loss barrier %% (r001 HARD_STOP_LOSS_PCT)")
    p.add_argument("--horizon", type=int, default=60, help="time barrier in 1m bars")
    p.add_argument("--slip-ticks", type=int, default=1)
    p.add_argument("--min-gap", type=int, default=5, help="min bars between events of one code")
    p.add_argument("--min-turnover", type=float, default=3e9, help="20d avg turnover KRW")
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--max-codes", type=int, default=150, help="per-day random sample size (0 = all)")
    p.add_argument("--live", action="store_true", help="also label live shadow records (edge_ai/live/*.jsonl)")
    p.add_argument("--settle-minutes", type=int, default=30,
                   help="skip a day whose newest 10s file is younger than this (g001 still collecting)")
    args = p.parse_args()

    days = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8)
    if args.dates:
        days = [d for d in days if d in set(args.dates)]
    out_dir = EDGE_AI_DIR / "dataset" / dataset_tag(args)
    if not args.rebuild:
        days = [d for d in days if not (out_dir / f"events_{d}.csv.gz").exists()]
    days = [d for d in days if _day_settled(DATA_ROOT / d, args.settle_minutes)]
    if args.live:
        for d in sorted(p.name.split("_")[-1].split(".")[0] for p in (EDGE_AI_DIR / "live").glob("edge_ai_*.jsonl")):
            if args.dates and d not in set(args.dates):
                continue
            if _day_settled(DATA_ROOT / d, args.settle_minutes):
                print(f"  live {d}: {build_live_day(d, args)} labeled signals", flush=True)
    if not days:
        print("nothing to build")
        return
    # newest folder that already has daily.csv (today's folder may only hold today_buys.txt mid-session)
    latest = next(d for d in sorted((d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit()), reverse=True)
                  if any(d.glob("*_daily.csv")))
    turnover_panel(latest, EDGE_AI_DIR / "cache" / "turnover_panel.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.json"
    params = generation_params(args)
    if params_path.exists():
        old = json.loads(params_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in params} != params:
            raise SystemExit(f"{params_path} differs from current generation settings - refusing to mix datasets")
    else:
        params_path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    tasks = [(d, {k: v for k, v in vars(args).items() if k != "dates"}) for d in days]
    print(f"building {len(days)} day(s) -> {out_dir} (workers={args.workers})", flush=True)
    with Pool(max(1, args.workers)) as pool:
        for trade_date, n, sec in pool.imap_unordered(build_day, tasks):
            print(f"  {trade_date}: {n} events ({sec:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
