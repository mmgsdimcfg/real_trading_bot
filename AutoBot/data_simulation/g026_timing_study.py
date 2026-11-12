# -*- coding: utf-8 -*-
"""Buy / sell / stop timing study on the 1-min history (2026-11-11 사용자 요청: 매수·매도·손절 타이밍 산출).

Questions
  1. stop width: how deep do positions that end up winning dip first (MAE) - is -1%, -1.5%, -2% ... the better stop?
  2. profit lock / trailing: arm + giveback, and a structure that lets a rising / surging position run
     (wider giveback while the 1-min trend holds, or once the position is far in profit)
  3. entry: which indicator states at the trigger are followed by a rise (bucket scan, train vs holdout)

Stages (all outputs under <runtime>/edge_ai/timing_study)
  cache  : 1-min session bars of the liquid universe, one pickle per day                 (--stage cache)
  events : entry events (1-min MA5 x MA20 golden cross = live trigger proxy, + causal random control), features at
           the trigger bar and the forward 1-min path up to 15:15                          (--stage events)
  real   : the bot's real buys (runtime/log/*_buy_sell.log) as events with the same path  (--stage real)
  panel  : every 3rd bar of every code-day: features + forward returns + net of 3 exit policies (--stage panel)
  The exit grid itself lives in g027_timing_grid.py (reads events_*.pkl).

Fill / cost model = g017 (fee 0.015% per side, sell tax 0.20%, market fills 1 tick slippage, 1-min path o->l->h->c on
bullish bars and o->h->l->c on bearish bars). Entry = next bar open + 1 tick. Nothing here is imported by r003.

Update log
- [2026-11-11] type=feat - first version
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))

import g013_edge_ai_dataset as g013  # noqa: E402
from r011_edge_ai import daily_features_from_frame, fast_indicators  # noqa: E402

OUT_DIR = g013.EDGE_AI_DIR / "timing_study"
BARS_DIR = OUT_DIR / "bars"
LOG_ROOT = g013.DATA_ROOT.parent / "log"
TOP_N = 250                    # liquid universe: top N by 20-day average turnover (common stocks)
MIN_PRICE = 2000.0
ENTRY_START_MIN = 9 * 60 + 5
ENTRY_END_MIN = 14 * 60 + 50
EOD_MIN = 15 * 60 + 15
WARM_BARS = 120                # prior-day bars prepended for the indicators (live r012 warmup)
MIN_GAP_BARS = 5               # min bars between two events of one code
PATH_COLS = ("minute", "open", "high", "low", "close", "ma5", "ma20", "vwap")


def day_dirs() -> list[Path]:
    return sorted(p for p in g013.DATA_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name))


def _index(day_dir: Path, suffix: str) -> dict[str, Path]:
    return {f.name.split("_", 1)[0]: f for f in day_dir.glob(f"*{suffix}")}


def real_buys() -> pd.DataFrame:
    """Completed real round trips from the daily buy_sell logs (one row per Sell block)."""
    rows = []
    pat_sell = re.compile(r">>> (\w{6})_(.*?) Sell - 수익금액=([-\d,]+)원 \| 수익율=([-\d.]+)% \| reason=(\S+)")
    pat_buy = re.compile(r"매수시각=([\d\- :]+) \| 매수가격=([\d,.]+) \| 매수수량=(\d+)")
    pat_out = re.compile(r"매도시각=([\d\- :]+) \| 매도가격=([\d,.]+) \| 매도수량=(\d+)")
    for f in sorted(LOG_ROOT.glob("*/2*_buy_sell.log")):
        if not re.fullmatch(r"\d{8}_buy_sell\.log", f.name):
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            m = pat_sell.search(line)
            if not m:
                continue
            b = pat_buy.search(lines[i + 1]) if i + 1 < len(lines) else None
            s = pat_out.search(lines[i + 2]) if i + 2 < len(lines) else None
            if not b or not s:
                continue
            rows.append({
                "code": m.group(1), "name": m.group(2), "pnl_krw": float(m.group(3).replace(",", "")),
                "pnl_pct": float(m.group(4)), "reason": m.group(5),
                "buy_time": pd.Timestamp(b.group(1).strip()), "buy_price": float(b.group(2).replace(",", "")),
                "buy_qty": int(b.group(3)), "sell_time": pd.Timestamp(s.group(1).strip()),
                "sell_price": float(s.group(2).replace(",", "")), "sell_qty": int(s.group(3)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["code", "buy_time", "sell_time", "sell_qty", "reason"]).reset_index(drop=True)
    df["date"] = df["buy_time"].dt.strftime("%Y%m%d")
    return df


# ---------------------------------------------------------------------------
# stage cache
# ---------------------------------------------------------------------------
def _universe(trade_date: str, panel: pd.DataFrame, extra: set[str]) -> list[str]:
    amount, close = panel["amount"], panel["close"]
    hist = amount[amount.index.astype(str) < trade_date].tail(20)
    if len(hist) < 10:
        return sorted(extra)
    avg = hist.mean()
    last_close = close[close.index.astype(str) < trade_date].iloc[-1]
    ok = avg[(last_close >= MIN_PRICE).reindex(avg.index).fillna(False)]
    ok = ok[[str(c).isdigit() and str(c).endswith("0") for c in ok.index]]
    return sorted(set(ok.sort_values(ascending=False).head(TOP_N).index) | extra)


def cache_day(task) -> tuple[str, int, float]:
    trade_date, codes = task
    t0 = time.time()
    out = BARS_DIR / f"{trade_date}.pkl"
    if out.exists():
        return trade_date, -1, 0.0
    day_dir = g013.DATA_ROOT / trade_date
    files = _index(day_dir, "_10s.txt")
    dfiles = _index(day_dir, "_daily.csv")
    day = pd.Timestamp(trade_date)
    bars, daily = {}, {}
    for code in codes:
        f = files.get(code)
        if f is None:
            continue
        m = g013.load_minute_bars(f)
        if m is None:
            continue
        m = m[m.index.normalize() == day]
        m = m[~m.index.duplicated(keep="last")].sort_index()
        minute = (m.index.hour * 60 + m.index.minute).to_numpy()
        keep = (minute > 9 * 60) & (minute <= 15 * 60 + 30)
        if keep.sum() < 60:
            continue
        arr = np.column_stack([minute[keep], m[["open", "high", "low", "close", "volume"]].to_numpy()[keep]])
        bars[code] = arr.astype(np.float64)
        df = dfiles.get(code)
        if df is not None:
            try:
                d = pd.read_csv(df, encoding="utf-8-sig", usecols=["date", "open", "high", "low", "close", "volume", "amount"],
                                dtype={"date": str})
                daily[code] = daily_features_from_frame(d, trade_date)
                prev = d[d["date"].str.slice(0, 8) < trade_date]
                daily[code]["prev_close"] = float(prev["close"].iloc[-1]) if len(prev) else float("nan")
            except Exception:
                pass
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"date": trade_date, "bars": bars, "daily": daily}, fh, protocol=4)
    tmp.replace(out)
    return trade_date, len(bars), time.time() - t0


def stage_cache(args) -> None:
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    dirs = day_dirs()
    panel = g013.turnover_panel(dirs[-1], g013.EDGE_AI_DIR / "cache" / "turnover_panel.csv")
    real = real_buys()
    traded = real.groupby("date")["code"].apply(set).to_dict() if not real.empty else {}
    tasks = []
    for d in dirs:
        if args.dates and d.name not in args.dates:
            continue
        extra = set(traded.get(d.name, set()))
        prev = g013.find_prev_data_dir(g013.DATA_ROOT, d.name)
        if prev is not None:
            picks = prev / f"_{prev.name}_picks.txt"
            if picks.exists():
                extra |= {ln.split(",", 1)[0].strip() for ln in picks.read_text(encoding="utf-8-sig").splitlines() if ln.strip()}
        tasks.append((d.name, _universe(d.name, panel, extra)))
    with Pool(args.workers) as pool:
        for name, n, sec in pool.imap_unordered(cache_day, tasks):
            print(f"cache {name} codes={n} {sec:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# stage events
# ---------------------------------------------------------------------------
def load_day(trade_date: str) -> dict | None:
    f = BARS_DIR / f"{trade_date}.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def _indicator_frame(arr: np.ndarray, prev: np.ndarray | None) -> tuple[pd.DataFrame, int]:
    """Indicators over [prior-day tail + session]; returns (frame, offset of the first session bar)."""
    cols = ["minute", "open", "high", "low", "close", "volume"]
    off = 0
    if prev is not None and len(prev):
        tail = prev[-WARM_BARS:]
        off = len(tail)
        arr = np.vstack([tail, arr])
    df = pd.DataFrame(arr, columns=cols)
    ind = fast_indicators(df[["open", "high", "low", "close", "volume"]])
    ind["minute"] = arr[:, 0]
    # session VWAP (fast_indicators' VWAP is cumulative over the frame = would include the prior day)
    c, v = arr[off:, 4], arr[off:, 5]
    vwap = np.full(len(arr), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap[off:] = np.cumsum(c * v) / np.cumsum(v)
    ind["VWAP"] = vwap
    return ind, off


def _pct(a, b):
    return (a / b - 1.0) * 100.0 if b and np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def _features(ind: pd.DataFrame, off: int, i: int, daily: dict, mkt: dict) -> dict:
    """State at the close of trigger bar i (index in ind). Everything is known at that close."""
    r = ind.iloc[i]
    c = float(r["close"])
    o_day = float(ind["open"].iloc[off])
    sess = ind.iloc[off:i + 1]
    hi_day, lo_day = float(sess["high"].max()), float(sess["low"].min())
    cl = ind["close"].to_numpy()
    mid = ind["BB_MIDDLE"].to_numpy()
    vol = ind["volume"].to_numpy()
    minute = int(r["minute"])
    f = {
        "minute": minute,
        "rsi": r["RSI"], "stoch_k": r["STOCH_K"], "stoch_d": r["STOCH_D"], "williams": r["WILLIAMS_R"], "mfi": r["MFI"],
        "macd_hist_pct": r["MACD_HIST"] / c * 100.0, "adx": r["ADX"], "di_spread": r["DI_PLUS"] - r["DI_MINUS"],
        "vol_ratio": r["volume"] / r["VOL_MA20"] if r["VOL_MA20"] > 0 else np.nan,
        "vol5_ratio": vol[i - 4:i + 1].mean() / r["VOL_MA20"] if i >= 4 and r["VOL_MA20"] > 0 else np.nan,
        "bb_width": (r["BB_UPPER"] - r["BB_LOWER"]) / r["BB_MIDDLE"] * 100.0,
        "bb_pctb": (c - r["BB_LOWER"]) / (r["BB_UPPER"] - r["BB_LOWER"]) if r["BB_UPPER"] > r["BB_LOWER"] else np.nan,
        "mid_gap": _pct(c, r["BB_MIDDLE"]),
        "mid_slope5": _pct(mid[i], mid[i - 5]) if i >= 5 else np.nan,
        "mid_slope20": _pct(mid[i], mid[i - 20]) if i >= 20 else np.nan,
        "ema20_60": _pct(r["EMA_20"], r["EMA_60"]),
        "vwap_gap": _pct(c, r["VWAP"]),
        "atr_pct": r["ATR"] / c * 100.0,
        "ret5": _pct(c, cl[i - 5]) if i >= 5 else np.nan,
        "ret15": _pct(c, cl[i - 15]) if i >= 15 else np.nan,
        "ret30": _pct(c, cl[i - 30]) if i >= 30 else np.nan,
        "day_ret": _pct(c, o_day),
        "gap_ret": _pct(o_day, daily.get("prev_close", np.nan)),
        "prev_ret": _pct(c, daily.get("prev_close", np.nan)),
        "hod_dist": _pct(c, hi_day),
        "range_pos": (c - lo_day) / (hi_day - lo_day) if hi_day > lo_day else np.nan,
        "candle_pct": _pct(c, float(r["open"])),
        "obv_up": float(r["OBV"] > r["OBV_MA"]),
        "mkt_ret": mkt["ret"].get(minute, np.nan), "mkt_ret15": mkt["ret15"].get(minute, np.nan),
        "mkt_breadth": mkt["breadth"].get(minute, np.nan),
    }
    for k, v in daily.items():
        if k != "prev_close":
            f[k] = v
    return f


def _market(bars: dict) -> dict:
    """Equal-weight universe return from the open per minute (market direction proxy, known at that minute's close)."""
    grid = np.arange(9 * 60 + 1, 15 * 60 + 31)
    rets = []
    for arr in bars.values():
        s = pd.Series(arr[:, 4], index=arr[:, 0].astype(int)).reindex(grid).ffill()
        rets.append((s / arr[0, 1] - 1.0) * 100.0)
    if not rets:
        return {"ret": {}, "ret15": {}, "breadth": {}}
    m = pd.concat(rets, axis=1)
    mean = m.mean(axis=1)
    return {"ret": mean.to_dict(), "ret15": (mean - mean.shift(15)).to_dict(), "breadth": (m > 0).mean(axis=1).to_dict()}


def _path(ind: pd.DataFrame, start: int) -> np.ndarray:
    """Forward bars from index start (the entry bar) up to the EOD bar: PATH_COLS."""
    sub = ind.iloc[start:]
    sub = sub[sub["minute"] <= EOD_MIN + 15]
    return np.column_stack([sub["minute"], sub["open"], sub["high"], sub["low"], sub["close"],
                            sub["MA_5"], sub["BB_MIDDLE"], sub["VWAP"]]).astype(np.float64)


def _setup_events(ind: pd.DataFrame, off: int, ok: np.ndarray, daily: dict, mkt: dict) -> list[tuple[int, str]]:
    """Three pre-declared setups (Codex review 2026-11-11; fixed definitions, no tuning). All need the daily uptrend
    D-1 close > MA20 and 20-day return > 0. Trigger = close of bar i, entry = next bar open.
      orb_rs    09:15~10:30 first close above the 09:00~09:15 range high, stock day return > universe return,
                bar volume >= 2 x VOL_MA20
      pb_vwap   universe return > 0, day high >= +2% from the open, price fell below VWAP without breaking the opening
                range low, first close back above VWAP
      dip_rev   stock was ahead of the universe 15 bars ago, fell >= 1% in 15 bars, universe not falling over the last
                5 minutes, first close above the highs of the preceding three bars
    """
    if not (daily.get("d_close_ma20_pct", np.nan) > 0 and daily.get("d_ret_20", np.nan) > 0):
        return []
    n = len(ind)
    minute = ind["minute"].to_numpy().astype(np.int64)
    o, h, l, c, v = (ind[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    vwap, vma = ind["VWAP"].to_numpy(), ind["VOL_MA20"].to_numpy()
    sess = np.arange(n) >= off
    orng = sess & (minute <= 9 * 60 + 15)
    if not orng.any():
        return []
    or_hi, or_lo, o_day = h[orng].max(), l[orng].min(), o[off]
    mret = np.array([mkt["ret"].get(int(m), np.nan) for m in minute])
    day_ret = (c / o_day - 1.0) * 100.0
    hi_run = np.full(n, np.nan)
    hi_run[off:] = np.maximum.accumulate(h[off:])
    out: list[tuple[int, str]] = []
    # orb_rs
    with np.errstate(invalid="ignore"):
        m = ok & (minute > 9 * 60 + 15) & (minute <= 10 * 60 + 30) & (c > or_hi) & (day_ret > mret) & (v >= 2.0 * vma)
    idx = np.flatnonzero(m)
    if len(idx):
        out.append((int(idx[0]), "orb_rs"))
    # pb_vwap
    below = False
    broke = False
    for i in range(off + 1, n):
        if l[i] < or_lo:
            broke = True
        if c[i] <= vwap[i]:
            below = True
            continue
        if below and ok[i] and not broke and mret[i] > 0 and hi_run[i] >= o_day * 1.02 and minute[i] > 9 * 60 + 15:
            out.append((i, "pb_vwap"))
            break
        below = False
    # dip_rev
    last = -10**9
    for i in range(off + 20, n):
        if not ok[i] or i - last < 15:
            continue
        with np.errstate(invalid="ignore"):
            ahead = day_ret[i - 15] - mret[i - 15] > 0
            fell = c[i - 1] / c[i - 16] - 1.0 <= -0.01
            calm = mret[i] - mret[i - 5] >= 0
        if ahead and fell and calm and c[i] > h[i - 3:i].max():
            out.append((i, "dip_rev"))
            last = i
    return out


def events_day(task) -> tuple[str, int, float]:
    trade_date, prev_date = task
    t0 = time.time()
    out = OUT_DIR / "events" / f"events_{trade_date}.pkl"
    if out.exists():
        return trade_date, -1, 0.0
    day = load_day(trade_date)
    if day is None:
        return trade_date, 0, 0.0
    prev = load_day(prev_date) if prev_date else None
    mkt = _market(day["bars"])
    rng = np.random.default_rng(int(trade_date))
    rows, paths = [], []
    for code, arr in day["bars"].items():
        ind, off = _indicator_frame(arr, prev["bars"].get(code) if prev else None)
        minute = ind["minute"].to_numpy()
        ma5, ma20 = ind["MA_5"].to_numpy(), ind["BB_MIDDLE"].to_numpy()
        close = ind["close"].to_numpy()
        n = len(ind)
        ok = (np.arange(n) >= off + 1) & (minute >= ENTRY_START_MIN) & (minute <= ENTRY_END_MIN) & (np.arange(n) < n - 2)
        with np.errstate(invalid="ignore"):
            above = ma5 > ma20
            cross = above & ~np.r_[False, above[:-1]] & np.isfinite(np.r_[np.nan, ma20[:-1]]) & (close > ma20)
        daily = day["daily"].get(code, {})
        picked: list[tuple[int, str]] = []
        last = -10**9
        for i in np.flatnonzero(cross & ok):
            if i - last >= MIN_GAP_BARS:
                picked.append((int(i), "ma_cross"))
                last = i
        picked += _setup_events(ind, off, ok, daily, mkt)
        cand = np.flatnonzero(ok)
        if len(cand):
            picked.append((int(rng.choice(cand)), "random"))       # drift control: any bar
            up = np.flatnonzero(ok & above & (close > ma20))
            if len(up):
                picked.append((int(rng.choice(up)), "random_up"))  # control: any bar in the 1m uptrend state (live "above")
        for i, family in picked:
            p = _path(ind, i + 1)
            if len(p) < 3:
                continue
            f = _features(ind, off, i, daily, mkt)
            f.update({"date": trade_date, "code": code, "family": family, "bar": int(i - off),
                      "trigger_close": float(close[i]), "entry_minute": int(p[0, 0]), "path": len(paths)})
            rows.append(f)
            paths.append(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"events": pd.DataFrame(rows), "paths": paths}, fh, protocol=4)
    tmp.replace(out)
    return trade_date, len(rows), time.time() - t0


def stage_events(args) -> None:
    dates = sorted(p.stem for p in BARS_DIR.glob("*.pkl"))
    tasks = [(d, dates[k - 1] if k else None) for k, d in enumerate(dates) if not args.dates or d in args.dates]
    with Pool(args.workers) as pool:
        for name, n, sec in pool.imap_unordered(events_day, tasks):
            print(f"events {name} n={n} {sec:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# stage real
# ---------------------------------------------------------------------------
def stage_real(args) -> None:
    real = real_buys()
    if real.empty:
        print("no real trades")
        return
    # one row per buy (a position sold in parts has several Sell blocks)
    g = real.groupby(["code", "buy_time"], as_index=False).agg(
        date=("date", "first"), name=("name", "first"), buy_price=("buy_price", "first"), buy_qty=("buy_qty", "max"),
        pnl_krw=("pnl_krw", "sum"), sell_time=("sell_time", "max"), reasons=("reason", lambda s: "|".join(s)),
        sold_value=("sell_price", lambda s: 0.0))
    sold = real.assign(v=real["sell_price"] * real["sell_qty"]).groupby(["code", "buy_time"]).agg(v=("v", "sum"), q=("sell_qty", "sum"))
    g = g.merge(sold, on=["code", "buy_time"])
    g["sell_avg"] = g["v"] / g["q"]
    g["real_ret_pct"] = (g["sell_avg"] / g["buy_price"] - 1.0) * 100.0
    dates = sorted(p.stem for p in BARS_DIR.glob("*.pkl"))
    rows, paths = [], []
    for date, grp in g.groupby("date"):
        day = load_day(date)
        if day is None:
            continue
        k = dates.index(date)
        prev = load_day(dates[k - 1]) if k else None
        mkt = _market(day["bars"])
        for _, t in grp.iterrows():
            arr = day["bars"].get(t["code"])
            if arr is None:
                continue
            ind, off = _indicator_frame(arr, prev["bars"].get(t["code"]) if prev else None)
            minute = ind["minute"].to_numpy()
            bm = t["buy_time"].hour * 60 + t["buy_time"].minute      # buy inside the bar labelled bm+1
            idx = np.flatnonzero((np.arange(len(ind)) >= off) & (minute == bm))  # last bar completed before the buy
            if not len(idx) or bm < 9 * 60 + 1 or bm > 15 * 60 + 19:
                continue
            i = int(idx[0])
            # the bar that contains the buy (i + 1) also holds prices from BEFORE the fill, so the replay starts at the next
            # bar (i + 2); up to 60 s right after the fill are not replayed (Codex review 2026-11-11)
            p = _path(ind, i + 2)
            if len(p) < 3:
                continue
            f = _features(ind, off, i, day["daily"].get(t["code"], {}), mkt)
            f.update({"date": date, "code": t["code"], "family": "real", "bar": int(i - off), "trigger_close": float(ind["close"].iloc[i]),
                      "entry_minute": int(p[0, 0]), "path": len(paths), "buy_price": t["buy_price"], "buy_time": t["buy_time"],
                      "sell_time": t["sell_time"], "real_ret_pct": t["real_ret_pct"], "real_pnl_krw": t["pnl_krw"],
                      "reasons": t["reasons"], "buy_qty": t["buy_qty"]})
            rows.append(f)
            paths.append(p)
    out = OUT_DIR / "events" / "real.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"events": pd.DataFrame(rows), "paths": paths}, fh, protocol=4)
    print(f"real buys={len(g)} with path={len(rows)} -> {out}")


# ---------------------------------------------------------------------------
# stage panel: every 3rd bar of every code-day with vectorized features + labels (for the model report in g027)
# ---------------------------------------------------------------------------
PANEL_STEP = 3
PANEL_POLICIES = {            # g027 PRM_COLS: stop, atr_k, hold, arm, give, mode, give2, p1, tp, cap
    "live": (1.7, 0, 4, 1.0, 0.4, 0, 0, 0, 0, 0),
    "tight": (2.0, 0, 4, 0.6, 0.3, 0, 0, 0, 0, 0),
    "run": (2.5, 0, 4, 2.0, 1.2, 0, 0, 0, 0, 0),
}


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def panel_day(task) -> tuple[str, int, float]:
    import g027_timing_grid as g027
    trade_date, prev_date = task
    t0 = time.time()
    out = OUT_DIR / "panel" / f"panel_{trade_date}.pkl"
    if out.exists():
        return trade_date, -1, 0.0
    day = load_day(trade_date)
    if day is None:
        return trade_date, 0, 0.0
    prev = load_day(prev_date) if prev_date else None
    mkt = _market(day["bars"])
    frames = []
    pol = {k: np.array(v, dtype=np.float64) for k, v in PANEL_POLICIES.items()}
    for code, arr in day["bars"].items():
        ind, off = _indicator_frame(arr, prev["bars"].get(code) if prev else None)
        n = len(ind)
        g = {k: ind[k].to_numpy(dtype=np.float64) for k in ind.columns}
        c, o, h, l, v, minute = g["close"], g["open"], g["high"], g["low"], g["volume"], g["minute"]
        mid = g["BB_MIDDLE"]
        pos = np.arange(n)
        sel = np.flatnonzero((pos >= off + 1) & (minute >= ENTRY_START_MIN) & (minute <= ENTRY_END_MIN) & (pos < n - 3)
                             & (minute.astype(np.int64) % PANEL_STEP == 0))
        if not len(sel):
            continue
        o_day = o[off]
        sess_hi = np.full(n, np.nan)
        sess_lo = np.full(n, np.nan)
        sess_hi[off:] = np.maximum.accumulate(h[off:])
        sess_lo[off:] = np.minimum.accumulate(l[off:])
        daily = day["daily"].get(code, {})
        pc = daily.get("prev_close", np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            above = (g["MA_5"] > mid) & (c > mid)
            age = np.full(n, -1.0)      # bars since MA5 went above the mid (0 = cross bar), -1 while below
            run = -1
            for i in range(n):
                run = run + 1 if above[i] else -1
                age[i] = run
            lo30 = pd.Series(l).rolling(30, min_periods=5).min().to_numpy()
            hi30 = pd.Series(h).rolling(30, min_periods=5).max().to_numpy()
            f = {
                "minute": minute, "rsi": g["RSI"], "stoch_k": g["STOCH_K"], "stoch_kd": g["STOCH_K"] - g["STOCH_D"],
                "williams": g["WILLIAMS_R"], "mfi": g["MFI"], "macd_hist_pct": g["MACD_HIST"] / c * 100.0,
                "macd_hist_d": (g["MACD_HIST"] - _shift(g["MACD_HIST"], 1)) / c * 100.0, "adx": g["ADX"],
                "di_spread": g["DI_PLUS"] - g["DI_MINUS"], "vol_ratio": v / g["VOL_MA20"],
                "vol5_ratio": pd.Series(v).rolling(5).mean().to_numpy() / g["VOL_MA20"],
                "bb_width": (g["BB_UPPER"] - g["BB_LOWER"]) / mid * 100.0,
                "bb_pctb": (c - g["BB_LOWER"]) / (g["BB_UPPER"] - g["BB_LOWER"]),
                "mid_gap": (c / mid - 1.0) * 100.0, "ma5_gap": (g["MA_5"] / mid - 1.0) * 100.0,
                "mid_slope5": (mid / _shift(mid, 5) - 1.0) * 100.0, "mid_slope20": (mid / _shift(mid, 20) - 1.0) * 100.0,
                "ema20_60": (g["EMA_20"] / g["EMA_60"] - 1.0) * 100.0, "vwap_gap": (c / g["VWAP"] - 1.0) * 100.0,
                "atr_pct": g["ATR"] / c * 100.0, "ret1": (c / _shift(c, 1) - 1.0) * 100.0, "ret3": (c / _shift(c, 3) - 1.0) * 100.0,
                "ret5": (c / _shift(c, 5) - 1.0) * 100.0, "ret15": (c / _shift(c, 15) - 1.0) * 100.0,
                "ret30": (c / _shift(c, 30) - 1.0) * 100.0, "ret60": (c / _shift(c, 60) - 1.0) * 100.0,
                "day_ret": (c / o_day - 1.0) * 100.0, "prev_ret": (c / pc - 1.0) * 100.0,
                "gap_ret": np.full(n, (o_day / pc - 1.0) * 100.0),
                "hod_dist": (c / sess_hi - 1.0) * 100.0, "lod_dist": (c / sess_lo - 1.0) * 100.0,
                "dd30": (c / hi30 - 1.0) * 100.0, "bounce30": (c / lo30 - 1.0) * 100.0,
                "candle_pct": (c / o - 1.0) * 100.0, "obv_up": (g["OBV"] > g["OBV_MA"]).astype(float), "above_age": age,
                "price": c,
            }
        df = pd.DataFrame({k: np.asarray(x)[sel] for k, x in f.items()})
        m_sel = minute[sel].astype(int)
        df["mkt_ret"] = [mkt["ret"].get(m, np.nan) for m in m_sel]
        df["mkt_ret15"] = [mkt["ret15"].get(m, np.nan) for m in m_sel]
        df["mkt_breadth"] = [mkt["breadth"].get(m, np.nan) for m in m_sel]
        for k, x in daily.items():
            if k != "prev_close":
                df[k] = x
        P = np.column_stack([minute, o, h, l, c, g["MA_5"], mid, g["VWAP"]]).astype(np.float64)
        eod_idx = np.flatnonzero(minute <= EOD_MIN)
        last = int(eod_idx[-1]) if len(eod_idx) else n - 1
        names = ["entry", "f15", "f30", "f60", "feod", "mfe60", "mae60", "mfe_eod"]
        names += [f"net_{p}" for p in pol] + [f"held_{p}" for p in pol]
        lab = {k: np.full(len(sel), np.nan) for k in names}
        for a, i in enumerate(sel):
            s = i + 1
            if s >= last:
                continue
            en = g027._up(o[s]) + g027._tick(o[s])
            lab["entry"][a] = en
            for name, k in (("f15", 15), ("f30", 30), ("f60", 60)):
                lab[name][a] = (c[min(s + k - 1, last)] / o[s] - 1.0) * 100.0
            lab["feod"][a] = (c[last] / o[s] - 1.0) * 100.0
            e60 = min(s + 60, last + 1)
            lab["mfe60"][a] = (h[s:e60].max() / o[s] - 1.0) * 100.0
            lab["mae60"][a] = (l[s:e60].min() / o[s] - 1.0) * 100.0
            lab["mfe_eod"][a] = (h[s:last + 1].max() / o[s] - 1.0) * 100.0
            for name, prm in pol.items():
                r, hb, _, _ = g027._sim_one(P, s, last - s + 1, en, f["atr_pct"][i], prm)
                lab[f"net_{name}"][a] = r
                lab[f"held_{name}"][a] = hb
        for k, x in lab.items():
            df[k] = x
        df["code"] = code
        df["bar"] = sel - off
        frames.append(df)
    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(res):
        res["date"] = trade_date
        fl = res.select_dtypes("float64").columns
        res[fl] = res[fl].astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    res.to_pickle(tmp)
    tmp.replace(out)
    return trade_date, len(res), time.time() - t0


def stage_panel(args) -> None:
    dates = sorted(p.stem for p in BARS_DIR.glob("*.pkl"))
    tasks = [(d, dates[k - 1] if k else None) for k, d in enumerate(dates) if not args.dates or d in args.dates]
    with Pool(args.workers) as pool:
        for name, n, sec in pool.imap_unordered(panel_day, tasks):
            print(f"panel {name} n={n} {sec:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("cache", "events", "real", "panel"), required=True)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"cache": stage_cache, "events": stage_events, "real": stage_real, "panel": stage_panel}[args.stage](args)


if __name__ == "__main__":
    main()
