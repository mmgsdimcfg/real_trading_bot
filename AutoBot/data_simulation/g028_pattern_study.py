# -*- coding: utf-8 -*-
"""Reversal / breakout pattern study on every 1-min bar (2026-11-12 사용자 요청: 차트의 V자 반등 / 박스 돌파 지점 매수 조건).

Uses the g026 bar cache (<runtime>/edge_ai/timing_study/bars). For every session bar in the entry window it stores the
state known at that bar's close (indicators of r002 + reversal / squeeze / breakout descriptors) and what a buy at the next
bar's open (+1 tick) would have returned under a few fixed exit policies (g027 simulator, costs included).

  python g028_pattern_study.py --stage scan [--workers 12]     -> timing_study/scan/scan_<date>.pkl
  python g028_pattern_study.py --stage show --code 006360 --date 20261111 --hhmm 1350 1400
  python g028_pattern_study.py --stage rules                   -> timing_study/report_rules.md
  python g028_pattern_study.py --stage model [--workers 12]    -> timing_study/report_pattern_model.md
  python g028_pattern_study.py --stage flow                    -> timing_study/report_flow.md (needs data/ticks)

Nothing here is imported by r003.

Update log
- [2026-11-12] type=feat - first version
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g026_timing_study as g026  # noqa: E402
import g027_timing_grid as g027  # noqa: E402

SCAN_DIR = g026.OUT_DIR / "scan"

# g027 PRM_COLS: stop, atr_k, hold, arm, give, mode, give2, p1, tp, cap
POLICIES = {
    "live": (1.7, 0, 4, 1.0, 0.4, 0, 0, 0, 0, 0),        # live proxy
    "tp3": (1.5, 0, 0, 99.0, 0.4, 0, 0, 0, 3.0, 0),      # +3% target / -1.5% stop, else EOD
    "tp3s1": (1.0, 0, 0, 99.0, 0.4, 0, 0, 0, 3.0, 0),    # +3% target / -1.0% stop
    "tp2": (1.0, 0, 0, 99.0, 0.4, 0, 0, 0, 2.0, 0),      # +2% target / -1.0% stop
    "run": (1.5, 0, 0, 1.5, 0.8, 0, 0, 0, 0, 0),         # -1.5% stop, trail 0.8% once +1.5%
    "run3": (1.5, 0, 0, 3.0, 1.0, 0, 0, 0, 0, 0),        # -1.5% stop, trail 1.0% once +3%
}


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _run_len(flag: np.ndarray) -> np.ndarray:
    """Bars since `flag` became true (0 = first true bar), -1 while false."""
    out = np.empty(len(flag))
    run = -1
    for i in range(len(flag)):
        run = run + 1 if flag[i] else -1
        out[i] = run
    return out


def _roll(x: np.ndarray, w: int, fn: str, minp: int | None = None) -> np.ndarray:
    return getattr(pd.Series(x).rolling(w, min_periods=minp or w), fn)().to_numpy()


def _since_argmin(x: np.ndarray, w: int) -> np.ndarray:
    """Bars since the lowest value of the last w bars (0 = the current bar is the low)."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        a = max(0, i - w + 1)
        seg = x[a:i + 1]
        if np.isfinite(seg).any():
            out[i] = len(seg) - 1 - int(np.nanargmin(seg))
    return out


@njit(cache=True)
def _labels(P, sel, last, atr_pct, prms):
    """Per selected trigger bar i: buy at bar i+1 open (+1 tick). -> entry, net[K], kind[K], held[K], mfe, mae."""
    n, K = sel.shape[0], prms.shape[0]
    entry = np.full(n, np.nan)
    net = np.full((n, K), np.nan)
    kind = np.zeros((n, K))
    held = np.zeros((n, K))
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for a in range(n):
        s = sel[a] + 1
        if s >= last:
            continue
        o = P[s, 1]
        en = g027._up(o) + g027._tick(o)
        entry[a] = en
        hi, lo = P[s, 2], P[s, 3]
        for r in range(s, last + 1):
            if P[r, 2] > hi:
                hi = P[r, 2]
            if P[r, 3] < lo:
                lo = P[r, 3]
        mfe[a] = (hi / en - 1.0) * 100.0
        mae[a] = (lo / en - 1.0) * 100.0
        for k in range(K):
            r_, hb, _, kd = g027._sim_one(P, s, last - s + 1, en, atr_pct[sel[a]], prms[k])
            net[a, k] = r_
            kind[a, k] = kd
            held[a, k] = hb
    return entry, net, kind, held, mfe, mae


def feature_arrays(ind: pd.DataFrame, off: int, prev_close: float) -> dict[str, np.ndarray]:
    """Per-bar state (all known at the bar close). ind = g026._indicator_frame output."""
    g = {k: ind[k].to_numpy(dtype=np.float64) for k in ind.columns}
    c, o, h, l, v, minute = g["close"], g["open"], g["high"], g["low"], g["volume"], g["minute"]
    n = len(c)
    mid = g["BB_MIDDLE"]
    o_day = o[off]
    sess_hi = np.full(n, np.nan)
    sess_lo = np.full(n, np.nan)
    sess_hi[off:] = np.maximum.accumulate(h[off:])
    sess_lo[off:] = np.minimum.accumulate(l[off:])
    with np.errstate(invalid="ignore", divide="ignore"):
        hi30, lo30 = _roll(h, 30, "max", 5), _roll(l, 30, "min", 5)
        hi60 = _roll(h, 60, "max", 5)
        hi20p, lo20p = _shift(_roll(h, 20, "max", 5), 1), _shift(_roll(l, 20, "min", 5), 1)   # previous 20 bars
        hi60p = _shift(hi60, 1)
        tenkan = (_roll(h, 9, "max") + _roll(l, 9, "min")) / 2.0
        kijun = (_roll(h, 26, "max") + _roll(l, 26, "min")) / 2.0
        bbw = (g["BB_UPPER"] - g["BB_LOWER"]) / mid * 100.0
        pctb = (c - g["BB_LOWER"]) / (g["BB_UPPER"] - g["BB_LOWER"])
        hist = g["MACD_HIST"]
        rsi_sig = g["RSI"] - g["RSI_SIGNAL"]
        dsp = g["DI_PLUS"] - g["DI_MINUS"]
        vr = v / g["VOL_MA20"]
        upv = np.where(c >= o, v, 0.0)
        rng = h - l
        f = {
            "minute": minute, "price": c,
            "rsi": g["RSI"], "rsi_sig": rsi_sig, "rsi_min10": _roll(g["RSI"], 10, "min", 3),
            "rsi_x_age": _run_len(rsi_sig > 0),
            "stoch_k": g["STOCH_K"], "stoch_kd": g["STOCH_K"] - g["STOCH_D"], "williams": g["WILLIAMS_R"], "mfi": g["MFI"],
            "macd_pct": g["MACD"] / c * 100.0, "macd_hist_pct": hist / c * 100.0,
            "macd_hist_d": (hist - _shift(hist, 1)) / c * 100.0, "macd_x_age": _run_len(hist > 0),
            "macd_min10": _roll(g["MACD"] / c * 100.0, 10, "min", 3),
            "adx": g["ADX"], "adx_d3": g["ADX"] - _shift(g["ADX"], 3), "adx_max10": _roll(g["ADX"], 10, "max", 3),
            "di_plus": g["DI_PLUS"], "di_minus": g["DI_MINUS"], "di_spread": dsp, "di_spread_d3": dsp - _shift(dsp, 3),
            "di_x_age": _run_len(dsp > 0),
            "vol_ratio": vr, "vol5_ratio": _roll(v, 5, "mean") / g["VOL_MA20"], "vol_max10": _roll(vr, 10, "max", 3),
            "upvol10": _roll(upv, 10, "sum", 3) / _roll(v, 10, "sum", 3),
            "bb_width": bbw, "bw_rel": bbw / _roll(bbw, 120, "median", 30), "bw_min20": _roll(bbw, 20, "min", 5),
            "bb_pctb": pctb, "pctb_min10": _roll(pctb, 10, "min", 3),
            "mid_gap": (c / mid - 1.0) * 100.0, "ma5_gap": (g["MA_5"] / mid - 1.0) * 100.0,
            "ma5_slope": (g["MA_5"] / _shift(g["MA_5"], 1) - 1.0) * 100.0,
            "mid_slope5": (mid / _shift(mid, 5) - 1.0) * 100.0, "mid_slope20": (mid / _shift(mid, 20) - 1.0) * 100.0,
            "ema20_60": (g["EMA_20"] / g["EMA_60"] - 1.0) * 100.0, "vwap_gap": (c / g["VWAP"] - 1.0) * 100.0,
            "ten_gap": (c / tenkan - 1.0) * 100.0, "kij_gap": (c / kijun - 1.0) * 100.0,
            "ten_kij": (tenkan / kijun - 1.0) * 100.0, "ten_x_age": _run_len(c > tenkan),
            "atr_pct": g["ATR"] / c * 100.0,
            "ret1": (c / _shift(c, 1) - 1.0) * 100.0, "ret3": (c / _shift(c, 3) - 1.0) * 100.0,
            "ret5": (c / _shift(c, 5) - 1.0) * 100.0, "ret15": (c / _shift(c, 15) - 1.0) * 100.0,
            "ret30": (c / _shift(c, 30) - 1.0) * 100.0, "ret60": (c / _shift(c, 60) - 1.0) * 100.0,
            "day_ret": (c / o_day - 1.0) * 100.0, "prev_ret": (c / prev_close - 1.0) * 100.0,
            "gap_ret": np.full(n, (o_day / prev_close - 1.0) * 100.0),
            "hod_dist": (c / sess_hi - 1.0) * 100.0, "lod_dist": (c / sess_lo - 1.0) * 100.0,
            "dd30": (c / hi30 - 1.0) * 100.0, "dd60": (c / hi60 - 1.0) * 100.0, "bounce30": (c / lo30 - 1.0) * 100.0,
            "low_age30": _since_argmin(l, 30), "fall30": (lo30 / hi30 - 1.0) * 100.0,
            "range20": (hi20p / lo20p - 1.0) * 100.0, "brk20": (c / hi20p - 1.0) * 100.0, "brk60": (c / hi60p - 1.0) * 100.0,
            "candle_pct": (c / o - 1.0) * 100.0, "clv": np.where(rng > 0, (c - l) / rng, 0.5),
            "bull3": _roll((c > o).astype(float), 3, "sum"), "hi_brk3": (c / _shift(_roll(h, 3, "max"), 1) - 1.0) * 100.0,
            "obv_up": (g["OBV"] > g["OBV_MA"]).astype(float),
            "above_age": _run_len((g["MA_5"] > mid) & (c > mid)),
            "tick_pct": g027.krx_tick(c) / c * 100.0,
        }
    return f


def scan_day(task) -> tuple[str, int, float]:
    trade_date, prev_date, force = task
    t0 = time.time()
    out = SCAN_DIR / f"scan_{trade_date}.pkl"
    if out.exists() and not force:
        return trade_date, -1, 0.0
    day = g026.load_day(trade_date)
    if day is None:
        return trade_date, 0, 0.0
    prev = g026.load_day(prev_date) if prev_date else None
    mkt = g026._market(day["bars"])
    prms = np.array(list(POLICIES.values()), dtype=np.float64)
    frames = []
    for code, arr in day["bars"].items():
        ind, off = g026._indicator_frame(arr, prev["bars"].get(code) if prev else None)
        n = len(ind)
        daily = day["daily"].get(code, {})
        f = feature_arrays(ind, off, daily.get("prev_close", np.nan))
        minute = f["minute"]
        pos = np.arange(n)
        sel = np.flatnonzero((pos >= off + 1) & (minute >= g026.ENTRY_START_MIN) & (minute <= g026.ENTRY_END_MIN) & (pos < n - 3))
        if not len(sel):
            continue
        df = pd.DataFrame({k: np.asarray(x)[sel] for k, x in f.items()})
        m_sel = minute[sel].astype(int)
        df["mkt_ret"] = [mkt["ret"].get(m, np.nan) for m in m_sel]
        df["mkt_ret15"] = [mkt["ret15"].get(m, np.nan) for m in m_sel]
        df["mkt_breadth"] = [mkt["breadth"].get(m, np.nan) for m in m_sel]
        for k, x in daily.items():
            if k != "prev_close":
                df[k] = x
        P = np.column_stack([minute, ind["open"], ind["high"], ind["low"], ind["close"], ind["MA_5"], ind["BB_MIDDLE"],
                             ind["VWAP"]]).astype(np.float64)
        eod_idx = np.flatnonzero(minute <= g026.EOD_MIN)
        last = int(eod_idx[-1]) if len(eod_idx) else n - 1
        entry, net, kind, held, mfe, mae = _labels(P, sel.astype(np.int64), last, f["atr_pct"], prms)
        df["entry"], df["mfe"], df["mae"] = entry, mfe, mae
        for k, name in enumerate(POLICIES):
            df[f"net_{name}"], df[f"kind_{name}"], df[f"held_{name}"] = net[:, k], kind[:, k], held[:, k]
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


def stage_scan(args) -> None:
    dates = sorted(p.stem for p in g026.BARS_DIR.glob("*.pkl"))
    tasks = [(d, dates[k - 1] if k else None, args.force) for k, d in enumerate(dates) if not args.dates or d in args.dates]
    with Pool(args.workers) as pool:
        for name, n, sec in pool.imap_unordered(scan_day, tasks):
            print(f"scan {name} n={n} {sec:.0f}s", flush=True)


def load_scan(dates: list[str] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(SCAN_DIR.glob("scan_*.pkl"))
    frames = []
    for f in files:
        if dates and f.stem[5:] not in dates:
            continue
        d = pd.read_pickle(f)
        if len(d):
            frames.append(d.reindex(columns=columns) if columns else d)   # a day without daily data lacks the d_* columns
    return pd.concat(frames, ignore_index=True)


def stage_show(args) -> None:
    df = load_scan([args.date])
    df = df[df["code"] == args.code]
    lo, hi = (int(x[:2]) * 60 + int(x[2:]) for x in args.hhmm)
    df = df[(df["minute"] >= lo) & (df["minute"] <= hi)]
    pd.set_option("display.width", 250, "display.max_columns", 200, "display.max_rows", 500)
    cols = [c for c in df.columns if c not in ("code", "date")]
    print(df[cols].T.to_string(float_format=lambda x: f"{x:.2f}"))


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
REPORT_POLICIES = ("live", "tp3", "tp3s1", "tp2", "run", "run3")


def _valid_scan(columns: list[str] | None = None) -> pd.DataFrame:
    d = load_scan(columns=columns)
    return d[np.isfinite(d["entry"])].reset_index(drop=True)


def rule_masks(g: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hand-written versions of the two chart examples of 2026-11-11 (006360 13:53~13:56, 010060 12:37~12:41).
    turn = DI+ crosses above DI- on this bar while the MACD histogram and RSI-vs-signal turned positive within the last
    4 bars and the close is above the Ichimoku conversion line. early = the 1-min MA5 is still below MA20 (the live
    trigger has not fired yet)."""
    early = g["above_age"] < 0
    turn = ((g["di_x_age"] == 0) & (g["macd_x_age"] >= 0) & (g["macd_x_age"] <= 4) & (g["rsi_x_age"] >= 0)
            & (g["rsi_x_age"] <= 4) & (g["ten_x_age"] >= 0))
    te = turn & early
    base = turn & (g["range20"] <= 1.2) & (g["bw_rel"] < 1.0)
    brk = (g["brk20"] > 0) & (g["vol_ratio"] >= 3) & (g["bw_rel"] < 1.0)
    rsi_now = (g["rsi_x_age"] == 0) & (g["dd30"] <= -3) & (g["rsi_min10"] < 20)
    return {
        "all bars": np.ones(len(early), bool),
        "turn": turn,
        "turn, MA5<MA20": te,
        "turn early, 30m drawdown <= -1%": te & (g["dd30"] <= -1),
        "turn early, 30m drawdown <= -2%": te & (g["dd30"] <= -2),
        "turn early, 30m drawdown <= -3%": te & (g["dd30"] <= -3),
        "turn early, 30m drawdown <= -4%": te & (g["dd30"] <= -4),
        "V: dd<=-3, RSI low<25, ADX high>=60": te & (g["dd30"] <= -3) & (g["rsi_min10"] < 25) & (g["adx_max10"] >= 60),
        "RSI crosses signal now, dd<=-3, RSI low<20": rsi_now,
        "  + daily uptrend (D-1 close > MA20)": rsi_now & (g["macd_x_age"] >= 0) & (g["d_close_ma20_pct"] > 0),
        "base: turn, 20-bar range<=1.2%, BB narrow": base,
        "  + daily uptrend": base & (g["d_close_ma20_pct"] > 0),
        "breakout: >20-bar high, vol>=3x, BB narrow": brk,
        "  + range<=1.2% + daily up + day up": brk & (g["range20"] <= 1.2) & (g["d_close_ma20_pct"] > 0) & (g["day_ret"] > 0),
        "turn early dd<=-2, tick<=0.13%": te & (g["dd30"] <= -2) & (g["tick_pct"] <= 0.13),
        "turn early dd<=-2, tick>0.13%": te & (g["dd30"] <= -2) & (g["tick_pct"] > 0.13),
    }


RULE_COLS = ["date", "above_age", "di_x_age", "macd_x_age", "rsi_x_age", "ten_x_age", "range20", "bw_rel", "brk20", "vol_ratio",
             "dd30", "rsi_min10", "adx_max10", "d_close_ma20_pct", "day_ret", "tick_pct", "mfe", "entry", "kind_tp3"]


def stage_rules(args) -> None:
    d = _valid_scan(RULE_COLS + [f"net_{p}" for p in REPORT_POLICIES])
    dates = sorted(d["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    tr = (d["date"] < cut).to_numpy()
    dt = d["date"].to_numpy()
    g = {c: d[c].to_numpy() for c in RULE_COLS if c != "date"}
    lines = [f"# g028 rules - {len(d):,} bars, {len(dates)} days, holdout from {cut}", "",
             "Buy at the next bar's open + 1 tick. mfe3 = +3% seen before EOD, hit3 = +3% reached before -1.5%, "
             "stop = -1.5% first. net = mean % per trade after costs (train / holdout, t = by-day t-stat).", "",
             "| rule | n | /day | mfe3 | hit3 | stop | " + " | ".join(REPORT_POLICIES) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(REPORT_POLICIES)]
    for name, m in rule_masks(g).items():
        n = int(m.sum())
        if n < 30:
            lines.append(f"| {name} | {n} |" + " |" * (4 + len(REPORT_POLICIES)))
            continue
        cells = [name, f"{n:,}", f"{n / len(dates):.1f}", f"{(g['mfe'][m] >= 3).mean() * 100:.1f}%",
                 f"{(g['kind_tp3'][m] == 3).mean() * 100:.1f}%", f"{(g['kind_tp3'][m] == 1).mean() * 100:.1f}%"]
        for p in REPORT_POLICIES:
            x = d[f"net_{p}"].to_numpy()[m]
            a, b = x[tr[m]], x[~tr[m]]
            sday = pd.Series(x).groupby(dt[m]).sum()
            t = sday.mean() / (sday.std(ddof=1) / np.sqrt(len(sday))) if len(sday) > 2 else np.nan
            cells.append(f"{x.mean():+.2f} (w{(x > 0).mean() * 100:.0f}%, {a.mean() if len(a) else np.nan:+.2f} / "
                         f"{b.mean() if len(b) else np.nan:+.2f}, t{t:+.1f})")
        lines.append("| " + " | ".join(cells) + " |")
    out = g026.OUT_DIR / "report_rules.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("->", out)


def _logit_fit(xt: np.ndarray, y: np.ndarray):
    from scipy.optimize import minimize
    mu, sd = np.nanmean(xt, 0), np.nanstd(xt, 0) + 1e-9
    prep = lambda x: np.c_[np.clip(np.nan_to_num((x - mu) / sd), -5, 5), np.ones(len(x))]  # noqa: E731
    z = prep(xt)

    def loss(w):
        pr = 1 / (1 + np.exp(-(z @ w)))
        val = -np.mean(y * np.log(pr + 1e-9) + (1 - y) * np.log(1 - pr + 1e-9)) + 1e-3 * (w[:-1] ** 2).sum()
        return val, z.T @ (pr - y) / len(y) + 2e-3 * np.r_[w[:-1], 0]

    w = minimize(loss, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 200}).x
    return lambda x: prep(x) @ w


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    rank = np.empty(len(s))
    rank[order] = np.arange(len(s))
    n1 = y.sum()
    if n1 == 0 or n1 == len(y):
        return float("nan")       # a fold without both classes
    return float((rank[y > 0].sum() - n1 * (n1 - 1) / 2) / (n1 * (len(y) - n1)))


def stage_model(args) -> None:
    """Walk-forward (6 test folds over the last 60% of days, 1-day embargo): LightGBM vs logistic regression, three labels.
    The cut is the test fold's own score quantile = optimistic; a model that is negative here has nothing to offer."""
    import lightgbm as lgb
    d = _valid_scan()
    lab = ["entry", "mfe", "mae", "code", "bar", "date", "price"] + [c for c in d.columns if c.startswith(("net_", "kind_", "held_"))]
    feats = [c for c in d.columns if c not in lab]
    dates = sorted(d["date"].unique())
    dt = d["date"].to_numpy()
    x = d[feats].to_numpy(dtype=np.float32)
    pol = ["live", "tp3", "tp2", "run", "run3"]
    net = {p: d[f"net_{p}"].to_numpy() for p in pol}
    hit3 = (d["kind_tp3"].to_numpy() == 3).astype(np.float32)
    targets = {"hit3 (+3% before -1.5%)": hit3, "run3 net > +0.5%": (net["run3"] > 0.5).astype(np.float32),
               "live exit net > 0": (net["live"] > 0).astype(np.float32)}
    bounds = np.linspace(int(len(dates) * 0.4), len(dates), 7).astype(int)
    folds = [(np.flatnonzero(np.isin(dt, dates[:bounds[k] - 1])), np.flatnonzero(np.isin(dt, dates[bounds[k]:bounds[k + 1]])))
             for k in range(6)]
    rng = np.random.default_rng(1)
    lines = [f"# g028 model - {len(d):,} bars, {len(feats)} features, {len(dates)} days", "",
             f"Base in the test folds: hit3 {hit3[np.concatenate([te for _, te in folds])].mean() * 100:.1f}%, "
             + ", ".join(f"{p} {net[p][np.concatenate([te for _, te in folds])].mean():+.3f}" for p in pol), ""]
    for tname, y in targets.items():
        for model in ("lgbm", "logit"):
            score = np.full(len(d), np.nan, dtype=np.float32)
            aucs = []
            for tr, te in folds:
                tr = rng.choice(tr, size=min(len(tr), 900_000 if model == "lgbm" else 300_000), replace=False)
                if model == "lgbm":
                    b = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 500,
                                   "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 5.0,
                                   "verbose": -1, "num_threads": args.workers}, lgb.Dataset(x[tr], y[tr]), num_boost_round=250)
                    score[te] = b.predict(x[te])
                else:
                    score[te] = _logit_fit(x[tr], y[tr])(x[te])
                aucs.append(_auc(y[te], score[te]))
            lines += [f"## label: {tname} / model: {model} - AUC by fold {' '.join(f'{a:.3f}' for a in aucs)}", "",
                      "| top share | n | hit3 | " + " | ".join(pol) + " |", "|---|---|---|" + "---|" * len(pol)]
            for q in (0.10, 0.01, 0.002, 0.0005):
                pick = np.zeros(len(d), bool)
                for _, te in folds:
                    pick[te[score[te] >= np.quantile(score[te], 1 - q)]] = True
                cells = [f"{q * 100:.2f}%", f"{pick.sum():,}", f"{hit3[pick].mean() * 100:.1f}%"]
                for p in pol:
                    v = net[p][pick]
                    cells.append(f"{v.mean():+.3f} (w{(v > 0).mean() * 100:.0f}%, day avg {pd.Series(v).groupby(dt[pick]).mean().mean():+.2f})")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
            print("\n".join(lines[-8:]), flush=True)
    out = g026.OUT_DIR / "report_pattern_model.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", out)


# ---------------------------------------------------------------------------
# order flow from the g011 tick files (ccld_dvsn 1 = buyer-initiated, 5 = seller-initiated)
# ---------------------------------------------------------------------------
TICK_ROOT = g026.g013.DATA_ROOT / "ticks"
BIG_TRADE_KRW = 30_000_000.0
FLOW_FEATS = ["flow1", "flow5", "flow15", "flow30", "strength_day", "big_flow15", "big_share15", "ntr_ratio", "size_ratio",
              "buy_n_share5", "spread_ticks", "flow5_d"]


def flow_minutes(path: Path) -> pd.DataFrame | None:
    """Tick file -> per-minute order-flow state, minute-END labels like the bars (minute M = ticks of [M-1, M)).
    Everything at label M uses ticks before M:00 only."""
    want = ("trade_time", "price", "cntg_vol", "ccld_dvsn", "askp1", "bidp1", "venue")
    try:
        t = pd.read_csv(path, usecols=lambda c: c in want, dtype={"trade_time": str})
    except Exception:
        return None
    if "venue" in t.columns:                       # files before 2026-11-06 have no venue column (KRX only)
        t = t[t["venue"] == "KRX"]
    t = t[(t["trade_time"] >= "090000") & (t["trade_time"] < "152000")]      # continuous session only
    if len(t) < 200:
        return None
    hh = t["trade_time"].str.slice(0, 2).astype(int)
    mm = t["trade_time"].str.slice(2, 4).astype(int)
    t = t.assign(minute=hh * 60 + mm + 1)
    vol, px = t["cntg_vol"].to_numpy(dtype=np.float64), t["price"].to_numpy(dtype=np.float64)
    buy = (t["ccld_dvsn"].to_numpy() == 1).astype(np.float64)
    sell = (t["ccld_dvsn"].to_numpy() == 5).astype(np.float64)
    big = (vol * px >= BIG_TRADE_KRW).astype(np.float64)
    spread = (t["askp1"].to_numpy(dtype=np.float64) - t["bidp1"].to_numpy(dtype=np.float64)) / g027.krx_tick(px)
    a = pd.DataFrame({"minute": t["minute"].to_numpy(), "b": vol * buy, "s": vol * sell, "n": 1.0, "bn": buy, "sn": sell,
                      "bb": vol * buy * big, "bs": vol * sell * big, "spread": spread})
    g = a.groupby("minute")
    m = g[["b", "s", "n", "bn", "sn", "bb", "bs"]].sum()
    m["spread"] = g["spread"].last()
    grid = np.arange(9 * 60 + 1, 15 * 60 + 31)
    m = m.reindex(grid)
    has = m["n"].notna()
    z = m.fillna(0.0)
    r = lambda col, w: z[col].rolling(w, min_periods=1).sum()  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(index=grid)
        for w in (1, 5, 15, 30):
            out[f"flow{w}"] = (r("b", w) - r("s", w)) / (r("b", w) + r("s", w))
        out["strength_day"] = z["b"].cumsum() / z["s"].cumsum()
        v15 = r("b", 15) + r("s", 15)
        out["big_flow15"] = (r("bb", 15) - r("bs", 15)) / v15
        out["big_share15"] = (r("bb", 15) + r("bs", 15)) / v15
        out["ntr_ratio"] = r("n", 5) / 5.0 / (r("n", 60) / has.rolling(60, min_periods=1).sum())
        out["size_ratio"] = (v15 / r("n", 15)) / ((r("b", 120) + r("s", 120)) / r("n", 120))
        out["buy_n_share5"] = r("bn", 5) / (r("bn", 5) + r("sn", 5))
        out["spread_ticks"] = m["spread"].ffill(limit=5)
        out["flow5_d"] = out["flow5"] - out["flow5"].shift(5)
        out["cov15"] = has.rolling(15, min_periods=1).mean()
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "minute"
    return out.reset_index()


def load_flow() -> pd.DataFrame:
    """Scan rows of the tick days joined with the order-flow state (only minutes whose last 15 minutes are fully covered)."""
    frames = []
    for day_dir in sorted(p for p in TICK_ROOT.iterdir() if p.is_dir()):
        f = SCAN_DIR / f"scan_{day_dir.name}.pkl"
        if not f.exists():
            continue
        sc = pd.read_pickle(f)
        parts = []
        for tf in sorted(day_dir.glob("*.csv")):
            fm = flow_minutes(tf)
            if fm is not None:
                fm["code"] = tf.stem
                parts.append(fm)
        if not parts:
            continue
        fl = pd.concat(parts, ignore_index=True)
        sc["minute"] = sc["minute"].astype(int)
        j = sc.merge(fl, on=["code", "minute"], how="inner")
        frames.append(j[(j["cov15"] >= 0.999) & np.isfinite(j["entry"])])
    return pd.concat(frames, ignore_index=True)


def _day_table(d: pd.DataFrame, mask: np.ndarray, name: str, cols=("net_live", "net_tp3", "net_run")) -> str:
    sub = d[mask]
    if len(sub) < 30:
        return f"| {name} | {len(sub)} |" + " |" * (3 + len(cols))
    cells = [name, f"{len(sub):,}", f"{sub['date'].nunique()}", f"{(sub['kind_tp3'] == 3).mean() * 100:.1f}%",
             f"{(sub['kind_tp3'] == 1).mean() * 100:.1f}%"]
    for c in cols:
        by = sub.groupby("date")[c].mean()
        cells.append(f"{sub[c].mean():+.2f} (w{(sub[c] > 0).mean() * 100:.0f}%, days+ {(by > 0).sum()}/{len(by)})")
    return "| " + " | ".join(cells) + " |"


def stage_flow(args) -> None:
    import lightgbm as lgb
    d = load_flow()
    days = sorted(d["date"].unique())
    head = "| {} | n | days | hit3 | stop | live | tp3 | run |\n|---|---|---|---|---|---|---|---|"
    lines = [f"# g028 order flow - {len(d):,} bars, {d.groupby(['date', 'code']).ngroups} code-days, {len(days)} days "
             f"({days[0]}~{days[-1]})", "",
             "Order flow from the g011 tick files joined to the 1-min scan. net = mean % after costs, days+ = days with a "
             "positive mean. The sample is the watchlist of each day only (8~41 symbols).", "",
             head.format("all"), _day_table(d, np.ones(len(d), bool), "all bars"), ""]
    for col in ("flow5", "flow15", "flow30", "strength_day", "big_flow15", "buy_n_share5", "ntr_ratio", "size_ratio", "flow5_d"):
        x = d[col].to_numpy()
        qs = np.nanquantile(x, [0.1, 0.3, 0.7, 0.9])
        lines += [head.format(col)]
        edges = [-np.inf, *qs, np.inf]
        for k, lab in enumerate(("bottom 10%", "10~30%", "30~70%", "70~90%", "top 10%")):
            lines.append(_day_table(d, (x > edges[k]) & (x <= edges[k + 1]), f"{lab} ({edges[k]:+.2f}~{edges[k + 1]:+.2f})"))
        lines.append("")
    g = {c: d[c].to_numpy() for c in RULE_COLS if c != "date"}
    rm = rule_masks(g)
    te = rm["turn, MA5<MA20"]
    f5, f15, bf = d["flow5"].to_numpy(), d["flow15"].to_numpy(), d["big_flow15"].to_numpy()
    lines += [head.format("chart pattern x order flow"),
              _day_table(d, te, "turn, MA5<MA20"),
              _day_table(d, te & (f5 > 0.2), "  + flow5 > +0.2"),
              _day_table(d, te & (f5 > 0.2) & (f15 > 0), "  + flow5 > +0.2 & flow15 > 0"),
              _day_table(d, te & (bf > 0.05), "  + big-trade flow15 > +0.05"),
              _day_table(d, te & (g["dd30"] <= -2), "turn early, drawdown <= -2%"),
              _day_table(d, te & (g["dd30"] <= -2) & (f5 > 0.2), "  + flow5 > +0.2"),
              _day_table(d, (g["dd30"] <= -2) & (f15 > 0.2), "any bar: drawdown <= -2% & flow15 > +0.2 (absorption)"),
              _day_table(d, (g["brk20"] > 0) & (f5 > 0.3), "any bar: 20-bar high breakout & flow5 > +0.3"),
              _day_table(d, (g["brk20"] > 0) & (f5 > 0.3) & (bf > 0.05), "  + big-trade flow15 > +0.05"), ""]
    # leave-one-day-out LightGBM, with and without the flow features
    lab = ["entry", "mfe", "mae", "code", "bar", "date", "price", "cov15"] + [c for c in d.columns if c.startswith(("net_", "kind_", "held_"))]
    base = [c for c in d.columns if c not in lab and c not in FLOW_FEATS]
    dt = d["date"].to_numpy()
    for tname, y in (("live exit net > 0", (d["net_live"] > 0).to_numpy(dtype=np.float32)),
                     ("hit3", (d["kind_tp3"] == 3).to_numpy(dtype=np.float32))):
        for fname, feats in (("price only", base), ("price + flow", base + FLOW_FEATS), ("flow only", FLOW_FEATS)):
            x = d[feats].to_numpy(dtype=np.float32)
            score = np.zeros(len(d), dtype=np.float32)
            for day in days:
                tr, tst = dt != day, dt == day
                b = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
                               "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 5.0,
                               "verbose": -1, "num_threads": args.workers, "seed": 1}, lgb.Dataset(x[tr], y[tr]),
                              num_boost_round=150)
                score[tst] = b.predict(x[tst])
            aucs = [_auc(y[dt == day], score[dt == day]) for day in days]
            pick = np.zeros(len(d), bool)
            for day in days:
                i = np.flatnonzero(dt == day)
                pick[i[score[i] >= np.quantile(score[i], 0.95)]] = True
            lines += [f"## leave-one-day-out LightGBM - label {tname}, features {fname}: AUC mean {np.nanmean(aucs):.3f} "
                      f"(by day {' '.join(f'{a:.2f}' for a in aucs)})", "", head.format("top 5% per day"),
                      _day_table(d, pick, "picked"), ""]
    out = g026.OUT_DIR / "report_flow.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("->", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("scan", "show", "rules", "model", "flow"), required=True)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--code")
    ap.add_argument("--date")
    ap.add_argument("--hhmm", nargs=2)
    args = ap.parse_args()
    {"scan": stage_scan, "show": stage_show, "rules": stage_rules, "model": stage_model, "flow": stage_flow}[args.stage](args)


if __name__ == "__main__":
    main()
