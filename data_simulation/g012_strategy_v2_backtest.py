# -*- coding: utf-8 -*-
"""g012_strategy_v2_backtest.py - portfolio backtest for Strategy V2 (r008) with costs and walk-forward.

Strategies
  A : daily filter + 5m context + 1m trigger                      (r008 only)
  B : A + 120-tick momentum filter at the trigger                 (needs runtime/data/ticks/<D>/)
  C : A + 60-tick momentum filter                                 (needs runtime/data/ticks/<D>/)
  B/C are evaluated only on dates that have tick files, and A is re-reported on the SAME dates/symbols
  so the comparison is like-for-like.

No look-ahead
  - Universe for day D uses daily bars with date < D only (daily_features row of D-1).
  - Universe membership requires a 10s file in runtime/data/<D>/ (symbol was trading on D).
  - 1m bars are the REAL minute bars (raw_bar==1 rows of the 10s file; volume x6 undoes g001's /6 split),
    labelled by bar END. Signals use completed bars only; entries fill at the NEXT 1m bar's open.
  - Exits are evaluated inside each 1m bar pessimistically (stop before target).

Costs
  commission each side + sell tax (r001 V2_COMMISSION_PCT / V2_SELL_TAX_PCT) + slippage in KRX ticks
  (+ extra bps for the small-liquidity tier). TP1 is a resting limit -> no slippage.

Known biases (reported, not hidden)
  - Survivorship: the daily panel is read from the latest collected folder, so symbols delisted after D are
    missing from D's universe (small for a 4-month window).
  - Minutes without trades have no 1m bar; if the next bar after a signal is > 2 minutes later the entry is skipped.
  - No order book history: spread filter is approximated by V2_MAX_TICK_PCT (tick/price) and the slippage model.

Run
  python g012_strategy_v2_backtest.py --start 20260701 --end 20261022
  python g012_strategy_v2_backtest.py --start 20260701 --end 20261022 --walk-forward --train-days 20 --test-days 5
  python g012_strategy_v2_backtest.py --start 20261026 --end 20261130 --strategies A B C

Update log:
- [2026-10-23] type=feat owner=claude
    summary: 신규 - 전략 V2 포트폴리오 백테스트(전략 A/B/C, 수수료·세금·슬리피지, 승률/손익비/PF/MDD/거래수,
      유동성 티어·셋업·시간대별 분해, 학습/검증 분리 + walk-forward).
    impact: sim
    compatibility: backward-compatible (신규 파일, 결과는 runtime/log/v2_backtest/)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))

import r001_define_config as CFGMOD  # noqa: E402
from r001_define_config import RUNTIME_DATA_DIR, RUNTIME_LOG_DIR  # noqa: E402
from r008_strategy_v2 import (  # noqa: E402
    DayRisk, EntryState, V2Config, V2Position, build_5m, completed_5m, daily_features, fill_cost_krw,
    judge_daily_row, market_breadth_ok, prepare_1m, select_universe, size_position, slippage_price, step_entry, step_exit,
)
from r009_tick_bars import build_tick_bars, iter_tick_file, tick_momentum  # noqa: E402

OUT_DIR = RUNTIME_LOG_DIR / "v2_backtest"
CACHE_DIR = RUNTIME_DATA_DIR / "_v2_cache"
SESSION_LAST_ENTRY_BAR_START = "15:19"  # continuous session ends 15:20 (closing auction after)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def trading_dates(start: str, end: str) -> list[str]:
    out = []
    for p in sorted(RUNTIME_DATA_DIR.iterdir()):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8 and start <= p.name <= end:
            if any(p.glob("*_10s.txt")):
                out.append(p.name)
    return out


def _code_files(day_dir: Path, suffix: str) -> dict[str, Path]:
    out = {}
    for f in day_dir.glob(f"*_{suffix}"):
        code = f.name.split("_", 1)[0]
        if code.isdigit():
            out[code] = f
    return out


def load_daily_panel(source_date: str, cfg: V2Config) -> dict[str, pd.DataFrame]:
    """Per-code daily_features frames indexed by int date, from <source_date>'s daily csv files (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"daily_feat_{source_date}_{cfg.daily_ma_fast}_{cfg.daily_ma_slow}_{cfg.daily_ma_fast_slope_days}_{cfg.daily_ma_slow_slope_days}.pkl"
    cache = CACHE_DIR / key
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    panel: dict[str, pd.DataFrame] = {}
    files = _code_files(RUNTIME_DATA_DIR / source_date, "daily.csv")
    for i, (code, path) in enumerate(sorted(files.items())):
        try:
            df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume", "amount"], encoding="utf-8-sig")
        except Exception:
            continue
        df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates("date", keep="last")
        if df.empty:
            continue
        df["date"] = df["date"].astype(int)
        df = df.set_index("date")
        panel[code] = daily_features(df, cfg)
        if (i + 1) % 500 == 0:
            print(f"  daily panel {i + 1}/{len(files)}", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(panel, f)
    return panel


def load_1m_bars(path: Path, day: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=["datetime", "open", "high", "low", "close", "volume", "market", "amount", "raw_bar"],
                         encoding="utf-8-sig")
    except Exception:
        return None
    df = df[(df["raw_bar"] == 1)]
    if "market" in df.columns:
        df = df[df["market"].astype(str).isin(["J", "UN"])]
    if df.empty:
        return None
    start = pd.to_datetime(df["datetime"])
    keep = ((start >= pd.Timestamp(f"{day} 09:00")) & (start <= pd.Timestamp(f"{day} {SESSION_LAST_ENTRY_BAR_START}"))).to_numpy()
    df = df.assign(volume=df["volume"].astype(float) * 6.0)
    df.index = pd.DatetimeIndex(start + pd.Timedelta(minutes=1))  # label by bar END
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        return None
    if df["amount"].isna().any() or float(df["amount"].fillna(0).sum()) <= 0:
        df = df.drop(columns=["amount"])
    return df[[c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]].astype(float)


# ---------------------------------------------------------------------------
# One day, one or more configs
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    date: str
    code: str
    setup: str
    tier: str
    variant: str
    entry_ts: str
    entry_price: float
    qty: int
    initial_risk: float
    exit_ts: str = ""
    exit_reasons: str = ""
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    r_net: float = 0.0
    hold_min: float = 0.0
    entry_hour: int = 0


def _tick_filter_for(variant: str) -> int | None:
    # "A_T" = strategy A restricted to the tick-covered symbol set (like-for-like baseline for B/C)
    return {"A": None, "A_T": None, "B": 120, "C": 60}[variant]


def simulate_day(day: str, universe, bars: dict[str, pd.DataFrame], cfg: V2Config, variant: str,
                 tick_dir: Path | None) -> tuple[list[Trade], dict]:
    tick_n = _tick_filter_for(variant)
    tiers = {u.code: u.tier for u in universe}
    prepared: dict[str, pd.DataFrame] = {}
    m5s: dict[str, pd.DataFrame] = {}
    for code, m1 in bars.items():
        p = prepare_1m(m1, cfg)
        prepared[code] = p
        m5s[code] = build_5m(p, cfg)
    tick_bars = {}
    if tick_n is not None:
        for code in prepared:
            f = tick_dir / f"{code}.csv" if tick_dir else None
            tick_bars[code] = build_tick_bars(iter_tick_file(f), code, tick_n).bars if f and f.exists() else []

    entry_states = {c: EntryState() for c in prepared}
    pos: dict[str, V2Position] = {}
    pos_meta: dict[str, dict] = {}
    pending: dict[str, object] = {}
    risk = DayRisk(equity=cfg.account_equity_krw)
    trades: list[Trade] = []
    diag = {"signals": 0, "fills": 0, "skip_gap": 0, "skip_risk": 0, "skip_size": 0, "skip_tick": 0, "blocked_breadth_bars": 0}
    arrays = {c: (list(p.columns), p.to_numpy(dtype=float)) for c, p in prepared.items()}

    def row_at(code: str, i: int) -> dict:
        cols, arr = arrays[code]
        return dict(zip(cols, arr[i]))

    timeline = sorted(set().union(*[set(p.index) for p in prepared.values()])) if prepared else []
    pos_idx = {c: {ts: i for i, ts in enumerate(p.index)} for c, p in prepared.items()}

    def close_trade(code: str, ts: pd.Timestamp) -> None:
        p = pos.pop(code)
        m = pos_meta.pop(code)
        t = m["trade"]
        t.exit_ts = str(ts)
        t.exit_reasons = "+".join(m["reasons"])
        t.gross_pnl = m["gross"]
        t.costs = m["costs"]
        t.net_pnl = m["gross"] - m["costs"]
        t.r_net = t.net_pnl / (p.initial_risk * p.qty) if p.initial_risk > 0 and p.qty > 0 else 0.0
        t.hold_min = (ts - p.entry_ts).total_seconds() / 60.0
        trades.append(t)  # risk.realized_pnl was already updated leg by leg

    def book_exit(code: str, p: V2Position, px: float, qty: int, reason: str) -> None:
        meta = pos_meta[code]
        sell_cost = fill_cost_krw("sell", px, qty, cfg)
        meta["gross"] += (px - p.entry_price) * qty
        meta["costs"] += sell_cost
        meta["reasons"].append(reason)
        risk.realized_pnl += (px - p.entry_price) * qty - sell_cost  # every leg hits the day risk immediately

    last_idx: dict[str, int] = {}
    for ts in timeline:
        # (1) OPEN of bar ts: fill signals from the previous completed bar. Only information available at the
        #     open is used: realized P&L so far and open positions marked at their PREVIOUS bar close.
        unreal_open = sum((row_at(c, last_idx[c])["close"] - q.entry_price) * q.remaining
                          for c, q in pos.items() if c in last_idx)
        for code in list(pending):
            i = pos_idx[code].get(ts)
            if i is None:
                continue
            sig = pending.pop(code)
            if code in pos:
                continue
            if (ts - sig.ts) > pd.Timedelta(minutes=2):
                diag["skip_gap"] += 1
                continue
            extra = cfg.slippage_extra_bps_small if tiers.get(code) == "small" else 0.0
            fill = slippage_price(float(row_at(code, i)["open"]), "buy", cfg.slippage_ticks_entry, extra)
            ok, why = risk.can_open(len(pos), unreal_open, cfg)
            if not ok:
                diag["skip_risk"] += 1
            elif fill <= sig.stop or (fill / sig.support - 1.0) * 100.0 > cfg.m1_trigger_max_above_support_pct + 0.5:
                diag["skip_gap"] += 1  # gapped through stop or ran away before the fill
            else:
                qty = size_position(fill, sig.stop, risk.equity, cfg)
                if qty <= 0:
                    diag["skip_size"] += 1
                else:
                    p = V2Position(code, sig.setup, ts - pd.Timedelta(minutes=1), fill, qty, sig.stop, fill - sig.stop)
                    buy_cost = fill_cost_krw("buy", fill, qty, cfg)
                    pos[code] = p
                    pos_meta[code] = {
                        "gross": 0.0, "costs": buy_cost, "reasons": [],
                        "trade": Trade(day, code, sig.setup, tiers.get(code, ""), variant, str(p.entry_ts), fill, qty,
                                       p.initial_risk, entry_hour=p.entry_ts.hour),
                    }
                    risk.realized_pnl -= buy_cost
                    diag["fills"] += 1
        # (2) bar ts becomes complete: advance marks, then intrabar / close-based exits
        for code in prepared:
            i = pos_idx[code].get(ts)
            if i is not None:
                last_idx[code] = i
        for code in list(pos):
            i = pos_idx[code].get(ts)
            if i is None:
                continue
            p = pos[code]
            for a in step_exit(p, row_at(code, i), ts, completed_5m(m5s[code], ts), cfg):
                if a.reason == "TP1":
                    px = a.price
                else:
                    extra = cfg.slippage_extra_bps_small if tiers.get(code) == "small" else 0.0
                    px = slippage_price(a.price, "sell", cfg.slippage_ticks_exit, extra)
                book_exit(code, p, px, a.qty, a.reason)
            if p.remaining <= 0:
                close_trade(code, ts)
        # (2) entry signals on this completed bar (filled next bar)
        above = total = 0
        for code in prepared:
            j = last_idx.get(code)
            if j is not None:
                row = row_at(code, j)
                total += 1
                above += int(float(row["close"]) > float(row["VWAP"]))
        mkt_ok, _breadth = market_breadth_ok(above, total, cfg)
        for code in prepared:
            i = pos_idx[code].get(ts)
            if i is None:
                continue
            n_before = entry_states[code].trades_taken
            gate = None
            if tick_n is not None:
                tb = tick_bars.get(code, [])

                def gate(_sig, _tb=tb, _now=ts.to_pydatetime()):
                    tm = tick_momentum(_tb, _now, CFGMOD.V2_TICK_LOOKBACK_BARS, CFGMOD.V2_TICK_MIN_UP_BARS,
                                       CFGMOD.V2_TICK_MIN_BUY_VOLUME_RATIO, CFGMOD.V2_TICK_MAX_GAP_RATIO,
                                       CFGMOD.V2_TICK_STALE_SECONDS)
                    if not tm.ok:
                        diag["skip_tick"] += 1
                    return tm.ok, tm.reason
            # The tick filter is applied BEFORE the signal is committed, so a rejection keeps the setup armed.
            sig = step_entry(code, entry_states[code], prepared[code].iloc[: i + 1], m5s[code], cfg,
                             code in pos or code in pending, market_ok=mkt_ok, signal_gate=gate)
            if sig is None:
                if not mkt_ok and entry_states[code].trades_taken == n_before:
                    diag["blocked_breadth_bars"] += 1
                continue
            diag["signals"] += 1
            pending[code] = sig
    # safety: anything still open at the end of data is closed at its last close
    for code in list(pos):
        p = pos[code]
        last = row_at(code, len(prepared[code]) - 1)
        px = slippage_price(float(last["close"]), "sell", cfg.slippage_ticks_exit)
        book_exit(code, p, px, p.remaining, "DATA_END")
        p.remaining = 0
        close_trade(code, prepared[code].index[-1])
    return trades, diag


def run_day(task) -> dict:
    day, cfgs, variants, panel_path = task
    with open(panel_path, "rb") as f:
        panel = pickle.load(f)
    day_dir = RUNTIME_DATA_DIR / day
    files10 = _code_files(day_dir, "10s.txt")
    di = int(day)
    base_cfg = cfgs[0][1]
    trends = []
    for code in files10:
        feat = panel.get(code)
        if feat is None:
            continue
        hist = feat[feat.index < di]
        if hist.empty:
            continue
        trends.append(judge_daily_row(code, hist.iloc[-1], base_cfg))
    universe = select_universe(trends, base_cfg)
    bars = {}
    for u in universe:
        b = load_1m_bars(files10[u.code], day)
        if b is not None and len(b) >= 30:
            bars[u.code] = b
    tick_dir = RUNTIME_DATA_DIR / "ticks" / day
    has_ticks = tick_dir.is_dir() and any(tick_dir.glob("*.csv"))
    out = {"day": day, "universe": len(universe), "with_bars": len(bars), "has_ticks": has_ticks,
           "tiers": {t: sum(1 for u in universe if u.tier == t) for t in ("large", "mid", "small")}, "results": {}}
    # B/C comparison set: symbols of the day's universe that actually have a tick file. A_T/B/C all run on
    # exactly this set (breadth included), so the tick filter is the only difference.
    tick_codes = {c for c in bars if has_ticks and (tick_dir / f"{c}.csv").exists()}
    bars_t = {c: b for c, b in bars.items() if c in tick_codes}
    out["tick_codes"] = sorted(tick_codes)
    universe_t = [u for u in universe if u.code in tick_codes]
    for name, cfg in cfgs:
        for v in variants:
            if v == "A":
                tr, dg = simulate_day(day, universe, bars, cfg, "A", None)
                out["results"][f"{name}|A"] = {"trades": [asdict(t) for t in tr], "diag": dg}
        if bars_t and any(v in ("B", "C") for v in variants):
            for v in ["A_T"] + [v for v in variants if v in ("B", "C")]:
                tr, dg = simulate_day(day, universe_t, bars_t, cfg, v, tick_dir)
                out["results"][f"{name}|{v}"] = {"trades": [asdict(t) for t in tr], "diag": dg}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(trades: pd.DataFrame, days: list[str], equity: float) -> dict:
    n = len(trades)
    daily = trades.groupby("date")["net_pnl"].sum().reindex(days, fill_value=0.0) if n else pd.Series(0.0, index=days)
    # Equity curve of closed-trade P&L by day, starting from 0 so a loss on day 1 counts as drawdown.
    # (Intraday mark-to-market drawdown is not included: day-trades are flat at the close.)
    curve = pd.concat([pd.Series([0.0]), daily.cumsum().reset_index(drop=True)])
    mdd = float((curve - curve.cummax()).min())
    if n == 0:
        return {"trades": 0, "days": len(days), "net_pnl": 0.0, "mdd_krw": mdd}
    wins = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] <= 0]["net_pnl"]
    gp, gl = float(wins.sum()), float(-losses.sum())
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(-losses.mean()) if len(losses) else 0.0
    return {
        "trades": n,
        "days": len(days),
        "trades_per_day": round(n / max(len(days), 1), 2),
        "win_rate_pct": round(len(wins) / n * 100.0, 1),
        "payoff_ratio": round(avg_w / avg_l, 2) if avg_l > 0 else float("inf"),
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "expectancy_krw": round(float(trades["net_pnl"].mean()), 0),
        "expectancy_r": round(float(trades["r_net"].mean()), 3),
        "net_pnl": round(float(trades["net_pnl"].sum()), 0),
        "net_pnl_pct_equity": round(float(trades["net_pnl"].sum()) / equity * 100.0, 2),
        "costs_krw": round(float(trades["costs"].sum()), 0),
        "mdd_krw": round(mdd, 0),
        "mdd_pct_equity": round(mdd / equity * 100.0, 2),
        "worst_day_krw": round(float(daily.min()), 0),
        "positive_days_pct": round(float((daily > 0).mean() * 100.0), 1),
        "avg_hold_min": round(float(trades["hold_min"].mean()), 1),
    }


def breakdown(trades: pd.DataFrame, key: str, days: list[str], equity: float) -> dict:
    if trades.empty:
        return {}
    return {str(k): metrics(g, days, equity) for k, g in trades.groupby(key)}


# ---------------------------------------------------------------------------
# Grid + walk-forward
# ---------------------------------------------------------------------------

def build_grid(base: V2Config, use_grid: bool) -> list[tuple[str, V2Config]]:
    if not use_grid:
        return [("base", base)]
    out = []
    # Kept deliberately small (Raspberry Pi shares the CPU with the live bot): 2 x 2 around the defaults.
    for vol, trail in itertools.product((base.m1_trigger_vol_ratio, 1.8), (base.trail_atr_mult, 2.0)):
        out.append((f"vol{vol}_trail{trail}", replace(base, m1_trigger_vol_ratio=vol, trail_atr_mult=trail)))
    out[0] = ("base", out[0][1])  # first combo == defaults
    return out


def _select_score(m: dict, min_trades: int) -> float:
    if m.get("trades", 0) < min_trades:
        return -1e18
    pf = m.get("profit_factor", 0.0)
    pf = 5.0 if pf == float("inf") else pf
    return pf * 1000.0 + m.get("expectancy_r", 0.0)


def walk_forward(all_trades: pd.DataFrame, days: list[str], names: list[str], train: int, test: int,
                 equity: float, min_trades: int) -> dict:
    windows = []
    oos_parts = []
    i = train
    while i < len(days):
        tr_days = days[i - train:i]
        te_days = days[i:i + test]
        best, best_score = None, -1e19
        for n in names:
            sub = all_trades[(all_trades["cfg"] == n) & (all_trades["date"].isin(tr_days))]
            s = _select_score(metrics(sub, tr_days, equity), min_trades)
            if s > best_score:
                best, best_score = n, s
        te = all_trades[(all_trades["cfg"] == best) & (all_trades["date"].isin(te_days))]
        oos_parts.append(te)
        windows.append({"train": f"{tr_days[0]}~{tr_days[-1]}", "test": f"{te_days[0]}~{te_days[-1]}", "chosen": best,
                        "oos": metrics(te, te_days, equity)})
        i += test
    oos_days = days[train:]
    oos = pd.concat(oos_parts) if oos_parts else all_trades.iloc[0:0]
    return {"windows": windows, "oos_total": metrics(oos, oos_days, equity)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def apply_overrides(cfg: V2Config, items: list[str]) -> V2Config:
    fields = {f: type(v) for f, v in asdict(cfg).items()}
    upd = {}
    for it in items:
        k, _, v = it.partition("=")
        if k not in fields:
            raise SystemExit(f"unknown V2Config field: {k}")
        t = fields[k]
        if t is bool:
            upd[k] = v.strip().lower() in ("1", "true", "yes", "on")
        elif t in (int, float):
            upd[k] = t(float(v)) if t is int else float(v)
        else:
            raise SystemExit(f"field {k} ({t.__name__}) cannot be overridden from the CLI")
    return replace(cfg, **upd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--strategies", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--walk-forward", action="store_true", help="run a small parameter grid + rolling walk-forward")
    ap.add_argument("--train-days", type=int, default=20)
    ap.add_argument("--test-days", type=int, default=5)
    ap.add_argument("--split", type=float, default=0.7, help="simple in-sample fraction for the train/test report")
    ap.add_argument("--min-trades", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--panel-date", default=None, help="folder to read daily history from (default: latest date)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="override V2Config fields for an experiment without editing r001 (e.g. time_stop_min=0)")
    args = ap.parse_args()

    base = apply_overrides(V2Config.from_module(CFGMOD), args.set)
    days = trading_dates(args.start, args.end)
    if not days:
        print("no trading dates with data")
        return
    all_dirs = trading_dates("00000000", "99999999")
    panel_date = args.panel_date or all_dirs[-1]
    t0 = time.time()
    print(f"[g012] days={len(days)} ({days[0]}~{days[-1]}) panel={panel_date} strategies={args.strategies}", flush=True)
    load_daily_panel(panel_date, base)
    panel_path = CACHE_DIR / f"daily_feat_{panel_date}_{base.daily_ma_fast}_{base.daily_ma_slow}_{base.daily_ma_fast_slope_days}_{base.daily_ma_slow_slope_days}.pkl"
    cfgs = build_grid(base, args.walk_forward)
    tasks = [(d, cfgs, args.strategies, panel_path) for d in days]
    results = []
    with Pool(max(1, args.workers)) as pool:
        for k, r in enumerate(pool.imap(run_day, tasks), 1):
            results.append(r)
            print(f"  [{k}/{len(days)}] {r['day']} universe={r['universe']} bars={r['with_bars']} ticks={r['has_ticks']} "
                  f"tiers={r['tiers']} elapsed={time.time() - t0:.0f}s", flush=True)

    rows = []
    diags: dict[str, dict] = {}
    for r in results:
        for key, res in r["results"].items():
            cfg_name, v = key.split("|")
            for t in res["trades"]:
                rows.append({**t, "cfg": cfg_name})
            d = diags.setdefault(key, {})
            for k2, v2 in res["diag"].items():
                d[k2] = d.get(k2, 0) + v2
    trades = pd.DataFrame(rows)
    equity = base.account_equity_krw
    tick_days = [r["day"] for r in results if r.get("tick_codes")]
    report: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "days": days, "panel_date": panel_date,
                    "overrides": args.set,
                    "config": {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in asdict(base).items()},
                    "universe": {r["day"]: {"n": r["universe"], "tiers": r["tiers"]} for r in results},
                    "diagnostics": diags, "tick_days": tick_days}

    if trades.empty:
        report["summary"] = "no trades"
    else:
        base_tr = trades[(trades["cfg"] == "base")]
        a = base_tr[base_tr["variant"] == "A"]
        report["A_full"] = metrics(a, days, equity)
        report["A_by_tier"] = breakdown(a, "tier", days, equity)
        report["A_by_setup"] = breakdown(a, "setup", days, equity)
        report["A_by_hour"] = breakdown(a, "entry_hour", days, equity)
        a2 = a.assign(month=a["date"].str[:6])
        report["A_by_month"] = breakdown(a2, "month", days, equity)
        a3 = a.assign(first_exit=a["exit_reasons"].str.split("+").str[-1])
        report["A_by_exit"] = breakdown(a3, "first_exit", days, equity)
        cut = max(1, int(len(days) * args.split))
        report["A_train"] = metrics(a[a["date"].isin(days[:cut])], days[:cut], equity)
        report["A_test"] = metrics(a[a["date"].isin(days[cut:])], days[cut:], equity)
        if tick_days:
            same = {}
            for v in ("A_T", "B", "C"):
                sub = base_tr[(base_tr["variant"] == v) & (base_tr["date"].isin(tick_days))]
                same[v] = metrics(sub, tick_days, equity)
                same[v + "_by_tier"] = breakdown(sub, "tier", tick_days, equity)
            report["ABC_same_days"] = same
        else:
            report["ABC_same_days"] = "B/C not evaluated: no tick files under runtime/data/ticks/<date>/ (start g011 collector)"
        if args.walk_forward:
            names = [n for n, _c in cfgs]
            report["grid_full_period_in_sample"] = {
                n: metrics(trades[(trades["cfg"] == n) & (trades["variant"] == "A")], days, equity) for n in names
            }
            report["walk_forward_A"] = walk_forward(trades[trades["variant"] == "A"], days, names, args.train_days,
                                                    args.test_days, equity, args.min_trades)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{days[0]}_{days[-1]}{'_wf' if args.walk_forward else ''}"
    (OUT_DIR / f"report_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if not trades.empty:
        trades.to_csv(OUT_DIR / f"trades_{tag}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: report[k] for k in ("A_full", "A_train", "A_test", "A_by_tier", "A_by_setup") if k in report},
                     ensure_ascii=False, indent=1, default=str))
    if "walk_forward_A" in report:
        print(json.dumps(report["walk_forward_A"]["oos_total"], ensure_ascii=False, indent=1))
    print(f"[g012] done in {time.time() - t0:.0f}s -> {OUT_DIR / f'report_{tag}.json'}")


if __name__ == "__main__":
    main()
