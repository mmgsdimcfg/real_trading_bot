# -*- coding: utf-8 -*-
"""g057 - VWAP trade simulation (read-only: no orders, no bot state, nothing written under runtime/).

Question (user 2026-11-28): what would the VWAP trade with REAL orders have done on the October / November data, and which of
its conditions decide the result?

Two data sets, two fidelities - they are never mixed in one number:
  tick  the days the tick collector (g011) ran (ticks/<day>/<code>.csv, from 2026-10-30): the signal exactly as r003 judges it -
        r029.vwap_signal on the day's 1-minute bars built from the prints + the 60-tick MA5/MA15 cross of r013 (TickReplayFeed,
        the live feed's own rule and abstain reasons) - evaluated once per completed 1-minute bar, EVAL_DELAY_S after its end.
        Entry = a limit order at the ask of that moment, filled only when a print at or under it follows within FILL_WAIT_S
        (r001 VWAP_LIVE_UNFILLED_CANCEL_SECONDS). Exits are looked at every POLL_S seconds on the last print (r003 looks at a
        holding once per loop turn, not on every print): take profit / stop loss / holding time / 15:20, sold at the bid.
  bars  any collected day (<day>/<code>_<name>_10s.txt): the same conditions WITHOUT the tick trigger (no tick data), on the
        day's watchlist (the picks of the previous scan). Entry = last price + 1 tick, exits on the 10-second closes - 1 tick.
        This says what conditions 1 / 2 / 4 alone do; it is NOT the live strategy.

Not simulated (each would only remove trades): the rule engine's shared safety conditions the live path also runs
(VWAP_BUY_CONDITIONS of r005 - market regime, cool-down, order book ...), mixed holdings, order holds, API failures.
Costs: commission COMMISSION_PCT per side + SELL_TAX_PCT on the sale (r001 V2_COMMISSION_PCT / V2_SELL_TAX_PCT).

usage: g057_vwap_trade_sim.py --out <dir> [--from YYYYMMDD] [--to YYYYMMDD] [--mode tick|bars|both] [--workers 4]
                              [--order-krw 500000] [--max-positions 5] [--max-entries 20] [--daily-loss 50000]
                              [--tp 3.0] [--sl 1.27] [--horizon 300]

Update log:
- [2026-11-28] type=feat owner=claude
    summary: 신규 - VWAP 매매 실전 조건 시뮬레이션(틱 데이터가 있는 날: 실제 신호 그대로 / 그 밖의 날: 틱 조건 없는 참고 분석),
      조건별 단계 분포, 청산 기준 격자, 시간대 / 이격 구간별 결과. 읽기 전용.
    impact: sim
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, time as dt_time, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))
sys.path.insert(0, str(HERE))
import r001_define_config as cfg  # noqa: E402
import r029_vwap_trade as r029  # noqa: E402
from r008_strategy_v2 import krx_tick_size  # noqa: E402
from r011_edge_ai import is_fund_product  # noqa: E402
from r013_tick_trigger import TickReplayFeed  # noqa: E402

EVAL_DELAY_S = 3.0          # a completed bar is judged this long after its end (confirm delay 2 s + the loop)
FILL_WAIT_S = float(getattr(cfg, "VWAP_LIVE_UNFILLED_CANCEL_SECONDS", 20))
POLL_S = 20.0               # a holding is looked at this often
ENTRY_START, ENTRY_END = dt_time(9, 5), dt_time(14, 50)
SESSION_START, FORCE_EXIT = dt_time(9, 0), dt_time(15, 20)
COMMISSION_PCT = float(getattr(cfg, "V2_COMMISSION_PCT", 0.015))
SELL_TAX_PCT = float(getattr(cfg, "V2_SELL_TAX_PCT", 0.20))
PARAMS = r029.VwapParams.from_config(cfg)


def net_pct(entry: float, exit_px: float) -> float:
    gross = exit_px / entry - 1.0
    return (gross - (COMMISSION_PCT * 2 + SELL_TAX_PCT) / 100.0 - gross * (COMMISSION_PCT + SELL_TAX_PCT) / 100.0) * 100.0


def net_krw(entry: float, exit_px: float, qty: int) -> float:
    return qty * exit_px - qty * entry - (exit_px * qty * (COMMISSION_PCT + SELL_TAX_PCT) + entry * qty * COMMISSION_PCT) / 100.0


# ---- data -----------------------------------------------------------------------------------------------------------------------
def load_ticks(path: Path, day: str) -> pd.DataFrame | None:
    """regular-session prints of a g011 file: recv (receive time), t (exchange time), price, vol, ask, bid - in receive order,
    reconnect replays dropped (acml_vol only grows)"""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ("recv_ts", "trade_time", "price", "cntg_vol", "acml_vol", "askp1", "bidp1", "session"),
                         dtype=str)
    except Exception:
        return None
    if df.empty or "recv_ts" not in df:
        return None
    if "session" in df:
        df = df[df["session"].fillna("regular") == "regular"]
    df["recv"] = pd.to_datetime(df["recv_ts"], errors="coerce")
    tt = df["trade_time"].fillna("").str.zfill(6)
    df["t"] = pd.to_datetime(day + tt, format="%Y%m%d%H%M%S", errors="coerce")
    for c in ("price", "cntg_vol", "acml_vol", "askp1", "bidp1"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["recv", "t", "price"])
    df = df[(df["price"] > 0) & (df["t"].dt.time >= SESSION_START) & (df["t"].dt.time < FORCE_EXIT)]
    if df.empty:
        return None
    df = df.sort_values("recv", kind="stable")
    acml = df["acml_vol"].ffill()
    df = df[(acml > acml.cummax().shift(fill_value=-1)) | acml.isna()]
    return df[["recv", "t", "price", "cntg_vol", "askp1", "bidp1"]].rename(columns={"cntg_vol": "vol", "askp1": "ask", "bidp1": "bid"}).reset_index(drop=True)


def minute_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """end-labelled 1-minute bars (09:00:xx prints -> the 09:01 bar), like r003's frame"""
    end = ticks["t"].dt.floor("min") + pd.Timedelta(minutes=1)
    g = ticks.groupby(end)
    return pd.DataFrame({"close": g["price"].last(), "volume": g["vol"].sum()})


def load_10s(path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(1-minute end-labelled bars, 10-second closes) of a g001 file"""
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in ("datetime", "close", "volume", "raw_bar"))
    except Exception:
        return None
    if raw.empty:
        return None
    raw.index = pd.to_datetime(raw.pop("datetime"), errors="coerce")
    raw = raw[raw.index.notna()]
    raw = raw[(raw.index.time >= SESSION_START) & (raw.index.time <= FORCE_EXIT)]
    for c in ("close", "volume", "raw_bar"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[raw["close"] > 0]
    if raw.empty:
        return None
    # a row at hh:mm:00 with raw_bar == 1 is the 1-minute bar that STARTS there: its close is known one minute later
    m = raw[(raw.index.second == 0) & (raw["raw_bar"] == 1)][["close", "volume"]].copy()
    m.index = m.index + pd.Timedelta(minutes=1)
    return m, raw[["close"]]


# ---- per symbol: every bar's stage + the entry candidates -------------------------------------------------------------------------
def symbol_tick(args) -> dict | None:
    day, code, name, ticks_root, ten_path = args
    ticks = load_ticks(Path(ticks_root) / day / f"{code}.csv", day)
    if ticks is None or len(ticks) < 200:
        return None
    bars = minute_bars(ticks)
    bar_src = "ticks"
    # the collector started after the open on some days: the day's VWAP cannot be made from its prints. r003 itself reads the
    # 1-minute bars from the broker (the whole session), so the collected 10-second file stands in for them on such a day
    if not len(bars) or bars.index[0].time() > r029.FULL_SESSION_FIRST_BAR:
        got = load_10s(Path(ten_path)) if ten_path else None
        if got is not None and len(got[0]) and got[0].index[0].time() <= r029.FULL_SESSION_FIRST_BAR:
            bars, bar_src = got[0], "10s file"
    feed = TickReplayFeed(
        Path(ticks_root), (int(getattr(cfg, "VWAP_TRADE_TICK_BAR_SIZE", 60)),), int(getattr(cfg, "VWAP_TRADE_TICK_MA_FAST", 5)),
        int(getattr(cfg, "VWAP_TRADE_TICK_MA_SLOW", 15)), float(getattr(cfg, "VWAP_TRADE_TICK_CROSS_MAX_AGE_SECONDS", 60.0)),
        float(getattr(cfg, "VWAP_TRADE_TICK_FEED_STALE_SECONDS", 15.0)),
        prior_day_warmup=bool(getattr(cfg, "TICK_TRIGGER_PRIOR_DAY_WARMUP", True)),
        min_prints_per_min=float(getattr(cfg, "VWAP_TRADE_MIN_PRINTS_PER_MIN", 30.0)),
        density_window_min=float(getattr(cfg, "VWAP_TRADE_MIN_PRINTS_WINDOW_MINUTES", 3)),
        warmup_extra_bars=int(getattr(cfg, "TICK_CROSS_WARMUP_EXTRA_BARS", 0) or 0))
    recv = ticks["recv"].values
    t_end = (ticks["t"].dt.floor("min") + pd.Timedelta(minutes=1)).values
    px_all = ticks["price"].values
    vol_all = ticks["vol"].fillna(0).values
    stages: Counter = Counter()
    cands: list[dict] = []
    cands_nt: list[dict] = []          # the same bars without the tick trigger (what the tick condition adds / removes)
    for i, end in enumerate(bars.index):
        if not (ENTRY_START <= end.time() <= ENTRY_END):
            continue
        now = end + pd.Timedelta(seconds=EVAL_DELAY_S)
        k = int(np.searchsorted(recv, np.datetime64(now), side="right")) - 1
        if k < 0:
            continue
        price = float(ticks["price"].iat[k])
        frame = bars.iloc[: i + 1]
        if bar_src == "ticks":
            # the last bar as it was known at `now`: only prints received by then
            m = (t_end[: k + 1] == np.datetime64(end))
            if m.any():
                frame = frame.copy()
                frame.iloc[-1, frame.columns.get_loc("close")] = float(px_all[: k + 1][m][-1])
                frame.iloc[-1, frame.columns.get_loc("volume")] = float(vol_all[: k + 1][m].sum())
        tick = feed.evaluate(code, now.to_pydatetime())
        ok, stage, info = r029.vwap_signal(frame, price, now.to_pydatetime(), tick, PARAMS)
        ok_nt, _st, info_nt = r029.vwap_signal(frame, price, now.to_pydatetime(), _TickYes, PARAMS)
        ask_k = ticks["ask"].iat[k]
        bid_k = ticks["bid"].iat[k]
        ask0 = float(ask_k) if ask_k == ask_k and ask_k > 0 else price + krx_tick_size(price)
        spread = (ask0 - float(bid_k)) / price * 100 if bid_k == bid_k and bid_k > 0 else None
        if ok_nt:
            cands_nt.append({"now": now, "price": price, "limit": ask0, "k": k, "gap": info_nt.get("vwap_gap_pct"), "ma5_up": info_nt.get("ma5_up"),
                             "mid_up": info_nt.get("bb_mid_up"), "cross_age": None, "spread": spread, "tick_pct": krx_tick_size(price) / price * 100})
        if stage == "no_tick_cross":
            r = str(info.get("tick", ""))
            stage = "tick_abstain" if "ABSTAIN" in r or "DENSITY" in r or "LOW_" in r else "no_tick_cross"
        stages[stage] += 1
        if ok:
            cands.append({"now": now, "price": price, "limit": ask0, "k": k, "gap": info.get("vwap_gap_pct"), "ma5_up": info.get("ma5_up"),
                          "mid_up": info.get("bb_mid_up"), "cross_age": (now - pd.Timestamp(info.get("tick_cross"))).total_seconds()
                          if info.get("tick_cross") not in (None, "None") else None, "spread": spread, "tick_pct": krx_tick_size(price) / price * 100})
    return {"day": day, "code": code, "name": name, "stages": dict(stages), "cands": cands, "cands_nt": cands_nt, "mode": "tick",
            "bar_src": bar_src, "px": ticks[["recv", "price", "bid"]]}


def symbol_bars(args) -> dict | None:
    day, code, name, path = args
    got = load_10s(Path(path))
    if got is None:
        return None
    bars, tens = got
    if len(bars) < 30:
        return None
    t10 = tens.index.values
    stages: Counter = Counter()
    cands: list[dict] = []
    for i, end in enumerate(bars.index):
        if not (ENTRY_START <= end.time() <= ENTRY_END):
            continue
        now = end + pd.Timedelta(seconds=EVAL_DELAY_S)
        k = int(np.searchsorted(t10, np.datetime64(now), side="right")) - 1
        if k < 0:
            continue
        price = float(tens["close"].iat[k])
        ok, stage, info = r029.vwap_signal(bars.iloc[: i + 1], price, now.to_pydatetime(), _TickYes, PARAMS)
        stages[stage] += 1
        if ok:
            cands.append({"now": now, "price": price, "limit": price + krx_tick_size(price), "k": k, "gap": info.get("vwap_gap_pct"),
                          "ma5_up": info.get("ma5_up"), "mid_up": info.get("bb_mid_up"), "cross_age": None})
    px = pd.DataFrame({"recv": tens.index, "price": tens["close"].values, "bid": np.nan})
    return {"day": day, "code": code, "name": name, "stages": dict(stages), "cands": cands, "mode": "bars", "px": px}


class _TickYes:
    """stands in for the tick trigger in `bars` mode: always there, no cross price (so only the VWAP-gap chase limit applies)"""
    ok, label, cross_time, reason = True, "NO_TICK_DATA", None, ""
    crosses: dict = {}


# ---- one position ---------------------------------------------------------------------------------------------------------------
def fill_and_exit(sym: dict, cand: dict, tp: float, sl: float, horizon: float) -> dict | None:
    """the trade of one candidate: None when the limit order would not have filled"""
    px: pd.DataFrame = sym["px"]
    recv = px["recv"].values
    price = px["price"].values
    bid = px["bid"].values
    t0 = np.datetime64(cand["now"])
    limit = float(cand["limit"])
    i0 = int(np.searchsorted(recv, t0, side="right"))
    if sym["mode"] == "tick":
        i1 = int(np.searchsorted(recv, t0 + np.timedelta64(int(FILL_WAIT_S * 1000), "ms"), side="right"))
        hit = np.nonzero(price[i0:i1] <= limit)[0]
        if not len(hit):
            return None
        i_fill = i0 + int(hit[0])
        t_fill = recv[i_fill]
    else:
        i_fill, t_fill = max(i0 - 1, 0), t0
    entry = limit
    def _mae(minutes: float) -> float | None:
        j = int(np.searchsorted(recv, t_fill + np.timedelta64(int(minutes * 60_000), "ms"), side="right"))
        seg = price[i_fill:j]
        return float((seg.min() / entry - 1.0) * 100.0) if len(seg) else None
    mae1, mae5 = _mae(1), _mae(5)
    day0 = pd.Timestamp(t_fill).normalize()
    force = np.datetime64(day0 + pd.Timedelta(hours=FORCE_EXIT.hour, minutes=FORCE_EXIT.minute))
    t = t_fill + np.timedelta64(int(POLL_S * 1000), "ms")
    reason, k = "close", len(recv) - 1
    while True:
        if t >= force:
            k = max(int(np.searchsorted(recv, force, side="left")) - 1, i_fill)
            reason = "close"
            break
        k = int(np.searchsorted(recv, t, side="right")) - 1
        if k > i_fill:
            p = (price[k] / entry - 1.0) * 100.0
            if p >= tp:
                reason = "tp"
                break
            if p <= -sl:
                reason = "sl"
                break
        if (t - t_fill) / np.timedelta64(1, "m") >= horizon:
            reason = "time"
            break
        t = t + np.timedelta64(int(POLL_S * 1000), "ms")
    k = max(k, i_fill)
    last = float(price[k])
    b = float(bid[k]) if bid[k] == bid[k] and bid[k] > 0 else last - krx_tick_size(last)
    exit_px = min(b, last)
    t_exit = min(max(t, t_fill), force)
    return {"entry": entry, "exit": exit_px, "t_in": pd.Timestamp(t_fill), "t_out": pd.Timestamp(t_exit), "reason": reason,
            "net_pct": net_pct(entry, exit_px), "held_min": float((t_exit - t_fill) / np.timedelta64(1, "m")), "mae1": mae1, "mae5": mae5}


# ---- the day's book ---------------------------------------------------------------------------------------------------------------
def run_day(syms: list[dict], o: dict) -> list[dict]:
    """every candidate of the day in time order under the day's limits (positions, entries, daily loss, no re-entry after a stop)"""
    events = sorted(((c["now"], s["code"], j) for s in syms for j, c in enumerate(s["cands"])), key=lambda x: (x[0], x[1]))
    by = {s["code"]: s for s in syms}
    open_until: dict[str, pd.Timestamp] = {}
    pending: list[tuple[pd.Timestamp, float, str, str]] = []   # (exit time, krw, code, reason)
    realized, entries = 0.0, 0
    stopped: set[str] = set()
    out: list[dict] = []
    for now, code, j in events:
        for e in sorted(p for p in pending if p[0] <= now):
            realized += e[1]
            if e[3] == "sl":
                stopped.add(e[2])
            pending.remove(e)
        for c in [c for c, t in open_until.items() if t <= now]:
            open_until.pop(c)
        s = by[code]
        if code in open_until or code in stopped or is_fund_product(s["name"]):
            continue
        if entries >= o["max_entries"] or len(open_until) >= o["max_positions"]:
            continue
        if o["daily_loss"] > 0 and realized <= -o["daily_loss"]:
            continue
        cand = s["cands"][j]
        qty = int(o["order_krw"] // cand["limit"])
        if qty <= 0:
            continue
        tr = fill_and_exit(s, cand, o["tp"], o["sl"], o["horizon"])
        if tr is None:
            continue
        entries += 1
        krw = net_krw(tr["entry"], tr["exit"], qty)
        open_until[code] = tr["t_out"]
        pending.append((tr["t_out"], krw, code, tr["reason"]))
        out.append({"day": s["day"], "code": code, "name": s["name"], "qty": qty, "krw": krw, "gap": cand["gap"], "cross_age": cand["cross_age"],
                    "spread": cand.get("spread"), "tick_pct": cand.get("tick_pct"), "bar_src": s.get("bar_src", ""),
                    "ma5_up": cand["ma5_up"], "mid_up": cand["mid_up"], "hour": pd.Timestamp(now).hour, **tr})
    return out


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame(trades)
    days = df.groupby("day")["krw"].sum()
    return {
        "n": int(len(df)), "days": int(df["day"].nunique()), "win_rate": float((df["net_pct"] > 0).mean()), "avg_net_pct": float(df["net_pct"].mean()),
        "median_net_pct": float(df["net_pct"].median()), "total_krw": float(df["krw"].sum()),
        "day_mean_krw": float(days.mean()), "day_min_krw": float(days.min()), "day_max_krw": float(days.max()),
        "losing_days": int((days < 0).sum()), "avg_held_min": float(df["held_min"].mean()),
        "by_reason": {k: {"n": int(len(g)), "avg_net_pct": float(g["net_pct"].mean()), "krw": float(g["krw"].sum())} for k, g in df.groupby("reason")},
    }


def bucket(trades: list[dict], key: str, bins: list[float], labels: list[str]) -> list[dict]:
    if not trades:
        return []
    df = pd.DataFrame(trades)
    df = df[df[key].notna()]
    if df.empty:
        return []
    df["b"] = pd.cut(df[key].astype(float), bins=bins, labels=labels, include_lowest=True)
    return [{"bucket": str(b), "n": int(len(g)), "win_rate": float((g["net_pct"] > 0).mean()), "avg_net_pct": float(g["net_pct"].mean()),
             "krw": float(g["krw"].sum())} for b, g in df.groupby("b", observed=True)]


# the same candidates under other exits (the entries do not depend on them, the day's limits do)
GRID = {f"g|{tp}|{sl}|{hz}": (tp, sl, hz) for tp in (2.0, 3.0, 5.0) for sl in (1.0, 1.27, 2.0, 3.0) for hz in (60.0, 300.0)}
GRID_KEYS = list(GRID)


def day_job(job) -> dict:
    """one day in one process: the symbols' candidates, then the day's book under the base settings, without the day's limits
    ("free": every signal that would have filled - what the signal itself is worth) and under the exit grid. Only trades and
    counts go back to the parent (the price series stay here)."""
    mode, day, syms, base = job
    fn = symbol_tick if mode == "tick" else symbol_bars
    res = [r for r in map(fn, syms) if r]
    stages: Counter = Counter()
    for r in res:
        stages.update(r["stages"])
    trades = {"base": run_day(res, base),
              "free": run_day(res, {**base, "max_positions": 10**6, "max_entries": 10**6, "daily_loss": 0})}
    for key, (tp, sl, hz) in GRID.items():
        trades[key] = run_day(res, {**base, "tp": tp, "sl": sl, "horizon": hz})
    if mode == "tick":
        nt = [{**r, "cands": r.get("cands_nt", [])} for r in res]
        trades["base_notick"] = run_day(nt, base)
        trades["free_notick"] = run_day(nt, {**base, "max_positions": 10**6, "max_entries": 10**6, "daily_loss": 0})
    return {"day": day, "symbol_days": len(res), "stages": dict(stages), "signals": sum(len(r["cands"]) for r in res), "trades": trades,
            "bar_src": dict(Counter(r.get("bar_src", "10s file") for r in res))}


# ---- driver ---------------------------------------------------------------------------------------------------------------------
def watchlist_of(data_root: Path, day: str, days: list[str]) -> list[tuple[str, str]]:
    """the symbols watched on `day`: the picks of the last scan before it"""
    for d in reversed([x for x in days if x < day]):
        f = data_root / d / f"_{d}_picks.txt"
        if f.is_file():
            rows = []
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                p = [x.strip() for x in line.split(",")]
                if p and p[0].isdigit() and len(p[0]) == 6:
                    rows.append((p[0], p[1] if len(p) > 1 else p[0]))
            return rows
    return []


def watched_on(ticks_root: Path, data_root: Path, day: str, days: list[str]) -> tuple[set[str], str]:
    """the symbols r003 watched on `day` (the VWAP trade buys watched symbols only; the tick collector also records extra /
    scan-fill symbols). From the collector's own "ROLES watch=..." line when the day has one, else the picks of the last scan
    before it. (codes, source)"""
    log = ticks_root / day / "_collector.log"
    if log.is_file():
        roles = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if " ROLES " in ln and "watch=" in ln]
        if roles:
            part = roles[-1].split("watch=", 1)[1].split()[0]
            codes = {c for c in part.split(",") if c.isdigit() and len(c) == 6}
            if codes:
                return codes, "collector"
    return {c for c, _n in watchlist_of(data_root, day, days)}, "picks"


def name_map(data_root: Path, day: str) -> dict[str, str]:
    out: dict[str, str] = {}
    d = data_root / day
    if d.is_dir():
        for p in d.glob("*_10s.txt"):
            parts = p.name.split("_")
            if len(parts) >= 3:
                out[parts[0]] = "_".join(parts[1:-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="d0", default="20261001")
    ap.add_argument("--to", dest="d1", default="20261130")
    ap.add_argument("--mode", choices=["tick", "bars", "both"], default="both")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--order-krw", type=float, default=500_000)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--max-entries", type=int, default=20)
    ap.add_argument("--daily-loss", type=float, default=50_000)
    ap.add_argument("--tp", type=float, default=float(getattr(cfg, "VWAP_TRADE_TP_PCT", 3.0)))
    ap.add_argument("--sl", type=float, default=float(getattr(cfg, "VWAP_TRADE_SL_PCT", 1.27)))
    ap.add_argument("--horizon", type=float, default=float(getattr(cfg, "VWAP_TRADE_HORIZON_MIN", 300)))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg.RUNTIME_ROOT) / "data"
    ticks_root = Path(cfg.TICK_DATA_DIR)
    all_days = sorted(d.name for d in data_root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8)
    days = [d for d in all_days if a.d0 <= d <= a.d1]
    base = {"order_krw": a.order_krw, "max_positions": a.max_positions, "max_entries": a.max_entries, "daily_loss": a.daily_loss,
            "tp": a.tp, "sl": a.sl, "horizon": a.horizon}
    report: dict = {"params": {**base, "eval_delay_s": EVAL_DELAY_S, "fill_wait_s": FILL_WAIT_S, "poll_s": POLL_S,
                               "commission_pct": COMMISSION_PCT, "sell_tax_pct": SELL_TAX_PCT, "signal": PARAMS.__dict__}}

    for mode in (["tick", "bars"] if a.mode == "both" else [a.mode]):
        jobs = []
        sources: dict[str, str] = {}
        for day in days:
            names = name_map(data_root, day)
            syms = []
            if mode == "tick":
                d = ticks_root / day
                if d.is_dir():
                    watched, src = watched_on(ticks_root, data_root, day, all_days)
                    sources[day] = src
                    tens = {q.name.split("_")[0]: str(q) for q in (data_root / day).glob("*_10s.txt")}
                    syms = [(day, p.stem, names.get(p.stem, p.stem), str(ticks_root), tens.get(p.stem, "")) for p in sorted(d.glob("*.csv"))
                            if p.stem.isdigit() and p.stem in watched]
            else:
                for code, name in watchlist_of(data_root, day, all_days):
                    hits = list((data_root / day).glob(f"{code}_*_10s.txt"))
                    if hits:
                        syms.append((day, code, name, str(hits[0])))
            if syms:
                jobs.append((mode, day, syms, base))
        if a.workers > 1 and len(jobs) > 1:
            with Pool(a.workers) as pool:
                res = list(pool.imap_unordered(day_job, jobs))
        else:
            res = [day_job(j) for j in jobs]
        res.sort(key=lambda r: r["day"])
        stages: Counter = Counter()
        for r in res:
            stages.update(r["stages"])
        cat = lambda key: [t for r in res for t in r["trades"].get(key, [])]   # noqa: E731
        trades, free = cat("base"), cat("free")
        grid = []
        for key in GRID_KEYS:
            tp, sl, hz = GRID[key]
            sm = summarize(cat(key))
            grid.append({"tp": tp, "sl": sl, "horizon": hz, **{k: sm.get(k) for k in ("n", "win_rate", "avg_net_pct", "total_krw", "losing_days")}})
        report[mode] = {
            "days": [r["day"] for r in res], "watch_source": sources, "bar_source": {r["day"]: r["bar_src"] for r in res}, "symbol_days": int(sum(r["symbol_days"] for r in res)), "bars_judged": int(sum(stages.values())),
            "stages": dict(stages.most_common()), "signals": int(sum(r["signals"] for r in res)),
            "summary": summarize(trades), "summary_no_limits": summarize(free), "exit_grid": grid,
            "by_hour": bucket(trades, "hour", [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 15], ["09", "10", "11", "12", "13", "14"]),
            "by_gap": bucket(free, "gap", [0, 0.3, 0.6, 1.0, 1.5, 2.01], ["0-0.3", "0.3-0.6", "0.6-1.0", "1.0-1.5", "1.5-2.0"]),
            "by_cross_age": bucket(free, "cross_age", [-1, 15, 30, 45, 61, 1e9], ["0-15s", "15-30s", "30-45s", "45-60s", "60s+"]),
            "by_day": [{"day": d, "n": int(len(g)), "krw": float(g["krw"].sum()), "win_rate": float((g["net_pct"] > 0).mean())}
                       for d, g in (pd.DataFrame(trades).groupby("day") if trades else [])],
        }
        if mode == "tick":
            fdf = pd.DataFrame(free)
            report[mode].update({
                # the same days, symbols and fill / exit model with the tick trigger switched off
                "no_tick_trigger": {"with_limits": summarize(cat("base_notick")), "no_limits": summarize(cat("free_notick"))},
                # days whose bars came from the tick file only (the 10s-file fallback days left out)
                "tick_bar_days_only": summarize([t for t in trades if t.get("bar_src") == "ticks"]),
                "by_tick_pct": bucket(free, "tick_pct", [0, 0.1, 0.2, 0.3, 0.6], ["<0.1%", "0.1-0.2%", "0.2-0.3%", "0.3%+"]),
                "by_spread": bucket(free, "spread", [-1, 0.1, 0.2, 0.3, 5], ["<=0.1%", "0.1-0.2%", "0.2-0.3%", "0.3%+"]),
                "adverse_move": ({} if fdf.empty else {
                    "median_mae_1min": float(fdf["mae1"].median()), "median_mae_5min": float(fdf["mae5"].median()),
                    "share_below_sl_in_1min": float((fdf["mae1"] <= -a.sl).mean()), "share_below_sl_in_5min": float((fdf["mae5"] <= -a.sl).mean()),
                    "median_spread_pct": float(fdf["spread"].median()), "median_tick_pct": float(fdf["tick_pct"].median())}),
            })
        pd.DataFrame(trades).to_csv(out_dir / f"vwap_sim_trades_{mode}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(free).to_csv(out_dir / f"vwap_sim_signals_{mode}.csv", index=False, encoding="utf-8-sig")
    (out_dir / "vwap_sim_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({m: {"days": len(report[m]["days"]), "symbol_days": report[m]["symbol_days"], "signals": report[m]["signals"],
                          "summary": report[m]["summary"]} for m in report if m != "params"}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
