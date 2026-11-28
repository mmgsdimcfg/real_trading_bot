# -*- coding: utf-8 -*-
"""g058 - entry-condition table (read-only study: no orders, no bot state, nothing written under runtime/).

Question (user 2026-11-28): pool the entry conditions of the rule trade, the VWAP trade, the Edge AI trade and the rebound alert,
leave them out one at a time, and look at a same-day take-profit / stop grid of 1..15 % each.

One row per watched symbol and completed 1-minute bar 09:05..14:50 (judged EVAL_DELAY_S after the bar's end):
  R_*  rule trade, r005 _009 triple trigger and the gates after it that look at market data only (r002 functions called as r005 does):
       R_1m / R_tick / R_guard / R_mid / R_drop / R_chase, R_regime, R_atr1 / R_atr3, R_liq, R_antichase (+ rise from the previous
       close), R_spread. NOT judged (state / account / order book depth): cool-down, positions, daily loss, opening-gap gate
       (before 09:05), quote status, 2-poll confirm, ask size.
  V_*  VWAP trade, the parts of r029.vwap_signal each by itself (V_live_now = the function's own answer with the running r001 values)
  E_*  Edge AI trade (r022): close above the 1-minute BB mid, the hybrid 1-minute trigger, and the model probability / threshold of
       the newest walk-forward OOS file (runtime/edge_ai/reports/oos_*.csv.gz) - only bars that are g013 events have one
  outcome: bought at the price + 1 tick, then the following 1-minute bars to 15:20 - for each level 1..15 % the first bar whose high
       reaches the take-profit / whose low reaches the stop (index + that bar's open), and the 15:20 close
The rebound-alert columns come from AutoBotHtsH/scripts/rebound-conds.ts (the app's own detector) on the bars_<day>.json written here.

Modes
  tick (default)  days with g011 tick files, the symbols r003 watched (g057.watched_on); price / spread from the prints
  bars            any collected day, the day's watchlist (picks of the last scan before it - sparse before 2026-10), no tick
                  conditions (R_tick / R_chase / V_tick are not judged, R_spread is False), price = the next bar's open

Limits: 1-minute bars come from the g001 10-second file (raw_bar == 1 rows); the 3-minute frame is completed buckets only (r003
also sees the forming one); the index ETF regime fails open when the ETF has no file that day.

usage: g058_entry_condition_table.py <out dir> [--mode tick|bars] [--from YYYYMMDD] [--to YYYYMMDD] [--workers 4]
       [--watch-json FILE]   bars mode: {trading day: [codes]} instead of the picks files - the same entry rules on another watchlist
       then: (AutoBotHtsH) npx vite-node scripts/rebound-conds.ts <out dir>
       then: g059_entry_condition_report.py <out dir> [bars]   /   g060_gap_atr_study.py <out dir> [bars]

Settings: the conditions use the r001 values of the process. The reported runs were made with AUTO_TRADING_LIVE_OVERRIDES=off (the
r001 defaults); without it the PC's runtime/config/live_overrides.json is applied and the samples differ (e.g. the 1-minute ATR floor).
entry_ok = the 1-minute bar the entry falls into exists; the reports leave the other rows out (their path after the fill is unknown).

Update log:
- [2026-11-28] type=feat owner=claude
    summary: --watch-json (봉 모드) - 선별 파일 대신 주어진 감시 목록으로 조건표 생성 (감시 목록끼리 같은 진입 규칙으로 비교).
    impact: sim (읽기 전용)
    compatibility: backward-compatible
- [2026-11-28] type=fix owner=claude
    summary: Codex 3차 검토 - entry_ok 열 추가 (진입 봉이 빠진 행 표시). 설정 조건(AUTO_TRADING_LIVE_OVERRIDES) 문서화.
    impact: sim (읽기 전용)
    compatibility: 이전에 만든 표는 g059 / g060 에서 다시 만들라고 중단됨
- [2026-11-28] type=feat owner=claude
    summary: 신규 - 매매 3종 + 반등 알림의 진입 조건을 봉마다 통과/실패로 기록하는 조건표 생성기 (틱 / 봉 모드), 익절 손절 1~15% 결과 배열 포함.
    impact: sim (읽기 전용: 주문 / 봇 상태 / runtime 쓰기 없음. 결과는 --out 폴더에만 쓴다)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import time as dt_time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

AB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AB / "live_trading"))
sys.path.insert(0, str(AB / "data_simulation"))
import r001_define_config as cfg  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
import r011_edge_ai as r011  # noqa: E402
import r029_vwap_trade as r029  # noqa: E402
import g013_edge_ai_dataset as g013  # noqa: E402
import g057_vwap_trade_sim as g057  # noqa: E402
from r013_tick_trigger import TickReplayFeed  # noqa: E402

ENTRY_START, ENTRY_END = dt_time(9, 5), dt_time(14, 50)
LAST_BAR = dt_time(15, 20)
EVAL_DELAY_S = 3.0
LEVELS = list(range(1, 16))
import os  # noqa: E402
BARS = os.environ.get("EA_MODE", "tick") == "bars"   # no tick files: tick conditions are not judged
DATA = Path(cfg.RUNTIME_ROOT) / "data"
TICKS = Path(cfg.TICK_DATA_DIR)
G = lambda k, d=None: getattr(cfg, k, d)   # noqa: E731


def market_map() -> dict[str, str]:
    return r002.load_symbol_market_map(AB / "data_simulation" / "g004_universe_symbols_master.txt")


def bars_of(day: str, code: str) -> pd.DataFrame | None:
    hit = list((DATA / day).glob(f"{code}_*_10s.txt"))
    if not hit:
        return None
    m = g013.load_minute_bars(hit[0])
    if m is None:
        return None
    t = m.index.time
    return m[(t > dt_time(9, 0)) & (t <= dt_time(15, 30))]


def prev_close_of(day: str, code: str) -> float:
    hit = list((DATA / day).glob(f"{code}_*_daily.csv"))
    if not hit:
        return float("nan")
    try:
        d = pd.read_csv(hit[0], encoding="utf-8-sig", usecols=["date", "close"], dtype={"date": str})
        d = d[d["date"] < day]
        return float(d["close"].iat[-1]) if len(d) else float("nan")
    except Exception:
        return float("nan")


def with_ind(today: pd.DataFrame, prev: pd.DataFrame | None, n_warm: int) -> pd.DataFrame:
    warm = prev.iloc[-n_warm:] if prev is not None and len(prev) else None
    return r002.calculate_indicators_with_warmup(today, warm)


def r3(m: pd.DataFrame) -> pd.DataFrame:
    return m[["open", "high", "low", "close", "volume"]].resample("3min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])


def etf_state(day: str, prev_day: str | None) -> dict[str, pd.Series]:
    """index ETF: per bar end label -> market regime ok (None = unknown)"""
    out = {}
    for mkt, code in dict(G("MARKET_REGIME_ETF", {})).items():
        b = bars_of(day, code)
        if b is None:
            continue
        f = with_ind(b, bars_of(prev_day, code) if prev_day else None, int(G("PRIOR_DAY_WARMUP_MINUTES", 180) or 180))
        res = {}
        for i in range(len(f)):
            ok, _ = r002.market_regime_ok(f.iloc[max(0, i - 199): i + 1], int(G("MARKET_REGIME_MA_FAST", 5)), int(G("MARKET_REGIME_MA_SLOW", 20)),
                                          int(G("MARKET_REGIME_CONFIRM_BARS", 1) or 1))
            res[f.index[i]] = ok
        out[mkt] = pd.Series(res)
    return out


def symbol_rows(a) -> tuple[list[dict], dict | None]:
    day, prev_day, code, name, market, regime, oos = a
    today = bars_of(day, code)
    if today is None or len(today) < 40:
        return [], None
    prev = bars_of(prev_day, code) if prev_day else None
    ticks = None if BARS else g057.load_ticks(TICKS / day / f"{code}.csv", day)
    if not BARS and (ticks is None or len(ticks) < 200):
        return [], None
    recv = ticks["recv"].values if ticks is not None else np.array([], dtype="datetime64[ns]")
    tick_px = ticks["price"].to_numpy(float) if ticks is not None else np.array([])
    f1 = with_ind(today, prev, int(G("PRIOR_DAY_WARMUP_MINUTES", 180) or 180))
    t3 = r3(today)
    p3 = r3(prev) if prev is not None and len(prev) else None
    f3 = r002.calculate_indicators_with_warmup(t3, p3.iloc[-60:] if p3 is not None and len(p3) else None)
    sess = today[today.index.time <= LAST_BAR]
    e1 = r011.fast_indicators(sess[["open", "high", "low", "close", "volume"]])
    e3 = r011.completed_3min(e1)
    sizes = tuple(int(x) for x in G("TICK_TRIGGER_BAR_SIZES", (60, 120)))
    feed_r = TickReplayFeed(TICKS, sizes, int(G("TICK_TRIGGER_MA_FAST", 5)), int(G("TICK_TRIGGER_MA_SLOW", 20)),
                            float(G("TICK_TRIGGER_CROSS_MAX_AGE_SECONDS", 240)), float(G("TICK_TRIGGER_FEED_STALE_SECONDS", 120)),
                            prior_day_warmup=bool(G("TICK_TRIGGER_PRIOR_DAY_WARMUP", True)), min_prints_per_min=float(G("TICK_MIN_PRINTS_PER_MIN", 0) or 0),
                            warmup_extra_bars=int(G("TICK_CROSS_WARMUP_EXTRA_BARS", 0) or 0), density_window_min=float(G("TICK_MIN_PRINTS_WINDOW_MINUTES", 30)))
    feed_v = TickReplayFeed(TICKS, (int(G("VWAP_TRADE_TICK_BAR_SIZE", 60)),), int(G("VWAP_TRADE_TICK_MA_FAST", 5)), int(G("VWAP_TRADE_TICK_MA_SLOW", 15)),
                            float(G("VWAP_TRADE_TICK_CROSS_MAX_AGE_SECONDS", 60)), float(G("VWAP_TRADE_TICK_FEED_STALE_SECONDS", 15)),
                            prior_day_warmup=bool(G("TICK_TRIGGER_PRIOR_DAY_WARMUP", True)), min_prints_per_min=float(G("VWAP_TRADE_MIN_PRINTS_PER_MIN", 30)),
                            density_window_min=float(G("VWAP_TRADE_MIN_PRINTS_WINDOW_MINUTES", 3)), warmup_extra_bars=int(G("TICK_CROSS_WARMUP_EXTRA_BARS", 0) or 0))
    P = r029.VwapParams.from_config(cfg)
    prev_close = prev_close_of(day, code)
    day_open = float(today["open"].iat[0]) if today.index[0].time() <= dt_time(9, 1) else float("nan")
    open_gap = (day_open / prev_close - 1) * 100 if prev_close == prev_close and prev_close > 0 and day_open == day_open else float("nan")
    close = today["close"].astype(float)
    vol = today["volume"].astype(float)
    cum_v = vol.cumsum()
    vwap_s = (close * vol).cumsum() / cum_v.where(cum_v > 0)
    ma5 = close.rolling(P.ma_fast).mean()
    mid20 = close.rolling(P.bb_period).mean()
    o, h, l, c = (sess[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sidx = sess.index
    fast, slow = int(G("TRIPLE_1MIN_MA_FAST", 5)), int(G("TRIPLE_1MIN_MA_SLOW", 20))
    reg = regime.get(market)
    rows = []
    for i, T in enumerate(today.index):
        if not (ENTRY_START <= T.time() <= ENTRY_END):
            continue
        now = T + pd.Timedelta(seconds=EVAL_DELAY_S)
        k = int(np.searchsorted(recv, np.datetime64(now), side="right")) - 1
        j = int(sidx.searchsorted(T, side="right"))            # the first bar after the judged one
        if j >= len(sidx):
            continue
        fill = float(ticks["price"].iat[k]) if k >= 0 else float(o[j])      # what the buy is priced from
        price = fill if k >= 0 else float(close.iat[i])                     # what the conditions see (no print: the confirmed close)
        tick_sz = r002.krx_tick_size(fill)
        last = float(close.iat[i])
        fr = f1.iloc[max(0, i - 199): i + 1]
        npy = now.to_pydatetime()
        # ---- rule trade (r005 _009 triple, then the context / pre-order gates that look at market data) ----
        tr = None if BARS else feed_r.evaluate(code, npy)
        crosses = getattr(tr, "crosses", None) or {}
        ma = r002.find_1min_ma_golden_cross(fr, fast, slow)
        age = (now - pd.Timestamp(ma["cross_bar_time"])).total_seconds() if ma["ok"] else float("nan")
        below = None
        if G("TRIPLE_1MIN_BELOW_TURN_PASS", False) and not ma["ok"] and str(ma.get("reason", "")).endswith("_NOT_ABOVE"):
            below = r002.evaluate_1min_below_turn(fr, fast, int(G("TRIPLE_1MIN_BELOW_TURN_LOOKBACK_BARS", 3)))
        trig = r002.evaluate_triple_trigger(ma, age, crosses, sizes, G("TRIPLE_1MIN_CONDITION", "above"), G("TRIPLE_TICK_CONDITION", "any"),
                                            float(G("TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS", 240)), fast, slow,
                                            above_max_age_s=float(G("TRIPLE_1MIN_ABOVE_MAX_AGE_SECONDS", 0) or 0), below_turn=below)
        miss = trig["missing"]
        r_tick = bool(trig["tick_cross_times"])
        r_1m = not [m for m in miss if not m.startswith("TICK")]
        guard_ok, _ = r002.check_buy_condition_1min_hybrid_trigger(fr, info={}, tick_cross_label=trig["label"], external_bars_since_cross=trig["bars_since_cross"])
        drop_ok, _ = r002.check_1min_trigger_live_guard(fr, price, require_above_mid=False)
        bbm = float(fr["BB_MIDDLE"].iat[-1]) if "BB_MIDDLE" in fr else float("nan")
        req_mid = bool(G("TRIPLE_REQUIRE_1MIN_ABOVE_MID", True)) and (not trig.get("below_turn") or bool(G("TRIPLE_BELOW_TURN_REQUIRE_ABOVE_MID", False)))
        r_mid = (not req_mid) or not (bbm == bbm) or price > bbm
        chase_ok, _ = r002.check_chase_from_cross(price, trig.get("cross_ref_price", float("nan")), float(G("TRIPLE_CHASE_FROM_CROSS_MAX_PCT", 0.5)))
        rg = reg.get(T) if reg is not None else None
        r_regime = True if rg is None else bool(rg)
        atr1 = float(fr["ATR"].iat[-1]) if "ATR" in fr else float("nan")
        r_atr1 = False if not (atr1 == atr1) else atr1 / last * 100 >= float(G("GC_VOLATILITY_MIN_ATR_1MIN_PCT", 0.5)) - 1e-9
        c3 = f3[f3.index <= T]
        r_atr3 = r_liq = True
        if len(c3):
            cur = c3.iloc[-1]
            a3 = float(cur.get("ATR", float("nan")))
            if a3 == a3:
                r_atr3 = a3 / price >= r002.min_entry_atr_threshold_pct(npy, True)
            ok_l, _c, _r = r002._gate_min_liquidity_safety(r002.BuyEvalContext(**{**_CTX_DEFAULT, "cur": cur, "live_price": price})) if _CTX_DEFAULT is not None else (True, None, None)
            r_liq = bool(ok_l)
        vw = float(vwap_s.iat[i])
        ac_ok, _ = r002.anti_chase_day_gate(price, day_open if day_open == day_open else None, vw if vw == vw else None)
        r_rise = True if not (prev_close == prev_close and prev_close > 0) else (price / prev_close - 1) < float(G("MAX_BUY_RISE_PCT_FROM_PREV_CLOSE", 0.23))
        ask = float(ticks["ask"].iat[k]) if k >= 0 else float("nan")
        bid = float(ticks["bid"].iat[k]) if k >= 0 else float("nan")
        spread_ticks = (ask - bid) / r002.krx_tick_size(bid) if ask == ask and bid == bid and bid > 0 and ask >= bid else float("nan")
        r_spread = spread_ticks == spread_ticks and spread_ticks <= float(G("MAX_ENTRY_SPREAD_TICKS", 2)) + 1e-9
        # ---- VWAP trade (r029.vwap_signal, each part by itself) ----
        v_above = vw == vw and vw > 0 and last > vw and price > vw
        v_hist = (i + 1) >= r029.MIN_BARS and today.index[0].time() <= r029.FULL_SESSION_FIRST_BAR
        v_trend = i >= 1 and bool((ma5.iat[i] > ma5.iat[i - 1]) or (mid20.iat[i] > mid20.iat[i - 1]))
        v_gap = open_gap == open_gap and open_gap >= P.min_open_gap_pct - 1e-9
        tv = None if BARS else feed_v.evaluate(code, npy)
        v_tick = bool(getattr(tv, "ok", False))
        cross = next((x for x in (getattr(tv, "crosses", None) or {}).values() if x.ok), None)
        cc = float(getattr(cross, "cross_close", float("nan"))) if cross is not None else float("nan")
        v_tchase = not (cc == cc and cc > 0 and price > cc * (1 + P.chase_max_pct / 100))
        v_cap = vw == vw and vw > 0 and price <= vw * (1 + P.max_vwap_gap_pct / 100)
        v_drop = price >= last * (1 - P.live_drop_max_pct / 100)
        v_live = r029.vwap_signal(today.iloc[: i + 1], price, npy, (g057._TickYes if BARS else tv), P, open_gap_pct=open_gap if open_gap == open_gap else None)[0]
        # ---- Edge AI trade (r022): close above the 1-minute BB mid, the hybrid 1-minute trigger, the model ----
        e_mid = e_trig = False
        if T in e1.index:
            ei = e1.index.get_loc(T)
            el = e1.iloc[ei]
            e_mid = bool(float(el["close"]) > float(el["BB_MIDDLE"])) if el["BB_MIDDLE"] == el["BB_MIDDLE"] else False
            m3 = e3[e3.index <= T] if e3 is not None else None
            try:
                e_trig, _ = r011.hybrid_trigger_with_context(e1.iloc[max(0, ei - 199): ei + 1], m3, float(el["close"]), {})
            except Exception:
                e_trig = False
        pr = oos.get((code, T))
        # ---- outcome: bought at the price + 1 tick right after the bar, the following 1-minute bars to 15:20 ----
        entry = fill + tick_sz
        hh, ll, oo = h[j:].copy(), l[j:].copy(), o[j:].copy()
        if k >= 0 and sidx[j] == T + pd.Timedelta(minutes=1):   # (Codex 1) the entry bar counts from the fill on, not from its first second
            oo[0] = fill
            m = (recv > np.datetime64(now)) & (recv < np.datetime64(sidx[j]))   # receive clock on both ends, like the decision
            hh[0] = max(fill, float(tick_px[m].max())) if m.any() else fill
            ll[0] = min(fill, float(tick_px[m].min())) if m.any() else fill
        tp_i, sl_i, tp_o, sl_o = [], [], [], []
        for u in LEVELS:
            w = np.flatnonzero(hh >= entry * (1 + u / 100) - 1e-6)
            tp_i.append(int(w[0]) if len(w) else 9999)
            tp_o.append(float(oo[w[0]]) if len(w) else float("nan"))
            w = np.flatnonzero(ll <= entry * (1 - u / 100) + 1e-6)
            sl_i.append(int(w[0]) if len(w) else 9999)
            sl_o.append(float(oo[w[0]]) if len(w) else float("nan"))
        rows.append({
            "day": day, "code": code, "name": name, "t": T, "price": price, "entry": entry, "tick": tick_sz, "fill": fill, "entry_ok": bool(sidx[j] == T + pd.Timedelta(minutes=1)), "eod": (float(c[-1]) if sidx[-1].time() >= LAST_BAR else float("nan")), "n_after": len(hh),
            "R_1m": r_1m, "R_tick": r_tick, "R_guard": bool(guard_ok), "R_mid": bool(r_mid), "R_drop": bool(drop_ok), "R_chase": bool(chase_ok),
            "R_regime": r_regime, "R_atr1": bool(r_atr1), "R_atr3": bool(r_atr3), "R_liq": r_liq, "R_antichase": bool(ac_ok) and bool(r_rise), "R_spread": bool(r_spread),
            "V_hist": bool(v_hist), "V_above": bool(v_above), "V_trend": bool(v_trend), "V_gap": bool(v_gap), "V_tick": v_tick, "V_tchase": bool(v_tchase), "V_cap": bool(v_cap), "V_drop": bool(v_drop),
            "V_live_now": bool(v_live),
            "E_mid": e_mid, "E_trig": bool(e_trig), "E_scored": bool(pr is not None and np.isfinite(pr[0]) and np.isfinite(pr[1])), "E_p": (pr[0] if pr else float("nan")), "E_thr": (pr[1] if pr else float("nan")),
            "tick_cov": bool(k >= 0 and (np.datetime64(now) - recv[0]) >= np.timedelta64(30, "m")
                             and (np.datetime64(now) - recv[k]) <= np.timedelta64(int(float(G("TICK_TRIGGER_FEED_STALE_SECONDS", 120)) * 1000), "ms")), "open_gap": open_gap, "vwap_gap": (price / vw - 1) * 100 if vw == vw and vw > 0 else float("nan"), "spread_ticks": spread_ticks,
            "tp_i": tp_i, "sl_i": sl_i, "tp_o": tp_o, "sl_o": sl_o,
        })
    # bars for the rebound detector (bar start, epoch ms; the file's clock is KST)
    both = pd.concat([prev, today]) if prev is not None and len(prev) else today
    ms = (both.index - pd.Timedelta(minutes=1) - pd.Timedelta(hours=9)).astype("datetime64[ms]").astype("int64").tolist()
    bars = [[int(t), float(a), float(b), float(c_), float(d), float(v)] for t, a, b, c_, d, v in
            zip(ms, both["open"], both["high"], both["low"], both["close"], both["volume"])]
    return rows, {"code": code, "bars": bars}


def _ctx_default():
    import dataclasses
    try:
        out = {}
        for f in dataclasses.fields(r002.BuyEvalContext):
            if f.default is not dataclasses.MISSING:
                out[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:   # type: ignore[misc]
                out[f.name] = f.default_factory()                 # type: ignore[misc]
            else:
                out[f.name] = None
        return out
    except Exception:
        return None


_CTX_DEFAULT = _ctx_default()


def day_job(a) -> str:
    day, prev_day, syms, out, oos_path = a
    regime = etf_state(day, prev_day)
    oos: dict = {}
    if oos_path:
        d = pd.read_csv(oos_path, dtype={"code": str, "date": str})
        d = d[d["date"] == day]
        oos = {(c, pd.Timestamp(t)): (float(p), float(th)) for c, t, p, th in zip(d["code"].str.zfill(6), d["ts"], d["p"], d["threshold"])}
    mm = market_map()
    rows, bars = [], []
    for code, name in syms:
        r, b = symbol_rows((day, prev_day, code, name, mm.get(code, ""), regime, oos))
        rows += r
        if b:
            bars.append(b)
    pd.DataFrame(rows).to_pickle(Path(out) / f"rows_{day}.pkl")
    (Path(out) / f"bars_{day}.json").write_text(json.dumps({"day": day, "syms": bars}), encoding="utf-8")
    return f"{day} symbols={len(bars)} rows={len(rows)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--from", dest="d0", default="20261030")
    ap.add_argument("--to", dest="d1", default="20261130")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=["tick", "bars"], default=os.environ.get("EA_MODE", "tick"))
    ap.add_argument("--watch-json", default="", help="bars mode: {trading day: [codes]} to use instead of the picks files (compare watchlists)")
    a = ap.parse_args()
    watch = json.loads(Path(a.watch_json).read_text(encoding="utf-8")) if a.watch_json else None
    if watch is not None and a.mode != "bars":
        raise SystemExit("--watch-json needs --mode bars")
    global BARS
    BARS = a.mode == "bars"
    os.environ["EA_MODE"] = a.mode          # the worker processes read it on import
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_days = sorted(d.name for d in DATA.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8 and any(d.glob("*_10s.txt")))
    oos_files = sorted((Path(cfg.RUNTIME_ROOT) / "edge_ai" / "reports").glob("oos_*.csv.gz"))
    oos_path = str(oos_files[-1]) if oos_files else ""
    jobs = []
    for day in all_days:
        if not (a.d0 <= day <= a.d1) or (not BARS and not (TICKS / day).is_dir()):
            continue
        names = g057.name_map(DATA, day)
        if watch is not None:   # [2026-11-28] a given watchlist of that trading day (a day without one is left out)
            if day not in watch:
                continue
            syms = [(c, names[c]) for c in watch[day] if c in names and not r011.is_fund_product(names[c])]
        elif BARS:
            syms = [(c, names[c]) for c, _n in g057.watchlist_of(DATA, day, all_days) if c in names and not r011.is_fund_product(names[c])]
        else:
            watched, _src = g057.watched_on(TICKS, DATA, day, all_days)
            syms = [(p.stem, names.get(p.stem, p.stem)) for p in sorted((TICKS / day).glob("*.csv"))
                    if p.stem.isdigit() and p.stem in watched and p.stem in names and not r011.is_fund_product(names.get(p.stem, ""))]
        prev = [d for d in all_days if d < day]
        jobs.append((day, prev[-1] if prev else None, syms, str(out), oos_path))
    print("days", len(jobs), "oos", oos_path, flush=True)
    if a.workers > 1 and len(jobs) > 1:
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(day_job, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(day_job(j), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
