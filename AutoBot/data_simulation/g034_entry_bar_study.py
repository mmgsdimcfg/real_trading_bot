# -*- coding: utf-8 -*-
"""G034 entry bar-configuration study (stage 2 of the entry-engine review, 2026-11-15).

Same days, same symbols, same exit - only the ENTRY differs.

Bar configurations (request stage 2)
  A  1-minute bars only            B  3-minute bars only
  C  60-tick bars only             D  120-tick bars only
  E  current style: 1-minute signal valid (15 min) AND a 60- or 120-tick signal valid (4 min)
     ("E_live" = the real r005 _009 trigger replayed every 10 s: 1m MA5>MA20 run <= 15 min, price > 1m MA20, 60/120-tick
     MA5xMA20 cross <= 4 min, chase <= 0.5 % over the earliest valid cross close)
  F1 / F3  hybrid: 1-minute (F1) or 3-minute (F3) bar = regime filter (close > MA20 and MA20 rising over 3 bars),
     60- or 120-tick signal = trigger
Indicator combinations per bar series
  SEQ   BB middle (MA20) up-cross -> MACD(12,26,9) line over signal -> RSI(14) over 50, in this order, each within W bars of
        the previous one, close still above the middle                     (the request's "current" combination)
  WIN   all three signals inside the last W bars, all three states still true (any order)
  LEAD  BB middle up-cross + leading filters at the cross: RVOL(3 min vs 30 min) >= 1.5, buy-share change (60 s vs the
        5 min before) > 0, price > session VWAP, 30 s price acceleration > 0. Order-book imbalance: NO DATA (g011 keeps
        level-1 prices only)
  MA    MA5 x MA20 up-cross (the live trigger's indicator, reference)
Fill model (no bar-close fills): the first KRX print received more than LAT_S after the signal became available; buy at
that print's ask1 (else price + 1 tick). Signal price = the signal bar's close. One position per symbol at a time.
Exit: the r024 adaptive exit engine (the live sell engine since v2.3.19) replayed on the same ticks through the r026 runner
(g032 style), NET of fee and tax; positions still open at 15:20 are marked at the last print (flag mtm).

Not the live bot: the gates after the trigger (_011.._025: ATR, liquidity, anti-chase, Edge AI, spread, 2-poll confirm,
market regime) are NOT applied to any configuration, so E_live is the raw trigger, not the live buy list.

  python g034_entry_bar_study.py [--workers 14] [--dates ...] [--w 10]
Output: <runtime>/data/sim/g034/<run>/trades.csv, signals.csv (every BB cross with its leading features), report.md

Update log
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 매수 엔진 2단계 봉 구성(A~F) x 지표 조합(SEQ/WIN/LEAD/MA) 비교, 청산은 r024 엔진 틱 리플레이로 통일.
    impact: sim only
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))
sys.path.insert(0, str(HERE))

from r001_define_config import RUNTIME_DATA_DIR, TICK_DATA_DIR  # noqa: E402
from r009_tick_bars import Tick  # noqa: E402
from r024_adaptive_exit import ExitConfig, pnl_pct, tick_size  # noqa: E402

ENTRY_START_S = 9 * 3600 + 5 * 60
ENTRY_END_S = 14 * 3600 + 50 * 60
SESSION_START_S = 9 * 3600
SESSION_END_S = 15 * 3600 + 20 * 60
LAT_S = 1.0                  # decision -> order (event-driven engine); the same for every configuration
TIME_BAR_DELAY_S = 5.0       # a time bar is used 5 s after its end (r003 CANDLE_CONFIRM_DELAY_SECONDS)
TICK_VALID_S = 240.0         # a tick-bar signal stays valid 4 min (live TICK_TRIGGER_MAX_AGE)
M1_VALID_S = 900.0           # a 1-minute signal stays valid 15 min (live 1m run age)
M3_VALID_S = 900.0
CHASE_PCT = 0.5
QTY = 100
NET = ExitConfig()

SERIES = ("m1", "m3", "t60", "t120")
COMBOS = ("SEQ", "WIN", "LEAD", "MA")
CONFIGS = ("A", "B", "C", "D", "E", "F1", "F3")
CONFIG_SERIES = {"A": "m1", "B": "m3", "C": "t60", "D": "t120"}
CONFIG_LABEL = {"A": "1분봉만", "B": "3분봉만", "C": "60틱봉만", "D": "120틱봉만", "E": "1분 + (60틱|120틱) 동시",
                "F1": "1분 레짐 + 60/120틱 트리거", "F3": "3분 레짐 + 60/120틱 트리거", "E_live": "현행 r005 _009 트리거 그대로"}


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def _sod(dt: pd.Series) -> np.ndarray:
    return (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second + dt.dt.microsecond / 1e6).to_numpy(dtype=np.float64)


def load_ticks(path: Path) -> pd.DataFrame:
    """KRX regular-session prints in receive order: rt (receive second of day), xt (exchange second), price, qty, side,
    ask, bid, ts (exchange datetime), recv (datetime)."""
    cols = ("recv_ts", "bsop_date", "trade_time", "price", "cntg_vol", "acml_vol", "ccld_dvsn", "askp1", "bidp1", "venue",
            "session", "seln_cntg_csnu", "shnu_cntg_csnu")
    try:
        t = pd.read_csv(path, usecols=lambda c: c in cols, dtype={"trade_time": str, "bsop_date": str}, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if "venue" in t.columns:
        t = t[t["venue"].fillna("KRX") == "KRX"]
    t = t[pd.to_numeric(t["price"], errors="coerce") > 0]
    t = t.dropna(subset=["recv_ts", "trade_time"])
    tt = t["trade_time"].astype(str).str.zfill(6)
    t = t[(tt >= "090000") & (tt < "152000")]
    if t.empty:
        return pd.DataFrame()
    tt = t["trade_time"].astype(str).str.zfill(6)
    out = pd.DataFrame({
        "recv": pd.to_datetime(t["recv_ts"], errors="coerce"),
        "ts": pd.to_datetime(t["bsop_date"].astype(str) + tt, format="%Y%m%d%H%M%S", errors="coerce"),
        "price": t["price"].astype(float), "qty": pd.to_numeric(t["cntg_vol"], errors="coerce").fillna(0).astype(np.int64),
        "acml": pd.to_numeric(t["acml_vol"], errors="coerce").fillna(0).astype(np.int64),
        "side": pd.to_numeric(t["ccld_dvsn"], errors="coerce").fillna(0).astype(int),
        "ask": pd.to_numeric(t.get("askp1"), errors="coerce").fillna(0.0) if "askp1" in t else 0.0,
        "bid": pd.to_numeric(t.get("bidp1"), errors="coerce").fillna(0.0) if "bidp1" in t else 0.0,
    }).dropna(subset=["recv", "ts"])
    out = out.drop_duplicates(subset=["ts", "acml", "price", "qty"])          # same print logged twice
    out = out.sort_values("recv", kind="stable").reset_index(drop=True)
    out["rt"] = _sod(out["recv"])
    out["xt"] = _sod(out["ts"])
    return out


def minute_bars_from_ticks(t: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Time bars on the exchange clock, anchored at 09:00. avail = end + TIME_BAR_DELAY_S (receive clock ~ exchange)."""
    if t.empty:
        return pd.DataFrame(columns=["avail", "open", "high", "low", "close", "volume"])
    b = ((t["xt"] - SESSION_START_S) // (60 * minutes)).astype(int)
    g = t.groupby(b, sort=True)
    bars = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                         "close": g["price"].last(), "volume": g["qty"].sum()})
    bars["avail"] = SESSION_START_S + (bars.index.to_numpy() + 1) * 60 * minutes + TIME_BAR_DELAY_S
    return bars.reset_index(drop=True)


def tick_bars(t: pd.DataFrame, n: int) -> pd.DataFrame:
    """Bars of n prints in receive order (approximation of r009: missed prints are not counted). avail = receive time of
    the print that completed the bar."""
    m = len(t) // n
    if m == 0:
        return pd.DataFrame(columns=["avail", "open", "high", "low", "close", "volume"])
    p = t["price"].to_numpy()[: m * n].reshape(m, n)
    q = t["qty"].to_numpy()[: m * n].reshape(m, n)
    rt = t["rt"].to_numpy()[: m * n].reshape(m, n)
    return pd.DataFrame({"avail": rt[:, -1], "open": p[:, 0], "high": p.max(1), "low": p.min(1), "close": p[:, -1],
                         "volume": q.sum(1)})


def prior_minute_closes(day: str, code: str, minutes: int) -> np.ndarray:
    """Previous collected day's 1-minute closes from the 10 s file (raw_bar=1), resampled to `minutes`. Warm-up only."""
    days = sorted(p.name for p in RUNTIME_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8
                  and p.name < day)
    for d in reversed(days[-5:]):
        files = sorted((RUNTIME_DATA_DIR / d).glob(f"{code}_*_10s.txt"))
        if not files:
            continue
        try:
            df = pd.read_csv(files[0], encoding="utf-8-sig", usecols=["datetime", "close", "raw_bar"])
        except Exception:
            continue
        df = df[df["raw_bar"].astype(str).isin(["1", "1.0", "True"])]
        if df.empty:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        s = df.set_index("datetime")["close"].astype(float).sort_index()
        s = s[(s.index.time >= datetime.strptime("09:01", "%H:%M").time())
              & (s.index.time <= datetime.strptime("15:30", "%H:%M").time())]
        if minutes > 1:
            s = s.resample(f"{minutes}min", label="right", closed="right", origin="start_day",
                           offset="9h").last().dropna()
        return s.to_numpy()[-120:]
    return np.array([])


def prior_tick_closes(day: str, code: str, n: int) -> np.ndarray:
    days = sorted(p.name for p in TICK_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit() and p.name < day)
    for d in reversed(days[-3:]):
        f = TICK_DATA_DIR / d / f"{code}.csv"
        if f.exists():
            t = load_ticks(f)
            if len(t) >= n * 30:
                return tick_bars(t, n)["close"].to_numpy()[-120:]
    return np.array([])


# ---------------------------------------------------------------------------------------------------------------------
# indicators / signals
# ---------------------------------------------------------------------------------------------------------------------
def indicators(closes: np.ndarray) -> dict:
    s = pd.Series(closes, dtype=float)
    mid = s.rolling(20).mean()
    ma5 = s.rolling(5).mean()
    ema12, ema26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi = rsi.fillna(100.0).where(s.index >= 14, np.nan)
    hist = (macd - sig).where(s.index >= 33, np.nan)
    return {"close": s.to_numpy(), "mid": mid.to_numpy(), "ma5": ma5.to_numpy(), "hist": hist.to_numpy(),
            "rsi": rsi.to_numpy()}


def _up(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """a crosses above b at k (a[k] > b[k] and a[k-1] <= b[k-1]); NaN -> False."""
    b = np.broadcast_to(b, a.shape) if np.isscalar(b) else b
    now = a > b
    prev = np.r_[False, (a[:-1] <= b[:-1])]
    return now & prev


def series_signals(bars: pd.DataFrame, warm: np.ndarray, w: int) -> dict:
    """Fire flags per combination for TODAY's bars (warm-up closes only feed the indicators)."""
    closes = np.r_[warm, bars["close"].to_numpy(dtype=float)]
    ind = indicators(closes)
    o = len(warm)
    c, mid, ma5, hist, rsi = ind["close"], ind["mid"], ind["ma5"], ind["hist"], ind["rsi"]
    with np.errstate(invalid="ignore"):
        bbx = _up(c, mid)
        macdx = _up(hist, 0.0)
        rsix = _up(rsi, 50.0)
        max_ = _up(ma5, mid)
        above = c > mid
        hpos = hist > 0
        rpos = rsi > 50
    n = len(c)
    seq = np.zeros(n, bool)
    stage, last_k = 0, -1
    for k in range(n):
        if not above[k]:
            stage = 0
        if stage == 0 and bbx[k]:
            stage, last_k = 1, k
        if stage == 1 and macdx[k] and k - last_k <= w:
            stage, last_k = 2, k
        elif stage == 1 and k - last_k > w:
            stage = 0
        if stage == 2 and rsix[k] and k - last_k <= w and above[k]:
            seq[k] = True
            stage = 0
        elif stage == 2 and k - last_k > w:
            stage = 0
    win = np.zeros(n, bool)
    lb = lm = lr = -10 ** 9
    prev_ok = False
    for k in range(n):
        if bbx[k]:
            lb = k
        if macdx[k]:
            lm = k
        if rsix[k]:
            lr = k
        ok = bool(above[k] and hpos[k] and rpos[k] and k - lb < w and k - lm < w and k - lr < w)
        win[k] = ok and not prev_ok
        prev_ok = ok
    regime = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        regime[3:] = above[3:] & (mid[3:] > mid[:-3])
    sl = slice(o, None)
    return {"avail": bars["avail"].to_numpy(dtype=float), "close": bars["close"].to_numpy(dtype=float),
            "SEQ": seq[sl], "WIN": win[sl], "BBX": bbx[sl], "MA": max_[sl], "above": above[sl], "regime": regime[sl],
            "ma5_above": (ma5 > mid)[sl], "mid": mid[sl]}


class Lead:
    """Leading features at a receive time from the raw prints (all strictly before or at t: no look-ahead)."""

    def __init__(self, t: pd.DataFrame):
        self.rt = t["rt"].to_numpy()
        self.p = t["price"].to_numpy()
        q = t["qty"].to_numpy(dtype=float)
        side = t["side"].to_numpy()
        self.cq = np.r_[0.0, np.cumsum(q)]
        self.cb = np.r_[0.0, np.cumsum(np.where(side == 1, q, 0.0))]
        self.cs = np.r_[0.0, np.cumsum(np.where(side == 5, q, 0.0))]
        self.cpq = np.r_[0.0, np.cumsum(q * self.p)]

    def _i(self, t: float) -> int:
        return int(np.searchsorted(self.rt, t, side="right"))     # prints [0, i) are visible at t

    def px(self, t: float) -> float:
        i = self._i(t)
        return float(self.p[i - 1]) if i else float("nan")

    def features(self, t: float) -> dict:
        i = self._i(t)
        if i == 0:
            return {}
        i3, i33 = self._i(t - 180), self._i(t - 1980)
        v3 = self.cq[i] - self.cq[i3]
        hist_span = min(1800.0, t - 180 - SESSION_START_S)
        vh = self.cq[i3] - self.cq[i33]
        rvol = v3 / (vh / (hist_span / 180.0)) if hist_span >= 360 and vh > 0 else float("nan")
        i1, i6 = self._i(t - 60), self._i(t - 360)
        b1, s1 = self.cb[i] - self.cb[i1], self.cs[i] - self.cs[i1]
        b0, s0 = self.cb[i1] - self.cb[i6], self.cs[i1] - self.cs[i6]
        sh1 = b1 / (b1 + s1) if b1 + s1 > 0 else float("nan")
        sh0 = b0 / (b0 + s0) if b0 + s0 > 0 else float("nan")
        vwap = self.cpq[i] / self.cq[i] if self.cq[i] > 0 else float("nan")
        p, p30, p60 = self.px(t), self.px(t - 30), self.px(t - 60)
        r1 = (p / p30 - 1) * 100 if p30 == p30 and p30 > 0 else float("nan")
        r0 = (p30 / p60 - 1) * 100 if p60 == p60 and p60 > 0 else float("nan")
        return {"rvol": rvol, "dshare": sh1 - sh0, "share60": sh1, "vwap_gap": (p / vwap - 1) * 100 if vwap == vwap else float("nan"),
                "accel": r1 - r0, "ret30s": r1}


def lead_ok(f: dict) -> bool:
    """LEAD filter (가정 thresholds, backtest to confirm). A NaN feature (too early in the day) passes that check."""
    def ok(v, test):
        return v != v or test(v)
    return (ok(f.get("rvol", np.nan), lambda v: v >= 1.5) and ok(f.get("dshare", np.nan), lambda v: v > 0)
            and ok(f.get("vwap_gap", np.nan), lambda v: v > 0) and ok(f.get("accel", np.nan), lambda v: v > 0))


# ---------------------------------------------------------------------------------------------------------------------
# configurations -> signal events (time, signal price)
# ---------------------------------------------------------------------------------------------------------------------
def _events(sig: dict, flag: np.ndarray) -> list[tuple[float, float]]:
    return [(float(sig["avail"][k]), float(sig["close"][k])) for k in np.flatnonzero(flag)]


def _valid_at(ev: list[tuple[float, float]], t: float, span: float) -> tuple[float, float] | None:
    """Earliest event in [t - span, t] (the chase reference is the earliest valid signal, like live)."""
    for et, ep in ev:
        if t - span <= et <= t:
            return et, ep
    return None


def _state_at(sig: dict, key: str, t: float) -> bool:
    k = int(np.searchsorted(sig["avail"], t, side="right")) - 1
    return bool(k >= 0 and sig[key][k])


def config_events(cfg: str, combo: str, sigs: dict, lead: Lead) -> list[tuple[float, float]]:
    def raw(series: str) -> list[tuple[float, float]]:
        s = sigs[series]
        if combo == "LEAD":
            return [(t, p) for (t, p) in _events(s, s["BBX"]) if lead_ok(lead.features(t))]
        return _events(s, s[combo])

    if cfg in CONFIG_SERIES:
        return raw(CONFIG_SERIES[cfg])
    ticks_ev = sorted(raw("t60") + raw("t120"))
    if cfg == "E":
        m1 = raw("m1")
        out = []
        for t, p in sorted(set(ticks_ev + m1)):
            a = _valid_at(m1, t, M1_VALID_S)
            b = _valid_at(ticks_ev, t, TICK_VALID_S)
            if a and b and _state_at(sigs["m1"], "above", t):
                out.append((t, min(a, b)[1] if a[0] <= b[0] else b[1]))
        return out
    reg = "m1" if cfg == "F1" else "m3"
    return [(t, p) for (t, p) in ticks_ev if _state_at(sigs[reg], "regime", t)]


def live_events(sigs: dict, lead: Lead) -> list[tuple[float, float]]:
    """r005 _009 'triple' trigger polled every 10 s (g030 'live')."""
    m1 = sigs["m1"]
    runs = []
    for s in ("t60", "t120"):
        sg = sigs[s]
        runs.append(_events(sg, sg["MA"]))
    m1_starts = _events(m1, m1["MA"])
    out = []
    for T in range(ENTRY_START_S, ENTRY_END_S + 1, 10):
        k = int(np.searchsorted(m1["avail"], T, side="right")) - 1
        if k < 0 or not m1["ma5_above"][k]:
            continue
        px = lead.px(T)
        if not (px == px) or not (px > m1["mid"][k]):
            continue
        st = [e for e in m1_starts if e[0] <= T]
        if st and T - st[-1][0] > M1_VALID_S + 60:            # run started long ago (bar avail vs label: +1 bar)
            continue
        ref = None
        for (ev, s) in zip(runs, ("t60", "t120")):
            sg = sigs[s]
            j = int(np.searchsorted(sg["avail"], T, side="right")) - 1
            if j < 0 or not sg["ma5_above"][j]:
                continue
            past = [e for e in ev if e[0] <= T]
            if past and T - past[-1][0] <= TICK_VALID_S:
                if ref is None or past[-1][0] < ref[0]:
                    ref = past[-1]
        if ref is None:
            continue
        if (px / ref[1] - 1) * 100 > CHASE_PCT:
            continue
        out.append((float(T), float(ref[1])))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# exit: r024 engine replay (g032 style)
# ---------------------------------------------------------------------------------------------------------------------
class _Tail:
    def __init__(self, ticks: list[Tick], start: int):
        self.ticks, self.i, self.now, self.resets = ticks, start, None, 0

    def poll(self):
        out = []
        while self.i < len(self.ticks) and self.ticks[self.i].recv <= self.now:
            out.append(self.ticks[self.i])
            self.i += 1
        return out


def to_tick_objects(t: pd.DataFrame, code: str) -> list[Tick]:
    return [Tick(code=code, ts=r.ts.to_pydatetime(), price=float(r.price), qty=int(r.qty), acml_vol=int(r.acml),
                 side=int(r.side), cum_count=None, ask1=float(r.ask), bid1=float(r.bid), recv=r.recv.to_pydatetime())
            for r in t.itertuples(index=False)]


def engine_exit(day: str, code: str, ticks: list[Tick], recv_keys: list[datetime], entry_dt: datetime, entry_px: float,
                tmp: Path) -> tuple[float, datetime, str, bool]:
    """(net %, exit time, exit kinds, marked-to-market) for one position bought at entry_px at entry_dt."""
    import bisect
    from r026_exit_shadow import ShadowExitRunner
    out_dir = tmp / f"{code}_{entry_dt:%H%M%S}"
    runner = ShadowExitRunner(TICK_DATA_DIR, RUNTIME_DATA_DIR, out_dir, day, lambda _m: None, ExitConfig(), cleanup=False)
    pos = {"buy_price": entry_px, "quantity": QTY, "entry_buy_time": entry_dt}
    k0 = bisect.bisect_right(recv_keys, entry_dt)
    last_px = ticks[k0 - 1].price if k0 else entry_px
    runner.observe(code, code, entry_dt, pos, QTY, last_px, price_ts=entry_dt)
    try:
        if len(runner.positions) != 1:
            return float("nan"), entry_dt, "NO_OPEN", True
        vp = next(iter(runner.positions.values()))
        feed = _Tail(ticks, 0)
        while feed.i < len(ticks) and ticks[feed.i].recv <= entry_dt - timedelta(minutes=35):
            feed.i += 1                                          # the window needs ~30 min of history, not the whole day
        vp.tail = feed
        now = entry_dt.replace(microsecond=0) + timedelta(seconds=1)
        end = entry_dt.replace(hour=15, minute=20, second=0, microsecond=0)
        n = 0
        while now <= end and runner.positions:
            feed.now = now
            if n % 5 == 0:
                k = bisect.bisect_right(recv_keys, now)
                if k:
                    last = ticks[k - 1]
                    runner.observe(code, code, now, pos, QTY, last.price, price_ts=last.recv)
            runner.tick(now)
            now += timedelta(seconds=1)
            n += 1
        if runner.positions:
            runner.finalize(now, "REPLAY_END")
        tpath = out_dir / f"exit_shadow_trades_{day}.jsonl"
        trades = [json.loads(x) for x in tpath.read_text(encoding="utf-8").splitlines()] if tpath.exists() else []
        tr = trades[-1] if trades else None
        if tr and tr["status"] == "CLOSED" and tr["fills"]:
            return (float(tr["net_pct_sold"]), datetime.fromisoformat(tr["fills"][-1]["ts"]),
                    "+".join(f["kind"] for f in tr["fills"]), False)
        sold = tr["fills"] if tr else []
        k = bisect.bisect_right(recv_keys, end)
        last_px = ticks[k - 1].price if k else entry_px
        sold_q = sum(int(f["qty"]) for f in sold)
        proceeds = sum(float(f["price"]) * int(f["qty"]) for f in sold) + last_px * (QTY - sold_q)
        avg = proceeds / QTY
        return pnl_pct(NET, entry_px, avg), end, "+".join([f["kind"] for f in sold] + ["MTM"]), True
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------
# one symbol-day
# ---------------------------------------------------------------------------------------------------------------------
def _dt(day: str, sod: float) -> datetime:
    return datetime.strptime(day, "%Y%m%d") + timedelta(seconds=float(sod))


def prev_turnover(day: str, code: str) -> float:
    files = sorted((RUNTIME_DATA_DIR / day).glob(f"{code}_*_daily.csv"))
    if not files:
        return float("nan")
    try:
        d = pd.read_csv(files[0], encoding="utf-8-sig", usecols=["date", "amount"], dtype={"date": str})
        prev = d[d["date"].str.slice(0, 8) < day]
        return float(prev["amount"].iloc[-1]) if len(prev) else float("nan")
    except Exception:
        return float("nan")


def run_code_day(task) -> tuple[list[dict], list[dict]]:
    day, code, w = task
    t = load_ticks(TICK_DATA_DIR / day / f"{code}.csv")
    if len(t) < 2000:                                            # too thin for 60/120-tick bars (g030 rule)
        return [], []
    lead = Lead(t)
    sigs = {
        "m1": series_signals(minute_bars_from_ticks(t, 1), prior_minute_closes(day, code, 1), w),
        "m3": series_signals(minute_bars_from_ticks(t, 3), prior_minute_closes(day, code, 3), w),
        "t60": series_signals(tick_bars(t, 60), prior_tick_closes(day, code, 60), w),
        "t120": series_signals(tick_bars(t, 120), prior_tick_closes(day, code, 120), w),
    }
    ticks = to_tick_objects(t, code)
    recv_keys = [x.recv for x in ticks]
    rt, price, ask = t["rt"].to_numpy(), t["price"].to_numpy(), t["ask"].to_numpy()
    turnover = prev_turnover(day, code)
    cache: dict[tuple[float, float], tuple] = {}
    tmp = Path(tempfile.mkdtemp(prefix=f"g034_{day}_{code}_"))
    trades, sig_rows = [], []
    try:
        plans = [(c, combo) for c in CONFIGS for combo in COMBOS] + [("E_live", "MA")]
        for cfg, combo in plans:
            ev = live_events(sigs, lead) if cfg == "E_live" else config_events(cfg, combo, sigs, lead)
            busy = -1.0
            for (ts, sp) in ev:
                if ts < ENTRY_START_S or ts > ENTRY_END_S or ts < busy:
                    continue
                i = int(np.searchsorted(rt, ts + LAT_S, side="right"))
                if i >= len(rt) or rt[i] - ts > 120:
                    continue
                fill = float(ask[i]) if ask[i] > 0 and ask[i] >= price[i] else float(price[i] + tick_size(price[i]))
                key = (round(float(rt[i]), 3), fill)
                if key not in cache:
                    entry_dt = _dt(day, rt[i])
                    net, exit_dt, kinds, mtm = engine_exit(day, code, ticks, recv_keys, entry_dt, fill, tmp)
                    j10 = int(np.searchsorted(rt, rt[i] + 600, side="right"))
                    j30 = int(np.searchsorted(rt, rt[i] + 1800, side="right"))
                    seg = price[i:j30] if j30 > i else price[i:i + 1]
                    cache[key] = (net, exit_dt, kinds, mtm, (price[max(i, j10 - 1)] / fill - 1) * 100,
                                  (price[max(i, j30 - 1)] / fill - 1) * 100, (seg.max() / fill - 1) * 100,
                                  (seg.min() / fill - 1) * 100)
                net, exit_dt, kinds, mtm, r10, r30, mfe, mae = cache[key]
                f = lead.features(ts)
                trades.append({"date": day, "code": code, "config": cfg, "combo": combo, "sig_t": ts, "sig_px": sp,
                               "fill_t": float(rt[i]), "fill": fill, "slip_pct": (fill / sp - 1) * 100 if sp > 0 else np.nan,
                               "net": net, "exit_t": exit_dt.isoformat(timespec="seconds"), "exit_kinds": kinds, "mtm": mtm,
                               "ret10": r10, "ret30": r30, "mfe30": mfe, "mae30": mae, "turnover_prev": turnover, **f})
                busy = (exit_dt - datetime.strptime(day, "%Y%m%d")).total_seconds()
        # every BB-middle cross of every series with its leading features and the forward path (no position logic)
        for s in SERIES:
            sg = sigs[s]
            for ts, sp in _events(sg, sg["BBX"]):
                if ts < ENTRY_START_S or ts > ENTRY_END_S:
                    continue
                i = int(np.searchsorted(rt, ts + LAT_S, side="right"))
                if i >= len(rt):
                    continue
                fill = float(ask[i]) if ask[i] > 0 and ask[i] >= price[i] else float(price[i] + tick_size(price[i]))
                j30 = int(np.searchsorted(rt, rt[i] + 1800, side="right"))
                seg = price[i:max(j30, i + 1)]
                sig_rows.append({"date": day, "code": code, "series": s, "sig_t": ts, "sig_px": sp, "fill": fill,
                                 "ret30": (price[max(i, j30 - 1)] / fill - 1) * 100, "mfe30": (seg.max() / fill - 1) * 100,
                                 "mae30": (seg.min() / fill - 1) * 100, **lead.features(ts)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return trades, sig_rows


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------
def _q(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.mean():+.3f} / {s.median():+.3f}" if len(s) else "-"


def stats(g: pd.DataFrame, n_days: int) -> dict:
    net = pd.to_numeric(g["net"], errors="coerce").dropna()
    if net.empty:
        return {"n": 0}
    wins, losses = net[net > 0], net[net <= 0]
    seq = g.sort_values(["date", "fill_t"])["net"].astype(float).fillna(0).cumsum()
    mdd = float((seq.cummax() - seq).max()) if len(seq) else 0.0
    by = g.groupby("date")["net"].sum()
    return {"n": len(net), "per_day": len(net) / n_days, "win": (net > 0).mean() * 100, "mean": net.mean(),
            "median": net.median(), "payoff": (wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
            "sum": net.sum(), "mdd": mdd, "slip": pd.to_numeric(g["slip_pct"], errors="coerce").mean(),
            "mfe": g["mfe30"].mean(), "mae": g["mae30"].mean(), "days_pos": int((by > 0).sum()), "mtm": int(g["mtm"].sum())}


def _row(name: str, label: str, s: dict) -> str:
    if not s.get("n"):
        return f"| {name} | {label} | 0 | | | | | | | | | |"
    small = " (표본 작음)" if s["n"] < 30 else ""
    return (f"| {name} | {label} | {s['n']}{small} | {s['per_day']:.1f} | {s['win']:.0f}% | {s['payoff']:.2f} | "
            f"{s['mean']:+.3f} | {s['median']:+.3f} | {s['mdd']:.1f} | {s['slip']:+.3f} | {s['mfe']:+.2f} / {s['mae']:+.2f} | "
            f"{s['days_pos']} |")


HEAD = ["| 구성 | 설명 | 거래 | /일 | 승률 | 손익비 | 기대값(순 %) | 중앙값 | MDD(%p) | 진입 슬리피지 % | MFE / MAE 30분 | 플러스 일 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]


def walk_forward(d: pd.DataFrame, days: list[str], min_train: int = 5) -> tuple[list[str], pd.DataFrame]:
    """Expanding window: for each test day choose the configuration x combination with the best mean net on the earlier days
    (>= 30 trades), then record its trades on the test day."""
    lines, picked = [], []
    for k in range(min_train, len(days)):
        train = d[d["date"].isin(days[:k])]
        g = train.groupby(["config", "combo"])["net"].agg(["mean", "count"])
        g = g[g["count"] >= 30]
        if g.empty:
            continue
        best = g["mean"].idxmax()
        test = d[(d["date"] == days[k]) & (d["config"] == best[0]) & (d["combo"] == best[1])]
        live = d[(d["date"] == days[k]) & (d["config"] == "E_live")]
        picked.append(test)
        lines.append(f"| {days[k]} | {best[0]}/{best[1]} ({g.loc[best, 'mean']:+.3f}, n={int(g.loc[best, 'count'])}) | "
                     f"{len(test)} | {_q(test['net'])} | {len(live)} | {_q(live['net'])} |")
    return lines, (pd.concat(picked) if picked else pd.DataFrame())


def report(d: pd.DataFrame, sg: pd.DataFrame, days: list[str], w: int) -> str:
    n_days = len(days)
    split = days[: max(1, min(n_days - 1, int(round(n_days * 2 / 3))))] if n_days > 1 else days
    L = [f"# g034 매수 봉 구성 비교 ({days[0]} ~ {days[-1]}, {n_days}거래일, 종목-일 {d.groupby(['date', 'code']).ngroups})", "",
         f"청산은 모든 구성에 r024 적응형 엔진(틱 리플레이), 비용(수수료 양방향 + 세금) 포함 순손익 %. 진입 = 신호 {LAT_S:.0f}초 뒤 첫 체결의 매도1호가. "
         f"SEQ/WIN 창 W={w}봉. 주문 직전 게이트(_011~_025)는 어느 구성에도 적용하지 않았다 (E_live 도 트리거만).", "",
         "## 1. 전체 (구성 x 지표 조합)", ""] + HEAD
    for cfg in list(CONFIGS) + ["E_live"]:
        for combo in (COMBOS if cfg != "E_live" else ("MA",)):
            g = d[(d["config"] == cfg) & (d["combo"] == combo)]
            L.append(_row(f"{cfg}/{combo}", CONFIG_LABEL.get(cfg, cfg), stats(g, n_days)))
    L += ["", "MDD = 거래를 시간순으로 1슬롯씩 누적한 순손익 합의 최대 낙폭(%p). 표본 30건 미만은 통계적 의미가 약하다.", ""]
    L += [f"## 2. 학습 / 검증 분할 (학습 {split[0]}~{split[-1]}, 검증 {days[min(len(split), n_days - 1)]}~{days[-1]})", "",
          "| 구성 | 학습 거래 | 학습 기대값 | 검증 거래 | 검증 기대값 |", "|---|---|---|---|---|"]
    tr, te = d[d["date"].isin(split)], d[~d["date"].isin(split)]
    for (cfg, combo), _g in d.groupby(["config", "combo"]):
        a, b = tr[(tr["config"] == cfg) & (tr["combo"] == combo)], te[(te["config"] == cfg) & (te["combo"] == combo)]
        L.append(f"| {cfg}/{combo} | {len(a)} | {_q(a['net'])} | {len(b)} | {_q(b['net'])} |")
    wf_lines, wf = walk_forward(d, days)
    L += ["", "## 3. 워크포워드 (확장 창: 이전 날들로 최고 구성 선택 -> 다음 날 적용)", "",
          "| 검증일 | 선택 (학습 기대값, 건수) | 거래 | 기대값 | 현행 E_live 거래 | 현행 기대값 |", "|---|---|---|---|---|---|"] + wf_lines
    if not wf.empty:
        live_te = d[d["date"].isin(wf["date"].unique()) & (d["config"] == "E_live")]
        L += ["", f"- 워크포워드 합계: 선택 구성 {len(wf)}건 {_q(wf['net'])} vs 현행 E_live {len(live_te)}건 {_q(live_te['net'])}"]
    # segments for the configurations with the most trades of each family
    L += ["", "## 4. 종목 유형·시간대별 (기대값 순 % 평균 / 건수)", ""]
    d = d.copy()
    d["band_px"] = pd.cut(d["fill"], [0, 10000, 50000, np.inf], labels=["1만원 이하", "1~5만원", "5만원 이상"])
    d["band_tod"] = pd.cut(d["sig_t"], [0, 10 * 3600, 14 * 3600, 24 * 3600], labels=["09:05~10:00", "10:00~14:00", "14:00~14:50"])
    q = d["turnover_prev"].quantile([1 / 3, 2 / 3]).to_numpy() if d["turnover_prev"].notna().any() else [np.nan, np.nan]
    d["band_liq"] = pd.cut(d["turnover_prev"], [-np.inf, q[0], q[1], np.inf], labels=["전일 거래대금 하위", "중위", "상위"]) \
        if q[0] == q[0] else "?"
    for col, title in (("band_px", "가격대"), ("band_tod", "시간대"), ("band_liq", "유동성(전일 거래대금 3분위)")):
        cats = [c for c in d[col].dropna().unique()]
        cats = sorted(cats, key=str)
        L += [f"### {title}", "", "| 구성 | " + " | ".join(str(c) for c in cats) + " |", "|---|" + "---|" * len(cats)]
        for (cfg, combo), g in d.groupby(["config", "combo"]):
            cells = []
            for c in cats:
                x = g.loc[g[col] == c, "net"]
                cells.append(f"{x.mean():+.2f} / {len(x)}" if len(x) else "-")
            L.append(f"| {cfg}/{combo} | " + " | ".join(cells) + " |")
        L.append("")
    # leading features on every BB cross
    if not sg.empty:
        L += ["## 5. 선행 지표별 BB 중간선 돌파 이후 30분 수익 (모든 돌파, 포지션 무관, 비용 전 %)", "",
              "| 지표 | 구간 | 1분 | 3분 | 60틱 | 120틱 |", "|---|---|---|---|---|---|"]
        bins = {"rvol": [0, 0.7, 1.0, 1.5, 2.5, np.inf], "dshare": [-1, -0.1, 0, 0.1, 1],
                "vwap_gap": [-np.inf, -0.5, 0, 0.5, 1.5, np.inf], "accel": [-np.inf, -0.2, 0, 0.2, np.inf]}
        for f, b in bins.items():
            cut = pd.cut(sg[f], b)
            for k in cut.cat.categories:
                cells = []
                for s in SERIES:
                    x = sg.loc[(cut == k) & (sg["series"] == s), "ret30"]
                    cells.append(f"{x.mean():+.2f} ({len(x)})" if len(x) else "-")
                L.append(f"| {f} | {k} | " + " | ".join(cells) + " |")
        L += ["", "| 전체 돌파 | | " + " | ".join(f"{sg.loc[sg['series'] == s, 'ret30'].mean():+.2f} ({int((sg['series'] == s).sum())})"
                                              for s in SERIES) + " |", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N symbol-days only (smoke test)")
    args = ap.parse_args()
    days = sorted(p.name for p in TICK_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()
                  and (not args.dates or p.name in args.dates))
    tasks = [(d, f.stem, args.w) for d in days for f in sorted((TICK_DATA_DIR / d).glob("*.csv"))]
    if args.limit:
        tasks = tasks[: args.limit]
    out = Path(args.out) if args.out else RUNTIME_DATA_DIR / "sim" / "g034" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    trades, sigs = [], []
    with Pool(args.workers) as pool:
        for k, (tr, sg) in enumerate(pool.imap_unordered(run_code_day, tasks), 1):
            trades += tr
            sigs += sg
            if k % 10 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} symbol-days, trades {len(trades)}", flush=True)
    d, sg = pd.DataFrame(trades), pd.DataFrame(sigs)
    d.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    sg.to_csv(out / "signals.csv", index=False, encoding="utf-8-sig")
    used = sorted(d["date"].unique()) if not d.empty else days
    (out / "report.md").write_text(report(d, sg, used, args.w), encoding="utf-8")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
