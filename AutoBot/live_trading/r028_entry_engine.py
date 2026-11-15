# -*- coding: utf-8 -*-
"""r028 entry engine (stage 3 of the entry-engine review, 2026-11-15) - entry_engine_analysis_20261115.md section 7.

Modules (each one is a class that can be unit-tested on its own; no broker import, no r003 import):
  BarBuilder      time bars (regime) + tick bars (trigger) from raw prints, plus the forming bar and running features
                  (session VWAP, RVOL, buy-share change, 30 s acceleration)
  SignalDetector  incremental BB(20) / MACD(12,26,9) / RSI(14) per bar series; BB-mid cross, MACD cross, RSI 50 cross,
                  SEQ / WIN / MA combinations; records T0 (first print of the forming bar at which the combination would
                  complete) and T1 (bar close)
  EntryFilter     time window, chase vs the first signal price, VWAP band, RSI / BB-upper overheat, min price, RVOL,
                  buy-share change, tick cost
  EdgeAIGate      optional model: may only VETO or SHRINK; timeout / error -> rules only
  EntryEngine     candidates from the trigger series, regime check on the regime series, validity time, one decision per
                  candidate, exit hints for the exit engine
  OrderExecutor   order plan: limit at the best ask / KIS best-limit / market, reprice and cancel schedule, one active
                  order per symbol (pure state machine - NOT wired to live orders in this version)
  LatencyLogger   T0 .. T5 per candidate / real buy into entry_latency_YYYYMMDD.jsonl
EntryShadowService (bottom) runs the engine on its own thread inside r003 when ENTRY_ENGINE_MODE="shadow": it reads the
g011 tick files of the watchlist, writes decisions to entry_engine_YYYYMMDD.jsonl and never places an order.

Parameters: r028_entry_engine.yaml (+ r001 ENTRY_ENGINE_CONFIG overrides).

Update log
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 매수 엔진 3단계(봉 생성, 신호, 필터, Edge AI 게이트, 결정, 주문 계획, 지연 기록) + r003 shadow 서비스.
      기본 구성은 g034 2단계 결과(3분 레짐 + 60/120틱 SEQ, VWAP 0~1.5 %, 09:05~14:00).
    impact: live (ENTRY_ENGINE_MODE="shadow": 기록만, 주문 없음) / sim (g035)
    compatibility: backward-compatible (기존 매수 경로 무변경)
"""
from __future__ import annotations

import bisect
import json
import math
import queue
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import yaml
except Exception:  # noqa: BLE001 - yaml missing -> built-in defaults only
    yaml = None

TAG = "  [ENTRY ENGINE]"
LAT_TAG = "  [ENTRY LAT   ]"
YAML_PATH = Path(__file__).with_name("r028_entry_engine.yaml")
SESSION_START = dt_time(9, 0)
SESSION_END = dt_time(15, 20)

DEFAULTS: dict = {
    "bars": {"regime_minutes": 3, "trigger_tick_sizes": [60, 120], "time_bar_delay_s": 5.0, "max_bars": 400,
             "warmup_prior_day": True},
    "signal": {"combo": "SEQ", "window_bars": 10, "bb_period": 20, "macd": [12, 26, 9], "rsi_period": 14, "rsi_level": 50.0,
               "intrabar": False, "intrabar_confirm_prints": 3, "regime_rising_bars": 3},
    "entry": {"valid_s": 60.0, "rearm_after_s": 0.0},
    "filter": {"start": "09:05:00", "end": "14:00:00", "chase_max_pct": 0.3, "vwap_gap_min_pct": 0.0,
               "vwap_gap_max_pct": 1.5, "rsi_max": 75.0, "bb_upper_gap_min_pct": 0.0, "min_price": 0, "rvol_min": 0.0,
               "buy_share_change_min": None, "max_tick_cost_pct": 0.0},
    "edge_ai": {"mode": "off", "timeout_s": 0.3, "min_prob": 0.5, "shrink_below_prob": 0.6, "shrink_factor": 0.5},
    "order": {"style": "limit_best_ask", "reprice_after_s": 3.0, "max_reprices": 2, "cancel_after_s": 10.0,
              "max_reprice_chase_pct": 0.3},
}
COMBOS = ("SEQ", "WIN", "MA")
ORDER_STYLES = ("limit_best_ask", "best_limit", "market")


# ---------------------------------------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------------------------------------
def _merge(base: dict, over: dict | None) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _hms(text: str) -> dt_time:
    return datetime.strptime(str(text), "%H:%M:%S").time()


@dataclass
class EntryParams:
    raw: dict

    @classmethod
    def load(cls, path: Path | None = YAML_PATH, overrides: dict | None = None) -> "EntryParams":
        data = deepcopy(DEFAULTS)
        if path is not None and Path(path).exists() and yaml is not None:
            with open(path, encoding="utf-8") as fh:
                data = _merge(data, yaml.safe_load(fh) or {})
        p = cls(_merge(data, overrides))
        p.validate()
        return p

    def __getitem__(self, key: str) -> dict:
        return self.raw[key]

    def validate(self) -> None:
        b, s, e, f, a, o = (self.raw[k] for k in ("bars", "signal", "entry", "filter", "edge_ai", "order"))
        if int(b["regime_minutes"]) not in (1, 3, 5):
            raise ValueError("bars.regime_minutes must be 1, 3 or 5")
        if not 0 <= float(b["time_bar_delay_s"]) < 60 * int(b["regime_minutes"]):
            # one pending bar at a time: a third bucket can only start after the pending bar's end + delay
            raise ValueError("bars.time_bar_delay_s must be >= 0 and shorter than one regime bar")
        sizes = [int(x) for x in b["trigger_tick_sizes"]]
        if not sizes or any(n < 10 for n in sizes):
            raise ValueError("bars.trigger_tick_sizes: at least one size >= 10")
        if str(s["combo"]).upper() not in COMBOS:
            raise ValueError(f"signal.combo must be one of {COMBOS}")
        if int(s["window_bars"]) < 1 or int(s["bb_period"]) < 5:
            raise ValueError("signal.window_bars >= 1, signal.bb_period >= 5")
        fast, slow, sig = (int(x) for x in s["macd"])
        if not 0 < fast < slow or sig < 1:
            raise ValueError("signal.macd = [fast < slow, signal >= 1]")
        if float(e["valid_s"]) <= 0:
            raise ValueError("entry.valid_s > 0")
        if _hms(f["start"]) >= _hms(f["end"]):
            raise ValueError("filter.start < filter.end")
        if float(f["vwap_gap_min_pct"]) > float(f["vwap_gap_max_pct"]):
            raise ValueError("filter.vwap_gap_min_pct <= vwap_gap_max_pct")
        if float(f["chase_max_pct"]) < 0:
            raise ValueError("filter.chase_max_pct >= 0")
        if str(a["mode"]).lower() not in ("off", "veto"):
            raise ValueError("edge_ai.mode: off | veto")
        if not 0 < float(a["shrink_factor"]) <= 1:
            raise ValueError("edge_ai.shrink_factor in (0, 1]")
        if str(o["style"]) not in ORDER_STYLES:
            raise ValueError(f"order.style must be one of {ORDER_STYLES}")
        if int(o["max_reprices"]) < 0 or float(o["cancel_after_s"]) <= 0:
            raise ValueError("order.max_reprices >= 0, order.cancel_after_s > 0")


# ---------------------------------------------------------------------------------------------------------------------
# tick size (KRX, same table as r024)
# ---------------------------------------------------------------------------------------------------------------------
def tick_size(price: float) -> int:
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if price < lim:
            return t
    return 1000


# ---------------------------------------------------------------------------------------------------------------------
# BarBuilder
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Bar:
    key: Any
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    n: int = 0
    first_recv: datetime | None = None
    avail: datetime | None = None      # when the bar may be used (tick bar: recv of its last print; time bar: end + delay)


@dataclass
class Print:
    """Minimal print (r009 Tick has more fields; anything with ts / recv / price / qty / side / acml_vol works)."""
    ts: datetime
    recv: datetime
    price: float
    qty: int
    side: int = 0
    acml_vol: int = 0
    ask1: float = 0.0
    bid1: float = 0.0


class BarSeries:
    """kind="tick" (n prints per bar, receive order) or kind="time" (minutes on the exchange clock from 09:00).

    A time bar is CONFIRMED only at end + delay (Codex review #2): a print of the next bucket moves it to `pending`, late
    prints of the pending bucket still count until then, and only advance(now >= end + delay) hands it to the indicators."""

    def __init__(self, kind: str, size: int, delay_s: float = 5.0, max_bars: int = 400):
        self.kind, self.size, self.delay_s = kind, int(size), float(delay_s)
        self.bars: deque[Bar] = deque(maxlen=max_bars)
        self.forming: Bar | None = None
        self.pending: Bar | None = None               # time bars: ended, waiting for end + delay

    def _time_key(self, ts: datetime) -> tuple:
        sec = (ts.hour * 3600 + ts.minute * 60 + ts.second) - 9 * 3600
        return (ts.date(), sec // (60 * self.size))

    def _time_end(self, key: tuple) -> datetime:
        return datetime.combine(key[0], SESSION_START) + timedelta(seconds=(key[1] + 1) * 60 * self.size)

    def _avail(self, b: Bar) -> datetime:
        return self._time_end(b.key) + timedelta(seconds=self.delay_s)

    @staticmethod
    def _put(b: Bar, p) -> None:
        b.high, b.low, b.close = max(b.high, p.price), min(b.low, p.price), p.price
        b.volume += int(p.qty or 0)
        b.n += 1

    def add(self, p, recv: datetime) -> list[Bar]:
        if self.kind == "tick":
            b = self.forming
            if b is None:
                b = self.forming = Bar(None, p.price, p.price, p.price, p.price, 0, 0, recv)
            self._put(b, p)
            if b.n >= self.size:
                b.avail = recv
                self.bars.append(b)
                self.forming = None
                return [b]
            return []
        closed = self.advance(recv)                    # anything whose confirmation time has passed goes first
        key = self._time_key(p.ts)
        if self.bars and key <= self.bars[-1].key:
            return closed                              # late print of a confirmed bar: dropped
        if self.pending is not None and key == self.pending.key:
            self._put(self.pending, p)                 # late print inside the confirmation delay: still counts
            return closed
        if self.forming is not None and key != self.forming.key:
            if key < self.forming.key:
                return closed                          # older than the forming bar and not pending: dropped
            if self.pending is not None:               # two buckets ahead: the old pending one is final now
                closed.append(self._confirm(self.pending))
                self.pending = None
            self.pending, self.forming = self.forming, None
        if self.forming is None:
            self.forming = Bar(key, p.price, p.price, p.price, p.price, 0, 0, recv)
        self._put(self.forming, p)
        return closed

    def _confirm(self, b: Bar) -> Bar:
        b.avail = self._avail(b)
        self.bars.append(b)
        return b

    def advance(self, now: datetime) -> list[Bar]:
        """Time bars are confirmed by the clock (end + delay) even without a later print."""
        out: list[Bar] = []
        if self.kind != "time":
            return out
        if self.pending is not None and now >= self._avail(self.pending):
            out.append(self._confirm(self.pending))
            self.pending = None
        if self.forming is not None and now >= self._avail(self.forming):
            out.append(self._confirm(self.forming))
            self.forming = None
        return out

    def drop_unfinished(self) -> None:
        """After the prior-day warm-up: a half-built bar of another session must not be completed by today's prints."""
        self.forming = None
        self.pending = None


class BarBuilder:
    """All series of one symbol + running intraday features. warm=True prints (prior day) feed the bars only."""

    def __init__(self, params: EntryParams):
        b = params["bars"]
        self.series: dict[str, BarSeries] = {"regime": BarSeries("time", int(b["regime_minutes"]), float(b["time_bar_delay_s"]),
                                                                 int(b["max_bars"]))}
        for n in b["trigger_tick_sizes"]:
            self.series[f"t{int(n)}"] = BarSeries("tick", int(n), max_bars=int(b["max_bars"]))
        self.last_acml: int | None = None
        self.last_print = None
        self.last_recv: datetime | None = None
        # running session features (today only)
        self._rt: list[float] = []
        self._cq: list[float] = [0.0]
        self._cb: list[float] = [0.0]
        self._cs: list[float] = [0.0]
        self._cpq: list[float] = [0.0]
        self._px: list[float] = []
        self._next_trim = self.TRIM_AT

    def ingest(self, p, warm: bool = False) -> dict[str, list[Bar]]:
        """-> closed bars per series. Duplicate prints (same or lower cumulative volume) are dropped."""
        acml = int(getattr(p, "acml_vol", 0) or 0)
        if acml > 0 and self.last_acml is not None and acml <= self.last_acml and not warm:
            return {}
        if p.ts.time() < SESSION_START or p.ts.time() >= SESSION_END or p.price <= 0:
            return {}
        recv = p.recv or p.ts
        if not warm:
            if acml > 0:
                self.last_acml = acml
            self.last_print, self.last_recv = p, recv
            q = float(p.qty or 0)
            self._rt.append(recv.timestamp())
            self._px.append(float(p.price))
            self._cq.append(self._cq[-1] + q)
            self._cb.append(self._cb[-1] + (q if p.side == 1 else 0.0))
            self._cs.append(self._cs[-1] + (q if p.side == 5 else 0.0))
            self._cpq.append(self._cpq[-1] + q * p.price)
            if len(self._rt) > self._next_trim:
                self._trim(self._rt[-1] - self.KEEP_S)
                self._next_trim = len(self._rt) + self.TRIM_AT   # amortised: at most one compaction per TRIM_AT prints
        return {name: s.add(p, recv) for name, s in self.series.items()}

    KEEP_S = 2400.0      # features look back at most 1980 s (RVOL); older prints are dropped (memory on the Pi)
    TRIM_AT = 20000

    def _trim(self, cutoff: float) -> None:
        """Drops prints received before cutoff. The cumulative lists keep their absolute values (VWAP stays a session
        VWAP) - only the prefix goes."""
        k = bisect.bisect_left(self._rt, cutoff)
        if k <= 0:
            return
        del self._rt[:k]
        del self._px[:k]
        for lst in (self._cq, self._cb, self._cs, self._cpq):
            del lst[:k]                                          # element 0 is the cumulative value before _rt[0]

    def advance(self, now: datetime) -> dict[str, list[Bar]]:
        return {name: s.advance(now) for name, s in self.series.items()}

    # -- features (no look-ahead: only prints received at or before t) ----------------------------------------------
    def _i(self, t: float) -> int:
        return bisect.bisect_right(self._rt, t)

    def price_at(self, t: float) -> float:
        i = self._i(t)
        return self._px[i - 1] if i else float("nan")

    def features(self, now: datetime) -> dict:
        t = now.timestamp()
        i = self._i(t)
        if i == 0:
            return {}
        open_t = datetime.combine(now.date(), SESSION_START).timestamp()
        i3, i33 = self._i(t - 180), self._i(t - 1980)
        v3 = self._cq[i] - self._cq[i3]
        span = min(1800.0, t - 180 - open_t)
        vh = self._cq[i3] - self._cq[i33]
        rvol = v3 / (vh / (span / 180.0)) if span >= 360 and vh > 0 else float("nan")
        i1, i6 = self._i(t - 60), self._i(t - 360)
        b1, s1 = self._cb[i] - self._cb[i1], self._cs[i] - self._cs[i1]
        b0, s0 = self._cb[i1] - self._cb[i6], self._cs[i1] - self._cs[i6]
        sh1 = b1 / (b1 + s1) if b1 + s1 > 0 else float("nan")
        sh0 = b0 / (b0 + s0) if b0 + s0 > 0 else float("nan")
        vwap = self._cpq[i] / self._cq[i] if self._cq[i] > 0 else float("nan")
        p, p30, p60 = self._px[i - 1], self.price_at(t - 30), self.price_at(t - 60)
        r1 = (p / p30 - 1) * 100 if p30 == p30 and p30 > 0 else float("nan")
        r0 = (p30 / p60 - 1) * 100 if p60 == p60 and p60 > 0 else float("nan")
        return {"price": p, "rvol": rvol, "dshare": sh1 - sh0, "vwap": vwap,
                "vwap_gap": (p / vwap - 1) * 100 if vwap == vwap and vwap > 0 else float("nan"), "accel": r1 - r0}


# ---------------------------------------------------------------------------------------------------------------------
# SignalDetector
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class _Ind:
    """Incremental indicator state after the last closed bar (pandas rolling / ewm(adjust=False) / Wilder RSI)."""
    n: int = 0
    closes: deque = field(default_factory=lambda: deque(maxlen=64))
    mids: deque = field(default_factory=lambda: deque(maxlen=64))
    ema_f: float = float("nan")
    ema_s: float = float("nan")
    sig: float = float("nan")
    up: float = float("nan")
    dn: float = float("nan")
    prev_close: float = float("nan")


@dataclass
class SeriesSignal:
    series: str
    fired: bool
    t1: datetime | None                 # bar close (available)
    p1: float
    t0: datetime | None = None          # first print of that bar at which the combination would already complete
    p0: float = float("nan")
    rsi: float = float("nan")
    mid: float = float("nan")
    upper: float = float("nan")
    hist: float = float("nan")
    intrabar: bool = False


class SignalDetector:
    """One per (symbol, series). update() on every closed bar; peek() on every print of the forming bar (T0)."""

    def __init__(self, series: str, params: EntryParams):
        s = params["signal"]
        self.series = series
        self.combo = str(s["combo"]).upper()
        self.w = int(s["window_bars"])
        self.period = int(s["bb_period"])
        self.fast, self.slow, self.sigp = (int(x) for x in s["macd"])
        self.rsi_p = int(s["rsi_period"])
        self.rsi_level = float(s["rsi_level"])
        self.rising = int(s["regime_rising_bars"])
        self.ind = _Ind()
        self.k = -1                                   # index of the last closed bar
        self.stage, self.stage_k = 0, -1              # SEQ state machine
        self.last_b = self.last_m = self.last_r = -10 ** 9
        self.win_prev_ok = False
        self.last = {}                                # values of the last closed bar
        self.t0: datetime | None = None               # T0 of the forming bar
        self.p0 = float("nan")

    # -- indicator math ---------------------------------------------------------------------------------------------
    def _next(self, close: float) -> dict:
        """Indicator values if `close` were the next bar close (does not mutate)."""
        st = self.ind
        n = st.n + 1
        closes = list(st.closes)[-(self.period - 1):] + [close]
        mid = sum(closes) / self.period if n >= self.period and len(closes) == self.period else float("nan")
        c5 = (list(st.closes)[-4:] + [close])
        ma5 = sum(c5) / 5 if n >= 5 else float("nan")
        af, as_ = 2 / (self.fast + 1), 2 / (self.slow + 1)
        ema_f = close if st.n == 0 else st.ema_f + af * (close - st.ema_f)
        ema_s = close if st.n == 0 else st.ema_s + as_ * (close - st.ema_s)
        macd = ema_f - ema_s
        asig = 2 / (self.sigp + 1)
        sig = macd if st.n == 0 else st.sig + asig * (macd - st.sig)
        hist = macd - sig if n > self.slow + self.sigp - 2 else float("nan")
        up, dn = st.up, st.dn
        if st.n >= 1:
            d = close - st.prev_close
            g, l = max(d, 0.0), max(-d, 0.0)
            a = 1.0 / self.rsi_p
            up = g if st.n == 1 else up + a * (g - up)
            dn = l if st.n == 1 else dn + a * (l - dn)
        rsi = float("nan")
        if n > self.rsi_p and up == up and dn == dn:
            rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
        var = sum((c - mid) ** 2 for c in closes) / self.period if mid == mid else float("nan")
        upper = mid + 2 * math.sqrt(var) if var == var else float("nan")
        return {"close": close, "mid": mid, "ma5": ma5, "ema_f": ema_f, "ema_s": ema_s, "sig": sig, "hist": hist, "up": up,
                "dn": dn, "rsi": rsi, "upper": upper}

    @staticmethod
    def _cross(now: float, prev: float, level: float = 0.0, prev_level: float | None = None) -> bool:
        pl = level if prev_level is None else prev_level
        return now == now and prev == prev and pl == pl and level == level and now > level and prev <= pl

    def _flags(self, v: dict) -> dict:
        last = self.last
        bbx = self._cross(v["close"], last.get("close", float("nan")), v["mid"], last.get("mid", float("nan")))
        macdx = self._cross(v["hist"], last.get("hist", float("nan")))
        rsix = self._cross(v["rsi"], last.get("rsi", float("nan")), self.rsi_level)
        max_ = self._cross(v["ma5"], last.get("ma5", float("nan")), v["mid"], last.get("mid", float("nan")))
        return {"bbx": bbx, "macdx": macdx, "rsix": rsix, "max": max_, "above": v["close"] > v["mid"] if v["mid"] == v["mid"] else False,
                "hpos": v["hist"] > 0 if v["hist"] == v["hist"] else False, "rpos": v["rsi"] > self.rsi_level if v["rsi"] == v["rsi"] else False}

    def _combo(self, f: dict, k: int, commit: bool) -> bool:
        """SEQ / WIN / MA fire at bar k. commit=False evaluates a provisional (forming) bar without changing state."""
        if self.combo == "MA":
            return f["max"]
        if self.combo == "WIN":
            lb = k if f["bbx"] else self.last_b
            lm = k if f["macdx"] else self.last_m
            lr = k if f["rsix"] else self.last_r
            ok = bool(f["above"] and f["hpos"] and f["rpos"] and k - lb < self.w and k - lm < self.w and k - lr < self.w)
            fire = ok and not self.win_prev_ok
            if commit:
                self.last_b, self.last_m, self.last_r, self.win_prev_ok = lb, lm, lr, ok
            return fire
        stage, sk = self.stage, self.stage_k                         # SEQ (same order of checks as g034)
        fire = False
        if not f["above"]:
            stage = 0
        if stage == 0 and f["bbx"]:
            stage, sk = 1, k
        if stage == 1 and f["macdx"] and k - sk <= self.w:
            stage, sk = 2, k
        elif stage == 1 and k - sk > self.w:
            stage = 0
        if stage == 2 and f["rsix"] and k - sk <= self.w and f["above"]:
            fire, stage = True, 0
        elif stage == 2 and k - sk > self.w:
            stage = 0
        if commit:
            self.stage, self.stage_k = stage, sk
        return fire

    # -- public -----------------------------------------------------------------------------------------------------
    def peek(self, price: float, recv: datetime) -> bool:
        """Forming bar with `price` as its provisional close: would the combination complete? Records T0 (first such
        print of this bar)."""
        if self.ind.n < self.period:
            return False
        v = self._next(price)
        fire = self._combo(self._flags(v), self.k + 1, commit=False)
        if fire and self.t0 is None:
            self.t0, self.p0 = recv, price
        return fire

    def update(self, bar: Bar) -> SeriesSignal:
        v = self._next(bar.close)
        f = self._flags(v)
        self.k += 1
        fire = self._combo(f, self.k, commit=True)
        st = self.ind
        st.n += 1
        st.closes.append(bar.close)
        st.mids.append(v["mid"])
        st.ema_f, st.ema_s, st.sig, st.up, st.dn, st.prev_close = v["ema_f"], v["ema_s"], v["sig"], v["up"], v["dn"], bar.close
        self.last = {**v, **f}
        out = SeriesSignal(self.series, fire, bar.avail, bar.close, self.t0 if fire else None, self.p0 if fire else float("nan"),
                           v["rsi"], v["mid"], v["upper"], v["hist"])
        self.t0, self.p0 = None, float("nan")
        return out

    def regime_ok(self) -> bool:
        """Close above MA20 and MA20 above its value `rising` bars ago (last closed bar)."""
        mids = list(self.ind.mids)
        if len(mids) <= self.rising or not self.last:
            return False
        m, m_old = mids[-1], mids[-1 - self.rising]
        return bool(self.last.get("above") and m == m and m_old == m_old and m > m_old)


# ---------------------------------------------------------------------------------------------------------------------
# EntryFilter
# ---------------------------------------------------------------------------------------------------------------------
class EntryFilter:
    def __init__(self, params: EntryParams):
        f = params["filter"]
        self.start, self.end = _hms(f["start"]), _hms(f["end"])
        self.chase = float(f["chase_max_pct"])
        self.vmin, self.vmax = float(f["vwap_gap_min_pct"]), float(f["vwap_gap_max_pct"])
        self.rsi_max = float(f["rsi_max"]) if f.get("rsi_max") is not None else None
        self.bb_upper_gap = float(f.get("bb_upper_gap_min_pct") or 0.0)
        self.min_price = float(f.get("min_price") or 0)
        self.rvol_min = float(f.get("rvol_min") or 0.0)
        self.share_min = f.get("buy_share_change_min")
        self.tick_cost = float(f.get("max_tick_cost_pct") or 0.0)

    def check(self, now: datetime, price: float, ref_price: float, sig: SeriesSignal, feat: dict) -> tuple[bool, str]:
        """(ok, reason). A NaN feature never blocks (too early in the day / no data) except the price itself."""
        if not (self.start <= now.time() < self.end):
            return False, f"TIME_{now:%H:%M:%S}_OUTSIDE_{self.start:%H:%M}-{self.end:%H:%M}"
        if not (price == price and price > 0):
            return False, "NO_PRICE"
        if self.min_price > 0 and price < self.min_price:
            return False, f"PRICE_{price:,.0f}_LT_{self.min_price:,.0f}"
        if ref_price == ref_price and ref_price > 0:
            chase = (price / ref_price - 1) * 100
            if chase > self.chase:
                return False, f"CHASE_{chase:.2f}%_GT_{self.chase:.2f}%"
        gap = feat.get("vwap_gap", float("nan"))
        if gap == gap and not (self.vmin < gap <= self.vmax):
            return False, f"VWAP_GAP_{gap:+.2f}%_OUTSIDE_({self.vmin:+.2f},{self.vmax:+.2f}]"
        if self.rsi_max is not None and sig.rsi == sig.rsi and sig.rsi >= self.rsi_max:
            return False, f"OVERHEAT_RSI_{sig.rsi:.1f}_GE_{self.rsi_max:.0f}"
        if self.bb_upper_gap > 0 and sig.upper == sig.upper and price >= sig.upper * (1 - self.bb_upper_gap / 100):
            return False, f"OVERHEAT_NEAR_BB_UPPER_{(sig.upper / price - 1) * 100:.2f}%"
        rv = feat.get("rvol", float("nan"))
        if self.rvol_min > 0 and rv == rv and rv < self.rvol_min:
            return False, f"RVOL_{rv:.2f}_LT_{self.rvol_min:.2f}"
        ds = feat.get("dshare", float("nan"))
        if self.share_min is not None and ds == ds and ds < float(self.share_min):
            return False, f"BUY_SHARE_CHANGE_{ds:+.2f}_LT_{float(self.share_min):+.2f}"
        if self.tick_cost > 0 and tick_size(price) / price * 100 > self.tick_cost:
            return False, f"TICK_COST_{tick_size(price) / price * 100:.2f}%"
        return True, "OK"


# ---------------------------------------------------------------------------------------------------------------------
# EdgeAIGate
# ---------------------------------------------------------------------------------------------------------------------
class EdgeAIGate:
    """predict(features) -> probability of a good entry. The gate can only refuse or shrink; a timeout or an error falls
    back to the rules (allow, factor 1.0). mode="off" -> always (True, 1.0)."""

    BREAKER_TIMEOUTS = 3       # consecutive timeouts -> the model is skipped for BREAKER_S (Codex review #7)
    BREAKER_S = 60.0

    def __init__(self, params: EntryParams, predict: Callable[[dict], float] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        a = params["edge_ai"]
        self.mode = str(a["mode"]).lower()
        self.timeout_s = float(a["timeout_s"])
        self.min_prob, self.shrink_below, self.factor = float(a["min_prob"]), float(a["shrink_below_prob"]), float(a["shrink_factor"])
        self.predict = predict
        self.clock = clock
        self._busy: threading.Thread | None = None     # at most ONE inference in flight (a hung model is never stacked)
        self._timeouts = 0
        self._open_until = 0.0

    def check(self, features: dict) -> tuple[bool, float, str]:
        if self.mode == "off" or self.predict is None:
            return True, 1.0, "AI_OFF"
        if self.clock() < self._open_until:
            return True, 1.0, "AI_BREAKER_RULES_ONLY"
        if self._busy is not None and self._busy.is_alive():
            return True, 1.0, "AI_BUSY_RULES_ONLY"
        box: dict = {}

        def run() -> None:
            try:
                box["prob"] = float(self.predict(dict(features)))
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=run, name="entry-ai", daemon=True)   # daemon: a hung model never blocks exit
        th.start()
        th.join(self.timeout_s)
        if th.is_alive():
            self._busy = th
            self._timeouts += 1
            if self._timeouts >= self.BREAKER_TIMEOUTS:
                self._open_until = self.clock() + self.BREAKER_S
                self._timeouts = 0
            return True, 1.0, "AI_TIMEOUT_RULES_ONLY"
        self._busy = None
        self._timeouts = 0
        if "err" in box:
            return True, 1.0, f"AI_ERROR_RULES_ONLY_{type(box['err']).__name__}"
        prob = box.get("prob", float("nan"))
        if not prob == prob:
            return True, 1.0, "AI_NAN_RULES_ONLY"
        if prob < self.min_prob:
            return False, 0.0, f"AI_VETO_{prob:.2f}_LT_{self.min_prob:.2f}"
        if prob < self.shrink_below:
            return True, self.factor, f"AI_SHRINK_{prob:.2f}"
        return True, 1.0, f"AI_OK_{prob:.2f}"


# ---------------------------------------------------------------------------------------------------------------------
# EntryEngine
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    code: str
    series: str
    t1: datetime
    p1: float
    t0: datetime | None
    p0: float
    sig: SeriesSignal
    expires: datetime
    last_reason: str = ""
    rejects: int = 0


@dataclass
class EntryDecision:
    code: str
    action: str                     # BUY | EXPIRED
    at: datetime
    price: float
    reason: str
    series: str
    t0: datetime | None
    p0: float
    t1: datetime
    p1: float
    qty_factor: float = 1.0
    exit_hints: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("at", "t0", "t1"):
            d[k] = d[k].isoformat(timespec="milliseconds") if d[k] else None
        return {k: (round(v, 5) if isinstance(v, float) and v == v else (None if isinstance(v, float) else v)) for k, v in d.items()}


class _CodeState:
    def __init__(self, code: str, params: EntryParams):
        self.code = code
        self.builder = BarBuilder(params)
        self.det = {name: SignalDetector(name, params) for name in self.builder.series}
        self.cand: Candidate | None = None
        self.last_decision_at: datetime | None = None
        self.intrabar_hits: dict[str, int] = {}
        self.intrabar_fired: set[tuple[str, int]] = set()


class EntryEngine:
    """Feed prints with on_print (or on_prints), call evaluate(now) to get decisions. Pure: no I/O."""

    def __init__(self, params: EntryParams, ai: EdgeAIGate | None = None):
        self.params = params
        self.filter = EntryFilter(params)
        self.ai = ai or EdgeAIGate(params)
        self.valid = timedelta(seconds=float(params["entry"]["valid_s"]))
        self.rearm = timedelta(seconds=float(params["entry"]["rearm_after_s"]))
        self.intrabar = bool(params["signal"]["intrabar"])
        self.confirm_prints = int(params["signal"]["intrabar_confirm_prints"])
        self.codes: dict[str, _CodeState] = {}

    def state(self, code: str) -> _CodeState:
        st = self.codes.get(code)
        if st is None:
            st = self.codes[code] = _CodeState(code, self.params)
        return st

    def warm(self, code: str, prints: Iterable) -> None:
        st = self.state(code)
        for p in prints:
            for name, bars in st.builder.ingest(p, warm=True).items():
                for b in bars:
                    st.det[name].update(b)
        for name, s_ in st.builder.series.items():
            if s_.kind == "time":                       # the prior session is over: its last time bars are final
                for b in s_.advance(datetime(9999, 1, 1)):
                    st.det[name].update(b)
            s_.drop_unfinished()                        # Codex review #6: a half tick bar never mixes two sessions

    def on_print(self, code: str, p) -> None:
        st = self.state(code)
        recv = p.recv or p.ts
        self._advance(st, recv)
        closed = st.builder.ingest(p)
        if not closed:
            return                                                  # duplicate / outside the session: ignored
        for name, bars in closed.items():
            for b in bars:
                self._on_bar(st, name, b)
        # T0 is always recorded (LatencyLogger); with signal.intrabar the forming-bar signal also becomes a candidate
        for name, s in st.builder.series.items():
            if name == "regime" or s.forming is None:
                continue
            det = st.det[name]
            fire = det.peek(p.price, recv)
            if not self.intrabar:
                continue
            if not fire:
                st.intrabar_hits[name] = 0
                continue
            hits = st.intrabar_hits.get(name, 0) + 1
            st.intrabar_hits[name] = hits
            key = (name, det.k + 1)
            if hits >= self.confirm_prints and key not in st.intrabar_fired:
                st.intrabar_fired.add(key)
                sig = SeriesSignal(name, True, recv, p.price, det.t0, det.p0, det.last.get("rsi", float("nan")),
                                   det.last.get("mid", float("nan")), det.last.get("upper", float("nan")),
                                   det.last.get("hist", float("nan")), intrabar=True)
                self._candidate(st, sig)

    def on_prints(self, code: str, prints: Iterable) -> None:
        for p in prints:
            self.on_print(code, p)

    def _advance(self, st: _CodeState, now: datetime) -> None:
        for name, bars in st.builder.advance(now).items():
            for b in bars:
                self._on_bar(st, name, b)

    def _on_bar(self, st: _CodeState, name: str, bar: Bar) -> None:
        sig = st.det[name].update(bar)
        st.intrabar_hits[name] = 0
        if name == "regime" or not sig.fired:
            return
        if self.intrabar and (name, st.det[name].k) in st.intrabar_fired:
            return                                                  # already a candidate from the forming bar
        self._candidate(st, sig)

    def _candidate(self, st: _CodeState, sig: SeriesSignal) -> None:
        if st.cand is not None and st.cand.expires > sig.t1:
            return                                                  # the earliest valid signal stays the reference
        st.cand = Candidate(st.code, sig.series, sig.t1, sig.p1, sig.t0, sig.p0, sig, sig.t1 + self.valid)

    def evaluate(self, now: datetime, code: str | None = None, blocked: Callable[[str], str | None] | None = None
                 ) -> list[EntryDecision]:
        """Decisions at `now` for one code (or all). blocked(code) -> reason or None (position / pending order held)."""
        out = []
        if code is None:
            states = list(self.codes.values())
        else:
            states = [self.codes[code]] if code in self.codes else []
        for st in states:
            self._advance(st, now)
            c = st.cand
            if c is None:
                continue
            if now > c.expires:
                out.append(EntryDecision(st.code, "EXPIRED", now, st.builder.price_at(now.timestamp()), c.last_reason or "NO_EVAL",
                                         c.series, c.t0, c.p0, c.t1, c.p1))
                st.cand = None
                continue
            reason = blocked(st.code) if blocked else None
            if reason:
                c.last_reason, c.rejects = reason, c.rejects + 1
                continue
            if st.last_decision_at is not None and now - st.last_decision_at < self.rearm:
                c.last_reason = "REARM_WAIT"
                continue
            reg = st.det["regime"]
            if not reg.regime_ok():
                c.last_reason, c.rejects = "REGIME_NOT_UP", c.rejects + 1
                continue
            feat = st.builder.features(now)
            price = feat.get("price", float("nan"))
            ref = c.p0 if c.p0 == c.p0 else c.p1
            ok, why = self.filter.check(now, price, ref, c.sig, feat)
            if not ok:
                c.last_reason, c.rejects = why, c.rejects + 1
                continue
            allow, factor, ai_reason = self.ai.check({**feat, "rsi": c.sig.rsi, "series": c.series,
                                                      "chase_pct": (price / ref - 1) * 100})
            if not allow:
                c.last_reason, c.rejects = ai_reason, c.rejects + 1
                continue
            hints = {"regime": "up", "signal_series": c.series, "chase_pct": round((price / ref - 1) * 100, 4),
                     "signal_rsi": round(c.sig.rsi, 2) if c.sig.rsi == c.sig.rsi else None,
                     "vwap_gap": round(feat.get("vwap_gap", float("nan")), 4) if feat.get("vwap_gap") == feat.get("vwap_gap") else None}
            out.append(EntryDecision(st.code, "BUY", now, price, f"{c.series.upper()}_{self.params['signal']['combo']}|{ai_reason}",
                                     c.series, c.t0, c.p0, c.t1, c.p1, factor, hints,
                                     {k: (round(v, 5) if isinstance(v, float) and v == v else None) for k, v in feat.items()}))
            st.cand = None
            st.last_decision_at = now
        return out


# ---------------------------------------------------------------------------------------------------------------------
# OrderExecutor (order plan; the live bot keeps its own order path in this version)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class OrderPlan:
    code: str
    qty: int
    style: str
    price: float
    submitted_at: datetime
    ref_price: float
    reprices: int = 0
    filled: int = 0
    state: str = "ACTIVE"           # ACTIVE | FILLED | CANCELLED


class OrderExecutor:
    def __init__(self, params: EntryParams):
        o = params["order"]
        self.style = str(o["style"])
        self.reprice_after = timedelta(seconds=float(o["reprice_after_s"]))
        self.max_reprices = int(o["max_reprices"])
        self.cancel_after = timedelta(seconds=float(o["cancel_after_s"]))
        self.max_chase = float(o["max_reprice_chase_pct"])
        self.active: dict[str, OrderPlan] = {}

    def submit(self, code: str, qty: int, now: datetime, best_ask: float, ref_price: float) -> OrderPlan | None:
        """None when an order of this symbol is still active (duplicate guard) or qty <= 0."""
        a = self.active.get(code)
        if a is not None and a.state == "ACTIVE":
            return None
        if qty <= 0:
            return None
        price = float(best_ask) if self.style != "market" else float("nan")
        plan = OrderPlan(code, int(qty), self.style, price, now, float(ref_price))
        self.active[code] = plan
        return plan

    def on_fill(self, code: str, qty: int) -> None:
        a = self.active.get(code)
        if a is not None and a.state == "ACTIVE":
            a.filled = min(a.qty, a.filled + int(qty))
            if a.filled >= a.qty:
                a.state = "FILLED"

    def step(self, code: str, now: datetime, best_ask: float) -> tuple[str, float]:
        """-> ("NONE" | "REPRICE" | "CANCEL", price). Reprice to the new best ask at most max_reprices times and never
        above ref_price x (1 + max_reprice_chase_pct); cancel the remainder after cancel_after_s."""
        a = self.active.get(code)
        if a is None or a.state != "ACTIVE" or a.style == "market":
            return "NONE", float("nan")
        age = now - a.submitted_at
        if age >= self.cancel_after:
            a.state = "CANCELLED"
            return "CANCEL", float("nan")
        due = a.submitted_at + self.reprice_after * (a.reprices + 1)
        if now >= due and a.reprices < self.max_reprices and best_ask == best_ask and best_ask > a.price:
            cap = a.ref_price * (1 + self.max_chase / 100) if a.ref_price > 0 else float("inf")
            if best_ask <= cap:
                a.reprices += 1
                a.price = float(best_ask)
                return "REPRICE", a.price
        return "NONE", float("nan")


# ---------------------------------------------------------------------------------------------------------------------
# LatencyLogger
# ---------------------------------------------------------------------------------------------------------------------
STAGES = ("t0", "t1", "t_decision", "t_submit", "t_fill")


def _iso(at: datetime) -> str:
    return at.isoformat(timespec="milliseconds")


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if f == f else None


class LatencyLogger:
    """Two record kinds (Codex review #5 - an engine candidate never ends or rewrites a real order record):
      engine  T0 (forming-bar signal) / T1 (bar close) / decision of one r028 candidate -> SHADOW_BUY | EXPIRED_* | ...
      order   a REAL buy: T4 submit / T5 fill. At submit it copies the symbol's latest engine record when that record's T1
              is not after the submit and at most ATTACH_S older (time order checked), so one row holds T0..T5.
    Rows go to a jsonl file (path) and, with keep_rows=True, to self.rows (tests / g035)."""

    ATTACH_S = 600.0

    def __init__(self, path: Path | None, log: Callable[[str], None] | None = None, keep_rows: bool = True):
        self.path = Path(path) if path else None
        self.log = log or (lambda _m: None)
        self.keep_rows = keep_rows
        self.engine_open: dict[str, dict] = {}
        self.order_open: dict[str, dict] = {}
        self.recent: dict[str, dict] = {}          # last finished engine record per symbol (for attaching)
        self.rows: list[dict] = []

    # -- engine side ------------------------------------------------------------------------------------------------
    def engine_start(self, code: str, t0: datetime | None, p0: float, t1: datetime, p1: float, **extra) -> None:
        self.engine_end(code, "SUPERSEDED")
        rec = {"kind": "engine", "code": code, "t1": _iso(t1), "p1": _num(p1), **extra}
        if t0 is not None:
            rec.update({"t0": _iso(t0), "p0": _num(p0)})
        self.engine_open[code] = rec

    def engine_decision(self, code: str, at: datetime, price: float, reason: str = "") -> None:
        rec = self.engine_open.get(code)
        if rec is None:
            return
        rec.update({"t_decision": _iso(at), "p_decision": _num(price), "decision": reason})
        self.engine_end(code, "SHADOW_BUY")

    def engine_end(self, code: str, outcome: str) -> None:
        rec = self.engine_open.pop(code, None)
        if rec is not None:
            self.recent[code] = dict(rec)
            self._flush(rec, outcome)

    # -- real order side --------------------------------------------------------------------------------------------
    def order_submit(self, code: str, at: datetime, price: float, reason: str = "") -> None:
        old = self.order_open.pop(code, None)
        if old is not None:
            self._flush(old, "NO_FILL")
        rec = {"kind": "order", "code": code, "t_submit": _iso(at), "p_submit": _num(price), "live_reason": reason}
        eng = self.engine_open.get(code) or self.recent.get(code)
        if eng and eng.get("t1"):
            age = (at - datetime.fromisoformat(eng["t1"])).total_seconds()
            if 0.0 <= age <= self.ATTACH_S:
                for k in ("t0", "p0", "t1", "p1", "series"):
                    if k in eng:
                        rec[k] = eng[k]
                # Codex re-review: a decision logged after the real submit (worker step vs queue order) is not copied
                if eng.get("t_decision") and datetime.fromisoformat(eng["t_decision"]) <= at:
                    for k in ("t_decision", "p_decision", "decision"):
                        rec[k] = eng[k]
                rec["engine_attached"] = True
        self.order_open[code] = rec

    def order_fill(self, code: str, at: datetime, price: float) -> None:
        rec = self.order_open.pop(code, None)
        if rec is None:
            rec = {"kind": "order", "code": code}
        rec.update({"t_fill": _iso(at), "p_fill": _num(price)})
        self._flush(rec, "FILLED")

    def close(self, outcome: str) -> None:
        """Session stop / date change (Codex review #8): every open record is written with this outcome."""
        for code in list(self.engine_open):
            self.engine_end(code, outcome)
        for code in list(self.order_open):
            self._flush(self.order_open.pop(code), outcome)

    # -- output -----------------------------------------------------------------------------------------------------
    def _flush(self, rec: dict, outcome: str) -> None:
        rec["outcome"] = outcome
        times = {s_: datetime.fromisoformat(rec[s_]) for s_ in STAGES if rec.get(s_)}
        order = [s_ for s_ in STAGES if s_ in times]
        for a, b in zip(order, order[1:]):
            rec[f"{a}->{b}_s"] = round((times[b] - times[a]).total_seconds(), 3)
        p_first = next((rec.get(k) for k in ("p0", "p1", "p_decision", "p_submit") if rec.get(k)), None)
        if p_first and rec.get("p_fill"):
            rec["first_to_fill_pct"] = round((rec["p_fill"] / p_first - 1) * 100, 4)
        if self.keep_rows:
            self.rows.append(rec)
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as exc:
                self.log(f"{LAT_TAG} write failed: {exc}")
        if rec["kind"] == "order" and outcome == "FILLED":
            seg = " ".join(f"{k[:-2]}={v:.1f}s" for k, v in rec.items() if k.endswith("_s"))
            self.log(f"{LAT_TAG} {rec['code']} | {seg} | first->fill={rec.get('first_to_fill_pct')}%")


def make_file_logger(path: Path) -> Callable[[str], None]:
    """Worker-thread log (Codex review #1): its own file, never the r003 logger (no shared handler lock with trading)."""
    path = Path(path)

    def write(msg: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
        except OSError:
            pass
    return write


# ---------------------------------------------------------------------------------------------------------------------
# shadow service for r003 (worker thread; reads tick files; never orders)
# ---------------------------------------------------------------------------------------------------------------------
class EntryShadowRunner:
    def __init__(self, tick_dir: Path, out_dir: Path, date_str: str, params: EntryParams, log: Callable[[str], None]):
        from r026_exit_shadow import TickFileTail   # lazy: r026 imports r024/r025 (pure modules)
        self._Tail = TickFileTail
        self.tick_dir, self.out_dir, self.date_str = Path(tick_dir), Path(out_dir), date_str
        self.params, self.log = params, log
        self.engine = EntryEngine(params)
        self.tails: dict[str, Any] = {}
        self.latency = LatencyLogger(self.out_dir / f"entry_latency_{date_str}.jsonl", log, keep_rows=False)
        self.decisions_path = self.out_dir / f"entry_engine_{date_str}.jsonl"
        self.watch: set[str] = set()
        self.blocked: dict[str, str] = {}
        self.decisions = 0
        self.errors = 0
        self._last_cand: dict[str, tuple] = {}

    def set_watch(self, codes: Iterable[str], blocked: dict[str, str] | None = None) -> None:
        """Only watched symbols are fed AND evaluated. A symbol that leaves the list loses its state (Codex review #4);
        if it comes back it is warmed up again from the files."""
        new = {str(c).zfill(6) for c in codes}
        for code in set(self.engine.codes) | set(self.tails):
            if code not in new:
                self.engine.codes.pop(code, None)
                self.tails.pop(code, None)
                self._last_cand.pop(code, None)
                self.latency.engine_end(code, "UNWATCHED")
        self.watch = new
        self.blocked = {str(k).zfill(6): v for k, v in (blocked or {}).items()}

    def _open(self, code: str) -> None:
        if self.params["bars"]["warmup_prior_day"]:
            try:
                from r013_tick_trigger import find_prior_tick_file, iter_prior_tail
                prior = find_prior_tick_file(self.tick_dir, self.date_str, code)
                if prior is not None:
                    self.engine.warm(code, iter_prior_tail(prior, 20000))
            except Exception as exc:  # noqa: BLE001
                self.log(f"{TAG} {code} | prior-day warm-up failed: {exc}")
        self.tails[code] = self._Tail(self.tick_dir / self.date_str / f"{code}.csv")

    def step(self, now: datetime) -> list[EntryDecision]:
        out: list[EntryDecision] = []
        for code in sorted(self.watch):
            try:
                if code not in self.tails:
                    self._open(code)
                self.engine.on_prints(code, self.tails[code].poll())
                st = self.engine.codes.get(code)
                c = st.cand if st else None
                if c is not None and self._last_cand.get(code) != (c.series, c.t1):
                    self._last_cand[code] = (c.series, c.t1)
                    self.latency.engine_start(code, c.t0, c.p0, c.t1, c.p1, series=c.series)
                out += self.engine.evaluate(now, code, blocked=lambda c_: self.blocked.get(c_))
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                self.log(f"{TAG} {code} | error: {type(exc).__name__}: {exc}")
        for d in out:
            self.decisions += 1
            self._write(d)
            if d.action == "BUY":
                self.latency.engine_decision(d.code, d.at, d.price, d.reason)
                ref = f"T0={d.t0:%H:%M:%S}" if d.t0 else f"T1={d.t1:%H:%M:%S}"
                self.log(f"{TAG} {d.code} | SHADOW BUY {d.price:,.0f} | {d.reason} | {ref}")
            else:
                self.latency.engine_end(d.code, f"EXPIRED_{d.reason}")
        return out

    def note_submit(self, code: str, at: datetime, price: float, reason: str) -> None:
        self.latency.order_submit(str(code).zfill(6), at, price, reason)

    def note_fill(self, code: str, at: datetime, price: float) -> None:
        self.latency.order_fill(str(code).zfill(6), at, price)

    def close(self, outcome: str = "STOP") -> None:
        self.latency.close(outcome)

    def _write(self, d: EntryDecision) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.decisions_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
        except OSError as exc:
            self.errors += 1
            self.log(f"{TAG} write failed: {exc}")


class EntryShadowService:
    """r003 side: set_watch / note_submit / note_fill only enqueue (never block, never raise). The worker evaluates every
    interval_s on its own thread and logs only through its own file logger (never the r003 logger)."""

    def __init__(self, factory: Callable[[], EntryShadowRunner], interval_s: float = 1.0,
                 clock: Callable[[], datetime] = datetime.now, log: Callable[[str], None] | None = None):
        self._factory, self.interval_s, self.clock = factory, interval_s, clock
        self.log = log or (lambda _m: None)
        self.runner: EntryShadowRunner | None = None
        self._q: queue.Queue = queue.Queue(maxsize=2000)
        self.dropped = 0
        self._stop = threading.Event()
        self._outcome = "STOP"
        self.ready = threading.Event()
        self.finished = threading.Event()            # set after the final drain + latency close (or a failed init)
        self._thread = threading.Thread(target=self._worker, name="entry-shadow", daemon=True)
        self._thread.start()

    def _put(self, item: tuple) -> None:
        try:
            self._q.put_nowait(item)
        except Exception:  # noqa: BLE001
            self.dropped += 1

    def set_watch(self, codes: Iterable[str], blocked: dict[str, str] | None = None) -> None:
        self._put(("watch", tuple(codes), dict(blocked or {})))

    def note_submit(self, code: str, at: datetime, price: float, reason: str = "") -> None:
        self._put(("submit", str(code), at, float(price), str(reason)))

    def note_fill(self, code: str, at: datetime, price: float) -> None:
        self._put(("fill", str(code), at, float(price)))

    def close(self, timeout_s: float = 3.0, outcome: str = "STOP") -> bool:
        """outcome is written on every open latency record (STOP / DATE_CHANGE). timeout_s=0: do not wait.
        Returns True when the worker has finished its final write (self.finished)."""
        self._outcome = outcome
        self._stop.set()
        if timeout_s > 0:
            self._thread.join(timeout_s)
        return self.finished.is_set()

    def _drain(self, limit: int | None = 500) -> None:
        n = 0
        while limit is None or n < limit:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            n += 1
            try:                                     # one bad note never stops the others
                kind = item[0]
                if kind == "watch":
                    self.runner.set_watch(item[1], item[2])
                elif kind == "submit":
                    self.runner.note_submit(*item[1:])
                elif kind == "fill":
                    self.runner.note_fill(*item[1:])
            except Exception as exc:  # noqa: BLE001
                self.runner.errors += 1
                self.log(f"{TAG} note error ({item[0]}): {type(exc).__name__}: {exc}")

    def _worker(self) -> None:
        try:
            self.runner = self._factory()
        except Exception as exc:  # noqa: BLE001
            self.log(f"{TAG} init failed on worker, shadow disabled: {type(exc).__name__}: {exc}")
            self.finished.set()
            return
        self.ready.set()
        while not self._stop.is_set():
            try:
                self._drain()
                self.runner.step(self.clock())
            except Exception as exc:  # noqa: BLE001
                self.runner.errors += 1
                self.log(f"{TAG} worker error: {type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_s)
        try:
            self._drain(limit=None)                  # every queued submit / fill before the final write
        except Exception as exc:  # noqa: BLE001
            self.log(f"{TAG} final drain error: {type(exc).__name__}: {exc}")
        try:
            if self.dropped:
                self.log(f"{TAG} {self.dropped} queued note(s) dropped (queue full)")
            self.runner.close(self._outcome)
        except Exception as exc:  # noqa: BLE001
            self.log(f"{TAG} worker stop error: {type(exc).__name__}: {exc}")
        finally:
            self.finished.set()
