# -*- coding: utf-8 -*-
"""G033 entry latency decomposition (stage 1 of the entry-engine review, 2026-11-15).

What it answers (from r003 logs + g011 tick files, no new data needed)
- Per REAL buy: T0 intrabar cross print -> T1 tick-bar cross (bar close) -> T1L r003 saw the cross -> TP first poll where every
  gate passed (BUY HOLD 1/2) -> T4 order sent (BUY SUBMIT) -> T5 fill (buy block avg price). Time and price rise of every
  segment, which reject reasons held the order back between T1L and T4, slippage, MFE / MAE after the fill and "had we
  bought at T0".
- Per LOGGED tick cross (all watchlist crosses r003 saw, bought or not): forward returns from T0 / T1 / T1+60 s / T1+180 s,
  cross-bar volume ratio and buy share (fake breakout check), 60- vs 120-tick bar delay.

The live entry pipeline (ENTRY_TRIGGER_MODE="triple") has no sequential MACD / RSI step: the trigger is 1-min MA5>MA20
state + 60/120-tick MA5xMA20 cross, then 3-min context gates, Edge AI gate, spread gate and a 2-poll confirm. T2/T3 of the
request map to "TP" (all gates passed once) and the confirm wait.

Usage:  python g033_entry_latency.py [--days 20261030,20261031] [--out <dir>]
Output: <runtime>/data/sim/g033/<run>/report.md + real_buys.csv + crosses.csv
"""
from __future__ import annotations

import argparse
import bisect
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

from r001_define_config import RUNTIME_DATA_DIR, TICK_DATA_DIR  # noqa: E402
from r009_tick_bars import iter_tick_file  # noqa: E402
from r013_tick_trigger import TickReplayFeed  # noqa: E402

LOG_ROOT = RUNTIME_DATA_DIR.parent / "log"
COST_PCT = 0.23            # fees both sides + sell tax (no slippage)
FAST, SLOW = 5, 20
END_CLIP = "15:20:00"

_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} \[INFO\]\s+\[([A-Z0-9 _]+?)\s*\] (.*)$")
_CROSS = re.compile(r"^(\d{6}) \| tick(\d+) MA5>MA20 at (\d\d:\d\d:\d\d) ma5=([\d,.]+) ma20=([\d,.]+) bars=(\d+)")
_HOLD = re.compile(r"^(\d{6})\S*\s*\| reason=WAIT_NEXT_POLL_CONFIRM \| count=(\d+)/(\d+) \| live=([\d,]+)")
_SUBMIT = re.compile(r"^(\d{6})\S*\s*\| qty=(\d+) \| price=([\d,]+) \| (market|limit) \| (\w+) \| order_no=(\S*) \| reason=(\S+)")
_REJECT = re.compile(r"^(\d{6})\S*\s*\| (\S+)")
_CHECK = re.compile(r"^(\d{6})\S*\s*\| bars=\d+ live=([\d,]+).*?BB_MID=([\d.]+) BB_UP=([\d.]+).*?\| RSI=([\d.]+) ")
_EVAL = re.compile(r"^(\d{6})\S*\s*\| OK (\S+) \| [\d:]+ \| LIVE ([\d,]+) .*?RSI=([\d.]+) .*?MACD ([-\d.]+)->([-\d.]+) SIG=([-\d.]+)"
                   r".*?VOL=([\d,]+) VOLMA=([\d,]+)")
_BUYBLK = re.compile(r">>> (\d{6})_(.*?) Buy - reason=(\S+) .*?live=([\d,]+).*?vol_ratio=([\d.]+)")
_BUYROW = re.compile(r"매수시각=([\d\- :]+) \| 매수가격=([\d,.]+) \| 매수수량=(\d+)")
_BLKTIME = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _f(s: str) -> float:
    return float(str(s).replace(",", ""))


def _pct(a: float, b: float) -> float:
    return (b / a - 1.0) * 100.0 if a and b else float("nan")


# ---------------------------------------------------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class DayLog:
    crosses: list[dict] = field(default_factory=list)
    holds: list[dict] = field(default_factory=list)
    submits: list[dict] = field(default_factory=list)
    rejects: dict[str, list[tuple[datetime, str]]] = field(default_factory=lambda: defaultdict(list))
    checks: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    evals: list[dict] = field(default_factory=list)
    buys: list[dict] = field(default_factory=list)


def parse_day_logs(day: str) -> DayLog:
    out = DayLog()
    ddir = LOG_ROOT / day
    seen_cross: set[tuple] = set()
    for f in sorted(ddir.glob("r003_trade_live_execute_*.log")):
        if f.name.endswith("_buy_sell.log"):
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _LINE.match(line.rstrip("\n"))
                if not m:
                    continue
                ts, tag, rest = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2), m.group(3)
                if tag == "TICK CROSS":
                    c = _CROSS.match(rest)
                    if c:
                        at = datetime.strptime(f"{day} {c.group(3)}", "%Y%m%d %H:%M:%S")
                        key = (c.group(1), int(c.group(2)), at)
                        if key not in seen_cross:              # restarts re-log the same cross
                            seen_cross.add(key)
                            out.crosses.append({"code": c.group(1), "size": int(c.group(2)), "t1": at, "t1_log": ts,
                                                "bars": int(c.group(6))})
                elif tag == "BUY HOLD":
                    h = _HOLD.match(rest)
                    if h:
                        out.holds.append({"code": h.group(1), "t": ts, "count": int(h.group(2)), "live": _f(h.group(4))})
                elif tag == "BUY SUBMIT":
                    s = _SUBMIT.match(rest)
                    if s:
                        out.submits.append({"code": s.group(1), "t": ts, "qty": int(s.group(2)), "price": _f(s.group(3)),
                                            "type": s.group(4), "reason": s.group(7)})
                elif tag == "REJECT":
                    r = _REJECT.match(rest)
                    if r:
                        out.rejects[r.group(1)].append((ts, r.group(2)))
                elif tag == "CHECK":
                    c = _CHECK.match(rest)
                    if c:
                        out.checks[c.group(1)].append({"t": ts, "live": _f(c.group(2)), "bb_mid": float(c.group(3)),
                                                       "bb_up": float(c.group(4)), "rsi": float(c.group(5))})
                elif tag == "BUY EVAL":
                    e = _EVAL.match(rest)
                    if e:
                        out.evals.append({"code": e.group(1), "t": ts, "rsi3": float(e.group(4)),
                                          "macd": float(e.group(6)), "macd_sig": float(e.group(7)),
                                          "vol": _f(e.group(8)), "volma": _f(e.group(9))})
    bs = ddir / f"{day}_buy_sell.log"
    if bs.exists():
        lines = bs.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            b = _BUYBLK.search(line)
            if not b or i + 1 >= len(lines):
                continue
            row = _BUYROW.search(lines[i + 1])
            tm = _BLKTIME.match(line)
            if not row or not tm:
                continue
            out.buys.append({"code": b.group(1), "name": b.group(2), "reason": b.group(3), "live": _f(b.group(4)),
                             "vol_ratio": float(b.group(5)), "logged": datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S"),
                             "buy_time": datetime.strptime(row.group(1).strip(), "%Y-%m-%d %H:%M:%S"),
                             "fill": _f(row.group(2)), "qty": int(row.group(3))})
    out.holds.sort(key=lambda x: x["t"])
    out.submits.sort(key=lambda x: x["t"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ticks
# ---------------------------------------------------------------------------------------------------------------------
class DayTicks:
    def __init__(self, day: str, feed: TickReplayFeed):
        self.day = day
        self.feed = feed
        self._px: dict[str, tuple[list[datetime], list[float]]] = {}
        self._ticks: dict[str, list] = {}

    def ticks(self, code: str) -> list:
        if code not in self._ticks:
            p = TICK_DATA_DIR / self.day / f"{code}.csv"
            ts = [t for t in iter_tick_file(p) if t.recv is not None and t.price > 0] if p.exists() else []
            ts.sort(key=lambda t: t.recv)
            self._ticks[code] = ts
            self._px[code] = ([t.recv for t in ts], [t.price for t in ts])
        return self._ticks[code]

    def price_at(self, code: str, t: datetime) -> float:
        self.ticks(code)
        rs, ps = self._px[code]
        i = bisect.bisect_right(rs, t)
        return ps[i - 1] if i > 0 else float("nan")

    def path(self, code: str, t0: datetime, minutes: float) -> tuple[float, float, float]:
        """(max, min, last) traded price in (t0, t0+minutes], clipped at 15:20."""
        self.ticks(code)
        rs, ps = self._px[code]
        end = min(t0 + timedelta(minutes=minutes), datetime.strptime(f"{self.day} {END_CLIP}", "%Y%m%d %H:%M:%S"))
        i, j = bisect.bisect_right(rs, t0), bisect.bisect_right(rs, end)
        if j <= i:
            return float("nan"), float("nan"), float("nan")
        seg = ps[i:j]
        return max(seg), min(seg), seg[-1]

    def cross_detail(self, code: str, size: int, t1: datetime) -> dict:
        """T0 = first print INSIDE the cross bar at which MA5 > MA20 would already hold with that print as the bar close.
        Also the cross bar's volume ratio (vs the 20 bars before) and buyer share."""
        loaded = self.feed._load(code, self.day)
        if loaded is None:
            return {}
        bars = loaded[0][size].bars
        idx = None
        for i, b in enumerate(bars):
            if b.end_ts.date() != t1.date() and b.available_at.date() != t1.date():
                continue
            if abs((b.available_at - t1).total_seconds()) <= 1.0:
                idx = i
                break
        if idx is None or idx < SLOW:
            return {"matched": False}
        closes = [b.close for b in bars]
        prev = bars[idx - 1]
        lo = prev.available_at
        hi = bars[idx].available_at
        s4 = sum(closes[idx - FAST + 1:idx])
        s19 = sum(closes[idx - SLOW + 1:idx])
        t0, p0 = hi, bars[idx].close
        for t in self.ticks(code):
            if t.recv <= lo:
                continue
            if t.recv > hi:
                break
            if (s4 + t.price) / FAST > (s19 + t.price) / SLOW:
                t0, p0 = t.recv, t.price
                break
        vols = [b.volume for b in bars[idx - SLOW:idx] if b.volume > 0]
        cb = bars[idx]
        side = cb.buy_vol + cb.sell_vol
        return {"matched": True, "t0": t0, "p0": p0, "bar_open_t": lo, "cross_close": cb.close,
                "bar_secs": (cb.available_at - lo).total_seconds(),
                "vol_ratio_bar": cb.volume / np.mean(vols) if vols else float("nan"),
                "buy_share_bar": cb.buy_vol / side if side else float("nan")}


# ---------------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------------
def _gate_of(reason: str) -> str:
    r = reason.upper()
    for key, label in (("TRIPLE_CROSS_MISSING_1MIN", "1분 MA5>MA20 미충족"), ("CHASE", "교차 대비 추격 한도"),
                       ("CROSS_EXPIRED", "틱 교차 만료"), ("EDGE_AI", "Edge AI 게이트"), ("ATR", "변동성(ATR) 게이트"),
                       ("SPREAD", "호가 비용"), ("TICK_COST", "호가 비용"), ("ANTI_CHASE", "당일 과열"),
                       ("CONFIRM", "연속 확인"), ("MKT", "시장 추세"), ("REGIME", "시장 추세"), ("LIVE_DROP", "1분 하락"),
                       ("BB_GAP", "BB 갭 추격 가드"), ("CANDLE", "캔들 추격 가드"), ("VOL", "거래량/유동성")):
        if key in r:
            return label
    return r.split("_")[0] if r else "?"


def real_buy_rows(day: str, log: DayLog, ticks: DayTicks) -> list[dict]:
    rows = []
    used_blocks: set[int] = set()
    attempts: list[list[dict]] = []
    for s in log.submits:                                         # legs within 5 s = one attempt
        if attempts and attempts[-1][0]["code"] == s["code"] and (s["t"] - attempts[-1][0]["t"]).total_seconds() <= 5:
            attempts[-1].append(s)
        else:
            attempts.append([s])
    for legs in attempts:
        s0 = legs[0]
        code, t4 = s0["code"], s0["t"]
        if "REENTRY" in s0["reason"] or "REPRICE" in s0["reason"]:
            pass
        m = re.search(r"TICK(\d+)_GOLDEN_CROSS", s0["reason"])
        size = int(m.group(1)) if m else None
        blk = None
        for k, b in enumerate(log.buys):
            if k in used_blocks or b["code"] != code:
                continue
            if t4 - timedelta(seconds=60) <= b["logged"] <= t4 + timedelta(minutes=15):
                blk, _ = b, used_blocks.add(k)
                break
        cr = None
        if size:
            cands = [c for c in log.crosses if c["code"] == code and c["size"] == size and c["t1_log"] <= t4
                     and (t4 - c["t1"]).total_seconds() <= 300]
            cr = min(cands, key=lambda c: c["t1"]) if cands else None    # chase reference = earliest (r001)
        row = {"day": day, "code": code, "size": size, "t4": t4, "submit_px": s0["price"],
               "legs": "+".join(sorted({x["type"] for x in legs})), "reason": s0["reason"]}
        if blk:
            row.update({"name": blk["name"], "fill": blk["fill"], "t5_log": blk["logged"], "vol_ratio_3m": blk["vol_ratio"]})
        if cr:
            row.update({"t1": cr["t1"], "t1_log": cr["t1_log"]})
            det = ticks.cross_detail(code, size, cr["t1"])
            if det.get("matched"):
                row.update({"t0": det["t0"], "p0": det["p0"], "bar_open_t": det["bar_open_t"], "cross_close": det["cross_close"],
                            "bar_secs": det["bar_secs"], "vol_ratio_bar": det["vol_ratio_bar"],
                            "buy_share_bar": det["buy_share_bar"]})
            holds = [h for h in log.holds if h["code"] == code and cr["t1_log"] <= h["t"] <= t4]
            if holds:
                row["tp"] = holds[0]["t"]
            rj = [r for (t, r) in log.rejects.get(code, []) if cr["t1_log"] <= t <= t4]
            if rj:
                cnt = Counter(_gate_of(r) for r in rj)
                row["wait_gates"] = "; ".join(f"{k}x{v}" for k, v in cnt.most_common(3))
                last = [r for (t, r) in log.rejects.get(code, []) if cr["t1_log"] <= t <= t4][-1]
                row["last_gate"] = _gate_of(last)
        chk = [c for c in log.checks.get(code, []) if c["t"] <= t4]
        if chk:
            c = chk[-1]
            row.update({"rsi3": c["rsi"], "bb_up_gap_pct": _pct(c["live"], c["bb_up"]),
                        "pctb": (c["live"] - c["bb_mid"]) / (c["bb_up"] - c["bb_mid"]) if c["bb_up"] > c["bb_mid"] else float("nan")})
        ev = [e for e in log.evals if e["code"] == code and abs((e["t"] - t4).total_seconds()) <= 5]
        if ev:
            row.update({"macd_hist": ev[0]["macd"] - ev[0]["macd_sig"], "vol_over_ma_3m": ev[0]["vol"] / ev[0]["volma"]
                        if ev[0]["volma"] else float("nan")})
        # prices at each point (traded price available at that time)
        for k in ("t1", "t1_log", "tp", "t4"):
            if row.get(k):
                row[f"px_{k}"] = ticks.price_at(code, row[k])
        base_t = row.get("t4")
        fill = row.get("fill")
        if fill and base_t:
            for w in (5, 10, 30):
                hi, lo, last = ticks.path(code, base_t, w)
                row[f"mfe{w}"], row[f"mae{w}"], row[f"ret{w}"] = _pct(fill, hi), _pct(fill, lo), _pct(fill, last)
            if row.get("p0") and row.get("t0"):
                hi, lo, last = ticks.path(code, row["t0"], 30 + (base_t - row["t0"]).total_seconds() / 60.0)
                row["ret30_from_t0"] = _pct(row["p0"], last)
                row["mfe30_from_t0"], row["mae30_from_t0"] = _pct(row["p0"], hi), _pct(row["p0"], lo)
        rows.append(row)
    return rows


def cross_rows(day: str, log: DayLog, ticks: DayTicks) -> list[dict]:
    rows = []
    for c in log.crosses:
        if c["t1"].strftime("%H:%M:%S") < "09:05:00" or c["t1"].strftime("%H:%M:%S") > "14:50:00":
            continue
        det = ticks.cross_detail(c["code"], c["size"], c["t1"])
        if not det.get("matched"):
            continue
        r = {"day": day, "code": c["code"], "size": c["size"], "t0": det["t0"], "t1": c["t1"], "t1_log": c["t1_log"],
             "p0": det["p0"], "bar_secs": det["bar_secs"], "vol_ratio_bar": det["vol_ratio_bar"],
             "buy_share_bar": det["buy_share_bar"]}
        for name, t in (("t0", det["t0"]), ("t1", c["t1"]), ("t1log", c["t1_log"]),
                        ("t1p60", c["t1"] + timedelta(seconds=60)), ("t1p180", c["t1"] + timedelta(seconds=180))):
            px = ticks.price_at(c["code"], t)
            r[f"px_{name}"] = px
            for w in (10, 30):
                _hi, _lo, last = ticks.path(c["code"], t, w)
                r[f"ret{w}_{name}"] = _pct(px, last)
            hi, lo, _ = ticks.path(c["code"], t, 30)
            r[f"mfe30_{name}"], r[f"mae30_{name}"] = _pct(px, hi), _pct(px, lo)
        rows.append(r)
    return rows


def _q(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "-"
    return f"{s.mean():+.3f} / {s.median():+.3f} (n={len(s)})"


def _secs(a: pd.Series, b: pd.Series) -> pd.Series:
    return (pd.to_datetime(b) - pd.to_datetime(a)).dt.total_seconds()


def report(rb: pd.DataFrame, cx: pd.DataFrame, days: list[str]) -> str:
    L = [f"# g033 매수 지연 분해 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
         "평균 / 중앙값 (표본 수). 가격 변화는 T0 가격 기준 %. 비용 0.23%는 수익률 표에 미반영(별도 표기).", ""]
    if not rb.empty:
        rb = rb.copy()
        L += ["## 1. 실매수 지연 분해", ""]
        segs = [("T0 봉 내부 교차 -> T1 틱봉 마감", "t0", "t1", "p0", "px_t1"),
                ("T1 -> T1L r003 인지", "t1", "t1_log", "px_t1", "px_t1_log"),
                ("T1L -> TP 모든 게이트 1회 통과", "t1_log", "tp", "px_t1_log", "px_tp"),
                ("TP -> T4 주문 (연속 확인)", "tp", "t4", "px_tp", "px_t4"),
                ("T1L -> T4 (TP 없음 포함)", "t1_log", "t4", "px_t1_log", "px_t4")]
        L += ["| 구간 | 시간(초) | 가격 변화(%) |", "|---|---|---|"]
        for label, a, b, pa, pb in segs:
            if a in rb and b in rb:
                ok = rb[a].notna() & rb[b].notna()
                L.append(f"| {label} | {_q(_secs(rb.loc[ok, a], rb.loc[ok, b]))} | "
                         f"{_q(((rb.loc[ok, pb] / rb.loc[ok, pa]) - 1) * 100)} |")
        if "fill" in rb and "px_t4" in rb:
            ok = rb["fill"].notna() & rb["px_t4"].notna()
            L.append(f"| T4 -> T5 체결 (주문 시점 체결가 대비 평균 체결가) | - | {_q((rb.loc[ok, 'fill'] / rb.loc[ok, 'px_t4'] - 1) * 100)} |")
            ok2 = rb["fill"].notna() & rb["submit_px"].notna()
            L.append(f"| 슬리피지: 판정 현재가(submit price) 대비 체결가 | - | {_q((rb.loc[ok2, 'fill'] / rb.loc[ok2, 'submit_px'] - 1) * 100)} |")
        if "p0" in rb and "fill" in rb:
            ok = rb["fill"].notna() & rb["p0"].notna()
            L.append(f"| **합계 T0 -> 체결가** | {_q(_secs(rb.loc[ok, 't0'], rb.loc[ok, 't4']))} | "
                     f"{_q((rb.loc[ok, 'fill'] / rb.loc[ok, 'p0'] - 1) * 100)} |")
        L += ["", f"- 실매수 시도 {len(rb)}건, 틱 교차 매칭 {int(rb.get('p0', pd.Series(dtype=float)).notna().sum())}건, "
              f"체결 블록 매칭 {int(rb.get('fill', pd.Series(dtype=float)).notna().sum())}건.", ""]
        if "size" in rb:
            L += ["### 1.1 봉 종류별 (실매수 트리거)", "", "| 트리거 | 건수 | T0->T1 초 | T0->T1 % | T0->체결가 % |", "|---|---|---|---|---|"]
            for sz, g in rb.groupby("size"):
                okg = g["p0"].notna() if "p0" in g else pd.Series(False, index=g.index)
                L.append(f"| {int(sz)}틱 | {len(g)} | {_q(_secs(g.loc[okg, 't0'], g.loc[okg, 't1']))} | "
                         f"{_q((g.loc[okg, 'px_t1'] / g.loc[okg, 'p0'] - 1) * 100)} | "
                         f"{_q((g.loc[okg & g['fill'].notna(), 'fill'] / g.loc[okg & g['fill'].notna(), 'p0'] - 1) * 100)} |")
            L.append("")
        if "last_gate" in rb:
            L += ["### 1.2 T1L -> T4 사이 매수를 늦춘 조건 (마지막 반려 사유 기준)", "", "| 조건 | 건수 |", "|---|---|"]
            for k, v in rb["last_gate"].value_counts().items():
                L.append(f"| {k} | {v} |")
            L.append("")
        L += ["### 1.3 매수 후 결과 (체결가 기준, 비용 전 %)", "", "| 창 | MFE | MAE | 종료 수익 |", "|---|---|---|---|"]
        for w in (5, 10, 30):
            if f"mfe{w}" in rb:
                L.append(f"| {w}분 | {_q(rb[f'mfe{w}'])} | {_q(rb[f'mae{w}'])} | {_q(rb[f'ret{w}'])} |")
        if "ret30_from_t0" in rb:
            L += ["", f"- T0 가격으로 샀다면(같은 종료 시각까지): 수익 {_q(rb['ret30_from_t0'])}, MFE {_q(rb['mfe30_from_t0'])}, "
                  f"MAE {_q(rb['mae30_from_t0'])}. 실제(체결가, 30분): {_q(rb.get('ret30', pd.Series(dtype=float)))}."]
        L.append("")
        L += ["### 1.4 매수 시점 과열·거래량 (H4/H5)", ""]
        for col, label in (("rsi3", "3분 RSI"), ("pctb", "%B (중심선 0, 상단 1)"), ("bb_up_gap_pct", "상단까지 남은 %"),
                           ("vol_ratio_3m", "3분 거래량/MA20"), ("vol_ratio_bar", "교차 틱봉 거래량/직전 20봉"),
                           ("buy_share_bar", "교차 틱봉 매수 체결 비중")):
            if col in rb:
                L.append(f"- {label}: {_q(rb[col])}")
        if "rsi3" in rb and "ret30" in rb:
            ok = rb["rsi3"].notna() & rb["ret30"].notna()
            hi = rb.loc[ok & (rb["rsi3"] >= 70), "ret30"]
            lo = rb.loc[ok & (rb["rsi3"] < 70), "ret30"]
            L.append(f"- 30분 수익: RSI>=70 {_q(hi)} / RSI<70 {_q(lo)}")
        L.append("")
    if not cx.empty:
        L += ["## 2. r003 이 본 모든 틱 교차 (매수 여부 무관, 09:05~14:50)", "",
              "같은 교차에서 사는 시점을 늦출수록 이후 수익이 어떻게 변하는지(지연 비용). 비용 전 %.", "",
              "| 매수 시점 | 10분 수익 | 30분 수익 | 30분 MFE | 30분 MAE |", "|---|---|---|---|---|"]
        for name, label in (("t0", "T0 봉 내부 교차"), ("t1", "T1 틱봉 마감"), ("t1log", "T1L r003 인지"),
                            ("t1p60", "T1+60초"), ("t1p180", "T1+180초")):
            L.append(f"| {label} | {_q(cx[f'ret10_{name}'])} | {_q(cx[f'ret30_{name}'])} | {_q(cx[f'mfe30_{name}'])} | "
                     f"{_q(cx[f'mae30_{name}'])} |")
        L += ["", "| 봉 | 교차 수 | 봉 길이(초) | T0->T1 초 | T0->T1 % |", "|---|---|---|---|---|"]
        for sz, g in cx.groupby("size"):
            L.append(f"| {int(sz)}틱 | {len(g)} | {_q(g['bar_secs'])} | {_q(_secs(g['t0'], g['t1']))} | "
                     f"{_q((g['px_t1'] / g['p0'] - 1) * 100)} |")
        L += ["", "### 2.1 교차 틱봉 거래량 (H5: 거래량 없는 돌파)", "", "| 교차 봉 거래량/직전 20봉 | 교차 수 | 30분 수익(T1) | 30분 MAE |",
              "|---|---|---|---|"]
        b = pd.cut(cx["vol_ratio_bar"], [0, 0.7, 1.0, 1.5, 2.5, np.inf])
        for k, g in cx.groupby(b, observed=True):
            L.append(f"| {k} | {len(g)} | {_q(g['ret30_t1'])} | {_q(g['mae30_t1'])} |")
        L += ["", "| 교차 봉 매수 체결 비중 | 교차 수 | 30분 수익(T1) |", "|---|---|---|"]
        b = pd.cut(cx["buy_share_bar"], [0, 0.4, 0.5, 0.6, 0.7, 1.0001])
        for k, g in cx.groupby(b, observed=True):
            L.append(f"| {k} | {len(g)} | {_q(g['ret30_t1'])} |")
        L += ["", "### 2.2 날짜별 (T1 기준 30분 수익 평균)", "", "| 날짜 | 교차 수 | 30분 수익 |", "|---|---|---|"]
        for d, g in cx.groupby("day"):
            L.append(f"| {d} | {len(g)} | {_q(g['ret30_t1'])} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------------
# stage bbmid: every real buy since the logs start vs the 1-minute BB middle (MA20) cross that preceded it (no ticks needed)
# ---------------------------------------------------------------------------------------------------------------------
def _minute_bars(day: str, code: str) -> pd.DataFrame | None:
    files = sorted((RUNTIME_DATA_DIR / day).glob(f"{code}_*_10s.txt"))
    if not files:
        return None
    df = pd.read_csv(files[0], encoding="utf-8-sig", usecols=["datetime", "open", "high", "low", "close", "raw_bar"])
    df = df[df["raw_bar"].astype(str).isin(["1", "1.0", "True"])]
    if df.empty:
        return None
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")[["open", "high", "low", "close"]].astype(float)


def bbmid_rows() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "data_simulation"))
    from g026_timing_study import real_buys   # noqa: WPS433 - same parser as the timing study
    rb = real_buys()
    if rb.empty:
        return rb
    rb = rb.sort_values("sell_time").drop_duplicates(["code", "buy_time"], keep="first")
    days = sorted(p.name for p in RUNTIME_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8)
    rows = []
    for _, b in rb.iterrows():
        day, code, bt, bp = b["date"], b["code"], b["buy_time"], float(b["buy_price"])
        if day not in days:
            continue
        prev = [d for d in days if d < day][-1:] if any(d < day for d in days) else []
        frames = [f for f in (_minute_bars(d, code) for d in prev + [day]) if f is not None]
        if not frames:
            continue
        m = pd.concat(frames).sort_index()
        m = m[~m.index.duplicated(keep="last")]
        m["mid"] = m["close"].rolling(20).mean()
        m["ma5"] = m["close"].rolling(5).mean()
        m["end"] = m.index + pd.Timedelta(minutes=1)
        today = m[(m.index.strftime("%Y%m%d") == day) & (m["end"] <= bt)]
        if len(today) < 2:
            continue
        up = (today["close"] > today["mid"]) & (today["close"].shift(1) <= today["mid"].shift(1))
        ups = today[up & (today["end"] >= bt - pd.Timedelta(minutes=60))]
        row = {"day": day, "code": code, "name": b["name"], "buy_time": bt, "buy_price": bp, "pnl_pct": b["pnl_pct"],
               "reason": b["reason"]}
        if not ups.empty:
            cb = ups.iloc[-1]
            row.update({"bbmid_cross_end": cb["end"], "bbmid_close": cb["close"], "bbmid_level": cb["mid"],
                        "bbmid_secs": (bt - cb["end"]).total_seconds(),
                        "gap_vs_cross_close_pct": _pct(cb["close"], bp), "gap_vs_mid_pct": _pct(cb["mid"], bp)})
        ma = (today["ma5"] > today["mid"]) & (today["ma5"].shift(1) <= today["mid"].shift(1))
        mas = today[ma & (today["end"] >= bt - pd.Timedelta(minutes=60))]
        if not mas.empty:
            cm = mas.iloc[-1]
            row.update({"ma5x20_end": cm["end"], "ma5x20_secs": (bt - cm["end"]).total_seconds(),
                        "gap_vs_ma5x20_close_pct": _pct(cm["close"], bp)})
        rows.append(row)
    return pd.DataFrame(rows)


def report_bbmid(df: pd.DataFrame) -> str:
    L = ["# g033 실매수 vs 직전 1분 BB 중간선(MA20) 상향 돌파", "",
         f"실매수 {len(df)}건 ({df['day'].min()} ~ {df['day'].max()}), 매수 전 60분 안의 마지막 돌파 기준. 평균 / 중앙값 (n).", "",
         "| 기준 | 경과 시간(초) | 체결가 차이(%) |", "|---|---|---|",
         f"| 1분 종가가 BB 중간선 상향 돌파한 봉의 종가 | {_q(df.get('bbmid_secs', pd.Series(dtype=float)))} | "
         f"{_q(df.get('gap_vs_cross_close_pct', pd.Series(dtype=float)))} |",
         f"| 같은 봉의 BB 중간선 값(장중 돌파 가격) | - | {_q(df.get('gap_vs_mid_pct', pd.Series(dtype=float)))} |",
         f"| 1분 MA5 x MA20 골든크로스 봉 종가 | {_q(df.get('ma5x20_secs', pd.Series(dtype=float)))} | "
         f"{_q(df.get('gap_vs_ma5x20_close_pct', pd.Series(dtype=float)))} |", ""]
    if "gap_vs_cross_close_pct" in df:
        g = pd.to_numeric(df["gap_vs_cross_close_pct"], errors="coerce").dropna()
        L += ["| 돌파 봉 종가 대비 체결가 | 건수 | 비율 | 실현 손익(%) |", "|---|---|---|---|"]
        for lo, hi in ((-99, 0), (0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 99)):
            sel = df[(df["gap_vs_cross_close_pct"] > lo) & (df["gap_vs_cross_close_pct"] <= hi)]
            L.append(f"| ({lo}, {hi}] | {len(sel)} | {len(sel) / max(1, len(g)):.0%} | {_q(sel['pnl_pct'])} |")
        L.append(f"| 돌파 없음(60분 내) | {int(df['gap_vs_cross_close_pct'].isna().sum())} | - | "
                 f"{_q(df.loc[df['gap_vs_cross_close_pct'].isna(), 'pnl_pct'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--stage", default="ticks", choices=("ticks", "bbmid"))
    args = ap.parse_args()
    if args.stage == "bbmid":
        out = Path(args.out) if args.out else RUNTIME_DATA_DIR / "sim" / "g033" / ("bbmid_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)
        df = bbmid_rows()
        df.to_csv(out / "bbmid.csv", index=False, encoding="utf-8-sig")
        (out / "report_bbmid.md").write_text(report_bbmid(df), encoding="utf-8")
        print(f"rows {len(df)} out: {out}")
        return
    tick_days = {p.name for p in TICK_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()}
    log_days = {p.name for p in LOG_ROOT.iterdir() if p.is_dir() and p.name.isdigit()}
    days = sorted(tick_days & log_days)
    if args.days:
        days = [d for d in days if d in set(args.days.split(","))]
    feed = TickReplayFeed(TICK_DATA_DIR, (60, 120), FAST, SLOW, 240.0, 120.0, prior_day_warmup=True)
    rb_all, cx_all = [], []
    for day in days:
        log = parse_day_logs(day)
        ticks = DayTicks(day, feed)
        rb = real_buy_rows(day, log, ticks)
        cx = cross_rows(day, log, ticks)
        rb_all += rb
        cx_all += cx
        print(f"{day}: submits={len(log.submits)} buys={len(log.buys)} crosses={len(log.crosses)} -> rows {len(rb)} / {len(cx)}",
              flush=True)
        feed._loaded.clear()
    out = Path(args.out) if args.out else RUNTIME_DATA_DIR / "sim" / "g033" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rb_df, cx_df = pd.DataFrame(rb_all), pd.DataFrame(cx_all)
    rb_df.to_csv(out / "real_buys.csv", index=False, encoding="utf-8-sig")
    cx_df.to_csv(out / "crosses.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text(report(rb_df, cx_df, days), encoding="utf-8")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
