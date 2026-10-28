# -*- coding: utf-8 -*-
"""Strategy V2 shadow runner for r003 (virtual entries/exits on live data, NO orders).

Enabled only when r001 V2_LIVE_MODE == "shadow". r003 calls V2ShadowRunner.on_symbol() once per symbol per
loop with the confirmed 1m frame it already maintains and the live price. Everything here is wrapped so an
exception can never reach the r003 trading loop.

Data handling
- Keeps its OWN append-only 1m store per code (r003's 1m cache is capped at 200 bars and would shift the
  session VWAP/EMA after ~12:20). Only bars >= 09:00 (KRX regular session) are used.
- If the first stored bar starts after 09:05 (late start / restart), VWAP is not session-anchored: the code is
  marked partial and no virtual entries are taken for it (logged once).
- Daily filter uses the latest collected daily csv BEFORE today (runtime/data/<prev date>/<code>_*_daily.csv).
- Virtual fill = live price at signal time + slippage ticks (what an aggressive order would have paid).
  Exits are evaluated on completed 1m bars with r008.step_exit (same rules as the g012 backtest).

Output: runtime/data/live_runtime/v2_shadow_YYYYMMDD.jsonl (ENTRY / EXIT / SKIP records) + "[V2 SHADOW]" log lines.

Limitations (shadow v1)
- Evaluates V2 on r003's active watchlist codes that pass the V2 daily filter, not on the full V2 universe.
- One extra 1m chart request per code per minute (r003 _get_or_refresh_1min_frame) while enabled.

Update log:
- [2026-10-23] type=feat owner=claude
    summary: 신규 - V2 전략 섀도우(가상 체결) 러너. r003에서 V2_LIVE_MODE="shadow"일 때만 호출, 주문 없음.
    impact: live (기록만)
    compatibility: backward-compatible (기본 V2_LIVE_MODE="off")
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from r008_strategy_v2 import (
    DayRisk, EntryState, V2Config, V2Position, build_5m, completed_5m, evaluate_daily_trend, fill_cost_krw,
    market_breadth_ok, prepare_1m, size_position, slippage_price, step_entry, step_exit,
)

SESSION_START = "09:00"
PARTIAL_IF_FIRST_BAR_AFTER = "09:06"  # first bar END later than this -> VWAP not session-anchored


class V2ShadowRunner:
    def __init__(self, cfg: V2Config, data_dir: Path, out_dir: Path, date_str: str, log: Callable[[str], None]):
        self.cfg = cfg
        self.data_dir = data_dir
        self.out_path = out_dir / f"v2_shadow_{date_str}.jsonl"
        self.date_str = date_str
        self.log = log
        self.daily_ok: dict[str, object] = {}
        self.bars: dict[str, pd.DataFrame] = {}
        self.last_bar_end: dict[str, pd.Timestamp] = {}
        self.entry_state: dict[str, EntryState] = {}
        self.positions: dict[str, V2Position] = {}
        self.pos_meta: dict[str, dict] = {}
        self.partial_logged: set[str] = set()
        self.last_close_vs_vwap: dict[str, bool] = {}
        self.risk = DayRisk(equity=cfg.account_equity_krw)
        self.disabled_reason = ""

    # ---- helpers -------------------------------------------------------------------------
    def _emit(self, rec: dict) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), **rec}
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            self.log(f"  [V2 SHADOW   ] write failed: {exc}")

    def _prev_daily(self, code: str) -> pd.DataFrame | None:
        dirs = sorted(p for p in self.data_dir.iterdir() if p.is_dir() and p.name.isdigit() and p.name < self.date_str)
        for d in reversed(dirs[-10:]):
            files = list(d.glob(f"{code}_*_daily.csv"))
            if files:
                df = pd.read_csv(files[0], usecols=["date", "open", "high", "low", "close", "volume", "amount"], encoding="utf-8-sig")
                df = df[df["date"].astype(int) < int(self.date_str)].sort_values("date")
                return df.reset_index(drop=True)
        return None

    def _daily_pass(self, code: str):
        if code not in self.daily_ok:
            try:
                df = self._prev_daily(code)
                self.daily_ok[code] = evaluate_daily_trend(code, df, self.cfg) if df is not None else None
            except Exception as exc:
                self.log(f"  [V2 SHADOW   ] {code} daily load failed: {exc}")
                self.daily_ok[code] = None
            t = self.daily_ok[code]
            self._emit({"event": "DAILY", "code": code, "ok": bool(t and t.ok), "reason": t.reason if t else "NO_DAILY",
                        "tier": t.tier if t else ""})
        t = self.daily_ok[code]
        return t if (t is not None and t.ok) else None

    def _append_bars(self, code: str, frame_1m: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in frame_1m.columns]
        day0 = pd.Timestamp(f"{self.date_str[:4]}-{self.date_str[4:6]}-{self.date_str[6:]} {SESSION_START}")
        new = frame_1m[cols]
        new = new[new.index > day0]
        old = self.bars.get(code)
        merged = new if old is None else pd.concat([old, new[new.index > old.index[-1]]]) if len(old) else new
        merged = merged[~merged.index.duplicated(keep="last")].sort_index().astype(float)
        self.bars[code] = merged
        return merged

    # ---- main entry point -------------------------------------------------------------------
    def on_symbol(self, code: str, name: str, now: datetime, frame_1m: pd.DataFrame | None, live_price: float) -> None:
        try:
            self._on_symbol(code, name, now, frame_1m, live_price)
        except Exception as exc:  # never break the live loop
            self.log(f"  [V2 SHADOW   ] {code} error: {type(exc).__name__}: {exc}")

    def _on_symbol(self, code: str, name: str, now: datetime, frame_1m, live_price: float) -> None:
        if frame_1m is None or frame_1m.empty or not (live_price and live_price > 0):
            return
        trend = self._daily_pass(code)
        if trend is None and code not in self.positions:
            return
        m1 = self._append_bars(code, frame_1m)
        if m1.empty:
            return
        last_end = m1.index[-1]
        if self.last_bar_end.get(code) == last_end:
            return  # no new completed bar
        self.last_bar_end[code] = last_end
        partial = m1.index[0] > pd.Timestamp(f"{last_end:%Y-%m-%d} {PARTIAL_IF_FIRST_BAR_AFTER}")
        if partial and code not in self.partial_logged:
            self.partial_logged.add(code)
            self.log(f"  [V2 SHADOW   ] {code} partial session (first bar {m1.index[0]:%H:%M}) - VWAP not anchored, no entries")
            self._emit({"event": "SKIP", "code": code, "reason": "PARTIAL_SESSION", "first_bar": m1.index[0]})
        p = prepare_1m(m1, self.cfg)
        m5 = build_5m(p, self.cfg)
        m5c = completed_5m(m5, last_end)
        bar = p.iloc[-1]
        self.last_close_vs_vwap[code] = float(bar["close"]) > float(bar["VWAP"])

        # exits first (position state is independent from entry state)
        pos = self.positions.get(code)
        if pos is not None:
            for a in step_exit(pos, {k: float(bar[k]) for k in ("open", "high", "low", "close")}, last_end, m5c, self.cfg):
                px = a.price if a.reason == "TP1" else slippage_price(a.price, "sell", self.cfg.slippage_ticks_exit)
                meta = self.pos_meta[code]
                sell_cost = fill_cost_krw("sell", px, a.qty, self.cfg)
                meta["gross"] += (px - pos.entry_price) * a.qty
                meta["costs"] += sell_cost
                meta["reasons"].append(a.reason)
                self.risk.realized_pnl += (px - pos.entry_price) * a.qty - sell_cost  # each leg hits day risk now
                self._emit({"event": "EXIT_PART" if pos.remaining > 0 else "EXIT", "code": code, "name": name,
                            "reason": a.reason, "qty": a.qty, "price": px, "bar_end": last_end})
            if pos.remaining <= 0:
                meta = self.pos_meta.pop(code)
                self.positions.pop(code)
                net = meta["gross"] - meta["costs"]
                r = net / (pos.initial_risk * pos.qty) if pos.initial_risk > 0 and pos.qty else 0.0
                self.log(f"  [V2 SHADOW   ] EXIT {code} {name} | {'+'.join(meta['reasons'])} | net={net:,.0f} R={r:.2f}")
                self._emit({"event": "CLOSED", "code": code, "name": name, "setup": pos.setup, "net_pnl": round(net),
                            "r_net": round(r, 3), "reasons": meta["reasons"], "day_realized": round(self.risk.realized_pnl)})

        if partial:
            return
        st = self.entry_state.setdefault(code, EntryState())
        n_up = sum(1 for v in self.last_close_vs_vwap.values() if v)
        mkt_ok, breadth = market_breadth_ok(n_up, len(self.last_close_vs_vwap), self.cfg)
        sig = step_entry(code, st, p, m5, self.cfg, code in self.positions, market_ok=mkt_ok)
        if sig is None:
            return
        unreal = 0.0
        for c, q in self.positions.items():
            b = self.bars.get(c)
            if b is not None and len(b):
                unreal += (float(b["close"].iloc[-1]) - q.entry_price) * q.remaining
        ok, why = self.risk.can_open(len(self.positions), unreal, self.cfg)
        fill = slippage_price(float(live_price), "buy", self.cfg.slippage_ticks_entry)
        qty = size_position(fill, sig.stop, self.risk.equity, self.cfg) if ok else 0
        if not ok or qty <= 0 or fill <= sig.stop:
            reason = why if not ok else ("SIZE_ZERO" if qty <= 0 else "PRICE_BELOW_STOP")
            self._emit({"event": "SKIP", "code": code, "reason": reason, "signal": asdict(sig)})
            return
        pos = V2Position(code, sig.setup, pd.Timestamp(now), fill, qty, sig.stop, fill - sig.stop)
        self.positions[code] = pos
        buy_cost = fill_cost_krw("buy", fill, qty, self.cfg)
        self.pos_meta[code] = {"gross": 0.0, "costs": buy_cost, "reasons": []}
        self.risk.realized_pnl -= buy_cost
        self.log(f"  [V2 SHADOW   ] ENTRY {code} {name} | {sig.setup} | px={fill:,.0f} stop={sig.stop:,.0f} qty={qty} "
                 f"breadth={breadth:.0f}% | {sig.reason}")
        self._emit({"event": "ENTRY", "code": code, "name": name, "setup": sig.setup, "price": fill, "stop": sig.stop,
                    "qty": qty, "support": sig.support, "atr5": sig.atr5, "breadth_pct": breadth,
                    "tier": trend.tier if trend else "", "reason": sig.reason, "bar_end": sig.ts})
