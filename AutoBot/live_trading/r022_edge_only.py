# -*- coding: utf-8 -*-
"""Edge-AI-only strategy runner for r003 - PAPER trading (virtual entries/exits on live data, NO orders).

Enabled only when r001 EDGE_ONLY_MODE == "paper". r003 calls EdgeOnlyRunner.on_symbol() once per symbol per loop
(same hook as the V2 shadow, r010) with the confirmed 1m frame it already maintains and the live price. Everything is
wrapped so an exception can never reach the r003 trading loop. The normal rule-based buying/selling of r003 is untouched.

Policy (identical to data_simulation/g029_edge_only_sim.py, which measures it on past data)
  buy   on a newly completed 1m bar: the hybrid 1m trigger fires (the event type the model was trained on, g013) and the
        Edge AI model's probability >= EDGE_ONLY_MIN_PROB (0 = the threshold stored with the model). No other rule gate.
        09:05~14:50 only, one position per symbol, EDGE_ONLY_MAX_POSITIONS at a time, EDGE_ONLY_MAX_ENTRIES_PER_DAY a day.
  sell  the model's label definition (current_meta.json "label"): +tp% / -sl% / horizon minutes / 15:15, whichever first.
        Checked with the live price on every poll.
  fill  virtual: buy = live price + 1 tick, sell = the target price for a take-profit, else live price - 1 tick.
        Net % includes the commission (0.015% per side) and the sell tax (0.20%).

Data handling
- Keeps its own append-only regular-session 1m store per code (r003's 1m cache is capped at 200 bars, which would lose the
  day's open / VWAP after ~12:20).
- Open paper positions are restored from the day's file after an r003 restart.

Output: runtime/data/live_runtime/edge_only_YYYYMMDD.jsonl (ENTRY / EXIT records) + "[EDGE ONLY   ]" log lines.
Runs the same on Linux (Raspberry Pi) and Windows. Each host uses its own model (runtime/edge_ai, trained by the
post-close job from that host's data folder).

Update log:
- [2026-11-12] type=feat owner=claude
    summary: 신규 - Edge AI 메타 모델만으로 매매하는 방식의 가상 매매(paper) 러너. r003에서 EDGE_ONLY_MODE="paper"일 때만
      호출, 주문 없음. 과거 데이터 검증은 g029_edge_only_sim.py.
    impact: live (기록만 - 실제 매수/매도 판정 무변경)
    compatibility: backward-compatible (기본 EDGE_ONLY_MODE="off")
"""
from __future__ import annotations

import json
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Callable

import pandas as pd

import r011_edge_ai as r011
from r002_strategy_core_shared import krx_tick_size

ENTRY_START = dt_time(9, 5)
ENTRY_END = dt_time(14, 50)
FORCE_EXIT = dt_time(15, 15)
TRIGGER_CACHE_BARS = 200      # the live trigger never sees more history (r003 1m cache cap, g013 TRIGGER_CACHE_BARS)
FEE_RATE = 0.00015
SELL_TAX_RATE = 0.0020
DEFAULT_LABEL = {"tp": 2.0, "sl": 1.7, "horizon": 60}
TAG = "  [EDGE ONLY   ]"


def net_pct(entry: float, exit_px: float) -> float:
    """Net % of a round trip after commission (both sides) and sell tax."""
    cost = entry * FEE_RATE + exit_px * (FEE_RATE + SELL_TAX_RATE)
    return (exit_px - entry - cost) / entry * 100.0


class EdgeOnlyRunner:
    def __init__(self, model_dir: Path, data_root: Path, out_dir: Path, date_str: str, log: Callable[[str], None], config,
                 min_prob: float = 0.0, max_positions: int = 3, max_entries: int = 10, model=None):
        self.model = model if model is not None else r011.EdgeAIModel(Path(model_dir))
        self.data_root = Path(data_root)
        self.out_path = Path(out_dir) / f"edge_only_{date_str}.jsonl"
        self.date_str = date_str
        self.log = log
        self.config = config
        self.min_prob = float(min_prob or 0.0)
        self.max_positions = int(max_positions)
        self.max_entries = int(max_entries)
        self.bars: dict[str, pd.DataFrame] = {}
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.positions: dict[str, dict] = {}
        self.entries_today = 0
        self.closed: list[float] = []
        self._daily: dict[str, dict] = {}
        self._prev_close: dict[str, float] = {}
        self._restore()

    # ---- persistence ------------------------------------------------------------------------
    def _emit(self, rec: dict) -> None:
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with self.out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _restore(self) -> None:
        """Positions still open and the day's counters, from the day's file (an r003 restart must not forget them)."""
        if not self.out_path.is_file():
            return
        try:
            for line in self.out_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                code = str(rec.get("code") or "")
                if rec.get("event") == "ENTRY":
                    self.entries_today += 1
                    self.positions[code] = {"name": rec.get("name"), "entry": float(rec["entry"]), "ts": pd.Timestamp(rec["ts"]),
                                            "tp": float(rec["tp_px"]), "sl": float(rec["sl_px"]), "horizon": float(rec["horizon"]),
                                            "prob": rec.get("prob")}
                elif rec.get("event") == "EXIT":
                    self.positions.pop(code, None)
                    self.closed.append(float(rec.get("net_pct") or 0.0))
        except Exception as exc:
            self.log(f"{TAG} restore failed ({type(exc).__name__}: {exc}) - starting the day empty")
            self.positions, self.entries_today, self.closed = {}, 0, []

    # ---- helpers ----------------------------------------------------------------------------
    def _label(self) -> dict:
        lab = {}
        try:
            lab = dict((self.model.meta or {}).get("label") or {})
        except Exception:
            lab = {}
        return {k: float(lab.get(k) or DEFAULT_LABEL[k]) for k in DEFAULT_LABEL}

    def _threshold(self) -> float | None:
        if self.min_prob > 0:
            return self.min_prob
        try:
            t = (self.model.meta or {}).get("threshold")
        except Exception:
            t = None
        return float(t) if r011.valid_prob(t) else None

    def _append(self, code: str, raw: pd.DataFrame) -> pd.DataFrame:
        old = self.bars.get(code)
        merged = raw if old is None or old.empty else pd.concat([old, raw[raw.index > old.index[-1]]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.bars[code] = merged
        return merged

    def _prev_close_of(self, code: str) -> float:
        if code not in self._prev_close:
            val = float("nan")
            try:
                prev_dir = r011.find_prev_data_dir(self.data_root, self.date_str)
                files = sorted(prev_dir.glob(f"{code}_*_daily.csv")) if prev_dir is not None else []
                if files:
                    d = pd.read_csv(files[0], encoding="utf-8-sig", dtype={"date": str})
                    d = d[d["date"].astype(str).str.slice(0, 8) < self.date_str]
                    if not d.empty:
                        val = float(pd.to_numeric(d["close"], errors="coerce").iloc[-1])
            except Exception:
                val = float("nan")
            self._prev_close[code] = val
        return self._prev_close[code]

    # ---- main entry point -------------------------------------------------------------------
    def on_symbol(self, code: str, name: str, now: datetime, frame_1m: pd.DataFrame | None, live_price: float) -> None:
        try:
            self._on_symbol(str(code).zfill(6), name, now, frame_1m, float(live_price))
        except Exception as exc:  # never break the live loop
            self.log(f"{TAG} {code} error: {type(exc).__name__}: {exc}")

    def _on_symbol(self, code: str, name: str, now: datetime, frame_1m, price: float) -> None:
        if not (price > 0):
            return
        was_open = code in self.positions
        if was_open:
            self._check_exit(code, now, price)
        if frame_1m is None or len(frame_1m) == 0:
            return
        raw = r011.session_raw_1min(frame_1m, now.date())
        if raw is None or raw.empty:
            return
        store = self._append(code, raw)
        key = store.index[-1]
        if self.last_bar.get(code) == key:
            return  # no new completed bar
        self.last_bar[code] = key
        # a symbol that was held at the start of this poll is not bought on this bar (also right after its sale: the
        # next entry needs a new bar, like the simulator where a position blocks the symbol until it is closed)
        if was_open or len(store) < r011.MIN_SESSION_BARS:
            return
        t = pd.Timestamp(key).time()
        if t < ENTRY_START or t > ENTRY_END or now.time() > ENTRY_END:
            return
        if len(self.positions) >= self.max_positions or self.entries_today >= self.max_entries:
            return
        self._check_entry(code, name, now, price, store)

    def _check_entry(self, code: str, name: str, now: datetime, price: float, store: pd.DataFrame) -> None:
        m1_all = r011.fast_indicators(store)
        last = m1_all.iloc[-1]
        if not float(last["close"]) > float(last["BB_MIDDLE"]):   # every trigger path needs the close above the 1m BB mid
            return
        m3_all = r011.completed_3min(m1_all)
        info: dict = {}
        ok, reason = r011.hybrid_trigger_with_context(m1_all.iloc[-TRIGGER_CACHE_BARS:], m3_all, float(last["close"]), info)
        if not ok:
            return
        frames = r011.feature_frames(store)
        if frames is None:
            return   # history window incomplete (late start): no prediction, no entry
        threshold = self._threshold()
        if threshold is None:
            return
        if code not in self._daily:
            self._daily[code] = r011.load_daily_features(self.data_root, code, self.date_str)
        first = store.index[0]
        full_session = pd.Timestamp(first).time() <= dt_time(9, 2)
        day_open = float(store["open"].iloc[0]) if full_session else float("nan")
        vol = store["volume"].astype(float)
        vwap = float((store["close"].astype(float) * vol).sum() / vol.sum()) if full_session and vol.sum() > 0 else float("nan")
        feats = r011.compute_features(frames[0], frames[1], pd.Timestamp(store.index[-1]), float(last["close"]),
                                      self._prev_close_of(code), day_open, vwap, self._daily[code], self.config,
                                      trigger_info=info)
        prob, meta = self.model.predict_with_meta(feats)
        if prob is None or not r011.valid_prob(prob) or prob < threshold:
            return
        label = self._label()
        entry = price + krx_tick_size(price)
        pos = {"name": name, "entry": entry, "ts": pd.Timestamp(now), "tp": entry * (1 + label["tp"] / 100.0),
               "sl": entry * (1 - label["sl"] / 100.0), "horizon": label["horizon"], "prob": float(prob)}
        self.positions[code] = pos
        self.entries_today += 1
        self._emit({"event": "ENTRY", "ts": now.isoformat(timespec="seconds"), "code": code, "name": name, "entry": entry,
                    "price": price, "prob": float(prob), "threshold": threshold, "trigger": reason, "tp_px": pos["tp"],
                    "sl_px": pos["sl"], "horizon": label["horizon"], "model_id": (meta or {}).get("model_id")})
        self.log(f"{TAG} {code}_{name} | PAPER BUY {entry:,.0f} p={prob:.3f} thr={threshold:.3f} {reason} "
                 f"| tp {pos['tp']:,.0f} sl {pos['sl']:,.0f} {label['horizon']:.0f}min | open={len(self.positions)} "
                 f"entries={self.entries_today}/{self.max_entries}")

    def _check_exit(self, code: str, now: datetime, price: float) -> None:
        pos = self.positions[code]
        held_min = (pd.Timestamp(now) - pos["ts"]).total_seconds() / 60.0
        reason, exit_px = None, None
        if price >= pos["tp"]:
            reason, exit_px = "tp", pos["tp"]                      # a resting sell at the target
        elif price <= pos["sl"]:
            reason, exit_px = "sl", price - krx_tick_size(price)
        elif now.time() >= FORCE_EXIT:
            reason, exit_px = "eod", price - krx_tick_size(price)
        elif held_min >= pos["horizon"]:
            reason, exit_px = "time", price - krx_tick_size(price)
        if reason is None:
            return
        net = net_pct(pos["entry"], exit_px)
        self.positions.pop(code, None)
        self.closed.append(net)
        wins = sum(1 for x in self.closed if x > 0)
        self._emit({"event": "EXIT", "ts": now.isoformat(timespec="seconds"), "code": code, "name": pos.get("name"),
                    "reason": reason, "entry": pos["entry"], "exit": exit_px, "net_pct": net, "held_min": round(held_min, 1),
                    "prob": pos.get("prob")})
        self.log(f"{TAG} {code}_{pos.get('name')} | PAPER SELL {exit_px:,.0f} {reason} net={net:+.2f}% held={held_min:.0f}min "
                 f"| day: {len(self.closed)} trades {wins} wins sum={sum(self.closed):+.2f}%")
