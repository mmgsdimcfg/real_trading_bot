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

Collector symbols (2026-11-19, r001 EDGE_ONLY_COLLECTOR): the tick collector's symbols outside the watchlist (g011 scan fill /
extra codes) get the same policy. r003 fetches no KIS minutes for them, so CollectorTickFeed builds their regular-session 1m bars
and the live price from the day's KRX tick file (runtime/data/ticks/YYYYMMDD/<code>.csv, read incrementally) - no extra KIS call.
Their limits are separate (EDGE_ONLY_COLLECTOR_MAX_POSITIONS / _MAX_ENTRIES_PER_DAY); records carry "group": "collector"
(the watchlist's "watch"; a record without group is "watch").

Output: runtime/data/live_runtime/edge_only_YYYYMMDD.jsonl (ENTRY / EXIT records) + "[EDGE ONLY   ]" log lines.
Status (2026-11-19, AutoBotH 가상 매매 tab): runtime/data/live_runtime/edge_only_status_YYYYMMDD.json, replaced atomically at most
every STATUS_EVERY_S by write_status() - every symbol the runner looks at (group, price, last bar, the outcome of its last
evaluation = stage, probability) + the group limits and the model threshold.
Runs the same on Linux (Raspberry Pi) and Windows. Each host uses its own model (runtime/edge_ai, trained by the
post-close job from that host's data folder).

Update log:
- [2026-11-20] type=feat owner=claude
    summary: 사용자 결정(가상 매매 검토 3차, Codex 합의) - log-only 모드(r001 EDGE_ONLY_LOG_ONLY): 신규 가상 매수를 하지 않고 트리거가 뜬
      봉마다 CANDIDATE 한 줄(55개 피처, 점수·컷·모델, 막힌 사유 fund/sl_today, 가격·틱, 라벨 설정)을 edge_only_candidates_<날짜>.jsonl
      에 남긴다(id 로 재시작 중복 제거). 기존 가상 보유의 청산(tp/sl/15:15/시간, poll_orphans)은 그대로. 상태 파일에 log_only.
      근거: 65일 OOS 재생·회귀·상대강도/평균회귀 실험 모두 비용(왕복 약 0.5%)을 넘지 못함 - 모델은 방향이 아니라 변동폭을 순위화.
    impact: live paper only (주문 없음)
    compatibility: backward-compatible (log_only=False 면 종전과 동일; 롤백: r001 EDGE_ONLY_LOG_ONLY=False)
- [2026-11-20] type=fix+feat owner=claude
    summary: 가상 매매 분석(20261120, Codex 검토) 후 사용자 결정 0/2/3번.
      (0) 버그: r003 이 on_symbol 을 부르지 않은 가상 보유 종목(감시 이탈, 직접 매수로 [HOLD SKIP], 재시작 후 목록 밖)이 청산되지
          않던 문제 - poll_orphans(): 그 턴에 평가 안 된 가상 보유를 틱 파일(없으면 마지막 가격)로 tp/sl/15:15/시간 청산.
          20261120 443060 은 14:41 직접 매수 뒤 목표가(14:53)와 15:15 를 지나 미청산.
      (2) ETF·ETN 가상 매수 제외 (r011.is_fund_product, r001 EDGE_ONLY_EXCLUDE_FUNDS, stage "fund").
      (3) 당일 손절(sl) 종목 재매수 금지 (r001 EDGE_ONLY_NO_REENTRY_AFTER_SL, stage "sl_today", 재시작 시 파일에서 복원).
    impact: live paper only (주문 없음) - r003 재시작 후
    compatibility: backward-compatible (기록 형식 불변, 새 stage 2개; 롤백: r001 두 스위치 False)
- [2026-11-19] type=feat owner=claude
    summary: 사용자 요청(가상 매매 탭에 대상 종목 상태 표시) - 종목별 상태(구분, 현재가, 마지막 봉, 마지막 판단 결과 stage, 확률,
      보유 중이면 매수가/목표/손절) 를 edge_only_status_<날짜>.json 으로 STATUS_EVERY_S 마다 원자적 교체(write_status). r003 이 호출.
    impact: live (파일 하나 추가 - 매매 판단 무변경)
    compatibility: backward-compatible
- [2026-11-19] type=feat owner=claude
    summary: 사용자 요청(틱 수집기 종목을 가상 매매에 포함) - CollectorTickFeed(틱 파일 -> 확정 1분봉 + 현재가, 증분 읽기,
      acml_vol 로 재접속 중복 제거, 정규장 체결만) + EdgeOnlyRunner.poll_collector(). 그룹별 상한(watch / collector),
      ENTRY/EXIT 에 "group" 기록, 재시작 복원도 그룹별.
    impact: live (가상 매매 기록만 - 실제 매수/매도 판정 무변경, KIS 호출 추가 없음)
    compatibility: backward-compatible (group 없는 예전 기록 = watch, 감시 종목 상한·동작 동일)
- [2026-11-19] type=fix owner=claude
    summary: _threshold() 가 모델을 읽기 전(meta = {})에 기준값을 찾아 None -> 예측 자체를 하지 않아, EDGE_ONLY_MIN_PROB=0(2026-11-12~)
      이후 가상 매수가 한 건도 없었다(edge_only_*.jsonl 파일 없음). 기준값을 읽기 전에 model.maybe_reload().
    impact: live (가상 매매가 실제로 기록되기 시작함 - 실매매 무관)
    compatibility: backward-compatible
- [2026-11-12] type=feat owner=claude
    summary: 신규 - Edge AI 메타 모델만으로 매매하는 방식의 가상 매매(paper) 러너. r003에서 EDGE_ONLY_MODE="paper"일 때만
      호출, 주문 없음. 과거 데이터 검증은 g029_edge_only_sim.py.
    impact: live (기록만 - 실제 매수/매도 판정 무변경)
    compatibility: backward-compatible (기본 EDGE_ONLY_MODE="off")
"""
from __future__ import annotations

import json
import re
from datetime import datetime, time as dt_time, timedelta
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
GROUPS = ("watch", "collector")
STATUS_EVERY_S = 5.0          # edge_only_status_<date>.json at most this often
STATUS_STALE_S = 600.0        # a symbol not looked at for this long (left the watchlist / collector) leaves the status
# outcome of the last evaluation of a symbol (AutoBotH shows these)
STAGES = ("wait_bars", "outside_window", "limit", "held", "below_mid", "no_trigger", "no_history", "no_model", "below_threshold",
          "bought", "sold", "fund", "sl_today", "candidate")   # [2026-11-20] fund = ETF/ETN not bought, sl_today = stopped out today
GROUP_KO = {"watch": "", "collector": " [수집기]"}
_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")


def net_pct(entry: float, exit_px: float) -> float:
    """Net % of a round trip after commission (both sides) and sell tax."""
    cost = entry * FEE_RATE + exit_px * (FEE_RATE + SELL_TAX_RATE)
    return (exit_px - entry - cost) / entry * 100.0


class EdgeOnlyRunner:
    def __init__(self, model_dir: Path, data_root: Path, out_dir: Path, date_str: str, log: Callable[[str], None], config,
                 min_prob: float = 0.0, max_positions: int = 3, max_entries: int = 10, model=None,
                 collector_max_positions: int = 0, collector_max_entries: int = 0,
                 exclude_funds: bool = True, no_reentry_after_sl: bool = True, log_only: bool = False):
        self.model = model if model is not None else r011.EdgeAIModel(Path(model_dir))
        self.data_root = Path(data_root)
        self.out_path = Path(out_dir) / f"edge_only_{date_str}.jsonl"
        self.date_str = date_str
        self.log = log
        self.config = config
        self.min_prob = float(min_prob or 0.0)
        self.max_positions = int(max_positions)
        self.max_entries = int(max_entries)
        # [2026-11-19] per-group limits: the watchlist (r003's symbols) and the collector-only symbols count separately
        self.limits = {"watch": (self.max_positions, self.max_entries),
                       "collector": (int(collector_max_positions), int(collector_max_entries))}
        self.bars: dict[str, pd.DataFrame] = {}
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.positions: dict[str, dict] = {}
        self.entries = {g: 0 for g in GROUPS}
        self.closed: list[float] = []
        self._names: dict[str, str] = {}
        self._collector_next: datetime | None = None
        # [2026-11-19] per-symbol status for the app (write_status)
        self.sym_state: dict[str, dict] = {}
        self.status_path = Path(out_dir) / f"edge_only_status_{date_str}.json"
        self._status_next: datetime | None = None
        self._daily: dict[str, dict] = {}
        self._prev_close: dict[str, float] = {}
        # [2026-11-20] user decision (paper review 20261120): no ETF / ETN buys (r011.is_fund_product), and a symbol stopped out (sl)
        # today is not bought again the same day (5 such re-entries on 11/20, all sl)
        self.exclude_funds = bool(exclude_funds)
        self.no_reentry_after_sl = bool(no_reentry_after_sl)
        self.sl_today: set[str] = set()
        self._eval_at: dict[str, datetime] = {}   # wall clock of each symbol's last on_symbol (poll_orphans)
        # [2026-11-20] log-only (r001 EDGE_ONLY_LOG_ONLY, paper review round 3): no NEW paper buys; every trigger bar is written as a
        # CANDIDATE row (features, score, cut, block reason, price) to edge_only_candidates_<date>.jsonl for later studies.
        # Open positions keep their exits (tp / sl / 15:15 / time, poll_orphans).
        self.log_only = bool(log_only)
        self.cand_path = Path(out_dir) / f"edge_only_candidates_{date_str}.jsonl"
        self._cand_ids: set[str] = set()
        if self.log_only and self.cand_path.is_file():
            try:
                for line in self.cand_path.read_text(encoding="utf-8").splitlines():
                    self._cand_ids.add(str(json.loads(line).get("id") or ""))
            except Exception as exc:
                self.log(f"{TAG} candidate log restore failed ({type(exc).__name__}: {exc})")
        self._restore()

    @property
    def entries_today(self) -> int:
        """the watchlist's entries of the day (the original counter)"""
        return self.entries["watch"]

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
                    group = rec.get("group") if rec.get("group") in GROUPS else "watch"
                    self.entries[group] += 1
                    self.positions[code] = {"name": rec.get("name"), "entry": float(rec["entry"]), "ts": pd.Timestamp(rec["ts"]),
                                            "tp": float(rec["tp_px"]), "sl": float(rec["sl_px"]), "horizon": float(rec["horizon"]),
                                            "prob": rec.get("prob"), "group": group}
                elif rec.get("event") == "EXIT":
                    self.positions.pop(code, None)
                    self.closed.append(float(rec.get("net_pct") or 0.0))
                    if rec.get("reason") == "sl":
                        self.sl_today.add(code)
        except Exception as exc:
            self.log(f"{TAG} restore failed ({type(exc).__name__}: {exc}) - starting the day empty")
            self.positions, self.entries, self.closed = {}, {g: 0 for g in GROUPS}, []
            self.sl_today = set()

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
            # [2026-11-19] fix: the model (and its meta) is loaded lazily on the first prediction - read before it, the meta was
            # {} -> no threshold -> no prediction ever: with EDGE_ONLY_MIN_PROB = 0 (since 2026-11-12) nothing was paper-bought
            reload = getattr(self.model, "maybe_reload", None)
            if callable(reload):
                reload()
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
    def on_symbol(self, code: str, name: str, now: datetime, frame_1m: pd.DataFrame | None, live_price: float,
                  group: str = "watch") -> None:
        try:
            self._eval_at[str(code).zfill(6)] = datetime.now()
            self._on_symbol(str(code).zfill(6), name, now, frame_1m, float(live_price), group if group in GROUPS else "watch")
        except Exception as exc:  # never break the live loop
            self.log(f"{TAG} {code} error: {type(exc).__name__}: {exc}")

    def _open_count(self, group: str) -> int:
        return sum(1 for p in self.positions.values() if p.get("group", "watch") == group)

    def name_of(self, code: str) -> str:
        """symbol name from the previous data folder's <code>_<name>_daily.csv (collector symbols have no WATCH line)"""
        if code not in self._names:
            name = ""
            try:
                prev_dir = r011.find_prev_data_dir(self.data_root, self.date_str)
                files = sorted(prev_dir.glob(f"{code}_*_daily.csv")) if prev_dir is not None else []
                if files:
                    name = files[0].name[len(code) + 1:-len("_daily.csv")]
            except Exception:
                name = ""
            self._names[code] = name
        return self._names[code]

    def poll_collector(self, feed: "CollectorTickFeed", now: datetime, exclude: set[str], every_s: float = 5.0) -> int:
        """[2026-11-19] the collector-only symbols (tick files of the day minus `exclude` = r003's watched / held symbols) through
        the same policy as group "collector". At most once per every_s. Returns how many symbols were looked at."""
        if self._collector_next is not None and now < self._collector_next:
            return 0
        self._collector_next = now + timedelta(seconds=max(0.0, float(every_s)))
        n = 0
        try:
            codes = [c for c in feed.codes() if c not in exclude]
        except Exception as exc:
            self.log(f"{TAG} collector list error: {type(exc).__name__}: {exc}")
            return 0
        held = [c for c, p in self.positions.items() if p.get("group") == "collector" and c not in exclude]
        for code in dict.fromkeys(codes + held):
            try:
                frame, price = feed.poll(code, now)
            except Exception as exc:
                self.log(f"{TAG} {code} collector feed error: {type(exc).__name__}: {exc}")
                continue
            if price is None or not price > 0:
                continue
            n += 1
            self.on_symbol(code, self.name_of(code), now, frame, price, group="collector")
        return n

    def poll_orphans(self, feed: "CollectorTickFeed | None", now: datetime, since: datetime) -> int:
        """[2026-11-20] Open paper positions that r003 did not hand to on_symbol this turn (the symbol left the watchlist, was
        bought by hand -> [HOLD SKIP], or r003 restarted without it): exits (tp / sl / 15:15 / time) from the day's KRX tick file,
        else from the last price seen. Bug 20261120: 443060 bought by hand at 14:41 kept its paper position past its target and
        15:15. `since` = start of r003's turn: a position looked at after it (symbol loop or poll_collector) is skipped.
        Returns how many were checked."""
        n = 0
        for code in [c for c in list(self.positions) if not (self._eval_at.get(c) and self._eval_at[c] > since)]:
            pos = self.positions.get(code)
            if pos is None:
                continue
            frame, price = None, None
            if feed is not None:
                try:
                    frame, price = feed.poll(code, now)
                except Exception as exc:
                    self.log(f"{TAG} {code} orphan feed error: {type(exc).__name__}: {exc}")
            if price is None or not price > 0:
                price = (self.sym_state.get(code) or {}).get("price")
            if price is None or not float(price) > 0:
                continue
            n += 1
            self.on_symbol(code, pos.get("name") or self.name_of(code), now, frame, float(price), group=pos.get("group", "watch"))
        return n

    def write_status(self, now: datetime, force: bool = False) -> bool:
        """[2026-11-19] edge_only_status_<date>.json for the app, at most every STATUS_EVERY_S (force: now). Never raises."""
        return _write_status(self, now, force)

    def _stage(self, code: str, stage: str, prob: float | None = None) -> None:
        st = self.sym_state.get(code)
        if st is not None:
            st["stage"] = stage
            st["prob"] = None if prob is None else round(float(prob), 4)

    def _on_symbol(self, code: str, name: str, now: datetime, frame_1m, price: float, group: str = "watch") -> None:
        if not (price > 0):
            return
        st = self.sym_state.setdefault(code, {"stage": "wait_bars", "prob": None, "bar": None, "bars": 0})
        st.update(name=name, group=group, price=price, at=now.isoformat(timespec="seconds"))
        was_open = code in self.positions
        if was_open:
            self._check_exit(code, now, price)
            if code not in self.positions:
                self._stage(code, "sold")
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
        st.update(bar=pd.Timestamp(key).strftime("%H:%M"), bars=int(len(store)))
        # a symbol that was held at the start of this poll is not bought on this bar (also right after its sale: the
        # next entry needs a new bar, like the simulator where a position blocks the symbol until it is closed)
        if was_open:
            if code in self.positions:
                self._stage(code, "held", self.positions[code].get("prob"))
            return
        if len(store) < r011.MIN_SESSION_BARS:
            self._stage(code, "wait_bars")
            return
        t = pd.Timestamp(key).time()
        if t < ENTRY_START or t > ENTRY_END or now.time() > ENTRY_END:
            self._stage(code, "outside_window")
            return
        block = None
        if self.exclude_funds and r011.is_fund_product(name):
            block = "fund"
        elif self.no_reentry_after_sl and code in self.sl_today:
            block = "sl_today"
        if self.log_only:                      # no new positions: limits do not apply, a block is only recorded
            self._check_entry(code, name, now, price, store, group, block=block)
            return
        if block:
            self._stage(code, block)
            return
        max_pos, max_ent = self.limits[group]
        if self._open_count(group) >= max_pos or self.entries[group] >= max_ent:
            self._stage(code, "limit")
            return
        self._check_entry(code, name, now, price, store, group)

    def _candidate(self, code: str, name: str, now: datetime, price: float, store: pd.DataFrame, group: str, trigger: str,
                   info: dict, feats: dict | None, prob, threshold, meta: dict | None, block: str | None, reason: str) -> None:
        """[2026-11-20] log-only: one CANDIDATE row per trigger bar (id = date:code:bar:model - a restart does not repeat it)"""
        bar = pd.Timestamp(store.index[-1])
        model_id = (meta or {}).get("model_id") or (getattr(self.model, "meta", None) or {}).get("model_id")
        cid = f"{self.date_str}:{code}:{bar.strftime('%H%M')}:{model_id}"
        if cid in self._cand_ids:
            return
        self._cand_ids.add(cid)
        tick = krx_tick_size(price)
        p = float(prob) if prob is not None and r011.valid_prob(prob) else None
        rec = {"schema_version": 1, "event": "CANDIDATE", "id": cid, "date": self.date_str, "code": code, "name": name, "group": group,
               "bar": bar.isoformat(), "ts": now.isoformat(timespec="seconds"), "trigger": trigger,
               "trigger_info": {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, bool)) or v is None},
               "feature_version": r011.FEATURE_VERSION, "features": feats, "prob": p, "threshold": threshold, "model_id": model_id,
               "would_buy": bool(p is not None and threshold is not None and p >= threshold and not block), "block": block,
               "reason": reason, "price": price, "price_source": "live", "tick": tick, "tick_pct": tick / price * 100 if price else None,
               "label": self._label()}
        try:
            self.cand_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cand_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            self.log(f"{TAG} candidate log write failed ({type(exc).__name__}: {exc})")
        self._stage(code, "candidate", p)

    def _check_entry(self, code: str, name: str, now: datetime, price: float, store: pd.DataFrame, group: str = "watch",
                     block: str | None = None) -> None:
        m1_all = r011.fast_indicators(store)
        last = m1_all.iloc[-1]
        if not float(last["close"]) > float(last["BB_MIDDLE"]):   # every trigger path needs the close above the 1m BB mid
            self._stage(code, "below_mid")
            return
        m3_all = r011.completed_3min(m1_all)
        info: dict = {}
        ok, reason = r011.hybrid_trigger_with_context(m1_all.iloc[-TRIGGER_CACHE_BARS:], m3_all, float(last["close"]), info)
        if not ok:
            self._stage(code, "no_trigger")
            return
        frames = r011.feature_frames(store)
        if frames is None:
            if self.log_only:
                self._candidate(code, name, now, price, store, group, reason, info, None, None, None, None, block, "no_history")
                return
            self._stage(code, "no_history")
            return   # history window incomplete (late start): no prediction, no entry
        threshold = self._threshold()
        if threshold is None:
            if self.log_only:
                self._candidate(code, name, now, price, store, group, reason, info, None, None, None, None, block, "no_model")
                return
            self._stage(code, "no_model")
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
        if self.log_only:
            ok_p = prob is not None and r011.valid_prob(prob)
            why = "no_model" if not ok_p else (block or ("below_threshold" if prob < threshold else "would_buy"))
            self._candidate(code, name, now, price, store, group, reason, info,
                            {k: (None if v is None or (isinstance(v, float) and v != v) else float(v)) for k, v in feats.items()},
                            prob, threshold, meta, block, why)
            return
        if prob is None or not r011.valid_prob(prob) or prob < threshold:
            self._stage(code, "no_model" if prob is None else "below_threshold", prob if r011.valid_prob(prob) else None)
            return
        self._stage(code, "bought", prob)
        label = self._label()
        entry = price + krx_tick_size(price)
        pos = {"name": name, "entry": entry, "ts": pd.Timestamp(now), "tp": entry * (1 + label["tp"] / 100.0),
               "sl": entry * (1 - label["sl"] / 100.0), "horizon": label["horizon"], "prob": float(prob), "group": group}
        self.positions[code] = pos
        self.entries[group] += 1
        self._emit({"event": "ENTRY", "ts": now.isoformat(timespec="seconds"), "code": code, "name": name, "entry": entry,
                    "price": price, "prob": float(prob), "threshold": threshold, "trigger": reason, "tp_px": pos["tp"],
                    "sl_px": pos["sl"], "horizon": label["horizon"], "model_id": (meta or {}).get("model_id"), "group": group})
        self.log(f"{TAG} {code}_{name}{GROUP_KO[group]} | PAPER BUY {entry:,.0f} p={prob:.3f} thr={threshold:.3f} {reason} "
                 f"| tp {pos['tp']:,.0f} sl {pos['sl']:,.0f} {label['horizon']:.0f}min | open={self._open_count(group)} "
                 f"entries={self.entries[group]}/{self.limits[group][1]}")

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
        if reason == "sl":
            self.sl_today.add(code)
        self.closed.append(net)
        wins = sum(1 for x in self.closed if x > 0)
        group = pos.get("group", "watch")
        self._emit({"event": "EXIT", "ts": now.isoformat(timespec="seconds"), "code": code, "name": pos.get("name"),
                    "reason": reason, "entry": pos["entry"], "exit": exit_px, "net_pct": net, "held_min": round(held_min, 1),
                    "prob": pos.get("prob"), "group": group})
        self.log(f"{TAG} {code}_{pos.get('name')}{GROUP_KO.get(group, '')} | PAPER SELL {exit_px:,.0f} {reason} net={net:+.2f}% held={held_min:.0f}min "
                 f"| day: {len(self.closed)} trades {wins} wins sum={sum(self.closed):+.2f}%")


class CollectorTickFeed:
    """[2026-11-19] Confirmed 1m bars + the last price of a symbol from the day's KRX tick file of the collector (g011).

    - incremental: each poll reads only the bytes appended since the last one (a torn last line waits for the next poll)
    - regular-session prints only (session column "regular"; files without it: 09:00 <= time < 15:20)
    - reconnect replays are dropped by acml_vol (it only grows)
    - bars are end-labeled like r003's 1m frame (09:00:xx prints -> the 09:01 bar) and handed out only once complete
      (end <= now - confirm_delay_s, floored to the minute), and only when a new one completed since the last poll
      (None otherwise - the runner then checks exits with the price)
    """

    def __init__(self, ticks_dir: Path, date_str: str, confirm_delay_s: float = 2.0):
        self.dir = Path(ticks_dir)
        self.date_str = date_str
        self.confirm_delay_s = float(confirm_delay_s)
        self.state: dict[str, dict] = {}

    def codes(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.csv") if _CODE_RE.fullmatch(p.stem))

    @staticmethod
    def _new() -> dict:
        return {"off": 0, "rest": b"", "cols": None, "acml": -1.0, "buckets": {}, "price": None, "done": None}

    def _read(self, code: str, st: dict) -> None:
        path = self.dir / f"{code}.csv"
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < st["off"]:          # replaced / truncated file: start over
            st.clear()
            st.update(self._new())
        if size == st["off"]:
            return
        with path.open("rb") as fh:
            fh.seek(st["off"])
            chunk = fh.read(size - st["off"])
        st["off"] += len(chunk)
        data = st["rest"] + chunk
        cut = data.rfind(b"\n")
        if cut < 0:
            st["rest"] = data
            return
        st["rest"] = data[cut + 1:]
        for raw in data[:cut].split(b"\n"):
            line = raw.decode("utf-8", "replace").rstrip("\r")
            if not line:
                continue
            fields = line.split(",")
            if st["cols"] is None:
                st["cols"] = {c: i for i, c in enumerate(fields)}
                st["width"] = len(fields)
                continue
            self._add(st, fields)

    def _add(self, st: dict, fields: list[str]) -> None:
        cols = st["cols"]
        if len(fields) != st["width"]:
            return            # torn / foreign row (r009 rule)
        try:
            t = fields[cols["trade_time"]].strip().zfill(6)
            price = float(fields[cols["price"]])
            qty = float(fields[cols["cntg_vol"]] or 0)
            acml = float(fields[cols["acml_vol"]])
            day = fields[cols["bsop_date"]].strip() if "bsop_date" in cols else self.date_str
        except (KeyError, ValueError):
            return
        if day != self.date_str or not price > 0 or not t.isdigit():
            return
        sess = fields[cols["session"]].strip() if "session" in cols else ("regular" if "090000" <= t < "152000" else "")
        if sess != "regular":
            return
        if acml <= st["acml"]:
            return            # replayed print after a reconnect
        st["acml"] = acml
        st["price"] = price
        start = t[:4]         # HHMM of the minute the print falls in
        b = st["buckets"].get(start)
        if b is None:
            st["buckets"][start] = [price, price, price, price, qty]
        else:
            b[1] = max(b[1], price)
            b[2] = min(b[2], price)
            b[3] = price
            b[4] += qty

    def poll(self, code: str, now: datetime) -> tuple[pd.DataFrame | None, float | None]:
        st = self.state.setdefault(code, self._new())
        self._read(code, st)
        if not st["buckets"]:
            return None, st["price"]
        cutoff = pd.Timestamp(now - timedelta(seconds=self.confirm_delay_s)).floor("1min")
        base = datetime.strptime(self.date_str, "%Y%m%d")
        ends = {k: pd.Timestamp(base + timedelta(hours=int(k[:2]), minutes=int(k[2:]) + 1)) for k in st["buckets"]}
        done = [k for k in sorted(st["buckets"]) if ends[k] <= cutoff]
        if not done or done[-1] == st["done"]:
            return None, st["price"]
        st["done"] = done[-1]
        frame = pd.DataFrame([st["buckets"][k] for k in done], columns=["open", "high", "low", "close", "volume"],
                             index=pd.DatetimeIndex([ends[k] for k in done]))
        return frame, st["price"]


def _write_status_file(runner: "EdgeOnlyRunner", now: datetime) -> None:
    """edge_only_status_<date>.json: atomic replace (tmp file + os.replace) so the app never reads half a file."""
    import os
    import tempfile
    cutoff = now - timedelta(seconds=STATUS_STALE_S)
    symbols = []
    for code, st in list(runner.sym_state.items()):
        pos = runner.positions.get(code)
        seen = datetime.fromisoformat(st["at"]) if st.get("at") else now
        if pos is None and seen < cutoff:
            runner.sym_state.pop(code, None)      # left the watchlist / the collector
            continue
        row = {"code": code, "name": st.get("name") or "", "group": st.get("group", "watch"), "price": st.get("price"),
               "bar": st.get("bar"), "bars": st.get("bars", 0), "stage": st.get("stage"), "prob": st.get("prob"),
               "at": st.get("at")}
        if pos is not None:
            row["position"] = {"entry": pos["entry"], "tp": pos["tp"], "sl": pos["sl"], "ts": pd.Timestamp(pos["ts"]).isoformat(),
                               "prob": pos.get("prob")}
        symbols.append(row)
    meta = {}
    try:
        meta = dict(runner.model.meta or {})
    except Exception:
        meta = {}
    payload = {
        "asof": now.isoformat(timespec="seconds"), "date": runner.date_str, "log_only": bool(getattr(runner, "log_only", False)),
        "model_id": meta.get("model_id"), "threshold": runner._threshold(), "label": runner._label(),
        "entry_window": [ENTRY_START.strftime("%H:%M"), ENTRY_END.strftime("%H:%M")], "force_exit": FORCE_EXIT.strftime("%H:%M"),
        "groups": {g: {"open": runner._open_count(g), "max_positions": runner.limits[g][0], "entries": runner.entries[g],
                       "max_entries": runner.limits[g][1], "enabled": runner.limits[g][1] > 0} for g in GROUPS},
        "symbols": symbols,
    }
    runner.status_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".edge_only_status.", suffix=".tmp", dir=str(runner.status_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, default=str)
        os.replace(tmp, runner.status_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_status(runner: "EdgeOnlyRunner", now: datetime, force: bool = False) -> bool:
    """[2026-11-19] the status file at most every STATUS_EVERY_S (force: now). Never raises. True when written."""
    try:
        if not force and runner._status_next is not None and now < runner._status_next:
            return False
        runner._status_next = now + timedelta(seconds=STATUS_EVERY_S)
        _write_status_file(runner, now)
        return True
    except Exception as exc:
        try:
            runner.log(f"{TAG} status write error: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return False

