# -*- coding: utf-8 -*-
"""VWAP trade (user request 2026-11-27): the buy signal and its PAPER runner (virtual entries / exits on live data, NO orders).

The fourth way of trading beside the rule trade, the Edge AI trade and the rebound alert. Enabled by r001 VWAP_TRADE_MODE = "paper"
(default "off"). r003 calls the runner once per watched symbol per loop, next to the Edge AI paper runner (r022) - the rule trade and
its orders are untouched, and an exception here never reaches the trading loop.

Buy signal (all four, on a newly completed 1-minute bar of the regular session, 09:05 ~ 14:50)
  1 day direction   the last 1-minute close AND the live price are above the session VWAP
                    (VWAP = sum(close x volume) / sum(volume) of the day's regular 1-minute bars; needs bars from the open)
  2 short trend     1-minute MA5 is rising (MA5 > MA5 one bar ago) OR the Bollinger middle (SMA20) is rising
  3 trigger         60-tick bars: MA5 crossed above MA15 within VWAP_TRADE_TICK_CROSS_MAX_AGE_SECONDS, and the recent print density
                    is at least VWAP_TRADE_MIN_PRINTS_PER_MIN (r013 TickTriggerFeed on the tick collector's files)
  4 safety          chase: live price <= tick cross close x (1 + VWAP_TRADE_CHASE_MAX_PCT %) and <= VWAP x (1 + VWAP_TRADE_MAX_VWAP_GAP_PCT %)
                    drop : live price >= last 1-minute close x (1 - VWAP_TRADE_LIVE_DROP_MAX_PCT %)
                    fresh: the last 1-minute bar ended <= VWAP_TRADE_BAR_STALE_SECONDS ago; the tick feed's last print is
                           <= VWAP_TRADE_TICK_FEED_STALE_SECONDS old (r013 abstains otherwise)
Sell (user decision): + VWAP_TRADE_TP_PCT % / - VWAP_TRADE_SL_PCT % / VWAP_TRADE_HORIZON_MIN minutes / the 15:20 session close,
whichever comes first - the paper runner's exits (r022).
Fill (virtual, as r022): buy = live price + 1 tick, take profit = the target, other sells = live price - 1 tick; net % includes the
commission (0.015 % per side) and the sell tax (0.20 %).

Files: runtime/data/live_runtime/vwap_paper_YYYYMMDD.jsonl (ENTRY / EXIT) and vwap_paper_status_YYYYMMDD.json (per-symbol stage for the
app), same record layout as r022's edge_only files. Log lines "[VWAP PAPER  ]".
Earlier studies found no forward edge in "price above VWAP" (vwap_indicator_review_20261126.md 3) - this runs on paper first.

Update log:
- [2026-11-27] type=feat owner=claude
    summary: 신규 - VWAP 매매 신호(vwap_signal)와 가상 매매 러너(VwapPaperRunner, r022 EdgeOnlyRunner 재사용). 주문 없음.
    impact: paper (r001 VWAP_TRADE_MODE="paper" 일 때만, 기본 off). 규칙 매매 / Edge AI 매매 판정 불변
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

import r022_edge_only as r022
from r008_strategy_v2 import krx_tick_size

TAG = "  [VWAP PAPER  ]"
FILE_PREFIX = "vwap_paper"
MODEL_ID = "vwap_rule_v1"
FULL_SESSION_FIRST_BAR = pd.Timestamp("09:02").time()   # the store must start at the open or the VWAP is not the day's VWAP
MIN_BARS = 21                                            # SMA20 now and one bar ago


@dataclass(frozen=True)
class VwapParams:
    chase_max_pct: float = 0.5
    max_vwap_gap_pct: float = 2.0
    live_drop_max_pct: float = 0.4
    bar_stale_s: float = 90.0
    ma_fast: int = 5
    bb_period: int = 20

    @classmethod
    def from_config(cls, cfg) -> "VwapParams":
        g = lambda k, d: float(getattr(cfg, k, d))   # noqa: E731
        return cls(chase_max_pct=g("VWAP_TRADE_CHASE_MAX_PCT", 0.5), max_vwap_gap_pct=g("VWAP_TRADE_MAX_VWAP_GAP_PCT", 2.0),
                   live_drop_max_pct=g("VWAP_TRADE_LIVE_DROP_MAX_PCT", 0.4), bar_stale_s=g("VWAP_TRADE_BAR_STALE_SECONDS", 90.0))


def session_vwap(raw: pd.DataFrame) -> float:
    """VWAP of the bars given (close x volume / volume); nan without volume"""
    vol = raw["volume"].astype(float)
    total = float(vol.sum())
    return float((raw["close"].astype(float) * vol).sum() / total) if total > 0 else float("nan")


def vwap_signal(raw: pd.DataFrame, price: float, now: datetime, tick, p: VwapParams,
                vwap: float | None = None) -> tuple[bool, str, dict]:
    """(buy?, stage or reason, info). `raw` = the day's regular 1-minute bars up to the last completed one (index = bar end),
    `tick` = r013 TickTriggerResult of the symbol (None = no tick feed). The stage names are what the app shows:
    no_history / below_vwap / no_trend / no_tick_cross / chase / price_drop / stale / signal.
    `vwap`: the day's VWAP when the caller has it from elsewhere (r003's 1-minute cache holds 200 bars, so the live path passes
    the day reference price) - then `raw` need not start at the open."""
    info: dict = {}
    if raw is None or len(raw) < MIN_BARS or (vwap is None and pd.Timestamp(raw.index[0]).time() > FULL_SESSION_FIRST_BAR):
        return False, "no_history", info
    close = raw["close"].astype(float)
    last = float(close.iloc[-1])
    age = (pd.Timestamp(now) - pd.Timestamp(raw.index[-1])).total_seconds()
    info.update(bar=str(raw.index[-1]), bar_age_s=round(age, 1))
    if age < -5 or age > p.bar_stale_s:
        return False, "stale", info
    vwap = session_vwap(raw) if vwap is None else float(vwap)
    info.update(vwap=vwap, vwap_gap_pct=(price / vwap - 1) * 100 if vwap == vwap and vwap > 0 else None)
    if not (vwap == vwap and vwap > 0) or not (last > vwap and price > vwap):
        return False, "below_vwap", info
    ma = close.rolling(p.ma_fast).mean()
    mid = close.rolling(p.bb_period).mean()
    ma_up = bool(ma.iloc[-1] > ma.iloc[-2])
    mid_up = bool(mid.iloc[-1] > mid.iloc[-2])
    info.update(ma5_up=ma_up, bb_mid_up=mid_up)
    if not (ma_up or mid_up):
        return False, "no_trend", info
    if tick is None or not getattr(tick, "ok", False):
        info["tick"] = str(getattr(tick, "reason", "") or "no tick feed")[:160]
        return False, "no_tick_cross", info
    cross = next((c for c in tick.crosses.values() if c.ok), None)
    cross_close = float(getattr(cross, "cross_close", float("nan"))) if cross is not None else float("nan")
    info.update(tick=tick.label, tick_cross=str(tick.cross_time), cross_close=cross_close if cross_close == cross_close else None)
    if cross_close == cross_close and cross_close > 0 and price > cross_close * (1 + p.chase_max_pct / 100.0):
        return False, "chase", info
    if price > vwap * (1 + p.max_vwap_gap_pct / 100.0):
        return False, "chase", info
    if price < last * (1 - p.live_drop_max_pct / 100.0):
        return False, "price_drop", info
    return True, "signal", info


def trace_fields(stage: str, info: dict | None) -> dict:
    """[2026-11-27] what the timeline trace (r030) keeps of one vwap_signal answer: the values the decision used, as they are"""
    i = info or {}
    out = {"stage": str(stage)}
    for src, dst in (("vwap", "vwap"), ("vwap_gap_pct", "gap"), ("bar_age_s", "bar_age"), ("tick", "tick"),
                     ("tick_cross", "tick_cross"), ("cross_close", "cross_close"), ("ma5_up", "ma5_up"), ("bb_mid_up", "mid_up")):
        if i.get(src) is not None:
            out[dst] = i[src]
    return out


class _RuleModel:
    """stands in for the Edge AI model object the base runner reads (meta only; nothing is predicted)"""
    meta = {"model_id": MODEL_ID}
    error = ""

    def maybe_reload(self) -> None:
        return None


class VwapPaperRunner(r022.EdgeOnlyRunner):
    """r022's paper book (store, limits, exits, session close, status file, restore) with the VWAP signal as the entry"""
    FILE_PREFIX = FILE_PREFIX
    TRACE_STRATEGY = "vwap"

    def __init__(self, out_dir: Path, date_str: str, log: Callable[[str], None], params: VwapParams,
                 tick_eval: Callable[[str, datetime], object] | None, max_positions: int = 3, max_entries: int = 20,
                 tp_pct: float = 3.0, sl_pct: float = 1.27, horizon_min: float = 300, exclude_funds: bool = True,
                 no_reentry_after_sl: bool = True, close_only: bool = False):
        self.params = params
        self.tick_eval = tick_eval
        super().__init__(Path(out_dir), Path(out_dir), out_dir, date_str, lambda m: log(str(m).replace(r022.TAG, TAG)), None,
                         max_positions=max_positions, max_entries=max_entries, model=_RuleModel(),
                         exclude_funds=exclude_funds, no_reentry_after_sl=no_reentry_after_sl, log_only=False, nxt=False,
                         tp_pct=tp_pct, sl_pct=sl_pct, horizon_min=horizon_min, close_only=close_only)

    def _threshold(self):
        return None     # no probability cut: the status file shows none

    def _check_entry(self, code: str, name: str, now: datetime, price: float, store: pd.DataFrame, group: str = "watch",
                     block: str | None = None, session: str = "regular") -> None:
        tick = None
        if self.tick_eval is not None:
            try:
                tick = self.tick_eval(code, now)
            except Exception as exc:            # the tick feed never breaks the runner
                self.log(f"{TAG} {code} tick feed error ({type(exc).__name__}: {exc})")
        ok, stage, info = vwap_signal(store, float(price), now, tick, self.params)
        try:                                    # [2026-11-27] timeline trace (record only; never stops the evaluation)
            self._trace_info = {k: v for k, v in trace_fields(stage, info).items() if k != "stage"}
        except Exception:
            self._trace_info = None
        if not ok:
            self._stage(code, stage)
            return
        reason = (f"VWAP_gap={info.get('vwap_gap_pct'):+.2f}%_{'MA5' if info.get('ma5_up') else 'BBMID'}_UP_{info.get('tick')}"
                  f"@{pd.Timestamp(info.get('tick_cross')).strftime('%H:%M:%S') if info.get('tick_cross') else '-'}")
        self._open_position(code, name, now, price, group, session, prob=None, threshold=None, reason=reason, model_id=MODEL_ID)


def tick_feed_from_config(cfg, log: Callable[[str], None] | None = None):
    """r013 TickTriggerFeed with the VWAP trade's own tick settings (60-tick MA5 / MA15, 60 s, density)"""
    from r013_tick_trigger import TickTriggerFeed
    return TickTriggerFeed(
        Path(cfg.TICK_DATA_DIR), (int(getattr(cfg, "VWAP_TRADE_TICK_BAR_SIZE", 60)),),
        int(getattr(cfg, "VWAP_TRADE_TICK_MA_FAST", 5)), int(getattr(cfg, "VWAP_TRADE_TICK_MA_SLOW", 15)),
        float(getattr(cfg, "VWAP_TRADE_TICK_CROSS_MAX_AGE_SECONDS", 60.0)),
        float(getattr(cfg, "VWAP_TRADE_TICK_FEED_STALE_SECONDS", 15.0)), log=log,
        prior_day_warmup=bool(getattr(cfg, "TICK_TRIGGER_PRIOR_DAY_WARMUP", True)),
        min_prints_per_min=float(getattr(cfg, "VWAP_TRADE_MIN_PRINTS_PER_MIN", 30.0)),
        density_window_min=float(getattr(cfg, "VWAP_TRADE_MIN_PRINTS_WINDOW_MINUTES", 3)),
        warmup_extra_bars=int(getattr(cfg, "TICK_CROSS_WARMUP_EXTRA_BARS", 0) or 0))


def make_runner(cfg, out_dir: Path, date_str: str, log: Callable[[str], None], confirm_delay_s: float = 2.0):
    """The runner r003 uses, or None. VWAP_TRADE_MODE "paper" -> trading; "off" with today's paper positions still open -> a
    close-only runner (they still get their exits); otherwise None. Earlier days' open positions are closed first. Never raises."""
    try:
        out_dir = Path(out_dir)
        mode = str(getattr(cfg, "VWAP_TRADE_MODE", "off")).strip().lower()
        r022.close_missed_days(out_dir, Path(cfg.TICK_DATA_DIR), date_str, lambda m: log(str(m).replace(r022.TAG, TAG)),
                               prefix=FILE_PREFIX)
        close_only = False
        if mode == "paper" and bool(getattr(cfg, "VWAP_TRADE_LIVE_ORDERS", False)):
            mode = "live"                       # [2026-11-27] live orders on: no new paper buys (the same signal is not counted twice)
            log(f"{TAG} VWAP live orders on - no new paper buys")
        if mode != "paper":
            if not r022.open_codes_in(out_dir / f"{FILE_PREFIX}_{date_str}.jsonl"):
                return None
            log(f"{TAG} VWAP trade off - closing today's open paper position(s) only")
            close_only = True
        runner = VwapPaperRunner(
            out_dir, date_str, log, VwapParams.from_config(cfg), tick_feed_from_config(cfg).evaluate,
            max_positions=int(getattr(cfg, "VWAP_TRADE_MAX_POSITIONS", 3)),
            max_entries=int(getattr(cfg, "VWAP_TRADE_MAX_ENTRIES_PER_DAY", 20)),
            tp_pct=float(getattr(cfg, "VWAP_TRADE_TP_PCT", 3.0)), sl_pct=float(getattr(cfg, "VWAP_TRADE_SL_PCT", 1.27)),
            horizon_min=float(getattr(cfg, "VWAP_TRADE_HORIZON_MIN", 300)),
            exclude_funds=bool(getattr(cfg, "VWAP_TRADE_EXCLUDE_FUNDS", True)),
            no_reentry_after_sl=bool(getattr(cfg, "VWAP_TRADE_NO_REENTRY_AFTER_SL", True)), close_only=close_only)
        runner.trace_enabled = True             # [2026-11-27] timeline trace (r030): one row per judged bar
        # the day's KRX tick files price the paper positions r003 did not evaluate this turn (r022 poll_orphans)
        runner.tick_feed = r022.CollectorTickFeed(Path(cfg.TICK_DATA_DIR) / date_str, date_str, confirm_delay_s)
        runner.nxt_feed = None            # regular session only
        runner.collector_feed = None      # watched symbols only
        if not close_only:
            lab = runner._label()
            log(f"{TAG} paper mode enabled (no orders) | max_positions={runner.max_positions} max_entries={runner.max_entries} | "
                f"exits +{lab['tp']:g}% / -{lab['sl']:g}% / {lab['horizon']:g}min | out={runner.out_path}")
        return runner
    except Exception as exc:
        log(f"{TAG} init failed, VWAP paper mode disabled: {type(exc).__name__}: {exc}")
        return None


__all__ = ["VwapParams", "VwapPaperRunner", "vwap_signal", "session_vwap", "make_runner", "tick_feed_from_config", "TAG",
           "FILE_PREFIX", "krx_tick_size"]
