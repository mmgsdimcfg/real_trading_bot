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
- [2026-11-25] type=feat owner=claude
    summary: 남은 과제 B(Codex 계획 검토 반영) - close_missed_days(): r003 이 마감 전에 멈춰 지난 날짜 파일에 남은 가상 보유를 그날 파일에
      EXIT(reason close_missed) 로 정리. 시각 = 그날 예정 청산(15:20 / NXT 19:59:50), 가격 = 단일가 체결 / NXT 마지막 체결(19:59:50 까지)
      / 마지막 정규장 체결(15:20 까지, estimated) / 매수 기록(estimated), price_at / recorded_at 기록. 7일(달력) 안만, 그보다 오래된 것은
      개수만 로그. 같은 ENTRY 에 두 번 쓰지 않음. last_print_until(): 시각 상한이 있는 틱 파일 마지막 체결.
      Codex 코드 검토 반영: 끝이 잘린 파일엔 줄바꿈부터 넣고 기록, 마지막 체결은 체결 시각 기준(수신 순서 아님), NXT 체결이 19:59:50 보다
      180초 넘게 오래됐으면 estimated, 기간은 달력 7일(연휴 포함).
    impact: paper only (기록 파일)
    compatibility: backward-compatible
- [2026-11-25] type=feat owner=claude
    summary: 사용자 결정(매매 방식 조합, Codex 검토: 가상 매매를 꺼도 열린 가상 보유는 청산) - EdgeOnlyRunner(close_only=True): 보유
      청산(tp/sl/time/세션 마감)만 하고 신규 가상 매수·후보 기록·수집기 신규 평가는 하지 않는다. open_codes_in(path): 그날 기록에서
      아직 열린 종목. 상태 파일에 close_only. _restore 는 잘린 줄만 건너뛴다(예전: 파일 전체를 버림, Codex 검토).
    impact: live paper only (주문 없음)
    compatibility: backward-compatible (close_only=False 기본)
- [2026-11-25] type=feat owner=claude
    summary: 사용자 결정(가상 매매 청산을 운용 시간·종목의 NXT 가능 여부에 맞춤, Codex 검토 권고안) -
      (1) 청산 시각: 정규장 종목(또는 정규장만 운용) = 15:20 종가 단일가 청산, 가상 체결가 = 15:30 단일가 체결가(틱 파일 close_auction),
          15:32 까지 없으면 마지막 정규장 가격 - 1틱 + estimated. NXT 가능 종목(NXT 포함 운용) = 19:59:50, NXT 현재가 - 1틱
          (3분 넘게 묵은 가격이면 estimated). 장전 / 장후 / 정규장 사이를 넘어 보유(08:45 / 15:15 / 19:50 강제 청산 없앰).
          reason close_regular / close_nxt (예전 eod 기록은 그대로).
      (6) Codex 코드 검토 반영: 단일가 체결가는 15:30 부터만 읽음(15:20 실주문 청산 앞에서 파일 읽기 없음), 정규장만 운용의 15:30 루프
          종료는 15:32 까지 기다림, 폴백 가격 = 틱 파일 마지막 정규장 체결 -> 마지막 본 가격 -> 매수 기록 가격(재시작 후에도 청산),
          고아 처리는 틱 파일 체결 시각이 3분 이내일 때만(메모리 가격 재사용 없음), EXIT 기록이 실패하면 보유 유지 후 다음 턴 재시도,
          종목별 예외 격리, 1틱 차감은 호가 단위 경계 반영(marketable_sell_limit).
      (7) Codex 재검토 반영: 15:30 종료 대기는 실제 경과 시간(monotonic)과 요청한 대기 중 큰 값으로 15:32 를 넘기지 않음, 15:30 전에는
          단일가 파일을 읽지 않음. NXT 고아 보유는 NXT 틱 파일(ticks/<날짜>/nxt, nxt_feed)의 3분 이내 체결로 tp/sl/time 판정.
          수집기 종목·고아 모두 같은 읽기의 체결 시각으로 0~180초 신선도 확인(미래 시각 거부). 최종 청산 기록 실패는 1초 뒤 2번 재시도.
          NXT 19:59:50 청산가 = NXT 틱 파일의 3분 이내 체결 -> r003 이 넘긴 3분 이내 시세 -> 마지막 가격(estimated).
          (15:32 이후의 단일가 파일 조회는 유지 - 15:32 는 가상 체결가 기준일 뿐 실주문과 무관하고, 늦게 보인 체결가가 더 정확하다)
      (2) session_close(now, final): r003 이 매 턴 종목 순회 전(15:20~15:30 단일가 구간, NXT 청산 대기 구간 포함)과 루프 종료 직전에 부른다
          - 종목 순회로는 마감 청산이 불리지 않던 문제(Codex 검토).
      (3) on_symbol(nxt_tradeable): 종목별 NXT 가능 여부(r003 nxt_map). ENTRY / 보유 상태에 "nxt" 저장(재시작 복원, 없는 예전 기록 =
          정규장 종목). 수집기 종목은 NXT 여부를 모름 -> 정규장 종목.
      (4) poll_orphans: 틱 파일은 지금이 정규장일 때만 쓰고, 마지막 가격은 3분 이내 것만 쓴다(지난 세션 가격으로 tp/sl 판정 방지).
          _eval_at 은 평가가 끝난 뒤 기록.
      (5) nxt 인자는 r003 이 nxt_trading_enabled()(자동매매 시간) 로 준다 - r001 EDGE_ONLY_NXT 는 쓰지 않음.
    impact: live paper only (주문 없음)
    compatibility: 기록 형식 추가(nxt, fill, estimated, close_regular / close_nxt). 예전 기록은 그대로 읽힌다.
- [2026-11-25] type=feat owner=claude
    summary: 사용자 요청(가상 매매 익절/손절을 앱에서 설정) - EdgeOnlyRunner(tp_pct, sl_pct, horizon_min): 0 보다 크면 모델 라벨
      (current_meta.json label) 대신 그 값으로 새 가상 매수의 목표/손절/보유 시간을 정한다(r001 EDGE_ONLY_TP_PCT /
      EDGE_ONLY_SL_PCT / EDGE_ONLY_HORIZON_MIN, 0 = 모델 라벨). 이미 있는 가상 보유는 매수 때 정한 값 그대로. 상태 파일 label =
      실제 쓰는 값, model_label = 모델 값. CANDIDATE 의 label 은 모델 값 그대로(학습 정의와 맞추는 연구용).
    impact: live paper only (주문 없음)
    compatibility: backward-compatible (세 값 모두 0 이면 종전과 동일)
- [2026-11-23] type=feat owner=claude
    summary: 사용자 결정(가상 매매를 NXT 시간에도, 세션 구분 기록) - EdgeOnlyRunner(nxt=True): NXT 장전(nxt_pre 08:00~08:50) /
      장후(nxt_after 15:30~20:00)에도 r003 이 넘긴 1분봉(NXT 봉)으로 같은 정책을 평가한다. 세션별 매수 창과 강제 청산
      (NXT_SESSIONS: 08:05~08:35 / 08:45, 15:35~19:30 / 19:50). 장후는 정규장 봉에 이어 붙인 봉으로, 장전은 장전 봉만으로
      지표/피처를 만든다(첫 봉이 08:10 이후면 no_history). ENTRY / EXIT / CANDIDATE 와 보유 상태에 "session"
      (regular / nxt_pre / nxt_after; 없는 예전 기록 = regular). 정규장 보유는 종전대로 15:15, NXT 보유는 그 세션의 청산 시각.
      수집기 종목(KRX 틱 파일)은 정규장만. 모델은 정규장 데이터로 학습된 것 - NXT 성과는 session 으로 나눠 본다.
    impact: live paper only (주문 없음)
    compatibility: backward-compatible (nxt=False 또는 정규장 시간에는 종전과 동일)
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
import math
import re
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

import r011_edge_ai as r011
from r002_strategy_core_shared import krx_tick_size, marketable_sell_limit

ENTRY_START = dt_time(9, 5)
ENTRY_END = dt_time(14, 50)
# [2026-11-25] session close of the paper positions (user decision, Codex review): a regular-only holding (or every holding when
# the runner does not trade NXT) is closed into the KRX closing auction at 15:20 and filled at the 15:30 auction print; an NXT
# holding is closed at 19:59:50 (the same time as r003 AFTERNOON_NXT_FORCE_EXIT). The old 15:15 / 08:45 / 19:50 exits are gone.
REGULAR_CLOSE = dt_time(15, 20)
AUCTION_WAIT_UNTIL = dt_time(15, 32)   # no auction print by then -> the last regular price - 1 tick, marked estimated
AUCTION_PRINT_FROM = dt_time(15, 30)   # the auction fill prints at the close: the tick file is not read for it before
NXT_CLOSE = dt_time(19, 59, 50)
FORCE_EXIT = REGULAR_CLOSE             # kept for the status file / older callers
PRICE_FRESH_S = 180.0                  # a remembered price older than this is not used for tp / sl (session gaps are >= 10 min)
# [2026-11-23] NXT sessions of the paper runner: bars with lo < label <= hi belong to the session; buys between entry_start and
# entry_end. [2026-11-25] force_exit is no longer used: NXT holdings are kept across the sessions until NXT_CLOSE
NXT_SESSIONS = {
    "nxt_pre": {"lo": dt_time(8, 0), "hi": dt_time(8, 50), "entry_start": dt_time(8, 5), "entry_end": dt_time(8, 35),
                "force_exit": dt_time(8, 45)},
    "nxt_after": {"lo": dt_time(15, 30), "hi": dt_time(20, 0), "entry_start": dt_time(15, 35), "entry_end": dt_time(19, 30),
                  "force_exit": dt_time(19, 50)},
}
NXT_PRE_FIRST_BAR_BY = dt_time(8, 10)   # nxt_pre: no prediction unless the session's bars start by this label
SESSIONS = ("regular",) + tuple(NXT_SESSIONS)


def nxt_session_of(t: dt_time) -> str | None:
    """The NXT session a wall-clock time falls in (None: regular hours or outside)."""
    for name, w in NXT_SESSIONS.items():
        if w["lo"] <= t <= w["hi"] and not (name == "nxt_after" and t == w["lo"]):
            return name
    return None


def force_exit_of(session: str | None) -> dt_time:
    """[2026-11-23] kept for older callers; [2026-11-25] the close of a position is close_time_of()"""
    w = NXT_SESSIONS.get(str(session or "regular"))
    return w["force_exit"] if w else FORCE_EXIT


def close_time_of(pos: dict, runner_nxt: bool) -> dt_time:
    """[2026-11-25] when a paper position is closed: NXT_CLOSE for an NXT holding of a runner that trades NXT, else REGULAR_CLOSE"""
    return NXT_CLOSE if (runner_nxt and pos.get("nxt")) else REGULAR_CLOSE


def nxt_raw_1min(frame_1min: pd.DataFrame | None, trade_date, session: str) -> pd.DataFrame | None:
    """End-labeled OHLCV 1m bars of one NXT session on trade_date (r011.session_raw_1min for the NXT hours)."""
    if frame_1min is None or len(frame_1min) == 0:
        return None
    w = NXT_SESSIONS[session]
    idx = pd.DatetimeIndex(frame_1min.index)
    t = idx.time
    mask = (idx.normalize() == pd.Timestamp(trade_date).normalize()) & (t > w["lo"]) & (t <= w["hi"])
    raw = frame_1min.loc[mask, ["open", "high", "low", "close", "volume"]]
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return None if raw.empty else raw
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


def _pos_or_zero(v) -> float:
    """a finite number > 0, else 0.0 (= not set)"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) and f > 0 else 0.0


def open_codes_in(path: Path) -> set[str]:
    """[2026-11-25] codes with an ENTRY and no EXIT yet in a day's edge_only_<date>.jsonl (empty when there is no file)"""
    path = Path(path)
    if not path.is_file():
        return set()
    open_: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        code = str(rec.get("code") or "")
        if rec.get("event") == "ENTRY":
            open_.add(code)
        elif rec.get("event") == "EXIT":
            open_.discard(code)
    return open_


# ---- [2026-11-25] open paper positions of earlier days (r003 stopped before the session close) ---------------------------------

def last_print_until(path: Path, date_str: str, until_hhmmss: str, sessions: tuple[str, ...],
                     since_hhmmss: str = "000000") -> tuple[float, str] | None:
    """(price, HHMMSS) of the latest print (by trade time - the collector writes in arrival order) of a tick file between
    `since_hhmmss` and `until_hhmmss` in `sessions` (None without one). Same row rules as CollectorTickFeed (header widths, bsop_date,
    session column; files without it: 09:00 <= t < 15:20 = regular)."""
    path = Path(path)
    if not path.is_file():
        return None
    best = None
    cols = None
    width = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = line.rstrip("\r\n").split(",")
            if cols is None:
                cols = {c: i for i, c in enumerate(fields)}
                width = len(fields)
                continue
            if len(fields) != width:
                continue
            try:
                t = fields[cols["trade_time"]].strip().zfill(6)
                price = float(fields[cols["price"]])
                day = fields[cols["bsop_date"]].strip() if "bsop_date" in cols else date_str
            except (KeyError, ValueError):
                continue
            if day != date_str or not price > 0 or not t.isdigit() or t > until_hhmmss or t < since_hhmmss:
                continue
            sess = fields[cols["session"]].strip() if "session" in cols else ("regular" if "090000" <= t < "152000" else "")
            if sess in sessions and (best is None or t >= best[1]):
                best = (price, t)
    return best


def close_missed_days(out_dir: Path, ticks_root: Path, today: str, log: Callable[[str], None], lookback_days: int = 7,
                      now: datetime | None = None, prefix: str = "edge_only") -> int:
    """[2026-11-25] Paper positions still open in an earlier day's edge_only_<date>.jsonl (r003 stopped before that day's session
    close) get an EXIT in that day's file: reason close_missed, at that day's scheduled close (15:20 regular / 19:59:50 NXT holding).
    Price: regular -> the closing auction print, else the last regular print up to 15:20 (estimated), else the buy record (estimated);
    NXT holding -> the last NXT print up to 19:59:50, else the same regular fallbacks (estimated). Fields: price_at (HHMMSS of the price
    used, if any), recorded_at (now). Days within lookback_days (calendar) before `today`; older open positions are only counted.
    Idempotent (an ENTRY with an EXIT is not touched). Returns how many were closed. Never raises."""
    n = 0
    try:
        now = now or datetime.now()
        out_dir = Path(out_dir)
        ticks_root = Path(ticks_root)
        t0 = datetime.strptime(today, "%Y%m%d")
        older = 0
        for path in sorted(out_dir.glob(f"{prefix}_*.jsonl")):   # [2026-11-27] prefix: the VWAP paper files (r029) too
            day = path.stem[len(prefix) + 1:]
            if not (len(day) == 8 and day.isdigit()) or day >= today:
                continue
            open_ = _open_entries(path)
            if not open_:
                continue
            if (t0 - datetime.strptime(day, "%Y%m%d")).days > lookback_days:
                older += len(open_)
                continue
            base = datetime.strptime(day, "%Y%m%d")
            for code, rec in open_.items():
                try:
                    nxt = rec.get("nxt") is True
                    close_t = NXT_CLOSE if nxt else REGULAR_CLOSE
                    fill, estimated, px, px_at = None, True, None, None
                    if nxt:
                        lp = last_print_until(ticks_root / day / "nxt" / f"{code}.csv", day, NXT_CLOSE.strftime("%H%M%S"),
                                              ("pre", "regular", "post"))
                        if lp:
                            # a print older than PRICE_FRESH_S before the close is used but marked estimated (r022 rule)
                            age = (datetime.strptime(NXT_CLOSE.strftime("%H%M%S"), "%H%M%S")
                                   - datetime.strptime(lp[1], "%H%M%S")).total_seconds()
                            px, px_at, fill, estimated = tick_down(lp[0]), lp[1], "last", age > PRICE_FRESH_S
                    if px is None and not nxt:
                        lp = last_print_until(ticks_root / day / f"{code}.csv", day, "153200", ("close_auction",))
                        if lp:
                            px, px_at, fill, estimated = lp[0], lp[1], "auction", False
                    if px is None:
                        lp = last_print_until(ticks_root / day / f"{code}.csv", day, REGULAR_CLOSE.strftime("%H%M%S"), ("regular",))
                        if lp:
                            px, px_at, fill = tick_down(lp[0]), lp[1], "last"
                    if px is None:
                        px, fill = tick_down(float(rec.get("price") or rec["entry"])), "entry_record"
                    at = base.replace(hour=close_t.hour, minute=close_t.minute, second=close_t.second)
                    entry = float(rec["entry"])
                    held = (pd.Timestamp(at) - pd.Timestamp(rec["ts"])).total_seconds() / 60.0
                    out = {"event": "EXIT", "ts": at.isoformat(timespec="seconds"), "code": code, "name": rec.get("name"),
                           "reason": "close_missed", "entry": entry, "exit": px, "net_pct": net_pct(entry, px),
                           "held_min": round(held, 1), "prob": rec.get("prob"), "group": rec.get("group") or "watch",
                           "session": rec.get("session") or "regular", "nxt": nxt, "fill": fill,
                           "recorded_at": now.isoformat(timespec="seconds")}
                    if px_at:
                        out["price_at"] = px_at
                    if estimated:
                        out["estimated"] = True
                    _append_line(path, json.dumps(out, ensure_ascii=False, default=str))
                    n += 1
                    log(f"{TAG} {code}_{rec.get('name')} | PAPER SELL {px:,.0f} close_missed ({day}, {fill}"
                        f"{', estimated' if estimated else ''}) net={out['net_pct']:+.2f}% - r003 stopped before that day's close")
                except Exception as exc:
                    log(f"{TAG} {code} close_missed ({day}) failed: {type(exc).__name__}: {exc}")
        if older:
            log(f"{TAG} {older} open paper position(s) older than {lookback_days} days left as they are")
    except Exception as exc:
        try:
            log(f"{TAG} close_missed error: {type(exc).__name__}: {exc}")
        except Exception:
            pass
    return n


def _append_line(path: Path, line: str) -> None:
    """append one JSONL record; a file whose last line is torn (no newline) gets a newline first so the record stays whole"""
    path = Path(path)
    lead = ""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            if fh.tell() > 0:
                fh.seek(-1, 2)
                lead = "" if fh.read(1) == b"\n" else "\n"
    except OSError:
        lead = ""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(lead + line + "\n")
        fh.flush()


def _open_entries(path: Path) -> dict[str, dict]:
    """code -> the ENTRY record still without an EXIT in a day's file (torn lines skipped)"""
    open_: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        code = str(rec.get("code") or "")
        if rec.get("event") == "ENTRY":
            open_[code] = rec
        elif rec.get("event") == "EXIT":
            open_.pop(code, None)
    return open_


def tick_down(price: float) -> float:
    """[2026-11-25] one price step below (tick table across bands: 2,000 -> 1,999, 5,000 -> 4,995)"""
    return float(marketable_sell_limit(None, float(price), 1))


def _fresh_print(feed, code: str, now: datetime, read: bool = False) -> bool:
    """[2026-11-25] the feed's last print of the symbol is 0..PRICE_FRESH_S old (a feed without last_print: True)"""
    last_print = getattr(feed, "last_print", None)
    if not callable(last_print):
        return True
    lp = last_print(code, read=read)
    if not lp:
        return False
    age = (pd.Timestamp(now) - pd.Timestamp(lp[1])).total_seconds()
    return 0 <= age <= PRICE_FRESH_S


def net_pct(entry: float, exit_px: float) -> float:
    """Net % of a round trip after commission (both sides) and sell tax."""
    cost = entry * FEE_RATE + exit_px * (FEE_RATE + SELL_TAX_RATE)
    return (exit_px - entry - cost) / entry * 100.0


class EdgeOnlyRunner:
    # [2026-11-27] file name prefix of the day's records / status / candidates (r029 VwapPaperRunner: "vwap_paper")
    FILE_PREFIX = "edge_only"
    # [2026-11-27] strategy name in the timeline trace file strategy_eval_<name>_<date>.jsonl (r030)
    TRACE_STRATEGY = "edge"
    # [2026-11-27] the method's target codes (r001 EDGE_TARGET_CODES / VWAP_TARGET_CODES, set by r003 after the runner is made).
    # Empty = everything as before. New entries only: an open paper position is closed whatever the list says.
    targets: frozenset = frozenset()
    # off unless the owner switches it on (r003 / r029.make_runner do): a runner built by a test or a study writes no trace
    trace_enabled = False

    def __init__(self, model_dir: Path, data_root: Path, out_dir: Path, date_str: str, log: Callable[[str], None], config,
                 min_prob: float = 0.0, max_positions: int = 3, max_entries: int = 10, model=None,
                 collector_max_positions: int = 0, collector_max_entries: int = 0,
                 exclude_funds: bool = True, no_reentry_after_sl: bool = True, log_only: bool = False, nxt: bool = False,
                 tp_pct: float = 0.0, sl_pct: float = 0.0, horizon_min: float = 0.0, close_only: bool = False):
        # [2026-11-27] the trade model (edge_ai/trade, g014 --role trade) when there is one, else the filter model as before
        self.model = model if model is not None else r011.EdgeAIModel(Path(model_dir), prefer_subdir=r011.TRADE_MODEL_SUBDIR)
        self.data_root = Path(data_root)
        self.out_path = Path(out_dir) / f"{self.FILE_PREFIX}_{date_str}.jsonl"
        self.date_str = date_str
        self.log = log
        self.config = config
        self.min_prob = float(min_prob or 0.0)
        # [2026-11-25] paper trading switched off with positions still open: close them only (no new buys / candidates)
        self.close_only = bool(close_only)
        # [2026-11-25] exit overrides of new paper buys (% / % / minutes); 0 or less = the model's label
        self.exit_override = {"tp": _pos_or_zero(tp_pct), "sl": _pos_or_zero(sl_pct), "horizon": _pos_or_zero(horizon_min)}
        self.max_positions = int(max_positions)
        self.max_entries = int(max_entries)
        # [2026-11-19] per-group limits: the watchlist (r003's symbols) and the collector-only symbols count separately
        self.limits = {"watch": (self.max_positions, self.max_entries),
                       "collector": (int(collector_max_positions), int(collector_max_entries))}
        self.bars: dict[str, pd.DataFrame] = {}
        # [2026-11-23] NXT sessions too (r001 EDGE_ONLY_NXT and ENABLE_NXT_SESSION): their bars are kept apart from the regular store
        self.nxt = bool(nxt)
        self.xbars: dict[tuple[str, str], pd.DataFrame] = {}
        # [2026-11-25] per-symbol NXT eligibility as r003 reports it (nxt_map); unknown = regular only
        self.nxt_ok: dict[str, bool] = {}
        self.sleep = __import__("time").sleep      # session_close(final) waits for the auction print (tests replace it)
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.positions: dict[str, dict] = {}
        self.entries = {g: 0 for g in GROUPS}
        self.closed: list[float] = []
        self._names: dict[str, str] = {}
        self._collector_next: datetime | None = None
        # [2026-11-19] per-symbol status for the app (write_status)
        self.sym_state: dict[str, dict] = {}
        self.status_path = Path(out_dir) / f"{self.FILE_PREFIX}_status_{date_str}.json"
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
        self.cand_path = Path(out_dir) / f"{self.FILE_PREFIX}_candidates_{date_str}.jsonl"
        self._cand_ids: set[str] = set()
        if self.log_only and self.cand_path.is_file():
            try:
                for line in self.cand_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    self._cand_ids.add(str(json.loads(line).get("id") or ""))
            except Exception as exc:
                self.log(f"{TAG} candidate log restore failed ({type(exc).__name__}: {exc})")
        self._restore()

    @property
    def entries_today(self) -> int:
        """the watchlist's entries of the day (the original counter)"""
        return self.entries["watch"]

    # ---- persistence ------------------------------------------------------------------------
    def _emit(self, rec: dict) -> bool:
        """append one record; [2026-11-25] False when it could not be written"""
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with self.out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            return True
        except Exception:
            return False

    def _restore(self) -> None:
        """Positions still open and the day's counters, from the day's file (an r003 restart must not forget them)."""
        if not self.out_path.is_file():
            return
        try:
            for line in self.out_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue      # [2026-11-25] a torn line is skipped, the rest of the day is kept (same as open_codes_in)
                code = str(rec.get("code") or "")
                if rec.get("event") == "ENTRY":
                    group = rec.get("group") if rec.get("group") in GROUPS else "watch"
                    self.entries[group] += 1
                    self.positions[code] = {"name": rec.get("name"), "entry": float(rec["entry"]), "ts": pd.Timestamp(rec["ts"]),
                                            "tp": float(rec["tp_px"]), "sl": float(rec["sl_px"]), "horizon": float(rec["horizon"]),
                                            "prob": rec.get("prob"), "group": group,
                                            "session": rec.get("session") if rec.get("session") in SESSIONS else "regular",
                                            "nxt": rec.get("nxt") is True,   # [2026-11-25] older records: regular only
                                            # [2026-11-25] the last known price after a restart (session close fallback)
                                            "last_px": float(rec.get("price") or rec["entry"])}
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
    def _model_label(self) -> dict:
        """the model's label definition (current_meta.json label, DEFAULT_LABEL for a missing key)"""
        lab = {}
        try:
            lab = dict((self.model.meta or {}).get("label") or {})
        except Exception:
            lab = {}
        return {k: float(lab.get(k) or DEFAULT_LABEL[k]) for k in DEFAULT_LABEL}

    def _label(self) -> dict:
        """the exits of a new paper buy: the model's label, each value replaced by its r001 override when that is > 0"""
        lab = self._model_label()
        ov = getattr(self, "exit_override", None) or {}
        return {k: (ov[k] if ov.get(k, 0) > 0 else v) for k, v in lab.items()}

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
                  group: str = "watch", nxt_tradeable: bool | None = None) -> None:
        try:
            code = str(code).zfill(6)
            if nxt_tradeable is not None:
                self.nxt_ok[code] = bool(nxt_tradeable)
            self._on_symbol(code, name, now, frame_1m, float(live_price), group if group in GROUPS else "watch")
            self._eval_at[code] = datetime.now()   # [2026-11-25] after the evaluation (a failed one leaves it to poll_orphans)
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
        if self.close_only:
            codes = []                # [2026-11-25] closing the open positions only
        for code in dict.fromkeys(codes + held):
            try:
                frame, price = feed.poll(code, now)
                if not _fresh_print(feed, code, now):     # [2026-11-25] no new print for 3 min: nothing to evaluate
                    continue
            except Exception as exc:
                self._safe_log(f"{TAG} {code} collector feed error: {type(exc).__name__}: {exc}")
                continue
            if price is None or not price > 0:
                continue
            n += 1
            self.on_symbol(code, self.name_of(code), now, frame, price, group="collector")
        return n

    def poll_orphans(self, feed: "CollectorTickFeed | None", now: datetime, since: datetime) -> int:
        """[2026-11-20] Open paper positions that r003 did not hand to on_symbol this turn (the symbol left the watchlist, was
        bought by hand -> [HOLD SKIP], or r003 restarted without it): exits (tp / sl / time) from the day's tick files.
        Bug 20261120: 443060 bought by hand at 14:41 kept its paper position past its target.
        `since` = start of r003's turn: a position looked at after it (symbol loop or poll_collector) is skipped.
        [2026-11-25] (Codex review) regular session: the KRX tick file (`feed`); NXT sessions: an NXT holding from the NXT tick file
        (self.nxt_feed, ticks/<date>/nxt), exits only (no bars). A print is used only when it is 0..PRICE_FRESH_S old; the
        remembered last price is not re-used. The session close is session_close()'s job. Returns how many were checked."""
        n = 0
        t = now.time()
        regular = dt_time(9, 0) <= t < REGULAR_CLOSE
        nxt_now = self.nxt and nxt_session_of(t) is not None
        if not regular and not nxt_now:
            return 0
        src = feed if regular else getattr(self, "nxt_feed", None)
        if src is None:
            return 0
        for code in [c for c in list(self.positions) if not (self._eval_at.get(c) and self._eval_at[c] > since)]:
            pos = self.positions.get(code)
            if pos is None or (nxt_now and not regular and not pos.get("nxt")):
                continue
            try:
                frame, price = src.poll(code, now)
                if not _fresh_print(src, code, now):
                    continue
            except Exception as exc:
                self._safe_log(f"{TAG} {code} orphan feed error: {type(exc).__name__}: {exc}")
                continue
            if price is None or not float(price) > 0:
                continue
            n += 1
            self.on_symbol(code, pos.get("name") or self.name_of(code), now, frame if regular else None, float(price),
                           group=pos.get("group", "watch"))
        return n

    def _fresh_price(self, code: str, now: datetime, max_age_s: float = PRICE_FRESH_S) -> float | None:
        """[2026-11-25] the last price seen for the symbol when it is at most max_age_s old (None otherwise)"""
        st = self.sym_state.get(code) or {}
        price, at = st.get("price"), st.get("at")
        if price is None or not at:
            return None
        try:
            age = (pd.Timestamp(now) - pd.Timestamp(at)).total_seconds()
        except Exception:
            return None
        return float(price) if 0 <= age <= max_age_s and float(price) > 0 else None

    def _auction_price(self, code: str) -> float | None:
        feed = getattr(self, "tick_feed", None)
        if feed is None:
            return None
        try:
            return feed.auction_price(code)
        except Exception as exc:
            self._safe_log(f"{TAG} {code} auction price error: {type(exc).__name__}: {exc}")
            return None

    def _safe_log(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:
            pass

    def _last_regular_print(self, code: str) -> float | None:
        """the day's last regular-session print of the tick file (None without the file / feed)"""
        feed = getattr(self, "tick_feed", None)
        last_print = getattr(feed, "last_print", None) if feed is not None else None
        if not callable(last_print):
            return None
        try:
            lp = last_print(code)
            return float(lp[0]) if lp and float(lp[0]) > 0 else None
        except Exception:
            return None

    def _known_price(self, code: str, pos: dict) -> float | None:
        """the last price seen for the symbol, else the buy record's price (after a restart)"""
        px = (self.sym_state.get(code) or {}).get("price") or pos.get("last_px")
        try:
            return float(px) if px and float(px) > 0 else None
        except (TypeError, ValueError):
            return None

    def session_close(self, now: datetime, final: bool = False, wait_s: float = 0.0, codes: list[str] | None = None) -> int:
        """[2026-11-25] Close the paper positions whose session close has come (r003 calls it every turn after its own scheduled
        liquidations, also in the 15:20~15:30 auction and the NXT close wait, and once more with final=True when its market loop
        ends). `codes` limits it to those symbols (an exit check of one symbol).
        - regular close (15:20): filled at the closing auction print (read from 15:30 on). Until AUCTION_WAIT_UNTIL the position
          waits for it (final: waits up to wait_s, not past AUCTION_WAIT_UNTIL, polling every 5 s); after that it is closed at the
          day's last regular print, else the last price seen / the buy record's price - 1 tick, marked estimated
        - NXT close (19:59:50): the NXT price seen in the last PRICE_FRESH_S - 1 tick, else the last known price (estimated)
        final=True closes every open position. A failed EXIT write keeps the position (retried next turn). Returns how many were
        closed. Never raises."""
        n = 0
        try:
            t = now.time()
            pool = [c for c in (codes if codes is not None else list(self.positions)) if c in self.positions]
            due = [c for c in pool if final or t >= close_time_of(self.positions[c], self.nxt)]
            if not due:
                return 0
            regular_due = [c for c in due if close_time_of(self.positions[c], self.nxt) == REGULAR_CLOSE]
            if final and wait_s > 0 and regular_due and t < AUCTION_WAIT_UNTIL:
                # wait for the auction print, at most wait_s and never past AUCTION_WAIT_UNTIL. Elapsed = the larger of the real
                # (monotonic) time and the requested sleeps, so slow file reads count; the file is read only from 15:30 on
                import time as _time
                limit = min(float(wait_s), (datetime.combine(now.date(), AUCTION_WAIT_UNTIL) - now).total_seconds())
                start, slept = _time.monotonic(), 0.0

                def elapsed() -> float:
                    return max(_time.monotonic() - start, slept)

                def missing() -> bool:
                    if (now + timedelta(seconds=elapsed())).time() < AUCTION_PRINT_FROM:
                        return True
                    return any(self._auction_price(c) is None for c in regular_due if c in self.positions)

                while elapsed() < limit and missing():
                    step = min(5.0, limit - elapsed())
                    if step <= 0:
                        break
                    self.sleep(step)
                    slept += step
                now = now + timedelta(seconds=min(elapsed(), max(limit, 0.0)))
                t = now.time()
            for attempt in range(3 if final else 1):
                for code in [c for c in due if c in self.positions]:
                    try:
                        if self._close_due(code, now, t, final):
                            n += 1
                    except Exception as exc:
                        self._safe_log(f"{TAG} {code} session close error: {type(exc).__name__}: {exc}")
                if not final or attempt == 2 or not any(c in self.positions for c in due):
                    break
                self.sleep(1.0)          # [2026-11-25] the last chance of the day: retry a failed EXIT write
        except Exception as exc:
            self._safe_log(f"{TAG} session close error: {type(exc).__name__}: {exc}")
        return n

    def _close_due(self, code: str, now: datetime, t: dt_time, final: bool) -> bool:
        pos = self.positions.get(code)
        if pos is None:
            return False
        if close_time_of(pos, self.nxt) == REGULAR_CLOSE:
            px = self._auction_price(code) if t >= AUCTION_PRINT_FROM else None
            if px:
                return self._close(code, now, "close_regular", float(px), fill="auction")
            if not final and t < AUCTION_WAIT_UNTIL:
                return False                      # the auction print comes at 15:30
            last = self._last_regular_print(code) or self._known_price(code, pos)
            if last is None:
                self._safe_log(f"{TAG} {code} close_regular: no price - position kept")
                return False
            return self._close(code, now, "close_regular", tick_down(last), fill="last", estimated=True)
        # [2026-11-25] the NXT tick file's print when it is 0..PRICE_FRESH_S old (time of the trade), else the live quote r003
        # handed over in the last PRICE_FRESH_S, else the last known price (estimated)
        fresh = None
        nf = getattr(self, "nxt_feed", None)
        if nf is not None:
            try:
                lp = nf.last_print(code)
                if lp and 0 <= (pd.Timestamp(now) - pd.Timestamp(lp[1])).total_seconds() <= PRICE_FRESH_S and float(lp[0]) > 0:
                    fresh = float(lp[0])
            except Exception as exc:
                self._safe_log(f"{TAG} {code} NXT tick read error: {type(exc).__name__}: {exc}")
        if fresh is None:
            fresh = self._fresh_price(code, now)
        last = fresh if fresh is not None else self._known_price(code, pos)
        if last is None:
            self._safe_log(f"{TAG} {code} close_nxt: no price - position kept")
            return False
        return self._close(code, now, "close_nxt", tick_down(last), fill="last", estimated=fresh is None)

    def write_status(self, now: datetime, force: bool = False) -> bool:
        """[2026-11-19] edge_only_status_<date>.json for the app, at most every STATUS_EVERY_S (force: now). Never raises."""
        return _write_status(self, now, force)

    def _stage(self, code: str, stage: str, prob: float | None = None) -> None:
        st = self.sym_state.get(code)
        if st is not None:
            st["stage"] = stage
            st["prob"] = None if prob is None else round(float(prob), 4)
        # [2026-11-27] timeline trace (r030): the stage a NEW bar of the symbol ended at - once per bar, a record only. The
        # stage calls of the exit / close-only polling above the new-bar check are not bars and are not traced
        pend = getattr(self, "_trace_pending", None)
        if pend is not None and pend[0] == code:
            self._trace_pending = None
            info, self._trace_info = getattr(self, "_trace_info", None), None
            # watched symbols only: the tick-collector group is many symbols the timeline does not show
            if not self.trace_enabled or (st or {}).get("group", "watch") != "watch":
                return
            try:                                  # [Codex round 2] only values the decision already has - nothing is asked again
                import r030_strategy_trace as r030
                r030.trace(self.out_path.parent, self.TRACE_STRATEGY, self.date_str, pend[1], code, "paper", stage,
                           bar=pend[2], price=(st or {}).get("price"), prob=prob, **(info if isinstance(info, dict) else {}))
            except Exception:
                pass

    def _on_symbol(self, code: str, name: str, now: datetime, frame_1m, price: float, group: str = "watch") -> None:
        if not (price > 0):
            return
        st = self.sym_state.setdefault(code, {"stage": "wait_bars", "prob": None, "bar": None, "bars": 0})
        st.update(name=name, group=group, price=price, at=now.isoformat(timespec="seconds"))
        self._trace_pending = None                 # [2026-11-27] set again below when this poll brings a new bar
        self._trace_info = None
        was_open = code in self.positions
        if was_open:
            self._check_exit(code, now, price)
            if code not in self.positions:
                self._stage(code, "sold")
        if self.close_only:
            if code in self.positions:
                self._stage(code, "held", self.positions[code].get("prob"))
            return
        if frame_1m is None or len(frame_1m) == 0:
            return
        sess = nxt_session_of(now.time()) if self.nxt else None
        if sess is not None:                   # [2026-11-23] NXT pre / after session: its own bars and windows
            self._on_symbol_nxt(code, name, now, frame_1m, price, group, sess, was_open, st)
            return
        raw = r011.session_raw_1min(frame_1m, now.date())
        if raw is None or raw.empty:
            return
        store = self._append(code, raw)
        key = store.index[-1]
        if self.last_bar.get(code) == key:
            return  # no new completed bar
        self.last_bar[code] = key
        self._trace_pending = (code, now, pd.Timestamp(key).strftime("%H:%M"))   # [2026-11-27] the next _stage of this bar is traced
        st.update(bar=pd.Timestamp(key).strftime("%H:%M"), bars=int(len(store)))
        # a symbol that was held at the start of this poll is not bought on this bar (also right after its sale: the
        # next entry needs a new bar, like the simulator where a position blocks the symbol until it is closed)
        if was_open:
            if code in self.positions:
                self._stage(code, "held", self.positions[code].get("prob"))
            else:
                self._stage(code, "sold")          # [Codex round 2] sold in this poll: the same stage again, so the bar has its trace row
            return
        if len(store) < r011.MIN_SESSION_BARS:
            self._stage(code, "wait_bars")
            return
        t = pd.Timestamp(key).time()
        if t < ENTRY_START or t > ENTRY_END or now.time() > ENTRY_END:
            self._stage(code, "outside_window")
            return
        block = None
        if self.targets and code not in self.targets:      # [2026-11-27] the method's target list: no new entry outside it
            block = "not_target"
        elif self.exclude_funds and r011.is_fund_product(name):
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

    def _on_symbol_nxt(self, code: str, name: str, now: datetime, frame_1m, price: float, group: str, sess: str,
                       was_open: bool, st: dict) -> None:
        """[2026-11-23] The same policy in an NXT session. nxt_after continues the day's regular bars (day open / VWAP / history
        from 09:01); nxt_pre has only its own bars. Entry window and force exit come from NXT_SESSIONS."""
        raw = nxt_raw_1min(frame_1m, now.date(), sess)
        if raw is None or raw.empty:
            return
        old = self.xbars.get((code, sess))
        own = raw if old is None or old.empty else pd.concat([old, raw[raw.index > old.index[-1]]])
        own = own[~own.index.duplicated(keep="last")].sort_index()
        self.xbars[(code, sess)] = own
        key = own.index[-1]
        if self.last_bar.get(code) == key:
            return  # no new completed bar
        self.last_bar[code] = key
        self._trace_pending = (code, now, pd.Timestamp(key).strftime("%H:%M"))   # [2026-11-27] the next _stage of this bar is traced
        regular = self.bars.get(code) if sess == "nxt_after" else None
        store = own if regular is None or regular.empty else pd.concat([regular, own])
        st.update(bar=pd.Timestamp(key).strftime("%H:%M"), bars=int(len(own)))
        if was_open:
            if code in self.positions:
                self._stage(code, "held", self.positions[code].get("prob"))
            else:
                self._stage(code, "sold")          # [Codex round 2] sold in this poll: the bar still has its trace row
            return
        if len(own) < r011.MIN_SESSION_BARS:
            self._stage(code, "wait_bars")
            return
        w = NXT_SESSIONS[sess]
        t = pd.Timestamp(key).time()
        if t < w["entry_start"] or t > w["entry_end"] or now.time() > w["entry_end"]:
            self._stage(code, "outside_window")
            return
        block = None
        if self.targets and code not in self.targets:      # [2026-11-27] the method's target list: no new entry outside it
            block = "not_target"
        elif self.exclude_funds and r011.is_fund_product(name):
            block = "fund"
        elif self.no_reentry_after_sl and code in self.sl_today:
            block = "sl_today"
        if self.log_only:
            self._check_entry(code, name, now, price, store, group, block=block, session=sess)
            return
        if block:
            self._stage(code, block)
            return
        max_pos, max_ent = self.limits[group]
        if self._open_count(group) >= max_pos or self.entries[group] >= max_ent:
            self._stage(code, "limit")
            return
        self._check_entry(code, name, now, price, store, group, session=sess)

    @staticmethod
    def _frames(store: pd.DataFrame, session: str):
        """(m1, m3) for compute_features. regular / nxt_after: r011.feature_frames (history window anchored at 09:01).
        nxt_pre: the session's own bars, only when they start by NXT_PRE_FIRST_BAR_BY."""
        if session != "nxt_pre":
            return r011.feature_frames(store)
        if store is None or store.empty or pd.Timestamp(store.index[0]).time() > NXT_PRE_FIRST_BAR_BY:
            return None
        m1 = r011.fast_indicators(store)
        return m1, r011.completed_3min(m1)

    def _candidate(self, code: str, name: str, now: datetime, price: float, store: pd.DataFrame, group: str, trigger: str,
                   info: dict, feats: dict | None, prob, threshold, meta: dict | None, block: str | None, reason: str,
                   session: str = "regular") -> None:
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
               "session": session,
               "bar": bar.isoformat(), "ts": now.isoformat(timespec="seconds"), "trigger": trigger,
               "trigger_info": {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, bool)) or v is None},
               "feature_version": r011.FEATURE_VERSION, "features": feats, "prob": p, "threshold": threshold, "model_id": model_id,
               "would_buy": bool(p is not None and threshold is not None and p >= threshold and not block), "block": block,
               "reason": reason, "price": price, "price_source": "live", "tick": tick, "tick_pct": tick / price * 100 if price else None,
               "label": self._model_label()}
        try:
            self.cand_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cand_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            self.log(f"{TAG} candidate log write failed ({type(exc).__name__}: {exc})")
        self._stage(code, "candidate", p)

    def _check_entry(self, code: str, name: str, now: datetime, price: float, store: pd.DataFrame, group: str = "watch",
                     block: str | None = None, session: str = "regular") -> None:
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
        frames = self._frames(store, session)
        if frames is None:
            if self.log_only:
                self._candidate(code, name, now, price, store, group, reason, info, None, None, None, None, block, "no_history", session)
                return
            self._stage(code, "no_history")
            return   # history window incomplete (late start): no prediction, no entry
        threshold = self._threshold()
        if threshold is None:
            if self.log_only:
                self._candidate(code, name, now, price, store, group, reason, info, None, None, None, None, block, "no_model", session)
                return
            self._stage(code, "no_model")
            return
        self._trace_info = {"thr": threshold}      # [2026-11-27] timeline trace: the cut this bar is judged against
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
                            prob, threshold, meta, block, why, session)
            return
        if prob is None or not r011.valid_prob(prob) or prob < threshold:
            self._stage(code, "no_model" if prob is None else "below_threshold", prob if r011.valid_prob(prob) else None)
            return
        self._open_position(code, name, now, price, group, session, prob=float(prob), threshold=threshold, reason=reason,
                            model_id=(meta or {}).get("model_id"))

    def _open_position(self, code: str, name: str, now: datetime, price: float, group: str, session: str, prob: float | None,
                       threshold: float | None, reason: str, model_id=None) -> None:
        """[2026-11-27] the paper buy itself (book, ENTRY record, log line) - split from _check_entry so another signal (r029 VWAP
        trade, prob None) opens positions the same way"""
        self._stage(code, "bought", prob)
        label = self._label()
        entry = price + krx_tick_size(price)
        pos = {"name": name, "entry": entry, "ts": pd.Timestamp(now), "tp": entry * (1 + label["tp"] / 100.0),
               "sl": entry * (1 - label["sl"] / 100.0), "horizon": label["horizon"], "prob": prob, "group": group,
               "session": session,
               # [2026-11-25] an NXT holding (kept until NXT_CLOSE): the runner trades NXT and r003 says the symbol is NXT-tradable
               "nxt": bool(self.nxt and (session != "regular" or (group == "watch" and self.nxt_ok.get(code, False))))}
        self.positions[code] = pos
        self.entries[group] += 1
        self._emit({"event": "ENTRY", "ts": now.isoformat(timespec="seconds"), "code": code, "name": name, "entry": entry,
                    "price": price, "prob": prob, "threshold": threshold, "trigger": reason, "tp_px": pos["tp"],
                    "sl_px": pos["sl"], "horizon": label["horizon"], "model_id": model_id, "group": group,
                    "session": session, "nxt": pos["nxt"]})
        self.log(f"{TAG} {code}_{name}{GROUP_KO[group]}{'' if session == 'regular' else f' [{session}]'} | PAPER BUY {entry:,.0f} {f'p={prob:.3f} thr={threshold:.3f} ' if prob is not None and threshold is not None else ''}{reason} "
                 f"| tp {pos['tp']:,.0f} sl {pos['sl']:,.0f} {label['horizon']:.0f}min | open={self._open_count(group)} "
                 f"entries={self.entries[group]}/{self.limits[group][1]}")

    def _check_exit(self, code: str, now: datetime, price: float) -> None:
        pos = self.positions[code]
        if now.time() >= close_time_of(pos, self.nxt):
            # [2026-11-25] the session close has come: session_close() fills it (auction print / NXT close), no tp / sl now
            self.session_close(now, codes=[code])
            return
        held_min = (pd.Timestamp(now) - pos["ts"]).total_seconds() / 60.0
        reason, exit_px = None, None
        if price >= pos["tp"]:
            reason, exit_px = "tp", pos["tp"]                      # a resting sell at the target
        elif price <= pos["sl"]:
            reason, exit_px = "sl", tick_down(price)
        elif held_min >= pos["horizon"]:
            reason, exit_px = "time", tick_down(price)
        if reason is None:
            return
        self._close(code, now, reason, exit_px)

    def _close(self, code: str, now: datetime, reason: str, exit_px: float, fill: str | None = None,
               estimated: bool = False) -> bool:
        """EXIT record + log line. [2026-11-25] fill = how a session close was priced (auction / last), estimated = no fresh
        price. The position is dropped only after the record was written (a failed write is retried next turn)."""
        pos = self.positions[code]
        held_min = (pd.Timestamp(now) - pos["ts"]).total_seconds() / 60.0
        net = net_pct(pos["entry"], exit_px)
        group = pos.get("group", "watch")
        rec = {"event": "EXIT", "ts": now.isoformat(timespec="seconds"), "code": code, "name": pos.get("name"),
               "reason": reason, "entry": pos["entry"], "exit": exit_px, "net_pct": net, "held_min": round(held_min, 1),
               "prob": pos.get("prob"), "group": group, "session": pos.get("session", "regular"), "nxt": bool(pos.get("nxt"))}
        if fill:
            rec["fill"] = fill
        if estimated:
            rec["estimated"] = True
        if not self._emit(rec):
            self._safe_log(f"{TAG} {code} EXIT record could not be written - position kept, retried next turn")
            return False
        self.positions.pop(code, None)
        if reason == "sl":
            self.sl_today.add(code)
        self.closed.append(net)
        wins = sum(1 for x in self.closed if x > 0)
        self._stage(code, "sold")
        self._safe_log(f"{TAG} {code}_{pos.get('name')}{GROUP_KO.get(group, '')} | PAPER SELL {exit_px:,.0f} {reason}"
                       f"{' (' + fill + (', estimated' if estimated else '') + ')' if fill else ''} net={net:+.2f}% held={held_min:.0f}min "
                       f"| day: {len(self.closed)} trades {wins} wins sum={sum(self.closed):+.2f}%")
        return True

class CollectorTickFeed:
    """[2026-11-19] Confirmed 1m bars + the last price of a symbol from the day's KRX tick file of the collector (g011).

    - incremental: each poll reads only the bytes appended since the last one (a torn last line waits for the next poll)
    - regular-session prints only (session column "regular"; files without it: 09:00 <= time < 15:20)
    - reconnect replays are dropped by acml_vol (it only grows)
    - bars are end-labeled like r003's 1m frame (09:00:xx prints -> the 09:01 bar) and handed out only once complete
      (end <= now - confirm_delay_s, floored to the minute), and only when a new one completed since the last poll
      (None otherwise - the runner then checks exits with the price)
    """

    def __init__(self, ticks_dir: Path, date_str: str, confirm_delay_s: float = 2.0, sessions: tuple[str, ...] = ("regular",)):
        self.dir = Path(ticks_dir)
        self.date_str = date_str
        self.confirm_delay_s = float(confirm_delay_s)
        self.state: dict[str, dict] = {}
        # [2026-11-25] which session column values count as prints: KRX file "regular"; the NXT file (ticks/<date>/nxt) is read
        # with ("pre", "regular", "post") for NXT orphan exits
        self.sessions = tuple(sessions)

    def codes(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.csv") if _CODE_RE.fullmatch(p.stem))

    @staticmethod
    def _new() -> dict:
        return {"off": 0, "rest": b"", "cols": None, "acml": -1.0, "buckets": {}, "price": None, "done": None, "auction": None,
                "price_t": None}

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
        if sess == "close_auction" and st.get("auction") is None:
            st["auction"] = price     # [2026-11-25] the 15:30 closing auction fill (paper regular close)
            return
        if sess not in self.sessions:
            return
        if acml <= st["acml"]:
            return            # replayed print after a reconnect
        st["acml"] = acml
        st["price"] = price
        st["price_t"] = t      # [2026-11-25] HHMMSS of the last regular print (freshness of the orphan price)
        start = t[:4]         # HHMM of the minute the print falls in
        b = st["buckets"].get(start)
        if b is None:
            st["buckets"][start] = [price, price, price, price, qty]
        else:
            b[1] = max(b[1], price)
            b[2] = min(b[2], price)
            b[3] = price
            b[4] += qty

    def last_print(self, code: str, read: bool = True) -> tuple[float, datetime] | None:
        """[2026-11-25] (price, time) of the day's last print read so far (None before the first one). read=False: as of the
        last poll() (the price and the time come from the same read)"""
        st = self.state.setdefault(code, self._new())
        if read:
            self._read(code, st)
        if st.get("price") is None or not st.get("price_t"):
            return None
        t = st["price_t"]
        base = datetime.strptime(self.date_str, "%Y%m%d")
        return float(st["price"]), base.replace(hour=int(t[:2]), minute=int(t[2:4]), second=int(t[4:6]))

    def auction_price(self, code: str) -> float | None:
        """[2026-11-25] the day's closing auction price of the symbol (the first close_auction print), None before it"""
        st = self.state.setdefault(code, self._new())
        self._read(code, st)
        return st.get("auction")

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
                               "prob": pos.get("prob"), "session": pos.get("session", "regular"), "nxt": bool(pos.get("nxt")),
                               "close": close_time_of(pos, runner.nxt).strftime("%H:%M:%S")}
        symbols.append(row)
    meta = {}
    try:
        meta = dict(runner.model.meta or {})
    except Exception:
        meta = {}
    payload = {
        "asof": now.isoformat(timespec="seconds"), "date": runner.date_str, "log_only": bool(getattr(runner, "log_only", False)),
        "close_only": bool(getattr(runner, "close_only", False)),
        "model_id": meta.get("model_id"), "threshold": runner._threshold(), "label": runner._label(),
        "model_label": runner._model_label(),
        "entry_window": [ENTRY_START.strftime("%H:%M"), ENTRY_END.strftime("%H:%M")], "force_exit": FORCE_EXIT.strftime("%H:%M"),
        # [2026-11-25] session closes: regular-only holdings / NXT holdings (null when the runner does not trade NXT)
        "close_regular": REGULAR_CLOSE.strftime("%H:%M"),
        "close_nxt": NXT_CLOSE.strftime("%H:%M:%S") if getattr(runner, "nxt", False) else None,
        # [2026-11-23] NXT sessions the runner also trades on paper (empty when off)
        "nxt": bool(getattr(runner, "nxt", False)),
        "nxt_sessions": ({k: {"entry_window": [w["entry_start"].strftime("%H:%M"), w["entry_end"].strftime("%H:%M")],
                              "force_exit": NXT_CLOSE.strftime("%H:%M:%S")} for k, w in NXT_SESSIONS.items()}
                         if getattr(runner, "nxt", False) else {}),
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

