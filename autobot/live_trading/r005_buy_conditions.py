"""R005 live buy conditions - numbered checklist for a NEW entry (used by the r003 live executor).

Overview
- r003 used to decide a new entry with ~200 inline lines inside run(). Those checks now live here, one object per
  condition, named `_001_...`, `_002_...`, ... in the exact order they are evaluated.
- `check_buy_conditions(ctx)` is the main function: it walks BUY_CONDITIONS in order and stops at the first condition
  that does not pass. A condition returns True (passed) or False (stop). A failing condition logs / mutates state at
  the same point, with the same text, as the old inline code did.
- The numeric logic of the 1-min trigger and of the 3-min context gates stays in r002 (shared with the g003
  backtester); the `_009`..`_014` conditions (numbers as of 2026-11-08) only delegate to it. The refactor itself changed no threshold.
- [2026-09-22] `_021`(당시 `_020`) now also requires the same 3-min bar and a bounded price drift between the two consecutive
  passes, and its gap window is 45 s (r001 BUY_CONFIRM_MAX_GAP_SECONDS / _REQUIRE_SAME_BAR / _MAX_PRICE_DRIFT_PCT).
  That is the only behavior change in this file since the refactor.
- [2026-11-04] `_007` stop-loss circuit breaker removed (user decision); `_009` marks the 3-min eval context with
  `triple_cross` so r002 can apply the relaxed entry-ATR floor (r001 ENABLE_RELAXED_ENTRY_ATR).
- [2026-11-04] soft gates `_010/_011/_012/_013/_016`, the score threshold of `_017` and the whole TRIPLE_CROSS_BYPASS
  mechanism removed (user decision after the 11/04 review: 14 of 17 triple buys bypassed stochastic, so the gates decided
  nothing). Decision gates are now r002 DECISION_CONTEXT_GATES (ATR floor + liquidity); `_017` only records the score.
- [2026-11-08] Renumbered `_001`..`_025` in evaluation order (numbers had drifted: added gates kept 25~31 and removed
  ones left gaps). Notes and update-log entries above this line use the numbers of their date - see the 2026-11-08 entry
  for the old -> new table.
- No broker imports: everything that talks to the broker / clock is injected through `BuyServices` and `ctx.api`, so the
  whole checklist can be unit-tested without authenticating.

Order (stage)  [numbers = evaluation order since 2026-11-08, old number in brackets]
  STATE     _001 entry time window                is_new_entry_allowed
            _002 session open warm-up             STARTUP_WARMUP_SECONDS
            _003 trade cool-down                  api._in_cooldown
            _004 same-bar signal dedup            signal_buy_bar
            _005 no open buy exposure             api.has_buy_exposure
            _006 hard-stop re-entry block         hard_stop_today_codes
            _007 daily loss limit        [_025]   (2026-10-23 신규) 포트폴리오 가드
            _008 max open positions      [_026]   (2026-10-23 신규) 포트폴리오 가드
  TRIGGER   _009 hybrid 1-min trigger    [_008]   check_buy_condition_1min_hybrid_trigger
  CONTEXT   _010 market regime (index ETF) [_029] (2026-11-04 신규) r002 market_regime_ok
            _011 3-min context frame ready [_009] run_3min_context_pipeline (prefix)
            _012 min_entry_atr_volatility [_014]  r002 DECISION_CONTEXT_GATES
            _013 min_liquidity_safety    [_015]
            _014 context score record    [_017]   score only, no threshold (2026-11-04)
  PRE_ORDER _015 excessive rise from prev close [_018] MAX_BUY_RISE_PCT_FROM_PREV_CLOSE
            _016 anti-chase day extension [_019]  (2026-10-04 신규) 당일 시가/VWAP 대비 과열 차단
            _017 opening gap / volume gate [_020] passes_opening_gap_volume_gate
            _018 quote status (VI/halt/warning) [_031] (2026-11-04 신규) 현재가 응답 상태 코드 차단
            _019 account sync freshness  [_030]   (2026-11-04 신규) 잔고 동기화 오래되면 매수 차단
            _020 edge AI meta-label filter [_028] (2026-10-26 신규) r011 LightGBM P(승) - 기본 shadow(기록만)
            _021 entry spread            [_027]   (2026-10-23 신규) 매도1-매수1 호가 스프레드
            _022 consecutive poll confirm [_021]  BUY_CONSECUTIVE_CONFIRM_COUNT
            _023 affordable buy qty      [_022]   api.get_affordable_buy_qty
            _024 fresh live price        [_023]   _is_stale_live_price_source
            _025 order-book ask not thin [_024]   _fetch_orderbook_totals (+ reservation, see docstring)

Behavior notes kept from the old inline code (do not "fix" them here - they are strategy decisions):
- _001~_004 rejects are silent and do NOT clear the confirm state; _009~_017 rejects clear it; _023/_024/_025 rejects and
  a failed order do NOT clear it.
- _022 rewrites the confirm state on every pass, also once the count is >= BUY_CONSECUTIVE_CONFIRM_COUNT.
- The 1-min frame fetch (_009), prev-close fetch (_015), day open/VWAP quote (_016), buying-power query (_023) and
  order-book query (_025) are API calls: they run only when reached, in this order.

Update log (append only):
- [2026-11-08] type=refactor owner=claude
    summary: 사용자 요청(AutoBotH 매수 평가 y축 번호가 아래->위로 순차 증가하지 않음) - 번호를 평가 순서대로 _001~_025로
      다시 매김(함수명/BuyCondition.no/_COND_* 변수 함께). 평가 순서와 로직은 무변경. 이전 -> 새 번호:
      _025->_007, _026->_008, _008->_009, _029->_010, _009->_011, _014->_012, _015->_013, _017->_014, _018->_015,
      _019->_016, _020->_017, _031->_018, _030->_019, _028->_020, _027->_021, _021->_022, _022->_023, _023->_024,
      _024->_025 (_001~_006 그대로), basic 트리거 _b08 -> _b09. SHADOW_ENTRY_CONDITIONS는 번호 범위(2~18) 대신 같은
      조건 집합을 새 번호로 명시. 로그 문구에는 번호가 없어(앱은 사유 문자열 -> 번호 매핑) 과거 로그도 그대로 읽힘.
    impact: live(r003 표시/조건 JSON r015·r016) / app(autoboth mapping.ts 동시 수정)
    compatibility: 조건 JSON의 no/func 값 변경 - AutoBotH와 함께 배포해야 함. 매도 조건 번호는 무변경.
- [2026-11-04] type=feat owner=claude
    summary: 사용자 승인(Codex 2차 안전 점검 + 시장 추세) - _029 시장 추세(지수 ETF 1분 MA5>MA20, r002 market_regime_ok),
      _030 계좌 동기화 신선도(fail-closed), _031 시세 상태(VI/임시정지/시장경고/종목상태), _025에 당일 봇 포지션 평가손익 포함.
      BuyServices: account_sync_age/quote_status/market_regime 주입(None이면 통과).
    impact: live
    compatibility: breaking (매수 감소 방향. 롤백: r001 플래그)
- [2026-11-04] type=feat owner=claude
    summary: 로그 패턴 제안 적용 - (1) _008 triple: 틱 크로스 봉 종가 대비 추격폭 가드(r001 ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD/
      TRIPLE_CHASE_FROM_CROSS_MAX_PCT, r002 check_chase_from_cross, 사유 TRIPLE_CROSS_CHASE_FROM_CROSS_x%_GT_y%).
      (2) 정규장 개장 전 TICK_ABSTAIN_NO_FILE는 종목당 1회만 로그하고 조용히 반려(BuyState.tick_nofile_logged, 11/04 3,057건/일).
      틱 밀도 하한은 r013 피드(TICK_LOW_DENSITY 기권)가 담당.
    impact: live
    compatibility: breaking (매수 감소 방향. 롤백: ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD=False)
- [2026-11-02] type=feat owner=claude
    summary: 사용자 요청(20261102 091580/105740 등 60틱 크로스 미매수) - _008 3중 트리거를 r002 evaluate_triple_trigger로 판정,
      r001 TRIPLE_1MIN_CONDITION("above": 1분 MA5>MA20 상태) / TRIPLE_TICK_CONDITION("any": 60틱 또는 120틱) 적용.
    impact: live (r003)
    compatibility: breaking (매수 증가 방향. 롤백: "cross" / "all")
- [2026-10-30] type=feat+fix owner=claude
    summary: 사용자 요청(326030 SK바이오팜 09:36 크로스 없는 continuation 매수 / 09:22 3중 골든크로스 미매수) + Codex 검토.
      r001 ENTRY_TRIGGER_MODE="triple"이면 _008이 1분 MA5>MA20(r002 find_1min_ma_golden_cross, 크로스 봉 확정 후
      TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS 이내) AND 60틱 AND 120틱 MA5>MA20 골든크로스(r013 crosses 각각 ok)일 때만 통과.
      틱 서비스 없음/예외/stale이면 반려(fail-closed). 1분 캔들/BB갭 추격가드(경과봉=1분 MA 크로스 이후 봉 수)와 실시간
      가드는 유지. 통과 시 ctx.triple_cross=True -> _context_gate가 TRIPLE_CROSS_BYPASS_CONTEXT_GATES 실패를 통과 처리,
      _017은 TRIPLE_CROSS_BYPASS_SCORE면 점수 미달도 통과(buy_reason에 BYPASS 게이트 기록). triple 모드는 basic보다 우선.
    impact: live (r003)
    compatibility: breaking (매수 조건 변경. 롤백: r001 ENTRY_TRIGGER_MODE="legacy")
- [2026-10-29] type=feat owner=claude
    summary: _008에 틱 트리거 OR 추가(사용자 요청, 003160 디아이 10:42 사례) - services.tick_trigger(r013)가 60/120틱봉
      MA5>MA20 골든크로스를 보고하고 1분 쪽에 유효한 크로스가 없으면(크로스 없음/continuation BB중심선 아래/1분 크로스
      CROSS_MAX_AGE 초과) r002 트리거를 tick_cross_label로 다시 평가(캔들/BB갭 가드 동일, 경과봉 0). 틱 경로는 실시간
      가드의 1분 BB중심선 조건을 생략(r001 TICK_TRIGGER_REQUIRE_1MIN_ABOVE_MID=False), 하락폭 조건은 유지.
      통과 시 gate_decisions에 TRIGGER_AGE/TICK_CROSS_PASS 기록.
    impact: live (r003)
    compatibility: breaking (매수 증가 방향. 롤백: r001 TICK_TRIGGER_MODE="off")
- [2026-10-28] type=fix owner=claude
    summary: _008이 트리거 통과·유효시간 판정 뒤 r002 check_1min_trigger_live_guard(frame_1min, ctx.price)를 호출 -
      형성 중인 1분봉이 직전 확정 종가 대비 HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT(0.4%) 초과 하락이거나 1분 BB중심선
      이하이면 반려(091580 상신이디피 20261028 09:35 하락 중 매수 사례, r001/r002 Update log 참조). 반려 시 기존 트리거
      반려와 같이 확인 상태 초기화. 트리거 유효시간 타이머는 이 반려로 리셋되지 않음.
    impact: live (r003)
    compatibility: breaking (매수 감소 방향. 롤백: r001 HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT=0)
- [2026-10-27] type=fix owner=claude
    summary: _008이 통과시킨 트리거 info를 ctx.trigger_info로 보관, _028이 r011에 전달(Edge AI trig_path 피처를
      실제 트리거 기준으로 계산 - Codex 1차 리뷰 ② 반영). 매수 판정 무변경.
    impact: live (shadow 기록 내용만)
    compatibility: backward-compatible
- [2026-10-26] type=feat owner=claude
    summary: 신규 _028_edge_ai_meta_filter(r011 LightGBM 메타 라벨링) - _020 뒤 / _027·_021 앞에서 평가(차단 시 확인·예약
      부작용 없음). r001 EDGE_AI_MODE: off/shadow(기본, 기록만)/auto(승격 모델만 차단)/gate. 모델·라이브러리 없음/예외는
      fail-open. 전일 종가는 _018 값, 당일 시가/VWAP은 _019 값이 없으면 fetch_day_ref_prices로 조회(정규장만, 15초 캐시).
      basic 모드와 시간차단 섀도 평가(SHADOW_ENTRY_CONDITIONS)에는 포함하지 않음.
    impact: live (r003) - 기본 shadow는 runtime/edge_ai/live/*.jsonl 기록만, 매수 판정 무변경
    compatibility: backward-compatible
- [2026-10-23] type=feat owner=claude
    summary: 포트폴리오 가드 3종 추가(기본 OFF) - _025 일일 손실 한도(live_state realized_pnl_krw), _026 봇 동시
      보유 종목 수 상한(traded_today 중 노출 남은 종목만, 장기 보유분 제외), _027 매도1-매수1 호가 스프레드(틱 수).
      번호 1~24는 유지하고 평가 순서는 튜플 위치로 정함(_025/_026은 _007 뒤, _027은 _021 연속확인 앞 -
      차단 시 확인/예약 부작용이 남지 않음). basic 모드에도 _025/_026 포함.
    impact: live (r001 ENABLE_DAILY_LOSS_LIMIT_GATE / ENABLE_MAX_OPEN_POSITIONS_GATE / ENABLE_SPREAD_GATE=True일 때만)
    compatibility: backward-compatible
- [2026-10-05] type=feat owner=claude
    summary: 사용자 요청(Codex 2차 검토 권고 "게이트 판정 기록") - _019가 판정(통과/차단)마다 BuyServices.record_gate_decision
      (선택 필드, 기본 None)으로 시가/VWAP/이격률/차단 여부를 기록. 신규 evaluate_entry_time_shadow(): _001에서
      점심 금지/진입창 때문에 막힌 종목을 상태 사본 + 무음 로그로 _002~_018 평가해 '주문 직전 단계 도달' 여부와
      과열 지표를 기록(실제 매수 판정/상태 영향 없음). r001 ENABLE_GATE_DECISION_LOG.
    impact: live (r003) - 기록 파일만 추가, 매매 판단 무변경
    compatibility: backward-compatible
- [2026-10-04] type=feat owner=claude
    summary: 사용자 요청(전체 매매내역/감시 로그 기반 조건 재검토) - 신규 _019_anti_chase_day_extension:
      전일종가 체크(_018) 직후, 당일 시가/VWAP(BuyServices.fetch_day_ref_prices, KIS 현재가 시세) 대비
      과열이면 반려(r002 anti_chase_day_gate, r001 ENABLE_ANTI_CHASE_DAY_GATE). 기존 _019~_023은 _020~_024로
      한 칸씩 밀림(로직 무변경). 점심 구간 신규 매수 금지는 _001(r003 is_new_entry_allowed)에서 처리.
      r001 Update log 2026-10-04 참조.
    impact: live (r003), g003도 같은 r002 함수로 동일 판정
    compatibility: breaking (롤백: r001 ENABLE_ANTI_CHASE_DAY_GATE=False)
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("매수/매도 컨셉 재검토", 204620 글로벌텍스프리 09:17 매수 지연 사례) + Codex
      설계검토 - _008_hybrid_1min_trigger에 진입 이벤트 유효기한 추가. update_timed_condition_state()로
      1분봉 트리거가 "계속 유효" 상태로 지속된 시간을 재고, HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS(r001,
      기본 300초) 초과 시 HYBRID_1MIN_TRIGGER_EXPIRED_*로 반려한다. 신규 상태 BuyState.buy_trigger_age_state
      추가(r003에서 주입/정리). r001 Update log 2026-09-23 참조.
    impact: live (r003), g003도 동일 로직 적용(백테스트 정합성)
    compatibility: breaking (HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS<=0이면 기존과 동일)
- [2026-09-21] type=refactor owner=claude
    summary: r003 run()의 신규 매수 판정 인라인 블록을 번호 붙은 조건 객체(_001~_023)와 메인 함수
      check_buy_conditions()로 분리. 판정 순서/로그 문구/상태 변경/API 호출 시점은 기존과 동일(동작 불변).
      Codex 사전 검토 반영: STEPS 진단은 기존과 같은 지점(_008, 트리거 판정 직후)에서 계산, 연속 손절 상태는
      run() 수명 동안 하나인 RiskState 인스턴스로 공유, session/buy_detail은 예약 직전에 생성.
    impact: live (r003)
    compatibility: backward-compatible
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from r001_define_config import (
    ANTI_CHASE_MAX_RISE_FROM_OPEN_PCT,
    ANTI_CHASE_MAX_VWAP_GAP_PCT,
    BUY_CONFIRM_MAX_GAP_SECONDS,
    ENABLE_ANTI_CHASE_DAY_GATE,
    BUY_CONFIRM_MAX_PRICE_DRIFT_PCT,
    BUY_CONFIRM_REQUIRE_SAME_BAR,
    BUY_CONSECUTIVE_CONFIRM_COUNT,
    ENABLE_OPENING_GAP_VOLUME_GATE,
    HARD_STOP_BLOCK_REENTRY_TODAY,
    HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS,
    HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS,
    ENTRY_1MIN_FRAME_REQUIRE_LATEST,
    TRADING_STRATEGY_MODE,
    CANDLE_CONFIRM_DELAY_SECONDS,
    LIVE_PRICE_STALE_TTL_SECONDS,
    MAX_BUY_RISE_PCT_FROM_PREV_CLOSE,
    MAX_BUY_RISE_SHADOW_PCT,
    STARTUP_WARMUP_SECONDS,
    ENABLE_DAILY_LOSS_LIMIT_GATE,
    DAILY_LOSS_LIMIT_KRW,
    ENABLE_MAX_OPEN_POSITIONS_GATE,
    MAX_OPEN_POSITIONS,
    ENABLE_SPREAD_GATE,
    MAX_ENTRY_SPREAD_TICKS,
    EDGE_AI_DIR,
    EDGE_AI_MIN_PROB_OVERRIDE,
    EDGE_AI_MODE,
    RUNTIME_DATA_DIR,
    TICK_TRIGGER_MODE,
    TICK_TRIGGER_REQUIRE_1MIN_ABOVE_MID,
    ENTRY_TRIGGER_MODE,
    TICK_TRIGGER_BAR_SIZES,
    TRIPLE_1MIN_MA_FAST,
    TRIPLE_1MIN_MA_SLOW,
    TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS,
    TRIPLE_1MIN_CONDITION,
    TRIPLE_TICK_CONDITION,
    TRIPLE_REQUIRE_1MIN_ABOVE_MID,
    ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD,
    ENABLE_MARKET_REGIME_GATE,
    MARKET_REGIME_FAIL_OPEN,
    ACCOUNT_SYNC_MAX_STALE_SECONDS,
    DAILY_LOSS_LIMIT_INCLUDE_UNREALIZED,
    ENABLE_QUOTE_STATUS_GATE,
    QUOTE_STATUS_BLOCK_WARN_CODES,
    QUOTE_STATUS_BLOCK_STAT_CODES,
    QUOTE_STATUS_MAX_AGE_SECONDS,
    TRIPLE_CHASE_FROM_CROSS_MAX_PCT,
    REGULAR_START,
)
from r008_strategy_v2 import krx_tick_size
from r002_strategy_core_shared import (
    BUY_SCORE_RULES,
    DECISION_CONTEXT_GATES,
    DECISION_CONTEXT_GATE_NAMES,
    _compute_bb_slope_pct,
    _evaluate_bb_mid_cross,
    _num,
    anti_chase_day_gate,
    anti_chase_day_metrics,
    build_context_eval,
    check_buy_condition_1min_hybrid_trigger,
    check_1min_trigger_live_guard,
    find_1min_ma_golden_cross,
    evaluate_triple_trigger,
    check_chase_from_cross,
    gate_steps_diagnostic,
    passes_opening_gap_volume_gate,
    update_timed_condition_state,
)


# ---------------------------------------------------------------------------
# Context / services / decision
# ---------------------------------------------------------------------------

@dataclass
class RiskState:
    """당일 손절 누적 상태. r003 run() 수명 동안 인스턴스 하나를 매도(손절 조건 갱신)가 공유한다(2026-11-04 서킷브레이커
    삭제 후 카운트는 기록/로그용).

    기존에는 run()의 재바인딩되는 지역변수 2개(hard_stop_daily_count, hard_stop_circuit_breaker_until)였다.
    날짜가 바뀌면 reset()으로 초기화한다(새 객체로 갈아끼우지 않는다 - 양쪽이 같은 인스턴스를 봐야 함)."""

    hard_stop_daily_count: int = 0

    def reset(self) -> None:
        self.hard_stop_daily_count = 0


@dataclass
class BuyState:
    """r003 run()의 가변 상태 컨테이너들(참조 전달 - 조건이 직접 변경). 틱마다 새로 재바인딩되는 traded_today를
    쓰므로 종목 순회마다 BuyState를 새로 만든다."""

    buy_confirm_state: dict[str, dict[str, object]]
    buy_trigger_age_state: dict[str, dict[str, object]]
    signal_buy_bar: dict[str, object]
    hard_stop_today_codes: set[str]
    gap_blocked_codes: set[str]
    traded_today: set[str]
    tick_nofile_logged: set[str] = field(default_factory=set)  # [2026-11-04] 개장 전 NO_FILE 1회 로그용 (r003이 하루 동안 유지)


@dataclass
class BuyServices:
    """브로커/시간/로그에 의존하는 함수들(r003이 주입). 이 모듈은 브로커를 import하지 않는다."""

    log: Callable[[str], None]
    is_new_entry_allowed: Callable[[datetime, bool], bool]
    get_session_open_datetime: Callable[[datetime, bool], datetime | None]
    get_frame_1min: Callable[[str, datetime, bool], pd.DataFrame | None]
    fetch_prev_close: Callable[[str, datetime, bool], float | None]
    rise_from_prev_close: Callable[[float, float], float | None]
    fetch_day_ref_prices: Callable[[str, datetime, bool], tuple[float | None, float | None]]
    is_stale_live_price_source: Callable[[str], bool]
    classify_buy_session: Callable[[datetime], str]
    get_order_spec: Callable[[datetime, bool], dict | None]
    fetch_orderbook_totals: Callable[[str, str], tuple[float | None, float | None]]
    format_reject_detail: Callable[..., str]
    # [2026-10-05] 게이트 판정 기록(r001 ENABLE_GATE_DECISION_LOG). None이면 기록하지 않는다.
    record_gate_decision: Callable[[dict], None] | None = None
    # [2026-10-23] best bid/ask for the spread gate (_021). None -> gate passes (flag is OFF by default anyway).
    fetch_bid_ask: Callable[[str, str], tuple[float | None, float | None]] | None = None
    # [2026-10-29] r013 틱 트리거 (code, now) -> TickTriggerResult. None이면 틱 트리거 없음(1분 트리거만).
    tick_trigger: Callable[[str, datetime], Any] | None = None
    # [2026-11-04] 안전 점검/시장 추세 (None이면 해당 조건 통과)
    account_sync_age: Callable[[], float | None] | None = None      # 마지막 성공 잔고 동기화 이후 초 (None = 없음)
    quote_status: Callable[[str], dict | None] | None = None         # 최근 현재가 응답의 상태 필드(VI/정지/경고/종목상태)
    market_regime: Callable[[str], tuple[bool | None, str]] | None = None  # 종목 -> (지수 ETF 추세 OK, 사유)


@dataclass
class BuyContext:
    """종목 1개 x 틱 1회의 신규 매수 판정 입력 + 조건들이 채우는 파생값."""

    code: str
    symbol_label: str
    current_dt: datetime
    nxt_tradeable: bool
    price: float
    price_source: str
    bar_time: Any
    buy_frame: pd.DataFrame
    cross_info: dict[str, object]
    config: Any  # r002.StrategyConfig
    api: Any  # TradingAPI (duck-typed: _in_cooldown / has_buy_exposure / get_affordable_buy_qty / live_state)
    state: BuyState
    risk: RiskState
    services: BuyServices

    # 조건이 채우는 파생값
    session_open_dt: datetime | None = None
    trigger_flag: str = "-"       # STEPS 로그의 1min_trigger(P|F|-)
    trigger_reason: str = ""
    steps_text: str = ""          # "1min_trigger(P) bb_slope_rising(P) ..." (_009에서 계산)
    eval_ctx: Any = None          # r002.BuyEvalContext
    buy_reason: str = ""
    prev_close: float | None = None
    rise_ratio: float | None = None
    day_open: float | None = None
    day_vwap: float | None = None
    qty: int = 0
    session: str = ""
    buy_detail: str = ""
    # [2026-10-15] 게이트 섀도 기록 구분용 - 실제 판정이면 "live", evaluate_entry_time_shadow의 시간차단 관찰이면 그 kind
    eval_mode: str = "live"
    # [2026-10-27] _009가 통과시킨 1분 트리거의 info(path/bars_since_cross) - _020 Edge AI 피처가 재계산 대신 사용
    trigger_info: dict | None = None
    # [2026-10-30] _009 triple 모드에서 1분 MA + 60틱 + 120틱 골든크로스가 모두 통과했으면 True(3분 소프트 게이트 BYPASS 대상)
    triple_cross: bool = False

    @property
    def norm_code(self) -> str:
        return str(self.code).zfill(6)

    @property
    def prev_bar(self) -> pd.Series:
        frame = self.buy_frame
        return frame.iloc[-2] if len(frame) >= 2 else frame.iloc[-1]

    def log(self, msg: str) -> None:
        self.services.log(msg)


@dataclass(frozen=True)
class BuyCondition:
    no: int
    func_name: str
    stage: str
    title: str
    description: str
    related_defines: tuple[str, ...]
    check: Callable[[BuyContext], bool]


@dataclass(frozen=True)
class BuyDecision:
    """approved=True면 모든 조건 통과(이때 _025의 예약이 남아 있음 - r003이 주문 후 실패 시 해제).
    approved=False면 failed가 처음 통과하지 못한 조건."""

    approved: bool
    failed: BuyCondition | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _steps_text(ctx: BuyContext, triple_cross: bool = False) -> str:
    ctx_steps = gate_steps_diagnostic(
        ctx.buy_frame, ctx.current_dt, ctx.price, ctx.cross_info, ctx.config,
        gates=DECISION_CONTEXT_GATES, triple_cross=triple_cross,
    )
    return f"1min_trigger({ctx.trigger_flag}) {ctx_steps}"


def _reject_hybrid(ctx: BuyContext, buy_reason: str) -> None:
    """트리거/3분 컨텍스트(_009~_014) 공통 리젝 로그: 확인 상태 초기화 + 지표 스냅샷 + STEPS 진단."""
    ctx.state.buy_confirm_state.pop(ctx.code, None)
    detail = ctx.services.format_reject_detail(
        buy_reason,
        ctx.buy_frame.iloc[-1],
        ctx.prev_bar,
        live_price=ctx.price,
        cross_info=ctx.cross_info,
        frame=ctx.buy_frame,
    )
    if ctx.steps_text:
        detail = f"{detail} | STEPS {ctx.steps_text}"
    ctx.log(f"  [REJECT      ] {ctx.symbol_label} | {detail}")


def _context_gate(ctx: BuyContext, gate_name: str) -> bool:
    gate = _GATES_BY_NAME[gate_name]  # [2026-11-04] 판정 게이트는 DECISION_CONTEXT_GATES 2개뿐 - 소프트 게이트/BYPASS 삭제
    passed, ctx.eval_ctx, reason = gate.eval_fn(ctx.eval_ctx)
    if not passed:
        _reject_hybrid(ctx, f"HYBRID_3MIN_CTX_{reason}")
        return False
    return True


# ---------------------------------------------------------------------------
# STATE stage
# ---------------------------------------------------------------------------

def _001_entry_time_window(ctx: BuyContext) -> bool:
    return bool(ctx.services.is_new_entry_allowed(ctx.current_dt, ctx.nxt_tradeable))


def _002_session_open_warmup(ctx: BuyContext) -> bool:
    ctx.session_open_dt = ctx.services.get_session_open_datetime(ctx.current_dt, ctx.nxt_tradeable)
    if ctx.session_open_dt is not None:
        elapsed = (ctx.current_dt - ctx.session_open_dt).total_seconds()
        if elapsed < STARTUP_WARMUP_SECONDS:
            ctx.log(
                f"  [REJECT      ] {ctx.symbol_label} | SESSION_OPEN_WARMUP | "
                f"elapsed={elapsed:.0f}s / {STARTUP_WARMUP_SECONDS}s | "
                f"session_open={ctx.session_open_dt:%H:%M:%S}"
            )
            return False
    return True


def _003_trade_cooldown(ctx: BuyContext) -> bool:
    return not ctx.api._in_cooldown(ctx.code, ctx.current_dt)


def _004_same_bar_signal_dedup(ctx: BuyContext) -> bool:
    return ctx.state.signal_buy_bar.get(ctx.code) != ctx.bar_time


def _005_no_open_buy_exposure(ctx: BuyContext) -> bool:
    if ctx.api.has_buy_exposure(ctx.norm_code):
        ctx.log(f"  [BUY SKIP    ] {ctx.symbol_label} | ALREADY_TRADED_TODAY_UNTIL_SELL")
        return False
    return True


def _006_hard_stop_reentry_block(ctx: BuyContext) -> bool:
    # 당일 HARD_STOP 발생 종목 재진입 차단
    if HARD_STOP_BLOCK_REENTRY_TODAY and ctx.norm_code in ctx.state.hard_stop_today_codes:
        ctx.log(f"  [BUY SKIP    ] {ctx.symbol_label} | HARD_STOP_REENTRY_BLOCKED_TODAY")
        return False
    return True


# [2026-11-04] _007_stoploss_circuit_breaker 삭제(사용자 결정) - 손절 3회 누적 시 60분 신규 매수 차단이 20261104
# 11:24~13:02 동안 모든 감시 종목 매수를 막았다. 번호 7은 비워 둔다(로그/분석 번호 안정성).


# [2026-10-23] Portfolio guards (r001 ENABLE_*_GATE, all OFF by default). Evaluated right after _006 (now _007/_008) / _021 before _022
# so a block never leaves confirm/reservation side effects behind. Numbers 25-27 keep the existing 1-24 stable.
def _007_daily_loss_limit(ctx: BuyContext) -> bool:
    if not ENABLE_DAILY_LOSS_LIMIT_GATE:
        return True
    realized = float((ctx.api.live_state or {}).get("realized_pnl_krw", 0.0) or 0.0)
    unrealized = 0.0
    if DAILY_LOSS_LIMIT_INCLUDE_UNREALIZED:  # [2026-11-04] 당일 봇 보유 포지션 평가손익(현재가 기준) 포함 (Codex #17)
        positions = getattr(ctx.api, "positions", None) or {}
        for _c in ctx.state.traded_today:
            _pos = positions.get(_c) or positions.get(str(_c).zfill(6))
            if not isinstance(_pos, dict):
                continue
            try:
                _q = int(_pos.get("quantity", 0) or 0)
                _bp = float(_pos.get("buy_price", 0.0) or 0.0)
                _cp = float(_pos.get("current_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if _q > 0 and _bp > 0 and _cp > 0:
                unrealized += (_cp - _bp) * _q
    total = realized + unrealized
    if total <= -abs(float(DAILY_LOSS_LIMIT_KRW)):
        ctx.log(f"  [BUY SKIP    ] {ctx.symbol_label} | DAILY_LOSS_LIMIT realized={realized:,.0f} unrealized={unrealized:,.0f} "
                f"limit=-{abs(DAILY_LOSS_LIMIT_KRW):,.0f}")
        return False
    return True


def _008_max_open_positions(ctx: BuyContext) -> bool:
    if not ENABLE_MAX_OPEN_POSITIONS_GATE:
        return True
    # Bot exposure only: codes bought today (traded_today is cleared on a full exit) that are still held/pending.
    # Long-term holdings in the same account are not counted.
    open_n = sum(1 for c in ctx.state.traded_today if c != ctx.norm_code and ctx.api.has_buy_exposure(c))
    if open_n >= int(MAX_OPEN_POSITIONS):
        ctx.log(f"  [BUY SKIP    ] {ctx.symbol_label} | MAX_OPEN_POSITIONS open={open_n} max={MAX_OPEN_POSITIONS}")
        return False
    return True


def _021_entry_spread(ctx: BuyContext) -> bool:
    if not ENABLE_SPREAD_GATE or ctx.services.fetch_bid_ask is None:
        return True
    spec = ctx.services.get_order_spec(ctx.current_dt, ctx.nxt_tradeable)
    mkt = "NX" if (spec and spec.get("exchange") == "NXT") else "J"
    try:
        bid, ask = ctx.services.fetch_bid_ask(ctx.norm_code, mkt)
    except Exception as exc:  # [2026-11-04 Codex] API 장애는 fail-open(통과) - 정상 응답의 무호가만 차단
        ctx.log(f"  [SPREAD      ] {ctx.symbol_label} | quote query failed ({type(exc).__name__}) - pass (fail-open)")
        return True
    if not (bid and ask and bid > 0 and ask >= bid):
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | SPREAD_NO_QUOTE bid={bid} ask={ask}")
        return False
    ticks = (ask - bid) / krx_tick_size(bid)
    if ticks > MAX_ENTRY_SPREAD_TICKS + 1e-9:
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | SPREAD_TOO_WIDE {ticks:.0f}ticks bid={bid:,.0f} ask={ask:,.0f} max={MAX_ENTRY_SPREAD_TICKS}")
        return False
    return True


def _010_market_regime(ctx: BuyContext) -> bool:
    """[2026-11-04] 시장 추세 게이트: 종목 시장의 지수 ETF 1분봉 MA5>MA20이 아니면 차단 (r002 market_regime_ok, r003 주입)."""
    fn = ctx.services.market_regime
    if not ENABLE_MARKET_REGIME_GATE or fn is None:
        return True
    try:
        ok, detail = fn(ctx.norm_code)
    except Exception as exc:  # 지수 프레임 오류가 매수 루프를 막으면 안 된다
        ok, detail = None, f"ERROR_{type(exc).__name__}"
    if ok is None:
        if MARKET_REGIME_FAIL_OPEN:
            return True
        ok = False
    if ok:
        return True
    ctx.state.buy_confirm_state.pop(ctx.code, None)
    ctx.log(f"  [REJECT      ] {ctx.symbol_label} | MARKET_REGIME_DOWN_{detail}")
    return False


def _019_account_sync_fresh(ctx: BuyContext) -> bool:
    """[2026-11-04] 계좌 동기화 신선도(Codex #17): 마지막 성공 잔고 조회가 ACCOUNT_SYNC_MAX_STALE_SECONDS보다 오래됐거나 없으면
    신규 매수 차단(fail-closed). 서비스 미주입이면 통과."""
    fn = ctx.services.account_sync_age
    if fn is None or ACCOUNT_SYNC_MAX_STALE_SECONDS <= 0:
        return True
    age = fn()
    if age is None or age > ACCOUNT_SYNC_MAX_STALE_SECONDS:
        ctx.state.buy_confirm_state.pop(ctx.code, None)
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | ACCOUNT_SYNC_STALE age={'none' if age is None else f'{age:.0f}s'} "
                f"max={ACCOUNT_SYNC_MAX_STALE_SECONDS:.0f}s")
        return False
    return True


def _018_quote_status(ctx: BuyContext) -> bool:
    """[2026-11-04] 시세 상태(Codex #18): 최근 현재가 응답의 VI/임시정지/시장경고/종목상태 코드가 차단 대상이면 반려.
    응답 없음/오래됨은 통과(fail-open)."""
    fn = ctx.services.quote_status
    if not ENABLE_QUOTE_STATUS_GATE or fn is None:
        return True
    st = fn(ctx.norm_code)
    if not isinstance(st, dict):
        return True
    at = st.get("at")
    if isinstance(at, datetime) and (ctx.current_dt - at).total_seconds() > QUOTE_STATUS_MAX_AGE_SECONDS:
        return True
    reasons = []
    if str(st.get("vi_cls_code", "")).strip().upper() == "Y":
        reasons.append("VI_ACTIVE")
    if str(st.get("temp_stop_yn", "")).strip().upper() == "Y":
        reasons.append("TEMP_STOP")
    warn = str(st.get("mrkt_warn_cls_code", "")).strip()
    if warn in QUOTE_STATUS_BLOCK_WARN_CODES:
        reasons.append(f"MARKET_WARN_{warn}")
    stat = str(st.get("iscd_stat_cls_code", "")).strip()
    if stat in QUOTE_STATUS_BLOCK_STAT_CODES:
        reasons.append(f"STOCK_STATUS_{stat}")
    if not reasons:
        return True
    ctx.state.buy_confirm_state.pop(ctx.code, None)
    ctx.log(f"  [REJECT      ] {ctx.symbol_label} | QUOTE_STATUS_BLOCK_{'+'.join(reasons)}")
    return False


# ---------------------------------------------------------------------------
# TRIGGER stage
# ---------------------------------------------------------------------------

# [2026-10-07] 트리거 나이/최신성 판정 기록(gate jsonl kind=TRIGGER_AGE). (결과종류, 종목, 키)별 최초 1회만 기록하고,
# 전체 발생 횟수는 별도 카운터로 누적해 기록마다 함께 남긴다(Codex 2차: dedup 건수와 전체 카운터 분리).
_TRIGGER_AGE_SEEN: set[tuple] = set()
_TRIGGER_AGE_COUNTS: dict[str, int] = {}


def _ts_text(value) -> str | None:
    try:
        return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _record_trigger_age(ctx: BuyContext, result: str, key, **extra) -> None:
    day = ctx.current_dt.strftime("%Y%m%d")
    _TRIGGER_AGE_COUNTS[result] = _TRIGGER_AGE_COUNTS.get(result, 0) + 1
    seen_key = (day, result, ctx.norm_code, _ts_text(key))
    if seen_key in _TRIGGER_AGE_SEEN:
        return
    _TRIGGER_AGE_SEEN.add(seen_key)
    rec_fn = ctx.services.record_gate_decision
    if rec_fn is None:
        return
    try:
        rec = {
            "ts": ctx.current_dt.isoformat(timespec="seconds"), "code": ctx.norm_code, "kind": "TRIGGER_AGE",
            "result": result, "price": ctx.price, "bar_time": str(ctx.bar_time),
            "cross_max_age_s": HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS, "counts": dict(_TRIGGER_AGE_COUNTS),
        }
        rec.update({k: (_ts_text(v) if isinstance(v, (pd.Timestamp, datetime)) else v) for k, v in extra.items()})
        rec_fn(rec)
    except Exception as exc:  # 기록 실패는 매수 판정에 영향 없음
        ctx.log(f"  [GATE LOG    ] {ctx.symbol_label} | TRIGGER_AGE record failed: {exc}")


def _cross_age_seconds(ctx: BuyContext, cross_bar_time) -> float:
    try:
        if cross_bar_time is not None and not pd.isna(cross_bar_time):
            return (pd.Timestamp(ctx.current_dt) - pd.Timestamp(cross_bar_time)).total_seconds()
    except Exception:
        pass
    return float("nan")


def _evaluate_tick_trigger(ctx: BuyContext, required: bool = False):
    """r013 틱 트리거 결과(TickTriggerResult) 또는 None(모드 off / 서비스 미주입 / 예외).
    required=True(triple 모드)면 TICK_TRIGGER_MODE와 무관하게 서비스가 있으면 평가한다."""
    fn = getattr(ctx.services, "tick_trigger", None)
    if fn is None or (not required and TICK_TRIGGER_MODE != "live"):
        return None
    try:
        return fn(ctx.norm_code, ctx.current_dt)
    except Exception as exc:  # 틱 피드 문제로 1분 트리거 판정을 막지 않는다
        ctx.log(f"  [TICK TRIG   ] {ctx.symbol_label} | evaluate failed: {exc}")
        return None


def _tick_may_replace_1min(ctx: BuyContext, trigger_ok: bool, trigger_reason: str, trigger_info: dict) -> bool:
    """틱 후보를 만들지 여부. 1분 트리거가 통과했으면 항상 만든다(1분 후보가 실시간 가드/크로스 나이에 걸릴 때의 대안 -
    Codex 리뷰). 1분이 실패했으면 크로스 자체가 없을 때만 만든다 - 1분이 크로스를 찾고 캔들/BB갭 가드에 걸린 경우는
    틱 경로도 같은 가드(더 좁은 BB갭 상한)를 받으므로 결과가 같다."""
    if trigger_ok:
        return True
    return trigger_reason in ("1MIN_NO_BB_MID_GOLDEN_CROSS", "1MIN_CONTINUATION_BELOW_BB_MID")


def _load_entry_frame_1min(ctx: BuyContext) -> pd.DataFrame | None:
    """_009 공통: 진입 판단용 1분봉(직전 확정분까지 최신) 또는 None(반려 로그 완료)."""
    frame_1min = ctx.services.get_frame_1min(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
    if frame_1min is None or frame_1min.empty or len(frame_1min) < 2:
        ctx.trigger_flag = "-"
        ctx.steps_text = _steps_text(ctx)
        _reject_hybrid(ctx, "HYBRID_1MIN_FRAME_UNAVAILABLE")
        return None

    # [2026-10-07 Codex 1/2차] 진입 판단용 1분봉은 직전 확정분까지 반드시 있어야 한다(캐시가 한 봉 늦으면 이전 봉으로 매수).
    # KIS는 무거래 분도 거래량 0 행을 주므로(실측) 마지막 라벨이 기대 라벨과 정확히 같아야 한다. 미래 라벨도 반려.
    if ENTRY_1MIN_FRAME_REQUIRE_LATEST:
        expected_label = pd.Timestamp(ctx.current_dt - timedelta(seconds=CANDLE_CONFIRM_DELAY_SECONDS)).floor("1min")
        try:
            last_label = pd.Timestamp(frame_1min.index[-1])
        except Exception:
            last_label = None
        if last_label is None or pd.isna(last_label) or last_label != expected_label:
            ctx.trigger_flag = "-"
            ctx.steps_text = _steps_text(ctx)
            _record_trigger_age(ctx, "FRAME_STALE", expected_label, expected_label=expected_label, last_label=last_label)
            _reject_hybrid(
                ctx,
                f"HYBRID_1MIN_FRAME_STALE_last={last_label:%H:%M}_expected={expected_label:%H:%M}"
                if last_label is not None and not pd.isna(last_label) else "HYBRID_1MIN_FRAME_STALE_last=NA",
            )
            return None
    return frame_1min


def _008_triple_cross_trigger(ctx: BuyContext, frame_1min: pd.DataFrame) -> bool:
    """[2026-10-30] triple 모드 트리거: 1분 MA5>MA20 + 60틱 + 120틱 MA5>MA20 골든크로스가 모두 유효할 때만 통과.

    - 1분: r002 find_1min_ma_golden_cross, 크로스 봉 라벨(종료시각)부터 TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS 이내.
    - 틱: r013 TickTriggerResult.crosses의 각 봉 크기(TICK_TRIGGER_BAR_SIZES)가 모두 ok(나이는 r013이 판정).
      틱 서비스 없음/예외/ABSTAIN(파일 없음, stale)이면 반려(fail-closed).
    - 그 다음 1분 캔들/BB갭 추격가드(r002, 경과봉 = 1분 MA 크로스 이후 봉 수)와 실시간 가드(하락폭 + 1분 BB중심선 위)를 적용.
    continuation/연속유효 타이머는 쓰지 않는다 - 세 크로스 각각의 나이 상한이 신호 수명을 대신한다.
    [2026-11-02] r001 TRIPLE_1MIN_CONDITION="above"면 1분은 MA5>MA20 상태만, TRIPLE_TICK_CONDITION="any"면 틱은 60/120 중 하나만
    요구(r002 evaluate_triple_trigger). 크로스가 1분 프레임 밖이면 BB갭 가드 경과봉 = HYBRID_1MIN_TRIGGER_LOOKBACK_BARS.
    """
    ma = find_1min_ma_golden_cross(frame_1min, TRIPLE_1MIN_MA_FAST, TRIPLE_1MIN_MA_SLOW)
    ma_age = _cross_age_seconds(ctx, ma["cross_bar_time"]) if ma["ok"] else float("nan")
    tick_result = _evaluate_tick_trigger(ctx, required=True)
    crosses = (getattr(tick_result, "crosses", None) or {}) if tick_result is not None else {}
    # [2026-11-02] 1분/틱 조건 판정은 r002 evaluate_triple_trigger(g003 공용) - TRIPLE_1MIN_CONDITION / TRIPLE_TICK_CONDITION
    trig = evaluate_triple_trigger(
        ma, ma_age, crosses, TICK_TRIGGER_BAR_SIZES, TRIPLE_1MIN_CONDITION, TRIPLE_TICK_CONDITION,
        TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS, TRIPLE_1MIN_MA_FAST, TRIPLE_1MIN_MA_SLOW,
    )
    missing = list(trig["missing"])
    if not crosses:  # 피드 없음/ABSTAIN: 봉 크기별 사유 대신 피드 사유 하나
        missing = [m for m in missing if not m.startswith("TICK")]
        missing.append("TICK_FEED_UNAVAILABLE" if tick_result is None else (tick_result.reason or "TICK_NO_RESULT"))
        # [2026-11-04] 정규장 개장 전(NXT 프리마켓)에는 당일 틱 파일이 아직 없어 매 폴링마다 TICK_ABSTAIN_NO_FILE 리젝이
        # 쌓였다(11/04 3,057건/일). 개장 전 NO_FILE은 종목당 1회만 로그하고 조용히 반려한다.
        if (tick_result is not None and str(tick_result.reason).startswith("TICK_ABSTAIN_NO_FILE")
                and ctx.current_dt.time() < REGULAR_START):
            ctx.state.buy_confirm_state.pop(ctx.code, None)
            if ctx.norm_code not in ctx.state.tick_nofile_logged:
                ctx.state.tick_nofile_logged.add(ctx.norm_code)
                ctx.log(f"  [TICK TRIG   ] {ctx.symbol_label} | no tick file before {REGULAR_START:%H:%M} - triple trigger idle (logged once)")
            return False
    tick_cross_times = trig["tick_cross_times"]
    ctx.trigger_flag = "F" if missing else "P"
    ctx.steps_text = _steps_text(ctx, triple_cross=True)  # triple 경로: 3분 게이트는 완화 ATR 하한으로 판정됨
    if missing:
        _reject_hybrid(ctx, f"TRIPLE_CROSS_MISSING_{'|'.join(missing)}")
        return False

    label = trig["label"]
    guard_info: dict = {}
    guard_ok, guard_reason = check_buy_condition_1min_hybrid_trigger(
        frame_1min, info=guard_info, tick_cross_label=label,
        external_bars_since_cross=trig["bars_since_cross"],
    )
    if not guard_ok:
        _reject_hybrid(ctx, f"TRIPLE_CROSS_{guard_reason}")
        return False
    live_ok, live_reason = check_1min_trigger_live_guard(
        frame_1min, ctx.price, require_above_mid=TRIPLE_REQUIRE_1MIN_ABOVE_MID,
    )
    if not live_ok:
        _reject_hybrid(ctx, f"TRIPLE_CROSS_{live_reason}")
        return False
    if ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD:  # [2026-11-04] 크로스 봉 종가 대비 추격폭 가드
        chase_ok, chase_reason = check_chase_from_cross(ctx.price, trig.get("cross_ref_price", float("nan")),
                                                        TRIPLE_CHASE_FROM_CROSS_MAX_PCT)
        if not chase_ok:
            _reject_hybrid(ctx, f"TRIPLE_CROSS_{chase_reason}")
            return False
    trigger_info = {
        "path": "triple", "cross_bar_time": ma["cross_bar_time"], "bars_since_cross": trig["bars_since_cross"],
        "run_start_time": guard_info.get("run_start_time"),
        "tick_cross_times": {n: _ts_text(t) for n, t in tick_cross_times.items()},
    }
    _record_trigger_age(
        ctx, "TRIPLE_CROSS_PASS", ma["cross_bar_time"], path="triple", trigger=guard_reason,
        cross_bar_time=ma["cross_bar_time"], cross_age_s=None if pd.isna(ma_age) else round(ma_age, 1),
        bars_since_cross=trig["bars_since_cross"], tick_cross_times=trigger_info["tick_cross_times"],
        last_1min_label=frame_1min.index[-1], last_3min_label=ctx.bar_time,
    )
    ctx.triple_cross = True
    ctx.trigger_reason = guard_reason
    ctx.trigger_info = trigger_info
    return True


def _009_hybrid_1min_trigger(ctx: BuyContext) -> bool:
    ctx.triple_cross = False
    frame_1min = _load_entry_frame_1min(ctx)
    if frame_1min is None:
        return False
    if str(ENTRY_TRIGGER_MODE).lower() == "triple":
        return _008_triple_cross_trigger(ctx, frame_1min)

    # [2026-09-09] 3분 컨텍스트가 이미 uptrend_continuation으로 판정한 상태면 그 신호를 1분 트리거에도
    # 그대로 전달한다 - 452190 한빛레이저 사례(1분 트리거가 룩백 밖의 오래/강하게 지속된 랠리를 계속 놓침)
    # 대책 중 하나.
    frame = ctx.buy_frame
    cur3 = frame.iloc[-1]
    prev3 = frame.iloc[-2] if len(frame) >= 2 else cur3
    cur3_bb = _num(cur3, "BB_MIDDLE")
    prev3_bb = _num(prev3, "BB_MIDDLE")
    context_uptrend_continuation = False
    if not any(pd.isna(v) for v in (cur3_bb, prev3_bb)):
        bb_slope = _compute_bb_slope_pct(frame)
        cross_eval = _evaluate_bb_mid_cross(
            frame, cur3, prev3, cur3_bb, prev3_bb, ctx.price, bb_slope, ctx.cross_info,
        )
        context_uptrend_continuation = bool(cross_eval.get("uptrend_continuation"))

    trigger_info: dict = {}
    trigger_ok, trigger_reason = check_buy_condition_1min_hybrid_trigger(
        frame_1min, context_uptrend_continuation=context_uptrend_continuation, info=trigger_info,
    )
    # [2026-10-29] 틱 트리거(r013): 60/120틱봉 MA5>MA20 골든크로스를 1분 트리거와 OR. 틱 후보는 같은 1분 캔들/BB갭 가드를
    # 받고(r002 tick_cross_label), 아래에서 1분 후보와 각자의 실시간 가드/크로스 나이로 평가한 뒤 하나라도 통과하면 인정.
    tick_result = _evaluate_tick_trigger(ctx)
    tick_ok, tick_reason, tick_info = False, "", {}
    if tick_result is not None and tick_result.ok and _tick_may_replace_1min(ctx, trigger_ok, trigger_reason, trigger_info):
        tick_ok, tick_reason = check_buy_condition_1min_hybrid_trigger(
            frame_1min, info=tick_info, tick_cross_label=tick_result.label,
        )
        tick_info["cross_bar_time"] = tick_result.cross_time
    any_trigger_ok = trigger_ok or tick_ok
    ctx.trigger_flag = "P" if any_trigger_ok else "F"
    ctx.steps_text = _steps_text(ctx)  # 기존과 동일하게 트리거 판정 직후, 통과 여부와 무관하게 계산
    # [2026-09-23] 1분봉 트리거가 "계속 유효" 상태로 지속된 시간을 추적한다 - 크로스 자체는 유효한데
    # 3분봉 컨텍스트 게이트가 못 따라와 대기만 길어지다가, 뒤늦게 게이트가 맞아떨어진 시점엔 이미
    # 추격매수 구간인 사례(204620 글로벌텍스프리 2026-09-23 09:04 트리거 유효 -> 09:17 매수, 13분
    # 지연) 대응. r002 check_buy_condition_1min_hybrid_trigger 자체는 무변경 - 이 함수가 반환한
    # trigger_ok를 그대로 사용해 별도로 체류시간만 잰다(g003도 동일 패턴 사용, r001 Update log 참조).
    trigger_age_seconds = update_timed_condition_state(
        ctx.state.buy_trigger_age_state, ctx.code, "1min_trigger", ctx.current_dt, any_trigger_ok,
    )
    if not any_trigger_ok:
        # 1분 크로스가 없어 틱 후보를 평가했다면 그 반려 사유(가드)가 더 구체적이다
        _reject_hybrid(ctx, f"HYBRID_1MIN_TRIGGER_{tick_reason or trigger_reason}")
        return False
    if HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS > 0 and trigger_age_seconds > HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS:
        _reject_hybrid(
            ctx,
            f"HYBRID_1MIN_TRIGGER_EXPIRED_{trigger_age_seconds:.0f}s_GT_{HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS:.0f}s",
        )
        return False
    reject_reason = None
    if trigger_ok:
        reject_reason = _1min_candidate_reject(ctx, frame_1min, trigger_info, trigger_age_seconds)
        if reject_reason is None:
            ctx.trigger_reason = trigger_reason
            ctx.trigger_info = trigger_info
            return True
    if tick_ok:
        # 틱 경로 실시간 가드: 하락폭 조건만(1분 BB중심선 조건은 r001 TICK_TRIGGER_REQUIRE_1MIN_ABOVE_MID일 때만).
        # 틱 크로스 나이는 r013이 TICK_TRIGGER_CROSS_MAX_AGE_SECONDS로 이미 판정.
        live_ok, live_reason = check_1min_trigger_live_guard(
            frame_1min, ctx.price, require_above_mid=TICK_TRIGGER_REQUIRE_1MIN_ABOVE_MID,
        )
        if live_ok:
            cross_bar_time = tick_info.get("cross_bar_time")
            _record_trigger_age(
                ctx, "TICK_CROSS_PASS", cross_bar_time, path="tick", trigger=tick_reason, cross_bar_time=cross_bar_time,
                cross_age_s=round(_cross_age_seconds(ctx, cross_bar_time), 1),
                continuity_age_s=round(trigger_age_seconds, 1), tick_reason=tick_result.reason,
                one_min_reject=reject_reason, last_1min_label=frame_1min.index[-1], last_3min_label=ctx.bar_time,
            )
            ctx.trigger_reason = tick_reason
            ctx.trigger_info = tick_info
            return True
        reject_reason = reject_reason or f"HYBRID_1MIN_TRIGGER_{live_reason}"
    _reject_hybrid(ctx, reject_reason)
    return False


def _1min_candidate_reject(ctx: BuyContext, frame_1min: pd.DataFrame, trigger_info: dict,
                           trigger_age_seconds: float) -> str | None:
    """1분 트리거 후보의 실시간 가드 + 크로스 나이 판정(기존 _009 후반부). 통과면 None, 반려면 리젝 사유 문자열."""
    # [2026-10-28] 형성 중인 1분봉 실시간 하락/BB중심선 이탈 반려(091580 사례) - 유효시간 타이머 갱신 뒤에 둬서
    # 이 반려가 확정봉 트리거의 수명을 리셋하지 않게 한다(Codex 검토).
    live_ok, live_reason = check_1min_trigger_live_guard(frame_1min, ctx.price)
    if not live_ok:
        return f"HYBRID_1MIN_TRIGGER_{live_reason}"
    path = trigger_info.get("path")
    if HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS > 0 and path == "cross":
        # [2026-10-07] 크로스 봉 확정 시각(종료 라벨)부터 잰 나이 - 연속통과 타이머와 달리 추격가드 반려로 리셋되지 않는다.
        cross_bar_time = trigger_info.get("cross_bar_time")
        cross_age = _cross_age_seconds(ctx, cross_bar_time)
        common = dict(
            path=path, cross_bar_time=cross_bar_time, bars_since_cross=trigger_info.get("bars_since_cross"),
            run_start_time=trigger_info.get("run_start_time"), continuity_age_s=round(trigger_age_seconds, 1),
            last_1min_label=frame_1min.index[-1], last_3min_label=ctx.bar_time,
        )
        if pd.isna(cross_age) or cross_age < 0:
            _record_trigger_age(ctx, "CROSS_TIME_INVALID", cross_bar_time, cross_age_s=None, **common)
            return "HYBRID_1MIN_TRIGGER_CROSS_TIME_INVALID"
        if cross_age > HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS:
            _record_trigger_age(ctx, "CROSS_EXPIRED", cross_bar_time, cross_age_s=round(cross_age, 1), **common)
            return (
                f"HYBRID_1MIN_TRIGGER_CROSS_EXPIRED_{cross_age:.1f}s_GT_{HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS:.0f}s"
                f"_cross={pd.Timestamp(cross_bar_time):%H:%M}"
            )
        _record_trigger_age(ctx, "CROSS_PASS", cross_bar_time, cross_age_s=round(cross_age, 1), **common)
    elif path in ("uptrend_3min", "uptrend_1min"):
        # continuation 통과 기록(같은 상승구간 재허용 측정용, 구간 시작 시각 기준 dedup)
        _record_trigger_age(
            ctx, "CONTINUATION_PASS", trigger_info.get("run_start_time"), path=path,
            run_start_time=trigger_info.get("run_start_time"), continuity_age_s=round(trigger_age_seconds, 1),
            last_1min_label=frame_1min.index[-1], last_3min_label=ctx.bar_time,
        )
    return None


# ---------------------------------------------------------------------------
# CONTEXT stage (3-min) - numeric logic lives in r002 gates
# ---------------------------------------------------------------------------

def _011_context_frame_ready(ctx: BuyContext) -> bool:
    eval_ctx, reason = build_context_eval(ctx.buy_frame, ctx.current_dt, ctx.price, ctx.cross_info, ctx.config)
    if eval_ctx is None:
        _reject_hybrid(ctx, reason)
        return False
    # [2026-11-04] triple 경로 표시 - r002 진입 ATR 하한 완화(min_entry_atr_threshold_pct)
    ctx.eval_ctx = replace(eval_ctx, triple_cross=bool(ctx.triple_cross)) if is_dataclass(eval_ctx) else eval_ctx
    return True


def _012_min_entry_atr_volatility(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "min_entry_atr_volatility")


def _013_min_liquidity_safety(ctx: BuyContext) -> bool:
    return _context_gate(ctx, "min_liquidity_safety")


def _014_context_score_record(ctx: BuyContext) -> bool:
    """[2026-11-04] 가점 합계는 임계값 없이 매수 사유에 기록만 한다(종전 _017_context_score_threshold - triple 경로에서
    TRIPLE_CROSS_BYPASS_SCORE로 항상 통과돼 판정 가치 0, 사용자 결정으로 삭제). 항상 True."""
    score = sum(rule.eval_fn(ctx.eval_ctx) for rule in BUY_SCORE_RULES)
    reason = f"HYBRID_3MIN_CTX_SCORE_{score}"
    if ctx.triple_cross:
        ctx.buy_reason = f"{ctx.trigger_reason}+{reason}"
    else:
        ctx.buy_reason = f"HYBRID_1MIN_TRIGGER_{ctx.trigger_reason}+{reason}"
    return True


# ---------------------------------------------------------------------------
# PRE_ORDER stage
# ---------------------------------------------------------------------------

def _015_excessive_rise_from_prev_close(ctx: BuyContext) -> bool:
    # 상승률 제한이 꺼져 있어도 전일 종가 조회(API)는 항상 수행한다(기존 동작 유지).
    prev_close = ctx.services.fetch_prev_close(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
    rise_ratio = ctx.services.rise_from_prev_close(ctx.price, float(prev_close or 0.0))
    ctx.prev_close = prev_close
    ctx.rise_ratio = rise_ratio
    if (
        MAX_BUY_RISE_PCT_FROM_PREV_CLOSE > 0
        and rise_ratio is not None
        and rise_ratio >= MAX_BUY_RISE_PCT_FROM_PREV_CLOSE
    ):
        ctx.state.buy_confirm_state.pop(ctx.code, None)
        ctx.log(
            f"  [REJECT      ] {ctx.symbol_label} | EXCESSIVE_RISE_FROM_PREV_CLOSE_"
            f"{rise_ratio*100:.2f}%_GE_{MAX_BUY_RISE_PCT_FROM_PREV_CLOSE*100:.2f}% | "
            f"prev_close={float(prev_close):,.0f} live={ctx.price:,.0f}"
        )
        return False
    if (
        MAX_BUY_RISE_SHADOW_PCT > 0
        and rise_ratio is not None
        and rise_ratio >= MAX_BUY_RISE_SHADOW_PCT
    ):
        # [2026-10-15] 관찰 전용(주문 영향 없음, r001 MAX_BUY_RISE_SHADOW_PCT) - 이 단계 통과 기록이며 이후 조건
        # (_016~_025) 통과까지 뜻하지는 않는다. record_gate_decision이 종목/kind/result별 1분 1건으로 중복을 줄이므로
        # 시간차단 관찰(eval_mode != "live") 기록은 result에 모드를 붙여 live 기록과 서로 지워지지 않게 한다(Codex 4차).
        _record_gate(ctx, kind="PREV_CLOSE_RISE_SHADOW",
                     result="would_block" if ctx.eval_mode == "live" else f"would_block@{ctx.eval_mode}", reason=None,
                     evaluation_mode=ctx.eval_mode, prev_close=prev_close, rise_ratio=rise_ratio,
                     shadow_threshold=MAX_BUY_RISE_SHADOW_PCT, live_cap=MAX_BUY_RISE_PCT_FROM_PREV_CLOSE)
    return True


def _016_anti_chase_day_extension(ctx: BuyContext) -> bool:
    """[2026-10-04] 당일 시가/VWAP 대비 이미 크게 오른 종목은 신규 매수하지 않는다(r002 anti_chase_day_gate).
    플래그가 꺼져 있거나(임계값 둘 다 <=0 포함) 정규장이 아니면 시세 조회(API)도 하지 않는다."""
    if not ENABLE_ANTI_CHASE_DAY_GATE or (
        ANTI_CHASE_MAX_RISE_FROM_OPEN_PCT <= 0 and ANTI_CHASE_MAX_VWAP_GAP_PCT <= 0
    ):
        return True
    # 정규장 진입에만 적용(NXT 세션의 시가/VWAP 기준은 미검증 - r001 주석 참조).
    if ctx.services.classify_buy_session(ctx.current_dt) != "regular":
        return True
    ctx.day_open, ctx.day_vwap = ctx.services.fetch_day_ref_prices(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
    if ctx.day_open is None or ctx.day_vwap is None:
        ctx.log(
            f"  [ANTI_CHASE  ] {ctx.symbol_label} | REF_MISSING(fail-open) | "
            f"open={ctx.day_open} vwap={ctx.day_vwap} live={ctx.price:,.0f}"
        )
    ok, reason = anti_chase_day_gate(ctx.price, ctx.day_open, ctx.day_vwap)
    _record_gate(ctx, "ANTI_CHASE", "pass" if ok else "block", reason)
    if not ok:
        ctx.state.buy_confirm_state.pop(ctx.code, None)
        open_txt = f"{ctx.day_open:,.0f}" if ctx.day_open else "nan"
        vwap_txt = f"{ctx.day_vwap:,.1f}" if ctx.day_vwap else "nan"
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | {reason} | open={open_txt} vwap={vwap_txt} live={ctx.price:,.0f}")
        return False
    return True


def _record_gate(ctx: BuyContext, kind: str, result: str, reason: str | None, **extra) -> None:
    """[2026-10-05] 게이트 판정 1건을 services.record_gate_decision으로 넘긴다(기록 실패는 매매에 영향 없음)."""
    rec_fn = ctx.services.record_gate_decision
    if rec_fn is None:
        return
    try:
        rec = {
            "ts": ctx.current_dt.isoformat(timespec="seconds"), "code": ctx.norm_code, "kind": kind,
            "result": result, "reason": reason, "price": ctx.price, "price_source": ctx.price_source,
            "bar_time": str(ctx.bar_time), "day_open": ctx.day_open, "day_vwap": ctx.day_vwap,
            "buy_reason": ctx.buy_reason or None,
        }
        rec.update(anti_chase_day_metrics(ctx.price, ctx.day_open, ctx.day_vwap))
        rec.update(extra)
        rec_fn(rec)
    except Exception as exc:  # 기록 실패는 매수 판정에 영향 주지 않는다
        ctx.log(f"  [GATE LOG    ] {ctx.symbol_label} | record failed: {exc}")


def _017_opening_gap_volume_gate(ctx: BuyContext) -> bool:
    if not ENABLE_OPENING_GAP_VOLUME_GATE:
        return True
    # passes_opening_gap_volume_gate는 심한 갭하락 종목을 gap_blocked_codes에 추가할 수 있다.
    gap_ok, gap_reason = passes_opening_gap_volume_gate(
        ctx.code, ctx.current_dt, ctx.session_open_dt, ctx.rise_ratio, ctx.buy_frame,
        ctx.state.gap_blocked_codes,
    )
    if not gap_ok:
        ctx.state.buy_confirm_state.pop(ctx.code, None)
        gap_txt = f"{ctx.rise_ratio*100:.2f}%" if ctx.rise_ratio is not None else "nan"
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | {gap_reason} | gap={gap_txt} live={ctx.price:,.0f}")
        return False
    return True


def _edge_ai_gate(config):
    from r011_edge_ai import EdgeAIGate, get_gate  # lazy: EDGE_AI_MODE="off" imports nothing
    return get_gate(lambda: EdgeAIGate(EDGE_AI_MODE, EDGE_AI_DIR, RUNTIME_DATA_DIR, EDGE_AI_DIR / "live", config,
                                       EDGE_AI_MIN_PROB_OVERRIDE))


def _020_edge_ai_meta_filter(ctx: BuyContext) -> bool:
    """[2026-10-26] 기본 매수 신호에 대한 LightGBM 메타 라벨 필터(r011). 정규장만 평가하고, 예외/모델 없음은 통과(fail-open)."""
    if str(EDGE_AI_MODE).lower() == "off" or ctx.services.classify_buy_session(ctx.current_dt) != "regular":
        return True
    try:
        if ctx.day_open is None or ctx.day_vwap is None:
            ctx.day_open, ctx.day_vwap = ctx.services.fetch_day_ref_prices(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
        frame_1min = ctx.services.get_frame_1min(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
        decision = _edge_ai_gate(ctx.config).evaluate(
            ctx.code, ctx.current_dt, float(ctx.price), frame_1min, ctx.prev_close, ctx.day_open, ctx.day_vwap,
            trigger_info=ctx.trigger_info,
        )
    except Exception as exc:
        ctx.log(f"  [EDGE AI     ] {ctx.symbol_label} | error(fail-open): {exc}")
        return True
    if decision.prob is not None:
        thr = f"{decision.threshold:.3f}" if decision.threshold is not None else "-"
        ctx.log(f"  [EDGE AI     ] {ctx.symbol_label} | p={decision.prob:.3f} thr={thr} {decision.reason} mode={EDGE_AI_MODE}")
    if decision.allow:
        return True
    ctx.state.buy_confirm_state.pop(ctx.code, None)
    ctx.log(f"  [REJECT      ] {ctx.symbol_label} | EDGE_AI_LOW_PROB_p={decision.prob:.3f}_LT_{decision.threshold:.3f}")
    return False


def _022_consecutive_poll_confirm(ctx: BuyContext) -> bool:
    """연속 BUY_CONSECUTIVE_CONFIRM_COUNT회 통과 확인. 이 지점에 도달할 때마다(결과와 무관) 상태를 갱신한다.

    직전 통과와 아래 중 하나라도 어긋나면 횟수를 1로 되돌리고 이번 통과를 새 기준으로 삼는다:
      (a) 간격 > BUY_CONFIRM_MAX_GAP_SECONDS
      (b) BUY_CONFIRM_REQUIRE_SAME_BAR이면 3분봉(bar_time)이 다름
      (c) BUY_CONFIRM_MAX_PRICE_DRIFT_PCT > 0이면 직전 통과 시점 실시간가 대비 이탈(절대값)이 이를 초과
    [2026-09-22] 정규장 활성 50종목의 평가주기(중앙 약 22초)가 예전 20초 창보다 길어 2회 연속 확인이 거의
    성립하지 않아 창을 45초로 넓히고, (b)(c)로 낡은 신호를 확인하지 않도록 했다(r001 BUY_CONFIRM_* 참조)."""
    confirm_state = ctx.state.buy_confirm_state.get(ctx.code)
    last_confirmed = confirm_state.get("confirmed_at") if confirm_state else None
    elapsed = (
        (ctx.current_dt - last_confirmed).total_seconds()
        if isinstance(last_confirmed, datetime) else None
    )
    reset_reason = ""
    if elapsed is None:
        confirm_count = 1
    elif elapsed > BUY_CONFIRM_MAX_GAP_SECONDS:
        confirm_count = 1
        reset_reason = f"GAP_{elapsed:.0f}s_GT_{BUY_CONFIRM_MAX_GAP_SECONDS:.0f}s"
    elif BUY_CONFIRM_REQUIRE_SAME_BAR and confirm_state.get("bar_time") != ctx.bar_time:
        confirm_count = 1
        reset_reason = "BAR_CHANGED"
    else:
        prev_price = confirm_state.get("price")
        drift_pct = (
            abs(ctx.price / prev_price - 1.0) * 100.0
            if isinstance(prev_price, (int, float)) and prev_price > 0 else 0.0
        )
        if BUY_CONFIRM_MAX_PRICE_DRIFT_PCT > 0 and drift_pct > BUY_CONFIRM_MAX_PRICE_DRIFT_PCT:
            confirm_count = 1
            reset_reason = f"PRICE_DRIFT_{drift_pct:.2f}%_GT_{BUY_CONFIRM_MAX_PRICE_DRIFT_PCT:.2f}%"
        else:
            confirm_count = int(confirm_state.get("count", 0)) + 1

    ctx.state.buy_confirm_state[ctx.code] = {
        "confirmed_at": ctx.current_dt, "count": confirm_count, "bar_time": ctx.bar_time, "price": ctx.price,
    }
    if confirm_count < BUY_CONSECUTIVE_CONFIRM_COUNT:
        reset_txt = f" | confirm_reset={reset_reason}" if reset_reason else ""
        ctx.log(
            f"  [BUY HOLD    ] {ctx.symbol_label} | reason=WAIT_NEXT_POLL_CONFIRM | "
            f"count={confirm_count}/{BUY_CONSECUTIVE_CONFIRM_COUNT} | "
            f"live={ctx.price:,.0f} bb_mid={_num(ctx.buy_frame.iloc[-1], 'BB_MIDDLE'):.1f} bar={ctx.bar_time:%H:%M:%S}"
            f"{reset_txt}"
        )
        return False
    return True


def _023_affordable_buy_qty(ctx: BuyContext) -> bool:
    ctx.qty = ctx.api.get_affordable_buy_qty(ctx.code, ctx.price, ctx.current_dt, ctx.nxt_tradeable)
    if ctx.qty <= 0:
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price={ctx.price:,.0f}")
        return False
    return True


def _024_fresh_live_price(ctx: BuyContext) -> bool:
    if ctx.services.is_stale_live_price_source(ctx.price_source):
        ctx.log(
            f"  [REJECT      ] {ctx.symbol_label} | STALE_LIVE_PRICE | "
            f"source={ctx.price_source} ttl={LIVE_PRICE_STALE_TTL_SECONDS}s"
        )
        return False
    return True


def _025_orderbook_ask_not_thin(ctx: BuyContext) -> bool:
    """세션/매수 상세(buy_detail)를 만들고 신규 매수를 '예약'(traded_today/signal_buy_bar)한 뒤 호가를 조회한다.
    통과하면 예약이 남는다 - r003이 이어서 place_buy_order를 호출하고 실패하면 예약을 해제한다. 매도잔량이
    얇아 거절할 때는 여기서 예약을 해제한다(확인 상태는 지우지 않음 - 기존 동작)."""
    ctx.session = ctx.services.classify_buy_session(ctx.current_dt)
    cur = ctx.buy_frame.iloc[-1]
    latest_vol = _num(cur, "volume")
    latest_vol_ma = _num(cur, "VOL_MA20")
    latest_vol_ratio = (
        latest_vol / latest_vol_ma
        if not any(pd.isna(v) for v in (latest_vol, latest_vol_ma)) and latest_vol_ma > 0
        else float("nan")
    )
    ctx.buy_detail = (
        f"reason={ctx.buy_reason} signal={ctx.cross_info.get('signal')} "
        f"live={ctx.price:,.0f} bb_mid={_num(cur, 'BB_MIDDLE'):.1f} "
        f"bar_close={_num(cur, 'close'):,.0f} ma5={_num(cur, 'MA_5'):.1f} "
        f"VOL={latest_vol:,.0f} VOLMA={latest_vol_ma:,.0f} vol_ratio={latest_vol_ratio:.4f}"
    )
    ctx.state.traded_today.add(ctx.norm_code)
    ctx.api.live_state["traded_today"] = ctx.state.traded_today
    ctx.state.signal_buy_bar[ctx.code] = ctx.bar_time

    spec = ctx.services.get_order_spec(ctx.current_dt, ctx.nxt_tradeable)
    mkt = "NX" if (spec and spec.get("exchange") == "NXT") else "J"
    ask_total, bid_total = ctx.services.fetch_orderbook_totals(ctx.norm_code, mkt)
    if ask_total is not None and bid_total is not None and bid_total > 0:
        if ask_total < bid_total * 0.5:
            ctx.log(
                f"  [REJECT      ] {ctx.symbol_label} | ORDERBOOK_ASK_THIN | "
                f"ask={ask_total:,.0f} bid={bid_total:,.0f} ratio={ask_total / bid_total:.2f}"
            )
            ctx.state.traded_today.discard(ctx.norm_code)
            ctx.api.live_state["traded_today"] = ctx.state.traded_today
            ctx.state.signal_buy_bar.pop(ctx.code, None)
            return False
    return True


# ---------------------------------------------------------------------------
# Registry + main function
# ---------------------------------------------------------------------------

def _b09_basic_1min_golden_cross(ctx: BuyContext) -> bool:
    """[2026-10-07] basic 모드 매수 트리거: 최신 확정 1분봉이 1분봉 BB중심선을 아래->위로 돌파한 봉이고
    현재가도 그 중심선 위일 때만 통과. 1분봉 캐시가 직전 확정분이 아니면 보류(ENTRY_1MIN_FRAME_REQUIRE_LATEST)."""
    frame_1min = ctx.services.get_frame_1min(ctx.code, ctx.current_dt, ctx.nxt_tradeable)
    ctx.trigger_flag = "-"
    ctx.steps_text = "basic"
    if frame_1min is None or len(frame_1min) < 2:
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | BASIC_1MIN_FRAME_UNAVAILABLE")
        return False
    expected_label = pd.Timestamp(ctx.current_dt - timedelta(seconds=CANDLE_CONFIRM_DELAY_SECONDS)).floor("1min")
    last_label = pd.Timestamp(frame_1min.index[-1])
    if ENTRY_1MIN_FRAME_REQUIRE_LATEST and last_label != expected_label:
        ctx.log(
            f"  [REJECT      ] {ctx.symbol_label} | BASIC_1MIN_FRAME_STALE_last={last_label:%H:%M}_expected={expected_label:%H:%M}"
        )
        return False
    cur, prev = frame_1min.iloc[-1], frame_1min.iloc[-2]
    cur_c, cur_m, prev_c, prev_m = _num(cur, "close"), _num(cur, "BB_MIDDLE"), _num(prev, "close"), _num(prev, "BB_MIDDLE")
    if any(pd.isna(v) for v in (cur_c, cur_m, prev_c, prev_m)):
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | BASIC_1MIN_MISSING_INDICATOR")
        return False
    if not (prev_c <= prev_m and cur_c > cur_m):
        ctx.log(f"  [REJECT      ] {ctx.symbol_label} | BASIC_NO_1MIN_GOLDEN_CROSS | close={cur_c:,.0f} mid={cur_m:,.1f}")
        return False
    if not (ctx.price > cur_m):
        ctx.log(
            f"  [REJECT      ] {ctx.symbol_label} | BASIC_LIVE_BELOW_1MIN_MID | live={ctx.price:,.0f} mid={cur_m:,.1f}"
        )
        return False
    ctx.trigger_flag = "P"
    ctx.trigger_reason = "BASIC_1MIN_BB_MID_GOLDEN_CROSS"
    ctx.buy_reason = (
        f"BASIC_1MIN_BB_MID_GOLDEN_CROSS_bar={last_label:%H:%M}_close={cur_c:,.0f}_mid1={cur_m:,.1f}"
    )
    return True


_GATES_BY_NAME = {gate.name: gate for gate in DECISION_CONTEXT_GATES}

# [2026-10-23] Portfolio guards - tuple position (not number) decides evaluation order.
_COND_007 = BuyCondition(7, "_007_daily_loss_limit", "STATE", "일일 손실 한도",
                         "당일 실현손익(live_state realized_pnl_krw, 비용 차감 추정) <= -DAILY_LOSS_LIMIT_KRW면 신규 매수 중단",
                         ("ENABLE_DAILY_LOSS_LIMIT_GATE", "DAILY_LOSS_LIMIT_KRW"), _007_daily_loss_limit)
_COND_008 = BuyCondition(8, "_008_max_open_positions", "STATE", "동시 보유 종목 수 상한",
                         "봇이 당일 매수해 보유/미체결 중인 종목 수 >= MAX_OPEN_POSITIONS면 신규 매수 중단",
                         ("ENABLE_MAX_OPEN_POSITIONS_GATE", "MAX_OPEN_POSITIONS"), _008_max_open_positions)
_COND_020 = BuyCondition(20, "_020_edge_ai_meta_filter", "PRE_ORDER", "Edge AI 메타 라벨 필터",
                         "r011 LightGBM이 예측한 P(비용 차감 순수익>0) < 임계값이면 차단(EDGE_AI_MODE auto=승격 모델만/gate). "
                         "shadow는 기록만. 모델 없음/오류는 통과",
                         ("EDGE_AI_MODE", "EDGE_AI_DIR", "EDGE_AI_MIN_PROB_OVERRIDE"), _020_edge_ai_meta_filter)
_COND_021 = BuyCondition(21, "_021_entry_spread", "PRE_ORDER", "호가 스프레드",
                         "(매도1호가-매수1호가)/호가단위 > MAX_ENTRY_SPREAD_TICKS면 차단, 호가 없음도 차단 (API: 호가 조회)",
                         ("ENABLE_SPREAD_GATE", "MAX_ENTRY_SPREAD_TICKS"), _021_entry_spread)
_COND_010 = BuyCondition(10, "_010_market_regime", "CONTEXT", "시장 추세(지수 ETF)",
                         "종목 시장(KOSPI/KOSDAQ)의 지수 ETF 1분봉 MA5 > MA20이 아니면 차단. ETF 데이터 없으면 MARKET_REGIME_FAIL_OPEN",
                         ("ENABLE_MARKET_REGIME_GATE", "MARKET_REGIME_ETF", "MARKET_REGIME_MA_FAST", "MARKET_REGIME_MA_SLOW",
                          "MARKET_REGIME_FAIL_OPEN"), _010_market_regime)
_COND_019 = BuyCondition(19, "_019_account_sync_fresh", "PRE_ORDER", "계좌 동기화 신선도",
                         "마지막 성공 잔고 조회가 ACCOUNT_SYNC_MAX_STALE_SECONDS보다 오래됐거나 없으면 신규 매수 차단(fail-closed)",
                         ("ACCOUNT_SYNC_MAX_STALE_SECONDS",), _019_account_sync_fresh)
_COND_018 = BuyCondition(18, "_018_quote_status", "PRE_ORDER", "시세 상태(VI/정지/경고)",
                         "최근 현재가 응답의 vi_cls_code=Y / temp_stop_yn=Y / 시장경고 02·03 / 종목상태 51·52·53·58·59면 차단 (오래된 응답은 통과)",
                         ("ENABLE_QUOTE_STATUS_GATE", "QUOTE_STATUS_BLOCK_WARN_CODES", "QUOTE_STATUS_BLOCK_STAT_CODES",
                          "QUOTE_STATUS_MAX_AGE_SECONDS"), _018_quote_status)

BUY_CONDITIONS: tuple[BuyCondition, ...] = (
    BuyCondition(1, "_001_entry_time_window", "STATE", "신규 진입 허용 시간대",
                 "정규장/NXT 세션별 신규 진입 마감 시각 이전인가 (마감 후에는 청산만)", (), _001_entry_time_window),
    BuyCondition(2, "_002_session_open_warmup", "STATE", "세션 개장 워밍업",
                 "세션(NXT 프리/정규/NXT 애프터) 시작 후 STARTUP_WARMUP_SECONDS 경과", ("STARTUP_WARMUP_SECONDS",), _002_session_open_warmup),
    BuyCondition(3, "_003_trade_cooldown", "STATE", "종목 거래 쿨다운",
                 "직전 주문 후 TRADE_COOLDOWN_MINUTES 경과 (api._in_cooldown)", ("TRADE_COOLDOWN_MINUTES",), _003_trade_cooldown),
    BuyCondition(4, "_004_same_bar_signal_dedup", "STATE", "같은 봉 재신호 차단",
                 "이 3분봉에서 이미 매수 신호를 낸 종목은 건너뜀 (signal_buy_bar)", (), _004_same_bar_signal_dedup),
    BuyCondition(5, "_005_no_open_buy_exposure", "STATE", "미청산 매수 노출 없음",
                 "당일 매수 후 아직 매도되지 않았거나 미체결 매수가 있으면 차단 (has_buy_exposure)", (), _005_no_open_buy_exposure),
    BuyCondition(6, "_006_hard_stop_reentry_block", "STATE", "손절 종목 당일 재진입 차단",
                 "HARD_STOP/ATR 손절이 난 종목은 당일 재진입 금지", ("HARD_STOP_BLOCK_REENTRY_TODAY",), _006_hard_stop_reentry_block),
    _COND_007,
    _COND_008,
    BuyCondition(9, "_009_hybrid_1min_trigger", "TRIGGER", "1분봉 트리거",
                 "1분봉 BB중심선 골든크로스(룩백)/우상향 지속 + 1분 캔들·BB갭 추격 가드 (r002 check_buy_condition_1min_hybrid_trigger) "
                 "+ 트리거 유효 지속시간이 HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS 초과 시 만료 반려 + 크로스 경로는 크로스 봉 확정 후 "
                 "HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS 초과 시 반려 + 1분봉 캐시가 직전 확정분이 아니면 반려 "
                 "+ 실시간가가 확정 1분봉 종가 대비 HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT 초과 하락/1분 BB중심선 이하이면 반려 "
                 "+ 60/120틱봉 MA5>MA20 골든크로스(r013)를 1분 크로스와 OR (TICK_TRIGGER_MODE)",
                 ("HYBRID_1MIN_TRIGGER_LOOKBACK_BARS", "HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT", "HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT",
                  "HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT", "HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR",
                  "HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT", "HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT",
                  "HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS", "HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS",
                  "HYBRID_1MIN_CONTINUATION_REQUIRE_ABOVE_MID", "ENTRY_1MIN_FRAME_REQUIRE_LATEST",
                  "HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT", "TICK_TRIGGER_MODE", "TICK_TRIGGER_BAR_SIZES",
                  "TICK_TRIGGER_MA_FAST", "TICK_TRIGGER_MA_SLOW", "TICK_TRIGGER_CROSS_MAX_AGE_SECONDS",
                  "TICK_TRIGGER_FEED_STALE_SECONDS", "TICK_TRIGGER_REQUIRE_1MIN_ABOVE_MID"), _009_hybrid_1min_trigger),
    _COND_010,
    BuyCondition(11, "_011_context_frame_ready", "CONTEXT", "3분봉 컨텍스트 준비",
                 "3분봉 2개 이상 + BB 지표 산출 완료", (), _011_context_frame_ready),
    BuyCondition(12, "_012_min_entry_atr_volatility", "CONTEXT", "최소 변동성(ATR%)",
                 "ATR% >= STAGED_TP1_PCT x MIN_ENTRY_ATR_TO_TP1_RATIO", ("ENABLE_MIN_ENTRY_ATR_FILTER", "MIN_ENTRY_ATR_TO_TP1_RATIO"), _012_min_entry_atr_volatility),
    BuyCondition(13, "_013_min_liquidity_safety", "CONTEXT", "최소 유동성",
                 "거래량/거래량MA/비율/거래대금 최소치", ("MIN_ENTRY_VOL_MA", "MIN_ENTRY_VOLUME", "MIN_ENTRY_TURNOVER_KRW"), _013_min_liquidity_safety),
    BuyCondition(14, "_014_context_score_record", "CONTEXT", "가점 합계 기록",
                 "가점 12항목 합계를 매수 사유(HYBRID_3MIN_CTX_SCORE_n)에 기록만 한다 - 임계값 없음(2026-11-04 삭제)",
                 (), _014_context_score_record),
    BuyCondition(15, "_015_excessive_rise_from_prev_close", "PRE_ORDER", "전일 종가 대비 과도 상승",
                 "전일 종가 대비 상승률 < MAX_BUY_RISE_PCT_FROM_PREV_CLOSE (API: 전일 종가 조회)",
                 ("MAX_BUY_RISE_PCT_FROM_PREV_CLOSE",), _015_excessive_rise_from_prev_close),
    BuyCondition(16, "_016_anti_chase_day_extension", "PRE_ORDER", "당일 과열(추격) 차단",
                 "현재가 >= 당일시가 x (1+ANTI_CHASE_MAX_RISE_FROM_OPEN_PCT%) 또는 >= 당일VWAP x (1+ANTI_CHASE_MAX_VWAP_GAP_PCT%)면 차단 "
                 "(API: 현재가 시세 stck_oprc/wghn_avrg_stck_prc, 조회 실패 시 통과)",
                 ("ENABLE_ANTI_CHASE_DAY_GATE", "ANTI_CHASE_MAX_RISE_FROM_OPEN_PCT", "ANTI_CHASE_MAX_VWAP_GAP_PCT"),
                 _016_anti_chase_day_extension),
    BuyCondition(17, "_017_opening_gap_volume_gate", "PRE_ORDER", "개장 초반 갭/거래량 게이트",
                 "개장 후 5분 이내: 갭 범위 + 거래량 폭발 확인, 큰 갭하락은 당일 영구 차단",
                 ("ENABLE_OPENING_GAP_VOLUME_GATE", "OPENING_GAP_GATE_WINDOW_MINUTES", "OPENING_GAP_MIN_PCT", "OPENING_GAP_MAX_PCT",
                  "OPENING_GAP_HARD_FLOOR_PCT", "OPENING_MIN_EARLY_VOLUME_RATIO"), _017_opening_gap_volume_gate),
    _COND_018,
    _COND_019,
    _COND_020,
    _COND_021,
    BuyCondition(22, "_022_consecutive_poll_confirm", "PRE_ORDER", "연속 폴링 확인",
                 "모든 조건이 연속 BUY_CONSECUTIVE_CONFIRM_COUNT회 통과해야 매수 - 직전 통과와 간격 BUY_CONFIRM_MAX_GAP_SECONDS 이내, "
                 "같은 3분봉(BUY_CONFIRM_REQUIRE_SAME_BAR), 실시간가 이탈 BUY_CONFIRM_MAX_PRICE_DRIFT_PCT 이내여야 연속으로 인정",
                 ("BUY_CONSECUTIVE_CONFIRM_COUNT", "BUY_CONFIRM_MAX_GAP_SECONDS", "BUY_CONFIRM_REQUIRE_SAME_BAR",
                  "BUY_CONFIRM_MAX_PRICE_DRIFT_PCT", "POLL_INTERVAL_SECONDS"), _022_consecutive_poll_confirm),
    BuyCondition(23, "_023_affordable_buy_qty", "PRE_ORDER", "매수 가능 수량",
                 "예수금/1회 주문 한도로 1주 이상 살 수 있는가 (API: 계좌 조회)", ("MAX_ORDER_AMOUNT_KRW",), _023_affordable_buy_qty),
    BuyCondition(24, "_024_fresh_live_price", "PRE_ORDER", "신선한 현재가",
                 "현재가가 stale 캐시/봉 종가 대체값이 아닌가", ("LIVE_PRICE_STALE_TTL_SECONDS",), _024_fresh_live_price),
    BuyCondition(25, "_025_orderbook_ask_not_thin", "PRE_ORDER", "호가 잔량 확인",
                 "매도호가 총잔량 >= 매수호가 총잔량 x 0.5 (API: 호가 조회). 통과 시 매수 예약 상태로 남음", (), _025_orderbook_ask_not_thin),
)

# r002에 3분 컨텍스트 게이트가 추가/삭제되면 위 번호 목록도 함께 고치도록 import 시점에 바로 실패시킨다.
_CONTEXT_GATE_FUNC_TO_NAME = {
    "_012_min_entry_atr_volatility": "min_entry_atr_volatility",
    "_013_min_liquidity_safety": "min_liquidity_safety",
}
_registered_gate_names = {
    _CONTEXT_GATE_FUNC_TO_NAME[c.func_name] for c in BUY_CONDITIONS if c.func_name in _CONTEXT_GATE_FUNC_TO_NAME
}
# [2026-11-04] r005 등록 게이트 == r002 DECISION_CONTEXT_GATE_NAMES (정확히 일치해야 한다).
if _registered_gate_names != set(DECISION_CONTEXT_GATE_NAMES):
    raise RuntimeError(
        "r005 BUY_CONDITIONS의 3분 컨텍스트 게이트와 r002 DECISION_CONTEXT_GATE_NAMES 불일치: "
        f"{sorted(_registered_gate_names)} vs {sorted(DECISION_CONTEXT_GATE_NAMES)}"
    )


def active_buy_conditions() -> tuple[BuyCondition, ...]:
    """[2026-10-07] TRADING_STRATEGY_MODE에 따른 매수 조건 목록 ("basic"이면 기본 조건, 그 외 기존 normal)."""
    # [2026-10-30] ENTRY_TRIGGER_MODE="triple"이면 basic보다 우선(basic 트리거는 3중 크로스를 강제하지 않음 - Codex 검토)
    if str(ENTRY_TRIGGER_MODE).lower() == "triple":
        return BUY_CONDITIONS
    return BASIC_BUY_CONDITIONS if str(TRADING_STRATEGY_MODE).lower() == "basic" else BUY_CONDITIONS


def check_buy_conditions(ctx: BuyContext, conditions: tuple[BuyCondition, ...] | None = None) -> BuyDecision:
    """신규 매수 메인 판정: 조건을 번호 순서대로 평가하고 처음 통과하지 못한 조건에서 멈춘다."""
    for cond in (active_buy_conditions() if conditions is None else conditions):
        if not cond.check(ctx):
            return BuyDecision(approved=False, failed=cond)
    return BuyDecision(approved=True)


# [2026-10-05] 진입 시간 관찰(shadow)에 쓰는 조건 - 주문 직전 단계(_016 이후: 과열 차단/개장 갭/연속 확인/
# 수량/현재가 신선도/호가+예약)는 제외. _016 과열 여부는 지표로만 함께 기록한다.
# [2026-11-08] 번호 재정렬 전 "2 <= no <= 18"이 고르던 집합 그대로(포트폴리오 가드 _007/_008, 시장 추세 _010 제외).
_SHADOW_ENTRY_NOS = (2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15)
SHADOW_ENTRY_CONDITIONS: tuple[BuyCondition, ...] = tuple(c for c in BUY_CONDITIONS if c.no in _SHADOW_ENTRY_NOS)


def evaluate_entry_time_shadow(ctx: BuyContext, kind: str) -> bool:
    """[2026-10-05] _001(진입 시간)에서 점심 금지/진입 허용창 때문에 막힌 종목이 그 제한이 없었다면 _002~_015(shadow 조건)를
    통과했을지 관찰한다(Codex 2차 검토: 실제 시세 기준 관찰 기록). 상태(확인/트리거 경과/중복 방지/차단 목록)는
    사본으로 평가하고 로그는 끄므로 실제 매수 판정과 13:00 이후 동작에는 영향이 없다. 모두 통과하면 당일
    시가/VWAP 과열 지표(_016 기준)를 붙여 kind로 기록하고 True를 돌려준다. 연속 확인(_022)은 평가하지 않으므로
    '주문 직전 단계 도달' 기준이다."""
    if str(TRADING_STRATEGY_MODE).lower() == "basic":
        return False  # [2026-10-07] 점심/진입창 관찰은 normal 조건 기준이라 basic 모드에서는 기록하지 않는다
    shadow_state = BuyState(
        buy_confirm_state=copy.deepcopy(ctx.state.buy_confirm_state),
        buy_trigger_age_state=copy.deepcopy(ctx.state.buy_trigger_age_state),
        signal_buy_bar=dict(ctx.state.signal_buy_bar),
        hard_stop_today_codes=set(ctx.state.hard_stop_today_codes),
        gap_blocked_codes=set(ctx.state.gap_blocked_codes),
        traded_today=set(ctx.state.traded_today),
    )
    shadow_ctx = replace(ctx, state=shadow_state, services=replace(ctx.services, log=lambda _msg: None), eval_mode=kind)
    if not check_buy_conditions(shadow_ctx, SHADOW_ENTRY_CONDITIONS).approved:
        return False
    if shadow_ctx.services.classify_buy_session(shadow_ctx.current_dt) == "regular":
        shadow_ctx.day_open, shadow_ctx.day_vwap = shadow_ctx.services.fetch_day_ref_prices(
            shadow_ctx.code, shadow_ctx.current_dt, shadow_ctx.nxt_tradeable,
        )
    _ok, _reason = anti_chase_day_gate(shadow_ctx.price, shadow_ctx.day_open, shadow_ctx.day_vwap)
    _record_gate(ctx=replace(shadow_ctx, services=ctx.services), kind=kind, result="would_reach_preorder",
                 reason=None, anti_chase_ok=_ok, anti_chase_reason=_reason)
    return True


def describe_buy_conditions() -> list[dict[str, object]]:
    """문서/발표자료용 조건 목록(번호, 함수명, 단계, 제목, 설명, 관련 상수)."""
    return [
        {"no": c.no, "func": c.func_name, "stage": c.stage, "title": c.title,
         "description": c.description, "defines": list(c.related_defines)}
        for c in BUY_CONDITIONS
    ]


# [2026-10-07] basic 모드 매수 조건 - normal의 운영 안전 조건(_001~_008, _023~_025)은 같은 함수 그대로 재사용하고
# 전략 필터(_009~_022: 1분 트리거 룩백/3분봉 컨텍스트/추격·갭 차단/연속확인)는 basic 트리거 하나로 대체한다.
_NORMAL_BY_NO = {c.no: c for c in BUY_CONDITIONS}
BASIC_BUY_CONDITIONS: tuple[BuyCondition, ...] = tuple(
    [_NORMAL_BY_NO[n] for n in range(1, 9)]  # _001~_006 + 포트폴리오 가드 _007/_008
    + [BuyCondition(9, "_b09_basic_1min_golden_cross", "TRIGGER", "[basic] 1분봉 BB중심선 골든크로스",
                    "최신 확정 1분봉이 BB중심선을 아래->위로 돌파 + 현재가 중심선 위 (1분봉 캐시 최신 필수)",
                    ("TRADING_STRATEGY_MODE", "ENTRY_1MIN_FRAME_REQUIRE_LATEST"), _b09_basic_1min_golden_cross)]
    + [_NORMAL_BY_NO[n] for n in (23, 24, 25)]
)
