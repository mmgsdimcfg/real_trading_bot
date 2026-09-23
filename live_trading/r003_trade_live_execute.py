# -*- coding: utf-8 -*-

"""R76 live trading executor - BB middle cross strategy with multi indicators.

Core idea:
1) Buy when live price crosses above BB middle and stays there long enough.
2) Sell when live price crosses below BB middle and stays there long enough.
3) Use Stochastic Fast, RSI, Williams %R as confirmation filters.
4) Use take-profit, stop-loss, and trailing-stop for risk control.

Run examples:
- python xgraph/auto_trading/r003_trade_live_execute.py
- python xgraph/auto_trading/r003_trade_live_execute.py --date 20260508
- python xgraph/auto_trading/r003_trade_live_execute.py --fake

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("매수/매도 컨셉 재검토" 4단계 적용) + Codex 설계검토 - r006 신규
      _012_peak_retracement_guard에 맞춰 peak_retrace_guard_state dict 배선(선언/SellState 주입/날짜변경
      clear()/포지션청산 시 pop()). 매도측 조건 번호가 다시 한 칸씩 밀려 "_004, _018" RiskState 공유
      주석도 "_004, _019"로 갱신. 상세는 r001/r006 Update log 2026-09-23 참조.
    impact: live
    compatibility: backward-compatible (배선만 추가, 판정 로직은 r006/r002에 있음)
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("매수/매도 컨셉 재검토" 3단계 적용) + Codex 설계검토 - r006 신규
      _011_hybrid_1min_dead_cross_exit에 맞춰 hybrid_1min_dead_cross_state dict 배선(선언/SellState
      주입/날짜변경 clear()/포지션청산 시 pop()) + SellServices에 get_frame_1min 추가(buy_services와
      동일한 _get_or_refresh_1min_frame 캐시 재사용). 매도측 조건 번호가 한 칸씩 밀려 "_004, _017"
      RiskState 공유 주석도 "_004, _018"로 갱신. 상세는 r001/r006 Update log 2026-09-23 참조.
    impact: live
    compatibility: backward-compatible (배선만 추가, 판정 로직은 r006/r002에 있음)
- [2026-09-23] type=feat owner=claude
    summary: 사용자 요청("매수/매도 컨셉 재검토" + Codex 설계검토) - r005 _008_hybrid_1min_trigger의 신규
      진입 이벤트 유효기한(HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS)에 맞춰 buy_trigger_age_state dict를
      신설(선언/BuyState 주입/날짜변경 clear()/포지션보유전환 및 매수체결 시 pop()) - buy_confirm_state와
      동일한 배선 패턴. 상세는 r001/r005 Update log 2026-09-23 참조.
    impact: live
    compatibility: backward-compatible (배선만 추가, 판정 로직 변경은 r005/r002에 있음)
- [2026-09-23] type=fix owner=claude
    summary: 사용자 요청(119850 지엔씨에너지 2026-09-23 09:02 하드손절 직후 반등 사례 + Codex 검토) - r006
      _004_hard_stop_loss에 확인창(HARD_STOP_CONFIRM_SECONDS) 추가에 맞춰 이 파일에 hard_stop_confirm_state
      dict를 신설(선언/SellState 주입/날짜변경 clear()/포지션청산 시 pop()) - 다른 확인창 상태
      (atr_stop_confirm_state 등)와 동일한 배선 패턴. 상세는 r001/r006 Update log 2026-09-23 참조.
    impact: live
    compatibility: backward-compatible (배선만 추가, 판정 로직 변경은 r006에 있음)
- [2026-09-21] type=refactor owner=claude
    summary: 사용자 요청 - run()의 신규 매수 판정(약 200줄)과 보유 종목 매도/포지션 관리(약 680줄) 인라인 블록을
      번호 붙은 조건 객체와 메인 함수로 분리(동작 불변). 매수 = r005_buy_conditions(_001~_023,
      check_buy_conditions), 매도 = r006_sell_conditions(_001~_020, evaluate_sell_conditions). 두 모듈은 브로커를
      import하지 않고 이 파일의 함수(BuyServices/SellServices)와 TradingAPI(ctx.api)를 주입받는다. run()의
      재바인딩 지역변수 hard_stop_circuit_breaker_until/hard_stop_daily_count는 run() 수명 동안 하나인
      RiskState(r005) 인스턴스로 바뀌었고 날짜 변경 시 risk.reset()으로 초기화한다. 로그 문구/순서, 주문·API
      호출 시점, 상태 변경은 기존과 같다(원본 인라인 텍스트와 새 구조를 같은 시나리오/가짜 API로 돌려 로그·API
      호출·상태를 비교하는 차등 테스트로 확인). 리팩터링으로 쓰지 않게 된 r001/r002 import 54개 삭제(pyflakes).
      알려진 기존 동작(이번에 미변경, 별도 결정 필요): (1) _020 연속 확인창 20초 < 정규장 종목당 평가주기 ~22초라
      2회 연속 확인이 거의 성립하지 않음 -> [2026-09-22] 45초 + 같은 3분봉 + 가격 이탈 <=0.3%로 완화 적용(r001
      Update log 2026-09-22 참조), (2) 확인 상태는 _021/_022/_023 거절과 주문 실패 때 지워지지 않음,
      (3) hard_stop_daily_count는 "연속"이 아니라 하루 누적 횟수. 상세는 r005 _020 docstring 참조.
    impact: live
    compatibility: backward-compatible
- [2026-09-21] type=fix owner=claude
    summary: 사용자 요청(r001 Update log 2026-09-21 참조). (1) 1차 익절 목표를 r002 compute_staged_tp1_target_pct로
      계산(ATR 익절선 상한) - 023160 태광 2026-09-21 09:01 전량 일괄 매도(고점 +2.79%, 트레일이 TP1보다 먼저 무장)
      재발 방지. (2) 급등 사다리 익절 - 1차 익절 시점에 detect_price_surge로 급등이면 pos["surge_ladder"]=
      {base(TP1 체결가), tp2_done, tp3_done}를 세우고, next_surge_ladder_action으로 +2%(진입수량 30%)/+4%(잔량 전량)
      순차 익절(reason TP2_/TP3_SURGE_LADDER_..., TP1에는 _SURGE 접미사). 종목별 최근 가격 표본은 run()의
      recent_price_samples(메모리 전용), TP1 체결가는 TradingAPI.last_sell_fill_price(_confirm_pending_sell 기록)에서
      읽고 체결 확인 전이면 트리거 시점 현재가로 대체. surge_ladder는 재시작/계좌 동기화에도 유지되도록 live_state
      저장·복원 6개 경로(_serialize_live_state/load_live_state/_apply_persisted_position_meta/_record_position_meta/
      sync_positions_from_account/신규 매수 체결 시 초기화)에 추가하고 _sanitize_surge_ladder로 검증. 사다리 가동 중엔
      기존 TP2(STAGED_TP2_RATIO>0)와 TP3_TRAIL_ARMED 로그를 건너뜀. 사다리 TP3(잔량 전량)는 포지션이 닫히므로
      _record_position_meta를 호출하지 않는다(닫힌 포지션 메타 부활 방지).
      주의(기존 동작, 이번에 미변경): place_sell_order는 모든 매도 후 TRADE_COOLDOWN_MINUTES(3분)간 같은 종목의
      후속 매도를 막는다 - TP1 직후 3분간 사다리/트레일/손절 주문이 지연됨(플래그는 주문 성공 시에만 세워지므로
      사다리는 쿨다운이 풀린 뒤 다음 틱에 재시도됨).
    impact: live
    compatibility: breaking (r001 두 플래그 ENABLE_TP1_CAP_AT_ATR_TP/ENABLE_SURGE_LADDER_TP=False로 즉시 롤백)
- [2026-09-20] type=refactor owner=claude
    summary: 불필요 코드 정리 + 사용자 결정 반영(r002/r001/g003 Update log 2026-09-20 참조).
      (1) 미사용 import 22개(inspect, datetime.time as dt_time, typing.Optional, r001 상수 19개),
      미사용 지역변수 2개(code_label, intrabar_elapsed_seconds), 어디서도 호출하지 않는 함수 2개
      (load_today_buy_codes, is_open_trading_day) 삭제(pyflakes 기준, 동작 불변).
      (2) 죽은 매수 경로 삭제 - 하이브리드가 유일한 신규 매수 경로: 1분봉 골든크로스 단독
      (ENABLE_1MIN_GOLDEN_CROSS_BUY) 분기와 gc_confirm_state, 1분봉 Entry Score 게이트
      (ENABLE_1MIN_ENTRY_SCORE_GATE - 하이브리드가 켜져 있으면 도달 불가였음), 3분봉 단독 else 분기,
      check_buy_condition_1min/check_buy_condition 래퍼, bar_time_for_buy 별칭(=bar_time), 관련 r001
      플래그 import. _buy_reject_detail의 3분봉 단독 사유 상세 분기(BB_SLOPE_NOT_RISING/
      NO_BB_MID_CROSS_UP/CANDLE_NOT_BULLISH/BB_UPPER_GAP_TOO_SMALL/LOW_VOLUME_RATIO/LOW_SCORE)도
      삭제 - 하이브리드 사유는 항상 "HYBRID_"로 시작해 한 번도 걸리지 않았으므로 로그 출력 불변.
      (3) check_buy_condition_1min_hybrid_trigger/_passes_opening_gap_volume_gate를 r002 공용 함수
      (check_buy_condition_1min_hybrid_trigger/passes_opening_gap_volume_gate)로 이동.
      (4) 같은 날 재매수 허용(사용자 결정): ALLOW_REBUY_SAME_CODE는 실전이 읽은 적 없는 죽은
      설정이라 import/정의를 삭제 - 기존 동작 그대로(보유/주문중, 쿨다운 3분, HARD_STOP 재진입
      차단만 적용). 단, _rebalance_active_watchlist가 매수 체결 종목을 active_set에서 졸업시키고
      청산 뒤에도 되돌리지 않아 실전에서는 그날 재평가 자체가 안 된다(g003은 재진입 허용) -
      재매수를 실제로 가능하게 하는 변경은 하지 않았고 별도 결정 사항으로 남김.
      (5) 거래량 하한 통일: 1분봉 트리거의 전용 하한(HYBRID_1MIN_MIN_ENTRY_*)을 삭제하고 3분봉
      min_liquidity_safety만 사용(전략 변경, r002 Update log 참조).
    impact: live ((1)~(4) 로직 불변 / (5) 하이브리드 트리거 통과 조건 완화)
    compatibility: (1)~(4) backward-compatible / (5) breaking - 매수 신호 빈도가 소폭 늘 수 있음
- [2026-09-18] type=feat owner=claude
    summary: 사용자 요청 - _passes_intrabar_entry_gate()의 RSI 밴드를 50~70에서 45~65
      (INTRABAR_RSI_MIN/MAX, r001)로 낮추고, 65 초과~75(신규 INTRABAR_RSI_EXT_MAX)
      구간은 무조건 INTRABAR_RSI_..._OUT_OF로 거부(확정 3분봉 폴백)하는 대신 조건부
      허용 경로를 신설했다: RSI가 직전봉 대비 상승 중 + MACD가 직전봉 대비 상승 중 +
      거래량이 VOL_MA20을 상회 + 종가가 BB중간선 위, 4개 조건을 전부 만족해야
      통과("INTRABAR_RSI_{rsi}_EXT_COND_NOT_MET"으로 실패 사유 구분). 거래량 조건은
      "거래량증가"를 직전봉 대비 단순 비교가 아니라 VOL_MA20 대비로 해석했다 - 이
      저장소에 직전봉 대비 거래량 비교를 노이즈성 신호로 보고 폐기한 선례가 있음
      (R77-B, volume_up_direction 가점 삭제, r002 Update log 2026-09-13 참조).
      20260918 000500 가온전선 사례(14:23~14:25 RSI 72.9~79.6로 반복 BAR_SKIP, 그
      시간대 실제로는 강상승 지속 중)를 계기로 요청됨. 75 초과는 기존과 동일하게
      무조건 거부.
    impact: live (_passes_intrabar_entry_gate만 해당 - 인트라바 프레임을 신뢰할지
      여부만 바뀌고, 최종 매수 판정 게이트/점수 로직 자체는 불변)
    compatibility: breaking (RSI 45~65는 기존보다 넓어 통과가 늘고, 65~75는 4조건
      충족 시에만 통과 - 순효과는 g003 백테스트로 확인 필요. 45 미만~50 구간은
      기존엔 거부였다가 이번에 새로 허용되는 점도 주의)
- [2026-09-18] type=fix owner=claude
    summary: 사용자 요청 - [REJECT] 로그가 콘솔에 GATES/STEPS 진단 블록까지 전부 찍혀
      너무 길어지는 문제 완화. 신규 Formatter _ConsoleTruncateRejectFormatter를 콘솔용
      StreamHandler에만 setFormatter로 덮어씌워, 메시지에 "[REJECT"가 있으면 " | GATES"
      이전까지만(reason + _buy_reject_detail이 reason에 붙이는 짧은 수치, 예:
      BB_SLOPE_NOT_RISING_-2.07% | bb_mid=... bb_slope=...%)만 남기고 그 뒤(GATES
      스냅샷 + STEPS 게이트별 pass/fail 요약)는 잘라낸다. FileHandler/
      _PerSymbolFileHandler는 이 Formatter를 쓰지 않아(각자 basicConfig 공용
      Formatter를 그대로 유지) 파일에는 전체 내용이 그대로 남는다 - 로그 분석(REJECT
      사유 상세 확인)은 파일 기준으로 계속 가능.
    impact: live (콘솔 출력만 변경 - 파일 로그/판정 로직 전부 불변)
    compatibility: backward-compatible (로그 내용/파일 위치 변화 없음, 콘솔 가독성만 개선)
- [2026-09-18] type=fix owner=claude
    summary: 사용자 요청 - [CHECK] 로그가 종목 순회마다 찍혀 콘솔이 CHECK 줄로 도배되는
      문제 완화. _rotate_logging_for_date()가 붙이는 콘솔 StreamHandler(sys.stdout)에만
      신규 필터 _SuppressConsoleCheckLogs를 추가해 메시지에 "[CHECK"가 포함된 로그를
      콘솔 출력에서만 제외 - FileHandler(메인 로그 파일)와 _PerSymbolFileHandler(종목별
      txt)는 그대로 전부 기록되므로 사후 로그 분석에는 영향 없음. FileHandler가
      StreamHandler의 서브클래스라 isinstance()로는 콘솔 핸들러만 골라낼 수 없어
      type() 정확 비교로 구분.
    impact: live (콘솔 출력만 변경 - 파일 로그/판정 로직 전부 불변)
    compatibility: backward-compatible (로그 내용/파일 위치 변화 없음, 콘솔 가독성만 개선)
- [2026-09-18] type=fix owner=claude
    summary: 사용자 요청("오늘 매수 3건, 익절 가능했던 매수를 놓친 경우 점검") -
      20260918 REJECT 로그 분석 결과 score 15/24 이상 고품질 신호가 CHASE_BUY_BB_GAP
      하나로 반복 반려된 종목 다수(000500 가온전선 +22.7%/043260 성호전자 +21.3%/
      024840 KBI메탈 +16.0% 등, 반려 시점 대비 당일 고점) 확인. check_buy_condition_
      1min_hybrid_trigger()의 BB갭 상한(HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT=1.1%)이
      uptrend_continuation 경로(trigger_reason이 1MIN_UPTREND_CONTINUATION_3MIN_CTX
      또는 1MIN_UPTREND_CONTINUATION - 추세 지속이 다른 지표로 이미 검증된 경우)에도
      신선한 크로스와 동일하게 적용되고 있었음 - 1분봉 BB중간선(후행 SMA)이 급등을
      못 따라가는 날엔 갭이 영구히 이 상한을 넘어 남은 장중 내내 진입이 막히는 구조
      (028050 삼성E&A는 09:11 score=20으로 이 사유 반려 후 09:34 score=13으로 신호가
      식고 나서야 다른 경로로 겨우 진입). uptrend_continuation 경로에서만 신규 상수
      HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT(2.2%, r001)를 쓰도록 분기 추가 -
      신선한 크로스 경로는 기존 1.1%를 그대로 유지.
    impact: common (r003 실전/g003 check_buy_condition_1min_hybrid_trigger_sim 동시
      반영 - live/sim parity 유지)
    compatibility: breaking (uptrend_continuation 경로의 매수 판정 결과가 바뀜 - 그
      경로의 매수 빈도가 늘어날 것으로 예상, g003 --date 20260918 백테스트로 검증
      권장, 라이브 반영 전 사용자 승인 필요)
- [2026-09-18] type=fix owner=claude
    summary: 사용자가 183300 코미코 실매매 사례(08:27:28 매수 신호/주문 제출 -> 08:28:29
      최종 체결, 61초 소요)를 보고 매수 체결이 늦은 것 아니냐고 질의, 로그 분석 결과
      신호/주문 자체는 제때 나갔지만(08:27:26봉 확정 직후 제출) 체결 확인 경로가 느렸던
      것으로 확인됨 - refresh_pending_orders()가 ORDER_STATUS_POLL_INTERVAL_SECONDS(당시
      15초) 전역 타이머로 스로틀되어 있어, 미체결 재주문 사이클의 (1)미체결 확인+취소
      트리거 (2)취소 확인+재주문 (3)체결 확인 3단계가 전부 이 15초 간격에 순차적으로
      걸려 누적 지연됨(2026-09-14에 메인 루프 자체 재시작은 즉시화했지만 미결 주문 상태
      폴링은 별도 타이머라 그 개선 혜택을 못 받고 있었음, r001 Update log 2026-09-18
      참조). ORDER_STATUS_POLL_INTERVAL_SECONDS를 5초로 단축.
    impact: live (refresh_pending_orders 폴링 간격만 변경, 판정 로직 불변)
    compatibility: breaking (미결 주문 상태 조회 API 호출 빈도 최대 3배 증가 - KIS
      레이트리밋 여부 관찰 권장)
- [2026-09-18] type=feat owner=claude
    summary: 사용자 요청 - place_buy_order()의 신규 진입 매수(피라미딩 제외)를 시장가
      leg + 매수1호가 지정가 leg 두 조각(각각 BUY_SPLIT_MARKET_RATIO=50%)으로 분할
      제출한다. 시장가 leg는 KRX 정규장은 실제 시장가(ord_dvsn=01), NXT는 시장가
      미지원이라 기존 관례(place_sell_order/재주문 최종단계와 동일)대로 매도1호가
      크로싱 지정가로 대체 - 급등 추격 중에도 즉시 체결 가능성을 확보한다. 지정가 leg는
      기존과 동일하게 매수1호가에서 시작해 BUY_ORDER_REPRICE_AFTER_SECONDS 경과 시
      추격 재주문(bid->ask->market) 한다 - 평균 매수가를 낮추는 역할. 두 leg는
      pending_orders[code]["legs"]에 독립 추적되고(_refresh_split_buy_legs가 처리)
      모두 종결돼야 _confirm_pending_buy를 합산 1회 호출한다. 수량이 1주뿐이면 분할해도
      지정가 쪽에 남는 수량이 없어 무의미하므로 분할하지 않고 전량 시장가 1건으로
      즉시 매수한다. 한쪽 leg 제출이 실패해도 다른 leg만으로 정상 진행(legs 1개 -> 기존
      단일주문 경로로 자연 축소).
    impact: live (신규 진입 매수만 해당 - 피라미딩 추가매수/매도 로직은 불변)
    compatibility: breaking (신규 진입 시 브로커 주문이 1건->최대 2건으로 증가, 평균
      체결가/체결 속도가 달라짐 - 실매매로 체결가 개선 여부 확인 권장)
- [2026-09-17] type=fix owner=claude
    summary: 매수 0건 로그 분석(r002 Update log 2026-09-17 참조, 사용자 요청 "20년차
      트레이더 관점 매수조건 점검") 중 발견한 표시 전용 버그 수정. _buy_condition_
      snapshot()의 "score=X/22"와 _buy_reject_detail() LOW_SCORE 분기의 "total=X/22"가
      전부 이 파일 로컬 _buy_support_score()(r002 BUY_SCORE_RULES를 손으로 복제한
      2026-09-07 당시 버전)를 쓰고 있었는데, 2026-09-13 R77 리팩터가 r002에만 반영되고
      이 복제본엔 반영되지 않아 드리프트됨 - 이미 삭제된 volume_up_direction(+1) 가점을
      여전히 더하고, 신규 pre_cross_accumulation(+1)/bb_upper_room_atr(최대 +2) 가점은
      누락, 분모도 실제 24점 만점을 22로 표시. 즉 2026-09-13 이후 CHECK/REJECT 로그에
      찍힌 모든 점수 표시가 실제 매수 판정(run_3min_context_pipeline이 쓰는 r002 실제
      BUY_SCORE_RULES 합계)과 달랐음 - 판정 로직 자체는 항상 올바른 함수를 썼으므로 실제
      매수/거부 결과에는 영향 없었고, 사람이 로그를 읽고 진단할 때만 오도했다(이번
      분석에서도 로그의 "score=X/22"를 신뢰하지 않고 소스코드로 직접 재검증해서 발견).
      로컬 재구현을 지우고 r002._buy_support_score를 그대로 import해 위임하도록 변경,
      "/22" 하드코딩 2곳을 BUY_SCORE_RULES에서 동적으로 계산한 BUY_SCORE_MAX(현재 24)로
      교체 - 앞으로 BUY_SCORE_RULES가 바뀌어도 표시가 다시 드리프트되지 않는다. 이제
      쓰이지 않게 된 EMA_TREND_ALIGN_SCORE/REQUIRE_OBV_SIGNAL_CROSS/
      OBV_BREAKOUT_LOOKBACK_BARS/OBV_CONFIRM_SCORE import도 함께 제거(죽은 코드).
    impact: live (로그 표시값만 변경, 실제 매수/거부 판정 로직은 원래도 항상 정확한
      r002 함수를 썼으므로 불변)
    compatibility: backward-compatible (로그에 찍히는 점수 숫자와 만점 표기만 정확해짐 -
      이 숫자를 파싱해 특정 값과 비교하는 외부 스크립트가 있다면 만점이 22->24로 바뀐 점
      확인 필요)
- [2026-09-14] type=fix owner=claude
    summary: 메인 루프 턴 종료 후 대기 로직을 LIVE_PRICE_POLL_INTERVAL_SECONDS(10초)
      벽시계 정렬 틱(_sleep_until_next_tick) 방식에서 즉시 재시작 방식으로 변경 - 사용자가
      실매매 로그(예: 13:19:40경 한 턴의 종목 순회가 1초 미만으로 끝났는데 다음 종목
      CHECK 로그가 13:19:50에야 찍힘)를 보고, 턴 처리 자체는 빨리 끝나는데도 다음
      00/10/.../50초 정렬 틱까지 최대 10초를 그냥 대기하는 것을 지적하며 턴 종료 즉시
      다음 턴을 시작하도록 요청. run() 최하단(while True 루프 끝)의
      `served_tick = _sleep_until_next_tick(served_tick, LIVE_PRICE_POLL_INTERVAL_SECONDS)`
      를 제거하고, 이번 턴 시작 시각(current_dt) 기준 경과시간이 신규
      MAIN_LOOP_MIN_CYCLE_SECONDS(r001, 1초)에 못 미칠 때만 그 차이만큼만 대기하도록
      변경 - 활성 종목이 있으면 사실상 종목별 API 왕복시간(수백 ms)만큼만 지나면 바로
      다음 턴이 시작되고, 활성 종목이 0개라 턴이 즉시 끝나는 경우에만 계좌/미체결 동기화
      API 스팸을 막는 안전 바닥으로 최소 간격이 적용됨. 정규장/NXT 외 유휴 대기 및
      동시호가 구간 대기(4109줄/4129줄, `continue`로 되돌아가는 두 지점)는 "턴을 아예
      돌지 않는" 경우라 기존 LIVE_PRICE_POLL_INTERVAL_SECONDS 정렬 틱 방식 그대로 유지.
    impact: live
    compatibility: breaking (활성 종목이 있을 때 신규 진입/청산 재평가 빈도가 종목당
      API 왕복시간 수준으로 크게 증가 - 이전엔 최소 10초 고정 간격이었음. 브로커(KIS)
      API 호출 빈도가 늘어나므로 실매매에서 레이트리밋/오류 발생 여부 관찰 필요)
- [2026-09-13] type=refactor owner=claude
    summary: 로그 출력 포맷 재정리 (사용자 요청, 실제 로그 파일 라인 기준). 종목별 폴링
      루프 태그(CHECK/REJECT/SKIP/PENDING/BUY HOLD, 그리고 INTRABAR_SKIP->BAR_SKIP로
      개명)를 전부 "[태그(8자 고정폭)] {symbol_label} | detail" 순서로 통일 - 이전엔
      SKIP/PENDING은 태그가 먼저, CHECK/REJECT/BUY HOLD/INTRABAR_SKIP은 종목이 먼저
      오는 등 태그별로 순서가 제각각이었음. log_trade()의 공용 [TRADE] 래퍼 태그도
      동일하게 8자 고정폭([TRADE   ])으로 맞추고, BUY REPRICE RESUBMIT 로그 1건을
      예시로 삼아 메시지 내용도 "{symbol_label} | 이벤트 설명 | ..." 순서로 재구성
      (기존 code(name) 괄호 표기 대신 다른 태그들과 동일한 _symbol_log_label 사용) -
      다른 log_trade() 호출부(BUY pending/SELL EXECUTED 등)는 메시지 구조가 제각각이라
      이번엔 손대지 않음, 필요시 후속 작업.
    impact: live
    compatibility: backward-compatible (로그 표시 형식만 변경, 판정/실행 로직 무관)
- [2026-09-13] type=fix owner=claude
    summary: BUY_CHECK_SECONDS_OF_MINUTE(2026-09-01 도입, 분당 [3,18,33,48]초에만 신규
      진입 평가 실행) 제거 - 사용자가 실매매에서 1분봉/3분봉 매수 타이밍이 계속 틀어진다고
      지적, 분석 결과 이 게이트가 신호 발생~평가 사이에 최대 15초+폴링 간격의 추가 지연을
      만들고 있었음이 확인됨(A/B/C/D 실험 중 B=재확인 대기 축소와 별개로, 이 게이트 자체는
      백테스터(g003)에는 애초에 이식되지 않아 백테스트 성과에는 영향을 준 적이 없었음).
      _latest_buy_check_slot()/last_buy_check_slot/buy_check_due 전부 제거하고, 신규
      진입 평가를 원래대로(2026-09-01 이전과 동일하게) 매 폴링 틱(LIVE_PRICE_POLL_
      INTERVAL_SECONDS 간격 - 한 턴의 종목 순회가 끝나는 즉시 다음 턴이 시작됨)마다
      active_set 전 종목에 대해 수행하도록 원복. 매도/청산 감시 등 나머지 루프 동작은
      애초에 이 게이트의 영향을 받지 않았으므로 변경 없음.
    impact: live
    compatibility: breaking (신규 진입 평가 빈도가 분당 4회 고정에서 다시 폴링 주기와
      동일한 빈도로 증가 - 2026-09-01 변경 이전 동작으로 원복. 필요 시 r001에서
      LIVE_PRICE_POLL_INTERVAL_SECONDS를 늘려 빈도를 다시 낮출 수 있음)
- [2026-09-11] type=fix owner=claude
    summary: 종목당 처리 루프에서 보유중이지만 오늘 매수분이 아닌 포지션(NOT_TODAY_
      BUY_POSITION, 매매 없이 모니터링만 됨)의 [HOLD SKIP] 판정을 루프 최상단(can_trade_
      code_now 직후)으로 끌어올림 (사용자 요청 - 로그에서 매번 [INTRABAR_SKIP]/[CHECK]가
      찍힌 뒤에야 [HOLD SKIP]으로 continue되는 것을 보고 불필요하다고 지적). api.
      get_open_positions()는 인메모리 dict 조회라 비용이 없어 위치를 옮겨도 부작용
      없음 - HOLD SKIP 대상이면 프레임 리프레시/실시간가 조회/인트라바 게이트/[CHECK]
      로그를 전부 건너뛰고 즉시 continue. 기존에 포지션 관리 블록 안에 있던 동일 판정은
      도달 불가능한 중복 코드가 되어 제거.
    impact: live (로그 출력 순서/빈도만 변경, 매매 판정 로직 불변)
    compatibility: backward-compatible ([HOLD SKIP] 로그 자체는 그대로 남고 [INTRABAR_SKIP]/
      [CHECK]가 해당 종목에서 더 이상 찍히지 않음 - 그 두 태그를 이 케이스에서도 기대하고
      파싱하는 스크립트가 있다면 갱신 필요)
- [2026-09-09] type=fix owner=claude
    summary: check_buy_condition_1min_hybrid_trigger()에 uptrend_continuation 예외
      추가 (452190 한빛레이저 사례 - 사용자 요청). 3분봉은 11:54~11:58에 골든크로스
      확정(score 17/22)했는데 12:07~12:30 폭등 구간(4,655->5,160) 전체가 1분 트리거의
      1MIN_NO_BB_MID_GOLDEN_CROSS로 100% 리젝됨 - 돌파는 12:07~09에 발생했으나 그 후
      가격이 BB중간선 위에서 계속 강하게 올라 "크로스 시점"이 룩백창 밖으로 벗어나
      역설적으로 추세가 강하고 오래갈수록 못 통과하는 구조였음(r001 HYBRID_1MIN_TRIGGER_
      LOOKBACK_BARS 3->8 완화와 별개 대책). 두 경로로 예외 인정: (a) 함수 자체가 1분봉
      ADX/+DI/-DI/MA5/BB슬로프로 우상향 지속을 판정(_evaluate_bb_mid_cross 재사용,
      3분 게이트와 동일 공식), (b) 호출측이 3분 컨텍스트에서 이미 uptrend_continuation
      으로 판정한 신호를 context_uptrend_continuation 파라미터로 전달받아 인정. 두
      경로 모두 크로스 "발견" 여부만 대체하고 이후 캔들/BB갭/거래량 안전장치는 그대로
      전부 적용. g003의 동일 로직 복제본(check_buy_condition_1min_hybrid_trigger_sim)도
      함께 수정(g003 Update log 참조). 합성 데이터 4개 시나리오로 단위 검증 완료 -
      실제 20260909 데이터 백테스트는 원본 틱 데이터 미수집으로 보류.
    impact: live/sim
    compatibility: backward-compatible (기존 신선한 크로스 경로는 그대로 동작, 새
      예외 경로는 기존에 리젝되던 케이스만 추가로 통과시킴 - 매수 빈도 증가 예상,
      데이터 확보 후 g003 --date 20260909 --codes 452190 백테스트 재검증 권장)
- [2026-09-09] type=feat owner=claude
    summary: 라이브 로그 출력 포맷 정리 (사용자 요청). (1) [BUY REJECT] 태그를 [REJECT]로
      통일(SELL REJECT는 그대로 유지) - 매수 거부 관련 로그가 여러 지점에서 서로 다른
      부가 사유로 찍히는데 태그가 길어 가독성이 떨어졌음. (2) [INTRABAR SKIP]을
      {symbol_label} [INTRABAR_SKIP] 순서로 바꿔 [CHECK]/[REJECT]와 동일하게 종목코드_
      종목명이 태그보다 먼저 오도록 정렬(_symbol_log_label의 고정폭 패딩 덕분에 컬럼이
      맞춰짐), 태그 자체도 밑줄로 통일. (3) 주 매수조건([REJECT] 중 _buy_reject_detail로
      찍히는 케이스)에 gate_steps_diagnostic()(r002, 신규) 결과를 "STEPS ..." 로 추가 -
      실제 활성 모드인 하이브리드 경로(ENABLE_1MIN_TRIGGER_3MIN_CONTEXT)에서는 1분
      트리거 pass/fail 1개 + 3분 컨텍스트 게이트 7개(HYBRID_3MIN_CONTEXT_GATES) 전체를
      조기종료 없이 항상 계산해 P/F로 보여주고, 다른 모드(순정 9게이트 파이프라인)에서는
      BUY_GATE_CONDITIONS 9개 전체를 보여준다 - 리젝된 종목이 어느 게이트에서 막혔는지뿐
      아니라 나머지 게이트도 통과 상태인지 로그만 보고 바로 진단 가능해짐. 실제 매수
      판정 로직(buy_ok/buy_reason)에는 전혀 영향 없음 - 표시 전용.
    impact: live (로그 포맷/내용만 변경, 판정 로직 불변)
    compatibility: backward-compatible (기존 로그 파싱 스크립트가 "[BUY REJECT]"나
      "[INTRABAR SKIP]" 문자열을 그대로 grep하고 있다면 새 태그명으로 갱신 필요)
- [2026-09-07] type=feat owner=claude
    summary: ENABLE_WATCHLIST_ROTATION(r001, 기본 False) 배선 - False면
      _rebalance_active_watchlist() 호출 자체를 건너뛰어 active_set이 최초 로드된
      상위 ACTIVE_WATCHLIST_SIZE(50->20으로 함께 축소)개로 장중 내내 고정된다. 같은
      조건으로 REGULAR_START TIME_LIMIT 타이머 리셋 블록도 함께 건너뜀(로테이션이
      꺼져있으면 그 타이머는 어차피 안 읽히므로 로그 노이즈만 줄이는 목적). 배경은
      r001 Update log 2026-09-07 참조(50종목 1턴 30초 실측 - 서버 API 왕복 지연이
      지배적이라 종목당 처리시간 자체는 줄이기 어려워 active_set 축소 + 로테이션
      정지로 대응).
    impact: live
    compatibility: breaking (ENABLE_WATCHLIST_ROTATION=True로 되돌리면 기존
      active/backup 교체 동작 그대로 복원됨)
- [2026-09-07] type=fix owner=claude
    summary: _rotate_logging_for_date()의 로그 파일명 순서를 {timestamp}_{script_stem}
      -> {script_stem}_{timestamp}(buy_sell 로그는 {script_stem}_buy_sell_{timestamp})로
      변경 (사용자 요청) - 같은 날짜 폴더 안에서 파일명만 보고도 어떤 스크립트의 로그인지
      바로 구분되도록. _trade_log_target_paths()는 _LOG_CTX["trade_log"]를 동적으로 읽으므로
      별도 수정 불필요.
    impact: live
    compatibility: backward-compatible (신규 실행분부터 새 파일명 형식 적용, 기존 로그
      파일명은 그대로 유지됨 - 별도 마이그레이션 없음)
- [2026-09-07] type=fix owner=claude
    summary: check_buy_condition_1min_hybrid_trigger()의 CHASE_BUY_BB_GAP 상한을 고정값
      (HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT)에서 크로스 후 경과봉 수에 비례해 완화되는
      값으로 변경 (r001 HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR/_CEILING_PCT 참조).
      20260907 실매매 로그 분석(사용자 요청) 결과 388050 지투파워 09:16 골든크로스가
      매집봉 조건으로 리젝된 뒤 09:18~09:24 눌림으로 무효화, 09:26~09:28 재돌파는
      BB_MID(후행지표)가 못 따라와 갭이 고정 상한 0.5%를 넘어 CHASE_BUY_BB_GAP로 매
      폴링 리젝된 사례, 025980 아난티 13:45~13:50도 매집봉/거래량/갭이 서로 다른
      시점에 하나씩만 걸려 결국 룩백 만료된 사례로 확인 - 크로스가 BB_MID 위로 계속
      유지되고 있다는 것 자체가 유효한 추세 지속의 증거인데도 갭 상한이 고정이라
      시간이 지날수록 무조건 리젝 확률만 높아지는 구조였음. 크로스 인정 룩백
      (HYBRID_1MIN_TRIGGER_LOOKBACK_BARS) 안에서 실제 크로스 발생봉까지의 경과봉 수를
      계산해 상한에 반영 (신선한 크로스=경과봉 0=기존 0.5% 그대로, 경과봉이 늘수록
      봉당 0.15%p씩 완화, CEILING 1.1%로 상한). g003의 동일 로직 복제본
      (check_buy_condition_1min_hybrid_trigger_sim)도 함께 수정 - 겸사겸사 그 함수가
      2026-09-07 HYBRID_1MIN_MIN_ENTRY_VOL_MA/VOLUME 유동성 상수 분리(위 vol_ma 체크
      부근 주석 참조)를 반영하지 못하고 3분봉 기준 MIN_ENTRY_VOL_MA/VOLUME을 그대로
      쓰고 있던 기존 drift도 같이 정정함(g003 Update log 참조).
    impact: live/sim
    compatibility: backward-compatible (신선한 크로스는 기존과 동일, 경과봉이 있는
      크로스만 상한이 완화되어 매수 빈도가 소폭 늘 수 있음; g003 --date 20260904
      백테스트로 리젝 사유 재분포 확인 권장)
- [2026-09-05] type=fix owner=claude
    summary: 8/17~9/4 로그 분석(사용자 요청) 결과 도출. active_set/backup_pool 교체(2026-08-31
      도입)의 TIME_LIMIT 탈락이 ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES(60분) 순수 타이머였을 뿐,
      사용자가 알고 있던 것과 달리 실제 가격이 박스권(정체)인지는 전혀 확인하지 않고 있었음이
      확인됨. 실거래 체결 건수가 이 기능 도입 이후(8/31~9/4) 일 0~1건으로 급감(그 이전
      8/18~8/28은 일 30~200건대) - 아직 방향성을 만들어가는 중인 종목까지 60분 타이머만으로
      무차별 탈락시켜 매수 직전에 감시 대상에서 빠지는 경우가 다수였던 것으로 추정(active=50
      < watch_map=100이라 탈락한 종목은 최소 한 사이클 뒤에나 재편입됨). _rebalance_active_
      watchlist()에 frame_cache를 추가로 전달받아, 60분 경과 시점부터는 _is_box_range_hold_
      zone()(기존 매도측 박스권 판정 재사용, r002)으로 실제 정체 여부를 확인해 박스권이 확인된
      종목만 TIME_LIMIT으로 교체하고, 아직 박스권이 아닌(방향성이 남아있는) 종목은 신규
      ACTIVE_WATCHLIST_HARD_TIME_LIMIT_MINUTES(120분, r001)까지 계속 감시를 유지하도록 변경.
      120분 상한은 9/3 backup_pool 고갈 사고(Update log 9/3 참조) 재발을 막기 위한 안전장치로
      유지. 프레임 미확보 종목은 안전하게 기존과 동일하게 60분에 탈락.
    impact: live
    compatibility: backward-compatible (TIME_LIMIT 탈락 판정 기준만 변경 - GRADUATE/HARD_STOP/
      GAP_BLOCKED 및 backup_pool 재편입 동작은 동일. dropout 로그 reason이 TIME_LIMIT 외에
      TIME_LIMIT_BOX_.../TIME_LIMIT_HARD로도 나올 수 있음 - _append_watchlist_change_history
      등 reason 문자열을 직접 파싱하는 외부 도구가 있다면 startswith("TIME_LIMIT") 기준으로
      갱신 필요)
- [2026-09-04] type=fix owner=claude
    summary: 직전 커밋에서 <날짜>_buy_sell.log(flat)를 완전히 제거했는데, 이 파일은 트레이드
      로거의 실행별(run별) 타임스탬프 파일과 달리 같은 날짜 안에서 r003을 재시작해도
      계속 누적되는 일별 트레이드 로그로 쓰이고 있음이 확인됨(사용자 지적). 파일 자체는
      유지하되 저장 위치만 data_simulation/logs/(flat)에서 data_simulation/logs/<날짜>/로
      이동 - _trade_log_target_paths()가 다시 <날짜>_buy_sell.log를 후보에 포함하되 경로를
      log_date_dir 하위로 변경. <날짜>_trade_events.log(사용자가 불필요하다고 확인한 파일)는
      계속 미생성 상태로 둠.
    impact: live
    compatibility: backward-compatible (파일명은 동일하게 유지되고 저장 위치만
      data_simulation/logs/<날짜>/ 하위로 이동 - 이 파일을 flat 경로로 직접 참조하던 외부
      스크립트가 있다면 경로 갱신 필요)
- [2026-09-04] type=fix owner=claude
    summary: 직전 커밋(flat 로그 이중 기록 제거)이 놓친 잔여 중복 파일 3건 추가 수정.
      _bind_session_trade_log()가 실행 시작 시각으로 data_simulation/logs/(flat)에
      <timestamp>_r003_trade_live_execute_buy_sell.log를 별도로 다시 만들고 있었고(트레이드
      로거의 날짜 폴더본과 내용 동일), _trade_log_target_paths()도 data_simulation/logs/
      <날짜>_buy_sell.log / <날짜>_trade_events.log를 매 트레이드 로그 라인마다 flat
      폴더에 추가로 미러링하고 있었음 - 셋 다 trade_logger의 날짜 폴더 핸들러가 이미 쓰는
      내용을 그대로 복제하는 보조 안전장치였을 뿐이라 실사용 가치가 없었음(사용자가 4개
      flat 파일 중 트레이드 로거가 관리하는 날짜 폴더의 <timestamp>_..._buy_sell.log 한
      개만 남기고 나머지는 없어도 된다고 확인). _bind_session_trade_log() 삭제(호출부
      포함) 및 _trade_log_target_paths()를 _LOG_CTX["trade_log"](날짜 폴더 경로) 단일
      후보만 반환하도록 축소 - log_trade()/_log_trade_block()의 수동 폴백 루프는 이미
      로거가 관리 중인 경로라 그대로 스킵되어 사실상 무해한 안전장치로만 남음.
    impact: live
    compatibility: breaking (data_simulation/logs/(flat) 바로 아래에 buy_sell/trade_events
      관련 파일이 더 이상 생성되지 않음 - 이 경로를 직접 참조하던 외부 스크립트/모니터링이
      있다면 갱신 필요)
- [2026-09-04] type=feat owner=claude
    summary: 사용자 요청 3건 반영. (1) 현재 매수 감시 중인 종목(active_set)을 스크립트와 같은
      폴더의 active_watchlist.txt에 "종목코드,종목명" 형식으로 매 틱 기록하는
      _write_active_watchlist_txt() 신규 도입(기존 ACTIVE_WATCHLIST_STATE_PATH json과 별도로,
      grep 없이 바로 열어볼 수 있는 텍스트본 요청). (2) _rebalance_active_watchlist()의
      GRADUATE/DROPOUT/PROMOTE 이벤트를 data_simulation/logs/<날짜>/<날짜>_change_history.txt에
      타임스탬프와 함께 append하는 _append_watchlist_change_history() 신규 도입 - 감시 대상
      교체 이력을 날짜별로 추적 가능하게 함. (3) _rotate_logging_for_date()가 메인 로그/거래
      로그를 flat 폴더(data_simulation/logs/)와 날짜 폴더(data_simulation/logs/<날짜>/)에
      동일한 내용으로 이중 기록하던 것을 날짜 폴더 전용으로 변경(flat_log_filename/
      flat_trade_log_filename 제거) - 동일 파일이 두 곳에 중복 저장되던 문제 수정.
    impact: live
    compatibility: breaking (main/buy_sell 로그 파일이 더 이상 data_simulation/logs/ 바로
      아래에는 생성되지 않고 data_simulation/logs/<날짜>/ 하위에만 생성됨 - 이 경로를 직접
      참조하는 외부 스크립트/모니터링이 있다면 갱신 필요)
- [2026-09-03] type=fix owner=claude
    summary: active_set(실시간 감시 상한 50)이 TIME_LIMIT 탈락만으로 서서히 소진되어
      backup_pool까지 완전히 바닥나면(둘 다 0) 이후 장중 재보충 수단이 전혀 없어 감시
      종목이 통째로 사라지는 문제 수정. 실사례: 09/03 12:09:50 backup_pool 50개 전량
      active로 승격(backup=0), 12:49 GRADUATE 1건으로 backup 소진 경고 발생, 13:10:14
      당시 active 전원이 거의 동시에 편입돼 TIME_LIMIT 타이머도 거의 동시에 만료 ->
      대체할 backup이 하나도 없어 active_set/backup_pool 모두 0으로 떨어진 채 장 마감까지
      복구되지 않음(active_watchlist.json active_count=0, backup_count=0). watch_map/
      active_set/backup_pool이 하루 시작 시 1회만 채워지고 장중 재스캔이 없는 구조라,
      TIME_LIMIT 탈락 종목을 그냥 버리면 스캐너 100종목 풀이 시간이 지날수록 단조 감소해
      결국 고갈되는 게 구조적 필연이었음. _rebalance_active_watchlist()에서 TIME_LIMIT
      사유로 탈락한 종목만 backup_pool 맨 뒤로 재편입하도록 변경(HARD_STOP/GAP_BLOCKED는
      오늘 재진입 불가 사유이므로 기존대로 영구 제외 유지) - 신호가 없었을 뿐 나쁜 종목이
      아니므로 아직 안 써본 backup 종목들에 우선권을 준 뒤 순환 재도전하게 함.
    impact: live
    compatibility: backward-compatible (TIME_LIMIT 탈락 종목이 이후 backup_pool을 통해
      다시 active_set에 편입될 수 있음 - 기존엔 영구 제외였음. HARD_STOP/GAP_BLOCKED/
      GRADUATE 탈락 동작은 변경 없음)
- [2026-09-02] type=feat owner=claude
    summary: 현재 실시간 감시 중인 종목(active_set)/대기 중인 종목(backup_pool) 현황을
      매 틱마다 LIVE_RUNTIME_DIR/active_watchlist.json으로 기록하는 기능 추가 (사용자
      요청 - 로그 파일을 grep하지 않고도 지금 어떤 종목이 실시간 감시 대상인지 바로 확인
      가능하게 해달라는 요청). _write_active_watchlist_state() 신규 도입 -
      _rebalance_active_watchlist() 호출 직후(같은 틱의 최신 active_set/backup_pool
      반영) 실행. active는 first_active_at(편입 시각) 오름차순 정렬(다음 TIME_LIMIT
      탈락에 가까운 순서), 각 종목의 code/name/active_since/holding(현재 보유 여부)을
      포함. backup은 deque 순서(=스캐너 점수 순위) 그대로 code/name만 기록. 임시파일 작성
      후 os.replace로 교체하는 원자적 쓰기로, 파일을 동시에 읽는 도중 부분 기록이 보이는
      경우를 방지.
    impact: live (신규 파일 기록만 추가, 매매 판단 로직 변경 없음)
    compatibility: backward-compatible
- [2026-09-02] type=fix owner=claude
    summary: 계좌 보유 중이지만 오늘 watch_map(스캐너 picks)에는 없는 종목의 로그 표기가
      "000660_000660"처럼 코드가 이름 자리에 중복 표시되던 문제 수정 (000660 SK하이닉스,
      005930 삼성전자 사례 - "WARNING: account holdings not in watchlist" 로그로 존재가
      드러남). 원인: 메인 루프의 name = watch_map.get(code) or code가 watch_map에 없으면
      바로 code로 폴백했는데, 그 종목이 왜 감시 로그에 나타나는지 자체는 정상 동작 -
      position_codes(보유 포지션)는 active_set/watch_map 소속과 무관하게 항상
      iter_codes에 합쳐져 청산 감시가 끊기지 않도록 하는 기존 설계(2026-08-31 ACTIVE/BACKUP
      SPLIT)이고, 이 종목들은 NOT_TODAY_BUY_POSITION으로 매매 없이 모니터링만 됨(2026-05-24
      "당일 매수만 거래" 기능). 실제 이름 표시 문제만 수정 - TradingAPI.sync_positions_
      from_account()가 이미 조회하는 잔고 API(inquire_balance_rlz_pl) 응답에 종목명
      (prdt_name)이 포함돼 있는데 이를 버리고 있었음. 이제 보유 종목 동기화 시 prdt_name을
      전역 _SYMBOL_NAME_MAP에 등록하고, 메인 루프 name 폴백을 watch_map -> _SYMBOL_NAME_MAP
      -> code 순으로 확장. 부수 효과: _PerSymbolFileHandler(종목별 로그 파일)가 코드의
      _SYMBOL_NAME_MAP 존재 여부로 기록 대상을 판단하므로, watch_map 밖의 보유 종목도 계좌
      동기화 이후부터는 종목별 로그 파일이 정상 생성됨(이전엔 전혀 기록 안 됐음).
    impact: live (로그 표기 전용, 매매 판단 로직 변경 없음)
    compatibility: backward-compatible
- [2026-09-02] type=fix owner=claude
    summary: active_set(실시간 감시 상한 50종목, 2026-08-31 도입)의 TIME_LIMIT 탈락 타이머가
      장전 NXT 세션(08:00~08:50) 대기시간을 그대로 소모하던 문제 수정. 126730 코칩 사례
      (2026-09-02 08:03:26 active_set 편입 -> 09:03:41 TIME_LIMIT 탈락, 이후 09:07 신호
      여부와 무관하게 완전히 감시 대상에서 빠짐) 분석 결과, NXT=False 종목은 08:00~08:50
      구간 내내 can_trade_code_now()가 False라 매수 평가 자체가 불가능한데도
      ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES(60분) 카운트는 그대로 흘러, 정규장(09:00)이
      열리고 불과 3분여 만에 탈락하는 경우가 발생함(당일 로그에서 초기 편입 종목 다수가
      09:03:41에 동시 탈락). regular_session_watchlist_reset_done 플래그 신규 도입 -
      정규장 시작(REGULAR_START) 도달 첫 틱에서 그 순간의 active_set 전원의
      first_active_at을 current_dt로 1회 재설정해, TIME_LIMIT 타이머가 프리마켓 대기시간을
      제외하고 정규장 실거래 시간부터 60분을 온전히 확보하도록 함. _rebalance_active_watchlist
      호출 직전에 위치해 같은 틱에서 재설정된 시각 기준으로 탈락 판정이 이뤄짐. 날짜 변경 시
      다른 일일 상태와 함께 플래그도 리셋.
    impact: live
    compatibility: backward-compatible (정규장 시작 시점에 active_set 종목들의 TIME_LIMIT
      탈락이 최대 60분 더 늦게 발생함 - 프리마켓에 편입된 종목이 정규장에서 더 오래
      감시되나, HARD_STOP/GAP_BLOCKED/GRADUATE 탈락 사유와 backup_pool 교체 로직 자체는
      변경 없음)
- [2026-09-01] type=feat owner=claude
    summary: 신규 매수 체크(진입 평가) 시작 시각을 벽시계 기준 특정 초(BUY_CHECK_SECONDS_OF_MINUTE,
      r001, 기본값 [3, 18, 33, 48])로만 실행하도록 변경. 기존에는 메인 루프가
      LIVE_PRICE_POLL_INTERVAL_SECONDS(10초) 간격으로 돌 때마다(00/10/20/.../50초) 포지션이
      없는 모든 종목에 대해 매번 신규 진입 평가(check_buy_condition* 계열)를 수행했음.
      _latest_buy_check_slot() 신규 도입 - 이번 틱 시각 기준 가장 최근에 지난
      BUY_CHECK_SECONDS_OF_MINUTE 슬롯을 계산하고, 직전에 처리한 슬롯(last_buy_check_slot)과
      다르면(=새 슬롯이 지났으면) 이번 순회에서만 buy_check_due=True로 신규 진입 평가를
      수행, 그 외 틱에서는 종목별 else(무포지션) 분기 진입 즉시 continue로 건너뜀. 슬롯
      시각이 폴링 간격(10초)의 배수가 아니어도(예: 3초) 그 다음 폴링 틱에서 슬롯이 지난
      것으로 감지되어 실행됨(최대 폴링 간격만큼 지연 가능, _next_aligned_tick과 동일 철학).
      매도/청산 감시·시세·프레임 갱신·계좌 동기화 등 루프의 나머지 동작은 이 게이트와 무관하게
      기존과 동일하게 매 틱 실행됨 - 신규 진입 평가 실행 빈도만 변경.
    impact: live
    compatibility: breaking (신규 매수 진입 평가가 매 폴링 틱이 아닌 분당 4회(기본값)로만
      실행됨 - 진입 신호 발생 시점부터 실제 평가까지 최대 15초+폴링 간격 지연 가능;
      BUY_CHECK_SECONDS_OF_MINUTE를 폴링 간격의 배수 리스트로 설정하면 기존과 동일한 빈도로
      복원 가능. 매도/손절/트레일링 등 청산 로직은 영향 없음)
- [2026-08-28] type=feat owner=claude
    summary: 매수 미체결 재시도(추격 지정가) 신규 추가 - place_buy_order()는 매수1호가
      순수 지정가만 쓰고 체결 안 되면 [BUY STALE] 경고만 반복될 뿐 아무 조치가 없었음
      (000720 현대건설 2026-08-28 13:14 매수 신호 이후 500초 넘게 미체결로 발견).
      refresh_pending_orders의 "미체결 장기 대기" 분기를 BUY_ORDER_REPRICE_AFTER_SECONDS
      (10초, r001) 경과 시 _request_buy_reprice() 호출로 교체 - order_rvsecncl(정정취소
      취소구분)로 기존 주문을 취소하고, pending에 reprice_pending/cancel_inflight 상태를
      건다. "BUY closed without fill" 분기(취소 확인 지점)도 cancel_inflight면
      PENDING_BUY_GRACE_SECONDS(90초, 브로커 예상 밖 거부용 유예라 우리가 건 취소엔
      불필요) 기다리지 않고 즉시 reprice_pending 여부로 분기 - True면
      _resubmit_repriced_buy_order()로 재주문(1차 신선한 매수1호가/2차 매도1호가/최종
      시장가, place_buy_order의 has_buy_exposure/쿨다운 검사를 우회하는 전용 경로 -
      그 검사들은 "새 진입"용이지 "같은 진입 시도 재주문"에는 안 맞음), False면(추격
      상한 초과로 포기) 그냥 정리. place_buy_order()에 order_org_no(krx_fwdg_ord_orgno,
      취소/정정에 필수)·entry_reference_price(재주문해도 불변, 추격 상한 계산용)·
      reprice_attempt/cancel_inflight/reprice_pending 초기값을 pending_orders에 추가.
      상세 배경은 r001 Update log 2026-08-28 참조.
    impact: live
    compatibility: backward-compatible (미체결 상태 처리 경로만 변경, 체결 확인/포지션
      로직은 그대로; order_rvsecncl은 이 저장소에서 최초로 실사용 - 프로덕션 실행 전
      모의투자(env_dv=demo)로 취소/재주문 왕복이 실제로 되는지 먼저 확인 권장)
- [2026-08-28] type=feat owner=claude
    summary: ENABLE_1MIN_TRIGGER_3MIN_CONTEXT(r001) 배선 추가 - 신규 elif 분기로
      ENABLE_1MIN_GOLDEN_CROSS_BUY(1분봉 완전대체)와 최종 else(기존 3분봉 단독) 사이에
      끼워넣음. 1분봉 프레임을 별도로 조회해 check_buy_condition_1min으로 트리거를
      먼저 확인하고(buy_frame/3분봉은 그대로 유지, ENABLE_1MIN_GOLDEN_CROSS_BUY처럼
      갈아끼우지 않음), 통과 시에만 r002 run_3min_context_pipeline으로 3분봉 컨텍스트를
      재확인한다. ENABLE_1MIN_ENTRY_SCORE_GATE(12-b) 조건에 not
      ENABLE_1MIN_TRIGGER_3MIN_CONTEXT 추가 - 하이브리드 트리거와 목적이 겹치는 이중
      1분봉 게이트 방지. 배경은 r001/r002 Update log 2026-08-28 참조(403870 HPSP
      09:09/09:27 반복 CHASE_BUY_BB_GAP 반려 분석).
    impact: live
    compatibility: backward-compatible (ENABLE_1MIN_TRIGGER_3MIN_CONTEXT 기본값 False라
      미사용 시 동작 변화 없음)
- [2026-08-27] type=fix owner=copilot
    summary: 시그널 매도 억제 판정(_strong_uptrend)에 가격 기반 보조 조건 추가 - 기존 ADX>28 &
      +DI>-DI 단일 조건 외에, MA5가 상승 중이고 현재가가 BB_MID 위에 있으면(_price_uptrend)도
      상승추세로 인정해 STOCH_K_LT_D/SIGNAL_EXIT_MACD_HIST_DOWN_2BARS를 억제한다. 000500
      가온전선 2026-08-27 09:06 매매 - MACD_HIST가 2봉 연속 하락(146.9->-247.8->-441.7)해
      -0.81%에 청산됐으나 ADX가 28을 못 넘겨 억제되지 않았고, 매도 직후 한 봉만 눌린 뒤 상승이
      재개됨. ADX는 후행지표라 추세 초입 눌림목 구간에서 못 잡는 사각지대가 있어 보완.
    impact: live
    compatibility: backward-compatible (기존 ADX 조건은 그대로 유지, OR로 조건 추가 -
      _strong_uptrend가 더 쉽게 True가 되어 신호 매도가 이전보다 더 자주 억제될 수 있음)
- [2026-08-24] type=feat owner=copilot
    summary: PEAK_NEXT_BAR_BEARISH_EXIT 신규 매도 규칙 추가 - 포지션의 고점(highest_price)이
      갱신될 때마다 그 틱이 속한 확정봉을 pos["peak_bar_time"]으로 기록해두고, 그 바로 다음
      확정봉이 음봉이면서 고점 대비 PEAK_NEXT_BAR_DROP_PCT(0.5%, r003) 이상 하락하면
      TP_EXTENSION_TRAIL(고점 대비 -1.0%) 도달을 기다리지 않고 즉시 시장가 매도한다.
      017670 SK텔레콤 2026-08-24 12:40 매매 분석 계기: peak pnl 1.26%(12:58) -> 고점 바로
      다음 3분봉(13:00봉, 종가 104,300)이 이미 하락 시작했는데도 giveback이 1.0%에 못 미쳐
      13:08:24까지 안 팔리고 실현 0.19%(거래세 포함 시 순손실)로 끝남. ATR_TP 도달 이후
      (peak_pnl_pct>=atr_tp_pct)에만 작동 - 매수 직후 초반은 POST_BUY_BB_DROP_GUARD/
      ATR_STOP_LOSS가 별도로 담당하므로 겹치지 않음. pos["peak_bar_time"]/
      ["peak_next_bar_checked"]는 positions_meta 직렬화 화이트리스트에 없어 재시작 시
      리셋됨(다른 in-memory 전용 상태들과 동일한 수준).
    impact: live
    compatibility: backward-compatible (ENABLE_PEAK_NEXT_BAR_BEARISH_EXIT=False로 즉시
      롤백 가능; 기존 TP_EXTENSION_TRAIL/신호 매도 등 다른 매도 경로는 그대로 유지)
- [2026-08-24] type=refactor owner=copilot
    summary: _buy_condition_snapshot()(BUY REJECT 로그의 GATES 진단 문구)이 close_cross를
      즉시 2봉 비교만으로 자체 계산하고 있어, 실제 판정 함수(run_buy_condition_pipeline_comment,
      r005)의 5봉 룩백/우상향추세지속 로직으로는 통과했을 케이스도 로그엔 close_cross=False로
      찍히던 문제 발견(051900 LG생활건강 09:12 사례로 사용자가 지적). r005에 새로 추출한
      _evaluate_bb_mid_cross()를 import해 동일 로직을 재사용하도록 교체 - GATES 라인에
      cross_pass(최종 통과 여부)/uptrend_cont 필드 신규 추가. 로그 전용 변경이라 실매매
      판정에는 영향 없음(frame/live_price가 없는 예외 상황에서만 기존 단순 2봉 fallback 유지).
    impact: live (로그만, 판정 로직 아님)
    compatibility: backward-compatible (로그 필드 추가만, 기존 필드 의미도 그대로 유지)
- [2026-08-24] type=fix owner=copilot
    summary: check_buy_condition_1min(비활성 대체경로, ENABLE_1MIN_GOLDEN_CROSS_BUY=False)의
      최소 봉 수 요건을 BB_PERIOD(20)봉 -> 2봉으로 되돌림. 사용자가 MTS에서 051900 LG생활건강
      1분봉/3분봉 볼린저밴드가 09:00 장 시작 즉시 정상 렌더링되고 골든크로스도 유효함을 실측 확인 -
      2026-08-17 당시 전제(HTS 미표시=BB_MIDDLE이 20봉 미만이면 통계적으로 불안정)가 HTS 자체
      렌더링 문제였을 가능성이 높음. r005 Update log 2026-08-24 참조(실제 매매를 결정하는
      run_buy_condition_pipeline_comment에도 동일 revert 적용).
    impact: live
    compatibility: breaking (경로 자체는 여전히 비활성 - 향후 재활성화 시 개장 직후 20분 차단이
      없어짐)
- [2026-08-21] type=fix owner=copilot
    summary: place_sell_order()의 has_pending_order() 차단 조건이 매도(sell) 대기 주문뿐 아니라
      매수(buy) 대기 주문(특히 피라미딩 추가매수)에도 걸려, 피라미딩 매수 주문이 미체결로 남아있는
      동안 익절(TP1/TP2/트레일링)·손절(ATR_STOP/HARD_STOP) 매도가 전부 조용히 실패(return False)하고
      다음 폴링에서 [SELL TRIGGER] 로그만 재출력되던 문제 수정. logs/20260817~20260821 실매매 로그
      분석 결과 003010(혜인) 사례에서 13:28:09 피라미딩 매수 주문(order_no=0017559900)이 대기 중인
      3분14초 동안 TP1(목표 도달가=1.46%, 당시가=8.84%) 매도가 8회 연속 실패하다가 주문이 풀린
      13:31:23에야 체결되었고, 그 사이 가격이 고점(12,500) 대비 하락해 이어진 TP_EXTENSION_TRAIL
      청산까지 포함하면 peak_pnl 8.84% 중 8.53%p를 반납(giveback)함. 같은 유형(피라미딩 대기 중
      익절/손절 지연)의 TP_EXTENSION_TRAIL/TRAILING_STOP_GIVEBACK 사례가 5일간 30건, 총 66.80%p
      giveback(건당 평균 2.23%p)으로 확인됨. pos["quantity"]는 브로커 확정 체결분만 반영하므로
      (피라미딩 확정 시 sync_positions_from_account(force=True)로 별도 반영) 대기 중인 매수 주문과
      무관하게 이미 보유 중인 확정 수량은 즉시 매도 가능 - place_sell_order()의 가드를
      "대기 주문이 있으면 무조건 차단"에서 "대기 주문이 매도(side=sell)일 때만 차단(중복 매도 제출
      방지)"으로 변경. 대기 중인 피라미딩 매수가 매도 이후 뒤늦게 체결되는 경우는 sync_positions_
      from_account가 브로커의 실제 잔고를 그대로 반영해 신규 포지션으로 자연 처리되며(다른 포지션과
      동일하게 손절/익절 로직의 보호를 받음), r007 시뮬레이터는 대기 주문 개념이 없어 해당 없음.
    impact: live
    compatibility: breaking (피라미딩 대기 중에도 익절/손절 매도가 즉시 실행됨 - 매도 타이밍이
      빨라지고 수익 반납/손절 지연이 감소하나, 드물게 피라미딩 매수가 매도 이후 체결되어 의도치
      않은 소량 재진입 포지션이 생길 수 있음(기존 손절/익절 로직으로 계속 보호됨))

Note: This script cannot guarantee profit. Always paper-test before live trading.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import json
import os
import signal
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from r001_define_config import (
    ACCOUNT_SYNC_INTERVAL_SECONDS,
    AFTERNOON_NXT_END,
    AFTERNOON_NXT_FORCE_EXIT,
    AFTERNOON_NXT_NEW_ENTRY_CUTOFF,
    AFTERNOON_NXT_START,
    AUX_SELL_MIN_PNL_SCORE2,
    AUX_SELL_MIN_PNL_SCORE3,
    AUX_SELL_MIN_PNL_SCORE4,
    BOX_RANGE_HOLD_LOOKBACK_BARS,
    BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT,
    BOX_RANGE_HOLD_MAX_RANGE_PCT,
    DATA_DIR_NAME,
    DEFINE_TODAY_CODE_PATH,
    ENABLE_BOX_RANGE_HOLD_TECH_SELL,
    ENABLE_NXT_SESSION,
    LIVE_PRICE_BB_BUFFER_PCT,
    LIVE_PRICE_CROSS_CONFIRM_POLLS,
    LIVE_PRICE_CROSS_CONFIRM_SECONDS,
    LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS,
    LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS,
    MA5_BB_DOWN_CROSS_IMMEDIATE_PNL,
    MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE,
    MA5_BB_DOWN_CROSS_MIN_PNL,
    MAX_ORDER_AMOUNT_KRW,
    ENABLE_BUY_SPLIT_MARKET_LIMIT,
    BUY_SPLIT_MARKET_RATIO,
    MORNING_NXT_END,
    MORNING_NXT_START,
    ATR_STOP_MULTIPLIER,
    ATR_TAKE_PROFIT_MULTIPLIER,
    REGULAR_END,
    REGULAR_FORCE_EXIT,
    REGULAR_NEW_ENTRY_CUTOFF,
    REGULAR_START,
    STARTUP_WARMUP_SECONDS,
    STOP_LOSS_PERCENT,
    STOCH_OVERBOUGHT,
    STAGED_TP1_PCT,
    TRADE_COOLDOWN_MINUTES,
    TRAILING_STOP_FROM_PEAK,
    AUX_SELL_MIN_REALIZED_TARGET_PCT,
    AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT,
    BB_BUY_SCORE_THRESHOLD,
    BUY_CONSECUTIVE_CONFIRM_COUNT,
    ENABLE_INTRABAR_LIVE_ENTRY_FILTER,
    ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP,
    FRAME_BACKFILL_SYNC_SECONDS,
    FRAME_POLL_INTERVAL_SECONDS,
    INTRABAR_ADX_MIN,
    INTRABAR_MFI_MAX,
    INTRABAR_MFI_MIN,
    INTRABAR_MIN_ELAPSED_SECONDS,
    INTRABAR_RSI_MAX,
    INTRABAR_RSI_MIN,
    INTRABAR_RSI_EXT_MAX,
    LIVE_PRICE_BACKOFF_BASE_SECONDS,
    LIVE_PRICE_BACKOFF_MAX_SECONDS,
    LIVE_PRICE_POLL_INTERVAL_SECONDS,
    LIVE_PRICE_STALE_TTL_SECONDS,
    LIVE_STATE_SAVE_INTERVAL_SECONDS,
    MAIN_LOOP_MAX_CONSECUTIVE_ERRORS,
    MAIN_LOOP_MIN_CYCLE_SECONDS,
    MARKET_DAY_FAIL_CLOSED,
    MORNING_NXT_NEW_ENTRY_CUTOFF,
    ORDER_STATUS_POLL_INTERVAL_SECONDS,
    PENDING_BUY_GRACE_SECONDS,
    BUY_ORDER_STALE_WARN_SECONDS,
    BUY_ORDER_REPRICE_AFTER_SECONDS,
    BUY_ORDER_REPRICE_MAX_ATTEMPTS,
    BUY_ORDER_REPRICE_MAX_CHASE_PCT,
    PENDING_STATUS_BACKOFF_MAX_SECONDS,
    SESSION_FORCE_CLOSE_ALL_AT_CUTOFF,
    WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS,
    ACTIVE_WATCHLIST_SIZE,
    ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES,
    ACTIVE_WATCHLIST_HARD_TIME_LIMIT_MINUTES,
    ENABLE_WATCHLIST_ROTATION,
    CANDLE_CONFIRM_DELAY_SECONDS,
)
from r002_strategy_core_shared import (
    R76StrategyConfig,
    calculate_indicators,
    check_sell_condition as shared_check_sell_condition,
    update_live_price_cross_state as shared_update_live_price_cross_state,
    _compute_bb_slope_pct,
    _evaluate_bb_mid_cross,
    _is_box_range_hold_zone,
    _buy_support_score as _shared_buy_support_score,
    BUY_SCORE_RULES,
)

from r005_buy_conditions import BuyContext, BuyServices, BuyState, RiskState, check_buy_conditions
from r006_sell_conditions import SellContext, SellServices, SellState, evaluate_sell_conditions

BUY_SCORE_MAX = sum(rule.max_score for rule in BUY_SCORE_RULES)

current_dir = Path(__file__).resolve().parent
data_sim_dir = current_dir.parent / "data_simulation"
sys.path.insert(0, str(data_sim_dir))  # g005_watchlist_bridge
from g005_watchlist_bridge import resolve_watchlist_path

project_root = Path(os.environ.get("OPEN_TRADING_API_ROOT", str(Path.home() / "git" / "open-trading-api")))
sys.path.insert(0, str(project_root / "examples_llm"))
sys.path.insert(0, str(project_root / "examples_user" / "domestic_stock"))
sys.path.insert(0, str(project_root / "examples_llm" / "domestic_stock" / "inquire_time_itemchartprice"))
sys.path.insert(0, str(current_dir))

import kis_auth as ka
import domestic_stock_functions as dsf

try:
    from inquire_time_itemchartprice import inquire_time_itemchartprice
except Exception:
    inquire_time_itemchartprice = None

TODAY_CODE_FILE = current_dir / DEFINE_TODAY_CODE_PATH
DATA_DIR = data_sim_dir / DATA_DIR_NAME  # data/ lives under data_simulation/
TODAY_BUYS_FILENAME = "today_buys.txt"

LIVE_RUNTIME_DIR = DATA_DIR / "live_runtime"
LIVE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
POSITION_META_PATH = LIVE_RUNTIME_DIR / "position_meta.json"
ACTIVE_WATCHLIST_STATE_PATH = LIVE_RUNTIME_DIR / "active_watchlist.json"
ACTIVE_WATCHLIST_TXT_PATH = current_dir / "active_watchlist.txt"
# Legacy alias (daily traded_today + optional combined state)
LIVE_STATE_DIR = LIVE_RUNTIME_DIR

KIS_ENV_DV = (os.environ.get("KIS_ENV_DV", "real") or "real").strip()
_LIVE_DRY_RUN_RAW = (os.environ.get("LIVE_DRY_RUN", "") or "").strip().lower()
LIVE_DRY_RUN = _LIVE_DRY_RUN_RAW in ("1", "true", "yes", "on", "y")

ENABLE_LIVE_DRY_RUN = LIVE_DRY_RUN

SHARED_R76_CONFIG = R76StrategyConfig(
    live_price_bb_buffer_pct=LIVE_PRICE_BB_BUFFER_PCT,
    live_price_cross_confirm_polls=LIVE_PRICE_CROSS_CONFIRM_POLLS,
    live_price_cross_confirm_seconds=LIVE_PRICE_CROSS_CONFIRM_SECONDS,
    live_price_down_cross_confirm_polls=LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS,
    live_price_down_cross_confirm_seconds=LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS,
    stoch_overbought=STOCH_OVERBOUGHT,
    stop_loss_percent=STOP_LOSS_PERCENT,
    # [2026-09-07] config.take_profit_percent는 check_sell_condition의 박스권 홀드
    # 상한으로만 쓰인다(실제 익절은 ENABLE_STAGED_TAKE_PROFIT 경로가 별도 담당).
    # 레거시 TAKE_PROFIT_PERCENT(2.5%)를 그대로 쓰면 실제 1차 익절 목표인
    # STAGED_TP1_PCT(3.0%)와 어긋나 2.5~3.0% 구간에서 박스권 홀드 보호가 빠지는
    # 공백이 있었다 - 실제 TP1 기준과 일치시킨다.
    take_profit_percent=STAGED_TP1_PCT,
    enable_box_range_hold_tech_sell=ENABLE_BOX_RANGE_HOLD_TECH_SELL,
    box_range_hold_lookback_bars=BOX_RANGE_HOLD_LOOKBACK_BARS,
    box_range_hold_max_range_pct=BOX_RANGE_HOLD_MAX_RANGE_PCT,
    box_range_hold_max_bb_width_pct=BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT,
    ma5_bb_down_cross_min_pnl=MA5_BB_DOWN_CROSS_MIN_PNL,
    ma5_bb_down_cross_immediate_pnl=MA5_BB_DOWN_CROSS_IMMEDIATE_PNL,
    ma5_bb_down_cross_immediate_score=MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE,
    aux_sell_min_pnl_score2=AUX_SELL_MIN_PNL_SCORE2,
    aux_sell_min_pnl_score3=AUX_SELL_MIN_PNL_SCORE3,
    aux_sell_min_pnl_score4=AUX_SELL_MIN_PNL_SCORE4,
    bb_buy_score_threshold=BB_BUY_SCORE_THRESHOLD,
)

INDICATOR_WARMUP_BARS = 1  # backtest uses MIN_BARS_REQUIRED=3; live needs only 1 closed bar (indicators use min_periods=1)

# Market-day status cache (holiday API call reduction).
_MARKET_DAY_STATUS_CACHE: dict[str, tuple[bool, str]] = {}

# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
class _SuppressLibLogs(logging.Filter):
    _SUPPRESS = frozenset(["Data fetch complete.", "Call Next page...", "Max recursive depth reached."])
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in self._SUPPRESS

_suppress_filter = _SuppressLibLogs()

class _SuppressConsoleCheckLogs(logging.Filter):
    """[CHECK] 로그는 파일에는 남기되 콘솔(stdout)에는 찍지 않는다 (사용자 요청 -
    종목 순회마다 찍혀 콘솔이 CHECK 줄로만 채워지는 문제). 이 필터는 콘솔용
    StreamHandler에만 붙이고 FileHandler에는 붙이지 않는다."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "[CHECK" not in record.getMessage()

_console_check_filter = _SuppressConsoleCheckLogs()

class _ConsoleTruncateRejectFormatter(logging.Formatter):
    """[REJECT] 로그는 콘솔에는 reason(및 그에 딸린 짧은 수치, _buy_reject_detail
    참조)까지만 보여주고, 그 뒤의 GATES/STEPS 진단 블록은 생략한다 (사용자 요청 -
    콘솔이 REJECT 상세로 도배되는 문제). 이 포매터는 콘솔용 StreamHandler에만 붙이고
    FileHandler/_PerSymbolFileHandler는 별도 Formatter를 그대로 쓰므로 파일에는
    _buy_reject_detail이 반환한 전체 내용이 그대로 남는다."""
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if "[REJECT" in formatted:
            cut = formatted.find(" | GATES")
            if cut != -1:
                return formatted[:cut]
        return formatted

_LOG_CTX: dict[str, object] = {"date_str": datetime.now().strftime("%Y%m%d")}

import threading

_TRADE_LOG_WRITE_LOCK = threading.Lock()

_SYMBOL_CODE_PATTERN = re.compile(r"\b(\d{6})\b")
_INVALID_FILENAME_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
_SYMBOL_NAME_MAP: dict[str, str] = {}


def _sanitize_log_filename(name: str) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(name).strip())
    cleaned = cleaned.rstrip(" .")
    return cleaned or "UNKNOWN"


def register_symbol_names(symbol_name_map: dict[str, str]) -> None:
    _SYMBOL_NAME_MAP.clear()
    for code, name in symbol_name_map.items():
        normalized_code = str(code).zfill(6)
        normalized_name = str(name).strip() if name else ""
        _SYMBOL_NAME_MAP[normalized_code] = normalized_name


class _PerSymbolFileHandler(logging.Handler):
    """Write log lines to per-symbol files based on 6-digit code in message."""

    def __init__(self, base_dir: Path, buy_sell: bool = False):
        super().__init__(level=logging.INFO)
        self.base_dir = base_dir
        self.buy_sell = buy_sell
        self._streams: dict[Path, object] = {}

    def _resolve_path(self, code: str) -> Path:
        symbol_name = _SYMBOL_NAME_MAP.get(code, "")
        date_str = str(_LOG_CTX.get("date_str") or datetime.now().strftime("%Y%m%d"))
        label = f"{date_str}_{code}_{symbol_name}" if symbol_name else f"{date_str}_{code}"
        stem = _sanitize_log_filename(label)
        suffix = "_buy_sell.txt" if self.buy_sell else ".txt"
        return self.base_dir / f"{stem}{suffix}"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            match = _SYMBOL_CODE_PATTERN.search(msg)
            if not match:
                return
            code = match.group(1)
            if code not in _SYMBOL_NAME_MAP:
                return
            path = self._resolve_path(code)
            stream = self._streams.get(path)
            if stream is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = open(path, "a", encoding="utf-8")
                self._streams[path] = stream
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception:
                pass
        self._streams.clear()
        super().close()

def _rotate_logging_for_date(date_str: str) -> None:
    _LOG_CTX["date_str"] = date_str
    log_dir = data_sim_dir / "logs"
    log_date_dir = log_dir / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    log_date_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_stem = Path(__file__).stem
    log_filename = log_date_dir / f"{script_stem}_{timestamp}.log"
    trade_log_filename = log_date_dir / f"{script_stem}_{timestamp}_buy_sell.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    global logger, trade_logger, _trade_handler, _symbol_general_handler, _symbol_trade_handler
    logger = logging.getLogger(__name__)
    logging.getLogger("domestic_stock_functions").setLevel(logging.WARNING)
    logging.getLogger("inquire_time_itemchartprice").setLevel(logging.WARNING)
    for _handler in logging.getLogger().handlers:
        if not any(isinstance(f, _SuppressLibLogs) for f in getattr(_handler, "filters", [])):
            _handler.addFilter(_suppress_filter)
        # type()을 정확히 비교 - FileHandler는 StreamHandler의 서브클래스라
        # isinstance(h, StreamHandler)로는 콘솔 핸들러만 골라낼 수 없다.
        if type(_handler) is logging.StreamHandler:
            if not any(isinstance(f, _SuppressConsoleCheckLogs) for f in getattr(_handler, "filters", [])):
                _handler.addFilter(_console_check_filter)
            # FileHandler는 basicConfig가 준 공용 Formatter를 그대로 쓰고(전체 내용
            # 보존), 콘솔 StreamHandler만 별도 인스턴스로 덮어써 REJECT를 잘라 보여준다.
            _handler.setFormatter(_ConsoleTruncateRejectFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    trade_logger = logging.getLogger("trade_events")
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False
    for h in list(trade_logger.handlers):
        trade_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    _trade_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _trade_handler = logging.FileHandler(trade_log_filename, encoding="utf-8")
    _trade_handler.setFormatter(_trade_formatter)
    trade_logger.addHandler(_trade_handler)

    for h in list(logger.handlers):
        if isinstance(h, _PerSymbolFileHandler):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    _symbol_general_handler = _PerSymbolFileHandler(log_date_dir, buy_sell=False)
    _symbol_general_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_symbol_general_handler)

    for h in list(trade_logger.handlers):
        if isinstance(h, _PerSymbolFileHandler):
            trade_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    _symbol_trade_handler = _PerSymbolFileHandler(log_date_dir, buy_sell=True)
    _symbol_trade_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    trade_logger.addHandler(_symbol_trade_handler)

    for handler in trade_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    _LOG_CTX["trade_log"] = trade_log_filename

_rotate_logging_for_date(str(_LOG_CTX["date_str"]))

def log(msg: str) -> None:
    logger.info(msg)





def _trade_log_target_paths() -> list[Path]:
    date_str = str(_LOG_CTX.get("date_str") or datetime.now().strftime("%Y%m%d"))
    candidates: list[Path] = [
        data_sim_dir / "logs" / date_str / f"{date_str}_buy_sell.log",
    ]
    value = _LOG_CTX.get("trade_log")
    if value:
        candidates.append(Path(value))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _trade_logger_file_paths() -> set[str]:
    paths: set[str] = set()
    for handler in getattr(trade_logger, "handlers", []):
        file_name = getattr(handler, "baseFilename", None)
        if not file_name:
            continue
        try:
            paths.add(str(Path(file_name).resolve()))
        except Exception:
            paths.add(str(file_name))
    return paths


def log_trade(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [INFO] {msg}\n"
    logger_managed_paths = _trade_logger_file_paths()
    with _TRADE_LOG_WRITE_LOCK:
        for target_path in _trade_log_target_paths():
            try:
                target_key = str(target_path.resolve())
                if target_key in logger_managed_paths:
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "a", encoding="utf-8") as trade_f:
                    trade_f.write(line)
                    trade_f.flush()
                    os.fsync(trade_f.fileno())
            except Exception as exc:
                logger.warning(f"trade log append failed ({target_path}): {exc}")
    trade_logger.info(msg)
    for handler in trade_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    log(f"[TRADE   ] {msg}")


def _log_trade_block(lines: list[str], event_time: datetime | None = None, mirror_main_log: bool = False) -> None:
    ts = event_time or datetime.now()
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    payload = [f"{stamp} [INFO] {line}\n" for line in lines]

    with _TRADE_LOG_WRITE_LOCK:
        for target_path in _trade_log_target_paths():
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "a", encoding="utf-8") as trade_f:
                    trade_f.writelines(payload)
                    trade_f.flush()
                    os.fsync(trade_f.fileno())
            except Exception as exc:
                logger.warning(f"trade log append failed ({target_path}): {exc}")

    if mirror_main_log:
        for line in lines:
            log(line)


# ---------------------------------------------------------------------------
# 파일 유틸
# ---------------------------------------------------------------------------

def _load_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=encoding) as file_obj:
                return [line.strip() for line in file_obj if line.strip()]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("define_today_code", b"", 0, 1, "Unable to decode")


def load_today_codes(code_file: Path | None = None) -> dict[str, str]:
    source_file = code_file or TODAY_CODE_FILE

    if not source_file.exists():
        return {}

    result: dict[str, str] = {}
    for line in _load_text_lines(source_file):
        if line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        # Backward compatible formats:
        # - code,name
        # - YYYYMMDD,code,name
        if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 8:
            code = parts[1].zfill(6)
            name = parts[2] if parts[2] else code
        else:
            code = parts[0].zfill(6)
            name = parts[1] if len(parts) >= 2 and parts[1] else code
        result[code] = name

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R76 real trading runner")
    parser.add_argument("--date", type=str, help="Watchlist date (YYYYMMDD). Use data/YYYYMMDD/picks.txt first")
    parser.add_argument(
        "--watchlist-source",
        type=str,
        default="auto",
        choices=["auto", "r004", "scan-picks", "picks"],
        help="Watchlist resolution: auto, r004, scan-picks, or legacy picks.txt",
    )
    parser.add_argument("--dry-run", "--fake", dest="dry_run", action="store_true", help="Log orders without sending to broker")
    parser.add_argument("--env-dv", type=str, default=None, help="KIS env_dv override (default: env KIS_ENV_DV or real)")
    return parser.parse_args()


def _resolve_watchlist_file(target_date: str | None, watchlist_source: str = "auto") -> Path:
    return resolve_watchlist_path(current_dir, target_date, watchlist_source, DATA_DIR)


def _today_buys_file_path(date_str: str) -> Path:
    return DATA_DIR / date_str / TODAY_BUYS_FILENAME


def save_today_buy_codes(date_str: str, codes: set[str]) -> None:
    path = _today_buys_file_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(sorted({str(code).zfill(6) for code in (codes or set())}))
    if payload:
        payload += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _is_today_buy_position(code: str, pos: dict, date_str: str, today_buy_codes: set[str]) -> bool:
    norm = str(code).zfill(6)
    if norm in today_buy_codes:
        return True

    buy_time = pos.get("buy_time")
    if isinstance(buy_time, datetime):
        return buy_time.strftime("%Y%m%d") == date_str

    if isinstance(buy_time, str) and buy_time.strip():
        try:
            return datetime.fromisoformat(buy_time.strip()).strftime("%Y%m%d") == date_str
        except ValueError:
            return False

    return False


# ---------------------------------------------------------------------------
# 주문 결과 파싱
# ---------------------------------------------------------------------------

def _order_succeeded(result) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        rt_cd = result.get("rt_cd")
        if rt_cd is not None:
            return str(rt_cd).strip() == "0"
        # Some wrappers return output-only dict without rt_cd on success.
        return bool(result)
    try:
        if hasattr(result, "empty") and bool(result.empty):
            return False
        if hasattr(result, "columns"):
            lowered = {str(col).lower() for col in result.columns}
            if "rt_cd" in lowered:
                col_name = next(col for col in result.columns if str(col).lower() == "rt_cd")
                return str(result.iloc[0][col_name]).strip() == "0"
            # domestic_stock_functions.order_cash() returns a non-empty DataFrame
            # containing output fields (for example odno/ord_tmd) when successful,
            # and an empty DataFrame when failed.
            return len(result) > 0
    except Exception:
        pass
    return False


def _extract_order_value(result, candidates: tuple[str, ...]):
    if result is None:
        return None

    def _pick(mapping):
        if not isinstance(mapping, dict):
            return None

        lowered = {str(key).lower(): value for key, value in mapping.items()}
        for key in candidates:
            value = lowered.get(str(key).lower())
            if value not in (None, ""):
                return value

        for nested in ("output", "output1", "output2", "data"):
            inner = mapping.get(nested)
            if isinstance(inner, dict):
                picked = _pick(inner)
                if picked is not None:
                    return picked
            elif isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        picked = _pick(item)
                        if picked is not None:
                            return picked
        return None

    if isinstance(result, dict):
        return _pick(result)

    try:
        if hasattr(result, "to_dict"):
            records = result.to_dict(orient="records")
            if records:
                return _pick(records[0])
    except Exception:
        pass

    return None


def _extract_order_price(result):
    value = _extract_order_value(
        result,
        ("avg_pric", "avg_price", "avg_prvs", "stck_prpr", "ord_unpr", "fill_pric", "ccld_pric", "prpr"),
    )
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _extract_order_number(result) -> str:
    value = _extract_order_value(result, ("odno", "ord_no", "order_no"))
    text = str(value).strip() if value is not None else ""
    return text


def _extract_order_org_no(result) -> str:
    """정정취소주문(order_rvsecncl)의 필수 파라미터 krx_fwdg_ord_orgno(한국거래소전송주문
    조직번호) 추출. order_cash 성공 응답의 output에 KRX_FWDG_ORD_ORGNO로 포함됨."""
    value = _extract_order_value(result, ("krx_fwdg_ord_orgno", "ord_orgno"))
    text = str(value).strip() if value is not None else ""
    return text


def _extract_aux_score_from_reason(reason: str) -> int | None:
    m = re.search(r"AUX_REVERSAL_SCORE_(\d+)", str(reason or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _aux_min_pnl_for_score(score: int) -> float | None:
    if score >= 4:
        return float(AUX_SELL_MIN_PNL_SCORE4)
    if score == 3:
        return float(AUX_SELL_MIN_PNL_SCORE3)
    if score == 2:
        return float(AUX_SELL_MIN_PNL_SCORE2)
    return None


def _extract_order_time(result) -> str:
    value = _extract_order_value(result, ("ord_tmd", "ord_time", "order_time"))
    text = str(value).strip() if value is not None else ""
    return text


def _format_pending_pnl(reason_prefix: str, pnl_pct: float) -> str:
    return f"{reason_prefix}_{pnl_pct * 100:.2f}%"


def _session_exit_plan(reason_prefix: str, pnl_pct: float) -> tuple[str, str]:
    if SESSION_FORCE_CLOSE_ALL_AT_CUTOFF:
        return "sell", _format_pending_pnl(f"{reason_prefix}_SESSION_FORCE_CLOSE_ALL", pnl_pct)
    if pnl_pct > 0:
        return "sell", _format_pending_pnl(f"{reason_prefix}_PROFIT_CLOSE", pnl_pct)
    if ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP and pnl_pct >= STOP_LOSS_PERCENT:
        return "hold", _format_pending_pnl(f"{reason_prefix}_OVERNIGHT_HOLD_WITHIN_STOP", pnl_pct)
    return "sell", _format_pending_pnl(f"{reason_prefix}_STOP_LOSS_BREACH", pnl_pct)


def _extract_order_error_detail(result) -> str:
    """Best-effort extraction of broker error code/message from order response."""
    if result is None:
        return "NO_RESULT"

    key_candidates = ("rt_cd", "msg_cd", "msg1", "msg", "error_code", "error_message")

    def _pull(mapping) -> dict[str, str]:
        found: dict[str, str] = {}
        if not isinstance(mapping, dict):
            return found

        for key in key_candidates:
            value = mapping.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                found[key] = text

        for nested in ("output", "output1", "output2", "data"):
            inner = mapping.get(nested)
            if isinstance(inner, dict):
                found.update({k: v for k, v in _pull(inner).items() if k not in found})
            elif isinstance(inner, list):
                for item in inner:
                    if isinstance(item, dict):
                        found.update({k: v for k, v in _pull(item).items() if k not in found})

        return found

    extracted: dict[str, str] = {}
    if isinstance(result, dict):
        extracted = _pull(result)
    else:
        try:
            if hasattr(result, "to_dict"):
                records = result.to_dict(orient="records")
                if records:
                    extracted = _pull(records[0])
        except Exception:
            extracted = {}

    if not extracted:
        return "UNKNOWN_ORDER_ERROR"

    parts = []
    for key in ("rt_cd", "msg_cd", "msg1", "msg", "error_code", "error_message"):
        value = extracted.get(key)
        if value:
            parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else "UNKNOWN_ORDER_ERROR"


def _format_code_label(code: str, code_name: str = "") -> str:
    return f"{code}({code_name})" if code_name else code


def _format_trade_time_label(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "N/A"
        # Broker order time may arrive as HHMMSS.
        if text.isdigit() and len(text) == 6:
            return f"{text[0:2]}:{text[2:4]}:{text[4:6]}"
        try:
            return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return text
    return "N/A"


def _log_trade_event_banner(event: str, code: str, qty: int, price: float, detail: str = "", code_name: str = "") -> None:
    """Prints a high-visibility trade event block to both console and file logs."""
    if str(event).strip().upper() == "BUY SUBMITTED":
        return
    line = "=" * 110
    title = f"*** {event} | {_format_code_label(code, code_name)} | qty={qty} | price={price:,.0f} ***"
    log(line)
    log(title)
    log_trade(line)
    log_trade(title)
    if detail:
        log(f"*** DETAIL: {detail}")
        log_trade(f"*** DETAIL: {detail}")
    log(line)
    log_trade(line)


# ---------------------------------------------------------------------------
# NXT 거래 가능 종목 판단
# ---------------------------------------------------------------------------

def _is_truthy_flag(value) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if text in {"Y", "1", "TRUE", "T", "O", "YES"}:
        return True
    if text in {"N", "0", "FALSE", "F", "X", "NO"}:
        return False
    return None


NXT_TRADABLE_CACHE: dict[str, bool] = {}
_NXT_PROBE_FAILED_LOGGED = False


def _probe_nxt_tradeable_via_stock_info(code: str) -> bool | None:
    stock_info_fn = getattr(dsf, "search_stock_info", None)
    if not callable(stock_info_fn):
        return None

    try:
        result = stock_info_fn(prdt_type_cd="300", pdno=code)
    except Exception as exc:
        log(f"WARNING: search_stock_info failed for {code}: {exc}")
        return None

    if result is None or getattr(result, "empty", True):
        return None

    row = result.iloc[-1]
    for key in ("cptt_trad_tr_psbl_yn", "nxt_tr_stop_yn", "tr_stop_yn"):
        if key not in result.columns:
            continue
        flag = _is_truthy_flag(row.get(key))
        if flag is None:
            continue
        return flag if key == "cptt_trad_tr_psbl_yn" else not flag

    return None


def is_nxt_tradeable(code: str) -> bool:
    global _NXT_PROBE_FAILED_LOGGED

    if code in NXT_TRADABLE_CACHE:
        return NXT_TRADABLE_CACHE[code]

    tradeable = _probe_nxt_tradeable_via_stock_info(code)
    if tradeable is not None:
        NXT_TRADABLE_CACHE[code] = tradeable
        return tradeable

    if not _NXT_PROBE_FAILED_LOGGED:
        log("WARNING: NXT probe failed; defaulting to False for unknown codes")
        _NXT_PROBE_FAILED_LOGGED = True

    NXT_TRADABLE_CACHE[code] = False
    return False


# ---------------------------------------------------------------------------
# 세션 헬퍼
# ---------------------------------------------------------------------------

def is_weekday_market_day(now: datetime) -> bool:
    return now.weekday() < 5


def _market_day__market_day_fail_closed_result(date_label: str, reason: str) -> tuple[bool, str]:
    return (
        False,
        f"MARKET DAY CHECK | {date_label} | CLOSED | source=fail_closed | reason={reason}",
    )


def get_market_day_status(now: datetime) -> tuple[bool, str]:
    cache_key = now.strftime("%Y%m%d")
    cached = _MARKET_DAY_STATUS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    date_label = now.strftime("%Y-%m-%d")
    if not is_weekday_market_day(now):
        result = (False, f"MARKET DAY CHECK | {date_label} | CLOSED | source=weekday | reason=weekend")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    holiday_fn = getattr(dsf, "chk_holiday", None)
    if not callable(holiday_fn):
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, "chk_holiday_unavailable")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday | reason=chk_holiday_unavailable")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    try:
        df = holiday_fn(bass_dt=now.strftime("%Y%m%d"))
    except Exception as exc:
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, f"chk_holiday_failed:{exc}")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=chk_holiday_failed:{exc}")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    if df is None or df.empty:
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, "empty_holiday_response")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=empty_holiday_response")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    date_str = now.strftime("%Y%m%d")
    if "bass_dt" in df.columns:
        row = df[df["bass_dt"].astype(str) == date_str]
        if row.empty:
            row = df.iloc[[0]]
    else:
        row = df.iloc[[0]]

    flag = _is_truthy_flag(row.iloc[-1].get("opnd_yn"))
    if flag is None:
        if MARKET_DAY_FAIL_CLOSED:
            result = _market_day__market_day_fail_closed_result(date_label, "unknown_opnd_yn")
        else:
            result = (True, f"MARKET DAY CHECK | {date_label} | OPEN | source=weekday_fallback | reason=unknown_opnd_yn")
        _MARKET_DAY_STATUS_CACHE[cache_key] = result
        return result

    status_text = "OPEN" if flag else "CLOSED"
    bass_dt = str(row.iloc[-1].get("bass_dt", date_str)).strip() or date_str
    opnd_yn = str(row.iloc[-1].get("opnd_yn", "")).strip() or "UNKNOWN"
    result = (flag, f"MARKET DAY CHECK | {date_label} | {status_text} | source=chk_holiday | bass_dt={bass_dt} | opnd_yn={opnd_yn}")
    _MARKET_DAY_STATUS_CACHE[cache_key] = result
    return result


def is_regular_session(now: datetime) -> bool:
    return REGULAR_START <= now.time() <= REGULAR_END


def is_regular_call_auction(now: datetime) -> bool:
    current_time = now.time()
    return REGULAR_NEW_ENTRY_CUTOFF <= current_time < REGULAR_END


def is_nxt_session(now: datetime) -> bool:
    if not ENABLE_NXT_SESSION:
        return False
    current_time = now.time()
    return (MORNING_NXT_START <= current_time <= MORNING_NXT_END) or (AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END)


def classify_buy_session(now: datetime) -> str:
    if not ENABLE_NXT_SESSION:
        return "regular"
    current_time = now.time()
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return "morning_nxt"
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return "afternoon_nxt"
    return "regular"


def can_trade_code_now(now: datetime, nxt_tradeable: bool) -> bool:
    current_time = now.time()
    if REGULAR_START <= current_time <= REGULAR_END:
        return True
    if not ENABLE_NXT_SESSION:
        return False
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return nxt_tradeable
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return nxt_tradeable
    return False


def is_new_entry_allowed(now: datetime, nxt_tradeable: bool) -> bool:
    if is_regular_session(now):
        return now.time() < REGULAR_NEW_ENTRY_CUTOFF
    if not ENABLE_NXT_SESSION or not nxt_tradeable or not is_nxt_session(now):
        return False
    current_time = now.time()
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return current_time < MORNING_NXT_NEW_ENTRY_CUTOFF
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return current_time < AFTERNOON_NXT_NEW_ENTRY_CUTOFF
    return False


def _fetch_bid_ask_price(code: str, market_div: str) -> tuple[float | None, float | None]:
    """매수1호가(bid)와 매도1호가(ask) 조회. 실패 시 (None, None) 반환."""
    try:
        df1, _ = dsf.inquire_asking_price_exp_ccn(
            env_dv=KIS_ENV_DV,
            fid_cond_mrkt_div_code=market_div,
            fid_input_iscd=str(code).zfill(6),
        )
        if df1.empty:
            return None, None
        row = df1.iloc[0]
        bid = float(row.get("bidp1") or 0) or None
        ask = float(row.get("askp1") or 0) or None
        return bid, ask
    except Exception as exc:
        log(f"WARNING: 호가 조회 실패 {code}: {exc}")
        return None, None


def _fetch_orderbook_totals(code: str, market_div: str) -> tuple[float | None, float | None]:
    """총 매도호가 잔량(ask)과 총 매수호가 잔량(bid) 조회. 실패 시 (None, None) 반환."""
    try:
        df1, _ = dsf.inquire_asking_price_exp_ccn(
            env_dv=KIS_ENV_DV,
            fid_cond_mrkt_div_code=market_div,
            fid_input_iscd=str(code).zfill(6),
        )
        if df1.empty:
            return None, None
        row = df1.iloc[0]
        ask_total = float(row.get("total_askp_rsqn") or 0) or None
        bid_total = float(row.get("total_bidp_rsqn") or 0) or None
        return ask_total, bid_total
    except Exception as exc:
        log(f"WARNING: 호가잔량 조회 실패 {code}: {exc}")
        return None, None


def get_order_spec(now: datetime, nxt_tradeable: bool) -> dict | None:
    if is_regular_session(now):
        return {"exchange": "KRX", "ord_dvsn": "01", "ord_unpr": "0"}
    if not ENABLE_NXT_SESSION:
        return None
    if is_nxt_session(now) and nxt_tradeable:
        return {"exchange": "NXT", "ord_dvsn": "00", "ord_unpr": None}
    return None


def get_session_open_datetime(now: datetime, nxt_tradeable: bool) -> datetime | None:
    current_time = now.time()
    if REGULAR_START <= current_time <= REGULAR_END:
        return datetime.combine(now.date(), REGULAR_START)
    if not ENABLE_NXT_SESSION:
        return None
    if nxt_tradeable and MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return datetime.combine(now.date(), MORNING_NXT_START)
    if nxt_tradeable and AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return datetime.combine(now.date(), AFTERNOON_NXT_START)
    return None


# ---------------------------------------------------------------------------
# 이익 실현
# ---------------------------------------------------------------------------

def _is_allowed_intraday_time(ts: pd.Timestamp, nxt_tradeable: bool) -> bool:
    t = ts.time()
    if REGULAR_START <= t <= REGULAR_END:
        return True
    if not ENABLE_NXT_SESSION:
        return False
    if nxt_tradeable and (
        (MORNING_NXT_START <= t <= MORNING_NXT_END)
        or (AFTERNOON_NXT_START <= t <= AFTERNOON_NXT_END)
    ):
        return True
    return False


def _normalize_intraday_frame(df: pd.DataFrame, target_date: str, nxt_tradeable: bool, bar_interval: str = "3min") -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    out = df.copy()
    rename_map = {
        "stck_cntg_hour": "time",
        "stck_oprc": "open",
        "stck_hgpr": "high",
        "stck_lwpr": "low",
        "stck_prpr": "close",
        "cntg_vol": "volume",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    if not {"time", "open", "high", "low", "close", "volume"}.issubset(set(out.columns)):
        return None

    out["time"] = out["time"].astype(str).str.zfill(6)
    out["datetime"] = pd.to_datetime(target_date + out["time"], format="%Y%m%d%H%M%S", errors="coerce")
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    out = out[~out.index.duplicated(keep="last")]

    allowed_mask = pd.Series([_is_allowed_intraday_time(ts, nxt_tradeable) for ts in out.index], index=out.index)
    out = out[allowed_mask]
    if out.empty:
        return None

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[(out[["open", "high", "low", "close"]].max(axis=1) > 0)].copy()
    if out.empty:
        return None

    out = out.resample(bar_interval, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])

    return out if not out.empty else None


def fetch_3min_frame(code: str, now: datetime, nxt_tradeable: bool) -> pd.DataFrame | None:
    if inquire_time_itemchartprice is None:
        return None

    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]
    today_str = now.strftime("%Y%m%d")

    for market_div in candidates:
        try:
            _, raw_df = inquire_time_itemchartprice(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
                fid_input_hour_1=now.strftime("%H%M%S"),
                fid_pw_data_incu_yn="Y",
                fid_etc_cls_code="",
            )
        except Exception as exc:
            log(f"WARNING: chart fetch failed for {code} ({market_div}): {exc}")
            continue

        frame = _normalize_intraday_frame(raw_df, today_str, nxt_tradeable)
        if frame is not None and not frame.empty:
            # fetch 시점 기준으로 확정 완료된 3분봉까지만 사용 (거래소 API 반영 지연 감안,
            # CANDLE_CONFIRM_DELAY_SECONDS만큼 여유를 두고 확정 판정)
            last_closed_bar = (pd.Timestamp(now) - pd.Timedelta(seconds=CANDLE_CONFIRM_DELAY_SECONDS)).floor("3min")
            frame = frame[frame.index <= last_closed_bar]
            if frame.empty:
                continue
            return calculate_indicators(frame)

    return None


def fetch_1min_frame(code: str, now: datetime, nxt_tradeable: bool) -> pd.DataFrame | None:
    """1분봉 프레임 조회 (1분봉 BB 중간값 골든크로스 매수 판단 전용)."""
    if inquire_time_itemchartprice is None:
        return None

    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]
    today_str = now.strftime("%Y%m%d")

    for market_div in candidates:
        try:
            _, raw_df = inquire_time_itemchartprice(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
                fid_input_hour_1=now.strftime("%H%M%S"),
                fid_pw_data_incu_yn="Y",
                fid_etc_cls_code="",
            )
        except Exception as exc:
            log(f"WARNING: 1min chart fetch failed for {code} ({market_div}): {exc}")
            continue

        frame = _normalize_intraday_frame(raw_df, today_str, nxt_tradeable, bar_interval="1min")
        if frame is not None and not frame.empty:
            # fetch 시점 기준으로 확정 완료된 1분봉까지만 사용 (거래소 API 반영 지연 감안,
            # CANDLE_CONFIRM_DELAY_SECONDS만큼 여유를 두고 확정 판정)
            last_closed_bar = (pd.Timestamp(now) - pd.Timedelta(seconds=CANDLE_CONFIRM_DELAY_SECONDS)).floor("1min")
            frame = frame[frame.index <= last_closed_bar]
            if frame.empty:
                continue
            return calculate_indicators(frame)

    return None


def fetch_live_price(code: str, now: datetime, nxt_tradeable: bool) -> float | None:
    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]

    for market_div in candidates:
        try:
            quote_df = dsf.inquire_price(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
            )
        except Exception as exc:
            log(f"WARNING: live price fetch failed for {code} ({market_div}): {exc}")
            continue

        if quote_df is None or quote_df.empty:
            continue

        row = quote_df.iloc[-1]
        for key in ("stck_prpr", "prpr"):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return None


def fetch_prev_close(code: str, now: datetime, nxt_tradeable: bool) -> float | None:
    candidates = ["NX", "UN", "J"] if (is_nxt_session(now) and nxt_tradeable) else ["J", "UN"]

    for market_div in candidates:
        try:
            quote_df = dsf.inquire_price(
                env_dv=KIS_ENV_DV,
                fid_cond_mrkt_div_code=market_div,
                fid_input_iscd=code,
            )
        except Exception:
            continue

        if quote_df is None or quote_df.empty:
            continue

        row = quote_df.iloc[-1]
        for key in ("stck_sdpr", "bfdy_clpr", "prdy_clpr", "stck_prdy_clpr"):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return None


def should_refresh_3min_frame(
    now: datetime,
    cached_frame: pd.DataFrame | None,
    last_refresh_at: datetime | None,
) -> bool:
    if cached_frame is None or cached_frame.empty:
        return True

    if last_refresh_at is None:
        return True

    # Poll frame data every 20 seconds even if no new 3-minute bar closed yet.
    # This keeps server-side data updates in sync with live checks.
    elapsed_seconds = (now - last_refresh_at).total_seconds()
    if elapsed_seconds >= FRAME_POLL_INTERVAL_SECONDS:
        return True

    # Extra safety backfill in case refresh timestamps drift unexpectedly.
    if elapsed_seconds >= FRAME_BACKFILL_SYNC_SECONDS:
        return True

    return False


# KIS's inquire_time_itemchartprice only returns a recent rolling window per call
# (observed: ~10 three-minute bars after resampling), so simply replacing the cache
# with each fresh call permanently starves rolling indicators that need more history
# (BB_PERIOD/VOLUME_MA_PERIOD=20 never actually see 20 bars). Accumulate raw OHLCV
# across refreshes instead, then recompute indicators once over the full history.
FRAME_CACHE_MAX_BARS = 200  # generous cap (~10h of 3-min bars); covers a full NXT+regular session


def _merge_bar_frame(previous: pd.DataFrame | None, refreshed: pd.DataFrame) -> pd.DataFrame:
    """봉 간격에 무관한 공용 병합 로직 (3분봉/1분봉 캐시 겸용)."""
    if previous is None or previous.empty:
        combined = refreshed
    else:
        combined = pd.concat([previous, refreshed])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    if len(combined) > FRAME_CACHE_MAX_BARS:
        combined = combined.tail(FRAME_CACHE_MAX_BARS)
    return calculate_indicators(combined[["open", "high", "low", "close", "volume"]])


def _get_or_refresh_1min_frame(
    code: str,
    current_dt: datetime,
    nxt_tradeable: bool,
    frame_cache_1min: dict[str, pd.DataFrame],
    frame_last_refresh_at_1min: dict[str, datetime],
) -> pd.DataFrame | None:
    """1분봉 캐시 조회/갱신 헬퍼 (하이브리드 1분봉 트리거용)."""
    cached_frame_1min = frame_cache_1min.get(code)
    last_frame_refresh_1min = frame_last_refresh_at_1min.get(code)
    frame_1min = cached_frame_1min
    if should_refresh_3min_frame(current_dt, cached_frame_1min, last_frame_refresh_1min):
        try:
            refreshed_frame_1min = fetch_1min_frame(code, current_dt, nxt_tradeable)
        except Exception as exc:
            log(f"{code} 1min frame error: {exc}")
            refreshed_frame_1min = None
        if refreshed_frame_1min is not None and not refreshed_frame_1min.empty:
            frame_1min = _merge_bar_frame(cached_frame_1min, refreshed_frame_1min)
            frame_cache_1min[code] = frame_1min
            frame_last_refresh_at_1min[code] = current_dt
    return frame_1min


def _live_price_backoff_seconds(fail_count: int) -> int:
    # Exponential backoff: 5s, 10s, 20s, 40s, 60s cap.
    seconds = LIVE_PRICE_BACKOFF_BASE_SECONDS * (2 ** max(0, fail_count - 1))
    return int(min(LIVE_PRICE_BACKOFF_MAX_SECONDS, seconds))


def _pending_status_backoff_seconds(fail_count: int) -> int:
    # Exponential backoff with wider cap for repeated status-query failures.
    seconds = ORDER_STATUS_POLL_INTERVAL_SECONDS * (2 ** max(0, fail_count - 1))
    return int(min(PENDING_STATUS_BACKOFF_MAX_SECONDS, seconds))


def _next_aligned_tick(after: datetime, interval_seconds: int) -> datetime:
    """`after` 이후 가장 가까운 정렬 시각을 반환한다 (interval_seconds=20이면 매분 00/20/40초).
    반환값은 항상 `after`보다 뒤(다음 틱)이다."""
    if interval_seconds <= 0:
        return after
    minute_start = after.replace(second=0, microsecond=0)
    elapsed = (after - minute_start).total_seconds()
    steps = int(elapsed // interval_seconds) + 1
    return minute_start + timedelta(seconds=steps * interval_seconds)


def _sleep_until_next_tick(scheduled_tick: datetime, interval_seconds: int) -> datetime:
    """다음 정렬 틱(예: 매분 00/20/40초)까지 대기하고 그 시각을 반환한다.
    직전 처리가 한 틱 구간을 넘겨 지연됐다면 이미 지나간 틱은 건너뛰고 그 다음 정렬
    시각으로 넘어간다 - 밀린 틱을 몰아서 처리하지 않기 위함."""
    next_tick = _next_aligned_tick(scheduled_tick, interval_seconds)
    now = datetime.now()
    while next_tick <= now:
        next_tick = _next_aligned_tick(next_tick, interval_seconds)
    time.sleep(max(0.0, (next_tick - now).total_seconds()))
    return next_tick


def update_live_price_cross_state(
    cross_state: dict[str, dict],
    code: str,
    now: datetime,
    live_price: float,
    bb_middle: float,
) -> dict[str, object]:
    return shared_update_live_price_cross_state(
        cross_state=cross_state,
        code=code,
        now=pd.Timestamp(now),
        live_price=live_price,
        bb_middle=bb_middle,
        config=SHARED_R76_CONFIG,
    )
def _build_realtime_entry_frame(
    frame: pd.DataFrame,
    code: str,
    now: datetime,
    live_price: float,
    realtime_bar_state: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, float]:
    now_ts = pd.Timestamp(now)
    bar_start = now_ts.floor("3min")
    bar_end = bar_start + pd.Timedelta(minutes=3)
    elapsed_seconds = max(0.0, (now_ts - bar_start).total_seconds())

    if frame is None or frame.empty or pd.isna(live_price) or live_price <= 0:
        return frame, elapsed_seconds

    state = realtime_bar_state.get(code)
    if state is None or state.get("bar_end") != bar_end:
        state = {
            "bar_end": bar_end,
            "open": float(live_price),
            "high": float(live_price),
            "low": float(live_price),
            "close": float(live_price),
        }
    else:
        state["high"] = max(float(state.get("high", live_price)), float(live_price))
        state["low"] = min(float(state.get("low", live_price)), float(live_price))
        state["close"] = float(live_price)

    realtime_bar_state[code] = state

    working = frame.copy()
    # 현재 미완성봉 거래량: API가 10초 누적을 신뢰성 있게 제공하지 않으므로
    # 직전 확정봉의 거래량을 대리값으로 사용해 volume=0 으로 인한 리젝을 방지한다.
    prev_bar_volume = float(frame.iloc[-1]["volume"]) if not frame.empty else 0.0
    realtime_row = {
        "open": float(state["open"]),
        "high": float(state["high"]),
        "low": float(state["low"]),
        "close": float(state["close"]),
        "volume": prev_bar_volume,
    }
    if len(working) > 0 and working.index[-1] == bar_end:
        for key, val in realtime_row.items():
            working.at[bar_end, key] = val
    else:
        working.loc[bar_end, ["open", "high", "low", "close", "volume"]] = [
            realtime_row["open"],
            realtime_row["high"],
            realtime_row["low"],
            realtime_row["close"],
            realtime_row["volume"],
        ]
    working = working.sort_index()
    return calculate_indicators(working), elapsed_seconds


def _passes_intrabar_entry_gate(frame: pd.DataFrame, elapsed_seconds: float) -> tuple[bool, str]:
    """미완성(인트라바) 3분봉을 매수 판단에 신뢰해도 되는지 여부.
    r003의 INTRABAR_MIN_ELAPSED_SECONDS/RSI/MFI/ADX 기준을 모두 통과해야 True.
    실패 시 호출자는 반드시 진짜 확정된 3분봉으로 폴백해야 한다 - 그렇지 않으면 실시간가
    변동이 확정 종가 크로스(CLOSE_BB_UP_CROSS)로 오인되어 매수될 수 있다.

    RSI는 [2026-09-18]부터 2단계: INTRABAR_RSI_MIN~MAX(45~65)는 무조건 통과, MAX 초과
    ~INTRABAR_RSI_EXT_MAX(65~75)는 RSI/MACD 상승 + 거래량 VOL_MA20 상회 + BB중간선
    위 4조건을 전부 만족해야 통과, EXT_MAX 초과는 무조건 거부."""
    if elapsed_seconds < INTRABAR_MIN_ELAPSED_SECONDS:
        return False, f"INTRABAR_ELAPSED_{elapsed_seconds:.0f}s_LT_{INTRABAR_MIN_ELAPSED_SECONDS:.0f}s"
    if frame is None or frame.empty:
        return False, "INTRABAR_FRAME_EMPTY"
    cur = frame.iloc[-1]
    rsi = _num(cur, "RSI")
    if pd.isna(rsi) or rsi < INTRABAR_RSI_MIN or rsi > INTRABAR_RSI_EXT_MAX:
        return False, f"INTRABAR_RSI_{rsi:.1f}_OUT_OF_{INTRABAR_RSI_MIN:.0f}-{INTRABAR_RSI_EXT_MAX:.0f}"
    if rsi > INTRABAR_RSI_MAX:
        # [2026-09-18] 65~75 확장 구간(사용자 요청) - 이미 강하게 상승 중인 종목의
        # 인트라바 RSI가 기본 상한(65)을 넘었다고 무조건 거부하지 않고, RSI/MACD가
        # 실제로 계속 오르고 있고(직전봉 대비) 거래량도 VOL_MA20을 웃돌며 가격이
        # BB중간선 위인 경우에만 인트라바 데이터를 신뢰한다 - 4개 모두 충족해야 통과.
        if len(frame) < 2:
            return False, f"INTRABAR_RSI_{rsi:.1f}_EXT_COND_NOT_MET"
        prev = frame.iloc[-2]
        prev_rsi = _num(prev, "RSI")
        cur_macd = _num(cur, "MACD")
        prev_macd = _num(prev, "MACD")
        vol = _num(cur, "volume")
        vol_ma = _num(cur, "VOL_MA20")
        cur_close = _num(cur, "close")
        cur_bb_mid = _num(cur, "BB_MIDDLE")
        ext_ok = (
            not pd.isna(prev_rsi) and rsi > prev_rsi
            and not any(pd.isna(v) for v in (cur_macd, prev_macd)) and cur_macd > prev_macd
            and not any(pd.isna(v) for v in (vol, vol_ma)) and vol > vol_ma
            and not any(pd.isna(v) for v in (cur_close, cur_bb_mid)) and cur_close > cur_bb_mid
        )
        if not ext_ok:
            return False, f"INTRABAR_RSI_{rsi:.1f}_EXT_COND_NOT_MET"
    mfi = _num(cur, "MFI")
    if pd.isna(mfi) or not (INTRABAR_MFI_MIN <= mfi <= INTRABAR_MFI_MAX):
        return False, f"INTRABAR_MFI_{mfi:.1f}_OUT_OF_{INTRABAR_MFI_MIN:.0f}-{INTRABAR_MFI_MAX:.0f}"
    adx = _num(cur, "ADX")
    if pd.isna(adx) or adx < INTRABAR_ADX_MIN:
        return False, f"INTRABAR_ADX_{adx:.1f}_LT_{INTRABAR_ADX_MIN:.0f}"
    return True, "INTRABAR_GATE_PASS"


# ---------------------------------------------------------------------------
# 전략 로직
# ---------------------------------------------------------------------------

def _num(candle: pd.Series, key: str) -> float:
    value = candle.get(key)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


SYMBOL_LOG_WIDTH = 25


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _symbol_log_label(code: str, name: str, width: int = SYMBOL_LOG_WIDTH) -> str:
    label = f"{code}_{name}" if name else code
    pad = max(0, width - _display_width(label))
    return label + " " * pad


def _buy_reject_detail(
    buy_reason: str,
    cur: pd.Series,
    prev: pd.Series,
    live_price: float | None = None,
    cross_info: dict | None = None,
    frame: pd.DataFrame | None = None,
) -> str:
    """리젝 사유 + 지표 스냅샷. 하이브리드 경로의 사유(HYBRID_1MIN_TRIGGER_*/HYBRID_3MIN_CTX_*)는 값이
    이미 사유 문자열에 들어 있어 스냅샷만 덧붙인다.

    [2026-09-20] 3분봉 단독 경로(삭제됨)의 사유(BB_SLOPE_NOT_RISING/NO_BB_MID_CROSS_UP/CANDLE_NOT_
    BULLISH/BB_UPPER_GAP_TOO_SMALL/LOW_VOLUME_RATIO/LOW_SCORE)별 상세 분기를 삭제했다 - 하이브리드
    사유는 항상 "HYBRID_"로 시작해 이 분기들에 한 번도 걸리지 않았다(로그 출력 불변)."""
    snapshot = _buy_condition_snapshot(cur, prev, live_price=live_price, cross_info=cross_info, frame=frame)
    return f"{buy_reason} | {snapshot}"



def _buy_support_score(cur: pd.Series, prev: pd.Series, frame: pd.DataFrame | None = None) -> int:
    """로그 표시용 점수 계산 - r002._buy_support_score(실제 매수 판정이 쓰는 BUY_SCORE_RULES
    그 자체)에 위임한다.

    [2026-09-17] 이 함수는 원래 r002의 BUY_SCORE_RULES를 손으로 복제한 별도 구현이었는데,
    2026-09-13 R77 리팩터(pre_cross_accumulation을 가점으로 전환, bb_upper_room_atr 신규
    추가, volume_up_direction 가점 삭제)가 r002에만 반영되고 이 로컬 복제본엔 반영되지
    않아 드리프트가 발생했다 - 사용자 요청으로 진행한 매수 0건 로그 분석 중 발견(CHECK/
    REJECT 로그의 "score=X/22"가 2026-09-13 이후 전부 실제 판정 점수와 다른 값을 표시하고
    있었음: 이미 삭제된 volume_up_direction(+1)을 여전히 더하고, 신규 pre_cross_
    accumulation(+1)/bb_upper_room_atr(+2)는 누락). 2026-08-24에 close_cross 스냅샷에서
    겪은 것과 동일한 계열의 버그(표시용 재구현이 실제 판정 로직과 따로 놀다 어긋남) -
    이번엔 재구현 대신 실제 판정 함수를 그대로 재사용해 원천적으로 드리프트를 차단한다.
    """
    if frame is None:
        return -1
    return _shared_buy_support_score(cur, prev, frame, SHARED_R76_CONFIG)


def _buy_condition_snapshot(
    cur: pd.Series,
    prev: pd.Series,
    live_price: float | None = None,
    cross_info: dict | None = None,
    frame: pd.DataFrame | None = None,
) -> str:
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_bb = _num(cur, "BB_MIDDLE")
    cur_bb_upper = _num(cur, "BB_UPPER")
    cur_open = _num(cur, "open")
    support_score = _buy_support_score(cur, prev, frame=frame) if frame is not None else -1
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    vol_ratio = vol / vol_ma if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0 else float("nan")
    adx_c = _num(cur, "ADX")
    di_plus = _num(cur, "DI_PLUS"); di_minus = _num(cur, "DI_MINUS")
    rsi_c = _num(cur, "RSI")
    macd_c = _num(cur, "MACD"); msig_c = _num(cur, "MACD_SIGNAL")
    bb_slope_pct = _compute_bb_slope_pct(frame) if frame is not None else float("nan")
    # 실제 매수 판정 함수(run_buy_condition_pipeline_comment)와 동일한 _evaluate_bb_mid_cross()를
    # 그대로 재사용 - 예전엔 여기서 즉시 2봉 close_cross만 따로 계산해서, 실제로는 5봉 룩백/
    # 우상향추세지속으로 통과하는 케이스가 로그엔 close_cross=False로 잘못 찍히곤 했음
    # (2026-08-24, r005 Update log 참조).
    if frame is not None and not frame.empty and live_price is not None and pd.notna(live_price):
        _cross_eval = _evaluate_bb_mid_cross(
            frame, cur, prev, cur_bb, prev_bb, float(live_price), bb_slope_pct, cross_info or {}
        )
        close_cross = _cross_eval["close_cross"]
        uptrend_cont = _cross_eval["uptrend_continuation"]
        cross_pass = _cross_eval["passed"]
    else:
        prev_close = _num(prev, "close")
        cur_close = _num(cur, "close")
        close_cross = (
            not any(pd.isna(v) for v in (prev_close, cur_close, prev_bb, cur_bb))
            and prev_close <= prev_bb and cur_close > cur_bb
        )
        uptrend_cont = False
        cross_pass = close_cross or ((cross_info or {}).get("signal") == "cross_up")
    candle_gain_pct = (live_price - cur_open) / cur_open * 100.0 if (live_price and not pd.isna(cur_open) and cur_open > 0) else float("nan")
    bb_upper_gap_pct = (cur_bb_upper - live_price) / live_price * 100.0 if (live_price and live_price > 0 and not pd.isna(cur_bb_upper)) else float("nan")
    live_signal = cross_info.get("signal") if cross_info else None
    live_part = f"live={live_price:,.0f}" if live_price is not None and pd.notna(live_price) else "live=nan"
    return (
        f"GATES cross_pass={cross_pass} close_cross={close_cross} uptrend_cont={uptrend_cont} signal={live_signal} {live_part} "
        f"bb_mid={cur_bb:.1f} bb_upper={cur_bb_upper:.1f} "
        f"bb_slope={bb_slope_pct:.3f}% bb_upper_gap={bb_upper_gap_pct:.2f}% candle_gain={candle_gain_pct:.2f}% "
        f"RSI={rsi_c:.1f} ADX={adx_c:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f} MACD={macd_c:.3f} SIG={msig_c:.3f} "
        f"vol={vol:,.0f} vol_ma={vol_ma:,.0f} vol_ratio={vol_ratio:.4f} score={support_score}/{BUY_SCORE_MAX}"
    )


def _rise_from_prev_close(live_price: float, prev_close: float) -> float | None:
    if live_price <= 0 or prev_close <= 0:
        return None
    return (float(live_price) / float(prev_close)) - 1.0


def check_sell_condition(frame: pd.DataFrame, pnl_pct: float, live_price: float, cross_info: dict[str, object]) -> tuple[bool, str]:
    return shared_check_sell_condition(
        frame=frame,
        pnl_pct=pnl_pct,
        live_price=live_price,
        cross_info=cross_info,
        config=SHARED_R76_CONFIG,
    )


# ---------------------------------------------------------------------------
# Live state persistence (DATA_DIR/live_state/YYYYMMDD.json)
# ---------------------------------------------------------------------------

def _live_state_path(date_str: str) -> Path:
    return LIVE_STATE_DIR / f"{date_str}.json"


def _sanitize_surge_ladder(raw: object) -> dict | None:
    """포지션 pos["surge_ladder"](급등 사다리 익절 상태)를 정규화한다. 사다리가 없거나 값이 깨졌으면 None.

    형식: {"base": 1차 익절 체결가(float, >0), "tp2_done": bool, "tp3_done": bool}. live_state JSON 저장/
    복원, 계좌 동기화(sync_positions_from_account)로 pos dict가 새로 만들어지는 경로 모두에서 같은 검증을 쓴다.
    """
    if not isinstance(raw, dict):
        return None
    try:
        base = float(raw.get("base", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not (base > 0.0):
        return None
    return {
        "base": base,
        "tp2_done": bool(raw.get("tp2_done", False)),
        "tp3_done": bool(raw.get("tp3_done", False)),
    }


def _serialize_live_state(live_state: dict) -> dict:
    def _json_safe(value):
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "isoformat") and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value.isoformat()
            except Exception:
                pass
        if isinstance(value, set):
            return sorted(_json_safe(item) for item in value)
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return value

    positions_meta = {}
    for code, meta in (live_state.get("positions_meta") or {}).items():
        positions_meta[str(code).zfill(6)] = {
            "buy_time": _json_safe(meta.get("buy_time")),
            "entry_buy_time": _json_safe(meta.get("entry_buy_time") or meta.get("buy_time")),
            "buy_session": _json_safe(meta.get("buy_session")),
            "highest_price": _json_safe(meta.get("highest_price")),
            "entry_quantity": int(meta.get("entry_quantity", 0) or 0),
            "tp1_done": bool(meta.get("tp1_done", False)),
            "tp2_done": bool(meta.get("tp2_done", False)),
            "tp3_done": bool(meta.get("tp3_done", False)),
            "pyramid_done": bool(meta.get("pyramid_done", False)),
            "surge_ladder": _sanitize_surge_ladder(meta.get("surge_ladder")),
        }
    traded = sorted({str(c).zfill(6) for c in (live_state.get("traded_today") or set())})
    return {"positions_meta": positions_meta, "traded_today": traded}


def load_live_state(date_str: str) -> dict:
    path = _live_state_path(date_str)
    if not path.exists():
        return {"date": date_str, "positions_meta": {}, "traded_today": set()}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"WARNING: live state load failed ({path}): {exc}")
        return {"date": date_str, "positions_meta": {}, "traded_today": set()}

    positions_meta: dict[str, dict] = {}
    for code, meta in (raw.get("positions_meta") or {}).items():
        norm = str(code).zfill(6)
        buy_time_raw = (meta or {}).get("buy_time")
        entry_buy_time_raw = (meta or {}).get("entry_buy_time")
        buy_time = None
        if buy_time_raw:
            try:
                buy_time = datetime.fromisoformat(str(buy_time_raw))
            except ValueError:
                buy_time = _parse_buy_time_from_holding_fields(meta)
        entry_buy_time = None
        if entry_buy_time_raw:
            try:
                entry_buy_time = datetime.fromisoformat(str(entry_buy_time_raw))
            except ValueError:
                entry_buy_time = buy_time
        if entry_buy_time is None:
            entry_buy_time = buy_time
        positions_meta[norm] = {
            "buy_time": buy_time,
            "entry_buy_time": entry_buy_time,
            "buy_session": (meta or {}).get("buy_session"),
            "highest_price": (meta or {}).get("highest_price"),
            "entry_quantity": int((meta or {}).get("entry_quantity", 0) or 0),
            "tp1_done": bool((meta or {}).get("tp1_done", False)),
            "tp2_done": bool((meta or {}).get("tp2_done", False)),
            "tp3_done": bool((meta or {}).get("tp3_done", False)),
            "pyramid_done": bool((meta or {}).get("pyramid_done", False)),
            "surge_ladder": _sanitize_surge_ladder((meta or {}).get("surge_ladder")),
        }
    traded = {str(c).zfill(6) for c in (raw.get("traded_today") or [])}
    return {"date": date_str, "positions_meta": positions_meta, "traded_today": traded}


def save_live_state(live_state: dict, date_str: str | None = None) -> None:
    date_key = date_str or str(live_state.get("date") or _LOG_CTX.get("date_str") or datetime.now().strftime("%Y%m%d"))
    live_state["date"] = date_key
    path = _live_state_path(date_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_live_state(live_state)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    save_today_buy_codes(date_key, {str(c).zfill(6) for c in (payload.get("traded_today") or [])})


def _parse_buy_time_from_holding_fields(row_or_meta: dict) -> datetime | None:
    if not isinstance(row_or_meta, dict):
        return None
    for key in ("pchs_dt", "buy_dt", "ord_dt", "pchs_date"):
        raw = row_or_meta.get(key)
        if raw in (None, ""):
            continue
        text = str(raw).strip()
        try:
            if len(text) >= 14 and text[:14].isdigit():
                return datetime.strptime(text[:14], "%Y%m%d%H%M%S")
            if len(text) >= 8 and text[:8].isdigit():
                return datetime.strptime(text[:8], "%Y%m%d")
        except ValueError:
            continue
    return None


def _is_stale_live_price_source(price_source: str) -> bool:
    src = str(price_source or "")
    return "stale_live=" in src or src.startswith("bar_close(stale")


def _append_watchlist_change_history(current_dt: datetime, lines: list[str]) -> None:
    if not lines:
        return
    date_str = current_dt.strftime("%Y%m%d")
    history_path = data_sim_dir / "logs" / date_str / f"{date_str}_change_history.txt"
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = current_dt.strftime("%Y-%m-%d %H:%M:%S")
        with open(history_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{stamp} {line}\n")
    except Exception as exc:
        log(f"WARNING: watchlist change history write failed: {exc}")


def _log_account_watchlist_mismatch(api, watch_map: dict[str, str]) -> None:
    open_codes = {str(c).zfill(6) for c, p in api.get_open_positions().items() if int(p.get("quantity", 0) or 0) > 0}
    watch_codes = {str(c).zfill(6) for c in watch_map}
    extra = sorted(open_codes - watch_codes)
    if extra:
        log(f"WARNING: account holdings not in watchlist: {', '.join(extra)}")


def _rebalance_active_watchlist(
    current_dt: datetime,
    api,
    watch_map: dict[str, str],
    active_set: set[str],
    backup_pool: "collections.deque[str]",
    first_active_at: dict[str, datetime],
    hard_stop_today_codes: set[str],
    gap_blocked_codes: set[str],
    warn_state: dict[str, bool],
    frame_cache: dict[str, "pd.DataFrame"],
) -> None:
    """[2026-08-31] active_set(실시간 폴링 상한, ACTIVE_WATCHLIST_SIZE)과 backup_pool(대기,
    미폴링) 사이의 교체를 매 틱 처리한다. 이탈 사유:
    - GRADUATE: 매수 체결/미체결주문/체결대기(api.has_buy_exposure) - 진입후보 역할이
      끝났으므로 active_set에서는 빠지지만, 청산 감시(TP/SL/트레일링)는 메인 루프의
      position_codes union이 계속 커버하므로 문제 없다.
    - HARD_STOP / GAP_BLOCKED: 오늘 더 이상 신규 진입 대상이 될 수 없는 종목.
    - TIME_LIMIT: ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES(60분) 경과 후에도 여전히 매수신호가
      없는 종목. [2026-09-05] 기존엔 이 60분 타이머가 실제 가격이 박스권(정체)인지와 무관하게
      무조건 탈락시켰는데, 8/17~9/4 로그 분석 결과 이 기능이 도입된 8/31 이후 실거래 체결이
      일 0~1건으로 급감(그 이전 8/18~8/28은 일 30~200건대)한 것과 시기가 정확히 겹침 - 아직
      방향성을 만들어가는 중(박스권을 이미 벗어났거나 벗어나는 과정)인 종목까지 순수 타이머로
      무차별 탈락시켜 backup과 교체해버린 것이 매수 기회 자체를 놓친 원인으로 추정됨. 이제
      60분 경과 시점부터는 _is_box_range_hold_zone()(기존 매도측 "박스권 보유" 판정과 동일
      기준 재사용)으로 실제 박스권 여부를 확인해, 박스권이 확인된 종목만 TIME_LIMIT으로
      교체하고 아직 방향성이 살아있는(박스권이 아닌) 종목은 ACTIVE_WATCHLIST_HARD_TIME_LIMIT_
      MINUTES(120분, 9/3 backup_pool 고갈 사고 재발 방지용 안전장치)까지 계속 감시를 유지한다.
      프레임이 없거나(아직 미체결 조회 등) 판정 불가한 경우는 안전하게 기존과 동일하게 60분에
      탈락시킨다. HARD_STOP/GAP_BLOCKED와 달리 이 종목이 나쁜 게 아니라 그 시점에 신호가 없었을
      뿐이므로 backup_pool 맨 뒤로 재편입시켜 재도전 기회를 준다(아직 안 써본 다른
      backup 종목들이 우선권을 가지도록 뒤에 붙인다). [2026-09-03] 이 재편입이 없으면
      watch_map(스캐너 100개)이 하루 동안 TIME_LIMIT으로만 서서히 소진되어 active_set과
      backup_pool이 둘 다 0이 되는 사고가 발생한다(장중 재스캔이 없으므로 완전히
      감시 종목이 사라짐) - 실제 09/03 13:10경 이 현상으로 감시 종목이 전멸했던 것을
      수정.
    빠진 자리는 backup_pool 선두(스캐너 점수 순위 순서)부터 채운다. 승격된 종목은 추가
    확인 API 호출 없이 다음 폴링(이미 iter_codes에 포함되어 정상 진행됨)이 그 역할을 한다.
    """
    graduated: list[str] = []
    dropped: list[tuple[str, str]] = []

    for code in list(active_set):
        if api.has_buy_exposure(code):
            active_set.discard(code)
            first_active_at.pop(code, None)
            graduated.append(code)
            continue

        if code in hard_stop_today_codes:
            reason = "HARD_STOP"
        elif code in gap_blocked_codes:
            reason = "GAP_BLOCKED"
        else:
            started = first_active_at.get(code, current_dt)
            elapsed_minutes = (current_dt - started).total_seconds() / 60.0
            if elapsed_minutes < ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES:
                continue
            if elapsed_minutes >= ACTIVE_WATCHLIST_HARD_TIME_LIMIT_MINUTES:
                reason = "TIME_LIMIT_HARD"
            else:
                frame = frame_cache.get(code)
                if frame is None:
                    reason = "TIME_LIMIT"
                else:
                    is_box, box_info = _is_box_range_hold_zone(frame, SHARED_R76_CONFIG)
                    if not is_box:
                        continue
                    reason = f"TIME_LIMIT_BOX_{box_info}"

        active_set.discard(code)
        first_active_at.pop(code, None)
        dropped.append((code, reason))

    history_lines: list[str] = []
    for code in graduated:
        log(
            f"[ACTIVE GRADUATE] {code}_{watch_map.get(code, code)} | reason=POSITION_OPENED | "
            f"active={len(active_set)} backup={len(backup_pool)}"
        )
        history_lines.append(f"[GRADUATE] {code}_{watch_map.get(code, code)} | reason=POSITION_OPENED")
    for code, reason in dropped:
        if reason.startswith("TIME_LIMIT"):
            backup_pool.append(code)
        log(
            f"[ACTIVE DROPOUT] {code}_{watch_map.get(code, code)} | reason={reason} | "
            f"active={len(active_set)} backup={len(backup_pool)}"
        )
        history_lines.append(f"[DROPOUT] {code}_{watch_map.get(code, code)} | reason={reason}")

    promoted: list[str] = []
    while len(active_set) < ACTIVE_WATCHLIST_SIZE and backup_pool:
        code = backup_pool.popleft()
        active_set.add(code)
        first_active_at[code] = current_dt
        promoted.append(code)
    for code in promoted:
        log(f"[ACTIVE PROMOTE] {code}_{watch_map.get(code, code)} | active={len(active_set)} backup={len(backup_pool)}")
        history_lines.append(f"[PROMOTE] {code}_{watch_map.get(code, code)}")
    _append_watchlist_change_history(current_dt, history_lines)

    if not backup_pool and len(active_set) < ACTIVE_WATCHLIST_SIZE:
        if not warn_state.get("backup_pool_exhausted_logged"):
            log(
                f"WARNING: backup_pool 소진 - active_set이 상한({ACTIVE_WATCHLIST_SIZE}) 밑으로 "
                f"유지됨 (active={len(active_set)})"
            )
            warn_state["backup_pool_exhausted_logged"] = True
    elif backup_pool and warn_state.get("backup_pool_exhausted_logged"):
        warn_state["backup_pool_exhausted_logged"] = False


def _write_active_watchlist_state(
    current_dt: datetime,
    watch_map: dict[str, str],
    active_set: set[str],
    backup_pool: "collections.deque[str]",
    first_active_at: dict[str, datetime],
    position_codes: set[str],
) -> None:
    """[2026-09-02] active_set/backup_pool 현재 상태를 ACTIVE_WATCHLIST_STATE_PATH에
    매 틱 스냅샷으로 기록한다 - 실행 중인 봇이 지금 실시간으로 어떤 종목을 감시 중인지
    로그 파일을 grep하지 않고도 바로 확인할 수 있도록 함(사용자 요청). active는
    active_since(편입 시각) 오름차순 정렬 - 다음 TIME_LIMIT 탈락에 가장 가까운 순서.
    backup은 deque 순서(=스캐너 점수 순위) 그대로. 임시파일 작성 후 os.replace로 갈아끼워
    이 파일을 동시에 읽는 중에도 부분 기록이 보이지 않도록 함."""
    try:
        active_sorted = sorted(active_set, key=lambda c: first_active_at.get(c, current_dt))
        payload = {
            "updated_at": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "active_count": len(active_set),
            "backup_count": len(backup_pool),
            "active": [
                {
                    "code": code,
                    "name": watch_map.get(code, code),
                    "active_since": first_active_at.get(code, current_dt).strftime("%Y-%m-%d %H:%M:%S"),
                    "holding": code in position_codes,
                }
                for code in active_sorted
            ],
            "backup": [
                {"code": code, "name": watch_map.get(code, code)}
                for code in backup_pool
            ],
        }
        tmp_path = ACTIVE_WATCHLIST_STATE_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, ACTIVE_WATCHLIST_STATE_PATH)
    except Exception as exc:
        log(f"WARNING: active watchlist state write failed: {exc}")


def _write_active_watchlist_txt(
    current_dt: datetime,
    watch_map: dict[str, str],
    active_set: set[str],
    first_active_at: dict[str, datetime],
) -> None:
    """[2026-09-04] 현재 매수 감시 중인 종목(active_set)을 ACTIVE_WATCHLIST_TXT_PATH(스크립트와
    같은 폴더의 active_watchlist.txt)에 "종목코드,종목명" 형식으로 매 틱 기록한다(사용자 요청).
    active_since 오름차순 정렬로 ACTIVE_WATCHLIST_STATE_PATH(json)와 순서를 맞춤."""
    try:
        active_sorted = sorted(active_set, key=lambda c: first_active_at.get(c, current_dt))
        tmp_path = ACTIVE_WATCHLIST_TXT_PATH.with_suffix(".txt.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for code in active_sorted:
                f.write(f"{code},{watch_map.get(code, code)}\n")
        os.replace(tmp_path, ACTIVE_WATCHLIST_TXT_PATH)
    except Exception as exc:
        log(f"WARNING: active watchlist txt write failed: {exc}")


# ---------------------------------------------------------------------------
# Trading API
# ---------------------------------------------------------------------------

class TradingAPI:
    def __init__(self, env_dv: str | None = None, dry_run: bool = False, live_state: dict | None = None):
        self.env_dv = (env_dv or KIS_ENV_DV).strip() or "real"
        self.dry_run = bool(dry_run)
        self.live_state = live_state if live_state is not None else {"positions_meta": {}, "traded_today": set()}
        ka.auth()

        trenv = ka.getTREnv()
        self.cano = trenv.my_acct
        self.acnt_prdt_cd = trenv.my_prod
        self.positions: dict[str, dict] = {}
        self.pending_orders: dict[str, dict] = {}
        self.buy_inflight_codes: set[str] = set()
        self.startup_position_codes: set[str] = set()
        self.trade_lock_until: dict[str, datetime] = {}
        # 종목별 가장 최근 매도 체결가(_confirm_pending_sell 기록). 1차 익절 체결가를 급등 사다리 기준가로
        # 쓰기 위한 용도이며 메모리 전용(재시작 시 사다리 기준가는 live_state의 surge_ladder에서 복원).
        self.last_sell_fill_price: dict[str, float] = {}
        self._last_sync_at: datetime | None = None
        self._last_pending_poll_at: datetime | None = None
        self._last_live_state_save_at: datetime | None = None
        self.sync_positions_from_account(force=True)
        self._apply_persisted_position_meta()
        for code, pos in self.positions.items():
            if int(pos.get("quantity", 0) or 0) > 0:
                norm_code = str(code).zfill(6)
                self.buy_inflight_codes.add(norm_code)
                self.startup_position_codes.add(norm_code)
        log(
            f"TradingAPI (r76 MA5-BB multi-indicator) initialized | env_dv={self.env_dv} | dry_run={self.dry_run}"
        )

    def _apply_persisted_position_meta(self) -> None:
        meta_map = self.live_state.get("positions_meta") or {}
        for code, pos in self.positions.items():
            meta = meta_map.get(code) or {}
            if meta.get("buy_time") is not None:
                pos["buy_time"] = meta["buy_time"]
            if meta.get("entry_buy_time") is not None:
                pos["entry_buy_time"] = meta["entry_buy_time"]
            if meta.get("buy_session"):
                pos["buy_session"] = meta["buy_session"]
            if meta.get("highest_price") is not None:
                pos["highest_price"] = max(
                    float(pos.get("highest_price", 0.0)),
                    float(meta.get("highest_price", 0.0)),
                    float(pos.get("current_price", 0.0)),
                )
            pos["entry_quantity"] = int(meta.get("entry_quantity", 0) or pos.get("entry_quantity", 0) or pos.get("quantity", 0) or 0)
            pos["tp1_done"] = bool(meta.get("tp1_done", pos.get("tp1_done", False)))
            pos["tp2_done"] = bool(meta.get("tp2_done", pos.get("tp2_done", False)))
            pos["tp3_done"] = bool(meta.get("tp3_done", pos.get("tp3_done", False)))
            pos["pyramid_done"] = bool(meta.get("pyramid_done", pos.get("pyramid_done", False)))
            pos["surge_ladder"] = _sanitize_surge_ladder(meta.get("surge_ladder") or pos.get("surge_ladder"))

    def _record_position_meta(self, code: str, pos: dict) -> None:
        meta_map = self.live_state.setdefault("positions_meta", {})
        meta_map[str(code).zfill(6)] = {
            "buy_time": pos.get("entry_buy_time") or pos.get("buy_time"),
            "entry_buy_time": pos.get("entry_buy_time") or pos.get("buy_time"),
            "buy_session": pos.get("buy_session"),
            "highest_price": pos.get("highest_price"),
            "entry_quantity": int(pos.get("entry_quantity", 0) or 0),
            "tp1_done": bool(pos.get("tp1_done", False)),
            "tp2_done": bool(pos.get("tp2_done", False)),
            "tp3_done": bool(pos.get("tp3_done", False)),
            "pyramid_done": bool(pos.get("pyramid_done", False)),
            "surge_ladder": _sanitize_surge_ladder(pos.get("surge_ladder")),
        }

    def _sync_live_state_from_positions(self) -> None:
        for code, pos in self.positions.items():
            if int(pos.get("quantity", 0) or 0) > 0:
                self._record_position_meta(code, pos)

    def persist_live_state(self, date_str: str | None = None) -> None:
        self._sync_live_state_from_positions()
        save_live_state(self.live_state, date_str=date_str)
        self._last_live_state_save_at = datetime.now()

    def maybe_persist_live_state_interval(self, now: datetime, date_str: str) -> None:
        if self._last_live_state_save_at is None:
            self.persist_live_state(date_str=date_str)
            return
        elapsed = (now - self._last_live_state_save_at).total_seconds()
        if elapsed >= LIVE_STATE_SAVE_INTERVAL_SECONDS:
            self.persist_live_state(date_str=date_str)

    def sync_positions_from_account(self, force: bool = False) -> None:
        if not force and self._last_sync_at is not None:
            elapsed = (datetime.now() - self._last_sync_at).total_seconds()
            if elapsed < ACCOUNT_SYNC_INTERVAL_SECONDS:
                return

        balance_fn = getattr(dsf, "inquire_balance_rlz_pl", None)
        if balance_fn is None:
            return

        try:
            df_holdings, _ = balance_fn(
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                afhr_flpr_yn="N",
                inqr_dvsn="02",
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00",
            )
        except Exception as exc:
            log(f"WARNING: holdings sync failed: {exc}")
            return

        self._last_sync_at = datetime.now()

        if df_holdings is None or df_holdings.empty:
            if self.positions:
                log(
                    "WARNING: holdings API empty but local positions non-empty; keeping previous positions"
                )
            else:
                self.positions.clear()
            self._reconcile_pending_positions()
            return

        updated: dict[str, dict] = {}
        for _, row in df_holdings.iterrows():
            code = str(row.get("pdno", "")).zfill(6)
            try:
                qty = int(float(row.get("hldg_qty", 0)))
                avg_price = float(row.get("pchs_avg_pric", 0))
                current_price = float(row.get("prpr", avg_price))
            except (TypeError, ValueError):
                continue
            if qty <= 0 or avg_price <= 0:
                continue

            holding_name = str(row.get("prdt_name", "") or "").strip()
            if holding_name and code not in _SYMBOL_NAME_MAP:
                _SYMBOL_NAME_MAP[code] = holding_name

            prev = self.positions.get(code, {})
            persisted = (self.live_state.get("positions_meta") or {}).get(code) or {}
            buy_time = prev.get("buy_time") or persisted.get("buy_time")
            entry_buy_time = prev.get("entry_buy_time") or persisted.get("entry_buy_time") or buy_time
            if buy_time is None:
                buy_time = _parse_buy_time_from_holding_fields(row.to_dict() if hasattr(row, "to_dict") else dict(row))
            if entry_buy_time is None:
                entry_buy_time = buy_time
            buy_session = prev.get("buy_session") or persisted.get("buy_session") or "synced"
            highest = max(
                float(prev.get("highest_price", current_price)),
                float(persisted.get("highest_price", current_price) or current_price),
                current_price,
            )
            entry_quantity = int(prev.get("entry_quantity", 0) or persisted.get("entry_quantity", 0) or qty)
            updated[code] = {
                "buy_price": avg_price,
                "quantity": qty,
                "buy_time": buy_time,
                "entry_buy_time": entry_buy_time,
                "buy_session": buy_session,
                "current_price": current_price,
                "highest_price": highest,
                "entry_quantity": entry_quantity,
                "tp1_done": bool(prev.get("tp1_done", persisted.get("tp1_done", False))),
                "tp2_done": bool(prev.get("tp2_done", persisted.get("tp2_done", False))),
                "tp3_done": bool(prev.get("tp3_done", persisted.get("tp3_done", False))),
                "pyramid_done": bool(prev.get("pyramid_done", persisted.get("pyramid_done", False))),
                "surge_ladder": _sanitize_surge_ladder(prev.get("surge_ladder") or persisted.get("surge_ladder")),
            }
            self._record_position_meta(code, updated[code])

        self.positions = updated
        self._apply_persisted_position_meta()
        self._reconcile_pending_positions()

    def get_open_positions(self) -> dict[str, dict]:
        return self.positions

    def has_pending_order(self, code: str) -> bool:
        return str(code).zfill(6) in self.pending_orders

    def has_buy_exposure(self, code: str) -> bool:
        norm = str(code).zfill(6)
        if norm in self.buy_inflight_codes:
            return True
        if self.has_pending_order(norm):
            return True
        pos = self.positions.get(norm)
        if pos is not None and int(pos.get("quantity", 0) or 0) > 0:
            return True
        return False

    def get_pending_order(self, code: str) -> dict | None:
        return self.pending_orders.get(str(code).zfill(6))

    def _in_cooldown(self, code: str, now: datetime) -> bool:
        key = str(code).zfill(6)
        until = self.trade_lock_until.get(key)
        return until is not None and now < until

    def _mark_trade_lock(self, code: str, now: datetime) -> None:
        key = str(code).zfill(6)
        self.trade_lock_until[key] = now + timedelta(minutes=TRADE_COOLDOWN_MINUTES)

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            text = str(value).replace(",", "").strip()
            if text == "":
                return default
            return int(float(text))
        except Exception:
            return default

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            text = str(value).replace(",", "").strip()
            if text == "":
                return default
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _pending_side_to_ccld_code(side: str) -> str:
        return "02" if side == "buy" else "01"

    def _reconcile_pending_positions(self) -> None:
        for code, pending in self.pending_orders.items():
            if pending.get("side") != "buy":
                continue
            pos = self.positions.get(code)
            if pos is None:
                continue
            pos["buy_time"] = pending.get("submitted_at", pos.get("buy_time", datetime.now()))
            pos["buy_session"] = pending.get("session", pos.get("buy_session", "synced"))
            pos["highest_price"] = max(
                float(pos.get("highest_price", 0.0)),
                float(pos.get("current_price", 0.0)),
                float(pos.get("buy_price", 0.0)),
            )

    def _fetch_today_order_status(self, code: str, side: str, now: datetime, order_no: str = "") -> dict[str, object] | None:
        ccld_fn = getattr(dsf, "inquire_daily_ccld", None)
        if not callable(ccld_fn):
            return None

        date_str = now.strftime("%Y%m%d")
        try:
            df_orders, _ = ccld_fn(
                env_dv=self.env_dv,
                pd_dv="inner",
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                inqr_strt_dt=date_str,
                inqr_end_dt=date_str,
                sll_buy_dvsn_cd=self._pending_side_to_ccld_code(side),
                ccld_dvsn="00",
                inqr_dvsn="00",
                inqr_dvsn_3="00",
                pdno=code,
                odno=order_no,
                excg_id_dvsn_cd="ALL",
            )
        except Exception as exc:
            log(f"WARNING: order status query failed for {code}({side}): {exc}")
            return None

        if df_orders is None or df_orders.empty:
            return None

        orders = df_orders.copy()
        if "pdno" in orders.columns:
            orders = orders[orders["pdno"].astype(str).str.zfill(6) == code]
        if order_no and "odno" in orders.columns:
            matched = orders[orders["odno"].astype(str).str.strip() == order_no]
            if not matched.empty:
                orders = matched
        if orders.empty:
            return None

        sort_columns = [col for col in ("ord_dt", "ord_tmd", "odno") if col in orders.columns]
        if sort_columns:
            orders = orders.sort_values(sort_columns)

        row = orders.iloc[-1].to_dict()
        order_qty = self._to_int(row.get("ord_qty"), 0)
        filled_qty = self._to_int(row.get("tot_ccld_qty"), 0)
        remaining_qty = self._to_int(row.get("rmn_qty"), max(0, order_qty - filled_qty))
        rejected_qty = self._to_int(row.get("rjct_qty"), 0)
        avg_price = self._to_float(row.get("avg_prvs"), 0.0)
        cancel_yn = str(row.get("cncl_yn", "")).strip().upper()

        return {
            "order_no": str(row.get("odno", "")).strip(),
            "order_time": str(row.get("ord_tmd", "")).strip(),
            "order_qty": order_qty,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "rejected_qty": rejected_qty,
            "avg_price": avg_price,
            "cancel_yn": cancel_yn,
        }

    def _maybe_log_pending_progress(self, pending: dict, message: str, signature: str) -> None:
        if pending.get("last_status_signature") == signature:
            return
        pending["last_status_signature"] = signature
        log(message)

    def _confirm_pending_buy(self, code: str, pending: dict, pos: dict, status: dict | None) -> None:
        requested_qty = int(pending.get("quantity", 0))
        filled_qty = int(status.get("filled_qty", 0)) if status else int(pos.get("quantity", 0))
        avg_price = float(status.get("avg_price", 0.0)) if status else 0.0
        fill_price = avg_price if avg_price > 0 else float(pos.get("buy_price") or pos.get("current_price") or pending.get("requested_price", 0.0))

        if bool(pending.get("pyramid")):
            # 피라미딩(불타기) 추가매수 체결 확정: 기존 포지션의 평단가/보유시간/TP진행상태는
            # 그대로 두고, 브로커 계좌의 실제 가중평균 단가·수량(sync)만 신뢰해서 반영한다.
            # (직접 가중평균을 계산하면 sync 타이밍에 따라 이중계산될 위험이 있어 회피)
            add_qty = filled_qty if filled_qty > 0 else requested_qty
            pos["entry_quantity"] = int(pos.get("entry_quantity", 0) or 0) + add_qty
            self.sync_positions_from_account(force=True)
            pos = self.positions.get(code) or pos
            if fill_price > 0:
                pos["highest_price"] = max(
                    float(pos.get("highest_price", fill_price)),
                    float(pos.get("current_price", fill_price)),
                    fill_price,
                )
            code_name = str(pending.get("code_name", ""))
            compact_code_label = f"{code}_{code_name}" if code_name else code
            buy_detail_text = str(pending.get("buy_detail", "")).strip()
            title = f">>> {compact_code_label} Pyramid Buy - {buy_detail_text}" if buy_detail_text else f">>> {compact_code_label} Pyramid Buy"
            _log_trade_block(
                [
                    "=" * 110,
                    title,
                    f"*** 추가매수수량={add_qty} 추가매수가={fill_price:,.0f} | "
                    f"신규평단가={float(pos.get('buy_price', 0.0)):,.0f} 신규수량={int(pos.get('quantity', 0))}",
                ],
                event_time=datetime.now(),
                mirror_main_log=False,
            )
            self._record_position_meta(code, pos)
            self.persist_live_state()
            self.pending_orders.pop(str(code).zfill(6), None)
            return

        entry_time = pending.get("submitted_at", pos.get("entry_buy_time", pos.get("buy_time", datetime.now())))
        pos["buy_time"] = entry_time
        pos["entry_buy_time"] = entry_time
        pos["buy_session"] = pending.get("session", pos.get("buy_session", "synced"))
        pos["tp1_done"] = False
        pos["tp2_done"] = False
        pos["tp3_done"] = False
        pos["surge_ladder"] = None
        pos["entry_quantity"] = max(int(pos.get("entry_quantity", 0) or 0), filled_qty)
        if fill_price > 0:
            pos["buy_price"] = fill_price
            pos["current_price"] = float(pos.get("current_price") or fill_price)
            pos["highest_price"] = max(float(pos.get("highest_price", fill_price)), float(pos["current_price"]), fill_price)

        code_name = str(pending.get("code_name", ""))
        compact_code_label = f"{code}_{code_name}" if code_name else code
        event_time = datetime.now()
        buy_time_label = _format_trade_time_label(pending.get("submitted_at"))
        buy_price_label = f"{fill_price:,.0f}" if fill_price > 0 else "N/A"
        buy_detail_text = str(pending.get("buy_detail", "")).strip()
        summary_line = f">>> {compact_code_label} Buy - {buy_detail_text}" if buy_detail_text else f">>> {compact_code_label} Buy"
        _log_trade_block(
            [
                "=" * 110,
                summary_line,
                f"*** 매수시각={buy_time_label} | 매수가격={buy_price_label} | 매수수량={filled_qty}",
            ],
            event_time=event_time,
            mirror_main_log=False,
        )
        self._record_position_meta(code, pos)
        self.startup_position_codes.discard(str(code).zfill(6))
        self.persist_live_state()
        self.pending_orders.pop(str(code).zfill(6), None)

    def _confirm_pending_sell(self, code: str, pending: dict, status: dict | None, remaining_qty: int) -> None:
        requested_qty = int(pending.get("quantity", 0))
        filled_qty = int(status.get("filled_qty", requested_qty)) if status else requested_qty
        avg_price = float(status.get("avg_price", 0.0)) if status else 0.0
        fill_price = avg_price if avg_price > 0 else float(pending.get("requested_price", 0.0))
        if fill_price > 0:
            self.last_sell_fill_price[str(code).zfill(6)] = fill_price
        buy_price = float(pending.get("buy_price", 0.0))
        pnl_pct = ((fill_price / buy_price) - 1.0) * 100.0 if buy_price > 0 and fill_price > 0 else float("nan")
        profit_amount = (fill_price - buy_price) * filled_qty if buy_price > 0 and fill_price > 0 and filled_qty > 0 else float("nan")
        event_time = datetime.now()
        buy_time_label = _format_trade_time_label(pending.get("buy_time"))
        sell_time_label = _format_trade_time_label(event_time)
        buy_price_label = f"{buy_price:,.0f}" if buy_price > 0 else "N/A"
        fill_price_label = f"{fill_price:,.0f}" if fill_price > 0 else "N/A"
        profit_amount_label = f"{profit_amount:,.0f}원" if pd.notna(profit_amount) else "N/A"
        pnl_pct_label = f"{pnl_pct:.2f}%" if pd.notna(pnl_pct) else "N/A"
        code_name = str(pending.get("code_name", "")).strip() or _SYMBOL_NAME_MAP.get(str(code).zfill(6), "")
        code_label = _format_code_label(code, code_name)
        compact_code_label = f"{code}_{code_name}" if code_name else code
        buy_qty = int(pending.get("pre_submit_qty", filled_qty))
        reason_text = str(pending.get("reason", "UNKNOWN"))
        exch_text = str(pending.get("exchange", "UNKNOWN"))
        log(
            f"{code_label} SELL executed  | filled_qty={filled_qty}/{requested_qty} | "
            f"price={fill_price:,.0f} | remaining={remaining_qty} | pnl={pnl_pct:.2f}% | "
            f"reason={reason_text} | exch={exch_text}"
        )
        _log_trade_block(
            [
                "=" * 110,
                f">>> {compact_code_label} Sell - 수익금액={profit_amount_label} | 수익율={pnl_pct_label} | reason={reason_text} exch={exch_text}",
                f"*** 매수시각={buy_time_label} | 매수가격={buy_price_label} | 매수수량={buy_qty}",
                f"*** 매도시각={sell_time_label} | 매도가격={fill_price_label} | 매도수량={filled_qty}",
            ],
            event_time=event_time,
            mirror_main_log=False,
        )
        if remaining_qty <= 0:
            self.buy_inflight_codes.discard(str(code).zfill(6))
            self.startup_position_codes.discard(str(code).zfill(6))
            self.positions.pop(code, None)
            (self.live_state.get("positions_meta") or {}).pop(str(code).zfill(6), None)
            traded = self.live_state.setdefault("traded_today", set())
            traded.discard(str(code).zfill(6))
        self.persist_live_state()
        self.pending_orders.pop(str(code).zfill(6), None)

    def refresh_pending_orders(self, now: datetime) -> None:
        if not self.pending_orders:
            return
        if self._last_pending_poll_at is not None:
            elapsed = (now - self._last_pending_poll_at).total_seconds()
            if elapsed < ORDER_STATUS_POLL_INTERVAL_SECONDS:
                return

        self._last_pending_poll_at = now
        self.sync_positions_from_account(force=False)

        for code, pending in list(self.pending_orders.items()):
            next_poll_at = pending.get("next_status_poll_at")
            if isinstance(next_poll_at, datetime) and now < next_poll_at:
                continue

            side = str(pending.get("side", ""))

            # [2026-09-18] 시장가+지정가 분할 매수(place_buy_order 참조)는 leg마다
            # 별도 order_no를 가지므로 아래 단일-주문 전용 로직(order_no 1개 기준)을
            # 타지 않고 전용 처리기로 위임한다.
            if side == "buy" and pending.get("legs"):
                self._refresh_split_buy_legs(code, pending, now)
                continue

            pos = self.positions.get(code)
            status = self._fetch_today_order_status(code, side, now, str(pending.get("order_no", "")))

            if status is None:
                fail_count = int(pending.get("status_fail_count", 0)) + 1
                pending["status_fail_count"] = fail_count
                pending["next_status_poll_at"] = now + timedelta(seconds=_pending_status_backoff_seconds(fail_count))
            else:
                pending["status_fail_count"] = 0
                pending.pop("next_status_poll_at", None)

            if side == "buy":
                if bool(pending.get("pyramid")):
                    # 피라미딩 추가매수는 이미 보유 중(quantity>0)이라 기존 "pos.quantity>0 = 체결됨"
                    # 휴리스틱을 쓸 수 없다 - 주문 상태(status)의 체결/잔량으로 직접 종결 여부를 판정한다.
                    if status is not None:
                        filled_qty = int(status.get("filled_qty", 0))
                        remaining_qty = int(status.get("remaining_qty", 0))
                        rejected_qty = int(status.get("rejected_qty", 0))
                        order_qty = int(status.get("order_qty", pending.get("quantity", 0)))
                        cancel_yn = str(status.get("cancel_yn", ""))
                        terminal = remaining_qty <= 0 or cancel_yn == "Y" or rejected_qty >= order_qty
                        if filled_qty > 0 and terminal:
                            if pos is None:
                                log(f"  [PYRAMID BUY WARN] {code} | 체결되었으나 기존 포지션을 찾을 수 없음 - 확정 보류")
                            else:
                                self._confirm_pending_buy(code, pending, pos, status)
                            continue
                        if filled_qty <= 0 and terminal:
                            self._maybe_log_pending_progress(
                                pending,
                                f"PYRAMID BUY closed without fill | {code} | order_no={pending.get('order_no', '')}",
                                "pyramid_buy_closed_without_fill",
                            )
                            self.pending_orders.pop(str(code).zfill(6), None)
                            continue
                        submitted_at = pending.get("submitted_at")
                        if (
                            isinstance(submitted_at, datetime)
                            and filled_qty <= 0
                            and (now - submitted_at).total_seconds() >= BUY_ORDER_STALE_WARN_SECONDS
                        ):
                            log(
                                f"  [PYRAMID BUY STALE] {code} | {int((now - submitted_at).total_seconds())}s 미체결 대기중 | "
                                f"order_no={pending.get('order_no', '')}"
                            )
                        self._maybe_log_pending_progress(
                            pending,
                            f"PYRAMID BUY pending | {code} | filled={filled_qty}/{order_qty} | remaining={remaining_qty}",
                            f"pyramid_buy_pending:{filled_qty}:{remaining_qty}",
                        )
                    continue

                if pos is not None and pos.get("quantity", 0) > 0:
                    if status and int(status.get("remaining_qty", 0)) > 0:
                        self._maybe_log_pending_progress(
                            pending,
                            f"BUY partial | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                            f"| remaining={int(status.get('remaining_qty', 0))} | avg={float(status.get('avg_price', 0.0)):,.0f}",
                            f"buy_partial:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}:{float(status.get('avg_price', 0.0))}",
                        )
                        continue
                    self._confirm_pending_buy(code, pending, pos, status)
                    continue

                if status is not None:
                    if int(status.get("filled_qty", 0)) <= 0 and (
                        int(status.get("remaining_qty", 0)) <= 0
                        or str(status.get("cancel_yn", "")) == "Y"
                        or int(status.get("rejected_qty", 0)) >= int(status.get("order_qty", pending.get("quantity", 0)))
                    ):
                        # 우리가 _request_buy_reprice로 취소를 요청한 경우(cancel_inflight)는
                        # 원래 의도한 종결(취소 확인)이므로 PENDING_BUY_GRACE_SECONDS를 기다리지
                        # 않고 바로 처리한다 - 그 유예는 브로커 쪽에서 예상 못한 거부/취소를
                        # 상태 API 지연과 혼동하지 않기 위한 것으로, 우리가 건 취소에는 해당 없음.
                        if pending.get("cancel_inflight"):
                            if pending.get("reprice_pending"):
                                self._resubmit_repriced_buy_order(code, pending, now, is_nxt_tradeable(code))
                            else:
                                self._maybe_log_pending_progress(
                                    pending,
                                    f"BUY reprice giveup confirmed | {code} | order_no={pending.get('order_no', '')}",
                                    "buy_reprice_giveup_confirmed",
                                )
                                self.buy_inflight_codes.discard(str(code).zfill(6))
                                self.pending_orders.pop(str(code).zfill(6), None)
                            continue

                        submitted_at = pending.get("submitted_at")
                        if isinstance(submitted_at, datetime):
                            if (now - submitted_at).total_seconds() < PENDING_BUY_GRACE_SECONDS:
                                continue
                        self._maybe_log_pending_progress(
                            pending,
                            f"BUY closed without fill | {code} | order_no={pending.get('order_no', '')} | exch={pending.get('exchange', 'UNKNOWN')}",
                            "buy_closed_without_fill",
                        )
                        self.buy_inflight_codes.discard(str(code).zfill(6))
                        self.pending_orders.pop(str(code).zfill(6), None)
                        continue

                    # 미체결 장기 대기 - BUY_ORDER_REPRICE_AFTER_SECONDS 지나면 취소 후
                    # 더 공격적인 가격으로 재주문 시도(cancel_inflight 아닐 때만 - 이미 취소
                    # 요청이 진행중이면 확인될 때까지 중복 취소하지 않음). 재시도 소진 후에는
                    # 기존과 동일하게 경고만 남긴다.
                    _sub_at = pending.get("submitted_at")
                    if (
                        isinstance(_sub_at, datetime)
                        and int(status.get("filled_qty", 0)) <= 0
                        and not pending.get("cancel_inflight")
                        and int(pending.get("reprice_attempt", 0)) < BUY_ORDER_REPRICE_MAX_ATTEMPTS
                        and (now - _sub_at).total_seconds() >= BUY_ORDER_REPRICE_AFTER_SECONDS
                    ):
                        self._request_buy_reprice(code, pending, now)
                    elif (
                        isinstance(_sub_at, datetime)
                        and int(status.get("filled_qty", 0)) <= 0
                        and (now - _sub_at).total_seconds() >= BUY_ORDER_STALE_WARN_SECONDS
                    ):
                        _age = int((now - _sub_at).total_seconds())
                        log(
                            f"  [BUY STALE] {code} | {_age}s 미체결 대기중 "
                            f"| order_no={pending.get('order_no', '')} | 시장 변화로 체결 불가 가능성 높음"
                        )
                    self._maybe_log_pending_progress(
                        pending,
                        f"BUY pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                        f"| remaining={int(status.get('remaining_qty', 0))} | order_no={status.get('order_no', '')}",
                        f"buy_pending:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}",
                    )
                    log_trade(
                        f"BUY pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                        f"| remaining={int(status.get('remaining_qty', 0))} | order_no={status.get('order_no', '')}"
                    )
                continue

            pre_submit_qty = int(pending.get("pre_submit_qty", pending.get("quantity", 0)))
            current_qty = int(pos.get("quantity", 0)) if pos is not None else 0
            expected_remaining_qty = max(0, pre_submit_qty - int(pending.get("quantity", 0)))

            if status is not None:
                filled_qty = int(status.get("filled_qty", 0))
                remaining_qty = int(status.get("remaining_qty", 0))
                rejected_qty = int(status.get("rejected_qty", 0))
                order_qty = int(status.get("order_qty", pending.get("quantity", 0)))
                cancel_yn = str(status.get("cancel_yn", ""))
                terminal = remaining_qty <= 0 or cancel_yn == "Y" or rejected_qty >= order_qty
                if filled_qty > 0 and terminal:
                    self._confirm_pending_sell(code, pending, status, remaining_qty)
                    continue

            if current_qty <= expected_remaining_qty:
                self._confirm_pending_sell(code, pending, status, current_qty)
                continue

            if status is not None:
                if int(status.get("filled_qty", 0)) <= 0 and (
                    int(status.get("remaining_qty", 0)) <= 0
                    or str(status.get("cancel_yn", "")) == "Y"
                    or int(status.get("rejected_qty", 0)) >= int(status.get("order_qty", pending.get("quantity", 0)))
                ):
                    self._maybe_log_pending_progress(
                        pending,
                        f"SELL closed without fill | {code} | reason={pending.get('reason', 'UNKNOWN')} | order_no={pending.get('order_no', '')}",
                        "sell_closed_without_fill",
                    )
                    self.pending_orders.pop(code, None)
                    continue

                self._maybe_log_pending_progress(
                    pending,
                    f"SELL pending | {code} | filled={int(status.get('filled_qty', 0))}/{int(status.get('order_qty', pending.get('quantity', 0)))} "
                    f"| remaining={int(status.get('remaining_qty', 0))} | current_qty={current_qty} | reason={pending.get('reason', 'UNKNOWN')}",
                    f"sell_pending:{int(status.get('filled_qty', 0))}:{int(status.get('remaining_qty', 0))}:{current_qty}",
                )

    def get_affordable_buy_qty(self, code: str, price: float, now: datetime, nxt_tradeable: bool) -> int:
        price_krw = max(1, int(round(price)))
        qty_by_budget = int(MAX_ORDER_AMOUNT_KRW / price_krw)
        if qty_by_budget <= 0:
            return 0

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return 0

        psbl_fn = getattr(dsf, "inquire_psbl_order", None)
        if not callable(psbl_fn):
            return qty_by_budget

        try:
            psbl_df = psbl_fn(
                env_dv=self.env_dv,
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                pdno=code,
                ord_unpr=str(price_krw),
                ord_dvsn=order_spec["ord_dvsn"],
                cma_evlu_amt_icld_yn="N",
                ovrs_icld_yn="N",
            )
        except Exception as exc:
            log(f"WARNING: inquire_psbl_order failed for {code}: {exc}")
            return qty_by_budget

        if psbl_df is None or psbl_df.empty:
            return qty_by_budget

        row = psbl_df.iloc[-1].to_dict()
        qty_candidates = ("nrcvb_buy_qty", "max_buy_qty", "ord_psbl_qty")
        qty_by_psbl = 0
        for key in qty_candidates:
            qty_by_psbl = max(qty_by_psbl, self._to_int(row.get(key), 0))

        # 응답 포맷 차이를 대비해 유사 컬럼명도 허용
        if qty_by_psbl <= 0:
            for key, value in row.items():
                key_text = str(key).lower()
                if "qty" in key_text and ("buy" in key_text or "psbl" in key_text or "ord" in key_text):
                    qty_by_psbl = max(qty_by_psbl, self._to_int(value, 0))

        if qty_by_psbl <= 0:
            amt_candidates = ("nrcvb_buy_amt", "max_buy_amt", "ord_psbl_cash", "ord_psbl_amt")
            psbl_amt = 0
            for key in amt_candidates:
                psbl_amt = max(psbl_amt, self._to_int(row.get(key), 0))

            if psbl_amt <= 0:
                for key, value in row.items():
                    key_text = str(key).lower()
                    if "amt" in key_text or "cash" in key_text:
                        if "buy" in key_text or "psbl" in key_text or "ord" in key_text:
                            psbl_amt = max(psbl_amt, self._to_int(value, 0))

            if psbl_amt > 0:
                qty_by_psbl = int(psbl_amt / price_krw)

        if qty_by_psbl <= 0:
            # 주문가 응답이 예상과 다르거나 수량/금액 해석이 안되면 과주문 방지 위해 보수적으로 0 처리
            log(f"WARNING: psbl-order parse failed for {code}; force qty=0 to avoid over-order")
            return 0

        return max(0, min(qty_by_budget, qty_by_psbl))

    def _refresh_split_buy_legs(self, code: str, pending: dict, now: datetime) -> None:
        """시장가+지정가로 분할 제출된 신규 진입 매수(place_buy_order 참조)의 leg별
        체결/재주문을 처리한다. 각 leg는 order_no가 서로 달라 독립적으로 상태를
        조회해야 하므로, 단일 order_no를 가정하는 refresh_pending_orders의 일반
        buy 경로 대신 이 전용 처리기를 탄다. 두 leg가 모두 종결(체결/취소/거부)되면
        합산 결과로 _confirm_pending_buy를 한 번만 호출한다."""
        legs = pending.get("legs") or []
        nxt_tradeable_cache: bool | None = None

        for leg in legs:
            if leg.get("done"):
                continue
            order_no = str(leg.get("order_no", ""))
            if not order_no:
                leg["done"] = True
                continue

            status = self._fetch_today_order_status(code, "buy", now, order_no)
            if status is None:
                continue  # 다음 폴링에서 재시도

            leg["last_status"] = status
            filled_qty = int(status.get("filled_qty", 0))
            remaining_qty = int(status.get("remaining_qty", 0))
            rejected_qty = int(status.get("rejected_qty", 0))
            order_qty = int(status.get("order_qty", leg.get("quantity", 0)))
            cancel_yn = str(status.get("cancel_yn", ""))
            terminal = remaining_qty <= 0 or cancel_yn == "Y" or rejected_qty >= order_qty

            if leg.get("cancel_inflight"):
                # 우리가 _request_leg_reprice로 취소를 요청한 leg - 확인되는 대로
                # 재주문하거나(reprice_pending) 포기(giveup)하고 leg를 종결시킨다.
                if terminal:
                    if leg.get("reprice_pending"):
                        if nxt_tradeable_cache is None:
                            nxt_tradeable_cache = is_nxt_tradeable(code)
                        self._resubmit_leg(code, leg, now, nxt_tradeable_cache)
                    else:
                        leg["done"] = True
                continue

            if terminal:
                if (
                    filled_qty <= 0
                    and not leg.get("is_market_order")
                    and isinstance(leg.get("submitted_at"), datetime)
                    and (now - leg["submitted_at"]).total_seconds() < PENDING_BUY_GRACE_SECONDS
                ):
                    continue  # 브로커 상태 API 지연 가능성 - grace 기간 내엔 종결 처리 보류
                leg["done"] = True
                self._maybe_log_pending_progress(
                    leg,
                    f"  BUY leg closed | {code} | role={leg.get('role')} | filled={filled_qty}/{order_qty}",
                    f"leg_closed:{filled_qty}:{order_qty}",
                )
                continue

            if leg.get("is_market_order"):
                # 시장가는 재주문 대상이 아님(이미 최우선 가격) - 정상적으로는 즉시
                # 체결/거부로 terminal이 되므로, 오래 pending이면 경고만 남긴다.
                submitted_at = leg.get("submitted_at")
                if (
                    isinstance(submitted_at, datetime)
                    and (now - submitted_at).total_seconds() >= BUY_ORDER_STALE_WARN_SECONDS
                ):
                    log(
                        f"  [BUY LEG STALE] {code} | role=market | "
                        f"{int((now - submitted_at).total_seconds())}s 미체결 대기중 | 시장가인데 미체결 - 확인 필요"
                    )
                self._maybe_log_pending_progress(
                    leg,
                    f"  BUY leg pending | {code} | role=market | filled={filled_qty}/{order_qty}",
                    f"leg_pending:{filled_qty}:{order_qty}",
                )
                continue

            submitted_at = leg.get("submitted_at")
            if (
                isinstance(submitted_at, datetime)
                and filled_qty <= 0
                and int(leg.get("reprice_attempt", 0)) < BUY_ORDER_REPRICE_MAX_ATTEMPTS
                and (now - submitted_at).total_seconds() >= BUY_ORDER_REPRICE_AFTER_SECONDS
            ):
                self._request_leg_reprice(code, leg, now)
            else:
                self._maybe_log_pending_progress(
                    leg,
                    f"  BUY leg pending | {code} | role={leg.get('role')} | filled={filled_qty}/{order_qty}",
                    f"leg_pending:{filled_qty}:{order_qty}",
                )

        if not all(leg.get("done") for leg in legs):
            return

        total_requested = sum(int(leg.get("quantity", 0)) for leg in legs)
        total_filled = sum(int((leg.get("last_status") or {}).get("filled_qty", 0)) for leg in legs)
        norm_code = str(code).zfill(6)

        if total_filled <= 0:
            self._maybe_log_pending_progress(
                pending,
                f"BUY closed without fill | {code} | 분할매수 두 leg 모두 미체결로 종결",
                "split_buy_closed_without_fill",
            )
            self.buy_inflight_codes.discard(norm_code)
            self.pending_orders.pop(norm_code, None)
            return

        pos = self.positions.get(code)
        if pos is None:
            return  # 계좌 동기화 반영 전 - 다음 폴링에서 재시도

        fill_amount = sum(
            int((leg.get("last_status") or {}).get("filled_qty", 0))
            * float((leg.get("last_status") or {}).get("avg_price", 0.0) or leg.get("requested_price", 0.0))
            for leg in legs
        )
        avg_price = fill_amount / total_filled if total_filled > 0 else 0.0
        pending["quantity"] = total_requested
        self._confirm_pending_buy(code, pending, pos, {"filled_qty": total_filled, "avg_price": avg_price})

    def _request_leg_reprice(self, code: str, leg: dict, now: datetime) -> None:
        """분할매수 지정가 leg의 미체결 재주문 요청. 취소만 하고, 확인되면(다음 폴링에서
        _refresh_split_buy_legs가 처리) reprice_pending 여부에 따라 재주문하거나 포기한다.
        단일주문 경로의 _request_buy_reprice와 동일한 정책(추격상한/bid->ask->market
        단계)을 leg 단위로 독립 적용한다."""
        norm_code = str(code).zfill(6)
        order_no = str(leg.get("order_no", ""))
        order_org_no = str(leg.get("order_org_no", ""))
        if not order_no or not order_org_no:
            return

        attempt = int(leg.get("reprice_attempt", 0)) + 1
        exchange = str(leg.get("exchange", "KRX"))
        market_div = "NX" if exchange == "NXT" else "J"
        reference_price = float(leg.get("entry_reference_price") or leg.get("requested_price") or 0.0)
        is_final_attempt = attempt >= BUY_ORDER_REPRICE_MAX_ATTEMPTS

        bid_price, ask_price = _fetch_bid_ask_price(norm_code, market_div)
        candidate = ask_price if attempt >= 2 else bid_price
        candidate = candidate or bid_price or ask_price
        if not candidate or candidate <= 0:
            log(f"  [BUY REPRICE SKIP] {code} | leg={leg.get('role')} | 호가 조회 실패 - 다음 폴링에서 재시도")
            return

        give_up = False
        if reference_price > 0:
            max_price = reference_price * (1 + BUY_ORDER_REPRICE_MAX_CHASE_PCT / 100.0)
            if candidate > max_price:
                log(
                    f"  [BUY REPRICE ABANDON] {code} | leg={leg.get('role')} | candidate={candidate:,.0f} > "
                    f"max_chase={max_price:,.0f}(ref={reference_price:,.0f}+{BUY_ORDER_REPRICE_MAX_CHASE_PCT:.1f}%) "
                    f"| 추격 포기, 취소만 진행"
                )
                give_up = True

        if is_final_attempt and exchange != "NXT":
            next_price_mode = "market"
        else:
            next_price_mode = "ask" if attempt >= 2 else "bid"

        try:
            cancel_result = dsf.order_rvsecncl(
                env_dv=self.env_dv,
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                krx_fwdg_ord_orgno=order_org_no,
                orgn_odno=order_no,
                ord_dvsn="00",
                rvse_cncl_dvsn_cd="02",
                ord_qty=str(int(leg.get("quantity", 0))),
                ord_unpr="0",
                qty_all_ord_yn="Y",
                excg_id_dvsn_cd=exchange,
            )
        except Exception as exc:
            log(f"  [BUY REPRICE CANCEL ERROR] {code} | leg={leg.get('role')} | {exc}")
            return

        if not _order_succeeded(cancel_result):
            log(f"  [BUY REPRICE CANCEL FAILED] {code} | leg={leg.get('role')} | {_extract_order_error_detail(cancel_result)}")
            return

        leg["reprice_attempt"] = attempt
        leg["cancel_inflight"] = True
        if give_up:
            leg["reprice_pending"] = False
            log(f"  [BUY REPRICE GIVEUP] {code} | leg={leg.get('role')} | attempt={attempt} | 취소 요청 완료, 재주문 없이 포기")
        else:
            leg["reprice_pending"] = True
            leg["reprice_next_mode"] = next_price_mode
            log(
                f"  [BUY REPRICE CANCEL] {code} | leg={leg.get('role')} | attempt={attempt}/{BUY_ORDER_REPRICE_MAX_ATTEMPTS} | "
                f"next_mode={next_price_mode} | 취소 요청 완료, 확인되는 대로 재주문"
            )

    def _resubmit_leg(self, code: str, leg: dict, now: datetime, nxt_tradeable: bool) -> None:
        """_request_leg_reprice가 요청한 취소가 확인된 뒤 새 가격/방식으로 leg를 재주문한다.
        단일주문 경로의 _resubmit_repriced_buy_order와 동일하되, 실패해도 pending_orders
        전체를 정리하지 않고 이 leg만 done 처리한다(다른 leg는 독립적으로 계속 진행)."""
        norm_code = str(code).zfill(6)
        price_mode = str(leg.get("reprice_next_mode", "bid"))
        qty = int(leg.get("quantity", 0))
        exchange = str(leg.get("exchange", "KRX"))
        market_div = "NX" if exchange == "NXT" else "J"
        reference_price = float(leg.get("entry_reference_price") or 0.0)
        attempt = int(leg.get("reprice_attempt", 0))

        affordable_qty = self.get_affordable_buy_qty(code, reference_price or 1.0, now, nxt_tradeable)
        qty = min(qty, int(affordable_qty))
        if qty <= 0:
            log(f"  [BUY REPRICE ABORT] {code} | leg={leg.get('role')} | 재주문 여력 부족 - 포기")
            leg["done"] = True
            return

        if price_mode == "market" and exchange != "NXT":
            ord_dvsn, ord_unpr, price_for_log = "01", "0", None
        else:
            bid_price, ask_price = _fetch_bid_ask_price(norm_code, market_div)
            candidate = ask_price if price_mode == "ask" else bid_price
            candidate = candidate or bid_price or ask_price
            if not candidate or candidate <= 0:
                log(f"  [BUY REPRICE SKIP] {code} | leg={leg.get('role')} | 호가 조회 실패 - 다음 폴링에서 재시도")
                return  # leg 그대로 유지(reprice_pending=True) -> 다음 폴링에서 재시도
            ord_dvsn, ord_unpr, price_for_log = "00", str(int(round(candidate))), candidate

        order_result = self._submit_buy_order_cash(code, qty, ord_dvsn, ord_unpr, exchange, price_for_log or 0.0)
        if order_result is None or not _order_succeeded(order_result):
            detail = _extract_order_error_detail(order_result) if order_result is not None else "SUBMIT_EXCEPTION"
            log(f"  [BUY REPRICE RESUBMIT FAILED] {code} | leg={leg.get('role')} | {detail}")
            leg["done"] = True
            return

        new_price = float(_extract_order_price(order_result) or price_for_log or reference_price)
        leg.update({
            "quantity": qty,
            "submitted_at": now,
            "requested_price": new_price,
            "order_no": _extract_order_number(order_result),
            "order_org_no": _extract_order_org_no(order_result),
            "order_time": _extract_order_time(order_result),
            "cancel_inflight": False,
            "reprice_pending": False,
        })
        leg.pop("last_status_signature", None)
        leg.pop("last_status", None)
        log(
            f"  [BUY REPRICE RESUBMIT] {code} | leg={leg.get('role')} | attempt={attempt}/{BUY_ORDER_REPRICE_MAX_ATTEMPTS} | "
            f"mode={price_mode} | price={new_price:,.0f} | order_no={leg['order_no'] or 'UNKNOWN'}"
        )
        log_trade(
            f"{_symbol_log_label(code, str(leg.get('code_name', '')))} | BUY REPRICE RESUBMIT | qty={qty} | "
            f"leg={leg.get('role')} | mode={price_mode} | price={new_price:,.0f} | order_no={leg['order_no'] or 'UNKNOWN'}"
        )

    def _request_buy_reprice(self, code: str, pending: dict, now: datetime) -> None:
        """미체결 매수 지정가 주문을 취소 요청한다. 취소가 확인되면(closed without fill,
        refresh_pending_orders에서 처리) reprice_pending 여부에 따라 새 가격으로 재주문하거나
        포기한다. BUY_ORDER_REPRICE_MAX_ATTEMPTS번째 시도이거나 다음 후보가(candidate)가
        entry_reference_price(최초 신호가) 대비 BUY_ORDER_REPRICE_MAX_CHASE_PCT를 넘으면
        추격을 포기(취소만 하고 재주문 안 함) - r002 추격매수 방지 게이트와 같은 철학.
        (r001 Update log 2026-08-28 참조, 000720 현대건설 13:14 미체결 사례로 발견)
        """
        norm_code = str(code).zfill(6)
        order_no = str(pending.get("order_no", ""))
        order_org_no = str(pending.get("order_org_no", ""))
        if not order_no or not order_org_no:
            return  # 취소에 필요한 원주문 식별자를 확보하지 못했으면 재시도 불가 - 그대로 대기

        attempt = int(pending.get("reprice_attempt", 0)) + 1
        exchange = str(pending.get("exchange", "KRX"))
        market_div = "NX" if exchange == "NXT" else "J"
        reference_price = float(pending.get("entry_reference_price") or pending.get("requested_price") or 0.0)
        is_final_attempt = attempt >= BUY_ORDER_REPRICE_MAX_ATTEMPTS

        # 시장가(마지막 시도)도 체결 예상가는 결국 매도1호가 근방이므로, 추격 상한 검사는
        # 주문유형과 무관하게 항상 현재 호가로 수행한다 - 그렇지 않으면 마지막 시도에서만
        # 상한 검사를 건너뛰게 되어 가장 위험한 시장가 주문이 오히려 무방비로 추격매수를
        # 해버리는 모순이 생긴다(추격 상한을 두는 취지 자체가 무의미해짐).
        bid_price, ask_price = _fetch_bid_ask_price(norm_code, market_div)
        candidate = ask_price if attempt >= 2 else bid_price
        candidate = candidate or bid_price or ask_price
        if not candidate or candidate <= 0:
            log(f"  [BUY REPRICE SKIP] {code} | 호가 조회 실패 - 다음 폴링에서 재시도")
            return

        give_up = False
        if reference_price > 0:
            max_price = reference_price * (1 + BUY_ORDER_REPRICE_MAX_CHASE_PCT / 100.0)
            if candidate > max_price:
                log(
                    f"  [BUY REPRICE ABANDON] {code} | candidate={candidate:,.0f} > "
                    f"max_chase={max_price:,.0f}(ref={reference_price:,.0f}+{BUY_ORDER_REPRICE_MAX_CHASE_PCT:.1f}%) "
                    f"| 추격 포기, 취소만 진행"
                )
                give_up = True

        if is_final_attempt and exchange != "NXT":
            next_price_mode = "market"
        else:
            next_price_mode = "ask" if attempt >= 2 else "bid"

        try:
            cancel_result = dsf.order_rvsecncl(
                env_dv=self.env_dv,
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                krx_fwdg_ord_orgno=order_org_no,
                orgn_odno=order_no,
                ord_dvsn="00",
                rvse_cncl_dvsn_cd="02",
                ord_qty=str(int(pending.get("quantity", 0))),
                ord_unpr="0",
                qty_all_ord_yn="Y",
                excg_id_dvsn_cd=exchange,
            )
        except Exception as exc:
            log(f"  [BUY REPRICE CANCEL ERROR] {code} | {exc}")
            return

        if not _order_succeeded(cancel_result):
            log(f"  [BUY REPRICE CANCEL FAILED] {code} | {_extract_order_error_detail(cancel_result)}")
            return

        pending["reprice_attempt"] = attempt
        pending["cancel_inflight"] = True
        if give_up:
            pending["reprice_pending"] = False
            log(f"  [BUY REPRICE GIVEUP] {code} | attempt={attempt} | 취소 요청 완료, 재주문 없이 포기")
        else:
            pending["reprice_pending"] = True
            pending["reprice_next_mode"] = next_price_mode
            log(
                f"  [BUY REPRICE CANCEL] {code} | attempt={attempt}/{BUY_ORDER_REPRICE_MAX_ATTEMPTS} | "
                f"next_mode={next_price_mode} | 취소 요청 완료, 확인되는 대로 재주문"
            )

    def _resubmit_repriced_buy_order(self, code: str, pending: dict, now: datetime, nxt_tradeable: bool) -> None:
        """_request_buy_reprice가 요청한 취소가 확인된 뒤 새 가격/방식으로 재주문한다.
        place_buy_order()는 has_buy_exposure/쿨다운 검사가 있어 같은 진입 시도의 재주문에는
        쓸 수 없으므로(자기 자신의 pending 항목·방금 세팅한 trade_lock에 막힘) 그 검사들을
        건너뛰는 전용 경로. reference_price(최초 신호가) 등 pending에 보존된 컨텍스트를
        그대로 이어받는다."""
        norm_code = str(code).zfill(6)
        price_mode = str(pending.get("reprice_next_mode", "bid"))
        qty = int(pending.get("quantity", 0))
        exchange = str(pending.get("exchange", "KRX"))
        market_div = "NX" if exchange == "NXT" else "J"
        code_name = str(pending.get("code_name", ""))
        buy_detail = str(pending.get("buy_detail", ""))
        reference_price = float(pending.get("entry_reference_price") or 0.0)
        attempt = int(pending.get("reprice_attempt", 0))

        # 재주문 직전 잔여 매수 여력 재확인 (취소~재주문 사이 다른 체결/입출금으로 감소했을 수 있음)
        affordable_qty = self.get_affordable_buy_qty(code, reference_price or 1.0, now, nxt_tradeable)
        qty = min(qty, int(affordable_qty))
        if qty <= 0:
            log(f"  [BUY REPRICE ABORT] {code} | 재주문 여력 부족 - 포기")
            self.buy_inflight_codes.discard(norm_code)
            self.pending_orders.pop(norm_code, None)
            return

        if price_mode == "market" and exchange != "NXT":
            ord_dvsn, ord_unpr, price_for_log = "01", "0", None
        else:
            bid_price, ask_price = _fetch_bid_ask_price(norm_code, market_div)
            candidate = ask_price if price_mode == "ask" else bid_price
            candidate = candidate or bid_price or ask_price
            if not candidate or candidate <= 0:
                log(f"  [BUY REPRICE SKIP] {code} | 호가 조회 실패 - 다음 폴링에서 재시도")
                return  # pending 그대로 유지(reprice_pending=True) -> 다음 폴링에서 재시도
            ord_dvsn, ord_unpr, price_for_log = "00", str(int(round(candidate))), candidate

        if self.dry_run:
            log(f"DRY_RUN BUY REPRICE | {code} | qty={qty} | mode={price_mode} | price={price_for_log or 0:,.0f}")
            order_result = {
                "rt_cd": "0", "odno": "DRYRUN", "avg_pric": ord_unpr, "krx_fwdg_ord_orgno": "DRYRUN",
            }
        else:
            try:
                order_result = dsf.order_cash(
                    env_dv=self.env_dv,
                    ord_dv="buy",
                    cano=self.cano,
                    acnt_prdt_cd=self.acnt_prdt_cd,
                    pdno=code,
                    ord_dvsn=ord_dvsn,
                    ord_qty=str(qty),
                    ord_unpr=ord_unpr,
                    excg_id_dvsn_cd=exchange,
                )
            except Exception as exc:
                log(f"  [BUY REPRICE RESUBMIT ERROR] {code} | {exc}")
                return

        if not _order_succeeded(order_result):
            log(f"  [BUY REPRICE RESUBMIT FAILED] {code} | {_extract_order_error_detail(order_result)}")
            self.buy_inflight_codes.discard(norm_code)
            self.pending_orders.pop(norm_code, None)
            return

        new_order_no = _extract_order_number(order_result)
        new_order_org_no = _extract_order_org_no(order_result)
        new_price = _extract_order_price(order_result) or price_for_log or reference_price

        pending.update({
            "quantity": qty,
            "submitted_at": now,
            "requested_price": float(new_price or 0.0),
            "order_no": new_order_no,
            "order_org_no": new_order_org_no,
            "order_time": _extract_order_time(order_result),
            "cancel_inflight": False,
            "reprice_pending": False,
        })
        pending.pop("last_status_signature", None)
        detail_suffix = f" | {buy_detail}" if buy_detail else ""
        code_label = _format_code_label(code, code_name)
        log(
            f"  [BUY REPRICE RESUBMIT] {code_label} | attempt={attempt}/{BUY_ORDER_REPRICE_MAX_ATTEMPTS} | "
            f"mode={price_mode} | price={new_price or 0:,.0f} | order_no={new_order_no or 'UNKNOWN'}{detail_suffix}"
        )
        log_trade(
            f"{_symbol_log_label(code, code_name)} | BUY REPRICE RESUBMIT | qty={qty} | mode={price_mode} | "
            f"price={new_price or 0:,.0f} | order_no={new_order_no or 'UNKNOWN'}"
        )

    def _submit_buy_order_cash(self, code: str, qty: int, ord_dvsn: str, ord_unpr: str, exchange: str, log_price: float) -> dict | None:
        """매수 주문 제출(시장가/지정가 공용, place_buy_order/_resubmit_leg에서 재사용).
        dry_run이면 가짜 체결 응답을 반환. 예외 발생 시 None(호출측이 'BUY error' 로그)."""
        if self.dry_run:
            log(f"DRY_RUN BUY | {code} | qty={qty} | ord_dvsn={ord_dvsn} | price={log_price:,.0f} | exch={exchange}")
            return {"rt_cd": "0", "odno": "DRYRUN", "avg_pric": str(int(round(log_price)))}
        try:
            return dsf.order_cash(
                env_dv=self.env_dv,
                ord_dv="buy",
                cano=self.cano,
                acnt_prdt_cd=self.acnt_prdt_cd,
                pdno=code,
                ord_dvsn=ord_dvsn,
                ord_qty=str(qty),
                ord_unpr=ord_unpr,
                excg_id_dvsn_cd=exchange,
            )
        except Exception as exc:
            log(f"BUY error | {code} | {exc}")
            return None

    def place_buy_order(self, code: str, price: float, qty: int, now: datetime, nxt_tradeable: bool, session: str, buy_detail: str = "", code_name: str = "", pyramid: bool = False) -> bool:
        norm_code = str(code).zfill(6)
        if not pyramid and self.has_buy_exposure(norm_code):
            log(f"BUY skipped | {code} | reason=BUY_EXPOSURE_ACTIVE")
            return False
        if qty <= 0 or self._in_cooldown(norm_code, now) or self.has_pending_order(norm_code):
            return False

        affordable_qty = self.get_affordable_buy_qty(code, price, now, nxt_tradeable)
        qty = min(int(qty), int(affordable_qty))
        if qty <= 0:
            log(f"BUY skipped | {code} | reason=INSUFFICIENT_BUYING_POWER_AT_ORDER_TIME")
            return False

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return False

        exchange = order_spec["exchange"]
        market_div = "NX" if exchange == "NXT" else "J"
        bid_price, ask_price = _fetch_bid_ask_price(norm_code, market_div)
        limit_price = int(round(bid_price)) if (bid_price and bid_price > 0) else int(round(price))
        cross_price = int(round(ask_price)) if (ask_price and ask_price > 0) else limit_price

        # [2026-09-18] 신규 진입(비-피라미딩) 매수를 시장가 leg(즉시체결)+매수1호가
        # 지정가 leg(추격재주문)로 분할한다 - 183300 코미코 사례(지정가 전량이 급등을
        # 못 따라가 체결까지 61초)처럼 느린 체결을 완화(r001/r003 Update log 2026-09-18
        # 참조). 수량이 1주뿐이면 분할해도 지정가 쪽에 남는 게 없어 무의미하므로 그대로
        # 시장가 1건으로 즉시 매수. 피라미딩은 기존과 동일하게 매수1호가 지정가 단일 주문.
        if pyramid or not ENABLE_BUY_SPLIT_MARKET_LIMIT or qty < 2:
            role = "market" if (not pyramid and qty <= 1 and ENABLE_BUY_SPLIT_MARKET_LIMIT) else "limit"
            legs_spec = [(role, qty)]
        else:
            market_qty = int(round(qty * BUY_SPLIT_MARKET_RATIO))
            market_qty = max(1, min(market_qty, qty - 1))
            legs_spec = [("market", market_qty), ("limit", qty - market_qty)]

        self.buy_inflight_codes.add(norm_code)
        legs: list[dict] = []
        for role, leg_qty in legs_spec:
            if leg_qty <= 0:
                continue
            if role == "market" and exchange != "NXT":
                ord_dvsn, ord_unpr, log_price, is_market_order = "01", "0", price, True
            elif role == "market":
                # NXT는 시장가 미지원 취급(기존 매도/재주문 정책과 동일) - 매도1호가
                # 크로싱 지정가로 즉시체결에 가깝게 낸다.
                ord_dvsn, ord_unpr, log_price, is_market_order = "00", str(cross_price), cross_price, False
            else:
                ord_dvsn, ord_unpr, log_price, is_market_order = "00", str(limit_price), limit_price, False

            order_result = self._submit_buy_order_cash(code, leg_qty, ord_dvsn, ord_unpr, exchange, log_price)
            if order_result is None:
                continue
            if not _order_succeeded(order_result):
                error_detail = _extract_order_error_detail(order_result)
                log(f"BUY failed | {code} | qty={leg_qty} | role={role} | {error_detail}")
                continue

            requested_price = float(_extract_order_price(order_result) or log_price)
            leg = {
                "role": role,
                "quantity": int(leg_qty),
                "order_no": _extract_order_number(order_result),
                "order_org_no": _extract_order_org_no(order_result),
                "order_time": _extract_order_time(order_result),
                "requested_price": requested_price,
                "entry_reference_price": requested_price,
                "exchange": exchange,
                "code_name": code_name,
                "submitted_at": now,
                "reprice_attempt": 0,
                "cancel_inflight": False,
                "reprice_pending": False,
                "is_market_order": is_market_order,
                "done": False,
            }
            legs.append(leg)
            detail_suffix = f" | {buy_detail}" if buy_detail else ""
            role_suffix = f" | role={role}" if len(legs_spec) > 1 else ""
            code_label = _format_code_label(code, code_name)
            log(
                f"BUY submitted | {code_label} | qty={leg_qty} | requested={requested_price:,.0f}{role_suffix} | "
                f"session={session} | exch={exchange} | order_no={leg['order_no'] or 'UNKNOWN'}{detail_suffix}"
            )
            log_trade(
                f"BUY submitted | {code_label} | qty={leg_qty} | requested={requested_price:,.0f}{role_suffix} | "
                f"session={session} | exch={exchange} | order_no={leg['order_no'] or 'UNKNOWN'}{detail_suffix}"
            )
            _log_trade_event_banner(
                event="BUY SUBMITTED",
                code=code,
                qty=int(leg_qty),
                price=requested_price,
                detail=f"{buy_detail}{role_suffix}" if buy_detail else role_suffix.strip(" |"),
                code_name=code_name,
            )

        if not legs:
            self.buy_inflight_codes.discard(norm_code)
            return False

        total_qty = sum(leg["quantity"] for leg in legs)
        primary_price = legs[0]["requested_price"]
        self.pending_orders[norm_code] = {
            "side": "buy",
            "quantity": total_qty,
            "submitted_at": now,
            "session": session,
            "requested_price": primary_price,
            "entry_reference_price": primary_price,  # 추격 상한 계산용 원 신호가, 재주문해도 불변
            "exchange": exchange,
            "order_no": legs[0]["order_no"],
            "order_org_no": legs[0]["order_org_no"],
            "order_time": legs[0]["order_time"],
            "buy_detail": buy_detail,
            "code_name": code_name,
            "pyramid": pyramid,
            "reprice_attempt": 0,
            "cancel_inflight": False,
            "reprice_pending": False,
            "legs": legs if len(legs) > 1 else None,
        }
        self._mark_trade_lock(norm_code, now)
        traded = self.live_state.setdefault("traded_today", set())
        traded.add(norm_code)
        self.persist_live_state()
        return True

    def place_sell_order(self, code: str, qty: int, now: datetime, reason: str, nxt_tradeable: bool, price: float | None = None, code_name: str = "", market_order: bool = False) -> bool:
        self.sync_positions_from_account(force=False)
        pos = self.positions.get(code)
        if not pos or pos.get("quantity", 0) <= 0:
            return False

        norm_code = str(code).zfill(6)
        qty = min(int(qty), int(pos["quantity"]))
        if qty <= 0 or self._in_cooldown(norm_code, now):
            return False
        # 피라미딩 추가매수 주문이 대기 중이어도(side=buy) 이미 확정 체결된 기존 수량(pos["quantity"])에
        # 대한 매도(익절/손절)는 막지 않는다. 매도는 중복 제출 방지를 위해 대기 중인 매도 주문이
        # 있을 때만(side=sell) 차단한다.
        pending_order = self.pending_orders.get(norm_code)
        if pending_order is not None and pending_order.get("side") != "buy":
            return False

        order_spec = get_order_spec(now, nxt_tradeable)
        if order_spec is None:
            return False

        current_price = float(price or pos.get("current_price") or pos["buy_price"])
        market_div = "NX" if order_spec["exchange"] == "NXT" else "J"
        # 정규장(KRX)은 항상 시장가(01), NXT는 지정가(ask) 유지
        use_market = order_spec["exchange"] != "NXT"
        if use_market:
            ord_dvsn = "01"  # 시장가
            ord_unpr = "0"
        else:
            _, ask_price = _fetch_bid_ask_price(norm_code, market_div)
            limit_price = int(round(ask_price)) if (ask_price and ask_price > 0) else int(round(current_price))
            ord_dvsn = "00"  # 지정가
            ord_unpr = str(limit_price)

        if self.dry_run:
            log(f"DRY_RUN SELL | {code} | qty={qty} | reason={reason} | exch={order_spec['exchange']}")
            order_result = {"rt_cd": "0", "odno": "DRYRUN", "avg_pric": str(int(round(current_price)))}
        else:
            try:
                order_result = dsf.order_cash(
                    env_dv=self.env_dv,
                    ord_dv="sell",
                    cano=self.cano,
                    acnt_prdt_cd=self.acnt_prdt_cd,
                    pdno=code,
                    ord_dvsn=ord_dvsn,
                    ord_qty=str(qty),
                    ord_unpr=ord_unpr,
                    excg_id_dvsn_cd=order_spec["exchange"],
                )
            except Exception as exc:
                log(f"SELL error | {code} | {exc}")
                return False

        if not _order_succeeded(order_result):
            error_detail = _extract_order_error_detail(order_result)
            log(f"SELL failed | {code} | qty={qty} | reason={reason} | {error_detail}")
            return False

        requested_price = _extract_order_price(order_result) or current_price
        order_no = _extract_order_number(order_result)
        order_time = _extract_order_time(order_result)
        self.pending_orders[norm_code] = {
            "side": "sell",
            "quantity": int(qty),
            "submitted_at": now,
            "requested_price": float(requested_price),
            "exchange": order_spec["exchange"],
            "order_no": order_no,
            "order_time": order_time,
            "reason": reason,
            "code_name": code_name,
            "buy_price": float(pos.get("buy_price", 0.0)),
            "buy_time": pos.get("buy_time"),
            "pre_submit_qty": int(pos.get("quantity", qty)),
        }
        self._mark_trade_lock(code, now)
        code_label = _format_code_label(code, code_name)
        log(
            f"SELL submitted | {code_label} | qty={qty} | requested={requested_price:,.0f} | "
            f"reason={reason} | exch={order_spec['exchange']} | order_no={order_no or 'UNKNOWN'}"
        )
        self.refresh_pending_orders(now)
        return True


# ---------------------------------------------------------------------------
# 예약 청산
# ---------------------------------------------------------------------------

def run_scheduled_liquidations(
    current_dt: datetime,
    api: TradingAPI,
    nxt_map: dict[str, bool],
    watch_map: dict[str, str],
    state: dict,
    date_str: str,
    today_buy_codes: set[str],
) -> None:
    trade_date = current_dt.date()
    current_time = current_dt.time()

    if state.get("date") != trade_date:
        state["date"] = trade_date
        state["done_1520"] = False
        state["done_1959"] = False

    if not state["done_1520"] and current_time >= REGULAR_FORCE_EXIT:
        state["done_1520"] = True
        for code, pos in list(api.get_open_positions().items()):
            if code not in watch_map:
                continue
            if not _is_today_buy_position(code, pos, date_str, today_buy_codes):
                log(f"  [REGULAR CLOSE SKIP] {code} | NOT_TODAY_BUY_POSITION")
                continue
            if api.has_pending_order(code):
                log(f"  [REGULAR CLOSE SKIP] {code} | pending_order_active")
                continue
            price = float(pos.get("current_price") or pos["buy_price"])
            buy_price = float(pos.get("buy_price") or 0)
            if buy_price <= 0 or price <= 0:
                log(f"  [REGULAR CLOSE HOLD] {code} | INVALID_PRICE | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            pnl_pct = (price / buy_price) - 1.0
            action, reason = _session_exit_plan("REGULAR_CLOSE", pnl_pct)
            if action == "hold":
                log(f"  [REGULAR CLOSE HOLD] {code} | {reason} | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            api.trade_lock_until.pop(code, None)
            api.place_sell_order(code, int(pos["quantity"]), current_dt, reason, nxt_map.get(code, False), price=price, code_name=watch_map.get(code, ""))

    if ENABLE_NXT_SESSION and (not state["done_1959"]) and current_time >= AFTERNOON_NXT_FORCE_EXIT:
        state["done_1959"] = True
        for code, pos in list(api.get_open_positions().items()):
            if code not in watch_map:
                continue
            if not _is_today_buy_position(code, pos, date_str, today_buy_codes):
                log(f"  [NXT CLOSE SKIP] {code} | NOT_TODAY_BUY_POSITION")
                continue
            if api.has_pending_order(code):
                log(f"  [NXT CLOSE SKIP] {code} | pending_order_active")
                continue
            if not nxt_map.get(code, False):
                log(f"  [NXT CLOSE HOLD] {code} | NXT_NOT_TRADABLE")
                continue

            price = float(pos.get("current_price") or pos["buy_price"])
            buy_price = float(pos.get("buy_price") or 0)
            if buy_price <= 0 or price <= 0:
                log(f"  [NXT CLOSE HOLD] {code} | INVALID_PRICE | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            pnl_pct = (price / buy_price) - 1.0
            action, reason = _session_exit_plan("NXT_CLOSE", pnl_pct)
            if action == "hold":
                log(f"  [NXT CLOSE HOLD] {code} | {reason} | price={price:,.0f} buy={buy_price:,.0f}")
                continue

            api.trade_lock_until.pop(code, None)
            api.place_sell_order(code, int(pos["quantity"]), current_dt, reason, nxt_map.get(code, False), price=price, code_name=watch_map.get(code, ""))



_SHUTDOWN_API: TradingAPI | None = None


def _ensure_log_date_for(now: datetime | None = None) -> None:
    dt = now or datetime.now()
    date_str = dt.strftime("%Y%m%d")
    if str(_LOG_CTX.get("date_str")) != date_str:
        _rotate_logging_for_date(date_str)


def _log_pending_orders_on_shutdown(api: TradingAPI | None) -> None:
    if api is None or not getattr(api, "pending_orders", None):
        log("SHUTDOWN | no pending orders")
        return
    if not api.pending_orders:
        log("SHUTDOWN | no pending orders")
        return
    log(f"SHUTDOWN | pending_orders={len(api.pending_orders)}")
    for code, pending in api.pending_orders.items():
        log(
            f"  PENDING | {code} | side={pending.get('side')} qty={pending.get('quantity')} "
            f"order_no={pending.get('order_no', '')} submitted={pending.get('submitted_at')}"
        )


def _install_shutdown_handlers(api: TradingAPI) -> None:
    global _SHUTDOWN_API
    _SHUTDOWN_API = api

    def _atexit_shutdown() -> None:
        _log_pending_orders_on_shutdown(_SHUTDOWN_API)
        _shutdown_save_live_state(_SHUTDOWN_API)

    atexit.register(_atexit_shutdown)

    def _handler(signum, frame):  # noqa: ARG001
        _log_pending_orders_on_shutdown(api)
        _shutdown_save_live_state(api)
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def _shutdown_save_live_state(api: TradingAPI | None) -> None:
    if api is None:
        return
    try:
        api.persist_live_state()
    except Exception as exc:
        log(f"WARNING: live state save on shutdown failed: {exc}")


def run(target_date: str | None = None, env_dv: str | None = None, dry_run: bool | None = None, watchlist_source: str | None = None) -> None:
    now = datetime.now()
    _ensure_log_date_for(now)
    print(f"[R003 START] {now:%Y-%m-%d %H:%M:%S} | initializing live executor", flush=True)
    log("R003 START | initializing live executor")

    try:
        print("[R003 AUTH] starting ka.auth()", flush=True)
        ka.auth()
        print("[R003 AUTH] success", flush=True)
    except Exception as exc:
        print(f"[R003 AUTH ERROR] {exc}", flush=True)
        log(f"R003 AUTH ERROR | {exc}")
        return

    is_open_day, market_day_log = get_market_day_status(now)
    print(f"[R003 MARKET] {market_day_log}", flush=True)
    log(market_day_log)

    if not is_open_day:
        print("[R003 STOP] market closed day", flush=True)
        return

    watch_file = _resolve_watchlist_file(target_date, watchlist_source=watchlist_source or "auto")
    print(f"[R003 WATCHLIST FILE] {watch_file}", flush=True)
    try:
        watch_map = load_today_codes(watch_file)
    except Exception as exc:
        print(f"[R003 WATCHLIST ERROR] {exc}", flush=True)
        log(f"Failed to load code list: {exc}")
        watch_map = {}

    if not watch_map:
        print("[R003 WATCHLIST] No codes loaded", flush=True)
        log("No codes loaded")
        return

    print(f"[R003 WATCHLIST] loaded {len(watch_map)} codes", flush=True)

    register_symbol_names(watch_map)

    # [2026-08-31] active/backup 워치리스트 분리 - watch_map(g002 점수 내림차순, 최대 100개)
    # 중 상위 ACTIVE_WATCHLIST_SIZE개만 실시간 폴링(active_set)하고 나머지는 backup_pool로
    # 대기(미폴링)시킨다. 100개 미만이면 슬라이싱이 그대로 안전하게 축소 동작한다(예: 30개면
    # active=30, backup=0). 상세 배경은 r001_define_config.py의 ACTIVE_WATCHLIST_SIZE 정의부
    # 및 g002 Update log 2026-08-31 참조.
    watch_codes_ordered = list(watch_map.keys())
    active_set: set[str] = set(watch_codes_ordered[:ACTIVE_WATCHLIST_SIZE])
    backup_pool: "collections.deque[str]" = collections.deque(watch_codes_ordered[ACTIVE_WATCHLIST_SIZE:])
    first_active_at: dict[str, datetime] = {code: now for code in active_set}
    active_watchlist_warn_state: dict[str, bool] = {}
    log(f"[ACTIVE/BACKUP SPLIT] active={len(active_set)} backup={len(backup_pool)} total={len(watch_map)}")

    log(f"[FEATURE_R004_WATCHLIST] Watchlist source: {watch_file}")

    if ENABLE_NXT_SESSION:
        log("MODE BANNER: REGULAR+NXT_MODE")
    else:
        log("MODE BANNER: REGULAR_ONLY_MODE")

    nxt_map = {code: is_nxt_tradeable(code) for code in watch_map}
    for code, name in watch_map.items():
#        print(f"[R003 WATCH] {code} | {name} | NXT={nxt_map[code]}", flush=True)
        log(f"WATCH | {code} | {name} | NXT={nxt_map[code]}")

    log("Strategy: live price cross over buffered BB middle + Stoch/RSI/Williams confirmation")
    log(
        f"Live-cross filter: BB buffer={LIVE_PRICE_BB_BUFFER_PCT*100:.3f}% | "
        f"confirm polls={LIVE_PRICE_CROSS_CONFIRM_POLLS} | confirm seconds={LIVE_PRICE_CROSS_CONFIRM_SECONDS}"
    )
    log(
        f"Polling: turn restarts immediately (min_cycle={MAIN_LOOP_MIN_CYCLE_SECONDS}s floor) | "
        f"idle/session-gap wait={LIVE_PRICE_POLL_INTERVAL_SECONDS}s | "
        f"frame refresh every {FRAME_POLL_INTERVAL_SECONDS}s (3min bars) + backfill {FRAME_BACKFILL_SYNC_SECONDS}s | "
        f"buy consecutive confirms={BUY_CONSECUTIVE_CONFIRM_COUNT}"
    )
    log(
        f"ATR model: stop={ATR_STOP_MULTIPLIER:.1f}x | tp={ATR_TAKE_PROFIT_MULTIPLIER:.1f}x | "
        f"trail={TRAILING_STOP_FROM_PEAK*100:.1f}%"
    )
    log(f"Indicator warmup: require bars >= {INDICATOR_WARMUP_BARS} before new entries")
    date_str = now.strftime("%Y%m%d")
    live_state = load_live_state(date_str)
    traded_today: set[str] = set(live_state.get("traded_today") or [])
    effective_env = (env_dv or KIS_ENV_DV).strip() or "real"
    effective_dry = LIVE_DRY_RUN if dry_run is None else bool(dry_run)
    if ENABLE_LIVE_DRY_RUN:
        effective_dry = True
    log(f"CONFIG BANNER | env_dv={effective_env} | dry_run={effective_dry}")
    log(f"CONFIG BANNER | MARKET_DAY_FAIL_CLOSED={MARKET_DAY_FAIL_CLOSED} | SESSION_FORCE_CLOSE_ALL_AT_CUTOFF={SESSION_FORCE_CLOSE_ALL_AT_CUTOFF}")
    log(
        "CONFIG BANNER | aux-trigger guard | "
        f"min_target={AUX_SELL_MIN_REALIZED_TARGET_PCT*100:.2f}% "
        f"slippage_buffer={AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT*100:.2f}%"
    )
    api = TradingAPI(env_dv=effective_env, dry_run=effective_dry, live_state=live_state)
    _install_shutdown_handlers(api)
    api.live_state["traded_today"] = traded_today
    signal_buy_bar: dict[str, object] = {}
    signal_sell_bar: dict[str, object] = {}
    trailing_sell_confirm_state: dict[str, dict[str, object]] = {}
    # 급등 판정(detect_price_surge)용 종목별 최근 (시각, 현재가) 표본 - 보유 중인 종목만 쌓고 청산 시 비운다.
    recent_price_samples: dict[str, list[tuple[datetime, float]]] = {}
    buy_confirm_state: dict[str, dict[str, object]] = {}
    buy_trigger_age_state: dict[str, dict[str, object]] = {}  # [2026-09-23] _008 트리거 유효기한 상태
    post_buy_bb_drop_state: dict[str, dict] = {}
    breakeven_fail_state: dict[str, dict] = {}
    no_trend_exit_state: dict[str, dict] = {}
    atr_stop_confirm_state: dict[str, dict] = {}
    hard_stop_confirm_state: dict[str, dict] = {}  # [2026-09-23] _004 하드손절 확인창 상태
    hybrid_1min_dead_cross_state: dict[str, dict] = {}  # [2026-09-23] _011 1분봉 데드크로스 확인창 상태
    peak_retrace_guard_state: dict[str, dict] = {}  # [2026-09-23] _012 고점대비 익절가드 확인창 상태
    # 연속 HARD_STOP 서킷브레이커 상태
    hard_stop_today_codes: set[str] = set()  # 당일 HARD_STOP 발생 종목 (재진입 영구 차단)
    gap_blocked_codes: set[str] = set()  # 개장초 갭하락으로 당일 신규매수 차단된 종목
    # 당일 손절 누적 횟수/서킷브레이커 해제 시각 - 매수(_007)/매도(_004, _019) 조건이 같은 인스턴스를 공유한다
    # (2026-09-23: 매도측 신규 _011/_012 삽입으로 옛 _017 atr_stop_loss가 _019로 밀림)
    risk = RiskState()
    live_price_cross_state: dict[str, dict] = {}
    live_price_cache: dict[str, float] = {}
    live_price_cache_at: dict[str, datetime] = {}
    live_price_fail_count: dict[str, int] = {}
    live_price_backoff_until: dict[str, datetime] = {}
    frame_cache: dict[str, pd.DataFrame] = {}
    frame_last_refresh_at: dict[str, datetime] = {}
    frame_cache_1min: dict[str, pd.DataFrame] = {}
    frame_last_refresh_at_1min: dict[str, datetime] = {}
    realtime_entry_bar_state: dict[str, dict[str, object]] = {}
    liquidation_state: dict = {}
    # 번호 붙은 매수/매도 조건 모듈(r005/r006)에 주입하는 상태 컨테이너/함수 묶음. 컨테이너는 날짜 변경 시
    # .clear()만 하므로(재바인딩 없음) 한 번 만들어 계속 쓴다. traded_today는 틱마다 재바인딩되므로 종목 순회
    # 안에서 BuyState를 새로 만든다.
    sell_state = SellState(
        recent_price_samples=recent_price_samples,
        trailing_sell_confirm_state=trailing_sell_confirm_state,
        post_buy_bb_drop_state=post_buy_bb_drop_state,
        breakeven_fail_state=breakeven_fail_state,
        no_trend_exit_state=no_trend_exit_state,
        atr_stop_confirm_state=atr_stop_confirm_state,
        hard_stop_confirm_state=hard_stop_confirm_state,
        hybrid_1min_dead_cross_state=hybrid_1min_dead_cross_state,
        peak_retrace_guard_state=peak_retrace_guard_state,
        signal_sell_bar=signal_sell_bar,
        hard_stop_today_codes=hard_stop_today_codes,
    )
    sell_services = SellServices(
        log=log,
        is_stale_live_price_source=_is_stale_live_price_source,
        sanitize_surge_ladder=_sanitize_surge_ladder,
        extract_aux_score_from_reason=_extract_aux_score_from_reason,
        aux_min_pnl_for_score=_aux_min_pnl_for_score,
        check_sell_condition=check_sell_condition,
        classify_buy_session=classify_buy_session,
        get_frame_1min=lambda _code, _dt, _nxt: _get_or_refresh_1min_frame(
            _code, _dt, _nxt, frame_cache_1min, frame_last_refresh_at_1min,
        ),
    )
    buy_services = BuyServices(
        log=log,
        is_new_entry_allowed=is_new_entry_allowed,
        get_session_open_datetime=get_session_open_datetime,
        get_frame_1min=lambda _code, _dt, _nxt: _get_or_refresh_1min_frame(
            _code, _dt, _nxt, frame_cache_1min, frame_last_refresh_at_1min,
        ),
        fetch_prev_close=fetch_prev_close,
        rise_from_prev_close=_rise_from_prev_close,
        is_stale_live_price_source=_is_stale_live_price_source,
        classify_buy_session=classify_buy_session,
        get_order_spec=get_order_spec,
        fetch_orderbook_totals=_fetch_orderbook_totals,
        format_reject_detail=_buy_reject_detail,
    )
    current_trade_date = now.date()
    loop_consecutive_errors = 0
    last_watchlist_mismatch_log_at: datetime | None = None
    if ENABLE_NXT_SESSION:
        log(
            f"Session-open warmup: new entries blocked for {STARTUP_WARMUP_SECONDS}s "
            f"after each session start ({MORNING_NXT_START:%H:%M}, {REGULAR_START:%H:%M}, {AFTERNOON_NXT_START:%H:%M})"
        )
    else:
        log(
            f"Session-open warmup: new entries blocked for {STARTUP_WARMUP_SECONDS}s "
            f"after regular session start ({REGULAR_START:%H:%M})"
        )

    served_tick = datetime.now()
    regular_session_watchlist_reset_done = False

    while True:
        current_dt = datetime.now()
        _ensure_log_date_for(current_dt)

        if current_dt.date() != current_trade_date:
            current_trade_date = current_dt.date()
            date_str = current_dt.strftime("%Y%m%d")
            live_state = load_live_state(date_str)
            _rotate_logging_for_date(date_str)
            live_state = load_live_state(date_str)
            traded_today = set(live_state.get("traded_today") or [])
            api.live_state = live_state
            api.live_state["traded_today"] = traded_today
            api._apply_persisted_position_meta()
            # 날짜 변경: 서킷브레이커 상태 초기화
            hard_stop_today_codes.clear()
            gap_blocked_codes.clear()
            risk.reset()
            post_buy_bb_drop_state.clear()
            breakeven_fail_state.clear()
            no_trend_exit_state.clear()
            atr_stop_confirm_state.clear()
            hard_stop_confirm_state.clear()
            hybrid_1min_dead_cross_state.clear()
            peak_retrace_guard_state.clear()
            signal_buy_bar.clear()
            signal_sell_bar.clear()
            trailing_sell_confirm_state.clear()
            buy_confirm_state.clear()
            buy_trigger_age_state.clear()
            live_price_cross_state.clear()
            live_price_cache.clear()
            live_price_cache_at.clear()
            live_price_fail_count.clear()
            live_price_backoff_until.clear()
            frame_cache.clear()
            frame_last_refresh_at.clear()
            frame_cache_1min.clear()
            frame_last_refresh_at_1min.clear()
            realtime_entry_bar_state.clear()
            regular_session_watchlist_reset_done = False

        is_open_day, market_day_log = get_market_day_status(current_dt)
        if not is_open_day:
            log(market_day_log)
            log("MARKET LOOP STOP | reason=market_closed_day")
            break

        market_end_time = AFTERNOON_NXT_END if ENABLE_NXT_SESSION else REGULAR_END
        if current_dt.time() >= market_end_time:
            run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state, date_str, traded_today)
            log(f"{market_end_time:%H:%M} reached. Stopping.")
            break

        if not is_regular_session(current_dt) and not is_nxt_session(current_dt):
            served_tick = _sleep_until_next_tick(served_tick, LIVE_PRICE_POLL_INTERVAL_SECONDS)
            continue

        try:
            # MAIN_LOOP_TICK_START
            if (
                last_watchlist_mismatch_log_at is None
                or (current_dt - last_watchlist_mismatch_log_at).total_seconds() >= WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS
            ):
                _log_account_watchlist_mismatch(api, watch_map)
                last_watchlist_mismatch_log_at = current_dt
            api.sync_positions_from_account(force=False)
            api.refresh_pending_orders(current_dt)
            traded_today = {str(c).zfill(6) for c in (api.live_state.get("traded_today") or [])}
            api.live_state["traded_today"] = traded_today
            run_scheduled_liquidations(current_dt, api, nxt_map, watch_map, liquidation_state, date_str, traded_today)

            # 15:20~15:30 정규장 마감 구간은 종목별 매도 체크 건너뛰고
            # 동시호가 예약 청산 로직(당일 매수 + 수익 구간)만 실행한다.
            if is_regular_call_auction(current_dt):
                served_tick = _sleep_until_next_tick(served_tick, LIVE_PRICE_POLL_INTERVAL_SECONDS)
                continue

            # [2026-08-31] 이 틱의 실제 순회 대상 = active_set(실시간 감시 상한) ∪ 보유 포지션
            # 전체. 보유 포지션은 랭크/backup 여부와 무관하게 항상 포함해야 청산 감시(TP/SL/
            # 트레일링, 이 루프 안에서만 실행됨)가 끊기지 않는다. 정렬은 로그 가독성 유지 목적.
            position_codes = {
                str(c).zfill(6) for c, p in api.get_open_positions().items()
                if int(p.get("quantity", 0) or 0) > 0
            }
            iter_codes = sorted(active_set | position_codes)

            for code in iter_codes:
                name = watch_map.get(code) or _SYMBOL_NAME_MAP.get(code) or code
                nxt_tradeable = nxt_map.get(code, False)
                symbol_label = _symbol_log_label(code, name)
                if not can_trade_code_now(current_dt, nxt_tradeable):
                    log(f"  [SKIP    ] {symbol_label} | can_trade_code_now=False | time={current_dt:%H:%M:%S} nxt={nxt_tradeable}")
                    continue

                pos = api.get_open_positions().get(code)
                if pos is not None and pos.get("quantity", 0) > 0 and not _is_today_buy_position(code, pos, date_str, traded_today):
                    log(f"  {symbol_label} [HOLD SKIP] | NOT_TODAY_BUY_POSITION")
                    continue

                cached_frame = frame_cache.get(code)
                last_frame_refresh = frame_last_refresh_at.get(code)
                frame: pd.DataFrame | None = cached_frame

                if should_refresh_3min_frame(current_dt, cached_frame, last_frame_refresh):
                    try:
                        refreshed_frame = fetch_3min_frame(code, current_dt, nxt_tradeable)
                    except Exception as exc:
                        log(f"{code} frame error: {exc}")
                        refreshed_frame = None

                    if refreshed_frame is not None and not refreshed_frame.empty:
                        merged_frame = _merge_bar_frame(cached_frame, refreshed_frame)
                        frame_cache[code] = merged_frame
                        frame_last_refresh_at[code] = current_dt
                        frame = merged_frame

                if frame is None:
                    log(f"  [SKIP    ] {symbol_label} | frame=None (fetch failed)")
                    continue
                if len(frame) < INDICATOR_WARMUP_BARS:
                    log(f"  [SKIP    ] {symbol_label} | bars={len(frame)} < INDICATOR_WARMUP_BARS={INDICATOR_WARMUP_BARS}")
                    continue

                bar_time = frame.index[-1]
                last_closed_bar = pd.Timestamp(current_dt).floor("3min")
                bar_age_sec = max(0.0, (pd.Timestamp(current_dt) - bar_time).total_seconds())
                cur = frame.iloc[-1]
                live_backoff_until = live_price_backoff_until.get(code)
                can_fetch_live = live_backoff_until is None or current_dt >= live_backoff_until
                price_source = "live"
                price = None

                if can_fetch_live:
                    price = fetch_live_price(code, current_dt, nxt_tradeable)
                    if price is None or price <= 0:
                        fail_count = int(live_price_fail_count.get(code, 0)) + 1
                        live_price_fail_count[code] = fail_count
                        backoff_seconds = _live_price_backoff_seconds(fail_count)
                        live_price_backoff_until[code] = current_dt + timedelta(seconds=backoff_seconds)
                        price_source = f"fallback_backoff_{backoff_seconds}s"
                    else:
                        live_price_cache[code] = float(price)
                        live_price_cache_at[code] = current_dt
                        live_price_fail_count[code] = 0
                        live_price_backoff_until.pop(code, None)
                else:
                    price_source = "cached_live(backoff_active)"

                if price is None or price <= 0:
                    cached_live = live_price_cache.get(code)
                    cached_at = live_price_cache_at.get(code)
                    cache_age_sec = (current_dt - cached_at).total_seconds() if isinstance(cached_at, datetime) else None
                    cache_fresh = cache_age_sec is not None and cache_age_sec <= LIVE_PRICE_STALE_TTL_SECONDS

                    if cached_live is not None and cached_live > 0 and cache_fresh:
                        price = float(cached_live)
                        if price_source == "live":
                            price_source = "cached_live"
                    else:
                        price = float(cur["close"])
                        age_text = f"{cache_age_sec:.0f}s" if cache_age_sec is not None else "unknown"
                        price_source = f"bar_close(stale_live={age_text})"

                buy_frame = frame
                if ENABLE_INTRABAR_LIVE_ENTRY_FILTER and price is not None and price > 0:
                    try:
                        realtime_frame, realtime_elapsed = _build_realtime_entry_frame(
                            frame,
                            code,
                            current_dt,
                            float(price),
                            realtime_entry_bar_state,
                        )
                        gate_ok, gate_reason = _passes_intrabar_entry_gate(realtime_frame, realtime_elapsed)
                        if gate_ok:
                            buy_frame = realtime_frame
                        elif not gate_reason.startswith("INTRABAR_ELAPSED_"):
                            log(f"  [BAR_SKIP] {symbol_label} | {gate_reason} | using confirmed 3min bar instead")
                    except Exception as exc:
                        log(f"  [WARN] {symbol_label} | realtime entry-frame build failed: {exc}")
                        buy_frame = frame

                buy_cur = buy_frame.iloc[-1] if buy_frame is not None and not buy_frame.empty else cur

                cross_info = update_live_price_cross_state(
                    live_price_cross_state,
                    code,
                    current_dt,
                    float(price),
                    _num(buy_cur, "BB_MIDDLE"),
                )
                pending = api.get_pending_order(code)

                if pending is not None:
                    pending_side = str(pending.get("side", "")).upper() or "UNKNOWN"
                    pending_qty = int(pending.get("quantity", 0))
                    pending_time = pending.get("submitted_at", current_dt)
                    is_pyramid_pending = (
                        bool(pending.get("pyramid"))
                        and pos is not None
                        and int(pos.get("quantity", 0) or 0) > 0
                    )
                    log(
                        f"  [PENDING ] {symbol_label} | side={pending_side} qty={pending_qty} "
                        f"submitted={pending_time:%H:%M:%S} | order_no={pending.get('order_no', '') or 'UNKNOWN'}"
                        + (" | PYRAMID_ADD_IN_FLIGHT - position monitoring continues" if is_pyramid_pending else "")
                    )
                    if not is_pyramid_pending:
                        continue
                    # 피라미딩 추가매수 체결 대기 중에도 기존 보유수량에 대한 손절/익절 감시는
                    # 계속 진행한다 (아래 포지션 관리 블록으로 그대로 진입).

                log(
                    f"  [CHECK   ] {symbol_label} | bars={len(frame)} live={price:,.0f}  bar_close={float(cur['close']):,.0f} | "
                    f"confirmed_bar={bar_time:%H:%M:%S} cutoff={last_closed_bar:%H:%M:%S} bar_age={bar_age_sec:.0f}s | "
                    f"MA5={_num(cur, 'MA_5'):.1f} BB_MID={_num(cur, 'BB_MIDDLE'):.1f} BB_UP={_num(cur, 'BB_UPPER'):.1f} BB_LW={_num(cur, 'BB_LOWER'):.1f} | "
                    f"CROSS relation={cross_info.get('relation')} upper={float(cross_info.get('upper_trigger', 0.0)):.1f} lower={float(cross_info.get('lower_trigger', 0.0)):.1f} pending={cross_info.get('pending_side')} cnt={cross_info.get('pending_count')} sec={float(cross_info.get('pending_seconds', 0.0)):.0f} signal={cross_info.get('signal')} | "
                    f"RSI={_num(cur, 'RSI'):.1f} SIG={_num(cur, 'RSI_SIGNAL'):.1f} | "
                    f"K={_num(cur, 'STOCH_K'):.1f} D={_num(cur, 'STOCH_D'):.1f} | "
                    f"WR={_num(cur, 'WILLIAMS_R'):.1f} WD={_num(cur, 'WILLIAMS_D'):.1f} | "
                    f"MACD={_num(cur, 'MACD'):.2f} SIG={_num(cur, 'MACD_SIGNAL'):.2f} HIST={_num(cur, 'MACD_HIST'):.2f} | "
                    f"ADX={_num(cur, 'ADX'):.1f} +DI={_num(cur, 'DI_PLUS'):.1f} -DI={_num(cur, 'DI_MINUS'):.1f} | "
                    f"VOL={_num(cur, 'volume'):,.0f} VOLMA={_num(cur, 'VOL_MA20'):,.0f} | "
                    f"VWAP={_num(cur, 'VWAP'):,.0f} OBV={_num(cur, 'OBV'):,.0f} OBVMA={_num(cur, 'OBV_MA'):,.0f}"
                )

                if pos is None or pos.get("quantity", 0) <= 0:
                    recent_price_samples.pop(code, None)
                    post_buy_bb_drop_state.pop(code, None)
                    breakeven_fail_state.pop(code, None)
                    no_trend_exit_state.pop(code, None)
                    atr_stop_confirm_state.pop(code, None)
                    hard_stop_confirm_state.pop(code, None)
                    hybrid_1min_dead_cross_state.pop(code, None)
                    peak_retrace_guard_state.pop(code, None)

                if pos is not None and pos.get("quantity", 0) > 0:
                    # 보유 종목: 번호 붙은 매도/포지션 관리 조건(r006_sell_conditions)을 순서대로 평가한다.
                    # 조건이 이 종목의 이번 틱 처리를 끝내면(주문 시도/보류) 평가가 거기서 멈춘다.
                    buy_confirm_state.pop(code, None)
                    buy_trigger_age_state.pop(code, None)
                    evaluate_sell_conditions(SellContext(
                        code=code, name=name, symbol_label=symbol_label, current_dt=current_dt,
                        nxt_tradeable=nxt_tradeable, pos=pos, price=price, price_source=price_source,
                        frame=frame, cur=cur, bar_time=bar_time, cross_info=cross_info, date_str=date_str,
                        api=api, state=sell_state, risk=risk, services=sell_services,
                    ))
                else:
                    # 신규 진입: 번호 붙은 매수 조건(r005_buy_conditions)을 순서대로 평가한다. 하이브리드 매수 경로가
                    # 유일한 신규 매수 경로다(2026-09-20 1분봉 골든크로스 단독/1분봉 Entry Score/3분봉 단독 경로 삭제):
                    # 1분봉 트리거 -> 3분봉 컨텍스트 -> 주문 직전 안전장치 순서.
                    trailing_sell_confirm_state.pop(code, None)
                    buy_ctx = BuyContext(
                        code=code, symbol_label=symbol_label, current_dt=current_dt, nxt_tradeable=nxt_tradeable,
                        price=price, price_source=price_source, bar_time=bar_time, buy_frame=buy_frame,
                        cross_info=cross_info, config=SHARED_R76_CONFIG, api=api,
                        state=BuyState(
                            buy_confirm_state=buy_confirm_state, buy_trigger_age_state=buy_trigger_age_state,
                            signal_buy_bar=signal_buy_bar,
                            hard_stop_today_codes=hard_stop_today_codes, gap_blocked_codes=gap_blocked_codes,
                            traded_today=traded_today,
                        ),
                        risk=risk, services=buy_services,
                    )
                    if check_buy_conditions(buy_ctx).approved:
                        # 모든 조건 통과 - _023이 traded_today/signal_buy_bar를 '예약'해 둔 상태다.
                        buy_reason, prev_bar, qty = buy_ctx.buy_reason, buy_ctx.prev_bar, buy_ctx.qty
                        session, buy_detail, norm_code = buy_ctx.session, buy_ctx.buy_detail, buy_ctx.norm_code
                        if api.place_buy_order(code, price, qty, current_dt, nxt_tradeable, session, buy_detail=buy_detail, code_name=name):
                            log(
                                f"  {symbol_label} [BUY EVAL] | OK {buy_reason} | {current_dt:%H:%M:%S} | "
                                f"LIVE {price:,.0f} | BB {_num(prev_bar, 'BB_MIDDLE'):.1f}->{_num(buy_frame.iloc[-1], 'BB_MIDDLE'):.1f} | "
                                f"RSI={_num(buy_frame.iloc[-1], 'RSI'):.1f} SIG={_num(buy_frame.iloc[-1], 'RSI_SIGNAL'):.1f} | "
                                f"K={_num(prev_bar, 'STOCH_K'):.1f}->{_num(buy_frame.iloc[-1], 'STOCH_K'):.1f} D={_num(buy_frame.iloc[-1], 'STOCH_D'):.1f} | "
                                f"WR={_num(prev_bar, 'WILLIAMS_R'):.1f}->{_num(buy_frame.iloc[-1], 'WILLIAMS_R'):.1f} WD={_num(buy_frame.iloc[-1], 'WILLIAMS_D'):.1f} | "
                                f"MACD {_num(prev_bar, 'MACD'):.2f}->{_num(buy_frame.iloc[-1], 'MACD'):.2f} SIG={_num(buy_frame.iloc[-1], 'MACD_SIGNAL'):.2f} | "
                                f"ADX={_num(buy_frame.iloc[-1], 'ADX'):.1f} +DI={_num(buy_frame.iloc[-1], 'DI_PLUS'):.1f} -DI={_num(buy_frame.iloc[-1], 'DI_MINUS'):.1f} | "
                                f"VOL={_num(buy_frame.iloc[-1], 'volume'):,.0f} VOLMA={_num(buy_frame.iloc[-1], 'VOL_MA20'):,.0f} | "
                                f"VWAP={_num(buy_frame.iloc[-1], 'VWAP'):,.0f} OBV={_num(buy_frame.iloc[-1], 'OBV'):,.0f} OBVMA={_num(buy_frame.iloc[-1], 'OBV_MA'):,.0f}"
                            )
                            log(f"  {symbol_label} [BUY EXECUTED] | {buy_reason} | qty={qty} price={price:,.0f} session={session}")
                            buy_confirm_state.pop(code, None)
                            buy_trigger_age_state.pop(code, None)
                            log("=" * 110)
                        else:
                            traded_today.discard(norm_code)
                            api.live_state["traded_today"] = traded_today
                            signal_buy_bar.pop(code, None)

            # [2026-09-02] 정규장 시작(REGULAR_START) 도달 시 active_set의 TIME_LIMIT
            # 카운트다운을 리셋 - 리셋 전에는 first_active_at이 장전 NXT 세션(08:00~08:50)
            # 편입 시각이라, NXT=False 종목은 그 구간 내내 can_trade_code_now가 False라
            # 매수평가 자체가 불가능한데도 ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES(60분)
            # 타이머는 그대로 흘러 정규장이 열리자마자 몇 분 만에 TIME_LIMIT으로 탈락하는
            # 문제가 있었음(126730 코칩 2026-09-02 09:03:41 탈락 사례 - 08:03:26 편입 후
            # 정규장 진입 3분여 만에 탈락, 이후 09:07 신호 여부와 무관하게 감시 대상에서
            # 완전히 빠져 매수 평가 자체가 이뤄지지 않음). 정규장 진입 시점 1회만 그 순간의
            # active_set 전원에 대해 first_active_at을 current_dt로 재설정해, TIME_LIMIT
            # 타이머가 프리마켓 대기시간을 소모하지 않고 정규장 실거래 시간부터 60분을
            # 온전히 확보하도록 함.
            if ENABLE_WATCHLIST_ROTATION:
                if not regular_session_watchlist_reset_done and is_regular_session(current_dt):
                    _reset_count = len(active_set)
                    for _code in active_set:
                        first_active_at[_code] = current_dt
                    regular_session_watchlist_reset_done = True
                    log(
                        f"[ACTIVE WATCHLIST] REGULAR_START 도달 - TIME_LIMIT"
                        f"({ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES}min) 타이머 리셋"
                        f"(프리마켓 NXT 대기시간 제외) | reset_count={_reset_count}"
                    )

                _rebalance_active_watchlist(
                    current_dt, api, watch_map, active_set, backup_pool, first_active_at,
                    hard_stop_today_codes, gap_blocked_codes, active_watchlist_warn_state,
                    frame_cache,
                )
            _write_active_watchlist_state(
                current_dt, watch_map, active_set, backup_pool, first_active_at, position_codes,
            )
            _write_active_watchlist_txt(current_dt, watch_map, active_set, first_active_at)

            api.maybe_persist_live_state_interval(current_dt, current_dt.strftime("%Y%m%d"))
            loop_consecutive_errors = 0
        except KeyboardInterrupt:
            _log_pending_orders_on_shutdown(api)
            try:
                api.persist_live_state(date_str=current_dt.strftime("%Y%m%d"))
            except Exception:
                pass
            log("MAIN LOOP STOP | reason=keyboard_interrupt")
            break
        except Exception as exc:
            loop_consecutive_errors += 1
            log(f"MAIN LOOP ERROR ({loop_consecutive_errors}/{MAIN_LOOP_MAX_CONSECUTIVE_ERRORS}): {exc}")
            if loop_consecutive_errors >= MAIN_LOOP_MAX_CONSECUTIVE_ERRORS:
                _log_pending_orders_on_shutdown(api)
                try:
                    api.persist_live_state(date_str=current_dt.strftime("%Y%m%d"))
                except Exception:
                    pass
                log("MAIN LOOP STOP | reason=max_consecutive_errors")
                break

        # [2026-09-14] 턴 종료 즉시 다음 턴 시작 (사용자 요청) - 이전엔 여기서
        # _sleep_until_next_tick으로 다음 10초 정렬 틱까지 대기했으나, 턴 처리 자체는
        # 보통 1초 미만이라 매번 최대 10초의 불필요한 유휴 시간이 발생했음. 활성 종목이
        # 0개라 턴이 사실상 즉시 끝나는 경우에만 계좌/미체결 동기화 API를 스팸하지
        # 않도록 최소 간격(MAIN_LOOP_MIN_CYCLE_SECONDS)만 바닥으로 보장.
        turn_elapsed_seconds = (datetime.now() - current_dt).total_seconds()
        if turn_elapsed_seconds < MAIN_LOOP_MIN_CYCLE_SECONDS:
            time.sleep(MAIN_LOOP_MIN_CYCLE_SECONDS - turn_elapsed_seconds)


if __name__ == "__main__":
    args = _parse_args()
    run(target_date=args.date, env_dv=args.env_dv, dry_run=args.dry_run, watchlist_source=args.watchlist_source)


