# -*- coding: utf-8 -*-

# Update log
# - [2026-09-23] type=feat owner=claude
#     summary: 사용자 요청("매수/매도 컨셉 재검토" 4단계 적용 - "급등 후 꺾이면 고점 대비 -0.8%서 익절")
#       + Codex 설계검토 - r006에 신규 _012_peak_retracement_guard 추가. TP1(+3.0%)/ATR익절선 도달 전
#       구간(현재 무보호)에서 고점 대비 되돌림이 문턱(ATR% 기반, 하한 0.8%) 이상이면 남은 이익을 지킨다.
#       신규 상수 ENABLE_PEAK_RETRACE_GUARD/PEAK_RETRACE_GUARD_ARM_PNL/_MIN_PCT/_ATR_MULT/
#       _CONFIRM_SECONDS. 매도 조건 번호 _012~_021이 _013~_022로 다시 한 칸씩 밀림(로직 무변경).
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: breaking (익절 빈도 증가 - TP1 미도달 구간에서 더 일찍 이익 실현. 롤백:
#       ENABLE_PEAK_RETRACE_GUARD=False)
# - [2026-09-23] type=feat owner=claude
#     summary: 사용자 요청("매수/매도 컨셉 재검토" 3~4단계 적용) + Codex 설계검토 - r006에 신규
#       _011_hybrid_1min_dead_cross_exit 추가(매수측 1분봉 골든크로스와 대칭되는 매도 조건, 1분봉 BB중심선
#       데드크로스 + 손실구간 수익요건 없이 즉시청산/수익구간 BB기울기 추가확인 + 확인창). 신규 상수
#       ENABLE_HYBRID_1MIN_DEADCROSS_EXIT/HYBRID_1MIN_DEADCROSS_LOOKBACK_BARS/_MIN_HOLD_SECONDS/
#       _CONFIRM_SECONDS/_LOSS_EXIT_PNL_MAX. 매도 조건 번호 _011~_020이 _012~_021로 한 칸씩 밀림(기존
#       조건 로직은 무변경, 이름/번호만 이동) - r003의 "_004, _017 RiskState 공유" 주석도 "_004, _018"로
#       갱신. r006/buy_condition_flow.txt 순서표 갱신.
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: breaking (매도 빈도 증가 - 이전에 늦은 시그널 청산/하드손절까지 버티던 손실 포지션이
#       더 일찍 청산됨. 롤백: ENABLE_HYBRID_1MIN_DEADCROSS_EXIT=False)
# - [2026-09-23] type=feat owner=claude
#     summary: 사용자 요청(204620 글로벌텍스프리 2026-09-23 09:17 매수 지연 사례 - "매수/매도 컨셉
#       재검토" 중 발견 + Codex 설계검토) - 신규 HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS(300초=5분, 탐색적
#       값) 추가. 1분봉 골든크로스는 유효한데 3분봉 컨텍스트 게이트가 못 따라와 대기만 길어지다가, 뒤늦게
#       게이트가 맞아떨어진 시점엔 이미 가격이 추격매수 구간까지 오른 사례(204620: 09:04 트리거 유효 ->
#       09:17 실제 매수, 13분 지연) 대응. Codex가 제안한 "진입 이벤트에 유효기한을 두자"는 아이디어를
#       구현 - r005 _008_hybrid_1min_trigger에서 update_timed_condition_state()로 1분봉 트리거가
#       "계속 유효" 상태로 지속된 시간을 추적, 초과 시 HYBRID_1MIN_TRIGGER_EXPIRED로 반려한다(r002
#       check_buy_condition_1min_hybrid_trigger 자체는 무변경, r005/g003 양쪽에서 동일 로직 사용).
#       <=0이면 비활성(기존과 동일). 아직 백테스트로 최적값을 검증하지 않았다 - 실거래 관찰 필요.
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: breaking (1분봉 트리거가 이 시간 이상 대기 상태로 지속된 뒤 매수하던 케이스가
#       이제 거부됨 - 매수 빈도 감소 방향. 롤백: HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS=0.0)
# - [2026-09-23] type=fix owner=claude
#     summary: 사용자 요청(119850 지엔씨에너지 2026-09-23 09:02 하드손절 직후 반등 사례 분석 + Codex 검토) -
#       HARD_STOP_LOSS(_004, r006)가 20개 매도조건 중 유일하게 확인창(디바운스) 없이 단일 폴링 틱에서 즉시
#       시장가 전량매도됨을 발견(ATR_STOP_LOSS/POST_BUY_DROP/BREAKEVEN_FAIL/NO_TREND_EXIT은 전부 확인창 보유).
#       119850 건: 09:02:10 -1.27% -> 09:02:31 -2.00%로 21초 폴링 간격에 임계값을 그냥 통과해 확인 없이 즉시
#       발동, 체결가는 시장가 슬리피지로 -2.18%. 신규 HARD_STOP_CONFIRM_SECONDS(10초) 추가 - ATR_STOP_LOSS와
#       동일한 update_timed_condition_state() 헬퍼 재사용. Codex 검토 결과 반영: 이 확인창은 "손실 상한 유지"가
#       아니라 "일시적 휩쏘 방지 vs 지속 하락 시 확인 시간만큼 손실 확대"의 트레이드오프임 - 무료 개선이 아님을
#       명시. g003도 동일 로직(r006 parity) 반영해 백테스트 정합성 유지(PaperStrategyTracker의 별도
#       STOP_LOSS_PERCENT 비교용 경로는 이 변경과 무관 - MULTI_FILTER/BASIC_CROSS 비교 전용, 실제 백테스트
#       경로 아님을 Codex 검토로 확인).
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: breaking (하드손절이 최소 10초 확인 후 발동 - 지속 하락 시 이전보다 다소 늦게/낮은
#       가격에 체결될 수 있음. 롤백: HARD_STOP_CONFIRM_SECONDS=0.0)
# - [2026-09-22] type=tuning owner=claude
#     summary: 사용자 요청("완화 검토안 적용") - 20260921 매수 2건 분석(analysis_20260921_missed_buys.md)에서 도출한
#       완화 3건 적용. (1) 연속 매수 확인창(r005 _020): BUY_CONFIRM_MAX_GAP_SECONDS 20->45초 + 같은 3분봉 유지
#       (신규 BUY_CONFIRM_REQUIRE_SAME_BAR) + 직전 통과 대비 가격 이탈 <=0.3%(신규 BUY_CONFIRM_MAX_PRICE_DRIFT_PCT).
#       정규장 종목당 평가주기(중앙 22초)가 20초 창보다 길어 2회 연속 확인이 5일 425건 중 2건만 성립했고, 제안 규칙은
#       425건 중 342건이 통과. (2) STOCH_D_BUY_MIN 50->40 (K>D & 40<=D<50 근접실패 n=82, TP36.6%/SL32.9%).
#       (3) HYBRID_1MIN_TRIGGER_LOOKBACK_BARS 8->12 (새로 통과 n=59, TP39.0%/SL28.8%). (2)(3)은 5일 소표본·사후
#       라벨 탐색 결과라 표본외 검증 전 잠정 적용이다. 확인창 3개 상수는 r003 실전 전용(g003은 확인을 연속 횟수로만
#       셈), STOCH_D/룩백은 r002 공용이라 g003 백테스트에도 함께 반영된다.
#     impact: common (r003 실전/g003 백테스트 공용, 확인창은 실전만)
#     compatibility: breaking (매수 빈도 증가. 롤백: BUY_CONFIRM_MAX_GAP_SECONDS=20, BUY_CONFIRM_REQUIRE_SAME_BAR=False,
#       BUY_CONFIRM_MAX_PRICE_DRIFT_PCT=0.0, STOCH_D_BUY_MIN=50.0, HYBRID_1MIN_TRIGGER_LOOKBACK_BARS=8)
# - [2026-09-21] type=refactor owner=claude
#     summary: 사용자 요청 - r003 매수/매도 조건을 번호 붙은 조건 객체(r005_buy_conditions/r006_sell_conditions)와
#       메인 함수로 분리(동작 불변 리팩터링). 이 파일은 신규 상수 BUY_CONFIRM_MAX_GAP_SECONDS만 추가 -
#       r003이 하드코딩하던 "연속 매수 확인" 간격 상한(POLL_INTERVAL_SECONDS*2)을 이름 붙여 노출한 것이며
#       값은 기존과 같은 20초라 동작 변화 없음. (참고: 20260921 로그 분석에서 정규장 종목당 평가주기가
#       중앙 22초라 이 20초 창이 사실상 성립하지 않는 것이 확인됨 - 값 변경은 전략 변경이라 이번엔
#       바꾸지 않음. r003 Update log 2026-09-21 참조)
#     impact: live (r003/r005)
#     compatibility: backward-compatible
# - [2026-09-21] type=fix owner=claude
#     summary: 사용자 요청 2건. (1) 023160 태광 2026-09-21 09:01 전량 일괄 매도 원인 수정 - 트레일(TP_EXTENSION)은
#       peak >= ATR 익절선(3xATR%, 태광 +1.91%)에서 무장하는데 1차 분할익절(TP1)은 max(+3.0%, 1.2xATR%)라서
#       ATR% < 1.0% 종목은 트레일이 TP1보다 먼저 무장된다. 태광은 고점 +2.79%(TP1 +3.0% 미달)에서 -1.0% 되돌림이
#       나오자 TP1 없이 18주 전량이 트레일로 매도됨. 신규 ENABLE_TP1_CAP_AT_ATR_TP(기본 True): TP1 목표를 ATR
#       익절선으로 상한 처리해 TP1이 트레일 무장보다 늦지 않게 함(ATR%>2.5% 고변동 종목의 ATR 동적 목표는 그대로).
#       (2) 급등 감지 + 사다리 익절 신설 - ENABLE_SURGE_LADDER_TP(기본 True)와 SURGE_* 상수 9개. 1차 익절 시점에
#       급등(60초 내 최저가 대비 상승폭 >= max(0.8%, 1.5xATR%) + BB 상단 돌파/봉 거래량 1.5배 중 1개 이상)이면
#       잔량을 TP1 체결가 +2%(진입수량 30%)/+4%(잔량 전량)에서 순차 익절, 트레일/손절은 병행. 급등이 아니면 기존
#       (TP1 후 잔량 전량 트레일 위임)과 동일. 공용 함수는 r002, 적용은 r003 실전/g003 백테스트.
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: breaking (저변동성 종목의 1차 익절이 앞당겨지고 급등 시 잔량 청산 방식이 바뀜; 두 플래그를
#       False로 두면 즉시 이전 동작으로 롤백)
# - [2026-09-20] type=refactor owner=claude
#     summary: 사용자 결정 반영(r002/r003 Update log 2026-09-20 참조). 하이브리드가 유일한 매수 경로가
#       되어 ENABLE_1MIN_GOLDEN_CROSS_BUY/ENABLE_1MIN_ENTRY_SCORE_GATE/ENABLE_1MIN_TRIGGER_3MIN_CONTEXT
#       플래그와 전용 상수(ENTRY_*, EMA_9_PERIOD) 삭제. 같은 날 재매수 허용(사용자 결정) - 실전이 읽은
#       적 없는 ALLOW_REBUY_SAME_CODE와 g003 전용 SIM_ALLOW_REENTRY_AFTER_COMPLETED_SELL 삭제.
#       거래량 하한 통일 - HYBRID_1MIN_MIN_ENTRY_VOL_MA/VOLUME 삭제(MIN_ENTRY_VOL_MA/VOLUME/
#       TURNOVER_KRW 하나만 사용). 신규 ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK(기본 True=현행 유지) -
#       bb_mid_downtrend_block 게이트를 끄는 결정은 별도 전략 변경으로 분리.
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: breaking (거래량 하한 통일로 하이브리드 트리거 통과 조건이 소폭 완화됨, 그 외 불변)
# - [2026-09-18] type=fix owner=claude
#     summary: 사용자 요청("오늘 매수 3건, 익절 가능했던 매수를 놓친 경우 점검") -
#       20260918 REJECT 로그 46,385건 분석 결과 score 15/24 이상 고품질 신호가 반려된
#       종목 41개 중 상당수(000500 가온전선 +22.7%/043260 성호전자 +21.3%/456010
#       아이씨티케이 +20.7%/024840 KBI메탈 +16.0% 등, 반려 시점 대비 당일 고점 기준)가
#       CHASE_BUY_BB_GAP 하나로 수렴 - HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT(1.1%)가
#       uptrend_continuation 경로(추세 지속이 다른 지표로 이미 검증된 경우)에도 동일하게
#       적용되면서, 1분봉 BB중간선이 급등을 못 따라가는 날엔 갭이 영구히 상한을 넘어
#       남은 장중 내내 진입이 막히는 구조적 결함 확인(2026-09-07에 경과봉 decay는
#       도입했으나 상한 자체는 그대로였음). 신규 상수 HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_
#       UPTREND_PCT(2.2%) 추가 - uptrend_continuation 경로에서만 상한을 완화하고, 신선한
#       크로스 경로는 기존 1.1%를 그대로 유지해 저품질 스파이크 추격은 계속 차단한다.
#       r003 check_buy_condition_1min_hybrid_trigger/g003 _sim 동시 반영, g003
#       --date 20260918 백테스트로 검증 후 라이브 반영 예정(r003/g003 Update log 참조).
#     impact: common (r003 실전/g003 백테스트 공용)
#     compatibility: backward-compatible (신선한 크로스 경로는 기존과 동일, uptrend_
#       continuation 경로에서만 갭 상한이 완화되어 그 경로의 매수 빈도가 늘어날 수 있음)
# - [2026-09-18] type=fix owner=claude
#     summary: 사용자가 183300 코미코 실매매 사례(08:27:28 매수 신호/주문 제출 ->
#       08:28:29 최종 체결, 61초 소요)를 보고 "매수는 걸었는데 체결이 늦은 것 아니냐"고
#       질의, 로그 분석 결과 신호/주문 자체는 제때(08:27:26봉 확정 직후) 나갔지만 체결
#       확인 경로가 느렸던 것으로 확인됨: (1) 08:27:28 지정가(매수1호가 30,450) 제출 ->
#       (2) 다음 ORDER_STATUS_POLL_INTERVAL_SECONDS(15초) 폴링(08:27:47, 19초 경과)에서야
#       미체결 확인 + BUY_ORDER_REPRICE_AFTER_SECONDS(10초) 초과로 재주문(취소) 트리거 ->
#       (3) 다시 다음 15초 폴링(08:28:09)에서야 취소 확인 + 재주문(매수1호가 30,600) ->
#       (4) 다시 다음 15초 폴링(08:28:29)에서야 체결 확인. 즉 취소확인/재주문/체결확인
#       3단계가 전부 refresh_pending_orders()의 전역 15초 폴링 간격에 순차적으로 걸려
#       누적 지연됨(2026-09-14에 메인 루프 자체는 즉시 재시작하도록 바꿨으나 미체결
#       주문 상태 폴링은 별도 타이머라 그 개선 혜택을 못 받고 있었음). 미결 주문 개수는
#       보통 0~1건이라(활성 종목 50개를 매턴 조회하는 현재가 폴링과 달리) KIS TPS 여유가
#       있다고 보고 15->5초로 단축 - 3단계 누적 지연이 최대 약 45초 이내로 줄어든다.
#       추가로 BUY_ORDER_REPRICE_AFTER_SECONDS 주석의 "폴링 주기 15초" 표현을 갱신.
#     impact: live (r003 refresh_pending_orders 폴링 간격만 변경, 판정 로직 불변)
#     compatibility: breaking (미결 주문 상태 조회 API 호출 빈도가 최대 3배 증가 -
#       실매매에서 KIS 레이트리밋/오류 발생 여부 관찰 권장, 문제 시 5->10 등으로 조정)
# - [2026-09-18] type=feat owner=claude
#     summary: 사용자 요청 - 신규 진입 매수 1건을 시장가/지정가 두 조각으로 분할 제출.
#       위 183300 코미코 사례처럼 매수1호가 지정가 전량이 급등을 따라가지 못해 체결까지
#       61초 걸리는 경우를 완화하기 위해, 수량의 BUY_SPLIT_MARKET_RATIO(50%)는 즉시
#       체결(시장가, NXT는 시장가 미지원이라 매도1호가 크로싱으로 대체)로 진입 타이밍을
#       확보하고 나머지는 기존과 동일하게 매수1호가 지정가(+추격 재주문)로 평균 매수가를
#       낮춘다. 수량이 1주뿐이면 분할해도 지정가 쪽에 남는 게 없어 무의미하므로 그대로
#       전량 시장가 1건으로 즉시 매수(r003 place_buy_order 참조). 피라미딩(불타기)
#       추가매수는 이번 변경 대상에서 제외 - 기존과 동일하게 매수1호가 지정가 단일
#       주문 유지.
#     impact: live (r003 place_buy_order/refresh_pending_orders - 신규 진입 매수만
#       해당, 피라미딩/매도는 불변)
#     compatibility: breaking (신규 진입 시 브로커 주문이 1건->최대 2건으로 늘어남,
#       평균 체결가/체결 속도가 달라짐 - 실매매로 체결가 개선 여부 확인 권장)
# - [2026-09-14] type=fix owner=claude
#     summary: 메인 루프 턴 간격을 LIVE_PRICE_POLL_INTERVAL_SECONDS(10초) 벽시계 정렬
#       틱(_next_aligned_tick/_sleep_until_next_tick, 00/10/20/.../50초)에서 분리 -
#       사용자가 실매매 로그(13:19:40 턴 종료 -> 13:19:50 다음 턴 시작)를 보고 한 턴의
#       종목 순회가 1초도 안 걸려 끝났는데도 다음 정렬 틱까지 최대 10초를 그냥 대기하는
#       현상을 지적, 즉시 재시작하도록 요청. r003의 턴 종료부(while True 루프 최하단)
#       가 이제 신규 MAIN_LOOP_MIN_CYCLE_SECONDS만 최소 바닥으로 보장하고 나머지는
#       그대로 다음 턴을 시작한다(r003 Update log 참조). LIVE_PRICE_POLL_INTERVAL_SECONDS
#       자체는 삭제하지 않음 - 정규장/NXT 외 유휴 대기, 동시호가 구간 대기 등 "턴을
#       아예 안 도는" 구간에서는 여전히 이 값으로 폴링한다.
#     impact: live
#     compatibility: breaking (활성 종목이 있을 때 신규 진입/청산 재평가 빈도가 종목당
#       API 왕복시간(관측상 수백 ms) 수준으로 크게 증가 - 이전엔 최소 10초 고정 간격
#       이었음. 브로커(KIS) API 호출 빈도가 늘어나므로 레이트리밋 여부 관찰 필요,
#       필요시 MAIN_LOOP_MIN_CYCLE_SECONDS를 올려 다시 늦출 수 있음)
# - [2026-09-14] type=feat owner=claude
#     summary: 사용자 요청 ①②(실거래 141건 분석 기반, 2026-09-13 세션에서 도출) -
#       (①) 신규 필수 게이트 STOCH_D_BUY_MIN=50.0 추가: 실거래 로그에서 매수 시점
#       %D<50 구간 승률 24-33%(n=64) vs %D 50-80 구간 42%(n=50)로 격차가 뚜렷했음에도
#       기존 _gate_stochastic_buy_signal은 %K만 확인하고 %D 자체엔 하한이 전혀 없었음.
#       (②) 신규 필수 게이트 DI_SPREAD_MIN_REQUIRED=10.0 추가: DI스프레드(+DI--DI)
#       0-10구간 25.6%(n=39) vs 10-20구간 36.4%(n=33) vs 20+구간 62.5%(n=8, 표본작음)
#       - 가장 나쁜 0-10구간만 확실히 배제하는 보수적 임계치로 시작(백테스트로 재검증
#       후 필요시 상향 예정). 기존 R77-D의 ADX_DI_SCORE_*(가점, adx_strength 룰)는
#       그대로 유지 - 이건 별도의 필수 게이트로 얹는 것이라 서로 배타적이지 않음.
#     impact: common (r003 실전/g003 백테스트 공용 - BUY_GATE_CONDITIONS 8개->10개
#       (①은 기존 stochastic_buy_signal 게이트 내부에 조건 추가, ②는 신규 게이트 1개))
#     compatibility: breaking (매수 판정 결과가 바뀜 - 진입 승인율이 낮아질 것으로 예상,
#       백테스트로 거래빈도/승률 트레이드오프 검증 필요)
# - [2026-09-13] type=refactor owner=claude
#     summary: "dead config 정리"(사용자 요청) - 코드 전체(r002/r003/g001~g003) 실제
#       호출부를 grep으로 검증해 완전히 도달 불가능하거나 참조되지 않는 설정 31개 삭제.
#       (1) 근접교차 ARM/Early + 가격선행돌파 클러스터(ENABLE_NEAR_CROSS_ARM,
#       NEAR_CROSS_ARM_GAP_MAX/MA_RISE_MIN/EXPIRE_BARS, ENABLE_EARLY_NEAR_CROSS_ENTRY,
#       NEAR_CROSS_EARLY_GAP_MAX/MA_RISE_MIN, EARLY_NEAR_CROSS_ALLOWED_START/END/
#       ALLOW_NXT/MIN_VOLUME/MIN_VOL_MA/MIN_TURNOVER_KRW, ENABLE_PRICE_LEAD_BB_BREAKOUT,
#       PRICE_LEAD_BREAKOUT_MIN_SCORE/MIN_ADX/ALLOW_OVERBOUGHT) - 이 값들을 쓰는 함수
#       (r002._near_cross_momentum_flags/_passes_early_near_cross_liquidity,
#       g003.is_early_near_cross_allowed/_price_lead_breakout_context_sim)가 코드
#       어디서도 호출되지 않아 함수 자체를 통째로 삭제. (2) 강추세 과열우회 클러스터
#       (ENABLE_STRONG_TREND_OVERBOUGHT_BYPASS, STRONG_TREND_OVERBOUGHT_MIN_ADX/
#       MIN_SCORE/MIN_VOL_RATIO) - 도입만 되고 실제 배선된 적 없음. (3) 단독 항목:
#       BB_SQUEEZE_MIN_WIDTH_PCT/ENABLE_STRICT_MA5_BB_GOLDEN_CROSS/MFI_BUY_MIN/
#       POST_BUY_BB_DROP_POLLS/RSI_BUY_MIN/RSI_BUY_MAX/SAME_DAY_MIN_BARS/
#       SIM_WARMUP_TAIL_BARS/STAGED_TP3_PCT(r003 임포트만 되고 미사용) - 옛 구현이
#       현재의 BuyScoreRule/BuyGateCondition 객체 구조로 리팩터링되며 남은 잔재.
#       (4) R004_EXPORT_TOP_N - 이름/주석상 "r004 export를 상위 N개로 제한"한다고
#       설명돼 있었으나 실제 export 호출부(g002_data_scan_trade_candidates.py)가 이
#       상수를 전혀 참조하지 않아 한 번도 동작한 적 없는 죽은 설정이었음(발견 계기:
#       오늘 앞서 이 값을 20->100으로 바꿨던 변경이 실제로는 아무 효과가 없었고,
#       같은 날의 g002 max_picks 50->100 변경이 진짜 원인이었음을 재검증 중 확인).
#       risk_profiles/*.json에서 이름으로 참조하는 STOP_LOSS_EARLY_PERCENT/
#       STOP_LOSS_MIN_HOLD_SECONDS는 소스코드 grep만으로는 죽은 것처럼 보이지만
#       실제로는 살아있어 삭제 대상에서 제외(교차검증으로 확인).
#     impact: common (live/sim 양쪽에서 삭제, 동작 변화 없음 - 전부 도달 불가능하거나
#       미배선 상태였음)
#     compatibility: backward-compatible (죽은 코드 제거만, 실행 경로/판정 로직 변화 없음)
# - [2026-09-09] type=fix owner=claude
#     summary: HYBRID_1MIN_TRIGGER_LOOKBACK_BARS 3->8 (사용자 요청 - 452190 한빛레이저
#       사례). 3분봉은 11:54~11:58에 골든크로스 확정(score 17/22)했는데 12:07~12:30
#       폭등 구간(4,655->5,160) 전체가 1분 트리거의 1MIN_NO_BB_MID_GOLDEN_CROSS로 100%
#       리젝됨 - 돌파는 12:07~09에 발생했으나 그 후 가격이 BB중간선 위에서 계속 강하게
#       올라 "크로스 시점"이 룩백창(3->5봉) 밖으로 벗어나 역설적으로 추세가 강하고
#       오래갈수록 못 통과하는 구조였음. r003 check_buy_condition_1min_hybrid_trigger의
#       uptrend_continuation 예외 추가(r003 Update log 2026-09-09 참조)와 별개의 보완책 -
#       룩백 자체도 넓혀 "완화"는 하되(무한정은 아니고 BB_GAP_CEILING_PCT가 추격 상한 유지)
#       비교적 흔한 케이스는 uptrend_continuation 예외까지 갈 필요 없이 룩백만으로 해결.
#     impact: live/sim
#     compatibility: backward-compatible (룩백 창만 넓어져 매수 빈도가 소폭 늘 수 있음)
# - [2026-09-08] type=fix owner=claude
#     summary: PRE_CROSS_ACCUM_VOL_RATIO_MIN 0.8->0.6 (매집봉 판정 거래량 기준을
#       VOL_MA20의 80%->60%로 완화). 20260908 실매매 로그 분석(사용자 요청) 결과
#       NO_PRE_CROSS_ACCUM_BAR 리젝이 052690 한전기술 한 종목에서만 46건 발생 - ATR
#       필터(13건)보다 더 자주 걸렸고, 그중 11:55 케이스는 close_cross=True score=18/22
#       (당일 최고점수) + vol_ratio=2.13으로 완전히 유효한 확정 크로스였는데도 "BB중간값
#       포함+양봉+거래량≥VOL_MA20*0.8" 3중 AND 매집봉 조건 하나를 8봉 룩백 내에서
#       충족 못 해 리젝됨. 매집봉 자체가 없었던 게 아니라 거래량 기준(0.8배)만 근소하게
#       충족 못한 것으로 보여, 룩백 확대보다 거래량 기준을 완화하는 쪽을 선택.
#     impact: live/sim (r002 _gate_pre_cross_accumulation_bar, g003도 동일 상수 재사용)
#     compatibility: backward-compatible (필터 자체는 유지, 통과 폭만 넓어짐 - 매수 빈도
#       소폭 증가 예상)
# - [2026-09-08] type=fix owner=claude
#     summary: MIN_ENTRY_ATR_TO_TP1_RATIO 0.3->0.2 (진입 최소 ATR% 하한 0.90%->0.60%).
#       20260908 실매매 로그 분석(사용자 요청) 결과 LOW_ENTRY_ATR 매수거부가 하루 100건
#       발생, 이 중 절대다수(052690 한전기술 score 17/22, vol_ratio 1.96 등 포함)가 나머지
#       8개 게이트(BB돌파/매집봉/캔들/스토캐스틱 등)와 점수 기준(BB_BUY_SCORE_THRESHOLD=10)
#       을 모두 충족한 상태였고 오직 ATR 필터에만 걸려 리젝됨 - 즉 "불안정해서 걸러진" 게
#       아니라 "저변동성 장세에서 정상 신호까지 과도하게 걸러진" 경우가 대부분이었음.
#       052690은 실제로 이날 ATR이 0.59~0.72% 구간을 오갔고(간헐적으로만 0.9% 상회 -
#       그 순간에만 매수 2건 체결, 결과 -0.94%/+0.95%로 TP1 3.0% 근처도 못 감), 반면
#       024060/001820/083650 등 진짜 저변동 종목(ATR 0.23~0.41%)은 새 하한(0.6%)에서도
#       여전히 차단됨 - 필터 완전 제거가 아니라 임계값만 완화.
#     impact: live/sim (r002 _gate_min_entry_atr_volatility, g003도 동일 상수 재사용)
#     compatibility: backward-compatible (필터 자체는 유지, 통과 폭만 넓어짐 - 매수 빈도
#       증가 예상되므로 며칠 실매매 결과 재확인 권장)
# - [2026-09-07] type=feat owner=claude
#     summary: ACTIVE_WATCHLIST_SIZE 50->20, ENABLE_WATCHLIST_ROTATION(신규, 기본 False),
#       R004_EXPORT_TOP_N(신규, 20) 추가 (사용자 요청). 20260907 실매매 로그 실측 결과
#       active=50종목 1턴(전체 순회)이 평균 30초 걸렸는데(종목당 0.6초, API 왕복 지연이
#       지배적이라 서버 연동 구조상 종목당 처리시간 자체는 단축이 어려움 - 진짜 줄이려면
#       비동기/병렬 폴링으로 아키텍처를 바꿔야 하는데 KIS API 자체의 초당 호출 제한(TPS)에
#       걸려 효과가 제한적이고 리스크도 커서 이번엔 보류), LIVE_PRICE_POLL_INTERVAL_SECONDS
#       (10초) 설계 목표의 3배로 밀려 있었음. active_set을 20으로 줄여 1턴을 설계 목표에
#       가깝게 되돌리는 대신, 그동안 active_set 크기를 지탱하던 backup_pool 로테이션
#       (TIME_LIMIT 교체) 자체를 ENABLE_WATCHLIST_ROTATION=False로 정지 - active_set이
#       처음 로드된 상위 20개로 장중 고정된다. g002가 내부적으로 선별하는 후보 수(max_picks,
#       100)는 그대로 두되 r004로 내보내는 건 R004_EXPORT_TOP_N(20)개로 제한(g002/g005
#       Update log 참조) - active_set 크기와 r004 export 개수를 일치시켜 backup_pool이
#       항상 비도록 함.
#     impact: live
#     compatibility: breaking (감시 종목 수가 50->20으로 줄어 매수 기회 자체가 줄어들 수
#       있음 - 대신 종목당 재평가 주기가 30초->~12초로 빨라져 개별 종목의 신호 포착
#       정확도는 올라감. 로테이션 재활성화가 필요하면 ENABLE_WATCHLIST_ROTATION=True로
#       되돌리면 기존 동작 그대로 복원됨)
# - [2026-09-07] type=fix owner=claude
#     summary: HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR(0.15)/_CEILING_PCT(1.1) 신규
#       추가. 20260907 실매매 로그 분석(사용자 요청) 결과 388050 지투파워 09:16(정상
#       골든크로스, 매집봉 미충족으로 리젝) -> 09:18~09:24(눌림으로 크로스 무효화) ->
#       09:26~09:28(재돌파했으나 BB_MID가 후행지표라 못 따라와 갭 0.84~0.99%가 고정
#       상한 0.5%를 넘어 CHASE_BUY_BB_GAP로 매 폴링 리젝) 패턴, 025980 아난티 13:45~13:50도
#       동일 패턴(매집봉/거래량/갭이 서로 다른 시점에 하나씩만 걸려 결국 룩백 만료)으로
#       확인됨 - 갭 상한이 고정값이라 크로스가 유효하게 지속 중(BB_MID 위 연속 유지)이어도
#       시간이 지날수록 무조건 리젝 확률만 높아지는 구조적 결함. 크로스 이후 경과봉 수만큼
#       상한을 소폭(봉당 0.15%p) 완화하되 CEILING_PCT(1.1%)로 무한 완화는 방지.
#     impact: live/sim (r003 check_buy_condition_1min_hybrid_trigger, g003
#       check_buy_condition_1min_hybrid_trigger_sim 동시 적용 - 상세는 각 파일 Update log)
#     compatibility: backward-compatible (경과봉=0, 즉 신선한 크로스는 기존 0.5% 그대로;
#       경과봉이 있을 때만 상한이 완화되어 매수 빈도가 소폭 늘 수 있음)
# - [2026-09-06] type=fix owner=claude
#     summary: UPTREND_CONT_SLOPE_MIN_PCT(-0.05) 신규 추가. 8/17~9/4 로그(사용자 요청 -
#       매수/매도 조건 세분화 검토) 분석 결과, r002의 _evaluate_bb_mid_cross() uptrend_
#       continuation(크로스 이벤트 없이 지속 추세만으로 진입 허용하는 경로) 판정이
#       bb_slope_pct > 0.0을 요구하는데, 이는 필수 게이트 bb_slope_rising이 요구하는
#       BB_SLOPE_MIN_PCT(-2.0%)보다 훨씬 엄격함 - 대체(보조) 진입 경로가 필수 게이트보다
#       더 까다로운 역전 상태였음. 실제 196170 알테오젠 2026-08-21 09:33~09:40 사례에서
#       ADX 80대, +DI>>-DI, MA5 상승 등 나머지 조건은 전부 충족했는데 bb_slope_pct가
#       BB_MIDDLE의 후행지표 특성상 0 근방(-0.016%~0.071%)에서 노이즈로 진동하는 바람에
#       09:33에 리젝, 09:36에야 순전히 노이즈 타이밍으로 통과 - 매수 기회가 몇 분 단위로
#       운에 좌우됨. r002의 하드코딩된 0.0 리터럴을 이 상수로 교체.
#     impact: common
#     compatibility: backward-compatible (조건이 소폭 완화되어 uptrend_continuation 경로
#       진입 빈도가 약간 늘어날 수 있음 - 나머지 7개 필수조건(ADX>=30, DI 우세, 3/5봉 BB
#       위 유지, MA5 상승 등)은 그대로라 저품질 신호가 크게 늘지는 않을 것으로 판단)
# - [2026-09-05] type=fix owner=claude
#     summary: ACTIVE_WATCHLIST_HARD_TIME_LIMIT_MINUTES(120) 신규 추가. 8/17~9/4 로그 분석
#       결과, ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES(60분)가 실제 가격이 박스권(정체)인지와
#       무관하게 무조건 탈락시키는 순수 타이머였던 것이 확인됨(사용자는 "박스권 횡보 시
#       교체"로 이해하고 있었으나 실제로는 박스권 판정 로직 자체가 없었음). 이 기능이 도입된
#       8/31 이후 실거래 체결이 8/31=1건, 9/1=1건, 9/2=0건, 9/3=1건, 9/4=1건으로 급감(그 이전
#       8/18~8/28은 일 30~200건대) - 60분 타이머가 아직 박스권을 벗어나 방향성을 만들어가는
#       중인 종목까지 무차별적으로 backup과 교체해버려 매수 직전 종목이 자주 감시 대상에서
#       빠진 것으로 추정됨. r003의 _rebalance_active_watchlist()에서 60분 경과 시점부터는
#       _is_box_range_hold_zone()(기존 매도측 박스권 판정 재사용)으로 실제 정체 여부를
#       확인해 박스권이 확인된 종목만 교체하고, 아직 방향성이 살아있는 종목은 이 상한(120분)
#       까지 계속 감시하도록 변경 - 상한은 9/3 backup_pool 고갈 사고 재발 방지용 안전장치.
#     impact: live
#     compatibility: backward-compatible (신규 상수만 추가, 실제 판정 변경은 r003 쪽과 짝을
#       이뤄야 발동)
# - [2026-08-31] type=feat owner=claude
#     summary: ACTIVE_WATCHLIST_SIZE(50)/ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES(60) 신규 추가.
#       g002 스캐너 선정 종목 수를 50->100으로 확대(같은 날짜 g002 Update log 참조)하면서,
#       r003 메인 루프가 매 틱 watch_map 전종목을 완전 순차(동시성 없음) API 폴링하는 구조상
#       종목 수에 선형 비례해 부하가 늘어 틱 주기(10초)를 넘길 위험이 있어, r003에
#       active_set(실시간 폴링 상한)/backup_pool(대기, 미폴링) 분리를 도입하며 함께 추가.
#       상세 설계는 r003 Update log 2026-08-31 및 _rebalance_active_watchlist() 참조.
#     impact: live
#     compatibility: backward-compatible (신규 상수만 추가, 기존 상수/동작 변화 없음 - 실제
#       분리 동작은 r003 쪽 변경이 짝을 이뤄야 발동)
# - [2026-08-28] type=feat owner=claude
#     summary: BUY_ORDER_REPRICE_AFTER_SECONDS/MAX_ATTEMPTS/MAX_CHASE_PCT 신규 추가 -
#       place_buy_order()가 매수1호가(최우선 매수호가) 순수 지정가로만 주문을 내다 보니
#       (매도 정규장은 이미 시장가를 쓰는 것과 비대칭) 방금 돌파한 종목처럼 가격이 계속
#       올라가는 경우 영원히 미체결로 남는 문제 발견(000720 현대건설 2026-08-28 13:14
#       사례: 500초 넘게 미체결, [BUY STALE] 경고만 반복되고 실제 조치는 없었음). 10초마다
#       취소 후 더 공격적인 가격(1차 신선한 매수1호가 -> 2차 매도1호가 -> 최종 시장가)으로
#       최대 3회 재주문하되, 최초 신호가 대비 0.5%를 넘는 후보가는 추격을 포기하고 취소만
#       한다(r002 추격매수 방지 게이트와 동일 철학 - 재시도가 그 게이트들을 무력화하지
#       않도록). 상세 배경/설계는 이 상수 정의부 및 r003 Update log 2026-08-28 참조.
#     impact: live
#     compatibility: backward-compatible (미체결 상태에서만 개입, 체결/취소 판정 로직
#       자체는 그대로 - 다만 place_buy_order()가 내는 지정가 자체는 항상 매수1호가라는
#       근본 동작은 안 바뀜, 이건 미체결 "이후" 대응만 추가한 것)
# - [2026-08-28] type=feat owner=claude
#     summary: ENABLE_1MIN_TRIGGER_3MIN_CONTEXT 신규 추가 - 1분봉 골든크로스를 트리거로,
#       3분봉(BB기울기/다운트렌드/매집봉/BB상단여유/스토캐스틱/윌리엄스/유동성+가점)을
#       컨텍스트 필터로 쓰는 하이브리드 3번째 경로. 403870 HPSP 2026-08-28 실매매
#       로그에서 3분봉 크로스 확정 지연으로 인한 반복 추격매수 반려(09:09, 09:27) 확인 후
#       추가. 상세 배경/설계는 이 플래그 정의부 및 r002 Update log 2026-08-28 참조.
#       1차 검증(트리거를 check_buy_condition_1min 그대로 재사용)은 매수 0건으로 실패 -
#       원인은 순서가 아니라 1분봉 단일봉 크로스 판정(룩백 없음)이었음. HYBRID_1MIN_TRIGGER_*
#       전용 상수(룩백 3봉 + 완화된 캔들/추격 문턱) 신설 후 2차 검증에서 09:07 매수/09:25
#       매도 +2.19% 1건 체결 성공(상세는 이 플래그 정의부 참조) - 단 n=1 검증이라 기본값은
#       False 유지, 다일자/다종목 백테스트로 일반화 확인 후 전환 권장.
#     impact: common
#     compatibility: backward-compatible (기본값 False, 기존 경로 동작 변화 없음; 다른
#       날짜/종목 다건 백테스트로 일반화 여부 확인 전까지 opt-in 유지)
# - [2026-08-25] type=fix owner=claude
#     summary: 2026-08-25 실매매 로그 분석 결과 필수조건 2-b(직전 매집봉 확인)가 크로스 이후
#       다운스트림 리젝의 압도적 1위(정규화 기준 1,284건/블록시간 2,385분, 2위인 CHASE_BUY_BB_GAP
#       986분의 2.4배)로 확인됨 - 09:00부터 정규장 중반 상승장 전환 이후까지 하루 종일 매수가
#       6건뿐이었던 핵심 원인. 원인은 "그 봉의 저가~고가가 BB중간값 포함 AND 양봉 AND 거래량>
#       VOL_MA20"을 같은 한 봉에서 동시에 요구하는 3중 AND 조건이 너무 희귀한 조합이라는 점.
#       (1) PRE_CROSS_ACCUM_LOOKBACK_BARS 5->8봉으로 재확장, (2) 거래량 기준을 "VOL_MA20 초과"에서
#       "VOL_MA20의 PRE_CROSS_ACCUM_VOL_RATIO_MIN(0.8) 이상"으로 완화(신규 상수) - 2026-08-21에
#       이미 "직전봉 대비"->"VOL_MA20 대비"로 한 차례 완화했으나 그 후로도 여전히 최대 병목으로
#       남아있어 추가 완화. CANDLE_GAIN_MIN_PCT도 0.0%->-0.1%로 소폭 완화 - 라이브 폴링 시점의
#       미세한 틱 노이즈로 정상 돌파 구간에서도 순간적으로 음전(-0.01~-0.2%대)되어 CANDLE_NOT_BULLISH로
#       리젝되는 사례(669건, 블록시간 586분)를 완화하기 위함.
#     impact: common
#     compatibility: breaking (매수 필수조건 2-b/3 통과 기준이 완화되어 매수 빈도가 늘어날 것으로 예상;
#       r007 --date 20260825 백테스트로 리젝 사유 재분포 확인 필요)
# - [2026-08-21] type=fix owner=copilot
#     summary: PRE_CROSS_ACCUM_LOOKBACK_BARS 3->5, 매집봉 거래량 증가 판정 기준을 "직전봉 대비"에서
#       "VOL_MA20(20봉 평균) 대비"로 변경(r005 Update log 참조). 어제(2026-08-20) 신규 추가된
#       필수조건 2-b가 257720 실리콘투 2026-08-21 09:57 BB중앙선 상향 돌파 사례를 막은 것을 사용자가
#       발견 - 돌파 직전 3봉(09:48/09:51/09:54) 각각의 거래량을 직전봉과 비교하면 09:48봉이 그 직전
#       09:45봉(장중 거래량 스파이크)보다 낮게 나와 탈락했는데, VOL_MA20 대비로는 낮지 않아 정상
#       매집 국면이 직전봉 대비 잡음 때문에 오탈락한 사례로 판단됨. 사용자 요청으로 (1) 판정 기준을
#       변동성이 큰 "직전 1봉"에서 안정적인 "20봉 평균" 대비로 변경, (2) 룩백도 3->5봉으로 확장.
#     impact: common
#     compatibility: breaking (매수 필수조건 2-b 통과 기준이 완화되어 매수 빈도가 다시 늘어날 수 있음)
# - [2026-08-20] type=feat owner=copilot
#     summary: ENABLE_PRE_CROSS_ACCUM_BAR_CHECK/PRE_CROSS_ACCUM_LOOKBACK_BARS 신규 추가 -
#       BB 중앙선 상향 돌파 직전에 매집(거래량 증가) 양봉이 있었는지 확인하는 필수조건.
#       사용자 요청으로 run_buy_condition_pipeline_comment에 신규 필수조건으로 삽입(r005
#       Update log 참조). 매수 빈도 급감을 막기 위해 직전봉 1개가 아니라 최근
#       PRE_CROSS_ACCUM_LOOKBACK_BARS(3)개 봉 중 하나라도 만족하면 통과시킴.
#     impact: live
#     compatibility: breaking (신규 필수조건 추가로 매수 빈도가 줄어들 수 있음; 플래그로 즉시 롤백 가능)
# - [2026-08-17] type=fix owner=copilot
#     summary: BB_SLOPE_MIN_PCT(-2.0%) 신규 추가 - run_buy_condition_pipeline_comment의
#       필수조건 1(BB 기울기) 하드코딩값 -0.7%를 상수로 분리하며 완화. 사용자 제공 실제 차트
#       (2026-08-14 삼성전기)에서 급등 후 되돌림 국면 중 가격/스토캐스틱/윌리엄스%R은 이미
#       반전됐는데 60분 BB기울기만 아직 음(-)이라 매수가 막히는 사례를 확인해 조정.
#     impact: common
#     compatibility: breaking (매수 필수조건 1 통과 기준 완화)
# - [2026-08-17] type=fix owner=copilot
#     summary: ENABLE_1MIN_GOLDEN_CROSS_BUY True->False - 사용자가 원래 의도한 매수 설계
#       ("3분봉 BB 중앙선 골든크로스 + 스토캐스틱 패스트/윌리엄스 %R 매수신호 확인")를
#       확인해보니 1분봉 골든크로스 경로는 스토캐스틱/윌리엄스를 전혀 참조하지 않는 것으로
#       드러남(2026-06-28 리팩토링 때 매수 스코어링에서 제거된 뒤 복구되지 않았음). 3분봉
#       다중필터 경로(check_buy_condition)로 되돌리고, 그 안에 스토캐스틱/윌리엄스 %R
#       매수신호 필수조건을 신규 추가함(r005 Update log 참조).
#     impact: live
#     compatibility: breaking (매수 판단 타임프레임이 1분봉 -> 3분봉으로 되돌아가고, 매수
#       빈도/타이밍이 최근 한 달간 라이브 운용과 크게 달라짐)
# - [2026-07-25] type=feat owner=copilot
#     summary: 매매 파이프라인 개선 4건 반영 (r006 Update log에 상세 배경 기술).
#       (1) CANDLE_CONFIRM_DELAY_SECONDS 신규 - 봉 마감 직후 거래소 API 지연(1~2초)으로
#       인한 미확정 데이터 사용 방지. (2) TP1_ATR_MULTIPLIER 신규 - 1차 익절 목표를
#       고정 1.0%에서 max(1.0%, ATR*1.2) 동적 목표로 전환. (3) STAGED_TP3_PCT는 더 이상
#       r006 라이브 매매의 고정 청산 트리거로 쓰이지 않음 - 1,2차 익절 이후 잔량은
#       기존 트레일링 스탑 로직에 위임(3단계=트레일링 30%). r007 시뮬레이션 참고용으로만 유지.
#       (4) ENABLE_PYRAMIDING/PYRAMID_TRIGGER_PNL_PCT 신규 - 추세 지속 확인 시 1회 추가진입(불타기).
#     impact: live
#     compatibility: breaking (익절 3단계/매수 확정 타이밍/봉 확정 판정이 변경됨; 플래그로 즉시 롤백 가능)
# - [2026-07-22] type=feat owner=copilot
#     summary: 개장 초반(09:01~09:05류) 갭/거래량폭발 라이브 게이트용 설정값 신규 추가
#       (ENABLE_OPENING_GAP_VOLUME_GATE 등). r002 스캐너는 전일 종가 기준 데이터라 당일
#       시가 갭이나 거래량 급변을 반영하지 못하는 한계가 있어, r006 실시간 매수 직전에
#       한 번 더 검증하기 위함 (r006 Update log 참조).
#     impact: live
#     compatibility: backward-compatible (기본 True, 플래그로 즉시 롤백 가능)
# - [2026-07-21] type=fix owner=copilot
#     summary: STAGED_TP3_PCT 1.8%->3.0% - 사용자 요청으로 3단계 익절 중 3차(잔량 전체 청산) 임계값만
#       확장(1차 +1.0%/2차 +1.4%는 유지). 최대 익절폭이 기존 대비 넓어짐.
#     impact: live
#     compatibility: backward-compatible (값만 변경, 로직 동일)
# - [2026-07-21] type=feat owner=copilot
#     summary: (1) 1분봉 BB 중간값 골든크로스 매수 신규 도입 - ENABLE_1MIN_GOLDEN_CROSS_BUY 플래그로
#       기존 3분봉 다중필터 매수 파이프라인과 전환 가능. (2) 익절을 3단계 분할청산(40%/30%/30%,
#       +1.0%/+1.4%/+1.8%)으로 변경 - ENABLE_STAGED_TAKE_PROFIT 플래그로 기존 2단계(50%@1.0%,
#       전량@2.0%)와 전환 가능. 상세 배경은 r006 Update log 참조.
#     impact: live
#     compatibility: breaking (매수 판단 타임프레임/신호 및 익절 단계가 모두 변경됨; 플래그로 즉시 롤백 가능)
# - [2026-07-21] type=fix owner=copilot
#     summary: BB_BUY_SCORE_THRESHOLD 8->10 - logs/20260720 r006 실매매 로그 분석(16건) 결과 score=8
#       매수 6건 전부 손실(-18,975원, 당일 총손실의 87%), score=9 4건도 순손실(-2,295원). score>=10
#       매수만 순이익 포함(순 -550원). 2026-07-01에 9->8로 완화했던 조치가 이번 표본에서는 저품질
#       진입을 다수 통과시켜 역효과. 표본 1일뿐이라 추가 로그로 지속 검증 필요.
#     impact: live
#     compatibility: breaking (매수 빈도 감소 예상)
# - [2026-07-20] type=fix owner=copilot
#     summary: LIVE_PRICE_POLL_INTERVAL_SECONDS 20->10 원복 - 직전 커밋에서 사용자가 실제로 알고
#       있던 기존 값(10초)을 확인 없이 20초로 임의 변경했던 것을 사용자 확인 후 되돌림. r006의
#       틱 정렬 로직(_next_aligned_tick/_sleep_until_next_tick)은 이 상수를 그대로 읽어 매분
#       00/10/20/30/40/50초(6틱/분)에 정렬되도록 자동 적용되므로 이 값 변경 외 다른 수정은 불필요.
#     impact: live
#     compatibility: backward-compatible
# - [2026-07-20] type=fix owner=copilot
#     summary: LIVE_PRICE_POLL_INTERVAL_SECONDS 10->20 - r006 메인 루프를 매분 00/20/40초
#       벽시계 정렬 틱(3틱/분)으로 고정 스케줄링하도록 바꾸면서, 20초 배수가 되도록 조정.
#       (r006 Update log 참조)
#     impact: live
#     compatibility: backward-compatible
# - [2026-07-18] type=fix owner=copilot
#     summary: BB_UPPER_GAP_MIN_PCT 0.5% → 0.25% 하향 조정 - 저변동성 종목 오후 구간에서 BB 폭 좁아져 매수 차단 완화.
#     impact: common
#     compatibility: backward-compatible
# - [2026-07-16] type=fix owner=copilot
#     summary: 매도/손절 조건 완화 - r007로 D일 picks를 D+1일 실데이터에 매매 시뮬레이션(look-ahead 없는 교차일 백테스트)한
#       결과, 급락 후 반등을 놓치는 조기청산 비중이 높게 나타나 값 조정.
#       (1) HARD_STOP_LOSS_PCT 1.2%->1.7%, HARD_STOP_MIN_HOLD_SECONDS 180->240 (58건 중 손절 후 34.5%가 +2%까지 회복)
#       (2) BREAKEVEN_FAIL_GIVEBACK_PCT 2.0%->2.8%, CONFIRM_SECONDS 120->150 (14건 중 92.9%가 본전 이상 회복)
#       (3) POST_BUY_BB_DROP_PCT 1.0%->1.4%, CONFIRM_SECONDS 60->90 (12건 중 58.3%가 +2%까지 회복, 평균 최대회복 +4.17%)
#       (4) TP_EXTENSION_TRAIL_FROM_PEAK 0.4%->0.6% (익절 후 트레일링이 과도하게 타이트하여 수익 조기 제한)
#       동일 근거로 r006/r007의 시그널 매도(STOCH K<D, MACD_HIST 2봉하락) pnl 임계값도 함께 완화함
#       (-0.8%->-1.2%, -0.5%->-0.8%, 해당 파일 Update log 참조).
#     impact: live
#     compatibility: breaking (매도/손절 발동이 이전보다 늦어짐 - 손실 트레이드당 손실폭은 커질 수 있으나 조기청산으로
#       인한 기회손실 감소가 목표)
# - [2026-07-12] type=refactor owner=copilot
#     summary: 매매(손절/익절/트레일링스탑/진입필터 등) 판단에 쓰이는 모든 설정값을
#       파일 상단 "LIVE TRADING TUNABLE PARAMETERS" 섹션 하나로 모아 재배치.
#       값은 전혀 변경하지 않음(재배치만 수행) - 지표 계산 주기/세션 시간/폴링
#       간격/시뮬레이션 전용 값은 하단 별도 섹션으로 분리.
#     impact: common
#     compatibility: backward-compatible (값 변경 없음, 위치만 이동)
# - [2026-07-01] type=fix owner=copilot summary=BB_BUY_SCORE_THRESHOLD 9->8 (실매매 중 유효 반전신호 과잉반려 완화, 093370 14:51 크로스 사례)
# - [2026-06-28] type=refactor owner=copilot summary=(3) BB_BUY_SCORE_THRESHOLD 8->9 (VWAP/거래량방향/BB폭/MA5 신호 추가 점수 체계 반영)
# - [2026-06-28] type=fix owner=copilot summary=(2) NO_TREND_EXIT peak 0.3%->0.6% + 연속HARD_STOP 서킷브레이커 + 당일HARD_STOP 종목 재진입 차단 상수 추가
# - [2026-06-28] type=fix owner=copilot summary=저거래량 추격매수 차단 강화 + 거래량 hard_floor 상향 + 개장 직후 보호 강화 + BB 갭 추격 기준 엄격화
# - [2026-06-26] type=fix owner=copilot summary=BREAKEVEN_FAIL 완화(giveback 2.0%, confirm 120s) + VOLUME_RATIO_MIDDAY 강화(0.50)

"""R76 shared configuration for live trading and simulation.

Risk profile presets:
- Set AUTO_TRADING_RISK_PROFILE to one of conservative|neutral|aggressive.
- Korean aliases are supported: 보수|중립|공격.
- If no profile is specified, neutral is applied by default.
- Preset JSON files live under xgraph/auto_trading/risk_profiles/.

튜닝 가이드:
- 실전 매매(r003_trade_live_execute.py)의 손절/익절/트레일링스탑/진입필터 등
  "매매 판단"에 직접 영향을 주는 값은 전부 아래 "LIVE TRADING TUNABLE
  PARAMETERS" 섹션에 모여 있다. 실전매매 결과를 보고 값을 튜닝할 때는 이
  섹션만 확인하면 된다.
- 지표 계산 주기(봉 개수 등), 세션/시간대, 폴링/백오프 같은 시스템 운영값,
  r007 시뮬레이션 전용 값은 그 아래 별도 섹션에 분리되어 있다.
"""

import json
import os
from datetime import time as dt_time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path / file constants
# ---------------------------------------------------------------------------
DEFINE_TODAY_CODE_PATH = "r004_trade_watchlist_today.txt"
DATA_DIR_NAME = "data"
# ---------------------------------------------------------------------------
# Feature: R004 watchlist pipeline (scanner -> r004 -> r003 live)
# ---------------------------------------------------------------------------
FEATURE_R004_WATCHLIST = True
FEATURE_SCAN_EXPORT_TO_R004 = True
FEATURE_WATCHLIST_RESOLVE_SCAN_PICKS = True

R004_WATCHLIST_FILENAME = DEFINE_TODAY_CODE_PATH
SCAN_PICKS_LEGACY_FILENAME = "picks.txt"
SCAN_PICKS_PREFIX_TEMPLATE = "_{date}_picks.txt"

# [2026-09-13] R004_EXPORT_TOP_N 삭제 ("dead config 정리" 중 발견) - g002가 실제로
# export_picks_to_r004()를 호출하는 지점(g002_data_scan_trade_candidates.py)에서
# 이 상수를 전혀 참조하지 않아, r004에는 항상 max_picks(스캐너 내부 후보 수 상한)
# 전체가 그대로 나갔음(이름/주석은 "상위 N개로 제한"이라고 설명하고 있었지만 실제
# 배선이 안 돼 있던 죽은 설정). 같은 날 ACTIVE_WATCHLIST_SIZE(50)/g002 max_picks(100)
# 원복이 의도한 대로 동작한 건 이 상수 때문이 아니라 max_picks 자체를 50->100으로
# 늘려서였음 - 별도 export 제한이 필요하면 g002 호출부에 picks[:N] 슬라이싱을
# 새로 추가해야 한다.


# =============================================================================
# LIVE TRADING TUNABLE PARAMETERS
#   r003_trade_live_execute.py(실전 매매)가 매수/매도/리스크 판단에 직접
#   사용하는 값 모음. 실전매매 결과를 보고 튜닝할 때는 이 섹션만 수정하면 된다.
#   (g003_trade_simulate_by_date.py도 동일 로직 재현을 위해 이 값들을 그대로 사용)
# =============================================================================

# --- 1. 손절 (Stop Loss) ----------------------------------------------------
TAKE_PROFIT_PERCENT = 0.025  # 고정 익절 비율(+2.5%) - 섹션 2와 함께 관리
STOP_LOSS_PERCENT = -0.021  # 기본 손절 기준(-2.1%)
STOP_LOSS_EARLY_PERCENT = -0.020  # 보유 초기 구간에서 사용하는 완화 손절 기준
STOP_LOSS_MIN_HOLD_SECONDS = 600  # 손절 로직이 본격 적용되기 전 최소 보유 시간(초)
HARD_STOP_LOSS_PCT = 0.017  # 하드스탑 손절 기준 (0.8%->1.2%->1.7%: 백테스트 결과 하드스탑 58건 중 63.8%가 손절 후 본전 이상 회복, 34.5%는 +2%까지 회복 - 급락 후 반등을 놓치지 않도록 완화)
HARD_STOP_MIN_HOLD_SECONDS = 240.0  # 하드스탑 활성화 최소 보유 시간(초) (180->240: 초기 변동성 노이즈에 덜 민감하도록 연장)
# [2026-09-23] 단일 폴링 틱 노이즈로 즉시 손절되는 걸 막기 위한 확인창(ATR_STOP_LOSS와 동일 헬퍼/취지,
# 119850 2026-09-23 09:02 사례+Codex 검토). 0 이하면 기존과 동일(즉시 발동, 디바운스 없음)으로 되돌아간다 -
# 지속 하락 시 확인 시간만큼 체결가가 더 나빠질 수 있는 트레이드오프이므로 손실 상한이 아니라 트리거 지연임에 유의.
HARD_STOP_CONFIRM_SECONDS = 10.0
ATR_STOP_MULTIPLIER = 1.5  # ATR 기반 손절 배수

# [2026-09-07] 진입 최소 변동성(ATR%) 필터. 손절/익절(TP1/ATR_STOP_MULTIPLIER)은 ATR을 쓰면서
# 정작 진입 시점에는 ATR을 전혀 확인하지 않아, ATR이 TP1(STAGED_TP1_PCT=3.0%)에 한참
# 못 미치는 저변동 종목도 그대로 진입 게이트를 통과했다. 일반적인 변동성 필터 관행(진입 시
# ATR이 가격 대비 최소 임계치 이상이어야 함)을 참고해, TP1 대비 ATR 비율 최소 하한을 둔다.
ENABLE_MIN_ENTRY_ATR_FILTER = True
MIN_ENTRY_ATR_TO_TP1_RATIO = 0.2  # ATR% >= STAGED_TP1_PCT * 0.2 (0.3->0.2) 이어야 진입 허용 - 20260908 로그 분석 결과 0.3(0.90%)이 정상 신호까지 과도 차단(위 Update log 참조)

ATR_STOP_CONFIRM_SECONDS = 20.0  # ATR 손절 조건이 이 시간 이상 연속 유지돼야 실제 매도 (033790 피노
# 2026-08-31 13:35 사례: sl=7,434 대비 단 한 틱(7,430, 폴링 1회)만 하회하고 다음 폴링(20초 후)엔
# 이미 7,440으로 회복 - 시장가 즉시 매도 후 몇 분 만에 7,590까지 반등. 순간 틱노이즈로 손절이
# 확정되는 걸 막기 위해 다른 매도가드(POST_BUY_DROP/BREAKEVEN_FAIL/NO_TREND_EXIT)처럼 지속시간
# 확인 게이트 추가 - 다만 ATR_STOP은 자본 보호용 최종 방어선이라 확인시간은 최소(20초=폴링 1~2회)로 짧게.

# --- 2. 익절 / 트레일링 스탑 (Take Profit / Trailing Stop) ------------------
TRAILING_STOP_FROM_PEAK = 0.012  # 트레일링 스탑: 고점 대비 되돌림 허용폭 (1.2%)
ENABLE_TP_EXTENSION_TRAILING = True  # 익절 후 연장 트레일링 기능 사용 여부
TP_EXTENSION_TRAIL_FROM_PEAK = 0.010  # 익절 연장 구간 트레일링 폭 (0.4%->0.6%->1.0%: 1차 익절(+3.0%) 이후 잔량 60%를 관리하는 폭이라 -1.0%로 확대)
ATR_TAKE_PROFIT_MULTIPLIER = 3.0  # ATR 기반 익절 배수

# [2026-08-24] 고점봉 바로 다음 확정봉 반전 매도 - TP_EXTENSION_TRAIL_FROM_PEAK(1.0%)만으로는
# 반응이 느려, 017670 SK텔레콤 2026-08-24 12:40 매매(peak pnl 1.26% -> 실현 0.19%, giveback
# 1.07%p, 거래세 포함 시 순손실)처럼 고점 바로 다음 3분봉이 이미 음봉으로 꺾였는데도 트레일 폭을
# 다 기다리다 익절분 대부분을 반납하는 사례가 있었음. 고점을 만든 확정봉의 바로 다음 확정봉이
# 음봉이면서 고점 대비 이 폭 이상 하락하면 트레일 도달을 기다리지 않고 즉시 매도한다.
ENABLE_PEAK_NEXT_BAR_BEARISH_EXIT = True
PEAK_NEXT_BAR_DROP_PCT = 0.005  # 고점 대비 하락폭 (0.5%)

# 익절 단계 분할청산 (40%/60%, +3.0%/트레일링) - 기존 TP1(40%@1.0%)+TP2(30%@1.4%)+잔량(30%) 대체
# [2026-08-23] 사용자 요청: 급등 구간에서 TP1(+1.0%)이 너무 일찍 40%를 털어 추세를 못 태운다는
# 피드백에 따라 1차 익절 목표를 +1.0%->+3.0%로 상향. 2차 단계는 STAGED_TP2_RATIO=0으로 비활성화하고
# 1차 이후 잔량(60%)은 곧바로 TP_EXTENSION_TRAIL_FROM_PEAK(고점대비 -1.0%) 트레일링에 위임한다.
# False로 설정 시 즉시 기존 2단계 방식으로 롤백된다.
ENABLE_STAGED_TAKE_PROFIT = True
STAGED_TP1_PCT = 0.030   # 1차 익절 기준 (+3.0%, 기존 +1.0%) - 진입 수량의 40% 청산
STAGED_TP1_RATIO = 0.40
STAGED_TP2_PCT = 0.014   # 2차 익절 기준값 (STAGED_TP2_RATIO=0으로 비활성화되어 현재 미사용)
STAGED_TP2_RATIO = 0.00  # 0.30 -> 0.00: 2차 분할청산 비활성화 (1차 이후 잔량은 트레일링으로 일괄 관리)
# 1차 익절 목표를 고정 STAGED_TP1_PCT 대신 종목 변동성(ATR)에 연동해 동적으로 산출한다.
# 목표익절 = max(STAGED_TP1_PCT, (ATR/entry_price) * TP1_ATR_MULTIPLIER)
TP1_ATR_MULTIPLIER = 1.2
# [2026-09-21] 1차 익절 목표 상한 = ATR 익절선(ATR% x ATR_TAKE_PROFIT_MULTIPLIER, 곧 TP_EXTENSION 트레일
# 무장선). 트레일은 peak >= ATR 익절선(3xATR%)에서 무장하는데, TP1이 +3.0% 고정 하한이면 ATR% < 1.0%
# 종목은 트레일이 TP1보다 먼저 무장되어 1차 분할 없이 트레일이 전량 매도한다(023160 태광 2026-09-21:
# 고점 +2.79% < TP1 +3.0%, 트레일 무장 +1.91% -> 18주 전량 매도). 목표 =
# min(max(STAGED_TP1_PCT, ATR%*TP1_ATR_MULTIPLIER), ATR%*ATR_TAKE_PROFIT_MULTIPLIER) 로 두어 TP1이 트레일
# 무장보다 늦어지지 않게 한다(ATR%>2.5% 고변동 종목은 기존 ATR 동적 목표가 그대로 유지됨).
# False면 기존 max(STAGED_TP1_PCT, ATR%*TP1_ATR_MULTIPLIER)로 즉시 롤백.
ENABLE_TP1_CAP_AT_ATR_TP = True
# [2026-09-21] 급등 감지 + 사다리 익절: 1차 익절 시점에 급등 중이면 잔량을 TP1 체결가 기준 +2%/+4%에서 순서대로
# 추가 익절한다(추세를 더 태우기 위해 잔량 전량을 트레일에만 맡기지 않음). 급등 여부는 TP1 시점에 1회 판정.
# 급등 = (속도) 최근 SURGE_LOOKBACK_SECONDS 내 최저가 대비 상승폭 >= max(SURGE_SPEED_MIN_PCT,
#        ATR% x SURGE_SPEED_ATR_MULT) 이면서, (확인) BB 상단 돌파 / 봉 거래량 >= VOL_MA20 x
#        SURGE_VOLUME_RATIO_MIN 중 SURGE_MIN_CONFIRMS개 이상.
# 사다리: TP2 = TP1 체결가 x (1+SURGE_TP2_PCT)에서 진입수량의 SURGE_TP2_RATIO, TP3 = x (1+SURGE_TP3_PCT)에서
# 잔량 전량. 트레일링 스탑/손절은 그대로 병행(사다리 사이에서 되돌림이 나오면 트레일이 잔량을 청산).
# False면 급등 판정/사다리 자체가 비활성화되어 기존(TP1 후 잔량 전량 트레일 위임) 동작과 동일.
ENABLE_SURGE_LADDER_TP = True
SURGE_LOOKBACK_SECONDS = 60.0   # 급등 속도 측정 구간(초) - 라이브 틱 약 10초 간격이라 구간당 5~6표본
SURGE_SPEED_MIN_PCT = 0.008     # 구간 내 최저가 대비 상승폭 절대 하한 (+0.8%)
SURGE_SPEED_ATR_MULT = 1.5      # 상승폭 >= ATR% x 1.5 (변동성 정규화: 원래 잘 흔들리는 종목의 노이즈를 급등으로 오인하지 않음)
SURGE_VOLUME_RATIO_MIN = 1.5    # 확인 신호: 최근 확정봉 거래량 / VOL_MA20 >= 1.5
SURGE_MIN_CONFIRMS = 1          # 속도 외 확인 신호(BB 상단 돌파, 거래량 급증) 최소 충족 개수 (0~2)
SURGE_TP2_PCT = 0.02            # 사다리 2단계: TP1 체결가 대비 +2%
SURGE_TP3_PCT = 0.04            # 사다리 3단계: TP1 체결가 대비 +4% (잔량 전량)
SURGE_TP2_RATIO = 0.30          # 2단계 청산 수량 = 진입수량 x 30% (TP1 40% + 30% + 3단계 잔량 30%)
# 3차 익절(잔량 처리) 방식: 고정 목표가 청산 대신, 1/2차 완료 후 잔량을 트레일링 스탑에 위임한다.
# (기존 TRAILING_STOP_FROM_PEAK / TP_EXTENSION_TRAIL_FROM_PEAK 로직을 그대로 재사용)

# --- 3. 매수 후 보호 가드 (Post-buy protective exits) -----------------------
# 매수 직후 급락 방지: 매수 후 일정 시간 동안 현재가가 매수가 대비
# POST_BUY_BB_DROP_PCT 이상 낮은 상태가 POST_BUY_DROP_CONFIRM_SECONDS 동안
# 유지되면 손절 이전에 조기 매도한다.
POST_BUY_BB_DROP_PCT = 0.014  # 매수가 대비 이탈 임계치 (-1.0%->-1.4%: 백테스트 결과 12건 중 83.3%가 본전 이상 회복, 58.3%는 +2%까지 회복(평균 최대회복 +4.17%) - 매수직후 정상 노이즈 조기청산 완화)
POST_BUY_BB_DROP_ARMED_SECONDS = 180.0  # 매수 후 가드 활성 구간 (3분, 하드스탑 활성 전까지)
POST_BUY_DROP_CONFIRM_SECONDS = 90.0  # 급락 지속 확인 시간 (60초->90초)

# 초기 수익 반납 실패 보호 (본전 부근에서 반납 시 조기 청산)
BREAKEVEN_FAIL_ARM_PNL = 0.010  # 한 번이라도 +1.0% 이익 도달 시 활성화 (0.8%->1.0%)
BREAKEVEN_FAIL_GIVEBACK_PCT = 0.028  # 고점 대비 2.8% 이상 반납 (0.8%->2.0%->2.8%: 백테스트 결과 14건 중 92.9%가 본전 이상 회복, 42.9%는 +2%까지 회복 - 조기청산 완화)
BREAKEVEN_FAIL_CONFIRM_SECONDS = 150.0  # 실패 지속 확인 시간 (60초->120초->150초, 일시 pullback 무시 강화)

# 무추세 시간 손절
NO_TREND_EXIT_ARM_SECONDS = 1200.0  # 20분 동안 추세 미발생 시 점검 시작
NO_TREND_EXIT_MAX_PEAK_PNL = 0.006  # 0.3%->0.6%: 24건 전패(-58,653원) 개선 - 단기 노이즈 무추세 오판 방지
NO_TREND_EXIT_MIN_PNL = -0.005  # 현재 손익이 -0.5% 이하일 때만 적용
NO_TREND_EXIT_CONFIRM_SECONDS = 90.0  # BB 하단 약세 지속 확인 시간 (초)

# --- 4. 매도 보조 시그널 / 박스권 홀드 / MA5-BB 하향크로스 --------------------
# 보조 매도 시그널 score별 최소 요구 수익률
AUX_SELL_MIN_PNL_SCORE2 = 0.015
AUX_SELL_MIN_PNL_SCORE3 = 0.008
AUX_SELL_MIN_PNL_SCORE4 = 0.003
# AUX 매도 슬리피지 버퍼
AUX_SELL_MIN_REALIZED_TARGET_PCT = 0.010
AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT = 0.005

# MA5-BB 하향 크로스 시 즉시 매도 허용 손익 기준
MA5_BB_DOWN_CROSS_IMMEDIATE_PNL = -0.007
# MA5-BB 하향 크로스 즉시 매도 최소 점수
MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE = 2
# MA5-BB 하향 크로스 확정 최소 점수
MA5_BB_DOWN_CROSS_CONFIRM_MIN_SCORE = 1
# MA5-BB 하향 크로스 매도 허용 최소 손익
MA5_BB_DOWN_CROSS_MIN_PNL = 0.000

# 박스권(횡보) 구간에서는 기술적 매도 홀드 기능 사용
ENABLE_BOX_RANGE_HOLD_TECH_SELL = True
# 박스권 판정 시 참조할 최근 봉 개수
BOX_RANGE_HOLD_LOOKBACK_BARS = 8
# 박스권 판정용 가격 범위 상한
BOX_RANGE_HOLD_MAX_RANGE_PCT = 0.0065
# 박스권 판정용 BB 폭 상한
BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT = 0.0080

# --- 5. 매수 진입 - 실시간가/BB 크로스 확인 & 추격매수 방지 -------------------
# BB 크로스 판정용 버퍼(노이즈 필터)
LIVE_PRICE_BB_BUFFER_PCT = 0.0005  # 0.0008
# 상향 크로스 확정에 필요한 연속 관측 횟수 / 최소 유지 시간(초)
LIVE_PRICE_CROSS_CONFIRM_POLLS = 2  # 3 -> 2: 크로스 확인 속도 향상 (30s -> ~10s)
LIVE_PRICE_CROSS_CONFIRM_SECONDS = 10  # 20 -> 10 (보다 빠른 진입 확인)
# 하향 크로스 확정에 필요한 연속 관측 횟수 / 최소 유지 시간(초)
LIVE_PRICE_DOWN_CROSS_CONFIRM_POLLS = 1
LIVE_PRICE_DOWN_CROSS_CONFIRM_SECONDS = 0
# 추격매수 방지: 전일 종가 대비 현재가 상승률이 임계치 이상이면 매수 차단
MAX_BUY_RISE_PCT_FROM_PREV_CLOSE = 0.23  # 23%
# 추격매수 방지: 실시간 BB 상향 크로스 없이(신호 없음) MA5/BB 후행 진입 시
# 현재가가 BB 중심선 대비 과도하게 이격되면 매수 차단
MA5_BB_FOLLOW_CHASE_MAX_GAP_PCT = 0.002  # 0.20% -- tightened: buy only when price is within 0.2% of BB middle
# 매수 연속 확인 횟수 / 진입 전 필요한 최소 확정 봉 개수
BUY_CONSECUTIVE_CONFIRM_COUNT = 2
MIN_BARS_REQUIRED = 3

# --- 5-b. 매수 진입 - 직전 매집봉(Pre-cross accumulation bar) 확인 -----------
# BB 중앙선 상향 돌파 직전에 "매집(거래량 증가) 양봉"이 있었는지 확인하는 필수조건.
# 최근 PRE_CROSS_ACCUM_LOOKBACK_BARS개 봉 중 하나라도 아래 3가지를 모두 만족하면 통과:
#   1) 그 봉의 저가~고가 범위가 그 봉 시점의 BB중간값을 포함
#   2) 그 봉이 양봉(종가>시가)
#   3) 그 봉의 거래량이 그 봉 시점의 VOL_MA20(20봉 평균)보다 큼
# 직전봉(-2) 1개만 보면 매수 빈도가 크게 줄어들어, close_cross 판정처럼 최근 몇 봉으로
# 룩백을 넓혀 빈도 감소를 완화한다. False로 설정 시 이 필수조건 자체를 건너뛴다(롤백용).
# [2026-08-21] 3->5봉, 거래량 기준 "직전봉 대비"->"VOL_MA20 대비"로 완화 (r005 Update log 참조) -
#   257720 실리콘투 09:57 사례처럼 매집 구간에서도 봉 하나하나의 직전봉 대비 거래량은 들쭉날쭉해
#   정상적인 매집 국면을 놓치는 오탈락이 발생해, 더 안정적인 평균 대비 기준 + 넓은 룩백으로 변경.
ENABLE_PRE_CROSS_ACCUM_BAR_CHECK = True
PRE_CROSS_ACCUM_LOOKBACK_BARS = 8
# [2026-08-25] 매집봉 거래량 판정을 "VOL_MA20 초과"에서 "VOL_MA20의 이 비율 이상"으로 완화 -
#   BB중간값 포함+양봉+거래량 조건을 같은 한 봉에서 동시 요구하는 3중 AND라 100% 기준은 너무 희귀함.
PRE_CROSS_ACCUM_VOL_RATIO_MIN = 0.6  # 0.8->0.6: 20260908 로그 분석 결과 0.8도 score=18/22 확정크로스를 리젝시킬 만큼 빡빡함(위 Update log 참조)

# [2026-09-13] 근접교차(Near-cross ARM/Early)/가격선행돌파 절 전체 삭제(사용자 요청
# "dead config 정리") - 관련 함수(_near_cross_momentum_flags, _passes_early_near_cross_
# liquidity, is_early_near_cross_allowed, _price_lead_breakout_context_sim)가 코드
# 전체에서 실제로 호출되는 곳이 전혀 없어(호출부 grep 0건) 완전히 도달 불가능한 코드였음.
# 매수 진입 거래량 MA20 최소치 / 현재봉 거래량 최소치 (저유동성 차단, 공통, 3분봉 기준)
MIN_ENTRY_VOL_MA = 1000
MIN_ENTRY_VOLUME = 1500

# [2026-09-20] 1분봉 하이브리드 트리거 전용 유동성 최소치(HYBRID_1MIN_MIN_ENTRY_VOL_MA=340/
# HYBRID_1MIN_MIN_ENTRY_VOLUME=500 - 위 3분봉 기준값을 1/3로 환산한 복사본)를 삭제했다.
# 거래량/유동성 하한은 위 MIN_ENTRY_VOL_MA/MIN_ENTRY_VOLUME/MIN_ENTRY_TURNOVER_KRW(3분봉
# min_liquidity_safety 게이트) 하나로 통일한다.

# [2026-09-07] 거래대금(turnover=종가*거래량) 기반 유동성 하한 - MIN_ENTRY_VOLUME(주수)은
# 가격대별 형평성이 없다(저가주는 쉽게 통과, 고가주는 동일 주수라도 거래대금이 훨씬 큼에도
# 주수 기준으로는 오히려 불리). 시가총액/가격에 무관한(cap-neutral) 유동성 지표로 거래대금을
# 추가 하한선으로 병행 적용한다. 이미 존재하던 EARLY_NEAR_CROSS_MIN_TURNOVER_KRW(조기진입
# 전용, 5백만원)보다 완만한 일반 매수 게이트용 기본값.
MIN_ENTRY_TURNOVER_KRW = 10_000_000

# BB 중앙선 상승 돌파 전략 파라미터 (BB slope break cross strategy)
BB_SLOPE_LOOKBACK_BARS = 20      # BB 기울기 측정 봉 수 (3분봉 기준 약 1시간)
# BB 중앙선 기울기 최소 허용치(%) - 이 값 이하면 매수 차단 (기존 -0.7 하드코딩값을 상수로 분리 + 완화)
# 60분 lookback은 급등 후 되돌림 국면에서 BB중앙선이 실제 저점 형성보다 한참 늦게 양전환되어,
# 가격/스토캐스틱/윌리엄스%R이 이미 반전된 뒤에도 한동안 매수를 막는 경우가 있어 -0.7%->-2.0%로 완화.
BB_SLOPE_MIN_PCT = -2.0
BB_MID_DOWNTREND_BARS = 5        # BB 중간선 우하향 감지 봉 수 (3분봉 기준 약 15분): 연속 하락 시 매수 차단
# [2026-09-20] 하이브리드 경로(HYBRID_3MIN_CONTEXT_GATES)에서 bb_mid_downtrend_block 게이트를 쓸지 여부.
# True = 현재 동작 유지. False로 바꾸면 이 게이트가 빠져 진입이 완화된다 - 별도 전략 변경으로 다룬다:
# 최근 라이브 5거래일 단독 차단 0건, g003 20260916~18 x 랭킹 상위 8종목(24 종목-일)에서 게이트
# ON/OFF 거래가 완전히 동일했지만 다른 게이트가 논리적으로 포함하지는 않으므로(Codex 검토) 표본이
# 이 정도로는 켜둔 채 유지하고, 바꾸려면 더 많은 일자 백테스트/섀도 검증 후 결정. 비하이브리드
# 3분봉 단독 경로(g003 비교 트래커)에는 영향 없음.
ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK = True
BB_UPPER_GAP_MIN_PCT = 0.25      # BB 상단 여유 최소치 (%) - 상단까지 여유 없으면 매수 차단 (0.5->0.25->0.5->0.25)
CANDLE_GAIN_MIN_PCT = -0.1       # 현재봉 양봉 최소 상승률 (%) - 미세 틱노이즈 허용 (0.1->0.0->-0.1)
CANDLE_GAIN_MAX_PCT = 0.8        # 현재봉 최대 허용 상승률 (%) - 초과 시 추격 매수 차단
BB_MID_CHASE_MAX_GAP_PCT = 0.35  # BB 중간선 대비 현재가 최대 허용 갭 (%) - 초과 시 추격 매수 차단 (1.0->0.7)
# [2026-08-24] uptrend_continuation(크로스 이벤트 없이 추세 지속만으로 진입) 경로 전용 추격 기준.
# 017670 SK텔레콤 2026-08-24 13:12 반등 사례: uptrend_continuation 조건은 13:17에야 확정됐는데,
# BB_MID가 후행지표라 그땐 이미 가격이 BB_MID_CHASE_MAX_GAP_PCT(0.35%)보다 멀리 가있어 재진입
# 자체가 계속 막혔음(그 시간대엔 CHASE_BUY_BB_GAP 리젝만 반복). 이 경로에서는 BB_MID 갭 상한을
# 넓히는 대신 RSI 과열 여부로 "아직 쫓아가도 되는 건강한 지속 구간"인지를 추가로 검증한다.
UPTREND_CONT_CHASE_MAX_GAP_PCT = 0.6   # uptrend_continuation 진입 시 BB_MID 갭 상한 (%)
UPTREND_CONT_CHASE_RSI_MAX = 75.0      # 이 값 이상이면 과열로 보고 차단
# [2026-09-06] uptrend_continuation 판정 자체의 BB 기울기 조건. 196170 알테오젠 2026-08-21
# 09:33~09:40 사례: ADX 80~88, +DI>>-DI, MA5 상승 등 나머지 조건은 전부 강한 지속 추세를
# 가리켰는데 bb_slope_pct가 -0.016%~0.071% 사이에서 미세하게 진동(BB_MIDDLE이 급등을
# 뒤늦게 따라잡는 후행지표 특성상 기울기가 0 근방에서 노이즈로 흔들림)하는 바람에 09:33에는
# 엄격한 ">0.0" 기준에 막혀 리젝됐다가 09:36에야 겨우 통과 - 그마저도 순전히 0 근방 노이즈
# 타이밍 운에 좌우됨. 더 눈에 띄는 점은 이 값(0.0)이 필수 게이트인 bb_slope_rising의
# BB_SLOPE_MIN_PCT(-2.0%)보다 훨씬 엄격하다는 것 - uptrend_continuation은 필수 게이트를
# 이미 통과한 지속 추세용 대체 진입 경로인데 그 자체 조건이 필수 게이트보다 더 까다로운
# 역전 상태였음. 작은 음수 허용치를 둬 노이즈성 미세 하락을 지속 추세 이탈로 오판하지
# 않도록 함.
UPTREND_CONT_SLOPE_MIN_PCT = -0.05      # uptrend_continuation 전용 BB 기울기 하한 (%) - 0 근방 노이즈 허용
BB_BUY_SCORE_THRESHOLD = 10  # 8->10: 2026-07-20 실매매 로그 분석 결과 score=8(구 임계값) 매수 6건이
  # 전부 손실(합계 -18,975원, 당일 총손실 -21,820원의 87%). score=9 매수 4건도 순손실(-2,295원).
  # score>=10 매수만 순이익 포함(GS +1,400원 익절 등, 순 -550원 vs 조정 전 -21,820원 전체손익).
  # 표본 1일(16건)이라 지속 모니터링 필요. 매수 최소 점수 (공격형=6, 중립형=8, 보수형=10)
# 개장 직후 보호: 장 시작(09:00) 후 이 분 수 이내에는 score threshold를 높여 추격매수 방지
OPENING_GUARD_MINUTES = 15       # 개장 후 15분(첫 5봉) 동안 강화된 필터 적용
OPENING_GUARD_SCORE_THRESHOLD = 12  # 개장 직후 요구 최소 점수 (일반=8, 개장보호=12)

# --- 7. 매수 진입 - 보조지표 임계값 (과열/모멘텀 필터) ------------------------
STOCH_OVERBOUGHT = 96.0  # 85.0 -> 92.0 -> 96.0 (과열 차단 기준 완화)
STOCH_BUY_MIN = 20.0  # 매수 스토캐스틱 K 하한
STOCH_BUY_MAX = 50.0  # 매수 스토캐스틱 K 상한
STOCH_D_BUY_MIN = 40.0  # [2026-09-14][사용자 요청①] 매수 스토캐스틱 %D 하한 신규 추가 (실거래 분석: %D<50 승률 24-33%(n=64) vs 50-80 42%(n=50)); [2026-09-22] 50->40 잠정 완화(K>D & 40<=D<50 근접실패 n=82, TP36.6%/SL32.9%, 5일 소표본 - 표본외 검증 필요, 롤백=50.0)
RSI_BUY_MOMENTUM_MAX = 60.0  # RSI 모멘텀 허용 상한(기본: 50~60 구간 유지)
WILLIAMS_BUY_FLOOR = -70.0  # 매수 허용 Williams %R 하한
WILLIAMS_OVERBOUGHT_CEIL = -10  # -20 -> -10 (Williams R 완화)
BB_UPPER_PROXIMITY_MAX = 1.05  # 0.85 -> 1.05 (BB 상단에서 충분히 진입 허용)

ADX_MIN_TREND = 15.0  # 20.0 -> 15.0 (ADX 최소값 완화)
ADX_STRONG_TREND = 40.0

# [2026-09-07] 시그널 매도 억제(_strong_uptrend) 판정에 쓰이던 하드코딩 리터럴을 상수로
# 분리 - r003 매도 루프(Signal-based full exits)에 600.0/28/-0.012/-0.008이 그대로 박혀
# 있어 r001만 보면 튜닝이 끝난다는 원칙과 어긋났다. SIGNAL_EXIT_STRONG_TREND_ADX_MIN(28)은
# ADX_STRONG_TREND(40, 다른 곳에서 쓰이는 "매우 강한 추세" 기준)와 별개의, 시그널 매도
# 억제 전용 기준값이라 이름을 구분한다.
SIGNAL_EXIT_MIN_HOLD_SECONDS = 600.0  # 시그널 기반 청산(스토캐스틱/MACD) 최소 보유 시간(초)
SIGNAL_EXIT_STRONG_TREND_ADX_MIN = 28.0  # 이 값 초과 + DI+>DI- 이면 스토캐스틱 매도신호 억제
SIGNAL_EXIT_STOCH_SUPPRESS_PNL_MIN = -0.012  # 이 손익률 초과(덜 손해)면 스토캐스틱 매도신호 억제
SIGNAL_EXIT_MACD_PNL_MAX = -0.008  # 이 손익률 이하여야 MACD 히스토그램 2봉 하락 매도 발동
ADX_BUY_MIN = 25.0  # 매수 진입용 ADX 최소값
REQUIRE_ADX_RISING = True  # 매수 시 ADX가 직전봉 대비 우상향이어야 하는지 여부
REQUIRE_DI_PLUS_DOMINANT = True  # 매수 시 +DI가 -DI보다 커야 하는지 여부

MFI_OVERBOUGHT_MAX = 80.0  # MFI 과열 상한(이상일 경우 추격 매수 금지)

# [2026-09-07] REQUIRE_OBV_SIGNAL_CROSS는 원래 "필수 게이트"로 설계된 이름이지만
# 실제로는 어떤 매수 파이프라인에도 배선되어 있지 않았다(값만 있고 소비처가 없는
# 상태). 거래량 방향 확인(OBV가 OBV_MA를 상향 돌파했는가)은 필수 게이트로 걸면
# 거래 빈도를 과도하게 줄일 위험이 있어, 대신 BUY_SCORE_RULES 가점 항목으로
# 구현한다(OBV 상승 확인 시 가점 - 가격 돌파와 거래량 방향이 같이 확인되면 가짜
# 돌파(fakeout) 가능성이 낮다는 일반적인 거래량 분석 관행 참고). 이 플래그는 이제
# 해당 가점 항목의 활성화 여부를 제어한다.
REQUIRE_OBV_SIGNAL_CROSS = True  # OBV 골든크로스+돌파 가점 활성화 여부
OBV_BREAKOUT_LOOKBACK_BARS = 5  # OBV 돌파 판정에 사용할 과거 봉 개수
OBV_CONFIRM_SCORE = 2  # OBV가 OBV_MA를 lookback 내에서 상향 돌파했을 때 가점

# --- 8. 거래량 필터 ----------------------------------------------------------
VOLUME_RATIO_OPEN = 0.75  # 0.40 -> 0.75 (개장 초반 저거래량 종목 진입 차단 강화)
VOLUME_RATIO_MIDDAY = 0.75  # 0.50 -> 0.75 (저거래량 추격매수 차단)
VOLUME_RATIO_CLOSE = 0.80  # 0.60 -> 0.80 (장마감 저거래량 차단 강화)
VOLUME_RATIO_NXT = 0.30  # 0.40 -> 0.30 (NXT 세션 거래량 필터 더 완화)
VOLUME_RATIO_STRONG_RELAX = 0.15  # 0.10 -> 0.15 (강한 추세 보너스 증가)
VOLUME_RATIO_FLOOR = 0.70  # 0.55 -> 0.70 (최소 하한 강화)

# --- 9. 인트라바 실시간 진입 필터 --------------------------------------------
ENABLE_INTRABAR_LIVE_ENTRY_FILTER = True
INTRABAR_MIN_ELAPSED_SECONDS = 90.0
INTRABAR_MFI_MIN = 50.0
INTRABAR_MFI_MAX = 75.0
# [2026-09-18] 사용자 요청 - 기본 구간 50~70을 45~65로 낮추고(이미 확정봉 대비 소폭
# 완화), 65 초과 구간은 무조건 거부하지 않고 65~75(INTRABAR_RSI_EXT_MAX)까지 조건부
# 허용 구간을 신설한다 - RSI/MACD 상승 + 거래량(VOL_MA20 상회) + BB중간선 위 4개
# 확인 조건을 전부 만족할 때만 통과(_passes_intrabar_entry_gate 참조). 이미 강하게
# 상승 중인 종목의 인트라바 RSI가 65~70대에서 자주 걸려 확정봉으로 폴백되는 사례
# (20260918 000500 가온전선 등) 분석 후 조정 - 75 초과는 여전히 무조건 거부.
INTRABAR_RSI_MIN = 45.0
INTRABAR_RSI_MAX = 65.0
INTRABAR_RSI_EXT_MAX = 75.0
INTRABAR_ADX_MIN = 20.0

# --- 10. 리스크 서킷브레이커 / 재진입 & 안전장치 -----------------------------
# HARD_STOP이 N번 연속 발생하면 N분간 신규 매수 차단 (6/22, 6/23 전손절 방지)
HARD_STOP_CIRCUIT_BREAKER_COUNT = 3      # 연속 HARD_STOP 발생 횟수 임계치
HARD_STOP_CIRCUIT_BREAKER_COOLDOWN_MIN = 60  # 서킷브레이커 발동 시 신규 매수 차단 시간(분)
# 당일 HARD_STOP_LOSS 발생 종목 재진입 차단 여부 (같은 날 같은 종목 손절 후 재매수 금지)
HARD_STOP_BLOCK_REENTRY_TODAY = True
# 같은 날 같은 종목 재매수를 막는 명시적 규칙은 두지 않는다(2026-09-20 사용자 결정 - 종전
# ALLOW_REBUY_SAME_CODE=False는 실전이 읽은 적 없는 죽은 설정이라 삭제). 재진입 제어는
# has_buy_exposure(보유/주문중)/TRADE_COOLDOWN_MINUTES/HARD_STOP_BLOCK_REENTRY_TODAY뿐이다.
# 주의: 실전은 매수 체결 종목을 감시목록에서 졸업(GRADUATE)시키고 청산 뒤에도 되돌리지 않으므로
# (r003 _rebalance_active_watchlist) 그날은 다시 평가되지 않는다 - g003은 재진입을 허용해 둘이
# 다르다. 재매수를 실전에서 실제로 가능하게 하려면 졸업/재편입 정책을 따로 바꿔야 한다.
# 동일 종목 재진입 쿨다운(분):
TRADE_COOLDOWN_MINUTES = 3
# 시장일 확인 실패 시 보수적으로 비거래 처리
MARKET_DAY_FAIL_CLOSED = True

# --- 11. 주문/자금 관리 -------------------------------------------------------
MAX_ORDER_AMOUNT_KRW = 500_000  # 1회 매수 주문 최대 금액(KRW)

# [2026-09-18] 신규 진입 매수 분할 - 절반은 즉시 체결(시장가), 절반은 매수1호가
# 지정가(+기존 추격 재주문)로 낸다. 수량이 1주뿐이면 분할하지 않고 전량 시장가로
# 즉시 매수한다(r003 place_buy_order 참조, r001 Update log 2026-09-18 참조).
ENABLE_BUY_SPLIT_MARKET_LIMIT = True
BUY_SPLIT_MARKET_RATIO = 0.5  # 시장가 즉시체결 비중 (나머지는 매수1호가 지정가)

# --- 11-b. 피라미딩(불타기) - 추세 지속 시 1회 추가 진입 ----------------------
# 평균단가 대비 PYRAMID_TRIGGER_PNL_PCT 이상 이익이고 MA5/BB중간선/ADX가 모두
# 상승(직전봉 대비) 중일 때 1회에 한해 추가 매수한다(같은 포지션당 최대 1회).
# 추가매수 금액도 MAX_ORDER_AMOUNT_KRW 한도를 그대로 따른다.
ENABLE_PYRAMIDING = False  # 2026-08-20: 보유중 종목 중복 재매수 방지 위해 비활성화
PYRAMID_TRIGGER_PNL_PCT = 0.005  # 평균단가 대비 +0.5%

# --- 12. (삭제됨 2026-09-20) 1분봉 골든크로스 단독 매수 / 1분봉 Entry Score 게이트 ---
# 하이브리드(아래 12-c: 1분봉 트리거 + 3분봉 컨텍스트)가 유일한 매수 경로가 되면서 두 경로를 켜고
# 끄던 플래그(ENABLE_1MIN_GOLDEN_CROSS_BUY/ENABLE_1MIN_ENTRY_SCORE_GATE)와 전용 상수(ENTRY_*,
# EMA_9_PERIOD)를 삭제했다 - 도입/검증 이력은 git history 참조.
EMA_20_PERIOD = 20   # calculate_indicators의 EMA_20(3분봉 ema_trend_align 가점)이 사용

# --- 12-c. 1분봉 트리거 + 3분봉 컨텍스트 하이브리드 (섹션 12/12-b와 별개 3번째 경로) ---
# 섹션 12(ENABLE_1MIN_GOLDEN_CROSS_BUY)는 1분봉이 3분봉을 완전히 대체하고, 섹션
# 12-b(ENABLE_1MIN_ENTRY_SCORE_GATE)는 3분봉 필수조건 통과 "이후"에만 1분봉을 보조로
# 확인한다 - 즉 3분봉의 bb_mid_cross_up/candle_bullish_and_chase_guard 게이트에서
# 막히면 1분봉 확인 자체가 실행되지 않는다. 2026-08-28 HPSP(403870) 실매매 로그 분석
# 결과, BB_MID(3분봉)가 후행지표라 크로스가 "확정"되는 시점엔 이미 가격이 추격매수
# 갭 문턱(BB_MID_CHASE_MAX_GAP_PCT)을 넘어있어 신호 자체는 났는데(cross_pass=True)
# 매번 CHASE_BUY_BB_GAP/CANDLE_NOT_BULLISH로 반려되는 패턴이 반복 확인됨(09:09, 09:27
# 두 차례). 이 플래그가 True면 순서를 뒤집는다: (1) 1분봉 자체 골든크로스+양봉+추격가드
# (check_buy_condition_1min, 1분봉 자신의 open/BB를 기준점으로 판정 - 트리거 시점과
# 판정 기준 프레임 불일치 방지)를 트리거로 먼저 확인하고, (2) 통과 시에만 3분봉을
# 컨텍스트 필터(HYBRID_3MIN_CONTEXT_GATES = BUY_GATE_CONDITIONS 중 bb_mid_cross_up/
# candle_bullish_and_chase_guard 2개를 제외한 나머지 - BB기울기/다운트렌드차단/매집봉/
# BB상단여유/스토캐스틱/윌리엄스/유동성 + 가점 스코어)로 재확인한다. 크로스 감지가
# 최대 2분 빨라져 gap이 작을 때 신호가 뜨므로 추격가드 반려가 줄어들 것으로 기대되나,
# 1분봉은 3분봉보다 휩쏘(가짜 돌파)에 약해 오탐이 늘 수도 있음.
# True 활성화 시 ENABLE_1MIN_ENTRY_SCORE_GATE(12-b)는 자동으로 건너뛴다(같은 1분봉
# 트리거를 이미 확인했으므로 중복 게이트로 과도하게 좁아지는 것을 방지).
# 2026-08-28 1차 검증(r007 --date 20260828 --codes 403870, check_buy_condition_1min을
# 트리거로 그대로 재사용): 여전히 매수 0건. 리젝 사유 분석 결과 진짜 병목은 트리거
# 순서가 아니라 check_buy_condition_1min의 require_fresh_cross가 "바로 이 1분봉"만
# 인정하는 단일봉 판정이었음(469/507건이 1MIN_NO_BB_MID_GOLDEN_CROSS) - 3분봉의
# 5봉 룩백+우상향지속 관용도가 1분봉에는 전혀 없었던 것. 실제로 크로스가 잡힌 소수
# 케이스도 3분봉과 동일한 CANDLE_GAIN_MAX_PCT(0.8%)/BB_MID_CHASE_MAX_GAP_PCT(0.35%)를
# 1분봉에 그대로 적용하다 보니 HPSP처럼 빠른 종목은 그마저도 초과(0.87~1.67%)해 반려됨.
# 이에 HYBRID_1MIN_TRIGGER_* 전용 상수(아래)를 신설 - check_buy_condition_1min(섹션12
# 전용, 원본 그대로 유지)과 분리된 check_buy_condition_1min_hybrid_trigger가 사용한다.
# 2026-08-28 2차 검증(위 HYBRID_1MIN_TRIGGER_* 완화 적용 후 동일 백테스트 재실행):
# 09:07 매수(51,325)/09:25 매도(52,450) +2.19%(+10,125 KRW) 1건 체결 성공 - 기존
# 3분봉 경로/1차 하이브리드 모두 하루 종일 매수 0건이었던 종목에서 최초로 체결됨.
# 단 단일 종목·단일 일자(n=1) 검증이라 과최적화 위험이 있음 - 기본값은 False로 유지,
# 다른 날짜/종목 다건 백테스트로 일반화 여부를 추가 확인한 뒤에만 True 전환 권장.
# 2026-08-28 3차 검증(다른 2개 일자 x 랭킹 상위 3종목, r007 --date 20260826 --codes
# 047040 001820 388050 / --date 20260827 --codes 403870 069540 237690): 결과 혼재.
# (1) 20260826 047040: baseline 0건 -> hybrid 1건(13:33 매수 19,393/14:07 SIGNAL_EXIT
# 매도 19,233, -0.83%/-4,000원 손실) - 신규 체결이지만 손실.
# (2) 20260827 403870 HPSP: baseline 1건(13:59 매수 50,433/15:20 강제청산 50,500,
# +0.13%/+600원)을 hybrid가 더 늦은 크로스로 대체(14:07 매수 50,500/15:20 강제청산
# 50,500, +0.00%) - 진입가가 늦어져 기존 소폭 이익이 손익분기로 악화.
# (3) 20260828 403870 HPSP: 위 2차 검증 결과(+2.19%/+10,125원, baseline 0건) 유지.
# 3개 일자 합산 손익은 +6,125원으로 플러스지만, 4건 중 순수 신규 이익 1건/신규 손실
# 1건/기존 이익 악화 1건으로 방향성이 일관되지 않음 - 표본이 여전히 작고(n=3일,
# 4건) 결과가 혼재돼 있어 기본값 False 유지. 추가로 검증하려면 (a) 더 많은 일자/
# 종목으로 표본 확대, (b) 20260827 케이스처럼 "더 늦은 크로스로 대체"되는 원인
# (HYBRID_3MIN_CTX의 OPENING_GUARD/스코어 재계산 타이밍 차이 추정) 규명 필요.
# [2026-09-20] 하이브리드가 유일한 매수 경로가 되어 ENABLE_1MIN_TRIGGER_3MIN_CONTEXT 플래그를 삭제했다
# (위 이력의 "기본값 False 유지" 등은 도입 당시 기준 기록).
# [2026-09-09] 452190 한빛레이저 사례: 3분봉은 11:54~11:58에 이미 골든크로스 확정(score
# 17/22)했는데, 12:07~12:30 폭등 구간(4,655->5,160) 전체가 HYBRID_1MIN_TRIGGER_
# 1MIN_NO_BB_MID_GOLDEN_CROSS로 100% 리젝됨 - 돌파 자체는 12:07~09에 발생했지만 그 후
# 가격이 BB중간선 위에서 계속 강하게 올라, "크로스 시점"이 룩백창(3->5봉) 밖으로
# 벗어나 버려 매 폴링마다 크로스가 "없다"고 판정됨(역설적으로 추세가 강하고 오래갈수록
# 못 통과). 3->8로 완화(그래도 무한정은 아니고, BB_GAP_CEILING_PCT가 추격 상한을 유지).
# [2026-09-22] 8->12 잠정 완화(20260921 분석: 새로 통과 n=59, TP39.0%/SL28.8%, 5일 소표본 - 표본외 검증 필요, 롤백=8).
# 추격 상한은 그대로(HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT가 경과봉 수와 무관하게 상한).
HYBRID_1MIN_TRIGGER_LOOKBACK_BARS = 12       # 크로스 인정 룩백(1분봉 12봉 = 12분 = 3분봉 4봉)
HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT = -0.4  # 1분봉 자체 틱노이즈가 더 커서 3분봉(-0.1%)보다 완화
HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT = 1.8   # 1분봉 급등 캔들은 3분봉 환산 시 정상 범위일 수 있어 완화
HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT = 0.5        # 3분봉(0.35%)보다 소폭 완화 - 트리거를 빨리 잡는 목적과 상충 방지

# [2026-09-07] 크로스가 유효하게 지속(BB_MID 위 연속 유지) 중인데도 BB_MID가 후행지표라
# 갭이 계속 벌어져 고정 상한(HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT)에 매 폴링 걸리는 문제
# 완화용 - 크로스 후 경과봉 수 x DECAY만큼 상한을 늘리되 CEILING으로 상한선을 둔다.
HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR = 0.15
HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT = 1.1

# [2026-09-18] CEILING_PCT(1.1%)가 신선한 크로스 오남용 방지용으로는 적절하지만,
# uptrend_continuation 경로(호출측 3분 컨텍스트 또는 자체 ADX/DI/MA5 판정으로 상승추세
# 지속이 이미 별도 확인된 경우)에도 동일하게 적용되면서, 하루 20%+ 급등하는 종목처럼
# 1분봉 BB중간선(후행 SMA)이 가격을 아예 못 따라가는 날엔 갭이 2~3%대로 영구히 벌어져
# 남은 장중 내내 진입 자체가 막히는 사례가 확인됨(20260918 로그 분석 - 000500 가온전선
# +22.7%/043260 성호전자 +21.3%/024840 KBI메탈 +16.0% 등 score 20+/24 고품질 신호가
# CHASE_BUY_BB_GAP에 반복 반려, 그 중 028050 삼성E&A는 09:11 score=20으로 반려된 뒤
# 09:34 score=13으로 신호 품질이 식고 나서야 다른 경로로 겨우 진입). uptrend_continuation
# 경로는 신선한 크로스와 달리 다른 지표들로 추세 지속이 이미 검증된 상태이므로, 이
# 경로에서만 상한을 완화한다 - 신선한 크로스 경로의 상한(위 CEILING_PCT)은 그대로 유지해
# 저품질 스파이크 추격은 계속 차단한다.
HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT = 2.2

# [2026-09-23] 사용자 요청(204620 글로벌텍스프리 2026-09-23 09:17 매수 - 1분봉 골든크로스는 09:04에
# 이미 유효했는데 3분봉 컨텍스트 게이트(stochastic_buy_signal/bb_upper_gap_min)가 계속 막아 실제 매수가
# 13분 뒤, 가격이 이미 그 구간 상단 근처로 오른 뒤에야 체결된 사례 + Codex 설계검토) - 1분봉 트리거가
# "계속 유효한 채로 대기" 상태로 이 시간(초) 이상 지속되면(HYBRID_1MIN_TRIGGER_LOOKBACK_BARS의 12봉
# 룩백과 별개로, 3분봉 컨텍스트가 못 따라와 대기만 길어지는 경우를 겨냥) 매수를 포기한다 - 게이트가
# 뒤늦게 전부 맞아떨어져도 그땐 이미 추격매수가 된 경우가 많다는 진단(Codex 동의). <=0이면 비활성(기존과
# 동일). 아직 백테스트로 정확한 값을 검증하지 않은 탐색적 상수 - 실거래 관찰 후 조정 필요.
HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS = 300.0   # 5분

# [2026-09-23] 사용자 요청("매수/매도 컨셉 재검토") + Codex 설계검토 - r006 _011_hybrid_1min_dead_cross_exit.
# 매수측 1분봉 골든크로스 트리거와 대칭되는 매도 조건. Codex의 5가지 권고를 반영:
# (1) 신선한 크로스 또는 룩백+연속유지(check_1min_dead_cross, r002) - 매수측과 동일 패턴이나 룩백은 더
#     짧게(빠른 반응 목적, 매수 12봉보다 좁힘). (2) 최소 보유시간(MIN_HOLD_SECONDS)으로 매수 직후 노이즈
#     방지 - 이 구간은 _015_post_buy_entry_drop_guard가 별도 담당. (3) 손실 구간은 수익 요건 없이 즉시
#     청산(LOSS_EXIT_PNL_MAX) - _021_shared_reversal_sell(AUX_REVERSAL_SCORE)의 "손실 포지션을 절대
#     청산 못 하는" 구조적 결함(2026-09-23 컨셉 재검토 메모 참조)을 이 조건으로 해결한다. (4) 손실이
#     아직 그 정도가 아니면(수익 중이거나 얕은 손실) BB중심선 자체 기울기(1분봉)도 꺾였는지 추가 확인 -
#     정상 상승추세 눌림목에서의 오매도(휩쏘) 방지. (5) 확인창(CONFIRM_SECONDS)으로 단일 틱 노이즈 방지 -
#     HARD_STOP_CONFIRM_SECONDS/ATR_STOP_CONFIRM_SECONDS와 동일 패턴.
# ENABLE 플래그로 명확한 킬스위치를 둔다(다른 상수처럼 <=0 암묵적 비활성보다 이 조건은 영향이 커서 명시적으로).
ENABLE_HYBRID_1MIN_DEADCROSS_EXIT = True
HYBRID_1MIN_DEADCROSS_LOOKBACK_BARS = 5          # 매수측(12봉)보다 좁힘 - 손절/추세이탈 반응은 빠르게
HYBRID_1MIN_DEADCROSS_MIN_HOLD_SECONDS = 60.0    # 매수 직후 최소 유예(초) - 기존 시그널 청산군(600초)보다 훨씬 짧음
HYBRID_1MIN_DEADCROSS_CONFIRM_SECONDS = 15.0     # 노이즈 방지 확인창(초) - HARD_STOP(10s)~ATR_STOP(20s) 사이
HYBRID_1MIN_DEADCROSS_LOSS_EXIT_PNL_MAX = -0.003  # -0.3%: 이 손익 이하면 수익요건 없이 즉시 청산

# [2026-09-23] 사용자 요청("급등 후 꺾이면 고점 대비 -0.8%서 익절") + Codex 설계검토 - r006
# _012_peak_retracement_guard. TP1(STAGED_TP1_PCT=3.0%)/ATR익절선 도달 전 구간은 현재 아무 보호장치가
# 없다(그 밑에서 피크를 찍고 반납해도 하드손절/시그널청산까지 아무 것도 안 잡음) - 이 공백을 메운다.
# 주의: 이 파일의 BREAKEVEN_FAIL_GIVEBACK_PCT 이력(0.8%->2.0%->2.8%로 점진 완화, 백테스트 근거: 92.9%가
# 본전 이상 회복)은 고정 0.8% 되돌림처럼 좁은 문턱이 정상적인 되돌림도 조기청산할 위험을 보여준다 -
# Codex 권고대로 ATR% 기반으로 문턱을 정해(고정 MIN_PCT는 하한선일 뿐) 변동성 큰 종목은 자동으로 문턱이
# 넓어지게 한다. 아직 백테스트로 정확한 값을 검증하지 않은 탐색적 구현 - 실거래 관찰 후 조정 필요.
ENABLE_PEAK_RETRACE_GUARD = True
PEAK_RETRACE_GUARD_ARM_PNL = 0.010        # +1.0% 이상 찍어야 활성화 (BREAKEVEN_FAIL_ARM_PNL과 동일)
PEAK_RETRACE_GUARD_MIN_PCT = 0.008        # 0.8%: 문턱 하한(사용자 요청값 그대로, ATR가 이보다 작을 때 하한 역할)
PEAK_RETRACE_GUARD_ATR_MULT = 1.0         # 문턱 = max(MIN_PCT, ATR% x 이 배수)
PEAK_RETRACE_GUARD_CONFIRM_SECONDS = 30.0  # 노이즈 방지 확인창(초)

# 3분봉 가점(_buy_support_score)용 장기 추세 정합성: EMA20 > EMA60이면 상위 추세가
# 우상향이라는 뜻으로 +2점. EMA_20_PERIOD는 위 1분봉 게이트와 공유(같은 컬럼, 프레임만 다름).
EMA_60_PERIOD = 60
EMA_TREND_ALIGN_SCORE = 2             # EMA20 > EMA60 (3분봉 기준)

# [2026-09-13][R77-A] pre_cross_accumulation_bar를 필수 게이트에서 가점으로 전환
# (사용자 요청) - 이 조건 하나가 2026-08-25 로그 분석에서 리젝 사유 1위(1,284건,
# 2,385분 차단, 2위 대비 2.4배)였던 이력이 있어, 필수조건 대신 가점으로 완화한다.
# 판정 로직(룩백/거래량비율)은 그대로 유지, 통과/실패가 매수 자체를 막지 않고
# 점수에만 반영되도록 변경.
PRE_CROSS_ACCUM_SCORE = 1

# [2026-09-13][R77-D] ADX 추세강도 점수를 "크기만" 보던 것에서 DI(+/-) 방향성도
# 함께 확인하도록 재구성 (사용자 요청) - ADX는 하락추세에서도 높게 나올 수 있어
# 크기만으로는 방향을 확인 못함. +DI가 -DI보다 우세하지 않으면 0점, 우세해도
# ADX/DI스프레드가 약하면 2점, 둘 다 강하면 3점 (2~3점 구조, 기존 1점 티어 제거).
ADX_DI_SCORE_MIN_ADX = 25.0            # 이 미만이면 DI 우세해도 0점
ADX_DI_SCORE_STRONG_ADX = 30.0         # 이 이상 + 스프레드 조건 충족 시 3점
ADX_DI_SCORE_STRONG_SPREAD = 15.0      # DI_PLUS - DI_MINUS 최소치 (3점 조건)

# [2026-09-13][R77-G] BB 상단까지 남은 여유폭을 종목 변동성(ATR) 대비로 정규화한
# 가점 신규 추가 (사용자 요청) - 기존 bb_upper_gap_min 게이트(고정 0.25%)는 필수조건
# 으로 그대로 유지하고, 이건 그 위에 얹는 보너스: 저변동 종목의 0.3% 여유와
# 고변동 종목의 0.3% 여유는 의미가 다르므로 ATR 배수로 환산해 평가한다.
BB_UPPER_ROOM_ATR_TIER1 = 0.8          # room/ATR >= 이 값이면 1점
BB_UPPER_ROOM_ATR_TIER2 = 1.5          # room/ATR >= 이 값이면 2점

# [2026-09-14][사용자 요청②] DI스프레드(+DI--DI) 최소치 신규 필수 게이트. 실거래 분석:
# 0-10구간 승률 25.6%(n=39), 10-20구간 36.4%(n=33), 20+구간 62.5%(n=8, 표본 작음) -
# 가장 나쁜 구간만 배제하는 보수적 값으로 시작, 위 ADX_DI_SCORE_*(가점)와는 별개.
DI_SPREAD_MIN_REQUIRED = 10.0

# --- 13. 개장 초반 갭/거래량폭발 라이브 게이트 (Opening gap/volume live gate) -------
# r002 스캐너는 전일 종가 기준 데이터로 랭킹을 매기므로, 당일 아침 뉴스/해외증시
# 영향으로 갭상승/갭하락 출발하거나 거래량이 급변하는 경우를 반영하지 못한다.
# 이를 보완하기 위해 개장 후 일정 시간(윈도우) 동안 신규 매수 직전에 한 번 더
# 당일 시가 갭 비율과 초반 거래량을 실시간으로 검증한다.
ENABLE_OPENING_GAP_VOLUME_GATE = True
OPENING_GAP_GATE_WINDOW_MINUTES = 5     # 개장 후 이 분(分) 이내에만 게이트 적용
OPENING_GAP_MIN_PCT = 0.0               # 선호 갭 하한 (0%)
OPENING_GAP_MAX_PCT = 0.05              # 선호 갭 상한 (+5%)
OPENING_GAP_HARD_FLOOR_PCT = -0.02      # 이 미만 갭하락이면 해당 종목 당일 신규매수 자체를 차단 (-2%)
OPENING_MIN_EARLY_VOLUME_RATIO = 1.5    # 개장초반 거래량 폭발 판정 배수 (현재봉 거래량 / VOL_MA20)

# --- 14. 봉 확정 지연 유예 (Candle confirmation delay) -----------------------
# 거래소/브로커 API가 봉 마감 시각 직후 1~2초 정도 최종 체결 데이터 반영이 지연될
# 수 있어, 정확히 마감된 시각까지만 포함하면 아직 데이터가 덜 채워진 봉을 '확정봉'
# 으로 오인할 수 있다. 봉 확정 판정 시각을 이 값만큼 뒤로 미뤄 안전 마진을 둔다.
CANDLE_CONFIRM_DELAY_SECONDS = 2


# ---------------------------------------------------------------------------
# Indicator periods (지표 계산 주기) - 매매 판단 임계값이 아닌 지표 자체의
# 계산 기간/배수. 값을 바꾸면 지표 모양 자체가 달라지므로 위 튜닝 섹션과는
# 분리해서 관리한다.
# ---------------------------------------------------------------------------
# 볼린저밴드 기준 기간(봉 개수)
BB_PERIOD = 20
# 볼린저밴드 표준편차 배수(상/하단 폭)
BB_STD_MULTIPLIER = 2.0
# 기본 이동평균 기간(공용)
MA_PERIOD = 5
# 스토캐스틱 %K 계산 기간
STOCH_K_PERIOD = 10
# 스토캐스틱 %D 평활 기간
STOCH_D_PERIOD = 5
# RSI 계산 기간
RSI_PERIOD = 14
# RSI 시그널선 평활 기간
RSI_SIGNAL_PERIOD = 6
# Williams %R 계산 기간
WILLIAMS_R_PERIOD = 10
# Williams %D(평활) 기간
WILLIAMS_D_PERIOD = 9
# MFI 계산 기간
MFI_PERIOD = 14
# 거래량 이동평균 기간
VOLUME_MA_PERIOD = 20
# OBV 이동평균 기간
OBV_MA_PERIOD = 10
# MACD 단기 EMA 기간
MACD_FAST = 5
# MACD 장기 EMA 기간
MACD_SLOW = 12
# MACD 시그널 EMA 기간
MACD_SIGNAL_PERIOD = 4
# ADX 계산 기간
ADX_PERIOD = 7
# ATR 계산 기간
ATR_PERIOD = 14

# ---------------------------------------------------------------------------
# Session / time constants
# ---------------------------------------------------------------------------
# NXT 세션 활성화 여부 및 시간 설정
ENABLE_NXT_SESSION = True  # NXT 세션 포함 운용 여부
MORNING_NXT_START = dt_time(8, 0)
MORNING_NXT_END = dt_time(8, 50)
REGULAR_START = dt_time(9, 0)
REGULAR_END = dt_time(15, 30)
# 신규 진입 허용 종료 시각(정규장)
REGULAR_NEW_ENTRY_CUTOFF = dt_time(15, 20)
# 강제 청산 시작 시각(정규장)
REGULAR_FORCE_EXIT = dt_time(15, 20)
AFTERNOON_NXT_START = dt_time(15, 30)
AFTERNOON_NXT_END = dt_time(20, 0)
# 신규 진입 허용 종료 시각(오후 NXT)
AFTERNOON_NXT_NEW_ENTRY_CUTOFF = dt_time(19, 59)
# 강제 청산 시작 시각(오후 NXT)
AFTERNOON_NXT_FORCE_EXIT = dt_time(19, 59)
# 세션 종료 컷오프에서 전량 청산 강제 여부
SESSION_FORCE_CLOSE_ALL_AT_CUTOFF = True
# 세션 종료 홀드 예외 허용 여부
ENABLE_SESSION_EXIT_HOLD_WITHIN_STOP = False
# 오전 NXT 신규 진입 종료 시각(별도 변수)
MORNING_NXT_NEW_ENTRY_CUTOFF = MORNING_NXT_END

# ---------------------------------------------------------------------------
# Live execution operational parameters (r003_trade_live_execute) - 매매
# 판단값이 아닌 시스템 운영(폴링 주기/백오프/재시도 등) 파라미터.
# ---------------------------------------------------------------------------
STARTUP_WARMUP_SECONDS = 90
# 메인 루프 폴링 간격(초)
POLL_INTERVAL_SECONDS = 10  # 15 -> 10 (더 빠른 대응)
# 연속 매수 확인(BUY_CONSECUTIVE_CONFIRM_COUNT) 사이 허용 간격 상한(초) - 이보다 오래 벌어지면 확인 횟수를
# 1로 되돌린다. [2026-09-21] r003에 POLL_INTERVAL_SECONDS * 2로 하드코딩돼 있던 값을 이름 붙여 노출(20초).
# [2026-09-22] 20->45초: 정규장 활성 50종목의 종목당 평가주기가 중앙 약 22초라 20초 창에서는 2회 연속 확인이
# 5일 425건 중 2건만 성립했다(20260921 분석). 창이 넓어진 만큼 아래 두 조건으로 낡은 신호의 확인을 막는다.
# 롤백=20.
BUY_CONFIRM_MAX_GAP_SECONDS = 45
# 연속 확인의 두 통과가 같은 3분봉(r005 ctx.bar_time 동일)이어야 한다. False면 봉이 바뀌어도 확인 유지(이전 동작).
BUY_CONFIRM_REQUIRE_SAME_BAR = True
# 연속 확인 두 통과 사이 실시간가 이탈 상한(%, 직전 통과 시점 실시간가 대비 절대값). 초과하면 확인 횟수를 1로 되돌린다.
# 0 이하면 검사 안 함(이전 동작). 5일 두 통과 사이 이탈은 중앙 0.09%, p90 0.32%였다.
BUY_CONFIRM_MAX_PRICE_DRIFT_PCT = 0.3
# 실시간 현재가 재조회 간격(초) - [2026-09-14] 활성 종목 순회 턴에는 더 이상 쓰이지 않음
# (MAIN_LOOP_MIN_CYCLE_SECONDS로 대체). 정규장/NXT 외 유휴 대기, 동시호가 구간 대기 등
# 턴 자체를 돌지 않는 구간의 폴링 간격으로만 남아있음(r003 참고).
LIVE_PRICE_POLL_INTERVAL_SECONDS = 10  # 20->10 원복: 사용자 확인 결과 기존 10초 유지가 맞음
# [2026-09-14] 메인 루프 턴 최소 간격(초) - 활성 종목이 없어 순회가 즉시 끝나는 경우에도
# 계좌/미체결 동기화 API를 무한정 스팸하지 않도록 하는 안전 바닥값. 사용자 요청(턴 종료
# 즉시 다음 턴 시작)에 따라 기존 LIVE_PRICE_POLL_INTERVAL_SECONDS 고정 대기를 대체.
MAIN_LOOP_MIN_CYCLE_SECONDS = 1
# 계좌/체결 상태 동기화 주기(초)
ACCOUNT_SYNC_INTERVAL_SECONDS = 90
# 3분봉 프레임 갱신 주기(초)
FRAME_POLL_INTERVAL_SECONDS = 20
# 시작 시 과거 바 백필 동기화 범위(초)
FRAME_BACKFILL_SYNC_SECONDS = 600
# 주문 상태 폴링 간격(초) - [2026-09-18] 15->5: 미결 매수 재주문(취소확인/재주문/
# 체결확인)이 전부 이 간격에 순차적으로 걸려 누적 지연되는 문제 확인(183300 코미코
# 08:27:28 제출->08:28:29 체결, 61초 사례), r001 Update log 2026-09-18 참조. 미결
# 주문은 보통 0~1건이라 현재가 폴링(활성종목 전체)만큼의 TPS 부담은 없음.
ORDER_STATUS_POLL_INTERVAL_SECONDS = 5
# 현재가 조회 backoff 초기값(초) / 최대값(초)
LIVE_PRICE_BACKOFF_BASE_SECONDS = 5
LIVE_PRICE_BACKOFF_MAX_SECONDS = 60
# 현재가 stale TTL(초)
LIVE_PRICE_STALE_TTL_SECONDS = 20
# 미결 주문 상태 backoff 최대값(초)
PENDING_STATUS_BACKOFF_MAX_SECONDS = 120
# 메인 루프 연속 오류 허용 최대 횟수
MAIN_LOOP_MAX_CONSECUTIVE_ERRORS = 20
# 라이브 상태 저장 주기(초)
LIVE_STATE_SAVE_INTERVAL_SECONDS = 60
# 미결 매수 주문 대기 여유 시간(초)
PENDING_BUY_GRACE_SECONDS = 90
# 매수 미체결 경고 기준 시간(초)
BUY_ORDER_STALE_WARN_SECONDS = 60
# [2026-09-13] BUY_CHECK_SECONDS_OF_MINUTE(분당 특정 초에만 신규 진입 평가, 2026-09-01
# 도입) 제거 - 진입 신호 발생 시점부터 실제 평가까지 최대 15초+폴링 간격이 추가로 지연되어
# 진입 타이밍이 틀어진다는 실매매 피드백에 따라, 신규 진입 평가를 다시 매 폴링 틱
# (LIVE_PRICE_POLL_INTERVAL_SECONDS 간격, 한 턴 끝나는 즉시 다음 턴 시작)마다 수행하도록
# 원복. r003 Update log 2026-09-01 항목 참조(당시 "폴링 간격의 배수 리스트로 설정하면
# 기존과 동일한 빈도로 복원 가능"이라 명시했던 롤백에 해당).

# --- 매수 미체결 재시도(지정가 추격) ---------------------------------------
# place_buy_order()는 매수1호가(최우선 매수호가)로 순수 지정가 주문을 낸다 - 매도
# (정규장)는 이미 시장가(01)를 쓰는 것과 비대칭. 방금 돌파 신호가 난 종목은 가격이
# 계속 오르는 경우가 많아, 그 가격까지 밀려 내려오지 않으면 영원히 미체결로 남는다
# (2026-08-28 13:14 000720 현대건설 실매매 사례: 500초 넘게 미체결, [BUY STALE]
# 경고만 반복되고 아무 조치 없었음 - r003 Update log 2026-08-28 참조).
# BUY_ORDER_REPRICE_AFTER_SECONDS(폴링 주기 ORDER_STATUS_POLL_INTERVAL_SECONDS=5초
# 단위라 실제로는 다음 폴링 시점에 반영, 10초 정각 보장은 아님)마다 취소 후 더 공격적인
# 가격으로 재주문한다 - 1차=신선한 매수1호가, 2차=매도1호가(스프레드 crossing으로 체결
# 보장), 최종(BUY_ORDER_REPRICE_MAX_ATTEMPTS번째)=시장가(정규장 매도가 이미 쓰는 방식과
# 통일). BUY_ORDER_REPRICE_MAX_CHASE_PCT는 최초 신호가(entry_reference_price) 대비 이
# 비율을 넘으면 추격을 포기하고 취소만 한다 - r002의 추격매수 방지 게이트(CHASE_BUY_BB_GAP
# 등)와 같은 철학을 재시도 로직에도 유지하기 위함(무한정 쫓아가서 사면 그 게이트들이
# 무의미해짐).
BUY_ORDER_REPRICE_AFTER_SECONDS = 10
BUY_ORDER_REPRICE_MAX_ATTEMPTS = 3
BUY_ORDER_REPRICE_MAX_CHASE_PCT = 0.5

# 계좌-감시종목 불일치 로그 출력 최소 간격(초)
WATCHLIST_MISMATCH_LOG_INTERVAL_SECONDS = 300

# --- active/backup 워치리스트 분리 (2026-08-31) -----------------------------
# g002 스캐너 선정 종목 수를 50->100으로 확대하면서, r003가 매 틱 전종목을 완전
# 순차(동시성 없음) API 폴링하는 구조상 부하가 그대로 2배가 되는 걸 막기 위해 도입.
# watch_map(스캐너 점수 내림차순 100개) 중 상위 ACTIVE_WATCHLIST_SIZE개만 실시간
# 폴링(active_set)하고 나머지는 backup_pool로 대기(미폴링)시킨다. active_set 종목이
# 매수 체결(졸업)/HARD_STOP/갭차단/시간초과로 빠지면 backup_pool 선두를 순위 순서대로
# 승격시켜 채운다 - 보유 포지션은 active_set 소속 여부와 무관하게 항상 폴링 대상에
# 포함된다(청산 감시가 끊기면 안 되므로).
# [2026-09-07] 50종목 1턴(active_set 전체 순회) 실측 결과 약 30초(종목당 평균 0.6초,
# API 왕복 지연이 지배적 - 서버 연동 구조상 종목당 처리시간 자체를 줄이기 어려움) -
# LIVE_PRICE_POLL_INTERVAL_SECONDS(10초) 설계 목표의 3배라는 이유로 한때 active_set을
# 20으로 줄이고 ENABLE_WATCHLIST_ROTATION=False로 로테이션을 정지했었음.
# [2026-09-13] 사용자 요청으로 2026-08-31 이전 원래 구조로 완전 원복 - active_set을
# 다시 50으로, 로테이션도 다시 켠다(아래). r004 export도 100으로 늘려(위 R004_EXPORT_TOP_N
# 참조) 스캐너 전체 후보(g002 max_picks=100, 같은 날 g002 Update log 참조)를 그대로
# r004에 반영 - active(50)+backup(50) 여유가 있어야 로테이션이 실제로 교체 대상을
# 찾을 수 있음. 1턴 소요시간이 다시 ~30초대로 늘어나는 트레이드오프는 사용자가 인지하고
# 감수하기로 함(신규 진입 평가 자체는 같은 날 BUY_CHECK_SECONDS_OF_MINUTE 제거로 매 턴마다
# 실행되도록 이미 바꿔둔 상태 - r003 Update log 2026-09-13 참조).
ACTIVE_WATCHLIST_SIZE = 50                    # 실시간 감시(active_set) 상한 (20->50, 2026-09-13 원복)
ACTIVE_WATCHLIST_TIME_DROPOUT_MINUTES = 60    # 이 시간(분) 경과 후부터 박스권(정체) 여부를 확인해 교체를 시작
ACTIVE_WATCHLIST_HARD_TIME_LIMIT_MINUTES = 120  # 박스권이 아니어도 강제로 교체하는 상한(분) - backup_pool 고갈 방지

# [2026-09-07] active/backup 로테이션(TIME_LIMIT 기반 교체) 자체를 켜고 끄는 스위치.
# False면 _rebalance_active_watchlist()가 아예 호출되지 않아 active_set이 최초 로드된
# 상위 ACTIVE_WATCHLIST_SIZE개로 장중 내내 고정된다(교체 없음).
# [2026-09-13] True로 원복 (사용자 요청) - ACTIVE_WATCHLIST_SIZE(50) < R004_EXPORT_TOP_N(100)
# 이라 backup_pool에 실제로 교체 후보가 생기므로 로테이션이 다시 의미를 가짐.
ENABLE_WATCHLIST_ROTATION = True

# ---------------------------------------------------------------------------
# Simulation parameters (g003_trade_simulate_by_date) - r006 실전 매매에는
# 적용되지 않는 시뮬레이션 전용 값.
# ---------------------------------------------------------------------------
# 초기 시뮬레이션 자본금(KRW)
SIM_INITIAL_CAPITAL = 5_000_000
# 웜업 이전 데이터 최대 조회일수
SIM_WARMUP_PRIOR_MAX_DAYS = 20
# 기술적 매도 최소 보유 시간(초)
TECH_SELL_MIN_HOLD_SECONDS = 300
# 시뮬레이션 내부 체크 간격(초)
SIM_CHECK_INTERVAL_SECONDS = 10
# 10초 그리드 시뮬레이션 기본값
SIMULATE_10S_GRID_DEFAULT = True
# 인트라바 볼륨 폴백 활성화
ENABLE_INTRABAR_VOLUME_FALLBACK = True
# 인트라바 볼륨 폴백 최소 진행률
INTRABAR_VOLUME_FALLBACK_MIN_PROGRESS = 0.30

# 시뮬레이션 완화 게이트 (공유 진입 게이트가 거래를 0건으로 막는 것을 방지)
SIM_RELAXED_SHARED_GATES = True
SIM_RELAXED_MIN_SUPPORT_SCORE = 0
SIM_RELAXED_VWAP_MAX_UNDER_PCT = 0.0025
SIM_RELAXED_REQUIRE_CONFIRMED_ABOVE = False
SIM_RELAXED_REQUIRE_MA5_BIAS = False
SIM_RELAXED_ALLOW_BELOW_BB = True
SIM_RELAXED_ALLOW_FALLING_TREND = True


def _apply_risk_profile_overrides() -> None:
	raw_profile = (os.environ.get("AUTO_TRADING_RISK_PROFILE") or os.environ.get("RISK_PROFILE") or "").strip()
	if not raw_profile:
		raw_profile = "neutral"

	alias = {
		"safe": "conservative",
		"conservative": "conservative",
		"보수": "conservative",
		"balanced": "neutral",
		"neutral": "neutral",
		"중립": "neutral",
		"aggressive": "aggressive",
		"attack": "aggressive",
		"공격": "aggressive",
	}
	profile = alias.get(raw_profile.lower())
	if profile is None:
		print(
			"[WARN] Unknown AUTO_TRADING_RISK_PROFILE "
			f"'{raw_profile}'. Expected conservative|neutral|aggressive (or 보수|중립|공격)."
		)
		return

	profile_path = Path(__file__).resolve().parent / "risk_profiles" / f"{profile}.json"
	if not profile_path.is_file():
		print(f"[WARN] Risk profile file not found: {profile_path}")
		return

	try:
		overrides = json.loads(profile_path.read_text(encoding="utf-8"))
	except Exception as exc:
		print(f"[WARN] Failed to load risk profile '{profile}': {exc}")
		return

	if not isinstance(overrides, dict):
		print(f"[WARN] Invalid risk profile format (expected object): {profile_path}")
		return

	applied_keys: list[str] = []
	for key, value in overrides.items():
		if key in globals():
			globals()[key] = value
			applied_keys.append(key)

	print(f"[INFO] Applied risk profile '{profile}' ({len(applied_keys)} overrides)")


_apply_risk_profile_overrides()
