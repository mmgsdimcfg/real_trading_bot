"""R76 shared strategy core.

Overview:
- Single source of truth for live/sim buy-sell decision logic.
- Keep this file as the only place for core rule changes.

Used by:
- r76_trade_live_execute.py
- r76_trade_simulate_by_date.py

Quick usage:
- from r76_strategy_core_shared import R76StrategyConfig, check_buy_condition, check_sell_condition, update_live_price_cross_state

Update log format (append only):
- [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
    summary: <one line>
    impact: <live/sim/common>
    compatibility: <backward-compatible|breaking>

Update log:
- [2026-09-23] type=fix owner=claude
    summary: 사용자 요청(204620 글로벌텍스프리 2026-09-23 09:33 매도 지연 사례 - "매수/매도 컨셉 재검토" 전체
      분석 중 재발견 + Codex 검토) - check_sell_condition()의 price_cross_down 분기에서 즉시손절
      (ma5_bb_down_cross_immediate_pnl, 기본 -0.7%) 체크가 min_pnl(0.0%) 차단보다 뒤에 있어 영원히 도달
      불가능하던 죽은 코드를 수정 - 즉시손절 체크를 먼저 수행하도록 순서를 바꿨다. 이 문제 자체는
      2026-09-21 세션에서 이미 발견/기록됐지만(project 메모 참조) 그때는 고치지 않고 넘어갔던 것.
      pnl<=-0.7% 구간에서만 동작이 바뀐다(기존: 항상 BLOCKED_PNL로 반려 -> 신규: 즉시 손절, score>=1이면
      LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_{score} 아니면 LIVE_PRICE_BB_DOWN_CROSS). pnl>=0.0% 구간(기존
      BLOCKED_PNL/AUX 점수별 최소수익 요건)은 전부 동일 - 경계값(0%, -0.5%, -0.69%, -0.70%, -0.71%, -2%)
      단위 테스트로 확인. g003은 이 함수를 shared_check_sell_condition으로 직접 import해 쓰므로(g003
      check_sell_condition_r76_sim) 별도 수정 없이 동일하게 반영된다 - PaperStrategyTracker(BASIC_CROSS/
      MULTI_FILTER 비교 전용)의 로컬 check_sell_condition은 다른 함수라 영향 없음.
    impact: common (r003 실전/g003 백테스트 공용, _020_shared_reversal_sell을 통해 라이브 매도 경로에 반영)
    compatibility: breaking (pnl<=-0.7% + BB중심선 하향돌파 live cross_down 신호가 겹치는 순간에만 매도
      빈도가 늘어남 - 이전에는 이 조합에서 절대 매도되지 않았음)
- [2026-09-21] type=refactor owner=claude
    summary: r003 매수 조건을 번호 붙은 조건 객체(r005_buy_conditions)로 분리하면서 run_3min_context_pipeline의
      앞/뒤 단계를 공용 함수 build_context_eval(봉 수/지표 준비 + BuyEvalContext 생성)과
      evaluate_context_score(가점 합산 + 개장 직후 보호 + 임계값)로 추출. run_3min_context_pipeline은 시그니처/
      결과(통과 여부, 사유 문자열)가 그대로라 g003 등 기존 호출자는 영향 없음(분리 전후 프레임 대량 비교로 확인).
    impact: common (r003 실전/g003 백테스트 공용)
    compatibility: backward-compatible
- [2026-09-21] type=feat owner=claude
    summary: 1차 익절 상한 + 급등 사다리 익절용 공용 함수 4개 추가(r003 실전/g003 백테스트가 같은 함수를 호출해
      복사본 드리프트 방지). compute_staged_tp1_target_pct(TP1 목표를 ATR 익절선=트레일 무장선으로 상한),
      update_recent_price_samples(종목별 최근 (시각,가격) 표본 갱신), detect_price_surge(60초 내 최저가 대비
      상승폭 >= max(절대 하한, ATR% x 배수) + BB 상단 돌파/거래량 확인), next_surge_ladder_action(TP1 체결가
      기준 +2%/+4% 사다리의 다음 단계와 수량 결정; TP2 미실행 상태에서 +4% 갭이면 TP2+TP3 합산 잔량 전량).
      모두 순수 함수(설정은 인자로 받음)라 r001을 import하지 않는다. 상세 배경은 r001 Update log 참조.
    impact: common (r003 실전/g003 백테스트 공용)
    compatibility: backward-compatible (신규 함수 추가만, 기존 함수 변경 없음)
- [2026-09-20] type=refactor owner=claude
    summary: 사용자 결정(죽은 코드 삭제 / r002로 이동 / 같은 날 재매수 허용 / 거래량 하한 통일 /
      bb_mid_downtrend_block은 별도 전략 변경) 반영.
      (1) r003/g003에 따로 있던 하이브리드 1분봉 트리거(check_buy_condition_1min_hybrid_trigger,
      g003 _sim 복사본)와 개장 갭/거래량 게이트(passes_opening_gap_volume_gate, g003 _sim 복사본)를
      이 파일의 공용 함수로 통합 - 실전/백테스트가 같은 코드를 호출해 복사본 드리프트(2026-09-07/
      09-09/09-18에 반복 발생)를 원천 차단. 옮기기 전 HEAD의 r003/g003 복사본 3벌과 신규 함수를
      실제 1분봉(3일 x 12종목 9,294건)/그리드(2,268건)로 비교해 불일치 0건 확인.
      (2) 죽은 코드 삭제: check_entry_condition_1min/_entry_score_1min(1분봉 Entry Score 게이트 전용)과
      EMA_9 컬럼(그 함수만 사용). 3분봉 단독 파이프라인(check_buy_condition/
      run_buy_condition_pipeline_comment)과 BUY_GATE_CONDITIONS는 g003의 전략 비교 트래커
      (MULTI_FILTER)가 쓰므로 유지.
      (3) 거래량 하한 통일: 트리거에서 1분봉 전용 하한(340/500/비율 0.10)을 삭제하고 3분봉
      min_liquidity_safety를 유일한 하한으로 함(전략 변경).
      (4) bb_mid_downtrend_block은 r001 ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK(기본 True=현행 유지)로
      별도 스위치화 - 최근 라이브 5거래일 단독 차단 0건이지만 다른 게이트가 논리적으로 포함하지는
      않아(Codex 검토) 끄는 결정은 백테스트/섀도 검증 뒤로 분리.
      (5) _score_volume_ratio의 결과가 같은 중복 분기(>=1.2 -> 1점, >=0.7 -> 1점) 병합(40,014개
      입력 변경 전후 비교 불일치 0건).
    impact: common (r003 실전/g003 백테스트 공용)
    compatibility: (1)(2)(4)(5) backward-compatible / (3) breaking - 1분봉 하한만 막고 3분봉 하한은
      통과하던 틱이 이제 트리거를 통과한다(최근 라이브 5거래일 기준 점수 단계 도달 217틱, 그 뒤
      점수 10점/2회 연속 확인이 다시 걸러냄)
- [2026-09-17] type=fix owner=claude
    summary: 사용자가 "예수금 부족으로 매수가 막힌 경우가 있나" 질의를 계기로 실매매
      매수 0건이 여러 날 이어지는 것을 발견, 확대 요청("20년차 트레이더 관점에서 현재
      매수 진입 조건 점검 + 타이트한 조건으로 익절 가능했던 매수를 놓친 경우가 없는지
      검토/보완")에 따른 로그 분석 기반 수정. 2026-09-14/16/17(전체/근전체 세션, 매수
      0건) gate_steps_diagnostic STEPS 집계(178,572건) 결과 stochastic_buy_signal이
      전체 REJECT의 94.6%에서 실패(전체 8개 판정 중 최다) - 특히 "나머지 7개는 전부
      통과"인 근접리젝(exactly-1-gate-fail, 2,759건)의 50.5%(1,394건)가 이 게이트
      단독 실패였음. 원인 분리 결과 실패의 상당수가 _gate_stochastic_buy_signal의
      williams_equiv_ok 조건 중 "stoch_k > stoch_k_prev"(직전 3분봉 대비 %K 상승
      필수) 단일 틱 요건 - 175330 JB금융지주 2026-09-14 09:09 사례로 실증(RSI/ADX
      100 포화, %K가 90.0->81.82로 밴드(30~90) 안에서 살짝 눌렸다는 이유만으로 리젝,
      그 시점 매수 가정 시 09:27까지 20분 내 +4.25% - STAGED_TP1_PCT 3.0% 손절 없이
      달성). 해당 조건 제거(아래 _gate_stochastic_buy_signal 본문 참조), 밴드 범위
      체크는 유지. 2026-09-11(부분세션)/09-13(주말 휴장)/09-15(장마감 후 2분만 실행)는
      코드가 아니라 운영/시장 요인으로 확인되어 이번 분석/수정 대상에서 제외 - 실제
      코드 귀책 매수 0건 구간은 09-14~09-17(R78, 2026-09-14 STOCH_D_BUY_MIN/
      DI_SPREAD_MIN_REQUIRED 추가 직후부터)로 특정됨. di_spread_min(근접리젝
      359건, 13%)도 동일 계열 의심 사례(047040 대우건설/138080 오이솔루션 등 신선한
      cross_up 신호가 후행 DI스프레드에 막힌 경우) 확인했으나, DI_PLUS/DI_MINUS는
      hybrid 경로(HYBRID_3MIN_CONTEXT_GATES)에서 cross_eval 컨텍스트가 없어(1분
      트리거가 크로스를 이미 판정) 안전한 예외 조건을 결합 파이프라인 양쪽에 드리프트
      없이 걸기 어려워 이번엔 수정 보류 - g003 백테스트로 DI_SPREAD_MIN_REQUIRED
      완화 여지를 별도 검증 권장(아래 r003 Update log 동일 날짜 참조, 표시 스코어
      버그도 함께 수정).
    impact: common (r003 실전/g003 백테스트 공용 - stochastic_buy_signal 통과 기준 완화)
    compatibility: breaking (매수 판정 결과가 바뀜 - 리젝 감소/매수 빈도 증가 예상,
      g003 --date 20260914,20260916,20260917 백테스트로 재검증 권장)
- [2026-09-14] type=feat owner=claude
    summary: R78(사용자 요청 ①②, 실거래 141건 로그 분석 기반) - 신규 필수 게이트 2개
      추가. (R78-①) _gate_stochastic_buy_signal에 STOCH_D_BUY_MIN(50.0) 하한 조건
      추가 - 기존엔 %K만 확인하고 %D 자체엔 하한이 없었음(실거래 분석: %D<50 승률
      24-33%(n=64) vs 50-80 42%(n=50)). (R78-②) 신규 함수 _gate_di_spread_min +
      BUY_GATE_CONDITIONS 9번째 항목("di_spread_min") 추가 - DI스프레드(+DI--DI)가
      DI_SPREAD_MIN_REQUIRED(10.0) 미만이면 매수 차단(실거래 분석: 0-10구간 25.6%
      (n=39) vs 10-20구간 36.4%(n=33) vs 20+구간 62.5%(n=8, 표본작음), 가장 나쁜
      구간만 배제하는 보수적 값으로 시작). R77-D의 adx_strength 가점(ADX_DI_SCORE_*)
      과는 독립적인 별개의 필수조건 - 서로 배타적이지 않음.
    impact: common (r003 실전/g003 백테스트 공용 - BUY_GATE_CONDITIONS 8개->9개)
    compatibility: breaking (매수 판정 결과가 바뀜 - 백테스트로 거래빈도/승률
      트레이드오프 검증 필요, 아래 참조)
- [2026-09-13] type=refactor owner=claude
    summary: 사용자가 제시한 R77(전략 개선안 - 코드상으로는 g003/r002 등 기존 파일에
      적용, 별도 파일 아님) 항목 중 사전검토 결과 문제 없다고 판단된 4건 적용:
      (R77-A) pre_cross_accumulation_bar 필수 게이트를 삭제하고 동일 판정 로직을
      가점(+1, PRE_CROSS_ACCUM_SCORE)으로 전환 - 2026-08-25 로그 분석에서 리젝 사유
      1위(1,284건)였던 과거 이력 근거. (R77-B) volume_up_direction(+1, 직전봉 대비
      거래량 증가 단순비교) 가점 삭제 - 노이즈성 신호로 판단. (R77-D) adx_strength를
      "ADX 크기만" 보던 3단계 티어에서 "+DI>-DI 방향성 필수 + ADX/DI스프레드" 기준
      2~3점 구조로 재설계 - 기존 티어는 하락추세에서도 ADX가 높으면 만점을 줄 수
      있었음. DI가 2026-06-28에 스코어에서 "이중반영"으로 제거된 이력이 있으나 그건
      당시 스코어 레벨 중복이었고, 현재 DI 방향 확인은 bb_mid_cross_up 게이트의
      uptrend_continuation 예외에만 국한돼 live_cross_up/close_cross 경로엔 방향성
      확인이 전혀 없었던 공백을 메움. (R77-G) BB 상단 여유폭을 ATR로 정규화한 가점
      (+1/+2, bb_upper_room_atr) 신규 추가 - 기존 bb_upper_gap_min 게이트(고정 0.25%)
      는 그대로 유지, 그 위에 저변동/고변동 종목 형평성을 보정하는 보너스.
      R77-C(BB slope 하드컷 -1.5~-2.0)는 검토 결과 이미 BB_SLOPE_MIN_PCT=-2.0으로
      요청 범위 안이라 코드 변경 없음. R77-E(시간대별 RVOL)/R77-F(시장 대비 상대강도)
      는 라이브에 필요한 인프라(과거 종목별 시간대 거래량 프로파일, 실시간 지수 시세)가
      전혀 없어 이번엔 보류 - 별도 스코핑 필요.
    impact: common (r003 실전/g003 백테스트 공용 - BUY_GATE_CONDITIONS 9개->8개,
      BUY_SCORE_RULES 11개->12개, 만점 22점->24점 변경. BB_BUY_SCORE_THRESHOLD(10)/
      OPENING_GUARD_SCORE_THRESHOLD(12)는 그대로 두고 백테스트로 재검증 예정)
    compatibility: breaking (매수 판정 결과가 바뀔 수 있음 - 5일 백테스트 검증 권장,
      기존 A-D 실험과 동일 방법론으로 재검증 예정)
- [2026-09-09] type=feat owner=claude
    summary: gate_steps_diagnostic() 신규 추가 - BUY_GATE_CONDITIONS(또는 호출측이 넘긴
      임의의 gates 리스트, 예: HYBRID_3MIN_CONTEXT_GATES)를 조기종료 없이 전부 평가해
      "gate_name(P|F)" 문자열로 이어붙여 반환하는 로그 표시 전용 진단 함수(사용자 요청 -
      r003 [REJECT] 로그에 매수조건 스텝별 pass/fail 요약 추가). run_buy_condition_
      pipeline_comment/run_3min_context_pipeline과 달리 순차 통과형 short-circuit을
      하지 않고 모든 게이트를 독립적으로 재평가하므로, 리젝된 종목이 실제로 걸린 게이트
      외에 나머지 게이트도 통과했는지 한눈에 볼 수 있다. 실제 매수 판정 로직에는 전혀
      관여하지 않는 순수 진단/로그용 함수(r003 Update log 2026-09-09 참조).
    impact: live/sim (로그 전용, 판정 로직 변경 없음)
    compatibility: backward-compatible (신규 함수 추가만, 기존 판정 함수는 그대로 유지)
- [2026-09-06] type=fix owner=claude
    summary: _evaluate_bb_mid_cross()의 uptrend_continuation(크로스 이벤트 없이 지속
      추세만으로 진입 허용) 판정에서 bb_slope_pct > 0.0 하드코딩 리터럴을 신규 상수
      UPTREND_CONT_SLOPE_MIN_PCT(-0.05, r001)로 교체. 8/17~9/4 실매매 로그 분석(사용자
      요청) 결과, 이 0.0 기준이 필수 게이트 bb_slope_rising의 BB_SLOPE_MIN_PCT(-2.0%)
      보다 훨씬 엄격해 대체 진입 경로가 필수 게이트보다 더 까다로운 역전 상태였음을 발견.
      196170 알테오젠 2026-08-21 09:33~09:40 사례: ADX 80대/+DI>>-DI/MA5 상승 등 나머지
      조건은 전부 강한 지속 추세를 가리켰는데, BB_MIDDLE이 급등을 뒤늦게 따라잡는
      후행지표 특성상 bb_slope_pct가 0 근방(-0.016%~0.071%)에서 노이즈로 진동해 09:33엔
      리젝, 09:36에야 순전히 노이즈 타이밍으로 통과 - 진입 여부가 몇 분 단위 운에 좌우됨.
      나머지 7개 조건(ADX>=30/DI우세/3-5봉 BB위 유지/MA5상승/모멘텀 등)은 그대로 유지.
    impact: common
    compatibility: backward-compatible (조건이 소폭 완화되어 uptrend_continuation 경로
      진입 빈도가 약간 늘어날 수 있음)
- [2026-08-28] type=feat owner=claude
    summary: run_3min_context_pipeline()/HYBRID_3MIN_CONTEXT_GATES 신규 추가 -
      ENABLE_1MIN_TRIGGER_3MIN_CONTEXT(r001) 하이브리드 경로 전용. 403870 HPSP
      2026-08-28 실매매 로그 분석 결과, run_buy_condition_pipeline_comment의
      bb_mid_cross_up 게이트가 3분봉 종가 확정(최대 180초 지연) 시점에만 크로스를
      인정하다 보니, 크로스가 확정된 그 순간엔 이미 candle_bullish_and_chase_guard의
      갭 문턱(BB_MID_CHASE_MAX_GAP_PCT 등)을 넘어버려 신호는 났는데(cross_pass=True)
      매번 CHASE_BUY_BB_GAP/CANDLE_NOT_BULLISH로 반려되는 패턴이 하루 중 2차례(09:09,
      09:27) 반복 확인됨 - BB_MID가 3분봉 기준 후행지표라는 구조적 한계. 호출측(r003/
      g003)이 1분봉 자체 기준(check_buy_condition_1min, 1분봉 자신의 open/BB로 크로스+
      양봉+추격가드를 판정 - 트리거 프레임과 판정 기준 프레임을 일치시켜 "1분봉에서
      막 크로스했는데 3분봉 open 대비로는 이미 갭 초과" 같은 불일치 재현을 방지)로
      먼저 트리거를 확인한 뒤, 이 함수로 3분봉 컨텍스트(BB기울기/다운트렌드차단/매집봉/
      BB상단여유/스토캐스틱/윌리엄스/유동성 + 가점 스코어)만 재검증하는 2단계 구조.
      BUY_GATE_CONDITIONS/BUY_SCORE_RULES를 그대로 재사용해 판정 로직 드리프트 방지
      (bb_mid_cross_up/candle_bullish_and_chase_guard 2개만 HYBRID_3MIN_CONTEXT_GATES에서
      제외 - 1분봉 트리거가 이미 확인한 항목의 중복 판정 방지).
    impact: common
    compatibility: backward-compatible (신규 함수 추가만, ENABLE_1MIN_TRIGGER_3MIN_CONTEXT
      기본값 False라 기존 경로 미사용 시 동작 변화 없음; r007 --date 20260828 --codes
      403870 백테스트로 검증 필요 - r001 Update log 2026-08-28 참조)
- [2026-08-26] type=fix owner=claude
    summary: check_sell_condition()의 price_cross_down(라이브가 BB중심선 하향 돌파) 분기가
      AUX_REVERSAL_SCORE 분기와 동일한 _sell_support_score를 쓰면서도 훨씬 느슨한 수익 요건을
      쓰고 있던 불일치 수정. AUX_REVERSAL_SCORE 분기는 score 4/3/2에 각각 pnl>=0.3%/0.8%/1.5%를
      요구하는데, price_cross_down 분기는 ma5_bb_down_cross_min_pnl(0.0%)만 넘으면 score>=2로
      즉시 확정 매도였음 - 083650 비에이치아이 2026-08-26 실매매 로그에서 진입 자체가 BB중심선
      근처 얕은 마진 진입(live=65,900 < bb_mid=65912.5)이었던 탓에 매수 14분 후 자연스러운
      되돌림만으로 score=4, pnl≈0%가 되어 즉시 매도(LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_4) ->
      -0.13% 손실 확정된 사례로 발견. pnl<=ma5_bb_down_cross_immediate_pnl(-0.7%) 하드손절
      분기는 그대로 두고, score 기반(비-하드손절) 확정 매도에는 AUX_REVERSAL_SCORE와 동일한
      점수별 최소수익 요건(aux_sell_min_pnl_score2/3/4)을 적용 - 미달 시
      LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_SCORE_N으로 보류.
    impact: common
    compatibility: breaking (BB중심선 하향돌파 직후 손익분기점 근처 확정매도 빈도 감소 - 동일
      신호가 진짜 하락추세로 이어지면 손절액이 더 커질 수 있으나, 얕은 되돌림 노이즈로 인한
      조기 손절/이익축소는 줄어듦. r003 [SELL HOLD] 로그에 LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_SCORE
      prefix 추가)
- [2026-08-25] type=fix owner=claude
    summary: 2026-08-25 실매매 로그 분석(163k줄) 결과 필수조건 2-b(직전 매집봉 확인,
      NO_PRE_CROSS_ACCUM_BAR_LOOKBACK_5)가 크로스 이후 다운스트림 리젝 중 압도적 1위로
      확인됨(정규화 기준 1,284개 개별 발생, 블록 시간 합계 2,385분 - 2위 CHASE_BUY_BB_GAP
      986분의 2.4배). 정규장이 중반부터 상승장으로 전환됐음에도 하루 종일 매수가 6건뿐이었던
      핵심 병목. 원인은 "그 봉 저가~고가가 BB중간값 포함 AND 양봉 AND 거래량>VOL_MA20"을
      같은 한 봉에서 동시 요구하는 3중 AND 조건이 너무 희귀한 조합이라는 점 - 2026-08-21에
      이미 거래량 기준을 완화했으나(직전봉 대비->VOL_MA20 대비) 그 후로도 최대 병목으로 남음.
      (1) PRE_CROSS_ACCUM_LOOKBACK_BARS 5->8봉(r003) 재확장, (2) 거래량 기준을 "VOL_MA20
      초과"에서 "VOL_MA20의 PRE_CROSS_ACCUM_VOL_RATIO_MIN(0.8, r003) 이상"으로 완화.
      CANDLE_GAIN_MIN_PCT도 0.0%->-0.1%(r003)로 완화 - 라이브 폴링 시점 미세 틱노이즈로
      정상 돌파 구간에서도 순간 음전(-0.01~-0.2%대)되어 CANDLE_NOT_BULLISH로 리젝되는
      사례(669건, 블록시간 586분)를 완화.
    impact: common
    compatibility: breaking (매수 필수조건 2-b/3 통과 기준 완화로 매수 빈도 증가 예상; r007
      --date 20260825 백테스트로 리젝 사유 재분포 및 신규 매수 건 확인 완료)
- [2026-08-24] type=feat owner=copilot
    summary: 필수조건 2(추격 매수 방지, "CHASE_BUY_BB_GAP")에서 uptrend_continuation 경로
      전용 분기 추가 - live_cross_up/close_cross 없이 우상향 추세 지속만으로 진입하는 경우엔
      BB_MID 갭 상한을 BB_MID_CHASE_MAX_GAP_PCT(0.35%)->UPTREND_CONT_CHASE_MAX_GAP_PCT(0.6%,
      r003)로 완화하고, 대신 RSI<UPTREND_CONT_CHASE_RSI_MAX(75, r003) 과열 여부를 추가
      검증한다. 017670 SK텔레콤 2026-08-24 13:12 반등 사례: BB_MID가 후행지표라
      uptrend_continuation이 13:17에야 확정됐는데 그땐 이미 갭이 0.35% 문턱을 넘어있어 그날
      내내 CHASE_BUY_BB_GAP로만 리젝되고 재진입 자체가 봉쇄됐음. live_cross_up/close_cross
      경로(신선한 크로스 이벤트)는 기존 0.35% 그대로 유지 - 그쪽은 BB_MID 지연이 문제되지 않음.
    impact: common
    compatibility: backward-compatible (uptrend_continuation 경로에만 영향, 기존 크로스
      기반 진입의 추격 방지 기준은 그대로)
- [2026-08-24] type=refactor owner=copilot
    summary: run_buy_condition_pipeline_comment의 필수조건 2(BB 중앙선 상향 돌파 판정: live
      cross signal / close 기준 크로스 / 5봉 룩백 / 우상향 추세 지속) 블록을 신규 함수
      _evaluate_bb_mid_cross()로 추출. 사용자가 051900 LG생활건강 09:12 사례(live가 BB_MID
      위에 있는데도 signal=None인 이유)를 물어보며 r006의 진단 로그 함수 _buy_condition_
      snapshot이 이 실제 판정과 무관하게 즉시 2봉 close_cross만 따로 계산하고 있던 걸 발견 -
      실제로는 5봉 룩백으로 통과했을 케이스가 로그엔 close_cross=False로 오표시되고 있었음.
      공유 함수로 추출해 r006의 로그 스냅샷도 동일 함수를 재사용하도록 배선(r006 Update log
      2026-08-24 참조) - 두 곳이 따로 계산하는 구조 자체를 없애 향후 드리프트 재발을 원천 차단.
      순수 추출 리팩터라 실제 매수 판정 로직/스코어/문턱값은 전혀 바뀌지 않음.
    impact: common
    compatibility: backward-compatible (r007 --date 20260821 --codes 051900 백테스트로 REJECT
      사유 집계가 리팩터 전후 완전히 동일함을 확인)
- [2026-08-24] type=fix owner=copilot
    summary: 2026-08-23에 run_buy_condition_pipeline_comment에 추가한 len(frame)<BB_PERIOD(20)
      최소 봉 수 가드를 되돌리고 원래의 len(frame)<2로 복원. 그 가드는 "HTS 차트에서 볼린저밴드가
      3분봉 10시/1분봉 09:20 이전엔 그려지지 않는다"는 관측에 근거했었는데, 사용자가 MTS에서
      동일 종목(051900 LG생활건강)을 확인한 결과 1분봉/3분봉 모두 09:00 장 시작 즉시 볼린저밴드가
      정상적으로 그려지고 그 구간에서 골든크로스도 유효하게 발생함을 확인 - 즉 20봉 미만 구간의
      BB_MIDDLE/BB_STD(rolling min_periods=1 성장형 평균)이 "가짜"라는 전제 자체가 HTS 자체
      표시 문제(혹은 다른 원인)였을 가능성이 높고, MTS/코드의 실제 계산 방식과는 무관했음.
      calculate_indicators()의 BB 계산 로직 자체는 애초에 변경한 적이 없음(rolling(window=
      BB_PERIOD, min_periods=1) 그대로) - 문제는 오직 그 값을 매수 판정에 쓰기 시작하는 시점을
      인위적으로 늦춘 게이트였음.
    impact: common
    compatibility: breaking (장 시작 후 약 60분간 다시 3분봉 신규 매수 판정이 재개됨 - 2026-08-23
      이전 동작으로 복귀)
- [2026-08-23] type=fix owner=copilot
    summary: run_buy_condition_pipeline_comment(현재 활성 3분봉 다중필터 경로, r006에서
      ENABLE_1MIN_GOLDEN_CROSS_BUY=False라 실제로 매매를 결정하는 그 함수)의 최소 봉 수
      요건이 len(frame)<2(사실상 3분봉 기준 장 시작 후 약 6분부터 판정 시작)로만 걸려있어,
      calculate_indicators()가 BB_PERIOD(20) 미만 구간에서 rolling(min_periods=1)으로
      만든 불안정한 BB_MIDDLE/BB_STD(사실상 2~3봉짜리 초단기 평균)를 그대로 판정에 썼음 -
      사용자가 HTS 차트에서 볼린저밴드가 3분봉은 10시부터, 1분봉은 09:20부터만 그려지는
      것을 보고 발견. 완전히 동일한 원인이 check_buy_condition_1min(1분봉 골든크로스
      대체경로, 현재 비활성)에서는 2026-08-17에 이미 len(frame)<BB_PERIOD 가드로 수정된
      바 있었는데, 실제 매매를 결정하는 이 3분봉 경로에는 그 수정이 적용되지 않고 있었음.
      동일한 가드로 통일. OPENING_GUARD_MINUTES(15분, 점수 상향 방식)는 그대로 두되, 이제
      BB_PERIOD 미만 구간(3분봉 기준 최초 약 60분)에는 그보다 먼저 이 가드가 걸려 판정
      자체가 발생하지 않는다.
    impact: common
    compatibility: breaking (장 시작 후 약 60분간 3분봉 신규 매수가 전면 차단됨 - 그 구간에서
      나오던 신호는 통계적으로 신뢰하기 어려운 "가짜 돌파"였을 가능성이 높음)
- [2026-08-23] type=feat owner=copilot
    summary: _buy_support_score()에 장기 추세 정합성 가점 신규 추가 - EMA20 > EMA60이면
      +EMA_TREND_ALIGN_SCORE(기본 2, r003)점. 사용자가 제안한 3분봉 Signal V1 스코어
      구성안(EMA20>EMA60/BB Middle상승/RSI/ADX/VolumeRatio/StochK>D) 중 기존 코드에
      전혀 없던 유일한 신호였음 - 나머지(BB돌파/RSI/ADX/VolumeRatio/Stochastic)는 이미
      필수조건 또는 가점으로 반영돼 있었고, 제안안의 V2 항목(윌리엄스%R/BB폭/추격매수
      방지/개장시간필터/거래대금/시장상대강도)도 대부분 이미 구현돼 있었음(r002 RS15,
      turnover 필터 등). calculate_indicators()에 EMA_60 컬럼 추가(EMA_20은 1분봉
      게이트용으로 기존에 이미 계산되던 컬럼을 재사용, 3분봉 프레임에도 동일하게 존재).
    impact: common
    compatibility: backward-compatible (기존 가점 항목에 최대 +2점 추가되는 것뿐, 새 필수조건
      아님 - 매수 문턱을 낮추는 방향이라 빈도가 소폭 늘어날 수 있음)
- [2026-08-23] type=feat owner=copilot
    summary: check_entry_condition_1min() 신규 추가 - 3분봉 다중필터(run_buy_condition_pipeline_
      comment) 통과 이후 1분봉에서 진입 타이밍을 한 번 더 점수제로 검증하는 2단계 게이트.
      필수조건 없이 EMA9>EMA20(+2)/Close>EMA9(+1)/직전고점돌파(+2)/거래량>VOL_MA20(+2) 4개를
      점수화해 ENTRY_SCORE_THRESHOLD(기본 5, r003) 이상이면 통과. calculate_indicators()에
      EMA_9/EMA_20 컬럼 신규 추가(EMA 계열 지표가 기존에 전혀 없었음, MA_5 SMA만 존재).
      r006/r007의 실제 배선은 ENABLE_1MIN_ENTRY_SCORE_GATE(r003) 플래그로 제어되며, 기존
      ENABLE_1MIN_GOLDEN_CROSS_BUY(단독 대체 경로, 섹션12)와는 완전히 별개.
    impact: common
    compatibility: breaking (신규 2단계 게이트로 매수 빈도가 줄어들 수 있음, False로 즉시 롤백 가능)
- [2026-08-21] type=fix owner=copilot
    summary: 필수조건 2-b(직전 매집봉 확인)의 거래량 증가 판정을 "직전봉 대비"에서 "VOL_MA20(20봉
      평균) 대비"로 변경, 룩백도 3->5봉으로 확장(PRE_CROSS_ACCUM_LOOKBACK_BARS, r003). 어제
      추가된 이 조건이 257720 실리콘투 2026-08-21 09:57 BB중앙선 상향 돌파를 막은 사례를 사용자가
      발견 - 돌파 직전 3봉 중 09:48봉이 직전 09:45봉(장중 거래량 스파이크)보다 거래량이 낮게
      나와 "거래량 증가" 조건에서 탈락했는데, VOL_MA20 대비로는 낮지 않았음. 봉 대 봉 비교는
      매집 국면에서도 노이즈로 인해 오탈락하기 쉬워, 더 안정적인 평균 대비 기준으로 교체.
    impact: common
    compatibility: breaking (필수조건 2-b 통과 기준 완화로 매수 빈도가 다시 늘어날 수 있음)
- [2026-08-20] type=feat owner=copilot
    summary: run_buy_condition_pipeline_comment에 필수조건 2-b(직전 매집봉 확인) 신규 추가 -
      BB 중앙선 상향 돌파 직전 PRE_CROSS_ACCUM_LOOKBACK_BARS(3, r003)개 봉 중 하나라도
      (1) 그 봉의 저가~고가가 그 봉 시점 BB중간값을 포함, (2) 양봉, (3) 거래량이 바로 이전
      봉보다 큼을 모두 만족해야 통과. 직전봉 1개만 보면 매수 빈도가 크게 줄어들어 최근 3봉
      룩백으로 완화. ENABLE_PRE_CROSS_ACCUM_BAR_CHECK(r003) False 시 건너뜀(즉시 롤백 가능).
    impact: common
    compatibility: breaking (신규 필수조건 추가로 매수 빈도가 줄어들 수 있음)
- [2026-08-17] type=fix owner=copilot
    summary: 사용자가 캡처한 실제 차트(2026-08-14 현대차/삼성전기 3분봉)에서 육안상 매수가
      나와야 할 지점이 실제로는 걸러지는 것을 발견해 원인 2건 수정. (1) 필수조건 5(스토캐스틱)의
      상한을 STOCH_BUY_MAX(50)에서 config.stoch_overbought(96, 진짜 과열 기준)로 완화 -
      %K는 10봉 오실레이터라 BB중앙선 돌파 순간엔 이미 50을 넘는 게 정상적인 움직임인데,
      50을 상한으로 걸어놓아 정상적인 돌파 대부분을 걸러내고 있었음(현대차 12:12 사례로 확인).
      (2) 필수조건 1(BB 기울기)의 하드코딩 임계값 -0.7%를 신규 상수 BB_SLOPE_MIN_PCT(-2.0%,
      r003)로 분리 및 완화 - 60분 lookback 기울기가 급등 후 되돌림 국면에서 가격/스토캐스틱/
      윌리엄스%R이 이미 반전된 뒤에도 한참 뒤늦게 양전환되어 진입을 막고 있었음(삼성전기 10:42
      사례로 확인). 두 수정 모두 2026-08-14 실데이터 재생으로 검증 완료.
    impact: common
    compatibility: breaking (매수 필수조건 통과 기준이 완화되어 매수 빈도/타이밍이 달라짐)
- [2026-08-17] type=feat owner=copilot
    summary: run_buy_condition_pipeline_comment(3분봉 다중필터 매수)에 필수조건 5/6으로
      스토캐스틱 패스트/윌리엄스 %R 매수신호를 신규 추가. 사용자가 원래 의도한 설계는
      "3분봉 BB 중앙선 골든크로스 시점에 스토캐스틱 패스트와 윌리엄스 %R의 매수신호가
      함께 나올 때 매수"였는데, 실제 코드는 2026-06-28 리팩토링(불필요 로직 제거 명목)
      이후 스토캐스틱은 매도 판단에만, 윌리엄스 %R은 어디에도 쓰이지 않는 상태였음
      (STOCH_BUY_MIN/MAX, WILLIAMS_BUY_FLOOR/OVERBOUGHT_CEIL 등 관련 config 필드도
      R76StrategyConfig에 정의만 되고 아무 곳에서도 읽히지 않는 죽은 필드였음). 필수조건 5는
      스토캐스틱 %K가 %D를 상향 돌파하거나(골든크로스) %K>%D이면서 STOCH_BUY_MIN~MAX
      구간(과열 아님)일 때 통과, 필수조건 6은 윌리엄스 %R이 직전봉 대비 상승 중이면서
      WILLIAMS_BUY_FLOOR~WILLIAMS_OVERBOUGHT_CEIL 구간(바닥권 탈출~과열 직전)일 때
      통과. r003의 ENABLE_1MIN_GOLDEN_CROSS_BUY도 False로 되돌려 이 3분봉 경로가 다시
      기본 매수 경로가 되도록 함(r003 Update log 참조).
    impact: common
    compatibility: breaking (매수 필수조건이 2개 늘어나 매수 빈도가 줄어들 수 있음)
- [2026-08-17] type=refactor owner=copilot
    summary: 죽은 코드 제거 - buy_1st_live_price_above_bb_mid_within_gap_comment ~
      buy_10th_prev_close_and_volume_soft_guard_comment (10개 함수, "가짜" 10단계 매수 파이프라인)가
      r006/r007 어디에서도 호출되지 않는 것으로 확인됨(run_buy_condition_pipeline_comment는 자체
      로직을 재구현하고 있어 이 함수들과 무관). 실제 사용되지 않는 죽은 코드라 전체 삭제.
    impact: common
    compatibility: backward-compatible (미사용 함수 삭제만, 동작 변화 없음)
- [2026-07-18] type=fix owner=copilot
    summary: BB_MID_DOWNTREND 조건 완화 - 가격이 이미 BB_MID 위에 있을 경우 BB_MID 하락추세 차단 해제. 후행 지표 아티팩트로 인한 오진입 차단 방지.
    impact: common
    compatibility: backward-compatible (BB_MID 하락 시 가격이 이미 돌파한 경우 추가 허용)
- [2026-07-02] type=fix owner=copilot
    summary: 거래량비율 점수에 0.7~1.2배 구간(+1점) 추가. 093370 사례처럼 vol_ratio 0.7~1.2 사이에서 0점 처리되어 유효 반전신호가 과도하게 반려되던 문제 완화.
    impact: common
    compatibility: backward-compatible (more entries pass score gate)
- [2026-06-28] type=refactor owner=copilot
    summary: 불필요 로직 제거(Williams/RSI_SIGNAL/DI이중/EMA미사용) + VWAP·거래량방향·BB폭확장·MA5방향 신호 추가; BB_BUY_SCORE_THRESHOLD 8->9
    impact: common
    compatibility: backward-compatible
- [2026-06-28] type=fix owner=copilot
    summary: buy_9th hard_floor max(0.30,req*0.75)->max(0.65,req*0.90) 저거래량 진입 완전 차단
    impact: common
    compatibility: backward-compatible
- [2026-06-26] type=fix owner=copilot
    summary: buy_9th hard_floor 공식 강화 max(0.15,req*0.65)->max(0.30,req*0.75) 저거래량 종목 매수 차단
- [2026-06-25] type=feat owner=copilot
    summary: (1) BB_MID_CHASE_MAX_GAP_PCT 1.0%→0.7% 강화; (2) BB_UPPER_GAP_MIN_PCT 0.25%→0.5% 상향; (3) 개장 직후(09:00~09:08, 첫 3봉) CLOSE/UPTREND_CONT 진입 시 score threshold를 10으로 상향(OPENING_GUARD).
    impact: common
    compatibility: backward-compatible
- [2026-06-17] type=feat owner=copilot
    summary: BB 중간선 최근 4봉(12분) 연속 우하향 시 매수 차단 조건 추가 (BB_MID_DOWNTREND_4BARS); _bb_middle_is_downtrend 함수 신규.
    impact: common
    compatibility: backward-compatible
- [2026-06-17] type=fix owner=copilot
    summary: (1) 크로스 룩백을 3봉→5봉으로 확장하여 최근 15분 내 크로스 인정, (2) UPTREND_CONT 진입경로 추가: live>BB중간선+종가상승+ADX30이상++DI>-DI+BB위3봉이상이면 크로스 없이도 매수 허용.
    impact: common
    compatibility: backward-compatible
- [2026-06-05] type=fix owner=copilot
    summary: (한글) BB 중간값 진입 보조조건을 최근 3봉 유지(2/3)에서 "현재가 > 직전 3분봉 종가 && 현재가 > BB 중간값"으로 변경.
    impact: common
    compatibility: backward-compatible
- [2026-05-10] type=refactor owner=copilot
    summary: created r76 shared core module and centralized core strategy logic.
    impact: common
    compatibility: backward-compatible
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd
from r001_define_config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD_MULTIPLIER,
    BB_SLOPE_LOOKBACK_BARS,
    BB_SLOPE_MIN_PCT,
    BB_MID_DOWNTREND_BARS,
    BB_MID_CHASE_MAX_GAP_PCT,
    UPTREND_CONT_CHASE_MAX_GAP_PCT,
    UPTREND_CONT_CHASE_RSI_MAX,
    UPTREND_CONT_SLOPE_MIN_PCT,
    BB_UPPER_GAP_MIN_PCT,
    CANDLE_GAIN_MAX_PCT,
    CANDLE_GAIN_MIN_PCT,
    MIN_ENTRY_VOL_MA,
    MIN_ENTRY_VOLUME,
    MIN_ENTRY_TURNOVER_KRW,
    ENABLE_MIN_ENTRY_ATR_FILTER,
    MIN_ENTRY_ATR_TO_TP1_RATIO,
    STAGED_TP1_PCT,
    REQUIRE_OBV_SIGNAL_CROSS,
    OBV_BREAKOUT_LOOKBACK_BARS,
    OBV_CONFIRM_SCORE,
    MACD_FAST,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW,
    MA_PERIOD,
    MFI_PERIOD,
    OBV_MA_PERIOD,
    RSI_PERIOD,
    RSI_SIGNAL_PERIOD,
    STOCH_BUY_MIN,
    STOCH_D_BUY_MIN,
    STOCH_D_PERIOD,
    STOCH_K_PERIOD,
    VOLUME_MA_PERIOD,
    WILLIAMS_BUY_FLOOR,
    WILLIAMS_D_PERIOD,
    WILLIAMS_OVERBOUGHT_CEIL,
    WILLIAMS_R_PERIOD,
    OPENING_GUARD_MINUTES,
    OPENING_GUARD_SCORE_THRESHOLD,
    ENABLE_PRE_CROSS_ACCUM_BAR_CHECK,
    PRE_CROSS_ACCUM_LOOKBACK_BARS,
    PRE_CROSS_ACCUM_VOL_RATIO_MIN,
    EMA_20_PERIOD,
    EMA_60_PERIOD,
    EMA_TREND_ALIGN_SCORE,
    PRE_CROSS_ACCUM_SCORE,
    ADX_DI_SCORE_MIN_ADX,
    ADX_DI_SCORE_STRONG_ADX,
    ADX_DI_SCORE_STRONG_SPREAD,
    BB_UPPER_ROOM_ATR_TIER1,
    BB_UPPER_ROOM_ATR_TIER2,
    DI_SPREAD_MIN_REQUIRED,
    ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK,
    HYBRID_1MIN_TRIGGER_LOOKBACK_BARS,
    HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT,
    HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT,
    HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT,
    HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR,
    HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT,
    HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT,
    OPENING_GAP_GATE_WINDOW_MINUTES,
    OPENING_GAP_MIN_PCT,
    OPENING_GAP_MAX_PCT,
    OPENING_GAP_HARD_FLOOR_PCT,
    OPENING_MIN_EARLY_VOLUME_RATIO,
)


@dataclass(frozen=True)
class R76StrategyConfig:
    live_price_bb_buffer_pct: float
    live_price_cross_confirm_polls: int
    live_price_cross_confirm_seconds: float
    live_price_down_cross_confirm_polls: int
    live_price_down_cross_confirm_seconds: float

    stoch_overbought: float

    stop_loss_percent: float
    take_profit_percent: float
    enable_box_range_hold_tech_sell: bool
    box_range_hold_lookback_bars: int
    box_range_hold_max_range_pct: float
    box_range_hold_max_bb_width_pct: float

    ma5_bb_down_cross_min_pnl: float
    ma5_bb_down_cross_immediate_pnl: float
    ma5_bb_down_cross_immediate_score: int
    aux_sell_min_pnl_score2: float
    aux_sell_min_pnl_score3: float
    aux_sell_min_pnl_score4: float

    bb_buy_score_threshold: int = 8


def _num(candle: pd.Series, key: str) -> float:
    value = candle.get(key)
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def update_timed_condition_state(
    state_by_code: dict[str, dict[str, object]],
    code: str,
    position_token: object,
    ts: object,
    condition: bool,
) -> float:
    state = state_by_code.get(code)
    if not condition:
        if state is not None and state.get("position_token") == position_token:
            state_by_code.pop(code, None)
        return 0.0

    if state is None or state.get("position_token") != position_token:
        state_by_code[code] = {"position_token": position_token, "start": ts}
        return 0.0

    start = state.get("start")
    if start is None:
        state["start"] = ts
        return 0.0

    try:
        return max((ts - start).total_seconds(), 0.0)
    except Exception:
        state["start"] = ts
        return 0.0


def compute_staged_tp1_target_pct(
    atr_pct: float,
    atr_tp_pct: float,
    *,
    staged_tp1_pct: float,
    tp1_atr_multiplier: float,
    cap_at_atr_tp: bool,
) -> tuple[float, float]:
    """1차 분할익절 목표(소수, 0.03=+3%)와 그 ATR 동적 성분(로그용)을 반환한다.

    기본 목표 = max(staged_tp1_pct, ATR% x tp1_atr_multiplier). cap_at_atr_tp=True면 ATR 익절선
    (atr_tp_pct = ATR% x ATR_TAKE_PROFIT_MULTIPLIER = TP_EXTENSION 트레일 무장선)을 상한으로 두어,
    TP1이 트레일 무장보다 늦어져 1차 분할 없이 트레일 전량 매도로 넘어가는 것을 막는다(ATR%>2.5%
    고변동 종목은 ATR 동적 목표가 이미 무장선보다 낮아 상한이 걸리지 않는다). ATR을 모르면(NaN)
    staged_tp1_pct 그대로. 반환: (tp1_target_pct, atr_based_pct) - ATR 미상이면 atr_based_pct는 NaN.
    """
    atr_based_pct = atr_pct * tp1_atr_multiplier if not pd.isna(atr_pct) else float("nan")
    target = max(staged_tp1_pct, atr_based_pct) if not pd.isna(atr_based_pct) else staged_tp1_pct
    if cap_at_atr_tp and not pd.isna(atr_tp_pct) and atr_tp_pct > 0:
        target = min(target, atr_tp_pct)
    return target, atr_based_pct


def update_recent_price_samples(
    samples_by_code: dict[str, list[tuple[object, float]]],
    code: str,
    ts: object,
    price: float,
    keep_seconds: float,
) -> list[tuple[object, float]]:
    """종목별 최근 (시각, 가격) 표본에 현재 틱을 추가하고 keep_seconds보다 오래된 표본은 버린다.

    표본은 프로세스 메모리에만 있다(재시작하면 비어서 시작 -> 표본이 쌓일 때까지 급등 판정은 False로
    보수적으로 동작). 정렬된(오래된 것 먼저) 표본 리스트를 반환한다.
    """
    samples = samples_by_code.get(code) or []
    if price is not None and price > 0:
        samples.append((ts, float(price)))
    try:
        samples = [(t, p) for (t, p) in samples if (ts - t).total_seconds() <= keep_seconds]
    except Exception:
        samples = [(ts, float(price))] if price is not None and price > 0 else []
    samples_by_code[code] = samples
    return samples


def detect_price_surge(
    samples: list[tuple[object, float]],
    now: object,
    price: float,
    atr_pct: float,
    bb_upper: float,
    volume: float,
    vol_ma: float,
    *,
    lookback_seconds: float,
    speed_min_pct: float,
    speed_atr_mult: float,
    volume_ratio_min: float,
    min_confirms: int,
) -> tuple[bool, str]:
    """급등 여부와 판정 근거 문자열(로그용)을 반환한다.

    급등 = (속도, 필수) 최근 lookback_seconds 내 최저가 대비 현재가 상승폭이
           max(speed_min_pct, ATR% x speed_atr_mult) 이상
         + (확인) BB 상단 돌파(현재가 > BB_UPPER), 봉 거래량 >= VOL_MA20 x volume_ratio_min 중
           min_confirms개 이상 충족.
    속도 기준을 ATR%로 정규화해, 평소 잘 흔들리는 종목의 정상 변동을 급등으로 오인하지 않게 한다.
    표본이 2개 미만이면(재시작 직후 등) 판정 불가로 False.
    """
    if price is None or price <= 0:
        return False, "NO_PRICE"
    window: list[float] = []
    for t, p in samples:
        try:
            if 0.0 <= (now - t).total_seconds() <= lookback_seconds and p > 0:
                window.append(float(p))
        except Exception:
            continue
    if len(window) < 2:
        return False, f"NO_SAMPLES(n={len(window)})"

    low = min(window)
    speed = price / low - 1.0
    speed_need = max(speed_min_pct, speed_atr_mult * atr_pct) if not pd.isna(atr_pct) else speed_min_pct
    speed_ok = speed >= speed_need
    bb_ok = (not pd.isna(bb_upper)) and bb_upper > 0 and price > bb_upper
    vol_ratio = (
        volume / vol_ma
        if (not pd.isna(volume) and not pd.isna(vol_ma) and vol_ma > 0)
        else float("nan")
    )
    vol_ok = (not pd.isna(vol_ratio)) and vol_ratio >= volume_ratio_min
    confirms = int(bb_ok) + int(vol_ok)
    is_surge = speed_ok and confirms >= min_confirms
    detail = (
        f"speed={speed*100:.2f}%(need>={speed_need*100:.2f}%,low={low:,.0f},{lookback_seconds:.0f}s,n={len(window)}) "
        f"bb_break={bb_ok}(bb_up={bb_upper:,.0f}) "
        f"vol_ratio={vol_ratio:.2f}(need>={volume_ratio_min:.2f}) confirms={confirms}/{min_confirms}"
    )
    return is_surge, detail


def next_surge_ladder_action(
    price: float,
    base_price: float,
    remaining_qty: int,
    entry_qty: int,
    tp2_done: bool,
    tp3_done: bool,
    *,
    tp2_pct: float,
    tp3_pct: float,
    tp2_ratio: float,
) -> tuple[str, int, float] | None:
    """급등 사다리 익절의 다음 실행 단계를 결정한다. 실행할 단계가 없으면 None.

    base_price는 1차 익절 체결가. 반환: ("TP2"|"TP3", 매도수량, 그 단계 목표가) - 한 번에 한 단계만.
    - TP3: price >= base x (1+tp3_pct) 이면 잔량 전량. TP2가 아직이어도 TP2+TP3를 합쳐 이 틱에 잔량
      전량을 청산한다(급등이 한 틱에 +4%를 넘어도, 부분 매도 뒤 쿨다운으로 TP3가 밀려 놓치지 않도록).
      호출측은 TP3 실행 시 tp2_done/tp3_done을 모두 True로 둔다.
    - TP2: price >= base x (1+tp2_pct) 이면 진입수량 x tp2_ratio(최소 1주)를 매도하되 TP3용 1주는
      남긴다. 잔량이 1주뿐이라 나눌 수 없으면 TP2는 건너뛰고 TP3(또는 트레일)가 처리한다.
    """
    if base_price <= 0 or remaining_qty <= 0 or (tp2_done and tp3_done):
        return None
    tp2_price = base_price * (1.0 + tp2_pct)
    tp3_price = base_price * (1.0 + tp3_pct)
    if (not tp3_done) and price >= tp3_price:
        return "TP3", int(remaining_qty), tp3_price
    if (not tp2_done) and price >= tp2_price:
        base_qty = entry_qty if entry_qty > 0 else remaining_qty
        qty = min(max(1, int(round(base_qty * tp2_ratio))), int(remaining_qty) - 1)
        if qty >= 1:
            return "TP2", qty, tp2_price
    return None


def _compute_bb_slope_pct(frame: pd.DataFrame, lookback: int = BB_SLOPE_LOOKBACK_BARS) -> float:
    if "BB_MIDDLE" not in frame.columns or len(frame) < 2:
        return float("nan")
    n = min(lookback, len(frame) - 1)
    bb_now = _num(frame.iloc[-1], "BB_MIDDLE")
    bb_ago = _num(frame.iloc[-(n + 1)], "BB_MIDDLE")
    if pd.isna(bb_now) or pd.isna(bb_ago) or bb_ago <= 0:
        return float("nan")
    return (bb_now - bb_ago) / bb_ago * 100.0



def _bb_middle_is_downtrend(frame: pd.DataFrame, n: int = BB_MID_DOWNTREND_BARS) -> bool:
    """최근 n봉(=n*3분) 동안 BB 중간선이 연속으로 하락 중이면 True."""
    if "BB_MIDDLE" not in frame.columns or len(frame) < n + 1:
        return False
    vals = [_num(frame.iloc[-(i + 1)], "BB_MIDDLE") for i in range(n)]
    if any(pd.isna(v) for v in vals):
        return False
    return all(vals[i] < vals[i + 1] for i in range(n - 1))

@dataclass(frozen=True)
class BuyEvalContext:
    """매수 조건 평가에 필요한 공유 상태를 한 곳에 모은 컨텍스트.

    check_buy_condition()/run_buy_condition_pipeline_comment() 진입 시 한 번
    만들어지고, BUY_GATE_CONDITIONS를 순서대로 통과하며 각 게이트가 계산한
    파생값(cross_eval 등)을 dataclasses.replace()로 다음 게이트에 전달한다 -
    기존의 "뒤 조건이 앞 조건 계산값에 의존"하는 흐름을 그대로 유지하되 암묵적인
    지역변수 전달 대신 명시적 타입으로 드러낸다.
    """

    frame: pd.DataFrame
    now: pd.Timestamp | None
    live_price: float | None
    cross_info: dict[str, object] | None
    config: "R76StrategyConfig"
    cur: pd.Series
    prev: pd.Series
    cur_bb: float
    cur_bb_upper: float | None
    prev_bb: float
    bb_slope_pct: float

    # 게이트가 진행되며 채워지는 파생값
    cross_eval: dict[str, object] | None = None
    candle_gain_pct: float | None = None
    bb_mid_gap_pct: float | None = None


@dataclass(frozen=True)
class BuyGateCondition:
    """매수 필수조건(순차 통과형) 1개. BUY_GATE_CONDITIONS 리스트 순서대로
    평가되며 하나라도 실패하면 즉시 매수 거부(short-circuit) - 조건을
    추가/삭제/순서변경하려면 이 객체들이 모인 리스트만 건드리면 된다."""

    name: str
    description: str
    related_defines: tuple[str, ...]
    eval_fn: Callable[["BuyEvalContext"], tuple[bool, "BuyEvalContext", str | None]]


@dataclass(frozen=True)
class BuyScoreRule:
    """매수 가점항목(독립 합산형) 1개. BUY_SCORE_RULES 각 항목의 eval_fn이
    돌려주는 점수를 모두 더해 config.bb_buy_score_threshold와 비교한다."""

    name: str
    max_score: int
    description: str
    related_defines: tuple[str, ...]
    eval_fn: Callable[["BuyEvalContext"], int]


# ============================================================================
# BUY CONDITIONS - 매수 조건 모음
# 조건을 추가/삭제/순서변경하려면 이 섹션의 BUY_GATE_CONDITIONS(필수 게이트,
# 순차 통과형) / BUY_SCORE_RULES(가점항목, 독립 합산형) 리스트만 수정하면 된다.
# 각 조건이 쓰는 r001_define_config.py 상수는 related_defines에 모아둬서
# 어느 define이 어느 조건 소속인지 한눈에 확인 가능.
# ============================================================================

# ---- 가점항목 (BUY_SCORE_RULES, 독립 합산형) -------------------------------


def _score_rsi_band(ctx: BuyEvalContext) -> int:
    rsi_c = _num(ctx.cur, "RSI")
    if pd.isna(rsi_c):
        return 0
    if 50.0 <= rsi_c <= 65.0:
        return 2
    if 45.0 <= rsi_c < 50.0 or 65.0 < rsi_c <= 70.0:
        return 1
    return 0


def _score_ema_trend_align(ctx: BuyEvalContext) -> int:
    ema20_c = _num(ctx.cur, "EMA_20")
    ema60_c = _num(ctx.cur, "EMA_60")
    if any(pd.isna(v) for v in (ema20_c, ema60_c)):
        return 0
    return EMA_TREND_ALIGN_SCORE if ema20_c > ema60_c else 0


def _score_volume_ratio(ctx: BuyEvalContext) -> int:
    vol = _num(ctx.cur, "volume")
    vol_ma = _num(ctx.cur, "VOL_MA20")
    if any(pd.isna(v) for v in (vol, vol_ma)) or vol_ma <= 0:
        return 0
    vol_ratio = vol / vol_ma
    if vol_ratio >= 2.0:
        return 3
    if vol_ratio >= 1.5:
        return 2
    if vol_ratio >= 0.7:
        return 1
    return 0


def _score_adx_strength(ctx: BuyEvalContext) -> int:
    # [2026-09-13][R77-D] ADX 크기만으로는 추세 방향을 확인 못한다(하락추세에서도 ADX는
    # 높을 수 있음) - +DI가 -DI보다 우세할 때만 점수를 주도록 방향성 확인을 추가한다.
    # DI가 스코어에 쓰인 적이 2026-06-28에 "이중반영"으로 한 번 제거됐으나, 그건 당시
    # 스코어 레벨에서의 중복이었고 현재 DI 방향 확인은 bb_mid_cross_up 게이트의
    # uptrend_continuation 예외 경로에만 국한돼 있어 live_cross_up/close_cross 경로는
    # 방향성 확인이 전혀 없었다 - 그 공백을 메우는 재도입(2~3점 구조, 기존 1점 티어 제거).
    adx_c = _num(ctx.cur, "ADX")
    di_plus = _num(ctx.cur, "DI_PLUS")
    di_minus = _num(ctx.cur, "DI_MINUS")
    if any(pd.isna(v) for v in (adx_c, di_plus, di_minus)):
        return 0
    if di_plus <= di_minus or adx_c < ADX_DI_SCORE_MIN_ADX:
        return 0
    di_spread = di_plus - di_minus
    if adx_c >= ADX_DI_SCORE_STRONG_ADX and di_spread >= ADX_DI_SCORE_STRONG_SPREAD:
        return 3
    return 2


def _score_vwap_position(ctx: BuyEvalContext) -> int:
    # DI이중반영 제거 후 대체: 현재가 > VWAP → 기관 평균 매수가 상회
    vwap_v = _num(ctx.cur, "VWAP")
    close_v = _num(ctx.cur, "close")
    if any(pd.isna(v) for v in (vwap_v, close_v)) or vwap_v <= 0:
        return 0
    if close_v > vwap_v * 1.002:
        return 2
    if close_v > vwap_v:
        return 1
    return 0


def _score_bb_width_expansion(ctx: BuyEvalContext) -> int:
    # 스퀴즈 해소 → 추세 발생 초기 신호
    bb_upper_c = _num(ctx.cur, "BB_UPPER")
    bb_lower_c = _num(ctx.cur, "BB_LOWER")
    bb_upper_p = _num(ctx.prev, "BB_UPPER")
    bb_lower_p = _num(ctx.prev, "BB_LOWER")
    if any(pd.isna(v) for v in (bb_upper_c, bb_lower_c, bb_upper_p, bb_lower_p)):
        return 0
    return 1 if (bb_upper_c - bb_lower_c) > (bb_upper_p - bb_lower_p) else 0


def _score_ma5_short_term_up(ctx: BuyEvalContext) -> int:
    # MA5[t] > MA5[t-1] → 단기 추세 유지
    ma5_c = _num(ctx.cur, "MA_5")
    ma5_p = _num(ctx.prev, "MA_5")
    if any(pd.isna(v) for v in (ma5_c, ma5_p)) or ma5_p <= 0:
        return 0
    return 1 if ma5_c > ma5_p else 0


def _score_macd_golden_cross(ctx: BuyEvalContext) -> int:
    macd_c = _num(ctx.cur, "MACD")
    msig_c = _num(ctx.cur, "MACD_SIGNAL")
    if any(pd.isna(v) for v in (macd_c, msig_c)):
        return 0
    return 2 if macd_c > msig_c else 0


def _score_bb_mid_slope_strength(ctx: BuyEvalContext) -> int:
    bb_slope_pct = ctx.bb_slope_pct
    if pd.isna(bb_slope_pct):
        return 0
    if bb_slope_pct >= 1.5:
        return 3
    if bb_slope_pct >= 1.0:
        return 2
    if bb_slope_pct >= 0.5:
        return 1
    return 0


def _score_obv_breakout(ctx: BuyEvalContext) -> int:
    # [2026-09-07] OBV가 OBV_MA를 최근 OBV_BREAKOUT_LOOKBACK_BARS 내에서 상향 돌파했는지
    # 확인한다. 가격 돌파와 거래량 누적방향(OBV)이 함께 확인되면 가짜 돌파(fakeout)일
    # 가능성이 낮다는 거래량 분석의 일반적인 관행을 반영한 가점 - REQUIRE_OBV_SIGNAL_CROSS로
    # on/off 가능. 이전에는 OBV_MA/OBV가 계산만 되고 매수 판정 어디에도 쓰이지 않았다.
    if not REQUIRE_OBV_SIGNAL_CROSS:
        return 0
    frame = ctx.frame
    if frame is None or len(frame) < 2:
        return 0
    obv = frame["OBV"] if "OBV" in frame.columns else None
    obv_ma = frame["OBV_MA"] if "OBV_MA" in frame.columns else None
    if obv is None or obv_ma is None:
        return 0
    lookback = min(OBV_BREAKOUT_LOOKBACK_BARS, len(frame) - 1)
    obv_now = _num(ctx.cur, "OBV")
    obv_ma_now = _num(ctx.cur, "OBV_MA")
    if pd.isna(obv_now) or pd.isna(obv_ma_now) or obv_now <= obv_ma_now:
        return 0
    for back in range(1, lookback + 1):
        prior_obv = obv.iloc[-1 - back]
        prior_obv_ma = obv_ma.iloc[-1 - back]
        if pd.isna(prior_obv) or pd.isna(prior_obv_ma):
            continue
        if prior_obv <= prior_obv_ma:
            return OBV_CONFIRM_SCORE  # lookback 내에서 OBV가 OBV_MA 아래였다가 지금 위로 돌파
    return 0


def _score_pre_cross_accumulation(ctx: BuyEvalContext) -> int:
    # [2026-09-13][R77-A] 기존 _gate_pre_cross_accumulation_bar(필수 게이트)를 그대로
    # 가점으로 전환 - 판정 로직(룩백/거래량비율)은 동일, 실패해도 매수를 막지 않고
    # 점수만 안 준다.
    if not ENABLE_PRE_CROSS_ACCUM_BAR_CHECK:
        return 0
    frame = ctx.frame
    for _ofs in range(2, 2 + PRE_CROSS_ACCUM_LOOKBACK_BARS):
        if _ofs + 1 > len(frame):
            break
        _cand = frame.iloc[-_ofs]
        _cand_open = _num(_cand, "open")
        _cand_close = _num(_cand, "close")
        _cand_low = _num(_cand, "low")
        _cand_high = _num(_cand, "high")
        _cand_bb = _num(_cand, "BB_MIDDLE")
        _cand_vol = _num(_cand, "volume")
        _cand_vol_ma = _num(_cand, "VOL_MA20")
        if any(pd.isna(v) for v in (
            _cand_open, _cand_close, _cand_low, _cand_high, _cand_bb, _cand_vol, _cand_vol_ma,
        )):
            continue
        _contains_bb_mid = _cand_low <= _cand_bb <= _cand_high
        _is_bullish = _cand_close > _cand_open
        _vol_rising = _cand_vol >= _cand_vol_ma * PRE_CROSS_ACCUM_VOL_RATIO_MIN
        if _contains_bb_mid and _is_bullish and _vol_rising:
            return PRE_CROSS_ACCUM_SCORE
    return 0


def _score_bb_upper_room_atr(ctx: BuyEvalContext) -> int:
    # [2026-09-13][R77-G] BB 상단까지 남은 여유폭을 종목 변동성(ATR)으로 정규화한 가점.
    # 기존 bb_upper_gap_min 게이트(고정 0.25%)는 필수조건으로 그대로 유지 - 이건 그
    # 위에 얹는 보너스로, 저변동/고변동 종목에 동일한 고정 %가 다른 의미를 갖는 문제를
    # 완화한다.
    bb_upper = _num(ctx.cur, "BB_UPPER")
    close_v = _num(ctx.cur, "close")
    atr = _num(ctx.cur, "ATR")
    if any(pd.isna(v) for v in (bb_upper, close_v, atr)) or close_v <= 0 or atr <= 0:
        return 0
    room_atr = (bb_upper - close_v) / atr
    if room_atr >= BB_UPPER_ROOM_ATR_TIER2:
        return 2
    if room_atr >= BB_UPPER_ROOM_ATR_TIER1:
        return 1
    return 0


BUY_SCORE_RULES: list[BuyScoreRule] = [
    BuyScoreRule("rsi_band", 2, "RSI 구간 점수 (50~65=2점, 45~50/65~70=1점)", (), _score_rsi_band),
    BuyScoreRule("ema_trend_align", EMA_TREND_ALIGN_SCORE, "장기 추세 정합성: EMA20 > EMA60(3분봉) → 상위 추세 우상향", ("EMA_TREND_ALIGN_SCORE",), _score_ema_trend_align),
    BuyScoreRule("volume_ratio", 3, "거래량 비율 점수 (VOL_MA20 대비 >=2.0=3점, >=1.5=2점, >=0.7=1점)", (), _score_volume_ratio),
    BuyScoreRule("adx_strength", 3, "[R77-D] ADX+DI 방향성 점수 (+DI>-DI 필수: ADX>=30&스프레드>=15=3점, ADX>=25=2점, 그 외 0점)", ("ADX_DI_SCORE_MIN_ADX", "ADX_DI_SCORE_STRONG_ADX", "ADX_DI_SCORE_STRONG_SPREAD"), _score_adx_strength),
    BuyScoreRule("vwap_position", 2, "VWAP 대비 현재가 위치 (VWAP*1.002 초과=2점, VWAP 초과=1점)", (), _score_vwap_position),
    BuyScoreRule("bb_width_expansion", 1, "BB 폭 확장: 스퀴즈 해소 → 추세 발생 초기 신호", (), _score_bb_width_expansion),
    BuyScoreRule("ma5_short_term_up", 1, "MA5 단기 상승: MA5[t] > MA5[t-1]", (), _score_ma5_short_term_up),
    BuyScoreRule("macd_golden_cross", 2, "MACD 골든크로스: MACD > MACD_SIGNAL", (), _score_macd_golden_cross),
    BuyScoreRule("bb_mid_slope_strength", 3, "BB 중앙선 기울기 강도 점수 (>=1.5%=3점, >=1.0%=2점, >=0.5%=1점)", (), _score_bb_mid_slope_strength),
    BuyScoreRule("obv_breakout", OBV_CONFIRM_SCORE, "OBV가 OBV_MA를 lookback 내 상향 돌파 (거래량 방향 확인, fakeout 방지)", ("REQUIRE_OBV_SIGNAL_CROSS", "OBV_BREAKOUT_LOOKBACK_BARS"), _score_obv_breakout),
    BuyScoreRule("pre_cross_accumulation", PRE_CROSS_ACCUM_SCORE, "[R77-A] 직전 매집봉 존재 (기존 필수게이트를 가점으로 전환)", ("ENABLE_PRE_CROSS_ACCUM_BAR_CHECK", "PRE_CROSS_ACCUM_LOOKBACK_BARS", "PRE_CROSS_ACCUM_VOL_RATIO_MIN"), _score_pre_cross_accumulation),
    BuyScoreRule("bb_upper_room_atr", 2, "[R77-G] BB상단 여유폭/ATR 정규화 점수 (>=1.5=2점, >=0.8=1점)", ("BB_UPPER_ROOM_ATR_TIER1", "BB_UPPER_ROOM_ATR_TIER2"), _score_bb_upper_room_atr),
]


def _buy_support_score(
    cur: pd.Series,
    prev: pd.Series,
    frame: pd.DataFrame,
    config: R76StrategyConfig,
) -> int:
    """가점 조건 합산 (BUY_SCORE_RULES 참조). 실제 최대 합산 점수는
    sum(rule.max_score for rule in BUY_SCORE_RULES) - 항목이 추가될 때마다
    바뀌므로 이 함수 docstring에는 숫자를 하드코딩하지 않는다."""
    ctx = BuyEvalContext(
        frame=frame, now=None, live_price=None, cross_info=None, config=config,
        cur=cur, prev=prev, cur_bb=_num(cur, "BB_MIDDLE"), cur_bb_upper=None,
        prev_bb=_num(prev, "BB_MIDDLE"), bb_slope_pct=_compute_bb_slope_pct(frame),
    )
    return sum(rule.eval_fn(ctx) for rule in BUY_SCORE_RULES)


def _sell_support_score(cur: pd.Series, prev: pd.Series, config: R76StrategyConfig) -> int:
    score = 0

    k_c = _num(cur, "STOCH_K")
    d_c = _num(cur, "STOCH_D")
    k_p = _num(prev, "STOCH_K")
    d_p = _num(prev, "STOCH_D")
    if not any(pd.isna(v) for v in (k_c, d_c, k_p, d_p)):
        if k_c < d_c:
            score += 1

    # RSI 하락 방향 (빠른 모멘텀 약화 감지): RSI[t] < RSI[t-1] — RSI_SIGNAL 크로스보다 빠름
    rsi_c = _num(cur, "RSI")
    rsi_p = _num(prev, "RSI")
    if not any(pd.isna(v) for v in (rsi_c, rsi_p)):
        if rsi_c < rsi_p:
            score += 1

    # VWAP 이탈 (기관 매도 압력): 현재가 < VWAP — Williams %R/%D 제거 후 대체
    vwap_v = _num(cur, "VWAP")
    close_v = _num(cur, "close")
    if not any(pd.isna(v) for v in (vwap_v, close_v)) and vwap_v > 0:
        if close_v < vwap_v:
            score += 1

    macd_c = _num(cur, "MACD")
    msig_c = _num(cur, "MACD_SIGNAL")
    macd_p = _num(prev, "MACD")
    msig_p = _num(prev, "MACD_SIGNAL")
    if not any(pd.isna(v) for v in (macd_c, msig_c, macd_p, msig_p)):
        if macd_p >= msig_p and macd_c < msig_c:
            score += 1

    obv_c = _num(cur, "OBV")
    obv_ma_c = _num(cur, "OBV_MA")
    obv_p = _num(prev, "OBV")
    if not any(pd.isna(v) for v in (obv_c, obv_ma_c, obv_p)):
        if obv_c < obv_ma_c and obv_c < obv_p:
            score += 1

    return score


def _is_box_range_hold_zone(frame: pd.DataFrame, config: R76StrategyConfig) -> tuple[bool, str]:
    if len(frame) < config.box_range_hold_lookback_bars:
        return False, "INSUFFICIENT_BOX_BARS"

    recent = frame.tail(config.box_range_hold_lookback_bars)
    high_v = pd.to_numeric(recent["high"], errors="coerce").max()
    low_v = pd.to_numeric(recent["low"], errors="coerce").min()
    close_v = _num(recent.iloc[-1], "close")
    bb_up = _num(recent.iloc[-1], "BB_UPPER")
    bb_low = _num(recent.iloc[-1], "BB_LOWER")

    if any(pd.isna(v) for v in (high_v, low_v, close_v, bb_up, bb_low)) or close_v <= 0:
        return False, "BOX_DATA_NAN"

    range_pct = (float(high_v) - float(low_v)) / float(close_v)
    bb_width_pct = (float(bb_up) - float(bb_low)) / float(close_v)
    is_box = range_pct <= config.box_range_hold_max_range_pct and bb_width_pct <= config.box_range_hold_max_bb_width_pct

    return is_box, f"RANGE_{range_pct*100:.2f}%_BBW_{bb_width_pct*100:.2f}%"


def update_live_price_cross_state(
    cross_state: dict[str, dict],
    code: str,
    now: pd.Timestamp,
    live_price: float,
    bb_middle: float,
    config: R76StrategyConfig,
) -> dict[str, object]:
    relation = "on"
    upper_trigger = bb_middle * (1.0 + config.live_price_bb_buffer_pct)
    lower_trigger = bb_middle * (1.0 - config.live_price_bb_buffer_pct)

    if live_price >= upper_trigger:
        relation = "above"
    elif live_price <= lower_trigger:
        relation = "below"

    tracker = cross_state.get(code)
    if tracker is None:
        tracker = {
            "confirmed_relation": relation if relation in {"above", "below"} else None,
            "pending": None,
        }
        cross_state[code] = tracker
        return {
            "relation": relation,
            "confirmed_relation": tracker["confirmed_relation"],
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    confirmed_relation = tracker.get("confirmed_relation")
    pending = tracker.get("pending")

    if relation not in {"above", "below"}:
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    if relation == confirmed_relation:
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    if pending is None or pending.get("side") != relation:
        tracker["pending"] = {"side": relation, "started_at": now, "count": 1}
        return {
            "relation": relation,
            "confirmed_relation": confirmed_relation,
            "pending_side": relation,
            "pending_count": 1,
            "pending_seconds": 0.0,
            "signal": None,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    pending["count"] = int(pending.get("count", 0)) + 1
    pending_seconds = max(0.0, (now - pending["started_at"]).total_seconds())

    if relation == "below":
        req_polls = config.live_price_down_cross_confirm_polls
        req_seconds = config.live_price_down_cross_confirm_seconds
    else:
        req_polls = config.live_price_cross_confirm_polls
        req_seconds = config.live_price_cross_confirm_seconds

    if pending["count"] >= req_polls and pending_seconds >= req_seconds:
        tracker["confirmed_relation"] = relation
        tracker["pending"] = None
        return {
            "relation": relation,
            "confirmed_relation": relation,
            "pending_side": None,
            "pending_count": 0,
            "pending_seconds": pending_seconds,
            "signal": "cross_up" if relation == "above" else "cross_down",
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
        }

    return {
        "relation": relation,
        "confirmed_relation": confirmed_relation,
        "pending_side": relation,
        "pending_count": int(pending["count"]),
        "pending_seconds": pending_seconds,
        "signal": None,
        "upper_trigger": upper_trigger,
        "lower_trigger": lower_trigger,
    }


def _evaluate_bb_mid_cross(
    frame: pd.DataFrame,
    cur: pd.Series,
    prev: pd.Series,
    cur_bb: float,
    prev_bb: float,
    live_price: float,
    bb_slope_pct: float,
    cross_info: dict[str, object],
) -> dict[str, object]:
    """BB 중앙선 상향 돌파 판정 (live cross signal OR close 기준 크로스, 5봉 룩백,
    우상향 추세 지속 포함). run_buy_condition_pipeline_comment의 실제 매수 판정과
    r006의 진단 로그 스냅샷(_buy_condition_snapshot)이 모두 이 함수 하나를 공유해서
    쓴다 - 둘이 따로 계산하면 로그 표시가 실제 판정과 어긋날 수 있음(2026-08-24 확인:
    로그가 즉시 2봉 close_cross만 보여줘 실제로는 통과하는 케이스를 False로 오표시).
    """
    live_cross_up = cross_info.get("signal") == "cross_up"
    prev_close = _num(prev, "close")
    cur_close = _num(cur, "close")
    close_cross = (
        not any(pd.isna(v) for v in (prev_close, cur_close, prev_bb, cur_bb))
        and prev_close <= prev_bb
        and cur_close > cur_bb
    )
    # N봉 이전 크로스 체크 (5봉으로 확장): 전환봉 이후 현재까지 BB 위 연속 유지 시 크로스 인정
    if not close_cross:
        for _lb in range(3, min(6, len(frame)) + 1):
            _bar_n_close = _num(frame.iloc[-_lb], "close")
            _bar_n_bb = _num(frame.iloc[-_lb], "BB_MIDDLE")
            if any(pd.isna(v) for v in (_bar_n_close, _bar_n_bb)) or _bar_n_close > _bar_n_bb:
                continue  # 이 봉도 BB 위이면 더 이전 탐색 or NaN
            # 전환봉 이후 현재까지 모든 봉이 연속으로 BB 위인지 확인
            _all_above = all(
                not any(pd.isna(v) for v in (_num(frame.iloc[-_k], "close"), _num(frame.iloc[-_k], "BB_MIDDLE")))
                and _num(frame.iloc[-_k], "close") > _num(frame.iloc[-_k], "BB_MIDDLE")
                for _k in range(1, _lb)
            )
            if _all_above:
                close_cross = True
                break

    # 우상향 추세 지속 진입: 크로스 이벤트 없이도 강한 추세 + BB 위 지속이면 매수 허용
    # 크로스가 5봉 이전에 발생했지만 추세가 계속 이어지는 구간 포착
    uptrend_continuation = False
    if not live_cross_up and not close_cross:
        _adx = _num(cur, "ADX")
        _di_plus = _num(cur, "DI_PLUS")
        _di_minus = _num(cur, "DI_MINUS")
        _ma5_cur = _num(cur, "MA_5")
        _ma5_prev = _num(prev, "MA_5")
        # 최근 최대 5봉 중 종가가 BB 위에 있는 봉 수 카운트
        _n_above = 0
        for _i in range(1, min(6, len(frame)) + 1):
            _bc = _num(frame.iloc[-_i], "close")
            _bbb = _num(frame.iloc[-_i], "BB_MIDDLE")
            if not any(pd.isna(v) for v in (_bc, _bbb)) and _bc > _bbb:
                _n_above += 1
        if (
            not any(pd.isna(v) for v in (cur_close, cur_bb, prev_close, _adx, _di_plus, _di_minus))
            and live_price > cur_bb        # 현재가 BB 중간선 위
            and cur_close > cur_bb         # 현재봉 종가 BB 위
            and cur_close > prev_close     # 현재봉 종가 > 전봉 종가 (상승 모멘텀)
            and bb_slope_pct > UPTREND_CONT_SLOPE_MIN_PCT  # BB 중간선 상승 추세 (0 근방 후행지표 노이즈 허용)
            and _adx >= 30.0               # ADX 30 이상 (뚜렷한 추세)
            and _di_plus > _di_minus       # +DI > -DI (상승 방향성 우세)
            and _n_above >= 3              # 최근 5봉 중 3봉 이상 BB 위 유지
            and not any(pd.isna(v) for v in (_ma5_cur, _ma5_prev))
            and _ma5_cur > _ma5_prev       # MA5 단기 상승 중 (이미 꺾인 추세 추격 차단)
        ):
            uptrend_continuation = True

    return {
        "live_cross_up": live_cross_up,
        "close_cross": close_cross,
        "uptrend_continuation": uptrend_continuation,
        "passed": live_cross_up or close_cross or uptrend_continuation,
    }


def check_buy_condition(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
) -> tuple[bool, str]:
    return run_buy_condition_pipeline_comment(
        frame=frame,
        now=now,
        live_price=live_price,
        cross_info=cross_info,
        config=config,
    )


# ---- 필수 게이트 (BUY_GATE_CONDITIONS, 순차 통과형) --------------------------


def _gate_bb_slope_rising(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    bb_slope_pct = ctx.bb_slope_pct
    if pd.isna(bb_slope_pct) or bb_slope_pct <= BB_SLOPE_MIN_PCT:
        slope_str = f"{bb_slope_pct:.3f}" if not pd.isna(bb_slope_pct) else "nan"
        return False, ctx, f"BB_SLOPE_NOT_RISING_{slope_str}%"
    return True, ctx, None


def _gate_bb_mid_downtrend_block(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # BB 중간선 최근 12분(4봉) 연속 우하향 시 매수 차단. 단, 가격이 이미 BB_MID를
    # 상향 돌파한 경우(후행 지표 아티팩트)는 차단 해제.
    frame, cur_bb, live_price = ctx.frame, ctx.cur_bb, ctx.live_price
    if _bb_middle_is_downtrend(frame):
        _price_above_bb_pct = (live_price - cur_bb) / cur_bb * 100.0 if cur_bb > 0 else 0.0
        if _price_above_bb_pct <= 0.0:
            bb_vals = [_num(frame.iloc[-(i + 1)], "BB_MIDDLE") for i in range(4)] if len(frame) >= 4 else []
            vals_str = " > ".join(f"{v:.1f}" for v in reversed(bb_vals))
            return False, ctx, f"BB_MID_DOWNTREND_4BARS_{vals_str}"
    return True, ctx, None


def _gate_bb_mid_cross_up(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # BB 중앙선 상향 돌파 (live cross signal OR close 기준 크로스, 5봉 룩백, 우상향
    # 추세 지속 포함) - _evaluate_bb_mid_cross()에 위임. 이 판정 로직은 r003의
    # 진단용 로그 스냅샷(_buy_condition_snapshot)에서도 그대로 재사용되므로, 여기서
    # 바꾸면 로그 표시도 항상 실제 판정과 일치한다(2026-08-24: 로그가 즉시 2봉
    # close_cross만 보여줘 실제로는 5봉 룩백/우상향 지속으로 통과하는 케이스를
    # False로 오표시하던 문제를 이 공유화로 근본 해결 - r002 Update log 참조).
    cross_eval = _evaluate_bb_mid_cross(
        ctx.frame, ctx.cur, ctx.prev, ctx.cur_bb, ctx.prev_bb, ctx.live_price,
        ctx.bb_slope_pct, ctx.cross_info,
    )
    ctx = dataclasses.replace(ctx, cross_eval=cross_eval)
    if not cross_eval["passed"]:
        return False, ctx, "NO_BB_MID_CROSS_UP"
    return True, ctx, None


def _gate_candle_bullish_and_chase_guard(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # 현재 진행중인 3분봉 양봉(+CANDLE_GAIN_MIN_PCT% 이상) 확인 + 추격매수 방지 2건:
    # (1) 현재봉 과도 상승 차단(급등봉 추격), (2) BB 중간선 대비 현재가 갭 과도 차단
    # (스파이크 추격). uptrend_continuation 경로(크로스 이벤트 없이 추세 지속만으로
    # 진입)는 BB_MID가 후행지표라 조건이 확정되는 시점엔 이미 가격이
    # BB_MID_CHASE_MAX_GAP_PCT보다 멀리 가있는 경우가 흔함(017670 SK텔레콤
    # 2026-08-24 13:12 반등 사례: uptrend_continuation이 13:17에야 확정됐는데 그땐
    # 이미 갭이 0.35% 문턱을 넘어 있어 재진입 자체가 매번 CHASE_BUY_BB_GAP로 막힘).
    # 이 경로에서만 갭 상한을 넓히는 대신, RSI 과열 여부(모멘텀)로 "아직 쫓아가도
    # 되는 건강한 지속 구간"인지 추가 검증한다 - MA5 상승은 uptrend_continuation
    # 자체 조건에 이미 포함됨.
    cur, live_price, cur_bb = ctx.cur, ctx.live_price, ctx.cur_bb
    cross_eval = ctx.cross_eval
    live_cross_up = cross_eval["live_cross_up"]
    close_cross = cross_eval["close_cross"]
    _uptrend_continuation = cross_eval["uptrend_continuation"]

    cur_open = _num(cur, "open")
    if pd.isna(cur_open) or cur_open <= 0:
        return False, ctx, "CANDLE_OPEN_MISSING"
    candle_gain_pct = (live_price - cur_open) / cur_open * 100.0
    if candle_gain_pct < CANDLE_GAIN_MIN_PCT:
        return False, ctx, f"CANDLE_NOT_BULLISH_{candle_gain_pct:.2f}%_LT_{CANDLE_GAIN_MIN_PCT:.1f}%"
    if candle_gain_pct > CANDLE_GAIN_MAX_PCT:
        return False, ctx, f"CHASE_BUY_INTRABAR_{candle_gain_pct:.2f}%_GT_{CANDLE_GAIN_MAX_PCT:.1f}%"

    _bb_mid_gap_pct = (live_price - cur_bb) / cur_bb * 100.0 if cur_bb > 0 else 0.0
    ctx = dataclasses.replace(ctx, candle_gain_pct=candle_gain_pct, bb_mid_gap_pct=_bb_mid_gap_pct)

    if _uptrend_continuation and not live_cross_up and not close_cross:
        _rsi_for_chase = _num(cur, "RSI")
        _uptrend_chase_ok = (
            not pd.isna(_rsi_for_chase)
            and _rsi_for_chase < UPTREND_CONT_CHASE_RSI_MAX
            and _bb_mid_gap_pct <= UPTREND_CONT_CHASE_MAX_GAP_PCT
        )
        if not _uptrend_chase_ok:
            _rsi_str = f"{_rsi_for_chase:.1f}" if not pd.isna(_rsi_for_chase) else "nan"
            return False, ctx, (
                f"CHASE_BUY_UPTREND_CONT_RSI_{_rsi_str}_GAP_{_bb_mid_gap_pct:.2f}%_"
                f"LT_RSI{UPTREND_CONT_CHASE_RSI_MAX:.0f}_GAP{UPTREND_CONT_CHASE_MAX_GAP_PCT:.1f}%"
            )
    elif _bb_mid_gap_pct > BB_MID_CHASE_MAX_GAP_PCT:
        return False, ctx, f"CHASE_BUY_BB_GAP_{_bb_mid_gap_pct:.2f}%_GT_{BB_MID_CHASE_MAX_GAP_PCT:.1f}%"

    return True, ctx, None


def _gate_bb_upper_gap_min(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    live_price, cur_bb_upper = ctx.live_price, ctx.cur_bb_upper
    if live_price <= 0:
        return False, ctx, "LIVE_PRICE_INVALID"
    bb_upper_gap_pct = (cur_bb_upper - live_price) / live_price * 100.0
    if bb_upper_gap_pct < BB_UPPER_GAP_MIN_PCT:
        return False, ctx, f"BB_UPPER_GAP_TOO_SMALL_{bb_upper_gap_pct:.2f}%_LT_{BB_UPPER_GAP_MIN_PCT:.1f}%"
    return True, ctx, None


def _gate_stochastic_buy_signal(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # 스토캐스틱 패스트 매수신호 (%K가 %D를 상향 돌파, 또는 %K>%D이면서 진짜
    # 과열(stoch_overbought) 아닌 구간). 주의: %K는 10봉 오실레이터라 BB중앙선
    # 돌파 순간엔 이미 50을 넘어서는 게 정상적인 움직임이라 STOCH_BUY_MAX(50)를
    # 상한으로 쓰면 정상적인 돌파 대부분을 걸러내 버린다. 상한은
    # config.stoch_overbought(진짜 과열 기준, 기본 96)로 완화한다.
    #
    # [2026-09-07] 기존에 별도 게이트였던 williams_r_buy_signal을 여기로 통합했다.
    # WILLIAMS_R_PERIOD == STOCH_K_PERIOD(둘 다 10)로 동일 window의 동일 high/low를
    # 쓰기 때문에 수학적으로 WILLIAMS_R = STOCH_K - 100이 항상 성립한다(검증 완료) -
    # "상승 중 & 바닥권탈출~과열직전" 조건은 STOCH_K가 직전봉 대비 상승 중이고
    # WILLIAMS_BUY_FLOOR/WILLIAMS_OVERBOUGHT_CEIL을 STOCH_K 스케일로 환산한 밴드
    # 안에 있는지로 완전히 동일하게 표현된다. 서로 다른 두 지표처럼 보였지만 실제로는
    # 같은 원천 데이터의 재포장이라 게이트 2개를 유지해도 필터링 다양성에 기여하는 바가
    # 없었다 - 통합하고 남은 게이트 자리는 거래대금/ATR 등 독립적인 정보로 대체한다.
    cur, prev, config = ctx.cur, ctx.prev, ctx.config
    stoch_k = _num(cur, "STOCH_K")
    stoch_d = _num(cur, "STOCH_D")
    stoch_k_prev = _num(prev, "STOCH_K")
    stoch_d_prev = _num(prev, "STOCH_D")
    if any(pd.isna(v) for v in (stoch_k, stoch_d, stoch_k_prev, stoch_d_prev)):
        return False, ctx, "STOCH_DATA_MISSING"
    stoch_golden_cross = stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d
    stoch_buy_signal = stoch_golden_cross or (
        stoch_k > stoch_d and STOCH_BUY_MIN <= stoch_k < config.stoch_overbought
    )
    if not stoch_buy_signal:
        return False, ctx, f"NO_STOCH_BUY_SIGNAL_K_{stoch_k:.1f}_D_{stoch_d:.1f}"

    # [2026-09-14][사용자 요청①] %D 하한 신규 추가. 실거래 141건 분석: 매수 시점 %D<50
    # 구간 승률 24-33%(n=64) vs %D 50-80 구간 42%(n=50) - %K만 보고 %D 자체엔 하한이
    # 없었던 공백을 메운다.
    if stoch_d < STOCH_D_BUY_MIN:
        return False, ctx, f"STOCH_D_TOO_LOW_{stoch_d:.1f}_LT_{STOCH_D_BUY_MIN:.1f}"

    # WILLIAMS_R = STOCH_K - 100 이므로 WILLIAMS_BUY_FLOOR/CEIL을 STOCH_K 스케일로 환산.
    # [2026-09-17] "stoch_k > stoch_k_prev"(직전 3분봉 대비 %K가 반드시 더 높아야 함)
    # 하드 요건을 제거한다. 2026-09-14~17 실매매 로그 분석(사용자 요청 - 예수금 부족 여부
    # 질의 중 매수 0건이 발견되어 확대 분석) 결과, 이 단일 틱 상승 요건이 gate_steps_
    # diagnostic 기준 전체 REJECT 스냅샷의 94.6%에서 실패로 찍히고, "나머지 7개 조건은
    # 전부 통과(근접 리젝, exactly-1-gate-fail)"인 2,759건 중 1,394건(50.5%)의 유일한
    # 실패 원인으로 확인됨 - 단일 최대 병목. 175330 JB금융지주 2026-09-14 09:09 사례로
    # 실증: RSI=100/ADX=100/-DI=0.0(포화값, 눌림 없는 강한 상승) 상태에서 %K가 직전봉
    # 90.0->81.82로 "여전히 30~90 밴드 안"인데도 한 틱 주춤했다는 이유만으로 리젝, 그
    # 시점(live=31,750) 매수를 가정하면 09:27 고점 33,100까지 20분 내 +4.25% 상승해
    # STAGED_TP1_PCT(3.0%)를 손절 없이 달성했을 조건이었음(3m 데이터 재생 확인). 이미
    # golden_cross 또는 (K>D and 밴드 내) + STOCH_D_BUY_MIN 하한을 통과한 뒤에 얹는
    # 추가 요건이었는데, WILLIAMS_R이 STOCH_K의 완전한 재표현(수학적 항등)이라 이 "상승
    # 중" 조건은 원래의 %K/%D 신호가 이미 확인한 것 이상의 정보를 주지 않으면서, 강한
    # 추세가 자연스럽게 만드는 오실레이터 미세 진동에는 취약했다 - BB 기울기(2026-09-06)/
    # 캔들 양봉(2026-08-25)에서 이미 반복된 "후행/노이즈성 지표에 단일 틱 하드컷을 걸면
    # 오히려 진짜 강한 추세를 더 못 잡는다" 패턴과 동일. 밴드 범위(floor~ceil) 확인은
    # 과열/침체 구간 배제 목적이 여전히 유효해 그대로 유지한다.
    stoch_k_floor = WILLIAMS_BUY_FLOOR + 100.0
    stoch_k_ceil = WILLIAMS_OVERBOUGHT_CEIL + 100.0
    williams_equiv_ok = stoch_k_floor <= stoch_k <= stoch_k_ceil
    if not williams_equiv_ok:
        return False, ctx, f"NO_WILLIAMS_BUY_SIGNAL_R_{stoch_k - 100.0:.1f}"
    return True, ctx, None


def _gate_min_entry_atr_volatility(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # [2026-09-07] 진입 최소 변동성(ATR%) 필터. 손절(ATR_STOP_MULTIPLIER)/1차 익절
    # (TP1_ATR_MULTIPLIER)은 ATR에 연동돼 있는데 정작 진입 게이트에는 ATR 확인이 전혀
    # 없어, ATR이 TP1(STAGED_TP1_PCT) 근처에도 못 미치는 저변동 종목이 그대로 진입을
    # 통과했다. ATR% >= STAGED_TP1_PCT * MIN_ENTRY_ATR_TO_TP1_RATIO 이어야 진입 허용.
    if not ENABLE_MIN_ENTRY_ATR_FILTER:
        return True, ctx, None
    cur, live_price = ctx.cur, ctx.live_price
    atr_val = _num(cur, "ATR")
    if pd.isna(atr_val) or not live_price or live_price <= 0:
        return True, ctx, None  # ATR 미산출 구간(초기 warmup)은 다른 게이트에 판단을 맡김
    atr_pct = atr_val / float(live_price)
    min_atr_pct = STAGED_TP1_PCT * MIN_ENTRY_ATR_TO_TP1_RATIO
    if atr_pct < min_atr_pct:
        return False, ctx, f"LOW_ENTRY_ATR_{atr_pct*100:.2f}%_LT_{min_atr_pct*100:.2f}%"
    return True, ctx, None


def _gate_min_liquidity_safety(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # 안전장치: 거래량 절대/상대 최소치 (저유동성 종목 차단)
    cur = ctx.cur
    vol = _num(cur, "volume")
    vol_ma = _num(cur, "VOL_MA20")
    if not any(pd.isna(v) for v in (vol, vol_ma)):
        if vol_ma < MIN_ENTRY_VOL_MA:
            return False, ctx, f"LOW_VOL_MA_ABS_{vol_ma:.0f}_LT_{MIN_ENTRY_VOL_MA}"
        if vol < MIN_ENTRY_VOLUME:
            return False, ctx, f"LOW_ABS_VOLUME_{vol:.0f}_LT_{MIN_ENTRY_VOLUME}"
        if vol_ma > 0:
            vol_ratio = vol / vol_ma
            if vol_ratio < 0.10:
                return False, ctx, f"LOW_VOLUME_RATIO_{vol_ratio:.4f}_LT_0.10"

    # [2026-09-07] 거래대금(turnover) 기반 유동성 하한 - 가격대에 무관한(cap-neutral)
    # 유동성 지표를 병행한다 (참고: turnover 필터는 시가총액/가격대와 무관하게 동일
    # 기준을 적용할 수 있다는 점이 절대 거래량(주수) 필터 대비 장점으로 꼽힌다).
    close_v = _num(cur, "close")
    if not any(pd.isna(v) for v in (vol, close_v)):
        turnover = close_v * vol
        if turnover < MIN_ENTRY_TURNOVER_KRW:
            return False, ctx, f"LOW_TURNOVER_{turnover:,.0f}_LT_{MIN_ENTRY_TURNOVER_KRW:,}"
    return True, ctx, None


def _gate_di_spread_min(ctx: BuyEvalContext) -> tuple[bool, BuyEvalContext, str | None]:
    # [2026-09-14][사용자 요청②] DI스프레드(+DI--DI) 최소치 신규 필수 게이트. 실거래
    # 141건 분석: 0-10구간 승률 25.6%(n=39), 10-20구간 36.4%(n=33), 20+구간 62.5%
    # (n=8, 표본 작음) - 스프레드가 음수(-DI 우세, 하락추세)인 경우도 자동으로 걸러진다.
    # R77-D의 adx_strength 가점(ADX_DI_SCORE_*)과는 별개의 독립 필수조건.
    di_plus = _num(ctx.cur, "DI_PLUS")
    di_minus = _num(ctx.cur, "DI_MINUS")
    if any(pd.isna(v) for v in (di_plus, di_minus)):
        return False, ctx, "DI_DATA_MISSING"
    di_spread = di_plus - di_minus
    if di_spread < DI_SPREAD_MIN_REQUIRED:
        return False, ctx, f"DI_SPREAD_TOO_SMALL_{di_spread:.1f}_LT_{DI_SPREAD_MIN_REQUIRED:.1f}"
    return True, ctx, None


BUY_GATE_CONDITIONS: list[BuyGateCondition] = [
    BuyGateCondition(
        "bb_slope_rising",
        "BB 중앙선 상승 추세 확인 (최근 lookback봉 대비 기울기 > BB_SLOPE_MIN_PCT)",
        ("BB_SLOPE_LOOKBACK_BARS", "BB_SLOPE_MIN_PCT"),
        _gate_bb_slope_rising,
    ),
    BuyGateCondition(
        "bb_mid_downtrend_block",
        "BB 중간선 최근 4봉 연속 우하향 시 매수 차단 (가격이 이미 돌파했으면 해제)",
        ("BB_MID_DOWNTREND_BARS",),
        _gate_bb_mid_downtrend_block,
    ),
    BuyGateCondition(
        "bb_mid_cross_up",
        "BB 중앙선 상향 돌파 (live cross / close 크로스 5봉 룩백 / 우상향 추세 지속)",
        (),
        _gate_bb_mid_cross_up,
    ),
    BuyGateCondition(
        "candle_bullish_and_chase_guard",
        "현재봉 양봉 확인 + 추격매수 방지 (인트라바 급등 차단, BB중간선 갭 차단, uptrend_continuation 전용 RSI/갭 기준)",
        ("CANDLE_GAIN_MIN_PCT", "CANDLE_GAIN_MAX_PCT", "BB_MID_CHASE_MAX_GAP_PCT",
         "UPTREND_CONT_CHASE_MAX_GAP_PCT", "UPTREND_CONT_CHASE_RSI_MAX"),
        _gate_candle_bullish_and_chase_guard,
    ),
    BuyGateCondition(
        "bb_upper_gap_min",
        "BB 상단까지 충분한 공간 확보",
        ("BB_UPPER_GAP_MIN_PCT",),
        _gate_bb_upper_gap_min,
    ),
    BuyGateCondition(
        "stochastic_buy_signal",
        "스토캐스틱+윌리엄스%R 통합 매수신호 (%K가 %D 상향돌파, 또는 %K>%D且 진짜 과열 아님 "
        "+ WILLIAMS_R 환산 밴드 내 상승, %D 자체도 STOCH_D_BUY_MIN 이상[R78-①]) - "
        "WILLIAMS_R=STOCH_K-100 수학적 등가라 게이트 통합",
        ("STOCH_BUY_MIN", "STOCH_D_BUY_MIN", "WILLIAMS_BUY_FLOOR", "WILLIAMS_OVERBOUGHT_CEIL"),
        _gate_stochastic_buy_signal,
    ),
    BuyGateCondition(
        "min_entry_atr_volatility",
        "진입 최소 변동성 확인 (ATR% >= STAGED_TP1_PCT*MIN_ENTRY_ATR_TO_TP1_RATIO) - "
        "TP1 도달이 통계적으로 가능한 최소 변동성 종목만 진입",
        ("ENABLE_MIN_ENTRY_ATR_FILTER", "MIN_ENTRY_ATR_TO_TP1_RATIO"),
        _gate_min_entry_atr_volatility,
    ),
    BuyGateCondition(
        "min_liquidity_safety",
        "저유동성 종목 차단 (거래량/거래량MA 절대치 및 비율 최소치 + 거래대금 최소치)",
        ("MIN_ENTRY_VOL_MA", "MIN_ENTRY_VOLUME", "MIN_ENTRY_TURNOVER_KRW"),
        _gate_min_liquidity_safety,
    ),
    BuyGateCondition(
        "di_spread_min",
        "[R78-②] DI스프레드(+DI--DI) 최소치 확보 (음수/약한 추세 방향성 배제)",
        ("DI_SPREAD_MIN_REQUIRED",),
        _gate_di_spread_min,
    ),
]


def run_buy_condition_pipeline_comment(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
) -> tuple[bool, str]:
    """BB 중앙선 상승 돌파 전략: BUY_GATE_CONDITIONS(9개 필수조건) 순차 통과 +
    BUY_SCORE_RULES 가점 합계가 config.bb_buy_score_threshold 이상.

    BB_MIDDLE/BB_STD는 calculate_indicators()에서 rolling(window=BB_PERIOD,
    min_periods=1)로 계산되는 성장형(growing-window) 평균이다 - 이는 실제 HTS/MTS가
    분봉 차트에 볼린저밴드를 그리는 방식과 동일하다(2026-08-24 MTS 실측: 1분봉/3분봉
    모두 09:00 장 시작 즉시 정상적으로 그려지고, 그 구간에서 골든크로스도 유효하게
    발생함을 확인 - r002 Update log 2026-08-24 참조). 판정에 필요한 최소 봉 수는
    prev/cur 두 봉뿐이다.

    조건을 추가/삭제/순서변경하려면 이 함수 본문이 아니라 위쪽의
    BUY_GATE_CONDITIONS / BUY_SCORE_RULES 리스트를 수정한다. 개장가드와 최종
    점수임계값 체크 2개는 점수 자체에 의존하는 후처리라 의도적으로 게이트
    리스트에 넣지 않고 아래 그대로 남겨둔다.
    """
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")
    cur_bb_upper = _num(cur, "BB_UPPER")
    prev_bb = _num(prev, "BB_MIDDLE")

    if any(pd.isna(v) for v in (cur_bb, cur_bb_upper, prev_bb)):
        return False, "MISSING_INDICATOR"

    ctx = BuyEvalContext(
        frame=frame, now=now, live_price=live_price, cross_info=cross_info, config=config,
        cur=cur, prev=prev, cur_bb=cur_bb, cur_bb_upper=cur_bb_upper, prev_bb=prev_bb,
        bb_slope_pct=_compute_bb_slope_pct(frame),
    )

    for gate in BUY_GATE_CONDITIONS:
        passed, ctx, reason = gate.eval_fn(ctx)
        if not passed:
            return False, reason

    # 가점 조건 합산 (BUY_SCORE_RULES 참조, 실제 최대 점수는 rule.max_score 합계)
    score = sum(rule.eval_fn(ctx) for rule in BUY_SCORE_RULES)
    live_cross_up = ctx.cross_eval["live_cross_up"]
    close_cross = ctx.cross_eval["close_cross"]

    # 개장 직후 보호: 09:00~09:08 (첫 3봉) 동안 추격매수 차단 강화 (BUY_GATE_CONDITIONS
    # 밖에 남겨둔 의도적 예외 - 점수가 계산된 이후에만 판단 가능함).
    # live_cross_up(실시간 돌파)은 제외, close/uptrend 기반 진입만 제한
    if not live_cross_up and now.hour == 9 and now.minute < OPENING_GUARD_MINUTES:
        if score < OPENING_GUARD_SCORE_THRESHOLD:
            return False, f"OPENING_GUARD_{now.strftime('%H:%M')}_{score}_LT_{OPENING_GUARD_SCORE_THRESHOLD}"

    if score < config.bb_buy_score_threshold:
        return False, f"LOW_SCORE_{score}_LT_{config.bb_buy_score_threshold}"

    if live_cross_up:
        trigger = "LIVE_PRICE_BB_UP_CROSS"
    elif close_cross:
        trigger = "CLOSE_BB_UP_CROSS"
    else:
        trigger = "UPTREND_CONT"
    return True, f"{trigger}_SCORE_{score}"


# HYBRID_3MIN_CONTEXT_GATES: ENABLE_1MIN_TRIGGER_3MIN_CONTEXT(r001) 전용. 크로스 감지
# (bb_mid_cross_up)와 캔들/추격가드(candle_bullish_and_chase_guard)는 호출측이 1분봉
# 자체 기준(check_buy_condition_1min)으로 이미 확인했으므로 제외하고, 나머지 3분봉
# 컨텍스트 게이트(추세/매집/공간/모멘텀/유동성)만 재사용한다.
# [2026-09-20] bb_mid_downtrend_block은 r001 ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK(기본 True=포함)로
# 별도 전략 변경 스위치를 둔다 - 최근 라이브 5거래일 단독 차단 0건이었지만 다른 게이트가 논리적으로
# 포함하지는 않아(Codex 검토) 기본 동작은 그대로 두고 백테스트/섀도 검증 후 판단하도록 분리했다.
_HYBRID_EXCLUDED_GATE_NAMES = ("bb_mid_cross_up", "candle_bullish_and_chase_guard") + (
    () if ENABLE_HYBRID_BB_MID_DOWNTREND_BLOCK else ("bb_mid_downtrend_block",)
)
HYBRID_3MIN_CONTEXT_GATES: list[BuyGateCondition] = [
    gate for gate in BUY_GATE_CONDITIONS
    if gate.name not in _HYBRID_EXCLUDED_GATE_NAMES
]


def build_context_eval(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
) -> tuple[BuyEvalContext | None, str]:
    """3분봉 컨텍스트 판정 준비 단계: 봉 수/지표 산출 여부를 확인하고 BuyEvalContext를 만든다.

    run_3min_context_pipeline(g003 포함 기존 호출자)과 r005_buy_conditions(_009)가 같은 구현을 쓰도록
    run_3min_context_pipeline의 앞부분을 그대로 추출한 것이다(2026-09-21, 동작 불변). 준비가 안 됐으면
    (None, 사유), 됐으면 (BuyEvalContext, "")를 반환한다."""
    if len(frame) < 2:
        return None, "HYBRID_3MIN_CTX_INSUFFICIENT_BARS"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")
    cur_bb_upper = _num(cur, "BB_UPPER")
    prev_bb = _num(prev, "BB_MIDDLE")

    if any(pd.isna(v) for v in (cur_bb, cur_bb_upper, prev_bb)):
        return None, "HYBRID_3MIN_CTX_MISSING_INDICATOR"

    ctx = BuyEvalContext(
        frame=frame, now=now, live_price=live_price, cross_info=cross_info, config=config,
        cur=cur, prev=prev, cur_bb=cur_bb, cur_bb_upper=cur_bb_upper, prev_bb=prev_bb,
        bb_slope_pct=_compute_bb_slope_pct(frame),
    )
    return ctx, ""


def evaluate_context_score(
    ctx: BuyEvalContext,
    now: pd.Timestamp,
    config: R76StrategyConfig,
) -> tuple[bool, str]:
    """3분봉 컨텍스트 판정 마무리 단계: 가점 합산 + 개장 직후 보호 + 최종 점수 임계값.

    run_3min_context_pipeline의 뒷부분을 그대로 추출한 것(2026-09-21, 동작 불변) - r005_buy_conditions
    (_017)와 공유한다."""
    score = sum(rule.eval_fn(ctx) for rule in BUY_SCORE_RULES)

    # 개장 직후 보호는 run_buy_condition_pipeline_comment와 동일하게 유지 - 1분봉
    # 트리거는 live_cross_up(실시간 돌파)만큼 즉각적이지 않으므로 예외 처리하지 않는다.
    if now.hour == 9 and now.minute < OPENING_GUARD_MINUTES:
        if score < OPENING_GUARD_SCORE_THRESHOLD:
            return False, f"HYBRID_3MIN_CTX_OPENING_GUARD_{now.strftime('%H:%M')}_{score}_LT_{OPENING_GUARD_SCORE_THRESHOLD}"

    if score < config.bb_buy_score_threshold:
        return False, f"HYBRID_3MIN_CTX_LOW_SCORE_{score}_LT_{config.bb_buy_score_threshold}"

    return True, f"HYBRID_3MIN_CTX_SCORE_{score}"


def run_3min_context_pipeline(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
) -> tuple[bool, str]:
    """ENABLE_1MIN_TRIGGER_3MIN_CONTEXT 하이브리드 경로의 2단계: 1분봉 트리거(호출측에서
    이미 확인 완료)가 통과된 뒤, 3분봉 상위 프레임에서 "아직 추세/유동성 컨텍스트가
    우호적인가"만 재확인한다. bb_mid_cross_up/candle_bullish_and_chase_guard를 제외한
    HYBRID_3MIN_CONTEXT_GATES + BUY_SCORE_RULES를 run_buy_condition_pipeline_comment와
    동일하게 재사용해 판정 로직 드리프트를 방지한다(2026-08-28 r002 Update log 참조).

    [2026-09-21] 준비 단계(build_context_eval)와 점수 단계(evaluate_context_score)를 공용 함수로
    분리했다 - r005_buy_conditions가 같은 두 함수를 쓰며 게이트는 번호 붙은 조건으로 하나씩 평가한다.
    결과(통과 여부/사유 문자열)는 분리 전과 동일하다.
    """
    ctx, reason = build_context_eval(frame, now, live_price, cross_info, config)
    if ctx is None:
        return False, reason

    for gate in HYBRID_3MIN_CONTEXT_GATES:
        passed, ctx, reason = gate.eval_fn(ctx)
        if not passed:
            return False, f"HYBRID_3MIN_CTX_{reason}"

    return evaluate_context_score(ctx, now, config)


def check_buy_condition_1min_hybrid_trigger(
    frame_1min: pd.DataFrame,
    context_uptrend_continuation: bool = False,
) -> tuple[bool, str]:
    """하이브리드 매수 경로(1분봉 트리거 -> 3분봉 컨텍스트)의 1단계 1분봉 트리거 - r003 실전/g003
    백테스트 공용(2026-09-20 r003/g003에 따로 있던 복사본을 여기로 통합. 이 함수의 원형이던
    check_buy_condition_1min 1분봉 단독 경로는 같은 날 삭제됨).

    2026-08-28 1차 검증(HYBRID_1MIN_TRIGGER 상수 도입 전)에서 check_buy_condition_1min을
    그대로 재사용했더니 매수 0건 - 리젝 469/507건이 require_fresh_cross(이 1분봉에서 "막"
    크로스했을 때만 인정, 룩백 없음)에서 발생. 3분봉 bb_mid_cross_up은 5봉 룩백+우상향
    지속 예외가 있는데 1분봉엔 그런 관용도가 전혀 없었던 것 - 이 함수는 그 룩백을
    HYBRID_1MIN_TRIGGER_LOOKBACK_BARS만큼 추가한다(_evaluate_bb_mid_cross의 close_cross
    N봉 룩백과 동일 패턴: 전환봉 이후 현재까지 BB 위 연속 유지 시 인정). 또한 캔들/추격
    가드 문턱도 3분봉 값(CANDLE_GAIN_MAX_PCT 등)을 그대로 쓰지 않고 HYBRID_1MIN_TRIGGER_*
    전용 값을 쓴다 - 1분봉은 3분봉보다 캔들 하나의 시간폭이 짧아 같은 % 문턱이 상대적으로
    더 쉽게 초과됨(HPSP 1차 검증에서 candle_gain 0.87~1.67%로 3분봉 문턱 0.8% 초과 반복 확인).

    [2026-09-09] 452190 한빛레이저 사례: 룩백(3->8봉으로 완화했음에도) 밖에서 돌파한 뒤
    오래/강하게 지속되는 랠리는 여전히 놓칠 수 있다(룩백은 "완화"일 뿐 무제한이 아님) -
    3분봉 bb_mid_cross_up 게이트의 uptrend_continuation과 동일한 예외를 추가한다.
    두 경로:
    (a) 이 함수 자체가 1분봉 자체 지표(ADX/+DI/-DI/MA5/BB슬로프)로 우상향 지속을 판정
        (_evaluate_bb_mid_cross 재사용 - 3분 게이트와 동일 공식, 프레임만 1분봉).
    (b) 호출측이 3분봉 컨텍스트에서 이미 uptrend_continuation으로 판정했으면
        context_uptrend_continuation=True로 전달 - 그 신호를 그대로 인정한다.
    두 경로 모두 크로스 "발견" 여부만 대체할 뿐, 이후의 캔들/BB갭 안전장치는 그대로
    전부 적용한다(거래량 하한은 3분봉 컨텍스트 min_liquidity_safety가 단독 담당).
    """
    if frame_1min is None or len(frame_1min) < 2:
        return False, "1MIN_INSUFFICIENT_BARS"

    cur = frame_1min.iloc[-1]
    prev = frame_1min.iloc[-2]

    cur_bb = _num(cur, "BB_MIDDLE")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_close = _num(cur, "close")
    prev_close = _num(prev, "close")
    cur_open = _num(cur, "open")

    if any(pd.isna(v) for v in (cur_bb, prev_bb, cur_close, prev_close, cur_open)) or cur_open <= 0:
        return False, "1MIN_MISSING_INDICATOR"

    golden_cross = prev_close <= prev_bb and cur_close > cur_bb
    bars_since_cross = 0  # 신선한 크로스(golden_cross=True) 기본값 - 경과봉 0, 갭 상한 완화 없음
    trigger_reason = "1MIN_BB_MID_GOLDEN_CROSS_LOOKBACK"
    if not golden_cross:
        _found = False
        for _lb in range(3, min(HYBRID_1MIN_TRIGGER_LOOKBACK_BARS + 2, len(frame_1min)) + 1):
            _bar_n_close = _num(frame_1min.iloc[-_lb], "close")
            _bar_n_bb = _num(frame_1min.iloc[-_lb], "BB_MIDDLE")
            if any(pd.isna(v) for v in (_bar_n_close, _bar_n_bb)) or _bar_n_close > _bar_n_bb:
                continue  # 이 봉도 BB 위이면 더 이전 탐색 or NaN
            _all_above = all(
                not any(pd.isna(v) for v in (
                    _num(frame_1min.iloc[-_k], "close"), _num(frame_1min.iloc[-_k], "BB_MIDDLE"),
                ))
                and _num(frame_1min.iloc[-_k], "close") > _num(frame_1min.iloc[-_k], "BB_MIDDLE")
                for _k in range(1, _lb)
            )
            if _all_above:
                _found = True
                # 실제 돌파봉은 -_lb(미돌파 마지막봉) 바로 다음인 -(_lb-1) - 그 봉부터
                # cur(-1)까지 경과한 봉 수 = (_lb-1)의 위치 차이 = _lb-2.
                bars_since_cross = _lb - 2
                break

        if not _found:
            if context_uptrend_continuation:
                _found = True
                trigger_reason = "1MIN_UPTREND_CONTINUATION_3MIN_CTX"
            else:
                _bb_slope_1min = _compute_bb_slope_pct(frame_1min)
                _uptrend_eval = _evaluate_bb_mid_cross(
                    frame_1min, cur, prev, cur_bb, prev_bb, cur_close, _bb_slope_1min, {},
                )
                if _uptrend_eval.get("uptrend_continuation"):
                    _found = True
                    trigger_reason = "1MIN_UPTREND_CONTINUATION"

        if not _found:
            return False, "1MIN_NO_BB_MID_GOLDEN_CROSS"

        if trigger_reason != "1MIN_BB_MID_GOLDEN_CROSS_LOOKBACK":
            # 우상향 지속 경로로 인정된 경우 - 정확한 경과봉을 알 수 없으므로 BB갭
            # 완화도(아래) 최대치를 적용해 추격매수 가드가 과도하게 좁아지지 않게 한다.
            bars_since_cross = HYBRID_1MIN_TRIGGER_LOOKBACK_BARS

    candle_gain_pct = (cur_close - cur_open) / cur_open * 100.0
    if candle_gain_pct < HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT:
        return False, f"1MIN_CANDLE_NOT_BULLISH_{candle_gain_pct:.2f}%_LT_{HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT:.1f}%"
    if candle_gain_pct > HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT:
        return False, f"1MIN_CHASE_BUY_INTRABAR_{candle_gain_pct:.2f}%_GT_{HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT:.1f}%"

    if cur_bb > 0:
        bb_gap_pct = (cur_close - cur_bb) / cur_bb * 100.0
        # [2026-09-07] 크로스가 BB_MID 위로 계속 유지 중(=유효한 추세 지속)인데도 BB_MID가
        # 후행지표라 갭이 계속 벌어져 고정 상한에 매 폴링 걸리는 문제 완화 - 크로스 후
        # 경과봉 수만큼 상한을 소폭 완화하되 CEILING_PCT로 무한 완화는 방지한다.
        # [2026-09-18] uptrend_continuation 경로(추세 지속이 다른 지표로 이미 검증된
        # 경우)는 신선한 크로스보다 높은 CEILING_UPTREND_PCT를 쓴다 - 1분봉 BB중간선이
        # 급등을 못 따라가는 날(20260918 000500/043260 등 분석)엔 고정 1.1% 상한이
        # 영구 차단으로 이어졌음(20260918 r001 Update log 참조).
        is_uptrend_continuation = trigger_reason in (
            "1MIN_UPTREND_CONTINUATION_3MIN_CTX", "1MIN_UPTREND_CONTINUATION",
        )
        gap_ceiling_pct = (
            HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT if is_uptrend_continuation
            else HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT
        )
        allowed_gap_pct = min(
            HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT + bars_since_cross * HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR,
            gap_ceiling_pct,
        )
        if bb_gap_pct > allowed_gap_pct:
            return False, f"1MIN_CHASE_BUY_BB_GAP_{bb_gap_pct:.2f}%_GT_{allowed_gap_pct:.2f}%"

    # [2026-09-20] 거래량/유동성 하한 통일: 여기(1분봉)에 있던 시간프레임 환산 복사본
    # (HYBRID_1MIN_MIN_ENTRY_VOL_MA/VOLUME 340/500, 비율 0.10)을 삭제하고, 3분봉 컨텍스트의
    # min_liquidity_safety 게이트(MIN_ENTRY_VOL_MA/MIN_ENTRY_VOLUME/MIN_ENTRY_TURNOVER_KRW)를
    # 유일한 하한으로 쓴다. 최근 라이브 5거래일(0914~0918)에서 1분봉 하한이 막았지만 3분봉
    # 하한은 통과한 틱이 12,070건이고, 그중 나머지 3분봉 게이트까지 전부 통과해 점수 단계에
    # 도달했을 틱은 217건(그 뒤 점수 10점/2회 연속 확인이 다시 걸러냄).
    return True, trigger_reason


def check_1min_dead_cross(
    frame_1min: pd.DataFrame,
    lookback_bars: int,
) -> tuple[bool, str, int]:
    """1분봉 BB중심선 데드크로스(하향 이탈) 판정 - check_buy_condition_1min_hybrid_trigger의 골든크로스
    감지 로직을 그대로 대칭 반전한 것(신선한 크로스 OR N봉 룩백 + 그 이후 계속 BB 아래 유지 확인).

    [2026-09-23] 사용자 요청("매수/매도 컨셉 재검토", 204620 글로벌텍스프리 사례) + Codex 설계검토 - 매도측에
    매수측 1분봉 트리거와 대칭되는 조건이 없다는 지적에 대한 응답. r006 _011_hybrid_1min_dead_cross_exit가
    이 함수를 호출한다. 매수측과 달리 캔들/BB갭 추격가드는 넣지 않는다 - 매도는 "빨리 자르는" 게 목적이라
    추격 방지 로직이 오히려 반대 효과(청산 지연)를 낸다.

    반환: (감지 여부, 사유 문자열, 크로스 이후 경과 봉 수 - 신선한 크로스는 0)
    """
    if frame_1min is None or len(frame_1min) < 2:
        return False, "1MIN_INSUFFICIENT_BARS", 0

    cur = frame_1min.iloc[-1]
    prev = frame_1min.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")
    prev_bb = _num(prev, "BB_MIDDLE")
    cur_close = _num(cur, "close")
    prev_close = _num(prev, "close")
    if any(pd.isna(v) for v in (cur_bb, prev_bb, cur_close, prev_close)):
        return False, "1MIN_MISSING_INDICATOR", 0

    dead_cross = prev_close >= prev_bb and cur_close < cur_bb
    if dead_cross:
        return True, "1MIN_BB_MID_DEAD_CROSS", 0

    for lb in range(2, min(lookback_bars + 1, len(frame_1min)) + 1):
        bar_close = _num(frame_1min.iloc[-lb], "close")
        bar_bb = _num(frame_1min.iloc[-lb], "BB_MIDDLE")
        if any(pd.isna(v) for v in (bar_close, bar_bb)) or bar_close < bar_bb:
            continue  # 이 봉이 여전히 BB 아래(또는 NaN)이면 아직 크로스 지점(BB 위였던 마지막 봉)을 못 찾은 것 - 더 이전 탐색
        all_below = all(
            not any(pd.isna(v) for v in (
                _num(frame_1min.iloc[-k], "close"), _num(frame_1min.iloc[-k], "BB_MIDDLE"),
            ))
            and _num(frame_1min.iloc[-k], "close") < _num(frame_1min.iloc[-k], "BB_MIDDLE")
            for k in range(1, lb)
        )
        if all_below:
            return True, "1MIN_BB_MID_DEAD_CROSS_LOOKBACK", lb - 2

    return False, "1MIN_NO_BB_MID_DEAD_CROSS", 0


def passes_opening_gap_volume_gate(
    code: str,
    current_dt: datetime,
    session_open_dt: datetime | None,
    gap_pct: float | None,
    buy_frame: pd.DataFrame,
    gap_blocked_codes: set[str],
) -> tuple[bool, str]:
    """개장 초반 갭/거래량폭발 게이트 - r003 실전/g003 백테스트 공용(2026-09-20 통합).

    개장 초반 갭/거래량폭발 게이트.

    r002 스캐너는 전일 종가 기준 데이터라 당일 시가 갭이나 초반 거래량 급변을
    반영하지 못한다. 개장 후 OPENING_GAP_GATE_WINDOW_MINUTES 이내에만 적용되며,
    심한 갭하락(OPENING_GAP_HARD_FLOOR_PCT 미만)은 당일 재검사 없이 영구 차단한다.
    """
    norm_code = str(code).zfill(6)
    if norm_code in gap_blocked_codes:
        return False, "OPENING_GAP_BLOCKED_TODAY"

    if session_open_dt is None or gap_pct is None:
        return True, "OK"

    elapsed = (current_dt - session_open_dt).total_seconds()
    if elapsed < 0 or elapsed > OPENING_GAP_GATE_WINDOW_MINUTES * 60:
        return True, "OK"  # 게이트 적용 윈도우 밖이면 통과 (일반 로직으로 복귀)

    if gap_pct < OPENING_GAP_HARD_FLOOR_PCT:
        gap_blocked_codes.add(norm_code)
        return False, f"OPENING_GAP_DOWN_BLOCKED_{gap_pct*100:.2f}%"

    if not (OPENING_GAP_MIN_PCT <= gap_pct <= OPENING_GAP_MAX_PCT):
        return False, f"OPENING_GAP_OUT_OF_RANGE_{gap_pct*100:.2f}%"

    cur_row = buy_frame.iloc[-1] if buy_frame is not None and not buy_frame.empty else None
    vol = _num(cur_row, "volume") if cur_row is not None else float("nan")
    vol_ma = _num(cur_row, "VOL_MA20") if cur_row is not None else float("nan")
    if not any(pd.isna(v) for v in (vol, vol_ma)) and vol_ma > 0:
        vol_ratio = vol / vol_ma
        if vol_ratio < OPENING_MIN_EARLY_VOLUME_RATIO:
            return False, f"OPENING_VOLUME_INSUFFICIENT_{vol_ratio:.2f}x"

    return True, "OK"


def gate_steps_diagnostic(
    frame: pd.DataFrame,
    now: pd.Timestamp,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
    gates: list[BuyGateCondition] | None = None,
) -> str:
    """로그 표시 전용 진단 함수. gates(기본값 BUY_GATE_CONDITIONS) 리스트의 각 게이트를
    run_buy_condition_pipeline_comment/run_3min_context_pipeline처럼 실제 매수 판정에
    쓰지 않고, 조기종료 없이 전부 평가해 "gate_name(P|F)" 형태로 이어붙여 반환한다.
    실제 판정 로직(순차 통과형)에는 전혀 영향을 주지 않는다 - 리젝된 종목이 어느
    게이트에서 걸렸는지뿐 아니라 나머지 게이트도 통과했는지 한눈에 보기 위함.
    """
    gates = gates if gates is not None else BUY_GATE_CONDITIONS
    if len(frame) < 2:
        return " ".join(f"{gate.name}(-)" for gate in gates)

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")
    cur_bb_upper = _num(cur, "BB_UPPER")
    prev_bb = _num(prev, "BB_MIDDLE")
    if any(pd.isna(v) for v in (cur_bb, cur_bb_upper, prev_bb)):
        return " ".join(f"{gate.name}(-)" for gate in gates)

    ctx = BuyEvalContext(
        frame=frame, now=now, live_price=live_price, cross_info=cross_info, config=config,
        cur=cur, prev=prev, cur_bb=cur_bb, cur_bb_upper=cur_bb_upper, prev_bb=prev_bb,
        bb_slope_pct=_compute_bb_slope_pct(frame),
    )

    parts: list[str] = []
    for gate in gates:
        try:
            passed, ctx, _reason = gate.eval_fn(ctx)
            mark = "P" if passed else "F"
        except Exception:
            mark = "-"
        parts.append(f"{gate.name}({mark})")
    return " ".join(parts)


def check_sell_condition(
    frame: pd.DataFrame,
    pnl_pct: float,
    live_price: float,
    cross_info: dict[str, object],
    config: R76StrategyConfig,
) -> tuple[bool, str]:
    if len(frame) < 2:
        return False, "INSUFFICIENT_BARS"

    if config.enable_box_range_hold_tech_sell and config.stop_loss_percent < pnl_pct < config.take_profit_percent:
        is_box, box_info = _is_box_range_hold_zone(frame, config)
        if is_box:
            return False, f"BOX_RANGE_HOLD_{box_info}"

    cur = frame.iloc[-1]
    prev = frame.iloc[-2]
    cur_bb = _num(cur, "BB_MIDDLE")

    price_cross_down = (
        cross_info.get("signal") == "cross_down"
        and not pd.isna(cur_bb)
        and live_price < cur_bb
    )
    if price_cross_down:
        score = _sell_support_score(cur, prev, config)
        # [2026-09-23] 즉시손절(ma5_bb_down_cross_immediate_pnl, 기본 -0.7%) 체크를 min_pnl(0.0%) 차단보다
        # 먼저 수행한다. immediate_pnl이 항상 min_pnl보다 더 마이너스라, 순서가 반대였던 기존 코드는 pnl<0.0%
        # 이면 바로 위에서 False를 반환해버려 이 즉시손절 분기가 영원히 도달 불가능했다(2026-09-21 세션에서
        # 발견/기록만 되고 방치된 죽은 코드 - 204620 글로벌텍스프리 2026-09-23 09:33 매도 지연 사례 분석 중
        # 재확인, Codex 검토로 fix 순서 확정). pnl>=0.0% 구간의 기존 동작(BLOCKED_PNL/점수별 최소수익 요건)은
        # 그대로 유지 - 이 재정렬은 pnl<=-0.7% 구간에서만 동작을 바꾼다(기존: 항상 BLOCKED_PNL로 반려 ->
        # 신규: 즉시 손절).
        if pnl_pct <= config.ma5_bb_down_cross_immediate_pnl:
            # 실손실이 하드 임계치까지 커졌으면 점수/수익 요건과 무관하게 즉시 손절.
            if score >= 1:
                return True, f"LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_{score}"
            return True, "LIVE_PRICE_BB_DOWN_CROSS"

        if pnl_pct < config.ma5_bb_down_cross_min_pnl:
            return False, (
                f"LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_PNL_{pnl_pct * 100:.2f}%"
                f"_LT_{config.ma5_bb_down_cross_min_pnl * 100:.2f}%"
            )

        if score >= config.ma5_bb_down_cross_immediate_score:
            # [2026-08-26] AUX_REVERSAL_SCORE 분기와 동일한 점수별 최소 수익 요건을 적용한다.
            # 기존에는 ma5_bb_down_cross_min_pnl(0.0%, 위에서 이미 통과)만 넘으면 score>=2로
            # 즉시 확정 매도였는데, 같은 _sell_support_score를 쓰는 AUX_REVERSAL_SCORE 분기는
            # score4/3/2에 각각 0.3%/0.8%/1.5% 이상 실현수익을 요구한다 - 매수 직후 BB중심선
            # 부근의 얕은 하향돌파(진입 자체가 BB중심선 근처 마진 진입일 때 자연스러운 되돌림)만
            # 으로도 손익분기점 근처에서 바로 매도되던 문제(083650 비에이치아이 2026-08-26 사례:
            # 매수 14분 후 pnl≈0%, score=4로 즉시 매도 후 -0.13% 손실 확정)를 이 분기도 동일한
            # 수익 요건으로 맞춰 방지한다.
            min_pnl_req = (
                config.aux_sell_min_pnl_score4 if score >= 4
                else config.aux_sell_min_pnl_score3 if score == 3
                else config.aux_sell_min_pnl_score2
            )
            if pnl_pct >= min_pnl_req:
                return True, f"LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_{score}"
            return False, (
                f"LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_SCORE_{score}_PNL_{pnl_pct * 100:.2f}%"
                f"_LT_{min_pnl_req * 100:.2f}%"
            )
        return False, f"LIVE_PRICE_BB_DOWN_CROSS_WEAK_SCORE_{score}"

    score = _sell_support_score(cur, prev, config)
    if score >= 4:
        min_pnl_req: float | None = config.aux_sell_min_pnl_score4
    elif score == 3:
        min_pnl_req = config.aux_sell_min_pnl_score3
    elif score == 2:
        min_pnl_req = config.aux_sell_min_pnl_score2
    else:
        return False, "NO_SELL_SIGNAL"

    if pnl_pct >= min_pnl_req:
        return True, f"AUX_REVERSAL_SCORE_{score}"
    return False, f"AUX_BLOCKED_SCORE_{score}_PNL_{pnl_pct * 100:.2f}%_LT_{min_pnl_req * 100:.2f}%"


# ---------------------------------------------------------------------------
# Shared indicator calculation (used by r006 and r007)
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["MA_5"] = out["close"].rolling(window=MA_PERIOD, min_periods=1).mean()
    out["VOL_MA20"] = out["volume"].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean()

    out["EMA_20"] = out["close"].ewm(span=EMA_20_PERIOD, adjust=False).mean()
    out["EMA_60"] = out["close"].ewm(span=EMA_60_PERIOD, adjust=False).mean()

    out["BB_MIDDLE"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).mean()
    out["BB_STD"] = out["close"].rolling(window=BB_PERIOD, min_periods=1).std()
    out["BB_UPPER"] = out["BB_MIDDLE"] + out["BB_STD"] * BB_STD_MULTIPLIER
    out["BB_LOWER"] = out["BB_MIDDLE"] - out["BB_STD"] * BB_STD_MULTIPLIER

    delta = out["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / RSI_PERIOD, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out["RSI"] = 100 - (100 / (1 + rs))
    out.loc[avg_loss == 0, "RSI"] = 100.0
    out["RSI_SIGNAL"] = out["RSI"].rolling(window=RSI_SIGNAL_PERIOD, min_periods=1).mean()

    low_n = out["low"].rolling(window=STOCH_K_PERIOD, min_periods=1).min()
    high_n = out["high"].rolling(window=STOCH_K_PERIOD, min_periods=1).max()
    denom = (high_n - low_n).replace(0, float("nan"))
    out["STOCH_K"] = 100.0 * (out["close"] - low_n) / denom
    out["STOCH_D"] = out["STOCH_K"].rolling(window=STOCH_D_PERIOD, min_periods=1).mean()

    high_w = out["high"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).max()
    low_w = out["low"].rolling(window=WILLIAMS_R_PERIOD, min_periods=1).min()
    wr_denom = (high_w - low_w).replace(0, float("nan"))
    out["WILLIAMS_R"] = -100.0 * (high_w - out["close"]) / wr_denom
    out["WILLIAMS_D"] = out["WILLIAMS_R"].rolling(window=WILLIAMS_D_PERIOD, min_periods=1).mean()

    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    raw_money_flow = typical_price * out["volume"]
    price_delta = typical_price.diff()
    positive_flow = raw_money_flow.where(price_delta > 0, 0.0)
    negative_flow = raw_money_flow.where(price_delta < 0, 0.0)
    positive_sum = positive_flow.rolling(window=MFI_PERIOD, min_periods=1).sum()
    negative_sum = negative_flow.rolling(window=MFI_PERIOD, min_periods=1).sum()
    money_ratio = positive_sum / negative_sum.replace(0, float("nan"))
    out["MFI"] = 100.0 - (100.0 / (1.0 + money_ratio))
    out.loc[(positive_sum == 0) & (negative_sum == 0), "MFI"] = 50.0
    out.loc[(negative_sum == 0) & (positive_sum > 0), "MFI"] = 100.0
    out.loc[(positive_sum == 0) & (negative_sum > 0), "MFI"] = 0.0

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    high_diff = out["high"] - out["high"].shift(1)
    low_diff = out["low"].shift(1) - out["low"]
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    ema_tr = tr.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    ema_plus = plus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    ema_minus = minus_dm.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()
    out["DI_PLUS"] = 100.0 * ema_plus / ema_tr.replace(0, float("nan"))
    out["DI_MINUS"] = 100.0 * ema_minus / ema_tr.replace(0, float("nan"))
    di_sum = (out["DI_PLUS"] + out["DI_MINUS"]).replace(0, float("nan"))
    dx = 100.0 * (out["DI_PLUS"] - out["DI_MINUS"]).abs() / di_sum
    out["ADX"] = dx.ewm(alpha=1.0 / ADX_PERIOD, min_periods=1, adjust=False).mean()

    cum_vol = out["volume"].cumsum()
    out["VWAP"] = (out["close"] * out["volume"]).cumsum() / cum_vol.replace(0, float("nan"))

    # ATR - 변동성 기반 손익비 판단용 평균 진폭
    true_range = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = true_range.ewm(alpha=1.0 / ATR_PERIOD, min_periods=1, adjust=False).mean()

    close_diff = out["close"].diff()
    obv_vol = out["volume"] * close_diff.gt(0).astype(float) - out["volume"] * close_diff.lt(0).astype(float)
    out["OBV"] = obv_vol.cumsum()
    out["OBV_MA"] = out["OBV"].rolling(window=OBV_MA_PERIOD, min_periods=1).mean()

    return out


