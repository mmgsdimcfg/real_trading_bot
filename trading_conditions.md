# 매수/매도 조건 정리 (R76 전략)

- 기준 파일: `live_trading/r002_strategy_core_shared.py` (핵심 판정 로직), `live_trading/r001_define_config.py` (define/설정값)
- 매수 필수조건/가점항목은 `r002_strategy_core_shared.py`의 `BUY_GATE_CONDITIONS` / `BUY_SCORE_RULES` 리스트에서 자동 추출 (2026-08-26 기준, `neutral` 리스크 프로파일 적용값)
- 매도 조건은 `check_sell_condition()` / `_sell_support_score()`를 직접 읽어 수기 정리 (객체화 대상 아님)
- 값이 바뀌면 이 문서도 다시 생성/갱신 필요 (매수 표는 `gen_buy_doc.py` 스타일 스크립트로 재생성 가능)

---

## 1. 매수 조건

매수는 3단계로 판정된다: **① 3분봉 필수조건 9개(순차 통과) → ② 가점 합산이 임계값 이상 → ③(선택) 1분봉 진입 타이밍 재검증**.

### 1-1. 필수조건 (BUY_GATE_CONDITIONS) — 순서대로 통과, 하나라도 실패하면 즉시 REJECT

| # | 조건명 | 설명 | 관련 define (현재값) |
|---|--------|------|----------------------|
| 1 | `bb_slope_rising` | BB 중앙선 상승 추세 확인 (최근 lookback봉 대비 기울기 > 최소값) | `BB_SLOPE_LOOKBACK_BARS`=20, `BB_SLOPE_MIN_PCT`=-2.0 |
| 2 | `bb_mid_downtrend_block` | BB 중간선이 최근 4봉 연속 우하향이면 매수 차단 (단, 가격이 이미 BB중간선을 상향 돌파했으면 차단 해제) | `BB_MID_DOWNTREND_BARS`=5 |
| 3 | `bb_mid_cross_up` | BB 중앙선 상향 돌파: 실시간 크로스 신호 OR 종가 기준 크로스(5봉 룩백) OR 크로스 없이도 강한 우상향 추세 지속(uptrend_continuation) 중 하나라도 만족 | - |
| 4 | `pre_cross_accumulation_bar` | 돌파 직전 "매집봉"(BB중간값 포함 + 양봉 + 거래량이 VOL_MA20 대비 증가) 존재 확인 | `ENABLE_PRE_CROSS_ACCUM_BAR_CHECK`=True, `PRE_CROSS_ACCUM_LOOKBACK_BARS`=8봉, `PRE_CROSS_ACCUM_VOL_RATIO_MIN`=0.8 |
| 5 | `candle_bullish_and_chase_guard` | 현재봉이 양봉인지 + 추격매수 방지 2건(인트라바 급등 차단, BB중간선 대비 가격 갭 과도 차단 — uptrend_continuation 경로는 RSI/갭 별도 기준 적용) | `CANDLE_GAIN_MIN_PCT`=-0.1%, `CANDLE_GAIN_MAX_PCT`=0.8%, `BB_MID_CHASE_MAX_GAP_PCT`=0.35%, `UPTREND_CONT_CHASE_MAX_GAP_PCT`=0.6%, `UPTREND_CONT_CHASE_RSI_MAX`=75.0 |
| 6 | `bb_upper_gap_min` | BB 상단까지 충분한 상승 여력(공간) 확보 | `BB_UPPER_GAP_MIN_PCT`=0.25% |
| 7 | `stochastic_buy_signal` | 스토캐스틱 매수신호: %K가 %D를 상향 돌파, 또는 %K>%D이면서 진짜 과열 구간이 아님 | `STOCH_BUY_MIN`=20.0, `STOCH_OVERBOUGHT`(프로파일)=96.0 |
| 8 | `williams_r_buy_signal` | 윌리엄스 %R 매수신호: 상승 중이면서 바닥권 탈출~과열 직전 구간 | `WILLIAMS_BUY_FLOOR`=-70.0, `WILLIAMS_OVERBOUGHT_CEIL`=-10 |
| 9 | `min_liquidity_safety` | 저유동성 종목 차단 (거래량/거래량이동평균 절대치 및 비율 최소치) | `MIN_ENTRY_VOL_MA`=1,000, `MIN_ENTRY_VOLUME`=1,500 |

### 1-2. 가점항목 (BUY_SCORE_RULES) — 필수조건 9개 통과 후에만 합산, 실제 최대 20점

| 항목명 | 배점 | 설명 | 관련 define (현재값) |
|--------|------|------|----------------------|
| `rsi_band` | 최대 2 | RSI 구간 점수 (50~65=2점, 45~50 또는 65~70=1점) | - |
| `ema_trend_align` | 2 | 장기 추세 정합성: EMA20 > EMA60(3분봉 기준) → 상위 추세 우상향 | `EMA_TREND_ALIGN_SCORE`=2 |
| `volume_ratio` | 최대 3 | 거래량/VOL_MA20 비율 (>=2.0배=3점, >=1.5배=2점, >=1.2배 또는 >=0.7배=1점) | - |
| `adx_strength` | 최대 3 | ADX 추세 강도 (>=35=3점, >=30=2점, >=25=1점) | - |
| `vwap_position` | 최대 2 | VWAP 대비 현재가 위치 (VWAP*1.002 초과=2점, VWAP 초과=1점) — 기관 평균매수가 상회 여부 | - |
| `volume_up_direction` | 1 | 거래량 증가 방향: 현재봉 거래량 > 직전봉 거래량 | - |
| `bb_width_expansion` | 1 | BB 밴드폭 확장: 스퀴즈 해소 → 추세 발생 초기 신호 | - |
| `ma5_short_term_up` | 1 | MA5 단기 상승 (MA5[t] > MA5[t-1]) | - |
| `macd_golden_cross` | 2 | MACD 골든크로스 (MACD > MACD_SIGNAL) | - |
| `bb_mid_slope_strength` | 최대 3 | BB 중앙선 기울기 강도 (>=1.5%=3점, >=1.0%=2점, >=0.5%=1점) | - |

**합산 후 처리 (점수 계산 이후에만 판단 가능해 게이트 리스트가 아닌 별도 로직으로 처리):**
- **개장가드**: 09:00~09:14(개장 첫 `OPENING_GUARD_MINUTES`=15분) 동안은 실시간 돌파(live_cross_up)가 아닌 진입에 한해 점수가 `OPENING_GUARD_SCORE_THRESHOLD`=12점 이상이어야 통과 (평상시보다 강화된 기준).
- **최종 점수 임계값**: 위 개장가드 통과 후, 최종적으로 점수가 `BB_BUY_SCORE_THRESHOLD`(config.bb_buy_score_threshold)=10점 이상이어야 매수 확정.

### 1-3. 1분봉 진입 타이밍 재검증 (선택적 2차 게이트)

`ENABLE_1MIN_ENTRY_SCORE_GATE`=True 일 때, 위 3분봉 매수 신호 통과 후 1분봉에서 한 번 더 타이밍을 검증한다 (`check_entry_condition_1min` / `_entry_score_1min`, 필수조건 없이 아래 4개를 점수화 — 만점 7점 중 `ENTRY_SCORE_THRESHOLD`=5점 이상이면 통과):

| 항목 | 배점 | 설명 |
|------|------|------|
| EMA 골든크로스 | `ENTRY_EMA_CROSS_SCORE`=2 | EMA9 > EMA20 (1분봉) |
| 종가 > EMA9 | `ENTRY_CLOSE_ABOVE_EMA9_SCORE`=1 | Close > EMA9 |
| 직전고점 돌파 | `ENTRY_PREV_HIGH_BREAKOUT_SCORE`=2 | Close > 직전 `ENTRY_PREV_HIGH_LOOKBACK_BARS`=10봉 고점 |
| 거래량 우위 | `ENTRY_VOLUME_ABOVE_MA_SCORE`=2 | Volume > VOL_MA20 (1분봉) |

---

## 2. 매도 조건

매도는 `check_sell_condition()` 하나에서 판정하며, 두 개의 서로 다른 경로로 나뉜다: **급락 크로스 즉시청산 경로**와 **보조 반전(가점) 경로**. 두 경로 모두 진입 전 "박스권 보유 유지" 억제 조건이 먼저 적용된다.

### 2-1. 박스권 보유 유지 억제 (매도 억제)

`ENABLE_BOX_RANGE_HOLD_TECH_SELL`=True 이고 손익률이 `STOP_LOSS_PERCENT`(-2.0%) ~ `TAKE_PROFIT_PERCENT`(+2.5%) 구간(즉 손절/익절 확정 구간이 아닐 때)이면, 최근 `BOX_RANGE_HOLD_LOOKBACK_BARS`=8봉의 가격 변동폭이 `BOX_RANGE_HOLD_MAX_RANGE_PCT`=0.65% 이내이고 BB 밴드폭이 `BOX_RANGE_HOLD_MAX_BB_WIDTH_PCT`=0.8% 이내(=박스권/횡보)이면, 아래 매도 신호들을 모두 무시하고 보유를 유지한다 (`BOX_RANGE_HOLD_*` 사유로 REJECT).

### 2-2. 가격 급락 크로스 즉시청산 경로

BB 중앙선 하향 크로스가 확정(`cross_info.signal == "cross_down"`)되고 현재가가 BB중앙선 아래일 때 진입하는 경로:

- 손익률이 `MA5_BB_DOWN_CROSS_MIN_PNL`=0.0%(본전) 미만이면 매도하지 않음 (손실 확정 방지, `LIVE_PRICE_BB_DOWN_CROSS_BLOCKED_PNL`).
- 그 외에는 `_sell_support_score`(아래 2-4) 점수를 계산해서, 손익률이 `MA5_BB_DOWN_CROSS_IMMEDIATE_PNL`=-0.7% 이하이거나 점수가 `MA5_BB_DOWN_CROSS_IMMEDIATE_SCORE`=2점 이상이면 즉시 매도 확정.
- 위 조건을 만족하지 못하면 매도 보류 (`LIVE_PRICE_BB_DOWN_CROSS_WEAK_SCORE`).

### 2-3. 보조 반전(가점) 경로 — 하향 크로스가 없을 때

`_sell_support_score`(아래 2-4) 점수 구간별로 요구하는 최소 손익률이 다르다:

| 점수 | 요구 최소 손익률 (매도 확정 조건) |
|------|-----------------------------------|
| >= 4 | `AUX_SELL_MIN_PNL_SCORE4`=0.3% 이상 |
| == 3 | `AUX_SELL_MIN_PNL_SCORE3`=0.8% 이상 |
| == 2 | `AUX_SELL_MIN_PNL_SCORE2`=1.5% 이상 |
| <= 1 | 매도 신호 없음 (`NO_SELL_SIGNAL`) |

### 2-4. 매도 가점항목 (`_sell_support_score`, 최대 5점 — 객체화 대상 아님, 5개 항목 단순 합산)

| 항목 | 배점 | 설명 |
|------|------|------|
| 스토캐스틱 데드크로스 | 1 | %K < %D |
| RSI 하락 방향 | 1 | RSI[t] < RSI[t-1] — 빠른 모멘텀 약화 감지 |
| VWAP 이탈 | 1 | 현재가 < VWAP — 기관 매도 압력 |
| MACD 데드크로스 | 1 | 직전봉 MACD>=SIGNAL이었다가 현재봉에서 MACD<SIGNAL로 전환 |
| OBV 하락 확인 | 1 | OBV < OBV_MA 이고 OBV[t] < OBV[t-1] — 거래량 흐름상 매도세 우위 |

---

## 참고

- 매수 조건 객체화(코드 구조): `live_trading/r002_strategy_core_shared.py`의 `BuyGateCondition`/`BuyScoreRule`/`BuyEvalContext` — 조건 추가/삭제/순서변경은 `BUY_GATE_CONDITIONS`/`BUY_SCORE_RULES` 리스트만 수정하면 됨.
- 매도 조건은 이번 작업에서 객체화하지 않음 (사용자 요청 범위 = 매수만).
- `r003_trade_live_execute.py`(실거래)와 `g003_trade_simulate_by_date.py`(백테스트)는 로그 표시 전용의 `_buy_support_score` 사본을 각자 별도로 갖고 있음 — 실제 매수판정과 무관하며 과거 실제 로직과 어긋난 이력이 있어 드리프트 위험 존재 (알려진 후속 과제).
