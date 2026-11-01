# 매수 게이트 조정안 (2026-11-02)

기준 설정: **new** = 손실 매도 3조건 OFF (ENABLE_HYBRID_1MIN_DEADCROSS_EXIT / ENABLE_SIGNAL_EXIT_STOCH_K_LT_D /
ENABLE_SIGNAL_EXIT_MACD_HIST_DOWN = False). 매도 쪽(r006)과 매수 게이트(r002)는 진입 트리거 모드(ENTRY_TRIGGER_MODE,
TICK_TRIGGER_MODE)와 무관하게 공통으로 적용되므로 라이브 3중(1분+60틱+120틱) 트리거에도 그대로 적용된다.
라이브 반영은 사용자가 r003을 재시작해야 한다.

## 근거 1 - g003 A/B (runtime/sim_oct_ab, 10/01·10/02·10/06·10/07, 49종목)

g003은 레거시 1분 트리거로 진입한다(틱 데이터가 10/30부터만 있어 3중 트리거는 재현 불가). 로그 pnl은 비용 전이므로
매도 금액 기준 왕복 비용을 차감해 비교했다.

| 시나리오 | 매도 | 승률 | 비용 전 | 비용 0.25% | 비용 0.45% |
|---|---|---|---|---|---|
| base (매도 3조건 ON) | 51 | 53% | +47,347 | -8,422 | -53,038 |
| **new (매도 3조건 OFF)** | 50 | 78% | +124,411 | **+74,718** | **+34,964** |
| nogate (new + _013 스토캐스틱/_014 ATR 게이트 OFF) | 225 | 61% | +210,242 | +8,889 | -152,193 |
| 실매매 같은 날짜 (10/01 로그 없음) | 104 | 19% | -48,114 (실현) | | |

- new는 base보다 일관되게 좋다. base 손실의 대부분인 HYBRID_1MIN_DEAD_CROSS_LOSS(22건 -58,753원)가 사라지고
  AUX_REVERSAL / TP1 / 트레일링으로 청산된다.
- nogate가 new 대비 추가로 잡은 종목의 매도 150건: 비용 0.25%에서 -57,560원, 0.45%에서 -160,192원.
  이 중 ATR_STOP_LOSS 49건(비용 0.25% 기준 -188,265원)이 손실을 주도한다. 일별 편차도 크다(+73.8k ~ -36.7k).
  -> **_013/_014 게이트는 정책 시뮬레이션에서 가치가 있다.**

## 근거 2 - g021 게이트 값 스캔 (runtime/edge_ai/gate_scan/report_20261101_001817.md)

82일 ma_cross 이벤트 118,774건, 게이트별 통과/실패 그룹의 사후 수익 비교(고정 청산 규칙, 실제 정책과 다름).

| 게이트 | 판정 |
|---|---|
| bb_mid_downtrend_block, di_spread_min | KEEP (train/holdout 모두 실패 그룹이 더 나쁨) |
| bb_upper_gap_min, min_entry_atr_volatility | INVERTED (실패 그룹이 더 좋음) |
| bb_slope_rising, stochastic, min_liquidity, score_ok | MIXED |

모든 그룹이 비용 후 마이너스이고 차이는 1건당 수백 원 수준이다. min_entry_atr_volatility(_014)는 g021에서 INVERTED지만
근거 1(실제 매도 로직 포함)에서는 끄면 손실이 커졌다. 청산 로직까지 포함한 근거 1을 우선한다.

## 조정안

1. **매도 3조건 OFF(new) 유지** - 이미 코드 반영, r003 재시작 후 라이브 적용.
2. **_013 스토캐스틱(ENABLE_STOCHASTIC_BUY_GATE) / _014 ATR(ENABLE_MIN_ENTRY_ATR_FILTER) 게이트 ON 유지.**
   킬스위치는 남겨 두고 값은 바꾸지 않는다.
3. **bb_mid_downtrend_block, di_spread_min 유지.**
4. **bb_upper_gap_min 완화는 보류** - g021 INVERTED지만 정책 A/B가 없다. 다음 실험 후보:
   g003에 BB_UPPER_GAP 게이트 환경변수 오버라이드를 추가해 new 대비 단독 OFF A/B를 같은 4일로 실행.
5. MIXED 게이트는 변경하지 않는다.
6. 재시작 후 5거래일 동안 라이브(3중 트리거) 매도 사유별 손익을 집계해 new 시뮬레이션과 비교한다.
   특히 데드크로스 매도가 사라진 대신 ATR_STOP / LIVE_PRICE_BB_DOWN_CROSS 손실이 커지는지 확인한다.

## 한계

- 4거래일, 49종목 표본. 10/01 하루가 nogate 비용 전 수익의 2/3을 차지한다.
- 레거시 1분 트리거 진입이므로 라이브 3중 트리거의 진입 빈도/품질과 다를 수 있다.
- 10월 실매매 사후 분석(게이트가 막은 후보의 15:15 수익률이 더 좋음)과 근거 1이 상충한다. 사후 분석은 청산 규칙 없이
  종가 보유 수익만 본 것이라 근거 1을 우선했다.
