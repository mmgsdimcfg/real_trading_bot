/*
 * Update log
 * - [2026-11-11] type=test - Korean wording of reason codes (shared/reasonKo): triple-cross tokens with numbers, chase guards,
 *   market regime, buy signal labels, unknown tokens kept
 */

import { describe, expect, it } from 'vitest';
import { reasonKo, reasonTokenKo } from '../src/shared/reasonKo';

describe('reasonKo', () => {
  it('translates triple-cross rejections token by token', () => {
    expect(reasonKo('TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_NOT_ABOVE|TICK60_MA5_NOT_ABOVE_MA20|TICK120_MA5_NOT_ABOVE_MA20')).toBe(
      '1분봉 MA5가 MA20 아래 · 60틱 MA5가 MA20 아래 · 120틱 MA5가 MA20 아래',
    );
    expect(reasonKo('TRIPLE_CROSS_MISSING_TICK_ABSTAIN_NO_FILE')).toBe('틱 파일 없음 (수집기가 이 종목을 구독하지 않음)');
    expect(reasonKo('TRIPLE_CROSS_MISSING_TICK60_ABSTAIN_FEW_BARS_3')).toBe('60틱봉 부족 3개 (판정 보류)');
    expect(reasonKo('TRIPLE_CROSS_MISSING_TICK_ABSTAIN_FEED_STALE_120s|TICK120_CROSS_EXPIRED_900s_cross=10:12')).toBe(
      '틱 수신 끊김 120초 (판정 보류) · 120틱 골든크로스 유효시간 지남 (900초 전 · 10:12)',
    );
  });
  it('translates the chase, regime and context guards with their numbers', () => {
    expect(reasonKo('TRIPLE_CROSS_CHASE_FROM_CROSS_0.86%_GT_0.50%')).toBe('크로스 시점보다 0.86% 올라 추격 매수 금지 (한도 0.50%)');
    expect(reasonKo('MARKET_REGIME_DOWN_KOSPI_069500_1MIN_MA5_MA20_NOT_ABOVE')).toBe('시장 하락 국면: KOSPI(069500) 1분봉 MA5가 MA20 아래');
    expect(reasonKo('HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.42%_LT_0.60%')).toBe('3분봉 변동성(ATR) 0.42% < 0.60% (너무 낮음)');
    expect(reasonKo('TRIPLE_CROSS_3MIN_CHASE_BUY_BB_GAP_1.2%_GT_1.0%')).toBe('3분봉 볼린저 중심선 대비 1.2% 위 (한도 1.0%, 추격 매수 금지)');
    expect(reasonKo('can_trade_code_now=False')).toBe('지금은 이 종목 매매 불가 (시간대·종목 상태)');
  });
  it('translates buy signal labels and keeps unknown tokens', () => {
    expect(reasonKo('TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20+TICK60_GOLDEN_CROSS+HYBRID_3MIN_CTX_SCORE_15')).toBe('1분봉 MA5 > MA20 · 60틱 골든크로스 · 3분 맥락 점수 15');
    expect(reasonTokenKo('SOMETHING_NEW_42')).toBe('SOMETHING_NEW_42');
    expect(reasonKo('')).toBe('');
    expect(reasonKo('reason=WAIT_NEXT_POLL_CONFIRM | count=1/2 | live=234,500 bb_mid=232500.0')).toBe('다음 폴링에서 재확인 대기 · 확인 1/2');
  });
});
