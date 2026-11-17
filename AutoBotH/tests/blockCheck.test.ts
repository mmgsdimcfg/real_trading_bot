/*
 * Update log
 * - [2026-11-16] type=test - explainBlock: real reasons from the 2026-11-16 log (triple trigger with three failed parts), comparison
 *   reasons with limits in the text, limits taken from the settings, and the fallback key match
 * - [2026-11-16] type=test - MAVAL segment (MA values of 1m / tick series) and the full reason read from the raw line
 */
import { describe, expect, it } from 'vitest';
import { blockedKeyNotes, explainBlock, fullReason, parseMaValues } from '../src/shared/blockCheck';

const SETTINGS: Record<string, unknown> = {
  TRIPLE_1MIN_ABOVE_MAX_AGE_SECONDS: 900,
  TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS: 240,
  TICK_TRIGGER_CROSS_MAX_AGE_SECONDS: 240,
  TICK_TRIGGER_MA_SLOW: 20,
  SOME_LIMIT_PCT: 1.5,
};
const v = (k: string) => SETTINGS[k];

describe('explainBlock', () => {
  it('splits the triple trigger reason into its failed checks', () => {
    const r = explainBlock('TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_ABOVE_TOO_OLD_1428s_GT_900s|TICK60_MA5_NOT_ABOVE_MA20|TICK120_MA5_NOT_ABOVE_MA20', v);
    expect(r).toHaveLength(3);
    expect(r[0]).toMatchObject({ actual: '1,428초 (23.8분)', limit: '≤ 900초 (15.0분)' });
    expect(r[0].keys[0]).toBe('TRIPLE_1MIN_ABOVE_MAX_AGE_SECONDS');
    expect(r[1].label).toBe('60틱봉 MA5가 MA20 아래');
    expect(r[1].keys).toContain('TICK_TRIGGER_MA_FAST');
    expect(r[2].label).toBe('120틱봉 MA5가 MA20 아래');
  });

  it('reads comparison reasons and maps them to their keys', () => {
    const [a] = explainBlock('TRIPLE_CROSS_1MIN_LIVE_DROP_0.55%_GT_0.40%', v);
    expect(a).toMatchObject({ actual: '0.55%', limit: '≤ 0.4%', keys: ['HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT'] });
    const [b] = explainBlock('TRIPLE_CROSS_1MIN_CANDLE_NOT_BULLISH_-0.52%_LT_-0.4%', v);
    expect(b).toMatchObject({ actual: '-0.52%', limit: '≥ -0.4%', keys: ['HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT'] });
    const [c] = explainBlock('TRIPLE_CROSS_MISSING_TICK_LOW_DENSITY_9/min_LT_15', v);
    expect(c).toMatchObject({ actual: '9회/분', limit: '≥ 15회/분', keys: ['TICK_MIN_PRINTS_PER_MIN', 'TICK_MIN_PRINTS_WINDOW_MINUTES'] });
    const [d] = explainBlock('GC_VOL_LOW_1MIN_ATR_0.312%_LT_0.500%', v);
    expect(d.keys[0]).toBe('GC_VOLATILITY_MIN_ATR_1MIN_PCT');
  });

  it('takes a limit the reason does not carry from the settings', () => {
    const [a] = explainBlock('TRIPLE_CROSS_MISSING_TICK60_CROSS_EXPIRED_300s', v);
    expect(a).toMatchObject({ actual: '300초 (5.0분)', limit: '≤ 240초 (4.0분)' });
    const [b] = explainBlock('TRIPLE_CROSS_MISSING_TICK120_ABSTAIN_FEW_BARS_12', v);
    expect(b).toMatchObject({ actual: '12봉', limit: '≥ 21봉' });
  });

  it('unknown comparison: key whose value equals the limit; unknown text stays as is', () => {
    const [a] = explainBlock('SOMETHING_ODD_2.10%_GT_1.50%', v, ['OTHER', 'SOME_LIMIT_PCT']);
    expect(a.keys).toEqual(['SOME_LIMIT_PCT']);
    const [b] = explainBlock('WEIRD_REASON', v);
    expect(b).toMatchObject({ label: 'WEIRD_REASON', keys: [] });
  });

  it('raw line: full reason and MA values', () => {
    const raw = '2026-11-16 09:02:09,836 [INFO]   [REJECT      ] 069960_현대백화점 | TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_ABOVE_TOO_OLD_1428s_GT_900s'
      + '|TICK60_MA5_NOT_ABOVE_MA20|TICK120_MA5_NOT_ABOVE_MA20 | GATES cross_pass=True live=86,500 | STEPS 1min_trigger(F) '
      + '| MAVAL 1m(5=86420.00,20=86100.50) t60(5=86380.00,20=86450.00,bars=45) t120(5=86300.00,20=86410.20,bars=30)';
    const first = 'TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_ABOVE_TOO_OLD_1428s_GT_900s'; // what the app keeps as the event reason
    expect(fullReason(first, raw)).toBe('TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_ABOVE_TOO_OLD_1428s_GT_900s|TICK60_MA5_NOT_ABOVE_MA20|TICK120_MA5_NOT_ABOVE_MA20');
    expect(parseMaValues(raw).t60).toEqual({ fast: 5, f: 86380, slow: 20, s: 86450, bars: 45 });
    const r = explainBlock(first, v, [], raw);
    expect(r).toHaveLength(3);
    expect(r[1].actual).toBe('MA5 86,380 / MA20 86,450 (-0.08%) · 45봉');
    expect(r[2].actual).toMatch(/^MA5 86,300 \/ MA20 86,410.2 \(-0.13%\) · 30봉$/);
    expect(explainBlock(first, v, [], 'no maval here')).toHaveLength(1); // reason not in raw: as given
  });

  it('notes: the threshold key gets the comparison, related keys a pointer', () => {
    const n = blockedKeyNotes(explainBlock('TRIPLE_CROSS_MISSING_1MIN_MA5_MA20_ABOVE_TOO_OLD_1428s_GT_900s', v));
    expect(n.TRIPLE_1MIN_ABOVE_MAX_AGE_SECONDS).toMatch(/^막힘: .* 현재 1,428초 .* 기준 ≤ 900초/);
    expect(n.TRIPLE_1MIN_CONDITION).toMatch(/^관련: /);
  });
});

describe('explainBlock - keys of reasons without a fallback guess (2026-11-17)', () => {
  const S: Record<string, unknown> = { GC_VOLATILITY_MIN_ATR_1MIN_PCT: 0.4, RELAXED_ENTRY_ATR_MIN_PCT: 0.004, MIN_ENTRY_ATR_TO_TP1_RATIO: 0.2, STAGED_TP1_PCT: 0.03 };
  const val = (k: string) => S[k];
  const cand = ['ENABLE_GC_VOLATILITY_GATE', 'GC_VOLATILITY_MIN_ATR_1MIN_PCT', 'MIN_ENTRY_ATR_TO_TP1_RATIO'];

  it('3-minute ATR floor: the relaxed floor, not the 1-minute key that happens to equal 0.4', () => {
    const [a] = explainBlock('HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.38%_LT_0.40%', val, cand);
    expect(a).toMatchObject({ actual: '0.38%', limit: '≥ 0.4%' });
    expect(a.keys[0]).toBe('RELAXED_ENTRY_ATR_MIN_PCT');
    expect(a.keys).not.toContain('GC_VOLATILITY_MIN_ATR_1MIN_PCT');
    const [b] = explainBlock('HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.45%_LT_0.60%', val, cand);
    expect(b.keys[0]).toBe('MIN_ENTRY_ATR_TO_TP1_RATIO');
  });

  it('liquidity, Edge AI and aux sell score', () => {
    expect(explainBlock('HYBRID_3MIN_CTX_LOW_VOL_MA_ABS_946_LT_1000', val)[0].keys[0]).toBe('MIN_ENTRY_VOL_MA');
    expect(explainBlock('HYBRID_3MIN_CTX_LOW_ABS_VOLUME_1266_LT_1500', val)[0].keys[0]).toBe('MIN_ENTRY_VOLUME');
    const [e] = explainBlock('EDGE_AI_LOW_PROB_p=0.293_LT_0.310', val);
    expect(e).toMatchObject({ actual: '0.293', limit: '≥ 0.31' });
    expect(e.keys[0]).toBe('EDGE_AI_MIN_PROB_OVERRIDE');
    expect(explainBlock('AUX_BLOCKED_SCORE_3_PNL_-0.62%_LT_0.80%', val)[0].keys[0]).toBe('AUX_SELL_MIN_PNL_SCORE3');
  });
});
