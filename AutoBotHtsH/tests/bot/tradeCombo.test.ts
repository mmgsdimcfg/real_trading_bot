/*
 * Update log
 * - [2026-11-25] type=test - 매매 방식 combination (규칙 매매 / Edge AI 가상 매매 / 반등 알림): RULE_BUY_ENABLED parsing, the set
 *   changes (never TRADING_STRATEGY_MODE, null on an older server for a rules change), the label
 */

import { describe, expect, it } from 'vitest';
import { comboChanges, comboLabel, parseRuleBuy } from '../../src/bot/shared/paper';

describe('trade combination', () => {
  it('RULE_BUY_ENABLED state; null on a server without it', () => {
    expect(parseRuleBuy({ values: { RULE_BUY_ENABLED: { effective: false, running: true, pending: true } } })).toEqual({ effective: 'false', running: 'true', pending: true });
    expect(parseRuleBuy({ values: {}, unknown: ['RULE_BUY_ENABLED'] })).toBeNull();
  });

  it('changes only what was switched, never the strategy', () => {
    expect(comboChanges({ rules: false, paper: true }, { rules: true, paper: true }, true)).toEqual({ RULE_BUY_ENABLED: false });
    expect(comboChanges({ rules: true, paper: false }, { rules: true, paper: true }, true)).toEqual({ EDGE_ONLY_MODE: 'off' });
    expect(comboChanges({ rules: true, paper: true }, { rules: true, paper: false }, false)).toEqual({ EDGE_ONLY_MODE: 'paper' });
    expect(comboChanges({ rules: false, paper: false }, { rules: true, paper: false }, false)).toBeNull();
    expect(comboChanges({ rules: true, paper: true }, { rules: true, paper: true }, true)).toBeNull();
  });

  it('label of the kinds that are on', () => {
    expect(comboLabel({ rules: true, paper: false, rebound: true })).toBe('규칙 매매 + 반등 알림');
    expect(comboLabel({ rules: false, paper: true, rebound: true }, ['rules', 'paper'])).toBe('Edge AI 가상 매매');
    expect(comboLabel({})).toBe('모두 꺼짐');
  });
});
