/*
 * Update log
 * - [2026-11-27] type=test - Per-condition applied / not applied of the rule trade (server r005.buy_condition_state): what the settings
 *   window claims from a condition row (condApplyInfo) and the BUY_CONDITION_OFF list it sends (nextBuyConditionOff)
 */
import { describe, expect, it } from 'vitest';
import { BUY_CONDITION_OFF_KEY, BUY_CONDITION_SWITCH_KEY, buyConditionSwitch, COND_APPLY_KO, condApplyInfo, nextBuyConditionOff } from '../src/shared/control';

describe('condApplyInfo', () => {
  it('claims nothing for a row of an older server', () => {
    const a = condApplyInfo({});
    expect([a.stored, a.running, a.switchable, a.restartPending]).toEqual([null, null, false, false]);
  });

  it('a required condition: applied, not switchable, with its reason', () => {
    const a = condApplyInfo({ apply: 'on', apply_running: 'on', off_by: [], switchable: false, locked_reason: '매수 신호 자체' });
    expect([a.stored, a.switchable, a.lockedReason, a.restartPending]).toEqual(['on', false, '매수 신호 자체', false]);
  });

  it('tells "switched off as a whole" from "its own settings make it pass"', () => {
    const list = condApplyInfo({ apply: 'off', off_by: [BUY_CONDITION_OFF_KEY], switchable: true });
    expect([list.offByList, list.offBySettings]).toEqual([true, []]);
    const own = condApplyInfo({ apply: 'off', off_by: ['ENABLE_DAILY_LOSS_LIMIT_GATE=False'], switchable: true });
    expect([own.offByList, own.offBySettings]).toEqual([false, ['ENABLE_DAILY_LOSS_LIMIT_GATE=False']]);
    const both = condApplyInfo({ apply: 'off', off_by: [BUY_CONDITION_OFF_KEY, 'EDGE_AI_MODE=off'], switchable: true });
    expect([both.offByList, both.offBySettings]).toEqual([true, ['EDGE_AI_MODE=off']]);
  });

  it('the running bot can differ from the stored settings until a restart', () => {
    const a = condApplyInfo({ apply: 'off', apply_running: 'on', off_by: [BUY_CONDITION_OFF_KEY], switchable: true });
    expect(a.restartPending).toBe(true);
    // a bot that is not running / an older r003: unknown, never "pending"
    expect(condApplyInfo({ apply: 'off', apply_running: null, switchable: true }).restartPending).toBe(false);
    expect(condApplyInfo({ apply: 'record', apply_running: 'record', switchable: false }).restartPending).toBe(false);
  });

  it('has a Korean word for every state', () => {
    expect(COND_APPLY_KO).toEqual({ on: '적용', off: '미적용', record: '기록만' });
  });
});

describe('nextBuyConditionOff', () => {
  it('adds / removes one number, sorted, without duplicates', () => {
    expect(nextBuyConditionOff([20], 6, false)).toEqual([6, 20]);
    expect(nextBuyConditionOff([6, 20], 6, false)).toEqual([6, 20]);
    expect(nextBuyConditionOff([6, 20], 20, true)).toEqual([6]);
    expect(nextBuyConditionOff([20], 20, true)).toEqual([]);
    expect(nextBuyConditionOff(undefined, 22, false)).toEqual([22]);
    expect(nextBuyConditionOff([20, 20, 3], 3, true)).toEqual([20]);
  });
});

describe('display defence and the one-number switch (Codex round 3)', () => {
  it('a row that says applied but nothing about switchable is neither locked nor editable', () => {
    const a = condApplyInfo({ apply: 'on' });
    expect([a.locked, a.switchable, a.editable]).toEqual([false, false, false]);
  });

  it('only a row of the live server answer (it carries apply_running) can be edited', () => {
    expect(condApplyInfo({ apply: 'on', switchable: true }).editable).toBe(false); // a cached / bundled list
    expect(condApplyInfo({ apply: 'on', switchable: true, apply_running: null }).editable).toBe(true);
    const locked = condApplyInfo({ apply: 'on', switchable: false, apply_running: 'on', locked_reason: 'x' });
    expect([locked.locked, locked.editable]).toEqual([true, false]);
  });

  it('sends one number, never the whole list', () => {
    expect(buyConditionSwitch(22, false)).toEqual({ [BUY_CONDITION_SWITCH_KEY]: { no: 22, apply: false } });
    expect(Object.keys(buyConditionSwitch(20, true))).toEqual(['BUY_CONDITION_SWITCH']);
  });
});
