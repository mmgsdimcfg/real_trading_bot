/*
 * Update log
 * - [2026-11-23] type=test - Stock info window, investor table: the year of rows + the newer days of the short list, newest first
 *   (mergeInvestor) and the sums of the last N months (investorSum: window after `months` before the newest row, amounts NaN
 *   when no row has them, rows without numbers left out)
 */

import { describe, expect, it } from 'vitest';
import { cumLabel, investorSum, mergeInvestor } from '../src/shared/investor';

const day = (date: string, person: number, foreign: number, institution: number, amt?: number) =>
  ({ date, person, foreign, institution, close: 100, change: 1, ...(amt === undefined ? {} : { personAmt: amt, foreignAmt: amt * 2, institutionAmt: amt * 3 }) });

describe('investor table', () => {
  it('merge: the year wins, only newer days come from the short list, newest first', () => {
    const year = [day('20261119', 1, 2, 3), day('20261120', 4, 5, 6)];
    const recent = [{ ...day('20261123', 7, 8, 9), close: 110, change: 0 }, { ...day('20261120', 99, 99, 99), close: 1, change: 0 }];
    const m = mergeInvestor(year, recent);
    expect(m.map((r) => [r.date, r.person])).toEqual([['20261123', 7], ['20261120', 4], ['20261119', 1]]);
    expect(mergeInvestor(null, recent).map((r) => r.date)).toEqual(['20261123', '20261120']);
    expect(mergeInvestor(undefined, undefined)).toEqual([]);
  });
  it('sum: rows after N months before the newest row', () => {
    const rows = [day('20261123', 1, 10, 100, 1), day('20261026', 2, 20, 200, 2), day('20261023', 4, 40, 400, 4), day('20260824', 8, 80, 800, 8)];
    expect(investorSum(rows, 1)).toMatchObject({ from: '20261026', to: '20261123', days: 2, person: 3, foreign: 30, institution: 300, personAmt: 3, foreignAmt: 6, institutionAmt: 9 });
    expect(investorSum(rows, 3)).toMatchObject({ from: '20260824', days: 4, person: 15, foreignAmt: 30 });
    expect(investorSum(rows, 12)).toMatchObject({ days: 4 });
    expect(investorSum([], 1)).toBeNull();
  });
  it('sum: amounts are NaN when no row has them; a row without numbers (today before the close) is left out', () => {
    const s = investorSum([{ date: '20261123', person: NaN, foreign: NaN, institution: NaN }, day('20261120', 1, 2, 3)], 1)!;
    expect([s.days, s.person, s.to]).toEqual([1, 1, '20261120']);
    expect(Number.isNaN(s.foreignAmt)).toBe(true);
    expect([cumLabel(1), cumLabel(6), cumLabel(12)]).toEqual(['1달', '6달', '1년']);
  });
});
