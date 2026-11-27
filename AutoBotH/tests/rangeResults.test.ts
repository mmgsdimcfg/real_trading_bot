/*
 * Update log
 * - [2026-11-27] type=test - shared/rangeResults: a day's results per method and mode, their sum over days, week / month ranges
 */
import { describe, expect, it } from 'vitest';
import { addDays, dayResults, isYmd, periodRange, shiftPeriod, sumResults } from '../src/shared/rangeResults';
import type { PaperTrade } from '../src/shared/paper';
import type { Trade } from '../src/shared/types';

const sell = (strategy: Trade['strategy'], pnlKrw: number, buyPrice = 10000, price = 10000 + pnlKrw / 10, qty = 10): Trade => ({ code: '000001', name: 'X', side: 'SELL', ts: 0, price, qty, reason: 'R', raw: '', pnlKrw, buyPrice, strategy });
const buy = (strategy: Trade['strategy']): Trade => ({ code: '000001', name: 'X', side: 'BUY', ts: 0, price: 10000, qty: 10, reason: '', raw: '', strategy });
const paper = (entry: number, netPct?: number): PaperTrade => ({ code: '000001', group: 'watch', session: 'regular', name: 'X', entryTs: 0, entry, ...(netPct === undefined ? {} : { exitTs: 1, netPct }) });

describe('results of a day and of a range', () => {
  it('real trades per method (gross and net of costs), paper per book sized with the budget', () => {
    const d = dayResults('20261127', [buy('rule'), sell('rule', 1000), buy('vwap'), sell('vwap', -500), sell('vwap', 100)], { edge: [paper(10000, 2), paper(10000, -1), paper(5000)], vwap: null }, 100_000);
    expect(d.real.rule).toMatchObject({ buys: 1, sells: 1, wins: 1, pnlKrw: 1000 });
    expect(d.real.rule.pnlNet).toBeLessThan(1000);
    expect(d.real.vwap).toMatchObject({ buys: 1, sells: 2, wins: 1, pnlKrw: -400 });
    // +100 gross on a 100,000 trade is a loss after commission and sell tax
    expect(d.real.vwap.winsNet).toBe(0);
    expect(d.real.edge).toMatchObject({ buys: 0, sells: 0, pnlKrw: 0 });
    // 10 shares x 10,000: +2% and -1% of 100,000
    expect(d.paper.edge).toEqual({ buys: 3, sells: 2, wins: 1, open: 1, pnlKrw: 1000 });
    expect(d.paper.vwap).toBeNull();
  });

  it('sums days; a book that could not be read on a day adds nothing; open = the last day', () => {
    const a = dayResults('20261126', [sell('rule', 1000)], { edge: [paper(10000, 2), paper(10000)], vwap: [] }, 100_000);
    const b = dayResults('20261127', [sell('rule', -300), sell('edge', 50)], { edge: null, vwap: [paper(10000, 1), paper(10000)] }, 100_000);
    const s = sumResults([a, b]);
    expect(s.real.rule).toMatchObject({ sells: 2, wins: 1, pnlKrw: 700 });
    expect(s.real.edge).toMatchObject({ sells: 1, pnlKrw: 50 });
    expect(s.paper.edge).toMatchObject({ buys: 2, sells: 1, pnlKrw: 2000, open: 0 });
    expect(s.paper.vwap).toMatchObject({ buys: 2, sells: 1, pnlKrw: 1000, open: 1 });
    expect(sumResults([]).real.vwap.sells).toBe(0);
  });
});

describe('periods', () => {
  it('week = Monday - Sunday, month = first - last day', () => {
    expect(periodRange('week', '20261127')).toEqual(['20261123', '20261129']); // a Friday
    expect(periodRange('week', '20261123')).toEqual(['20261123', '20261129']);
    expect(periodRange('week', '20261129')).toEqual(['20261123', '20261129']); // Sunday
    expect(periodRange('month', '20261127')).toEqual(['20261101', '20261130']);
    expect(periodRange('month', '20280215')).toEqual(['20280201', '20280229']);
  });
  it('moves to the previous / next week or month, across a year', () => {
    expect(shiftPeriod('week', '20261127', -1)).toBe('20261120');
    expect(periodRange('month', shiftPeriod('month', '20261231', 1))).toEqual(['20270101', '20270131']);
    expect(periodRange('month', shiftPeriod('month', '20270131', -1))).toEqual(['20261201', '20261231']);
    expect(addDays('20261231', 1)).toBe('20270101');
  });
  it('rejects what is not a real date', () => {
    expect(isYmd('20261127')).toBe(true);
    expect(isYmd('20261131')).toBe(false);
    expect(isYmd('2026-11-27')).toBe(false);
    expect(isYmd('')).toBe(false);
  });
});
