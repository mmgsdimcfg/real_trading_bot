/*
 * Update log
 * - [2026-11-11] type=test - Trading costs (shared/fees): round-trip costs, gross vs net P&L in KRW and percent, guards for
 *   missing prices / quantities
 */

import { describe, expect, it } from 'vitest';
import { FEE_COMMISSION_PCT, FEE_SELL_TAX_PCT, grossPnl, pnlKrw, pnlPct, tradeCosts } from '../src/shared/fees';

describe('fees', () => {
  it('matches the bot rates (r001 V2_COMMISSION_PCT / V2_SELL_TAX_PCT)', () => {
    expect(FEE_COMMISSION_PCT).toBe(0.015);
    expect(FEE_SELL_TAX_PCT).toBe(0.2);
  });
  it('round-trip costs: commission on both sides, tax on the sell', () => {
    // buy 10,000 x 10 = 100,000 -> 15; sell 10,200 x 10 = 102,000 -> 15.3 + 204
    expect(tradeCosts(10000, 10200, 10)).toBeCloseTo(15 + 15.3 + 204, 6);
    expect(tradeCosts(0, 10200, 10)).toBe(0);
    expect(tradeCosts(10000, 10200, 0)).toBe(0);
  });
  it('gross vs net P&L', () => {
    expect(grossPnl(10000, 10200, 10)).toBe(2000);
    expect(pnlKrw(10000, 10200, 10, false)).toBe(2000);
    expect(pnlKrw(10000, 10200, 10, true)).toBeCloseTo(2000 - 234.3, 6);
    expect(pnlPct(10000, 10200, 10, false)).toBeCloseTo(2, 9);
    expect(pnlPct(10000, 10200, 10, true)).toBeCloseTo(((2000 - 234.3) / 100000) * 100, 9);
    expect(Number.isNaN(pnlPct(0, 10200, 10, true))).toBe(true);
  });
  it('a small gain becomes a loss after costs', () => {
    expect(pnlKrw(10000, 10010, 10, false)).toBe(100);
    expect(pnlKrw(10000, 10010, 10, true)).toBeLessThan(0);
  });
});
