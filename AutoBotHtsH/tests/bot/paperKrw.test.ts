/*
 * Update log
 * - [2026-11-20] type=test - 가상 매매 수익금: shares the bot would buy with its per-order budget, open = (price - buy) x qty, closed = net %
 *   of the bought amount, a price above the budget = no amount
 * - [2026-11-20] type=test - 매수 / 매도 totals of the closed trades (open trades left out, no exit price = buy + net)
 */

import { describe, expect, it } from 'vitest';
import { paperKrw, paperQty, paperTotals } from '../../src/bot/renderer/src/views/PaperView';

describe('paper trade KRW', () => {
  it('sizes like the bot and computes the amount', () => {
    expect(paperQty(9100, 500_000)).toBe(54);
    expect(paperKrw(9100, 500_000, { price: 9300 })).toBe(54 * 200);
    expect(paperKrw(10_000, 500_000, { netPct: 1.5 })).toBeCloseTo(7500);
    expect(paperKrw(10_000, 500_000, { netPct: -2 })).toBeCloseTo(-10_000);
    expect(paperKrw(600_000, 500_000, { price: 610_000 })).toBeUndefined();
    expect(paperKrw(9100, 500_000, {})).toBeUndefined();
  });
});

describe('paper trade totals', () => {
  it('sums buy and sell amounts of closed trades', () => {
    const base = { group: 'watch' as const, name: 'x', entryTs: 0 };
    const t = paperTotals(
      [
        { ...base, code: '000001', entry: 10_000, exitTs: 1, exit: 10_200, netPct: 1.8 },
        { ...base, code: '000002', entry: 20_000, exitTs: 2, netPct: -1 },
        { ...base, code: '000003', entry: 5_000 },
      ],
      500_000,
    );
    expect(t.buy).toBe(50 * 10_000 + 25 * 20_000);
    expect(t.sell).toBeCloseTo(50 * 10_200 + 25 * 20_000 * 0.99);
  });
});
