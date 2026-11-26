// [2026-11-27] vwapHoldBars / vwapHoldLabel: how long the close has stayed on one side of VWAP (shown in the rebound alert only)
import { describe, expect, it } from 'vitest';
import { detectRebound, vwapHoldBars, vwapHoldLabel } from '../src/shared/rebound';
import type { Bar } from '../src/shared/price';

const KST9 = Date.UTC(2026, 10, 26, 0, 0, 0); // 2026-11-26 09:00 KST
const mk = (closes: number[], t0 = KST9): Bar[] => closes.map((c, k) => ({ t: t0 + k * 60000, o: c, h: c, l: c, c, v: 100 }));

describe('vwapHoldBars', () => {
  it('counts the bars in a row on the side of the last bar', () => {
    const bars = mk([100, 100, 99, 98, 101, 102, 103]);
    const V = [100, 100, 100, 100, 100, 100, 100];
    expect(vwapHoldBars(bars, V)).toBe(3); // 101, 102, 103 above
    expect(vwapHoldBars(bars, V, 3)).toBe(-2); // 99, 98 below
    expect(vwapHoldBars(bars, V, 1)).toBe(0); // on the line
    expect(vwapHoldBars(bars, V, 4)).toBe(1); // the bar that crossed
  });

  it('stops at a bar without VWAP, at the day change, and is 0 outside the bars', () => {
    const bars = mk([101, 102, 103, 104]);
    expect(vwapHoldBars(bars, [100, NaN, 100, 100])).toBe(2);
    expect(vwapHoldBars(bars, [100, 100, 100, NaN])).toBe(0);
    const two = [...mk([105, 106], KST9 - 86400000 + 6 * 3600000), ...mk([101, 102, 103])]; // yesterday 15:00, 15:01 + today
    expect(vwapHoldBars(two, [100, 100, 100, 100, 100])).toBe(3);
    expect(vwapHoldBars([], [])).toBe(0);
    expect(vwapHoldBars(bars, [100, 100, 100, 100], 9)).toBe(0);
  });

  it('is part of the snapshot and has a short label', () => {
    const r = detectRebound(mk([100, 101, 102, 103, 104]));
    expect(r.snap.vwapHold).toBe(4); // the first bar's close is its own VWAP
    expect(vwapHoldLabel(23)).toBe('위 23분째');
    expect(vwapHoldLabel(-5)).toBe('아래 5분째');
    expect(vwapHoldLabel(0)).toBe('');
    expect(vwapHoldLabel(undefined)).toBe('');
    expect(vwapHoldLabel(NaN)).toBe('');
  });
});
