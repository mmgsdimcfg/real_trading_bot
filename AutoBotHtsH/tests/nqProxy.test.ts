import { describe, expect, it } from 'vitest';
import { retOverMinutes } from '../src/shared/market';

const T0 = Date.UTC(2026, 10, 26, 0, 0); // 09:00 KST
const bar = (min: number, c: number) => ({ t: T0 + min * 60_000, c });

describe('retOverMinutes (index window Nasdaq-100 ETF card)', () => {
  const bars = Array.from({ length: 41 }, (_, i) => bar(i, 100 + i)); // 09:00 .. 09:40
  const now = T0 + 40 * 60_000 + 30_000;
  it('compares the newest close with the close N minutes earlier', () => {
    expect(retOverMinutes(bars, 5, now)).toBeCloseTo((140 / 135 - 1) * 100, 6);
    expect(retOverMinutes(bars, 30, now)).toBeCloseTo((140 / 110 - 1) * 100, 6);
  });
  it('is empty before the day has that much history', () => {
    expect(retOverMinutes(bars.slice(0, 20), 30, T0 + 19 * 60_000)).toBeNull();
    expect(retOverMinutes([], 5, now)).toBeNull();
  });
  it('never uses bars of the day before', () => {
    const yesterday = Array.from({ length: 30 }, (_, i) => ({ t: T0 - 86_400_000 + (350 + i) * 60_000, c: 90 }));
    expect(retOverMinutes([...yesterday, ...bars.slice(0, 10)], 30, T0 + 9 * 60_000)).toBeNull();
  });
  it('is empty when the newest bar is stale or a bar is missing around the reference time', () => {
    expect(retOverMinutes(bars, 5, now + 20 * 60_000)).toBeNull();
    const gap = bars.filter((b) => b.t < T0 + 20 * 60_000 || b.t > T0 + 38 * 60_000); // 09:20..09:38 missing
    expect(retOverMinutes(gap, 5, now)).toBeNull();
  });
});
