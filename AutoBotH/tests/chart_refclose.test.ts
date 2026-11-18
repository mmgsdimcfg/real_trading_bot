import { describe, expect, it } from 'vitest';
import { barWhen, kstDayStart, refCloseAt } from '../src/renderer/src/components/PriceChart';
import { ymdEpoch, type Bar } from '../src/shared/price';

const at = (ymd: string, hh: number, mm: number) => ymdEpoch(ymd) + (hh * 60 + mm) * 60000;
const bar = (t: number, c: number): Bar => ({ t, o: c, h: c, l: c, c, v: 1 });

describe('chart hover reference close (2026-11-18)', () => {
  it('kstDayStart is 00:00 KST of the bar day', () => {
    expect(kstDayStart(at('20261118', 9, 5))).toBe(ymdEpoch('20261118'));
    expect(kstDayStart(at('20261118', 23, 59))).toBe(ymdEpoch('20261118'));
  });

  it('intraday: previous day close from prevCloseOf, else the last bar of the day before (bars, then warm)', () => {
    const warm = [bar(at('20261116', 15, 30), 900)];
    const bars = [bar(at('20261117', 9, 0), 1000), bar(at('20261117', 15, 30), 1100), bar(at('20261118', 9, 0), 1210), bar(at('20261118', 9, 1), 1220)];
    expect(refCloseAt(bars, 3, true, warm)).toBe(1100);
    expect(refCloseAt(bars, 0, true, warm)).toBe(900);
    expect(refCloseAt(bars, 3, true, warm, () => 1050)).toBe(1050);
    expect(refCloseAt(bars, 3, true, warm, () => undefined)).toBe(1100);
  });

  it('daily kinds: the previous bar', () => {
    const bars = [bar(ymdEpoch('20261117'), 100), bar(ymdEpoch('20261118'), 110)];
    expect(refCloseAt(bars, 1, false)).toBe(100);
    expect(refCloseAt(bars, 0, false)).toBeUndefined();
  });

  it('barWhen shows date and time for intraday bars', () => {
    expect(barWhen(bar(at('20261118', 9, 5), 1), 'min')).toBe('2026-11-18 09:05:00');
    expect(barWhen(bar(ymdEpoch('20261118'), 1), 'day')).toBe('2026-11-18');
    expect(barWhen({ ...bar(at('20261118', 9, 5), 1), t2: at('20261118', 9, 6) }, 'tick')).toBe('2026-11-18 09:05:00 ~ 09:06:00');
  });
});

describe('yearly bars (2026-11-18)', () => {
  it('groups daily bars per KST year and labels them', async () => {
    const { groupDaily, samePeriod } = await import('../src/shared/price');
    const d = (ymd: string, c: number): Bar => ({ t: ymdEpoch(ymd), o: c, h: c + 5, l: c - 5, c, v: 10 });
    const daily = [d('20241230', 100), d('20250102', 110), d('20251230', 130), d('20260102', 140), d('20261118', 150)];
    const y = groupDaily(daily, 'year');
    expect(y.map((b) => b.t)).toEqual([ymdEpoch('20241230'), ymdEpoch('20250102'), ymdEpoch('20260102')]);
    expect(y[1]).toMatchObject({ o: 110, c: 130, h: 135, l: 105, v: 20 });
    expect(samePeriod('year', y[2].t, '20261118')).toBe(true);
    expect(samePeriod('year', y[1].t, '20261118')).toBe(false);
    expect(barWhen(y[2], 'year')).toBe('2026년');
  });
});
