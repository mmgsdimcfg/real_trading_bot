// [2026-11-23] rebound history minute chart: trading days to ask for and the split of the alert's day / warm-up bars
import { describe, expect, it } from 'vitest';
import { daysBack, kstDay0, splitDay } from '../src/renderer/src/components/ReboundPanels';

const kst = (s: string) => Date.parse(`${s}+09:00`);

describe('rebound history chart', () => {
  it('kstDay0 is the KST midnight of the day', () => {
    expect(kstDay0(kst('2026-11-23T14:24:00'))).toBe(kst('2026-11-23T00:00:00'));
    expect(kstDay0(kst('2026-11-23T00:30:00'))).toBe(kst('2026-11-23T00:00:00'));
  });
  it('daysBack counts weekdays from the alert day to today', () => {
    const now = kst('2026-11-23T18:00:00'); // Monday
    expect(daysBack(kst('2026-11-23T09:34:00'), now)).toBe(1);
    expect(daysBack(kst('2026-11-20T10:00:00'), now)).toBe(2); // Friday -> Monday
    expect(daysBack(kst('2026-11-16T10:00:00'), now)).toBe(6); // a week ago Monday
  });
  it('splitDay keeps the alert day as bars and earlier days as warm-up', () => {
    const bar = (s: string) => ({ t: kst(s), o: 1, h: 1, l: 1, c: 1, v: 1 });
    const bars = [bar('2026-11-20T15:19:00'), bar('2026-11-23T09:00:00'), bar('2026-11-23T14:23:00'), bar('2026-11-24T09:00:00')];
    const { day, warm } = splitDay(bars, kst('2026-11-23T14:23:00'));
    expect(day.map((b) => b.t)).toEqual([kst('2026-11-23T09:00:00'), kst('2026-11-23T14:23:00')]);
    expect(warm.map((b) => b.t)).toEqual([kst('2026-11-20T15:19:00')]);
  });
});
