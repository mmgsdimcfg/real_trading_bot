/*
 * Update log
 * - [2026-11-16] type=test - tradingSessionRange (timeline axis = whole trading session of the day: NXT 08:00-20:00, KRX 09:00-15:30)
 * - [2026-11-19] type=test - KRX only: 09:00-20:00 (KRX after-hours market)
 */
import { describe, expect, it } from 'vitest';
import { fmtKstDateTime, kstToEpoch, shownSessionRange, tradingSessionRange } from '../../src/bot/shared/time';
import { regularOnly } from '../../src/bot/shared/price';

describe('tradingSessionRange', () => {
  it('NXT symbol: 08:00 - 20:00 KST', () => {
    const r = tradingSessionRange('20261116', true)!;
    expect(r.map(fmtKstDateTime)).toEqual(['2026-11-16 08:00:00', '2026-11-16 20:00:00']);
  });
  it('KRX only: 09:00 - 20:00 KST (after-hours market, no pre-market), dashed date accepted', () => {
    const r = tradingSessionRange('2026-11-16', false)!;
    expect(r.map(fmtKstDateTime)).toEqual(['2026-11-16 09:00:00', '2026-11-16 20:00:00']);
  });
  it('[2026-11-20] regular session only (운용 시간): 09:00 - 15:30 for every symbol', () => {
    expect(tradingSessionRange('20261116', true, false)!.map(fmtKstDateTime)).toEqual(['2026-11-16 09:00:00', '2026-11-16 15:30:00']);
    expect(tradingSessionRange('20261116', false, false)!.map(fmtKstDateTime)).toEqual(['2026-11-16 09:00:00', '2026-11-16 15:30:00']);
  });
  it('[2026-11-23] shownSessionRange: regular-only still shows the hours traded right now on the live day', () => {
    const at = (hm: string) => kstToEpoch('2026-11-23', hm + ':00');
    const show = (nxt: boolean, nxtOn: boolean, now: number | null) => shownSessionRange('20261123', nxt, nxtOn, now)!.map((t) => fmtKstDateTime(t).slice(11, 16)).join('-');
    expect(show(true, false, at('08:43'))).toBe('08:00-15:30'); // NXT pre-market
    expect(show(false, false, at('08:43'))).toBe('09:00-15:30'); // KRX only: no pre-market
    expect(show(true, false, at('10:00'))).toBe('09:00-15:30');
    expect(show(true, false, at('16:10'))).toBe('09:00-20:00'); // after-market
    expect(show(true, false, at('07:30'))).toBe('09:00-15:30');
    expect(show(true, false, at('20:30'))).toBe('09:00-15:30');
    expect(show(true, false, null)).toBe('09:00-15:30'); // past / replayed day
    expect(show(true, true, at('10:00'))).toBe('08:00-20:00');
    expect(shownSessionRange('', true, false, 1)).toBeNull();
  });
  it('unknown date: null', () => {
    expect(tradingSessionRange('', true)).toBeNull();
    expect(tradingSessionRange(undefined, false)).toBeNull();
  });
});

describe('regularOnly with the live range', () => {
  it('[2026-11-23] keeps the bars of the range traded right now', () => {
    const b = (hms: string) => ({ t: kstToEpoch('2026-11-23', hms) });
    const keep = shownSessionRange('20261123', true, false, kstToEpoch('2026-11-23', '08:43:00'));
    const out = regularOnly([{ t: kstToEpoch('2026-11-20', '08:30:00') }, b('08:10:00'), b('09:00:00'), b('16:00:00')], keep);
    expect(out.map((x) => fmtKstDateTime(x.t).slice(5, 16))).toEqual(['11-23 08:10', '11-23 09:00']);
  });
});

describe('regularOnly', () => {
  it('keeps the bars of 09:00 - 15:30 KST', () => {
    const b = (hm: string) => ({ t: kstToEpoch('2026-11-16', hm), o: 1, h: 1, l: 1, c: 1, v: 1 });
    const out = regularOnly([b('08:59:00'), b('09:00:00'), b('15:30:00'), b('15:31:00'), b('19:00:00')]);
    expect(out.map((x) => fmtKstDateTime(x.t).slice(11))).toEqual(['09:00:00', '15:30:00']);
  });
});
