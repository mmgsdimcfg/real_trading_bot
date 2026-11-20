/*
 * Update log
 * - [2026-11-16] type=test - tradingSessionRange (timeline axis = whole trading session of the day: NXT 08:00-20:00, KRX 09:00-15:30)
 * - [2026-11-19] type=test - KRX only: 09:00-20:00 (KRX after-hours market)
 */
import { describe, expect, it } from 'vitest';
import { fmtKstDateTime, kstToEpoch, tradingSessionRange } from '../src/shared/time';
import { regularOnly } from '../src/shared/price';

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
  it('unknown date: null', () => {
    expect(tradingSessionRange('', true)).toBeNull();
    expect(tradingSessionRange(undefined, false)).toBeNull();
  });
});

describe('regularOnly', () => {
  it('keeps the bars of 09:00 - 15:30 KST', () => {
    const b = (hm: string) => ({ t: kstToEpoch('2026-11-16', hm), o: 1, h: 1, l: 1, c: 1, v: 1 });
    const out = regularOnly([b('08:59:00'), b('09:00:00'), b('15:30:00'), b('15:31:00'), b('19:00:00')]);
    expect(out.map((x) => fmtKstDateTime(x.t).slice(11))).toEqual(['09:00:00', '15:30:00']);
  });
});
