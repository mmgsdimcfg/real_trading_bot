/*
 * Update log
 * - [2026-11-16] type=test - tradingSessionRange (timeline axis = whole trading session of the day: NXT 08:00-20:00, KRX 09:00-15:30)
 */
import { describe, expect, it } from 'vitest';
import { fmtKstDateTime, tradingSessionRange } from '../src/shared/time';

describe('tradingSessionRange', () => {
  it('NXT symbol: 08:00 - 20:00 KST', () => {
    const r = tradingSessionRange('20261116', true)!;
    expect(r.map(fmtKstDateTime)).toEqual(['2026-11-16 08:00:00', '2026-11-16 20:00:00']);
  });
  it('KRX only: 09:00 - 15:30 KST, dashed date accepted', () => {
    const r = tradingSessionRange('2026-11-16', false)!;
    expect(r.map(fmtKstDateTime)).toEqual(['2026-11-16 09:00:00', '2026-11-16 15:30:00']);
  });
  it('unknown date: null', () => {
    expect(tradingSessionRange('', true)).toBeNull();
    expect(tradingSessionRange(undefined, false)).toBeNull();
  });
});
