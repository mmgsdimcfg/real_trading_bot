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

import { toNqFutures } from '../src/shared/market';
import { assertInquiry, restEndpoint } from '../src/shared/kis';

describe('E-Mini Nasdaq 100 futures index (KIS 해외지수분봉조회)', () => {
  const o1 = { ovrs_nmix_prdy_vrss: '274.50', prdy_vrss_sign: '2', prdy_ctrt: '1.05', ovrs_nmix_prdy_clpr: '26100.25', ovrs_nmix_prpr: '26374.75', hts_kor_isnm: 'E-Mini Nasdaq 100 지수선물' };
  const rows: Record<string, string>[] = [
    { stck_bsop_date: '20261126', stck_cntg_hour: '131300', optn_prpr: '26374.75', optn_oprc: '26372.50', optn_hgpr: '26377.75', optn_lwpr: '26372.25', cntg_vol: '19' },
    { stck_bsop_date: '20261126', stck_cntg_hour: '131200', optn_prpr: '26370.00', optn_oprc: '26369.00', optn_hgpr: '26373.00', optn_lwpr: '26368.00', cntg_vol: '7' },
    { stck_bsop_date: '', stck_cntg_hour: '131100', optn_prpr: '1' },
    { stck_bsop_date: '20261126', stck_cntg_hour: '131000', optn_prpr: '0' },
  ];
  it('parses the quote and the bars (oldest first, KST)', () => {
    const f = toNqFutures(o1, rows);
    expect(f.value).toBe(26374.75);
    expect(f.change).toBe(274.5);
    expect(f.changePct).toBe(1.05);
    expect(f.bars.map((b) => b.c)).toEqual([26370, 26374.75]);
    expect(f.lastBarAt).toBe(Date.UTC(2026, 10, 26, 4, 13)); // 13:13 KST
  });
  it('a falling sign makes the change negative, empty rows give no bars', () => {
    const f = toNqFutures({ ...o1, prdy_vrss_sign: '5', ovrs_nmix_prdy_vrss: '120.00', prdy_ctrt: '0.46' }, []);
    expect(f.change).toBe(-120);
    expect(f.changePct).toBe(-0.46);
    expect(f.lastBarAt).toBeNull();
  });
  it('is the only path outside domestic-stock the allowlist accepts', () => {
    expect(restEndpoint('inquire_time_indexchartprice').trId).toBe('FHKST03030200');
    expect(() => assertInquiry('/uapi/overseas-price/v1/quotations/inquire-time-indexchartprice', 'FHKST03030200')).not.toThrow();
    expect(() => assertInquiry('/uapi/overseas-price/v1/quotations/price', 'HHDFS00000300')).toThrow();
    expect(() => assertInquiry('/uapi/overseas-stock/v1/trading/order', 'TTTT1002U')).toThrow();
    expect(() => assertInquiry('/uapi/overseas-futureoption/v1/trading/order', 'OTFM3001U')).toThrow();
  });
});
