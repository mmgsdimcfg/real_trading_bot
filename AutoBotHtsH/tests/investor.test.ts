/*
 * Update log
 * - [2026-11-20] type=test - 투자자 누적 pane: monthsBefore(), investorCum() over day / week bars (window start, NaN before it, carry
 *   to bars without a row), and KisMarket.investorDaily(): pages back by date to about a year, answers from the cache within the reuse
 *   window, a later refresh asks only the newest page, an error returns the cached rows
 * - [2026-11-20] type=test - KIS refuses today's date before 15:40 ("TIME LIMIT 00:00 ~ 15:40", OPSQ2001): the first page is asked from
 *   yesterday then; market UN on the real server
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, safeStorage: {} }));

import { investorCum, monthsBefore } from '../src/shared/indicators';
import { KisMarket } from '../src/main/kisMarket';
import type { ChartStore } from '../src/main/chartCache';
import type { Bar } from '../src/shared/price';

const at = (iso: string) => vi.setSystemTime(new Date(`${iso}+09:00`));
afterEach(() => vi.useRealTimers());

const KST = 9 * 3600000;
const t = (d: string) => Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8)) - KST;
const bar = (d: string): Bar => ({ t: t(d), o: 1, h: 1, l: 1, c: 1, v: 1 });

/** weekdays (YYYYMMDD) from `from` up to `to` */
function weekdays(from: string, to: string): string[] {
  const out: string[] = [];
  for (let x = t(from) + KST; ; x += 86400000) {
    const d = new Date(x);
    const s = d.toISOString().slice(0, 10).replace(/-/g, '');
    if (s > to) break;
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) out.push(s);
  }
  return out;
}

describe('investorCum', () => {
  it('monthsBefore clamps the day to the month', () => {
    expect(monthsBefore('20261120', 1)).toBe('20261020');
    expect(monthsBefore('20261120', 12)).toBe('20251120');
    expect(monthsBefore('20260331', 1)).toBe('20260228');
    expect(monthsBefore('20260115', 3)).toBe('20251015');
  });

  it('sums the rows after the window start; bars before it are NaN, a bar without a row carries the sum', () => {
    const rows = [
      { date: '20261016', person: 100, foreign: 1, institution: 10 },
      { date: '20261020', person: 100, foreign: 1, institution: 10 }, // = start (1 month before 11/20): not counted
      { date: '20261021', person: 5, foreign: -2, institution: 1 },
      { date: '20261119', person: -3, foreign: 4, institution: 0 },
      { date: '20261120', person: 1, foreign: 1, institution: 1 },
    ];
    const bars = ['20261016', '20261020', '20261021', '20261119', '20261120'].map(bar);
    const c = investorCum(bars, rows, 1);
    expect(c.start).toBe('20261020');
    expect(c.person.slice(0, 2).every(Number.isNaN)).toBe(true);
    expect(c.person.slice(2)).toEqual([5, 2, 3]);
    expect(c.foreign.slice(2)).toEqual([-2, 2, 3]);
    expect(c.institution.slice(2)).toEqual([1, 1, 2]);
    // today's row not there yet: the last bar keeps yesterday's sum
    const c2 = investorCum(bars, rows.slice(0, 4), 1);
    expect(c2.start).toBe('20261019');
    expect(c2.person[4]).toBe(c2.person[3]);
  });

  it('week bars take the sum up to the last row of their week', () => {
    const rows = weekdays('20261102', '20261120').map((d) => ({ date: d, person: 1, foreign: 2, institution: 3 }));
    const bars = ['20261102', '20261109', '20261116'].map(bar);
    const c = investorCum(bars, rows, 12);
    expect(c.person).toEqual([5, 10, 15]);
    expect(c.institution).toEqual([15, 30, 45]);
  });
});

class MemStore implements ChartStore {
  m = new Map<string, string>();
  get<T>(k: string) {
    const v = this.m.get(k);
    return v === undefined ? undefined : (JSON.parse(v) as T);
  }
  put<T>(k: string, v: T) {
    this.m.set(k, JSON.stringify(v));
  }
  flush() {}
}

/** fake KIS: investor rows up to FID_INPUT_DATE_1, newest first, `per` rows per call */
function server(days: string[], per = 30) {
  const calls: string[] = [];
  const mks: string[] = [];
  let fail = false;
  const client = {
    env: 'real',
    get: async (api: string, params: Record<string, string>) => {
      if (api !== 'investor_trade_by_stock_daily') throw new Error(api);
      calls.push(params.FID_INPUT_DATE_1);
      mks.push(params.FID_COND_MRKT_DIV_CODE);
      if (fail) throw new Error('network');
      const now = new Date(Date.now() + KST).toISOString();
      if (params.FID_INPUT_DATE_1 === now.slice(0, 10).replace(/-/g, '') && now.slice(11, 16) < '15:40') {
        throw Object.assign(new Error('TIME LIMIT 00:00 ~ 15:40'), { code: 'OPSQ2001' });
      }
      const rows = days
        .filter((d) => d <= params.FID_INPUT_DATE_1)
        .reverse()
        .slice(0, per)
        .map((d) => ({ stck_bsop_date: d, prsn_ntby_qty: '-3', frgn_ntby_qty: '2', orgn_ntby_qty: '1' }));
      return { output1: {}, output2: rows, trCont: '' };
    },
  };
  return { calls, mks, client, setFail: (v: boolean) => (fail = v) };
}

describe('KisMarket.investorDaily', () => {
  it('loads about a year, reuses it, refreshes with the newest page only, keeps the cache on an error', async () => {
    vi.useFakeTimers();
    at('2026-11-20T10:00:00');
    const days = weekdays('20250101', '20261120');
    const s = server(days);
    const disk = new MemStore();
    const m = new KisMarket(s.client as never, () => false, disk);
    const r1 = await m.investorDaily('005930');
    // 10:00: today's date is refused before 15:40 -> from yesterday, without a failed call
    expect(s.calls[0]).toBe('20261119');
    expect(s.mks.every((x) => x === 'UN')).toBe(true);
    expect(r1[r1.length - 1].date).toBe('20261119');
    expect(r1[0].date <= '20251120').toBe(true);
    expect(r1.every((r, i) => i === 0 || r.date > r1[i - 1].date)).toBe(true);
    expect(r1[0]).toMatchObject({ person: -3, foreign: 2, institution: 1 });
    const full = s.calls.length;
    expect(full).toBeGreaterThan(5);

    // within the reuse window: no call
    await m.investorDaily('005930');
    expect(s.calls.length).toBe(full);

    // next day, new run from disk: one page
    at('2026-11-23T16:00:00');
    days.push('20261123');
    const m2 = new KisMarket(s.client as never, () => false, disk);
    const r2 = await m2.investorDaily('005930');
    expect(s.calls.length).toBe(full + 1);
    expect(r2[r2.length - 1].date).toBe('20261123');

    // error: the cached rows
    at('2026-11-23T17:00:00');
    s.setFail(true);
    const r3 = await m2.investorDaily('005930');
    expect(r3.length).toBe(r2.length);
  });
});
