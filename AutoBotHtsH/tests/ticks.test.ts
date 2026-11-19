/*
 * Update log
 * - [2026-11-06] type=test - tick back-fill: inquire_time_itemconclusion rows -> trades (side from ask / bid), KisMarket.ticks() paging
 *   (boundary overlap removed by accumulated volume, a second with more than 30 trades stepped over, `until` stop, UN only on the real
 *   server), acml-based merge / realtime replace / ordering
 * - [2026-11-06] type=test - review fixes: realtime batch merged in one pass and a REST copy found beyond 500 trades; demo ticks keep
 *   their identity across calls and honour cursor / until
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, safeStorage: {} }));

import { toConclusionTrades, type TradeTick } from '../src/shared/kis';
import { acmlMismatches, addRealtimeTrade, addRealtimeTrades, mergeTrades } from '../src/shared/live';
import { DemoMarket } from '../src/main/demoMarket';
import { KisMarket } from '../src/main/kisMarket';

const row = (time: string, price: number, vol: number, acml: number) => ({
  stck_cntg_hour: time,
  stck_prpr: String(price),
  prdy_vrss: '-100',
  prdy_vrss_sign: '5',
  askp: '1001',
  bidp: '1000',
  tday_rltv: '100.00',
  cnqn: String(vol),
  acml_vol: String(acml),
});

describe('toConclusionTrades', () => {
  it('parses volume / acml and infers the side from the quote', () => {
    const t = toConclusionTrades([row('100001', 1001, 5, 500), row('100000', 1000, 3, 495), { ...row('100000', 1000, 1, 0) }]);
    expect(t).toHaveLength(2); // a row without acml is dropped
    expect(t[0]).toMatchObject({ time: '100001', price: 1001, volume: 5, acml: 500, side: 1 });
    expect(t[1].side).toBe(5);
  });
});

/** fake KIS: today's trades (newest first), answers "trades at or before hour", 30 per call */
function fakeMarket(all: ReturnType<typeof row>[], env: 'real' | 'paper' = 'real', nxt = true) {
  const calls: Record<string, string>[] = [];
  const client = {
    env,
    get: async (api: string, p: Record<string, string>) => {
      if (api === 'search_stock_info') return { output: { cptt_trad_tr_psbl_yn: nxt ? 'Y' : 'N', nxt_tr_stop_yn: 'N' } };
      calls.push(p);
      return { output2: all.filter((o) => o.stck_cntg_hour <= p.FID_INPUT_HOUR_1).slice(0, 30) };
    },
  };
  return { m: new KisMarket(client as never), calls };
}

/** n trades, `perSec` per second, newest first, ending at 10:00:59 */
function day(n: number, perSec: number) {
  const out: ReturnType<typeof row>[] = [];
  let acml = 100000;
  for (let i = 0; i < n; i++) {
    const sec = 10 * 3600 + 59 - Math.floor(i / perSec);
    const h = [Math.floor(sec / 3600), Math.floor(sec / 60) % 60, sec % 60].map((v) => String(v).padStart(2, '0')).join('');
    out.push(row(h, 1000, 1, acml));
    acml -= 1;
  }
  return out;
}

describe('KisMarket.ticks', () => {
  it('pages backwards without duplicates and reports the cursor', async () => {
    const { m, calls } = fakeMarket(day(200, 4));
    const r = await m.ticks({ code: '000660', cursor: { hour: '100059', acml: Infinity }, pages: 20 });
    const acmls = r.trades.map((t) => t.acml);
    expect(new Set(acmls).size).toBe(acmls.length);
    expect(r.trades).toHaveLength(200);
    expect(r.skipped).toBe(0);
    expect(r.next).toBeNull(); // last page was short: start of the day
    expect(calls[0].FID_COND_MRKT_DIV_CODE).toBe('UN');
  });
  it('steps over a second with more than 30 trades', async () => {
    const { m } = fakeMarket(day(120, 40));
    const r = await m.ticks({ code: '000660', cursor: { hour: '100059', acml: Infinity }, pages: 20 });
    expect(r.skipped).toBeGreaterThan(0);
    expect(new Set(r.trades.map((t) => t.acml)).size).toBe(r.trades.length);
  });
  it('stops at `until` and uses J on the paper server', async () => {
    const { m, calls } = fakeMarket(day(300, 2), 'paper');
    const r = await m.ticks({ code: '000660', cursor: { hour: '100059', acml: Infinity }, pages: 20, until: 99950 });
    expect(calls.length).toBeLessThan(5);
    expect(r.trades.some((t) => (t.acml as number) <= 99950)).toBe(true);
    expect(calls[0].FID_COND_MRKT_DIV_CODE).toBe('J');
  });
});

describe('acml merge', () => {
  const t = (time: string, acml: number, price = 100): TradeTick => ({ time, price, volume: 1, change: 0, side: 0, acml });
  it('dedupes and orders by (time, acml)', () => {
    const out = mergeTrades([t('100001', 10), t('100000', 8)], [t('100001', 11), t('100001', 10), t('100000', 9), t('099959', 7)]);
    expect(out.map((x) => x.acml)).toEqual([11, 10, 9, 8, 7]);
  });
  it('a realtime trade replaces its REST copy, a new one goes in order', () => {
    const seeded = mergeTrades([], [t('100001', 11), t('100000', 9)]);
    const same = addRealtimeTrade(seeded, { ...t('100001', 11), side: 1 });
    expect(same).toHaveLength(2);
    expect(same[0]).toMatchObject({ acml: 11, side: 1 });
    expect(addRealtimeTrade(seeded, t('100000', 10)).map((x) => x.acml)).toEqual([11, 10, 9]);
    expect(addRealtimeTrade(seeded, t('100002', 12))[0].acml).toBe(12);
  });
});

describe('review fixes', () => {
  const t = (time: string, acml: number): TradeTick => ({ time, price: 100, volume: 1, change: 0, side: 0, acml });
  it('replaces a REST copy far down the cache and adds a batch in order', () => {
    const list = Array.from({ length: 1000 }, (_, i) => ({ ...t('100000', 5000 - i), rest: true }));
    const out = addRealtimeTrades(list, [{ ...t('100000', 4300), side: 1 }, t('100001', 6000), t('100001', 6001)]);
    expect(out).toHaveLength(1002);
    expect(out.filter((x) => x.acml === 4300)).toHaveLength(1);
    expect(out.find((x) => x.acml === 4300)?.side).toBe(1);
    expect(out.slice(0, 3).map((x) => x.acml)).toEqual([6001, 6000, 5000]);
    expect(addRealtimeTrade(list, t('100000', 4300))).toHaveLength(1000);
  });
  it('demo ticks: same identity across calls, cursor continues, until stops', async () => {
    const d = new DemoMarket();
    const a = await d.ticks({ code: '005930', pages: 2 });
    const b = await d.ticks({ code: '005930', pages: 1 });
    if (!a.trades.length) return; // before 09:00 there is nothing to page
    // a second may pass between the calls: compare the trades both calls returned
    const byA = new Map(a.trades.map((x) => [x.acml, x.price]));
    const common = b.trades.filter((x) => byA.has(x.acml));
    expect(common.length).toBeGreaterThan(20);
    for (const x of common) expect(x.price).toBe(byA.get(x.acml));
    if (a.next) {
      const c = await d.ticks({ code: '005930', cursor: a.next, pages: 1 });
      expect(Math.max(...c.trades.map((x) => x.acml as number))).toBeLessThan(Math.min(...a.trades.map((x) => x.acml as number)));
    }
    const u = await d.ticks({ code: '005930', pages: 20, until: a.trades[5].acml });
    expect(u.trades.every((x) => (x.acml as number) > (a.trades[5].acml as number))).toBe(true);
    expect(u.trades.length).toBeLessThan(10);
  });
});

describe('acmlMismatches', () => {
  const t = (acml: number, rest?: boolean): TradeTick => ({ time: '100000', price: 100, volume: 5, change: 0, side: 0, acml, rest });
  it('flags the same trade shape with different REST / realtime acml only', () => {
    expect(acmlMismatches([t(10), t(10, true)])).toBe(0); // matched (would be one trade in practice)
    expect(acmlMismatches([t(500), t(10, true)])).toBe(1); // different bases
    expect(acmlMismatches([t(10, true), t(20, true)])).toBe(0); // REST only
  });
});
