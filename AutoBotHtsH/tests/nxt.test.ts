/*
 * Update log
 * - [2026-11-12] type=test - NXT: NXT master parsing, NXT-only hours, per-code market codes of the multi-price request, the watchlist
 *   asking NX for NXT symbols only while KRX is closed (real server), the KRX fallback of a symbol without an NX price, and the
 *   current price / order book using UN for NXT symbols with a J fallback
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, safeStorage: {} }));

import { multPriceParams, nxtOnlyHours, parseNxtCodes } from '../src/shared/kis';
import { KisMarket } from '../src/main/kisMarket';

/** KST wall time -> the fake system time */
const at = (hhmm: string) => vi.setSystemTime(new Date(`2026-11-12T${hhmm}:00+09:00`));

const NXT = new Set(['010060', '241560']);
const hint = (c: string) => NXT.has(c);

/** fake KIS client: records every call; multi-price rows per market code */
function fake(env: 'real' | 'paper', price: (code: string, mk: string) => number | null, fail: (api: string, mk: string) => boolean = () => false) {
  const calls: { api: string; params: Record<string, string> }[] = [];
  const client = {
    env,
    get: async (api: string, params: Record<string, string>) => {
      calls.push({ api, params });
      const mk = params.FID_COND_MRKT_DIV_CODE || '';
      if (fail(api, mk)) throw new Error('not served');
      if (api === 'intstock_multprice') {
        const output = [];
        for (let i = 1; i <= 30; i++) {
          const code = params[`FID_INPUT_ISCD_${i}`];
          if (!code) break;
          const p = price(code, params[`FID_COND_MRKT_DIV_CODE_${i}`]);
          if (p !== null) output.push({ inter_shrn_iscd: code, inter_kor_isnm: code, inter2_prpr: String(p), inter2_prdy_vrss: '0', prdy_vrss_sign: '3', prdy_ctrt: '0', acml_vol: '1' });
        }
        return { output };
      }
      if (api === 'search_stock_info') return { output: { cptt_trad_tr_psbl_yn: 'Y', nxt_tr_stop_yn: 'N' } };
      if (api === 'inquire_price') return { output: { stck_prpr: mk === 'UN' ? '207500' : '196600', prdy_vrss: '0', prdy_vrss_sign: '3', prdy_ctrt: '0' } };
      if (api === 'inquire_asking_price_exp_ccn') return { output1: mk === 'UN' ? { askp1: '207600', bidp1: '207500' } : { askp1: '196700', bidp1: '196600' }, output2: {} };
      return { output: [] };
    },
  };
  return { calls, market: new KisMarket(client as never, hint) };
}

afterEach(() => vi.useRealTimers());

describe('NXT helpers', () => {
  it('reads the codes of an NXT master file', () => {
    const text = ['000070   KR7000070003삼양홀딩스   ST3', '0008Z0   KR70008Z0005에스엔시스   ST2', '', 'bad row'].join('\n');
    expect(parseNxtCodes(text)).toEqual(['000070', '0008Z0']);
  });

  it('NXT-only hours: before 09:00 and from 15:40', () => {
    expect(nxtOnlyHours('0805')).toBe(true);
    expect(nxtOnlyHours('0859')).toBe(true);
    expect(nxtOnlyHours('0900')).toBe(false);
    expect(nxtOnlyHours('1530')).toBe(false);
    expect(nxtOnlyHours('1540')).toBe(true);
    expect(nxtOnlyHours('1959')).toBe(true);
  });

  it('multi-price params take a market code per symbol', () => {
    const p = multPriceParams(['010060', '006360'], (c) => (c === '010060' ? 'NX' : 'J'));
    expect(p.FID_COND_MRKT_DIV_CODE_1).toBe('NX');
    expect(p.FID_COND_MRKT_DIV_CODE_2).toBe('J');
  });
});

describe('watchlist prices in NXT hours', () => {
  it('asks NX for NXT symbols before the KRX open, J for the others, in one call', async () => {
    vi.useFakeTimers();
    at('08:05');
    const { calls, market } = fake('real', (_c, mk) => (mk === 'NX' ? 207500 : 196600));
    const rows = await market.multi(['010060', '006360', '241560']);
    expect(calls).toHaveLength(1);
    expect(calls[0].params).toMatchObject({ FID_COND_MRKT_DIV_CODE_1: 'NX', FID_COND_MRKT_DIV_CODE_2: 'J', FID_COND_MRKT_DIV_CODE_3: 'NX' });
    expect(rows.map((r) => [r.code, r.price])).toEqual([
      ['010060', 207500],
      ['006360', 196600],
      ['241560', 207500],
    ]);
  });

  it('asks J for every symbol while KRX is open, and on the paper server', async () => {
    vi.useFakeTimers();
    at('10:00');
    const open = fake('real', () => 1000);
    await open.market.multi(['010060', '006360']);
    expect(open.calls[0].params.FID_COND_MRKT_DIV_CODE_1).toBe('J');
    at('08:05');
    const paper = fake('paper', () => 1000);
    await paper.market.multi(['010060']);
    expect(paper.calls[0].params.FID_COND_MRKT_DIV_CODE_1).toBe('J');
  });

  it('after the KRX close NX is used again from 15:40', async () => {
    vi.useFakeTimers();
    at('15:35');
    const a = fake('real', () => 1000);
    await a.market.multi(['010060']);
    expect(a.calls[0].params.FID_COND_MRKT_DIV_CODE_1).toBe('J');
    at('15:45');
    const b = fake('real', () => 1000);
    await b.market.multi(['010060']);
    expect(b.calls[0].params.FID_COND_MRKT_DIV_CODE_1).toBe('NX');
  });

  it('a symbol without an NX price gets its KRX row and is asked with J for the next minute', async () => {
    vi.useFakeTimers();
    at('08:05');
    const { calls, market } = fake('real', (c, mk) => (mk === 'NX' && c === '241560' ? null : mk === 'NX' ? 207500 : 52800));
    const rows = await market.multi(['010060', '241560']);
    expect(calls).toHaveLength(2); // NX request, then the KRX request for the missing row
    expect(rows.map((r) => [r.code, r.price])).toEqual([
      ['010060', 207500],
      ['241560', 52800],
    ]);
    await market.multi(['010060', '241560']);
    expect(calls).toHaveLength(3);
    expect(calls[2].params).toMatchObject({ FID_COND_MRKT_DIV_CODE_1: 'NX', FID_COND_MRKT_DIV_CODE_2: 'J' });
    vi.setSystemTime(Date.now() + 61000);
    await market.multi(['010060', '241560']);
    expect(calls[3].params.FID_COND_MRKT_DIV_CODE_2).toBe('NX'); // tried again after a minute
  });

  it('an NX request that fails falls back to KRX for the whole request', async () => {
    vi.useFakeTimers();
    at('08:05');
    let n = 0;
    const { market } = fake(
      'real',
      () => 196600,
      (api) => api === 'intstock_multprice' && n++ === 0,
    );
    const rows = await market.multi(['010060', '006360']);
    expect(rows.map((r) => r.price)).toEqual([196600, 196600]);
  });
});

describe('current price / order book of NXT symbols', () => {
  it('use the combined market on the real server', async () => {
    const { calls, market } = fake('real', () => 1);
    expect((await market.quote('010060')).price).toBe(207500);
    expect((await market.orderbook('010060')).asks[0][0]).toBe(207600);
    expect(calls.filter((c) => c.api !== 'search_stock_info').map((c) => c.params.FID_COND_MRKT_DIV_CODE)).toEqual(['UN', 'UN']);
  });

  it('fall back to KRX when the combined market is not served', async () => {
    const { market } = fake(
      'real',
      () => 1,
      (_api, mk) => mk === 'UN',
    );
    expect((await market.quote('010060')).price).toBe(196600);
    expect((await market.orderbook('010060')).bids[0][0]).toBe(196600);
  });

  it('other symbols and the paper server ask KRX only, without the NXT lookup', async () => {
    const real = fake('real', () => 1);
    await real.market.quote('006360');
    expect(real.calls.map((c) => [c.api, c.params.FID_COND_MRKT_DIV_CODE])).toEqual([['inquire_price', 'J']]);
    const paper = fake('paper', () => 1);
    await paper.market.quote('010060');
    expect(paper.calls.map((c) => [c.api, c.params.FID_COND_MRKT_DIV_CODE])).toEqual([['inquire_price', 'J']]);
  });
});
