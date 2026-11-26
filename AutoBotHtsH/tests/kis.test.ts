/*
 * Update log
 * - [2026-11-06] type=test - HtsH core: inquiry-only allowlist (order / balance TRs and non-quotation paths rejected), realtime frame
 *   parsing with the generated column order, stock master row layout, minute / daily chart rows -> bars (minute = start time), quote /
 *   order book normalizers, multi-price params, zip reader (stored + deflated), symbol search incl. 초성, live bar updates from trades
 * - [2026-11-06] type=test - mergeTrades: REST trade lists merged without duplicates, newest first
 * - [2026-11-06] type=test - combined (KRX + NXT) realtime TRs allowed; their trade records map back to the KRX shape (side from CNTG_CLS_CODE)
 */

import { describe, expect, it, vi } from 'vitest';
import { deflateRawSync } from 'node:zlib';

vi.mock('electron', () => ({ app: { getPath: () => '.' } }));

import {
  assertInquiry,
  assertWsTr,
  toKrxRecord,
  wsToTrade,
  dailyRowsToBars,
  minuteRowsToBars,
  multPriceParams,
  parseMasterRow,
  parseWsFrame,
  recordToObject,
  restEndpoint,
  toOrderBook,
  toQuote,
  wsColumns,
  REST_APIS,
} from '../src/shared/kis';
import { KIS_SPEC } from '../src/shared/kisSpec.generated';
import { matchScore, choseong } from '../src/shared/search';
import { applyTrades, tickBarsFromTrades, tradeTime, kstToday } from '../src/shared/live';
import { unzip } from '../src/main/master';
import { fmtKstDateTime } from '../src/shared/time';

describe('inquiry-only allowlist', () => {
  it('every REST API HtsH uses is a GET quotation/ranking TR from the official spec', () => {
    for (const api of REST_APIS) {
      const ep = restEndpoint(api);
      // [2026-11-26] + the one overseas index inquiry (E-Mini Nasdaq 100 futures index of the index window)
      expect(ep.path).toMatch(/^\/uapi\/(domestic-stock\/v1\/(quotations|ranking)\/|overseas-price\/v1\/quotations\/inquire-time-indexchartprice$)/);
      expect(ep.trId).toMatch(/^(FH|CT)/);
    }
  });
  it('rejects anything not on the list (orders, balances)', () => {
    expect(() => restEndpoint('order_cash')).toThrow(/허용되지 않은/);
    expect(() => restEndpoint('inquire_balance')).toThrow(/허용되지 않은/);
  });
  it('rejects order / account tr_ids and trading paths even if asked directly', () => {
    expect(() => assertInquiry('/uapi/domestic-stock/v1/trading/order-cash', 'TTTC0802U')).toThrow();
    expect(() => assertInquiry('/uapi/domestic-stock/v1/quotations/inquire-price', 'TTTC8434R')).toThrow();
    expect(() => assertInquiry('/uapi/domestic-stock/v1/trading/inquire-balance', 'FHKST01010100')).toThrow();
    expect(() => assertInquiry('/uapi/hashkey', 'FHKST01010100')).toThrow();
    expect(() => assertInquiry('/uapi/domestic-stock/v1/quotations/inquire-price', 'FHKST01010100')).not.toThrow();
  });
  it('realtime: only KRX trades and order book', () => {
    expect(() => assertWsTr('H0STCNT0')).not.toThrow();
    expect(() => assertWsTr('H0STASP0')).not.toThrow();
    expect(() => assertWsTr('H0UNCNT0')).not.toThrow();
    expect(() => assertWsTr('H0UNASP0')).not.toThrow();
    expect(() => assertWsTr('H0STCNI0')).toThrow(); // 체결통보 (account)
    expect(() => assertWsTr('H0IFASP0')).toThrow();
  });
});

describe('realtime frames', () => {
  it('splits multiple records with the spec column count', () => {
    const cols = wsColumns('H0STCNT0');
    expect(cols[0]).toBe('MKSC_SHRN_ISCD');
    expect(cols.length).toBe(KIS_SPEC.ccnl_krx.wsColumns.length);
    const rec = (code: string, price: string) => cols.map((c) => (c === 'MKSC_SHRN_ISCD' ? code : c === 'STCK_PRPR' ? price : c === 'STCK_CNTG_HOUR' ? '093001' : '0'));
    const raw = `0|H0STCNT0|002|${[...rec('005930', '71200'), ...rec('005930', '71300')].join('^')}`;
    const f = parseWsFrame(raw, (tr) => wsColumns(tr).length)!;
    expect(f.records).toHaveLength(2);
    const o = recordToObject('H0STCNT0', f.records[1]);
    expect(o.STCK_PRPR).toBe('71300');
    expect(o.STCK_CNTG_HOUR).toBe('093001');
  });
  it('JSON control messages are not data frames', () => {
    expect(parseWsFrame('{"header":{"tr_id":"PINGPONG"}}', () => 0)).toBeNull();
  });
});

describe('stock master', () => {
  it('parses the fixed-width KOSPI row (code 9, std 12, name, 228 tail)', () => {
    const row = '005930   ' + 'KR7005930003' + '삼성전자' + 'ST' + ' '.repeat(226);
    expect(parseMasterRow(row, 'KOSPI')).toEqual({ code: '005930', std: 'KR7005930003', name: '삼성전자', market: 'KOSPI', group: 'ST' });
    const kq = '035720   ' + 'KR7035720002' + '카카오' + 'ST' + ' '.repeat(220);
    expect(parseMasterRow(kq, 'KOSDAQ')?.name).toBe('카카오');
    expect(parseMasterRow('short', 'KOSPI')).toBeNull();
  });
});

describe('charts', () => {
  it('minute rows: start-labelled, oldest first, de-duplicated', () => {
    const rows = [
      { stck_bsop_date: '20261106', stck_cntg_hour: '090100', stck_oprc: '71100', stck_hgpr: '71300', stck_lwpr: '71000', stck_prpr: '71200', cntg_vol: '1500' },
      { stck_bsop_date: '20261106', stck_cntg_hour: '090000', stck_oprc: '71000', stck_hgpr: '71200', stck_lwpr: '70900', stck_prpr: '71100', cntg_vol: '9000' },
      { stck_bsop_date: '20261106', stck_cntg_hour: '090100', stck_oprc: '71100', stck_hgpr: '71300', stck_lwpr: '71000', stck_prpr: '71250', cntg_vol: '1600' },
    ];
    const b = minuteRowsToBars(rows);
    expect(b).toHaveLength(2);
    expect(fmtKstDateTime(b[0].t)).toBe('2026-11-06 09:00:00');
    expect(b[1]).toMatchObject({ c: 71250, v: 1600 });
  });
  it('daily rows (stock and index fields)', () => {
    const s = dailyRowsToBars([{ stck_bsop_date: '20261105', stck_oprc: '1', stck_hgpr: '3', stck_lwpr: '1', stck_clpr: '2', acml_vol: '10' }]);
    expect(s[0]).toMatchObject({ o: 1, h: 3, l: 1, c: 2, v: 10 });
    const i = dailyRowsToBars([{ stck_bsop_date: '20261105', bstp_nmix_oprc: '2580.1', bstp_nmix_hgpr: '2600', bstp_nmix_lwpr: '2570', bstp_nmix_prpr: '2590.5', acml_vol: '5' }]);
    expect(i[0].c).toBe(2590.5);
  });
});

describe('normalizers', () => {
  it('quote: signed change from prdy_vrss_sign', () => {
    const q = toQuote('005930', { stck_prpr: '71200', prdy_vrss: '800', prdy_vrss_sign: '5', prdy_ctrt: '-1.11', stck_sdpr: '72000', hts_avls: '4250000', per: '12.8' });
    expect(q).toMatchObject({ price: 71200, change: -800, changePct: -1.11, prevClose: 72000, marketCapEok: 4250000, per: 12.8 });
  });
  it('order book: 10 levels, totals, expected price', () => {
    const o1: Record<string, string> = { total_askp_rsqn: '100', total_bidp_rsqn: '200' };
    for (let i = 1; i <= 10; i++) (o1[`askp${i}`] = String(71200 + i * 100)), (o1[`askp_rsqn${i}`] = '5'), (o1[`bidp${i}`] = String(71200 - (i - 1) * 100)), (o1[`bidp_rsqn${i}`] = '7');
    const ob = toOrderBook(o1, { antc_cnpr: '71250', antc_vol: '12', antc_cntg_vrss: '50', antc_cntg_vrss_sign: '2' });
    expect(ob.asks).toHaveLength(10);
    expect(ob.asks[0][0]).toBe(71300);
    expect(ob.bids[0][0]).toBe(71200);
    expect(ob.expected).toEqual({ price: 71250, volume: 12, change: 50 });
  });
  it('multi-price params: 30 codes max', () => {
    const p = multPriceParams(Array.from({ length: 35 }, (_, i) => String(100000 + i)));
    expect(p.FID_INPUT_ISCD_30).toBe('100029');
    expect(p.FID_INPUT_ISCD_31).toBeUndefined();
    expect(p.FID_COND_MRKT_DIV_CODE_1).toBe('J');
  });
});

describe('zip reader', () => {
  const zipOf = (name: string, data: Buffer, deflate: boolean) => {
    const comp = deflate ? deflateRawSync(data) : data;
    const nm = Buffer.from(name);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(deflate ? 8 : 0, 8);
    local.writeUInt32LE(comp.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nm.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(deflate ? 8 : 0, 10);
    central.writeUInt32LE(comp.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(nm.length, 28);
    central.writeUInt32LE(0, 42);
    const cdOff = 30 + nm.length + comp.length;
    const eocd = Buffer.alloc(22);
    eocd.writeUInt32LE(0x06054b50, 0);
    eocd.writeUInt16LE(1, 8);
    eocd.writeUInt16LE(1, 10);
    eocd.writeUInt32LE(46 + nm.length, 12);
    eocd.writeUInt32LE(cdOff, 16);
    return Buffer.concat([local, nm, comp, central, nm, eocd]);
  };
  it('reads deflated and stored entries', () => {
    const text = Buffer.from('005930   KR7005930003삼성전자\n'.repeat(50));
    for (const d of [true, false]) {
      const files = unzip(zipOf('kospi_code.mst', text, d));
      expect(files[0].name).toBe('kospi_code.mst');
      expect(files[0].data.equals(text)).toBe(true);
    }
  });
});

describe('symbol search', () => {
  it('code, name, 초성', () => {
    expect(matchScore('005930', '삼성전자', '005930')).toBe(100);
    expect(matchScore('005930', '삼성전자', '0059')).toBeGreaterThan(0);
    expect(matchScore('005930', '삼성전자', '삼성')).toBeGreaterThan(matchScore('028260', '삼성물산우', '전자'));
    expect(choseong('삼성전자')).toBe('ㅅㅅㅈㅈ');
    expect(matchScore('005930', '삼성전자', 'ㅅㅅㅈㅈ')).toBeGreaterThan(0);
    expect(matchScore('035420', 'NAVER', 'naver')).toBeGreaterThan(0);
    expect(matchScore('035720', '카카오', '현대')).toBe(0);
  });
});

describe('live bars from realtime trades', () => {
  const now = kstToday() + 10 * 3600000; // 10:00 KST today
  const t = (hms: string, price: number, volume = 1) => ({ time: hms, price, change: 0, volume, side: 1 });
  it('rolls trades into the current minute bucket and opens new buckets', () => {
    const base = [{ t: tradeTime('095900', now), o: 100, h: 101, l: 99, c: 100, v: 10 }];
    const trades = [t('100005', 103, 2), t('095930', 102, 3)]; // newest first
    const out = applyTrades(base, trades, 'min', 1, tradeTime('095920', now), now);
    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({ h: 102, c: 102, v: 13 });
    expect(out[1]).toMatchObject({ o: 103, c: 103, v: 2 });
  });
  it('ignores trades older than the REST data', () => {
    const base = [{ t: tradeTime('095900', now), o: 100, h: 101, l: 99, c: 100, v: 10 }];
    expect(applyTrades(base, [t('095910', 150)], 'min', 1, tradeTime('095930', now), now)).toEqual(base);
  });
  it('tick bars', () => {
    const tr = [t('100003', 4), t('100002', 3), t('100001', 2), t('100000', 1)];
    const b = tickBarsFromTrades(tr, 2, now);
    expect(b).toHaveLength(2);
    expect(b[0]).toMatchObject({ o: 1, c: 2, h: 2, l: 1 });
  });
});

describe('combined KRX + NXT realtime', () => {
  it('maps H0UNCNT0 records to H0STCNT0 with the trade side', () => {
    const m = toKrxRecord('H0UNCNT0', { MKSC_SHRN_ISCD: '000660', STCK_CNTG_HOUR: '173100', STCK_PRPR: '1151000', CNTG_VOL: '10', CNTG_CLS_CODE: '5' });
    expect(m.tr).toBe('H0STCNT0');
    expect(wsToTrade(m.rec).trade).toMatchObject({ time: '173100', price: 1151000, volume: 10, side: 5 });
    expect(toKrxRecord('H0UNASP0', {}).tr).toBe('H0STASP0');
    expect(toKrxRecord('H0STCNT0', { CCLD_DVSN: '1' }).rec.CCLD_DVSN).toBe('1');
  });
});

describe('mergeTrades', () => {
  const t = (time: string, price: number, volume: number) => ({ time, price, volume, change: 0, side: 0 });
  it('adds only trades not already cached, newest first', async () => {
    const { mergeTrades } = await import('../src/shared/live');
    const cur = [t('193000', 100, 5), t('192959', 101, 1)];
    const out = mergeTrades(cur, [t('193001', 102, 2), t('193000', 100, 5), t('192959', 101, 1), t('192959', 101, 1), t('192950', 99, 3)]);
    expect(out.map((x) => `${x.time}/${x.volume}`)).toEqual(['193001/2', '193000/5', '192959/1', '192959/1', '192950/3']);
    expect(mergeTrades(cur, [t('193000', 100, 5)])).toBe(cur);
  });
});

describe('REST / realtime trade merge (review cases)', () => {
  const t = (time: string, price: number, volume: number) => ({ time, price, volume, change: 0, side: 0 });
  it('keeps the REST order of same-second trades', async () => {
    const { mergeTrades } = await import('../src/shared/live');
    const out = mergeTrades([t('100000', 100, 1)], [t('100000', 101, 1), t('100000', 100, 1)]);
    expect(out.map((x) => x.price)).toEqual([101, 100]);
  });
  it('a realtime trade replaces its REST copy instead of doubling it', async () => {
    const { mergeTrades, addRealtimeTrade } = await import('../src/shared/live');
    const seeded = mergeTrades([], [t('100001', 102, 3), t('100000', 101, 2)]);
    const out = addRealtimeTrade(seeded, t('100001', 102, 3));
    expect(out).toHaveLength(2);
    expect(out[0].rest).toBeUndefined();
    expect(addRealtimeTrade(out, t('100001', 102, 3))).toHaveLength(3); // a second, real trade of the same shape
  });
  it('realtime trades after the fetch stay newest at the same second', async () => {
    const { mergeTrades } = await import('../src/shared/live');
    const out = mergeTrades([t('100001', 105, 9)], [t('100001', 104, 1), t('100000', 103, 1)]);
    expect(out.map((x) => x.price)).toEqual([105, 104, 103]);
  });
});

describe('trade identity', () => {
  it('a separate same-second trade with a different strength is added, not merged', async () => {
    const { mergeTrades, addRealtimeTrade } = await import('../src/shared/live');
    const t = (s: number) => ({ time: '100000', price: 100, volume: 1, change: 0, side: 0, strength: s });
    const seeded = mergeTrades([], [t(117.82)]);
    expect(addRealtimeTrade(seeded, t(117.83))).toHaveLength(2);
    expect(addRealtimeTrade(seeded, t(117.82))).toHaveLength(1);
  });
});
