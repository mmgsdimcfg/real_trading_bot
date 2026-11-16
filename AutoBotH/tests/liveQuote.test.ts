/*
 * Update log
 * - [2026-11-12] type=test - current price from r003's quote log: change in won derived from price + rate (tick grid), row parsing by
 *   header name, newest row per code and the replay cursor, the display text, and PriceService.liveQuotes reading the growing file
 *   by byte offset (rows of the real 2026-11-12 file)
 * - [2026-11-16] type=test - open / high / low / volume / value of a row, fmtKrwShort
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, dialog: {}, ipcMain: { handle: () => undefined, on: () => undefined } }));

import { fmtKrwShort, fromMarketQuote, krxTick, parseQuoteRows, quoteChange, quoteHeader, quoteText, QuoteBook } from '../src/shared/liveQuote';
import { PriceService } from '../src/main/prices';
import type { FileHost } from '../src/main/host';

const HEADER = 'ts,code,market,stck_prpr,acml_vol,acml_tr_pbmn,stck_oprc,stck_hgpr,stck_lwpr,wghn_avrg_stck_prc,prdy_ctrt,vi_cls_code,mrkt_warn_cls_code,temp_stop_yn,short_over_yn,iscd_stat_cls_code';
const ROWS = [
  '2026-11-12T08:03:05,010060,NX,207500,9035,1735441900,209500,210000,204500,206079.90,5.54,N,00,N,N,55',
  '2026-11-12T09:07:06,068760,J,41800,5463,226435750,41050,42000,40550,41448.97,1.21,N,00,N,N,55',
  '2026-11-12T09:08:52,030200,J,51200,17265,878927100,51100,51200,50400,50908.03,1.19,N,00,N,N,55',
  '2026-11-12T09:08:54,036570,J,192600,30909,6097278800,209000,209500,191900,197265.48,-7.85,N,00,N,N,55',
  '2026-11-12T09:09:10,010060,J,205000,40000,8200000000,209500,210000,204000,205900.00,4.27,N,00,N,N,55',
];
const kst = (hms: string) => Date.parse(`2026-11-12T${hms}+09:00`);

describe('change in won from price and rate', () => {
  it('KRX tick table', () => {
    expect([1999, 2000, 4999, 5000, 19990, 20000, 49950, 50000, 199900, 200000, 499500, 500000].map(krxTick)).toEqual([1, 5, 5, 10, 10, 50, 50, 100, 100, 500, 500, 1000]);
  });

  it('previous close lands on the tick grid', () => {
    expect(quoteChange(11650, 0.52)).toBe(60); // 11,590 -> 11,650
    expect(quoteChange(192600, -7.85)).toBe(-16400); // NC: 209,000 -> 192,600
    expect(quoteChange(51200, 1.19)).toBe(600); // KT: 50,600 -> 51,200
    expect(quoteChange(41800, 1.21)).toBe(500); // 41,300 -> 41,800
    expect(quoteChange(2490, 0)).toBe(0);
    expect(quoteChange(0, 1)).toBe(0);
    expect(quoteChange(1000, -100)).toBe(0);
  });

  it('display text', () => {
    expect(quoteText({ price: 11650, pct: 0.52, change: 60 })).toEqual({ price: '11,650', change: '▲ 60', pct: '(+0.52%)', dir: 1 });
    expect(quoteText({ price: 192600, pct: -7.85, change: -16400 })).toEqual({ price: '192,600', change: '▼ 16,400', pct: '(-7.85%)', dir: -1 });
    expect(quoteText({ price: 2490, pct: 0, change: 0 })).toEqual({ price: '2,490', change: '0', pct: '(0.00%)', dir: 0 });
  });
});

describe('quote log rows', () => {
  it('parses by header name and skips broken rows', () => {
    const q = parseQuoteRows([...ROWS, '', 'garbage', '2026-11-12T09:10:00,BAD,J,100,1,1,1,1,1,1,0.1', '2026-11-12T09:10:00,005930,J,0,1,1,1,1,1,1,0.1'], quoteHeader(HEADER));
    expect(q).toHaveLength(5);
    expect(q[0]).toMatchObject({ code: '010060', t: kst('08:03:05'), price: 207500, pct: 5.54, market: 'NX' });
    expect(q[3].change).toBe(-16400);
  });

  it('[2026-11-16] KIS quote answer as a LiveQuote', () => {
    const q = fromMarketQuote({ code: '403870', asof: '2026-11-16T10:21:30', price: 37800, change: -1650, change_pct: -4.18, prev_close: 39450,
      open: 41000, high: 41000, low: 37600, volume: 812345, amount: 31234567890 });
    expect(q).toMatchObject({ code: '403870', t: Date.parse('2026-11-16T10:21:30+09:00'), price: 37800, change: -1650, pct: -4.18, source: 'kis',
      open: 41000, high: 41000, low: 37600, vol: 812345, value: 31234567890 });
    expect(fromMarketQuote({ code: '403870', price: 37800, prev_close: 40000 })).toMatchObject({ pct: -5.5, change: -2200 });
    expect(fromMarketQuote({ code: '403870', price: 0 })).toBeNull();
  });

  it('[2026-11-16] keeps the day open / high / low / volume / value', () => {
    const [q] = parseQuoteRows(ROWS.slice(3, 4), quoteHeader(HEADER));
    expect(q).toMatchObject({ open: 209000, high: 209500, low: 191900, vol: 30909, value: 6097278800 });
    const [old] = parseQuoteRows(['2026-11-12T09:10:00,005930,J,100,0.1'], quoteHeader('ts,code,market,stck_prpr,prdy_ctrt'));
    expect(old.open).toBeUndefined();
    expect([fmtKrwShort(6097278800), fmtKrwShort(272823500), fmtKrwShort(34_500_000), fmtKrwShort(9800), fmtKrwShort(1.2e12)]).toEqual(['61억', '2.7억', '3,450만', '9,800', '1.2조']);
  });

  it('newest per code, and the newest at or before a replay time', () => {
    const book = new QuoteBook();
    book.add(parseQuoteRows(ROWS, quoteHeader(HEADER)));
    expect(book.latest()['010060'].price).toBe(205000);
    const early = book.latest(kst('09:07:30'));
    expect(early['010060'].price).toBe(207500); // the 08:03 NXT row
    expect(Object.keys(early).sort()).toEqual(['010060', '068760']);
    expect(book.latest(kst('08:00:00'))).toEqual({});
  });
});

describe('PriceService.liveQuotes', () => {
  it('reads the growing file by byte offset and restarts when it shrinks', async () => {
    let text = `${HEADER}\n${ROWS.slice(0, 2).join('\n')}\n${ROWS[2].slice(0, 30)}`; // last row still being written
    const reads: [number, number][] = [];
    const host: FileHost = {
      kind: 'local',
      connected: true,
      readdir: async () => [],
      stat: async (p) => (p.endsWith('quote_ticks_20261112.csv') ? { size: Buffer.byteLength(text), mtime: 0 } : null),
      readRange: async (_p, s, e) => (reads.push([s, e]), Buffer.from(text).subarray(s, e)),
      listPrefix: async () => [],
    };
    const svc = new PriceService(host, () => '/root');
    let q = await svc.liveQuotes('20261112');
    expect(Object.keys(q).sort()).toEqual(['010060', '068760']);
    const firstEnd = Buffer.byteLength(text);
    text = `${HEADER}\n${ROWS.join('\n')}\n`;
    q = await svc.liveQuotes('20261112');
    expect(q['030200'].price).toBe(51200); // the half-written row completed
    expect(q['010060'].price).toBe(205000);
    expect(reads[1][0]).toBe(firstEnd); // only the appended bytes
    expect((await svc.liveQuotes('20261112', kst('09:07:30')))['010060'].price).toBe(207500);
    text = `${HEADER}\n${ROWS[1]}\n`; // rewritten file
    q = await svc.liveQuotes('20261112');
    expect(Object.keys(q)).toEqual(['068760']);
    expect(await svc.liveQuotes('20261113')).toEqual({}); // no file for that day
    expect(await svc.liveQuotes('bad')).toEqual({});
  });
});
