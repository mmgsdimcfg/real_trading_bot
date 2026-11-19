/*
 * Update log
 * - [2026-11-18] type=test - Reuse of fetched data: the renderer's dataCache (keep / limit per kind / drop / clear), coversGaps,
 *   the price service keeps the NXT combined minutes of the day from 09:00 to the KRX close (one r016 call, not one a minute) and
 *   asks again after 15:30, and a symbol's tick file is not read from byte 0 again after a dozen other symbols were opened
 * - [2026-11-19] type=test - price limit 160 (raised for the background fetch, prefetch.ts)
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, dialog: {}, ipcMain: { handle: () => undefined, on: () => undefined } }));

import { clearAll, drop, peek, put, storedAt } from '../src/renderer/src/dataCache';
import { coversGaps, PriceService } from '../src/main/prices';
import type { FileHost } from '../src/main/host';
import type { R016Client } from '../src/main/control';
import type { Bar } from '../src/shared/price';
import { kstToEpoch } from '../src/shared/time';

const DATE = '20261112';
const at = (hm: string) => kstToEpoch('2026-11-12', `${hm}:00`);
const bar = (hm: string, c = 1): Bar => ({ t: at(hm), o: c, h: c, l: c, c, v: 1 });
const row = (hm: string, c: number) => [`2026-11-12T${hm}:00`, c, c, c, c, 10];
const TICK_HEADER = 'recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,bidp1,hour_cls_code,venue,session';
const tickText = (code: string) =>
  [TICK_HEADER, `2026-11-12T00:00:00.000,${code},${DATE},090005,1000,7,7,3,0,0,0,0,0,KRX,regular`, `2026-11-12T00:00:00.000,${code},${DATE},090105,1010,3,10,3,0,0,0,0,0,KRX,regular`].join('\n') + '\n';

afterEach(() => {
  vi.useRealTimers();
  clearAll();
});

describe('renderer dataCache', () => {
  it('keeps the last value per key, drops the oldest past the limit of its kind', () => {
    put('price:a', 1);
    put('price:a', 2);
    expect(peek('price:a')).toBe(2);
    expect(storedAt('price:a')).toBeGreaterThan(0);
    for (let i = 0; i < 160; i++) put(`price:x${i}`, i); // limit 160 for price
    expect(peek('price:a')).toBeUndefined();
    expect(peek('price:x79')).toBe(79);
    put('jobs:list', { n: 1 });
    expect(peek('jobs:list')).toEqual({ n: 1 }); // another kind is not pushed out by price entries
  });

  it('drop(prefix) and clearAll()', () => {
    put('collector:20261117', 1);
    put('collector:20261118', 2);
    put('paper:20261118', 3);
    drop('collector:');
    expect(peek('collector:20261117')).toBeUndefined();
    expect(peek('paper:20261118')).toBe(3);
    clearAll();
    expect(peek('paper:20261118')).toBeUndefined();
  });
});

describe('coversGaps', () => {
  it('true only when every gap minute has a bar', () => {
    const gaps: [number, number][] = [[at('09:10'), at('09:13')]];
    expect(coversGaps([bar('09:10'), bar('09:11'), bar('09:12')], gaps)).toBe(true);
    expect(coversGaps([bar('09:10'), bar('09:12')], gaps)).toBe(false);
    expect(coversGaps([], [])).toBe(true);
  });
});

function host(files: Record<string, string>, reads: [string, number][]): FileHost {
  return {
    kind: 'local',
    connected: true,
    readdir: async () => [],
    stat: async (p) => (p in files ? { size: Buffer.byteLength(files[p]), mtime: 0 } : null),
    readRange: async (p, s, e) => {
      reads.push([p, s]);
      return Buffer.from(files[p]).subarray(s, e);
    },
    listPrefix: async () => [],
  };
}

function r016() {
  const calls: string[][] = [];
  const client = {
    run: async (args: string[]) => {
      calls.push(args);
      return { ok: true, data: { ok: true, source: 'kis:UN', columns: ['t', 'o', 'h', 'l', 'c', 'v'], bars: [row('08:00', 900), row('08:30', 901)] } };
    },
  };
  return { calls, client: client as unknown as R016Client };
}

const tickPath = (code: string) => `/root/runtime/data/ticks/${DATE}/${code}.csv`;

describe('price service reuse', () => {
  it('NXT combined minutes: one r016 call from 09:00 to the KRX close, again after 15:30', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(at('10:00'));
    const s = r016();
    const svc = new PriceService(host({ [tickPath('010060')]: tickText('010060') }, []), () => '/root', () => s.client);
    const req = { date: DATE, code: '010060', kind: 'min' as const, n: 1, nxt: true };
    await svc.get(req, true);
    vi.setSystemTime(at('11:30')); // 90 minutes later - the old 60 s reuse would have asked again
    await svc.get(req, true);
    expect(s.calls.filter((a) => a.includes('kis-un'))).toHaveLength(1);
    vi.setSystemTime(at('15:35')); // after-market minutes start: ask again
    await svc.get(req, true);
    expect(s.calls.filter((a) => a.includes('kis-un'))).toHaveLength(2);
  });

  it('a symbol opened again after 12 others is not read from byte 0 (tick cache holds 40 entries)', async () => {
    const codes = Array.from({ length: 13 }, (_, i) => String(100000 + i));
    const files = Object.fromEntries(codes.map((c) => [tickPath(c), tickText(c)]));
    const reads: [string, number][] = [];
    const svc = new PriceService(host(files, reads), () => '/root', () => null);
    for (const c of codes) await svc.get({ date: DATE, code: c, kind: 'tick', n: 1 }, true);
    const before = reads.filter(([p]) => p === tickPath(codes[0])).length;
    await svc.get({ date: DATE, code: codes[0], kind: 'tick', n: 1 }, true);
    expect(reads.filter(([p]) => p === tickPath(codes[0])).length).toBe(before); // file unchanged: no read at all
  });
});
