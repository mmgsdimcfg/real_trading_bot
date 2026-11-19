/*
 * Update log
 * - [2026-11-12] type=test - price service before the KRX open: a tick chart without a tick file shows the day's 1-minute bars, a
 *   minute chart of an NXT symbol gets the combined-market minutes of the hours KRX is closed (they win over KRX pre-open prints),
 *   an NXT symbol's tick chart with pre-open prints only falls back to minute bars, other symbols keep their ticks
 * - [2026-11-19] type=test - minutes outside the KRX regular session for EVERY symbol (r016 `bars --source kis-ext`): the KRX
 *   after-hours market of a KRX-only symbol (403870 HPSP), the CSAT-day session from the server's hours, a known KRX-only symbol
 *   is not asked before the open, an older server without kis-ext falls back to kis-un for NXT symbols
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, dialog: {}, ipcMain: { handle: () => undefined, on: () => undefined } }));

import { mergeOutsideKrx, outsideKrxHours, PriceService } from '../src/main/prices';
import type { FileHost } from '../src/main/host';
import type { R016Client } from '../src/main/control';
import type { Bar } from '../src/shared/price';
import { kstToEpoch } from '../src/shared/time';

const DATE = '20261112';
const at = (hm: string) => kstToEpoch('2026-11-12', `${hm}:00`);
const bar = (hm: string, c: number, v = 10): Bar => ({ t: at(hm), o: c, h: c, l: c, c, v });
const row = (hm: string, c: number, v = 10) => [`2026-11-12T${hm}:00`, c, c, c, c, v];

const TICK_HEADER = 'recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,bidp1,hour_cls_code,venue,session';
const tick = (hms: string, price: number, vol: number, session: string) => `2026-11-12T00:00:00.000,010060,${DATE},${hms},${price},${vol},${vol},3,0,0,0,0,0,KRX,${session}`;

/** files: path -> text (only the tick file matters here) */
function host(files: Record<string, string>): FileHost {
  return {
    kind: 'local',
    connected: true,
    readdir: async () => [],
    stat: async (p) => (p in files ? { size: Buffer.byteLength(files[p]), mtime: 0 } : null),
    readRange: async (p, s, e) => Buffer.from(files[p]).subarray(s, e),
    listPrefix: async () => [],
  };
}

type Hours = { regular: [string, string]; nxt_pre: [string, string] | null; nxt_after: [string, string] | null; krx_after: [string, string] | null };
/** fake r016: `bars` answers by --source (auto -> `auto`, kis-un -> `un`, kis-ext -> `ext`; oldServer: no kis-ext choice) */
function r016(answers: { auto?: (string | number)[][]; un?: (string | number)[][]; ext?: (string | number)[][]; nxt?: boolean; hours?: Hours; oldServer?: boolean }) {
  const calls: string[][] = [];
  const client = {
    run: async (args: string[]) => {
      calls.push(args);
      if (args.includes('kis-ext') && answers.oldServer) return { ok: false, error: "argument --source: invalid choice: 'kis-ext'" };
      const src = args.includes('kis-ext') ? 'ext' : args.includes('kis-un') ? 'un' : args.includes('kis') ? 'kis' : 'auto';
      const bars = src === 'ext' ? answers.ext : src === 'un' ? answers.un : src === 'auto' ? answers.auto : undefined;
      if (!bars) return { ok: false, error: 'no bars' };
      return {
        ok: true,
        data: { ok: true, source: 'kis:inquire_time_itemchartprice:UN', columns: ['t', 'o', 'h', 'l', 'c', 'v'], bars, nxt: answers.nxt, hours: answers.hours },
      };
    },
  };
  return { calls, client: client as unknown as R016Client };
}
const CSAT: Hours = { regular: ['10:00', '16:30'], nxt_pre: null, nxt_after: ['16:40', '20:00'], krx_after: ['17:00', '20:00'] };

const tickPath = `/root/runtime/data/ticks/${DATE}/010060.csv`;

describe('bars of the hours KRX is closed', () => {
  it('outsideKrxHours keeps bars before 09:00 and from 15:30', () => {
    const kept = outsideKrxHours([bar('08:00', 1), bar('08:59', 1), bar('09:00', 1), bar('15:29', 1), bar('15:30', 1), bar('19:59', 1)]);
    expect(kept.map((b) => b.t)).toEqual([at('08:00'), at('08:59'), at('15:30'), at('19:59')]);
  });

  it('mergeOutsideKrx: the combined bar wins in its minute, the day bars stay elsewhere', () => {
    const merged = mergeOutsideKrx([bar('08:30', 100), bar('09:00', 101)], [bar('08:00', 200), bar('08:30', 201)]);
    expect(merged.map((b) => [b.t, b.c])).toEqual([
      [at('08:00'), 200],
      [at('08:30'), 201],
      [at('09:00'), 101],
    ]);
  });
});

describe('price service before the KRX open', () => {
  it('tick chart without a tick file: the day 1-minute bars with a note', async () => {
    const s = r016({ auto: [row('08:00', 206500), row('08:01', 207500)] });
    const svc = new PriceService(host({}), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'tick', n: 30 }, true);
    expect(r.ok).toBe(true);
    expect(r.bars.map((b) => b.c)).toEqual([206500, 207500]);
    expect(r.note).toContain('1분봉으로 표시');
    const args = s.calls[0];
    expect(args.slice(args.indexOf('--kind'), args.indexOf('--kind') + 4)).toEqual(['--kind', 'min', '--interval', '1']);
  });

  it('minute chart without a tick file asks the server (combined minutes come from its fallback)', async () => {
    const s = r016({ auto: [row('08:00', 206500)] });
    const svc = new PriceService(host({}), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1 }, true);
    expect(r.ok && r.bars.length).toBe(1);
    expect(r.note).toContain('KIS 시세 조회');
  });

  it('minute chart of an NXT symbol with a tick file: combined minutes outside KRX hours are merged and win there', async () => {
    const files = { [tickPath]: [TICK_HEADER, tick('083001', 204000, 5, 'pre'), tick('090005', 205000, 7, 'regular')].join('\n') + '\n' };
    const s = r016({ ext: [row('08:00', 206500), row('08:30', 204500), row('09:00', 999999)], nxt: true });
    const svc = new PriceService(host(files), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1, nxt: true }, false);
    expect(r.bars.map((b) => [b.t, b.c])).toEqual([
      [at('08:00'), 206500],
      [at('08:30'), 204500], // the combined bar, not the KRX pre-open print
      [at('09:00'), 205000], // KRX hours: the tick bar (what the bot saw)
    ]);
    expect(r.note).toContain('KIS 통합 분봉');
    expect(r.nxt).toBe(true);
    expect(s.calls.some((a) => a.includes('kis-ext'))).toBe(true);
  });

  it('no minutes outside the regular session on the server: the tick file is used as it is', async () => {
    const files = { [tickPath]: [TICK_HEADER, tick('083001', 204000, 5, 'pre'), tick('090005', 205000, 7, 'regular')].join('\n') + '\n' };
    const s = r016({ ext: [], nxt: false });
    const svc = new PriceService(host(files), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1 }, false);
    expect(r.bars.map((b) => b.c)).toEqual([204000, 205000]);
    expect(r.nxt).toBe(false);
    expect(s.calls.some((a) => a.includes('kis-un'))).toBe(false);
  });

  it('KRX-only symbol (HPSP): the KRX after-hours minutes are added to the tick bars', async () => {
    const files = { [tickPath]: [TICK_HEADER, tick('090005', 205000, 7, 'regular'), tick('152959', 205500, 7, 'regular')].join('\n') + '\n' };
    const s = r016({ ext: [row('16:00', 206000), row('19:59', 207000)], nxt: false });
    const svc = new PriceService(host(files), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1 }, false);
    expect(r.bars.map((b) => [b.t, b.c])).toEqual([
      [at('09:00'), 205000],
      [at('15:29'), 205500],
      [at('16:00'), 206000],
      [at('19:59'), 207000],
    ]);
    expect(r.note).toContain('KRX 장후 2분');
    expect(r.nxt).toBe(false);
  });

  it("CSAT day: the server's hours decide what is outside the regular session (10:00 - 16:30)", async () => {
    const files = { [tickPath]: [TICK_HEADER, tick('100005', 205000, 7, 'regular'), tick('160005', 205500, 7, 'regular')].join('\n') + '\n' };
    // 16:00 is inside the CSAT-day regular session: the tick bar stays; 17:00 is the KRX after-hours market
    const s = r016({ ext: [row('16:00', 999999), row('17:00', 206000)], nxt: false, hours: CSAT });
    const svc = new PriceService(host(files), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1 }, false);
    expect(r.bars.map((b) => [b.t, b.c])).toEqual([
      [at('10:00'), 205000],
      [at('16:00'), 205500],
      [at('17:00'), 206000],
    ]);
    expect(r.hours?.regular).toEqual(['10:00', '16:30']);
  });

  it('older server without kis-ext: NXT symbols fall back to the combined minutes (kis-un)', async () => {
    const files = { [tickPath]: [TICK_HEADER, tick('090005', 205000, 7, 'regular')].join('\n') + '\n' };
    const s = r016({ un: [row('08:00', 206500)], oldServer: true });
    const svc = new PriceService(host(files), () => '/root', () => s.client);
    const r = await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1, nxt: true }, false);
    expect(r.bars.map((b) => b.c)).toEqual([206500, 205000]);
    expect(s.calls.some((a) => a.includes('kis-un'))).toBe(true);
    const krx = r016({ un: [row('08:00', 206500)], oldServer: true });
    const r2 = await new PriceService(host(files), () => '/root', () => krx.client).get({ date: DATE, code: '010060', kind: 'min', n: 1 }, false);
    expect(r2.bars.map((b) => b.c)).toEqual([205000]);
    expect(krx.calls.some((a) => a.includes('kis-un'))).toBe(false);
  });

  it('tick chart with pre-open prints only: minute bars for an NXT symbol, the prints for the others', async () => {
    const files = { [tickPath]: [TICK_HEADER, tick('083001', 204000, 5, 'pre'), tick('083023', 204000, 3, 'pre')].join('\n') + '\n' };
    const s = r016({ auto: [row('08:00', 206500), row('08:01', 207500), row('08:02', 207000)] });
    const nxt = await new PriceService(host(files), () => '/root', () => s.client).get({ date: DATE, code: '010060', kind: 'tick', n: 1, nxt: true }, true);
    expect(nxt.bars).toHaveLength(3);
    expect(nxt.note).toContain('1분봉으로 표시');
    const krx = await new PriceService(host(files), () => '/root', () => s.client).get({ date: DATE, code: '010060', kind: 'tick', n: 1 }, true);
    expect(krx.source).toBe('ticks');
    expect(krx.bars).toHaveLength(2);
  });
});

describe('KRX-only symbol before the KRX open', () => {
  it('a code the server flagged KRX-only is not asked for outside minutes before the open (no pre-market)', async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(at('16:10'));
      const day2 = '20261113';
      const files = {
        [tickPath]: [TICK_HEADER, tick('090005', 205000, 7, 'regular')].join('\n') + '\n',
        [`/root/runtime/data/ticks/${day2}/010060.csv`]: [TICK_HEADER, tick('083001', 204000, 5, 'pre').replace(DATE, day2)].join('\n') + '\n',
      };
      const s = r016({ ext: [row('16:00', 206000)], nxt: false });
      const svc = new PriceService(host(files), () => '/root', () => s.client);
      await svc.get({ date: DATE, code: '010060', kind: 'min', n: 1 }, true);
      expect(s.calls.filter((a) => a.includes('kis-ext'))).toHaveLength(1);
      vi.setSystemTime(kstToEpoch('2026-11-13', '08:30:00'));
      await svc.get({ date: day2, code: '010060', kind: 'min', n: 1 }, true);
      expect(s.calls.filter((a) => a.includes('kis-ext'))).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
