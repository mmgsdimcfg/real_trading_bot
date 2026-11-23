/*
 * Update log
 * - [2026-11-19] type=test - Chart cache (memory + disk store): a cached chart answers without KIS calls (stale when old), day+ refresh
 *   merges the newest page and reloads everything when older adjusted prices changed; minute refresh asks only today's newest
 *   minutes and never the cached past days; a new day asks only the day that became past. Watchlist picks files: parsing and the
 *   MMDD tab rules
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ app: { getPath: () => '.' }, safeStorage: {} }));

import { KisMarket } from '../src/main/kisMarket';
import type { ChartStore } from '../src/main/chartCache';
import { parsePicks, PICKS_RE } from '../src/main/watchlist';
import { mergePicks } from '../src/shared/watchPicks';

const at = (iso: string) => vi.setSystemTime(new Date(`${iso}+09:00`));
afterEach(() => vi.useRealTimers());

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

const pad = (n: number) => String(n).padStart(2, '0');

/** weekdays (YYYYMMDD) from `from` up to `to` */
function weekdays(from: string, to: string): string[] {
  const out: string[] = [];
  for (let d = new Date(`${from.slice(0, 4)}-${from.slice(4, 6)}-${from.slice(6)}T00:00:00Z`); ; d = new Date(d.getTime() + 86400000)) {
    const s = d.toISOString().slice(0, 10).replace(/-/g, '');
    if (s > to) break;
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) out.push(s);
  }
  return out;
}

/** fake KIS server: daily rows and J minutes 09:00 - 15:30 of the trading days; "now" from the fake clock */
function server(opts: { days: string[]; scale?: () => number }) {
  const calls: { api: string; params: Record<string, string> }[] = [];
  const nowKst = () => new Date(Date.now() + 9 * 3600000).toISOString();
  const minutes = (d: string) => {
    const out: string[] = [];
    for (let m = 9 * 60; m <= 15 * 60 + 30; m++) out.push(`${pad(Math.floor(m / 60))}${pad(m % 60)}00`);
    const today = nowKst().slice(0, 10).replace(/-/g, '');
    const hhmm = nowKst().slice(11, 16).replace(':', '');
    return d === today ? out.filter((h) => h.slice(0, 4) <= hhmm) : d < today ? out : [];
  };
  const minRow = (d: string, h: string) => ({ stck_bsop_date: d, stck_cntg_hour: h, stck_prpr: String(1000 + +h.slice(0, 4)), stck_oprc: '1000', stck_hgpr: '3000', stck_lwpr: '900', cntg_vol: '10' });
  const client = {
    env: 'real',
    get: async (api: string, params: Record<string, string>) => {
      calls.push({ api, params });
      if (api === 'inquire_daily_itemchartprice') {
        const k = opts.scale?.() ?? 1;
        const rows = opts.days
          .filter((d) => d <= params.FID_INPUT_DATE_2)
          .reverse()
          .slice(0, 100)
          .map((d, i) => ({ stck_bsop_date: d, stck_clpr: String((10000 + i) * k), stck_oprc: '1', stck_hgpr: '1', stck_lwpr: '1', acml_vol: '5' }));
        return { output1: { hts_kor_isnm: 'X' }, output2: rows };
      }
      if (api === 'inquire_time_itemchartprice') {
        const today = nowKst().slice(0, 10).replace(/-/g, '');
        const rows = minutes(today)
          .filter((h) => h <= params.FID_INPUT_HOUR_1)
          .reverse()
          .slice(0, 30)
          .map((h) => minRow(today, h));
        return { output1: { hts_kor_isnm: 'X' }, output2: rows };
      }
      if (api === 'inquire_time_dailychartprice') {
        const rows: Record<string, string>[] = [];
        for (const d of [...opts.days].reverse()) {
          if (d > params.FID_INPUT_DATE_1) continue;
          for (const h of [...minutes(d)].reverse()) if (d < params.FID_INPUT_DATE_1 || h <= params.FID_INPUT_HOUR_1) rows.push(minRow(d, h));
          if (rows.length >= 120) break;
        }
        return { output2: rows.slice(0, 120) };
      }
      return { output: [] };
    },
  };
  return { calls, client };
}

describe('day chart cache', () => {
  it('serves memory / disk without calls, refreshes with one page, reloads when adjusted prices change', async () => {
    vi.useFakeTimers();
    at('2026-11-19T12:00:00');
    let scale = 1;
    const days = weekdays('20250101', '20261119');
    const s = server({ days, scale: () => scale });
    const disk = new MemStore();
    const m = new KisMarket(s.client as never, () => false, disk);
    const r1 = await m.chart({ code: '005930', kind: 'day', n: 1 });
    const full = s.calls.length;
    expect(full).toBeGreaterThan(1);
    expect(r1.bars.length).toBeGreaterThan(300);

    // memory, young: no call, not stale
    const r2 = await m.chart({ code: '005930', kind: 'day', n: 1 });
    expect(s.calls.length).toBe(full);
    expect(r2.stale).toBeUndefined();

    // another run (new backend) with the disk copy, two minutes later: answered at once, marked stale
    at('2026-11-19T12:02:00');
    const m2 = new KisMarket(s.client as never, () => false, disk);
    const r3 = await m2.chart({ code: '005930', kind: 'day', n: 1 });
    expect(s.calls.length).toBe(full);
    expect(r3.stale).toBe(true);
    expect(r3.bars.length).toBe(r1.bars.length);

    // refresh: newest page only
    await m2.chart({ code: '005930', kind: 'day', n: 1, refresh: true });
    expect(s.calls.length).toBe(full + 1);

    // a split: older adjusted closes halve -> whole history again
    scale = 0.5;
    const before = s.calls.length;
    const r4 = await m2.chart({ code: '005930', kind: 'day', n: 1, refresh: true });
    expect(s.calls.length - before).toBeGreaterThan(1);
    expect(r4.bars[0].c).toBeLessThan(r1.bars[0].c);
  });
});

describe('minute chart cache', () => {
  it('refresh asks only the newest minutes of today; a new day asks only the day that became past', async () => {
    vi.useFakeTimers();
    at('2026-11-19T10:00:30');
    const days = weekdays('20261101', '20261120');
    const s = server({ days });
    const disk = new MemStore();
    const m = new KisMarket(s.client as never, () => false, disk);
    const r1 = await m.chart({ code: '005930', kind: 'min', n: 1, days: 3 });
    const dates = (bars: { t: number }[]) => [...new Set(bars.map((b) => new Date(b.t + 9 * 3600000).toISOString().slice(0, 10)))];
    expect(dates(r1.bars)).toEqual(['2026-11-17', '2026-11-18', '2026-11-19']);
    expect(r1.bars.filter((b) => new Date(b.t + 9 * 3600000).toISOString().startsWith('2026-11-18')).length).toBe(391);

    // five minutes later: one call for today's newest page, none for 11/17 - 11/18
    at('2026-11-19T10:05:30');
    const before = s.calls.length;
    const r2 = await m.chart({ code: '005930', kind: 'min', n: 1, days: 3, refresh: true });
    expect(s.calls.slice(before).map((c) => c.api)).toEqual(['inquire_time_itemchartprice']);
    expect(r2.bars.length).toBe(r1.bars.length + 5);

    // next day, new run from disk: the cached chart answers at once (stale), the refresh asks today + 11/19 only
    at('2026-11-20T09:20:30');
    const m2 = new KisMarket(s.client as never, () => false, disk);
    const b3 = s.calls.length;
    const r3 = await m2.chart({ code: '005930', kind: 'min', n: 1, days: 3 });
    expect(r3.stale).toBe(true);
    expect(s.calls.length).toBe(b3);
    const r4 = await m2.chart({ code: '005930', kind: 'min', n: 1, days: 3, refresh: true });
    expect(dates(r4.bars)).toEqual(['2026-11-18', '2026-11-19', '2026-11-20']);
    const asked = s.calls.slice(b3).filter((c) => c.api === 'inquire_time_dailychartprice').map((c) => c.params.FID_INPUT_DATE_1);
    expect(asked.every((d) => d >= '20261119')).toBe(true); // 11/18 came from the cache
    expect(r4.bars.filter((b) => new Date(b.t + 9 * 3600000).toISOString().startsWith('2026-11-19')).length).toBe(391);
  });
});

describe('watchlist picks files', () => {
  it('parses "code,name" lines and matches the file names', () => {
    expect(parsePicks('﻿067920,이글루\r\n332570,PS일렉트로닉스\r\n\r\ncode,name\n145020 휴젤\n067920,dup\n')).toEqual(['067920', '332570', '145020']);
    expect(PICKS_RE.exec('_20261119_picks.txt')?.[1]).toBe('20261119');
    expect(PICKS_RE.exec('20261119_picks.txt')?.[1]).toBe('20261119');
    expect(PICKS_RE.exec('20261119_picks.txt.bak')).toBeNull();
  });

  it('makes an MMDD tab once, refreshes it when the file changes, does not bring back a deleted tab', () => {
    let n = 0;
    const id = () => `n${++n}`;
    const g0 = [{ id: 'g1', name: '관심 1', codes: ['005930'] }];
    const f = { file: '_20261119_picks.txt', date: '20261119', mtime: 100, codes: ['067920', '332570'] };
    const a = mergePicks(g0, {}, [f], id, 60);
    expect(a.added).toEqual(['11/19']);
    expect(a.groups[1]).toMatchObject({ name: '11/19', codes: ['067920', '332570'], src: f.file });
    // same file again: nothing
    expect(mergePicks(a.groups, a.imported, [f], id, 60).changed).toBe(false);
    // the user deletes the tab: not made again while the file is unchanged
    const del = a.groups.slice(0, 1);
    expect(mergePicks(del, a.imported, [f], id, 60).changed).toBe(false);
    // the file changes: the tab is made again with the new codes
    const c = mergePicks(del, a.imported, [{ ...f, mtime: 200, codes: ['145020'] }], id, 60);
    expect(c.groups[1]).toMatchObject({ name: '11/19', codes: ['145020'] });
    // an existing tab keeps its id when its file changes
    const d = mergePicks(c.groups, c.imported, [{ ...f, mtime: 300, codes: ['000660'] }], id, 60);
    expect(d.groups[1].id).toBe(c.groups[1].id);
    expect(d.groups[1].codes).toEqual(['000660']);
    // cap at the group limit
    const many = { ...f, file: '_20261120_picks.txt', date: '20261120', codes: Array.from({ length: 80 }, (_, i) => String(100000 + i)) };
    expect(mergePicks(g0, {}, [many], id, 60).groups[1].codes).toHaveLength(60);
  });
});
