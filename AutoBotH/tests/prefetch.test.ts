/*
 * Update log
 * - [2026-11-19] type=test - Background fetch (renderer prefetch.ts): the shared request keys equal the keys the timeline / chart
 *   views used before, one pass fetches the focus symbols first then every watched symbol (no quotes for the watched ones), a
 *   second pass inside the interval asks nothing, a hidden window or a lost connection asks nothing, failures wait FAIL_MS
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

const calls: string[] = [];
let priceOk = true;
const g = globalThis as unknown as { window: unknown; document: unknown };
g.document = { hidden: false, addEventListener: () => undefined };
g.window = {
  api: {
    getPrice: async (r: { code: string; kind: string }) => (calls.push(`price ${r.code} ${r.kind}`), { ok: priceOk, bars: [] }),
    getTimeline: async (code: string) => (calls.push(`events ${code}`), { events: [], trades: [] }),
    getQuote: async (code: string) => (calls.push(`quote ${code}`), { ok: true, data: { code } }),
  },
};

import { chartPriceFetch, timelinePriceFetch, __prefetchTest } from '../src/renderer/src/prefetch';
import { useStore } from '../src/renderer/src/store';
import { clearAll, peek } from '../src/renderer/src/dataCache';

const DAY = '20261119';
const chart = { kind: 'min', tickN: 30, minN: 1, ind: ['ma'] };

describe('shared keys', () => {
  it('timeline key = the view key of before', () => {
    const f = timelinePriceFetch(DAY, '005930', chart, true);
    expect(f.key).toBe(`price:tl/${DAY}/005930/min/1/wx`);
    expect(f.req).toEqual({ date: DAY, code: '005930', kind: 'min', n: 1, warmup: true, nxt: true });
    expect(timelinePriceFetch(DAY, '005930', { ...chart, ind: [] }, false).key).toBe(`price:tl/${DAY}/005930/min/1/`);
  });
  it('chart keys = the view keys of before', () => {
    const f = chartPriceFetch(DAY, '005930', { ...chart, kind: 'tick' }, false);
    expect(f.main.key).toBe(`price:ch/${DAY}/005930/tick/30//w`);
    expect(f.main.req).toEqual({ date: DAY, code: '005930', kind: 'tick', n: 30, warmup: true, withName: true, nxt: false });
    expect(f.day?.req).toEqual({ date: DAY, code: '005930', kind: 'day', n: 1 });
    const d = chartPriceFetch(DAY, '005930', { ...chart, kind: 'day' }, true);
    expect(d.main.key).toBe(`price:ch/${DAY}/005930/day/1/x/`);
    expect(d.day).toBeNull();
    expect(d.dayKey).toBe(`price:ch/${DAY}/005930/day`);
  });
});

describe('loop', () => {
  beforeEach(() => {
    calls.length = 0;
    priceOk = true;
    clearAll();
    __prefetchTest.tried.clear();
    (g.document as { hidden: boolean }).hidden = false;
    useStore.setState({
      status: { conn: 'connected', mode: 'local', date: DAY, live: true },
      snapshot: { date: DAY, symbols: [{ code: '000020' }, { code: '005930', nxt: true }, { code: '035720' }] } as never,
      selected: '005930',
      chart: chart as never,
      chartTab: { ...chart, kind: 'day', prevDay: true } as never,
      chartCode: { day: DAY, code: '068270' },
    });
  });

  it('focus first, then every watched symbol; quote only for the chart symbol', async () => {
    await __prefetchTest.tick();
    expect(calls).toEqual([
      'price 005930 min',
      'events 005930',
      'price 068270 day',
      'quote 068270',
      'price 000020 min',
      'events 000020',
      'price 035720 min',
      'events 035720',
    ]);
    expect(peek(`price:tl/${DAY}/005930/min/1/wx`)).toBeTruthy();
    expect(peek(`price:ch/${DAY}/068270/day/1//`)).toBeTruthy();
    expect(peek(`timeline:${DAY}/000020`)).toBeTruthy();
    expect(useStore.getState().timeline?.code).toBe('005930');
  });

  it('nothing again inside the interval', async () => {
    await __prefetchTest.tick();
    calls.length = 0;
    await __prefetchTest.tick();
    expect(calls).toEqual([]);
  });

  it('hidden window / no connection: nothing', async () => {
    (g.document as { hidden: boolean }).hidden = true;
    await __prefetchTest.tick();
    (g.document as { hidden: boolean }).hidden = false;
    useStore.setState({ status: { conn: 'error', date: DAY, live: true } as never });
    await __prefetchTest.tick();
    expect(calls).toEqual([]);
  });

  it('a failed price request waits FAIL_MS (60 s), not the 10 s live interval', async () => {
    priceOk = false;
    const t0 = Date.now();
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(t0);
    await __prefetchTest.tick();
    calls.length = 0;
    now.mockReturnValue(t0 + 15000);
    await __prefetchTest.tick();
    expect(calls.filter((c) => c === 'price 005930 min')).toEqual([]);
    now.mockReturnValue(t0 + 61000);
    await __prefetchTest.tick();
    expect(calls).toContain('price 005930 min');
    now.mockRestore();
  });
});
