// [2026-11-23] bot r004 -> watchlist tab of the trading day it is for, at the front
import { describe, expect, it } from 'vitest';
import { mergePicks, r004Entry, r004Src } from '../src/shared/watchPicks';

const doc = (header: string[], codes: string[], mtime = '2026-11-23T16:22:22') => ({ mtime, header, codes: codes.map((code) => ({ code })) });
const H = ['# scan_date=20261123', '# for_trading_day=20261124 (r014 next trading day)', '# [r016] applied from picks by autoboth@x at 2026-11-23T16:22:22'];
let n = 0;
const id = () => `g${++n}`;

describe('r004 watchlist tab', () => {
  it('reads the trading day from the header', () => {
    const e = r004Entry(doc(H, ['192820', '008930']));
    expect(e).toMatchObject({ file: r004Src('20261124'), date: '20261124', codes: ['192820', '008930'] });
    expect(r004Entry(doc(['# hand made'], ['005930']))).toBeNull();
    expect(r004Entry(doc(H, []))).toBeNull();
    expect(r004Entry(null)).toBeNull();
  });
  it('adds the tab MMDD first, refreshes it when r004 changes, keeps a deleted tab away until then', () => {
    const g0 = [{ id: 'g1', name: '관심 1', codes: ['005930'] }];
    const a = mergePicks(g0, {}, [r004Entry(doc(H, ['192820', '008930']))!], id, 60, true);
    expect(a.groups.map((g) => g.name)).toEqual(['1124', '관심 1']);
    expect(a.added).toEqual(['1124']);
    // same r004 again: nothing to do
    expect(mergePicks(a.groups, a.imported, [r004Entry(doc(H, ['192820', '008930']))!], id, 60, true).changed).toBe(false);
    // r004 written again (new mtime): the same tab is refreshed in place
    const b = mergePicks(a.groups, a.imported, [r004Entry(doc(H, ['281820'], '2026-11-23T20:10:00'))!], id, 60, true);
    expect(b.groups.map((g) => [g.name, g.codes.join()])).toEqual([['1124', '281820'], ['관심 1', '005930']]);
    // deleted by the user: not made again for the same r004
    const c = mergePicks(b.groups.slice(1), b.imported, [r004Entry(doc(H, ['281820'], '2026-11-23T20:10:00'))!], id, 60, true);
    expect(c.groups.map((g) => g.name)).toEqual(['관심 1']);
    // the next day's selection: a new tab first
    const H2 = ['# scan_date=20261124', '# for_trading_day=20261125 (r014 next trading day)'];
    const d = mergePicks(b.groups, b.imported, [r004Entry(doc(H2, ['005930'], '2026-11-24T20:10:00'))!], id, 60, true);
    expect(d.groups.map((g) => g.name)).toEqual(['1125', '1124', '관심 1']);
  });
});
