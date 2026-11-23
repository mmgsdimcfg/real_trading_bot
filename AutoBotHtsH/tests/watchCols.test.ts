/*
 * Update log
 * - [2026-11-23] type=test - Watchlist list window, 항목 menu: the default columns (those shown before the menu existed), unknown
 *   names dropped, table order kept
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('../src/renderer/src/store', () => ({ useStore: () => undefined, LINK_COLORS: [], WATCH_MAX: 60, newGroupId: () => 'g' }));
vi.mock('../src/renderer/src/hooks', () => ({ useCall: () => ({}), useOutsideClose: () => ({ current: null }) }));
vi.mock('../src/renderer/src/components/SymbolSearch', () => ({ SymbolSearch: () => null }));
vi.mock('../src/renderer/src/components/NxtTag', () => ({ NxtTag: () => null }));
vi.mock('../src/renderer/src/components/FloatPop', () => ({ FloatPop: () => null }));
vi.mock('../src/renderer/src/components/WatchMenu', () => ({ WatchMenu: () => null }));
vi.mock('../src/renderer/src/components/PriceChart', () => ({ PriceChart: () => null }));
vi.mock('../src/bot/renderer/src/components/AutoView', () => ({ clampSec: (v: unknown) => Number(v) || 10 }));

import { nextWatchSort, sortWatch, WATCH_COLS_DEFAULT, watchColsOf, watchSortOf } from '../src/renderer/src/windows/ListWindows';
import type { WatchRow } from '../src/shared/kis';

describe('watchlist columns', () => {
  it('default = the columns shown so far; a saved choice keeps table order and drops unknown names', () => {
    expect(watchColsOf(undefined)).toEqual(['code', 'change', 'changePct', 'volume', 'amount', 'high', 'low']);
    expect(watchColsOf(undefined)).toBe(WATCH_COLS_DEFAULT);
    expect(watchColsOf(['bid', 'nope', 'open', 'change'])).toEqual(['change', 'open', 'bid']);
    expect(watchColsOf([])).toEqual([]);
  });

  it('[2026-11-23] column title clicks: high to low, low to high, off; names start A to Z', () => {
    const a = nextWatchSort(null, 'changePct');
    expect(a).toEqual({ key: 'changePct', dir: -1 });
    const b = nextWatchSort(a, 'changePct');
    expect(b).toEqual({ key: 'changePct', dir: 1 });
    expect(nextWatchSort(b, 'changePct')).toBeNull();
    expect(nextWatchSort(b, 'volume')).toEqual({ key: 'volume', dir: -1 });
    expect(nextWatchSort(null, 'name')).toEqual({ key: 'name', dir: 1 });
    expect(nextWatchSort({ key: 'name', dir: 1 }, 'name')).toEqual({ key: 'name', dir: -1 });
    expect([watchSortOf('price', 1), watchSortOf('price', null), watchSortOf('nope', 1), watchSortOf(null, null)]).toEqual([{ key: 'price', dir: 1 }, { key: 'price', dir: -1 }, null, null]);
  });

  it('[2026-11-23] sortWatch: by value or name, symbols without a value last, ties in the registered order', () => {
    const row = (code: string, name: string, changePct: number, volume: number): WatchRow => ({ code, name, price: 100, change: changePct, changePct, volume });
    const rows = new Map([row('1', '다', 1, 5), row('2', '가', -2, 5), row('3', '나', 3, 9)].map((r) => [r.code, r]));
    const codes = ['1', 'X', '2', '3'];
    expect(sortWatch(codes, rows, null)).toBe(codes);
    expect(sortWatch(codes, rows, { key: 'changePct', dir: -1 })).toEqual(['3', '1', '2', 'X']);
    expect(sortWatch(codes, rows, { key: 'changePct', dir: 1 })).toEqual(['2', '1', '3', 'X']);
    expect(sortWatch(codes, rows, { key: 'name', dir: 1 })).toEqual(['2', '3', '1', 'X']);
    expect(sortWatch(codes, rows, { key: 'volume', dir: -1 })).toEqual(['3', '1', '2', 'X']); // 1 and 2 tie: registered order
    expect(sortWatch(codes, rows, { key: 'ask', dir: -1 })).toEqual(codes); // no values at all
  });
});
