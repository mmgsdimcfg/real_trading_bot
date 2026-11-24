/*
 * Update log
 * - [2026-11-23] type=test - 관심종목(타일): tile order (list / rate / traded value, symbols without a price last), the tint strength
 *   and the price position between the day's low and high
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('../src/renderer/src/store', () => ({ useStore: () => undefined }));
vi.mock('../src/renderer/src/hooks', () => ({ useCall: () => ({}), useOutsideClose: () => ({ current: null }) }));
vi.mock('../src/renderer/src/components/FloatPop', () => ({ FloatPop: () => null }));
vi.mock('../src/renderer/src/components/NxtTag', () => ({ NxtTag: () => null }));
vi.mock('../src/renderer/src/components/SymbolSearch', () => ({ SymbolSearch: () => null }));
vi.mock('../src/renderer/src/components/WatchMenu', () => ({ WatchMenu: () => null }));
vi.mock('../src/bot/renderer/src/components/AutoView', () => ({ clampSec: (v: unknown) => Number(v) || 10 }));

import { moveCode, rangePos, tileFieldsOf, tileHeat, TILE_FIELDS_DEFAULT, tileSizeOf, tileSortOf, tileTintOf } from '../src/renderer/src/windows/WatchTileWindow';
import { firstDir, SORT_LABEL, sortWatch } from '../src/renderer/src/watchSort';
import type { WatchRow } from '../src/shared/kis';

const row = (code: string, changePct: number, amount: number): WatchRow => ({ code, name: code, price: 100, change: changePct, changePct, volume: 1, amount });
const rows = new Map([row('A', 1.5, 300), row('B', -2, 900), row('C', 0, 100)].map((r) => [r.code, r]));
const codes = ['A', 'X', 'B', 'C']; // X has no price yet

describe('watchlist tiles', () => {
  it('orders the tiles; a symbol without a price stays last', () => {
    // [2026-11-24] nothing chosen = change rate high to low; registered order = sortKey null (or the older sort 'list')
    expect(tileSortOf({})).toEqual({ key: 'changePct', dir: -1 });
    expect(sortWatch(codes, rows, tileSortOf({}))).toEqual(sortWatch(codes, rows, tileSortOf({ sort: 'pctDesc' })));
    expect(sortWatch(codes, rows, tileSortOf({ sortKey: null }))).toEqual(['A', 'X', 'B', 'C']);
    expect(sortWatch(codes, rows, tileSortOf({ sort: 'pctDesc' }))).toEqual(['A', 'C', 'B', 'X']);
    expect(sortWatch(codes, rows, tileSortOf({ sort: 'pctAsc' }))).toEqual(['B', 'C', 'A', 'X']);
    expect(sortWatch(codes, rows, tileSortOf({ sort: 'amount' }))).toEqual(['B', 'A', 'C', 'X']);
  });
  it('[2026-11-23] sort select: sortKey / sortDir win over the older opts.sort; null key = registered order', () => {
    expect(tileSortOf({ sort: 'pctDesc', sortKey: 'volume', sortDir: 1 })).toEqual({ key: 'volume', dir: 1 });
    expect(tileSortOf({ sort: 'pctDesc', sortKey: null })).toBeNull();
    expect(tileSortOf({ sort: 'list' })).toBeNull();
    expect([firstDir('name'), firstDir('price')]).toEqual([1, -1]);
    expect(SORT_LABEL.changePct).toBe('등락률');
    expect(sortWatch(codes, rows, tileSortOf({ sortKey: 'name', sortDir: -1 }))).toEqual(['C', 'B', 'A', 'X']);
  });
  it('[2026-11-23] drag: the tile takes the place of the one it is dropped on', () => {
    const l = ['A', 'B', 'C', 'D'];
    expect(moveCode(l, 'A', 'C')).toEqual(['B', 'C', 'A', 'D']);
    expect(moveCode(l, 'D', 'A')).toEqual(['D', 'A', 'B', 'C']);
    expect(moveCode(l, 'B', 'C')).toEqual(['A', 'C', 'B', 'D']);
    expect(moveCode(l, 'B', 'B')).toBe(l);
    expect(moveCode(l, 'X', 'B')).toBe(l);
    expect(moveCode(l, 'B', 'X')).toBe(l);
  });
  it('[2026-11-23] tile size: unknown values are the normal size', () => {
    expect([tileSizeOf('s'), tileSizeOf('l'), tileSizeOf('m'), tileSizeOf(undefined), tileSizeOf('x')]).toEqual(['s', 'l', 'm', 'm', 'm']);
  });
  it('[2026-11-23] tile fields: default without a saved choice, unknown names dropped, menu order kept', () => {
    expect(tileFieldsOf(undefined)).toEqual(TILE_FIELDS_DEFAULT);
    expect(TILE_FIELDS_DEFAULT).not.toContain('ask');
    expect(tileFieldsOf(['bid', 'x', 'code', 'volume'])).toEqual(['code', 'volume', 'bid']);
    expect(tileFieldsOf([])).toEqual([]);
  });
  it('[2026-11-23] tint select: saved multiplier, anything else is normal', () => {
    expect([tileTintOf(0), tileTintOf(0.5), tileTintOf(3), tileTintOf(undefined), tileTintOf(7), tileTintOf('2')]).toEqual([0, 0.5, 3, 1, 1, 1]);
  });
  it('tint strength: 3 % is full, unknown is none', () => {
    expect([tileHeat(0), tileHeat(1.5), tileHeat(-3), tileHeat(12), tileHeat(undefined)]).toEqual([0, 0.5, 1, 1, 0]);
  });
  it('price position between low and high', () => {
    expect(rangePos({ price: 105, high: 110, low: 100 })).toBe(50);
    expect(rangePos({ price: 120, high: 110, low: 100 })).toBe(100);
    expect(rangePos({ price: 100, high: 100, low: 100 })).toBeNull();
    expect(rangePos({ price: 100, high: undefined, low: undefined })).toBeNull();
    expect(rangePos(undefined)).toBeNull();
  });
});
