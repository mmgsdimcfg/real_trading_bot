/*
 * Update log
 * - [2026-11-23] type=test - 관심종목(타일): tile order (list / rate / traded value, symbols without a price last), the tint strength
 *   and the price position between the day's low and high
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('../src/renderer/src/store', () => ({ useStore: () => undefined }));
vi.mock('../src/renderer/src/hooks', () => ({ useCall: () => ({}) }));
vi.mock('../src/renderer/src/components/NxtTag', () => ({ NxtTag: () => null }));
vi.mock('../src/renderer/src/components/SymbolSearch', () => ({ SymbolSearch: () => null }));
vi.mock('../src/renderer/src/components/WatchMenu', () => ({ WatchMenu: () => null }));
vi.mock('../src/bot/renderer/src/components/AutoView', () => ({ clampSec: (v: unknown) => Number(v) || 10 }));

import { moveCode, rangePos, sortTiles, tileHeat, tileSizeOf } from '../src/renderer/src/windows/WatchTileWindow';
import type { WatchRow } from '../src/shared/kis';

const row = (code: string, changePct: number, amount: number): WatchRow => ({ code, name: code, price: 100, change: changePct, changePct, volume: 1, amount });
const rows = new Map([row('A', 1.5, 300), row('B', -2, 900), row('C', 0, 100)].map((r) => [r.code, r]));
const codes = ['A', 'X', 'B', 'C']; // X has no price yet

describe('watchlist tiles', () => {
  it('orders the tiles; a symbol without a price stays last', () => {
    expect(sortTiles(codes, rows, 'list')).toEqual(['A', 'X', 'B', 'C']);
    expect(sortTiles(codes, rows, 'pctDesc')).toEqual(['A', 'C', 'B', 'X']);
    expect(sortTiles(codes, rows, 'pctAsc')).toEqual(['B', 'C', 'A', 'X']);
    expect(sortTiles(codes, rows, 'amount')).toEqual(['B', 'A', 'C', 'X']);
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
