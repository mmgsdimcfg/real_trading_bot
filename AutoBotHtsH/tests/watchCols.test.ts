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

import { WATCH_COLS_DEFAULT, watchColsOf } from '../src/renderer/src/windows/ListWindows';

describe('watchlist columns', () => {
  it('default = the columns shown so far; a saved choice keeps table order and drops unknown names', () => {
    expect(watchColsOf(undefined)).toEqual(['code', 'change', 'changePct', 'volume', 'amount', 'high', 'low']);
    expect(watchColsOf(undefined)).toBe(WATCH_COLS_DEFAULT);
    expect(watchColsOf(['bid', 'nope', 'open', 'change'])).toEqual(['change', 'open', 'bid']);
    expect(watchColsOf([])).toEqual([]);
  });
});
