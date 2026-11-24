// Update log
// - [2026-11-24] type=test (관심종목(요약) sort: change rate high to low when nothing was chosen, a stored choice (or the registered
//   order = sortKey null) is kept, keys of other windows' columns are ignored)

import { describe, expect, it } from 'vitest';
import { sumSortOf, SUM_SORT_DEFAULT } from '../src/renderer/src/windows/WatchSumWindow';

describe('관심종목(요약) sort', () => {
  it('a new window sorts by the change rate, high to low', () => {
    expect(sumSortOf({})).toEqual({ key: 'changePct', dir: -1 });
    expect(SUM_SORT_DEFAULT).toEqual({ key: 'changePct', dir: -1 });
  });
  it('keeps a chosen sort and the registered order', () => {
    expect(sumSortOf({ sortKey: 'price', sortDir: 1 })).toEqual({ key: 'price', dir: 1 });
    expect(sumSortOf({ sortKey: 'changePct', sortDir: 1 })).toEqual({ key: 'changePct', dir: 1 });
    expect(sumSortOf({ sortKey: null, sortDir: null })).toBeNull();
  });
  it('a key of a column this window does not show means the registered order', () => {
    expect(sumSortOf({ sortKey: 'volume', sortDir: -1 })).toBeNull();
  });
});
