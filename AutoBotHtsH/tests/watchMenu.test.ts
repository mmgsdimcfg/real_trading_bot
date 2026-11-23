/*
 * Update log
 * - [2026-11-23] type=test - Watchlist right-click menu: the group edits (copy / move to another tab, delete, top / bottom)
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('../src/renderer/src/store', () => ({ useStore: () => undefined, WATCH_MAX: 3 }));
vi.mock('../src/renderer/src/hooks', () => ({ useOutsideClose: () => ({ current: null }) }));

import { addBlock, copyToGroup, moveInGroup, moveToGroup, removeFromGroup } from '../src/renderer/src/components/WatchMenu';

const groups = [
  { id: 'a', name: 'A', codes: ['111111', '222222', '333333'] },
  { id: 'b', name: 'B', codes: ['222222'] },
  { id: 'c', name: 'C', codes: ['444444', '555555', '666666'] }, // full (WATCH_MAX 3 here)
];
const codes = (g: typeof groups) => Object.fromEntries(g.map((x) => [x.id, x.codes.join(',')]));

describe('watchlist menu edits', () => {
  it('copy: added at the end; not twice, not into a full tab', () => {
    expect(codes(copyToGroup(groups, 'b', '111111'))).toMatchObject({ a: '111111,222222,333333', b: '222222,111111' });
    expect(copyToGroup(groups, 'b', '222222')).toEqual(groups);
    expect(copyToGroup(groups, 'c', '111111')).toEqual(groups);
    expect([addBlock(groups[1], '222222'), addBlock(groups[2], '111111'), addBlock(groups[1], '111111')]).toEqual(['이미 있음', '가득 참', '']);
  });
  it('move: leaves the source only when the target took it', () => {
    expect(codes(moveToGroup(groups, 'a', 'b', '111111'))).toMatchObject({ a: '222222,333333', b: '222222,111111' });
    expect(moveToGroup(groups, 'a', 'b', '222222')).toBe(groups); // already there: nothing is lost
    expect(moveToGroup(groups, 'a', 'c', '111111')).toBe(groups); // full
    expect(moveToGroup(groups, 'a', 'a', '111111')).toBe(groups);
  });
  it('delete and top / bottom touch this tab only', () => {
    expect(codes(removeFromGroup(groups, 'a', '222222'))).toMatchObject({ a: '111111,333333', b: '222222' });
    expect(codes(moveInGroup(groups, 'a', '333333', 'top')).a).toBe('333333,111111,222222');
    expect(codes(moveInGroup(groups, 'a', '111111', 'bottom')).a).toBe('222222,333333,111111');
    expect(codes(moveInGroup(groups, 'b', '999999', 'top')).b).toBe('222222');
  });
});
