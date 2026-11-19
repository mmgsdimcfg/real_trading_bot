/*
 * Update log
 * - [2026-11-20] type=test - timeline 자동 보기: next symbol (wraps, starts from the first when the current one is not a
 *   target), seconds clamped to 3 - 600, target scopes (전체 / 감시 중 / 보유)
 */

import { describe, expect, it } from 'vitest';
import { clampSec, inScope, nextCode } from '../src/renderer/src/components/AutoView';
import type { SymbolSummary } from '../src/shared/types';

const sym = (code: string, p: Partial<SymbolSummary> = {}) =>
  ({ code, name: code, watch: 'active', furthestOrderToday: -1, blockCounts: {}, sellCounts: {}, realizedKrw: 0, evalCount: 0, ...p }) as SymbolSummary;

describe('timeline auto view', () => {
  it('steps through the codes and wraps', () => {
    const c = ['A', 'B', 'C'];
    expect(nextCode(c, 'A')).toBe('B');
    expect(nextCode(c, 'C')).toBe('A');
    expect(nextCode(c, 'X')).toBe('A');
    expect(nextCode(c, null)).toBe('A');
    expect(nextCode([], 'A')).toBeNull();
  });
  it('keeps the interval in 3 - 600 s', () => {
    expect(clampSec(1)).toBe(3);
    expect(clampSec('15')).toBe(15);
    expect(clampSec(9999)).toBe(600);
    expect(clampSec('abc')).toBe(10);
  });
  it('filters the target symbols', () => {
    const a = sym('A');
    const b = sym('B', { watch: 'backup' });
    const h = sym('H', { watch: 'backup', held: 'carried' });
    expect([a, b, h].filter((s) => inScope(s, 'all')).map((s) => s.code)).toEqual(['A', 'B', 'H']);
    expect([a, b, h].filter((s) => inScope(s, 'active')).map((s) => s.code)).toEqual(['A']);
    expect([a, b, h].filter((s) => inScope(s, 'held')).map((s) => s.code)).toEqual(['H']);
  });
});
