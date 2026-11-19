/*
 * Update log
 * - [2026-11-20] type=test - AutoBotHtsH timeline 자동 보기: next symbol (wraps, starts from the first when the current one is not a
 *   target), seconds clamped to 3 - 600, target scopes (전체 / 감시 중 / 보유)
 * - [2026-11-20] type=test - only symbols trading at the viewed time (NXT 08:00, KRX 09:00; replay cursor; whole past day = all)
 */

import { describe, expect, it } from 'vitest';
import { autoTargets, clampSec, inScope, nextCode, tradingAt, viewTime } from '../../src/bot/renderer/src/components/AutoView';
import type { Snapshot, SymbolSummary } from '../../src/bot/shared/types';

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
  it('takes only the symbols already trading at the viewed time', () => {
    const at = (hm: string) => Date.parse(`2026-11-20T${hm}:00+09:00`);
    const nxt = sym('N', { nxt: true });
    const krx = sym('K', { nxt: false });
    const unknown = sym('U');
    expect(tradingAt(nxt, '20261120', at('08:30'))).toBe(true);
    expect(tradingAt(krx, '20261120', at('08:30'))).toBe(false);
    expect(tradingAt(unknown, '20261120', at('08:30'))).toBe(false);
    expect(tradingAt(krx, '20261120', at('09:00'))).toBe(true);
    expect(tradingAt(nxt, '20261120', at('07:59'))).toBe(false);
    const snap = { date: '20261120', symbols: [nxt, krx, unknown] } as unknown as Snapshot;
    expect(autoTargets(snap, 'all', at('08:30'))).toEqual({ codes: ['N'], waiting: 2 });
    expect(autoTargets(snap, 'all', at('10:00'))).toEqual({ codes: ['N', 'K', 'U'], waiting: 0 });
    // live day = now, replayed past day = the cursor, whole past day = everything
    expect(viewTime(true, null, 5)).toBe(5);
    expect(viewTime(false, 7, 5)).toBe(7);
    expect(autoTargets(snap, 'all', viewTime(false, null, at('08:30'))).codes).toHaveLength(3);
  });
});
