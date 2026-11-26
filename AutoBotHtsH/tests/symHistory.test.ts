import { describe, expect, it } from 'vitest';
import { dropSymHistory, parseSymHistory, pushSymHistory, SYM_HISTORY_MAX } from '../src/shared/symHistory';

describe('symbol input history', () => {
  it('puts the newest code first without duplicates', () => {
    let l: string[] = [];
    l = pushSymHistory(l, '005930');
    l = pushSymHistory(l, '000660');
    l = pushSymHistory(l, '005930');
    expect(l).toEqual(['005930', '000660']);
    expect(pushSymHistory(l, '005930')).toBe(l); // already first: the same list
    expect(pushSymHistory(l, 'abc')).toBe(l); // not a code
    expect(pushSymHistory(l, '0001a0')).toEqual(['0001A0', '005930', '000660']);
  });
  it('keeps at most 20', () => {
    let l: string[] = [];
    for (let i = 0; i < 30; i++) l = pushSymHistory(l, String(100000 + i));
    expect(l.length).toBe(SYM_HISTORY_MAX);
    expect(l[0]).toBe('100029');
    expect(l[19]).toBe('100010');
  });
  it('removes one and reads a stored list safely', () => {
    expect(dropSymHistory(['005930', '000660'], '005930')).toEqual(['000660']);
    expect(parseSymHistory('["005930","005930","x",5,"000660"]')).toEqual(['005930', '000660']);
    expect(parseSymHistory('{bad')).toEqual([]);
    expect(parseSymHistory(null)).toEqual([]);
    expect(parseSymHistory(JSON.stringify(Array.from({ length: 40 }, (_, i) => String(200000 + i)))).length).toBe(20);
  });
});
