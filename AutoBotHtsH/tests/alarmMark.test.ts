/*
 * Update log
 * - [2026-11-23] type=test - Watchlist alert bell: today's (KST) rebound alerts per code, newest first; other days are left out
 */

import { describe, expect, it } from 'vitest';
import { todayAlerts } from '../src/renderer/src/components/AlarmMark';
import type { ReboundHistoryItem } from '../src/preload';

const KST = 9 * 3600000;
const at = (iso: string) => Date.parse(iso + 'Z') - KST; // KST wall clock
const item = (id: string, code: string, when: string) => ({ id, code, at: at(when) }) as unknown as ReboundHistoryItem;

describe('watchlist alert bell', () => {
  it('groups the alerts of the KST day by code, newest first', () => {
    const items = [
      item('a', '008930', '2026-11-23T10:58:04'),
      item('b', '008930', '2026-11-23T13:20:00'),
      item('c', '008770', '2026-11-23T00:05:00'), // just after midnight KST: today
      item('d', '008770', '2026-11-22T23:59:00'), // yesterday
      item('e', '005930', '2026-11-20T14:55:00'),
    ];
    const m = todayAlerts(items, at('2026-11-23T15:00:00'));
    expect([...m.keys()].sort()).toEqual(['008770', '008930']);
    expect(m.get('008930')!.map((x) => x.id)).toEqual(['b', 'a']);
    expect(m.get('008770')!.map((x) => x.id)).toEqual(['c']);
    expect(todayAlerts(items, at('2026-11-24T09:00:00')).size).toBe(0);
  });
});
