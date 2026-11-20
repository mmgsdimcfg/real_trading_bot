/*
 * Update log
 * - [2026-11-20] type=test - AutoBotHtsH: one window per kind (saved layouts keep the front window of each kind), only one auto view
 *   at a time (starting one stops the other; a saved-on auto view stays off when another started since)
 */

import { describe, expect, it, vi } from 'vitest';

// minimal browser globals for the renderer modules
const store = new Map<string, string>();
const listeners = new Map<string, Set<(e: Event) => void>>();
vi.stubGlobal('localStorage', { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => store.set(k, v), removeItem: (k: string) => store.delete(k) });
vi.stubGlobal('window', {
  addEventListener: (t: string, h: (e: Event) => void) => (listeners.get(t) || listeners.set(t, new Set()).get(t)!).add(h),
  removeEventListener: (t: string, h: (e: Event) => void) => listeners.get(t)?.delete(h),
  dispatchEvent: (e: Event) => (listeners.get(e.type)?.forEach((h) => h(e)), true),
});
vi.stubGlobal('CustomEvent', class extends Event {
  detail: unknown;
  constructor(t: string, o?: { detail?: unknown }) {
    super(t);
    this.detail = o?.detail;
  }
});

const { onePerKind } = await import('../src/renderer/src/store');
const { autoViewMayResume, autoViewStarted, onAutoViewStarted } = await import('../src/bot/renderer/src/components/AutoView');

describe('windows', () => {
  it('keeps the front window of each kind', () => {
    const w = (id: string, type: string, z: number) => ({ id, type, z }) as never;
    const out = onePerKind([w('a', 'watch', 1), w('b', 'chart', 2), w('c', 'watch', 5), w('d', 'bot-timeline', 3), w('e', 'bot-timeline', 1)]);
    expect(out.map((x: { id: string }) => x.id)).toEqual(['b', 'c', 'd']);
  });
});

describe('one auto view at a time', () => {
  it('stops the other one and remembers the last started', () => {
    let timelineOn = true;
    let watchOn = false;
    const offT = onAutoViewStarted('timeline', () => (timelineOn = false));
    const offW = onAutoViewStarted('watch', () => (watchOn = false));
    autoViewStarted('timeline');
    expect(timelineOn).toBe(true);
    watchOn = true;
    autoViewStarted('watch');
    expect(timelineOn).toBe(false);
    expect(watchOn).toBe(true);
    expect(autoViewMayResume('timeline')).toBe(false);
    expect(autoViewMayResume('watch')).toBe(true);
    offT();
    offW();
  });
});
