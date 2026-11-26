import { describe, expect, it, vi } from 'vitest';
import { parseCollectorLock } from '../src/shared/collectorLock';
import { KisFeed } from '../src/main/realtime';

describe('tick collector lock (bot first)', () => {
  it('reads the pid of a lock written today', () => {
    expect(parseCollectorLock('21148 2026-11-26 07:55:28 ppid=20040', '20261126')).toBe(21148);
    expect(parseCollectorLock('  77 2026-11-26T07:55:28\n', '20261126')).toBe(77);
  });
  it('ignores a lock of another day and broken text', () => {
    expect(parseCollectorLock('21148 2026-11-25 07:55:28 ppid=20040', '20261126')).toBeNull();
    expect(parseCollectorLock('', '20261126')).toBeNull();
    expect(parseCollectorLock('abc 2026-11-26 07:55:28', '20261126')).toBeNull();
    expect(parseCollectorLock('0 2026-11-26 07:55:28', '20261126')).toBeNull();
  });
});

describe('KisFeed pause / resume', () => {
  it('opens no session while paused and says why', async () => {
    const key = vi.fn(async () => 'k');
    const status: [string, string | undefined][] = [];
    const f = new KisFeed('real', key, () => undefined, (s, m) => status.push([s, m]));
    f.pause('봇 우선');
    expect(f.isPaused).toBe(true);
    expect(status[status.length - 1]).toEqual(['off', '봇 우선']);
    f.subscribe('H0STCNT0', '005930');
    await new Promise((r) => setTimeout(r, 20));
    expect(key).not.toHaveBeenCalled(); // no approval key asked = no connection attempt
    expect(f.status).toBe('off');
    f.pause('again'); // no second status line
    expect(status.filter((x) => x[1] === 'again').length).toBe(0);
    f.close();
  });
  it('resume connects again when something is subscribed', async () => {
    const key = vi.fn(async () => {
      throw new Error('no network in tests');
    });
    const f = new KisFeed('real', key, () => undefined, () => undefined);
    f.pause('봇 우선');
    f.resume(); // nothing subscribed: stays idle
    await new Promise((r) => setTimeout(r, 10));
    expect(key).not.toHaveBeenCalled();
    f.pause('봇 우선');
    f.subscribe('H0STCNT0', '005930');
    f.resume();
    await new Promise((r) => setTimeout(r, 20));
    expect(key).toHaveBeenCalledTimes(1);
    expect(f.isPaused).toBe(false);
    f.close();
  });
});
