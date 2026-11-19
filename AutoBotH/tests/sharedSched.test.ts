/*
 * Update log
 * - [2026-11-20] type=test - Shared local-mode scheduler (sharedSched.ts): one holder at a time, takeover when the holder's process is
 *   gone or its heartbeat is stale, release, and the post-close days merged (the later day wins)
 */

import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { createSharedSched, STALE_MS } from '../src/main/sharedSched';

describe('shared scheduler', () => {
  it('lets one app hold it, hands over on release / dead process / stale heartbeat, merges the days', () => {
    const dir = mkdtempSync(join(tmpdir(), 'sched-'));
    let t = 1_000_000;
    const now = () => t;
    // both pids alive: this test process and its parent
    const a = createSharedSched('AutoBotH', dir, process.pid, now);
    const b = createSharedSched('AutoBotHtsH', dir, process.ppid, now);
    expect(a.hold()).toBe(true);
    expect(b.hold()).toBe(false);
    expect(b.holder()).toBe('AutoBotH');
    expect(a.holder()).toBeNull();
    // heartbeat keeps it
    t += STALE_MS - 1000;
    expect(a.hold()).toBe(true);
    expect(b.hold()).toBe(false);
    // released -> the other app takes it
    a.release();
    expect(b.hold()).toBe(true);
    expect(a.hold()).toBe(false);
    // stale heartbeat -> taken over
    t += STALE_MS + 1;
    expect(a.hold()).toBe(true);
    // holder process gone -> taken over at once
    const dead = createSharedSched('AutoBotHtsH', dir, 2 ** 22 + 12345, now);
    a.release();
    expect(dead.hold()).toBe(true);
    expect(a.hold()).toBe(true);
    // days: the later one wins, own meta kept
    a.setMeta({ lastPostcloseYmd: '20261120' });
    b.setMeta({ lastPostcloseYmd: '20261119', lastPostcloseSyncYmd: '20261119' });
    expect(b.meta({ lastPostcloseYmd: '20261118', other: 1 })).toEqual({ lastPostcloseYmd: '20261120', lastPostcloseSyncYmd: '20261119', other: 1 });
  });
});
