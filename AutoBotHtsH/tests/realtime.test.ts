/*
 * Update log
 * - [2026-11-06] type=test - KisFeed opens one websocket even when several subscriptions arrive while the approval key is pending
 */

import { describe, expect, it, vi } from 'vitest';

const sockets: unknown[] = [];
vi.mock('ws', () => {
  class FakeWs {
    static OPEN = 1;
    readyState = 0;
    constructor(public url: string) {
      sockets.push(this);
    }
    on() {
      return this;
    }
    send() {}
    close() {}
  }
  return { default: FakeWs };
});

describe('KisFeed', () => {
  it('opens a single websocket for concurrent subscriptions', async () => {
    const { KisFeed } = await import('../src/main/realtime');
    let resolve!: (k: string) => void;
    const key = new Promise<string>((r) => (resolve = r));
    const feed = new KisFeed('real', () => key, () => {}, () => {});
    feed.subscribe('H0UNCNT0', '105560');
    feed.subscribe('H0UNASP0', '105560');
    resolve('k');
    await new Promise((r) => setTimeout(r, 0));
    expect(sockets).toHaveLength(1);
    feed.close();
  });
});
