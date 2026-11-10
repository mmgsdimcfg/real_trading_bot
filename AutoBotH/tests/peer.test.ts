/*
 * Update log
 * - [2026-11-10] type=feat - The cross-host start guard (peer.ts guardStart) and the local-mode scheduler with fake r016 hosts:
 *   unreachable / failing other host -> confirmation, other host trading -> refused, primary -> standby before the start, broken
 *   local settings after the local mode was used -> confirmation (not "no local bot"), exclusive role switch, collector only on an
 *   explicit primary, watchlist copy retried until it succeeds
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

const dialogAnswer = { response: 1 }; // 1 = 취소 (cancel), 0 = OK
vi.mock('electron', () => ({
  ipcMain: { handle: vi.fn() },
  dialog: { showMessageBox: vi.fn(async () => dialogAnswer) },
}));

import { R016Client } from '../src/main/control';
import { createPeer } from '../src/main/peer';
import { createScheduler } from '../src/main/scheduler';
import type { SshConnection } from '../src/main/ssh';
import type { R016Status } from '../src/shared/control';
import type { Settings } from '../src/shared/types';

type Handler = (args: string[]) => unknown;

/** fake host: answers r016 commands from `h` (JSON), records every call */
function fakeHost(h: Record<string, Handler>) {
  const calls: string[][] = [];
  const client = R016Client.local(async (args) => {
    calls.push(args);
    const fn = h[args.slice(0, 2).join(' ')] || h[args[0]];
    if (!fn) return { out: JSON.stringify({ ok: false, error: `invalid choice: ${args[0]}` }), err: '', code: 2 };
    const v = fn(args);
    if (v instanceof Error) throw v;
    const o = v as { ok?: boolean };
    return { out: JSON.stringify(v), err: '', code: o.ok === false ? 3 : 0 };
  });
  return { client, calls };
}

const st = (running: boolean, role: 'primary' | 'standby' | null): R016Status =>
  ({
    generated_at: '2026-11-10T10:00:00',
    service: { available: true, active_state: running ? 'active' : 'inactive' },
    running,
    external_r003: [],
    env: {},
    dry_run: false,
    risk_profile: 'neutral',
    override_count: 0,
    override_errors: [],
    pending: {},
    pending_restart: false,
    host: role ? { platform: 'linux', hostname: 'pi', backend: 'systemd', role: { role, set: true }, runtime_root: '/r' } : undefined,
  }) as R016Status;

function setup(opts: { mode?: 'local' | 'remote'; peer: Record<string, Handler>; self?: Record<string, Handler>; sshFails?: boolean; localProblem?: string; everUsed?: boolean }) {
  const peer = fakeHost(opts.peer);
  let selfRole: { role: string; set: boolean } = { role: 'primary', set: false };
  const self = fakeHost(
    opts.self || {
      'role get': () => ({ ok: true, ...selfRole }),
      'role set': (a) => {
        selfRole = { role: a[2], set: true };
        return { ok: true, ...selfRole };
      },
    },
  );
  const mode = opts.mode || 'local';
  const ssh = { connected: !opts.sshFails } as unknown as SshConnection;
  const p = createPeer({
    mode: () => mode,
    ssh,
    connectSsh: async () => {
      if (opts.sshFails) throw new Error('ECONNREFUSED');
    },
    remoteConfigured: () => true,
    localProblem: () => opts.localProblem || '',
    remoteClient: mode === 'local' ? peer.client : self.client,
    localClient: mode === 'local' ? self.client : peer.client,
    localEverUsed: () => !!opts.everUsed,
    win: () => null,
    log: () => undefined,
  });
  return { p, peer, self, selfRole: () => selfRole };
}

beforeEach(() => {
  dialogAnswer.response = 1;
});

describe('guardStart (local mode: the other host is the Raspberry Pi)', () => {
  it('Pi trading -> refused, nothing changed', async () => {
    const s = setup({ peer: { status: () => st(true, 'primary') } });
    const r = await s.p.guardStart(s.self.client);
    expect(r.ok).toBe(false);
    expect(r.error).toContain('실행 중');
    expect(s.peer.calls.some((c) => c[0] === 'role')).toBe(false);
  });
  it('Pi idle primary -> Pi standby first, then this PC explicitly primary', async () => {
    let piRole = 'primary';
    const s = setup({ peer: { status: () => st(false, piRole as 'primary'), 'role set': (a) => ((piRole = a[2]), { ok: true, role: a[2], set: true }) } });
    const r = await s.p.guardStart(s.self.client);
    expect(r.ok).toBe(true);
    expect(piRole).toBe('standby');
    expect(s.selfRole()).toEqual({ role: 'primary', set: true });
  });
  it('Pi refuses standby (its r003 just started) -> refused', async () => {
    const s = setup({ peer: { status: () => st(false, 'primary'), 'role set': () => ({ ok: false, error: 'r003 is running on this host' }) } });
    const r = await s.p.guardStart(s.self.client);
    expect(r.ok).toBe(false);
    expect(r.error).toContain('standby');
  });
  it('SSH down -> asks; cancel = cancelled, OK = go on', async () => {
    const s = setup({ peer: {}, sshFails: true });
    expect((await s.p.guardStart(s.self.client)).cancelled).toBe(true);
    dialogAnswer.response = 0;
    expect((await s.p.guardStart(s.self.client)).ok).toBe(true);
  });
  it('status call throws (timeout / dropped link) -> asks instead of failing', async () => {
    const s = setup({ peer: { status: () => new Error('원격 명령 시간 초과 (30s)') } });
    const r = await s.p.guardStart(s.self.client);
    expect(r.cancelled).toBe(true);
  });
  it('old Pi code without role -> asks before going on', async () => {
    const s = setup({ peer: { status: () => st(false, null) } });
    expect((await s.p.guardStart(s.self.client)).cancelled).toBe(true);
    dialogAnswer.response = 0;
    expect((await s.p.guardStart(s.self.client)).ok).toBe(true);
  });
});

describe('guardStart (remote mode: the other host is this PC)', () => {
  it('local mode never used and not set up -> old behaviour (no question)', async () => {
    const s = setup({ mode: 'remote', peer: {}, localProblem: 'no venv', everUsed: false });
    expect((await s.p.guardStart(s.self.client)).ok).toBe(true);
  });
  it('local mode used before but settings broken now -> asks (a detached local r003 may still trade)', async () => {
    const s = setup({ mode: 'remote', peer: {}, localProblem: 'no venv', everUsed: true });
    expect((await s.p.guardStart(s.self.client)).cancelled).toBe(true);
  });
  it('local r003 running -> refused', async () => {
    const s = setup({ mode: 'remote', peer: { status: () => st(true, 'primary') } });
    expect((await s.p.guardStart(s.self.client)).ok).toBe(false);
  });
});

describe('peer:role primary (other host becomes primary, this host standby)', () => {
  const handlerOf = async () => {
    const { ipcMain } = await import('electron');
    const calls = (ipcMain.handle as unknown as { mock: { calls: [string, (...a: unknown[]) => Promise<unknown>][] } }).mock.calls;
    return calls.filter((c) => c[0] === 'peer:role').pop()![1];
  };
  it('this host goes standby first; a lost answer with the other host primary keeps it that way', async () => {
    dialogAnswer.response = 0;
    let piRole = 'standby';
    let lose = true;
    const s = setup({
      peer: {
        'role set': (a) => {
          piRole = a[2];
          if (lose) return new Error('SSH 연결이 끊어졌습니다');
          return { ok: true, role: a[2], set: true };
        },
        'role get': () => ({ ok: true, role: piRole, set: true }),
      },
    });
    const h = await handlerOf();
    const r = (await h({}, 'primary')) as { ok: boolean };
    expect(r.ok).toBe(true);
    expect(piRole).toBe('primary');
    expect(s.selfRole().role).toBe('standby'); // never two primaries
    lose = false;
  });
  it('other host unreachable after this one went standby -> stays standby, reports it', async () => {
    dialogAnswer.response = 0;
    const s = setup({ peer: { 'role set': () => new Error('timeout'), 'role get': () => new Error('timeout') } });
    const h = await handlerOf();
    const r = (await h({}, 'primary')) as { ok: boolean; error?: string };
    expect(r.ok).toBe(false);
    expect(r.error).toContain('standby');
    expect(s.selfRole().role).toBe('standby');
  });
});

describe('local-mode scheduler', () => {
  it('copies the watchlist of a job that finished after midnight', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 10, 11, 0, 30));
    const meta: { lastPostcloseYmd?: string; lastPostcloseSyncYmd?: string } = { lastPostcloseYmd: '20261110' };
    const h = fakeHost({ 'job list': () => ({ ok: true, running: null, jobs: [{ name: 'postclose', last: { state: 'done', args: { date: '20261110' } } }] }) });
    const sync = vi.fn(async () => ({ ok: true, count: 3 }));
    const sc = createScheduler({
      active: () => true,
      settings: () => ({ autoCollector: false, autoPostclose: false, postcloseTime: '20:10', syncWatchlistToPeer: true }) as Settings,
      client: () => h.client,
      status: () => null,
      meta: () => meta,
      setMeta: (p) => Object.assign(meta, p),
      syncToPeer: sync,
      send: () => undefined,
    });
    await sc.tick();
    expect(sync).toHaveBeenCalledTimes(1);
    expect(meta.lastPostcloseSyncYmd).toBe('20261110');
    vi.useRealTimers();
  });

  const settings = { autoCollector: true, autoPostclose: false, postcloseTime: '20:10', postcloseEdgeAi: false, syncWatchlistToPeer: true } as Settings;

  it('starts the tick collector only on an explicitly primary host', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 10, 10, 9, 0));
    const h = fakeHost({ 'collector start': () => ({ ok: true, started: true, pid: 7 }) });
    let status: R016Status | null = st(false, null);
    const sc = createScheduler({
      active: () => true,
      settings: () => settings,
      client: () => h.client,
      status: () => status,
      meta: () => ({}),
      setMeta: () => undefined,
      syncToPeer: async () => ({ ok: true }),
      send: () => undefined,
    });
    await sc.tick();
    expect(h.calls.length).toBe(0); // no role file = not proof that the Pi stopped collecting
    expect(sc.state().lastCollectorMsg).toContain('primary');
    status = { ...st(false, null), host: { platform: 'windows', hostname: 'pc', backend: 'process', role: { role: 'primary', set: true }, runtime_root: 'x' } };
    vi.setSystemTime(new Date(2026, 10, 10, 9, 6));
    await sc.tick();
    expect(h.calls).toEqual([['collector', 'start']]);
    vi.useRealTimers();
  });

  it('watchlist copy to the Pi is retried until it succeeds', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 10, 10, 21, 0));
    const meta: { lastPostcloseYmd?: string; lastPostcloseSyncYmd?: string } = { lastPostcloseYmd: '20261110' };
    const h = fakeHost({ 'job list': () => ({ ok: true, running: null, jobs: [{ name: 'postclose', last: { state: 'done', args: { date: '20261110' } } }] }) });
    let ok = false;
    const sync = vi.fn(async () => (ok ? { ok: true, count: 5 } : { ok: false, error: 'SSH down' }));
    const sc = createScheduler({
      active: () => true,
      settings: () => ({ ...settings, autoCollector: false }),
      client: () => h.client,
      status: () => null,
      meta: () => meta,
      setMeta: (p) => Object.assign(meta, p),
      syncToPeer: sync,
      send: () => undefined,
    });
    await sc.tick();
    expect(sync).toHaveBeenCalledTimes(1);
    expect(meta.lastPostcloseSyncYmd).toBeUndefined();
    await sc.tick(); // within 10 min: no retry yet
    expect(sync).toHaveBeenCalledTimes(1);
    vi.setSystemTime(new Date(2026, 10, 10, 21, 11));
    ok = true;
    await sc.tick();
    expect(sync).toHaveBeenCalledTimes(2);
    expect(meta.lastPostcloseSyncYmd).toBe('20261110');
    vi.useRealTimers();
  });
});
