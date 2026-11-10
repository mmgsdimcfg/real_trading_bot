/*
 * Update log
 * - [2026-11-10] type=feat - Local mode: R016Client.local (argv, same argument checks, serial queue), the cross-host guard decision,
 *   scheduler time rules, the Windows stop button rule and LocalHost (ranged reads, prefix listing, missing files)
 */

import { describe, expect, it, vi } from 'vitest';
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { R016Client } from '../src/main/control';
import { LocalHost, HostSwitch } from '../src/main/host';
import { collectorWindow, localYmd, peerDecision, postcloseDue, serviceButtons, type R016Status } from '../src/shared/control';

const j = (o: unknown) => JSON.stringify(o);

function status(p: Partial<R016Status> & { role?: 'primary' | 'standby'; backend?: string }): R016Status {
  return {
    generated_at: '2026-11-10T10:00:00',
    service: { available: true, active_state: p.running ? 'active' : 'inactive', backend: p.backend },
    running: !!p.running,
    external_r003: p.external_r003 || [],
    env: {},
    dry_run: false,
    risk_profile: 'neutral',
    override_count: 0,
    override_errors: [],
    pending: {},
    pending_restart: false,
    host: p.role ? { platform: 'linux', hostname: 'pi', backend: 'systemd', role: { role: p.role, set: true }, runtime_root: '/r' } : undefined,
  } as R016Status;
}

describe('R016Client.local (argv runner)', () => {
  it('passes argv and stdin unquoted, parses the JSON', async () => {
    const run = vi.fn(async () => ({ out: j({ ok: true, count: 1 }), err: '', code: 0 }));
    const c = R016Client.local(run);
    const r = await c.run(['watchlist', 'set', '--stdin'], { stdin: '005930,삼성전자\n' });
    expect(r.ok).toBe(true);
    expect(run).toHaveBeenCalledWith(['watchlist', 'set', '--stdin'], 30000, '005930,삼성전자\n');
  });
  it('rejects the same unsafe arguments as the SSH client, without running anything', async () => {
    const run = vi.fn(async () => ({ out: '{}', err: '', code: 0 }));
    const c = R016Client.local(run);
    expect((await c.run(['set', 'X="a;b"'])).ok).toBe(false);
    expect((await c.run(['set', "X='a'"])).ok).toBe(false);
    expect(run).not.toHaveBeenCalled();
  });
  it('is a serial queue', async () => {
    let active = 0;
    let max = 0;
    const run = vi.fn(async () => {
      active++;
      max = Math.max(max, active);
      await new Promise((r) => setTimeout(r, 5));
      active--;
      return { out: '{"ok": true}', err: '', code: 0 };
    });
    const c = R016Client.local(run);
    await Promise.all([c.run(['status']), c.run(['job', 'list']), c.run(['doctor'])]);
    expect(max).toBe(1);
  });
});

describe('cross-host guard (peerDecision)', () => {
  it('unknown other host -> ask the user', () => {
    expect(peerDecision(null, 'SSH 연결 실패', '라즈베리파이').action).toBe('ask');
  });
  it('other host trading -> refuse', () => {
    expect(peerDecision(status({ running: true, role: 'primary' }), undefined, 'x').action).toBe('refuse');
    expect(peerDecision(status({ external_r003: [{ pid: 1, cmdline: 'r003' }], role: 'standby' }), undefined, 'x').action).toBe('refuse');
  });
  it('other host idle: primary / old server without role -> standby, already standby -> ok', () => {
    expect(peerDecision(status({ role: 'primary' }), undefined, 'x').action).toBe('standby');
    expect(peerDecision(status({}), undefined, 'x').action).toBe('standby');
    expect(peerDecision(status({ role: 'standby' }), undefined, 'x').action).toBe('ok');
  });
});

describe('scheduler rules', () => {
  const at = (y: number, mo: number, d: number, h: number, mi: number) => new Date(y, mo - 1, d, h, mi);
  it('post-close: weekday, at/after the time, once per day', () => {
    expect(postcloseDue(at(2026, 11, 10, 20, 9), '20:10', undefined)).toBe(false);
    expect(postcloseDue(at(2026, 11, 10, 20, 10), '20:10', undefined)).toBe(true);
    expect(postcloseDue(at(2026, 11, 10, 23, 0), '20:10', '20261109')).toBe(true);
    expect(postcloseDue(at(2026, 11, 10, 23, 0), '20:10', '20261110')).toBe(false);
    expect(postcloseDue(at(2026, 11, 8, 21, 0), '20:10', undefined)).toBe(false); // Sunday
    expect(postcloseDue(at(2026, 11, 10, 21, 0), 'bad', undefined)).toBe(false);
  });
  it('tick collector window Mon-Fri 08:40-15:35', () => {
    expect(collectorWindow(at(2026, 11, 10, 8, 39))).toBe(false);
    expect(collectorWindow(at(2026, 11, 10, 8, 40))).toBe(true);
    expect(collectorWindow(at(2026, 11, 10, 15, 34))).toBe(true);
    expect(collectorWindow(at(2026, 11, 10, 15, 35))).toBe(false);
    expect(collectorWindow(at(2026, 11, 7, 10, 0))).toBe(false); // Saturday
  });
  it('localYmd', () => {
    expect(localYmd(at(2026, 1, 5, 0, 1))).toBe('20260105');
  });
});

describe('service buttons on the Windows process backend', () => {
  it('stop is offered for an r003 started outside the supervisor; start says standby is switched', () => {
    const ext = status({ backend: 'process', external_r003: [{ pid: 5, cmdline: 'r003' }] });
    expect(serviceButtons(ext, false).stop).toBe(true);
    const sysd = status({ backend: undefined, external_r003: [{ pid: 5, cmdline: 'r003' }] });
    expect(serviceButtons(sysd, false).stop).toBe(false);
    const sb = status({ role: 'standby' });
    expect(serviceButtons(sb, false).start).toBe(true);
    expect(serviceButtons(sb, false).startReason).toContain('standby');
  });
});

describe('LocalHost / HostSwitch', () => {
  const dir = mkdtempSync(join(tmpdir(), 'autoboth-local-'));
  mkdirSync(join(dir, 'd'));
  writeFileSync(join(dir, 'd', '005930_삼성전자_10s.txt'), 'abcdefghij');
  writeFileSync(join(dir, 'd', '005930_삼성전자_daily.csv'), 'x');
  writeFileSync(join(dir, 'd', '000660_SK_daily.csv'), 'y');
  const h = new LocalHost();
  h.connected = true;

  it('reads byte ranges and stats (missing = null)', async () => {
    const p = `${dir}/d/005930_삼성전자_10s.txt`;
    expect((await h.readRange(p, 2, 5)).toString()).toBe('cde');
    expect((await h.readRange(p, 8, 20)).toString()).toBe('ij');
    expect((await h.stat(p))?.size).toBe(10);
    expect(await h.stat(`${dir}/nope`)).toBeNull();
  });
  it('lists by prefix and directory entries', async () => {
    expect(await h.listPrefix(`${dir}/d`, '005930_')).toEqual(['005930_삼성전자_10s.txt', '005930_삼성전자_daily.csv']);
    expect(await h.listPrefix(`${dir}/missing`, '0')).toEqual([]);
    const ents = await h.readdir(dir);
    expect(ents.find((e) => e.name === 'd')?.isDir).toBe(true);
  });
  it('HostSwitch forwards to the host of the run mode', async () => {
    const remote = { kind: 'remote' as const, connected: false, readdir: vi.fn(), stat: vi.fn(async () => null), readRange: vi.fn(), listPrefix: vi.fn() };
    let mode: 'remote' | 'local' = 'remote';
    const sw = new HostSwitch(remote, h, () => mode);
    expect(sw.kind).toBe('remote');
    expect(sw.connected).toBe(false);
    await sw.stat('x');
    expect(remote.stat).toHaveBeenCalled();
    mode = 'local';
    expect(sw.kind).toBe('local');
    expect(sw.connected).toBe(true);
    expect((await sw.stat(`${dir}/d/000660_SK_daily.csv`))?.size).toBe(1);
  });
});
