/*
 * Update log
 * - [2026-11-10] type=feat - Integration: the app's local path (localRunner -> R016Client.local -> live_trading/r016_control.py with
 *   the local venv) against a temporary runtime folder. Skipped when the venv / code is not on this PC (CI, other machines)
 */

import { afterAll, describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { R016Client } from '../src/main/control';
import { makeLocalRunner } from '../src/main/localRunner';
import type { JobList, R016Status, RoleInfo, WatchlistDoc } from '../src/shared/control';

const ROOT = resolve(__dirname, '..', '..', '..');
const PY = process.env.AUTOBOTH_TEST_PYTHON || join(ROOT, 'venv', 'Scripts', 'python.exe');
const CODE = join(ROOT, 'real_trading_bot', 'AutoBot');
const R016 = join(CODE, 'live_trading', 'r016_control.py');
const enabled = existsSync(PY) && existsSync(R016);

describe.skipIf(!enabled)('local r016 through the app runner', () => {
  const rt = mkdtempSync(join(tmpdir(), 'autoboth-rt-'));
  const prev = process.env.AUTO_TRADING_RUNTIME;
  process.env.AUTO_TRADING_RUNTIME = rt;
  const c = R016Client.local(makeLocalRunner(() => ({ python: PY, script: R016, cwd: CODE, by: 'autoboth@test' })));
  afterAll(() => {
    if (prev === undefined) delete process.env.AUTO_TRADING_RUNTIME;
    else process.env.AUTO_TRADING_RUNTIME = prev;
    rmSync(rt, { recursive: true, force: true });
  });

  it('status has the host block (Windows: process backend)', async () => {
    const r = await c.run<R016Status>(['status'], { timeoutMs: 60000 });
    expect(r.ok, r.error).toBe(true);
    expect(r.data?.host?.runtime_root).toBe(rt);
    if (process.platform === 'win32') expect(r.data?.service.backend).toBe('process');
  }, 70000);

  // never `watchlist set` here: the r004 file lives in the code folder (live_trading/), not in the temporary runtime
  it('role set/get (temporary runtime) and a standby host refuses start; Korean note round trip', async () => {
    const s = await c.run<RoleInfo>(['role', 'set', 'standby', '--note', '노트북 메인'], { timeoutMs: 60000 });
    expect(s.ok, s.error).toBe(true);
    const g = await c.run<RoleInfo>(['role', 'get'], { timeoutMs: 60000 });
    expect(g.data?.role).toBe('standby');
    expect(g.data?.note).toBe('노트북 메인');
    const wl = await c.run<WatchlistDoc>(['watchlist', 'get'], { timeoutMs: 60000 });
    expect(wl.ok, wl.error).toBe(true);
    const st = await c.run(['start'], { timeoutMs: 60000 });
    expect(st.ok).toBe(false); // standby host refuses
    expect(st.error).toContain('standby');
  }, 120000);

  it('job list returns the Korean titles', async () => {
    const r = await c.run<JobList>(['job', 'list'], { timeoutMs: 60000 });
    expect(r.ok, r.error).toBe(true);
    expect(r.data?.jobs.find((j) => j.name === 'postclose')?.title).toContain('장 마감');
  }, 70000);
});
