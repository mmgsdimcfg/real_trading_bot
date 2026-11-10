/*
 * Update log
 * - [2026-11-10] type=feat - Local mode: runs live_trading/r016_control.py with the local venv Python as a child process (argv, no
 *   shell - nothing to quote), UTF-8 stdio (PYTHONUTF8 / PYTHONIOENCODING), no console window, optional stdin. On timeout the whole
 *   process tree is ended (taskkill /T: the venv python.exe is a launcher with the real interpreter as its child). r016 itself
 *   starts the long-lived processes (r020 supervisor, r021 jobs, g011) detached, so ending a call never stops trading
 */

import { spawn } from 'node:child_process';

export type ArgvRunner = (args: string[], timeoutMs: number, stdin?: string) => Promise<{ out: string; err: string; code: number | null }>;

export interface LocalRunnerConfig {
  python: string;
  script: string;
  cwd: string;
  by: string;
}

function killTree(pid: number | undefined): void {
  if (!pid) return;
  if (process.platform === 'win32') {
    try {
      spawn('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
    } catch {
      /* best effort */
    }
  } else {
    try {
      process.kill(pid, 'SIGTERM');
    } catch {
      /* gone */
    }
  }
}

export function makeLocalRunner(cfg: () => LocalRunnerConfig): ArgvRunner {
  return (args, timeoutMs, stdin) =>
    new Promise((resolve, reject) => {
      const c = cfg();
      const child = spawn(c.python, [c.script, ...args], {
        cwd: c.cwd,
        windowsHide: true,
        env: { ...process.env, R016_BY: c.by, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      const out: Buffer[] = [];
      const err: Buffer[] = [];
      let done = false;
      const timer = setTimeout(() => {
        if (done) return;
        done = true;
        killTree(child.pid);
        reject(new Error(`로컬 명령 시간 초과 (${Math.round(timeoutMs / 1000)}s): r016 ${args[0] || ''}`));
      }, timeoutMs);
      child.stdout.on('data', (d: Buffer) => out.push(d));
      child.stderr.on('data', (d: Buffer) => err.push(d));
      child.on('error', (e) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        reject(new Error(`로컬 Python 실행 실패 (${c.python}): ${e.message}`));
      });
      child.on('close', (code) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        resolve({ out: Buffer.concat(out).toString('utf8'), err: Buffer.concat(err).toString('utf8'), code });
      });
      child.stdin.on('error', () => undefined); // the script may exit before reading stdin
      if (stdin !== undefined) child.stdin.end(stdin, 'utf8');
      else child.stdin.end();
    });
}
