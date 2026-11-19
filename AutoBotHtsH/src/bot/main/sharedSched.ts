/*
 * Update log
 * - [2026-11-20] type=feat - Local-mode scheduler shared by AutoBotH and AutoBotHtsH (same file in both apps: AutoBotH
 *   src/main/sharedSched.ts, AutoBotHtsH src/bot/main/sharedSched.ts). Both can run on one PC; only one may run the scheduler (tick
 *   collector checks, the daily post-close job, the watchlist copy to the Pi), or the post-close job would start twice. The holder is
 *   written to %APPDATA%/autoboth-shared/scheduler.json {owner: {app, pid, at}} and refreshed on every scheduler tick; another app
 *   takes over when the holder's process is gone or its heartbeat is older than STALE_MS. The file also keeps the last post-close
 *   day / watchlist-copy day, so the app that takes over does not run the same day's job again (dates are YYYYMMDD; the later one wins)
 */

import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const STALE_MS = 3 * 60000;

export interface SharedSchedFile {
  owner?: { app: string; pid: number; at: number };
  lastPostcloseYmd?: string;
  lastPostcloseSyncYmd?: string;
}

const later = (a?: string, b?: string) => (!a ? b : !b ? a : a > b ? a : b);

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    // EPERM: the process exists but belongs to someone else
    return (e as NodeJS.ErrnoException).code === 'EPERM';
  }
}

export function createSharedSched(appName: string, dir: string, pid = process.pid, now: () => number = Date.now) {
  const file = join(dir, 'scheduler.json');
  const read = (): SharedSchedFile => {
    try {
      return JSON.parse(readFileSync(file, 'utf8')) as SharedSchedFile;
    } catch {
      return {};
    }
  };
  const write = (f: SharedSchedFile): void => {
    try {
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      const tmp = `${file}.${pid}.tmp`;
      writeFileSync(tmp, JSON.stringify(f), 'utf8');
      renameSync(tmp, file);
    } catch {
      /* no shared folder: this app schedules alone (as before) */
    }
  };
  const mine = (f: SharedSchedFile) => f.owner?.pid === pid && f.owner.app === appName;
  const free = (f: SharedSchedFile) => !f.owner || now() - f.owner.at > STALE_MS || !alive(f.owner.pid);

  return {
    /** take or keep the scheduler (heartbeat); false = another app holds it */
    hold(): boolean {
      const f = read();
      if (!mine(f) && !free(f)) return false;
      write({ ...f, owner: { app: appName, pid, at: now() } });
      // two apps writing at the same moment: the file decides
      return mine(read());
    },
    /** the app holding the scheduler (null = this app or nobody) */
    holder(): string | null {
      const f = read();
      return mine(f) || free(f) ? null : f.owner!.app;
    },
    release(): void {
      const f = read();
      if (!mine(f)) return;
      delete f.owner;
      write(f);
    },
    /** the app's own meta merged with the shared days */
    meta<M extends { lastPostcloseYmd?: string; lastPostcloseSyncYmd?: string }>(own: M): M {
      const f = read();
      return { ...own, lastPostcloseYmd: later(own.lastPostcloseYmd, f.lastPostcloseYmd), lastPostcloseSyncYmd: later(own.lastPostcloseSyncYmd, f.lastPostcloseSyncYmd) };
    },
    setMeta(p: { lastPostcloseYmd?: string; lastPostcloseSyncYmd?: string }): void {
      const f = read();
      write({ ...f, lastPostcloseYmd: later(f.lastPostcloseYmd, p.lastPostcloseYmd), lastPostcloseSyncYmd: later(f.lastPostcloseSyncYmd, p.lastPostcloseSyncYmd) });
    },
    file,
    /** test helper */
    reset(): void {
      try {
        unlinkSync(file);
      } catch {
        /* none */
      }
    },
  };
}

export type SharedSched = ReturnType<typeof createSharedSched>;
