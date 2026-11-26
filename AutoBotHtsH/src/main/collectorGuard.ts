// Update log
// - [2026-11-26] type=feat (Bot first (user 2026-11-26): KIS allows one realtime session per app key. While the bot's tick collector
//   (AutoBot g011) runs on this PC, the app must not hold that session - on 2026-11-25 / 26 the collector could not connect for hours
//   and the rule strategy abstained for lack of tick files. localCollectorPid() tells whether the collector process of today is alive
//   (its lock file <localRoot>/runtime/data/ticks/.g011.lock: "<pid> <YYYY-MM-DD HH:MM:SS> ppid=<n>"); main/index.ts pauses the feed)

import { readFileSync } from 'fs';
import { localPaths } from '../bot/main/settings';
import { parseCollectorLock } from '../shared/collectorLock';
import { kstYmd } from '../shared/reboundReplay';

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0); // signal 0: existence test only
    return true;
  } catch (e) {
    return (e as NodeJS.ErrnoException).code === 'EPERM'; // exists, owned by someone else
  }
}

/** pid of the tick collector running on this PC today, or null (no local runtime, no lock, an old lock, a dead process) */
export function localCollectorPid(now = Date.now()): number | null {
  try {
    const root = localPaths().root;
    if (!root) return null;
    const pid = parseCollectorLock(readFileSync(`${root}/runtime/data/ticks/.g011.lock`, 'utf8'), kstYmd(now));
    return pid !== null && alive(pid) ? pid : null;
  } catch {
    return null;
  }
}
