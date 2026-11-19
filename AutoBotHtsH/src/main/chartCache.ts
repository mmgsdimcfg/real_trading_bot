// Update log
// - [2026-11-19] type=perf (Chart data kept on disk in %APPDATA%\htsh\chartcache\ (user 2026-11-19: charts waited for KIS on every open).
//   One JSON file per code / bar kind / server; read once into memory, written at most every WRITE_MS per key (and on quit) through a
//   temp file + rename; files unused for MAX_AGE_DAYS are removed at start. The KIS backend serves a cached chart at once and fetches
//   only what is missing)

import { app } from 'electron';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from 'fs';
import { join } from 'path';

const WRITE_MS = 30000;
const MAX_AGE_DAYS = 45;

export interface ChartStore {
  get<T>(key: string): T | undefined;
  put<T>(key: string, v: T): void;
  flush(): void;
}

const fileName = (key: string) => `${key.replace(/[^0-9A-Za-z_-]+/g, '_')}.json`;

export class ChartDiskCache implements ChartStore {
  private dir: string;
  private dirty = new Map<string, unknown>();
  private lastWrite = new Map<string, number>();
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(dir = join(app.getPath('userData'), 'chartcache')) {
    this.dir = dir;
    try {
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      const old = Date.now() - MAX_AGE_DAYS * 86400000;
      for (const f of readdirSync(dir)) {
        const p = join(dir, f);
        try {
          if (statSync(p).mtimeMs < old || f.endsWith('.tmp')) unlinkSync(p);
        } catch {
          /* in use - next start */
        }
      }
    } catch {
      /* no cache folder: works without it */
    }
  }

  get<T>(key: string): T | undefined {
    if (this.dirty.has(key)) return this.dirty.get(key) as T;
    try {
      return JSON.parse(readFileSync(join(this.dir, fileName(key)), 'utf8')) as T;
    } catch {
      return undefined;
    }
  }

  put<T>(key: string, v: T): void {
    this.dirty.set(key, v);
    const due = (this.lastWrite.get(key) || 0) + WRITE_MS - Date.now();
    if (due <= 0) this.write(key);
    else if (!this.timer) this.timer = setTimeout(() => ((this.timer = null), this.flush()), Math.max(1000, due));
  }

  flush(): void {
    for (const k of [...this.dirty.keys()]) this.write(k);
  }

  private write(key: string): void {
    const v = this.dirty.get(key);
    this.dirty.delete(key);
    this.lastWrite.set(key, Date.now());
    if (v === undefined) return;
    try {
      const p = join(this.dir, fileName(key));
      writeFileSync(`${p}.tmp`, JSON.stringify(v), 'utf8');
      renameSync(`${p}.tmp`, p);
    } catch {
      /* disk full / folder gone: the memory copy still serves */
    }
  }
}
