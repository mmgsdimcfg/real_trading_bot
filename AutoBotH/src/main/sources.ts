/*
 * Update log
 * - [2026-11-04] type=feat - Where a day's log files come from: the server over SFTP (with a local byte cache in
 *   %APPDATA%/autoboth/logcache/YYYYMMDD so a restart or a past date does not download the same bytes again), or a local folder
 *   (fixtures / the cache itself when offline). Cache folders older than the retention period are removed at start
 */

import { app } from 'electron';
import { appendFileSync, existsSync, mkdirSync, openSync, readSync, closeSync, readdirSync, rmSync, statSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import type { SshConnection } from './ssh';

export interface SourceFile {
  name: string;
  size: number;
}

export interface LogSource {
  readonly kind: 'remote' | 'local';
  readonly label: string;
  list(): Promise<SourceFile[]>;
  read(name: string, start: number, end: number): Promise<Buffer>;
  invalidate(name: string): void;
}

export const cacheRoot = () => join(app.getPath('userData'), 'logcache');
export const cacheDirFor = (ymd: string) => join(cacheRoot(), ymd);

function readLocal(path: string, start: number, end: number): Buffer {
  const len = Math.max(0, end - start);
  const buf = Buffer.alloc(len);
  const fd = openSync(path, 'r');
  try {
    let off = 0;
    while (off < len) {
      const n = readSync(fd, buf, off, len - off, start + off);
      if (n <= 0) break;
      off += n;
    }
    return off === len ? buf : buf.subarray(0, off);
  } finally {
    closeSync(fd);
  }
}

export class LocalDirSource implements LogSource {
  readonly kind = 'local' as const;
  constructor(
    readonly dir: string,
    readonly label = dir,
  ) {}
  async list(): Promise<SourceFile[]> {
    if (!existsSync(this.dir)) return [];
    return readdirSync(this.dir, { withFileTypes: true })
      .filter((d) => d.isFile())
      .map((d) => ({ name: d.name, size: statSync(join(this.dir, d.name)).size }));
  }
  async read(name: string, start: number, end: number): Promise<Buffer> {
    return readLocal(join(this.dir, name), start, end);
  }
  invalidate(): void {
    /* local files are not ours to touch */
  }
}

/** Only these files are cached (the ones the viewer actually reads). */
const CACHEABLE = /^(r003_trade_live_execute_\d{8}_\d{6}\.log|\d{8}_buy_sell\.log)$/;

export class RemoteDaySource implements LogSource {
  readonly kind = 'remote' as const;
  readonly label: string;
  private cacheDir: string;

  constructor(
    private ssh: SshConnection,
    readonly remoteDir: string,
    ymd: string,
  ) {
    this.label = remoteDir;
    this.cacheDir = cacheDirFor(ymd);
  }

  async list(): Promise<SourceFile[]> {
    const list = await this.ssh.readdir(this.remoteDir);
    return list.filter((e) => !e.isDir).map((e) => ({ name: e.name, size: e.size }));
  }

  private cachedSize(name: string): number {
    const p = join(this.cacheDir, name);
    return existsSync(p) ? statSync(p).size : 0;
  }

  async read(name: string, start: number, end: number): Promise<Buffer> {
    if (!CACHEABLE.test(name)) return this.ssh.readRange(`${this.remoteDir}/${name}`, start, end);
    const local = join(this.cacheDir, name);
    const have = this.cachedSize(name);
    const parts: Buffer[] = [];
    let pos = start;
    if (pos < have) {
      const upto = Math.min(end, have);
      parts.push(readLocal(local, pos, upto));
      pos = upto;
    }
    if (pos < end) {
      const remote = await this.ssh.readRange(`${this.remoteDir}/${name}`, pos, end);
      if (pos === have) {
        mkdirSync(this.cacheDir, { recursive: true });
        appendFileSync(local, remote);
      }
      parts.push(remote);
    }
    return parts.length === 1 ? parts[0] : Buffer.concat(parts);
  }

  /** remote file shrank (rotated/rewritten) - drop our copy */
  invalidate(name: string): void {
    const p = join(this.cacheDir, name);
    if (existsSync(p)) unlinkSync(p);
  }
}

/** Remove cache folders older than `days` (by folder date). */
export function pruneCache(days: number): void {
  const root = cacheRoot();
  if (!existsSync(root)) return;
  const limit = Date.now() - days * 86400000;
  for (const d of readdirSync(root)) {
    const m = /^(\d{4})(\d{2})(\d{2})$/.exec(d);
    if (!m) continue;
    if (Date.UTC(+m[1], +m[2] - 1, +m[3]) < limit) rmSync(join(root, d), { recursive: true, force: true });
  }
}

export function cachedDates(): string[] {
  const root = cacheRoot();
  if (!existsSync(root)) return [];
  return readdirSync(root).filter((d) => /^\d{8}$/.test(d));
}
