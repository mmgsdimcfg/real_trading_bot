/*
 * Update log
 * - [2026-11-07] type=feat - Local copy of a PAST day's price files in %APPDATA%/autoboth/pricecache/YYYYMMDD/ (a past day's server
 *   files no longer change): the tick file ticks/<code>.csv, the "<code>_*" file list of runtime/data/<date> and the files the
 *   charts read from it (_10s.txt / _daily.csv). Each file is downloaded whole into a temp file and renamed, so a dropped
 *   connection never leaves half a file. A file the server does not have is remembered too (index.json), so an offline view does
 *   not wait for the server. The list of runtime/data folders is kept in pricecache/folders.json for offline warm-up lookups.
 *   Folders older than the retention setting are removed with the log cache. Today's (live) files are never cached here
 */

import { app } from 'electron';
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync, appendFileSync, unlinkSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { SshConnection } from './ssh';

const CHUNK = 4 * 1024 * 1024;

export const priceCacheRoot = () => join(app.getPath('userData'), 'pricecache');
const dayDir = (ymd: string) => join(priceCacheRoot(), ymd);

interface DayIndex {
  /** relative paths the server did not have */
  missing: string[];
  /** code -> names of runtime/data/<date>/<code>_* */
  lists: Record<string, string[]>;
}

export class PriceFileCache {
  private idx = new Map<string, DayIndex>();

  constructor(private ssh: SshConnection) {}

  private index(ymd: string): DayIndex {
    let i = this.idx.get(ymd);
    if (!i) {
      i = { missing: [], lists: {} };
      try {
        const v = JSON.parse(readFileSync(join(dayDir(ymd), 'index.json'), 'utf8'));
        if (Array.isArray(v?.missing)) i.missing = v.missing.filter((x: unknown) => typeof x === 'string');
        if (v?.lists && typeof v.lists === 'object') i.lists = v.lists;
      } catch {
        /* no index yet */
      }
      this.idx.set(ymd, i);
    }
    return i;
  }

  private saveIndex(ymd: string): void {
    try {
      mkdirSync(dayDir(ymd), { recursive: true });
      const tmp = join(dayDir(ymd), `index.json.${process.pid}.tmp`);
      writeFileSync(tmp, JSON.stringify(this.index(ymd)));
      renameSync(tmp, join(dayDir(ymd), 'index.json'));
    } catch {
      /* the index is a convenience */
    }
  }

  /** local file only (no server access): Buffer, null = the server did not have it, undefined = not cached */
  peek(ymd: string, rel: string): Buffer | null | undefined {
    const p = join(dayDir(ymd), rel);
    if (existsSync(p)) return readFileSync(p);
    return this.index(ymd).missing.includes(rel) ? null : undefined;
  }

  /** The whole remote file through the local copy. null = the server has no such file. Throws when offline and not cached. */
  async file(ymd: string, rel: string, remotePath: string): Promise<Buffer | null> {
    const have = this.peek(ymd, rel);
    if (have !== undefined) return have;
    if (!this.ssh.connected) throw new Error('서버에 연결되어 있지 않고 이 파일의 로컬 사본도 없습니다.');
    const st = await this.ssh.stat(remotePath);
    if (!st) {
      const i = this.index(ymd);
      if (!i.missing.includes(rel)) i.missing.push(rel);
      this.saveIndex(ymd);
      return null;
    }
    const dest = join(dayDir(ymd), rel);
    const tmp = `${dest}.${process.pid}.part`;
    mkdirSync(dirname(dest), { recursive: true });
    try {
      writeFileSync(tmp, Buffer.alloc(0));
      for (let off = 0; off < st.size; ) {
        const buf = await this.ssh.readRange(remotePath, off, Math.min(st.size, off + CHUNK));
        if (!buf.length) break;
        appendFileSync(tmp, buf);
        off += buf.length;
      }
      if (statSync(tmp).size !== st.size) throw new Error(`파일을 끝까지 받지 못했습니다: ${remotePath}`);
      renameSync(tmp, dest);
    } finally {
      if (existsSync(tmp)) unlinkSync(tmp);
    }
    return readFileSync(dest);
  }

  /** "<code>_*" names of runtime/data/<date> (cached), listed with `ls` over SSH the first time */
  async list(ymd: string, code: string, fetch: () => Promise<string[]>): Promise<string[]> {
    const i = this.index(ymd);
    if (i.lists[code]) return i.lists[code];
    if (!this.ssh.connected) throw new Error('서버에 연결되어 있지 않고 이 종목의 파일 목록 사본도 없습니다.');
    const names = await fetch();
    i.lists[code] = names;
    this.saveIndex(ymd);
    return names;
  }

  /** small JSON answer kept with the day (e.g. the server's daily bars of a past day); undefined = none */
  json<T>(ymd: string, rel: string): T | undefined {
    try {
      return JSON.parse(readFileSync(join(dayDir(ymd), rel), 'utf8')) as T;
    } catch {
      return undefined;
    }
  }

  saveJson(ymd: string, rel: string, v: unknown): void {
    try {
      const dest = join(dayDir(ymd), rel);
      mkdirSync(dirname(dest), { recursive: true });
      const tmp = `${dest}.${process.pid}.tmp`;
      writeFileSync(tmp, JSON.stringify(v));
      renameSync(tmp, dest);
    } catch {
      /* optional */
    }
  }

  /** last known list of runtime/data/YYYYMMDD folders (offline warm-up / daily lookups) */
  saveFolders(list: string[]): void {
    try {
      mkdirSync(priceCacheRoot(), { recursive: true });
      writeFileSync(join(priceCacheRoot(), 'folders.json'), JSON.stringify(list));
    } catch {
      /* optional */
    }
  }

  loadFolders(): string[] {
    try {
      const v = JSON.parse(readFileSync(join(priceCacheRoot(), 'folders.json'), 'utf8'));
      return Array.isArray(v) ? v.filter((x) => typeof x === 'string' && /^\d{8}$/.test(x)) : [];
    } catch {
      return [];
    }
  }
}

/** Remove price cache folders older than `days` (by folder date), like the log cache. */
export function prunePriceCache(days: number): void {
  const root = priceCacheRoot();
  if (!existsSync(root)) return;
  const limit = Date.now() - days * 86400000;
  for (const d of readdirSync(root)) {
    const m = /^(\d{4})(\d{2})(\d{2})$/.exec(d);
    if (!m) continue;
    if (Date.UTC(+m[1], +m[2] - 1, +m[3]) < limit) rmSync(join(root, d), { recursive: true, force: true });
  }
}
