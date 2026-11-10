/*
 * Update log
 * - [2026-11-10] type=feat - Where the bot's files are read from: FileHost = the methods the log/price readers need (readdir, stat,
 *   ranged read, "<prefix>*" listing). SshConnection (Raspberry Pi over SFTP) and LocalHost (this PC's runtime folder, Node fs)
 *   implement it; HostSwitch forwards to the one of the current run mode, so DaySession / PriceService do not care where r003 runs
 */

import { open, readdir, stat } from 'node:fs/promises';
import type { RunMode } from '../shared/types';

export interface RemoteEntry {
  name: string;
  size: number;
  mtime: number;
  isDir: boolean;
}

export interface FileHost {
  readonly kind: RunMode;
  readonly connected: boolean;
  readdir(dir: string): Promise<RemoteEntry[]>;
  /** size / mtime of one file (null when it does not exist) */
  stat(path: string): Promise<{ size: number; mtime: number } | null>;
  /** bytes [start, end) */
  readRange(path: string, start: number, end: number): Promise<Buffer>;
  /** names in `dir` starting with `prefix` (missing dir = []) */
  listPrefix(dir: string, prefix: string): Promise<string[]>;
}

/** This PC's files (local mode). Always "connected" once the local root is valid. */
export class LocalHost implements FileHost {
  readonly kind = 'local' as const;
  connected = false;

  async readdir(dir: string): Promise<RemoteEntry[]> {
    const list = await readdir(dir, { withFileTypes: true });
    const out: RemoteEntry[] = [];
    for (const d of list) {
      try {
        const st = await stat(`${dir}/${d.name}`);
        out.push({ name: d.name, size: st.size, mtime: st.mtimeMs, isDir: st.isDirectory() });
      } catch {
        /* vanished between readdir and stat */
      }
    }
    return out;
  }

  async stat(path: string): Promise<{ size: number; mtime: number } | null> {
    try {
      const st = await stat(path);
      return { size: st.size, mtime: st.mtimeMs };
    } catch {
      return null;
    }
  }

  async readRange(path: string, start: number, end: number): Promise<Buffer> {
    if (end <= start) return Buffer.alloc(0);
    const fh = await open(path, 'r');
    try {
      const buf = Buffer.alloc(end - start);
      let off = 0;
      while (off < buf.length) {
        const { bytesRead } = await fh.read(buf, off, buf.length - off, start + off);
        if (bytesRead <= 0) break;
        off += bytesRead;
      }
      return off === buf.length ? buf : buf.subarray(0, off);
    } finally {
      await fh.close();
    }
  }

  async listPrefix(dir: string, prefix: string): Promise<string[]> {
    try {
      return (await readdir(dir)).filter((n) => n.startsWith(prefix)).sort();
    } catch {
      return [];
    }
  }
}

/** Forwards to the host of the current run mode. */
export class HostSwitch implements FileHost {
  constructor(
    private remote: FileHost,
    private local: FileHost,
    private mode: () => RunMode,
  ) {}

  private get cur(): FileHost {
    return this.mode() === 'local' ? this.local : this.remote;
  }
  get kind(): RunMode {
    return this.cur.kind;
  }
  get connected(): boolean {
    return this.cur.connected;
  }
  readdir(dir: string) {
    return this.cur.readdir(dir);
  }
  stat(path: string) {
    return this.cur.stat(path);
  }
  readRange(path: string, start: number, end: number) {
    return this.cur.readRange(path, start, end);
  }
  listPrefix(dir: string, prefix: string) {
    return this.cur.listPrefix(dir, prefix);
  }
}
