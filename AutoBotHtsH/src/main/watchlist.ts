// Update log
// - [2026-11-19] type=feat (Watchlist groups kept as a file in %APPDATA%\htsh\watchlist\watchlist.json (was only the renderer's
//   localStorage, which cannot be seen or copied). The same folder takes selection files "<...>YYYYMMDD_picks.txt" (one symbol per
//   line, "code,name" - AutoBot g002 _YYYYMMDD_picks.txt): the renderer turns each into a watchlist tab named MMDD (user 2026-11-19))

import { app } from 'electron';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'fs';
import { join } from 'path';

export interface WatchGroupFile {
  id: string;
  name: string;
  codes: string[];
  /** picks file the group came from (kept in step with it while the file changes) */
  src?: string;
}
export interface WatchFile {
  groups: WatchGroupFile[];
  /** picks file name -> its mtime (ms) when it was turned into a tab; a deleted tab is not made again until the file changes */
  imported: Record<string, number>;
}
export interface PicksFile {
  file: string;
  /** YYYYMMDD of the file name */
  date: string;
  mtime: number;
  codes: string[];
}

export const watchDir = () => join(app.getPath('userData'), 'watchlist');
const watchFile = () => join(watchDir(), 'watchlist.json');
/** "_20261119_picks.txt", "20261119_picks.txt", "xx_20261119_picks.txt" */
export const PICKS_RE = /(?:^|_)(\d{8})_picks\.txt$/i;

function ensureDir(): string {
  const d = watchDir();
  if (!existsSync(d)) mkdirSync(d, { recursive: true });
  return d;
}

export function loadWatchFile(): WatchFile | null {
  try {
    const v = JSON.parse(readFileSync(watchFile(), 'utf8')) as Partial<WatchFile>;
    if (!Array.isArray(v.groups)) return null;
    const groups = v.groups.filter((g) => g && typeof g.id === 'string' && Array.isArray(g.codes));
    return { groups, imported: v.imported && typeof v.imported === 'object' ? v.imported : {} };
  } catch {
    return null;
  }
}

/** written to a temp file and renamed, so a crash never leaves half a file */
export function saveWatchFile(w: WatchFile): void {
  ensureDir();
  const tmp = `${watchFile()}.tmp`;
  writeFileSync(tmp, JSON.stringify(w, null, 1), 'utf8');
  renameSync(tmp, watchFile());
}

/** codes of a picks file: the first field of each line ("005930,삼성전자" / "005930 삼성전자" / "005930"); header and blank lines skipped */
export function parsePicks(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.replace(/^﻿/, '').split(/\r?\n/)) {
    const m = /^\s*([0-9A-Z]{6})(?=$|[\s,;\t|])/.exec(raw);
    if (m && !out.includes(m[1])) out.push(m[1]);
  }
  return out;
}

/** picks files in the watchlist folder (the folder is created on first use so it can be found) */
export function scanPicks(): PicksFile[] {
  const d = ensureDir();
  const out: PicksFile[] = [];
  for (const name of readdirSync(d)) {
    const m = PICKS_RE.exec(name);
    if (!m) continue;
    try {
      const p = join(d, name);
      const st = statSync(p);
      if (!st.isFile() || st.size > 1_000_000) continue;
      out.push({ file: name, date: m[1], mtime: Math.floor(st.mtimeMs), codes: parsePicks(readFileSync(p, 'utf8')) });
    } catch {
      /* a file being written - next scan */
    }
  }
  return out.sort((a, b) => a.date.localeCompare(b.date) || a.file.localeCompare(b.file));
}
