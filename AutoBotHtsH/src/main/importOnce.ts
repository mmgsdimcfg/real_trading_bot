// Update log
// - [2026-11-20] type=feat (AutoBotHtsH first start: settings of the two apps it combines are copied into its own folder
//   (%APPDATA%\autobothtsh) when they are not there yet - AutoBotH connection settings (autoboth-settings.json without the encrypted
//   SSH / proxy secrets: safeStorage keys belong to each app, so the password is asked again), its meta and condition cache, and the
//   HtsH watchlist file. Imported before any settings store is created (first import of src/main/index.ts). Never writes to the
//   other apps' folders)
// - Screen tests: AUTOBOTHTSH_USERDATA=<dir> uses that folder as the app folder and imports nothing (no bot connection, no scheduler)

import { app } from 'electron';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export interface ImportResult {
  copied: string[];
  skipped: string[];
}

/** copy `from` to `to` unless `to` exists; `edit` may change parsed JSON on the way */
function copyOnce(from: string, to: string, out: ImportResult, edit?: (j: Record<string, unknown>) => void): void {
  if (existsSync(to) || !existsSync(from)) return void out.skipped.push(to);
  try {
    mkdirSync(dirname(to), { recursive: true });
    if (edit) {
      const j = JSON.parse(readFileSync(from, 'utf8')) as Record<string, unknown>;
      edit(j);
      writeFileSync(to, JSON.stringify(j, null, '\t'), 'utf8');
    } else copyFileSync(from, to);
    out.copied.push(to);
  } catch {
    out.skipped.push(to);
  }
}

export function importFromSiblingApps(appData = app.getPath('appData'), userData = app.getPath('userData')): ImportResult {
  const out: ImportResult = { copied: [], skipped: [] };
  const bot = join(appData, 'autoboth');
  copyOnce(join(bot, 'autoboth-settings.json'), join(userData, 'autoboth-settings.json'), out, (j) => {
    j.secretEnc = '';
    j.proxySecretEnc = '';
  });
  copyOnce(join(bot, 'autoboth-meta.json'), join(userData, 'autoboth-meta.json'), out);
  copyOnce(join(bot, 'conditions_cache.json'), join(userData, 'conditions_cache.json'), out);
  copyOnce(join(appData, 'htsh', 'watchlist', 'watchlist.json'), join(userData, 'watchlist', 'watchlist.json'), out);
  return out;
}

export const imported: ImportResult = (() => {
  const testDir = process.env.AUTOBOTHTSH_USERDATA;
  if (testDir) {
    app.setPath('userData', testDir);
    return { copied: [], skipped: [] };
  }
  try {
    return importFromSiblingApps();
  } catch {
    return { copied: [], skipped: [] };
  }
})();
