/*
 * Update log
 * - [2026-11-05] type=feat - Condition list now comes from `r016_control.py conditions` (same buy/sell structure as r015 plus
 *   values_meta / overrides / override_errors for the value editor), run through the control client's serial queue. A server
 *   without r016 falls back to r015_export_conditions.py (read-only display, no values_meta)
 * - [2026-11-04] type=feat - Condition list (source of truth) from the server: runs live_trading/r015_export_conditions.py over SSH
 *   (read-only export script) and parses its JSON stdout. The last good copy is cached in userData; if both fail, the snapshot
 *   bundled with the app (resources/conditions_snapshot.json) is used
 */

import { app } from 'electron';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ConditionsDoc } from '../shared/types';
import { shellPath, shellQuote, type SshConnection } from './ssh';
import { R016_CONDITIONS_TIMEOUT, type R016Client } from './control';

const cacheFile = () => join(app.getPath('userData'), 'conditions_cache.json');
const bundledFile = () => join(app.getAppPath(), 'resources', 'conditions_snapshot.json');

function validate(doc: unknown): ConditionsDoc {
  const d = doc as ConditionsDoc;
  if (!d || !Array.isArray(d.buy) || !Array.isArray(d.sell)) throw new Error('조건 JSON 형식이 올바르지 않습니다 (buy/sell 배열 없음).');
  for (const c of [...d.buy, ...d.sell]) {
    if (typeof c.order !== 'number' || typeof c.no !== 'number') throw new Error(`조건 항목 형식 오류: ${JSON.stringify(c).slice(0, 120)}`);
  }
  return d;
}

export function loadLocalConditions(): ConditionsDoc {
  try {
    if (existsSync(cacheFile())) return { ...validate(JSON.parse(readFileSync(cacheFile(), 'utf8'))), source: 'cache' };
  } catch {
    /* fall through to the bundled copy */
  }
  return { ...validate(JSON.parse(readFileSync(bundledFile(), 'utf8'))), source: 'bundled' };
}

async function fetchR015(ssh: SshConnection, remoteRoot: string, pythonPath: string): Promise<ConditionsDoc> {
  const cmd = `cd ${shellQuote(remoteRoot + '/auto_trading_src')} && ${shellPath(pythonPath)} live_trading/r015_export_conditions.py`;
  const { out, err, code } = await ssh.exec(cmd, 90000);
  const a = out.indexOf('{');
  const b = out.lastIndexOf('}');
  if (code !== 0 || a < 0 || b < a) {
    throw new Error(`조건 JSON 내보내기 실패 (exit=${code}): ${(err || out).trim().split('\n').slice(-3).join(' / ').slice(0, 300)}`);
  }
  return validate(JSON.parse(out.slice(a, b + 1)));
}

export async function fetchConditions(ssh: SshConnection, r016: R016Client, remoteRoot: string, pythonPath: string): Promise<ConditionsDoc> {
  let doc: ConditionsDoc;
  const r = await r016.run<ConditionsDoc>(['conditions'], { timeoutMs: R016_CONDITIONS_TIMEOUT });
  if (r.ok && r.data) doc = validate(r.data);
  else {
    try {
      doc = await fetchR015(ssh, remoteRoot, pythonPath);
    } catch (e) {
      throw new Error(`r016 conditions: ${r.error} / r015: ${(e as Error).message}`);
    }
  }
  try {
    writeFileSync(cacheFile(), JSON.stringify(doc), 'utf8');
  } catch {
    /* cache is best effort */
  }
  return { ...doc, source: 'server' };
}
