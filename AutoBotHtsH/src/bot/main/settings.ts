/*
 * Update log
 * - [2026-11-10] type=fix - detectLocalRoot() also tries D:/C:\make_utils\Trading (the PC folders moved under make_utils/Trading)
 * - [2026-11-04] type=feat - Settings in %APPDATA%/autoboth/autoboth-settings.json (electron-store). The key passphrase / password is
 *   stored only encrypted with safeStorage (DPAPI on Windows) - never in plain text and never sent back to the renderer.
 *   Settings from v1 (pemPath) are taken over as keyPath
 * - [2026-11-05] type=feat - SSH proxy settings (proxyType/proxyCommand/proxyHost/proxyPort/proxyUser); the proxy password is stored
 *   encrypted with safeStorage as proxySecretEnc, like the SSH secret. getProxyConfig() builds the ProxyConfig for ssh.connect
 * - [2026-11-07] type=feat - sessionOpenAuto (default on): the funnel start time follows the server NXT setting; stores without
 *   the key get it on
 * - [2026-11-07] type=feat - statusPollSec (r016 status poll, default 30 s, min 15 s) separate from pollSec (log poll); stores without
 *   the key get the default
 * - [2026-11-10] type=feat - Local mode settings (runMode, localRoot - auto-detected from the app/repo location, localPython,
 *   local poll intervals, scheduler switches, keepAwake) + localPaths() helper and a small meta store (last post-close run day)
 */

import Store from 'electron-store';
import { app, safeStorage } from 'electron';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import type { Settings } from '../shared/types';
import type { ProxyConfig } from './proxy';
import { STATUS_POLL_DEFAULT_SEC, STATUS_POLL_MIN_SEC } from '../shared/control';

interface Stored extends Omit<Settings, 'secret' | 'hasSecret' | 'proxySecret' | 'hasProxySecret'> {
  secretEnc: string;
  proxySecretEnc: string;
  pemPath?: string; // v1
}

const DEFAULTS: Stored = {
  host: '',
  port: 22,
  username: '',
  authMethod: 'key',
  keyPath: '',
  secretEnc: '',
  proxyType: '',
  proxyCommand: '',
  proxyHost: '',
  proxyPort: 0,
  proxyUser: '',
  proxySecretEnc: '',
  remoteRoot: '/data/srv/autobot',
  pythonPath: '~/venv/bin/python',
  pollSec: 10,
  statusPollSec: STATUS_POLL_DEFAULT_SEC,
  sessionOpen: '09:00',
  sessionOpenAuto: true,
  retentionDays: 30,
  runMode: 'remote',
  localRoot: '',
  localPython: '',
  localPollSec: 2,
  localStatusPollSec: 5,
  autoCollector: true,
  autoPostclose: false,
  postcloseTime: '20:10',
  postcloseEdgeAi: false,
  postcloseEmail: false,
  syncWatchlistToPeer: true,
  keepAwake: true,
};

const R016_REL = ['real_trading_bot', 'AutoBot', 'live_trading', 'r016_control.py'];
const isLocalRoot = (dir: string) => !!dir && existsSync(join(dir, ...R016_REL));

/** First folder above the app (dev: <root>/real_trading_bot/AutoBotH) or a usual place that holds real_trading_bot/AutoBot. */
export function detectLocalRoot(): string {
  const cands: string[] = [];
  try {
    let d = resolve(app.getAppPath());
    for (let i = 0; i < 5; i++) {
      cands.push(d);
      d = dirname(d);
    }
  } catch {
    /* no app path in tests */
  }
  cands.push('D:\\make_utils\\Trading', 'C:\\make_utils\\Trading', 'D:\\make_utils', 'C:\\make_utils');
  return cands.find(isLocalRoot) || '';
}

export interface LocalPaths {
  root: string;
  codeDir: string;
  r016: string;
  python: string;
  runtime: string;
  logDir: string;
}

/** Paths of the local mode (forward slashes - Node fs accepts them on Windows; same joins as the remote paths). */
export function localPaths(s: Settings = getSettings()): LocalPaths {
  const root = (s.localRoot || '').replace(/[\\/]+$/, '').replace(/\\/g, '/');
  const python = (s.localPython || '').trim() || `${root}/venv/Scripts/python.exe`;
  return {
    root,
    codeDir: `${root}/real_trading_bot/AutoBot`,
    r016: `${root}/real_trading_bot/AutoBot/live_trading/r016_control.py`,
    python,
    runtime: `${root}/runtime`,
    logDir: `${root}/runtime/log`,
  };
}

/** Why the local mode cannot start ('' = ok). */
export function localProblem(s: Settings = getSettings()): string {
  const p = localPaths(s);
  if (!p.root) return '로컬 루트 폴더가 설정되지 않았습니다 (real_trading_bot / runtime / venv 가 있는 폴더).';
  if (!existsSync(p.r016)) return `로컬 루트에 ${R016_REL.join('/')} 가 없습니다: ${p.root}`;
  if (!existsSync(p.python)) return `로컬 Python 이 없습니다: ${p.python} (venv 를 만들고 패키지를 설치하세요 - 설정/연결 탭의 안내 참고)`;
  return '';
}

/** small internal values that are not user settings */
interface Meta {
  lastPostcloseYmd?: string;
  lastPostcloseSyncYmd?: string;
  /** the local mode connected at least once (a detached local r003 may exist even when the local settings break later) */
  localEverUsed?: boolean;
}
const metaStore = new Store<Meta>({ name: 'autoboth-meta', defaults: {} });
export const getMeta = (): Meta => metaStore.store;
export const setMeta = (patch: Meta): void => metaStore.set({ ...metaStore.store, ...patch });

const store = new Store<Stored>({ name: 'autoboth-settings', defaults: DEFAULTS });
if (!store.get('localRoot')) {
  const d = detectLocalRoot();
  if (d) store.set('localRoot', d);
}

// v1 -> v2: pemPath was the key file
if (!store.get('keyPath') && store.get('pemPath')) store.set('keyPath', store.get('pemPath') as string);

export function getSettings(): Settings {
  const s = store.store;
  return {
    host: s.host,
    port: s.port,
    username: s.username,
    authMethod: s.authMethod,
    keyPath: s.keyPath,
    proxyType: s.proxyType || '',
    proxyCommand: s.proxyCommand || '',
    proxyHost: s.proxyHost || '',
    proxyPort: s.proxyPort || 0,
    proxyUser: s.proxyUser || '',
    hasProxySecret: !!s.proxySecretEnc,
    remoteRoot: s.remoteRoot,
    pythonPath: s.pythonPath,
    pollSec: s.pollSec,
    statusPollSec: Math.max(STATUS_POLL_MIN_SEC, Number(s.statusPollSec) || STATUS_POLL_DEFAULT_SEC),
    sessionOpen: s.sessionOpen,
    sessionOpenAuto: s.sessionOpenAuto !== false,
    retentionDays: s.retentionDays,
    hasSecret: !!s.secretEnc,
    runMode: s.runMode === 'local' ? 'local' : 'remote',
    localRoot: s.localRoot || '',
    localPython: s.localPython || '',
    localPollSec: Math.max(1, Number(s.localPollSec) || DEFAULTS.localPollSec),
    localStatusPollSec: Math.max(3, Number(s.localStatusPollSec) || DEFAULTS.localStatusPollSec),
    autoCollector: s.autoCollector !== false,
    autoPostclose: s.autoPostclose === true,
    postcloseTime: /^\d{1,2}:\d{2}$/.test(String(s.postcloseTime || '')) ? s.postcloseTime : DEFAULTS.postcloseTime,
    postcloseEdgeAi: s.postcloseEdgeAi === true,
    postcloseEmail: s.postcloseEmail === true,
    syncWatchlistToPeer: s.syncWatchlistToPeer !== false,
    keepAwake: s.keepAwake !== false,
  };
}

export function saveSettings(next: Settings): Settings {
  const { secret, hasSecret: _h, proxySecret, hasProxySecret: _hp, ...rest } = next;
  const proxyType = ['command', 'socks5', 'http'].includes(rest.proxyType) ? rest.proxyType : '';
  if (proxyType === 'command' && !String(rest.proxyCommand || '').trim()) throw new Error('ProxyCommand를 입력하세요.');
  if ((proxyType === 'socks5' || proxyType === 'http') && !String(rest.proxyHost || '').trim()) throw new Error('프록시 호스트를 입력하세요.');
  store.set({
    ...rest,
    proxyType,
    proxyCommand: String(rest.proxyCommand || '').trim(),
    proxyHost: String(rest.proxyHost || '').trim(),
    proxyPort: Math.max(0, Number(rest.proxyPort) || 0),
    proxyUser: String(rest.proxyUser || ''),
    port: Number(rest.port) || 22,
    pollSec: Math.max(3, Number(rest.pollSec) || 10),
    statusPollSec: Math.max(STATUS_POLL_MIN_SEC, Number(rest.statusPollSec) || STATUS_POLL_DEFAULT_SEC),
    retentionDays: Math.max(1, Number(rest.retentionDays) || 30),
    sessionOpenAuto: rest.sessionOpenAuto !== false,
    sessionOpen: /^\d{1,2}:\d{2}$/.test(String(rest.sessionOpen || '').trim()) ? String(rest.sessionOpen).trim() : '09:00',
    remoteRoot: String(rest.remoteRoot || DEFAULTS.remoteRoot).replace(/\/+$/, ''),
    runMode: rest.runMode === 'local' ? 'local' : 'remote',
    localRoot: String(rest.localRoot || '').trim().replace(/[\\/]+$/, ''),
    localPython: String(rest.localPython || '').trim(),
    localPollSec: Math.max(1, Number(rest.localPollSec) || DEFAULTS.localPollSec),
    localStatusPollSec: Math.max(3, Number(rest.localStatusPollSec) || DEFAULTS.localStatusPollSec),
    autoCollector: rest.autoCollector !== false,
    autoPostclose: rest.autoPostclose === true,
    postcloseTime: /^\d{1,2}:\d{2}$/.test(String(rest.postcloseTime || '').trim()) ? String(rest.postcloseTime).trim() : DEFAULTS.postcloseTime,
    postcloseEdgeAi: rest.postcloseEdgeAi === true,
    postcloseEmail: rest.postcloseEmail === true,
    syncWatchlistToPeer: rest.syncWatchlistToPeer !== false,
    keepAwake: rest.keepAwake !== false,
  });
  if (secret !== undefined) storeEncrypted('secretEnc', secret);
  if (proxySecret !== undefined) storeEncrypted('proxySecretEnc', proxySecret);
  return getSettings();
}

function storeEncrypted(key: 'secretEnc' | 'proxySecretEnc', value: string): void {
  if (!value) store.set(key, '');
  else {
    if (!safeStorage.isEncryptionAvailable()) throw new Error('OS 암호화(safeStorage)를 사용할 수 없어 비밀번호를 저장하지 않았습니다.');
    store.set(key, safeStorage.encryptString(value).toString('base64'));
  }
}

function readEncrypted(key: 'secretEnc' | 'proxySecretEnc'): string | undefined {
  const enc = store.get(key);
  if (!enc) return undefined;
  try {
    return safeStorage.decryptString(Buffer.from(enc, 'base64'));
  } catch {
    return undefined;
  }
}

/** Decrypted passphrase/password for the SSH connection (main process only). */
export function getSecret(): string | undefined {
  return readEncrypted('secretEnc');
}

/** Proxy settings for ssh.connect (main process only) - undefined = direct connection. */
export function getProxyConfig(): ProxyConfig | undefined {
  const s = store.store;
  if (s.proxyType === 'command') return { type: 'command', command: s.proxyCommand || '' };
  if (s.proxyType === 'socks5' || s.proxyType === 'http') {
    return {
      type: s.proxyType,
      host: s.proxyHost || '',
      port: Number(s.proxyPort) || (s.proxyType === 'socks5' ? 1080 : 8080),
      username: s.proxyUser || '',
      password: readEncrypted('proxySecretEnc') || '',
    };
  }
  return undefined;
}
