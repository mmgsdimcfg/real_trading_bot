/*
 * Update log
 * - [2026-11-04] type=feat - Settings in %APPDATA%/autoboth/autoboth-settings.json (electron-store). The key passphrase / password is
 *   stored only encrypted with safeStorage (DPAPI on Windows) - never in plain text and never sent back to the renderer.
 *   Settings from v1 (pemPath) are taken over as keyPath
 * - [2026-11-05] type=feat - SSH proxy settings (proxyType/proxyCommand/proxyHost/proxyPort/proxyUser); the proxy password is stored
 *   encrypted with safeStorage as proxySecretEnc, like the SSH secret. getProxyConfig() builds the ProxyConfig for ssh.connect
 * - [2026-11-07] type=feat - statusPollSec (r016 status poll, default 30 s, min 15 s) separate from pollSec (log poll); stores without
 *   the key get the default
 */

import Store from 'electron-store';
import { safeStorage } from 'electron';
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
  retentionDays: 30,
};

const store = new Store<Stored>({ name: 'autoboth-settings', defaults: DEFAULTS });

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
    retentionDays: s.retentionDays,
    hasSecret: !!s.secretEnc,
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
    remoteRoot: String(rest.remoteRoot || DEFAULTS.remoteRoot).replace(/\/+$/, ''),
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
