// Update log
// - [2026-11-06] type=feat (HtsH settings in %APPDATA%/HtsH/htsh-settings.json (electron-store). The KIS app key / app secret and the
//   issued access token are stored only encrypted with safeStorage (DPAPI, tied to the Windows user) and never sent to the renderer.
//   No account number is asked for or stored - HtsH is inquiry-only)

import Store from 'electron-store';
import { safeStorage } from 'electron';
import type { KisEnv } from '../shared/kis';

interface Stored {
  env: KisEnv;
  remember: boolean;
  /** encrypted JSON {appKey, appSecret} */
  credEnc: string;
  /** encrypted JSON {token, expiresAt, env, keyHash} */
  tokenEnc: string;
  /** last app key shown masked on the login screen (e.g. "PSab…x9") */
  keyHint: string;
}

const store = new Store<Stored>({
  name: 'htsh-settings',
  defaults: { env: 'real', remember: true, credEnc: '', tokenEnc: '', keyHint: '' },
});

const enc = (s: string) => {
  if (!safeStorage.isEncryptionAvailable()) throw new Error('OS 암호화(safeStorage)를 사용할 수 없어 저장하지 않았습니다.');
  return safeStorage.encryptString(s).toString('base64');
};
const dec = (s: string): string | null => {
  if (!s) return null;
  try {
    return safeStorage.decryptString(Buffer.from(s, 'base64'));
  } catch {
    return null;
  }
};

export interface Creds {
  env: KisEnv;
  appKey: string;
  appSecret: string;
}

export function loginDefaults(): { env: KisEnv; remember: boolean; keyHint: string; hasSaved: boolean } {
  const s = store.store;
  return { env: s.env, remember: s.remember, keyHint: s.keyHint, hasSaved: !!dec(s.credEnc) };
}

export function savedCreds(): Creds | null {
  const raw = dec(store.get('credEnc'));
  if (!raw) return null;
  try {
    const c = JSON.parse(raw) as { appKey: string; appSecret: string };
    return c.appKey && c.appSecret ? { env: store.get('env'), appKey: c.appKey, appSecret: c.appSecret } : null;
  } catch {
    return null;
  }
}

export const maskKey = (k: string) => (k.length > 8 ? `${k.slice(0, 4)}…${k.slice(-2)}` : '****');

export function rememberCreds(c: Creds, remember: boolean): void {
  store.set('env', c.env);
  store.set('remember', remember);
  store.set('keyHint', maskKey(c.appKey));
  store.set('credEnc', remember ? enc(JSON.stringify({ appKey: c.appKey, appSecret: c.appSecret })) : '');
}

export function forgetCreds(): void {
  store.set('credEnc', '');
  store.set('tokenEnc', '');
  store.set('keyHint', '');
}

export interface TokenCache {
  token: string;
  /** epoch ms */
  expiresAt: number;
  env: KisEnv;
  /** short hash of the app key the token belongs to */
  keyHash: string;
}

export const tokenStore = {
  load(): TokenCache | null {
    const raw = dec(store.get('tokenEnc'));
    if (!raw) return null;
    try {
      return JSON.parse(raw) as TokenCache;
    } catch {
      return null;
    }
  },
  save(t: TokenCache | null): void {
    // tokens are kept even when credentials are not remembered: KIS allows only one issue per minute and a token lasts 24 h
    store.set('tokenEnc', t ? enc(JSON.stringify(t)) : '');
  },
};
