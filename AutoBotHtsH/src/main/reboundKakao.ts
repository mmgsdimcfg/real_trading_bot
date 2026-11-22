// Update log
// - [2026-11-22] type=feat (KakaoTalk "나에게 보내기" for rebound alerts (user 2026-11-22). The user's own Kakao Developers app: REST API
//   key (+ client secret when the app has one switched on) and a redirect URI registered in the console. login(): a small window
//   opens Kakao's consent page (scope talk_message); the redirect to the registered URI is caught before it loads (no local server),
//   its code is exchanged for tokens. send(): the default text template through /v2/api/talk/memo/default/send; the access token
//   (6 h) is renewed with the refresh token (about 2 months, renewed by Kakao when it gets short) - a failed renewal asks for a new
//   login. Key, secret and tokens are kept only encrypted with safeStorage (DPAPI, this Windows user) in rebound\kakao.json and
//   never go to the renderer. Kakao error codes are turned into a hint about the console setting that is usually wrong)
// - [2026-11-22] type=feat (exportForBot(): a second Kakao login whose tokens are written, with the app key, to ~/.kakao/
//   kakao_config.json + kakao_token.json in the format of AutoBot g007_kakao_notify.py - the scan job (g002 --notify-email) then
//   sends the picks to KakaoTalk from Python, also while this app is closed. Its own token pair, so the app's login and the bot's
//   never renew each other's refresh token. These two files are plain text (g007's format): anyone who can read the user's home
//   folder could send KakaoTalk messages to the user with them)

import { BrowserWindow, safeStorage, session } from 'electron';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import { reboundDir } from './reboundAlert';

const AUTH_URL = 'https://kauth.kakao.com/oauth/authorize';
const TOKEN_URL = 'https://kauth.kakao.com/oauth/token';
const SEND_URL = 'https://kapi.kakao.com/v2/api/talk/memo/default/send';
export const KAKAO_DEFAULT_REDIRECT = 'http://localhost:5000/oauth';
/** the default text template carries at most 200 characters */
export const KAKAO_TEXT_MAX = 200;

interface Secret {
  restKey: string;
  clientSecret: string;
  accessToken: string;
  accessExpiresAt: number;
  refreshToken: string;
  refreshExpiresAt: number;
}
interface Stored {
  redirectUri: string;
  keyHint: string;
  /** encrypted JSON of Secret */
  enc: string;
}

export interface KakaoState {
  /** a REST API key is saved */
  configured: boolean;
  keyHint: string;
  hasSecret: boolean;
  redirectUri: string;
  /** tokens are held and the refresh token has not run out */
  loggedIn: boolean;
  accessExpiresAt: number;
  refreshExpiresAt: number;
  lastError: string;
  lastSentAt: number;
  /** ~/.kakao holds the login files of the bot (g007): the scan job can send its picks */
  botFiles: boolean;
  botDir: string;
}

const HINTS: Record<string, string> = {
  KOE004: '카카오 로그인이 꺼져 있습니다. 콘솔 > 카카오 로그인 > 사용 설정을 켜세요.',
  KOE006: '리다이렉트 URI가 콘솔에 등록된 값과 다릅니다. 콘솔에 등록한 주소와 이 창의 주소가 글자 하나까지 같아야 합니다.',
  KOE010: '클라이언트 시크릿이 맞지 않습니다. 콘솔에서 시크릿이 켜져 있으면 그 값을 입력하고, 꺼져 있으면 비워 두세요.',
  KOE101: 'REST API 키가 잘못됐습니다. 네이티브 / JavaScript / 어드민 키가 아니라 REST API 키를 넣어야 합니다.',
  KOE205: '카카오톡 메시지 전송(talk_message) 동의항목이 설정되지 않았습니다. 콘솔 > 카카오 로그인 > 동의항목에서 설정하세요.',
  KOE303: '리다이렉트 URI가 로그인 때와 다릅니다. 주소를 바꿨다면 다시 로그인하세요.',
  KOE320: '인가 코드가 만료됐거나 이미 쓰였습니다. 다시 로그인하세요.',
  KOE322: '로그인 정보(리프레시 토큰)가 만료됐습니다. 다시 로그인하세요.',
  '-401': '토큰이 유효하지 않습니다. 다시 로그인하세요.',
  '-402': '카카오톡 메시지 전송에 동의하지 않았습니다. 다시 로그인해서 동의 화면의 "카카오톡 메시지 전송"을 체크하세요.',
  '-10': '호출 한도를 넘었습니다. 잠시 뒤 다시 시도하세요.',
};

/** Kakao error body -> one line with the code and what to fix */
export function kakaoErrorText(status: number, body: unknown): string {
  const b = (body && typeof body === 'object' ? body : {}) as Record<string, unknown>;
  const code = String(b.error_code ?? b.code ?? '');
  const msg = String(b.error_description ?? b.msg ?? b.error ?? '');
  const hint = HINTS[code] || '';
  return [`카카오 오류 ${code || status}`, msg, hint].filter(Boolean).join(' - ');
}

/** folder of the bot's Kakao files (AutoBot g007_kakao_notify.py: Path.home() / ".kakao") */
export const kakaoBotDir = () => join(homedir(), '.kakao');

const mask = (k: string) => (k.length > 8 ? `${k.slice(0, 4)}…${k.slice(-4)}` : k ? '…' : '');

export class KakaoTalk {
  private lastError = '';
  private lastSentAt = 0;
  private refreshing: Promise<void> | null = null;

  constructor(private log: (line: string) => void) {}

  private file(): string {
    return join(reboundDir(), 'kakao.json');
  }

  private read(): { stored: Stored; secret: Secret } {
    const empty: Secret = { restKey: '', clientSecret: '', accessToken: '', accessExpiresAt: 0, refreshToken: '', refreshExpiresAt: 0 };
    let stored: Stored = { redirectUri: KAKAO_DEFAULT_REDIRECT, keyHint: '', enc: '' };
    try {
      const v = JSON.parse(readFileSync(this.file(), 'utf8')) as Partial<Stored>;
      stored = { redirectUri: typeof v.redirectUri === 'string' && v.redirectUri ? v.redirectUri : KAKAO_DEFAULT_REDIRECT, keyHint: String(v.keyHint || ''), enc: String(v.enc || '') };
    } catch {
      /* not set up yet */
    }
    let secret = empty;
    if (stored.enc) {
      try {
        secret = { ...empty, ...(JSON.parse(safeStorage.decryptString(Buffer.from(stored.enc, 'base64'))) as Partial<Secret>) };
      } catch {
        /* another Windows user / damaged: set up again */
      }
    }
    return { stored, secret };
  }

  private write(stored: Stored, secret: Secret): void {
    if (!safeStorage.isEncryptionAvailable()) throw new Error('OS 암호화(safeStorage)를 사용할 수 없어 저장하지 않았습니다.');
    if (!existsSync(reboundDir())) mkdirSync(reboundDir(), { recursive: true });
    const out: Stored = { redirectUri: stored.redirectUri, keyHint: mask(secret.restKey), enc: safeStorage.encryptString(JSON.stringify(secret)).toString('base64') };
    writeFileSync(this.file(), JSON.stringify(out, null, 1), 'utf8');
  }

  state(): KakaoState {
    const { stored, secret } = this.read();
    return {
      configured: !!secret.restKey,
      keyHint: secret.restKey ? mask(secret.restKey) : '',
      hasSecret: !!secret.clientSecret,
      redirectUri: stored.redirectUri,
      loggedIn: !!secret.refreshToken && (!secret.refreshExpiresAt || secret.refreshExpiresAt > Date.now()),
      accessExpiresAt: secret.accessExpiresAt,
      refreshExpiresAt: secret.refreshExpiresAt,
      lastError: this.lastError,
      lastSentAt: this.lastSentAt,
      botFiles: existsSync(join(kakaoBotDir(), 'kakao_config.json')) && existsSync(join(kakaoBotDir(), 'kakao_token.json')),
      botDir: kakaoBotDir(),
    };
  }

  /** app settings from the dialog. restKey '' keeps the saved key; clientSecret undefined keeps it, '' removes it. A new key drops the tokens */
  saveApp(a: { restKey?: string; clientSecret?: string; redirectUri?: string }): KakaoState {
    const { stored, secret } = this.read();
    const key = (a.restKey || '').trim();
    if (key && key !== secret.restKey) Object.assign(secret, { restKey: key, accessToken: '', accessExpiresAt: 0, refreshToken: '', refreshExpiresAt: 0 });
    if (a.clientSecret !== undefined) secret.clientSecret = a.clientSecret.trim();
    const uri = (a.redirectUri || '').trim();
    if (uri) stored.redirectUri = uri;
    this.write(stored, secret);
    this.lastError = '';
    return this.state();
  }

  logout(): KakaoState {
    const { stored, secret } = this.read();
    this.write(stored, { ...secret, accessToken: '', accessExpiresAt: 0, refreshToken: '', refreshExpiresAt: 0 });
    this.lastError = '';
    return this.state();
  }

  private async token(params: Record<string, string>): Promise<void> {
    const { stored, secret } = this.read();
    const body = new URLSearchParams({ client_id: secret.restKey, ...params });
    if (secret.clientSecret) body.set('client_secret', secret.clientSecret);
    const res = await fetch(TOKEN_URL, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8' }, body });
    const j = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    if (!res.ok || !j.access_token) throw new Error(kakaoErrorText(res.status, j));
    const now = Date.now();
    secret.accessToken = String(j.access_token);
    secret.accessExpiresAt = now + Number(j.expires_in || 21600) * 1000;
    // a refresh answer carries a new refresh token only when the old one is running short - otherwise the old one stays
    if (j.refresh_token) {
      secret.refreshToken = String(j.refresh_token);
      secret.refreshExpiresAt = now + Number(j.refresh_token_expires_in || 5184000) * 1000;
    }
    this.write(stored, secret);
  }

  /**
   * Kakao login in its own window (a session that is not kept): the user logs in and agrees to "카카오톡 메시지 전송"; Kakao then
   * redirects to the registered URI with ?code=... - caught before it loads, exchanged for tokens.
   */
  async login(parent: BrowserWindow | null): Promise<KakaoState> {
    const code = await this.authorize(parent);
    try {
      await this.token({ grant_type: 'authorization_code', redirect_uri: this.read().stored.redirectUri, code });
      this.lastError = '';
      this.log('[kakao] 로그인 완료');
    } catch (e) {
      this.lastError = (e as Error).message;
      throw e;
    }
    return this.state();
  }

  /**
   * Login files for the bot: one more Kakao login, its tokens and the app key written to ~/.kakao in g007's format (plain JSON).
   * The app's own login is not touched.
   */
  async exportForBot(parent: BrowserWindow | null): Promise<KakaoState> {
    const code = await this.authorize(parent);
    const { stored, secret } = this.read();
    try {
      const body = new URLSearchParams({ client_id: secret.restKey, grant_type: 'authorization_code', redirect_uri: stored.redirectUri, code });
      if (secret.clientSecret) body.set('client_secret', secret.clientSecret);
      const res = await fetch(TOKEN_URL, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8' }, body });
      const j = (await res.json().catch(() => ({}))) as Record<string, unknown>;
      if (!res.ok || !j.access_token || !j.refresh_token) throw new Error(kakaoErrorText(res.status, j));
      const dir = kakaoBotDir();
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, 'kakao_config.json'), JSON.stringify({ rest_api_key: secret.restKey, client_secret: secret.clientSecret, redirect_uri: stored.redirectUri }, null, 2), 'utf8');
      writeFileSync(join(dir, 'kakao_token.json'), JSON.stringify({ ...j, obtained_at: Date.now() / 1000 }, null, 2), 'utf8');
      this.lastError = '';
      this.log(`[kakao] 봇용 로그인 파일 저장 (${dir})`);
    } catch (e) {
      this.lastError = (e as Error).message;
      throw e;
    }
    return this.state();
  }

  /** the Kakao consent window; resolves with the authorization code of the redirect */
  private async authorize(parent: BrowserWindow | null): Promise<string> {
    const { stored, secret } = this.read();
    if (!secret.restKey) throw new Error('REST API 키를 먼저 저장하세요.');
    const redirect = stored.redirectUri;
    const url = `${AUTH_URL}?${new URLSearchParams({ client_id: secret.restKey, redirect_uri: redirect, response_type: 'code', scope: 'talk_message' })}`;
    return new Promise<string>((resolve, reject) => {
      const w = new BrowserWindow({
        width: 520,
        height: 720,
        parent: parent || undefined,
        modal: !!parent,
        title: '카카오 로그인',
        autoHideMenuBar: true,
        webPreferences: { session: session.fromPartition('kakao-login'), sandbox: true, contextIsolation: true, nodeIntegration: false },
      });
      let done = false;
      const finish = (fn: () => void) => {
        if (done) return;
        done = true;
        fn();
        if (!w.isDestroyed()) w.close();
      };
      const check = (e: { preventDefault: () => void }, target: string) => {
        if (!target.startsWith(redirect)) return;
        e.preventDefault();
        const q = new URL(target).searchParams;
        const c = q.get('code');
        if (c) finish(() => resolve(c));
        else finish(() => reject(new Error(kakaoErrorText(0, { error_code: q.get('error') || '', error_description: q.get('error_description') || '동의하지 않았거나 취소했습니다.' }))));
      };
      w.webContents.on('will-redirect', check);
      w.webContents.on('will-navigate', check);
      w.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
      w.on('closed', () => finish(() => reject(new Error('로그인 창을 닫았습니다.'))));
      void w.loadURL(url).catch(() => undefined); // an error page of Kakao (wrong key / URI) stays visible in the window
    }).catch((e) => {
      this.lastError = (e as Error).message;
      throw e;
    });
  }

  private async ensureToken(force = false): Promise<string> {
    let { secret } = this.read();
    if (!secret.restKey) throw new Error('카카오 REST API 키가 저장돼 있지 않습니다.');
    if (!secret.refreshToken) throw new Error('카카오 로그인이 필요합니다.');
    if (force || !secret.accessToken || secret.accessExpiresAt - Date.now() < 5 * 60000) {
      // one renewal at a time: two alerts in the same minute must not both spend the refresh token
      this.refreshing ??= this.token({ grant_type: 'refresh_token', refresh_token: secret.refreshToken }).finally(() => (this.refreshing = null));
      await this.refreshing;
      secret = this.read().secret;
    }
    return secret.accessToken;
  }

  /** text (cut to 200 characters) to the user's own KakaoTalk; link = where the message's button goes (its domain must be registered in the console) */
  async send(text: string, link?: string): Promise<void> {
    const post = async (tok: string) => {
      const web = link || 'https://finance.naver.com';
      const template = { object_type: 'text', text: text.length > KAKAO_TEXT_MAX ? `${text.slice(0, KAKAO_TEXT_MAX - 1)}…` : text, link: { web_url: web, mobile_web_url: web }, button_title: '종목 보기' };
      const res = await fetch(SEND_URL, {
        method: 'POST',
        headers: { Authorization: `Bearer ${tok}`, 'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8' },
        body: new URLSearchParams({ template_object: JSON.stringify(template) }),
      });
      const j = (await res.json().catch(() => ({}))) as Record<string, unknown>;
      return { res, j };
    };
    try {
      let { res, j } = await post(await this.ensureToken());
      if (res.status === 401) ({ res, j } = await post(await this.ensureToken(true))); // the token was revoked early: renew once
      if (!res.ok || Number(j.result_code) !== 0) throw new Error(kakaoErrorText(res.status, j));
      this.lastError = '';
      this.lastSentAt = Date.now();
    } catch (e) {
      this.lastError = (e as Error).message;
      this.log(`[kakao] 전송 실패: ${this.lastError}`);
      throw e;
    }
  }
}
