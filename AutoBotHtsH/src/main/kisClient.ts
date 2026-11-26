// Update log
// - [2026-11-06] type=feat (KIS REST client: access token from /oauth2/tokenP reused until 10 minutes before it expires (cached encrypted -
//   KIS allows one issue per minute and a token lasts 24 h), approval key for the websocket from /oauth2/Approval, every GET goes through
//   restEndpoint()/assertInquiry() (inquiry-only allowlist) and a client-side rate limit (real 8/s, paper 2/s - the app key may be shared
//   with other programs), EGW00201 (too many calls) waits and retries once, an expired token is reissued once. POST is used only for the
//   two oauth endpoints; there is no code path that posts to /uapi)
// - [2026-11-26] type=feat (Order window (user 2026-11-26): trade() - the account calls of shared/order.ts (its own allowlist of six
//   trading TRs; get() and the market-data allowlist are unchanged). An order POST is sent once: it is repeated only when the
//   gateway itself refused it before it reached the order system (rate limit EGW00201, expired token) - never after a network
//   error or a timeout, where the order may have been accepted)

import { createHash } from 'node:crypto';
import { KIS_DOMAINS, RATE_PER_SEC, restEndpoint, type KisEnv, type RestApi } from '../shared/kis';
import type { Creds, TokenCache } from './settings';
import { tradeEndpoint, type TradeApi } from '../shared/order';

export class KisError extends Error {
  constructor(
    message: string,
    readonly code?: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

export interface KisResponse {
  output?: unknown;
  output1?: unknown;
  output2?: unknown;
  /** tr_cont header of the response: 'F'/'M' = more pages, 'D'/'E' = last */
  trCont: string;
  msg?: string;
  /** [2026-11-26] continuation keys of a paged account inquiry (ctx_area_fk100 / ctx_area_nk100) */
  ctx?: { fk: string; nk: string };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
export const keyHash = (k: string) => createHash('sha256').update(k).digest('hex').slice(0, 12);

export class KisClient {
  private token: TokenCache | null;
  private tokenPromise: Promise<string> | null = null;
  private approval: { key: string; at: number } | null = null;
  private nextSlot = 0;
  calls = 0;

  constructor(
    readonly creds: Creds,
    private tokens: { load(): TokenCache | null; save(t: TokenCache | null): void },
    private log: (line: string) => void = () => undefined,
  ) {
    const t = tokens.load();
    this.token = t && t.env === creds.env && t.keyHash === keyHash(creds.appKey) ? t : null;
  }

  get env(): KisEnv {
    return this.creds.env;
  }

  private get base(): string {
    return KIS_DOMAINS[this.creds.env].rest;
  }

  /** client-side rate limit: one slot every 1000/rate ms */
  private async slot(): Promise<void> {
    const gap = 1000 / RATE_PER_SEC[this.creds.env];
    const now = Date.now();
    const at = Math.max(now, this.nextSlot);
    this.nextSlot = at + gap;
    if (at > now) await sleep(at - now);
  }

  private async postJson(path: string, body: unknown): Promise<Record<string, unknown>> {
    const res = await fetch(this.base + path, {
      method: 'POST',
      headers: { 'content-type': 'application/json; charset=utf-8' },
      body: JSON.stringify(body),
    });
    const j = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    if (!res.ok || j.error_code || j.error_description) {
      throw new KisError(String(j.error_description || j.msg1 || `HTTP ${res.status}`), String(j.error_code || j.msg_cd || ''), res.status);
    }
    return j;
  }

  /** valid access token (cached; issued at most once per call site) */
  async accessToken(force = false): Promise<string> {
    if (!force && this.token && this.token.expiresAt - Date.now() > 10 * 60000) return this.token.token;
    if (this.tokenPromise) return this.tokenPromise;
    this.tokenPromise = (async () => {
      const j = await this.postJson('/oauth2/tokenP', { grant_type: 'client_credentials', appkey: this.creds.appKey, appsecret: this.creds.appSecret });
      const token = String(j.access_token || '');
      if (!token) throw new KisError('접근 토큰을 받지 못했습니다.');
      // "2026-11-07 10:12:13" (KST) or expires_in seconds
      const expTxt = String(j.access_token_token_expired || '');
      const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(expTxt);
      const expiresAt = m
        ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) - 9 * 3600000
        : Date.now() + (Number(j.expires_in) || 86400) * 1000;
      this.token = { token, expiresAt, env: this.creds.env, keyHash: keyHash(this.creds.appKey) };
      this.tokens.save(this.token);
      this.log(`[auth] 접근 토큰 발급 (만료 ${expTxt || new Date(expiresAt).toISOString()})`);
      return token;
    })().finally(() => (this.tokenPromise = null));
    return this.tokenPromise;
  }

  tokenExpiresAt(): number | undefined {
    return this.token?.expiresAt;
  }

  /** websocket approval key (valid 24 h; one per session is enough) */
  async approvalKey(): Promise<string> {
    if (this.approval && Date.now() - this.approval.at < 20 * 3600000) return this.approval.key;
    const j = await this.postJson('/oauth2/Approval', { grant_type: 'client_credentials', appkey: this.creds.appKey, secretkey: this.creds.appSecret });
    const key = String(j.approval_key || '');
    if (!key) throw new KisError('실시간 접속키를 받지 못했습니다.');
    this.approval = { key, at: Date.now() };
    return key;
  }

  /** inquiry GET (allowlisted API only) */
  async get(api: RestApi, params: Record<string, string>, trCont = '', retry = true): Promise<KisResponse> {
    const ep = restEndpoint(api);
    const token = await this.accessToken();
    await this.slot();
    this.calls++;
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${this.base}${ep.path}?${qs}`, {
      method: 'GET',
      headers: {
        'content-type': 'application/json; charset=utf-8',
        authorization: `Bearer ${token}`,
        appkey: this.creds.appKey,
        appsecret: this.creds.appSecret,
        tr_id: ep.trId,
        tr_cont: trCont,
        custtype: 'P',
      },
    });
    const j = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    const code = String(j.msg_cd || '');
    if (j.rt_cd !== '0') {
      if (retry && code === 'EGW00201') {
        // too many calls per second (shared app key) - wait and try once more
        await sleep(1100);
        return this.get(api, params, trCont, false);
      }
      if (retry && (code === 'EGW00123' || code === 'EGW00121' || res.status === 401)) {
        await this.accessToken(true);
        return this.get(api, params, trCont, false);
      }
      throw new KisError(`${String(j.msg1 || `HTTP ${res.status}`).trim()}${code ? ` (${code})` : ''}`, code, res.status);
    }
    return { output: j.output, output1: j.output1, output2: j.output2, trCont: res.headers.get('tr_cont') || '', msg: String(j.msg1 || '') };
  }

  /**
   * [2026-11-26] one account call of the order window (shared/order.ts allowlist). GET: params as the query; POST: params as the JSON
   * body. A refused call throws KisError with KIS's message; see the update log for when a call is repeated
   */
  async trade(api: TradeApi, params: Record<string, string>, trCont = '', retry = true): Promise<KisResponse> {
    const ep = tradeEndpoint(api, this.creds.env);
    const token = await this.accessToken();
    await this.slot();
    this.calls++;
    const headers = {
      'content-type': 'application/json; charset=utf-8',
      authorization: `Bearer ${token}`,
      appkey: this.creds.appKey,
      appsecret: this.creds.appSecret,
      tr_id: ep.trId,
      tr_cont: trCont,
      custtype: 'P',
    };
    const res =
      ep.method === 'GET'
        ? await fetch(`${this.base}${ep.path}?${new URLSearchParams(params).toString()}`, { method: 'GET', headers })
        : await fetch(`${this.base}${ep.path}`, { method: 'POST', headers, body: JSON.stringify(params) });
    const j = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    const code = String(j.msg_cd || '');
    if (j.rt_cd !== '0') {
      // both are refusals of the gateway: the call never reached the order system, so sending it again cannot double an order
      if (retry && code === 'EGW00201') {
        await sleep(1100);
        return this.trade(api, params, trCont, false);
      }
      if (retry && (code === 'EGW00123' || code === 'EGW00121')) {
        await this.accessToken(true);
        return this.trade(api, params, trCont, false);
      }
      throw new KisError(`${String(j.msg1 || `HTTP ${res.status}`).trim()}${code ? ` (${code})` : ''}`, code, res.status);
    }
    return {
      output: j.output,
      output1: j.output1,
      output2: j.output2,
      trCont: res.headers.get('tr_cont') || '',
      msg: String(j.msg1 || ''),
      ctx: { fk: String(j.ctx_area_fk100 || ''), nk: String(j.ctx_area_nk100 || '') },
    };
  }
}
