/*
 * Update log
 * - [2026-11-22] type=test - Rebound alert history (main/reboundHistory.ts: kept in a file, unread count, mark read, delete, cap),
 *   the KakaoTalk summary text (shared/rebound.ts alertSummary: at most 200 characters) and the KakaoTalk client
 *   (main/reboundKakao.ts with a stubbed fetch: code -> tokens, send, an expired token is renewed first, a 401 is renewed once and
 *   retried, Kakao error codes become a hint, the key / tokens are not readable in the file)
 */

import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const userData = mkdtempSync(join(tmpdir(), 'rebound-extras-'));
vi.mock('electron', () => ({
  app: { getPath: () => userData },
  BrowserWindow: class {},
  screen: {},
  shell: { beep: () => undefined },
  session: { fromPartition: () => ({}) },
  // stand-in for DPAPI: reversible, and the plain text is not visible in the file
  safeStorage: {
    isEncryptionAvailable: () => true,
    encryptString: (s: string) => Buffer.from(s, 'utf8').map((b) => b ^ 0x5a),
    decryptString: (b: Buffer) => Buffer.from(b.map((x) => x ^ 0x5a)).toString('utf8'),
  },
}));

import { alertSummary, REBOUND_DEFAULTS } from '../src/shared/rebound';
import { sampleAlert } from '../src/main/reboundPoller';
import { HISTORY_MAX, ReboundHistory } from '../src/main/reboundHistory';
import { KAKAO_DEFAULT_REDIRECT, KAKAO_TEXT_MAX, kakaoErrorText, KakaoTalk } from '../src/main/reboundKakao';

afterAll(() => rmSync(userData, { recursive: true, force: true }));
const alertOf = (code: string, at: number) => ({ ...sampleAlert(REBOUND_DEFAULTS), code, name: `종목${code}`, at });

describe('alert summary (KakaoTalk text)', () => {
  it('names the symbol, bar, price and indicators within 200 characters', () => {
    const s = alertSummary(alertOf('448900', 0));
    expect(s.length).toBeLessThanOrEqual(KAKAO_TEXT_MAX);
    expect(s.split('\n')[0]).toBe('[바닥 반등 초입] 종목448900(448900)');
    expect(s).toContain('38,500원 (-2.41%) · 가점 3점');
    expect(s).toContain('RSI 53.3/48.8');
    expect(s).toContain('가점: ADX 반등, 볼린저, 거래량 감소');
  });

  it('drops lines from the end when the name is long, and gives a sample without a result', () => {
    const long = alertSummary({ ...alertOf('448900', 0), name: '가'.repeat(120) });
    expect(long.length).toBeLessThanOrEqual(KAKAO_TEXT_MAX);
    expect(long).toContain('448900');
    expect(alertSummary({ code: '005930', name: '삼성전자', price: 71200, changePct: 1.28, res: null })).toContain('71,200원 (+1.28%)');
  });
});

describe('alert history', () => {
  it('keeps alerts newest first, counts unread, marks read, deletes and survives a restart', () => {
    let changes = 0;
    const h = new ReboundHistory(() => changes++);
    h.remove(null);
    const a = h.add(alertOf('000001', 1000));
    const b = h.add(alertOf('000002', 2000));
    expect(h.list().map((x) => x.code)).toEqual(['000002', '000001']);
    expect(h.unread()).toBe(2);
    h.markRead([a.id]);
    expect(h.unread()).toBe(1);
    h.setKakao(b.id, 'failed', '카카오 오류 -402');
    expect(changes).toBeGreaterThan(0);
    // another instance reads the same file
    const again = new ReboundHistory();
    expect(again.list().map((x) => [x.code, x.read, x.kakao])).toEqual([['000002', false, 'failed'], ['000001', true, '']]);
    expect(again.get(b.id)?.kakaoError).toContain('-402');
    again.markRead(null);
    expect(again.unread()).toBe(0);
    again.remove([a.id]);
    expect(new ReboundHistory().list().map((x) => x.code)).toEqual(['000002']);
    again.remove(null);
    expect(new ReboundHistory().list()).toEqual([]);
  });

  it('keeps at most HISTORY_MAX alerts', () => {
    const h = new ReboundHistory();
    for (let k = 0; k < HISTORY_MAX + 5; k++) h.add(alertOf('000003', k));
    expect(h.list()).toHaveLength(HISTORY_MAX);
    expect(h.list()[0].at).toBe(HISTORY_MAX + 4);
    h.remove(null);
  });
});

describe('KakaoTalk client', () => {
  const calls: { url: string; body: URLSearchParams; auth: string }[] = [];
  let answers: { status: number; json: unknown }[] = [];
  const logs: string[] = [];
  beforeEach(() => {
    calls.length = 0;
    answers = [];
    vi.stubGlobal('fetch', async (url: string, init: { body: URLSearchParams; headers: Record<string, string> }) => {
      calls.push({ url, body: init.body, auth: init.headers.Authorization || '' });
      const a = answers.shift() || { status: 500, json: {} };
      return { ok: a.status >= 200 && a.status < 300, status: a.status, json: async () => a.json };
    });
  });
  const tokens = (access: string, refresh?: string) => ({ status: 200, json: { access_token: access, expires_in: 21600, ...(refresh ? { refresh_token: refresh, refresh_token_expires_in: 5184000 } : {}) } });
  const sentOk = { status: 200, json: { result_code: 0 } };
  const exchange = (k: KakaoTalk) => (k as unknown as { token: (p: Record<string, string>) => Promise<void> }).token({ grant_type: 'authorization_code', redirect_uri: KAKAO_DEFAULT_REDIRECT, code: 'CODE' });

  it('saves the app keys encrypted and reports the state', () => {
    const k = new KakaoTalk((l) => logs.push(l));
    expect(k.state()).toMatchObject({ configured: false, loggedIn: false, redirectUri: KAKAO_DEFAULT_REDIRECT });
    const st = k.saveApp({ restKey: 'abcd1234efgh5678', clientSecret: 'SECRETVALUE', redirectUri: 'http://localhost:5000/oauth' });
    expect(st).toMatchObject({ configured: true, hasSecret: true, keyHint: 'abcd…5678', loggedIn: false });
    const file = readFileSync(join(userData, 'rebound', 'kakao.json'), 'utf8');
    expect(file).not.toContain('abcd1234efgh5678');
    expect(file).not.toContain('SECRETVALUE');
    expect(k.saveApp({ clientSecret: '' }).hasSecret).toBe(false); // '' removes it, the key stays
    expect(k.state().configured).toBe(true);
  });

  it('code -> tokens, then a message with the bearer token and the text template', async () => {
    const k = new KakaoTalk((l) => logs.push(l));
    k.saveApp({ restKey: 'abcd1234efgh5678', clientSecret: 'SEC' });
    answers = [tokens('ACCESS1', 'REFRESH1'), sentOk];
    await exchange(k);
    expect(calls[0].url).toContain('/oauth/token');
    expect(Object.fromEntries(calls[0].body)).toMatchObject({ client_id: 'abcd1234efgh5678', client_secret: 'SEC', grant_type: 'authorization_code', code: 'CODE' });
    expect(k.state().loggedIn).toBe(true);
    await k.send('가'.repeat(300), 'https://m.stock.naver.com/domestic/stock/448900/total');
    expect(calls[1].url).toContain('/v2/api/talk/memo/default/send');
    expect(calls[1].auth).toBe('Bearer ACCESS1');
    const tpl = JSON.parse(calls[1].body.get('template_object') || '{}');
    expect(tpl.object_type).toBe('text');
    expect(tpl.text).toHaveLength(KAKAO_TEXT_MAX);
    expect(tpl.link.mobile_web_url).toContain('448900');
    expect(k.state().lastSentAt).toBeGreaterThan(0);
  });

  it('a 401 is renewed with the refresh token once and sent again; the old refresh token stays when Kakao sends no new one', async () => {
    const k = new KakaoTalk((l) => logs.push(l));
    answers = [{ status: 401, json: { code: -401, msg: 'this access token does not exist' } }, tokens('ACCESS2'), sentOk];
    await k.send('hello');
    expect(calls.map((c) => c.url.replace(/^https:\/\/[^/]+/, ''))).toEqual(['/v2/api/talk/memo/default/send', '/oauth/token', '/v2/api/talk/memo/default/send']);
    expect(Object.fromEntries(calls[1].body)).toMatchObject({ grant_type: 'refresh_token', refresh_token: 'REFRESH1' });
    expect(calls[2].auth).toBe('Bearer ACCESS2');
    // the next send uses the renewed token without another renewal
    answers = [sentOk];
    await k.send('again');
    expect(calls[3].auth).toBe('Bearer ACCESS2');
  });

  it('a refusal becomes an error with the code and a hint, kept as the last error', async () => {
    const k = new KakaoTalk((l) => logs.push(l));
    answers = [{ status: 403, json: { code: -402, msg: 'insufficient scopes.' } }];
    await expect(k.send('x')).rejects.toThrow(/-402.*동의/);
    expect(k.state().lastError).toContain('-402');
    expect(k.logout()).toMatchObject({ loggedIn: false, configured: true, lastError: '' });
    await expect(k.send('x')).rejects.toThrow('카카오 로그인이 필요합니다.');
    // a new key drops the old login
    k.saveApp({ restKey: 'zzzz1111yyyy2222' });
    expect(k.state()).toMatchObject({ keyHint: 'zzzz…2222', loggedIn: false });
  });

  it('state reports the bot login files (~/.kakao of AutoBot g007)', () => {
    const st = new KakaoTalk(() => undefined).state();
    expect(st.botDir.replace(/\\/g, '/')).toMatch(/\/\.kakao$/);
    expect(typeof st.botFiles).toBe('boolean');
  });

  it('kakaoErrorText: token errors (KOE...) and API errors (negative codes)', () => {
    expect(kakaoErrorText(400, { error: 'invalid_client', error_description: 'Bad client credentials', error_code: 'KOE010' })).toMatch(/KOE010.*시크릿/);
    expect(kakaoErrorText(400, { error_code: 'KOE006', error_description: 'redirect uri mismatch' })).toMatch(/리다이렉트 URI/);
    expect(kakaoErrorText(500, {})).toBe('카카오 오류 500');
  });
});
