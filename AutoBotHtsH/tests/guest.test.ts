/*
 * Update log
 * - [2026-11-23] type=test - Guest mode (main/guest.ts) over a real local HTTP server: hello and calls with the right password, a
 *   wrong password is refused and locks the address out after five tries, only POST /htsh/v1/hello|call exist, oversized and
 *   malformed bodies are refused, a host error reaches the guest as an Error with its code, a short password does not start the
 *   relay, an unreachable host gives a clear message; normalizeHostUrl
 */

import { afterEach, describe, expect, it } from 'vitest';
import { callerOf, GUEST_PATH, GUEST_PORT_DEFAULT, GuestHost, normalizeHostUrl, RemoteMarket, samePass } from '../src/main/guest';
import type { Res } from '../src/shared/market';

const PASS = 'correct-horse-9';
let host: GuestHost | null = null;
let now = 1_000_000;
const calls: { method: string; args: unknown[] }[] = [];
const logs: string[] = [];

async function start(pass = PASS) {
  calls.length = 0;
  logs.length = 0;
  host = new GuestHost({
    call: async (method, args): Promise<Res<unknown>> => {
      calls.push({ method, args });
      if (method === 'quote') return { ok: true, data: { code: args[0], price: 71200 } };
      if (method === 'multi') return { ok: true, data: (args[0] as string[]).map((code) => ({ code })) };
      return { ok: false, error: `허용되지 않은 호출: ${method}`, code: 'NOPE' };
    },
    info: () => ({ app: 'AutoBotHtsH', version: '1.0.0_001', mode: 'kis', env: 'real' }),
    log: (l) => logs.push(l),
    now: () => now,
  });
  const st = await host.start(0, pass);
  return { st, base: `http://127.0.0.1:${st.port}` };
}
afterEach(async () => {
  await host?.stop();
  host = null;
});

describe('guest mode', () => {
  it('hello and calls with the right password', async () => {
    const { st, base } = await start();
    expect(st).toMatchObject({ on: true, error: '' });
    const g = new RemoteMarket(base, PASS);
    expect(await g.hello()).toEqual({ app: 'AutoBotHtsH', version: '1.0.0_001', mode: 'kis', env: 'real' });
    expect(await g.quote('005930')).toEqual({ code: '005930', price: 71200 });
    expect(await g.multi(['005930', '000660'])).toHaveLength(2);
    expect(calls).toEqual([{ method: 'quote', args: ['005930'] }, { method: 'multi', args: [['005930', '000660']] }]);
    expect(host!.status()).toMatchObject({ clients: 1, calls: 2 });
    expect(g.kind).toBe('guest');
  });

  it('a host error reaches the guest with its message and code', async () => {
    const { base } = await start();
    const e = await new RemoteMarket(base, PASS).news('005930').catch((x) => x as Error & { code?: string });
    expect((e as Error).message).toContain('허용되지 않은 호출');
    expect((e as { code?: string }).code).toBe('NOPE');
  });

  it('a wrong password is refused, five tries lock the address out for a minute, the host is never called', async () => {
    const { base } = await start();
    const bad = new RemoteMarket(base, 'wrong-password');
    for (let k = 0; k < 5; k++) await expect(bad.quote('005930')).rejects.toThrow('암호가 맞지 않습니다');
    await expect(bad.quote('005930')).rejects.toThrow('잠시 막혔습니다');
    await expect(new RemoteMarket(base, PASS).quote('005930')).rejects.toThrow('잠시 막혔습니다'); // the address, not the password
    expect(calls).toEqual([]);
    expect(logs.some((l) => l.includes('5회 오류'))).toBe(true);
    now += 61000;
    expect(await new RemoteMarket(base, PASS).quote('005930')).toMatchObject({ price: 71200 });
  });

  it('only POST hello / call exist; oversized and malformed bodies are refused', async () => {
    const { base } = await start();
    const h = { 'x-htsh-key': PASS, 'content-type': 'application/json' };
    expect((await fetch(`${base}${GUEST_PATH}/call`, { headers: h })).status).toBe(404); // GET
    expect((await fetch(`${base}/`, { method: 'POST', headers: h, body: '{}' })).status).toBe(404);
    expect((await fetch(`${base}${GUEST_PATH}/call`, { method: 'POST', headers: h, body: 'not json' })).status).toBe(400);
    expect((await fetch(`${base}${GUEST_PATH}/call`, { method: 'POST', headers: h, body: JSON.stringify({ method: 'quote' }) })).status).toBe(400);
    expect((await fetch(`${base}${GUEST_PATH}/call`, { method: 'POST', headers: h, body: JSON.stringify({ method: 'quote', args: ['x'.repeat(70000)] }) })).status).toBe(413);
    expect((await fetch(`${base}${GUEST_PATH}/call`, { method: 'POST', body: '{}' })).status).toBe(401); // no key
    expect(calls).toEqual([]);
  });

  it('a short password does not start the relay; stop closes it; an unreachable host gives a clear message', async () => {
    const { st } = await start('short');
    expect(st.on).toBe(false);
    expect(st.error).toContain('8자');
    const ok = await start();
    await host!.stop();
    expect(host!.status().on).toBe(false);
    await expect(new RemoteMarket(ok.base, PASS).hello()).rejects.toThrow('호스트에 연결할 수 없습니다');
  });

  it('[2026-11-23] behind a local tunnel the lockout and the guest count go by the forwarded address', async () => {
    expect(callerOf('192.168.0.7', { 'cf-connecting-ip': '1.2.3.4', 'x-forwarded-for': '9.9.9.9' })).toBe('192.168.0.7'); // direct: headers not trusted
    expect(callerOf('127.0.0.1', { 'cf-connecting-ip': '1.2.3.4', 'x-forwarded-for': '9.9.9.9' })).toBe('1.2.3.4');
    expect(callerOf('::1', { 'x-forwarded-for': '6.6.6.6, 5.5.5.5' })).toBe('5.5.5.5'); // the entry the nearest proxy appended
    expect(callerOf('127.0.0.1', {})).toBe('127.0.0.1');
    const { base } = await start();
    const post = (ip: string, key: string) => fetch(`${base}${GUEST_PATH}/call`, { method: 'POST', headers: { 'x-htsh-key': key, 'cf-connecting-ip': ip, 'content-type': 'application/json' }, body: JSON.stringify({ method: 'quote', args: ['005930'] }) });
    for (let k = 0; k < 5; k++) expect((await post('1.1.1.1', 'wrong-password')).status).toBe(401);
    expect((await post('1.1.1.1', PASS)).status).toBe(429); // the stranger is locked out
    expect((await post('2.2.2.2', PASS)).status).toBe(200); // the real guest is not
    expect((await post('3.3.3.3', PASS)).status).toBe(200);
    expect(host!.status().clients).toBe(2);
  });

  it('normalizeHostUrl and samePass', () => {
    expect(normalizeHostUrl('192.168.0.10')).toBe(`http://192.168.0.10:${GUEST_PORT_DEFAULT}`);
    expect(normalizeHostUrl(' 192.168.0.10:5000/ ')).toBe('http://192.168.0.10:5000');
    expect(normalizeHostUrl('mypc')).toBe(`http://mypc:${GUEST_PORT_DEFAULT}`);
    expect(normalizeHostUrl('https://hts.example.com/')).toBe('https://hts.example.com');
    expect(normalizeHostUrl('https://hts.example.com:8443/relay/')).toBe('https://hts.example.com:8443/relay');
    expect([normalizeHostUrl(''), normalizeHostUrl('ftp://x'), normalizeHostUrl('http://user:pw@x'), normalizeHostUrl('http://')]).toEqual([null, null, null, null]);
    expect([samePass('abc', 'abc'), samePass('abc', 'abd'), samePass('abc', 'abcd')]).toEqual([true, false, false]);
  });
});
