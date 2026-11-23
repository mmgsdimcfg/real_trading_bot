// Update log
// - [2026-11-23] type=feat (Guest mode (user 2026-11-23): a PC without a KIS key shows real market data through another AutoBotHtsH that
//   is logged in to KIS. GuestHost - a small HTTP server on the key PC (off by default, the user sets a port and a password): POST
//   /htsh/v1/hello and /htsh/v1/call {method, args} are answered through the same dispatcher as the renderer's calls, so a guest can
//   reach exactly the inquiry methods of the app and nothing else (the app has no order code at all). RemoteMarket - the guest's
//   MarketBackend, every method one call to the host. The password travels in the x-htsh-key header, is compared in constant time,
//   five wrong tries from one address lock it out for a minute; bodies are limited to 64 KB. Plain HTTP: meant for the home network
//   or behind the user's own HTTPS tunnel. No realtime feed for guests - their windows poll)

import http from 'node:http';
import { createHash, timingSafeEqual } from 'node:crypto';
import { networkInterfaces } from 'node:os';
import type { ChartReq, ChartRes, DailyRow, IndexQuote, InfoField, InvestorDay, InvestorRow, MarketBackend, NewsRow, RankReq, Res, TicksReq, TicksRes } from '../shared/market';
import type { OrderBook, Quote, RankRow, TradeTick, WatchRow } from '../shared/kis';

export const GUEST_PORT_DEFAULT = 47800;
export const GUEST_PATH = '/htsh/v1';
export const GUEST_PASS_MIN = 8;
const BODY_MAX = 64 * 1024;
const FAIL_MAX = 5;
const FAIL_WINDOW_MS = 60000;
const CLIENT_SEEN_MS = 90000;
const CALL_TIMEOUT_MS = 25000;

const sha = (s: string) => createHash('sha256').update(s, 'utf8').digest();
/** constant-time comparison (of the hashes, so the lengths never differ) */
export const samePass = (a: string, b: string) => timingSafeEqual(sha(a), sha(b));

/**
 * What the user typed as the host -> base URL without a trailing slash, or null when it is no address.
 * "192.168.0.10" -> http://192.168.0.10:47800, "pc:5000" -> http://pc:5000, "https://x.example.com/" -> https://x.example.com
 * (http without a port gets the default port; https keeps its own)
 */
export function normalizeHostUrl(input: string): string | null {
  let s = String(input || '').trim();
  if (!s) return null;
  const typedScheme = /^https?:\/\//i.test(s);
  if (!typedScheme && s.includes('://')) return null; // some other scheme
  if (!typedScheme) s = `http://${s}`;
  let u: URL;
  try {
    u = new URL(s);
  } catch {
    return null;
  }
  if ((u.protocol !== 'http:' && u.protocol !== 'https:') || !u.hostname || u.username || u.password) return null;
  if (u.protocol === 'http:' && !u.port) u.port = String(GUEST_PORT_DEFAULT);
  return `${u.protocol}//${u.host}${u.pathname.replace(/\/+$/, '')}`;
}

/** IPv4 addresses of this PC other PCs can reach (shown on the host panel) */
export function lanAddresses(): string[] {
  const out: string[] = [];
  for (const list of Object.values(networkInterfaces())) {
    for (const a of list || []) if (a.family === 'IPv4' && !a.internal) out.push(a.address);
  }
  return out;
}

export interface HostInfo {
  app: 'AutoBotHtsH';
  version: string;
  /** the host's own login: guests get data only while it is 'kis' */
  mode: string | null;
  env: string;
}

export interface GuestHostStatus {
  on: boolean;
  port: number;
  /** guest addresses seen in the last 90 s */
  clients: number;
  calls: number;
  error: string;
}

export interface GuestHostDeps {
  call: (method: string, args: unknown[]) => Promise<Res<unknown>>;
  info: () => HostInfo;
  log: (line: string) => void;
  now?: () => number;
}

export class GuestHost {
  private srv: http.Server | null = null;
  private pass = '';
  private port = 0;
  private error = '';
  private calls = 0;
  private fails = new Map<string, { n: number; at: number }>();
  private seen = new Map<string, number>();
  constructor(private d: GuestHostDeps) {}
  private now = () => (this.d.now ? this.d.now() : Date.now());

  status(): GuestHostStatus {
    const t = this.now();
    for (const [ip, at] of this.seen) if (t - at > CLIENT_SEEN_MS) this.seen.delete(ip);
    return { on: !!this.srv, port: this.port, clients: this.seen.size, calls: this.calls, error: this.error };
  }

  /** the port actually bound (0 asks the OS for a free one - tests) */
  async start(port: number, password: string): Promise<GuestHostStatus> {
    await this.stop();
    this.error = '';
    if (password.length < GUEST_PASS_MIN) {
      this.error = `암호는 ${GUEST_PASS_MIN}자 이상이어야 합니다.`;
      return this.status();
    }
    this.pass = password;
    const srv = http.createServer((req, res) => void this.handle(req, res));
    srv.requestTimeout = 30000;
    try {
      await new Promise<void>((ok, bad) => {
        srv.once('error', bad);
        srv.listen(port, '0.0.0.0', () => (srv.off('error', bad), ok()));
      });
    } catch (e) {
      const code = (e as NodeJS.ErrnoException).code;
      this.error = code === 'EADDRINUSE' ? `포트 ${port} 를 다른 프로그램이 쓰고 있습니다.` : String((e as Error).message || e);
      this.d.log(`[guest] 중계 시작 실패: ${this.error}`);
      return this.status();
    }
    this.srv = srv;
    const a = srv.address();
    this.port = a && typeof a === 'object' ? a.port : port;
    srv.on('error', (e) => this.d.log(`[guest] 서버 오류: ${e.message}`));
    this.d.log(`[guest] 게스트 중계 시작 (포트 ${this.port})`);
    return this.status();
  }

  async stop(): Promise<void> {
    const s = this.srv;
    this.srv = null;
    this.seen.clear();
    this.fails.clear();
    if (!s) return;
    await new Promise<void>((ok) => {
      s.close(() => ok());
      s.closeAllConnections?.();
    });
    this.d.log('[guest] 게스트 중계 중지');
  }

  private locked(ip: string): boolean {
    const f = this.fails.get(ip);
    if (!f) return false;
    if (this.now() - f.at > FAIL_WINDOW_MS) {
      this.fails.delete(ip);
      return false;
    }
    return f.n >= FAIL_MAX;
  }

  private async handle(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const reply = (status: number, body: unknown) => {
      const text = JSON.stringify(body);
      res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'content-length': Buffer.byteLength(text) });
      res.end(text);
    };
    try {
      const ip = req.socket.remoteAddress || '?';
      const path = (req.url || '').split('?')[0];
      if (req.method !== 'POST' || (path !== `${GUEST_PATH}/hello` && path !== `${GUEST_PATH}/call`)) return reply(404, { ok: false, error: 'not found' });
      if (this.locked(ip)) return reply(429, { ok: false, error: '암호를 여러 번 틀려 잠시 막혔습니다. 1분 뒤 다시 시도하세요.' });
      const key = req.headers['x-htsh-key'];
      if (typeof key !== 'string' || !samePass(key, this.pass)) {
        const f = this.fails.get(ip);
        const t = this.now();
        this.fails.set(ip, f && t - f.at <= FAIL_WINDOW_MS ? { n: f.n + 1, at: f.at } : { n: 1, at: t });
        if (this.fails.get(ip)!.n === FAIL_MAX) this.d.log(`[guest] ${ip} 암호 ${FAIL_MAX}회 오류 - 1분 차단`);
        return reply(401, { ok: false, error: '암호가 맞지 않습니다.' });
      }
      this.fails.delete(ip);
      if (!this.seen.has(ip)) this.d.log(`[guest] 게스트 접속 ${ip}`);
      this.seen.set(ip, this.now());
      if (path === `${GUEST_PATH}/hello`) {
        req.resume();
        return reply(200, { ok: true, data: this.d.info() });
      }
      const chunks: Buffer[] = [];
      let size = 0;
      for await (const c of req) {
        size += (c as Buffer).length;
        if (size > BODY_MAX) return reply(413, { ok: false, error: '요청이 너무 큽니다.' });
        chunks.push(c as Buffer);
      }
      let body: { method?: unknown; args?: unknown };
      try {
        body = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
      } catch {
        return reply(400, { ok: false, error: '요청 형식 오류' });
      }
      if (typeof body.method !== 'string' || !Array.isArray(body.args)) return reply(400, { ok: false, error: '요청 형식 오류' });
      this.calls++;
      return reply(200, await this.d.call(body.method, body.args));
    } catch (e) {
      if (!res.headersSent) reply(500, { ok: false, error: String((e as Error)?.message || e).split('\n')[0] });
    }
  }
}

/** the guest's market backend: every method is one call to the host */
export class RemoteMarket implements MarketBackend {
  readonly kind = 'guest' as const;
  constructor(
    readonly base: string,
    private pass: string,
    private fetchFn: typeof fetch = fetch,
  ) {}

  private async post<T>(path: string, body: unknown): Promise<T> {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), CALL_TIMEOUT_MS);
    let r: Response;
    try {
      r = await this.fetchFn(`${this.base}${GUEST_PATH}${path}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-htsh-key': this.pass },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
    } catch (e) {
      throw new Error(ac.signal.aborted ? '호스트가 응답하지 않습니다 (시간 초과).' : `호스트에 연결할 수 없습니다 (${String((e as Error)?.message || e).split('\n')[0]}).`);
    } finally {
      clearTimeout(timer);
    }
    let j: Res<T> | null = null;
    try {
      j = (await r.json()) as Res<T>;
    } catch {
      /* not this app's answer */
    }
    if (!j || typeof j !== 'object' || typeof (j as { ok?: unknown }).ok !== 'boolean') throw new Error(`호스트 응답을 읽을 수 없습니다 (HTTP ${r.status}) - 주소와 포트를 확인하세요.`);
    if (!j.ok) throw Object.assign(new Error(j.error || `호스트 오류 (HTTP ${r.status})`), { code: j.code });
    return j.data;
  }

  hello(): Promise<HostInfo> {
    return this.post<HostInfo>('/hello', {});
  }
  private call<T>(method: string, ...args: unknown[]): Promise<T> {
    return this.post<T>('/call', { method, args });
  }

  quote(code: string): Promise<Quote> {
    return this.call('quote', code);
  }
  orderbook(code: string): Promise<OrderBook> {
    return this.call('orderbook', code);
  }
  trades(code: string): Promise<TradeTick[]> {
    return this.call('trades', code);
  }
  chart(req: ChartReq): Promise<ChartRes> {
    return this.call('chart', req);
  }
  ticks(req: TicksReq): Promise<TicksRes> {
    return this.call('ticks', req);
  }
  daily(code: string): Promise<DailyRow[]> {
    return this.call('daily', code);
  }
  investor(code: string): Promise<InvestorRow[]> {
    return this.call('investor', code);
  }
  investorDaily(code: string): Promise<InvestorDay[]> {
    return this.call('investorDaily', code);
  }
  info(code: string): Promise<InfoField[]> {
    return this.call('info', code);
  }
  multi(codes: string[]): Promise<WatchRow[]> {
    return this.call('multi', codes);
  }
  rank(req: RankReq): Promise<RankRow[]> {
    return this.call('rank', req);
  }
  index(code: string): Promise<IndexQuote> {
    return this.call('index', code);
  }
  news(code: string): Promise<NewsRow[]> {
    return this.call('news', code);
  }
}
