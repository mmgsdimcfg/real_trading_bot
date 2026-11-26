// Update log
// - [2026-11-06] type=feat (Realtime market data: KIS websocket (ws://ops.koreainvestment.com:21000 real / :31000 paper) with the approval key,
//   only H0STCNT0 (KRX trades) and H0STASP0 (KRX order book) - assertWsTr() rejects anything else (no 체결통보, which needs an HTS ID).
//   Subscriptions are reference-counted per (tr, code) so several windows share one, capped at 40 (KIS session limit 41), re-sent after
//   a reconnect (back-off 2 s -> 30 s). PINGPONG frames are echoed. Data frames are parsed with the realtime column order from the
//   generated spec and forwarded as objects. Three connections in a row that drop within 30 s of opening pause reconnecting for 5 minutes
//   (another program on the same app key's realtime session would otherwise be knocked off over and over). DemoFeed emits fake trades / order books once a second for the demo backend)
// - [2026-11-06] type=feat (H0UNCNT0 / H0UNASP0 (KRX + NXT combined trades / order book) are allowed too - main/index.ts uses them for
//   NXT-tradable symbols)
// - [2026-11-06] type=fix (Two subscriptions made before the approval key arrived each opened a websocket (this.ws was still null while
//   connect() awaited the key) - the same app key on two sockets gets one of them dropped, which kept realtime from staying on. connect() is
//   now single-flight (connecting flag) and re-checks closed / ws after the await)

import WebSocket from 'ws';
import { KIS_DOMAINS, assertWsTr, parseWsFrame, recordToObject, wsColumns, type KisEnv, type WsTr } from '../shared/kis';
import type { DemoMarket } from './demoMarket';

export interface RtData {
  tr: WsTr;
  code: string;
  rec: Record<string, string>;
}

export type RtStatus = 'off' | 'connecting' | 'on' | 'error';

export interface Feed {
  subscribe(tr: string, code: string): void;
  unsubscribe(tr: string, code: string): void;
  close(): void;
  readonly status: RtStatus;
}

export const MAX_SUBS = 40;

export class KisFeed implements Feed {
  private ws: WebSocket | null = null;
  /** connect() is awaiting the approval key (no socket yet) */
  private connecting = false;
  private subs = new Map<string, number>();
  private backoff = 2000;
  private timer: NodeJS.Timeout | null = null;
  private closed = false;
  /** when the current connection opened, and how many connections in a row dropped within 30 s of opening */
  private openedAt = 0;
  private quickDrops = 0;
  status: RtStatus = 'off';
  lastError = '';
  /** [2026-11-26] bot first: while set, no socket is opened (subscriptions are kept for the resume) */
  private paused = '';

  constructor(
    private env: KisEnv,
    private approvalKey: () => Promise<string>,
    private onData: (d: RtData) => void,
    private onStatus: (s: RtStatus, msg?: string) => void,
  ) {}

  private setStatus(s: RtStatus, msg?: string) {
    this.status = s;
    if (msg) this.lastError = msg;
    this.onStatus(s, msg);
  }

  /** [2026-11-26] give the realtime session up (another program on this PC needs the app key's one session): the socket is closed,
   *  nothing reconnects, the status is 'off' with `why`. The subscriptions stay and are sent again by resume() */
  pause(why: string): void {
    if (this.closed || this.paused) return;
    this.paused = why;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null; // the close handler sees another socket and does not retry
    try {
      ws?.close();
    } catch {
      /* already gone */
    }
    this.openedAt = 0;
    this.quickDrops = 0;
    this.backoff = 2000;
    this.setStatus('off', why);
  }

  resume(): void {
    if (this.closed || !this.paused) return;
    this.paused = '';
    this.setStatus('off');
    if (this.subs.size) void this.connect();
  }

  get isPaused(): boolean {
    return !!this.paused;
  }

  private async connect(): Promise<void> {
    if (this.closed || this.paused || this.ws || this.connecting) return;
    this.connecting = true;
    this.setStatus('connecting');
    let key: string;
    try {
      key = await this.approvalKey();
    } catch (e) {
      this.connecting = false;
      this.setStatus('error', `실시간 접속키: ${(e as Error).message}`);
      this.retry();
      return;
    }
    this.connecting = false;
    if (this.closed || this.paused || this.ws) return;
    const ws = new WebSocket(`${KIS_DOMAINS[this.env].ws}/tryitout/H0STCNT0`);
    this.ws = ws;
    ws.on('open', () => {
      this.openedAt = Date.now();
      this.setStatus('on');
      for (const k of this.subs.keys()) this.send(key, '1', k);
    });
    ws.on('message', (data) => {
      const raw = data.toString();
      if (raw[0] === '{') {
        try {
          const j = JSON.parse(raw);
          if (j?.header?.tr_id === 'PINGPONG') ws.send(raw);
          else if (j?.body?.rt_cd && j.body.rt_cd !== '0') this.setStatus('on', `실시간 구독: ${j.body.msg1 || j.body.msg_cd}`);
        } catch {
          /* ignore */
        }
        return;
      }
      const f = parseWsFrame(raw, (tr) => wsColumns(tr).length);
      if (!f || f.encrypted) return;
      for (const rec of f.records) {
        const o = recordToObject(f.trId, rec);
        const code = o.MKSC_SHRN_ISCD || rec[0];
        this.onData({ tr: f.trId as WsTr, code, rec: o });
      }
    });
    const drop = (why: string) => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.closed) return;
      // a session that keeps dropping right after it opens usually means another program uses the same app key's realtime
      // session (KIS ends one of them) - stop for 5 minutes instead of knocking the other program off again and again
      const quick = this.openedAt && Date.now() - this.openedAt < 30000;
      this.quickDrops = quick ? this.quickDrops + 1 : 0;
      if (!quick && this.openedAt) this.backoff = 2000;
      this.openedAt = 0;
      if (this.quickDrops >= 3) {
        this.quickDrops = 0;
        this.setStatus('error', `${why} - 연결 직후 끊김이 반복되어 5분간 실시간을 멈춥니다 (같은 앱키로 실시간을 쓰는 다른 프로그램이 있는지 확인). 그동안 주기 조회로 표시합니다.`);
        this.backoff = 300000;
      } else this.setStatus('error', why);
      this.retry();
    };
    ws.on('close', (code, reason) => drop(`실시간 연결 끊김 (${code}${reason?.length ? ` ${reason.toString()}` : ''})`));
    ws.on('error', (e) => drop(`실시간 오류: ${e.message}`));
  }

  private retry(): void {
    if (this.closed || this.paused || this.timer) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.connect();
    }, this.backoff);
    this.backoff = this.backoff >= 300000 ? 2000 : Math.min(30000, this.backoff * 2);
  }

  private send(key: string, type: '1' | '2', subKey: string): void {
    const [tr, code] = subKey.split('|');
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(
      JSON.stringify({
        header: { approval_key: key, custtype: 'P', tr_type: type, 'content-type': 'utf-8' },
        body: { input: { tr_id: tr, tr_key: code } },
      }),
    );
  }

  subscribe(tr: string, code: string): void {
    assertWsTr(tr);
    const k = `${tr}|${code}`;
    const n = this.subs.get(k) || 0;
    if (!n && this.subs.size >= MAX_SUBS) {
      this.onStatus(this.status, `실시간 구독은 ${MAX_SUBS}개까지입니다 (${code} 제외)`);
      return;
    }
    this.subs.set(k, n + 1);
    if (n) return;
    if (!this.ws) void this.connect();
    else void this.approvalKey().then((key) => this.send(key, '1', k));
  }

  unsubscribe(tr: string, code: string): void {
    const k = `${tr}|${code}`;
    const n = this.subs.get(k) || 0;
    if (n > 1) return void this.subs.set(k, n - 1);
    if (!n) return;
    this.subs.delete(k);
    void this.approvalKey().then((key) => this.send(key, '2', k));
  }

  close(): void {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
    this.subs.clear();
    this.setStatus('off');
  }
}

/** fake realtime for the demo backend */
export class DemoFeed implements Feed {
  private subs = new Map<string, number>();
  private timer: NodeJS.Timeout;
  status: RtStatus = 'on';

  constructor(
    private market: DemoMarket,
    private onData: (d: RtData) => void,
    onStatus: (s: RtStatus, msg?: string) => void,
  ) {
    onStatus('on');
    this.timer = setInterval(() => void this.emit(), 1000);
  }

  private async emit(): Promise<void> {
    const codes = new Set([...this.subs.keys()].map((k) => k.split('|')[1]));
    for (const code of codes) {
      if (this.subs.has(`H0STCNT0|${code}`)) {
        const t = this.market.tick(code);
        const q = await this.market.quote(code);
        this.onData({
          tr: 'H0STCNT0',
          code,
          rec: {
            MKSC_SHRN_ISCD: code,
            STCK_CNTG_HOUR: t.time,
            STCK_PRPR: String(t.price),
            PRDY_VRSS_SIGN: q.sign,
            PRDY_VRSS: String(Math.abs(q.change)),
            PRDY_CTRT: q.changePct.toFixed(2),
            STCK_OPRC: String(q.open),
            STCK_HGPR: String(q.high),
            STCK_LWPR: String(q.low),
            CNTG_VOL: String(t.volume),
            ACML_VOL: String((q.volume || 0) + t.volume),
            CCLD_DVSN: t.side === 1 ? '1' : '5',
            CTTR: (t.strength || 100).toFixed(2),
          },
        });
      }
      if (this.subs.has(`H0STASP0|${code}`)) {
        const ob = await this.market.orderbook(code);
        const rec: Record<string, string> = { MKSC_SHRN_ISCD: code, BSOP_HOUR: ob.time || '', TOTAL_ASKP_RSQN: String(ob.totalAsk), TOTAL_BIDP_RSQN: String(ob.totalBid) };
        ob.asks.forEach(([p, q], i) => ((rec[`ASKP${i + 1}`] = String(p)), (rec[`ASKP_RSQN${i + 1}`] = String(q))));
        ob.bids.forEach(([p, q], i) => ((rec[`BIDP${i + 1}`] = String(p)), (rec[`BIDP_RSQN${i + 1}`] = String(q))));
        this.onData({ tr: 'H0STASP0', code, rec });
      }
    }
  }

  subscribe(tr: string, code: string): void {
    assertWsTr(tr);
    const k = `${tr}|${code}`;
    this.subs.set(k, (this.subs.get(k) || 0) + 1);
  }

  unsubscribe(tr: string, code: string): void {
    const k = `${tr}|${code}`;
    const n = this.subs.get(k) || 0;
    if (n > 1) this.subs.set(k, n - 1);
    else this.subs.delete(k);
  }

  close(): void {
    clearInterval(this.timer);
    this.subs.clear();
  }
}
