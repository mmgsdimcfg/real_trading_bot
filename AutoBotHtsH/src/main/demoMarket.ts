// Update log
// - [2026-11-06] type=feat (Demo backend: deterministic synthetic market data (seeded random walks per code) with the same MarketBackend
//   contract as the KIS backend, plus a fake realtime feed - lets the UI be tried and screen-tested without a KIS app key. Every window
//   shows "데모" while it is active; nothing here touches the network)
// - [2026-11-06] type=feat (ticks(): fake earlier trades for the tick chart back-fill, identity (acml) fixed by the time of day)
// - [2026-11-20] type=feat (investorDaily(): a year of fake investor net buying for the chart pane 투자자 누적)

import type { Bar } from '../shared/price';
import { resampleMinuteBars, weekBars, monthBars, yearBars } from '../shared/price';
import type { OrderBook, Quote, RankRow, TradeTick, WatchRow, MasterItem } from '../shared/kis';
import type { ChartReq, ChartRes, TicksReq, TicksRes, DailyRow, IndexQuote, InfoField, InvestorDay, InvestorRow, MarketBackend, NewsRow, RankReq } from '../shared/market';
import { INDEXES } from '../shared/market';

export const DEMO_MASTER: MasterItem[] = [
  ['005930', '삼성전자', 71200], ['000660', 'SK하이닉스', 182000], ['035420', 'NAVER', 201500], ['035720', '카카오', 41250], ['005380', '현대차', 238000],
  ['051910', 'LG화학', 362000], ['006400', '삼성SDI', 398500], ['207940', '삼성바이오로직스', 812000], ['068270', '셀트리온', 178900], ['003490', '대한항공', 30100],
  ['015760', '한국전력', 21350], ['105560', 'KB금융', 81200], ['055550', '신한지주', 52300], ['000270', '기아', 99800], ['028260', '삼성물산', 141000],
  ['247540', '에코프로비엠', 211500], ['086520', '에코프로', 98700], ['091990', '셀트리온헬스케어', 72100], ['293490', '카카오게임즈', 21850], ['069500', 'KODEX 200', 36420],
].map(([code, name], i) => ({ code: code as string, name: name as string, market: i >= 15 && i < 19 ? 'KOSDAQ' : 'KOSPI', group: i === 19 ? 'EF' : 'ST', std: `KR7${code}00${i % 10}` }));

const BASE: Record<string, number> = {
  '005930': 71200, '000660': 182000, '035420': 201500, '035720': 41250, '005380': 238000, '051910': 362000, '006400': 398500, '207940': 812000,
  '068270': 178900, '003490': 30100, '015760': 21350, '105560': 81200, '055550': 52300, '000270': 99800, '028260': 141000, '247540': 211500,
  '086520': 98700, '091990': 72100, '293490': 21850, '069500': 36420,
};

const KST = 9 * 3600000;
const DAY = 86400000;

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
function rng(seed: number) {
  let s = seed || 1;
  return () => ((s = Math.imul(s ^ (s >>> 15), 2246822507) ^ Math.imul(s ^ (s >>> 13), 3266489909)), ((s ^= s >>> 16) >>> 0) / 4294967296);
}
/** KRX tick size */
export function tickSize(p: number): number {
  return p < 2000 ? 1 : p < 5000 ? 5 : p < 20000 ? 10 : p < 50000 ? 50 : p < 200000 ? 100 : p < 500000 ? 500 : 1000;
}
const roundTick = (p: number) => Math.max(1, Math.round(p / tickSize(p)) * tickSize(p));

/** trading days (Mon-Fri) ending at today's KST midnight */
function tradingDays(n: number): number[] {
  const out: number[] = [];
  let t = Math.floor((Date.now() + KST) / DAY) * DAY - KST;
  while (out.length < n) {
    const dow = new Date(t + KST).getUTCDay();
    if (dow !== 0 && dow !== 6) out.unshift(t);
    t -= DAY;
  }
  return out;
}

function dailySeries(code: string, n = 600): Bar[] {
  const r = rng(hash(code));
  const base = BASE[code] || 10000 + (hash(code) % 90000);
  const days = tradingDays(n);
  // walk backwards from today's base so the latest close stays near the base price
  const closes: number[] = new Array(n);
  closes[n - 1] = base;
  for (let i = n - 2; i >= 0; i--) closes[i] = closes[i + 1] / (1 + (r() - 0.5) * 0.045 + 0.0004);
  return days.map((t, i) => {
    const c = roundTick(closes[i]);
    const o = roundTick(i ? closes[i - 1] * (1 + (r() - 0.5) * 0.01) : c);
    const h = roundTick(Math.max(o, c) * (1 + r() * 0.02));
    const l = roundTick(Math.min(o, c) * (1 - r() * 0.02));
    return { t, o, h, l, c, v: Math.round((200000 + r() * 2000000) * (code === '005930' ? 8 : 1)) };
  });
}

function minuteSeries(code: string, dayT: number, close: number, open: number): Bar[] {
  const r = rng(hash(code + dayT));
  const out: Bar[] = [];
  const start = dayT + 9 * 3600000;
  const nowT = Date.now();
  let p = open;
  for (let i = 0; i < 381; i++) {
    const t = start + i * 60000;
    if (t > nowT) break;
    const drift = (close - p) / Math.max(30, 381 - i);
    const c = roundTick(p + drift + (r() - 0.5) * p * 0.004);
    const h = roundTick(Math.max(p, c) * (1 + r() * 0.0015));
    const l = roundTick(Math.min(p, c) * (1 - r() * 0.0015));
    out.push({ t, o: roundTick(p), h, l, c, v: Math.round(500 + r() * 20000) });
    p = c;
  }
  return out;
}

export class DemoMarket implements MarketBackend {
  readonly kind = 'demo' as const;
  private live = new Map<string, number>();

  private last(code: string): { price: number; prev: number; day: Bar } {
    const d = dailySeries(code);
    const day = d[d.length - 1];
    const prev = d[d.length - 2].c;
    const price = this.live.get(code) ?? day.c;
    return { price, prev, day };
  }

  /** fake realtime trade (called by the demo realtime feed) */
  tick(code: string): TradeTick {
    const { price, prev } = this.last(code);
    const r = Math.random();
    const next = roundTick(price + (r < 0.33 ? -1 : r > 0.66 ? 1 : 0) * tickSize(price));
    this.live.set(code, next);
    const t = new Date(Date.now() + KST).toISOString().slice(11, 19).replace(/:/g, '');
    return { time: t, price: next, change: next - prev, volume: Math.round(1 + Math.random() * 300), side: r > 0.5 ? 1 : 5, strength: 95 + Math.random() * 15 };
  }

  async quote(code: string): Promise<Quote> {
    const { price, prev, day } = this.last(code);
    const d = dailySeries(code, 260);
    const name = DEMO_MASTER.find((m) => m.code === code)?.name;
    const change = price - prev;
    return {
      code,
      price,
      change,
      changePct: (change / prev) * 100,
      sign: change > 0 ? '2' : change < 0 ? '5' : '3',
      open: day.o,
      high: Math.max(day.h, price),
      low: Math.min(day.l, price),
      prevClose: prev,
      upper: roundTick(prev * 1.3),
      lower: roundTick(prev * 0.7),
      volume: day.v,
      amount: day.v * price,
      volRatio: 87.4,
      per: 12.8,
      pbr: 1.31,
      eps: Math.round(price / 12.8),
      bps: Math.round(price / 1.31),
      marketCapEok: Math.round((price * 5_969_782_550) / 1e8 / (code === '005930' ? 1 : 20)),
      listed: 5_969_782_550,
      foreignRate: 53.2,
      w52High: Math.max(...d.map((b) => b.h)),
      w52Low: Math.min(...d.map((b) => b.l)),
      yearHigh: Math.max(...d.slice(-200).map((b) => b.h)),
      yearLow: Math.min(...d.slice(-200).map((b) => b.l)),
      tickUnit: tickSize(price),
      market: DEMO_MASTER.find((m) => m.code === code)?.market === 'KOSDAQ' ? 'KOSDAQ' : 'KOSPI',
      sector: '데모 업종',
      halted: false,
      warn: '',
      pivot: { r2: roundTick(prev * 1.04), r1: roundTick(prev * 1.02), p: prev, s1: roundTick(prev * 0.98), s2: roundTick(prev * 0.96) },
      raw: { hts_kor_isnm: name || code },
    };
  }

  async orderbook(code: string): Promise<OrderBook> {
    const { price } = this.last(code);
    const u = tickSize(price);
    const r = rng(hash(code) + Math.floor(Date.now() / 1000));
    const asks: [number, number, number][] = Array.from({ length: 10 }, (_, i) => [price + (i + 1) * u, Math.round(1000 + r() * 40000), Math.round((r() - 0.5) * 2000)]);
    const bids: [number, number, number][] = Array.from({ length: 10 }, (_, i) => [price - i * u, Math.round(1000 + r() * 40000), Math.round((r() - 0.5) * 2000)]);
    return {
      time: new Date(Date.now() + KST).toISOString().slice(11, 19).replace(/:/g, ''),
      asks,
      bids,
      totalAsk: asks.reduce((a, x) => a + x[1], 0),
      totalBid: bids.reduce((a, x) => a + x[1], 0),
    };
  }

  async trades(code: string): Promise<TradeTick[]> {
    return Array.from({ length: 30 }, () => this.tick(code)).reverse();
  }

  /**
   * fake earlier trades: one a second from 09:00 up to one second ago; the accumulated volume grows with the time of day (second * 400), so
   * every call - with or without a cursor - gives the same trade the same identity, and `until` / the cursor work as with KIS
   */
  async ticks(req: TicksReq): Promise<TicksRes> {
    const toSec = (h: string) => +h.slice(0, 2) * 3600 + +h.slice(2, 4) * 60 + +h.slice(4, 6);
    const hhmmss = (x: number) => [Math.floor(x / 3600), Math.floor(x / 60) % 60, x % 60].map((v) => String(v).padStart(2, '0')).join('');
    const now = new Date(Date.now() + KST).toISOString().slice(11, 19).replace(/:/g, '');
    const open = 9 * 3600;
    let sec = Math.min(toSec(now) - 1, req.cursor ? toSec(req.cursor.hour) - 1 : Infinity);
    // price from the day's bar and the second only, so a trade looks the same in every call
    const base = this.last(req.code).day.c;
    const trades: TradeTick[] = [];
    const max = Math.max(1, Math.min(20, req.pages || 1)) * 30;
    for (; trades.length < max && sec >= open; sec--) {
      const acml = sec * 400;
      if (req.until !== undefined && acml <= req.until) break;
      const h = hash(`${req.code}${sec}`);
      const price = roundTick(base + (((h >> 3) % 21) - 10) * tickSize(base));
      trades.push({ time: hhmmss(sec), price, change: 0, volume: 1 + (h % 300), side: h % 2 ? 1 : 5, acml, rest: true });
    }
    const done = sec < open || (req.until !== undefined && sec * 400 <= req.until);
    return { trades, next: done ? null : { hour: hhmmss(sec + 1), acml: (sec + 1) * 400 }, skipped: 0 };
  }

  async chart(req: ChartReq): Promise<ChartRes> {
    const code = req.index ? `IDX${req.code}` : req.code;
    if (req.kind === 'tick') return { bars: [], note: '틱봉은 실시간 체결로 만듭니다 (데모 체결 1초마다).' };
    const d = dailySeries(code);
    if (req.kind === 'min') {
      const days = Math.max(1, Math.min(5, req.days || 1));
      const bars: Bar[] = [];
      for (let k = days; k >= 1; k--) {
        const day = d[d.length - k];
        bars.push(...minuteSeries(code, day.t, k === 1 ? this.last(code).price : day.c, day.o));
      }
      return { bars: resampleMinuteBars(bars, Math.max(1, req.n)), name: DEMO_MASTER.find((m) => m.code === req.code)?.name };
    }
    // index charts: scale the synthetic walk to the index level shown in the index quote
    const k = req.index ? (req.code === '0001' ? 2584.3 : req.code === '1001' ? 742.1 : 342.6) / d[d.length - 1].c : 1;
    const ds = k === 1 ? d : d.map((b) => ({ ...b, o: b.o * k, h: b.h * k, l: b.l * k, c: b.c * k }));
    const bars = req.kind === 'day' ? ds : req.kind === 'week' ? weekBars(ds) : req.kind === 'month' ? monthBars(ds) : yearBars(ds);
    return { bars, name: DEMO_MASTER.find((m) => m.code === req.code)?.name };
  }

  async daily(code: string): Promise<DailyRow[]> {
    const d = dailySeries(code).slice(-31);
    const r = rng(hash(code + 'f'));
    return d
      .slice(1)
      .map((b, i) => ({ date: new Date(b.t + KST).toISOString().slice(0, 10).replace(/-/g, ''), open: b.o, high: b.h, low: b.l, close: b.c, change: b.c - d[i].c, changePct: ((b.c - d[i].c) / d[i].c) * 100, volume: b.v, foreignNet: Math.round((r() - 0.5) * 400000), foreignRate: 50 + r() * 5 }))
      .reverse();
  }

  async investor(code: string): Promise<InvestorRow[]> {
    const r = rng(hash(code + 'i'));
    return (await this.daily(code)).map((x) => {
      const f = Math.round((r() - 0.5) * 600000);
      const o = Math.round((r() - 0.5) * 400000);
      return { date: x.date, close: x.close, change: x.change, person: -f - o, foreign: f, institution: o, personAmt: Math.round(((-f - o) * x.close) / 1e6), foreignAmt: Math.round((f * x.close) / 1e6), institutionAmt: Math.round((o * x.close) / 1e6) };
    });
  }

  async investorDaily(code: string): Promise<InvestorDay[]> {
    const r = rng(hash(code + 'y'));
    let drift = 0;
    return dailySeries(code)
      .slice(-250)
      .map((b) => {
        drift = drift * 0.9 + (r() - 0.5) * 120000;
        const f = Math.round(drift + (r() - 0.5) * 300000);
        const o = Math.round(-drift * 0.5 + (r() - 0.5) * 200000);
        const amt = (n: number) => Math.round((n * b.c) / 1e6);
        return { date: new Date(b.t + KST).toISOString().slice(0, 10).replace(/-/g, ''), person: -f - o, foreign: f, institution: o, close: b.c, change: b.c - b.o, personAmt: amt(-f - o), foreignAmt: amt(f), institutionAmt: amt(o) };
      });
  }

  async info(code: string): Promise<InfoField[]> {
    const m = DEMO_MASTER.find((x) => x.code === code);
    return [
      { label: '상품약어명', value: m?.name || code },
      { label: '시장', value: m?.market || '-' },
      { label: '표준상품번호', value: m?.std || '-' },
      { label: '상장주수', value: '5,969,782,550' },
      { label: '액면가', value: '100' },
      { label: '결산월일', value: '1231' },
      { label: '비고', value: '데모 데이터입니다 - 로그인하면 KIS 종목정보가 표시됩니다.' },
    ];
  }

  async multi(codes: string[]): Promise<WatchRow[]> {
    return Promise.all(
      codes.map(async (code) => {
        const q = await this.quote(code);
        return { code, name: DEMO_MASTER.find((m) => m.code === code)?.name || code, market: q.market, price: q.price, change: q.change, changePct: q.changePct, volume: q.volume || 0, open: q.open, high: q.high, low: q.low, amount: q.amount };
      }),
    );
  }

  async rank(req: RankReq): Promise<RankRow[]> {
    const rows = await this.multi(DEMO_MASTER.map((m) => m.code));
    const sorted =
      req.type === 'rise' ? rows.sort((a, b) => b.changePct - a.changePct) : req.type === 'fall' ? rows.sort((a, b) => a.changePct - b.changePct) : req.type === 'cap' ? rows.sort((a, b) => b.price - a.price) : rows.sort((a, b) => b.volume - a.volume);
    return sorted.map((x, i) => ({ rank: i + 1, code: x.code, name: x.name, price: x.price, change: x.change, changePct: x.changePct, volume: x.volume, amount: x.amount }));
  }

  async index(code: string): Promise<IndexQuote> {
    const base = code === '0001' ? 2584.3 : code === '1001' ? 742.1 : 342.6;
    const d = dailySeries(`IDX${code}`);
    const ratio = base / d[d.length - 1].c;
    const v = d[d.length - 1].c * ratio;
    const p = d[d.length - 2].c * ratio;
    return { code, name: INDEXES.find((i) => i.code === code)?.name || code, value: v, change: v - p, changePct: ((v - p) / p) * 100, sign: v >= p ? '2' : '5', up: 512, upLimit: 3, flat: 88, down: 401, downLimit: 0 };
  }

  async news(code: string): Promise<NewsRow[]> {
    const name = DEMO_MASTER.find((m) => m.code === code)?.name || code;
    const today = new Date(Date.now() + KST).toISOString().slice(0, 10).replace(/-/g, '');
    return ['실적 발표 일정 안내', '외국인 순매수 지속', '신규 사업 진출 검토', '목표주가 상향'].map((t, i) => ({ date: today, time: `${String(15 - i).padStart(2, '0')}0000`, title: `[데모] ${name} ${t}`, source: '데모', codes: [code] }));
  }
}
