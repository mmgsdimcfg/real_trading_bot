// Update log
// - [2026-11-06] type=feat (Market data contract between main and renderer: chart request/response, investor / daily / info / index / news
//   rows, and the MarketBackend interface implemented by the KIS backend (main/kisMarket.ts) and the demo backend (main/demoMarket.ts))
// - [2026-11-06] type=feat (ticks(): today's individual trades paged backwards - tick charts back-fill earlier trades)
// - [2026-11-20] type=feat (investorDaily(): about one year of daily investor net buying (shares) for the chart pane 투자자 누적)

import type { Bar, ChartKind } from './price';
import type { OrderBook, Quote, RankRow, TradeTick, WatchRow } from './kis';

export interface ChartReq {
  code: string;
  kind: ChartKind;
  /** ticks per bar (tick) / minutes per bar (min) */
  n: number;
  /** minute charts: trading days to load (1 = today; more needs a real account - KIS keeps up to 1 year of minutes) */
  days?: number;
  /** index chart (KOSPI 0001, KOSDAQ 1001, KOSPI200 2001) */
  index?: boolean;
  /** refresh only the newest page and merge (live update) */
  refresh?: boolean;
}

/** today's individual trades, newest first, paged backwards (inquire_time_itemconclusion) */
export interface TicksReq {
  code: string;
  /** continue below this point (from a previous TicksRes.next); none = from now */
  cursor?: TicksCursor;
  /** calls to make (30 trades each), 1 - 20 */
  pages?: number;
  /** stop once a trade with acml <= until was reached (catching up to trades already held) */
  until?: number;
}
export interface TicksCursor {
  /** HHMMSS to ask from */
  hour: string;
  /** accumulated volume of the oldest trade received */
  acml: number;
}
export interface TicksRes {
  trades: TradeTick[];
  /** where to continue; null = reached the start of the day */
  next: TicksCursor | null;
  /** seconds with more than 30 trades - the rest of such a second cannot be fetched and was skipped */
  skipped: number;
}

export interface ChartRes {
  bars: Bar[];
  note?: string;
  name?: string;
  /** [2026-11-19] served from the cache and older than its reuse window - the window asks for a refresh right away */
  stale?: boolean;
}

export interface DailyRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  change: number;
  changePct: number;
  volume: number;
  foreignNet?: number;
  foreignRate?: number;
}

export interface InvestorRow {
  date: string;
  close: number;
  change: number;
  person: number;
  foreign: number;
  institution: number;
  /** net buy amount, million KRW */
  personAmt?: number;
  foreignAmt?: number;
  institutionAmt?: number;
}

/** [2026-11-20] one trading day of investor net buying (investor_trade_by_stock_daily), shares */
export interface InvestorDay {
  date: string;
  person: number;
  foreign: number;
  institution: number;
  /** [2026-11-23] close, change vs the previous close and net buy amounts (million KRW) - the stock info window's table */
  close?: number;
  change?: number;
  personAmt?: number;
  foreignAmt?: number;
  institutionAmt?: number;
}

export interface InfoField {
  label: string;
  value: string;
}

export interface IndexQuote {
  code: string;
  name: string;
  value: number;
  change: number;
  changePct: number;
  sign: string;
  volume?: number;
  amount?: number;
  open?: number;
  high?: number;
  low?: number;
  up?: number;
  upLimit?: number;
  flat?: number;
  down?: number;
  downLimit?: number;
}

export interface NewsRow {
  date: string;
  time: string;
  title: string;
  source: string;
  codes: string[];
}

export type RankType = 'volume' | 'amount' | 'rise' | 'fall' | 'cap';
export interface RankReq {
  type: RankType;
  /** 0000 all, 0001 KOSPI, 1001 KOSDAQ */
  market: '0000' | '0001' | '1001';
}

export interface MarketBackend {
  /** [2026-11-23] 'guest' = through another AutoBotHtsH that is logged in to KIS (main/guest.ts) */
  readonly kind: 'kis' | 'demo' | 'guest';
  quote(code: string): Promise<Quote>;
  orderbook(code: string): Promise<OrderBook>;
  trades(code: string): Promise<TradeTick[]>;
  chart(req: ChartReq): Promise<ChartRes>;
  ticks(req: TicksReq): Promise<TicksRes>;
  daily(code: string): Promise<DailyRow[]>;
  investor(code: string): Promise<InvestorRow[]>;
  /** [2026-11-20] about one year of daily investor net buying, oldest first (chart pane 투자자 누적) */
  investorDaily(code: string): Promise<InvestorDay[]>;
  info(code: string): Promise<InfoField[]>;
  multi(codes: string[]): Promise<WatchRow[]>;
  rank(req: RankReq): Promise<RankRow[]>;
  index(code: string): Promise<IndexQuote>;
  /** [2026-11-26] E-Mini Nasdaq 100 futures index (KIS 해외지수, delayed about 10 minutes): quote + today's 1-minute bars */
  nqFutures(): Promise<NqFutures>;
  news(code: string): Promise<NewsRow[]>;
}

export const INDEXES: { code: string; name: string }[] = [
  { code: '0001', name: 'KOSPI' },
  { code: '1001', name: 'KOSDAQ' },
  { code: '2001', name: 'KOSPI200' },
];

/** [2026-11-26] The index window's Nasdaq-100 card (user 2026-11-26: E-mini Nasdaq 100 futures as a reference). The KIS CME futures feed
 *  is refused for the account (EGW00550, paid subscription), so the card shows this KRX-listed ETF: during the regular session its
 *  5-minute returns move with the futures (correlation about 0.9, FX included). Display only - no trading decision or alert reads it
 *  (AutoBot/nq_futures_indicator_plan_20261126.md). */
export const NQ_PROXY = { code: '133690', name: '나스닥100 (ETF)', full: 'TIGER 미국나스닥100' } as const;

/** [2026-11-26] E-Mini Nasdaq 100 futures as KIS serves it in 해외지수 (mobile app: 투자정보 > 지수/환율/금리 > 해외지수; code of
 *  frgn_code.mst 'FENXH' -> ENXH, market N). Measured 2026-11-26: the newest 1-minute bar is about 10 minutes behind the clock. */
export const NQ_FUTURES = { code: 'ENXH', market: 'N', name: '나스닥100 선물', full: 'E-Mini Nasdaq 100 지수선물' } as const;

export interface NqFutures {
  name: string;
  value: number;
  change: number;
  changePct: number;
  sign: string;
  prevClose?: number;
  /** 1-minute bars, oldest first (KST) */
  bars: Bar[];
  /** start of the newest bar (epoch ms), null without bars */
  lastBarAt: number | null;
}

const fnum = (v: unknown): number => {
  const n = Number(String(v ?? '').replace(/,/g, ''));
  return Number.isFinite(n) ? n : 0;
};

/** 해외지수분봉조회 output1 / output2 -> NqFutures (rows: stck_bsop_date YYYYMMDD, stck_cntg_hour HHMMSS in KST, optn_* prices) */
export function toNqFutures(o1: Record<string, string>, rows: Record<string, string>[]): NqFutures {
  const bars: Bar[] = [];
  for (const r of rows) {
    const d = String(r.stck_bsop_date || '');
    const h = String(r.stck_cntg_hour || '').padStart(6, '0');
    const c = fnum(r.optn_prpr);
    if (!/^\d{8}$/.test(d) || !/^\d{6}$/.test(h) || !(c > 0)) continue;
    const t = Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8), +h.slice(0, 2), +h.slice(2, 4), 0) - 9 * 3600_000;
    bars.push({ t, o: fnum(r.optn_oprc) || c, h: fnum(r.optn_hgpr) || c, l: fnum(r.optn_lwpr) || c, c, v: fnum(r.cntg_vol) });
  }
  bars.sort((a, b) => a.t - b.t);
  const value = fnum(o1.ovrs_nmix_prpr);
  const sign = String(o1.prdy_vrss_sign || '3');
  const abs = Math.abs(fnum(o1.ovrs_nmix_prdy_vrss));
  const down = sign === '4' || sign === '5';
  return {
    name: String(o1.hts_kor_isnm || NQ_FUTURES.full).trim(),
    value,
    change: down ? -abs : abs,
    changePct: (down ? -1 : 1) * Math.abs(fnum(o1.prdy_ctrt)),
    sign,
    prevClose: fnum(o1.ovrs_nmix_prdy_clpr) || undefined,
    bars,
    lastBarAt: bars.length ? bars[bars.length - 1].t : null,
  };
}

/** % change of the newest bar's close against the close `minutes` earlier, from bars of ONE day (bar start times, ascending).
 *  null when the day has no bar that old yet or the newest bar is older than `maxAgeMs` at `now` (a stale value is not shown). */
export function retOverMinutes(bars: { t: number; c: number }[], minutes: number, now: number, maxAgeMs = 5 * 60_000, sameDay = true): number | null {
  if (!bars.length) return null;
  const last = bars[bars.length - 1];
  if (now - last.t > maxAgeMs + 60_000 || now < last.t) return null;
  const day = Math.floor((last.t + 9 * 3600_000) / 86_400_000); // KST day
  const want = last.t - minutes * 60_000;
  let ref: { t: number; c: number } | null = null;
  for (const b of bars) {
    if (sameDay && Math.floor((b.t + 9 * 3600_000) / 86_400_000) !== day) continue;
    if (b.t <= want) ref = b;
    else break;
  }
  if (!ref || !(ref.c > 0) || want - ref.t > 5 * 60_000) return null;
  return (last.c / ref.c - 1) * 100;
}

/** result wrapper over IPC */
export type Res<T> = { ok: true; data: T } | { ok: false; error: string; code?: string };
