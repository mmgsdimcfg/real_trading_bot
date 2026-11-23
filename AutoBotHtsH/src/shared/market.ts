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
  readonly kind: 'kis' | 'demo';
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
  news(code: string): Promise<NewsRow[]>;
}

export const INDEXES: { code: string; name: string }[] = [
  { code: '0001', name: 'KOSPI' },
  { code: '1001', name: 'KOSDAQ' },
  { code: '2001', name: 'KOSPI200' },
];

/** result wrapper over IPC */
export type Res<T> = { ok: true; data: T } | { ok: false; error: string; code?: string };
