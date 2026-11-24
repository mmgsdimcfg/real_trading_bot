// Update log
// - [2026-11-24] type=feat (Quote.vwap = 가중평균가: REST wghn_avrg_stck_prc, realtime WGHN_AVRG_STCK_PRC)
// - [2026-11-06] type=feat (KIS OpenAPI core for HtsH, pure functions (tests/kis.test.ts): server domains (real / paper), the inquiry-only
//   allowlist built from the generated spec (kisSpec.generated.ts) - only GET quotations/ranking TRs and the two realtime market-data TRs
//   are allowed, order / balance / account TRs are rejected by prefix as a second check -, realtime frame parsing ("0|H0STCNT0|003|a^b^..."),
//   the stock master file layout (kospi_code.mst / kosdaq_code.mst, same as stocks_info/kis_*_code_mst.py of the official repo), and the
//   normalizers that turn KIS responses into the app's quote / order book / trades / bars / ranking rows)
// - [2026-11-06] type=feat (inquire_time_itemconclusion (today's trades by time, 30 per call) allowed; toConclusionTrades(); trades carry
//   the accumulated volume (acml) as their identity, from REST acml_vol and realtime ACML_VOL)
// - [2026-11-12] type=feat (NXT: MasterItem.nxt + parseNxtCodes() for the public NXT master files, nxtOnlyHours() and a per-code
//   market code in multPriceParams() - the watchlist asks NXT prices (NX) for NXT symbols while KRX is closed)

import { KIS_SPEC } from './kisSpec.generated';
import type { Bar } from './price';
import { kstToEpoch } from './time';

export type KisEnv = 'real' | 'paper';

export const KIS_DOMAINS: Record<KisEnv, { rest: string; ws: string }> = {
  real: { rest: 'https://openapi.koreainvestment.com:9443', ws: 'ws://ops.koreainvestment.com:21000' },
  paper: { rest: 'https://openapivts.koreainvestment.com:29443', ws: 'ws://ops.koreainvestment.com:31000' },
};

/** REST calls per second the client allows itself (KIS limits are per app key; the bot may share it - stay well below) */
export const RATE_PER_SEC: Record<KisEnv, number> = { real: 8, paper: 2 };

// ---- inquiry-only allowlist ----

/** spec entries HtsH may call (REST) */
export const REST_APIS = [
  'inquire_price',
  'inquire_asking_price_exp_ccn',
  'inquire_ccnl',
  'inquire_time_itemconclusion',
  'inquire_time_itemchartprice',
  'inquire_time_dailychartprice',
  'inquire_daily_itemchartprice',
  'inquire_daily_price',
  'inquire_investor',
  'investor_trade_by_stock_daily',
  'search_stock_info',
  'intstock_multprice',
  'volume_rank',
  'fluctuation',
  'market_cap',
  'inquire_index_price',
  'inquire_daily_indexchartprice',
  'news_title',
  'chk_holiday',
] as const;
export type RestApi = (typeof REST_APIS)[number];

/** realtime TRs HtsH may subscribe to: KRX trades / order book, KRX + NXT combined trades / order book (no 체결통보 / account notices) */
export const WS_TRS = ['H0STCNT0', 'H0STASP0', 'H0UNCNT0', 'H0UNASP0'] as const;

/** renderer-facing TR -> combined (KRX + NXT) TR used for NXT-tradable symbols */
export const WS_COMBINED: Partial<Record<WsTr, WsTr>> = { H0STCNT0: 'H0UNCNT0', H0STASP0: 'H0UNASP0' };

/** combined-TR record -> the KRX record shape the renderer parses (tr renamed, 체결구분 CNTG_CLS_CODE -> CCLD_DVSN) */
export function toKrxRecord(tr: WsTr, rec: Record<string, string>): { tr: WsTr; rec: Record<string, string> } {
  if (tr === 'H0UNCNT0') return { tr: 'H0STCNT0', rec: { ...rec, CCLD_DVSN: rec.CCLD_DVSN ?? rec.CNTG_CLS_CODE ?? '' } };
  if (tr === 'H0UNASP0') return { tr: 'H0STASP0', rec };
  return { tr, rec };
}
export type WsTr = (typeof WS_TRS)[number];

/** tr_id prefixes of orders, balances and account services - never sent, whatever the allowlist says */
const FORBIDDEN_TR = /^(TTTC|VTTC|TTTS|VTTS|TTTK|VTTK|CTSC|CTRP|CTRGA|TTTO|VTTO|CTOS|JTCE|H0STCNI|H0IFCNI)/;
const ALLOWED_PATH = /^\/uapi\/domestic-stock\/v1\/(quotations|ranking)\/[a-z0-9-]+$/;

export interface RestEndpoint {
  api: RestApi;
  path: string;
  trId: string;
}

export function restEndpoint(api: string): RestEndpoint {
  if (!(REST_APIS as readonly string[]).includes(api)) throw new Error(`조회 전용: 허용되지 않은 API ${api}`);
  const spec = KIS_SPEC[api];
  if (!spec || !spec.url || !spec.trIds.length) throw new Error(`KIS 명세에 없는 API: ${api}`);
  const ep = { api: api as RestApi, path: spec.url, trId: spec.trIds[0] };
  assertInquiry(ep.path, ep.trId);
  return ep;
}

/** second line of defence: path and tr_id must look like a market-data inquiry */
export function assertInquiry(path: string, trId: string): void {
  if (FORBIDDEN_TR.test(trId)) throw new Error(`조회 전용: 주문/계좌 TR(${trId})은 사용할 수 없습니다`);
  if (!ALLOWED_PATH.test(path)) throw new Error(`조회 전용: 시세 조회 경로가 아닙니다 (${path})`);
}

export function assertWsTr(trId: string): asserts trId is WsTr {
  if (FORBIDDEN_TR.test(trId) || !(WS_TRS as readonly string[]).includes(trId)) throw new Error(`조회 전용: 실시간 TR ${trId}는 구독할 수 없습니다`);
}

export const isCode = (c: string) => /^[0-9A-Z]{6}$/.test(c);

// ---- realtime frames ----

export interface WsFrame {
  encrypted: boolean;
  trId: string;
  count: number;
  /** one string[] per record (count records of `columns` fields each) */
  records: string[][];
}

/** "0|H0STCNT0|002|f1^f2^...^f1^f2^..." -> records; JSON control messages (PINGPONG, subscribe acks) -> null */
export function parseWsFrame(raw: string, columnsOf: (trId: string) => number): WsFrame | null {
  if (!raw || (raw[0] !== '0' && raw[0] !== '1')) return null;
  const parts = raw.split('|');
  if (parts.length < 4) return null;
  const [enc, trId, cnt] = parts;
  const data = parts.slice(3).join('|');
  const fields = data.split('^');
  const n = columnsOf(trId);
  const count = Math.max(1, parseInt(cnt, 10) || 1);
  const records: string[][] = [];
  if (n > 0) for (let i = 0; i + n <= fields.length && records.length < count; i += n) records.push(fields.slice(i, i + n));
  else records.push(fields);
  return { encrypted: enc === '1', trId, count, records };
}

export function wsColumns(trId: string): string[] {
  const spec = Object.values(KIS_SPEC).find((s) => s.trIds.includes(trId) && s.wsColumns.length);
  return spec ? spec.wsColumns : [];
}

export function recordToObject(trId: string, rec: string[]): Record<string, string> {
  const cols = wsColumns(trId);
  const o: Record<string, string> = {};
  cols.forEach((c, i) => (o[c] = rec[i] ?? ''));
  return o;
}

// ---- stock master (kospi_code.mst / kosdaq_code.mst) ----

export type Market = 'KOSPI' | 'KOSDAQ';

export interface MasterItem {
  code: string;
  name: string;
  market: Market;
  /** 그룹코드: ST 주권, EF ETF, EN ETN, RT 리츠, ... */
  group: string;
  std: string;
  /** also tradable on NXT (nxt_kospi_code.mst / nxt_kosdaq_code.mst) */
  nxt?: boolean;
}

/** NXT master file text (rows start with the 9-character short code like the KRX master) -> codes tradable on NXT */
export function parseNxtCodes(text: string): string[] {
  const out: string[] = [];
  for (const row of text.split('\n')) {
    const c = row.slice(0, 9).trim();
    if (isCode(c)) out.push(c);
  }
  return out;
}

/**
 * KST "HHMM": KRX is closed and only NXT can trade (pre-market 08:00 - 08:50, after-market 15:40 - 20:00). Outside NXT hours the
 * answer stays true so the last NXT price (after-market close) is shown until KRX opens at 09:00.
 */
export const nxtOnlyHours = (hm: string): boolean => hm < '0900' || hm >= '1540';

/** fixed tail width of each row (stocks_info/kis_kospi_code_mst.py: 228, kis_kosdaq_code_mst.py: 222) */
export const MASTER_TAIL: Record<Market, number> = { KOSPI: 228, KOSDAQ: 222 };

/** one decoded (cp949 -> string) row of the master file */
export function parseMasterRow(row: string, market: Market): MasterItem | null {
  const line = row.replace(/\r?\n$/, '');
  const tail = MASTER_TAIL[market];
  if (line.length <= tail + 21) return null;
  const head = line.slice(0, line.length - tail);
  const code = head.slice(0, 9).trim();
  const std = head.slice(9, 21).trim();
  const name = head.slice(21).trim();
  const group = line.slice(line.length - tail, line.length - tail + 2).trim();
  if (!code || !name) return null;
  return { code, name, market, group, std };
}

export const GROUP_LABEL: Record<string, string> = { ST: '주식', EF: 'ETF', EN: 'ETN', RT: '리츠', MF: '펀드', SC: '선박', IF: '인프라', DR: 'DR', SW: '워런트', SR: '신주인수권', BC: '수익증권', FE: '해외ETF', FS: '외국주' };

// ---- normalizers ----

export const num = (v: unknown): number => {
  if (v === null || v === undefined || v === '') return NaN;
  const n = Number(String(v).replace(/,/g, ''));
  return Number.isFinite(n) ? n : NaN;
};
const opt = (v: unknown) => {
  const n = num(v);
  return Number.isFinite(n) ? n : undefined;
};

/** KIS 1상한 2상승 3보합 4하한 5하락 -> signed change */
export const signed = (diff: unknown, sign: unknown): number => {
  const d = Math.abs(num(diff));
  if (!Number.isFinite(d)) return NaN;
  return sign === '4' || sign === '5' ? -d : d;
};

export interface Quote {
  code: string;
  price: number;
  change: number;
  changePct: number;
  sign: string;
  open?: number;
  high?: number;
  low?: number;
  prevClose?: number;
  upper?: number;
  lower?: number;
  volume?: number;
  amount?: number;
  volRatio?: number;
  /** [2026-11-24] 가중평균가 - the day's volume weighted average price (REST wghn_avrg_stck_prc, realtime WGHN_AVRG_STCK_PRC) */
  vwap?: number;
  per?: number;
  pbr?: number;
  eps?: number;
  bps?: number;
  marketCapEok?: number;
  listed?: number;
  foreignRate?: number;
  w52High?: number;
  w52HighDate?: string;
  w52Low?: number;
  w52LowDate?: string;
  yearHigh?: number;
  yearLow?: number;
  tickUnit?: number;
  market?: string;
  sector?: string;
  vi?: string;
  halted?: boolean;
  warn?: string;
  /** pivot points */
  pivot?: { r2?: number; r1?: number; p?: number; s1?: number; s2?: number };
  raw: Record<string, string>;
}

/** inquire_price output -> Quote */
export function toQuote(code: string, o: Record<string, string>): Quote {
  const prev = opt(o.stck_sdpr);
  return {
    code,
    price: num(o.stck_prpr),
    change: signed(o.prdy_vrss, o.prdy_vrss_sign),
    changePct: num(o.prdy_ctrt),
    sign: o.prdy_vrss_sign || '3',
    open: opt(o.stck_oprc),
    high: opt(o.stck_hgpr),
    low: opt(o.stck_lwpr),
    prevClose: prev,
    upper: opt(o.stck_mxpr),
    lower: opt(o.stck_llam),
    volume: opt(o.acml_vol),
    amount: opt(o.acml_tr_pbmn),
    volRatio: opt(o.prdy_vrss_vol_rate),
    vwap: opt(o.wghn_avrg_stck_prc),
    per: opt(o.per),
    pbr: opt(o.pbr),
    eps: opt(o.eps),
    bps: opt(o.bps),
    marketCapEok: opt(o.hts_avls),
    listed: opt(o.lstn_stcn),
    foreignRate: opt(o.hts_frgn_ehrt),
    w52High: opt(o.w52_hgpr),
    w52HighDate: o.w52_hgpr_date,
    w52Low: opt(o.w52_lwpr),
    w52LowDate: o.w52_lwpr_date,
    yearHigh: opt(o.stck_dryy_hgpr),
    yearLow: opt(o.stck_dryy_lwpr),
    tickUnit: opt(o.aspr_unit),
    market: o.rprs_mrkt_kor_name,
    sector: o.bstp_kor_isnm,
    vi: o.vi_cls_code && o.vi_cls_code !== 'N' ? o.vi_cls_code : undefined,
    halted: o.temp_stop_yn === 'Y',
    warn: [o.invt_caful_yn === 'Y' ? '투자유의' : '', o.short_over_yn === 'Y' ? '단기과열' : '', o.mang_issu_cls_code === 'Y' ? '관리종목' : '', o.sltr_yn === 'Y' ? '정리매매' : '']
      .filter(Boolean)
      .join(' '),
    pivot: { r2: opt(o.pvt_scnd_dmrs_prc), r1: opt(o.pvt_frst_dmrs_prc), p: opt(o.pvt_pont_val), s1: opt(o.pvt_frst_dmsp_prc), s2: opt(o.pvt_scnd_dmsp_prc) },
    raw: o,
  };
}

export interface OrderBook {
  time?: string;
  /** level 1..10, [price, qty, qtyChange] */
  asks: [number, number, number][];
  bids: [number, number, number][];
  totalAsk: number;
  totalBid: number;
  expected?: { price: number; volume: number; change: number };
}

/** inquire_asking_price_exp_ccn output1 (+ output2 for the expected price) */
export function toOrderBook(o1: Record<string, string>, o2?: Record<string, string>): OrderBook {
  const asks: [number, number, number][] = [];
  const bids: [number, number, number][] = [];
  for (let i = 1; i <= 10; i++) {
    const ap = num(o1[`askp${i}`]);
    const bp = num(o1[`bidp${i}`]);
    if (ap > 0) asks.push([ap, num(o1[`askp_rsqn${i}`]) || 0, num(o1[`askp_rsqn_icdc${i}`]) || 0]);
    if (bp > 0) bids.push([bp, num(o1[`bidp_rsqn${i}`]) || 0, num(o1[`bidp_rsqn_icdc${i}`]) || 0]);
  }
  const ex = o2 && num(o2.antc_cnpr) > 0 ? { price: num(o2.antc_cnpr), volume: num(o2.antc_vol) || 0, change: signed(o2.antc_cntg_vrss, o2.antc_cntg_vrss_sign) } : undefined;
  return { time: o1.aspr_acpt_hour, asks, bids, totalAsk: num(o1.total_askp_rsqn) || 0, totalBid: num(o1.total_bidp_rsqn) || 0, expected: ex };
}

/** realtime H0STASP0 record -> OrderBook */
export function wsToOrderBook(r: Record<string, string>): OrderBook {
  const lower: Record<string, string> = {};
  for (const [k, v] of Object.entries(r)) lower[k.toLowerCase()] = v;
  const ob = toOrderBook(lower);
  ob.time = r.BSOP_HOUR;
  ob.totalAsk = num(r.TOTAL_ASKP_RSQN) || 0;
  ob.totalBid = num(r.TOTAL_BIDP_RSQN) || 0;
  if (num(r.ANTC_CNPR) > 0) ob.expected = { price: num(r.ANTC_CNPR), volume: num(r.ANTC_VOL) || 0, change: signed(r.ANTC_CNTG_VRSS, r.ANTC_CNTG_VRSS_SIGN) };
  return ob;
}

export interface TradeTick {
  /** HHMMSS */
  time: string;
  price: number;
  change: number;
  volume: number;
  /** 1 buy, 5 sell, 0 unknown */
  side: number;
  strength?: number;
  /** came from a REST trade list (a realtime copy of it replaces it instead of being added) */
  rest?: boolean;
  /** accumulated volume after this trade - unique per trade within a day (REST acml_vol / realtime ACML_VOL) */
  acml?: number;
}

/**
 * inquire_time_itemconclusion output2 rows (newest first) -> trades. Side from the quote at the time: price at / above the ask = buy (1),
 * at / below the bid = sell (5).
 */
export function toConclusionTrades(rows: Record<string, string>[]): TradeTick[] {
  const out: TradeTick[] = [];
  for (const o of rows) {
    const price = num(o.stck_prpr);
    const acml = num(o.acml_vol);
    if (!(price > 0) || !o.stck_cntg_hour || !(acml > 0)) continue;
    const ask = num(o.askp);
    const bid = num(o.bidp);
    out.push({
      time: o.stck_cntg_hour,
      price,
      change: signed(o.prdy_vrss, o.prdy_vrss_sign),
      volume: num(o.cnqn) || 0,
      side: ask > 0 && price >= ask ? 1 : bid > 0 && price <= bid ? 5 : 0,
      strength: opt(o.tday_rltv),
      acml,
    });
  }
  return out;
}

/** inquire_ccnl output rows (newest first) */
export function toTrades(rows: Record<string, string>[]): TradeTick[] {
  return rows.map((o) => ({ time: o.stck_cntg_hour, price: num(o.stck_prpr), change: signed(o.prdy_vrss, o.prdy_vrss_sign), volume: num(o.cntg_vol) || 0, side: 0, strength: opt(o.tday_rltv) }));
}

/** realtime H0STCNT0 record -> trade + the quote fields it updates */
export function wsToTrade(r: Record<string, string>): { trade: TradeTick; quote: Partial<Quote> } {
  const change = signed(r.PRDY_VRSS, r.PRDY_VRSS_SIGN);
  const trade: TradeTick = { time: r.STCK_CNTG_HOUR, price: num(r.STCK_PRPR), change, volume: num(r.CNTG_VOL) || 0, side: r.CCLD_DVSN === '1' ? 1 : r.CCLD_DVSN === '5' ? 5 : 0, strength: opt(r.CTTR), acml: opt(r.ACML_VOL) };
  return {
    trade,
    quote: {
      price: trade.price,
      change,
      changePct: num(r.PRDY_CTRT),
      sign: r.PRDY_VRSS_SIGN,
      open: opt(r.STCK_OPRC),
      high: opt(r.STCK_HGPR),
      low: opt(r.STCK_LWPR),
      volume: opt(r.ACML_VOL),
      amount: opt(r.ACML_TR_PBMN),
      vwap: opt(r.WGHN_AVRG_STCK_PRC),
      halted: r.TRHT_YN === 'Y',
    },
  };
}

/** KIS date "YYYYMMDD" + time "HHMMSS" -> epoch ms (KST) */
export function kisTime(date: string, time = '000000'): number {
  const t = time.padStart(6, '0');
  return kstToEpoch(`${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}`, `${t.slice(0, 2)}:${t.slice(2, 4)}:${t.slice(4, 6)}`);
}

/** minute chart rows (inquire_time_itemchartprice / inquire_time_dailychartprice output2) -> 1-minute bars, oldest first, no duplicates */
export function minuteRowsToBars(rows: Record<string, string>[]): Bar[] {
  const byT = new Map<number, Bar>();
  for (const o of rows) {
    const c = num(o.stck_prpr);
    if (!(c > 0) || !o.stck_bsop_date || !o.stck_cntg_hour) continue;
    // stck_cntg_hour is the START of the minute (102200 = 10:22:00-10:22:59; documented in the trading bot's r003, 2026-10-06)
    const t = kisTime(o.stck_bsop_date, o.stck_cntg_hour);
    byT.set(t, { t, o: num(o.stck_oprc) || c, h: num(o.stck_hgpr) || c, l: num(o.stck_lwpr) || c, c, v: num(o.cntg_vol) || 0 });
  }
  return [...byT.values()].sort((a, b) => a.t - b.t);
}

/** period chart rows (inquire_daily_itemchartprice / inquire_daily_indexchartprice output2) -> bars, oldest first */
export function dailyRowsToBars(rows: Record<string, string>[]): Bar[] {
  const byT = new Map<number, Bar>();
  for (const o of rows) {
    const c = num(o.stck_clpr ?? o.bstp_nmix_prpr);
    if (!(c > 0) || !o.stck_bsop_date) continue;
    const t = kisTime(o.stck_bsop_date);
    byT.set(t, {
      t,
      o: num(o.stck_oprc ?? o.bstp_nmix_oprc) || c,
      h: num(o.stck_hgpr ?? o.bstp_nmix_hgpr) || c,
      l: num(o.stck_lwpr ?? o.bstp_nmix_lwpr) || c,
      c,
      v: num(o.acml_vol) || 0,
    });
  }
  return [...byT.values()].sort((a, b) => a.t - b.t);
}

export interface RankRow {
  rank: number;
  code: string;
  name: string;
  price: number;
  change: number;
  changePct: number;
  volume: number;
  amount?: number;
  extra?: number;
}

/** volume_rank / fluctuation / market_cap rows */
export function toRankRows(rows: Record<string, string>[], extraField?: string): RankRow[] {
  return rows
    .map((o) => ({
      rank: num(o.data_rank),
      code: o.mksc_shrn_iscd || o.stck_shrn_iscd || '',
      name: o.hts_kor_isnm || '',
      price: num(o.stck_prpr),
      change: signed(o.prdy_vrss, o.prdy_vrss_sign),
      changePct: num(o.prdy_ctrt),
      volume: num(o.acml_vol) || 0,
      amount: opt(o.acml_tr_pbmn),
      extra: extraField ? opt(o[extraField]) : undefined,
    }))
    .filter((r) => r.code);
}

export interface WatchRow {
  code: string;
  name: string;
  market?: string;
  price: number;
  change: number;
  changePct: number;
  volume: number;
  open?: number;
  high?: number;
  low?: number;
  ask?: number;
  bid?: number;
  amount?: number;
}

/** intstock_multprice rows */
export function toWatchRows(rows: Record<string, string>[]): WatchRow[] {
  return rows
    .map((o) => ({
      code: o.inter_shrn_iscd || '',
      name: o.inter_kor_isnm || '',
      market: o.kospi_kosdaq_cls_name,
      price: num(o.inter2_prpr),
      change: signed(o.inter2_prdy_vrss, o.prdy_vrss_sign),
      changePct: num(o.prdy_ctrt),
      volume: num(o.acml_vol) || 0,
      open: opt(o.inter2_oprc),
      high: opt(o.inter2_hgpr),
      low: opt(o.inter2_lwpr),
      ask: opt(o.inter2_askp),
      bid: opt(o.inter2_bidp),
      amount: opt(o.acml_tr_pbmn),
    }))
    .filter((r) => r.code);
}

/** multi-price params for up to 30 codes; `marketOf` = market code per code (J KRX, NX NXT - the TR takes one per symbol) */
export function multPriceParams(codes: string[], marketOf: (code: string) => string = () => 'J'): Record<string, string> {
  const p: Record<string, string> = {};
  codes.slice(0, 30).forEach((c, i) => {
    p[`FID_COND_MRKT_DIV_CODE_${i + 1}`] = marketOf(c);
    p[`FID_INPUT_ISCD_${i + 1}`] = c;
  });
  return p;
}

/** Korean label of a response field (from the official repo's COLUMN_MAPPING) */
export function fieldLabel(api: string, field: string): string {
  return KIS_SPEC[api]?.labels[field] || field;
}
