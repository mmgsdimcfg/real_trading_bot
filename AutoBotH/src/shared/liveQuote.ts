/*
 * Update log
 * - [2026-11-12] type=feat - Current price per watched symbol from r003's quote log (runtime/data/live_runtime/quote_ticks_YYYYMMDD.csv,
 *   one row per symbol per poll: ts, code, market, stck_prpr, ..., prdy_ctrt). parseQuoteRows() reads rows by header name,
 *   QuoteBook keeps them per code (latestAt(ts) for the replay cursor), quoteChange() derives the change in won from the price and
 *   the change rate (the file has no previous close): previous close = price / (1 + rate) rounded to the KRX tick of that price
 * - [2026-11-16] type=feat - The day's open / high / low, accumulated volume and traded value of the row (stck_oprc / stck_hgpr /
 *   stck_lwpr / acml_vol / acml_tr_pbmn) for the quote line of the timeline; fmtKrwShort ("2.7억")
 * - [2026-11-16] type=feat - fromMarketQuote: a KIS current-price answer (r016 quote) as a LiveQuote (source 'kis') - the timeline
 *   shows it for symbols r003 does not quote (holdings outside the watch list)
 */

export interface LiveQuote {
  code: string;
  /** epoch ms of the row */
  t: number;
  price: number;
  /** change vs the previous close, % (prdy_ctrt) */
  pct: number;
  /** change in won, derived (see quoteChange) */
  change: number;
  /** J = KRX, NX = NXT - the market the price was asked on */
  market?: string;
  /** [2026-11-16] the day's open / high / low, accumulated volume (shares) and traded value (won) - absent when the file has no column */
  open?: number;
  high?: number;
  low?: number;
  vol?: number;
  value?: number;
  /** [2026-11-16] 'kis' = asked from KIS by the app (r016 quote), otherwise r003's quote log */
  source?: 'kis';
}

/** [2026-11-16] r016 `quote` answer (KIS inquire-price, KRX) -> LiveQuote; null without a price */
export function fromMarketQuote(q: { code: string; asof?: string; price: number; change?: number; change_pct?: number; prev_close?: number;
  open?: number; high?: number; low?: number; volume?: number; amount?: number } | null | undefined): LiveQuote | null {
  if (!q || !(q.price > 0)) return null;
  const t = q.asof ? tsToEpoch(q.asof) : NaN;
  const pct = Number.isFinite(q.change_pct as number) ? (q.change_pct as number)
    : q.prev_close && q.prev_close > 0 ? Math.round((q.price / q.prev_close - 1) * 10000) / 100 : 0;
  const change = Number.isFinite(q.change as number) ? (q.change as number)
    : q.prev_close && q.prev_close > 0 ? q.price - q.prev_close : quoteChange(q.price, pct);
  const out: LiveQuote = { code: q.code, t: Number.isFinite(t) ? t : Date.now(), price: q.price, pct, change, source: 'kis' };
  const pos = (v: number | undefined) => (v !== undefined && v !== null && Number.isFinite(v) && v > 0 ? v : undefined);
  const extra = { open: pos(q.open), high: pos(q.high), low: pos(q.low), vol: pos(q.volume), value: pos(q.amount) };
  for (const [k, v] of Object.entries(extra)) if (v !== undefined) (out as unknown as Record<string, number>)[k] = v;
  return out;
}

/** 2.7억 / 3,450만 / 9,800 - traded value in short Korean units */
export function fmtKrwShort(v: number): string {
  if (!Number.isFinite(v)) return '-';
  const a = Math.abs(v);
  if (a >= 1e12) return `${(v / 1e12).toFixed(1)}조`;
  if (a >= 1e8) return `${(v / 1e8).toLocaleString('ko-KR', { maximumFractionDigits: a >= 1e10 ? 0 : 1 })}억`;
  if (a >= 1e4) return `${Math.round(v / 1e4).toLocaleString('ko-KR')}만`;
  return Math.round(v).toLocaleString('ko-KR');
}

/** KRX tick size at a price (2023-01-25 table, KOSPI = KOSDAQ) */
export function krxTick(price: number): number {
  if (price < 2000) return 1;
  if (price < 5000) return 5;
  if (price < 20000) return 10;
  if (price < 50000) return 50;
  if (price < 200000) return 100;
  if (price < 500000) return 500;
  return 1000;
}

/** change in won from the price and the change rate: the previous close is a price on the tick grid near price / (1 + pct%) */
export function quoteChange(price: number, pct: number): number {
  if (!(price > 0) || !Number.isFinite(pct) || pct <= -100) return 0;
  if (pct === 0) return 0;
  const raw = price / (1 + pct / 100);
  const tick = krxTick(raw);
  return price - Math.round(raw / tick) * tick;
}

/** "11,650 ▲ 60 (+0.52%)" parts; dir: 1 up, -1 down, 0 flat */
export function quoteText(q: Pick<LiveQuote, 'price' | 'pct' | 'change'>): { price: string; change: string; pct: string; dir: -1 | 0 | 1 } {
  const dir = q.pct > 0 ? 1 : q.pct < 0 ? -1 : 0;
  const n = (v: number) => Math.round(v).toLocaleString('ko-KR');
  return {
    price: n(q.price),
    change: `${dir > 0 ? '▲' : dir < 0 ? '▼' : ''} ${n(Math.abs(q.change))}`.trim(),
    pct: `(${q.pct > 0 ? '+' : ''}${q.pct.toFixed(2)}%)`,
    dir,
  };
}

export function quoteHeader(line: string): Map<string, number> {
  return new Map(line.replace(/^﻿/, '').trim().split(',').map((c, i) => [c.trim(), i]));
}

function tsToEpoch(s: string): number {
  // "2026-11-12T09:07:06" is KST wall time
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(s);
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 9, +m[5], +m[6]) : NaN;
}

/** data rows -> quotes (rows without a time, a 6-character code or a positive price are skipped) */
export function parseQuoteRows(lines: string[], cols: Map<string, number>): LiveQuote[] {
  const iT = cols.get('ts') ?? 0;
  const iC = cols.get('code') ?? 1;
  const iM = cols.get('market');
  const iP = cols.get('stck_prpr') ?? 3;
  const iR = cols.get('prdy_ctrt');
  const opt = (name: string) => cols.get(name);
  const iO = opt('stck_oprc');
  const iH = opt('stck_hgpr');
  const iL = opt('stck_lwpr');
  const iV = opt('acml_vol');
  const iA = opt('acml_tr_pbmn');
  const val = (f: string[], i: number | undefined) => {
    if (i === undefined) return undefined;
    const n = Number(f[i]);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  };
  const out: LiveQuote[] = [];
  for (const line of lines) {
    if (!line) continue;
    const f = line.split(',');
    const t = tsToEpoch(f[iT] || '');
    const code = (f[iC] || '').trim();
    const price = Number(f[iP]);
    if (!Number.isFinite(t) || !/^[0-9A-Z]{6}$/.test(code) || !(price > 0)) continue;
    const pct = iR === undefined ? NaN : Number(f[iR]);
    const p = Number.isFinite(pct) ? pct : 0;
    const q: LiveQuote = { code, t, price, pct: p, change: quoteChange(price, p), market: iM === undefined ? undefined : (f[iM] || '').trim() || undefined };
    const extra = { open: val(f, iO), high: val(f, iH), low: val(f, iL), vol: val(f, iV), value: val(f, iA) };
    for (const [k, v] of Object.entries(extra)) if (v !== undefined) (q as unknown as Record<string, number>)[k] = v;
    out.push(q);
  }
  return out;
}

/** the day's quotes per code in file order (time ascending) */
export class QuoteBook {
  private by = new Map<string, LiveQuote[]>();

  add(rows: LiveQuote[]): void {
    for (const q of rows) {
      const list = this.by.get(q.code);
      if (list) list.push(q);
      else this.by.set(q.code, [q]);
    }
  }

  /** newest quote of every code at or before `upTo` (omitted = the newest) */
  latest(upTo?: number | null): Record<string, LiveQuote> {
    const out: Record<string, LiveQuote> = {};
    for (const [code, list] of this.by) {
      if (upTo === undefined || upTo === null) {
        out[code] = list[list.length - 1];
        continue;
      }
      // rows are appended in time order: binary search for the last one with t <= upTo
      let lo = 0;
      let hi = list.length - 1;
      let hit = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (list[mid].t <= upTo) (hit = mid), (lo = mid + 1);
        else hi = mid - 1;
      }
      if (hit >= 0) out[code] = list[hit];
    }
    return out;
  }
}
