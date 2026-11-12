/*
 * Update log
 * - [2026-11-12] type=feat - Current price per watched symbol from r003's quote log (runtime/data/live_runtime/quote_ticks_YYYYMMDD.csv,
 *   one row per symbol per poll: ts, code, market, stck_prpr, ..., prdy_ctrt). parseQuoteRows() reads rows by header name,
 *   QuoteBook keeps them per code (latestAt(ts) for the replay cursor), quoteChange() derives the change in won from the price and
 *   the change rate (the file has no previous close): previous close = price / (1 + rate) rounded to the KRX tick of that price
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

const tsToEpoch = (s: string): number => {
  // "2026-11-12T09:07:06" is KST wall time
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(s);
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 9, +m[5], +m[6]) : NaN;
};

/** data rows -> quotes (rows without a time, a 6-character code or a positive price are skipped) */
export function parseQuoteRows(lines: string[], cols: Map<string, number>): LiveQuote[] {
  const iT = cols.get('ts') ?? 0;
  const iC = cols.get('code') ?? 1;
  const iM = cols.get('market');
  const iP = cols.get('stck_prpr') ?? 3;
  const iR = cols.get('prdy_ctrt');
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
    out.push({ code, t, price, pct: p, change: quoteChange(price, p), market: iM === undefined ? undefined : (f[iM] || '').trim() || undefined });
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
