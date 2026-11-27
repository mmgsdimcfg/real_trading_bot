/*
 * Update log
 * - [2026-11-27] type=feat - Results per trading method and mode over a date range (user 2026-11-27: 주간 / 월간): one DayResults
 *   per day (real trades of the daily buy_sell log by method, paper trades of both books), their sum, and the week / month a day
 *   belongs to. Pure functions - the main process reads the files, the 매매 결과 tab shows them
 */

import { pnlKrw as feePnl } from './fees';
import { paperSummary, type PaperBook, type PaperTrade } from './paper';
import { strategyOfTrade, type TradeStrategy } from './strategy';
import type { Trade } from './types';

export interface MethodSum {
  buys: number;
  sells: number;
  wins: number;
  /** realized P&L as logged (before costs) */
  pnlKrw: number;
  /** net of the estimated commission + sell tax (shared/fees); a sell without prices stays gross */
  pnlNet: number;
  winsNet: number;
  /** [2026-11-28] sells without a known result: not in the sums, not in the win rate */
  unknown: number;
}
export interface PaperSum {
  buys: number;
  sells: number;
  wins: number;
  open: number;
  /** net KRW of the closed trades, sized with the bot's per-order budget */
  pnlKrw: number;
}
export interface DayResults {
  /** YYYYMMDD */
  date: string;
  real: Record<TradeStrategy, MethodSum>;
  /** null = that book's file could not be read (no file = zeros) */
  paper: Record<PaperBook, PaperSum | null>;
}
export interface RangeResults {
  ok: boolean;
  days: DayResults[];
  error?: string;
  /** days of the range that could not be read */
  failed?: string[];
}

const zeroReal = (): MethodSum => ({ buys: 0, sells: 0, wins: 0, pnlKrw: 0, pnlNet: 0, winsNet: 0, unknown: 0 });
const zeroPaper = (): PaperSum => ({ buys: 0, sells: 0, wins: 0, open: 0, pnlKrw: 0 });

export function paperDaySum(rows: PaperTrade[], budget: number): PaperSum {
  const s = paperSummary(rows);
  let krw = 0;
  for (const t of rows) {
    if (t.exitTs === undefined || !(t.entry > 0)) continue;
    const q = Math.floor(budget / t.entry);
    if (q > 0) krw += (q * t.entry * (t.netPct ?? 0)) / 100;
  }
  return { buys: rows.length, sells: s.closed, wins: s.wins, open: s.open, pnlKrw: krw };
}

export function dayResults(date: string, trades: Trade[], paper: Record<PaperBook, PaperTrade[] | null>, budget: number): DayResults {
  const real: Record<TradeStrategy, MethodSum> = { rule: zeroReal(), edge: zeroReal(), vwap: zeroReal() };
  for (const t of trades) {
    const e = real[strategyOfTrade(t)];
    if (t.side === 'BUY') {
      e.buys++;
      continue;
    }
    e.sells++;
    if (!Number.isFinite(t.pnlKrw as number) && !(t.buyPrice && t.price && t.qty)) {
      e.unknown++; // no result known: counted apart [Codex review]
      continue;
    }
    const priced = !!(t.buyPrice && t.price && t.qty);
    const gross = Number.isFinite(t.pnlKrw as number) ? (t.pnlKrw as number) : priced ? feePnl(t.buyPrice!, t.price, t.qty, false) : 0;
    const net = priced ? feePnl(t.buyPrice!, t.price, t.qty, true) : gross;
    e.pnlKrw += gross;
    e.pnlNet += net;
    if (gross > 0) e.wins++;
    if (net > 0) e.winsNet++;
  }
  return {
    date,
    real,
    paper: { edge: paper.edge ? paperDaySum(paper.edge, budget) : null, vwap: paper.vwap ? paperDaySum(paper.vwap, budget) : null },
  };
}

export function sumResults(days: DayResults[]): { real: Record<TradeStrategy, MethodSum>; paper: Record<PaperBook, PaperSum> } {
  const real: Record<TradeStrategy, MethodSum> = { rule: zeroReal(), edge: zeroReal(), vwap: zeroReal() };
  const paper: Record<PaperBook, PaperSum> = { edge: zeroPaper(), vwap: zeroPaper() };
  for (const d of days) {
    for (const k of Object.keys(real) as TradeStrategy[]) for (const f of Object.keys(real[k]) as (keyof MethodSum)[]) real[k][f] += d.real[k][f];
    for (const b of Object.keys(paper) as PaperBook[]) {
      const p = d.paper[b];
      if (p) for (const f of Object.keys(paper[b]) as (keyof PaperSum)[]) paper[b][f] += p[f];
    }
  }
  // open paper positions do not add up over days (each day closes its own): the last day's count
  const last = days[days.length - 1];
  for (const b of Object.keys(paper) as PaperBook[]) paper[b].open = last?.paper[b]?.open || 0;
  return { real, paper };
}

// ---- periods (YYYYMMDD, calendar arithmetic in UTC so no time zone shifts a day) ------------------------------------------
export type Period = 'day' | 'week' | 'month' | 'custom';
export const PERIODS: Period[] = ['day', 'week', 'month', 'custom'];
export const PERIOD_KO: Record<Period, string> = { day: '일간', week: '주간', month: '월간', custom: '직접 지정' };

const toDate = (ymd: string) => new Date(Date.UTC(+ymd.slice(0, 4), +ymd.slice(4, 6) - 1, +ymd.slice(6, 8)));
const toYmd = (d: Date) => `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, '0')}${String(d.getUTCDate()).padStart(2, '0')}`;
export const isYmd = (v: unknown): v is string => typeof v === 'string' && /^\d{8}$/.test(v) && toYmd(toDate(v)) === v;
export function addDays(ymd: string, n: number): string {
  const d = toDate(ymd);
  d.setUTCDate(d.getUTCDate() + n);
  return toYmd(d);
}
/** [from, to] of the week (Monday - Sunday) or the month the day is in */
export function periodRange(kind: 'week' | 'month', ymd: string): [string, string] {
  if (kind === 'week') {
    const dow = (toDate(ymd).getUTCDay() + 6) % 7; // Monday = 0
    const from = addDays(ymd, -dow);
    return [from, addDays(from, 6)];
  }
  const d = toDate(ymd);
  const last = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0));
  return [`${ymd.slice(0, 6)}01`, toYmd(last)];
}
/** a day in the previous (dir -1) / next (dir +1) week or month */
export function shiftPeriod(kind: 'week' | 'month', ymd: string, dir: -1 | 1): string {
  if (kind === 'week') return addDays(ymd, 7 * dir);
  const d = toDate(ymd);
  return toYmd(new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + dir, 1)));
}
/** most days one request reads */
export const RANGE_MAX_DAYS = 93;

// ---- [2026-11-27] CSV of a range (user 2026-11-27) ------------------------------------------------------------------------
const csvCell = (v: string | number): string => {
  const t = typeof v === 'number' ? String(Math.round(v)) : v;
  return /[",\r\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
};
/** one row per day and a total row: per method x mode the buys, sells, wins and P&L (real = net of costs when `net`), then the
 *  real total. A paper book that could not be read on a day leaves its cells empty. UTF-8 BOM + CRLF so Excel opens it as is */
export function rangeCsv(days: DayResults[], net: boolean): string {
  const cols: [string, (d: { real: Record<TradeStrategy, MethodSum>; paper: Record<PaperBook, PaperSum | null> }) => (number | '')[]][] = [];
  const real = (k: TradeStrategy) => (d: { real: Record<TradeStrategy, MethodSum> }): number[] => [d.real[k].buys, d.real[k].sells, net ? d.real[k].winsNet : d.real[k].wins, net ? d.real[k].pnlNet : d.real[k].pnlKrw];
  const pap = (k: PaperBook) => (d: { paper: Record<PaperBook, PaperSum | null> }): (number | '')[] => {
    const x = d.paper[k];
    return x ? [x.buys, x.sells, x.wins, x.pnlKrw] : ['', '', '', ''];
  };
  cols.push(['규칙 실제', real('rule')], ['Edge AI 실제', real('edge')], ['Edge AI 가상', pap('edge')], ['VWAP 실제', real('vwap')], ['VWAP 가상', pap('vwap')]);
  const pnlKo = net ? '손익(제세금 포함)' : '손익';
  const head = ['날짜', ...cols.flatMap(([n], i) => ['매수', '매도', '승', i === 0 || i === 1 || i === 3 ? pnlKo : '손익(가상)'].map((f) => `${n} ${f}`)), '실제 합계 매수', '실제 합계 매도', `실제 합계 ${pnlKo}`];
  const line = (label: string, d: { real: Record<TradeStrategy, MethodSum>; paper: Record<PaperBook, PaperSum | null> }) => {
    const r = (['rule', 'edge', 'vwap'] as TradeStrategy[]).map((k) => real(k)(d));
    return [label, ...cols.flatMap(([, f]) => f(d)), r.reduce((a, x) => a + x[0], 0), r.reduce((a, x) => a + x[1], 0), r.reduce((a, x) => a + x[3], 0)].map(csvCell).join(',');
  };
  const rows = days.map((d) => line(`${d.date.slice(0, 4)}-${d.date.slice(4, 6)}-${d.date.slice(6, 8)}`, d));
  if (days.length) rows.push(line('합계', sumResults(days)));
  return '\uFEFF' + [head.map(csvCell).join(','), ...rows].join('\r\n') + '\r\n';
}
