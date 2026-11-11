/*
 * Update log
 * - [2026-11-05] type=feat - Price chart data for the timeline tab: parsers for the server's tick files
 *   (runtime/data/ticks/YYYYMMDD/<code>.csv - real trades of watched symbols, kept ~5 days), the 10-second files
 *   (runtime/data/YYYYMMDD/<code>_<name>_10s.txt - only raw_bar=1 rows are real 1-minute bars, the rest are interpolated) and the
 *   daily files (<code>_<name>_daily.csv, ~200 trading days), and the bar builders: N-tick bars, M-minute bars (from ticks or from
 *   1-minute bars), weekly / monthly bars from daily bars, and today's daily bar from intraday data when the daily file has no row
 *   for it yet. Pure functions (tests/price.test.ts)
 * - [2026-11-06] type=feat - PriceResponse.warm / warmDate: previous trading day's bars for indicator warm-up (WARMUP_BARS = 120)
 * - [2026-11-06] type=feat - ServerBars / serverRowsToBars for r016 `bars` (server-side aggregation, daily_history, KIS fallback once added)
 *   and MarketQuote for the requested r016 `quote` (current price, 10-level order book, latest trades)
 */

import { kstToEpoch } from './time';

export type ChartKind = 'tick' | 'min' | 'day' | 'week' | 'month';

export interface Tick {
  /** epoch ms (KST trade time) */
  t: number;
  p: number;
  v: number;
  /** ccld_dvsn: 1 = buy-initiated, 5 = sell-initiated, 3 = pre-market (KIS) */
  s?: number;
  /** best ask / bid at the trade (askp1 / bidp1, 0 = unknown) */
  a?: number;
  b?: number;
}

/** current-price panel of the chart tab, from the day's tick file */
export interface Quote {
  /** last trade */
  t: number;
  price: number;
  ask?: number;
  bid?: number;
  /** volume of buy- / sell-initiated trades today (체결강도 = buyVol / sellVol * 100) */
  buyVol: number;
  sellVol: number;
  volume: number;
  trades: number;
  /** latest trades, newest first */
  recent: Tick[];
}

export interface Bar {
  /** epoch ms of the first trade / bar start (daily: KST midnight of the day) */
  t: number;
  /** epoch ms of the last trade in the bar (tick bars) */
  t2?: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

/** ticks / min1 / daily = files read over SFTP; server = r016 `bars` (daily_history, or KIS once the server adds it) */
export type PriceSource = 'ticks' | 'min1' | 'daily' | 'server';

/** r016_control.py `bars` output (autoboth_chart_data_reply_20261106 P3-1; `name` / KIS sources per the 2026-11-06 KIS request) */
export interface ServerBars {
  ok: boolean;
  kind: string;
  code: string;
  date: string;
  source: string;
  columns: string[];
  bars: (string | number)[][];
  warm?: (string | number)[][];
  warm_date?: string | null;
  next_since?: string | null;
  name?: string;
  error?: string;
}

/** r016_control.py `quote` (requested 2026-11-06: KIS current price + 10-level order book + latest trades) */
export interface MarketQuote {
  ok: boolean;
  code: string;
  name?: string;
  market?: string;
  asof?: string;
  source?: string;
  cached_ms?: number;
  price: number;
  change?: number;
  change_pct?: number;
  prev_close?: number;
  base_price?: number;
  open?: number;
  high?: number;
  low?: number;
  volume?: number;
  amount?: number;
  upper_limit?: number;
  lower_limit?: number;
  w52_high?: number;
  w52_high_date?: string;
  w52_low?: number;
  w52_low_date?: string;
  per?: number;
  pbr?: number;
  market_cap_eok?: number;
  vi?: boolean;
  halted?: boolean;
  strength?: number;
  orderbook?: { asof?: string; asks: [number, number][]; bids: [number, number][]; total_ask?: number; total_bid?: number };
  trades?: [string, number, number, string][];
  error?: string;
}

/** rows of a server `bars` response -> Bar[] (t = bar start, KST ISO "YYYY-MM-DDTHH:MM:SS") */
export function serverRowsToBars(columns: string[] | undefined, rows: (string | number)[][] | undefined): Bar[] {
  if (!rows?.length) return [];
  const cols = columns || ['t', 'o', 'h', 'l', 'c', 'v'];
  const ix = (k: string, d: number) => {
    const i = cols.indexOf(k);
    return i >= 0 ? i : d;
  };
  const [it, io, ih, il, ic, iv] = [ix('t', 0), ix('o', 1), ix('h', 2), ix('l', 3), ix('c', 4), ix('v', 5)];
  const out: Bar[] = [];
  for (const r of rows) {
    const m = /^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?/.exec(String(r[it] ?? ''));
    if (!m) continue;
    const b: Bar = { t: kstToEpoch(m[1], m[2] || '00:00:00'), o: Number(r[io]), h: Number(r[ih]), l: Number(r[il]), c: Number(r[ic]), v: Number(r[iv]) || 0 };
    if (b.o > 0 && b.h > 0 && b.l > 0 && b.c > 0) out.push(b);
  }
  return out;
}

export interface PriceRequest {
  date: string;
  code: string;
  kind: ChartKind;
  /** ticks per bar (tick) or minutes per bar (min); ignored for day/week/month */
  n: number;
  /** also return the previous trading day's bars (indicator warm-up) */
  warmup?: boolean;
  /** also return the symbol name (chart tab) */
  withName?: boolean;
}

export interface PriceResponse {
  ok: boolean;
  error?: string;
  bars: Bar[];
  source?: PriceSource;
  /** e.g. "틱 파일 없음 -> 1분봉(10초 파일)으로 대체" */
  note?: string;
  /** daily: the YYYYMMDD folder the daily file came from */
  folder?: string;
  /** tick/min: the previous trading day's bars of the same size (whole day, at most WARMUP_BARS_MAX) - MA lines / Ichimoku warm-up */
  warm?: Bar[];
  /** tick/min: the bot's warm-up window - the previous day's last PRIOR_DAY_WARMUP_MINUTES one-minute bars, resampled (tick: last 120 bars) */
  warmBot?: Bar[];
  /** tick/min with a tick file: current price, best ask/bid, buy/sell volume, latest trades */
  quote?: Quote;
  /** symbol name from the server file name <code>_<name>_daily.csv (codes typed in the chart tab) */
  name?: string;
  /** source detail for the status line (server bars: file path or "kis:<TR>") */
  sourceDetail?: string;
  /** YYYYMMDD of the warm-up bars */
  warmDate?: string;
  /** [2026-11-11] tick/min from a tick file: minute ranges (epoch ms, [from, to)) the tick collector missed inside the session */
  gaps?: [number, number][];
  /** min: how many 1-minute bars the KIS minute chart added for those gaps */
  filled?: number;
}

export const WARMUP_BARS = 120;
export const WARMUP_BARS_MAX = 400;

/** Current-price panel data from the day's trades. */
export function quoteFromTicks(ticks: Tick[], recent = 40): Quote | undefined {
  if (!ticks.length) return undefined;
  const last = ticks[ticks.length - 1];
  let buyVol = 0;
  let sellVol = 0;
  let volume = 0;
  let ask: number | undefined;
  let bid: number | undefined;
  for (const k of ticks) {
    volume += k.v;
    if (k.s === 1) buyVol += k.v;
    else if (k.s === 5) sellVol += k.v;
  }
  for (let i = ticks.length - 1; i >= 0 && (ask === undefined || bid === undefined); i--) {
    if (ask === undefined && ticks[i].a) ask = ticks[i].a;
    if (bid === undefined && ticks[i].b) bid = ticks[i].b;
  }
  return { t: last.t, price: last.p, ask, bid, buyVol, sellVol, volume, trades: ticks.length, recent: ticks.slice(-recent).reverse() };
}

export const TICK_SIZES = [1, 5, 10, 30, 60, 120, 300];
export const MIN_SIZES = [1, 3, 5, 10, 15, 30, 60];

const DAY_MS = 86400000;
const KST_MS = 9 * 3600000;

/** KST midnight (epoch ms) of a YYYYMMDD string */
export function ymdEpoch(ymd: string): number {
  return Date.UTC(+ymd.slice(0, 4), +ymd.slice(4, 6) - 1, +ymd.slice(6, 8)) - KST_MS;
}

const num = (s: string | undefined) => {
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
};

/** Column index map of a CSV header (BOM stripped). */
function header(line: string): Map<string, number> {
  return new Map(
    line
      .replace(/^﻿/, '')
      .split(',')
      .map((h, i) => [h.trim(), i] as [string, number]),
  );
}

/**
 * Tick CSV lines (header: recv_ts,code,bsop_date,trade_time,price,cntg_vol,...). `cols` is the header map of the file; pass the
 * lines after the header (incremental reads keep the map from the first chunk).
 */
export function parseTickLines(lines: string[], cols: Map<string, number>): Tick[] {
  const iD = cols.get('bsop_date') ?? 2;
  const iT = cols.get('trade_time') ?? 3;
  const iP = cols.get('price') ?? 4;
  const iV = cols.get('cntg_vol') ?? 5;
  const iS = cols.get('ccld_dvsn');
  const iA = cols.get('askp1');
  const iB = cols.get('bidp1');
  const out: Tick[] = [];
  for (const line of lines) {
    if (!line || line.charCodeAt(0) < 48 || line.charCodeAt(0) > 57) continue;
    const f = line.split(',');
    const d = f[iD];
    const tm = (f[iT] || '').padStart(6, '0');
    const p = num(f[iP]);
    if (!d || d.length !== 8 || !(p > 0)) continue;
    const t = kstToEpoch(`${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`, `${tm.slice(0, 2)}:${tm.slice(2, 4)}:${tm.slice(4, 6)}`);
    if (!Number.isFinite(t)) continue;
    const k: Tick = { t, p, v: num(f[iV]) || 0 };
    if (iS !== undefined) k.s = num(f[iS]) || 0;
    if (iA !== undefined) k.a = num(f[iA]) || 0;
    if (iB !== undefined) k.b = num(f[iB]) || 0;
    out.push(k);
  }
  return out;
}

export function tickHeader(line: string): Map<string, number> {
  return header(line);
}

/** 10-second file -> real 1-minute bars (raw_bar = 1 rows only). */
export function parse10sMinuteBars(text: string): Bar[] {
  const lines = text.split(/\r?\n/);
  if (!lines.length) return [];
  const h = header(lines[0]);
  const iDt = h.get('datetime') ?? 0;
  const iRaw = h.get('raw_bar');
  const [iO, iH, iL, iC, iV] = ['open', 'high', 'low', 'close', 'volume'].map((k, i) => h.get(k) ?? i + 1);
  const out: Bar[] = [];
  for (let i = 1; i < lines.length; i++) {
    const f = lines[i].split(',');
    if (f.length < 5) continue;
    if (iRaw !== undefined && f[iRaw]?.trim() !== '1') continue;
    const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})/.exec(f[iDt] || '');
    if (!m) continue;
    const b = { t: kstToEpoch(m[1], m[2]), o: num(f[iO]), h: num(f[iH]), l: num(f[iL]), c: num(f[iC]), v: num(f[iV]) || 0 };
    if (b.o > 0 && b.h > 0 && b.l > 0 && b.c > 0) out.push(b);
  }
  return out;
}

/** daily.csv -> daily bars (t = KST midnight), sorted by date */
export function parseDailyCsv(text: string): Bar[] {
  const lines = text.split(/\r?\n/);
  if (!lines.length) return [];
  const h = header(lines[0]);
  const iD = h.get('date') ?? 0;
  const [iO, iH, iL, iC, iV] = ['open', 'high', 'low', 'close', 'volume'].map((k, i) => h.get(k) ?? i + 1);
  const out: Bar[] = [];
  for (let i = 1; i < lines.length; i++) {
    const f = lines[i].split(',');
    const d = (f[iD] || '').trim();
    if (!/^\d{8}$/.test(d)) continue;
    const b = { t: ymdEpoch(d), o: num(f[iO]), h: num(f[iH]), l: num(f[iL]), c: num(f[iC]), v: num(f[iV]) || 0 };
    if (b.o > 0 && b.h > 0 && b.l > 0 && b.c > 0) out.push(b);
  }
  out.sort((a, b) => a.t - b.t);
  // the same date twice -> keep the last row
  return out.filter((b, i) => i === out.length - 1 || out[i + 1].t !== b.t);
}

/** Every `n` trades -> one bar. */
export function tickBars(ticks: Tick[], n: number): Bar[] {
  const size = Math.max(1, Math.floor(n));
  const out: Bar[] = [];
  for (let i = 0; i < ticks.length; i += size) {
    const seg = ticks.slice(i, i + size);
    let h = -Infinity;
    let l = Infinity;
    let v = 0;
    for (const k of seg) {
      if (k.p > h) h = k.p;
      if (k.p < l) l = k.p;
      v += k.v;
    }
    out.push({ t: seg[0].t, t2: seg[seg.length - 1].t, o: seg[0].p, h, l, c: seg[seg.length - 1].p, v });
  }
  return out;
}

/** bucket start of `t` for an m-minute grid aligned to KST midnight */
const bucket = (t: number, m: number) => {
  const span = m * 60000;
  const day = Math.floor((t + KST_MS) / DAY_MS) * DAY_MS - KST_MS;
  return day + Math.floor((t - day) / span) * span;
};

/** Trades -> m-minute bars (bars without trades are skipped). */
export function minuteBarsFromTicks(ticks: Tick[], m: number): Bar[] {
  const out: Bar[] = [];
  let cur: Bar | null = null;
  for (const k of ticks) {
    const b = bucket(k.t, Math.max(1, m));
    if (!cur || cur.t !== b) {
      cur = { t: b, o: k.p, h: k.p, l: k.p, c: k.p, v: 0 };
      out.push(cur);
    }
    if (k.p > cur.h) cur.h = k.p;
    if (k.p < cur.l) cur.l = k.p;
    cur.c = k.p;
    cur.v += k.v;
  }
  return out;
}

/** 1-minute bars -> m-minute bars */
export function resampleMinuteBars(bars: Bar[], m: number): Bar[] {
  if (m <= 1) return bars;
  const out: Bar[] = [];
  let cur: Bar | null = null;
  for (const x of bars) {
    const b = bucket(x.t, m);
    if (!cur || cur.t !== b) {
      cur = { t: b, o: x.o, h: x.h, l: x.l, c: x.c, v: 0 };
      out.push(cur);
    }
    cur.h = Math.max(cur.h, x.h);
    cur.l = Math.min(cur.l, x.l);
    cur.c = x.c;
    cur.v += x.v;
  }
  return out;
}

/** Monday 00:00 KST of the week holding `t` */
function weekKey(t: number): number {
  const d = new Date(t + KST_MS);
  const dow = (d.getUTCDay() + 6) % 7; // Mon = 0
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() - dow) - KST_MS;
}

function monthKey(t: number): number {
  const d = new Date(t + KST_MS);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1) - KST_MS;
}

function groupBars(daily: Bar[], key: (t: number) => number): Bar[] {
  const out: Bar[] = [];
  let cur: Bar | null = null;
  let curKey = NaN;
  for (const x of daily) {
    const k = key(x.t);
    if (!cur || k !== curKey) {
      curKey = k;
      cur = { t: x.t, o: x.o, h: x.h, l: x.l, c: x.c, v: 0 };
      out.push(cur);
    }
    cur.h = Math.max(cur.h, x.h);
    cur.l = Math.min(cur.l, x.l);
    cur.c = x.c;
    cur.v += x.v;
  }
  return out;
}

/** weekly bars (t = first trading day of the week) */
export const weekBars = (daily: Bar[]) => groupBars(daily, weekKey);
/** monthly bars (t = first trading day of the month) */
export const monthBars = (daily: Bar[]) => groupBars(daily, monthKey);

/** True when the bar of `t` belongs to the same week / month / day as `ymd`. */
export function samePeriod(kind: ChartKind, t: number, ymd: string): boolean {
  const d = ymdEpoch(ymd);
  if (kind === 'week') return weekKey(t) === weekKey(d);
  if (kind === 'month') return monthKey(t) === monthKey(d);
  return t === d;
}

/** Daily bars up to `ymd`; adds `ymd`'s bar from intraday bars when the daily file has no row for it yet (live day). */
export function dailyWithToday(daily: Bar[], ymd: string, intraday: Bar[]): Bar[] {
  const d = ymdEpoch(ymd);
  const upTo = daily.filter((b) => b.t <= d);
  if (upTo.length && upTo[upTo.length - 1].t === d) return upTo;
  // regular session only (09:00 - 15:30 close auction): the tick files also hold pre-market trades from 08:30, which the official
  // daily bar does not include (checked against a daily row: open/high/low match the 09:00-15:30 ticks exactly)
  const day = intraday.filter((b) => b.t >= d + 9 * 3600000 && b.t < d + (15 * 60 + 31) * 60000);
  if (!day.length) return upTo;
  const today: Bar = { t: d, o: day[0].o, h: Math.max(...day.map((b) => b.h)), l: Math.min(...day.map((b) => b.l)), c: day[day.length - 1].c, v: day.reduce((a, b) => a + b.v, 0) };
  return [...upTo, today];
}

// ---- [2026-11-11] tick-collector gaps: the minute chart fills them with the server's KIS minute bars ----
const MINUTE_MS = 60000;
/** regular session on `ymd`: [09:00, 15:30) KST as epoch ms */
export function sessionRange(ymd: string): [number, number] {
  const day = ymdEpoch(ymd);
  return [day + 9 * 3600000, day + 15.5 * 3600000];
}
/**
 * Minute slots missing from tick-derived 1-minute bars inside the regular session, as [from, to) epoch-ms ranges. Only runs of at
 * least `minLen` minutes count (a quiet minute without a print is not an outage). `until` caps the scan (now while live - the
 * current minute may still be forming, so the scan ends one minute before it).
 */
export function minuteGaps(m1: Bar[], ymd: string, until = Infinity, minLen = 3): [number, number][] {
  const [open, close] = sessionRange(ymd);
  const end = Math.min(close, until) - MINUTE_MS;
  if (end <= open) return [];
  const have = new Set(m1.map((b) => Math.floor(b.t / MINUTE_MS)));
  const out: [number, number][] = [];
  const first = Math.floor(open / MINUTE_MS);
  const last = Math.floor(end / MINUTE_MS);
  let run = -1;
  const flush = (upto: number) => {
    if (run >= 0 && upto - run >= minLen) out.push([run * MINUTE_MS, upto * MINUTE_MS]);
    run = -1;
  };
  for (let m = first; m <= last; m++) {
    if (have.has(m)) flush(m);
    else if (run < 0) run = m;
  }
  flush(last + 1);
  return out;
}
/** total minutes covered by gap ranges */
export const gapMinutes = (gaps: [number, number][]): number => gaps.reduce((s, [a, b]) => s + Math.round((b - a) / MINUTE_MS), 0);
/** "09:24~09:40, 11:02~11:05" (KST) */
export function gapText(gaps: [number, number][]): string {
  const hm = (t: number) => {
    const d = new Date(t + 9 * 3600000);
    return `${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
  };
  return gaps.map(([a, b]) => `${hm(a)}~${hm(b)}`).join(', ');
}
/**
 * KIS 1-minute bars (complete for the day) as the base; tick-derived bars for minutes KIS does not have and from KIS's last
 * minute on (that bar may be partial on the KIS side while the ticks are up to date).
 */
export function mergeMinuteBars(kis: Bar[], ticks: Bar[]): Bar[] {
  const by = new Map<number, Bar>();
  for (const b of kis) by.set(Math.floor(b.t / MINUTE_MS), b);
  const lastKis = kis.length ? Math.floor(kis[kis.length - 1].t / MINUTE_MS) : -1;
  for (const b of ticks) {
    const m = Math.floor(b.t / MINUTE_MS);
    if (!by.has(m) || m >= lastKis) by.set(m, b);
  }
  return [...by.entries()].sort((x, y) => x[0] - y[0]).map(([, b]) => b);
}
