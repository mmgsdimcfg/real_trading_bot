/*
 * Update log
 * - [2026-11-19] type=perf - Background fetch while the app runs (user 2026-11-19): the timeline / chart tabs asked the server only
 *   while they were open, so a tab switch drew old bars (or nothing for a symbol never seen) until the answer came. This loop keeps
 *   dataCache fresh on every tab, under the same keys the views read (built here, used by the views too):
 *   - focus symbols (timeline selected symbol + chart tab symbol): timeline-chart bars, chart-tab bars, the chart's daily side
 *     request, events/trades, and the r016 quote of the chart symbol only - live day every FOCUS_LIVE_MS
 *   - every watched symbol of the day (snapshot symbols incl. holdings, at most WATCH_MAX): timeline-chart bars + events/trades -
 *     live day every WATCH_LIVE_MS, one request at a time so a view's own request never waits behind a burst
 *   - not live (past day / bot stopped): every IDLE_MS (past days are served from the disk cache of main)
 *   No quotes of the other watched symbols (each r016 quote = Python on the server + KIS). Paused while the window is hidden or
 *   the connection is down; main merges a request already in flight, so a view and this loop asking together cost one call.
 */

import type { PriceRequest, PriceResponse } from '../../shared/price';
import { useStore } from './store';
import { put, storedAt } from './dataCache';

type ChartPrefsLike = { kind: string; tickN: number; minN: number; ind: string[] };
export type Fetch = { key: string; req: PriceRequest };

const intradayKind = (k: string) => k === 'tick' || k === 'min';
const barN = (p: ChartPrefsLike) => (p.kind === 'tick' ? p.tickN : p.minN);

/** timeline price chart request + dataCache key (TimelineView) */
export function timelinePriceFetch(day: string, code: string, chart: ChartPrefsLike, nxt: boolean): Fetch {
  const warmup = chart.ind.length > 0 && intradayKind(chart.kind);
  const n = barN(chart);
  return {
    key: `price:tl/${day}/${code}/${chart.kind}/${n}/${warmup ? 'w' : ''}${nxt ? 'x' : ''}`,
    req: { date: day, code, kind: chart.kind as PriceRequest['kind'], n, warmup, nxt },
  };
}

/** chart tab: main request + the daily side request (null on the daily chart itself) + their dataCache keys (ChartView) */
export function chartPriceFetch(day: string, code: string, prefs: ChartPrefsLike, nxt: boolean): { main: Fetch; day: Fetch | null; dayKey: string } {
  const n = barN(prefs);
  const intraday = intradayKind(prefs.kind);
  const dayKey = `price:ch/${day}/${code}/day`;
  return {
    main: {
      key: `price:ch/${day}/${code}/${prefs.kind}/${n}/${nxt ? 'x' : ''}/${intraday ? 'w' : ''}`,
      req: { date: day, code, kind: prefs.kind as PriceRequest['kind'], n, warmup: intraday, withName: true, nxt },
    },
    day: prefs.kind === 'day' ? null : { key: dayKey, req: { date: day, code, kind: 'day', n: 1 } },
    dayKey,
  };
}

export const timelineKey = (day: string, code: string) => `timeline:${day}/${code}`;
export const quoteKey = (code: string) => `quote:${code}`;

const TICK_MS = 5000;
const FOCUS_LIVE_MS = 10000;
const WATCH_LIVE_MS = 30000;
const IDLE_MS = 300000;
/** a failed request is not repeated before this */
const FAIL_MS = 60000;
const WATCH_MAX = 30;

/** last attempt per key (success or not) - failures are not in dataCache */
const tried = new Map<string, number>();
let busy = false;
let timer: ReturnType<typeof setInterval> | null = null;

function due(key: string, every: number, now: number): boolean {
  const last = Math.max(storedAt(key), tried.get(key) || 0);
  return now - last >= every;
}

/** run() answers whether the fetch succeeded (a failure is retried after FAIL_MS at the earliest) */
type Job = { key: string; every: number; run: () => Promise<boolean> };

function priceJob(f: Fetch, every: number): Job {
  return {
    key: f.key,
    every,
    run: async () => {
      const r: PriceResponse = await window.api.getPrice(f.req);
      if (r.ok) put(f.key, r);
      return r.ok;
    },
  };
}

function jobs(): Job[] {
  const st = useStore.getState();
  const snap = st.snapshot;
  const day = snap?.date || st.status.date;
  if (!day) return [];
  const live = !!st.status.live && day === st.status.date && !snap?.replay;
  const focusEvery = live ? FOCUS_LIVE_MS : IDLE_MS;
  const watchEvery = live ? WATCH_LIVE_MS : IDLE_MS;
  const nxtOf = (code: string) => !!snap?.symbols.find((s) => s.code === code)?.nxt;
  const out: Job[] = [];
  const seen = new Set<string>();
  const add = (j: Job) => {
    if (!seen.has(j.key)) (seen.add(j.key), out.push(j));
  };
  const events = (code: string, every: number): Job => ({
    key: timelineKey(day, code),
    every,
    run: async () => {
      const r = await window.api.getTimeline(code);
      if (useStore.getState().status.date !== day) return false;
      put(timelineKey(day, code), { code, ...r });
      // the store's timeline of the selected symbol follows too (TimelineView draws it without its own request)
      if (useStore.getState().selected === code) useStore.setState({ timeline: { code, ...r } });
      return true;
    },
  });
  const valid = (c: string | null | undefined): c is string => !!c && /^[0-9A-Z]{6}$/.test(c);

  // focus symbols first
  if (valid(st.selected)) {
    add(priceJob(timelinePriceFetch(day, st.selected, st.chart, nxtOf(st.selected)), focusEvery));
    if (day === st.status.date) add(events(st.selected, focusEvery));
  }
  const cc = st.chartCode && st.chartCode.day === day ? st.chartCode.code : null;
  if (valid(cc)) {
    const c = chartPriceFetch(day, cc, st.chartTab, nxtOf(cc));
    add(priceJob(c.main, focusEvery));
    if (c.day) add(priceJob(c.day, focusEvery));
    add({
      key: quoteKey(cc),
      every: focusEvery,
      run: async () => {
        const q = await window.api.getQuote(cc);
        if (q.ok && q.data) put(quoteKey(cc), q.data);
        return !!q.ok;
      },
    });
  }
  // every watched symbol of the day
  for (const s of (snap?.symbols || []).slice(0, WATCH_MAX)) {
    if (!valid(s.code)) continue;
    add(priceJob(timelinePriceFetch(day, s.code, st.chart, !!s.nxt), watchEvery));
    if (day === st.status.date) add(events(s.code, watchEvery));
  }
  return out;
}

async function tick(): Promise<void> {
  if (busy || document.hidden) return;
  if (useStore.getState().status.conn !== 'connected') return;
  busy = true;
  try {
    const mode = useStore.getState().status.mode;
    for (const j of jobs()) {
      const now = Date.now();
      if (!due(j.key, j.every, now)) continue;
      const st = useStore.getState();
      if (document.hidden || st.status.conn !== 'connected' || st.status.mode !== mode) break;
      tried.set(j.key, now);
      if (tried.size > 400) tried.delete(tried.keys().next().value as string);
      const ok = await j.run().catch(() => false);
      if (!ok) tried.set(j.key, now + Math.max(0, FAIL_MS - j.every)); // next try FAIL_MS later at the earliest
      // focus symbols may have changed meanwhile: the next tick rebuilds the list
      if (useStore.getState().status.mode !== mode) break;
    }
  } finally {
    busy = false;
  }
}

/** start the loop once (initStore) */
export function startPrefetch(): void {
  if (timer) return;
  timer = setInterval(() => void tick(), TICK_MS);
  // a host / mode switch forgets the attempts of the old host
  let mode = useStore.getState().status.mode;
  useStore.subscribe((s) => {
    if (s.status.mode !== mode) {
      mode = s.status.mode;
      tried.clear();
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) void tick();
  });
}


/** tests only: one pass of the loop and the attempt map */
export const __prefetchTest = { tick, tried };
