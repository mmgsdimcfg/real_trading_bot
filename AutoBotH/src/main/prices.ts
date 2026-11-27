/*
 * Update log
 * - [2026-11-05] type=feat - Price data for the timeline chart, read-only over SFTP (plus one `ls` per symbol/day to find the
 *   name-suffixed files): tick file runtime/data/ticks/YYYYMMDD/<code>.csv read incrementally by byte offset (the live file
 *   grows all day), real 1-minute bars from runtime/data/YYYYMMDD/<code>_<name>_10s.txt (fallback when there is no tick file),
 *   daily bars from <code>_<name>_daily.csv of that day's folder (or the latest earlier folder). Bars are built here
 *   (shared/price.ts) so only bars cross IPC. Small in-memory caches, re-read only when the remote size changes. Requests are
 *   serialized (a 1-minute request still reading a >4 MB tick file plus a 5-minute request read the same chunk twice and lost
 *   the bars after ~14:00)
 * - [2026-11-06] type=feat - Indicator warm-up: with req.warmup, tick/minute responses carry the previous trading day's bars of the
 *   same size (that day's tick file, or its 1-minute bars for minute charts), last 120
 * - [2026-11-06] type=feat - Any code (chart tab): daily bars come from the server's r016 `bars --kind day` first (daily_history, ~14 months,
 *   or KIS once the server adds it - see docs/server_kis_market_data_request_20261106.md), week/month are built from them here; tick /
 *   minute bars still read the files over SFTP first (incremental, quote from ticks) and fall back to `bars` when the server has no file
 *   for the code/day (not collected, today's minutes before g001 runs). An older server without `bars` keeps the SFTP-only behaviour
 * - [2026-11-07] type=perf - r016 `bars` answers are reused (daily 10 min, intraday outside market hours 5 min, failures 1 min):
 *   the chart refreshes every 10 s and each server call starts Python on the Raspberry Pi (and may cost KIS calls)
 * - [2026-11-06] type=feat - Warm-up like the bot (r012 load_prior_day_warmup): minute charts use the previous day's raw 1-minute bars
 *   of the 10-second file - warmBot = last PRIOR_DAY_WARMUP_MINUTES (180) resampled, warm = whole day (MA / Ichimoku); responses with
 *   a tick file carry a quote (last price, best ask/bid, buy/sell volume, latest 40 trades) for the chart tab
 * - [2026-11-11] type=feat - Tick-collector gaps: for tick / min charts the tick-derived 1-minute bars are scanned for missing
 *   minutes inside the regular session (shared/price minuteGaps, runs of 3+ minutes). The minute chart then merges the day's KIS
 *   1-minute bars (r016 `bars --kind min --source kis`, kisMinuteFill, one call per minute per day/code) under the tick bars
 *   (mergeMinuteBars); the tick chart only notes the gap. Response gains gaps / filled and a note
 * - [2026-11-07] type=feat - Past days read through PriceFileCache (priceCache.ts): tick / 10-second / daily files and the "<code>_*"
 *   lists are downloaded whole once into %APPDATA%/autoboth/pricecache/<date>/ and read locally after that - also offline. The
 *   server's daily answer (r016 bars --kind day) of a past day is kept there too. prefetch(date, codes) downloads the files of
 *   the day's symbols and the previous trading day's warm-up files in the background (one code per queue slot, so chart
 *   requests are not held up for the whole download). The live day keeps the incremental SFTP reads
 * - [2026-11-10] type=feat - Reads through a FileHost (host.ts): the Raspberry Pi over SFTP or this PC's runtime folder (local mode).
 *   The past-day download cache is used for the remote host only (local files are read in place); the "<code>_*" listing is
 *   FileHost.listPrefix
 * - [2026-11-12] type=fix - Charts were empty before the KRX open (NXT pre-market): the tick collector records KRX trades only and
 *   the server had no minute bars before 09:00. (1) Minute chart without files: r016 `bars` now answers with KRX + NXT combined
 *   minutes (server change); answers are reused 10 s in NXT-only hours. (2) Minute chart of an NXT symbol (req.nxt) with a tick
 *   file: the combined minutes of the hours KRX is closed (r016 `bars --source kis-un`) are merged under the tick bars.
 *   (3) Tick chart without a tick file: the day's 1-minute bars are shown instead of an error, with a note
 * - [2026-11-12] type=feat - liveQuotes(date, upTo): current price per watched symbol from r003's quote log
 *   (runtime/data/live_runtime/quote_ticks_<date>.csv), read incrementally by byte offset like the tick files
 * - [2026-11-18] type=perf - Symbol switches re-read whole files / re-asked the server: one 12-entry limit was shared by every
 *   cache and each symbol takes two tick entries (the day + the warm-up day), so cycling through ~10 watched symbols evicted them
 *   and today's tick file (several MB over SFTP) was read from byte 0 again. Limits are now per cache (ticks / 1-minute / daily 40,
 *   file lists 160, server answers 80) and a read marks the entry recently used. KIS minute bars (tick-gap fill, NXT pre/after
 *   market) are kept while they still answer the request: a past day's for good, gap fills while they cover every missing minute,
 *   the NXT combined minutes fetched after 09:00 until the KRX close (the pre-market minutes no longer change) - instead of a new
 *   r016 call (Python on the Pi + KIS) every minute
 * - [2026-11-18] type=perf - Past days' server answers are kept on disk (pricecache/<date>/): r016 `bars --kind min` answers
 *   (codes without a tick / 10-second file) as server_min/<code>_<n>[_w].json and the KIS minute bars (gap fill / NXT combined)
 *   as kis_min/<code>[_un].json - a past day is asked from the server once, also across app restarts
 * - [2026-11-18] type=feat - Chart history (the chart tab loads earlier days while panning left): tick / minute answers name the
 *   previous trading day (warmDate) even when that day has no file to warm up from, so the next older day can still be asked
 * - [2026-11-18] type=feat - Yearly bars (kind 'year', grouped here like week / month). The server's daily answer now reaches back
 *   to 2018 (r017 + g040 daily_backfill): a past day's saved daily answer is server_day/<code>.v2.json, so answers saved before
 *   (daily_history only, ~14 months) are asked again once
 * - [2026-11-19] type=feat - KRX after-hours market (user 2026-11-19: 403870 HPSP does not trade on NXT but trades on KRX 16:00 -
 *   20:00, its chart stopped at the KRX close): minute charts of EVERY symbol now get the minutes outside the KRX regular session
 *   from r016 `bars --source kis-ext` (server picks the venue: NXT symbol = KRX+NXT combined minutes, otherwise the KRX after-hours
 *   minutes) - was the NXT symbols only (kis-un). Also for the 10-second-file minutes. The regular session comes from the server's
 *   r014 hours (CSAT day 10:00 - 16:30) and the server's NXT flag is remembered per code: a known KRX-only symbol is not asked
 *   before the KRX open (no pre-market). Responses carry hours / nxt. An older server without kis-ext falls back to kis-un for
 *   symbols flagged NXT. Saved past-day answers use new names (server_min/*.v2.json, kis_min/<code>_ext.json): the old ones end
 *   at the KRX close
 * - [2026-11-19] type=fix - Daily chart ended at 11/17 (403870 HPSP; many symbols had no 11/18): the 11/18 collect failed (KIS
 *   outage) and HPSP has no tick file to build the day's bar from. The server (r017) now fills the trading days missing after
 *   daily_history / the folder daily.csv from KIS (source "+ kis", noted on the chart). Saved past-day daily answers are
 *   server_day/<code>.v3.json (v2 answers could lack those days)
 */

import { parseStrategyEvalRows, type StrategyEvalRowX } from '../shared/sideTimeline';
import { LineBuffer } from '../shared/lineBuffer';
import { CHART_CACHE_MS, isMarketHours, isNxtOnlyHours, kstNowIso } from '../shared/control';
import {
  dailyWithToday,
  gapMinutes,
  gapText,
  mergeMinuteBars,
  minuteBarsFromTicks,
  minuteGaps,
  groupDaily,
  parse10sMinuteBars,
  parseDailyCsv,
  parseTickLines,
  resampleMinuteBars,
  tickBars,
  tickHeader,
  ymdEpoch,
  serverRowsToBars,
  type ServerBars,
  WARMUP_BARS,
  WARMUP_BARS_MAX,
  quoteFromTicks,
  type Bar,
  type PriceRequest,
  type PriceResponse,
  type Tick,
  type TradingHours,
} from '../shared/price';
import type { FileHost } from './host';
import { parseQuoteRows, quoteHeader, QuoteBook, type LiveQuote } from '../shared/liveQuote';
import { BOT_PARAMS } from '../shared/indicators';
import type { R016Client } from './control';
import type { PriceFileCache } from './priceCache';

export interface PrefetchProgress {
  date: string;
  done: number;
  total: number;
  failed: number;
  /** last error (e.g. the connection dropped) */
  error?: string;
}

const MAX_READ = 4 * 1024 * 1024;
/** [2026-11-18] entries kept per cache (was one limit of 12 for all) */
const LIMIT = { ticks: 40, files: 160, min1: 40, daily: 40, server: 80, fill: 80 } as const;

interface TickCache {
  off: number;
  lb: LineBuffer;
  cols?: Map<string, number>;
  ticks: Tick[];
}

/** KIS minute bars for gap filling are re-asked at most this often per day/code (the server caches them too) */
const KIS_FILL_TTL_MS = 60000;
/** r016 `bars` answers reused this long in NXT-only hours (each call starts Python; the chart refreshes with every log poll) */
const NXT_HOURS_TTL_MS = 10000;
/** KST minute of the day of a bar */
const kstMinuteOfDay = (t: number) => Math.floor(((t + 9 * 3600000) % 86400000) / 60000);
/** [2026-11-19] "HH:MM" -> minute of the day */
const hmMinute = (s: string | undefined, dflt: number) => {
  const m = /^(\d{1,2}):(\d{2})/.exec(s || '');
  return m ? +m[1] * 60 + +m[2] : dflt;
};
/** [2026-11-19] KRX regular session [open, close) in minutes of the day (server r014 hours; 09:00 - 15:30 without them) */
export const regularMinutes = (hours?: TradingHours | null): [number, number] => [hmMinute(hours?.regular?.[0], 9 * 60), hmMinute(hours?.regular?.[1], 15 * 60 + 30)];
/** bars of the hours KRX's regular session is closed (before the open, from the close; 09:00 / 15:30 without hours) */
export const outsideKrxHours = (bars: Bar[], hours?: TradingHours | null): Bar[] => {
  const [open, close] = regularMinutes(hours);
  return bars.filter((b) => kstMinuteOfDay(b.t) < open || kstMinuteOfDay(b.t) >= close);
};
/** [2026-11-19] kind of KIS minute fill: tick-collector gaps (kis), combined market (kis-un, older servers), outside the regular session (kis-ext) */
type FillMode = 'gap' | 'un' | 'ext';
type FillRes = { bars?: Bar[]; error?: string; hours?: TradingHours; nxt?: boolean | null; unsupported?: boolean };
/**
 * 1-minute bars of the day + the combined-market bars of the hours KRX is closed. In those minutes the combined bar wins: the tick
 * file only has KRX pre-open / after-hours prints there, the combined bar has the NXT trades.
 */
export function mergeOutsideKrx(base: Bar[], outside: Bar[]): Bar[] {
  const taken = new Set(outside.map((b) => Math.floor(b.t / 60000)));
  return [...outside, ...base.filter((b) => !taken.has(Math.floor(b.t / 60000)))].sort((a, b) => a.t - b.t);
}

/** [2026-11-19] hours / nxt fields of a response (left out when unknown) */
function sessionInfo(x: { hours?: TradingHours | null; nxt?: boolean | null }): Pick<PriceResponse, 'hours' | 'nxt'> {
  return { ...(x.hours ? { hours: x.hours } : {}), ...(typeof x.nxt === 'boolean' ? { nxt: x.nxt } : {}) };
}

/** keep at most `limit` entries (least recently used dropped) */
function remember<K, V>(m: Map<K, V>, k: K, v: V, limit: number): V {
  m.delete(k);
  m.set(k, v);
  while (m.size > limit) m.delete(m.keys().next().value as K);
  return v;
}

/** the entry of `k`, marked recently used (moved to the end, so it is dropped last) */
function touch<K, V>(m: Map<K, V>, k: K): V | undefined {
  const v = m.get(k);
  if (v !== undefined) {
    m.delete(k);
    m.set(k, v);
  }
  return v;
}

/** [2026-11-18] every minute of the gap ranges ([start, end) ms) has a bar in `bars` */
export function coversGaps(bars: Bar[], gaps: [number, number][]): boolean {
  if (!gaps.length) return true;
  const have = new Set(bars.map((b) => Math.floor(b.t / 60000)));
  for (const [a, b] of gaps) for (let m = Math.floor(a / 60000); m < Math.floor(b / 60000); m++) if (!have.has(m)) return false;
  return true;
}

/** r016 `bars` result: null = unavailable (old server / no client) */
type ServerBarsResult = { ok: true; data: ServerBars } | { ok: false; error: string } | null;

/** most bytes of a strategy trace file read by one call */
const EVAL_CHUNK = 16_000_000;

export class PriceService {
  private ticks = new Map<string, TickCache>();
  private files = new Map<string, { at: number; names: string[] }>();
  private min1 = new Map<string, { size: number; bars: Bar[] }>();
  private daily = new Map<string, { size: number; bars: Bar[] }>();
  private dirs: { at: number; list: string[] } | null = null;
  /** r016 `bars` answers reused for a while (each call = Python on the server, maybe KIS): key = args joined */
  private serverCache = new Map<string, { at: number; ttl: number; res: ServerBarsResult }>();
  /** [2026-11-11] KIS minute bars used to fill tick-collector gaps, per day/code (re-asked at most every KIS_FILL_TTL_MS) */
  private fillCache = new Map<string, { at: number; res: FillRes }>();
  /** [2026-11-19] the server's NXT flag per code and r014 hours per day (from tick / minute answers) */
  private nxtOf = new Map<string, boolean>();
  private hoursOf = new Map<string, TradingHours>();
  /** r003 quote log of one day (the newest day asked): byte offset + parsed rows */
  private quoteLog: { key: string; off: number; lb: LineBuffer; cols?: Map<string, number>; book: QuoteBook } | null = null;
  /** requests run one at a time: two concurrent readers of the same incremental tick cache would read the same byte range twice */
  private tail: Promise<unknown> = Promise.resolve();

  /** the server has no `bars` command (older r016) - stop asking until reconnect */
  private noServerBars = false;

  constructor(
    private host: FileHost,
    private root: () => string,
    private r016: () => R016Client | null = () => null,
    /** local copies of past days' files; null = always read the server */
    private cache: PriceFileCache | null = null,
    /** true for a day before the server's today (its files no longer change) */
    private isPast: (date: string) => boolean = () => false,
  ) {}

  private cached(date: string): PriceFileCache | null {
    return this.cache && this.host.kind === 'remote' && this.isPast(date) ? this.cache : null;
  }

  resetServerSupport(): void {
    this.noServerBars = false;
    this.serverCache.clear();
  }

  /** the host changed (run mode switch): forget everything read from the old one */
  resetAll(): void {
    this.evalCache.clear();
    this.resetServerSupport();
    this.ticks.clear();
    this.files.clear();
    this.min1.clear();
    this.daily.clear();
    this.dirs = null;
    this.quoteLog = null;
  }

  /** Current price per code from r003's quote log of `date`, at or before `upTo` (replay cursor; omitted = newest). */
  liveQuotes(date: string, upTo?: number | null): Promise<Record<string, LiveQuote>> {
    const job = this.tail.then(() => this.readQuoteLog(date, upTo));
    this.tail = job.catch(() => undefined);
    return job;
  }

  /** [2026-11-12] text of the day's Edge-AI-only paper trade file (null = no file: the mode was off or nothing was bought) */
  paperLog(date: string, prefix = 'edge_only'): Promise<string | null> {
    const job = this.tail.then(async () => {
      if (!/^\d{8}$/.test(date) || !/^[a-z_]+$/.test(prefix)) return null;
      // [2026-11-27] prefix: 'edge_only' (Edge AI trade, r022) or 'vwap_paper' (VWAP trade, r029)
      const path = `${this.root()}/runtime/data/live_runtime/${prefix}_${date}.jsonl`;
      const st = await this.host.stat(path);
      if (!st || !st.size) return null;
      return (await this.host.readRange(path, 0, st.size)).toString('utf8');
    });
    this.tail = job.catch(() => undefined);
    return job;
  }

  /** [2026-11-27] text of a day's daily trade log runtime/log/<date>/<date>_buy_sell.log (null = none: no trade that day) */
  dayTradeLog(date: string): Promise<string | null> {
    const job = this.tail.then(async () => {
      if (!/^\d{8}$/.test(date)) return null;
      const path = `${this.root()}/runtime/log/${date}/${date}_buy_sell.log`;
      const st = await this.host.stat(path);
      if (!st || !st.size || st.size > 20_000_000) return null;
      return (await this.host.readRange(path, 0, st.size)).toString('utf8');
    });
    this.tail = job.catch(() => undefined);
    return job;
  }

  /** [2026-11-27] per-bar evaluation trace of a side strategy (server r030: strategy_eval_<strategy>_<date>.jsonl). The file
   *  only grows, so each call reads the bytes added since the last one and keeps the parsed rows (whole lines only). */
  private evalCache = new Map<string, { size: number; rows: StrategyEvalRowX[] }>();
  /** dropped with everything else when the host / folder changes (resetAll) */
  clearStrategyEvals(): void {
    this.evalCache.clear();
  }
  strategyEvals(date: string, strategy: string): Promise<StrategyEvalRowX[]> {
    const job = this.tail.then(async () => {
      if (!/^\d{8}$/.test(date) || !/^(edge|vwap)$/.test(strategy)) return [];
      const path = `${this.root()}/runtime/data/live_runtime/strategy_eval_${strategy}_${date}.jsonl`;
      const key = `${this.root()}|${path}`;
      const st = await this.host.stat(path);
      if (!st || !st.size) return [];
      let c = this.evalCache.get(key);
      if (!c || st.size < c.size) c = { size: 0, rows: [] }; // new or replaced file: from the start
      if (st.size > c.size) {
        // [Codex round 3] at most EVAL_CHUNK bytes a call (a longer unread stretch is caught up over the next calls); the offset
        // advances by BYTES up to the last line break of what was read, never by a re-encoded text length
        const buf = await this.host.readRange(path, c.size, Math.min(st.size, c.size + EVAL_CHUNK));
        const cut = buf.lastIndexOf(0x0a);
        if (cut >= 0) {
          c = { size: c.size + cut + 1, rows: c.rows.concat(parseStrategyEvalRows(buf.subarray(0, cut + 1).toString('utf8'))) };
          this.evalCache.set(key, c);
          if (this.evalCache.size > 6) this.evalCache.delete(this.evalCache.keys().next().value as string);
        }
      }
      return c.rows;
    });
    this.tail = job.then(() => undefined).catch(() => undefined);
    return job;
  }

  /** [2026-11-19] text of the day's paper status file (server r022 write_status; null = none: paper mode off / bot not run) */
  paperStatus(date: string, prefix = 'edge_only'): Promise<string | null> {
    const job = this.tail.then(async () => {
      if (!/^\d{8}$/.test(date) || !/^[a-z_]+$/.test(prefix)) return null;
      const path = `${this.root()}/runtime/data/live_runtime/${prefix}_status_${date}.json`;
      const st = await this.host.stat(path);
      if (!st || !st.size || st.size > 4_000_000) return null;
      return (await this.host.readRange(path, 0, st.size)).toString('utf8');
    });
    this.tail = job.catch(() => undefined);
    return job;
  }

  /** [2026-11-12] text of the Edge AI model meta the bot loads; null = no model published.
   *  [2026-11-27] the Edge AI trade model (runtime/edge_ai/trade/current_meta.json, g014 --role trade) when it exists - that is what
   *  the paper / live Edge AI trades read (r011 TRADE_MODEL_SUBDIR) - else runtime/edge_ai/current_meta.json as before */
  edgeMeta(): Promise<string | null> {
    const job = this.tail.then(async () => {
      for (const rel of ['trade/current_meta.json', 'current_meta.json']) {
        const path = `${this.root()}/runtime/edge_ai/${rel}`;
        const st = await this.host.stat(path).catch(() => null);
        if (!st || !st.size || st.size > 2_000_000) continue;
        return (await this.host.readRange(path, 0, st.size)).toString('utf8');
      }
      return null;
    });
    this.tail = job.catch(() => undefined);
    return job;
  }

  private async readQuoteLog(date: string, upTo?: number | null): Promise<Record<string, LiveQuote>> {
    if (!/^\d{8}$/.test(date)) return {};
    const path = `${this.root()}/runtime/data/live_runtime/quote_ticks_${date}.csv`;
    const st = await this.host.stat(path);
    if (!st) return {};
    let c = this.quoteLog;
    if (!c || c.key !== path || st.size < c.off) c = this.quoteLog = { key: path, off: 0, lb: new LineBuffer(), book: new QuoteBook() };
    while (c.off < st.size) {
      const buf = await this.host.readRange(path, c.off, Math.min(st.size, c.off + MAX_READ));
      if (!buf.length) break;
      c.off += buf.length;
      let lines = c.lb.push(new Uint8Array(buf.buffer, buf.byteOffset, buf.length));
      if (!c.cols && lines.length) {
        c.cols = quoteHeader(lines[0]);
        lines = lines.slice(1);
      }
      if (c.cols) c.book.add(parseQuoteRows(lines, c.cols));
    }
    return c.book.latest(upTo);
  }

  /**
   * r016 `bars` with reuse: daily answers 10 min, intraday answers outside market hours 5 min, failures 1 min
   * (a code the server cannot serve is not asked again every 10 s).
   */
  private async serverBars(args: string[]): Promise<ServerBarsResult> {
    const key = args.join(' ');
    const hit = this.serverCache.get(key);
    if (hit && Date.now() - hit.at < hit.ttl) return hit.res;
    const res = await this.serverBarsRaw(args);
    if (res) {
      const isDaily = args.includes('day');
      const now = kstNowIso();
      const ttl = !res.ok ? CHART_CACHE_MS.failed : isDaily ? CHART_CACHE_MS.daily : isMarketHours(now) ? 0 : isNxtOnlyHours(now) ? NXT_HOURS_TTL_MS : CHART_CACHE_MS.intradayIdle;
      if (ttl > 0) remember(this.serverCache, key, { at: Date.now(), ttl, res }, LIMIT.server);
    }
    return res;
  }

  /**
   * [2026-11-11] The day's KIS 1-minute bars (r016 `bars --kind min --source kis`) for filling minutes the tick collector missed.
   * One server call per KIS_FILL_TTL_MS per day/code - the chart refreshes with every snapshot, the fill does not have to.
   */
  private async kisMinuteFill(date: string, code: string, mode: FillMode = 'gap', gaps: [number, number][] = []): Promise<FillRes> {
    const suffix = mode === 'gap' ? '' : `_${mode}`;
    const key = `${date}/${code}${suffix}`;
    // [2026-11-18] a past day's KIS minutes never change: kept on disk with the day
    const fc = this.cached(date);
    const rel = `kis_min/${code}${suffix}.json`;
    if (fc && !this.fillCache.has(key)) {
      const saved = fc.json<Bar[] | FillRes>(date, rel);
      // [2026-11-19] kis-ext answers are saved with hours / nxt ({bars, hours, nxt}), the older ones as a bar list
      const res: FillRes | null = Array.isArray(saved) ? (saved.length ? { bars: saved } : null) : saved && Array.isArray(saved.bars) ? saved : null;
      if (res) remember(this.fillCache, key, { at: Date.now(), res }, LIMIT.fill);
    }
    const hit = touch(this.fillCache, key);
    // an empty kis-ext answer is an answer too (a KRX-only symbol before its after-hours market)
    const usable = (r: FillRes) => (mode === 'ext' ? !r.error && !!r.bars : !!r.bars?.length);
    if (hit && (Date.now() - hit.at < KIS_FILL_TTL_MS || (usable(hit.res) && this.fillStillGood(date, mode, hit.at, hit.res.bars || [], gaps, hit.res.hours)))) {
      return hit.res;
    }
    let res: FillRes;
    const source = mode === 'gap' ? 'kis' : mode === 'un' ? 'kis-un' : 'kis-ext';
    const c = this.r016();
    if (!c || this.noServerBars) res = { error: '서버에 bars 명령이 없습니다 (이전 버전 r016)' };
    else {
      try {
        const r = await c.run<ServerBars>(['bars', '--date', date, '--code', code, '--kind', 'min', '--interval', '1', '--source', source], { timeoutMs: 60000 });
        if (r.ok && r.data?.ok) {
          res = { bars: serverRowsToBars(r.data.columns, r.data.bars) };
          if (r.data.hours) res.hours = r.data.hours;
          if (typeof r.data.nxt === 'boolean') res.nxt = r.data.nxt;
        } else if (/unrecognized arguments|invalid choice/.test(`${r.error || ''} ${r.stderr || ''}`)) res = { error: `서버 r017이 이전 버전이라 --source ${source} 가 없습니다 (git pull 필요)`, unsupported: true };
        else res = { error: r.data?.error || r.error || 'bars 실패' };
      } catch (e) {
        res = { error: String((e as Error)?.message || e).split('\n')[0] };
      }
    }
    remember(this.fillCache, key, { at: Date.now(), res }, LIMIT.fill);
    if (fc && res.bars && (res.bars.length || mode === 'ext') && !res.error) fc.saveJson(date, rel, mode === 'ext' ? res : res.bars);
    return res;
  }

  /**
   * [2026-11-18] a kept KIS minute answer still serves the request (no new r016 call): a past day's never changes; gap fill = it
   * has a bar for every missing minute; minutes outside the regular session = fetched after the KRX open of that day while the
   * regular session is still open (the pre-market minutes are final, the after-market ones start at the close)
   * [2026-11-19] the open / close of the day from the server's r014 hours (CSAT day 10:00 - 16:30)
   */
  private fillStillGood(date: string, mode: FillMode, at: number, bars: Bar[], gaps: [number, number][], hours?: TradingHours | null): boolean {
    if (this.isPast(date)) return true;
    if (mode === 'gap') return coversGaps(bars, gaps);
    const day0 = ymdEpoch(date); // 00:00 KST of the day (epoch ms)
    const [open, close] = regularMinutes(hours || this.hoursOf.get(date));
    return at >= day0 + open * 60000 && Date.now() < day0 + close * 60000;
  }

  /**
   * [2026-11-19] 1-minute bars of the day + the KIS minutes outside the KRX regular session (r016 `bars --source kis-ext`: NXT
   * pre / after market of an NXT symbol, the KRX after-hours market of every symbol). Outside the session the KIS bar wins (the
   * tick file has KRX regular prints only). A code the server flagged KRX-only is not asked before the KRX open.
   */
  private async withOutside(date: string, code: string, req: PriceRequest, live: boolean, base: Bar[]): Promise<{ bars: Bar[]; added: number; note?: string; hours?: TradingHours; nxt?: boolean | null }> {
    const known = this.nxtOf.get(code);
    const hoursKnown = this.hoursOf.get(date);
    if (live && known === false && Date.now() < ymdEpoch(date) + regularMinutes(hoursKnown)[0] * 60000) {
      return { bars: base, added: 0, hours: hoursKnown, nxt: known };
    }
    let fill = await this.kisMinuteFill(date, code, 'ext');
    let viaUn = false;
    if (fill.unsupported && (req.nxt || known)) {
      fill = await this.kisMinuteFill(date, code, 'un'); // older server: the combined minutes, NXT symbols only (as before)
      viaUn = true;
    }
    if (typeof fill.nxt === 'boolean') remember(this.nxtOf, code, fill.nxt, LIMIT.files);
    if (fill.hours) remember(this.hoursOf, date, fill.hours, LIMIT.daily);
    const hours = fill.hours || hoursKnown;
    const nxt = typeof fill.nxt === 'boolean' ? fill.nxt : known ?? (viaUn ? true : null);
    const outside = fill.bars ? outsideKrxHours(fill.bars, hours) : [];
    if (!outside.length) return { bars: base, added: 0, hours, nxt };
    const note = nxt
      ? `NXT 장전/장후·KRX 장후 ${outside.length}분은 KIS 통합 분봉입니다 (틱 수집기는 KRX 정규장 체결만 수집)`
      : `KRX 장후 ${outside.length}분은 KIS 분봉입니다 (NXT 비대상 종목 - 장전 없음, 틱 수집기는 KRX 정규장 체결만 수집)`;
    return { bars: mergeOutsideKrx(base, outside), added: outside.length, note, hours, nxt };
  }

  /** r016 `bars` (read-only). null = unavailable (old server / no client) */
  private async serverBarsRaw(args: string[]): Promise<ServerBarsResult> {
    const c = this.r016();
    if (!c || this.noServerBars) return null;
    const r = await c.run<ServerBars>(['bars', ...args], { timeoutMs: 60000 });
    if (r.ok && r.data?.ok) return { ok: true, data: r.data };
    if (/invalid choice/.test(`${r.error || ''} ${r.stderr || ''}`)) {
      this.noServerBars = true;
      return null;
    }
    return { ok: false, error: r.data?.error || r.error || 'bars 실패' };
  }

  private static sourceNote(src: string): string {
    if (src.startsWith('kis')) return `KIS 시세 조회 (${src.replace(/^kis:?/, '') || 'REST'})`;
    // [2026-11-19] "+ kis": trading days missing from daily_history / the folder daily.csv (collect failed) came from KIS
    const kis = / \+ kis\b/.test(src) ? ' · 최근 누락 일봉은 KIS 보충' : '';
    if (src.includes('daily_backfill')) return `서버 일봉 이력 (daily_history + 장기 일봉 daily_backfill)${kis}`;
    if (src.includes('daily_history') && kis) return `서버 일봉 이력 (daily_history)${kis}`;
    if (src.includes('daily_history')) return '서버 일봉 이력 (daily_history)';
    if (src.includes('/ticks/')) return '서버 집계 (틱 파일)';
    if (src.endsWith('_10s.txt') || src.includes('_10s')) return '서버 집계 (1분봉, 10초 파일)';
    return `서버 집계 (${src.split('/').pop()})`;
  }

  private dataDir = (ymd: string) => `${this.root()}/runtime/data/${ymd}`;

  /** All trades of the day so far (null = no tick file). */
  private async loadTicks(date: string, code: string): Promise<Tick[] | null> {
    const path = `${this.root()}/runtime/data/ticks/${date}/${code}.csv`;
    const key = `${date}/${code}`;
    const fc = this.cached(date);
    if (fc) {
      const have = touch(this.ticks, key);
      if (have) return have.ticks;
      const buf = await fc.file(date, `ticks/${code}.csv`, path);
      if (!buf) return null;
      const c: TickCache = { off: buf.length, lb: new LineBuffer(), ticks: [] };
      let lines = c.lb.push(new Uint8Array(buf.buffer, buf.byteOffset, buf.length));
      lines = lines.concat(c.lb.push(new Uint8Array([0x0a]))); // a last line without a newline
      if (lines.length) {
        c.cols = tickHeader(lines[0]);
        for (const t of parseTickLines(lines.slice(1).filter(Boolean), c.cols)) c.ticks.push(t);
      }
      return remember(this.ticks, key, c, LIMIT.ticks).ticks;
    }
    const st = await this.host.stat(path);
    if (!st) {
      this.ticks.delete(key);
      return null;
    }
    let c = touch(this.ticks, key);
    if (!c || st.size < c.off) c = remember(this.ticks, key, { off: 0, lb: new LineBuffer(), ticks: [] }, LIMIT.ticks);
    while (c.off < st.size) {
      const end = Math.min(st.size, c.off + MAX_READ);
      const buf = await this.host.readRange(path, c.off, end);
      if (!buf.length) break;
      c.off += buf.length;
      let lines = c.lb.push(new Uint8Array(buf.buffer, buf.byteOffset, buf.length));
      if (!c.cols && lines.length) {
        c.cols = tickHeader(lines[0]);
        lines = lines.slice(1);
      }
      if (c.cols) for (const t of parseTickLines(lines, c.cols)) c.ticks.push(t);
    }
    return c.ticks;
  }

  /** file names in runtime/data/<date> that start with "<code>_" */
  private async filesFor(date: string, code: string, live: boolean): Promise<string[]> {
    const key = `${date}/${code}`;
    const hit = touch(this.files, key);
    if (hit && (!live || Date.now() - hit.at < 60000)) return hit.names;
    const ls = () => this.host.listPrefix(this.dataDir(date), `${code}_`);
    const fc = this.cached(date);
    const names = fc ? await fc.list(date, code, ls) : await ls();
    return remember(this.files, key, { at: Date.now(), names }, LIMIT.files).names;
  }

  /** real 1-minute bars from the 10-second file (null = no file) */
  private async loadMin1(date: string, code: string, live: boolean): Promise<Bar[] | null> {
    const name = (await this.filesFor(date, code, live)).find((n) => n.endsWith('_10s.txt'));
    if (!name) return null;
    const path = `${this.dataDir(date)}/${name}`;
    const fc = this.cached(date);
    if (fc) {
      const hit = touch(this.min1, path);
      if (hit) return hit.bars;
      const buf = await fc.file(date, `data/${name}`, path);
      return buf ? remember(this.min1, path, { size: buf.length, bars: parse10sMinuteBars(buf.toString('utf8')) }, LIMIT.min1).bars : null;
    }
    const st = await this.host.stat(path);
    if (!st) return null;
    const hit = touch(this.min1, path);
    if (hit && hit.size === st.size) return hit.bars;
    const text = (await this.host.readRange(path, 0, st.size)).toString('utf8');
    return remember(this.min1, path, { size: st.size, bars: parse10sMinuteBars(text) }, LIMIT.min1).bars;
  }

  /** YYYYMMDD folders under runtime/data (cached 10 min) */
  private async dataFolders(): Promise<string[]> {
    if (this.dirs && Date.now() - this.dirs.at < 600000) return this.dirs.list;
    const remote = this.host.kind === 'remote';
    if (!this.host.connected && this.cache && remote) {
      const saved = this.cache.loadFolders();
      if (saved.length) return saved; // offline: the last list seen
    }
    const list = (await this.host.readdir(`${this.root()}/runtime/data`))
      .filter((e) => e.isDir && /^\d{8}$/.test(e.name))
      .map((e) => e.name)
      .sort();
    this.dirs = { at: Date.now(), list };
    if (remote) this.cache?.saveFolders(list);
    return list;
  }

  private async loadDaily(date: string, code: string, live: boolean): Promise<{ bars: Bar[]; folder: string } | null> {
    const folders = (await this.dataFolders()).filter((d) => d <= date).reverse();
    // the day's own folder first, then up to 5 earlier ones (a symbol can be missing from a single day)
    for (const folder of folders.slice(0, 6)) {
      const name = (await this.filesFor(folder, code, live && folder === date)).find((n) => n.endsWith('_daily.csv'));
      if (!name) continue;
      const path = `${this.dataDir(folder)}/${name}`;
      const fc = this.cached(folder);
      if (fc) {
        const hit = touch(this.daily, path);
        if (hit) return { bars: hit.bars, folder };
        const buf = await fc.file(folder, `data/${name}`, path);
        if (!buf) continue;
        return { bars: remember(this.daily, path, { size: buf.length, bars: parseDailyCsv(buf.toString('utf8')) }, LIMIT.daily).bars, folder };
      }
      const st = await this.host.stat(path);
      if (!st) continue;
      const hit = touch(this.daily, path);
      if (hit && hit.size === st.size) return { bars: hit.bars, folder };
      const text = (await this.host.readRange(path, 0, st.size)).toString('utf8');
      return { bars: remember(this.daily, path, { size: st.size, bars: parseDailyCsv(text) }, LIMIT.daily).bars, folder };
    }
    return null;
  }

  /**
   * Download a past day's price files for `codes` (+ the previous trading day's warm-up files) into the local cache. One code per
   * queue slot so chart requests interleave. Files already cached are skipped without server access.
   */
  async prefetch(date: string, codes: string[], onProgress: (p: PrefetchProgress) => void, stopped: () => boolean): Promise<PrefetchProgress> {
    const fc = this.cached(date);
    const p: PrefetchProgress = { date, done: 0, total: codes.length, failed: 0 };
    if (!fc) return p;
    let prev: string | undefined;
    try {
      prev = (await this.dataFolders()).filter((d) => d < date).pop();
    } catch {
      /* no warm-up day */
    }
    onProgress({ ...p });
    for (const code of codes) {
      if (stopped()) break;
      const job = this.tail.then(async () => {
        const days = prev && this.isPast(prev) ? [date, prev] : [date];
        for (const d of days) {
          await fc.file(d, `ticks/${code}.csv`, `${this.root()}/runtime/data/ticks/${d}/${code}.csv`);
          const names = await this.filesFor(d, code, false);
          for (const n of names.filter((x) => x.endsWith('_10s.txt') || (d === date && x.endsWith('_daily.csv')))) {
            await fc.file(d, `data/${n}`, `${this.dataDir(d)}/${n}`);
          }
        }
      });
      this.tail = job.catch(() => undefined);
      try {
        await job;
      } catch (e) {
        p.failed++;
        p.error = String((e as Error)?.message || e).split('\n')[0];
      }
      p.done++;
      onProgress({ ...p });
      if (!this.host.connected) {
        p.error = p.error || '연결이 끊겨 중단했습니다.';
        break;
      }
    }
    return p;
  }

  get(req: PriceRequest, live: boolean): Promise<PriceResponse> {
    const job = this.tail.then(() => this.getNow(req, live));
    this.tail = job.catch(() => undefined);
    return job;
  }

  /** previous trading day's bars of the same size (indicator warm-up); tick bars need that day's tick file */
  private async warmup(date: string, code: string, kind: 'tick' | 'min', n: number): Promise<{ warm: Bar[]; warmBot: Bar[]; warmDate: string } | null> {
    const prev = (await this.dataFolders()).filter((d) => d < date).pop();
    if (!prev) return null;
    if (kind === 'tick') {
      const ticks = await this.loadTicks(prev, code);
      if (!ticks?.length) return { warm: [], warmBot: [], warmDate: prev }; // [2026-11-18] no warm-up, but the chart can still ask that day
      const all = tickBars(ticks, n);
      return { warm: all.slice(-WARMUP_BARS_MAX), warmBot: all.slice(-WARMUP_BARS), warmDate: prev };
    }
    // minute bars: like the bot (r012 load_prior_day_warmup) - the previous day's raw 1-minute bars of the 10-second file,
    // last PRIOR_DAY_WARMUP_MINUTES of them for the bot's indicators; the tick file is only a fallback source
    let m1 = (await this.loadMin1(prev, code, false)) || [];
    if (!m1.length) {
      const ticks = await this.loadTicks(prev, code);
      if (ticks?.length) m1 = minuteBarsFromTicks(ticks, 1);
    }
    if (!m1.length) return { warm: [], warmBot: [], warmDate: prev };
    return {
      warm: resampleMinuteBars(m1, n).slice(-WARMUP_BARS_MAX),
      warmBot: resampleMinuteBars(m1.slice(-BOT_PARAMS.warmupMinutes), n),
      warmDate: prev,
    };
  }

  private async withWarmup(r: PriceResponse, req: PriceRequest, kind: 'tick' | 'min', n: number): Promise<PriceResponse> {
    if (!r.ok || !req.warmup) return r;
    try {
      const w = await this.warmup(req.date, req.code, kind, n);
      return w ? { ...r, ...w } : r;
    } catch {
      return r; // warm-up is optional
    }
  }

  /** symbol name from "<code>_<name>_daily.csv" / "_10s.txt" of the day's folder (or the latest earlier one) */
  private async nameFor(date: string, code: string, live: boolean): Promise<string | undefined> {
    try {
      for (const folder of (await this.dataFolders()).filter((d) => d <= date).reverse().slice(0, 3)) {
        const f = (await this.filesFor(folder, code, live && folder === date)).find((x) => /_(daily\.csv|10s\.txt)$/.test(x));
        if (f) return f.slice(code.length + 1).replace(/_(daily\.csv|10s\.txt)$/, '');
      }
    } catch {
      /* the name is optional */
    }
    return undefined;
  }

  private async getNow(req: PriceRequest, live: boolean): Promise<PriceResponse> {
    const r = await this.getBars(req, live);
    if (r.ok && req.withName) r.name = await this.nameFor(req.date, req.code, live);
    return r;
  }

  private async getBars(req: PriceRequest, live: boolean): Promise<PriceResponse> {
    const { date, code, kind } = req;
    if (!/^\d{8}$/.test(date) || !/^[0-9A-Z]{6}$/.test(code)) return { ok: false, bars: [], error: '날짜/종목 코드 형식 오류' };
    const n = Math.max(1, Math.floor(Number(req.n) || 1));

    if (kind === 'tick' || kind === 'min') {
      const ticks = await this.loadTicks(date, code);
      // [2026-11-12] tick chart of an NXT symbol whose tick file has KRX pre-open prints only (08:30 - 09:00): not a tick chart
      // yet - the minute bars below (combined market) are shown instead
      const preOnly = kind === 'tick' && !!req.nxt && !!ticks?.length && !ticks.some((t) => kstMinuteOfDay(t.t) >= 9 * 60);
      if (ticks && ticks.length && !preOnly) {
        // [2026-11-11] minutes the tick collector missed (g011 outage, a code subscribed late): the minute chart fills them with
        // the server's KIS minute bars; tick bars cannot be filled (KIS keeps no tick history) and only say so
        const m1 = minuteBarsFromTicks(ticks, 1);
        const gaps = minuteGaps(m1, date, live ? Date.now() : Infinity);
        const quote = quoteFromTicks(ticks);
        if (kind === 'tick') {
          const note = gaps.length ? `틱 수집이 끊긴 ${gapText(gaps)} (${gapMinutes(gaps)}분)은 틱봉에 없습니다 - 분봉은 KIS 분봉으로 채워 보여 줍니다` : undefined;
          return this.withWarmup({ ok: true, source: 'ticks', bars: tickBars(ticks, n), quote, gaps, note }, req, kind, n);
        }
        let bars = resampleMinuteBars(m1, n);
        let note: string | undefined;
        let filled = 0;
        let base = m1;
        if (gaps.length) {
          const fill = await this.kisMinuteFill(date, code, 'gap', gaps);
          if (fill.bars?.length) {
            const merged = mergeMinuteBars(fill.bars, m1);
            filled = Math.max(0, merged.length - m1.length);
            base = merged;
            bars = resampleMinuteBars(merged, n);
            note = `틱 수집이 끊긴 ${gapText(gaps)} (${gapMinutes(gaps)}분)을 KIS 분봉으로 채웠습니다 - 봇은 이 구간의 틱을 보지 못했습니다`;
          } else note = `틱 수집이 끊긴 ${gapText(gaps)} (${gapMinutes(gaps)}분) - KIS 분봉으로 채우지 못함: ${fill.error || '분봉 없음'}`;
        }
        // [2026-11-19] every symbol: the minutes outside the KRX regular session (NXT pre / after market, KRX after-hours market)
        const ext = await this.withOutside(date, code, req, live, base);
        if (ext.added) {
          bars = resampleMinuteBars(ext.bars, n);
          note = [note, ext.note].filter(Boolean).join(' · ');
        }
        return this.withWarmup({ ok: true, source: 'ticks', bars, quote, gaps, filled, note, ...sessionInfo(ext) }, req, kind, n);
      }
      const m1 = kind === 'min' ? await this.loadMin1(date, code, live) : null;
      if (m1 && m1.length) {
        const ext = await this.withOutside(date, code, req, live, m1);
        const note = ['틱 파일이 없어 서버의 1분봉(10초 파일)으로 그렸습니다.', ext.added ? ext.note : undefined].filter(Boolean).join(' · ');
        return this.withWarmup({ ok: true, source: 'min1', bars: resampleMinuteBars(ext.bars, n), note, ...sessionInfo(ext) }, req, kind, n);
      }
      // no file for this code/day (not collected, or today's minutes before g001 runs) -> server `bars` (KIS once the server adds it)
      // [2026-11-12] tick chart without a tick file (before the KRX open the collector has nothing; a code that is not collected):
      // the day's 1-minute bars are drawn instead of an error
      const tickAsMin = kind === 'tick';
      const sizeArg = tickAsMin ? ['--interval', '1'] : ['--interval', String(n)];
      // [2026-11-18] a past day's answer is kept on disk (asked once)
      const fcMin = this.cached(date);
      // [2026-11-19] .v2: answers saved before end at the KRX close (no after-hours market)
      const relMin = `server_min/${code}_${tickAsMin ? 1 : n}${req.warmup ? '_w' : ''}.v2.json`;
      const savedMin = fcMin?.json<ServerBars>(date, relMin);
      let sb: ServerBarsResult = savedMin ? { ok: true, data: savedMin } : null;
      if (!sb) {
        sb = await this.serverBars(['--date', date, '--code', code, '--kind', 'min', ...sizeArg, ...(req.warmup ? ['--warmup', String(WARMUP_BARS_MAX)] : [])]);
        if (fcMin && sb?.ok && sb.data.bars?.length) fcMin.saveJson(date, relMin, sb.data);
      }
      if (sb?.ok) {
        const d = sb.data;
        const warm = serverRowsToBars(d.columns, d.warm || undefined);
        const botN = tickAsMin ? BOT_PARAMS.warmupMinutes : Math.ceil(BOT_PARAMS.warmupMinutes / n);
        if (typeof d.nxt === 'boolean') remember(this.nxtOf, code, d.nxt, LIMIT.files);
        if (d.hours) remember(this.hoursOf, date, d.hours, LIMIT.daily);
        return {
          ...sessionInfo({ hours: d.hours || undefined, nxt: d.nxt }),
          ok: true,
          source: 'server',
          bars: serverRowsToBars(d.columns, d.bars),
          warm: warm.length ? warm : undefined,
          warmBot: warm.length ? warm.slice(-botN) : undefined,
          warmDate: d.warm_date || undefined,
          name: d.name,
          sourceDetail: d.source,
          note: tickAsMin ? `틱 데이터가 없어 1분봉으로 표시합니다 (틱 수집기는 KRX 체결만 수집 - 09:00 이전에는 틱이 없습니다) · ${PriceService.sourceNote(d.source)}` : PriceService.sourceNote(d.source),
        };
      }
      const base =
        kind === 'tick'
          ? '이 날짜·종목의 틱 파일이 없습니다 (틱은 감시·후보 종목만 수집). 분봉으로 보세요.'
          : '이 날짜·종목의 분봉 데이터가 서버에 없습니다 (수집 대상이 아니거나, 오늘 분봉은 장 마감 후 생성).';
      return { ok: false, bars: [], error: sb ? `${base} - 서버: ${sb.error}` : base };
    }

    // daily: server `bars --kind day` first (daily_history ~14 months, KIS fallback on the server), then the day's folder file
    let daily: Bar[] | null = null;
    let note: string | undefined;
    let folder: string | undefined;
    let name: string | undefined;
    let sourceDetail: string | undefined;
    const fcDay = this.cached(date);
    let sb: ServerBarsResult = null;
    const savedDay = fcDay?.json<ServerBars>(date, `server_day/${code}.v3.json`);
    if (savedDay) sb = { ok: true, data: savedDay };
    else {
      sb = await this.serverBars(['--date', date, '--code', code, '--kind', 'day']);
      if (fcDay && sb?.ok) fcDay.saveJson(date, `server_day/${code}.v3.json`, sb.data);
    }
    if (sb?.ok) {
      daily = serverRowsToBars(sb.data.columns, sb.data.bars);
      name = sb.data.name;
      sourceDetail = sb.data.source;
      note = PriceService.sourceNote(sb.data.source);
    }
    if (!daily?.length) {
      const d = await this.loadDaily(date, code, live);
      if (!d) return { ok: false, bars: [], error: `일봉 데이터가 없습니다 (서버에 수집되지 않은 종목)${sb && !sb.ok ? ` - 서버: ${sb.error}` : ''}` };
      daily = d.bars;
      folder = d.folder;
      note = undefined;
    }
    return this.finishDaily(daily, req, live, { note, folder, name, sourceDetail });
  }

  private async finishDaily(all: Bar[], req: PriceRequest, live: boolean, extra: Partial<PriceResponse>): Promise<PriceResponse> {
    const { date, code, kind } = req;

    const dayT = ymdEpoch(date);
    let daily = all.filter((b) => b.t <= dayT);
    let note = extra.note;
    if (!daily.some((b) => b.t === dayT)) {
      // the daily file has no row for the day yet (live day / older folder) -> the day's bar from intraday data
      let intraday: Bar[] = [];
      try {
        const ticks = await this.loadTicks(date, code);
        intraday = ticks?.length ? minuteBarsFromTicks(ticks, 1) : (await this.loadMin1(date, code, live)) || [];
      } catch {
        /* intraday data is optional here */
      }
      daily = dailyWithToday(daily, date, intraday);
      if (daily.some((b) => b.t === dayT)) note = [note, '당일 봉은 장중 데이터로 만들었습니다.'].filter(Boolean).join(' · ');
    }
    const bars = groupDaily(daily, kind); // [2026-11-18] + year
    return { ok: true, source: extra.sourceDetail ? 'server' : 'daily', bars, ...extra, note };
  }
}
