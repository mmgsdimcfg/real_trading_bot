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
 * - [2026-11-07] type=feat - Past days read through PriceFileCache (priceCache.ts): tick / 10-second / daily files and the "<code>_*"
 *   lists are downloaded whole once into %APPDATA%/autoboth/pricecache/<date>/ and read locally after that - also offline. The
 *   server's daily answer (r016 bars --kind day) of a past day is kept there too. prefetch(date, codes) downloads the files of
 *   the day's symbols and the previous trading day's warm-up files in the background (one code per queue slot, so chart
 *   requests are not held up for the whole download). The live day keeps the incremental SFTP reads
 */

import { LineBuffer } from '../shared/lineBuffer';
import { CHART_CACHE_MS, isMarketHours, kstNowIso } from '../shared/control';
import {
  dailyWithToday,
  minuteBarsFromTicks,
  monthBars,
  parse10sMinuteBars,
  parseDailyCsv,
  parseTickLines,
  resampleMinuteBars,
  tickBars,
  tickHeader,
  weekBars,
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
} from '../shared/price';
import { shellQuote, type SshConnection } from './ssh';
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
const MAX_ENTRIES = 12;

interface TickCache {
  off: number;
  lb: LineBuffer;
  cols?: Map<string, number>;
  ticks: Tick[];
}

/** keep at most MAX_ENTRIES (oldest inserted dropped) */
function remember<K, V>(m: Map<K, V>, k: K, v: V): V {
  m.delete(k);
  m.set(k, v);
  while (m.size > MAX_ENTRIES) m.delete(m.keys().next().value as K);
  return v;
}

/** r016 `bars` result: null = unavailable (old server / no client) */
type ServerBarsResult = { ok: true; data: ServerBars } | { ok: false; error: string } | null;

export class PriceService {
  private ticks = new Map<string, TickCache>();
  private files = new Map<string, { at: number; names: string[] }>();
  private min1 = new Map<string, { size: number; bars: Bar[] }>();
  private daily = new Map<string, { size: number; bars: Bar[] }>();
  private dirs: { at: number; list: string[] } | null = null;
  /** r016 `bars` answers reused for a while (each call = Python on the server, maybe KIS): key = args joined */
  private serverCache = new Map<string, { at: number; ttl: number; res: ServerBarsResult }>();
  /** requests run one at a time: two concurrent readers of the same incremental tick cache would read the same byte range twice */
  private tail: Promise<unknown> = Promise.resolve();

  /** the server has no `bars` command (older r016) - stop asking until reconnect */
  private noServerBars = false;

  constructor(
    private ssh: SshConnection,
    private root: () => string,
    private r016: () => R016Client | null = () => null,
    /** local copies of past days' files; null = always read the server */
    private cache: PriceFileCache | null = null,
    /** true for a day before the server's today (its files no longer change) */
    private isPast: (date: string) => boolean = () => false,
  ) {}

  private cached(date: string): PriceFileCache | null {
    return this.cache && this.isPast(date) ? this.cache : null;
  }

  resetServerSupport(): void {
    this.noServerBars = false;
    this.serverCache.clear();
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
      const ttl = !res.ok ? CHART_CACHE_MS.failed : isDaily ? CHART_CACHE_MS.daily : isMarketHours(kstNowIso()) ? 0 : CHART_CACHE_MS.intradayIdle;
      if (ttl > 0) remember(this.serverCache, key, { at: Date.now(), ttl, res });
    }
    return res;
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
      const have = this.ticks.get(key);
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
      return remember(this.ticks, key, c).ticks;
    }
    const st = await this.ssh.stat(path);
    if (!st) {
      this.ticks.delete(key);
      return null;
    }
    let c = this.ticks.get(key);
    if (!c || st.size < c.off) c = remember(this.ticks, key, { off: 0, lb: new LineBuffer(), ticks: [] });
    while (c.off < st.size) {
      const end = Math.min(st.size, c.off + MAX_READ);
      const buf = await this.ssh.readRange(path, c.off, end);
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
    const hit = this.files.get(key);
    if (hit && (!live || Date.now() - hit.at < 60000)) return hit.names;
    const ls = async () => {
      const { out } = await this.ssh.exec(`cd ${shellQuote(this.dataDir(date))} 2>/dev/null && ls -1 -- ${code}_* 2>/dev/null`, 20000);
      return out.split('\n').map((s) => s.trim()).filter(Boolean);
    };
    const fc = this.cached(date);
    const names = fc ? await fc.list(date, code, ls) : await ls();
    return remember(this.files, key, { at: Date.now(), names }).names;
  }

  /** real 1-minute bars from the 10-second file (null = no file) */
  private async loadMin1(date: string, code: string, live: boolean): Promise<Bar[] | null> {
    const name = (await this.filesFor(date, code, live)).find((n) => n.endsWith('_10s.txt'));
    if (!name) return null;
    const path = `${this.dataDir(date)}/${name}`;
    const fc = this.cached(date);
    if (fc) {
      const hit = this.min1.get(path);
      if (hit) return hit.bars;
      const buf = await fc.file(date, `data/${name}`, path);
      return buf ? remember(this.min1, path, { size: buf.length, bars: parse10sMinuteBars(buf.toString('utf8')) }).bars : null;
    }
    const st = await this.ssh.stat(path);
    if (!st) return null;
    const hit = this.min1.get(path);
    if (hit && hit.size === st.size) return hit.bars;
    const text = (await this.ssh.readRange(path, 0, st.size)).toString('utf8');
    return remember(this.min1, path, { size: st.size, bars: parse10sMinuteBars(text) }).bars;
  }

  /** YYYYMMDD folders under runtime/data (cached 10 min) */
  private async dataFolders(): Promise<string[]> {
    if (this.dirs && Date.now() - this.dirs.at < 600000) return this.dirs.list;
    if (!this.ssh.connected && this.cache) {
      const saved = this.cache.loadFolders();
      if (saved.length) return saved; // offline: the last list seen
    }
    const list = (await this.ssh.readdir(`${this.root()}/runtime/data`))
      .filter((e) => e.isDir && /^\d{8}$/.test(e.name))
      .map((e) => e.name)
      .sort();
    this.dirs = { at: Date.now(), list };
    this.cache?.saveFolders(list);
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
        const hit = this.daily.get(path);
        if (hit) return { bars: hit.bars, folder };
        const buf = await fc.file(folder, `data/${name}`, path);
        if (!buf) continue;
        return { bars: remember(this.daily, path, { size: buf.length, bars: parseDailyCsv(buf.toString('utf8')) }).bars, folder };
      }
      const st = await this.ssh.stat(path);
      if (!st) continue;
      const hit = this.daily.get(path);
      if (hit && hit.size === st.size) return { bars: hit.bars, folder };
      const text = (await this.ssh.readRange(path, 0, st.size)).toString('utf8');
      return { bars: remember(this.daily, path, { size: st.size, bars: parseDailyCsv(text) }).bars, folder };
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
      if (!this.ssh.connected) {
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
      if (!ticks?.length) return null;
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
    if (!m1.length) return null;
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
      if (ticks && ticks.length) {
        const bars = kind === 'tick' ? tickBars(ticks, n) : minuteBarsFromTicks(ticks, n);
        return this.withWarmup({ ok: true, source: 'ticks', bars, quote: quoteFromTicks(ticks) }, req, kind, n);
      }
      const m1 = kind === 'min' ? await this.loadMin1(date, code, live) : null;
      if (m1 && m1.length) {
        return this.withWarmup({ ok: true, source: 'min1', bars: resampleMinuteBars(m1, n), note: '틱 파일이 없어 서버의 1분봉(10초 파일)으로 그렸습니다.' }, req, kind, n);
      }
      // no file for this code/day (not collected, or today's minutes before g001 runs) -> server `bars` (KIS once the server adds it)
      const sizeArg = kind === 'tick' ? ['--ticks', String(n)] : ['--interval', String(n)];
      const sb = await this.serverBars(['--date', date, '--code', code, '--kind', kind, ...sizeArg, ...(req.warmup ? ['--warmup', String(WARMUP_BARS_MAX)] : [])]);
      if (sb?.ok) {
        const d = sb.data;
        const warm = serverRowsToBars(d.columns, d.warm || undefined);
        const botN = kind === 'tick' ? WARMUP_BARS : Math.ceil(BOT_PARAMS.warmupMinutes / n);
        return {
          ok: true,
          source: 'server',
          bars: serverRowsToBars(d.columns, d.bars),
          warm: warm.length ? warm : undefined,
          warmBot: warm.length ? warm.slice(-botN) : undefined,
          warmDate: d.warm_date || undefined,
          name: d.name,
          sourceDetail: d.source,
          note: PriceService.sourceNote(d.source),
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
    const savedDay = fcDay?.json<ServerBars>(date, `server_day/${code}.json`);
    if (savedDay) sb = { ok: true, data: savedDay };
    else {
      sb = await this.serverBars(['--date', date, '--code', code, '--kind', 'day']);
      if (fcDay && sb?.ok) fcDay.saveJson(date, `server_day/${code}.json`, sb.data);
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
    const bars = kind === 'day' ? daily : kind === 'week' ? weekBars(daily) : monthBars(daily);
    return { ok: true, source: extra.sourceDetail ? 'server' : 'daily', bars, ...extra, note };
  }
}
