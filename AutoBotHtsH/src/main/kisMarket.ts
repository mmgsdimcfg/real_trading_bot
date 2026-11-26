// Update log
// - [2026-11-06] type=feat (MarketBackend on the KIS REST API (inquiry TRs only, through KisClient): quote, order book + expected price,
//   latest trades, minute charts (today: inquire_time_itemchartprice 30 bars/call paged back to 09:00; earlier days: inquire_time_dailychartprice
//   120 bars/call, real accounts only, up to 1 year), day / week / month / year charts (inquire_daily_itemchartprice 100 rows/call, adjusted
//   prices), index quote + chart, daily prices with foreign net buying, investor net buying, stock info (search_stock_info, labels from the
//   official repo's COLUMN_MAPPING), multi-symbol prices (30 per call), rankings (volume / amount / rise / fall / market cap) and news titles.
//   Charts are cached per code+kind; refresh=true fetches only the newest page and merges)
// - [2026-11-06] type=fix (Minute charts stopped at 15:20 and never showed NXT trading. NXT-tradable symbols (search_stock_info
//   cptt_trad_tr_psbl_yn = Y, not NXT-halted) now load minute bars with market code UN (KRX + NXT combined, 08:00 - 20:00) for today and
//   past days; other symbols keep J (KRX) with the first page asked from 16:00 so the 15:30 closing-auction bar is included. UN falls back
//   to J when it errors or returns nothing (e.g. paper server))
// - [2026-11-06] type=fix (isNxt() is public - the main process routes realtime trades / order book of NXT symbols to the combined TRs;
//   the REST trade list of NXT symbols uses UN too (falls back to J))
// - [2026-11-06] type=feat (ticks(): today's individual trades for tick charts (inquire_time_itemconclusion, 30 per call, paged backwards by
//   time, de-duplicated by accumulated volume; a second with more than 30 trades is stepped over and counted in `skipped`). tradeMarket()
//   picks UN only on the real server, matching the realtime routing, so REST and realtime accumulated volumes share one basis)
// - [2026-11-06] type=fix (Minute charts of N days showed N-1 days: the past query started at today and kept the newest N-1 dates, today
//   included. pastMinutes() now leaves out `skipDate` (today, already loaded) when counting and keeping dates, and gets one more day of
//   pages to read through it)

// - [2026-11-09] type=fix (Day / week / month / year charts of stocks load with market code UN on the real server (falls back to J on an
//   error or no rows). KIS J daily rows from 2026-09-14 on carry closes that differ from the real KRX close with the same volume (GS E&C
//   11/06 29,750 vs 29,800, 3S 1,150 vs 1,139), which moved MA / MACD / RSI / ADX away from the MTS; UN matches the exchange for
//   non-NXT symbols and is the combined KRX + NXT series (the MTS default "통합") for NXT symbols. The refresh uses the same code; a UN
//   refresh that errors or comes back empty reloads the whole history with J instead of keeping stale bars or mixing the two series)
// - [2026-11-12] type=fix (Prices stood still in NXT hours: the watchlist (intstock_multprice) and the current price / order book asked
//   KRX only (J). Watchlist: NXT symbols (stock master NXT flag) are asked with NX while KRX is closed (before 09:00, from 15:40) -
//   the TR takes a market code per symbol; a symbol NX has no price for falls back to its KRX row. Current price / order book of NXT
//   symbols on the real server use UN (KRX + NXT combined, like the charts and realtime), falling back to J)
// - [2026-11-19] type=perf (Charts open without waiting (user 2026-11-19): chart data is kept on disk (ChartStore, chartcache/) as well
//   as in memory. A request with a cached chart is answered at once (stale = true when it is older than the reuse window, and the
//   window then asks for a refresh right away). Refreshes fetch only what is missing: day+ charts the newest page (a full reload when
//   the cache is older than that page reaches or the adjusted prices of older bars changed, e.g. a split); minute charts today's
//   minutes back to the last cached bar of today and past days only for the dates not cached complete (a past day's minutes never
//   change) - before, every open past 30 s / 60 s paged the whole history again (minute charts: up to 26 + 7 pages per day))
// - [2026-11-20] type=feat (investorDaily(): about one year of daily investor net buying (개인 / 외국인 / 기관계, shares) from
//   investor_trade_by_stock_daily for the chart pane 투자자 누적. Paged backwards by date; kept in memory and on disk (inv_<code>), a
//   refresh asks only the newest page(s) until it reaches the cached rows (past days do not change); reused for INV_REUSE_MS; on an error
//   the cached rows are returned)
// - [2026-11-20] type=fix (The 투자자 누적 pane stayed empty: KIS answers this TR for today's date only after 15:40 - before that it returns
//   rt_cd 2 "TIME LIMIT 00:00 ~ 15:40" (OPSQ2001), so the first page failed and nothing was kept. Before 15:40 the first page is asked
//   from yesterday (and a TIME LIMIT answer steps one day back). 30 rows per call (checked 2026-11-20). Real server: market UN (KRX + NXT
//   combined, like the day charts), J when UN fails / is empty; the cache keeps the market code)

import { KisClient, type KisError } from './kisClient';
import type { ChartStore } from './chartCache';
import {
  dailyRowsToBars,
  fieldLabel,
  minuteRowsToBars,
  multPriceParams,
  nxtOnlyHours,
  num,
  signed,
  toOrderBook,
  toConclusionTrades,
  toQuote,
  toRankRows,
  toTrades,
  toWatchRows,
  type TradeTick,
  type WatchRow,
} from '../shared/kis';
import { resampleMinuteBars, weekBars, monthBars, yearBars, type Bar } from '../shared/price';
import type { ChartReq, ChartRes, TicksReq, TicksRes, DailyRow, IndexQuote, InfoField, InvestorDay, InvestorRow, MarketBackend, NewsRow, NqFutures, RankReq } from '../shared/market';
import { NQ_FUTURES, toNqFutures } from '../shared/market';
import { INDEXES } from '../shared/market';

const KST = 9 * 3600000;
const kstNow = () => new Date(Date.now() + KST);
const ymd = (d: Date) => d.toISOString().slice(0, 10).replace(/-/g, '');
const hms = (d: Date) => d.toISOString().slice(11, 19).replace(/:/g, '');
const rowsOf = (v: unknown): Record<string, string>[] => (Array.isArray(v) ? (v as Record<string, string>[]) : v && typeof v === 'object' ? [v as Record<string, string>] : []);
/** "HHMMSS" one second earlier ("000000" stays) */
const prevSecond = (h: string) => {
  const s = Math.max(0, +h.slice(0, 2) * 3600 + +h.slice(2, 4) * 60 + +h.slice(4, 6) - 1);
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map((v) => String(v).padStart(2, '0')).join('');
};
const objOf = (v: unknown): Record<string, string> => (Array.isArray(v) ? (v[0] as Record<string, string>) || {} : ((v as Record<string, string>) || {}));

interface ChartCache {
  bars: Bar[];
  at: number;
  name?: string;
  days: number;
  /** market code the bars were loaded with (minute charts, and day+ charts of stocks) */
  mk?: MinMarket;
  /** [2026-11-19] minute charts: past dates (YYYYMMDD) whose minutes are all in `bars` */
  done?: string[];
}
/** [2026-11-19] adjusted closes of older bars moved more than this (split / rights issue): reload the whole history */
const ADJ_TOL = 0.005;
const dayOf = (t: number) => ymd(new Date(t + KST));
const hmsOf = (t: number) => hms(new Date(t + KST));
const mergeBars = (...lists: Bar[][]): Bar[] => {
  const map = new Map<number, Bar>();
  for (const l of lists) for (const b of l) map.set(b.t, b);
  return [...map.values()].sort((a, b) => a.t - b.t);
};

/** [2026-11-20] investor net buying: reuse window, history kept (days back from today), page limit of one load */
const INV_REUSE_MS = 5 * 60000;
const INV_DAYS = 372;
const INV_PAGES = 15;
/** [2026-11-23] rows carry close / change / amounts from this cache version on; older saved rows are loaded again */
const INV_V = 2;
interface InvCache {
  rows: InvestorDay[];
  at: number;
  /** the rows are complete from this date (YYYYMMDD) on */
  from: string;
  mk?: 'J' | 'UN';
  v?: number;
}
const prevYmd = (d: string) => ymd(new Date(Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8)) - 86400000));

/** minute chart market code: J = KRX only, UN = KRX + NXT combined */
type MinMarket = 'J' | 'UN';
/** first page time (asked a little after the last bar; a future time returns up to now) / earliest bar of a session */
const SESSION: Record<MinMarket, { open: string; close: string; pagesPerDay: number }> = {
  // KRX 09:00 - 15:30 (closing auction 15:20 - 15:30, its bar is stamped 15:30)
  J: { open: '090000', close: '160000', pagesPerDay: 4 },
  // NXT pre-market 08:00 - 08:50, main 09:00 - 15:20, after-market 15:30 - 20:00 (+ KRX in between)
  UN: { open: '080000', close: '200000', pagesPerDay: 7 },
};

export class KisMarket implements MarketBackend {
  readonly kind = 'kis' as const;
  private charts = new Map<string, ChartCache>();
  /** NXT tradability per code (search_stock_info), kept for the session */
  private nxt = new Map<string, Promise<boolean>>();

  /** codes whose NX row came back without a price -> asked with J until this time (epoch ms) */
  private nxSkip = new Map<string, number>();

  /** `nxtHint`: NXT flag of the stock master (true / false; undefined = not known) - no API call; `disk`: chart data kept across runs */
  constructor(
    readonly client: KisClient,
    private nxtHint: (code: string) => boolean | undefined = () => undefined,
    private disk: ChartStore | null = null,
  ) {}

  /** [2026-11-19] cached chart of `key`: memory, else the disk copy (per server: real / paper) */
  private cachedChart(key: string): ChartCache | undefined {
    let hit = this.charts.get(key);
    if (!hit && this.disk) {
      const d = this.disk.get<ChartCache>(`${this.client.env}_${key}`);
      if (d && Array.isArray(d.bars) && d.bars.length) this.charts.set(key, (hit = d));
    }
    return hit;
  }

  private keepChart(key: string, c: ChartCache): void {
    this.charts.set(key, c);
    this.disk?.put(`${this.client.env}_${key}`, c);
  }

  async quote(code: string) {
    if ((await this.tradeMarket(code)) === 'UN') {
      try {
        const r = await this.client.get('inquire_price', { FID_COND_MRKT_DIV_CODE: 'UN', FID_INPUT_ISCD: code });
        const q = toQuote(code, objOf(r.output));
        if (q.price > 0) return q;
      } catch {
        /* combined market not served -> KRX */
      }
    }
    const r = await this.client.get('inquire_price', { FID_COND_MRKT_DIV_CODE: 'J', FID_INPUT_ISCD: code });
    return toQuote(code, objOf(r.output));
  }

  async orderbook(code: string) {
    if ((await this.tradeMarket(code)) === 'UN') {
      try {
        const r = await this.client.get('inquire_asking_price_exp_ccn', { FID_COND_MRKT_DIV_CODE: 'UN', FID_INPUT_ISCD: code });
        const ob = toOrderBook(objOf(r.output1), objOf(r.output2));
        if (ob.asks.length || ob.bids.length) return ob;
      } catch {
        /* combined market not served -> KRX */
      }
    }
    const r = await this.client.get('inquire_asking_price_exp_ccn', { FID_COND_MRKT_DIV_CODE: 'J', FID_INPUT_ISCD: code });
    return toOrderBook(objOf(r.output1), objOf(r.output2));
  }

  async trades(code: string) {
    if ((await this.tradeMarket(code)) === 'UN') {
      try {
        const r = await this.client.get('inquire_ccnl', { FID_COND_MRKT_DIV_CODE: 'UN', FID_INPUT_ISCD: code });
        const rows = rowsOf(r.output);
        if (rows.length) return toTrades(rows);
      } catch {
        /* combined market not served -> KRX */
      }
    }
    const r = await this.client.get('inquire_ccnl', { FID_COND_MRKT_DIV_CODE: 'J', FID_INPUT_ISCD: code });
    return toTrades(rowsOf(r.output));
  }

  /** market code for trades / ticks: UN (KRX + NXT) for NXT symbols on the real server - the same rule as the realtime routing in
   *  main/index.ts, so REST and realtime accumulated volumes (the trade identity) are on the same basis */
  async tradeMarket(code: string): Promise<'J' | 'UN'> {
    return this.client.env === 'real' && (await this.isNxt(code)) ? 'UN' : 'J';
  }

  /**
   * today's individual trades, newest first, paged backwards by time (inquire_time_itemconclusion, 30 per call, trades at or before
   * FID_INPUT_HOUR_1). Pages overlap at the boundary second and are de-duplicated by accumulated volume; when a whole page is one second
   * (more than 30 trades in it) the rest of that second cannot be reached and the next page starts one second earlier.
   */
  async ticks(req: TicksReq): Promise<TicksRes> {
    const mk = await this.tradeMarket(req.code);
    const { open, close } = SESSION[mk];
    const now = hms(kstNow());
    let hour = req.cursor?.hour || (now > close ? close : now);
    let minAcml = req.cursor?.acml ?? Infinity;
    const pages = Math.max(1, Math.min(20, req.pages || 1));
    const out: TradeTick[] = [];
    let skipped = 0;
    let next: TicksRes['next'] = null;
    for (let p = 0; p < pages; p++) {
      const r = await this.client.get('inquire_time_itemconclusion', { FID_COND_MRKT_DIV_CODE: mk, FID_INPUT_ISCD: req.code, FID_INPUT_HOUR_1: hour });
      const page = toConclusionTrades(rowsOf(r.output2));
      if (!page.length) break;
      for (const t of page) if ((t.acml as number) < minAcml) out.push(t);
      minAcml = Math.min(minAcml, ...page.map((t) => t.acml as number));
      const earliest = page[page.length - 1].time;
      let nextHour = earliest;
      if (earliest >= hour && page.length >= 30) {
        // a full page of the asked second only (more than 30 trades in it): step to the previous second
        nextHour = prevSecond(hour);
        skipped++;
      }
      next = page.length < 30 || nextHour < open ? null : { hour: nextHour, acml: minAcml };
      if (!next || (req.until !== undefined && minAcml <= req.until)) break;
      hour = nextHour;
    }
    return { trades: out, next, skipped };
  }

  // ---- charts ----

  /** true when the symbol also trades on NXT (and is not halted there); errors -> false and retried next time */
  isNxt(code: string): Promise<boolean> {
    // the stock master says it is not an NXT symbol: no API call needed
    if (this.nxtHint(code) === false) return Promise.resolve(false);
    let p = this.nxt.get(code);
    if (!p) {
      p = this.client
        .get('search_stock_info', { PRDT_TYPE_CD: '300', PDNO: code })
        .then((r) => {
          const o = objOf(r.output);
          return o.cptt_trad_tr_psbl_yn === 'Y' && o.nxt_tr_stop_yn !== 'Y';
        })
        .catch(() => {
          this.nxt.delete(code);
          return false;
        });
      this.nxt.set(code, p);
    }
    return p;
  }

  /** today's 1-minute bars, newest page first; `pages` = how far back (30 bars per page); `stopAt` = HHMMSS of the newest bar already held */
  private async todayMinutes(code: string, pages: number, mk: MinMarket = 'J', stopAt = ''): Promise<{ bars: Bar[]; name?: string }> {
    const now = kstNow();
    const { open, close } = SESSION[mk];
    let hour = hms(now) > close ? close : hms(now);
    const today = ymd(now);
    const rows: Record<string, string>[] = [];
    let name: string | undefined;
    for (let p = 0; p < pages; p++) {
      const r = await this.client.get('inquire_time_itemchartprice', {
        FID_ETC_CLS_CODE: '',
        FID_COND_MRKT_DIV_CODE: mk,
        FID_INPUT_ISCD: code,
        FID_INPUT_HOUR_1: hour,
        FID_PW_DATA_INCU_YN: 'N',
      });
      name = name || objOf(r.output1).hts_kor_isnm;
      const page = rowsOf(r.output2).filter((o) => o.stck_bsop_date === today);
      if (!page.length) break;
      rows.push(...page);
      const earliest = page.reduce((a, o) => (o.stck_cntg_hour < a ? o.stck_cntg_hour : a), page[0].stck_cntg_hour);
      if (earliest <= open || (stopAt && earliest <= stopAt)) break;
      // next page ends one minute before the earliest bar received
      const h = +earliest.slice(0, 2);
      const m = +earliest.slice(2, 4) - 1;
      hour = `${String(m < 0 ? h - 1 : h).padStart(2, '0')}${String((m + 60) % 60).padStart(2, '0')}00`;
    }
    return { bars: minuteRowsToBars(rows), name };
  }

  /** earlier days' 1-minute bars (real accounts), paged backwards from (date, hour) until `days` trading days are covered */
  private async pastMinutes(code: string, fromDate: string, days: number, mk: MinMarket = 'J', skipDate = '', known: Map<string, Bar[]> = new Map()): Promise<{ bars: Bar[]; dates: string[]; done: string[] }> {
    if (this.client.env !== 'real' || days <= 0) return { bars: [], dates: [], done: [] };
    const { open, close, pagesPerDay } = SESSION[mk];
    const rows: Record<string, string>[] = [];
    const dates = new Set<string>();
    /** [2026-11-19] complete dates taken from the cache instead of the server */
    const fromCache: Bar[] = [];
    let date = fromDate;
    let hour = close;
    const prevDay = (d: string) => ymd(new Date(Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8)) - 86400000));
    // [2026-11-19] start before `skipDate` (today, loaded by todayMinutes) instead of paging through it (up to 7 calls per refresh);
    // a holiday date returns the trading day before it, like the day change below
    if (skipDate && date === skipDate) date = prevDay(date);
    /** the oldest kept date is complete (paged past it, or cached complete) */
    let complete = false;
    for (let p = 0; p < (days + (skipDate ? 1 : 0)) * pagesPerDay + 2; p++) {
      if (date !== skipDate && known.has(date)) {
        // [2026-11-19] this date and the cached ones before it are complete: no more server pages for them
        const used = [...known.keys()].filter((d) => d <= date && !fromCache.some((b) => dayOf(b.t) === d));
        for (const d of used) (fromCache.push(...known.get(d)!), dates.add(d));
        if (dates.size >= days) {
          complete = true;
          break;
        }
        const oldest = [...known.keys()].sort()[0];
        date = prevDay(oldest);
        hour = close;
        known = new Map([...known].filter(([d]) => d < oldest));
        continue;
      }
      const r = await this.client.get('inquire_time_dailychartprice', {
        FID_COND_MRKT_DIV_CODE: mk,
        FID_INPUT_ISCD: code,
        FID_INPUT_HOUR_1: hour,
        FID_INPUT_DATE_1: date,
        FID_PW_DATA_INCU_YN: 'Y',
        FID_FAKE_TICK_INCU_YN: '',
      });
      const page = rowsOf(r.output2).filter((o) => o.stck_bsop_date && o.stck_cntg_hour);
      if (!page.length) break;
      // paging follows the whole page; rows of `skipDate` (loaded by todayMinutes) are not kept or counted
      for (const o of page) {
        if (o.stck_bsop_date === skipDate) continue;
        rows.push(o);
        dates.add(o.stck_bsop_date);
      }
      const last = page.reduce((a, o) => (o.stck_bsop_date + o.stck_cntg_hour < a.stck_bsop_date + a.stck_cntg_hour ? o : a), page[0]);
      if (dates.size > days) {
        complete = true;
        break;
      }
      date = last.stck_bsop_date;
      const h = +last.stck_cntg_hour.slice(0, 2);
      const m = +last.stck_cntg_hour.slice(2, 4) - 1;
      hour = `${String(m < 0 ? h - 1 : h).padStart(2, '0')}${String((m + 60) % 60).padStart(2, '0')}00`;
      if (hour < open) {
        // continue with the previous calendar day (KIS skips holidays when there is no data)
        date = prevDay(date);
        hour = close;
      }
    }
    const keep = [...dates].sort().slice(-days);
    const fetched = minuteRowsToBars(rows.filter((o) => keep.includes(o.stck_bsop_date)));
    // [2026-11-19] dates kept complete in the cache (never asked again); an unfinished oldest date (page limit) is asked next time
    const done = complete ? keep : keep.slice(1);
    return { bars: mergeBars(fromCache.filter((b) => keep.includes(dayOf(b.t))), fetched), dates: keep, done };
  }

  private async periodBars(code: string, period: 'D' | 'W' | 'M' | 'Y', index: boolean, pages: number, mk: MinMarket = 'J'): Promise<{ bars: Bar[]; name?: string }> {
    let date2 = ymd(kstNow());
    const rows: Record<string, string>[] = [];
    let name: string | undefined;
    for (let p = 0; p < pages; p++) {
      const r = index
        ? await this.client.get('inquire_daily_indexchartprice', {
            FID_COND_MRKT_DIV_CODE: 'U',
            FID_INPUT_ISCD: code,
            FID_INPUT_DATE_1: '19900101',
            FID_INPUT_DATE_2: date2,
            FID_PERIOD_DIV_CODE: period,
          })
        : await this.client.get('inquire_daily_itemchartprice', {
            FID_COND_MRKT_DIV_CODE: mk,
            FID_INPUT_ISCD: code,
            FID_INPUT_DATE_1: '19900101',
            FID_INPUT_DATE_2: date2,
            FID_PERIOD_DIV_CODE: period,
            FID_ORG_ADJ_PRC: '0',
          });
      name = name || objOf(r.output1).hts_kor_isnm;
      const page = rowsOf(r.output2).filter((o) => o.stck_bsop_date);
      if (!page.length) break;
      rows.push(...page);
      const earliest = page.reduce((a, o) => (o.stck_bsop_date < a ? o.stck_bsop_date : a), page[0].stck_bsop_date);
      if (page.length < 90) break;
      const d = new Date(Date.UTC(+earliest.slice(0, 4), +earliest.slice(4, 6) - 1, +earliest.slice(6, 8)) - 86400000);
      date2 = ymd(d);
    }
    return { bars: dailyRowsToBars(rows), name };
  }

  async chart(req: ChartReq): Promise<ChartRes> {
    const { code, kind } = req;
    if (kind === 'tick') return { bars: [], note: '틱봉은 실시간 체결(웹소켓)로 만듭니다 - 창을 연 뒤부터 쌓입니다.' };
    if (kind === 'min') {
      const days = Math.max(1, Math.min(20, req.days || 1));
      if (req.index) return { bars: [], note: '지수 분봉은 지원하지 않습니다 (일·주·월봉을 보세요).' };
      const key = `${code}|min|`;
      const hit = this.cachedChart(key);
      const out = (c: ChartCache, stale = false): ChartRes => {
        const note =
          days > 1 && this.client.env !== 'real' ? '모의투자 접속은 지난 날짜 분봉을 조회할 수 없어 오늘만 표시합니다.' : !c.bars.length ? '분봉 데이터가 없습니다 (장 시작 전이거나 거래가 없음).' : undefined;
        return { bars: resampleMinuteBars(c.bars, Math.max(1, req.n)), name: c.name, note, ...(stale ? { stale } : {}) };
      };
      // [2026-11-19] a cached chart answers at once; an old one says so (the window refreshes right away)
      if (hit && hit.days >= days && !req.refresh) return out(hit, Date.now() - hit.at >= 30000);
      const c = await this.loadMinutes(code, days, hit);
      this.keepChart(key, c);
      return out(c);
    }
    // day / week / month / year
    const period = kind === 'day' ? 'D' : kind === 'week' ? 'W' : kind === 'month' ? 'M' : 'Y';
    const key = `${code}|${period}|${req.index ? 'i' : ''}`;
    const hit = this.cachedChart(key);
    if (hit && !req.refresh) return { bars: hit.bars, name: hit.name, ...(Date.now() - hit.at >= 60000 ? { stale: true } : {}) };
    /** a UN refresh failed: reload the whole history with J, so the bars never mix the two series */
    let forceJ = false;
    if (hit) {
      let fresh: { bars: Bar[] } | null = null;
      try {
        fresh = await this.periodBars(code, period, !!req.index, 1, hit.mk || 'J');
      } catch (e) {
        if (hit.mk !== 'UN') throw e;
      }
      if (fresh && (fresh.bars.length || hit.mk !== 'UN')) {
        if (this.newestPageFits(hit.bars, fresh.bars)) {
          const bars = mergeBars(hit.bars, fresh.bars);
          this.keepChart(key, { ...hit, bars, at: Date.now() });
          return { bars, name: hit.name };
        }
        // the cache is older than the newest page reaches, or older adjusted prices changed: whole history again
      } else forceJ = true;
    }
    const pages = period === 'D' ? 5 : period === 'W' ? 3 : period === 'M' ? 3 : 1;
    let mk: MinMarket = !req.index && !forceJ && this.client.env === 'real' ? 'UN' : 'J';
    let res: { bars: Bar[]; name?: string } | null = null;
    if (mk === 'UN') {
      try {
        res = await this.periodBars(code, period, false, pages, 'UN');
      } catch {
        /* UN not served -> J */
      }
      if (!res?.bars.length) (mk = 'J'), (res = null);
    }
    let { bars, name } = res ?? (await this.periodBars(code, period, !!req.index, pages, 'J'));
    // KIS returns week/month/year bars directly; if a page came back as daily rows, group them here
    if (period !== 'D' && bars.length > 1 && bars[1].t - bars[0].t < 3 * 86400000) bars = period === 'W' ? weekBars(bars) : period === 'M' ? monthBars(bars) : yearBars(bars);
    this.keepChart(key, { bars, at: Date.now(), name: name || hit?.name, days: 1, mk });
    return { bars, name };
  }

  /**
   * [2026-11-19] the newest page can be merged into the cached bars: it reaches back into them (no hole between) and the older bars
   * they share have the same adjusted close (the newest two may still change)
   */
  private newestPageFits(cached: Bar[], page: Bar[]): boolean {
    if (!page.length || !cached.length) return !!cached.length;
    if (page[0].t > cached[cached.length - 1].t) return false;
    const byT = new Map(cached.map((b) => [b.t, b]));
    for (const b of page.slice(0, -2)) {
      const o = byT.get(b.t);
      if (o && o.c > 0 && Math.abs(b.c / o.c - 1) > ADJ_TOL) return false;
    }
    return true;
  }

  /**
   * [2026-11-19] minute chart of `days` trading days, reusing the cache: today's minutes back to the newest cached bar of today,
   * past days from the server only for dates not cached complete
   */
  private async loadMinutes(code: string, days: number, hit: ChartCache | undefined): Promise<ChartCache> {
    const today = ymd(kstNow());
    let mk: MinMarket = hit?.mk || ((await this.isNxt(code)) ? 'UN' : 'J');
    const cachedToday = hit && hit.mk === mk ? hit.bars.filter((b) => dayOf(b.t) === today) : [];
    let name = hit?.name;
    let todayBars: Bar[];
    if (cachedToday.length) {
      // from now back to the newest cached bar (refetched: it may have been unfinished)
      const fresh = await this.todayMinutes(code, mk === 'UN' ? 26 : 15, mk, hmsOf(cachedToday[cachedToday.length - 1].t));
      todayBars = mergeBars(cachedToday, fresh.bars);
      name = fresh.name || name;
    } else {
      const pages = mk === 'UN' ? 26 : 15;
      let t: { bars: Bar[]; name?: string } = { bars: [] };
      if (mk === 'UN') {
        try {
          t = await this.todayMinutes(code, pages, 'UN');
        } catch {
          /* combined market not served (paper server) -> KRX */
        }
        if (!t.bars.length) {
          const krx = await this.todayMinutes(code, 15, 'J');
          // before 08:00 both are empty: keep UN so past days include NXT; otherwise KRX answered and UN did not
          if (krx.bars.length) (t = krx), (mk = 'J');
        }
      } else t = await this.todayMinutes(code, pages, 'J');
      todayBars = t.bars;
      name = t.name || name;
    }
    const first = todayBars[0];
    const fromDate = first ? dayOf(first.t) : today;
    const pastDays = todayBars.length ? days - 1 : days;
    const skip = todayBars.length ? fromDate : '';
    // complete past dates of the cache (same market code only - J and UN minutes differ)
    const known = new Map<string, Bar[]>();
    if (hit && hit.mk === mk) for (const d of hit.done || []) if (d !== today) known.set(d, hit.bars.filter((b) => dayOf(b.t) === d));
    let past: { bars: Bar[]; dates: string[]; done: string[] } = { bars: [], dates: [], done: [] };
    const knownDates = [...known.keys()].sort();
    if (pastDays > 0 && hit && dayOf(hit.at) === today && knownDates.length >= pastDays) {
      // already worked out today which past dates belong to the chart (they cannot change during the day): no server call
      const use = knownDates.slice(-pastDays);
      past = { bars: mergeBars(...use.map((d) => known.get(d)!)), dates: use, done: use };
    } else if (pastDays > 0) {
      past = await this.pastMinutes(code, fromDate, pastDays, mk, skip, known).catch((e) => (mk === 'UN' ? { bars: [], dates: [], done: [] } : Promise.reject(e)));
      if (!past.bars.length && mk === 'UN' && !todayBars.length) (past = await this.pastMinutes(code, fromDate, pastDays, 'J', skip)), (mk = 'J');
    }
    return { bars: mergeBars(past.bars, todayBars), at: Date.now(), name, days, mk, done: past.done.filter((d) => d !== today) };
  }

  // ---- tables ----

  async daily(code: string): Promise<DailyRow[]> {
    const r = await this.client.get('inquire_daily_price', { FID_COND_MRKT_DIV_CODE: 'J', FID_INPUT_ISCD: code, FID_PERIOD_DIV_CODE: 'D', FID_ORG_ADJ_PRC: '0' });
    return rowsOf(r.output).map((o) => ({
      date: o.stck_bsop_date,
      open: num(o.stck_oprc),
      high: num(o.stck_hgpr),
      low: num(o.stck_lwpr),
      close: num(o.stck_clpr),
      change: signed(o.prdy_vrss, o.prdy_vrss_sign),
      changePct: num(o.prdy_ctrt),
      volume: num(o.acml_vol) || 0,
      foreignNet: num(o.frgn_ntby_qty),
      foreignRate: num(o.hts_frgn_ehrt),
    }));
  }

  async investor(code: string): Promise<InvestorRow[]> {
    const r = await this.client.get('inquire_investor', { FID_COND_MRKT_DIV_CODE: 'J', FID_INPUT_ISCD: code });
    return rowsOf(r.output)
      .filter((o) => o.stck_bsop_date)
      .map((o) => ({
        date: o.stck_bsop_date,
        close: num(o.stck_clpr),
        change: signed(o.prdy_vrss, o.prdy_vrss_sign),
        person: num(o.prsn_ntby_qty),
        foreign: num(o.frgn_ntby_qty),
        institution: num(o.orgn_ntby_qty),
        personAmt: num(o.prsn_ntby_tr_pbmn),
        foreignAmt: num(o.frgn_ntby_tr_pbmn),
        institutionAmt: num(o.orgn_ntby_tr_pbmn),
      }));
  }

  private inv = new Map<string, InvCache>();
  private invBusy = new Map<string, Promise<InvestorDay[]>>();

  async investorDaily(code: string): Promise<InvestorDay[]> {
    let hit = this.inv.get(code);
    if (!hit && this.disk) {
      const d = this.disk.get<InvCache>(`${this.client.env}_inv_${code}`);
      if (d && Array.isArray(d.rows) && d.v === INV_V) this.inv.set(code, (hit = d));
    }
    if (hit && hit.rows.length && Date.now() - hit.at < INV_REUSE_MS && dayOf(hit.at) === ymd(kstNow())) return hit.rows;
    let p = this.invBusy.get(code);
    if (!p) {
      p = this.loadInvestor(code, hit).finally(() => this.invBusy.delete(code));
      this.invBusy.set(code, p);
    }
    return p;
  }

  private async loadInvestor(code: string, hit: InvCache | undefined): Promise<InvestorDay[]> {
    const mk: 'J' | 'UN' = this.client.env === 'real' && hit?.mk !== 'J' ? 'UN' : 'J';
    try {
      return await this.loadInvestorMk(code, hit?.mk === mk || (!hit?.mk && mk === 'J') ? hit : undefined, mk);
    } catch (e) {
      if (mk !== 'UN') throw e;
      return this.loadInvestorMk(code, hit?.mk === 'J' ? hit : undefined, 'J');
    }
  }

  private async loadInvestorMk(code: string, hit: InvCache | undefined, mk: 'J' | 'UN'): Promise<InvestorDay[]> {
    const now = kstNow();
    const today = ymd(now);
    const start = ymd(new Date(now.getTime() - INV_DAYS * 86400000));
    const cached = (hit?.rows || []).filter((r) => r.date >= start);
    const newest = cached.length ? cached[cached.length - 1].date : '';
    const covered = !!hit && cached.length > 0 && (hit.from || '99999999') <= start;
    const map = new Map(cached.map((r) => [r.date, r]));
    // today's rows are served only after 15:40 (before: rt_cd 2 "TIME LIMIT 00:00 ~ 15:40")
    let date = hms(now) < '154000' ? prevYmd(today) : today;
    let from = start;
    let limited = 0;
    try {
      for (let page = 0; page < INV_PAGES; page++) {
        let r;
        try {
          r = await this.client.get('investor_trade_by_stock_daily', { FID_COND_MRKT_DIV_CODE: mk, FID_INPUT_ISCD: code, FID_INPUT_DATE_1: date, FID_ORG_ADJ_PRC: '', FID_ETC_CLS_CODE: '' });
        } catch (e) {
          const k = e as KisError;
          if (page === 0 && limited++ < 2 && (k.code === 'OPSQ2001' || /TIME LIMIT/i.test(k.message))) {
            date = prevYmd(date);
            page--;
            continue;
          }
          throw e;
        }
        const got = rowsOf(r.output2)
          .filter((o) => /^\d{8}$/.test(o.stck_bsop_date || '') && o.stck_bsop_date <= date)
          .map((o) => ({
            date: o.stck_bsop_date,
            person: num(o.prsn_ntby_qty) || 0,
            foreign: num(o.frgn_ntby_qty) || 0,
            institution: num(o.orgn_ntby_qty) || 0,
            close: num(o.stck_clpr),
            change: signed(o.prdy_vrss, o.prdy_vrss_sign),
            personAmt: num(o.prsn_ntby_tr_pbmn),
            foreignAmt: num(o.frgn_ntby_tr_pbmn),
            institutionAmt: num(o.orgn_ntby_tr_pbmn),
          }));
        if (!got.length) {
          if (page === 0 && mk === 'UN' && !cached.length) throw new Error('UN: no rows');
          break;
        }
        for (const g of got) map.set(g.date, g);
        const oldest = got.reduce((m, g) => (g.date < m ? g.date : m), got[0].date);
        // the page reached the cached rows (older days do not change) and the cache already goes back far enough
        if (newest && oldest <= newest && covered) break;
        if (oldest <= start) break;
        const next = prevYmd(oldest);
        if (next >= date) break;
        date = next;
        if (page === INV_PAGES - 1) from = oldest; // page limit: complete only back to here
      }
    } catch (e) {
      if (cached.length) return cached;
      throw e;
    }
    const rows = [...map.values()].filter((r) => r.date >= start).sort((a, b) => (a.date < b.date ? -1 : 1));
    const c: InvCache = { rows, at: Date.now(), from: covered ? start : from, mk, v: INV_V };
    this.inv.set(code, c);
    this.disk?.put(`${this.client.env}_inv_${code}`, c);
    return rows;
  }

  async info(code: string): Promise<InfoField[]> {
    const r = await this.client.get('search_stock_info', { PRDT_TYPE_CD: '300', PDNO: code });
    const o = objOf(r.output);
    const fields = [
      'prdt_abrv_name', 'prdt_eng_abrv_name', 'mket_id_cd', 'scty_grp_id_cd', 'std_idst_clsf_cd_name', 'idx_bztp_lcls_cd_name', 'idx_bztp_mcls_cd_name',
      'idx_bztp_scls_cd_name', 'scts_mket_lstg_dt', 'kosdaq_mket_lstg_dt', 'lstg_stqt', 'cpta', 'papr', 'setl_mmdd', 'kospi200_item_yn', 'tr_stop_yn',
      'admn_item_yn', 'etf_dvsn_cd', 'bfdy_clpr', 'thdt_clpr', 'cptt_trad_tr_psbl_yn', 'std_pdno',
    ];
    return fields
      .filter((f) => o[f] !== undefined && String(o[f]).trim() !== '')
      .map((f) => ({ label: fieldLabel('search_stock_info', f), value: String(o[f]).trim() }));
  }

  /** market code of a watchlist row now: NX for NXT symbols while KRX is closed (real server), else J */
  private watchMarket(code: string, now = Date.now()): 'J' | 'NX' {
    if (this.client.env !== 'real' || !this.nxtHint(code)) return 'J';
    if ((this.nxSkip.get(code) || 0) > now) return 'J';
    return nxtOnlyHours(hms(new Date(now + KST)).slice(0, 4)) ? 'NX' : 'J';
  }

  async multi(codes: string[]) {
    const out: WatchRow[] = [];
    for (let i = 0; i < codes.length; i += 30) {
      const chunk = codes.slice(i, i + 30);
      const now = Date.now();
      const nx = new Set(chunk.filter((c) => this.watchMarket(c, now) === 'NX'));
      if (!nx.size) {
        const r = await this.client.get('intstock_multprice', multPriceParams(chunk));
        out.push(...toWatchRows(rowsOf(r.output)));
        continue;
      }
      let rows: WatchRow[] = [];
      try {
        const r = await this.client.get('intstock_multprice', multPriceParams(chunk, (c) => (nx.has(c) ? 'NX' : 'J')));
        rows = toWatchRows(rowsOf(r.output));
      } catch {
        /* NX not served -> every symbol from KRX below */
      }
      const got = new Map(rows.map((x) => [x.code, x]));
      // an NXT symbol without an NX price (no row, or no trade on NXT yet): its KRX row, and KRX only for the next minute
      const bad = chunk.filter((c) => !(Number(got.get(c)?.price) > 0));
      if (bad.length) {
        for (const c of bad) if (nx.has(c)) this.nxSkip.set(c, now + 60000);
        const r = await this.client.get('intstock_multprice', multPriceParams(chunk));
        const krx = new Map(toWatchRows(rowsOf(r.output)).map((x) => [x.code, x]));
        for (const c of bad) {
          const x = krx.get(c);
          if (x) got.set(c, x);
        }
      }
      for (const c of chunk) {
        const x = got.get(c);
        if (x) out.push(x);
      }
    }
    return out;
  }

  async rank(req: RankReq) {
    if (req.type === 'volume' || req.type === 'amount') {
      const r = await this.client.get('volume_rank', {
        FID_COND_MRKT_DIV_CODE: 'J',
        FID_COND_SCR_DIV_CODE: '20171',
        FID_INPUT_ISCD: req.market,
        FID_DIV_CLS_CODE: '0',
        FID_BLNG_CLS_CODE: req.type === 'amount' ? '3' : '0',
        FID_TRGT_CLS_CODE: '111111111',
        FID_TRGT_EXLS_CLS_CODE: '0000000000',
        FID_INPUT_PRICE_1: '',
        FID_INPUT_PRICE_2: '',
        FID_VOL_CNT: '',
        FID_INPUT_DATE_1: '',
      });
      return toRankRows(rowsOf(r.output), 'vol_inrt');
    }
    if (req.type === 'cap') {
      const r = await this.client.get('market_cap', {
        fid_input_price_2: '',
        fid_cond_mrkt_div_code: 'J',
        fid_cond_scr_div_code: '20174',
        fid_div_cls_code: '0',
        fid_input_iscd: req.market,
        fid_trgt_cls_code: '0',
        fid_trgt_exls_cls_code: '0',
        fid_input_price_1: '',
        fid_vol_cnt: '',
      });
      return toRankRows(rowsOf(r.output), 'stck_avls');
    }
    const r = await this.client.get('fluctuation', {
      fid_rsfl_rate2: '',
      fid_cond_mrkt_div_code: 'J',
      fid_cond_scr_div_code: '20170',
      fid_input_iscd: req.market,
      fid_rank_sort_cls_code: req.type === 'rise' ? '0' : '1',
      fid_input_cnt_1: '0',
      fid_prc_cls_code: '0',
      fid_input_price_1: '',
      fid_input_price_2: '',
      fid_vol_cnt: '',
      fid_trgt_cls_code: '0',
      fid_trgt_exls_cls_code: '0',
      fid_div_cls_code: '0',
      fid_rsfl_rate1: '',
    });
    return toRankRows(rowsOf(r.output));
  }

  async index(code: string): Promise<IndexQuote> {
    const r = await this.client.get('inquire_index_price', { FID_COND_MRKT_DIV_CODE: 'U', FID_INPUT_ISCD: code });
    const o = objOf(r.output);
    return {
      code,
      name: INDEXES.find((i) => i.code === code)?.name || code,
      value: num(o.bstp_nmix_prpr),
      change: signed(o.bstp_nmix_prdy_vrss, o.prdy_vrss_sign),
      changePct: num(o.bstp_nmix_prdy_ctrt),
      sign: o.prdy_vrss_sign,
      volume: num(o.acml_vol),
      amount: num(o.acml_tr_pbmn),
      open: num(o.bstp_nmix_oprc),
      high: num(o.bstp_nmix_hgpr),
      low: num(o.bstp_nmix_lwpr),
      up: num(o.ascn_issu_cnt),
      upLimit: num(o.uplm_issu_cnt),
      flat: num(o.stnr_issu_cnt),
      down: num(o.down_issu_cnt),
      downLimit: num(o.lslm_issu_cnt),
    };
  }

  /** [2026-11-26] E-Mini Nasdaq 100 futures index, delayed (해외지수분봉조회). One call: the quote and the newest ~100 one-minute bars */
  async nqFutures(): Promise<NqFutures> {
    const r = await this.client.get('inquire_time_indexchartprice', {
      FID_COND_MRKT_DIV_CODE: NQ_FUTURES.market,
      FID_INPUT_ISCD: NQ_FUTURES.code,
      FID_HOUR_CLS_CODE: '0',
      FID_PW_DATA_INCU_YN: 'Y',
    });
    return toNqFutures(objOf(r.output1), rowsOf(r.output2));
  }

  async news(code: string): Promise<NewsRow[]> {
    const r = await this.client.get('news_title', {
      FID_NEWS_OFER_ENTP_CODE: '',
      FID_COND_MRKT_CLS_CODE: '',
      FID_INPUT_ISCD: code,
      FID_TITL_CNTT: '',
      FID_INPUT_DATE_1: '',
      FID_INPUT_HOUR_1: '',
      FID_RANK_SORT_CLS_CODE: '',
      FID_INPUT_SRNO: '',
    });
    return rowsOf(r.output).map((o) => ({
      date: o.data_dt,
      time: o.data_tm,
      title: (o.hts_pbnt_titl_cntt || '').trim(),
      source: o.dorg || '',
      codes: [o.iscd1, o.iscd2, o.iscd3, o.iscd4, o.iscd5].filter(Boolean),
    }));
  }
}
