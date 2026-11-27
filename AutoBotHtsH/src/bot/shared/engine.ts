/*
 * Update log
 * - [2026-11-04] type=feat - DayEngine: aggregates one trading day (main logs + daily buy_sell log) into per-symbol state,
 *   per-condition funnel counts, trades and the unknown list. Pure TypeScript (no Node/Electron APIs) so the same code runs in
 *   the worker thread and in vitest. Fed with complete lines only (see LineBuffer)
 * - [2026-11-05] type=feat - Status events ([CHECK]/[BAR_SKIP], sub STATUS_SUB) are thinned so they cannot push real evaluations out of
 *   the per-symbol timeline cap: one is kept only when the symbol's previous timeline event is at least STATUS_GAP_MS old, i.e. at
 *   most one gray "still checking" dot per minute and none while conditions are being evaluated
 * - [2026-11-05] type=fix - Symbol list cleanup: (1) the name from the WATCH line (full name, the bot's own watchlist) now wins over
 *   names from log labels / file names, which could be cut or carry a suffix; (2) a line with an ignored tag ([WARMUP] etc.) no
 *   longer creates a symbol by itself - the market-regime ETFs (069500 KOSPI, 229200 KOSDAQ) only appear in [WARMUP] lines and
 *   were listed as name-less symbols with no events
 * - [2026-11-08] type=fix - Condition keys from the log rules (and sell trades) are translated to the server's numbers (mapping
 *   keyRemap, rebuilt in setConditions) so an app/server version gap does not show "조건 목록에 없음"
 * - [2026-11-13] type=fix - a bot buy sold by hand (no sell in the trade log) stayed 보유 on the board: buys from before the last
 *   "[ACT REENTER]" / "[BOT LEDGER] ... account=N->0" of the symbol no longer count as an open position (SymState.closedTs)
 * - [2026-11-13] type=feat - held state 'manual': "[HOLD SKIP] ... NOT_BOT_POSITION" (bought by hand today, the bot leaves it
 *   alone under AUTO_SELL_SCOPE=bot_only); carried / manual holdings stay fresh on "[SKIP] ... can_trade_code_now=False"
 * - [2026-11-13] type=fix - (1) r003 restart: the watch state is rebuilt from the WATCH lines of the latest run ("R003 START").
 *   A symbol watched by an earlier run but not by the latest one (r004 edited, then restarted) becomes 'removed' and is left out
 *   of the snapshot when it has no events / trades - it used to stay on the board as 감시 for the rest of the day (251340 on
 *   2026-11-13: deleted in the watchlist editor at 07:56, r003 restarted with 8 symbols, the board kept showing it).
 *   (2) SymbolSummary.held: holdings not bought today - 'carried' from "[HOLD SKIP] ... NOT_TODAY_BUY_POSITION", 'unmanaged' from
 *   "[WARN] account holdings not in watchlist: ..." - so the board can say 보유 instead of 졸업 / -
 * - [2026-11-17] type=fix - per-symbol timeline cap: repeats are thinned over the whole day (compact) instead of cutting the oldest
 *   events, so a busy symbol keeps its morning on the 매수 평가 chart (068270 showed nothing before 11:00)
 */

import { sideSummaries } from './methodBoard';
import { buyStrategyOf, sellStrategyOf } from './strategy';
import { STRATEGY_TAGS, parseStrategyNote, parseStrategyRest, type SideStrategy, type StrategyEvent, type StrategyNote } from './strategyEvents';
import type {
  ConditionsDoc,
  EvalEvent,
  NodeStat,
  SellNodeStat,
  Snapshot,
  SymbolSummary,
  Trade,
  UnknownEntry,
  WatchState,
} from './types';
import { BS_BLOCK_RE, BS_DETAIL_RE, num, parseBuySellLine, parseKv, parseMainLine } from './logLine';
import { IGNORED_TAGS, STATUS_SUB, classify, condKeyOf, keyRemap, reasonShape, remapKey, sellReasonToCond, type Classified, type KeyRemap } from './mapping';

const STATUS_GAP_MS = 60000;
/** a held state is shown while its log line is this recent relative to the last line read ([HOLD SKIP] repeats every poll,
 *  the holdings warning every 5 minutes) */
const CARRIED_FRESH_MS = 3 * 60000;
const UNMANAGED_FRESH_MS = 12 * 60000;
import { hhmmToSeconds, kstDateTimeToEpoch, kstSecondOfDay } from './time';

export const TIMELINE_CAP = 2000;
/** compaction widens its bucket step by step until the timeline fits the cap; the newest COMPACT_KEEP_RECENT events stay as read */
const COMPACT_BUCKETS_MS = [30000, 60000, 120000, 300000, 600000, 1800000];
const COMPACT_KEEP_RECENT = 400;

/** [2026-11-27] strategy events kept per symbol / notes kept per day (a held symbol writes a check line every ~30 s) */
const STRATEGY_EVENT_CAP = 4000;
const STRATEGY_NOTE_CAP = 200;
/** a fill block belongs to a side order when its printed order time is within this of the order's SUBMIT line */
const SIDE_BLOCK_MATCH_MS = 15000;

/** a side-strategy buy order seen in the buy_sell log ([BUY SUBMIT] ... reason=VWAP_LIVE ..., then [BUY PENDING] filled=a/b) */
interface SideOrder {
  code: string;
  name: string;
  ts: number;
  price: number;
  tail: string;
  strategy: SideStrategy;
  filled: number;
  raw: string;
}

interface SymState {
  summary: SymbolSummary;
  timeline: EvalEvent[];
  /** [2026-11-27] what the side strategies wrote about the symbol */
  side: StrategyEvent[];
  /** watch state at the end of the previous r003 run (set on "R003 START") */
  prevWatch?: WatchState;
  carriedTs?: number;
  unmanagedTs?: number;
  /** [2026-11-13] last "[HOLD SKIP] ... NOT_BOT_POSITION": bought by hand today, the bot leaves it alone (AUTO_SELL_SCOPE=bot_only) */
  manualTs?: number;
  /** [2026-11-13] the account no longer holds the symbol although the trade log has no sell for it (sold by hand): time of the
   *  last "[ACT REENTER]" (r003 saw the bot's position gone and put the symbol back on watch) or "[BOT LEDGER] ... account=N->0" */
  closedTs?: number;
}

export interface EngineOptions {
  sessionOpen?: string; // "09:00"
  timelineCap?: number;
}

export class DayEngine {
  readonly date: string;
  private buyIndex = new Map<string, number>();
  private buyKeys: string[] = [];
  /** app condition key -> server key (see mapping keyRemap) */
  private remap: Record<'buy' | 'sell', KeyRemap> = { buy: new Map(), sell: new Map() };
  private syms = new Map<string, SymState>();
  private buyStats = new Map<string, NodeStat>();
  private sellStats = new Map<string, SellNodeStat>();
  private trades: Trade[] = [];
  private openTrade: Trade | null = null;
  /** [2026-11-27] method of the symbol's last buy block - a sell whose reason does not name a method takes it */
  private lastBuyStrategy = new Map<string, 'rule' | 'edge' | 'vwap'>();
  /** [2026-11-27] side-strategy buy orders of the buy_sell log by order number, and the trade list with them merged in */
  private sideOrders = new Map<string, SideOrder>();
  private mergedTrades: Trade[] | null = null;
  private strategyNotes: StrategyNote[] = [];
  /** the side strategy that last signalled / ordered the symbol (older logs tag every fill line "[EDGE LIVE]") */
  private lastSide = new Map<string, SideStrategy>();
  private unknown = new Map<string, UnknownEntry>();
  private unknownLines = 0;
  private regime?: string;
  private wl?: { active: number; backup: number };
  private lineCount = 0;
  private lastLineTs?: number;
  private nextId = 1;
  private sessionOpenSec: number;
  private cap: number;

  constructor(date: string, conditions: ConditionsDoc, opts: EngineOptions = {}) {
    this.date = date;
    this.sessionOpenSec = hhmmToSeconds(opts.sessionOpen || '09:00');
    this.cap = opts.timelineCap || TIMELINE_CAP;
    this.setConditions(conditions);
  }

  /** Evaluation order comes from Condition.order (not the number). Counts already collected stay keyed by "_NNN". */
  setConditions(doc: ConditionsDoc): void {
    const sorted = [...doc.buy].sort((a, b) => a.order - b.order);
    this.buyKeys = sorted.map((c) => condKeyOf(c.no));
    this.buyIndex = new Map(this.buyKeys.map((k, i) => [k, i]));
    this.remap = { buy: keyRemap(doc, 'buy'), sell: keyRemap(doc, 'sell') };
  }

  private watchNamed = new Set<string>();

  /** `fromWatch`: name from the WATCH line - authoritative; other sources only fill in or lengthen a name not set by WATCH. */
  setName(code: string, name: string, fromWatch = false): void {
    if (!name) return;
    const s = this.sym(code);
    if (fromWatch) {
      this.watchNamed.add(code);
      s.summary.name = name;
    } else if (!this.watchNamed.has(code) && (!s.summary.name || s.summary.name.length < name.length)) s.summary.name = name;
  }

  // ---- main log ------------------------------------------------------------------------

  feedMainLines(lines: string[]): void {
    for (const raw of lines) this.mainLine(raw);
  }

  private mainLine(raw: string): void {
    const p = parseMainLine(raw);
    if (p.kind === 'blank' || p.kind === 'continuation') return;
    this.lineCount++;
    this.lastLineTs = p.ts;

    if (p.kind === 'untagged') {
      if (p.body.startsWith('R003 START')) {
        // a new run re-reads r004: everything watched so far is 'removed' until this run logs its WATCH line
        for (const st of this.syms.values()) {
          if (st.summary.watch === 'unknown') continue;
          if (st.summary.watch !== 'removed') st.prevWatch = st.summary.watch;
          st.summary.watch = 'removed';
        }
        return;
      }
      const w = /^WATCH \| (\d{6}) \| ([^|]+?) \|(.*)$/.exec(p.body);
      if (w) {
        this.setName(w[1], w[2].trim(), true);
        const s = this.sym(w[1]);
        const nxt = /\bNXT=(True|False)\b/.exec(w[3]);
        if (nxt) s.summary.nxt = nxt[1] === 'True';
        // a symbol that graduated (position opened) stays out of the active set after a restart (r003 restores the reentry state
        // without logging [ACT GRADUATE] again)
        if (s.summary.watch === 'removed') s.summary.watch = s.prevWatch === 'graduated' ? 'graduated' : 'active';
        else if (s.summary.watch === 'unknown') s.summary.watch = 'active';
      }
      return;
    }

    // tagged line without a symbol: global state
    if (!p.code) {
      const st = STRATEGY_TAGS[p.tag];
      if (st) {
        const n = parseStrategyNote(p.body);
        // the same note repeats every 10 minutes while it holds: keep the first time it was said and the last
        const last = this.strategyNotes[this.strategyNotes.length - 1];
        if (last && last.strategy === st.strategy && last.mode === st.mode && last.kind === n.kind && last.text === n.text) last.lastTs = p.ts;
        else {
          this.strategyNotes.push({ ts: p.ts, lastTs: p.ts, ...st, ...n });
          if (this.strategyNotes.length > STRATEGY_NOTE_CAP) this.strategyNotes.shift();
        }
        return;
      }
      if (p.tag === 'MKT REGIME') this.regime = p.body.split('|')[0].trim();
      else if (p.tag === 'WL SPLIT') {
        const m = /active=(\d+)\s+backup=(\d+)/.exec(p.body);
        if (m) this.wl = { active: +m[1], backup: +m[2] };
      } else if (p.tag === 'WARN') {
        // "account holdings not in watchlist: 033780, 005930 - not managed by the bot (...)"
        const h = /account holdings not in watchlist:\s*(.*)$/.exec(p.body);
        if (h) for (const code of h[1].split(' - ')[0].match(/\d{6}/g) || []) this.sym(code).unmanagedTs = p.ts;
      }
      return;
    }

    // ignored tags (except ACT state changes) never introduce a symbol - e.g. [WARMUP] of the market-regime ETFs
    if (!this.syms.has(p.code) && IGNORED_TAGS.has(p.tag) && !p.tag.startsWith('ACT ')) return;
    const s = this.sym(p.code);
    if (p.name) this.setName(p.code, p.name);

    // [2026-11-27] a side strategy's line about the symbol: kept as it is for the timeline's method tabs, not a rule condition
    const side = STRATEGY_TAGS[p.tag];
    if (side) {
      const f = parseStrategyRest(side.mode, p.rest);
      let strategy = side.strategy;
      if (f.kind === 'signal' || f.kind === 'order' || f.kind === 'nobuy' || f.kind === 'hold' || f.kind === 'exit') this.lastSide.set(p.code, strategy);
      // fill / close lines of logs before 2026-11-27 are always tagged EDGE LIVE: those (only those) follow the strategy that last
      // signalled / ordered the symbol. A VWAP LIVE tag is always what it says [Codex round 3]
      else if (p.tag === 'EDGE LIVE') strategy = this.lastSide.get(p.code) || strategy;
      s.side.push({ id: this.nextId++, ts: p.ts, code: p.code, strategy, mode: side.mode, ...f, raw });
      if (s.side.length > STRATEGY_EVENT_CAP) {
        // thin the oldest holding checks first (one of two), the rarer events stay
        let dropped = false;
        s.side = s.side.filter((e, i) => !(e.kind === 'hold' && i < s.side.length / 2 && (dropped = !dropped)));
      }
      return;
    }

    if (p.tag === 'ACT REENTER' || (p.tag === 'BOT LEDGER' && /\baccount=\d+->0\b/.test(p.rest))) s.closedTs = p.ts;
    if (p.tag.startsWith('ACT ')) {
      const next: Record<string, WatchState> = {
        'ACT PROMOTE': 'active',
        'ACT REENTER': 'active',
        'ACT DROPOUT': 'dropped',
        'ACT GRADUATE': 'graduated',
      };
      if (next[p.tag]) s.summary.watch = next[p.tag];
      return;
    }

    if (p.tag === 'HOLD SKIP' && p.rest.trim().startsWith('NOT_TODAY_BUY_POSITION')) s.carriedTs = p.ts;
    if (p.tag === 'HOLD SKIP' && p.rest.trim().startsWith('NOT_BOT_POSITION')) s.manualTs = p.ts;
    // outside the symbol's session r003 logs [SKIP] can_trade_code_now=False instead of HOLD SKIP (a carried holding that is
    // not NXT-tradeable, after the regular close) - the holding is still there, so keep it fresh
    if (p.tag === 'SKIP' && p.rest.includes('can_trade_code_now=False')) {
      if (s.carriedTs !== undefined) s.carriedTs = p.ts;
      if (s.manualTs !== undefined) s.manualTs = p.ts;
    }

    const out = classify(p.tag, p.rest);
    if (out.kind === 'ignore') return;
    if (out.kind === 'unknown') {
      this.addUnknown(p.tag, out.reason, raw);
      this.push(s, p.ts, p.code, p.tag, { side: out.side, condKey: 'UNKNOWN', result: 'info', reason: out.reason }, raw);
      return;
    }
    if (out.ev.sub === STATUS_SUB) {
      const last = s.timeline[s.timeline.length - 1];
      if (last && p.ts - last.ts < STATUS_GAP_MS) return;
    }
    const ev = out.ev;
    this.push(s, p.ts, p.code, p.tag, { ...ev, condKey: remapKey(this.remap[ev.side], ev.condKey) }, raw);
  }

  /** [2026-11-17] Keep the whole day inside the cap. Cutting the oldest events left only the last ~2 hours of a busy symbol
   *  (068270: ~10k evaluations by 12:45, nothing before 11:00 on the chart). Repeats of the same (side, condition, step, result) are
   *  thinned to one per bucket; passes, orders and the newest COMPACT_KEEP_RECENT events are never dropped. Only if the widest
   *  bucket is still too many, the oldest events go. */
  private compact(s: SymState): void {
    const tl = s.timeline;
    const head = tl.slice(0, Math.max(0, tl.length - Math.min(COMPACT_KEEP_RECENT, Math.floor(this.cap / 5))));
    const tail = tl.slice(head.length);
    // compact well below the cap so the next ~cap/4 events fit before the next pass (and the oldest are never cut on read)
    const target = Math.floor(this.cap * 0.75);
    let kept = head;
    for (const bucket of COMPACT_BUCKETS_MS) {
      const lastTs = new Map<string, number>();
      kept = head.filter((ev) => {
        if (ev.result === 'pass' || ev.condKey === 'ORDER') return true;
        const sig = `${ev.side}|${ev.condKey}|${ev.sub || ''}|${ev.result}`;
        const prev = lastTs.get(sig);
        if (prev !== undefined && ev.ts - prev < bucket) return false;
        lastTs.set(sig, ev.ts);
        return true;
      });
      if (kept.length + tail.length <= target) break;
    }
    const out = kept.concat(tail);
    s.timeline = out.length > target ? out.slice(out.length - target) : out;
  }

  private push(s: SymState, ts: number, code: string, tag: string, c: Classified, raw: string): void {
    const ev: EvalEvent = { id: this.nextId++, ts, code, tag, raw, ...c };
    s.timeline.push(ev);
    if (s.timeline.length > this.cap) this.compact(s);
    const sum = s.summary;
    const inSession = kstSecondOfDay(ts) >= this.sessionOpenSec;

    if (c.side === 'buy') {
      if (c.condKey === 'ORDER' && c.result !== 'pass') {
        sum.lastOrder = ev;
        return;
      }
      if (c.condKey === 'UNKNOWN') return;
      if (c.result === 'info') return;
      sum.lastEval = ev;
      sum.evalCount++;
      const idx = c.condKey === 'ORDER' ? this.buyKeys.length : c.condKey === 'PREP' ? -1 : this.buyIndex.get(c.condKey);
      if (c.result === 'fail') {
        const st = this.stat(c.condKey);
        // sub-step counts of the trigger (TRIGGER_COND) are kept as "_009/trigger" etc. (not part of the evaluation order)
        const subSt = c.sub ? this.stat(`${c.condKey}/${c.sub}`) : null;
        if (inSession) {
          st.block++;
          if (subSt) subSt.block++;
          sum.blockCounts[c.condKey] = (sum.blockCounts[c.condKey] || 0) + 1;
        } else {
          st.preOpenBlock++;
          if (subSt) subSt.preOpenBlock++;
        }
      } else if (c.result === 'hold') {
        this.stat(c.condKey).hold++;
      }
      if (idx !== undefined) {
        // everything ordered before the stopping condition passed on this poll
        const passedUpTo = idx - 1; // ORDER pass: idx = buyKeys.length -> all conditions passed
        if (inSession) for (let i = 0; i <= passedUpTo && i < this.buyKeys.length; i++) this.stat(this.buyKeys[i]).pass++;
        const reached = c.condKey === 'ORDER' ? this.buyKeys.length : passedUpTo;
        if (reached > sum.furthestOrderToday) sum.furthestOrderToday = reached;
      }
      return;
    }

    // sell side
    if (c.condKey === 'ORDER') {
      sum.lastOrder = ev;
      if (c.sub === 'SELL FILLED' && /remaining=0\b/.test(raw)) sum.position = undefined;
      return;
    }
    if (c.condKey === 'UNKNOWN' || c.result === 'info') return;
    sum.lastSell = ev;
    const st = this.sellStats.get(c.condKey) || { trigger: 0, hold: 0 };
    if (c.result === 'hold') st.hold++;
    else st.trigger++;
    this.sellStats.set(c.condKey, st);
    sum.sellCounts[c.condKey] = (sum.sellCounts[c.condKey] || 0) + 1;
  }

  private stat(key: string): NodeStat {
    let st = this.buyStats.get(key);
    if (!st) {
      st = { block: 0, pass: 0, hold: 0, preOpenBlock: 0 };
      this.buyStats.set(key, st);
    }
    return st;
  }

  private addUnknown(tag: string, reason: string, raw: string): void {
    this.unknownLines++;
    const key = `${tag} | ${reasonShape(reason)}`;
    const e = this.unknown.get(key);
    if (e) e.count++;
    else this.unknown.set(key, { key, tag, count: 1, sample: raw });
  }

  private sym(code: string): SymState {
    let s = this.syms.get(code);
    if (!s) {
      s = {
        summary: {
          code,
          name: '',
          watch: 'unknown',
          furthestOrderToday: -1,
          blockCounts: {},
          sellCounts: {},
          realizedKrw: 0,
          evalCount: 0,
        },
        timeline: [],
        side: [],
      };
      this.syms.set(code, s);
    }
    return s;
  }

  // ---- daily buy_sell log --------------------------------------------------------------

  feedBuySellLines(lines: string[]): void {
    for (const raw of lines) this.buySellLine(raw);
  }

  private buySellLine(raw: string): void {
    const p = parseBuySellLine(raw);
    if (!p) return;
    // [2026-11-27] a side-strategy buy order: r003 (before 2026-11-27) could close it without writing the fill block, so the
    // order itself is remembered by its number - see allTrades()
    const so = /^\[BUY (SUBMIT|PENDING)\s*\]\s+(\d{6})_(\S*)\s*\|(.*)$/.exec(p.body);
    if (so) {
      const kv = parseKv(so[4]);
      const no = (kv.order_no || '').replace(/^0+/, '');
      if (!no) return;
      if (so[1] === 'SUBMIT') {
        const tail = (kv.reason || '').trim();
        const strategy = buyStrategyOf(tail);
        if (strategy !== 'rule' && /^(VWAP|EDGE)_LIVE\b/.test(tail))
          this.sideOrders.set(no, { code: so[2], name: so[3].replace(/~+$/, ''), ts: p.ts, price: num(kv.price), tail, strategy, filled: 0, raw });
      } else {
        const o = this.sideOrders.get(no);
        const f = /^(\d+)\/(\d+)/.exec(kv.filled || '');
        if (o && f && +f[1] !== o.filled) {
          o.filled = +f[1]; // the cumulative quantity as last seen (never summed)
          o.raw += '\n' + raw;
          this.mergedTrades = null;
        }
      }
      return;
    }
    const b = BS_BLOCK_RE.exec(p.body);
    if (b) {
      const [, code, name, side, tail] = b;
      const kv = parseKv(tail);
      const reason = (kv.reason || '').split(/\s+/)[0];
      const t: Trade = {
        code,
        name,
        side: side === 'Buy' ? 'BUY' : 'SELL',
        ts: p.ts,
        price: NaN,
        qty: NaN,
        reason,
        raw,
      };
      if (t.side === 'BUY') {
        t.strategy = buyStrategyOf(tail);
        this.lastBuyStrategy.set(code, t.strategy);
      } else t.strategy = sellStrategyOf(reason) || this.lastBuyStrategy.get(code) || 'rule';
      if (t.side === 'SELL') {
        t.pnlKrw = num(kv['수익금액']);
        t.pnlPct = num(kv['수익율']);
        const k = sellReasonToCond(reason);
        t.condKey = k ? remapKey(this.remap.sell, k) : 'UNKNOWN';
      }
      if (name) this.setName(code, name);
      this.trades.push(t);
      this.mergedTrades = null;
      this.openTrade = t;
      return;
    }
    const d = BS_DETAIL_RE.exec(p.body);
    if (d && this.openTrade) {
      const t = this.openTrade;
      const kv = parseKv(d[1]);
      t.raw += '\n' + raw;
      this.mergedTrades = null; // the detail line sets the block's time / price: the merged list is built again [Codex round 3]
      if (kv['매수시각']) {
        const bts = kstDateTimeToEpoch(kv['매수시각']);
        if (t.side === 'BUY') {
          t.ts = bts;
          t.price = num(kv['매수가격']);
          t.qty = num(kv['매수수량']);
        } else {
          t.buyTs = bts;
          t.buyPrice = num(kv['매수가격']);
        }
      }
      if (kv['매도시각'] && t.side === 'SELL') {
        t.ts = kstDateTimeToEpoch(kv['매도시각']);
        t.price = num(kv['매도가격']);
        t.qty = num(kv['매도수량']);
      }
      return;
    }
    if (!/^=+$/.test(p.body.trim())) this.openTrade = null;
  }

  // ---- output --------------------------------------------------------------------------

  /** [2026-11-27] the fill blocks of the trade log + the side-strategy buy orders that were filled but have no block (matched
   *  by symbol, method and the order time the block prints - only then it is certainly the same order). Time order. */
  private allTrades(): Trade[] {
    if (this.mergedTrades) return this.mergedTrades;
    const extra: Trade[] = [];
    // [Codex round 3] one block answers one order: each filled order takes the nearest block of its symbol + method that no other
    // order took (the block prints the order's submit time; SIDE_BLOCK_MATCH_MS allows for the API round trip). A side strategy
    // judges a symbol once per 1-minute bar and never reprices, so two of its orders are never that close.
    const used = new Set<Trade>();
    const orders = [...this.sideOrders].filter(([, o]) => o.filled > 0).sort((a, b) => a[1].ts - b[1].ts);
    for (const [no, o] of orders) {
      let best: Trade | null = null;
      for (const t of this.trades) {
        if (t.side !== 'BUY' || t.code !== o.code || t.strategy !== o.strategy || used.has(t)) continue;
        const gap = Math.abs(t.ts - o.ts);
        if (gap <= SIDE_BLOCK_MATCH_MS && (!best || gap < Math.abs(best.ts - o.ts))) best = t;
      }
      if (best) {
        used.add(best);
        continue;
      }
      extra.push({ code: o.code, name: o.name, side: 'BUY', ts: o.ts, price: o.price, qty: o.filled, reason: o.tail, raw: o.raw, strategy: o.strategy, approx: true, orderNo: no });
    }
    this.mergedTrades = extra.length ? [...this.trades, ...extra].sort((a, b) => a.ts - b.ts) : this.trades;
    return this.mergedTrades;
  }

  getTrades(): Trade[] {
    return this.allTrades();
  }

  timeline(code: string): { events: EvalEvent[]; trades: Trade[]; strategy: StrategyEvent[] } {
    const s = this.syms.get(code);
    const events = s ? s.timeline.slice() : [];
    return { events, trades: this.allTrades().filter((t) => t.code === code), strategy: s ? s.side.slice() : [] };
  }

  snapshot(): Snapshot {
    // positions and realized P&L from the trade log (buy_sell log is the source of truth for fills)
    const pos = new Map<string, { qty: number; buyPrice: number; buyTs: number }>();
    const realized = new Map<string, number>();
    // a buy from before the symbol's closedTs is no longer held (sold outside the bot: no sell in the trade log)
    const closedBefore = (code: string, ts: number) => {
      const c = this.syms.get(code)?.closedTs;
      return c !== undefined && ts <= c;
    };
    const trades = this.allTrades();
    for (const t of trades) {
      if (t.side === 'BUY') {
        if (closedBefore(t.code, t.ts)) continue;
        const cur = pos.get(t.code);
        const qty = Number.isFinite(t.qty) ? t.qty : 0;
        if (cur && cur.qty > 0) cur.qty += qty;
        else pos.set(t.code, { qty, buyPrice: t.price, buyTs: t.ts });
      } else {
        const cur = pos.get(t.code);
        if (cur) {
          cur.qty -= Number.isFinite(t.qty) ? t.qty : cur.qty;
          if (cur.qty <= 0) pos.delete(t.code);
        }
        if (Number.isFinite(t.pnlKrw)) realized.set(t.code, (realized.get(t.code) || 0) + (t.pnlKrw as number));
      }
      this.sym(t.code);
    }
    const symbols: SymbolSummary[] = [];
    const traded = new Set(trades.map((t) => t.code));
    const now = this.lastLineTs || 0;
    for (const [code, s] of this.syms) {
      const sum: SymbolSummary = { ...s.summary, realizedKrw: realized.get(code) || 0 };
      if (s.side.length) sum.side = sideSummaries(s.side); // [2026-11-27] per-method state for the method board
      const p = pos.get(code);
      sum.position = p && p.qty > 0 ? { ...p } : undefined;
      sum.held = sum.position
        ? undefined
        : s.carriedTs !== undefined && now - s.carriedTs <= CARRIED_FRESH_MS
          ? 'carried'
          : s.manualTs !== undefined && now - s.manualTs <= CARRIED_FRESH_MS
            ? 'manual'
          : s.unmanagedTs !== undefined && now - s.unmanagedTs <= UNMANAGED_FRESH_MS
            ? 'unmanaged'
            : undefined;
      // taken off the watchlist before anything happened to it: not a symbol of the day
      if (sum.watch === 'removed' && !s.timeline.length && !s.side.length && !traded.has(code) && !sum.held) continue;
      symbols.push(sum);
    }
    symbols.sort((a, b) => a.code.localeCompare(b.code));
    return {
      date: this.date,
      generatedAt: Date.now(),
      symbols,
      buyStats: Object.fromEntries(this.buyStats),
      sellStats: Object.fromEntries(this.sellStats),
      trades,
      strategyNotes: this.strategyNotes.slice(),
      unknown: [...this.unknown.values()].sort((a, b) => b.count - a.count),
      warnings: this.unknownLines,
      regime: this.regime,
      wl: this.wl,
      lines: this.lineCount,
      lastLineTs: this.lastLineTs,
      files: [],
    };
  }
}
