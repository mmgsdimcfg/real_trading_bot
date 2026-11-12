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
 */

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
import { hhmmToSeconds, kstDateTimeToEpoch, kstSecondOfDay } from './time';

export const TIMELINE_CAP = 2000;

interface SymState {
  summary: SymbolSummary;
  timeline: EvalEvent[];
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
      const w = /^WATCH \| (\d{6}) \| ([^|]+?) \|(.*)$/.exec(p.body);
      if (w) {
        this.setName(w[1], w[2].trim(), true);
        const s = this.sym(w[1]);
        const nxt = /\bNXT=(True|False)\b/.exec(w[3]);
        if (nxt) s.summary.nxt = nxt[1] === 'True';
        if (s.summary.watch === 'unknown') s.summary.watch = 'active';
      }
      return;
    }

    // tagged line without a symbol: global state
    if (!p.code) {
      if (p.tag === 'MKT REGIME') this.regime = p.body.split('|')[0].trim();
      else if (p.tag === 'WL SPLIT') {
        const m = /active=(\d+)\s+backup=(\d+)/.exec(p.body);
        if (m) this.wl = { active: +m[1], backup: +m[2] };
      }
      return;
    }

    // ignored tags (except ACT state changes) never introduce a symbol - e.g. [WARMUP] of the market-regime ETFs
    if (!this.syms.has(p.code) && IGNORED_TAGS.has(p.tag) && !p.tag.startsWith('ACT ')) return;
    const s = this.sym(p.code);
    if (p.name) this.setName(p.code, p.name);

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

  private push(s: SymState, ts: number, code: string, tag: string, c: Classified, raw: string): void {
    const ev: EvalEvent = { id: this.nextId++, ts, code, tag, raw, ...c };
    s.timeline.push(ev);
    if (s.timeline.length > this.cap + 200) s.timeline.splice(0, s.timeline.length - this.cap);
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
      if (t.side === 'SELL') {
        t.pnlKrw = num(kv['수익금액']);
        t.pnlPct = num(kv['수익율']);
        const k = sellReasonToCond(reason);
        t.condKey = k ? remapKey(this.remap.sell, k) : 'UNKNOWN';
      }
      if (name) this.setName(code, name);
      this.trades.push(t);
      this.openTrade = t;
      return;
    }
    const d = BS_DETAIL_RE.exec(p.body);
    if (d && this.openTrade) {
      const t = this.openTrade;
      const kv = parseKv(d[1]);
      t.raw += '\n' + raw;
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

  getTrades(): Trade[] {
    return this.trades;
  }

  timeline(code: string): { events: EvalEvent[]; trades: Trade[] } {
    const s = this.syms.get(code);
    const events = s ? s.timeline.slice(-this.cap) : [];
    return { events, trades: this.trades.filter((t) => t.code === code) };
  }

  snapshot(): Snapshot {
    // positions and realized P&L from the trade log (buy_sell log is the source of truth for fills)
    const pos = new Map<string, { qty: number; buyPrice: number; buyTs: number }>();
    const realized = new Map<string, number>();
    for (const t of this.trades) {
      if (t.side === 'BUY') {
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
    for (const [code, s] of this.syms) {
      const sum = { ...s.summary, realizedKrw: realized.get(code) || 0 };
      const p = pos.get(code);
      sum.position = p && p.qty > 0 ? { ...p } : undefined;
      symbols.push(sum);
    }
    symbols.sort((a, b) => a.code.localeCompare(b.code));
    return {
      date: this.date,
      generatedAt: Date.now(),
      symbols,
      buyStats: Object.fromEntries(this.buyStats),
      sellStats: Object.fromEntries(this.sellStats),
      trades: this.trades,
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
