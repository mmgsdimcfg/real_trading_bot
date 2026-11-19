// Update log
// - [2026-11-06] type=feat (Live chart update from realtime trades: KIS trade time "HHMMSS" -> epoch (today, KST); applyTrades() rolls trades
//   newer than the REST data into the bars - minute charts by M-minute bucket (start-labelled like KIS minute bars), day / week / month / year
//   update the current period's bar (new period -> new bar); tickBarsFromTrades() builds N-tick bars from the realtime trade list)
// - [2026-11-06] type=feat (acmlMismatches(): flags REST / realtime trades of the same shape with different accumulated volumes)
// - [2026-11-06] type=fix (addRealtimeTrades(): a realtime batch is merged in one pass (the 30,000-trade cache was copied once per trade)
//   and a REST copy is found anywhere in the cache by acml (the search stopped after 500 trades and doubled older ones))
// - [2026-11-06] type=feat (Trades with an accumulated volume (acml) are identified and ordered by it - REST back-fill pages and realtime
//   trades merge exactly; the cache holds up to 30,000 trades (TRADE_CAP) for tick charts)
// - [2026-11-06] type=fix (Trade identity includes the trade strength: a separate realtime trade with the same second, price and volume as a
//   REST trade no longer replaces it)
// - [2026-11-06] type=fix (mergeTrades() keeps the REST order for same-second trades and marks REST trades; addRealtimeTrade() replaces a
//   REST copy of a realtime trade instead of adding it twice)
// - [2026-11-06] type=fix (mergeTrades(): REST trade lists (inquire_ccnl, newest 30) merged into the trade cache without duplicates, so tick
//   charts fill even when the realtime connection is down)

import type { Bar, ChartKind } from './price';
import type { TradeTick } from './kis';

const KST = 9 * 3600000;
const DAY = 86400000;

/** today's KST midnight (epoch ms) */
export function kstToday(now = Date.now()): number {
  return Math.floor((now + KST) / DAY) * DAY - KST;
}

/** "HHMMSS" of today (KST) -> epoch ms */
export function tradeTime(hhmmss: string, now = Date.now()): number {
  const s = (hhmmss || '').padStart(6, '0');
  return kstToday(now) + (+s.slice(0, 2) * 3600 + +s.slice(2, 4) * 60 + +s.slice(4, 6)) * 1000;
}

function periodStart(kind: ChartKind, t: number): number {
  const d = new Date(t + KST);
  if (kind === 'day') return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) - KST;
  if (kind === 'week') {
    const dow = (d.getUTCDay() + 6) % 7;
    return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() - dow) - KST;
  }
  if (kind === 'month') return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1) - KST;
  return Date.UTC(d.getUTCFullYear(), 0, 1) - KST;
}

/**
 * Roll realtime trades (newest first, as stored) into bars. Only trades at or after `since` (epoch ms of the newest REST bar data)
 * are applied, so a REST refresh followed by the same trades does not double the volume.
 */
export function applyTrades(bars: Bar[], trades: TradeTick[], kind: ChartKind, n: number, since: number, now = Date.now()): Bar[] {
  if (!trades.length || kind === 'tick') return bars;
  const out = bars.slice();
  const span = Math.max(1, n) * 60000;
  for (let i = trades.length - 1; i >= 0; i--) {
    const tr = trades[i];
    const t = tradeTime(tr.time, now);
    if (t < since || !(tr.price > 0)) continue;
    let start: number;
    if (kind === 'min') {
      const day = kstToday(t);
      start = day + Math.floor((t - day) / span) * span;
    } else start = periodStart(kind, t);
    const last = out[out.length - 1];
    if (last && last.t === start) {
      out[out.length - 1] = { ...last, h: Math.max(last.h, tr.price), l: Math.min(last.l, tr.price), c: tr.price, v: last.v + tr.volume };
    } else if (!last || start > last.t) {
      out.push({ t: start, o: tr.price, h: tr.price, l: tr.price, c: tr.price, v: tr.volume });
    }
  }
  return out;
}

/** identity of a trade: time, price, volume and the trade strength after it (REST tday_rltv / realtime CTTR - it moves with every trade,
 *  so two separate same-second trades of the same price and volume still differ) */
const tradeKey = (t: TradeTick) => `${t.time}|${t.price}|${t.volume}|${t.strength !== undefined && Number.isFinite(t.strength) ? t.strength.toFixed(2) : ''}`;

/**
 * Merge `incoming` trades (newest first, e.g. a REST trade list) into `list` (newest first). Cached trades matching an incoming one (same
 * time, price and volume, one to one) are kept; the rest of `incoming` is added marked `rest`. Inside the incoming time range the incoming
 * order is authoritative (same-second trades keep the server's order); cached trades the list did not match go before it at its newest
 * second (they arrived after the fetch) and after its trades of the same second otherwise. Returns `list` itself when nothing is new.
 */
export const TRADE_CAP = 30000;

/** newest first: later time, then (same second) larger accumulated volume; trades without acml keep their order */
const newerFirst = (a: TradeTick, b: TradeTick) => (a.time !== b.time ? (a.time < b.time ? 1 : -1) : a.acml && b.acml ? b.acml - a.acml : 0);

export function mergeTrades(list: TradeTick[], incoming: TradeTick[], max = TRADE_CAP): TradeTick[] {
  const inc = incoming.filter((t) => t.time && t.price > 0);
  if (!inc.length) return list;
  if (inc.every((t) => t.acml)) {
    // accumulated volume identifies a trade exactly: add the unknown ones and order by (time, acml)
    const have = new Set(list.map((t) => t.acml).filter(Boolean));
    const add: TradeTick[] = [];
    for (const t of inc) {
      if (have.has(t.acml)) continue;
      have.add(t.acml);
      add.push({ ...t, rest: true });
    }
    if (!add.length) return list;
    const out = [...list, ...add].sort(newerFirst);
    return out.length > max ? out.slice(0, max) : out;
  }
  let newest = inc[0].time;
  let oldest = inc[0].time;
  for (const t of inc) {
    if (t.time > newest) newest = t.time;
    if (t.time < oldest) oldest = t.time;
  }
  const newer: TradeTick[] = [];
  const overlap: TradeTick[] = [];
  const older: TradeTick[] = [];
  for (const t of list) (t.time > newest ? newer : t.time < oldest ? older : overlap).push(t);
  const used = new Set<number>();
  let added = 0;
  const seq: TradeTick[] = inc.map((t) => {
    const k = tradeKey(t);
    const i = overlap.findIndex((o, j) => !used.has(j) && tradeKey(o) === k);
    if (i >= 0) {
      used.add(i);
      return overlap[i];
    }
    added++;
    return { ...t, rest: true };
  });
  if (!added) return list;
  const left = overlap.filter((_, j) => !used.has(j));
  const atNewest = left.filter((t) => t.time === newest);
  const rest = left.filter((t) => t.time !== newest);
  const mid: TradeTick[] = [];
  let a = 0;
  let b = 0;
  while (a < seq.length || b < rest.length) {
    if (b >= rest.length || (a < seq.length && seq[a].time >= rest[b].time)) mid.push(seq[a++]);
    else mid.push(rest[b++]);
  }
  const out = [...newer, ...atNewest, ...mid, ...older];
  return out.length > max ? out.slice(0, max) : out;
}

/**
 * Add one realtime trade to `list` (newest first). When a REST copy of it is cached (same time, price and volume, marked `rest`), the
 * realtime trade replaces that copy in place instead of being added a second time.
 */
export function addRealtimeTrade(list: TradeTick[], trade: TradeTick, max = TRADE_CAP): TradeTick[] {
  if (trade.acml) return addRealtimeTrades(list, [trade], max);
  const k = tradeKey(trade);
  const lim = Math.min(list.length, 300);
  for (let i = 0; i < lim; i++) {
    const t = list[i];
    if (t.rest && tradeKey(t) === k) {
      const out = list.slice();
      out[i] = trade;
      return out;
    }
    if (t.time < trade.time) break;
  }
  const out = [trade, ...list];
  return out.length > max ? out.slice(0, max) : out;
}

/**
 * Add one realtime batch (several trades of one code) to `list` (newest first) in one pass: trades with an accumulated volume replace
 * their REST copy anywhere in the list (realtime has the trade side) or are added in (time, acml) order; the cap is applied once.
 * The common case - every new trade is newer than the list head - is a plain prepend without sorting.
 */
export function addRealtimeTrades(list: TradeTick[], batch: TradeTick[], max = TRADE_CAP): TradeTick[] {
  if (!batch.length) return list;
  let out = list;
  const keyed = batch.filter((t) => t.acml);
  if (keyed.length) {
    const byAcml = new Map(keyed.map((t) => [t.acml as number, t]));
    let replaced = false;
    const rep = list.map((t) => {
      const r = t.acml ? byAcml.get(t.acml) : undefined;
      if (!r) return t;
      byAcml.delete(t.acml as number);
      replaced = true;
      return r;
    });
    const fresh = [...byAcml.values()].sort(newerFirst);
    if (!fresh.length) out = replaced ? rep : list;
    else if (!rep.length || newerFirst(fresh[fresh.length - 1], rep[0]) <= 0) out = [...fresh, ...rep];
    else out = [...fresh, ...rep].sort(newerFirst);
  }
  for (const t of batch) if (!t.acml) out = addRealtimeTrade(out, t, Infinity);
  return out.length > max ? out.slice(0, max) : out;
}

/**
 * Diagnostic: trades that look identical (same second, price, volume) in REST and realtime form but carry different accumulated volumes -
 * a sign the two sources count acml on different bases (KRX only vs KRX + NXT), which would double trades. Counts such shapes among the
 * newest `scan` trades.
 */
export function acmlMismatches(list: TradeTick[], scan = 3000): number {
  const shapes = new Map<string, { rest: Set<number>; rt: Set<number> }>();
  for (const t of list.slice(0, scan)) {
    if (!t.acml) continue;
    const k = `${t.time}|${t.price}|${t.volume}`;
    let e = shapes.get(k);
    if (!e) shapes.set(k, (e = { rest: new Set(), rt: new Set() }));
    (t.rest ? e.rest : e.rt).add(t.acml);
  }
  let n = 0;
  for (const e of shapes.values()) if (e.rest.size && e.rt.size && e.rest.size + e.rt.size > Math.max(e.rest.size, e.rt.size) && ![...e.rt].some((a) => e.rest.has(a))) n++;
  return n;
}

/** N-tick bars from realtime trades (newest first) */
export function tickBarsFromTrades(trades: TradeTick[], n: number, now = Date.now()): Bar[] {
  const size = Math.max(1, Math.floor(n));
  const asc = trades.slice().reverse();
  const out: Bar[] = [];
  for (let i = 0; i < asc.length; i += size) {
    const seg = asc.slice(i, i + size);
    const t = tradeTime(seg[0].time, now);
    out.push({
      t,
      t2: tradeTime(seg[seg.length - 1].time, now),
      o: seg[0].price,
      h: Math.max(...seg.map((s) => s.price)),
      l: Math.min(...seg.map((s) => s.price)),
      c: seg[seg.length - 1].price,
      v: seg.reduce((a, s) => a + s.volume, 0),
    });
  }
  return out;
}
