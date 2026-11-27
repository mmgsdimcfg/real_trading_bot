/*
 * Update log
 * - [2026-11-27] type=feat - "매매 방식 현황" (user 2026-11-27: which symbols does the VWAP trade look at, and one place per method for
 *   its symbols / state / results): the rows of one method's symbol table. Display only - a row says where the symbol comes from
 *   (watched / tick collector / only held or traded) and the LAST thing the bot wrote about it for that method (rule: the last
 *   condition evaluation; Edge AI / VWAP: the last judged bar of the r030 trace, the last log line, or the paper runner's status,
 *   whichever is newest). Nothing is judged here; a symbol the bot wrote nothing about says "기록 없음" [Codex review: the list is
 *   "watched + recorded", not a claim about what the bot will evaluate]
 */

import type { PaperStatusSymbol, PaperTrade } from './paper';
import { PAPER_STAGE_KO } from './paper';
import { evalRowKo, STAGE_KO, type StrategyEvalRowX } from './sideTimeline';
import { stageResult, stageRowOf, type SideStrategy, type StrategyEvent } from './strategyEvents';
import { strategyOfTrade, type TradeStrategy } from './strategy';
import type { SymbolSummary, Trade } from './types';

/** what the side strategies wrote about one symbol today, per strategy (kept on the snapshot's symbol summary) */
export interface SideSymSummary {
  /** the last line that is not a routine holding check (those are in `holding`) */
  last?: StrategyEvent;
  /** the holding the strategy reports: the last check line with no SALE after it. An exit decision alone does not end it - the
   *  sell order may not fill (2026-11-27: the 15:20 close order of 237690 was not filled); `exitAt` says a decision was written */
  holding?: StrategyEvent;
  exitAt?: number;
  signals: number;
  orders: number;
  nobuys: number;
  paperBuys: number;
}

export function sideSummaries(events: StrategyEvent[]): Partial<Record<SideStrategy, SideSymSummary>> {
  const out: Partial<Record<SideStrategy, SideSymSummary>> = {};
  for (const e of events) {
    const s = (out[e.strategy] ||= { signals: 0, orders: 0, nobuys: 0, paperBuys: 0 });
    if (e.kind === 'hold') {
      s.holding = e;
      s.exitAt = undefined;
      continue;
    }
    s.last = e;
    if (e.kind === 'signal') s.signals++;
    else if (e.kind === 'order') s.orders++;
    else if (e.kind === 'nobuy') s.nobuys++;
    else if (e.kind === 'paper_buy') s.paperBuys++;
    if (e.kind === 'exit' && s.holding) s.exitAt = e.ts;
    if ((e.kind === 'fill' && e.reason === 'sell') || e.kind === 'gone') {
      s.holding = undefined;
      s.exitAt = undefined;
    }
  }
  return out;
}

export type MethodRowKind = 'pass' | 'fail' | 'hold' | 'info' | 'none';
/** where the row's state text comes from */
export type MethodRowSrc = 'rule' | 'trace' | 'log' | 'paper' | '';

export interface MethodRow {
  code: string;
  name: string;
  nxt?: boolean;
  /** r003 watches it now (the method's evaluation starts from these) */
  watched: boolean;
  /** tick-collector symbol of the Edge AI paper runner */
  collector: boolean;
  /** on the list only because it is held / was traded / has a record (not counted as a target) */
  extra: boolean;
  state: string;
  stateKind: MethodRowKind;
  stateSrc: MethodRowSrc;
  stateTs?: number;
  /** the record is of the real-order path or of the paper runner (side methods; a paper candidate is not a real one) */
  stateMode?: 'live' | 'paper';
  /** the record's own text (tooltip) */
  stateRaw?: string;
  /** bars judged today (trace) / condition evaluations (rule) */
  evals: number;
  signals: number;
  orders: number;
  /** real holding of this method: quantity @ average as the bot printed (rule: today's fills) */
  holdQty?: number;
  holdAvg?: number;
  holdPnlPct?: number;
  /** time of the holding check the holding comes from, and of the method's last real sale */
  holdTs?: number;
  lastSellTs?: number;
  /** an exit was decided after the last holding check and no sale is recorded since (time of the decision) */
  holdExitAt?: number;
  /** open paper position (entry price) */
  paperEntry?: number;
  buys: number;
  sells: number;
  /** shares bought / sold today by this method's real trades (a partly sold position is still held) */
  buyQty: number;
  sellQty: number;
  buyAmt: number;
  realizedKrw: number;
  paperBuys: number;
  paperSells: number;
  /** sum of the net % of the day's closed paper trades */
  paperNetPct: number;
}

export interface MethodBoardInput {
  symbols: SymbolSummary[];
  /** the day's real trades (all methods) */
  trades: Trade[];
  /** rule: title of a buy condition key */
  condTitle?: (key: string) => string;
  /** rule: Korean text of a reason code ('' = none) */
  reasonKo?: (reason: string) => string;
  /** side: last trace row per code + rows per code */
  evalLast?: Record<string, StrategyEvalRowX>;
  evalCount?: Record<string, number>;
  paperStatus?: PaperStatusSymbol[];
  /** [2026-11-27] the method places REAL orders now: its paper runner only closes what it had open, so the stage it writes is
   *  not the method's state (user 2026-11-27: "실제 매매인데 왜 가상 러너 상태인가") - positions / groups are still read */
  paperStateOff?: boolean;
  paperTrades?: PaperTrade[];
}

const RESULT_KO: Record<string, string> = { pass: '전부 통과', fail: '막힘', hold: '대기', info: '' };
const EVENT_KO: Record<string, string> = {
  signal: '매수 신호',
  order: '매수 주문 접수',
  fill: '체결 반영',
  nobuy: '신호는 났지만 사지 않음',
  exit: '청산 결정',
  gone: '봇 보유 종료 (봇 장부: 봇 몫 0)',
  hand_sell: '직접 매도 기록',
  paper_buy: '가상 매수',
  paper_sell: '가상 매도',
  note: '',
};
const EVENT_KIND: Record<string, MethodRowKind> = { hand_sell: 'info', gone: 'info', signal: 'pass', order: 'pass', fill: 'info', nobuy: 'fail', exit: 'info', paper_buy: 'pass', paper_sell: 'info', note: 'info' };

const blank = (code: string, name: string): MethodRow => ({
  code,
  name,
  watched: false,
  collector: false,
  extra: false,
  state: '기록 없음',
  stateKind: 'none',
  stateSrc: '',
  evals: 0,
  signals: 0,
  orders: 0,
  buys: 0,
  sells: 0,
  buyQty: 0,
  sellQty: 0,
  buyAmt: 0,
  realizedKrw: 0,
  paperBuys: 0,
  paperSells: 0,
  paperNetPct: 0,
});

function addTrades(method: TradeStrategy, trades: Trade[], get: (code: string, name: string) => MethodRow): void {
  for (const t of trades) {
    if (t.paper || strategyOfTrade(t) !== method) continue;
    const r = get(t.code, t.name);
    const q = Number.isFinite(t.qty) ? t.qty : 0;
    if (t.side === 'BUY') {
      r.buys++;
      r.buyQty += q;
      if (Number.isFinite(t.price)) r.buyAmt += q * t.price;
    } else {
      r.sells++;
      r.sellQty += q;
      if (r.lastSellTs === undefined || t.ts > r.lastSellTs) r.lastSellTs = t.ts;
      if (Number.isFinite(t.pnlKrw)) r.realizedKrw += t.pnlKrw as number;
    }
  }
}

/** rows of one method, sorted: holdings first, then symbols with a signal / order, then by code */
export function methodRows(method: TradeStrategy, input: MethodBoardInput): MethodRow[] {
  const rows = new Map<string, MethodRow>();
  const byCode = new Map(input.symbols.map((s) => [s.code, s]));
  const get = (code: string, name: string): MethodRow => {
    let r = rows.get(code);
    if (!r) {
      r = blank(code, name || byCode.get(code)?.name || '');
      r.nxt = byCode.get(code)?.nxt;
      r.extra = true; // until a source says it is a target
      rows.set(code, r);
    } else if (!r.name && name) r.name = name;
    return r;
  };
  for (const s of input.symbols) {
    if (s.watch !== 'active') continue;
    const r = get(s.code, s.name);
    r.watched = true;
    r.extra = false;
  }

  if (method === 'rule') {
    for (const s of input.symbols) {
      const ruleHeld = !!s.position;
      // a refused / failed order is written after the evaluation that passed: the newer record is the state [Codex round 2]
      const ev = s.lastOrder && (!s.lastEval || s.lastOrder.ts >= s.lastEval.ts) ? s.lastOrder : s.lastEval;
      if (s.watch !== 'active' && !ev && !ruleHeld) continue;
      const r = get(s.code, s.name);
      r.evals = s.evalCount;
      if (ev) {
        const cond = ev.condKey === 'ORDER' ? '주문 단계' : input.condTitle ? input.condTitle(ev.condKey) : ev.condKey;
        const why = ev.reason ? (input.reasonKo && input.reasonKo(ev.reason)) || ev.reason : '';
        r.state = [RESULT_KO[ev.result] ?? ev.result, cond, why].filter(Boolean).join(' · ');
        r.stateKind = ev.result === 'pass' || ev.result === 'fail' || ev.result === 'hold' ? ev.result : 'info';
        r.stateSrc = 'rule';
        r.stateTs = ev.ts;
        r.stateRaw = ev.raw;
      }
    }
    addTrades('rule', input.trades, get);
    // rule holdings: today's fills of rule buys that are still open (the snapshot's position is all of today's fills of the symbol)
    // only the shares the rule trade bought and has not sold (the snapshot's position mixes every method's fills of the symbol)
    for (const r of rows.values()) {
      const left = r.buyQty - r.sellQty;
      if (left > 0 && byCode.get(r.code)?.position) {
        r.holdQty = left;
        r.holdAvg = r.buyQty > 0 && r.buyAmt > 0 ? r.buyAmt / r.buyQty : undefined;
      }
      r.orders = r.buys; // filled buy orders; refused ones are in the state column (the bot's last order record)
    }
  } else {
    const side = method as SideStrategy;
    for (const p of input.paperStatus || []) {
      const r = get(p.code, p.name);
      if (p.group === 'collector') {
        r.collector = true;
        r.extra = false;
      }
      if (p.position) r.paperEntry = p.position.entry;
      if (!input.paperStateOff && p.stage && p.at !== undefined && (r.stateTs === undefined || p.at > r.stateTs)) {
        const key = stageRowOf(side, { stage: p.stage });
        r.state = `${PAPER_STAGE_KO[p.stage] || STAGE_KO[p.stage] || p.stage}${p.prob !== undefined ? ` · 확률 ${p.prob.toFixed(3)}` : ''}${p.bar ? ` · ${p.bar} 봉` : ''}`;
        r.stateKind = p.position ? 'hold' : stageResult(key);
        r.stateSrc = 'paper';
        r.stateMode = 'paper';
        r.stateTs = p.at;
        r.stateRaw = `stage=${p.stage}`;
      }
    }
    for (const s of input.symbols) {
      const sum = (s.side as Partial<Record<SideStrategy, SideSymSummary>> | undefined)?.[side];
      if (!sum) continue;
      const r = get(s.code, s.name);
      r.signals = sum.signals;
      r.orders = sum.orders;
      r.paperBuys = sum.paperBuys;
      if (sum.holding) {
        r.holdQty = sum.holding.qty;
        r.holdAvg = sum.holding.avg;
        r.holdPnlPct = sum.holding.pnlPct;
        r.holdTs = sum.holding.ts;
        r.holdExitAt = sum.exitAt;
      }
      const e = sum.last;
      // at the same time the log line wins over a trace row: it is the more specific record (signal / order / refusal)
      if (e && (r.stateTs === undefined || e.ts >= r.stateTs)) {
        const ko = EVENT_KO[e.kind] ?? '';
        r.state = [ko, e.kind === 'note' || e.kind === 'gone' || e.kind === 'hand_sell' || !ko ? e.text : e.reason || ''].filter(Boolean).join(' · ');
        r.stateKind = EVENT_KIND[e.kind] || 'info';
        r.stateSrc = 'log';
        r.stateMode = e.mode;
        r.stateTs = e.ts;
        r.stateRaw = e.raw;
      }
    }
    for (const [code, row] of Object.entries(input.evalLast || {})) {
      const r = get(code, '');
      r.evals = input.evalCount?.[code] || 0;
      if (r.stateTs === undefined || row.ts > r.stateTs) {
        r.state = evalRowKo(row);
        r.stateMode = row.mode;
        r.stateKind = stageResult(stageRowOf(side, row));
        r.stateSrc = 'trace';
        r.stateTs = row.ts;
        r.stateRaw = JSON.stringify({ stage: row.stage, no: row.no, detail: row.detail, ...row.extra });
      }
    }
    addTrades(side, input.trades, get);
    // a recorded sale of everything bought ends the holding the log lines left open
    for (const r of rows.values()) {
      // ... unless the bot printed a holding check after that sale (shares carried over from an earlier day) [Codex review]
      if (r.holdQty && r.sellQty > 0 && r.sellQty >= r.buyQty && (r.holdTs === undefined || r.lastSellTs === undefined || r.lastSellTs >= r.holdTs)) {
        r.holdQty = r.holdAvg = r.holdPnlPct = r.holdExitAt = undefined;
      }
    }
    if (input.paperTrades) for (const r of rows.values()) r.paperBuys = 0; // the paper book itself counts its buys
    for (const t of input.paperTrades || []) {
      const r = get(t.code, t.name);
      r.paperBuys++;
      if (t.group === 'collector') r.collector = true; // where the trade came from; a target only while the runner's status lists it
      if (t.exitTs === undefined) {
        r.paperEntry = r.paperEntry ?? t.entry;
      } else {
        r.paperSells++;
        r.paperNetPct += t.netPct ?? 0;
      }
    }
    for (const r of rows.values()) if (r.holdQty && r.stateKind !== 'pass') r.stateKind = r.stateKind === 'none' ? 'hold' : r.stateKind;
  }

  const rank = (r: MethodRow) => (r.holdQty || r.paperEntry !== undefined ? 0 : r.signals + r.orders + r.buys + r.paperBuys > 0 ? 1 : r.extra ? 3 : 2);
  return [...rows.values()].sort((a, b) => rank(a) - rank(b) || a.code.localeCompare(b.code));
}

export interface MethodTotals {
  /** symbols the method starts from (watched + collector), not the extra rows */
  targets: number;
  watched: number;
  collector: number;
  recorded: number;
  holding: number;
  paperHolding: number;
  signals: number;
  orders: number;
  buys: number;
  sells: number;
  realizedKrw: number;
  paperBuys: number;
  paperSells: number;
}

export function methodTotals(rows: MethodRow[]): MethodTotals {
  const t: MethodTotals = { targets: 0, watched: 0, collector: 0, recorded: 0, holding: 0, paperHolding: 0, signals: 0, orders: 0, buys: 0, sells: 0, realizedKrw: 0, paperBuys: 0, paperSells: 0 };
  for (const r of rows) {
    if (!r.extra) t.targets++;
    if (r.watched) t.watched++;
    if (r.collector) t.collector++;
    if (r.stateSrc) t.recorded++;
    if (r.holdQty) t.holding++;
    if (r.paperEntry !== undefined) t.paperHolding++;
    t.signals += r.signals;
    t.orders += r.orders;
    t.buys += r.buys;
    t.sells += r.sells;
    t.realizedKrw += r.realizedKrw;
    t.paperBuys += r.paperBuys;
    t.paperSells += r.paperSells;
  }
  return t;
}

export type MethodRowFilter = 'all' | 'target' | 'recorded' | 'holding' | 'active';
export const METHOD_FILTER_KO: Record<MethodRowFilter, string> = { all: '전체', target: '감시 대상', recorded: '오늘 기록 있음', holding: '보유', active: '신호 · 주문 있음' };
export function filterMethodRows(rows: MethodRow[], f: MethodRowFilter): MethodRow[] {
  switch (f) {
    case 'target':
      return rows.filter((r) => !r.extra);
    case 'recorded':
      return rows.filter((r) => !!r.stateSrc);
    case 'holding':
      return rows.filter((r) => !!r.holdQty || r.paperEntry !== undefined);
    case 'active':
      return rows.filter((r) => r.signals + r.orders + r.buys + r.paperBuys > 0);
    default:
      return rows;
  }
}

export const METHOD_SRC_KO: Record<MethodRowSrc, string> = { rule: '조건 평가', trace: '봉별 판정', log: '봇 로그', paper: '가상 러너 상태', '': '' };
