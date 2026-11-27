/*
 * Update log
 * - [2026-11-27] type=feat - The timeline of a side strategy (Edge AI / VWAP) in the shape the timeline's evaluation charts draw:
 *   rows bottom-up + dots, from the strategy's log events (shared/strategyEvents) and the per-bar trace rows the bot writes
 *   (strategy_eval_<strategy>_<date>.jsonl). Also the paper trades as chart markers and which method tab to open first.
 *   Display only - every text is a value the bot wrote; nothing is judged or recomputed here
 */

import { PAPER_EXIT_KO, PAPER_STAGE_KO, type PaperTrade } from './paper';
import type { TradeStrategy } from './strategy';
import { STAGE_ROWS, stageResult, stageRowOf, type SideStrategy, type StrategyEvalRow, type StrategyEvent, type StrategyRunMode } from './strategyEvents';
import type { EvalEvent, Trade } from './types';

export const STAGE_KO: Record<string, string> = {
  ...PAPER_STAGE_KO,
  cond_before: '공통 안전 조건에서 멈춤',
  cond_after: '수량 · 가격 · 호가 확인에서 멈춤',
  ordered: '매수 주문 접수',
  order_refused: '신호는 났지만 주문이 나가지 않음',
  signal: '매수 신호',
  not_promoted: '승격되지 않은 모델 (실제 주문 안 함)',
  error: '판정 오류',
};
const MODE_KO: Record<StrategyRunMode, string> = { live: '실제', paper: '가상' };
const n0 = (v: number) => Math.round(v).toLocaleString('ko-KR');
const sp = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`;

export interface StrategyEvalRowX extends StrategyEvalRow {
  /** values the bot used for that bar, as written (vwap, gap, tick, prob, thr, bar ...) */
  extra?: Record<string, unknown>;
}

/** the trace file's rows with every key the bot wrote (the known ones typed, the rest in `extra`) */
export function parseStrategyEvalRows(text: string): StrategyEvalRowX[] {
  const out: StrategyEvalRowX[] = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      const o = JSON.parse(s) as Record<string, unknown>;
      const ts = Date.parse(String(o.t || ''));
      const code = String(o.code || '');
      if (!Number.isFinite(ts) || !/^[0-9A-Z]{6}$/.test(code) || typeof o.stage !== 'string') continue;
      const extra: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(o)) if (!['t', 'code', 'mode', 'stage', 'no', 'detail', 'price'].includes(k)) extra[k] = v;
      out.push({
        ts,
        code,
        mode: o.mode === 'paper' ? 'paper' : 'live',
        stage: o.stage,
        no: typeof o.no === 'number' ? o.no : undefined,
        detail: typeof o.detail === 'string' ? o.detail : undefined,
        price: typeof o.price === 'number' ? o.price : undefined,
        extra,
      });
    } catch {
      /* a half-written row */
    }
  }
  return out;
}

/** what the bot saw on a judged bar, for people - only values of the row itself */
export function evalRowKo(r: StrategyEvalRowX): string {
  const x = r.extra || {};
  const parts: string[] = [STAGE_KO[r.stage.toLowerCase()] || r.stage];
  if (typeof x.vwap === 'number') parts.push(`VWAP ${n0(x.vwap)}`);
  if (typeof x.gap === 'number') parts.push(`VWAP 대비 ${sp(x.gap)}`);
  if (typeof x.tick === 'string' && x.tick) parts.push(`틱 ${x.tick}`);
  if (typeof x.tick_cross === 'string' && x.tick_cross) parts.push(`크로스 ${x.tick_cross.slice(11, 19)}`);
  if (typeof x.bar_age === 'number') parts.push(`봉 지연 ${Math.round(x.bar_age)}초`);
  if (typeof x.prob === 'number') parts.push(`확률 ${x.prob.toFixed(3)}${typeof x.thr === 'number' ? ` / 기준 ${x.thr.toFixed(3)}` : ''}`);
  if (typeof r.price === 'number') parts.push(`현재가 ${n0(r.price)}`);
  if (typeof x.cond === 'string' && x.cond) parts.push(`조건 ${x.cond}`);
  if (r.detail) parts.push(r.detail);
  if (typeof x.reason === 'string' && x.reason) parts.push(x.reason);
  return parts.join(' · ');
}

export interface SideTimeline {
  /** rows bottom-up, like the rule chart's */
  buyRows: { key: string; label: string }[];
  sellRows: { key: string; label: string }[];
  events: EvalEvent[];
  /** the last judged bar (trace) of the symbol */
  lastEval?: StrategyEvalRowX;
  /** the holding the strategy reports now: the last check line after which no exit / sale was written */
  holding?: StrategyEvent;
  /** rows of the per-bar trace exist for this symbol (false: only the events the bot logged) */
  traced: boolean;
}
const SELL_ROWS = [
  { key: 'S:hold', label: '보유 점검' },
  { key: 'S:exit', label: '청산' },
];
/** "매수 안 함 | _023_affordable_buy_qty | ..." -> the row of that condition: before the signal (< 9) or after it */
const nobuyRow = (reason: string | undefined): string => {
  const m = /^_[a-z]?(\d{2,3})_/.exec(reason || '');
  return m && +m[1] < 9 ? 'before' : 'after';
};

/** the strategy's events + trace rows of one symbol as chart events. `upTo`: replay cursor (trace rows after it are left out;
 *  the engine already cuts the events). */
export function sideTimeline(strategy: SideStrategy, events: StrategyEvent[], evals: StrategyEvalRowX[], upTo: number | null = null): SideTimeline {
  const rows = STAGE_ROWS[strategy];
  const out: EvalEvent[] = [];
  let id = -1;
  const label = new Map(rows.map((r) => [r.key, r.label]));
  const L = (k: string) => label.get(k) || k;
  const push = (ts: number, code: string, side: 'buy' | 'sell', key: string, result: EvalEvent['result'], mode: StrategyRunMode, reason: string, ko: string, raw: string, rowLabel: string) =>
    out.push({ id: id--, ts, code, side, tag: MODE_KO[mode], condKey: `S:${key}`, result, reason, raw, ko, rowLabel });
  const mine = evals.filter((r) => upTo === null || r.ts <= upTo);
  for (const r of mine) {
    const key = stageRowOf(strategy, r);
    push(r.ts, r.code, 'buy', key, stageResult(key), r.mode, r.stage, evalRowKo(r), JSON.stringify({ stage: r.stage, no: r.no, detail: r.detail, price: r.price, ...r.extra }), L(key));
  }
  let holding: StrategyEvent | undefined;
  for (const e of events) {
    if (e.strategy !== strategy) continue;
    switch (e.kind) {
      case 'signal':
        push(e.ts, e.code, 'buy', 'signal', 'pass', e.mode, e.reason || '', `매수 신호${e.price ? ` · 현재가 ${n0(e.price)}` : ''}${e.qty ? ` · ${e.qty}주까지` : ''} · ${e.reason || ''}`, e.raw, L('signal'));
        break;
      case 'order':
        push(e.ts, e.code, 'buy', 'order', 'pass', e.mode, 'ORDER', `매수 주문 접수 · ${e.text.split('|').slice(1).join('|').trim()}`, e.raw, L('order'));
        break;
      case 'paper_buy':
        push(e.ts, e.code, 'buy', 'order', 'pass', e.mode, e.reason || '', `가상 매수${e.price ? ` ${n0(e.price)}` : ''} · ${e.reason || ''}`, e.raw, L('order'));
        break;
      case 'nobuy': {
        const k = nobuyRow(e.reason);
        push(e.ts, e.code, 'buy', k, 'fail', e.mode, e.reason || '', `신호는 났지만 사지 않음 · ${e.reason || ''}`, e.raw, L(k));
        break;
      }
      case 'fill':
        if (e.reason === 'sell') {
          push(e.ts, e.code, 'sell', 'exit', 'info', e.mode, 'FILL', `매도 체결 ${e.qty ?? ''}주 반영`, e.raw, '청산');
          holding = undefined;
        } else push(e.ts, e.code, 'buy', 'order', 'pass', e.mode, 'FILL', `매수 체결 ${e.qty ?? ''}주 반영${e.avg ? ` · 원가 ${n0(e.avg)}` : ''}`, e.raw, L('order'));
        break;
      case 'hold':
        push(
          e.ts,
          e.code,
          'sell',
          'hold',
          'hold',
          e.mode,
          'HOLD',
          `보유 ${e.qty ?? ''}주 · 손익 ${e.pnlPct === undefined ? '-' : sp(e.pnlPct)} · 현재가 ${e.price ? n0(e.price) : '-'} / 원가 ${e.avg ? n0(e.avg) : '-'}` +
            (e.tpPct !== undefined && e.slPct !== undefined ? ` · 익절 ${sp(e.tpPct)} / 손절 ${sp(e.slPct)}` : ''),
          e.raw,
          '보유 점검',
        );
        holding = e;
        break;
      case 'exit':
        push(e.ts, e.code, 'sell', 'exit', 'pass', e.mode, e.reason || '', `청산 결정 · ${e.reason || ''}${e.pnlPct === undefined ? '' : ` · 손익 ${sp(e.pnlPct)}`}${e.price ? ` · 현재가 ${n0(e.price)}` : ''}`, e.raw, '청산');
        // [2026-11-27] a decision is not a sale (the 15:20 close order of 237690 did not fill): the holding ends with a sale
        // the bot recorded or with its ledger going to 0
        break;
      case 'hand_sell':
        push(e.ts, e.code, 'sell', 'exit', 'info', e.mode, 'HAND_SELL', `직접 매도 ${e.qty ?? ''}주 (봇 기록)${e.price ? ` · 매도가 ${n0(e.price)}` : ''}${e.avg ? ` · 원가 ${n0(e.avg)}` : ''}${e.pnlKrw === undefined ? '' : ` · 손익 ${e.pnlKrw > 0 ? '+' : ''}${n0(e.pnlKrw)}원`}${e.pnlPct === undefined ? '' : ` (${sp(e.pnlPct)})`} · 전략 실현 손익 · 한도에는 넣지 않음`, e.raw, '청산');
        break;
      case 'gone':
        push(e.ts, e.code, 'sell', 'exit', 'info', e.mode, 'BOT_LEDGER', `봇 보유 종료 · 봇 장부에서 봇 몫 ${e.qty ?? ''}주 -> 0 (직접 매도 등) · ${e.text}`, e.raw, '청산');
        holding = undefined;
        break;
      case 'paper_sell':
        push(e.ts, e.code, 'sell', 'exit', 'pass', e.mode, e.reason || '', `가상 매도${e.price ? ` ${n0(e.price)}` : ''} · ${PAPER_EXIT_KO[e.reason || ''] || e.reason || ''}${e.pnlPct === undefined ? '' : ` · 순 ${sp(e.pnlPct)}`}`, e.raw, '청산');
        break;
      default:
        break;
    }
  }
  out.sort((a, b) => a.ts - b.ts);
  return {
    buyRows: [...rows].reverse().map((r) => ({ key: `S:${r.key}`, label: r.label })),
    sellRows: SELL_ROWS,
    events: out,
    lastEval: mine.length ? mine[mine.length - 1] : undefined,
    holding,
    traced: mine.length > 0,
  };
}

/** paper trades of the symbol as chart markers (buy at the entry, sell at the exit with its net %) */
export function paperTradesAsTrades(rows: PaperTrade[], code: string, strategy: SideStrategy, upTo: number | null = null): Trade[] {
  const out: Trade[] = [];
  for (const t of rows) {
    if (t.code !== code) continue;
    if (upTo === null || t.entryTs <= upTo)
      out.push({ code, name: t.name, side: 'BUY', ts: t.entryTs, price: t.entry, qty: NaN, reason: `가상 ${t.trigger || ''}`.trim(), raw: '', strategy, paper: true });
    if (t.exitTs !== undefined && (upTo === null || t.exitTs <= upTo))
      out.push({
        code,
        name: t.name,
        side: 'SELL',
        ts: t.exitTs,
        price: t.exit ?? NaN,
        qty: NaN,
        reason: `가상 ${PAPER_EXIT_KO[t.reason || ''] || t.reason || ''}`.trim(),
        raw: '',
        strategy,
        paper: true,
        buyPrice: t.entry,
        buyTs: t.entryTs,
        pnlPct: t.netPct,
      });
  }
  return out.sort((a, b) => a.ts - b.ts);
}

/** which timeline tab to open first for a symbol: the side strategy that holds / ordered it, else the method running real
 *  orders (rule first), else the one running paper, else rule. `side` = the symbol's strategy events (real-order ones count
 *  before paper ones). */
export function defaultTimelineMethod(modes: Record<TradeStrategy, { live: boolean | null; paper: boolean | null }>, side: StrategyEvent[], ruleTraded = false): TradeStrategy {
  const open = (mode: StrategyRunMode): SideStrategy | null => {
    let held: SideStrategy | null = null;
    for (const e of side) {
      if (e.mode !== mode) continue;
      if (e.kind === 'order' || e.kind === 'hold' || e.kind === 'paper_buy' || (e.kind === 'fill' && e.reason !== 'sell')) held = e.strategy;
      else if ((e.kind === 'gone' || e.kind === 'paper_sell' || (e.kind === 'fill' && e.reason === 'sell')) && held === e.strategy) held = null;
    }
    return held;
  };
  const real = open('live');
  if (real) return real;
  // `ruleTraded`: the rule strategy bought the symbol today - its tab, whatever else runs
  if (ruleTraded) return 'rule';
  for (const m of ['rule', 'vwap', 'edge'] as TradeStrategy[]) if (modes[m].live) return m;
  const paper = open('paper');
  if (paper) return paper;
  for (const m of ['vwap', 'edge'] as TradeStrategy[]) if (modes[m].paper) return m;
  return 'rule';
}
