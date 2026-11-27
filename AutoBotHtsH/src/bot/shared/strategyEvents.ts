/*
 * Update log
 * - [2026-11-27] type=feat - What the side strategies (Edge AI / VWAP; real orders and paper) wrote about a symbol, read from the
 *   r003 main log lines "[VWAP LIVE]" / "[EDGE LIVE]" / "[VWAP PAPER]" / "[EDGE ONLY]" and from the per-bar evaluation trace
 *   (server strategy_eval_<strategy>_<date>.jsonl) - for the timeline's method tabs (user 2026-11-27). Display only: nothing is
 *   judged here, every row is something the bot wrote
 */

import { num } from './logLine';
import type { TradeStrategy } from './strategy';

export type SideStrategy = Exclude<TradeStrategy, 'rule'>;
export type StrategyRunMode = 'live' | 'paper';
export type StrategyEventKind =
  | 'signal' // the buy signal of a bar (real)
  | 'order' // buy order accepted
  | 'fill' // fills applied to the strategy's book
  | 'nobuy' // signal conditions passed up to a condition that refused the buy
  | 'hold' // holding check: price / P&L against the exits
  | 'exit' // exit decided (sell order follows)
  | 'hand_sell' // [2026-11-27] the bot's record of the strategy's shares sold by hand (price, cost, result; not in its realized result)
  | 'gone' // [2026-11-27] the bot's ledger says its quantity of the symbol went to 0 ("[BOT LEDGER] ... bot=N->0": sold by hand, or a sale the strategy's own lines did not report)
  | 'paper_buy'
  | 'paper_sell'
  | 'note'; // anything else the strategy wrote about the symbol

export interface StrategyEvent {
  id: number;
  ts: number;
  code: string;
  strategy: SideStrategy;
  mode: StrategyRunMode;
  kind: StrategyEventKind;
  /** the line's own text after the symbol (shown as it is) */
  text: string;
  /** reason code: signal reason, refusing condition, exit reason */
  reason?: string;
  price?: number;
  /** the strategy's average buy price (hold / exit / fill lines) */
  avg?: number;
  qty?: number;
  pnlPct?: number;
  /** result in KRW as the bot printed it (hand_sell: net of costs) */
  pnlKrw?: number;
  /** exits printed on a hold line */
  tpPct?: number;
  slPct?: number;
  raw: string;
}

/** a line of a strategy that names no symbol: why new buys are on / stopped */
export interface StrategyNote {
  /** first time it was said */
  ts: number;
  /** last time the same note was repeated */
  lastTs: number;
  strategy: SideStrategy;
  mode: StrategyRunMode;
  /** 'blocked' = new buys stopped (why), 'on' / 'off' = the start line, 'info' = other */
  kind: 'blocked' | 'on' | 'off' | 'info';
  text: string;
}

export const STRATEGY_TAGS: Record<string, { strategy: SideStrategy; mode: StrategyRunMode }> = {
  'VWAP LIVE': { strategy: 'vwap', mode: 'live' },
  'EDGE LIVE': { strategy: 'edge', mode: 'live' },
  'VWAP PAPER': { strategy: 'vwap', mode: 'paper' },
  'EDGE ONLY': { strategy: 'edge', mode: 'paper' },
};

const pct = (s: string | undefined): number | undefined => {
  const v = s === undefined ? NaN : parseFloat(s);
  return Number.isFinite(v) ? v : undefined;
};
const price = (s: string | undefined): number | undefined => {
  const v = num(s);
  return Number.isFinite(v) ? v : undefined;
};
const seg = (rest: string, i: number): string => (rest.split('|')[i] || '').trim();

/** a symbol line of a strategy tag -> the event fields (without id / ts / code / raw) */
export function parseStrategyRest(mode: StrategyRunMode, rest: string): Pick<StrategyEvent, 'kind' | 'text' | 'reason' | 'price' | 'avg' | 'qty' | 'pnlPct' | 'pnlKrw' | 'tpPct' | 'slPct'> {
  const text = rest.trim();
  const head = seg(text, 0);
  if (mode === 'paper') {
    let m = /^PAPER BUY ([\d,]+)\s*(.*)$/.exec(head);
    if (m) return { kind: 'paper_buy', text, price: price(m[1]), reason: m[2].replace(/^p=\S+\s+thr=\S+\s*/, '').trim() || undefined };
    m = /^PAPER SELL ([\d,]+)\s+(\S+)(?:\s+net=([+-]?[\d.]+)%)?/.exec(head);
    if (m) return { kind: 'paper_sell', text, price: price(m[1]), reason: m[2], pnlPct: pct(m[3]) };
    return { kind: 'note', text };
  }
  if (head === '매수 신호') return { kind: 'signal', text, reason: seg(text, 1) || undefined, price: price(/live=([\d,]+)/.exec(text)?.[1]), qty: price(/qty<=(\d+)/.exec(text)?.[1]) };
  if (head === '매수 주문 접수') return { kind: 'order', text };
  if (head === '매수 안 함') return { kind: 'nobuy', text, reason: [seg(text, 1), seg(text, 2)].filter(Boolean).join(' | ') || undefined, price: price(/price=([\d,]+)/.exec(text)?.[1]) };
  let m = /^qty=(\d+)\s+원가=([\d,]+)\s+현재가=([\d,]+)\S*\s+손익=([+-]?[\d.]+)%/.exec(head);
  if (m) {
    const ex = /tp=([+-]?[\d.]+)%\s+sl=([+-]?[\d.]+)%/.exec(text);
    return { kind: 'hold', text, qty: +m[1], avg: price(m[2]), price: price(m[3]), pnlPct: pct(m[4]), tpPct: pct(ex?.[1]), slPct: pct(ex?.[2]) };
  }
  m = /^청산\s+(\S+)$/.exec(head);
  if (m) {
    const d = /qty=(\d+)\s+원가=([\d,]+)\s+현재가=([\d,]+)\S*\s+손익=([+-]?[\d.]+)%/.exec(text);
    return { kind: 'exit', text, reason: m[1], qty: d ? +d[1] : undefined, avg: price(d?.[2]), price: price(d?.[3]), pnlPct: pct(d?.[4]) };
  }
  const hs = parseHandSale(text);
  if (hs) return { kind: 'hand_sell', text, qty: hs.qty, price: hs.price, avg: hs.cost, pnlKrw: hs.pnlKrw, pnlPct: hs.pnlPct };
  m = /^(buy|sell) 체결 (\d+)주 반영/.exec(head);
  if (m) return { kind: 'fill', text, reason: m[1], qty: +m[2], avg: price(/원가 ([\d,]+)/.exec(text)?.[1]) };
  return { kind: 'note', text };
}

/** [2026-11-27] "직접 매도 5주 기록 | 매도가=87,300 원가=87,900 손익=-4,012원(-0.91%) | strategy=vwap order=... | ..." (r003
 *  _log_side_hand_sale) -> its values; N/A fields are undefined. null = not that line */
export function parseHandSale(text: string): { qty: number; price?: number; cost?: number; pnlKrw?: number; pnlPct?: number; strategy?: SideStrategy } | null {
  const m = /^직접 매도 (\d+)주 기록\s*\|/.exec(text.trim());
  if (!m) return null;
  const p = /손익=([+-][\d,]+)원\(([+-]?[\d.]+)%\)/.exec(text);
  const st = /\bstrategy=(edge|vwap)\b/.exec(text)?.[1] as SideStrategy | undefined;
  return { qty: +m[1], price: price(/매도가=([\d,]+)/.exec(text)?.[1]), cost: price(/원가=([\d,]+)/.exec(text)?.[1]), pnlKrw: p ? price(p[1].replace('+', '')) : undefined, pnlPct: pct(p?.[2]), strategy: st };
}

/** a strategy line without a symbol */
export function parseStrategyNote(body: string): Pick<StrategyNote, 'kind' | 'text'> {
  const text = body.trim();
  if (/^신규 매수 중단/.test(text)) return { kind: 'blocked', text: seg(text, 1) || text };
  if (/^신규 매수 켜짐/.test(text)) return { kind: 'on', text: seg(text, 1) };
  if (/^신규 매수 꺼짐/.test(text)) return { kind: 'off', text: (/\((.*)\)/.exec(text)?.[1] || '').trim() };
  return { kind: 'info', text };
}

/** why new buys of a real-order strategy are stopped, for people (server edge_live_block_reason codes; unknown = as written) */
export function blockReasonKo(why: string): string {
  const w = why.trim();
  const rules: [RegExp, (m: RegExpExecArray) => string][] = [
    [/^MIXED_HOLDING_(\d{6})/, (m) => `${m[1]} 종목에 직접 산 주식이 섞여 있습니다 (이 전략 보유분 + 직접 매수분). 봇은 직접 산 수량을 팔지 않으며, 섞여 있는 동안 이 전략은 새로 사지 않습니다`],
    [/^DAILY_LOSS_/, () => '하루 손실 한도에 도달했습니다'],
    [/^MAX_ENTRIES_PER_DAY_/, () => '하루 매수 횟수 한도에 도달했습니다'],
    [/^MAX_POSITIONS_/, () => '동시 보유 한도에 도달했습니다'],
    [/^STOPPED_OUT_TODAY/, () => '오늘 손절한 종목입니다 (다시 사지 않음)'],
    [/^ENTRY_WINDOW_CLOSED_(\d{2}:\d{2})/, (m) => `신규 매수 마감 시각(${m[1]})이 지났습니다 - 보유 종목의 청산만 합니다`],
    [/^STOPPING/, () => '봇이 종료 중입니다'],
    [/^UNRESOLVED_ORDER_HOLD/, () => '확인되지 않은 이전 주문이 있어 새로 사지 않습니다 (브로커 확인 대기)'],
    [/^LEDGER_BASELINE_PENDING/, () => '체결 장부를 계좌와 맞추는 중입니다 (끝나면 다시 삽니다)'],
    [/^REALIZED_PNL_UNKNOWN/, () => '오늘 매도 가격을 확인하지 못한 거래가 있어 손실 한도를 계산할 수 없습니다'],
  ];
  for (const [re, f] of rules) {
    const m = re.exec(w);
    if (m) return `${f(m)} (${w})`;
  }
  return w;
}

// ---- per-bar evaluation trace (server strategy_eval_<strategy>_<date>.jsonl) ------------------------------------------------
export interface StrategyEvalRow {
  ts: number;
  code: string;
  mode: StrategyRunMode;
  /** stage reached / reason the bar stopped at (vwap_signal stage, paper runner stage, cond_<no> for a shared condition) */
  stage: string;
  /** number of the buy condition that refused (real orders) */
  no?: number;
  detail?: string;
  price?: number;
}

export function parseStrategyEvalLines(text: string): StrategyEvalRow[] {
  const out: StrategyEvalRow[] = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      const o = JSON.parse(s) as Record<string, unknown>;
      const ts = Date.parse(String(o.t || ''));
      const code = String(o.code || '');
      if (!Number.isFinite(ts) || !/^[0-9A-Z]{6}$/.test(code) || typeof o.stage !== 'string') continue;
      out.push({
        ts,
        code,
        mode: o.mode === 'paper' ? 'paper' : 'live',
        stage: o.stage,
        no: typeof o.no === 'number' ? o.no : undefined,
        detail: typeof o.detail === 'string' ? o.detail : undefined,
        price: typeof o.price === 'number' ? o.price : undefined,
      });
    } catch {
      /* a half-written last line */
    }
  }
  return out;
}

/** rows of the stage chart, top = furthest. `stages` = the stage codes drawn on that row */
export interface StageRow {
  key: string;
  label: string;
  stages: string[];
}
const COMMON_TOP: StageRow[] = [
  { key: 'order', label: '주문', stages: ['ordered', 'bought'] },
  { key: 'after', label: '수량 · 가격 · 호가 확인', stages: ['order_refused', 'cond_after'] },
  { key: 'signal', label: '매수 신호', stages: ['signal', 'candidate'] },
];
const COMMON_BOTTOM: StageRow[] = [
  { key: 'before', label: '공통 안전 조건', stages: ['cond_before', 'limit', 'outside_window', 'fund', 'not_target', 'sl_today', 'blocked'] },
  { key: 'data', label: '데이터 준비', stages: ['wait_bars', 'no_history', 'stale', 'no_model', 'regular_session_only', '1min_frame_unavailable', 'insufficient_bars', 'no_day_vwap', 'error'] },
  { key: 'held', label: '보유 중', stages: ['held', 'sold'] },
];
export const STAGE_ROWS: Record<SideStrategy, StageRow[]> = {
  vwap: [
    ...COMMON_TOP,
    { key: 'chase', label: '추격 · 급락 제한', stages: ['chase', 'price_drop'] },
    { key: 'tick', label: '60틱 골든크로스', stages: ['no_tick_cross'] },
    { key: 'trend', label: '1분 추세 (MA5 · BB 중심선)', stages: ['no_trend'] },
    { key: 'vwap', label: 'VWAP 위', stages: ['below_vwap'] },
    ...COMMON_BOTTOM,
  ],
  edge: [
    ...COMMON_TOP,
    { key: 'prob', label: 'Edge AI 확률', stages: ['below_threshold'] },
    { key: 'trigger', label: '1분봉 트리거', stages: ['no_trigger'] },
    { key: 'mid', label: '1분 BB 중심선 위', stages: ['below_mid'] },
    ...COMMON_BOTTOM,
  ],
};
/** the row a trace row is drawn on (unknown stage -> the data row, shown with its own text) */
export function stageRowOf(strategy: SideStrategy, r: Pick<StrategyEvalRow, 'stage'>): string {
  const st = r.stage.toLowerCase();
  for (const row of STAGE_ROWS[strategy]) if (row.stages.includes(st)) return row.key;
  return 'data';
}
/** a bar that reached the signal or further counts as passed, 'held' rows are neutral, everything else stopped there */
export const stageResult = (rowKey: string): 'pass' | 'fail' | 'info' => (rowKey === 'signal' || rowKey === 'order' ? 'pass' : rowKey === 'held' ? 'info' : 'fail');
