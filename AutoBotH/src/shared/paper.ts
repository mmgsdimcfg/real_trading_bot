/*
 * Update log
 * - [2026-11-12] type=feat - Edge-AI-only paper trading (server r022, r001 EDGE_ONLY_MODE): the day's virtual trades from
 *   runtime/data/live_runtime/edge_only_YYYYMMDD.jsonl (ENTRY / EXIT lines -> one PaperTrade per entry, open while it has no
 *   EXIT), the day summary, and the mode as r016 `get EDGE_ONLY_MODE` reports it (stored / running / restart pending)
 * - [2026-11-13] type=feat - AUTO_SELL_SCOPE (r001): the automatic sells sell only the bot's own quantity (bot_only, default) or the
 *   whole holding (all) - key, labels, parse helpers for the control tab
 * - [2026-11-12] type=feat - Trade mode choice of the control tab: TRADING_STRATEGY_MODE (normal / edge = Edge-AI-only with real
 *   orders) + EDGE_ONLY_MODE (paper) read together (parseModes) and reduced to one of three choices (tradeModeOf);
 *   tradeModeChanges() is the `set` a choice sends
 */

export interface PaperTrade {
  code: string;
  name: string;
  /** epoch ms of the virtual buy */
  entryTs: number;
  /** virtual fill (live price + 1 tick) */
  entry: number;
  prob?: number;
  threshold?: number;
  trigger?: string;
  tpPx?: number;
  slPx?: number;
  /** minutes until the time exit */
  horizon?: number;
  exitTs?: number;
  exit?: number;
  /** tp | sl | time | eod */
  reason?: string;
  /** net % after commission and sell tax */
  netPct?: number;
  heldMin?: number;
}

export const PAPER_EXIT_KO: Record<string, string> = { tp: '익절 (목표가)', sl: '손절', time: '시간 경과', eod: '장 마감' };

const tsToEpoch = (s: unknown): number => {
  // "2026-11-12T09:30:05" is KST wall time
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(String(s || ''));
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4] - 9, +m[5], +m[6]) : NaN;
};
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

/** jsonl text -> trades in entry order (a broken line is skipped; an EXIT closes the open trade of its code) */
export function parsePaperLines(text: string): PaperTrade[] {
  const out: PaperTrade[] = [];
  const open = new Map<string, PaperTrade>();
  for (const line of String(text || '').split(/\r?\n/)) {
    if (!line.trim()) continue;
    let r: Record<string, unknown>;
    try {
      r = JSON.parse(line);
    } catch {
      continue;
    }
    const code = String(r.code || '');
    const ts = tsToEpoch(r.ts);
    if (!code || !Number.isFinite(ts)) continue;
    if (r.event === 'ENTRY') {
      const entry = num(r.entry);
      if (!entry) continue;
      const t: PaperTrade = {
        code,
        name: String(r.name || ''),
        entryTs: ts,
        entry,
        prob: num(r.prob),
        threshold: num(r.threshold),
        trigger: typeof r.trigger === 'string' ? r.trigger : undefined,
        tpPx: num(r.tp_px),
        slPx: num(r.sl_px),
        horizon: num(r.horizon),
      };
      out.push(t);
      open.set(code, t);
    } else if (r.event === 'EXIT') {
      const t = open.get(code);
      if (!t) continue;
      t.exitTs = ts;
      t.exit = num(r.exit);
      t.reason = typeof r.reason === 'string' ? r.reason : undefined;
      t.netPct = num(r.net_pct);
      t.heldMin = num(r.held_min);
      open.delete(code);
    }
  }
  return out;
}

export interface PaperSummary {
  closed: number;
  open: number;
  wins: number;
  /** sum of the closed trades' net % (each trade = one equal slot) */
  sumPct: number;
  meanPct: number;
  winRate: number;
  byReason: Record<string, number>;
}

export function paperSummary(trades: PaperTrade[]): PaperSummary {
  const closed = trades.filter((t) => t.exitTs !== undefined);
  const nets = closed.map((t) => t.netPct ?? 0);
  const sumPct = nets.reduce((a, b) => a + b, 0);
  const wins = nets.filter((x) => x > 0).length;
  const byReason: Record<string, number> = {};
  for (const t of closed) byReason[t.reason || '?'] = (byReason[t.reason || '?'] || 0) + 1;
  return {
    closed: closed.length,
    open: trades.length - closed.length,
    wins,
    sumPct,
    meanPct: closed.length ? sumPct / closed.length : 0,
    winRate: closed.length ? wins / closed.length : 0,
    byReason,
  };
}

/** unrealized % of an open trade at `price` (no costs - the trade is not closed) */
export const paperOpenPct = (t: PaperTrade, price: number | undefined): number | undefined =>
  price && price > 0 && t.entry > 0 ? (price / t.entry - 1) * 100 : undefined;

// ---- mode (r016 get EDGE_ONLY_MODE) ----

export const PAPER_KEY = 'EDGE_ONLY_MODE';

export interface PaperModeState {
  /** value r003 uses at its next start: 'off' | 'paper' */
  effective: string;
  /** value the running r003 uses (null = not running / unknown) */
  running: string | null;
  /** effective differs from running -> r003 restart needed */
  pending: boolean;
}

export const STRATEGY_KEY = 'TRADING_STRATEGY_MODE';

function modeOf(data: unknown, key: string): PaperModeState | null {
  const v = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values?.[key];
  if (!v || typeof v.effective !== 'string') return null;
  return { effective: v.effective.toLowerCase(), running: typeof v.running === 'string' ? v.running.toLowerCase() : null, pending: !!v.pending };
}

/** `get EDGE_ONLY_MODE` data -> state; null when the server does not know the key (older server code) */
export const parsePaperMode = (data: unknown): PaperModeState | null => modeOf(data, PAPER_KEY);

/** `get TRADING_STRATEGY_MODE EDGE_ONLY_MODE` data -> both states (each null when the server does not report it) */
export function parseModes(data: unknown): { strategy: PaperModeState | null; paper: PaperModeState | null } {
  return { strategy: modeOf(data, STRATEGY_KEY), paper: modeOf(data, PAPER_KEY) };
}

/**
 * The control tab's choice:
 *   rules        current rule-based trading only
 *   rules+paper  current rule-based trading, Edge AI as paper trading beside it (no orders)
 *   edge         Edge AI only, real orders (no paper trading - it would be the same policy twice)
 *   other        a strategy mode this choice does not cover (e.g. basic) - shown as it is, not changed by accident
 */
export type TradeMode = 'rules' | 'rules+paper' | 'edge' | 'other';

export function tradeModeOf(strategy: string | null | undefined, paper: string | null | undefined): TradeMode {
  const s = String(strategy || '').toLowerCase();
  if (s === 'edge') return 'edge';
  if (s !== 'normal') return 'other';
  return String(paper || '').toLowerCase() === 'paper' ? 'rules+paper' : 'rules';
}

/** r016 `set` values of a choice (null for 'other': nothing to send) */
export function tradeModeChanges(mode: TradeMode): Record<string, string> | null {
  if (mode === 'rules') return { [STRATEGY_KEY]: 'normal', [PAPER_KEY]: 'off' };
  if (mode === 'rules+paper') return { [STRATEGY_KEY]: 'normal', [PAPER_KEY]: 'paper' };
  if (mode === 'edge') return { [STRATEGY_KEY]: 'edge', [PAPER_KEY]: 'off' };
  return null;
}

/** [2026-11-12] r001 AUTO_SELL_MODE: which sells the bot makes by itself */
export const AUTO_SELL_KEY = 'AUTO_SELL_MODE';
export type AutoSellMode = 'all' | 'stop_only' | 'off';
export const AUTO_SELL_MODES: AutoSellMode[] = ['all', 'stop_only', 'off'];
export const AUTO_SELL_KO: Record<AutoSellMode, string> = {
  all: '전체 자동 (익절·트레일링·당일 청산 포함)',
  stop_only: '손절만 자동 (비용 포함 -2%)',
  off: '끔 (모든 매도 수동)',
};
/** `get AUTO_SELL_MODE` data -> state; null when the server does not know the key (older server code) */
export const parseAutoSell = (data: unknown): PaperModeState | null => modeOf(data, AUTO_SELL_KEY);
/** a server value as one of the three choices (anything else = the bot's fallback 'all') */
export const autoSellOf = (v: string | null | undefined): AutoSellMode => (v === 'stop_only' || v === 'off' ? v : 'all');

/** [2026-11-13] r001 AUTO_SELL_SCOPE: which quantity the automatic sells may sell (the bot's own buys only / the whole holding) */
export const AUTO_SELL_SCOPE_KEY = 'AUTO_SELL_SCOPE';
export type AutoSellScope = 'bot_only' | 'all';
export const AUTO_SELL_SCOPES: AutoSellScope[] = ['bot_only', 'all'];
export const AUTO_SELL_SCOPE_KO: Record<AutoSellScope, string> = {
  bot_only: '봇 매수분만',
  all: '직접 매수분 포함 (전량)',
};
/** `get AUTO_SELL_SCOPE` data -> state; null when the server does not know the key (older server code) */
export const parseAutoSellScope = (data: unknown): PaperModeState | null => modeOf(data, AUTO_SELL_SCOPE_KEY);
/** a server value as one of the two choices (anything but 'all' = the bot's fallback 'bot_only') */
export const autoSellScopeOf = (v: string | null | undefined): AutoSellScope => (v === 'all' ? 'all' : 'bot_only');

export const TRADE_MODE_KO: Record<TradeMode, string> = {
  rules: '현재 방식 (규칙 기반)',
  'rules+paper': '현재 방식 + Edge AI 가상 매매',
  edge: 'Edge AI 방식 (실제 주문)',
  other: '기타',
};

/** [2026-11-12] the Edge AI model the bot uses now (server file runtime/edge_ai/current_meta.json, written by g014) */
export interface EdgeModelInfo {
  modelId: string;
  created: string;
  threshold: number | null;
  promoted: boolean;
  /** label column: 'y_tp' = +tp% reached before the stop, 'y' = net return > 0 (older models have no target = 'y') */
  target: string;
  tp: number | null;
  sl: number | null;
  horizon: number | null;
  dataset: string;
  nDays: number | null;
  trainRange: string;
  aucOos: number | null;
  /** out-of-sample entries the model kept / all entries: count and mean net return (%) */
  kept: { n: number; meanNet: number } | null;
  base: { n: number; meanNet: number } | null;
  /** share of kept / all entries that reached +2 / +3 / +5 % before the stop, and the stopped share (new label only) */
  runKept: { tp2: number; tp3: number; tp5: number; stopped: number } | null;
  runBase: { tp2: number; tp3: number; tp5: number; stopped: number } | null;
}

const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** current_meta.json text -> summary; null when the text is not a model meta */
export function parseEdgeMeta(text: string): EdgeModelInfo | null {
  let m: Record<string, any>;
  try {
    m = JSON.parse(text);
  } catch {
    return null;
  }
  if (!m || typeof m !== 'object' || typeof m.model_id !== 'string') return null;
  const label = (m.label && typeof m.label === 'object' ? m.label : {}) as Record<string, unknown>;
  const pm = (m.promotion && typeof m.promotion === 'object' ? m.promotion : {}) as Record<string, any>;
  const stat = (s: any) => (s && numOrNull(s.n) !== null && numOrNull(s.mean_net) !== null ? { n: s.n as number, meanNet: s.mean_net as number } : null);
  const run = (s: any) => (s && numOrNull(s.tp2) !== null ? { tp2: s.tp2 as number, tp3: Number(s.tp3) || 0, tp5: Number(s.tp5) || 0, stopped: Number(s.stopped) || 0 } : null);
  return {
    modelId: m.model_id,
    created: typeof m.created === 'string' ? m.created : '',
    threshold: numOrNull(m.threshold),
    promoted: m.promoted === true,
    target: typeof label.target === 'string' ? label.target : 'y',
    tp: numOrNull(label.tp),
    sl: numOrNull(label.sl),
    horizon: numOrNull(label.horizon),
    dataset: typeof m.dataset === 'string' ? m.dataset : '',
    nDays: numOrNull(m.n_days),
    trainRange: typeof m.train_range === 'string' ? m.train_range : '',
    aucOos: numOrNull(pm.auc_oos),
    kept: stat(pm.kept),
    base: stat(pm.base),
    runKept: run(pm.run_kept),
    runBase: run(pm.run_base),
  };
}

/** "손절(-1.77%) 전에 +2% 도달" / "+2% / -1.7% / 60분 순손익 > 0" */
export function edgeLabelKo(m: EdgeModelInfo): string {
  const tp = m.tp ?? '?';
  const sl = m.sl ?? '?';
  return m.target === 'y_tp' ? `손절(-${sl}%) 전에 +${tp}% 도달` : `+${tp}% / -${sl}% / ${m.horizon ?? '?'}분 순손익 > 0`;
}
