/*
 * Update log
 * - [2026-11-25] type=feat - trading kinds combined in the control tab (user 2026-11-25, Codex review): 규칙 매매 (server r001
 *   RULE_BUY_ENABLED) / Edge AI 가상 매매 (EDGE_ONLY_MODE) / 반등 알림 (AutoBotHtsH app setting): RULE_BUY_KEY, parseRuleBuy, TRADE_KIND_KO,
 *   comboLabel, comboChanges (never touches TRADING_STRATEGY_MODE)
 * - [2026-11-25] type=feat - session close reasons close_regular (15:20 auction) / close_nxt (NXT 19:59:50) and the estimated flag
 *   (server r022: paper positions follow the trading hours and the symbol's NXT eligibility)
 * - [2026-11-25] type=feat - paper exits set by the user (server r001 EDGE_ONLY_TP_PCT / EDGE_ONLY_SL_PCT / EDGE_ONLY_HORIZON_MIN,
 *   0 = the model's label): PAPER_EXIT_KEYS, parsePaperExit, paperExitInput, paperExitChanges; status model_label
 * - [2026-11-23] type=feat - PaperTrade.session / position.session ('regular' | 'nxt_pre' | 'nxt_after', server r022 nxt=True;
 *   a record without it is regular), PaperStatus.nxt + nxtSessions (entry windows / force exits of the NXT sessions)
 * - [2026-11-12] type=feat - Edge-AI-only paper trading (server r022, r001 EDGE_ONLY_MODE): the day's virtual trades from
 *   runtime/data/live_runtime/edge_only_YYYYMMDD.jsonl (ENTRY / EXIT lines -> one PaperTrade per entry, open while it has no
 *   EXIT), the day summary, and the mode as r016 `get EDGE_ONLY_MODE` reports it (stored / running / restart pending)
 * - [2026-11-13] type=feat - AUTO_SELL_SCOPE (r001): the automatic sells sell only the bot's own quantity (bot_only, default) or the
 *   whole holding (all) - key, labels, parse helpers for the control tab
 * - [2026-11-12] type=feat - Trade mode choice of the control tab: TRADING_STRATEGY_MODE (normal / edge = Edge-AI-only with real
 *   orders) + EDGE_ONLY_MODE (paper) read together (parseModes) and reduced to one of three choices (tradeModeOf);
 *   tradeModeChanges() is the `set` a choice sends
 * - [2026-11-19] type=feat - PaperTrade.group: 'watch' (the bot's watchlist) | 'collector' (tick collector symbols outside it,
 *   r001 EDGE_ONLY_COLLECTOR - separate limits on the server). A record without group is 'watch'
 * - [2026-11-19] type=feat - PaperStatus: the server's edge_only_status_<date>.json (r022 write_status, every 5 s while the bot
 *   runs in the regular session) - every symbol the paper runner looks at with its group, price, last bar, the outcome of its
 *   last evaluation (stage) and probability, plus the group limits, threshold and model
 * - [2026-11-20] type=feat - status log_only (server EDGE_ONLY_LOG_ONLY) + stage candidate
 * - [2026-11-20] type=feat - stages fund (ETF/ETN not paper-bought) and sl_today (stopped out today, no re-entry) - server r022
 */

export type PaperGroup = 'watch' | 'collector';
export const PAPER_GROUP_KO: Record<PaperGroup, string> = { watch: '감시 종목', collector: '수집기 종목' };
/** [2026-11-23] session a paper trade was opened in */
export type PaperSession = 'regular' | 'nxt_pre' | 'nxt_after';
export const PAPER_SESSION_KO: Record<PaperSession, string> = { regular: '정규장', nxt_pre: 'NXT 장전', nxt_after: 'NXT 장후' };
export const paperSession = (v: unknown): PaperSession => (v === 'nxt_pre' || v === 'nxt_after' ? v : 'regular');

export interface PaperTrade {
  code: string;
  /** [2026-11-19] which symbols the trade came from */
  group: PaperGroup;
  /** [2026-11-23] regular | nxt_pre | nxt_after */
  session: PaperSession;
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
  /** tp | sl | time | eod (before 2026-11-25) | close_regular (15:20 auction) | close_nxt (NXT 19:59:50) */
  reason?: string;
  /** [2026-11-25] session close priced with the last price seen (no auction print / no fresh NXT price) */
  estimated?: boolean;
  /** net % after commission and sell tax */
  netPct?: number;
  heldMin?: number;
}

export const PAPER_EXIT_KO: Record<string, string> = {
  tp: '익절 (목표가)',
  sl: '손절',
  time: '시간 경과',
  eod: '장 마감',
  // [2026-11-25] session closes (server r022): regular-only holdings into the 15:20 closing auction, NXT holdings at 19:59:50
  close_regular: '정규장 종가 청산',
  close_nxt: 'NXT 마감 청산',
  // [2026-11-25] r003 stopped before that day's close: closed at its next start, at that day's close (server r022 close_missed_days)
  close_missed: '미청산 정리',
};

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
        group: r.group === 'collector' ? 'collector' : 'watch',
        session: paperSession(r.session),
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
      if (r.estimated === true) t.estimated = true;
      t.netPct = num(r.net_pct);
      t.heldMin = num(r.held_min);
      open.delete(code);
    }
  }
  return out;
}

/** [2026-11-19] outcome of a symbol's last evaluation (server r022 STAGES) */
export const PAPER_STAGE_KO: Record<string, string> = {
  wait_bars: '봉 대기 (5개 미만)',
  outside_window: '매수 시간 밖 (09:05~14:50)',
  limit: '상한 도달',
  held: '가상 보유 중',
  below_mid: '1분 BB 중심선 아래',
  no_trigger: '트리거 없음',
  no_history: '이력 부족 (늦게 시작)',
  no_model: '모델 없음',
  below_threshold: '확률 미달',
  bought: '가상 매수',
  sold: '가상 매도',
  fund: 'ETF·ETN 제외',
  sl_today: '오늘 손절한 종목 (재매수 안 함)',
  candidate: '후보 기록 (신규 매수 꺼짐)',
};

export interface PaperStatusSymbol {
  code: string;
  name: string;
  group: PaperGroup;
  price?: number;
  /** "HH:MM" end label of the last completed 1-minute bar */
  bar?: string;
  bars: number;
  stage?: string;
  prob?: number;
  /** epoch ms the runner last looked at the symbol */
  at?: number;
  position?: { entry: number; tp: number; sl: number; ts: number; prob?: number; session?: PaperSession };
}

export interface PaperGroupStatus {
  open: number;
  maxPositions: number;
  entries: number;
  maxEntries: number;
  enabled: boolean;
}

export interface PaperStatus {
  /** epoch ms of the file (server time) */
  asof: number;
  date: string;
  modelId?: string;
  threshold?: number;
  /** the exits new paper buys use (since 2026-11-25 the r001 EDGE_ONLY_TP_PCT / SL_PCT / HORIZON_MIN where set, else the model's) */
  label?: { tp?: number; sl?: number; horizon?: number };
  /** [2026-11-25] the model's own label (absent on an older server = same as label) */
  modelLabel?: { tp?: number; sl?: number; horizon?: number };
  entryWindow?: [string, string];
  /** [2026-11-20] server r001 EDGE_ONLY_LOG_ONLY: no new paper buys, trigger bars are only recorded as candidates */
  logOnly?: boolean;
  /** [2026-11-23] the runner also paper-trades the NXT sessions (r001 EDGE_ONLY_NXT) */
  nxt?: boolean;
  nxtSessions?: Partial<Record<PaperSession, { entryWindow: [string, string]; forceExit: string }>>;
  groups: Record<PaperGroup, PaperGroupStatus>;
  symbols: PaperStatusSymbol[];
}

function nxtSessions(raw: unknown): PaperStatus['nxtSessions'] {
  const out: NonNullable<PaperStatus['nxtSessions']> = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const key of ['nxt_pre', 'nxt_after'] as const) {
    const v = (raw as Record<string, { entry_window?: unknown; force_exit?: unknown }>)[key];
    if (v && Array.isArray(v.entry_window) && v.entry_window.length === 2) {
      out[key] = { entryWindow: [String(v.entry_window[0]), String(v.entry_window[1])], forceExit: String(v.force_exit || '') };
    }
  }
  return out;
}

/** edge_only_status_<date>.json text -> status; null when it is not a status file */
export function parsePaperStatus(text: string): PaperStatus | null {
  let j: Record<string, unknown>;
  try {
    j = JSON.parse(String(text || ''));
  } catch {
    return null;
  }
  if (!j || typeof j !== 'object' || !Array.isArray(j.symbols)) return null;
  const grp = (v: unknown): PaperGroupStatus => {
    const g = (v || {}) as Record<string, unknown>;
    return {
      open: num(g.open) ?? 0,
      maxPositions: num(g.max_positions) ?? 0,
      entries: num(g.entries) ?? 0,
      maxEntries: num(g.max_entries) ?? 0,
      enabled: g.enabled !== false,
    };
  };
  const groups = (j.groups || {}) as Record<string, unknown>;
  const symbols: PaperStatusSymbol[] = [];
  for (const raw of j.symbols as Record<string, unknown>[]) {
    const code = String(raw?.code || '');
    if (!/^[0-9A-Z]{6}$/.test(code)) continue;
    const p = raw.position as Record<string, unknown> | undefined;
    const entry = num(p?.entry);
    const at = tsToEpoch(raw.at);
    const pts = tsToEpoch(p?.ts);
    symbols.push({
      code,
      name: String(raw.name || ''),
      group: raw.group === 'collector' ? 'collector' : 'watch',
      price: num(raw.price),
      bar: typeof raw.bar === 'string' ? raw.bar : undefined,
      bars: num(raw.bars) ?? 0,
      stage: typeof raw.stage === 'string' ? raw.stage : undefined,
      prob: num(raw.prob),
      at: Number.isFinite(at) ? at : undefined,
      position: entry ? { entry, tp: num(p?.tp) ?? NaN, sl: num(p?.sl) ?? NaN, ts: Number.isFinite(pts) ? pts : NaN, prob: num(p?.prob), session: paperSession(p?.session) } : undefined,
    });
  }
  const win = Array.isArray(j.entry_window) && j.entry_window.length === 2 ? ([String(j.entry_window[0]), String(j.entry_window[1])] as [string, string]) : undefined;
  return {
    asof: tsToEpoch(j.asof),
    date: String(j.date || ''),
    modelId: typeof j.model_id === 'string' ? j.model_id : undefined,
    threshold: num(j.threshold),
    label: (j.label as PaperStatus['label']) || undefined,
    modelLabel: (j.model_label as PaperStatus['modelLabel']) || undefined,
    entryWindow: win,
    logOnly: j.log_only === true,
    nxt: j.nxt === true,
    nxtSessions: nxtSessions(j.nxt_sessions),
    groups: { watch: grp(groups.watch), collector: grp(groups.collector) },
    symbols,
  };
}

/** status rows in display order: held first, then the highest probability, then by code */
export function sortStatusRows(rows: PaperStatusSymbol[]): PaperStatusSymbol[] {
  return [...rows].sort(
    (a, b) => Number(!!b.position) - Number(!!a.position) || (b.prob ?? -1) - (a.prob ?? -1) || a.code.localeCompare(b.code),
  );
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

/** [2026-11-23] r001 EDGE_ONLY_LOG_ONLY: true = paper trading records candidates only, no new paper buys */
export const PAPER_LOG_ONLY_KEY = 'EDGE_ONLY_LOG_ONLY';
/** `get EDGE_ONLY_LOG_ONLY` data -> 'true' / 'false' state (the value is a boolean on the server); null on a server without the key */
export function parsePaperLogOnly(data: unknown): PaperModeState | null {
  const v = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values?.[PAPER_LOG_ONLY_KEY];
  if (!v || typeof v.effective !== 'boolean') return null;
  return { effective: String(v.effective), running: typeof v.running === 'boolean' ? String(v.running) : null, pending: !!v.pending };
}

/** [2026-11-25] r001 EDGE_ONLY_TP_PCT / EDGE_ONLY_SL_PCT (%) / EDGE_ONLY_HORIZON_MIN (minutes): the exits of new paper buys,
 *  0 = the model's label (user 2026-11-25: set them in the app) */
export const PAPER_EXIT_KEYS = { tp: 'EDGE_ONLY_TP_PCT', sl: 'EDGE_ONLY_SL_PCT', horizon: 'EDGE_ONLY_HORIZON_MIN' } as const;
export type PaperExitKind = keyof typeof PAPER_EXIT_KEYS;
export const PAPER_EXIT_KINDS: PaperExitKind[] = ['tp', 'sl', 'horizon'];
/** allowed range of a set value (0 = the model's label) */
export const PAPER_EXIT_MAX: Record<PaperExitKind, number> = { tp: 30, sl: 30, horizon: 720 };
export interface PaperExitValue {
  /** value r003 uses at its next start (0 = the model's label) */
  effective: number;
  /** value the running r003 uses (null = not running / unknown) */
  running: number | null;
  pending: boolean;
}
export type PaperExitState = Record<PaperExitKind, PaperExitValue>;

/** `get EDGE_ONLY_TP_PCT EDGE_ONLY_SL_PCT EDGE_ONLY_HORIZON_MIN` data -> state; null on a server without the keys */
export function parsePaperExit(data: unknown): PaperExitState | null {
  return parseExitKeys(data, PAPER_EXIT_KEYS);
}

/** the three exit values under the given server keys (all three or null) */
function parseExitKeys(data: unknown, keys: Record<PaperExitKind, string>): PaperExitState | null {
  const vals = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values;
  const out: Partial<PaperExitState> = {};
  for (const k of PAPER_EXIT_KINDS) {
    const v = vals?.[keys[k]];
    if (!v || typeof v.effective !== 'number' || !Number.isFinite(v.effective)) return null;
    out[k] = { effective: v.effective, running: typeof v.running === 'number' && Number.isFinite(v.running) ? v.running : null, pending: !!v.pending };
  }
  return out as PaperExitState;
}

/** an input box's text -> the value to store: '' = 0 (the model's label); null = not a number / out of range.
 *  Percentages keep two decimals, the minutes are whole */
export function paperExitInput(kind: PaperExitKind, text: string): number | null {
  const t = text.trim();
  if (t === '') return 0;
  if (!/^\d+(\.\d+)?$/.test(t)) return null;
  const n = Number(t);
  if (!Number.isFinite(n) || n < 0 || n > PAPER_EXIT_MAX[kind]) return null;
  if (kind === 'horizon') return Number.isInteger(n) ? n : null;
  return Math.round(n * 100) / 100;
}

/** the r016 set changes: percentages as floats, minutes as an integer (r001 kinds: float / float / int) */
export function paperExitChanges(v: Record<PaperExitKind, number>): Record<string, number> {
  return { [PAPER_EXIT_KEYS.tp]: v.tp, [PAPER_EXIT_KEYS.sl]: v.sl, [PAPER_EXIT_KEYS.horizon]: Math.round(v.horizon) };
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
  all: '전체 자동 (익절·트레일링·손절·당일 청산 포함)',
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

/** [2026-11-25] r001 RULE_BUY_ENABLED: 규칙 매매 (rule-based new buys) on / off - a server without it: null */
export const RULE_BUY_KEY = 'RULE_BUY_ENABLED';
export function parseRuleBuy(data: unknown): PaperModeState | null {
  const v = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values?.[RULE_BUY_KEY];
  if (!v || typeof v.effective !== 'boolean') return null;
  return { effective: String(v.effective), running: typeof v.running === 'boolean' ? String(v.running) : null, pending: !!v.pending };
}

/** [2026-11-25] r001 EDGE_LIVE_ORDERS: Edge AI live orders beside the rule strategy (the control tab's "가상 매수" unchecked).
 *  A server without the key: null (the checkbox stays locked) */
export const EDGE_LIVE_KEY = 'EDGE_LIVE_ORDERS';
export function parseEdgeLive(data: unknown): PaperModeState | null {
  const v = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values?.[EDGE_LIVE_KEY];
  if (!v || typeof v.effective !== 'boolean') return null;
  return { effective: String(v.effective), running: typeof v.running === 'boolean' ? String(v.running) : null, pending: !!v.pending };
}

/** [2026-11-25] the warning every editor shows before EDGE_LIVE_ORDERS is switched on (the control tab adds the limits) */
export const EDGE_LIVE_ON_WARNING =
  'Edge AI 실제 매매를 켭니다 - 실제 계좌로 Edge AI 매수 주문이 나갑니다 (규칙 매매와 병행).\n\n' +
  '주의: 이 방식의 과거 검증은 모두 손실이었습니다 (58일 재생 거래당 -0.44 ~ -0.62%, 가상 매매도 손절이 대부분).\n' +
  '수익 기대가 아니라 실제 체결·슬리피지를 작은 금액으로 확인하는 단계입니다.\n\n' +
  '조건: Edge AI 매매가 켜져 있고 자동 매도가 전체일 때만 새로 삽니다. r003 재시작 후 적용.\n\n켤까요?';

/** the server keys that together decide whether new Edge AI live buys can start (r003 _edge_live_config_ok) */
export const EDGE_LIVE_GATE_KEYS: string[] = [EDGE_LIVE_KEY, 'EDGE_ONLY_MODE', 'AUTO_SELL_MODE'];
/** what is true for shares the edge strategy bought, whatever the sell settings say */
export const EDGE_LIVE_EXIT_NOTE =
  'Edge AI 실제 매매로 산 수량은 자동 매도 설정과 무관하게 Edge 청산 기준(손절 / 익절 / 보유 시간 / 15:20)으로 자동 매도됩니다.';

/** [2026-11-25] the rule strategy's own exits (server r001 RULE_EXIT_TP_PCT / RULE_EXIT_SL_PCT (%) / RULE_EXIT_HORIZON_MIN,
 *  r006 _c03_common_exit_bounds) - apart from the Edge AI ones (user: two input rows). Same kinds, ranges and value types */
export const RULE_EXIT_KEYS: Record<PaperExitKind, string> = { tp: 'RULE_EXIT_TP_PCT', sl: 'RULE_EXIT_SL_PCT', horizon: 'RULE_EXIT_HORIZON_MIN' };
/** `get RULE_EXIT_TP_PCT ...` data -> state; null on a server without the keys */
export const parseRuleExit = (data: unknown): PaperExitState | null => parseExitKeys(data, RULE_EXIT_KEYS);
export function ruleExitChanges(v: Record<PaperExitKind, number>): Record<string, number> {
  return { [RULE_EXIT_KEYS.tp]: v.tp, [RULE_EXIT_KEYS.sl]: v.sl, [RULE_EXIT_KEYS.horizon]: Math.round(v.horizon) };
}
/** how the rule strategy's exits act on its positions */
export const COMMON_EXIT_NOTE =
  '값을 넣은 항목만 규칙 매매 보유분에 적용됩니다 (빈칸은 미적용). 기존 규칙 청산(분할 익절·트레일링·적응형 엔진·15:20)은 그대로 두고, ' +
  '익절 % 도달 / 손절 % 도달 / 보유 시간 경과 중 먼저 닿으면 잔량 전부를 매도합니다. 정규장에서만, 자동 매도가 전체일 때만 판정합니다. r003 재시작 후 적용.';

/** [2026-11-25] the limits of the Edge AI live orders (r001 EDGE_LIVE_*; the ranges are r016's EDGE_LIVE_RANGES) */
export const EDGE_LIVE_LIMIT_KEYS = {
  order: 'EDGE_LIVE_ORDER_KRW',
  positions: 'EDGE_LIVE_MAX_POSITIONS',
  entries: 'EDGE_LIVE_MAX_ENTRIES_PER_DAY',
  loss: 'EDGE_LIVE_DAILY_LOSS_KRW',
} as const;
export type EdgeLiveLimitKind = keyof typeof EDGE_LIVE_LIMIT_KEYS;
export const EDGE_LIVE_LIMIT_KINDS: EdgeLiveLimitKind[] = ['order', 'positions', 'entries', 'loss'];
export const EDGE_LIVE_LIMIT_RANGE: Record<EdgeLiveLimitKind, [number, number]> = {
  order: [10_000, 1_000_000],
  positions: [1, 5],
  entries: [1, 20],
  loss: [1_000, 1_000_000],
};
export const EDGE_LIVE_LIMIT_KO: Record<EdgeLiveLimitKind, string> = { order: '1회 금액', positions: '동시 보유', entries: '하루 횟수', loss: '하루 손실 한도' };
export const EDGE_LIVE_LIMIT_UNIT: Record<EdgeLiveLimitKind, string> = { order: '원', positions: '종목', entries: '회', loss: '원' };
export type EdgeLiveLimits = Record<EdgeLiveLimitKind, PaperExitValue>;

/** `get EDGE_LIVE_ORDER_KRW ...` data -> state; null on a server without the keys */
export function parseEdgeLiveLimits(data: unknown): EdgeLiveLimits | null {
  const vals = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values;
  const out: Partial<EdgeLiveLimits> = {};
  for (const k of EDGE_LIVE_LIMIT_KINDS) {
    const v = vals?.[EDGE_LIVE_LIMIT_KEYS[k]];
    if (!v || typeof v.effective !== 'number' || !Number.isFinite(v.effective)) return null;
    out[k] = { effective: v.effective, running: typeof v.running === 'number' && Number.isFinite(v.running) ? v.running : null, pending: !!v.pending };
  }
  return out as EdgeLiveLimits;
}

/** an input box's text -> the whole number to store ("100,000" is read too); null = not a number / out of range */
export function edgeLiveLimitInput(kind: EdgeLiveLimitKind, text: string): number | null {
  const t = text.trim().replace(/,/g, '');
  if (!/^\d+$/.test(t)) return null;
  const n = Number(t);
  const [lo, hi] = EDGE_LIVE_LIMIT_RANGE[kind];
  return Number.isSafeInteger(n) && n >= lo && n <= hi ? n : null;
}

/** the r016 set changes of the limits (all four are integers on the server) */
export function edgeLiveLimitChanges(v: Record<EdgeLiveLimitKind, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of EDGE_LIVE_LIMIT_KINDS) out[EDGE_LIVE_LIMIT_KEYS[k]] = Math.round(v[k]);
  return out;
}

/** [2026-11-25] what the edge strategy holds and did today (r016 status `edge_live`, from r003's fill ledger) */
export interface EdgeLiveSummary {
  held: { code: string; qty: number; avgPrice: number }[];
  openBuys: string[];
  entries: number;
  realized: number;
  slCodes: string[];
  /** sells whose price is not known: the day's result is not certain (the server stops new edge buys) */
  unpricedSells: number;
}

/** r016 status -> the summary; null on a server without it / before the fill ledger exists */
export function parseEdgeLiveSummary(status: unknown): EdgeLiveSummary | null {
  const e = (status as { edge_live?: unknown } | null)?.edge_live as
    | { held?: unknown; open_buys?: unknown; today?: Record<string, unknown> }
    | null
    | undefined;
  if (!e || typeof e !== 'object' || !Array.isArray(e.held)) return null;
  const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
  const t = e.today || {};
  return {
    held: (e.held as Record<string, unknown>[])
      .filter((h) => h && typeof h.code === 'string')
      .map((h) => ({ code: String(h.code), qty: num(h.qty), avgPrice: num(h.avg_price) })),
    openBuys: Array.isArray(e.open_buys) ? (e.open_buys as unknown[]).map(String) : [],
    entries: num(t.entries),
    realized: num(t.realized),
    slCodes: Array.isArray(t.sl_codes) ? (t.sl_codes as unknown[]).map(String) : [],
    unpricedSells: num(t.unpriced_sells),
  };
}

/** "오늘 매수 1/1회 · 실현 -1,200원 · 보유 005930 9주" */
export function edgeLiveSummaryText(s: EdgeLiveSummary, maxEntries: number | null): string {
  const parts = [`오늘 매수 ${s.entries}${maxEntries ? `/${maxEntries}` : ''}회`, `실현 ${s.realized >= 0 ? '+' : ''}${Math.round(s.realized).toLocaleString('ko-KR')}원`];
  if (s.held.length) parts.push(`보유 ${s.held.map((h) => `${h.code} ${h.qty}주`).join(', ')}`);
  if (s.openBuys.length) parts.push(`매수 주문 중 ${s.openBuys.join(', ')}`);
  if (s.slCodes.length) parts.push(`오늘 손절 ${s.slCodes.join(', ')}`);
  if (s.unpricedSells > 0) parts.push('손익 미상 (신규 매수 중단)');
  return parts.join(' · ');
}

/** [2026-11-25] the three trading kinds the control tab combines (user 2026-11-25): any of them on or off */
export type TradeKind = 'rules' | 'paper' | 'rebound';
export const TRADE_KINDS: TradeKind[] = ['rules', 'paper', 'rebound'];
export const TRADE_KIND_KO: Record<TradeKind, string> = { rules: '규칙 매매', paper: 'Edge AI 가상 매매', rebound: '반등 알림' };

/** "규칙 매매 + 반등 알림" for the kinds that are on (only the kinds this app has); "모두 꺼짐" when none */
export function comboLabel(on: Partial<Record<TradeKind, boolean>>, kinds: TradeKind[] = TRADE_KINDS): string {
  const names = kinds.filter((k) => on[k]).map((k) => TRADE_KIND_KO[k]);
  return names.length ? names.join(' + ') : '모두 꺼짐';
}

/** server keys for a change of the rule / paper checkboxes (TRADING_STRATEGY_MODE is never touched); null = nothing to change
 *  or the server lacks RULE_BUY_ENABLED for a rules change */
export function comboChanges(
  next: { rules: boolean; paper: boolean },
  cur: { rules: boolean; paper: boolean },
  ruleKeyKnown: boolean,
): Record<string, string | boolean> | null {
  const out: Record<string, string | boolean> = {};
  if (next.rules !== cur.rules) {
    if (!ruleKeyKnown) return null;
    out[RULE_BUY_KEY] = next.rules;
  }
  if (next.paper !== cur.paper) out[PAPER_KEY] = next.paper ? 'paper' : 'off';
  return Object.keys(out).length ? out : null;
}

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
  /** [2026-11-27] g014 found no cut with a positive expectancy: the Edge AI trades buy nothing (threshold is then 1) */
  noTrade: boolean;
  /** [2026-11-27] 'trade' = the Edge AI trade model (edge_ai/trade); 'gate' = the rule-trade filter model (also every older meta) */
  role: 'gate' | 'trade';
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

/** [2026-11-27] the Edge AI buy cut as text: '없음 (매수 안 함)' when the model has no cut with a positive expectancy */
export function edgeCutText(threshold: number | null | undefined, noTrade = false): string {
  if (threshold === null || threshold === undefined) return '-';
  return noTrade || threshold >= 1 ? '없음 (매수 안 함)' : threshold.toFixed(3);
}

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
    noTrade: m.no_trade === true,
    role: m.role === 'trade' ? 'trade' : 'gate',
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
