/*
 * Update log
 * - [2026-11-04] type=feat - Log -> condition mapping (guide 3.1 buy path, 3.2 sell path). Conditions are keyed by number ("_014"),
 *   which is matched to Condition.no of the server JSON, so a renamed function does not break the mapping. Reason prefixes are
 *   matched longest-first; anything that matches no rule is reported as unknown with the original line kept
 * - [2026-11-05] type=fix - New r003 info tags were listed as unknown: [BAR_SKIP] (intrabar 3min RSI/MFI out of range ->
 *   "using confirmed 3min bar instead", a data-source note, not a reject) and [WARMUP] (prior-day bars loaded for indicator warmup)
 *   added to IGNORED_TAGS
 * - [2026-11-05] type=feat - [CHECK] and [BAR_SKIP] are no longer ignored: they become status events (condKey 'PREP', result 'info',
 *   sub STATUS_SUB) so the timeline shows the bot is still checking a symbol while no condition is evaluated. The engine thins them
 *   (see DayEngine.push). [WARMUP] stays ignored
 * - [2026-11-05] type=feat - [HOLD SKIP] (logged as "[🔴HOLD SKIP ]": a held position that was not bought today, e.g.
 *   NOT_TODAY_BUY_POSITION, so the sell conditions are not evaluated for it) becomes a sell-side status event on the 'PREP' row
 * - [2026-11-07] type=fix - Unknown-list leftovers of 2026-11-04 mapped: [SELL EVAL] OK <reason> (r006 AUX / BB down-cross sell, no
 *   [SELL TRIGGER] line) -> sell condition by reason, pass; [TRADE] BUY REPRICE RESUBMIT and [BUY CXL REJ] -> buy ORDER info;
 *   [BUY SKIP] CIRCUIT_BREAKER_ACTIVE (breaker removed from r003 the same day) -> PREP status info (a global pause, not a block)
 * - [2026-11-08] type=refactor - Buy condition numbers follow the evaluation order (server r005 renumbered _001.._025): rule
 *   keys moved with it (_025->_007, _026->_008, _008->_009, _029->_010, _009->_011, _014->_012, _015->_013, _017->_014,
 *   _018->_015, _019->_016, _020->_017, _031->_018, _030->_019, _028->_020, _027->_021, _021->_022, _022->_023, _023->_024,
 *   _024->_025). Logs carry reason strings, not numbers, so older logs map to the new keys too. Sell keys unchanged.
 *   TRIGGER_COND names the trigger key for the pipeline's sub-step nodes
 * - [2026-11-08] type=fix - keyRemap / remapKey: app keys are translated to the server's numbers through the function names in the
 *   server condition list, so labels stay right when the server and the app were updated at different times (a sell trade was
 *   shown as "_022 (조건 목록에 없음)" by an app built before the sell renumbering)
 * - [2026-11-08] type=refactor - Sell condition numbers follow the evaluation order too (server r006 _001.._015): _006->_005,
 *   _007->_006, _023->_007, _015->_008, _016->_009, _017->_010, _018->_011, _019->_012, _020->_013, _021->_014, _022->_015
 */

import type { Condition, ConditionsDoc, EvalResult, Side } from './types';
import { parseSteps, splitSegments } from './logLine';

export interface Classified {
  side: Side;
  condKey: string;
  sub?: string;
  result: EvalResult;
  reason: string;
  steps?: Record<string, 'P' | 'F' | '-'>;
}

export type ClassifyOutcome =
  | { kind: 'event'; ev: Classified }
  | { kind: 'ignore' }
  | { kind: 'unknown'; side: Side; reason: string };

interface Rule {
  p: string;
  cond: string;
  sub?: string;
}

function sortRules(rules: Rule[]): Rule[] {
  return [...rules].sort((a, b) => b.p.length - a.p.length);
}

function matchLongest(rules: Rule[], text: string): Rule | undefined {
  for (const r of rules) if (text.startsWith(r.p)) return r;
  return undefined;
}

/** condKey of the entry trigger (its sub-steps are counted as "<key>/<sub>") */
export const TRIGGER_COND = '_009';

/** trigger (TRIGGER_COND) sub-steps, in display order */
export const TRIGGER_SUBS = ['trigger', 'candle', 'live', 'chase', 'legacy', 'other'] as const;
export const TRIGGER_SUB_LABEL: Record<string, string> = {
  trigger: '크로스 트리거',
  candle: '캔들/갭 가드',
  live: '현재가 가드',
  chase: '크로스 대비 추격',
  legacy: 'legacy 1분 트리거',
  other: '기타',
};

// [REJECT] reasons (guide 3.1)
const BUY_REJECT_RULES = sortRules([
  { p: 'SESSION_OPEN_WARMUP', cond: '_002' },
  { p: 'EDGE_SIGNAL_', cond: '_009', sub: 'trigger' }, // [2026-11-12] TRADING_STRATEGY_MODE=edge: _e09_edge_ai_signal
  { p: 'TRIPLE_CROSS_MISSING_', cond: '_009', sub: 'trigger' },
  { p: 'TICK_LOW_DENSITY_', cond: '_009', sub: 'trigger' },
  { p: 'TRIPLE_CROSS_1MIN_CHASE_BUY_BB_GAP', cond: '_009', sub: 'candle' },
  { p: 'TRIPLE_CROSS_1MIN_CHASE_BUY_INTRABAR', cond: '_009', sub: 'candle' },
  { p: 'TRIPLE_CROSS_1MIN_CANDLE_NOT_BULLISH', cond: '_009', sub: 'candle' },
  { p: 'TRIPLE_CROSS_CANDLE_NOT_BULLISH', cond: '_009', sub: 'candle' },
  { p: 'TRIPLE_CROSS_1MIN_LIVE_DROP', cond: '_009', sub: 'live' },
  { p: 'TRIPLE_CROSS_1MIN_LIVE_BELOW_BB_MID', cond: '_009', sub: 'live' },
  { p: 'TRIPLE_CROSS_CHASE_FROM_CROSS', cond: '_009', sub: 'chase' },
  { p: 'TRIPLE_CROSS_', cond: '_009', sub: 'other' },
  { p: 'HYBRID_1MIN_TRIGGER_', cond: '_009', sub: 'legacy' },
  { p: 'HYBRID_1MIN_FRAME_UNAVAILABLE', cond: '_009', sub: 'legacy' },
  { p: 'MARKET_REGIME_DOWN', cond: '_010' },
  { p: 'HYBRID_3MIN_CTX_INSUFFICIENT_BARS', cond: '_011' },
  { p: 'HYBRID_3MIN_CTX_MISSING_INDICATOR', cond: '_011' },
  { p: 'HYBRID_3MIN_CTX_LOW_ENTRY_ATR', cond: '_012' },
  { p: 'GC_VOL_', cond: '_012' }, // [2026-11-12] golden-cross volatility gate: 1-minute ATR% below the limit / unavailable
  { p: 'HYBRID_3MIN_CTX_LOW_VOL_MA_ABS', cond: '_013' },
  { p: 'HYBRID_3MIN_CTX_LOW_ABS_VOLUME', cond: '_013' },
  { p: 'HYBRID_3MIN_CTX_LOW_VOLUME_RATIO', cond: '_013' },
  { p: 'HYBRID_3MIN_CTX_LOW_TURNOVER', cond: '_013' },
  { p: 'EXCESSIVE_RISE_FROM_PREV_CLOSE', cond: '_015' },
  { p: 'ANTI_CHASE_', cond: '_016' },
  { p: 'OPENING_GAP', cond: '_017' },
  { p: 'QUOTE_STATUS_BLOCK', cond: '_018' },
  { p: 'ACCOUNT_SYNC_STALE', cond: '_019' },
  { p: 'EDGE_AI_LOW_PROB', cond: '_020' },
  { p: 'SPREAD_NO_QUOTE', cond: '_021' },
  { p: 'SPREAD_TOO_WIDE', cond: '_021' },
  { p: 'INSUFFICIENT_BUYING_POWER_OR_BUDGET', cond: '_023' },
  { p: 'STALE_LIVE_PRICE', cond: '_024' },
  { p: 'ORDERBOOK_ASK_THIN', cond: '_025' },
]);

// [BUY SKIP] reasons
const BUY_SKIP_RULES = sortRules([
  { p: 'ALREADY_TRADED_TODAY_UNTIL_SELL', cond: '_005' },
  { p: 'HARD_STOP_REENTRY_BLOCKED_TODAY', cond: '_006' },
  { p: 'DAILY_LOSS_LIMIT', cond: '_007' },
  { p: 'MAX_OPEN_POSITIONS', cond: '_008' },
]);

// [SELL TRIGGER] / [SELL HOLD] reasons (guide 3.2)
const SELL_RULES = sortRules([
  { p: 'AUX_BLOCKED_SCORE', cond: '_015' },
  { p: 'AUX_TRIGGER_BUFFER_BLOCK', cond: '_015' },
  { p: 'HARD_STOP_LOSS', cond: '_004' },
  { p: 'NET_STOP_LOSS', cond: '_004' }, // [2026-11-12] AUTO_SELL_MODE=stop_only: the only automatic sell (r006 _s04)
  { p: 'EDGE_TP_', cond: '_010' }, // [2026-11-12] edge mode sells (r006 _e10 / _e11 / _e12)
  { p: 'EDGE_SL_', cond: '_011' },
  { p: 'EDGE_TIME_', cond: '_012' },
  { p: 'TP1_PARTIAL', cond: '_005' },
  { p: 'TP2_SURGE_LADDER', cond: '_006' },
  { p: 'TP3_SURGE_LADDER', cond: '_006' },
  { p: 'PROFIT_LOCK_', cond: '_007' },
  { p: 'ATR_TAKE_PROFIT', cond: '_008' },
  { p: 'POST_BUY_ENTRY_DROP', cond: '_009' },
  { p: 'BREAKEVEN_FAIL', cond: '_010' },
  { p: 'NO_TREND_EXIT', cond: '_011' },
  { p: 'ATR_STOP_LOSS', cond: '_012' },
  { p: 'TRAILING_STOP_GIVEBACK', cond: '_013' },
  { p: 'TP_EXTENSION_TRAIL', cond: '_013' },
  { p: 'AUX_REVERSAL_SCORE', cond: '_015' },
  { p: 'LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED', cond: '_015' },
  { p: 'PEAK_NEXT_BAR_BEARISH', cond: '_003' },
  { p: 'REGULAR_CLOSE_SESSION_FORCE_CLOSE_ALL', cond: 'SESSION_CLOSE' },
  { p: 'NXT_CLOSE', cond: 'SESSION_CLOSE' },
]);

/** Sell reason -> condKey (also used for trades from the buy_sell log). */
export function sellReasonToCond(reason: string): string | undefined {
  const r = matchLongest(SELL_RULES, reason.replace(/^reason=/, ''));
  return r?.cond;
}

/** Buy reject reason -> condKey/sub (exported for tests and the unknown list). */
export function buyRejectToCond(reason: string): { cond: string; sub?: string } | undefined {
  const r = matchLongest(BUY_REJECT_RULES, reason);
  return r ? { cond: r.cond, sub: r.sub } : undefined;
}

const BUY_ORDER_TAGS = new Set([
  'BUY SUBMIT',
  'BUY PENDING',
  'BUY PARTIAL',
  'BUY PART CXL',
  'BUY RP CNCL',
  'BUY RP RSUB',
  'BUY RP QUIT',
  'BUY RP ABAND',
  'BUY EXEC',
  'BUY NOFILL',
  'BUY CXL REJ',
  'REENTRY LIVE',
]);
const SELL_ORDER_TAGS = new Set(['SELL SUBMIT', 'SELL FILLED', 'SELL EXEC', 'SELL PARTIAL', 'SEL RSUB ERR', 'SELL RSUB ERR']);

/** sub of status events ([CHECK] / [BAR_SKIP]) - info dots on the PREP row, not condition evaluations */
export const STATUS_SUB = 'status';
/** [CHECK]: bar count / live price / confirmed bar; [BAR_SKIP]: intrabar 3min RSI/MFI out of range -> confirmed 3min bar used (not a reject) */
const STATUS_TAGS = new Set(['CHECK', 'BAR_SKIP']);

/** Tags that are understood but carry no condition event (handled elsewhere or noise). */
export const IGNORED_TAGS = new Set([
  'TICK CROSS',
  'WL SPLIT',
  'MKT REGIME',
  'ACT PROMOTE',
  'ACT DROPOUT',
  'ACT GRADUATE',
  'ACT REENTER',
  'WARMUP', // prior-day bars loaded for indicator warmup
  'AUTO SELL', // [2026-11-12] AUTO_SELL_MODE: "scheduled liquidation skipped" notice, not a condition
  'BOT LEDGER', // [2026-11-13] r023 bot / manual quantity ledger changes, not a condition
  'SELL LIMIT', // [2026-11-13] AUTO_SELL_SCOPE=bot_only: a sell was cut down to the bot's quantity (the order line follows)
  'EDGE ONLY', // [2026-11-12] Edge-AI-only paper trading (r022): virtual buys / sells, not a condition of the real pipeline
]);

function reasonCandidates(rest: string): string[] {
  const segs = splitSegments(rest);
  const out: string[] = [];
  if (segs[0]) out.push(segs[0].replace(/^reason=/, ''));
  for (const s of segs) {
    const m = /(?:^|\s)reason=(\S+)/.exec(s);
    if (m) out.push(m[1]);
  }
  return out;
}

function firstMatch(rules: Rule[], rest: string): { rule: Rule; reason: string } | undefined {
  for (const c of reasonCandidates(rest)) {
    const rule = matchLongest(rules, c);
    if (rule) return { rule, reason: c };
  }
  return undefined;
}

/** Sell reasons can sit anywhere in the body - scan every token. */
function scanTokens(rules: Rule[], rest: string): { rule: Rule; reason: string } | undefined {
  const viaFields = firstMatch(rules, rest);
  if (viaFields) return viaFields;
  for (const tok of rest.split(/[\s|]+/)) {
    const t = tok.replace(/^reason=/, '');
    const rule = matchLongest(rules, t);
    if (rule) return { rule, reason: t };
  }
  return undefined;
}

const firstSeg = (rest: string) => splitSegments(rest)[0] || '';

/**
 * Classify one tagged main-log line that belongs to a symbol.
 * `tag` is trimmed; `rest` is the body after "{code}_{name} |".
 */
export function classify(tag: string, rest: string): ClassifyOutcome {
  const ev = (e: Classified): ClassifyOutcome => ({ kind: 'event', ev: e });

  switch (tag) {
    case 'SKIP': {
      if (rest.includes('can_trade_code_now=False')) return ev({ side: 'buy', condKey: '_001', result: 'fail', reason: 'can_trade_code_now=False' });
      if (/frame=None|INDICATOR_WARMUP_BARS|bars=\d+\s*</.test(rest)) return ev({ side: 'buy', condKey: 'PREP', result: 'fail', reason: firstSeg(rest) });
      return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
    }
    case 'REJECT':
    case 'BUY REJECT': {
      const hit = firstMatch(BUY_REJECT_RULES, rest);
      const steps = parseSteps(rest);
      if (!hit) return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
      return ev({ side: 'buy', condKey: hit.rule.cond, sub: hit.rule.sub, result: 'fail', reason: hit.reason, steps });
    }
    case 'BUY SKIP': {
      // hard-stop circuit breaker, removed from r003 on 2026-11-04 (only in older logs): a global pause of all buys, not a condition -
      // gray status dots on the PREP row (thinned like [CHECK]), not counted as blocks
      if (rest.includes('CIRCUIT_BREAKER_ACTIVE')) return ev({ side: 'buy', condKey: 'PREP', sub: STATUS_SUB, result: 'info', reason: firstSeg(rest) });
      const hit = firstMatch(BUY_SKIP_RULES, rest);
      if (!hit) return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
      return ev({ side: 'buy', condKey: hit.rule.cond, result: 'fail', reason: hit.reason });
    }
    case 'BUY HOLD': {
      if (rest.includes('WAIT_NEXT_POLL_CONFIRM')) return ev({ side: 'buy', condKey: '_022', result: 'hold', reason: firstSeg(rest).replace(/^reason=/, '') + (/(count=\S+)/.exec(rest) ? ' ' + /(count=\S+)/.exec(rest)![1] : '') });
      return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
    }
    case 'EDGE AI':
      return ev({ side: 'buy', condKey: '_020', result: 'info', reason: firstSeg(rest) });
    case 'SPREAD':
      return ev({ side: 'buy', condKey: '_021', result: 'info', reason: firstSeg(rest) });
    case 'TICK TRIG':
      return ev({ side: 'buy', condKey: '_009', sub: 'trigger', result: 'info', reason: firstSeg(rest) });
    case 'BUY EVAL': {
      const m = /^OK\s+(\S+)/.exec(rest.trim());
      if (m) return ev({ side: 'buy', condKey: 'ORDER', sub: 'EVAL OK', result: 'pass', reason: m[1] });
      return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
    }
    case 'PENDING': {
      const side: Side = /side=SELL/.test(rest) ? 'sell' : 'buy';
      return ev({ side, condKey: 'ORDER', sub: 'PENDING', result: 'hold', reason: firstSeg(rest) });
    }
    case 'SELL HOLD': {
      const hit = scanTokens(SELL_RULES, rest);
      if (!hit) return { kind: 'unknown', side: 'sell', reason: firstSeg(rest) };
      return ev({ side: 'sell', condKey: hit.rule.cond, result: 'hold', reason: hit.reason });
    }
    case 'SELL TRIGGER': {
      const hit = scanTokens(SELL_RULES, rest);
      if (!hit) return { kind: 'unknown', side: 'sell', reason: firstSeg(rest) };
      return ev({ side: 'sell', condKey: hit.rule.cond, result: 'pass', reason: hit.reason });
    }
    case 'SELL EVAL': {
      // r006 logs "OK <reason>" after the sell order was placed (AUX / BB down-cross path, no separate [SELL TRIGGER])
      const hit = /^OK\s/.test(rest.trim()) ? scanTokens(SELL_RULES, rest) : null;
      if (!hit) return { kind: 'unknown', side: 'sell', reason: firstSeg(rest) };
      return ev({ side: 'sell', condKey: hit.rule.cond, result: 'pass', reason: hit.reason });
    }
    case 'TRADE': {
      // "[TRADE] <code> | BUY REPRICE RESUBMIT | qty=..." (the other [TRADE] lines carry no symbol and are global)
      if (rest.includes('BUY REPRICE RESUBMIT')) return ev({ side: 'buy', condKey: 'ORDER', sub: 'BUY REPRICE RESUBMIT', result: 'info', reason: firstSeg(rest) });
      return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
    }
    case 'HOLD SKIP':
      return ev({ side: 'sell', condKey: 'PREP', sub: STATUS_SUB, result: 'info', reason: firstSeg(rest) });
    case 'TP_EXTENSION':
      return ev({ side: 'sell', condKey: '_008', result: 'info', reason: firstSeg(rest) });
  }
  if (BUY_ORDER_TAGS.has(tag)) {
    const result: EvalResult = tag === 'BUY EXEC' ? 'pass' : tag === 'BUY NOFILL' || tag === 'BUY RP ABAND' ? 'fail' : 'info';
    return ev({ side: 'buy', condKey: 'ORDER', sub: tag, result, reason: firstSeg(rest) });
  }
  if (SELL_ORDER_TAGS.has(tag)) {
    const result: EvalResult = tag === 'SELL FILLED' || tag === 'SELL EXEC' ? 'pass' : 'info';
    return ev({ side: 'sell', condKey: 'ORDER', sub: tag, result, reason: firstSeg(rest) });
  }
  if (STATUS_TAGS.has(tag)) return ev({ side: 'buy', condKey: 'PREP', sub: STATUS_SUB, result: 'info', reason: `${tag} ${firstSeg(rest)}`.trim() });
  if (IGNORED_TAGS.has(tag)) return { kind: 'ignore' };
  return { kind: 'unknown', side: 'buy', reason: firstSeg(rest) };
}

/** STEPS diagnostic names -> condKey (colors other gates on a rejected poll) */
export const STEP_TO_COND: Record<string, string> = {
  '1min_trigger': '_009',
  min_entry_atr_volatility: '_012',
  min_liquidity_safety: '_013',
};

/** Buy conditions that never log (shown as "silent") and the score-only condition (dashed). */
export const SILENT_BUY = new Set(['_003', '_004']);
export const INFO_ONLY_BUY = new Set(['_014']);

/** "_012" for a condition */
export const condKeyOf = (no: number) => '_' + String(no).padStart(3, '0');

// [2026-11-08] The rules above use the app's condition numbers (server r005 / r006 as of 2026-11-08). When the server numbers
// differ (server and app updated at different times), keys are translated through the function name: server func
// "_007_profit_lock" -> stem "profit_lock" -> server key "_007". Stems not found in the server list keep the app key.
const BUY_STEMS: Record<string, string> = {
  _001: 'entry_time_window', _002: 'session_open_warmup', _003: 'trade_cooldown', _004: 'same_bar_signal_dedup',
  _005: 'no_open_buy_exposure', _006: 'hard_stop_reentry_block', _007: 'daily_loss_limit', _008: 'max_open_positions',
  _009: 'hybrid_1min_trigger', _010: 'market_regime', _011: 'context_frame_ready', _012: 'min_entry_atr_volatility',
  _013: 'min_liquidity_safety', _014: 'context_score_record', _015: 'excessive_rise_from_prev_close',
  _016: 'anti_chase_day_extension', _017: 'opening_gap_volume_gate', _018: 'quote_status', _019: 'account_sync_fresh',
  _020: 'edge_ai_meta_filter', _021: 'entry_spread', _022: 'consecutive_poll_confirm', _023: 'affordable_buy_qty',
  _024: 'fresh_live_price', _025: 'orderbook_ask_not_thin',
};
const SELL_STEMS: Record<string, string> = {
  _001: 'stale_live_price_guard', _002: 'update_position_tracking', _003: 'peak_next_bar_bearish_exit', _004: 'hard_stop_loss',
  _005: 'staged_tp1_partial', _006: 'surge_ladder_tp', _007: 'profit_lock', _008: 'atr_take_profit',
  _009: 'post_buy_entry_drop_guard', _010: 'breakeven_fail_guard', _011: 'no_trend_time_exit', _012: 'atr_stop_loss',
  _013: 'trailing_stop', _014: 'same_bar_sell_dedup', _015: 'shared_reversal_sell',
};

/** server function name without its number prefix ("_007_profit_lock" -> "profit_lock", "_b09_basic_x" -> "basic_x") */
export const funcStem = (func: string) => func.replace(/^_(?:\d{3}|b\d{2})_/, '');

export type KeyRemap = Map<string, string>;

/** app key -> server key for one side, only for keys whose number differs on the server */
export function keyRemap(doc: ConditionsDoc | null | undefined, side: Side): KeyRemap {
  return keyRemapOf(doc ? (side === 'buy' ? doc.buy : doc.sell) : [], side);
}

/** keyRemap for one side's condition list */
export function keyRemapOf(list: Condition[] | undefined, side: Side): KeyRemap {
  const out: KeyRemap = new Map();
  if (!list?.length) return out;
  const byStem = new Map(list.map((c) => [funcStem(c.func), condKeyOf(c.no)]));
  for (const [appKey, stem] of Object.entries(side === 'buy' ? BUY_STEMS : SELL_STEMS)) {
    const k = byStem.get(stem);
    if (k && k !== appKey) out.set(appKey, k);
  }
  return out;
}

/** translate an app key (also "<key>/<sub>"); pseudo keys (PREP, ORDER, ...) pass through */
export function remapKey(remap: KeyRemap, key: string): string {
  if (!remap.size) return key;
  const slash = key.indexOf('/');
  const base = slash < 0 ? key : key.slice(0, slash);
  const to = remap.get(base);
  return to ? to + (slash < 0 ? '' : key.slice(slash)) : key;
}

/** Group similar reasons in the unknown list: digits -> # */
export function reasonShape(reason: string): string {
  return reason.replace(/-?\d[\d.,]*/g, '#').slice(0, 80);
}
