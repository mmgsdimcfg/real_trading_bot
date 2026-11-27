/*
 * Update log
 * - [2026-11-27] type=feat - 매매 방식 현황 dashboard (user 2026-11-27 mock-up + guide, Codex review): the five run units (규칙 /
 *   Edge AI 가상 · 실전 / VWAP 가상 · 실전) with what is STORED and what the RUNNING bot does kept apart, and the per-method
 *   target symbols (server RULE_ / EDGE_ / VWAP_TARGET_CODES). Everything here is read from the control state (r016) - no state
 *   is guessed: a bot that is not running or did not report gives 'botoff' / 'unknown', never "실행 중".
 *   Real and paper of one method are exclusive in the bot (real orders on -> its paper runner only closes), so the paper unit's
 *   switch is the method's own switch ("방식 사용") and the real unit's switch is its live-orders switch.
 */

import type { ControlState } from './control';
import type { PaperModeState } from './paper';
import { strategyModes, type TradeStrategy } from './strategy';

export type UnitKey = 'rule' | 'edge_paper' | 'edge_live' | 'vwap_paper' | 'vwap_live';
export const UNIT_KEYS: UnitKey[] = ['rule', 'edge_paper', 'edge_live', 'vwap_paper', 'vwap_live'];
export const UNIT_METHOD: Record<UnitKey, TradeStrategy> = { rule: 'rule', edge_paper: 'edge', edge_live: 'edge', vwap_paper: 'vwap', vwap_live: 'vwap' };
export const UNIT_MODE: Record<UnitKey, 'live' | 'paper'> = { rule: 'live', edge_paper: 'paper', edge_live: 'live', vwap_paper: 'paper', vwap_live: 'live' };
export const UNIT_KO: Record<UnitKey, string> = { rule: '규칙 매매', edge_paper: 'Edge AI 매매', edge_live: 'Edge AI 매매', vwap_paper: 'VWAP 매매', vwap_live: 'VWAP 매매' };
export const UNIT_SHORT: Record<UnitKey, string> = { rule: '규칙', edge_paper: 'AI(가상)', edge_live: 'AI(실전)', vwap_paper: 'VWAP(가상)', vwap_live: 'VWAP(실전)' };

/** run: new entries by the running bot · pending: running and stored differ (applied by a restart) · closing: paper unit while
 *  the method places real orders (no new paper entries) · stop: off · botoff: the bot is not running · unknown: not reported */
export type UnitState = 'run' | 'pending' | 'closing' | 'stop' | 'botoff' | 'unknown';

export interface UnitInfo {
  key: UnitKey;
  method: TradeStrategy;
  mode: 'live' | 'paper';
  /** what the unit's switch shows (stored): rule = rule buys, paper = the method itself, live = its real orders. null = the
   *  server does not know the key / not read yet */
  switchOn: boolean | null;
  /** the switch cannot be used here, and why ('' = usable) */
  locked: string;
  /** new entries of this unit under the STORED settings (after the next start) */
  entryStored: boolean | null;
  /** new entries of this unit in the RUNNING bot (null = not running / not reported) */
  entryRunning: boolean | null;
  state: UnitState;
  stateKo: string;
}

type Modes = Pick<ControlState, 'strategy' | 'ruleBuy' | 'paper' | 'paperLogOnly' | 'edgeLive' | 'vwapMode' | 'vwapLive'>;
const KEYS: (keyof Modes)[] = ['strategy', 'ruleBuy', 'paper', 'paperLogOnly', 'edgeLive', 'vwapMode', 'vwapLive'];
/** the same state with every running value dropped (what the next start would do) / required (what the running bot does) */
function view(c: Modes, which: 'stored' | 'running'): Modes {
  const out: Modes = {};
  for (const k of KEYS) {
    const m = c[k] as PaperModeState | undefined;
    if (!m) continue;
    out[k] = which === 'stored' ? { ...m, running: null } : { ...m, effective: m.running ?? '' };
  }
  return out;
}
const runningKnown = (c: Modes, keys: (keyof Modes)[]): boolean => keys.every((k) => !c[k] || (c[k] as PaperModeState).running !== null);

const STATE_KO: Record<UnitState, string> = { run: '실행 중', pending: '적용 대기', closing: '신규 진입 중지', stop: '중지', botoff: '봇 꺼짐', unknown: '확인 불가' };

/** what the bot itself said beyond the settings [Codex review: a switch that is on is not always a method that buys] */
export interface UnitHints {
  /** r016 status.dry_run: the bot sends no order at all */
  dryRun?: boolean;
  /** the last start / block line of a real-order strategy in the day's log: 'off' = it said new buys are off (why), 'blocked' =
   *  stopped for now (a recent line) */
  liveNote?: Partial<Record<'edge' | 'vwap', 'on' | 'off' | 'blocked'>>;
}

/** `botRunning`: r016 status.running (null = no status yet / unreachable) */
export function unitInfos(c: Modes, botRunning: boolean | null, hints: UnitHints = {}): UnitInfo[] {
  const stored = strategyModes(view(c, 'stored'));
  const running = strategyModes(view(c, 'running'));
  const sm = c.strategy?.effective ?? null;
  const edgeOnly = sm === 'edge';
  const edgeOnlyRunning = (c.strategy?.running ?? null) === 'edge';
  const normal = sm === null || sm === 'normal';
  const methodKeys: Record<TradeStrategy, (keyof Modes)[]> = { rule: ['strategy', 'ruleBuy'], edge: ['strategy', 'paper', 'edgeLive', 'paperLogOnly'], vwap: ['strategy', 'vwapMode', 'vwapLive'] };
  const methodOn: Record<TradeStrategy, boolean | null> = {
    rule: c.ruleBuy ? c.ruleBuy.effective === 'true' : c.strategy ? true : null,
    edge: c.paper ? c.paper.effective === 'paper' : null,
    vwap: c.vwapMode ? c.vwapMode.effective === 'paper' : null,
  };
  return UNIT_KEYS.map((key) => {
    const method = UNIT_METHOD[key];
    const mode = UNIT_MODE[key];
    const s = stored[method];
    const r = running[method];
    const entryStored = mode === 'live' ? s.live : s.paper;
    const known = botRunning === true && runningKnown(c, methodKeys[method]);
    let entryRunning = known ? (mode === 'live' ? r.live : r.paper) : null;
    // Edge AI real orders beside the rule trade need the Edge AI trade itself on in the running bot (r003 _edge_live_config_ok)
    if (key === 'edge_live' && entryRunning && !edgeOnlyRunning && c.paper && c.paper.running !== 'paper') entryRunning = false;
    let switchOn: boolean | null;
    let locked = '';
    if (key === 'rule') {
      switchOn = methodOn.rule;
      if (!c.ruleBuy) locked = '서버가 이 설정을 모릅니다';
      else if (!normal) locked = edgeOnly ? 'Edge AI 단독 모드' : '서버 매매 방식이 normal 이 아님';
    } else if (mode === 'paper') {
      switchOn = methodOn[method];
      if (switchOn === null) locked = '서버가 이 설정을 모릅니다';
      else if (method === 'edge' && !normal) locked = edgeOnly ? 'Edge AI 단독 모드 · 실전' : '서버 매매 방식이 normal 이 아님';
    } else {
      const live = method === 'edge' ? c.edgeLive : c.vwapLive;
      switchOn = live ? live.effective === 'true' && methodOn[method] === true : null;
      if (!live) locked = '서버가 이 설정을 모릅니다';
      else if (method === 'edge' && edgeOnly) locked = '단독 모드 · 실전';
      else if (methodOn[method] !== true) locked = '방식 꺼짐';
    }
    let state: UnitState;
    if (botRunning === null) state = 'unknown';
    else if (!botRunning) state = entryStored ? 'botoff' : 'stop';
    else if (entryRunning === null) state = 'unknown';
    else if (entryRunning) state = entryStored === false ? 'pending' : 'run';
    else if (entryStored) state = 'pending';
    else if (mode === 'paper' && r.live) state = 'closing';
    else state = 'stop';
    let stateKo = STATE_KO[state];
    if (state === 'pending') stateKo = entryRunning ? '실행 중 · 재시작 후 꺼짐' : '적용 대기 · 재시작 후 켜짐';
    if (state === 'closing') stateKo = '신규 진입 중지 · 실전 우선';
    if (state === 'botoff') stateKo = '봇 꺼짐 · 설정 ON';
    if (state === 'run' && mode === 'live') {
      const note = method === 'rule' ? undefined : hints.liveNote?.[method];
      if (hints.dryRun) {
        state = 'stop';
        stateKo = '주문 안 함 (dry-run)';
        entryRunning = false;
      } else if (note === 'off') {
        state = 'stop';
        stateKo = '신규 매수 꺼짐 (봇 기록)';
        entryRunning = false;
      } else if (note === 'blocked') {
        state = 'closing';
        stateKo = '신규 매수 중단 (봇 기록)';
        entryRunning = false;
      } else if (method !== 'rule') stateKo = '실행 중 (설정 기준)';
    }
    return { key, method, mode, switchOn, locked, entryStored, entryRunning, state, stateKo };
  });
}

/** units whose new entries the running bot confirmed */
export const runningUnits = (units: UnitInfo[]): number => units.filter((u) => u.entryRunning === true).length;

// ---- per-method target symbols ------------------------------------------------------------------------------------------------
export const METHOD_TARGET_KEYS: Record<TradeStrategy, string> = { rule: 'RULE_TARGET_CODES', edge: 'EDGE_TARGET_CODES', vwap: 'VWAP_TARGET_CODES' };
export const METHOD_TARGET_MAX = 300;

export interface MethodTargetState {
  /** stored list (the next start); empty = the method's whole scope */
  effective: string[];
  /** the running bot's list; null = not running / a bot from before the lists */
  running: string[] | null;
  pending: boolean;
}
export type MethodTargets = Partial<Record<TradeStrategy, MethodTargetState>>;

const codes = (v: unknown): string[] | null => (Array.isArray(v) && v.every((x) => typeof x === 'string') ? (v as string[]).slice().sort() : null);

/** `get RULE_TARGET_CODES EDGE_TARGET_CODES VWAP_TARGET_CODES` data -> state (a method the server does not know is left out) */
export function parseMethodTargets(data: unknown): MethodTargets {
  const vals = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values;
  const out: MethodTargets = {};
  for (const m of Object.keys(METHOD_TARGET_KEYS) as TradeStrategy[]) {
    const v = vals?.[METHOD_TARGET_KEYS[m]];
    const eff = codes(v?.effective);
    if (!v || !eff) continue;
    const run = v.running === null || v.running === undefined ? null : codes(v.running);
    out[m] = { effective: eff, running: run, pending: !!v.pending || (run !== null && run.join(',') !== eff.join(',')) };
  }
  return out;
}

/** the list the board marks rows with: the running bot's when it is known, else the stored one. `from` says which */
export function targetsInUse(t: MethodTargetState | undefined, botRunning: boolean | null = false): { list: string[]; from: 'running' | 'stored' | 'none' | 'unknown' } {
  if (!t) return { list: [], from: 'none' };
  if (t.running !== null) return { list: t.running, from: 'running' };
  // [Codex review] the bot runs but did not report its list: the stored list may not be what it uses - nothing is marked
  if (botRunning !== false) return { list: [], from: 'unknown' };
  return { list: t.effective, from: 'stored' };
}

/** what to store for a picked set: all watched symbols picked (or none) = no list. Codes outside `watched` that are kept stay. */
export function targetListToStore(picked: Iterable<string>, watched: string[]): string[] {
  const set = new Set(picked);
  const all = watched.length > 0 && watched.every((c) => set.has(c));
  if (!set.size || (all && [...set].every((c) => watched.includes(c)))) return [];
  return [...set].filter((c) => /^[0-9A-Z]{6}$/.test(c)).sort();
}

export const targetsKo = (list: string[]): string => (list.length ? `${list.length}종목 지정` : '전체 (지정 없음)');

// ---- automatic composition of the target lists (server r031 / r016 `targets`) -------------------------------------------------
/** [2026-11-27] user 2026-11-27: A = the scanner's picks; rule = A + the scan ranking up to sizes.rule (30); Edge AI = A + the
 *  ranking after that up to sizes.edge (50); VWAP = the union. include / exclude: kept from day to day. auto: the post-close job
 *  composes (false: it copies the picks as before; the composition can still be applied by hand). */
export interface TargetsConfig {
  auto: boolean;
  sizes: { rule: number; edge: number };
  exclude_funds: boolean;
  include: Record<TradeStrategy, string[]>;
  exclude: Record<TradeStrategy, string[]>;
  updated_at?: string;
  updated_by?: string;
}
/** where a watched symbol of the composition comes from */
export type TargetSrc = 'A' | 'rule' | 'edge' | 'manual';
export interface TargetRow {
  name: string;
  src: TargetSrc;
  /** rank / score / eligible / fail reasons of the scan (null: the scan does not know the symbol) */
  rank: number | null;
  score: number | null;
  eligible: boolean | null;
  fail: string;
  /** the methods whose list holds it */
  in: TradeStrategy[];
  /** the methods whose always-include list holds it */
  included: TradeStrategy[];
}
export interface TargetsComposition {
  a: string[];
  rule_extra: string[];
  edge_extra: string[];
  lists: Record<TradeStrategy, string[]>;
  watch: { code: string; name: string }[];
  rows: Record<string, TargetRow>;
  tick_cap: number;
  /** watched symbols past the tick collector's limit: no tick feed (the rule trade and the VWAP trade cannot buy them) */
  no_tick: string[];
  dropped: string[];
  sizes: { rule: number; edge: number };
  /** extras the ranking could not supply */
  short: { rule: number; edge: number };
  /** always-excluded codes that the composition would have listed */
  excluded: Record<TradeStrategy, string[]>;
  scan_file?: string | null;
  fund_filter?: boolean;
  /** fingerprint of the lists + the watch order: sent back with `apply` so what is stored is what was shown */
  digest?: string;
}
export interface TargetsDoc {
  ok: boolean;
  error?: string;
  /** scan date the composition was made from */
  date?: string | null;
  config: TargetsConfig;
  /** 'missing' = no config stored yet (defaults), 'broken' = the file cannot be read */
  config_state: 'missing' | 'ok' | 'broken';
  composition?: TargetsComposition;
  /** the three lists stored now (what r003 uses after its next start) */
  stored: Record<TradeStrategy, string[]>;
  watch_now: string[];
  tick_cap: number;
  /** watched symbols r003 polls at a time (the rest wait and rotate in); 0 = not told */
  active_size?: number;
  applied?: boolean;
  for_trading_day?: string | null;
}
export const TARGET_SRC_KO: Record<TargetSrc, string> = { A: 'A (선별)', rule: '순위 추가 (규칙)', edge: '순위 추가 (Edge)', manual: '수동 포함' };

/** the composed list of a method equals what is stored for it (order ignored) */
export function targetsApplied(doc: TargetsDoc | null | undefined, m: TradeStrategy): boolean | null {
  const c = doc?.composition;
  if (!doc || !c) return null;
  const a = [...c.lists[m]].sort().join(',');
  const b = [...(doc.stored[m] || [])].sort().join(',');
  return a === b;
}
