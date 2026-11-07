/*
 * Update log
 * - [2026-11-05] type=feat - r003 remote control (guide 20261105): response types of live_trading/r016_control.py, the command
 *   builder (cd + R016_BY + python + quoted args), the stdout JSON extractor/parser (rc 0 / 2 rejected / 3 systemctl), the
 *   service button rules, the market-hours check on the server's KST timestamp, value-editor parsing per r001 type, the changed-
 *   keys collector for `set`, the history formatter and the watchlist row checks. Pure functions only (unit-tested in
 *   tests/control.test.ts); the app never re-implements the server's validation - it only blocks input it cannot send safely
 */

import { shellPath, shellQuote } from './shell';

// ---- response types --------------------------------------------------------------------------

export interface ServiceState {
  available: boolean;
  unit?: string;
  loaded?: boolean;
  /** active | inactive | failed | activating | deactivating */
  active_state?: string;
  /** e.g. running / dead / auto-restart */
  sub_state?: string;
  main_pid?: number | null;
  started_at?: string | null;
  restarts?: number;
  result?: string;
}

export interface ExternalProc {
  pid: number;
  cmdline: string;
}

export interface PendingFile {
  path: string;
  mtime: string | null;
  changed_since_start: boolean;
}

export interface R016Status {
  generated_at: string;
  service: ServiceState;
  running: boolean;
  external_r003: ExternalProc[];
  env: Record<string, string>;
  dry_run: boolean;
  risk_profile: string;
  override_count: number;
  override_errors: string[];
  pending: { overrides?: PendingFile; env?: PendingFile; watchlist?: PendingFile };
  pending_restart: boolean;
  latest_main_log?: string | null;
  watchlist_codes?: number;
}

export type ValueType = 'bool' | 'int' | 'float' | 'str' | 'list' | 'dict' | 'null';

export interface SchemaKey {
  key: string;
  type: ValueType;
  base: unknown;
  override: unknown;
  has_override: boolean;
  effective?: unknown;
  pending: boolean;
  /** conditions.values_meta only - absent in `schema` (every schema key is editable) */
  editable?: boolean;
}

export interface SchemaDoc {
  generated_at: string;
  overrides_file?: string;
  errors?: string[];
  keys: SchemaKey[];
}

export interface SetResult {
  ok: boolean;
  accepted: Record<string, unknown>;
  rejected: Record<string, string>;
  restart_required?: boolean;
  error?: string;
}

export interface HistoryRow {
  ts: string;
  by: string;
  action: string;
  changes: Record<string, unknown>;
}

export interface WatchEntry {
  code: string;
  name: string;
}

export interface WatchlistDoc {
  path: string;
  mtime: string | null;
  header: string[];
  codes: WatchEntry[];
}

export interface WatchlistSetResult {
  ok: boolean;
  count?: number;
  backup?: string | null;
  path?: string;
  error?: string;
}

export interface ServiceResult {
  ok: boolean;
  verb?: string;
  stderr?: string;
  error?: string;
  service?: ServiceState;
  external_r003?: ExternalProc[];
}

/** One parsed r016 call. `ok` = exit 0 and the JSON does not say otherwise. */
export interface R016Result<T = unknown> {
  ok: boolean;
  rc: number | null;
  data: T | null;
  /** human readable reason when !ok (server message verbatim when there is one) */
  error?: string;
  stderr: string;
}

/** Renderer-facing control state pushed by the main process ('control' channel). */
export interface ControlState {
  status: R016Status | null;
  /** epoch ms of the last successful status */
  statusAt?: number;
  /** running service verb ('start' | 'stop' | 'restart') or another long action, null = idle */
  busy: string | null;
  /** last status error (SSH down, r016 missing, ...) */
  error?: string;
  /** result line of the last service action */
  lastAction?: { verb: string; ok: boolean; message: string; at: number };
}

export type RiskProfile = 'conservative' | 'neutral' | 'aggressive';
export const RISK_PROFILES: RiskProfile[] = ['conservative', 'neutral', 'aggressive'];

// ---- command building / parsing --------------------------------------------------------------

export const R016_SCRIPT = 'live_trading/r016_control.py';

/** Characters never sent inside an argument, even though every argument is single-quoted. */
const ARG_FORBIDDEN = /['`;\r\n\0]/;

export function argProblem(arg: string): string | null {
  return ARG_FORBIDDEN.test(arg) ? `허용되지 않는 문자(따옴표 ' · 백틱 · 세미콜론 · 줄바꿈)가 들어 있습니다: ${arg.slice(0, 60)}` : null;
}

/** "autoboth@<windows user>" - only [A-Za-z0-9._-] kept from the user name. */
export function writerName(osUser: string): string {
  const u = String(osUser || '').replace(/[^A-Za-z0-9._-]/g, '') || 'user';
  return `autoboth@${u}`;
}

/**
 * `cd <root>/auto_trading_src && R016_BY=<by> <python> live_trading/r016_control.py <args...>`, every argument single-quoted.
 * Throws when an argument holds a quote, backtick, semicolon or newline (checked before quoting, defence in depth).
 */
export function buildR016Command(remoteRoot: string, pythonPath: string, by: string, args: string[]): string {
  for (const a of [by, ...args]) {
    const p = argProblem(a);
    if (p) throw new Error(p);
  }
  const root = String(remoteRoot || '').replace(/\/+$/, '');
  return `cd ${shellQuote(root + '/auto_trading_src')} && R016_BY=${shellQuote(by)} ${shellPath(pythonPath)} ${R016_SCRIPT} ${args.map(shellQuote).join(' ')}`.trimEnd();
}

/** First "{" .. last "}" of stdout (tolerates warnings printed before/after the JSON). */
export function extractJson(out: string): unknown {
  const a = out.indexOf('{');
  const b = out.lastIndexOf('}');
  if (a < 0 || b < a) return undefined;
  try {
    return JSON.parse(out.slice(a, b + 1));
  } catch {
    return undefined;
  }
}

const lastLines = (s: string, n = 3) => s.trim().split('\n').slice(-n).join(' / ').slice(0, 400);

function rejectedText(rej: unknown): string {
  if (!rej || typeof rej !== 'object') return '';
  return Object.entries(rej as Record<string, unknown>)
    .map(([k, v]) => `${k}: ${String(v)}`)
    .join(', ');
}

export function parseR016Output<T = unknown>(out: string, err: string, code: number | null): R016Result<T> {
  const data = extractJson(out) as Record<string, unknown> | undefined;
  const stderr = err || '';
  if (!data || typeof data !== 'object') {
    return { ok: false, rc: code, data: null, stderr, error: `r016 응답에 JSON이 없습니다 (exit=${code}): ${lastLines(stderr || out) || '출력 없음'}` };
  }
  const saysFail = data.ok === false || typeof data.error === 'string';
  const ok = code === 0 && !saysFail;
  let error: string | undefined;
  if (!ok) {
    const rej = rejectedText(data.rejected);
    error =
      (typeof data.error === 'string' && data.error) ||
      (rej && `거부됨 - ${rej}`) ||
      (typeof data.stderr === 'string' && data.stderr.trim()) ||
      (code === 3 ? 'systemctl 실패' : '') ||
      lastLines(stderr) ||
      `exit=${code}`;
  }
  return { ok, rc: code, data: data as T, stderr, error };
}

// ---- service buttons / time ------------------------------------------------------------------

export interface ButtonState {
  start: boolean;
  stop: boolean;
  restart: boolean;
  /** why start is disabled (shown as a tooltip) */
  startReason?: string;
}

/**
 * start = !running && no external r003 (guide 2.1); stop/restart = service active. Stop is also allowed while systemd is
 * activating / waiting in auto-restart, so a crash loop can be ended from the app.
 */
export function serviceButtons(s: R016Status | null, busy: boolean): ButtonState {
  if (!s || busy) return { start: false, stop: false, restart: false, startReason: busy ? '작업 진행 중' : '상태 없음' };
  const st = s.service?.active_state || '';
  const ext = (s.external_r003?.length || 0) > 0;
  const avail = s.service?.available !== false;
  let startReason: string | undefined;
  if (!avail) startReason = 'systemd 서비스가 없습니다 (autobot-r003.service 설치 필요)';
  else if (ext) startReason = '서비스 밖에서 r003이 실행 중입니다';
  else if (s.running) startReason = '이미 실행 중입니다';
  return {
    start: avail && !ext && !s.running,
    stop: st === 'active' || st === 'activating',
    restart: st === 'active',
    startReason,
  };
}

/** Mon-Fri 09:00 <= t < 15:30 using the server's KST ISO string as is (no time-zone conversion). */
export function isMarketHours(serverIso: string | undefined): boolean {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(serverIso || '');
  if (!m) return false;
  const dow = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])).getUTCDay();
  if (dow === 0 || dow === 6) return false;
  const hm = +m[4] * 60 + +m[5];
  return hm >= 9 * 60 && hm < 15 * 60 + 30;
}

export type BotSummary = 'unknown' | 'running' | 'starting' | 'stopping' | 'stopped' | 'failed' | 'external';

export function botSummary(s: R016Status | null): BotSummary {
  if (!s) return 'unknown';
  const st = s.service?.active_state;
  if (st === 'active') return 'running';
  if (st === 'activating') return 'starting';
  if (st === 'deactivating') return 'stopping';
  if (s.external_r003?.length) return 'external';
  if (st === 'failed') return 'failed';
  return 'stopped';
}

/** Number of pending-restart items (changed files since the service start). */
export function pendingCount(s: R016Status | null): number {
  if (!s || !s.pending_restart) return 0;
  return Object.values(s.pending || {}).filter((p) => p?.changed_since_start).length;
}

// ---- value editor ----------------------------------------------------------------------------

/** draft value for one key: boolean for bool keys, raw input text for everything else */
export type Draft = boolean | string;

/** The value currently saved for a key: the override when there is one, else the base. */
export function savedValue(k: Pick<SchemaKey, 'base' | 'override' | 'has_override'>): unknown {
  return k.has_override ? k.override : k.base;
}

export function valueToText(type: ValueType, v: unknown): string {
  if (v === null || v === undefined) return type === 'str' ? '' : 'null';
  if (type === 'str' && typeof v === 'string') return v;
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export function initialDraft(k: Pick<SchemaKey, 'type' | 'base' | 'override' | 'has_override'>): Draft {
  const v = savedValue(k);
  if (k.type === 'bool') return v === true;
  return valueToText(k.type, v);
}

const STR_FORBIDDEN = /['"`;\\\r\n]/;
const JSON_FORBIDDEN = /['`;\r\n]/;

export type ParseOut = { ok: true; value: unknown } | { ok: false; error: string };

/**
 * Input -> JSON value to send. The app blocks only what is clearly not a value of the type (int with a fraction, text that is not
 * JSON, a dict that is not an object) or what it cannot quote safely; everything else goes to the server, which has the final say
 * (e.g. `true` in an int field is sent and comes back as a `rejected` message).
 */
export function parseDraft(type: ValueType, d: Draft): ParseOut {
  if (type === 'bool') {
    if (typeof d === 'boolean') return { ok: true, value: d };
    if (d === 'true' || d === 'false') return { ok: true, value: d === 'true' };
    return { ok: false, error: 'true / false 만 가능합니다' };
  }
  const text = String(d).trim();
  if (type === 'str') {
    if (STR_FORBIDDEN.test(String(d))) return { ok: false, error: '따옴표 · 백틱 · 세미콜론 · 역슬래시 · 줄바꿈은 쓸 수 없습니다' };
    return { ok: true, value: String(d) };
  }
  if (!text) return { ok: false, error: '값을 입력하세요' };
  if (JSON_FORBIDDEN.test(text)) return { ok: false, error: "작은따옴표(') · 백틱 · 세미콜론 · 줄바꿈은 쓸 수 없습니다" };
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    if (type === 'int' || type === 'float') return { ok: false, error: '숫자가 아닙니다' };
    return { ok: false, error: 'JSON 형식이 올바르지 않습니다' };
  }
  if (type === 'int' && typeof v === 'number' && (!Number.isInteger(v) || /[.eE]/.test(text))) return { ok: false, error: '정수만 입력할 수 있습니다' };
  if (type === 'list' && !Array.isArray(v)) return { ok: false, error: 'JSON 배열 [ ... ] 이어야 합니다' };
  if (type === 'dict' && (typeof v !== 'object' || v === null || Array.isArray(v))) return { ok: false, error: 'JSON 객체 { ... } 이어야 합니다' };
  return { ok: true, value: v };
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

export interface Collected {
  changes: Record<string, unknown>;
  errors: Record<string, string>;
}

/** Keys whose draft differs from the saved value (only those go into `set`), plus input errors that block saving. */
export function collectChanges(items: Pick<SchemaKey, 'key' | 'type' | 'base' | 'override' | 'has_override' | 'editable'>[], drafts: Record<string, Draft>): Collected {
  const changes: Record<string, unknown> = {};
  const errors: Record<string, string> = {};
  for (const it of items) {
    if (!(it.key in drafts) || it.editable === false) continue;
    const p = parseDraft(it.type, drafts[it.key]);
    if (!p.ok) {
      errors[it.key] = p.error;
      continue;
    }
    if (!same(p.value, savedValue(it))) changes[it.key] = p.value;
  }
  return { changes, errors };
}

/** ["KEY=<json>", ...] for `r016_control.py set` */
export function setArgs(changes: Record<string, unknown>): string[] {
  return Object.entries(changes).map(([k, v]) => `${k}=${JSON.stringify(v)}`);
}

/** "= 0.40%" next to *_PCT values (the code stores ratios: 0.004 = 0.4%) */
export function pctHint(key: string, v: unknown): string {
  if (!/_PCT$/.test(key) || typeof v !== 'number' || !Number.isFinite(v)) return '';
  return `= ${(v * 100).toFixed(2)}%`;
}

export function fmtValue(v: unknown): string {
  if (v === undefined) return '-';
  if (v === null) return 'null';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** "set ENABLE_X true->false, Y 1->2" style one-liner of a history row */
export function historyText(r: HistoryRow): string {
  const ch = r.changes || {};
  const parts = Object.entries(ch).map(([k, v]) => {
    if (v && typeof v === 'object' && !Array.isArray(v) && ('from' in v || 'to' in v)) {
      const o = v as { from?: unknown; to?: unknown };
      return `${k} ${fmtValue(o.from)}->${fmtValue(o.to)}`;
    }
    if (k === 'codes' && Array.isArray(v)) return `${v.length}종목`;
    return `${k}=${fmtValue(v)}`;
  });
  return parts.join(', ');
}

// ---- watchlist -------------------------------------------------------------------------------

export const WATCH_MAX = 100;

/** Per-row input errors (index -> message) and list-level errors. */
export function checkWatchRows(rows: WatchEntry[]): { rows: Record<number, string>; list?: string } {
  const out: Record<number, string> = {};
  const seen = new Map<string, number>();
  rows.forEach((r, i) => {
    const code = r.code.trim();
    if (!/^\d{6}$/.test(code)) out[i] = '코드는 숫자 6자리';
    else if (/[,;\r\n'"`\\]/.test(r.name)) out[i] = '이름에 쉼표 · 세미콜론 · 따옴표 · 줄바꿈은 쓸 수 없습니다';
    else if (seen.has(code)) out[i] = `중복 (${seen.get(code)! + 1}행과 같은 코드)`;
    if (/^\d{6}$/.test(code) && !seen.has(code)) seen.set(code, i);
  });
  const list = rows.length > WATCH_MAX ? `최대 ${WATCH_MAX}종목까지 저장할 수 있습니다 (현재 ${rows.length})` : undefined;
  return { rows: out, list };
}

/** stdin body for `watchlist set --stdin`: one "code,name" (or "code") per line */
export function watchlistStdin(rows: WatchEntry[]): string {
  return rows.map((r) => (r.name.trim() ? `${r.code.trim()},${r.name.trim()}` : r.code.trim())).join('\n') + '\n';
}
