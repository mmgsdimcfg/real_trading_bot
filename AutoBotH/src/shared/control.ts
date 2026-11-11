/*
 * Update log
 * - [2026-11-10] type=feat - Local mode: R016Status.host (platform / backend / host role) + collector, role / job / doctor
 *   response types, JOB_NAMES, scheduler helpers (postcloseDue, collectorWindow, localYmd) and peerDecision() - the cross-host
 *   guard before an r003 start (other host running -> refuse, primary -> switch it to standby, unknown -> ask)
 * - [2026-11-07] type=feat - NXT session: ControlState.nxt (r016 `get ENABLE_NXT_SESSION` - effective / running / pending),
 *   parseNxt() and resolveSessionOpen(): the app's funnel start time follows the server's NXT setting when sessionOpenAuto is on
 *   (NXT on -> 08:00 morning NXT session, off -> 09:00 regular session)
 * - [2026-11-09] type=refactor - Server code folder renamed real_trading_bot/autobot -> real_trading_bot/AutoBot (SERVER_CODE_SUBDIR);
 *   app builds before 2.2.1_003 point at the old folder and fail until rebuilt
 * - [2026-11-07] type=fix - Server code moved to the monorepo folder `<remote root>/real_trading_bot/autobot` (was
 *   `<remote root>/auto_trading_src`). The sub-folder is one constant (SERVER_CODE_SUBDIR) used by the r016 and r015 commands.
 *   r016 `status` gets its own poll interval (statusPollMs): each call starts a Python process on the Raspberry Pi, so it is
 *   not tied to the 10 s log poll - default 30 s, minimum 15 s, 120 s while the window is minimized/hidden.
 *   kstNowIso / quoteCacheMs / CHART_CACHE_MS: main-process reuse windows for r016 quote / bars answers (chart tab load rule)
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
  /** process backend (Windows r020 supervisor) */
  backend?: string;
  supervisor_pid?: number | null;
}

export interface ExternalProc {
  pid: number;
  cmdline: string;
  ppid?: number | null;
}

export interface PendingFile {
  path: string;
  mtime: string | null;
  changed_since_start: boolean;
}

/** r016 status.host / role get (2026-11-10) */
export interface HostRole {
  role: 'primary' | 'standby';
  /** false = no host_role.json (treated as primary) */
  set: boolean;
  by?: string | null;
  at?: string | null;
  note?: string | null;
}

export interface HostInfo {
  platform: string;
  hostname: string;
  machine?: string;
  /** systemd (Linux) | process (r020 supervisor, Windows) */
  backend: string;
  role: HostRole;
  runtime_root: string;
}

export type RoleInfo = HostRole & { ok: boolean; host?: HostInfo; error?: string };

export interface R016Status {
  generated_at: string;
  service: ServiceState;
  running: boolean;
  external_r003: ExternalProc[];
  /** older servers: missing */
  host?: HostInfo;
  collector?: {
    running: boolean;
    pid: number | null;
    started_at: string | null;
    symbols: number;
    last_write: string | null;
    /** codes g011 was started with (null = unknown) */
    subscribed?: string[] | null;
    /** watch codes g011 does not collect (started before the watchlist changed) */
    missing?: string[];
  };
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
  /** server ENABLE_NXT_SESSION (r016 get) - undefined until read */
  nxt?: NxtState;
}

/** r001 ENABLE_NXT_SESSION as r016 `get` reports it */
export interface NxtState {
  /** value r003 uses at its next start (base or override) */
  effective: boolean;
  /** value the running r003 uses (null = not running / unknown) */
  running: boolean | null;
  base: boolean;
  has_override: boolean;
  /** effective differs from running -> r003 restart needed */
  pending: boolean;
}

export const NXT_KEY = 'ENABLE_NXT_SESSION';
/** r001 MORNING_NXT_START / REGULAR_START */
export const NXT_OPEN = '08:00';
export const REGULAR_OPEN = '09:00';

/** `get ENABLE_NXT_SESSION` data -> NxtState (null when the key is missing or not a bool) */
export function parseNxt(data: unknown): NxtState | null {
  const v = (data as { values?: Record<string, Record<string, unknown>> } | null)?.values?.[NXT_KEY];
  if (!v || typeof v.effective !== 'boolean') return null;
  return {
    effective: v.effective,
    running: typeof v.running === 'boolean' ? v.running : null,
    base: typeof v.base === 'boolean' ? v.base : v.effective,
    has_override: !!v.has_override,
    pending: !!v.pending,
  };
}

/**
 * Funnel start time for the log engine: blocks before it are counted as "pre-open". Auto mode follows the NXT setting the
 * running r003 uses (or the stored one when it is not running); unknown -> the manual value.
 */
export function resolveSessionOpen(auto: boolean, manual: string, nxt: NxtState | undefined | null): string {
  if (auto && nxt) return (nxt.running ?? nxt.effective) ? NXT_OPEN : REGULAR_OPEN;
  return /^\d{1,2}:\d{2}$/.test(manual || '') ? manual : REGULAR_OPEN;
}

export type RiskProfile = 'conservative' | 'neutral' | 'aggressive';
export const RISK_PROFILES: RiskProfile[] = ['conservative', 'neutral', 'aggressive'];

// ---- command building / parsing --------------------------------------------------------------

export const R016_SCRIPT = 'live_trading/r016_control.py';
/** Server code folder under the remote root (monorepo `real_trading_bot`, server part `AutoBot/`). */
export const SERVER_CODE_SUBDIR = 'real_trading_bot/AutoBot';

export const STATUS_POLL_DEFAULT_SEC = 30;
export const STATUS_POLL_MIN_SEC = 15;
export const STATUS_POLL_HIDDEN_SEC = 120;

/** r016 `status` poll interval in ms: setting (default 30 s, at least 15 s); at least 120 s while the window is not visible. */
export function statusPollMs(statusPollSec: number | undefined, windowVisible: boolean): number {
  const n = Number(statusPollSec);
  const sec = Math.max(STATUS_POLL_MIN_SEC, Number.isFinite(n) && n > 0 ? n : STATUS_POLL_DEFAULT_SEC);
  return (windowVisible ? sec : Math.max(sec, STATUS_POLL_HIDDEN_SEC)) * 1000;
}

/** `<remote root>/real_trading_bot/AutoBot` (trailing slashes of the root removed). */
export function serverCodeDir(remoteRoot: string): string {
  return `${String(remoteRoot || '').replace(/\/+$/, '')}/${SERVER_CODE_SUBDIR}`;
}

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
 * `cd <root>/real_trading_bot/AutoBot && R016_BY=<by> <python> live_trading/r016_control.py <args...>`, every argument single-quoted.
 * Throws when an argument holds a quote, backtick, semicolon or newline (checked before quoting, defence in depth).
 */
export function buildR016Command(remoteRoot: string, pythonPath: string, by: string, args: string[]): string {
  for (const a of [by, ...args]) {
    const p = argProblem(a);
    if (p) throw new Error(p);
  }
  return `cd ${shellQuote(serverCodeDir(remoteRoot))} && R016_BY=${shellQuote(by)} ${shellPath(pythonPath)} ${R016_SCRIPT} ${args.map(shellQuote).join(' ')}`.trimEnd();
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
  const standby = s.host?.role?.role === 'standby';
  if (!avail) startReason = 'systemd 서비스가 없습니다 (autobot-r003.service 설치 필요)';
  else if (ext) startReason = '서비스 밖에서 r003이 실행 중입니다';
  else if (s.running) startReason = '이미 실행 중입니다';
  else if (standby) startReason = '이 호스트는 standby 입니다 - 실행하면 primary 로 바꾼 뒤 시작합니다';
  const processBackend = s.service?.backend === 'process';
  return {
    start: avail && !ext && !s.running,
    // Windows (process backend): stop also ends an r003 started outside the supervisor (there is no other manager there)
    stop: st === 'active' || st === 'activating' || (processBackend && ext),
    restart: st === 'active',
    startReason,
  };
}

/** Mon-Fri 09:00 <= t < 15:30 using the server's KST ISO string as is (no time-zone conversion). */
/** Current KST time as 'YYYY-MM-DDTHH:MM:SS' (independent of the PC's time zone). */
export function kstNowIso(nowMs: number = Date.now()): string {
  return new Date(nowMs + 9 * 3600 * 1000).toISOString().slice(0, 19);
}

/**
 * How long the main process reuses a server chart/quote answer before asking the server again (each ask starts Python on the
 * Raspberry Pi and may cost KIS calls). Market hours: just under the 10 s chart refresh; outside: values do not change.
 */
/** [2026-11-11] quoteLive 8 s -> 4 s: the chart tab now takes price / best quotes / trades from the tick file, so the KIS quote only
 *  refreshes the 10-level book - 3 KIS calls per 4 s for the viewed symbol stays inside the shared budget next to r003 (~1.6/s of 2/s) */
export const CHART_CACHE_MS = { quoteLive: 4000, quoteIdle: 300000, quoteFailed: 20000, daily: 600000, failed: 60000, intradayIdle: 300000 } as const;

export function quoteCacheMs(nowIso: string = kstNowIso()): number {
  return isMarketHours(nowIso) ? CHART_CACHE_MS.quoteLive : CHART_CACHE_MS.quoteIdle;
}

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

// ---- data jobs (r021 via r016 `job`) / doctor - 2026-11-10 --------------------------------------

export const JOB_NAMES = ['postclose', 'collect', 'history', 'scan', 'edge_ai', 'universe'];

export interface JobStep {
  name: string;
  argv: string[];
  state: 'running' | 'done' | 'failed' | 'stopped';
  rc: number | null;
  started_at: string;
  finished_at: string | null;
}

export interface JobRun {
  name: string;
  title: string;
  /** running | done | done_with_errors | failed | stopped */
  state: string;
  pid: number;
  by: string;
  run_id?: string | null;
  started_at: string;
  finished_at: string | null;
  rc: number | null;
  args: { date: string; workers: number | null; edge_ai: boolean; keep_watchlist?: boolean };
  log: string;
  steps: JobStep[];
  error: string | null;
}

export interface JobList {
  ok: boolean;
  running: { pid: number | null; started_at: string | null; name: string | null; run_id?: string | null } | null;
  jobs: { name: string; title: string; steps: string[]; last: JobRun | null }[];
  generated_at: string;
}

export interface JobLog {
  ok: boolean;
  name: string;
  log: string | null;
  lines: string[];
  state?: string;
}

export interface JobStartOpts {
  date?: string;
  workers?: number;
  edgeAi?: boolean;
  tradingDayOnly?: boolean;
  /** scan / postclose: build the picks files only, r004 watchlist untouched (past-date backfill, Edge AI) */
  keepWatchlist?: boolean;
  /** scheduler: no confirmation dialog */
  noConfirm?: boolean;
}

export interface DoctorReport {
  ok: boolean;
  problems: string[];
  generated_at: string;
  host: HostInfo;
  python: string;
  executable: string;
  packages: Record<string, string | null>;
  code: { dir: string; commit: string | null };
  open_trading_api: { path: string; exists: boolean; commit: string | null };
  kis_config: { dir: string; devlp_yaml: boolean; token_today: boolean };
  runtime: { root: string; exists: boolean };
}

export const JOB_STATE_LABEL: Record<string, string> = {
  running: '실행 중',
  done: '완료',
  done_with_errors: '완료 (일부 실패)',
  failed: '실패',
  stopped: '정지됨',
};

/** "HH:MM" -> minutes after midnight (NaN when malformed) */
export function hmMinutes(hm: string): number {
  const m = /^(\d{1,2}):(\d{2})$/.exec(String(hm || '').trim());
  return m ? +m[1] * 60 + +m[2] : NaN;
}

/** this PC's local date as YYYYMMDD (the notebook runs on KST like r003's datetime.now()) */
export function localYmd(d: Date = new Date()): string {
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
}

/**
 * Post-close scheduler: run once per day at/after `at` (local time) on Mon-Fri - the job itself skips closed days
 * (--trading-day-only). `lastYmd` = the day it last started.
 */
export function postcloseDue(now: Date, at: string, lastYmd: string | undefined): boolean {
  const dow = now.getDay();
  if (dow === 0 || dow === 6) return false;
  const t = hmMinutes(at);
  if (!Number.isFinite(t)) return false;
  return now.getHours() * 60 + now.getMinutes() >= t && lastYmd !== localYmd(now);
}

/** Tick collector morning window on this PC (Mon-Fri 08:40-15:35): r016 `collector start` checks the exact calendar itself. */
export function collectorWindow(now: Date): boolean {
  const dow = now.getDay();
  if (dow === 0 || dow === 6) return false;
  const m = now.getHours() * 60 + now.getMinutes();
  return m >= 8 * 60 + 40 && m < 15 * 60 + 35;
}

/**
 * Cross-host guard before starting r003 on `self` (decision only; the main process does the calls).
 *  - other host unknown (no SSH settings / not reachable): ask the user
 *  - other host runs r003: refuse
 *  - other host primary (or no role file): switch it to standby first
 */
export type PeerDecision = { action: 'refuse'; reason: string } | { action: 'ask'; reason: string } | { action: 'standby' } | { action: 'ok' };

export function peerDecision(peer: R016Status | null, peerError: string | undefined, peerLabel: string): PeerDecision {
  if (!peer) return { action: 'ask', reason: `${peerLabel} 상태를 확인하지 못했습니다${peerError ? ` (${peerError})` : ''}.` };
  if (peer.running || (peer.external_r003?.length || 0) > 0) {
    return { action: 'refuse', reason: `${peerLabel}에서 r003이 실행 중입니다. 같은 계좌를 두 곳에서 동시에 매매할 수 없으니 ${peerLabel}의 r003을 먼저 정지하세요.` };
  }
  if (peer.host && peer.host.role.role === 'standby') return { action: 'ok' };
  return { action: 'standby' };
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
