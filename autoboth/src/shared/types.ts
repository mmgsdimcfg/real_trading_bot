/*
 * Update log
 * - [2026-11-05] type=feat - Settings: SSH proxy fields proxyType/proxyCommand/proxyHost/proxyPort/proxyUser + proxySecret/hasProxySecret (same options as TermH)
 * - [2026-11-05] type=feat - Remote control: Condition.values_meta (editable/type/base/override/pending per value, from
 *   `r016_control.py conditions`) and ConditionsDoc.overrides / override_errors / overrides_file
 * - [2026-11-04] type=feat - Shared data model of the r003 condition pipeline viewer (main process, worker, renderer)
 */

export type Stage = 'STATE' | 'TRIGGER' | 'CONTEXT' | 'PRE_ORDER' | 'GUARD' | 'EXIT' | 'ORDER' | string;

export interface Condition {
  order: number;
  no: number;
  func: string;
  stage: Stage;
  title: string;
  description: string;
  defines?: string[];
  values?: Record<string, unknown>;
  /** r016 only: per-value edit metadata (absent in r015 / bundled copies -> read-only display) */
  values_meta?: Record<string, ValueMeta>;
}

export interface ValueMeta {
  editable: boolean;
  type: 'bool' | 'int' | 'float' | 'str' | 'list' | 'dict' | 'null';
  base: unknown;
  override: unknown;
  has_override: boolean;
  pending: boolean;
}

export interface ConditionsDoc {
  generated_at: string;
  entry_trigger_mode?: string;
  trading_strategy_mode?: string;
  buy: Condition[];
  sell: Condition[];
  /** r016 only: every stored override, entries r001 refused, and the override file path */
  overrides?: Record<string, unknown>;
  override_errors?: string[];
  overrides_file?: string;
  /** where this copy came from: 'server' (just fetched), 'cache' (last good fetch), 'bundled' (app snapshot) */
  source?: 'server' | 'cache' | 'bundled';
}

export type Side = 'buy' | 'sell';
export type EvalResult = 'pass' | 'fail' | 'hold' | 'info';

/**
 * condKey: "_014" style key (number of the condition, matched to Condition.no), or one of
 * 'PREP' (frame not ready - before the conditions), 'ORDER' (order state), 'SESSION_CLOSE'
 * (forced close at session end - outside the sell conditions), 'UNKNOWN'.
 */
export interface EvalEvent {
  id: number;
  ts: number;
  code: string;
  side: Side;
  tag: string;
  condKey: string;
  sub?: string;
  result: EvalResult;
  reason: string;
  raw: string;
  steps?: Record<string, 'P' | 'F' | '-'>;
}

export interface Trade {
  code: string;
  name: string;
  side: 'BUY' | 'SELL';
  ts: number;
  price: number;
  qty: number;
  reason: string;
  condKey?: string;
  buyTs?: number;
  buyPrice?: number;
  pnlKrw?: number;
  pnlPct?: number;
  raw: string;
}

export type WatchState = 'active' | 'backup' | 'graduated' | 'dropped' | 'unknown';

export interface Position {
  qty: number;
  buyPrice: number;
  buyTs: number;
}

export interface SymbolSummary {
  code: string;
  name: string;
  watch: WatchState;
  position?: Position;
  lastEval?: EvalEvent;
  lastSell?: EvalEvent;
  lastOrder?: EvalEvent;
  /** highest buy condition order passed today (-1 = none, buyCount = all passed) */
  furthestOrderToday: number;
  /** buy condKey -> number of failed polls today (regular session only) */
  blockCounts: Record<string, number>;
  /** sell condKey -> number of SELL TRIGGER / SELL HOLD lines */
  sellCounts: Record<string, number>;
  realizedKrw: number;
  evalCount: number;
}

export interface NodeStat {
  /** failed polls in the regular session (from sessionOpen) */
  block: number;
  /** polls that got past this condition (regular session) */
  pass: number;
  /** waiting polls (e.g. _021 WAIT_NEXT_POLL_CONFIRM) */
  hold: number;
  /** failed polls before sessionOpen (pre-market tick-file waits etc.) - shown separately */
  preOpenBlock: number;
}

export interface SellNodeStat {
  trigger: number;
  hold: number;
}

export interface UnknownEntry {
  key: string;
  tag: string;
  count: number;
  sample: string;
}

export interface FileProgress {
  name: string;
  size: number;
  offset: number;
}

export interface Snapshot {
  date: string;
  generatedAt: number;
  symbols: SymbolSummary[];
  buyStats: Record<string, NodeStat>;
  sellStats: Record<string, SellNodeStat>;
  trades: Trade[];
  unknown: UnknownEntry[];
  warnings: number;
  regime?: string;
  wl?: { active: number; backup: number };
  lines: number;
  lastLineTs?: number;
  files: FileProgress[];
  /** past day loaded with time replay: the day's log time range and the replay cursor (null = whole day) */
  replay?: ReplayInfo;
}

export interface ReplayInfo {
  /** epoch ms of the first / last log line */
  first: number;
  last: number;
  /** epoch ms the shown state is cut at; null = the whole day */
  cursor: number | null;
  /** log times of the buy/sell blocks in the daily buy_sell log (whole day) - replay jump targets */
  trades: number[];
}

export type ConnState = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'error' | 'local';

export interface AppStatus {
  conn: ConnState;
  message?: string;
  date: string;
  live: boolean;
  lastReceivedAt?: number;
  serverDate?: string;
  conditionsSource?: ConditionsDoc['source'];
  nextRetryAt?: number;
}

export interface Settings {
  host: string;
  port: number;
  username: string;
  authMethod: 'key' | 'password';
  keyPath: string;
  /** set by the renderer only when changed; main stores it encrypted (safeStorage) */
  secret?: string;
  hasSecret?: boolean;
  /** SSH proxy (same options as TermH): '' = direct, 'command' = OpenSSH-style ProxyCommand, 'socks5' / 'http' = proxy server */
  proxyType: '' | 'command' | 'socks5' | 'http';
  /** ProxyCommand line - %h = host, %p = port, %r = username, %% = % */
  proxyCommand: string;
  proxyHost: string;
  /** 0 = default (SOCKS5 1080, HTTP 8080) */
  proxyPort: number;
  proxyUser: string;
  /** proxy password - set by the renderer only when changed; stored encrypted like `secret` */
  proxySecret?: string;
  hasProxySecret?: boolean;
  remoteRoot: string;
  pythonPath: string;
  pollSec: number;
  /** r016 status poll interval (s) - separate from pollSec because every call starts Python on the server */
  statusPollSec: number;
  /** manual funnel start time ("09:00") - used when sessionOpenAuto is off or the server NXT setting is unknown */
  sessionOpen: string;
  /** funnel start time follows the server ENABLE_NXT_SESSION (08:00 / 09:00) */
  sessionOpenAuto: boolean;
  retentionDays: number;
}
