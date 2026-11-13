/*
 * Update log
 * - [2026-11-05] type=feat - Settings: SSH proxy fields proxyType/proxyCommand/proxyHost/proxyPort/proxyUser + proxySecret/hasProxySecret (same options as TermH)
 * - [2026-11-05] type=feat - Remote control: Condition.values_meta (editable/type/base/override/pending per value, from
 *   `r016_control.py conditions`) and ConditionsDoc.overrides / override_errors / overrides_file
 * - [2026-11-04] type=feat - Shared data model of the r003 condition pipeline viewer (main process, worker, renderer)
 * - [2026-11-10] type=feat - Local mode: RunMode, AppStatus.mode, Settings.runMode / localRoot / localPython / local poll intervals,
 *   scheduler switches (autoCollector, autoPostclose + postcloseTime / postcloseEdgeAi, syncWatchlistToPeer) and keepAwake
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

/** 'removed' [2026-11-13]: watched by an earlier r003 run of the day but not in the watchlist of the latest run (r004 was edited) */
export type WatchState = 'active' | 'backup' | 'graduated' | 'dropped' | 'removed' | 'unknown';

/** [2026-11-13] a holding that was not bought today: 'carried' = in the watchlist ([HOLD SKIP] NOT_TODAY_BUY_POSITION),
 *  'unmanaged' = not in the watchlist ("account holdings not in watchlist") */
export type HeldState = 'carried' | 'unmanaged' | 'manual';

export interface Position {
  qty: number;
  buyPrice: number;
  buyTs: number;
}

export interface SymbolSummary {
  code: string;
  name: string;
  /** [2026-11-12] also tradable on NXT - from the r003 WATCH line ("NXT=True"); undefined = not said (symbol not watched) */
  nxt?: boolean;
  watch: WatchState;
  position?: Position;
  /** [2026-11-13] held but not bought today (no fill in today's buy_sell log, so `position` is empty) */
  held?: HeldState;
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
  /** [from, to] stretches without main-log lines (r003 not running) - shaded on the replay bar, skipped while playing */
  gaps: [number, number][];
}

export type ConnState = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'error' | 'local';

/** where r003 runs and the logs are read: 'remote' = Raspberry Pi over SSH, 'local' = this Windows PC */
export type RunMode = 'remote' | 'local';

/** [2026-11-11] confirmation asked by the main process, shown by the renderer (components/ConfirmDialog) */
export interface ConfirmRow {
  k: string;
  v: string;
  /** 'live' = red (real orders), 'warn' = amber */
  tone?: 'live' | 'warn';
}
export interface ConfirmSpec {
  /** small pill left of the kicker, e.g. 'LIVE' */
  tag?: string;
  tagTone?: 'live' | 'warn';
  /** small line above the title, e.g. 'r003 실행' */
  kicker: string;
  /** the question, e.g. '실주문 모드로 실행할까요?' */
  title: string;
  message?: string;
  rows?: ConfirmRow[];
  okLabel: string;
  cancelLabel?: string;
}
export interface ConfirmRequest extends ConfirmSpec {
  id: number;
}

export interface AppStatus {
  conn: ConnState;
  /** run mode of the open connection ('connected' + 'local' = this PC's runtime folder, read directly) */
  mode?: RunMode;
  message?: string;
  date: string;
  live: boolean;
  lastReceivedAt?: number;
  serverDate?: string;
  conditionsSource?: ConditionsDoc['source'];
  nextRetryAt?: number;
  /** past day: background download of the day's price files (main/prices.ts PrefetchProgress) */
  prefetch?: { date: string; done: number; total: number; failed: number; error?: string };
  /** local mode: the PC is kept awake while r003 runs (powerSaveBlocker) */
  keepAwake?: boolean;
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

  // ---- [2026-11-10] local mode (r003 on this PC; the Raspberry Pi is the standby host) ----
  runMode: RunMode;
  /** folder that holds real_trading_bot/, runtime/, open-trading-api/ and venv/ (e.g. D:\make_utils\Trading) */
  localRoot: string;
  /** Python of the local venv ('' = <localRoot>\venv\Scripts\python.exe) */
  localPython: string;
  /** local log poll interval (s) - reading a local file is cheap */
  localPollSec: number;
  /** local r016 status interval (s) */
  localStatusPollSec: number;
  /** start the g011 tick collector on weekday mornings (r016 `collector start`, every 5 min in the window) */
  autoCollector: boolean;
  /** run the post-close job (collect -> history -> scan) on trading days at postcloseTime */
  autoPostclose: boolean;
  postcloseTime: string;
  /** add the Edge AI steps to the post-close job */
  postcloseEdgeAi: boolean;
  /** [2026-11-11] the scheduled post-close job mails the picks (r021 --email -> g002 --notify-email) */
  postcloseEmail: boolean;
  /** after a successful post-close job, copy the new watchlist to the Raspberry Pi (standby host) */
  syncWatchlistToPeer: boolean;
  /** keep the PC from sleeping while the local r003 runs (Electron powerSaveBlocker) */
  keepAwake: boolean;
}
