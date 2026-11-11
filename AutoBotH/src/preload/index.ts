/*
 * Update log
 * - [2026-11-04] type=feat - v2 bridge: window.api (settings, connection, day selection, conditions, snapshots, timelines) and
 *   window.updater for the shared update dialog. Read-only by design - no channel writes to the server
 * - [2026-11-05] type=feat - r003 remote control channels (ctl:*): state/status, service start/stop/restart, schema, set/unset/
 *   reset, history, watchlist get/set. They only reach the main process, which runs r016_control.py (confirmations are shown
 *   by main, so the renderer cannot skip them)
 * - [2026-11-05] type=feat - getPrice (price:get) for the timeline price chart
 * - [2026-11-06] type=feat - getQuote (market:quote) for the chart tab's current price / order book
 * - [2026-11-07] type=feat - setReplayCursor (replay:cursor) for the past-day time replay bar
 * - [2026-11-06] type=feat - window.winCtl (title-bar window buttons) and window.appAbout (shared About dialog of the ☰ menu)
 * - [2026-11-10] type=feat - Local mode channels: ctl:role / ctl:collectorStart / ctl:jobs / ctl:jobStart / ctl:jobStop / ctl:jobLog /
 *   ctl:doctor, peer:* (the other host: status, role, stop, doctor, watchlist copy, KIS config import), settings:localCheck,
 *   folder/file pickers with a title, scheduler state ('sched')
 * - [2026-11-11] type=feat - ctlJobData (ctl:jobData) for the pipeline card of the data jobs tab (ctlPicksApply existed for a few
 *   hours and was dropped - r004 is written by the job's apply step)
 */

import { contextBridge, ipcRenderer } from 'electron';
import type { AppStatus, ConditionsDoc, ConfirmRequest, Settings, Snapshot } from '../shared/types';
import type { MarketQuote, PriceRequest, PriceResponse } from '../shared/price';
import type {
  ControlState,
  DoctorReport,
  HistoryRow,
  JobDataInfo,
  JobList,
  JobLog,
  JobStartOpts,
  R016Status,
  RiskProfile,
  RoleInfo,
  SchemaDoc,
  ServiceResult,
  SetResult,
  WatchEntry,
  WatchlistDoc,
  WatchlistSetResult,
} from '../shared/control';
import type { SchedState } from '../main/scheduler';

export interface CtlResult<T> {
  ok: boolean;
  error?: string;
  cancelled?: boolean;
  data?: T | null;
}

const on = <T>(ch: string, cb: (d: T) => void) => {
  const h = (_e: unknown, d: T) => cb(d);
  ipcRenderer.on(ch, h);
  return () => ipcRenderer.removeListener(ch, h);
};

const api = {
  appVersion: ipcRenderer.sendSync('get-app-version') as string,
  getSettings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (s: Settings) => ipcRenderer.invoke('settings:save', s),
  pickKey: () => ipcRenderer.invoke('dialog:pickKey'),
  pickFolder: () => ipcRenderer.invoke('dialog:pickFolder'),
  /** connect, then open `ymd` (past date) or the server's today */
  connect: (ymd?: string) => ipcRenderer.invoke('conn:connect', ymd),
  disconnect: () => ipcRenderer.invoke('conn:disconnect'),
  openDay: (ymd: string) => ipcRenderer.invoke('day:open', ymd),
  openLocal: (dir: string) => ipcRenderer.invoke('day:openLocal', dir),
  listDates: () => ipcRenderer.invoke('day:listDates'),
  getConditions: () => ipcRenderer.invoke('conditions:get'),
  refreshConditions: () => ipcRenderer.invoke('conditions:refresh'),
  getSnapshot: () => ipcRenderer.invoke('snapshot:get'),
  getStatus: () => ipcRenderer.invoke('status:get'),
  getTimeline: (code: string) => ipcRenderer.invoke('timeline:get', code),
  /** past-day replay: show the state at ts (epoch ms), null = whole day */
  setReplayCursor: (ts: number | null): Promise<{ ok: boolean; error?: string }> => ipcRenderer.invoke('replay:cursor', ts),
  getPrice: (req: PriceRequest): Promise<PriceResponse> => ipcRenderer.invoke('price:get', req),
  getQuote: (code: string): Promise<{ ok: boolean; data?: MarketQuote; error?: string; unsupported?: boolean }> => ipcRenderer.invoke('market:quote', code),
  onStatus: (cb: (s: AppStatus) => void) => on('status', cb),
  onSnapshot: (cb: (s: Snapshot | null) => void) => on('snapshot', cb),
  onConditions: (cb: (c: ConditionsDoc) => void) => on('conditions', cb),

  // r003 remote control (r016_control.py)
  ctlState: (): Promise<ControlState> => ipcRenderer.invoke('ctl:state'),
  ctlStatus: (): Promise<ControlState> => ipcRenderer.invoke('ctl:status'),
  ctlService: (verb: 'start' | 'stop' | 'restart', opts: { dryRun?: boolean; profile?: RiskProfile }): Promise<CtlResult<ServiceResult>> =>
    ipcRenderer.invoke('ctl:service', verb, opts),
  ctlSchema: (all: boolean): Promise<CtlResult<SchemaDoc>> => ipcRenderer.invoke('ctl:schema', all),
  ctlSet: (changes: Record<string, unknown>): Promise<CtlResult<SetResult>> => ipcRenderer.invoke('ctl:set', changes),
  ctlUnset: (keys: string[]): Promise<CtlResult<unknown>> => ipcRenderer.invoke('ctl:unset', keys),
  ctlReset: (): Promise<CtlResult<unknown>> => ipcRenderer.invoke('ctl:reset'),
  ctlHistory: (n: number): Promise<CtlResult<{ rows: HistoryRow[] }>> => ipcRenderer.invoke('ctl:history', n),
  watchlistGet: (): Promise<CtlResult<WatchlistDoc>> => ipcRenderer.invoke('ctl:watchlistGet'),
  watchlistSet: (rows: WatchEntry[]): Promise<CtlResult<WatchlistSetResult>> => ipcRenderer.invoke('ctl:watchlistSet', rows),
  onControl: (cb: (s: ControlState) => void) => on('control', cb),

  // [2026-11-10] local mode: host role, tick collector, data jobs, doctor (connected host)
  ctlRole: (role?: 'primary' | 'standby'): Promise<CtlResult<RoleInfo>> => ipcRenderer.invoke('ctl:role', role),
  ctlCollectorStart: (force?: boolean): Promise<CtlResult<unknown>> => ipcRenderer.invoke('ctl:collectorStart', force),
  /** [2026-11-11] stop g011 and start it again with the current watchlist (r016 collector restart) */
  ctlCollectorRestart: (): Promise<CtlResult<{ restarted?: boolean; pid?: number; window?: string }>> => ipcRenderer.invoke('ctl:collectorRestart'),
  ctlJobs: (): Promise<CtlResult<JobList>> => ipcRenderer.invoke('ctl:jobs'),
  ctlJobStart: (name: string, opts: JobStartOpts): Promise<CtlResult<unknown>> => ipcRenderer.invoke('ctl:jobStart', name, opts),
  ctlJobStop: (name: string): Promise<CtlResult<unknown>> => ipcRenderer.invoke('ctl:jobStop', name),
  ctlJobLog: (name: string, n?: number): Promise<CtlResult<JobLog>> => ipcRenderer.invoke('ctl:jobLog', name, n),
  /** [2026-11-11] the date folder's collected data + g002 picks (r016 job data) - pipeline step defaults / refusals */
  ctlJobData: (date: string): Promise<CtlResult<JobDataInfo>> => ipcRenderer.invoke('ctl:jobData', date),
  ctlDoctor: (): Promise<CtlResult<DoctorReport>> => ipcRenderer.invoke('ctl:doctor'),
  // the other host (local mode: Raspberry Pi, remote mode: this PC)
  peerInfo: (): Promise<{ kind: 'remote' | 'local'; label: string }> => ipcRenderer.invoke('peer:info'),
  peerStatus: (): Promise<CtlResult<R016Status> & { notConfigured?: boolean }> => ipcRenderer.invoke('peer:status'),
  peerRole: (role?: 'primary' | 'standby'): Promise<CtlResult<RoleInfo>> => ipcRenderer.invoke('peer:role', role),
  peerStop: (): Promise<CtlResult<unknown>> => ipcRenderer.invoke('peer:stop'),
  peerDoctor: (): Promise<CtlResult<DoctorReport>> => ipcRenderer.invoke('peer:doctor'),
  peerSyncWatchlist: (): Promise<CtlResult<WatchlistSetResult> & { count?: number }> => ipcRenderer.invoke('peer:syncWatchlist'),
  peerImportKis: (): Promise<{ ok: boolean; error?: string; cancelled?: boolean; path?: string }> => ipcRenderer.invoke('peer:importKis'),
  localCheck: (s?: Settings): Promise<{ problem: string; paths: { root: string; codeDir: string; r016: string; python: string; runtime: string } }> =>
    ipcRenderer.invoke('settings:localCheck', s),
  pickFolderTitled: (title: string): Promise<string | null> => ipcRenderer.invoke('dialog:pickFolder', title),
  pickFile: (title: string): Promise<string | null> => ipcRenderer.invoke('dialog:pickFile', title),
  schedState: (): Promise<SchedState> => ipcRenderer.invoke('sched:get'),
  onSched: (cb: (s: SchedState) => void) => on('sched', cb),
  // [2026-11-11] app-styled confirmations asked by the main process (control.ts askConfirm)
  onConfirm: (cb: (req: ConfirmRequest) => void) => on('confirm:ask', cb),
  answerConfirm: (id: number, ok: boolean) => ipcRenderer.send('confirm:answer', id, ok),
};

export type Api = typeof api;

contextBridge.exposeInMainWorld('api', api);

// custom title bar window buttons
contextBridge.exposeInMainWorld('winCtl', {
  minimize: () => ipcRenderer.send('win-minimize'),
  toggleMaximize: () => ipcRenderer.send('win-toggle-maximize'),
  close: () => ipcRenderer.send('win-close'),
  isMaximized: (): Promise<boolean> => ipcRenderer.invoke('win-is-maximized'),
  onMaximized: (cb: (m: boolean) => void) => on('win-maximized-changed', cb),
});

// shared About dialog (app-menu.js looks for window.appAbout in contextIsolation apps)
contextBridge.exposeInMainWorld('appAbout', {
  info: () => ipcRenderer.invoke('app-about:info'),
  openWebsite: () => ipcRenderer.invoke('app-about:open-website'),
  contact: (msg: unknown) => ipcRenderer.invoke('app-about:contact', msg),
  copyEmail: () => ipcRenderer.invoke('app-about:copy-email'),
});

contextBridge.exposeInMainWorld('updater', {
  check: () => ipcRenderer.invoke('update-check'),
  download: () => ipcRenderer.invoke('update-download'),
  install: () => ipcRenderer.send('update-install'),
  onAvailable: (cb: (info: unknown) => void) => ipcRenderer.on('update-available', (_e, info) => cb(info)),
  onProgress: (cb: (p: unknown) => void) => ipcRenderer.on('update-progress', (_e, p) => cb(p)),
});
