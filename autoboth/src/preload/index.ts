/*
 * Update log
 * - [2026-11-04] type=feat - v2 bridge: window.api (settings, connection, day selection, conditions, snapshots, timelines) and
 *   window.updater for the shared update dialog. Read-only by design - no channel writes to the server
 * - [2026-11-05] type=feat - r003 remote control channels (ctl:*): state/status, service start/stop/restart, schema, set/unset/
 *   reset, history, watchlist get/set. They only reach the main process, which runs r016_control.py (confirmations are shown
 *   by main, so the renderer cannot skip them)
 * - [2026-11-05] type=feat - getPrice (price:get) for the timeline price chart
 * - [2026-11-06] type=feat - getQuote (market:quote) for the chart tab's current price / order book
 * - [2026-11-06] type=feat - window.winCtl (title-bar window buttons) and window.appAbout (shared About dialog of the ☰ menu)
 */

import { contextBridge, ipcRenderer } from 'electron';
import type { AppStatus, ConditionsDoc, Settings, Snapshot } from '../shared/types';
import type { MarketQuote, PriceRequest, PriceResponse } from '../shared/price';
import type {
  ControlState,
  HistoryRow,
  RiskProfile,
  SchemaDoc,
  ServiceResult,
  SetResult,
  WatchEntry,
  WatchlistDoc,
  WatchlistSetResult,
} from '../shared/control';

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
  connect: () => ipcRenderer.invoke('conn:connect'),
  disconnect: () => ipcRenderer.invoke('conn:disconnect'),
  openDay: (ymd: string) => ipcRenderer.invoke('day:open', ymd),
  openLocal: (dir: string) => ipcRenderer.invoke('day:openLocal', dir),
  listDates: () => ipcRenderer.invoke('day:listDates'),
  getConditions: () => ipcRenderer.invoke('conditions:get'),
  refreshConditions: () => ipcRenderer.invoke('conditions:refresh'),
  getSnapshot: () => ipcRenderer.invoke('snapshot:get'),
  getStatus: () => ipcRenderer.invoke('status:get'),
  getTimeline: (code: string) => ipcRenderer.invoke('timeline:get', code),
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
