// Update log
// - [2026-11-06] type=feat (HtsH bridge: window.hts (auth, market inquiry calls, master search, realtime subscribe + data/status events),
//   window.winCtl (title-bar window buttons), window.appAbout (shared About dialog), window.updater (shared update dialog). The app secret
//   goes to main once at login and never comes back)
// - [2026-11-19] type=feat (watchLoad / watchSave / watchPicks / watchDir / watchOpenDir - watchlist file and picks files)
// - [2026-11-20] type=feat (AutoBotHtsH: window.api = the AutoBotH bridge (src/bot/preload) for the bot windows)
// - [2026-11-20] type=feat (hts.setZoom: page zoom factor for the 글자 크기 setting (webFrame, this window only))
// - [2026-11-21] type=feat (rebound*: status / on-off / test popup / folder of the rebound watcher, onReboundOpen (a popup's "열기"))

import { api as botApi } from '../bot/preload/index';
import { contextBridge, ipcRenderer, webFrame } from 'electron';
import type { MasterItem, KisEnv } from '../shared/kis';
import type { Res } from '../shared/market';
import type { PickGroup, PicksEntry } from '../shared/watchPicks';
import type { ReboundConfig } from '../shared/rebound';

type WatchFile = { groups: PickGroup[]; imported: Record<string, number> };

export interface ReboundStatus {
  enabled: boolean;
  tab: string;
  state: 'off' | 'rest' | 'login' | 'notab' | 'run';
  codes: number;
  lastPass?: number;
  judged?: number;
  failed?: number;
  alerts: number;
  note?: string;
}

const on = <T>(ch: string, cb: (d: T) => void) => {
  const h = (_e: unknown, d: T) => cb(d);
  ipcRenderer.on(ch, h);
  return () => ipcRenderer.removeListener(ch, h);
};

export interface AuthState {
  status: 'in' | 'out';
  mode: 'kis' | 'demo' | null;
  env: KisEnv;
  keyHint: string;
  remember: boolean;
  hasSaved: boolean;
  tokenExpiresAt?: number;
  master: { size: number; date: string; error: string };
  rt: { status: 'off' | 'connecting' | 'on' | 'error'; msg?: string };
  specCommit: string;
}

const hts = {
  /** [2026-11-20] 글자 크기: zoom factor 0.7 - 1.6 */
  setZoom: (f: number) => webFrame.setZoomFactor(Math.min(1.6, Math.max(0.7, Number(f) || 1))),
  version: ipcRenderer.sendSync('get-app-version') as string,
  authState: (): Promise<AuthState> => ipcRenderer.invoke('auth:state'),
  login: (req: { env: KisEnv; appKey?: string; appSecret?: string; remember: boolean; useSaved?: boolean }): Promise<Res<AuthState>> => ipcRenderer.invoke('auth:login', req),
  demo: (): Promise<AuthState> => ipcRenderer.invoke('auth:demo'),
  logout: (forget: boolean): Promise<AuthState> => ipcRenderer.invoke('auth:logout', forget),
  onAuth: (cb: (s: AuthState) => void) => on('auth:state', cb),
  call: <T>(method: string, ...args: unknown[]): Promise<Res<T>> => ipcRenderer.invoke('mkt:call', method, ...args),
  search: (q: string): Promise<MasterItem[]> => ipcRenderer.invoke('master:search', q),
  symbol: (code: string): Promise<MasterItem | null> => ipcRenderer.invoke('master:get', code),
  refreshMaster: () => ipcRenderer.invoke('master:refresh'),
  sub: (tr: string, code: string) => ipcRenderer.send('rt:sub', tr, code),
  unsub: (tr: string, code: string) => ipcRenderer.send('rt:unsub', tr, code),
  onRt: (cb: (d: { tr: string; code: string; rec: Record<string, string> }[]) => void) => on('rt:data', cb),
  onRtStatus: (cb: (s: AuthState['rt']) => void) => on('rt:status', cb),
  log: (): Promise<string[]> => ipcRenderer.invoke('app:log'),
  // [2026-11-19] watchlist file + picks files (%APPDATA%\htsh\watchlist)
  watchLoad: (): Promise<WatchFile | null> => ipcRenderer.invoke('watch:load'),
  watchSave: (w: WatchFile): Promise<boolean> => ipcRenderer.invoke('watch:save', w),
  watchPicks: (): Promise<PicksEntry[]> => ipcRenderer.invoke('watch:picks'),
  watchDir: (): Promise<string> => ipcRenderer.invoke('watch:dir'),
  watchOpenDir: (): Promise<string> => ipcRenderer.invoke('watch:openDir'),
  // [2026-11-21] rebound watcher (watchlist tab "TEST": detection + popup only)
  reboundStatus: (): Promise<ReboundStatus> => ipcRenderer.invoke('rebound:status'),
  reboundEnable: (on: boolean): Promise<ReboundStatus> => ipcRenderer.invoke('rebound:enable', on),
  reboundTest: (): Promise<void> => ipcRenderer.invoke('rebound:test'),
  // [2026-11-21] settings dialog: the whole config (the main process makes it safe and writes config.json); refresh = read the tab again
  reboundConfig: (): Promise<ReboundConfig> => ipcRenderer.invoke('rebound:config'),
  reboundSaveConfig: (cfg: ReboundConfig): Promise<{ config: ReboundConfig; status: ReboundStatus }> => ipcRenderer.invoke('rebound:saveConfig', cfg),
  reboundRefresh: (): Promise<void> => ipcRenderer.invoke('rebound:refresh'),
  reboundOpenDir: (): Promise<string> => ipcRenderer.invoke('rebound:openDir'),
  onReboundStatus: (cb: (s: ReboundStatus) => void) => on('rebound:status', cb),
  onReboundOpen: (cb: (code: string) => void) => on('rebound:open', cb),
};

export type Hts = typeof hts;

contextBridge.exposeInMainWorld('hts', hts);
// [2026-11-20] AutoBotHtsH: the AutoBotH API (bot control, log sessions, price service) for the bot windows
contextBridge.exposeInMainWorld('api', botApi);

contextBridge.exposeInMainWorld('winCtl', {
  minimize: () => ipcRenderer.send('win-minimize'),
  toggleMaximize: () => ipcRenderer.send('win-toggle-maximize'),
  close: () => ipcRenderer.send('win-close'),
  isMaximized: (): Promise<boolean> => ipcRenderer.invoke('win-is-maximized'),
  onMaximized: (cb: (m: boolean) => void) => on('win-maximized-changed', cb),
});

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
