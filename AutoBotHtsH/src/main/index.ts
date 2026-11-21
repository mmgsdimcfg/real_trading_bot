// Update log
// - [2026-11-06] type=feat (HtsH main process: frameless window (TermH-style title bar in the renderer), login with a KIS app key / app
//   secret (real or paper server) or demo mode, the market backend (KisMarket / DemoMarket) behind one IPC dispatcher with a fixed list of
//   inquiry methods, stock master search, realtime trades / order book forwarded in 100 ms batches, shared updater + About dialog.
//   Inquiry-only by construction: no account number is asked for, the KIS client allows only quotation/ranking TRs, and there is no order
//   code anywhere in the app)
// - [2026-11-06] type=fix (Tick charts showed no NXT trades: the renderer always subscribes H0STCNT0 / H0STASP0 (KRX only). For NXT-tradable
//   symbols on the real server the main process now subscribes the combined KRX + NXT TRs H0UNCNT0 / H0UNASP0 instead (route decided once
//   per tr+code and reused by unsubscribe) and hands the records back under the KRX TR names, so every window gets NXT trades unchanged)
// - [2026-11-06] type=feat ('ticks' market method (tick chart back-fill), its code checked like the other per-symbol calls)
// - [2026-11-12] type=feat (The KIS backend gets the stock master's NXT flag (no API call) for watchlist prices and NXT routing)
// - [2026-11-19] type=feat (watch:* IPC - watchlist groups in %APPDATA%\htsh\watchlist\watchlist.json, picks files of that folder, open
//   the folder in Explorer; chart cache folder for the KIS backend (chartCache.ts))
// - [2026-11-20] type=feat (AutoBotHtsH: the HtsH main process plus the AutoBotH main process (src/bot/main/botMain.ts - SSH / local
//   r016 control, log sessions, price service, scheduler; IPC channels and window.api unchanged). One window: attachBotWindow() after it
//   is created, botQuit() when it closes. importOnce.ts runs first so AutoBotH / HtsH settings are copied before any store opens)
// - [2026-11-21] type=feat (Rebound watcher (reboundPoller.ts) started with the window: watches the watchlist tab "TEST" on 1-minute bars
//   through the logged-in KIS backend (not in demo mode) and shows always-on-top popups (reboundAlert.ts). rebound:* IPC - status,
//   on / off, test popup, open the config / log folder; a popup's "열기" sends rebound:open (the renderer shows the symbol).
//   Popups close with the main window so the app still quits)

import './importOnce';
import { app, BrowserWindow, ipcMain, Menu, shell } from 'electron';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { KisClient, KisError } from './kisClient';
import { KisMarket } from './kisMarket';
import { DemoMarket, DEMO_MASTER } from './demoMarket';
import { Master } from './master';
import { DemoFeed, KisFeed, type Feed, type RtData, type RtStatus } from './realtime';
import { ChartDiskCache } from './chartCache';
import { loadWatchFile, saveWatchFile, scanPicks, watchDir, type WatchFile } from './watchlist';
import { forgetCreds, loginDefaults, rememberCreds, savedCreds, tokenStore, maskKey, type Creds } from './settings';
import type { MarketBackend, Res } from '../shared/market';
import { isCode, toKrxRecord, WS_COMBINED, type KisEnv, type WsTr } from '../shared/kis';
import { KIS_SPEC_COMMIT } from '../shared/kisSpec.generated';
import { attachBotWindow, botQuit } from '../bot/main/botMain';
import { ReboundWatcher, sampleAlert } from './reboundPoller';
import { AlertPopups, reboundDir } from './reboundAlert';

const TASKBAR_TITLE_PAD = '⠀'.repeat(40);
app.on('browser-window-created', (_e, w) => {
  w.setTitle(String(w.getTitle() || '').replace(/⠀+$/, '') + TASKBAR_TITLE_PAD);
  w.on('page-title-updated', (e, title) => {
    e.preventDefault();
    w.setTitle(String(title || '').replace(/⠀+$/, '') + TASKBAR_TITLE_PAD);
  });
});

let win: BrowserWindow | null = null;
let backend: MarketBackend | null = null;
let client: KisClient | null = null;
let feed: Feed | null = null;
let rtStatus: { status: RtStatus; msg?: string } = { status: 'off' };
const master = new Master(DEMO_MASTER);
const logLines: string[] = [];

const log = (line: string) => {
  logLines.push(`${new Date().toISOString().slice(11, 19)} ${line}`);
  if (logLines.length > 300) logLines.shift();
};
const send = (ch: string, data: unknown) => {
  if (win && !win.isDestroyed()) win.webContents.send(ch, data);
};
const errMsg = (e: unknown) => String((e as Error)?.message || e).split('\n')[0];

function authState() {
  const d = loginDefaults();
  return {
    status: backend ? 'in' : 'out',
    mode: backend?.kind || null,
    env: client?.env || d.env,
    keyHint: client ? maskKey(client.creds.appKey) : d.keyHint,
    remember: d.remember,
    hasSaved: d.hasSaved,
    tokenExpiresAt: client?.tokenExpiresAt(),
    master: { size: master.size, date: master.date, error: master.error },
    rt: rtStatus,
    specCommit: KIS_SPEC_COMMIT,
  };
}

// ---- realtime batching ----
let rtBuf: RtData[] = [];
let rtTimer: NodeJS.Timeout | null = null;
function onRt(raw: RtData) {
  const d: RtData = { ...raw, ...toKrxRecord(raw.tr, raw.rec) };
  if (d.tr === 'H0STASP0') rtBuf = rtBuf.filter((x) => !(x.tr === 'H0STASP0' && x.code === d.code));
  rtBuf.push(d);
  if (!rtTimer)
    rtTimer = setTimeout(() => {
      rtTimer = null;
      const b = rtBuf;
      rtBuf = [];
      send('rt:data', b);
    }, 100);
}
function onRtStatus(status: RtStatus, msg?: string) {
  rtStatus = { status, msg };
  if (msg) log(`[rt] ${msg}`);
  send('rt:status', rtStatus);
}

function closeSession() {
  rtRoutes.clear();
  feed?.close();
  feed = null;
  backend = null;
  client = null;
  rtStatus = { status: 'off' };
}

async function login(req: { env: KisEnv; appKey?: string; appSecret?: string; remember: boolean; useSaved?: boolean }) {
  closeSession();
  let creds: Creds | null;
  if (req.useSaved) {
    creds = savedCreds();
    if (!creds) throw new Error('저장된 앱키가 없습니다.');
    creds = { ...creds, env: req.env };
  } else {
    const appKey = String(req.appKey || '').trim();
    const appSecret = String(req.appSecret || '').trim();
    if (appKey.length < 16 || appSecret.length < 16) throw new Error('앱키와 앱시크릿을 입력하세요.');
    creds = { env: req.env, appKey, appSecret };
  }
  const c = new KisClient(creds, tokenStore, log);
  await c.accessToken(); // validates the key (or reuses the cached token)
  client = c;
  backend = new KisMarket(c, (code) => (master.nxtKnown ? !!master.get(code)?.nxt : undefined), chartDisk());
  rememberCreds(creds, req.remember);
  feed = new KisFeed(creds.env, () => c.approvalKey(), onRt, onRtStatus);
  log(`[auth] 로그인 (${creds.env === 'real' ? '실전' : '모의'} 서버, 조회 전용)`);
  void master.ensure().then(() => send('auth:state', authState()));
  return authState();
}

/** market methods the renderer may call - every one of them is an inquiry */
const METHODS = ['quote', 'orderbook', 'trades', 'ticks', 'chart', 'daily', 'investor', 'investorDaily', 'info', 'multi', 'rank', 'index', 'news'] as const;
type Method = (typeof METHODS)[number];

ipcMain.handle('auth:state', () => authState());
ipcMain.handle('auth:login', async (_e, req) => {
  try {
    return { ok: true, data: await login(req) };
  } catch (e) {
    closeSession();
    const k = e as KisError;
    const hint =
      k.code === 'EGW00133' ? ' - 토큰은 1분에 한 번만 발급됩니다. 잠시 후 다시 시도하세요.' : k.code === 'EGW00103' || k.code === 'EGW00105' ? ' - 앱키/앱시크릿 또는 실전·모의 구분을 확인하세요.' : '';
    return { ok: false, error: errMsg(e) + hint, code: k.code };
  }
});
ipcMain.handle('auth:demo', () => {
  closeSession();
  const d = new DemoMarket();
  backend = d;
  feed = new DemoFeed(d, onRt, onRtStatus);
  log('[auth] 데모 모드');
  return authState();
});
ipcMain.handle('auth:logout', (_e, forget: boolean) => {
  closeSession();
  if (forget) forgetCreds();
  return authState();
});

ipcMain.handle('mkt:call', async (_e, method: Method, ...args: unknown[]): Promise<Res<unknown>> => {
  if (!backend) return { ok: false, error: '로그인하세요.' };
  if (!METHODS.includes(method)) return { ok: false, error: `허용되지 않은 호출: ${method}` };
  const first = args[0];
  if ((method === 'quote' || method === 'orderbook' || method === 'trades' || method === 'daily' || method === 'investor' || method === 'investorDaily' || method === 'info' || method === 'news') && !isCode(String(first))) {
    return { ok: false, error: '종목 코드 형식 오류' };
  }
  if (method === 'ticks' && !isCode(String((first as { code?: unknown })?.code))) return { ok: false, error: '종목 코드 형식 오류' };
  try {
    // screen tests: HTSH_DEMO_DELAY=<ms> slows the demo backend down to network-like latency
    const delay = backend.kind === 'demo' ? Number(process.env.HTSH_DEMO_DELAY) || 0 : 0;
    if (delay) await new Promise((r) => setTimeout(r, delay));
    const fn = backend[method] as (...a: unknown[]) => Promise<unknown>;
    return { ok: true, data: await fn.apply(backend, args) };
  } catch (e) {
    return { ok: false, error: errMsg(e), code: (e as KisError).code };
  }
});

ipcMain.handle('master:search', async (_e, q: string) => master.search(String(q || ''), 30));
ipcMain.handle('master:get', (_e, code: string) => master.get(String(code)) || DEMO_MASTER.find((m) => m.code === code) || null);
ipcMain.handle('master:refresh', async () => {
  await master.ensure(true);
  return authState().master;
});

// renderer tr|code -> TR actually subscribed (combined KRX + NXT for NXT symbols on the real server); kept for the session so
// subscribe / unsubscribe of the same key always hit the same TR and run in order
const rtRoutes = new Map<string, Promise<string>>();
function rtRoute(tr: string, code: string): Promise<string> {
  const k = `${tr}|${code}`;
  let p = rtRoutes.get(k);
  if (!p) {
    const un = WS_COMBINED[tr as WsTr];
    p = un && backend instanceof KisMarket && client?.env === 'real' ? backend.isNxt(code).then((nxt) => (nxt ? un : tr)) : Promise.resolve(tr);
    rtRoutes.set(k, p);
  }
  return p;
}
ipcMain.on('rt:sub', (_e, tr: string, code: string) => {
  if (!isCode(code)) return;
  const f = feed;
  void rtRoute(tr, code).then((t) => {
    try {
      f?.subscribe(t, code);
    } catch (e) {
      log(`[rt] ${errMsg(e)}`);
    }
  });
});
ipcMain.on('rt:unsub', (_e, tr: string, code: string) => {
  const f = feed;
  void rtRoute(tr, code).then((t) => f?.unsubscribe(t, code));
});
ipcMain.handle('app:log', () => logLines.slice(-200));

// ---- [2026-11-19] chart data on disk (%APPDATA%\htsh\chartcache), written in the background and on quit ----
let chartDiskInst: ChartDiskCache | null = null;
function chartDisk(): ChartDiskCache {
  return (chartDiskInst ??= new ChartDiskCache());
}
app.on('before-quit', () => chartDiskInst?.flush());

// ---- watchlist file + picks files (%APPDATA%\htsh\watchlist) ----
ipcMain.handle('watch:load', () => loadWatchFile());
ipcMain.handle('watch:save', (_e, w: WatchFile) => {
  try {
    if (w && Array.isArray(w.groups)) saveWatchFile({ groups: w.groups, imported: w.imported || {} });
    return true;
  } catch (e) {
    log(`watchlist save failed: ${(e as Error).message}`);
    return false;
  }
});
ipcMain.handle('watch:picks', () => {
  try {
    return scanPicks();
  } catch {
    return [];
  }
});
ipcMain.handle('watch:dir', () => watchDir());
ipcMain.handle('watch:openDir', () => {
  scanPicks(); // creates the folder when it is not there yet
  return shell.openPath(watchDir());
});

// ---- [2026-11-21] rebound watcher: watchlist tab "TEST", detection + popup only ----
const popups = new AlertPopups((code) => {
  send('rebound:open', code);
  if (win && !win.isDestroyed()) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});
const rebound = new ReboundWatcher({
  // demo prices are made up: no watching there (HTSH_REBOUND_DEMO=1 for screen tests)
  backend: () => (backend && (backend.kind === 'kis' || process.env.HTSH_REBOUND_DEMO === '1') ? backend : null),
  master,
  log,
  alert: (a, cfg) => popups.show(a, cfg),
  status: (s) => send('rebound:status', s),
});
ipcMain.handle('rebound:status', () => rebound.status);
ipcMain.handle('rebound:enable', (_e, on: boolean) => {
  rebound.setEnabled(!!on);
  return rebound.status;
});
ipcMain.handle('rebound:test', () => popups.show(sampleAlert(rebound.config), rebound.config));
ipcMain.handle('rebound:openDir', () => {
  rebound.loadConfig(); // creates the folder and the config file when they are not there yet
  return shell.openPath(reboundDir());
});

// ---- window ----
function versionLabel(): string {
  try {
    const line = readFileSync(join(app.getAppPath(), 'version.txt'), 'utf8')
      .split(/\r?\n/)
      .map((s) => s.trim())
      .find((s) => s && !s.startsWith('#'));
    const m = /^v?(\d+\.\d+\.\d+_\d{3})$/.exec(line || '');
    if (m) return `v${m[1]}`;
  } catch {
    /* fall back */
  }
  return `v${app.getVersion()}`;
}
ipcMain.on('get-app-version', (e) => (e.returnValue = versionLabel()));

function createWindow(): void {
  win = new BrowserWindow({
    width: 1600,
    height: 980,
    minWidth: 1024,
    minHeight: 640,
    backgroundColor: '#101114',
    title: 'AutoBotHtsH',
    frame: false,
    icon: join(app.getAppPath(), 'resources', 'icon.png'),
    webPreferences: { preload: join(__dirname, '../preload/index.js'), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  const w = win;
  w.on('maximize', () => w.webContents.send('win-maximized-changed', true));
  w.on('unmaximize', () => w.webContents.send('win-maximized-changed', false));
  w.on('closed', () => {
    rebound.stop();
    popups.closeAll(); // popups are windows too: left open they would keep the app from quitting
  });
  w.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) void shell.openExternal(url);
    return { action: 'deny' };
  });
  if (process.env.ELECTRON_RENDERER_URL) void w.loadURL(process.env.ELECTRON_RENDERER_URL);
  else void w.loadFile(join(__dirname, '../renderer/index.html'));
  attachBotWindow(w);
}

const appRequire = createRequire(join(app.getAppPath(), 'package.json'));
const { initUpdater } = appRequire('./updater.js');
initUpdater({ productName: 'AutoBotHtsH', getMainWindow: () => win, getVersionLabel: versionLabel });
const { initAppAbout } = appRequire('./app-about.js');
initAppAbout({ productName: 'AutoBotHtsH' });

Menu.setApplicationMenu(null);
const senderWin = (e: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent) => BrowserWindow.fromWebContents(e.sender);
ipcMain.on('win-minimize', (e) => senderWin(e)?.minimize());
ipcMain.on('win-toggle-maximize', (e) => {
  const x = senderWin(e);
  if (x) x.isMaximized() ? x.unmaximize() : x.maximize();
});
ipcMain.on('win-close', (e) => senderWin(e)?.close());
ipcMain.handle('win-is-maximized', (e) => !!senderWin(e)?.isMaximized());

void app.whenReady().then(() => {
  createWindow();
  rebound.start();
  // HTSH_DEMO=1: start in demo mode (screen tests)
  if (process.env.HTSH_DEMO === '1') {
    const d = new DemoMarket();
    backend = d;
    feed = new DemoFeed(d, onRt, onRtStatus);
  }
});

app.on('window-all-closed', () => {
  closeSession();
  botQuit();
  app.quit();
});
