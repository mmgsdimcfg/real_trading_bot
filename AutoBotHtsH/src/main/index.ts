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
// - [2026-11-21] type=feat (rebound:config / rebound:saveConfig / rebound:refresh for the settings dialog (☰ > 반등 감시 설정))
// - [2026-11-22] type=feat (Rebound alerts are kept (reboundHistory.ts): every popup has a history item, the number of unread ones
//   goes to the renderer (rebound:unread) - a popup closed by hand / opened is read, one that closed by itself is not.
//   rebound:history* IPC (list, mark read, delete, show the popup again). KakaoTalk (reboundKakao.ts): with alert.kakao on, the
//   summary of each alert is sent to the user's own KakaoTalk; rebound:kakao* IPC (state, app keys, login, logout, test message))

// - [2026-11-23] type=feat (Guest mode (user 2026-11-23, guest.ts): auth:guest logs this PC in through another AutoBotHtsH (RemoteMarket,
//   no realtime feed - the windows poll); guest:host* lets a KIS-logged-in PC relay its market calls to guests (GuestHost, off by
//   default, port + password). Both the renderer and the guests go through callMarket() - the fixed list of inquiry methods)
import './importOnce';
import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from 'electron';
import { createRequire } from 'node:module';
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { KisClient, KisError } from './kisClient';
import { KisMarket } from './kisMarket';
import { DemoTrader, KisTrader, type Trader } from './kisOrder';
import { maskAccount, orderProblem, orderText, parseAccount, reviseProblem, type OrderReq, type OrderState, type ReviseReq } from '../shared/order';
import { DemoMarket, DEMO_MASTER } from './demoMarket';
import { Master } from './master';
import { DemoFeed, KisFeed, type Feed, type RtData, type RtStatus } from './realtime';
import { ChartDiskCache } from './chartCache';
import { loadWatchFile, saveWatchFile, scanPicks, watchDir, type WatchFile } from './watchlist';
import { forgetCreds, guestHostCfg, loginDefaults, rememberCreds, rememberGuest, saveGuestHostCfg, savedCreds, savedGuest, tokenStore, maskKey, saveAccount, savedAccount, type Creds } from './settings';
import { GUEST_PASS_MIN, GUEST_PORT_DEFAULT, GuestHost, lanAddresses, normalizeHostUrl, RemoteMarket } from './guest';
import type { MarketBackend, Res } from '../shared/market';
import { isCode, toKrxRecord, WS_COMBINED, type KisEnv, type WsTr } from '../shared/kis';
import { KIS_SPEC_COMMIT } from '../shared/kisSpec.generated';
import { attachBotWindow, botQuit } from '../bot/main/botMain';
import { ReboundWatcher, sampleAlert } from './reboundPoller';
import { AlertPopups, reboundDir, type AlertInfo } from './reboundAlert';
import { ReboundHistory } from './reboundHistory';
import { KakaoTalk, type KakaoState } from './reboundKakao';
import { alertSummary } from '../shared/rebound';

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
/** [2026-11-23] server (real / paper) of the host this PC is a guest of */
let guestEnv: KisEnv | null = null;
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
    env: client?.env || guestEnv || d.env,
    keyHint: client ? maskKey(client.creds.appKey) : d.keyHint,
    remember: d.remember,
    hasSaved: d.hasSaved,
    tokenExpiresAt: client?.tokenExpiresAt(),
    master: { size: master.size, date: master.date, error: master.error },
    rt: rtStatus,
    specCommit: KIS_SPEC_COMMIT,
    guest: { url: backend instanceof RemoteMarket ? backend.base : savedGuest()?.url || '', hasSaved: !!savedGuest() },
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
  demoTrader = null;
  rtRoutes.clear();
  feed?.close();
  feed = null;
  backend = null;
  client = null;
  guestEnv = null;
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
  log(`[auth] 로그인 (${creds.env === 'real' ? '실전' : '모의'} 서버)`);
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

// ---- [2026-11-26] order window (user 2026-11-26): the only code that sends orders. It is not part of callMarket, so a guest of this
// PC can never reach it; every order / revise / cancel is confirmed in a native dialog here in the main process and written to
// %APPDATA%\autobothtsh\orders\orders_YYYYMMDD.log ----
let demoTrader: DemoTrader | null = null;
function trader(): Trader | null {
  if (backend?.kind === 'demo') {
    const b = backend;
    return (demoTrader ??= new DemoTrader(
      (code) => b.quote(code).then((q) => q.price),
      (code) => master.get(code)?.name || DEMO_MASTER.find((m) => m.code === code)?.name || code,
    ));
  }
  if (backend?.kind === 'kis' && client) {
    const a = savedAccount(client.env);
    return a ? new KisTrader(client, a) : null;
  }
  return null;
}
function orderState(): OrderState {
  const mode = backend?.kind || null;
  const env = client?.env || guestEnv || loginDefaults().env;
  if (mode === 'demo') return { ready: true, mode, env, account: '데모 계좌', reason: '' };
  if (mode === 'kis' && client) {
    const a = savedAccount(client.env);
    return { ready: !!a, mode, env, account: a ? maskAccount(a) : '', reason: a ? '' : '계좌번호를 입력하면 주문할 수 있습니다.' };
  }
  return { ready: false, mode, env, account: '', reason: mode === 'guest' ? '게스트 모드에서는 주문할 수 없습니다.' : 'KIS 로그인 후 주문할 수 있습니다.' };
}
function orderLog(line: string): void {
  log(`[order] ${line}`);
  try {
    const now = new Date(Date.now() + 9 * 3600000).toISOString();
    const dir = join(app.getPath('userData'), 'orders');
    mkdirSync(dir, { recursive: true });
    appendFileSync(join(dir, `orders_${now.slice(0, 10).replace(/-/g, '')}.log`), `${now.slice(0, 19).replace('T', ' ')} ${line}\n`, 'utf8');
  } catch {
    /* the log file is a convenience - an order is never blocked by it */
  }
}
/** native confirmation of one order; the demo account of a screen test may skip it (HTSH_ORDER_NOCONFIRM=1) */
async function confirmOrder(title: string, text: string): Promise<boolean> {
  if (backend?.kind === 'demo' && process.env.HTSH_ORDER_NOCONFIRM === '1') return true;
  if (!win || win.isDestroyed()) return false;
  const st = orderState();
  const where = st.mode === 'demo' ? '데모 계좌 (실제 주문 아님)' : `${st.env === 'real' ? '실전' : '모의'} 계좌 ${st.account}`;
  const r = await dialog.showMessageBox(win, {
    type: st.mode === 'kis' && st.env === 'real' ? 'warning' : 'question',
    title,
    message: text,
    detail: `${where}\n\n주문을 전송할까요?`,
    buttons: ['주문 전송', '취소'],
    defaultId: 1,
    cancelId: 1,
    noLink: true,
  });
  return r.response === 0;
}
const ordRes = async <T>(f: () => Promise<T>): Promise<Res<T>> => {
  try {
    return { ok: true, data: await f() };
  } catch (e) {
    return { ok: false, error: errMsg(e), code: (e as KisError).code };
  }
};
const noTrader = '주문할 수 없는 상태입니다 (로그인 / 계좌번호 확인).';
const nameOf = (code: string) => master.get(code)?.name || DEMO_MASTER.find((m) => m.code === code)?.name;

ipcMain.handle('ord:state', () => orderState());
ipcMain.handle('ord:setAccount', (_e, text: string | null) =>
  ordRes(async () => {
    if (backend?.kind !== 'kis' || !client) throw new Error('KIS 로그인 후 계좌번호를 저장할 수 있습니다.');
    if (text === null) saveAccount(client.env, null);
    else {
      const a = parseAccount(String(text));
      if (!a) throw new Error('계좌번호는 숫자 10자리(종합계좌번호 8자리 + 상품코드 2자리)로 입력하세요. 예: 12345678-01');
      saveAccount(client.env, a);
      orderLog(`계좌 저장 ${maskAccount(a)} (${client.env === 'real' ? '실전' : '모의'})`);
    }
    return orderState();
  }),
);
ipcMain.handle('ord:buyable', (_e, code: string, type: string, price: number) =>
  ordRes(async () => {
    const t = trader();
    if (!t) throw new Error(noTrader);
    if (!isCode(String(code))) throw new Error('종목 코드 형식 오류');
    return t.buyable(String(code), type === 'market' ? 'market' : 'limit', Math.max(0, Math.round(Number(price)) || 0));
  }),
);
ipcMain.handle('ord:balance', () =>
  ordRes(async () => {
    const t = trader();
    if (!t) throw new Error(noTrader);
    return t.balance();
  }),
);
ipcMain.handle('ord:open', () =>
  ordRes(async () => {
    const t = trader();
    if (!t) throw new Error(noTrader);
    return t.openOrders();
  }),
);
ipcMain.handle('ord:place', (_e, raw: Partial<OrderReq>) =>
  ordRes(async () => {
    const t = trader();
    if (!t) throw new Error(noTrader);
    if (raw?.side !== 'buy' && raw?.side !== 'sell') throw new Error('매수 / 매도를 고르세요.');
    const o: OrderReq = { code: String(raw.code || ''), side: raw.side, type: raw.type === 'market' ? 'market' : 'limit', qty: Number(raw.qty), price: Number(raw.price) || 0 };
    const bad = orderProblem(o);
    if (bad) throw new Error(bad);
    const text = orderText(o, nameOf(o.code));
    if (!(await confirmOrder(o.side === 'buy' ? '매수 주문 확인' : '매도 주문 확인', text))) throw new Error('주문을 취소했습니다 (전송하지 않음).');
    try {
      const r = await t.place(o);
      orderLog(`${t.kind} ${text} -> 접수 주문번호 ${r.no}`);
      return r;
    } catch (e) {
      orderLog(`${t.kind} ${text} -> 실패 ${errMsg(e)}`);
      throw e;
    }
  }),
);
ipcMain.handle('ord:revise', (_e, raw: Partial<ReviseReq> & { code?: string; side?: string }) =>
  ordRes(async () => {
    const t = trader();
    if (!t) throw new Error(noTrader);
    if (raw?.kind !== 'cancel' && raw?.kind !== 'revise') throw new Error('정정 / 취소를 고르세요.');
    const r: ReviseReq = {
      no: String(raw.no || ''),
      branch: String(raw.branch || ''),
      kind: raw.kind,
      type: raw.type === 'market' ? 'market' : 'limit',
      price: Number(raw.price) || 0,
      qty: Number(raw.qty),
      venue: raw.venue ? String(raw.venue) : undefined,
    };
    const bad = reviseProblem(r);
    if (bad) throw new Error(bad);
    const code = String(raw.code || '');
    const name = nameOf(code) || code;
    const won = (n: number) => n.toLocaleString('ko-KR');
    const side = raw.side === 'sell' ? '매도' : '매수';
    const text = r.kind === 'cancel' ? `취소 ${name} ${side} 주문 ${r.no} 잔량 ${won(r.qty)}주` : `정정 ${name} ${side} 주문 ${r.no} 잔량 ${won(r.qty)}주 -> ${won(r.price)}원`;
    if (!(await confirmOrder(r.kind === 'cancel' ? '주문 취소 확인' : '주문 정정 확인', text))) throw new Error('전송하지 않았습니다.');
    try {
      const res = await t.revise(r);
      orderLog(`${t.kind} ${text} -> 접수 주문번호 ${res.no}`);
      return res;
    } catch (e) {
      orderLog(`${t.kind} ${text} -> 실패 ${errMsg(e)}`);
      throw e;
    }
  }),
);

ipcMain.handle('mkt:call', (_e, method: Method, ...args: unknown[]) => callMarket(method, args));
/** one market call - the renderer's (mkt:call) and a guest's (GuestHost): only the METHODS above, arguments checked */
async function callMarket(method: Method, args: unknown[]): Promise<Res<unknown>> {
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
}

// ---- [2026-11-23] guest mode ----
ipcMain.handle('auth:guest', async (_e, req: { url?: string; password?: string; remember?: boolean; useSaved?: boolean }) => {
  closeSession();
  try {
    const saved = req.useSaved ? savedGuest() : null;
    if (req.useSaved && !saved) throw new Error('저장된 호스트가 없습니다.');
    const url = normalizeHostUrl(saved ? saved.url : String(req.url || ''));
    const password = saved ? saved.password : String(req.password || '');
    if (!url) throw new Error('호스트 주소를 확인하세요 (예: 192.168.0.10 또는 192.168.0.10:47800).');
    if (!password) throw new Error('암호를 입력하세요.');
    const r = new RemoteMarket(url, password);
    const info = await r.hello();
    if (info?.app !== 'AutoBotHtsH') throw new Error('AutoBotHtsH 호스트가 아닙니다.');
    if (info.mode !== 'kis') throw new Error('호스트 PC의 AutoBotHtsH가 KIS에 로그인되어 있지 않습니다.');
    backend = r;
    guestEnv = info.env === 'paper' ? 'paper' : 'real';
    if (!saved) rememberGuest(req.remember ? { url, password } : null);
    log(`[auth] 게스트 접속 (${url}, 호스트 v${info.version})`);
    void master.ensure().then(() => send('auth:state', authState()));
    return { ok: true, data: authState() };
  } catch (e) {
    closeSession();
    return { ok: false, error: errMsg(e) };
  }
});
ipcMain.handle('auth:guestForget', () => (rememberGuest(null), authState()));

const guestHost = new GuestHost({
  // guests are served only from this PC's own KIS login (a guest or demo PC does not relay)
  call: (method, args) => (backend?.kind === 'kis' ? callMarket(method as Method, args) : Promise.resolve({ ok: false as const, error: '호스트 PC의 AutoBotHtsH가 KIS에 로그인되어 있지 않습니다.' })),
  info: () => ({ app: 'AutoBotHtsH', version: versionLabel(), mode: backend?.kind || null, env: client?.env || 'real' }),
  log,
});
function guestHostState() {
  const c = guestHostCfg();
  return { ...guestHost.status(), enabled: !!c?.enabled, port: c?.port || GUEST_PORT_DEFAULT, hasPassword: !!c?.password, passMin: GUEST_PASS_MIN, addresses: lanAddresses() };
}
async function applyGuestHost(): Promise<void> {
  const c = guestHostCfg();
  if (c?.enabled && c.password) await guestHost.start(c.port, c.password);
  else await guestHost.stop();
}
ipcMain.handle('guest:hostState', () => guestHostState());
ipcMain.handle('guest:hostSave', async (_e, req: { enabled?: boolean; port?: number; password?: string }) => {
  try {
    const cur = guestHostCfg();
    const port = Math.round(Number(req.port)) || cur?.port || GUEST_PORT_DEFAULT;
    if (port < 1024 || port > 65535) throw new Error('포트는 1024 ~ 65535 사이로 입력하세요.');
    const password = req.password ? String(req.password) : cur?.password || '';
    if (req.enabled && password.length < GUEST_PASS_MIN) throw new Error(`암호는 ${GUEST_PASS_MIN}자 이상으로 정하세요.`);
    saveGuestHostCfg({ enabled: !!req.enabled, port, password });
    await applyGuestHost();
    return { ok: true, data: guestHostState() };
  } catch (e) {
    return { ok: false, error: errMsg(e) };
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
const history = new ReboundHistory(() => send('rebound:unread', history.unread()));
const kakao = new KakaoTalk(log);
const popups = new AlertPopups(
  (code) => {
    send('rebound:open', code);
    if (win && !win.isDestroyed()) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  },
  (id) => history.markRead([id]),
);
const stockLink = (code: string) => `https://m.stock.naver.com/domestic/stock/${code}/total`;
/** an alert: kept in the history, shown as a popup and (alert.kakao) sent to the user's KakaoTalk */
function raiseAlert(a: AlertInfo, cfg: typeof rebound.config): void {
  const item = history.add(a);
  popups.show(a, cfg, item.id);
  if (!cfg.alert.kakao) return;
  kakao.send(alertSummary(a), stockLink(a.code)).then(
    () => history.setKakao(item.id, 'sent'),
    (e) => history.setKakao(item.id, 'failed', errMsg(e)),
  );
}
const rebound = new ReboundWatcher({
  // demo prices are made up: no watching there (HTSH_REBOUND_DEMO=1 for screen tests)
  backend: () => (backend && (backend.kind === 'kis' || backend.kind === 'guest' || process.env.HTSH_REBOUND_DEMO === '1') ? backend : null),
  master,
  log,
  alert: (a, cfg) => raiseAlert(a, cfg),
  status: (s) => send('rebound:status', s),
});
ipcMain.handle('rebound:status', () => rebound.status);
ipcMain.handle('rebound:enable', (_e, on: boolean) => {
  rebound.setEnabled(!!on);
  return rebound.status;
});
ipcMain.handle('rebound:config', () => {
  rebound.loadConfig();
  return rebound.config;
});
ipcMain.handle('rebound:saveConfig', (_e, cfg: unknown) => ({ config: rebound.saveConfig(cfg), status: rebound.status }));
ipcMain.handle('rebound:refresh', () => rebound.refreshTab());
ipcMain.handle('rebound:test', () => popups.show(sampleAlert(rebound.config), rebound.config));
// ---- [2026-11-22] alert history + unread count ----
ipcMain.handle('rebound:history', () => history.list());
ipcMain.handle('rebound:unread', () => history.unread());
ipcMain.handle('rebound:historyRead', (_e, ids: string[] | null) => history.markRead(Array.isArray(ids) ? ids.map(String) : null));
ipcMain.handle('rebound:historyRemove', (_e, ids: string[] | null) => history.remove(Array.isArray(ids) ? ids.map(String) : null));
ipcMain.handle('rebound:historyPopup', (_e, id: string) => {
  const item = history.get(String(id));
  if (item) popups.show(item, { ...rebound.config, alert: { ...rebound.config.alert, sound: false } }, item.id);
});
// ---- [2026-11-22] KakaoTalk "나에게 보내기" (keys and tokens stay in the main process) ----
const kakaoRun = async (fn: () => Promise<unknown> | unknown): Promise<{ ok: boolean; state: KakaoState; error?: string }> => {
  try {
    await fn();
    return { ok: true, state: kakao.state() };
  } catch (e) {
    return { ok: false, state: kakao.state(), error: errMsg(e) };
  }
};
ipcMain.handle('rebound:kakaoState', () => kakao.state());
ipcMain.handle('rebound:kakaoSave', (_e, a: { restKey?: string; clientSecret?: string; redirectUri?: string }) =>
  kakaoRun(() => kakao.saveApp({ restKey: typeof a?.restKey === 'string' ? a.restKey : '', clientSecret: typeof a?.clientSecret === 'string' ? a.clientSecret : undefined, redirectUri: typeof a?.redirectUri === 'string' ? a.redirectUri : '' })),
);
ipcMain.handle('rebound:kakaoLogin', () => kakaoRun(() => kakao.login(win && !win.isDestroyed() ? win : null)));
ipcMain.handle('rebound:kakaoLogout', () => kakaoRun(() => kakao.logout()));
// [2026-11-22] login files for the bot's own sender (AutoBot g007): the scan job mails the picks and sends them to KakaoTalk
ipcMain.handle('rebound:kakaoBotFiles', () => kakaoRun(() => kakao.exportForBot(win && !win.isDestroyed() ? win : null)));
ipcMain.handle('rebound:kakaoTest', () => kakaoRun(() => kakao.send(`[AutoBotHtsH] 카카오톡 연결 테스트\n${alertSummary(sampleAlert(rebound.config))}`, stockLink('005930'))));
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
  void applyGuestHost();
  // HTSH_DEMO=1: start in demo mode (screen tests)
  if (process.env.HTSH_DEMO === '1') {
    const d = new DemoMarket();
    backend = d;
    feed = new DemoFeed(d, onRt, onRtStatus);
  }
});

app.on('window-all-closed', () => {
  void guestHost.stop();
  closeSession();
  botQuit();
  app.quit();
});
