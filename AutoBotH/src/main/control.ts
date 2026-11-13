/*
 * Update log
 * - [2026-11-05] type=feat - r003 remote control (guide 20261105). R016Client: every server change goes through
 *   `live_trading/r016_control.py` over SSH exec - one serial queue for all calls (never two at once), 30 s timeout (service
 *   verbs 120 s because systemctl waits for r003 to save its state, up to 90 s), stdin for `watchlist set --stdin`, stdout JSON
 *   parsed by shared/control.ts. Controller: status polling every pollSec while connected (coalesced), confirmation dialogs in
 *   the main process (live-order start, stop/restart during market hours, reset, watchlist save), the post-action settle loop
 *   (status every 2 s until the unit leaves activating/deactivating), and a condition reload when the service start time changes
 *   (so "pending" marks clear after a restart). Rejections (rc 2/3) are never retried; an unknown result (timeout / SSH drop) is
 *   resolved by reading `status` again
 * - [2026-11-07] type=feat - NXT session: state.nxt from `get ENABLE_NXT_SESSION` (by the main process right after connect, after set/unset/reset and when
 *   the service start time changes); deps.onNxt lets the main process re-derive the funnel start time
 * - [2026-11-07] type=perf - status polling uses its own interval (settings.statusPollSec, default 30 s; 120 s while the window is
 *   minimized/hidden) instead of the 10 s log poll - each call starts a Python process on the Raspberry Pi
 * - [2026-11-10] type=feat - Local mode: R016Client runs either a shell command over SSH (remote) or argv of a local child process
 *   (R016Client.local, main/localRunner.ts) - same argument checks, same serial queue. The controller takes the client of the
 *   connected host (deps.client), its poll interval (deps.pollMs) and a cross-host guard before start/restart (deps.beforeStart:
 *   the other host must not trade the same account). New channels: ctl:role (get/set primary|standby), ctl:collectorStart,
 *   ctl:jobs / ctl:jobStart / ctl:jobStop / ctl:jobLog (r021 data jobs), ctl:doctor. Saving the watchlist while r003 runs offers
 *   an immediate restart so the new list is watched right away
 * - [2026-11-10] type=fix - Codex review: a service verb pins its client (host) for the whole operation and passes it to the guard;
 *   clear() bumps a connection generation so status / NXT answers of the previous host are dropped; busy() lets the connect refuse
 *   a run-mode switch while a verb runs
 * - [2026-11-11] type=feat - ctl:jobStart keepWatchlist -> r021 --keep-watchlist (scan builds the picks files only, r004 stays);
 *   while r003 runs, a scan / postclose that WOULD rewrite r004 asks first (one dialog together with the KIS-load note)
 * - [2026-11-11] type=fix - Tick collector resubscribe: g011 takes the watchlist only when it starts, so codes added later had no
 *   tick files (_009 ABSTAIN_NO_FILE, empty tick chart). ctl:watchlistSet restarts g011 (r016 `collector restart`, which now
 *   respawns it) when status.collector.missing is not empty; new ctl:collectorRestart for the control tab button
 * - [2026-11-11] type=feat - askConfirm(): confirmations are drawn by the renderer (ConfirmDialog) in the app's style instead of the
 *   native message box; confirmBox() keeps its signature on top of it. The r003 start/restart/stop question is a card: LIVE / 장중
 *   tag, "실주문 모드로 실행할까요?", one-line reason and rows 모드 / 프로파일 / 실행 위치 (host label + hostname)
 * - [2026-11-11] type=feat - ctl:jobData (r016 `job data --date D`: collected data + g002 picks of a date, r004 comparison) and
 *   ctl:picksApply (`watchlist apply --date D` after a confirmation; same g011 restart / r003 restart offer as ctl:watchlistSet,
 *   now shared in afterWatchlistWrite) - the data jobs tab runs g002 alone on a collected date and applies the picks afterwards
 * - [2026-11-11] type=refactor - ctl:jobStart takes the pipeline steps (collect / scan / apply -> r021 --no-*) instead of
 *   keepWatchlist; the r003-running note names ③ r004 반영. ctl:picksApply removed - r004 is written by the job's apply step
 */

import { dialog, ipcMain, type BrowserWindow } from 'electron';
import { AUTO_SELL_KEY, AUTO_SELL_SCOPE_KEY, PAPER_KEY, STRATEGY_KEY, parseAutoSell, parseAutoSellScope, parseModes } from '../shared/paper';
import type { ArgvRunner } from './localRunner';
import type { ConfirmRequest, ConfirmRow, ConfirmSpec } from '../shared/types';
import {
  argProblem,
  buildR016Command,
  isMarketHours,
  NXT_KEY,
  parseNxt,
  parseR016Output,
  setArgs,
  watchlistStdin,
  checkWatchRows,
  JOB_NAMES,
  type ControlState,
  type DoctorReport,
  type HistoryRow,
  type HoldingsDoc,
  type JobDataInfo,
  type JobList,
  type JobLog,
  type JobStartOpts,
  type NxtState,
  type R016Result,
  type R016Status,
  type RiskProfile,
  type RoleInfo,
  type SchemaDoc,
  type ServiceResult,
  type SetResult,
  type WatchEntry,
  type WatchlistDoc,
  type WatchlistSetResult,
} from '../shared/control';

export type ExecFn = (cmd: string, timeoutMs: number, stdin?: string) => Promise<{ out: string; err: string; code: number | null }>;

export const R016_TIMEOUT = 30000;
export const R016_SERVICE_TIMEOUT = 120000;
export const R016_CONDITIONS_TIMEOUT = 90000;
/** job start waits for the runner lock (10 s); job stop for the current step to end (up to ~25 s) */
export const R016_JOB_TIMEOUT = 60000;

export interface R016Config {
  remoteRoot: string;
  pythonPath: string;
  by: string;
}

/**
 * Serial queue around r016_control.py: remote = one shell command over ssh exec (`exec`), local = argv of a child process
 * (`argv`, main/localRunner.ts). Arguments are checked the same way for both.
 */
export class R016Client {
  private tail: Promise<unknown> = Promise.resolve();
  private queued = 0;
  /** stderr lines of every call (kept for diagnostics, never parsed) */
  readonly stderrLog: string[] = [];

  constructor(
    private exec: ExecFn | null,
    private cfg: () => R016Config,
    private argv: ArgvRunner | null = null,
  ) {}

  /** local client: argv runner, no shell */
  static local(runner: ArgvRunner): R016Client {
    return new R016Client(null, () => ({ remoteRoot: '', pythonPath: '', by: '' }), runner);
  }

  get busy(): boolean {
    return this.queued > 0;
  }

  command(args: string[]): string {
    const c = this.cfg();
    return buildR016Command(c.remoteRoot, c.pythonPath, c.by, args);
  }

  run<T = unknown>(args: string[], opts: { timeoutMs?: number; stdin?: string } = {}): Promise<R016Result<T>> {
    let call: (timeoutMs: number, stdin?: string) => Promise<{ out: string; err: string; code: number | null }>;
    try {
      if (this.argv) {
        for (const a of args) {
          const p = argProblem(a);
          if (p) throw new Error(p);
        }
        const runner = this.argv;
        call = (t, s) => runner(args, t, s);
      } else {
        const cmd = this.command(args);
        const exec = this.exec!;
        call = (t, s) => exec(cmd, t, s);
      }
    } catch (e) {
      return Promise.resolve({ ok: false, rc: null, data: null, stderr: '', error: (e as Error).message });
    }
    this.queued++;
    const job = this.tail.then(async () => {
      const r = await call(opts.timeoutMs ?? R016_TIMEOUT, opts.stdin);
      if (r.err.trim()) {
        this.stderrLog.push(`[${new Date().toISOString()}] ${args[0]}: ${r.err.trim().slice(0, 2000)}`);
        if (this.stderrLog.length > 50) this.stderrLog.shift();
      }
      return parseR016Output<T>(r.out, r.err, r.code);
    });
    const done = job.finally(() => this.queued--);
    this.tail = done.catch(() => undefined);
    return done;
  }
}

// ---- controller ------------------------------------------------------------------------------

export interface ControlDeps {
  /** r016 client of the host the app is connected to (remote Raspberry Pi or this PC) */
  client: () => R016Client;
  isConnected: () => boolean;
  /** status poll interval in ms (run mode / window visibility) */
  pollMs: () => number;
  /** "라즈베리파이" / "이 PC" - used in messages */
  hostLabel: () => string;
  send: (ch: string, data: unknown) => void;
  win: () => BrowserWindow | null;
  /** reload the condition JSON (values_meta) - called after set/unset/reset and when the service start time changes */
  reloadConditions: () => Promise<unknown>;
  /** the server NXT setting was read and changed (or read for the first time) */
  onNxt?: (nxt: NxtState) => void;
  /** every successful status (keep-awake, scheduler) */
  onStatus?: (s: R016Status) => void;
  /**
   * Before start/restart: make sure the other host does not trade this account (its r003 stopped, its role standby, this
   * host primary). Returns an error to refuse, or cancelled when the user declined a confirmation.
   */
  beforeStart?: (verb: ServiceVerb, client: R016Client) => Promise<{ ok: boolean; error?: string; cancelled?: boolean }>;
}

const errMsg = (e: unknown) => String((e as Error)?.message || e).split('\n')[0];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export type ServiceVerb = 'start' | 'stop' | 'restart';
export interface ServiceOpts {
  dryRun?: boolean;
  profile?: RiskProfile;
}
export interface OpResult<T = unknown> {
  ok: boolean;
  error?: string;
  cancelled?: boolean;
  data?: T | null;
}

/**
 * [2026-11-11] Confirmations are shown by the renderer in the app's own style (components/ConfirmDialog): 'confirm:ask' carries the
 * ConfirmSpec + id, 'confirm:answer' brings the answer back. A closed window answers "no". Electron's native message box is used
 * only when there is no window to draw in (or it is still loading).
 */
const pendingConfirms = new Map<number, (ok: boolean) => void>();
let confirmSeq = 0;
let confirmIpcReady = false;
export async function askConfirm(win: BrowserWindow | null, spec: ConfirmSpec): Promise<boolean> {
  if (!win || win.isDestroyed() || win.webContents.isLoading()) {
    const detail = [spec.message, ...(spec.rows || []).map((r) => `${r.k}: ${r.v}`)].filter(Boolean).join('\n');
    const opts = { type: 'warning' as const, title: spec.kicker, message: spec.title, detail, buttons: [spec.okLabel, spec.cancelLabel || '취소'], defaultId: 1, cancelId: 1, noLink: true };
    const r = win && !win.isDestroyed() ? await dialog.showMessageBox(win, opts) : await dialog.showMessageBox(opts);
    return r.response === 0;
  }
  if (!confirmIpcReady) {
    confirmIpcReady = true;
    ipcMain.on('confirm:answer', (_e, id: number, ok: boolean) => {
      const done = pendingConfirms.get(id);
      if (done) done(!!ok);
    });
  }
  const id = ++confirmSeq;
  return new Promise<boolean>((resolve) => {
    const onClosed = () => done(false);
    const done = (ok: boolean) => {
      if (!pendingConfirms.delete(id)) return;
      win.removeListener('closed', onClosed);
      resolve(ok);
    };
    pendingConfirms.set(id, done);
    win.once('closed', onClosed);
    if (win.isMinimized()) win.restore();
    win.focus();
    win.webContents.send('confirm:ask', { id, ...spec } as ConfirmRequest);
  });
}

/** title = kicker line, message = the question, detail = the line under it (kept for the older call sites) */
export function confirmBox(win: BrowserWindow | null, title: string, message: string, detail: string, okLabel: string): Promise<boolean> {
  return askConfirm(win, { kicker: title, title: message, message: detail, okLabel });
}

export function createControl(deps: ControlDeps) {
  const cl = () => deps.client();
  let state: ControlState = { status: null, busy: null };
  let timer: NodeJS.Timeout | null = null;
  let statusInFlight: Promise<R016Status | null> | null = null;
  /** connection generation: bumped by clear() (disconnect / run-mode switch) - answers of an older generation are dropped */
  let gen = 0;

  const push = (patch: Partial<ControlState>) => {
    state = { ...state, ...patch };
    deps.send('control', state);
  };

  const confirm = (title: string, message: string, detail: string, okLabel: string) => confirmBox(deps.win(), title, message, detail, okLabel);
  const notConnected = () => `${deps.hostLabel()}에 연결되어 있지 않습니다.`;

  /** `status` (coalesced: concurrent callers share one call). */
  function refreshStatus(): Promise<R016Status | null> {
    if (statusInFlight) return statusInFlight;
    const myGen = gen;
    const p: Promise<R016Status | null> = (async () => {
      if (!deps.isConnected()) {
        push({ error: notConnected() });
        return null;
      }
      try {
        const r = await cl().run<R016Status>(['status']);
        if (myGen !== gen) return null; // asked before a disconnect / host switch: the answer is about the old host
        if (!r.ok || !r.data) {
          push({ error: r.error || 'status 실패' });
          return null;
        }
        const prev = state.status;
        const s = r.data;
        push({ status: s, statusAt: Date.now(), error: undefined });
        deps.onStatus?.(s);
        // service (re)started or stopped since the last status -> pending marks in the condition JSON changed
        if (
          prev &&
          (prev.service?.started_at !== s.service?.started_at || (prev.service?.active_state === 'active') !== (s.service?.active_state === 'active'))
        ) {
          void deps.reloadConditions();
          void refreshNxt();
        }
        return s;
      } catch (e) {
        if (myGen === gen) push({ error: errMsg(e) });
        return null;
      }
    })().finally(() => {
      if (statusInFlight === p) statusInFlight = null;
    });
    statusInFlight = p;
    return p;
  }

  /** ENABLE_NXT_SESSION (base / override / running). Failures keep the last known value. */
  async function refreshNxt(): Promise<NxtState | undefined> {
    if (!deps.isConnected()) return state.nxt;
    const myGen = gen;
    try {
      const r = await cl().run(['get', NXT_KEY]);
      if (myGen !== gen) return state.nxt;
      const nxt = r.ok ? parseNxt(r.data) : null;
      if (!nxt) return state.nxt;
      const prev = state.nxt;
      push({ nxt });
      if (!prev || JSON.stringify(prev) !== JSON.stringify(nxt)) deps.onNxt?.(nxt);
      return nxt;
    } catch {
      return state.nxt;
    }
  }

  function startPolling(): void {
    stopPolling();
    const loop = () => {
      timer = setTimeout(async () => {
        // skip a poll while a write is queued - the action itself refreshes the status afterwards
        if (deps.isConnected() && !state.busy && !cl().busy) await refreshStatus();
        if (timer) loop();
      }, deps.pollMs());
    };
    loop();
    void refreshStatus();
  }

  function stopPolling(): void {
    if (timer) clearTimeout(timer);
    timer = null;
  }

  /** After a service verb: status every 2 s until the unit settles (stop: up to 100 s, start/restart: 40 s). */
  async function settle(verb: ServiceVerb): Promise<R016Status | null> {
    const limit = Date.now() + (verb === 'stop' ? 100000 : 40000);
    let s = await refreshStatus();
    while (Date.now() < limit) {
      const st = s?.service?.active_state;
      const transitional = st === 'activating' || st === 'deactivating' || s?.service?.sub_state === 'auto-restart';
      if (s && !transitional && (verb !== 'stop' || st !== 'active')) break;
      await sleep(2000);
      if (!deps.isConnected()) break;
      s = await refreshStatus();
    }
    return s;
  }

  async function service(verb: ServiceVerb, opts: ServiceOpts, skipConfirm = false): Promise<OpResult<ServiceResult>> {
    if (state.busy) return { ok: false, error: `다른 작업(${state.busy}) 진행 중입니다.` };
    if (!deps.isConnected()) return { ok: false, error: notConnected() };
    // the host this operation acts on, fixed for its whole duration (connect refuses to switch the run mode while busy)
    const client = cl();
    const myGen = gen;
    push({ busy: verb });
    try {
      const cur = await refreshStatus();
      if (!cur) return { ok: false, error: state.error || '현재 상태를 읽지 못했습니다.' };
      if (verb === 'start' && cur.external_r003?.length) {
        return {
          ok: false,
          error:
            cur.host?.platform === 'windows'
              ? '이 PC에서 감시 프로세스 밖으로 실행된 r003이 있습니다 (콘솔 등). [정지]로 먼저 끝낸 뒤 실행하세요.'
              : '서비스 밖(screen/터미널)에서 r003이 실행 중입니다. 서버의 screen 세션에서 r003을 종료한 뒤 앱에서 실행하세요.',
        };
      }
      const args: string[] = [verb];
      if (verb !== 'stop') {
        // flags only when they change the stored env (r003.env); none = keep the previous settings
        if (opts.dryRun !== undefined && opts.dryRun !== cur.dry_run) args.push(opts.dryRun ? '--dry-run' : '--live');
        if (opts.profile && opts.profile !== cur.risk_profile) args.push('--profile', opts.profile);
      }
      const dryRun = opts.dryRun ?? cur.dry_run;
      const live = verb !== 'stop' && !dryRun; // real orders
      const market = verb !== 'start' && isMarketHours(cur.generated_at); // sell evaluation pauses while stopped
      if ((live || market) && !skipConfirm) {
        const label = verb === 'start' ? '실행' : verb === 'stop' ? '정지' : '재시작';
        const prof = opts.profile || cur.risk_profile;
        const where = `${deps.hostLabel()}${cur.host?.hostname ? ` · ${cur.host.hostname}` : ''}`;
        const rows: ConfirmRow[] = [];
        if (verb !== 'stop') {
          rows.push({ k: '모드', v: dryRun ? 'dry-run (주문 안 냄)' : '실주문 (dry-run 꺼짐)', tone: live ? 'live' : undefined }, { k: '프로파일', v: String(prof || '-') });
        }
        rows.push({ k: '실행 위치', v: where });
        const pause = verb === 'stop' ? '정지된 동안' : '재시작되는 동안';
        const spec: ConfirmSpec = live
          ? {
              tag: 'LIVE',
              tagTone: 'live',
              kicker: `r003 ${label}`,
              title: `실주문 모드로 ${label}할까요?`,
              message: `r003이 ${deps.hostLabel()}에서 실제 매수·매도 주문을 전송합니다.${market ? ` 장중이라 ${pause} 보유 종목의 매도 조건 평가가 멈춥니다.` : ''}`,
              rows,
              okLabel: `실주문으로 ${label}`,
            }
          : {
              tag: '장중',
              tagTone: 'warn',
              kicker: `r003 ${label}`,
              title: `장중에 ${label}할까요?`,
              message: `${pause} 보유 종목의 매도 조건 평가가 중단됩니다.`,
              rows,
              okLabel: label,
            };
        if (!(await askConfirm(deps.win(), spec))) return { ok: false, cancelled: true };
      }
      if (verb !== 'stop' && deps.beforeStart) {
        const g = await deps.beforeStart(verb, client);
        if (!g.ok) return { ok: false, error: g.error, cancelled: g.cancelled };
      }
      if (myGen !== gen || client !== cl()) return { ok: false, error: '연결이 바뀌어 취소했습니다.' };
      let r: R016Result<ServiceResult> | null = null;
      let unknown: string | undefined;
      try {
        r = await client.run<ServiceResult>(args, { timeoutMs: R016_SERVICE_TIMEOUT });
      } catch (e) {
        unknown = errMsg(e);
      }
      const after = await settle(verb);
      void refreshPaper(); // a (re)start applies a stored mode change: running / pending change with it
      const st = after?.service?.active_state || '?';
      let res: OpResult<ServiceResult>;
      if (unknown) res = { ok: false, error: `결과를 받지 못했습니다 (${unknown}). 상태를 다시 읽었습니다: ${st}` };
      else if (!r!.ok) res = { ok: false, error: r!.error, data: r!.data };
      else res = { ok: true, data: r!.data };
      push({ lastAction: { verb, ok: res.ok, message: res.ok ? `${verb} 완료 - 현재 ${st}` : res.error || '실패', at: Date.now() } });
      return res;
    } finally {
      push({ busy: null });
    }
  }

  /** [2026-11-12] EDGE_ONLY_MODE (paper trading on / off, restart pending). A server without the key leaves it undefined. */
  async function refreshPaper(): Promise<void> {
    if (!deps.isConnected()) return;
    const myGen = gen;
    try {
      const r = await cl().run(['get', STRATEGY_KEY, PAPER_KEY, AUTO_SELL_KEY, AUTO_SELL_SCOPE_KEY]);
      if (myGen !== gen) return;
      const m = parseModes(r.data);
      push({
        paper: m.paper || undefined,
        strategy: m.strategy || undefined,
        autoSell: parseAutoSell(r.data) || undefined,
        autoSellScope: parseAutoSellScope(r.data) || undefined,
      });
    } catch {
      /* keep the last known value */
    }
  }

  async function afterOverrideChange(): Promise<void> {
    await deps.reloadConditions();
    await refreshStatus();
    await refreshNxt();
    await refreshPaper();
  }

  // ---- IPC ----
  /** exec errors (timeout, SSH drop) become { ok:false } instead of a rejected invoke */
  const handle = (ch: string, fn: (...a: never[]) => unknown) =>
    ipcMain.handle(ch, async (_e, ...a: unknown[]) => {
      try {
        return await (fn as (...x: unknown[]) => unknown)(...a);
      } catch (e) {
        return { ok: false, error: errMsg(e) };
      }
    });
  const needConn = () => (deps.isConnected() ? null : { ok: false, error: notConnected() });
  ipcMain.handle('ctl:state', () => state);
  ipcMain.handle('ctl:status', async () => {
    await refreshStatus();
    return state;
  });
  handle('ctl:service', (verb: ServiceVerb, opts: ServiceOpts) => {
    if (!['start', 'stop', 'restart'].includes(verb)) return { ok: false, error: `알 수 없는 동작: ${verb}` };
    return service(verb, opts || {});
  });
  handle('ctl:schema', async (all: boolean) => {
    const r = await cl().run<SchemaDoc>(all ? ['schema', '--all'] : ['schema']);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:set', async (changes: Record<string, unknown>) => {
    const args = setArgs(changes || {});
    if (!args.length) return { ok: true, data: { ok: true, accepted: {}, rejected: {} } };
    const r = await cl().run<SetResult>(['set', ...args]);
    // accepted keys are stored even when others were rejected (rc 2) -> always reload
    if (r.data) await afterOverrideChange();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:unset', async (keys: string[]) => {
    if (!keys?.length) return { ok: true };
    const r = await cl().run(['unset', ...keys]);
    if (r.ok) await afterOverrideChange();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:reset', async () => {
    if (!(await confirm('설정값 전체 되돌리기', '저장된 오버라이드를 모두 지우고 기본값으로 되돌립니다.', '재시작 후 봇에 반영됩니다.', '전부 되돌리기'))) {
      return { ok: false, cancelled: true };
    }
    const r = await cl().run(['reset']);
    if (r.ok) await afterOverrideChange();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  // [2026-11-13] account holdings snapshot with the bot's / hand-bought quantity (r016 holdings, read-only, no KIS call)
  handle('ctl:holdings', async () => {
    const off = needConn();
    if (off) return off;
    const r = await cl().run<HoldingsDoc>(['holdings']);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:history', async (n: number) => {
    const r = await cl().run<{ rows: HistoryRow[] }>(['history', '-n', String(Math.max(1, Math.min(500, Number(n) || 20)))]);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:watchlistGet', async () => {
    const r = await cl().run<WatchlistDoc>(['watchlist', 'get']);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  /** [2026-11-11] after r004 was rewritten (watchlist set / apply): g011 subscribes the watchlist once at its start, so a list
   *  changed afterwards leaves the new codes without tick files (_009 1분봉 트리거 ABSTAIN_NO_FILE, empty tick chart) -> restart
   *  it right away with the current r004; then offer an r003 restart so the new list is watched now (held positions keep their
   *  sell evaluation - r003 restores today's positions from live_state and the account) */
  const afterWatchlistWrite = async (r: { ok: boolean; error?: string; data?: WatchlistSetResult | null }) => {
    let cur = await refreshStatus();
    let collectorRestart: unknown = undefined;
    if (r.ok && cur?.collector?.running && (cur.collector.missing?.length || 0) > 0) {
      try {
        const x = await cl().run(['collector', 'restart'], { timeoutMs: R016_SERVICE_TIMEOUT });
        collectorRestart = x.ok ? x.data : { ok: false, error: x.error };
      } catch (e) {
        collectorRestart = { ok: false, error: errMsg(e) };
      }
      cur = await refreshStatus();
    }
    if (r.ok && cur?.running && !state.busy) {
      if (await confirm('감시 다시 시작', '새 감시목록을 바로 감시하려면 r003을 재시작해야 합니다.', '지금 재시작할까요? (보유 종목은 재시작 후에도 계속 매도 평가합니다)', '재시작')) {
        void service('restart', {}, true);
      }
    }
    return { ok: r.ok, error: r.error, data: r.data && collectorRestart !== undefined ? { ...r.data, collector_restart: collectorRestart } : r.data };
  };
  handle('ctl:watchlistSet', async (rows: WatchEntry[]) => {
    const list = (rows || []).map((r) => ({ code: String(r.code || '').trim(), name: String(r.name || '').trim() }));
    const chk = checkWatchRows(list);
    const bad = Object.entries(chk.rows)[0];
    if (chk.list || bad) return { ok: false, error: chk.list || `${Number(bad[0]) + 1}행: ${bad[1]}` };
    if (
      !(await confirm(
        '감시목록 저장',
        `감시목록을 ${list.length}종목으로 바꿉니다 (${deps.hostLabel()}).`,
        '기존 파일을 watchlist_backup/에 백업한 뒤 교체합니다. r003 재시작 후 반영되며, 장 마감 후 g002 선별(감시 종목 선별 작업)이 다시 씁니다.',
        '저장',
      ))
    ) {
      return { ok: false, cancelled: true };
    }
    const r = await cl().run<WatchlistSetResult>(['watchlist', 'set', '--stdin'], { stdin: watchlistStdin(list) });
    return afterWatchlistWrite(r);
  });
  // [2026-11-11] pipeline card of the data jobs tab: what the date folder holds (step defaults / refusals, picks list)
  handle('ctl:jobData', async (date: string) => {
    const nc = needConn();
    if (nc) return nc;
    if (!/^\d{8}$/.test(String(date))) return { ok: false, error: '날짜는 YYYYMMDD' };
    const r = await cl().run<JobDataInfo>(['job', 'data', '--date', date]);
    return { ok: r.ok, error: r.error, data: r.data };
  });


  // [2026-11-10] host role / tick collector / data jobs / doctor (same r016 commands on both hosts)
  /** run `fn` with the busy flag and the pinned client, after the cross-host guard (other host standby, this host primary) */
  async function guarded<T>(label: string, fn: (c: R016Client) => Promise<T>): Promise<T | OpResult> {
    if (state.busy) return { ok: false, error: `다른 작업(${state.busy}) 진행 중입니다.` };
    const client = cl();
    push({ busy: label });
    try {
      if (deps.beforeStart) {
        const g = await deps.beforeStart('start', client);
        if (!g.ok) return { ok: false, error: g.error, cancelled: g.cancelled };
      }
      if (client !== cl()) return { ok: false, error: '연결이 바뀌어 취소했습니다.' };
      return await fn(client);
    } finally {
      push({ busy: null });
    }
  }
  handle('ctl:role', async (role?: 'primary' | 'standby') => {
    const nc = needConn();
    if (nc) return nc;
    if (role && role !== 'primary' && role !== 'standby') return { ok: false, error: `알 수 없는 역할: ${role}` };
    if (role === 'primary') {
      // becoming primary = the same cross-host guard as an r003 start (other host checked and made standby, then this one primary)
      const r = await guarded('role', async (c) => {
        const g = await c.run<RoleInfo>(['role', 'get']);
        return { ok: g.ok, error: g.error, data: g.data };
      });
      await refreshStatus();
      return r;
    }
    const r = await cl().run<RoleInfo>(role ? ['role', 'set', role, '--note', 'AutoBotH'] : ['role', 'get']);
    if (role) await refreshStatus();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:collectorStart', async (force?: boolean) => {
    const nc = needConn();
    if (nc) return nc;
    // a g011 here while the other host still collects would fight over the KIS websocket (one session per appkey)
    const r = await guarded('collector', async (c) => {
      const x = await c.run(['collector', 'start', ...(force ? ['--force'] : [])]);
      return { ok: x.ok, error: x.error, data: x.data };
    });
    await refreshStatus();
    return r;
  });
  handle('ctl:collectorRestart', async () => {
    const nc = needConn();
    if (nc) return nc;
    const r = await guarded('collector', async (c) => {
      const x = await c.run(['collector', 'restart'], { timeoutMs: R016_SERVICE_TIMEOUT });
      return { ok: x.ok, error: x.error, data: x.data };
    });
    await refreshStatus();
    return r;
  });
  handle('ctl:jobs', async () => {
    const nc = needConn();
    if (nc) return nc;
    const r = await cl().run<JobList>(['job', 'list']);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:jobStart', async (name: string, opts: JobStartOpts) => {
    const nc = needConn();
    if (nc) return nc;
    if (!JOB_NAMES.includes(name)) return { ok: false, error: `알 수 없는 작업: ${name}` };
    const o = opts || {};
    const args = ['job', 'start', name];
    if (o.date) {
      if (!/^\d{8}$/.test(o.date)) return { ok: false, error: '날짜는 YYYYMMDD' };
      args.push('--date', o.date);
    }
    if (o.workers) args.push('--workers', String(Math.max(1, Math.min(32, Math.floor(Number(o.workers)) || 1))));
    if (o.edgeAi) args.push('--edge-ai');
    if (o.tradingDayOnly) args.push('--trading-day-only');
    // [2026-11-11] pipeline steps: a step set to false is skipped (r021 --no-*); undefined keeps the server default (on)
    const stepOn = (k: 'collect' | 'scan' | 'apply') => o[k] !== false;
    for (const k of ['collect', 'scan', 'apply'] as const) if (!stepOn(k)) args.push(`--no-${k}`);
    // [2026-11-11] the scan step mails its picks (g002 --notify-email) - nothing to send without the scan step
    if (o.email && stepOn('scan') && (name === 'postclose' || name === 'scan')) args.push('--email');
    if (!o.noConfirm && state.status?.running) {
      const notes: string[] = [];
      if ((name === 'collect' || name === 'postclose') && stepOn('collect')) notes.push('수집(g001)은 KIS API를 많이 호출해 r003의 시세 조회와 호출 한도를 나눠 씁니다.');
      if ((name === 'scan' || name === 'postclose') && stepOn('apply')) {
        notes.push('③ r004 반영은 감시목록을 덮어씁니다 - 지금 감시 중인 종목이 다음 실행/재시작부터 바뀝니다. 감시목록을 그대로 두려면 ③을 끄세요.');
      }
      if (notes.length && !(await confirm('데이터 작업', 'r003이 실행 중입니다.', `${notes.join(' ')} 그래도 시작할까요?`, '시작'))) {
        return { ok: false, cancelled: true };
      }
    }
    const r = await cl().run(args, { timeoutMs: R016_JOB_TIMEOUT });
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:jobStop', async (name: string) => {
    const nc = needConn();
    if (nc) return nc;
    if (!JOB_NAMES.includes(name)) return { ok: false, error: `알 수 없는 작업: ${name}` };
    const r = await cl().run(['job', 'stop', name], { timeoutMs: R016_JOB_TIMEOUT });
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:jobLog', async (name: string, n?: number) => {
    const nc = needConn();
    if (nc) return nc;
    if (!JOB_NAMES.includes(name)) return { ok: false, error: `알 수 없는 작업: ${name}` };
    const r = await cl().run<JobLog>(['job', 'log', name, '-n', String(Math.max(10, Math.min(2000, Number(n) || 200)))]);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:doctor', async () => {
    const nc = needConn();
    if (nc) return nc;
    const r = await cl().run<DoctorReport>(['doctor']);
    return { ok: !!r.data, error: r.data ? undefined : r.error, data: r.data };
  });

  return {
    startPolling,
    stopPolling,
    refreshStatus,
    refreshNxt,
    refreshPaper,
    /** start / stop / restart without the renderer (scheduler) - confirmations still apply unless skipConfirm */
    service,
    status: () => state.status,
    nxt: () => state.nxt,
    /** a service verb is running (the run mode must not change under it) */
    busy: () => !!state.busy,
    /** disconnect / run-mode switch: forget the old host's state and drop its answers still in flight */
    clear: () => {
      gen++;
      statusInFlight = null;
      push({ status: null, statusAt: undefined, error: undefined, nxt: undefined, paper: undefined, strategy: undefined, autoSell: undefined, autoSellScope: undefined, lastAction: undefined });
    },
  };
}
