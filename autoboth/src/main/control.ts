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
 * - [2026-11-07] type=perf - status polling uses its own interval (settings.statusPollSec, default 30 s; 120 s while the window is
 *   minimized/hidden) instead of the 10 s log poll - each call starts a Python process on the Raspberry Pi
 */

import { dialog, ipcMain, type BrowserWindow } from 'electron';
import { userInfo } from 'node:os';
import {
  buildR016Command,
  isMarketHours,
  parseR016Output,
  setArgs,
  statusPollMs,
  watchlistStdin,
  writerName,
  checkWatchRows,
  type ControlState,
  type HistoryRow,
  type R016Result,
  type R016Status,
  type RiskProfile,
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

export interface R016Config {
  remoteRoot: string;
  pythonPath: string;
  by: string;
}

/** Serial queue around ssh exec of r016_control.py. */
export class R016Client {
  private tail: Promise<unknown> = Promise.resolve();
  private queued = 0;
  /** stderr lines of every call (kept for diagnostics, never parsed) */
  readonly stderrLog: string[] = [];

  constructor(
    private exec: ExecFn,
    private cfg: () => R016Config,
  ) {}

  get busy(): boolean {
    return this.queued > 0;
  }

  command(args: string[]): string {
    const c = this.cfg();
    return buildR016Command(c.remoteRoot, c.pythonPath, c.by, args);
  }

  run<T = unknown>(args: string[], opts: { timeoutMs?: number; stdin?: string } = {}): Promise<R016Result<T>> {
    let cmd: string;
    try {
      cmd = this.command(args);
    } catch (e) {
      return Promise.resolve({ ok: false, rc: null, data: null, stderr: '', error: (e as Error).message });
    }
    this.queued++;
    const job = this.tail.then(async () => {
      const r = await this.exec(cmd, opts.timeoutMs ?? R016_TIMEOUT, opts.stdin);
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
  exec: ExecFn;
  isConnected: () => boolean;
  settings: () => { remoteRoot: string; pythonPath: string; pollSec: number; statusPollSec?: number };
  send: (ch: string, data: unknown) => void;
  win: () => BrowserWindow | null;
  /** reload the condition JSON (values_meta) - called after set/unset/reset and when the service start time changes */
  reloadConditions: () => Promise<unknown>;
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

export function createControl(deps: ControlDeps) {
  const by = writerName(userInfo().username);
  const client = new R016Client(deps.exec, () => ({ ...deps.settings(), by }));
  let state: ControlState = { status: null, busy: null };
  let timer: NodeJS.Timeout | null = null;
  let statusInFlight: Promise<R016Status | null> | null = null;

  const push = (patch: Partial<ControlState>) => {
    state = { ...state, ...patch };
    deps.send('control', state);
  };

  async function confirm(title: string, message: string, detail: string, okLabel: string): Promise<boolean> {
    const w = deps.win();
    const opts = { type: 'warning' as const, title, message, detail, buttons: [okLabel, '취소'], defaultId: 1, cancelId: 1, noLink: true };
    const r = w ? await dialog.showMessageBox(w, opts) : await dialog.showMessageBox(opts);
    return r.response === 0;
  }

  /** `status` (coalesced: concurrent callers share one call). */
  function refreshStatus(): Promise<R016Status | null> {
    if (statusInFlight) return statusInFlight;
    statusInFlight = (async () => {
      if (!deps.isConnected()) {
        push({ error: 'SSH가 연결되어 있지 않습니다.' });
        return null;
      }
      try {
        const r = await client.run<R016Status>(['status']);
        if (!r.ok || !r.data) {
          push({ error: r.error || 'status 실패' });
          return null;
        }
        const prev = state.status;
        const s = r.data;
        push({ status: s, statusAt: Date.now(), error: undefined });
        // service (re)started or stopped since the last status -> pending marks in the condition JSON changed
        if (
          prev &&
          (prev.service?.started_at !== s.service?.started_at || (prev.service?.active_state === 'active') !== (s.service?.active_state === 'active'))
        ) {
          void deps.reloadConditions();
        }
        return s;
      } catch (e) {
        push({ error: errMsg(e) });
        return null;
      }
    })().finally(() => (statusInFlight = null));
    return statusInFlight;
  }

  const windowVisible = (): boolean => {
    const w = deps.win();
    return !!w && !w.isDestroyed() && w.isVisible() && !w.isMinimized();
  };

  function startPolling(): void {
    stopPolling();
    const loop = () => {
      timer = setTimeout(async () => {
        // skip a poll while a write is queued - the action itself refreshes the status afterwards
        if (deps.isConnected() && !state.busy && !client.busy) await refreshStatus();
        if (timer) loop();
      }, statusPollMs(deps.settings().statusPollSec, windowVisible()));
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

  async function service(verb: ServiceVerb, opts: ServiceOpts): Promise<OpResult<ServiceResult>> {
    if (state.busy) return { ok: false, error: `다른 작업(${state.busy}) 진행 중입니다.` };
    if (!deps.isConnected()) return { ok: false, error: 'SSH가 연결되어 있지 않습니다.' };
    push({ busy: verb });
    try {
      const cur = await refreshStatus();
      if (!cur) return { ok: false, error: state.error || '현재 상태를 읽지 못했습니다.' };
      if (verb === 'start' && cur.external_r003?.length) {
        return { ok: false, error: '서비스 밖(screen/터미널)에서 r003이 실행 중입니다. 서버의 screen 세션에서 r003을 종료한 뒤 앱에서 실행하세요.' };
      }
      const args: string[] = [verb];
      if (verb !== 'stop') {
        // flags only when they change the stored env (r003.env); none = keep the previous settings
        if (opts.dryRun !== undefined && opts.dryRun !== cur.dry_run) args.push(opts.dryRun ? '--dry-run' : '--live');
        if (opts.profile && opts.profile !== cur.risk_profile) args.push('--profile', opts.profile);
      }
      const dryRun = opts.dryRun ?? cur.dry_run;
      const warn: string[] = [];
      if (verb !== 'stop' && !dryRun) warn.push('실주문 모드로 실행합니다. r003이 실제 매수/매도 주문을 보냅니다.');
      if (verb !== 'start' && isMarketHours(cur.generated_at)) warn.push('장중입니다. 정지된 동안 보유 종목 매도 조건 평가가 중단됩니다.');
      if (warn.length) {
        const label = verb === 'start' ? '실행' : verb === 'stop' ? '정지' : '재시작';
        const prof = opts.profile || cur.risk_profile;
        const detail = verb === 'stop' ? '' : `모드: ${dryRun ? 'dry-run (주문 안 냄)' : '실주문'} · 프로파일: ${prof}`;
        if (!(await confirm(`r003 ${label}`, warn.join('\n'), detail, label))) return { ok: false, cancelled: true };
      }
      let r: R016Result<ServiceResult> | null = null;
      let unknown: string | undefined;
      try {
        r = await client.run<ServiceResult>(args, { timeoutMs: R016_SERVICE_TIMEOUT });
      } catch (e) {
        unknown = errMsg(e);
      }
      const after = await settle(verb);
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

  async function afterOverrideChange(): Promise<void> {
    await deps.reloadConditions();
    await refreshStatus();
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
    const r = await client.run<SchemaDoc>(all ? ['schema', '--all'] : ['schema']);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:set', async (changes: Record<string, unknown>) => {
    const args = setArgs(changes || {});
    if (!args.length) return { ok: true, data: { ok: true, accepted: {}, rejected: {} } };
    const r = await client.run<SetResult>(['set', ...args]);
    // accepted keys are stored even when others were rejected (rc 2) -> always reload
    if (r.data) await afterOverrideChange();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:unset', async (keys: string[]) => {
    if (!keys?.length) return { ok: true };
    const r = await client.run(['unset', ...keys]);
    if (r.ok) await afterOverrideChange();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:reset', async () => {
    if (!(await confirm('설정값 전체 되돌리기', '저장된 오버라이드를 모두 지우고 기본값으로 되돌립니다.', '재시작 후 봇에 반영됩니다.', '전부 되돌리기'))) {
      return { ok: false, cancelled: true };
    }
    const r = await client.run(['reset']);
    if (r.ok) await afterOverrideChange();
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:history', async (n: number) => {
    const r = await client.run<{ rows: HistoryRow[] }>(['history', '-n', String(Math.max(1, Math.min(500, Number(n) || 20)))]);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:watchlistGet', async () => {
    const r = await client.run<WatchlistDoc>(['watchlist', 'get']);
    return { ok: r.ok, error: r.error, data: r.data };
  });
  handle('ctl:watchlistSet', async (rows: WatchEntry[]) => {
    const list = (rows || []).map((r) => ({ code: String(r.code || '').trim(), name: String(r.name || '').trim() }));
    const chk = checkWatchRows(list);
    const bad = Object.entries(chk.rows)[0];
    if (chk.list || bad) return { ok: false, error: chk.list || `${Number(bad[0]) + 1}행: ${bad[1]}` };
    if (
      !(await confirm(
        '감시목록 저장',
        `감시목록을 ${list.length}종목으로 바꿉니다.`,
        '서버가 기존 파일을 watchlist_backup/에 백업한 뒤 교체합니다. r003 재시작 후 반영되며, 매일 아침 g002 스캐너 내보내기가 다시 씁니다.',
        '저장',
      ))
    ) {
      return { ok: false, cancelled: true };
    }
    const r = await client.run<WatchlistSetResult>(['watchlist', 'set', '--stdin'], { stdin: watchlistStdin(list) });
    await refreshStatus();
    return { ok: r.ok, error: r.error, data: r.data };
  });

  return {
    client,
    by,
    startPolling,
    stopPolling,
    refreshStatus,
    clear: () => push({ status: null, statusAt: undefined, error: undefined }),
  };
}
