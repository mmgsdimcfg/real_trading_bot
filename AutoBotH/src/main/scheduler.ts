/*
 * Update log
 * - [2026-11-10] type=feat - Local-mode scheduler (replaces the Raspberry Pi cron while this PC is the main host; runs only while the
 *   app runs - the jobs it starts are detached and finish even if the app is closed):
 *   - tick collector: Mon-Fri 07:55-19:55 (KRX + NXT sessions, [2026-11-12] was 08:40-15:35) every 5 min r016 `collector start` (g022 rules on the PC: calendar window, standby refusal,
 *     no second g011)
 *   - post-close: once per day at/after settings.postcloseTime `job start postclose --trading-day-only [--edge-ai] [--email]`, never while
 *     r003 still runs (NXT session until 20:00)
 *   - after that job finished OK: copy the new watchlist to the standby Raspberry Pi (settings.syncWatchlistToPeer) so it can take
 *     over tomorrow
 * - [2026-11-11] type=fix - The 5-min collector check restarts a running g011 whose subscription misses watch codes
 *   (status.collector.missing - the watchlist changed after it started), so new watch symbols get tick files within 5 min
 * - [2026-11-10] type=fix - Codex review: the tick collector only on a host whose role is explicitly primary (set by a guarded r003
 *   start); every action re-checks that the local mode is still active; the watchlist copy is marked done only after it succeeded
 *   (retried every 10 min)
 * - [2026-11-11] type=feat - settings.postcloseEmail adds --email to the scheduled post-close job (g002 --notify-email: picks by mail)
 * - [2026-11-11] type=feat - The scheduled post-close job is started with --scheduled (state args.scheduled -> 자동 실행 in the jobs tab)
 */

import { collectorWindow, localYmd, postcloseDue, type JobList, type R016Status } from '../shared/control';
import type { Settings } from '../shared/types';
import type { R016Client } from './control';

export interface SchedState {
  enabled: boolean;
  lastCollectorAt?: number;
  lastCollectorMsg?: string;
  lastPostcloseMsg?: string;
  lastSyncMsg?: string;
}

export interface SchedDeps {
  active: () => boolean;
  settings: () => Settings;
  client: () => R016Client;
  status: () => R016Status | null;
  meta: () => { lastPostcloseYmd?: string; lastPostcloseSyncYmd?: string };
  setMeta: (p: { lastPostcloseYmd?: string; lastPostcloseSyncYmd?: string }) => void;
  syncToPeer: () => Promise<{ ok: boolean; error?: string; count?: number }>;
  send: (s: SchedState) => void;
}

const COLLECTOR_EVERY_MS = 5 * 60 * 1000;
const SYNC_RETRY_MS = 10 * 60 * 1000;
const TICK_MS = 60 * 1000;
const errMsg = (e: unknown) => String((e as Error)?.message || e).split('\n')[0];
const hm = (d = new Date()) => `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;

export function createScheduler(deps: SchedDeps) {
  let timer: NodeJS.Timeout | null = null;
  let busy = false;
  let lastSyncTryAt = 0;
  let state: SchedState = { enabled: false };
  const push = (p: Partial<SchedState>) => {
    state = { ...state, ...p };
    deps.send(state);
  };

  async function tick(): Promise<void> {
    if (busy) return;
    if (!deps.active()) {
      if (state.enabled) push({ enabled: false });
      return;
    }
    busy = true;
    try {
      if (!state.enabled) push({ enabled: true });
      const s = deps.settings();
      const now = new Date();
      const role = deps.status()?.host?.role;
      if (s.autoCollector && collectorWindow(now) && (!state.lastCollectorAt || Date.now() - state.lastCollectorAt >= COLLECTOR_EVERY_MS)) {
        state.lastCollectorAt = Date.now();
        // only on a host that is explicitly primary (set by an r003 start through the cross-host guard): a PC without a role
        // file counts as primary for r016, and a g011 here while the Pi still collects would fight over the KIS websocket
        if (!role?.set || role.role !== 'primary') {
          push({ lastCollectorMsg: `${hm()} 틱 수집기 대기 - 이 PC가 명시적 primary 가 아님 (제어 탭 [실행] 시 라즈베리파이를 standby, 이 PC를 primary 로 정함)` });
        } else if (deps.active()) {
          try {
            // a running g011 that misses watch codes (list changed after its start) is restarted with the current list
            const coll = deps.status()?.collector;
            const missing = coll?.running ? coll.missing || [] : [];
            if (missing.length) {
              const r = await deps.client().run<{ restarted?: boolean; pid?: number; window?: string }>(['collector', 'restart'], { timeoutMs: 60000 });
              push({
                lastCollectorMsg: r.ok
                  ? `${hm()} 틱 수집기 재시작 - 구독 누락 ${missing.length}종목 (${missing.join(', ')})${r.data?.restarted ? ` -> pid ${r.data.pid}` : r.data?.window ? ` - 다시 시작 안 함: ${r.data.window}` : ''}`
                  : `${hm()} 틱 수집기 재시작 실패: ${r.error}`,
              });
            } else {
              const r = await deps.client().run<{ already_running?: boolean; started?: boolean; pid?: number }>(['collector', 'start']);
              const d = r.data;
              push({
                lastCollectorMsg: r.ok ? `${hm()} ${d?.started ? `틱 수집기 시작 (pid ${d.pid})` : '틱 수집기 실행 중'}` : `${hm()} 틱 수집기: ${r.error}`,
              });
            }
          } catch (e) {
            push({ lastCollectorMsg: `${hm()} 틱 수집기: ${errMsg(e)}` });
          }
        }
      }
      const meta = deps.meta();
      const today = localYmd(now);
      if (s.autoPostclose && deps.active() && postcloseDue(now, s.postcloseTime, meta.lastPostcloseYmd)) {
        if (deps.status()?.running) {
          push({ lastPostcloseMsg: `${hm()} 장 마감 작업 대기 - r003 실행 중 (종료 후 시작)` });
        } else {
          const args = ['job', 'start', 'postclose', '--date', today, '--trading-day-only', '--scheduled', ...(s.postcloseEdgeAi ? ['--edge-ai'] : []), ...(s.postcloseEmail ? ['--email'] : [])];
          const r = await deps.client().run<{ ok: boolean; skipped?: boolean; reason?: string; error?: string }>(args, { timeoutMs: 60000 });
          if (r.ok) {
            deps.setMeta({ lastPostcloseYmd: today });
            push({ lastPostcloseMsg: `${hm()} ${r.data?.skipped ? `장 마감 작업 건너뜀 - ${r.data.reason}` : '장 마감 작업 시작 (수집 -> 이력 -> 선별 -> r004 반영)'}` });
            if (r.data?.skipped) deps.setMeta({ lastPostcloseSyncYmd: today });
          } else push({ lastPostcloseMsg: `${hm()} 장 마감 작업 시작 실패: ${r.error} (다음 분에 다시 시도)` });
        }
      }
      // the copy belongs to the job's day (meta.lastPostcloseYmd = its --date), not to "today": a job that ends after midnight, or a
      // Pi that is back only after midnight, still gets that day's watchlist
      const jobDay = meta.lastPostcloseYmd;
      if (
        s.syncWatchlistToPeer &&
        deps.active() &&
        jobDay &&
        meta.lastPostcloseSyncYmd !== jobDay &&
        (!lastSyncTryAt || Date.now() - lastSyncTryAt >= SYNC_RETRY_MS)
      ) {
        const r = await deps.client().run<JobList>(['job', 'list']);
        const last = r.data?.jobs.find((j) => j.name === 'postclose')?.last;
        const finished = last && last.args?.date === jobDay && last.state !== 'running';
        if (finished) {
          if (last.state === 'done' || last.state === 'done_with_errors') {
            lastSyncTryAt = Date.now();
            const sr = await deps.syncToPeer();
            // done only once the Pi has it - a Pi that is off / a dropped link is retried every 10 min (also after an app restart)
            if (sr.ok) deps.setMeta({ lastPostcloseSyncYmd: jobDay });
            push({
              lastSyncMsg: `${hm()} ${sr.ok ? `라즈베리파이로 감시목록 ${sr.count}종목 보냄` : `라즈베리파이 감시목록 보내기 실패: ${sr.error} (10분 뒤 다시 시도)`}`,
            });
          } else {
            deps.setMeta({ lastPostcloseSyncYmd: jobDay });
            push({ lastSyncMsg: `${hm()} 장 마감 작업이 ${last.state} - 감시목록을 보내지 않았습니다` });
          }
        }
      }
    } catch (e) {
      push({ lastPostcloseMsg: `${hm()} 스케줄러 오류: ${errMsg(e)}` });
    } finally {
      busy = false;
    }
  }

  return {
    start(): void {
      if (timer) return;
      timer = setInterval(() => void tick(), TICK_MS);
      setTimeout(() => void tick(), 3000);
    },
    stop(): void {
      if (timer) clearInterval(timer);
      timer = null;
      push({ enabled: false });
    },
    state: () => state,
    tick,
  };
}
