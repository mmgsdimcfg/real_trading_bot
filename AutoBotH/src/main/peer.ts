/*
 * Update log
 * - [2026-11-10] type=feat - The other host (one KIS account, two places that can run r003): in local mode the Raspberry Pi over SSH
 *   (connected on demand), in remote mode this PC (when the local mode is set up). peer:* IPC - status, role get/set, stop its
 *   r003, doctor, copy this host's watchlist there, one-time KIS config import (Pi ~/KIS/config/kis_devlp.yaml -> this PC's
 *   %USERPROFILE%\KIS\config, never overwritten without asking). guardStart() runs before every r003 start/restart: the other
 *   host must not be trading (refuse), is switched to standby (its r003 then refuses to start, its g022 cron opens no websocket),
 *   and this host is switched to primary. An unreachable other host needs the user's explicit confirmation
 * - [2026-11-10] type=fix - Codex review: guardStart works on the client pinned by the caller's operation; any failure while asking
 *   the other host (timeout, dropped link, local Python error) is "unknown" -> confirmation; broken local settings count as unknown
 *   once the local mode was ever used (a detached local r003 may still trade); this host is always written primary explicitly;
 *   switching the other host to primary first makes this host standby (roles stay exclusive, undone on failure)
 */

import { ipcMain, type BrowserWindow } from 'electron';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import type { RunMode } from '../shared/types';
import {
  peerDecision,
  watchlistStdin,
  type DoctorReport,
  type R016Status,
  type RoleInfo,
  type WatchlistDoc,
  type WatchlistSetResult,
} from '../shared/control';
import { confirmBox, R016_SERVICE_TIMEOUT, type R016Client } from './control';
import type { SshConnection } from './ssh';

export interface PeerDeps {
  mode: () => RunMode;
  ssh: SshConnection;
  connectSsh: () => Promise<void>;
  /** remote host configured (SSH host set) */
  remoteConfigured: () => boolean;
  /** '' when the local mode can run here */
  localProblem: () => string;
  remoteClient: R016Client;
  localClient: R016Client;
  /** the local mode connected successfully at least once on this PC (a local r003 may exist even if the settings broke since) */
  localEverUsed: () => boolean;
  win: () => BrowserWindow | null;
  log: (msg: string) => void;
}

const errMsg = (e: unknown) => String((e as Error)?.message || e).split('\n')[0];
const isOldServer = (s: string | undefined) => /invalid choice/.test(s || '');

export function createPeer(deps: PeerDeps) {
  /** role switches in progress (the run mode must not change under them) */
  let busyCount = 0;
  const peerKind = (): RunMode => (deps.mode() === 'local' ? 'remote' : 'local');
  const peerLabel = () => (peerKind() === 'remote' ? '라즈베리파이' : '이 PC (로컬)');
  const selfClient = () => (deps.mode() === 'local' ? deps.localClient : deps.remoteClient);

  /** client of the other host, connecting SSH when needed. null + reason when it is not set up / not reachable. */
  async function peerClient(): Promise<{ client: R016Client | null; reason?: string; notConfigured?: boolean }> {
    if (peerKind() === 'remote') {
      if (!deps.remoteConfigured()) return { client: null, reason: 'SSH 설정이 없습니다', notConfigured: true };
      if (!deps.ssh.connected) {
        try {
          await deps.connectSsh();
        } catch (e) {
          return { client: null, reason: `SSH 연결 실패: ${errMsg(e)}` };
        }
      }
      return { client: deps.remoteClient };
    }
    const p = deps.localProblem();
    if (p) return { client: null, reason: p, notConfigured: true };
    return { client: deps.localClient };
  }

  /** never throws: a dropped SSH link / timeout / local Python failure is "unknown" (the guard then asks the user) */
  async function peerStatus(): Promise<{ ok: boolean; data?: R016Status; error?: string; notConfigured?: boolean }> {
    try {
      const pc = await peerClient();
      if (!pc.client) return { ok: false, error: pc.reason, notConfigured: pc.notConfigured };
      const r = await pc.client.run<R016Status>(['status']);
      return r.ok && r.data ? { ok: true, data: r.data } : { ok: false, error: r.error };
    } catch (e) {
      return { ok: false, error: errMsg(e) };
    }
  }

  async function setRoleOn(client: R016Client, role: 'primary' | 'standby', label: string): Promise<{ ok: boolean; error?: string; oldServer?: boolean; data?: RoleInfo | null }> {
    try {
      const r = await client.run<RoleInfo>(['role', 'set', role, '--note', `AutoBotH (${deps.mode() === 'local' ? '노트북 메인' : '라즈베리파이 메인'})`]);
      if (!r.ok && isOldServer(`${r.error} ${r.stderr}`)) {
        return { ok: false, oldServer: true, error: `${label}의 코드가 이전 버전이라 role 명령이 없습니다 (git pull 필요).` };
      }
      return { ok: r.ok, error: r.error, data: r.data };
    } catch (e) {
      return { ok: false, error: errMsg(e) };
    }
  }

  async function setPeerRole(role: 'primary' | 'standby'): Promise<{ ok: boolean; error?: string; oldServer?: boolean; data?: RoleInfo | null }> {
    const pc = await peerClient();
    if (!pc.client) return { ok: false, error: pc.reason };
    return setRoleOn(pc.client, role, peerLabel());
  }

  const confirm = (title: string, message: string, detail: string, ok: string) => confirmBox(deps.win(), title, message, detail, ok);

  /**
   * Before start/restart of r003 on the connected host (see header). `self` is the client of the host being started - pinned by
   * the caller for the whole operation, so a run-mode switch in between cannot redirect it.
   */
  async function guardStart(self: R016Client): Promise<{ ok: boolean; error?: string; cancelled?: boolean }> {
    const label = peerLabel();
    const ps = await peerStatus();
    // remote-only users (the local mode never used on this PC) keep the old behaviour - this PC cannot have an r003 started by
    // the app. Once it was used, broken local settings are NOT proof that no local r003 runs (it is detached) -> ask.
    if (!ps.ok && ps.notConfigured && peerKind() === 'local' && !deps.localEverUsed()) return selfPrimary(self);
    const d = peerDecision(ps.ok ? ps.data! : null, ps.error, label);
    if (d.action === 'refuse') return { ok: false, error: d.reason };
    if (d.action === 'ask') {
      const go = await confirm(
        'r003 실행 - 다른 호스트 확인',
        d.reason,
        `같은 계좌를 ${label}와 동시에 매매하면 주문이 두 번 나갈 수 있습니다. ${label}에서 r003이 돌고 있지 않음을 직접 확인했다면 계속하세요.`,
        '확인했고 계속',
      );
      if (!go) return { ok: false, cancelled: true };
    } else if (d.action === 'standby') {
      const r = await setPeerRole('standby');
      if (!r.ok) {
        if (r.oldServer) {
          const go = await confirm('r003 실행 - standby 설정 불가', r.error || '', `${label}에서 r003과 틱 수집기(g022 cron)가 돌지 않게 직접 막아야 합니다. 계속할까요?`, '계속');
          if (!go) return { ok: false, cancelled: true };
        } else return { ok: false, error: `${label}를 standby 로 바꾸지 못했습니다: ${r.error}` };
      } else deps.log(`${label}를 standby 로 바꿨습니다.`);
    }
    return selfPrimary(self);
  }

  /** this host: -> primary, written explicitly (also when there was no role file - the scheduler's collector needs it) */
  async function selfPrimary(self: R016Client): Promise<{ ok: boolean; error?: string }> {
    try {
      const r = await self.run<RoleInfo>(['role', 'get']);
      if (!r.ok) return isOldServer(`${r.error} ${r.stderr}`) ? { ok: true } : { ok: false, error: `역할 확인 실패: ${r.error}` };
      if (r.data?.role === 'primary' && r.data.set) return { ok: true };
      const s = await self.run<RoleInfo>(['role', 'set', 'primary', '--note', 'AutoBotH start']);
      return s.ok ? { ok: true } : { ok: false, error: `이 호스트를 primary 로 바꾸지 못했습니다: ${s.error}` };
    } catch (e) {
      return { ok: false, error: `역할 확인 실패: ${errMsg(e)}` };
    }
  }

  /** this host's watchlist -> the other host (r016 watchlist set there, with its backup) */
  async function syncWatchlist(): Promise<{ ok: boolean; error?: string; count?: number; data?: WatchlistSetResult | null }> {
    const src = await selfClient().run<WatchlistDoc>(['watchlist', 'get']);
    if (!src.ok || !src.data) return { ok: false, error: `감시목록 읽기 실패: ${src.error}` };
    const rows = src.data.codes || [];
    if (!rows.length) return { ok: false, error: '감시목록이 비어 있습니다.' };
    const pc = await peerClient();
    if (!pc.client) return { ok: false, error: pc.reason };
    const r = await pc.client.run<WatchlistSetResult>(['watchlist', 'set', '--stdin'], { stdin: watchlistStdin(rows) });
    return { ok: r.ok, error: r.error, count: rows.length, data: r.data };
  }

  /** Raspberry Pi ~/KIS/config/kis_devlp.yaml -> %USERPROFILE%\KIS\config\kis_devlp.yaml (local mode setup) */
  async function importKisConfig(): Promise<{ ok: boolean; error?: string; cancelled?: boolean; path?: string }> {
    if (!deps.remoteConfigured()) return { ok: false, error: 'SSH 설정이 없습니다.' };
    if (!deps.ssh.connected) await deps.connectSsh();
    const home = await deps.ssh.home();
    const buf = await deps.ssh.readSmallFile(`${home}/KIS/config/kis_devlp.yaml`, 256 * 1024);
    if (!buf) return { ok: false, error: `라즈베리파이에 ${home}/KIS/config/kis_devlp.yaml 이 없습니다.` };
    const dir = join(homedir(), 'KIS', 'config');
    const dest = join(dir, 'kis_devlp.yaml');
    const detail = `앱키·시크릿·계좌번호가 들어 있는 파일입니다. ${dest} 에 저장되며 이 Windows 사용자만 쓰는 폴더입니다.`;
    if (existsSync(dest)) {
      if (!(await confirm('KIS 설정 가져오기', `${dest} 가 이미 있습니다. 라즈베리파이의 파일로 덮어쓸까요?`, detail, '덮어쓰기'))) return { ok: false, cancelled: true };
    } else if (!(await confirm('KIS 설정 가져오기', '라즈베리파이의 KIS 설정(kis_devlp.yaml)을 이 PC로 복사합니다.', detail, '복사'))) {
      return { ok: false, cancelled: true };
    }
    mkdirSync(dir, { recursive: true });
    writeFileSync(dest, buf);
    return { ok: true, path: dest };
  }

  const wrap = (fn: (...a: never[]) => Promise<unknown>) => async (_e: unknown, ...a: unknown[]) => {
    try {
      return await (fn as (...x: unknown[]) => Promise<unknown>)(...a);
    } catch (e) {
      return { ok: false, error: errMsg(e) };
    }
  };
  ipcMain.handle('peer:info', () => ({ kind: peerKind(), label: peerLabel() }));
  ipcMain.handle('peer:status', wrap(peerStatus));
  ipcMain.handle(
    'peer:role',
    wrap(async (role?: 'primary' | 'standby') => {
      if (role) {
        if (role === 'primary') {
          if (busyCount) return { ok: false, error: '다른 역할 변경이 진행 중입니다.' };
          const go = await confirm(
            '역할 변경',
            `${peerLabel()}를 primary 로, 이 호스트를 standby 로 바꿉니다.`,
            'primary 호스트는 r003을 실행할 수 있고 틱 수집기를 띄웁니다. 두 호스트가 동시에 primary 가 되지 않도록 이 호스트를 먼저 standby 로 바꿉니다 (이 호스트의 r003이 실행 중이면 거부).',
            '변경',
          );
          if (!go) return { ok: false, cancelled: true };
          busyCount++;
          try {
            // both hosts pinned for the whole switch (connect refuses to change the run mode while busy())
            const self = selfClient();
            const label = peerLabel();
            const pc = await peerClient();
            if (!pc.client) return { ok: false, error: pc.reason };
            const peer = pc.client;
            // roles stay exclusive: this host first (r016 refuses while its r003 runs), then the other one
            const s = await setRoleOn(self, 'standby', '이 호스트');
            if (!s.ok && !s.oldServer) return { ok: false, error: `이 호스트를 standby 로 바꾸지 못했습니다: ${s.error}` };
            const p = await setRoleOn(peer, 'primary', label);
            if (p.ok) return p;
            // the answer may be lost although the switch happened: undo only when the other host is confirmed NOT primary
            let peerRole: string | undefined;
            try {
              const g = await peer.run<RoleInfo>(['role', 'get']);
              peerRole = g.ok ? g.data?.role : undefined;
            } catch {
              peerRole = undefined;
            }
            if (peerRole === 'primary') return { ok: true, data: null };
            if (peerRole === 'standby' && s.ok) await setRoleOn(self, 'primary', '이 호스트');
            else if (peerRole === undefined) {
              return { ok: false, error: `${label} 결과를 확인하지 못했습니다 (${p.error}). 이 호스트는 standby 로 두었습니다 - ${label} 역할을 확인하세요.` };
            }
            return p;
          } finally {
            busyCount--;
          }
        }
        return setPeerRole(role);
      }
      const pc = await peerClient();
      if (!pc.client) return { ok: false, error: pc.reason };
      const r = await pc.client.run<RoleInfo>(['role', 'get']);
      return { ok: r.ok, error: r.error, data: r.data };
    }),
  );
  ipcMain.handle(
    'peer:stop',
    wrap(async () => {
      const pc = await peerClient();
      if (!pc.client) return { ok: false, error: pc.reason };
      if (!(await confirm('r003 정지', `${peerLabel()}의 r003을 정지합니다.`, '장중이면 정지된 동안 그쪽 보유 종목의 매도 조건 평가가 중단됩니다.', '정지'))) {
        return { ok: false, cancelled: true };
      }
      const r = await pc.client.run(['stop'], { timeoutMs: R016_SERVICE_TIMEOUT });
      return { ok: r.ok, error: r.error, data: r.data };
    }),
  );
  ipcMain.handle(
    'peer:doctor',
    wrap(async () => {
      const pc = await peerClient();
      if (!pc.client) return { ok: false, error: pc.reason };
      const r = await pc.client.run<DoctorReport>(['doctor']);
      return { ok: !!r.data, error: r.data ? undefined : r.error, data: r.data };
    }),
  );
  ipcMain.handle(
    'peer:syncWatchlist',
    wrap(async () => {
      if (!(await confirm('감시목록 보내기', `이 호스트의 감시목록을 ${peerLabel()}로 복사합니다.`, `${peerLabel()}의 기존 파일은 watchlist_backup/에 백업됩니다.`, '보내기'))) {
        return { ok: false, cancelled: true };
      }
      return syncWatchlist();
    }),
  );
  ipcMain.handle('peer:importKis', wrap(importKisConfig));

  return { guardStart, peerStatus, syncWatchlist, peerLabel, busy: () => busyCount > 0 };
}
