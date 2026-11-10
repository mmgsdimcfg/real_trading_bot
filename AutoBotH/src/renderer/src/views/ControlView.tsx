/*
 * Update log
 * - [2026-11-05] type=feat - Control tab (guide 2.1): r003 service state (running/PID/start time/restarts/mode/profile), risk
 *   profile + dry-run choice, 실행/정지/재시작 with the guide's enable rules (start = not running and no external r003), a spinner
 *   while the main process waits for the unit to settle, the external-r003 warning, pending-restart items (overrides with
 *   [기본값으로], env, watchlist), override errors, and the last 20 history rows (`history -n 20`). Confirmations (live orders,
 *   market hours) are shown by the main process. Server times are shown as the server's KST strings, unconverted
 * - [2026-11-07] type=fix - Mode label: status.dry_run is null while the bot is stopped (no running snapshot); it was shown as a red
 *   '실주문'. Now '-' when stopped, '확인 불가' when running without a snapshot, dry-run/실주문 only from the snapshot
 * - [2026-11-10] type=feat - Local mode: host line (hostname, systemd / r020 supervisor, host role primary|standby with a switch,
 *   tick collector state + 시작), PeerPanel for the other host (status, its r003 stop, standby/primary, watchlist copy), local
 *   wording for the external-r003 warning and the restart note
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import type { RunMode } from '../../../shared/types';
import {
  botSummary,
  fmtValue,
  historyText,
  pctHint,
  pendingCount,
  RISK_PROFILES,
  serviceButtons,
  type HistoryRow,
  type R016Status,
  type RiskProfile,
  type SchemaKey,
} from '../../../shared/control';

const isoShow = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(0, 19) : '-');

export const BOT_LABEL: Record<string, string> = {
  unknown: '상태 모름',
  running: '실행 중',
  starting: '시작 중',
  stopping: '정지 중',
  stopped: '정지',
  failed: '실패(정지)',
  external: '서비스 밖에서 실행 중',
};

const BUSY_LABEL: Record<string, string> = {
  start: '실행 중… 서비스가 올라올 때까지 확인합니다',
  stop: '정지 중… r003이 상태를 저장하고 내려갈 때까지 최대 90초 걸릴 수 있습니다',
  restart: '재시작 중… 정지(최대 90초) 후 다시 시작합니다',
};

const ACTION_LABEL: Record<string, string> = {
  set: '설정',
  unset: '기본값',
  reset: '전체 되돌림',
  env: '실행 환경',
  watchlist: '감시목록',
  'service.start': '실행',
  'service.stop': '정지',
  'service.restart': '재시작',
  role: '호스트 역할',
  'collector.start': '틱 수집기 시작',
  'collector.restart': '틱 수집기 재시작',
};

const roleLabel = (r: string | undefined) => (r === 'standby' ? 'standby (매매 안 함)' : 'primary (매매 호스트)');

const HOST_NAME: Record<RunMode, string> = { local: '이 PC', remote: '라즈베리파이' };

/**
 * [2026-11-11] 매매 실행 위치: both hosts as two rows (role, r003, tick collector). The selected row is the host the [실행] button
 * starts r003 on (= settings.runMode / the open connection). "여기서 실행" on the other row saves runMode and reconnects (what
 * 설정/연결 > 실행 위치 > [저장 후 연결] did). Role switches, collector start, the other host's r003 stop and the watchlist copy
 * stay available under "고급".
 */
function HostPanel() {
  const control = useStore((s) => s.control);
  const status = useStore((s) => s.status);
  const settings = useStore((s) => s.settings);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const connected = status.conn === 'connected';
  const mode: RunMode = (connected && status.mode) || settings?.runMode || 'remote';
  const peerKind: RunMode = mode === 'local' ? 'remote' : 'local';
  const self = connected ? control.status : null;
  const [peer, setPeer] = useState<R016Status | null>(null);
  const [peerErr, setPeerErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [msg, setMsg] = useState('');
  const ctlBusy = !!control.busy;

  const loadPeer = async () => {
    setBusy(true);
    setPeerErr('');
    const r = await window.api.peerStatus();
    setBusy(false);
    if (r.ok && r.data) setPeer(r.data);
    else {
      setPeer(null);
      setPeerErr(r.notConfigured ? `설정되지 않음 (${r.error})` : `상태 확인 실패: ${r.error}`);
    }
  };
  useEffect(() => {
    if (connected) void loadPeer();
    else {
      setPeer(null);
      setPeerErr('');
    }
  }, [connected, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  /** make `kind` the host that [실행] starts r003 on: save runMode, then connect (the view, the control and the scheduler follow) */
  const switchTo = async (kind: RunMode) => {
    if (!settings || switching) return;
    setSwitching(true);
    setMsg(kind === mode ? `${HOST_NAME[kind]}에 연결 중…` : `실행 위치를 ${HOST_NAME[kind]}로 바꾸는 중…`);
    const s = await window.api.saveSettings({ ...settings, runMode: kind });
    if (!s.ok) {
      setMsg(`실패: ${s.error}`);
      setSwitching(false);
      return;
    }
    useStore.setState({ settings: s.settings });
    const r = await window.api.connect();
    setSwitching(false);
    setMsg(r.ok ? `이제 [실행]은 ${HOST_NAME[kind]}에서 r003을 시작합니다.` : `${HOST_NAME[kind]} 연결 실패: ${r.error} - 설정/연결 탭에서 ${HOST_NAME[kind]} 연결 정보를 확인하세요.`);
  };

  const act = async (fn: () => Promise<{ ok: boolean; error?: string; cancelled?: boolean }>, okMsg: string, reloadPeer = true) => {
    setBusy(true);
    setMsg('');
    const r = await fn();
    setBusy(false);
    if (r.cancelled) return;
    setMsg(r.ok ? okMsg : `실패: ${r.error}`);
    bumpHistory();
    if (reloadPeer) await loadPeer();
  };
  const startCollector = async () => {
    setMsg('틱 수집기 시작 중…');
    const res = await window.api.ctlCollectorStart(false);
    const d = res.data as { already_running?: boolean; pid?: number } | undefined;
    setMsg(res.ok ? (d?.already_running ? `이미 실행 중 (pid ${d.pid})` : `시작했습니다 (pid ${d?.pid})`) : `실패: ${res.error}`);
  };

  const selfRole = self?.host?.role?.role;
  const peerRole = peer?.host?.role?.role;
  const peerName = HOST_NAME[peerKind];
  const anyBusy = busy || switching || ctlBusy;

  const row = (kind: RunMode) => {
    const selected = kind === mode;
    const st = selected ? self : peer;
    const role = st?.host?.role?.role;
    const s = botSummary(st);
    const hostText = st?.host ? `${st.host.hostname} (${st.host.platform})` : kind === 'local' ? 'Windows' : 'Linux · SSH';
    return (
      <tr key={kind} className={`nohover${selected ? ' sel' : ''}`}>
        <td className="host-pick">
          {selected ? (
            <b className="pill live" title="[실행]을 누르면 이 호스트에서 r003이 시작됩니다">
              실행 대상
            </b>
          ) : (
            <button className="mini" disabled={anyBusy} title={`실행 위치를 ${HOST_NAME[kind]}로 바꿉니다 (설정 저장 후 다시 연결)`} onClick={() => void switchTo(kind)}>
              여기서 실행
            </button>
          )}
        </td>
        <td>
          <b>{HOST_NAME[kind]}</b> <span className="muted small">{hostText}</span>
        </td>
        <td>
          {st?.host ? (
            <span className={role === 'standby' ? 'pill pend' : 'pill live'} title={st.host.role?.set ? `${st.host.role.by || ''} ${st.host.role.at || ''} ${st.host.role.note || ''}` : '역할 파일 없음 = primary'}>
              {roleLabel(role)}
            </span>
          ) : st ? (
            <span className="muted small">이전 버전 서버 (git pull 필요)</span>
          ) : (
            '-'
          )}
        </td>
        <td>
          {st ? (
            <span className={`bot bot-${s}`}>
              <i className="dot" />
              r003 {BOT_LABEL[s]}
            </span>
          ) : (
            <span className="muted small">{selected ? (connected ? '상태 확인 중…' : '연결 안 됨') : peerErr || '확인 안 됨'}</span>
          )}
        </td>
        <td className="muted small">{st ? `틱 수집기 ${st.collector?.running ? '실행 중' : '정지'}` : ''}</td>
      </tr>
    );
  };

  return (
    <section>
      <h3>매매 실행 위치</h3>
      <p className="muted small">
        r003(자동매매)은 계좌가 하나라 <b>한 곳에서만</b> 돌립니다. 아래에서 "실행 대상"인 호스트에서 [실행]이 r003을 시작합니다. 다른 호스트로 바꾸려면 그 줄의 [여기서 실행]을
        누르세요.
      </p>
      <table className="grid compact host-table">
        <tbody>
          {row('local')}
          {row('remote')}
        </tbody>
      </table>
      <div className="actions">
        {!connected && (
          <button className="primary" disabled={switching} onClick={() => void switchTo(mode)}>
            {HOST_NAME[mode]}에 연결
          </button>
        )}
        <button disabled={anyBusy || !connected} onClick={() => void loadPeer()}>
          {peerName} 상태 확인
        </button>
        {(busy || switching) && <i className="spinner" />}
        {msg && <span className={/실패/.test(msg) ? 'err-text small' : 'muted small'}>{msg}</span>}
      </div>
      <details className="adv">
        <summary>고급: 역할 바꾸기 · 틱 수집기 · {peerName} r003 정지 · 감시목록 복사</summary>
        <p className="muted small">
          [실행]은 시작 전에 {peerName}를 확인해 r003이 돌고 있으면 거부하고, primary 이면 standby 로 바꾼 뒤 {HOST_NAME[mode]}를 primary 로 해서 시작합니다 (standby 호스트는
          r003 시작을 거부하고 틱 수집기 cron 을 띄우지 않습니다). 아래 버튼은 그 역할을 손으로 바꿀 때만 씁니다.
        </p>
        <div className="actions">
          <span className="muted small">{HOST_NAME[mode]}:</span>
          {selfRole === 'standby' ? (
            <button className="mini" disabled={anyBusy || !connected} onClick={() => void act(() => window.api.ctlRole('primary'), `${HOST_NAME[mode]}를 primary 로 바꿨습니다.`, false)}>
              primary 로
            </button>
          ) : (
            <button
              className="mini"
              disabled={anyBusy || !connected || !self?.host || !!self?.running}
              title={self?.running ? 'r003 실행 중에는 standby 로 바꿀 수 없습니다' : ''}
              onClick={() => void act(() => window.api.ctlRole('standby'), `${HOST_NAME[mode]}를 standby 로 바꿨습니다.`, false)}
            >
              standby 로
            </button>
          )}
          {mode === 'local' && !self?.collector?.running && (
            <button className="mini" disabled={anyBusy || !connected || selfRole === 'standby'} onClick={() => void startCollector()}>
              틱 수집기 시작
            </button>
          )}
          <span className="muted small">{peerName}:</span>
          <button className="mini" disabled={anyBusy || !peer?.running} onClick={() => void act(() => window.api.peerStop(), `${peerName} r003 을 정지했습니다.`)}>
            r003 정지
          </button>
          <button className="mini" disabled={anyBusy || !peer?.host || peerRole === 'standby' || !!peer?.running} onClick={() => void act(() => window.api.peerRole('standby'), `${peerName}를 standby 로 바꿨습니다.`)}>
            standby 로
          </button>
          <button className="mini" disabled={anyBusy || !peer?.host || peerRole !== 'standby'} onClick={() => void act(() => window.api.peerRole('primary'), `${peerName}를 primary 로 바꿨습니다.`)}>
            primary 로
          </button>
          <button className="mini" disabled={anyBusy || !peer} onClick={() => void act(() => window.api.peerSyncWatchlist(), `감시목록을 ${peerName}로 보냈습니다.`)}>
            감시목록 보내기 →
          </button>
        </div>
        <p className="muted small">
          노트북을 못 쓰는 날: 라즈베리파이 줄의 [여기서 실행]을 누른 뒤 [라즈베리파이에서 실행]. 노트북이 켜져 있으면 자동으로 standby 가 됩니다. 장 마감 후 자동 작업이 끝나면
          새 감시목록이 라즈베리파이로도 복사됩니다 (설정).
        </p>
      </details>
    </section>
  );
}

export function ControlView() {
  const control = useStore((s) => s.control);
  const connected = useStore((s) => s.status.conn === 'connected');
  const isLocal = useStore((s) => s.status.mode === 'local');
  const historyRev = useStore((s) => s.historyRev);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const st = control.status;
  const [mode, setMode] = useState<{ dryRun: boolean; profile: RiskProfile } | null>(null);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [histErr, setHistErr] = useState('');
  const [overrides, setOverrides] = useState<SchemaKey[]>([]);
  const [msg, setMsg] = useState('');

  // form follows the server until the user changes it
  const dryRun = mode?.dryRun ?? st?.dry_run ?? false;
  const profile = mode?.profile ?? ((st?.risk_profile as RiskProfile) || 'neutral');
  const busy = !!control.busy;
  const btn = serviceButtons(st, busy || !connected);
  const summary = botSummary(st);
  const svc = st?.service;
  const nPending = pendingCount(st);

  useEffect(() => {
    if (!connected) return;
    void window.api.ctlStatus();
  }, [connected]);

  useEffect(() => {
    if (!connected) return;
    let alive = true;
    void window.api.ctlHistory(20).then((r) => {
      if (!alive) return;
      if (r.ok && r.data) {
        setHistory([...r.data.rows].reverse());
        setHistErr('');
      } else setHistErr(r.error || '이력 조회 실패');
    });
    void window.api.ctlSchema(false).then((r) => {
      if (alive && r.ok && r.data) setOverrides(r.data.keys.filter((k) => k.has_override));
    });
    return () => {
      alive = false;
    };
  }, [connected, historyRev, st?.override_count, st?.pending_restart, st?.service?.started_at]);

  const run = async (verb: 'start' | 'stop' | 'restart') => {
    setMsg('');
    const r = await window.api.ctlService(verb, verb === 'stop' ? {} : { dryRun, profile });
    if (r.cancelled) return;
    setMsg(r.ok ? '' : `실패: ${r.error}`);
    if (r.ok) setMode(null);
    bumpHistory();
  };

  const unset = async (key: string) => {
    const r = await window.api.ctlUnset([key]);
    setMsg(r.ok ? `${key} 기본값으로 되돌림 - 재시작 후 반영` : `실패: ${r.error}`);
    bumpHistory();
  };

  const reset = async () => {
    const r = await window.api.ctlReset();
    if (r.cancelled) return;
    setMsg(r.ok ? '모든 오버라이드를 지웠습니다 - 재시작 후 반영' : `실패: ${r.error}`);
    bumpHistory();
  };

  const pendingRows = useMemo(() => {
    const rows: { k: string; text: string; key?: string; pending?: boolean }[] = [];
    for (const o of overrides) {
      const h = pctHint(o.key, o.override);
      rows.push({
        k: o.key,
        key: o.key,
        pending: o.pending,
        text: `${o.key} = ${fmtValue(o.override)}${h ? ` (${h.slice(2)})` : ''}  (기본 ${fmtValue(o.base)})`,
      });
    }
    if (st?.pending?.env?.changed_since_start) rows.push({ k: 'env', pending: true, text: `실행 환경(r003.env) 변경 ${isoShow(st.pending.env.mtime)}` });
    if (st?.pending?.watchlist?.changed_since_start) rows.push({ k: 'wl', pending: true, text: `감시목록 변경 ${isoShow(st.pending.watchlist.mtime)}` });
    return rows;
  }, [overrides, st]);

  if (!connected && !st) {
    return (
      <div className="control">
        <HostPanel />
        <div className="empty">연결되어 있지 않습니다 - 위에서 실행 위치를 고르고 [연결]을 누르세요 (연결 정보는 설정/연결 탭).</div>
      </div>
    );
  }

  return (
    <div className="control">
      <HostPanel />
      <section>
        <h3>
          봇 상태 (r003 · {isLocal ? '이 PC' : '라즈베리파이'} · {svc?.unit || 'autobot-r003.service'})
        </h3>
        <div className="botline">
          <span className={`bot bot-${summary}`}>
            <i className="dot" />
            {BOT_LABEL[summary]}
          </span>
          <span>PID {svc?.main_pid ?? '-'}</span>
          <span>시작 {isoShow(svc?.started_at)}</span>
          <span>재시작 {svc?.restarts ?? 0}회</span>
          <span>
            {/* dry_run is null while the bot is stopped (no running snapshot) - never show that as 실주문 */}
            모드{' '}
            {st?.dry_run === true ? (
              <b>dry-run</b>
            ) : st?.dry_run === false ? (
              <b className="live-mode">실주문</b>
            ) : (
              <b title={st?.running ? '실행 스냅샷이 없어 실행 모드를 확인할 수 없습니다 (재시작 후 정확)' : '정지 중 - 다음 실행 모드는 아래 dry-run 체크'}>
                {st?.running ? '확인 불가' : '-'}
              </b>
            )}
          </span>
          <span>프로파일 {st?.risk_profile || '-'}</span>
          {svc?.sub_state === 'auto-restart' && <span className="warn-text">크래시 후 재시작 대기 중</span>}
          {svc?.active_state === 'failed' && <span className="warn-text">마지막 실행 실패 (result={svc.result})</span>}
        </div>
        <div className="actions">
          <label>
            프로파일
            <select value={profile} disabled={busy} onChange={(e) => setMode({ dryRun, profile: e.target.value as RiskProfile })}>
              {RISK_PROFILES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={dryRun} disabled={busy} onChange={(e) => setMode({ dryRun: e.target.checked, profile })} /> dry-run (주문 안 냄)
          </label>
          {mode && (mode.dryRun !== st?.dry_run || mode.profile !== st?.risk_profile) && (
            <span className="muted small">다음 실행/재시작부터 적용 (r003.env에 저장)</span>
          )}
        </div>
        <div className="actions">
          <button className="primary" disabled={!btn.start} title={btn.startReason || ''} onClick={() => void run('start')}>
            {isLocal ? '이 PC' : '라즈베리파이'}에서 실행
          </button>
          <button disabled={!btn.stop} onClick={() => void run('stop')}>
            정지
          </button>
          <button disabled={!btn.restart} onClick={() => void run('restart')}>
            재시작
          </button>
          {nPending > 0 && st?.running && (
            <span className="badge-warn" title="서비스 시작 이후 바뀐 파일이 있습니다">
              ⚠ 미적용 변경 {nPending}건 → 재시작 필요
            </span>
          )}
          {busy && (
            <span className="busy">
              <i className="spinner" />
              {BUSY_LABEL[control.busy!] || `${control.busy} 진행 중…`}
            </span>
          )}
        </div>
        {msg && <p className="err-text">{msg}</p>}
        {control.lastAction && !busy && (
          <p className={control.lastAction.ok ? 'muted small' : 'err-text small'}>
            마지막 동작: {control.lastAction.message}
          </p>
        )}
        {control.error && <p className="err-text small">상태 조회 오류: {control.error}</p>}
        <div className={st?.external_r003?.length ? 'callout danger' : 'muted small'}>
          {st?.external_r003?.length ? (
            <>
              <b>외부 r003 감지</b> -{' '}
              {isLocal ? '감시 프로세스 밖에서 실행된 r003입니다 (콘솔 등). [정지]를 누르면 상태를 저장하고 끝내도록 요청합니다.' : '서버의 screen 세션에서 r003을 종료한 뒤 앱에서 실행하세요.'}
              <ul>
                {st.external_r003.map((p) => (
                  <li key={p.pid} className="mono">
                    PID {p.pid} · {p.cmdline}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            '외부 r003 감지: 없음'
          )}
        </div>
        {st && !st.running && st.override_count > 0 && (
          <p className="callout small">봇이 꺼져 있습니다. 저장된 오버라이드 {st.override_count}건은 다음 실행 때 적용됩니다.</p>
        )}
        <p className="muted small">
          {isLocal
            ? '이 PC의 r003은 앱과 별개로 실행되어 앱을 닫아도 계속됩니다 (PC 재부팅·절전 시 멈춤 - 재부팅 후에는 여기서 [실행]). 크래시하면 60초 뒤 다시 시작합니다 (10분에 3회까지).'
            : '서비스는 서버 부팅 시 자동으로 시작하지 않습니다. 서버를 재부팅한 뒤에는 여기서 [실행]을 눌러야 합니다.'}{' '}
          상태 기준 시각 {isoShow(st?.generated_at)} (KST)
        </p>
      </section>


      <section>
        <h3>미적용 변경 · 저장된 오버라이드 ({overrides.length})</h3>
        {(st?.override_errors?.length || 0) > 0 && (
          <div className="callout warn small">
            서버(r001)가 거부한 오버라이드 항목:
            <ul>
              {st!.override_errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          </div>
        )}
        {pendingRows.length ? (
          <table className="grid compact">
            <tbody>
              {pendingRows.map((r) => (
                <tr key={r.k} className="nohover">
                  <td>{r.pending ? <i className="pend-dot">미적용</i> : <i className="ovr-mark">{st?.running ? '◆ 적용 중' : '◆ 저장됨'}</i>}</td>
                  <td className="mono wrap">{r.text}</td>
                  <td>
                    {r.key && (
                      <button className="mini" disabled={busy || !connected} onClick={() => void unset(r.key!)}>
                        기본값으로
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">저장된 오버라이드나 재시작을 기다리는 변경이 없습니다.</p>
        )}
        <div className="actions">
          <button disabled={busy || !connected || !overrides.length} onClick={() => void reset()}>
            전체 기본값으로 되돌리기 (reset)
          </button>
          <span className="muted small">설정값은 파이프라인 탭에서 노드를 눌러 바꾸거나, 설정/연결 탭의 "고급: 전체 설정값"에서 바꿉니다.</span>
        </div>
      </section>

      <section>
        <h3>최근 변경 이력 (20건)</h3>
        {histErr && <p className="err-text small">{histErr}</p>}
        <div className="table-wrap short">
          <table className="grid compact">
            <thead>
              <tr>
                <th>시각 (KST)</th>
                <th>작성자</th>
                <th>동작</th>
                <th>내용</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={`${h.ts}-${i}`} className="nohover">
                  <td className="mono">{isoShow(h.ts)}</td>
                  <td>{h.by}</td>
                  <td>{ACTION_LABEL[h.action] || h.action}</td>
                  <td className="mono wrap">{historyText(h)}</td>
                </tr>
              ))}
              {!history.length && (
                <tr className="nohover">
                  <td colSpan={4} className="muted center">
                    이력 없음
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
