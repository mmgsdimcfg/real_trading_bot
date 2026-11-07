/*
 * Update log
 * - [2026-11-05] type=feat - Control tab (guide 2.1): r003 service state (running/PID/start time/restarts/mode/profile), risk
 *   profile + dry-run choice, 실행/정지/재시작 with the guide's enable rules (start = not running and no external r003), a spinner
 *   while the main process waits for the unit to settle, the external-r003 warning, pending-restart items (overrides with
 *   [기본값으로], env, watchlist), override errors, and the last 20 history rows (`history -n 20`). Confirmations (live orders,
 *   market hours) are shown by the main process. Server times are shown as the server's KST strings, unconverted
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import {
  botSummary,
  fmtValue,
  historyText,
  pctHint,
  pendingCount,
  RISK_PROFILES,
  serviceButtons,
  type HistoryRow,
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
};

export function ControlView() {
  const control = useStore((s) => s.control);
  const connected = useStore((s) => s.status.conn === 'connected');
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
        <div className="empty">서버에 연결되어 있지 않습니다 - 설정/연결 탭에서 연결하세요.</div>
      </div>
    );
  }

  return (
    <div className="control">
      <section>
        <h3>봇 상태 (r003 · {svc?.unit || 'autobot-r003.service'})</h3>
        <div className="botline">
          <span className={`bot bot-${summary}`}>
            <i className="dot" />
            {BOT_LABEL[summary]}
          </span>
          <span>PID {svc?.main_pid ?? '-'}</span>
          <span>시작 {isoShow(svc?.started_at)}</span>
          <span>재시작 {svc?.restarts ?? 0}회</span>
          <span>
            모드 <b className={st?.dry_run ? '' : 'live-mode'}>{st ? (st.dry_run ? 'dry-run' : '실주문') : '-'}</b>
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
            실행
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
              <b>외부 r003 감지</b> - 서버의 screen 세션에서 r003을 종료한 뒤 앱에서 실행하세요.
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
          서비스는 서버 부팅 시 자동으로 시작하지 않습니다. 서버를 재부팅한 뒤에는 여기서 [실행]을 눌러야 합니다. 상태 기준 시각 {isoShow(st?.generated_at)} (서버 KST)
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
