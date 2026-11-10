/*
 * Update log
 * - [2026-11-10] type=feat - 데이터 작업 tab (r016 `job` -> live_trading/r021_jobs.py on the connected host): post-close pipeline
 *   (collect -> daily history -> watchlist selection [-> Edge AI]) and each step alone, with the target date (default today),
 *   workers, start / stop, the last run's steps (rc, times) and a log tail refreshed while it runs. Only one job runs at a time
 *   (r021 lock); jobs are detached and keep running when the app closes. The scheduler line shows the local-mode automation
 *   (tick collector / post-close / watchlist copy to the Raspberry Pi)
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { JOB_STATE_LABEL, localYmd, type JobList, type JobLog, type JobRun } from '../../../shared/control';

const isoShow = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(5, 19) : '-');
const dur = (a: string | null | undefined, b: string | null | undefined) => {
  if (!a) return '';
  const t = (b ? Date.parse(b) : Date.now()) - Date.parse(a);
  if (!Number.isFinite(t) || t < 0) return '';
  const m = Math.floor(t / 60000);
  return m >= 60 ? `${Math.floor(m / 60)}시간 ${m % 60}분` : `${m}분 ${Math.floor((t % 60000) / 1000)}초`;
};

const DESC: Record<string, string> = {
  postclose: '장 마감 후 한 번에: 수집(g001) → 일봉 이력(g025) → 다음 거래일 감시 종목 선별(g002 → r004 감시목록). 수집이 실패하면 선별하지 않습니다.',
  collect: '분봉·일봉 데이터 수집 (KIS API, 전 종목 - 오래 걸림)',
  history: '종목별 일봉 이력 갱신 (차트 일/주/월봉)',
  scan: '수집된 데이터로 다음 거래일 감시 종목 선별 → r004 감시목록 저장',
  edge_ai: 'Edge AI 데이터셋 생성 → 학습 → 점검 (CPU 많이 씀)',
  universe: '수집 대상 종목 목록(g004) 갱신 - 신규상장/상장폐지/위험종목',
};

function stateClass(st: string | undefined) {
  if (st === 'running') return 'pill pend';
  if (st === 'done') return 'pill live';
  if (st === 'failed' || st === 'done_with_errors') return 'pill fail';
  return 'pill';
}

export function JobsView() {
  const connected = useStore((s) => s.status.conn === 'connected');
  const local = useStore((s) => s.status.mode === 'local');
  const sched = useStore((s) => s.sched);
  const settings = useStore((s) => s.settings);
  const setTab = useStore((s) => s.setTab);
  const [list, setList] = useState<JobList | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState('postclose');
  const [date, setDate] = useState(localYmd());
  const [workers, setWorkers] = useState('');
  const [edgeAi, setEdgeAi] = useState(false);
  const [log, setLog] = useState<JobLog | null>(null);
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);

  const running = list?.running?.name || null;
  const selRow = useMemo(() => list?.jobs.find((j) => j.name === sel), [list, sel]);
  const last: JobRun | null = selRow?.last || null;

  const load = async () => {
    const r = await window.api.ctlJobs();
    if (r.ok && r.data) {
      setList(r.data);
      setErr('');
    } else setErr(r.error || '작업 목록을 읽지 못했습니다');
  };
  const loadLog = async (name = sel) => {
    const r = await window.api.ctlJobLog(name, 300);
    if (r.ok && r.data) setLog(r.data);
  };

  // job list: every 5 s on this PC, 20 s on the Raspberry Pi (each call starts Python there); log while the selected job runs
  useEffect(() => {
    if (!connected) return;
    void load();
    const t = setInterval(() => void load(), local ? 5000 : 20000);
    return () => clearInterval(t);
  }, [connected, local]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!connected) return;
    void loadLog(sel);
    if (running !== sel) return;
    const t = setInterval(() => void loadLog(sel), local ? 3000 : 15000);
    return () => clearInterval(t);
  }, [connected, sel, running, local]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    if (!/^\d{8}$/.test(date)) return setMsg('날짜는 YYYYMMDD');
    setBusy(true);
    setMsg('시작 중…');
    const r = await window.api.ctlJobStart(sel, { date, workers: Number(workers) || undefined, edgeAi: sel === 'postclose' && edgeAi });
    setBusy(false);
    if (r.cancelled) return setMsg('');
    setMsg(r.ok ? '시작했습니다. 앱을 닫아도 작업은 계속됩니다.' : `실패: ${r.error}`);
    await load();
    await loadLog(sel);
  };
  const stop = async () => {
    setBusy(true);
    setMsg('정지 중… (지금 단계의 프로세스를 끝냅니다)');
    const r = await window.api.ctlJobStop(sel);
    setBusy(false);
    setMsg(r.ok ? '정지했습니다.' : `실패: ${r.error}`);
    await load();
  };

  if (!connected) return <div className="control"><div className="empty">연결되어 있지 않습니다 - 설정/연결 탭에서 연결하세요.</div></div>;

  return (
    <div className="control jobs">
      <section>
        <h3>데이터 작업 ({local ? '이 PC' : '라즈베리파이'})</h3>
        {err && <p className="err-text small">{err}</p>}
        <table className="grid compact">
          <thead>
            <tr>
              <th>작업</th>
              <th>마지막 실행</th>
              <th>상태</th>
              <th>기준일</th>
              <th>소요</th>
            </tr>
          </thead>
          <tbody>
            {(list?.jobs || []).map((j) => (
              <tr key={j.name} className={j.name === sel ? 'sel' : ''} onClick={() => setSel(j.name)}>
                <td>
                  <b>{j.title}</b>
                </td>
                <td className="mono">{isoShow(j.last?.started_at)}</td>
                <td>
                  <span className={stateClass(j.last?.state)}>{j.last ? JOB_STATE_LABEL[j.last.state] || j.last.state : '없음'}</span>
                </td>
                <td className="mono">{j.last?.args?.date || '-'}</td>
                <td>{j.last ? dur(j.last.started_at, j.last.finished_at) : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">한 번에 한 작업만 실행됩니다{running ? ` - 지금 실행 중: ${list?.jobs.find((j) => j.name === running)?.title || running}` : ''}.</p>
      </section>

      <section>
        <h3>{selRow?.title || sel}</h3>
        <p className="muted small">{DESC[sel]}</p>
        <div className="actions">
          <label>
            기준일 <input className="mono" style={{ width: 96 }} value={date} onChange={(e) => setDate(e.target.value.trim())} placeholder="YYYYMMDD" />
          </label>
          <button className="mini" onClick={() => setDate(localYmd())}>
            오늘
          </button>
          {(sel === 'collect' || sel === 'postclose' || sel === 'edge_ai') && (
            <label title="g001 수집 스레드 / Edge AI 프로세스 수 (비우면 기본값)">
              작업자 수 <input type="number" min={1} max={32} style={{ width: 60 }} value={workers} onChange={(e) => setWorkers(e.target.value)} placeholder="기본" />
            </label>
          )}
          {sel === 'postclose' && (
            <label className="check">
              <input type="checkbox" checked={edgeAi} onChange={(e) => setEdgeAi(e.target.checked)} /> Edge AI 재학습 포함
            </label>
          )}
          <button className="primary" disabled={busy || !!running} title={running ? '다른 작업이 실행 중입니다' : ''} onClick={() => void start()}>
            시작
          </button>
          <button disabled={busy || running !== sel} onClick={() => void stop()}>
            정지
          </button>
          {busy && <i className="spinner" />}
          <span className="muted small">{msg}</span>
        </div>
        {last && (
          <>
            {last.error && <p className={last.state === 'failed' ? 'err-text small' : 'muted small'}>{last.error}</p>}
            <table className="grid compact">
              <thead>
                <tr>
                  <th>단계</th>
                  <th>상태</th>
                  <th>rc</th>
                  <th>시작</th>
                  <th>소요</th>
                  <th>명령</th>
                </tr>
              </thead>
              <tbody>
                {last.steps.map((s, i) => (
                  <tr key={`${s.name}-${i}`} className="nohover">
                    <td>{s.name}</td>
                    <td>
                      <span className={stateClass(s.state)}>{JOB_STATE_LABEL[s.state] || s.state}</span>
                    </td>
                    <td className="mono">{s.rc ?? ''}</td>
                    <td className="mono">{isoShow(s.started_at)}</td>
                    <td>{dur(s.started_at, s.finished_at)}</td>
                    <td className="mono wrap small">{s.argv.slice(0, 1).map((a) => a.split(/[\\/]/).pop()).concat(s.argv.slice(1)).join(' ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {(sel === 'scan' || sel === 'postclose') && last.state.startsWith('done') && (
              <p className="small">
                새 감시목록은{' '}
                <button className="linklike" onClick={() => setTab('board')}>
                  종목 보드
                </button>
                의 감시목록 편집에서 확인·수정할 수 있습니다 (다음 r003 실행부터 적용).
              </p>
            )}
          </>
        )}
        <div className="log-box mono">
          {log?.lines?.length ? log.lines.map((l, i) => <div key={i}>{l}</div>) : <span className="muted">로그 없음</span>}
        </div>
        {log?.log && <p className="muted small mono">{log.log}</p>}
      </section>

      {local && (
        <section>
          <h3>자동 실행 (이 PC, 앱이 켜져 있을 때)</h3>
          <ul className="small">
            <li>
              틱 수집기: {settings?.autoCollector ? '평일 08:40~15:35 5분마다 확인해 없으면 시작' : '꺼짐'} {sched?.lastCollectorMsg && <span className="muted">- {sched.lastCollectorMsg}</span>}
            </li>
            <li>
              장 마감 작업: {settings?.autoPostclose ? `거래일 ${settings.postcloseTime} 이후 1회 (r003 종료 후)${settings.postcloseEdgeAi ? ' + Edge AI' : ''}` : '꺼짐'}{' '}
              {sched?.lastPostcloseMsg && <span className="muted">- {sched.lastPostcloseMsg}</span>}
            </li>
            <li>
              라즈베리파이로 감시목록 복사: {settings?.syncWatchlistToPeer ? '장 마감 작업 완료 후' : '꺼짐'} {sched?.lastSyncMsg && <span className="muted">- {sched.lastSyncMsg}</span>}
            </li>
          </ul>
          <p className="muted small">설정/연결 탭의 "실행 위치 / 자동 실행"에서 바꿉니다.</p>
        </section>
      )}
    </div>
  );
}
