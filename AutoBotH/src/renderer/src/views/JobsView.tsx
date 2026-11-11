/*
 * Update log
 * - [2026-11-10] type=feat - 데이터 작업 tab (r016 `job` -> live_trading/r021_jobs.py on the connected host): post-close pipeline
 *   (collect -> daily history -> watchlist selection [-> Edge AI]) and each step alone, with the target date (default today),
 *   workers, start / stop, the last run's steps (rc, times) and a log tail refreshed while it runs. Only one job runs at a time
 *   (r021 lock); jobs are detached and keep running when the app closes. The scheduler line shows the local-mode automation
 *   (tick collector / post-close / watchlist copy to the Raspberry Pi)
 * - [2026-11-10] type=feat - New layout: section title + muted subtitle, job name with its script code in a muted monospace,
 *   state as a dot + text, durations as m:ss, step commands shortened to the script code ("g001 --date ..." - full command in the
 *   tooltip), date / workers inputs with placeholders and 시작 / 정지 on the right; the date and step start time moved to tooltips
 * - [2026-11-10] type=feat - 기준일 is a date field (components/DatePicker): a click opens a small calendar, future days cannot be
 *   picked; the job still receives YYYYMMDD
 * - [2026-11-11] type=feat - "감시목록(r004) 갱신" checkbox for 장 마감 파이프라인 / 감시 종목 선별: on for today, off for a past date
 *   (auto until the user toggles it; follows the date again when the date changes). Off = r021 --keep-watchlist: the scan builds
 *   that day's picks files (what g013 reads for the Edge AI dataset) but the r004 watchlist stays, so a past-date backfill for
 *   Edge AI retraining does not change the stocks r003 watches. The last run shows whether it touched the watchlist
 * - [2026-11-11] type=feat - Scan-only flow: the 감시 종목 선별 / 장 마감 파이프라인 cards show what the date folder holds (r016 job data:
 *   10-second / daily files, the collected list, last file time) - 감시 종목 선별 cannot start on a date without collected data.
 *   For 감시 종목 선별 the "감시목록(r004) 갱신" box is off by default: g002 runs alone (--keep-watchlist), the picks of the date are
 *   listed under the run (name + code chips) with a "r004 감시목록에 적용" checkbox that applies them (r016 watchlist apply, after a
 *   confirmation); the box shows checked when r004 already holds exactly those codes
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { DatePicker } from '../components/DatePicker';
import { JOB_STATE_LABEL, localYmd, type JobDataInfo, type JobList, type JobLog, type JobRun } from '../../../shared/control';

const isoShow = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(5, 16) : '-');
const hhmm = (s: string | null | undefined) => (s ? s.slice(11, 16) : '');
const num = (n: number) => n.toLocaleString('ko-KR');
/** elapsed time as m:ss (h:mm:ss from one hour) */
const dur = (a: string | null | undefined, b: string | null | undefined) => {
  if (!a) return '';
  const t = (b ? Date.parse(b) : Date.now()) - Date.parse(a);
  if (!Number.isFinite(t) || t < 0) return '';
  const sec = Math.floor(t / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const ss = String(sec % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
};
/** "데이터 수집 (g001)" -> ["데이터 수집", "g001"] - the script codes are shown in a muted monospace after the name */
const splitTitle = (t: string): [string, string] => {
  const m = /^(.*?)\s*\(([^()]+)\)\s*$/.exec(t);
  if (!m) return [t, ''];
  // only script codes ("g001", "g013/g014/g024", "g002 -> r004") - a step list like "(수집 -> 이력 -> 선별)" is dropped
  return [m[1], /[a-z]\d{3}/i.test(m[2]) ? m[2] : ''];
};
/** short state words for this view ("완료 (일부 실패)" -> "일부 실패") */
const stateText = (st: string) => (st === 'done_with_errors' ? '일부 실패' : JOB_STATE_LABEL[st] || st);
/** step command without the folder and the long script name: "g001_data_collect_symbols_daily.py --date 1" -> "g001 --date 1" */
const shortCmd = (argv: string[]) => {
  const [script = '', ...args] = argv;
  const base = script.split(/[\\/]/).pop() || '';
  const code = /^([a-z]\d{3})_.*\.py$/i.exec(base);
  return [code ? code[1] : base, ...args].join(' ');
};

const DESC: Record<string, string> = {
  postclose: '수집 → 이력 → 다음 거래일 감시 종목 선별 [→ Edge AI] · 수집 실패 시 선별 생략 · 과거 날짜는 기본으로 감시목록을 바꾸지 않음',
  collect: '분봉·일봉 데이터 수집 · KIS API 전 종목 (오래 걸림)',
  history: '종목별 일봉 이력 갱신 · 차트 일/주/월봉',
  scan: '수집된 기준일 데이터로 감시 종목 선별(g002)만 실행 · 결과를 보고 아래 "r004 감시목록에 적용"을 체크하면 저장 ("감시목록 갱신"을 켜면 바로 저장)',
  edge_ai: 'Edge AI 데이터셋 생성 → 학습 → 점검 · CPU 많이 씀',
  universe: '수집 대상 종목 목록(g004) 갱신 · 신규상장/상장폐지/위험종목',
};

/** state as a dot + text (.st), colored by kind */
function stateClass(st: string | undefined) {
  if (st === 'running') return 'st st-run';
  if (st === 'done') return 'st st-ok';
  if (st === 'failed' || st === 'done_with_errors') return 'st st-fail';
  return 'st st-none';
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
  /** 감시목록(r004) 갱신: null = automatic (today -> on, past date -> off), the user's click fixes it until the date changes */
  const [wlChoice, setWlChoice] = useState<boolean | null>(null);
  const isToday = date === localYmd();
  const hasScan = sel === 'postclose' || sel === 'scan';
  // [2026-11-11] 감시 종목 선별 alone: picks first, apply afterwards (the checkbox under the result) - so off unless the user turns it on
  const wlUpdate = wlChoice ?? (sel === 'scan' ? false : isToday);
  useEffect(() => setWlChoice(null), [date, sel]);
  /** [2026-11-11] what the date folder holds (r016 job data) - for the scan / postclose cards */
  const [info, setInfo] = useState<JobDataInfo | null>(null);
  const infoOk = info && info.date === date ? info : null;
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
  const loadInfo = async (d = date) => {
    if (!/^\d{8}$/.test(d)) return;
    const r = await window.api.ctlJobData(d);
    if (r.ok && r.data) setInfo(r.data);
    else setInfo({ ok: false, date: d, dir: '', exists: false, scannable: false, collected: false, collected_count: 0, files_10s: 0, files_daily: 0, data_mtime: null, picks: null, picks_file: '', picks_mtime: null, scan_log_mtime: null, generated_at: '', error: r.error || '확인 실패' });
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
  // date folder info: on the scan / postclose cards, again when the date changes and when the selected job's run ends
  useEffect(() => {
    if (!connected || !hasScan) return;
    void loadInfo(date);
  }, [connected, hasScan, date, last?.finished_at, last?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const noData = sel === 'scan' && !!infoOk && !infoOk.scannable;
  const start = async () => {
    if (!/^\d{8}$/.test(date)) return setMsg('날짜는 YYYYMMDD');
    if (noData) return setMsg(`${date} 데이터가 없습니다 - 먼저 수집(g001) 또는 장 마감 파이프라인을 실행하세요.`);
    setBusy(true);
    setMsg('시작 중…');
    const r = await window.api.ctlJobStart(sel, {
      date,
      workers: Number(workers) || undefined,
      edgeAi: sel === 'postclose' && edgeAi,
      keepWatchlist: hasScan && !wlUpdate,
    });
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
  /** [2026-11-11] r004 <- the picks of the date (r016 watchlist apply; main asks first) */
  const apply = async () => {
    if (!infoOk?.picks?.length) return;
    setBusy(true);
    setMsg('감시목록에 적용 중…');
    const r = await window.api.ctlPicksApply(date, infoOk.picks.length);
    setBusy(false);
    if (r.cancelled) return setMsg('');
    setMsg(r.ok ? `r004 감시목록에 ${r.data?.count ?? infoOk.picks.length}종목을 적용했습니다 (r003 재시작 후 반영).` : `실패: ${r.error}`);
    await loadInfo(date);
  };
  const dataLine = () => {
    if (!hasScan) return null;
    if (!infoOk) return <span className="muted">기준일 데이터 확인 중…</span>;
    if (infoOk.error && !infoOk.ok) return <span className="err-text">기준일 데이터 확인 실패: {infoOk.error}</span>;
    if (!infoOk.exists || !infoOk.scannable) {
      return sel === 'scan' ? (
        <span className="warn-text">기준일 {date} 수집 데이터 없음 - 수집(g001) 또는 장 마감 파이프라인을 먼저 실행하세요</span>
      ) : (
        <span className="muted">기준일 {date} 수집 데이터 없음 - 수집부터 실행됩니다</span>
      );
    }
    return (
      <span className="muted" title={infoOk.dir}>
        기준일 {date} 데이터 있음 · 10초봉 {num(infoOk.files_10s)}종목 · 일봉 {num(infoOk.files_daily)}
        {infoOk.collected ? ` · 수집 목록 ${num(infoOk.collected_count)}종목` : ' · 수집 완료 표시 없음'}
        {infoOk.data_mtime ? ` · 마지막 파일 ${isoShow(infoOk.data_mtime)}` : ''}
        {infoOk.picks ? ` · 선별 결과 ${infoOk.picks.length}종목 (${hhmm(infoOk.picks_mtime)})` : ' · 선별 결과 없음'}
      </span>
    );
  };

  if (!connected) return <div className="control"><div className="empty">연결되어 있지 않습니다 - 설정/연결 탭에서 연결하세요.</div></div>;

  return (
    <div className="control jobs">
      <section>
        <h3>데이터 작업</h3>
        <p className="subtitle">
          {local ? '이 PC' : '라즈베리파이'}에서 실행 · 한 번에 한 작업만 실행됩니다
          {running ? ` · 지금 실행 중: ${splitTitle(list?.jobs.find((j) => j.name === running)?.title || running)[0]}` : ''}
        </p>
        {err && <p className="err-text small">{err}</p>}
        <table className="grid">
          <thead>
            <tr>
              <th>작업</th>
              <th>마지막 실행</th>
              <th>상태</th>
              <th>소요</th>
            </tr>
          </thead>
          <tbody>
            {(list?.jobs || []).map((j) => {
              const [name, code] = splitTitle(j.title);
              return (
                <tr key={j.name} className={j.name === sel ? 'sel' : ''} onClick={() => setSel(j.name)}>
                  <td>
                    <b>{name}</b> {code && <span className="code-tag">{code}</span>}
                  </td>
                  <td className="mono" title={j.last?.args?.date ? `기준일 ${j.last.args.date}` : ''}>
                    {isoShow(j.last?.started_at)}
                  </td>
                  <td>
                    <span className={stateClass(j.last?.state)}>{j.last ? stateText(j.last.state) : '없음'}</span>
                  </td>
                  <td className="mono">{j.last ? dur(j.last.started_at, j.last.finished_at) : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section>
        <h3>{splitTitle(selRow?.title || sel)[0]}</h3>
        <p className="subtitle">{DESC[sel]}</p>
        {hasScan && <p className="small data-line">{dataLine()}</p>}
        <div className="job-form">
          <DatePicker value={date} onChange={setDate} today={localYmd()} max={localYmd()} title="기준일 (누르면 달력, 기본은 오늘)" />
          {(sel === 'collect' || sel === 'postclose' || sel === 'edge_ai') && (
            <input
              type="number"
              min={1}
              max={32}
              style={{ width: 120 }}
              value={workers}
              onChange={(e) => setWorkers(e.target.value)}
              placeholder="작업자 수"
              title="g001 수집 스레드 / Edge AI 프로세스 수 (비우면 기본값)"
            />
          )}
          {sel === 'postclose' && (
            <label className="check" title="장 마감 파이프라인 끝에 Edge AI 재학습(g013/g014/g024)을 이어서 실행">
              <input type="checkbox" checked={edgeAi} onChange={(e) => setEdgeAi(e.target.checked)} /> Edge AI 포함
            </label>
          )}
          {hasScan && (
            <label
              className="check"
              title="켜짐: 선별 결과를 r004 감시목록에 저장 (다음 r003 실행부터 그 종목을 감시). 꺼짐: 기준일 폴더에 선별 파일(picks·리포트)만 만들고 감시목록은 그대로 - 과거 날짜 보충 수집 / Edge AI 데이터셋용"
            >
              <input type="checkbox" checked={wlUpdate} onChange={(e) => setWlChoice(e.target.checked)} /> 감시목록(r004) 갱신
            </label>
          )}
          {hasScan && !wlUpdate && (
            <span className="muted small">{sel === 'scan' ? '선별 결과를 보고 아래에서 r004에 적용합니다' : '선별 파일만 만들고 지금 감시 중인 종목은 그대로 둡니다'}</span>
          )}
          {hasScan && wlUpdate && !isToday && <span className="warn-text small">과거 날짜({date}) 기준으로 감시목록이 바뀝니다</span>}
          {busy && <i className="spinner" />}
          <span className="muted small">{msg}</span>
          <span className="push-right" />
          <button
            className="primary wide"
            disabled={busy || !!running || noData}
            title={running ? '다른 작업이 실행 중입니다' : noData ? `${date} 수집 데이터가 없습니다` : ''}
            onClick={() => void start()}
          >
            시작
          </button>
          <button className="wide" disabled={busy || running !== sel} onClick={() => void stop()}>
            정지
          </button>
        </div>
        {last && (
          <>
            {last.error && <p className={last.state === 'failed' ? 'err-text small' : 'muted small'}>{last.error}</p>}
            <table className="grid">
              <thead>
                <tr>
                  <th>단계</th>
                  <th>상태</th>
                  <th>rc</th>
                  <th>소요</th>
                  <th>명령</th>
                </tr>
              </thead>
              <tbody>
                {last.steps.map((s, i) => (
                  <tr key={`${s.name}-${i}`} className="nohover">
                    <td>{s.name}</td>
                    <td>
                      <span className={stateClass(s.state)}>{stateText(s.state)}</span>
                    </td>
                    <td className="mono">{s.rc ?? ''}</td>
                    <td className="mono" title={s.started_at ? `시작 ${isoShow(s.started_at)}` : ''}>
                      {dur(s.started_at, s.finished_at)}
                    </td>
                    <td className="mono wrap" title={s.argv.join(' ')}>
                      {shortCmd(s.argv)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {hasScan && last.state.startsWith('done') && last.args?.keep_watchlist && (
              <p className="muted small">
                감시목록(r004)은 바꾸지 않았습니다 - 기준일 {last.args.date} 폴더의 선별 파일만 만들었습니다
                {last.args.date === date ? ' (아래 선별 결과에서 적용할 수 있습니다).' : '.'}
              </p>
            )}
            {hasScan && last.state.startsWith('done') && !last.args?.keep_watchlist && (
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
        {hasScan && infoOk?.picks && (
          <div className="picks">
            <div className="picks-head">
              <b>선별 결과</b>
              <span className="muted small" title={infoOk.picks_file}>
                {date} · {infoOk.picks.length}종목{infoOk.picks_mtime ? ` · ${isoShow(infoOk.picks_mtime)}` : ''}
              </span>
              <label
                className="check small"
                title={
                  infoOk.r004?.applied
                    ? '지금 r004 감시목록이 이 선별 결과와 같습니다'
                    : '체크하면 이 선별 결과를 r004 감시목록으로 저장합니다 (기존 파일 백업, r003 재시작 후 반영)'
                }
              >
                <input type="checkbox" checked={!!infoOk.r004?.applied} disabled={busy || !!infoOk.r004?.applied || !infoOk.picks.length} onChange={() => void apply()} />
                r004 감시목록에 적용{infoOk.r004?.applied ? ' (적용됨)' : ''}
              </label>
              {infoOk.r004 && !infoOk.r004.applied && (
                <span className="muted small">
                  현재 r004: {infoOk.r004.count}종목{infoOk.r004.scan_date ? ` (${infoOk.r004.scan_date} 선별)` : ''}
                </span>
              )}
            </div>
            <div className="pick-chips">
              {infoOk.picks.map((p) => (
                <span className="pick-chip" key={p.code}>
                  {p.name} <span className="mono">{p.code}</span>
                </span>
              ))}
            </div>
          </div>
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
