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
 * - [2026-11-11] type=feat - "감시목록(r004) 갱신" checkbox (r021 --keep-watchlist when off) - replaced the same day, see below
 * - [2026-11-11] type=feat - Scan-only flow (job data: what the date folder holds, picks list, apply checkbox) - replaced the same
 *   day by the step model below
 * - [2026-11-11] type=refactor - Pipeline steps: the 장 마감 파이프라인 card is the one place for ① 데이터 수집 (g001 + g025) ② 종목 선별
 *   (g002) ③ r004 감시목록 반영 - each a checkbox, combined freely (1+2+3, 2+3, 3; r021 --no-collect / --no-scan / --no-apply).
 *   Defaults follow the date folder (r016 job data): ① off when the date is already collected, ② on, ③ on for today only. A step
 *   whose input is missing (② without collected data, ③ without a picks file) is refused with the reason and 시작 is disabled.
 *   The picks of the date are listed under the form (name + code) with whether r004 already holds them. The separate collect /
 *   history / scan rows left the job table (still r021 jobs for the CLI); Edge AI 재학습 and 유니버스 갱신 stay
 * - [2026-11-11] type=fix - Codex review: a collect / history / scan job started from the CLI stays in the table while it runs (stop / log)
 * - [2026-11-11] type=fix - Codex review: a skipped apply step (no picks of this run - r021 state note) is shown as 건너뜀 with its
 *   reason; after an applied run while r003 runs, a line says the new list needs an r003 restart (link to the control tab)
 * - [2026-11-11] type=feat - "선별 결과 메일" checkbox on the pipeline card (r021 --email -> g002 --notify-email: picks in the body,
 *   report files attached); only with step ②, off by default; the plan line and the last-run line show it, the automation line
 *   shows settings.postcloseEmail
 * - [2026-11-11] type=feat - The last-run line says 자동 실행 (스케줄러) / 수동 실행 (state args.scheduled, r021 --scheduled) and that
 *   the step checkboxes above apply to the next manual start only - a scheduled run with Edge AI under an unchecked
 *   "Edge AI 포함" box looked like a bug
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { DatePicker } from '../components/DatePicker';
import { JOB_STATE_LABEL, localYmd, type JobDataInfo, type JobList, type JobLog, type JobRun } from '../../../shared/control';

const isoShow = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(5, 16) : '-');
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

/** jobs shown in the table - collect / history / scan are steps of the pipeline card now (r021 keeps them for the CLI) */
const VISIBLE_JOBS = ['postclose', 'edge_ai', 'universe'];

const DESC: Record<string, string> = {
  postclose: '기준일을 고르고 실행할 단계를 체크 · ① 수집(g001 분봉·일봉 + g025 일봉 이력) → ② 선별(g002) → ③ r004 감시목록 반영 · 앞 단계를 끄면 그 산출물이 이미 있어야 합니다',
  edge_ai: 'Edge AI 데이터셋 생성 → 학습 → 점검 · CPU 많이 씀',
  universe: '수집 대상 종목 목록(g004) 갱신 · 신규상장/상장폐지/위험종목',
  collect: '분봉·일봉 데이터 수집 · KIS API 전 종목 (오래 걸림)',
  history: '종목별 일봉 이력 갱신 · 차트 일/주/월봉',
  scan: '수집된 데이터로 감시 종목 선별 후 r004 반영',
};

const STEP_LABEL: Record<string, string> = { collect: '수집', history: '이력', scan: '선별', apply: 'r004 반영', edge_dataset: 'Edge AI 데이터셋', edge_train: 'Edge AI 학습', edge_check: 'Edge AI 점검', universe: '유니버스' };

/** state as a dot + text (.st), colored by kind */
function stateClass(st: string | undefined) {
  if (st === 'running') return 'st st-run';
  if (st === 'done') return 'st st-ok';
  if (st === 'failed' || st === 'done_with_errors') return 'st st-fail';
  return 'st st-none';
}

type StepKey = 'collect' | 'scan' | 'apply';

export function JobsView() {
  const connected = useStore((s) => s.status.conn === 'connected');
  const local = useStore((s) => s.status.mode === 'local');
  const sched = useStore((s) => s.sched);
  const settings = useStore((s) => s.settings);
  const setTab = useStore((s) => s.setTab);
  const r003Running = useStore((s) => !!s.control.status?.running);
  const [list, setList] = useState<JobList | null>(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState('postclose');
  const [date, setDate] = useState(localYmd());
  const [workers, setWorkers] = useState('');
  const [edgeAi, setEdgeAi] = useState(false);
  const [email, setEmail] = useState(false);
  const [log, setLog] = useState<JobLog | null>(null);
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const isToday = date === localYmd();
  const pipeline = sel === 'postclose';

  /** what the date folder holds (r016 job data) - drives the step defaults and the refusals */
  const [info, setInfo] = useState<JobDataInfo | null>(null);
  const infoOk = info && info.date === date ? info : null;
  /** the user's step choices; a step not chosen follows its default (reset when the date changes) */
  const [choice, setChoice] = useState<Partial<Record<StepKey, boolean>>>({});
  useEffect(() => setChoice({}), [date]);
  const step: Record<StepKey, boolean> = {
    collect: choice.collect ?? !infoOk?.collected,
    scan: choice.scan ?? true,
    apply: choice.apply ?? isToday,
  };
  const choose = (k: StepKey, v: boolean) => setChoice((c) => ({ ...c, [k]: v }));

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
  // date folder info: on the pipeline card, again when the date changes and when the pipeline's run ends
  useEffect(() => {
    if (!connected || !pipeline) return;
    void loadInfo(date);
  }, [connected, pipeline, date, last?.finished_at, last?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- pipeline checks (the server refuses the same cases; here they disable 시작 with the reason) ----
  const chosen = pipeline ? (['collect', 'scan', 'apply'] as StepKey[]).filter((k) => step[k]) : [];
  let problem = '';
  if (pipeline) {
    if (!chosen.length) problem = '실행할 단계를 하나 이상 체크하세요';
    else if (step.scan && !step.collect && infoOk && !infoOk.scannable) problem = `${date} 수집 데이터가 없습니다 - ① 수집을 켜세요`;
    else if (step.apply && !step.scan && infoOk && !infoOk.picks) problem = `${date} 선별 결과가 없습니다 - ② 선별을 켜세요`;
  }
  const planText = chosen.length
    ? chosen.map((k) => ({ collect: '수집 → 이력', scan: step.scan && email ? '선별(메일 전송)' : '선별', apply: 'r004 반영' })[k]).join(' → ') + (edgeAi ? ' → Edge AI' : '')
    : '';

  const start = async () => {
    if (!/^\d{8}$/.test(date)) return setMsg('날짜는 YYYYMMDD');
    if (problem) return setMsg(problem);
    setBusy(true);
    setMsg('시작 중…');
    const r = await window.api.ctlJobStart(sel, {
      date,
      workers: Number(workers) || undefined,
      edgeAi: pipeline && edgeAi,
      ...(pipeline ? { collect: step.collect, scan: step.scan, apply: step.apply, email: step.scan && email } : {}),
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

  const dataLine = () => {
    if (!infoOk) return <span className="muted">기준일 데이터 확인 중…</span>;
    if (infoOk.error && !infoOk.ok) return <span className="err-text">기준일 데이터 확인 실패: {infoOk.error}</span>;
    if (!infoOk.exists || !infoOk.scannable) return <span className="muted">기준일 {date} 수집 데이터 없음</span>;
    return (
      <span className="muted" title={infoOk.dir}>
        기준일 {date} 데이터 있음 · 10초봉 {num(infoOk.files_10s)}종목 · 일봉 {num(infoOk.files_daily)}
        {infoOk.collected ? ` · 수집 완료 ${num(infoOk.collected_count)}종목` : ' · 수집 완료 표시 없음(중단됨?)'}
        {infoOk.data_mtime ? ` · 마지막 파일 ${isoShow(infoOk.data_mtime)}` : ''}
        {infoOk.picks ? ` · 선별 결과 ${infoOk.picks.length}종목 (${isoShow(infoOk.picks_mtime)})` : ' · 선별 결과 없음'}
      </span>
    );
  };
  /** last run of the pipeline: did it write r004? (older state files: keep_watchlist) */
  const lastApplied = last?.args ? (last.args.apply ?? !last.args.keep_watchlist) : false;

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
            {(list?.jobs || [])
              // a hidden job (collect / history / scan from the CLI) stays visible while it runs or is selected, so it can be stopped
              .filter((j) => VISIBLE_JOBS.includes(j.name) || j.name === running || j.name === sel)
              .map((j) => {
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
        {pipeline && <p className="small data-line">{dataLine()}</p>}
        <div className="job-form">
          <DatePicker value={date} onChange={setDate} today={localYmd()} max={localYmd()} title="기준일 (누르면 달력, 기본은 오늘)" />
          {(sel === 'postclose' || sel === 'edge_ai' || sel === 'collect') && (
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
          {pipeline && (
            <div className="steps">
              <label className="check step" title="g001 분봉·일봉 수집 + g025 일봉 이력 · KIS API 전 종목, 오래 걸림 · 이미 수집된 날짜는 기본으로 꺼짐 (켜면 끝난 종목은 건너뛰고 이어받음)">
                <input type="checkbox" checked={step.collect} onChange={(e) => choose('collect', e.target.checked)} />
                <span className="step-num">1</span> 데이터 수집 <span className="code-tag">g001</span>
              </label>
              <label className="check step" title="g002 감시 종목 선별 · 기준일 폴더에 선별 파일(_D_picks.txt, 리포트)을 만듭니다 · r004는 ③이 씁니다">
                <input type="checkbox" checked={step.scan} onChange={(e) => choose('scan', e.target.checked)} />
                <span className="step-num">2</span> 종목 선별 <span className="code-tag">g002</span>
              </label>
              <label className="check step" title="기준일의 선별 결과를 r004 감시목록으로 저장 · 기존 파일은 watchlist_backup/에 백업 · r003 재시작 후 반영 · 과거 날짜는 기본으로 꺼짐">
                <input type="checkbox" checked={step.apply} onChange={(e) => choose('apply', e.target.checked)} />
                <span className="step-num">3</span> r004 감시목록 반영
              </label>
              <label className="check" title="③ 뒤에 Edge AI 재학습(g013/g014/g024)을 이어서 실행">
                <input type="checkbox" checked={edgeAi} onChange={(e) => setEdgeAi(e.target.checked)} /> Edge AI 포함
              </label>
              <label className="check" title={`② 선별이 끝나면 선별 종목(본문)과 리포트 파일(첨부)을 이메일로 전송 (g002 --notify-email) · ${local ? '이 PC' : '라즈베리파이'}의 ~/.mail/mail_config.json 설정 필요 · ②를 켜야 보냅니다 · 전송 결과는 아래 로그의 [EMAIL] 줄 (실패해도 작업은 계속)`}>
                <input type="checkbox" checked={step.scan && email} disabled={!step.scan} onChange={(e) => setEmail(e.target.checked)} /> 선별 결과 메일
              </label>
            </div>
          )}
          {busy && <i className="spinner" />}
          <span className="push-right" />
          <button className="primary wide" disabled={busy || !!running || !!problem} title={running ? '다른 작업이 실행 중입니다' : problem} onClick={() => void start()}>
            시작
          </button>
          <button className="wide" disabled={busy || running !== sel} onClick={() => void stop()}>
            정지
          </button>
        </div>
        {pipeline && (
          <p className="small plan-line">
            {problem ? (
              <span className="warn-text">{problem}</span>
            ) : (
              <>
                <span className="muted">실행: </span>
                {planText}
                {step.collect && infoOk?.collected && <span className="muted"> · 이미 수집된 날짜입니다 - ①을 끄면 선별부터 합니다</span>}
                {step.apply && !isToday && <span className="warn-text"> · 과거 날짜({date}) 선별 기준으로 r004 감시목록이 바뀝니다</span>}
                {step.apply && !step.scan && infoOk?.picks && <span className="muted"> · {isoShow(infoOk.picks_mtime)}에 만든 선별 결과 {infoOk.picks.length}종목을 반영합니다</span>}
              </>
            )}
            {msg && <span className="muted"> · {msg}</span>}
          </p>
        )}
        {!pipeline && msg && <p className="muted small">{msg}</p>}
        {pipeline && infoOk?.picks && (
          <div className="picks">
            <div className="picks-head">
              <b>선별 결과</b>
              <span className="muted small" title={infoOk.picks_file}>
                {date} · {infoOk.picks.length}종목{infoOk.picks_mtime ? ` · ${isoShow(infoOk.picks_mtime)}` : ''}
              </span>
              {infoOk.r004 && (
                <span className={`small ${infoOk.r004.applied ? 'ok-text' : 'muted'}`}>
                  {infoOk.r004.applied
                    ? 'r004 감시목록에 반영됨'
                    : `현재 r004: ${infoOk.r004.count}종목${infoOk.r004.scan_date ? ` (${infoOk.r004.scan_date} 선별)` : ''} - 아직 반영되지 않음`}
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
        {last && (
          <>
            <p className="muted small">
              마지막 실행 {isoShow(last.started_at)}
              {last.args?.scheduled === true && <b title="스케줄러가 시작한 실행입니다 · 실행 단계는 설정/연결 탭의 자동 실행 설정을 따릅니다 (위 체크박스와 무관)"> · 자동 실행 (스케줄러)</b>}
              {last.args?.scheduled === false && <b title="[시작] 버튼 또는 명령줄로 시작한 실행입니다"> · 수동 실행</b>}
              {' '}· 기준일 {last.args?.date}
              {last.steps.length ? ` · ${last.steps.map((s) => STEP_LABEL[s.name] || s.name).join(' → ')}` : ''}
              {last.args?.email ? ' · 선별 결과 메일 전송' : ''}
            </p>
            {pipeline && last.args?.scheduled === true && (
              <p className="muted small">자동 실행의 단계(Edge AI·메일 포함 여부)는 설정/연결 탭의 자동 실행 설정을 따릅니다 - 위 체크박스는 [시작]으로 직접 실행할 때만 적용됩니다.</p>
            )}
            {last.error && <p className={last.state === 'failed' ? 'err-text small' : 'muted small'}>{last.error}</p>}
            {last.note && <p className="warn-text small">{last.note}</p>}
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
                    <td>{STEP_LABEL[s.name] || s.name}</td>
                    <td>
                      <span className={stateClass(s.state)}>{stateText(s.state)}</span>
                    </td>
                    <td className="mono">{s.rc ?? ''}</td>
                    <td className="mono" title={s.started_at ? `시작 ${isoShow(s.started_at)}` : ''}>
                      {dur(s.started_at, s.finished_at)}
                    </td>
                    <td className="mono wrap" title={s.argv.join(' ')}>
                      {s.argv.length ? shortCmd(s.argv) : <span className="muted">{s.note || ''}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {pipeline && last.state.startsWith('done') && lastApplied && last.steps.some((s) => s.name === 'apply' && s.state === 'done') && (
              <p className="small">
                새 감시목록은{' '}
                <button className="linklike" onClick={() => setTab('board')}>
                  종목 보드
                </button>
                의 감시목록 편집에서 확인·수정할 수 있습니다 (다음 r003 실행부터 적용).
                {r003Running && (
                  <>
                    {' '}
                    <span className="warn-text">r003이 실행 중입니다 - 새 감시목록은 재시작 후 반영됩니다</span> (
                    <button className="linklike" onClick={() => setTab('control')}>
                      제어
                    </button>
                    ).
                  </>
                )}
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
              틱 수집기: {settings?.autoCollector ? '평일 07:55~19:55 5분마다 확인해 없으면 시작 (KRX 정규장 + NXT 08:00~20:00)' : '꺼짐'} {sched?.lastCollectorMsg && <span className="muted">- {sched.lastCollectorMsg}</span>}
            </li>
            <li>
              장 마감 작업: {settings?.autoPostclose ? `거래일 ${settings.postcloseTime} 이후 1회 (r003 종료 후, ①②③ 전부)${settings.postcloseEdgeAi ? ' + Edge AI' : ''}${settings.postcloseEmail ? ' + 선별 결과 메일' : ''}` : '꺼짐'}{' '}
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
