/*
 * Update log
 * - [2026-11-04] type=feat - Settings / connection: SSH (host, port, user, key file + passphrase or password - stored encrypted by
 *   the main process), remote root, python path, poll interval, session open time, cache retention; date selection (server folders
 *   and local cache), local folder (fixtures), condition JSON refresh, and the list of unknown tags/reasons
 * - [2026-11-05] type=feat - SSH proxy option (same as TermH Add/Edit Host): 프록시 select (사용 안 함 / ProxyCommand / SOCKS5 / HTTP CONNECT)
 *   with the ProxyCommand line + %h/%p/%r hint, or proxy host/port/user/password (password stored encrypted like the SSH secret)
 * - [2026-11-07] type=feat - "매매 시간 (서버 r003)": NXT session toggle (ENABLE_NXT_SESSION via r016 set / unset - back to the r001
 *   value is an unset, so no override is left behind) with the stored / running value and the restart-needed mark. The funnel start
 *   time ("장 시작 (집계 기준)") is automatic by default: it follows the server NXT setting (08:00 / 09:00); unticking it enables
 *   the manual time
 * - [2026-11-07] type=fix - The date select starts at (and follows) the loaded date; "저장 후 연결" opens the selected past date
 * - [2026-11-10] type=feat - "실행 위치" section: 라즈베리파이 (SSH) / 이 PC (로컬), local root + venv Python (checked live), local poll
 *   intervals, automation (tick collector, post-close job + time + Edge AI, watchlist copy to the Pi, keep awake), doctor of both
 *   hosts side by side (version mismatches marked), KIS config import from the Pi and the one-time setup steps
 * - [2026-11-05] type=feat - "고급: 전체 설정값" (collapsed): `schema --all` (~380 r001 constants) with a search box and the same value
 *   editor as the pipeline panel (only changed keys are sent in one `set`); the read-only note now lists the r016 control calls
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import type { Settings } from '../../../shared/types';
import { fmtInt } from '../lib';
import { ymdDash } from '../../../shared/time';
import { NXT_KEY, resolveSessionOpen, type DoctorReport, type SchemaDoc } from '../../../shared/control';
import { EditorActions, useValueEditing, ValuesEditor, type EditItem } from '../components/ValuesEditor';

export function SettingsView() {
  const stored = useStore((s) => s.settings);
  const status = useStore((s) => s.status);
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const [form, setForm] = useState<Settings | null>(stored);
  const [secret, setSecret] = useState('');
  const [proxySecret, setProxySecret] = useState('');
  const [msg, setMsg] = useState('');
  const [dates, setDates] = useState<{ remote: string[]; cached: string[] }>({ remote: [], cached: [] });
  // the date select shows the loaded day (kept when the tab is left and re-entered)
  const [date, setDate] = useState(status.date || '');
  useEffect(() => setDate(status.date || ''), [status.date]);
  const nxt = useStore((s) => s.control.nxt);

  useEffect(() => setForm(stored), [stored]);
  const loadDates = () => window.api.listDates().then(setDates);
  useEffect(() => {
    void loadDates();
  }, [status.conn]);

  if (!form) return null;
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setForm({ ...form, [k]: v });

  const save = async (): Promise<boolean> => {
    const r = await window.api.saveSettings({ ...form, ...(secret ? { secret } : {}), ...(proxySecret ? { proxySecret } : {}) });
    if (!r.ok) {
      setMsg(r.error);
      return false;
    }
    useStore.setState({ settings: r.settings });
    setSecret('');
    setProxySecret('');
    setMsg('저장했습니다.');
    return true;
  };
  const connect = async () => {
    if (!(await save())) return;
    setMsg('연결 중…');
    const r = await window.api.connect(date && date !== status.serverDate ? date : undefined);
    setMsg(r.ok ? '연결했습니다.' : `연결 실패: ${r.error}`);
  };
  const allDates = [...new Set([...dates.remote, ...dates.cached])].sort().reverse();

  return (
    <div className="settings">
      <RunModeSection form={form} set={set} onConnect={connect} onSave={save} msg={msg} />

      <section>
        <h3>라즈베리파이 SSH 연결 {form.runMode === 'local' ? '(예비 호스트 - 상태 확인 / standby 전환 / 감시목록 복사용)' : ''}</h3>
        <div className="form">
          <label>호스트</label>
          <input value={form.host} onChange={(e) => set('host', e.target.value)} placeholder="raspberrypi.local / 192.168.0.x" />
          <label>포트</label>
          <input type="number" value={form.port} onChange={(e) => set('port', Number(e.target.value))} />
          <label>사용자</label>
          <input value={form.username} onChange={(e) => set('username', e.target.value)} />
          <label>인증</label>
          <div className="seg">
            <button className={form.authMethod === 'key' ? 'on' : ''} onClick={() => set('authMethod', 'key')}>
              개인키 (PEM)
            </button>
            <button className={form.authMethod === 'password' ? 'on' : ''} onClick={() => set('authMethod', 'password')}>
              비밀번호
            </button>
          </div>
          {form.authMethod === 'key' && (
            <>
              <label>개인키 파일</label>
              <div className="row">
                <input value={form.keyPath} onChange={(e) => set('keyPath', e.target.value)} />
                <button onClick={async () => { const p = await window.api.pickKey(); if (p) set('keyPath', p); }}>찾기…</button>
              </div>
            </>
          )}
          <label>{form.authMethod === 'key' ? '키 암호 (선택)' : '비밀번호'}</label>
          <div className="row">
            <input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} placeholder={form.hasSecret ? '저장됨 (바꾸려면 입력)' : '없음'} />
            {form.hasSecret && (
              <button onClick={async () => { const r = await window.api.saveSettings({ ...form, secret: '' }); if (r.ok) useStore.setState({ settings: r.settings }); }}>지우기</button>
            )}
          </div>
          <label>프록시</label>
          <select value={form.proxyType || ''} onChange={(e) => set('proxyType', e.target.value as Settings['proxyType'])}>
            <option value="">사용 안 함 (직접 연결)</option>
            <option value="command">ProxyCommand</option>
            <option value="socks5">SOCKS5</option>
            <option value="http">HTTP CONNECT</option>
          </select>
          {form.proxyType === 'command' && (
            <>
              <label>ProxyCommand</label>
              <div className="field-col">
                <input
                  className="mono"
                  spellCheck={false}
                  value={form.proxyCommand || ''}
                  onChange={(e) => set('proxyCommand', e.target.value)}
                  placeholder="cloudflared access ssh --hostname %h"
                />
                <p className="muted small">
                  %h = 호스트, %p = 포트, %r = 사용자명, %% = %<br />
                  예: <code>cloudflared access ssh --hostname %h</code>
                  <br />
                  예: <code>ssh -W %h:%p user@jumphost</code>
                </p>
              </div>
            </>
          )}
          {(form.proxyType === 'socks5' || form.proxyType === 'http') && (
            <>
              <label>프록시 호스트</label>
              <input spellCheck={false} value={form.proxyHost || ''} onChange={(e) => set('proxyHost', e.target.value)} />
              <label>프록시 포트</label>
              <input
                type="number"
                value={form.proxyPort || ''}
                onChange={(e) => set('proxyPort', Number(e.target.value) || 0)}
                placeholder={form.proxyType === 'socks5' ? '1080' : '8080'}
              />
              <label>프록시 사용자 (선택)</label>
              <input value={form.proxyUser || ''} onChange={(e) => set('proxyUser', e.target.value)} />
              <label>프록시 비밀번호 (선택)</label>
              <div className="row">
                <input
                  type="password"
                  value={proxySecret}
                  onChange={(e) => setProxySecret(e.target.value)}
                  placeholder={form.hasProxySecret ? '저장됨 (바꾸려면 입력)' : '없음'}
                />
                {form.hasProxySecret && (
                  <button onClick={async () => { const r = await window.api.saveSettings({ ...form, proxySecret: '' }); if (r.ok) useStore.setState({ settings: r.settings }); }}>지우기</button>
                )}
              </div>
            </>
          )}
          <label>원격 루트</label>
          <input value={form.remoteRoot} onChange={(e) => set('remoteRoot', e.target.value)} />
          <label>Python</label>
          <input value={form.pythonPath} onChange={(e) => set('pythonPath', e.target.value)} />
          <label>폴링 주기 (초)</label>
          <input type="number" min={3} value={form.pollSec} onChange={(e) => set('pollSec', Number(e.target.value))} />
          <label>봇 상태 확인 주기 (초)</label>
          <input type="number" min={15} value={form.statusPollSec} onChange={(e) => set('statusPollSec', Number(e.target.value))} title="r016 status 호출 주기 - 호출마다 서버(라즈베리파이)에서 Python이 실행되므로 15초 이상. 창을 최소화하면 120초" />
          <label>장 시작 (집계 기준)</label>
          <div className="inline-row" title="이 시각 전의 반려는 '개장 전'으로 따로 집계합니다 (깔때기에서 제외). r003의 매매 시간과는 무관합니다.">
            <label className="check">
              <input type="checkbox" checked={form.sessionOpenAuto} onChange={(e) => set('sessionOpenAuto', e.target.checked)} /> 자동 (서버 NXT 설정)
            </label>
            {form.sessionOpenAuto ? (
              <span className="muted">
                {resolveSessionOpen(true, form.sessionOpen, nxt)}
                {nxt ? (nxt.running ?? nxt.effective ? ' - NXT 운용' : ' - 정규장만') : ' - 서버 값 모름, 수동 값 사용'}
              </span>
            ) : (
              <input value={form.sessionOpen} onChange={(e) => set('sessionOpen', e.target.value)} placeholder="09:00" style={{ width: 90 }} />
            )}
          </div>
          <label>로컬 캐시 보관 (일)</label>
          <input type="number" min={1} value={form.retentionDays} onChange={(e) => set('retentionDays', Number(e.target.value))} />
        </div>
        <div className="actions">
          <button onClick={save}>저장</button>
          <button className="primary" onClick={connect}>
            저장 후 연결
          </button>
          <button onClick={() => window.api.disconnect()} disabled={status.conn === 'idle' || status.conn === 'local'}>
            연결 해제
          </button>
          <span className="muted">{msg}</span>
        </div>
        <p className="muted small">
          로그는 SFTP로 읽기만 합니다. 서버 변경(봇 실행/정지, 설정값, 감시목록)은 모두 <code>live_trading/r016_control.py</code>를 통해서만 하며, 주문 API와 계좌 정보에는 접근하지 않습니다. 비밀번호/키 암호/프록시 비밀번호는 Windows 사용자 계정으로 암호화(safeStorage)해 저장합니다.
          장 시작(집계 기준) 시각이 바뀌면 열려 있는 날짜를 다시 집계합니다.
        </p>
      </section>

      <NxtSection />

      <section>
        <h3>날짜 / 데이터 소스</h3>
        <div className="actions">
          <select value={date} onChange={(e) => setDate(e.target.value)}>
            <option value="">날짜 선택…</option>
            {allDates.map((d) => (
              <option key={d} value={d}>
                {ymdDash(d)} {dates.remote.includes(d) ? '' : '(캐시)'}
                {d === status.serverDate ? ' - 오늘' : ''}
              </option>
            ))}
          </select>
          <button disabled={!date} onClick={async () => { const r = await window.api.openDay(date); setMsg(r.ok ? `${ymdDash(date)} 로드` : r.error); }}>
            열기
          </button>
          <button onClick={loadDates}>목록 새로고침</button>
          <button
            onClick={async () => {
              const d = await window.api.pickFolder();
              if (!d) return;
              const r = await window.api.openLocal(d);
              setMsg(r.ok ? `로컬 폴더: ${d}` : r.error);
            }}
          >
            로컬 폴더 열기…
          </button>
        </div>
        <p className="muted small">오늘(서버 날짜)은 {form.pollSec}초마다 증분 수신(봇 상태는 {form.statusPollSec}초마다, 창 최소화 시 120초), 지난 날짜는 폴링 없이 1회 로드합니다. 로컬 폴더는 runtime/log/YYYYMMDD 복사본(픽스처)을 SSH 없이 엽니다.</p>
      </section>

      <section>
        <h3>조건 목록</h3>
        <p>
          출처: <b>{conditions?.source === 'server' ? '서버' : conditions?.source === 'cache' ? '마지막 성공본 (캐시)' : '앱 내장 스냅샷'}</b> · 생성 {conditions?.generated_at || '-'} · 매수{' '}
          {conditions?.buy.length || 0} / 매도 {conditions?.sell.length || 0} · entry_trigger_mode={conditions?.entry_trigger_mode || '-'} · trading_strategy_mode=
          {conditions?.trading_strategy_mode || '-'}
        </p>
        <div className="actions">
          <button
            onClick={async () => {
              setMsg('조건 JSON 받는 중…');
              const r = await window.api.refreshConditions();
              setMsg(r.ok ? '조건 JSON을 새로 받았습니다.' : `실패: ${r.error}`);
            }}
          >
            조건 JSON 새로고침
          </button>
        </div>
      </section>

      <AdvancedValues />

      <section>
        <h3>알 수 없는 태그 / 사유 ({fmtInt(snap?.warnings || 0)}줄)</h3>
        <p className="muted small">매핑 표에 없는 줄은 버리지 않고 여기에 모읍니다 (숫자는 # 로 묶음). 새 조건이 추가되면 이 목록에 나타납니다.</p>
        <div className="table-wrap short">
          <table className="grid compact">
            <thead>
              <tr>
                <th>태그 | 사유 형태</th>
                <th className="num">건수</th>
                <th>예시 원문</th>
              </tr>
            </thead>
            <tbody>
              {(snap?.unknown || []).map((u) => (
                <tr key={u.key}>
                  <td className="mono">{u.key}</td>
                  <td className="num">{fmtInt(u.count)}</td>
                  <td className="mono raw-cell" title={u.sample}>
                    {u.sample}
                  </td>
                </tr>
              ))}
              {!snap?.unknown.length && (
                <tr>
                  <td colSpan={3} className="muted center">
                    없음
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

type SetFn = <K extends keyof Settings>(k: K, v: Settings[K]) => void;

/** [2026-11-10] where r003 runs + local mode paths + automation + environment check */
function RunModeSection({ form, set, onConnect, onSave, msg }: { form: Settings; set: SetFn; onConnect: () => void; onSave: () => Promise<boolean>; msg: string }) {
  const status = useStore((s) => s.status);
  const [check, setCheck] = useState<{ problem: string; paths: { python: string; r016: string; runtime: string } } | null>(null);
  const [doc, setDoc] = useState<{ self?: DoctorReport | null; peer?: DoctorReport | null; selfErr?: string; peerErr?: string; peerLabel?: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [kisMsg, setKisMsg] = useState('');
  const local = form.runMode === 'local';

  useEffect(() => {
    let alive = true;
    void window.api.localCheck(form).then((r) => alive && setCheck(r));
    return () => {
      alive = false;
    };
  }, [form.localRoot, form.localPython]); // eslint-disable-line react-hooks/exhaustive-deps

  const runDoctor = async () => {
    setBusy(true);
    setDoc(null);
    const [a, info, b] = await Promise.all([window.api.ctlDoctor(), window.api.peerInfo(), window.api.peerDoctor()]);
    setDoc({ self: a.data, selfErr: a.ok ? undefined : a.error, peer: b.data, peerErr: b.ok ? undefined : b.error, peerLabel: info.label });
    setBusy(false);
  };
  const importKis = async () => {
    setKisMsg('가져오는 중…');
    const r = await window.api.peerImportKis();
    setKisMsg(r.cancelled ? '' : r.ok ? `저장했습니다: ${r.path}` : `실패: ${r.error}`);
  };

  return (
    <section>
      <h3>실행 위치 (r003 자동매매)</h3>
      <div className="actions">
        <div className="seg">
          <button className={!local ? 'on' : ''} onClick={() => set('runMode', 'remote')}>
            라즈베리파이 (SSH)
          </button>
          <button className={local ? 'on' : ''} onClick={() => set('runMode', 'local')}>
            이 PC (로컬)
          </button>
        </div>
        <span className="muted small">
          현재 연결: {status.conn === 'connected' ? (status.mode === 'local' ? '이 PC' : '라즈베리파이') : '없음'} - 바꾼 뒤 [저장 후 연결]을 누르면 적용됩니다. 제어 탭 맨 위 "매매 실행 위치"에서 [여기서 실행]으로도 바꿀 수 있습니다.
        </span>
      </div>
      {local && (
        <>
          <div className="form">
            <label>로컬 루트 폴더</label>
            <div className="row">
              <input value={form.localRoot} onChange={(e) => set('localRoot', e.target.value)} placeholder="D:\make_utils\Trading (real_trading_bot / runtime / venv 가 있는 폴더)" />
              <button
                onClick={async () => {
                  const p = await window.api.pickFolderTitled('로컬 루트 폴더 (real_trading_bot 이 있는 폴더)');
                  if (p) set('localRoot', p);
                }}
              >
                찾기…
              </button>
            </div>
            <label>Python (venv)</label>
            <div className="row">
              <input value={form.localPython} onChange={(e) => set('localPython', e.target.value)} placeholder={check?.paths.python || '<루트>\\venv\\Scripts\\python.exe'} />
              <button
                onClick={async () => {
                  const p = await window.api.pickFile('venv 의 python.exe');
                  if (p) set('localPython', p);
                }}
              >
                찾기…
              </button>
            </div>
            <label>로그 폴링 (초)</label>
            <input type="number" min={1} value={form.localPollSec} onChange={(e) => set('localPollSec', Number(e.target.value))} />
            <label>봇 상태 확인 (초)</label>
            <input type="number" min={3} value={form.localStatusPollSec} onChange={(e) => set('localStatusPollSec', Number(e.target.value))} />
          </div>
          {check?.problem ? <p className="err-text small">{check.problem}</p> : <p className="muted small">로컬 실행 준비됨 · 런타임 {check?.paths.runtime}</p>}

          <h3>자동 실행 (앱이 켜져 있을 때)</h3>
          <div className="checks">
            <label className="check">
              <input type="checkbox" checked={form.autoCollector} onChange={(e) => set('autoCollector', e.target.checked)} /> 평일 아침 틱 수집기(g011) 자동 시작 (08:45부터, 5분마다 확인)
            </label>
            <label className="check">
              <input type="checkbox" checked={form.autoPostclose} onChange={(e) => set('autoPostclose', e.target.checked)} /> 장 마감 작업 자동 실행 (수집 → 이력 → 감시 종목 선별)
              <input value={form.postcloseTime} style={{ width: 64, marginLeft: 6 }} onChange={(e) => set('postcloseTime', e.target.value)} placeholder="20:10" /> 이후
            </label>
            <label className="check">
              <input type="checkbox" checked={form.postcloseEdgeAi} disabled={!form.autoPostclose} onChange={(e) => set('postcloseEdgeAi', e.target.checked)} /> 장 마감 작업에 Edge AI 재학습 포함
            </label>
            <label className="check">
              <input type="checkbox" checked={form.syncWatchlistToPeer} onChange={(e) => set('syncWatchlistToPeer', e.target.checked)} /> 장 마감 작업이 끝나면 새 감시목록을 라즈베리파이로 복사 (예비 실행 대비)
            </label>
            <label className="check">
              <input type="checkbox" checked={form.keepAwake} onChange={(e) => set('keepAwake', e.target.checked)} /> 이 PC의 r003이 실행 중이면 절전 방지
            </label>
          </div>
          <p className="muted small">
            NXT 운용 시 r003은 20:00까지 돌므로 장 마감 작업은 r003이 끝난 뒤 시작합니다. 덮개 닫힘·Windows 업데이트 재부팅은 절전 방지로 막을 수 없으니 전원 설정에서 "덮개를 닫을 때: 아무 것도
            안 함"과 업데이트 사용 시간(장 시간 포함)을 설정하세요.
          </p>
        </>
      )}
      <div className="actions">
        <button onClick={() => void onSave()}>저장</button>
        <button className="primary" onClick={onConnect}>
          저장 후 연결
        </button>
        <span className="muted">{msg}</span>
      </div>

      <h3>환경 점검 / 설정 가져오기</h3>
      <div className="actions">
        <button disabled={busy || status.conn !== 'connected'} onClick={() => void runDoctor()}>
          두 호스트 환경 비교 (doctor)
        </button>
        <button disabled={!form.host} onClick={() => void importKis()} title="라즈베리파이 ~/KIS/config/kis_devlp.yaml -> %USERPROFILE%\KIS\config">
          KIS 설정 가져오기 (라즈베리파이 → 이 PC)
        </button>
        {busy && <i className="spinner" />}
        <span className="muted small">{kisMsg}</span>
      </div>
      {doc && <DoctorTable a={doc.self} b={doc.peer} aErr={doc.selfErr} bErr={doc.peerErr} aLabel={status.mode === 'local' ? '이 PC' : '라즈베리파이'} bLabel={doc.peerLabel || '다른 호스트'} />}
      {local && (
        <details className="small">
          <summary>이 PC 처음 준비 (한 번만)</summary>
          <ol>
            <li>Python 3.12 설치 (r003 은 3.12 문법을 씁니다): <code>winget install Python.Python.3.12</code></li>
            <li>
              venv: <code>py -3.12 -m venv {form.localRoot || 'D:\\make_utils\\Trading'}\venv</code> → <code>venv\Scripts\python -m pip install "pandas&lt;3" numpy requests pyyaml websockets pycryptodome holidays lightgbm numba</code>
            </li>
            <li>
              KIS 공식 코드: <code>git clone https://github.com/koreainvestment/open-trading-api {form.localRoot || 'D:\\make_utils\\Trading'}\open-trading-api</code> (라즈베리파이와 같은 커밋인지 환경 비교로 확인)
            </li>
            <li>KIS 설정: 위 [KIS 설정 가져오기] (앱키/시크릿/계좌가 든 kis_devlp.yaml)</li>
            <li>데이터: 선별(g002)은 지난 거래일들의 runtime/data 가 필요합니다 - 첫날은 [데이터 작업] 탭에서 며칠치를 수집하거나 라즈베리파이의 runtime/data 를 복사하세요.</li>
          </ol>
        </details>
      )}
    </section>
  );
}

function DoctorTable({ a, b, aErr, bErr, aLabel, bLabel }: { a?: DoctorReport | null; b?: DoctorReport | null; aErr?: string; bErr?: string; aLabel: string; bLabel: string }) {
  const rows: [string, (d: DoctorReport) => string][] = [
    ['호스트', (d) => `${d.host.hostname} (${d.host.platform})`],
    ['역할', (d) => d.host.role.role + (d.host.role.set ? '' : ' (파일 없음)')],
    ['Python', (d) => d.python],
    ['코드 커밋', (d) => d.code.commit || '-'],
    ['open-trading-api', (d) => (d.open_trading_api.exists ? d.open_trading_api.commit || '(git 아님)' : '없음')],
    ['KIS 설정', (d) => (d.kis_config.devlp_yaml ? '있음' : '없음') + (d.kis_config.token_today ? ' · 오늘 토큰 있음' : '')],
    ...Object.keys({ ...(a?.packages || {}), ...(b?.packages || {}) }).map((k): [string, (d: DoctorReport) => string] => [k, (d) => d.packages[k] || '없음']),
  ];
  const major = (v: string) => v.split('.').slice(0, 2).join('.');
  return (
    <>
      <table className="grid compact">
        <thead>
          <tr>
            <th>항목</th>
            <th>{aLabel}</th>
            <th>{bLabel}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, f]) => {
            const va = a ? f(a) : aErr || '-';
            const vb = b ? f(b) : bErr || '-';
            const diff = !!a && !!b && k !== '호스트' && k !== '역할' && k !== 'KIS 설정' && major(va) !== major(vb);
            return (
              <tr key={k} className={diff ? 'nohover warn-row' : 'nohover'}>
                <td>{k}</td>
                <td className="mono">{va}</td>
                <td className="mono">{vb}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {[a, b].map((d, i) =>
        d?.problems?.length ? (
          <p key={i} className="err-text small">
            {i === 0 ? aLabel : bLabel}: {d.problems.join(' / ')}
          </p>
        ) : null,
      )}
      <p className="muted small">노란 줄 = 두 호스트의 버전(앞 두 자리)이 다름. 전략 결과가 같으려면 코드 커밋과 pandas/numpy 버전을 맞추는 것이 좋습니다.</p>
    </>
  );
}

/** "매매 시간 (서버 r003)" - ENABLE_NXT_SESSION toggle */
function NxtSection() {
  const connected = useStore((s) => s.status.conn === 'connected');
  const nxt = useStore((s) => s.control.nxt);
  const running = useStore((s) => !!s.control.status?.running);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');

  const change = async (on: boolean) => {
    if (!nxt || busy) return;
    setBusy(true);
    setMsg('저장 중…');
    // back to the r001 value = remove the override (no leftover key); otherwise store the override
    const r = on === nxt.base ? await window.api.ctlUnset([NXT_KEY]) : await window.api.ctlSet({ [NXT_KEY]: on });
    setBusy(false);
    bumpHistory();
    setMsg(r.ok ? `저장했습니다.${running ? ' r003을 재시작하면 반영됩니다.' : ' 다음 r003 실행부터 적용됩니다.'}` : `실패: ${r.error}`);
  };
  const onOff = (v: boolean | null | undefined) => (v === true ? '켬' : v === false ? '끔' : '-');

  return (
    <section>
      <h3>매매 시간 (서버 r003)</h3>
      {!connected && <p className="muted">서버에 연결되어 있을 때만 바꿀 수 있습니다.{nxt ? ` (마지막으로 읽은 값: ${onOff(nxt.effective)})` : ''}</p>}
      <div className="actions">
        <label className="check" title="r001 ENABLE_NXT_SESSION">
          <input type="checkbox" checked={!!nxt?.effective} disabled={!connected || !nxt || busy} onChange={(e) => void change(e.target.checked)} /> NXT 세션 운용
          (오전 08:00~08:50, 오후 15:30~20:00)
        </label>
        {nxt && (
          <span className="muted small">
            저장값 {onOff(nxt.effective)}
            {nxt.has_override ? ` (r001 기본 ${onOff(nxt.base)}에서 변경)` : ' (r001 기본값)'} · 실행 중 r003 {running ? onOff(nxt.running) : '정지'}
            {nxt.pending && running && <b className="pill pend">재시작 필요</b>}
          </span>
        )}
        <span className="muted small">{msg}</span>
      </div>
      <p className="muted small">
        끄면 정규장(09:00~15:30)만 매매합니다: 15:20 신규 진입 마감·당일 매수분 청산, 15:30 종료. 켜면 당일 매수분 중 NXT 거래 가능 종목은 15:20 청산 대신
        19:59:58에 청산하고(NXT 불가 종목은 15:20) 20:00에 종료합니다. 서버{' '}
        <code>live_overrides.json</code>에 저장되며 r003을 다시 시작해야 반영됩니다.
      </p>
    </section>
  );
}

/** "고급: 전체 설정값" - every editable r001 constant (`schema --all`), collapsed by default */
function AdvancedValues() {
  const connected = useStore((s) => s.status.conn === 'connected');
  const historyRev = useStore((s) => s.historyRev);
  const [open, setOpen] = useState(false);
  const [doc, setDoc] = useState<SchemaDoc | null>(null);
  const [err, setErr] = useState('');
  const [q, setQ] = useState('');
  const [onlyOverride, setOnlyOverride] = useState(false);

  useEffect(() => {
    if (!open || !connected) return;
    let alive = true;
    setErr('불러오는 중…');
    void window.api.ctlSchema(true).then((r) => {
      if (!alive) return;
      if (r.ok && r.data) {
        setDoc(r.data);
        setErr('');
      } else setErr(`불러오기 실패: ${r.error}`);
    });
    return () => {
      alive = false;
    };
  }, [open, connected, historyRev]);

  const all: EditItem[] = useMemo(() => (doc?.keys || []).map((k) => ({ ...k, editable: k.editable !== false, current: k.effective })), [doc]);
  const items = useMemo(() => {
    const words = q.trim().toUpperCase().split(/\s+/).filter(Boolean);
    return all.filter((k) => (!onlyOverride || k.has_override) && words.every((w) => k.key.includes(w)));
  }, [all, q, onlyOverride]);
  const ed = useValueEditing(all);
  const shown = items.slice(0, 200);

  return (
    <section>
      <details onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
        <summary>
          <h3 className="inline">고급: 전체 설정값 (r001 상수{doc ? ` ${doc.keys.length}개` : ''})</h3>
        </summary>
        {!connected && <p className="muted">서버에 연결되어 있을 때만 볼 수 있습니다.</p>}
        {connected && (
          <>
            <div className="actions">
              <input className="search" placeholder="키 검색 (예: PROFIT LOCK)" value={q} onChange={(e) => setQ(e.target.value)} />
              <label className="check">
                <input type="checkbox" checked={onlyOverride} onChange={(e) => setOnlyOverride(e.target.checked)} /> 오버라이드된 키만
              </label>
              <span className="muted small">
                {items.length}개{items.length > shown.length ? ` 중 ${shown.length}개 표시 - 검색어로 좁히세요` : ''} {err}
              </span>
            </div>
            {(doc?.errors?.length || 0) > 0 && <p className="err-text small">{doc!.errors!.join(' / ')}</p>}
            <div className="adv-table">
              <ValuesEditor items={shown} ed={ed} disabled={!connected} />
            </div>
            <EditorActions ed={ed} disabled={!connected} />
            <p className="muted small">
              값은 서버 <code>{doc?.overrides_file || 'runtime/config/live_overrides.json'}</code>에 저장되고, r003을 재시작해야 반영됩니다. 퍼센트 상수(*_PCT)는 비율로
              입력합니다 (0.004 = 0.4%).
            </p>
          </>
        )}
      </details>
    </section>
  );
}
