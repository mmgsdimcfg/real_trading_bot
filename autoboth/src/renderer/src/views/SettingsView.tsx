/*
 * Update log
 * - [2026-11-04] type=feat - Settings / connection: SSH (host, port, user, key file + passphrase or password - stored encrypted by
 *   the main process), remote root, python path, poll interval, session open time, cache retention; date selection (server folders
 *   and local cache), local folder (fixtures), condition JSON refresh, and the list of unknown tags/reasons
 * - [2026-11-05] type=feat - SSH proxy option (same as TermH Add/Edit Host): 프록시 select (사용 안 함 / ProxyCommand / SOCKS5 / HTTP CONNECT)
 *   with the ProxyCommand line + %h/%p/%r hint, or proxy host/port/user/password (password stored encrypted like the SSH secret)
 * - [2026-11-05] type=feat - "고급: 전체 설정값" (collapsed): `schema --all` (~380 r001 constants) with a search box and the same value
 *   editor as the pipeline panel (only changed keys are sent in one `set`); the read-only note now lists the r016 control calls
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import type { Settings } from '../../../shared/types';
import { fmtInt } from '../lib';
import { ymdDash } from '../../../shared/time';
import type { SchemaDoc } from '../../../shared/control';
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
  const [date, setDate] = useState('');

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
    const r = await window.api.connect();
    setMsg(r.ok ? '연결했습니다.' : `연결 실패: ${r.error}`);
  };
  const allDates = [...new Set([...dates.remote, ...dates.cached])].sort().reverse();

  return (
    <div className="settings">
      <section>
        <h3>SSH 연결</h3>
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
          <label>정규장 시작</label>
          <input value={form.sessionOpen} onChange={(e) => set('sessionOpen', e.target.value)} title="이 시각 전의 반려는 '개장 전'으로 따로 집계 (깔때기에서 제외)" />
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
          정규장 시작 시각 변경은 다음 로드부터 적용됩니다.
        </p>
      </section>

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
        <p className="muted small">오늘(서버 날짜)은 {form.pollSec}초마다 증분 수신, 지난 날짜는 폴링 없이 1회 로드합니다. 로컬 폴더는 runtime/log/YYYYMMDD 복사본(픽스처)을 SSH 없이 엽니다.</p>
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
