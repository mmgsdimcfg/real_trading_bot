/*
 * Update log
 * - [2026-11-16] type=feat - New 수집기 tab (tick collector g011 of the connected host, r016 collector detail / start / restart /
 *   stop / extra / prune): state + actions, the settings that decide what is collected, the day's subscription plan and one row per
 *   symbol (role 감시 / 추가 / 스캔 채움, KRX / NXT prints, first / last print, idle time - stale rows marked), the extra codes the
 *   user adds (next start), the daily history with disk use and a prune of old days (preview, then confirm; at least 30 days kept),
 *   the collector's own log events, and what to know about the collector. Past days are read-only views of their folder.
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { DatePicker } from '../components/DatePicker';
import { fmtBytes, localYmd, parseTickCodes, type CollectorDetail, type CollectorPrune, type CollectorRole, type CollectorSymbol } from '../../../shared/control';

const ROLE_LABEL: Record<CollectorRole, string> = { watch: '감시', extra: '추가', scan: '스캔 채움', file_only: '파일만' };
const ROLE_TITLE: Record<CollectorRole, string> = {
  watch: 'r004 감시 종목 - 봇의 틱 트리거가 이 파일을 읽습니다 (항상 먼저 구독)',
  extra: '이 탭에서 추가한 수집 종목',
  scan: '남는 자리를 최근 g002 스캔 상위 후보로 채운 종목 (학습용)',
  file_only: '이번 실행 목록에는 없지만 오늘 파일이 있는 종목 (이전 실행에서 수집)',
};
const n = (v: number | null | undefined) => (v === null || v === undefined ? '-' : Math.round(v).toLocaleString('ko-KR'));
const hhmm = (t: string | null | undefined) => (t ? `${t.slice(0, 2)}:${t.slice(2, 4)}:${t.slice(4, 6)}` : '-');
const ymd = (d: string) => `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
const isoTime = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(5, 19) : '-');
function ago(sec: number | null): string {
  if (sec === null || sec === undefined) return '-';
  if (sec < 60) return `${sec}초 전`;
  if (sec < 3600) return `${Math.floor(sec / 60)}분 전`;
  return `${Math.floor(sec / 3600)}시간 전`;
}
const cfgText = (v: unknown) => (v === null || v === undefined ? '-' : typeof v === 'boolean' ? (v ? '켜짐' : '꺼짐') : Array.isArray(v) ? v.join(', ') : String(v));

export function CollectorView() {
  const connected = useStore((s) => s.status.conn === 'connected');
  const local = useStore((s) => s.status.mode === 'local');
  const busyCtl = useStore((s) => s.control.busy);
  const [date, setDate] = useState(localYmd());
  const [d, setD] = useState<CollectorDetail | null>(null);
  const [err, setErr] = useState('');
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const [roleFilter, setRoleFilter] = useState<'all' | CollectorRole | 'stale'>('all');
  const [extraText, setExtraText] = useState('');
  const [extraDirty, setExtraDirty] = useState(false);
  const [keepDays, setKeepDays] = useState('60');
  const [prune, setPrune] = useState<CollectorPrune | null>(null);
  const isToday = date === localYmd();
  const host = local ? '이 PC' : '라즈베리파이';

  const load = async () => {
    const r = await window.api.ctlCollectorDetail(date);
    if (r.ok && r.data) {
      setD(r.data);
      setErr('');
      if (!extraDirty) setExtraText(r.data.extra.codes.join(', '));
    } else setErr(r.error || '불러오지 못했습니다');
  };
  useEffect(() => {
    if (!connected) return;
    void load();
    if (!isToday) return;
    const t = setInterval(() => void load(), 15000);
    return () => clearInterval(t);
  }, [connected, date]); // eslint-disable-line react-hooks/exhaustive-deps

  const act = async (label: string, fn: () => Promise<{ ok: boolean; error?: string; cancelled?: boolean; data?: unknown }>, done: (data: unknown) => string) => {
    setBusy(true);
    setMsg(`${label} 중…`);
    const r = await fn();
    setBusy(false);
    if (r.cancelled) return setMsg('');
    setMsg(r.ok ? done(r.data) : `${label} 실패: ${r.error}`);
    await load();
  };
  const start = () => act('시작', () => window.api.ctlCollectorStart(false), () => '시작했습니다');
  const restart = () =>
    act('재시작', () => window.api.ctlCollectorRestart(), (x) => {
      const r = x as { restarted?: boolean; window?: string } | undefined;
      return r?.restarted ? '재시작했습니다 (추가 종목·감시 목록이 새로 반영됨)' : `멈췄습니다 - 수집 시간이 아니라 다시 띄우지 않았습니다 (${r?.window || ''})`;
    });
  const stop = () => {
    if (!window.confirm('틱 수집기를 멈출까요? 수집이 멈춘 동안의 틱은 나중에 받을 수 없습니다.\n(r003 이 실행 중이면 1분 안에 다시 시작됩니다)')) return;
    void act('중지', () => window.api.ctlCollectorStop(), (x) => {
      const r = x as { note?: string | null } | undefined;
      return r?.note ? `멈췄습니다 - ${r.note.includes('r003') ? 'r003 이 실행 중이라 1분 안에 다시 시작됩니다' : r.note}` : '멈췄습니다';
    });
  };

  const parsed = useMemo(() => parseTickCodes(extraText), [extraText]);
  const saveExtra = async (thenRestart: boolean) => {
    if (parsed.bad.length) return setMsg(`종목코드 형식 오류: ${parsed.bad.join(', ')}`);
    setBusy(true);
    const r = await window.api.ctlCollectorExtra(parsed.codes);
    setBusy(false);
    if (!r.ok) return setMsg(`추가 종목 저장 실패: ${r.error}`);
    setExtraDirty(false);
    setMsg(`추가 종목 ${parsed.codes.length}개 저장했습니다${thenRestart ? '' : ' - 다음 수집기 시작부터 적용됩니다'}`);
    if (thenRestart) await restart();
    else await load();
  };

  const previewPrune = async () => {
    setBusy(true);
    const r = await window.api.ctlCollectorPrune(Number(keepDays), false);
    setBusy(false);
    if (!r.ok || !r.data) return setMsg(`정리 계획 실패: ${r.error}`);
    setPrune(r.data);
    setMsg('');
  };
  const applyPrune = async () => {
    if (!prune || !prune.delete.length) return;
    const list = prune.delete.map((x) => ymd(x.date)).join(', ');
    if (!window.confirm(`틱 데이터 ${prune.delete.length}일치(${fmtBytes(prune.free_bytes)})를 삭제합니다. 되돌릴 수 없습니다.\n${list}`)) return;
    setBusy(true);
    const r = await window.api.ctlCollectorPrune(prune.keep_days, true);
    setBusy(false);
    setPrune(null);
    setMsg(r.ok ? `${r.data?.removed?.length || 0}일치 삭제했습니다${r.data?.errors?.length ? ` (오류 ${r.data.errors.length})` : ''}` : `삭제 실패: ${r.error}`);
    await load();
  };

  const st = d?.status;
  const symbols = d?.symbols || [];
  const counts = useMemo(() => {
    const c: Record<string, number> = { all: symbols.length, stale: symbols.filter((s) => s.stale).length };
    for (const s of symbols) c[s.role] = (c[s.role] || 0) + 1;
    return c;
  }, [symbols]);
  const shown = symbols.filter((s) => roleFilter === 'all' || (roleFilter === 'stale' ? s.stale : s.role === roleFilter));
  const sumRows = (k: 'krx' | 'nxt') => symbols.reduce((a, s) => a + (s[k]?.rows_est || 0), 0);
  const cfg = d?.config || {};

  if (!connected) return <div className="control"><div className="empty">연결되어 있지 않습니다 - 설정/연결 탭에서 연결하세요.</div></div>;

  return (
    <div className="control collector-view">
      <section>
        <h3>틱 수집기 (g011)</h3>
        <p className="subtitle">
          {host}의 KIS 실시간 체결 수집 · 봇의 60/120틱 트리거와 틱 연구(g036/g037)의 원자료 ·{' '}
          <span className="mono">{d?.ticks_dir || ''}</span>
        </p>
        {err && <p className="err-text small">{err}</p>}
        <div className="coll-head">
          <DatePicker value={date} onChange={setDate} today={localYmd()} max={localYmd()} title="수집 날짜 (지난 날짜는 그날 폴더를 보여 줍니다)" />
          {isToday ? (
            st?.running ? (
              <span className="coll-state ok" title={`PID ${st.pid}`}>
                ● 수집 중 <span className="muted small">· 시작 {isoTime(st.started_at)} · 마지막 기록 {isoTime(st.last_write)}</span>
              </span>
            ) : (
              <span className="coll-state bad">● 멈춤 {d && !d.window.ok && <span className="muted small">· {d.window.reason}</span>}</span>
            )
          ) : (
            <span className="muted">지난 날짜 - 그날 폴더의 기록입니다</span>
          )}
          {d?.role?.role === 'standby' && <span className="warn-text small">이 호스트는 standby - 수집은 다른 호스트가 합니다 (KIS 웹소켓은 앱키당 1개)</span>}
          {(busy || busyCtl) && <i className="spinner" />}
          <span className="push-right" />
          {isToday && (
            <>
              <button className="primary" disabled={busy || !!busyCtl || !!st?.running} onClick={() => void start()} title="g011 을 빈자리 채우기(--fill-from-scan)로 띄웁니다 (수집 시간 안에서만)">
                시작
              </button>
              <button disabled={busy || !!busyCtl || !st?.running} onClick={() => void restart()} title="멈춘 뒤 현재 감시 목록·추가 종목으로 다시 띄웁니다 (몇 초간 수집 공백)">
                재시작
              </button>
              <button disabled={busy || !!busyCtl || !st?.running} onClick={stop} title="수집기를 멈춥니다 - 그동안의 틱은 나중에 받을 수 없습니다">
                중지
              </button>
            </>
          )}
          <button disabled={busy} onClick={() => void load()}>
            새로고침
          </button>
        </div>
        {msg && <p className="small muted">{msg}</p>}

        {d && (
          <div className="coll-stats">
            <div>
              <b>{n(d.plan.codes.length || symbols.length)}</b>
              <span>수집 종목</span>
              <i>
                감시 {counts.watch || 0} · 추가 {counts.extra || 0} · 스캔 {counts.scan || 0}
              </i>
            </div>
            <div>
              <b>
                {d.plan.subscribed ? `${n(d.plan.subscribed.krx)} / ${n(d.plan.subscribed.nxt)}` : '-'}
              </b>
              <span>구독 KRX / NXT</span>
              <i>{d.plan.subscribed ? `${isoTime(d.plan.subscribed.at)} 기준 · 최대 ${n(Number(cfg.max_subscriptions) || 40)}` : '로그 없음'}</i>
            </div>
            <div>
              <b>{n(sumRows('krx'))}</b>
              <span>KRX 체결 (추정)</span>
              <i>NXT {n(sumRows('nxt'))}</i>
            </div>
            <div className={counts.stale ? 'bad' : ''}>
              <b>{n(counts.stale)}</b>
              <span>끊긴 종목</span>
              <i>정규장 중 {Math.round(180 / 60)}분 이상 기록 없음</i>
            </div>
            <div className={d.plan.reconnects || d.plan.errors ? 'warn' : ''}>
              <b>
                {n(d.plan.reconnects)} / {n(d.plan.errors)}
              </b>
              <span>재접속 / 오류</span>
              <i>이번 실행 · 시작 {n(d.plan.starts)}회</i>
            </div>
            <div>
              <b>{fmtBytes(d.total_size)}</b>
              <span>틱 데이터 전체</span>
              <i>
                {n(d.day_count)}일 · 디스크 여유 {fmtBytes(d.disk?.free)}
              </i>
            </div>
          </div>
        )}
      </section>

      {d && (
        <section>
          <h3>수집 설정</h3>
          <p className="subtitle">무엇을 언제 받는지 정하는 값 (r001) · 바꾸려면 파이프라인 탭 설정값 또는 r016 set, r003 재시작 후 적용</p>
          <table className="grid coll-cfg">
            <tbody>
              <tr className="nohover">
                <td>자동 기동</td>
                <td className="mono">{cfgText(cfg.TICK_COLLECTOR_AUTOSTART)}</td>
                <td className="muted wrap">r003 이 시작할 때 수집기를 띄우고 1분마다 살아 있는지 확인 (TICK_COLLECTOR_AUTOSTART)</td>
              </tr>
              <tr className="nohover">
                <td>빈자리 채우기</td>
                <td className="mono">{cfgText(cfg.TICK_COLLECTOR_FILL_FROM_SCAN)}</td>
                <td className="muted wrap">감시·추가 종목 뒤 남는 자리를 최근 스캔 상위 후보로 40까지 (TICK_COLLECTOR_FILL_FROM_SCAN) · 스캔 {d.scan_date ? ymd(d.scan_date) : '없음'}</td>
              </tr>
              <tr className="nohover">
                <td>NXT 수집</td>
                <td className="mono">
                  {cfgText(cfg.TICK_COLLECT_NXT)} · {String(cfg.TICK_COLLECT_NXT_REGULAR || 'all') === 'held' ? '정규장엔 보유만' : '정규장엔 남는 자리'}
                </td>
                <td className="muted wrap">
                  {String(cfg.MORNING_NXT_START || '').slice(0, 5)}~{String(cfg.AFTERNOON_NXT_END || '').slice(0, 5)} · 장전·장후는 감시 종목 전부, 정규장(개장 5분 전~)은 KRX 우선
                  {d.plan.nxt_keep.length ? ` · 지금 정규장 NXT 유지: ${d.plan.nxt_keep.join(', ')}` : ''} (TICK_COLLECT_NXT / _REGULAR)
                </td>
              </tr>
              <tr className="nohover">
                <td>KRX 종료</td>
                <td className="mono">{cfgText(cfg.TICK_COLLECTOR_UNTIL)}</td>
                <td className="muted wrap">r003 이 띄운 수집기의 KRX 구독 종료 시각 (TICK_COLLECTOR_UNTIL) · 틱봉 {cfgText(cfg.TICK_TRIGGER_BAR_SIZES)}</td>
              </tr>
            </tbody>
          </table>
        </section>
      )}

      {d && (
        <section>
          <h3>수집 종목</h3>
          <p className="subtitle">
            {d.plan.start ? `${isoTime(d.plan.start)} 시작한 실행의 구독 목록` : '수집 로그 없음'}
            {d.plan.start_note ? ` · ${d.plan.start_note.replace(/ out=.*$/, '')}` : ''}
          </p>
          <div className="seg small-seg">
            {(['all', 'watch', 'extra', 'scan', 'file_only', 'stale'] as const).map((k) => (
              <button key={k} className={roleFilter === k ? 'on' : ''} onClick={() => setRoleFilter(k)}>
                {k === 'all' ? '전체' : k === 'stale' ? '끊김' : ROLE_LABEL[k]} {counts[k] || 0}
              </button>
            ))}
          </div>
          <table className="grid">
            <thead>
              <tr>
                <th>종목</th>
                <th>구분</th>
                <th className="num">KRX 체결</th>
                <th>첫 / 마지막 체결</th>
                <th>마지막 기록</th>
                <th className="num">크기</th>
                <th className="num">NXT 체결</th>
                <th>NXT 첫 / 마지막</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s) => (
                <SymbolRow key={s.code} s={s} isToday={isToday} />
              ))}
              {!shown.length && (
                <tr className="nohover">
                  <td colSpan={8} className="muted">
                    해당 종목 없음
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      )}

      {d && (
        <section>
          <h3>추가 수집 종목</h3>
          <p className="subtitle">
            감시 종목 외에 매일 받고 싶은 종목 (최대 40, 감시 종목 다음 순서로 구독 → 남는 자리를 스캔 후보가 채움) · 다음 수집기 시작부터 적용
            {d.extra.updated_at ? ` · 마지막 저장 ${isoTime(d.extra.updated_at)}` : ''}
          </p>
          <div className="coll-extra">
            <textarea
              className="mono"
              rows={2}
              spellCheck={false}
              placeholder="005930, 000660 …"
              value={extraText}
              onChange={(e) => {
                setExtraText(e.target.value);
                setExtraDirty(true);
              }}
            />
            <div className="coll-extra-actions">
              <span className={parsed.bad.length ? 'err-text small' : 'muted small'}>
                {parsed.codes.length}종목{parsed.bad.length ? ` · 형식 오류 ${parsed.bad.join(', ')}` : ''}
                {parsed.codes.length + d.watch.length > 40 ? ' · 감시 종목과 합쳐 40을 넘는 종목은 구독되지 않습니다' : ''}
              </span>
              <button disabled={busy || !extraDirty || !!parsed.bad.length} onClick={() => void saveExtra(false)}>
                저장
              </button>
              <button
                className="primary"
                disabled={busy || !!busyCtl || !!parsed.bad.length || !isToday || !st?.running}
                title="저장하고 수집기를 재시작해 바로 반영 (몇 초간 수집 공백)"
                onClick={() => void saveExtra(true)}
              >
                저장 후 재시작
              </button>
            </div>
          </div>
        </section>
      )}

      {d && (
        <section>
          <h3>일별 수집 이력</h3>
          <p className="subtitle">
            전체 {n(d.day_count)}일 · {fmtBytes(d.total_size)} · 디스크 여유 {fmtBytes(d.disk?.free)} / {fmtBytes(d.disk?.total)} · 하루 40종목이면 약 70~100MB
          </p>
          <table className="grid coll-days">
            <thead>
              <tr>
                <th>날짜</th>
                <th className="num">KRX 종목</th>
                <th className="num">NXT 종목</th>
                <th className="num">크기</th>
              </tr>
            </thead>
            <tbody>
              {d.days.map((x) => (
                <tr key={x.date} className={x.date === date ? 'sel' : ''} onClick={() => setDate(x.date)} title="누르면 그날 수집 종목을 봅니다">
                  <td className="mono">{ymd(x.date)}</td>
                  <td className={`num${x.krx < 20 ? ' warn-text' : ''}`}>{x.krx}</td>
                  <td className="num">{x.nxt}</td>
                  <td className="num">{fmtBytes(x.size)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="coll-prune">
            <span>오래된 날짜 정리: 최근</span>
            <input type="number" min={d.prune_min_keep_days} value={keepDays} onChange={(e) => { setKeepDays(e.target.value); setPrune(null); }} style={{ width: 90 }} />
            <span>일만 남기기 (최소 {d.prune_min_keep_days}일, 오늘 제외)</span>
            <button disabled={busy} onClick={() => void previewPrune()}>
              미리보기
            </button>
            {prune && (
              <>
                <span className="muted small">
                  {prune.delete.length ? `${prune.delete.length}일치 삭제 · ${fmtBytes(prune.free_bytes)} 확보 (${ymd(prune.delete[prune.delete.length - 1].date)} ~ ${ymd(prune.delete[0].date)})` : '삭제할 날짜 없음'}
                </span>
                {prune.delete.length > 0 && (
                  <button className="danger-text" disabled={busy} onClick={() => void applyPrune()}>
                    삭제
                  </button>
                )}
              </>
            )}
          </div>
          <p className="muted small">틱은 KIS 에서 지난날 것을 다시 받을 수 없습니다 - 삭제한 날짜는 연구에 다시 쓸 수 없습니다. 이 PC 디스크가 충분하면 정리하지 않아도 됩니다.</p>
        </section>
      )}

      {d && (
        <section>
          <h3>수집기 로그</h3>
          <p className="subtitle">
            {ymd(d.date)} <span className="mono">_collector.log</span> 의 시작·연결·구독·재접속·오류 (최근 {d.events.length}줄)
          </p>
          <div className="coll-log">
            {[...d.events].reverse().map((e, i) => (
              <div key={i} className={`ev ${e.kind === 'ERROR' ? 'err' : e.kind === 'WARN' || e.kind === 'RECONNECT' ? 'warn' : ''}`}>
                <span className="mono">{e.ts.slice(11)}</span> <b>{e.kind}</b> <span className="mono">{e.text}</span>
              </div>
            ))}
            {!d.events.length && <p className="muted">로그 없음</p>}
          </div>
        </section>
      )}

      <section>
        <h3>알아 둘 점</h3>
        <ul className="coll-notes">
          <li>KIS 실시간 연결은 앱키당 1개, 구독은 최대 40개입니다. 감시 종목이 항상 먼저 구독되고, 남는 자리를 추가 종목 → 스캔 후보 순으로 채웁니다.</li>
          <li>NXT 체결도 같은 40자리를 씁니다. 정규장(개장 5분 전~15:31)에는 KRX 를 우선하고 NXT 는 보유 종목만 받습니다. 장전·장후에는 감시 종목 NXT 를 받습니다.</li>
          <li>봇의 60/120틱 트리거는 감시 종목의 KRX 틱 파일을 읽습니다. 감시 종목이 끊기면(위 "끊김") 그 종목은 3중 트리거를 통과할 수 없습니다.</li>
          <li>수집기는 시작할 때의 목록만 구독합니다. 감시 목록이나 추가 종목을 바꾸면 재시작해야 반영됩니다 (감시 목록 저장 시에는 앱이 자동 재시작).</li>
          <li>틱은 장중 실시간으로만 쌓입니다. KIS 는 지난날 틱을 주지 않으므로 수집기가 멈춘 시간의 틱은 되찾을 수 없습니다.</li>
          <li>standby 호스트에서는 수집기를 시작할 수 없습니다. 매매하는 호스트(primary)에서만 수집합니다.</li>
          <li>KRX 체결 수는 파일 크기로 추정한 값입니다. 끊김 판정은 정규장 중 3분 이상 기록이 없는 경우입니다.</li>
        </ul>
      </section>
    </div>
  );
}

function SymbolRow({ s, isToday }: { s: CollectorSymbol; isToday: boolean }) {
  return (
    <tr className={`nohover${s.stale ? ' warn-row' : ''}`}>
      <td>
        <span className="mono">{s.code}</span> {s.name}
        {s.nxt_keep && <span className="code-tag" title="보유 종목 - 정규장에도 NXT 체결을 받습니다">NXT 유지</span>}
      </td>
      <td title={ROLE_TITLE[s.role]}>
        <span className={`role-tag ${s.role}`}>{ROLE_LABEL[s.role]}</span>
      </td>
      <td className="num">{s.krx ? n(s.krx.rows_est) : <span className="muted">파일 없음</span>}</td>
      <td className="mono">{s.krx ? `${hhmm(s.krx.first)} / ${hhmm(s.krx.last)}` : '-'}</td>
      <td title={s.krx?.mtime || ''}>{s.krx ? (isToday ? ago(s.idle_s) : isoTime(s.krx.mtime)) : '-'}{s.stale ? ' · 끊김' : ''}</td>
      <td className="num">{fmtBytes(s.krx?.size)}</td>
      <td className="num">{s.nxt ? n(s.nxt.rows_est) : '-'}</td>
      <td className="mono">{s.nxt ? `${hhmm(s.nxt.first)} / ${hhmm(s.nxt.last)}` : '-'}</td>
    </tr>
  );
}
