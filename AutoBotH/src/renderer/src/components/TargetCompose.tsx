/*
 * Update log
 * - [2026-11-27] type=feat - 대상 구성 (user 2026-11-27): the target symbols of a trading method composed from the scan - A (the
 *   scanner's picks) + the scan ranking up to the method's size (rule 30, Edge AI 50), VWAP = the union - with the user's
 *   always-include / always-exclude lists (kept from day to day). This card shows the composition the server makes (r016
 *   `targets get`: source, scan rank, score, eligible flag, fail reasons, tick coverage of every symbol), stores the edits
 *   (`targets config`) and applies it (`targets apply`: r004 + the three target lists, confirmed in a native dialog, effective
 *   after an r003 restart). Nothing is composed or judged in the app.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { fmtInt } from '../lib';
import { TRADE_STRATEGY_KO, type TradeStrategy } from '../../../shared/strategy';
import { METHOD_TARGET_KEYS, TARGET_SRC_KO, targetsApplied, type TargetsDoc } from '../../../shared/methodUnits';

const SIZE_KEY: Partial<Record<TradeStrategy, 'rule' | 'edge'>> = { rule: 'rule', edge: 'edge' };
const HOW: Record<TradeStrategy, string> = {
  rule: 'A(선별 종목) + 스캔 순위 다음 종목으로 채웁니다',
  edge: 'A(선별 종목) + 규칙 추가분 다음 순위 종목으로 채웁니다',
  vwap: 'A + 규칙 추가 종목 + Edge AI 추가 종목 (합집합)',
};

export function TargetCompose(props: { method: TradeStrategy; enabled: boolean; hostKey: string; onApplied: () => void }) {
  const { method, enabled } = props;
  const [doc, setDoc] = useState<TargetsDoc | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [code, setCode] = useState('');
  const [size, setSize] = useState('');
  const [msg, setMsg] = useState('');

  const load = useCallback(async (): Promise<TargetsDoc | null> => {
    const r = await window.api.targetsGet();
    if (r.data) {
      setDoc(r.data);
      setErr(r.data.ok ? '' : r.data.error || '');
      return r.data;
    }
    setErr(r.error || '구성을 읽지 못했습니다.');
    return null;
  }, []);
  useEffect(() => {
    setDoc(null);
    setErr('');
    if (enabled) void load();
  }, [enabled, props.hostKey, load]);
  useEffect(() => {
    setMsg('');
    setCode('');
    setSize('');
  }, [method]);

  const cfg = doc?.config;
  const comp = doc?.composition;
  const list = comp?.lists[method] || [];
  const noTick = useMemo(() => new Set(comp?.no_tick || []), [comp]);
  const sizeKey = SIZE_KEY[method];
  const applied = targetsApplied(doc, method);
  const allApplied = doc && comp ? (['rule', 'edge', 'vwap'] as const).every((m) => targetsApplied(doc, m)) && comp.watch.map((w) => w.code).join(',') === doc.watch_now.join(',') : null;
  const counts = useMemo(() => {
    if (!comp) return null;
    const rows = list.map((c) => comp.rows[c]).filter(Boolean);
    return {
      a: rows.filter((r) => r.src === 'A').length,
      extra: rows.filter((r) => r.src === 'rule' || r.src === 'edge').length,
      manual: rows.filter((r) => r.src === 'manual').length,
      eligible: rows.filter((r) => r.eligible).length,
      noTick: list.filter((c) => noTick.has(c)).length,
    };
  }, [comp, list, method, noTick]);

  /** stores a config change; the fresh document, or null when it was not stored */
  const change = async (patch: Record<string, unknown>, done: string): Promise<TargetsDoc | null> => {
    if (!enabled || busy) return null;
    setBusy(true);
    setMsg('');
    try {
      const r = await window.api.targetsConfig(patch);
      if (!r.ok) {
        setMsg(`저장하지 못했습니다: ${r.error || r.data?.error || '알 수 없는 오류'}`);
        return null;
      }
      setMsg(done);
      return await load();
    } finally {
      setBusy(false);
    }
  };
  // one code at a time, applied by the server on the lists stored now (an edit made elsewhere is not overwritten)
  const exclude = (c: string) => void change({ exclude_add: { [method]: [c] } }, `${c} 제외 (구성 적용 전까지는 저장만 된 상태)`);
  const restore = (c: string) => void change({ exclude_remove: { [method]: [c] } }, `${c} 제외 해제`);
  const include = () => {
    const c = code.trim().toUpperCase();
    if (!cfg || !/^[0-9A-Z]{6}$/.test(c)) return setMsg('종목코드 6자리를 넣으세요.');
    setCode('');
    void change({ include_add: { [method]: [c] } }, `${c} 항상 포함 (구성 적용 전까지는 저장만 된 상태)`);
  };
  const uninclude = (c: string) => void change({ include_remove: { [method]: [c] } }, `${c} 포함 해제`);
  const saveSize = () => {
    const n = Number(size);
    if (!sizeKey || !Number.isInteger(n) || n < 1 || n > 100) return setMsg('종목 수는 1~100 사이 정수입니다.');
    setSize('');
    void change({ sizes: { [sizeKey]: n } }, `목표 종목 수 ${n}`);
  };
  const setAuto = (on: boolean) => {
    if (
      on &&
      !window.confirm(
        '자동 구성을 켭니다.\n\n매일 장 마감 작업이 선별(g002) 뒤에 이 구성으로 감시목록과 매매법별 대상을 다시 만듭니다 (감시 종목이 A 만이 아니라 순위 추가 종목까지 늘어납니다).\n' +
          '순위로 추가되는 종목은 대부분 선별에서 탈락한 종목입니다. 아래 구성 목록에서 확인하세요.\n\n켤까요?',
      )
    )
      return;
    if (on) return void change({ auto: true }, '자동 구성 켬 (다음 장 마감 작업부터)');
    void (async () => {
      // [Codex review] only when the switch was really stored, and from the lists stored NOW (not what the card read earlier)
      const fresh = await change({ auto: false }, '자동 구성 끔 (장 마감 작업은 선별 종목만 감시목록에 넣습니다)');
      if (!fresh) return;
      const doc = fresh;
      // the three stored lists stay as they are: tomorrow's picks outside them would not be bought
      const kept = (['rule', 'edge', 'vwap'] as const).filter((m) => (doc.stored[m] || []).length > 0);
      if (!kept.length) return;
      if (
        !window.confirm(
          `저장된 대상 목록이 남아 있습니다 (${kept.map((m) => `${TRADE_STRATEGY_KO[m]} ${doc.stored[m].length}종목`).join(', ')}).\n\n` +
            '그대로 두면 장 마감 작업이 넣는 새 선별 종목 중 이 목록에 없는 종목은 사지 않습니다.\n"전체(지정 없음)"로 되돌릴까요? (r003 재시작 후 적용)',
        )
      )
        return setMsg('자동 구성 끔 - 저장된 대상 목록은 그대로 둡니다 (새 선별 종목이 목록에 없으면 사지 않습니다)');
      const r = await window.api.ctlSet(Object.fromEntries(kept.map((m) => [METHOD_TARGET_KEYS[m], []])));
      setMsg(r.ok ? '자동 구성 끔 - 대상 목록을 전체(지정 없음)로 되돌렸습니다 (r003 재시작 후 적용)' : `대상 목록을 되돌리지 못했습니다: ${r.error || ''}`);
      props.onApplied();
      await load();
    })();
  };
  const apply = async () => {
    if (!enabled || busy) return;
    setBusy(true);
    setMsg('');
    try {
      const r = await window.api.targetsApply(undefined, comp?.digest);
      if (r.cancelled) return;
      if (!r.ok) setMsg(`적용하지 못했습니다: ${r.error || '알 수 없는 오류'}`);
      else {
        setMsg('적용했습니다 - 감시목록과 세 방식의 대상을 저장했습니다 (r003 재시작 후 적용)');
        props.onApplied();
      }
      await load();
    } finally {
      setBusy(false);
    }
  };

  if (!enabled) return null;
  return (
    <div className="tcompose">
      <div className="tc-head">
        <b>대상 구성</b>
        <span className="muted small">{HOW[method]}</span>
        {doc?.date && <span className="muted small">· 스캔 {doc.date.slice(0, 4)}-{doc.date.slice(4, 6)}-{doc.date.slice(6, 8)}</span>}
        <button type="button" className="chip link push-right" onClick={() => setOpen((o) => !o)}>
          {open ? '구성 목록 닫기 ▲' : '구성 목록 보기 ▼'}
        </button>
      </div>
      {err && <p className="badge-warn">{err}</p>}
      {doc?.config_state === 'broken' && <p className="badge-warn">구성 설정 파일을 읽을 수 없습니다 (자동 구성은 동작하지 않습니다).</p>}
      {comp && counts && cfg && (
        <>
          <div className="tc-line">
            <span>
              <b>{list.length}종목</b> = A {counts.a} + 순위 추가 {counts.extra}
              {counts.manual ? ` + 수동 포함 ${counts.manual}` : ''}
              {comp.excluded[method].length ? ` (제외 ${comp.excluded[method].length})` : ''}
            </span>
            <span className={counts.eligible < list.length ? 'badge-warn' : 'muted small'} title="스캔에서 선별 적격(eligible)으로 판정된 종목 수. 순위로 추가된 종목은 대부분 선별에서 탈락한 종목입니다">
              선별 적격 {counts.eligible} / {list.length}
            </span>
            {counts.noTick > 0 && (
              <span className="badge-warn" title={`틱 수집기는 감시목록 앞에서부터 ${comp.tick_cap}종목만 구독합니다 (적용 후 수집기 재시작 기준, 실제 구독은 수집기 탭에서 확인). 그 밖의 종목은 틱 신호가 없어 규칙 매매와 VWAP 매매가 사지 못합니다 (Edge AI 만 가능)`}>
                틱 수집 밖 {counts.noTick}
              </span>
            )}
            {sizeKey && comp.short[sizeKey] > 0 && <span className="badge-warn">순위 종목 부족 {comp.short[sizeKey]}</span>}
            {comp.dropped.length > 0 && (
              <span className="badge-warn" title={`감시목록 상한(100종목)을 넘어 구성에서 빠진 종목: ${comp.dropped.join(', ')}`}>
                상한 초과로 빠짐 {comp.dropped.length}
              </span>
            )}
            <span className={applied ? 'muted small' : 'badge-warn'}>{applied ? '저장된 대상과 같음' : '저장된 대상과 다름 (적용 전)'}</span>
          </div>
          <div className="tc-line muted small">
            감시목록 전체 {comp.watch.length}종목
            {doc.active_size ? ` · r003 동시 감시 ${Math.min(doc.active_size, comp.watch.length)}종목${comp.watch.length > doc.active_size ? ` (나머지 ${comp.watch.length - doc.active_size}종목은 대기)` : ''}` : ''}
            {' · '}감시 종목이 늘면 한 바퀴 도는 시간이 길어집니다 (종목당 약 0.6초)
          </div>
          <div className="tc-line">
            <label className="check" title="켜면 매일 장 마감 작업이 이 구성으로 감시목록(r004)과 세 방식의 대상을 다시 만듭니다. 끄면 선별 종목만 감시목록에 넣습니다(지금까지와 같음)">
              <input type="checkbox" checked={cfg.auto} disabled={busy} onChange={(e) => setAuto(e.target.checked)} /> 매일 자동 구성
            </label>
            {sizeKey && (
              <span className="tc-size">
                목표 {cfg.sizes[sizeKey]}종목
                <input type="number" min={1} max={100} placeholder="변경" value={size} disabled={busy} onChange={(e) => setSize(e.target.value)} />
                <button type="button" disabled={busy || !size} onClick={saveSize}>
                  저장
                </button>
              </span>
            )}
            <span className="tc-size">
              <input type="text" maxLength={6} placeholder="종목코드" value={code} disabled={busy} onChange={(e) => setCode(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && include()} />
              <button type="button" disabled={busy || !code.trim()} onClick={include} title={`${TRADE_STRATEGY_KO[method]}에 항상 포함 (매일 유지)`}>
                항상 포함
              </button>
            </span>
            <button type="button" className="primary" disabled={busy || allApplied === true} onClick={() => void apply()} title="감시목록(r004)과 세 방식의 대상 종목을 이 구성으로 저장합니다 (확인 창, r003 재시작 후 적용)">
              {allApplied ? '구성 적용됨' : '구성 적용 (감시목록 + 대상)'}
            </button>
          </div>
          {(cfg.include[method].length > 0 || cfg.exclude[method].length > 0) && (
            <div className="tc-line">
              {cfg.include[method].length > 0 && (
                <span>
                  항상 포함:{' '}
                  {cfg.include[method].map((c) => (
                    <button key={c} type="button" className="chip" disabled={busy} title="누르면 포함을 해제합니다" onClick={() => uninclude(c)}>
                      {c} {comp.rows[c]?.name && comp.rows[c].name !== c ? comp.rows[c].name : ''} ×
                    </button>
                  ))}
                </span>
              )}
              {cfg.exclude[method].length > 0 && (
                <span>
                  항상 제외:{' '}
                  {cfg.exclude[method].map((c) => (
                    <button key={c} type="button" className="chip" disabled={busy} title="누르면 제외를 해제합니다" onClick={() => restore(c)}>
                      {c} ×
                    </button>
                  ))}
                </span>
              )}
            </div>
          )}
          {msg && <p className="muted small">{msg}</p>}
          {open && (
            <div className="table-wrap tc-table">
              <table className="grid compact">
                <thead>
                  <tr>
                    <th className="num">#</th>
                    <th>코드</th>
                    <th>종목명</th>
                    <th>출처</th>
                    <th className="num" title="스캔 전체 순위 (점수순)">스캔 순위</th>
                    <th className="num">점수</th>
                    <th title="스캔의 선별 적격 판정">적격</th>
                    <th title="스캔이 적은 탈락 사유">탈락 사유</th>
                    <th title={`틱 수집기 구독 예정 여부: 감시목록 앞 ${comp.tick_cap}종목 (실제 구독은 수집기 탭)`}>틱 (예정)</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {list.map((c, i) => {
                    const r = comp.rows[c];
                    if (!r) return null;
                    return (
                      <tr key={c} className="nohover">
                        <td className="num">{i + 1}</td>
                        <td className="mono">{c}</td>
                        <td>{r.name}</td>
                        <td>
                          <span className={`tag ${r.src === 'A' ? 'apply-on' : r.src === 'manual' ? 'apply-record' : ''}`}>{TARGET_SRC_KO[r.src]}</span>
                        </td>
                        <td className="num">{r.rank ?? '-'}</td>
                        <td className="num">{r.score === null ? '-' : fmtInt(r.score)}</td>
                        <td className={r.eligible ? 'res-pass' : 'muted'}>{r.eligible === null ? '-' : r.eligible ? 'Y' : 'N'}</td>
                        <td className="reason" title={r.fail}>
                          {r.fail || '-'}
                        </td>
                        <td className={noTick.has(c) ? 'res-fail' : 'muted'}>{noTick.has(c) ? '밖' : '안'}</td>
                        <td>
                          <button type="button" className="chip" disabled={busy} title={`${TRADE_STRATEGY_KO[method]} 대상에서 항상 제외 (매일 유지)`} onClick={() => exclude(c)}>
                            제외
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      {!doc && !err && <p className="muted small">구성을 읽는 중…</p>}
    </div>
  );
}
