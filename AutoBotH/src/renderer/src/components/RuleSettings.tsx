/*
 * Update log
 * - [2026-11-27] type=feat - Rule-trade settings window (user request: a dedicated window like the rebound alert's "세부 설정").
 *   Left: the buy / sell conditions of the server (r016 `conditions`) in evaluation order - number, title, stage, ON / OFF of the
 *   condition's ENABLE_* switch, marks for a saved override (◆), a value the running bot does not have yet (미적용) and unsaved edits.
 *   Right: the selected condition's description and its r001 values in the same editor the pipeline tab uses (ValuesEditor).
 *   ONE editing state over every condition of both sides: edits made in several conditions are saved together with one 저장 (only
 *   the changed keys go to the server in one `set`, as everywhere). Nothing here decides a trade - it edits r001 values through r016.
 *   RuleSettingsPanel = the content (AutoBotHtsH shows it in its own MDI window), RuleSettingsModal = AutoBotH's dialog
 * - [2026-11-27] type=fix - The list showed ON / OFF from a condition's first ENABLE_* key, which read as "the condition is off"
 *   (09 1분봉 트리거 "OFF" was only its chase guard ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD). Now "옵션 on/total" with the keys in the
 *   tooltip, and a note above the switches of the selected condition
 * - [2026-11-27] type=feat - StrategySettingsPanel / StrategySettingsModal (user request: "세부 설정" on the Edge AI and the VWAP trade
 *   cards too): the r001 values of that way of trading (shared/paper STRATEGY_SECTIONS) in sections with a line of what each does,
 *   read with `schema --all` and edited with the same editor and one 저장. The cards' own on / off switches are not in the list
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { editItemsFor, isEnableItem, sortedConds, STAGE_LABEL, unionEditItems } from '../lib';
import { EditorActions, ValuesEditor, useValueEditing } from './ValuesEditor';
import type { Condition } from '../../../shared/types';
import type { SchemaDoc } from '../../../shared/control';
import { STRATEGY_KO, STRATEGY_SECTIONS, strategyKeys, type StrategyKind } from '../../../shared/paper';
import { peek, put } from '../dataCache';
import type { EditItem } from './ValuesEditor';

type Side = 'buy' | 'sell';
const SIDE_KO: Record<Side, string> = { buy: '매수 조건', sell: '매도 조건' };
const condId = (side: Side, c: Condition) => `${side}:${c.no}:${c.func}`;

export function RuleSettingsPanel() {
  const doc = useStore((s) => s.conditions);
  const connected = useStore((s) => s.status.conn === 'connected');
  const [side, setSide] = useState<Side>('buy');
  const [sel, setSel] = useState<Record<Side, string>>({ buy: '', sell: '' });
  const buy = useMemo(() => sortedConds(doc, 'buy'), [doc]);
  const sell = useMemo(() => sortedConds(doc, 'sell'), [doc]);
  const conds = side === 'buy' ? buy : sell;
  const allItems = useMemo(() => unionEditItems([...buy, ...sell]), [buy, sell]);
  const ed = useValueEditing(allItems);
  const cur = conds.find((c) => condId(side, c) === sel[side]) || conds[0];
  const items = useMemo(() => editItemsFor(cur), [cur]);
  const editable = allItems.some((i) => i.editable);

  // [2026-11-27] the ENABLE_* switches of a condition and what each would be after 저장 (draft, else the saved override, else the
  // base). They are OPTIONS inside the condition (a guard, a sub-gate) - most conditions, the trigger above all, have no switch
  // that turns the whole condition off, so the list shows "옵션 on/total", never a bare ON / OFF of the condition
  const switchesOf = (c: Condition): { key: string; on: boolean }[] =>
    editItemsFor(c)
      .filter((i) => isEnableItem(i))
      .map((it) => {
        const d = ed.drafts[it.key];
        return { key: it.key, on: typeof d === 'boolean' ? d : !!(it.has_override ? it.override : it.base) };
      });

  if (!doc)
    return (
      <div className="rule-set">
        <p className="muted">조건 목록을 아직 받지 못했습니다. 연결 후 다시 열어 주세요.</p>
      </div>
    );

  return (
    <div className="rule-set">
      <div className="rule-set-head">
        <div className="seg">
          {(['buy', 'sell'] as Side[]).map((s) => (
            <button key={s} type="button" className={side === s ? 'on' : ''} onClick={() => setSide(s)}>
              {SIDE_KO[s]} {(s === 'buy' ? buy : sell).length}
            </button>
          ))}
        </div>
        <span className="muted small">
          규칙 매매(r005 / r006)의 조건별 기준값 · 저장한 값은 r003 재시작 후 적용
          {doc.entry_trigger_mode ? ` · 트리거 ${doc.entry_trigger_mode}` : ''}
        </span>
        {!editable && <span className="badge-warn">읽기 전용 (서버에 연결되면 바꿀 수 있습니다)</span>}
      </div>
      <div className="rule-set-body">
        <ul className="rule-set-list">
          {conds.map((c) => {
            const its = editItemsFor(c);
            const sw = switchesOf(c);
            const nOn = sw.filter((x) => x.on).length;
            const dirty = its.some((i) => i.key in ed.changes || ed.errors[i.key]);
            return (
              <li key={condId(side, c)}>
                <button type="button" className={c === cur ? 'on' : ''} onClick={() => setSel((s) => ({ ...s, [side]: condId(side, c) }))} title={c.description}>
                  <span className="mono rs-no">{String(c.no).padStart(2, '0')}</span>
                  <span className="rs-title">{c.title}</span>
                  {its.some((i) => i.pending) && <i className="pend-dot">미적용</i>}
                  {its.some((i) => i.has_override) && <i className="ovr-mark" title="저장된 변경값이 있습니다">◆</i>}
                  {dirty && <i className="rs-dirty" title="저장하지 않은 변경" />}
                  {sw.length > 0 && (
                    <span
                      className={`tag ${nOn === sw.length ? 'ok' : 'off'}`}
                      title={[`이 조건 안의 옵션 스위치 ${sw.length}개 중 ${nOn}개 켜짐 (조건 자체를 끄는 표시가 아닙니다)`, ...sw.map((x) => `${x.key} = ${x.on ? 'ON' : 'OFF'}`)].join('\n')}
                    >
                      옵션 {nOn}/{sw.length}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
          {!conds.length && <li className="muted small">조건이 없습니다.</li>}
        </ul>
        <div className="rule-set-detail">
          {cur && (
            <>
              <h3>
                <span className="mono">{String(cur.no).padStart(2, '0')}</span> {cur.title}
                <span className="muted small"> · {STAGE_LABEL[cur.stage] || cur.stage} · {cur.func}</span>
              </h3>
              <p className="rs-desc">{cur.description}</p>
              {items.some((i) => isEnableItem(i)) && (
                <p className="muted small rs-switch-note">
                  아래 ENABLE_* 스위치는 이 조건 안의 옵션(가드 · 하위 게이트)을 켜고 끕니다. 끈다고 조건 전체가 꺼지는 것은 아닙니다.
                </p>
              )}
              {items.length ? <ValuesEditor items={items} ed={ed} disabled={!connected || ed.saving} /> : <p className="muted">이 조건에는 바꿀 수 있는 설정값이 없습니다.</p>}
            </>
          )}
        </div>
      </div>
      <div className="rule-set-foot">
        <EditorActions ed={ed} disabled={!connected} />
      </div>
    </div>
  );
}

/** AutoBotH: the settings as a dialog over the app (window event 'rule-settings', App.tsx) */
export function RuleSettingsModal({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal rule-set-modal" role="dialog" aria-label="규칙 매매 세부 설정">
        <header>
          <h3>규칙 매매 세부 설정</h3>
          <button type="button" className="mini" onClick={onClose}>
            닫기
          </button>
        </header>
        <RuleSettingsPanel />
      </div>
    </div>
  );
}

/** [2026-11-27] the r001 values of the Edge AI trade / the VWAP trade, in sections (control > card > 세부 설정) */
export function StrategySettingsPanel({ kind }: { kind: StrategyKind }) {
  const connected = useStore((s) => s.status.conn === 'connected');
  const historyRev = useStore((s) => s.historyRev);
  const [doc, setDoc] = useState<SchemaDoc | null>(() => peek<SchemaDoc>('settings:schema') || null);
  const [err, setErr] = useState('');
  useEffect(() => {
    if (!connected) return;
    let alive = true;
    const kept = peek<SchemaDoc>('settings:schema');
    if (kept) setDoc(kept);
    setErr(kept ? '' : '불러오는 중…');
    void window.api.ctlSchema(true).then((r) => {
      if (!alive) return;
      if (r.ok && r.data) {
        setDoc(put('settings:schema', r.data));
        setErr('');
      } else setErr(`불러오기 실패: ${r.error}`);
    });
    return () => {
      alive = false;
    };
  }, [connected, historyRev]);
  const all: EditItem[] = useMemo(() => {
    const want = new Set(strategyKeys(kind));
    return (doc?.keys || []).filter((k) => want.has(k.key)).map((k) => ({ ...k, editable: k.editable !== false, current: k.effective }));
  }, [doc, kind]);
  const byKey = useMemo(() => new Map(all.map((i) => [i.key, i])), [all]);
  const ed = useValueEditing(all);
  const sections = STRATEGY_SECTIONS[kind]
    .map((sec) => ({ ...sec, rows: sec.keys.filter(([k]) => byKey.has(k)) }))
    .filter((sec) => sec.rows.length > 0);
  // what each key does, under its row (the editor's own note line)
  const notes = useMemo(() => Object.fromEntries(STRATEGY_SECTIONS[kind].flatMap((sec) => sec.keys)), [kind]);

  return (
    <div className="rule-set strat-set">
      <div className="rule-set-head">
        <b>{STRATEGY_KO[kind]} 세부 설정</b>
        <span className="muted small">저장한 값은 r003 재시작 후 적용 · 켜기 / 끄기는 제어 탭 카드에서</span>
        {!connected && <span className="badge-warn">연결되면 볼 수 있습니다</span>}
        {err && <span className="muted small">{err}</span>}
      </div>
      <div className="strat-set-body">
        {sections.map((sec) => (
          <section key={sec.title}>
            <h3>{sec.title}</h3>
            {sec.note && <p className="rs-desc">{sec.note}</p>}
            <ValuesEditor items={sec.rows.map(([k]) => byKey.get(k)!)} ed={ed} disabled={!connected || ed.saving} alerts={notes} />
          </section>
        ))}
        {connected && doc && !sections.length && <p className="muted">서버에 이 매매 방식의 설정값이 없습니다 (서버 코드 갱신 필요).</p>}
      </div>
      <div className="rule-set-foot">
        <EditorActions ed={ed} disabled={!connected} />
      </div>
    </div>
  );
}

/** AutoBotH: the Edge AI / VWAP settings as a dialog (window event 'strategy-settings' with detail 'edge' | 'vwap', App.tsx) */
export function StrategySettingsModal({ kind, onClose }: { kind: StrategyKind; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal rule-set-modal strat-set-modal" role="dialog" aria-label={`${STRATEGY_KO[kind]} 세부 설정`}>
        <header>
          <h3>{STRATEGY_KO[kind]} 세부 설정</h3>
          <button type="button" className="mini" onClick={onClose}>
            닫기
          </button>
        </header>
        <StrategySettingsPanel kind={kind} />
      </div>
    </div>
  );
}
