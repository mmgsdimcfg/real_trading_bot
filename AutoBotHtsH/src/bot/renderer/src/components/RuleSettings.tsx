/*
 * Update log
 * - [2026-11-27] type=feat - Rule-trade settings window (user request: a dedicated window like the rebound alert's "세부 설정").
 *   Left: the buy / sell conditions of the server (r016 `conditions`) in evaluation order - number, title, stage, ON / OFF of the
 *   condition's ENABLE_* switch, marks for a saved override (◆), a value the running bot does not have yet (미적용) and unsaved edits.
 *   Right: the selected condition's description and its r001 values in the same editor the pipeline tab uses (ValuesEditor).
 *   ONE editing state over every condition of both sides: edits made in several conditions are saved together with one 저장 (only
 *   the changed keys go to the server in one `set`, as everywhere). Nothing here decides a trade - it edits r001 values through r016.
 *   RuleSettingsPanel = the content (AutoBotHtsH shows it in its own MDI window), RuleSettingsModal = AutoBotH's dialog
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { editItemsFor, isEnableItem, sortedConds, STAGE_LABEL, unionEditItems } from '../lib';
import { EditorActions, ValuesEditor, useValueEditing } from './ValuesEditor';
import type { Condition } from '../../../shared/types';

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

  // the value an ENABLE_* switch would have after 저장 (draft first, then the saved override, then the base)
  const enabledOf = (c: Condition): boolean | null => {
    const it = editItemsFor(c).find((i) => isEnableItem(i));
    if (!it) return null;
    const d = ed.drafts[it.key];
    if (typeof d === 'boolean') return d;
    return !!(it.has_override ? it.override : it.base);
  };

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
            const on = enabledOf(c);
            const dirty = its.some((i) => i.key in ed.changes || ed.errors[i.key]);
            return (
              <li key={condId(side, c)}>
                <button type="button" className={c === cur ? 'on' : ''} onClick={() => setSel((s) => ({ ...s, [side]: condId(side, c) }))} title={c.description}>
                  <span className="mono rs-no">{String(c.no).padStart(2, '0')}</span>
                  <span className="rs-title">{c.title}</span>
                  {its.some((i) => i.pending) && <i className="pend-dot">미적용</i>}
                  {its.some((i) => i.has_override) && <i className="ovr-mark" title="저장된 변경값이 있습니다">◆</i>}
                  {dirty && <i className="rs-dirty" title="저장하지 않은 변경" />}
                  {on !== null && <span className={`tag ${on ? 'ok' : 'off'}`}>{on ? 'ON' : 'OFF'}</span>}
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
