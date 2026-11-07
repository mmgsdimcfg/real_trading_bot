/*
 * Update log
 * - [2026-11-05] type=feat - r001 value editor (guide 2.2): one row per key - key + base/saved override + "미적용" mark, a widget
 *   picked by type (bool toggle, int/float/str input with a "= 0.40%" hint for *_PCT, list/dict JSON textarea), a [기본값] button
 *   (unset) and the row error (input error or the server's `rejected` message verbatim). Nothing is sent while typing:
 *   useValueEditing().save() sends only the changed keys in one `set` call; input errors block the save
 */

import { Fragment, useMemo, useState } from 'react';
import { collectChanges, fmtValue, initialDraft, parseDraft, pctHint, type Draft, type ValueType } from '../../../shared/control';
import { useStore } from '../store';

export interface EditItem {
  key: string;
  type: ValueType;
  base: unknown;
  override: unknown;
  has_override: boolean;
  pending: boolean;
  editable: boolean;
  /** shown for read-only rows */
  current?: unknown;
}

export function useValueEditing(items: EditItem[]) {
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [serverErrors, setServerErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');
  const bumpHistory = useStore((s) => s.bumpHistory);

  const collected = useMemo(() => collectChanges(items, drafts), [items, drafts]);
  const dirty = Object.keys(collected.changes).length > 0 || Object.keys(collected.errors).length > 0;
  const errors = { ...serverErrors, ...collected.errors };

  const setDraft = (key: string, v: Draft) => {
    setDrafts((d) => ({ ...d, [key]: v }));
    setServerErrors(({ [key]: _drop, ...rest }) => rest);
  };
  const cancel = () => {
    setDrafts({});
    setServerErrors({});
    setMsg('');
  };
  const save = async () => {
    if (Object.keys(collected.errors).length) {
      setMsg('입력 오류가 있는 행을 고친 뒤 저장하세요.');
      return;
    }
    const changes = collected.changes;
    if (!Object.keys(changes).length) return;
    setSaving(true);
    setMsg('저장 중…');
    try {
      const r = await window.api.ctlSet(changes);
      const accepted = Object.keys(r.data?.accepted || {});
      const rejected = r.data?.rejected || {};
      setDrafts((d) => {
        const n = { ...d };
        for (const k of accepted) delete n[k];
        return n;
      });
      setServerErrors(rejected);
      if (accepted.length) bumpHistory();
      if (r.ok) setMsg(`${accepted.length}건 저장 - r003 재시작 후 반영됩니다.`);
      else if (Object.keys(rejected).length) setMsg(`${accepted.length}건 저장, ${Object.keys(rejected).length}건 거부 (행의 서버 메시지 참고)`);
      else setMsg(`저장 실패: ${r.error}`);
    } finally {
      setSaving(false);
    }
  };
  const unset = async (key: string) => {
    setMsg('기본값으로 되돌리는 중…');
    const r = await window.api.ctlUnset([key]);
    if (r.ok) {
      setDrafts(({ [key]: _d, ...rest }) => rest);
      setServerErrors(({ [key]: _e, ...rest }) => rest);
      bumpHistory();
      setMsg(`${key} 기본값으로 되돌림 - r003 재시작 후 반영됩니다.`);
    } else setMsg(`되돌리기 실패: ${r.error}`);
  };
  return { drafts, setDraft, errors, dirty, changes: collected.changes, saving, msg, save, cancel, unset, setDrafts };
}

export type ValueEditing = ReturnType<typeof useValueEditing>;

function Widget({ it, ed, disabled }: { it: EditItem; ed: ValueEditing; disabled: boolean }) {
  const d = it.key in ed.drafts ? ed.drafts[it.key] : initialDraft(it);
  if (!it.editable) return <span className="ro mono">{fmtValue(it.current ?? it.base)}</span>;
  if (it.type === 'bool') {
    return (
      <label className="toggle" title={d === true ? '켜짐' : '꺼짐'}>
        <input type="checkbox" disabled={disabled} checked={d === true} onChange={(e) => ed.setDraft(it.key, e.target.checked)} />
        <span>{d === true ? 'ON' : 'OFF'}</span>
      </label>
    );
  }
  if (it.type === 'list' || it.type === 'dict' || it.type === 'null') {
    return (
      <textarea
        className="mono"
        spellCheck={false}
        rows={Math.min(5, Math.max(1, Math.ceil(String(d).length / 34)))}
        disabled={disabled}
        value={String(d)}
        onChange={(e) => ed.setDraft(it.key, e.target.value)}
      />
    );
  }
  const p = parseDraft(it.type, d);
  const hint = p.ok ? pctHint(it.key, p.value) : '';
  return (
    <div className="vinput">
      <input
        className="mono"
        spellCheck={false}
        disabled={disabled}
        inputMode={it.type === 'str' ? 'text' : 'decimal'}
        value={String(d)}
        onChange={(e) => ed.setDraft(it.key, e.target.value)}
      />
      {hint && <span className="muted small">{hint}</span>}
    </div>
  );
}

export function ValuesEditor({ items, ed, disabled }: { items: EditItem[]; ed: ValueEditing; disabled: boolean }) {
  return (
    <table className="values edit">
      <tbody>
        {items.map((it) => {
          const err = ed.errors[it.key];
          const changed = it.key in ed.changes;
          return (
            <Fragment key={it.key}>
              <tr className={`${err ? 'verr-row' : ''} ${changed ? 'changed' : ''} ${it.editable ? '' : 'readonly'}`}>
                <td>
                  <div className="vk mono">
                    {it.key}
                    {it.pending && (
                      <i className="pend-dot" title="저장됐지만 실행 중인 봇에는 아직 반영되지 않음 (재시작 필요)">
                        미적용
                      </i>
                    )}
                  </div>
                  <div className="vb">
                    기본 {fmtValue(it.base)}
                    {pctHint(it.key, it.base) && ` (${pctHint(it.key, it.base).slice(2)})`}
                    {it.has_override && <b> · 저장값 {fmtValue(it.override)}</b>}
                    {!it.editable && ' · 읽기 전용'}
                  </div>
                </td>
                <td className="vw">
                  <Widget it={it} ed={ed} disabled={disabled} />
                </td>
                <td className="va">
                  {it.editable && it.has_override && (
                    <button className="mini" disabled={disabled} onClick={() => void ed.unset(it.key)} title="오버라이드를 지우고 기본값으로 (unset)">
                      기본값
                    </button>
                  )}
                </td>
              </tr>
              {err && (
                <tr className="verr">
                  <td colSpan={3}>{err}</td>
                </tr>
              )}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

/** Save / cancel bar under an editor. */
export function EditorActions({ ed, disabled }: { ed: ValueEditing; disabled: boolean }) {
  return (
    <div className="actions editor-actions">
      <button className="primary" disabled={disabled || ed.saving || !ed.dirty} onClick={() => void ed.save()}>
        저장{Object.keys(ed.changes).length ? ` (${Object.keys(ed.changes).length})` : ''}
      </button>
      <button disabled={ed.saving || !ed.dirty} onClick={ed.cancel}>
        취소
      </button>
      {ed.msg && <span className="muted small">{ed.msg}</span>}
    </div>
  );
}
