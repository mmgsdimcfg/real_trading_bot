/*
 * Update log
 * - [2026-11-11] type=feat - App-styled confirmation dialog: the main process no longer shows Electron's native message box
 *   (control.ts askConfirm -> 'confirm:ask'); this component renders the request as a card - tag pill (LIVE / 장중) + kicker,
 *   question title, one-line message, key/value rows (mode, profile, host) and [취소] [ok] - and answers over 'confirm:answer'.
 *   Requests queue up (one at a time); Esc = cancel; the cancel button has the focus (Enter never confirms by accident)
 */

import { useEffect, useRef, useState } from 'react';
import type { ConfirmRequest } from '../../../shared/types';

export function ConfirmDialog() {
  const [queue, setQueue] = useState<ConfirmRequest[]>([]);
  const cancelRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const off = window.api.onConfirm((req) => setQueue((q) => [...q, req]));
    return () => {
      off();
    };
  }, []);
  const cur = queue[0];
  const answer = (ok: boolean) => {
    if (!cur) return;
    window.api.answerConfirm(cur.id, ok);
    setQueue((q) => q.slice(1));
  };
  useEffect(() => {
    if (!cur) return;
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        answer(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cur?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!cur) return null;
  return (
    <div className="modal-back confirm-back" role="presentation">
      <div className="modal confirm" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <div className="confirm-kicker">
          {cur.tag && <span className={`confirm-tag ${cur.tagTone || ''}`}>{cur.tag}</span>}
          <span>{cur.kicker}</span>
        </div>
        <h2 id="confirm-title">{cur.title}</h2>
        {cur.message && <p className="confirm-msg">{cur.message}</p>}
        {!!cur.rows?.length && (
          <div className="confirm-rows">
            {cur.rows.map((r, i) => (
              <div key={i} className="confirm-row">
                <span className="k">{r.k}</span>
                <span className={`v ${r.tone || ''}`}>{r.v}</span>
              </div>
            ))}
          </div>
        )}
        <div className="confirm-btns">
          <button ref={cancelRef} className="wide" onClick={() => answer(false)}>
            {cur.cancelLabel || '취소'}
          </button>
          <button className="primary wide" onClick={() => answer(true)}>
            {cur.okLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
