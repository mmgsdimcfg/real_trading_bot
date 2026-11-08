/*
 * Update log
 * - [2026-11-05] type=feat - r004 watchlist editor (guide 2.3): `watchlist get` fills the table; rows can be added, deleted and
 *   reordered (drag handle or ↑/↓ - order = r003's activation priority). Code must be 6 digits, the name must not hold a comma /
 *   semicolon / quote; duplicates and more than 100 rows block the save. 저장 -> main asks for confirmation -> `watchlist set
 *   --stdin`; the result shows the backup path. The fixed notice explains that g002 rewrites the file every morning
 */

import { useEffect, useMemo, useState } from 'react';
import { checkWatchRows, WATCH_MAX, type WatchEntry, type WatchlistDoc } from '../../../shared/control';
import { useStore } from '../store';

interface Row extends WatchEntry {
  id: number;
}

let nextId = 1;
const withIds = (rows: WatchEntry[]): Row[] => rows.map((r) => ({ ...r, id: nextId++ }));

export function WatchlistEditor({ onClose }: { onClose: () => void }) {
  const connected = useStore((s) => s.status.conn === 'connected');
  const bumpHistory = useStore((s) => s.bumpHistory);
  const [doc, setDoc] = useState<WatchlistDoc | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [orig, setOrig] = useState('');
  const [msg, setMsg] = useState('');
  const [toast, setToast] = useState('');
  const [saving, setSaving] = useState(false);
  const [drag, setDrag] = useState<number | null>(null);

  const load = async () => {
    setMsg('불러오는 중…');
    const r = await window.api.watchlistGet();
    if (r.ok && r.data) {
      setDoc(r.data);
      setRows(withIds(r.data.codes));
      setOrig(JSON.stringify(r.data.codes));
      setMsg('');
    } else setMsg(`불러오기 실패: ${r.error}`);
  };
  useEffect(() => {
    if (connected) void load();
  }, [connected]);

  const plain = rows.map(({ code, name }) => ({ code, name }));
  const dirty = JSON.stringify(plain) !== orig;
  const check = useMemo(() => checkWatchRows(plain), [rows]); // eslint-disable-line react-hooks/exhaustive-deps
  const blocked = !!check.list || Object.keys(check.rows).length > 0;

  const update = (id: number, patch: Partial<WatchEntry>) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const move = (from: number, to: number) =>
    setRows((rs) => {
      if (to < 0 || to >= rs.length || from === to) return rs;
      const n = [...rs];
      const [x] = n.splice(from, 1);
      n.splice(to, 0, x);
      return n;
    });

  const save = async () => {
    setSaving(true);
    setMsg('저장 중…');
    try {
      const r = await window.api.watchlistSet(plain);
      if (r.cancelled) return setMsg('');
      if (r.ok && r.data) {
        setToast(`저장했습니다 (${r.data.count ?? plain.length}종목). 백업: ${r.data.backup || '없음 (기존 파일 없음)'} - r003 재시작 후 반영됩니다.`);
        setMsg('');
        bumpHistory();
        await load();
      } else setMsg(`저장 실패: ${r.error}`);
    } finally {
      setSaving(false);
    }
  };

  const close = () => {
    if (dirty && !window.confirm('저장하지 않은 감시목록 변경이 있습니다. 닫을까요?')) return;
    onClose();
  };

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal wl-editor">
        <header>
          <h3>감시목록 편집 (r004)</h3>
          <button className="mini" onClick={close}>
            닫기
          </button>
        </header>
        <p className="callout warn small">
          이 목록은 매일 아침 g002 스캐너 내보내기가 다시 씁니다. 수정은 당일 재시작까지만 유효합니다.
        </p>
        <p className="muted small">
          {doc ? `${doc.path} · 수정 ${doc.mtime?.replace('T', ' ') || '-'} (서버 KST)` : ''} · 순서 = r003의 활성 우선순위 (위가 먼저) · 최대 {WATCH_MAX}종목
        </p>
        {toast && <p className="toast">{toast}</p>}
        <div className="wl-table">
          <table className="grid compact">
            <thead>
              <tr>
                <th />
                <th className="num">#</th>
                <th>코드</th>
                <th>종목명</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const err = check.rows[i];
                return (
                  <tr
                    key={r.id}
                    className={`nohover ${drag === i ? 'dragging' : ''} ${err ? 'verr-row' : ''}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      if (drag !== null && drag !== i) {
                        move(drag, i);
                        setDrag(i);
                      }
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDrag(null);
                    }}
                  >
                    <td className="handle" draggable onDragStart={() => setDrag(i)} onDragEnd={() => setDrag(null)} title="끌어서 순서 변경">
                      ⠿
                    </td>
                    <td className="num muted">{i + 1}</td>
                    <td>
                      <input
                        className="mono code-in"
                        maxLength={6}
                        value={r.code}
                        onChange={(e) => update(r.id, { code: e.target.value.replace(/\D/g, '') })}
                        placeholder="000000"
                      />
                    </td>
                    <td>
                      <input value={r.name} onChange={(e) => update(r.id, { name: e.target.value })} placeholder="이름 (선택)" />
                      {err && <div className="err-text small">{err}</div>}
                    </td>
                    <td className="row-btns">
                      <button className="mini" disabled={i === 0} onClick={() => move(i, i - 1)} title="위로">
                        ↑
                      </button>
                      <button className="mini" disabled={i === rows.length - 1} onClick={() => move(i, i + 1)} title="아래로">
                        ↓
                      </button>
                      <button className="mini" onClick={() => setRows((rs) => rs.filter((x) => x.id !== r.id))} title="삭제">
                        삭제
                      </button>
                    </td>
                  </tr>
                );
              })}
              {!rows.length && (
                <tr className="nohover">
                  <td colSpan={5} className="muted center">
                    {doc ? '종목 없음' : connected ? '' : '서버에 연결되어 있지 않습니다.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {check.list && <p className="err-text small">{check.list}</p>}
        <div className="actions">
          <button disabled={!doc || rows.length >= WATCH_MAX} onClick={() => setRows((rs) => [...rs, ...withIds([{ code: '', name: '' }])])}>
            + 행 추가
          </button>
          <button className="primary" disabled={!connected || !doc || !dirty || blocked || saving} onClick={() => void save()}>
            저장
          </button>
          <button disabled={!dirty || saving} onClick={() => doc && setRows(withIds(doc.codes))}>
            되돌리기
          </button>
          <button disabled={saving || !connected} onClick={() => (!dirty || window.confirm('저장하지 않은 변경을 버리고 서버에서 다시 불러올까요?')) && void load()}>
            다시 불러오기
          </button>
          <span className="muted small">
            {rows.length}종목{dirty ? ' · 저장 안 됨' : ''} {msg}
          </span>
        </div>
      </div>
    </div>
  );
}
