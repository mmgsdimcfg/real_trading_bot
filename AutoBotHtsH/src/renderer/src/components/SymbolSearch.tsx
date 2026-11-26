// Update log
// - [2026-11-26] type=feat (History (user 2026-11-26): with an empty box the list shows the last 20 symbols picked or shown (newest first,
//   shared by every symbol box - symHistory.ts); arrow keys + Enter pick as in the search list, ✕ removes one entry, "모두 지우기" all)
// - [2026-11-06] type=feat (Symbol search box: code, name or Korean initial consonants (ㅅㅅㅈㅈ) against the KIS stock master; arrow keys +
//   Enter pick, Esc closes; shows market and type (ETF / ETN ...). Used by the toolbar and by each window's header)

import { useEffect, useRef, useState } from 'react';
import { NxtMark } from './NxtTag';
import type { MasterItem } from '../../../shared/kis';
import { GROUP_LABEL } from '../../../shared/kis';
import { clearSymHistory, forgetSymbol, rememberSymbol, useSymHistory } from '../symHistory';

/** master entries of the history codes (asked once per code and session; a code the master does not know stays code-only) */
const histCache = new Map<string, MasterItem | null>();

export function SymbolSearch({ value, onPick, placeholder = '종목명 / 코드 / 초성', compact }: { value?: string; onPick: (code: string) => void; placeholder?: string; compact?: boolean }) {
  const [q, setQ] = useState('');
  const [items, setItems] = useState<MasterItem[]>([]);
  const [open, setOpen] = useState(false);
  const [sel, setSel] = useState(0);
  const [, setRev] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const history = useSymHistory();
  const typing = !!q.trim();

  useEffect(() => {
    if (!q.trim()) return setItems([]);
    let alive = true;
    const t = setTimeout(() => {
      void window.hts.search(q).then((r) => {
        if (!alive) return;
        setItems(r);
        setSel(0);
      });
    }, 80);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [q]);

  // names of the history entries, when the list is shown
  useEffect(() => {
    if (!open || typing) return;
    let alive = true;
    const missing = history.filter((c) => !histCache.has(c));
    if (!missing.length) return;
    void Promise.all(missing.map((c) => window.hts.symbol(c).then((m) => histCache.set(c, m)).catch(() => undefined))).then(() => alive && setRev((n) => n + 1));
    return () => {
      alive = false;
    };
  }, [open, typing, history]);

  useEffect(() => {
    const h = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener('mousedown', h);
    return () => window.removeEventListener('mousedown', h);
  }, []);

  const rows: { code: string; item: MasterItem | null }[] = typing ? items.map((it) => ({ code: it.code, item: it })) : history.map((c) => ({ code: c, item: histCache.get(c) ?? null }));
  const at = Math.min(sel, Math.max(0, rows.length - 1));

  const pick = (code?: string) => {
    const c = code || (/^[0-9A-Z]{6}$/i.test(q.trim()) ? q.trim().toUpperCase() : '');
    if (!c) return;
    rememberSymbol(c);
    onPick(c);
    setQ('');
    setSel(0);
    setOpen(false);
  };

  return (
    <div className={`sym-search ${compact ? 'compact' : ''}`} ref={box}>
      <input
        value={q}
        placeholder={value ? `${value}  ·  ${placeholder}` : placeholder}
        spellCheck={false}
        onFocus={() => (setOpen(true), setSel(0))}
        onMouseDown={() => setOpen(true)}
        onChange={(e) => {
          setQ(e.target.value);
          setSel(0);
          setOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') (setOpen(true), setSel(Math.min(rows.length - 1, at + 1)), e.preventDefault());
          else if (e.key === 'ArrowUp') (setSel(Math.max(0, at - 1)), e.preventDefault());
          else if (e.key === 'Enter') pick(open ? rows[at]?.code : undefined);
          else if (e.key === 'Escape') setOpen(false);
        }}
      />
      {open && rows.length > 0 && (
        <ul className="sym-list">
          {!typing && (
            <li className="sym-hist-head" onMouseDown={(e) => e.preventDefault()}>
              <span>최근 조회 {rows.length}</span>
              <button type="button" onMouseDown={(e) => (e.preventDefault(), e.stopPropagation(), clearSymHistory())} title="이력을 모두 지웁니다">
                모두 지우기
              </button>
            </li>
          )}
          {rows.map(({ code, item }, i) => (
            <li key={code} className={`${i === at ? 'sel' : ''}${typing ? '' : ' hist'}`} onMouseDown={(e) => (e.preventDefault(), pick(code))} onMouseEnter={() => setSel(i)}>
              <span className="mono">{code}</span>
              <b>
                {item?.name || code} {item?.nxt && <NxtMark />}
              </b>
              <small>
                {item?.market || ''}
                {item?.group && item.group !== 'ST' ? ` · ${GROUP_LABEL[item.group] || item.group}` : ''}
              </small>
              {!typing && (
                <button type="button" className="sym-hist-x" title="이력에서 지우기" onMouseDown={(e) => (e.preventDefault(), e.stopPropagation(), forgetSymbol(code))}>
                  ✕
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
