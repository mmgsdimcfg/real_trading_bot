// Update log
// - [2026-11-06] type=feat (Symbol search box: code, name or Korean initial consonants (ㅅㅅㅈㅈ) against the KIS stock master; arrow keys +
//   Enter pick, Esc closes; shows market and type (ETF / ETN ...). Used by the toolbar and by each window's header)

import { useEffect, useRef, useState } from 'react';
import { NxtMark } from './NxtTag';
import type { MasterItem } from '../../../shared/kis';
import { GROUP_LABEL } from '../../../shared/kis';

export function SymbolSearch({ value, onPick, placeholder = '종목명 / 코드 / 초성', compact }: { value?: string; onPick: (code: string) => void; placeholder?: string; compact?: boolean }) {
  const [q, setQ] = useState('');
  const [items, setItems] = useState<MasterItem[]>([]);
  const [open, setOpen] = useState(false);
  const [sel, setSel] = useState(0);
  const box = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    const h = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener('mousedown', h);
    return () => window.removeEventListener('mousedown', h);
  }, []);

  const pick = (it?: MasterItem) => {
    const code = it?.code || (/^[0-9A-Z]{6}$/i.test(q.trim()) ? q.trim().toUpperCase() : '');
    if (!code) return;
    onPick(code);
    setQ('');
    setOpen(false);
  };

  return (
    <div className={`sym-search ${compact ? 'compact' : ''}`} ref={box}>
      <input
        value={q}
        placeholder={value ? `${value}  ·  ${placeholder}` : placeholder}
        spellCheck={false}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') (setSel((s) => Math.min(items.length - 1, s + 1)), e.preventDefault());
          else if (e.key === 'ArrowUp') (setSel((s) => Math.max(0, s - 1)), e.preventDefault());
          else if (e.key === 'Enter') pick(items[sel]);
          else if (e.key === 'Escape') setOpen(false);
        }}
      />
      {open && items.length > 0 && (
        <ul className="sym-list">
          {items.map((it, i) => (
            <li key={it.code} className={i === sel ? 'sel' : ''} onMouseDown={(e) => (e.preventDefault(), pick(it))} onMouseEnter={() => setSel(i)}>
              <span className="mono">{it.code}</span>
              <b>
                {it.name} {it.nxt && <NxtMark />}
              </b>
              <small>
                {it.market}
                {it.group && it.group !== 'ST' ? ` · ${GROUP_LABEL[it.group] || it.group}` : ''}
              </small>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
