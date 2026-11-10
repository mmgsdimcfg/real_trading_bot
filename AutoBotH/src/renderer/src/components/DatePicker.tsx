/*
 * Update log
 * - [2026-11-10] type=feat - Small calendar date field (YYYYMMDD value): the field shows YYYY-MM-DD, a click opens a month
 *   calendar under it (일~토, ‹ › months, today / selected marked, days after `max` disabled, 오늘 button); outside click or Esc
 *   closes it. Drawn in the app (not the native <input type="date">, whose text follows the OS locale - 11/10/2026)
 */

import { useEffect, useRef, useState } from 'react';

const WEEK = ['일', '월', '화', '수', '목', '금', '토'];
const pad = (n: number) => String(n).padStart(2, '0');
const ymd = (y: number, m: number, d: number) => `${y}${pad(m + 1)}${pad(d)}`;
const show = (v: string) => (/^\d{8}$/.test(v) ? `${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)}` : '');

interface Props {
  value: string; // YYYYMMDD
  onChange: (v: string) => void;
  max?: string; // YYYYMMDD - later days cannot be picked
  today: string; // YYYYMMDD
  title?: string;
}

export function DatePicker({ value, onChange, max, today, title }: Props) {
  const [open, setOpen] = useState(false);
  const base = /^\d{8}$/.test(value) ? value : today;
  const [view, setView] = useState({ y: Number(base.slice(0, 4)), m: Number(base.slice(4, 6)) - 1 });
  const box = useRef<HTMLDivElement>(null);

  // reopen on the month of the current value; close on a click outside or Esc
  useEffect(() => {
    if (!open) return;
    setView({ y: Number(base.slice(0, 4)), m: Number(base.slice(4, 6)) - 1 });
    const onDown = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const first = new Date(view.y, view.m, 1).getDay();
  const days = new Date(view.y, view.m + 1, 0).getDate();
  const cells: (number | null)[] = [...Array(first).fill(null), ...Array.from({ length: days }, (_, i) => i + 1)];
  while (cells.length % 7) cells.push(null);
  const move = (delta: number) => setView((v) => ({ y: v.y + Math.floor((v.m + delta) / 12), m: (((v.m + delta) % 12) + 12) % 12 }));
  // the next month cannot be opened when its first day is already after `max`
  const nextFirst = view.m === 11 ? ymd(view.y + 1, 0, 1) : ymd(view.y, view.m + 1, 1);
  const nextDisabled = !!max && nextFirst > max;
  const pick = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  return (
    <div className="datepicker" ref={box}>
      <button type="button" className={`dp-field mono${open ? ' open' : ''}`} title={title} onClick={() => setOpen((o) => !o)}>
        <span>{show(value) || 'YYYY-MM-DD'}</span>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <rect x="2" y="3" width="12" height="11" rx="1.5" />
          <path d="M2 6.5h12M5.5 1.5v3M10.5 1.5v3" />
        </svg>
      </button>
      {open && (
        <div className="dp-pop" role="dialog" aria-label="날짜 선택">
          <div className="dp-head">
            <button type="button" className="dp-nav" onClick={() => move(-1)} title="이전 달">
              ‹
            </button>
            <b>
              {view.y}년 {view.m + 1}월
            </b>
            <button type="button" className="dp-nav" onClick={() => move(1)} disabled={nextDisabled} title="다음 달">
              ›
            </button>
          </div>
          <div className="dp-grid">
            {WEEK.map((w, i) => (
              <span key={w} className={`dp-week${i === 0 ? ' sun' : i === 6 ? ' sat' : ''}`}>
                {w}
              </span>
            ))}
            {cells.map((d, i) => {
              if (d === null) return <span key={`e${i}`} />;
              const v = ymd(view.y, view.m, d);
              const cls = ['dp-day', v === value ? 'sel' : '', v === today ? 'today' : '', i % 7 === 0 ? 'sun' : i % 7 === 6 ? 'sat' : ''].filter(Boolean).join(' ');
              return (
                <button key={v} type="button" className={cls} disabled={!!max && v > max} onClick={() => pick(v)}>
                  {d}
                </button>
              );
            })}
          </div>
          <div className="dp-foot">
            <button type="button" className="mini" onClick={() => pick(today)}>
              오늘
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
