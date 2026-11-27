/*
 * Update log
 * - [2026-11-27] type=feat - Drag-to-resize table columns (user 2026-11-27, symbol board): useColWidths keeps one px width per
 *   column (localStorage), ColGrip is the handle on the right edge of a header cell. Until a column is first dragged the table keeps
 *   its default (percentage) widths; the first drag freezes every column at the width it has on screen, so only the dragged one
 *   changes. Double click on a handle = back to the default widths.
 * - [2026-11-27] type=feat - Helpers for tables without fixed widths (trade results, holdings): colTable (ref + fixed layout once
 *   a column was dragged), ColGroup, ColReset ("열 너비 초기화", shown only while widths are set)
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent, RefObject } from 'react';

export const COL_MIN = 32;
export const COL_MAX = 1200;

/** stored widths are used only when they are n finite numbers in range (a changed column count drops them) */
export function validColWidths(v: unknown, n: number): number[] | null {
  if (!Array.isArray(v) || v.length !== n) return null;
  if (!v.every((x) => typeof x === 'number' && Number.isFinite(x) && x >= COL_MIN && x <= COL_MAX)) return null;
  return v.map((x) => Math.round(x as number));
}

function load(key: string, n: number): number[] | null {
  try {
    return validColWidths(JSON.parse(localStorage.getItem(key) || 'null'), n);
  } catch {
    return null;
  }
}

function save(key: string, w: number[] | null): void {
  try {
    if (w) localStorage.setItem(key, JSON.stringify(w));
    else localStorage.removeItem(key);
  } catch {
    /* storage unavailable: the widths last for this window only */
  }
}

export interface ColWidths {
  tableRef: RefObject<HTMLTableElement>;
  /** px width per column, null = default widths */
  widths: number[] | null;
  /** sum of the widths (the table's own width), undefined = default */
  total: number | undefined;
  startDrag: (e: ReactMouseEvent, i: number) => void;
  reset: () => void;
}

export function useColWidths(storageKey: string, n: number): ColWidths {
  const tableRef = useRef<HTMLTableElement>(null);
  const [widths, setWidths] = useState<number[] | null>(() => load(storageKey, n));
  const cur = useRef(widths);
  cur.current = widths;
  const stop = useRef<(() => void) | null>(null);
  useEffect(() => () => stop.current?.(), []);

  const startDrag = useCallback(
    (e: ReactMouseEvent, i: number) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      let base = cur.current;
      if (!base) {
        const ths = tableRef.current ? Array.from(tableRef.current.querySelectorAll('thead th')) : [];
        if (ths.length !== n) return;
        base = ths.map((t) => Math.max(COL_MIN, Math.round((t as HTMLElement).offsetWidth)));
      }
      const from = base;
      const x0 = e.clientX;
      const prevCursor = document.body.style.cursor;
      document.body.style.cursor = 'col-resize';
      let moved = false;
      const move = (ev: MouseEvent) => {
        const next = from.slice();
        next[i] = Math.round(Math.min(COL_MAX, Math.max(COL_MIN, from[i] + ev.clientX - x0)));
        moved = true;
        cur.current = next;
        setWidths(next);
      };
      // the click that ends a drag must not reach the header (it would change the sort order)
      const eatClick = (ev: MouseEvent) => {
        ev.stopPropagation();
        ev.preventDefault();
      };
      const up = () => {
        stop.current?.();
        if (moved) {
          save(storageKey, cur.current);
          window.addEventListener('click', eatClick, true);
          setTimeout(() => window.removeEventListener('click', eatClick, true), 0);
        }
      };
      stop.current = () => {
        window.removeEventListener('mousemove', move);
        window.removeEventListener('mouseup', up);
        document.body.style.cursor = prevCursor;
        stop.current = null;
      };
      window.addEventListener('mousemove', move);
      window.addEventListener('mouseup', up);
    },
    [n, storageKey],
  );

  const reset = useCallback(() => {
    cur.current = null;
    setWidths(null);
    save(storageKey, null);
  }, [storageKey]);

  return { tableRef, widths, total: widths ? widths.reduce((a, b) => a + b, 0) : undefined, startDrag, reset };
}

/** handle on the right edge of a header cell (the cell must be positioned - .grid th is sticky) */
export function ColGrip(props: { cols: ColWidths; i: number }) {
  return (
    <span
      className="col-grip"
      title="끌어서 열 너비 조절 · 두 번 누르면 기본 너비로"
      onMouseDown={(e) => props.cols.startDrag(e, props.i)}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => {
        e.stopPropagation();
        props.cols.reset();
      }}
    />
  );
}

/** props of a <table> whose columns size themselves until one is dragged: then fixed layout at the stored widths */
export function colTable(cols: ColWidths, className: string): { ref: RefObject<HTMLTableElement>; className: string; style: CSSProperties | undefined } {
  return { ref: cols.tableRef, className: cols.widths ? `${className} cols-fixed` : className, style: cols.total ? { width: cols.total } : undefined };
}

/** first child of such a table */
export function ColGroup(props: { cols: ColWidths }) {
  if (!props.cols.widths) return null;
  return (
    <colgroup>
      {props.cols.widths.map((w, i) => (
        <col key={i} style={{ width: w }} />
      ))}
    </colgroup>
  );
}

/** "열 너비 초기화" line above a table, only while its widths were changed */
export function ColReset(props: { cols: ColWidths }) {
  if (!props.cols.widths) return null;
  return (
    <div className="col-reset">
      <button type="button" className="chip link" onClick={props.cols.reset} title="이 표의 모든 열을 기본 너비로 되돌립니다">
        열 너비 초기화
      </button>
    </div>
  );
}
