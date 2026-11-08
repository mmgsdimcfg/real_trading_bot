/*
 * Update log
 * - [2026-11-08] type=feat - Drag-to-resize heights: startHeightDrag (shared by the HTML grip bar below a chart and the SVG pane
 *   separators of PriceChart) and HeightGrip. Drag up/down changes the height, double-click restores the default size
 */

import type { MouseEvent as ReactMouseEvent } from 'react';

export const clampH = (h: number, min: number, max: number) => Math.round(Math.min(max, Math.max(min, h)));

/** follows the mouse until the button is released: onChange(startH + dy), clamped to [min, max] */
export function startHeightDrag(e: ReactMouseEvent, startH: number, min: number, max: number, onChange: (h: number) => void): void {
  if (e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();
  const y0 = e.clientY;
  const prevCursor = document.body.style.cursor;
  document.body.style.cursor = 'row-resize';
  const move = (ev: MouseEvent) => onChange(clampH(startH + ev.clientY - y0, min, max));
  const up = () => {
    window.removeEventListener('mousemove', move);
    window.removeEventListener('mouseup', up);
    document.body.style.cursor = prevCursor;
  };
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
}

/** thin bar under a block: drag to change the block's height, double-click = default size */
export function HeightGrip(props: { height: number; min: number; max: number; onChange: (h: number) => void; onReset: () => void }) {
  const { height, min, max, onChange, onReset } = props;
  return (
    <div
      className="height-grip"
      title="위아래로 끌어 높이 조절 · 더블클릭: 기본 높이"
      onMouseDown={(e) => startHeightDrag(e, height, min, max, onChange)}
      onDoubleClick={onReset}
    />
  );
}
