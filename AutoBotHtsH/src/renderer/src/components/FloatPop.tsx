// Update log
// - [2026-11-20] type=fix (A drop-down / popover inside a window (지표, 기간 설정, 관심종목 ☆) was cut off at the window's edge - the
//   window clips its content. It now floats over the app (position: fixed) next to its button and opens toward the side that has
//   room: below, or above when the space below is short; left-aligned with the button, or right-aligned when it would run past
//   the right edge; finally kept inside the app window - user 2026-11-20. Same file in HtsH and AutoBotHtsH)

import { useLayoutEffect, useRef, type HTMLAttributes, type ReactNode } from 'react';

const GAP = 3;
const MARGIN = 4;

export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** where a popover of size w x h goes next to the anchor box, inside a view of size vw x vh */
export function placePop(anchor: Box, w: number, h: number, vw: number, vh: number): { left: number; top: number } {
  let left = anchor.left;
  if (left + w > vw - MARGIN) left = anchor.right - w; // no room on the right: align the right edges
  left = Math.max(MARGIN, Math.min(left, vw - MARGIN - w));
  let top = anchor.bottom + GAP;
  const below = vh - MARGIN - top;
  const above = anchor.top - GAP - MARGIN;
  if (h > below && above > below) top = Math.max(MARGIN, anchor.top - GAP - h); // more room above: open upward
  top = Math.max(MARGIN, Math.min(top, vh - MARGIN - h));
  return { left, top };
}

/** a popover placed next to its parent element (the button's wrapper), floating over the windows */
export function FloatPop({ children, tag = 'span', style, ...rest }: HTMLAttributes<HTMLElement> & { children: ReactNode; tag?: 'span' | 'div' }) {
  const ref = useRef<HTMLElement | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const place = () => {
      const a = (el.parentElement || el).getBoundingClientRect();
      const { left, top } = placePop(a, el.offsetWidth, el.offsetHeight, window.innerWidth, window.innerHeight);
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
      el.style.visibility = 'visible';
    };
    place();
    const ro = new ResizeObserver(place);
    ro.observe(el);
    window.addEventListener('resize', place);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', place);
    };
  }, []);
  const Tag = tag;
  return (
    <Tag
      ref={ref as never}
      {...rest}
      style={{ ...style, position: 'fixed', left: 0, top: 0, right: 'auto', bottom: 'auto', marginTop: 0, maxHeight: `calc(100vh - ${MARGIN * 2}px)`, overflowY: 'auto', visibility: 'hidden' }}
    >
      {children}
    </Tag>
  );
}
