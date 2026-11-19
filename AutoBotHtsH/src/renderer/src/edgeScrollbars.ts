// Update log
// - [2026-11-20] type=feat (Thin, hidden scrollbars (user 2026-11-20): every scroll area still scrolls with the mouse wheel, but its
//   scrollbar (6 px, styles.css) is drawn only while the cursor is at the area's right edge (vertical bar) or bottom edge (horizontal
//   bar), and while a scrollbar is being dragged. One document listener marks the scroll area under the cursor with .sb-y / .sb-x)

/** distance from the right / bottom edge (px) that shows the scrollbar */
const EDGE = 14;

const scrollsY = (el: HTMLElement) => el.scrollHeight > el.clientHeight + 1 && /(auto|scroll)/.test(getComputedStyle(el).overflowY);
const scrollsX = (el: HTMLElement) => el.scrollWidth > el.clientWidth + 1 && /(auto|scroll)/.test(getComputedStyle(el).overflowX);

/** installs the listener; returns the remover */
export function installEdgeScrollbars(): () => void {
  let shown: HTMLElement[] = [];
  let frame = 0;
  let last: MouseEvent | null = null;

  const update = () => {
    frame = 0;
    const e = last;
    if (!e) return;
    // dragging a scrollbar: keep what is shown until the button is released
    if (e.buttons & 1 && shown.length) return;
    const next = new Map<HTMLElement, string[]>();
    let y = false;
    let x = false;
    for (let el = e.target instanceof HTMLElement ? e.target : null; el && el !== document.body && !(y && x); el = el.parentElement) {
      const r = el.getBoundingClientRect();
      const c: string[] = [];
      if (!y && e.clientX >= r.right - EDGE && e.clientX <= r.right && scrollsY(el)) (c.push('sb-y'), (y = true));
      if (!x && e.clientY >= r.bottom - EDGE && e.clientY <= r.bottom && scrollsX(el)) (c.push('sb-x'), (x = true));
      if (c.length) next.set(el, c);
    }
    for (const el of shown) if (!next.has(el)) el.classList.remove('sb-y', 'sb-x');
    for (const [el, c] of next) {
      el.classList.toggle('sb-y', c.includes('sb-y'));
      el.classList.toggle('sb-x', c.includes('sb-x'));
    }
    shown = [...next.keys()];
  };
  const onMove = (e: MouseEvent) => {
    last = e;
    if (!frame) frame = requestAnimationFrame(update);
  };
  const onUp = (e: MouseEvent) => onMove(e);
  const onLeave = () => {
    for (const el of shown) el.classList.remove('sb-y', 'sb-x');
    shown = [];
  };

  document.addEventListener('mousemove', onMove, { passive: true });
  document.addEventListener('mouseup', onUp, { passive: true });
  document.documentElement.addEventListener('mouseleave', onLeave);
  return () => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    document.documentElement.removeEventListener('mouseleave', onLeave);
    if (frame) cancelAnimationFrame(frame);
    onLeave();
  };
}
