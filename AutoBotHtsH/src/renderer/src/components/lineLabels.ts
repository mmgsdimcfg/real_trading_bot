// Update log
// - [2026-11-24] type=feat (user 2026-11-24): LabelLine.value - the line's latest value shown after the name, counted in the tag width
// - [2026-11-24] type=feat (Name tags on the price-pane indicator lines (MA5 / VWAP / 볼린저 중 / 전환 / 기준 ...) (user 2026-11-24): each tag
//   is put on its own line where it covers nothing - no candle, no other line, no legend, no tag placed before it. Candidates run
//   from the newest bar to the oldest, above and below the line (3 / 9 / 16 / 24 px away); the first free spot wins, else the spot with the least overlap)

export interface LabelLine {
  name: string;
  /** [2026-11-24] value shown after the name (the line's latest value), counted in the tag width */
  value?: string;
  /** pixel points of the line, sorted by x (gaps already dropped) */
  pts: [number, number][];
  color?: string;
  cls?: string;
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PlacedLabel extends Box {
  name: string;
  value?: string;
  color?: string;
  cls?: string;
}

/** distance of a tag from its line, tried in this order */
const GAPS = [3, 9, 16, 24];

const hit = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** first index with pts[i][0] >= x */
function lowerBound(pts: [number, number][], x: number): number {
  let lo = 0;
  let hi = pts.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (pts[m][0] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/** does the line pass through the box (a point inside, or two neighbours on both sides of it vertically) */
function crosses(pts: [number, number][], b: Box): boolean {
  let i = Math.max(0, lowerBound(pts, b.x) - 1);
  for (; i < pts.length && pts[i][0] <= b.x + b.w; i++) {
    const [x, y] = pts[i];
    if (x >= b.x && y >= b.y && y <= b.y + b.h) return true;
    const n = pts[i + 1];
    if (n && n[0] >= b.x && x <= b.x + b.w && Math.min(y, n[1]) <= b.y + b.h && Math.max(y, n[1]) >= b.y) return true;
  }
  return false;
}

/** candle (wick) boxes sorted by x */
function candlesHit(candles: Box[], b: Box): number {
  let lo = 0;
  let hi = candles.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (candles[m].x + candles[m].w < b.x) lo = m + 1;
    else hi = m;
  }
  let n = 0;
  for (let i = lo; i < candles.length && candles[i].x <= b.x + b.w; i++) if (hit(candles[i], b)) n++;
  return n;
}

/**
 * Places one tag per line inside `area`. `candles` = bar boxes sorted by x, `blocked` = other things to avoid (legend, price tag).
 * Tag size: `charW` px per character + padding, `h` high.
 */
export function placeLineLabels(lines: LabelLine[], candles: Box[], blocked: Box[], area: Box, charW = 6.4, h = 14): PlacedLabel[] {
  const placed: PlacedLabel[] = [];
  const avoid = [...blocked];
  for (const ln of lines) {
    const pts = ln.pts.filter(([x, y]) => x >= area.x && x <= area.x + area.w && y >= area.y && y <= area.y + area.h);
    if (pts.length < 2) continue;
    const w = Math.ceil((ln.name.length + (ln.value ? ln.value.length + 1 : 0)) * charW + 8);
    const stride = Math.max(1, Math.floor(pts.length / 60));
    let best = null as { b: Box; cost: number } | null;
    for (let k = pts.length - 1, step = 0; k >= 0; k -= stride, step++) {
      const [x, y] = pts[k];
      // right above / below the line first, then a little further away (lines that hug the candles, like MA5, have no room
      // right next to them)
      for (const gap of GAPS) for (const dy of [-h - gap, gap]) {
        const b = { x: Math.min(Math.max(x - w / 2, area.x + 2), area.x + area.w - w - 2), y: y + dy, w, h };
        if (b.y < area.y || b.y + h > area.y + area.h) continue;
        let cost = candlesHit(candles, b) * 4;
        for (const o of avoid) if (hit(o, b)) cost += 20;
        for (const other of lines) if (other !== ln && crosses(other.pts, b)) cost += 6;
        if (crosses(pts, b)) cost += 3;
        cost += step * 0.02 + (gap - GAPS[0]) * 0.04; // nearer the newest bars / the line is better when the rest is equal
        if (!best || cost < best.cost) best = { b, cost };
      }
      if (best && best.cost < 1) break;
    }
    if (!best) continue;
    const p: PlacedLabel = { ...best.b, name: ln.name, value: ln.value, color: ln.color, cls: ln.cls };
    placed.push(p);
    avoid.push(p);
  }
  return placed;
}

/**
 * [2026-11-28] Where a note box of w x h goes near an anchor (the alert bell): the place inside `area` that covers the least of the
 * candles, the line points and the blocked boxes - an empty part of the pane above or below the anchor (user). Among equally
 * free places the one nearest to the anchor wins. Returns the box's top-left corner.
 */
export function placeNoteBox(anchor: { x: number; y: number }, w: number, h: number, candles: Box[], linePts: [number, number][], blocked: Box[], area: Box): { x: number; y: number } {
  const x0 = area.x + 2;
  const x1 = Math.max(x0, area.x + area.w - w - 2);
  const y0 = area.y + 2;
  const y1 = Math.max(y0, area.y + area.h - h - 2);
  const clampX = (x: number) => Math.min(x1, Math.max(x0, x));
  const xs = [...new Set([anchor.x + 12, anchor.x - 12 - w, anchor.x - w / 2, anchor.x - w / 4, anchor.x - (3 * w) / 4].map((x) => Math.round(clampX(x))))];
  const PAD = 3;
  let best = { x: xs[0], y: y0 };
  let bestCost = Infinity;
  for (const x of xs) {
    const inX = candles.filter((c) => c.x + c.w > x - PAD && c.x < x + w + PAD);
    const ptsX = linePts.filter(([px]) => px >= x - PAD && px <= x + w + PAD);
    for (let y = y0; y <= y1; y += 4) {
      let cover = 0;
      for (const c of inX) {
        const v = Math.min(c.y + c.h, y + h + PAD) - Math.max(c.y, y - PAD);
        if (v > 0) cover += v * Math.max(1, c.w);
      }
      for (const [, py] of ptsX) if (py >= y - PAD && py <= y + h + PAD) cover += 30;
      for (const b of blocked) {
        const ox = Math.min(b.x + b.w, x + w) - Math.max(b.x, x);
        const oy = Math.min(b.y + b.h, y + h) - Math.max(b.y, y);
        if (ox > 0 && oy > 0) cover += ox * oy;
      }
      // the bell itself stays visible
      if (anchor.x > x - 8 && anchor.x < x + w + 8 && anchor.y > y - 4 && anchor.y - 16 < y + h) cover += 2000;
      const dist = Math.abs(y + h / 2 - anchor.y) + Math.abs(x + w / 2 - anchor.x) * 0.5;
      const cost = cover * 1000 + dist;
      if (cost < bestCost) {
        bestCost = cost;
        best = { x, y };
      }
    }
  }
  return best;
}
