/*
 * Update log
 * - [2026-11-07] type=feat - Chart drawings (HTS-style tools): horizontal line at a price, vertical line at a time, trend line between
 *   two (time, price) points. Kept per symbol code in localStorage (best effort), shared by the timeline and chart tabs; a small
 *   subscribe() lets every chart of the same code redraw when one of them changes
 */

export type Drawing =
  | { id: string; type: 'h'; p: number }
  | { id: string; type: 'v'; t: number }
  | { id: string; type: 't'; t1: number; p1: number; t2: number; p2: number };

export type DrawTool = 'pointer' | 'h' | 'v' | 't';

const KEY = 'autoboth.drawings';
const listeners = new Set<() => void>();

function loadAll(): Record<string, Drawing[]> {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || '{}');
    return v && typeof v === 'object' ? v : {};
  } catch {
    return {};
  }
}

let all = loadAll();

function save(): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable - drawings stay for this session */
  }
  for (const l of listeners) l();
}

export const getDrawings = (code: string): Drawing[] => all[code] || [];

export function addDrawing(code: string, d: Omit<Drawing, 'id'>): void {
  const id = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  all = { ...all, [code]: [...getDrawings(code), { ...d, id } as Drawing] };
  save();
}

export function removeDrawing(code: string, id: string): void {
  all = { ...all, [code]: getDrawings(code).filter((d) => d.id !== id) };
  save();
}

export function clearDrawings(code: string): void {
  const { [code]: _gone, ...rest } = all;
  all = rest;
  save();
}

export function subscribeDrawings(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
