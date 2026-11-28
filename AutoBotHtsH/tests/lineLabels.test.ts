// [2026-11-24] line name tags: placed on their line, away from candles / other lines / tags
import { describe, expect, it } from 'vitest';
import { placeLineLabels, type Box, type LabelLine } from '../src/renderer/src/components/lineLabels';

const area: Box = { x: 0, y: 0, w: 400, h: 200 };
const flat = (name: string, y: number): LabelLine => ({ name, pts: Array.from({ length: 81 }, (_, i) => [i * 5, y] as [number, number]) });
const hit = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

describe('placeLineLabels', () => {
  it('puts each tag next to its own line, newest side first', () => {
    const [t] = placeLineLabels([flat('MA5', 100)], [], [], area);
    expect(t.name).toBe('MA5');
    expect(t.y + t.h <= 100 || t.y >= 100).toBe(true);
    expect(Math.abs(t.y + (t.y < 100 ? t.h : 0) - 100)).toBeLessThanOrEqual(3);
    expect(t.x).toBeGreaterThan(300);
  });
  it('moves away from candles on the right and from the other tag', () => {
    const candles: Box[] = Array.from({ length: 20 }, (_, i) => ({ x: 300 + i * 5, y: 60, w: 3, h: 80 }));
    const tags = placeLineLabels([flat('MA5', 100), flat('MA20', 104)], candles, [], area);
    expect(tags).toHaveLength(2);
    for (const t of tags) expect(candles.some((c) => hit(c, t))).toBe(false);
    expect(hit(tags[0], tags[1])).toBe(false);
  });
  it('skips the legend block', () => {
    const [t] = placeLineLabels([flat('VWAP', 20)], [], [{ x: 0, y: 0, w: 400, h: 10 }, { x: 0, y: 0, w: 400, h: 16 }], area);
    expect(t.y).toBeGreaterThanOrEqual(16);
  });
});

describe('placeLineLabels value', () => {
  it('carries the value and widens the tag for it', () => {
    const [a] = placeLineLabels([flat('MA5', 100)], [], [], area);
    const [b] = placeLineLabels([{ ...flat('MA5', 100), value: '70,450' }], [], [], area);
    expect(b.value).toBe('70,450');
    expect(b.w).toBeGreaterThan(a.w);
  });
});

import { placeNoteBox } from '../src/renderer/src/components/lineLabels';

describe('placeNoteBox', () => {
  const pane: Box = { x: 0, y: 0, w: 600, h: 300 };
  const band = (y: number, h: number): Box[] => Array.from({ length: 120 }, (_, k) => ({ x: k * 5, y, w: 3, h }));
  it('goes under the candles when they fill the top of the pane', () => {
    const p = placeNoteBox({ x: 300, y: 30 }, 200, 60, band(10, 100), [], [], pane);
    expect(p.y).toBeGreaterThanOrEqual(110);
  });
  it('goes above the candles when they fill the bottom of the pane', () => {
    const p = placeNoteBox({ x: 300, y: 180 }, 200, 60, band(190, 100), [], [], pane);
    expect(p.y + 60).toBeLessThanOrEqual(190);
  });
  it('keeps off a line and a blocked box, and stays inside the pane', () => {
    const line: [number, number][] = Array.from({ length: 120 }, (_, k) => [k * 5, 250]);
    const p = placeNoteBox({ x: 590, y: 30 }, 200, 60, band(10, 100), line, [{ x: 0, y: 110, w: 600, h: 40 }], pane);
    expect(p.x + 200).toBeLessThanOrEqual(600);
    expect(p.y).toBeGreaterThanOrEqual(150);
    expect(p.y + 60).toBeLessThan(250 - 3);
  });
});
