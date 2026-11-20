// Update log
// - [2026-11-20] type=test (FloatPop placement: below / above, left / right-aligned, kept inside the view)

import { describe, expect, it } from 'vitest';
import { placePop } from '../src/renderer/src/components/FloatPop';

const a = { left: 900, top: 60, right: 1010, bottom: 90 };
describe('placePop', () => {
  it('opens below, left-aligned when there is room', () => {
    expect(placePop(a, 200, 100, 1400, 800)).toEqual({ left: 900, top: 93 });
  });
  it('right-aligns when it would run past the right edge', () => {
    expect(placePop(a, 360, 100, 1200, 800)).toEqual({ left: 650, top: 93 });
  });
  it('opens upward when there is more room above', () => {
    const b = { left: 10, top: 600, right: 60, bottom: 630 };
    expect(placePop(b, 100, 300, 1200, 800)).toEqual({ left: 10, top: 297 });
  });
  it('stays inside a view that is too small', () => {
    expect(placePop(a, 2000, 1000, 1200, 800)).toEqual({ left: 4, top: 4 });
  });
});
