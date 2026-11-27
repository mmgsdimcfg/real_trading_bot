import { describe, expect, it } from 'vitest';
import { COL_MAX, COL_MIN, validColWidths } from '../../src/bot/renderer/src/components/useColWidths';

describe('validColWidths', () => {
  it('accepts n widths in range, rounded', () => {
    expect(validColWidths([40.4, 100, 250.6], 3)).toEqual([40, 100, 251]);
  });
  it('drops a list of another length, out of range values and non-numbers', () => {
    expect(validColWidths([40, 100], 3)).toBeNull();
    expect(validColWidths([COL_MIN - 1, 100, 100], 3)).toBeNull();
    expect(validColWidths([COL_MAX + 1, 100, 100], 3)).toBeNull();
    expect(validColWidths(['40', 100, 100], 3)).toBeNull();
    expect(validColWidths([NaN, 100, 100], 3)).toBeNull();
    expect(validColWidths(null, 3)).toBeNull();
  });
});
