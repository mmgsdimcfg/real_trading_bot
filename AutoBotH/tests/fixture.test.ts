/*
 * Update log
 * - [2026-11-04] type=test - Acceptance checks against the real 2026-11-04 log copy in fixtures/20261104/ (guide 7).
 *   Skipped when the fixture folder is not present (it holds trading data and is not committed)
 */

import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DayEngine } from '../src/shared/engine';
import { parseMainLine } from '../src/shared/logLine';
import { LineBuffer } from '../src/shared/lineBuffer';
import { classifyLogFile } from '../src/shared/files';
import type { ConditionsDoc } from '../src/shared/types';

const DIR = join(__dirname, '../fixtures/20261104');
const has = existsSync(DIR);
const conditions = JSON.parse(readFileSync(join(__dirname, '../resources/conditions_snapshot.json'), 'utf8')) as ConditionsDoc;

describe.skipIf(!has)('fixture 20261104', () => {
  const files = has ? readdirSync(DIR).sort() : [];
  const mains = files.filter((f) => classifyLogFile(f, '20261104').kind === 'main');
  const bs = files.find((f) => classifyLogFile(f, '20261104').kind === 'daily_buy_sell');

  const engine = new DayEngine('20261104', conditions);
  let lines = 0;
  for (const f of mains) {
    const lb = new LineBuffer();
    const ls = lb.push(readFileSync(join(DIR, f)));
    lines += ls.length;
    engine.feedMainLines(ls);
  }
  if (bs) engine.feedBuySellLines(new LineBuffer().push(readFileSync(join(DIR, bs))));
  const snap = engine.snapshot();

  it('(a) every main-log line parses or is classified (no exceptions)', () => {
    let n = 0;
    for (const f of mains) {
      for (const l of readFileSync(join(DIR, f), 'utf8').split('\n')) {
        expect(() => parseMainLine(l)).not.toThrow();
        n++;
      }
    }
    expect(n).toBeGreaterThan(50000);
    console.log(`main files=${mains.length} lines=${lines} unknownLines=${snap.warnings}`);
    console.table(snap.unknown.slice(0, 15).map((u) => ({ key: u.key, count: u.count })));
  });

  it('(c) 12 sells totalling -20,265 KRW', () => {
    const sells = snap.trades.filter((t) => t.side === 'SELL');
    const total = sells.reduce((a, t) => a + (t.pnlKrw || 0), 0);
    expect(sells).toHaveLength(12);
    expect(total).toBe(-20265);
  });

  it('acceptance: 159010 sold by _004 hard stop; _014 is the most blocked node (257)', () => {
    const sell = snap.trades.find((t) => t.code === '159010' && t.side === 'SELL');
    expect(sell?.condKey).toBe('_004');
    const ranked = Object.entries(snap.buyStats).sort((a, b) => b[1].block - a[1].block);
    console.table(ranked.slice(0, 8).map(([k, v]) => ({ cond: k, ...v })));
    expect(ranked[0][0]).toBe('_014');
    expect(snap.buyStats._014.block).toBe(257);
  });
});
