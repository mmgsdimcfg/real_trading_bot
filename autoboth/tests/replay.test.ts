/*
 * Update log
 * - [2026-11-07] type=test - Past-day time replay (DayFeed in shared/replay.ts): the cut follows the log time, moving forward feeds
 *   only new lines, moving back rebuilds, null = whole day, live feeds keep no lines; trade marks; barsUpTo for the charts
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DayFeed, barsUpTo } from '../src/shared/replay';
import { kstToEpoch } from '../src/shared/time';
import type { ConditionsDoc } from '../src/shared/types';

const conditions = JSON.parse(readFileSync(join(__dirname, '../resources/conditions_snapshot.json'), 'utf8')) as ConditionsDoc;
const D = '2026-11-04';
const at = (hms: string) => kstToEpoch(D, hms);

const MAIN = [
  '2026-11-04 08:00:01,000 [INFO] R003 START dry_run=0',
  '2026-11-04 08:00:02,000 [INFO] WATCH | 159010 | 아스플로 | x',
  '2026-11-04 09:00:05,000 [INFO] [MKT REGIME  ] BULL | kospi',
  '  continuation of the line above',
  '2026-11-04 09:30:00,000 [INFO] [MKT REGIME  ] BEAR | kospi',
  '2026-11-04 15:00:00,000 [INFO] R003 STOP',
];
const BS = [
  '2026-11-04 09:09:26 [INFO] >>> 159010_아스플로 Buy - reason=TRIPLE_CROSS signal=None live=29,050 bb_mid=28485.0',
  '2026-11-04 09:09:26 [INFO] *** 매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17',
  '2026-11-04 09:15:56 [INFO] >>> 159010_아스플로 Sell - 수익금액=-10,413원 | 수익율=-2.10% | reason=HARD_STOP_LOSS_1.7PCT exch=KRX',
  '2026-11-04 09:15:56 [INFO] *** 매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17',
  '2026-11-04 09:15:56 [INFO] *** 매도시각=2026-11-04 09:15:56 | 매도가격=28,511 | 매도수량=17',
];

function feed(replay = true): DayFeed {
  const f = new DayFeed('20261104', conditions, { sessionOpen: '09:00' }, replay);
  f.setName('159010', '아스플로');
  f.feedMain(MAIN);
  f.feedBuySell(BS);
  return f;
}

describe('DayFeed replay', () => {
  it('whole day by default, with the log range and trade marks', () => {
    const s = feed().snapshot();
    expect(s.trades).toHaveLength(2);
    expect(s.regime).toBe('BEAR');
    expect(s.replay).toEqual({ first: at('08:00:01'), last: at('15:00:00'), cursor: null, trades: [at('09:09:26'), at('09:15:56')] });
  });

  it('cuts every part of the state at the cursor', () => {
    const f = feed();
    f.setCursor(at('09:10:00'));
    let s = f.snapshot();
    expect(s.replay?.cursor).toBe(at('09:10:00'));
    expect(s.regime).toBe('BULL');
    expect(s.trades.map((t) => t.side)).toEqual(['BUY']);
    expect(s.symbols.find((x) => x.code === '159010')?.position?.qty).toBe(17);
    expect(s.lastLineTs).toBe(at('09:00:05'));

    // forward: only the new lines
    f.setCursor(at('09:20:00'));
    s = f.snapshot();
    expect(s.trades.map((t) => t.side)).toEqual(['BUY', 'SELL']);
    expect(s.symbols.find((x) => x.code === '159010')?.position).toBeUndefined();
    expect(s.regime).toBe('BULL');

    // back: rebuilt from the first line
    f.setCursor(at('08:30:00'));
    s = f.snapshot();
    expect(s.trades).toHaveLength(0);
    expect(s.regime).toBeUndefined();
    expect(s.symbols.find((x) => x.code === '159010')?.name).toBe('아스플로');

    // null = whole day again
    f.setCursor(null);
    s = f.snapshot();
    expect(s.trades).toHaveLength(2);
    expect(s.replay?.cursor).toBeNull();
  });

  it('forward and rebuilt states are the same', () => {
    const a = feed();
    a.setCursor(at('09:00:00'));
    a.setCursor(at('09:31:00'));
    const b = feed();
    b.setCursor(at('15:00:00'));
    b.setCursor(at('09:31:00'));
    const strip = (s: ReturnType<DayFeed['snapshot']>) => ({ ...s, generatedAt: 0 });
    expect(strip(a.snapshot())).toEqual(strip(b.snapshot()));
  });

  it('lines arriving after a cursor was set stay in the whole-day state and are reached by moving on', () => {
    const f = new DayFeed('20261104', conditions, {}, true);
    f.feedBuySell(BS.slice(0, 2));
    f.setCursor(at('12:00:00'));
    f.feedBuySell(BS.slice(2));
    expect(f.snapshot().trades).toHaveLength(1);
    f.setCursor(at('12:00:01'));
    expect(f.snapshot().trades).toHaveLength(2);
  });

  it('a live feed keeps no lines and ignores the cursor', () => {
    const f = feed(false);
    f.setCursor(at('09:10:00'));
    const s = f.snapshot();
    expect(s.replay).toBeUndefined();
    expect(s.trades).toHaveLength(2);
  });
});

describe('barsUpTo', () => {
  const bars = [{ t: 1 }, { t: 2 }, { t: 3 }];
  it('drops intraday bars that start after the cursor', () => {
    expect(barsUpTo(bars, 2, true)).toEqual([{ t: 1 }, { t: 2 }]);
    expect(barsUpTo(bars, 0, true)).toEqual([]);
  });
  it('keeps everything without a cursor or for daily kinds', () => {
    expect(barsUpTo(bars, null, true)).toBe(bars);
    expect(barsUpTo(bars, 1, false)).toBe(bars);
  });
});
