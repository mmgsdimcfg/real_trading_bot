/*
 * Update log
 * - [2026-11-05] type=test - Timeline price chart data: tick CSV parsing (KST trade time, incremental lines), 10-second file ->
 *   real 1-minute bars (raw_bar=1 only), daily CSV with BOM, N-tick / M-minute bars, minute resampling, weekly / monthly grouping,
 *   the viewed day's bar from intraday data, and the period check used for the highlight
 */

import { describe, expect, it } from 'vitest';
import {
  dailyWithToday,
  minuteBarsFromTicks,
  monthBars,
  parse10sMinuteBars,
  parseDailyCsv,
  parseTickLines,
  resampleMinuteBars,
  samePeriod,
  tickBars,
  tickHeader,
  weekBars,
  ymdEpoch,
  type Tick,
} from '../src/shared/price';
import { fmtKstDateTime } from '../src/shared/time';

const TICK_HEAD = 'recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,bidp1,hour_cls_code';
const tick = (time: string, price: number, vol: number) => `2026-11-05T${time.slice(0, 2)}:${time.slice(2, 4)}:${time.slice(4, 6)}.100,003490,20261105,${time},${price},${vol},0,1,0,0,0,0,0`;

describe('tick CSV', () => {
  it('parses trade time as KST and skips the header / broken lines', () => {
    const cols = tickHeader('﻿' + TICK_HEAD);
    const ticks = parseTickLines([tick('090027', 29650, 5), 'garbage', '', tick('090101', 29700, 3)], cols);
    expect(ticks).toHaveLength(2);
    expect(fmtKstDateTime(ticks[0].t)).toBe('2026-11-05 09:00:27');
    expect(ticks[1]).toMatchObject({ p: 29700, v: 3 });
  });
});

const T = (hhmmss: string) => ymdEpoch('20261105') + (+hhmmss.slice(0, 2) * 3600 + +hhmmss.slice(2, 4) * 60 + +hhmmss.slice(4, 6)) * 1000;
const ticks: Tick[] = [
  { t: T('090005'), p: 100, v: 1 },
  { t: T('090030'), p: 105, v: 2 },
  { t: T('090059'), p: 98, v: 3 },
  { t: T('090100'), p: 101, v: 4 },
  { t: T('090430'), p: 110, v: 5 },
];

describe('bar builders', () => {
  it('N-tick bars', () => {
    const b = tickBars(ticks, 2);
    expect(b).toHaveLength(3);
    expect(b[0]).toMatchObject({ o: 100, h: 105, l: 100, c: 105, v: 3, t: T('090005'), t2: T('090030') });
    expect(b[2]).toMatchObject({ o: 110, c: 110, v: 5 });
  });

  it('1-minute and 3-minute bars from ticks (aligned to KST minutes)', () => {
    const m1 = minuteBarsFromTicks(ticks, 1);
    expect(m1.map((b) => fmtKstDateTime(b.t).slice(11, 16))).toEqual(['09:00', '09:01', '09:04']);
    expect(m1[0]).toMatchObject({ o: 100, h: 105, l: 98, c: 98, v: 6 });
    const m3 = minuteBarsFromTicks(ticks, 3);
    expect(m3.map((b) => fmtKstDateTime(b.t).slice(11, 16))).toEqual(['09:00', '09:03']);
    expect(m3[0]).toMatchObject({ o: 100, h: 105, l: 98, c: 101, v: 10 });
  });

  it('resamples 1-minute bars', () => {
    const m5 = resampleMinuteBars(minuteBarsFromTicks(ticks, 1), 5);
    expect(m5).toHaveLength(1);
    expect(m5[0]).toMatchObject({ o: 100, h: 110, l: 98, c: 110, v: 15 });
  });
});

describe('10-second file', () => {
  it('keeps only raw_bar=1 rows (real 1-minute bars)', () => {
    const text = [
      '﻿datetime,open,high,low,close,volume,market,MA_5,raw_bar',
      '2026-11-05 09:00:00,57700,58400,57700,58400,651.8,J,,1',
      '2026-11-05 09:00:10,57767,58500,57767,58500,651.8,J,,0',
      '2026-11-05 09:01:00,58400,58600,58300,58500,120,J,,1',
    ].join('\n');
    const b = parse10sMinuteBars(text);
    expect(b).toHaveLength(2);
    expect(b[1]).toMatchObject({ o: 58400, h: 58600, l: 58300, c: 58500, v: 120 });
    expect(fmtKstDateTime(b[1].t)).toBe('2026-11-05 09:01:00');
  });
});

describe('daily bars', () => {
  const csv = [
    '﻿date,open,high,low,close,volume,amount,MA_5',
    '20261030,100,110,95,105,10,0,0', // Fri
    '20261102,105,120,100,118,20,0,0', // Mon
    '20261103,118,119,110,112,30,0,0',
    '20261105,112,125,111,124,40,0,0', // Thu
  ].join('\r\n');
  const daily = parseDailyCsv(csv);

  it('parses with BOM and CRLF', () => {
    expect(daily).toHaveLength(4);
    expect(daily[0].t).toBe(ymdEpoch('20261030'));
  });

  it('weekly bars group Mon-Sun', () => {
    const w = weekBars(daily);
    expect(w).toHaveLength(2);
    expect(w[1]).toMatchObject({ o: 105, h: 125, l: 100, c: 124, v: 90, t: ymdEpoch('20261102') });
  });

  it('monthly bars', () => {
    const m = monthBars(daily);
    expect(m).toHaveLength(2);
    expect(m[0]).toMatchObject({ o: 100, c: 105 });
    expect(m[1]).toMatchObject({ o: 105, h: 125, l: 100, c: 124 });
  });

  it('adds the viewed day from intraday bars when the daily file has no row for it', () => {
    const upTo = dailyWithToday(daily.slice(0, 3), '20261105', minuteBarsFromTicks(ticks, 1));
    expect(upTo).toHaveLength(4);
    expect(upTo[3]).toMatchObject({ t: ymdEpoch('20261105'), o: 100, h: 110, l: 98, c: 110, v: 15 });
    // pre-market trades (before 09:00) are not part of the day's bar
    const pre = dailyWithToday(daily.slice(0, 3), '20261105', minuteBarsFromTicks([{ t: T('083005'), p: 90, v: 9 }, ...ticks], 1));
    expect(pre[3]).toMatchObject({ o: 100, l: 98, v: 15 });
    // already present -> unchanged; rows after the day are dropped
    expect(dailyWithToday(daily, '20261103', [])).toHaveLength(3);
  });

  it('highlight period check', () => {
    expect(samePeriod('day', ymdEpoch('20261105'), '20261105')).toBe(true);
    expect(samePeriod('week', ymdEpoch('20261102'), '20261105')).toBe(true);
    expect(samePeriod('week', ymdEpoch('20261030'), '20261105')).toBe(false);
    expect(samePeriod('month', ymdEpoch('20261102'), '20261105')).toBe(true);
  });
});

describe('server bars (r016 bars output)', () => {
  it('converts rows by column name; day rows are KST midnight, minute rows keep their time', async () => {
    const { serverRowsToBars } = await import('../src/shared/price');
    const day = serverRowsToBars(['t', 'o', 'h', 'l', 'c', 'v'], [['2025-09-03T00:00:00', 23500, 23750, 23350, 23450, 621917]]);
    expect(day[0]).toMatchObject({ t: ymdEpoch('20250903'), o: 23500, c: 23450, v: 621917 });
    const min = serverRowsToBars(['t', 'o', 'h', 'l', 'c', 'v'], [['2026-11-05T15:19:38', 30350, 30350, 30250, 30350, 104708], ['bad', 1, 1, 1, 1, 1]]);
    expect(min).toHaveLength(1);
    expect(fmtKstDateTime(min[0].t)).toBe('2026-11-05 15:19:38');
    // reordered columns
    const re = serverRowsToBars(['c', 't', 'o', 'h', 'l', 'v'], [[5, '2026-11-05', 4, 6, 3, 10]]);
    expect(re[0]).toMatchObject({ o: 4, h: 6, l: 3, c: 5, v: 10, t: ymdEpoch('20261105') });
  });
});

// [2026-11-11] tick-collector gaps and the KIS minute fill
import { gapMinutes, gapText, mergeMinuteBars, minuteGaps, sessionRange } from '../src/shared/price';

const MIN = 60000;
const bar = (t: number, c = 100): { t: number; o: number; h: number; l: number; c: number; v: number } => ({ t, o: c, h: c, l: c, c, v: 1 });

describe('minuteGaps', () => {
  const ymd = '20261111';
  const [open] = sessionRange(ymd);
  it('finds a run of missing minutes inside the session and ignores short holes', () => {
    const m1 = [];
    for (let m = 0; m < 60; m++) {
      if (m >= 24 && m < 40) continue; // 09:24-09:39 missing (collector down)
      if (m === 50) continue; // one quiet minute
      m1.push(bar(open + m * MIN));
    }
    const gaps = minuteGaps(m1, ymd, open + 60 * MIN);
    expect(gaps).toEqual([[open + 24 * MIN, open + 40 * MIN]]);
    expect(gapMinutes(gaps)).toBe(16);
    expect(gapText(gaps)).toBe('09:24~09:40');
  });
  it('counts a late start and a dead tail, stops one minute before `until`', () => {
    const m1 = [bar(open + 5 * MIN), bar(open + 6 * MIN)];
    // scanning until 09:12 -> the scan ends at 09:11 (one minute before `until`), so the tail 09:07..09:11 (5 minutes) counts
    expect(minuteGaps(m1, ymd, open + 12 * MIN)).toEqual([
      [open, open + 5 * MIN],
      [open + 7 * MIN, open + 12 * MIN],
    ]);
  });
  it('is empty before the session and for complete bars', () => {
    expect(minuteGaps([], ymd, open)).toEqual([]);
    const m1 = Array.from({ length: 30 }, (_, m) => bar(open + m * MIN));
    expect(minuteGaps(m1, ymd, open + 30 * MIN)).toEqual([]);
  });
});

describe('mergeMinuteBars', () => {
  const t0 = Date.UTC(2026, 10, 11, 0, 0); // 09:00 KST
  it('uses KIS bars as the base and tick bars for KIS-less minutes and from the last KIS minute on', () => {
    const kis = [bar(t0, 1), bar(t0 + MIN, 2), bar(t0 + 2 * MIN, 3), bar(t0 + 3 * MIN, 4)];
    const ticks = [bar(t0, 11), bar(t0 + 3 * MIN, 44), bar(t0 + 4 * MIN, 55)];
    const out = mergeMinuteBars(kis, ticks);
    expect(out.map((b) => b.c)).toEqual([1, 2, 3, 44, 55]);
  });
  it('keeps tick bars where KIS has none (a minute KIS skipped)', () => {
    const kis = [bar(t0, 1), bar(t0 + 2 * MIN, 3)];
    const ticks = [bar(t0 + MIN, 22)];
    expect(mergeMinuteBars(kis, ticks).map((b) => b.c)).toEqual([1, 22, 3]);
  });
});
