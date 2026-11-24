/*
 * Update log
 * - [2026-11-24] type=test - Lower highs (downtrend) check of the rebound detector (shared/rebound.ts lowerHighs, user 2026-11-24):
 *   H1 > H2 > Hc with the % and tick thresholds, equal-high runs, the day / session limit, the RSI divergence, block / divExempt, and
 *   the reference case 448900 2026-11-20 14:55 (a real rebound under lower highs, kept by the divergence exemption)
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { lowerHighs, mergeConfig, krxTick, ticksDown, REBOUND_DEFAULTS, type ReboundConfig } from '../src/shared/rebound';
import { parseBars, replay } from '../src/shared/reboundReplay';
import type { Bar } from '../src/shared/price';

const KST = 9 * 3600000;
const T0 = Date.UTC(2026, 10, 24, 9, 0) - KST; // 2026-11-24 09:00 KST
const hm = (t: number) => new Date(t + KST).toISOString().slice(11, 16);

/** 46 one-minute bars from 09:00: H1 at bar 10, H2 at bar 25, the plunge at bar 35, the rebound 36..45 up to `hc` */
function scene(o: { h1?: number; h2?: number; hc?: number; t0?: number; h1Run?: boolean } = {}): Bar[] {
  const bars: Bar[] = [];
  for (let k = 0; k < 46; k++) bars.push({ t: (o.t0 ?? T0) + k * 60000, o: 9950, h: 10000, l: 9900, c: 9950, v: 100 });
  bars[10].h = o.h1 ?? 10500;
  if (o.h1Run) bars[11].h = bars[10].h;
  bars[25].h = o.h2 ?? 10300;
  bars[18].l = 9800; // the lowest low between H1 and H2
  bars[35] = { ...bars[35], l: 9700, c: 9750, v: 5000 }; // the plunge
  for (let k = 36; k <= 45; k++) bars[k].h = 9950;
  bars[44].h = o.hc ?? 10200;
  return bars;
}
const cfgOf = (lh: Partial<ReboundConfig['lowerHigh']> = {}) => mergeConfig({ ...REBOUND_DEFAULTS, lowerHigh: { ...REBOUND_DEFAULTS.lowerHigh, ...lh } });
const flatRsi = (n: number) => Array.from({ length: n }, () => 50);

describe('lowerHighs', () => {
  it('H1 > H2 > current high: lower highs, with the times and prices', () => {
    const b = scene();
    const r = lowerHighs(b, 45, 35, cfgOf(), flatRsi(46));
    expect(r.state).toBe('lh');
    expect([hm(r.h1t!), r.h1, hm(r.h2t!), r.h2, r.hc]).toEqual(['09:10', 10500, '09:25', 10300, 10200]);
    expect(r.blocked).toBe(false); // block is off by default: shown only
    expect(r.note).toContain('09:10 10,500 → 09:25 10,300 → 현재 10,200');
  });

  it('the rebound went above H2: not lower', () => {
    expect(lowerHighs(scene({ hc: 10350 }), 45, 35, cfgOf(), flatRsi(46)).state).toBe('no');
  });

  it('a step smaller than minDropPct or minTicks does not count', () => {
    // 10500 -> 10490 = 0.095 %
    expect(lowerHighs(scene({ h2: 10490, hc: 10400 }), 45, 35, cfgOf(), flatRsi(46)).state).toBe('no');
    // without the % limit: 10 won = 1 tick at this price, under minTicks 2
    expect(krxTick(10500)).toBe(10);
    expect(lowerHighs(scene({ h2: 10490, hc: 10400 }), 45, 35, cfgOf({ minDropPct: 0 }), flatRsi(46)).state).toBe('no');
    expect(lowerHighs(scene({ h2: 10480, hc: 10400 }), 45, 35, cfgOf({ minDropPct: 0 }), flatRsi(46)).state).toBe('lh');
  });

  it('fewer than two swing highs before the plunge: not judged', () => {
    const b = scene();
    b[25].h = 10000; // no H2
    const r = lowerHighs(b, 45, 35, cfgOf(), flatRsi(46));
    expect(r.state).toBe('na');
    expect(r.blocked).toBe(false);
  });

  it('a run of equal highs is one swing high, the last bar of the run standing for it', () => {
    const r = lowerHighs(scene({ h1Run: true }), 45, 35, cfgOf(), flatRsi(46));
    expect(r.state).toBe('lh');
    expect(hm(r.h1t!)).toBe('09:11');
  });

  it('swing highs of the previous day or another session window are not used', () => {
    // the same shape, but H1 / H2 on the day before: only today's bars (none before the plunge with two highs) count
    const prev = scene({ t0: T0 - 86400000 }).slice(0, 30);
    const today = scene().slice(30).map((x, k) => ({ ...x, t: T0 + k * 60000 }));
    const bars = [...prev, ...today];
    const i = bars.length - 1;
    const pj = prev.length + 5;
    expect(lowerHighs(bars, i, pj, cfgOf(), flatRsi(bars.length)).state).toBe('na');
    // NXT pre-market (08:00 - 08:50) highs are not used for a KRX bar
    const pre = scene({ t0: Date.UTC(2026, 10, 24, 8, 0) - KST }).slice(0, 30);
    const krx = scene().slice(30);
    const all = [...pre, ...krx];
    expect(lowerHighs(all, all.length - 1, pre.length + 5, cfgOf(), flatRsi(all.length)).state).toBe('na');
  });

  it('RSI bullish divergence: lower low, higher RSI', () => {
    const rsi = flatRsi(46);
    rsi[18] = 20; // the low between H1 and H2
    rsi[35] = 30; // the plunge bar (lower low)
    const r = lowerHighs(scene(), 45, 35, cfgOf(), rsi);
    expect(r.div).toBe(true);
    rsi[35] = 15;
    expect(lowerHighs(scene(), 45, 35, cfgOf(), rsi).div).toBe(false);
  });

  it('block: lower highs are blocked unless the divergence exempts them', () => {
    const rsi = flatRsi(46);
    expect(lowerHighs(scene(), 45, 35, cfgOf({ block: true }), rsi).blocked).toBe(true);
    rsi[18] = 20;
    rsi[35] = 30;
    expect(lowerHighs(scene(), 45, 35, cfgOf({ block: true }), rsi).blocked).toBe(false);
    expect(lowerHighs(scene(), 45, 35, cfgOf({ block: true, divExempt: false }), rsi).blocked).toBe(true);
    // not lower / not judged: never blocked
    expect(lowerHighs(scene({ hc: 10350 }), 45, 35, cfgOf({ block: true }), flatRsi(46)).blocked).toBe(false);
  });

  it('(Codex) not judged - and never blocked - outside the session windows or with too few swing highs, block on', () => {
    // 15:31 .. 15:39, between KRX and the NXT after-market, all of it: with pivotBars 1 it would read H1 > H2 > Hc
    const hs = [10000, 10500, 10000, 10300, 10000, 10000, 9950, 10200, 9950];
    const night: Bar[] = hs.map((h, k) => ({ t: Date.UTC(2026, 10, 24, 15, 31 + k) - KST, o: 9950, h, l: k === 5 ? 9700 : 9900, c: 9950, v: 100 }));
    const r = lowerHighs(night, 8, 5, cfgOf({ block: true, divExempt: false, pivotBars: 1 }), flatRsi(9));
    expect([r.state, r.blocked]).toEqual(['na', false]);
    // the same bars inside the after-market (15:41 ..): judged
    const day = night.map((b) => ({ ...b, t: b.t + 10 * 60000 }));
    expect(lowerHighs(day, 8, 5, cfgOf({ block: true, divExempt: false, pivotBars: 1 }), flatRsi(9))).toMatchObject({ state: 'lh', blocked: true });
    const one = scene();
    one[25].h = 10000;
    expect(lowerHighs(one, 45, 35, cfgOf({ block: true, divExempt: false }), flatRsi(46))).toMatchObject({ state: 'na', blocked: false });
  });

  it('(Codex) ticks counted with the tick of each price on the way down: 5,000 -> 4,995 -> 4,990', () => {
    expect(ticksDown(5000, 2)).toBe(4990);
    expect(ticksDown(5100, 2)).toBe(5080);
    expect(ticksDown(2000, 1)).toBe(1999);
    expect(ticksDown(5000, 1e20)).toBe(-Infinity); // (Codex) a huge minTicks ends at once (nothing is that far below)
    expect(ticksDown(5000, 1000)).toBe(1600); // 600 ticks of 5 down to 2,000, then 400 of 1
    expect(ticksDown(5000, 1001)).toBe(1599);
    expect(ticksDown(5000, 2600)).toBe(0);
    expect(ticksDown(5000, 2601)).toBe(-Infinity);
    expect(ticksDown(510000, 12)).toBe(499000); // 10 ticks of 1000 to 500,000, then 2 of 500
    // 5,100 -> 5,000 -> 4,990: the second step is two real ticks and 0.2 %
    const b: Bar[] = Array.from({ length: 46 }, (_, k) => ({ t: T0 + k * 60000, o: 4880, h: 4900, l: 4850, c: 4880, v: 100 }));
    b[10].h = 5100;
    b[25].h = 5000;
    b[35].l = 4700;
    b[44].h = 4990;
    expect(lowerHighs(b, 45, 35, cfgOf(), flatRsi(46)).state).toBe('lh');
    b[44].h = 4995; // one tick
    expect(lowerHighs(b, 45, 35, cfgOf({ minDropPct: 0 }), flatRsi(46)).state).toBe('no');
  });

  it('(Codex) a swing high needs pivotBars completed bars on its right: k + L = i is confirmed, k + L = i + 1 is not', () => {
    // H2 at bar 25, L = 5: bars 26..30 must exist. Plunge at 29, judged bar 30 -> confirmed; judged bar 29 (plunge 28) -> not
    const b = scene();
    b[29] = { ...b[29], l: 9700 };
    expect(lowerHighs(b, 30, 29, cfgOf(), flatRsi(46)).h2).toBe(10300);
    const c = scene();
    c[28] = { ...c[28], l: 9700 };
    expect(lowerHighs(c, 29, 28, cfgOf(), flatRsi(46)).state).toBe('na');
  });

  it('the defaults: shown only, divergence exempt, 5 bars, 150 bars, 0.2 %, 2 ticks', () => {
    expect(REBOUND_DEFAULTS.lowerHigh).toEqual({ block: false, divExempt: true, pivotBars: 5, rangeBars: 150, minDropPct: 0.2, minTicks: 2 });
  });
});

describe('reference case 448900 2026-11-20 14:55 (a real rebound)', () => {
  const ref = parseBars(readFileSync(join(__dirname, 'rebound_448900_20261120.csv'), 'utf8'));
  const T = (h: string) => Date.UTC(2026, 10, 20, +h.slice(0, 2), +h.slice(3, 5)) - KST;

  it('is under lower highs but has the RSI divergence', () => {
    const r = replay(ref, REBOUND_DEFAULTS, T('14:55'), T('14:55'))[0];
    expect(r.res.hit).toBe(true);
    expect(r.res.lowerHigh).toMatchObject({ state: 'lh', h1: 39050, h2: 38800, hc: 38700, div: true, blocked: false });
  });

  it('block on: still alerted thanks to the divergence; without the exemption it is held back', () => {
    const on = replay(ref, cfgOf({ block: true }), T('13:00'), T('15:20'));
    expect(on.filter((r) => r.alert).map((r) => hm(r.t))).toEqual(['14:55']);
    const strict = replay(ref, cfgOf({ block: true, divExempt: false }), T('13:00'), T('15:20'));
    expect(strict.filter((r) => r.alert)).toEqual([]);
    expect(strict.filter((r) => r.res.hit && r.res.lowerHigh?.blocked).length).toBeGreaterThan(0);
  });
});
