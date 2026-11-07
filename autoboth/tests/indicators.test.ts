/*
 * Update log
 * - [2026-11-06] type=test - Indicator parity with the bot: tests/indicators_ref.json was produced on the server by the live bot's
 *   live_trading/r002_strategy_core_shared.py calculate_indicators() (r001 periods) for a synthetic 40-bar series (one zero-volume bar)
 *   with flat stretches (zero loss -> RSI 100, zero range -> NaN DX); BB / RSI / MACD / DI / ADX must match to 1e-6. Ichimoku
 *   window/shift checks against hand-computed values
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { adx, bollinger, BOT_PARAMS, computeIndicators, ewm, ichimoku, macd, mfi, obv, rollingMean, rsi, stochastic, vwap, williams } from '../src/shared/indicators';
import type { Bar } from '../src/shared/price';

const ref = JSON.parse(readFileSync(join(__dirname, 'indicators_ref.json'), 'utf8')) as Record<string, (number | null)[]> & {
  bars: { open: number; high: number; low: number; close: number; volume: number }[];
};
const bars: Bar[] = ref.bars.map((b, i) => ({ t: i, o: b.open, h: b.high, l: b.low, c: b.close, v: b.volume }));
const close = bars.map((b) => b.c);

function same(name: string, got: number[]) {
  const want = ref[name];
  expect(got).toHaveLength(want.length);
  got.forEach((g, i) => {
    if (want[i] === null) expect(Number.isNaN(g), `${name}[${i}] should be NaN, got ${g}`).toBe(true);
    else expect(Math.abs(g - (want[i] as number)), `${name}[${i}] ${g} vs ${want[i]}`).toBeLessThan(1e-6);
  });
}

describe('indicator parity with the bot (pandas reference)', () => {
  it('Bollinger 20 / 2 (sample std)', () => {
    const b = bollinger(close);
    same('BB_MIDDLE', b.mid);
    same('BB_UPPER', b.upper);
    same('BB_LOWER', b.lower);
  });
  it('RSI 14 Wilder + signal 6', () => {
    const r = rsi(close);
    same('RSI', r.rsi);
    same('RSI_SIGNAL', r.signal);
  });
  it('MACD 5 / 12 / 4', () => {
    const m = macd(close);
    same('MACD', m.macd);
    same('MACD_SIGNAL', m.signal);
    same('MACD_HIST', m.hist);
  });
  it('ADX 7 / DI+ / DI-', () => {
    const a = adx(bars);
    same('DI_PLUS', a.diPlus);
    same('DI_MINUS', a.diMinus);
    same('ADX', a.adx);
  });
});

describe('more bot indicators (r002 reference)', () => {
  it('stochastic 10 / 5', () => {
    const k = stochastic(bars);
    same('STOCH_K', k.k);
    same('STOCH_D', k.d);
  });
  it('Williams %R 10 / 9', () => {
    const w = williams(bars);
    same('WILLIAMS_R', w.r);
    same('WILLIAMS_D', w.d);
  });
  it('MFI 14', () => same('MFI', mfi(bars)));
  it('VWAP (close weighted, zero-volume bar)', () => same('VWAP', vwap(bars)));
  it('OBV + OBV_MA 10', () => {
    const o = obv(bars);
    same('OBV', o.obv);
    same('OBV_MA', o.ma);
  });
  it('MA_5 (min_periods=1)', () => same('MA_5', rollingMean(close, 5)));
});

describe('computeIndicators with warm-up', () => {
  it('bot indicators continue from warmBot, VWAP/OBV restart at the day', () => {
    const warmBot = bars.slice(0, 25);
    const day = bars.slice(25);
    const s = computeIndicators(day, ['rsi', 'vwap', 'obv'], { warmBot });
    // same as computing over all 40 bars and keeping the last 15
    expect(s.rsi!.rsi).toEqual(rsi(close).rsi.slice(25));
    expect(s.vwap).toEqual(vwap(day));
    expect(s.obv!.obv[0]).toBe(0);
  });
});

describe('ewm (pandas adjust=False)', () => {
  it('starts at the first value and skips leading NaN', () => {
    expect(ewm([NaN, 2, 4], 0.5)).toEqual([NaN, 2, 3]);
  });
  it('a NaN gap decays the old weight', () => {
    // pandas 2.x on the server: Series([1, nan, 3]).ewm(alpha=0.5, adjust=False).mean() -> [1.0, 1.0, 2.3333333333333335]
    const r = ewm([1, NaN, 3], 0.5);
    expect(r[1]).toBe(1);
    expect(r[2]).toBeCloseTo(7 / 3, 12);
  });
});

describe('ichimoku', () => {
  const P = { ...BOT_PARAMS, ichiTenkan: 2, ichiKijun: 3, ichiSenkouB: 4, ichiShift: 2 };
  const b: Bar[] = [10, 12, 11, 14, 13, 15].map((c, i) => ({ t: i, o: c, h: c + 1, l: c - 1, c, v: 1 }));
  const k = ichimoku(b, P);
  it('full windows only', () => {
    expect(Number.isNaN(k.tenkan[0])).toBe(true);
    expect(k.tenkan[1]).toBe((13 + 9) / 2);
    expect(Number.isNaN(k.kijun[1])).toBe(true);
    expect(k.kijun[2]).toBe((13 + 9) / 2);
  });
  it('senkou spans shifted forward, chikou back', () => {
    expect(k.senkouA).toHaveLength(b.length + 2);
    expect(k.senkouA[4]).toBe((k.tenkan[2] + k.kijun[2]) / 2);
    expect(k.senkouB[5]).toBe((15 + 9) / 2); // window bars 0-3: high max 15, low min 9
    expect(k.chikou[0]).toBe(11);
    expect(Number.isNaN(k.chikou[4])).toBe(true);
  });
});
