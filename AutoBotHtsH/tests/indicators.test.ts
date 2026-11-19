/*
 * Update log
 * - [2026-11-06] type=test - Indicator parity with the bot: tests/indicators_ref.json was produced on the server by the live bot's
 *   live_trading/r002_strategy_core_shared.py calculate_indicators() (r001 periods) for a synthetic 40-bar series (one zero-volume bar)
 *   with flat stretches (zero loss -> RSI 100, zero range -> NaN DX); BB / RSI / MACD / DI / ADX must match to 1e-6. Ichimoku
 *   window/shift checks against hand-computed values
 * - [2026-11-06] type=test - HtsH indicators (envelope, price channel, disparity, ROC, psychological line, CCI, ATR, parabolic SAR) and
 *   the buy / sell signal rules (zone exits, signal-line crossings, stochastic zones, NaN gaps)
 * - [2026-11-09] type=test - HTS RSI (simple average, flat window = 50), RSI marks = crossing its signal line, MFI zone rule
 * - [2026-11-09] type=test - MTS method: RSI signal = EMA, ADX/DMI with EMA alpha 2/(n+1) (Wilder kept for the bot params);
 *   normalizeParams (whole periods, method names, 'simple' -> 'mts', bad values -> defaults)
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { adx, atr, bollinger, BOT_PARAMS, cci, computeIndicators, disparity, donchian, envelope, ewm, HTS_PARAMS, normalizeParams, ichimoku, macd, mfi, obv, parabolicSar, psychological, roc, rollingMean, rsi, signalLine, signals, stochastic, vwap, williams } from '../src/shared/indicators';
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

describe('HtsH indicators', () => {
  const P = { ...HTS_PARAMS, cciPeriod: 3, atrPeriod: 3, rocPeriod: 2, rocSignal: 2, disparity: 3, psyPeriod: 3, donchian: 2, envPeriod: 3, envPct: 10 };
  const b: Bar[] = [10, 12, 11, 14, 13, 15].map((c, i) => ({ t: i, o: c, h: c + 1, l: c - 1, c, v: 1 }));
  const c = b.map((x) => x.c);
  it('envelope = SMA +- pct', () => {
    const e = envelope(c, P);
    expect(Number.isNaN(e.mid[1])).toBe(true);
    expect(e.mid[2]).toBeCloseTo(11);
    expect(e.upper[2]).toBeCloseTo(12.1);
    expect(e.lower[2]).toBeCloseTo(9.9);
  });
  it('price channel uses the previous bars', () => {
    const d = donchian(b, P);
    expect(Number.isNaN(d.upper[1])).toBe(true);
    expect(d.upper[2]).toBe(13); // bars 0-1 highs 11, 13
    expect(d.lower[2]).toBe(9);
  });
  it('disparity / ROC / psychological line', () => {
    expect(disparity(c, P)[2]).toBeCloseTo((11 / 11) * 100);
    expect(roc(c, P).roc[2]).toBeCloseTo((11 / 10 - 1) * 100);
    expect(psychological(c, P)[3]).toBeCloseTo((2 / 3) * 100); // 12>10, 11<12, 14>11
  });
  it('CCI of a constant typical price is 0, ATR of constant ranges is the range', () => {
    const flat: Bar[] = Array.from({ length: 6 }, (_, i) => ({ t: i, o: 10, h: 11, l: 9, c: 10, v: 1 }));
    expect(cci(flat, P)[5]).toBe(0);
    expect(atr(flat, P)[5]).toBeCloseTo(2);
    expect(Number.isNaN(atr(flat, P)[1])).toBe(true);
  });
  it('parabolic SAR flips below / above the price', () => {
    const upDown: Bar[] = [...[10, 11, 12, 13, 14, 15], ...[12, 9, 7, 5]].map((x, i) => ({ t: i, o: x, h: x + 0.5, l: x - 0.5, c: x, v: 1 }));
    const s = parabolicSar(upDown, HTS_PARAMS);
    expect(s.trend[5]).toBe(1);
    expect(s.sar[5]).toBeLessThan(upDown[5].l);
    expect(s.trend[9]).toBe(-1);
    expect(s.sar[9]).toBeGreaterThan(upDown[9].h);
  });
  it('computeIndicators returns the new keys aligned to the bars', () => {
    const S = computeIndicators(b, ['env', 'donchian', 'sar', 'cci', 'atr', 'roc', 'trix', 'disparity', 'psy'], {}, P, false);
    for (const s of [S.env!.mid, S.donchian!.upper, S.sar!.sar, S.cci!, S.atr!, S.roc!.roc, S.trix!.trix, S.disparity!, S.psy!]) expect(s).toHaveLength(b.length);
  });
});

describe('signals', () => {
  it('MFI zone rules: leaving 20 is a buy, leaving 80 a sell', () => {
    expect(signals('mfi', { mfi: [50, 15, 25, 85, 75] })).toEqual([
      { i: 2, side: 'buy' },
      { i: 4, side: 'sell' },
    ]);
  });
  it('RSI crossing its signal line (like the MTS)', () => {
    const r = [40, 45, 55, 60, 50];
    const s = [50, 50, 50, 52, 52];
    expect(signals('rsi', { rsi: { rsi: r, signal: s } })).toEqual([
      { i: 2, side: 'buy' },
      { i: 4, side: 'sell' },
    ]);
  });
  it('HTS RSI = simple average of gains / losses over N bars', () => {
    const close = [10, 11, 10, 12, 12, 11];
    const r = rsi(close, { ...HTS_PARAMS, rsiPeriod: 3, rsiSignal: 2 });
    // i3: +1 -1 +2 -> 3/4; i4: -1 +2 0 -> 2/3; i5: +2 0 -1 -> 2/3
    expect(r.rsi.slice(0, 3).every((v) => Number.isNaN(v))).toBe(true);
    expect(r.rsi[3]).toBeCloseTo(75);
    expect(r.rsi[4]).toBeCloseTo(200 / 3);
    expect(r.rsi[5]).toBeCloseTo(200 / 3);
    expect(rsi([5, 5, 5, 5], { ...HTS_PARAMS, rsiPeriod: 3 }).rsi[3]).toBe(50);
  });
  it('MTS RSI signal is an EMA of the RSI (span = signal period), Wilder keeps the SMA', () => {
    const close = [10, 11, 10, 12, 12, 11, 13, 12];
    const P = { ...HTS_PARAMS, rsiPeriod: 3, rsiSignal: 3 };
    const r = rsi(close, P);
    const a = 2 / (3 + 1);
    let e = r.rsi[3];
    for (let i = 4; i < close.length; i++) e = e + (r.rsi[i] - e) * a;
    expect(r.signal[close.length - 1]).toBeCloseTo(e);
    const w = rsi(close, { ...P, rsiMethod: 'wilder' });
    expect(w.signal).toEqual(rollingMean(w.rsi, 3));
  });
  it('MTS ADX/DMI smooth TR, DM and DX with EMA alpha 2/(n+1); BOT_PARAMS stay Wilder', () => {
    const bars = [
      { t: 0, o: 10, h: 11, l: 9, c: 10, v: 1 },
      { t: 1, o: 10, h: 13, l: 10, c: 12, v: 1 },
      { t: 2, o: 12, h: 12, l: 8, c: 9, v: 1 },
      { t: 3, o: 9, h: 14, l: 9, c: 13, v: 1 },
    ];
    const n = 3;
    const ema = (x: number[], al: number) => x.reduce<number[]>((o, v, i) => (o.push(i ? o[i - 1] + (v - o[i - 1]) * al : v), o), []);
    const tr = [2, 3, 4, 5];
    const pdm = [0, 2, 0, 2];
    const mdm = [0, 0, 2, 0];
    const al = 2 / (n + 1);
    const T = ema(tr, al), Pp = ema(pdm, al), M = ema(mdm, al);
    const di = Pp.map((v, i) => (100 * v) / T[i]);
    const r = adx(bars, { ...HTS_PARAMS, adxPeriod: n });
    expect(r.diPlus[3]).toBeCloseTo(di[3]);
    expect(r.diMinus[3]).toBeCloseTo((100 * M[3]) / T[3]);
    expect(adx(bars, { ...HTS_PARAMS, adxPeriod: n, adxMethod: 'wilder' }).diPlus[3]).not.toBeCloseTo(di[3]);
    expect(BOT_PARAMS.adxMethod).toBe('wilder');
    expect(BOT_PARAMS.rsiMethod).toBe('wilder');
  });
  it('normalizeParams: whole periods, valid methods, defaults for bad values', () => {
    const p = normalizeParams({ rsiPeriod: 14.5, rsiSignal: 0, bbMult: 2.5, sarStep: 0, macdFast: 'x', rsiMethod: 'simple', adxMethod: 'foo' } as never);
    expect(p.rsiPeriod).toBe(15);
    expect(p.rsiSignal).toBe(1);
    expect(p.bbMult).toBe(2.5);
    expect(p.sarStep).toBe(0.01);
    expect(p.macdFast).toBe(HTS_PARAMS.macdFast);
    expect(p.rsiMethod).toBe('mts');
    expect(p.adxMethod).toBe(HTS_PARAMS.adxMethod);
    expect(normalizeParams(undefined)).toEqual(HTS_PARAMS);
  });
  it('MACD crossing its signal', () => {
    const m = [-1, -0.5, 0.5, 1, -1];
    const s = [0, 0, 0, 0, 0];
    expect(signals('macd', { macd: { macd: m, signal: s, hist: m } })).toEqual([
      { i: 2, side: 'buy' },
      { i: 4, side: 'sell' },
    ]);
  });
  it('stochastic crosses count only from the oversold / overbought zone', () => {
    const k = [10, 30, 50, 40, 85, 70];
    const d = [20, 20, 45, 45, 80, 80];
    // i1: K crosses up from 10 (<20) -> buy; i3: crosses down from 50 (not >80) -> none; i4 up from 40 -> none; i5 down from 85 -> sell
    expect(signals('stoch', { stoch: { k, d } })).toEqual([
      { i: 1, side: 'buy' },
      { i: 5, side: 'sell' },
    ]);
  });
  it('NaN gaps give no signal, ATR has none', () => {
    expect(signals('disparity', { disparity: [NaN, 99, NaN, 101] })).toEqual([]);
    expect(signals('atr', { atr: [1, 2, 3] })).toEqual([]);
    expect(signalLine('macd', { macd: { macd: [1], signal: [1], hist: [0] } })).toEqual([1]);
  });
});
