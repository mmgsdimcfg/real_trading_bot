/*
 * Update log
 * - [2026-11-06] type=feat - Chart indicators computed the same way as the bot (live_trading/r002_strategy_core_shared.py
 *   calculate_indicators / data_simulation/shared/indicators.py): Bollinger (rolling mean + sample std ddof=1, min_periods=1),
 *   RSI with Wilder smoothing (ewm alpha=1/N, adjust=False) + simple-mean signal, MACD (ewm span, adjust=False), ADX / DI+ / DI-
 *   (Wilder). ewm() follows pandas' adjust=False, ignore_na=False loop exactly (NaN gaps decay the old weight), so values match
 *   the bot's columns. Default periods = r001 (BB 20/2, RSI 14/6, MACD 5/12/4, ADX 7). Ichimoku (not used by the bot): standard
 *   9/26/52 with full windows, senkou spans shifted 26 bars forward, chikou 26 back. Pure functions (tests/indicators.test.ts)
 * - [2026-11-06] type=feat - (HtsH copy) HTS_PARAMS: HTS default periods; every function still takes the params as an argument
 * - [2026-11-06] type=feat - Stochastic (10/5), Williams %R (10/9), MFI (14, bot's 50/100/0 overrides), OBV + OBV_MA (10), VWAP
 *   (close-weighted, session) and MA lines 5/20/60/120 for the chart tab. computeIndicators() now takes the warm-up bars and returns
 *   series aligned to the day's bars, mirroring the bot's calculate_indicators_with_warmup(): bot indicators over the previous
 *   day's last PRIOR_DAY_WARMUP_MINUTES (180) one-minute bars + the day, VWAP/OBV over the day only
 * - [2026-11-06] type=feat - (HtsH) more HTS indicators: overlays Envelope (20, 6%), price channel (Donchian 20, previous bars),
 *   Parabolic SAR (0.02 / 0.2); panes CCI (20), ATR (14, Wilder), ROC (12 + signal 9), TRIX (12 + signal 9), disparity (20) and
 *   psychological line (10). signals(): buy / sell points of each pane by the usual HTS rules (crossing the signal line, a zero /
 *   100 line or leaving the overbought / oversold zone) for the triangle marks
 * - [2026-11-09] type=fix - RSI like the Korean HTS / MTS (checked against Samsung Securities MTS, GS E&C 2026-11-09: RSI(14) 56.16):
 *   rsiMethod 'simple' = sum of gains / sum of gains + losses over the last N bars (HTS_PARAMS default; the bot keeps 'wilder'),
 *   HTS signal period 9 -> 6, RSI buy / sell marks = RSI crossing its signal line (was: leaving 30 / 70 only, so almost never shown)
 * - [2026-11-09] type=fix - MTS method checked against the MTS screen (GS E&C, KIS combined daily bars): RSI signal is an EMA (span 6) -
 *   its crossings land on all 17 MTS arrows, the SMA gave extra ones; ADX/DMI smooth TR / DM and DX with EMA (alpha 2/(n+1)): ADX 36.11
 *   exact and the three lines fit the MTS pane within 3.4 px (Wilder: 23 px). rsiMethod / adxMethod 'mts' | 'wilder' ('simple' -> 'mts')
 * - [2026-11-20] type=feat - Pane 'investor' (투자자 누적, user 2026-11-20): cumulative net buying of 개인 / 외국인 / 기관 over the last
 *   1 / 3 / 6 / 12 months from daily investor rows, aligned to day / week / month / year bars (investorCum; not for intraday charts)
 */

import type { Bar } from './price';

export type Series = number[];
export type CalcMethod = 'mts' | 'wilder';

export interface IndicatorParams {
  bbPeriod: number;
  bbMult: number;
  rsiPeriod: number;
  rsiSignal: number;
  /** 'mts' = gains / (gains + losses) over the last N bars + EMA signal (Korean MTS / HTS), 'wilder' = Wilder smoothing + SMA signal (the bot) */
  rsiMethod: CalcMethod;
  macdFast: number;
  macdSlow: number;
  macdSignal: number;
  adxPeriod: number;
  /** 'mts' = EMA (alpha 2/(n+1)) of TR / DM and of DX (Korean MTS), 'wilder' = Wilder smoothing alpha 1/n (the bot) */
  adxMethod: CalcMethod;
  ichiTenkan: number;
  ichiKijun: number;
  ichiSenkouB: number;
  /** forward shift of the senkou spans / backward shift of chikou */
  ichiShift: number;
  stochK: number;
  stochD: number;
  williamsR: number;
  williamsD: number;
  mfiPeriod: number;
  obvMa: number;
  /** r001 PRIOR_DAY_WARMUP_MINUTES: the bot prepends the previous day's last N one-minute bars */
  warmupMinutes: number;
  envPeriod: number;
  envPct: number;
  donchian: number;
  sarStep: number;
  sarMax: number;
  cciPeriod: number;
  atrPeriod: number;
  rocPeriod: number;
  rocSignal: number;
  trixPeriod: number;
  trixSignal: number;
  disparity: number;
  psyPeriod: number;
}

/** HtsH-only indicator defaults shared by both parameter sets */
const EXTRA_PARAMS = {
  envPeriod: 20,
  envPct: 6,
  donchian: 20,
  sarStep: 0.02,
  sarMax: 0.2,
  cciPeriod: 20,
  atrPeriod: 14,
  rocPeriod: 12,
  rocSignal: 9,
  trixPeriod: 12,
  trixSignal: 9,
  disparity: 20,
  psyPeriod: 10,
};

/** r001_define_config.py values (BB_PERIOD, BB_STD_MULTIPLIER, RSI_PERIOD, RSI_SIGNAL_PERIOD, MACD_*, ADX_PERIOD) */
/** HTS-style defaults (MACD 12/26/9, ADX 14, slow stochastic 14/3, ...) - the user can change them in the chart settings */
export const HTS_PARAMS: IndicatorParams = {
  bbPeriod: 20,
  bbMult: 2,
  rsiPeriod: 14,
  rsiSignal: 6,
  rsiMethod: 'mts',
  macdFast: 12,
  macdSlow: 26,
  macdSignal: 9,
  adxPeriod: 14,
  adxMethod: 'mts',
  ichiTenkan: 9,
  ichiKijun: 26,
  ichiSenkouB: 52,
  ichiShift: 26,
  stochK: 14,
  stochD: 3,
  williamsR: 14,
  williamsD: 9,
  mfiPeriod: 14,
  obvMa: 9,
  warmupMinutes: 0,
  ...EXTRA_PARAMS,
};

export const BOT_PARAMS: IndicatorParams = {
  bbPeriod: 20,
  bbMult: 2,
  rsiPeriod: 14,
  rsiSignal: 6,
  rsiMethod: 'wilder',
  macdFast: 5,
  macdSlow: 12,
  macdSignal: 4,
  adxPeriod: 7,
  adxMethod: 'wilder',
  ichiTenkan: 9,
  ichiKijun: 26,
  ichiSenkouB: 52,
  ichiShift: 26,
  stochK: 10,
  stochD: 5,
  williamsR: 10,
  williamsD: 9,
  mfiPeriod: 14,
  obvMa: 10,
  warmupMinutes: 180,
  ...EXTRA_PARAMS,
};

const isNum = (x: number) => x === x; // not NaN

/** parameters that may be fractional (multiplier / percent / SAR step) - every other number is a period (whole bars) */
export const FRACTION_PARAMS: Partial<Record<keyof IndicatorParams, number>> = { bbMult: 0.5, envPct: 0.5, sarStep: 0.01, sarMax: 0.05 };

/**
 * Saved / typed parameters made safe: unknown or missing values fall back to `base`, periods become whole numbers >= 1 (a period like
 * 14.5 indexed between bars and every value came out NaN), fractional ones stay >= their minimum, methods are 'mts' | 'wilder'
 */
export function normalizeParams(p: Partial<Record<keyof IndicatorParams, unknown>> | undefined, base: IndicatorParams = HTS_PARAMS): IndicatorParams {
  const out = { ...base } as IndicatorParams;
  const w = out as unknown as Record<string, unknown>;
  for (const k of Object.keys(base) as (keyof IndicatorParams)[]) {
    const v = p?.[k];
    if (v === undefined) continue;
    const def = base[k];
    if (typeof def === 'number') {
      const n = Number(v);
      if (!Number.isFinite(n)) continue;
      const min = FRACTION_PARAMS[k];
      w[k] = min !== undefined ? Math.max(min, n) : k === 'warmupMinutes' ? Math.max(0, Math.round(n)) : Math.max(1, Math.round(n));
    } else if (v === 'mts' || v === 'wilder') w[k] = v;
    else if (v === 'simple') w[k] = 'mts'; // 0.1.0_004 name
  }
  return out;
}

/** pandas Series.rolling(window, min_periods).mean() */
export function rollingMean(x: Series, window: number, minPeriods = 1): Series {
  const out: Series = new Array(x.length).fill(NaN);
  for (let i = 0; i < x.length; i++) {
    let s = 0;
    let n = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) if (isNum(x[j])) (s += x[j]), n++;
    if (n >= minPeriods && n > 0) out[i] = s / n;
  }
  return out;
}

/** pandas rolling(window, min_periods).std() - sample std (ddof = 1), NaN for a single value */
export function rollingStd(x: Series, window: number, minPeriods = 1): Series {
  const out: Series = new Array(x.length).fill(NaN);
  for (let i = 0; i < x.length; i++) {
    const v: number[] = [];
    for (let j = Math.max(0, i - window + 1); j <= i; j++) if (isNum(x[j])) v.push(x[j]);
    if (v.length < Math.max(minPeriods, 2)) continue;
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    out[i] = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
  }
  return out;
}

function rollingExtreme(x: Series, window: number, max: boolean, minPeriods: number): Series {
  const out: Series = new Array(x.length).fill(NaN);
  for (let i = 0; i < x.length; i++) {
    let best = NaN;
    let n = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) {
      if (!isNum(x[j])) continue;
      n++;
      if (!isNum(best) || (max ? x[j] > best : x[j] < best)) best = x[j];
    }
    if (n >= minPeriods) out[i] = best;
  }
  return out;
}

/** pandas ewm(alpha=..., adjust=False, ignore_na=False, min_periods).mean() */
export function ewm(x: Series, alpha: number, minPeriods = 0): Series {
  const out: Series = new Array(x.length).fill(NaN);
  if (!x.length) return out;
  let weighted = x[0];
  let nobs = isNum(weighted) ? 1 : 0;
  let oldWt = 1;
  out[0] = nobs >= Math.max(1, minPeriods) ? weighted : NaN;
  for (let i = 1; i < x.length; i++) {
    const cur = x[i];
    const obs = isNum(cur);
    if (obs) nobs++;
    if (isNum(weighted)) {
      oldWt *= 1 - alpha;
      if (obs) {
        if (weighted !== cur) weighted = (oldWt * weighted + alpha * cur) / (oldWt + alpha);
        oldWt = 1;
      }
    } else if (obs) weighted = cur;
    out[i] = nobs >= Math.max(1, minPeriods) ? weighted : NaN;
  }
  return out;
}

const ewmSpan = (x: Series, span: number) => ewm(x, 2 / (span + 1));

export interface Bollinger {
  mid: Series;
  upper: Series;
  lower: Series;
}

export function bollinger(close: Series, p = BOT_PARAMS): Bollinger {
  const mid = rollingMean(close, p.bbPeriod);
  const sd = rollingStd(close, p.bbPeriod);
  return { mid, upper: mid.map((m, i) => m + sd[i] * p.bbMult), lower: mid.map((m, i) => m - sd[i] * p.bbMult) };
}

export function rsi(close: Series, p = BOT_PARAMS): { rsi: Series; signal: Series } {
  const delta = close.map((c, i) => (i ? c - close[i - 1] : NaN));
  if (p.rsiMethod === 'mts') {
    // Korean MTS / HTS: 100 * (sum of gains) / (sum of gains + sum of losses) over the last N changes; a flat window is 50; EMA signal
    const n = p.rsiPeriod;
    const r = close.map((_, i) => {
      if (i < n) return NaN;
      let g = 0;
      let l = 0;
      for (let j = i - n + 1; j <= i; j++) {
        const d = delta[j];
        if (!isNum(d)) return NaN;
        if (d > 0) g += d;
        else l -= d;
      }
      return g + l === 0 ? 50 : (100 * g) / (g + l);
    });
    return { rsi: r, signal: ewmSpan(r, p.rsiSignal) };
  }
  const a = 1 / p.rsiPeriod;
  const gain = ewm(delta.map((d) => (isNum(d) ? Math.max(d, 0) : NaN)), a, 1);
  const loss = ewm(delta.map((d) => (isNum(d) ? Math.max(-d, 0) : NaN)), a, 1);
  const r = gain.map((g, i) => {
    const l = loss[i];
    if (l === 0) return 100;
    if (!isNum(l) || !isNum(g)) return NaN;
    return 100 - 100 / (1 + g / l);
  });
  return { rsi: r, signal: rollingMean(r, p.rsiSignal) };
}

export function macd(close: Series, p = BOT_PARAMS): { macd: Series; signal: Series; hist: Series } {
  const fast = ewmSpan(close, p.macdFast);
  const slow = ewmSpan(close, p.macdSlow);
  const m = fast.map((f, i) => f - slow[i]);
  const s = ewmSpan(m, p.macdSignal);
  return { macd: m, signal: s, hist: m.map((v, i) => v - s[i]) };
}

export function adx(bars: Bar[], p = BOT_PARAMS): { adx: Series; diPlus: Series; diMinus: Series } {
  const n = bars.length;
  const tr: Series = new Array(n);
  const pdm: Series = new Array(n);
  const mdm: Series = new Array(n);
  for (let i = 0; i < n; i++) {
    const b = bars[i];
    if (!i) {
      tr[i] = b.h - b.l; // max(axis=1) skips the NaN shift terms
      pdm[i] = 0;
      mdm[i] = 0;
      continue;
    }
    const pc = bars[i - 1].c;
    tr[i] = Math.max(b.h - b.l, Math.abs(b.h - pc), Math.abs(b.l - pc));
    const hd = b.h - bars[i - 1].h;
    const ld = bars[i - 1].l - b.l;
    pdm[i] = hd > ld && hd > 0 ? hd : 0;
    mdm[i] = ld > hd && ld > 0 ? ld : 0;
  }
  const a = p.adxMethod === 'mts' ? 2 / (p.adxPeriod + 1) : 1 / p.adxPeriod;
  const etr = ewm(tr, a, 1);
  const ep = ewm(pdm, a, 1);
  const em = ewm(mdm, a, 1);
  const diPlus = ep.map((v, i) => (etr[i] === 0 ? NaN : (100 * v) / etr[i]));
  const diMinus = em.map((v, i) => (etr[i] === 0 ? NaN : (100 * v) / etr[i]));
  const dx = diPlus.map((v, i) => {
    const s = v + diMinus[i];
    return !isNum(s) || s === 0 ? NaN : (100 * Math.abs(v - diMinus[i])) / s;
  });
  return { adx: ewm(dx, a, 1), diPlus, diMinus };
}

export interface Ichimoku {
  tenkan: Series;
  kijun: Series;
  /** length = bars + shift: index i is drawn at bar i (i >= bars -> future slots) */
  senkouA: Series;
  senkouB: Series;
  /** chikou[i] = close of bar i + shift (drawn at bar i) */
  chikou: Series;
}

export function ichimoku(bars: Bar[], p = BOT_PARAMS): Ichimoku {
  const hi = bars.map((b) => b.h);
  const lo = bars.map((b) => b.l);
  const mid = (w: number) => {
    const h = rollingExtreme(hi, w, true, w);
    const l = rollingExtreme(lo, w, false, w);
    return h.map((v, i) => (v + l[i]) / 2);
  };
  const tenkan = mid(p.ichiTenkan);
  const kijun = mid(p.ichiKijun);
  const b52 = mid(p.ichiSenkouB);
  const s = p.ichiShift;
  const total = bars.length + s;
  const senkouA: Series = new Array(total).fill(NaN);
  const senkouB: Series = new Array(total).fill(NaN);
  for (let i = 0; i < bars.length; i++) {
    senkouA[i + s] = (tenkan[i] + kijun[i]) / 2;
    senkouB[i + s] = b52[i];
  }
  const chikou = bars.map((_, i) => (i + s < bars.length ? bars[i + s].c : NaN));
  return { tenkan, kijun, senkouA, senkouB, chikou };
}

// ---- more bot indicators (r002 calculate_indicators) ----

function rollingSum(x: Series, window: number): Series {
  return x.map((_, i) => {
    let s = 0;
    let n = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) if (isNum(x[j])) (s += x[j]), n++;
    return n ? s : NaN;
  });
}

/** fast stochastic K(10) / D(5): rolling min/max with min_periods=1, D = mean of K */
export function stochastic(bars: Bar[], p = BOT_PARAMS): { k: Series; d: Series } {
  const hh = rollingExtreme(bars.map((b) => b.h), p.stochK, true, 1);
  const ll = rollingExtreme(bars.map((b) => b.l), p.stochK, false, 1);
  const k = bars.map((b, i) => (hh[i] - ll[i] === 0 ? NaN : (100 * (b.c - ll[i])) / (hh[i] - ll[i])));
  return { k, d: rollingMean(k, p.stochD) };
}

/** Williams %R(10) / %D(9) */
export function williams(bars: Bar[], p = BOT_PARAMS): { r: Series; d: Series } {
  const hh = rollingExtreme(bars.map((b) => b.h), p.williamsR, true, 1);
  const ll = rollingExtreme(bars.map((b) => b.l), p.williamsR, false, 1);
  const r = bars.map((b, i) => (hh[i] - ll[i] === 0 ? NaN : (-100 * (hh[i] - b.c)) / (hh[i] - ll[i])));
  return { r, d: rollingMean(r, p.williamsD) };
}

/** MFI(14) with the bot's 50 / 100 / 0 overrides for empty flows */
export function mfi(bars: Bar[], p = BOT_PARAMS): Series {
  const tp = bars.map((b) => (b.h + b.l + b.c) / 3);
  const pos = bars.map((b, i) => (i && tp[i] - tp[i - 1] > 0 ? tp[i] * b.v : 0));
  const neg = bars.map((b, i) => (i && tp[i] - tp[i - 1] < 0 ? tp[i] * b.v : 0));
  const ps = rollingSum(pos, p.mfiPeriod);
  const ns = rollingSum(neg, p.mfiPeriod);
  return ps.map((a, i) => {
    const b = ns[i];
    if (a === 0 && b === 0) return 50;
    if (b === 0 && a > 0) return 100;
    if (a === 0 && b > 0) return 0;
    return 100 - 100 / (1 + a / b);
  });
}

const KST_MS = 9 * 3600000;
const kstDay = (t: number) => Math.floor((t + KST_MS) / 86400000);
/** session start: first bar or the first bar of a new KST day (intraday bars spanning several days) */
const newSession = (bars: Bar[], i: number, intraday: boolean) => i === 0 || (intraday && kstDay(bars[i].t) !== kstDay(bars[i - 1].t));

/** OBV (cumulative per session, like the bot's day-only OBV) + OBV_MA(10) */
export function obv(bars: Bar[], p = BOT_PARAMS, intraday = false): { obv: Series; ma: Series } {
  let acc = 0;
  const o = bars.map((b, i) => {
    if (newSession(bars, i, intraday)) acc = 0;
    else acc += b.c > bars[i - 1].c ? b.v : b.c < bars[i - 1].c ? -b.v : 0;
    return acc;
  });
  return { obv: o, ma: rollingMean(o, p.obvMa) };
}

/** VWAP of the bot: cumsum(close * volume) / cumsum(volume), restarting each session (KST day) for intraday bars */
export function vwap(bars: Bar[], intraday = true): Series {
  let pv = 0;
  let v = 0;
  return bars.map((b, i) => {
    if (newSession(bars, i, intraday)) (pv = 0), (v = 0);
    pv += b.c * b.v;
    v += b.v;
    return v === 0 ? NaN : pv / v;
  });
}

/** simple moving average with a full window (chart MA lines, like an HTS) */
export function sma(close: Series, period: number): Series {
  return rollingMean(close, period, period);
}

// ---- more HTS indicators (HtsH) ----

/** Envelope: SMA(n) +- pct % */
export function envelope(close: Series, p = HTS_PARAMS): { mid: Series; upper: Series; lower: Series } {
  const mid = sma(close, p.envPeriod);
  return { mid, upper: mid.map((m) => m * (1 + p.envPct / 100)), lower: mid.map((m) => m * (1 - p.envPct / 100)) };
}

/** price channel (Donchian): highest high / lowest low of the previous n bars (a close beyond it is a breakout) */
export function donchian(bars: Bar[], p = HTS_PARAMS): { upper: Series; lower: Series } {
  const hh = rollingExtreme(bars.map((b) => b.h), p.donchian, true, p.donchian);
  const ll = rollingExtreme(bars.map((b) => b.l), p.donchian, false, p.donchian);
  return { upper: bars.map((_, i) => (i ? hh[i - 1] : NaN)), lower: bars.map((_, i) => (i ? ll[i - 1] : NaN)) };
}

/** Parabolic SAR (Wilder): value per bar + trend (1 = up, SAR below the price; -1 = down; 0 = none) */
export function parabolicSar(bars: Bar[], p = HTS_PARAMS): { sar: Series; trend: number[] } {
  const n = bars.length;
  const sar: Series = new Array(n).fill(NaN);
  const trend: number[] = new Array(n).fill(0);
  if (n < 2) return { sar, trend };
  let up = bars[1].c >= bars[0].c;
  let af = p.sarStep;
  let ep = up ? Math.max(bars[0].h, bars[1].h) : Math.min(bars[0].l, bars[1].l);
  let s = up ? Math.min(bars[0].l, bars[1].l) : Math.max(bars[0].h, bars[1].h);
  sar[1] = s;
  trend[1] = up ? 1 : -1;
  for (let i = 2; i < n; i++) {
    const b = bars[i];
    s = s + af * (ep - s);
    // the SAR may not move into the previous two bars
    s = up ? Math.min(s, bars[i - 1].l, bars[i - 2].l) : Math.max(s, bars[i - 1].h, bars[i - 2].h);
    if (up && b.l < s) {
      up = false;
      s = ep;
      ep = b.l;
      af = p.sarStep;
    } else if (!up && b.h > s) {
      up = true;
      s = ep;
      ep = b.h;
      af = p.sarStep;
    } else if (up ? b.h > ep : b.l < ep) {
      ep = up ? b.h : b.l;
      af = Math.min(p.sarMax, af + p.sarStep);
    }
    sar[i] = s;
    trend[i] = up ? 1 : -1;
  }
  return { sar, trend };
}

/** CCI(n): (TP - SMA(TP)) / (0.015 * mean deviation) */
export function cci(bars: Bar[], p = HTS_PARAMS): Series {
  const tp = bars.map((b) => (b.h + b.l + b.c) / 3);
  const m = sma(tp, p.cciPeriod);
  return tp.map((v, i) => {
    if (!isNum(m[i])) return NaN;
    let md = 0;
    for (let j = i - p.cciPeriod + 1; j <= i; j++) md += Math.abs(tp[j] - m[i]);
    md /= p.cciPeriod;
    return md === 0 ? 0 : (v - m[i]) / (0.015 * md);
  });
}

/** ATR(n): Wilder smoothing of the true range */
export function atr(bars: Bar[], p = HTS_PARAMS): Series {
  const tr = bars.map((b, i) => (i ? Math.max(b.h - b.l, Math.abs(b.h - bars[i - 1].c), Math.abs(b.l - bars[i - 1].c)) : b.h - b.l));
  return ewm(tr, 1 / p.atrPeriod, p.atrPeriod);
}

/** rate of change (%) over n bars + SMA signal */
export function roc(close: Series, p = HTS_PARAMS): { roc: Series; signal: Series } {
  const r = close.map((c, i) => (i >= p.rocPeriod && close[i - p.rocPeriod] ? (c / close[i - p.rocPeriod] - 1) * 100 : NaN));
  return { roc: r, signal: rollingMean(r, p.rocSignal, p.rocSignal) };
}

/** TRIX: 1-bar % change of the triple EMA(n) + SMA signal */
export function trix(close: Series, p = HTS_PARAMS): { trix: Series; signal: Series } {
  const e3 = ewmSpan(ewmSpan(ewmSpan(close, p.trixPeriod), p.trixPeriod), p.trixPeriod);
  const t = e3.map((v, i) => (i && e3[i - 1] ? (v / e3[i - 1] - 1) * 100 : NaN));
  return { trix: t, signal: rollingMean(t, p.trixSignal, p.trixSignal) };
}

/** disparity (이격도): close / SMA(n) * 100 */
export function disparity(close: Series, p = HTS_PARAMS): Series {
  const m = sma(close, p.disparity);
  return close.map((c, i) => (isNum(m[i]) && m[i] ? (c / m[i]) * 100 : NaN));
}

/** psychological line (심리도): share of up closes over the last n bars, % */
export function psychological(close: Series, p = HTS_PARAMS): Series {
  const n = p.psyPeriod;
  return close.map((_, i) => {
    if (i < n) return NaN;
    let up = 0;
    for (let j = i - n + 1; j <= i; j++) if (close[j] > close[j - 1]) up++;
    return (up / n) * 100;
  });
}

export type IndicatorKey =
  | 'ma'
  | 'bb'
  | 'env'
  | 'donchian'
  | 'sar'
  | 'ichimoku'
  | 'vwap'
  | 'rsi'
  | 'macd'
  | 'adx'
  | 'stoch'
  | 'williams'
  | 'mfi'
  | 'obv'
  | 'cci'
  | 'atr'
  | 'roc'
  | 'trix'
  | 'disparity'
  | 'psy'
  | 'investor';
/** drawn on the price pane */
export const OVERLAYS: IndicatorKey[] = ['ma', 'bb', 'env', 'donchian', 'sar', 'ichimoku', 'vwap'];
/** one pane each below the price pane */
export const OSCILLATORS: IndicatorKey[] = ['rsi', 'macd', 'adx', 'stoch', 'williams', 'mfi', 'obv', 'cci', 'atr', 'roc', 'trix', 'disparity', 'psy', 'investor'];
export const INDICATOR_LABEL: Record<IndicatorKey, string> = {
  ma: '이동평균',
  bb: '볼린저',
  env: '엔벨로프',
  donchian: '가격채널',
  sar: 'Parabolic SAR',
  ichimoku: '일목',
  vwap: 'VWAP',
  rsi: 'RSI',
  macd: 'MACD',
  adx: 'ADX/DMI',
  stoch: '스토캐스틱',
  williams: 'Williams %R',
  mfi: 'MFI',
  obv: 'OBV',
  cci: 'CCI',
  atr: 'ATR',
  roc: 'ROC',
  trix: 'TRIX',
  disparity: '이격도',
  psy: '심리도',
  investor: '투자자 누적',
};
/** moving-average lines of the 'ma' overlay */
export const MA_PERIODS = [5, 20, 60, 120];

// ---- [2026-11-20] investor net buying, cumulative ----

/** cumulative window of the 투자자 누적 pane, months */
export const INV_MONTHS = [1, 3, 6, 12] as const;
export type InvMonths = (typeof INV_MONTHS)[number];
export const INV_MONTHS_LABEL: Record<InvMonths, string> = { 1: '1개월', 3: '3개월', 6: '6개월', 12: '1년' };

export interface InvestorDayRow {
  date: string;
  person: number;
  foreign: number;
  institution: number;
}
export interface InvestorCum {
  person: Series;
  foreign: Series;
  institution: Series;
  /** YYYYMMDD: rows after this date are summed */
  start: string;
}

/** YYYYMMDD `months` calendar months before `d` (day clamped to the month's length) */
export function monthsBefore(d: string, months: number): string {
  let y = +d.slice(0, 4);
  let m = +d.slice(4, 6) - months;
  while (m < 1) (m += 12), y--;
  const last = new Date(Date.UTC(y, m, 0)).getUTCDate();
  return `${y}${String(m).padStart(2, '0')}${String(Math.min(+d.slice(6, 8), last)).padStart(2, '0')}`;
}

/**
 * Net buying of each investor group summed over the rows after `months` before the newest row, value at bar i = the sum up to the
 * last row inside the bar (rows dated before the next bar's date). Bars before the window (or with no row yet) are NaN.
 */
export function investorCum(bars: Bar[], rows: InvestorDayRow[], months: number): InvestorCum {
  const n = bars.length;
  const out: InvestorCum = { person: new Array(n).fill(NaN), foreign: new Array(n).fill(NaN), institution: new Array(n).fill(NaN), start: '' };
  if (!rows.length || !n) return out;
  const sorted = [...rows].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  out.start = monthsBefore(sorted[sorted.length - 1].date, months);
  const win = sorted.filter((r) => r.date > out.start);
  const day = (t: number) => new Date(t + 9 * 3600000).toISOString().slice(0, 10).replace(/-/g, '');
  let j = 0;
  let p = 0;
  let f = 0;
  let o = 0;
  for (let i = 0; i < n; i++) {
    const end = i + 1 < n ? day(bars[i + 1].t) : '99999999';
    while (j < win.length && win[j].date < end) (p += win[j].person), (f += win[j].foreign), (o += win[j].institution), j++;
    if (j === 0) continue;
    out.person[i] = p;
    out.foreign[i] = f;
    out.institution[i] = o;
  }
  return out;
}

export interface IndicatorSet {
  ma?: Record<number, Series>;
  bb?: Bollinger;
  ichimoku?: Ichimoku;
  vwap?: Series;
  rsi?: ReturnType<typeof rsi>;
  macd?: ReturnType<typeof macd>;
  adx?: ReturnType<typeof adx>;
  stoch?: ReturnType<typeof stochastic>;
  williams?: ReturnType<typeof williams>;
  mfi?: Series;
  obv?: ReturnType<typeof obv>;
  env?: ReturnType<typeof envelope>;
  donchian?: ReturnType<typeof donchian>;
  sar?: ReturnType<typeof parabolicSar>;
  cci?: Series;
  atr?: Series;
  roc?: ReturnType<typeof roc>;
  trix?: ReturnType<typeof trix>;
  disparity?: Series;
  psy?: Series;
  investor?: InvestorCum;
}

export interface WarmupBars {
  /** previous-day bars in the bot's warm-up window (PRIOR_DAY_WARMUP_MINUTES) - used for the bot's own indicators */
  warmBot?: Bar[];
  /** longer previous-day history - used for MA lines and Ichimoku (not bot decisions) */
  warm?: Bar[];
}

/**
 * Indicators of `on`, every series aligned to `bars` (index i = bars[i]; Ichimoku senkou spans have `ichiShift` extra slots).
 * Like the bot's calculate_indicators_with_warmup(): BB/RSI/MACD/ADX/stochastic/Williams/MFI run over warmBot + bars and keep the
 * day's rows; VWAP and OBV use the day's bars only. MA lines and Ichimoku use the longer `warm`.
 */
export function computeIndicators(bars: Bar[], on: IndicatorKey[], w: WarmupBars = {}, p = BOT_PARAMS, intraday = true): IndicatorSet {
  const out: IndicatorSet = {};
  const has = (k: IndicatorKey) => on.includes(k);
  const wb = w.warmBot || [];
  const wl = w.warm || [];
  const bot = wb.length ? [...wb, ...bars] : bars;
  const ext = wl.length ? [...wl, ...bars] : bars;
  const cutBot = (s: Series) => (wb.length ? s.slice(wb.length) : s);
  const cutExt = (s: Series) => (wl.length ? s.slice(wl.length) : s);
  const botClose = bot.map((b) => b.c);
  if (has('ma')) {
    const c = ext.map((b) => b.c);
    out.ma = Object.fromEntries(MA_PERIODS.map((n) => [n, cutExt(sma(c, n))]));
  }
  if (has('bb')) {
    const b = bollinger(botClose, p);
    out.bb = { mid: cutBot(b.mid), upper: cutBot(b.upper), lower: cutBot(b.lower) };
  }
  if (has('ichimoku')) {
    const k = ichimoku(ext, p);
    out.ichimoku = { tenkan: cutExt(k.tenkan), kijun: cutExt(k.kijun), senkouA: cutExt(k.senkouA), senkouB: cutExt(k.senkouB), chikou: cutExt(k.chikou) };
  }
  if (has('vwap')) out.vwap = vwap(bars, intraday);
  if (has('rsi')) {
    const r = rsi(botClose, p);
    out.rsi = { rsi: cutBot(r.rsi), signal: cutBot(r.signal) };
  }
  if (has('macd')) {
    const m = macd(botClose, p);
    out.macd = { macd: cutBot(m.macd), signal: cutBot(m.signal), hist: cutBot(m.hist) };
  }
  if (has('adx')) {
    const a = adx(bot, p);
    out.adx = { adx: cutBot(a.adx), diPlus: cutBot(a.diPlus), diMinus: cutBot(a.diMinus) };
  }
  if (has('stoch')) {
    const s = stochastic(bot, p);
    out.stoch = { k: cutBot(s.k), d: cutBot(s.d) };
  }
  if (has('williams')) {
    const s = williams(bot, p);
    out.williams = { r: cutBot(s.r), d: cutBot(s.d) };
  }
  if (has('mfi')) out.mfi = cutBot(mfi(bot, p));
  if (has('obv')) out.obv = obv(bars, p, intraday);
  // HtsH indicators: over the longer history, like the MA lines
  const extClose = ext.map((b) => b.c);
  if (has('env')) {
    const e = envelope(extClose, p);
    out.env = { mid: cutExt(e.mid), upper: cutExt(e.upper), lower: cutExt(e.lower) };
  }
  if (has('donchian')) {
    const d = donchian(ext, p);
    out.donchian = { upper: cutExt(d.upper), lower: cutExt(d.lower) };
  }
  if (has('sar')) {
    const r = parabolicSar(ext, p);
    out.sar = { sar: cutExt(r.sar), trend: wl.length ? r.trend.slice(wl.length) : r.trend };
  }
  if (has('cci')) out.cci = cutExt(cci(ext, p));
  if (has('atr')) out.atr = cutExt(atr(ext, p));
  if (has('roc')) {
    const r = roc(extClose, p);
    out.roc = { roc: cutExt(r.roc), signal: cutExt(r.signal) };
  }
  if (has('trix')) {
    const r = trix(extClose, p);
    out.trix = { trix: cutExt(r.trix), signal: cutExt(r.signal) };
  }
  if (has('disparity')) out.disparity = cutExt(disparity(extClose, p));
  if (has('psy')) out.psy = cutExt(psychological(extClose, p));
  return out;
}

// ---- buy / sell signals of the indicator panes ----

export interface Signal {
  /** bar index */
  i: number;
  side: 'buy' | 'sell';
}

const level = (b: Series | number, i: number) => (typeof b === 'number' ? b : b[i]);
/** a crosses above b at i */
const crossUp = (a: Series, b: Series | number, i: number) => {
  const [a0, a1, b0, b1] = [a[i - 1], a[i], level(b, i - 1), level(b, i)];
  return isNum(a0) && isNum(a1) && isNum(b0) && isNum(b1) && a0 <= b0 && a1 > b1;
};
/** a crosses below b at i */
const crossDn = (a: Series, b: Series | number, i: number) => {
  const [a0, a1, b0, b1] = [a[i - 1], a[i], level(b, i - 1), level(b, i)];
  return isNum(a0) && isNum(a1) && isNum(b0) && isNum(b1) && a0 >= b0 && a1 < b1;
};

/** the series a pane's signal marks are drawn on */
export function signalLine(key: IndicatorKey, S: IndicatorSet): Series | undefined {
  switch (key) {
    case 'rsi': return S.rsi?.rsi;
    case 'macd': return S.macd?.macd;
    case 'adx': return S.adx?.diPlus;
    case 'stoch': return S.stoch?.k;
    case 'williams': return S.williams?.r;
    case 'mfi': return S.mfi;
    case 'obv': return S.obv?.obv;
    case 'cci': return S.cci;
    case 'roc': return S.roc?.roc;
    case 'trix': return S.trix?.trix;
    case 'disparity': return S.disparity;
    case 'psy': return S.psy;
    default: return undefined;
  }
}

/**
 * Buy / sell points of a pane by the usual HTS rules:
 * leaving the oversold / overbought zone - MFI 20 / 80, Williams %R -80 / -20, CCI -100 / +100, psychological line 25 / 75;
 * crossing the signal line - RSI (like the MTS), MACD, ROC, TRIX, OBV (its MA); ADX/DMI: DI+ crossing DI-; stochastic: %K crossing %D from below 20 / above 80;
 * disparity: crossing 100. ATR has none.
 */
export function signals(key: IndicatorKey, S: IndicatorSet): Signal[] {
  const out: Signal[] = [];
  const zone = (s: Series | undefined, lo: number, hi: number) => {
    if (!s) return;
    for (let i = 1; i < s.length; i++) {
      if (crossUp(s, lo, i)) out.push({ i, side: 'buy' });
      else if (crossDn(s, hi, i)) out.push({ i, side: 'sell' });
    }
  };
  const cross = (a: Series | undefined, b: Series | number | undefined, keep: (i: number, side: Signal['side']) => boolean = () => true) => {
    if (!a || b === undefined) return;
    for (let i = 1; i < a.length; i++) {
      if (crossUp(a, b, i)) keep(i, 'buy') && out.push({ i, side: 'buy' });
      else if (crossDn(a, b, i)) keep(i, 'sell') && out.push({ i, side: 'sell' });
    }
  };
  switch (key) {
    case 'rsi': cross(S.rsi?.rsi, S.rsi?.signal); break;
    case 'mfi': zone(S.mfi, 20, 80); break;
    case 'williams': zone(S.williams?.r, -80, -20); break;
    case 'cci': zone(S.cci, -100, 100); break;
    case 'psy': zone(S.psy, 25, 75); break;
    case 'macd': cross(S.macd?.macd, S.macd?.signal); break;
    case 'roc': cross(S.roc?.roc, S.roc?.signal); break;
    case 'trix': cross(S.trix?.trix, S.trix?.signal); break;
    case 'obv': cross(S.obv?.obv, S.obv?.ma); break;
    case 'adx': cross(S.adx?.diPlus, S.adx?.diMinus); break;
    case 'disparity': cross(S.disparity, 100); break;
    case 'stoch': {
      const k = S.stoch?.k;
      if (k) cross(k, S.stoch?.d, (i, side) => (side === 'buy' ? k[i - 1] < 20 : k[i - 1] > 80));
      break;
    }
  }
  return out;
}
