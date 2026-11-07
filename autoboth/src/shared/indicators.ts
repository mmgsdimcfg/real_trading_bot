/*
 * Update log
 * - [2026-11-06] type=feat - Chart indicators computed the same way as the bot (live_trading/r002_strategy_core_shared.py
 *   calculate_indicators / data_simulation/shared/indicators.py): Bollinger (rolling mean + sample std ddof=1, min_periods=1),
 *   RSI with Wilder smoothing (ewm alpha=1/N, adjust=False) + simple-mean signal, MACD (ewm span, adjust=False), ADX / DI+ / DI-
 *   (Wilder). ewm() follows pandas' adjust=False, ignore_na=False loop exactly (NaN gaps decay the old weight), so values match
 *   the bot's columns. Default periods = r001 (BB 20/2, RSI 14/6, MACD 5/12/4, ADX 7). Ichimoku (not used by the bot): standard
 *   9/26/52 with full windows, senkou spans shifted 26 bars forward, chikou 26 back. Pure functions (tests/indicators.test.ts)
 * - [2026-11-06] type=feat - Stochastic (10/5), Williams %R (10/9), MFI (14, bot's 50/100/0 overrides), OBV + OBV_MA (10), VWAP
 *   (close-weighted, session) and MA lines 5/20/60/120 for the chart tab. computeIndicators() now takes the warm-up bars and returns
 *   series aligned to the day's bars, mirroring the bot's calculate_indicators_with_warmup(): bot indicators over the previous
 *   day's last PRIOR_DAY_WARMUP_MINUTES (180) one-minute bars + the day, VWAP/OBV over the day only
 * - [2026-11-07] type=feat - crossSignal / SIGNAL_RULES: per-oscillator buy / sell crossings for the chart's ▲ / ▼ marks (RSI 30/70,
 *   MACD-signal, DI+/DI-, stochastic %K/%D in 20/80 zones, Williams -80/-20, MFI 20/80, OBV/MA). Display only
 */

import type { Bar } from './price';

export type Series = number[];

export interface IndicatorParams {
  bbPeriod: number;
  bbMult: number;
  rsiPeriod: number;
  rsiSignal: number;
  macdFast: number;
  macdSlow: number;
  macdSignal: number;
  adxPeriod: number;
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
}

/** r001_define_config.py values (BB_PERIOD, BB_STD_MULTIPLIER, RSI_PERIOD, RSI_SIGNAL_PERIOD, MACD_*, ADX_PERIOD) */
export const BOT_PARAMS: IndicatorParams = {
  bbPeriod: 20,
  bbMult: 2,
  rsiPeriod: 14,
  rsiSignal: 6,
  macdFast: 5,
  macdSlow: 12,
  macdSignal: 4,
  adxPeriod: 7,
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
};

const isNum = (x: number) => x === x; // not NaN

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
  const a = 1 / p.adxPeriod;
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

export type IndicatorKey = 'ma' | 'bb' | 'ichimoku' | 'vwap' | 'rsi' | 'macd' | 'adx' | 'stoch' | 'williams' | 'mfi' | 'obv';
/** drawn on the price pane */
export const OVERLAYS: IndicatorKey[] = ['ma', 'bb', 'ichimoku', 'vwap'];
/** one pane each below the price pane */
export const OSCILLATORS: IndicatorKey[] = ['rsi', 'macd', 'adx', 'stoch', 'williams', 'mfi', 'obv'];
export const INDICATOR_LABEL: Record<IndicatorKey, string> = {
  ma: '이동평균',
  bb: '볼린저',
  ichimoku: '일목',
  vwap: 'VWAP',
  rsi: 'RSI',
  macd: 'MACD',
  adx: 'ADX',
  stoch: '스토캐스틱',
  williams: 'Williams %R',
  mfi: 'MFI',
  obv: 'OBV',
};
/** moving-average lines of the 'ma' overlay */
export const MA_PERIODS = [5, 20, 60, 120];

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
  return out;
}

// ---- crossing signals (HTS style ▲ / ▼ on the indicator panes) ----------------------------------

export type CrossSignal = 'buy' | 'sell';

/** what each pane's ▲ / ▼ means (tooltips / legend) */
export const SIGNAL_RULES: Partial<Record<IndicatorKey, string>> = {
  rsi: '매수: RSI가 30을 상향 돌파 · 매도: 70을 하향 돌파',
  macd: '매수: MACD가 시그널 상향 돌파(골든크로스) · 매도: 하향 돌파(데드크로스)',
  adx: '매수: DI+가 DI-를 상향 돌파 · 매도: 하향 돌파',
  stoch: '매수: 20 아래에서 %K가 %D 상향 돌파 · 매도: 80 위에서 하향 돌파',
  williams: '매수: %R이 -80을 상향 돌파 · 매도: -20을 하향 돌파',
  mfi: '매수: MFI가 20을 상향 돌파 · 매도: 80을 하향 돌파',
  obv: '매수: OBV가 이동평균 상향 돌파 · 매도: 하향 돌파',
};

const fin = (v: number | undefined): v is number => v !== undefined && Number.isFinite(v);

/** a crosses above b between bar i-1 and i (b may be a level) */
function crossUp(a: Series, b: Series | number, i: number): boolean {
  const a0 = a[i - 1];
  const a1 = a[i];
  const b0 = typeof b === 'number' ? b : b[i - 1];
  const b1 = typeof b === 'number' ? b : b[i];
  return fin(a0) && fin(a1) && fin(b0) && fin(b1) && a0 <= b0 && a1 > b1;
}
function crossDown(a: Series, b: Series | number, i: number): boolean {
  const a0 = a[i - 1];
  const a1 = a[i];
  const b0 = typeof b === 'number' ? b : b[i - 1];
  const b1 = typeof b === 'number' ? b : b[i];
  return fin(a0) && fin(a1) && fin(b0) && fin(b1) && a0 >= b0 && a1 < b1;
}

/** Buy / sell crossing of an oscillator at bar i (rules in SIGNAL_RULES), null = none. Display only - not the bot's logic. */
export function crossSignal(key: IndicatorKey, S: IndicatorSet, i: number): CrossSignal | null {
  if (i < 1) return null;
  const pick = (buy: boolean, sell: boolean): CrossSignal | null => (buy ? 'buy' : sell ? 'sell' : null);
  switch (key) {
    case 'rsi':
      return S.rsi ? pick(crossUp(S.rsi.rsi, 30, i), crossDown(S.rsi.rsi, 70, i)) : null;
    case 'macd':
      return S.macd ? pick(crossUp(S.macd.macd, S.macd.signal, i), crossDown(S.macd.macd, S.macd.signal, i)) : null;
    case 'adx':
      return S.adx ? pick(crossUp(S.adx.diPlus, S.adx.diMinus, i), crossDown(S.adx.diPlus, S.adx.diMinus, i)) : null;
    case 'stoch': {
      if (!S.stoch) return null;
      const k0 = S.stoch.k[i - 1];
      return pick(crossUp(S.stoch.k, S.stoch.d, i) && fin(k0) && k0 < 20, crossDown(S.stoch.k, S.stoch.d, i) && fin(k0) && k0 > 80);
    }
    case 'williams':
      return S.williams ? pick(crossUp(S.williams.r, -80, i), crossDown(S.williams.r, -20, i)) : null;
    case 'mfi':
      return S.mfi ? pick(crossUp(S.mfi, 20, i), crossDown(S.mfi, 80, i)) : null;
    case 'obv':
      return S.obv ? pick(crossUp(S.obv.obv, S.obv.ma, i), crossDown(S.obv.obv, S.obv.ma, i)) : null;
    default:
      return null;
  }
}
