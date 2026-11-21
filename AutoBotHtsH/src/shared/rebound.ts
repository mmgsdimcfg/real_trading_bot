/*
 * Update log
 * - [2026-11-21] type=feat - "바닥 반등 초입" (bottom-rebound start) detector on 1-minute bars (user 2026-11-21): a plunge bar on heavy
 *   volume whose low holds, then RSI / MACD / DMI / MA5-MA20 crossing up in the last N bars with MA5 rising at an angle, plus a score
 *   of six optional conditions. detectRebound() is a pure function of the completed bars and the config (no API, no GUI) - the
 *   poller (main/reboundPoller.ts), the replay (main/reboundReplay.ts) and the tests call the same function. Indicators come from
 *   shared/indicators.ts, the code the chart draws with, so the values equal the chart's for the same parameters. Detection and
 *   alert only: there is no order code
 * - [2026-11-21] type=feat - Two more rules after a random replay check (user 2026-11-21): (1) requireCurrentAbove - RSI > signal and
 *   DI+ > DI- must also hold on the judged bar itself (a bar that had already turned down still passed "3 of the last 5");
 *   (2) edge - the first edge.openMin and the last edge.closeMin minutes of each session window are left out: a plunge bar there
 *   does not count and a bar there is not detected (opening / closing volume passes the "3x volume" test by itself -
 *   161390 09:13 from a 09:02 bar, 236200 15:42 across the close)
 * - [2026-11-21] type=feat - plunge.maxFallPct / fallWindow (user 2026-11-21): a crash is not a "plunge before a rebound" - when
 *   the fall from the highest high of the fallWindow bars up to the plunge bar down to its low is deeper than maxFallPct,
 *   condition A fails (003010 혜인 2026-11-13: -17 % in 12 minutes, detected at 11:38, went on falling). Default 8 %: of 36 sampled alerts the next deepest falls were -6.8 / -5.4 / -5.2 %. 0 = no limit
 */

import { adx, bollinger, ichimoku, macd, normalizeParams, rsi, sma, vwap, HTS_PARAMS, type CalcMethod, type IndicatorParams, type Series } from './indicators';
import type { Bar } from './price';

export interface ReboundConfig {
  /** master switch of the watcher */
  enabled: boolean;
  /** name of the watchlist tab to watch (case and outer spaces ignored) */
  tab: string;
  /** indicator parameters - the chart's defaults (HTS_PARAMS); change them together with the chart settings */
  indicator: {
    rsiPeriod: number;
    rsiSignal: number;
    /** 'mts' = the chart default (Korean MTS / HTS), 'wilder' = Wilder smoothing + simple-mean signal */
    rsiMethod: CalcMethod;
    macdFast: number;
    macdSlow: number;
    macdSignal: number;
    adxPeriod: number;
    adxMethod: CalcMethod;
    bbPeriod: number;
    bbMult: number;
    ichiTenkan: number;
    ichiKijun: number;
    ichiSenkouB: number;
    ichiShift: number;
  };
  /** completed bars needed before anything is judged (indicator warm-up) */
  minBars: number;
  /** N: the crossings must have happened within the last N bars */
  lookback: number;
  /** M and the count: of the last M bars at least `recentMin` keep RSI > signal / DI+ > DI- */
  recent: number;
  recentMin: number;
  /** RSI > signal and DI+ > DI- must hold on the judged bar too */
  requireCurrentAbove: boolean;
  /** minutes left out at the start / end of each session window (0 = none): no plunge bar and no detection there */
  edge: { openMin: number; closeMin: number };
  plunge: {
    /** the plunge bar is within the last `window` bars */
    window: number;
    /** close vs the previous close, percent (<= this) ... */
    dropPct: number;
    /** ... or lower wick >= wickRatio x body, the wick being at least wickMinPct of the price */
    wickRatio: number;
    wickMinPct: number;
    /** volume >= volMult x the mean volume of the `volWindow` bars before it */
    volWindow: number;
    volMult: number;
    /** the fall into the plunge bar (highest high of the `fallWindow` bars up to it -> its low), percent; deeper than maxFallPct = a crash, not judged (0 = no limit) */
    fallWindow: number;
    maxFallPct: number;
  };
  rsi: { min: number; max: number };
  macd: {
    /** the oscillator (MACD - signal) grew on each of the last `oscRising` bars */
    oscRising: number;
  };
  /** MA5 angle: slope = (MA5 - MA5[lag]) / MA5[lag] * 100 / lag (% per bar), angle = atan(slope * k) in degrees */
  angle: { lag: number; k: number; min: number };
  bonus: {
    /** alert when the score reaches this */
    minScore: number;
    /** ADX: its low of the last N bars is within the last `adxTroughBars` bars, at least adxDropMin below the high before it */
    adxTroughBars: number;
    adxDropMin: number;
    /** close within maNearPct % of MA60 / MA120 */
    maNearPct: number;
    /** |MACD| <= macdZeroPct % of the price, or MACD crossed above 0 within macdZeroBars bars */
    macdZeroPct: number;
    macdZeroBars: number;
  };
  poll: {
    /** seconds after the minute turns (the previous bar is complete) */
    delaySec: number;
    /** trading days of minutes to load (the previous day warms the indicators up) */
    days: number;
    /** pause between two symbols, ms (the KIS client itself keeps the per-second limit) */
    gapMs: number;
    /** re-read the watchlist tab every this many seconds */
    tabReloadSec: number;
    /** tries per symbol and the first back-off (doubled each time) */
    retries: number;
    backoffMs: number;
  };
  /** KST "HH:MM" windows; outside all of them (and on weekends) the watcher rests */
  sessions: { krx: [string, string]; nxtPre: [string, string] | null; nxtAfter: [string, string] | null };
  alert: {
    /** the same symbol is not alerted again within this many minutes */
    cooldownMin: number;
    sound: boolean;
    /** close a popup by itself after this many seconds (0 = stays until closed) */
    autoCloseSec: number;
    /** popups kept on screen (the oldest closes) */
    maxPopups: number;
  };
}

export const REBOUND_DEFAULTS: ReboundConfig = {
  enabled: true,
  tab: 'TEST',
  indicator: {
    rsiPeriod: HTS_PARAMS.rsiPeriod,
    rsiSignal: HTS_PARAMS.rsiSignal,
    rsiMethod: HTS_PARAMS.rsiMethod,
    macdFast: HTS_PARAMS.macdFast,
    macdSlow: HTS_PARAMS.macdSlow,
    macdSignal: HTS_PARAMS.macdSignal,
    adxPeriod: HTS_PARAMS.adxPeriod,
    adxMethod: HTS_PARAMS.adxMethod,
    bbPeriod: HTS_PARAMS.bbPeriod,
    bbMult: HTS_PARAMS.bbMult,
    ichiTenkan: HTS_PARAMS.ichiTenkan,
    ichiKijun: HTS_PARAMS.ichiKijun,
    ichiSenkouB: HTS_PARAMS.ichiSenkouB,
    ichiShift: HTS_PARAMS.ichiShift,
  },
  minBars: 200,
  lookback: 25,
  recent: 5,
  recentMin: 3,
  requireCurrentAbove: true,
  edge: { openMin: 15, closeMin: 15 },
  plunge: { window: 40, dropPct: -0.8, wickRatio: 2, wickMinPct: 0.3, volWindow: 60, volMult: 3, fallWindow: 60, maxFallPct: 8 },
  rsi: { min: 45, max: 68 },
  macd: { oscRising: 3 },
  angle: { lag: 3, k: 10, min: 14 },
  bonus: { minScore: 3, adxTroughBars: 4, adxDropMin: 2, maNearPct: 0.2, macdZeroPct: 0.05, macdZeroBars: 3 },
  poll: { delaySec: 2, days: 2, gapMs: 150, tabReloadSec: 300, retries: 3, backoffMs: 500 },
  sessions: { krx: ['09:00', '15:30'], nxtPre: ['08:00', '08:50'], nxtAfter: ['15:40', '20:00'] },
  alert: { cooldownMin: 30, sound: true, autoCloseSec: 0, maxPopups: 6 },
};

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** a saved config over the defaults: unknown keys are dropped, a value of the wrong type keeps the default */
export function mergeConfig(saved: unknown, base: ReboundConfig = REBOUND_DEFAULTS): ReboundConfig {
  const walk = (def: unknown, got: unknown): unknown => {
    if (isObj(def)) {
      const out: Record<string, unknown> = {};
      for (const k of Object.keys(def)) out[k] = walk(def[k], isObj(got) ? got[k] : undefined);
      return out;
    }
    if (got === undefined) return def;
    if (Array.isArray(def) || def === null) {
      // a session window: ["HH:MM", "HH:MM"] or null (off)
      if (got === null) return null;
      return Array.isArray(got) && got.length === 2 && got.every((x) => typeof x === 'string' && /^\d{1,2}:\d{2}$/.test(x)) ? got : def;
    }
    if (typeof def === 'number') return typeof got === 'number' && Number.isFinite(got) ? got : def;
    if (typeof def === 'boolean') return typeof got === 'boolean' ? got : def;
    if (typeof def === 'string') return typeof got === 'string' && got.trim() ? got : def;
    return def;
  };
  const c = walk(base, saved) as ReboundConfig;
  if (c.indicator.rsiMethod !== 'mts' && c.indicator.rsiMethod !== 'wilder') c.indicator.rsiMethod = base.indicator.rsiMethod;
  if (c.indicator.adxMethod !== 'mts' && c.indicator.adxMethod !== 'wilder') c.indicator.adxMethod = base.indicator.adxMethod;
  return c;
}

export type CondKey = 'plunge' | 'rsi' | 'macd' | 'dmi' | 'ma' | 'angle';
export type BonusKey = 'adxTurn' | 'maLong' | 'bbSqueeze' | 'ichimoku' | 'macdZero' | 'volDry';

export const COND_LABEL: Record<CondKey, string> = {
  plunge: 'A 급락 선행',
  rsi: 'B RSI 상향 돌파',
  macd: 'C MACD 상향 돌파',
  dmi: 'D DI+ 상향 돌파',
  ma: 'E MA5/MA20 골든크로스',
  angle: 'F MA5 각도',
};
export const BONUS_LABEL: Record<BonusKey, string> = {
  adxTurn: 'ADX 바닥 후 반등',
  maLong: 'MA60/120 돌파·근접',
  bbSqueeze: '볼린저 폭 축소 + 중심선 위',
  ichimoku: '전환선>기준선 + 구름 안/위',
  macdZero: 'MACD 0선 근처·돌파',
  volDry: '반등 구간 거래량 감소',
};
export const COND_KEYS = Object.keys(COND_LABEL) as CondKey[];
export const BONUS_KEYS = Object.keys(BONUS_LABEL) as BonusKey[];

export interface Check {
  ok: boolean;
  /** what was measured, for the popup / the replay table */
  note: string;
}

export interface ReboundSnapshot {
  /** epoch ms of the judged (last completed) bar */
  t: number;
  close: number;
  ma5: number;
  ma20: number;
  ma60: number;
  ma120: number;
  rsi: number;
  rsiSignal: number;
  macd: number;
  macdSignal: number;
  macdOsc: number;
  diPlus: number;
  diMinus: number;
  adx: number;
  /** degrees */
  angle: number;
  vwap: number;
  /** close vs VWAP, percent (information only) */
  vwapPct: number;
  bbMid: number;
  bbWidthPct: number;
  /** epoch ms / low / volume of the plunge bar (NaN when there is none) */
  plungeT: number;
  plungeLow: number;
  plungeVol: number;
  /** fall into the plunge bar: highest high of the plunge.fallWindow bars up to it -> its low, percent (negative) */
  plungeFallPct: number;
}

export interface ReboundResult {
  /** every required condition holds, the score reaches bonus.minScore and the bar is not in a session edge */
  hit: boolean;
  /** the judged bar is in the first / last minutes of a session window (edge.*): never a detection */
  edge: boolean;
  /** every required condition holds */
  required: boolean;
  score: number;
  conds: Record<CondKey, Check>;
  bonus: Record<BonusKey, Check>;
  snap: ReboundSnapshot;
  /** why nothing was judged (too few bars) */
  skip?: string;
}

const isNum = (x: number) => x === x && Number.isFinite(x);
const f1 = (x: number) => (isNum(x) ? x.toFixed(1) : '-');
const f2 = (x: number) => (isNum(x) ? x.toFixed(2) : '-');
const KST = 9 * 3600000;
/** epoch ms -> "HH:MM" (KST) */
export const hhmm = (t: number) => (isNum(t) ? new Date(t + KST).toISOString().slice(11, 16) : '-');

/**
 * Which session `now` (epoch ms) falls in: 'krx' (regular hours - every symbol), 'nxt' (NXT pre / after market - NXT symbols only)
 * or null (weekend / outside every window: the watcher rests). Holidays are not known here; on one no new bar arrives, so nothing
 * is judged.
 */
export function sessionOf(cfg: ReboundConfig, now: number): 'krx' | 'nxt' | null {
  const d = new Date(now + KST);
  const dow = d.getUTCDay();
  if (dow === 0 || dow === 6) return null;
  const hm = d.toISOString().slice(11, 16);
  const pad = (s: string) => s.padStart(5, '0');
  const within = (w: [string, string] | null) => !!w && hm >= pad(w[0]) && hm <= pad(w[1]);
  if (within(cfg.sessions.krx)) return 'krx';
  if (within(cfg.sessions.nxtPre) || within(cfg.sessions.nxtAfter)) return 'nxt';
  return null;
}

/** `t` (epoch ms) is within the first edge.openMin or the last edge.closeMin minutes of a session window */
export function inEdge(cfg: ReboundConfig, t: number): boolean {
  const d = new Date(t + KST);
  const m = d.getUTCHours() * 60 + d.getUTCMinutes();
  const min = (s: string) => +s.split(':')[0] * 60 + +s.split(':')[1];
  for (const w of [cfg.sessions.krx, cfg.sessions.nxtPre, cfg.sessions.nxtAfter]) {
    if (!w) continue;
    const s = min(w[0]);
    const e = min(w[1]);
    if ((m >= s && m < s + cfg.edge.openMin) || (m <= e && m > e - cfg.edge.closeMin)) return true;
  }
  return false;
}

/** the bars whose minute is over at `now`: the bar of the running minute (still changing) is left out */
export function completedBars(bars: Bar[], now: number): Bar[] {
  const cut = Math.floor(now / 60000) * 60000;
  let n = bars.length;
  while (n > 0 && bars[n - 1].t >= cut) n--;
  return n === bars.length ? bars : bars.slice(0, n);
}

/** ms from `now` until `delaySec` seconds past the next minute */
export function msToNextPoll(now: number, delaySec: number): number {
  const d = Math.max(0, Math.min(50, delaySec)) * 1000;
  const next = Math.floor(now / 60000) * 60000 + d;
  return (next > now + 500 ? next : next + 60000) - now;
}

/** codes of the watchlist tab named `tab` (case and outer spaces ignored); null when there is no such tab */
export function tabCodes(groups: { name: string; codes: string[] }[], tab: string): string[] | null {
  const want = tab.trim().toLowerCase();
  const g = groups.find((x) => String(x.name || '').trim().toLowerCase() === want);
  return g ? [...new Set(g.codes.filter((c) => /^[0-9A-Z]{6}$/.test(c)))] : null;
}

export function indicatorParams(cfg: ReboundConfig): IndicatorParams {
  return normalizeParams(cfg.indicator, HTS_PARAMS);
}

/** MA angle in degrees: atan of the per-bar % change of the MA over `lag` bars, scaled by k */
export function maAngle(ma: Series, lag: number, k: number): Series {
  return ma.map((v, i) => {
    const p = i >= lag ? ma[i - lag] : NaN;
    if (!isNum(v) || !isNum(p) || p === 0) return NaN;
    return (Math.atan((((v - p) / p) * 100) / lag * k) * 180) / Math.PI;
  });
}

/** last index in [from, to] where a crosses above b (a[k-1] <= b[k-1], a[k] > b[k]); -1 when there is none */
export function lastCrossUp(a: Series, b: Series, from: number, to: number): number {
  for (let k = to; k >= Math.max(1, from); k--) {
    if (isNum(a[k]) && isNum(b[k]) && isNum(a[k - 1]) && isNum(b[k - 1]) && a[k] > b[k] && a[k - 1] <= b[k - 1]) return k;
  }
  return -1;
}

const countAbove = (a: Series, b: Series, from: number, to: number) => {
  let n = 0;
  for (let k = Math.max(0, from); k <= to; k++) if (isNum(a[k]) && isNum(b[k]) && a[k] > b[k]) n++;
  return n;
};

/**
 * The plunge bar of the last `window` bars: a drop (or a long lower wick) on heavy volume whose low was not undercut afterwards.
 * Of several the earliest is taken - its low is the lowest (each later one held above it), so it is the bottom.
 */
export function findPlunge(bars: Bar[], cfg: ReboundConfig): number {
  const i = bars.length - 1;
  const p = cfg.plunge;
  let lowAfter = Infinity; // lowest low of the bars after j
  let found = -1;
  for (let j = i; j >= Math.max(1, i - p.window + 1); j--) {
    const b = bars[j];
    const pc = bars[j - 1].c;
    const drop = pc > 0 ? (b.c / pc - 1) * 100 : 0;
    const body = Math.abs(b.c - b.o);
    const wick = Math.min(b.o, b.c) - b.l;
    const shape = drop <= p.dropPct || (wick > 0 && wick >= p.wickRatio * body && (wick / b.c) * 100 >= p.wickMinPct);
    if (shape && lowAfter >= b.l && !inEdge(cfg, b.t)) {
      let s = 0;
      let n = 0;
      for (let k = Math.max(0, j - p.volWindow); k < j; k++) (s += bars[k].v), n++;
      if (n > 0 && b.v >= p.volMult * (s / n)) found = j;
    }
    lowAfter = Math.min(lowAfter, b.l);
  }
  return found;
}

/** fall into bar j: the highest high of the `window` bars up to and including j -> the low of j, percent (<= 0) */
export function plungeFall(bars: Bar[], j: number, window: number): number {
  let hi = -Infinity;
  for (let k = Math.max(0, j - Math.max(1, window) + 1); k <= j; k++) hi = Math.max(hi, bars[k].h);
  return hi > 0 ? (bars[j].l / hi - 1) * 100 : NaN;
}

/**
 * Judge the last bar of `bars` (all of them completed 1-minute bars, oldest first; earlier days may be included for warm-up -
 * VWAP restarts each KST day). Returns the conditions met, the score and the indicator values at that bar.
 */
export function detectRebound(bars: Bar[], cfg: ReboundConfig = REBOUND_DEFAULTS): ReboundResult {
  const i = bars.length - 1;
  const P = indicatorParams(cfg);
  const close = bars.map((b) => b.c);
  const ma5 = sma(close, 5);
  const ma20 = sma(close, 20);
  const ma60 = sma(close, 60);
  const ma120 = sma(close, 120);
  const R = rsi(close, P);
  const M = macd(close, P);
  const D = adx(bars, P);
  const B = bollinger(close, P);
  const I = ichimoku(bars, P);
  const V = vwap(bars, true);
  const ang = maAngle(ma5, cfg.angle.lag, cfg.angle.k);
  const at = (s: Series) => (i >= 0 ? s[i] : NaN);
  const bbW = (k: number) => (isNum(B.mid[k]) && B.mid[k] !== 0 ? ((B.upper[k] - B.lower[k]) / B.mid[k]) * 100 : NaN);

  const pj = i >= 1 ? findPlunge(bars, cfg) : -1;
  const snap: ReboundSnapshot = {
    t: i >= 0 ? bars[i].t : NaN,
    close: at(close),
    ma5: at(ma5),
    ma20: at(ma20),
    ma60: at(ma60),
    ma120: at(ma120),
    rsi: at(R.rsi),
    rsiSignal: at(R.signal),
    macd: at(M.macd),
    macdSignal: at(M.signal),
    macdOsc: at(M.hist),
    diPlus: at(D.diPlus),
    diMinus: at(D.diMinus),
    adx: at(D.adx),
    angle: at(ang),
    vwap: at(V),
    vwapPct: isNum(at(V)) && at(V) !== 0 ? (at(close) / at(V) - 1) * 100 : NaN,
    bbMid: at(B.mid),
    bbWidthPct: i >= 0 ? bbW(i) : NaN,
    plungeT: pj >= 0 ? bars[pj].t : NaN,
    plungeLow: pj >= 0 ? bars[pj].l : NaN,
    plungeVol: pj >= 0 ? bars[pj].v : NaN,
    plungeFallPct: pj >= 0 ? plungeFall(bars, pj, cfg.plunge.fallWindow) : NaN,
  };
  const no = (note: string): Check => ({ ok: false, note });
  const conds = Object.fromEntries(COND_KEYS.map((k) => [k, no('-')])) as Record<CondKey, Check>;
  const bonus = Object.fromEntries(BONUS_KEYS.map((k) => [k, no('-')])) as Record<BonusKey, Check>;
  const edge = i >= 0 && inEdge(cfg, bars[i].t);
  if (bars.length < cfg.minBars) return { hit: false, edge, required: false, score: 0, conds, bonus, snap, skip: `봉 ${bars.length}개 < ${cfg.minBars}개` };
  const cur = cfg.requireCurrentAbove;

  const N = cfg.lookback;
  // the crossings come after the plunge: searched from the bar after it (or the last N bars, whichever starts later)
  const from = Math.max(i - N + 1, pj >= 0 ? pj + 1 : 0);
  const recFrom = i - cfg.recent + 1;

  // A. plunge
  const crash = pj >= 0 && cfg.plunge.maxFallPct > 0 && -snap.plungeFallPct > cfg.plunge.maxFallPct;
  conds.plunge =
    pj < 0
      ? no(`최근 ${cfg.plunge.window}봉에 없음`)
      : crash
        ? no(`${hhmm(bars[pj].t)} 저가 ${bars[pj].l} - 낙폭 ${f1(snap.plungeFallPct)}% 가 기준 -${cfg.plunge.maxFallPct}% 보다 큼 (급락 과대, 제외)`)
        : { ok: true, note: `${hhmm(bars[pj].t)} 저가 ${bars[pj].l} 거래량 ${bars[pj].v} 낙폭 ${f1(snap.plungeFallPct)}%` };

  // B. RSI crossed its signal, stays above on most of the last M bars, not overbought yet
  {
    const k = pj >= 0 ? lastCrossUp(R.rsi, R.signal, from, i) : -1;
    const n = countAbove(R.rsi, R.signal, recFrom, i);
    const inBand = snap.rsi >= cfg.rsi.min && snap.rsi <= cfg.rsi.max;
    const now = !cur || snap.rsi > snap.rsiSignal;
    conds.rsi = { ok: k >= 0 && n >= cfg.recentMin && inBand && now, note: `돌파 ${k >= 0 ? hhmm(bars[k].t) : '없음'}, 최근 ${cfg.recent}봉 중 ${n}봉 위, RSI ${f1(snap.rsi)}/${f1(snap.rsiSignal)}${now ? '' : ' (현재 봉은 시그널 아래)'}` };
  }

  // C. MACD crossed its signal below the zero line and stayed above since; the oscillator keeps growing
  {
    let k = i; // start of the current run of MACD > signal
    while (k >= 1 && isNum(M.macd[k]) && isNum(M.signal[k]) && M.macd[k] > M.signal[k]) k--;
    const cross = k < i && isNum(M.macd[k]) && isNum(M.signal[k]) && M.macd[k] <= M.signal[k] ? k + 1 : -1;
    const inWin = cross >= from && pj >= 0;
    const below0 = cross >= 0 && M.macd[cross] < 0;
    let grow = 0;
    for (let t = i; t >= 1 && isNum(M.hist[t]) && isNum(M.hist[t - 1]) && M.hist[t] > M.hist[t - 1]; t--) grow++;
    conds.macd = {
      ok: inWin && below0 && grow >= cfg.macd.oscRising,
      note: `돌파 ${cross >= 0 ? `${hhmm(bars[cross].t)}(MACD ${f1(M.macd[cross])})` : '없음'}, 오실레이터 ${grow}봉 연속 확대, MACD ${f1(snap.macd)}/${f1(snap.macdSignal)}`,
    };
  }

  // D. DI+ crossed DI-, stays above on most of the last M bars
  {
    const k = pj >= 0 ? lastCrossUp(D.diPlus, D.diMinus, from, i) : -1;
    const n = countAbove(D.diPlus, D.diMinus, recFrom, i);
    const now = !cur || snap.diPlus > snap.diMinus;
    conds.dmi = { ok: k >= 0 && n >= cfg.recentMin && now, note: `돌파 ${k >= 0 ? hhmm(bars[k].t) : '없음'}, 최근 ${cfg.recent}봉 중 ${n}봉 위, DI ${f1(snap.diPlus)}/${f1(snap.diMinus)}${now ? '' : ' (현재 봉은 DI- 아래)'}` };
  }

  // E. MA5 golden cross over MA20, close > MA5 > MA20
  {
    const k = pj >= 0 ? lastCrossUp(ma5, ma20, from, i) : -1;
    const order = snap.close > snap.ma5 && snap.ma5 > snap.ma20;
    conds.ma = { ok: k >= 0 && order, note: `골든크로스 ${k >= 0 ? hhmm(bars[k].t) : '없음'}, 종가 ${snap.close} / MA5 ${f1(snap.ma5)} / MA20 ${f1(snap.ma20)}` };
  }

  // F. MA5 angle
  conds.angle = { ok: isNum(snap.angle) && snap.angle >= cfg.angle.min, note: `${f1(snap.angle)}° (기준 ${cfg.angle.min}°)` };

  // ---- bonus ----
  const bz = cfg.bonus;
  {
    // ADX came down from a high, bottomed within the last few bars and turned up
    let lo = i;
    for (let k = Math.max(0, i - N + 1); k <= i; k++) if (isNum(D.adx[k]) && !(D.adx[k] > D.adx[lo])) lo = k; // the latest lowest
    let hi = -Infinity;
    for (let k = Math.max(0, i - N + 1); k < lo; k++) if (isNum(D.adx[k])) hi = Math.max(hi, D.adx[k]);
    const turned = lo < i && i - lo <= bz.adxTroughBars && D.adx[i] > D.adx[i - 1];
    bonus.adxTurn = { ok: turned && hi - D.adx[lo] >= bz.adxDropMin, note: `저점 ${hhmm(bars[lo].t)} ${f1(D.adx[lo])} (앞 고점 ${f1(hi)}), 현재 ${f1(snap.adx)}` };
  }
  {
    const near = (ma: number) => isNum(ma) && ma !== 0 && Math.abs(snap.close / ma - 1) * 100 <= bz.maNearPct;
    const crossed = (ma: Series) => lastCrossUp(close, ma, Math.max(1, i - N + 1), i) >= 0 && snap.close > ma[i];
    const ok = near(snap.ma60) || near(snap.ma120) || crossed(ma60) || crossed(ma120);
    bonus.maLong = { ok, note: `종가 ${snap.close} / MA60 ${f1(snap.ma60)} / MA120 ${f1(snap.ma120)}` };
  }
  {
    const w0 = pj >= 0 ? bbW(pj) : NaN;
    bonus.bbSqueeze = { ok: isNum(w0) && snap.bbWidthPct < w0 && snap.close >= snap.bbMid, note: `폭 ${f2(snap.bbWidthPct)}% (급락 때 ${f2(w0)}%), 중심선 ${f1(snap.bbMid)}` };
  }
  {
    const a = I.senkouA[i];
    const b = I.senkouB[i];
    const bottom = Math.min(a, b);
    const ok = isNum(I.tenkan[i]) && isNum(I.kijun[i]) && I.tenkan[i] > I.kijun[i] && isNum(bottom) && snap.close >= bottom;
    bonus.ichimoku = { ok, note: `전환 ${f1(I.tenkan[i])} / 기준 ${f1(I.kijun[i])}, 구름 ${f1(bottom)}~${f1(Math.max(a, b))}` };
  }
  {
    const lim = (snap.close * bz.macdZeroPct) / 100;
    let crossed = false;
    for (let k = Math.max(1, i - bz.macdZeroBars + 1); k <= i; k++) if (M.macd[k] > 0 && M.macd[k - 1] <= 0) crossed = true;
    bonus.macdZero = { ok: Math.abs(snap.macd) <= lim || (crossed && snap.macd > 0), note: `MACD ${f1(snap.macd)} (0선 기준 ±${f1(lim)})${crossed ? ', 0선 돌파' : ''}` };
  }
  {
    let s = 0;
    let n = 0;
    if (pj >= 0) for (let k = pj + 1; k <= i; k++) (s += bars[k].v), n++;
    const avg = n ? s / n : NaN;
    bonus.volDry = { ok: n > 0 && avg < bars[pj].v, note: `반등 구간 평균 ${f1(avg)} / 급락 봉 ${pj >= 0 ? bars[pj].v : '-'}` };
  }

  const required = COND_KEYS.every((k) => conds[k].ok);
  const score = BONUS_KEYS.filter((k) => bonus[k].ok).length;
  return { hit: required && score >= bz.minScore && !edge, edge, required, score, conds, bonus, snap };
}
