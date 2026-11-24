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
 *   condition A fails (003010 혜인 2026-11-13: -17 % in 12 minutes, detected at 11:38, went on falling).
 * - [2026-11-23] type=feat - plunge.gapPct (user 2026-11-23: 008770 호텔신라 opened 2.4 % under the previous close at 08:00 and
 *   rebounded, nothing was detected - an overnight gap is no single plunge bar): when no plunge bar is found and the day's first
 *   bar opened gapPct or more under the previous bar's close and is still within plunge.window, the gap counts as the plunge -
 *   its bar is the lowest low since the open (findGapPlunge). No volume test (the opening bar is always heavy) and the session
 *   edge does not exclude it; the crash limit (maxFallPct) and the edge rule for the judged bar still apply. 0 = off
 * - [2026-11-23] type=feat - edge.preOpenMin / preCloseMin (user 2026-11-23): the NXT pre-market window (08:00 - 08:50) has its own,
 *   shorter edges (5 / 5 minutes) - with 15 / 15 only 08:15 - 08:35 of its 50 minutes was judged. KRX and the after-market keep
 *   edge.openMin / closeMin
 * - [2026-11-22] type=feat - alert.kakao (send each alert's summary to the user's own KakaoTalk, off by default), alertSummary()
 *   (the text of that message, at most 200 characters), alert.autoCloseSec default 0 -> 60: a popup that closes by itself leaves
 *   the alert unread and the title bar counts it Default 8 %: of 36 sampled alerts the next deepest falls were -6.8 / -5.4 / -5.2 %. 0 = no limit
 * - [2026-11-24] type=feat - Lower highs (downtrend) check (user 2026-11-24, Codex review "수정 후 채택"): lowerHighs() - the last two
 *   swing highs before the plunge bar (H1, H2) and the high of the rebound since it (Hc). H1 > H2 > Hc, each step at least
 *   lowerHigh.minDropPct % and minTicks KRX ticks lower = 'lh' (the rebound is still under a falling sequence of highs).
 *   Swing high = a run of equal highs (the last bar of the run stands for it) higher than pivotBars bars on each side, all of them in
 *   the judged bar's own KST day and session window, centred within rangeBars bars; fewer than two = 'na'. Also: RSI bullish
 *   divergence (the plunge low under the lowest low between H1 and H2 while RSI on the plunge bar is higher). Always computed and
 *   shown (popup / history / CSV); res.lowerHigh.blocked only when lowerHigh.block (default off) - the poller then logs the detection
 *   without an alert. hit itself is unchanged
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
  /** [2026-11-23] preOpenMin / preCloseMin: the same for the NXT pre-market window (sessions.nxtPre) */
  edge: { openMin: number; closeMin: number; preOpenMin: number; preCloseMin: number };
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
    /** [2026-11-23] gap-down open: the day's first bar opening this much (percent, negative) or more under the previous close counts as
     *  the plunge when there is no plunge bar (0 = off) */
    gapPct: number;
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
  /** [2026-11-24] lower highs (downtrend) check - always shown; block = no alert for 'lh' (detection still logged) */
  lowerHigh: {
    block: boolean;
    /** with block: an 'lh' detection with an RSI bullish divergence is still alerted */
    divExempt: boolean;
    /** swing high = higher than this many completed bars on each side */
    pivotBars: number;
    /** swing highs centred within this many bars before the judged bar */
    rangeBars: number;
    /** each step (H1 -> H2, H2 -> Hc) at least this % lower ... */
    minDropPct: number;
    /** ... and at least this many KRX ticks lower */
    minTicks: number;
  };
  /** KST "HH:MM" windows; outside all of them (and on weekends) the watcher rests */
  sessions: { krx: [string, string]; nxtPre: [string, string] | null; nxtAfter: [string, string] | null };
  alert: {
    /** the same symbol is not alerted again within this many minutes */
    cooldownMin: number;
    sound: boolean;
    /** close a popup by itself after this many seconds (0 = stays until closed); closed that way the alert stays "unread" */
    autoCloseSec: number;
    /** send a summary of each alert to the user's own KakaoTalk (needs the Kakao login of the settings dialog) */
    kakao: boolean;
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
  edge: { openMin: 15, closeMin: 15, preOpenMin: 5, preCloseMin: 5 },
  plunge: { window: 40, dropPct: -0.8, wickRatio: 2, wickMinPct: 0.3, volWindow: 60, volMult: 3, fallWindow: 60, maxFallPct: 8, gapPct: -1.5 },
  rsi: { min: 45, max: 68 },
  macd: { oscRising: 3 },
  angle: { lag: 3, k: 10, min: 14 },
  bonus: { minScore: 3, adxTroughBars: 4, adxDropMin: 2, maNearPct: 0.2, macdZeroPct: 0.05, macdZeroBars: 3 },
  lowerHigh: { block: false, divExempt: true, pivotBars: 5, rangeBars: 150, minDropPct: 0.2, minTicks: 2 },
  poll: { delaySec: 2, days: 2, gapMs: 150, tabReloadSec: 300, retries: 3, backoffMs: 500 },
  sessions: { krx: ['09:00', '15:30'], nxtPre: ['08:00', '08:50'], nxtAfter: ['15:40', '20:00'] },
  alert: { cooldownMin: 30, sound: true, autoCloseSec: 60, kakao: false, maxPopups: 6 },
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
  /** [2026-11-23] gap of the day's open vs the previous close, percent - set only when the gap-down open is the plunge */
  plungeGapPct?: number;
}

/** [2026-11-24] lower highs check of a detection */
export interface LowerHigh {
  /** 'lh' H1 > H2 > Hc (downtrend), 'no' not lower, 'na' not judged (fewer than two swing highs, no bar after the plunge) */
  state: 'lh' | 'no' | 'na';
  /** epoch ms / price of the two swing highs (older first) */
  h1t?: number;
  h1?: number;
  h2t?: number;
  h2?: number;
  /** highest high from the bar after the plunge to the judged bar */
  hc?: number;
  /** RSI bullish divergence: plunge low under the lowest low between H1 and H2, RSI higher */
  div: boolean;
  /** lowerHigh.block is on and this one is not alerted ('lh', no divergence exemption) */
  blocked: boolean;
  note: string;
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
  /** [2026-11-24] lower highs check (missing in history items from before it) */
  lowerHigh?: LowerHigh;
  /** why nothing was judged (too few bars) */
  skip?: string;
}

const isNum = (x: number) => x === x && Number.isFinite(x);
const f1 = (x: number) => (isNum(x) ? x.toFixed(1) : '-');
const f2 = (x: number) => (isNum(x) ? x.toFixed(2) : '-');
const KST = 9 * 3600000;
/** epoch ms -> "HH:MM" (KST) */
export const hhmm = (t: number) => (isNum(t) ? new Date(t + KST).toISOString().slice(11, 16) : '-');

const BONUS_SHORT: Record<BonusKey, string> = { adxTurn: 'ADX 반등', maLong: 'MA60/120', bbSqueeze: '볼린저', ichimoku: '일목', macdZero: 'MACD 0선', volDry: '거래량 감소' };

/**
 * Summary of an alert for a text message (KakaoTalk: at most `max` characters). When it gets too long the VWAP, DI and RSI lines
 * are dropped first, the bonus list last. res = null gives the sample shown in the settings dialog.
 */
export function alertSummary(a: { code: string; name: string; price?: number; changePct?: number; res: ReboundResult | null }, max = 200): string {
  const n = (x: number | undefined, d = 1) => (x === undefined || !isNum(x) ? '-' : x.toFixed(d));
  const won = (x: number | undefined) => (x === undefined || !isNum(x) ? '-' : Math.round(x).toLocaleString('ko-KR'));
  const pct = a.changePct === undefined || !isNum(a.changePct) ? '' : ` (${a.changePct >= 0 ? '+' : ''}${a.changePct.toFixed(2)}%)`;
  const s = a.res?.snap;
  const lines = s
    ? [
        `[바닥 반등 초입] ${a.name}(${a.code})`,
        // [2026-11-24] lower highs: a short tag on this line (a line of its own pushed the bonus line past 200 characters)
        `${hhmm(s.t)} 봉 · ${won(a.price ?? s.close)}원${pct} · 가점 ${a.res!.score}점${a.res!.lowerHigh?.state === 'lh' ? ` · 고점↓${a.res!.lowerHigh.div ? '(다이버전스)' : ''}` : ''}`,
        `RSI ${n(s.rsi)}/${n(s.rsiSignal)} · MACD ${n(s.macd)}/${n(s.macdSignal)}`,
        `DI ${n(s.diPlus)}/${n(s.diMinus)} · ADX ${n(s.adx)} · 각도 ${n(s.angle)}°`,
        `VWAP ${isNum(s.vwapPct) && s.vwapPct >= 0 ? '+' : ''}${n(s.vwapPct, 2)}% · 급락 ${hhmm(s.plungeT)} (${n(s.plungeFallPct)}%)`,
        `가점: ${BONUS_KEYS.filter((k) => a.res!.bonus[k].ok).map((k) => BONUS_SHORT[k]).join(', ') || '-'}`,
      ]
    : [`[바닥 반등 초입] ${a.name}(${a.code})`, `14:55 봉 · ${won(a.price)}원${pct} · 가점 3점`, 'RSI 53.3/48.8 · MACD -40.3/-61.7', 'DI 32.2/12.6 · ADX 35.6 · 각도 14.6°', 'VWAP -3.29% · 급락 14:33 (-2.7%)', '가점: ADX 반등, 볼린저, 거래량 감소'];
  // [2026-11-24] (Codex) too long: the indicator lines go first (VWAP, DI, RSI), the bonus list last
  const kept = lines.map((text, k) => ({ text, k }));
  for (const drop of [4, 3, 2, 5]) {
    if (kept.map((x) => x.text).join('\n').length <= max) break;
    const at = kept.findIndex((x) => x.k === drop);
    if (at >= 0) kept.splice(at, 1);
  }
  return kept.map((x) => x.text).join('\n').slice(0, max);
}

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
    const pre = w === cfg.sessions.nxtPre;
    const open = pre ? cfg.edge.preOpenMin : cfg.edge.openMin;
    const close = pre ? cfg.edge.preCloseMin : cfg.edge.closeMin;
    if ((m >= s && m < s + open) || (m <= e && m > e - close)) return true;
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

const kstDayOf = (t: number) => Math.floor((t + KST) / 86400000);

/** KRX tick size of a price */
export function krxTick(p: number): number {
  return p < 2000 ? 1 : p < 5000 ? 5 : p < 20000 ? 10 : p < 50000 ? 50 : p < 200000 ? 100 : p < 500000 ? 500 : 1000;
}

/** the price `n` KRX ticks below `p`, each step with the tick of the price under it (5,000 -> 4,995 -> 4,990) */
export function ticksDown(p: number, n: number): number {
  // (Codex) a whole tick band at a time, so a huge minTicks costs at most 7 steps; below 0 = no price is that far below
  const LOWS = [500000, 200000, 50000, 20000, 5000, 2000, 0];
  let x = p;
  let left = Math.max(0, Math.round(n));
  while (left > 0) {
    if (x <= 0) return -Infinity;
    const t = krxTick(x - 1);
    const low = LOWS.find((l) => x - 1 >= l) ?? 0;
    const room = Math.ceil((x - low) / t); // ticks of size t until the band's low
    if (left <= room) return x - left * t;
    x -= room * t;
    left -= room;
  }
  return x;
}

/** index of the session window (krx 0, nxtPre 1, nxtAfter 2) `t` falls in, -1 outside all */
function sessionIdx(cfg: ReboundConfig, t: number): number {
  const d = new Date(t + KST);
  const m = d.getUTCHours() * 60 + d.getUTCMinutes();
  const min = (s: string) => +s.split(':')[0] * 60 + +s.split(':')[1];
  const ws = [cfg.sessions.krx, cfg.sessions.nxtPre, cfg.sessions.nxtAfter];
  for (let k = 0; k < ws.length; k++) {
    const w = ws[k];
    if (w && m >= min(w[0]) && m <= min(w[1])) return k;
  }
  return -1;
}

const won = (x: number) => Math.round(x).toLocaleString('ko-KR');

export function lowerHighLabel(x: LowerHigh | undefined): string {
  if (!x) return '-';
  return x.state === 'lh' ? '고점 낮아짐 (우하향)' : x.state === 'no' ? '고점 낮아짐 아님' : '고점 판정 불가';
}

/**
 * [2026-11-24] Lower highs before a rebound (see the update log): bars[i] = the judged bar, pj = the plunge bar, rsiS = RSI series.
 */
export function lowerHighs(bars: Bar[], i: number, pj: number, cfg: ReboundConfig, rsiS: Series): LowerHigh {
  const c = cfg.lowerHigh;
  const na = (note: string): LowerHigh => ({ state: 'na', div: false, blocked: false, note });
  if (i < 1 || pj < 0 || pj >= i) return na('급락 뒤 봉 없음');
  const L = Math.max(1, Math.round(c.pivotBars));
  // the judged bar's own day and session window
  const day = kstDayOf(bars[i].t);
  const ses = sessionIdx(cfg, bars[i].t);
  if (ses < 0) return na('운영 시간 밖 봉'); // (Codex 2026-11-24) bars between the windows are not one session
  let s0 = i;
  while (s0 > 0 && kstDayOf(bars[s0 - 1].t) === day && sessionIdx(cfg, bars[s0 - 1].t) === ses) s0--;
  const piv: number[] = [];
  const lo = Math.max(s0, i - Math.max(1, Math.round(c.rangeBars)));
  for (let a = s0 + L; a < pj; a++) {
    // a run of equal highs a..b (the last bar stands for it)
    let b = a;
    while (b + 1 < pj && bars[b + 1].h === bars[a].h) b++;
    const h = bars[a].h;
    let ok = b >= lo && b + L <= i;
    for (let k = a - L; ok && k < a; k++) ok = bars[k].h < h;
    for (let k = b + 1; ok && k <= b + L; k++) ok = bars[k].h < h;
    if (ok) piv.push(b);
    a = b;
  }
  if (piv.length < 2) return na(`급락 전 전고점 ${piv.length}개 (2개 필요)`);
  const k1 = piv[piv.length - 2];
  const k2 = piv[piv.length - 1];
  const h1 = bars[k1].h;
  const h2 = bars[k2].h;
  let hc = -Infinity;
  for (let k = pj + 1; k <= i; k++) hc = Math.max(hc, bars[k].h);
  const lower = (a: number, b: number) => a > b && ((a - b) / a) * 100 >= c.minDropPct && b <= ticksDown(a, c.minTicks);
  const lh = lower(h1, h2) && lower(h2, hc);
  let sl = -1;
  for (let k = k1 + 1; k < k2; k++) if (sl < 0 || bars[k].l < bars[sl].l) sl = k;
  const div = sl >= 0 && bars[pj].l < bars[sl].l && isNum(rsiS[pj]) && isNum(rsiS[sl]) && rsiS[pj] > rsiS[sl];
  const blocked = c.block && lh && !(c.divExempt && div);
  const seq = `${hhmm(bars[k1].t)} ${won(h1)} → ${hhmm(bars[k2].t)} ${won(h2)} → 현재 ${won(hc)}`;
  const dv = div ? `, RSI 상승 다이버전스 (${f1(rsiS[sl])} → ${f1(rsiS[pj])})` : '';
  const tail = blocked ? ' - 알림 안 보냄' : lh && c.block && div && c.divExempt ? ' - 다이버전스라 알림' : '';
  return { state: lh ? 'lh' : 'no', h1t: bars[k1].t, h1, h2t: bars[k2].t, h2, hc, div, blocked, note: `${seq}${dv}${tail}` };
}

/**
 * [2026-11-23] Gap-down open as the plunge: the first bar of the last bar's KST day opened plunge.gapPct or more under the previous
 * bar's close and lies within the last plunge.window bars. The plunge bar is the lowest low since that open (the earliest of equals).
 */
export function findGapPlunge(bars: Bar[], cfg: ReboundConfig): { j: number; open: number; gapPct: number } | null {
  const i = bars.length - 1;
  const p = cfg.plunge;
  if (!(p.gapPct < 0) || i < 1) return null;
  const day = kstDayOf(bars[i].t);
  let open = i;
  while (open > 0 && kstDayOf(bars[open - 1].t) === day) open--;
  if (open < 1 || open < i - p.window + 1) return null;
  const pc = bars[open - 1].c;
  const gapPct = pc > 0 ? (bars[open].o / pc - 1) * 100 : 0;
  if (!(gapPct <= p.gapPct)) return null;
  let j = open;
  for (let k = open + 1; k <= i; k++) if (bars[k].l < bars[j].l) j = k;
  return { j, open, gapPct };
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

  const pBar = i >= 1 ? findPlunge(bars, cfg) : -1;
  const gap = pBar < 0 ? findGapPlunge(bars, cfg) : null;
  const pj = gap ? gap.j : pBar;
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
    ...(gap ? { plungeGapPct: gap.gapPct } : {}),
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
        : gap
          ? { ok: true, note: `갭 하락 시가 ${hhmm(bars[gap.open].t)} ${f1(gap.gapPct)}% (전봉 종가 대비), 저점 ${hhmm(bars[pj].t)} ${bars[pj].l} 낙폭 ${f1(snap.plungeFallPct)}%` }
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
  const lowerHigh = lowerHighs(bars, i, pj, cfg, R.rsi);
  return { hit: required && score >= bz.minScore && !edge, edge, required, score, conds, bonus, snap, lowerHigh };
}
