/*
// - [2026-11-23] type=feat (props.alerts: bells over the bars where a rebound alert was raised (user 2026-11-23) - drawn above the high
//   of the bar that holds the alert's bar time, tooltip with the alert, a click calls onAlertClick)
// - [2026-11-23] type=feat (Tick / minute charts draw the previous trading day's close as a dotted reference line with its
//    price (user 2026-11-23): the last bar of the day before among the bars, else the last warm-up bar, else props.refClose (the
//    quote's previous close). The price scale reaches it when it is within 15 % of the shown range)
 * Update log
 * - [2026-11-05] type=feat - Candlestick price chart for the timeline tab (custom SVG, one <path> per color so thousands of 1-tick
 *   bars stay cheap). The plot area uses the same margins as the recharts charts around it (left 8 + 150 axis, right 24), so for
 *   tick / minute bars the x axis is the buy-evaluation chart's time domain and candles line up with the evaluation dots above.
 *   Daily / weekly / monthly bars use an index axis ending at the viewed day, whose bar is highlighted. Korean colors (up red,
 *   down blue), volume pane, buy/sell fills (vertical lines intraday, price lines on D/W/M), position price line, hover crosshair
 *   with OHLCV and change vs the previous close
 * - [2026-11-05] type=fix - Tick bars sit at their last trade with the median spacing as width (a pre-market gap stretched one bar)
 * - [2026-11-06] type=feat - Indicators (shared/indicators.ts, same formulas/periods as the bot): Bollinger and Ichimoku on the price
 *   pane, RSI / MACD / ADX panes; computed with the previous day's warm-up bars, drawn for the visible bars only
 * - [2026-11-06] type=feat - Generalized for the HTS-style chart tab: mode 'index' (evenly spaced bars for every bar kind) with wheel
 *   zoom / drag pan (follows the newest bar while live), price axis on the right with a last-price tag and a hover price tag,
 *   per-pane legends with values, more overlays (MA 5/20/60/120, VWAP) and panes (stochastic, Williams %R, MFI, OBV), a separator at
 *   each new KST day of intraday bars, trade marks placed by time on an index axis, onHover for an outside data panel. The
 *   timeline keeps mode 'time' with the left axis
 * - [2026-11-06] type=feat - (HtsH copy) indicator periods as a prop (default HTS_PARAMS), yearly bars, chart style candle / line / OHLC
 *   bars, drawing tools (horizontal line, trend line; click on the price pane), lines drawn in price/time so they survive zoom and pan
 * - [2026-11-06] type=feat - (HtsH) legends move to the right edge while the cursor is in the left half (they no longer cover the bars
 *   being read); buy / sell signal marks on the panes (red ▲ under the line = buy, blue ▼ over it = sell, rules in indicators.signals);
 *   new overlays Envelope / price channel / Parabolic SAR (dots) and panes CCI / ATR / ROC / TRIX / disparity / psychological line
 * - [2026-11-09] type=fix - ADX/DMI guide line at 20 with the MTS method (like the MTS), 25 with Wilder
 * - [2026-11-19] type=feat - Readability like the AutoBotH chart (user 2026-11-19): candles drawn on top of the context layers,
 *   Bollinger as a faint band with a dashed mid line (no upper / lower curves), Ichimoku cloud as a fill only (no span lines, no
 *   lagging span), MAs as a grey ramp (red / blue stay with the candles); legends always at the top left of each pane with colour
 *   chips and a halo (they no longer jump to the right edge); the hovered (else the last) bar's date, OHLC and change vs the
 *   previous close on the first legend line (was a separate data window). Text in the chart can no longer be selected (a drag
 *   selected and highlighted every label)
 * - [2026-11-20] type=feat - Pane 투자자 누적: cumulative net buying (shares) of 개인 / 외국인 / 기관 over the chosen 1 / 3 / 6 / 12
 *   months (props investor + invMonths), zero line, day / week / month / year bars only
 * - [2026-11-20] type=feat - Pane heights by drag (user 2026-11-20): a handle between the price and volume areas sets the volume share
 *   (volRatio / onVolRatio), one between the volume area and the indicator panes sets the indicator pane height (onOscHeight) - the
 *   price + volume area takes the rest; a double click on a handle goes back to the default. invNote: status text of the 투자자 누적
 *   pane while it has no data (loading / KIS error)
 * - [2026-11-21] type=feat - (user 2026-11-21) (1) each of 시 / 고 / 저 / 종 on the first legend line carries its change vs the
 *   reference close; (2) while hovering, the values of that bar are shown in a small box beside the vertical line in every pane
 *   (price overlays, volume, each indicator pane) - on the side with more room (right of the line in the left half, else left);
 *   the top-left legends then keep their names only; (3) prop `volume` (default on): the volume area can be switched off - the
 *   price pane takes its room, the handle and the volume legend go away
 * - [2026-11-21] type=feat - (user 2026-11-21) Volume is a pane of its own like the indicator panes: 가격 / 거래량 / RSI / MACD ...
 *   are separate boxes. Every pane has its own height (paneHeights / onPaneHeights; default oscHeight): the handle above a pane
 *   moves the border between it and the pane (or the price pane) above, a double click gives the pane its default height back;
 *   the price pane takes what is left (at least 120 px - panes shrink in proportion when the window is small). A small x at the
 *   top right of a pane removes it (onRemovePane). Replaces volRatio / onVolRatio / onOscHeight
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { samePeriod, type Bar, type ChartKind } from '../../../shared/price';
import { HTS_PARAMS, computeIndicators, investorCum, INV_MONTHS_LABEL, type InvestorDayRow, type InvMonths, MA_PERIODS, OSCILLATORS, signalLine, signals as signalsOf, type IndicatorKey, type IndicatorParams, type IndicatorSet, type Series } from '../../../shared/indicators';
import type { Trade } from '../../../shared/types';
import { fmtKstDateTime, fmtKstTime } from '../../../shared/time';
import { fmtInt } from '../lib';

/** must match the ScatterChart margins + YAxis width in TimelineView */
export const PLOT_LEFT = 8 + 150;
export const PLOT_RIGHT = 24;
const TOP = 10;
const AXIS_H = 22;
const GAP = 8;
export const PRICE_PANE_H = 300;
export const OSC_PANE_H = 84;
/** smallest price pane / pane heights */
const MIN_PRICE_H = 120;
const MIN_PANE_H = 28;
const KST = 9 * 3600000;
const MIN_BARS = 10;

/** [2026-11-19] grey ramp (theme tokens): the short average is the brightest; red / blue belong to the candles */
export const MA_COLORS: Record<number, string> = { 5: 'var(--ma-1)', 20: 'var(--ma-2)', 60: 'var(--ma-3)', 120: 'var(--ma-4)' };
/** colours of the price-pane overlays (legend chips) - the lines themselves are styled by the .ind-* classes */
export const OVERLAY_COLOR = { bb: '#9ca3af', vwap: '#a78bfa', tenkan: '#f59e0b', kijun: '#10b981', env: '#2aa1c9', donchian: '#b07cf2' } as const;

/** [2026-11-19] reference close for the change of bar i: intraday = the last bar of the day before, otherwise the previous bar */
export function refCloseAt(bars: Bar[], i: number, intraday: boolean): number | undefined {
  const b = bars[i];
  if (!b) return undefined;
  if (!intraday) return i > 0 ? bars[i - 1].c : undefined;
  const d0 = Math.floor((b.t + KST) / 86400000) * 86400000 - KST;
  for (let k = i - 1; k >= 0; k--) if (bars[k].t < d0) return bars[k].c;
  return undefined;
}

/** [2026-11-19] label of a bar: intraday "YYYY-MM-DD HH:MM" (tick: start ~ end time), D/W/M/Y the date */
function barWhen(b: Bar, kind: ChartKind): string {
  if (kind === 'tick' || kind === 'min') {
    const head = fmtKstDateTime(b.t).slice(0, kind === 'tick' ? 19 : 16);
    return kind === 'tick' && b.t2 && b.t2 !== b.t ? `${head} ~ ${fmtKstTime(b.t2)}` : head;
  }
  if (kind === 'year') return `${fmtKstDateTime(b.t).slice(0, 4)}년`;
  if (kind === 'month') return fmtKstDateTime(b.t).slice(0, 7);
  return fmtKstDateTime(b.t).slice(0, 10) + (kind === 'week' ? ' 주' : '');
}
const tone = (v: number, ref: number | undefined) => (ref === undefined ? '' : v > ref ? 'pc-legend-up' : v < ref ? 'pc-legend-dn' : '');
const signedPct = (v: number, ref: number) => `${v >= ref ? '+' : ''}${(((v - ref) / ref) * 100).toFixed(2)}%`;

interface Props {
  bars: Bar[];
  /** previous-day bars for MA lines / Ichimoku (not drawn) */
  warm?: Bar[];
  /** the bot's warm-up window for its own indicators (not drawn) */
  warmBot?: Bar[];
  kind: ChartKind;
  /** minutes per bar (min) - used for the candle width in time mode */
  n: number;
  indicators: IndicatorKey[];
  trades: Trade[];
  /** open position price (horizontal line) */
  buyPrice?: number;
  /** [2026-11-23] previous trading day's close when the bars do not hold that day (tick / minute charts) */
  refClose?: number;
  /** YYYYMMDD of the viewed day (highlighted on D/W/M) */
  date: string;
  /** 'time': x = time domain (timeline, intraday); 'index': evenly spaced bars. Default: time for tick/min, index otherwise */
  mode?: 'time' | 'index';
  /** time mode: x domain shared with the evaluation charts (null = fit the bars) */
  domain?: [number, number] | null;
  xTicks?: number[];
  /** plot margins and price axis side (timeline: left axis 158 / 24) */
  left?: number;
  right?: number;
  axis?: 'left' | 'right';
  /** total height; default = 300 + one pane per oscillator */
  height?: number;
  oscHeight?: number;
  /** index mode: wheel zoom + drag pan */
  interactive?: boolean;
  /** floating OHLC/indicator tooltip (off when the page shows its own data panel) */
  tooltip?: boolean;
  /** hovered bar index (null = none); left = the cursor is in the left half of the plot */
  onHover?: (i: number | null, left?: boolean) => void;
  /** buy (red ▲) / sell (blue ▼) marks on the indicator panes (default on) */
  signals?: boolean;
  /** indicator periods (default HTS_PARAMS) */
  params?: IndicatorParams;
  /** chart style */
  style?: 'candle' | 'line' | 'ohlc';
  /** horizontal price lines drawn by the user */
  hlines?: number[];
  onAddHline?: (price: number) => void;
  /** [2026-11-23] alert marks: a bell above the bar holding time t (epoch ms) */
  alerts?: { t: number; id: string; title: string }[];
  onAlertClick?: (id: string) => void;
  /** trend lines drawn by the user (bar time / price pairs) */
  tlines?: { t1: number; p1: number; t2: number; p2: number }[];
  onAddTline?: (l: { t1: number; p1: number; t2: number; p2: number }) => void;
  /** drawing tool: 'hline' = click adds a horizontal line, 'tline' = click twice for a trend line */
  tool?: 'none' | 'hline' | 'tline';
  /** daily investor net buying (pane 'investor') and its cumulative window in months */
  investor?: InvestorDayRow[];
  invMonths?: InvMonths;
  /** shown in the 투자자 누적 pane while it has nothing to draw */
  invNote?: string;
  /** heights (px) of the panes under the price pane by key ('volume' or an indicator key); a pane without one gets the default */
  paneHeights?: Record<string, number>;
  /** a handle was dragged (or double clicked = that key removed): the new heights */
  onPaneHeights?: (h: Record<string, number>) => void;
  /** the x of a pane was clicked ('volume' or an indicator key) */
  onRemovePane?: (key: string) => void;
  /** volume pane under the price pane (default on) */
  volume?: boolean;
}

function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(+v.toFixed(6));
  return out;
}

const iso = (t: number) => new Date(t + KST).toISOString();
function fmtDay(t: number, kind: ChartKind, withYear: boolean): string {
  const s = iso(t);
  if (kind === 'month') return s.slice(0, 7);
  if (kind === 'year') return s.slice(0, 4);
  return withYear ? s.slice(2, 10) : s.slice(5, 10);
}

const ok = (v: number | undefined): v is number => v !== undefined && v === v && Number.isFinite(v);

/** polyline with gaps at NaN */
function line(pts: [number, number | undefined][], sy: (v: number) => number): string {
  let d = '';
  let pen = false;
  for (const [x, v] of pts) {
    if (!ok(v)) {
      pen = false;
      continue;
    }
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)} ${sy(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** cloud polygons between two series, split where the leading span changes (up = a >= b) */
function cloud(pts: { x: number; a: number; b: number }[], sy: (v: number) => number): { up: string; dn: string } {
  let up = '';
  let dn = '';
  let run: typeof pts = [];
  const flush = () => {
    if (run.length > 1) {
      const d =
        'M' +
        run.map((p) => `${p.x.toFixed(1)} ${sy(p.a).toFixed(1)}`).join('L') +
        'L' +
        [...run].reverse().map((p) => `${p.x.toFixed(1)} ${sy(p.b).toFixed(1)}`).join('L') +
        'Z';
      if (run[0].a >= run[0].b) up += d;
      else dn += d;
    }
    run = [];
  };
  for (const p of pts) {
    if (!ok(p.a) || !ok(p.b)) {
      flush();
      continue;
    }
    if (run.length && run[run.length - 1].a >= run[run.length - 1].b !== p.a >= p.b) {
      const last = run[run.length - 1];
      flush();
      run = [last];
    }
    run.push(p);
  }
  flush();
  return { up, dn };
}

const oscOf = (indicators: IndicatorKey[]) => OSCILLATORS.filter((k) => indicators.includes(k));

export function chartHeight(indicators: IndicatorKey[], oscH = OSC_PANE_H): number {
  return PRICE_PANE_H + oscOf(indicators).length * (oscH + GAP);
}

interface PaneLine {
  cls?: string;
  color?: string;
  d: string;
}
interface Legend {
  label: string;
  color?: string;
  value?: number;
  digits?: number;
}

/** per-pane config: range, guide lines, lines, legend values at bar i */
function paneOf(key: IndicatorKey, S: IndicatorSet, P: IndicatorParams, invMonths: InvMonths = 3) {
  switch (key) {
    case 'investor':
      return {
        label: `투자자 누적 ${INV_MONTHS_LABEL[invMonths]}`,
        fixed: null,
        guides: [0],
        series: S.investor
          ? [{ s: S.investor.person, cls: 'ind-inv-person', name: '개인' }, { s: S.investor.foreign, cls: 'ind-inv-foreign', name: '외국인' }, { s: S.investor.institution, cls: 'ind-inv-inst', name: '기관' }]
          : [],
      };
    case 'rsi':
      return { label: `RSI(${P.rsiPeriod},${P.rsiSignal})`, fixed: [0, 100] as [number, number], guides: [30, 70], series: S.rsi ? [{ s: S.rsi.rsi, cls: 'ind-main', name: 'RSI' }, { s: S.rsi.signal, cls: 'ind-signal', name: '시그널' }] : [] };
    case 'adx':
      return { label: `ADX/DMI(${P.adxPeriod})`, fixed: null, min: 0, guides: [P.adxMethod === 'wilder' ? 25 : 20], series: S.adx ? [{ s: S.adx.diPlus, cls: 'ind-di-plus', name: 'DI+' }, { s: S.adx.diMinus, cls: 'ind-di-minus', name: 'DI-' }, { s: S.adx.adx, cls: 'ind-main', name: 'ADX' }] : [] };
    case 'macd':
      return { label: `MACD(${P.macdFast},${P.macdSlow},${P.macdSignal})`, fixed: null, symmetric: true, guides: [0], hist: S.macd?.hist, series: S.macd ? [{ s: S.macd.macd, cls: 'ind-main', name: 'MACD' }, { s: S.macd.signal, cls: 'ind-signal', name: '시그널' }] : [] };
    case 'stoch':
      return { label: `Stoch(${P.stochK},${P.stochD})`, fixed: [0, 100] as [number, number], guides: [20, 80], series: S.stoch ? [{ s: S.stoch.k, cls: 'ind-main', name: '%K' }, { s: S.stoch.d, cls: 'ind-signal', name: '%D' }] : [] };
    case 'williams':
      return { label: `Williams %R(${P.williamsR},${P.williamsD})`, fixed: [-100, 0] as [number, number], guides: [-80, -20], series: S.williams ? [{ s: S.williams.r, cls: 'ind-main', name: '%R' }, { s: S.williams.d, cls: 'ind-signal', name: '%D' }] : [] };
    case 'mfi':
      return { label: `MFI(${P.mfiPeriod})`, fixed: [0, 100] as [number, number], guides: [20, 80], series: S.mfi ? [{ s: S.mfi, cls: 'ind-main', name: 'MFI' }] : [] };
    case 'obv':
      return { label: `OBV (MA ${P.obvMa})`, fixed: null, guides: [], series: S.obv ? [{ s: S.obv.obv, cls: 'ind-main', name: 'OBV' }, { s: S.obv.ma, cls: 'ind-signal', name: 'MA' }] : [] };
    case 'cci':
      return { label: `CCI(${P.cciPeriod})`, fixed: null, guides: [-100, 100], series: S.cci ? [{ s: S.cci, cls: 'ind-main', name: 'CCI' }] : [] };
    case 'atr':
      return { label: `ATR(${P.atrPeriod})`, fixed: null, min: 0, guides: [], series: S.atr ? [{ s: S.atr, cls: 'ind-main', name: 'ATR' }] : [] };
    case 'roc':
      return { label: `ROC(${P.rocPeriod},${P.rocSignal})`, fixed: null, guides: [0], series: S.roc ? [{ s: S.roc.roc, cls: 'ind-main', name: 'ROC' }, { s: S.roc.signal, cls: 'ind-signal', name: '시그널' }] : [] };
    case 'trix':
      return { label: `TRIX(${P.trixPeriod},${P.trixSignal})`, fixed: null, guides: [0], series: S.trix ? [{ s: S.trix.trix, cls: 'ind-main', name: 'TRIX' }, { s: S.trix.signal, cls: 'ind-signal', name: '시그널' }] : [] };
    case 'disparity':
      return { label: `이격도(${P.disparity})`, fixed: null, guides: [100], series: S.disparity ? [{ s: S.disparity, cls: 'ind-main', name: '이격도' }] : [] };
    case 'psy':
      return { label: `심리도(${P.psyPeriod})`, fixed: [0, 100] as [number, number], guides: [25, 75], series: S.psy ? [{ s: S.psy, cls: 'ind-main', name: '심리도' }] : [] };
    default:
      return { label: key, fixed: null, guides: [], series: [] };
  }
}

const fmtNum = (v: number | undefined, d = 1) => {
  if (!ok(v)) return '-';
  if (Math.abs(v) >= 100000) return Math.round(v).toLocaleString('ko-KR');
  return v.toLocaleString('ko-KR', { maximumFractionDigits: d, minimumFractionDigits: d });
};

export function PriceChart(props: Props) {
  const { bars, warm = [], warmBot = [], kind, n, indicators, trades, buyPrice, date, xTicks = [], interactive = false, tooltip = true, onHover } = props;
  const P = props.params ?? HTS_PARAMS;
  const chartStyle = props.style ?? 'candle';
  const showSignals = props.signals ?? true;
  const intraday = kind === 'tick' || kind === 'min';
  // [2026-11-23] previous trading day's close - the reference line of tick / minute charts
  const refPrice = intraday ? (refCloseAt(bars, bars.length - 1, true) ?? (warm.length ? warm[warm.length - 1].c : undefined) ?? props.refClose) : undefined;
  const mode = props.mode ?? (intraday ? 'time' : 'index');
  const axis = props.axis ?? 'left';
  // [2026-11-20] live values while a pane handle is dragged (committed through onVolRatio / onOscHeight on release)
  const [liveH, setLiveH] = useState<Record<string, number> | null>(null);
  const oscH = props.oscHeight ?? OSC_PANE_H;
  const showVol = props.volume ?? true;
  const height = props.height ?? chartHeight(indicators, oscH);
  const L = props.left ?? PLOT_LEFT;
  const R = props.right ?? PLOT_RIGHT;

  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [hoverY, setHoverY] = useState<number | null>(null);
  /** index mode viewport: last visible index (exclusive) + bar count; null = follow the newest bars */
  const [view, setView] = useState<{ end: number; count: number } | null>(null);
  const drag = useRef<{ x: number; end: number; count: number; slot: number } | null>(null);
  const downAt = useRef<{ x: number; y: number } | null>(null);
  /** first point of a trend line being drawn */
  const [pending, setPending] = useState<{ t: number; p: number } | null>(null);
  const tool = props.tool ?? 'none';
  useEffect(() => setPending(null), [tool]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const osc = oscOf(indicators);
  const W = Math.max(10, width - L - R);
  // [2026-11-21] panes under the price pane: volume (when on), then one per indicator - each with its own height
  const paneKeys: string[] = [...(showVol ? ['volume'] : []), ...osc];
  const inner = height - TOP - AXIS_H;
  const hmap = liveH ?? props.paneHeights ?? {};
  const defH = (k: string) => (k === 'volume' ? Math.min(oscH, Math.round(inner * 0.2)) : oscH);
  let hs = paneKeys.map((k) => Math.max(MIN_PANE_H, hmap[k] ?? defH(k)));
  const room = Math.max(paneKeys.length * 16, inner - Math.min(MIN_PRICE_H, inner * 0.45) - paneKeys.length * GAP);
  const sumH = hs.reduce((a, b) => a + b, 0);
  if (sumH > room) hs = hs.map((v) => (v * room) / sumH); // a small window: every pane gives way in proportion
  const priceH = inner - hs.reduce((a, b) => a + b + GAP, 0);
  const paneTop = (j: number) => TOP + priceH + GAP + hs.slice(0, j).reduce((a, b) => a + b + GAP, 0);
  const vOff = showVol ? 1 : 0;
  const volTop = showVol ? paneTop(0) : TOP + priceH;
  const volH = showVol ? hs[0] : 0;
  const oscTop = (j: number) => paneTop(j + vOff);
  const oscHt = (j: number) => hs[j + vOff];
  const bottom = paneKeys.length ? paneTop(paneKeys.length - 1) + hs[paneKeys.length - 1] : TOP + priceH;
  const shift = indicators.includes('ichimoku') ? P.ichiShift : 0;
  const defaultCount = Math.max(MIN_BARS, Math.floor(W / (interactive ? 8 : 7)) - shift);

  // indicators over warm-up + bars (recomputed only when the data or the selection changes)
  const invMonths = props.invMonths ?? 3;
  const S = useMemo(() => {
    const s = computeIndicators(bars, indicators, { warm, warmBot }, P, intraday);
    if (indicators.includes('investor') && props.investor?.length) s.investor = investorCum(bars, props.investor, invMonths);
    return s;
  }, [bars, warm, warmBot, indicators, intraday, P, props.investor, invMonths]);

  // viewport: follow the newest bar unless the user zoomed/panned away from it
  const vp = useMemo(() => {
    const count = Math.min(bars.length, Math.max(MIN_BARS, view?.count ?? defaultCount));
    const end = view ? Math.min(bars.length, Math.max(Math.min(count, bars.length), view.end)) : bars.length;
    return { start: Math.max(0, end - count), end, count };
  }, [view, bars.length, defaultCount]);

  const model = useMemo(() => {
    if (!bars.length || width <= 0) return null;
    type Slot = { b: Bar; i: number; x: number; w: number; prevClose?: number };
    const slots: Slot[] = [];
    let d0 = 0;
    let d1 = 1;
    let highlight: { x: number; w: number } | null = null;
    let step = 6;
    const dayLines: number[] = [];

    if (mode === 'time') {
      const end = (i: number) => (kind === 'min' ? bars[i].t + n * 60000 : i + 1 < bars.length ? bars[i + 1].t : (bars[i].t2 ?? bars[i].t) + 1000);
      [d0, d1] = props.domain && props.domain[1] > props.domain[0] ? props.domain : [bars[0].t, end(bars.length - 1)];
      const sx = (t: number) => L + ((t - d0) / (d1 - d0)) * W;
      if (kind === 'min') {
        step = sx(d0 + n * 60000) - sx(d0);
        bars.forEach((b, i) => {
          const e = end(i);
          if (e < d0 || b.t > d1) return;
          const x0 = sx(b.t);
          const x1 = sx(e);
          slots.push({ b, i, x: (x0 + x1) / 2, w: Math.max(1, (x1 - x0) * 0.7), prevClose: i > 0 ? bars[i - 1].c : undefined });
        });
      } else {
        // tick bars cover uneven time: each candle sits at its last trade, width = median spacing
        const at = (b: Bar) => b.t2 ?? b.t;
        const vis = bars.map((b, i) => ({ b, i })).filter(({ b }) => at(b) >= d0 && b.t <= d1);
        const gaps = vis.slice(1).map(({ b }, j) => sx(at(b)) - sx(at(vis[j].b))).sort((a, b) => a - b);
        step = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 6;
        const w = Math.max(1, Math.min(12, step * 0.8));
        for (const { b, i } of vis) slots.push({ b, i, x: sx(at(b)), w, prevClose: i > 0 ? bars[i - 1].c : undefined });
      }
    } else {
      const { start, end, count } = vp;
      const slot = W / Math.max(count + shift, interactive ? 0 : Math.min(defaultCount, 40));
      step = slot;
      for (let i = start; i < end; i++) {
        // right-aligned, leaving `shift` empty slots for the ichimoku cloud ahead
        const x = L + W - slot * (end - i - 0.5 + shift);
        slots.push({ b: bars[i], i, x, w: Math.max(1, Math.min(24, slot * 0.7)), prevClose: i > 0 ? bars[i - 1].c : undefined });
        if (!intraday && samePeriod(kind, bars[i].t, date)) highlight = { x: x - slot / 2, w: slot };
        if (intraday && i > start && iso(bars[i].t).slice(0, 10) !== iso(bars[i - 1].t).slice(0, 10)) dayLines.push(x - slot / 2);
      }
    }
    if (!slots.length) return { empty: true as const };

    const lastSlot = slots[slots.length - 1];
    const futureIdx = lastSlot.i === bars.length - 1 && shift ? Array.from({ length: shift }, (_, k) => ({ x: lastSlot.x + step * (k + 1), idx: bars.length + k })) : [];

    // ---- price y scale ----
    let lo = Infinity;
    let hi = -Infinity;
    let vMax = 0;
    const take = (v: number | undefined) => {
      if (ok(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    };
    for (const s of slots) {
      take(s.b.l);
      take(s.b.h);
      vMax = Math.max(vMax, s.b.v);
      if (S.bb) (take(S.bb.upper[s.i]), take(S.bb.lower[s.i]));
      if (S.ichimoku) (take(S.ichimoku.senkouA[s.i]), take(S.ichimoku.senkouB[s.i]), take(S.ichimoku.kijun[s.i]));
      if (S.ma) for (const p of MA_PERIODS) take(S.ma[p][s.i]);
      if (S.vwap) take(S.vwap[s.i]);
      if (S.env) (take(S.env.upper[s.i]), take(S.env.lower[s.i]));
      if (S.donchian) (take(S.donchian.upper[s.i]), take(S.donchian.lower[s.i]));
      if (S.sar) take(S.sar.sar[s.i]);
    }
    if (S.ichimoku) for (const f of futureIdx) (take(S.ichimoku.senkouA[f.idx]), take(S.ichimoku.senkouB[f.idx]));
    const t0 = slots[0].b.t;
    const t1 = (lastSlot.b.t2 ?? lastSlot.b.t) + (kind === 'min' ? n * 60000 : 1000);
    const inRange = (t: number) => (mode === 'time' ? t >= d0 && t <= d1 : !intraday || (t >= t0 && t <= t1));
    const marks = trades.filter((t) => Number.isFinite(t.price) && t.price > 0 && inRange(t.ts));
    for (const t of marks) take(t.price);
    if (buyPrice && buyPrice > 0 && buyPrice > lo * 0.8 && buyPrice < hi * 1.2) take(buyPrice);
    if (refPrice && refPrice > 0 && refPrice > lo * 0.85 && refPrice < hi * 1.15) take(refPrice);
    const pad = (hi - lo) * 0.06 || hi * 0.01 || 1;
    lo -= pad;
    hi += pad;
    const sy = (p: number) => TOP + ((hi - p) / (hi - lo)) * priceH;
    const py = (y: number) => hi - ((y - TOP) / priceH) * (hi - lo);
    const sv = (v: number) => volTop + volH - (vMax > 0 ? (v / vMax) * volH : 0);

    // x of a trade time: time mode -> scale; index mode -> the bar holding it
    const xOfTime = (t: number): number | null => {
      if (mode === 'time') return L + ((t - d0) / (d1 - d0)) * W;
      let best: (typeof slots)[number] | null = null;
      for (const s of slots) if (s.b.t <= t) best = s;
      return best ? best.x : null;
    };

    // ---- candles ----
    let upW = '';
    let dnW = '';
    let upB = '';
    let dnB = '';
    let upV = '';
    let dnV = '';
    for (const s of slots) {
      const { b, x, w } = s;
      const up = b.c >= b.o;
      const wick = `M${x.toFixed(1)} ${sy(b.h).toFixed(1)}V${sy(b.l).toFixed(1)}`;
      const yTop = sy(Math.max(b.o, b.c));
      const bh = Math.max(1, sy(Math.min(b.o, b.c)) - yTop);
      const body = `M${(x - w / 2).toFixed(1)} ${yTop.toFixed(1)}h${w.toFixed(1)}v${bh.toFixed(1)}h${(-w).toFixed(1)}Z`;
      const vy = sv(b.v);
      const vol = showVol && b.v > 0 ? `M${(x - w / 2).toFixed(1)} ${vy.toFixed(1)}h${w.toFixed(1)}V${(volTop + volH).toFixed(1)}h${(-w).toFixed(1)}Z` : '';
      if (chartStyle === 'ohlc') {
        // OHLC bar: high-low line, open tick left, close tick right (drawn as "wick" paths)
        const tw = Math.max(1.5, w / 2);
        const bar = `${wick}M${(x - tw).toFixed(1)} ${sy(b.o).toFixed(1)}H${x.toFixed(1)}M${x.toFixed(1)} ${sy(b.c).toFixed(1)}H${(x + tw).toFixed(1)}`;
        if (up) (upW += bar), (upV += vol);
        else (dnW += bar), (dnV += vol);
        continue;
      }
      if (up) (upW += wick), (upB += body), (upV += vol);
      else (dnW += wick), (dnB += body), (dnV += vol);
    }
    const closeLine = chartStyle === 'line' ? line(slots.map((sl) => [sl.x, sl.b.c]), sy) : '';
    // x of a bar time on any axis (drawn trend lines keep their bar times)
    const xAtTime = (t: number): number | null => {
      if (mode === 'time') return L + ((t - d0) / (d1 - d0)) * W;
      const i = bars.findIndex((b) => b.t >= t);
      const idx = i < 0 ? bars.length - 1 : bars[i].t === t ? i : Math.max(0, i - 1);
      const first = slots[0];
      return first.x + (idx - first.i) * step;
    };

    // ---- overlays ----
    const pts = (s: Series | undefined) => slots.map((sl) => [sl.x, s?.[sl.i]] as [number, number | undefined]);
    const overlay: PaneLine[] = [];
    let cloudPaths: { up: string; dn: string } | null = null;
    if (S.ma) for (const p of MA_PERIODS) overlay.push({ color: MA_COLORS[p], d: line(pts(S.ma[p]), sy) });
    // [2026-11-19] Bollinger: a faint band between upper and lower instead of two more curves, the mid line stays
    let bandPath = '';
    if (S.bb) {
      const bb = cloud(slots.map((sl) => ({ x: sl.x, a: S.bb!.upper[sl.i], b: S.bb!.lower[sl.i] })), sy);
      bandPath = bb.up + bb.dn;
      overlay.push({ cls: 'ind-bb-mid', d: line(pts(S.bb.mid), sy) });
    }
    if (S.vwap) overlay.push({ cls: 'ind-vwap', d: line(pts(S.vwap), sy) });
    if (S.env) overlay.push({ cls: 'ind-env', d: line(pts(S.env.upper), sy) }, { cls: 'ind-env', d: line(pts(S.env.lower), sy) }, { cls: 'ind-env-mid', d: line(pts(S.env.mid), sy) });
    if (S.donchian) overlay.push({ cls: 'ind-donchian', d: line(pts(S.donchian.upper), sy) }, { cls: 'ind-donchian', d: line(pts(S.donchian.lower), sy) });
    // parabolic SAR: a small dot per bar (red under the price in an uptrend, blue over it in a downtrend)
    let sarUp = '';
    let sarDn = '';
    if (S.sar) {
      const r = Math.max(1.2, Math.min(2.5, step * 0.18));
      for (const sl of slots) {
        const v = S.sar.sar[sl.i];
        if (!ok(v)) continue;
        const dot = `M${(sl.x - r).toFixed(1)} ${sy(v).toFixed(1)}a${r} ${r} 0 1 0 ${(2 * r).toFixed(1)} 0a${r} ${r} 0 1 0 ${(-2 * r).toFixed(1)} 0`;
        if (S.sar.trend[sl.i] > 0) sarUp += dot;
        else sarDn += dot;
      }
    }
    if (S.ichimoku) {
      const k = S.ichimoku;
      const cp = [...slots.map((sl) => ({ x: sl.x, a: k.senkouA[sl.i], b: k.senkouB[sl.i] })), ...futureIdx.map((f) => ({ x: f.x, a: k.senkouA[f.idx], b: k.senkouB[f.idx] }))];
      cloudPaths = cloud(cp, sy);
      // [2026-11-19] the cloud is a fill only (no span lines) and the lagging span is not drawn: both crowded the candles
      overlay.push({ cls: 'ind-tenkan', d: line(pts(k.tenkan), sy) }, { cls: 'ind-kijun', d: line(pts(k.kijun), sy) });
    }

    // ---- oscillator panes ----
    const slotOf = new Map(slots.map((sl) => [sl.i, sl]));
    const panes = osc.map((key, j) => {
      const cfg = paneOf(key, S, P, invMonths);
      const top = oscTop(j);
      const ph = oscHt(j);
      let vlo = cfg.fixed ? cfg.fixed[0] : Infinity;
      let vhi = cfg.fixed ? cfg.fixed[1] : -Infinity;
      if (!cfg.fixed) {
        for (const sl of slots) for (const sr of [...cfg.series.map((x) => x.s), ...(cfg.hist ? [cfg.hist] : [])]) {
          const v = sr[sl.i];
          if (ok(v)) (vlo = Math.min(vlo, v)), (vhi = Math.max(vhi, v));
        }
        if (!(vhi > vlo)) (vlo = (ok(vlo) ? vlo : 0) - 1), (vhi = (ok(vhi) ? vhi : 0) + 1);
        if ('symmetric' in cfg && cfg.symmetric) {
          const m = Math.max(Math.abs(vlo), Math.abs(vhi));
          (vlo = -m), (vhi = m);
        }
        if ('min' in cfg && cfg.min !== undefined) vlo = Math.min(vlo, cfg.min);
        const pd = (vhi - vlo) * 0.08;
        (vlo -= pd), (vhi += pd);
      }
      const y = (v: number) => top + ((vhi - v) / (vhi - vlo)) * ph;
      let histUp = '';
      let histDn = '';
      if (cfg.hist) {
        for (const sl of slots) {
          const h = cfg.hist[sl.i];
          if (!ok(h)) continue;
          const y0 = y(0);
          const y1 = y(h);
          const r = `M${(sl.x - sl.w / 2).toFixed(1)} ${Math.min(y0, y1).toFixed(1)}h${sl.w.toFixed(1)}v${Math.max(0.5, Math.abs(y1 - y0)).toFixed(1)}h${(-sl.w).toFixed(1)}Z`;
          if (h >= 0) histUp += r;
          else histDn += r;
        }
      }
      const lines = cfg.series.map((sr) => ({ cls: sr.cls, d: line(pts(sr.s), y) }));
      const ticks = cfg.fixed ? cfg.guides : niceTicks(vlo, vhi, 2);
      // buy ▲ under / sell ▼ over the pane's main line at each signal bar in view
      let sigBuy = '';
      let sigSell = '';
      const on = signalLine(key, S);
      if (showSignals && on) {
        const a = Math.max(3, Math.min(5.5, step * 0.45));
        for (const g of signalsOf(key, S)) {
          const sl = slotOf.get(g.i);
          const v = on[g.i];
          if (!sl || !ok(v)) continue;
          const x = sl.x;
          if (g.side === 'buy') {
            const yb = Math.min(top + ph - 1, y(v) + 3 + a * 1.6);
            sigBuy += `M${x.toFixed(1)} ${(yb - a * 1.6).toFixed(1)}L${(x + a).toFixed(1)} ${yb.toFixed(1)}L${(x - a).toFixed(1)} ${yb.toFixed(1)}Z`;
          } else {
            const yt = Math.max(top + 1, y(v) - 3 - a * 1.6);
            sigSell += `M${x.toFixed(1)} ${(yt + a * 1.6).toFixed(1)}L${(x + a).toFixed(1)} ${yt.toFixed(1)}L${(x - a).toFixed(1)} ${yt.toFixed(1)}Z`;
          }
        }
      }
      return { key, top, h: ph, y, lines, guides: cfg.guides, ticks, label: cfg.label, histUp, histDn, series: cfg.series, sigBuy, sigSell };
    });

    // ---- x labels ----
    const labels: { x: number; text: string }[] = [];
    if (mode === 'time') {
      const sx = (t: number) => L + ((t - d0) / (d1 - d0)) * W;
      for (const t of xTicks) if (t >= d0 && t <= d1) labels.push({ x: sx(t), text: fmtKstTime(t).slice(0, 5) });
    } else {
      const gap = slots.length > 1 ? slots[1].x - slots[0].x : 70;
      const every = Math.max(1, Math.ceil(80 / gap));
      let lastKey = '';
      slots.forEach((s, j) => {
        if (j % every) return;
        if (s.x < L + 32 || s.x > L + W - 32) return; // a centered label there would be cut off
        if (intraday) {
          const day = iso(s.b.t).slice(5, 10);
          labels.push({ x: s.x, text: day !== lastKey ? `${day} ${fmtKstTime(s.b.t).slice(0, 5)}` : fmtKstTime(s.b.t).slice(0, 5) });
          lastKey = day;
        } else {
          const yr = iso(s.b.t).slice(0, 4);
          labels.push({ x: s.x, text: fmtDay(s.b.t, kind, yr !== lastKey) });
          lastKey = yr;
        }
      });
    }
    return { empty: false as const, vMax, sv, slots, sy, py, lo, hi, upW, dnW, upB, dnB, upV, dnV, labels, highlight, marks, overlay, cloudPaths, bandPath, panes, xOfTime, dayLines, step, closeLine, xAtTime, sarUp, sarDn };
  }, [bars, S, indicators, kind, n, mode, props.domain, xTicks, trades, buyPrice, refPrice, date, width, intraday, L, W, priceH, volTop, volH, hs.join(), osc.join(), vp, shift, defaultCount, interactive, chartStyle, showSignals, invMonths, showVol]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- interaction ----
  const nearest = (mx: number): number | null => {
    if (!model || model.empty) return null;
    let best = 0;
    let bd = Infinity;
    model.slots.forEach((s, j) => {
      const d = Math.abs(s.x - mx);
      if (d < bd) (bd = d), (best = j);
    });
    return best;
  };
  const setHov = (j: number | null) => {
    setHover(j);
    if (j === null || !model || model.empty) onHover?.(null);
    else onHover?.(model.slots[j].i, model.slots[j].x < L + W / 2);
  };
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!model || model.empty) return;
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    if (drag.current && mode === 'index') {
      const dx = mx - drag.current.x;
      const moved = Math.round(dx / drag.current.slot);
      const end = Math.min(bars.length, Math.max(drag.current.count, drag.current.end - moved));
      setView({ end, count: drag.current.count });
      return;
    }
    if (mx < L || mx > L + W) return setHov(null);
    setHoverY(my >= TOP && my <= TOP + priceH ? my : null);
    setHov(nearest(mx));
  };
  const onClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (tool === 'none' || !model || model.empty) return;
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    // a drag (pan) is not a click
    if (downAt.current && Math.hypot(downAt.current.x - mx, downAt.current.y - my) > 4) return;
    if (my < TOP || my > TOP + priceH || mx < L || mx > L + W) return;
    const price = model.py(my);
    if (tool === 'hline') props.onAddHline?.(Math.round(price));
    else if (tool === 'tline') {
      const j = nearest(mx);
      if (j === null) return;
      const t = model.slots[j].b.t;
      if (!pending) setPending({ t, p: price });
      else {
        props.onAddTline?.({ t1: pending.t, p1: pending.p, t2: t, p2: price });
        setPending(null);
      }
    }
  };
  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const rr = e.currentTarget.getBoundingClientRect();
    downAt.current = { x: e.clientX - rr.left, y: e.clientY - rr.top };
    if (!interactive || mode !== 'index' || e.button !== 0 || !model || model.empty) return;
    const r = e.currentTarget.getBoundingClientRect();
    drag.current = { x: e.clientX - r.left, end: vp.end, count: vp.count, slot: Math.max(1, model.step) };
  };
  useEffect(() => {
    const up = () => (drag.current = null);
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);
  // wheel zoom (non-passive listener so the page does not scroll)
  useEffect(() => {
    const el = ref.current;
    if (!el || !interactive || mode !== 'index') return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const count = Math.round(Math.min(bars.length, Math.max(MIN_BARS, vp.count * (e.deltaY > 0 ? 1.15 : 1 / 1.15))));
      // keep the bar under the cursor in place
      const r = el.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientX - r.left - L) / W));
      const anchor = vp.start + frac * vp.count;
      const end = Math.round(Math.min(bars.length, Math.max(count, anchor + (1 - frac) * count)));
      setView(end >= bars.length && !view ? { end: bars.length, count } : { end, count });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [interactive, mode, vp, bars.length, L, W, view]);
  // live: a viewport that ends at the newest bar keeps following it
  const prevLen = useRef(bars.length);
  useEffect(() => {
    if (view && view.end === prevLen.current && bars.length > prevLen.current) setView({ ...view, end: bars.length });
    prevLen.current = bars.length;
  }, [bars.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // [2026-11-21] the handle above pane j moves the border between it and what is above (pane j - 1, or the price pane):
  // down = pane j smaller and the one above larger. Double click: pane j back to its default height
  const startSplit = (j: number) => (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const y0 = e.clientY;
    const base = [...hs];
    const maxDown = base[j] - MIN_PANE_H;
    const maxUp = j > 0 ? base[j - 1] - MIN_PANE_H : priceH - Math.min(MIN_PRICE_H, priceH);
    let next: Record<string, number> = {};
    const move = (ev: MouseEvent) => {
      const dy = Math.max(-maxUp, Math.min(maxDown, ev.clientY - y0));
      const v = [...base];
      v[j] = base[j] - dy;
      if (j > 0) v[j - 1] = base[j - 1] + dy;
      next = Object.fromEntries(paneKeys.map((k, i) => [k, Math.round(v[i])]));
      setLiveH(next);
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      // every shown pane keeps the height it has now; heights of panes not shown at the moment are kept as saved
      if (Object.keys(next).length) props.onPaneHeights?.({ ...(props.paneHeights || {}), ...next });
      setLiveH(null);
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };
  const resetPane = (key: string) => {
    const m = { ...(props.paneHeights || {}) };
    delete m[key];
    props.onPaneHeights?.(m);
  };

  const h = model && !model.empty && hover !== null ? model.slots[hover] : null;
  const last = model && !model.empty ? model.slots[model.slots.length - 1] : null;
  const axisX = axis === 'left' ? L - 6 : L + W + 6;
  const anchorAxis = axis === 'left' ? 'end' : 'start';
  // [2026-11-19] legends always at the top left of their pane (they used to jump to the right edge with the cursor)
  const legendX = L + 4;
  const legendAnchor = 'start';
  const li = h ? h.i : last ? last.i : -1;
  const showLegend = axis === 'right';
  // [2026-11-19] the hovered (else the last) bar: when + OHLC + change vs the previous close, first line of the price legend
  const infoSlot = h || last;
  const infoRef = infoSlot ? refCloseAt(bars, infoSlot.i, intraday) : undefined;
  const infoY = showLegend ? 14 : 0;

  return (
    <div className={`price-chart ${interactive ? 'interactive' : ''}`} ref={ref} style={{ height }}>
      {model && !model.empty && width > 0 && (
        <svg width={width} height={height} onMouseMove={onMove} onMouseDown={onDown} onClick={onClick} className={tool !== 'none' ? 'drawing' : undefined} onMouseLeave={() => (setHov(null), setHoverY(null))} onDoubleClick={() => interactive && setView(null)}>
          <defs>
            <clipPath id={`pc-clip-${L}-${R}`}>
              <rect x={L} y={0} width={W} height={height} />
            </clipPath>
          </defs>
          {/* price grid + axis */}
          {niceTicks(model.lo, model.hi).map((p) => (
            <g key={p}>
              <line x1={L} x2={L + W} y1={model.sy(p)} y2={model.sy(p)} className="pc-grid" />
              <text x={axisX} y={model.sy(p) + 3.5} className="pc-axis" textAnchor={anchorAxis}>
                {fmtInt(p)}
              </text>
            </g>
          ))}
          {showVol && showLegend && model.vMax > 0 && volH > 40 && (
            <text x={axisX} y={model.sv(model.vMax) + 9} className="pc-axis" textAnchor={anchorAxis}>
              {fmtNum(model.vMax, 0)}
            </text>
          )}
          {!showLegend && showVol && (
            <text x={axisX} y={volTop + 10} className="pc-axis" textAnchor={anchorAxis}>
              거래량
            </text>
          )}
          {model.labels.map((l, i) => (
            <g key={i}>
              <line x1={l.x} x2={l.x} y1={TOP} y2={bottom} className="pc-grid" />
              <text x={l.x} y={height - 6} className="pc-axis" textAnchor="middle">
                {l.text}
              </text>
            </g>
          ))}
          {model.dayLines.map((x, i) => (
            <line key={i} x1={x} x2={x} y1={TOP} y2={bottom} className="pc-dayline" />
          ))}
          {model.highlight && <rect x={model.highlight.x} y={TOP} width={model.highlight.w} height={bottom - TOP} className="pc-today" />}

          <g clipPath={`url(#pc-clip-${L}-${R})`}>
            {/* [2026-11-19] back to front: context fills, reference lines, then the candles on top */}
            {model.bandPath && <path d={model.bandPath} className="ind-bb-band" />}
            {model.cloudPaths && (
              <>
                <path d={model.cloudPaths.up} className="ind-cloud-up" />
                <path d={model.cloudPaths.dn} className="ind-cloud-dn" />
              </>
            )}
            {model.overlay.map((o, i) => (
              <path key={i} d={o.d} className={`ind-line ${o.cls || ''}`} style={o.color ? { stroke: o.color } : undefined} />
            ))}
            {chartStyle === 'line' ? (
              <path d={model.closeLine} className="pc-close-line" />
            ) : (
              <>
                <path d={model.upW} className={`pc-up-line ${chartStyle === 'ohlc' ? 'ohlc' : ''}`} />
                <path d={model.dnW} className={`pc-dn-line ${chartStyle === 'ohlc' ? 'ohlc' : ''}`} />
                <path d={model.upB} className="pc-up" />
                <path d={model.dnB} className="pc-dn" />
              </>
            )}
            {showVol && <rect x={L} y={volTop} width={W} height={volH} className="pc-pane" />}
            <path d={model.upV} className="pc-up vol" />
            <path d={model.dnV} className="pc-dn vol" />
            {model.panes.map((p) => (
              <g key={p.key}>
                <rect x={L} y={p.top} width={W} height={p.h} className="pc-pane" />
                {p.guides.map((g) => (
                  <line key={g} x1={L} x2={L + W} y1={p.y(g)} y2={p.y(g)} className="pc-guide" />
                ))}
                <path d={p.histUp} className="pc-up vol" />
                <path d={p.histDn} className="pc-dn vol" />
                {p.lines.map((l, i) => (
                  <path key={i} d={l.d} className={`ind-line ${l.cls}`} />
                ))}
                {p.sigBuy && <path d={p.sigBuy} className="pc-sig-buy" />}
                {p.sigSell && <path d={p.sigSell} className="pc-sig-sell" />}
                {p.key === 'investor' && props.invNote && !p.lines.some((l) => l.d) && (
                  <text x={L + W / 2} y={p.top + p.h / 2 + 4} className="pc-axis" textAnchor="middle">
                    {props.invNote}
                  </text>
                )}
              </g>
            ))}
            {model.sarUp && <path d={model.sarUp} className="ind-sar-up" />}
            {model.sarDn && <path d={model.sarDn} className="ind-sar-dn" />}
          </g>
          {model.panes.map((p) => (
            <g key={p.key}>
              {!showLegend && (
                <text x={axisX} y={p.top + 11} className="pc-axis pc-pane-label" textAnchor={anchorAxis}>
                  {p.label}
                </text>
              )}
              {p.ticks.map((g) => (
                <text key={g} x={axisX} y={p.y(g) + 3.5} className="pc-axis" textAnchor={anchorAxis}>
                  {fmtNum(g, Math.abs(g) < 10 ? 1 : 0)}
                </text>
              ))}
              {showLegend && (
                <text x={legendX} y={p.top + 12} className="pc-legend pc-legend-halo" textAnchor={legendAnchor}>
                  <tspan className="pc-legend-name">{p.label}</tspan>
                  {p.series.map((s) => (
                    <tspan key={s.name} className={`lg-${s.cls}`} dx={8}>
                      {s.name} {!h && <tspan className="pc-legend-val">{fmtNum(s.s[li], key2digits(p.key))}</tspan>}
                    </tspan>
                  ))}
                </text>
              )}
            </g>
          ))}
          {showLegend && <Legends S={S} P={P} i={li} x={legendX} anchor={legendAnchor} y={TOP + 12 + infoY} volY={volTop + 11} vol={li >= 0 ? bars[li]?.v : undefined} names={!!h} showVol={showVol} />}
          {showLegend && infoSlot && (
            <text x={legendX} y={TOP + 12} className="pc-legend pc-legend-halo" pointerEvents="none">
              <tspan className="pc-legend-name">{barWhen(infoSlot.b, kind)}</tspan>
              {([['시', infoSlot.b.o], ['고', infoSlot.b.h], ['저', infoSlot.b.l], ['종', infoSlot.b.c]] as [string, number][]).map(([name, v], k) => (
                <tspan key={name} dx={k ? 7 : 9}>
                  {name} <tspan className={`pc-legend-val ${tone(v, infoRef)}`}>{fmtInt(v)}</tspan>
                  {infoRef ? <tspan className={`pc-legend-pct ${tone(v, infoRef)}`} dx={3}>{signedPct(v, infoRef)}</tspan> : null}
                </tspan>
              ))}
              <tspan dx={7}>량 <tspan className="pc-legend-val">{fmtInt(infoSlot.b.v)}</tspan></tspan>
              {infoRef ? <tspan dx={9} className="pc-legend-pct">{intraday || kind === 'day' ? '(전일 종가 대비)' : '(직전 봉 종가 대비)'}</tspan> : null}
            </text>
          )}

          {/* user drawings: horizontal lines (price) and trend lines (bar time + price) */}
          <g clipPath={`url(#pc-clip-${L}-${R})`}>
            {(props.hlines || []).filter((p) => p > model.lo && p < model.hi).map((p, i) => (
              <g key={`h${i}`}>
                <line x1={L} x2={L + W} y1={model.sy(p)} y2={model.sy(p)} className="pc-hline" />
                <text x={L + 4} y={model.sy(p) - 3} className="pc-label">
                  {fmtInt(p)}
                </text>
              </g>
            ))}
            {(props.tlines || []).map((l, i) => {
              const x1 = model.xAtTime(l.t1);
              const x2 = model.xAtTime(l.t2);
              if (x1 === null || x2 === null) return null;
              return <line key={`t${i}`} x1={x1} x2={x2} y1={model.sy(l.p1)} y2={model.sy(l.p2)} className="pc-tline" />;
            })}
            {pending && hoverY !== null && h && (
              <line x1={model.xAtTime(pending.t) ?? 0} x2={h.x} y1={model.sy(pending.p)} y2={hoverY} className="pc-tline pending" />
            )}
          </g>

          {/* [2026-11-23] rebound alert bells */}
          {(props.alerts || []).map((a) => {
            const x = model.xAtTime(a.t);
            if (x === null || x < L || x > L + W) return null;
            let bar: Bar | undefined;
            for (let k = bars.length - 1; k >= 0; k--) {
              if (bars[k].t <= a.t) {
                bar = bars[k];
                break;
              }
            }
            const y = Math.max(16, (bar ? model.sy(bar.h) : model.sy(model.hi)) - 8);
            return (
              <text key={a.id} x={x} y={y} textAnchor="middle" className="pc-bell" onMouseDown={(e) => e.stopPropagation()} onClick={(e) => (e.stopPropagation(), props.onAlertClick?.(a.id))}>
                🔔<title>{a.title}</title>
              </text>
            );
          })}

          {/* position + fills */}
          {refPrice && refPrice > model.lo && refPrice < model.hi && (
            <g>
              <line x1={L} x2={L + W} y1={model.sy(refPrice)} y2={model.sy(refPrice)} className="pc-ref" />
              <text x={L + 4} y={model.sy(refPrice) - 3} className="pc-label pc-ref-label">
                전일 종가 {fmtInt(refPrice)}
              </text>
            </g>
          )}
          {buyPrice && buyPrice > model.lo && buyPrice < model.hi && (
            <g>
              <line x1={L} x2={L + W} y1={model.sy(buyPrice)} y2={model.sy(buyPrice)} className="pc-pos" />
              <text x={L + W - 4} y={model.sy(buyPrice) - 4} className="pc-label" textAnchor="end">
                보유 평단 {fmtInt(buyPrice)}
              </text>
            </g>
          )}
          {model.marks.map((t, i) => {
            if (intraday) {
              const x = model.xOfTime(t.ts);
              if (x === null) return null;
              return (
                <g key={i}>
                  <line x1={x} x2={x} y1={TOP} y2={bottom} className={t.side === 'BUY' ? 'pc-buy' : 'pc-sell'} />
                  <circle cx={x} cy={model.sy(t.price)} r={4} className={t.side === 'BUY' ? 'pc-buy-dot' : 'pc-sell-dot'} />
                </g>
              );
            }
            return (
              <g key={i}>
                <line x1={L} x2={L + W} y1={model.sy(t.price)} y2={model.sy(t.price)} className={t.side === 'BUY' ? 'pc-buy' : 'pc-sell'} />
                <text x={L + 4} y={model.sy(t.price) - 3} className="pc-label">
                  {t.side === 'BUY' ? '매수' : '매도'} {fmtInt(t.price)} ({fmtKstTime(t.ts).slice(0, 5)})
                </text>
              </g>
            );
          })}

          {/* last price tag (right axis) */}
          {axis === 'right' && last && (
            <PriceTag x={L + W + 2} y={model.sy(last.b.c)} text={fmtInt(last.b.c)} cls={last.b.c >= last.b.o ? 'up' : 'dn'} />
          )}

          {/* hover */}
          {h && (
            <g pointerEvents="none">
              <line x1={h.x} x2={h.x} y1={TOP} y2={bottom} className="pc-cross" />
              {hoverY !== null ? (
                <>
                  <line x1={L} x2={L + W} y1={hoverY} y2={hoverY} className="pc-cross" />
                  {axis === 'right' && <PriceTag x={L + W + 2} y={hoverY} text={fmtInt(model.py(hoverY))} cls="cross" />}
                </>
              ) : (
                <line x1={L} x2={L + W} y1={model.sy(h.b.c)} y2={model.sy(h.b.c)} className="pc-cross" />
              )}
            </g>
          )}
        </svg>
      )}
      {/* [2026-11-21] values of the hovered bar beside the vertical line, one box per pane, on the side with more room */}
      {showLegend && h && model && !model.empty && (() => {
        const rightSide = h.x < L + W / 2;
        const pos = rightSide ? { left: h.x + 8 } : { right: width - h.x + 8 };
        // the names stay at the top left of each pane: a box that would cover them starts below them
        const near = (labelW: number, boxW: number) => (rightSide ? h.x + 8 < L + labelW : h.x - 8 - boxW < L + labelW);
        const rows = legendRows(S, P, h.i);
        return (
          <>
            {rows.length > 0 && (
              <div className="pc-fly" style={{ ...pos, top: TOP + 18 + (near(340, 300) ? 14 * rows.length : 0) }}>
                {rows.map((row, j) => (
                  <div key={j}>
                    {row.map((l, k) => (
                      <span key={k}>
                        {l.color && <i style={{ background: l.color }} />}
                        {l.label} <b>{fmtNum(l.value, l.digits ?? 1)}</b>
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            )}
            {showVol && volH > 26 && (
              <div className="pc-fly" style={{ ...pos, top: volTop + (near(50, 120) ? 14 : 1) }}>
                <span>
                  거래량 <b>{fmtInt(h.b.v)}</b>
                </span>
              </div>
            )}
            {model.panes.map((p) => {
              const hist = p.key === 'macd' ? S.macd?.hist[h.i] : undefined;
              if (!p.series.length) return null;
              return (
                <div key={p.key} className="pc-fly" style={{ ...pos, top: p.top + (near(230, 130) ? 15 : 1) }}>
                  {p.series.map((s) => (
                    <div key={s.name} className={`lg-${s.cls}`}>
                      {s.name} <b>{fmtNum(s.s[h.i], key2digits(p.key))}</b>
                    </div>
                  ))}
                  {p.key === 'macd' && (
                    <div>
                      오실레이터 <b>{fmtNum(hist, 2)}</b>
                    </div>
                  )}
                </div>
              );
            })}
          </>
        );
      })()}
      {model &&
        !model.empty &&
        width > 0 &&
        paneKeys.map((k, j) => (
          <span key={k}>
            {props.onPaneHeights && (
              <div className="pc-split" style={{ top: paneTop(j) - GAP / 2 - 4, left: L, width: W }} onMouseDown={startSplit(j)} onDoubleClick={() => resetPane(k)} title="끌어서 위 칸과 이 칸의 높이 조절 · 더블클릭: 이 칸 기본 높이" />
            )}
            {props.onRemovePane && (
              <button type="button" className="pc-pane-x" style={{ top: paneTop(j) + 2, left: L + W - 17 }} onClick={() => props.onRemovePane?.(k)} title="이 칸 닫기 (지표 메뉴에서 다시 켤 수 있음)">
                ×
              </button>
            )}
          </span>
        ))}
      {tooltip && h && (
        <div className="tip pc-tip" style={{ left: Math.min(Math.max(h.x + 12, 0), Math.max(0, width - 250)), top: 8 }}>
          <b>
            {intraday
              ? kind === 'tick' && h.b.t2 && h.b.t2 !== h.b.t
                ? `${fmtKstTime(h.b.t)} ~ ${fmtKstTime(h.b.t2)}`
                : fmtKstTime(h.b.t)
              : fmtKstDateTime(h.b.t).slice(0, 10) + (kind === 'week' ? ' 주' : kind === 'month' ? ' 월' : '')}
          </b>
          <div className="pc-ohlc">
            <span>시 {fmtInt(h.b.o)}</span>
            <span>고 {fmtInt(h.b.h)}</span>
            <span>저 {fmtInt(h.b.l)}</span>
            <span>종 {fmtInt(h.b.c)}</span>
            <span>량 {fmtInt(h.b.v)}</span>
            {h.prevClose ? (
              <span className={h.b.c > h.prevClose ? 'gain' : h.b.c < h.prevClose ? 'loss' : ''}>
                {h.b.c >= h.prevClose ? '+' : ''}
                {(((h.b.c - h.prevClose) / h.prevClose) * 100).toFixed(2)}%
              </span>
            ) : null}
          </div>
          <IndicatorLines S={S} i={h.i} />
        </div>
      )}
      {model && model.empty && <div className="empty small">이 구간에 해당하는 봉이 없습니다.</div>}
    </div>
  );
}

const key2digits = (k: IndicatorKey) => (k === 'obv' || k === 'investor' ? 0 : k === 'macd' || k === 'roc' ? 2 : k === 'trix' ? 3 : 1);

function PriceTag({ x, y, text, cls }: { x: number; y: number; text: string; cls: string }) {
  const w = text.length * 7 + 10;
  return (
    <g className={`pc-tag pc-tag-${cls}`} pointerEvents="none">
      <rect x={x} y={y - 9} width={w} height={18} rx={3} />
      <text x={x + 5} y={y + 4}>
        {text}
      </text>
    </g>
  );
}

/** price-pane overlay values at bar i, one row per group: MAs / Bollinger + VWAP / envelope + channel + SAR / Ichimoku */
function legendRows(S: IndicatorSet, P: IndicatorParams, i: number): Legend[][] {
  const rows: Legend[][] = [];
  if (S.ma) rows.push(MA_PERIODS.map((p) => ({ label: `MA${p}`, color: MA_COLORS[p], value: S.ma![p][i], digits: 0 })));
  const C = OVERLAY_COLOR;
  const r2: Legend[] = [];
  if (S.bb) r2.push({ label: `볼린저(${P.bbPeriod},${P.bbMult}) 상`, color: C.bb, value: S.bb.upper[i], digits: 0 }, { label: '중', value: S.bb.mid[i], digits: 0 }, { label: '하', value: S.bb.lower[i], digits: 0 });
  if (S.vwap) r2.push({ label: 'VWAP', color: C.vwap, value: S.vwap[i], digits: 0 });
  if (r2.length) rows.push(r2);
  const r3: Legend[] = [];
  if (S.env) r3.push({ label: `엔벨로프(${P.envPeriod},${P.envPct}%) 상`, color: C.env, value: S.env.upper[i], digits: 0 }, { label: '하', value: S.env.lower[i], digits: 0 });
  if (S.donchian) r3.push({ label: `채널(${P.donchian}) 상`, color: C.donchian, value: S.donchian.upper[i], digits: 0 }, { label: '하', value: S.donchian.lower[i], digits: 0 });
  if (S.sar) r3.push({ label: 'SAR', value: S.sar.sar[i], digits: 0 });
  if (r3.length) rows.push(r3);
  if (S.ichimoku) {
    const k = S.ichimoku;
    rows.push([
      { label: '전환', color: C.tenkan, value: k.tenkan[i], digits: 0 },
      { label: '기준', color: C.kijun, value: k.kijun[i], digits: 0 },
      { label: '구름 A', value: k.senkouA[i], digits: 0 },
      { label: 'B', value: k.senkouB[i], digits: 0 },
    ]);
  }
  return rows;
}

/** price-pane legend (right-axis layout); names = names only (the values are shown beside the hover line) */
function Legends({ S, P, i, x, y, volY, vol, anchor, names, showVol }: { S: IndicatorSet; P: IndicatorParams; i: number; x: number; y: number; volY: number; vol?: number; anchor: 'start' | 'end'; names: boolean; showVol: boolean }) {
  const rows = legendRows(S, P, i);
  return (
    <g pointerEvents="none">
      {rows.map((row, j) => (
        <text key={j} x={x} y={y + j * 14} className="pc-legend pc-legend-halo" textAnchor={anchor}>
          {row.map((l, k) => (
            <tspan key={k} dx={k ? 9 : 0}>
              {l.color && <tspan style={{ fill: l.color }}>■ </tspan>}
              {l.label} {!names && <tspan className="pc-legend-val">{fmtNum(l.value, l.digits ?? 1)}</tspan>}
            </tspan>
          ))}
        </text>
      ))}
      {showVol && (
        <text x={x} y={volY} className="pc-legend pc-legend-halo" textAnchor={anchor}>
          거래량 {names ? '' : fmtInt(vol)}
        </text>
      )}
    </g>
  );
}

/** indicator values of bar i (tooltip / data panel) */
export function IndicatorLines({ S, i }: { S: IndicatorSet; i: number }) {
  const v = (s: Series | undefined, d = 1) => fmtNum(s?.[i], d);
  return (
    <>
      {S.ma && <div className="pc-ind">{MA_PERIODS.map((p) => `MA${p} ${v(S.ma![p], 0)}`).join(' · ')}</div>}
      {S.bb && (
        <div className="pc-ind">
          BB 상 {v(S.bb.upper, 0)} · 중 {v(S.bb.mid, 0)} · 하 {v(S.bb.lower, 0)}
        </div>
      )}
      {S.vwap && <div className="pc-ind">VWAP {v(S.vwap, 0)}</div>}
      {S.ichimoku && (
        <div className="pc-ind">
          일목 전환 {v(S.ichimoku.tenkan, 0)} · 기준 {v(S.ichimoku.kijun, 0)} · 선행A {v(S.ichimoku.senkouA, 0)} · 선행B {v(S.ichimoku.senkouB, 0)}
        </div>
      )}
      {S.rsi && (
        <div className="pc-ind">
          RSI {v(S.rsi.rsi)} · 시그널 {v(S.rsi.signal)}
        </div>
      )}
      {S.macd && (
        <div className="pc-ind">
          MACD {v(S.macd.macd, 2)} · 시그널 {v(S.macd.signal, 2)} · 히스토 {v(S.macd.hist, 2)}
        </div>
      )}
      {S.adx && (
        <div className="pc-ind">
          ADX {v(S.adx.adx)} · DI+ {v(S.adx.diPlus)} · DI- {v(S.adx.diMinus)}
        </div>
      )}
      {S.stoch && (
        <div className="pc-ind">
          Stoch %K {v(S.stoch.k)} · %D {v(S.stoch.d)}
        </div>
      )}
      {S.williams && (
        <div className="pc-ind">
          Williams %R {v(S.williams.r)} · %D {v(S.williams.d)}
        </div>
      )}
      {S.mfi && <div className="pc-ind">MFI {v(S.mfi)}</div>}
      {S.obv && (
        <div className="pc-ind">
          OBV {v(S.obv.obv, 0)} · MA {v(S.obv.ma, 0)}
        </div>
      )}
      {S.env && (
        <div className="pc-ind">
          엔벨로프 상 {v(S.env.upper, 0)} · 중 {v(S.env.mid, 0)} · 하 {v(S.env.lower, 0)}
        </div>
      )}
      {S.donchian && (
        <div className="pc-ind">
          가격채널 상 {v(S.donchian.upper, 0)} · 하 {v(S.donchian.lower, 0)}
        </div>
      )}
      {S.sar && <div className="pc-ind">SAR {v(S.sar.sar, 0)}</div>}
      {S.cci && <div className="pc-ind">CCI {v(S.cci)}</div>}
      {S.atr && <div className="pc-ind">ATR {v(S.atr)}</div>}
      {S.roc && (
        <div className="pc-ind">
          ROC {v(S.roc.roc, 2)} · 시그널 {v(S.roc.signal, 2)}
        </div>
      )}
      {S.trix && (
        <div className="pc-ind">
          TRIX {v(S.trix.trix, 3)} · 시그널 {v(S.trix.signal, 3)}
        </div>
      )}
      {S.disparity && <div className="pc-ind">이격도 {v(S.disparity)}</div>}
      {S.psy && <div className="pc-ind">심리도 {v(S.psy)}</div>}
    </>
  );
}
