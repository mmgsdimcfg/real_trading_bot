/*
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
 * - [2026-11-07] type=fix - The floating tooltip flips to the left of the cursor when it does not fit on the right (measured width;
 *   before, a fixed 250 px guess let wide tooltips run off the right edge)
 * - [2026-11-07] type=feat - Indicator panes mark each indicator's own buy / sell crossings (HTS style, shared crossSignal): red ▲ under
 *   / blue ▼ over the pane's main line, the rule in the hover title. The bot's fills stay as the dashed vertical lines. Volume pane
 *   2/3 of its former height (VOL_RATIO 0.2 -> 0.133)
 * - [2026-11-07] type=feat - Volume is a switch ('vol' in indicators, 거래량 check box): off = no volume pane, the price pane takes the
 *   space. Volume pane half of the previous height (VOL_RATIO 0.133 -> 0.067)
 * - [2026-11-07] type=feat - Drawing tools (with drawKey = symbol code): 선택 / 가로선 / 세로선 / 추세선 (two clicks) / 지우기 at the top
 *   right, lines kept per code (drawings.ts) in price / time coordinates so they stay put when zooming, panning or switching bar
 *   size; right-click a line to delete it, Esc cancels. Drawn on the price pane only
 * - [2026-11-07] type=feat - A drawing tool stays selected (draw several lines; ↖ / Esc to stop). Horizontal lines show the price, the
 *   difference to the latest close and to the position's average price; vertical lines show the time and that bar's OHLC / volume
 * - [2026-11-08] type=feat - Per-pane heights (oscHeights) and, with onResizePane, drag handles on the bottom edge of the price +
 *   volume area and of every indicator pane (drag = resize, double-click = default size); the chart grows / shrinks with them
 * - [2026-11-11] type=feat - hoverTime prop: another chart's hovered time selects the bar here (timeline buy / sell evaluation
 *   charts drive the price chart's crosshair and tooltip)
 * - [2026-11-12] type=fix - The floating tooltip no longer sits under the drawing tools (top right): when its box reaches them it
 *   starts below the tool bar
 * - [2026-11-12] type=fix - the tooltip closed about a second after the mouse stopped on the chart (timeline): the external
 *   hover effect also ran when the bars were refreshed and, with no external hover time, cleared the chart's own hover.
 *   Now it clears only when an external hover ends (time -> null), never while the mouse itself is on this chart
 * - [2026-11-12] type=feat - readability + zoom. (1) layer order: candles on top of the context layers - Bollinger is a faint
 *   band (no upper / lower curves) with a dashed mid line, the Ichimoku cloud is a faint fill without span lines, the lagging
 *   span is no longer drawn; one colour per family (red / blue stay with the candles, MAs are a grey ramp) (2) inline legend
 *   with colour chips and the hovered bar's values at the top left of the price pane in every layout (3) time mode: mouse
 *   wheel zooms the time axis around the cursor (shift + wheel pans), dragging with the left button selects a span to zoom
 *   into, double click resets - reported through onZoom so the parent's other charts follow; index mode: drag = zoom into the
 *   selected bars, shift + drag = pan (was plain drag)
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { samePeriod, type Bar, type ChartKind } from '../../../shared/price';
import {
  BOT_PARAMS,
  computeIndicators,
  crossSignal,
  MA_PERIODS,
  OSCILLATORS,
  SIGNAL_RULES,
  type CrossSignal,
  type IndicatorKey,
  type IndicatorSet,
  type Series,
} from '../../../shared/indicators';
import type { Trade } from '../../../shared/types';
import { fmtKstDateTime, fmtKstTime } from '../../../shared/time';
import { fmtInt } from '../lib';
import { startHeightDrag } from './HeightGrip';
import { addDrawing, clearDrawings, getDrawings, removeDrawing, subscribeDrawings, type DrawTool, type Drawing } from '../drawings';

/** must match the ScatterChart margins + YAxis width in TimelineView */
export const PLOT_LEFT = 8 + 150;
export const PLOT_RIGHT = 24;
const TOP = 10;
const AXIS_H = 22;
const GAP = 8;
export const PRICE_PANE_H = 300;
export const OSC_PANE_H = 84;
const VOL_RATIO = 0.2 / 3; // volume pane share of the price+volume area (0.2 -> 0.133 -> half of that)
const KST = 9 * 3600000;
const MIN_BARS = 10;

/** [2026-11-12] grey ramp (theme tokens): the short average is the brightest; red / blue belong to the candles */
export const MA_COLORS: Record<number, string> = { 5: 'var(--ma-1)', 20: 'var(--ma-2)', 60: 'var(--ma-3)', 120: 'var(--ma-4)' };
/** colours of the price-pane overlays (legend chips, check boxes) - the lines themselves are styled by the .ind-* classes */
export const OVERLAY_COLOR = { bb: '#9ca3af', vwap: '#a78bfa', tenkan: '#f59e0b', kijun: '#10b981' } as const;
/** shortest time span the wheel / a drag selection may zoom into */
const MIN_ZOOM_MS = 2 * 60000;

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
  /** per indicator pane height (missing = oscHeight) */
  oscHeights?: Partial<Record<IndicatorKey, number>>;
  /** shows resize handles under the price + volume area ('price') and under each indicator pane; h undefined = default size.
   *  The parent owns the heights: it passes them back through height (chartHeight) and oscHeights */
  onResizePane?: (key: 'price' | IndicatorKey, h: number | undefined) => void;
  /** index mode: wheel zoom + drag pan */
  interactive?: boolean;
  /** floating OHLC/indicator tooltip (off when the page shows its own data panel) */
  tooltip?: boolean;
  /** hovered bar index (null = none) */
  onHover?: (i: number | null) => void;
  /** [2026-11-11] external hover: epoch ms of a point in another chart - the bar holding that time is highlighted as if hovered
   *  (null clears it, undefined = not driven from outside) */
  hoverTime?: number | null;
  /** symbol code: shows the drawing tools (horizontal / vertical / trend line, clear) and the code's saved drawings */
  drawKey?: string;
  /** [2026-11-12] time mode: the user asked for another time span (wheel / drag selection); null = show everything again.
   *  The parent owns the domain (it is shared with other charts) and passes it back through `domain` */
  onZoom?: (domain: [number, number] | null) => void;
  /** time mode: the widest span onZoom may ask for (the whole day) */
  zoomLimit?: [number, number];
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

export function chartHeight(
  indicators: IndicatorKey[],
  oscH = OSC_PANE_H,
  priceH = PRICE_PANE_H,
  oscHeights?: Partial<Record<IndicatorKey, number>>,
): number {
  return priceH + oscOf(indicators).reduce((sum, k) => sum + (oscHeights?.[k] ?? oscH) + GAP, 0);
}

/** resize limits (px) of the price + volume area and of one indicator pane */
export const PRICE_AREA_MIN = 140;
export const PRICE_AREA_MAX = 1600;
export const OSC_PANE_MIN = 40;
export const OSC_PANE_MAX = 600;

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
function paneOf(key: IndicatorKey, S: IndicatorSet) {
  const P = BOT_PARAMS;
  switch (key) {
    case 'rsi':
      return { label: `RSI(${P.rsiPeriod},${P.rsiSignal})`, fixed: [0, 100] as [number, number], guides: [30, 70], series: S.rsi ? [{ s: S.rsi.rsi, cls: 'ind-main', name: 'RSI' }, { s: S.rsi.signal, cls: 'ind-signal', name: '시그널' }] : [] };
    case 'adx':
      return { label: `ADX(${P.adxPeriod})`, fixed: null, min: 0, guides: [25], series: S.adx ? [{ s: S.adx.diPlus, cls: 'ind-di-plus', name: 'DI+' }, { s: S.adx.diMinus, cls: 'ind-di-minus', name: 'DI-' }, { s: S.adx.adx, cls: 'ind-main', name: 'ADX' }] : [] };
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
  const { bars, warm = [], warmBot = [], kind, n, indicators, trades, buyPrice, date, xTicks = [], interactive = false, tooltip = true, onHover, drawKey, onResizePane } = props;
  const intraday = kind === 'tick' || kind === 'min';
  const mode = props.mode ?? (intraday ? 'time' : 'index');
  const axis = props.axis ?? 'left';
  const oscH = props.oscHeight ?? OSC_PANE_H;
  const height = props.height ?? chartHeight(indicators, oscH, PRICE_PANE_H, props.oscHeights);
  const L = props.left ?? PLOT_LEFT;
  const R = props.right ?? PLOT_RIGHT;

  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [hoverY, setHoverY] = useState<number | null>(null);
  /** [2026-11-12] the hover currently comes from props.hoverTime (another chart), not from the mouse on this chart */
  const extHoverRef = useRef(false);
  /** index mode viewport: last visible index (exclusive) + bar count; null = follow the newest bars */
  const [view, setView] = useState<{ end: number; count: number } | null>(null);
  const drag = useRef<{ x: number; end: number; count: number; slot: number } | null>(null);
  /** [2026-11-12] drag selection (zoom into a span): x of the press and of the pointer */
  const selRef = useRef<{ x0: number; x1: number } | null>(null);
  const [sel, setSel] = useState<{ x0: number; x1: number } | null>(null);
  const applySel = useRef<(x0: number, x1: number) => void>(() => undefined);

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
  const oscHs = osc.map((k) => props.oscHeights?.[k] ?? oscH);
  const oscSum = oscHs.reduce((a, h) => a + h + GAP, 0);
  /** the price + volume area as chartHeight counts it (resize handle start value) */
  const priceArea = height - oscSum;
  const mainH = priceArea - TOP - AXIS_H;
  const showVol = indicators.includes('vol');
  const priceH = showVol ? mainH * (1 - VOL_RATIO) - GAP : mainH;
  const volTop = TOP + priceH + (showVol ? GAP : 0);
  const volH = showVol ? mainH * VOL_RATIO : 0;
  const oscTop = (j: number) => volTop + volH + GAP + oscHs.slice(0, j).reduce((a, h) => a + h + GAP, 0);
  const bottom = osc.length ? oscTop(osc.length - 1) + oscHs[osc.length - 1] : volTop + volH;
  const shift = indicators.includes('ichimoku') ? BOT_PARAMS.ichiShift : 0;
  const defaultCount = Math.max(MIN_BARS, Math.floor(W / (interactive ? 8 : 7)) - shift);

  // indicators over warm-up + bars (recomputed only when the data or the selection changes)
  const S = useMemo(() => computeIndicators(bars, indicators, { warm, warmBot }, BOT_PARAMS, intraday), [bars, warm, warmBot, indicators, intraday]);

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
    }
    if (S.ichimoku) for (const f of futureIdx) (take(S.ichimoku.senkouA[f.idx]), take(S.ichimoku.senkouB[f.idx]));
    const t0 = slots[0].b.t;
    const t1 = (lastSlot.b.t2 ?? lastSlot.b.t) + (kind === 'min' ? n * 60000 : 1000);
    const inRange = (t: number) => (mode === 'time' ? t >= d0 && t <= d1 : !intraday || (t >= t0 && t <= t1));
    const marks = trades.filter((t) => Number.isFinite(t.price) && t.price > 0 && inRange(t.ts));
    for (const t of marks) take(t.price);
    if (buyPrice && buyPrice > 0 && buyPrice > lo * 0.8 && buyPrice < hi * 1.2) take(buyPrice);
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
      const vol = b.v > 0 ? `M${(x - w / 2).toFixed(1)} ${vy.toFixed(1)}h${w.toFixed(1)}V${(volTop + volH).toFixed(1)}h${(-w).toFixed(1)}Z` : '';
      if (up) (upW += wick), (upB += body), (upV += vol);
      else (dnW += wick), (dnB += body), (dnV += vol);
    }

    // ---- overlays ----
    const pts = (s: Series | undefined) => slots.map((sl) => [sl.x, s?.[sl.i]] as [number, number | undefined]);
    const overlay: PaneLine[] = [];
    let cloudPaths: { up: string; dn: string } | null = null;
    if (S.ma) for (const p of MA_PERIODS) overlay.push({ color: MA_COLORS[p], d: line(pts(S.ma[p]), sy) });
    // [2026-11-12] Bollinger: a faint band between upper and lower instead of two more curves, the mid line stays
    let bandPath = '';
    if (S.bb) {
      const b = cloud(slots.map((sl) => ({ x: sl.x, a: S.bb!.upper[sl.i], b: S.bb!.lower[sl.i] })), sy);
      bandPath = b.up + b.dn;
      overlay.push({ cls: 'ind-bb-mid', d: line(pts(S.bb.mid), sy) });
    }
    if (S.vwap) overlay.push({ cls: 'ind-vwap', d: line(pts(S.vwap), sy) });
    if (S.ichimoku) {
      const k = S.ichimoku;
      const cp = [...slots.map((sl) => ({ x: sl.x, a: k.senkouA[sl.i], b: k.senkouB[sl.i] })), ...futureIdx.map((f) => ({ x: f.x, a: k.senkouA[f.idx], b: k.senkouB[f.idx] }))];
      cloudPaths = cloud(cp, sy);
      // [2026-11-12] the cloud is a fill only (no span lines) and the lagging span is not drawn: both crowded the candles
      overlay.push({ cls: 'ind-tenkan', d: line(pts(k.tenkan), sy) }, { cls: 'ind-kijun', d: line(pts(k.kijun), sy) });
    }

    // ---- oscillator panes ----
    const panes = osc.map((key, j) => {
      const cfg = paneOf(key, S);
      const top = oscTop(j);
      const ph = oscHs[j];
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
      // the indicator's own buy / sell crossings on the visible bars, drawn on its main line
      const main = cfg.series.find((x) => x.cls === 'ind-main') || cfg.series[0];
      const signals: { i: number; x: number; y: number; side: CrossSignal }[] = [];
      if (main) {
        for (const sl of slots) {
          const side = crossSignal(key, S, sl.i);
          const v = main.s[sl.i];
          if (side && ok(v)) signals.push({ i: sl.i, x: sl.x, y: y(v), side });
        }
      }
      return { key, top, h: ph, y, lines, guides: cfg.guides, ticks, label: cfg.label, histUp, histDn, series: cfg.series, signals };
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
    // drawings: time <-> x for the whole bar range (index mode: by bar position, may fall outside the view - clipped)
    const xOfT = (t: number): number => {
      if (mode === 'time') return L + ((t - d0) / (d1 - d0)) * W;
      let lo2 = 0;
      let hi2 = bars.length - 1;
      let i = -1;
      while (lo2 <= hi2) {
        const mid = (lo2 + hi2) >> 1;
        if (bars[mid].t <= t) (i = mid), (lo2 = mid + 1);
        else hi2 = mid - 1;
      }
      return L + W - step * (vp.end - i - 0.5 + shift);
    };
    const tOfX = (x: number): number => {
      if (mode === 'time') return d0 + ((x - L) / W) * (d1 - d0);
      let best = slots[0];
      for (const sl of slots) if (Math.abs(sl.x - x) < Math.abs(best.x - x)) best = sl;
      return best.b.t;
    };
    return { empty: false as const, slots, sy, py, lo, hi, upW, dnW, upB, dnB, upV, dnV, labels, highlight, marks, overlay, cloudPaths, bandPath, panes, xOfTime, xOfT, tOfX, dayLines, step, d0, d1 };
  }, [bars, S, indicators, kind, n, mode, props.domain, xTicks, trades, buyPrice, date, width, intraday, L, W, priceH, volTop, volH, oscHs.join(), osc.join(), vp, shift, defaultCount, interactive]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- drawings (HTS-style tools) ----
  const [tool, setTool] = useState<DrawTool>('pointer');
  const [drawings, setDrawings] = useState<Drawing[]>(() => (drawKey ? getDrawings(drawKey) : []));
  const [pending, setPending] = useState<{ t: number; p: number } | null>(null);
  const [ghost, setGhost] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    if (!drawKey) return setDrawings([]);
    setDrawings(getDrawings(drawKey));
    setPending(null);
    return subscribeDrawings(() => setDrawings(getDrawings(drawKey)));
  }, [drawKey]);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && (setPending(null), setTool('pointer'));
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, []);
  /** a click with a drawing tool (price pane only). Trend line = two clicks. Returns true when handled. */
  const drawClick = (mx: number, my: number): boolean => {
    if (!drawKey || tool === 'pointer' || !model || model.empty) return false;
    if (mx < L || mx > L + W || my < TOP || my > TOP + priceH) return true;
    const t = model.tOfX(mx);
    const p = model.py(my);
    if (tool === 'h') addDrawing(drawKey, { type: 'h', p });
    else if (tool === 'v') addDrawing(drawKey, { type: 'v', t });
    else if (!pending) {
      setPending({ t, p });
      return true;
    } else {
      addDrawing(drawKey, { type: 't', t1: pending.t, p1: pending.p, t2: t, p2: p });
      setPending(null);
    }
    // the tool stays selected: several lines in a row; ↖ (or Esc) ends drawing
    return true;
  };

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
    onHover?.(j === null || !model || model.empty ? null : model.slots[j].i);
  };
  // external hover (timeline evaluation chart -> this chart): the last slot whose bar starts at or before the time
  useEffect(() => {
    if (props.hoverTime === undefined) return;
    const hadExternal = extHoverRef.current;
    extHoverRef.current = props.hoverTime !== null;
    if (props.hoverTime === null) {
      // only the end of an external hover clears; a bar refresh (model change) must not close the mouse's own tooltip
      if (hadExternal) setHover(null);
      return;
    }
    if (!model || model.empty) {
      setHover(null);
      return;
    }
    let j: number | null = null;
    model.slots.forEach((s, k) => {
      if (s.b.t <= props.hoverTime!) j = k;
    });
    setHover(j);
  }, [props.hoverTime, model]); // eslint-disable-line react-hooks/exhaustive-deps
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
    if (pending) setGhost({ x: mx, y: my });
    if (selRef.current) {
      selRef.current = { x0: selRef.current.x0, x1: Math.min(L + W, Math.max(L, mx)) };
      setSel(selRef.current);
    }
    if (mx < L || mx > L + W) return setHov(null);
    setHoverY(my >= TOP && my <= TOP + priceH ? my : null);
    setHov(nearest(mx));
  };
  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button === 0 && tool !== 'pointer') {
      const r0 = e.currentTarget.getBoundingClientRect();
      if (drawClick(e.clientX - r0.left, e.clientY - r0.top)) return;
    }
    if (e.button !== 0 || !model || model.empty) return;
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const canZoom = mode === 'time' ? !!props.onZoom : interactive;
    if (!canZoom || mx < L || mx > L + W) return;
    if (mode === 'index' && e.shiftKey) {
      drag.current = { x: mx, end: vp.end, count: vp.count, slot: Math.max(1, model.step) }; // shift + drag = pan
      return;
    }
    // [2026-11-12] left drag = select a span to zoom into (applied on release)
    e.preventDefault();
    selRef.current = { x0: mx, x1: mx };
    setSel(selRef.current);
  };
  applySel.current = (x0: number, x1: number) => {
    if (!model || model.empty || Math.abs(x1 - x0) < 6) return; // a click, not a selection
    const [a, b] = x0 < x1 ? [x0, x1] : [x1, x0];
    if (mode === 'time') {
      const t0 = model.tOfX(a);
      const t1 = model.tOfX(b);
      const mid = (t0 + t1) / 2;
      props.onZoom?.(t1 - t0 >= MIN_ZOOM_MS ? [t0, t1] : [mid - MIN_ZOOM_MS / 2, mid + MIN_ZOOM_MS / 2]);
      return;
    }
    const idx = (x: number) => vp.end - shift - (L + W - x) / Math.max(1, model.step);
    const i0 = Math.max(0, Math.floor(idx(a)));
    const i1 = Math.min(bars.length, Math.ceil(idx(b)));
    const count = Math.max(MIN_BARS, i1 - i0);
    setView({ end: Math.min(bars.length, Math.max(count, i1)), count });
  };
  useEffect(() => {
    const up = () => {
      drag.current = null;
      const s0 = selRef.current;
      if (s0) {
        selRef.current = null;
        setSel(null);
        applySel.current(s0.x0, s0.x1);
      }
    };
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);
  // [2026-11-12] time mode: wheel = zoom the time axis around the cursor, shift + wheel (or a sideways wheel) = pan
  const zoomRef = useRef<{ d0: number; d1: number; lim?: [number, number]; onZoom?: Props['onZoom'] }>({ d0: 0, d1: 0 });
  zoomRef.current = { d0: model && !model.empty ? model.d0 : 0, d1: model && !model.empty ? model.d1 : 0, lim: props.zoomLimit, onZoom: props.onZoom };
  const hasZoom = mode === 'time' && !!props.onZoom;
  useEffect(() => {
    const el = ref.current;
    if (!el || !hasZoom) return;
    const onWheel = (e: WheelEvent) => {
      const { d0, d1, lim, onZoom } = zoomRef.current;
      if (!onZoom || !(d1 > d0)) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientX - r.left - L) / W));
      const span = d1 - d0;
      let n0: number;
      let n1: number;
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const move = span * 0.15 * Math.sign(e.deltaX || e.deltaY);
        [n0, n1] = [d0 + move, d1 + move];
      } else {
        const next = Math.max(MIN_ZOOM_MS, span * (e.deltaY > 0 ? 1.25 : 0.8));
        const at = d0 + frac * span;
        [n0, n1] = [at - frac * next, at + (1 - frac) * next];
      }
      if (lim) {
        if (n1 - n0 >= lim[1] - lim[0]) return onZoom(null); // zoomed out to the whole range
        if (n0 < lim[0]) [n0, n1] = [lim[0], lim[0] + (n1 - n0)];
        if (n1 > lim[1]) [n0, n1] = [lim[1] - (n1 - n0), lim[1]];
      }
      onZoom([n0, n1]);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [hasZoom, L, W]);
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

  const h = model && !model.empty && hover !== null ? model.slots[hover] : null;
  const last = model && !model.empty ? model.slots[model.slots.length - 1] : null;
  const axisX = axis === 'left' ? L - 6 : L + W + 6;
  const anchorAxis = axis === 'left' ? 'end' : 'start';
  const legendX = axis === 'left' ? L + 4 : L + 4;
  const li = h ? h.i : last ? last.i : -1;
  const showLegend = axis === 'right';
  const hasOverlayLegend = !!(S.ma || S.bb || S.vwap || S.ichimoku);

  // floating tooltip: right of the cursor when it fits, otherwise left of it (measured after render - its width depends on
  // the indicators shown)
  const tipRef = useRef<HTMLDivElement>(null);
  const toolsRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = tipRef.current;
    if (!el || !h) return;
    const w = el.offsetWidth;
    const right = h.x + 12;
    const left = right + w <= width ? right : Math.max(0, h.x - 12 - w);
    el.style.left = `${left}px`;
    // [2026-11-12] the drawing tools sit at the top right: a tooltip that reaches under them starts below them instead
    const tools = toolsRef.current;
    const box = ref.current;
    let top = 8;
    if (tools && box) {
      const tr = tools.getBoundingClientRect();
      const br = box.getBoundingClientRect();
      const toolsLeft = tr.left - br.left;
      const toolsBottom = tr.bottom - br.top;
      if (tr.width > 0 && left + w > toolsLeft - 4 && left < toolsLeft + tr.width + 4) top = toolsBottom + 4;
    }
    el.style.top = `${top}px`;
  });

  return (
    <div className={`price-chart ${interactive ? 'interactive' : ''}`} ref={ref} style={{ height }}>
      {model && !model.empty && width > 0 && (
        <svg className={tool !== 'pointer' ? 'drawing' : undefined} width={width} height={height} onMouseMove={onMove} onMouseDown={onDown} onMouseLeave={() => (setHov(null), setHoverY(null))} onDoubleClick={() => (mode === 'time' ? props.onZoom?.(null) : interactive && setView(null))}>
          <defs>
            <clipPath id={`pc-clip-${L}-${R}`}>
              <rect x={L} y={0} width={W} height={height} />
            </clipPath>
            <clipPath id={`pc-clip-price-${L}-${R}`}>
              <rect x={L} y={TOP} width={W} height={priceH} />
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
          <line x1={L} x2={L + W} y1={volTop + volH} y2={volTop + volH} className="pc-grid" />
          {showVol && !showLegend && (
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
            {/* [2026-11-12] back to front: context fills, reference lines, then the candles on top */}
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
            <path d={model.upW} className="pc-up-line" />
            <path d={model.dnW} className="pc-dn-line" />
            <path d={model.upB} className="pc-up" />
            <path d={model.dnB} className="pc-dn" />
            {showVol && <path d={model.upV} className="pc-up vol" />}
            {showVol && <path d={model.dnV} className="pc-dn vol" />}
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
              </g>
            ))}
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
                <text x={legendX} y={p.top + 12} className="pc-legend">
                  <tspan className="pc-legend-name">{p.label}</tspan>
                  {p.series.map((s) => (
                    <tspan key={s.name} className={`lg-${s.cls}`} dx={8}>
                      {s.name} {fmtNum(s.s[li], key2digits(p.key))}
                    </tspan>
                  ))}
                </text>
              )}
            </g>
          ))}
          {(showLegend || hasOverlayLegend) && (
            <Legends S={S} i={li} x={legendX} y={TOP + 12} volY={volTop + 11} vol={showLegend && showVol && li >= 0 ? bars[li]?.v : undefined} />
          )}

          {/* position + fills */}
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

          {/* HTS style: each indicator's own buy ▲ (red, under the line) / sell ▼ (blue, over the line) crossings (crossSignal) */}
          {model.panes.map((p) =>
            p.signals.map((g) => {
              const y = Math.min(p.top + p.h - 2, Math.max(p.top + 2, g.side === 'buy' ? g.y + 4 : g.y - 4));
              return (
                <path
                  key={`${p.key}-${g.i}`}
                  d={g.side === 'buy' ? `M${g.x},${y} l-4.5,8 h9 z` : `M${g.x},${y} l-4.5,-8 h9 z`}
                  className={g.side === 'buy' ? 'pc-buy-dot' : 'pc-sell-dot'}
                >
                  <title>{`${p.label} ${g.side === 'buy' ? '매수' : '매도'} 신호 · ${SIGNAL_RULES[p.key] || ''}`}</title>
                </path>
              );
            }),
          )}

          {/* user drawings: right-click a line to delete it */}
          {drawKey && (
            <g clipPath={`url(#pc-clip-price-${L}-${R})`}>
              {drawings.map((d) => {
                const del = (e: React.MouseEvent) => {
                  e.preventDefault();
                  removeDrawing(drawKey, d.id);
                };
                let x1: number, y1: number, x2: number, y2: number;
                let label = '';
                let vlabel = '';
                if (d.type === 'h') {
                  [x1, x2, y1, y2] = [L, L + W, model.sy(d.p), model.sy(d.p)];
                  label = hLineInfo(d.p, last?.b.c, buyPrice);
                } else if (d.type === 'v') {
                  const x = model.xOfT(d.t);
                  [x1, x2, y1, y2] = [x, x, TOP, TOP + priceH];
                  vlabel = vLineInfo(d.t, barAtTime(bars, d.t), intraday);
                } else [x1, y1, x2, y2] = [model.xOfT(d.t1), model.sy(d.p1), model.xOfT(d.t2), model.sy(d.p2)];
                return (
                  <g key={d.id} className="pc-draw" onContextMenu={del}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2} className="pc-draw-hit" />
                    <line x1={x1} y1={y1} x2={x2} y2={y2} className="pc-draw-line" />
                    {label && (
                      <text x={L + W - 4} y={y1 - 3} className="pc-draw-label" textAnchor="end">
                        {label}
                      </text>
                    )}
                    {vlabel && (
                      <text x={x1 + 4} y={TOP + 12} className="pc-draw-label">
                        {vlabel}
                      </text>
                    )}
                    <title>{d.type === 'h' ? `가로선 ${label}` : d.type === 'v' ? `세로선 ${vlabel}` : `추세선 ${fmtInt(d.p1)} → ${fmtInt(d.p2)}`} · 우클릭 = 삭제</title>
                  </g>
                );
              })}
              {pending && ghost && (
                <line x1={model.xOfT(pending.t)} y1={model.sy(pending.p)} x2={ghost.x} y2={ghost.y} className="pc-draw-line ghost" />
              )}
            </g>
          )}

          {/* last price tag (right axis) */}
          {axis === 'right' && last && (
            <PriceTag x={L + W + 2} y={model.sy(last.b.c)} text={fmtInt(last.b.c)} cls={last.b.c >= last.b.o ? 'up' : 'dn'} />
          )}

          {/* drag selection (released = zoom into it) */}
          {sel && Math.abs(sel.x1 - sel.x0) >= 2 && (
            <rect x={Math.min(sel.x0, sel.x1)} y={TOP} width={Math.abs(sel.x1 - sel.x0)} height={bottom - TOP} className="pc-sel" pointerEvents="none" />
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

          {/* [2026-11-08] resize handles: bottom edge of the price + volume area and of each indicator pane */}
          {onResizePane &&
            [
              { key: 'price' as const, y: volTop + volH + GAP / 2, cur: priceArea, min: PRICE_AREA_MIN, max: PRICE_AREA_MAX, label: '주가' },
              ...model.panes.map((p) => ({ key: p.key, y: p.top + p.h + GAP / 2, cur: p.h, min: OSC_PANE_MIN, max: OSC_PANE_MAX, label: p.label })),
            ].map((g) => (
              <g key={g.key} className="pc-resize" onMouseDown={(e) => startHeightDrag(e, g.cur, g.min, g.max, (hh) => onResizePane(g.key, hh))} onDoubleClick={(e) => (e.stopPropagation(), onResizePane(g.key, undefined))}>
                <title>{`${g.label} 영역 높이: 위아래로 끌어 조절 · 더블클릭: 기본 높이`}</title>
                <rect x={L} y={g.y - 4} width={W} height={8} className="pc-resize-hit" />
                <line x1={L} x2={L + W} y1={g.y} y2={g.y} className="pc-resize-line" />
              </g>
            ))}
        </svg>
      )}
      {drawKey && (
        <div className="draw-tools" ref={toolsRef} onMouseDown={(e) => e.stopPropagation()}>
          {(
            [
              ['pointer', '↖', '선택 (그리기 끝)'],
              ['h', '―', '가로선: 가격 칸을 클릭 (여러 개 연속, ↖ 또는 Esc로 끝)'],
              ['v', '│', '세로선: 가격 칸을 클릭 (여러 개 연속, ↖ 또는 Esc로 끝)'],
              ['t', '╱', '추세선: 시작점과 끝점을 차례로 클릭 (여러 개 연속)'],
            ] as [DrawTool, string, string][]
          ).map(([k, icon, title]) => (
            <button key={k} className={tool === k ? 'on' : ''} title={title} onClick={() => (setTool(k), setPending(null))}>
              {icon}
            </button>
          ))}
          <button
            title="이 종목에 그린 선을 모두 지웁니다 (하나만 지우려면 선을 우클릭)"
            disabled={!drawings.length}
            onClick={() => (clearDrawings(drawKey), setPending(null))}
          >
            지우기
          </button>
        </div>
      )}
      {tooltip && h && (
        <div ref={tipRef} className="tip pc-tip" style={{ left: Math.max(h.x + 12, 0), top: 8 }}>
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

/** the bar holding time t (last bar starting at or before it) */
function barAtTime(bars: Bar[], t: number): Bar | undefined {
  let lo = 0;
  let hi = bars.length - 1;
  let found: Bar | undefined;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].t <= t) (found = bars[mid]), (lo = mid + 1);
    else hi = mid - 1;
  }
  return found;
}

const signedPct = (a: number, b: number) => {
  const v = ((a - b) / b) * 100;
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
};

/** horizontal line: "22,450 · 현재가 대비 +1.20% · 평단 대비 -0.50%" */
function hLineInfo(p: number, lastClose: number | undefined, buyPrice: number | undefined): string {
  const parts = [fmtInt(p)];
  if (lastClose && lastClose > 0) parts.push(`최근가 ${fmtInt(lastClose)} 대비 ${signedPct(p, lastClose)}`);
  if (buyPrice && buyPrice > 0) parts.push(`평단 대비 ${signedPct(p, buyPrice)}`);
  return parts.join(' · ');
}

/** vertical line: "11:30:00 · 시 21,600 고 21,750 저 21,600 종 21,650 · 량 964" (daily kinds: the date) */
function vLineInfo(t: number, b: Bar | undefined, intraday: boolean): string {
  const when = intraday ? fmtKstTime(t) : fmtKstDateTime(t).slice(0, 10);
  if (!b) return when;
  return `${when} · 시 ${fmtInt(b.o)} 고 ${fmtInt(b.h)} 저 ${fmtInt(b.l)} 종 ${fmtInt(b.c)} · 량 ${fmtInt(b.v)}`;
}

const key2digits = (k: IndicatorKey) => (k === 'obv' ? 0 : k === 'macd' ? 2 : 1);

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

/** price-pane legend (right-axis layout): MA / BB / VWAP / Ichimoku values at bar i */
function Legends({ S, i, x, y, volY, vol }: { S: IndicatorSet; i: number; x: number; y: number; volY: number; vol?: number }) {
  const rows: Legend[][] = [];
  if (S.ma) rows.push(MA_PERIODS.map((p) => ({ label: `MA${p}`, color: MA_COLORS[p], value: S.ma![p][i], digits: 0 })));
  const r2: Legend[] = [];
  const C = OVERLAY_COLOR;
  if (S.bb) r2.push({ label: `볼린저(${BOT_PARAMS.bbPeriod},${BOT_PARAMS.bbMult}) 상`, color: C.bb, value: S.bb.upper[i], digits: 0 }, { label: '중', color: C.bb, value: S.bb.mid[i], digits: 0 }, { label: '하', color: C.bb, value: S.bb.lower[i], digits: 0 });
  if (S.vwap) r2.push({ label: 'VWAP', color: C.vwap, value: S.vwap[i], digits: 0 });
  if (r2.length) rows.push(r2);
  if (S.ichimoku) {
    const k = S.ichimoku;
    rows.push([
      { label: '전환', color: C.tenkan, value: k.tenkan[i], digits: 0 },
      { label: '기준', color: C.kijun, value: k.kijun[i], digits: 0 },
      { label: '구름 A', value: k.senkouA[i], digits: 0 },
      { label: 'B', value: k.senkouB[i], digits: 0 },
    ]);
  }
  return (
    <g pointerEvents="none">
      {rows.map((row, j) => (
        <text key={j} x={x} y={y + j * 14} className="pc-legend pc-legend-halo">
          {row.map((l, k) => (
            <tspan key={k} dx={k ? 9 : 0}>
              {l.color && <tspan style={{ fill: l.color }}>■ </tspan>}
              {l.label} <tspan className="pc-legend-val">{fmtNum(l.value, l.digits ?? 1)}</tspan>
            </tspan>
          ))}
        </text>
      ))}
      {vol !== undefined && (
        <text x={x} y={volY} className="pc-legend">
          거래량 {fmtInt(vol)}
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
    </>
  );
}
