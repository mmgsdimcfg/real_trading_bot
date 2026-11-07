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
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { samePeriod, type Bar, type ChartKind } from '../../../shared/price';
import { BOT_PARAMS, computeIndicators, MA_PERIODS, OSCILLATORS, type IndicatorKey, type IndicatorSet, type Series } from '../../../shared/indicators';
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
const VOL_RATIO = 0.2;
const KST = 9 * 3600000;
const MIN_BARS = 10;

export const MA_COLORS: Record<number, string> = { 5: '#e0559b', 20: '#e08a1e', 60: '#2aa1c9', 120: '#0ca30c' };

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
  /** index mode: wheel zoom + drag pan */
  interactive?: boolean;
  /** floating OHLC/indicator tooltip (off when the page shows its own data panel) */
  tooltip?: boolean;
  /** hovered bar index (null = none) */
  onHover?: (i: number | null) => void;
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
  const { bars, warm = [], warmBot = [], kind, n, indicators, trades, buyPrice, date, xTicks = [], interactive = false, tooltip = true, onHover } = props;
  const intraday = kind === 'tick' || kind === 'min';
  const mode = props.mode ?? (intraday ? 'time' : 'index');
  const axis = props.axis ?? 'left';
  const oscH = props.oscHeight ?? OSC_PANE_H;
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
  const mainH = height - osc.length * (oscH + GAP) - TOP - AXIS_H;
  const priceH = mainH * (1 - VOL_RATIO) - GAP;
  const volTop = TOP + priceH + GAP;
  const volH = mainH * VOL_RATIO;
  const oscTop = (j: number) => volTop + volH + GAP + j * (oscH + GAP);
  const bottom = osc.length ? oscTop(osc.length - 1) + oscH : volTop + volH;
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
    if (S.bb) overlay.push({ cls: 'ind-bb', d: line(pts(S.bb.upper), sy) }, { cls: 'ind-bb', d: line(pts(S.bb.lower), sy) }, { cls: 'ind-bb-mid', d: line(pts(S.bb.mid), sy) });
    if (S.vwap) overlay.push({ cls: 'ind-vwap', d: line(pts(S.vwap), sy) });
    if (S.ichimoku) {
      const k = S.ichimoku;
      const cp = [...slots.map((sl) => ({ x: sl.x, a: k.senkouA[sl.i], b: k.senkouB[sl.i] })), ...futureIdx.map((f) => ({ x: f.x, a: k.senkouA[f.idx], b: k.senkouB[f.idx] }))];
      cloudPaths = cloud(cp, sy);
      overlay.push(
        { cls: 'ind-span-a', d: line(cp.map((p) => [p.x, p.a]), sy) },
        { cls: 'ind-span-b', d: line(cp.map((p) => [p.x, p.b]), sy) },
        { cls: 'ind-tenkan', d: line(pts(k.tenkan), sy) },
        { cls: 'ind-kijun', d: line(pts(k.kijun), sy) },
        { cls: 'ind-chikou', d: line(pts(k.chikou), sy) },
      );
    }

    // ---- oscillator panes ----
    const panes = osc.map((key, j) => {
      const cfg = paneOf(key, S);
      const top = oscTop(j);
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
      const y = (v: number) => top + ((vhi - v) / (vhi - vlo)) * oscH;
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
      return { key, top, y, lines, guides: cfg.guides, ticks, label: cfg.label, histUp, histDn, series: cfg.series };
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
    return { empty: false as const, slots, sy, py, lo, hi, upW, dnW, upB, dnB, upV, dnV, labels, highlight, marks, overlay, cloudPaths, panes, xOfTime, dayLines, step };
  }, [bars, S, indicators, kind, n, mode, props.domain, xTicks, trades, buyPrice, date, width, intraday, L, W, priceH, volTop, volH, oscH, osc.join(), vp, shift, defaultCount, interactive]); // eslint-disable-line react-hooks/exhaustive-deps

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
  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
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

  const h = model && !model.empty && hover !== null ? model.slots[hover] : null;
  const last = model && !model.empty ? model.slots[model.slots.length - 1] : null;
  const axisX = axis === 'left' ? L - 6 : L + W + 6;
  const anchorAxis = axis === 'left' ? 'end' : 'start';
  const legendX = axis === 'left' ? L + 4 : L + 4;
  const li = h ? h.i : last ? last.i : -1;
  const showLegend = axis === 'right';

  return (
    <div className={`price-chart ${interactive ? 'interactive' : ''}`} ref={ref} style={{ height }}>
      {model && !model.empty && width > 0 && (
        <svg width={width} height={height} onMouseMove={onMove} onMouseDown={onDown} onMouseLeave={() => (setHov(null), setHoverY(null))} onDoubleClick={() => interactive && setView(null)}>
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
          <line x1={L} x2={L + W} y1={volTop + volH} y2={volTop + volH} className="pc-grid" />
          {!showLegend && (
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
            {model.cloudPaths && (
              <>
                <path d={model.cloudPaths.up} className="ind-cloud-up" />
                <path d={model.cloudPaths.dn} className="ind-cloud-dn" />
              </>
            )}
            <path d={model.upW} className="pc-up-line" />
            <path d={model.dnW} className="pc-dn-line" />
            <path d={model.upB} className="pc-up" />
            <path d={model.dnB} className="pc-dn" />
            <path d={model.upV} className="pc-up vol" />
            <path d={model.dnV} className="pc-dn vol" />
            {model.overlay.map((o, i) => (
              <path key={i} d={o.d} className={`ind-line ${o.cls || ''}`} style={o.color ? { stroke: o.color } : undefined} />
            ))}
            {model.panes.map((p) => (
              <g key={p.key}>
                <rect x={L} y={p.top} width={W} height={oscH} className="pc-pane" />
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
          {showLegend && <Legends S={S} i={li} x={legendX} y={TOP + 12} volY={volTop + 11} vol={li >= 0 ? bars[li]?.v : undefined} />}

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
  if (S.bb) r2.push({ label: `BB(${BOT_PARAMS.bbPeriod},${BOT_PARAMS.bbMult}) 상`, value: S.bb.upper[i], digits: 0 }, { label: '중', value: S.bb.mid[i], digits: 0 }, { label: '하', value: S.bb.lower[i], digits: 0 });
  if (S.vwap) r2.push({ label: 'VWAP', value: S.vwap[i], digits: 0 });
  if (r2.length) rows.push(r2);
  if (S.ichimoku) {
    const k = S.ichimoku;
    rows.push([
      { label: '전환', value: k.tenkan[i], digits: 0 },
      { label: '기준', value: k.kijun[i], digits: 0 },
      { label: '선행A', value: k.senkouA[i], digits: 0 },
      { label: '선행B', value: k.senkouB[i], digits: 0 },
    ]);
  }
  return (
    <g pointerEvents="none">
      {rows.map((row, j) => (
        <text key={j} x={x} y={y + j * 14} className="pc-legend">
          {row.map((l, k) => (
            <tspan key={k} dx={k ? 8 : 0} style={l.color ? { fill: l.color } : undefined}>
              {l.label} {fmtNum(l.value, l.digits ?? 1)}
            </tspan>
          ))}
        </text>
      ))}
      <text x={x} y={volY} className="pc-legend">
        거래량 {fmtInt(vol)}
      </text>
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
