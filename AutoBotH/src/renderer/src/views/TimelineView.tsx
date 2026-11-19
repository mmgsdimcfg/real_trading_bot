/*
 * Update log
 * - [2026-11-04] type=feat - Timeline of the selected symbol: x = KST time, y = depth reached in the buy pipeline (condition order),
 *   dot color/shape = result, buy/sell fills as vertical markers; hovering a dot shows the raw log line. Sell evaluations get their
 *   own strip below (one y-scale per chart)
 * - [2026-11-05] type=feat - Sell strip: new bottom row '평가 제외' (condKey 'PREP') for [HOLD SKIP] status dots - positions whose sell
 *   conditions are skipped (e.g. NOT_TODAY_BUY_POSITION); before this, such symbols showed an empty timeline
 * - [2026-11-05] type=feat - Price chart between the buy and sell evaluation charts (PriceChart): tick bars (N ticks), minute bars
 *   (M minutes), daily / weekly / monthly bars. Tick/minute bars use the buy chart's time domain and ticks so candles line up with
 *   the evaluation dots; D/W/M end at the viewed day. Reloaded with every snapshot while the day is live
 * - [2026-11-06] type=feat - Indicator toggles (볼린저 / 일목 / RSI / MACD / ADX, bot periods in the tooltips); tick/minute requests ask
 *   for the previous trading day's bars as indicator warm-up while any indicator is on; the chart box grows with the panes
 * - [2026-11-06] type=feat - All indicators of the chart tab available here too (shared IndicatorToggles); the bot's warm-up window
 *   (warmBot) is passed so the bot's indicators match r003
 * - [2026-11-07] type=feat - Past-day replay: tick/minute bars that start after the replay time are hidden (barsUpTo); the
 *   evaluation dots and trades already come cut from the replay snapshot. The time axis ends at the replay time, so it keeps
 *   moving through stretches without events (r003 stopped)
 * - [2026-11-07] type=fix - The time axis also covers the day's tick / minute bars (it ended at the symbol's last evaluation, so
 *   later bars could not be seen); "보기 범위" (전체 / 2시간 / 1시간 / 30분) + a slider to move the window left and right, shared by
 *   the buy-evaluation and price charts
 * - [2026-11-07] type=feat - Sell fill markers show the sell price and return ("매도 22,450 (+1.43%) <reason>"), like the buy price
 * - [2026-11-07] type=feat - Price chart twice as tall (TL_SCALE: price + volume area 300 -> 600 px, indicator panes 84 -> 168 px)
 * - [2026-11-07] type=feat - Buy chart y-axis rows can be hidden ("y축 항목" popover: per-row checkboxes, 모두 표시, 이 종목 평가된 항목만);
 *   hidden rows and their dots are left out, the remaining rows close up and the chart gets shorter. Kept per PC (localStorage)
 * - [2026-11-08] type=feat - Heights set by dragging (store layout, kept per PC): grip bars under the buy / sell evaluation charts,
 *   handles on the bottom edge of the price area and of each indicator pane (PriceChart onResizePane); double-click = default size.
 *   The y-axis row list follows the evaluation order, which the server numbers _001.._025 since 2026-11-08
 * - [2026-11-08] type=feat - Sell chart gets the same "y축 항목" picker (YPick: check boxes, 모두 표시, 이 종목 평가된 항목만 - sell
 *   events and sell fills); hidden rows close up and the default height follows the row count
 * - [2026-11-08] type=feat - "이 종목 평가된 항목만" is a checkbox (kept per PC): while checked, the shown rows follow the selected
 *   symbol and change with it (the old button picked the rows once, for the symbol selected at the click). "모두 표시" or
 *   changing a row by hand unchecks it and continues from the rows shown at that moment
 * - [2026-11-08] type=merge - Ubuntu (YPick shared by both charts, sell picker, heights) + Windows ("이 종목 평가된 항목만" checkbox
 *   that follows the selected symbol): the checkbox mode is now useRowPick and works on the buy and the sell chart
 * - [2026-11-11] type=feat - Hover details of the buy / sell evaluation charts move to a dock (TipDock) with the reason in Korean
 *   (shared/reasonKo) + the code + the raw line; the floating tooltip no longer covers the dots
 * - [2026-11-11] type=feat - The dock sits under its evaluation chart (between it and the price chart) as a full-width strip, and the
 *   hovered evaluation time drives the price chart's crosshair (PriceChart hoverTime) so both charts stay in sync
 * - [2026-11-12] type=feat - (1) Clicking a dot of the buy / sell evaluation chart pins it: the dock keeps showing that dot
 *   (marked 고정) after the mouse leaves, a line marks its time, a click on empty space or the ✕ releases it; hovering another
 *   dot shows that one while the mouse is on it (2) the crosshair is linked both ways: hovering the price chart (tick / minute
 *   bars) draws a vertical line at that bar's time on the buy and sell evaluation charts - before, only the evaluation charts
 *   drove the price chart
 * - [2026-11-12] type=feat - zoom: the price chart's wheel / drag selection sets a time span that all three charts follow
 *   (전체 보기 or a double click on the price chart resets it); time ticks adapt to the span (30 min ... 1 min); the buy
 *   evaluation chart is as tall as its rows need (one row no longer takes 200 px)
 * - [2026-11-16] type=feat - Symbols r003 does not quote (holdings outside the watch list) show the KIS price the store asks for
 *   (store quotes, source 'kis') - live day only
 * - [2026-11-16] type=feat - Quote line next to the symbol: price / change / rate + open, high, low, volume, traded value (PriceTag detail)
 * - [2026-11-16] type=feat - The time axis spans the whole trading session of the day from the start (NXT symbol 08:00 - 20:00,
 *   KRX 09:00 - 15:30): at the open the buy evaluation and price charts no longer stretch the first minutes over the full width,
 *   they fill in from the left at the scale of the finished day. Events / bars outside the session still widen the axis
 * - [2026-11-17] type=fix - Hover / click on the buy and sell evaluation charts picks the dot nearest the cursor (pixel distance
 *   with the drawn axis scales, within 18 px) - recharts' activePayload named another row's dot at the same time (cursor on _022,
 *   dock showed _020). The hovered dot gets a ring, a pinned dot an accent ring, and a pinned dot stays in the dock while the
 *   mouse moves over other dots (click another dot to re-pin, empty space or ✕ to release)
 * - [2026-11-18] type=feat - One time axis for every symbol: the frame is always 08:00 - 20:00 (the NXT day), so switching symbols
 *   keeps the same x positions. A KRX-only symbol shows its evaluations and price bars of the KRX session (09:00 - 15:30) only -
 *   r003's pre-open SKIP rows (can_trade_code_now=False) and KRX pre-open prints are left out
 * - [2026-11-18] type=perf - The price chart of a symbol / chart size seen before is drawn at once from the renderer's memory
 *   (dataCache, key = day/code/kind/size/warm-up/NXT) and refreshed in the background - a symbol switch no longer waits for the
 *   main process / server before drawing
 * - [2026-11-19] type=perf - Price request / dataCache key from prefetch.ts (timelinePriceFetch): the background fetch keeps every
 *   watched symbol's bars and events fresh, so opening the tab or switching symbols draws current data at once
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { timelinePriceFetch } from '../prefetch';
import { PriceTag } from '../components/PriceTag';
import { NxtMark } from '../components/NxtTag';
import { CartesianGrid, Customized, ReferenceDot, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { reasonKo } from '../../../shared/reasonKo';
import { useStore } from '../store';
import { SymbolPicker } from '../components/SymbolPicker';
import { condTitle, fmtInt, fmtKrw, sortedConds } from '../lib';
import { condKeyOf } from '../../../shared/mapping';
import { fmtKstTime, tradingSessionRange } from '../../../shared/time';
import type { EvalEvent, Trade } from '../../../shared/types';
import { MIN_SIZES, TICK_SIZES, type ChartKind, type PriceResponse } from '../../../shared/price';
import { OSC_PANE_H, PRICE_PANE_H, PriceChart, chartHeight } from '../components/PriceChart';
import { HeightGrip } from '../components/HeightGrip';

/** the timeline's price chart is twice the default size (price + volume area and every indicator pane) */
const TL_SCALE = 2;
const TL_OSC_H = OSC_PANE_H * TL_SCALE;
const TL_PRICE_H = PRICE_PANE_H * TL_SCALE;
/** buy / sell evaluation chart heights: default of the sell chart, drag limits of both */
const SELL_DEFAULT_H = 300;
const GRIP_MIN = 120;
const GRIP_MAX = 1600;
import { IndicatorToggles } from '../components/IndicatorToggles';
import { barsUpTo } from '../../../shared/replay';
import { peek, put } from '../dataCache';

const KIND_LABEL: [ChartKind, string][] = [
  ['tick', '틱봉'],
  ['min', '분봉'],
  ['day', '일봉'],
  ['week', '주봉'],
  ['month', '월봉'],
];
const SOURCE_LABEL: Record<string, string> = { ticks: '체결 틱', min1: '1분봉 (10초 파일)', daily: '일봉 파일' };

interface Pt {
  x: number;
  y: number;
  ev: EvalEvent;
}

/** [2026-11-17] pixel scales of an evaluation chart (captured from recharts while it draws) */
type Scales = { x: (v: number) => number; y: (v: number) => number };
/** a dot is picked only when the cursor is within this many pixels of it */
const PICK_R = 18;

/** recharts <Customized> renderer that only records the axis scales of the chart into `ref` (draws nothing) */
function scaleGrabber(ref: { current: Scales | null }) {
  return (props: unknown) => {
    const p = props as { xAxisMap?: Record<string, { scale?: (v: number) => number }>; yAxisMap?: Record<string, { scale?: (v: number) => number }> };
    const xa = Object.values(p.xAxisMap || {})[0];
    const ya = Object.values(p.yAxisMap || {})[0];
    if (xa?.scale && ya?.scale) ref.current = { x: xa.scale, y: ya.scale };
    return null;
  };
}

/**
 * [2026-11-17] The dot under the cursor. recharts' activePayload of a multi-series ScatterChart is not the dot under the mouse
 * (it could name a dot of another row at the same time - e.g. _020 while the cursor was on _022), so the nearest shown dot in
 * pixel space is chosen here; none within PICK_R -> null.
 */
function nearestPt(e: unknown, pts: Pt[], scales: Scales | null): Pt | null {
  const m = e as { chartX?: number; chartY?: number } | null;
  if (!m || m.chartX == null || m.chartY == null || !scales) return null;
  let best: Pt | null = null;
  let bestD = PICK_R * PICK_R;
  for (const p of pts) {
    const dx = scales.x(p.x) - m.chartX;
    const dy = scales.y(p.y) - m.chartY;
    const d = dx * dx + dy * dy;
    if (d <= bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}

const SERIES: { key: EvalEvent['result']; label: string; color: string; shape: 'circle' | 'cross' | 'diamond' | 'square' }[] = [
  { key: 'fail', label: '✕ 막힘', color: 'var(--st-fail)', shape: 'cross' },
  { key: 'hold', label: '… 대기', color: 'var(--st-hold)', shape: 'diamond' },
  { key: 'pass', label: '✓ 통과/주문', color: 'var(--st-pass)', shape: 'circle' },
  { key: 'info', label: '· 정보', color: 'var(--st-idle)', shape: 'square' },
];

/**
 * Rows shown on an evaluation chart. Normal mode: the user's hidden list. "이 종목 평가된 항목만" (evalOnly): the hidden rows follow the
 * selected symbol (rows without an evaluation for it), recomputed on every symbol change; until the new symbol's timeline arrives
 * (ready=false) the previous rows stay, so the chart does not flash all rows. evalOnly on a symbol without evaluations shows no rows.
 */
function useRowPick(rows: Row[], hidden: string[], evalOnly: boolean, evaluated: Set<string>, ready: boolean) {
  const evalHiddenRef = useRef<string[]>([]);
  const hiddenKeys = useMemo(() => {
    if (!evalOnly) return hidden;
    if (ready) evalHiddenRef.current = rows.filter((r) => !evaluated.has(r.key)).map((r) => r.key);
    return evalHiddenRef.current;
  }, [evalOnly, hidden, ready, rows, evaluated]);
  const shownRows = useMemo(() => {
    const hide = new Set(hiddenKeys);
    const shown = rows.filter((r) => !hide.has(r.key));
    return shown.length || evalOnly ? shown : rows;
  }, [rows, hiddenKeys, evalOnly]);
  return { hiddenKeys, shownRows };
}

type Row = { key: string; label: string };

/** "y축 항목" popover of an evaluation chart: per-row check boxes, 모두 표시, 이 종목 평가된 항목만 (checkbox that follows the selected
 *  symbol); "모두 표시" or changing a row by hand ends that mode and continues from the rows shown now. List top-down = chart top-down */
function YPick(props: {
  side: string;
  rows: Row[];
  shown: number;
  /** rows hidden right now (useRowPick hiddenKeys) */
  hidden: string[];
  setHidden: (keys: string[]) => void;
  evalOnly: boolean;
  setEvalOnly: (on: boolean) => void;
  evaluated: Set<string>;
}) {
  const { side, rows, shown, hidden, setHidden, evalOnly, setEvalOnly, evaluated } = props;
  return (
    <>
      <details className="ypick">
        <summary>
          y축 항목 ({shown}/{rows.length}) ▾
        </summary>
        <div className="ypick-panel">
          <div className="actions">
            <button
              className="mini"
              onClick={() => {
                setEvalOnly(false);
                setHidden([]);
              }}
            >
              모두 표시
            </button>
            <label className="check" title={`체크하면 종목을 바꿀 때마다 그 종목의 ${side} 평가 로그에 나온 항목만 표시합니다`}>
              <input type="checkbox" checked={evalOnly} onChange={(e) => setEvalOnly(e.target.checked)} />
              <span>이 종목 평가된 항목만</span>
            </label>
          </div>
          {[...rows].reverse().map((r) => (
            <label key={r.key} className="check ypick-row">
              <input
                type="checkbox"
                checked={!hidden.includes(r.key)}
                onChange={(e) => {
                  // picking rows by hand ends the per-symbol mode and starts from the rows shown now
                  setEvalOnly(false);
                  setHidden(e.target.checked ? hidden.filter((k) => k !== r.key) : [...hidden, r.key]);
                }}
              />
              <span className={evaluated.has(r.key) ? '' : 'muted'}>{r.label}</span>
            </label>
          ))}
          <p className="muted small">숨긴 항목의 점은 표시하지 않습니다. 흐린 글자 = 이 종목에서 평가 로그가 없는 항목.</p>
        </div>
      </details>
      {evalOnly && shown === 0 && <span className="muted small">이 종목은 {side} 평가 기록이 없습니다.</span>}
    </>
  );
}

/** [2026-11-11] hover details are shown in a dock beside the chart (TipDock) - a floating tooltip covered the dots around the cursor */
function TipDock({ pt, side, pinned, onUnpin }: { pt: Pt | null; side: '매수' | '매도'; pinned?: boolean; onUnpin?: () => void }) {
  const conditions = useStore((s) => s.conditions);
  if (!pt) {
    return (
      <div className="tip-dock">
        <div className="muted small">{side} 평가 차트의 점 위에 마우스를 올리면 여기에 내용이 나옵니다. 점을 누르면 고정됩니다.</div>
      </div>
    );
  }
  const ev = pt.ev;
  return (
    <div className={`tip-dock ${pinned ? 'pinned' : ''}`}>
      {/* [2026-11-17] one summary line (time, condition, Korean reason, reason code), the raw log line below it */}
      <div className="tip-line">
        {pinned && (
          <button className="pin-chip" onClick={onUnpin} title="고정 해제 (차트의 빈 곳을 눌러도 해제)">
            고정 ✕
          </button>
        )}
        <span>
          <b>{fmtKstTime(ev.ts)}</b> [{ev.tag}] {condTitle(conditions, ev.side, ev.condKey)}
          {ev.sub ? ` / ${ev.sub}` : ''}
        </span>
        <span className="reason-ko">{reasonKo(ev.reason) || ev.reason}</span>
        <span className="muted small mono">{ev.reason}</span>
      </div>
      <pre className="raw">{ev.raw}</pre>
    </div>
  );
}

/** stable empty list (memo inputs) */
const NO_EVENTS: EvalEvent[] = [];

export function TimelineView() {
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const selected = useStore((s) => s.selected);
  const timeline = useStore((s) => s.timeline);
  const status = useStore((s) => s.status);
  const chart = useStore((s) => s.chart);
  const setChart = useStore((s) => s.setChart);
  const [price, setPrice] = useState<PriceResponse | null>(null);
  const [priceLoading, setPriceLoading] = useState(false);
  const reqSeq = useRef(0);
  /** hovered dots of the buy / sell charts (shown in the docks beside them) */
  const [hovBuy, setHovBuy] = useState<Pt | null>(null);
  const [hovSell, setHovSell] = useState<Pt | null>(null);
  /** [2026-11-12] dots pinned by a click (shown in the dock while nothing is hovered) */
  const [pinBuy, setPinBuy] = useState<Pt | null>(null);
  const [pinSell, setPinSell] = useState<Pt | null>(null);
  /** [2026-11-12] time of the bar hovered on the price chart (tick / minute bars) - a vertical line on the evaluation charts */
  const [priceHovT, setPriceHovT] = useState<number | null>(null);
  // [2026-11-17] axis scales of the buy / sell evaluation charts (nearestPt picks the dot under the cursor with them)
  const buyScales = useRef<Scales | null>(null);
  const sellScales = useRef<Scales | null>(null);
  const grabBuyScales = useMemo(() => scaleGrabber(buyScales), []);
  const grabSellScales = useMemo(() => scaleGrabber(sellScales), []);

  const buyConds = useMemo(() => sortedConds(conditions, 'buy'), [conditions]);
  const sellConds = useMemo(() => sortedConds(conditions, 'sell'), [conditions]);
  const buyHidden = useStore((s) => s.buyHidden);
  const setBuyHidden = useStore((s) => s.setBuyHidden);
  const layout = useStore((s) => s.layout);
  const setLayout = useStore((s) => s.setLayout);
  /** every buy row bottom-up: 준비, conditions in evaluation order, 주문 */
  const buyRows = useMemo(
    () => [
      { key: 'PREP', label: '준비' },
      ...buyConds.map((c) => ({ key: condKeyOf(c.no), label: condKeyOf(c.no) + ' ' + c.title.slice(0, 10) })),
      { key: 'ORDER', label: '주문' },
    ],
    [buyConds],
  );
  const buyEvalOnly = useStore((s) => s.buyEvalOnly);
  const setBuyEvalOnly = useStore((s) => s.setBuyEvalOnly);
  const sellHidden = useStore((s) => s.sellHidden);
  const setSellHidden = useStore((s) => s.setSellHidden);
  const sellEvalOnly = useStore((s) => s.sellEvalOnly);
  const setSellEvalOnly = useStore((s) => s.setSellEvalOnly);
  /** every sell row bottom-up: 평가 제외 ('PREP': [HOLD SKIP], e.g. not bought today), conditions, 세션 종료, 주문 */
  const sellRows = useMemo(
    () => [
      { key: 'PREP', label: '평가 제외' },
      ...sellConds.map((c) => ({ key: condKeyOf(c.no), label: condKeyOf(c.no) + ' ' + c.title.slice(0, 10) })),
      { key: 'SESSION_CLOSE', label: '세션 종료' },
      { key: 'ORDER', label: '주문' },
    ],
    [sellConds],
  );

  // [2026-11-18] a KRX-only symbol: only what falls in the KRX session (09:00 - 15:30) is drawn
  // [2026-11-19] ... 09:00 - 20:00 (KRX after-hours market); NXT = the day's WATCH line or the server's flag on the price answer
  // (holdings outside the watchlist have no WATCH line)
  const selNxt = !!snap?.symbols.find((s) => s.code === selected)?.nxt || (price?.ok === true && price.nxt === true);
  const krxRange = selNxt ? null : tradingSessionRange(snap?.date || status.date, false);
  const inShown = (t: number) => !krxRange || (t >= krxRange[0] && t <= krxRange[1]);
  const allEvents = timeline && timeline.code === selected ? timeline.events : NO_EVENTS;
  const events = useMemo(() => (krxRange ? allEvents.filter((e) => inShown(e.ts)) : allEvents), [allEvents, krxRange?.[0]]); // eslint-disable-line react-hooks/exhaustive-deps
  const trades = timeline && timeline.code === selected ? timeline.trades : [];
  const buyKeysWithEvents = useMemo(() => new Set(events.filter((e) => e.side === 'buy').map((e) => e.condKey)), [events]);
  // sell rows with evaluations of this symbol: sell events + the condition of each sell fill
  const sellKeysWithEvents = useMemo(
    () => new Set([...events.filter((e) => e.side === 'sell').map((e) => e.condKey), ...trades.filter((t) => t.side === 'SELL' && t.condKey).map((t) => t.condKey as string)]),
    [events, trades],
  );
  const timelineReady = !!timeline && timeline.code === selected;
  const { hiddenKeys, shownRows } = useRowPick(buyRows, buyHidden, buyEvalOnly, buyKeysWithEvents, timelineReady);
  const { hiddenKeys: sellHiddenKeys, shownRows: shownSellRows } = useRowPick(sellRows, sellHidden, sellEvalOnly, sellKeysWithEvents, timelineReady);
  const buyY = useMemo(() => new Map(shownRows.map((r, i) => [r.key, i])), [shownRows]);
  const sellY = useMemo(() => new Map(shownSellRows.map((r, i) => [r.key, i])), [shownSellRows]);
  const buyPts = useMemo(() => events.filter((e) => e.side === 'buy' && buyY.has(e.condKey)).map((ev) => ({ x: ev.ts, y: buyY.get(ev.condKey)!, ev })), [events, buyY]);
  const sellPts = useMemo(() => events.filter((e) => e.side === 'sell' && sellY.has(e.condKey)).map((ev) => ({ x: ev.ts, y: sellY.get(ev.condKey)!, ev })), [events, sellY]);
  // per-series arrays kept stable: the charts re-render with every hovered bar of the price chart
  const buySeries = useMemo(() => SERIES.map((sr) => ({ ...sr, data: buyPts.filter((p) => p.ev.result === sr.key) })), [buyPts]);
  const sellSeries = useMemo(
    () => ({ pass: sellPts.filter((p) => p.ev.result === 'pass'), hold: sellPts.filter((p) => p.ev.result === 'hold'), info: sellPts.filter((p) => p.ev.result === 'info') }),
    [sellPts],
  );
  /** [2026-11-17] the sell dots actually drawn (the three series above) - what the cursor can pick */
  const sellShownPts = useMemo(() => [...sellSeries.pass, ...sellSeries.hold, ...sellSeries.info], [sellSeries]);
  // a pinned dot belongs to one symbol's events: another symbol (or a reloaded day) releases it
  useEffect(() => {
    setPinBuy(null);
    setPinSell(null);
    setPriceHovT(null);
  }, [selected, snap?.date]);
  const replayAt = snap?.replay?.cursor ?? null;
  // tick / minute bars of the day widen the axis too (evaluations may stop before the close, e.g. r003 stopped at 13:40)
  const intradayKind = chart.kind === 'tick' || chart.kind === 'min';
  const keptBars = useMemo(() => (price?.ok ? (krxRange && intradayKind ? price.bars.filter((b) => inShown(b.t)) : price.bars) : []), [price, krxRange?.[0], intradayKind]); // eslint-disable-line react-hooks/exhaustive-deps
  const barTs = intradayKind && keptBars.length ? [keptBars[0].t, keptBars[keptBars.length - 1].t2 ?? keptBars[keptBars.length - 1].t] : [];
  const xs = [...events.map((e) => e.ts), ...trades.map((t) => t.ts), ...barTs].filter(Number.isFinite);
  // [2026-11-16] the axis spans the day's whole trading session from the start, so the charts fill in from the left at the final
  // scale instead of stretching the first minutes over the whole width; events or bars outside it widen it.
  // [2026-11-18] the same 08:00 - 20:00 frame for every symbol (KRX-only too) - the x axis does not move between symbols
  const session = tradingSessionRange(snap?.date || status.date, true);
  const full: [number, number] = session
    ? [Math.min(session[0], ...xs), Math.max(session[1], ...xs)]
    : xs.length
      ? [Math.min(...xs), replayAt !== null ? Math.max(replayAt, Math.min(...xs) + 60000) : Math.max(...xs)]
      : replayAt !== null
        ? [replayAt - 30 * 60000, replayAt]
        : [0, 1];
  // visible window: whole range or a span moved with the slider (both charts follow it)
  const [span, setSpan] = useState(0);
  const [startPct, setStartPct] = useState(100);
  const fullLen = full[1] - full[0];
  const win = span > 0 && span < fullLen ? span : 0;
  const winStart = win ? full[0] + ((fullLen - win) * startPct) / 100 : full[0];
  /** [2026-11-12] span picked on the price chart (wheel / drag selection); null = the range chosen above */
  const [zoom, setZoom] = useState<[number, number] | null>(null);
  useEffect(() => setZoom(null), [selected, snap?.date, span, chart.kind]);
  const zoomOk = zoom && intradayKind && zoom[1] > zoom[0] && zoom[0] < full[1] && zoom[1] > full[0];
  const domain: [number, number] = zoomOk ? zoom : win ? [winStart, winStart + win] : full;
  // explicit ticks (recharts' automatic ticks on a time scale bunch up at the start); the step follows the visible span
  const xTicks = useMemo(() => {
    const len = (domain[1] - domain[0]) / 60000;
    const step = (len > 180 ? 30 : len > 90 ? 15 : len > 40 ? 10 : len > 15 ? 5 : len > 6 ? 2 : 1) * 60 * 1000;
    const out: number[] = [];
    for (let t = Math.ceil(domain[0] / step) * step; t <= domain[1]; t += step) out.push(t);
    return out;
  }, [domain[0], domain[1]]);
  const sym = snap?.symbols.find((s) => s.code === selected);

  // price bars: on symbol/day/chart change, and with every snapshot while the day is live
  const chartN = chart.kind === 'tick' ? chart.tickN : chart.minN;
  const day = snap?.date || status.date;
  const liveTick = status.live ? snap?.generatedAt : 0;
  const warmup = chart.ind.length > 0 && (chart.kind === 'tick' || chart.kind === 'min');
  const reqKey = `${day}/${selected}/${chart.kind}/${chartN}/${warmup ? 'w' : ''}${sym?.nxt ? 'x' : ''}`;
  // [2026-11-19] request + dataCache key shared with the background fetch (prefetch.ts)
  const fx = selected && day ? timelinePriceFetch(day, selected, chart, !!sym?.nxt) : null;
  const lastKey = useRef('');
  useEffect(() => {
    if (!selected || !day || !fx) return setPrice(null);
    const seq = ++reqSeq.current;
    const changed = lastKey.current !== reqKey;
    lastKey.current = reqKey;
    const ck = fx.key;
    if (changed) {
      // [2026-11-18] seen before: draw the kept bars now, refresh below
      const kept = peek<PriceResponse>(ck);
      setPrice(kept || null);
      setPriceLoading(!kept);
    }
    void window.api.getPrice(fx.req).then((r) => {
      if (r.ok) put(ck, r);
      if (seq !== reqSeq.current) return;
      setPriceLoading(false);
      // a failed live refresh keeps the bars already shown (also the kept bars of a switch)
      if (r.ok || (changed && !peek(ck))) setPrice(r);
    });
  }, [reqKey, liveTick, status.conn]); // eslint-disable-line react-hooks/exhaustive-deps

  // past-day replay: intraday bars after the replay time are not shown yet
  const shownBars = useMemo(
    () => (price?.ok ? barsUpTo(keptBars, replayAt, chart.kind === 'tick' || chart.kind === 'min') : []),
    [price, keptBars, replayAt, chart.kind],
  );
  // memoized so hovering the price chart does not rebuild its paths
  const intradayDomain = useMemo<[number, number] | null>(() => (xs.length || session ? [domain[0], domain[1]] : null), [xs.length > 0 || !!session, domain[0], domain[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  const buyLabel = (v: number) => shownRows[v]?.label || '';
  // heights: dragged by the user (layout) or the defaults
  const buyH = layout.buyH ?? Math.max(GRIP_MIN, Math.min(520, 92 + shownRows.length * 16));
  const sellH = layout.sellH ?? Math.max(160, Math.min(SELL_DEFAULT_H, 70 + shownSellRows.length * 14));
  const priceBoxH = chartHeight(chart.ind, TL_OSC_H, layout.priceH ?? TL_PRICE_H, layout.osc);
  const sellLabel = (v: number) => shownSellRows[v]?.label || '';

  const markers = trades.map((t, i) => (
    <ReferenceLine
      key={i}
      x={t.ts}
      stroke={t.side === 'BUY' ? 'var(--gain)' : 'var(--loss)'}
      strokeDasharray="4 3"
      label={{ value: tradeLabel(t), position: t.side === 'BUY' ? 'insideTopRight' : 'insideBottomLeft', fill: 'var(--text-2)', fontSize: 11 }}
    />
  ));

  return (
    <div className="timeline">
      <div className="toolbar">
        <SymbolPicker allowAll={false} />
        {sym?.nxt && <NxtMark />}
        {selected && <PriceTag code={selected} detail />}
        {sym && (
          <span className="symline">
            이벤트 {events.length.toLocaleString()}건 (최근 2,000건) · 거래 {trades.length}건{sym.realizedKrw ? ` · 실현 ${fmtKrw(sym.realizedKrw)}` : ''}
          </span>
        )}
        <span className="muted">점 위에 마우스를 올리면 로그 원문을 보여 주고, 누르면 고정합니다. 세로 점선: 매수/매도 체결</span>
      </div>
      {!selected && <div className="empty">종목을 선택하세요.</div>}
      {selected && (
        <div className="charts">
          <div className="chart-head">
            <h4>매수 평가 - 도달 깊이 (조건 평가 순서)</h4>
            <YPick
              side="매수"
              rows={buyRows}
              shown={shownRows.length}
              hidden={hiddenKeys}
              setHidden={setBuyHidden}
              evalOnly={buyEvalOnly}
              setEvalOnly={setBuyEvalOnly}
              evaluated={buyKeysWithEvents}
            />
            {/* [2026-11-12] legend moved out of the plot (was a recharts Legend above the grid) to the right end of the head row */}
            <span className="tl-legend">
              {SERIES.map((s) => (
                <span key={s.key}>
                  <i style={{ background: s.color }} />
                  {s.label}
                </span>
              ))}
            </span>
          </div>
          <div className="chart-buy" style={{ height: buyH }}>
            <ResponsiveContainer>
              <ScatterChart
                margin={{ top: 16, right: 24, bottom: 8, left: 8 }}
                onMouseMove={(e) => setHovBuy(nearestPt(e, buyPts, buyScales.current))}
                onMouseLeave={() => setHovBuy(null)}
                onClick={(e) => setPinBuy(nearestPt(e, buyPts, buyScales.current))}
              >
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" domain={domain} allowDataOverflow ticks={xTicks} tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} scale="time" />
                <YAxis type="number" dataKey="y" domain={[0, Math.max(1, shownRows.length - 1)]} ticks={[...Array(shownRows.length).keys()]} tickFormatter={buyLabel} width={150} stroke="var(--text-3)" fontSize={10} interval={0} />
                <Tooltip content={() => null} isAnimationActive={false} />
                {markers}
                {pinBuy && <ReferenceLine x={pinBuy.x} stroke="var(--accent)" strokeWidth={1.5} />}
                {priceHovT !== null && <ReferenceLine x={priceHovT} stroke="var(--text-2)" strokeDasharray="3 3" ifOverflow="hidden" />}
                {buySeries.map((s) => (
                  <Scatter key={s.key} name={s.label} data={s.data} fill={s.color} shape={s.shape} isAnimationActive={false} />
                ))}
                {/* [2026-11-17] focus rings: pinned dot (accent) and the dot under the cursor */}
                {pinBuy && <ReferenceDot x={pinBuy.x} y={pinBuy.y} r={9} fill="none" stroke="var(--accent)" strokeWidth={2.5} ifOverflow="hidden" />}
                {hovBuy && hovBuy !== pinBuy && <ReferenceDot x={hovBuy.x} y={hovBuy.y} r={9} fill="none" stroke="var(--text)" strokeWidth={2} ifOverflow="hidden" />}
                <Customized component={grabBuyScales} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          {/* [2026-11-17] a pinned dot stays in the dock while the mouse moves over other dots (click another dot to re-pin) */}
          <TipDock pt={pinBuy || hovBuy} side="매수" pinned={!!pinBuy} onUnpin={() => setPinBuy(null)} />
          <HeightGrip height={buyH} min={GRIP_MIN} max={GRIP_MAX} onChange={(h) => setLayout({ buyH: h })} onReset={() => setLayout({ buyH: undefined })} />
          {fullLen > 30 * 60000 && (
            <div className="time-window">
              <span className="muted small">보기 범위</span>
              <select value={span} onChange={(e) => setSpan(Number(e.target.value))} title="두 차트(매수 평가·주가)의 시간 범위">
                {[
                  [0, '전체'],
                  [2 * 3600000, '2시간'],
                  [3600000, '1시간'],
                  [1800000, '30분'],
                ].map(([v, label]) => (
                  <option key={v} value={v}>
                    {label}
                  </option>
                ))}
              </select>
              {zoomOk && (
                <>
                  <span className="badge-warn" title="주가 차트에서 휠 또는 Ctrl+끌기로 확대한 구간 (그냥 끌면 좌우 이동)">
                    확대 {fmtKstTime(domain[0]).slice(0, 5)} ~ {fmtKstTime(domain[1]).slice(0, 5)}
                  </span>
                  <button className="small" onClick={() => setZoom(null)} title="확대를 풀고 위에서 고른 범위로 돌아갑니다 (주가 차트 더블클릭도 같음)">
                    전체 보기
                  </button>
                </>
              )}
              {!zoomOk && win > 0 && (
                <>
                  <input type="range" min={0} max={100} step={0.5} value={startPct} onChange={(e) => setStartPct(Number(e.target.value))} title="좌우로 끌어 시간대 이동" />
                  <span className="mono small">
                    {fmtKstTime(domain[0]).slice(0, 5)} ~ {fmtKstTime(domain[1]).slice(0, 5)}
                  </span>
                </>
              )}
              <span className="muted small">
                {fmtKstTime(full[0]).slice(0, 5)} ~ {fmtKstTime(full[1]).slice(0, 5)}
              </span>
            </div>
          )}
          <div className="price-head">
            <h4>주가</h4>
            <div className="seg">
              {KIND_LABEL.map(([k, label]) => (
                <button key={k} className={chart.kind === k ? 'on' : ''} onClick={() => setChart({ kind: k })}>
                  {label}
                </button>
              ))}
            </div>
            {chart.kind === 'tick' && (
              <select value={chart.tickN} onChange={(e) => setChart({ tickN: Number(e.target.value) })} title="봉 하나에 들어가는 체결 수">
                {TICK_SIZES.map((v) => (
                  <option key={v} value={v}>
                    {v}틱
                  </option>
                ))}
              </select>
            )}
            {chart.kind === 'min' && (
              <select value={chart.minN} onChange={(e) => setChart({ minN: Number(e.target.value) })} title="봉 하나의 분">
                {MIN_SIZES.map((v) => (
                  <option key={v} value={v}>
                    {v}분
                  </option>
                ))}
              </select>
            )}
            <IndicatorToggles value={chart.ind} onChange={(ind) => setChart({ ind })} kind={chart.kind} />
            <span className="muted small" title="주가 차트: 휠 = 시간축 확대/축소, 끌기 = 좌우 이동, Ctrl+끌기 = 구간 확대, 더블클릭 = 전체">
              {priceLoading
                ? '불러오는 중…'
                : price?.ok
                  ? `${SOURCE_LABEL[price.source || ''] || ''} · ${price.bars.length.toLocaleString()}봉${price.warm?.length ? ` (+ 지표 예열 ${ymdShort(price.warmDate || '')} ${price.warm.length}봉)` : ''}${chart.kind === 'tick' || chart.kind === 'min' ? (intradayDomain ? ' · 매수 평가와 같은 시간축' : '') : ` · ${ymdShort(day)}까지`}${price.note ? ` · ${price.note}` : ''}`
                  : ''}
            </span>
          </div>
          <div className="chart-price" style={{ height: priceBoxH }}>
            {price && !price.ok && <div className="empty small">{price.error}</div>}
            {price?.ok && shownBars.length === 0 && <div className="empty small">{price.bars.length ? '재현 시각까지의 봉이 없습니다.' : '봉이 없습니다.'}</div>}
            {price?.ok && shownBars.length > 0 && (
              <PriceChart
                bars={shownBars}
                drawKey={selected}
                height={priceBoxH}
                oscHeight={TL_OSC_H}
                oscHeights={layout.osc}
                onResizePane={(k, h) => (k === 'price' ? setLayout({ priceH: h }) : setLayout({ osc: { [k]: h } }))}
                warm={price.warm}
                warmBot={price.warmBot}
                indicators={chart.ind}
                kind={chart.kind}
                n={chartN}
                domain={intradayDomain}
                xTicks={xTicks}
                onZoom={intradayKind ? setZoom : undefined}
                zoomLimit={full}
                trades={trades}
                buyPrice={sym?.position?.buyPrice}
                date={day}
                hoverTime={hovBuy ? hovBuy.ev.ts : hovSell ? hovSell.ev.ts : null}
                onHover={(i) => {
                  // only tick / minute bars share the evaluation charts' time axis
                  const b = i === null || !intradayKind ? undefined : shownBars[i];
                  const t = b ? b.t : null;
                  setPriceHovT((cur) => (cur === t ? cur : t));
                }}
              />
            )}
          </div>
          <div className="chart-head">
            <h4>매도 평가 (보유 중 발동/보류)</h4>
            <YPick
              side="매도"
              rows={sellRows}
              shown={shownSellRows.length}
              hidden={sellHiddenKeys}
              setHidden={setSellHidden}
              evalOnly={sellEvalOnly}
              setEvalOnly={setSellEvalOnly}
              evaluated={sellKeysWithEvents}
            />
          </div>
          <div className="chart-sell" style={{ height: sellH }}>
            {sellPts.length === 0 && trades.length === 0 ? (
              <div className="empty small">오늘 이 종목의 매도 평가 로그가 없습니다.</div>
            ) : (
              <ResponsiveContainer>
                <ScatterChart
                  margin={{ top: 8, right: 24, bottom: 8, left: 8 }}
                  onMouseMove={(e) => setHovSell(nearestPt(e, sellShownPts, sellScales.current))}
                  onMouseLeave={() => setHovSell(null)}
                  onClick={(e) => setPinSell(nearestPt(e, sellShownPts, sellScales.current))}
                >
                  <CartesianGrid stroke="var(--grid)" />
                  <XAxis type="number" dataKey="x" domain={domain} allowDataOverflow ticks={xTicks} tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} scale="time" />
                  <YAxis type="number" dataKey="y" domain={[0, Math.max(1, shownSellRows.length - 1)]} ticks={[...Array(shownSellRows.length).keys()]} tickFormatter={sellLabel} width={150} stroke="var(--text-3)" fontSize={10} interval={0} />
                  <Tooltip content={() => null} isAnimationActive={false} />
                  {markers}
                  {pinSell && <ReferenceLine x={pinSell.x} stroke="var(--accent)" strokeWidth={1.5} />}
                  {priceHovT !== null && <ReferenceLine x={priceHovT} stroke="var(--text-2)" strokeDasharray="3 3" ifOverflow="hidden" />}
                  <Scatter name="발동" data={sellSeries.pass} fill="var(--st-fail)" shape="cross" isAnimationActive={false} />
                  <Scatter name="보류" data={sellSeries.hold} fill="var(--st-hold)" shape="diamond" isAnimationActive={false} />
                  <Scatter name="정보" data={sellSeries.info} fill="var(--st-idle)" shape="square" isAnimationActive={false} />
                  {pinSell && <ReferenceDot x={pinSell.x} y={pinSell.y} r={9} fill="none" stroke="var(--accent)" strokeWidth={2.5} ifOverflow="hidden" />}
                  {hovSell && hovSell !== pinSell && <ReferenceDot x={hovSell.x} y={hovSell.y} r={9} fill="none" stroke="var(--text)" strokeWidth={2} ifOverflow="hidden" />}
                  <Customized component={grabSellScales} />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
          <TipDock pt={pinSell || hovSell} side="매도" pinned={!!pinSell} onUnpin={() => setPinSell(null)} />
          <HeightGrip height={sellH} min={GRIP_MIN} max={GRIP_MAX} onChange={(h) => setLayout({ sellH: h })} onReset={() => setLayout({ sellH: undefined })} />
          {trades.length > 0 && (
            <table className="grid compact">
              <thead>
                <tr>
                  <th>시각</th>
                  <th>구분</th>
                  <th className="num">가격</th>
                  <th className="num">수량</th>
                  <th>사유 / 조건</th>
                  <th className="num">손익</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => (
                  <tr key={i} title={t.raw}>
                    <td className="mono">{fmtKstTime(t.ts)}</td>
                    <td>{t.side === 'BUY' ? '매수' : '매도'}</td>
                    <td className="num">{fmtInt(t.price)}</td>
                    <td className="num">{fmtInt(t.qty)}</td>
                    <td>
                      {t.side === 'SELL' ? condTitle(conditions, 'sell', t.condKey) + ' · ' : ''}
                      {t.reason}
                    </td>
                    <td className="num">{t.side === 'SELL' ? `${fmtKrw(t.pnlKrw)} (${t.pnlPct?.toFixed(2)}%)` : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

/** fill marker text: "매수 22,133" / "매도 22,450 (+1.43%) AUX_REVERSAL_SCORE" */
function tradeLabel(t: Trade): string {
  const price = Number.isFinite(t.price) ? ` ${fmtInt(t.price)}` : '';
  if (t.side === 'BUY') return `매수${price}`;
  const pct = Number.isFinite(t.pnlPct) ? ` (${(t.pnlPct as number) >= 0 ? '+' : ''}${(t.pnlPct as number).toFixed(2)}%)` : '';
  return `매도${price}${pct} ${t.reason.slice(0, 18)}`.trim();
}

const ymdShort = (ymd: string) => (ymd ? `${ymd.slice(4, 6)}-${ymd.slice(6, 8)}` : '');
