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
 * - [2026-11-11] type=feat - Hover details of the buy / sell evaluation charts move to a dock on the right of each chart (TipDock,
 *   .chart-row) with the reason in Korean (shared/reasonKo) + the code + the raw line; the floating tooltip no longer covers the dots
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Legend } from 'recharts';
import { reasonKo } from '../../../shared/reasonKo';
import { useStore } from '../store';
import { SymbolPicker } from '../components/SymbolPicker';
import { condTitle, fmtInt, fmtKrw, sortedConds } from '../lib';
import { condKeyOf } from '../../../shared/mapping';
import { fmtKstTime } from '../../../shared/time';
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
function TipDock({ pt, side }: { pt: Pt | null; side: '매수' | '매도' }) {
  const conditions = useStore((s) => s.conditions);
  if (!pt) {
    return (
      <div className="tip-dock">
        <div className="muted small">{side} 평가 차트의 점 위에 마우스를 올리면 여기에 내용이 나옵니다.</div>
      </div>
    );
  }
  const ev = pt.ev;
  return (
    <div className="tip-dock">
      <div>
        <b>{fmtKstTime(ev.ts)}</b> [{ev.tag}] {condTitle(conditions, ev.side, ev.condKey)}
        {ev.sub ? ` / ${ev.sub}` : ''}
      </div>
      <div className="reason-ko">{reasonKo(ev.reason) || ev.reason}</div>
      <div className="muted small mono">{ev.reason}</div>
      <pre className="raw">{ev.raw}</pre>
    </div>
  );
}

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
  // recharts gives the active payload with every mouse move; the Tooltip itself stays invisible (content null) and only drives the cursor
  const pickPt = (e: unknown): Pt | null => {
    const ap = (e as { activePayload?: { payload?: Pt }[] } | null)?.activePayload;
    return ap && ap.length ? ap[0].payload || null : null;
  };

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

  const events = timeline && timeline.code === selected ? timeline.events : [];
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
  const replayAt = snap?.replay?.cursor ?? null;
  // tick / minute bars of the day widen the axis too (evaluations may stop before the close, e.g. r003 stopped at 13:40)
  const intradayKind = chart.kind === 'tick' || chart.kind === 'min';
  const barTs = intradayKind && price?.ok && price.bars.length ? [price.bars[0].t, price.bars[price.bars.length - 1].t2 ?? price.bars[price.bars.length - 1].t] : [];
  const xs = [...events.map((e) => e.ts), ...trades.map((t) => t.ts), ...barTs].filter(Number.isFinite);
  // replay: the axis runs to the replay time, so it moves while playing even when the symbol has no new events (r003 stopped)
  const full: [number, number] = xs.length
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
  const domain: [number, number] = win ? [winStart, winStart + win] : full;
  // explicit 30-minute ticks (recharts' automatic ticks on a time scale bunch up at the start)
  const xTicks = useMemo(() => {
    const step = 30 * 60 * 1000;
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
  const reqKey = `${day}/${selected}/${chart.kind}/${chartN}/${warmup ? 'w' : ''}`;
  const lastKey = useRef('');
  useEffect(() => {
    if (!selected || !day) return setPrice(null);
    const seq = ++reqSeq.current;
    const changed = lastKey.current !== reqKey;
    lastKey.current = reqKey;
    if (changed) {
      setPrice(null);
      setPriceLoading(true);
    }
    void window.api.getPrice({ date: day, code: selected, kind: chart.kind, n: chartN, warmup }).then((r) => {
      if (seq !== reqSeq.current) return;
      setPriceLoading(false);
      // a failed live refresh keeps the bars already shown
      if (r.ok || changed) setPrice(r);
    });
  }, [reqKey, liveTick, status.conn]); // eslint-disable-line react-hooks/exhaustive-deps

  // past-day replay: intraday bars after the replay time are not shown yet
  const shownBars = useMemo(
    () => (price?.ok ? barsUpTo(price.bars, replayAt, chart.kind === 'tick' || chart.kind === 'min') : []),
    [price, replayAt, chart.kind],
  );
  // memoized so hovering the price chart does not rebuild its paths
  const intradayDomain = useMemo<[number, number] | null>(() => (xs.length ? [domain[0], domain[1]] : null), [xs.length > 0, domain[0], domain[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  const buyLabel = (v: number) => shownRows[v]?.label || '';
  // heights: dragged by the user (layout) or the defaults
  const buyH = layout.buyH ?? Math.max(200, Math.min(520, 110 + shownRows.length * 15));
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
        {sym && (
          <span className="symline">
            이벤트 {events.length.toLocaleString()}건 (최근 2,000건) · 거래 {trades.length}건{sym.realizedKrw ? ` · 실현 ${fmtKrw(sym.realizedKrw)}` : ''}
          </span>
        )}
        <span className="muted">점 위에 마우스를 올리면 로그 원문을 보여 줍니다. 세로 점선: 매수/매도 체결</span>
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
          </div>
          <div className="chart-row">
          <div className="chart-buy" style={{ height: buyH }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 24, bottom: 8, left: 8 }} onMouseMove={(e) => setHovBuy(pickPt(e))} onMouseLeave={() => setHovBuy(null)}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" domain={domain} allowDataOverflow ticks={xTicks} tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} scale="time" />
                <YAxis type="number" dataKey="y" domain={[0, Math.max(1, shownRows.length - 1)]} ticks={[...Array(shownRows.length).keys()]} tickFormatter={buyLabel} width={150} stroke="var(--text-3)" fontSize={10} interval={0} />
                <Tooltip content={() => null} isAnimationActive={false} />
                <Legend verticalAlign="top" height={24} />
                {markers}
                {SERIES.map((s) => (
                  <Scatter key={s.key} name={s.label} data={buyPts.filter((p) => p.ev.result === s.key)} fill={s.color} shape={s.shape} isAnimationActive={false} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <TipDock pt={hovBuy} side="매수" />
          </div>
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
              {win > 0 && (
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
            <span className="muted small">
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
                trades={trades}
                buyPrice={sym?.position?.buyPrice}
                date={day}
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
          <div className="chart-row">
          <div className="chart-sell" style={{ height: sellH }}>
            {sellPts.length === 0 && trades.length === 0 ? (
              <div className="empty small">오늘 이 종목의 매도 평가 로그가 없습니다.</div>
            ) : (
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 8 }} onMouseMove={(e) => setHovSell(pickPt(e))} onMouseLeave={() => setHovSell(null)}>
                  <CartesianGrid stroke="var(--grid)" />
                  <XAxis type="number" dataKey="x" domain={domain} allowDataOverflow ticks={xTicks} tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} scale="time" />
                  <YAxis type="number" dataKey="y" domain={[0, Math.max(1, shownSellRows.length - 1)]} ticks={[...Array(shownSellRows.length).keys()]} tickFormatter={sellLabel} width={150} stroke="var(--text-3)" fontSize={10} interval={0} />
                  <Tooltip content={() => null} isAnimationActive={false} />
                  {markers}
                  <Scatter name="발동" data={sellPts.filter((p) => p.ev.result === 'pass')} fill="var(--st-fail)" shape="cross" isAnimationActive={false} />
                  <Scatter name="보류" data={sellPts.filter((p) => p.ev.result === 'hold')} fill="var(--st-hold)" shape="diamond" isAnimationActive={false} />
                  <Scatter name="정보" data={sellPts.filter((p) => p.ev.result === 'info')} fill="var(--st-idle)" shape="square" isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
          <TipDock pt={hovSell} side="매도" />
          </div>
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
