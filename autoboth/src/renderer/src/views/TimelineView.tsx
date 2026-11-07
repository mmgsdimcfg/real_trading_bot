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
 * - [2026-11-07] type=feat - Buy chart y-axis rows can be hidden ("y축 항목" popover: per-row checkboxes, 모두 표시, 이 종목 평가된 항목만);
 *   hidden rows and their dots are left out, the remaining rows close up and the chart gets shorter. Kept per PC (localStorage)
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Legend } from 'recharts';
import { useStore } from '../store';
import { SymbolPicker } from '../components/SymbolPicker';
import { condTitle, fmtInt, fmtKrw, sortedConds } from '../lib';
import { condKeyOf } from '../../../shared/mapping';
import { fmtKstTime } from '../../../shared/time';
import type { EvalEvent } from '../../../shared/types';
import { MIN_SIZES, TICK_SIZES, type ChartKind, type PriceResponse } from '../../../shared/price';
import { PriceChart, chartHeight } from '../components/PriceChart';
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

function TipBox({ active, payload }: { active?: boolean; payload?: { payload: Pt }[] }) {
  const conditions = useStore((s) => s.conditions);
  if (!active || !payload?.length) return null;
  const ev = payload[0].payload.ev;
  return (
    <div className="tip">
      <div>
        <b>{fmtKstTime(ev.ts)}</b> [{ev.tag}] {condTitle(conditions, ev.side, ev.condKey)}
        {ev.sub ? ` / ${ev.sub}` : ''}
      </div>
      <div>{ev.reason}</div>
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

  const buyConds = useMemo(() => sortedConds(conditions, 'buy'), [conditions]);
  const sellConds = useMemo(() => sortedConds(conditions, 'sell'), [conditions]);
  const buyHidden = useStore((s) => s.buyHidden);
  const setBuyHidden = useStore((s) => s.setBuyHidden);
  /** every buy row bottom-up: 준비, conditions in evaluation order, 주문 */
  const buyRows = useMemo(
    () => [
      { key: 'PREP', label: '준비' },
      ...buyConds.map((c) => ({ key: condKeyOf(c.no), label: condKeyOf(c.no) + ' ' + c.title.slice(0, 10) })),
      { key: 'ORDER', label: '주문' },
    ],
    [buyConds],
  );
  const shownRows = useMemo(() => {
    const hide = new Set(buyHidden);
    const rows = buyRows.filter((r) => !hide.has(r.key));
    return rows.length ? rows : buyRows;
  }, [buyRows, buyHidden]);
  const buyY = useMemo(() => new Map(shownRows.map((r, i) => [r.key, i])), [shownRows]);
  const sellY = useMemo(() => {
    // row 0 = 'PREP': sell evaluation skipped for the position ([HOLD SKIP], e.g. not bought today)
    const m = new Map<string, number>([['PREP', 0]]);
    sellConds.forEach((c, i) => m.set(condKeyOf(c.no), i + 1));
    m.set('SESSION_CLOSE', sellConds.length + 1);
    m.set('ORDER', sellConds.length + 2);
    return m;
  }, [sellConds]);

  const events = timeline && timeline.code === selected ? timeline.events : [];
  const trades = timeline && timeline.code === selected ? timeline.trades : [];
  const buyPts = useMemo(() => events.filter((e) => e.side === 'buy' && buyY.has(e.condKey)).map((ev) => ({ x: ev.ts, y: buyY.get(ev.condKey)!, ev })), [events, buyY]);
  const sellPts = useMemo(() => events.filter((e) => e.side === 'sell' && sellY.has(e.condKey)).map((ev) => ({ x: ev.ts, y: sellY.get(ev.condKey)!, ev })), [events, sellY]);
  const xs = [...events.map((e) => e.ts), ...trades.map((t) => t.ts)].filter(Number.isFinite);
  const replayAt = snap?.replay?.cursor ?? null;
  // replay: the axis runs to the replay time, so it moves while playing even when the symbol has no new events (r003 stopped)
  const domain: [number, number] = xs.length
    ? [Math.min(...xs), replayAt !== null ? Math.max(replayAt, Math.min(...xs) + 60000) : Math.max(...xs)]
    : replayAt !== null
      ? [replayAt - 30 * 60000, replayAt]
      : [0, 1];
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
  const buyKeysWithEvents = useMemo(() => new Set(events.filter((e) => e.side === 'buy').map((e) => e.condKey)), [events]);
  const sellLabel = (v: number) =>
    v === 0 ? '평가 제외' : v === sellConds.length + 1 ? '세션 종료' : v === sellConds.length + 2 ? '주문' : sellConds[v - 1] ? condKeyOf(sellConds[v - 1].no) + ' ' + sellConds[v - 1].title.slice(0, 10) : '';

  const markers = trades.map((t, i) => (
    <ReferenceLine
      key={i}
      x={t.ts}
      stroke={t.side === 'BUY' ? 'var(--gain)' : 'var(--loss)'}
      strokeDasharray="4 3"
      label={{ value: t.side === 'BUY' ? `매수 ${fmtInt(t.price)}` : `매도 ${t.reason.slice(0, 18)}`, position: t.side === 'BUY' ? 'insideTopRight' : 'insideBottomLeft', fill: 'var(--text-2)', fontSize: 11 }}
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
            <details className="ypick">
              <summary>
                y축 항목 ({shownRows.length}/{buyRows.length}) ▾
              </summary>
              <div className="ypick-panel">
                <div className="actions">
                  <button className="mini" onClick={() => setBuyHidden([])}>
                    모두 표시
                  </button>
                  <button
                    className="mini"
                    title="선택한 종목의 매수 평가 로그에 나온 항목만 남깁니다"
                    disabled={!buyKeysWithEvents.size}
                    onClick={() => setBuyHidden(buyRows.filter((r) => !buyKeysWithEvents.has(r.key)).map((r) => r.key))}
                  >
                    이 종목 평가된 항목만
                  </button>
                </div>
                {[...buyRows].reverse().map((r) => (
                  <label key={r.key} className="check ypick-row">
                    <input
                      type="checkbox"
                      checked={!buyHidden.includes(r.key)}
                      onChange={(e) => setBuyHidden(e.target.checked ? buyHidden.filter((k) => k !== r.key) : [...buyHidden, r.key])}
                    />
                    <span className={buyKeysWithEvents.has(r.key) ? '' : 'muted'}>{r.label}</span>
                  </label>
                ))}
                <p className="muted small">숨긴 항목의 점은 표시하지 않습니다. 흐린 글자 = 이 종목에서 평가 로그가 없는 항목.</p>
              </div>
            </details>
          </div>
          <div className="chart-buy" style={{ height: Math.max(200, Math.min(520, 110 + shownRows.length * 15)) }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" domain={domain} ticks={xTicks} tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} scale="time" />
                <YAxis type="number" dataKey="y" domain={[0, Math.max(1, shownRows.length - 1)]} ticks={[...Array(shownRows.length).keys()]} tickFormatter={buyLabel} width={150} stroke="var(--text-3)" fontSize={10} interval={0} />
                <Tooltip content={<TipBox />} isAnimationActive={false} />
                <Legend verticalAlign="top" height={24} />
                {markers}
                {SERIES.map((s) => (
                  <Scatter key={s.key} name={s.label} data={buyPts.filter((p) => p.ev.result === s.key)} fill={s.color} shape={s.shape} isAnimationActive={false} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
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
          <div className="chart-price" style={{ height: chartHeight(chart.ind) }}>
            {price && !price.ok && <div className="empty small">{price.error}</div>}
            {price?.ok && shownBars.length === 0 && <div className="empty small">{price.bars.length ? '재현 시각까지의 봉이 없습니다.' : '봉이 없습니다.'}</div>}
            {price?.ok && shownBars.length > 0 && (
              <PriceChart
                bars={shownBars}
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
          <h4>매도 평가 (보유 중 발동/보류)</h4>
          <div className="chart-sell">
            {sellPts.length === 0 && trades.length === 0 ? (
              <div className="empty small">오늘 이 종목의 매도 평가 로그가 없습니다.</div>
            ) : (
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
                  <CartesianGrid stroke="var(--grid)" />
                  <XAxis type="number" dataKey="x" domain={domain} ticks={xTicks} tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} scale="time" />
                  <YAxis type="number" dataKey="y" domain={[0, sellConds.length + 2]} ticks={[...Array(sellConds.length + 3).keys()]} tickFormatter={sellLabel} width={150} stroke="var(--text-3)" fontSize={10} interval={0} />
                  <Tooltip content={<TipBox />} isAnimationActive={false} />
                  {markers}
                  <Scatter name="발동" data={sellPts.filter((p) => p.ev.result === 'pass')} fill="var(--st-fail)" shape="cross" isAnimationActive={false} />
                  <Scatter name="보류" data={sellPts.filter((p) => p.ev.result === 'hold')} fill="var(--st-hold)" shape="diamond" isAnimationActive={false} />
                  <Scatter name="정보" data={sellPts.filter((p) => p.ev.result === 'info')} fill="var(--st-idle)" shape="square" isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
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

const ymdShort = (ymd: string) => (ymd ? `${ymd.slice(4, 6)}-${ymd.slice(6, 8)}` : '');
