/*
 * Update log
 * - [2026-11-04] type=feat - Trade results from YYYYMMDD_buy_sell.log: headline numbers, cumulative realized P&L over the day,
 *   P&L by sell condition, and the trade table (Korean convention: gain red, loss blue)
 * - [2026-11-11] type=feat - "제세금 포함" switch (store.feesIncluded, shared with 보유 현황): every P&L figure (KPIs, cumulative
 *   line, per-condition bars, table) is shown net of the estimated commission + sell tax (shared/fees) - the log's 수익금액 is gross
 * - [2026-11-27] type=feat - 매매 방식 · 모드별 결과 (user 2026-11-27): a table on top with one row per method and mode - 규칙 매매
 *   (실제), Edge AI 매매 (실제 / 가상), VWAP 매매 (실제 / 가상) - what each does now, its buys / sells / win rate / P&L of the
 *   day. A real row filters this tab to that method (KPIs, charts, table), a paper row opens the 가상 매매 tab on that book. Each
 *   trade row carries its method (shared/strategy: from the buy block / sell reason r003 logs)
 * - [2026-11-27] type=feat - 기간 (user 2026-11-27): 일간 / 주간 / 월간 / 직접 지정. A range shows the same method x mode table summed
 *   over its days (main 'results:range': each day's buy_sell log and both paper books) and a per-day table under it; a day row
 *   opens that day. The day-level KPIs / charts / list are shown for 일간 only
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { NxtTag } from '../components/NxtTag';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, LabelList } from 'recharts';
import { useStore } from '../store';
import { condTitle, fmtInt, fmtKrw, fmtPct, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import { feeLabel, pnlKrw as netKrw, pnlPct as netPct } from '../../../shared/fees';
import { sideExitKo, strategyModeKo, strategyModes, strategyOfTrade, strategySummary, TRADE_STRATEGIES, TRADE_STRATEGY_KO, TRADE_STRATEGY_SHORT, type TradeStrategy } from '../../../shared/strategy';
import { paperSummary, type PaperBook, type PaperTrade } from '../../../shared/paper';
import { peek, put } from '../dataCache';
import { paperKrw, paperOrderKrw } from './PaperView';
import { isYmd, PERIODS, PERIOD_KO, periodRange, shiftPeriod, sumResults, type DayResults, type Period, type RangeResults } from '../../../shared/rangeResults';
import { ymdDash } from '../../../shared/time';

type PaperRes = { day: string; ok: boolean; trades: PaperTrade[]; error?: string };
/** the day's paper trades of one book (same cache entry as the 가상 매매 tab) */
function usePaperTrades(book: PaperBook, day: string | undefined, tick: number | undefined, conn: string): PaperTrade[] | null {
  const key = `paper:${book}:${day}`;
  const [res, setRes] = useState<PaperRes | null>(() => (day ? peek<PaperRes>(key) || null : null));
  useEffect(() => {
    if (!day) return;
    let alive = true;
    const kept = peek<PaperRes>(key);
    setRes(kept || null);
    void window.api.getPaper(day, book).then((r) => {
      if (r.ok) put(key, { day, ...r });
      if (alive && (r.ok || !kept)) setRes({ day, ...r });
    });
    return () => {
      alive = false;
    };
  }, [day, tick, conn, book]); // eslint-disable-line react-hooks/exhaustive-deps
  return res && res.day === day && res.ok ? res.trades : null;
}

export function TradesView() {
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const net = useStore((s) => s.feesIncluded);
  const setNet = useStore((s) => s.setFeesIncluded);
  const allTrades = snap?.trades || [];
  const status = useStore((s) => s.status);
  const control = useStore((s) => s.control);
  const filter = useStore((s) => s.tradeFilter);
  const setFilter = useStore((s) => s.setTradeFilter);
  const openResults = useStore((s) => s.openResults);
  const day = snap?.date || status.date;
  const tick = status.live ? snap?.generatedAt : 0;
  const paperEdge = usePaperTrades('edge', day, tick, status.conn);
  const paperVwap = usePaperTrades('vwap', day, tick, status.conn);
  const modes = useMemo(() => strategyModes(control), [control]);
  // every method's real trades (net when the switch is on) for the summary table, whatever the filter
  const allRows = useMemo(
    () =>
      allTrades.map((t) => {
        const can = net && t.side === 'SELL' && !!t.buyPrice && !!t.price && !!t.qty;
        return can ? { ...t, pnlKrw: netKrw(t.buyPrice!, t.price, t.qty, true) } : t;
      }),
    [allTrades, net],
  );
  const byStrategy = useMemo(() => strategySummary(allRows), [allRows]);
  const paperRow = (rows: PaperTrade[] | null) => {
    if (!rows) return null;
    const s = paperSummary(rows);
    const budget = paperOrderKrw();
    const krw = rows.filter((t) => t.exitTs !== undefined).reduce((a, t) => a + (paperKrw(t.entry, budget, { netPct: t.netPct ?? 0 }) || 0), 0);
    return { buys: rows.length, sells: s.closed, wins: s.wins, pnlKrw: krw, open: s.open };
  };
  const paperSum: Record<PaperBook, ReturnType<typeof paperRow>> = { edge: paperRow(paperEdge), vwap: paperRow(paperVwap) };
  const trades = useMemo(() => (filter === 'all' ? allTrades : allTrades.filter((t) => strategyOfTrade(t) === filter)), [allTrades, filter]);
  // net figures need the buy price, the sell price and the quantity of the sell block; a sell without them stays gross
  const sells = useMemo(
    () =>
      trades
        .filter((t) => t.side === 'SELL')
        .map((t) => {
          const can = net && !!t.buyPrice && !!t.price && !!t.qty;
          return { ...t, pnlKrw: can ? netKrw(t.buyPrice!, t.price, t.qty, true) : t.pnlKrw, pnlPct: can ? netPct(t.buyPrice!, t.price, t.qty, true) : t.pnlPct };
        }),
    [trades, net],
  );
  const total = sells.reduce((a, t) => a + (t.pnlKrw || 0), 0);
  const wins = sells.filter((t) => (t.pnlKrw || 0) > 0).length;

  const cum = useMemo(() => {
    let acc = 0;
    const pts = [...sells]
      .sort((a, b) => a.ts - b.ts)
      .map((t) => {
        acc += t.pnlKrw || 0;
        return { x: t.ts, y: acc, t: t as (typeof sells)[number] | null };
      });
    // start the line at 0 at the first buy of the day
    const first = trades.reduce((m, t) => Math.min(m, Number.isFinite(t.ts) ? t.ts : m), Infinity);
    if (pts.length && Number.isFinite(first)) pts.unshift({ x: Math.min(first, pts[0].x), y: 0, t: null });
    return pts;
  }, [sells, trades]);

  const byCond = useMemo(() => {
    const m = new Map<string, { key: string; sum: number; n: number; label: string }>();
    for (const t of sells) {
      const side = strategyOfTrade(t) !== 'rule' ? sideExitKo(t.reason) : '';
      const k = side ? `${strategyOfTrade(t)}:${side}` : t.condKey || 'UNKNOWN';
      const e = m.get(k) || { key: k, sum: 0, n: 0, label: side ? `${TRADE_STRATEGY_SHORT[strategyOfTrade(t)]} ${side}` : '' };
      e.sum += t.pnlKrw || 0;
      e.n++;
      m.set(k, e);
    }
    return [...m.values()].sort((a, b) => a.sum - b.sum).map((e) => ({ ...e, label: e.label || condTitle(conditions, 'sell', e.key) }));
  }, [sells, conditions]);

  // ---- [2026-11-27] period: the viewed day, or a range of days read by the main process -------------------------------------
  const [period, setPeriod] = useState<Period>('day');
  /** a day inside the shown week / month (starts at the viewed day) */
  const [anchor, setAnchor] = useState<string>('');
  const [custom, setCustom] = useState<{ from: string; to: string }>({ from: '', to: '' });
  const base = isYmd(anchor) ? anchor : day || '';
  const range: [string, string] | null =
    period === 'day' || !isYmd(base)
      ? null
      : period === 'custom'
        ? isYmd(custom.from) && isYmd(custom.to) && custom.from <= custom.to
          ? [custom.from, custom.to]
          : null
        : periodRange(period, base);
  const rangeKey = range ? `${range[0]}-${range[1]}` : '';
  const [rangeRes, setRangeRes] = useState<{ key: string; res: RangeResults } | null>(null);
  const [rangeBusy, setRangeBusy] = useState(false);
  const [rangeRev, setRangeRev] = useState(0);
  const reqRef = useRef(0);
  useEffect(() => {
    if (!rangeKey) return;
    const [from, to] = rangeKey.split('-');
    const id = ++reqRef.current;
    setRangeBusy(true);
    void window.api
      .getRangeResults(from, to, paperOrderKrw())
      .then((res) => {
        if (id === reqRef.current) setRangeRes({ key: rangeKey, res });
      })
      .catch((e) => {
        if (id === reqRef.current) setRangeRes({ key: rangeKey, res: { ok: false, days: [], error: String((e as Error)?.message || e) } });
      })
      .finally(() => {
        if (id === reqRef.current) setRangeBusy(false);
      });
  }, [rangeKey, rangeRev, status.conn]);
  const rangeDays: DayResults[] | null = range && rangeRes && rangeRes.key === rangeKey && rangeRes.res.ok ? rangeRes.res.days : null;
  const rangeSum = useMemo(() => (rangeDays ? sumResults(rangeDays) : null), [rangeDays]);
  const pickPeriod = (p: Period) => {
    if (p === 'custom' && !isYmd(custom.from) && isYmd(base)) setCustom({ from: periodRange('month', base)[0], to: base });
    setPeriod(p);
  };
  const realOf = (d: DayResults['real'][TradeStrategy]) => ({ buys: d.buys, sells: d.sells, wins: net ? d.winsNet : d.wins, pnlKrw: net ? d.pnlNet : d.pnlKrw });
  const openDay = (ymd: string) => {
    setPeriod('day');
    if (ymd !== day) void window.api.openDay(ymd);
  };
  const cell = (d: { sells: number; pnlKrw: number; buys: number } | null) =>
    !d ? <td className="num muted">읽지 못함</td> : d.buys || d.sells ? (
      <td className={`num ${pnlClass(d.sells ? d.pnlKrw : undefined)}`}>
        {d.sells ? fmtKrw(d.pnlKrw) : '-'} <span className="muted small">{d.buys}/{d.sells}</span>
      </td>
    ) : (
      <td className="num muted">-</td>
    );

  const sellLabel = (t: (typeof sells)[number]) => {
    const ex = strategyOfTrade(t) !== 'rule' ? sideExitKo(t.reason) : '';
    return ex ? `${TRADE_STRATEGY_SHORT[strategyOfTrade(t)]} ${ex}` : condTitle(conditions, 'sell', t.condKey);
  };
  const modeRows: { s: TradeStrategy; mode: 'live' | 'paper' }[] = [
    { s: 'rule', mode: 'live' },
    { s: 'edge', mode: 'live' },
    { s: 'edge', mode: 'paper' },
    { s: 'vwap', mode: 'live' },
    { s: 'vwap', mode: 'paper' },
  ];

  return (
    <div className="trades">
      <div className="strat-sum">
        <div className="strat-bar period">
          <div className="seg" title="결과를 볼 기간">
            {PERIODS.map((p) => (
              <button key={p} type="button" className={period === p ? 'on' : ''} onClick={() => pickPeriod(p)}>
                {PERIOD_KO[p]}
              </button>
            ))}
          </div>
          {(period === 'week' || period === 'month') && isYmd(base) && (
            <>
              <button type="button" className="chip link" onClick={() => setAnchor(shiftPeriod(period, base, -1))}>
                ◀ 이전
              </button>
              <b className="mono">{range ? `${ymdDash(range[0])} ~ ${ymdDash(range[1])}` : ''}</b>
              <button type="button" className="chip link" onClick={() => setAnchor(shiftPeriod(period, base, 1))}>
                다음 ▶
              </button>
              {day && base !== day && (
                <button type="button" className="chip link" title="보고 있는 날짜가 든 기간으로" onClick={() => setAnchor('')}>
                  {ymdDash(day)} 기준
                </button>
              )}
            </>
          )}
          {period === 'custom' && (
            <>
              <input className="mono ymd" value={custom.from} placeholder="YYYYMMDD" maxLength={8} onChange={(e) => setCustom((c) => ({ ...c, from: e.target.value.replace(/\D/g, '') }))} />
              <span>~</span>
              <input className="mono ymd" value={custom.to} placeholder="YYYYMMDD" maxLength={8} onChange={(e) => setCustom((c) => ({ ...c, to: e.target.value.replace(/\D/g, '') }))} />
              {!range && <span className="badge-warn">날짜 8자리 (시작 ≤ 끝)</span>}
            </>
          )}
          {period === 'day' ? (
            <span className="muted small">{day ? `${ymdDash(day)} 하루` : ''}</span>
          ) : (
            <>
              <button type="button" className="chip link" disabled={!range || rangeBusy} onClick={() => setRangeRev((n) => n + 1)}>
                새로 읽기
              </button>
              <span className="muted small">
                {rangeBusy ? '읽는 중…' : rangeDays ? `거래일 ${rangeDays.length}일 (로그가 있는 날)` : ''}
                {rangeRes && rangeRes.key === rangeKey && rangeRes.res.failed?.length ? ` · 읽지 못한 날 ${rangeRes.res.failed.map(ymdDash).join(', ')}` : ''}
              </span>
              {range && rangeRes && rangeRes.key === rangeKey && !rangeRes.res.ok && <span className="badge-warn">{rangeRes.res.error || '읽지 못했습니다'}</span>}
              <label className="check" title={`손익에서 예상 거래 비용을 뺍니다: ${feeLabel()}`}>
                <input type="checkbox" checked={net} onChange={(e) => setNet(e.target.checked)} /> 제세금 포함
              </label>
            </>
          )}
        </div>
        <table className="grid">
          <thead>
            <tr>
              <th>매매 방식</th>
              <th>모드</th>
              <th title="실행 중인 봇의 설정 (봇이 꺼져 있으면 저장된 설정). 오늘 도중에 바꿨다면 그 전 거래도 아래 숫자에 들어 있습니다">지금 상태</th>
              <th className="num">매수</th>
              <th className="num">매도 (청산)</th>
              <th className="num">승률</th>
              <th className="num" title="실제 = 로그의 실현 손익 (제세금 포함 스위치 반영) · 가상 = 1회 매수 상한 금액으로 샀다고 본 순손익 (비용 차감)">손익</th>
              <th>내역 위치</th>
            </tr>
          </thead>
          <tbody>
            {modeRows.map(({ s, mode }) => {
              const m = modes[s];
              const on = mode === 'live' ? m.live : m.paper;
              const isDay = period === 'day';
              const d = isDay
                ? mode === 'live'
                  ? byStrategy[s]
                  : paperSum[s as PaperBook]
                : !rangeSum
                  ? null
                  : mode === 'live'
                    ? realOf(rangeSum.real[s])
                    : rangeSum.paper[s as PaperBook];
              const sel = isDay && mode === 'live' && filter === s;
              const openN = mode === 'paper' ? (isDay ? paperSum[s as PaperBook]?.open : rangeSum?.paper[s as PaperBook].open) || 0 : 0;
              return (
                <tr
                  key={`${s}:${mode}`}
                  className={`${sel ? 'sel' : ''} ${on ? 'run' : ''}`}
                  title={!isDay ? '기간 합계 - 날짜별 내역은 아래 표의 날짜를 누르세요' : mode === 'live' ? '누르면 아래 내역을 이 매매 방식으로 거릅니다 (다시 누르면 전체)' : '누르면 가상 매매 탭의 이 장부를 엽니다'}
                  onClick={() => (!isDay ? undefined : mode === 'live' ? setFilter(sel ? 'all' : s) : openResults(s, 'paper'))}
                >
                  <td>
                    <b>{TRADE_STRATEGY_KO[s]}</b>
                  </td>
                  <td>
                    <span className={`tag mode-${mode}`}>{mode === 'live' ? '실제' : '가상'}</span>
                  </td>
                  <td>
                    <span className={`st-dot ${on ? 'on' : ''}`} />
                    {on === null ? '확인 안 됨' : on ? (mode === 'live' ? '실제 주문 중' : '가상 매매 중') : '꺼짐'}
                    {on === false && mode === 'paper' && m.live ? <span className="muted small"> (실제 매매 중이라 가상 매수 안 함)</span> : null}
                    {m.pending ? <span className="muted small"> · 재시작 대기</span> : null}
                    {!m.fromRunning && on !== null ? <span className="muted small"> · 저장된 설정</span> : null}
                  </td>
                  <td className="num">{d ? `${d.buys}건` : '-'}</td>
                  <td className="num">
                    {d ? `${d.sells}건` : '-'}
                    {openN > 0 ? <span className="muted small"> · 보유 {openN}</span> : null}
                  </td>
                  <td className="num">{d && d.sells ? `${Math.round((d.wins / d.sells) * 100)}%` : '-'}</td>
                  <td className={`num ${pnlClass(d && d.sells ? d.pnlKrw : undefined)}`}>{d && d.sells ? fmtKrw(d.pnlKrw) : '-'}</td>
                  <td className="muted small">{!isDay ? '아래 날짜별 표' : mode === 'live' ? (sel ? '아래 표시 중 ▼' : '이 탭 (누르면 거름)') : '가상 매매 탭 →'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {period === 'day' && (
        <div className="strat-bar">
          <div className="seg" title="아래 요약 · 차트 · 내역에 보일 실제 매매">
            <button type="button" className={filter === 'all' ? 'on' : ''} onClick={() => setFilter('all')}>
              전체 {allTrades.length}
            </button>
            {TRADE_STRATEGIES.map((s) => (
              <button key={s} type="button" className={filter === s ? 'on' : ''} onClick={() => setFilter(s)}>
                {TRADE_STRATEGY_SHORT[s]} {byStrategy[s].buys + byStrategy[s].sells}
              </button>
            ))}
          </div>
          <span className="muted small">
            실제 주문 내역{filter === 'all' ? ' 전체' : `: ${TRADE_STRATEGY_KO[filter]}`} · 현재 모드 - {TRADE_STRATEGIES.map((s) => `${TRADE_STRATEGY_SHORT[s]} ${strategyModeKo(modes[s])}`).join(' · ')}
          </span>
        </div>
        )}
      </div>
      {period !== 'day' && (
        <div className="table-wrap range-days">
          <table className="grid">
            <thead>
              <tr>
                <th>날짜</th>
                <th className="num">규칙 (실제)</th>
                <th className="num">Edge AI (실제)</th>
                <th className="num">Edge AI (가상)</th>
                <th className="num">VWAP (실제)</th>
                <th className="num">VWAP (가상)</th>
                <th className="num" title="규칙 + Edge AI + VWAP 의 실제 매매 손익">실제 합계</th>
              </tr>
            </thead>
            <tbody>
              {(rangeDays || []).map((d) => {
                const r = { rule: realOf(d.real.rule), edge: realOf(d.real.edge), vwap: realOf(d.real.vwap) };
                const tot = { buys: r.rule.buys + r.edge.buys + r.vwap.buys, sells: r.rule.sells + r.edge.sells + r.vwap.sells, pnlKrw: r.rule.pnlKrw + r.edge.pnlKrw + r.vwap.pnlKrw };
                return (
                  <tr key={d.date} className={d.date === day ? 'sel' : ''} title="누르면 이 날짜를 열어 일간 내역을 봅니다" onClick={() => openDay(d.date)}>
                    <td className="mono">{ymdDash(d.date)}</td>
                    {cell(r.rule)}
                    {cell(r.edge)}
                    {cell(d.paper.edge)}
                    {cell(r.vwap)}
                    {cell(d.paper.vwap)}
                    {cell(tot)}
                  </tr>
                );
              })}
              {rangeDays && rangeSum && rangeDays.length > 0 && (() => {
                const r = { rule: realOf(rangeSum.real.rule), edge: realOf(rangeSum.real.edge), vwap: realOf(rangeSum.real.vwap) };
                const tot = { buys: r.rule.buys + r.edge.buys + r.vwap.buys, sells: r.rule.sells + r.edge.sells + r.vwap.sells, pnlKrw: r.rule.pnlKrw + r.edge.pnlKrw + r.vwap.pnlKrw };
                return (
                  <tr className="total">
                    <td>
                      <b>합계</b>
                    </td>
                    {cell(r.rule)}
                    {cell(r.edge)}
                    {cell(rangeSum.paper.edge)}
                    {cell(r.vwap)}
                    {cell(rangeSum.paper.vwap)}
                    {cell(tot)}
                  </tr>
                );
              })()}
              {(!rangeDays || !rangeDays.length) && (
                <tr>
                  <td colSpan={7} className="muted center">
                    {rangeBusy ? '읽는 중…' : !range ? '기간을 정해 주세요.' : rangeDays ? '이 기간에 로그가 있는 날이 없습니다.' : '불러오지 못했습니다.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <p className="muted small">칸 = 손익 · 매수 / 매도(청산) 건수. 실제 = 로그의 실현 손익{net ? ' (제세금 포함)' : ''}, 가상 = 1회 매수 상한 금액으로 샀다고 본 순손익. 날짜를 누르면 그날 내역을 엽니다.</p>
        </div>
      )}
      {period === 'day' && (<>
      <div className="kpis">
        <div className="kpi">
          <span>매도 건수</span>
          <b>{sells.length}건</b>
        </div>
        <div className="kpi">
          <span>실현 손익 합계{net ? ' (제세금 포함)' : ''}</span>
          <b className={pnlClass(total)}>{fmtKrw(total)}</b>
        </div>
        <div className="kpi">
          <span>승률</span>
          <b>{sells.length ? `${Math.round((wins / sells.length) * 100)}%` : '-'}</b>
          <small>
            {wins}승 {sells.length - wins}패
          </small>
        </div>
        <div className="kpi">
          <span>매수 건수</span>
          <b>{trades.length - sells.length}건</b>
        </div>
        <label className="check fee-switch" title={`손익에서 예상 거래 비용을 뺍니다: ${feeLabel()} (봇 설정 r001과 같은 비율). 로그의 수익금액은 비용 전 금액입니다`}>
          <input type="checkbox" checked={net} onChange={(e) => setNet(e.target.checked)} /> 제세금 포함
        </label>
      </div>
      <div className="trade-charts">
        <div className="panel">
          <h4>누적 실현 손익 (원{net ? ', 제세금 포함' : ''})</h4>
          <div className="chart-mid">
            {cum.length === 0 ? (
              <div className="empty small">매도 없음</div>
            ) : (
              <ResponsiveContainer>
                <LineChart data={cum} margin={{ top: 12, right: 24, bottom: 4, left: 8 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} scale="time" tickFormatter={(v) => fmtKstTime(v).slice(0, 5)} stroke="var(--text-3)" fontSize={11} />
                  <YAxis tickFormatter={(v) => fmtInt(v)} stroke="var(--text-3)" fontSize={11} width={70} />
                  <ReferenceLine y={0} stroke="var(--text-3)" />
                  <Tooltip
                    isAnimationActive={false}
                    content={({ active, payload }) =>
                      active && payload?.length ? (
                        <div className="tip">
                          <b>{fmtKstTime(payload[0].payload.x)}</b> 누적 {fmtKrw(payload[0].payload.y)}
                          {payload[0].payload.t && <div>
                            {payload[0].payload.t.code} {payload[0].payload.t.name} {fmtKrw(payload[0].payload.t.pnlKrw)} · {payload[0].payload.t.reason}
                          </div>}
                        </div>
                      ) : null
                    }
                  />
                  <Line type="stepAfter" dataKey="y" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, strokeWidth: 2, fill: 'var(--surface)' }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
        <div className="panel">
          <h4>매도 조건별 손익 합계 (원{net ? ', 제세금 포함' : ''})</h4>
          <div className="chart-mid">
            {byCond.length === 0 ? (
              <div className="empty small">매도 없음</div>
            ) : (
              <ResponsiveContainer>
                <BarChart data={byCond} layout="vertical" margin={{ top: 8, right: 90, bottom: 4, left: 8 }} barCategoryGap={6}>
                  <CartesianGrid stroke="var(--grid)" horizontal={false} />
                  <XAxis type="number" tickFormatter={(v) => fmtInt(v)} stroke="var(--text-3)" fontSize={11} />
                  <YAxis type="category" dataKey="label" width={170} stroke="var(--text-3)" fontSize={11} />
                  <ReferenceLine x={0} stroke="var(--text-3)" />
                  <Tooltip
                    isAnimationActive={false}
                    cursor={{ fill: 'var(--hover)' }}
                    content={({ active, payload }) =>
                      active && payload?.length ? (
                        <div className="tip">
                          <b>{payload[0].payload.label}</b> {payload[0].payload.n}건 · {fmtKrw(payload[0].payload.sum)}
                        </div>
                      ) : null
                    }
                  />
                  <Bar dataKey="sum" radius={4} isAnimationActive={false}>
                    {byCond.map((e) => (
                      <Cell key={e.key} fill={e.sum >= 0 ? 'var(--gain)' : 'var(--loss)'} />
                    ))}
                    <LabelList dataKey="sum" position="right" formatter={(v: number) => fmtKrw(v)} fill="var(--text-2)" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>시각</th>
              <th>종목</th>
              <th>방식</th>
              <th>구분</th>
              <th className="num">가격</th>
              <th className="num">수량</th>
              <th>매도 조건</th>
              <th>사유</th>
              <th className="num">손익{net ? ' (순)' : ''}</th>
              <th className="num">수익률{net ? ' (순)' : ''}</th>
            </tr>
          </thead>
          <tbody>
            {[...trades.filter((t) => t.side === 'BUY'), ...sells].sort((a, b) => a.ts - b.ts).map((t, i) => (
              <tr
                key={i}
                title={t.raw}
                onClick={() => {
                  select(t.code);
                  setTab('timeline');
                }}
              >
                <td className="mono">{fmtKstTime(t.ts)}</td>
                <td>
                  {t.code} {t.name} <NxtTag code={t.code} />
                </td>
                <td>
                  <span className={`tag strat-${strategyOfTrade(t)}`}>{TRADE_STRATEGY_SHORT[strategyOfTrade(t)]}</span>
                </td>
                <td>{t.side === 'BUY' ? '매수' : '매도'}</td>
                <td className="num">{fmtInt(t.price)}</td>
                <td className="num">{fmtInt(t.qty)}</td>
                <td>{t.side === 'SELL' ? sellLabel(t as (typeof sells)[number]) : ''}</td>
                <td className="reason">{t.reason}</td>
                <td className={`num ${pnlClass(t.pnlKrw)}`}>{t.side === 'SELL' ? fmtKrw(t.pnlKrw) : ''}</td>
                <td className={`num ${pnlClass(t.pnlPct)}`}>{t.side === 'SELL' ? fmtPct(t.pnlPct) : ''}</td>
              </tr>
            ))}
            {!trades.length && (
              <tr>
                <td colSpan={10} className="muted center">
                  {filter === 'all' ? '거래가 없습니다.' : `${TRADE_STRATEGY_KO[filter]}의 실제 거래가 없습니다.`}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      </>)}
    </div>
  );
}
