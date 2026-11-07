/*
 * Update log
 * - [2026-11-04] type=feat - Trade results from YYYYMMDD_buy_sell.log: headline numbers, cumulative realized P&L over the day,
 *   P&L by sell condition, and the trade table (Korean convention: gain red, loss blue)
 */

import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, LabelList } from 'recharts';
import { useStore } from '../store';
import { condTitle, fmtInt, fmtKrw, fmtPct, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';

export function TradesView() {
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const trades = snap?.trades || [];
  const sells = trades.filter((t) => t.side === 'SELL');
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
    const m = new Map<string, { key: string; sum: number; n: number }>();
    for (const t of sells) {
      const k = t.condKey || 'UNKNOWN';
      const e = m.get(k) || { key: k, sum: 0, n: 0 };
      e.sum += t.pnlKrw || 0;
      e.n++;
      m.set(k, e);
    }
    return [...m.values()].sort((a, b) => a.sum - b.sum).map((e) => ({ ...e, label: condTitle(conditions, 'sell', e.key) }));
  }, [sells, conditions]);

  return (
    <div className="trades">
      <div className="kpis">
        <div className="kpi">
          <span>매도 건수</span>
          <b>{sells.length}건</b>
        </div>
        <div className="kpi">
          <span>실현 손익 합계</span>
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
      </div>
      <div className="trade-charts">
        <div className="panel">
          <h4>누적 실현 손익 (원)</h4>
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
          <h4>매도 조건별 손익 합계 (원)</h4>
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
              <th>구분</th>
              <th className="num">가격</th>
              <th className="num">수량</th>
              <th>매도 조건</th>
              <th>사유</th>
              <th className="num">손익</th>
              <th className="num">수익률</th>
            </tr>
          </thead>
          <tbody>
            {[...trades].sort((a, b) => a.ts - b.ts).map((t, i) => (
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
                  {t.code} {t.name}
                </td>
                <td>{t.side === 'BUY' ? '매수' : '매도'}</td>
                <td className="num">{fmtInt(t.price)}</td>
                <td className="num">{fmtInt(t.qty)}</td>
                <td>{t.side === 'SELL' ? condTitle(conditions, 'sell', t.condKey) : ''}</td>
                <td className="reason">{t.reason}</td>
                <td className={`num ${pnlClass(t.pnlKrw)}`}>{t.side === 'SELL' ? fmtKrw(t.pnlKrw) : ''}</td>
                <td className={`num ${pnlClass(t.pnlPct)}`}>{t.side === 'SELL' ? fmtPct(t.pnlPct) : ''}</td>
              </tr>
            ))}
            {!trades.length && (
              <tr>
                <td colSpan={9} className="muted center">
                  거래가 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
