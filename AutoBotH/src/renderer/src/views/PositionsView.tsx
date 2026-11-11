/*
 * Update log
 * - [2026-11-11] type=feat - 보유 현황 tab: every symbol with an open position (snapshot positions from the buy_sell log) with the
 *   buy time / price / quantity, the current price (market:quote per held code, refreshed every 10 s while the tab is open), the
 *   unrealized P&L in KRW and percent, the holding time and the last sell evaluation. KPIs on top (positions, invested amount,
 *   unrealized total, today's realized total). "제세금 포함" (shared with the trade-results tab) subtracts the estimated
 *   commission + sell tax (shared/fees). A row opens the symbol on the timeline
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { condTitle, fmtInt, fmtKrw, fmtPct, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import { feeLabel, pnlKrw, pnlPct } from '../../../shared/fees';
import type { MarketQuote } from '../../../shared/price';

const QUOTE_EVERY_MS = 10000;

const holdText = (buyTs: number, now: number) => {
  const s = Math.max(0, Math.floor((now - buyTs) / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}시간 ${m}분` : `${m}분`;
};

export function PositionsView() {
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const status = useStore((s) => s.status);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const net = useStore((s) => s.feesIncluded);
  const setNet = useStore((s) => s.setFeesIncluded);
  const [quotes, setQuotes] = useState<Record<string, { q?: MarketQuote; error?: string; at: number }>>({});
  const [now, setNow] = useState(Date.now());

  const held = useMemo(() => (snap?.symbols || []).filter((s) => s.position && s.position.qty > 0), [snap]);
  const codesKey = held.map((s) => s.code).join(',');

  // current prices: one market:quote per held code, every 10 s while this tab is open (main caches 8 s in market hours)
  useEffect(() => {
    if (!codesKey || status.conn !== 'connected') return;
    let alive = true;
    const load = async () => {
      for (const code of codesKey.split(',')) {
        const r = await window.api.getQuote(code);
        if (!alive) return;
        setQuotes((m) => ({ ...m, [code]: { q: r.ok ? r.data : undefined, error: r.ok ? undefined : r.error, at: Date.now() } }));
      }
      setNow(Date.now());
    };
    void load();
    const t = setInterval(() => void load(), QUOTE_EVERY_MS);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [codesKey, status.conn]);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 30000);
    return () => clearInterval(t);
  }, []);

  const rows = useMemo(
    () =>
      held.map((s) => {
        const p = s.position!;
        const q = quotes[s.code]?.q;
        const price = q?.price && q.price > 0 ? q.price : undefined;
        const krw = price !== undefined ? pnlKrw(p.buyPrice, price, p.qty, net) : undefined;
        const pct = price !== undefined ? pnlPct(p.buyPrice, price, p.qty, net) : undefined;
        return { s, p, price, krw, pct, invested: p.buyPrice * p.qty, err: quotes[s.code]?.error };
      }),
    [held, quotes, net],
  );
  const invested = rows.reduce((a, r) => a + r.invested, 0);
  const priced = rows.filter((r) => r.krw !== undefined);
  const unrealized = priced.reduce((a, r) => a + (r.krw || 0), 0);
  const realized = (snap?.trades || []).filter((t) => t.side === 'SELL').reduce((a, t) => a + (net && t.buyPrice && t.price && t.qty ? pnlKrw(t.buyPrice, t.price, t.qty, true) : t.pnlKrw || 0), 0);
  const quoteErr = rows.find((r) => r.err)?.err;

  return (
    <div className="trades positions">
      <div className="kpis">
        <div className="kpi">
          <span>보유 종목</span>
          <b>{rows.length}종목</b>
        </div>
        <div className="kpi">
          <span>투자 금액 (매수가 기준)</span>
          <b>{fmtInt(invested)}원</b>
        </div>
        <div className="kpi">
          <span>평가 손익 합계{net ? ' (제세금 포함)' : ''}</span>
          <b className={pnlClass(unrealized)}>{priced.length ? fmtKrw(unrealized) : '-'}</b>
          {priced.length < rows.length && <small>{rows.length - priced.length}종목 현재가 없음</small>}
        </div>
        <div className="kpi">
          <span>오늘 실현 손익{net ? ' (제세금 포함)' : ''}</span>
          <b className={pnlClass(realized)}>{fmtKrw(realized)}</b>
        </div>
        <label className="check fee-switch" title={`평가 손익에서 예상 거래 비용을 뺍니다: ${feeLabel()} (봇 설정 r001과 같은 비율)`}>
          <input type="checkbox" checked={net} onChange={(e) => setNet(e.target.checked)} /> 제세금 포함
        </label>
      </div>
      {quoteErr && <p className="muted small" style={{ padding: '0 16px' }}>현재가: {quoteErr}</p>}
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>종목</th>
              <th>매수 시각</th>
              <th>보유 시간</th>
              <th className="num">매수가</th>
              <th className="num">수량</th>
              <th className="num">현재가</th>
              <th className="num">평가 손익{net ? ' (순)' : ''}</th>
              <th className="num">수익률</th>
              <th>마지막 매도 평가</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ s, p, price, krw, pct }) => (
              <tr
                key={s.code}
                onClick={() => {
                  select(s.code);
                  setTab('timeline');
                }}
              >
                <td>
                  <b>{s.code}</b> {s.name}
                </td>
                <td className="mono">{fmtKstTime(p.buyTs)}</td>
                <td>{holdText(p.buyTs, now)}</td>
                <td className="num">{fmtInt(p.buyPrice)}</td>
                <td className="num">{fmtInt(p.qty)}</td>
                <td className="num" title={quotes[s.code]?.q?.asof ? `기준 ${quotes[s.code]!.q!.asof}` : ''}>
                  {price !== undefined ? fmtInt(price) : '-'}
                </td>
                <td className={`num ${pnlClass(krw)}`}>{fmtKrw(krw)}</td>
                <td className={`num ${pnlClass(pct)}`}>{fmtPct(pct)}</td>
                <td className="reason" title={s.lastSell?.raw || ''}>
                  {s.lastSell ? `${fmtKstTime(s.lastSell.ts)} ${condTitle(conditions, 'sell', s.lastSell.condKey)}${s.lastSell.reason ? ` · ${s.lastSell.reason}` : ''}` : '-'}
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr className="nohover">
                <td colSpan={9} className="muted center">
                  보유 중인 종목이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ padding: '8px 16px' }}>
        보유 수량과 매수가는 매매 로그(buy_sell)의 체결 기준입니다. 현재가는 KIS 현재가 조회(10초마다)이며, 제세금은 {feeLabel()} 로 추정합니다.
      </p>
    </div>
  );
}
