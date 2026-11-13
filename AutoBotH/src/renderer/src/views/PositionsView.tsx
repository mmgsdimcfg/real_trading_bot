/*
 * Update log
 * - [2026-11-11] type=feat - 보유 현황 tab: every symbol with an open position (snapshot positions from the buy_sell log) with the
 *   buy time / price / quantity, the current price (market:quote per held code, refreshed every 10 s while the tab is open), the
 *   unrealized P&L in KRW and percent, the holding time and the last sell evaluation. KPIs on top (positions, invested amount,
 *   unrealized total, today's realized total). "제세금 포함" (shared with the trade-results tab) subtracts the estimated
 *   commission + sell tax (shared/fees). A row opens the symbol on the timeline
 * - [2026-11-13] type=feat - the whole account, split by who bought it: rows come from r016 `holdings` (the r003 account sync +
 *   the r023 ledger) - every held symbol with its quantity as 봇 / 직접(MTS) / 혼합, the two quantities, the account's average
 *   price. Filter 전체 / 봇 매수 / 직접 매수 (a mixed holding shows in both with that side's quantity). Symbols the bot bought
 *   today keep their buy time and last sell evaluation from the log. Falls back to the log-only view (bot buys of today) when the
 *   server has no holdings snapshot yet (older server code, or r003 not started since the update)
 */

import { useEffect, useMemo, useState } from 'react';
import { NxtMark } from '../components/NxtTag';
import { useStore } from '../store';
import { condTitle, fmtInt, fmtKrw, fmtPct, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import { feeLabel, pnlKrw, pnlPct } from '../../../shared/fees';
import type { MarketQuote } from '../../../shared/price';
import type { HoldingsDoc } from '../../../shared/control';
import type { SymbolSummary } from '../../../shared/types';

const QUOTE_EVERY_MS = 10000;
const HOLDINGS_EVERY_MS = 30000;

type Origin = 'bot' | 'manual' | 'mixed';
type OriginFilter = 'all' | 'bot' | 'manual';
const ORIGIN_KO: Record<Origin, string> = { bot: '봇', manual: '직접', mixed: '혼합' };
const ORIGIN_TITLE: Record<Origin, string> = {
  bot: 'AutoBot 이 매수한 수량입니다',
  manual: 'MTS / HTS 등에서 직접 매수한 수량입니다 (봇 매수 주문 없이 계좌에 늘어난 수량)',
  mixed: '같은 종목을 봇과 직접 매수로 함께 보유 중입니다',
};

interface Row {
  code: string;
  name: string;
  nxt: boolean;
  origin: Origin;
  qty: number;
  botQty: number;
  manualQty: number;
  buyPrice: number;
  buyTs?: number;
  sym?: SymbolSummary;
}

const holdText = (buyTs: number, now: number) => {
  const s = Math.max(0, Math.floor((now - buyTs) / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}시간 ${m}분` : `${m}분`;
};

const ageText = (sec: number | null | undefined) => {
  if (sec === null || sec === undefined) return '';
  if (sec < 90) return `${Math.round(sec)}초 전`;
  if (sec < 5400) return `${Math.round(sec / 60)}분 전`;
  return `${Math.round(sec / 3600)}시간 전`;
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
  const [doc, setDoc] = useState<HoldingsDoc | null>(null);
  const [docErr, setDocErr] = useState<string | undefined>();
  const [filter, setFilter] = useState<OriginFilter>('all');
  const connected = status.conn === 'connected';

  // account holdings with the bot's / hand-bought quantity (r016 holdings: a file r003 writes at every account sync)
  useEffect(() => {
    if (!connected) return;
    let alive = true;
    const load = async () => {
      const r = await window.api.ctlHoldings();
      if (!alive) return;
      if (r.ok && r.data) {
        setDoc(r.data);
        setDocErr(undefined);
      } else setDocErr(r.error || '보유 현황을 읽지 못했습니다');
    };
    void load();
    const t = setInterval(() => void load(), HOLDINGS_EVERY_MS);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [connected]);

  const symByCode = useMemo(() => new Map((snap?.symbols || []).map((s) => [s.code, s])), [snap]);
  const fromAccount = !!doc?.available;

  const allRows = useMemo<Row[]>(() => {
    if (doc?.available) {
      return doc.holdings
        .filter((h) => h.qty > 0)
        .map((h) => {
          const sym = symByCode.get(h.code);
          const origin: Origin = h.bot_qty <= 0 ? 'manual' : h.manual_qty <= 0 ? 'bot' : 'mixed';
          const t = h.buy_time ? Date.parse(h.buy_time) : NaN;
          return {
            code: h.code,
            name: h.name || sym?.name || '',
            nxt: !!sym?.nxt,
            origin,
            qty: h.qty,
            botQty: h.bot_qty,
            manualQty: h.manual_qty,
            buyPrice: h.avg_price,
            buyTs: sym?.position?.buyTs ?? (Number.isFinite(t) ? t : undefined),
            sym,
          };
        });
    }
    // no account snapshot on this server: today's bot buys from the trade log (the view before 2026-11-13)
    return (snap?.symbols || [])
      .filter((s) => s.position && s.position.qty > 0)
      .map((s) => ({
        code: s.code,
        name: s.name,
        nxt: !!s.nxt,
        origin: 'bot' as Origin,
        qty: s.position!.qty,
        botQty: s.position!.qty,
        manualQty: 0,
        buyPrice: s.position!.buyPrice,
        buyTs: s.position!.buyTs,
        sym: s,
      }));
  }, [doc, snap, symByCode]);

  const codesKey = allRows.map((r) => r.code).join(',');

  // current prices: one market:quote per held code, every 10 s while this tab is open (main caches 8 s in market hours)
  useEffect(() => {
    if (!codesKey || !connected) return;
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
  }, [codesKey, connected]);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 30000);
    return () => clearInterval(t);
  }, []);

  // the quantity a row counts for: the whole holding, or only the bot's / the hand-bought part under that filter
  const rows = useMemo(
    () =>
      allRows
        .map((r) => ({ r, shown: filter === 'bot' ? r.botQty : filter === 'manual' ? r.manualQty : r.qty }))
        .filter((x) => x.shown > 0)
        .map(({ r, shown }) => {
          const q = quotes[r.code]?.q;
          const live = q?.price && q.price > 0 ? q.price : undefined;
          const snapPrice = doc?.available ? doc.holdings.find((h) => h.code === r.code)?.current_price : undefined;
          const price = live ?? (snapPrice && snapPrice > 0 ? snapPrice : undefined);
          const krw = price !== undefined && r.buyPrice > 0 ? pnlKrw(r.buyPrice, price, shown, net) : undefined;
          const pct = price !== undefined && r.buyPrice > 0 ? pnlPct(r.buyPrice, price, shown, net) : undefined;
          return { r, shown, price, live: live !== undefined, krw, pct, invested: r.buyPrice * shown, err: quotes[r.code]?.error };
        }),
    [allRows, filter, quotes, net, doc],
  );
  const invested = rows.reduce((a, x) => a + x.invested, 0);
  const priced = rows.filter((x) => x.krw !== undefined);
  const unrealized = priced.reduce((a, x) => a + (x.krw || 0), 0);
  const realized = (snap?.trades || []).filter((t) => t.side === 'SELL').reduce((a, t) => a + (net && t.buyPrice && t.price && t.qty ? pnlKrw(t.buyPrice, t.price, t.qty, true) : t.pnlKrw || 0), 0);
  const quoteErr = rows.find((x) => x.err)?.err;
  const botShares = allRows.reduce((a, r) => a + (r.botQty > 0 ? 1 : 0), 0);
  const manualShares = allRows.reduce((a, r) => a + (r.manualQty > 0 ? 1 : 0), 0);
  const stale = fromAccount && (!doc!.running || (doc!.age_seconds ?? 0) > 300);

  return (
    <div className="trades positions">
      <div className="kpis">
        <div className="kpi">
          <span>보유 종목</span>
          <b>{rows.length}종목</b>
          {fromAccount && (
            <small>
              봇 {botShares} · 직접 {manualShares}
            </small>
          )}
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
        {fromAccount && (
          <label title="봇 매수 / 직접 매수는 수량 기준입니다. 같은 종목을 함께 보유하면 양쪽에 그 수량만큼 나옵니다.">
            구분{' '}
            <select value={filter} onChange={(e) => setFilter(e.target.value as OriginFilter)}>
              <option value="all">전체</option>
              <option value="bot">봇 매수</option>
              <option value="manual">직접 매수 (MTS)</option>
            </select>
          </label>
        )}
        <label className="check fee-switch" title={`평가 손익에서 예상 거래 비용을 뺍니다: ${feeLabel()} (봇 설정 r001과 같은 비율)`}>
          <input type="checkbox" checked={net} onChange={(e) => setNet(e.target.checked)} /> 제세금 포함
        </label>
      </div>
      {fromAccount && (
        <p className={stale ? 'badge-warn' : 'muted small'} style={{ margin: '0 16px 6px' }}>
          계좌 보유 기준 {doc!.updated_at?.replace('T', ' ')} ({ageText(doc!.age_seconds)})
          {!doc!.running ? ' · r003 이 꺼져 있어 마지막으로 읽은 계좌 상태입니다' : stale ? ' · 계좌 동기화가 늦고 있습니다' : ''}
          {doc!.auto_sell_scope ? ` · 자동 매도 대상: ${doc!.auto_sell_scope === 'all' ? '직접 매수분 포함' : '봇 매수분만'}` : ''}
        </p>
      )}
      {!fromAccount && connected && (
        <p className="muted small" style={{ padding: '0 16px' }}>
          {docErr
            ? `계좌 보유 현황을 읽지 못해 오늘 봇 매수분만 표시합니다: ${docErr}`
            : doc
              ? '계좌 보유 스냅샷이 아직 없습니다 (r003 을 이 버전으로 한 번 시작하면 직접 매수분까지 표시됩니다). 지금은 오늘 봇 매수분만 표시합니다.'
              : ''}
        </p>
      )}
      {quoteErr && <p className="muted small" style={{ padding: '0 16px' }}>현재가: {quoteErr}</p>}
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>종목</th>
              <th>구분</th>
              <th>매수 시각</th>
              <th>보유 시간</th>
              <th className="num">{fromAccount ? '평균 매수가' : '매수가'}</th>
              <th className="num">수량</th>
              <th className="num">현재가</th>
              <th className="num">평가 손익{net ? ' (순)' : ''}</th>
              <th className="num">수익률</th>
              <th>마지막 매도 평가</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ r, shown, price, live, krw, pct }) => {
              const today = r.buyTs !== undefined && new Date(r.buyTs).toDateString() === new Date(now).toDateString();
              const ls = r.sym?.lastSell;
              return (
                <tr
                  key={r.code}
                  onClick={() => {
                    select(r.code);
                    setTab('timeline');
                  }}
                >
                  <td>
                    <b>{r.code}</b> {r.name} {r.nxt && <NxtMark />}
                  </td>
                  <td title={ORIGIN_TITLE[r.origin]}>
                    <span className={`origin origin-${r.origin}`}>{ORIGIN_KO[r.origin]}</span>
                  </td>
                  <td className="mono">{r.buyTs === undefined ? '-' : today ? fmtKstTime(r.buyTs) : '이전 거래일'}</td>
                  <td>{r.buyTs !== undefined ? holdText(r.buyTs, now) : '-'}</td>
                  <td className="num">{fmtInt(r.buyPrice)}</td>
                  <td className="num" title={r.origin === 'mixed' ? `계좌 ${r.qty}주 = 봇 ${r.botQty}주 + 직접 ${r.manualQty}주` : ''}>
                    {fmtInt(shown)}
                    {r.origin === 'mixed' && filter === 'all' && (
                      <small className="muted">
                        {' '}
                        (봇 {fmtInt(r.botQty)} · 직접 {fmtInt(r.manualQty)})
                      </small>
                    )}
                  </td>
                  <td className="num" title={live ? (quotes[r.code]?.q?.asof ? `기준 ${quotes[r.code]!.q!.asof}` : '') : price !== undefined ? '계좌 동기화 시점의 가격' : ''}>
                    {price !== undefined ? fmtInt(price) : '-'}
                  </td>
                  <td className={`num ${pnlClass(krw)}`}>{fmtKrw(krw)}</td>
                  <td className={`num ${pnlClass(pct)}`}>{fmtPct(pct)}</td>
                  <td className="reason" title={ls?.raw || ''}>
                    {ls ? `${fmtKstTime(ls.ts)} ${condTitle(conditions, 'sell', ls.condKey)}${ls.reason ? ` · ${ls.reason}` : ''}` : '-'}
                  </td>
                </tr>
              );
            })}
            {!rows.length && (
              <tr className="nohover">
                <td colSpan={10} className="muted center">
                  {filter === 'all' ? '보유 중인 종목이 없습니다.' : filter === 'bot' ? '봇이 매수한 보유 종목이 없습니다.' : '직접 매수한 보유 종목이 없습니다.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ padding: '8px 16px' }}>
        {fromAccount
          ? '보유 수량과 평균 매수가는 계좌 조회(r003 계좌 동기화) 기준입니다. 봇 / 직접 구분은 수량 기준이며, 봇 매수 주문이 걸려 있던 동안 늘어난 수량을 봇 매수분으로 봅니다. 같은 종목을 함께 보유하면 평균 매수가는 두 매수분을 합친 값입니다. '
          : '보유 수량과 매수가는 매매 로그(buy_sell)의 체결 기준입니다. '}
        현재가는 KIS 현재가 조회(10초마다)이며, 제세금은 {feeLabel()} 로 추정합니다.
      </p>
    </div>
  );
}
