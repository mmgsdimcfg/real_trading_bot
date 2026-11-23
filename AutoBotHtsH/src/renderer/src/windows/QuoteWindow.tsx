// Update log
// - [2026-11-06] type=fix (Trades showed twice (realtime + REST copy): the REST list now comes from 'ticks' (accumulated volume = exact
//   match with realtime) instead of inquire_ccnl, whose strength differs from the realtime CTTR)
// - [2026-11-06] type=fix (Trade list: the cached trades (now up to 30,000 with the tick chart back-fill) and the polled REST list are
//   merged and the newest 100 shown - REST trades newer than the cache were dropped before)
// - [2026-11-06] type=feat (☆ button next to the code: add / remove the symbol in the watchlist groups)
// - [2026-11-06] type=fix (No more blinking: the window kept going blank while the realtime state changed (see hooks.ts useCall); the
//   loading screen is now shown only until the first quote of the code arrives)
// - [2026-11-06] type=feat (Current price window: price / change / rate with KRX sign colours, VI / halt / warning badges, info grid (open
//   high low, upper / lower limit, volume and amount, volume vs previous day, 52-week and yearly high / low, PER PBR EPS BPS, market cap,
//   listed shares, foreign ownership, tick size, pivot points), 10-level order book with remaining-quantity bars, changes and the expected
//   price, tabs: trades (realtime + latest 30 from REST, 체결강도), daily prices (foreign net buying), investors (individual / foreign /
//   institution). Realtime trades + order book over the websocket; REST polling as fallback (3-5 s) when realtime is not connected)

import { useMemo, useState } from 'react';
import { useStore, type Win } from '../store';
import { mergeTrades } from '../../../shared/live';
import { useCall, useRealtime, useSymbol } from '../hooks';
import { WatchStar } from './ListWindows';
import { NxtMark } from '../components/NxtTag';
import type { OrderBook, Quote, TradeTick } from '../../../shared/kis';
import type { DailyRow, TicksRes } from '../../../shared/market';
import { InvestorPanel } from '../components/InvestorPanel';
import { arrow, fmtInt, fmtKrwBig, fmtNum, fmtPct, signOf, upDown } from '../lib';

const NO_TRADES: TradeTick[] = [];
const hms = (s?: string) => (s && s.length >= 6 ? `${s.slice(0, 2)}:${s.slice(2, 4)}:${s.slice(4, 6)}` : s || '');
const ymd = (s?: string) => (s && s.length === 8 ? `${s.slice(2, 4)}.${s.slice(4, 6)}.${s.slice(6, 8)}` : s || '');

export function cls(v: number | undefined, base?: number) {
  if (v === undefined || base === undefined || !Number.isFinite(v) || !Number.isFinite(base)) return '';
  return v > base ? 'up' : v < base ? 'down' : '';
}

export function OrderBookView({ ob, price, prevClose }: { ob: OrderBook; price?: number; prevClose?: number }) {
  const asks = ob.asks.slice(0, 10).reverse();
  const bids = ob.bids.slice(0, 10);
  const max = Math.max(1, ...ob.asks.map((a) => a[1]), ...ob.bids.map((b) => b[1]));
  const pct = (p: number) => (prevClose ? ((p - prevClose) / prevClose) * 100 : undefined);
  const row = (p: number, q: number, d: number, side: 'a' | 'b', key: string) => (
    <tr key={key} className={p === price ? 'cur' : ''}>
      <td className="ob-q ask">
        {side === 'a' && (
          <>
            <i style={{ width: `${(q / max) * 100}%` }} />
            <span>{fmtInt(q)}</span>
          </>
        )}
      </td>
      <td className="ob-d">{side === 'a' && d ? <small className={d > 0 ? 'up' : 'down'}>{d > 0 ? '+' : ''}{fmtInt(d)}</small> : null}</td>
      <td className={`ob-p ${cls(p, prevClose)}`}>
        {fmtInt(p)}
        {pct(p) !== undefined && <small> {fmtPct(pct(p), 2)}</small>}
      </td>
      <td className="ob-d">{side === 'b' && d ? <small className={d > 0 ? 'up' : 'down'}>{d > 0 ? '+' : ''}{fmtInt(d)}</small> : null}</td>
      <td className="ob-q bid">
        {side === 'b' && (
          <>
            <i style={{ width: `${(q / max) * 100}%` }} />
            <span>{fmtInt(q)}</span>
          </>
        )}
      </td>
    </tr>
  );
  return (
    <table className="ob">
      <tbody>
        {asks.map(([p, q, d], i) => row(p, q, d, 'a', `a${i}`))}
        {bids.map(([p, q, d], i) => row(p, q, d, 'b', `b${i}`))}
        <tr className="ob-total">
          <td className="ob-q ask">{fmtInt(ob.totalAsk)}</td>
          <td />
          <td className="center muted">{ob.expected ? `예상 ${fmtInt(ob.expected.price)}` : '잔량 합계'}</td>
          <td />
          <td className="ob-q bid">{fmtInt(ob.totalBid)}</td>
        </tr>
      </tbody>
    </table>
  );
}

export function QuoteWindow({ win }: { win: Win }) {
  const code = win.code;
  const sym = useSymbol(code);
  const rtOn = useStore((s) => s.auth?.rt.status === 'on');
  useRealtime(code, { trades: true, book: true });
  const rq = useStore((s) => s.rt.quotes[code]);
  const rbook = useStore((s) => s.rt.books[code]);
  const rtrades = useStore((s) => s.rt.trades[code]) || NO_TRADES;
  const q = useCall<Quote>('quote', [code], rtOn ? 30000 : 5000);
  const ob = useCall<OrderBook>('orderbook', [code], rtOn && rbook ? 0 : 3000);
  // newest 30 trades by time (inquire_time_itemconclusion): they carry the accumulated volume, so they match realtime trades exactly
  // (inquire_ccnl's strength differs from the realtime one and its trades showed up twice)
  const tr = useCall<TicksRes>('ticks', [{ code, pages: 1 }], rtOn ? 0 : 5000);
  const [tab, setTab] = useState<'trades' | 'daily' | 'investor'>('trades');

  const base = q.data;
  // realtime fields newer than the REST quote win
  const live = rq && base && rq.at > q.at ? { ...base, ...rq } : base;
  const book = rbook && rbook.at > ob.at ? rbook : ob.data;
  // cached trades (realtime + tick chart back-fill, up to 30,000) and the polled REST list merged, newest 100 shown
  const trades = useMemo(() => mergeTrades(rtrades.slice(0, 100), tr.data?.trades || []).slice(0, 100), [rtrades, tr.data]);
  const s = live ? signOf(live.sign, live.change) : 0;
  const strength = trades.find((t) => t.strength !== undefined)?.strength;

  // loading / error screens only before the first quote of this code (later errors keep the last data on screen)
  if (!live) return <div className="empty">{q.error || '불러오는 중…'}</div>;
  const prev = live.prevClose;
  return (
    <div className="quote-win">
      <div className="q-head">
        <div>
          <b className="q-name">{sym?.name || live.raw?.hts_kor_isnm || code}</b> <span className="mono muted">{code}</span> {sym?.nxt && <NxtMark />} <WatchStar code={code} />{' '}
          <span className="muted small">
            {live.market} {live.sector}
          </span>
          {live.vi && <b className="flag">VI</b>}
          {live.halted && <b className="flag">거래정지</b>}
          {live.warn && <b className="flag warn">{live.warn}</b>}
        </div>
        <div className={`q-price ${upDown(s)}`}>
          {fmtInt(live.price)}
          <span className="q-chg">
            {arrow(live.sign, live.change)} {fmtInt(Math.abs(live.change))} {fmtPct(live.changePct)}
          </span>
        </div>
      </div>
      <div className="q-main">
        <div className="q-grid">
          <span>시가</span>
          <b className={cls(live.open, prev)}>{fmtInt(live.open)}</b>
          <span>고가</span>
          <b className={cls(live.high, prev)}>{fmtInt(live.high)}</b>
          <span>저가</span>
          <b className={cls(live.low, prev)}>{fmtInt(live.low)}</b>
          <span>전일 종가</span>
          <b>{fmtInt(prev)}</b>
          <span>상한가</span>
          <b className="up">{fmtInt(live.upper)}</b>
          <span>하한가</span>
          <b className="down">{fmtInt(live.lower)}</b>
          <span>거래량</span>
          <b>{fmtInt(live.volume)}</b>
          <span>거래대금</span>
          <b>{fmtKrwBig(live.amount)}</b>
          <span>전일 대비 거래량</span>
          <b>{fmtNum(live.volRatio)}%</b>
          <span>체결강도</span>
          <b className={strength !== undefined ? (strength >= 100 ? 'up' : 'down') : ''}>{strength !== undefined ? `${fmtNum(strength)}%` : '-'}</b>
          <span>52주 최고</span>
          <b>{fmtInt(live.w52High)}</b>
          <span>52주 최저</span>
          <b>{fmtInt(live.w52Low)}</b>
          <span>연중 최고 / 최저</span>
          <b>
            {fmtInt(live.yearHigh)} / {fmtInt(live.yearLow)}
          </b>
          <span>PER / PBR</span>
          <b>
            {fmtNum(live.per)} / {fmtNum(live.pbr)}
          </b>
          <span>EPS / BPS</span>
          <b>
            {fmtInt(live.eps)} / {fmtInt(live.bps)}
          </b>
          <span>시가총액</span>
          <b>{live.marketCapEok !== undefined ? fmtKrwBig(live.marketCapEok * 1e8) : '-'}</b>
          <span>상장주식수</span>
          <b>{fmtInt(live.listed)}</b>
          <span>외국인 소진율</span>
          <b>{fmtNum(live.foreignRate)}%</b>
          <span>호가단위</span>
          <b>{fmtInt(live.tickUnit)}</b>
          {live.pivot?.p ? (
            <>
              <span title="피봇: 2차저항 / 1차저항 / 피봇 / 1차지지 / 2차지지">피봇</span>
              <b className="small">
                {fmtInt(live.pivot.r2)} / {fmtInt(live.pivot.r1)} / <u>{fmtInt(live.pivot.p)}</u> / {fmtInt(live.pivot.s1)} / {fmtInt(live.pivot.s2)}
              </b>
            </>
          ) : null}
        </div>
        <div className="q-book">
          {book ? <OrderBookView ob={book} price={live.price} prevClose={prev} /> : <div className="empty small">{ob.error || '호가 불러오는 중…'}</div>}
        </div>
      </div>
      <div className="q-tabs">
        <span className="seg">
          <button className={tab === 'trades' ? 'on' : ''} onClick={() => setTab('trades')}>
            체결
          </button>
          <button className={tab === 'daily' ? 'on' : ''} onClick={() => setTab('daily')}>
            일별
          </button>
          <button className={tab === 'investor' ? 'on' : ''} onClick={() => setTab('investor')}>
            투자자
          </button>
        </span>
        <span className="muted small">{rtOn ? '실시간' : '주기 조회'}</span>
      </div>
      <div className="q-tab-body">
        {tab === 'trades' && (
          <table className="grid">
            <thead>
              <tr>
                <th>시각</th>
                <th className="num">체결가</th>
                <th className="num">대비</th>
                <th className="num">체결량</th>
                <th className="num">체결강도</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i}>
                  <td className="mono">{hms(t.time)}</td>
                  <td className={`num ${cls(t.price, prev)}`}>{fmtInt(t.price)}</td>
                  <td className={`num ${upDown(Math.sign(t.change))}`}>{fmtInt(t.change)}</td>
                  <td className={`num ${t.side === 1 ? 'up' : t.side === 5 ? 'down' : ''}`}>{fmtInt(t.volume)}</td>
                  <td className="num">{t.strength !== undefined ? fmtNum(t.strength) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {tab === 'daily' && <DailyTab code={code} />}
        {tab === 'investor' && <InvestorPanel win={win} inTab />}
      </div>
    </div>
  );
}

function DailyTab({ code }: { code: string }) {
  const d = useCall<DailyRow[]>('daily', [code], 60000);
  if (d.error) return <div className="empty small">{d.error}</div>;
  return (
    <table className="grid">
      <thead>
        <tr>
          <th>일자</th>
          <th className="num">종가</th>
          <th className="num">대비</th>
          <th className="num">등락률</th>
          <th className="num">거래량</th>
          <th className="num">외국인 순매수</th>
          <th className="num">외인 소진율</th>
        </tr>
      </thead>
      <tbody>
        {(d.data || []).map((r) => (
          <tr key={r.date}>
            <td className="mono">{ymd(r.date)}</td>
            <td className={`num ${upDown(Math.sign(r.change))}`}>{fmtInt(r.close)}</td>
            <td className={`num ${upDown(Math.sign(r.change))}`}>{fmtInt(r.change)}</td>
            <td className={`num ${upDown(Math.sign(r.change))}`}>{fmtPct(r.changePct)}</td>
            <td className="num">{fmtInt(r.volume)}</td>
            <td className={`num ${upDown(Math.sign(r.foreignNet || 0))}`}>{fmtInt(r.foreignNet)}</td>
            <td className="num">{fmtNum(r.foreignRate)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
