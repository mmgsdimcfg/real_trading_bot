// Update log
// - [2026-11-26] type=feat (Symbol box in the header, the same as every other symbol window (user 2026-11-26))
// - [2026-11-26] type=feat (Order window 주문 (user 2026-11-26): opened from the current price window (click a price of the order book ->
//   매수 / 매도) or the toolbar. Left: the order book (a click puts that price into the form). Right: 매수 / 매도, 지정가 / 시장가,
//   price and quantity with tick / share steppers and 10 / 25 / 50 / 100 % of the possible quantity, order amount, 예수금 / 주문가능금액
//   / 매수가능수량 (buy) or 보유 / 매도가능수량 / 평균단가 (sell). Below: the open orders with 정정 (new price for the rest) and 취소.
//   The main process asks for a confirmation of every order in a native dialog; nothing is sent from here without it. No account
//   saved / guest mode / logged out: the form says why and sends nothing)
// - [2026-11-26] type=feat (잔고 tab beside 미체결 (user 2026-11-26): every holding - 보유 / 매도가능 / 평균단가 / 현재가 / 평가금액 /
//   평가손익 / 수익률 - with the account's 총평가 and summed 평가손익; a row click shows that symbol, 매도 prepares a sell at the current
//   price (nothing is sent before the order button and its confirmation). The tab shown is kept in win.opts.tab)
// - [2026-11-26] type=feat (체결 tab (user 2026-11-26): today's filled orders of the account - 주문시각 / 종목 / 구분 / 체결가(평균) /
//   체결량 / 체결금액 / 주문가 / 주문량 with the day's buy and sell amounts; asked only while the tab is shown. The bot's orders of the
//   same account are in it too)

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore, type Win } from '../store';
import { useCall, useRealtime, useSymbol } from '../hooks';
import { OrderBookView } from './QuoteWindow';
import { NxtMark } from '../components/NxtTag';
import { SymbolSearch } from '../components/SymbolSearch';
import type { OrderBook, Quote } from '../../../shared/kis';
import { orderProblem, reviseProblem, stepPrice, tickSize, type Balance, type Buyable, type Fill, type OpenOrder, type OrderSide, type OrderState, type OrderType } from '../../../shared/order';
import { arrow, fmtInt, fmtPct, signOf, upDown } from '../lib';

const digits = (s: string) => s.replace(/[^\d]/g, '');
const toNum = (s: string) => (s === '' ? 0 : Number(s));
const hms = (s: string) => (s.length >= 6 ? `${s.slice(0, 2)}:${s.slice(2, 4)}:${s.slice(4, 6)}` : s);
const PCTS = [10, 25, 50, 100];

export function OrderWindow({ win }: { win: Win }) {
  const code = win.code;
  const sym = useSymbol(code);
  const mode = useStore((s) => s.auth?.mode);
  const env = useStore((s) => s.auth?.env);
  const rtOn = useStore((s) => s.auth?.rt.status === 'on');
  const setOpts = useStore((s) => s.setOpts);
  const setCode = useStore((s) => s.setCode);
  const openOrder = useStore((s) => s.openOrder);
  type Tab = 'open' | 'fills' | 'hold';
  const tab: Tab = win.opts.tab === 'hold' ? 'hold' : win.opts.tab === 'fills' ? 'fills' : 'open';
  const setTab = (t: Tab) => setOpts(win.id, { tab: t });
  useRealtime(code, { trades: true, book: true });
  const rq = useStore((s) => s.rt.quotes[code]);
  const rbook = useStore((s) => s.rt.books[code]);
  const q = useCall<Quote>('quote', [code], rtOn ? 30000 : 5000);
  const ob = useCall<OrderBook>('orderbook', [code], rtOn && rbook ? 0 : 3000);
  const live = rq && q.data && rq.at > q.at ? { ...q.data, ...rq } : q.data;
  const book = rbook && rbook.at > ob.at ? rbook : ob.data;

  const side: OrderSide = win.opts.side === 'sell' ? 'sell' : 'buy';
  const [type, setType] = useState<OrderType>('limit');
  const [price, setPrice] = useState('');
  const [qty, setQty] = useState('');
  const [st, setSt] = useState<OrderState | null>(null);
  const [bal, setBal] = useState<Balance | null>(null);
  const [buyable, setBuyable] = useState<Buyable | null>(null);
  const [open, setOpen] = useState<OpenOrder[]>([]);
  const [fills, setFills] = useState<Fill[] | null>(null);
  const [loadErr, setLoadErr] = useState('');
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  /** the open order being revised (the form then sets its new price) */
  const [edit, setEdit] = useState<OpenOrder | null>(null);
  const [acctText, setAcctText] = useState('');
  const [acctOpen, setAcctOpen] = useState(false);

  const ready = !!st?.ready;
  const priceNum = toNum(price);
  const qtyNum = toNum(qty);

  // ---- the session: can this window order at all ----
  useEffect(() => {
    let alive = true;
    void window.hts.orderState().then((s) => alive && setSt(s));
    return () => {
      alive = false;
    };
  }, [mode, env]);

  // ---- a new symbol: empty form; the price asked for by the opener (order book click), else the current price ----
  const priced = useRef('');
  useEffect(() => {
    setQty('');
    setEdit(null);
    setMsg(null);
    setBuyable(null);
    priced.current = '';
    setPrice('');
  }, [code]);
  const askedAt = Number(win.opts.at) || 0;
  useEffect(() => {
    const p = Number(win.opts.price);
    if (askedAt && p > 0) {
      setPrice(String(p));
      setType('limit');
      setEdit(null);
      priced.current = code;
    }
  }, [askedAt, code]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (priced.current !== code && live?.price) {
      priced.current = code;
      setPrice(String(live.price));
    }
  }, [code, live?.price]);

  // ---- account data: balance / open orders polled while the window is shown, the buyable amount on every price change ----
  const loadAccount = useCallback(async () => {
    // today's fills only while their tab is shown (one more account call)
    const [b, o, f] = await Promise.all([window.hts.orderBalance(), window.hts.orderOpen(), tab === 'fills' ? window.hts.orderFills() : null]);
    if (b.ok) setBal(b.data);
    if (o.ok) setOpen(o.data);
    if (f?.ok) setFills(f.data);
    setLoadErr(b.ok ? (o.ok ? (f && !f.ok ? f.error : '') : o.error) : b.error);
  }, [tab]);
  useEffect(() => {
    if (!ready || win.min) return;
    void loadAccount();
    const id = window.setInterval(() => void loadAccount(), 5000);
    return () => window.clearInterval(id);
  }, [ready, win.min, loadAccount, st?.account]);
  const loadBuyable = useCallback(async () => {
    const r = await window.hts.orderBuyable(code, type, type === 'market' ? 0 : priceNum);
    setBuyable(r.ok ? r.data : null);
  }, [code, type, priceNum]);
  useEffect(() => {
    if (!ready || side !== 'buy' || (type === 'limit' && !(priceNum > 0))) return;
    const id = window.setTimeout(() => void loadBuyable(), 400);
    return () => window.clearTimeout(id);
  }, [ready, side, type, priceNum, loadBuyable]);

  const holding = useMemo(() => bal?.holdings.find((h) => h.code === code) || null, [bal, code]);
  const pnlSum = useMemo(() => (bal?.holdings || []).reduce((s, h) => s + h.pnl, 0), [bal]);
  const fillSum = useMemo(() => (fills || []).reduce((s, f) => ({ ...s, [f.side]: s[f.side] + f.amount }), { buy: 0, sell: 0 }), [fills]);
  const costSum = useMemo(() => (bal?.holdings || []).reduce((s, h) => s + h.avgPrice * h.qty, 0), [bal]);
  const maxQty = edit ? edit.remain : side === 'buy' ? buyable?.qty ?? 0 : holding?.sellable ?? 0;
  const unit = type === 'market' ? live?.price || 0 : priceNum;
  const amount = unit * qtyNum;

  const problem = edit
    ? reviseProblem({ no: edit.no, branch: edit.branch, kind: 'revise', type: 'limit', price: priceNum, qty: edit.remain })
    : orderProblem({ code, side, type, qty: qtyNum, price: priceNum }, { upper: live?.upper, lower: live?.lower });
  const over = !edit && qtyNum > 0 && ready && (side === 'buy' ? buyable !== null : bal !== null) && qtyNum > maxQty;

  const setSide = (s: OrderSide) => {
    setOpts(win.id, { side: s });
    setEdit(null);
    setMsg(null);
  };
  const after = async (ok: boolean, text: string) => {
    setMsg({ ok, text });
    await loadAccount();
    if (side === 'buy') await loadBuyable();
  };
  const submit = async () => {
    if (busy || !ready || problem) return;
    setBusy(true);
    setMsg(null);
    try {
      if (edit) {
        const r = await window.hts.orderRevise({ no: edit.no, branch: edit.branch, kind: 'revise', type: 'limit', price: priceNum, qty: edit.remain, venue: edit.venue, code: edit.code, side: edit.side });
        await after(r.ok, r.ok ? `정정 접수 - 주문번호 ${r.data.no}` : r.error);
        if (r.ok) setEdit(null);
      } else {
        const r = await window.hts.orderPlace({ code, side, type, qty: qtyNum, price: type === 'market' ? 0 : priceNum });
        await after(r.ok, r.ok ? `${side === 'buy' ? '매수' : '매도'} 주문 접수 - 주문번호 ${r.data.no}` : r.error);
        if (r.ok) setQty('');
      }
    } finally {
      setBusy(false);
    }
  };
  const cancel = async (o: OpenOrder) => {
    if (busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await window.hts.orderRevise({ no: o.no, branch: o.branch, kind: 'cancel', type: o.type, price: 0, qty: o.remain, venue: o.venue, code: o.code, side: o.side });
      await after(r.ok, r.ok ? `취소 접수 - 주문번호 ${r.data.no}` : r.error);
      if (edit?.no === o.no) setEdit(null);
    } finally {
      setBusy(false);
    }
  };
  const startEdit = (o: OpenOrder) => {
    setEdit(o);
    setType('limit');
    setPrice(String(o.price));
    setMsg(null);
  };
  const saveAccount = async (text: string | null) => {
    const r = await window.hts.orderSetAccount(text);
    if (r.ok) {
      setSt(r.data);
      setAcctText('');
      setAcctOpen(false);
      setMsg(null);
    } else setMsg({ ok: false, text: r.error });
  };

  const s = live ? signOf(live.sign, live.change) : 0;
  const sideKo = side === 'buy' ? '매수' : '매도';
  const needAccount = st?.mode === 'kis' && !st.account;
  return (
    <div className={`ord-win ord-${side}`}>
      <div className="q-head">
        <div className="q-id">
          <SymbolSearch compact value={code} onPick={(c) => setCode(win.id, c)} />
          <b className="q-name">{sym?.name || code}</b> <span className="mono muted">{code}</span> {sym?.nxt && <NxtMark />}
          {st?.mode === 'demo' ? <b className="flag">데모</b> : st?.mode === 'kis' ? <b className={`flag${st.env === 'real' ? ' warn' : ''}`}>{st.env === 'real' ? '실전' : '모의'}</b> : null}
        </div>
        {live && (
          <div className={`q-price ${upDown(s)}`}>
            {fmtInt(live.price)}
            <span className="q-chg">
              {arrow(live.sign, live.change)} {fmtInt(Math.abs(live.change))} {fmtPct(live.changePct)}
            </span>
          </div>
        )}
      </div>
      <div className="ord-main">
        <div className="ord-book">
          {book ? (
            <OrderBookView ob={book} price={live?.price} prevClose={live?.prevClose} onPick={(p) => (setPrice(String(p)), setType('limit'))} />
          ) : (
            <div className="empty small">{ob.error || '호가 불러오는 중…'}</div>
          )}
        </div>
        <div className="ord-form">
          <div className="ord-side">
            <button className={side === 'buy' ? 'on buy' : 'buy'} onClick={() => setSide('buy')}>
              매수
            </button>
            <button className={side === 'sell' ? 'on sell' : 'sell'} onClick={() => setSide('sell')}>
              매도
            </button>
          </div>
          <div className="ord-acct">
            <span className="muted">계좌</span>
            <b>{st?.account || '-'}</b>
            {st?.mode === 'kis' && st.account && (
              <button className="mini" onClick={() => setAcctOpen(!acctOpen)}>
                변경
              </button>
            )}
          </div>
          {(needAccount || acctOpen) && (
            <div className="ord-acct-form">
              <input
                value={acctText}
                placeholder="계좌번호 10자리 (예: 12345678-01)"
                inputMode="numeric"
                onChange={(e) => setAcctText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void saveAccount(acctText)}
              />
              <button className="primary" onClick={() => void saveAccount(acctText)}>
                저장
              </button>
              {st?.account && (
                <button onClick={() => window.confirm('저장된 계좌번호를 지울까요? 지우면 주문할 수 없습니다.') && void saveAccount(null)} title="저장된 계좌번호 삭제">
                  삭제
                </button>
              )}
              <p className="muted small">이 PC 에 암호화해 저장합니다 ({st?.env === 'paper' ? '모의' : '실전'} 서버용). 로그인한 앱키의 계좌여야 합니다.</p>
            </div>
          )}
          {!ready && !needAccount && <p className="ord-note">{st?.reason || '주문 상태 확인 중…'}</p>}
          {edit && (
            <p className="ord-note edit">
              정정: {edit.name} {edit.side === 'buy' ? '매수' : '매도'} {fmtInt(edit.price)}원 · 잔량 {fmtInt(edit.remain)}주 (주문 {edit.no}){' '}
              <button className="mini" onClick={() => setEdit(null)}>
                정정 그만
              </button>
            </p>
          )}
          <div className="ord-rows">
            <span>유형</span>
            <span className="seg">
              <button className={type === 'limit' ? 'on' : ''} onClick={() => setType('limit')}>
                지정가
              </button>
              <button className={type === 'market' ? 'on' : ''} disabled={!!edit} onClick={() => setType('market')}>
                시장가
              </button>
            </span>
            <span>가격</span>
            <span className="ord-step">
              <button disabled={type === 'market'} onClick={() => setPrice(String(stepPrice(priceNum || live?.price || 0, -1)))} title="한 호가 아래">
                −
              </button>
              <input
                className="num"
                inputMode="numeric"
                disabled={type === 'market'}
                value={type === 'market' ? '' : price === '' ? '' : Number(price).toLocaleString('ko-KR')}
                placeholder={type === 'market' ? '시장가' : '가격'}
                onChange={(e) => setPrice(digits(e.target.value))}
                onKeyDown={(e) => e.key === 'Enter' && void submit()}
              />
              <button disabled={type === 'market'} onClick={() => setPrice(String(stepPrice(priceNum || live?.price || 0, 1)))} title="한 호가 위">
                +
              </button>
              <u>원</u>
            </span>
            <span>수량</span>
            <span className="ord-step">
              <button disabled={!!edit} onClick={() => setQty(String(Math.max(0, qtyNum - 1) || ''))}>
                −
              </button>
              <input
                className="num"
                inputMode="numeric"
                disabled={!!edit}
                value={edit ? edit.remain.toLocaleString('ko-KR') : qty === '' ? '' : Number(qty).toLocaleString('ko-KR')}
                placeholder="수량"
                onChange={(e) => setQty(digits(e.target.value))}
                onKeyDown={(e) => e.key === 'Enter' && void submit()}
              />
              <button disabled={!!edit} onClick={() => setQty(String(qtyNum + 1))}>
                +
              </button>
              <u>주</u>
            </span>
            <span />
            <span className="ord-pct">
              {PCTS.map((p) => (
                <button key={p} className="mini" disabled={!!edit || !(maxQty > 0)} onClick={() => setQty(String(Math.floor((maxQty * p) / 100) || ''))} title={`${side === 'buy' ? '매수' : '매도'}가능수량의 ${p}%`}>
                  {p}%
                </button>
              ))}
            </span>
            <span>주문금액</span>
            <b className="num">
              {amount > 0 ? `${fmtInt(amount)}원` : '-'}
              {type === 'market' && amount > 0 && <small className="muted"> (현재가 기준)</small>}
            </b>
          </div>
          <div className="kv-grid ord-info">
            <span>예수금</span>
            <b>{bal ? `${fmtInt(bal.cash)}원` : '-'}</b>
            <span title="D+2 예수금 (가수도 정산 금액)">D+2 예수금</span>
            <b>{bal ? `${fmtInt(bal.cashD2)}원` : '-'}</b>
            {side === 'buy' ? (
              <>
                <span title="미수 없이 살 수 있는 금액">주문가능금액</span>
                <b>{buyable ? `${fmtInt(buyable.amount)}원` : '-'}</b>
                <span title={`미수 없이 살 수 있는 수량 (${type === 'market' ? '시장가' : '입력한 가격'} 기준)`}>매수가능수량</span>
                <b>{buyable ? `${fmtInt(buyable.qty)}주` : '-'}</b>
              </>
            ) : (
              <>
                <span>보유수량</span>
                <b>{bal ? `${fmtInt(holding?.qty ?? 0)}주` : '-'}</b>
                <span>매도가능수량</span>
                <b>{bal ? `${fmtInt(holding?.sellable ?? 0)}주` : '-'}</b>
                <span>평균단가</span>
                <b>{holding ? `${fmtInt(holding.avgPrice)}원` : '-'}</b>
                <span>평가손익</span>
                <b className={holding ? upDown(Math.sign(holding.pnl)) : ''}>{holding ? `${fmtInt(holding.pnl)}원 (${fmtPct(holding.pnlPct)})` : '-'}</b>
              </>
            )}
          </div>
          <p className={`ord-msg${msg ? (msg.ok ? ' ok' : ' bad') : over ? ' bad' : ''}`}>
            {msg ? msg.text : over ? `${sideKo}가능수량(${fmtInt(maxQty)}주)을 넘는 수량입니다.` : (qtyNum > 0 || edit) && problem ? problem : loadErr || `호가 단위 ${fmtInt(tickSize(priceNum || live?.price || 0))}원`}
          </p>
          <button className={`ord-go ${edit ? 'edit' : side}`} disabled={busy || !ready || !!problem} onClick={() => void submit()}>
            {busy ? '전송 중…' : edit ? '정정 주문' : `현금 ${sideKo}`}
          </button>
        </div>
      </div>
      <div className="q-tabs">
        {/* [2026-11-26] 잔고 tab beside the open orders (user 2026-11-26) */}
        <span className="seg">
          <button className={tab === 'open' ? 'on' : ''} onClick={() => setTab('open')}>
            미체결 {open.length}
          </button>
          <button className={tab === 'fills' ? 'on' : ''} onClick={() => setTab('fills')} title="오늘 체결된 주문 (같은 계좌의 봇 주문 포함)">
            체결{fills ? ` ${fills.length}` : ''}
          </button>
          <button className={tab === 'hold' ? 'on' : ''} onClick={() => setTab('hold')}>
            잔고 {bal ? bal.holdings.length : 0}
          </button>
        </span>
        {tab === 'fills' && fills && (
          <span className="ord-sum">
            <span className="muted">매수</span> <b className="up">{fmtInt(fillSum.buy)}원</b> <span className="muted">매도</span> <b className="down">{fmtInt(fillSum.sell)}원</b>
          </span>
        )}
        {tab === 'hold' && bal && (
          <span className="ord-sum">
            <span className="muted">총평가</span> <b>{fmtInt(bal.total)}원</b> <span className="muted">평가손익</span>{' '}
            <b className={upDown(Math.sign(pnlSum))}>
              {fmtInt(pnlSum)}원{costSum > 0 ? ` (${fmtPct((pnlSum / costSum) * 100)})` : ''}
            </b>
          </span>
        )}
        <span>
          {loadErr && <span className="down small">{loadErr} </span>}
          <button className="mini" disabled={!ready} onClick={() => void loadAccount()}>
            새로고침
          </button>
        </span>
      </div>
      <div className="q-tab-body">
        {tab === 'fills' && (
          <table className="grid">
            <thead>
              <tr>
                <th title="주문 시각 (한 주문의 체결은 합쳐서 한 줄)">주문시각</th>
                <th>종목</th>
                <th>구분</th>
                <th className="num">체결가</th>
                <th className="num">체결량</th>
                <th className="num">체결금액</th>
                <th className="num">주문가</th>
                <th className="num">주문량</th>
              </tr>
            </thead>
            <tbody>
              {(fills || []).map((f) => (
                <tr key={f.no + f.side} className={f.code === code ? 'sel' : ''} onClick={() => setCode(win.id, f.code)} title="누르면 이 종목으로 바꿉니다">
                  <td className="mono">{hms(f.time)}</td>
                  <td>
                    {f.name || f.code} <span className="mono muted">{f.code}</span>
                  </td>
                  <td className={f.side === 'buy' ? 'up' : 'down'}>{f.side === 'buy' ? '매수' : '매도'}</td>
                  <td className="num">{fmtInt(f.price)}</td>
                  <td className="num">{fmtInt(f.qty)}</td>
                  <td className="num">{fmtInt(f.amount)}</td>
                  <td className="num">{f.ordPrice > 0 ? fmtInt(f.ordPrice) : '시장가'}</td>
                  <td className="num">{fmtInt(f.ordQty)}</td>
                </tr>
              ))}
              {!(fills || []).length && (
                <tr>
                  <td colSpan={8} className="center muted">
                    {ready ? (fills ? '오늘 체결된 주문이 없습니다' : '체결 내역 불러오는 중…') : st?.reason || ''}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
        {tab === 'hold' && (
          <table className="grid">
            <thead>
              <tr>
                <th>종목</th>
                <th className="num">보유</th>
                <th className="num">매도가능</th>
                <th className="num">평균단가</th>
                <th className="num">현재가</th>
                <th className="num">평가금액</th>
                <th className="num">평가손익</th>
                <th className="num">수익률</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(bal?.holdings || []).map((h) => (
                <tr key={h.code} className={h.code === code ? 'sel' : ''} onClick={() => setCode(win.id, h.code)} title="누르면 이 종목으로 바꿉니다">
                  <td>
                    {h.name || h.code} <span className="mono muted">{h.code}</span>
                  </td>
                  <td className="num">{fmtInt(h.qty)}</td>
                  <td className="num">{fmtInt(h.sellable)}</td>
                  <td className="num">{fmtInt(h.avgPrice)}</td>
                  <td className={`num ${upDown(Math.sign(h.price - h.avgPrice))}`}>{fmtInt(h.price)}</td>
                  <td className="num">{fmtInt(h.price * h.qty)}</td>
                  <td className={`num ${upDown(Math.sign(h.pnl))}`}>{fmtInt(h.pnl)}</td>
                  <td className={`num ${upDown(Math.sign(h.pnl))}`}>{fmtPct(h.pnlPct)}</td>
                  <td className="ord-act">
                    <button
                      className="mini"
                      disabled={!(h.sellable > 0)}
                      title="이 종목의 매도 주문을 현재가로 준비합니다 (전송은 주문 버튼과 확인 창을 거칩니다)"
                      onClick={(e) => (e.stopPropagation(), openOrder(h.code, 'sell', h.price))}
                    >
                      매도
                    </button>
                  </td>
                </tr>
              ))}
              {!(bal?.holdings.length) && (
                <tr>
                  <td colSpan={9} className="center muted">
                    {ready ? (bal ? '보유 종목이 없습니다' : '잔고 불러오는 중…') : st?.reason || ''}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
        <table className="grid" hidden={tab !== 'open'}>
          <thead>
            <tr>
              <th>시각</th>
              <th>종목</th>
              <th>구분</th>
              <th className="num">주문가</th>
              <th className="num">주문</th>
              <th className="num">미체결</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {open.map((o) => (
              <tr key={o.no} className={edit?.no === o.no ? 'sel' : ''}>
                <td className="mono">{hms(o.time)}</td>
                <td>{o.name || o.code}</td>
                <td className={o.side === 'buy' ? 'up' : 'down'}>{o.side === 'buy' ? '매수' : '매도'}</td>
                <td className="num">{o.type === 'market' ? '시장가' : fmtInt(o.price)}</td>
                <td className="num">{fmtInt(o.qty)}</td>
                <td className="num">{fmtInt(o.remain)}</td>
                <td className="ord-act">
                  <button className="mini" disabled={busy || o.type === 'market'} onClick={() => startEdit(o)} title="남은 수량의 가격을 바꿉니다">
                    정정
                  </button>
                  <button className="mini" disabled={busy} onClick={() => void cancel(o)} title="남은 수량을 모두 취소합니다">
                    취소
                  </button>
                </td>
              </tr>
            ))}
            {!open.length && (
              <tr>
                <td colSpan={7} className="center muted">
                  {ready ? '미체결 주문이 없습니다' : st?.reason || ''}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
