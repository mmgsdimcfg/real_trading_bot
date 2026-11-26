// [2026-11-26] order window: the trading allowlist, request bodies, validation, answer normalizers and the demo account
import { describe, expect, it } from 'vitest';
import { assertInquiry, restEndpoint } from '../src/shared/kis';
import {
  maskAccount,
  onTick,
  openOrderParams,
  orderBody,
  orderProblem,
  orderText,
  parseAccount,
  psblOrderParams,
  reviseBody,
  reviseProblem,
  stepPrice,
  tickSize,
  toBalance,
  toBuyable,
  toFills,
  dailyParams,
  toOpenOrders,
  toOrderResult,
  tradeEndpoint,
  TRADE_APIS,
} from '../src/shared/order';
import { DemoTrader, KisTrader } from '../src/main/kisOrder';
import type { KisClient } from '../src/main/kisClient';

const acct = { cano: '12345678', prod: '01' };

describe('trading allowlist', () => {
  it('knows exactly the TRs of the official examples, per server', () => {
    expect(tradeEndpoint('order_buy', 'real')).toEqual({ api: 'order_buy', method: 'POST', path: '/uapi/domestic-stock/v1/trading/order-cash', trId: 'TTTC0012U' });
    expect(tradeEndpoint('order_sell', 'real').trId).toBe('TTTC0011U');
    expect(tradeEndpoint('order_buy', 'paper').trId).toBe('VTTC0012U');
    expect(tradeEndpoint('order_sell', 'paper').trId).toBe('VTTC0011U');
    expect(tradeEndpoint('order_rvsecncl', 'real')).toMatchObject({ method: 'POST', path: '/uapi/domestic-stock/v1/trading/order-rvsecncl', trId: 'TTTC0013U' });
    expect(tradeEndpoint('psbl_order', 'real')).toMatchObject({ method: 'GET', trId: 'TTTC8908R' });
    expect(tradeEndpoint('balance', 'paper')).toMatchObject({ method: 'GET', trId: 'VTTC8434R' });
    expect(tradeEndpoint('psbl_rvsecncl', 'real').trId).toBe('TTTC0084R');
    expect(tradeEndpoint('daily_ccld', 'paper').trId).toBe('VTTC0081R');
  });

  it('refuses anything else, and the paper server has no 정정취소가능주문조회', () => {
    expect(() => tradeEndpoint('order_credit', 'real')).toThrow('허용되지 않은 API');
    expect(() => tradeEndpoint('inquire_price', 'real')).toThrow('허용되지 않은 API');
    expect(() => tradeEndpoint('psbl_rvsecncl', 'paper')).toThrow('모의 서버에 없는 API');
    for (const a of TRADE_APIS) if (a !== 'psbl_rvsecncl') expect(tradeEndpoint(a, 'paper').path).toMatch(/^\/uapi\/domestic-stock\/v1\/trading\//);
  });

  it('the market-data path still rejects every trading TR', () => {
    for (const a of TRADE_APIS) {
      const ep = tradeEndpoint(a, 'real');
      expect(() => assertInquiry(ep.path, ep.trId)).toThrow('조회 전용');
      expect(() => restEndpoint(a)).toThrow('조회 전용');
    }
  });
});

describe('account number', () => {
  it('reads 8 + 2 digits in the usual spellings and masks it', () => {
    expect(parseAccount('12345678-01')).toEqual(acct);
    expect(parseAccount(' 1234567801 ')).toEqual(acct);
    expect(parseAccount('12345678 01')).toEqual(acct);
    expect(parseAccount('1234567-01')).toBeNull();
    expect(parseAccount('12345678-0a')).toBeNull();
    expect(parseAccount('')).toBeNull();
    expect(maskAccount(acct)).toBe('1234****-01');
  });
});

describe('prices', () => {
  it('KRX tick sizes', () => {
    expect([1999, 2000, 4995, 5000, 19990, 20000, 49950, 50000, 199900, 200000, 499500, 500000].map(tickSize)).toEqual([1, 5, 5, 10, 10, 50, 50, 100, 100, 500, 500, 1000]);
    expect(onTick(70000)).toBe(true);
    expect(onTick(70050)).toBe(false);
    expect(onTick(0)).toBe(false);
  });

  it('steps one tick and crosses a band on the right tick', () => {
    expect(stepPrice(70000, 1)).toBe(70100);
    expect(stepPrice(70000, -1)).toBe(69900);
    expect(stepPrice(1999, 1)).toBe(2000);
    expect(stepPrice(2000, 1)).toBe(2005);
    expect(stepPrice(2000, -1)).toBe(1999);
    expect(stepPrice(50000, -1)).toBe(49950);
    expect(stepPrice(70030, 1)).toBe(70100); // off the tick: to the next valid one
    expect(stepPrice(70030, -1)).toBe(70000);
    expect(stepPrice(1, -1)).toBe(1);
  });
});

describe('order validation and bodies', () => {
  const o = { code: '005930', side: 'buy' as const, type: 'limit' as const, qty: 10, price: 70000 };

  it('finds what is wrong with an order', () => {
    expect(orderProblem(o)).toBe('');
    expect(orderProblem({ ...o, qty: 0 })).toContain('수량');
    expect(orderProblem({ ...o, qty: 1.5 })).toContain('수량');
    expect(orderProblem({ ...o, price: 70050 })).toContain('호가 단위(100원)');
    expect(orderProblem({ ...o, price: 0 })).toContain('가격');
    expect(orderProblem({ ...o, code: '5930' })).toContain('종목 코드');
    expect(orderProblem({ ...o, price: 92000 }, { upper: 91000, lower: 49000 })).toContain('상한가');
    expect(orderProblem({ ...o, price: 48000 }, { upper: 91000, lower: 49000 })).toContain('하한가');
    // a market order needs no price
    expect(orderProblem({ ...o, type: 'market', price: 0 })).toBe('');
  });

  it('builds the 현금주문 body (KRX, 지정가 00 / 시장가 01, 매도 SLL_TYPE 01)', () => {
    expect(orderBody(acct, o)).toEqual({ CANO: '12345678', ACNT_PRDT_CD: '01', PDNO: '005930', ORD_DVSN: '00', ORD_QTY: '10', ORD_UNPR: '70000', EXCG_ID_DVSN_CD: 'KRX', SLL_TYPE: '', CNDT_PRIC: '' });
    expect(orderBody(acct, { ...o, side: 'sell', type: 'market', price: 70000 })).toMatchObject({ ORD_DVSN: '01', ORD_UNPR: '0', SLL_TYPE: '01' });
    expect(orderText(o, '삼성전자')).toBe('매수 삼성전자(005930) 지정가 70,000원 × 10주 = 700,000원');
    expect(orderText({ ...o, side: 'sell', type: 'market' })).toBe('매도 005930 시장가 × 10주');
  });

  it('builds the 정정 / 취소 body for the whole remaining quantity', () => {
    const r = { no: '0000012345', branch: '91252', kind: 'revise' as const, type: 'limit' as const, price: 69900, qty: 7 };
    expect(reviseBody(acct, r)).toEqual({ CANO: '12345678', ACNT_PRDT_CD: '01', KRX_FWDG_ORD_ORGNO: '91252', ORGN_ODNO: '0000012345', ORD_DVSN: '00', RVSE_CNCL_DVSN_CD: '01', ORD_QTY: '7', ORD_UNPR: '69900', QTY_ALL_ORD_YN: 'Y', EXCG_ID_DVSN_CD: 'KRX' });
    expect(reviseBody(acct, { ...r, kind: 'cancel' })).toMatchObject({ RVSE_CNCL_DVSN_CD: '02', ORD_UNPR: '0', QTY_ALL_ORD_YN: 'Y' });
    expect(reviseProblem(r)).toBe('');
    expect(reviseProblem({ ...r, price: 69950 })).toContain('호가 단위');
    expect(reviseProblem({ ...r, qty: 0 })).toContain('잔량');
    expect(reviseProblem({ ...r, kind: 'cancel', price: 0 })).toBe('');
    expect(reviseProblem({ ...r, no: '' })).toContain('주문번호');
  });

  it('inquiry parameters', () => {
    expect(psblOrderParams(acct, '005930', 'limit', 70000)).toMatchObject({ PDNO: '005930', ORD_UNPR: '70000', ORD_DVSN: '00', CMA_EVLU_AMT_ICLD_YN: 'N', OVRS_ICLD_YN: 'N' });
    expect(psblOrderParams(acct, '005930', 'market', 70000)).toMatchObject({ ORD_UNPR: '0', ORD_DVSN: '01' });
    expect(openOrderParams(acct)).toEqual({ CANO: '12345678', ACNT_PRDT_CD: '01', INQR_DVSN_1: '0', INQR_DVSN_2: '0', CTX_AREA_FK100: '', CTX_AREA_NK100: '' });
  });
});

describe('answers', () => {
  it('매수가능조회', () => {
    expect(toBuyable({ ord_psbl_cash: '1500000', nrcvb_buy_amt: '1499000', nrcvb_buy_qty: '21', max_buy_amt: '3700000', max_buy_qty: '52' })).toEqual({ cash: 1500000, amount: 1499000, qty: 21, maxAmount: 3700000, maxQty: 52 });
    expect(toBuyable(undefined)).toEqual({ cash: 0, amount: 0, qty: 0, maxAmount: 0, maxQty: 0 });
  });

  it('잔고조회: holdings without quantity are dropped', () => {
    const b = toBalance(
      [
        { pdno: '005930', prdt_name: '삼성전자', hldg_qty: '10', ord_psbl_qty: '7', pchs_avg_pric: '69500.0000', prpr: '70000', evlu_pfls_amt: '5000', evlu_pfls_rt: '0.72' },
        { pdno: '000660', prdt_name: 'SK하이닉스', hldg_qty: '0', ord_psbl_qty: '0' },
      ],
      [{ dnca_tot_amt: '2000000', prvs_rcdl_excc_amt: '1800000', tot_evlu_amt: '2700000' }],
    );
    expect(b).toEqual({ cash: 2000000, cashD2: 1800000, total: 2700000, holdings: [{ code: '005930', name: '삼성전자', qty: 10, sellable: 7, avgPrice: 69500, price: 70000, pnl: 5000, pnlPct: 0.72 }] });
  });

  it('미체결: both TR shapes, filled orders dropped, newest first', () => {
    const a = toOpenOrders([
      { odno: '0000000001', ord_gno_brno: '91252', pdno: '005930', prdt_name: '삼성전자', sll_buy_dvsn_cd: '02', ord_dvsn_cd: '00', ord_unpr: '69000', ord_qty: '10', tot_ccld_qty: '3', psbl_qty: '7', ord_tmd: '091500', excg_id_dvsn_cd: 'KRX' },
      { odno: '0000000002', ord_gno_brno: '91252', pdno: '000660', prdt_name: 'SK하이닉스', sll_buy_dvsn_cd: '01', ord_dvsn_cd: '00', ord_unpr: '180000', ord_qty: '2', tot_ccld_qty: '0', psbl_qty: '2', ord_tmd: '100000' },
      { odno: '0000000003', pdno: '035720', sll_buy_dvsn_cd: '02', ord_qty: '5', tot_ccld_qty: '5', psbl_qty: '0', ord_tmd: '110000' },
    ]);
    expect(a.map((o) => o.no)).toEqual(['0000000002', '0000000001']);
    expect(a[1]).toEqual({ no: '0000000001', branch: '91252', code: '005930', name: '삼성전자', side: 'buy', type: 'limit', price: 69000, qty: 10, filled: 3, remain: 7, time: '091500', venue: 'KRX' });
    expect(a[0]).toMatchObject({ side: 'sell', venue: 'KRX' });
    // 일별주문체결조회 (paper): rmn_qty
    expect(toOpenOrders([{ odno: '9', pdno: '005930', sll_buy_dvsn_cd: '02', ord_qty: '10', tot_ccld_qty: '4', rmn_qty: '6', ord_tmd: '120000' }])[0].remain).toBe(6);
  });

  it('당일 체결: orders with a filled quantity only, newest first; every exchange on the real server', () => {
    const f = toFills([
      { odno: '0000000011', pdno: '005930', prdt_name: '삼성전자', sll_buy_dvsn_cd: '02', ord_tmd: '091500', ord_qty: '10', ord_unpr: '70000', tot_ccld_qty: '10', avg_prvs: '69950', tot_ccld_amt: '699500' },
      { odno: '0000000012', pdno: '000660', prdt_name: 'SK하이닉스', sll_buy_dvsn_cd: '01', ord_tmd: '101010', ord_qty: '5', ord_unpr: '0', tot_ccld_qty: '3', avg_prvs: '0', tot_ccld_amt: '546000' },
      { odno: '0000000013', pdno: '035720', sll_buy_dvsn_cd: '02', ord_tmd: '111111', ord_qty: '5', tot_ccld_qty: '0' },
    ]);
    expect(f.map((x) => x.no)).toEqual(['0000000012', '0000000011']);
    expect(f[1]).toEqual({ no: '0000000011', code: '005930', name: '삼성전자', side: 'buy', time: '091500', qty: 10, price: 69950, amount: 699500, ordQty: 10, ordPrice: 70000 });
    // no average price in the row: amount / quantity
    expect(f[0]).toMatchObject({ side: 'sell', qty: 3, price: 182000, ordPrice: 0 });
    expect(dailyParams(acct, '20261126', '01', 'ALL')).toMatchObject({ INQR_STRT_DT: '20261126', INQR_END_DT: '20261126', CCLD_DVSN: '01', EXCG_ID_DVSN_CD: 'ALL', SLL_BUY_DVSN_CD: '00' });
  });

  it('주문 접수 결과 (upper-case keys of 현금주문, lower-case of 정정취소)', () => {
    expect(toOrderResult({ KRX_FWDG_ORD_ORGNO: '91252', ODNO: '0000117057', ORD_TMD: '121052' })).toEqual({ no: '0000117057', branch: '91252', time: '121052' });
    expect(toOrderResult({ krx_fwdg_ord_orgno: '91252', odno: '0000117058', ord_tmd: '121100' })).toEqual({ no: '0000117058', branch: '91252', time: '121100' });
  });
});

describe('KisTrader', () => {
  const calls: { api: string; params: Record<string, string> }[] = [];
  const fake = (env: 'real' | 'paper') =>
    ({
      env,
      trade: async (api: string, params: Record<string, string>) => {
        calls.push({ api, params });
        return { output: { ODNO: '1', KRX_FWDG_ORD_ORGNO: '2', ORD_TMD: '090000' }, output1: [], output2: [], trCont: '' };
      },
    }) as unknown as KisClient;

  it('sends a buy / sell through its own TR and never an invalid order', async () => {
    calls.length = 0;
    const t = new KisTrader(fake('real'), acct);
    await t.place({ code: '005930', side: 'buy', type: 'limit', qty: 1, price: 70000 });
    await t.place({ code: '005930', side: 'sell', type: 'market', qty: 1, price: 0 });
    expect(calls.map((c) => c.api)).toEqual(['order_buy', 'order_sell']);
    expect(calls[0].params.ORD_UNPR).toBe('70000');
    await expect(t.place({ code: '005930', side: 'buy', type: 'limit', qty: 1, price: 70050 })).rejects.toThrow('호가 단위');
    await expect(t.revise({ no: '', branch: '', kind: 'cancel', type: 'limit', price: 0, qty: 1 })).rejects.toThrow('주문번호');
    expect(calls).toHaveLength(2);
  });

  it('open orders: 정정취소가능주문조회 on the real server, 일별주문체결조회 on the paper server', async () => {
    calls.length = 0;
    await new KisTrader(fake('real'), acct).openOrders();
    await new KisTrader(fake('paper'), acct).openOrders();
    expect(calls.map((c) => c.api)).toEqual(['psbl_rvsecncl', 'daily_ccld']);
    expect(calls[1].params.CCLD_DVSN).toBe('02');
    calls.length = 0;
    await new KisTrader(fake('real'), acct).fills();
    await new KisTrader(fake('paper'), acct).fills();
    expect(calls.map((c) => [c.api, c.params.CCLD_DVSN, c.params.EXCG_ID_DVSN_CD])).toEqual([
      ['daily_ccld', '01', 'ALL'],
      ['daily_ccld', '01', 'KRX'],
    ]);
  });
});

describe('DemoTrader', () => {
  const mk = () =>
    new DemoTrader(
      async () => 70000,
      () => '삼성전자',
    );

  it('fills a crossing order, keeps a resting one open, revises and cancels it', async () => {
    const t = mk();
    expect((await t.buyable('005930', 'limit', 70000)).qty).toBe(142);
    await t.place({ code: '005930', side: 'buy', type: 'market', qty: 10, price: 0 });
    let b = await t.balance();
    expect(b.cash).toBe(9_300_000);
    expect(b.holdings[0]).toMatchObject({ code: '005930', qty: 10, sellable: 10, avgPrice: 70000 });
    // a buy below the price rests and ties up its cash
    await t.place({ code: '005930', side: 'buy', type: 'limit', qty: 10, price: 69000 });
    let open = await t.openOrders();
    expect(open).toHaveLength(1);
    expect((await t.buyable('005930', 'limit', 70000)).cash).toBe(9_300_000 - 690_000);
    // a resting sell reduces what can still be sold
    await t.place({ code: '005930', side: 'sell', type: 'limit', qty: 4, price: 71000 });
    b = await t.balance();
    expect(b.holdings[0].sellable).toBe(6);
    await expect(t.place({ code: '005930', side: 'sell', type: 'limit', qty: 7, price: 71000 })).rejects.toThrow('매도가능수량');
    // revise the buy up to the price: it fills
    open = await t.openOrders();
    const buy = open.find((o) => o.side === 'buy')!;
    await t.revise({ no: buy.no, branch: buy.branch, kind: 'revise', type: 'limit', price: 70000, qty: buy.remain });
    b = await t.balance();
    expect(b.holdings[0].qty).toBe(20);
    // cancel the sell
    const sell = (await t.openOrders()).find((o) => o.side === 'sell')!;
    await t.revise({ no: sell.no, branch: sell.branch, kind: 'cancel', type: 'limit', price: 0, qty: sell.remain });
    expect(await t.openOrders()).toHaveLength(0);
    // two fills so far: the market buy and the revised buy; the cancelled sell never filled
    expect((await t.fills()).map((f) => [f.side, f.qty, f.price])).toEqual([
      ['buy', 10, 70000],
      ['buy', 10, 70000],
    ]);
    await expect(t.place({ code: '005930', side: 'buy', type: 'limit', qty: 1000, price: 70000 })).rejects.toThrow('주문가능금액');
  });
});
