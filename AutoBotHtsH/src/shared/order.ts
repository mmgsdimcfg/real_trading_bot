// Update log
// - [2026-11-26] type=feat (Order window (user 2026-11-26: real orders from the app). Pure part (tests/order.test.ts): the trading
//   allowlist - exactly the six account TRs the order window needs, each with its own path and tr_id per server (taken from the
//   official examples_llm/domestic_stock/{order_cash,order_rvsecncl,inquire_psbl_order,inquire_balance,inquire_psbl_rvsecncl,
//   inquire_daily_ccld}); the market-data allowlist of kis.ts is untouched and still rejects every one of them -, account number
//   parsing / masking, KRX tick sizes, order validation, the request bodies and the normalizers of the answers)
// - [2026-11-26] type=feat (Today's fills (user 2026-11-26): Fill + toFills() from 주식일별주문체결조회 with CCLD_DVSN 01 - one row per
//   order that has a filled quantity (the TR sums an order's fills: quantity, average price, amount; the time is the order's).
//   dailyParams() is the one builder of that TR's query (01 체결 / 02 미체결; every exchange on the real server, KRX on the paper one))

import type { KisEnv } from './kis';

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'limit' | 'market';

export interface Account {
  /** 종합계좌번호, 8 digits */
  cano: string;
  /** 계좌상품코드, 2 digits */
  prod: string;
}

export interface OrderReq {
  code: string;
  side: OrderSide;
  type: OrderType;
  qty: number;
  /** limit price; ignored for a market order */
  price: number;
}

/** an open (unfilled) order as the window lists it */
export interface OpenOrder {
  /** 주문번호 */
  no: string;
  /** 주문채번지점번호 (KRX_FWDG_ORD_ORGNO of a revise / cancel) */
  branch: string;
  code: string;
  name: string;
  side: OrderSide;
  type: OrderType;
  price: number;
  qty: number;
  filled: number;
  /** quantity that can still be revised / cancelled */
  remain: number;
  /** HHMMSS */
  time: string;
  venue: string;
}

export interface Holding {
  code: string;
  name: string;
  qty: number;
  /** quantity that can be sold now */
  sellable: number;
  avgPrice: number;
  price: number;
  pnl: number;
  pnlPct: number;
}

export interface Balance {
  /** 예수금 */
  cash: number;
  /** D+2 예수금 (가수도정산금액) */
  cashD2: number;
  /** 총평가금액 */
  total: number;
  holdings: Holding[];
}

export interface Buyable {
  /** 주문가능현금 */
  cash: number;
  /** 미수 없는 매수 금액 / 수량 - what the window offers */
  amount: number;
  qty: number;
  /** 최대 매수 금액 / 수량 (with margin) - shown, not used for the % buttons */
  maxAmount: number;
  maxQty: number;
}

/** one order of today with a filled quantity (the TR reports an order's fills summed) */
export interface Fill {
  no: string;
  code: string;
  name: string;
  side: OrderSide;
  /** order time, HHMMSS */
  time: string;
  /** filled quantity, average fill price and filled amount */
  qty: number;
  price: number;
  amount: number;
  /** ordered quantity and price (0 = market) */
  ordQty: number;
  ordPrice: number;
}

export interface OrderResult {
  no: string;
  branch: string;
  time: string;
}

// ---- trading allowlist ----

export const TRADE_APIS = ['order_buy', 'order_sell', 'order_rvsecncl', 'psbl_order', 'balance', 'psbl_rvsecncl', 'daily_ccld'] as const;
export type TradeApi = (typeof TRADE_APIS)[number];

interface TradeSpec {
  method: 'GET' | 'POST';
  path: string;
  trId: Record<KisEnv, string>;
}

const T = '/uapi/domestic-stock/v1/trading/';
const TRADE_SPEC: Record<TradeApi, TradeSpec> = {
  order_buy: { method: 'POST', path: `${T}order-cash`, trId: { real: 'TTTC0012U', paper: 'VTTC0012U' } },
  order_sell: { method: 'POST', path: `${T}order-cash`, trId: { real: 'TTTC0011U', paper: 'VTTC0011U' } },
  order_rvsecncl: { method: 'POST', path: `${T}order-rvsecncl`, trId: { real: 'TTTC0013U', paper: 'VTTC0013U' } },
  psbl_order: { method: 'GET', path: `${T}inquire-psbl-order`, trId: { real: 'TTTC8908R', paper: 'VTTC8908R' } },
  balance: { method: 'GET', path: `${T}inquire-balance`, trId: { real: 'TTTC8434R', paper: 'VTTC8434R' } },
  // 정정취소가능주문조회: the real server only (the paper server has no such TR - open orders come from daily_ccld there)
  psbl_rvsecncl: { method: 'GET', path: `${T}inquire-psbl-rvsecncl`, trId: { real: 'TTTC0084R', paper: '' } },
  daily_ccld: { method: 'GET', path: `${T}inquire-daily-ccld`, trId: { real: 'TTTC0081R', paper: 'VTTC0081R' } },
};

export interface TradeEndpoint {
  api: TradeApi;
  method: 'GET' | 'POST';
  path: string;
  trId: string;
}

/** the endpoint of one trading call; anything that is not on the list above throws */
export function tradeEndpoint(api: string, env: KisEnv): TradeEndpoint {
  if (!(TRADE_APIS as readonly string[]).includes(api)) throw new Error(`주문창: 허용되지 않은 API ${api}`);
  const s = TRADE_SPEC[api as TradeApi];
  const trId = s.trId[env];
  if (!trId) throw new Error(`주문창: ${env === 'paper' ? '모의' : '실전'} 서버에 없는 API ${api}`);
  return { api: api as TradeApi, method: s.method, path: s.path, trId };
}

// ---- account number ----

/** "12345678-01", "1234567801" or "12345678 01" -> account; null when it is not 8 + 2 digits */
export function parseAccount(text: string): Account | null {
  const d = String(text || '').replace(/[\s-]/g, '');
  return /^\d{10}$/.test(d) ? { cano: d.slice(0, 8), prod: d.slice(8) } : null;
}

/** "1234****-01" */
export const maskAccount = (a: Account) => `${a.cano.slice(0, 4)}****-${a.prod}`;

// ---- prices ----

/** KRX tick size of a price (the 2023 table, the same for KOSPI and KOSDAQ) */
export function tickSize(price: number): number {
  if (price < 2000) return 1;
  if (price < 5000) return 5;
  if (price < 20000) return 10;
  if (price < 50000) return 50;
  if (price < 200000) return 100;
  if (price < 500000) return 500;
  return 1000;
}

/** one tick up (dir 1) or down (dir -1) from a price, landing on a valid tick; never below 1 */
export function stepPrice(price: number, dir: 1 | -1): number {
  const p = Math.max(0, Math.round(price) || 0);
  if (dir > 0) {
    const t = tickSize(p);
    return Math.floor(p / t) * t + t;
  }
  const t = tickSize(Math.max(1, p - 1));
  return Math.max(1, Math.ceil(p / t) * t - t);
}

export const onTick = (price: number): boolean => Number.isInteger(price) && price > 0 && price % tickSize(price) === 0;

/** why an order cannot be sent ('' = it can). limits = today's upper / lower price limit when known */
export function orderProblem(o: OrderReq, limits?: { upper?: number; lower?: number }): string {
  if (!/^[0-9A-Z]{6}$/.test(o.code)) return '종목 코드를 확인하세요.';
  if (o.side !== 'buy' && o.side !== 'sell') return '매수 / 매도를 고르세요.';
  if (o.type !== 'limit' && o.type !== 'market') return '주문 유형을 고르세요.';
  if (!Number.isInteger(o.qty) || o.qty <= 0) return '수량을 1주 이상의 정수로 입력하세요.';
  if (o.qty > 1_000_000) return '수량이 너무 큽니다 (100만 주 초과).';
  if (o.type === 'market') return '';
  if (!Number.isInteger(o.price) || o.price <= 0) return '가격을 입력하세요.';
  if (!onTick(o.price)) return `호가 단위(${tickSize(o.price).toLocaleString('ko-KR')}원)에 맞지 않는 가격입니다.`;
  if (limits?.upper && o.price > limits.upper) return `상한가(${limits.upper.toLocaleString('ko-KR')}원)보다 높은 가격입니다.`;
  if (limits?.lower && o.price < limits.lower) return `하한가(${limits.lower.toLocaleString('ko-KR')}원)보다 낮은 가격입니다.`;
  return '';
}

/** ORD_DVSN: 00 지정가, 01 시장가 */
export const ordDvsn = (t: OrderType) => (t === 'market' ? '01' : '00');

/** one line a person reads before confirming: "매수 005930 지정가 70,000원 × 10주 = 700,000원" */
export function orderText(o: OrderReq, name?: string): string {
  const won = (n: number) => n.toLocaleString('ko-KR');
  const head = `${o.side === 'buy' ? '매수' : '매도'} ${name ? `${name}(${o.code})` : o.code}`;
  return o.type === 'market' ? `${head} 시장가 × ${won(o.qty)}주` : `${head} 지정가 ${won(o.price)}원 × ${won(o.qty)}주 = ${won(o.price * o.qty)}원`;
}

// ---- request bodies (KRX only: the window sends every order to the regular exchange) ----

const VENUE = 'KRX';

export function orderBody(a: Account, o: OrderReq): Record<string, string> {
  return {
    CANO: a.cano,
    ACNT_PRDT_CD: a.prod,
    PDNO: o.code,
    ORD_DVSN: ordDvsn(o.type),
    ORD_QTY: String(o.qty),
    ORD_UNPR: o.type === 'market' ? '0' : String(o.price),
    EXCG_ID_DVSN_CD: VENUE,
    SLL_TYPE: o.side === 'sell' ? '01' : '',
    CNDT_PRIC: '',
  };
}

export interface ReviseReq {
  no: string;
  branch: string;
  /** 'revise' = a new limit price for the whole remaining quantity, 'cancel' = cancel all of it */
  kind: 'revise' | 'cancel';
  type: OrderType;
  price: number;
  /** remaining quantity (sent with QTY_ALL_ORD_YN = Y) */
  qty: number;
  venue?: string;
}

export function reviseBody(a: Account, r: ReviseReq): Record<string, string> {
  return {
    CANO: a.cano,
    ACNT_PRDT_CD: a.prod,
    KRX_FWDG_ORD_ORGNO: r.branch,
    ORGN_ODNO: r.no,
    ORD_DVSN: ordDvsn(r.type),
    RVSE_CNCL_DVSN_CD: r.kind === 'cancel' ? '02' : '01',
    ORD_QTY: String(r.qty),
    ORD_UNPR: r.kind === 'cancel' || r.type === 'market' ? '0' : String(r.price),
    QTY_ALL_ORD_YN: 'Y',
    EXCG_ID_DVSN_CD: r.venue || VENUE,
  };
}

export function reviseProblem(r: ReviseReq): string {
  if (!r.no) return '주문번호가 없습니다.';
  if (r.kind !== 'revise' && r.kind !== 'cancel') return '정정 / 취소를 고르세요.';
  if (!Number.isInteger(r.qty) || r.qty <= 0) return '정정 / 취소할 잔량이 없습니다.';
  if (r.kind === 'revise') {
    if (r.type !== 'limit') return '정정은 지정가 주문만 할 수 있습니다.';
    if (!Number.isInteger(r.price) || r.price <= 0 || !onTick(r.price)) return '정정 가격을 호가 단위에 맞게 입력하세요.';
  }
  return '';
}

export const psblOrderParams = (a: Account, code: string, type: OrderType, price: number): Record<string, string> => ({
  CANO: a.cano,
  ACNT_PRDT_CD: a.prod,
  PDNO: code,
  ORD_UNPR: type === 'market' ? '0' : String(Math.max(0, Math.round(price) || 0)),
  ORD_DVSN: ordDvsn(type),
  CMA_EVLU_AMT_ICLD_YN: 'N',
  OVRS_ICLD_YN: 'N',
});

export const balanceParams = (a: Account, fk = '', nk = ''): Record<string, string> => ({
  CANO: a.cano,
  ACNT_PRDT_CD: a.prod,
  AFHR_FLPR_YN: 'N',
  OFL_YN: '',
  INQR_DVSN: '02',
  UNPR_DVSN: '01',
  FUND_STTL_ICLD_YN: 'N',
  FNCG_AMT_AUTO_RDPT_YN: 'N',
  PRCS_DVSN: '00',
  CTX_AREA_FK100: fk,
  CTX_AREA_NK100: nk,
});

export const openOrderParams = (a: Account, fk = '', nk = ''): Record<string, string> => ({
  CANO: a.cano,
  ACNT_PRDT_CD: a.prod,
  INQR_DVSN_1: '0',
  INQR_DVSN_2: '0',
  CTX_AREA_FK100: fk,
  CTX_AREA_NK100: nk,
});

/** 주식일별주문체결조회 of one day: ccld '01' = filled orders, '02' = unfilled ones; venue KRX / NXT / SOR / ALL */
export const dailyParams = (a: Account, ymd: string, ccld: '01' | '02', venue = 'KRX', fk = '', nk = ''): Record<string, string> => ({
  CANO: a.cano,
  ACNT_PRDT_CD: a.prod,
  INQR_STRT_DT: ymd,
  INQR_END_DT: ymd,
  SLL_BUY_DVSN_CD: '00',
  PDNO: '',
  CCLD_DVSN: ccld,
  INQR_DVSN: '00',
  INQR_DVSN_3: '00',
  ORD_GNO_BRNO: '',
  ODNO: '',
  INQR_DVSN_1: '',
  CTX_AREA_FK100: fk,
  CTX_AREA_NK100: nk,
  EXCG_ID_DVSN_CD: venue,
});
/** paper server: today's unfilled orders through 주식일별주문체결조회 (CCLD_DVSN 02) */
export const dailyOpenParams = (a: Account, ymd: string, fk = '', nk = ''): Record<string, string> => dailyParams(a, ymd, '02', 'KRX', fk, nk);

// ---- answers ----

const num = (v: unknown): number => {
  const n = Number(String(v ?? '').replace(/,/g, ''));
  return Number.isFinite(n) ? n : 0;
};
const str = (v: unknown) => String(v ?? '').trim();
const rows = (v: unknown): Record<string, unknown>[] => (Array.isArray(v) ? (v as Record<string, unknown>[]) : v && typeof v === 'object' ? [v as Record<string, unknown>] : []);

export function toBuyable(output: unknown): Buyable {
  const o = rows(output)[0] || {};
  return { cash: num(o.ord_psbl_cash), amount: num(o.nrcvb_buy_amt), qty: num(o.nrcvb_buy_qty), maxAmount: num(o.max_buy_amt), maxQty: num(o.max_buy_qty) };
}

export function toHoldings(output1: unknown): Holding[] {
  return rows(output1)
    .map((r) => ({
      code: str(r.pdno),
      name: str(r.prdt_name),
      qty: num(r.hldg_qty),
      sellable: num(r.ord_psbl_qty),
      avgPrice: num(r.pchs_avg_pric),
      price: num(r.prpr),
      pnl: num(r.evlu_pfls_amt),
      pnlPct: num(r.evlu_pfls_rt),
    }))
    .filter((h) => h.code && h.qty > 0);
}

export function toBalance(output1: unknown, output2: unknown): Balance {
  const s = rows(output2)[0] || {};
  return { cash: num(s.dnca_tot_amt), cashD2: num(s.prvs_rcdl_excc_amt), total: num(s.tot_evlu_amt), holdings: toHoldings(output1) };
}

/** rows of 정정취소가능주문조회 (psbl_qty) or 일별주문체결조회 (rmn_qty) -> open orders, newest first */
export function toOpenOrders(output: unknown): OpenOrder[] {
  return rows(output)
    .map((r) => {
      const qty = num(r.ord_qty);
      const filled = num(r.tot_ccld_qty);
      const remain = r.psbl_qty !== undefined ? num(r.psbl_qty) : r.rmn_qty !== undefined ? num(r.rmn_qty) : Math.max(0, qty - filled);
      return {
        no: str(r.odno),
        branch: str(r.ord_gno_brno),
        code: str(r.pdno),
        name: str(r.prdt_name),
        side: (str(r.sll_buy_dvsn_cd) === '01' ? 'sell' : 'buy') as OrderSide,
        type: (str(r.ord_dvsn_cd) === '01' ? 'market' : 'limit') as OrderType,
        price: num(r.ord_unpr),
        qty,
        filled,
        remain,
        time: str(r.ord_tmd),
        venue: str(r.excg_id_dvsn_cd) || VENUE,
      };
    })
    .filter((o) => o.no && o.remain > 0)
    .sort((a, b) => (a.time < b.time ? 1 : a.time > b.time ? -1 : 0));
}

/** rows of 일별주문체결조회 -> orders with a filled quantity, newest first (cancelled / unfilled rows dropped) */
export function toFills(output: unknown): Fill[] {
  return rows(output)
    .map((r) => {
      const qty = num(r.tot_ccld_qty);
      const amount = num(r.tot_ccld_amt);
      return {
        no: str(r.odno),
        code: str(r.pdno),
        name: str(r.prdt_name),
        side: (str(r.sll_buy_dvsn_cd) === '01' ? 'sell' : 'buy') as OrderSide,
        time: str(r.ord_tmd),
        qty,
        price: num(r.avg_prvs) || (qty > 0 ? Math.round(amount / qty) : 0),
        amount,
        ordQty: num(r.ord_qty),
        ordPrice: num(r.ord_unpr),
      };
    })
    .filter((f) => f.no && f.qty > 0)
    .sort((a, b) => (a.time < b.time ? 1 : a.time > b.time ? -1 : 0));
}

export function toOrderResult(output: unknown): OrderResult {
  const o = rows(output)[0] || {};
  return { no: str(o.ODNO ?? o.odno), branch: str(o.KRX_FWDG_ORD_ORGNO ?? o.krx_fwdg_ord_orgno), time: str(o.ORD_TMD ?? o.ord_tmd) };
}

/** what the order window knows about the session (main -> renderer; the account number only masked) */
export interface OrderState {
  /** orders can be sent now */
  ready: boolean;
  mode: 'kis' | 'demo' | 'guest' | null;
  env: KisEnv;
  /** masked account ("1234****-01"), '' when none is saved */
  account: string;
  /** why not ready (shown in the window) */
  reason: string;
}
