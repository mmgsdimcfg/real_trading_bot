// Update log
// - [2026-11-26] type=feat (Order window backends (user 2026-11-26). Trader = what the window can do with an account: buyable amount,
//   balance, open orders, place / revise / cancel. KisTrader sends them through KisClient.trade() (shared/order.ts allowlist);
//   DemoTrader keeps a made-up account in memory for the demo mode and the screen tests - it never touches the network.
//   The main process (index.ts) asks the user to confirm every order in a native dialog before it calls place / revise / cancel)
// - [2026-11-26] type=feat (fills(): today's filled orders - 일별주문체결조회 CCLD_DVSN 01 (every exchange on the real server);
//   the demo account records its own fills)

import type { KisClient } from './kisClient';
import {
  balanceParams,
  dailyOpenParams,
  dailyParams,
  openOrderParams,
  orderBody,
  orderProblem,
  psblOrderParams,
  reviseBody,
  reviseProblem,
  toBalance,
  toBuyable,
  toFills,
  toHoldings,
  toOpenOrders,
  toOrderResult,
  type Account,
  type Balance,
  type Buyable,
  type Fill,
  type Holding,
  type OpenOrder,
  type OrderReq,
  type OrderResult,
  type OrderType,
  type ReviseReq,
} from '../shared/order';

export interface Trader {
  readonly kind: 'kis' | 'demo';
  buyable(code: string, type: OrderType, price: number): Promise<Buyable>;
  balance(): Promise<Balance>;
  openOrders(): Promise<OpenOrder[]>;
  fills(): Promise<Fill[]>;
  place(o: OrderReq): Promise<OrderResult>;
  revise(r: ReviseReq): Promise<OrderResult>;
}

const KST = 9 * 3600000;
const kstYmd = () => new Date(Date.now() + KST).toISOString().slice(0, 10).replace(/-/g, '');
const kstHms = () => new Date(Date.now() + KST).toISOString().slice(11, 19).replace(/:/g, '');
/** pages of one inquiry at most (100 rows a page on the real server) */
const MAX_PAGES = 5;

export class KisTrader implements Trader {
  readonly kind = 'kis' as const;
  constructor(
    private client: KisClient,
    private acct: Account,
  ) {}

  async buyable(code: string, type: OrderType, price: number): Promise<Buyable> {
    const r = await this.client.trade('psbl_order', psblOrderParams(this.acct, code, type, price));
    return toBuyable(r.output);
  }

  async balance(): Promise<Balance> {
    let fk = '';
    let nk = '';
    let cont = '';
    const holdings: Holding[] = [];
    let first: Balance | null = null;
    for (let page = 0; page < MAX_PAGES; page++) {
      const r = await this.client.trade('balance', balanceParams(this.acct, fk, nk), cont);
      if (!first) first = toBalance(r.output1, r.output2);
      else holdings.push(...toHoldings(r.output1));
      if (r.trCont !== 'F' && r.trCont !== 'M') break;
      // more rows: the continuation keys come back in the body of the answer
      fk = r.ctx?.fk || '';
      nk = r.ctx?.nk || '';
      if (!nk.trim()) break;
      cont = 'N';
    }
    const b = first || { cash: 0, cashD2: 0, total: 0, holdings: [] };
    return { ...b, holdings: [...b.holdings, ...holdings] };
  }

  async openOrders(): Promise<OpenOrder[]> {
    if (this.client.env === 'paper') {
      const r = await this.client.trade('daily_ccld', dailyOpenParams(this.acct, kstYmd()));
      return toOpenOrders(r.output1);
    }
    const r = await this.client.trade('psbl_rvsecncl', openOrderParams(this.acct));
    return toOpenOrders(r.output);
  }

  async fills(): Promise<Fill[]> {
    const venue = this.client.env === 'real' ? 'ALL' : 'KRX';
    const out: Fill[] = [];
    let fk = '';
    let nk = '';
    let cont = '';
    for (let page = 0; page < MAX_PAGES; page++) {
      const r = await this.client.trade('daily_ccld', dailyParams(this.acct, kstYmd(), '01', venue, fk, nk), cont);
      out.push(...toFills(r.output1));
      if (r.trCont !== 'F' && r.trCont !== 'M') break;
      fk = r.ctx?.fk || '';
      nk = r.ctx?.nk || '';
      if (!nk.trim()) break;
      cont = 'N';
    }
    return out.sort((a, b) => (a.time < b.time ? 1 : a.time > b.time ? -1 : 0));
  }

  async place(o: OrderReq): Promise<OrderResult> {
    const bad = orderProblem(o);
    if (bad) throw new Error(bad);
    const r = await this.client.trade(o.side === 'buy' ? 'order_buy' : 'order_sell', orderBody(this.acct, o));
    return toOrderResult(r.output);
  }

  async revise(req: ReviseReq): Promise<OrderResult> {
    const bad = reviseProblem(req);
    if (bad) throw new Error(bad);
    const r = await this.client.trade('order_rvsecncl', reviseBody(this.acct, req));
    return toOrderResult(r.output);
  }
}

/** demo account: 10,000,000 won, orders kept in memory. A limit order that crosses the current price and every market order fill at
 *  once at the current price; the rest stay open until they are revised across the price or cancelled */
export class DemoTrader implements Trader {
  readonly kind = 'demo' as const;
  private cash = 10_000_000;
  private held = new Map<string, { qty: number; cost: number }>();
  private open: OpenOrder[] = [];
  private done: Fill[] = [];
  private seq = 1000;

  constructor(
    private priceOf: (code: string) => Promise<number>,
    private nameOf: (code: string) => string,
  ) {}

  /** cash not tied up in open buy orders */
  private freeCash(): number {
    return this.cash - this.open.filter((o) => o.side === 'buy').reduce((s, o) => s + o.price * o.remain, 0);
  }
  private sellable(code: string): number {
    return (this.held.get(code)?.qty || 0) - this.open.filter((o) => o.side === 'sell' && o.code === code).reduce((s, o) => s + o.remain, 0);
  }

  async buyable(code: string, type: OrderType, price: number): Promise<Buyable> {
    const p = type === 'market' || !(price > 0) ? await this.priceOf(code) : price;
    const cash = Math.max(0, this.freeCash());
    const qty = p > 0 ? Math.floor(cash / p) : 0;
    return { cash, amount: cash, qty, maxAmount: cash, maxQty: qty };
  }

  async balance(): Promise<Balance> {
    const holdings: Holding[] = [];
    let value = 0;
    for (const [code, h] of this.held) {
      if (h.qty <= 0) continue;
      const price = await this.priceOf(code);
      const avg = h.cost / h.qty;
      value += price * h.qty;
      holdings.push({ code, name: this.nameOf(code), qty: h.qty, sellable: Math.max(0, this.sellable(code)), avgPrice: Math.round(avg), price, pnl: Math.round((price - avg) * h.qty), pnlPct: avg ? ((price - avg) / avg) * 100 : 0 });
    }
    return { cash: this.cash, cashD2: this.cash, total: this.cash + value, holdings };
  }

  async openOrders(): Promise<OpenOrder[]> {
    return [...this.open].reverse();
  }

  async fills(): Promise<Fill[]> {
    return [...this.done].reverse();
  }

  private fill(code: string, side: 'buy' | 'sell', qty: number, price: number, no: string, ordPrice: number): void {
    this.done.push({ no, code, name: this.nameOf(code), side, time: kstHms(), qty, price, amount: price * qty, ordQty: qty, ordPrice });
    const h = this.held.get(code) || { qty: 0, cost: 0 };
    if (side === 'buy') {
      this.cash -= price * qty;
      this.held.set(code, { qty: h.qty + qty, cost: h.cost + price * qty });
    } else {
      const avg = h.qty ? h.cost / h.qty : 0;
      this.cash += price * qty;
      this.held.set(code, { qty: h.qty - qty, cost: avg * (h.qty - qty) });
    }
  }

  async place(o: OrderReq): Promise<OrderResult> {
    const bad = orderProblem(o);
    if (bad) throw new Error(bad);
    const now = await this.priceOf(o.code);
    const price = o.type === 'market' ? now : o.price;
    if (o.side === 'buy' && price * o.qty > this.freeCash()) throw new Error('주문가능금액을 초과했습니다. (데모)');
    if (o.side === 'sell' && o.qty > this.sellable(o.code)) throw new Error('매도가능수량을 초과했습니다. (데모)');
    const no = String(++this.seq).padStart(10, '0');
    const time = kstHms();
    const crosses = o.type === 'market' || (o.side === 'buy' ? o.price >= now : o.price <= now);
    if (crosses) this.fill(o.code, o.side, o.qty, now, no, o.type === 'market' ? 0 : o.price);
    else this.open.push({ no, branch: '00950', code: o.code, name: this.nameOf(o.code), side: o.side, type: o.type, price, qty: o.qty, filled: 0, remain: o.qty, time, venue: 'KRX' });
    return { no, branch: '00950', time };
  }

  async revise(r: ReviseReq): Promise<OrderResult> {
    const bad = reviseProblem(r);
    if (bad) throw new Error(bad);
    const i = this.open.findIndex((o) => o.no === r.no);
    if (i < 0) throw new Error('정정 / 취소할 주문이 없습니다. (데모)');
    const [old] = this.open.splice(i, 1);
    if (r.kind === 'cancel') return { no: String(++this.seq).padStart(10, '0'), branch: old.branch, time: kstHms() };
    return this.place({ code: old.code, side: old.side, type: 'limit', qty: old.remain, price: r.price });
  }
}
