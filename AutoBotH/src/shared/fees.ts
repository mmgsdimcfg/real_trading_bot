/*
 * Update log
 * - [2026-11-11] type=feat - Trading costs for the "제세금 포함" switch of the trade-results / positions tabs: the bot's own rates
 *   (r001 V2_COMMISSION_PCT per side incl. exchange fees, V2_SELL_TAX_PCT securities transaction tax + rural tax on sells -
 *   r003 uses the same formula for its daily net P&L). The buy_sell log's 수익금액 is gross (price difference x qty), so the
 *   net figure is gross minus these costs
 */

/** commission per side, percent of the traded value (KIS non-face-to-face ~0.014% + exchange fees) - r001 V2_COMMISSION_PCT */
export const FEE_COMMISSION_PCT = 0.015;
/** securities transaction tax (+ rural development tax) on sells, percent - r001 V2_SELL_TAX_PCT (2026 rate) */
export const FEE_SELL_TAX_PCT = 0.2;

/** estimated costs of a round trip (KRW): commission on the buy, commission + tax on the sell */
export function tradeCosts(buyPrice: number, sellPrice: number, qty: number): number {
  if (!(buyPrice > 0) || !(sellPrice > 0) || !(qty > 0)) return 0;
  return (buyPrice * qty * FEE_COMMISSION_PCT + sellPrice * qty * (FEE_COMMISSION_PCT + FEE_SELL_TAX_PCT)) / 100;
}

/** gross P&L of a position (KRW) */
export const grossPnl = (buyPrice: number, price: number, qty: number): number => (price - buyPrice) * qty;

/** P&L (KRW) with or without the estimated costs */
export function pnlKrw(buyPrice: number, price: number, qty: number, net: boolean): number {
  const g = grossPnl(buyPrice, price, qty);
  return net ? g - tradeCosts(buyPrice, price, qty) : g;
}

/** P&L in percent of the buy amount, with or without the estimated costs */
export function pnlPct(buyPrice: number, price: number, qty: number, net: boolean): number {
  if (!(buyPrice > 0) || !(qty > 0)) return NaN;
  return (pnlKrw(buyPrice, price, qty, net) / (buyPrice * qty)) * 100;
}

/** label for the switch: "수수료 0.015% x 2 + 거래세 0.20%" */
export const feeLabel = (): string => `수수료 ${FEE_COMMISSION_PCT}% x 2 + 거래세 ${FEE_SELL_TAX_PCT.toFixed(2)}%`;
