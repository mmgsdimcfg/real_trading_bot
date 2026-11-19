// Update log
// - [2026-11-06] type=feat (HtsH shared types: Trade marker used by PriceChart (copied from AutoBotH) - HtsH has no trading, the chart
//   gets an empty list or user-placed markers)

export interface Trade {
  side: 'BUY' | 'SELL';
  ts: number;
  price: number;
  qty?: number;
  reason: string;
}
