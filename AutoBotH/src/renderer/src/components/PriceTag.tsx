/*
 * Update log
 * - [2026-11-12] type=feat - Current price of a symbol next to its name: "11,650 ▲ 60 (+0.52%)" (red up / blue down), from the store's
 *   quotes (r003 quote log of the viewed day, refreshed with every snapshot; on a replayed past day the price at the replay time)
 */

import { useStore } from '../store';
import { quoteText } from '../../../shared/liveQuote';
import { fmtKstTime } from '../../../shared/time';

export function PriceTag({ code }: { code: string | null | undefined }) {
  const q = useStore((s) => (code ? s.quotes[code] : undefined));
  if (!q) return <span className="price-tag muted">-</span>;
  const x = quoteText(q);
  const cls = x.dir > 0 ? 'up' : x.dir < 0 ? 'down' : '';
  return (
    <span className={`price-tag ${cls}`} title={`${fmtKstTime(q.t)} 기준${q.market === 'NX' ? ' · NXT 시세' : ''} (r003 현재가 기록)`}>
      <b>{x.price}</b>
      {x.change ? ` ${x.change}` : ''} {x.pct}
    </span>
  );
}
