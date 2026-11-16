/*
 * Update log
 * - [2026-11-16] type=feat - `fallback`: a quote the caller fetched itself (KIS, for symbols r003 does not quote), used when the
 *   store has none
 * - [2026-11-16] type=feat - `detail`: the timeline's quote line like a trading screen - price, ▲/▼ change, rate, then the day's
 *   open / high / low (each colored against the previous close), volume and traded value, and the time of the quote. Without a quote
 *   it says why ("현재가 기록 없음" - r003 writes its first quote at its first poll of the session)
 * - [2026-11-12] type=feat - Current price of a symbol next to its name: "11,650 ▲ 60 (+0.52%)" (red up / blue down), from the store's
 *   quotes (r003 quote log of the viewed day, refreshed with every snapshot; on a replayed past day the price at the replay time)
 */

import { useStore } from '../store';
import { fmtKrwShort, quoteText, type LiveQuote } from '../../../shared/liveQuote';
import { fmtKstTime } from '../../../shared/time';

const n = (v: number) => Math.round(v).toLocaleString('ko-KR');

export function PriceTag({ code, detail = false, fallback }: { code: string | null | undefined; detail?: boolean; fallback?: LiveQuote | null }) {
  const logged = useStore((s) => (code ? s.quotes[code] : undefined));
  const q = logged || (fallback && fallback.code === code ? fallback : undefined);
  if (!q) {
    return detail ? (
      <span className="price-tag muted" title="r003 이 이 종목의 현재가를 아직 기록하지 않았습니다 (장 시작 후 첫 폴링부터 기록, NXT 종목은 08:00 이후)">
        현재가 기록 없음
      </span>
    ) : (
      <span className="price-tag muted">-</span>
    );
  }
  const x = quoteText(q);
  const cls = x.dir > 0 ? 'up' : x.dir < 0 ? 'down' : '';
  const title = `${fmtKstTime(q.t)} 기준${q.market === 'NX' ? ' · NXT 시세' : ''} (${q.source === 'kis' ? 'KIS 현재가 조회 - 감시 종목이 아니라 r003 기록이 없음' : 'r003 현재가 기록'})`;
  const main = (
    <span className={`price-tag ${cls}`} title={title}>
      <b>{x.price}</b>
      {x.change ? ` ${x.change}` : ''} {x.pct}
    </span>
  );
  if (!detail) return main;
  const prev = q.price - q.change;
  const side = (v: number | undefined) => (v === undefined || !(prev > 0) ? '' : v > prev ? 'up' : v < prev ? 'down' : '');
  const cell = (label: string, v: number | undefined) =>
    v === undefined ? null : (
      <span className="qd-cell">
        {label} <b className={side(v)}>{n(v)}</b>
      </span>
    );
  return (
    <span className="quote-line">
      {main}
      <span className="quote-detail" title={title}>
        {cell('시', q.open)}
        {cell('고', q.high)}
        {cell('저', q.low)}
        {prev > 0 && q.change !== 0 && (
          <span className="qd-cell">
            전일 <b>{n(prev)}</b>
          </span>
        )}
        {q.vol !== undefined && (
          <span className="qd-cell">
            거래량 <b>{n(q.vol)}</b>
          </span>
        )}
        {q.value !== undefined && (
          <span className="qd-cell">
            거래대금 <b>{fmtKrwShort(q.value)}</b>
          </span>
        )}
        <span className="qd-cell muted">
          {fmtKstTime(q.t).slice(0, 5)}
          {q.market === 'NX' ? ' NXT' : ''}
          {q.source === 'kis' ? ' KIS' : ''}
        </span>
      </span>
    </span>
  );
}
