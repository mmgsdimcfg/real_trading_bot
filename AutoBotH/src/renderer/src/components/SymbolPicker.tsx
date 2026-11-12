import { useStore } from '../store';
import { quoteText, type LiveQuote } from '../../../shared/liveQuote';

/** [2026-11-12] "11,650 ▲ 60 (+0.52%)" for the option text (an <option> cannot hold coloured markup) */
const quoteLine = (q: LiveQuote) => {
  const x = quoteText(q);
  return `${x.price} ${x.change} ${x.pct}`.replace(/\s+/g, ' ');
};

export function SymbolPicker({ allowAll = true }: { allowAll?: boolean }) {
  const snap = useStore((s) => s.snapshot);
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  const quotes = useStore((s) => s.quotes);
  return (
    <select className="picker" value={selected || ''} onChange={(e) => select(e.target.value || null)}>
      {allowAll && <option value="">전체 종목 (깔때기)</option>}
      {!allowAll && <option value="">종목 선택…</option>}
      {snap?.symbols.map((s) => (
        <option key={s.code} value={s.code}>
          {s.position ? '● ' : ''}
          {s.code} {s.name}
          {s.nxt ? ' [NXT]' : ''}
          {quotes[s.code] ? `  ${quoteLine(quotes[s.code])}` : ''}
        </option>
      ))}
    </select>
  );
}
