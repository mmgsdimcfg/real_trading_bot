/*
 * Update log
 * - [2026-11-12] type=feat - NXT mark next to a symbol name: shown for symbols r003 watches with NXT=True (WATCH line of the day's
 *   log -> SymbolSummary.nxt). Symbols that are not on the day's watchlist have no flag and get no mark
 */

import { useStore } from '../store';

const TITLE = 'NXT(넥스트레이드) 거래 가능 종목 - 08:00~08:50, 15:40~20:00 에도 거래됩니다';

export function NxtMark() {
  return (
    <span className="nxt-tag" title={TITLE}>
      NXT
    </span>
  );
}

/** NXT mark of `code` from the day's snapshot (nothing when the symbol is not an NXT symbol or not watched) */
export function NxtTag({ code }: { code: string | null | undefined }) {
  const nxt = useStore((s) => (code ? s.snapshot?.symbols.find((x) => x.code === code)?.nxt : undefined));
  return nxt ? <NxtMark /> : null;
}
