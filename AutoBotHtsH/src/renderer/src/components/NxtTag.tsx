// Update log
// - [2026-11-12] type=feat (NXT mark next to a symbol: shown when the stock master says the code also trades on NXT (넥스트레이드).
//   Used wherever a symbol is shown - watchlist / ranking rows, window titles, current price / info headers, chart toolbar, search list)

import { useSymbol } from '../hooks';

const TITLE = 'NXT(넥스트레이드) 거래 가능 종목 - 08:00~08:50, 15:40~20:00 에도 거래됩니다';

/** the mark itself (when the caller already knows the flag, e.g. search results) */
export function NxtMark() {
  return (
    <span className="nxt-tag" title={TITLE}>
      NXT
    </span>
  );
}

/** NXT mark of `code` from the stock master (nothing for other symbols) */
export function NxtTag({ code }: { code: string }) {
  const sym = useSymbol(code);
  return sym?.nxt ? <NxtMark /> : null;
}
