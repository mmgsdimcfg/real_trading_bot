// Update log
// - [2026-11-23] type=feat (Sort key 'code' (종목코드, ascending first) - user 2026-11-23)
// - [2026-11-23] type=refactor (Watchlist columns and sorting shared by the list window (column title clicks) and the tile window
//   (정렬 select + direction): the column names / labels, the saved sort (watchSortOf), the click cycle (nextWatchSort) and the sort
//   itself (sortWatch: symbols without a value last, ties in the registered order). Moved out of windows/ListWindows.tsx, which
//   re-exports them)

import type { WatchRow } from '../../shared/kis';

export type WatchCol = 'change' | 'changePct' | 'volume' | 'amount' | 'open' | 'high' | 'low' | 'ask' | 'bid';
/** menu entries: 'code' (shown inside the name cell) and the value columns, in table order */
export const WATCH_COLS: ('code' | WatchCol)[] = ['code', 'change', 'changePct', 'volume', 'amount', 'open', 'high', 'low', 'ask', 'bid'];
export const WATCH_COL_LABEL: Record<'code' | WatchCol, string> = {
  code: '종목코드',
  change: '대비',
  changePct: '등락률',
  volume: '거래량',
  amount: '거래대금',
  open: '시가',
  high: '고가',
  low: '저가',
  ask: '매도호가',
  bid: '매수호가',
};

export type WatchSortKey = 'name' | 'code' | 'price' | WatchCol;
export const SORT_KEYS: WatchSortKey[] = ['name', 'code', 'price', 'change', 'changePct', 'volume', 'amount', 'open', 'high', 'low', 'ask', 'bid'];
export interface WatchSort {
  key: WatchSortKey;
  dir: 1 | -1;
}
export const watchSortOf = (key: unknown, dir: unknown): WatchSort | null =>
  SORT_KEYS.includes(key as WatchSortKey) ? { key: key as WatchSortKey, dir: dir === 1 ? 1 : -1 } : null;
/** the next state of a click on column `key`: first click = high to low (names A to Z), second = the other way, third = no sort */
export function nextWatchSort(cur: WatchSort | null, key: WatchSortKey): WatchSort | null {
  const first = firstDir(key);
  if (!cur || cur.key !== key) return { key, dir: first };
  return cur.dir === first ? { key, dir: first === 1 ? -1 : 1 } : null;
}
/** codes in the sorted order (dir 1 = ascending); symbols without a value last, ties in the registered order */
export function sortWatch(codes: string[], rows: Map<string, WatchRow>, sort: WatchSort | null): string[] {
  if (!sort) return codes;
  const val = (c: string): number | string | null => {
    if (sort.key === 'code') return c;
    const x = rows.get(c);
    if (!x) return null;
    if (sort.key === 'name') return x.name || null;
    const v = x[sort.key];
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  };
  return codes
    .map((c, i) => ({ c, i, v: val(c) }))
    .sort((a, b) => {
      if (a.v === null || b.v === null) return (a.v === null ? 1 : 0) - (b.v === null ? 1 : 0) || a.i - b.i;
      const d = typeof a.v === 'string' || typeof b.v === 'string' ? String(a.v).localeCompare(String(b.v), 'ko') : a.v - b.v;
      return d * sort.dir || a.i - b.i;
    })
    .map((x) => x.c);
}

/** sort key -> label of the tile window's select */
export const SORT_LABEL: Record<WatchSortKey, string> = { name: '종목명', price: '현재가', ...WATCH_COL_LABEL } as Record<WatchSortKey, string>;
/** the direction a key starts with: names A to Z, numbers high to low */
export function firstDir(key: WatchSortKey): 1 | -1 {
  return key === 'name' || key === 'code' ? 1 : -1;
}
