// Update log
// - [2026-11-26] type=feat (Symbol input history shared by every symbol box (user 2026-11-26): the last 20 codes picked in a box or
//   shown in a symbol window, kept in localStorage (htsh.symHistory.v1); useSymHistory re-renders the open boxes on a change)

import { useSyncExternalStore } from 'react';
import { dropSymHistory, parseSymHistory, pushSymHistory } from '../../shared/symHistory';

const KEY = 'htsh.symHistory.v1';

let list: string[] = (() => {
  try {
    return parseSymHistory(localStorage.getItem(KEY));
  } catch {
    return [];
  }
})();
const subs = new Set<() => void>();

function set(next: string[]): void {
  if (next === list) return;
  list = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* the list still works for this session */
  }
  subs.forEach((f) => f());
}

/** a symbol was picked / shown: it becomes the first entry */
export const rememberSymbol = (code: string): void => set(pushSymHistory(list, code));
export const forgetSymbol = (code: string): void => set(dropSymHistory(list, code));
export const clearSymHistory = (): void => set([]);

const subscribe = (f: () => void) => {
  subs.add(f);
  return () => {
    subs.delete(f);
  };
};

export const useSymHistory = (): string[] => useSyncExternalStore(subscribe, () => list);
