// Update log
// - [2026-11-06] type=fix (useCall: a changed poll interval no longer clears the data - the current price window went blank ("불러오는 중")
//   every time the realtime state changed, because the window polls faster while realtime is down; only a new method/args resets it)
// - [2026-11-06] type=feat (Renderer hooks: useCall - one market inquiry with optional polling (skipped while the window is hidden or
//   another call of the same window is in flight), useRealtime - reference-counted realtime subscription of a code (trades and/or order
//   book) for as long as the window shows it, useSymbol - master entry (name, market) of a code)

// - [2026-11-12] type=fix (useSymbol cache is keyed by the stock master's date: entries read before today's master (with the NXT
//   flags) arrived are asked again instead of staying for the whole session)

import { useEffect, useRef, useState } from 'react';
import type { MasterItem } from '../../shared/kis';
import { useStore } from './store';

export interface CallState<T> {
  data: T | null;
  error: string;
  loading: boolean;
  at: number;
  reload: () => void;
}

/** market inquiry `method(...args)`; pollMs > 0 repeats it (newest result kept on errors) */
export function useCall<T>(method: string, args: unknown[], pollMs = 0, enabled = true): CallState<T> {
  const [st, setSt] = useState<{ data: T | null; error: string; loading: boolean; at: number }>({ data: null, error: '', loading: false, at: 0 });
  const key = JSON.stringify([method, args]);
  const busy = useRef(false);
  const lastKey = useRef('');
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    // only a different query clears the data; a new poll interval (e.g. realtime connected / dropped) keeps what is shown
    const newQuery = lastKey.current !== key;
    lastKey.current = key;
    const run = async (first: boolean) => {
      if (busy.current) {
        // a call of the previous effect is still running - try again shortly instead of dropping the poll loop
        timer = setTimeout(() => void run(first), 150);
        return;
      }
      busy.current = true;
      if (first && newQuery) setSt((s) => ({ ...s, loading: true, data: null, error: '' }));
      const r = await window.hts.call<T>(method, ...args);
      busy.current = false;
      if (!alive) {
        // the effect restarted while this call was running (interval change): keep its result if it is still the same query
        if (lastKey.current === key && r.ok) setSt({ data: r.data, error: '', loading: false, at: Date.now() });
        return;
      }
      if (r.ok) setSt({ data: r.data, error: '', loading: false, at: Date.now() });
      else setSt((s) => ({ ...s, error: r.error, loading: false }));
      if (pollMs > 0) timer = setTimeout(() => void run(false), pollMs);
    };
    void run(true);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [key, pollMs, enabled, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  return { ...st, reload: () => setTick((t) => t + 1) };
}

/** subscribe realtime trades / order book of `code` while mounted */
export function useRealtime(code: string, what: { trades?: boolean; book?: boolean }): void {
  useEffect(() => {
    if (!/^[0-9A-Z]{6}$/.test(code)) return;
    if (what.trades) window.hts.sub('H0STCNT0', code);
    if (what.book) window.hts.sub('H0STASP0', code);
    return () => {
      if (what.trades) window.hts.unsub('H0STCNT0', code);
      if (what.book) window.hts.unsub('H0STASP0', code);
    };
  }, [code, what.trades, what.book]);
}

const symCache = new Map<string, MasterItem | null>();
export function useSymbol(code: string): MasterItem | null {
  const rev = useStore((st) => st.auth?.master.date || '');
  const key = `${rev}|${code}`;
  const [s, setS] = useState<MasterItem | null>(symCache.get(key) ?? null);
  useEffect(() => {
    if (symCache.has(key)) return setS(symCache.get(key) ?? null);
    let alive = true;
    void window.hts.symbol(code).then((m) => {
      symCache.set(key, m);
      if (alive) setS(m);
    });
    return () => {
      alive = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return s;
}
