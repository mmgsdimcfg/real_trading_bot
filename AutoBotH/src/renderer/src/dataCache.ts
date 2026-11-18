/*
 * Update log
 * - [2026-11-18] type=perf - Renderer-side memory of data already fetched (charts, quotes, timeline, job / collector / control panels).
 *   Views unmount on every tab switch and refetched everything: a symbol switch on the timeline / chart tab waited for the main
 *   process (and r016 on the Raspberry Pi) before drawing anything. A view now shows the last answer for the same key at once and
 *   refreshes it in the background (stale-while-revalidate); only successful answers are kept. Bounded per kind (LRU).
 */

interface Entry {
  at: number;
  value: unknown;
}

/** entries kept per kind (first key segment before ':'); everything else shares the default */
const LIMITS: Record<string, number> = { price: 80, quote: 80, timeline: 40 };
const DEFAULT_LIMIT = 60;

const kinds = new Map<string, Map<string, Entry>>();

function bucket(key: string): Map<string, Entry> {
  const kind = key.split(':', 1)[0];
  let m = kinds.get(kind);
  if (!m) kinds.set(kind, (m = new Map()));
  return m;
}

/** the last value stored under `key` (undefined = never fetched) */
export function peek<T>(key: string): T | undefined {
  const e = bucket(key).get(key);
  return e ? (e.value as T) : undefined;
}

/** when `key` was stored (0 = never) */
export function storedAt(key: string): number {
  return bucket(key).get(key)?.at || 0;
}

/** store `value` under `key` (most recent last; the oldest of the kind is dropped past its limit) */
export function put<T>(key: string, value: T): T {
  const m = bucket(key);
  m.delete(key);
  m.set(key, { at: Date.now(), value });
  const limit = LIMITS[key.split(':', 1)[0]] ?? DEFAULT_LIMIT;
  while (m.size > limit) m.delete(m.keys().next().value as string);
  return value;
}

/** forget one key, or every key starting with `prefix` (e.g. after a host / connection change) */
export function drop(prefix: string): void {
  for (const m of kinds.values()) for (const k of [...m.keys()]) if (k === prefix || k.startsWith(prefix)) m.delete(k);
}

/** forget everything (run mode switch / reconnect to another host) */
export function clearAll(): void {
  kinds.clear();
}

/**
 * Fetch and remember: `fetcher()` result is stored under `key` when `keep(result)` (default: always).
 * Returns the fresh result; the caller shows peek(key) first if it wants the cached one meanwhile.
 */
export async function refresh<T>(key: string, fetcher: () => Promise<T>, keep: (v: T) => boolean = () => true): Promise<T> {
  const v = await fetcher();
  if (keep(v)) put(key, v);
  return v;
}
