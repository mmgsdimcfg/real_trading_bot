// Update log
// - [2026-11-26] type=feat (Symbol input history (user 2026-11-26): the codes picked in a symbol box or shown in a symbol window, newest
//   first, no duplicates, at most 20 - pure list functions; the renderer keeps the list in localStorage (renderer/src/symHistory.ts))

export const SYM_HISTORY_MAX = 20;

const CODE_RE = /^[0-9A-Z]{6}$/;

/** the list with `code` moved / added to the front (unchanged list object when it is already first or not a code) */
export function pushSymHistory(list: readonly string[], code: string, max = SYM_HISTORY_MAX): string[] {
  const c = String(code || '').trim().toUpperCase();
  if (!CODE_RE.test(c) || list[0] === c) return list as string[];
  return [c, ...list.filter((x) => x !== c)].slice(0, max);
}

export function dropSymHistory(list: readonly string[], code: string): string[] {
  return list.filter((x) => x !== code);
}

/** a stored value -> a clean list (anything else: empty) */
export function parseSymHistory(text: string | null | undefined, max = SYM_HISTORY_MAX): string[] {
  try {
    const v: unknown = JSON.parse(text || '[]');
    if (!Array.isArray(v)) return [];
    const out: string[] = [];
    for (const x of v) {
      const c = typeof x === 'string' ? x.trim().toUpperCase() : '';
      if (CODE_RE.test(c) && !out.includes(c)) out.push(c);
      if (out.length >= max) break;
    }
    return out;
  } catch {
    return [];
  }
}
