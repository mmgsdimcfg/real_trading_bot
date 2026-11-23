// Update log
// - [2026-11-19] type=feat (Selection files "<...>YYYYMMDD_picks.txt" in the watchlist folder become watchlist tabs named MMDD (user
//   2026-11-19). A new or changed file (mtime) makes / refreshes its tab; a tab the user deleted is not made again until the file
//   changes; codes are capped at the group limit)
// - [2026-11-23] type=feat (user 2026-11-23: the bot watchlist r004 written by the 종목 선별 -> 감시목록 반영 job becomes a tab named
//   MMDD of the trading day it is for (header "# for_trading_day=YYYYMMDD", e.g. 11/23 data -> 1124) at the front of the watchlist -
//   r004Entry(); mergePicks(..., front) puts a new tab first)

export interface PickGroup {
  id: string;
  name: string;
  codes: string[];
  src?: string;
}
export interface PicksEntry {
  file: string;
  date: string;
  mtime: number;
  codes: string[];
}

/** tab name of a picks file / r004: MM/DD of its date ([2026-11-24] was MMDD) */
export const picksTabName = (date: string) => `${date.slice(4, 6)}/${date.slice(6, 8)}`;

/** [2026-11-24] tabs made from a picks file / r004 under the old MMDD name get the MM/DD name (user tabs are left alone) */
export function renamePickTabs<G extends PickGroup>(groups: G[]): { groups: G[]; changed: boolean } {
  let changed = false;
  const out = groups.map((g) => {
    if (!g.src || !/^\d{4}$/.test(g.name)) return g;
    changed = true;
    return { ...g, name: `${g.name.slice(0, 2)}/${g.name.slice(2)}` };
  });
  return { groups: changed ? out : groups, changed };
}

/** [2026-11-23] key of the r004 tab of a trading day (PickGroup.src / imported) */
export const r004Src = (day: string) => `r004:${day}`;

/** [2026-11-23] r004 (r016 `watchlist get`: header lines + codes) as a picks entry of the trading day it is for; null when the header
 *  has no for_trading_day (an r004 not written by the selection job) or the file is empty */
export function r004Entry(doc: { mtime: string | null; header: string[]; codes: { code: string }[] } | null | undefined): PicksEntry | null {
  if (!doc || !doc.codes?.length) return null;
  const line = doc.header.find((h) => /^#\s*for_trading_day=/.test(h));
  const day = line ? /for_trading_day=(\d{8})/.exec(line)?.[1] : undefined;
  if (!day) return null;
  const mtime = doc.mtime ? Date.parse(doc.mtime) || 1 : 1;
  return { file: r004Src(day), date: day, mtime, codes: doc.codes.map((c) => c.code) };
}

export function mergePicks(
  groups: PickGroup[],
  imported: Record<string, number>,
  picks: PicksEntry[],
  newId: () => string,
  max: number,
  /** [2026-11-23] a new tab goes first instead of last */
  front = false,
): { groups: PickGroup[]; imported: Record<string, number>; changed: boolean; added: string[] } {
  let out = groups;
  const imp = { ...imported };
  let changed = false;
  const added: string[] = [];
  for (const p of picks) {
    if (imp[p.file] === p.mtime) continue;
    imp[p.file] = p.mtime;
    changed = true;
    if (!p.codes.length) continue;
    const codes = p.codes.slice(0, max);
    const name = picksTabName(p.date);
    // the tab made from this file before (else one with the same MMDD name that no other file owns)
    const at = out.findIndex((g) => g.src === p.file) >= 0 ? out.findIndex((g) => g.src === p.file) : out.findIndex((g) => g.name === name && !g.src);
    if (at >= 0) out = out.map((g, i) => (i === at ? { ...g, codes, src: p.file } : g));
    else {
      const g = { id: newId(), name, codes, src: p.file };
      out = front ? [g, ...out] : [...out, g];
      added.push(name);
    }
  }
  return { groups: out, imported: imp, changed, added };
}
