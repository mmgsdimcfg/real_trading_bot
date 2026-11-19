// Update log
// - [2026-11-19] type=feat (Selection files "<...>YYYYMMDD_picks.txt" in the watchlist folder become watchlist tabs named MMDD (user
//   2026-11-19). A new or changed file (mtime) makes / refreshes its tab; a tab the user deleted is not made again until the file
//   changes; codes are capped at the group limit)

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

/** tab name of a picks file: MMDD of its date */
export const picksTabName = (date: string) => date.slice(4, 8);

export function mergePicks(
  groups: PickGroup[],
  imported: Record<string, number>,
  picks: PicksEntry[],
  newId: () => string,
  max: number,
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
      out = [...out, { id: newId(), name, codes, src: p.file }];
      added.push(name);
    }
  }
  return { groups: out, imported: imp, changed, added };
}
