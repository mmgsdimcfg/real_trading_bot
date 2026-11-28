// rebound-alert conditions per judged bar, with the app's own detector (shared/rebound.ts) and its default settings.
// [2026-11-28] study helper for AutoBot/data_simulation/g058_entry_condition_table.py (read-only: files in the given folder only).
// usage (from AutoBotHtsH): npx vite-node scripts/rebound-conds.ts <dir with bars_YYYYMMDD.json>   -> rebound_YYYYMMDD.csv beside each
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { COND_KEYS, detectRebound, REBOUND_DEFAULTS } from '../src/shared/rebound';

const dir = process.argv[process.argv.length - 1];
const KST = 9 * 3600_000;
const iso = (ms: number) => new Date(ms + KST).toISOString().slice(0, 19).replace('T', ' ');
for (const f of readdirSync(dir).filter((x) => /^bars_\d{8}\.json$/.test(x))) {
  const { day, syms } = JSON.parse(readFileSync(join(dir, f), 'utf8')) as { day: string; syms: { code: string; bars: number[][] }[] };
  const out = ['day,code,t,hit,required,score,' + COND_KEYS.join(',') + ',setup,edge'];
  for (const s of syms) {
    const bars = s.bars.map(([t, o, h, l, c, v]) => ({ t, o, h, l, c, v }));
    for (let k = 0; k < bars.length; k++) {
      const end = iso(bars[k].t + 60_000);
      if (end.slice(0, 10).replace(/-/g, '') !== day) continue;
      const hm = end.slice(11, 16);
      if (hm < '09:05' || hm > '14:50') continue;
      const r = detectRebound(bars.slice(0, k + 1), REBOUND_DEFAULTS);
      out.push([day, s.code, end, r.hit ? 1 : 0, r.required ? 1 : 0, r.score, ...COND_KEYS.map((c) => (r.conds[c].ok ? 1 : 0)), r.setup ?? '', r.edge ? 1 : 0].join(','));
    }
  }
  writeFileSync(join(dir, `rebound_${day}.csv`), out.join('\n'), 'utf8');
  console.log(day, out.length - 1);
}
