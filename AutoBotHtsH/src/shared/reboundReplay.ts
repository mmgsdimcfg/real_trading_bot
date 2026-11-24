/*
 * Update log
 * - [2026-11-21] type=feat - Replay of the rebound detector (user 2026-11-21): saved 1-minute bars (CSV, or a chart cache JSON of the
 *   app) are fed one bar at a time to detectRebound(), exactly as the poller does with completed bars, and every detection is listed
 *   (`npm run replay -- <file> [--date YYYYMMDD] [--from HH:MM] [--to HH:MM] [--explain HH:MM] [--all] [--config file]
 *   [--set key=value ...]`). parseBars / replay are pure (tests/rebound.test.ts); main() is the CLI (scripts/rebound-replay.cjs)
 * - [2026-11-24] type=feat - lowerHigh: a detection blocked by lowerHigh.block is listed as "감지(고점 낮아짐, 알림 안 함)" and does not
 *   start the cooldown (as in the poller); --explain / --all also print the lower highs check
 */

import { readFileSync } from 'node:fs';
import { BONUS_KEYS, BONUS_LABEL, COND_KEYS, COND_LABEL, detectRebound, hhmm, lowerHighLabel, mergeConfig, REBOUND_DEFAULTS, type ReboundConfig, type ReboundResult } from './rebound';
import type { Bar } from './price';

const KST = 9 * 3600000;
export { REBOUND_DEFAULTS as DEFAULTS } from './rebound';
export const kstYmd = (t: number) => new Date(t + KST).toISOString().slice(0, 10).replace(/-/g, '');

/**
 * Bars of a file: the app's chart cache JSON ({"bars":[{t,o,h,l,c,v}]}, or the array itself) or CSV with a header -
 * `date,time,open,high,low,close,volume` (date YYYYMMDD or YYYY-MM-DD, time HH:MM[:SS] or HHMMSS, KST) or `t,o,h,l,c,v` (t = epoch ms).
 * Sorted by time; rows that do not parse are dropped.
 */
export function parseBars(text: string): Bar[] {
  const s = text.replace(/^﻿/, '').trim();
  let bars: Bar[] = [];
  if (s.startsWith('{') || s.startsWith('[')) {
    const j = JSON.parse(s) as { bars?: unknown } | unknown[];
    const arr = (Array.isArray(j) ? j : (j as { bars?: unknown }).bars) as Record<string, unknown>[] | undefined;
    bars = (arr || []).map((b) => ({ t: Number(b.t), o: Number(b.o), h: Number(b.h), l: Number(b.l), c: Number(b.c), v: Number(b.v) }));
  } else {
    const lines = s.split(/\r?\n/);
    const head = lines[0].split(',').map((x) => x.trim().toLowerCase());
    const col = (...names: string[]) => head.findIndex((h) => names.includes(h));
    const ci = { t: col('t'), date: col('date'), time: col('time'), o: col('open', 'o'), h: col('high', 'h'), l: col('low', 'l'), c: col('close', 'c'), v: col('volume', 'v') };
    for (const line of lines.slice(1)) {
      const f = line.split(',').map((x) => x.trim());
      let t = NaN;
      if (ci.t >= 0) t = Number(f[ci.t]);
      else if (ci.date >= 0 && ci.time >= 0) {
        const d = f[ci.date].replace(/-/g, '');
        const m = f[ci.time].replace(/:/g, '').padEnd(6, '0');
        if (/^\d{8}$/.test(d) && /^\d{6}$/.test(m)) t = Date.UTC(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8), +m.slice(0, 2), +m.slice(2, 4), +m.slice(4, 6)) - KST;
      }
      bars.push({ t, o: Number(f[ci.o]), h: Number(f[ci.h]), l: Number(f[ci.l]), c: Number(f[ci.c]), v: Number(f[ci.v]) });
    }
  }
  return bars.filter((b) => [b.t, b.o, b.h, b.l, b.c, b.v].every((x) => Number.isFinite(x))).sort((a, b) => a.t - b.t);
}

export function barsToCsv(bars: Bar[]): string {
  const rows = bars.map((b) => `${kstYmd(b.t)},${hhmm(b.t)},${b.o},${b.h},${b.l},${b.c},${b.v}`);
  return ['date,time,open,high,low,close,volume', ...rows].join('\n') + '\n';
}

export interface ReplayRow {
  t: number;
  res: ReboundResult;
  /** would be alerted (the first detection, or the cooldown has passed) */
  alert: boolean;
}

/**
 * Feed the bars one at a time: bar k is judged with the bars up to k of its own day and the poll.days - 1 trading days before it -
 * what the poller holds at that minute. `from` / `to` (epoch ms, inclusive) limit the bars judged. Returns one row per judged bar.
 */
export function replay(bars: Bar[], cfg: ReboundConfig = REBOUND_DEFAULTS, from = -Infinity, to = Infinity): ReplayRow[] {
  const out: ReplayRow[] = [];
  let lastAlert = -Infinity;
  const dayStart: number[] = []; // index of the first bar of each KST day
  const days = Math.max(1, Math.round(cfg.poll.days));
  for (let k = 0; k < bars.length; k++) {
    const t = bars[k].t;
    if (!k || kstYmd(t) !== kstYmd(bars[k - 1].t)) dayStart.push(k);
    if (t < from || t > to) continue;
    const res = detectRebound(bars.slice(dayStart[Math.max(0, dayStart.length - days)], k + 1), cfg);
    const alert = res.hit && !res.lowerHigh?.blocked && t - lastAlert >= cfg.alert.cooldownMin * 60000; // a blocked one does not start the cooldown
    if (alert) lastAlert = t;
    out.push({ t, res, alert });
  }
  return out;
}

/** "a.b.c=1" overrides on a config (numbers, true / false, "mts" ...) */
export function applySets(cfg: ReboundConfig, sets: string[]): ReboundConfig {
  const raw = JSON.parse(JSON.stringify(cfg)) as Record<string, unknown>;
  for (const s of sets) {
    const m = /^([\w.]+)=(.*)$/.exec(s);
    if (!m) throw new Error(`--set ${s}: key=value 형식이 아닙니다`);
    const path = m[1].split('.');
    let o = raw;
    for (const k of path.slice(0, -1)) {
      if (!o[k] || typeof o[k] !== 'object') throw new Error(`--set ${s}: 없는 설정 항목`);
      o = o[k] as Record<string, unknown>;
    }
    const last = path[path.length - 1];
    if (!(last in o)) throw new Error(`--set ${s}: 없는 설정 항목`);
    const v = m[2];
    o[last] = v === 'true' ? true : v === 'false' ? false : v !== '' && Number.isFinite(Number(v)) ? Number(v) : v;
  }
  return mergeConfig(raw);
}

function table(res: ReboundResult): string[] {
  const mark = (ok: boolean) => (ok ? 'O' : 'x');
  return [
    ...COND_KEYS.map((k) => `    [${mark(res.conds[k].ok)}] ${COND_LABEL[k]}: ${res.conds[k].note}`),
    ...BONUS_KEYS.map((k) => `    (${mark(res.bonus[k].ok)}) ${BONUS_LABEL[k]}: ${res.bonus[k].note}`),
    ...(res.lowerHigh ? [`    {${res.lowerHigh.state}} ${lowerHighLabel(res.lowerHigh)}: ${res.lowerHigh.note}`] : []),
  ];
}

export function main(argv: string[], print: (s: string) => void = console.log): number {
  const opt: Record<string, string> = {};
  const sets: string[] = [];
  const files: string[] = [];
  for (let a = 0; a < argv.length; a++) {
    const x = argv[a];
    if (x === '--all') opt.all = '1';
    else if (x === '--set') sets.push(argv[++a] || '');
    else if (x.startsWith('--')) opt[x.slice(2)] = argv[++a] || '';
    else files.push(x);
  }
  if (!files.length) {
    print('사용법: npm run replay -- <분봉 CSV 또는 차트 캐시 JSON> [--date YYYYMMDD] [--from HH:MM] [--to HH:MM] [--explain HH:MM] [--all] [--config 파일] [--set 항목=값]');
    return 2;
  }
  let cfg = REBOUND_DEFAULTS;
  if (opt.config) cfg = mergeConfig(JSON.parse(readFileSync(opt.config, 'utf8')));
  if (sets.length) cfg = applySets(cfg, sets);
  const bars = parseBars(readFileSync(files[0], 'utf8'));
  if (!bars.length) {
    print('봉을 읽지 못했습니다.');
    return 1;
  }
  const date = opt.date || kstYmd(bars[bars.length - 1].t);
  const at = (hm: string) => Date.UTC(+date.slice(0, 4), +date.slice(4, 6) - 1, +date.slice(6, 8), +hm.slice(0, 2), +hm.slice(3, 5)) - KST;
  const from = at(opt.from || '00:00');
  const to = at(opt.to || '23:59');
  const rows = replay(bars, cfg, from, to);
  print(`${files[0]}: 봉 ${bars.length}개, ${date} ${opt.from || '00:00'}~${opt.to || '23:59'} 판정 ${rows.length}봉`);
  let hits = 0;
  for (const r of rows) {
    const show = r.res.hit || (opt.all && r.res.required) || (opt.explain && hhmm(r.t) === opt.explain);
    if (r.res.hit) hits++;
    if (!show) continue;
    const s = r.res.snap;
    const tag = r.res.hit ? (r.alert ? '감지(알림)' : r.res.lowerHigh?.blocked ? '감지(고점 낮아짐, 알림 안 함)' : '감지(쿨다운)') : r.res.required ? (r.res.edge && r.res.score >= cfg.bonus.minScore ? '장 경계 제외' : '필수만 충족') : '미충족';
    print(`${hhmm(r.t)} ${tag} 점수 ${r.res.score} 종가 ${s.close} RSI ${s.rsi.toFixed(1)}/${s.rsiSignal.toFixed(1)} MACD ${s.macd.toFixed(1)}/${s.macdSignal.toFixed(1)} DI ${s.diPlus.toFixed(1)}/${s.diMinus.toFixed(1)} ADX ${s.adx.toFixed(1)} 각도 ${s.angle.toFixed(1)}° VWAP ${s.vwapPct.toFixed(2)}%`);
    if (opt.explain || opt.all) for (const l of table(r.res)) print(l);
  }
  print(`감지 ${hits}봉, 알림 ${rows.filter((r) => r.alert).length}회`);
  return 0;
}
