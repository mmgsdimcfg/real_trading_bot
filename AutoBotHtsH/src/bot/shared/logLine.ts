/*
 * Update log
 * - [2026-11-04] type=feat - r003 main log line grammar (guide 2.2) and buy_sell summary log grammar (guide 2.3)
 * - [2026-11-05] type=fix - Tags may start with an emoji ("[🔴HOLD SKIP ]"); such lines did not match TAGGED_RE and were dropped as
 *   untagged. Non-ASCII characters before the tag name are now skipped (tag = "HOLD SKIP")
 */

import { kstToEpoch } from './time';

// "2026-11-04 09:08:12,593 [INFO]   [BUY EVAL    ] 159010_아스플로   | OK ..."
const TAGGED_RE =
  /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3}) \[(\w+)\]\s+\[[^\x00-\x7F\]]*([A-Z][A-Z0-9 _-]*?)\s*\] (.*)$/;
// Same prefix without a [TAG] ("R003 START ...", "WATCH | 159010 | ...")
const UNTAGGED_RE = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3}) \[(\w+)\]\s?(.*)$/;
// "159010_아스플로   | ..." / "028260 | ..." / "017900_광전자    (광전자) | ..."
const SYMBOL_RE = /^(\d{6})(?:_(\S*))?[^|]*?\|\s?(.*)$/;

export type MainLine =
  | { kind: 'tagged'; ts: number; level: string; tag: string; body: string; code?: string; name?: string; rest: string }
  | { kind: 'untagged'; ts: number; level: string; body: string }
  | { kind: 'continuation'; body: string }
  | { kind: 'blank' };

export function parseMainLine(raw: string): MainLine {
  if (!raw.trim()) return { kind: 'blank' };
  let m = TAGGED_RE.exec(raw);
  if (m) {
    const ts = kstToEpoch(m[1], m[2], +m[3]);
    const tag = m[5].trim();
    const body = m[6];
    const s = SYMBOL_RE.exec(body);
    if (s) return { kind: 'tagged', ts, level: m[4], tag, body, code: s[1], name: cleanName(s[2]), rest: s[3] };
    return { kind: 'tagged', ts, level: m[4], tag, body, rest: body };
  }
  m = UNTAGGED_RE.exec(raw);
  if (m) return { kind: 'untagged', ts: kstToEpoch(m[1], m[2], +m[3]), level: m[4], body: m[5] };
  return { kind: 'continuation', body: raw };
}

/** Names in log labels may be cut ("카카오게~") - drop the cut marker so it is not shown as the real name. */
function cleanName(n: string | undefined): string | undefined {
  if (!n) return undefined;
  return n.replace(/~+$/, '') || undefined;
}

export function isTruncatedLabel(raw: string): boolean {
  return /^\d{6}_\S*~/.test(raw);
}

/** "29,050" -> 29050, "-10,413원" -> -10413 */
export function num(s: string | undefined): number {
  if (s === undefined) return NaN;
  const n = parseFloat(String(s).replace(/[,원%\s]/g, ''));
  return Number.isFinite(n) ? n : NaN;
}

/** Splits "{reason} | GATES ... | STEPS a(P) b(F)" into segments and parses STEPS. */
export function splitSegments(rest: string): string[] {
  return rest.split('|').map((s) => s.trim());
}

export function parseSteps(rest: string): Record<string, 'P' | 'F' | '-'> | undefined {
  const seg = splitSegments(rest).find((s) => s.startsWith('STEPS'));
  if (!seg) return undefined;
  const out: Record<string, 'P' | 'F' | '-'> = {};
  const re = /([A-Za-z0-9_]+)\(([PF-])\)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(seg))) out[m[1]] = m[2] as 'P' | 'F' | '-';
  return Object.keys(out).length ? out : undefined;
}

// ---- buy_sell summary log ---------------------------------------------------------------

// "2026-11-04 09:09:26 [INFO] >>> 159010_아스플로 Buy - reason=..."
const BS_LINE_RE = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(?:,\d{3})? \[(\w+)\]\s?(.*)$/;
export const BS_BLOCK_RE = /^>>> (\d{6})_(\S*)\s+(Buy|Sell) - (.*)$/;
export const BS_DETAIL_RE = /^\*\*\* (.*)$/;

export function parseBuySellLine(raw: string): { ts: number; body: string } | null {
  const m = BS_LINE_RE.exec(raw);
  if (!m) return null;
  return { ts: kstToEpoch(m[1], m[2]), body: m[4] };
}

/** "매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17" -> { 매수시각: ..., ... } */
export function parseKv(body: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const seg of body.split('|')) {
    const i = seg.indexOf('=');
    if (i > 0) out[seg.slice(0, i).trim()] = seg.slice(i + 1).trim();
  }
  return out;
}
