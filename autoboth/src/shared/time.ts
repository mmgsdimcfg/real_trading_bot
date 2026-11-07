/*
 * Update log
 * - [2026-11-04] type=feat - KST time helpers (server logs are KST local time; the app may run in any time zone)
 */

const KST_OFFSET_MS = 9 * 3600 * 1000;

/** "2026-11-04", "09:08:12", ms -> epoch ms (KST wall clock) */
export function kstToEpoch(date: string, time: string, ms = 0): number {
  const y = +date.slice(0, 4);
  const mo = +date.slice(5, 7);
  const d = +date.slice(8, 10);
  const h = +time.slice(0, 2);
  const mi = +time.slice(3, 5);
  const s = +time.slice(6, 8);
  return Date.UTC(y, mo - 1, d, h, mi, s, ms) - KST_OFFSET_MS;
}

/** "2026-11-04 09:07:56" -> epoch ms, or NaN */
export function kstDateTimeToEpoch(s: string): number {
  const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})/.exec(s.trim());
  return m ? kstToEpoch(m[1], m[2]) : NaN;
}

function kstParts(ts: number): Date {
  return new Date(ts + KST_OFFSET_MS);
}

export function fmtKstTime(ts: number | undefined): string {
  if (ts === undefined || !Number.isFinite(ts)) return '-';
  return kstParts(ts).toISOString().slice(11, 19);
}

export function fmtKstDateTime(ts: number | undefined): string {
  if (ts === undefined || !Number.isFinite(ts)) return '-';
  return kstParts(ts).toISOString().slice(0, 19).replace('T', ' ');
}

/** seconds since KST midnight */
export function kstSecondOfDay(ts: number): number {
  const d = kstParts(ts);
  return d.getUTCHours() * 3600 + d.getUTCMinutes() * 60 + d.getUTCSeconds();
}

/** "09:00" -> 32400 */
export function hhmmToSeconds(hhmm: string): number {
  const m = /^(\d{1,2}):(\d{2})/.exec(hhmm || '');
  return m ? +m[1] * 3600 + +m[2] * 60 : 9 * 3600;
}

/** "20261104" -> "2026-11-04" */
export function ymdDash(ymd: string): string {
  return `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}`;
}
