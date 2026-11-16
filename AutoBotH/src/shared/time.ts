/*
 * Update log
 * - [2026-11-04] type=feat - KST time helpers (server logs are KST local time; the app may run in any time zone)
 * - [2026-11-16] type=feat - tradingSessionRange: the day's whole trading session (KRX 09:00-15:30, NXT 08:00-20:00) for chart axes
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

/** [2026-11-16] The day's whole trading session as epoch ms: NXT symbols 08:00 - 20:00, KRX only 09:00 - 15:30 (KST).
 *  `date` is "20261116" or "2026-11-16"; null when it is neither. */
export function tradingSessionRange(date: string | undefined, nxt: boolean): [number, number] | null {
  const d = /^\d{8}$/.test(date || '') ? ymdDash(date as string) : date || '';
  if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) return null;
  return nxt ? [kstToEpoch(d, '08:00:00'), kstToEpoch(d, '20:00:00')] : [kstToEpoch(d, '09:00:00'), kstToEpoch(d, '15:30:00')];
}

/** "20261104" -> "2026-11-04" */
export function ymdDash(ymd: string): string {
  return `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}`;
}
