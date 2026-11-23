// Update log
// - [2026-11-23] type=feat (Stock info window, investor net buying (user 2026-11-23): the year of daily rows (investorDaily) with the
//   newest days of the short list (investor) that the year does not hold yet, newest first (mergeInvestor), and the sums of the last
//   1 / 3 / 6 / 12 months (investorSum - the same window rule as the chart pane 투자자 누적: rows dated after `months` before the
//   newest row))

import { monthsBefore } from './indicators';
import type { InvestorDay, InvestorRow } from './market';

export const CUM_MONTHS = [1, 3, 6, 12] as const;
export const cumLabel = (m: number) => (m === 12 ? '1년' : `${m}달`);

/** rows of the year (oldest first) + the days of `recent` newer than them, newest first */
export function mergeInvestor(year: InvestorDay[] | null | undefined, recent: InvestorRow[] | null | undefined): InvestorDay[] {
  const map = new Map<string, InvestorDay>();
  let newest = '';
  for (const r of year || []) {
    map.set(r.date, r);
    if (r.date > newest) newest = r.date;
  }
  for (const r of recent || []) if (r.date > newest) map.set(r.date, r);
  return [...map.values()].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
}

export interface InvestorSum {
  /** first / last date summed (YYYYMMDD) and the number of days */
  from: string;
  to: string;
  days: number;
  person: number;
  foreign: number;
  institution: number;
  /** million KRW; NaN when no row of the window carries amounts */
  personAmt: number;
  foreignAmt: number;
  institutionAmt: number;
}

/** sums over the rows dated after `months` before the newest row; null without rows */
export function investorSum(rows: InvestorDay[], months: number): InvestorSum | null {
  if (!rows.length) return null;
  const to = rows.reduce((m, r) => (r.date > m ? r.date : m), rows[0].date);
  const start = monthsBefore(to, months);
  const win = rows.filter((r) => r.date > start && Number.isFinite(r.person) && Number.isFinite(r.foreign) && Number.isFinite(r.institution));
  if (!win.length) return null;
  const add = (f: (r: InvestorDay) => number | undefined) => {
    let s = 0;
    let n = 0;
    for (const r of win) {
      const v = f(r);
      if (typeof v === 'number' && Number.isFinite(v)) (s += v), n++;
    }
    return n ? s : NaN;
  };
  return {
    from: win.reduce((m, r) => (r.date < m ? r.date : m), win[0].date),
    to: win.reduce((m, r) => (r.date > m ? r.date : m), win[0].date),
    days: win.length,
    person: add((r) => r.person),
    foreign: add((r) => r.foreign),
    institution: add((r) => r.institution),
    personAmt: add((r) => r.personAmt),
    foreignAmt: add((r) => r.foreignAmt),
    institutionAmt: add((r) => r.institutionAmt),
  };
}
