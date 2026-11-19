// Update log
// - [2026-11-06] type=feat (Number / sign formats shared by the HtsH windows; Korean market colours: up = red (gain), down = blue (loss))

export const fmtInt = (n: number | undefined | null) => (n === undefined || n === null || !Number.isFinite(n) ? '-' : Math.round(n).toLocaleString('ko-KR'));
export const fmtNum = (n: number | undefined | null, d = 2) =>
  n === undefined || n === null || !Number.isFinite(n) ? '-' : n.toLocaleString('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d });
export const fmtPct = (n: number | undefined | null, d = 2) => (n === undefined || n === null || !Number.isFinite(n) ? '-' : `${n > 0 ? '+' : ''}${n.toFixed(d)}%`);
/** 1,234,567,890,000 -> 1조 2,346억 style (KRW) */
export function fmtKrwBig(n: number | undefined | null): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return '-';
  const eok = n / 1e8;
  if (Math.abs(eok) >= 10000) return `${(eok / 10000).toFixed(2)}조`;
  if (Math.abs(eok) >= 1) return `${Math.round(eok).toLocaleString('ko-KR')}억`;
  return `${Math.round(n / 1e4).toLocaleString('ko-KR')}만`;
}
/** KIS prdy_vrss_sign: 1 상한, 2 상승, 3 보합, 4 하한, 5 하락 */
export const signOf = (sign: string | undefined, diff?: number): number =>
  sign === '1' || sign === '2' ? 1 : sign === '4' || sign === '5' ? -1 : sign === '3' ? 0 : diff ? Math.sign(diff) : 0;
export const upDown = (s: number) => (s > 0 ? 'up' : s < 0 ? 'down' : 'flat');
export const arrow = (sign: string | undefined, diff?: number) => {
  if (sign === '1') return '↑';
  if (sign === '4') return '↓';
  const s = signOf(sign, diff);
  return s > 0 ? '▲' : s < 0 ? '▼' : '';
};
