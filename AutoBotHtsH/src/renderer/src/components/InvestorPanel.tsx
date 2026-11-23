// Update log
// - [2026-11-23] type=feat (Investor net buying table shared by the stock info window and the 투자자 tab of the current price window
//   (user 2026-11-23): about one year of daily rows (investorDaily, cached) with the newest day from the 30-day list, newest first,
//   and a sum row for the last 1 / 3 / 6 / 12 months chosen with the 누적 select (kept per window: opts.cum, 0 = none). `inTab`: the
//   panel sits in a tab body that scrolls by itself (the header row and the sum row stay on top))

import { useMemo } from 'react';
import { useStore, type Win } from '../store';
import { useCall } from '../hooks';
import type { InvestorDay, InvestorRow } from '../../../shared/market';
import { CUM_MONTHS, cumLabel, investorSum, mergeInvestor } from '../../../shared/investor';
import { fmtInt, upDown } from '../lib';

const ymd = (s?: string) => (s && s.length === 8 ? `${s.slice(2, 4)}.${s.slice(4, 6)}.${s.slice(6, 8)}` : s || '');
const sign = (v: number | undefined) => upDown(Math.sign(v || 0));

export function InvestorPanel({ win, inTab = false }: { win: Win; inTab?: boolean }) {
  const setOpts = useStore((s) => s.setOpts);
  const year = useCall<InvestorDay[]>('investorDaily', [win.code], 300000);
  const recent = useCall<InvestorRow[]>('investor', [win.code], 60000);
  const rows = useMemo(() => mergeInvestor(year.data, recent.data), [year.data, recent.data]);
  const cum = CUM_MONTHS.includes(win.opts.cum as never) ? (win.opts.cum as number) : win.opts.cum === 0 ? 0 : 1;
  const sum = useMemo(() => (cum ? investorSum(rows, cum) : null), [rows, cum]);
  const err = !rows.length ? year.error || recent.error : '';
  return (
    <div className={`info-inv ${inTab ? 'in-tab' : ''}`}>
      <div className="inv-head">
        <h4>투자자별 순매수 (주)</h4>
        <label className="inv-cum" title="가장 최근 거래일부터 거슬러 올라간 기간의 합계를 표 맨 위에 표시합니다">
          누적
          <select value={cum} onChange={(e) => setOpts(win.id, { cum: Number(e.target.value) })}>
            <option value={0}>표시 안 함</option>
            {CUM_MONTHS.map((m) => (
              <option key={m} value={m}>
                {cumLabel(m)}
              </option>
            ))}
          </select>
        </label>
        <span className="muted small" title="실전 계좌는 KRX+NXT 통합 기준 (종가도 통합 종가). 약 1년치를 한 번 받아 두고 이후에는 새 날짜만 받습니다">
          {rows.length ? `${rows.length}일 · ${ymd(rows[rows.length - 1].date)} ~` : year.loading ? '불러오는 중…' : ''}
        </span>
      </div>
      {err && <div className="empty small">{err}</div>}
      <div className="tab-scroll">
        <table className="grid inv-table">
          <thead>
            <tr>
              <th>일자</th>
              <th className="num">종가</th>
              <th className="num">개인</th>
              <th className="num">외국인</th>
              <th className="num">기관</th>
              <th className="num" title="백만원">외국인 금액</th>
              <th className="num" title="백만원">기관 금액</th>
            </tr>
            {sum && (
              <tr className="inv-sum" title={`${ymd(sum.from)} ~ ${ymd(sum.to)} (${sum.days}거래일) 합계`}>
                <th>
                  누적 {cumLabel(cum)} <small>{sum.days}일</small>
                </th>
                <th className="num muted">{ymd(sum.from)}~</th>
                <th className={`num ${sign(sum.person)}`}>{fmtInt(sum.person)}</th>
                <th className={`num ${sign(sum.foreign)}`}>{fmtInt(sum.foreign)}</th>
                <th className={`num ${sign(sum.institution)}`}>{fmtInt(sum.institution)}</th>
                <th className={`num ${sign(sum.foreignAmt)}`}>{fmtInt(sum.foreignAmt)}</th>
                <th className={`num ${sign(sum.institutionAmt)}`}>{fmtInt(sum.institutionAmt)}</th>
              </tr>
            )}
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.date}>
                <td className="mono">{ymd(r.date)}</td>
                <td className={`num ${sign(r.change)}`}>{fmtInt(r.close)}</td>
                <td className={`num ${sign(r.person)}`}>{fmtInt(r.person)}</td>
                <td className={`num ${sign(r.foreign)}`}>{fmtInt(r.foreign)}</td>
                <td className={`num ${sign(r.institution)}`}>{fmtInt(r.institution)}</td>
                <td className={`num ${sign(r.foreignAmt)}`}>{fmtInt(r.foreignAmt)}</td>
                <td className={`num ${sign(r.institutionAmt)}`}>{fmtInt(r.institutionAmt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
