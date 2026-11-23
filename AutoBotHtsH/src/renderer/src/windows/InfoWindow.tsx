// Update log
// - [2026-11-06] type=feat (Stock info window: KIS search_stock_info fields with the official Korean labels (listing date, shares, capital,
//   par value, sector classes, KOSPI200 membership, halt / administrative flags, NXT tradability ...), valuation from the current quote
//   (market cap, PER / PBR / EPS / BPS, foreign ownership, 52-week range position) and investor net buying for the last days)
// - [2026-11-23] type=feat (투자자별 순매수 sits to the right of the stock information (user 2026-11-23); under it when the window is too
//   narrow for both columns)
// - [2026-11-23] type=feat (user 2026-11-23: (1) the window opens tall enough to show the whole stock information - fitted once per
//   window after the data arrived (opts.fit), never taller than the work area; (2) the investor table scrolls inside the window;
//   (3) it lists about one year (investorDaily, the chart pane's cached rows) instead of the last 30 days; (4) a sum row for the last
//   1 / 3 / 6 / 12 months, chosen with the 누적 select (opts.cum, 0 = none))

import { useEffect, useMemo, useRef } from 'react';
import { useStore, type Win } from '../store';
import { useCall, useSymbol } from '../hooks';
import { NxtMark } from '../components/NxtTag';
import type { Quote } from '../../../shared/kis';
import type { InfoField, InvestorDay, InvestorRow } from '../../../shared/market';
import { CUM_MONTHS, cumLabel, investorSum, mergeInvestor } from '../../../shared/investor';
import { fmtInt, fmtKrwBig, fmtNum, upDown } from '../lib';

const fmtVal = (label: string, v: string) => {
  if (/^\d{8}$/.test(v) && /일자|일$/.test(label)) return `${v.slice(0, 4)}.${v.slice(4, 6)}.${v.slice(6, 8)}`;
  if (/^-?\d+(\.\d+)?$/.test(v) && v.length > 3 && !/코드|번호|월일/.test(label)) return Number(v).toLocaleString('ko-KR');
  if (v === 'Y') return '예';
  if (v === 'N') return '아니오';
  return v;
};
const ymd = (s?: string) => (s && s.length === 8 ? `${s.slice(2, 4)}.${s.slice(4, 6)}.${s.slice(6, 8)}` : s || '');
const sign = (v: number | undefined) => upDown(Math.sign(v || 0));
const FIT_MIN_W = 900;
const FIT_W = 960;

function InvestorPanel({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const year = useCall<InvestorDay[]>('investorDaily', [win.code], 300000);
  const recent = useCall<InvestorRow[]>('investor', [win.code], 60000);
  const rows = useMemo(() => mergeInvestor(year.data, recent.data), [year.data, recent.data]);
  const cum = CUM_MONTHS.includes(win.opts.cum as never) ? (win.opts.cum as number) : win.opts.cum === 0 ? 0 : 1;
  const sum = useMemo(() => (cum ? investorSum(rows, cum) : null), [rows, cum]);
  const err = !rows.length ? year.error || recent.error : '';
  return (
    <div className="info-inv">
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

export function InfoWindow({ win }: { win: Win }) {
  const sym = useSymbol(win.code);
  const info = useCall<InfoField[]>('info', [win.code], 0);
  const q = useCall<Quote>('quote', [win.code], 30000);
  const d = q.data;
  const pos = d?.w52High && d.w52Low && d.w52High > d.w52Low ? ((d.price - d.w52Low) / (d.w52High - d.w52Low)) * 100 : undefined;

  // fit once: wide enough for both columns, then tall enough for the whole stock information (the left column must not scroll)
  const main = useRef<HTMLDivElement | null>(null);
  const ready = !!d && !!info.data?.length;
  useEffect(() => {
    if (win.opts.fit || win.max || win.min || !ready) return;
    const st = useStore.getState();
    if (win.w < FIT_MIN_W && st.area.w >= FIT_MIN_W) {
      st.resize(win.id, Math.min(FIT_W, st.area.w), win.h); // the height is fitted on the next pass, in the two-column layout
      return;
    }
    const el = main.current;
    if (!el) return;
    const extra = el.scrollHeight - el.clientHeight;
    if (extra > 0) st.resize(win.id, win.w, Math.min(win.h + extra + 4, Math.max(win.h, st.area.h - win.y)));
    st.setOpts(win.id, { fit: true });
  }, [ready, win.w, win.h, win.max, win.min, win.opts.fit]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="info-win">
      <h3>
        {sym?.name || win.code} <span className="mono muted">{win.code}</span> {sym?.nxt && <NxtMark />} <span className="muted small">{sym ? `${sym.market}` : ''}</span>
      </h3>
      <div className="info-cols">
        <div className="info-main" ref={main}>
          {d && (
            <div className="kv-grid">
              <span>시가총액</span>
              <b>{d.marketCapEok !== undefined ? fmtKrwBig(d.marketCapEok * 1e8) : '-'}</b>
              <span>PER / PBR</span>
              <b>
                {fmtNum(d.per)} / {fmtNum(d.pbr)}
              </b>
              <span>EPS / BPS</span>
              <b>
                {fmtInt(d.eps)} / {fmtInt(d.bps)}
              </b>
              <span>외국인 소진율</span>
              <b>{fmtNum(d.foreignRate)}%</b>
              <span>52주 범위</span>
              <b>
                {fmtInt(d.w52Low)} ~ {fmtInt(d.w52High)}
                {pos !== undefined && (
                  <span className="range-bar" title={`52주 범위 중 현재 위치 ${pos.toFixed(0)}%`}>
                    <i style={{ left: `${Math.min(100, Math.max(0, pos))}%` }} />
                  </span>
                )}
              </b>
              <span>업종</span>
              <b>{d.sector || '-'}</b>
            </div>
          )}
          <h4>기본 정보</h4>
          {info.error && <p className="err">{info.error}</p>}
          <div className="kv-grid">
            {(info.data || []).map((f) => (
              <span key={f.label} style={{ display: 'contents' }}>
                <span>{f.label}</span>
                <b>{fmtVal(f.label, f.value)}</b>
              </span>
            ))}
          </div>
        </div>
        <InvestorPanel win={win} />
      </div>
    </div>
  );
}
