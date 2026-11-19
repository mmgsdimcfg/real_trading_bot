// Update log
// - [2026-11-06] type=feat (Stock info window: KIS search_stock_info fields with the official Korean labels (listing date, shares, capital,
//   par value, sector classes, KOSPI200 membership, halt / administrative flags, NXT tradability ...), valuation from the current quote
//   (market cap, PER / PBR / EPS / BPS, foreign ownership, 52-week range position) and investor net buying for the last days)

import type { Win } from '../store';
import { useCall, useSymbol } from '../hooks';
import { NxtMark } from '../components/NxtTag';
import type { Quote } from '../../../shared/kis';
import type { InfoField } from '../../../shared/market';
import { fmtInt, fmtKrwBig, fmtNum } from '../lib';
import { InvestorTab } from './QuoteWindow';

const fmtVal = (label: string, v: string) => {
  if (/^\d{8}$/.test(v) && /일자|일$/.test(label)) return `${v.slice(0, 4)}.${v.slice(4, 6)}.${v.slice(6, 8)}`;
  if (/^-?\d+(\.\d+)?$/.test(v) && v.length > 3 && !/코드|번호|월일/.test(label)) return Number(v).toLocaleString('ko-KR');
  if (v === 'Y') return '예';
  if (v === 'N') return '아니오';
  return v;
};

export function InfoWindow({ win }: { win: Win }) {
  const sym = useSymbol(win.code);
  const info = useCall<InfoField[]>('info', [win.code], 0);
  const q = useCall<Quote>('quote', [win.code], 30000);
  const d = q.data;
  const pos = d?.w52High && d.w52Low && d.w52High > d.w52Low ? ((d.price - d.w52Low) / (d.w52High - d.w52Low)) * 100 : undefined;
  return (
    <div className="info-win">
      <h3>
        {sym?.name || win.code} <span className="mono muted">{win.code}</span> {sym?.nxt && <NxtMark />} <span className="muted small">{sym ? `${sym.market}` : ''}</span>
      </h3>
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
      <h4>투자자별 순매수 (주)</h4>
      <div className="tab-scroll">
        <InvestorTab code={win.code} />
      </div>
    </div>
  );
}
