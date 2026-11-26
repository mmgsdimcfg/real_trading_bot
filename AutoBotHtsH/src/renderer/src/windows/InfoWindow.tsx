// Update log
// - [2026-11-26] type=feat (Symbol box in the header, the same as every other symbol window (user 2026-11-26))
// - [2026-11-06] type=feat (Stock info window: KIS search_stock_info fields with the official Korean labels (listing date, shares, capital,
//   par value, sector classes, KOSPI200 membership, halt / administrative flags, NXT tradability ...), valuation from the current quote
//   (market cap, PER / PBR / EPS / BPS, foreign ownership, 52-week range position) and investor net buying for the last days)
// - [2026-11-23] type=feat (투자자별 순매수 sits to the right of the stock information (user 2026-11-23); under it when the window is too
//   narrow for both columns)
// - [2026-11-23] type=feat (user 2026-11-23: (1) the window opens tall enough to show the whole stock information - fitted once per
//   window after the data arrived (opts.fit), never taller than the work area; (2) the investor table scrolls inside the window;
//   (3) it lists about one year (investorDaily, the chart pane's cached rows) instead of the last 30 days; (4) a sum row for the last
//   1 / 3 / 6 / 12 months, chosen with the 누적 select (opts.cum, 0 = none))

import { useEffect, useRef } from 'react';
import { useStore, type Win } from '../store';
import { useCall, useSymbol } from '../hooks';
import { NxtMark } from '../components/NxtTag';
import { SymbolSearch } from '../components/SymbolSearch';
import type { Quote } from '../../../shared/kis';
import type { InfoField } from '../../../shared/market';
import { InvestorPanel } from '../components/InvestorPanel';
import { fmtInt, fmtKrwBig, fmtNum } from '../lib';

const fmtVal = (label: string, v: string) => {
  if (/^\d{8}$/.test(v) && /일자|일$/.test(label)) return `${v.slice(0, 4)}.${v.slice(4, 6)}.${v.slice(6, 8)}`;
  if (/^-?\d+(\.\d+)?$/.test(v) && v.length > 3 && !/코드|번호|월일/.test(label)) return Number(v).toLocaleString('ko-KR');
  if (v === 'Y') return '예';
  if (v === 'N') return '아니오';
  return v;
};
const FIT_MIN_W = 900;
const FIT_W = 960;

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
      <h3 className="q-id">
        <SymbolSearch compact value={win.code} onPick={(c) => useStore.getState().setCode(win.id, c)} />
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
