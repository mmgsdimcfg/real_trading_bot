/*
 * Update log
 * - [2026-11-04] type=feat - Symbol board: current state of every watched symbol (last evaluation, stopping condition, depth reached
 *   today, position, realized P&L); sortable, filter by stopping condition / holdings; a click opens the pipeline for that symbol
 * - [2026-11-05] type=feat - "감시목록 편집" button opens the r004 watchlist editor (WatchlistEditor)
 * - [2026-11-11] type=feat - 사유 in Korean (shared/reasonKo; the code and the raw log line stay in the tooltip); fixed column
 *   widths (colgroup, table-layout fixed) so the table never scrolls sideways - 종목명 / 멈춘 조건 / 사유 / 최다 막힘 wrap instead
 */

import { useMemo, useState } from 'react';
import { PriceTag } from '../components/PriceTag';
import { NxtMark } from '../components/NxtTag';
import { useStore } from '../store';
import { condTitle, fmtInt, fmtKrw, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import type { SymbolSummary } from '../../../shared/types';
import { WatchlistEditor } from '../components/WatchlistEditor';
import { reasonKo } from '../../../shared/reasonKo';

type SortKey = 'code' | 'name' | 'price' | 'last' | 'cond' | 'depth' | 'blocks' | 'pnl';

const WATCH_LABEL: Record<string, string> = { active: '감시', backup: '백업', graduated: '졸업', dropped: '탈락', unknown: '-' };
const RESULT_LABEL: Record<string, string> = { pass: '✓ 전부 통과', fail: '✕ 막힘', hold: '… 대기' };

export function BoardView() {
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const selected = useStore((s) => s.selected);
  const quotes = useStore((s) => s.quotes);
  const [sort, setSort] = useState<{ k: SortKey; asc: boolean }>({ k: 'code', asc: true });
  const [condFilter, setCondFilter] = useState('');
  const [holdOnly, setHoldOnly] = useState(false);
  const [wlOpen, setWlOpen] = useState(false);
  const connected = useStore((s) => s.status.conn === 'connected');
  const total = conditions?.buy.length || 0;

  const topBlock = (s: SymbolSummary) => Object.entries(s.blockCounts).sort((a, b) => b[1] - a[1])[0];
  const condsInUse = useMemo(() => [...new Set((snap?.symbols || []).map((s) => s.lastEval?.condKey).filter(Boolean) as string[])].sort(), [snap]);

  const rows = useMemo(() => {
    let list = [...(snap?.symbols || [])];
    if (condFilter) list = list.filter((s) => s.lastEval?.condKey === condFilter);
    if (holdOnly) list = list.filter((s) => s.position);
    const val = (s: SymbolSummary): number | string => {
      switch (sort.k) {
        case 'code': return s.code;
        case 'name': return s.name;
        case 'price': return quotes[s.code]?.pct ?? -1e9;
        case 'last': return s.lastEval?.ts || 0;
        case 'cond': return s.lastEval?.condKey || '';
        case 'depth': return s.furthestOrderToday;
        case 'blocks': return topBlock(s)?.[1] || 0;
        case 'pnl': return s.realizedKrw;
      }
    };
    list.sort((a, b) => {
      const x = val(a);
      const y = val(b);
      const c = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
      return sort.asc ? c : -c;
    });
    return list;
  }, [snap, sort, condFilter, holdOnly, quotes]);

  const th = (k: SortKey, label: string, cls = '') => (
    <th className={cls} onClick={() => setSort((s) => ({ k, asc: s.k === k ? !s.asc : true }))}>
      {label}
      {sort.k === k ? (sort.asc ? ' ▲' : ' ▼') : ''}
    </th>
  );

  return (
    <div className="board">
      <div className="toolbar">
        <label>
          막힌 조건
          <select value={condFilter} onChange={(e) => setCondFilter(e.target.value)}>
            <option value="">전체</option>
            {condsInUse.map((k) => (
              <option key={k} value={k}>
                {condTitle(conditions, 'buy', k)}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={holdOnly} onChange={(e) => setHoldOnly(e.target.checked)} /> 보유 종목만
        </label>
        <span className="muted">
          {rows.length} / {snap?.symbols.length || 0} 종목 · 행을 누르면 파이프라인으로 이동
        </span>
        <button className="push-right" disabled={!connected} onClick={() => setWlOpen(true)} title="r004 감시목록 보기/고치기 (재시작 후 반영)">
          감시목록 편집
        </button>
      </div>
      {wlOpen && <WatchlistEditor onClose={() => setWlOpen(false)} />}
      <div className="table-wrap">
        <table className="grid board-grid">
          <colgroup>
            <col style={{ width: '6%' }} />
            <col style={{ width: '10%' }} />
            <col style={{ width: '13%' }} />
            <col style={{ width: '4%' }} />
            <col style={{ width: '7%' }} />
            <col style={{ width: '6%' }} />
            <col style={{ width: '10%' }} />
            <col style={{ width: '17%' }} />
            <col style={{ width: '6%' }} />
            <col style={{ width: '9%' }} />
            <col style={{ width: '6%' }} />
            <col style={{ width: '6%' }} />
          </colgroup>
          <thead>
            <tr>
              {th('code', '코드')}
              {th('name', '종목명')}
              {th('price', '현재가 (전일 대비)', 'num')}
              <th>감시</th>
              {th('last', '마지막 평가')}
              <th>상태</th>
              {th('cond', '멈춘 조건')}
              <th>사유</th>
              {th('depth', '오늘 최대 도달', 'num')}
              {th('blocks', '오늘 최다 막힘')}
              <th className="num">보유</th>
              {th('pnl', '실현 손익', 'num')}
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => {
              const tb = topBlock(s);
              const ev = s.lastEval;
              return (
                <tr
                  key={s.code}
                  className={s.code === selected ? 'sel' : ''}
                  onClick={() => {
                    select(s.code);
                    setTab('pipeline');
                  }}
                >
                  <td className="mono">{s.code}</td>
                  <td>
                    {s.name} {s.nxt && <NxtMark />}
                  </td>
                  <td className="num">
                    <PriceTag code={s.code} />
                  </td>
                  <td>{WATCH_LABEL[s.watch]}</td>
                  <td className="mono">{fmtKstTime(ev?.ts)}</td>
                  <td className={`res-${ev?.result || ''}`}>{ev ? RESULT_LABEL[ev.result] || ev.result : '-'}</td>
                  <td>{ev && ev.condKey !== 'ORDER' ? condTitle(conditions, 'buy', ev.condKey) : ev ? '주문 단계' : '-'}</td>
                  <td className="reason" title={ev ? `${ev.reason}\n${ev.raw}` : ''}>
                    {ev?.reason ? reasonKo(ev.reason) || ev.reason : '-'}
                  </td>
                  <td className="num">{s.furthestOrderToday >= total ? '전부' : `${s.furthestOrderToday + 1}/${total}`}</td>
                  <td>{tb ? `${condTitle(conditions, 'buy', tb[0])} (${fmtInt(tb[1])})` : '-'}</td>
                  <td className="num">{s.position ? `${s.position.qty}주 @ ${fmtInt(s.position.buyPrice)}` : ''}</td>
                  <td className={`num ${pnlClass(s.realizedKrw)}`}>{s.realizedKrw ? fmtKrw(s.realizedKrw) : ''}</td>
                </tr>
              );
            })}
            {!rows.length && (
              <tr>
                <td colSpan={11} className="muted center">
                  표시할 종목이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
