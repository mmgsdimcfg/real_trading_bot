/*
 * Update log
 * - [2026-11-12] type=feat - 가상 매매 tab: the Edge-AI-only paper trades of the viewed day (server r022, no orders) - whether the
 *   running bot has the mode on, the day summary (closed / open, wins, sum and mean net %), the open virtual positions with the
 *   current price and the unrealized %, and the closed trades (newest first). Re-read with every snapshot while the day is live
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { NxtTag } from '../components/NxtTag';
import { fmtInt, fmtPct, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import { PAPER_EXIT_KO, paperOpenPct, paperSummary, type PaperTrade } from '../../../shared/paper';

export function PaperView() {
  const snap = useStore((s) => s.snapshot);
  const status = useStore((s) => s.status);
  const mode = useStore((s) => s.control.paper);
  const quotes = useStore((s) => s.quotes);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const day = snap?.date || status.date;
  const tick = status.live ? snap?.generatedAt : 0;
  const [res, setRes] = useState<{ day: string; ok: boolean; trades: PaperTrade[]; error?: string } | null>(null);

  useEffect(() => {
    if (!day) return;
    let alive = true;
    void window.api.getPaper(day).then((r) => {
      if (alive) setRes({ day, ...r });
    });
    return () => {
      alive = false;
    };
  }, [day, tick, status.conn]);

  const trades = res && res.day === day ? res.trades : [];
  const sum = useMemo(() => paperSummary(trades), [trades]);
  const open = trades.filter((t) => t.exitTs === undefined);
  const closed = trades.filter((t) => t.exitTs !== undefined).sort((a, b) => (b.exitTs || 0) - (a.exitTs || 0));
  const go = (code: string) => {
    if (snap?.symbols.some((s) => s.code === code)) {
      select(code);
      setTab('timeline');
    }
  };

  const strategy = useStore((s) => s.control.strategy);
  const realEdge = (strategy?.running ?? strategy?.effective) === 'edge';
  const running = mode?.running ?? null;
  const modeText = !mode
    ? '서버가 가상 매매 설정을 모릅니다 (서버 코드 갱신 필요)'
    : running === 'paper'
      ? '실행 중인 봇: 가상 매매 진행 중 (주문 없음)'
      : running === null
        ? `봇이 실행 중이 아닙니다 · 저장된 설정: ${mode.effective === 'paper' ? '켜짐' : '꺼짐'}`
        : '실행 중인 봇: 가상 매매 꺼짐';

  return (
    <div className="paper-view">
      <div className="toolbar">
        <span className={`badge-mode ${running === 'paper' ? 'on' : ''}`}>{modeText}</span>
        {mode?.pending && <span className="badge-warn">⚠ 설정이 바뀌었습니다 → r003 재시작 후 적용</span>}
        {realEdge && <span className="badge-warn">매매 방식이 Edge AI(실제 주문)입니다 - 가상 매매는 돌지 않고, 실제 내역은 매매 결과 탭에 있습니다</span>}
        <span className="muted">Edge AI 확률만으로 판단한 가상 매수/매도 기록입니다. 실제 주문·계좌와 무관합니다. 켜고 끄기는 제어 탭.</span>
      </div>
      <div className="kpis">
        <div className="kpi">
          <span>청산 / 보유</span>
          <b>
            {sum.closed} / {sum.open}
          </b>
        </div>
        <div className="kpi">
          <span>승률</span>
          <b>{sum.closed ? `${(sum.winRate * 100).toFixed(0)}% (${sum.wins}/${sum.closed})` : '-'}</b>
        </div>
        <div className="kpi">
          <span>순손익 합계</span>
          <b className={pnlClass(sum.closed ? sum.sumPct : undefined)}>{sum.closed ? fmtPct(sum.sumPct) : '-'}</b>
        </div>
        <div className="kpi">
          <span>거래당 평균</span>
          <b className={pnlClass(sum.closed ? sum.meanPct : undefined)}>{sum.closed ? fmtPct(sum.meanPct) : '-'}</b>
        </div>
        <div className="kpi">
          <span>청산 사유</span>
          <b className="small">
            {Object.entries(sum.byReason)
              .map(([k, n]) => `${PAPER_EXIT_KO[k] || k} ${n}`)
              .join(' · ') || '-'}
          </b>
        </div>
      </div>
      {res && !res.ok && <p className="err">{res.error}</p>}

      <h4>가상 보유 ({open.length})</h4>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>매수 시각</th>
              <th>종목</th>
              <th className="num">매수가</th>
              <th className="num">확률</th>
              <th className="num">현재가</th>
              <th className="num">평가 수익률</th>
              <th className="num">목표가</th>
              <th className="num">손절가</th>
              <th>트리거</th>
            </tr>
          </thead>
          <tbody>
            {open.map((t) => {
              const price = quotes[t.code]?.price;
              const pct = paperOpenPct(t, price);
              return (
                <tr key={`${t.code}-${t.entryTs}`} onClick={() => go(t.code)}>
                  <td className="mono">{fmtKstTime(t.entryTs)}</td>
                  <td>
                    {t.code} {t.name} <NxtTag code={t.code} />
                  </td>
                  <td className="num">{fmtInt(t.entry)}</td>
                  <td className="num">{t.prob !== undefined ? t.prob.toFixed(3) : '-'}</td>
                  <td className="num">{fmtInt(price)}</td>
                  <td className={`num ${pnlClass(pct)}`}>{fmtPct(pct)}</td>
                  <td className="num">{fmtInt(t.tpPx)}</td>
                  <td className="num">{fmtInt(t.slPx)}</td>
                  <td className="reason">{t.trigger}</td>
                </tr>
              );
            })}
            {!open.length && (
              <tr>
                <td colSpan={9} className="muted center">
                  가상 보유가 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <h4>청산 내역 ({closed.length})</h4>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>매수 시각</th>
              <th>매도 시각</th>
              <th>종목</th>
              <th className="num">매수가</th>
              <th className="num">매도가</th>
              <th>청산 사유</th>
              <th className="num">보유(분)</th>
              <th className="num">확률</th>
              <th className="num">순수익률</th>
            </tr>
          </thead>
          <tbody>
            {closed.map((t) => (
              <tr key={`${t.code}-${t.entryTs}`} onClick={() => go(t.code)}>
                <td className="mono">{fmtKstTime(t.entryTs)}</td>
                <td className="mono">{fmtKstTime(t.exitTs)}</td>
                <td>
                  {t.code} {t.name} <NxtTag code={t.code} />
                </td>
                <td className="num">{fmtInt(t.entry)}</td>
                <td className="num">{fmtInt(t.exit)}</td>
                <td>{PAPER_EXIT_KO[t.reason || ''] || t.reason}</td>
                <td className="num">{t.heldMin !== undefined ? Math.round(t.heldMin) : '-'}</td>
                <td className="num">{t.prob !== undefined ? t.prob.toFixed(3) : '-'}</td>
                <td className={`num ${pnlClass(t.netPct)}`}>{fmtPct(t.netPct)}</td>
              </tr>
            ))}
            {!closed.length && (
              <tr>
                <td colSpan={9} className="muted center">
                  {trades.length ? '아직 청산된 가상 거래가 없습니다.' : '이 날짜의 가상 매매 기록이 없습니다 (가상 매매가 꺼져 있었거나 매수 신호가 없었음).'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
