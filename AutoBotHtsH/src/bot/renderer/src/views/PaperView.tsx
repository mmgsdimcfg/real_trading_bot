/*
 * Update log
 * - [2026-11-12] type=feat - 가상 매매 tab: the Edge-AI-only paper trades of the viewed day (server r022, no orders) - whether the
 *   running bot has the mode on, the day summary (closed / open, wins, sum and mean net %), the open virtual positions with the
 *   current price and the unrealized %, and the closed trades (newest first). Re-read with every snapshot while the day is live
 * - [2026-11-18] type=perf - The day's last answer is shown at once when the tab is opened again (dataCache), then refreshed
 * - [2026-11-19] type=feat - Tick collector symbols are paper-traded too (server r022 group "collector", separate limits): a
 *   전체 / 감시 종목 / 수집기 종목 switch filters the summary and both tables, collector trades carry a "수집기" tag. Their current
 *   price is not in the bot's quote log (the bot does not poll them) - shown as "-"
 * - [2026-11-19] type=feat - 가상 매매 대상 (user 2026-11-19): every symbol the paper runner looks at, from the server's
 *   edge_only_status_<date>.json (r022, every 5 s) - group, current price, last 1-minute bar, the outcome of its last evaluation
 *   and the probability against the threshold; per group the open / entry counts against their limits. Follows the group switch.
 *   The status price also fills the current price of open collector positions (was "-")
 * - [2026-11-20] type=feat - 가상 매매 대상 table: 매수가 and 수익률 next to 현재가 (user 2026-11-20) - a virtual holding shows its buy
 *   price and the return at the current price; a symbol sold today shows its last trade's buy price and net return (청산); none = -
 * - [2026-11-20] type=feat - 수익금 (KRW) next to the returns (user 2026-11-20): virtual trades have no quantity, so the amount is what the
 *   real bot would have made - floor(MAX_ORDER_AMOUNT_KRW / buy price) shares (the bot's per-order budget, read once from the connected
 *   host's settings, 500,000 when it cannot be read): 대상 / 가상 보유 = at the current price (before costs), 청산 = net of costs;
 *   순손익 합계 shows the KRW sum too. 마지막 판단 says only "가상 보유 중" (the buy price has its own column). The view scrolls
 *   (it was cut off when the list grew)
 * - [2026-11-20] type=feat - 수량 and 총매수액 next to 매수가 in all three tables (the same sizing: shares x buy price)
 * - [2026-11-20] type=feat - Double click on a row shows the symbol in the other windows (showCode: AutoBotHtsH chart / current price /
 *   stock info / news, AutoBotH the 차트 tab); a single click still opens the timeline
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { NxtTag } from '../components/NxtTag';
import { fmtInt, fmtKrw, fmtPct, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import {
  PAPER_EXIT_KO,
  PAPER_GROUP_KO,
  PAPER_STAGE_KO,
  paperOpenPct,
  paperSummary,
  sortStatusRows,
  type PaperGroup,
  type PaperStatus,
  type PaperTrade,
} from '../../../shared/paper';
import { peek, put } from '../dataCache';
import { showCode } from '../showCode';

/** [2026-11-20] the bot's per-order budget (r001 MAX_ORDER_AMOUNT_KRW) - virtual trades are sized like real ones for the KRW columns */
export const ORDER_KRW_DEFAULT = 500_000;
let orderKrwCache: number | null = null;
/** shares the bot would buy at `price` with `budget` (0 = it would not buy) */
export const paperQty = (price: number | undefined, budget: number) => (price && price > 0 ? Math.floor(budget / price) : 0);
/** KRW result of a virtual trade: open = (price - buy) x qty (before costs), closed = net % of the bought amount */
export function paperKrw(entry: number | undefined, budget: number, opts: { price?: number; netPct?: number }): number | undefined {
  const q = paperQty(entry, budget);
  if (!q || !entry) return undefined;
  if (opts.netPct !== undefined) return (q * entry * opts.netPct) / 100;
  if (opts.price && opts.price > 0) return q * (opts.price - entry);
  return undefined;
}

/** [2026-11-20] 수량 / 총매수액 cells of a virtual buy (shares the bot's budget buys at `entry`) */
function BuySize({ entry, budget }: { entry: number | undefined; budget: number }) {
  const q = paperQty(entry, budget);
  return (
    <>
      <td className="num">{q ? `${fmtInt(q)}주` : '-'}</td>
      <td className="num">{q && entry ? fmtInt(q * entry) : '-'}</td>
    </>
  );
}

const DBL_TIP = '누르면 타임라인 · 두 번 누르면 차트 등 다른 창에 이 종목 표시';

type PaperRes = { day: string; ok: boolean; trades: PaperTrade[]; error?: string };
type StatusRes = { day: string; ok: boolean; status: PaperStatus | null; error?: string };
/** a status file older than this (while the day is live) is called stale */
const STATUS_STALE_MS = 60000;

export function PaperView() {
  const snap = useStore((s) => s.snapshot);
  const status = useStore((s) => s.status);
  const mode = useStore((s) => s.control.paper);
  const quotes = useStore((s) => s.quotes);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const day = snap?.date || status.date;
  const tick = status.live ? snap?.generatedAt : 0;
  const [res, setRes] = useState<PaperRes | null>(() => (day ? peek<PaperRes>(`paper:${day}`) || null : null));

  useEffect(() => {
    if (!day) return;
    let alive = true;
    const kept = peek<PaperRes>(`paper:${day}`);
    if (kept) setRes(kept);
    void window.api.getPaper(day).then((r) => {
      if (r.ok) put(`paper:${day}`, { day, ...r });
      if (alive && (r.ok || !kept)) setRes({ day, ...r });
    });
    return () => {
      alive = false;
    };
  }, [day, tick, status.conn]);

  // [2026-11-19] status of the paper-traded symbols (server r022 edge_only_status_<date>.json)
  const [stRes, setStRes] = useState<StatusRes | null>(() => (day ? peek<StatusRes>(`paperStatus:${day}`) || null : null));
  useEffect(() => {
    if (!day) return;
    let alive = true;
    const kept = peek<StatusRes>(`paperStatus:${day}`);
    if (kept) setStRes(kept);
    void window.api.getPaperStatus(day).then((r) => {
      if (r.ok) put(`paperStatus:${day}`, { day, ...r });
      if (alive && (r.ok || !kept)) setStRes({ day, ...r });
    });
    return () => {
      alive = false;
    };
  }, [day, tick, status.conn]);
  const pst = stRes && stRes.day === day ? stRes.status : null;
  const statusPrice = useMemo(() => new Map((pst?.symbols || []).map((x) => [x.code, x.price])), [pst]);

  const [grp, setGrp] = useState<'all' | PaperGroup>('all');
  const allTrades = res && res.day === day ? res.trades : [];
  const trades = useMemo(() => (grp === 'all' ? allTrades : allTrades.filter((t) => t.group === grp)), [allTrades, grp]);
  const groupCount = (g: PaperGroup) => allTrades.filter((t) => t.group === g).length;
  const stRows = useMemo(() => sortStatusRows((pst?.symbols || []).filter((x) => grp === 'all' || x.group === grp)), [pst, grp]);
  /** [2026-11-20] per code: the open virtual buy, else the last closed one of the day */
  const tradeOf = useMemo(() => {
    const m = new Map<string, { open?: PaperTrade; last?: PaperTrade }>();
    for (const t of allTrades) {
      const e = m.get(t.code) || {};
      if (t.exitTs === undefined) e.open = t;
      else if (!e.last || (t.exitTs || 0) > (e.last.exitTs || 0)) e.last = t;
      m.set(t.code, e);
    }
    return m;
  }, [allTrades]);
  const stale = !!pst && !!status.live && Date.now() - pst.asof > STATUS_STALE_MS;
  const sum = useMemo(() => paperSummary(trades), [trades]);
  // [2026-11-20] per-order budget from the host's settings (once per session)
  const connected = status.conn === 'connected' || status.conn === 'local';
  const [orderKrw, setOrderKrw] = useState<number>(orderKrwCache ?? ORDER_KRW_DEFAULT);
  useEffect(() => {
    if (orderKrwCache !== null || !connected) return;
    let alive = true;
    void window.api.ctlSchema(true).then((r) => {
      const k = r.ok ? r.data?.keys.find((x) => x.key === 'MAX_ORDER_AMOUNT_KRW') : undefined;
      const v = Number(k?.effective ?? k?.base);
      if (Number.isFinite(v) && v > 0) {
        orderKrwCache = v;
        if (alive) setOrderKrw(v);
      }
    });
    return () => {
      alive = false;
    };
  }, [connected]);
  const sumKrw = useMemo(
    () => trades.filter((t) => t.exitTs !== undefined).reduce((a, t) => a + (paperKrw(t.entry, orderKrw, { netPct: t.netPct ?? 0 }) || 0), 0),
    [trades, orderKrw],
  );
  const krwTip = `수익금 = ${fmtInt(orderKrw)}원(1회 매수 상한 MAX_ORDER_AMOUNT_KRW)으로 살 수 있는 주수 기준`;
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
      <div className="toolbar">
        <div className="seg">
          {(['all', 'watch', 'collector'] as const).map((g) => (
            <button
              key={g}
              className={grp === g ? 'on' : ''}
              onClick={() => setGrp(g)}
              title={g === 'collector' ? '틱 수집기가 모으는 감시목록 밖 종목 - 틱 파일 1분봉으로 판단, 상한은 감시 종목과 별도' : undefined}
            >
              {g === 'all' ? `전체 (${allTrades.length})` : `${PAPER_GROUP_KO[g]} (${groupCount(g)})`}
            </button>
          ))}
        </div>
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
          <b className={pnlClass(sum.closed ? sum.sumPct : undefined)} title={krwTip}>
            {sum.closed ? `${fmtPct(sum.sumPct)} · ${fmtKrw(sumKrw)}` : '-'}
          </b>
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

      <h4>
        가상 매매 대상 ({stRows.length})
        {pst && <span className="muted small"> · {fmtKstTime(pst.asof)} 기준{stale ? ' (오래됨 - 봇이 멈췄거나 정규장 밖)' : ''}</span>}
      </h4>
      {pst ? (
        <div className="toolbar small">
          {(['watch', 'collector'] as const)
            .filter((g) => grp === 'all' || grp === g)
            .map((g) => {
              const gs = pst.groups[g];
              return (
                <span key={g} className="muted">
                  {PAPER_GROUP_KO[g]}:{' '}
                  {gs.enabled ? `보유 ${gs.open}/${gs.maxPositions} · 오늘 매수 ${gs.entries}/${gs.maxEntries}` : '꺼짐'}
                </span>
              );
            })}
          <span className="muted">
            확률 기준 {pst.threshold !== undefined ? pst.threshold.toFixed(3) : '-'}
            {pst.label?.tp !== undefined ? ` · 목표 +${pst.label.tp}% / 손절 -${pst.label.sl}% / ${pst.label.horizon}분` : ''}
            {pst.entryWindow ? ` · 매수 ${pst.entryWindow[0]}~${pst.entryWindow[1]}` : ''}
            {pst.modelId ? ` · 모델 ${pst.modelId}` : ''}
          </span>
        </div>
      ) : null}
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>종목</th>
              <th>구분</th>
              <th className="num">현재가</th>
              <th className="num" title="가상 보유 중이면 그 매수가, 오늘 청산했으면 마지막 거래의 매수가">매수가</th>
              <th className="num" title={krwTip}>수량</th>
              <th className="num" title={krwTip}>총매수액</th>
              <th className="num" title="가상 보유 중 = 현재가 기준 평가 수익률, 청산 = 마지막 거래의 순수익률 (수수료·세금 뺌)">수익률</th>
              <th className="num" title={krwTip}>수익금</th>
              <th className="num">마지막 봉</th>
              <th className="num">봉 수</th>
              <th>마지막 판단</th>
              <th className="num">확률</th>
              <th className="num">확인 시각</th>
            </tr>
          </thead>
          <tbody>
            {stRows.map((x) => {
              const tr = tradeOf.get(x.code);
              const price = quotes[x.code]?.price ?? x.price;
              const buy = x.position?.entry ?? tr?.open?.entry ?? tr?.last?.entry;
              const held = !!(x.position || tr?.open);
              const pct = held ? (buy && price ? (price / buy - 1) * 100 : undefined) : tr?.last?.netPct;
              return (
              <tr key={x.code} onClick={() => go(x.code)} onDoubleClick={() => showCode(x.code)} title={DBL_TIP} className={x.position ? 'row-held' : undefined}>
                <td>
                  {x.code} {x.name} <NxtTag code={x.code} />
                </td>
                <td>{PAPER_GROUP_KO[x.group]}</td>
                <td className="num">{fmtInt(x.price)}</td>
                <td className="num">{buy ? fmtInt(buy) : '-'}</td>
                <BuySize entry={buy} budget={orderKrw} />
                <td className={`num ${pnlClass(pct)}`} title={held ? '가상 보유 - 현재가 기준 평가' : tr?.last ? '오늘 청산한 마지막 거래의 순수익률' : ''}>
                  {pct !== undefined ? `${fmtPct(pct)}${held ? '' : ' 청산'}` : '-'}
                </td>
                <td className={`num ${pnlClass(pct)}`} title={krwTip}>
                  {fmtKrw(held ? paperKrw(buy, orderKrw, { price }) : tr?.last ? paperKrw(buy, orderKrw, { netPct: tr.last.netPct ?? 0 }) : undefined)}
                </td>
                <td className="num mono">{x.bar || '-'}</td>
                <td className="num">{x.bars || '-'}</td>
                <td>{x.position ? '가상 보유 중' : PAPER_STAGE_KO[x.stage || ''] || x.stage || '-'}</td>
                <td className={`num ${x.prob !== undefined && pst?.threshold !== undefined && x.prob >= pst.threshold ? 'prob-ok' : ''}`}>
                  {x.prob !== undefined ? x.prob.toFixed(3) : '-'}
                </td>
                <td className="num mono">{x.at ? fmtKstTime(x.at) : '-'}</td>
              </tr>
              );
            })}
            {!stRows.length && (
              <tr>
                <td colSpan={13} className="muted center">
                  {stRes && !stRes.ok
                    ? stRes.error
                    : pst
                      ? '이 구분의 가상 매매 대상이 없습니다.'
                      : '가상 매매 상태 파일이 없습니다 (가상 매매가 꺼져 있거나, 봇이 정규장에 돌지 않았거나, 서버 코드 갱신 전).'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <h4>가상 보유 ({open.length})</h4>
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>매수 시각</th>
              <th>종목</th>
              <th className="num">매수가</th>
              <th className="num" title={krwTip}>수량</th>
              <th className="num" title={krwTip}>총매수액</th>
              <th className="num">확률</th>
              <th className="num">현재가</th>
              <th className="num">평가 수익률</th>
              <th className="num" title={krwTip}>평가 손익</th>
              <th className="num">목표가</th>
              <th className="num">손절가</th>
              <th>트리거</th>
            </tr>
          </thead>
          <tbody>
            {open.map((t) => {
              const price = quotes[t.code]?.price ?? statusPrice.get(t.code);
              const pct = paperOpenPct(t, price);
              return (
                <tr key={`${t.code}-${t.entryTs}`} onClick={() => go(t.code)} onDoubleClick={() => showCode(t.code)} title={DBL_TIP}>
                  <td className="mono">{fmtKstTime(t.entryTs)}</td>
                  <td>
                    {t.code} {t.name} <NxtTag code={t.code} />
                    {t.group === 'collector' && <span className="code-tag">수집기</span>}
                  </td>
                  <td className="num">{fmtInt(t.entry)}</td>
                  <BuySize entry={t.entry} budget={orderKrw} />
                  <td className="num">{t.prob !== undefined ? t.prob.toFixed(3) : '-'}</td>
                  <td className="num">{fmtInt(price)}</td>
                  <td className={`num ${pnlClass(pct)}`}>{fmtPct(pct)}</td>
                  <td className={`num ${pnlClass(pct)}`}>{fmtKrw(paperKrw(t.entry, orderKrw, { price }))}</td>
                  <td className="num">{fmtInt(t.tpPx)}</td>
                  <td className="num">{fmtInt(t.slPx)}</td>
                  <td className="reason">{t.trigger}</td>
                </tr>
              );
            })}
            {!open.length && (
              <tr>
                <td colSpan={12} className="muted center">
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
              <th className="num" title={krwTip}>수량</th>
              <th className="num" title={krwTip}>총매수액</th>
              <th className="num">매도가</th>
              <th>청산 사유</th>
              <th className="num">보유(분)</th>
              <th className="num">확률</th>
              <th className="num">순수익률</th>
              <th className="num" title={krwTip}>순손익</th>
            </tr>
          </thead>
          <tbody>
            {closed.map((t) => (
              <tr key={`${t.code}-${t.entryTs}`} onClick={() => go(t.code)} onDoubleClick={() => showCode(t.code)} title={DBL_TIP}>
                <td className="mono">{fmtKstTime(t.entryTs)}</td>
                <td className="mono">{fmtKstTime(t.exitTs)}</td>
                <td>
                  {t.code} {t.name} <NxtTag code={t.code} />
                  {t.group === 'collector' && <span className="code-tag">수집기</span>}
                </td>
                <td className="num">{fmtInt(t.entry)}</td>
                <BuySize entry={t.entry} budget={orderKrw} />
                <td className="num">{fmtInt(t.exit)}</td>
                <td>{PAPER_EXIT_KO[t.reason || ''] || t.reason}</td>
                <td className="num">{t.heldMin !== undefined ? Math.round(t.heldMin) : '-'}</td>
                <td className="num">{t.prob !== undefined ? t.prob.toFixed(3) : '-'}</td>
                <td className={`num ${pnlClass(t.netPct)}`}>{fmtPct(t.netPct)}</td>
                <td className={`num ${pnlClass(t.netPct)}`}>{fmtKrw(paperKrw(t.entry, orderKrw, { netPct: t.netPct ?? 0 }))}</td>
              </tr>
            ))}
            {!closed.length && (
              <tr>
                <td colSpan={12} className="muted center">
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
