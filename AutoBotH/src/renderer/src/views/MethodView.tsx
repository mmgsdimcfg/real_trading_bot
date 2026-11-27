/*
 * Update log
 * - [2026-11-27] type=feat - "매매 방식 현황" (user 2026-11-27: which symbols does the VWAP trade look at; one place per trading method
 *   for its symbols, their state and the results; plan reviewed with Codex). One view, a method selector (규칙 / Edge AI / VWAP):
 *   the mode the running bot uses (and "재시작 후 적용" when the stored one differs), today's numbers, the bot's "신규 매수 중단"
 *   note, the symbol table (감시 / 수집기 / 보유 badges, the last record of the bot for that method with its source and time,
 *   signals / orders, holding, results) and the method's recent fills. A row opens the timeline on that method's tab. Display
 *   only: switches and settings stay in 제어 (buttons lead there), rows come from shared/methodBoard
 * - [2026-11-27] type=feat - On / off on this page (user 2026-11-27): the method's switch and its "실제 매매" switch, through the
 *   same handlers as the control tab (components/useMethodSwitch: same warnings, confirmations and server checks; stored through
 *   r016, applied by an r003 restart - the restart itself stays in 제어)
 */

import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../store';
import { ColGrip, ColGroup, ColReset, colTable, useColWidths } from '../components/useColWidths';
import { PriceTag } from '../components/PriceTag';
import { NxtMark } from '../components/NxtTag';
import { condTitle, fmtInt, fmtKrw, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import { peek, put } from '../dataCache';
import { showCode } from '../showCode';
import { useMethodSwitch } from '../components/useMethodSwitch';
import { reasonKo } from '../../../shared/reasonKo';
import { TRADE_STRATEGIES, TRADE_STRATEGY_KO, strategyModeKo, strategyModes, strategyOfTrade, sideExitKo, type TradeStrategy } from '../../../shared/strategy';
import { blockReasonKo, type SideStrategy } from '../../../shared/strategyEvents';
import type { StrategyEvalRowX } from '../../../shared/sideTimeline';
import { edgeLiveSummaryText, parseEdgeLiveSummary, parseVwapLiveSummary, type EdgeModelInfo, type PaperStatus, type PaperTrade } from '../../../shared/paper';
import { METHOD_FILTER_KO, METHOD_SRC_KO, filterMethodRows, methodRows, methodTotals, type MethodRow, type MethodRowFilter } from '../../../shared/methodBoard';

const REFRESH_MS = 15000;
const BLOCK_NOTE_FRESH_MS = 11 * 60 * 1000;
const FILTERS: MethodRowFilter[] = ['all', 'target', 'recorded', 'active', 'holding'];
const TARGET_NOTE: Record<TradeStrategy, string> = {
  rule: '규칙 매매는 r003 이 감시하는 종목(r004 감시목록)을 조건 순서대로 평가합니다.',
  edge: 'Edge AI 매매는 감시 종목을 평가합니다. 가상 매매는 틱 수집기 종목도 함께 봅니다 (실제 주문은 감시 종목만).',
  vwap: 'VWAP 매매는 규칙 매매와 같은 감시 종목(r004 감시목록)을 1분봉마다 평가합니다. ETF · ETN 제외 등은 봇이 적용합니다.',
};
type EvalRes = { key: string; last: Record<string, StrategyEvalRowX>; count: Record<string, number>; total: number };
type PaperRes = { day: string; ok: boolean; trades: PaperTrade[] };
type StatusRes = { day: string; ok: boolean; status: PaperStatus | null };

const sp = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`;

export function MethodView() {
  const method = useStore((s) => s.methodTab);
  const setMethod = useStore((s) => s.setMethodTab);
  const snap = useStore((s) => s.snapshot);
  const status = useStore((s) => s.status);
  const control = useStore((s) => s.control);
  const conditions = useStore((s) => s.conditions);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const setTimelineMethod = useStore((s) => s.setTimelineMethod);
  const openResults = useStore((s) => s.openResults);
  const selected = useStore((s) => s.selected);
  const cols = useColWidths('autoboth.cols.methodCols', 10);
  const fillCols = useColWidths('autoboth.cols.methodFillCols', 7);
  const [filter, setFilter] = useState<MethodRowFilter>('all');
  // [2026-11-27] on / off here: the control tab's own handlers (same confirmations); the Edge AI model is read for the
  // "real orders only from a promoted model" check, as the control tab does
  const connected = status.conn === 'connected';
  const [swBusy, setSwBusy] = useState(false);
  const [edgeModel, setEdgeModel] = useState<EdgeModelInfo | null | undefined>(() => peek<EdgeModelInfo | null>('control:edge'));
  useEffect(() => {
    if (!connected) return setEdgeModel(undefined);
    if (method !== 'edge') return;
    let alive = true;
    void window.api.getEdgeModel().then((r) => {
      if (r.ok) put('control:edge', r.model);
      if (alive) setEdgeModel(r.ok ? r.model : peek<EdgeModelInfo | null>('control:edge'));
    });
    return () => {
      alive = false;
    };
  }, [connected, method, status.mode]);
  const sw = useMethodSwitch({ setBusy: setSwBusy, edgeModel });
  const swOff = !connected || swBusy;

  const modes = useMemo(() => strategyModes(control), [control]);
  const mode = modes[method];
  const side: SideStrategy | null = method === 'rule' ? null : method;
  const day = snap?.date || status.date;
  const replayX = snap?.replay?.cursor ?? null;
  // live day: re-read the bot's files at most every REFRESH_MS (the snapshot itself is pushed)
  const tick = status.live && snap ? Math.floor(snap.generatedAt / REFRESH_MS) : 0;
  // the host is part of the key: the same day / method on the other host is other data [Codex round 2]
  const key = side && day ? `${status.mode}/${status.conn}/${day}/${side}/${replayX ?? ''}` : '';

  const [evalRes, setEvalRes] = useState<EvalRes | null>(null);
  const [paperRes, setPaperRes] = useState<(PaperRes & { key: string }) | null>(null);
  const [stRes, setStRes] = useState<(StatusRes & { key: string }) | null>(null);
  useEffect(() => {
    if (!side || !day) return;
    let alive = true;
    const k = key;
    void window.api.getStrategyEvalLatest(day, side, replayX).then((r) => {
      // a failed read shows nothing rather than the answer of an earlier read
      if (alive) setEvalRes(r.ok ? { key: k, last: r.last, count: r.count, total: r.total } : null);
    });
    if (replayX !== null) return () => void (alive = false); // the paper files are the day's last state: not read against a replay cursor
    const pk = `paper:${side}:${day}`;
    const kept = peek<PaperRes>(pk);
    if (kept?.ok) setPaperRes({ key: k, ...kept });
    void window.api.getPaper(day, side).then((r) => {
      if (r.ok) put(pk, { day, ...r });
      if (alive && r.ok) setPaperRes({ key: k, day, ...r });
    });
    const sk = `paperStatus:${side}:${day}`;
    const keptS = peek<StatusRes>(sk);
    if (keptS?.ok) setStRes({ key: k, ...keptS });
    void window.api.getPaperStatus(day, side).then((r) => {
      if (r.ok) put(sk, { day, ...r });
      if (alive && r.ok) setStRes({ key: k, day, ...r });
    });
    return () => {
      alive = false;
    };
  }, [key, tick, status.conn]); // eslint-disable-line react-hooks/exhaustive-deps

  const evals = evalRes && evalRes.key === key ? evalRes : null;
  // the paper files are the day's last state: not shown against a replay cursor
  const paperTrades = replayX === null && paperRes && paperRes.key === key ? paperRes.trades : undefined;
  const pst = replayX === null && stRes && stRes.key === key ? stRes.status : null;

  const rows = useMemo(
    () =>
      methodRows(method, {
        symbols: snap?.symbols || [],
        trades: snap?.trades || [],
        condTitle: (k) => condTitle(conditions, 'buy', k),
        reasonKo: (r) => reasonKo(r) || '',
        evalLast: side ? evals?.last : undefined,
        evalCount: side ? evals?.count : undefined,
        paperStatus: side ? pst?.symbols : undefined,
        paperTrades: side ? paperTrades : undefined,
      }),
    [method, snap, conditions, side, evals, pst, paperTrades],
  );
  const totals = useMemo(() => methodTotals(rows), [rows]);
  const shown = useMemo(() => filterMethodRows(rows, filter), [rows, filter]);
  const fills = useMemo(() => (snap?.trades || []).filter((t) => strategyOfTrade(t) === method).sort((a, b) => b.ts - a.ts), [snap, method]);

  const blockNote = (() => {
    if (!side) return null;
    const notes = (snap?.strategyNotes || []).filter((n) => n.strategy === side && n.mode === 'live');
    const last = notes[notes.length - 1];
    const ref = snap?.lastLineTs || 0;
    return last && last.kind === 'blocked' && ref - last.lastTs <= BLOCK_NOTE_FRESH_MS ? last : null;
  })();
  const liveSum = method === 'edge' ? parseEdgeLiveSummary(control.status) : method === 'vwap' ? parseVwapLiveSummary(control.status) : null;
  const limits = method === 'edge' ? control.edgeLiveLimits : method === 'vwap' ? control.vwapLiveLimits : undefined;
  const maxEntries = limits ? limits.entries.running ?? limits.entries.effective : null;

  const inSnap = (code: string) => !!snap?.symbols.some((s) => s.code === code);
  const go = (r: MethodRow) => {
    // a symbol the day's bot log does not have (tick-collector symbols) has no timeline: show it in the other windows instead
    if (!inSnap(r.code)) return showCode(r.code);
    setTimelineMethod(method);
    select(r.code);
    setTab('timeline');
  };
  const openSettings = () =>
    method === 'rule' ? window.dispatchEvent(new Event('rule-settings')) : window.dispatchEvent(new CustomEvent('strategy-settings', { detail: method }));

  const modeCls = mode.live ? 'live' : mode.paper ? 'paper' : 'off';
  const traceNote =
    side && status.live && replayX === null && (mode.live || mode.paper) && evals && evals.total === 0
      ? '오늘 봉별 판정 기록이 없습니다 (2026-11-27 이후 버전의 r003 이 판정한 1분봉부터 쌓입니다). 지금은 봇 로그 · 가상 러너 상태만 표시합니다.'
      : '';

  const th = (i: number, label: string, cls = '', title?: string) => (
    <th className={cls} title={title}>
      {label}
      <ColGrip cols={cols} i={i} />
    </th>
  );

  return (
    <div className="method-view">
      <div className="toolbar method-pick">
        <div className="seg" title="현황을 볼 매매 방식">
          {TRADE_STRATEGIES.map((m) => (
            <button key={m} type="button" className={method === m ? 'on' : ''} onClick={() => setMethod(m)}>
              {TRADE_STRATEGY_KO[m]}
              <i className={`tl-mode ${modes[m].live ? 'live' : modes[m].paper ? 'paper' : 'off'}`}>{strategyModeKo(modes[m]).replace(' 매매', '')}</i>
            </button>
          ))}
        </div>
        <span className="muted small">{day ? `${day.slice(0, 4)}-${day.slice(4, 6)}-${day.slice(6, 8)}` : ''}{replayX !== null ? ' · 리플레이 시점 기준' : status.live ? ' · 실시간' : ''}</span>
      </div>

      <div className="method-head">
        <div className="method-mode">
          <b>{TRADE_STRATEGY_KO[method]}</b>
          <span className={`tl-mode ${modeCls}`} title={mode.fromRunning ? '실행 중인 봇(r003)의 설정' : '봇이 꺼져 있어 저장된 설정입니다'}>
            {strategyModeKo(mode)}
          </span>
          <span className="muted small">{mode.fromRunning ? '실행 중인 봇 기준' : '저장된 설정 기준 (봇 꺼짐)'}</span>
          {mode.pending && <span className="badge-warn">저장된 설정이 다릅니다 → r003 재시작 후 적용</span>}
        </div>
        <div className="chips">
          <button type="button" className="chip link" title="봇 시작 / 정지 / 재시작, 한도 · 청산 기준 등은 제어 탭에서 합니다" onClick={() => setTab('control')}>
            제어 (재시작 · 한도) →
          </button>
          <button type="button" className="chip link" title={`${TRADE_STRATEGY_KO[method]} 세부 설정 창`} onClick={openSettings}>
            세부 설정 →
          </button>
          <button type="button" className="chip link" title="매매 결과 탭에서 이 방식의 실제 매매만 봅니다 (기간 합계 · CSV 포함)" onClick={() => openResults(method, 'live')}>
            실제 결과 →
          </button>
          {side && (
            <button type="button" className="chip link" title="가상 매매 탭의 이 방식 장부" onClick={() => openResults(side, 'paper')}>
              가상 결과 →
            </button>
          )}
        </div>
      </div>

      <div className="method-switch">
        {method === 'rule' && (
          <label className={`opt-card${sw.rulesOn ? ' on' : ''}${swOff || !sw.has.rules || !sw.normalMode ? ' dis' : ''}`} title="규칙 매매 신규 매수 (서버 RULE_BUY_ENABLED). 끄면 조건을 통과해도 주문하지 않습니다(로그만). r003 재시작 후 적용">
            <input type="checkbox" checked={sw.rulesOn} disabled={swOff || !sw.has.rules || !sw.normalMode} onChange={(e) => void sw.setServerKind('rules', e.target.checked)} />
            <span className="opt-txt">
              <b>규칙 매매</b>
              <span>{sw.rulesOn ? '켜짐 · 조건 통과 시 실제 매수 주문' : '꺼짐 · 신규 매수 주문 없음'}</span>
            </span>
          </label>
        )}
        {method === 'edge' && (
          <>
            <label className={`opt-card${sw.edgeOn ? ' on' : ''}${swOff || !sw.has.edge || !sw.normalMode ? ' dis' : ''}`} title="Edge AI 매매 (서버 EDGE_ONLY_MODE). 켜면 가상 매매로 시작합니다. r003 재시작 후 적용">
              <input type="checkbox" checked={sw.edgeOn} disabled={swOff || !sw.has.edge || !sw.normalMode} onChange={(e) => void sw.setServerKind('paper', e.target.checked)} />
              <span className="opt-txt">
                <b>Edge AI 매매</b>
                <span>{sw.edgeOn ? '켜짐' : '꺼짐'}</span>
              </span>
            </label>
            {sw.edgeOn && (
              <label className={`opt-card danger${sw.edgeLiveOn ? ' on' : ''}${swOff || !sw.has.edgeLive ? ' dis' : ''}`} title="실제 계좌로 Edge AI 주문 (서버 EDGE_LIVE_ORDERS). 승격된 모델 · 자동 매도 전체일 때만 켤 수 있습니다. r003 재시작 후 적용">
                <input type="checkbox" checked={sw.edgeLiveOn} disabled={swOff || !sw.has.edgeLive} onChange={(e) => void sw.setEdgeLive(e.target.checked)} />
                <span className="opt-txt">
                  <b>
                    실제 매매 <span className="tag danger">LIVE</span>
                  </b>
                  <span>{sw.edgeLiveOn ? '켜짐 · 실제 계좌로 주문' : '꺼짐 · 가상 매매만'}</span>
                </span>
              </label>
            )}
          </>
        )}
        {method === 'vwap' && (
          <>
            <label className={`opt-card${sw.vwapOn ? ' on' : ''}${swOff || !sw.has.vwap ? ' dis' : ''}`} title="VWAP 매매 (서버 VWAP_TRADE_MODE off / paper). 켜면 가상 매매로 시작합니다. r003 재시작 후 적용">
              <input type="checkbox" checked={sw.vwapOn} disabled={swOff || !sw.has.vwap} onChange={(e) => void sw.setVwap(e.target.checked)} />
              <span className="opt-txt">
                <b>VWAP 매매</b>
                <span>{sw.vwapOn ? '켜짐' : '꺼짐'}</span>
              </span>
            </label>
            {sw.vwapOn && (
              <label className={`opt-card danger${sw.vwapLiveOn ? ' on' : ''}${swOff || !sw.has.vwapLive ? ' dis' : ''}`} title="실제 계좌로 VWAP 매매 주문 (서버 VWAP_TRADE_LIVE_ORDERS). 자동 매도가 전체일 때만 켤 수 있습니다. r003 재시작 후 적용">
                <input type="checkbox" checked={sw.vwapLiveOn} disabled={swOff || !sw.has.vwapLive} onChange={(e) => void sw.setVwapLive(e.target.checked)} />
                <span className="opt-txt">
                  <b>
                    실제 매매 <span className="tag danger">LIVE</span>
                  </b>
                  <span>{sw.vwapLiveOn ? '켜짐 · 실제 계좌로 주문' : '꺼짐 · 가상 매매만'}</span>
                </span>
              </label>
            )}
          </>
        )}
        <span className="muted small">
          {!connected
            ? '연결되어 있지 않아 바꿀 수 없습니다.'
            : method !== 'vwap' && !sw.normalMode
              ? '서버 매매 방식이 규칙 기반(normal)이 아니라 여기서 바꿀 수 없습니다 (제어 탭에서 확인).'
              : '저장되는 설정입니다. 바꾼 뒤 r003 을 재시작해야 적용됩니다 (재시작은 제어 탭).'}
        </span>
      </div>

      <div className="kpis">
        <div className="kpi" title={TARGET_NOTE[method]}>
          <span>감시 대상</span>
          <b>{totals.targets}종목</b>
          <small className="muted">
            감시 {totals.watched}
            {totals.collector ? ` · 수집기 ${totals.collector}` : ''} · 오늘 기록 {totals.recorded}
          </small>
        </div>
        <div className="kpi">
          <span>{method === 'rule' ? '오늘 매수' : '오늘 신호 · 주문'}</span>
          <b>{method === 'rule' ? `${totals.buys}건` : `${totals.signals} · ${totals.orders}`}</b>
          {side && <small className="muted">실제 매수 체결 {totals.buys}건{maxEntries ? ` / 한도 ${maxEntries}회` : ''}</small>}
        </div>
        <div className="kpi">
          <span>오늘 매도 · 실현 손익 (실제)</span>
          <b className={pnlClass(totals.sells ? totals.realizedKrw : undefined)}>{totals.sells ? `${totals.sells}건 · ${fmtKrw(totals.realizedKrw)}` : '0건'}</b>
        </div>
        <div className="kpi">
          <span>보유 (실제)</span>
          <b>{totals.holding}종목</b>
        </div>
        {side && (totals.paperBuys > 0 || totals.paperHolding > 0 || mode.paper) && (
          <div className="kpi">
            <span>가상 매매</span>
            <b>
              매수 {totals.paperBuys} · 청산 {totals.paperSells}
            </b>
            <small className="muted">가상 보유 {totals.paperHolding}종목</small>
          </div>
        )}
      </div>

      {(blockNote || liveSum || traceNote) && (
        <div className="method-notes">
          {blockNote && (
            <span
              className="badge-warn"
              title={`봇 로그 그대로: "신규 매수 중단 | ${blockNote.text}" - 처음 ${fmtKstTime(blockNote.ts)}, 마지막 ${fmtKstTime(blockNote.lastTs)} 기록. 마지막 기록 뒤 11분이 지나면 이 표시는 사라집니다`}
            >
              봇 기록 {fmtKstTime(blockNote.ts).slice(0, 5)}~{fmtKstTime(blockNote.lastTs).slice(0, 5)} 신규 매수 중단: {blockReasonKo(blockNote.text)}
            </span>
          )}
          {liveSum && (
            <span className={liveSum.unpricedSells > 0 ? 'badge-warn' : 'muted small'} title="봇의 체결 장부 기준 (제어 탭과 같은 값)">
              체결 장부: {edgeLiveSummaryText(liveSum, maxEntries)}
            </span>
          )}
          {traceNote && <span className="muted small">{traceNote}</span>}
        </div>
      )}

      <div className="toolbar method-bar">
        <h4>감시 · 기록 종목</h4>
        <div className="seg">
          {FILTERS.map((f) => (
            <button key={f} type="button" className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>
              {METHOD_FILTER_KO[f]} {filterMethodRows(rows, f).length}
            </button>
          ))}
        </div>
        <span className="muted small">{TARGET_NOTE[method]} 행을 누르면 타임라인({TRADE_STRATEGY_KO[method]} 탭)으로 이동</span>
      </div>
      <div className="table-wrap">
        <ColReset cols={cols} />
        <table {...colTable(cols, 'grid compact method-grid')}>
          <ColGroup cols={cols} />
          <thead>
            <tr>
              {th(0, '코드')}
              {th(1, '종목명')}
              {th(2, '현재가 (전일 대비)', 'num')}
              {th(3, '구분', '', '감시 = r003 이 지금 감시하는 종목 · 수집기 = 틱 수집기 종목(Edge AI 가상) · 기록 = 감시 대상은 아니지만 이 방식의 보유 / 체결 / 기록이 있는 종목')}
              {th(4, '지금 상태 (봇의 마지막 기록)', '', '이 방식에 대해 봇이 마지막으로 남긴 기록입니다. 앱이 판정하지 않습니다')}
              {th(5, '기록 시각')}
              {th(6, method === 'rule' ? '평가 수 · 매수 체결' : '판정 봉 · 신호 · 주문', 'num', method === 'rule' ? '오늘 조건 평가 횟수 · 매수 체결 건수' : '오늘 판정한 1분봉 수(봉별 판정 기록) · 매수 신호 · 매수 주문')}
              {th(7, '보유', 'num', '실제 보유: 수량 @ 원가 (봇이 마지막 보유 점검에 적은 손익) · 가상 보유는 "가상"')}
              {th(8, '오늘 매수 / 매도', 'num')}
              {th(9, '실현 손익', 'num', '실제 매매의 실현 손익 · 가상 매매는 순수익률 합')}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr
                key={r.code}
                className={`${r.code === selected ? 'sel' : ''}${r.holdQty || r.paperEntry !== undefined ? ' row-held' : ''}`}
                onClick={() => go(r)}
                onDoubleClick={() => showCode(r.code)}
                title={inSnap(r.code) ? '누르면 타임라인 · 두 번 누르면 다른 창에 종목 표시' : '봇의 오늘 로그에 없는 종목(수집기 종목 등)이라 타임라인이 없습니다. 누르면 다른 창(차트 등)에 종목을 표시합니다'}
              >
                <td className="mono">{r.code}</td>
                <td title={r.name}>
                  {r.name} {r.nxt && <NxtMark />}
                </td>
                <td className="num">
                  <PriceTag code={r.code} />
                </td>
                <td>
                  {r.watched && <span className="tag">감시</span>}
                  {r.collector && <span className="tag">수집기</span>}
                  {r.extra && <span className="tag lock">기록</span>}
                  {r.holdQty ? <span className="tag apply-on">보유</span> : null}
                  {r.paperEntry !== undefined && <span className="tag apply-record">가상 보유</span>}
                </td>
                <td className={`reason res-${r.stateKind === 'none' || r.stateKind === 'info' ? '' : r.stateKind}`} title={r.stateRaw ? `${METHOD_SRC_KO[r.stateSrc]}: ${r.state}\n${r.stateRaw}` : r.state}>
                  {r.stateSrc && r.stateSrc !== 'rule' && (
                    <i className="src" title="이 기록이 실제 주문 경로의 것인지 가상 매매 러너의 것인지">
                      {METHOD_SRC_KO[r.stateSrc]}
                      {r.stateMode ? ` · ${r.stateMode === 'live' ? '실제' : '가상'}` : ''}
                    </i>
                  )}
                  {r.state}
                </td>
                <td className="mono">{r.stateTs ? fmtKstTime(r.stateTs) : '-'}</td>
                <td className="num">{method === 'rule' ? `${fmtInt(r.evals)} · ${r.orders}` : `${r.evals || '-'} · ${r.signals} · ${r.orders}`}</td>
                <td className="num">
                  {r.holdQty ? `${r.holdQty}주${r.holdAvg ? ` @ ${fmtInt(r.holdAvg)}` : ''}` : ''}
                  {r.holdQty && r.holdPnlPct !== undefined ? <span className={pnlClass(r.holdPnlPct)}> {sp(r.holdPnlPct)}</span> : null}
                  {r.holdQty && r.holdExitAt ? (
                    <span className="badge-warn" title={`봇이 ${fmtKstTime(r.holdExitAt)} 에 청산을 결정했지만 그 뒤 매도 체결 기록이 없습니다. 계좌에서 확인하세요`}>
                      {' '}
                      청산 결정 {fmtKstTime(r.holdExitAt).slice(0, 5)} · 체결 기록 없음
                    </span>
                  ) : null}
                  {r.paperEntry !== undefined ? `${r.holdQty ? ' · ' : ''}가상 @ ${fmtInt(r.paperEntry)}` : ''}
                </td>
                <td className="num">
                  {r.buys || r.sells ? `${r.buys} / ${r.sells}` : ''}
                  {r.paperBuys || r.paperSells ? <span className="muted">{`${r.buys || r.sells ? ' · ' : ''}가상 ${r.paperBuys} / ${r.paperSells}`}</span> : null}
                </td>
                <td className={`num ${pnlClass(r.sells ? r.realizedKrw : undefined)}`}>
                  {r.sells ? fmtKrw(r.realizedKrw) : ''}
                  {r.paperSells ? <span className={`muted ${pnlClass(r.paperNetPct)}`}>{`${r.sells ? ' · ' : ''}가상 ${sp(r.paperNetPct)}`}</span> : null}
                </td>
              </tr>
            ))}
            {!shown.length && (
              <tr className="nohover">
                <td colSpan={10} className="muted center">
                  {rows.length ? '이 조건에 맞는 종목이 없습니다.' : '표시할 종목이 없습니다 (이 날짜의 봇 로그에 감시 종목이 없습니다).'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="toolbar method-bar">
        <h4>오늘 실제 체결 ({TRADE_STRATEGY_KO[method]})</h4>
        <span className="muted small">최근 {Math.min(fills.length, 8)}건 / 전체 {fills.length}건</span>
        <button type="button" className="chip link" onClick={() => openResults(method, 'live')}>
          매매 결과에서 전체 보기 →
        </button>
      </div>
      <div className="table-wrap">
        <ColReset cols={fillCols} />
        <table {...colTable(fillCols, 'grid compact')}>
          <ColGroup cols={fillCols} />
          <thead>
            <tr>
              {['시각', '종목', '구분', '가격', '수량', '사유', '손익'].map((h, i) => (
                <th key={h} className={i === 3 || i === 4 || i === 6 ? 'num' : ''}>
                  {h}
                  <ColGrip cols={fillCols} i={i} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {fills.slice(0, 8).map((t, i) => (
              <tr
                key={`${t.code}-${t.ts}-${i}`}
                onClick={() => {
                  setTimelineMethod(method);
                  select(t.code);
                  setTab('timeline');
                }}
              >
                <td className="mono">{fmtKstTime(t.ts)}</td>
                <td>
                  <span className="mono">{t.code}</span> {t.name}
                </td>
                <td className={t.side === 'BUY' ? 'gain' : 'loss'}>{t.side === 'BUY' ? '매수' : '매도'}</td>
                <td className="num" title={t.approx ? '체결 기록이 없어 주문가로 표시 (주문번호로 이은 거래)' : undefined}>
                  {t.approx ? '≈ ' : ''}
                  {fmtInt(t.price)}
                </td>
                <td className="num">{fmtInt(t.qty)}</td>
                <td className="reason" title={t.reason}>
                  {sideExitKo(t.reason) || reasonKo(t.reason) || t.reason}
                </td>
                <td className={`num ${pnlClass(t.pnlKrw)}`}>{t.side === 'SELL' ? fmtKrw(t.pnlKrw) : ''}</td>
              </tr>
            ))}
            {!fills.length && (
              <tr className="nohover">
                <td colSpan={7} className="muted center">
                  오늘 이 방식의 실제 체결이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
