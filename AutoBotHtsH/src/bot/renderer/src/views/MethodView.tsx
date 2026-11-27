/*
 * Update log
 * - [2026-11-27] type=feat - "매매 방식 현황" (user 2026-11-27: which symbols does the VWAP trade look at; one place per trading method
 *   for its symbols, their state and the results; plan reviewed with Codex). Display only: rows come from shared/methodBoard
 * - [2026-11-27] type=feat - On / off on this page through the control tab's own handlers (components/useMethodSwitch)
 * - [2026-11-27] type=feat - Dashboard (user mock-up + guide 2026-11-27, Codex review): tabs 전체 현황 / 규칙 / Edge AI / VWAP; five
 *   unit cards (규칙, Edge AI 가상 · 실전, VWAP 가상 · 실전) with the STORED switch and the RUNNING bot's state apart
 *   (shared/methodUnits); 전체 현황 = summary (real and paper never added together) + one row per symbol with every method's
 *   last record; a method tab = left panel (basic info, the conditions / exits / limits the app can read, today's result) + the
 *   symbol table + fills; a recent-record line at the bottom. Per-method target symbols (server RULE_ / EDGE_ / VWAP_TARGET_CODES)
 *   are picked on the method's table and stored through r016 (after a restart). Nothing is judged here and nothing the app
 *   cannot read is shown (no strategy version names, no "conditions met n/m", no measured values per condition). Controls are
 *   locked while a past day / replay is shown.
 * - [2026-11-27] type=feat - 대상 구성 card on a method tab (components/TargetCompose): the composition from the scan, its edits and apply
 */

import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../store';
import { ColGrip, ColGroup, ColReset, colTable, useColWidths, type ColWidths } from '../components/useColWidths';
import { PriceTag } from '../components/PriceTag';
import { NxtMark } from '../components/NxtTag';
import { useMethodSwitch } from '../components/useMethodSwitch';
import { TargetCompose } from '../components/TargetCompose';
import { condTitle, fmtInt, fmtKrw, pnlClass } from '../lib';
import { fmtKstTime } from '../../../shared/time';
import { peek, put } from '../dataCache';
import { showCode } from '../showCode';
import { reasonKo } from '../../../shared/reasonKo';
import { TRADE_STRATEGIES, TRADE_STRATEGY_KO, TRADE_STRATEGY_SHORT, strategyModeKo, strategyModes, strategyOfTrade, strategySummary, sideExitKo, type TradeStrategy } from '../../../shared/strategy';
import { blockReasonKo, STAGE_ROWS, type SideStrategy } from '../../../shared/strategyEvents';
import type { StrategyEvalRowX } from '../../../shared/sideTimeline';
import {
  EDGE_LIVE_LIMIT_KINDS,
  EDGE_LIVE_LIMIT_KO,
  EDGE_LIVE_LIMIT_UNIT,
  edgeLiveSummaryText,
  parseEdgeLiveSummary,
  parseVwapLiveSummary,
  type EdgeModelInfo,
  type PaperExitState,
  type PaperStatus,
  type PaperTrade,
} from '../../../shared/paper';
import { METHOD_FILTER_KO, METHOD_SRC_KO, filterMethodRows, methodRows, methodTotals, type MethodRow, type MethodRowFilter, type MethodTotals } from '../../../shared/methodBoard';
import { METHOD_TARGET_KEYS, UNIT_KO, UNIT_SHORT, runningUnits, targetListToStore, targetsInUse, targetsKo, unitInfos, type UnitInfo, BLOCK_NOTE_FRESH_MS, liveNotes } from '../../../shared/methodUnits';

const REFRESH_MS = 15000;
const FILTERS: MethodRowFilter[] = ['all', 'target', 'recorded', 'active', 'holding'];
const TARGET_NOTE: Record<TradeStrategy, string> = {
  rule: '규칙 매매는 r003 이 감시하는 종목(r004 감시목록)을 조건 순서대로 평가합니다.',
  edge: 'Edge AI 매매는 감시 종목을 평가합니다. 가상 매매는 틱 수집기 종목도 함께 봅니다 (실제 주문은 감시 종목만).',
  vwap: 'VWAP 매매는 규칙 매매와 같은 감시 종목(r004 감시목록)을 1분봉마다 평가합니다. ETF · ETN 제외 등은 봇이 적용합니다.',
};
type EvalRes = { key: string; last: Record<string, StrategyEvalRowX>; count: Record<string, number>; total: number };
type PaperRes = { day: string; ok: boolean; trades: PaperTrade[] };
type StatusRes = { day: string; ok: boolean; status: PaperStatus | null };
type SideData = { evals: EvalRes | null; paperTrades: PaperTrade[] | undefined; pst: PaperStatus | null };
type AllFilter = 'all' | 'watch' | 'traded' | 'holding';
const ALL_FILTER_KO: Record<AllFilter, string> = { all: '전체', watch: '감시', traded: '체결', holding: '보유' };
type Go = (code: string, m: TradeStrategy | 'auto') => void;

/** [2026-11-28, Codex review] the day the page shows: "오늘" only on the live day, else its date (a past day is not today) */
const DayCtx = createContext('오늘');

const sp = (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`;
const ymd = (d: string | undefined) => (d ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : '');

/** the bot's files of one side strategy for the day: last judged bar per symbol (trace), paper trades, paper runner status.
 *  Re-read at most every REFRESH_MS on the live day; the paper files are the day's last state and are not read in replay */
function useSideData(side: SideStrategy, day: string | undefined, replayX: number | null, tick: number, hostKey: string): SideData {
  const key = day ? `${hostKey}/${day}/${side}/${replayX ?? ''}` : '';
  const [evalRes, setEvalRes] = useState<EvalRes | null>(null);
  const [paperRes, setPaperRes] = useState<(PaperRes & { key: string }) | null>(null);
  const [stRes, setStRes] = useState<(StatusRes & { key: string }) | null>(null);
  useEffect(() => {
    if (!day) return;
    let alive = true;
    const k = key;
    void window.api.getStrategyEvalLatest(day, side, replayX).then((r) => {
      if (alive) setEvalRes(r.ok ? { key: k, last: r.last, count: r.count, total: r.total } : null);
    });
    if (replayX !== null) return () => void (alive = false);
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
  }, [key, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  return {
    evals: evalRes && evalRes.key === key ? evalRes : null,
    paperTrades: replayX === null && paperRes && paperRes.key === key ? paperRes.trades : undefined,
    pst: replayX === null && stRes && stRes.key === key ? stRes.status : null,
  };
}

export function MethodView() {
  const tab = useStore((s) => s.methodTab);
  const setTabM = useStore((s) => s.setMethodTab);
  const snap = useStore((s) => s.snapshot);
  const status = useStore((s) => s.status);
  const control = useStore((s) => s.control);
  const conditions = useStore((s) => s.conditions);
  const select = useStore((s) => s.select);
  const setTab = useStore((s) => s.setTab);
  const setTimelineMethod = useStore((s) => s.setTimelineMethod);
  const openResults = useStore((s) => s.openResults);
  const selected = useStore((s) => s.selected);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const cols = useColWidths('autoboth.cols.methodCols', 10);
  const fillCols = useColWidths('autoboth.cols.methodFillCols', 7);
  const allCols = useColWidths('autoboth.cols.methodAllCols', 9);
  const [filter, setFilter] = useState<MethodRowFilter>('all');
  const [allFilter, setAllFilter] = useState<AllFilter>('all');

  const method: TradeStrategy | null = tab === 'all' ? null : tab;
  const connected = status.conn === 'connected';
  const day = snap?.date || status.date;
  const replayX = snap?.replay?.cursor ?? null;
  // a past day or a replay: everything is read-only (the switches and the target lists are today's settings)
  const liveNow = !!status.live && replayX === null;
  const tick = status.live && snap ? Math.floor(snap.generatedAt / REFRESH_MS) : 0;
  const hostKey = `${status.mode}/${status.conn}`;

  // on / off: the control tab's own handlers (same confirmations)
  const [swBusy, setSwBusy] = useState(false);
  const [edgeModel, setEdgeModel] = useState<EdgeModelInfo | null | undefined>(() => peek<EdgeModelInfo | null>('control:edge'));
  useEffect(() => {
    if (!connected) return setEdgeModel(undefined);
    let alive = true;
    void window.api.getEdgeModel().then((r) => {
      if (r.ok) put('control:edge', r.model);
      if (alive) setEdgeModel(r.ok ? r.model : peek<EdgeModelInfo | null>('control:edge'));
    });
    return () => {
      alive = false;
    };
  }, [connected, status.mode]);
  const sw = useMethodSwitch({ setBusy: setSwBusy, edgeModel });
  const swOff = !connected || swBusy || !liveNow;

  const modes = useMemo(() => strategyModes(control), [control]);
  const botRunning = control.status ? !!control.status.running : null;
  // what the bot itself said about its real-order strategies today (start line / block line), newest per strategy
  const liveNote = useMemo(() => liveNotes(snap?.strategyNotes, snap?.lastLineTs || 0), [snap]);
  const units = useMemo(() => unitInfos(control, botRunning, liveNow ? { dryRun: !!control.status?.dry_run, liveNote } : {}), [control, botRunning, liveNow, liveNote]);
  const edge = useSideData('edge', day, replayX, tick, hostKey);
  const vwap = useSideData('vwap', day, replayX, tick, hostKey);
  const sideData: Record<SideStrategy, SideData> = { edge, vwap };

  const rowsBy = useMemo(() => {
    const out = {} as Record<TradeStrategy, MethodRow[]>;
    for (const m of TRADE_STRATEGIES) {
      const d = m === 'rule' ? null : sideData[m];
      out[m] = methodRows(m, {
        symbols: snap?.symbols || [],
        trades: snap?.trades || [],
        condTitle: (k) => condTitle(conditions, 'buy', k),
        reasonKo: (r) => reasonKo(r) || '',
        evalLast: d?.evals?.last,
        evalCount: d?.evals?.count,
        paperStatus: d?.pst?.symbols,
        paperTrades: d?.paperTrades,
        paperStateOff: liveNow && modes[m].live === true,
        targets: liveNow ? targetsInUse(control.targets?.[m], botRunning).list : undefined,
      });
    }
    return out;
  }, [snap, conditions, edge.evals, edge.pst, edge.paperTrades, vwap.evals, vwap.pst, vwap.paperTrades, liveNow, modes, control.targets, botRunning]); // eslint-disable-line react-hooks/exhaustive-deps
  const totalsBy = useMemo(() => ({ rule: methodTotals(rowsBy.rule), edge: methodTotals(rowsBy.edge), vwap: methodTotals(rowsBy.vwap) }), [rowsBy]);
  const lastTsBy = useMemo(() => {
    const out = {} as Record<TradeStrategy, number>;
    for (const m of TRADE_STRATEGIES) out[m] = rowsBy[m].reduce((a, r) => Math.max(a, r.stateTs || 0), 0);
    return out;
  }, [rowsBy]);

  const inSnap = (code: string) => !!snap?.symbols.some((s) => s.code === code);
  const go: Go = (code, m) => {
    if (!inSnap(code)) return showCode(code);
    setTimelineMethod(m);
    select(code);
    setTab('timeline');
  };

  // [2026-11-28] a method's paper and real cards: one of the two runs (user 2026-11-28).
  //   paper ON  = the method on, in paper: switches real orders off when they are on, else switches the method on
  //   paper OFF = the method off
  //   real ON   = real orders on (the method is switched on first when it is off - it stops there if that was not stored)
  //   real OFF  = back to paper (the method stays on; the paper card switches it off)
  const setUnit = (u: UnitInfo, on: boolean) => {
    if (u.key === 'rule') return void sw.setServerKind('rules', on);
    const edgeM = u.method === 'edge';
    const methodOn = edgeM ? sw.edgeOn : sw.vwapOn;
    const liveOn = edgeM ? sw.edgeLiveOn : sw.vwapLiveOn;
    const setMethod = (v: boolean) => (edgeM ? sw.setServerKind('paper', v) : sw.setVwap(v));
    const setLive = (v: boolean) => (edgeM ? sw.setEdgeLive(v) : sw.setVwapLive(v));
    void (async () => {
      if (u.mode === 'paper') {
        if (!on) return void (await setMethod(false));
        if (!methodOn) return void (await setMethod(true));
        if (liveOn) await setLive(false);
        return;
      }
      if (!on) return void (await setLive(false));
      if (!methodOn) {
        // [Codex review] two stored steps: say so first - cancelling the second leaves the method on in paper
        if (!window.confirm(`${TRADE_STRATEGY_KO[u.method]}가 꺼져 있습니다.\n\n먼저 이 방식을 켜고(가상 매매로 저장) 이어서 실제 매매를 켭니다.\n실제 매매 확인을 취소하거나 조건이 맞지 않으면 가상 매매로 켜진 상태로 남습니다.\n\n진행할까요?`)) return;
        if (!(await setMethod(true))) return;
      }
      await setLive(true);
    })();
  };

  // recent records: the last thing the bot wrote per symbol and method + the real fills
  const logItems = useMemo(() => {
    const items: { ts: number; method: TradeStrategy; mode?: 'live' | 'paper'; code: string; name: string; text: string }[] = [];
    for (const m of TRADE_STRATEGIES) {
      if (method && m !== method) continue;
      for (const r of rowsBy[m]) if (r.stateTs && r.stateSrc) items.push({ ts: r.stateTs, method: m, mode: r.stateMode ?? (m === 'rule' ? 'live' : undefined), code: r.code, name: r.name, text: r.state });
    }
    for (const t of snap?.trades || []) {
      const m = strategyOfTrade(t);
      if (method && m !== method) continue;
      items.push({ ts: t.ts, method: m, mode: 'live', code: t.code, name: t.name, text: `${t.side === 'BUY' ? '매수' : '매도'} 체결 ${fmtInt(t.qty)}주 @ ${t.approx ? '≈' : ''}${fmtInt(t.price)}${t.side === 'SELL' && Number.isFinite(t.pnlKrw) ? ` · ${fmtKrw(t.pnlKrw)}` : ''}` });
    }
    return items.sort((a, b) => b.ts - a.ts).slice(0, 6);
  }, [rowsBy, snap, method]);

  const dayKo = liveNow || !day ? '오늘' : `${day.slice(4, 6)}-${day.slice(6, 8)}`;
  return (
    <DayCtx.Provider value={dayKo}>
    <div className="method-view">
      <div className="toolbar method-pick">
        <div className="seg mtabs" title="전체 현황 또는 매매 방식 하나">
          <button type="button" className={tab === 'all' ? 'on' : ''} onClick={() => setTabM('all')}>
            전체 현황
          </button>
          {TRADE_STRATEGIES.map((m) => (
            <button key={m} type="button" className={`m-${m} ${tab === m ? 'on' : ''}`} onClick={() => setTabM(m)}>
              {TRADE_STRATEGY_KO[m]}
              <i className={`tl-mode ${modes[m].live ? 'live' : modes[m].paper ? 'paper' : 'off'}`}>{strategyModeKo(modes[m]).replace(' 매매', '')}</i>
            </button>
          ))}
        </div>
        <span className={`run-dot ${liveNow ? 'on' : ''}`} />
        <span className="muted small">
          {ymd(day)}
          {replayX !== null ? ' · 리플레이 시점 기준 (조작 잠김)' : status.live ? ` · 실시간${snap ? ` (${fmtKstTime(snap.generatedAt)} 갱신)` : ''}` : ' · 지난 날짜 (조작 잠김)'}
        </span>
        <span className="chips push-right">
          <button type="button" className="chip link" title="봇 시작 / 정지 / 재시작, 한도 · 청산 기준" onClick={() => setTab('control')}>
            제어 →
          </button>
          <button type="button" className="chip link" title="실제 매매 결과 (기간 합계 · CSV)" onClick={() => openResults(method ?? 'all', 'live')}>
            매매 결과 →
          </button>
          <button type="button" className="chip link" title="가상 매매 장부" onClick={() => setTab('paper')}>
            가상 매매 →
          </button>
        </span>
      </div>

      <div className="unit-cards">
        {units.map((u) => (
          <UnitCard
            key={u.key}
            u={u}
            selected={method === u.method}
            totals={totalsBy[u.method]}
            lastTs={lastTsBy[u.method]}
            targets={targetsInUse(control.targets?.[u.method], botRunning)}
            disabled={swOff}
            onSelect={() => setTabM(u.method)}
            onSwitch={(on) => setUnit(u, on)}
          />
        ))}
      </div>
      {!liveNow ? null : !connected ? (
        <p className="muted small method-line">연결되어 있지 않아 스위치를 바꿀 수 없습니다.</p>
      ) : (
        <p className="muted small method-line">
          스위치는 저장되는 설정입니다 (확인 창 뒤 저장, r003 재시작 후 적용 - 재시작은 제어 탭). 카드를 누르면 그 방식의 상세로 이동합니다. 한 방식의 가상 · 실전은 둘 중 하나만 켜집니다 (실전을 켜면 가상은 꺼지고, 실전을 끄면 가상으로 돌아갑니다. 방식을 완전히 끄려면 가상 카드를 끕니다).
        </p>
      )}

      {tab === 'all' ? (
        <Overview units={units} rowsBy={rowsBy} totalsBy={totalsBy} allFilter={allFilter} setAllFilter={setAllFilter} cols={allCols} selected={selected} go={go} />
      ) : (
        <MethodDetail
          method={tab}
          rows={rowsBy[tab]}
          totals={totalsBy[tab]}
          filter={filter}
          setFilter={setFilter}
          cols={cols}
          fillCols={fillCols}
          liveNow={liveNow}
          connected={connected}
          busy={swBusy}
          botRunning={botRunning}
          hostKey={hostKey}
          lastTs={lastTsBy[tab]}
          side={tab === 'rule' ? null : sideData[tab]}
          go={go}
          inSnap={inSnap}
          onSaved={bumpHistory}
        />
      )}

      <div className="method-log" title="봇이 종목 · 방식별로 마지막에 남긴 기록과 실제 체결 (최근 순)">
        <b>최근 기록</b>
        {logItems.length ? (
          logItems.map((l, i) => (
            <span key={`${l.ts}-${l.code}-${i}`} className="log-item" onClick={() => go(l.code, l.method)}>
              <i className="mono">{fmtKstTime(l.ts)}</i>
              <em className={`m-${l.method}`}>
                [{TRADE_STRATEGY_SHORT[l.method]}
                {l.mode ? `-${l.mode === 'live' ? '실전' : '가상'}` : ''}]
              </em>
              {l.name || l.code} {l.text}
            </span>
          ))
        ) : (
          <span className="muted">기록 없음</span>
        )}
      </div>
    </div>
    </DayCtx.Provider>
  );
}

// ---- unit card ----------------------------------------------------------------------------------------------------------------

function UnitCard(props: {
  u: UnitInfo;
  selected: boolean;
  totals: MethodTotals;
  lastTs: number;
  targets: { list: string[]; from: 'running' | 'stored' | 'none' | 'unknown' };
  disabled: boolean;
  onSelect: () => void;
  onSwitch: (on: boolean) => void;
}) {
  const { u, totals: t } = props;
  const dayKo = useContext(DayCtx);
  const off = props.disabled || !!u.locked || u.switchOn === null;
  const swTitle =
    u.key === 'rule'
      ? '규칙 매매 신규 매수 (저장되는 설정, r003 재시작 후 적용)'
      : u.mode === 'paper'
        ? `${UNIT_KO[u.key]} 가상 매매 (켜면 실전은 꺼집니다 · 끄면 이 방식 전체가 꺼집니다 · r003 재시작 후 적용)`
        : `${UNIT_KO[u.key]} 실제 주문 (켜면 가상은 꺼집니다 · 끄면 가상 매매로 돌아갑니다 · 확인 창 2회, r003 재시작 후 적용)`;
  return (
    <div className={`unit-card m-${u.method} ${u.mode} ${props.selected ? 'sel' : ''}`} onClick={props.onSelect} title="누르면 이 방식의 상세 (매매를 켜거나 끄지 않습니다)">
      <div className="uc-head">
        <b>{UNIT_KO[u.key]}</b>
        {u.key !== 'rule' && <i className={`tl-mode ${u.mode === 'live' ? 'live' : 'paper'}`}>{u.mode === 'live' ? '실전' : '가상'}</i>}
        <label className={`uc-switch ${u.switchOn ? 'on' : ''} ${off ? 'dis' : ''}`} title={u.locked ? `${swTitle}\n잠김: ${u.locked}` : swTitle} onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={!!u.switchOn} disabled={off} onChange={(e) => props.onSwitch(e.target.checked)} />
          <span>{u.switchOn === null ? '-' : u.switchOn ? 'ON' : 'OFF'}</span>
        </label>
      </div>
      <dl>
        <dt>모드</dt>
        <dd>{u.mode === 'live' ? '실전 매매' : '가상 매매'}</dd>
        <dt>상태</dt>
        <dd className={`ust-${u.state}`} title="실행 중인 봇이 알려 준 값 기준 (봇이 꺼져 있거나 알려 주지 않으면 실행 중으로 표시하지 않습니다)">
          {u.stateKo}
          {u.locked ? <span className="muted small"> · {u.locked}</span> : null}
        </dd>
        <dt>대상</dt>
        <dd title={props.targets.from === 'running' ? '실행 중인 봇의 대상 종목 설정' : props.targets.from === 'stored' ? '저장된 대상 종목 설정 (봇 꺼짐)' : props.targets.from === 'unknown' ? '실행 중인 봇의 대상 종목 설정을 확인할 수 없습니다 (감시 종목 수만 표시)' : '대상 종목 설정을 읽지 못했습니다'}>
          {t.targets}종목{props.targets.list.length ? ` (지정 ${props.targets.list.length})` : ''}
        </dd>
        <dt>{dayKo}</dt>
        {u.mode === 'live' ? (
          <dd>
            매수 {t.buys} · 매도 {t.sells} · 보유 {t.holding}
            {t.sells ? <b className={pnlClass(t.realizedKrw)}> {fmtKrw(t.realizedKrw)}</b> : null}
          </dd>
        ) : (
          <dd>
            가상 매수 {t.paperBuys} · 청산 {t.paperSells} · 보유 {t.paperHolding}
          </dd>
        )}
        <dt>기록</dt>
        <dd className="mono">{props.lastTs ? fmtKstTime(props.lastTs) : '-'}</dd>
      </dl>
    </div>
  );
}

// ---- overview -----------------------------------------------------------------------------------------------------------------

function cellOf(r: MethodRow | undefined): { text: string; cls: string; title: string } {
  if (!r) return { text: '-', cls: 'muted', title: '이 방식의 대상 · 기록이 아닙니다' };
  const title = r.stateRaw ? `${r.state}\n${r.stateRaw}` : r.state;
  if (r.holdQty) return { text: `보유 ${r.holdQty}주${r.holdPnlPct !== undefined ? ` ${sp(r.holdPnlPct)}` : ''}`, cls: 'held', title };
  if (r.paperEntry !== undefined) return { text: `가상 보유 @ ${fmtInt(r.paperEntry)}`, cls: 'pheld', title };
  if (r.notTarget) return { text: '대상 아님', cls: 'muted', title: '이 방식의 대상 종목 지정에 없는 종목입니다 (신규 매수 안 함)' };
  if (!r.stateSrc) return { text: '기록 없음', cls: 'muted', title };
  return { text: `${r.stateMode === 'paper' ? '[가상] ' : ''}${r.state}`, cls: `res-${r.stateKind === 'none' || r.stateKind === 'info' ? '' : r.stateKind}`, title };
}

function Overview(props: {
  units: UnitInfo[];
  rowsBy: Record<TradeStrategy, MethodRow[]>;
  totalsBy: Record<TradeStrategy, MethodTotals>;
  allFilter: AllFilter;
  setAllFilter: (f: AllFilter) => void;
  cols: ColWidths;
  selected: string | null;
  go: Go;
}) {
  const { rowsBy, totalsBy: tb, cols } = props;
  const dayKo = useContext(DayCtx);
  const by = useMemo(() => {
    const m = {} as Record<TradeStrategy, Map<string, MethodRow>>;
    for (const s of TRADE_STRATEGIES) m[s] = new Map(rowsBy[s].map((r) => [r.code, r]));
    return m;
  }, [rowsBy]);
  const all = useMemo(() => {
    const codes = new Map<string, { code: string; name: string; nxt?: boolean; watched: boolean; collector: boolean }>();
    for (const s of TRADE_STRATEGIES)
      for (const r of rowsBy[s]) {
        const e = codes.get(r.code) || { code: r.code, name: r.name, nxt: r.nxt, watched: false, collector: false };
        e.watched = e.watched || r.watched;
        e.collector = e.collector || r.collector;
        if (!e.name) e.name = r.name;
        codes.set(r.code, e);
      }
    const rank = (c: string) => {
      const rs = TRADE_STRATEGIES.map((s) => by[s].get(c));
      if (rs.some((r) => r?.holdQty)) return 0;
      if (rs.some((r) => r && r.buys + r.sells > 0)) return 1;
      if (rs.some((r) => r && (r.paperEntry !== undefined || r.signals + r.orders + r.paperBuys > 0))) return 2;
      return 3;
    };
    return [...codes.values()].sort((a, b) => rank(a.code) - rank(b.code) || a.code.localeCompare(b.code));
  }, [rowsBy, by]);
  const pick = (f: AllFilter) =>
    all.filter((c) => {
      const rs = TRADE_STRATEGIES.map((s) => by[s].get(c.code));
      if (f === 'watch') return c.watched;
      if (f === 'traded') return rs.some((r) => r && r.buys + r.sells > 0);
      if (f === 'holding') return rs.some((r) => r?.holdQty);
      return true;
    });
  const shown = pick(props.allFilter);
  const watched = all.filter((c) => c.watched).length;
  const real = TRADE_STRATEGIES.reduce((a, s) => ({ buys: a.buys + tb[s].buys, sells: a.sells + tb[s].sells, pnl: a.pnl + tb[s].realizedKrw, hold: a.hold + tb[s].holding }), { buys: 0, sells: 0, pnl: 0, hold: 0 });
  const paper = (['edge', 'vwap'] as const).reduce((a, s) => ({ buys: a.buys + tb[s].paperBuys, sells: a.sells + tb[s].paperSells, hold: a.hold + tb[s].paperHolding, net: a.net + rowsBy[s].reduce((x, r) => x + r.paperNetPct, 0) }), { buys: 0, sells: 0, hold: 0, net: 0 });
  const nRun = runningUnits(props.units);
  const th = (i: number, label: string, cls = '', title?: string) => (
    <th className={cls} title={title}>
      {label}
      <ColGrip cols={cols} i={i} />
    </th>
  );
  return (
    <>
      <div className="method-sum">
        <div className="ms-box">
          <b>종합 현황 ({dayKo})</b>
          <div className="ms-item" title="r003 이 지금 감시하는 종목 (방식이 겹쳐도 한 번만 셉니다)">
            <span>감시 종목</span>
            <b>{watched}</b>
          </div>
          <div className="ms-item" title="실제 체결 (봇 거래 로그 기준, 세 방식 합)">
            <span>실전 체결</span>
            <b>
              매수 {real.buys} · 매도 {real.sells}
            </b>
          </div>
          <div className="ms-item" title="실제 매매의 실현 손익 합 (가상 매매와 더하지 않습니다)">
            <span>실현 손익 (실전)</span>
            <b className={pnlClass(real.sells ? real.pnl : undefined)}>{real.sells ? fmtKrw(real.pnl) : '-'}</b>
          </div>
          <div className="ms-item">
            <span>보유 (실전)</span>
            <b>{real.hold}종목</b>
          </div>
          <div className="ms-item paper" title="가상 매매: 매수 / 청산 건수와 청산 건의 순수익률 합 (금액이 아니며 실전 손익과 더하지 않습니다)">
            <span>가상 매매</span>
            <b>
              매수 {paper.buys} · 청산 {paper.sells}
              {paper.sells ? <i className={pnlClass(paper.net)}> 순수익률 합 {sp(paper.net)}</i> : null}
            </b>
          </div>
        </div>
        <div className="ms-box run">
          <b title="실행 중인 봇이 신규 진입을 확인해 준 실행 단위 수">
            실행 중인 매매 방식 {nRun} / {props.units.length}
          </b>
          {props.units.map((u) => (
            <span key={u.key} className={`run-chip ust-${u.state}`} title={u.stateKo}>
              {UNIT_SHORT[u.key]} {u.entryRunning === true ? '✓' : u.state === 'unknown' ? '?' : '×'}
            </span>
          ))}
        </div>
      </div>

      <div className="toolbar method-bar">
        <h4>종목별 현황 (방식별 마지막 기록)</h4>
        <div className="seg">
          {(Object.keys(ALL_FILTER_KO) as AllFilter[]).map((f) => (
            <button key={f} type="button" className={props.allFilter === f ? 'on' : ''} onClick={() => props.setAllFilter(f)}>
              {ALL_FILTER_KO[f]} {pick(f).length}
            </button>
          ))}
        </div>
        <span className="muted small">칸 = 그 방식에 대해 봇이 마지막으로 남긴 기록 (앱이 판정하지 않습니다). 칸을 누르면 그 방식의 타임라인</span>
      </div>
      <div className="table-wrap">
        <ColReset cols={cols} />
        <table {...colTable(cols, 'grid compact method-grid method-all')}>
          <ColGroup cols={cols} />
          <thead>
            <tr>
              {th(0, '코드')}
              {th(1, '종목명')}
              {th(2, '현재가 (전일 대비)', 'num')}
              {th(3, '구분')}
              {th(4, '규칙 매매', 'm-rule')}
              {th(5, 'Edge AI 매매', 'm-edge')}
              {th(6, 'VWAP 매매', 'm-vwap')}
              {th(7, `${dayKo} 매매 (실전)`, 'num', '실제 체결: 방식 매수 / 매도')}
              {th(8, '실현 · 가상', 'num', '실제 실현 손익 · 가상 매매 순수익률 합 (따로 표시)')}
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => {
              const rs = TRADE_STRATEGIES.map((s) => by[s].get(c.code));
              const traded = TRADE_STRATEGIES.map((s, i) => {
                const r = rs[i];
                return r && r.buys + r.sells > 0 ? `${TRADE_STRATEGY_SHORT[s]} ${r.buys}/${r.sells}` : '';
              }).filter(Boolean);
              const pnl = rs.reduce((a, r) => a + (r && r.sells ? r.realizedKrw : 0), 0);
              const sold = rs.some((r) => r && r.sells > 0);
              const pNet = rs.reduce((a, r) => a + (r && r.paperSells ? r.paperNetPct : 0), 0);
              const pSold = rs.some((r) => r && r.paperSells > 0);
              return (
                <tr key={c.code} className={`${c.code === props.selected ? 'sel' : ''}${rs.some((r) => r?.holdQty) ? ' row-held' : ''}`} onClick={() => props.go(c.code, 'auto')} onDoubleClick={() => showCode(c.code)}>
                  <td className="mono">{c.code}</td>
                  <td title={c.name}>
                    {c.name} {c.nxt && <NxtMark />}
                  </td>
                  <td className="num">
                    <PriceTag code={c.code} />
                  </td>
                  <td>
                    {c.watched && <span className="tag">감시</span>}
                    {c.collector && <span className="tag">수집기</span>}
                    {!c.watched && !c.collector && <span className="tag lock">기록</span>}
                  </td>
                  {TRADE_STRATEGIES.map((s, i) => {
                    const cell = cellOf(rs[i]);
                    return (
                      <td
                        key={s}
                        className={`reason mcell ${cell.cls}`}
                        title={cell.title}
                        onClick={(e) => {
                          e.stopPropagation();
                          props.go(c.code, s);
                        }}
                      >
                        {cell.text}
                      </td>
                    );
                  })}
                  <td className="num">{traded.join(' · ')}</td>
                  <td className="num">
                    {sold ? <span className={pnlClass(pnl)}>{fmtKrw(pnl)}</span> : null}
                    {pSold ? <span className={`muted ${pnlClass(pNet)}`}>{`${sold ? ' · ' : ''}가상 ${sp(pNet)}`}</span> : null}
                  </td>
                </tr>
              );
            })}
            {!shown.length && (
              <tr className="nohover">
                <td colSpan={9} className="muted center">
                  표시할 종목이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ---- one method ---------------------------------------------------------------------------------------------------------------

const exitText = (st: PaperExitState | undefined, zero: string): { text: string; pending: boolean } | null => {
  if (!st) return null;
  const v = (k: 'tp' | 'sl' | 'horizon') => st[k].running ?? st[k].effective;
  const f = (k: 'tp' | 'sl' | 'horizon', pre: string, unit: string) => (v(k) > 0 ? `${pre}${v(k)}${unit}` : zero);
  return { text: `익절 ${f('tp', '+', '%')} · 손절 ${f('sl', '-', '%')} · 보유 ${f('horizon', '', '분')}`, pending: st.tp.pending || st.sl.pending || st.horizon.pending };
};

function MethodDetail(props: {
  method: TradeStrategy;
  rows: MethodRow[];
  totals: MethodTotals;
  filter: MethodRowFilter;
  setFilter: (f: MethodRowFilter) => void;
  cols: ColWidths;
  fillCols: ColWidths;
  liveNow: boolean;
  connected: boolean;
  busy: boolean;
  botRunning: boolean | null;
  hostKey: string;
  lastTs: number;
  side: SideData | null;
  go: Go;
  inSnap: (code: string) => boolean;
  onSaved: () => void;
}) {
  const { method, rows, totals, cols, fillCols, liveNow } = props;
  const dayKo = useContext(DayCtx);
  const snap = useStore((s) => s.snapshot);
  const control = useStore((s) => s.control);
  const conditions = useStore((s) => s.conditions);
  const selected = useStore((s) => s.selected);
  const openResults = useStore((s) => s.openResults);
  const side: SideStrategy | null = method === 'rule' ? null : method;
  const mode = useMemo(() => strategyModes(control)[method], [control, method]);
  const shown = useMemo(() => filterMethodRows(rows, props.filter), [rows, props.filter]);
  const fills = useMemo(() => (snap?.trades || []).filter((t) => strategyOfTrade(t) === method).sort((a, b) => b.ts - a.ts), [snap, method]);
  const daySum = useMemo(() => strategySummary(snap?.trades || [])[method], [snap, method]);

  const blockNote = (() => {
    if (!side) return null;
    const notes = (snap?.strategyNotes || []).filter((n) => n.strategy === side && n.mode === 'live');
    const last = notes[notes.length - 1];
    const ref = snap?.lastLineTs || 0;
    return last && last.kind === 'blocked' && ref - last.lastTs <= BLOCK_NOTE_FRESH_MS ? last : null;
  })();
  // the bot's fill ledger is its state NOW: shown on the live day only (not beside a past day's numbers)
  const liveSum = !liveNow ? null : method === 'edge' ? parseEdgeLiveSummary(control.status) : method === 'vwap' ? parseVwapLiveSummary(control.status) : null;
  const limits = method === 'edge' ? control.edgeLiveLimits : method === 'vwap' ? control.vwapLiveLimits : undefined;
  const maxEntries = limits ? limits.entries.running ?? limits.entries.effective : null;
  const exits = exitText(method === 'rule' ? control.ruleExit : method === 'edge' ? control.paperExit : control.vwapExit, method === 'rule' ? '적용 안 함' : method === 'edge' ? '모델 기준' : '-');
  const traceNote =
    side && liveNow && (mode.live || mode.paper) && props.side?.evals && props.side.evals.total === 0
      ? '오늘 봉별 판정 기록이 없습니다 (2026-11-27 이후 버전의 r003 이 장중에 판정한 1분봉부터 쌓입니다).'
      : '';

  // ---- target symbols of this method ----
  const tg = control.targets?.[method];
  const inUse = targetsInUse(tg, props.botRunning);
  const [draft, setDraft] = useState<Set<string> | null>(null);
  const [tgMsg, setTgMsg] = useState('');
  const [saving, setSaving] = useState(false);
  const methodRef = useRef(method);
  methodRef.current = method;
  const canEdit = liveNow && props.connected && !props.busy && !saving && !!tg;
  useEffect(() => {
    setDraft(null);
    setTgMsg('');
  }, [method]);
  // [Codex review] a draft does not outlive the live day: on a past day / replay / lost connection it is dropped
  useEffect(() => {
    if (!liveNow || !props.connected) setDraft(null);
  }, [liveNow, props.connected]);
  const watchedCodes = useMemo(() => rows.filter((r) => r.watched || r.collector).map((r) => r.code), [rows]);
  const startEdit = () => setDraft(new Set(tg && tg.effective.length ? tg.effective : watchedCodes));
  const toggle = (code: string) =>
    setDraft((d) => {
      const n = new Set(d || []);
      if (n.has(code)) n.delete(code);
      else n.add(code);
      return n;
    });
  const draftOutside = draft ? [...draft].filter((c) => !watchedCodes.includes(c)).sort() : [];
  const saveTargets = async () => {
    if (!draft || !tg || !canEdit) return;
    const forMethod = method;
    const list = targetListToStore(draft, watchedCodes);
    if (list.join(',') === tg.effective.join(',')) return setDraft(null);
    const text = list.length
      ? `${TRADE_STRATEGY_KO[method]}의 대상 종목을 ${list.length}종목으로 지정합니다.\n\n${list.join(', ')}\n\n이 방식은 지정한 종목만 새로 삽니다. 이미 산 종목의 매도 · 청산은 그대로 계속됩니다.` +
        (method === 'rule' ? '\n규칙 매매 대상에서 빠진 종목도 Edge AI / VWAP 매매가 켜져 있으면 그 방식이 살 수 있습니다.' : '') +
        (method === 'edge' ? '\nEdge AI 가상 매매의 수집기 종목도 지정한 종목만 새로 삽니다.' : '')
      : `${TRADE_STRATEGY_KO[method]}의 대상 종목 지정을 해제합니다 (전체 = 지금까지처럼 감시 종목 전부).\n\n모두 선택했거나 모두 해제하면 "전체"로 저장됩니다.`;
    if (!window.confirm(`${text}\n\nr003 재시작 후부터 적용됩니다. 저장할까요?`)) return;
    const key = METHOD_TARGET_KEYS[forMethod];
    setSaving(true);
    let r;
    try {
      r = await window.api.ctlSet({ [key]: list });
    } finally {
      setSaving(false);
    }
    if (methodRef.current !== forMethod) return; // the answer of another method's save: this page's draft is not touched
    const rej = r.data?.rejected?.[key];
    if (!r.ok || rej) return setTgMsg(`저장하지 못했습니다: ${rej || r.error || '알 수 없는 오류'}`);
    setTgMsg('저장했습니다 (r003 재시작 후 적용)');
    setDraft(null);
    props.onSaved();
  };
  const th = (i: number, label: string, cls = '', title?: string) => (
    <th className={cls} title={title}>
      {label}
      <ColGrip cols={cols} i={i} />
    </th>
  );
  const buyConds = method === 'rule' ? [...(conditions?.buy || [])].sort((a, b) => a.order - b.order) : [];
  const condFromRunning = buyConds.some((c) => c.apply_running !== undefined && c.apply_running !== null);

  return (
    <div className="method-split">
      <aside className={`method-side m-${method}`}>
        <section>
          <h5>기본 정보</h5>
          <dl>
            <dt>방식</dt>
            <dd>
              <b>{TRADE_STRATEGY_KO[method]}</b> <span className={`tl-mode ${mode.live ? 'live' : mode.paper ? 'paper' : 'off'}`}>{strategyModeKo(mode)}</span>
            </dd>
            <dt>기준</dt>
            <dd>
              {mode.fromRunning ? '실행 중인 봇' : '저장된 설정 (봇 꺼짐)'}
              {mode.pending ? <span className="badge-warn"> 저장값 다름 → 재시작 후 적용</span> : null}
            </dd>
            <dt>대상 종목</dt>
            <dd>
              {tg ? (
                <>
                  {inUse.from === 'running' ? `실행 중: ${targetsKo(inUse.list)}` : inUse.from === 'unknown' ? `실행 중: 확인 불가 · 저장: ${targetsKo(tg.effective)}` : `저장: ${targetsKo(inUse.list)}`}
                  {tg.pending && inUse.from === 'running' ? <span className="badge-warn"> 저장: {targetsKo(tg.effective)} → 재시작 후 적용</span> : null}
                </>
              ) : (
                <span className="muted">확인 불가 (서버가 대상 종목 설정을 모름)</span>
              )}
            </dd>
            <dt>마지막 기록</dt>
            <dd className="mono">{props.lastTs ? fmtKstTime(props.lastTs) : '-'}</dd>
            {side && (
              <>
                <dt>봉별 판정</dt>
                <dd>{props.side?.evals ? `${fmtInt(props.side.evals.total)}행 (${dayKo})` : '-'}</dd>
              </>
            )}
            {method === 'edge' && props.side?.pst?.modelId && (
              <>
                <dt>모델</dt>
                <dd className="mono" title="가상 러너 상태 파일의 모델 · 기준 확률">
                  {props.side.pst.modelId}
                  {props.side.pst.threshold !== undefined ? ` · 기준 ${props.side.pst.threshold.toFixed(3)}` : ''}
                </dd>
              </>
            )}
          </dl>
        </section>

        <section>
          <h5>
            매수 조건 <i>{method === 'rule' ? (condFromRunning ? '실행 중인 봇 기준' : '저장된 설정 기준') : '판정 단계 (위에서 아래 순서)'}</i>
          </h5>
          {method === 'rule' ? (
            buyConds.length ? (
              <ul className="cond-list">
                {buyConds.map((c) => {
                  const ap = c.apply_running ?? c.apply;
                  return (
                    <li key={c.no} className={ap === 'off' ? 'off' : ap === 'record' ? 'rec' : ''} title={c.description}>
                      <span className="mono">{String(c.no).padStart(3, '0')}</span> {c.title}
                      {ap === 'off' ? <i>끔</i> : ap === 'record' ? <i>기록만</i> : null}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="muted small">조건 목록을 읽지 못했습니다.</p>
            )
          ) : (
            <ul className="cond-list">
              {[...STAGE_ROWS[side as SideStrategy]]
                .reverse()
                .filter((r) => r.key !== 'held')
                .map((r) => (
                  <li key={r.key}>{r.label}</li>
                ))}
            </ul>
          )}
          <p className="muted small">종목별로 어디서 멈췄는지는 오른쪽 표의 "지금 상태"와 타임라인에서 봅니다. 세부 값은 세부 설정 창.</p>
        </section>

        <section>
          <h5>
            매도 · 청산 <i>실행 값 (없으면 저장값)</i>
          </h5>
          <p>
            {exits ? exits.text : <span className="muted">확인 불가</span>}
            {exits?.pending ? <span className="badge-warn"> 재시작 후 적용</span> : null}
          </p>
          {method === 'rule' && conditions?.sell?.length ? <p className="muted small">매도 조건 {conditions.sell.length}개 (규칙 파이프라인 · 세부 설정에서 확인)</p> : null}
          {side && <p className="muted small">남으면 15:20 장 마감 청산 (봇 규칙)</p>}
        </section>

        {side && (
          <section>
            <h5>
              매매 제한 <i>실제 주문 한도</i>
            </h5>
            {limits ? (
              <dl>
                {EDGE_LIVE_LIMIT_KINDS.map((k) => (
                  <span key={k} className="dl-row">
                    <dt>{EDGE_LIVE_LIMIT_KO[k]}</dt>
                    <dd>
                      {k === 'entries' && liveSum ? `${liveSum.entries} / ` : ''}
                      {fmtInt(limits[k].running ?? limits[k].effective)}
                      {EDGE_LIVE_LIMIT_UNIT[k]}
                      {limits[k].pending ? <span className="badge-warn"> 대기</span> : null}
                    </dd>
                  </span>
                ))}
              </dl>
            ) : (
              <p className="muted small">확인 불가</p>
            )}
          </section>
        )}

        <section>
          <h5>{dayKo} 성과</h5>
          <dl>
            <dt>실전</dt>
            <dd>
              매수 {daySum.buys} · 매도 {daySum.sells}
              {daySum.sells ? ` · ${daySum.wins}승 ${daySum.sells - daySum.wins}패` : ''}
            </dd>
            <dt>실현 손익</dt>
            <dd className={pnlClass(daySum.sells ? daySum.pnlKrw : undefined)}>{daySum.sells ? fmtKrw(daySum.pnlKrw) : '-'}</dd>
            {side && (
              <>
                <dt>신호 · 주문</dt>
                <dd>
                  {totals.signals} · {totals.orders}
                </dd>
                <dt>가상</dt>
                <dd>
                  매수 {totals.paperBuys} · 청산 {totals.paperSells} · 보유 {totals.paperHolding}
                </dd>
              </>
            )}
          </dl>
          <div className="chips">
            <button type="button" className="chip link" onClick={() => (method === 'rule' ? window.dispatchEvent(new Event('rule-settings')) : window.dispatchEvent(new CustomEvent('strategy-settings', { detail: method })))}>
              세부 설정 →
            </button>
            <button type="button" className="chip link" onClick={() => openResults(method, 'live')}>
              실제 결과 →
            </button>
            {side && (
              <button type="button" className="chip link" onClick={() => openResults(side, 'paper')}>
                가상 결과 →
              </button>
            )}
          </div>
        </section>
      </aside>

      <div className="method-main">
        {(blockNote || liveSum || traceNote) && (
          <div className="method-notes">
            {blockNote && (
              <span className="badge-warn" title={`봇 로그 그대로: "신규 매수 중단 | ${blockNote.text}" - 처음 ${fmtKstTime(blockNote.ts)}, 마지막 ${fmtKstTime(blockNote.lastTs)} 기록`}>
                봇 기록 {fmtKstTime(blockNote.ts).slice(0, 5)}~{fmtKstTime(blockNote.lastTs).slice(0, 5)} 신규 매수 중단: {blockReasonKo(blockNote.text)}
              </span>
            )}
            {liveSum && (
              <span className={liveSum.unpricedSells > 0 ? 'badge-warn' : 'muted small'} title="봇의 체결 장부 기준 (직접 매도한 전략 보유분은 들어가지 않습니다)">
                체결 장부: {edgeLiveSummaryText(liveSum, maxEntries)}
              </span>
            )}
            {traceNote && <span className="muted small">{traceNote}</span>}
          </div>
        )}

        <TargetCompose method={method} enabled={liveNow && props.connected} hostKey={props.hostKey} onApplied={props.onSaved} />

        <div className="toolbar method-bar">
          <h4>감시 · 기록 종목</h4>
          <div className="seg">
            {FILTERS.map((f) => (
              <button key={f} type="button" className={props.filter === f ? 'on' : ''} onClick={() => props.setFilter(f)}>
                {METHOD_FILTER_KO[f]} {filterMethodRows(rows, f).length}
              </button>
            ))}
          </div>
          {draft ? (
            <>
              <button type="button" onClick={() => setDraft(new Set([...watchedCodes, ...draftOutside]))}>
                전체 선택
              </button>
              <button type="button" onClick={() => setDraft(new Set())}>
                전체 해제
              </button>
              <button type="button" className="primary" disabled={!canEdit} onClick={() => void saveTargets()}>
                대상 저장 ({targetListToStore(draft, watchedCodes).length || '전체'})
              </button>
              <button type="button" onClick={() => setDraft(null)}>
                취소
              </button>
            </>
          ) : (
            <button type="button" disabled={!canEdit} title={tg ? '이 방식이 새로 살 종목을 고릅니다 (저장되는 설정, r003 재시작 후 적용)' : '서버가 대상 종목 설정을 모릅니다 (봇 코드 갱신 필요)'} onClick={startEdit}>
              대상 종목 지정
            </button>
          )}
          {tgMsg && <span className="muted small">{tgMsg}</span>}
        </div>
        {draft ? (
          <p className="muted small method-line">
            "구분" 칸의 체크로 고릅니다. 모두 선택하거나 모두 해제하면 "전체(지정 없음)"로 저장됩니다. 감시목록(r004)에 없는 종목은 지정해도 봇이 보지 않습니다.
            {draftOutside.length ? (
              <>
                {' '}
                감시 종목이 아닌 저장 코드:{' '}
                {draftOutside.map((c) => (
                  <button key={c} type="button" className="chip" title="누르면 목록에서 뺍니다" onClick={() => toggle(c)}>
                    {c} ×
                  </button>
                ))}
              </>
            ) : null}
          </p>
        ) : (
          <p className="muted small method-line">
            {TARGET_NOTE[method]} 행을 누르면 타임라인({TRADE_STRATEGY_KO[method]} 탭).
          </p>
        )}
        <div className="table-wrap">
          <ColReset cols={cols} />
          <table {...colTable(cols, 'grid compact method-grid')}>
            <ColGroup cols={cols} />
            <thead>
              <tr>
                {th(0, '코드')}
                {th(1, '종목명')}
                {th(2, '현재가 (전일 대비)', 'num')}
                {th(3, '구분', '', '감시 = r003 이 지금 감시 · 수집기 = 틱 수집기 종목(Edge AI 가상) · 기록 = 감시 대상은 아니지만 보유 / 체결 / 기록 있음 · 대상 아님 = 대상 종목 지정에 없음')}
                {th(4, '지금 상태 (봇의 마지막 기록)', '', '이 방식에 대해 봇이 마지막으로 남긴 기록입니다. 앱이 판정하지 않습니다')}
                {th(5, '기록 시각')}
                {th(6, method === 'rule' ? '평가 수 · 매수 체결' : '판정 봉 · 신호 · 주문', 'num')}
                {th(7, '보유', 'num', '실제 보유: 수량 @ 원가 · 가상 보유는 "가상"')}
                {th(8, `${dayKo} 매수 / 매도`, 'num')}
                {th(9, '실현 손익', 'num', '실제 매매의 실현 손익 · 가상 매매는 순수익률 합')}
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr
                  key={r.code}
                  className={`${r.code === selected ? 'sel' : ''}${r.holdQty || r.paperEntry !== undefined ? ' row-held' : ''}${r.notTarget ? ' not-target' : ''}`}
                  onClick={() => (draft ? undefined : props.go(r.code, method))}
                  onDoubleClick={() => showCode(r.code)}
                  title={draft ? undefined : props.inSnap(r.code) ? '누르면 타임라인 · 두 번 누르면 다른 창에 종목 표시' : '봇의 오늘 로그에 없는 종목이라 타임라인이 없습니다. 누르면 다른 창에 종목을 표시합니다'}
                >
                  <td className="mono">{r.code}</td>
                  <td title={r.name}>
                    {r.name} {r.nxt && <NxtMark />}
                  </td>
                  <td className="num">
                    <PriceTag code={r.code} />
                  </td>
                  <td>
                    {draft && (r.watched || r.collector) ? (
                      <label className="check" onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" checked={draft.has(r.code)} onChange={() => toggle(r.code)} /> 대상
                      </label>
                    ) : (
                      <>
                        {r.watched && <span className="tag">감시</span>}
                        {r.collector && <span className="tag">수집기</span>}
                        {r.extra && <span className="tag lock">기록</span>}
                        {r.notTarget && <span className="tag lock">대상 아님</span>}
                        {r.holdQty ? <span className="tag apply-on">보유</span> : null}
                        {r.paperEntry !== undefined && <span className="tag apply-record">가상 보유</span>}
                      </>
                    )}
                  </td>
                  <td className={`reason res-${r.stateKind === 'none' || r.stateKind === 'info' ? '' : r.stateKind}`} title={r.stateRaw ? `${METHOD_SRC_KO[r.stateSrc]}: ${r.state}\n${r.stateRaw}` : r.state}>
                    {r.stateSrc && r.stateSrc !== 'rule' && (
                      <i className="src">
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
          <h4>{dayKo} 실제 체결 ({TRADE_STRATEGY_KO[method]})</h4>
          <span className="muted small">
            최근 {Math.min(fills.length, 8)}건 / 전체 {fills.length}건
          </span>
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
                <tr key={`${t.code}-${t.ts}-${i}`} onClick={() => props.go(t.code, method)}>
                  <td className="mono">{fmtKstTime(t.ts)}</td>
                  <td>
                    <span className="mono">{t.code}</span> {t.name}
                  </td>
                  <td className={t.side === 'BUY' ? 'gain' : 'loss'}>{t.hand ? '직접 매도' : t.side === 'BUY' ? '매수' : '매도'}</td>
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
                    {dayKo} 이 방식의 실제 체결이 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
