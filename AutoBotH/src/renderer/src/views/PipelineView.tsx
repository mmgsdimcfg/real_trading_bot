/*
 * Update log
 * - [2026-11-04] type=feat - Pipeline diagram (main view). Buy conditions in evaluation order (Condition.order) laid out top-down
 *   with dagre, stage bands behind them, _008 expanded into its sub-steps. No symbol selected: funnel - node width and bar follow
 *   today's block count, badges show blocked/passed polls. Symbol selected: passed nodes green, the stopping node red (or amber for
 *   a wait), unreached nodes gray, STEPS diagnostics as dashed outlines. Sell tab maps SELL TRIGGER / SELL HOLD the same way
 * - [2026-11-05] type=feat - Remote control (guide 2.2): the side panel's 설정값 table is an editor (ValuesEditor, values_meta from
 *   `r016 conditions`) with 저장/취소 - only the changed keys go to the server in one `set`. Each node header mirrors its first
 *   ENABLE_* constant as an ON/OFF toggle (edits the panel draft of that node, not saved until 저장), a yellow "미적용" dot when a
 *   saved override is not yet in the running bot, and a ◆ mark while an override is active. Unsaved edits pin the focused node;
 *   switching nodes asks before discarding them
 * - [2026-11-08] type=fix - Trigger / STEPS / silent / info-only keys follow the server's numbering (mapping keyRemapOf)
 */

import { createContext, memo, useContext, useEffect, useMemo, useState } from 'react';
import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import dagre from '@dagrejs/dagre';
import { useStore } from '../store';
import { SymbolPicker } from '../components/SymbolPicker';
import { condIndex, editItemsFor, fmtInt, fmtKrw, nodeEnableKey, PSEUDO_LABEL, sortedConds, STAGE_LABEL } from '../lib';
import { EditorActions, useValueEditing, ValuesEditor } from '../components/ValuesEditor';
import { savedValue } from '../../../shared/control';
import { INFO_ONLY_BUY, SILENT_BUY, STEP_TO_COND, TRIGGER_COND, TRIGGER_SUBS, TRIGGER_SUB_LABEL, condKeyOf, keyRemapOf, remapKey } from '../../../shared/mapping';
import type { Condition, EvalEvent, Side, Snapshot, SymbolSummary } from '../../../shared/types';
import { fmtKstTime } from '../../../shared/time';

type NState = 'pass' | 'fail' | 'hold' | 'idle' | 'none';

interface CondData extends Record<string, unknown> {
  key: string;
  /** the entry trigger node (has the sub-step handle on the right) */
  trigger?: boolean;
  title: string;
  state: NState;
  badge?: string;
  bar?: number;
  dashed?: boolean;
  note?: string;
  diag?: 'P' | 'F';
  top?: boolean;
  focus?: boolean;
  width: number;
  sub?: boolean;
  /** first ENABLE_* constant of the condition (header toggle) */
  enableKey?: string;
  enableOn?: boolean;
  enableDirty?: boolean;
  /** an override is saved but the running bot has not picked it up */
  pendingMark?: boolean;
  /** an override is active for one of the values */
  ovr?: boolean;
}

/** node-header toggle -> PipelineView (keeps the memoized node free of callbacks in data) */
const ToggleCtx = createContext<{ toggle: (node: string, key: string) => void; disabled: boolean }>({ toggle: () => undefined, disabled: true });

interface NodeExt {
  enableKey?: string;
  enableOn?: boolean;
  enableDirty?: boolean;
  pendingMark?: boolean;
  ovr?: boolean;
}

interface BandData extends Record<string, unknown> {
  label: string;
  stage: string;
  width: number;
  height: number;
}

const ICON: Record<NState, string> = { pass: '✓', fail: '✕', hold: '…', idle: '·', none: '' };

const CondNode = memo(function CondNode({ id, data }: NodeProps<Node<CondData>>) {
  const tg = useContext(ToggleCtx);
  return (
    <div
      className={`cnode st-${data.state} ${data.dashed ? 'dashed' : ''} ${data.diag ? 'diag-' + data.diag : ''} ${data.focus ? 'focus' : ''} ${data.sub ? 'sub' : ''}`}
      style={{ width: data.width }}
    >
      {!data.sub && <Handle type="target" position={Position.Top} />}
      {data.sub && <Handle type="target" position={Position.Left} />}
      <div className="cnode-head">
        <span className="cnode-key">{data.key.startsWith('_') && !data.sub ? data.key : ''}</span>
        <span className="cnode-title">{data.title}</span>
        {data.pendingMark && <i className="pend-dot node" title="저장한 설정값이 실행 중인 봇에 아직 반영되지 않음 (재시작 필요)" />}
        {data.ovr && !data.pendingMark && (
          <i className="ovr-mark" title="오버라이드된 설정값 적용 중">
            ◆
          </i>
        )}
        {data.state !== 'none' && <span className="cnode-icon">{ICON[data.state]}</span>}
        {data.enableKey && (
          <button
            className={`ntoggle nodrag ${data.enableOn ? 'on' : 'off'} ${data.enableDirty ? 'dirty' : ''}`}
            disabled={tg.disabled}
            title={`${data.enableKey} = ${data.enableOn ? 'true' : 'false'}${data.enableDirty ? ' (저장 전)' : ''} - 누르면 바꾸고 오른쪽 패널에서 저장`}
            onClick={(e) => {
              e.stopPropagation();
              tg.toggle(id, data.enableKey!);
            }}
          >
            {data.enableOn ? 'ON' : 'OFF'}
          </button>
        )}
      </div>
      {(data.badge || data.note) && (
        <div className="cnode-badge">
          {data.top && <b className="top">최다</b>}
          {data.badge}
          {data.note && <span className="note">{data.note}</span>}
        </div>
      )}
      {data.bar !== undefined && data.bar > 0 && (
        <div className="cnode-bar">
          <i style={{ width: `${Math.max(2, data.bar * 100)}%` }} />
        </div>
      )}
      {!data.sub && <Handle type="source" position={Position.Bottom} />}
      {data.trigger && <Handle type="source" id="r" position={Position.Right} />}
    </div>
  );
});

const BandNode = memo(function BandNode({ data }: NodeProps<Node<BandData>>) {
  return (
    <div className={`band band-${data.stage}`} style={{ width: data.width, height: data.height }}>
      <span>{data.label}</span>
    </div>
  );
});

const nodeTypes = { cond: CondNode, band: BandNode };

const NODE_H = 58;
const SUB_H = 40;
const BASE_W = 250;
const EXTRA_W = 190;

interface ChainItem {
  key: string;
  stage: string;
  cond?: Condition;
}

function chainFor(conds: Condition[], side: Side): ChainItem[] {
  const items: ChainItem[] = conds.map((c) => ({ key: condKeyOf(c.no), stage: c.stage, cond: c }));
  if (side === 'buy') return [{ key: 'PREP', stage: 'PREP' }, ...items, { key: 'ORDER', stage: 'ORDER' }];
  return [...items, { key: 'SESSION_CLOSE', stage: 'ORDER' }, { key: 'ORDER', stage: 'ORDER' }];
}

/** state of each chain node for a selected symbol */
function buyStates(sym: SymbolSummary, keys: string[]): Map<string, NState> {
  const m = new Map<string, NState>();
  const ev = sym.lastEval;
  keys.forEach((k) => m.set(k, 'idle'));
  if (!ev) return m;
  const stopAt = ev.condKey === 'ORDER' ? keys.length - 1 : keys.indexOf(ev.condKey);
  keys.forEach((k, i) => {
    if (stopAt < 0) return;
    if (i < stopAt) m.set(k, 'pass');
    else if (i === stopAt) m.set(k, ev.condKey === 'ORDER' ? 'pass' : ev.result === 'hold' ? 'hold' : 'fail');
  });
  return m;
}

function sellStates(sym: SymbolSummary, keys: string[]): Map<string, NState> {
  const m = new Map<string, NState>();
  keys.forEach((k) => m.set(k, 'idle'));
  const ev = sym.lastSell;
  if (ev) m.set(ev.condKey, ev.result === 'hold' ? 'hold' : 'fail');
  if (sym.lastOrder && sym.lastOrder.side === 'sell') m.set('ORDER', sym.lastOrder.result === 'pass' ? 'pass' : 'hold');
  return m;
}

function build(snap: Snapshot | null, conds: Condition[], side: Side, sym: SymbolSummary | undefined, focus: string | null, ext: Map<string, NodeExt>) {
  const chain = chainFor(conds, side);
  const keys = chain.map((c) => c.key);
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'TB', nodesep: 20, ranksep: 22, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));

  const buyStats = snap?.buyStats || {};
  const sellStats = snap?.sellStats || {};
  const countOf = (k: string) => (side === 'buy' ? buyStats[k]?.block || 0 : (sellStats[k]?.trigger || 0) + (sellStats[k]?.hold || 0));
  const max = Math.max(1, ...keys.map(countOf));
  const topKey = keys.reduce((a, k) => (countOf(k) > countOf(a) ? k : a), keys[0]);
  const states = sym ? (side === 'buy' ? buyStates(sym, keys) : sellStates(sym, keys)) : null;

  // app keys of the trigger / STEPS / silent / info-only conditions in the server's numbering (mapping keyRemap)
  const rm = keyRemapOf(conds, side);
  const trig = remapKey(rm, TRIGGER_COND);
  const infoOnly = new Set([...INFO_ONLY_BUY].map((k) => remapKey(rm, k)));
  const silent = new Set([...SILENT_BUY].map((k) => remapKey(rm, k)));
  const data = new Map<string, CondData>();
  for (const it of chain) {
    const k = it.key;
    const title = it.cond?.title || PSEUDO_LABEL[k] || k;
    const d: CondData = { key: k, title, state: 'none', width: BASE_W, focus: focus === k, trigger: side === 'buy' && k === trig, ...ext.get(k) };
    if (!sym) {
      const c = countOf(k);
      d.bar = c / max;
      d.width = BASE_W + EXTRA_W * d.bar;
      d.top = c > 0 && k === topKey;
      if (side === 'buy') {
        const st = buyStats[k];
        if (k === 'ORDER') d.badge = `전체 통과 ${fmtInt(snap?.symbols.reduce((a, s) => a + (s.furthestOrderToday >= conds.length ? 1 : 0), 0))}종목`;
        else d.badge = `막힘 ${fmtInt(st?.block || 0)} · 통과 ${fmtInt(st?.pass || 0)}${st?.hold ? ` · 대기 ${fmtInt(st.hold)}` : ''}${st?.preOpenBlock ? ` · 개장 전 ${fmtInt(st.preOpenBlock)}` : ''}`;
      } else {
        const st = sellStats[k];
        d.badge = k === 'ORDER' ? `매도 ${snap?.trades.filter((t) => t.side === 'SELL').length || 0}건` : `발동 ${st?.trigger || 0} · 보류 ${st?.hold || 0}`;
      }
    } else {
      d.state = states!.get(k) || 'idle';
      if (side === 'buy') {
        const n = sym.blockCounts[k];
        if (n) d.badge = `오늘 막힘 ${fmtInt(n)}`;
        if (sym.lastEval?.condKey === k || (k === 'ORDER' && sym.lastEval?.condKey === 'ORDER')) d.note = sym.lastEval.reason;
        if (k === 'ORDER' && sym.lastOrder && sym.lastOrder.side === 'buy') d.note = `${sym.lastOrder.tag} ${fmtKstTime(sym.lastOrder.ts)}`;
        const idx = keys.indexOf(k);
        if (sym.furthestOrderToday >= 0 && idx - 1 === sym.furthestOrderToday && k !== 'ORDER') d.note = (d.note ? d.note + ' · ' : '') + '오늘 여기까지 도달';
      } else {
        const n = sym.sellCounts[k];
        if (n) d.badge = `발동/보류 ${n}`;
        if (sym.lastSell?.condKey === k) d.note = sym.lastSell.reason;
      }
    }
    if (side === 'buy' && infoOnly.has(k)) {
      d.dashed = true;
      d.note = d.note || '반려 없음 (점수 정보)';
    }
    if (side === 'buy' && silent.has(k)) {
      d.dashed = true;
      d.note = d.note || '로그 없음 (silent)';
    }
    data.set(k, d);
    g.setNode(k, { width: d.width, height: NODE_H });
  }
  // STEPS diagnostics of the last rejected poll
  if (sym && side === 'buy' && sym.lastEval?.steps) {
    for (const [step, v] of Object.entries(sym.lastEval.steps)) {
      const k = STEP_TO_COND[step] && remapKey(rm, STEP_TO_COND[step]);
      const d = k && data.get(k);
      if (d && v !== '-' && d.state === 'idle') d.diag = v;
    }
  }
  for (let i = 1; i < keys.length; i++) g.setEdge(keys[i - 1], keys[i]);
  dagre.layout(g);

  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const pos = new Map<string, { x: number; y: number }>();
  for (const k of keys) {
    const n = g.node(k);
    pos.set(k, { x: n.x, y: n.y });
    const d = data.get(k)!;
    nodes.push({ id: k, type: 'cond', position: { x: n.x - d.width / 2, y: n.y - NODE_H / 2 }, data: d, zIndex: 2 });
  }
  for (let i = 1; i < keys.length; i++) {
    const a = data.get(keys[i - 1])!;
    const b = data.get(keys[i])!;
    const on = a.state === 'pass' && (b.state === 'pass' || b.state === 'fail' || b.state === 'hold');
    edges.push({ id: `e${i}`, source: keys[i - 1], target: keys[i], className: on ? 'edge-on' : 'edge-off' });
  }

  const colRight = BASE_W / 2 + EXTRA_W / 2 + 60;
  // trigger (TRIGGER_COND in server numbering) sub-steps to the right of the trigger node
  if (side === 'buy' && pos.has(trig)) {
    const p = pos.get(trig)!;
    const subs = TRIGGER_SUBS.filter((s) => s !== 'legacy' || (buyStats[`${trig}/legacy`]?.block || 0) > 0);
    const subMax = Math.max(1, ...subs.map((s) => buyStats[`${trig}/${s}`]?.block || 0));
    subs.forEach((s, i) => {
      const id = `${trig}/${s}`;
      const st = buyStats[id];
      let state: NState = 'none';
      if (sym) {
        const ev = sym.lastEval;
        state = ev?.condKey === trig ? (ev.sub === s ? 'fail' : 'idle') : data.get(trig)!.state === 'pass' ? 'pass' : 'idle';
      }
      const d: CondData = {
        key: id,
        title: TRIGGER_SUB_LABEL[s],
        state,
        sub: true,
        width: 210,
        badge: sym ? undefined : `막힘 ${fmtInt(st?.block || 0)}${st?.preOpenBlock ? ` · 개장 전 ${fmtInt(st.preOpenBlock)}` : ''}`,
        bar: sym ? undefined : (st?.block || 0) / subMax,
        focus: focus === id,
      };
      const y = p.y - ((subs.length - 1) * (SUB_H + 8)) / 2 + i * (SUB_H + 8) - SUB_H / 2;
      nodes.push({ id, type: 'cond', position: { x: p.x + colRight, y }, data: d, zIndex: 2 });
      edges.push({ id: `es${i}`, source: trig, sourceHandle: 'r', target: id, className: 'edge-sub' });
    });
  }

  // stage bands
  const width = colRight + 210 + BASE_W / 2 + EXTRA_W / 2 + 140;
  const left = Math.min(...[...pos.values()].map((p) => p.x)) - BASE_W / 2 - EXTRA_W / 2 - 120;
  let i = 0;
  while (i < chain.length) {
    const stage = chain[i].stage;
    let j = i;
    while (j + 1 < chain.length && chain[j + 1].stage === stage) j++;
    const y0 = pos.get(chain[i].key)!.y - NODE_H / 2 - 9;
    const y1 = pos.get(chain[j].key)!.y + NODE_H / 2 + 9;
    nodes.push({
      id: `band-${i}`,
      type: 'band',
      position: { x: left, y: y0 },
      data: { label: STAGE_LABEL[stage] || stage, stage, width, height: y1 - y0 },
      selectable: false,
      draggable: false,
      zIndex: 0,
    });
    i = j + 1;
  }
  return { nodes, edges };
}

function lastEventAt(events: EvalEvent[] | undefined, key: string, side: Side): EvalEvent | undefined {
  if (!events) return undefined;
  const [k, sub] = key.split('/');
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.side === side && e.condKey === k && (!sub || e.sub === sub)) return e;
  }
  return undefined;
}

export function PipelineView() {
  const snap = useStore((s) => s.snapshot);
  const conditions = useStore((s) => s.conditions);
  const side = useStore((s) => s.side);
  const setSide = useStore((s) => s.setSide);
  const selected = useStore((s) => s.selected);
  const timeline = useStore((s) => s.timeline);
  const connected = useStore((s) => s.status.conn === 'connected');
  const overrideErrors = conditions?.override_errors || [];
  const [focus, setFocus] = useState<string | null>(null);

  const conds = useMemo(() => sortedConds(conditions, side), [conditions, side]);
  const sym = snap?.symbols.find((s) => s.code === selected);

  const focusKey = focus || (sym ? (side === 'buy' ? sym.lastEval?.condKey : sym.lastSell?.condKey) : null) || null;
  const condKey = focusKey ? focusKey.split('/')[0] : null;
  const cond = condKey ? condIndex(conditions, side).get(condKey) : undefined;
  const items = useMemo(() => editItemsFor(cond), [cond]);
  const ed = useValueEditing(items);
  const canEdit = connected && conditions?.source === 'server';

  // unsaved edits pin the node (a new snapshot must not move the panel to another condition)
  useEffect(() => {
    if (ed.dirty && !focus && focusKey) setFocus(focusKey);
  }, [ed.dirty, focus, focusKey]);

  const changeFocus = (next: string | null): boolean => {
    if (next === focusKey) return true;
    if (ed.dirty && !window.confirm('저장하지 않은 설정값 변경이 있습니다. 버리고 이동할까요?')) return false;
    ed.cancel();
    setFocus(next);
    return true;
  };

  // header toggles + marks of every node
  const ext = useMemo(() => {
    const m = new Map<string, NodeExt>();
    for (const c of conds) {
      const k = condKeyOf(c.no);
      const meta = Object.values(c.values_meta || {});
      const en = nodeEnableKey(c);
      const e: NodeExt = { pendingMark: meta.some((v) => v.pending), ovr: meta.some((v) => v.has_override) };
      if (en) {
        const saved = savedValue(en) === true;
        const on = k === condKey && en.key in ed.drafts ? ed.drafts[en.key] === true : saved;
        Object.assign(e, { enableKey: en.key, enableOn: on, enableDirty: on !== saved });
      }
      m.set(k, e);
    }
    return m;
  }, [conds, condKey, ed.drafts]);

  const toggleCtx = {
    disabled: !canEdit,
    toggle: (node: string, key: string) => {
      const target = conds.find((c) => condKeyOf(c.no) === node);
      const it = editItemsFor(target).find((i) => i.key === key);
      if (!it) return;
      const isCurrent = node === condKey;
      if (!isCurrent && !changeFocus(node)) return;
      const cur = isCurrent && key in ed.drafts ? ed.drafts[key] === true : savedValue(it) === true;
      ed.setDraft(key, !cur);
    },
  };

  const { nodes, edges } = useMemo(() => build(snap, conds, side, sym, focus, ext), [snap, conds, side, sym, focus, ext]);
  const ev = focusKey && sym ? lastEventAt(timeline?.code === sym.code ? timeline.events : undefined, focusKey, side) : undefined;
  const stat = focusKey ? (side === 'buy' ? snap?.buyStats[focusKey] : undefined) : undefined;
  const sstat = focusKey && side === 'sell' ? snap?.sellStats[focusKey] : undefined;

  return (
    <div className="pipeline">
      <div className="toolbar">
        <div className="seg">
          <button className={side === 'buy' ? 'on' : ''} onClick={() => { if (side !== 'buy' && changeFocus(null)) setSide('buy'); }}>
            매수 파이프라인 ({conditions?.buy.length || 0})
          </button>
          <button className={side === 'sell' ? 'on' : ''} onClick={() => { if (side !== 'sell' && changeFocus(null)) setSide('sell'); }}>
            매도 파이프라인 ({conditions?.sell.length || 0})
          </button>
        </div>
        <SymbolPicker />
        {sym && (
          <span className="symline">
            {sym.position ? `보유 ${sym.position.qty}주 @ ${fmtInt(sym.position.buyPrice)}` : '미보유'}
            {sym.realizedKrw ? ` · 실현 ${fmtKrw(sym.realizedKrw)}` : ''}
            {side === 'buy' && sym.lastEval ? ` · 마지막 평가 ${fmtKstTime(sym.lastEval.ts)}` : ''}
          </span>
        )}
        <span className="legend">
          {sym ? (
            side === 'buy' ? (
              <>
                <i className="lg pass">✓ 통과</i>
                <i className="lg fail">✕ 막힘</i>
                <i className="lg hold">… 대기</i>
                <i className="lg idle">· 미도달</i>
                <i className="lg diag">진단 P/F (점선)</i>
              </>
            ) : (
              <>
                <i className="lg fail">✕ 발동</i>
                <i className="lg hold">… 보류</i>
                <i className="lg idle">· 해당 없음</i>
              </>
            )
          ) : (
            <span className="muted">노드 너비·막대 = 오늘 정규장 막힌 폴링 수 (개장 전 별도 표기)</span>
          )}
        </span>
      </div>
      <div className="pipeline-body">
        <div className="flow">
          {!snap && <div className="empty-overlay">데이터 없음 - 설정/연결 탭에서 서버에 연결하거나 로컬 폴더를 여세요.</div>}
          <ToggleCtx.Provider value={toggleCtx}>
          <ReactFlow
            key={side}
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.04, minZoom: 0.6, maxZoom: 1 }}
            minZoom={0.2}
            panOnScroll
            zoomOnScroll={false}
            zoomOnPinch
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            onNodeClick={(_e, n) => n.type === 'cond' && changeFocus(n.id === focus ? null : n.id)}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={24} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
          </ToggleCtx.Provider>
        </div>
        <aside className="detail">
          {!focusKey && <p className="muted">노드를 누르면 조건 설명과 마지막 로그 원문을 보여 줍니다.</p>}
          {focusKey && (
            <>
              <h3>
                {focusKey.startsWith('_') ? focusKey + ' ' : ''}
                {cond?.title || PSEUDO_LABEL[focusKey] || focusKey}
              </h3>
              {cond && (
                <div className="kv">
                  <span>함수</span>
                  <code>{cond.func}</code>
                  <span>단계</span>
                  <span>{cond.stage}</span>
                  <span>평가 순서</span>
                  <span>{cond.order}</span>
                </div>
              )}
              {cond?.description && <p className="desc">{cond.description}</p>}
              {stat && (
                <p>
                  오늘 전체: 막힘 {fmtInt(stat.block)} · 통과 {fmtInt(stat.pass)}
                  {stat.hold ? ` · 대기 ${fmtInt(stat.hold)}` : ''}
                  {stat.preOpenBlock ? ` · 개장 전 막힘 ${fmtInt(stat.preOpenBlock)}` : ''}
                </p>
              )}
              {sstat && (
                <p>
                  오늘 전체: 발동 {sstat.trigger} · 보류 {sstat.hold}
                </p>
              )}
              {sym && (
                <>
                  <h4>
                    {sym.code} {sym.name} - 이 조건의 마지막 로그
                  </h4>
                  {ev ? (
                    <>
                      <p>
                        {fmtKstTime(ev.ts)} [{ev.tag}] {ev.reason}
                      </p>
                      <pre className="raw">{ev.raw}</pre>
                    </>
                  ) : (
                    <p className="muted">오늘 이 조건에 해당하는 로그가 없습니다 (최근 2,000건 기준).</p>
                  )}
                </>
              )}
              {items.length > 0 && (
                <>
                  <h4>설정값</h4>
                  {overrideErrors.length > 0 && (
                    <div className="callout warn small">
                      서버가 거부한 오버라이드 항목:
                      <ul>
                        {overrideErrors.map((e) => (
                          <li key={e}>{e}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {!canEdit && (
                    <p className="muted small">
                      {connected ? '서버의 r016 조건 JSON이 아니어서 읽기 전용입니다 (설정/연결 탭에서 조건 JSON 새로고침).' : '서버에 연결되어 있지 않아 읽기 전용입니다.'}
                    </p>
                  )}
                  <ValuesEditor items={items} ed={ed} disabled={!canEdit} />
                  {canEdit && items.some((i) => i.editable) && (
                    <>
                      <EditorActions ed={ed} disabled={!canEdit} />
                      <p className="muted small">저장한 값은 서버 오버라이드 파일에 기록되고, r003을 재시작해야 봇에 반영됩니다 (핫리로드 없음).</p>
                    </>
                  )}
                </>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
