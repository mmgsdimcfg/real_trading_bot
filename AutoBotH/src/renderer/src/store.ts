/*
 * Update log
 * - [2026-11-04] type=feat - Renderer global state (zustand): status, latest snapshot, conditions, selected symbol/side/tab and
 *   the selected symbol's timeline (fetched from the worker on demand, refreshed with each snapshot)
 * - [2026-11-05] type=feat - Control tab ('control') and the r016 control state pushed by main; a history refresh counter that
 *   the control tab watches so every write (set/unset/reset/watchlist/service) shows up in the history table
 * - [2026-11-05] type=feat - Timeline price chart choice (kind tick/min/day/week/month + ticks/minutes per bar), kept in localStorage
 * - [2026-11-06] type=feat - Chart indicator selection (chart.ind: bb / ichimoku / rsi / macd / adx), kept with the chart choice
 * - [2026-11-07] type=feat - buyHidden: buy-evaluation rows (condition keys, 'PREP', 'ORDER') hidden on the timeline's y axis,
 *   kept in localStorage
 * - [2026-11-08] type=feat - layout: timeline chart heights (buy / sell evaluation charts, price area, each indicator pane) set by
 *   dragging, kept in localStorage
 * - [2026-11-08] type=feat - sellHidden: rows hidden on the timeline's sell chart y axis (same picker as the buy chart)
 * - [2026-11-08] type=fix - buyHidden moved to key '.v2' with the old keys translated once to the renumbered buy conditions
 * - [2026-11-08] type=feat - buyEvalOnly: "이 종목 평가된 항목만" checkbox state, kept in localStorage
 * - [2026-11-08] type=feat - sellEvalOnly: the same checkbox for the sell chart (merge of the Windows buyEvalOnly change)
 * - [2026-11-06] type=feat - 'chart' tab and its own preferences (chartTab: kind, bar sizes, indicators, previous day on/off)
 * - [2026-11-10] type=feat - 'jobs' tab (데이터 작업) and the local-mode scheduler state (sched)
 * - [2026-11-11] type=feat - 'positions' tab (보유 현황) and feesIncluded ("제세금 포함" of the trade-results / positions tabs,
 *   kept in localStorage)
 * - [2026-11-12] type=fix - The viewed day changed (status.date): selected symbol, its timeline and the old snapshot are dropped.
 *   Before, the reset compared two snapshots, but main sends a null snapshot between two days, so yesterday's symbol stayed
 *   selected on the timeline / chart tabs after a date change
 * - [2026-11-12] type=feat - quotes: current price per watched symbol (r003 quote log), re-read with every snapshot
 */

import { create } from 'zustand';
import type { AppStatus, ConditionsDoc, EvalEvent, Settings, Side, Snapshot, Trade } from '../../shared/types';
import type { ControlState } from '../../shared/control';
import type { ChartKind } from '../../shared/price';
import type { IndicatorKey } from '../../shared/indicators';
import type { LiveQuote } from '../../shared/liveQuote';

type ChartPrefs = { kind: ChartKind; tickN: number; minN: number; ind: IndicatorKey[] };
const IND_KEYS: IndicatorKey[] = ['ma', 'bb', 'ichimoku', 'vwap', 'vol', 'rsi', 'macd', 'adx', 'stoch', 'williams', 'mfi', 'obv'];
/** prefs saved before volume became a switch (no `v`) keep showing volume */
const PREFS_V = 2;
const withVol = (v: { v?: number }, ind: IndicatorKey[]): IndicatorKey[] => (v.v === PREFS_V || ind.includes('vol') ? ind : ['vol', ...ind]);

/** chart tab (HTS-style) choice, kept separately from the timeline chart */
type ChartTabPrefs = ChartPrefs & { prevDay: boolean };
const CHART_TAB_KEY = 'autoboth.chartTab';
function loadChartTab(): ChartTabPrefs {
  const d: ChartTabPrefs = { kind: 'day', tickN: 30, minN: 3, ind: ['ma', 'bb', 'vol', 'rsi', 'macd'], prevDay: true };
  try {
    const v = JSON.parse(localStorage.getItem(CHART_TAB_KEY) || 'null');
    if (v && ['tick', 'min', 'day', 'week', 'month'].includes(v.kind)) {
      const ind = Array.isArray(v.ind) ? withVol(v, (v.ind as IndicatorKey[]).filter((k) => IND_KEYS.includes(k))) : d.ind;
      return { kind: v.kind, tickN: Number(v.tickN) || d.tickN, minN: Number(v.minN) || d.minN, ind, prevDay: v.prevDay !== false };
    }
  } catch {
    /* storage unavailable */
  }
  return d;
}

/** price chart choice of the timeline tab, remembered per PC (localStorage, best effort) */
const CHART_KEY = 'autoboth.priceChart';
function loadChart(): ChartPrefs {
  const d: ChartPrefs = { kind: 'min', tickN: 30, minN: 1, ind: ['bb', 'vol'] };
  try {
    const v = JSON.parse(localStorage.getItem(CHART_KEY) || 'null');
    if (v && ['tick', 'min', 'day', 'week', 'month'].includes(v.kind)) {
      const ind = Array.isArray(v.ind) ? withVol(v, (v.ind as IndicatorKey[]).filter((k) => IND_KEYS.includes(k))) : d.ind;
      return { kind: v.kind, tickN: Number(v.tickN) || d.tickN, minN: Number(v.minN) || d.minN, ind };
    }
  } catch {
    /* storage unavailable */
  }
  return d;
}

/** [2026-11-08] v2 = buy condition keys after the server renumbered them in evaluation order (r005 _001.._025) */
const BUY_HIDDEN_KEY = 'autoboth.timelineBuyHidden.v2';
const BUY_HIDDEN_KEY_V1 = 'autoboth.timelineBuyHidden';
/** old (v1) buy condition key -> new key; keys not listed (_001.._006, PREP, ORDER) kept */
const BUY_KEY_V1_TO_V2: Record<string, string> = {
  _025: '_007', _026: '_008', _008: '_009', _029: '_010', _009: '_011', _014: '_012', _015: '_013', _017: '_014',
  _018: '_015', _019: '_016', _020: '_017', _031: '_018', _030: '_019', _028: '_020', _027: '_021', _021: '_022',
  _022: '_023', _023: '_024', _024: '_025',
};
const strList = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);
function loadBuyHidden(): string[] {
  try {
    const cur = localStorage.getItem(BUY_HIDDEN_KEY);
    if (cur !== null) return strList(JSON.parse(cur));
    const old = localStorage.getItem(BUY_HIDDEN_KEY_V1);
    if (old === null) return [];
    const keys = strList(JSON.parse(old)).map((k) => BUY_KEY_V1_TO_V2[k] ?? k);
    localStorage.setItem(BUY_HIDDEN_KEY, JSON.stringify(keys));
    localStorage.removeItem(BUY_HIDDEN_KEY_V1);
    return keys;
  } catch {
    return [];
  }
}

/** [2026-11-08] timeline sell chart: hidden y-axis rows (sell condition keys, 'PREP', 'SESSION_CLOSE', 'ORDER') */
const SELL_HIDDEN_KEY = 'autoboth.timelineSellHidden';
function loadSellHidden(): string[] {
  try {
    return strList(JSON.parse(localStorage.getItem(SELL_HIDDEN_KEY) || '[]'));
  } catch {
    return [];
  }
}

/** [2026-11-08] timeline heights dragged by the user (px; missing = default size). price = price + volume area,
 *  osc = one entry per indicator pane. Kept per PC (localStorage) */
export interface TimelineLayout {
  buyH?: number;
  sellH?: number;
  priceH?: number;
  osc?: Partial<Record<IndicatorKey, number>>;
}
const LAYOUT_KEY = 'autoboth.timelineLayout';
const posNum = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : undefined);
function loadLayout(): TimelineLayout {
  try {
    const v = JSON.parse(localStorage.getItem(LAYOUT_KEY) || 'null');
    if (!v || typeof v !== 'object') return {};
    const osc: Partial<Record<IndicatorKey, number>> = {};
    for (const k of IND_KEYS) {
      const h = posNum(v.osc?.[k]);
      if (h) osc[k] = h;
    }
    return { buyH: posNum(v.buyH), sellH: posNum(v.sellH), priceH: posNum(v.priceH), osc };
  } catch {
    return {};
  }
}

const BUY_EVAL_ONLY_KEY = 'autoboth.timelineBuyEvalOnly';
/** [2026-11-11] 매매 결과 / 보유 현황: "제세금 포함" (net of the estimated commission + sell tax) */
const FEES_INCLUDED_KEY = 'autoboth.feesIncluded';
const SELL_EVAL_ONLY_KEY = 'autoboth.timelineSellEvalOnly';
function loadFlag(key: string): boolean {
  try {
    return localStorage.getItem(key) === '1';
  } catch {
    return false;
  }
}

export type Tab = 'control' | 'jobs' | 'pipeline' | 'board' | 'timeline' | 'chart' | 'positions' | 'trades' | 'paper' | 'settings';

/** local-mode scheduler state pushed by main ('sched') */
export interface SchedView {
  enabled: boolean;
  lastCollectorAt?: number;
  lastCollectorMsg?: string;
  lastPostcloseMsg?: string;
  lastSyncMsg?: string;
}

interface State {
  sched: SchedView | null;
  status: AppStatus;
  snapshot: Snapshot | null;
  /** current price per code of the viewed day (empty until read / when r003 wrote no quote log) */
  quotes: Record<string, LiveQuote>;
  conditions: ConditionsDoc | null;
  settings: Settings | null;
  tab: Tab;
  side: Side;
  selected: string | null;
  timeline: { code: string; events: EvalEvent[]; trades: Trade[] } | null;
  control: ControlState;
  /** bumped after every write so the history table reloads */
  historyRev: number;
  bumpHistory: () => void;
  chart: ChartPrefs;
  setChart: (patch: Partial<ChartPrefs>) => void;
  chartTab: ChartTabPrefs;
  setChartTab: (patch: Partial<ChartTabPrefs>) => void;
  /** timeline buy chart: y-axis rows hidden by the user */
  buyHidden: string[];
  setBuyHidden: (keys: string[]) => void;
  /** timeline sell chart: y-axis rows hidden by the user */
  sellHidden: string[];
  setSellHidden: (keys: string[]) => void;
  /** timeline chart / pane heights (undefined value = back to the default size) */
  layout: TimelineLayout;
  setLayout: (patch: Partial<Omit<TimelineLayout, 'osc'>> & { osc?: Partial<Record<IndicatorKey, number | undefined>> }) => void;
  /** timeline buy chart: "이 종목 평가된 항목만" checked - the rows follow the selected symbol (buyHidden unused while on) */
  buyEvalOnly: boolean;
  setBuyEvalOnly: (on: boolean) => void;
  /** timeline sell chart: "이 종목 평가된 항목만" checked (sellHidden unused while on) */
  sellEvalOnly: boolean;
  setSellEvalOnly: (on: boolean) => void;
  /** 매매 결과 / 보유 현황: P&L shown net of the estimated costs (shared/fees), kept in localStorage */
  feesIncluded: boolean;
  setFeesIncluded: (on: boolean) => void;
  setTab: (t: Tab) => void;
  setSide: (s: Side) => void;
  select: (code: string | null) => void;
  refreshTimeline: () => Promise<void>;
}

export const useStore = create<State>()((set, get) => ({
  sched: null,
  status: { conn: 'idle', date: '', live: false },
  snapshot: null,
  quotes: {},
  conditions: null,
  settings: null,
  tab: 'pipeline',
  side: 'buy',
  selected: null,
  timeline: null,
  control: { status: null, busy: null },
  historyRev: 0,
  bumpHistory: () => set({ historyRev: get().historyRev + 1 }),
  buyHidden: loadBuyHidden(),
  setBuyHidden: (keys) => {
    set({ buyHidden: keys });
    try {
      localStorage.setItem(BUY_HIDDEN_KEY, JSON.stringify(keys));
    } catch {
      /* storage unavailable */
    }
  },
  sellHidden: loadSellHidden(),
  setSellHidden: (keys) => {
    set({ sellHidden: keys });
    try {
      localStorage.setItem(SELL_HIDDEN_KEY, JSON.stringify(keys));
    } catch {
      /* storage unavailable */
    }
  },
  layout: loadLayout(),
  setLayout: (patch) => {
    const cur = get().layout;
    const osc = { ...cur.osc };
    for (const [k, h] of Object.entries(patch.osc || {})) {
      if (h) osc[k as IndicatorKey] = h;
      else delete osc[k as IndicatorKey];
    }
    const layout: TimelineLayout = { ...cur, ...patch, osc };
    set({ layout });
    try {
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
    } catch {
      /* storage unavailable */
    }
  },
  feesIncluded: loadFlag(FEES_INCLUDED_KEY),
  setFeesIncluded: (on) => {
    set({ feesIncluded: on });
    try {
      localStorage.setItem(FEES_INCLUDED_KEY, on ? '1' : '0');
    } catch {
      /* storage unavailable */
    }
  },
  buyEvalOnly: loadFlag(BUY_EVAL_ONLY_KEY),
  setBuyEvalOnly: (on) => {
    set({ buyEvalOnly: on });
    try {
      localStorage.setItem(BUY_EVAL_ONLY_KEY, on ? '1' : '0');
    } catch {
      /* storage unavailable */
    }
  },
  sellEvalOnly: loadFlag(SELL_EVAL_ONLY_KEY),
  setSellEvalOnly: (on) => {
    set({ sellEvalOnly: on });
    try {
      localStorage.setItem(SELL_EVAL_ONLY_KEY, on ? '1' : '0');
    } catch {
      /* storage unavailable */
    }
  },
  chartTab: loadChartTab(),
  setChartTab: (patch) => {
    const chartTab = { ...get().chartTab, ...patch };
    set({ chartTab });
    try {
      localStorage.setItem(CHART_TAB_KEY, JSON.stringify({ ...chartTab, v: PREFS_V }));
    } catch {
      /* storage unavailable */
    }
  },
  chart: loadChart(),
  setChart: (patch) => {
    const chart = { ...get().chart, ...patch };
    set({ chart });
    try {
      localStorage.setItem(CHART_KEY, JSON.stringify({ ...chart, v: PREFS_V }));
    } catch {
      /* storage unavailable */
    }
  },
  setTab: (tab) => set({ tab }),
  setSide: (side) => set({ side }),
  select: (code) => {
    set({ selected: code, timeline: code && get().timeline?.code === code ? get().timeline : null });
    void get().refreshTimeline();
  },
  refreshTimeline: async () => {
    const code = get().selected;
    if (!code) return set({ timeline: null });
    const r = await window.api.getTimeline(code);
    if (get().selected === code) set({ timeline: { code, ...r } });
  },
}));

export function initStore(): void {
  const api = window.api;
  void api.getStatus().then((status: AppStatus) => useStore.setState({ status }));
  void api.getSnapshot().then((snapshot: Snapshot | null) => useStore.setState({ snapshot }));
  void api.getConditions().then((conditions: ConditionsDoc) => useStore.setState({ conditions }));
  void api.getSettings().then((settings: Settings) => useStore.setState({ settings }));
  api.onStatus((status) => {
    const prev = useStore.getState().status.date;
    // another day: nothing of the old day may stay on screen (the new day's snapshot follows)
    if (prev && status.date && prev !== status.date) useStore.setState({ status, snapshot: null, quotes: {}, selected: null, timeline: null });
    else useStore.setState({ status });
  });
  api.onConditions((conditions) => useStore.setState({ conditions }));
  void api.ctlState().then((control) => useStore.setState({ control }));
  void api.schedState().then((sched) => useStore.setState({ sched }));
  api.onSched((sched) => useStore.setState({ sched }));
  api.onControl((control) => {
    const prev = useStore.getState().control;
    useStore.setState({ control });
    if (control.lastAction && control.lastAction.at !== prev.lastAction?.at) useStore.getState().bumpHistory();
  });
  api.onSnapshot((snapshot) => {
    const prevDate = useStore.getState().snapshot?.date;
    useStore.setState({ snapshot });
    if (snapshot && prevDate && prevDate !== snapshot.date) useStore.setState({ selected: null, timeline: null });
    const st = useStore.getState();
    if (st.selected && (st.tab === 'timeline' || st.tab === 'pipeline')) void st.refreshTimeline();
    // current prices of the day (at the replay time on a replayed past day)
    if (snapshot) {
      const day = snapshot.date;
      void api.getQuotes(day, snapshot.replay?.cursor ?? null).then((quotes) => {
        if (useStore.getState().snapshot?.date === day) useStore.setState({ quotes });
      });
    }
  });
}
