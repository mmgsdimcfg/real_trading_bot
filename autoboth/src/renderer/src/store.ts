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
 * - [2026-11-06] type=feat - 'chart' tab and its own preferences (chartTab: kind, bar sizes, indicators, previous day on/off)
 */

import { create } from 'zustand';
import type { AppStatus, ConditionsDoc, EvalEvent, Settings, Side, Snapshot, Trade } from '../../shared/types';
import type { ControlState } from '../../shared/control';
import type { ChartKind } from '../../shared/price';
import type { IndicatorKey } from '../../shared/indicators';

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

export type Tab = 'control' | 'pipeline' | 'board' | 'timeline' | 'chart' | 'trades' | 'settings';

interface State {
  status: AppStatus;
  snapshot: Snapshot | null;
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
  setTab: (t: Tab) => void;
  setSide: (s: Side) => void;
  select: (code: string | null) => void;
  refreshTimeline: () => Promise<void>;
}

export const useStore = create<State>()((set, get) => ({
  status: { conn: 'idle', date: '', live: false },
  snapshot: null,
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
  api.onStatus((status) => useStore.setState({ status }));
  api.onConditions((conditions) => useStore.setState({ conditions }));
  void api.ctlState().then((control) => useStore.setState({ control }));
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
  });
}
