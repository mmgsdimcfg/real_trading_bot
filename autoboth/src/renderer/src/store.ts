/*
 * Update log
 * - [2026-11-04] type=feat - Renderer global state (zustand): status, latest snapshot, conditions, selected symbol/side/tab and
 *   the selected symbol's timeline (fetched from the worker on demand, refreshed with each snapshot)
 * - [2026-11-05] type=feat - Control tab ('control') and the r016 control state pushed by main; a history refresh counter that
 *   the control tab watches so every write (set/unset/reset/watchlist/service) shows up in the history table
 * - [2026-11-05] type=feat - Timeline price chart choice (kind tick/min/day/week/month + ticks/minutes per bar), kept in localStorage
 * - [2026-11-06] type=feat - Chart indicator selection (chart.ind: bb / ichimoku / rsi / macd / adx), kept with the chart choice
 * - [2026-11-06] type=feat - 'chart' tab and its own preferences (chartTab: kind, bar sizes, indicators, previous day on/off)
 */

import { create } from 'zustand';
import type { AppStatus, ConditionsDoc, EvalEvent, Settings, Side, Snapshot, Trade } from '../../shared/types';
import type { ControlState } from '../../shared/control';
import type { ChartKind } from '../../shared/price';
import type { IndicatorKey } from '../../shared/indicators';

type ChartPrefs = { kind: ChartKind; tickN: number; minN: number; ind: IndicatorKey[] };
const IND_KEYS: IndicatorKey[] = ['ma', 'bb', 'ichimoku', 'vwap', 'rsi', 'macd', 'adx', 'stoch', 'williams', 'mfi', 'obv'];

/** chart tab (HTS-style) choice, kept separately from the timeline chart */
type ChartTabPrefs = ChartPrefs & { prevDay: boolean };
const CHART_TAB_KEY = 'autoboth.chartTab';
function loadChartTab(): ChartTabPrefs {
  const d: ChartTabPrefs = { kind: 'day', tickN: 30, minN: 3, ind: ['ma', 'bb', 'rsi', 'macd'], prevDay: true };
  try {
    const v = JSON.parse(localStorage.getItem(CHART_TAB_KEY) || 'null');
    if (v && ['tick', 'min', 'day', 'week', 'month'].includes(v.kind)) {
      const ind = Array.isArray(v.ind) ? (v.ind as IndicatorKey[]).filter((k) => IND_KEYS.includes(k)) : d.ind;
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
  const d: ChartPrefs = { kind: 'min', tickN: 30, minN: 1, ind: ['bb'] };
  try {
    const v = JSON.parse(localStorage.getItem(CHART_KEY) || 'null');
    if (v && ['tick', 'min', 'day', 'week', 'month'].includes(v.kind)) {
      const ind = Array.isArray(v.ind) ? (v.ind as IndicatorKey[]).filter((k) => IND_KEYS.includes(k)) : d.ind;
      return { kind: v.kind, tickN: Number(v.tickN) || d.tickN, minN: Number(v.minN) || d.minN, ind };
    }
  } catch {
    /* storage unavailable */
  }
  return d;
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
  chartTab: loadChartTab(),
  setChartTab: (patch) => {
    const chartTab = { ...get().chartTab, ...patch };
    set({ chartTab });
    try {
      localStorage.setItem(CHART_TAB_KEY, JSON.stringify(chartTab));
    } catch {
      /* storage unavailable */
    }
  },
  chart: loadChart(),
  setChart: (patch) => {
    const chart = { ...get().chart, ...patch };
    set({ chart });
    try {
      localStorage.setItem(CHART_KEY, JSON.stringify(chart));
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
