// Update log
// - [2026-11-24] type=feat (Window type 'watchsum' 관심종목(요약) (user 2026-11-24): 종목명 / 현재가 / 등락률 only, windows/WatchSumWindow.tsx)
// - [2026-11-06] type=feat (HtsH renderer state (zustand): login state; the MDI window list (type, position, size, z-order, minimized /
//   maximized, link group, symbol, per-window options) with focus / move / resize / arrange (cascade, tile, rows, columns, minimize /
//   restore / close all); link groups 1-4 - changing the symbol of a linked window changes every window of the same colour; realtime caches
//   per code (quote fields, order book, latest trades) fed by the main process; the layout is saved in localStorage)
// - [2026-11-06] type=feat (Watchlist groups shared by every watchlist window (htsh.watch.v1; the codes of an older watchlist window become
//   the first group), setCodeAll - one symbol for every open symbol window regardless of link colour (watchlist double click), theme
//   system / light / dark (htsh.theme, applied as <html data-theme>))
// - [2026-11-06] type=fix (Realtime trades are merged once per code and batch (addRealtimeTrades); the trade cache of a code is dropped
//   when the KST date changes (rt.tradeDay) - accumulated volumes restart every day)
// - [2026-11-06] type=fix (realtime trades go through addRealtimeTrade - a trade the REST list delivered first is not counted twice)
// - [2026-11-06] type=fix (mergeTrades: REST trade lists join the realtime trade cache (no duplicates) - tick charts fill without realtime)
// - [2026-11-19] type=feat (Watchlist groups are saved to %APPDATA%\htsh\watchlist\watchlist.json through the main process (localStorage
//   stays as a copy; on the first start the localStorage groups move into the file). syncPicks: "<...>YYYYMMDD_picks.txt" files in that
//   folder become tabs named MMDD (shared/watchPicks) - checked at start and every 30 s (user 2026-11-19))
// - [2026-11-20] type=feat (AutoBotHtsH: the AutoBotH tabs are window types 'bot-<tab>' (one window each - openBot() focuses an open
//   one); names that clash with HtsH get the 봇 prefix (봇 차트, 봇 설정·연결))
// - [2026-11-20] type=feat (fontScale: 글자 크기 (page zoom 80 - 150 %, kept per PC, applied at start); followCode: every open symbol
//   window shows the code without opening one - used for the symbol picked in a bot window)
// - [2026-11-20] type=feat (글자 크기 in points (user 2026-11-20): fontPt 8 - 14 pt (default 9 - user 2026-11-20; was 10) sets the CSS size variable --fs that every
//   font size of the app uses; the page zoom of the earlier setting is reset to 100 %)
// - [2026-11-23] type=feat (Window type 'rebound' (user 2026-11-23): the rebound watcher settings / history / KakaoTalk are a normal MDI
//   window instead of a dialog over the app; opts.tab keeps the tab shown)
// - [2026-11-23] type=feat (syncR004 (user 2026-11-23): the bot's r004 after the 종목 선별 -> 감시목록 반영 job is the watchlist tab of
//   the next trading day (MMDD of the r004 header for_trading_day) at the front; refreshed when r004 changes, not made again after
//   the user deleted it until r004 changes)
// - [2026-11-20] type=feat (One window per kind (user 2026-11-20): open() focuses the open window of that kind instead of adding another
//   (a symbol window then shows the asked code); a saved layout with several windows of one kind keeps the front one)

import { create } from 'zustand';
import type { AuthState } from '../../preload/index';
import type { OrderBook, Quote, TradeTick } from '../../shared/kis';
import { addRealtimeTrades, mergeTrades } from '../../shared/live';
import { mergePicks, r004Entry, renamePickTabs } from '../../shared/watchPicks';

/** KST trading date "YYYY-MM-DD" - the trade cache of a code is dropped when it changes (accumulated volumes restart every day) */
export const kstDay = () => new Date(Date.now() + 9 * 3600000).toISOString().slice(0, 10);
import { wsToOrderBook, wsToTrade } from '../../shared/kis';

/** [2026-11-20] AutoBotH tabs (src/bot/renderer/src/store.ts Tab) as window types */
export const BOT_TABS = ['control', 'jobs', 'collector', 'pipeline', 'board', 'timeline', 'chart', 'positions', 'trades', 'paper', 'settings'] as const;
export type BotTab = (typeof BOT_TABS)[number];
export type BotWinType = `bot-${BotTab}`;
export type WinType = 'chart' | 'quote' | 'info' | 'watch' | 'watchtile' | 'watchsum' | 'rank' | 'index' | 'news' | 'rebound' | BotWinType;
export const isBotWin = (t: WinType): t is BotWinType => t.startsWith('bot-');
export const botWinOf = (tab: BotTab): BotWinType => `bot-${tab}`;

export const BOT_TITLE: Record<BotTab, string> = {
  control: '제어',
  jobs: '데이터 작업',
  collector: '수집기',
  pipeline: '파이프라인',
  board: '종목 보드',
  timeline: '타임라인',
  chart: '봇 차트',
  positions: '보유 현황',
  trades: '매매 결과',
  paper: '가상 매매',
  settings: '봇 설정·연결',
};
/** bot menu groups (as the AutoBotH sidebar) */
export const BOT_GROUPS: { group: string; tabs: BotTab[] }[] = [
  { group: '운영', tabs: ['control', 'jobs', 'collector', 'pipeline'] },
  { group: '분석', tabs: ['board', 'timeline', 'chart', 'positions', 'trades', 'paper'] },
  { group: '시스템', tabs: ['settings'] },
];

export const WIN_TITLE: Record<WinType, string> = {
  chart: '차트',
  quote: '현재가',
  info: '종목정보',
  watch: '관심(목록)', // [2026-11-24] 관심종목 -> 관심(목록) (user 2026-11-24)
  watchtile: '관심(타일)', // [2026-11-24] shorter titles (user 2026-11-24): 관심종목(타일) -> 관심(타일)
  watchsum: '관심(요약)',
  rank: '순위',
  index: '지수',
  news: '뉴스',
  rebound: '바닥 반등 초입 감시',
  ...(Object.fromEntries(BOT_TABS.map((t) => [botWinOf(t), BOT_TITLE[t]])) as Record<BotWinType, string>),
};

/** windows that follow a symbol (others ignore setCode) */
export const SYMBOL_WINS: WinType[] = ['chart', 'quote', 'info', 'news'];

const DEFAULT_SIZE: Record<WinType, [number, number]> = {
  chart: [760, 520],
  quote: [520, 560],
  info: [960, 700],
  watch: [560, 380],
  watchtile: [720, 480],
  watchsum: [330, 460],
  rank: [600, 460],
  index: [480, 360],
  news: [520, 360],
  rebound: [1360, 760],
  // bot views were made for a full window
  ...(Object.fromEntries(BOT_TABS.map((t) => [botWinOf(t), [1180, 760]])) as Record<BotWinType, [number, number]>),
};

export const LINK_COLORS = ['', '#e5484d', '#3e9bf4', '#30a46c', '#f5a524'];

export interface Win {
  id: string;
  type: WinType;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  min: boolean;
  max: boolean;
  /** 0 = not linked, 1-4 = colour group */
  link: number;
  code: string;
  opts: Record<string, unknown>;
}

export interface Area {
  w: number;
  h: number;
}

export type ArrangeMode = 'cascade' | 'tile' | 'rows' | 'cols';

export interface WatchGroup {
  id: string;
  name: string;
  codes: string[];
  /** [2026-11-19] picks file the group was made from */
  src?: string;
}
export const WATCH_MAX = 60;
export type Theme = 'system' | 'light' | 'dark';

interface RtCache {
  quotes: Record<string, Partial<Quote> & { at: number }>;
  books: Record<string, OrderBook & { at: number }>;
  trades: Record<string, TradeTick[]>;
  /** KST date the cached trades of a code belong to */
  tradeDay: Record<string, string>;
}

interface State {
  auth: AuthState | null;
  wins: Win[];
  active: string | null;
  zTop: number;
  area: Area;
  rt: RtCache;
  /** last symbol chosen anywhere (new symbol windows start with it) */
  lastCode: string;
  setAuth: (a: AuthState) => void;
  setArea: (a: Area) => void;
  open: (type: WinType, init?: Partial<Pick<Win, 'code' | 'link' | 'opts'>>) => string;
  /** [2026-11-20] the window of a bot tab: focus the open one, else open it */
  openBot: (tab: BotTab) => string;
  close: (id: string) => void;
  focus: (id: string) => void;
  move: (id: string, x: number, y: number) => void;
  resize: (id: string, w: number, h: number, x?: number, y?: number) => void;
  toggleMin: (id: string) => void;
  toggleMax: (id: string) => void;
  setCode: (id: string | null, code: string) => void;
  setLink: (id: string, link: number) => void;
  setOpts: (id: string, patch: Record<string, unknown>) => void;
  arrange: (mode: ArrangeMode) => void;
  minimizeAll: () => void;
  restoreAll: () => void;
  closeAll: () => void;
  applyRt: (batch: { tr: string; code: string; rec: Record<string, string> }[]) => void;
  /** REST trade list (newest first) merged into the trade cache of `code` */
  mergeTrades: (code: string, list: TradeTick[]) => void;
  resetLayout: () => void;
  /** every open symbol window shows `code` (opens a chart when there is none) */
  setCodeAll: (code: string) => void;
  watch: WatchGroup[];
  setWatch: (groups: WatchGroup[]) => void;
  /** [2026-11-19] picks file name -> mtime when it became a tab */
  watchImported: Record<string, number>;
  /** [2026-11-19] read the watchlist file (once, at start); before that, changes are kept in localStorage only */
  initWatch: () => Promise<void>;
  /** [2026-11-19] new / changed picks files -> tabs; returns the names of the tabs made */
  syncPicks: () => Promise<string[]>;
  /** [2026-11-23] bot r004 -> watchlist tab of its trading day (first); names of the tabs made */
  syncR004: () => Promise<string[]>;
  /** add / remove a code in a group */
  toggleWatch: (groupId: string, code: string) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
  /** [2026-11-20] 글자 크기, points (8 - 14, default 10) */
  fontPt: number;
  setFontPt: (pt: number) => void;
  /** [2026-11-20] the open symbol windows (chart / current price / info / news) show `code`; nothing is opened */
  followCode: (code: string) => void;
}

const LAYOUT_KEY = 'htsh.layout.v1';
let seq = 1;
const newId = () => `w${Date.now().toString(36)}${seq++}`;

/** [2026-11-20] one window per kind: the front one (highest z) of each kind is kept */
export function onePerKind(wins: Win[]): Win[] {
  const best = new Map<string, Win>();
  for (const w of wins) {
    const b = best.get(w.type);
    if (!b || (w.z || 0) > (b.z || 0)) best.set(w.type, w);
  }
  return wins.filter((w) => best.get(w.type) === w);
}

function loadLayout(): { wins: Win[]; lastCode: string } {
  try {
    const v = JSON.parse(localStorage.getItem(LAYOUT_KEY) || 'null');
    if (v && Array.isArray(v.wins)) return { wins: onePerKind(v.wins.filter((w: Win) => w && WIN_TITLE[w.type])), lastCode: v.lastCode || '005930' };
  } catch {
    /* storage unavailable */
  }
  return { wins: [], lastCode: '005930' };
}

let saveTimer: ReturnType<typeof setTimeout> | null = null;
function saveLayout(s: { wins: Win[]; lastCode: string }) {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(LAYOUT_KEY, JSON.stringify({ wins: s.wins, lastCode: s.lastCode }));
    } catch {
      /* storage unavailable */
    }
  }, 400);
}

const clampWin = (w: Win, a: Area): Win => {
  const ww = Math.max(260, Math.min(w.w, a.w));
  const hh = Math.max(160, Math.min(w.h, a.h));
  return { ...w, w: ww, h: hh, x: Math.max(0, Math.min(w.x, a.w - 80)), y: Math.max(0, Math.min(w.y, a.h - 32)) };
};

const initial = loadLayout();

const WATCH_KEY = 'htsh.watch.v1';
const THEME_KEY = 'htsh.theme';
const FONT_KEY = 'autobothtsh.fontPt';
export const FONT_PTS = [8, 9, 10, 11, 12, 13, 14];
export const FONT_PT_DEFAULT = 9;
export function loadFontPt(): number {
  try {
    const v = Number(localStorage.getItem(FONT_KEY));
    if (FONT_PTS.includes(v)) return v;
  } catch {
    /* storage unavailable */
  }
  return FONT_PT_DEFAULT;
}
/** the app's text size variable (styles.css --fs) */
export function applyFontPt(pt: number): void {
  document.documentElement.style.setProperty('--fs', `${pt}pt`);
}
export const DEFAULT_WATCH = ['005930', '000660', '035420', '035720', '005380', '051910', '068270', '105560'];
export const newGroupId = () => `g${Date.now().toString(36)}${seq++}`;

function loadWatch(): WatchGroup[] {
  try {
    const v = JSON.parse(localStorage.getItem(WATCH_KEY) || 'null');
    if (Array.isArray(v) && v.length) return v.filter((g) => g && g.id && Array.isArray(g.codes));
  } catch {
    /* storage unavailable */
  }
  // first run: an older watchlist window's codes become the first group
  const old = initial.wins.find((w) => w.type === 'watch' && Array.isArray(w.opts.codes))?.opts.codes as string[] | undefined;
  return [{ id: 'g1', name: '관심 1', codes: old?.length ? old : DEFAULT_WATCH }];
}
/** the watchlist file was read: from now on every change is written to it */
let watchFileReady = false;
function saveWatch(g: WatchGroup[], imported: Record<string, number>) {
  try {
    localStorage.setItem(WATCH_KEY, JSON.stringify(g));
  } catch {
    /* storage unavailable */
  }
  if (watchFileReady) void window.hts.watchSave({ groups: g, imported }).catch(() => undefined);
}

function loadTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    if (t === 'light' || t === 'dark') return t;
  } catch {
    /* storage unavailable */
  }
  return 'system';
}
export function applyTheme(t: Theme): void {
  if (t === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
}

export const useStore = create<State>()((set, get) => {
  const commit = (patch: Partial<State>) => {
    set(patch);
    const s = get();
    saveLayout({ wins: s.wins, lastCode: s.lastCode });
  };
  const upd = (id: string, f: (w: Win) => Win) => commit({ wins: get().wins.map((w) => (w.id === id ? f(w) : w)) });

  return {
    auth: null,
    wins: initial.wins,
    active: initial.wins.length ? initial.wins.reduce((a, w) => (w.z > a.z ? w : a)).id : null,
    zTop: Math.max(1, ...initial.wins.map((w) => w.z)),
    area: { w: 1200, h: 800 },
    rt: { quotes: {}, books: {}, trades: {}, tradeDay: {} },
    lastCode: initial.lastCode,

    setAuth: (auth) => set({ auth }),
    setArea: (area) => set({ area, wins: get().wins.map((w) => clampWin(w, area)) }),

    open: (type, init) => {
      const s = get();
      // [2026-11-20] one window per kind: bring the open one to the front
      const had = s.wins.find((x) => x.type === type);
      if (had) {
        s.focus(had.id);
        if (init?.code && SYMBOL_WINS.includes(type) && had.code !== init.code) get().setCode(had.id, init.code);
        return had.id;
      }
      const [w, h] = DEFAULT_SIZE[type];
      const n = s.wins.filter((x) => !x.min).length;
      const x = Math.min(24 + (n % 8) * 28, Math.max(0, s.area.w - w));
      const y = Math.min(16 + (n % 8) * 28, Math.max(0, s.area.h - h));
      const id = newId();
      const z = s.zTop + 1;
      const win = clampWin({ id, type, x, y, w, h, z, min: false, max: false, link: init?.link ?? (SYMBOL_WINS.includes(type) ? 1 : 0), code: init?.code || s.lastCode, opts: init?.opts || {} }, s.area);
      commit({ wins: [...s.wins, win], active: id, zTop: z });
      return id;
    },
    openBot: (tab) => {
      const s = get();
      const w = s.wins.find((x) => x.type === botWinOf(tab));
      if (w) {
        s.focus(w.id);
        return w.id;
      }
      return s.open(botWinOf(tab));
    },
    close: (id) => {
      const wins = get().wins.filter((w) => w.id !== id);
      commit({ wins, active: wins.length ? wins.reduce((a, w) => (w.z > a.z ? w : a)).id : null });
    },
    focus: (id) => {
      const s = get();
      if (s.active === id && s.wins.find((w) => w.id === id)?.z === s.zTop) return;
      const z = s.zTop + 1;
      commit({ wins: s.wins.map((w) => (w.id === id ? { ...w, z, min: false } : w)), active: id, zTop: z });
    },
    move: (id, x, y) => upd(id, (w) => clampWin({ ...w, x: Math.round(x), y: Math.round(y), max: false }, get().area)),
    resize: (id, w, h, x, y) => upd(id, (v) => clampWin({ ...v, w: Math.round(w), h: Math.round(h), x: Math.round(x ?? v.x), y: Math.round(y ?? v.y), max: false }, get().area)),
    toggleMin: (id) => {
      const s = get();
      const w = s.wins.find((x) => x.id === id);
      if (!w) return;
      if (w.min) get().focus(id);
      else commit({ wins: s.wins.map((x) => (x.id === id ? { ...x, min: true } : x)), active: s.active === id ? null : s.active });
    },
    toggleMax: (id) => upd(id, (w) => ({ ...w, max: !w.max, min: false })),
    setCode: (id, code) => {
      const s = get();
      const src = id ? s.wins.find((w) => w.id === id) : null;
      const wins = s.wins.map((w) => {
        if (!SYMBOL_WINS.includes(w.type)) return w;
        if (w.id === id) return { ...w, code };
        if (src && src.link && w.link === src.link) return { ...w, code };
        return w;
      });
      commit({ wins, lastCode: code });
    },
    setLink: (id, link) => upd(id, (w) => ({ ...w, link })),
    setOpts: (id, patch) => upd(id, (w) => ({ ...w, opts: { ...w.opts, ...patch } })),

    arrange: (mode) => {
      const s = get();
      const vis = s.wins.filter((w) => !w.min).sort((a, b) => a.z - b.z);
      const n = vis.length;
      if (!n) return;
      const { w: AW, h: AH } = s.area;
      const pos = new Map<string, Partial<Win>>();
      if (mode === 'cascade') {
        const ww = Math.round(AW * 0.62);
        const hh = Math.round(AH * 0.66);
        vis.forEach((w, i) => pos.set(w.id, { x: (i * 30) % Math.max(30, AW - ww), y: (i * 30) % Math.max(30, AH - hh), w: ww, h: hh }));
      } else if (mode === 'rows' || mode === 'cols') {
        vis.forEach((w, i) =>
          pos.set(
            w.id,
            mode === 'rows'
              ? { x: 0, y: Math.round((AH / n) * i), w: AW, h: Math.round(AH / n) }
              : { x: Math.round((AW / n) * i), y: 0, w: Math.round(AW / n), h: AH },
          ),
        );
      } else {
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        vis.forEach((w, i) => {
          const r = Math.floor(i / cols);
          const inRow = r === rows - 1 ? n - cols * (rows - 1) : cols;
          const c = i - r * cols;
          const cw = AW / inRow;
          pos.set(w.id, { x: Math.round(c * cw), y: Math.round((AH / rows) * r), w: Math.round(cw), h: Math.round(AH / rows) });
        });
      }
      let z = s.zTop;
      commit({ wins: s.wins.map((w) => (pos.has(w.id) ? { ...w, ...pos.get(w.id), max: false, z: ++z } : w)), zTop: z });
    },
    minimizeAll: () => commit({ wins: get().wins.map((w) => ({ ...w, min: true })), active: null }),
    restoreAll: () => commit({ wins: get().wins.map((w) => ({ ...w, min: false, max: false })) }),
    closeAll: () => commit({ wins: [], active: null }),

    applyRt: (batch) => {
      const rt = get().rt;
      const quotes = { ...rt.quotes };
      const books = { ...rt.books };
      const trades = { ...rt.trades };
      const tradeDay = { ...rt.tradeDay };
      const now = Date.now();
      const day = kstDay();
      const newTrades = new Map<string, TradeTick[]>();
      for (const d of batch) {
        if (d.tr === 'H0STCNT0') {
          const { trade, quote } = wsToTrade(d.rec);
          quotes[d.code] = { ...quotes[d.code], ...Object.fromEntries(Object.entries(quote).filter(([, v]) => v !== undefined && !(typeof v === 'number' && Number.isNaN(v)))), at: now };
          newTrades.set(d.code, [...(newTrades.get(d.code) || []), trade]);
        } else if (d.tr === 'H0STASP0') {
          books[d.code] = { ...wsToOrderBook(d.rec), at: now };
        }
      }
      // one merge per code and batch (the cache can hold 30,000 trades)
      for (const [code, list] of newTrades) {
        const base = tradeDay[code] === day ? trades[code] || [] : [];
        tradeDay[code] = day;
        trades[code] = addRealtimeTrades(base, list);
      }
      set({ rt: { quotes, books, trades, tradeDay } });
    },
    mergeTrades: (code, list) => {
      const rt = get().rt;
      const day = kstDay();
      const cur = rt.tradeDay[code] === day ? rt.trades[code] || [] : [];
      const next = mergeTrades(cur, list);
      if (next !== cur || rt.tradeDay[code] !== day) set({ rt: { ...rt, trades: { ...rt.trades, [code]: next }, tradeDay: { ...rt.tradeDay, [code]: day } } });
    },
    resetLayout: () => {
      try {
        localStorage.removeItem(LAYOUT_KEY);
      } catch {
        /* storage unavailable */
      }
      commit({ wins: [], active: null });
    },

    followCode: (code) => {
      const s = get();
      if (!s.wins.some((w) => SYMBOL_WINS.includes(w.type) && w.code !== code)) return;
      commit({ wins: s.wins.map((w) => (SYMBOL_WINS.includes(w.type) ? { ...w, code } : w)), lastCode: code });
    },
    setCodeAll: (code) => {
      const s = get();
      if (!s.wins.some((w) => SYMBOL_WINS.includes(w.type))) {
        commit({ lastCode: code });
        get().open('chart', { code });
        return;
      }
      commit({ wins: s.wins.map((w) => (SYMBOL_WINS.includes(w.type) ? { ...w, code } : w)), lastCode: code });
    },
    watch: loadWatch(),
    setWatch: (groups) => {
      set({ watch: groups });
      saveWatch(groups, get().watchImported);
    },
    watchImported: {},
    initWatch: async () => {
      if (watchFileReady) return;
      const f = await window.hts.watchLoad().catch(() => null);
      watchFileReady = true;
      if (f && f.groups.length) {
        const rn = renamePickTabs(f.groups);
        set({ watch: rn.groups, watchImported: f.imported || {} });
        if (rn.changed) saveWatch(rn.groups, f.imported || {});
        try {
          localStorage.setItem(WATCH_KEY, JSON.stringify(rn.groups));
        } catch {
          /* storage unavailable */
        }
      } else saveWatch(get().watch, get().watchImported); // first start: the localStorage groups move into the file
      await get().syncPicks();
    },
    syncPicks: async () => {
      if (!watchFileReady) return [];
      const picks = await window.hts.watchPicks().catch(() => []);
      const r = mergePicks(get().watch, get().watchImported, picks, newGroupId, WATCH_MAX);
      if (!r.changed) return [];
      set({ watch: r.groups, watchImported: r.imported });
      saveWatch(r.groups, r.imported);
      return r.added;
    },
    syncR004: async () => {
      if (!watchFileReady || !window.api?.watchlistGet) return [];
      const res = await window.api.watchlistGet().catch(() => null);
      const e = res?.ok ? r004Entry(res.data) : null;
      if (!e) return [];
      const r = mergePicks(get().watch, get().watchImported, [e], newGroupId, WATCH_MAX, true);
      if (!r.changed) return [];
      set({ watch: r.groups, watchImported: r.imported });
      saveWatch(r.groups, r.imported);
      return r.added;
    },
    toggleWatch: (groupId, code) => {
      const groups = get().watch.map((g) => {
        if (g.id !== groupId) return g;
        if (g.codes.includes(code)) return { ...g, codes: g.codes.filter((c) => c !== code) };
        return g.codes.length >= WATCH_MAX ? g : { ...g, codes: [...g.codes, code] };
      });
      get().setWatch(groups);
    },
    fontPt: loadFontPt(),
    setFontPt: (pt) => {
      const v = FONT_PTS.includes(pt) ? pt : FONT_PT_DEFAULT;
      set({ fontPt: v });
      applyFontPt(v);
      try {
        localStorage.setItem(FONT_KEY, String(v));
      } catch {
        /* storage unavailable */
      }
    },
    theme: loadTheme(),
    setTheme: (theme) => {
      set({ theme });
      applyTheme(theme);
      try {
        if (theme === 'system') localStorage.removeItem(THEME_KEY);
        else localStorage.setItem(THEME_KEY, theme);
      } catch {
        /* storage unavailable */
      }
    },
  };
});

export function initStore(): void {
  applyTheme(useStore.getState().theme);
  // [2026-11-20] 글자 크기 (points): saved size, Ctrl + = / Ctrl + - / Ctrl + 0 (= 9pt); the old zoom setting is reset to 100 %
  window.hts.setZoom(1);
  applyFontPt(useStore.getState().fontPt);
  window.addEventListener('keydown', (e) => {
    if (!e.ctrlKey || e.altKey || e.metaKey) return;
    const s = useStore.getState();
    const i = Math.max(0, FONT_PTS.indexOf(s.fontPt));
    if (e.key === '=' || e.key === '+') s.setFontPt(FONT_PTS[Math.min(FONT_PTS.length - 1, i + 1)]);
    else if (e.key === '-') s.setFontPt(FONT_PTS[Math.max(0, i - 1)]);
    else if (e.key === '0') s.setFontPt(FONT_PT_DEFAULT);
    else return;
    e.preventDefault();
  });
  void window.hts.authState().then((a) => useStore.getState().setAuth(a));
  window.hts.onAuth((a) => useStore.getState().setAuth(a));
  window.hts.onRtStatus((rt) => {
    const a = useStore.getState().auth;
    if (a) useStore.getState().setAuth({ ...a, rt });
  });
  window.hts.onRt((batch) => useStore.getState().applyRt(batch));
  // screen-test hook: change the displayed realtime state (UI only, nothing is sent to KIS)
  (window as unknown as { __htsTest: unknown }).__htsTest = {
    setRt: (status: 'off' | 'connecting' | 'on' | 'error') => {
      const a = useStore.getState().auth;
      if (a) useStore.getState().setAuth({ ...a, rt: { status } });
    },
  };
}
