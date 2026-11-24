// Update log
// - [2026-11-23] type=feat (관심종목(타일) (user 2026-11-23): the watchlist groups as tiles - name, code, NXT mark, current price, change
//   and rate, the day's open / high / low with a low-high bar marking the price, volume and traded value; the tile is tinted by the
//   change rate. Same groups and 3 s polling as the 관심종목 window (groups are edited there); a click shows the symbol in every open
//   symbol window, a double click opens a chart. Order: as registered / by rate / by traded value (kept with the window))
// - [2026-11-23] type=feat (자동 보기 on the tile window (user 2026-11-23): the group's tiles are shown one after another every N s in
//   every open symbol window, in the order on screen (the chosen sort), the current tile framed and kept in view. Its own setting
//   (autobothtsh.watchTileAuto); one auto view at a time - starting it stops the timeline's / the list window's and the other way round)
// - [2026-11-23] type=feat (종목 추가: the symbol search of the list window on the tile toolbar too - the picked symbol is added to the
//   shown group (not twice, up to WATCH_MAX))
// - [2026-11-23] type=feat (Tiles can be dragged to a new place (user 2026-11-23) - in 등록순 only: the other orders are computed from
//   the prices, a drop there would have no place to keep. moveCode() is the group edit)
// - [2026-11-23] type=feat (Tile size (user 2026-11-23): 작게 (name, price, change only - more tiles per row) / 보통 / 크게 (wider, larger
//   text), kept with the window (opts.size))
// - [2026-11-23] type=feat (항목 menu (user 2026-11-23): which values a tile shows - 종목코드, 대비·등락률, 위치 막대, 시가, 고가, 저가,
//   거래량, 거래대금, 매도호가, 매수호가 (the two quotes are off at first). Kept with the window (opts.fields); the name and the
//   price are always shown. The small tile size still shows the name, price and change only)
// - [2026-11-23] type=feat (The small tile size shows the chosen fields too (user 2026-11-23) - narrower tiles, smaller text, one value
//   per line; it used to show the name, price and change only)
// - [2026-11-23] type=feat (정렬 like the list window's column titles (user 2026-11-23): 등록순 or any value - 종목명, 현재가, 대비, 등락률,
//   거래량, 거래대금, 시가, 고가, 저가, 매도호가, 매수호가 - with a direction button (높은순 / 낮은순, names 가나다순 / 역순). Same sort as
//   the list window (watchSort.ts). Kept with the window (opts.sortKey / sortDir); the older opts.sort values are read as before)
// - [2026-11-23] type=feat (색 select (user 2026-11-23): how strongly a tile is tinted by its change rate - 끔 / 약하게 / 보통 / 진하게 /
//   아주 진하게 (multiplier 0 / 0.5 / 1 / 2 / 3 of the tint), kept with the window (opts.tint))

import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore, WATCH_MAX, type Win } from '../store';
import { useCall, useOutsideClose } from '../hooks';
import { NxtTag } from '../components/NxtTag';
import { SymbolSearch } from '../components/SymbolSearch';
import { FloatPop } from '../components/FloatPop';
import { WatchMenu, type MenuAt } from '../components/WatchMenu';
import { AlarmMark, useTodayAlerts } from '../components/AlarmMark';
import type { WatchRow } from '../../../shared/kis';
import { fmtInt, fmtKrwBig, fmtPct, upDown } from '../lib';
import { firstDir, SORT_KEYS, SORT_LABEL, sortWatch, watchSortOf, type WatchSort, type WatchSortKey } from '../watchSort';
import { AUTO_MAX_SEC, AUTO_MIN_SEC, autoViewMayResume, autoViewStarted, clampSec, nextCode, onAutoViewStarted } from '../../../bot/renderer/src/components/AutoView';

const TILE_AUTO_KEY = 'autobothtsh.watchTileAuto';
const AUTO_NAME = 'watchtile';
interface TileAutoCfg {
  on: boolean;
  sec: number;
}
function loadTileAuto(): TileAutoCfg {
  try {
    const v = JSON.parse(localStorage.getItem(TILE_AUTO_KEY) || '{}') as Partial<TileAutoCfg>;
    return { on: v.on === true, sec: clampSec(v.sec) };
  } catch {
    return { on: false, sec: 10 };
  }
}

export type TileField = 'code' | 'change' | 'bar' | 'open' | 'high' | 'low' | 'volume' | 'amount' | 'ask' | 'bid';
export const TILE_FIELDS: TileField[] = ['code', 'change', 'bar', 'open', 'high', 'low', 'volume', 'amount', 'ask', 'bid'];
export const FIELD_LABEL: Record<TileField, string> = {
  code: '종목코드',
  change: '대비 · 등락률',
  bar: '저가~고가 위치 막대',
  open: '시가',
  high: '고가',
  low: '저가',
  volume: '거래량',
  amount: '거래대금',
  ask: '매도호가',
  bid: '매수호가',
};
export const TILE_FIELDS_DEFAULT: TileField[] = ['code', 'change', 'bar', 'open', 'high', 'low', 'volume', 'amount'];
/** the saved choice (unknown names dropped, kept in menu order); the default when nothing was saved */
export function tileFieldsOf(v: unknown): TileField[] {
  if (!Array.isArray(v)) return TILE_FIELDS_DEFAULT;
  return TILE_FIELDS.filter((f) => v.includes(f));
}
/** the fields shown as label + value rows under the price, in menu order */
const ROW_FIELDS: TileField[] = ['open', 'high', 'low', 'volume', 'amount', 'ask', 'bid'];

const TINTS: [number, string][] = [
  [0, '색 끔'],
  [0.5, '색 약하게'],
  [1, '색 보통'],
  [2, '색 진하게'],
  [3, '색 아주 진하게'],
];
/** the saved tint multiplier; anything else is the normal strength */
export const tileTintOf = (v: unknown): number => (TINTS.some(([k]) => k === v) ? (v as number) : 1);

export type TileSize = 's' | 'm' | 'l';
const SIZES: [TileSize, string][] = [
  ['s', '작게'],
  ['m', '보통'],
  ['l', '크게'],
];
export const tileSizeOf = (v: unknown): TileSize => (v === 's' || v === 'l' ? v : 'm');

/** the saved sort; windows saved before sortKey / sortDir existed carry opts.sort ('pctDesc' | 'pctAsc' | 'amount' | 'list') */
export function tileSortOf(opts: Record<string, unknown>): WatchSort | null {
  if (opts.sortKey !== undefined) return watchSortOf(opts.sortKey, opts.sortDir);
  if (opts.sort === 'pctDesc') return { key: 'changePct', dir: -1 };
  if (opts.sort === 'pctAsc') return { key: 'changePct', dir: 1 };
  if (opts.sort === 'amount') return { key: 'amount', dir: -1 };
  return null;
}

/** `code` moved to the place of `target` in the list (the others keep their order); unchanged when either is missing */
export function moveCode(codes: string[], code: string, target: string): string[] {
  const to = codes.indexOf(target);
  if (code === target || to < 0 || !codes.includes(code)) return codes;
  const rest = codes.filter((c) => c !== code);
  rest.splice(to, 0, code);
  return rest;
}

/** tint strength 0 - 1 of a change rate (3 % = full) */
export const tileHeat = (pct: number | undefined) => (typeof pct === 'number' && Number.isFinite(pct) ? Math.min(1, Math.abs(pct) / 3) : 0);

/** position 0 - 100 of the price between the day's low and high (null when they are not known or equal) */
export function rangePos(x: Pick<WatchRow, 'price' | 'high' | 'low'> | undefined): number | null {
  if (!x || !x.high || !x.low || !(x.high > x.low)) return null;
  return Math.max(0, Math.min(100, ((x.price - x.low) / (x.high - x.low)) * 100));
}

export function WatchTileWindow({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const groups = useStore((s) => s.watch);
  const setCodeAll = useStore((s) => s.setCodeAll);
  const group = groups.find((g) => g.id === win.opts.group) || groups[0];
  const codes = group?.codes || [];
  const sort = tileSortOf(win.opts);
  const size = tileSizeOf(win.opts.size);
  const tint = tileTintOf(win.opts.tint);
  const fields = useMemo(() => tileFieldsOf(win.opts.fields), [win.opts.fields]);
  const has = (f: TileField) => fields.includes(f);
  const rowFields = ROW_FIELDS.filter(has) as ('open' | 'high' | 'low' | 'volume' | 'amount' | 'ask' | 'bid')[];
  const [fieldsOpen, setFieldsOpen] = useState(false);
  const fieldsRef = useOutsideClose<HTMLSpanElement>(fieldsOpen, () => setFieldsOpen(false));
  const toggleField = (f: TileField) => setOpts(win.id, { fields: TILE_FIELDS.filter((k) => (k === f ? !has(k) : has(k))) });
  const r = useCall<WatchRow[]>('multi', [codes], 3000, codes.length > 0);
  const [sel, setSel] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuAt | null>(null);
  const alerts = useTodayAlerts();
  const [dragFrom, setDragFrom] = useState<string | null>(null);
  const canDrag = !sort;
  const dropOn = (target: string) => {
    if (!group || !dragFrom) return;
    const st = useStore.getState();
    st.setWatch(st.watch.map((g) => (g.id === group.id ? { ...g, codes: moveCode(g.codes, dragFrom, target) } : g)));
  };
  const byCode = useMemo(() => new Map((r.data || []).map((x) => [x.code, x])), [r.data]);
  const shown = useMemo(() => sortWatch(codes, byCode, sort), [codes, byCode, sort?.key, sort?.dir]); // eslint-disable-line react-hooks/exhaustive-deps
  const ups = shown.filter((c) => (byCode.get(c)?.change || 0) > 0).length;
  const downs = shown.filter((c) => (byCode.get(c)?.change || 0) < 0).length;
  // 자동 보기 (same rules as the list window's)
  const [auto, setAuto] = useState<TileAutoCfg>(loadTileAuto);
  const [secText, setSecText] = useState(String(auto.sec));
  const saveAuto = (p: Partial<TileAutoCfg>) => {
    setAuto((cur) => {
      const next = { ...cur, ...p };
      try {
        localStorage.setItem(TILE_AUTO_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
    if (p.on) autoViewStarted(AUTO_NAME);
  };
  useEffect(() => onAutoViewStarted(AUTO_NAME, () => saveAuto({ on: false })), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (auto.on && !autoViewMayResume(AUTO_NAME)) saveAuto({ on: false });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const shownRef = useRef(shown);
  shownRef.current = shown;
  const groupKey = codes.join(',');
  useEffect(() => {
    if (!auto.on || !codes.length) return;
    // the first step at once when nothing of this group is selected, then one step per interval (a click restarts the interval);
    // the order is the one on screen at the moment of the step (a re-sort between steps does not restart the interval)
    const step = () => {
      const n = nextCode(shownRef.current, sel);
      if (n) (setSel(n), setCodeAll(n));
    };
    if (!sel || !codes.includes(sel)) {
      step();
      return;
    }
    const t = setTimeout(step, auto.sec * 1000);
    return () => clearTimeout(t);
  }, [auto.on, auto.sec, sel, groupKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const curTile = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    if (auto.on) curTile.current?.scrollIntoView({ block: 'nearest' });
  }, [auto.on, sel]);

  return (
    <div className="list-win tile-win">
      <div className="watch-groups">
        {groups.map((g) => (
          <button key={g.id} className={`watch-tab ${g.id === group?.id ? 'on' : ''}`} onClick={() => (setOpts(win.id, { group: g.id }), setSel(null))} title={g.src ? `${g.src} 에서 자동 등록` : undefined}>
            {g.name} <small>{g.codes.length}</small>
          </button>
        ))}
      </div>
      <div className="win-tb">
        <SymbolSearch
          compact
          placeholder="종목 추가"
          onPick={(c) => {
            if (!group || codes.includes(c) || codes.length >= WATCH_MAX) return;
            const st = useStore.getState();
            st.setWatch(st.watch.map((g) => (g.id === group.id ? { ...g, codes: [...g.codes, c] } : g)));
            setSel(c);
          }}
        />
        <select
          value={sort?.key || ''}
          onChange={(e) => {
            const k = e.target.value as WatchSortKey | '';
            setOpts(win.id, { sortKey: k || null, sortDir: k ? firstDir(k) : null });
          }}
          title="타일 순서"
        >
          <option value="">등록순</option>
          {SORT_KEYS.map((k) => (
            <option key={k} value={k}>
              {SORT_LABEL[k]}
            </option>
          ))}
        </select>
        {sort && (
          <button className="mini" onClick={() => setOpts(win.id, { sortKey: sort.key, sortDir: sort.dir === 1 ? -1 : 1 })} title="정렬 방향 바꾸기">
            {sort.key === 'name' ? (sort.dir === 1 ? '가나다순 ▲' : '역순 ▼') : sort.dir === -1 ? '높은순 ▼' : '낮은순 ▲'}
          </button>
        )}
        <select value={size} onChange={(e) => setOpts(win.id, { size: e.target.value })} title="타일 크기 (작게: 좁은 타일과 작은 글자, 항목은 한 줄에 하나씩)">
          {SIZES.map(([k, label]) => (
            <option key={k} value={k}>
              {label}
            </option>
          ))}
        </select>
        <select value={tint} onChange={(e) => setOpts(win.id, { tint: Number(e.target.value) })} title="등락률에 따른 타일 색의 진하기 (등락률 3%에서 가장 진함)">
          {TINTS.map(([k, label]) => (
            <option key={k} value={k}>
              {label}
            </option>
          ))}
        </select>
        <span className="dropdown" ref={fieldsRef}>
          <button className="mini" onClick={() => setFieldsOpen((o) => !o)} title="타일에 표시할 항목 (종목명과 현재가는 항상 표시)">
            항목 ({fields.length}/{TILE_FIELDS.length}) ▾
          </button>
          {fieldsOpen && (
            <FloatPop className="dd-menu" onMouseDown={(e) => e.stopPropagation()}>
              {TILE_FIELDS.map((f) => (
                <label key={f} className="check dd-check">
                  <input type="checkbox" checked={has(f)} onChange={() => toggleField(f)} /> {FIELD_LABEL[f]}
                </label>
              ))}
              <button type="button" onClick={() => setOpts(win.id, { fields: TILE_FIELDS_DEFAULT })}>
                기본값으로
              </button>
            </FloatPop>
          )}
        </span>
        <span className="muted small" title="NXT 표시 종목은 KRX 장이 닫힌 시간(09:00 이전, 15:40 이후)에 NXT 시세로 표시합니다">
          {codes.length}/{WATCH_MAX}종목 · <span className="up">상승 {ups}</span> · <span className="down">하락 {downs}</span> · 3초마다
        </span>
        <span className="watch-auto" title={`이 그룹 종목을 화면에 보이는 순서대로 ${auto.sec}초마다 열린 종목 창에 표시 (타일을 누르면 그 종목부터 이어감). 타임라인 / 관심(목록) 창의 자동 보기와 동시에 실행되지 않습니다`}>
          <label className="check">
            <input type="checkbox" checked={auto.on} onChange={(e) => saveAuto({ on: e.target.checked })} /> 자동 보기
          </label>
          <input
            className="auto-sec"
            type="number"
            min={AUTO_MIN_SEC}
            max={AUTO_MAX_SEC}
            value={secText}
            onChange={(e) => setSecText(e.target.value)}
            onBlur={() => (saveAuto({ sec: clampSec(secText) }), setSecText(String(clampSec(secText))))}
            onKeyDown={(e) => (e.stopPropagation(), e.key === 'Enter' && (e.currentTarget as HTMLInputElement).blur())}
          />
          초{auto.on && shown.length ? ` · ${sel && shown.includes(sel) ? shown.indexOf(sel) + 1 : '-'}/${shown.length}` : ''}
        </span>
        <span className="muted small watch-hint">클릭 → 열린 창 모두 · 더블클릭 → 차트 · 우클릭 → 메뉴(삭제 / 복사) · {canDrag ? '끌어서 순서 변경' : '순서 변경은 등록순에서'}</span>
      </div>
      {r.error && <p className="err">{r.error}</p>}
      <div className={`tile-grid tile-${size}`} style={{ ['--tint' as string]: tint }}>
        {shown.map((c) => {
          const x = byCode.get(c);
          const s = x ? Math.sign(x.change) : 0;
          const pos = rangePos(x);
          return (
            <button
              type="button"
              key={c}
              ref={auto.on && sel === c ? curTile : undefined}
              className={`tile ${upDown(s)} ${sel === c ? 'sel' : ''} ${auto.on && sel === c ? 'auto-cur' : ''} ${dragFrom === c ? 'dragging' : dragFrom ? 'drop-ok' : ''}`}
              draggable={canDrag}
              onDragStart={(e) => (setDragFrom(c), (e.dataTransfer.effectAllowed = 'move'))}
              onDragOver={(e) => dragFrom && dragFrom !== c && e.preventDefault()}
              onDrop={(e) => (e.preventDefault(), dropOn(c), setDragFrom(null))}
              onDragEnd={() => setDragFrom(null)}
              style={{ ['--heat' as string]: tileHeat(x?.changePct) }}
              onClick={() => (setSel(c), setCodeAll(c))}
              onDoubleClick={() => useStore.getState().open('chart', { code: c, link: win.link })}
              onContextMenu={(e) => (e.preventDefault(), setSel(c), setMenu({ code: c, name: x?.name || '', x: e.clientX, y: e.clientY }))}
            >
              <span className="tile-head">
                <b className="tile-name">{x?.name || c}</b>
                <AlarmMark alerts={alerts.get(c)} />
                <NxtTag code={c} />
                {has('code') && <span className="mono muted small tile-code">{c}</span>}
              </span>
              <span className={`tile-price ${upDown(s)}`}>{fmtInt(x?.price)}</span>
              {has('change') && (
                <span className={`tile-chg ${upDown(s)}`}>
                  {s > 0 ? '▲' : s < 0 ? '▼' : ''} {fmtInt(x ? Math.abs(x.change) : undefined)} <b>{fmtPct(x?.changePct)}</b>
                </span>
              )}
              {has('bar') && (
                <span className="tile-range" title="저가 ~ 고가 사이의 현재가 위치">
                  <span className="tile-bar">{pos !== null && <i style={{ left: `${pos}%` }} />}</span>
                </span>
              )}
              {rowFields.length > 0 && (
                <span className={`tile-rows ${rowFields.length === 1 ? 'one' : ''}`}>
                  {rowFields.map((f) => (
                    <span key={f}>
                      <em>{FIELD_LABEL[f]}</em> {f === 'amount' ? fmtKrwBig(x?.amount) : fmtInt(x?.[f])}
                    </span>
                  ))}
                </span>
              )}
            </button>
          );
        })}
        {!codes.length && <p className="muted center tile-empty">이 그룹에 종목이 없습니다. 위 검색칸에서 종목을 추가하세요.</p>}
      </div>
      {menu && group && <WatchMenu at={menu} groupId={group.id} link={win.link} onClose={() => setMenu(null)} onRemoved={(c) => sel === c && setSel(null)} />}
    </div>
  );
}
