// Update log
// - [2026-11-23] type=feat (관심종목(타일) (user 2026-11-23): the watchlist groups as tiles - name, code, NXT mark, current price, change
//   and rate, the day's open / high / low with a low-high bar marking the price, volume and traded value; the tile is tinted by the
//   change rate. Same groups and 3 s polling as the 관심종목 window (groups are edited there); a click shows the symbol in every open
//   symbol window, a double click opens a chart. Order: as registered / by rate / by traded value (kept with the window))
// - [2026-11-23] type=feat (자동 보기 on the tile window (user 2026-11-23): the group's tiles are shown one after another every N s in
//   every open symbol window, in the order on screen (the chosen sort), the current tile framed and kept in view. Its own setting
//   (autobothtsh.watchTileAuto); one auto view at a time - starting it stops the timeline's / the list window's and the other way round)

import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore, type Win } from '../store';
import { useCall } from '../hooks';
import { NxtTag } from '../components/NxtTag';
import { WatchMenu, type MenuAt } from '../components/WatchMenu';
import type { WatchRow } from '../../../shared/kis';
import { fmtInt, fmtKrwBig, fmtPct, upDown } from '../lib';
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

export type TileSort = 'list' | 'pctDesc' | 'pctAsc' | 'amount';
const SORTS: [TileSort, string][] = [
  ['list', '등록순'],
  ['pctDesc', '등락률 높은순'],
  ['pctAsc', '등락률 낮은순'],
  ['amount', '거래대금순'],
];

/** codes in the chosen order; symbols without a price yet keep their place at the end (list order among themselves) */
export function sortTiles(codes: string[], rows: Map<string, WatchRow>, sort: TileSort): string[] {
  if (sort === 'list') return codes;
  const val = (c: string): number | null => {
    const x = rows.get(c);
    if (!x) return null;
    const v = sort === 'amount' ? x.amount : x.changePct;
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  };
  const dir = sort === 'pctAsc' ? 1 : -1;
  return codes
    .map((c, i) => ({ c, i, v: val(c) }))
    .sort((a, b) => (a.v === null || b.v === null ? (a.v === null ? 1 : 0) - (b.v === null ? 1 : 0) || a.i - b.i : (a.v - b.v) * dir || a.i - b.i))
    .map((x) => x.c);
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
  const sort = (SORTS.some(([k]) => k === win.opts.sort) ? win.opts.sort : 'list') as TileSort;
  const r = useCall<WatchRow[]>('multi', [codes], 3000, codes.length > 0);
  const [sel, setSel] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuAt | null>(null);
  const byCode = useMemo(() => new Map((r.data || []).map((x) => [x.code, x])), [r.data]);
  const shown = useMemo(() => sortTiles(codes, byCode, sort), [codes, byCode, sort]);
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
        <select value={sort} onChange={(e) => setOpts(win.id, { sort: e.target.value })} title="타일 순서">
          {SORTS.map(([k, label]) => (
            <option key={k} value={k}>
              {label}
            </option>
          ))}
        </select>
        <span className="muted small" title="NXT 표시 종목은 KRX 장이 닫힌 시간(09:00 이전, 15:40 이후)에 NXT 시세로 표시합니다">
          {codes.length}종목 · <span className="up">상승 {ups}</span> · <span className="down">하락 {downs}</span> · 3초마다
        </span>
        <span className="watch-auto" title={`이 그룹 종목을 화면에 보이는 순서대로 ${auto.sec}초마다 열린 종목 창에 표시 (타일을 누르면 그 종목부터 이어감). 타임라인 / 관심종목 창의 자동 보기와 동시에 실행되지 않습니다`}>
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
        <span className="muted small watch-hint">클릭 → 열린 창 모두 · 더블클릭 → 차트 · 우클릭 → 메뉴 · 종목 추가는 관심종목 창에서</span>
      </div>
      {r.error && <p className="err">{r.error}</p>}
      <div className="tile-grid">
        {shown.map((c) => {
          const x = byCode.get(c);
          const s = x ? Math.sign(x.change) : 0;
          const pos = rangePos(x);
          return (
            <button
              type="button"
              key={c}
              ref={auto.on && sel === c ? curTile : undefined}
              className={`tile ${upDown(s)} ${sel === c ? 'sel' : ''} ${auto.on && sel === c ? 'auto-cur' : ''}`}
              style={{ ['--heat' as string]: tileHeat(x?.changePct) }}
              onClick={() => (setSel(c), setCodeAll(c))}
              onDoubleClick={() => useStore.getState().open('chart', { code: c, link: win.link })}
              onContextMenu={(e) => (e.preventDefault(), setSel(c), setMenu({ code: c, name: x?.name || '', x: e.clientX, y: e.clientY }))}
            >
              <span className="tile-head">
                <b className="tile-name">{x?.name || c}</b>
                <NxtTag code={c} />
                <span className="mono muted small tile-code">{c}</span>
              </span>
              <span className={`tile-price ${upDown(s)}`}>{fmtInt(x?.price)}</span>
              <span className={`tile-chg ${upDown(s)}`}>
                {s > 0 ? '▲' : s < 0 ? '▼' : ''} {fmtInt(x ? Math.abs(x.change) : undefined)} <b>{fmtPct(x?.changePct)}</b>
              </span>
              <span className="tile-range" title="저가 ~ 고가 사이의 현재가 위치">
                <span className="tile-bar">{pos !== null && <i style={{ left: `${pos}%` }} />}</span>
              </span>
              <span className="tile-rows">
                <span>
                  <em>시가</em> {fmtInt(x?.open)}
                </span>
                <span>
                  <em>거래량</em> {fmtInt(x?.volume)}
                </span>
                <span>
                  <em>고가</em> {fmtInt(x?.high)}
                </span>
                <span>
                  <em>거래대금</em> {fmtKrwBig(x?.amount)}
                </span>
                <span>
                  <em>저가</em> {fmtInt(x?.low)}
                </span>
              </span>
            </button>
          );
        })}
        {!codes.length && <p className="muted center tile-empty">이 그룹에 종목이 없습니다. 관심종목 창에서 종목을 추가하세요.</p>}
      </div>
      {menu && group && <WatchMenu at={menu} groupId={group.id} link={win.link} onClose={() => setMenu(null)} onRemoved={(c) => sel === c && setSel(null)} />}
    </div>
  );
}
