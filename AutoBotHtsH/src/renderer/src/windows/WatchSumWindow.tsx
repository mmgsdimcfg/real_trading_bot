// Update log
// - [2026-11-24] type=feat (관심종목(요약) (user 2026-11-24): the watchlist groups as a narrow list of three columns only - 종목명, 현재가,
//   등락률. Same groups and 3 s polling as the 관심종목 window (groups are edited there); the group is picked from a drop-down to keep the
//   window narrow. A click shows the symbol in every open symbol window, a double click opens a chart, right-click = the watchlist menu
//   (open windows / copy / move / delete), ↑ ↓ move the selection. Column titles sort (high to low / low to high / registered order),
//   kept with the window (opts.group / sortKey / sortDir))

import { useMemo, useState } from 'react';
import { useStore, WATCH_MAX, type Win } from '../store';
import { useCall } from '../hooks';
import { NxtTag } from '../components/NxtTag';
import { WatchMenu, type MenuAt } from '../components/WatchMenu';
import { AlarmMark, useTodayAlerts } from '../components/AlarmMark';
import type { WatchRow } from '../../../shared/kis';
import { fmtInt, fmtPct, upDown } from '../lib';
import { nextWatchSort, sortWatch, watchSortOf, type WatchSortKey } from '../watchSort';

/** the columns of the summary list (the sort keys of their titles) */
export const SUM_SORT_KEYS: WatchSortKey[] = ['name', 'price', 'changePct'];
/** the saved sort, kept only for the three columns shown */
export function sumSortOf(opts: Record<string, unknown>) {
  const s = watchSortOf(opts.sortKey, opts.sortDir);
  return s && SUM_SORT_KEYS.includes(s.key) ? s : null;
}

export function WatchSumWindow({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const groups = useStore((s) => s.watch);
  const setCodeAll = useStore((s) => s.setCodeAll);
  const group = groups.find((g) => g.id === win.opts.group) || groups[0];
  const codes = group?.codes || [];
  const sort = sumSortOf(win.opts);
  const r = useCall<WatchRow[]>('multi', [codes], 3000, codes.length > 0);
  const [sel, setSel] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuAt | null>(null);
  const alerts = useTodayAlerts();
  const byCode = useMemo(() => new Map((r.data || []).map((x) => [x.code, x])), [r.data]);
  const shown = useMemo(() => sortWatch(codes, byCode, sort), [codes, byCode, sort?.key, sort?.dir]); // eslint-disable-line react-hooks/exhaustive-deps
  const ups = shown.filter((c) => (byCode.get(c)?.change || 0) > 0).length;
  const downs = shown.filter((c) => (byCode.get(c)?.change || 0) < 0).length;
  const sortBy = (key: WatchSortKey) => {
    const n = nextWatchSort(sort, key);
    setOpts(win.id, { sortKey: n?.key ?? null, sortDir: n?.dir ?? null });
  };
  const mark = (key: WatchSortKey) => (sort?.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : '');
  const pick = (c: string) => (setSel(c), setCodeAll(c));
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    const i = sel ? shown.indexOf(sel) : -1;
    const n = shown[i < 0 ? 0 : i + (e.key === 'ArrowUp' ? -1 : 1)];
    if (n) pick(n);
  };

  return (
    <div className="list-win watch-win sum-win" tabIndex={0} onKeyDown={onKey}>
      <div className="win-tb sum-tb">
        <select value={group?.id || ''} onChange={(e) => (setOpts(win.id, { group: e.target.value }), setSel(null))} title="관심 그룹 (그룹 편집은 관심종목 창에서)">
          {groups.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name} ({g.codes.length})
            </option>
          ))}
        </select>
        <span className="muted small" title={`${codes.length}/${WATCH_MAX}종목 · 3초마다 · 장외 시간은 NXT 시세`}>
          <span className="up">▲{ups}</span> <span className="down">▼{downs}</span>
        </span>
      </div>
      {r.error && <p className="err">{r.error}</p>}
      <div className="tab-scroll">
        <table className="grid hover sum-grid">
          <thead>
            <tr>
              <th className="sortable" onClick={() => sortBy('name')} title="종목명으로 정렬 (가나다순 → 역순 → 등록순)">
                종목명{mark('name')}
              </th>
              <th className="num sortable" onClick={() => sortBy('price')} title="누르면 정렬 (높은순 → 낮은순 → 등록순)">
                현재가{mark('price')}
              </th>
              <th className="num sortable" onClick={() => sortBy('changePct')} title="누르면 정렬 (높은순 → 낮은순 → 등록순)">
                등락률{mark('changePct')}
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => {
              const x = byCode.get(c);
              const s = x ? Math.sign(x.change) : 0;
              return (
                <tr
                  key={c}
                  className={sel === c ? 'sel' : ''}
                  onClick={() => pick(c)}
                  onDoubleClick={() => useStore.getState().open('chart', { code: c, link: win.link })}
                  onContextMenu={(e) => (e.preventDefault(), setSel(c), setMenu({ code: c, name: x?.name || '', x: e.clientX, y: e.clientY }))}
                  title={c}
                >
                  <td className="sum-name">
                    <b>{x?.name || c}</b> <NxtTag code={c} /> <AlarmMark alerts={alerts.get(c)} />
                  </td>
                  <td className={`num ${upDown(s)}`}>{fmtInt(x?.price)}</td>
                  <td className={`num ${upDown(s)}`}>{fmtPct(x?.changePct)}</td>
                </tr>
              );
            })}
            {!codes.length && (
              <tr>
                <td colSpan={3} className="muted center">
                  이 그룹에 종목이 없습니다. 관심종목 창에서 추가하세요.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {menu && group && <WatchMenu at={menu} groupId={group.id} link={win.link} onClose={() => setMenu(null)} onRemoved={(c) => sel === c && setSel(null)} />}
    </div>
  );
}
