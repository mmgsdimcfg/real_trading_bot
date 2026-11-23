// Update log
// - [2026-11-06] type=feat (List windows: watchlist (add with the symbol search, remove, move up / down, up to 60 codes polled every 3 s with
//   the multi-symbol price TR - 30 codes per call), rankings (volume / amount / rise / fall / market cap, all / KOSPI / KOSDAQ, every 10 s),
//   indices (KOSPI / KOSDAQ / KOSPI200 with advancers / decliners and a daily chart), news titles of the symbol (every 60 s). A click on a
//   row sends the code to the windows of the same link colour; a double click opens a chart)
// - [2026-11-06] type=feat (Watchlist manager: groups as tabs (add, rename by double click, delete) shared by every watchlist window and
//   kept in their own storage; add / delete / move up-down / drag to reorder; keys Delete, Enter, arrows, Alt+arrows. Double click or
//   Enter shows the code in every open symbol window (any link colour). WatchStar: the star button of the chart / current price windows)
// - [2026-11-12] type=feat (NXT mark on watchlist and ranking rows; the watchlist toolbar says NXT prices are shown outside KRX hours)
// - [2026-11-19] type=feat (Watchlist '폴더' button opens the watchlist folder (picks files there become MMDD tabs); tabs made from a
//   picks file name it in their hover title)
// - [2026-11-20] type=feat (Watchlist rows: one click shows the symbol in every open symbol window (was a double click; a click only
//   selected the row / sent it to the same link colour) - user 2026-11-20)
// - [2026-11-20] type=fix (drop-downs inside the window (지표 / 기간 설정 / 관심종목 ☆) float over the app (FloatPop) and open toward the
//   side with room instead of being cut off at the window edge)

// - [2026-11-20] type=fix (☆ group menu closes on a click outside it or Esc)
// - [2026-11-20] type=feat (Watchlist (user 2026-11-20): group tabs can be dragged to a new order; 자동 보기 - the group's symbols are shown
//   in every open symbol window one after another every N s (3 - 600, kept per PC), the current row framed by a thin line; only one
//   auto view runs - starting it here stops the timeline's and the other way round (AUTO_VIEW_EVENT))
// - [2026-11-23] type=feat (Watchlist rows: right-click menu (components/WatchMenu) - open windows, copy / move to another tab, delete)
import { useEffect, useMemo, useRef, useState } from 'react';
import { LINK_COLORS, newGroupId, useStore, WATCH_MAX, type Win } from '../store';
import { useCall, useOutsideClose } from '../hooks';
import { AUTO_MAX_SEC, AUTO_MIN_SEC, autoViewMayResume, autoViewStarted, clampSec, nextCode, onAutoViewStarted } from '../../../bot/renderer/src/components/AutoView';

const WATCH_AUTO_KEY = 'autobothtsh.watchAuto';
interface WatchAutoCfg {
  on: boolean;
  sec: number;
}
function loadWatchAuto(): WatchAutoCfg {
  try {
    const v = JSON.parse(localStorage.getItem(WATCH_AUTO_KEY) || '{}') as Partial<WatchAutoCfg>;
    return { on: v.on === true, sec: clampSec(v.sec) };
  } catch {
    return { on: false, sec: 10 };
  }
}
import { SymbolSearch } from '../components/SymbolSearch';
import { NxtTag } from '../components/NxtTag';
import { FloatPop } from '../components/FloatPop';
import { WatchMenu, type MenuAt } from '../components/WatchMenu';
import { PriceChart } from '../components/PriceChart';
import type { RankRow, WatchRow } from '../../../shared/kis';
import type { ChartRes, IndexQuote, NewsRow, RankReq, RankType } from '../../../shared/market';
import { INDEXES } from '../../../shared/market';
import { fmtInt, fmtKrwBig, fmtNum, fmtPct, signOf, upDown } from '../lib';

function useRowActions(win: Win) {
  const { setCode, open } = useStore.getState();
  return {
    pick: (code: string) => setCode(win.id, code),
    openChart: (code: string) => open('chart', { code, link: win.link }),
  };
}

function LinkHint({ win }: { win: Win }) {
  return win.link ? (
    <span className="muted small">
      <i className="dot" style={{ background: LINK_COLORS[win.link] }} /> 클릭 → 같은 색 창에 표시 · 더블클릭 → 차트
    </span>
  ) : (
    <span className="muted small">제목의 색 점으로 연결 그룹을 고르면 클릭한 종목이 그 그룹 창에 표시됩니다 · 더블클릭 → 차트</span>
  );
}

/** watchlist manager: groups (tabs) shared by every watchlist window; click = link colour, double click = every open window */
export function WatchWindow({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const groups = useStore((s) => s.watch);
  const setWatch = useStore((s) => s.setWatch);
  const setCodeAll = useStore((s) => s.setCodeAll);
  const group = groups.find((g) => g.id === win.opts.group) || groups[0];
  const codes = group?.codes || [];
  const r = useCall<WatchRow[]>('multi', [codes], 3000, codes.length > 0);
  const act = useRowActions(win);
  const [sel, setSel] = useState<string | null>(null);
  const [rename, setRename] = useState<string | null>(null);
  const [dragFrom, setDragFrom] = useState<string | null>(null);
  const [dragTab, setDragTab] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuAt | null>(null);
  // [2026-11-20] 자동 보기
  const [auto, setAuto] = useState<WatchAutoCfg>(loadWatchAuto);
  const [secText, setSecText] = useState(String(auto.sec));
  const saveAuto = (p: Partial<WatchAutoCfg>) => {
    setAuto((cur) => {
      const next = { ...cur, ...p };
      try {
        localStorage.setItem(WATCH_AUTO_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
    if (p.on) autoViewStarted('watch');
  };
  useEffect(() => onAutoViewStarted('watch', () => saveAuto({ on: false })), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (auto.on && !autoViewMayResume('watch')) saveAuto({ on: false });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const codesKey = codes.join(',');
  useEffect(() => {
    if (!auto.on || !codes.length) return;
    // the first step at once when nothing of this group is selected, then one step per interval (a click restarts the interval)
    const step = () => {
      const n = nextCode(codes, sel);
      if (n) (setSel(n), setCodeAll(n));
    };
    if (!sel || !codes.includes(sel)) {
      step();
      return;
    }
    const t = setTimeout(step, auto.sec * 1000);
    return () => clearTimeout(t);
  }, [auto.on, auto.sec, sel, codesKey]); // eslint-disable-line react-hooks/exhaustive-deps
  // keep the shown row in view
  const curRow = useRef<HTMLTableRowElement | null>(null);
  useEffect(() => {
    if (auto.on) curRow.current?.scrollIntoView({ block: 'nearest' });
  }, [auto.on, sel]);
  const moveGroup = (from: string, toId: string) => {
    const a = groups.findIndex((g) => g.id === from);
    const b = groups.findIndex((g) => g.id === toId);
    if (a < 0 || b < 0 || a === b) return;
    const next = groups.slice();
    const [g] = next.splice(a, 1);
    next.splice(b, 0, g);
    setWatch(next);
  };
  const byCode = useMemo(() => new Map((r.data || []).map((x) => [x.code, x])), [r.data]);

  const saveCodes = (c: string[]) => group && setWatch(groups.map((g) => (g.id === group.id ? { ...g, codes: c } : g)));
  const pickGroup = (id: string) => (setOpts(win.id, { group: id }), setSel(null));
  const addGroup = () => {
    const id = newGroupId();
    let n = groups.length + 1;
    while (groups.some((g) => g.name === `관심 ${n}`)) n++;
    setWatch([...groups, { id, name: `관심 ${n}`, codes: [] }]);
    pickGroup(id);
  };
  const delGroup = () => {
    if (!group || groups.length < 2) return;
    if (!window.confirm(`'${group.name}' 그룹(${codes.length}종목)을 삭제할까요?`)) return;
    setWatch(groups.filter((g) => g.id !== group.id));
    pickGroup(groups.find((g) => g.id !== group.id)!.id);
  };
  const doRename = () => {
    const name = (rename || '').trim();
    if (group && name) setWatch(groups.map((g) => (g.id === group.id ? { ...g, name: name.slice(0, 20) } : g)));
    setRename(null);
  };
  const move = (code: string, to: number) => {
    const i = codes.indexOf(code);
    if (i < 0 || to < 0 || to >= codes.length || to === i) return;
    const c = codes.slice();
    c.splice(i, 1);
    c.splice(to, 0, code);
    saveCodes(c);
  };
  const remove = (code: string) => {
    const i = codes.indexOf(code);
    saveCodes(codes.filter((c) => c !== code));
    setSel(codes[i + 1] || codes[i - 1] || null);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (!sel) return;
    const i = codes.indexOf(sel);
    if (e.key === 'Delete') remove(sel);
    else if (e.key === 'Enter') setCodeAll(sel);
    else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const j = i + (e.key === 'ArrowUp' ? -1 : 1);
      if (e.altKey) move(sel, j);
      else if (codes[j]) (setSel(codes[j]), act.pick(codes[j]));
    }
  };

  return (
    <div className="list-win watch-win" tabIndex={0} onKeyDown={onKey}>
      <div className="watch-groups">
        {groups.map((g) =>
          rename !== null && g.id === group?.id ? (
            <input key={g.id} autoFocus className="watch-rename" value={rename} maxLength={20} onChange={(e) => setRename(e.target.value)} onBlur={doRename} onKeyDown={(e) => (e.stopPropagation(), e.key === 'Enter' ? doRename() : e.key === 'Escape' && setRename(null))} />
          ) : (
            <button
              key={g.id}
              className={`watch-tab ${g.id === group?.id ? 'on' : ''} ${dragTab && dragTab !== g.id ? 'drop-ok' : ''}`}
              draggable
              onDragStart={(e) => (setDragTab(g.id), (e.dataTransfer.effectAllowed = 'move'))}
              onDragOver={(e) => dragTab && e.preventDefault()}
              onDrop={(e) => (e.preventDefault(), dragTab && moveGroup(dragTab, g.id), setDragTab(null))}
              onDragEnd={() => setDragTab(null)}
              onClick={() => pickGroup(g.id)} onDoubleClick={() => (pickGroup(g.id), setRename(g.name))} title={g.src ? `${g.src} 에서 자동 등록 (파일이 바뀌면 다시 반영) · 더블클릭: 이름 바꾸기 · 끌어서 순서 변경` : '더블클릭: 이름 바꾸기 · 끌어서 순서 변경'}>
              {g.name} <small>{g.codes.length}</small>
            </button>
          ),
        )}
        <button className="mini" onClick={addGroup} title="새 관심 그룹">
          +
        </button>
        <span className="watch-gtools">
          <button className="mini" onClick={() => group && setRename(group.name)} title="그룹 이름 바꾸기">
            이름
          </button>
          <button className="mini" onClick={delGroup} disabled={groups.length < 2} title="그룹 삭제">
            그룹 삭제
          </button>
          <button
            className="mini"
            onClick={() => void window.hts.watchOpenDir().then(() => useStore.getState().syncPicks())}
            title="관심종목 폴더 열기 - 이 폴더에 YYYYMMDD_picks.txt (한 줄에 '종목코드,종목명') 를 넣으면 MMDD 탭이 자동으로 생깁니다 (30초마다 확인)"
          >
            폴더
          </button>
        </span>
      </div>
      <div className="win-tb">
        <SymbolSearch compact placeholder="종목 추가" onPick={(c) => !codes.includes(c) && codes.length < WATCH_MAX && saveCodes([...codes, c])} />
        <button className="mini" disabled={!sel} onClick={() => sel && move(sel, codes.indexOf(sel) - 1)} title="위로 (Alt+↑)">
          ↑
        </button>
        <button className="mini" disabled={!sel} onClick={() => sel && move(sel, codes.indexOf(sel) + 1)} title="아래로 (Alt+↓)">
          ↓
        </button>
        <button className="mini" disabled={!sel} onClick={() => sel && remove(sel)} title="삭제 (Delete)">
          삭제
        </button>
        <span className="muted small" title="NXT 표시 종목은 KRX 장이 닫힌 시간(09:00 이전, 15:40 이후)에 NXT 시세로 표시합니다">
          {codes.length}/{WATCH_MAX}종목 · 3초마다 · 장외 시간은 NXT 시세
        </span>
        <span className="watch-auto" title={`이 그룹 종목을 ${auto.sec}초마다 차례로 열린 종목 창에 표시 (종목을 누르면 그 종목부터 이어감). 타임라인 자동 보기와 동시에 실행되지 않습니다`}>
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
          초{auto.on && codes.length ? ` · ${sel && codes.includes(sel) ? codes.indexOf(sel) + 1 : '-'}/${codes.length}` : ''}
        </span>
        <span className="muted small watch-hint">클릭·Enter → 열린 창 모두 · 끌어서 순서 변경 · 우클릭 → 메뉴</span>
      </div>
      {r.error && <p className="err">{r.error}</p>}
      <div className="tab-scroll">
        <table className="grid hover">
          <thead>
            <tr>
              <th>종목</th>
              <th className="num">현재가</th>
              <th className="num">대비</th>
              <th className="num">등락률</th>
              <th className="num">거래량</th>
              <th className="num">거래대금</th>
              <th className="num">고가</th>
              <th className="num">저가</th>
            </tr>
          </thead>
          <tbody>
            {codes.map((c) => {
              const x = byCode.get(c);
              const s = x ? Math.sign(x.change) : 0;
              return (
                <tr
                  key={c}
                  ref={auto.on && sel === c ? curRow : undefined}
                  className={`${sel === c ? 'sel' : ''} ${auto.on && sel === c ? 'auto-cur' : ''} ${dragFrom && dragFrom !== c ? 'drop-ok' : ''}`}
                  draggable
                  onDragStart={(e) => (setDragFrom(c), (e.dataTransfer.effectAllowed = 'move'))}
                  onDragOver={(e) => dragFrom && e.preventDefault()}
                  onDrop={(e) => (e.preventDefault(), dragFrom && move(dragFrom, codes.indexOf(c)), setDragFrom(null))}
                  onDragEnd={() => setDragFrom(null)}
                  onClick={() => (setSel(c), setCodeAll(c))}
                  onContextMenu={(e) => (e.preventDefault(), setSel(c), setMenu({ code: c, name: x?.name || '', x: e.clientX, y: e.clientY }))}
                >
                  <td>
                    <b>{x?.name || c}</b> <span className="mono muted small">{c}</span> <NxtTag code={c} />
                  </td>
                  <td className={`num ${upDown(s)}`}>{fmtInt(x?.price)}</td>
                  <td className={`num ${upDown(s)}`}>{fmtInt(x?.change)}</td>
                  <td className={`num ${upDown(s)}`}>{fmtPct(x?.changePct)}</td>
                  <td className="num">{fmtInt(x?.volume)}</td>
                  <td className="num">{fmtKrwBig(x?.amount)}</td>
                  <td className="num">{fmtInt(x?.high)}</td>
                  <td className="num">{fmtInt(x?.low)}</td>
                </tr>
              );
            })}
            {!codes.length && (
              <tr>
                <td colSpan={8} className="muted center">
                  위 검색칸에서 종목을 추가하세요.
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

/** ☆ button: add / remove a code in the watchlist groups */
export function WatchStar({ code }: { code: string }) {
  const groups = useStore((s) => s.watch);
  const toggle = useStore((s) => s.toggleWatch);
  const [open, setOpen] = useState(false);
  const ref = useOutsideClose<HTMLSpanElement>(open, () => setOpen(false));
  const inAny = groups.some((g) => g.codes.includes(code));
  return (
    <span className="dropdown" ref={ref} onMouseLeave={() => setOpen(false)}>
      <button className={`mini star ${inAny ? 'in' : ''}`} onClick={() => setOpen(!open)} title="관심종목 그룹에 추가 / 빼기">
        {inAny ? '★' : '☆'}
      </button>
      {open && (
        <FloatPop className="dd-menu" onMouseDown={(e) => e.stopPropagation()}>
          {groups.map((g) => (
            <label key={g.id} className="check dd-check">
              <input type="checkbox" checked={g.codes.includes(code)} disabled={!g.codes.includes(code) && g.codes.length >= WATCH_MAX} onChange={() => toggle(g.id, code)} /> {g.name}
            </label>
          ))}
        </FloatPop>
      )}
    </span>
  );
}


const RANK_TABS: [RankType, string][] = [
  ['volume', '거래량'],
  ['amount', '거래대금'],
  ['rise', '상승률'],
  ['fall', '하락률'],
  ['cap', '시가총액'],
];

export function RankWindow({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const type = (win.opts.type as RankType) || 'volume';
  const market = (win.opts.market as RankReq['market']) || '0000';
  const r = useCall<RankRow[]>('rank', [{ type, market }], 10000);
  const act = useRowActions(win);
  return (
    <div className="list-win">
      <div className="win-tb">
        <span className="seg">
          {RANK_TABS.map(([k, l]) => (
            <button key={k} className={type === k ? 'on' : ''} onClick={() => setOpts(win.id, { type: k })}>
              {l}
            </button>
          ))}
        </span>
        <select value={market} onChange={(e) => setOpts(win.id, { market: e.target.value })}>
          <option value="0000">전체</option>
          <option value="0001">코스피</option>
          <option value="1001">코스닥</option>
        </select>
        <span className="muted small">10초마다</span>
        <LinkHint win={win} />
      </div>
      {r.error && <p className="err">{r.error}</p>}
      <div className="tab-scroll">
        <table className="grid hover">
          <thead>
            <tr>
              <th className="num">순위</th>
              <th>종목</th>
              <th className="num">현재가</th>
              <th className="num">등락률</th>
              <th className="num">거래량</th>
              <th className="num">{type === 'cap' ? '시가총액' : type === 'volume' || type === 'amount' ? '거래증가율' : '거래대금'}</th>
            </tr>
          </thead>
          <tbody>
            {(r.data || []).map((x) => {
              const s = Math.sign(x.change);
              return (
                <tr key={x.code} onClick={() => act.pick(x.code)} onDoubleClick={() => act.openChart(x.code)}>
                  <td className="num">{x.rank}</td>
                  <td>
                    <b>{x.name}</b> <span className="mono muted small">{x.code}</span> <NxtTag code={x.code} />
                  </td>
                  <td className={`num ${upDown(s)}`}>{fmtInt(x.price)}</td>
                  <td className={`num ${upDown(s)}`}>{fmtPct(x.changePct)}</td>
                  <td className="num">{fmtInt(x.volume)}</td>
                  <td className="num">{type === 'cap' ? fmtKrwBig((x.extra || 0) * 1e8) : type === 'volume' || type === 'amount' ? (x.extra !== undefined ? `${fmtNum(x.extra)}%` : '-') : fmtKrwBig(x.amount)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function IndexCard({ code, on, onPick }: { code: string; on: boolean; onPick: () => void }) {
  const r = useCall<IndexQuote>('index', [code], 10000);
  const x = r.data;
  const s = x ? signOf(x.sign, x.change) : 0;
  return (
    <button className={`idx-card ${on ? 'sel' : ''}`} onClick={onPick}>
      <span>{INDEXES.find((i) => i.code === code)?.name}</span>
      <b className={upDown(s)}>{x ? fmtNum(x.value) : '-'}</b>
      <small className={upDown(s)}>
        {x ? `${fmtNum(x.change)} ${fmtPct(x.changePct)}` : r.error || '…'}
      </small>
      {x?.up !== undefined && (
        <small className="muted">
          <span className="up">▲{fmtInt(x.up)}</span> <span className="down">▼{fmtInt(x.down)}</span> ={fmtInt(x.flat)}
        </small>
      )}
    </button>
  );
}

export function IndexWindow({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const sel = (win.opts.index as string) || '0001';
  const c = useCall<ChartRes>('chart', [{ code: sel, kind: 'day', n: 1, index: true }], 60000);
  const box = useRef<HTMLDivElement>(null);
  const [ch, setCh] = useState(220);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setCh(Math.max(120, el.clientHeight - 4)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div className="index-win">
      <div className="idx-row">
        {INDEXES.map((i) => (
          <IndexCard key={i.code} code={i.code} on={sel === i.code} onPick={() => setOpts(win.id, { index: i.code })} />
        ))}
      </div>
      <div className="idx-chart" ref={box}>
        {c.data?.bars.length ? (
          <PriceChart bars={c.data.bars} kind="day" n={1} indicators={['ma']} trades={[]} date="" mode="index" axis="right" left={4} right={64} height={ch} interactive tooltip={false} />
        ) : (
          <div className="empty small">{c.error || '지수 차트 불러오는 중…'}</div>
        )}
      </div>
    </div>
  );
}

export function NewsWindow({ win }: { win: Win }) {
  const r = useCall<NewsRow[]>('news', [win.code], 60000);
  return (
    <div className="list-win">
      {r.error && <p className="err">{r.error}</p>}
      <div className="tab-scroll">
        <table className="grid">
          <tbody>
            {(r.data || []).map((n, i) => (
              <tr key={i}>
                <td className="mono muted small nowrap">
                  {n.date.slice(4, 6)}.{n.date.slice(6, 8)} {n.time.slice(0, 2)}:{n.time.slice(2, 4)}
                </td>
                <td className="wrap">{n.title}</td>
                <td className="muted small nowrap">{n.source}</td>
              </tr>
            ))}
            {!r.loading && !r.data?.length && !r.error && (
              <tr>
                <td className="muted center">뉴스가 없습니다.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
