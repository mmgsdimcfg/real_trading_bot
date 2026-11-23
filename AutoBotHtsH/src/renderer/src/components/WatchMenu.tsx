// Update log
// - [2026-11-23] type=feat (Right-click menu of a watchlist symbol (user 2026-11-23; list and tile windows): show in every open window,
//   open 차트 / 현재가 / 종목정보 / 뉴스, copy or move to another watchlist tab, copy the code, move to the top / bottom, delete from
//   this tab. Floats at the pointer (kept inside the app window), closes on a click outside or Esc. The group edits are pure
//   functions (copyToGroup / moveToGroup / removeFromGroup / moveInGroup) so both windows and the tests share them)

import { useLayoutEffect, useState } from 'react';
import { useStore, WATCH_MAX, type WatchGroup, type WinType } from '../store';
import { useOutsideClose } from '../hooks';
import { placePop } from './FloatPop';

export interface MenuAt {
  code: string;
  name: string;
  x: number;
  y: number;
}

/** why `code` cannot be added to group `g` ('' = it can) */
export function addBlock(g: WatchGroup, code: string): string {
  if (g.codes.includes(code)) return '이미 있음';
  if (g.codes.length >= WATCH_MAX) return '가득 참';
  return '';
}

export function copyToGroup(groups: WatchGroup[], toId: string, code: string): WatchGroup[] {
  return groups.map((g) => (g.id === toId && !addBlock(g, code) ? { ...g, codes: [...g.codes, code] } : g));
}

export function removeFromGroup(groups: WatchGroup[], fromId: string, code: string): WatchGroup[] {
  return groups.map((g) => (g.id === fromId ? { ...g, codes: g.codes.filter((c) => c !== code) } : g));
}

/** copy to `toId` and remove from `fromId`; nothing changes when the target cannot take it (it would only be lost) */
export function moveToGroup(groups: WatchGroup[], fromId: string, toId: string, code: string): WatchGroup[] {
  const to = groups.find((g) => g.id === toId);
  if (!to || fromId === toId || addBlock(to, code)) return groups;
  return removeFromGroup(copyToGroup(groups, toId, code), fromId, code);
}

/** `code` to the top / bottom of its group */
export function moveInGroup(groups: WatchGroup[], groupId: string, code: string, where: 'top' | 'bottom'): WatchGroup[] {
  return groups.map((g) => {
    if (g.id !== groupId || !g.codes.includes(code)) return g;
    const rest = g.codes.filter((c) => c !== code);
    return { ...g, codes: where === 'top' ? [code, ...rest] : [...rest, code] };
  });
}

const OPEN_WINS: [WinType, string][] = [
  ['chart', '차트'],
  ['quote', '현재가'],
  ['info', '종목정보'],
  ['news', '뉴스'],
];

export function WatchMenu({ at, groupId, link, onClose, onRemoved }: { at: MenuAt; groupId: string; link: number; onClose: () => void; onRemoved?: (code: string) => void }) {
  const groups = useStore((s) => s.watch);
  const [sub, setSub] = useState<'copy' | 'move' | null>(null);
  const ref = useOutsideClose<HTMLDivElement>(true, onClose);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { left, top } = placePop({ left: at.x, right: at.x, top: at.y, bottom: at.y }, el.offsetWidth, el.offsetHeight, window.innerWidth, window.innerHeight);
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
    el.style.visibility = 'visible';
  }, [at.x, at.y, sub]); // eslint-disable-line react-hooks/exhaustive-deps
  const st = useStore.getState();
  const done = (f?: () => void) => () => (f?.(), onClose());
  const others = groups.filter((g) => g.id !== groupId);
  const here = groups.find((g) => g.id === groupId);
  const idx = here ? here.codes.indexOf(at.code) : -1;
  const pickGroup = (toId: string) => {
    if (sub === 'copy') st.setWatch(copyToGroup(groups, toId, at.code));
    else {
      st.setWatch(moveToGroup(groups, groupId, toId, at.code));
      onRemoved?.(at.code);
    }
    onClose();
  };

  return (
    <div ref={ref} className="ctx-menu" style={{ position: 'fixed', left: 0, top: 0, visibility: 'hidden' }} onMouseDown={(e) => e.stopPropagation()} onContextMenu={(e) => e.preventDefault()}>
      <div className="ctx-head">
        <b>{at.name || at.code}</b> <span className="mono muted small">{at.code}</span>
      </div>
      {sub ? (
        <>
          <button type="button" className="ctx-item" onClick={() => setSub(null)}>
            ‹ {sub === 'copy' ? '복사할 탭 선택' : '이동할 탭 선택'}
          </button>
          <div className="ctx-sep" />
          {others.map((g) => {
            const why = addBlock(g, at.code);
            return (
              <button type="button" key={g.id} className="ctx-item" disabled={!!why} onClick={() => pickGroup(g.id)}>
                {g.name} <small>{why || `${g.codes.length}종목`}</small>
              </button>
            );
          })}
          {!others.length && <div className="ctx-note">다른 탭이 없습니다</div>}
        </>
      ) : (
        <>
          <button type="button" className="ctx-item" onClick={done(() => st.setCodeAll(at.code))}>
            열린 창 모두에 표시
          </button>
          {OPEN_WINS.map(([t, label]) => (
            <button type="button" key={t} className="ctx-item" onClick={done(() => st.open(t, { code: at.code, link }))}>
              {label}
            </button>
          ))}
          <div className="ctx-sep" />
          <button type="button" className="ctx-item" disabled={!others.length} onClick={() => setSub('copy')}>
            다른 탭으로 복사 <small>›</small>
          </button>
          <button type="button" className="ctx-item" disabled={!others.length} onClick={() => setSub('move')}>
            다른 탭으로 이동 <small>›</small>
          </button>
          <button type="button" className="ctx-item" onClick={done(() => void navigator.clipboard?.writeText(at.code).catch(() => undefined))}>
            종목코드 복사
          </button>
          <div className="ctx-sep" />
          <button type="button" className="ctx-item" disabled={idx <= 0} onClick={done(() => st.setWatch(moveInGroup(groups, groupId, at.code, 'top')))}>
            맨 위로
          </button>
          <button type="button" className="ctx-item" disabled={idx < 0 || idx === (here?.codes.length || 0) - 1} onClick={done(() => st.setWatch(moveInGroup(groups, groupId, at.code, 'bottom')))}>
            맨 아래로
          </button>
          <div className="ctx-sep" />
          <button type="button" className="ctx-item danger" onClick={done(() => (st.setWatch(removeFromGroup(groups, groupId, at.code)), onRemoved?.(at.code)))}>
            이 탭에서 삭제
          </button>
        </>
      )}
    </div>
  );
}
