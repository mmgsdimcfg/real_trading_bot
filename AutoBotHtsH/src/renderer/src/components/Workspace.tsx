// Update log
// - [2026-11-06] type=feat (MDI workspace: every HtsH window is a frame inside the main window - title bar (type icon, title, symbol, link
//   colour picker, minimize / maximize / close), drag to move, 8-direction resize, double-click title to maximize, click to focus (z-order).
//   Minimized windows sit in the tray at the bottom. The workspace size is reported to the store so arrange / maximize use the real area)
// - [2026-11-20] type=feat (AutoBotHtsH: bot-<tab> windows show the AutoBotH view (BotWindow); they have no link colour picker)
// - [2026-11-20] type=fix (link colour picker closes on a click outside it or Esc)

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { isBotWin, LINK_COLORS, WIN_TITLE, SYMBOL_WINS, useStore, type Win } from '../store';
import { ChartWindow } from '../windows/ChartWindow';
import { QuoteWindow } from '../windows/QuoteWindow';
import { InfoWindow } from '../windows/InfoWindow';
import { WatchWindow } from '../windows/WatchWindow';
import { WatchTileWindow } from '../windows/WatchTileWindow';
import { RankWindow } from '../windows/RankWindow';
import { IndexWindow } from '../windows/IndexWindow';
import { NewsWindow } from '../windows/NewsWindow';
import { BotWindow } from '../windows/BotWindow';
import { useOutsideClose, useSymbol } from '../hooks';
import { NxtMark } from './NxtTag';

const BODY: Record<string, (p: { win: Win }) => ReactNode> = {
  chart: ChartWindow,
  quote: QuoteWindow,
  info: InfoWindow,
  watch: WatchWindow,
  watchtile: WatchTileWindow,
  rank: RankWindow,
  index: IndexWindow,
  news: NewsWindow,
};
/** [2026-11-20] bot-<tab> windows -> BotWindow */
const bodyOf = (t: Win['type']) => BODY[t] || (t.startsWith('bot-') ? BotWindow : null);

type Edge = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

function LinkPicker({ win }: { win: Win }) {
  const setLink = useStore((s) => s.setLink);
  const [open, setOpen] = useState(false);
  const ref = useOutsideClose<HTMLSpanElement>(open, () => setOpen(false));
  return (
    <span className="link-pick" ref={ref} onMouseDown={(e) => e.stopPropagation()}>
      <button className="link-dot" style={{ background: LINK_COLORS[win.link] || 'transparent' }} title={win.link ? `연결 그룹 ${win.link} - 같은 색 창은 종목이 함께 바뀝니다` : '연결 안 함'} onClick={() => setOpen(!open)} />
      {open && (
        <span className="link-menu">
          {LINK_COLORS.map((c, i) => (
            <button key={i} className={`link-dot ${win.link === i ? 'on' : ''}`} style={{ background: c || 'transparent' }} title={i ? `그룹 ${i}` : '연결 안 함'} onClick={() => (setLink(win.id, i), setOpen(false))}>
              {i ? '' : '×'}
            </button>
          ))}
        </span>
      )}
    </span>
  );
}

function Frame({ win, active }: { win: Win; active: boolean }) {
  const { focus, move, resize, toggleMin, toggleMax, close } = useStore.getState();
  const area = useStore((s) => s.area);
  const sym = useSymbol(win.code);
  const drag = useRef<{ kind: 'move' | Edge; sx: number; sy: number; x: number; y: number; w: number; h: number } | null>(null);

  useEffect(() => {
    const mm = (e: MouseEvent) => {
      const d = drag.current;
      if (!d) return;
      const dx = e.clientX - d.sx;
      const dy = e.clientY - d.sy;
      if (d.kind === 'move') return move(win.id, d.x + dx, d.y + dy);
      let { x, y, w, h } = d;
      if (d.kind.includes('e')) w = d.w + dx;
      if (d.kind.includes('s')) h = d.h + dy;
      if (d.kind.includes('w')) (w = d.w - dx), (x = d.x + dx);
      if (d.kind.includes('n')) (h = d.h - dy), (y = d.y + dy);
      if (w < 260) (x -= d.kind.includes('w') ? 260 - w : 0), (w = 260);
      if (h < 160) (y -= d.kind.includes('n') ? 160 - h : 0), (h = 160);
      resize(win.id, w, h, x, y);
    };
    const mu = () => {
      drag.current = null;
      document.body.classList.remove('dragging');
    };
    window.addEventListener('mousemove', mm);
    window.addEventListener('mouseup', mu);
    return () => {
      window.removeEventListener('mousemove', mm);
      window.removeEventListener('mouseup', mu);
    };
  }, [win.id, move, resize]);

  const start = (kind: 'move' | Edge) => (e: React.MouseEvent) => {
    if (e.button !== 0 || (kind === 'move' && win.max)) return;
    e.preventDefault();
    e.stopPropagation();
    focus(win.id);
    drag.current = { kind, sx: e.clientX, sy: e.clientY, x: win.x, y: win.y, w: win.w, h: win.h };
    document.body.classList.add('dragging');
  };

  const Body = bodyOf(win.type);
  const style = win.max ? { left: 0, top: 0, width: area.w, height: area.h, zIndex: win.z } : { left: win.x, top: win.y, width: win.w, height: win.h, zIndex: win.z };
  const hasSym = SYMBOL_WINS.includes(win.type);
  return (
    <div className={`mdi-win ${active ? 'active' : ''} ${win.max ? 'max' : ''}`} style={style} onMouseDown={() => focus(win.id)}>
      <div className="mdi-title" onMouseDown={start('move')} onDoubleClick={() => toggleMax(win.id)} style={{ borderTopColor: LINK_COLORS[win.link] || undefined }}>
        {!isBotWin(win.type) && <LinkPicker win={win} />}
        <b>{WIN_TITLE[win.type]}</b>
        {hasSym && (
          <span className="mdi-sym">
            {sym?.name || ''} <span className="mono muted">{win.code}</span> {sym?.nxt && <NxtMark />}
          </span>
        )}
        <span className="mdi-btns" onMouseDown={(e) => e.stopPropagation()}>
          <button title="최소화" onClick={() => toggleMin(win.id)}>
            ─
          </button>
          <button title={win.max ? '이전 크기로' : '최대화'} onClick={() => toggleMax(win.id)}>
            {win.max ? '❐' : '□'}
          </button>
          <button className="close" title="닫기" onClick={() => close(win.id)}>
            ✕
          </button>
        </span>
      </div>
      <div className="mdi-body">
        {Body ? <Body win={win} /> : <div className="empty">알 수 없는 창</div>}
      </div>
      {!win.max && (['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'] as Edge[]).map((k) => <i key={k} className={`rz rz-${k}`} onMouseDown={start(k)} />)}
    </div>
  );
}

export function Workspace() {
  const wins = useStore((s) => s.wins);
  const active = useStore((s) => s.active);
  const setArea = useStore((s) => s.setArea);
  const focus = useStore((s) => s.focus);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setArea({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setArea({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, [setArea]);
  const minimized = wins.filter((w) => w.min);
  return (
    <div className="workspace-wrap">
      <div className="workspace" ref={ref}>
        {wins.filter((w) => !w.min).map((w) => (
          <Frame key={w.id} win={w} active={w.id === active} />
        ))}
        {!wins.length && <div className="ws-empty">위 도구 모음에서 창을 여세요 (차트 · 현재가 · 종목정보 · 관심종목 · 순위 · 지수 · 뉴스)</div>}
      </div>
      {minimized.length > 0 && (
        <div className="tray">
          {minimized.map((w) => (
            <button key={w.id} onClick={() => focus(w.id)} style={{ borderLeftColor: LINK_COLORS[w.link] || undefined }}>
              {WIN_TITLE[w.type]} {SYMBOL_WINS.includes(w.type) ? w.code : ''}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
