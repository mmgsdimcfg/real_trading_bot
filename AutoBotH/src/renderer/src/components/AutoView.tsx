/*
 * Update log
 * - [2026-11-20] type=feat - 자동 보기 on the timeline (user 2026-11-20; AutoBotHtsH first, then AutoBotH - same file in both apps:
 *   AutoBotH src/renderer/src/components, AutoBotHtsH src/bot/renderer/src/components). While checked, the timeline steps through the
 *   target symbols (대상: 전체 / 감시 중 / 보유) every N seconds (3 - 600, default 10), in the symbol picker's order, wrapping around.
 *   Each step is an ordinary symbol selection (in AutoBotHtsH the HTS windows follow it too). A symbol picked by hand stays for a full
 *   interval before the loop moves on. Settings kept per PC (localStorage); the loop runs only while the timeline is on screen
 */

import { useEffect, useState } from 'react';
import { useStore } from '../store';
import type { SymbolSummary } from '../../../shared/types';

export type AutoScope = 'all' | 'active' | 'held';
interface AutoCfg {
  on: boolean;
  sec: number;
  scope: AutoScope;
}
const KEY = 'autoboth.timelineAuto';
export const AUTO_MIN_SEC = 3;
export const AUTO_MAX_SEC = 600;
const SCOPE_LABEL: Record<AutoScope, string> = { all: '전체 종목', active: '감시 중', held: '보유 종목' };

function load(): AutoCfg {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || '{}') as Partial<AutoCfg>;
    return { on: v.on === true, sec: clampSec(v.sec), scope: v.scope === 'active' || v.scope === 'held' ? v.scope : 'all' };
  } catch {
    return { on: false, sec: 10, scope: 'all' };
  }
}
export const clampSec = (v: unknown) => {
  const n = Math.round(Number(v));
  return Number.isFinite(n) && n > 0 ? Math.min(AUTO_MAX_SEC, Math.max(AUTO_MIN_SEC, n)) : 10;
};

export const inScope = (s: SymbolSummary, scope: AutoScope) =>
  scope === 'all' ? true : scope === 'active' ? s.watch === 'active' : !!(s.position || s.held);

/** next code after `cur` among `codes` (wraps; the first one when `cur` is not in the list) */
export function nextCode(codes: string[], cur: string | null): string | null {
  if (!codes.length) return null;
  const i = cur ? codes.indexOf(cur) : -1;
  return codes[(i + 1) % codes.length];
}

export function AutoView() {
  const [cfg, setCfg] = useState<AutoCfg>(load);
  const [secText, setSecText] = useState(String(cfg.sec));
  const selected = useStore((s) => s.selected);
  const snap = useStore((s) => s.snapshot);
  const codes = (snap?.symbols || []).filter((s) => inScope(s, cfg.scope)).map((s) => s.code);
  const save = (p: Partial<AutoCfg>) => {
    const next = { ...cfg, ...p };
    setCfg(next);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* storage unavailable */
    }
  };
  // one step per interval; a new selection (by hand or by the loop) starts the next interval
  const key = codes.join(',');
  useEffect(() => {
    if (!cfg.on) return;
    const t = setTimeout(() => {
      const st = useStore.getState();
      const list = (st.snapshot?.symbols || []).filter((s) => inScope(s, cfg.scope)).map((s) => s.code);
      const n = nextCode(list, st.selected);
      if (n && n !== st.selected) st.select(n);
    }, cfg.sec * 1000);
    return () => clearTimeout(t);
  }, [cfg.on, cfg.sec, cfg.scope, selected, key]);

  const pos = selected ? codes.indexOf(selected) : -1;
  return (
    <span className="auto-view" title={`대상 종목을 ${cfg.sec}초마다 차례로 보여 줍니다 (종목을 직접 고르면 그 종목부터 이어감)`}>
      <label className="check">
        <input type="checkbox" checked={cfg.on} onChange={(e) => save({ on: e.target.checked })} /> 자동 보기
      </label>
      <input
        className="auto-sec"
        type="number"
        min={AUTO_MIN_SEC}
        max={AUTO_MAX_SEC}
        value={secText}
        onChange={(e) => setSecText(e.target.value)}
        onBlur={() => (save({ sec: clampSec(secText) }), setSecText(String(clampSec(secText))))}
        onKeyDown={(e) => e.key === 'Enter' && (e.currentTarget as HTMLInputElement).blur()}
      />
      초
      <select value={cfg.scope} onChange={(e) => save({ scope: e.target.value as AutoScope })} title="자동 보기 대상">
        {(Object.keys(SCOPE_LABEL) as AutoScope[]).map((k) => (
          <option key={k} value={k}>
            {SCOPE_LABEL[k]}
          </option>
        ))}
      </select>
      {cfg.on && <span className="muted">{codes.length ? `${pos >= 0 ? pos + 1 : '-'}/${codes.length}` : '대상 없음'}</span>}
    </span>
  );
}
