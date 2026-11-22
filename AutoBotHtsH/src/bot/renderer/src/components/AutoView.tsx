/*
 * Update log
 * - [2026-11-20] type=feat - 자동 보기 on the timeline (user 2026-11-20; AutoBotHtsH first, then AutoBotH - same file in both apps:
 *   AutoBotH src/renderer/src/components, AutoBotHtsH src/bot/renderer/src/components). While checked, the timeline steps through the
 *   target symbols (대상: 전체 / 감시 중 / 보유) every N seconds (3 - 600, default 10), in the symbol picker's order, wrapping around.
 *   Each step is an ordinary symbol selection (in AutoBotHtsH the HTS windows follow it too). A symbol picked by hand stays for a full
 *   interval before the loop moves on. Settings kept per PC (localStorage); the loop runs only while the timeline is on screen
 * - [2026-11-20] type=feat - Only symbols that already trade at the viewed time (user 2026-11-20): today live = now, a replayed past
 *   day = the replay cursor, a whole past day = all. NXT symbols from 08:00, KRX-only symbols (and symbols whose NXT flag is not known)
 *   from 09:00 (shared/time tradingSessionRange) - before 09:00 the loop shows the NXT symbols only. The count says how many wait
 * - [2026-11-20] type=feat - One auto view at a time (AutoBotHtsH: timeline and watchlist): turning one on sends AUTO_VIEW_EVENT with
 *   its name; any other auto view that is on unchecks itself and stops. autoViewStarted() / onAutoViewStarted() are the two ends.
 *   The last one started is kept (localStorage autoview.owner): one saved as on whose window opens later stays off if another started since
 * - [2026-11-20] type=feat - 운용 시간 = 정규장만 (r001 ENABLE_NXT_SESSION off): every symbol from 09:00 (no NXT pre-market)
 * - [2026-11-23] type=fix - Nothing trades yet (e.g. 08:02 with 운용 시간 = 정규장만): the status read '대상 없음 (거래 전 14 제외)' and
 *   looked like a stopped loop. It now says when the loop starts: '09:00부터 시작 (정규장만 운용 · 14종목 대기)'
 * - [2026-11-23] type=feat - 운용 시간 = 정규장만 no longer holds the loop until 09:00 (user 2026-11-23): NXT symbols from 08:00 in
 *   either mode, KRX-only symbols from 09:00 (their prices come from KIS before the bot starts quoting)
 */

import { useEffect, useState } from 'react';
import { useStore } from '../store';
import type { Snapshot, SymbolSummary } from '../../../shared/types';
import { tradingSessionRange } from '../../../shared/time';

export type AutoScope = 'all' | 'active' | 'held';

/** [2026-11-20] window event "an auto view started" (detail = its name: 'timeline' | 'watch') */
export const AUTO_VIEW_EVENT = 'autoview:start';
const OWNER_KEY = 'autoview.owner';
export const autoViewStarted = (who: string) => {
  try {
    localStorage.setItem(OWNER_KEY, who);
  } catch {
    /* storage unavailable */
  }
  window.dispatchEvent(new CustomEvent(AUTO_VIEW_EVENT, { detail: who }));
};
/** an auto view saved as on when its window opens keeps running only if it was the last one started */
export function autoViewMayResume(who: string): boolean {
  try {
    const o = localStorage.getItem(OWNER_KEY);
    return !o || o === who;
  } catch {
    return true;
  }
}
/** calls `stop` when another auto view starts; returns the remover */
export function onAutoViewStarted(me: string, stop: () => void): () => void {
  const h = (e: Event) => (e as CustomEvent<string>).detail !== me && stop();
  window.addEventListener(AUTO_VIEW_EVENT, h);
  return () => window.removeEventListener(AUTO_VIEW_EVENT, h);
}
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

/** time the targets are judged at: now on a live day, the replay cursor on a replayed past day, null = the whole day (all) */
export function viewTime(live: boolean, cursor: number | null | undefined, now: number): number | null {
  if (live) return now;
  return cursor ?? null;
}

/** the symbol trades at `t` already (NXT from 08:00, KRX-only from 09:00); `t` null = always */
export function tradingAt(s: SymbolSummary, date: string | undefined, t: number | null, nxtOn = true): boolean {
  if (t === null) return true;
  const r = tradingSessionRange(date, s.nxt === true, nxtOn);
  return !r || t >= r[0];
}

/** codes the loop goes through: in scope and already trading at `t`; `waiting` = in scope but not trading yet */
export function autoTargets(snap: Snapshot | null, scope: AutoScope, t: number | null, nxtOn = true): { codes: string[]; waiting: number } {
  const inS = (snap?.symbols || []).filter((s) => inScope(s, scope));
  const codes = inS.filter((s) => tradingAt(s, snap?.date, t, nxtOn)).map((s) => s.code);
  return { codes, waiting: inS.length - codes.length };
}

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
  const live = useStore((s) => s.status.live);
  const { codes, waiting } = autoTargets(snap, cfg.scope, viewTime(live, snap?.replay?.cursor, Date.now()));
  const save = (p: Partial<AutoCfg>) => {
    setCfg((cur) => {
      const next = { ...cur, ...p };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
    if (p.on) autoViewStarted('timeline');
  };
  // another auto view (watchlist) started: stop here; saved as on but another one started since: stays off
  useEffect(() => onAutoViewStarted('timeline', () => save({ on: false })), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (cfg.on && !autoViewMayResume('timeline')) save({ on: false });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // one step per interval; a new selection (by hand or by the loop) starts the next interval
  const key = codes.join(',');
  useEffect(() => {
    if (!cfg.on) return;
    const t = setTimeout(() => {
      const st = useStore.getState();
      const list = autoTargets(st.snapshot, cfg.scope, viewTime(st.status.live, st.snapshot?.replay?.cursor, Date.now())).codes;
      const n = nextCode(list, st.selected);
      if (n && n !== st.selected) st.select(n);
    }, cfg.sec * 1000);
    return () => clearTimeout(t);
  }, [cfg.on, cfg.sec, cfg.scope, selected, key]);

  const pos = selected ? codes.indexOf(selected) : -1;
  return (
    <span className="auto-view" title={`대상 종목 중 지금 거래 중인 종목을 ${cfg.sec}초마다 차례로 보여 줍니다 (종목을 직접 고르면 그 종목부터 이어감)`}>
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
      {cfg.on && (
        <span className="muted" title={waiting ? `거래 시작 전 ${waiting}종목은 빠짐 (NXT 종목 08:00, KRX 종목 09:00부터)` : undefined}>
          {codes.length
            ? `${pos >= 0 ? pos + 1 : '-'}/${codes.length}${waiting ? ` (거래 전 ${waiting} 제외)` : ''}`
            : waiting
              ? `거래 시작(NXT 08:00 / KRX 09:00)부터 시작 (${waiting}종목 대기)`
              : '대상 없음'}
        </span>
      )}
    </span>
  );
}
