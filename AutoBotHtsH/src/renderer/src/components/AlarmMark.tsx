// Update log
// - [2026-11-23] type=feat (Watchlist (list and tiles): a bell on the symbols that raised a rebound alert today (user 2026-11-23) with
//   the number of alerts; its tooltip lists the times, a click shows the latest alert's popup again (the full detail). The alerts
//   come from the rebound history (main/reboundHistory.ts), read again whenever it changes and once a minute (the day may turn))

import { useEffect, useState } from 'react';
import type { ReboundHistoryItem } from '../../../preload';

const KST = 9 * 3600000;
const dayOf = (t: number) => Math.floor((t + KST) / 86400000);
const hhmm = (t: number) => new Date(t + KST).toISOString().slice(11, 16);

/** today's (KST) alerts per code, newest first */
export function todayAlerts(items: ReboundHistoryItem[], now: number): Map<string, ReboundHistoryItem[]> {
  const today = dayOf(now);
  const out = new Map<string, ReboundHistoryItem[]>();
  for (const it of items) {
    if (dayOf(it.at) !== today) continue;
    const list = out.get(it.code);
    if (list) list.push(it);
    else out.set(it.code, [it]);
  }
  for (const list of out.values()) list.sort((a, b) => b.at - a.at);
  return out;
}

/** today's alerts per code, kept current */
export function useTodayAlerts(): Map<string, ReboundHistoryItem[]> {
  const [map, setMap] = useState<Map<string, ReboundHistoryItem[]>>(new Map());
  useEffect(() => {
    let alive = true;
    const load = () =>
      void window.hts
        .reboundHistory()
        .then((items) => alive && setMap(todayAlerts(items, Date.now())))
        .catch(() => undefined);
    load();
    const off = window.hts.onReboundUnread(() => load());
    const timer = setInterval(load, 60000);
    return () => {
      alive = false;
      void off();
      clearInterval(timer);
    };
  }, []);
  return map;
}

export function AlarmMark({ alerts }: { alerts: ReboundHistoryItem[] | undefined }) {
  if (!alerts?.length) return null;
  const latest = alerts[0];
  const times = alerts.map((a) => hhmm(a.at)).join(', ');
  return (
    <span
      role="button"
      className={`alarm-mark ${alerts.some((a) => !a.read) ? 'unread' : ''}`}
      title={`오늘 반등 알림 ${alerts.length}회 (${times}) · 누르면 최근 알림 내역 표시`}
      onClick={(e) => {
        e.stopPropagation();
        void window.hts.reboundHistoryPopup(latest.id);
      }}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      🔔{alerts.length > 1 ? alerts.length : ''}
    </span>
  );
}
