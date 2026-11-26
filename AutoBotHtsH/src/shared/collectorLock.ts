// Update log
// - [2026-11-26] type=feat (The tick collector's lock file text -> its pid, for the "bot first" realtime rule (main/collectorGuard.ts))

/** shown as the realtime status while the app yields the session */
export const RT_YIELD_MSG = '봇의 틱 수집기가 이 PC에서 KIS 실시간 접속을 쓰고 있어 앱의 실시간을 껐습니다 (앱키당 접속 1개). 시세는 주기 조회로 표시하고, 수집기가 끝나면 다시 연결합니다.';

/** "<pid> <YYYY-MM-DD HH:MM:SS> ppid=<n>" -> pid when the lock was written on `todayYmd` (YYYYMMDD, KST); anything else -> null.
 *  A lock of another day is a leftover (the collector starts once a day) - its pid may belong to another program by now. */
export function parseCollectorLock(text: string, todayYmd: string): number | null {
  const m = /^\s*(\d{1,10})\s+(\d{4})-(\d{2})-(\d{2})[ T]\d{2}:\d{2}:\d{2}/.exec(text || '');
  if (!m) return null;
  const pid = Number(m[1]);
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  return `${m[2]}${m[3]}${m[4]}` === todayYmd ? pid : null;
}
