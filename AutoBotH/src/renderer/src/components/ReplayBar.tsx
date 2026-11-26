/*
 * Update log
 * - [2026-11-07] type=feat - Past-day time replay bar (under the tabs, only for past days / local folders): time slider over the
 *   day's log range, play / pause with speed (1x-600x), -1 min / +1 min, previous / next trade, a HH:MM:SS jump box and
 *   "전체" (whole day). Every tab (pipeline, board, timeline, chart markers, trades) shows the state cut at the cursor because
 *   the main process re-sends the snapshot after each move. Cursor calls are coalesced: while one is running only the latest
 *   requested time is kept and sent next, so dragging or fast playback never queues work in the engine worker
 * - [2026-11-27] type=feat - Replay-by-date menu (user request: replay the rule-trade history of any day): the bar is now always
 *   there and starts with "리플레이 날짜" - a calendar marking the days that have logs, ◀ 전날 / 다음날 ▶ over those days, and
 *   "불러오기" which reads the WHOLE day again from the start (also today: opened once like a past day so it can be replayed).
 *   "실시간으로" goes back to today's live view. While the live day is shown only this date menu is drawn
 * - [2026-11-07] type=feat - Log gaps (r003 not running, ReplayInfo.gaps) are shaded on the slider, the time shows "r003 로그 없음"
 *   inside one, and playback jumps over them
 */

import { useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import { fmtKstTime, kstToEpoch, ymdDash } from '../../../shared/time';
import { neighborDate, replayDates } from '../../../shared/replay';
import { DatePicker, type DayMark } from './DatePicker';

/** [2026-11-27] date menu: pick a day with logs and load the whole day for replay */
function ReplayDateMenu({ date, live, onLoaded }: { date: string; live: boolean; onLoaded: () => void }) {
  const serverDate = useStore((s) => s.status.serverDate);
  const conn = useStore((s) => s.status.conn);
  const [dates, setDates] = useState<string[]>([]);
  const [pick, setPick] = useState(date);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  useEffect(() => setPick(date), [date]);
  useEffect(() => {
    let alive = true;
    void window.api
      .listDates()
      .then((l: { remote?: string[]; cached?: string[] }) => alive && setDates(replayDates(l)))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [conn, serverDate, date]);
  const today = serverDate || dates[dates.length - 1] || date;
  const marks: Record<string, DayMark> = {};
  for (const d of dates) marks[d] = 'data';
  const load = async (ymd: string, replay: boolean) => {
    if (!/^\d{8}$/.test(ymd) || busy) return;
    setBusy(true);
    setMsg('');
    try {
      const r = await window.api.openDay(ymd, replay);
      if (!r.ok) setMsg(r.error || '불러오지 못했습니다');
      else onLoaded();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };
  const prev = neighborDate(dates, pick || date, -1);
  const next = neighborDate(dates, pick || date, 1);
  return (
    <>
      <b className={live ? 'pill live' : 'pill replay-pill'}>{live ? '리플레이' : '재현'}</b>
      <span className="muted">날짜</span>
      <DatePicker value={pick} onChange={setPick} today={today} max={today} marks={marks} dataLabel="로그 있음" title="리플레이할 날짜 (점 = 로그가 있는 날)" />
      <button className="mini" title="로그가 있는 이전 날을 불러옵니다" disabled={busy || !prev} onClick={() => prev && void load(prev, true)}>
        ◀ 전날
      </button>
      <button className="mini" title="로그가 있는 다음 날을 불러옵니다" disabled={busy || !next} onClick={() => next && void load(next, true)}>
        다음날 ▶
      </button>
      <button
        className="mini primary"
        disabled={busy || !pick}
        title="이 날의 로그 전체를 처음부터 다시 읽어 리플레이합니다 (오늘도 가능 - 불러온 시점까지)"
        onClick={() => void load(pick, true)}
      >
        {busy ? '불러오는 중…' : '불러오기'}
      </button>
      {!live && serverDate && (
        <button className="mini" disabled={busy} title="오늘의 실시간 화면으로 돌아갑니다" onClick={() => void load(serverDate, false)}>
          실시간으로
        </button>
      )}
      {msg && <span className="err-text small">{msg}</span>}
    </>
  );
}

const SPEEDS = [1, 10, 60, 300, 600];
const TICK_MS = 500;
const MIN = 60000;

/** Sends cursor times one at a time; a request made while one is running replaces the waiting one. */
function useCursorSender() {
  const busy = useRef(false);
  const next = useRef<{ ts: number | null } | null>(null);
  const [error, setError] = useState('');
  const send = (ts: number | null) => {
    next.current = { ts };
    if (busy.current) return;
    const run = async () => {
      busy.current = true;
      while (next.current) {
        const { ts: t } = next.current;
        next.current = null;
        const r = await window.api.setReplayCursor(t);
        setError(r.ok ? '' : r.error || '실패');
      }
      busy.current = false;
    };
    void run();
  };
  return { send, error };
}

export function ReplayBar() {
  const replay = useStore((s) => s.snapshot?.replay);
  const date = useStore((s) => s.snapshot?.date || s.status.date);
  const live = useStore((s) => s.status.live);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(60);
  const [pos, setPos] = useState<number | null>(null);
  const [jump, setJump] = useState('');
  const { send, error } = useCursorSender();
  const posRef = useRef<number | null>(null);

  // follow the cursor the engine reports (after a load, a re-read, another date) unless we are moving it ourselves
  const serverCursor = replay?.cursor ?? null;
  useEffect(() => {
    if (!playing) {
      setPos(serverCursor);
      posRef.current = serverCursor;
    }
  }, [serverCursor, playing]);
  useEffect(() => setPlaying(false), [date]);

  const first = replay?.first ?? 0;
  const last = replay?.last ?? 0;

  const moveTo = (ts: number | null) => {
    const t = ts === null ? null : Math.max(first, Math.min(last, Math.round(ts)));
    posRef.current = t;
    setPos(t);
    send(t);
  };

  useEffect(() => {
    if (!playing || !replay) return;
    const id = setInterval(() => {
      const cur = posRef.current ?? first;
      let nxt = cur + speed * TICK_MS;
      // nothing happens while r003 was not logging - jump to where the logs continue
      const gap = (replay.gaps || []).find(([a, b]) => nxt > a && nxt < b);
      if (gap) nxt = gap[1];
      if (nxt >= last) {
        moveTo(last);
        setPlaying(false);
      } else moveTo(nxt);
    }, TICK_MS);
    return () => clearInterval(id);
  }, [playing, speed, first, last]); // eslint-disable-line react-hooks/exhaustive-deps

  const dateMenu = <ReplayDateMenu date={date || ''} live={!!live} onLoaded={() => setPlaying(false)} />;
  if (live || !replay || !last) return <div className="replaybar">{dateMenu}</div>;

  const shown = pos ?? last;
  const trades = replay.trades || [];
  const prevTrade = [...trades].reverse().find((t) => t < shown - 999);
  const nextTrade = trades.find((t) => t > shown);
  const doJump = () => {
    const m = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(jump.trim());
    if (!m || !date) return;
    const hms = `${m[1].padStart(2, '0')}:${m[2]}:${m[3] || '00'}`;
    moveTo(kstToEpoch(ymdDash(date), hms));
  };
  const pct = (t: number) => `${((t - first) / Math.max(1, last - first)) * 100}%`;
  const inGap = (replay.gaps || []).find(([a, b]) => shown > a && shown < b);

  return (
    <div className="replaybar">
      {dateMenu}
      <span className="replay-sep" />
      <button className="mini" title="처음" onClick={() => moveTo(first)}>
        ⏮
      </button>
      <button className="mini" title="이전 매매" disabled={prevTrade === undefined} onClick={() => prevTrade !== undefined && moveTo(prevTrade)}>
        ◀매매
      </button>
      <button className="mini" title="1분 뒤로" onClick={() => moveTo(shown - MIN)}>
        -1분
      </button>
      <button
        className={playing ? 'mini on' : 'mini primary'}
        onClick={() => {
          if (!playing && (pos === null || pos >= last)) moveTo(first);
          setPlaying(!playing);
        }}
      >
        {playing ? '⏸ 일시정지' : '▶ 재생'}
      </button>
      <button className="mini" title="1분 앞으로" onClick={() => moveTo(shown + MIN)}>
        +1분
      </button>
      <button className="mini" title="다음 매매" disabled={nextTrade === undefined} onClick={() => nextTrade !== undefined && moveTo(nextTrade)}>
        매매▶
      </button>
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} title="재생 속도 (로그 시간 / 실제 시간)">
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}x
          </option>
        ))}
      </select>
      <div className="replay-track">
        <input
          type="range"
          min={first}
          max={last}
          step={1000}
          value={shown}
          onChange={(e) => {
            setPlaying(false);
            moveTo(Number(e.target.value));
          }}
        />
        {(replay.gaps || []).map(([a, b], i) => (
          <i key={`g${i}`} className="replay-gap" style={{ left: pct(a), width: `calc(${pct(b)} - ${pct(a)})` }} title={`r003 로그 없음 ${fmtKstTime(a)} ~ ${fmtKstTime(b)}`} />
        ))}
        {trades.map((t, i) => (
          <i key={i} className="replay-mark" style={{ left: pct(t) }} title={`매매 ${fmtKstTime(t)}`} />
        ))}
      </div>
      <span className="replay-time mono" title={`로그 범위 ${fmtKstTime(first)} ~ ${fmtKstTime(last)}`}>
        {pos === null ? `전체 (~${fmtKstTime(last)})` : fmtKstTime(pos)}
      </span>
      {inGap && (
        <span className="muted small" title="이 구간에는 r003 로그가 없습니다 (r003 정지). 재생하면 건너뜁니다.">
          r003 로그 없음 ~{fmtKstTime(inGap[1])}
        </span>
      )}
      <input
        className="replay-jump mono"
        placeholder="HH:MM"
        value={jump}
        onChange={(e) => setJump(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && doJump()}
        title="이 시각으로 이동 (Enter)"
      />
      <button
        className="mini"
        title="재현을 끝내고 그날 전체를 봅니다"
        disabled={pos === null}
        onClick={() => {
          setPlaying(false);
          moveTo(null);
        }}
      >
        전체
      </button>
      {error && <span className="err-text small">{error}</span>}
    </div>
  );
}
