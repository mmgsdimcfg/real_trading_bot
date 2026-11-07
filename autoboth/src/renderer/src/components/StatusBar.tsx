/*
 * Update log
 * - [2026-11-04] type=feat - Top status bar: connection, day (live/past), last receive time, files/bytes, lines, parse warnings,
 *   market regime, watch-list split, condition source
 * - [2026-11-06] type=feat - The status bar is now the window's title bar (TermH style, frame:false): app icon + "AutoBotH", a gap,
 *   the status items, then ☰ (shared app-menu.js: 설정 / 업데이트 확인 / AutoBotH 정보) and minimize / maximize-restore / close.
 *   Empty space drags the window; double-click toggles maximize
 * - [2026-11-05] type=feat - Bot summary (봇: 실행 중 / 정지 / 미적용 변경 N, dry-run marker, external r003 warning) - opens the 제어 tab
 */

import { useEffect, useState } from 'react';
import { useStore } from '../store';
import { fmtBytes, fmtInt } from '../lib';
import { fmtKstTime, ymdDash } from '../../../shared/time';
import { botSummary, pendingCount } from '../../../shared/control';
import { BOT_LABEL } from '../views/ControlView';

/** ☰ menu (app-menu.js is a classic script loaded before the React module runs) */
function useAppMenu(openSettings: () => void) {
  useEffect(() => {
    window.AppMenu?.init({
      button: '#appMenuBtn',
      productName: 'AutoBotH',
      icon: './icon.png',
      tagline: { en: 'Condition pipeline viewer and remote control for the r003 trading bot', ko: 'r003 자동매매 봇 조건 파이프라인 뷰어·원격 제어' },
      description: {
        en: 'Reads the r003 trading bot logs over SSH and shows how each symbol moves through the buy and sell conditions, with price charts, indicators and trade results. Starts and stops the bot, edits its settings and watchlist through the server control script.',
        ko: 'SSH로 r003 자동매매 봇의 로그를 읽어 종목별 매수·매도 조건 진행을 보여 주고, 주가 차트·보조지표·매매 결과를 함께 표시합니다. 서버 제어 스크립트를 통해 봇 실행·정지, 설정값과 감시목록 수정을 할 수 있습니다.',
      },
      onSettings: openSettings,
      lang: () => 'ko',
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
}

function WindowButtons() {
  const [max, setMax] = useState(false);
  useEffect(() => {
    void window.winCtl.isMaximized().then(setMax);
    return window.winCtl.onMaximized(setMax);
  }, []);
  return (
    <div className="win-controls">
      <button type="button" id="appMenuBtn" className="win-btn win-menu" title="메뉴">
        ☰
      </button>
      <button type="button" className="win-btn" title="최소화" onClick={() => window.winCtl.minimize()}>
        ─
      </button>
      <button type="button" className="win-btn" title={max ? '이전 크기로' : '최대화'} onClick={() => window.winCtl.toggleMaximize()}>
        {max ? '❐' : '□'}
      </button>
      <button type="button" className="win-btn win-close" title="닫기" onClick={() => window.winCtl.close()}>
        ✕
      </button>
    </div>
  );
}

const CONN_LABEL: Record<string, string> = {
  idle: '연결 안 됨',
  connecting: '연결 중…',
  connected: '연결됨',
  reconnecting: '재연결 대기',
  error: '연결 오류',
  local: '로컬 폴더',
};
const SRC_LABEL: Record<string, string> = { server: '서버', cache: '캐시', bundled: '내장본' };

export function StatusBar() {
  const status = useStore((s) => s.status);
  const snap = useStore((s) => s.snapshot);
  const setTab = useStore((s) => s.setTab);
  useAppMenu(() => setTab('settings'));
  const control = useStore((s) => s.control);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  const bytes = snap?.files.reduce((a, f) => a + f.offset, 0) || 0;
  const retry = status.nextRetryAt ? Math.max(0, Math.round((status.nextRetryAt - now) / 1000)) : null;
  const ago = status.lastReceivedAt ? Math.round((now - status.lastReceivedAt) / 1000) : null;
  const bot = botSummary(control.status);
  const nPend = control.status?.running ? pendingCount(control.status) : 0;

  return (
    <header className="titlebar" onDoubleClick={(e) => e.target === e.currentTarget && window.winCtl.toggleMaximize()}>
      <span className="tb-brand">
        <img src="./icon.png" alt="" draggable={false} />
        AutoBotH
      </span>
      <div className="statusbar" onDoubleClick={(e) => e.target === e.currentTarget && window.winCtl.toggleMaximize()}>
      <span className={`conn conn-${status.conn}`} title={status.message || ''}>
        <i className="dot" />
        {CONN_LABEL[status.conn] || status.conn}
        {retry !== null && status.conn === 'reconnecting' ? ` (${retry}s)` : ''}
      </span>
      {(control.status || control.busy) && (
        <button
          className={`sb-item linklike bot bot-${bot}`}
          onClick={() => setTab('control')}
          title={control.error ? `상태 조회 오류: ${control.error}` : '제어 탭 열기'}
        >
          <i className="dot" />
          봇: {control.busy ? `${control.busy}…` : BOT_LABEL[bot]}
          {control.status?.running && control.status.dry_run ? ' (dry-run)' : ''}
          {nPend > 0 && <b className="pill pend">미적용 변경 {nPend}</b>}
        </button>
      )}
      <span className="sb-item">
        {status.date ? ymdDash(status.date) : '날짜 없음'}
        <b className={status.live ? 'pill live' : 'pill'}>{status.live ? 'LIVE' : '1회 로드'}</b>
      </span>
      <span className="sb-item" title="마지막으로 새 바이트를 받은 시각">
        수신 {status.lastReceivedAt ? `${fmtKstTime(status.lastReceivedAt)} KST (${ago}s 전)` : '-'}
      </span>
      <span className="sb-item" title={snap?.files.map((f) => `${f.name}  ${f.offset}/${f.size}`).join('\n') || ''}>
        파일 {snap?.files.length || 0} · {fmtBytes(bytes)} · {fmtInt(snap?.lines)}줄
      </span>
      <button className={`sb-item linklike ${snap?.warnings ? 'warn' : ''}`} onClick={() => setTab('settings')} title="알 수 없는 태그/사유 목록 (설정 화면)">
        경고 {fmtInt(snap?.warnings || 0)}
      </button>
      <span className="sb-item">시장 {snap?.regime || 'N/A'}</span>
      {snap?.wl && (
        <span className="sb-item">
          감시 {snap.wl.active}/{snap.wl.backup}
        </span>
      )}
      <span className="sb-item">조건 {SRC_LABEL[status.conditionsSource || ''] || '-'}</span>
      {status.message && <span className="sb-msg" title={status.message}>{status.message}</span>}
      </div>
      <WindowButtons />
    </header>
  );
}
