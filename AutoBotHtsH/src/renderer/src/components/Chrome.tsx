// Update log
// - [2026-11-06] type=feat (Title bar (TermH style, frame:false): icon + HtsH, connection badges (실전 / 모의 / 데모, 조회 전용, realtime
//   state), KOSPI / KOSDAQ ticker, ☰ (shared app-menu.js: 설정 = 로그인 정보, 종목 마스터 새로고침, 레이아웃 초기화, 업데이트 확인, HtsH 정보)
//   and window buttons. Toolbar: symbol search for the active window, buttons that open windows, the arrange menu)
// - [2026-11-06] type=feat (Theme button left of ☰ (system -> light -> dark, saved) and ☰ > 테마 전환)
// - [2026-11-20] type=feat (AutoBotHtsH: brand / About text of the combined app, ☰ > 봇 설정·연결, toolbar menu '봇 ▾' (the AutoBotH tabs
//   in its 운영 / 분석 / 시스템 groups) that opens or focuses each bot window)
// - [2026-11-20] type=fix (봇 ▾ / 창 정렬 ▾ close on a click outside them or Esc)
// - [2026-11-21] type=feat (Title bar chip "반등 감시 <탭> · N종목": state of the rebound watcher (main/reboundPoller.ts), a click
//   switches it on / off; ☰ > 반등 감시 설정·기록 폴더, 반등 알람 테스트)
// - [2026-11-21] type=feat (The chip only shows the state now (no click); ☰ > 반등 감시 설정 opens the settings dialog
//   (components/ReboundSettings.tsx: on / off, tab, symbols, thresholds; folder and test popup are buttons there))
// - [2026-11-22] type=feat (Title bar "미확인 알람 N": alerts whose popup closed by itself and were not looked at yet; a click
//   opens the history tab of the rebound settings dialog)

import { useEffect, useRef, useState } from 'react';
import { BOT_GROUPS, BOT_TITLE, botWinOf, useStore, WIN_TITLE, type WinType, type ArrangeMode, type Theme } from '../store';
import { SymbolSearch } from './SymbolSearch';
import { useCall, useOutsideClose } from '../hooks';
import type { IndexQuote } from '../../../shared/market';
import { fmtNum, fmtPct, signOf, upDown } from '../lib';
import type { ReboundStatus } from '../../../preload/index';

const THEME_NEXT: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' };
const THEME_LABEL: Record<Theme, [string, string]> = { system: ['◐', '시스템 설정'], light: ['☀', '라이트'], dark: ['☾', '다크'] };
export const cycleTheme = () => {
  const s = useStore.getState();
  s.setTheme(THEME_NEXT[s.theme]);
};

function WindowButtons() {
  const theme = useStore((s) => s.theme);
  const [max, setMax] = useState(false);
  useEffect(() => {
    void window.winCtl.isMaximized().then(setMax);
    return window.winCtl.onMaximized(setMax);
  }, []);
  return (
    <div className="win-controls">
      <button type="button" className="win-btn theme-btn" title={`테마: ${THEME_LABEL[theme][1]} (클릭 → ${THEME_LABEL[THEME_NEXT[theme]][1]})`} onClick={cycleTheme}>
        {THEME_LABEL[theme][0]}
      </button>
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

function IndexTicker() {
  const k = useCall<IndexQuote>('index', ['0001'], 15000);
  const q = useCall<IndexQuote>('index', ['1001'], 15000);
  return (
    <span className="ticker">
      {[k.data, q.data].map((x, i) =>
        x ? (
          <span key={i} className={upDown(signOf(x.sign, x.change))}>
            {x.name} <b>{fmtNum(x.value)}</b> {fmtPct(x.changePct)}
          </span>
        ) : null,
      )}
    </span>
  );
}

const REBOUND_STATE: Record<ReboundStatus['state'], string> = { off: '꺼짐', rest: '장외 대기', login: '로그인 대기', notab: '탭 없음', run: '감시 중' };

/** [2026-11-21] rebound watcher state (shown only - it is set in ☰ > 반등 감시 설정) */
function ReboundChip() {
  const [st, setSt] = useState<ReboundStatus | null>(null);
  useEffect(() => {
    void window.hts.reboundStatus().then(setSt);
    const off = window.hts.onReboundStatus(setSt);
    return () => void off();
  }, []);
  if (!st) return null;
  const dot = !st.enabled ? 'off' : st.state === 'notab' ? 'error' : st.state === 'run' ? (st.failed ? 'connecting' : 'on') : 'off';
  const label = st.enabled && st.state === 'run' ? `${st.codes}종목` : REBOUND_STATE[st.enabled ? st.state : 'off'];
  const tip = [
    `관심종목 "${st.tab}" 탭의 1분봉 바닥 반등 초입 감시 (감지·알림 전용, 주문 없음)`,
    st.note || '',
    st.lastPass ? `마지막 순회 ${new Date(st.lastPass).toLocaleTimeString('ko-KR', { hour12: false })} - 새 봉 판정 ${st.judged ?? 0}종목` : '',
    `알림 ${st.alerts}회`,
    '설정: ☰ > 반등 감시 설정',
  ]
    .filter(Boolean)
    .join('\n');
  return (
    <span className={`rt rt-${dot}`} title={tip}>
      <i /> 반등 감시 {st.tab} · {label}
      {st.alerts ? ` · 알림 ${st.alerts}` : ''}
    </span>
  );
}

/** [2026-11-22] number of rebound alerts not looked at yet (their popup closed by itself); a click opens the history */
function ReboundUnread() {
  const [n, setN] = useState(0);
  useEffect(() => {
    void window.hts.reboundUnread().then(setN);
    const off = window.hts.onReboundUnread(setN);
    return () => void off();
  }, []);
  if (!n) return null;
  return (
    <button type="button" className="rebound-unread" title="확인하지 않은 반등 알람 - 클릭하면 히스토리를 엽니다" onClick={() => window.dispatchEvent(new CustomEvent('rebound-settings', { detail: 'history' }))}>
      미확인 알람 {n}
    </button>
  );
}

export function TitleBar({ onSettings }: { onSettings: () => void }) {
  const auth = useStore((s) => s.auth);
  const resetLayout = useStore((s) => s.resetLayout);
  const menuInit = useRef(false);
  useEffect(() => {
    if (menuInit.current || !window.AppMenu) return;
    menuInit.current = true;
    window.AppMenu.init({
      button: '#appMenuBtn',
      productName: 'AutoBotHtsH',
      icon: './icon.png',
      tagline: { en: 'Personal HTS (KIS quotes) + r003 trading bot viewer and control', ko: '개인 HTS(KIS 조회) + r003 자동매매 봇 뷰어·제어' },
      description: {
        en: 'HtsH windows (charts, current price, stock information, watchlist, rankings, indices, news - KIS OpenAPI, inquiry only) together with the AutoBotH windows (bot control, data jobs, tick collector, condition pipeline, symbol board, timeline, bot chart, holdings, trade results, paper trading, bot connection settings). Bot actions go through the server control script; the app itself never places orders.',
        ko: 'HtsH 창(차트, 현재가, 종목정보, 관심종목, 순위, 지수, 뉴스 - KIS OpenAPI 조회 전용)과 AutoBotH 창(봇 제어, 데이터 작업, 수집기, 조건 파이프라인, 종목 보드, 타임라인, 봇 차트, 보유 현황, 매매 결과, 가상 매매, 봇 설정·연결)을 한 화면에서 씁니다. 봇 조작은 서버 제어 스크립트를 거치며 앱이 직접 주문하지 않습니다.',
      },
      onSettings,
      items: [
        { id: 'master', label: { en: 'Refresh stock master', ko: '종목 마스터 새로고침' }, icon: '↻', onClick: () => void window.hts.refreshMaster().then(() => window.hts.authState().then((a) => useStore.getState().setAuth(a))) },
        { id: 'bot', label: { en: 'Bot connection settings', ko: '봇 설정·연결' }, icon: '⚙', onClick: () => void useStore.getState().openBot('settings') },
        { id: 'rebound', label: { en: 'Rebound watcher settings', ko: '반등 감시 설정' }, icon: '◉', onClick: () => window.dispatchEvent(new Event('rebound-settings')) },
        { id: 'theme', label: { en: 'Switch theme (system / light / dark)', ko: '테마 전환 (시스템 / 라이트 / 다크)' }, icon: '◐', onClick: cycleTheme },
        { id: 'layout', label: { en: 'Reset window layout', ko: '창 배치 초기화' }, icon: '▦', onClick: () => window.confirm('열린 창을 모두 닫고 배치를 초기화할까요?') && resetLayout() },
      ],
      lang: () => 'ko',
    });
  }, [onSettings, resetLayout]);

  const mode = auth?.mode;
  const rt = auth?.rt;
  return (
    <header className="titlebar" onDoubleClick={(e) => e.target === e.currentTarget && window.winCtl.toggleMaximize()}>
      <span className="tb-brand">
        <img src="./icon.png" alt="" draggable={false} />
        AutoBotHtsH
      </span>
      <div className="tb-status" onDoubleClick={(e) => e.target === e.currentTarget && window.winCtl.toggleMaximize()}>
        {auth?.status === 'in' && (
          <>
            <b className={`badge ${mode === 'demo' ? 'demo' : auth.env}`}>{mode === 'demo' ? '데모' : auth.env === 'real' ? '실전 서버' : '모의 서버'}</b>
            <b className="badge ro" title="주문·계좌 기능이 없습니다. 시세 조회 TR만 허용합니다.">조회 전용</b>
            <span className={`rt rt-${rt?.status || 'off'}`} title={rt?.msg || ''}>
              <i /> 실시간 {rt?.status === 'on' ? '연결' : rt?.status === 'connecting' ? '연결 중' : rt?.status === 'error' ? '오류' : '대기'}
            </span>
            <ReboundChip />
            <ReboundUnread />
            {mode === 'kis' && auth.keyHint && <span className="muted">앱키 {auth.keyHint}</span>}
            <span className="muted" title={auth.master.error || ''}>
              종목 {auth.master.size.toLocaleString()}
              {auth.master.error ? ' (마스터 갱신 실패)' : ''}
            </span>
            <IndexTicker />
          </>
        )}
      </div>
      <WindowButtons />
    </header>
  );
}

const OPEN: WinType[] = ['chart', 'quote', 'info', 'watch', 'rank', 'index', 'news'];
const ARR: [ArrangeMode | 'min' | 'restore' | 'close', string][] = [
  ['cascade', '계단식'],
  ['tile', '바둑판식'],
  ['rows', '가로로 나란히'],
  ['cols', '세로로 나란히'],
  ['min', '모두 최소화'],
  ['restore', '모두 복원'],
  ['close', '모두 닫기'],
];

export function Toolbar() {
  const active = useStore((s) => s.wins.find((w) => w.id === s.active));
  const lastCode = useStore((s) => s.lastCode);
  const { open, openBot, setCode, arrange, minimizeAll, restoreAll, closeAll } = useStore.getState();
  const [arrOpen, setArrOpen] = useState(false);
  const [botOpen, setBotOpen] = useState(false);
  const botRef = useOutsideClose<HTMLSpanElement>(botOpen, () => setBotOpen(false));
  const arrRef = useOutsideClose<HTMLSpanElement>(arrOpen, () => setArrOpen(false));
  const openWins = useStore((s) => s.wins);
  return (
    <div className="toolbar main-tb">
      <SymbolSearch
        value={active?.code || lastCode}
        onPick={(code) => {
          if (active && ['chart', 'quote', 'info', 'news'].includes(active.type)) setCode(active.id, code);
          else open('chart', { code });
        }}
      />
      <span className="tb-sep" />
      {OPEN.map((t) => (
        <button key={t} onClick={() => open(t, { code: active?.code || lastCode })} title={`${WIN_TITLE[t]} 창 열기`}>
          + {WIN_TITLE[t]}
        </button>
      ))}
      <span className="tb-sep" />
      <span className="dropdown" ref={botRef}>
        <button onClick={() => setBotOpen(!botOpen)} title="AutoBotH 창 (봇 제어 · 파이프라인 · 타임라인 · 매매 결과 ...)">
          봇 ▾
        </button>
        {botOpen && (
          <span className="dd-menu bot-menu" onMouseLeave={() => setBotOpen(false)}>
            {BOT_GROUPS.map(({ group, tabs }) => (
              <span key={group} className="dd-group">
                <span className="dd-caption">{group}</span>
                {tabs.map((t) => (
                  <button
                    key={t}
                    className={openWins.some((w) => w.type === botWinOf(t)) ? 'on' : ''}
                    onClick={() => {
                      setBotOpen(false);
                      openBot(t);
                    }}
                  >
                    {BOT_TITLE[t]}
                  </button>
                ))}
              </span>
            ))}
          </span>
        )}
      </span>
      <span className="dropdown" ref={arrRef}>
        <button onClick={() => setArrOpen(!arrOpen)}>창 정렬 ▾</button>
        {arrOpen && (
          <span className="dd-menu" onMouseLeave={() => setArrOpen(false)}>
            {ARR.map(([k, label]) => (
              <button
                key={k}
                onClick={() => {
                  setArrOpen(false);
                  if (k === 'min') minimizeAll();
                  else if (k === 'restore') restoreAll();
                  else if (k === 'close') window.confirm('열린 창을 모두 닫을까요?') && closeAll();
                  else arrange(k);
                }}
              >
                {label}
              </button>
            ))}
          </span>
        )}
      </span>
    </div>
  );
}
