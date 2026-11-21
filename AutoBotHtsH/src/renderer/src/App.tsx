// Update log
// - [2026-11-06] type=feat (App shell: title bar, toolbar, MDI workspace; the login screen covers the workspace until a KIS login or demo
//   mode; ☰ > 설정 reopens it (account switch / logout / forget saved key). A first login opens a starter layout (chart + current price +
//   watchlist, linked) when no saved layout exists)
// - [2026-11-19] type=feat (Watchlist file read at start; picks files of the watchlist folder checked every 30 s)
// - [2026-11-20] type=feat (Thin scrollbars shown only at the right / bottom edge of a scroll area - installEdgeScrollbars)
// - [2026-11-20] type=feat (AutoBotHtsH: AutoBotH status line under the toolbar (BotStatusStrip) and its confirmation dialog)
// - [2026-11-20] type=fix (<details> popovers (bot y축 항목) close on a click outside them - installDetailsAutoClose)
// - [2026-11-21] type=feat (A rebound alert popup's "열기" shows its symbol in every symbol window (setCodeAll))
import { useEffect, useState } from 'react';
import { useStore } from './store';
import { TitleBar, Toolbar } from './components/Chrome';
import { Workspace } from './components/Workspace';
import { Login } from './components/Login';
import { installEdgeScrollbars } from './edgeScrollbars';
import { installDetailsAutoClose } from './hooks';
import { BotConfirm, BotStatusStrip } from './windows/BotWindow';

export function App() {
  const auth = useStore((s) => s.auth);
  const [settings, setSettings] = useState(false);
  const loggedIn = auth?.status === 'in';

  // [2026-11-20] scrollbars appear at the edge under the cursor
  useEffect(() => installEdgeScrollbars(), []);
  useEffect(() => installDetailsAutoClose(), []);

  // [2026-11-19] watchlist file (%APPDATA%\htsh\watchlist) + picks files -> MMDD tabs
  useEffect(() => {
    void useStore.getState().initWatch();
    const t = setInterval(() => void useStore.getState().syncPicks(), 30000);
    return () => clearInterval(t);
  }, []);

  // [2026-11-21] rebound alert popup > 열기
  useEffect(() => {
    const off = window.hts.onReboundOpen((code) => useStore.getState().setCodeAll(code));
    return () => void off();
  }, []);

  useEffect(() => {
    if (!loggedIn) return;
    const s = useStore.getState();
    if (s.wins.length) return;
    // starter layout
    s.open('chart', { code: s.lastCode, link: 1 });
    s.open('quote', { code: s.lastCode, link: 1 });
    s.open('watch', { link: 1 });
    s.arrange('tile');
  }, [loggedIn]);

  return (
    <div className="app">
      <TitleBar onSettings={() => setSettings(true)} />
      {loggedIn && <Toolbar />}
      {loggedIn && <BotStatusStrip />}
      {loggedIn ? <Workspace /> : <div className="workspace-wrap" />}
      {auth && (!loggedIn || settings) && <Login onClose={loggedIn ? () => setSettings(false) : undefined} />}
      {loggedIn && auth.mode === 'demo' && <div className="demo-banner">데모 모드 - 가상 데이터입니다. ☰ → 설정에서 KIS 앱키로 로그인하세요.</div>}
      <BotConfirm />
    </div>
  );
}
