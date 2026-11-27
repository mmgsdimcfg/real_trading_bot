// Update log
// - [2026-11-06] type=feat (HtsH renderer entry)
// - [2026-11-20] type=feat (AutoBotHtsH: AutoBotH styles scoped to .bot-root, React Flow styles (pipeline), the bot store started
//   (IPC listeners of window.api) and its tab requests linked to the workspace windows)
import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import '../../bot/renderer/src/styles.scoped.css';
import { App } from './App';
import { initStore } from './store';
import { initStore as initBotStore } from '../../bot/renderer/src/store';
import { linkBotTabs } from './windows/BotWindow';

// [2026-11-27] window.alert / window.confirm -> a message box of the main process (Electron's own leave the page without keyboard
// focus on Windows: no text box took typing afterwards). Same synchronous use as before for every caller.
{
  const nd = (window as unknown as { nativeDialog?: { alert: (t: string) => void; confirm: (t: string) => boolean } }).nativeDialog;
  if (nd) {
    window.alert = (m?: unknown) => nd.alert(String(m ?? ''));
    window.confirm = (m?: unknown) => nd.confirm(String(m ?? ''));
  }
}

initStore();
initBotStore();
linkBotTabs();
createRoot(document.getElementById('root')!).render(<App />);
