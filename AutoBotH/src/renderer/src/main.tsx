import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { App } from './App';
import { initStore } from './store';
import { applyTheme, loadTheme } from './theme';

// [2026-11-27] window.alert / window.confirm -> a message box of the main process (Electron's own leave the page without keyboard
// focus on Windows: no text box took typing afterwards). Same synchronous use as before for every caller.
{
  const nd = (window as unknown as { nativeDialog?: { alert: (t: string) => void; confirm: (t: string) => boolean } }).nativeDialog;
  if (nd) {
    window.alert = (m?: unknown) => nd.alert(String(m ?? ''));
    window.confirm = (m?: unknown) => nd.confirm(String(m ?? ''));
  }
}

applyTheme(loadTheme());
initStore();
createRoot(document.getElementById('root')!).render(<App />);
