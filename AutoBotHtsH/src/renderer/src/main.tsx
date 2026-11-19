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

initStore();
initBotStore();
linkBotTabs();
createRoot(document.getElementById('root')!).render(<App />);
