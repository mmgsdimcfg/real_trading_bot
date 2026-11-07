import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { App } from './App';
import { initStore } from './store';
import { applyTheme, loadTheme } from './theme';

applyTheme(loadTheme());
initStore();
createRoot(document.getElementById('root')!).render(<App />);
