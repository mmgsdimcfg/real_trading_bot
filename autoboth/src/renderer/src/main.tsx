import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { App } from './App';
import { initStore } from './store';

initStore();
createRoot(document.getElementById('root')!).render(<App />);
