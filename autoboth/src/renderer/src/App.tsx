/*
 * Update log
 * - [2026-11-04] type=feat - App shell: status bar + tabs (pipeline / board / timeline / trades / settings)
 * - [2026-11-06] type=feat - 차트 tab (HTS-style chart) after 타임라인
 * - [2026-11-05] type=feat - 제어 tab (r003 start/stop/restart, pending changes, history) left of the pipeline tab
 */

import { useStore, type Tab } from './store';
import { StatusBar } from './components/StatusBar';
import { PipelineView } from './views/PipelineView';
import { ControlView } from './views/ControlView';
import { BoardView } from './views/BoardView';
import { TimelineView } from './views/TimelineView';
import { ChartView } from './views/ChartView';
import { TradesView } from './views/TradesView';
import { SettingsView } from './views/SettingsView';

const TABS: [Tab, string][] = [
  ['control', '제어'],
  ['pipeline', '파이프라인'],
  ['board', '종목 보드'],
  ['timeline', '타임라인'],
  ['chart', '차트'],
  ['trades', '매매 결과'],
  ['settings', '설정/연결'],
];

export function App() {
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);
  return (
    <div className="app">
      <StatusBar />
      <nav className="tabs">
        {TABS.map(([k, label]) => (
          <button key={k} className={tab === k ? 'tab active' : 'tab'} onClick={() => setTab(k)}>
            {label}
          </button>
        ))}
      </nav>
      <main className="view">
        {tab === 'control' && <ControlView />}
        {tab === 'pipeline' && <PipelineView />}
        {tab === 'board' && <BoardView />}
        {tab === 'timeline' && <TimelineView />}
        {tab === 'chart' && <ChartView />}
        {tab === 'trades' && <TradesView />}
        {tab === 'settings' && <SettingsView />}
      </main>
    </div>
  );
}
