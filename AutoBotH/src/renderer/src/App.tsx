/*
 * Update log
 * - [2026-11-04] type=feat - App shell: status bar + tabs (pipeline / board / timeline / trades / settings)
 * - [2026-11-06] type=feat - 차트 tab (HTS-style chart) after 타임라인
 * - [2026-11-07] type=feat - ReplayBar under the tabs (past day / local folder): time replay of the loaded day for every tab
 * - [2026-11-05] type=feat - 제어 tab (r003 start/stop/restart, pending changes, history) left of the pipeline tab
 * - [2026-11-10] type=feat - 데이터 작업 tab (post-close collection / watchlist selection jobs) after 제어
 * - [2026-11-10] type=feat - Tabs moved to a vertical list on the left (.sidenav, a rule between rows); the replay bar and the tab
 *   content fill the area to its right (.app-main)
 * - [2026-11-10] type=feat - Sidebar in groups (운영 / 분석 / 시스템) with a line icon per tab; the current tab is a rounded
 *   highlight instead of rules between rows
 * - [2026-11-11] type=feat - ConfirmDialog mounted in the shell: app-styled confirmations asked by the main process
 * - [2026-11-11] type=feat - 보유 현황 tab (PositionsView) between 차트 and 매매 결과
 */

import { useStore, type Tab } from './store';
import { StatusBar } from './components/StatusBar';
import { ConfirmDialog } from './components/ConfirmDialog';
import { ReplayBar } from './components/ReplayBar';
import { PipelineView } from './views/PipelineView';
import { ControlView } from './views/ControlView';
import { JobsView } from './views/JobsView';
import { BoardView } from './views/BoardView';
import { TimelineView } from './views/TimelineView';
import { ChartView } from './views/ChartView';
import { TradesView } from './views/TradesView';
import { PositionsView } from './views/PositionsView';
import { SettingsView } from './views/SettingsView';

// 16 px line icons (stroke = currentColor, so they follow the item's text color and the theme)
const ICON: Record<Tab, JSX.Element> = {
  control: (
    <>
      <path d="M2 4.5h7M12 4.5h2M2 11.5h2M7 11.5h7" />
      <circle cx="10.5" cy="4.5" r="1.5" />
      <circle cx="5.5" cy="11.5" r="1.5" />
    </>
  ),
  jobs: (
    <>
      <ellipse cx="8" cy="3.5" rx="5" ry="1.8" />
      <path d="M3 3.5v9c0 1 2.2 1.8 5 1.8s5-.8 5-1.8v-9M3 8c0 1 2.2 1.8 5 1.8S13 9 13 8" />
    </>
  ),
  pipeline: (
    <>
      <circle cx="4" cy="3.5" r="1.5" />
      <circle cx="4" cy="12.5" r="1.5" />
      <circle cx="12" cy="6" r="1.5" />
      <path d="M4 5v6M12 7.5c0 2.5-3 3-6.6 4.3" />
    </>
  ),
  board: <path d="M2.5 3.5h2M2.5 8h2M2.5 12.5h2M7 3.5h6.5M7 8h6.5M7 12.5h6.5" />,
  timeline: <path d="M1.5 11.5l3.5-4 3 2.5 5.5-6M11 4h2.5v2.5" />,
  chart: <path d="M4 2v12M4 5h-1.5v6H5.5V5H4M10.5 3v11M9 6h3v4.5H9z" />,
  positions: (
    <>
      <rect x="2" y="4.5" width="12" height="9" rx="1.5" />
      <path d="M5.5 4.5v-2h5v2M2 8.5h12" />
    </>
  ),
  trades: (
    <>
      <path d="M3.5 1.5h9v13l-1.5-1-1.5 1-1.5-1-1.5 1-1.5-1-1.5 1z" />
      <path d="M6 5.5h4M6 8.5h4" />
    </>
  ),
  settings: (
    <>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" />
    </>
  ),
};

// Sidebar groups: 운영 (run the bot and its data jobs) / 분석 (look at the day) / 시스템
const NAV: { group: string; tabs: [Tab, string][] }[] = [
  { group: '운영', tabs: [['control', '제어'], ['jobs', '데이터 작업'], ['pipeline', '파이프라인']] },
  { group: '분석', tabs: [['board', '종목 보드'], ['timeline', '타임라인'], ['chart', '차트'], ['positions', '보유 현황'], ['trades', '매매 결과']] },
  { group: '시스템', tabs: [['settings', '설정·연결']] },
];

export function App() {
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);
  return (
    <div className="app">
      <ConfirmDialog />
      <StatusBar />
      <div className="app-body">
        {/* sidebar: group captions, then one row per tab (icon + label); the current tab sits on a rounded highlight */}
        <nav className="sidenav" aria-label="탭">
          {NAV.map(({ group, tabs }) => (
            <div key={group} className="nav-group">
              <div className="nav-caption">{group}</div>
              {tabs.map(([k, label]) => (
                <button key={k} className={tab === k ? 'tab active' : 'tab'} onClick={() => setTab(k)} aria-current={tab === k ? 'page' : undefined}>
                  <svg className="nav-icon" viewBox="0 0 16 16" aria-hidden="true">
                    {ICON[k]}
                  </svg>
                  {label}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="app-main">
          {tab !== 'control' && tab !== 'settings' && tab !== 'jobs' && tab !== 'positions' && <ReplayBar />}
          <main className="view">
            {tab === 'control' && <ControlView />}
            {tab === 'jobs' && <JobsView />}
            {tab === 'pipeline' && <PipelineView />}
            {tab === 'board' && <BoardView />}
            {tab === 'timeline' && <TimelineView />}
            {tab === 'chart' && <ChartView />}
            {tab === 'positions' && <PositionsView />}
            {tab === 'trades' && <TradesView />}
            {tab === 'settings' && <SettingsView />}
          </main>
        </div>
      </div>
    </div>
  );
}
