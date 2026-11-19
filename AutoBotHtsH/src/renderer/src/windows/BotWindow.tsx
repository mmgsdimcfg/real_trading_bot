// Update log
// - [2026-11-20] type=feat (AutoBotHtsH: one AutoBotH view (src/bot/renderer/src/views) inside a workspace window, under .bot-root so
//   the AutoBotH stylesheet (styles.scoped.css) and colour tokens apply only there. The views keep their own state (bot store); the
//   window that gets the focus becomes the bot store's current tab. The replay bar is shown where AutoBotH showed it (not on 제어 /
//   데이터 작업 / 수집기 / 보유 현황 / 가상 매매 / 설정). BotStatusStrip: the AutoBotH status line under the toolbar; BotConfirm: its
//   app-styled confirmation dialog)
// - [2026-11-20] type=feat (A symbol picked in a bot window (timeline, board, positions, trades, paper, bot chart, timeline 자동 보기)
//   is shown in every open HTS symbol window too - chart, current price, stock info, news (followCode; nothing is opened). One way
//   only: a symbol picked in an HTS window does not change the bot's selection - user 2026-11-20)

import { useEffect, type ReactNode } from 'react';
import { BOT_TABS, isBotWin, useStore, type BotTab, type Win } from '../store';
import { setTabHandlers, useStore as useBot } from '../../../bot/renderer/src/store';
import { StatusBar } from '../../../bot/renderer/src/components/StatusBar';
import { ConfirmDialog } from '../../../bot/renderer/src/components/ConfirmDialog';
import { ReplayBar } from '../../../bot/renderer/src/components/ReplayBar';
import { PipelineView } from '../../../bot/renderer/src/views/PipelineView';
import { ControlView } from '../../../bot/renderer/src/views/ControlView';
import { JobsView } from '../../../bot/renderer/src/views/JobsView';
import { CollectorView } from '../../../bot/renderer/src/views/CollectorView';
import { BoardView } from '../../../bot/renderer/src/views/BoardView';
import { TimelineView } from '../../../bot/renderer/src/views/TimelineView';
import { ChartView } from '../../../bot/renderer/src/views/ChartView';
import { TradesView } from '../../../bot/renderer/src/views/TradesView';
import { PositionsView } from '../../../bot/renderer/src/views/PositionsView';
import { PaperView } from '../../../bot/renderer/src/views/PaperView';
import { SettingsView } from '../../../bot/renderer/src/views/SettingsView';

const VIEW: Record<BotTab, () => ReactNode> = {
  control: ControlView,
  jobs: JobsView,
  collector: CollectorView,
  pipeline: PipelineView,
  board: BoardView,
  timeline: TimelineView,
  chart: ChartView,
  positions: PositionsView,
  trades: TradesView,
  paper: PaperView,
  settings: SettingsView,
};
const NO_REPLAY: BotTab[] = ['control', 'settings', 'jobs', 'collector', 'positions', 'paper'];

export const botTabOf = (w: Win): BotTab | null => {
  if (!isBotWin(w.type)) return null;
  const t = w.type.slice(4) as BotTab;
  return (BOT_TABS as readonly string[]).includes(t) ? t : null;
};

/** bot store <-> workspace: a tab request opens / focuses its window; a tab is "shown" while its window is open and not minimized */
export function linkBotTabs(): void {
  setTabHandlers({
    open: (t) => void useStore.getState().openBot(t),
    shown: (t) => useStore.getState().wins.some((w) => botTabOf(w) === t && !w.min),
  });
  useBot.subscribe((s, prev) => {
    if (s.selected && s.selected !== prev.selected && /^[0-9A-Z]{6}$/.test(s.selected)) useStore.getState().followCode(s.selected);
  });
}

export function BotWindow({ win }: { win: Win }) {
  const tab = botTabOf(win);
  const active = useStore((s) => s.active === win.id);
  useEffect(() => {
    if (tab && (active || !useBot.getState().tab) && useBot.getState().tab !== tab) useBot.setState({ tab });
  }, [active, tab]);
  if (!tab) return <div className="empty">알 수 없는 봇 창입니다.</div>;
  const View = VIEW[tab];
  return (
    <div className="bot-root bot-win">
      <div className="app-main">
        {!NO_REPLAY.includes(tab) && <ReplayBar />}
        <main className="view">
          <View />
        </main>
      </div>
    </div>
  );
}

/** AutoBotH status line (connection · bot · day · details); its items open the bot windows */
export function BotStatusStrip() {
  return (
    <div className="bot-root bot-strip">
      <StatusBar embedded />
    </div>
  );
}

export function BotConfirm() {
  return (
    <div className="bot-root bot-confirm">
      <ConfirmDialog />
    </div>
  );
}
