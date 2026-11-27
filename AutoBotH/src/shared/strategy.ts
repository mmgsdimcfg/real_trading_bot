/*
 * Update log
 * - [2026-11-27] type=feat - Which trading method a real trade of the daily buy_sell log belongs to (rule / Edge AI / VWAP), the
 *   mode each method runs in (real orders / paper / off) and the per-method day summary - for the 매매 결과 tab (user 2026-11-27:
 *   "각 매매방식의 모드에 따른 매매 결과를 쉽게 확인"). Display only: read from what r003 already logs
 *   (buy block "VWAP_LIVE ..." / "EDGE_LIVE ...", sell reasons VWAP_* / EDGE_*), no server change
 */

import type { ControlState } from './control';
import type { Trade } from './types';

export type TradeStrategy = 'rule' | 'edge' | 'vwap';
export const TRADE_STRATEGIES: TradeStrategy[] = ['rule', 'edge', 'vwap'];
export const TRADE_STRATEGY_KO: Record<TradeStrategy, string> = { rule: '규칙 매매', edge: 'Edge AI 매매', vwap: 'VWAP 매매' };
export const TRADE_STRATEGY_SHORT: Record<TradeStrategy, string> = { rule: '규칙', edge: 'Edge AI', vwap: 'VWAP' };
export type TradeFilter = 'all' | TradeStrategy;
export const tradeFilterOf = (v: unknown): TradeFilter => (v === 'rule' || v === 'edge' || v === 'vwap' ? v : 'all');

/** the method of a buy block: the text after ">>> <code>_<name> Buy - " (r003 writes "VWAP_LIVE VWAP_SIGNAL_..." /
 *  "EDGE_LIVE EDGE_AI_SIGNAL_..." for the side strategies; TRADING_STRATEGY_MODE=edge buys carry EDGE_AI_SIGNAL_ as the reason) */
export function buyStrategyOf(tail: string): TradeStrategy {
  const s = String(tail || '').trim().replace(/^reason=/, '');
  if (/^VWAP_LIVE\b/.test(s) || /^VWAP_SIGNAL_/.test(s)) return 'vwap';
  if (/^EDGE_LIVE\b/.test(s) || /^EDGE_AI_SIGNAL_/.test(s) || /^EDGE_SIGNAL_/.test(s)) return 'edge';
  return 'rule';
}

/** the method of a sell by its reason code (VWAP_TP_ / VWAP_SL_ / VWAP_TIME_ / VWAP_CLOSE_1520, EDGE_...); null = the reason does
 *  not say (a rule exit, a manual sell ...) - the caller then takes the method of the symbol's last buy */
export function sellStrategyOf(reason: string): TradeStrategy | null {
  const s = String(reason || '').trim().replace(/^reason=/, '');
  if (/^VWAP_(TP|SL|TIME|CLOSE)/.test(s)) return 'vwap';
  if (/^EDGE_(TP|SL|TIME|CLOSE)/.test(s)) return 'edge';
  return null;
}

/** exit kind of a side-strategy sell ("익절" ...), '' when the reason is not one of theirs */
export function sideExitKo(reason: string | undefined): string {
  const m = /^(?:VWAP|EDGE)_(TP|SL|TIME|CLOSE)/.exec(String(reason || '').trim());
  return m ? { TP: '익절', SL: '손절', TIME: '보유 시간', CLOSE: '장 마감 청산' }[m[1] as 'TP' | 'SL' | 'TIME' | 'CLOSE'] : '';
}

export const strategyOfTrade = (t: Pick<Trade, 'strategy'>): TradeStrategy => t.strategy || 'rule';

export interface StrategyDaySummary {
  buys: number;
  sells: number;
  wins: number;
  pnlKrw: number;
}
/** real trades of the day per method (pnl from the sells as given - the caller passes net or gross rows) */
export function strategySummary(trades: Pick<Trade, 'side' | 'strategy' | 'pnlKrw'>[]): Record<TradeStrategy, StrategyDaySummary> {
  const out: Record<TradeStrategy, StrategyDaySummary> = {
    rule: { buys: 0, sells: 0, wins: 0, pnlKrw: 0 },
    edge: { buys: 0, sells: 0, wins: 0, pnlKrw: 0 },
    vwap: { buys: 0, sells: 0, wins: 0, pnlKrw: 0 },
  };
  for (const t of trades) {
    const e = out[strategyOfTrade(t)];
    if (t.side === 'BUY') e.buys++;
    else {
      e.sells++;
      const p = Number.isFinite(t.pnlKrw as number) ? (t.pnlKrw as number) : 0;
      e.pnlKrw += p;
      if (p > 0) e.wins++;
    }
  }
  return out;
}

/** what a method does now. null = the server did not say (not read yet / older server) */
export interface StrategyModeInfo {
  /** real orders */
  live: boolean | null;
  /** paper (virtual) trades; rule trading has none */
  paper: boolean | null;
  /** the stored setting differs from the running bot (applied after an r003 restart) */
  pending: boolean;
  /** from the running bot (false = the bot is not running / unknown: the stored settings) */
  fromRunning: boolean;
}
type Mode = { effective: string; running: string | null; pending: boolean } | undefined;
const val = (m: Mode): string | null => (m ? m.running ?? m.effective : null);

/** the mode of each method from the control state (r016 settings: running value when the bot runs, else the stored one) */
export function strategyModes(c: Pick<ControlState, 'strategy' | 'ruleBuy' | 'paper' | 'paperLogOnly' | 'edgeLive' | 'vwapMode' | 'vwapLive'>): Record<TradeStrategy, StrategyModeInfo> {
  const sm = val(c.strategy);
  const edgeOnly = sm === 'edge'; // TRADING_STRATEGY_MODE=edge: the Edge AI strategy places the real orders, no rule trade
  const normal = sm === null ? null : sm === 'normal';
  const pend = (...ms: Mode[]) => ms.some((m) => !!m?.pending);
  const run = (...ms: Mode[]) => ms.some((m) => !!m && m.running !== null);
  const ruleBuy = val(c.ruleBuy);
  const edgeLive = val(c.edgeLive);
  const paper = val(c.paper);
  const logOnly = val(c.paperLogOnly) === 'true';
  const vwapMode = val(c.vwapMode);
  const vwapLive = val(c.vwapLive);
  const edgeLiveOn = edgeOnly ? true : edgeLive === null ? null : edgeLive === 'true' && normal !== false;
  const vwapLiveOn = vwapLive === null || vwapMode === null ? null : vwapLive === 'true' && vwapMode === 'paper' && normal !== false;
  return {
    rule: {
      live: edgeOnly ? false : ruleBuy === null ? (sm === null ? null : true) : ruleBuy === 'true',
      paper: false,
      pending: pend(c.strategy, c.ruleBuy),
      fromRunning: run(c.strategy, c.ruleBuy),
    },
    edge: {
      live: edgeLiveOn,
      // with real orders on (or log-only) the paper runner records candidates only: no new paper buys
      paper: paper === null ? null : paper === 'paper' && !edgeOnly && edgeLiveOn !== true && !logOnly,
      pending: pend(c.strategy, c.edgeLive, c.paper),
      fromRunning: run(c.strategy, c.edgeLive, c.paper),
    },
    vwap: {
      live: vwapLiveOn,
      paper: vwapMode === null ? null : vwapMode === 'paper' && vwapLiveOn !== true,
      pending: pend(c.vwapMode, c.vwapLive),
      fromRunning: run(c.vwapMode, c.vwapLive),
    },
  };
}

export const strategyModeKo = (m: StrategyModeInfo): string => (m.live === null && m.paper === null ? '확인 안 됨' : m.live ? '실제 매매' : m.paper ? '가상 매매' : 'OFF');
