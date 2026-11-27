/*
 * Update log
 * - [2026-11-27] type=test - shared/strategy: the method of a real trade (buy block / sell reason), the per-method day summary and
 *   the mode of each method from the control state
 */
import { describe, expect, it } from 'vitest';
import { DayEngine } from '../src/shared/engine';
import { buyStrategyOf, sellStrategyOf, sideExitKo, strategyModeKo, strategyModes, strategySummary } from '../src/shared/strategy';

const mode = (effective: string, running: string | null = effective) => ({ effective, running, pending: effective !== running && running !== null });

describe('trade method of a real trade', () => {
  it('reads the buy block and the sell reason', () => {
    expect(buyStrategyOf('VWAP_LIVE VWAP_SIGNAL_gap=+0.42%_tick=60')).toBe('vwap');
    expect(buyStrategyOf('EDGE_LIVE EDGE_AI_SIGNAL_p=0.612_thr=0.550_X_bar=10:01')).toBe('edge');
    expect(buyStrategyOf('reason=EDGE_AI_SIGNAL_p=0.6 signal=None')).toBe('edge');
    expect(buyStrategyOf('reason=TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20 signal=None live=1 bb_mid=1.0')).toBe('rule');
    expect(buyStrategyOf('')).toBe('rule');
    expect(sellStrategyOf('VWAP_TP_3.00')).toBe('vwap');
    expect(sellStrategyOf('VWAP_CLOSE_1520')).toBe('vwap');
    expect(sellStrategyOf('EDGE_SL_1.27')).toBe('edge');
    expect(sellStrategyOf('HARD_STOP_LOSS')).toBeNull();
    expect(sideExitKo('VWAP_TIME_300')).toBe('보유 시간');
    expect(sideExitKo('HARD_STOP_LOSS')).toBe('');
  });

  it('the engine tags each trade; a sell without a method reason takes the method of the last buy of the symbol', () => {
    const e = new DayEngine('20261127', { buy: [], sell: [] } as never);
    const blk = (t: string, code: string, side: string, tail: string, d: string[]) => [`2026-11-27 ${t} [INFO] >>> ${code}_X ${side} - ${tail}`, ...d.map((x) => `2026-11-27 ${t} [INFO] *** ${x}`)];
    e.feedBuySellLines([
      ...blk('09:10:00', '111111', 'Buy', 'reason=TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20 signal=None', ['매수시각=2026-11-27 09:10:00 | 매수가격=1,000 | 매수수량=10']),
      ...blk('09:20:00', '222222', 'Buy', 'VWAP_LIVE VWAP_SIGNAL_gap=+0.40%', ['매수시각=2026-11-27 09:20:00 | 매수가격=2,000 | 매수수량=5']),
      ...blk('09:30:00', '222222', 'Sell', '수익금액=300원 | 수익율=3.00% | reason=VWAP_TP_3.00 exch=KRX', ['매수시각=2026-11-27 09:20:00 | 매수가격=2,000 | 매수수량=5', '매도시각=2026-11-27 09:30:00 | 매도가격=2,060 | 매도수량=5']),
      ...blk('09:40:00', '333333', 'Buy', 'EDGE_LIVE EDGE_AI_SIGNAL_p=0.61', ['매수시각=2026-11-27 09:40:00 | 매수가격=3,000 | 매수수량=3']),
      ...blk('09:50:00', '333333', 'Sell', '수익금액=-90원 | 수익율=-1.00% | reason=MANUAL_SELL exch=KRX', ['매수시각=2026-11-27 09:40:00 | 매수가격=3,000 | 매수수량=3', '매도시각=2026-11-27 09:50:00 | 매도가격=2,970 | 매도수량=3']),
      ...blk('10:00:00', '111111', 'Sell', '수익금액=-100원 | 수익율=-1.00% | reason=HARD_STOP_LOSS exch=KRX', ['매수시각=2026-11-27 09:10:00 | 매수가격=1,000 | 매수수량=10', '매도시각=2026-11-27 10:00:00 | 매도가격=990 | 매도수량=10']),
    ]);
    const t = e.getTrades();
    expect(t.map((x) => x.strategy)).toEqual(['rule', 'vwap', 'vwap', 'edge', 'edge', 'rule']);
    const s = strategySummary(t);
    expect(s.vwap).toEqual({ buys: 1, sells: 1, wins: 1, pnlKrw: 300 });
    expect(s.edge).toEqual({ buys: 1, sells: 1, wins: 0, pnlKrw: -90 });
    expect(s.rule).toEqual({ buys: 1, sells: 1, wins: 0, pnlKrw: -100 });
  });
});

describe('mode of each method', () => {
  it('defaults: rule real, Edge AI paper, VWAP off', () => {
    const m = strategyModes({ strategy: mode('normal'), ruleBuy: mode('true'), paper: mode('paper'), edgeLive: mode('false'), vwapMode: mode('off'), vwapLive: mode('false') });
    expect([m.rule.live, m.edge.live, m.edge.paper, m.vwap.live, m.vwap.paper]).toEqual([true, false, true, false, false]);
    expect([strategyModeKo(m.rule), strategyModeKo(m.edge), strategyModeKo(m.vwap)]).toEqual(['실제 매매', '가상 매매', 'OFF']);
  });
  it('VWAP real orders need the VWAP trade on; with them on there are no new paper buys', () => {
    const on = strategyModes({ strategy: mode('normal'), vwapMode: mode('paper'), vwapLive: mode('true') });
    expect([on.vwap.live, on.vwap.paper]).toEqual([true, false]);
    const off = strategyModes({ strategy: mode('normal'), vwapMode: mode('off'), vwapLive: mode('true') });
    expect([off.vwap.live, off.vwap.paper]).toEqual([false, false]);
    const paper = strategyModes({ strategy: mode('normal'), vwapMode: mode('paper'), vwapLive: mode('false') });
    expect([paper.vwap.live, paper.vwap.paper]).toEqual([false, true]);
  });
  it('the running bot wins over the stored setting, and the difference is "pending"', () => {
    const m = strategyModes({ strategy: mode('normal'), vwapMode: mode('paper'), vwapLive: mode('true', 'false') });
    expect([m.vwap.live, m.vwap.paper, m.vwap.pending, m.vwap.fromRunning]).toEqual([false, true, true, true]);
  });
  it('nothing read: nothing is claimed', () => {
    const m = strategyModes({});
    expect([m.rule.live, m.edge.live, m.vwap.live, m.vwap.paper]).toEqual([null, null, null, null]);
    expect(strategyModeKo(m.vwap)).toBe('확인 안 됨');
  });
  it('TRADING_STRATEGY_MODE=edge: Edge AI places the real orders, no rule trade', () => {
    const m = strategyModes({ strategy: mode('edge'), ruleBuy: mode('true'), paper: mode('paper') });
    expect([m.rule.live, m.edge.live, m.edge.paper]).toEqual([false, true, false]);
  });
});
