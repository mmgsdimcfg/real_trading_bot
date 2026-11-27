import { describe, expect, it } from 'vitest';
import { filterMethodRows, methodRows, methodTotals, sideSummaries } from '../../src/bot/shared/methodBoard';
import type { StrategyEvent } from '../../src/bot/shared/strategyEvents';
import type { SymbolSummary, Trade } from '../../src/bot/shared/types';
import type { StrategyEvalRowX } from '../../src/bot/shared/sideTimeline';

const ev = (p: Partial<StrategyEvent>): StrategyEvent => ({ id: 1, ts: 1000, code: '000001', strategy: 'vwap', mode: 'live', kind: 'note', text: '', raw: 'raw', ...p });
const sym = (p: Partial<SymbolSummary>): SymbolSummary => ({ code: '000001', name: 'A', watch: 'active', furthestOrderToday: -1, blockCounts: {}, sellCounts: {}, realizedKrw: 0, evalCount: 0, ...p });
const trade = (p: Partial<Trade>): Trade => ({ code: '000001', name: 'A', side: 'BUY', ts: 1000, price: 100, qty: 1, reason: '', raw: '', ...p });

describe('sideSummaries', () => {
  it('counts signals / orders per strategy and keeps the holding until an exit', () => {
    const s = sideSummaries([
      ev({ ts: 1, kind: 'signal', reason: 'VWAP_SIGNAL' }),
      ev({ ts: 2, kind: 'order' }),
      ev({ ts: 3, kind: 'hold', qty: 5, avg: 100, pnlPct: 1.2 }),
      ev({ ts: 4, kind: 'signal', strategy: 'edge' }),
    ]);
    expect(s.vwap).toMatchObject({ signals: 1, orders: 1 });
    expect(s.vwap?.holding?.qty).toBe(5);
    expect(s.vwap?.last?.kind).toBe('order'); // the holding check is not the "last record"
    expect(s.edge?.signals).toBe(1);
    const closed = sideSummaries([ev({ ts: 3, kind: 'hold', qty: 5 }), ev({ ts: 5, kind: 'exit', reason: 'VWAP_TP' })]);
    expect(closed.vwap?.holding?.qty).toBe(5); // a decision is not a sale: the order may not fill
    expect(closed.vwap?.exitAt).toBe(5);
    const sold = sideSummaries([ev({ ts: 3, kind: 'hold', qty: 5 }), ev({ ts: 5, kind: 'exit' }), ev({ ts: 6, kind: 'fill', reason: 'sell' })]);
    expect(sold.vwap?.holding).toBeUndefined();
    // the bot's ledger going to 0 ends it too (sold by hand: no line of the strategy says so)
    const gone = sideSummaries([ev({ ts: 3, kind: 'hold', qty: 5 }), ev({ ts: 5, kind: 'exit' }), ev({ ts: 9, kind: 'gone', qty: 5, text: 'account=6->0 bot=5->0' })]);
    expect(gone.vwap?.holding).toBeUndefined();
    expect(gone.vwap?.exitAt).toBeUndefined();
    expect(gone.vwap?.last?.kind).toBe('gone');
  });
});

describe('methodRows', () => {
  const symbols = [
    sym({ code: '000001', name: 'A', side: sideSummaries([ev({ ts: 5000, kind: 'signal', reason: 'VWAP_SIGNAL' }), ev({ ts: 6000, kind: 'hold', qty: 5, avg: 100, pnlPct: -0.5 })]) }),
    sym({ code: '000002', name: 'B' }),
    sym({ code: '000003', name: 'C', watch: 'graduated' }),
  ];

  it('side method: watched symbols are targets, a symbol without a record says so', () => {
    const rows = methodRows('vwap', { symbols, trades: [] });
    expect(rows.map((r) => r.code)).toEqual(['000001', '000002']); // the holding first; the graduated symbol has nothing of VWAP
    expect(rows[0]).toMatchObject({ watched: true, extra: false, holdQty: 5, holdAvg: 100, signals: 1, stateSrc: 'log', stateKind: 'pass' });
    expect(rows[1]).toMatchObject({ watched: true, state: '기록 없음', stateSrc: '', stateKind: 'none' });
    expect(methodTotals(rows)).toMatchObject({ targets: 2, watched: 2, recorded: 1, holding: 1, signals: 1 });
  });

  it('the newest record wins: a later trace row replaces the log line, an older one does not', () => {
    const row = (ts: number): StrategyEvalRowX => ({ ts, code: '000001', mode: 'live', stage: 'below_vwap', extra: { vwap: 101 } }) as StrategyEvalRowX;
    const newer = methodRows('vwap', { symbols, trades: [], evalLast: { '000001': row(9000) }, evalCount: { '000001': 12 } })[0];
    expect(newer.stateSrc).toBe('trace');
    expect(newer.evals).toBe(12);
    expect(newer.stateKind).toBe('fail');
    const older = methodRows('vwap', { symbols, trades: [], evalLast: { '000001': row(10) } })[0];
    expect(older.stateSrc).toBe('log');
  });

  it('a symbol known only from a trade or the trace is listed as a record, not as a target', () => {
    const rows = methodRows('vwap', { symbols, trades: [trade({ code: '000009', name: 'Z', strategy: 'vwap' }), trade({ code: '000002', strategy: 'rule' })] });
    const z = rows.find((r) => r.code === '000009');
    expect(z).toMatchObject({ extra: true, watched: false, buys: 1 });
    expect(rows.find((r) => r.code === '000002')?.buys).toBe(0); // the rule buy is not VWAP's
    expect(methodTotals(rows).targets).toBe(2);
    expect(filterMethodRows(rows, 'target').map((r) => r.code)).not.toContain('000009');
    expect(filterMethodRows(rows, 'holding').map((r) => r.code)).toEqual(['000001']);
  });

  it('edge paper: collector symbols of the paper status are targets, paper trades are counted apart from real ones', () => {
    const rows = methodRows('edge', {
      symbols,
      trades: [],
      paperStatus: [{ code: '000050', name: 'K', group: 'collector', bars: 10, stage: 'no_trigger', at: 7000 }],
      paperTrades: [
        { code: '000050', name: 'K', group: 'collector', session: 'regular', entryTs: 1, entry: 50, exitTs: 2, exit: 51, netPct: 1.5 },
        { code: '000050', name: 'K', group: 'collector', session: 'regular', entryTs: 3, entry: 52 },
      ] as never,
    });
    const k = rows.find((r) => r.code === '000050');
    expect(k).toMatchObject({ collector: true, extra: false, stateSrc: 'paper', stateMode: 'paper', paperBuys: 2, paperSells: 1, paperEntry: 52, buys: 0 });
    // a collector symbol with a past paper trade but no status row is a record, not a current target
    const past = methodRows('edge', { symbols, trades: [], paperTrades: [{ code: '000060', name: 'P', group: 'collector', session: 'regular', entryTs: 1, entry: 5, exitTs: 2, exit: 5, netPct: 0 }] as never });
    expect(past.find((r) => r.code === '000060')).toMatchObject({ collector: true, extra: true });
    expect(k?.paperNetPct).toBeCloseTo(1.5);
    // the method places real orders: the paper runner's stage is not its state (positions / groups still are)
    const live = methodRows('edge', { symbols, trades: [], paperStateOff: true, paperStatus: [{ code: '000050', name: 'K', group: 'collector', bars: 10, stage: 'no_trigger', at: 7000, position: { entry: 52, tp: 1, sl: 1, ts: 1 } }] });
    expect(live.find((r) => r.code === '000050')).toMatchObject({ collector: true, stateSrc: '', state: '기록 없음', paperEntry: 52 });
  });

  it('rule: the last evaluation with the condition title, holdings from rule buys only', () => {
    const rs = [
      sym({ code: '000001', evalCount: 7, lastEval: { id: 1, ts: 4000, code: '000001', side: 'buy', tag: 'BUY', condKey: '_009', result: 'fail', reason: 'MA5_BELOW', raw: 'r' } as never, position: { qty: 3, buyPrice: 100, buyTs: 1 } }),
      sym({ code: '000002', position: { qty: 2, buyPrice: 50, buyTs: 1 } }),
    ];
    const rows = methodRows('rule', {
      symbols: rs,
      trades: [trade({ code: '000001', qty: 3, price: 100 }), trade({ code: '000001', side: 'SELL', qty: 1, ts: 2000 }), trade({ code: '000002', strategy: 'vwap' })],
      condTitle: (k) => `조건${k}`,
      reasonKo: () => '',
    });
    const a = rows.find((r) => r.code === '000001');
    expect(a).toMatchObject({ state: '막힘 · 조건_009 · MA5_BELOW', stateKind: 'fail', stateSrc: 'rule', evals: 7, holdQty: 2, holdAvg: 100, buys: 1, sells: 1 }); // 3 bought, 1 sold: 2 still held
    expect(rows.find((r) => r.code === '000002')?.holdQty).toBeUndefined(); // bought by VWAP, not a rule holding
  });

  it('rule: a later order record replaces the evaluation that passed', () => {
    const rs = [sym({ lastEval: { id: 1, ts: 4000, code: '000001', side: 'buy', tag: 'BUY', condKey: '_025', result: 'pass', reason: '', raw: 'e' } as never, lastOrder: { id: 2, ts: 4100, code: '000001', side: 'buy', tag: 'ORDER', condKey: 'ORDER', result: 'fail', reason: 'ORDER_REJECTED', raw: 'o' } as never })];
    const r = methodRows('rule', { symbols: rs, trades: [] })[0];
    expect(r).toMatchObject({ stateKind: 'fail', stateTs: 4100, stateRaw: 'o' });
  });

  it('side: a partly sold position is still held, a fully sold one is not', () => {
    const held = [sym({ side: sideSummaries([ev({ ts: 1, kind: 'hold', qty: 5, avg: 100 })]) })];
    const part = methodRows('vwap', { symbols: held, trades: [trade({ strategy: 'vwap', qty: 5 }), trade({ strategy: 'vwap', side: 'SELL', qty: 2 })] })[0];
    expect(part.holdQty).toBe(5); // as the bot last printed it
    const all = methodRows('vwap', { symbols: held, trades: [trade({ strategy: 'vwap', qty: 5 }), trade({ strategy: 'vwap', side: 'SELL', qty: 5 })] })[0];
    expect(all.holdQty).toBeUndefined();
    // a holding check written AFTER the sale: the bot still holds (shares carried over from an earlier day)
    const later = [sym({ side: sideSummaries([ev({ ts: 9000, kind: 'hold', qty: 5, avg: 100 })]) })];
    const kept = methodRows('vwap', { symbols: later, trades: [trade({ strategy: 'vwap', qty: 2, ts: 1000 }), trade({ strategy: 'vwap', side: 'SELL', qty: 2, ts: 2000 })] })[0];
    expect(kept.holdQty).toBe(5);
  });
});
