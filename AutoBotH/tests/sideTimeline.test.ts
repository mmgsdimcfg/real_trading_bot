/*
 * Update log
 * - [2026-11-27] type=test - Side strategies on the timeline: the engine keeps their log lines as strategy events and notes, a
 *   filled side order without a fill block becomes a trade (by order number), the chart rows / dots of a method tab, the
 *   paper trades as markers, which tab opens first. Lines are the ones r003 wrote on 2026-11-27 (VWAP real orders on at 14:00)
 */
import { describe, expect, it } from 'vitest';
import { DayEngine } from '../src/shared/engine';
import { blockReasonKo, parseStrategyRest, stageRowOf, STAGE_ROWS } from '../src/shared/strategyEvents';
import { defaultTimelineMethod, evalRowKo, paperTradesAsTrades, parseStrategyEvalRows, sideTimeline } from '../src/shared/sideTimeline';
import { strategySummary } from '../src/shared/strategy';
import type { PaperTrade } from '../src/shared/paper';

const D = '2026-11-27';
const MAIN = [
  `${D} 09:00:00,000 [INFO] WATCH | 237690 | 에스티팜 | NXT=True`,
  `${D} 09:00:00,000 [INFO] WATCH | 068270 | 셀트리온 | NXT=True`,
  `${D} 09:30:23,432 [INFO]   [VWAP PAPER  ] 237690_에스티팜 | PAPER BUY 86,500 VWAP_gap=+1.62%_MA5_UP_TICK60_MA5_MA15@09:30:02 | tp 89,095 sl 85,401 300min | open=1 entries=1/20`,
  `${D} 14:00:31,515 [INFO]   [VWAP LIVE   ] 신규 매수 켜짐 | 1회 500,000원 동시 5 하루 20회 손실 한도 50,000원`,
  `${D} 14:08:18,029 [INFO]   [VWAP LIVE   ] 237690_에스티팜   | 매수 신호 | VWAP_SIGNAL_gap=+1.34%_MA5_UP_TICK60_MA5_MA15 | live=87,900 qty<=5`,
  `${D} 14:08:18,625 [INFO]   [VWAP LIVE   ] 237690_에스티팜   | 매수 주문 접수 | 상한 500,000원`,
  // before 2026-11-27 the fill lines were always tagged EDGE LIVE - they belong to the strategy that ordered
  `${D} 14:08:21,737 [INFO]   [EDGE LIVE   ] 237690_에스티팜   | buy 체결 5주 반영 (order 0014965900) -> Edge 5주 원가 87,900`,
  `${D} 14:08:37,508 [INFO]   [VWAP LIVE   ] 237690_에스티팜   | qty=5 원가=87,900 현재가=87,800(live) 손익=-0.11% 보유=0분 | tp=+3.00% sl=-1.27% 300분`,
  `${D} 14:12:29,558 [INFO]   [VWAP LIVE   ] 068270_셀트리온   | 매수 안 함 | _023_affordable_buy_qty | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price=222,000`,
  `${D} 14:22:22,657 [INFO]   [VWAP LIVE   ] 신규 매수 중단 | MIXED_HOLDING_237690`,
  `${D} 14:25:42,637 [INFO]   [VWAP LIVE   ] 237690_에스티팜   | qty=5 원가=87,900 현재가=87,400(live) 손익=-0.57% 보유=17분 | tp=+3.00% sl=-1.27% 300분 | 계좌 6주 (직접 보유분은 팔지 않음)`,
  `${D} 14:30:18,239 [INFO]   [VWAP PAPER  ] 237690_에스티팜 | PAPER SELL 87,500 time net=+0.92% held=300min | day: 4 trades 2 wins sum=+0.31%`,
  `${D} 14:32:23,364 [INFO]   [VWAP LIVE   ] 신규 매수 중단 | MIXED_HOLDING_237690`,
];
const BUYSELL = [
  `${D} 14:08:18 [INFO] [BUY SUBMIT  ] 237690_에스티팜   | qty=5 | price=87,900 | KRX | order_no=0014965900 | reason=VWAP_LIVE VWAP_SIGNAL_gap=+1.34%_MA5_UP_TICK60_MA5_MA15`,
  `${D} 14:08:21 [INFO] [BUY PENDING ] 237690_에스티팜   | filled=5/5 | remaining=0 | order_no=0014965900`,
];
const engine = () => {
  const e = new DayEngine('20261127', { buy: [], sell: [] } as never);
  e.feedMainLines(MAIN);
  e.feedBuySellLines(BUYSELL);
  return e;
};

describe('the lines a side strategy writes', () => {
  it('are read as they are', () => {
    expect(parseStrategyRest('live', '매수 신호 | VWAP_SIGNAL_gap=+1.34% | live=87,900 qty<=5')).toMatchObject({ kind: 'signal', reason: 'VWAP_SIGNAL_gap=+1.34%', price: 87900, qty: 5 });
    expect(parseStrategyRest('live', 'qty=5 원가=87,900 현재가=87,400(live) 손익=-0.57% 보유=17분 | tp=+3.00% sl=-1.27% 300분')).toMatchObject({ kind: 'hold', qty: 5, avg: 87900, price: 87400, pnlPct: -0.57, tpPct: 3, slPct: -1.27 });
    expect(parseStrategyRest('live', '청산 VWAP_SL_1.27 | qty=5 원가=87,900 현재가=86,700(live) 손익=-1.37%')).toMatchObject({ kind: 'exit', reason: 'VWAP_SL_1.27', pnlPct: -1.37 });
    expect(parseStrategyRest('live', '매수 안 함 | _023_affordable_buy_qty | INSUFFICIENT_BUYING_POWER_OR_BUDGET | price=222,000')).toMatchObject({ kind: 'nobuy', reason: '_023_affordable_buy_qty | INSUFFICIENT_BUYING_POWER_OR_BUDGET', price: 222000 });
    expect(parseStrategyRest('paper', 'PAPER SELL 18,159 tp net=+2.76% held=22min | day: 1 trades')).toMatchObject({ kind: 'paper_sell', price: 18159, reason: 'tp', pnlPct: 2.76 });
    expect(parseStrategyRest('live', '청산 VWAP_TP_3.00 대기 - 매수 잔량 취소 확인 뒤 매도').kind).toBe('note');
    expect(blockReasonKo('MIXED_HOLDING_237690')).toContain('직접 산 주식');
    expect(blockReasonKo('SOMETHING_NEW')).toBe('SOMETHING_NEW');
  });

  it('the engine keeps them per symbol, not as unknown lines, and the notes per strategy', () => {
    const e = engine();
    const tl = e.timeline('237690');
    expect(tl.events).toEqual([]);
    expect(tl.strategy.map((x) => `${x.strategy}/${x.mode}/${x.kind}`)).toEqual([
      'vwap/paper/paper_buy',
      'vwap/live/signal',
      'vwap/live/order',
      'vwap/live/fill', // tagged EDGE LIVE in the log: the strategy that ordered the symbol
      'vwap/live/hold',
      'vwap/live/hold',
      'vwap/paper/paper_sell',
    ]);
    const snap = e.snapshot();
    expect(snap.warnings).toBe(0);
    const notes = snap.strategyNotes!;
    expect(notes.map((n) => n.kind)).toEqual(['on', 'blocked']);
    // the repeated note is one note with its first and last time
    expect(notes[1].text).toBe('MIXED_HOLDING_237690');
    expect(notes[1].lastTs - notes[1].ts).toBe(10 * 60 * 1000 + 707);
  });
});

describe('a filled side order without a fill block', () => {
  it('is a trade of its strategy, built from the order lines by order number', () => {
    const e = engine();
    const t = e.getTrades();
    expect(t).toHaveLength(1);
    expect(t[0]).toMatchObject({ code: '237690', side: 'BUY', strategy: 'vwap', price: 87900, qty: 5, approx: true, orderNo: '14965900' });
    expect(strategySummary(t).vwap.buys).toBe(1);
    expect(e.snapshot().symbols.find((s) => s.code === '237690')?.position).toMatchObject({ qty: 5, buyPrice: 87900 });
    // the cumulative filled quantity is the last one seen, never a sum
    e.feedBuySellLines([`${D} 14:08:25 [INFO] [BUY PENDING ] 237690_에스티팜   | filled=5/5 | remaining=0 | order_no=0014965900`]);
    expect(e.getTrades()[0].qty).toBe(5);
  });

  it('is not doubled when the block exists, and an unfilled or rule order adds nothing', () => {
    const e = engine();
    e.feedBuySellLines([
      `${D} 14:08:30 [INFO] >>> 237690_에스티팜 Buy - VWAP_LIVE VWAP_SIGNAL_gap=+1.34%_MA5_UP_TICK60_MA5_MA15`,
      `${D} 14:08:30 [INFO] *** 매수시각=2026-11-27 14:08:18 | 매수가격=87,850 | 매수수량=5`,
      `${D} 14:40:00 [INFO] [BUY SUBMIT  ] 068270_셀트리온   | qty=2 | price=222,000 | KRX | order_no=0015000000 | reason=VWAP_LIVE VWAP_SIGNAL_x`,
      `${D} 14:41:00 [INFO] [BUY SUBMIT  ] 068270_셀트리온   | qty=2 | price=222,000 | KRX | order_no=0015000001 | reason=TRIPLE_CROSS_1MIN`,
      `${D} 14:41:05 [INFO] [BUY PENDING ] 068270_셀트리온   | filled=2/2 | remaining=0 | order_no=0015000001`,
    ]);
    const t = e.getTrades();
    expect(t.map((x) => [x.code, x.price, !!x.approx])).toEqual([['237690', 87850, false]]);
  });

  it('one block answers one order: a second order next to it is not swallowed, and a late detail line is still matched', () => {
    // [Codex round 3] the block header arrives first, its detail line (with the order time) after the list was read once
    const e = engine();
    e.feedBuySellLines([`${D} 14:08:30 [INFO] >>> 237690_에스티팜 Buy - VWAP_LIVE VWAP_SIGNAL_gap=+1.34%_MA5_UP_TICK60_MA5_MA15`]);
    expect(e.getTrades()).toHaveLength(1); // header time 14:08:30 is within the window of the order: the block answers it
    e.feedBuySellLines([`${D} 14:08:30 [INFO] *** 매수시각=2026-11-27 14:08:10 | 매수가격=87,850 | 매수수량=5`]);
    expect(e.getTrades().map((x) => [x.price, !!x.approx])).toEqual([[87850, false]]); // 8 s before the SUBMIT line: still the same order
    // another filled order 10 s later has no block of its own: it stays a trade
    e.feedBuySellLines([
      `${D} 14:08:28 [INFO] [BUY SUBMIT  ] 237690_에스티팜   | qty=1 | price=87,950 | KRX | order_no=0014965999 | reason=VWAP_LIVE VWAP_SIGNAL_z`,
      `${D} 14:08:29 [INFO] [BUY PENDING ] 237690_에스티팜   | filled=1/1 | remaining=0 | order_no=0014965999`,
    ]);
    expect(e.getTrades().map((x) => [x.price, x.qty, !!x.approx])).toEqual([[87850, 5, false], [87950, 1, true]]);
  });

  it('only the old EDGE LIVE fill lines follow the last ordering strategy; a VWAP LIVE tag is what it says', () => {
    const e = new DayEngine('20261127', { buy: [], sell: [] } as never);
    e.feedMainLines([
      `${D} 09:00:00,000 [INFO] WATCH | 237690 | 에스티팜 | NXT=True`,
      `${D} 10:00:00,000 [INFO]   [EDGE LIVE   ] 237690_에스티팜   | 매수 신호 | EDGE_AI_SIGNAL_p=0.6 | live=87,000 qty<=5`,
      `${D} 10:00:03,000 [INFO]   [VWAP LIVE   ] 237690_에스티팜   | buy 체결 5주 반영 (order 1) -> VWAP 5주 원가 87,000`,
      `${D} 10:00:04,000 [INFO]   [EDGE LIVE   ] 237690_에스티팜   | buy 체결 5주 반영 (order 2) -> Edge 5주 원가 87,000`,
    ]);
    expect(e.timeline('237690').strategy.map((x) => x.strategy)).toEqual(['edge', 'vwap', 'edge']);
  });

  it('a second order of the symbol later in the day is its own trade', () => {
    const e = engine();
    e.feedBuySellLines([
      `${D} 14:50:00 [INFO] [BUY SUBMIT  ] 237690_에스티팜   | qty=3 | price=88,000 | KRX | order_no=0015100000 | reason=VWAP_LIVE VWAP_SIGNAL_y`,
      `${D} 14:50:04 [INFO] [BUY PENDING ] 237690_에스티팜   | filled=2/3 | remaining=1 | order_no=0015100000`,
    ]);
    expect(e.getTrades().map((x) => [x.orderNo, x.qty])).toEqual([['14965900', 5], ['15100000', 2]]);
  });
});

describe('a method tab of the timeline', () => {
  const evals = parseStrategyEvalRows(
    [
      '{"t": "2026-11-27T14:05:02", "code": "237690", "mode": "live", "stage": "below_vwap", "bar": "14:04", "price": 87300.0, "vwap": 87400.0, "gap": -0.11, "no": 9, "cond": "_v09_vwap_signal", "detail": "VWAP_SIGNAL_BELOW_VWAP"}',
      '{"t": "2026-11-27T14:06:02", "code": "237690", "mode": "live", "stage": "no_tick_cross", "bar": "14:05", "price": 87600.0, "vwap": 87400.0, "gap": 0.23, "tick": "TICK60_NO_CROSS"}',
      '{"t": "2026-11-27T14:07:02", "code": "237690", "mode": "live", "stage": "cond_before", "bar": "14:06", "no": 8, "detail": "MAX_OPEN_POSITIONS open=6 max=6"}',
      '{"t": "2026-11-27T14:08:18", "code": "237690", "mode": "live", "stage": "ordered", "bar": "14:07", "price": 87900.0, "gap": 1.34, "reason": "VWAP_SIGNAL_x", "qty": 5}',
      '{"t": "2026-11-27T14:09:0',
    ].join('\n'),
  );

  it('reads the trace rows with the values the bot used and skips a half-written row', () => {
    expect(evals).toHaveLength(4);
    expect(evals[0]).toMatchObject({ code: '237690', mode: 'live', stage: 'below_vwap', no: 9, price: 87300 });
    expect(evalRowKo(evals[0])).toBe('VWAP 아래 · VWAP 87,400 · VWAP 대비 -0.11% · 현재가 87,300 · 조건 _v09_vwap_signal · VWAP_SIGNAL_BELOW_VWAP');
    expect(evals.map((r) => stageRowOf('vwap', r))).toEqual(['vwap', 'tick', 'before', 'order']);
    expect(stageRowOf('edge', { stage: 'below_threshold' })).toBe('prob');
    expect(stageRowOf('vwap', { stage: 'something_new' })).toBe('data');
  });

  it('draws the stage of each judged bar and the strategy events on its own rows, bottom-up', () => {
    const tl = sideTimeline('vwap', engine().timeline('237690').strategy, evals);
    expect(tl.buyRows.map((r) => r.key)).toEqual([...STAGE_ROWS.vwap].reverse().map((r) => `S:${r.key}`));
    expect(tl.buyRows[tl.buyRows.length - 1].label).toBe('주문');
    const buy = tl.events.filter((e) => e.side === 'buy').map((e) => [e.condKey, e.result]);
    expect(buy).toEqual([
      ['S:order', 'pass'], // 09:30 paper buy
      ['S:vwap', 'fail'],
      ['S:tick', 'fail'],
      ['S:before', 'fail'],
      ['S:order', 'pass'], // the trace's ordered row (14:08:18.000)
      ['S:signal', 'pass'], // the log's signal line (14:08:18.029)
      ['S:order', 'pass'], // order accepted
      ['S:order', 'pass'], // fill applied
    ]);
    const sell = tl.events.filter((e) => e.side === 'sell');
    expect(sell.map((e) => [e.condKey, e.result])).toEqual([['S:hold', 'hold'], ['S:hold', 'hold'], ['S:exit', 'pass']]);
    expect(sell[1].ko).toBe('보유 5주 · 손익 -0.57% · 현재가 87,400 / 원가 87,900 · 익절 +3.00% / 손절 -1.27%');
    expect(tl.holding).toMatchObject({ kind: 'hold', pnlPct: -0.57, qty: 5 });
    expect(tl.traced).toBe(true);
    expect(tl.lastEval?.stage).toBe('ordered');
    // every dot has a row
    const keys = new Set([...tl.buyRows, ...tl.sellRows].map((r) => r.key));
    expect(tl.events.every((e) => keys.has(e.condKey) && !!e.ko && !!e.rowLabel)).toBe(true);
    // the other strategy's tab has none of it; without a trace only the log events remain
    expect(sideTimeline('edge', engine().timeline('237690').strategy, []).events).toEqual([]);
    const noTrace = sideTimeline('vwap', engine().timeline('237690').strategy, []);
    expect(noTrace.traced).toBe(false);
    expect(noTrace.events.length).toBe(tl.events.length - 4);
    // replay: trace rows after the cursor are left out
    expect(sideTimeline('vwap', [], evals, evals[1].ts).events).toHaveLength(2);
  });

  it('a condition after the signal that refused the buy is drawn above the signal', () => {
    const tl = sideTimeline('vwap', engine().timeline('068270').strategy, []);
    expect(tl.events.map((e) => [e.condKey, e.result])).toEqual([['S:after', 'fail']]);
    expect(tl.events[0].ko).toContain('INSUFFICIENT_BUYING_POWER_OR_BUDGET');
    expect(tl.holding).toBeUndefined();
  });

  it('paper trades are markers of their strategy', () => {
    const rows: PaperTrade[] = [
      { code: '237690', group: 'watch', session: 'regular', name: '에스티팜', entryTs: 1000, entry: 86500, trigger: 'VWAP_gap', exitTs: 5000, exit: 87500, reason: 'time', netPct: 0.92 },
      { code: '237690', group: 'watch', session: 'regular', name: '에스티팜', entryTs: 9000, entry: 88000 },
      { code: '000001', group: 'watch', session: 'regular', name: 'x', entryTs: 1, entry: 1 },
    ];
    const t = paperTradesAsTrades(rows, '237690', 'vwap');
    expect(t.map((x) => [x.side, x.ts, x.price, x.strategy])).toEqual([['BUY', 1000, 86500, 'vwap'], ['SELL', 5000, 87500, 'vwap'], ['BUY', 9000, 88000, 'vwap']]);
    expect(t[1]).toMatchObject({ pnlPct: 0.92, buyPrice: 86500 });
    expect(t.every((x) => x.paper === true)).toBe(true); // never drawn as a real fill
    expect(paperTradesAsTrades(rows, '237690', 'vwap', 3000)).toHaveLength(1);
  });

  it('opens on the method that holds the symbol, else on what runs', () => {
    const off = { live: false, paper: false };
    const modes = { rule: { live: true, paper: false }, edge: { live: false, paper: true }, vwap: { live: true, paper: false } };
    const ev = engine().timeline('237690').strategy;
    expect(defaultTimelineMethod(modes, ev)).toBe('vwap'); // VWAP holds it with real orders
    expect(defaultTimelineMethod(modes, [])).toBe('rule'); // nothing held: real orders, rule first
    expect(defaultTimelineMethod({ ...modes, rule: off }, [])).toBe('vwap');
    // a symbol the rule strategy bought today opens on the rule tab even while the rule buys are off now
    expect(defaultTimelineMethod({ ...modes, rule: off }, [], true)).toBe('rule');
    expect(defaultTimelineMethod(modes, ev, true)).toBe('vwap'); // ... unless a side strategy holds it with real orders
    expect(defaultTimelineMethod({ rule: off, edge: { live: false, paper: true }, vwap: off }, [])).toBe('edge');
    expect(defaultTimelineMethod({ rule: off, edge: off, vwap: off }, [])).toBe('rule');
    // a paper holding does not take the tab from a method placing real orders
    const paperOnly = ev.filter((e) => e.mode === 'paper').slice(0, 1);
    expect(defaultTimelineMethod(modes, paperOnly)).toBe('rule');
    expect(defaultTimelineMethod({ rule: off, edge: { live: false, paper: true }, vwap: { live: false, paper: true } }, paperOnly)).toBe('vwap');
  });
});
