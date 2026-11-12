/*
 * Update log
 * - [2026-11-04] type=test - Parser unit tests (guide 7: (a) line grammar, (b) every row of the 3.1/3.2 tables,
 *   (c) buy_sell block parsing, (d) incremental reads in two pieces, (e) incomplete last line held back)
 * - [2026-11-05] type=test - Symbol list: emoji tag [HOLD SKIP], _buy_sell file-name suffix, WATCH name priority, [WARMUP]-only ETFs
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { LineBuffer } from '../src/shared/lineBuffer';
import { parseMainLine } from '../src/shared/logLine';
import { classify, keyRemap, remapKey } from '../src/shared/mapping';
import { DayEngine } from '../src/shared/engine';
import { classifyLogFile } from '../src/shared/files';
import type { ConditionsDoc } from '../src/shared/types';
import { fmtKstDateTime } from '../src/shared/time';

const conditions = JSON.parse(readFileSync(join(__dirname, '../resources/conditions_snapshot.json'), 'utf8')) as ConditionsDoc;

const L = (time: string, tag: string, body: string) => `2026-11-04 ${time},593 [INFO]   [${tag.padEnd(12)}] ${body}`;

describe('(a) main log line grammar', () => {
  it('parses a tagged symbol line', () => {
    const p = parseMainLine(
      '2026-11-04 09:08:12,593 [INFO]   [BUY EVAL    ] 159010_아스플로   | OK TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20+TICK120_GOLDEN_CROSS+HYBRID_3MIN_CTX_SCORE_12 | 09:07:56 | LIVE 29,050 | x',
    );
    expect(p.kind).toBe('tagged');
    if (p.kind !== 'tagged') return;
    expect(p.tag).toBe('BUY EVAL');
    expect(p.code).toBe('159010');
    expect(p.name).toBe('아스플로');
    expect(p.rest.startsWith('OK TRIPLE_CROSS')).toBe(true);
    expect(fmtKstDateTime(p.ts)).toBe('2026-11-04 09:08:12');
  });
  it('parses symbol with alias and no-name forms', () => {
    const a = parseMainLine(L('10:50:37', 'REENTRY LIVE', '017900_광전자    (광전자) | BUY qty=55'));
    expect(a.kind === 'tagged' && a.code).toBe('017900');
    const b = parseMainLine(L('09:00:57', 'TICK CROSS', '028260 | tick120 MA5>MA20 at 09:00:57'));
    expect(b.kind === 'tagged' && b.code).toBe('028260');
    const c = parseMainLine(L('09:00:00', 'BUY SKIP', '035720_카카오게~ | ALREADY_TRADED_TODAY_UNTIL_SELL'));
    expect(c.kind === 'tagged' && c.name).toBe('카카오게');
  });
  it('classifies untagged and continuation lines without throwing', () => {
    expect(parseMainLine('2026-11-04 08:30:00,001 [INFO] R003 START pid=1').kind).toBe('untagged');
    expect(parseMainLine('2026-11-04 08:30:00,001 [INFO] WATCH | 159010 | 아스플로 | NXT=False').kind).toBe('untagged');
    expect(parseMainLine('Traceback (most recent call last):').kind).toBe('continuation');
    expect(parseMainLine('').kind).toBe('blank');
  });
});

describe('(b) log -> condition mapping', () => {
  const buyCases: [string, string, string, string?][] = [
    ['SKIP', 'can_trade_code_now=False', '_001'],
    ['SKIP', 'frame=None', 'PREP'],
    ['SKIP', 'bars=12 < INDICATOR_WARMUP_BARS(30)', 'PREP'],
    ['REJECT', 'SESSION_OPEN_WARMUP | GATES x', '_002'],
    ['BUY SKIP', 'ALREADY_TRADED_TODAY_UNTIL_SELL', '_005'],
    ['BUY SKIP', 'HARD_STOP_REENTRY_BLOCKED_TODAY', '_006'],
    ['BUY SKIP', 'DAILY_LOSS_LIMIT -1.20% <= -1.00%', '_007'],
    ['BUY SKIP', 'MAX_OPEN_POSITIONS 5/5', '_008'],
    ['BUY SKIP', 'CIRCUIT_BREAKER_ACTIVE until=12:24:03 count=3', 'PREP', 'status'],
    ['TRADE', 'BUY REPRICE RESUBMIT | qty=10 | leg=limit', 'ORDER', 'BUY REPRICE RESUBMIT'],
    ['BUY CXL REJ', 'leg=limit | 주문 취소 불가', 'ORDER', 'BUY CXL REJ'],
    ['REJECT', 'TRIPLE_CROSS_MISSING_TICK_ABSTAIN_NO_FILE | GATES', '_009', 'trigger'],
    ['REJECT', 'TICK_LOW_DENSITY_12_LT_30', '_009', 'trigger'],
    ['REJECT', 'TRIPLE_CROSS_1MIN_CHASE_BUY_BB_GAP_2.1%', '_009', 'candle'],
    ['REJECT', 'TRIPLE_CROSS_1MIN_CHASE_BUY_INTRABAR', '_009', 'candle'],
    ['REJECT', 'TRIPLE_CROSS_1MIN_CANDLE_NOT_BULLISH_0.2', '_009', 'candle'],
    ['REJECT', 'TRIPLE_CROSS_1MIN_LIVE_DROP_0.5%', '_009', 'live'],
    ['REJECT', 'TRIPLE_CROSS_1MIN_LIVE_BELOW_BB_MID', '_009', 'live'],
    ['REJECT', 'TRIPLE_CROSS_CHASE_FROM_CROSS_2.10%_GT_1.50%', '_009', 'chase'],
    ['REJECT', 'HYBRID_1MIN_TRIGGER_NONE', '_009', 'legacy'],
    ['REJECT', 'HYBRID_1MIN_FRAME_UNAVAILABLE', '_009', 'legacy'],
    ['TICK TRIG', 'no tick file before 09:00', '_009', 'trigger'],
    ['REJECT', 'MARKET_REGIME_DOWN_KOSPI', '_010'],
    ['REJECT', 'HYBRID_3MIN_CTX_INSUFFICIENT_BARS', '_011'],
    ['REJECT', 'HYBRID_3MIN_CTX_MISSING_INDICATOR', '_011'],
    ['REJECT', 'HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.41%_LT_0.60% | GATES a | STEPS 1min_trigger(P) min_entry_atr_volatility(F) min_liquidity_safety(-)', '_012'],
    ['REJECT', 'HYBRID_3MIN_CTX_LOW_VOL_MA_ABS_1000', '_013'],
    ['REJECT', 'HYBRID_3MIN_CTX_LOW_ABS_VOLUME_1000', '_013'],
    ['REJECT', 'HYBRID_3MIN_CTX_LOW_VOLUME_RATIO_0.8', '_013'],
    ['REJECT', 'HYBRID_3MIN_CTX_LOW_TURNOVER_1e8', '_013'],
    ['REJECT', 'EXCESSIVE_RISE_FROM_PREV_CLOSE_25%', '_015'],
    ['REJECT', 'ANTI_CHASE_VWAP_GAP_3.95%_GE_3.00%', '_016'],
    ['REJECT', 'OPENING_GAP_5%', '_017'],
    ['REJECT', 'QUOTE_STATUS_BLOCK_51', '_018'],
    ['REJECT', 'ACCOUNT_SYNC_STALE age=120s', '_019'],
    ['EDGE AI', 'p=0.372 thr=0.310 pass mode=shadow', '_020'],
    ['REJECT', 'EDGE_AI_LOW_PROB_p=0.20_LT_0.31', '_020'],
    ['REJECT', 'SPREAD_NO_QUOTE', '_021'],
    ['REJECT', 'SPREAD_TOO_WIDE 0.8%', '_021'],
    ['SPREAD', 'quote query failed timeout pass', '_021'],
    ['BUY HOLD', 'reason=WAIT_NEXT_POLL_CONFIRM | count=1/2', '_022'],
    ['REJECT', 'INSUFFICIENT_BUYING_POWER_OR_BUDGET', '_023'],
    ['REJECT', 'STALE_LIVE_PRICE', '_024'],
    ['REJECT', 'ORDERBOOK_ASK_THIN', '_025'],
    ['BUY EVAL', 'OK TRIPLE_CROSS_X+HYBRID_3MIN_CTX_SCORE_12 | 09:07:56 | LIVE 29,050', 'ORDER'],
    ['BUY SUBMIT', 'qty=8 | price=29,050', 'ORDER'],
    ['BUY EXEC', 'qty=8', 'ORDER'],
    ['PENDING', 'side=BUY qty=8 submitted=09:08:12', 'ORDER'],
  ];
  for (const [tag, rest, cond, sub] of buyCases) {
    it(`${tag} ${rest.slice(0, 40)} -> ${cond}${sub ? '/' + sub : ''}`, () => {
      const out = classify(tag, rest);
      expect(out.kind).toBe('event');
      if (out.kind !== 'event') return;
      expect(out.ev.side).toBe('buy');
      expect(out.ev.condKey).toBe(cond);
      if (sub) expect(out.ev.sub).toBe(sub);
    });
  }
  it('parses STEPS diagnostics', () => {
    const out = classify('REJECT', buyCases.find((c) => c[2] === '_012')![1]);
    expect(out.kind === 'event' && out.ev.steps).toEqual({ '1min_trigger': 'P', min_entry_atr_volatility: 'F', min_liquidity_safety: '-' });
  });

  const sellCases: [string, string, string, string][] = [
    ['SELL HOLD', 'AUX_BLOCKED_SCORE_3_PNL_0.20%_LT_0.50%', '_015', 'hold'],
    ['SELL HOLD', 'AUX_TRIGGER_BUFFER_BLOCK 1/2', '_015', 'hold'],
    ['SELL TRIGGER', 'HARD_STOP_LOSS_1.7PCT | pnl=-2.10%', '_004', 'pass'],
    ['SELL TRIGGER', 'TP1_PARTIAL_40PCT_1.20PCT_SURGE', '_005', 'pass'],
    ['SELL TRIGGER', 'TP2_SURGE_LADDER_3PCT', '_006', 'pass'],
    ['SELL TRIGGER', 'TP3_SURGE_LADDER_5PCT', '_006', 'pass'],
    ['SELL TRIGGER', 'PROFIT_LOCK_100PCT_GIVEBACK_0.4PCT', '_007', 'pass'],
    ['SELL TRIGGER', 'PROFIT_LOCK_REST_BREAKEVEN', '_007', 'pass'],
    ['SELL TRIGGER', 'ATR_TAKE_PROFIT_3.0x', '_008', 'pass'],
    ['SELL TRIGGER', 'POST_BUY_ENTRY_DROP_1.0%', '_009', 'pass'],
    ['SELL TRIGGER', 'BREAKEVEN_FAIL_30MIN', '_010', 'pass'],
    ['SELL TRIGGER', 'NO_TREND_EXIT_40MIN', '_011', 'pass'],
    ['SELL TRIGGER', 'ATR_STOP_LOSS_1.5x', '_012', 'pass'],
    ['SELL TRIGGER', 'TRAILING_STOP_GIVEBACK_0.8%', '_013', 'pass'],
    ['SELL TRIGGER', 'TP_EXTENSION_TRAIL_1.0%', '_013', 'pass'],
    ['SELL TRIGGER', 'AUX_REVERSAL_SCORE_4', '_015', 'pass'],
    ['SELL TRIGGER', 'LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_2', '_015', 'pass'],
    ['SELL TRIGGER', 'PEAK_NEXT_BAR_BEARISH_0.5%', '_003', 'pass'],
    ['SELL TRIGGER', 'qty=17 | reason=REGULAR_CLOSE_SESSION_FORCE_CLOSE_ALL_1520', 'SESSION_CLOSE', 'pass'],
    ['SELL FILLED', 'filled=17/17 | price=28,511 | remaining=0 | pnl=-2.10% | reason=HARD_STOP_LOSS_1.7PCT', 'ORDER', 'pass'],
    // 2026-11-04 unknown-list leftovers (real lines)
    ['SELL EVAL', 'OK AUX_REVERSAL_SCORE_3 | 11:14:10 | LIVE 22,550 | BB 21945.0->21940.0 | RSI=48.5 SIG=46.7', '_015', 'pass'],
    ['SELL EVAL', 'OK LIVE_PRICE_BB_DOWN_CROSS_CONFIRMED_2 | 10:14:00 | LIVE 30,100', '_015', 'pass'],
  ];
  for (const [tag, rest, cond, result] of sellCases) {
    it(`${tag} ${rest.slice(0, 40)} -> ${cond}`, () => {
      const out = classify(tag, rest);
      expect(out.kind).toBe('event');
      if (out.kind !== 'event') return;
      expect(out.ev.side).toBe('sell');
      expect(out.ev.condKey).toBe(cond);
      expect(out.ev.result).toBe(result);
    });
  }

  it('keeps unknown reasons as unknown (never guesses)', () => {
    expect(classify('REJECT', 'SOME_NEW_GATE_42').kind).toBe('unknown');
    expect(classify('NEW TAG', 'whatever').kind).toBe('unknown');
    for (const [tag, rest] of [
      ['CHECK', 'bars=9 live=116,800  bar_close=116,900 | confirmed_bar=08:27:00 cutoff=08:27:00 bar_age=9'],
      ['BAR_SKIP', 'INTRABAR_RSI_79.6_OUT_OF_45-75 | using confirmed 3min bar instead'],
      ['BAR_SKIP', 'INTRABAR_RSI_66.5_EXT_COND_NOT_MET | using confirmed 3min bar instead'],
    ]) {
      const o = classify(tag, rest);
      expect(o.kind).toBe('event');
      if (o.kind === 'event') expect(o.ev).toMatchObject({ condKey: 'PREP', result: 'info', sub: 'status' });
    }
    expect(classify('WARMUP', 'ON | prior_day=20261104 1min=180 3min=59').kind).toBe('ignore');
  });
});

const BUY_SELL_SAMPLE = [
  '2026-11-04 09:08:12 [INFO] [BUY SUBMIT  ] 159010_아스플로   | qty=8 | price=29,050 | market | KRX | order_no=0003529700 | reason=TRIPLE_CROSS_X',
  '2026-11-04 09:09:26 [INFO] ==============================================================================================================',
  '2026-11-04 09:09:26 [INFO] >>> 159010_아스플로 Buy - reason=TRIPLE_CROSS_1MIN_MA5_ABOVE_MA20+TICK120_GOLDEN_CROSS+HYBRID_3MIN_CTX_SCORE_12 signal=None live=29,050 bb_mid=28485.0',
  '2026-11-04 09:09:26 [INFO] *** 매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17',
  '2026-11-04 09:15:56 [INFO] >>> 159010_아스플로 Sell - 수익금액=-10,413원 | 수익율=-2.10% | reason=HARD_STOP_LOSS_1.7PCT exch=KRX',
  '2026-11-04 09:15:56 [INFO] *** 매수시각=2026-11-04 09:07:56 | 매수가격=29,124 | 매수수량=17',
  '2026-11-04 09:15:56 [INFO] *** 매도시각=2026-11-04 09:15:56 | 매도가격=28,511 | 매도수량=17',
  '2026-11-04 10:50:37 [INFO] [REENTRY LIVE] 017900_광전자    (광전자) | BUY qty=55 price=8,970 used=1/1 | reason=X',
];

describe('(c) buy_sell trade blocks', () => {
  it('groups >>> lines with their *** lines', () => {
    const e = new DayEngine('20261104', conditions);
    e.feedBuySellLines(BUY_SELL_SAMPLE);
    const t = e.getTrades();
    expect(t).toHaveLength(2);
    expect(t[0]).toMatchObject({ code: '159010', side: 'BUY', price: 29124, qty: 17 });
    expect(fmtKstDateTime(t[0].ts)).toBe('2026-11-04 09:07:56');
    expect(t[1]).toMatchObject({ side: 'SELL', price: 28511, qty: 17, pnlKrw: -10413, pnlPct: -2.1, condKey: '_004', buyPrice: 29124 });
    const snap = e.snapshot();
    const s = snap.symbols.find((x) => x.code === '159010')!;
    expect(s.position).toBeUndefined();
    expect(s.realizedKrw).toBe(-10413);
  });
});

const MAIN_SAMPLE = [
  '2026-11-04 08:30:00,001 [INFO] WATCH | 159010 | 아스플로 | NXT=False',
  L('08:59:00', 'REJECT', '159010_아스플로   | TRIPLE_CROSS_MISSING_TICK_ABSTAIN_NO_FILE | GATES x'),
  L('09:01:00', 'REJECT', '159010_아스플로   | SESSION_OPEN_WARMUP | GATES x'),
  L('09:05:00', 'REJECT', '159010_아스플로   | HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.4%_LT_0.6% | GATES x | STEPS 1min_trigger(P) min_entry_atr_volatility(F) min_liquidity_safety(-)'),
  L('09:06:00', 'EDGE AI', '159010_아스플로   | p=0.372 thr=0.310 pass mode=live'),
  L('09:06:00', 'BUY HOLD', '159010_아스플로   | reason=WAIT_NEXT_POLL_CONFIRM | count=1/2'),
  L('09:08:12', 'BUY EVAL', '159010_아스플로   | OK TRIPLE_CROSS_X | 09:07:56 | LIVE 29,050'),
  L('09:08:12', 'BUY SUBMIT', '159010_아스플로   | qty=8 | price=29,050'),
  L('09:10:00', 'SELL HOLD', '159010_아스플로   | AUX_BLOCKED_SCORE_3_PNL_0.20%_LT_0.50%'),
  L('09:15:56', 'SELL TRIGGER', '159010_아스플로   | HARD_STOP_LOSS_1.7PCT'),
  L('09:16:00', 'REJECT', '159010_아스플로   | BRAND_NEW_GATE_1'),
];

function runEngine(chunks: Uint8Array[]): string {
  const e = new DayEngine('20261104', conditions);
  const lb = new LineBuffer();
  for (const c of chunks) e.feedMainLines(lb.push(c));
  const snap = e.snapshot();
  return JSON.stringify({ ...snap, generatedAt: 0, tl: e.timeline('159010').events.map((x) => [x.condKey, x.result]) });
}

describe('engine aggregation', () => {
  it('counts blocks, passes, pre-open blocks and unknowns', () => {
    const e = new DayEngine('20261104', conditions);
    e.feedMainLines(MAIN_SAMPLE);
    const snap = e.snapshot();
    expect(snap.buyStats._009.preOpenBlock).toBe(1);
    expect(snap.buyStats._009.block).toBe(0);
    expect(snap.buyStats._002.block).toBe(1);
    expect(snap.buyStats._012.block).toBe(1);
    // in-session polls that got past _001: _002 reject, _012 reject, _022 hold, EVAL OK
    expect(snap.buyStats._001.pass).toBe(4);
    expect(snap.buyStats._022.hold).toBe(1);
    const s = snap.symbols[0];
    expect(s.name).toBe('아스플로');
    expect(s.furthestOrderToday).toBe(conditions.buy.length);
    expect(s.lastEval?.condKey).toBe('ORDER');
    expect(s.lastSell?.condKey).toBe('_004');
    expect(snap.sellStats._015.hold).toBe(1);
    expect(snap.unknown).toHaveLength(1);
  });
});

describe('status events ([CHECK] / [BAR_SKIP])', () => {
  it('keeps at most one per minute, none right after another event, and leaves the stats alone', () => {
    const e = new DayEngine('20261104', conditions);
    const C = (t: string) => L(t, 'CHECK', '159010_아스플로   | bars=9 live=29,000 bar_close=29,000 | confirmed_bar=08:27:00');
    const B = (t: string) => L(t, 'BAR_SKIP', '159010_아스플로   | INTRABAR_MFI_36.4_OUT_OF_50-75 | using confirmed 3min bar instead');
    e.feedMainLines([
      C('08:10:00'), // kept (first event)
      B('08:10:20'), // dropped (< 60s)
      C('08:10:59'), // dropped
      C('08:11:00'), // kept
      L('08:11:30', 'REJECT', '159010_아스플로   | SESSION_OPEN_WARMUP | GATES x'),
      C('08:12:00'), // dropped (30s after the REJECT)
      B('08:12:40'), // kept
    ]);
    const tl = e.timeline('159010').events.map((x) => [x.tag, x.condKey, x.result]);
    expect(tl).toEqual([
      ['CHECK', 'PREP', 'info'],
      ['CHECK', 'PREP', 'info'],
      ['REJECT', '_002', 'fail'],
      ['BAR_SKIP', 'PREP', 'info'],
    ]);
    const snap = e.snapshot();
    expect(snap.unknown).toHaveLength(0);
    expect(snap.symbols[0].evalCount).toBe(1);
  });
});

describe('(d)(e) incremental reads', () => {
  const bytes = new TextEncoder().encode(MAIN_SAMPLE.join('\n') + '\n');
  it('same result when fed in two pieces (split inside a multi-byte character)', () => {
    const whole = runEngine([bytes]);
    // find a split point inside a Korean character (3-byte UTF-8)
    let cut = bytes.indexOf(0xec, 200);
    if (cut < 0) cut = 300;
    expect(runEngine([bytes.slice(0, cut + 1), bytes.slice(cut + 1)])).toBe(whole);
    for (const n of [1, 7, 100, 333]) {
      const parts: Uint8Array[] = [];
      for (let i = 0; i < bytes.length; i += n) parts.push(bytes.slice(i, i + n));
      expect(runEngine(parts)).toBe(whole);
    }
  });
  it('holds back an incomplete last line until the newline arrives', () => {
    const lb = new LineBuffer();
    const enc = new TextEncoder();
    expect(lb.push(enc.encode('line one\nline tw'))).toEqual(['line one']);
    expect(lb.pending).toBe(7);
    expect(lb.push(enc.encode('o\r\n'))).toEqual(['line two']);
    expect(lb.pending).toBe(0);
  });
});

describe('symbol list (2026-11-05 log changes)', () => {
  it('parses an emoji-prefixed tag ([🔴HOLD SKIP ])', () => {
    const p = parseMainLine('2026-11-05 10:20:30,767 [INFO]   [🔴HOLD SKIP ] 033780_KT&G       | NOT_TODAY_BUY_POSITION');
    expect(p.kind).toBe('tagged');
    if (p.kind !== 'tagged') return;
    expect(p.tag).toBe('HOLD SKIP');
    expect(p.code).toBe('033780');
    expect(p.name).toBe('KT&G');
    const c = classify(p.tag, p.rest);
    expect(c).toMatchObject({ kind: 'event', ev: { side: 'sell', condKey: 'PREP', result: 'info', reason: 'NOT_TODAY_BUY_POSITION' } });
  });

  it('drops the _buy_sell suffix of per-symbol file names', () => {
    expect(classifyLogFile('20261105_034730_SK_buy_sell.txt', '20261105')).toEqual({ kind: 'symbol', code: '034730', name: 'SK' });
    expect(classifyLogFile('20261105_102780_KODEX 삼성그룹_buy_sell.txt', '20261105')).toMatchObject({ name: 'KODEX 삼성그룹' });
    expect(classifyLogFile('20261104_159010_아스플로.txt', '20261104')).toMatchObject({ name: '아스플로' });
  });

  it('WATCH name wins; [WARMUP] alone does not create a symbol', () => {
    const e = new DayEngine('20261105', conditions);
    e.feedMainLines([
      '2026-11-05 08:32:45,008 [INFO] WATCH | 034730 | SK | NXT=True',
      '2026-11-05 09:01:47,404 [INFO]   [WARMUP      ] 069500 | OFF | prior_day=20261104 close=89,410',
      '2026-11-05 09:01:48,404 [INFO]   [WARMUP      ] 034730 | ON | prior_day=20261104 1min=180 3min=59',
    ]);
    e.setName('034730', 'SK_buy_sell');
    const snap = e.snapshot();
    expect(snap.symbols.map((s) => s.code)).toEqual(['034730']);
    expect(snap.symbols[0].name).toBe('SK');
  });

  it('NXT flag comes from the WATCH line; symbols without one have none', () => {
    const e = new DayEngine('20261112', conditions);
    e.feedMainLines([
      '2026-11-12 07:58:01,000 [INFO] WATCH | 010060 | OCI홀딩스 | NXT=True',
      '2026-11-12 07:58:01,001 [INFO] WATCH | 003380 | 하림지주 | NXT=False',
      '2026-11-12 07:58:01,002 [INFO] WATCH | 159010 | 아스플로 | x',
    ]);
    const by = Object.fromEntries(e.snapshot().symbols.map((s) => [s.code, s.nxt]));
    expect(by).toEqual({ '010060': true, '003380': false, '159010': undefined });
  });
});

describe('server key remap (app / server numbering gap)', () => {
  // a server still on the numbering before 2026-11-08: sell profit lock _023, reversal _022, buy trigger _008
  const cond = (order: number, no: number, func: string) => ({ order, no, func, stage: 'EXIT', title: func, description: '' });
  const oldDoc = {
    generated_at: 'x',
    buy: [cond(0, 1, '_001_entry_time_window'), cond(1, 8, '_008_hybrid_1min_trigger'), cond(2, 14, '_014_min_entry_atr_volatility')],
    sell: [cond(0, 4, '_004_hard_stop_loss'), cond(1, 23, '_023_profit_lock'), cond(2, 22, '_022_shared_reversal_sell')],
  } as unknown as ConditionsDoc;

  it('maps app keys to the server numbers through the function names', () => {
    const sell = keyRemap(oldDoc, 'sell');
    expect(remapKey(sell, '_007')).toBe('_023');
    expect(remapKey(sell, '_015')).toBe('_022');
    expect(remapKey(sell, '_004')).toBe('_004');
    expect(remapKey(sell, 'ORDER')).toBe('ORDER');
    const buy = keyRemap(oldDoc, 'buy');
    expect(remapKey(buy, '_009/trigger')).toBe('_008/trigger');
    expect(remapKey(buy, '_012')).toBe('_014');
  });

  it('stub or missing functions keep the app key', () => {
    const stub = { generated_at: 'x', buy: [cond(0, 2, '_002')], sell: [] } as unknown as ConditionsDoc;
    expect(keyRemap(stub, 'buy').size).toBe(0);
    expect(keyRemap(null, 'sell').size).toBe(0);
  });

  it('engine events and sell trades use the server keys', () => {
    const e = new DayEngine('20261104', oldDoc);
    e.feedMainLines([
      L('09:05:00', 'REJECT', '159010_아스플로   | HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.4%_LT_0.6% | GATES x'),
      L('09:10:00', 'SELL HOLD', '159010_아스플로   | AUX_BLOCKED_SCORE_3_PNL_0.20%_LT_0.50%'),
    ]);
    const snap = e.snapshot();
    expect(snap.buyStats._014.block).toBe(1);
    expect(snap.sellStats._022.hold).toBe(1);
  });
});
