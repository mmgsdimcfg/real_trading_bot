/*
 * Update log
 * - [2026-11-12] type=test - paper trading file parsing (ENTRY / EXIT pairing, open trades, broken lines), day summary, the
 *   trade mode choice (server values -> one of three choices and back), the edge mode's log reasons mapped to its
 *   conditions, and the [EDGE ONLY] paper lines not counted as unknown
 */

import { describe, expect, it } from 'vitest';
import { PAPER_KEY, STRATEGY_KEY, paperOpenPct, paperSummary, parseModes, parsePaperLines, parsePaperMode, tradeModeChanges, tradeModeOf } from '../src/shared/paper';
import { classify, IGNORED_TAGS } from '../src/shared/mapping';

const kst = (hms: string) => Date.parse(`2026-11-12T${hms}+09:00`);
const entry = (ts: string, code: string, extra: object = {}) =>
  JSON.stringify({ event: 'ENTRY', ts: `2026-11-12T${ts}`, code, name: `n${code}`, entry: 10310, price: 10300, prob: 0.6, threshold: 0.45, trigger: 'T', tp_px: 10516.2, sl_px: 10134.73, horizon: 60, ...extra });
const exit = (ts: string, code: string, reason: string, net: number, px = 10516.2) =>
  JSON.stringify({ event: 'EXIT', ts: `2026-11-12T${ts}`, code, reason, entry: 10310, exit: px, net_pct: net, held_min: 12.4 });

describe('paper trade file', () => {
  const text = [entry('09:30:05', '000001'), entry('09:31:00', '000002'), exit('09:42:30', '000001', 'tp', 1.77), '{broken', '', entry('10:10:00', '000001'), exit('11:10:01', '000002', 'time', -0.4, 10290)].join('\n');

  it('pairs an EXIT with the open trade of its code; later entries of the same code are new trades', () => {
    const t = parsePaperLines(text);
    expect(t.map((x) => [x.code, x.entryTs, x.reason ?? 'open'])).toEqual([
      ['000001', kst('09:30:05'), 'tp'],
      ['000002', kst('09:31:00'), 'time'],
      ['000001', kst('10:10:00'), 'open'],
    ]);
    expect(t[0]).toMatchObject({ name: 'n000001', entry: 10310, prob: 0.6, threshold: 0.45, tpPx: 10516.2, slPx: 10134.73, horizon: 60, exit: 10516.2, netPct: 1.77, heldMin: 12.4, exitTs: kst('09:42:30') });
  });

  it('summary counts closed trades only', () => {
    const s = paperSummary(parsePaperLines(text));
    expect(s).toMatchObject({ closed: 2, open: 1, wins: 1, winRate: 0.5, byReason: { tp: 1, time: 1 } });
    expect(s.sumPct).toBeCloseTo(1.37);
    expect(s.meanPct).toBeCloseTo(0.685);
    expect(paperSummary([])).toMatchObject({ closed: 0, open: 0, meanPct: 0, winRate: 0 });
  });

  it('ignores an EXIT without an entry, rows without a time or code, and an empty file', () => {
    expect(parsePaperLines(exit('09:42:30', '000009', 'sl', -2))).toEqual([]);
    expect(parsePaperLines(JSON.stringify({ event: 'ENTRY', code: '000001', entry: 1 }))).toEqual([]);
    expect(parsePaperLines('')).toEqual([]);
  });

  it('unrealized % of an open trade', () => {
    const t = parsePaperLines(entry('09:30:05', '000001'))[0];
    expect(paperOpenPct(t, 10413.1)).toBeCloseTo(1.0);
    expect(paperOpenPct(t, undefined)).toBeUndefined();
  });
});

describe('trade mode', () => {
  const get = (strategy: string, paper: string | null, running: [string | null, string | null] = [null, null]) => ({
    values: {
      [STRATEGY_KEY]: { base: 'normal', effective: strategy, running: running[0], pending: false },
      ...(paper === null ? {} : { [PAPER_KEY]: { base: 'off', effective: paper, running: running[1], pending: true } }),
    },
    unknown: paper === null ? [PAPER_KEY] : [],
  });

  it('parses both keys; a server without the paper key reports only the strategy', () => {
    expect(parseModes(get('normal', 'paper', ['normal', 'off']))).toEqual({
      strategy: { effective: 'normal', running: 'normal', pending: false },
      paper: { effective: 'paper', running: 'off', pending: true },
    });
    const old = parseModes(get('normal', null));
    expect(old.paper).toBeNull();
    expect(old.strategy?.effective).toBe('normal');
    expect(parsePaperMode({ values: {} })).toBeNull();
    expect(parseModes(null)).toEqual({ strategy: null, paper: null });
  });

  it('three choices from the two server values', () => {
    expect(tradeModeOf('normal', 'off')).toBe('rules');
    expect(tradeModeOf('normal', 'paper')).toBe('rules+paper');
    expect(tradeModeOf('edge', 'off')).toBe('edge');
    expect(tradeModeOf('edge', 'paper')).toBe('edge'); // the bot skips the paper runner in edge mode
    expect(tradeModeOf('basic', 'paper')).toBe('other');
    expect(tradeModeOf(undefined, undefined)).toBe('other');
  });

  it('a choice sets both keys, and reading them back gives the same choice', () => {
    for (const m of ['rules', 'rules+paper', 'edge'] as const) {
      const c = tradeModeChanges(m)!;
      expect(Object.keys(c).sort()).toEqual([PAPER_KEY, STRATEGY_KEY].sort());
      expect(tradeModeOf(c[STRATEGY_KEY], c[PAPER_KEY])).toBe(m);
    }
    expect(tradeModeChanges('edge')).toEqual({ TRADING_STRATEGY_MODE: 'edge', EDGE_ONLY_MODE: 'off' });
    expect(tradeModeChanges('other')).toBeNull();
  });
});

describe('edge mode log lines', () => {
  it('reject reasons of the Edge AI signal map to the trigger condition', () => {
    for (const body of ['EDGE_SIGNAL_LOW_PROB_p=0.400_LT_0.450 | 1MIN_BB_MID_GOLDEN_CROSS_LOOKBACK', 'EDGE_SIGNAL_NO_TRIGGER | NO_CROSS', 'EDGE_SIGNAL_REGULAR_SESSION_ONLY']) {
      const r = classify('REJECT', body);
      expect(r.kind).toBe('event');
      if (r.kind === 'event') expect([r.ev.side, r.ev.condKey, r.ev.result]).toEqual(['buy', '_009', 'fail']);
    }
    // the normal-mode filter keeps its own condition
    const normal = classify('REJECT', 'EDGE_AI_LOW_PROB_p=0.300_LT_0.320');
    expect(normal.kind === 'event' && normal.ev.condKey).toBe('_020');
  });

  it('sell triggers map to the edge sell conditions', () => {
    const cases: [string, string][] = [
      ['EDGE_TP_2.0PCT | qty=7/7 price=10,200 entry=10,000 pnl=2.00% peak=2.00%', '_010'],
      ['EDGE_SL_1.7PCT | qty=7/7 price=9,830 entry=10,000 pnl=-1.70% peak=0.10%', '_011'],
      ['EDGE_TIME_60MIN | qty=7/7 price=10,050 entry=10,000 pnl=0.50% peak=0.90% held=60min', '_012'],
    ];
    for (const [body, key] of cases) {
      const r = classify('SELL TRIGGER', body);
      expect(r.kind === 'event' && [r.ev.side, r.ev.condKey, r.ev.result]).toEqual(['sell', key, 'pass']);
    }
  });

  it('paper trading lines are a known tag', () => {
    expect(IGNORED_TAGS.has('EDGE ONLY')).toBe(true);
    expect(classify('EDGE ONLY', 'PAPER BUY 10,310 p=0.600').kind).toBe('ignore');
  });
});

describe('automatic sell mode (AUTO_SELL_MODE)', () => {
  it('reads the server value and falls back to all', async () => {
    const { AUTO_SELL_KEY, autoSellOf, parseAutoSell } = await import('../src/shared/paper');
    const data = { values: { [AUTO_SELL_KEY]: { effective: 'STOP_ONLY', running: 'all', pending: true } } };
    expect(parseAutoSell(data)).toEqual({ effective: 'stop_only', running: 'all', pending: true });
    expect(parseAutoSell({ values: {} })).toBeNull(); // older server code
    expect(autoSellOf('stop_only')).toBe('stop_only');
    expect(autoSellOf('off')).toBe('off');
    expect(autoSellOf('all')).toBe('all');
    expect(autoSellOf('typo')).toBe('all'); // the bot treats an unknown value as all
    expect(autoSellOf(undefined)).toBe('all');
  });

  it('maps the new log reasons to their conditions', () => {
    const of = (tag: string, rest: string) => {
      const out = classify(tag, rest);
      return out.kind === 'event' ? [out.ev.condKey, out.ev.result] : [out.kind];
    };
    expect(of('REJECT', 'GC_VOL_LOW_1MIN_ATR_0.310%_LT_0.500% | GATES a')).toEqual(['_012', 'fail']);
    expect(of('REJECT', 'GC_VOL_1MIN_ATR_UNAVAILABLE | GATES a')).toEqual(['_012', 'fail']);
    expect(of('SELL TRIGGER', 'NET_STOP_LOSS_2.0PCT | price=9,822 entry=10,000 pnl=-1.78% net=-2.01%')).toEqual(['_004', 'pass']);
    expect(IGNORED_TAGS.has('AUTO SELL')).toBe(true);
  });
});
