/*
 * Update log
 * - [2026-11-12] type=test - paper trading file parsing (ENTRY / EXIT pairing, open trades, broken lines), day summary, the
 *   trade mode choice (server values -> one of three choices and back), the edge mode's log reasons mapped to its
 *   conditions, and the [EDGE ONLY] paper lines not counted as unknown
 * - [2026-11-19] type=test - paper status file (parsePaperStatus / sortStatusRows) on the exact text r022 writes
 * - [2026-11-19] type=test - group of a trade: "collector" (tick collector symbols), anything else / missing = "watch"
 */

import { describe, expect, it } from 'vitest';
import { PAPER_KEY, PAPER_STAGE_KO, STRATEGY_KEY, paperOpenPct, paperSummary, parseModes, parsePaperLines, parsePaperMode, parsePaperStatus, sortStatusRows, tradeModeChanges, tradeModeOf } from '../src/shared/paper';
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

  it('group: collector records are tagged, the others (and old records) are watch', () => {
    const t = parsePaperLines([entry('09:30:05', '000001'), entry('09:31:00', '000003', { group: 'collector' }), entry('09:32:00', '000004', { group: 'x' })].join('\n'));
    expect(t.map((x) => x.group)).toEqual(['watch', 'collector', 'watch']);
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
    expect(of('SELL TRIGGER', 'ADAPTIVE_TP1_NORMAL_2.0PCT | qty=40/100 price=10,200')).toEqual(['_010', 'pass']);
    expect(of('SELL TRIGGER', 'ADAPTIVE_STOP_INITIAL_-1.50PCT | qty=100/100')).toEqual(['_010', 'pass']);
    expect(IGNORED_TAGS.has('AUTO SELL')).toBe(true);
    expect(IGNORED_TAGS.has('EXIT ENGINE')).toBe(true);
    expect(IGNORED_TAGS.has('ENTRY ENGINE')).toBe(true);
    expect(IGNORED_TAGS.has('ENTRY LAT')).toBe(true);
    expect(classify('REENTRY PNL', '125490_한라캐스트 | last_exit_profit=False limit=10 used=1').kind).toBe('ignore');
  });
});

describe('Edge AI model meta (current_meta.json)', () => {
  it('summarises the new-label model', async () => {
    const { parseEdgeMeta, edgeLabelKo } = await import('../src/shared/paper');
    const meta = {
      model_id: '20261112_165714', created: '2026-11-12T16:57:14', threshold: 0.3388, promoted: false, dataset: 'v2_tp2_sl1.77_h400_0e91bb',
      n_days: 89, train_range: '20260814~20261111', label: { tp: 2, sl: 1.77, horizon: 400, slip_ticks: 1, target: 'y_tp' },
      promotion: { auc_oos: 0.718, base: { n: 244054, mean_net: -0.5636 }, kept: { n: 47583, mean_net: -0.586 },
        run_base: { n: 244054, tp2: 0.196, tp3: 0.112, tp5: 0.042, stopped: 0.326 }, run_kept: { n: 47583, tp2: 0.355, tp3: 0.241, tp5: 0.111, stopped: 0.592 } },
    };
    const m = parseEdgeMeta(JSON.stringify(meta))!;
    expect([m.modelId, m.threshold, m.promoted, m.target, m.nDays, m.aucOos]).toEqual(['20261112_165714', 0.3388, false, 'y_tp', 89, 0.718]);
    expect(m.kept).toEqual({ n: 47583, meanNet: -0.586 });
    expect(m.runKept).toEqual({ tp2: 0.355, tp3: 0.241, tp5: 0.111, stopped: 0.592 });
    expect(edgeLabelKo(m)).toBe('손절(-1.77%) 전에 +2% 도달');
  });

  it('reads an older meta without target / run-up stats and rejects other text', async () => {
    const { parseEdgeMeta, edgeLabelKo } = await import('../src/shared/paper');
    const m = parseEdgeMeta(JSON.stringify({ model_id: '20261111_222939', threshold: 0.52, promoted: true, label: { tp: 2, sl: 1.7, horizon: 60 }, promotion: {} }))!;
    expect([m.target, m.promoted, m.runKept, m.kept, m.aucOos]).toEqual(['y', true, null, null, null]);
    expect(edgeLabelKo(m)).toBe('+2% / -1.7% / 60분 순손익 > 0');
    expect([m.noTrade, m.role]).toEqual([false, 'gate']);
    // [2026-11-27] the trade model: no cut with a positive expectancy = threshold 1 + no_trade
    const { edgeCutText } = await import('../src/shared/paper');
    const n = parseEdgeMeta(JSON.stringify({ model_id: 'x', threshold: 1.0, no_trade: true, role: 'trade', promoted: false, label: { tp: 3, sl: 1.27, horizon: 300, target: 'y' } }))!;
    expect([n.noTrade, n.role, n.threshold]).toEqual([true, 'trade', 1]);
    expect(edgeCutText(n.threshold, n.noTrade)).toBe('없음 (매수 안 함)');
    expect([edgeCutText(1), edgeCutText(0.3312), edgeCutText(undefined), edgeCutText(null)]).toEqual(['없음 (매수 안 함)', '0.331', '-', '-']);
    expect(parseEdgeMeta('{broken')).toBeNull();
    expect(parseEdgeMeta('{"threshold":0.5}')).toBeNull();
  });
});

describe('paper status file (server r022 edge_only_status_<date>.json)', () => {
  // the exact text r022 write_status produced in a test run
  const text =
    '{"asof": "2026-11-12T09:30:06", "date": "20261112", "model_id": "test", "threshold": 0.45, "label": {"tp": 2.0, "sl": 1.7, "horizon": 60.0}, "entry_window": ["09:05", "14:50"], "force_exit": "15:15", "groups": {"watch": {"open": 1, "max_positions": 3, "entries": 1, "max_entries": 10, "enabled": true}, "collector": {"open": 0, "max_positions": 5, "entries": 0, "max_entries": 20, "enabled": true}}, "symbols": [{"code": "000001", "name": "가", "group": "watch", "price": 10300.0, "bar": "09:30", "bars": 30, "stage": "bought", "prob": 0.6, "at": "2026-11-12T09:30:05", "position": {"entry": 10310.0, "tp": 10516.2, "sl": 10134.73, "ts": "2026-11-12T09:30:05", "prob": 0.6}}, {"code": "000002", "name": "나", "group": "collector", "price": 9000.0, "bar": "09:30", "bars": 30, "stage": "below_mid", "prob": null, "at": "2026-11-12T09:30:05"}]}';

  it('parses groups, symbols, stages and the open position', () => {
    const s = parsePaperStatus(text)!;
    expect(s.asof).toBe(kst('09:30:06'));
    expect(s.threshold).toBe(0.45);
    expect(s.entryWindow).toEqual(['09:05', '14:50']);
    expect(s.groups.watch).toEqual({ open: 1, maxPositions: 3, entries: 1, maxEntries: 10, enabled: true });
    expect(s.groups.collector.maxEntries).toBe(20);
    expect(s.symbols.map((x) => [x.code, x.group, x.stage, x.prob])).toEqual([
      ['000001', 'watch', 'bought', 0.6],
      ['000002', 'collector', 'below_mid', undefined],
    ]);
    expect(s.symbols[0].position).toMatchObject({ entry: 10310, tp: 10516.2, ts: kst('09:30:05') });
    expect(PAPER_STAGE_KO.below_mid).toContain('BB');
    expect(s.logOnly).toBe(false);
    // [2026-11-20] log-only server (EDGE_ONLY_LOG_ONLY)
    expect(parsePaperStatus(text.replace('"date": "20261112",', '"date": "20261112", "log_only": true,'))!.logOnly).toBe(true);
    expect(PAPER_STAGE_KO.candidate).toContain('후보');
  });

  it('held first, then by probability; broken text is null', () => {
    const s = parsePaperStatus(text)!;
    const rows = sortStatusRows([{ ...s.symbols[1], prob: 0.9 }, s.symbols[0], { ...s.symbols[1], code: '000003', prob: 0.2 }]);
    expect(rows.map((x) => x.code)).toEqual(['000001', '000002', '000003']);
    expect(parsePaperStatus('{bad')).toBeNull();
    expect(parsePaperStatus('{"symbols": 1}')).toBeNull();
  });
});

// [2026-11-27] rule trade variant, VWAP trade keys and the paper books
describe('rule variant / VWAP trade / paper books', () => {
  it('parses the variant and the VWAP keys from an r016 get answer', async () => {
    const p = await import('../src/shared/paper');
    const v = (effective: unknown, running: unknown = effective) => ({ effective, running, pending: effective !== running });
    const data = { values: { RULE_TRADE_VARIANT: v('extended', 'normal'), VWAP_TRADE_MODE: v('paper'), VWAP_TRADE_TP_PCT: v(3.0), VWAP_TRADE_SL_PCT: v(1.27), VWAP_TRADE_HORIZON_MIN: v(300) } };
    expect(p.parseRuleVariant(data)).toEqual({ effective: 'extended', running: 'normal', pending: true });
    expect(p.ruleVariantOf('extended')).toBe('extended');
    expect(p.ruleVariantOf('weird')).toBe('normal');
    expect(p.parseVwapMode(data)!.effective).toBe('paper');
    const ex = p.parseVwapExit(data)!;
    expect([ex.tp.effective, ex.sl.effective, ex.horizon.effective]).toEqual([3, 1.27, 300]);
    expect(p.vwapExitChanges({ tp: 2.5, sl: 1, horizon: 120.4 })).toEqual({ VWAP_TRADE_TP_PCT: 2.5, VWAP_TRADE_SL_PCT: 1, VWAP_TRADE_HORIZON_MIN: 120 });
    // an older server without the keys
    expect([p.parseRuleVariant({ values: {} }), p.parseVwapMode({ values: {} }), p.parseVwapExit({ values: {} })]).toEqual([null, null, null]);
  });

  it('maps the paper books to their file prefixes and knows the VWAP stages', async () => {
    const p = await import('../src/shared/paper');
    expect([p.paperBookOf('vwap'), p.paperBookOf(undefined), p.paperBookOf('x')]).toEqual(['vwap', 'edge', 'edge']);
    expect(p.PAPER_BOOK_PREFIX).toEqual({ edge: 'edge_only', vwap: 'vwap_paper' });
    for (const s of ['below_vwap', 'no_trend', 'no_tick_cross', 'chase', 'price_drop', 'stale']) expect(p.PAPER_STAGE_KO[s]).toBeTruthy();
    // a VWAP ENTRY has no probability
    const t = p.parsePaperLines('{"event":"ENTRY","ts":"2026-11-27T09:30:30","code":"005930","name":"삼성전자","entry":103.2,"prob":null,"threshold":null,"trigger":"VWAP_gap=+1.50%_MA5_UP_TICK60_MA5_MA15@09:30:20","tp_px":106.3,"sl_px":101.9,"horizon":300,"group":"watch"}');
    expect([t.length, t[0].prob, t[0].tpPx]).toEqual([1, undefined, 106.3]);
  });
});

// [2026-11-27] VWAP trade live orders
describe('VWAP live keys', () => {
  it('parses the switch, the limits and the day summary', async () => {
    const p = await import('../src/shared/paper');
    const v = (effective: unknown, running: unknown = effective) => ({ effective, running, pending: effective !== running });
    const data = { values: { VWAP_TRADE_LIVE_ORDERS: v(true, false), VWAP_LIVE_ORDER_KRW: v(100000), VWAP_LIVE_MAX_POSITIONS: v(1), VWAP_LIVE_MAX_ENTRIES_PER_DAY: v(2), VWAP_LIVE_DAILY_LOSS_KRW: v(10000) } };
    expect(p.parseVwapLive(data)).toEqual({ effective: 'true', running: 'false', pending: true });
    const lim = p.parseVwapLiveLimits(data)!;
    expect([lim.order.effective, lim.positions.effective, lim.entries.effective, lim.loss.effective]).toEqual([100000, 1, 2, 10000]);
    expect(p.parseEdgeLiveLimits(data)).toBeNull(); // the edge keys are not in this answer
    expect(p.vwapLiveLimitChanges({ order: 150000, positions: 2, entries: 3, loss: 20000 })).toEqual({
      VWAP_LIVE_ORDER_KRW: 150000, VWAP_LIVE_MAX_POSITIONS: 2, VWAP_LIVE_MAX_ENTRIES_PER_DAY: 3, VWAP_LIVE_DAILY_LOSS_KRW: 20000,
    });
    const status = { edge_live: { held: [], open_buys: [], today: { entries: 0 } }, vwap_live: { held: [{ code: '005930', qty: 9, avg_price: 10050 }], open_buys: [], today: { entries: 1, realized: -1200 } } };
    const s = p.parseVwapLiveSummary(status)!;
    expect([s.held[0].code, s.held[0].qty, s.entries, s.realized]).toEqual(['005930', 9, 1, -1200]);
    expect(p.parseEdgeLiveSummary(status)!.held).toEqual([]);
    expect([p.parseVwapLive({ values: {} }), p.parseVwapLiveLimits({ values: {} }), p.parseVwapLiveSummary({})]).toEqual([null, null, null]);
  });
});
