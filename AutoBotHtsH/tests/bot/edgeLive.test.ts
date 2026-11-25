// [2026-11-25] Edge AI live orders (server r001 EDGE_LIVE_ORDERS): the control tab's parsers and inputs
import { describe, expect, it } from 'vitest';
import {
  EDGE_LIVE_KEY,
  EDGE_LIVE_LIMIT_KEYS,
  edgeLiveLimitChanges,
  edgeLiveLimitInput,
  edgeLiveSummaryText,
  parseEdgeLive,
  parseEdgeLiveLimits,
  parseEdgeLiveSummary,
  parsePaperExit,
  parseRuleExit,
  ruleExitChanges,
} from '../../src/bot/shared/paper';

const val = (effective: unknown, running: unknown = effective, pending = false) => ({ effective, running, pending });

describe('RULE_EXIT_* (the rule strategy has its own exits row)', () => {
  it('reads the three values under their own keys, apart from the Edge AI ones', () => {
    const values = {
      RULE_EXIT_TP_PCT: val(0), RULE_EXIT_SL_PCT: val(1.5, 0, true), RULE_EXIT_HORIZON_MIN: val(0),
      EDGE_ONLY_TP_PCT: val(2), EDGE_ONLY_SL_PCT: val(1.77), EDGE_ONLY_HORIZON_MIN: val(400),
    };
    expect(parseRuleExit({ values })?.sl).toEqual({ effective: 1.5, running: 0, pending: true });
    expect(parseRuleExit({ values })?.tp.effective).toBe(0);
    expect(parsePaperExit({ values })?.tp.effective).toBe(2);
    expect(parseRuleExit({ values: { EDGE_ONLY_TP_PCT: val(2) } })).toBeNull(); // an older server: no rule row
    expect(ruleExitChanges({ tp: 0, sl: 1.5, horizon: 90.4 })).toEqual({ RULE_EXIT_TP_PCT: 0, RULE_EXIT_SL_PCT: 1.5, RULE_EXIT_HORIZON_MIN: 90 });
  });
});

describe('EDGE_LIVE_ORDERS', () => {
  it('reads the switch, null on a server without the key', () => {
    expect(parseEdgeLive({ values: { [EDGE_LIVE_KEY]: val(true, false, true) } })).toEqual({ effective: 'true', running: 'false', pending: true });
    expect(parseEdgeLive({ values: { [EDGE_LIVE_KEY]: val(false, null) } })).toEqual({ effective: 'false', running: null, pending: false });
    expect(parseEdgeLive({ values: {}, unknown: [EDGE_LIVE_KEY] })).toBeNull();
    expect(parseEdgeLive(null)).toBeNull();
  });

  it('reads the four limits together or not at all', () => {
    const values = {
      [EDGE_LIVE_LIMIT_KEYS.order]: val(100000),
      [EDGE_LIVE_LIMIT_KEYS.positions]: val(1),
      [EDGE_LIVE_LIMIT_KEYS.entries]: val(1, 3, true),
      [EDGE_LIVE_LIMIT_KEYS.loss]: val(10000),
    };
    const st = parseEdgeLiveLimits({ values });
    expect(st?.order.effective).toBe(100000);
    expect(st?.entries).toEqual({ effective: 1, running: 3, pending: true });
    const { [EDGE_LIVE_LIMIT_KEYS.loss]: _drop, ...partial } = values;
    expect(parseEdgeLiveLimits({ values: partial })).toBeNull();
  });

  it('accepts whole numbers inside the server ranges only', () => {
    expect(edgeLiveLimitInput('order', '100,000')).toBe(100000);
    expect(edgeLiveLimitInput('order', '9999')).toBeNull();
    expect(edgeLiveLimitInput('order', '1000001')).toBeNull();
    expect(edgeLiveLimitInput('positions', '5')).toBe(5);
    expect(edgeLiveLimitInput('positions', '0')).toBeNull();
    expect(edgeLiveLimitInput('entries', '1.5')).toBeNull();
    expect(edgeLiveLimitInput('entries', '')).toBeNull();
    expect(edgeLiveLimitInput('loss', '-1')).toBeNull();
    expect(edgeLiveLimitChanges({ order: 200000, positions: 2, entries: 3, loss: 20000 })).toEqual({
      EDGE_LIVE_ORDER_KRW: 200000,
      EDGE_LIVE_MAX_POSITIONS: 2,
      EDGE_LIVE_MAX_ENTRIES_PER_DAY: 3,
      EDGE_LIVE_DAILY_LOSS_KRW: 20000,
    });
  });

  it('reads the status summary and words it', () => {
    expect(parseEdgeLiveSummary({ edge_live: null })).toBeNull();
    expect(parseEdgeLiveSummary({})).toBeNull();
    const s = parseEdgeLiveSummary({
      edge_live: {
        held: [{ code: '005930', qty: 9, avg_price: 10050, entry_at: '2026-11-25T10:00:00' }],
        open_buys: ['000660'],
        today: { entries: 1, realized: -1234.5, sl_codes: ['035420'], unpriced_sells: 0 },
      },
    });
    expect(s).toEqual({ held: [{ code: '005930', qty: 9, avgPrice: 10050 }], openBuys: ['000660'], entries: 1, realized: -1234.5, slCodes: ['035420'], unpricedSells: 0 });
    expect(edgeLiveSummaryText(s!, 1)).toBe('오늘 매수 1/1회 · 실현 -1,234원 · 보유 005930 9주 · 매수 주문 중 000660 · 오늘 손절 035420');
    const empty = parseEdgeLiveSummary({ edge_live: { held: [], open_buys: [], today: { entries: 0, realized: 0, sl_codes: [], unpriced_sells: 2 } } });
    expect(edgeLiveSummaryText(empty!, null)).toBe('오늘 매수 0회 · 실현 +0원 · 손익 미상 (신규 매수 중단)');
  });
});
