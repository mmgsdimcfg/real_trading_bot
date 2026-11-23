/**
 * Update log
 * - [2026-11-23] type=test - 자동매매 시간: parseNxt reads ENABLE_NXT_TRADING next to ENABLE_NXT_SESSION (absent on an older server),
 *   nxtTradingOn; paper records / status carry the session (regular when missing) and the NXT windows; post-close default 20:05
 */

import { describe, expect, it } from 'vitest';
import { NXT_TRADE_KEY, POSTCLOSE_NXT_DEFAULT, nxtTradingOn, parseNxt, postcloseAt } from '../../src/bot/shared/control';
import { PAPER_SESSION_KO, parsePaperLines, parsePaperStatus } from '../../src/bot/shared/paper';

const flag = (effective: boolean, running: boolean | null = null) => ({ base: true, override: null, has_override: !effective, effective, running, pending: false });

describe('trading hours next to the monitoring hours', () => {
  it('reads both keys of one r016 get', () => {
    const n = parseNxt({ values: { ENABLE_NXT_SESSION: flag(true, true), [NXT_TRADE_KEY]: flag(false, true) }, unknown: [] });
    expect(n?.effective).toBe(true);
    expect(n?.trade).toEqual({ effective: false, running: true, base: true, has_override: true, pending: false });
  });

  it('an older server has no trading key', () => {
    const n = parseNxt({ values: { ENABLE_NXT_SESSION: flag(true) }, unknown: [NXT_TRADE_KEY] });
    expect(n?.trade).toBeUndefined();
    expect(nxtTradingOn(n)).toBe(true); // moves with the monitoring hours
  });

  it('orders in the NXT sessions need both', () => {
    const mk = (session: boolean, trade: boolean, running: boolean | null = null) =>
      parseNxt({ values: { ENABLE_NXT_SESSION: flag(session, running === null ? null : session), [NXT_TRADE_KEY]: flag(trade, running) } });
    expect(nxtTradingOn(mk(true, true))).toBe(true);
    expect(nxtTradingOn(mk(true, false))).toBe(false);
    expect(nxtTradingOn(mk(false, true))).toBe(false);
    expect(nxtTradingOn(mk(true, false, true))).toBe(true); // the running r003 still trades the NXT hours until its restart
    expect(nxtTradingOn(null)).toBe(true);
  });

  it('post-close job after the NXT close is 20:05', () => {
    expect(POSTCLOSE_NXT_DEFAULT).toBe('20:05');
    expect(postcloseAt({ postcloseTime: '20:05', postcloseTimeRegular: '15:40' }, true)).toBe('20:05');
    expect(postcloseAt({ postcloseTime: '20:05', postcloseTimeRegular: '15:40' }, false)).toBe('15:40');
  });
});

describe('paper trades of the NXT sessions', () => {
  const entry = (ts: string, code: string, extra: object = {}) =>
    JSON.stringify({ event: 'ENTRY', ts: `2026-11-23T${ts}`, code, name: 'x', entry: 10000, tp_px: 10200, sl_px: 9830, horizon: 60, ...extra });

  it('keeps the session of a record, regular when missing or unknown', () => {
    const t = parsePaperLines([entry('09:30:05', '000001'), entry('16:00:05', '000002', { session: 'nxt_after' }), entry('08:20:05', '000003', { session: 'nxt_pre' }), entry('10:00:00', '000004', { session: 'zzz' })].join('\n'));
    expect(t.map((x) => x.session)).toEqual(['regular', 'nxt_after', 'nxt_pre', 'regular']);
    expect(PAPER_SESSION_KO.nxt_after).toBe('NXT 장후');
  });

  it('reads the NXT windows of the status file', () => {
    const base = { asof: '2026-11-23T16:00:05', date: '20261123', groups: {}, symbols: [{ code: '000002', name: 'x', bars: 30, position: { entry: 10000, tp: 10200, sl: 9830, ts: '2026-11-23T16:00:05', session: 'nxt_after' } }] };
    const on = parsePaperStatus(JSON.stringify({ ...base, nxt: true, nxt_sessions: { nxt_pre: { entry_window: ['08:05', '08:35'], force_exit: '08:45' }, nxt_after: { entry_window: ['15:35', '19:30'], force_exit: '19:50' } } }));
    expect(on?.nxt).toBe(true);
    expect(on?.nxtSessions?.nxt_after).toEqual({ entryWindow: ['15:35', '19:30'], forceExit: '19:50' });
    expect(on?.symbols[0].position?.session).toBe('nxt_after');
    const off = parsePaperStatus(JSON.stringify(base)); // a server before the change
    expect(off?.nxt).toBe(false);
    expect(off?.nxtSessions).toEqual({});
  });
});
