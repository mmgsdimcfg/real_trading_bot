/*
 * Update log
 * - [2026-11-25] type=test - paper exits set in the control tab (server r001 EDGE_ONLY_TP_PCT / EDGE_ONLY_SL_PCT /
 *   EDGE_ONLY_HORIZON_MIN, 0 = the model's label): r016 get parsing, input boxes, the set changes, status model_label
 */

import { describe, expect, it } from 'vitest';
import { paperExitChanges, paperExitInput, parsePaperExit, parsePaperStatus } from '../../src/bot/shared/paper';

const row = (effective: unknown, running: unknown = effective, pending = false) => ({ base: 0, override: null, effective, running, pending });

describe('paper exits', () => {
  it('r016 get -> state; null on a server without the keys', () => {
    const data = { values: { EDGE_ONLY_TP_PCT: row(1.5, 0.0, true), EDGE_ONLY_SL_PCT: row(0.0), EDGE_ONLY_HORIZON_MIN: row(90, null) } };
    expect(parsePaperExit(data)).toEqual({
      tp: { effective: 1.5, running: 0, pending: true },
      sl: { effective: 0, running: 0, pending: false },
      horizon: { effective: 90, running: null, pending: false },
    });
    expect(parsePaperExit({ values: { EDGE_ONLY_TP_PCT: row(1.5) }, unknown: ['EDGE_ONLY_SL_PCT', 'EDGE_ONLY_HORIZON_MIN'] })).toBeNull();
    expect(parsePaperExit(null)).toBeNull();
  });

  it('input boxes: empty = 0 (the model), range and number checks, minutes whole', () => {
    expect(paperExitInput('tp', '')).toBe(0);
    expect(paperExitInput('tp', ' 1.5 ')).toBe(1.5);
    expect(paperExitInput('sl', '1.777')).toBe(1.78);
    expect(paperExitInput('tp', '31')).toBeNull();
    expect(paperExitInput('tp', '-1')).toBeNull();
    expect(paperExitInput('tp', 'abc')).toBeNull();
    expect(paperExitInput('horizon', '90')).toBe(90);
    expect(paperExitInput('horizon', '90.5')).toBeNull();
    expect(paperExitInput('horizon', '721')).toBeNull();
  });

  it('set changes: the three r001 keys, minutes as an integer', () => {
    expect(paperExitChanges({ tp: 1.5, sl: 0, horizon: 90 })).toEqual({ EDGE_ONLY_TP_PCT: 1.5, EDGE_ONLY_SL_PCT: 0, EDGE_ONLY_HORIZON_MIN: 90 });
  });

  it('status: model_label kept beside label', () => {
    const st = parsePaperStatus(
      JSON.stringify({ asof: '2026-11-25T10:00:00', date: '20261125', label: { tp: 1.5, sl: 1.77, horizon: 90 }, model_label: { tp: 2, sl: 1.77, horizon: 400 }, groups: {}, symbols: [] }),
    );
    expect(st?.label).toEqual({ tp: 1.5, sl: 1.77, horizon: 90 });
    expect(st?.modelLabel).toEqual({ tp: 2, sl: 1.77, horizon: 400 });
  });
});

describe('[2026-11-25] session close reasons', () => {
  it('close_regular / close_nxt labels and the estimated flag', async () => {
    const { PAPER_EXIT_KO, parsePaperLines } = await import('../../src/bot/shared/paper');
    expect([PAPER_EXIT_KO.close_regular, PAPER_EXIT_KO.close_nxt, PAPER_EXIT_KO.eod]).toEqual(['정규장 종가 청산', 'NXT 마감 청산', '장 마감']);
    const lines = [
      JSON.stringify({ event: 'ENTRY', ts: '2026-11-25T09:30:05', code: '010060', name: 'x', entry: 10310, prob: 0.6 }),
      JSON.stringify({ event: 'EXIT', ts: '2026-11-25T19:59:51', code: '010060', reason: 'close_nxt', exit: 10290, net_pct: -0.4, estimated: true }),
    ].join('\n');
    const t = parsePaperLines(lines)[0];
    expect([t.reason, t.estimated]).toEqual(['close_nxt', true]);
  });
});
