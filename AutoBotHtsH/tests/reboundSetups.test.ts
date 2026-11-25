/*
 * Update log
 * - [2026-11-25] type=test - the extra setups of the rebound watcher (user 2026-11-25): reference case 377300 카카오페이 2026-11-25
 *   (tests/rebound_377300_20261125.csv, 1-minute bars of 11-24 and 11-25 up to 15:30 from the app's chart cache) - the three points
 *   the user circled: 09:15 V자 반전, 09:41 눌림목 반등, 12:07 VWAP 지지 반등
 */
import { readFileSync } from 'fs';
import { join } from 'path';
import { describe, expect, it } from 'vitest';
import { alertSummary, cooldownKey, detectRebound, hhmm, kindLabel, mergeConfig, REBOUND_DEFAULTS, type ReboundConfig } from '../src/shared/rebound';
import { applySets, parseBars, replay } from '../src/shared/reboundReplay';

const bars = parseBars(readFileSync(join(__dirname, 'rebound_377300_20261125.csv'), 'utf8'));
const KST = 9 * 3600000;
const at = (hm: string) => Date.UTC(2026, 10, 25, +hm.slice(0, 2), +hm.slice(3, 5)) - KST;
const upTo = (hm: string) => bars.filter((b) => b.t <= at(hm));
const day = (cfg: ReboundConfig = REBOUND_DEFAULTS) => replay(bars, cfg, at('00:00'), at('23:59'));

describe('rebound setups - 377300 2026-11-25', () => {
  it('alerts the three circled points, each as its own kind', () => {
    const alerts = day().filter((r) => r.alert).map((r) => `${hhmm(r.t)} ${r.res.setup}`);
    expect(alerts).toContain('09:15 vTurn');
    expect(alerts).toContain('09:41 pullback');
    expect(alerts).toContain('12:07 vwapHold');
  });

  it('09:15 is a V turn off the 09:13 climax low (a bar inside the opening edge)', () => {
    const r = detectRebound(upTo('09:15'));
    expect(r.hit).toBe(true);
    expect(r.setup).toBe('vTurn');
    expect(r.required).toBe(false); // not the classic detector
    expect(r.setups!.vTurn.note).toContain('09:13');
    expect(r.setups!.vTurn.note).toContain('33,350');
    // 09:14 already closed above the climax bar, but it is inside the opening edge: never a detection
    const e = detectRebound(upTo('09:14'));
    expect(e.edge).toBe(true);
    expect(e.hit).toBe(false);
  });

  it('09:41 is the pullback bounce, not the earlier 09:35 bar (MA5 had not come down to MA20 yet)', () => {
    expect(detectRebound(upTo('09:41')).setup).toBe('pullback');
    const early = detectRebound(upTo('09:35'));
    expect(early.hit).toBe(false);
    expect(early.setups!.pullback.note).toContain('MA5 가 MA20 까지 내려오지 않음');
  });

  it('[2026-11-25] stricter pullback: 10:31 (leg +2.8 %, under the 3 % minimum) is no longer a hit; the old thresholds bring it back', () => {
    const r = detectRebound(upTo('10:31'));
    expect(r.hit).toBe(false);
    expect(r.setups!.pullback.note).toContain('상승폭 부족 (기준 +3%)');
    const old = applySets(REBOUND_DEFAULTS, ['setups.pullback.legMinPct=2', 'setups.pullback.retraceMax=0.62', 'setups.pullback.volDryMax=0', 'setups.pullback.trigVolMult=0']);
    expect(detectRebound(upTo('10:31'), old).setup).toBe('pullback');
    // the volume tests: an impossible bound fails the 09:41 bar with its own note
    expect(detectRebound(upTo('09:41'), applySets(REBOUND_DEFAULTS, ['setups.pullback.volDryMax=0.1'])).setups!.pullback.note).toContain('눌림 구간 거래량이 줄지 않음');
    expect(detectRebound(upTo('09:41'), applySets(REBOUND_DEFAULTS, ['setups.pullback.trigVolMult=5'])).setups!.pullback.note).toContain('회복 봉 거래량 부족');
  });

  it('12:07 is the VWAP hold; the bar before it (a down close) is not', () => {
    const r = detectRebound(upTo('12:07'));
    expect(r.setup).toBe('vwapHold');
    expect(r.setups!.vwapHold.note).toContain('34,400');
    expect(detectRebound(upTo('12:06')).hit).toBe(false);
  });

  it('switches: setups.enabled off gives the classic detector only; each kind has its own switch', () => {
    const off = day(applySets(REBOUND_DEFAULTS, ['setups.enabled=false']));
    expect(off.some((r) => r.res.hit)).toBe(false);
    expect(off[0].res.setups).toBeUndefined();
    const noV = detectRebound(upTo('09:15'), applySets(REBOUND_DEFAULTS, ['setups.vTurn.on=false']));
    expect(noV.hit).toBe(false);
    expect(noV.setups!.vTurn.note).toBe('꺼짐');
  });

  it('regular session only: no setup is judged in the NXT pre-market', () => {
    const r = detectRebound(upTo('08:30'));
    expect(r.setups!.vTurn.note).toBe('정규장 밖 봉');
    expect(r.hit).toBe(false);
  });

  it('cooldown is per kind, the title and the summary name the kind, an old saved config gets the defaults', () => {
    const r = detectRebound(upTo('09:41'));
    expect(cooldownKey('377300', r)).toBe('377300|pullback');
    expect(cooldownKey('377300', { })).toBe('377300');
    expect(kindLabel(r)).toBe('눌림목 반등');
    expect(kindLabel(null)).toBe('바닥 반등 초입');
    const text = alertSummary({ code: '377300', name: '카카오페이', price: 34350, changePct: 1.2, res: r });
    expect(text.split('\n')[0]).toBe('[눌림목 반등] 카카오페이(377300)');
    expect(text.length).toBeLessThanOrEqual(200);
    const merged = mergeConfig({ enabled: true, plunge: { window: 30 } });
    expect(merged.setups).toEqual(REBOUND_DEFAULTS.setups);
    expect(mergeConfig({ setups: { vTurn: { on: false, bars: 'x' } } }).setups.vTurn).toEqual({ ...REBOUND_DEFAULTS.setups.vTurn, on: false });
  });
});
