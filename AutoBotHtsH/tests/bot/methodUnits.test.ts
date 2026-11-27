import { describe, expect, it } from 'vitest';
import { parseMethodTargets, runningUnits, targetListToStore, targetsInUse, unitInfos } from '../../src/bot/shared/methodUnits';
import { methodRows, methodTotals, filterMethodRows } from '../../src/bot/shared/methodBoard';
import type { SymbolSummary } from '../../src/bot/shared/types';

const m = (effective: string, running: string | null = effective) => ({ effective, running, pending: running !== null && running !== effective });
/** today's settings (2026-11-27): rule off, Edge AI paper, VWAP real orders */
const today = { strategy: m('normal'), ruleBuy: m('false'), paper: m('paper'), paperLogOnly: m('false'), edgeLive: m('false'), vwapMode: m('paper'), vwapLive: m('true') };
const by = (us: ReturnType<typeof unitInfos>) => Object.fromEntries(us.map((u) => [u.key, u]));

describe('unitInfos', () => {
  it('running bot: each unit says what the bot does, the paper unit of a method with real orders only closes', () => {
    const u = by(unitInfos(today, true));
    expect(u.rule).toMatchObject({ switchOn: false, state: 'stop', entryRunning: false });
    expect(u.edge_paper).toMatchObject({ switchOn: true, state: 'run', entryRunning: true });
    expect(u.edge_live).toMatchObject({ switchOn: false, state: 'stop', locked: '' });
    expect(u.vwap_paper).toMatchObject({ switchOn: false, state: 'closing', entryRunning: false }); // real orders run: the paper card is off
    expect(u.vwap_live).toMatchObject({ switchOn: true, state: 'run', entryRunning: true });
    expect(runningUnits(Object.values(u))).toBe(2);
  });

  it('a stored switch is not a running state: bot off / no status / an unreported value never read "실행 중"', () => {
    const off = by(unitInfos(today, false));
    expect(off.vwap_live).toMatchObject({ switchOn: true, state: 'botoff', entryRunning: null });
    expect(off.rule.state).toBe('stop');
    expect(runningUnits(Object.values(off))).toBe(0);
    const none = by(unitInfos(today, null));
    expect(none.vwap_live.state).toBe('unknown');
    const unreported = by(unitInfos({ ...today, vwapLive: m('true', null) }, true)); // the bot runs but did not report the key
    expect(unreported.vwap_live).toMatchObject({ state: 'unknown', entryRunning: null });
    expect(unreported.edge_paper.state).toBe('run'); // the other methods are not affected
  });

  it('stored differs from running: pending, both ways', () => {
    const on = by(unitInfos({ ...today, ruleBuy: m('true', 'false') }, true));
    expect(on.rule).toMatchObject({ switchOn: true, state: 'pending', entryRunning: false, stateKo: '적용 대기 · 재시작 후 켜짐' });
    const offNext = by(unitInfos({ ...today, vwapLive: m('false', 'true') }, true));
    expect(offNext.vwap_live).toMatchObject({ switchOn: false, state: 'pending', entryRunning: true, stateKo: '실행 중 · 재시작 후 꺼짐' });
  });

  it('locks: real orders need the method on; Edge AI alone locks the rule and the Edge switches; an unknown key locks', () => {
    const methodOff = by(unitInfos({ ...today, vwapMode: m('off'), vwapLive: m('false') }, true));
    expect(methodOff.vwap_live).toMatchObject({ switchOn: false, locked: '' }); // can be switched on (the method goes on first)
    expect(methodOff.vwap_paper).toMatchObject({ switchOn: false, locked: '', state: 'stop' });
    const edgeOnly = by(unitInfos({ ...today, strategy: m('edge') }, true));
    expect(edgeOnly.rule.locked).toBe('Edge AI 단독 모드');
    expect(edgeOnly.edge_live.locked).toBe('단독 모드 · 실전');
    expect(edgeOnly.edge_live.entryRunning).toBe(true);
    const old = by(unitInfos({ strategy: m('normal'), paper: m('paper') }, true)); // a server without the newer keys
    expect(old.vwap_paper).toMatchObject({ switchOn: null, locked: '서버가 이 설정을 모릅니다' });
    expect(old.vwap_live.switchOn).toBeNull();
  });
});

describe('of a method\'s paper and real cards at most one is on', () => {
  it('follows the stored settings: paper, real, or neither', () => {
    const pair = (c: Parameters<typeof unitInfos>[0], k: 'vwap' | 'edge') => {
      const u = by(unitInfos(c, true));
      return [u[`${k}_paper`].switchOn, u[`${k}_live`].switchOn];
    };
    expect(pair(today, 'vwap')).toEqual([false, true]); // VWAP real orders
    expect(pair(today, 'edge')).toEqual([true, false]); // Edge AI paper
    expect(pair({ ...today, vwapLive: m('false') }, 'vwap')).toEqual([true, false]);
    expect(pair({ ...today, vwapMode: m('off'), vwapLive: m('false') }, 'vwap')).toEqual([false, false]);
    // real orders stored on while the method itself is off: nothing runs, nothing is shown on
    expect(pair({ ...today, vwapMode: m('off'), vwapLive: m('true') }, 'vwap')).toEqual([false, false]);
    expect(pair({ ...today, edgeLive: m('true') }, 'edge')).toEqual([false, true]);
  });
});

describe('what the bot itself said overrides the settings', () => {
  it('dry-run, a "new buys off" start line and a recent block line stop a real unit from reading "실행 중"', () => {
    expect(by(unitInfos(today, true, { dryRun: true })).vwap_live).toMatchObject({ state: 'stop', entryRunning: false, stateKo: '주문 안 함 (dry-run)' });
    expect(by(unitInfos(today, true, { liveNote: { vwap: 'off' } })).vwap_live).toMatchObject({ state: 'stop', entryRunning: false });
    expect(by(unitInfos(today, true, { liveNote: { vwap: 'blocked' } })).vwap_live).toMatchObject({ state: 'closing', entryRunning: false });
    expect(by(unitInfos(today, true, { liveNote: { vwap: 'on' } })).vwap_live).toMatchObject({ state: 'run', entryRunning: true, stateKo: '실행 중 (설정 기준)' });
    expect(by(unitInfos(today, true, { dryRun: true })).edge_paper.state).toBe('run'); // paper trading is not an order
  });
  it('Edge AI real orders with the Edge AI trade itself off in the running bot are not running', () => {
    const u = by(unitInfos({ ...today, paper: m('off'), edgeLive: m('true') }, true));
    expect(u.edge_live.entryRunning).toBe(false);
    expect(u.edge_live).toMatchObject({ switchOn: false, locked: '' });
  });
});

describe('method targets', () => {
  const data = { values: { RULE_TARGET_CODES: { effective: [], running: [], pending: false }, VWAP_TARGET_CODES: { effective: ['237690', '005930'], running: [], pending: true } } };
  it('reads stored and running lists, a method the server does not know is absent', () => {
    const t = parseMethodTargets(data);
    expect(t.rule).toEqual({ effective: [], running: [], pending: false });
    expect(t.vwap).toEqual({ effective: ['005930', '237690'], running: [], pending: true });
    expect(t.edge).toBeUndefined();
    expect(targetsInUse(t.vwap)).toEqual({ list: [], from: 'running' }); // the running bot still buys everything
    expect(targetsInUse({ effective: ['005930'], running: null, pending: false })).toEqual({ list: ['005930'], from: 'stored' });
    expect(targetsInUse(undefined).from).toBe('none');
    // the bot runs but did not report its list: the stored list is not used to mark rows
    expect(targetsInUse({ effective: ['005930'], running: null, pending: false }, true)).toEqual({ list: [], from: 'unknown' });
  });
  it('picking everything or nothing stores no list; a kept code outside the watchlist stays', () => {
    const watched = ['000001', '000002', '000003'];
    expect(targetListToStore(new Set(watched), watched)).toEqual([]);
    expect(targetListToStore(new Set(), watched)).toEqual([]);
    expect(targetListToStore(new Set(['000002', '000001']), watched)).toEqual(['000001', '000002']);
    expect(targetListToStore(new Set([...watched, '999999']), watched)).toEqual(['000001', '000002', '000003', '999999']);
    expect(targetListToStore(new Set(['000001', 'bad']), watched)).toEqual(['000001']);
  });
  it('rows outside the list in use are marked and not counted as targets', () => {
    const sym = (code: string): SymbolSummary => ({ code, name: code, watch: 'active', furthestOrderToday: -1, blockCounts: {}, sellCounts: {}, realizedKrw: 0, evalCount: 0 });
    const rows = methodRows('vwap', { symbols: [sym('000001'), sym('000002'), sym('000003')], trades: [], targets: ['000002'] });
    expect(rows.filter((r) => r.notTarget).map((r) => r.code)).toEqual(['000001', '000003']);
    expect(methodTotals(rows).targets).toBe(1);
    expect(filterMethodRows(rows, 'target').map((r) => r.code)).toEqual(['000002']);
    expect(methodRows('vwap', { symbols: [sym('000001')], trades: [], targets: [] })[0].notTarget).toBe(false);
  });
});
