/*
 * Update log
 * - [2026-11-21] type=test - Rebound detector (shared/rebound.ts), replay (shared/reboundReplay.ts) and watcher (main/reboundPoller.ts):
 *   reference case 448900 한국피아이엠 2026-11-20 (tests/rebound_448900_20261120.csv, 1-minute KRX + NXT bars of 11-19 and 11-20 from
 *   the app's chart cache) - detected inside 14:40 ~ 14:58 and never in the falling stretch 13:00 ~ 14:35; indicator values at the
 *   detected bar re-computed here from their definitions; plunge / crossing / angle / session helpers; config merge; the watcher's
 *   pass with a fake backend and clock (completed bars only, cooldown, CSV, a failing symbol does not stop the pass)
 */

import { describe, expect, it, vi, beforeAll, afterAll } from 'vitest';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const userData = mkdtempSync(join(tmpdir(), 'rebound-test-'));
vi.mock('electron', () => ({ app: { getPath: () => userData }, BrowserWindow: class {}, screen: {}, shell: { beep: () => undefined } }));

import { completedBars, detectRebound, findPlunge, lastCrossUp, maAngle, mergeConfig, msToNextPoll, REBOUND_DEFAULTS, sessionOf, tabCodes, type ReboundConfig } from '../src/shared/rebound';
import { applySets, barsToCsv, main, parseBars, replay } from '../src/shared/reboundReplay';
import { ReboundWatcher, sampleAlert, configPath } from '../src/main/reboundPoller';
import { alertHtml, csvRow, type AlertInfo } from '../src/main/reboundAlert';
import type { Bar } from '../src/shared/price';
import type { MarketBackend } from '../src/shared/market';

const FIXTURE = join(__dirname, 'rebound_448900_20261120.csv');
const ref = parseBars(readFileSync(FIXTURE, 'utf8'));
const KST = 9 * 3600000;
const at = (hm: string, date = '20261120') => Date.UTC(+date.slice(0, 4), +date.slice(4, 6) - 1, +date.slice(6, 8), +hm.slice(0, 2), +hm.slice(3, 5)) - KST;
const hm = (t: number) => new Date(t + KST).toISOString().slice(11, 16);
const upTo = (t: number) => ref.filter((b) => b.t <= t);

afterAll(() => rmSync(userData, { recursive: true, force: true }));

describe('reference case 448900 2026-11-20', () => {
  const rows = replay(ref, REBOUND_DEFAULTS, at('13:00'), at('15:20'));
  const hits = rows.filter((r) => r.res.hit).map((r) => hm(r.t));

  it('is detected inside 14:40 ~ 14:58', () => {
    expect(hits.some((h) => h >= '14:40' && h <= '14:58')).toBe(true);
    expect(hits[0]).toBe('14:55');
  });

  it('is not detected in the falling stretch 13:00 ~ 14:35', () => {
    expect(hits.filter((h) => h <= '14:35')).toEqual([]);
    // not even the required conditions alone
    expect(rows.filter((r) => r.res.required && hm(r.t) <= '14:35')).toEqual([]);
  });

  it('alerts once - later detections fall into the cooldown', () => {
    expect(rows.filter((r) => r.alert).map((r) => hm(r.t))).toEqual(['14:55']);
  });

  it('finds the 14:33 plunge bar and reports every condition at 14:55', () => {
    const r = detectRebound(upTo(at('14:55')));
    expect(r.hit).toBe(true);
    expect(hm(r.snap.plungeT)).toBe('14:33');
    expect(r.snap.plungeLow).toBe(38000);
    expect(Object.values(r.conds).every((c) => c.ok)).toBe(true);
    expect(r.score).toBe(3);
    expect(r.bonus.adxTurn.ok && r.bonus.bbSqueeze.ok && r.bonus.volDry.ok).toBe(true);
    expect(r.snap.vwapPct).toBeLessThan(0); // still under VWAP: shown, not a condition
  });

  it('the original thresholds (angle 15, 4 of 5 bars, ADX low within 3 bars) fire only after the window', () => {
    const strict = applySets(REBOUND_DEFAULTS, ['angle.min=15', 'recentMin=4', 'bonus.adxTroughBars=3']);
    const h = replay(ref, strict, at('13:00'), at('15:20')).filter((r) => r.res.hit).map((r) => hm(r.t));
    expect(h.filter((x) => x <= '14:58')).toEqual([]);
  });
});

describe('indicator values at the detected bar, from their definitions', () => {
  const bars = upTo(at('14:55'));
  const c = bars.map((b) => b.c);
  const n = c.length;
  const s = detectRebound(bars).snap;

  it('MA5 / MA20 and the MA5 angle', () => {
    const mean = (k: number, end = n) => c.slice(end - k, end).reduce((a, b) => a + b, 0) / k;
    expect(s.ma5).toBeCloseTo(mean(5), 9);
    expect(s.ma20).toBeCloseTo(mean(20), 9);
    const prev = mean(5, n - 3);
    expect(s.angle).toBeCloseTo((Math.atan((((mean(5) - prev) / prev) * 100) / 3 * 10) * 180) / Math.PI, 9);
  });

  it('RSI(14), chart method: gains / (gains + losses) of the last 14 changes', () => {
    let g = 0;
    let l = 0;
    for (let i = n - 14; i < n; i++) {
      const d = c[i] - c[i - 1];
      if (d > 0) g += d;
      else l -= d;
    }
    expect(s.rsi).toBeCloseTo((100 * g) / (g + l), 9);
  });

  it('MACD(12, 26, 9) with recursive EMAs', () => {
    const ema = (x: number[], span: number) => {
      const a = 2 / (span + 1);
      const out = [x[0]];
      for (let i = 1; i < x.length; i++) out.push(a * x[i] + (1 - a) * out[i - 1]);
      return out;
    };
    const m = ema(c, 12).map((v, i) => v - ema(c, 26)[i]);
    const sig = ema(m, 9);
    expect(s.macd).toBeCloseTo(m[n - 1], 6);
    expect(s.macdSignal).toBeCloseTo(sig[n - 1], 6);
    expect(s.macdOsc).toBeCloseTo(m[n - 1] - sig[n - 1], 6);
  });

  it('VWAP restarts with the day', () => {
    const day = bars.filter((b) => b.t >= at('00:00'));
    const pv = day.reduce((a, b) => a + b.c * b.v, 0);
    const v = day.reduce((a, b) => a + b.v, 0);
    expect(s.vwap).toBeCloseTo(pv / v, 6);
  });

  it('Wilder variant differs and stays selectable', () => {
    const w = detectRebound(bars, applySets(REBOUND_DEFAULTS, ['indicator.rsiMethod=wilder', 'indicator.adxMethod=wilder'])).snap;
    expect(Math.abs(w.rsi - s.rsi)).toBeGreaterThan(0.5);
    expect(w.macd).toBeCloseTo(s.macd, 9);
  });
});

describe('helpers', () => {
  const bar = (t: number, o: number, h: number, l: number, c: number, v: number): Bar => ({ t: t * 60000, o, h, l, c, v });
  const flat = (k: number, px = 10000, v = 100) => Array.from({ length: k }, (_, i) => bar(i, px, px, px, px, v));

  it('maAngle: 0.1 % per bar with k = 10 is 45 degrees', () => {
    const ma = [100, 100.1, 100.2, 100.3];
    expect(maAngle(ma, 3, 10)[3]).toBeCloseTo(45, 6);
    expect(Number.isNaN(maAngle(ma, 3, 10)[2])).toBe(true);
    expect(maAngle([100, 100, 100, 99.7], 3, 10)[3]).toBeCloseTo(-45, 6);
  });

  it('lastCrossUp: the latest crossing inside the range', () => {
    const a = [1, 1, 3, 1, 3, 3];
    const b = [2, 2, 2, 2, 2, 2];
    expect(lastCrossUp(a, b, 0, 5)).toBe(4);
    expect(lastCrossUp(a, b, 0, 3)).toBe(2);
    expect(lastCrossUp(a, b, 5, 5)).toBe(-1);
  });

  it('findPlunge: a drop on 3x volume whose low holds', () => {
    const bars = flat(70);
    bars.push(bar(70, 10000, 10000, 9880, 9900, 400)); // -1 %, 4x volume
    bars.push(bar(71, 9900, 9950, 9900, 9950, 100));
    expect(findPlunge(bars, REBOUND_DEFAULTS)).toBe(70);
    // a later bar undercuts the low: no longer the bottom
    expect(findPlunge([...bars, bar(72, 9950, 9950, 9870, 9900, 100)], REBOUND_DEFAULTS)).toBe(-1);
    // thin volume: not a plunge
    const thin = flat(70);
    thin.push(bar(70, 10000, 10000, 9880, 9900, 200));
    expect(findPlunge(thin, REBOUND_DEFAULTS)).toBe(-1);
    // out of the 40-bar window
    expect(findPlunge([...bars, ...Array.from({ length: 45 }, (_, i) => bar(72 + i, 9950, 9950, 9950, 9950, 100))], REBOUND_DEFAULTS)).toBe(-1);
  });

  it('findPlunge: a long lower wick counts, a tiny one does not', () => {
    const wick = flat(70);
    wick.push(bar(70, 10000, 10000, 9900, 9990, 400)); // wick 90 vs body 10, 0.9 % of the price
    expect(findPlunge(wick, REBOUND_DEFAULTS)).toBe(70);
    const tiny = flat(70);
    tiny.push(bar(70, 10000, 10000, 9990, 10000, 400)); // 0.1 % wick on a doji
    expect(findPlunge(tiny, REBOUND_DEFAULTS)).toBe(-1);
  });

  it('too few bars: nothing is judged', () => {
    const r = detectRebound(flat(50));
    expect(r.hit).toBe(false);
    expect(r.skip).toContain('50');
  });

  it('sessionOf: KRX hours, NXT pre / after market, rest', () => {
    const c = REBOUND_DEFAULTS;
    expect(sessionOf(c, at('09:00'))).toBe('krx');
    expect(sessionOf(c, at('15:30'))).toBe('krx');
    expect(sessionOf(c, at('08:10'))).toBe('nxt');
    expect(sessionOf(c, at('18:00'))).toBe('nxt');
    expect(sessionOf(c, at('08:55'))).toBe(null);
    expect(sessionOf(c, at('20:01'))).toBe(null);
    expect(sessionOf(c, at('10:00', '20261121'))).toBe(null); // Saturday
    const noNxt = mergeConfig({ sessions: { nxtPre: null, nxtAfter: null } });
    expect(sessionOf(noNxt, at('18:00'))).toBe(null);
  });

  it('completedBars drops the running minute only', () => {
    const bars = [bar(10, 1, 1, 1, 1, 1), bar(11, 1, 1, 1, 1, 1), bar(12, 1, 1, 1, 1, 1)];
    expect(completedBars(bars, 12 * 60000 + 2000).map((b) => b.t / 60000)).toEqual([10, 11]);
    expect(completedBars(bars, 13 * 60000 + 2000)).toHaveLength(3);
  });

  it('msToNextPoll: a few seconds after the next minute', () => {
    expect(msToNextPoll(60000 * 5 + 30000, 2)).toBe(32000);
    expect(msToNextPoll(60000 * 5, 2)).toBe(2000);
    expect(msToNextPoll(60000 * 5 + 1900, 2)).toBe(60100); // too close: the minute after
  });

  it('tabCodes: the tab by name, case ignored; null when missing', () => {
    const g = [{ name: '관심 1', codes: ['005930'] }, { name: ' test ', codes: ['448900', '448900', 'bad', '000660'] }];
    expect(tabCodes(g, 'TEST')).toEqual(['448900', '000660']);
    expect(tabCodes(g, 'none')).toBe(null);
  });

  it('mergeConfig: wrong types and unknown keys fall back to the defaults', () => {
    const c = mergeConfig({ tab: 'ABC', lookback: 'x', angle: { min: 20, extra: 1 }, indicator: { rsiMethod: 'nope' }, sessions: { krx: ['9', 'x'] }, junk: 1 });
    expect(c.tab).toBe('ABC');
    expect(c.lookback).toBe(REBOUND_DEFAULTS.lookback);
    expect(c.angle).toEqual({ ...REBOUND_DEFAULTS.angle, min: 20 });
    expect(c.indicator.rsiMethod).toBe(REBOUND_DEFAULTS.indicator.rsiMethod);
    expect(c.sessions.krx).toEqual(REBOUND_DEFAULTS.sessions.krx);
    expect((c as unknown as Record<string, unknown>).junk).toBeUndefined();
    expect(mergeConfig(undefined)).toEqual(REBOUND_DEFAULTS);
  });
});

describe('replay files and CLI', () => {
  it('CSV and chart-cache JSON parse to the same bars', () => {
    const some = ref.slice(0, 50);
    expect(parseBars(barsToCsv(some))).toEqual(some);
    expect(parseBars(JSON.stringify({ bars: some }))).toEqual(some);
    expect(parseBars('t,o,h,l,c,v\n60000,1,2,0,1,5\nbad,row\n')).toEqual([{ t: 60000, o: 1, h: 2, l: 0, c: 1, v: 5 }]);
  });

  it('main() lists the detection of the reference file', () => {
    const out: string[] = [];
    expect(main([FIXTURE, '--date', '20261120', '--from', '13:00', '--to', '15:20'], (s) => out.push(s))).toBe(0);
    expect(out.some((l) => l.startsWith('14:55 감지(알림)'))).toBe(true);
    expect(out[out.length - 1]).toBe('감지 3봉, 알림 1회');
    expect(main([], () => undefined)).toBe(2);
    expect(() => applySets(REBOUND_DEFAULTS, ['no.such=1'])).toThrow();
  });
});

describe('alert text', () => {
  const a: AlertInfo = { ...sampleAlert(REBOUND_DEFAULTS), name: '<b>A&B</b>, "x"' };

  it('popup HTML escapes the name and carries no script', () => {
    const h = alertHtml(a, 3);
    expect(h).toContain('&lt;b&gt;A&amp;B&lt;/b&gt;');
    expect(h).not.toContain('<script');
    expect(h).toContain('https://rebound.invalid/open/000000');
    expect(h).toContain('VWAP 대비');
  });

  it('CSV row keeps its columns whatever the name holds', () => {
    expect(csvRow(a, true).split(',')).toHaveLength(29);
    expect(csvRow(a, false).split(',')[4]).toBe('0');
  });
});

describe('watcher pass', () => {
  let clock = at('14:56') + 2000;
  const alerts: AlertInfo[] = [];
  const logs: string[] = [];
  const calls: string[] = [];
  const backend = {
    kind: 'kis',
    chart: async (req: { code: string }) => {
      calls.push(req.code);
      if (req.code === '000001') throw new Error('EGW00201 초당 거래건수 초과');
      // what KIS would return at `clock`: every bar started so far, the running minute included
      return { bars: ref.filter((b) => b.t <= clock), name: '한국피아이엠' };
    },
    quote: async () => ({ code: '448900', price: 38500, change: -950, changePct: -2.41, sign: '5' }),
  } as unknown as MarketBackend;
  let current: MarketBackend | null = backend;
  const watcher = new ReboundWatcher({
    backend: () => current,
    master: { get: () => undefined, nxtKnown: false },
    log: (l) => logs.push(l),
    alert: (a) => alerts.push(a),
    status: () => undefined,
    now: () => clock,
  });
  const writeTab = (name: string, codes: string[]) => {
    mkdirSync(join(userData, 'watchlist'), { recursive: true });
    writeFileSync(join(userData, 'watchlist', 'watchlist.json'), JSON.stringify({ groups: [{ id: 'g1', name, codes }], imported: {} }));
  };
  const setCfg = (over: Partial<ReboundConfig>) => {
    mkdirSync(join(userData, 'rebound'), { recursive: true });
    const cfg = mergeConfig({ ...REBOUND_DEFAULTS, poll: { ...REBOUND_DEFAULTS.poll, gapMs: 0, backoffMs: 1, tabReloadSec: 0 }, ...over });
    writeFileSync(configPath(), JSON.stringify(cfg));
    return new Promise((r) => setTimeout(r, 15)); // a new mtime
  };
  const csv = () => readFileSync(join(userData, 'rebound', 'detections_20261120.csv'), 'utf8').trim().split('\r\n');

  beforeAll(async () => {
    writeTab('TEST', ['000001', '448900']);
    await setCfg({});
  });

  it('writes the default config on first use', () => {
    rmSync(configPath());
    watcher.loadConfig();
    expect(mergeConfig(JSON.parse(readFileSync(configPath(), 'utf8')))).toEqual(REBOUND_DEFAULTS);
    return setCfg({});
  });

  it('14:56:02 - the 14:55 bar is judged, one popup; the failing symbol is retried and skipped', async () => {
    await watcher.tick();
    expect(alerts).toHaveLength(1);
    expect(alerts[0].code).toBe('448900');
    expect(hm(alerts[0].res.snap.t)).toBe('14:55'); // not the running 14:56 bar
    expect(alerts[0].changePct).toBe(-2.41);
    expect(calls.filter((c) => c === '000001')).toHaveLength(REBOUND_DEFAULTS.poll.retries);
    expect(watcher.status).toMatchObject({ state: 'run', codes: 2, judged: 1, failed: 1, alerts: 1 });
    expect(csv()).toHaveLength(2);
    expect(csv()[1].split(',').slice(1, 6)).toEqual(['14:55', '448900', '한국피아이엠', '1', '3']);
  });

  it('the same minute again: nothing new is judged', async () => {
    await watcher.tick();
    expect(alerts).toHaveLength(1);
    expect(watcher.status.judged).toBe(0);
  });

  it('15:01:02 - detected again but inside the cooldown: logged, no popup', async () => {
    clock = at('15:01') + 2000;
    await watcher.tick();
    expect(alerts).toHaveLength(1);
    expect(csv()).toHaveLength(3);
    expect(csv()[2].split(',').slice(1, 5)).toEqual(['15:00', '448900', '한국피아이엠', '0']);
  });

  it('rests outside the sessions, waits for a login, reports a missing tab, can be switched off', async () => {
    const n = calls.length;
    clock = at('20:30') + 2000;
    await watcher.tick();
    expect(watcher.status.state).toBe('rest');
    clock = at('15:02') + 2000;
    current = null;
    await watcher.tick();
    expect(watcher.status.state).toBe('login');
    current = backend;
    writeTab('다른 탭', ['448900']);
    await watcher.tick();
    expect(watcher.status.state).toBe('notab');
    watcher.setEnabled(false);
    await watcher.tick();
    expect(watcher.status).toMatchObject({ enabled: false, state: 'off' });
    expect(calls.length).toBe(n);
    expect(JSON.parse(readFileSync(configPath(), 'utf8')).enabled).toBe(false);
    expect(existsSync(configPath())).toBe(true);
  });
});
