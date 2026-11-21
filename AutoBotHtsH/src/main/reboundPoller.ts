// Update log
// - [2026-11-21] type=feat (Rebound watcher (user 2026-11-21): once a minute, a few seconds after the minute turns, the symbols of the
//   watchlist tab "TEST" (config) are walked one by one - 1-minute bars through the market backend's chart() (the chart's own cache:
//   after the first load one call per symbol), the running bar dropped, detectRebound() on the rest. A detection is written to the
//   CSV and, unless the symbol is in its cooldown, shown as a popup. Runs on timers in the main process (never blocks the window);
//   rests outside the session windows, re-reads the tab every 5 minutes and the config when its file changes, retries a failed
//   symbol with a doubling back-off and goes on with the next one. Inquiry calls only - there is no order code)

import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { join } from 'path';
import { completedBars, detectRebound, hhmm, mergeConfig, msToNextPoll, REBOUND_DEFAULTS, sessionOf, tabCodes, type ReboundConfig } from '../shared/rebound';
import type { MarketBackend } from '../shared/market';
import type { MasterItem } from '../shared/kis';
import { logDetection, reboundDir, type AlertInfo } from './reboundAlert';
import { loadWatchFile } from './watchlist';

export interface ReboundStatus {
  enabled: boolean;
  tab: string;
  /** 'off' switched off, 'rest' outside the sessions, 'login' no market login, 'notab' the tab does not exist, 'run' watching */
  state: 'off' | 'rest' | 'login' | 'notab' | 'run';
  /** symbols of the tab */
  codes: number;
  /** epoch ms of the last finished pass and what it did */
  lastPass?: number;
  judged?: number;
  failed?: number;
  /** alerts shown since the app started */
  alerts: number;
  note?: string;
}

interface Deps {
  backend: () => MarketBackend | null;
  master: { get(code: string): MasterItem | undefined; readonly nxtKnown: boolean };
  log: (line: string) => void;
  alert: (a: AlertInfo, cfg: ReboundConfig) => void;
  status: (s: ReboundStatus) => void;
  now?: () => number;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
export const configPath = () => join(reboundDir(), 'config.json');

export class ReboundWatcher {
  private cfg: ReboundConfig = REBOUND_DEFAULTS;
  private cfgMtime = -1;
  private timer: NodeJS.Timeout | null = null;
  private busy = false;
  private stopped = true;
  /** stop() was called while a pass was running: the pass ends after the current symbol */
  private halt = false;
  private codes: string[] | null = null;
  private tabAt = 0;
  /** code -> time of the last bar judged (a minute without a trade is not judged twice) */
  private judgedBar = new Map<string, number>();
  /** code -> epoch ms of the last popup */
  private lastAlert = new Map<string, number>();
  private st: ReboundStatus = { enabled: true, tab: 'TEST', state: 'rest', codes: 0, alerts: 0 };

  constructor(private d: Deps) {}

  private now(): number {
    return this.d.now ? this.d.now() : Date.now();
  }

  get status(): ReboundStatus {
    return this.st;
  }
  get config(): ReboundConfig {
    return this.cfg;
  }

  /** read the config file when it changed; the first run writes the defaults so the file can be found and edited */
  loadConfig(): void {
    const p = configPath();
    try {
      if (!existsSync(p)) {
        if (!existsSync(reboundDir())) mkdirSync(reboundDir(), { recursive: true });
        writeFileSync(p, JSON.stringify(REBOUND_DEFAULTS, null, 2), 'utf8');
      }
      const m = statSync(p).mtimeMs;
      if (m === this.cfgMtime) return;
      this.cfgMtime = m;
      this.cfg = mergeConfig(JSON.parse(readFileSync(p, 'utf8').replace(/^﻿/, '')));
      this.tabAt = 0; // the tab name may have changed
      this.d.log(`[rebound] 설정 읽음 (탭 "${this.cfg.tab}", ${this.cfg.enabled ? '켜짐' : '꺼짐'})`);
    } catch (e) {
      this.d.log(`[rebound] 설정 파일 오류 - 이전 설정 유지: ${(e as Error).message}`);
    }
  }

  /** switch the watcher on / off (written to the config file, the other values kept) */
  setEnabled(on: boolean): void {
    this.loadConfig();
    this.cfg = { ...this.cfg, enabled: on };
    try {
      writeFileSync(configPath(), JSON.stringify(this.cfg, null, 2), 'utf8');
      this.cfgMtime = statSync(configPath()).mtimeMs;
    } catch (e) {
      this.d.log(`[rebound] 설정 저장 실패: ${(e as Error).message}`);
    }
    this.setStatus({ enabled: on, state: on ? 'rest' : 'off', note: undefined });
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.halt = false;
    this.loadConfig();
    this.setStatus({ enabled: this.cfg.enabled, tab: this.cfg.tab, state: this.cfg.enabled ? 'rest' : 'off' });
    this.schedule();
  }

  stop(): void {
    this.stopped = true;
    this.halt = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  private schedule(): void {
    if (this.stopped) return;
    this.timer = setTimeout(() => {
      void this.tick().finally(() => this.schedule());
    }, msToNextPoll(this.now(), this.cfg.poll.delaySec));
  }

  private setStatus(p: Partial<ReboundStatus>): void {
    this.st = { ...this.st, ...p };
    this.d.status(this.st);
  }

  private loadTab(): void {
    const now = this.now();
    if (this.codes && now - this.tabAt < this.cfg.poll.tabReloadSec * 1000) return;
    this.tabAt = now;
    const before = this.codes?.join(',');
    this.codes = tabCodes(loadWatchFile()?.groups || [], this.cfg.tab);
    if (this.codes && this.codes.join(',') !== before) this.d.log(`[rebound] "${this.cfg.tab}" 탭 ${this.codes.length}종목`);
    for (const c of [...this.judgedBar.keys()]) if (!this.codes?.includes(c)) this.judgedBar.delete(c);
  }

  /** one symbol: bars (with retries), the completed ones judged; true when a new bar was judged */
  private async one(backend: MarketBackend, code: string): Promise<boolean> {
    const p = this.cfg.poll;
    let lastErr: unknown;
    for (let a = 0; a < Math.max(1, p.retries); a++) {
      if (a) await sleep(p.backoffMs * 2 ** (a - 1));
      try {
        const res = await backend.chart({ code, kind: 'min', n: 1, days: Math.max(1, Math.round(p.days)), refresh: true });
        const now = this.now();
        const bars = completedBars(res.bars, now);
        const last = bars.length ? bars[bars.length - 1].t : 0;
        if (!last || this.judgedBar.get(code) === last) return false;
        this.judgedBar.set(code, last);
        if (now - last > 3 * 60000) return false; // an old bar (no trade for minutes, or the first pass of the day): nothing new to tell
        const r = detectRebound(bars, this.cfg);
        if (r.hit) await this.detected(backend, code, res.name || this.d.master.get(code)?.name || code, r);
        return true;
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr;
  }

  private async detected(backend: MarketBackend, code: string, name: string, res: AlertInfo['res']): Promise<void> {
    const now = this.now();
    const info: AlertInfo = { code, name, at: now, res };
    const cool = now - (this.lastAlert.get(code) ?? -Infinity) < this.cfg.alert.cooldownMin * 60000;
    if (!cool) {
      try {
        const q = await backend.quote(code);
        info.price = q.price;
        info.changePct = q.changePct;
      } catch {
        /* the popup shows the bar's close instead */
      }
    }
    try {
      logDetection(info, !cool);
    } catch (e) {
      this.d.log(`[rebound] 감지 기록 실패: ${(e as Error).message}`);
    }
    this.d.log(`[rebound] 감지 ${name}(${code}) 봉 ${hhmm(res.snap.t)} 점수 ${res.score}${cool ? ' - 쿨다운 중, 알림 생략' : ''}`);
    if (cool) return;
    this.lastAlert.set(code, now);
    this.st.alerts++;
    this.d.alert(info, this.cfg);
  }

  /** one pass over the tab; never throws */
  async tick(): Promise<void> {
    if (this.busy) {
      this.d.log('[rebound] 앞 순회가 아직 끝나지 않아 이번 분은 건너뜀');
      return;
    }
    this.busy = true;
    try {
      this.loadConfig();
      const cfg = this.cfg;
      if (!cfg.enabled) return this.setStatus({ enabled: false, tab: cfg.tab, state: 'off', note: undefined });
      const session = sessionOf(cfg, this.now());
      if (!session) return this.setStatus({ enabled: true, tab: cfg.tab, state: 'rest', note: undefined });
      const backend = this.d.backend();
      if (!backend) return this.setStatus({ enabled: true, tab: cfg.tab, state: 'login', note: undefined });
      this.loadTab();
      if (!this.codes) return this.setStatus({ enabled: true, tab: cfg.tab, state: 'notab', codes: 0, note: `관심종목에 "${cfg.tab}" 탭이 없습니다` });
      let judged = 0;
      let failed = 0;
      let firstErr = '';
      for (const code of this.codes) {
        if (this.halt || this.d.backend() !== backend) break; // logged out / switched
        // NXT pre / after market: symbols that do not trade on NXT have no new bars
        if (session === 'nxt' && this.d.master.nxtKnown && !this.d.master.get(code)?.nxt) continue;
        try {
          if (await this.one(backend, code)) judged++;
        } catch (e) {
          failed++;
          firstErr ||= `${code}: ${String((e as Error)?.message || e).split('\n')[0]}`;
        }
        if (cfg.poll.gapMs > 0) await sleep(cfg.poll.gapMs);
      }
      if (failed) this.d.log(`[rebound] 조회 실패 ${failed}종목 (${firstErr})`);
      this.setStatus({ enabled: true, tab: cfg.tab, state: 'run', codes: this.codes.length, lastPass: this.now(), judged, failed, note: failed ? `조회 실패 ${failed}종목 - ${firstErr}` : undefined });
    } catch (e) {
      this.d.log(`[rebound] 순회 오류: ${(e as Error).message}`);
    } finally {
      this.busy = false;
    }
  }
}

/** a sample detection for the "알람 테스트" menu (no market call) */
export function sampleAlert(cfg: ReboundConfig): AlertInfo {
  const ok = (note: string) => ({ ok: true, note });
  const t = Math.floor(Date.now() / 60000) * 60000 - 60000;
  return {
    code: '000000',
    name: '알람 테스트',
    at: Date.now(),
    price: 38500,
    changePct: -2.41,
    res: {
      hit: true,
      edge: false,
      required: true,
      score: 3,
      conds: { plunge: ok('14:33 저가 38000 거래량 2147'), rsi: ok('돌파 14:53, RSI 53.3/48.8'), macd: ok('돌파 14:46(MACD -84.5)'), dmi: ok('돌파 14:50, DI 32.2/12.6'), ma: ok('골든크로스 14:55'), angle: ok(`14.6° (기준 ${cfg.angle.min}°)`) },
      bonus: { adxTurn: ok('저점 14:51 33.9'), maLong: { ok: false, note: 'MA60 38530.0' }, bbSqueeze: ok('폭 0.89%'), ichimoku: { ok: false, note: '구름 아래' }, macdZero: { ok: false, note: 'MACD -40.3' }, volDry: ok('평균 147.0 / 급락 봉 2147') },
      snap: { t, close: 38500, ma5: 38450, ma20: 38445, ma60: 38530, ma120: 38800, rsi: 53.3, rsiSignal: 48.8, macd: -40.3, macdSignal: -61.7, macdOsc: 21.4, diPlus: 32.2, diMinus: 12.6, adx: 35.6, angle: 14.6, vwap: 39810, vwapPct: -3.29, bbMid: 38445, bbWidthPct: 0.89, plungeT: t - 22 * 60000, plungeLow: 38000, plungeVol: 2147, plungeFallPct: -2.7 },
    },
  };
}
