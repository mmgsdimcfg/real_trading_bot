// Update log
// - [2026-11-21] type=feat (Rebound alerts (user 2026-11-21): a small always-on-top window per detection, stacked from the bottom-right
//   corner of the primary screen (a new column to the left when a column is full), shown without taking the focus. The page is plain
//   HTML in a data: URL with scripts off; its two links ("열기" / "닫기") are caught in will-navigate, so the popup needs no preload.
//   Optional beep, optional auto-close, the oldest popup closes beyond alert.maxPopups. Every detection is appended to
//   %APPDATA%\<app>\rebound\detections_YYYYMMDD.csv (alerted or held back by the cooldown))

import { app, BrowserWindow, screen, shell } from 'electron';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { BONUS_KEYS, BONUS_LABEL, COND_KEYS, COND_LABEL, hhmm, type ReboundConfig, type ReboundResult } from '../shared/rebound';

export interface AlertInfo {
  code: string;
  name: string;
  /** epoch ms the detection was made */
  at: number;
  /** current price / change vs the previous close (%), when the quote call answered */
  price?: number;
  changePct?: number;
  res: ReboundResult;
}

export const reboundDir = () => join(app.getPath('userData'), 'rebound');
const KST = 9 * 3600000;
const kst = (t: number) => new Date(t + KST).toISOString();

const CSV_HEAD = [
  'detected_at', 'bar_time', 'code', 'name', 'alerted', 'score', 'conds', 'bonus', 'close', 'price', 'change_pct', 'rsi', 'rsi_signal', 'macd', 'macd_signal', 'macd_osc',
  'di_plus', 'di_minus', 'adx', 'ma5_angle', 'vwap', 'vwap_pct', 'ma5', 'ma20', 'ma60', 'ma120', 'plunge_time', 'plunge_low', 'plunge_volume', 'plunge_fall_pct',
];
const n = (x: number | undefined, d = 2) => (x === undefined || !Number.isFinite(x) ? '' : Number(x.toFixed(d)).toString());

export function csvRow(a: AlertInfo, alerted: boolean): string {
  const s = a.res.snap;
  const f = [
    kst(a.at).slice(0, 19).replace('T', ' '), hhmm(s.t), a.code, a.name.replace(/[",\r\n]/g, ' '), alerted ? '1' : '0', String(a.res.score),
    COND_KEYS.filter((k) => a.res.conds[k].ok).join(' '), BONUS_KEYS.filter((k) => a.res.bonus[k].ok).join(' '),
    n(s.close, 0), n(a.price, 0), n(a.changePct), n(s.rsi), n(s.rsiSignal), n(s.macd), n(s.macdSignal), n(s.macdOsc), n(s.diPlus), n(s.diMinus), n(s.adx), n(s.angle),
    n(s.vwap, 1), n(s.vwapPct), n(s.ma5, 1), n(s.ma20, 1), n(s.ma60, 1), n(s.ma120, 1), hhmm(s.plungeT), n(s.plungeLow, 0), n(s.plungeVol, 0), n(s.plungeFallPct),
  ];
  return f.join(',');
}

/** append one detection to today's CSV (created with a header) */
export function logDetection(a: AlertInfo, alerted: boolean): void {
  const dir = reboundDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const file = join(dir, `detections_${kst(a.at).slice(0, 10).replace(/-/g, '')}.csv`);
  const head = existsSync(file) ? '' : `﻿${CSV_HEAD.join(',')}\r\n`;
  appendFileSync(file, `${head}${csvRow(a, alerted)}\r\n`, 'utf8');
}

const esc = (s: unknown) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string);
const num = (x: number | undefined, d = 1) => (x === undefined || !Number.isFinite(x) ? '-' : x.toLocaleString('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d }));

/** the popup page (no script; "https://rebound.invalid/open/<code>" and "https://rebound.invalid/close" are handled by the main process) */
export function alertHtml(a: AlertInfo, minScore: number): string {
  const s = a.res.snap;
  const up = (a.changePct ?? 0) >= 0;
  const price = a.price ?? s.close;
  const li = (ok: boolean, label: string, note: string) => `<li class="${ok ? 'ok' : 'no'}" title="${esc(note)}"><b>${ok ? '✓' : '·'}</b> ${esc(label)}<i>${esc(note)}</i></li>`;
  const row = (k: string, v: string) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`;
  return `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>바닥 반등 초입 ${esc(a.name)}</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{font:12px/1.45 "Malgun Gothic","Segoe UI",sans-serif;background:#15171c;color:#d7dae0;border:1px solid #e0a23a;overflow:hidden;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:6px;padding:6px 8px;background:#2a2212;-webkit-app-region:drag}
header b{color:#ffcf73;font-size:13px}header span{color:#9aa0aa;margin-left:auto}
header a{-webkit-app-region:no-drag;color:#d7dae0;text-decoration:none;padding:0 6px;border:1px solid #555;border-radius:3px}
header a:hover{background:#3a3f4a}
main{padding:6px 8px;overflow:auto;flex:1}
main::-webkit-scrollbar{width:8px}main::-webkit-scrollbar-thumb{background:#3a3f4a;border-radius:4px}
h1{margin:0;font-size:15px}h1 small{color:#9aa0aa;font-weight:normal}
.px{font-size:14px;margin:2px 0 6px}.up{color:#ff5b5b}.dn{color:#4d8dff}
ul{list-style:none;margin:0 0 6px;padding:0}li{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
li b{display:inline-block;width:12px}li.ok b{color:#5fd38d}li.no{color:#737985}li i{font-style:normal;color:#8a909b;margin-left:6px;font-size:11px}
h2{margin:4px 0 2px;font-size:11px;color:#9aa0aa;font-weight:normal;border-bottom:1px solid #2c3038}
table{border-collapse:collapse;width:100%}th{text-align:left;color:#9aa0aa;font-weight:normal;padding:0 6px 0 0;white-space:nowrap}td{text-align:right}
footer{padding:3px 8px;color:#737985;font-size:11px;border-top:1px solid #2c3038}
</style></head><body>
<header><b>바닥 반등 초입</b><span>${esc(kst(a.at).slice(11, 19))} 감지 · 봉 ${esc(hhmm(s.t))}</span><a href="https://rebound.invalid/open/${esc(a.code)}" title="앱에서 이 종목 열기">열기</a><a href="https://rebound.invalid/close" title="닫기">✕</a></header>
<main>
<h1>${esc(a.name)} <small>(${esc(a.code)})</small></h1>
<div class="px ${up ? 'up' : 'dn'}">${num(price, 0)}${a.changePct === undefined ? '' : ` &nbsp;${up ? '+' : ''}${num(a.changePct, 2)}%`}</div>
<h2>필수 조건</h2><ul>${COND_KEYS.map((k) => li(a.res.conds[k].ok, COND_LABEL[k], a.res.conds[k].note)).join('')}</ul>
<h2>가점 ${a.res.score}점 (기준 ${minScore}점)</h2><ul>${BONUS_KEYS.map((k) => li(a.res.bonus[k].ok, BONUS_LABEL[k], a.res.bonus[k].note)).join('')}</ul>
<h2>지표</h2><table>
${row('RSI / 시그널', `${num(s.rsi)} / ${num(s.rsiSignal)}`)}
${row('MACD / 시그널', `${num(s.macd)} / ${num(s.macdSignal)}`)}
${row('DI+ / DI- / ADX', `${num(s.diPlus)} / ${num(s.diMinus)} / ${num(s.adx)}`)}
${row('MA5 각도', `${num(s.angle)}°`)}
${row('VWAP 대비', `${Number.isFinite(s.vwapPct) ? `${s.vwapPct >= 0 ? '+' : ''}${num(s.vwapPct, 2)}% (${s.vwapPct >= 0 ? '위' : '아래'}, VWAP ${num(s.vwap, 0)})` : '-'}`)}
</table></main>
<footer>감지·알림 전용 - 주문은 내지 않습니다</footer></body></html>`;
}

const W = 440;
const H = 490;
const GAP = 8;

export class AlertPopups {
  private wins: BrowserWindow[] = [];

  constructor(private onOpen: (code: string) => void) {}

  show(a: AlertInfo, cfg: ReboundConfig): void {
    // the oldest go first. Taken off the list here: close() only asks - 'closed' comes later, so waiting for it would loop forever
    while (this.wins.length >= Math.max(1, cfg.alert.maxPopups)) {
      const old = this.wins.shift();
      if (old && !old.isDestroyed()) old.destroy();
    }
    const w = new BrowserWindow({
      width: W,
      height: H,
      frame: false,
      resizable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      show: false,
      backgroundColor: '#15171c',
      title: `바닥 반등 초입 ${a.name}`,
      webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false, javascript: false },
    });
    w.setAlwaysOnTop(true, 'floating');
    w.setMenuBarVisibility(false);
    w.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    w.webContents.on('will-navigate', (e, url) => {
      e.preventDefault();
      const m = /^https:\/\/rebound\.invalid\/open\/([0-9A-Z]{6})$/.exec(url);
      if (m) this.onOpen(m[1]);
      else if (url.startsWith('https://rebound.invalid/close')) w.close();
    });
    w.on('closed', () => {
      this.wins = this.wins.filter((x) => x !== w);
      this.layout();
    });
    this.wins.push(w);
    this.layout();
    w.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(alertHtml(a, cfg.bonus.minScore))}`).then(
      () => !w.isDestroyed() && w.showInactive(),
      () => undefined, // closed while loading
    );
    if (cfg.alert.sound) shell.beep();
    if (cfg.alert.autoCloseSec > 0) setTimeout(() => !w.isDestroyed() && w.close(), cfg.alert.autoCloseSec * 1000);
  }

  /** newest at the bottom-right corner, older ones above it, then the next column to the left */
  private layout(): void {
    const a = screen.getPrimaryDisplay().workArea;
    const perCol = Math.max(1, Math.floor((a.height - GAP) / (H + GAP)));
    [...this.wins].reverse().forEach((w, k) => {
      if (w.isDestroyed()) return;
      const col = Math.floor(k / perCol);
      const rowI = k % perCol;
      w.setBounds({ x: Math.max(a.x, a.x + a.width - (col + 1) * (W + GAP)), y: a.y + a.height - (rowI + 1) * (H + GAP), width: W, height: H });
    });
  }

  closeAll(): void {
    for (const w of [...this.wins]) if (!w.isDestroyed()) w.close();
  }
}
