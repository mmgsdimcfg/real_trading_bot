// Update log
// - [2026-11-21] type=feat (Rebound alerts (user 2026-11-21): a small always-on-top window per detection, stacked from the bottom-right
//   corner of the primary screen (a new column to the left when a column is full), shown without taking the focus. The page is plain
//   HTML in a data: URL with scripts off; its two links ("열기" / "닫기") are caught in will-navigate, so the popup needs no preload.
//   Optional beep, optional auto-close, the oldest popup closes beyond alert.maxPopups. Every detection is appended to
//   %APPDATA%\<app>\rebound\detections_YYYYMMDD.csv (alerted or held back by the cooldown))
// - [2026-11-22] type=feat (show(a, cfg, id): a popup knows its history id; onSeen(id) is called when the user closes it (x) or
//   opens the symbol - a popup that closes by itself (alert.autoCloseSec) or is pushed out by newer ones is not "seen")
// - [2026-11-24] type=feat (lower highs check: popup section "추세 구조" (고점 낮아짐 / 아님 / 판정 불가 with H1 -> H2 -> current high and
//   the RSI divergence); CSV columns lower_high, lh_h1_time, lh_h1, lh_h2_time, lh_h2, lh_hc, lh_div, suppressed - a day file written
//   by an older version keeps its layout and the new rows go to detections_YYYYMMDD_2.csv)

import { app, BrowserWindow, screen, shell } from 'electron';
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readSync, statSync } from 'fs';
import { join } from 'path';
import { BONUS_KEYS, BONUS_LABEL, COND_KEYS, COND_LABEL, hhmm, lowerHighLabel, type ReboundConfig, type ReboundResult } from '../shared/rebound';

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
  // [2026-11-24] lower highs check: state lh / no / na, H1 / H2 / Hc, RSI divergence, why the alert was not sent ('' sent / cooldown)
  'lower_high', 'lh_h1_time', 'lh_h1', 'lh_h2_time', 'lh_h2', 'lh_hc', 'lh_div', 'suppressed',
];
const n = (x: number | undefined, d = 2) => (x === undefined || !Number.isFinite(x) ? '' : Number(x.toFixed(d)).toString());

export function csvRow(a: AlertInfo, alerted: boolean, suppressed = ''): string {
  const s = a.res.snap;
  const lh = a.res.lowerHigh;
  const f = [
    kst(a.at).slice(0, 19).replace('T', ' '), hhmm(s.t), a.code, a.name.replace(/[",\r\n]/g, ' '), alerted ? '1' : '0', String(a.res.score),
    COND_KEYS.filter((k) => a.res.conds[k].ok).join(' '), BONUS_KEYS.filter((k) => a.res.bonus[k].ok).join(' '),
    n(s.close, 0), n(a.price, 0), n(a.changePct), n(s.rsi), n(s.rsiSignal), n(s.macd), n(s.macdSignal), n(s.macdOsc), n(s.diPlus), n(s.diMinus), n(s.adx), n(s.angle),
    n(s.vwap, 1), n(s.vwapPct), n(s.ma5, 1), n(s.ma20, 1), n(s.ma60, 1), n(s.ma120, 1), hhmm(s.plungeT), n(s.plungeLow, 0), n(s.plungeVol, 0), n(s.plungeFallPct),
    lh?.state ?? '', lh?.h1t === undefined ? '' : hhmm(lh.h1t), n(lh?.h1, 0), lh?.h2t === undefined ? '' : hhmm(lh.h2t), n(lh?.h2, 0), n(lh?.hc, 0),
    lh ? (lh.div ? '1' : '0') : '', suppressed,
  ];
  return f.join(',');
}

/** the header line of a CSV file, from its first 4 KB only ('' when empty or unreadable) */
function headOf(file: string): string {
  let fd = -1;
  try {
    fd = openSync(file, 'r');
    const buf = Buffer.alloc(4096);
    const n = readSync(fd, buf, 0, buf.length, 0);
    return buf.subarray(0, n).toString('utf8').split(/\r?\n/, 1)[0].replace(/^﻿/, '');
  } catch {
    return '';
  } finally {
    if (fd >= 0) closeSync(fd);
  }
}

/**
 * Append one detection to today's CSV (created with a header). [2026-11-24] A file of the day in another layout (written by an older
 * version) is left as it is: the rows go to detections_YYYYMMDD_2.csv (then _3, ...) - the first one that does not exist, is empty or
 * has this header - so no file mixes two layouts.
 */
export function logDetection(a: AlertInfo, alerted: boolean, suppressed = ''): void {
  const dir = reboundDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const ymd = kst(a.at).slice(0, 10).replace(/-/g, '');
  const want = CSV_HEAD.join(',');
  const empty = (f: string) => !existsSync(f) || statSync(f).size === 0;
  const fits = (f: string) => empty(f) || headOf(f) === want;
  let file = join(dir, `detections_${ymd}.csv`);
  for (let k = 2; k < 100 && !fits(file); k++) file = join(dir, `detections_${ymd}_${k}.csv`);
  if (!fits(file)) throw new Error(`detections_${ymd} ~ _99.csv 가 모두 다른 형식입니다`); // (Codex) the last one is checked too
  const head = empty(file) ? `﻿${want}\r\n` : '';
  appendFileSync(file, `${head}${csvRow(a, alerted, suppressed)}\r\n`, 'utf8');
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
li b{display:inline-block;width:12px}li.ok b{color:#5fd38d}li.warn{color:#ffb347}li.warn b{color:#ffb347}li.wrap{white-space:normal}li.wrap i{display:block;margin-left:12px}li.no{color:#737985}li i{font-style:normal;color:#8a909b;margin-left:6px;font-size:11px}
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
${a.res.lowerHigh ? `<h2>추세 구조</h2><ul><li class="wrap ${a.res.lowerHigh.state === 'lh' ? 'warn' : a.res.lowerHigh.state === 'no' ? 'ok' : 'no'}" title="${esc(a.res.lowerHigh.note)}"><b>${a.res.lowerHigh.state === 'lh' ? '!' : a.res.lowerHigh.state === 'no' ? '✓' : '·'}</b> ${esc(lowerHighLabel(a.res.lowerHigh))}<i>${esc(a.res.lowerHigh.note)}</i></li></ul>` : ''}
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
const H = 490; // kept with the "추세 구조" section (2026-11-24): taller would put only one popup per column on a 1080p screen - the body scrolls
const GAP = 8;

export class AlertPopups {
  private wins: BrowserWindow[] = [];

  constructor(
    private onOpen: (code: string) => void,
    /** the user looked at the alert of this history id: closed its popup by hand or opened the symbol */
    private onSeen: (id: string) => void = () => undefined,
  ) {}

  show(a: AlertInfo, cfg: ReboundConfig, id = ''): void {
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
      else if (!url.startsWith('https://rebound.invalid/close')) return;
      if (id) this.onSeen(id);
      if (!m) w.close();
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
