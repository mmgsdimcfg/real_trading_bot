/*
 * Update log
 * - [2026-11-06] type=feat - Chart tab (HTS-style): any symbol (watched symbols in a list, or any 6-digit code), tick / minute /
 *   daily / weekly / monthly bars with zoom (wheel) and pan (drag, double-click = back to the newest bars), the previous trading day
 *   shown before intraday bars, every indicator (MA, Bollinger, Ichimoku, VWAP / RSI, MACD, ADX, stochastic, Williams %R, MFI, OBV)
 *   with legends on a right price axis. Side panel: current price with change vs the previous close, day OHLC / volume, best ask/bid
 *   and 체결강도 from the tick file, 200-day high/low, the hovered bar's data window, an indicator signal summary, the bot's state
 *   for the symbol, and the latest trades. Reloads with every snapshot while the day is live. Read-only (SFTP via price:get)
 * - [2026-11-06] type=feat - Any code, not only watched symbols: bars come from the server's r016 `bars` when there is no file (daily
 *   history, KIS once the server adds it) and the side panel uses r016 `quote` when the server has it (KIS current price, change, day
 *   OHLC, 10-level order book, latest trades, 체결강도, upper/lower limit, 52-week high/low, PER/PBR, market cap, VI / halt). Today's
 *   daily/weekly/monthly candle is completed from the current price when the daily data has no row for today yet
 * - [2026-11-07] type=perf - Periodic refreshes are skipped while the window is hidden/minimized (server load rule); symbol or kind
 *   changes still load; a missing server KIS token (r016 quote token_not_ready) is shown as a plain notice
 * - [2026-11-07] type=feat - Past-day replay: tick/minute bars of the viewed day that start after the replay time are hidden
 *   (barsUpTo); trade markers come from the replay snapshot. Daily / weekly / monthly bars are not cut
 * - [2026-11-11] type=fix - Symbol list: a <select> with every watched symbol (+ recent codes) left of the code input, showing the
 *   current symbol (like the timeline picker) - the datalist only shows entries matching the typed text, so with a code in the box
 *   it listed just that one symbol
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../store';
import { PriceChart, IndicatorLines } from '../components/PriceChart';
import { IndicatorToggles } from '../components/IndicatorToggles';
import { MIN_SIZES, TICK_SIZES, monthBars, weekBars, ymdEpoch, type Bar, type ChartKind, type MarketQuote, type PriceResponse } from '../../../shared/price';
import { computeIndicators, OSCILLATORS, type IndicatorKey, type IndicatorSet } from '../../../shared/indicators';
import { condTitle, fmtInt } from '../lib';
import { barsUpTo } from '../../../shared/replay';
import { fmtKstDateTime, fmtKstTime } from '../../../shared/time';

const KINDS: [ChartKind, string][] = [
  ['tick', '틱'],
  ['min', '분'],
  ['day', '일'],
  ['week', '주'],
  ['month', '월'],
];
const SUMMARY_KEYS: IndicatorKey[] = ['ma', 'bb', 'ichimoku', 'rsi', 'macd', 'adx', 'stoch', 'mfi'];
const RECENT_KEY = 'autoboth.chartRecent';
const NONE: Bar[] = [];

const loadRecent = (): string[] => {
  try {
    const v = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
    return Array.isArray(v) ? v.filter((x) => typeof x === 'string').slice(0, 12) : [];
  } catch {
    return [];
  }
};
const saveRecent = (v: string[]) => {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(v.slice(0, 12)));
  } catch {
    /* storage unavailable */
  }
};

const ok = (v: number | undefined): v is number => v !== undefined && v === v && Number.isFinite(v);
const sign = (v: number) => (v > 0 ? '▲' : v < 0 ? '▼' : '');
const cls = (v: number | undefined) => (!ok(v) || v === 0 ? '' : v > 0 ? 'gain' : 'loss');
const pct = (a: number, b: number) => ((a - b) / b) * 100;

interface Signal {
  name: string;
  text: string;
  tone: 'up' | 'down' | 'flat';
}

/** plain-language state of each indicator at bar i */
function signals(S: IndicatorSet, bars: Bar[], i: number): Signal[] {
  const out: Signal[] = [];
  const c = bars[i]?.c;
  if (!ok(c)) return out;
  const v = (s: number[] | undefined, k = i) => s?.[k];
  if (S.ma) {
    const [m5, m20, m60, m120] = [5, 20, 60, 120].map((p) => v(S.ma![p]));
    if ([m5, m20, m60].every(ok)) {
      const up = m5! > m20! && m20! > m60! && (!ok(m120) || m60! > m120!);
      const dn = m5! < m20! && m20! < m60! && (!ok(m120) || m60! < m120!);
      out.push({ name: '이동평균', text: up ? '정배열' : dn ? '역배열' : `혼조 (종가 ${c >= m20! ? 'MA20 위' : 'MA20 아래'})`, tone: up ? 'up' : dn ? 'down' : 'flat' });
    }
  }
  if (S.bb) {
    const u = v(S.bb.upper);
    const l = v(S.bb.lower);
    if (ok(u) && ok(l) && u > l) {
      const b = ((c - l) / (u - l)) * 100;
      out.push({ name: '볼린저 %B', text: `${b.toFixed(0)}% ${b >= 100 ? '(상단 돌파)' : b <= 0 ? '(하단 이탈)' : b >= 80 ? '(상단 근접)' : b <= 20 ? '(하단 근접)' : ''}`, tone: b >= 80 ? 'up' : b <= 20 ? 'down' : 'flat' });
    }
  }
  if (S.ichimoku) {
    const a = v(S.ichimoku.senkouA);
    const b = v(S.ichimoku.senkouB);
    if (ok(a) && ok(b)) {
      const top = Math.max(a, b);
      const bot = Math.min(a, b);
      out.push({ name: '일목 구름', text: c > top ? '구름 위' : c < bot ? '구름 아래' : '구름 안', tone: c > top ? 'up' : c < bot ? 'down' : 'flat' });
    }
  }
  if (S.rsi) {
    const r = v(S.rsi.rsi);
    if (ok(r)) out.push({ name: 'RSI', text: `${r.toFixed(1)} ${r >= 70 ? '과매수' : r <= 30 ? '과매도' : ''}`, tone: r >= 70 ? 'up' : r <= 30 ? 'down' : 'flat' });
  }
  if (S.macd) {
    const m = v(S.macd.macd);
    const s = v(S.macd.signal);
    const pm = v(S.macd.macd, i - 1);
    const ps = v(S.macd.signal, i - 1);
    if (ok(m) && ok(s)) {
      const cross = ok(pm) && ok(ps) ? (pm <= ps && m > s ? ' · 골든크로스' : pm >= ps && m < s ? ' · 데드크로스' : '') : '';
      out.push({ name: 'MACD', text: `${m > s ? '시그널 위' : '시그널 아래'}${cross}`, tone: m > s ? 'up' : 'down' });
    }
  }
  if (S.adx) {
    const a = v(S.adx.adx);
    const p = v(S.adx.diPlus);
    const m = v(S.adx.diMinus);
    if (ok(a) && ok(p) && ok(m)) out.push({ name: 'ADX', text: `${a.toFixed(1)} ${a >= 25 ? '추세' : '비추세'} · ${p > m ? 'DI+ 우위' : 'DI- 우위'}`, tone: a >= 25 ? (p > m ? 'up' : 'down') : 'flat' });
  }
  if (S.stoch) {
    const k = v(S.stoch.k);
    if (ok(k)) out.push({ name: '스토캐스틱', text: `%K ${k.toFixed(1)} ${k >= 80 ? '과열' : k <= 20 ? '침체' : ''}`, tone: k >= 80 ? 'up' : k <= 20 ? 'down' : 'flat' });
  }
  if (S.mfi) {
    const m = v(S.mfi);
    if (ok(m)) out.push({ name: 'MFI', text: `${m.toFixed(1)} ${m >= 80 ? '과매수' : m <= 20 ? '과매도' : ''}`, tone: m >= 80 ? 'up' : m <= 20 ? 'down' : 'flat' });
  }
  return out;
}

export function ChartView() {
  const snap = useStore((s) => s.snapshot);
  const status = useStore((s) => s.status);
  const conditions = useStore((s) => s.conditions);
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  const timeline = useStore((s) => s.timeline);
  const prefs = useStore((s) => s.chartTab);
  const setPrefs = useStore((s) => s.setChartTab);

  const [code, setCode] = useState<string>(selected || '');
  const [input, setInput] = useState(selected || '');
  const [recent, setRecent] = useState<string[]>(loadRecent);
  const [data, setData] = useState<PriceResponse | null>(null);
  const [daily, setDaily] = useState<PriceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [quoteRes, setQuoteRes] = useState<{ q: MarketQuote | null; unsupported?: boolean; error?: string } | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [chartKey, setChartKey] = useState(0);
  const seq = useRef(0);
  const areaRef = useRef<HTMLDivElement>(null);
  const [areaH, setAreaH] = useState(600);

  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setAreaH(el.clientHeight));
    ro.observe(el);
    setAreaH(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // default symbol: the one selected elsewhere, else the first watched symbol
  useEffect(() => {
    if (!code && (selected || snap?.symbols[0]?.code)) {
      const c = selected || snap!.symbols[0].code;
      setCode(c);
      setInput(c);
    }
  }, [selected, snap, code]);

  const day = snap?.date || status.date;
  const n = prefs.kind === 'tick' ? prefs.tickN : prefs.minN;
  const intraday = prefs.kind === 'tick' || prefs.kind === 'min';
  const live = status.live ? snap?.generatedAt : 0;
  const key = `${day}/${code}/${prefs.kind}/${n}`;
  const lastKey = useRef('');

  useEffect(() => {
    if (!code || !day || !/^[0-9A-Z]{6}$/.test(code)) return;
    const changed = lastKey.current !== key;
    // periodic refresh (new log poll) while the window is minimized/hidden: skip - every refresh starts Python on the server
    // (bars / quote, maybe KIS calls). A symbol/kind change still loads; the next poll after the window is shown refreshes.
    if (!changed && document.hidden) return;
    const my = ++seq.current;
    lastKey.current = key;
    if (changed) {
      setLoading(true);
      setHover(null);
      setChartKey((k) => k + 1);
    }
    if (changed) setQuoteRes(null);
    void Promise.all([
      window.api.getPrice({ date: day, code, kind: prefs.kind, n, warmup: intraday, withName: true }),
      prefs.kind === 'day' ? null : window.api.getPrice({ date: day, code, kind: 'day', n: 1 }),
      window.api.getQuote(code),
    ]).then(([r, d, q]) => {
      if (my !== seq.current) return;
      if (q.ok && q.data) setQuoteRes({ q: q.data });
      else if (q.unsupported) setQuoteRes({ q: null, unsupported: true });
      else if (changed || !quoteRes?.q) setQuoteRes({ q: null, error: q.error });
      setLoading(false);
      if (r.ok || changed) setData(r);
      if (prefs.kind === 'day') setDaily(r);
      else if (d && (d.ok || changed)) setDaily(d);
    });
  }, [key, live, status.conn]); // eslint-disable-line react-hooks/exhaustive-deps

  // shown bars: intraday = previous day + the day (HTS style), else as returned
  const prevDay = intraday && prefs.prevDay && data?.ok ? data.warm || NONE : NONE;
  const isToday = !!day && day === status.serverDate;
  const q = quoteRes?.q && quoteRes.q.code === code ? quoteRes.q : null;
  const dayT0 = day ? ymdEpoch(day) : 0;
  /** today's daily candle from the KIS quote when the daily data has no row for today yet */
  const quoteBar: Bar | null =
    isToday && q && q.open && q.high && q.low && q.price ? { t: dayT0, o: q.open, h: q.high, l: q.low, c: q.price, v: q.volume || 0 } : null;
  const replayCursor = snap?.replay?.cursor ?? null;
  const bars = useMemo<Bar[]>(() => {
    if (!data?.ok) return [];
    if (!intraday && quoteBar) {
      // daily bars of the side request (or the chart itself on 'day') + today's candle, regrouped for week / month
      const src = prefs.kind === 'day' ? data.bars : daily?.ok ? daily.bars : null;
      if (src && !src.some((b) => b.t === dayT0)) {
        const plus = [...src.filter((b) => b.t < dayT0), quoteBar];
        return prefs.kind === 'day' ? plus : prefs.kind === 'week' ? weekBars(plus) : monthBars(plus);
      }
    }
    return [...prevDay, ...barsUpTo(data.bars, replayCursor, intraday)];
  }, [data, daily, prefs.prevDay, prefs.kind, quoteBar?.c, quoteBar?.h, quoteBar?.l, quoteBar?.v, replayCursor]); // eslint-disable-line react-hooks/exhaustive-deps
  const warm = prevDay.length ? NONE : data?.warm;
  const warmBot = prevDay.length ? NONE : data?.warmBot;

  // indicators for the side panel (summary keys + the chart's)
  const allKeys = useMemo(() => [...new Set([...SUMMARY_KEYS, ...prefs.ind])].filter((k) => k !== 'vwap' || intraday), [prefs.ind, intraday]);
  const S = useMemo(() => computeIndicators(bars, allKeys, { warm, warmBot }, undefined, intraday), [bars, allKeys, warm, warmBot, intraday]);
  const shownS = useMemo(() => Object.fromEntries(Object.entries(S).filter(([k]) => prefs.ind.includes(k as IndicatorKey))) as IndicatorSet, [S, prefs.ind]);

  // header numbers from the daily bars (day bar = viewed day, previous close = the bar before it)
  const dayT = day ? ymdEpoch(day) : 0;
  const dBars = daily?.ok ? daily.bars : [];
  const di = dBars.findIndex((b) => b.t === dayT);
  const dayBar = di >= 0 ? dBars[di] : undefined;
  const prevClose = di > 0 ? dBars[di - 1].c : dBars.length && dBars[dBars.length - 1].t < dayT ? dBars[dBars.length - 1].c : undefined;
  const quote = data?.ok ? data.quote : undefined;
  const lastBar = bars[bars.length - 1];
  const useQ = isToday && q ? q : null;
  const price = useQ?.price ?? quote?.price ?? (intraday ? lastBar?.c : dayBar?.c) ?? lastBar?.c;
  const prevC = useQ?.prev_close ?? prevClose;
  const chg = useQ?.change ?? (ok(price) && ok(prevC) ? price - prevC : undefined);
  const chgPct = useQ?.change_pct ?? (ok(price) && ok(prevC) ? pct(price, prevC) : undefined);
  const dOpen = useQ?.open ?? dayBar?.o;
  const dHigh = useQ?.high ?? dayBar?.h;
  const dLow = useQ?.low ?? dayBar?.l;
  const dVol = useQ?.volume ?? quote?.volume ?? dayBar?.v;
  const hi200 = dBars.length ? Math.max(...dBars.map((b) => b.h)) : undefined;
  const lo200 = dBars.length ? Math.min(...dBars.map((b) => b.l)) : undefined;
  const strength = q?.strength ?? (quote && quote.sellVol > 0 ? (quote.buyVol / quote.sellVol) * 100 : undefined);
  const ask1 = q?.orderbook?.asks?.[0]?.[0] ?? quote?.ask;
  const bid1 = q?.orderbook?.bids?.[0]?.[0] ?? quote?.bid;

  const sym = snap?.symbols.find((s) => s.code === code);
  const name = sym?.name || q?.name || data?.name || '';
  const trades = timeline && timeline.code === code ? timeline.trades : [];
  const hi = hover ?? bars.length - 1;
  const hb = bars[hi];
  const sigs = useMemo(() => signals(S, bars, bars.length - 1), [S, bars]);

  const nOsc = OSCILLATORS.filter((k) => prefs.ind.includes(k)).length;
  const oscH = Math.max(60, Math.min(110, Math.floor((areaH * 0.45) / Math.max(1, nOsc)) - 8));

  const choose = (raw: string) => {
    const m = /^\s*([0-9A-Z]{6})/i.exec(raw) || /\(([0-9A-Z]{6})\)/.exec(raw);
    const byName = !m && snap?.symbols.find((s) => s.name && raw.trim() && s.name.includes(raw.trim()));
    const c = m ? m[1].toUpperCase() : byName ? byName.code : '';
    if (!c) return;
    setCode(c);
    setInput(c);
    const r = [c, ...recent.filter((x) => x !== c)].slice(0, 12);
    setRecent(r);
    saveRecent(r);
    if (snap?.symbols.some((s) => s.code === c)) select(c); // loads the day's trades for the markers
  };

  const symbolOptions = useMemo(() => {
    const seen = new Set<string>();
    const out: { code: string; label: string }[] = [];
    for (const s of snap?.symbols || []) if (!seen.has(s.code)) (seen.add(s.code), out.push({ code: s.code, label: `${s.code} ${s.name}${s.position ? ' (보유)' : ''}` }));
    for (const c of recent) if (!seen.has(c)) (seen.add(c), out.push({ code: c, label: `${c} (최근)` }));
    return out;
  }, [snap, recent]);

  return (
    <div className="chartview">
      <div className="toolbar">
        {/* today's watched symbols (+ recent codes) like the timeline picker; it shows the current symbol when it is one of them.
            The code input next to it takes any 6-digit code (the datalist only offers entries matching the typed text) */}
        <select className="picker sym-select" value={symbolOptions.some((o) => o.code === code) ? code : ''} title="감시 종목 / 최근 본 종목" onChange={(e) => e.target.value && choose(e.target.value)}>
          <option value="">감시 종목 선택…</option>
          {symbolOptions.map((o) => (
            <option key={o.code} value={o.code}>
              {o.label}
            </option>
          ))}
        </select>
        <input
          className="sym-input"
          list="chart-symbols"
          value={input}
          placeholder="종목 코드 또는 이름"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && choose(input)}
          onBlur={() => input !== code && choose(input)}
        />
        <datalist id="chart-symbols">
          {symbolOptions.map((o) => (
            <option key={o.code} value={o.code}>
              {o.label}
            </option>
          ))}
        </datalist>

        <span className="sym-name">{name}</span>
        <div className="seg">
          {KINDS.map(([k, label]) => (
            <button key={k} className={prefs.kind === k ? 'on' : ''} onClick={() => setPrefs({ kind: k })}>
              {label}
            </button>
          ))}
        </div>
        {prefs.kind === 'tick' && (
          <select value={prefs.tickN} onChange={(e) => setPrefs({ tickN: Number(e.target.value) })}>
            {TICK_SIZES.map((v) => (
              <option key={v} value={v}>
                {v}틱
              </option>
            ))}
          </select>
        )}
        {prefs.kind === 'min' && (
          <select value={prefs.minN} onChange={(e) => setPrefs({ minN: Number(e.target.value) })}>
            {MIN_SIZES.map((v) => (
              <option key={v} value={v}>
                {v}분
              </option>
            ))}
          </select>
        )}
        {intraday && (
          <label className="check" title="분봉·틱봉 앞에 전일 봉을 이어 붙여 보여 줍니다 (HTS처럼). 끄면 당일만 보이고 지표는 봇과 같은 예열(전일 마지막 180분)로 계산합니다.">
            <input type="checkbox" checked={prefs.prevDay} onChange={(e) => setPrefs({ prevDay: e.target.checked })} /> 전일 포함
          </label>
        )}
        <button className="mini" onClick={() => setChartKey((k) => k + 1)} title="확대·이동을 처음 상태로 (차트 더블클릭과 같음)">
          화면 초기화
        </button>
        <span className="muted small">
          {loading ? '불러오는 중…' : data?.ok ? `${bars.length.toLocaleString()}봉${data.note ? ` · ${data.note}` : ''} · 휠 확대/축소 · 끌어서 이동` : ''}
        </span>
      </div>
      <div className="toolbar ind-bar">
        <IndicatorToggles value={prefs.ind} onChange={(ind) => setPrefs({ ind })} kind={prefs.kind} />
      </div>
      <div className="chartview-body">
        <div className="chart-area" ref={areaRef}>
          {!code && <div className="empty">종목 코드를 입력하세요.</div>}
          {data && !data.ok && <div className="empty">{data.error}</div>}
          {data?.ok && bars.length > 0 && (
            <PriceChart
              key={`${chartKey}/${key}`}
              drawKey={code}
              bars={bars}
              warm={warm}
              warmBot={warmBot}
              kind={prefs.kind}
              n={n}
              indicators={prefs.ind.filter((k) => k !== 'vwap' || intraday)}
              trades={trades}
              buyPrice={sym?.position?.buyPrice}
              date={day}
              mode="index"
              axis="right"
              left={8}
              right={72}
              height={Math.max(300, areaH - 4)}
              oscHeight={oscH}
              interactive
              tooltip={false}
              onHover={setHover}
            />
          )}
        </div>
        <aside className="chart-side">
          <section className="quote">
            <div className="q-name">
              <b>{name || code}</b> <span className="muted mono">{code}</span>
            </div>
            <div className={`q-price ${cls(chg)}`}>
              {fmtInt(price)}
              {ok(chg) && (
                <span className="q-chg">
                  {sign(chg)} {fmtInt(Math.abs(chg))} ({chg >= 0 ? '+' : ''}
                  {ok(chgPct) ? chgPct.toFixed(2) : '-'}%)
                </span>
              )}
            </div>
            <div className="muted small">
              {useQ
                ? `${useQ.asof ? useQ.asof.slice(11, 19) : ''} KIS 현재가`
                : quote
                  ? `${fmtKstTime(quote.t)} 체결 기준`
                  : dayBar
                    ? `${day.slice(4, 6)}-${day.slice(6)} 일봉 기준`
                    : ''}{' '}
              · 서버 KST
              {q?.vi && <b className="q-flag">VI</b>}
              {q?.halted && <b className="q-flag">거래정지</b>}
            </div>
            <div className="q-grid">
              <span>전일 종가</span>
              <b>{fmtInt(prevC)}</b>
              <span>시가</span>
              <b className={cls(ok(dOpen) && ok(prevC) ? dOpen - prevC : undefined)}>{fmtInt(dOpen)}</b>
              <span>고가</span>
              <b className={cls(ok(dHigh) && ok(prevC) ? dHigh - prevC : undefined)}>{fmtInt(dHigh)}</b>
              <span>저가</span>
              <b className={cls(ok(dLow) && ok(prevC) ? dLow - prevC : undefined)}>{fmtInt(dLow)}</b>
              <span>거래량</span>
              <b>{fmtInt(dVol)}</b>
              {useQ?.amount ? (
                <>
                  <span>거래대금</span>
                  <b>{fmtInt(useQ.amount / 1e8)}억</b>
                </>
              ) : null}
              {(ask1 || bid1) && (
                <>
                  <span>매도 1호가</span>
                  <b className="loss">{fmtInt(ask1)}</b>
                  <span>매수 1호가</span>
                  <b className="gain">{fmtInt(bid1)}</b>
                </>
              )}
              {ok(strength) && (
                <>
                  <span title={q?.strength !== undefined ? 'KIS 체결강도' : '매수 체결량 / 매도 체결량 × 100 (틱 파일의 체결 구분 기준)'}>체결강도</span>
                  <b className={strength >= 100 ? 'gain' : 'loss'}>{strength.toFixed(1)}%</b>
                </>
              )}
              {q?.upper_limit ? (
                <>
                  <span>상한가 / 하한가</span>
                  <b>
                    <span className="gain">{fmtInt(q.upper_limit)}</span> / <span className="loss">{fmtInt(q.lower_limit)}</span>
                  </b>
                </>
              ) : null}
              {q?.w52_high ? (
                <>
                  <span title={q.w52_high_date || ''}>52주 최고</span>
                  <b>
                    {fmtInt(q.w52_high)}
                    {ok(price) ? <small className="muted"> ({pct(price, q.w52_high).toFixed(1)}%)</small> : null}
                  </b>
                  <span title={q.w52_low_date || ''}>52주 최저</span>
                  <b>
                    {fmtInt(q.w52_low)}
                    {ok(price) && q.w52_low ? <small className="muted"> (+{pct(price, q.w52_low).toFixed(1)}%)</small> : null}
                  </b>
                </>
              ) : (
                <>
                  <span title="서버 일봉 데이터 기간 기준">{dBars.length}일 최고</span>
                  <b>
                    {fmtInt(hi200)}
                    {ok(hi200) && ok(price) ? <small className="muted"> ({pct(price, hi200).toFixed(1)}%)</small> : null}
                  </b>
                  <span>{dBars.length}일 최저</span>
                  <b>
                    {fmtInt(lo200)}
                    {ok(lo200) && ok(price) ? <small className="muted"> (+{pct(price, lo200).toFixed(1)}%)</small> : null}
                  </b>
                </>
              )}
              {q?.per !== undefined && (
                <>
                  <span>PER / PBR</span>
                  <b>
                    {q.per ?? '-'} / {q.pbr ?? '-'}
                  </b>
                </>
              )}
              {q?.market_cap_eok ? (
                <>
                  <span>시가총액</span>
                  <b>{q.market_cap_eok >= 10000 ? `${(q.market_cap_eok / 10000).toFixed(1)}조` : `${fmtInt(q.market_cap_eok)}억`}</b>
                </>
              ) : null}
            </div>
            {quoteRes?.unsupported && (
              <p className="muted small q-note">
                현재가·호가·체결: 서버에 <code>quote</code> 명령이 추가되면 표시됩니다 (KIS 조회 요청서 docs/server_kis_market_data_request_20261106.md).
              </p>
            )}
            {quoteRes?.error && (
              <p className="muted small q-note">
                {/token file|token_not_ready/.test(quoteRes.error)
                  ? '현재가: 서버의 KIS 토큰이 아직 없습니다 (봇 r003이 시작되면 생깁니다).'
                  : `현재가 조회 실패: ${quoteRes.error}`}
              </p>
            )}
          </section>

          {q?.orderbook && (q.orderbook.asks.length > 0 || q.orderbook.bids.length > 0) && (
            <section className="orderbook">
              <h4>
                호가 {q.orderbook.asof ? <span className="muted small">{q.orderbook.asof}</span> : null}
              </h4>
              <OrderBook ob={q.orderbook} price={q.price} prevClose={prevC} />
            </section>
          )}

          <section>
            <h4>데이터 창 {hover !== null ? '' : '(마지막 봉)'}</h4>
            {hb ? (
              <>
                <div className="mono small">
                  {intraday ? `${fmtKstDateTime(hb.t).slice(5, 16)}${hb.t2 && hb.t2 !== hb.t ? ` ~ ${fmtKstTime(hb.t2)}` : ''}` : fmtKstDateTime(hb.t).slice(0, 10)}
                </div>
                <div className="pc-ohlc">
                  <span>시 {fmtInt(hb.o)}</span>
                  <span>고 {fmtInt(hb.h)}</span>
                  <span>저 {fmtInt(hb.l)}</span>
                  <span>종 {fmtInt(hb.c)}</span>
                  <span>량 {fmtInt(hb.v)}</span>
                </div>
                <IndicatorLines S={shownS} i={hi} />
              </>
            ) : (
              <p className="muted small">-</p>
            )}
          </section>

          <section>
            <h4>지표 신호 ({KINDS.find((k) => k[0] === prefs.kind)?.[1]}봉 마지막)</h4>
            {sigs.length ? (
              <table className="sig">
                <tbody>
                  {sigs.map((s) => (
                    <tr key={s.name}>
                      <td>{s.name}</td>
                      <td className={s.tone === 'up' ? 'gain' : s.tone === 'down' ? 'loss' : ''}>{s.text}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted small">봉이 부족합니다.</p>
            )}
          </section>

          {sym && (
            <section>
              <h4>봇 상태 (오늘)</h4>
              <div className="q-grid">
                <span>감시</span>
                <b>{sym.watch}</b>
                <span>마지막 평가</span>
                <b>{sym.lastEval ? `${fmtKstTime(sym.lastEval.ts)} ${sym.lastEval.condKey === 'ORDER' ? '주문' : condTitle(conditions, 'buy', sym.lastEval.condKey)}` : '-'}</b>
                <span>보유</span>
                <b>{sym.position ? `${sym.position.qty}주 @ ${fmtInt(sym.position.buyPrice)}` : '-'}</b>
              </div>
            </section>
          )}

          {q?.trades?.length ? (
            <section className="ticks">
              <h4>체결 내역 (KIS, 최근 {q.trades.length}건)</h4>
              <table className="grid compact">
                <tbody>
                  {q.trades.map(([t, p, v, side], j) => (
                    <tr key={j} className="nohover">
                      <td className="mono">{t}</td>
                      <td className={`num ${ok(prevC) ? cls(p - prevC) : ''}`}>{fmtInt(p)}</td>
                      <td className={`num ${side === 'buy' ? 'gain' : side === 'sell' ? 'loss' : ''}`}>{fmtInt(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : quote ? (
            <section className="ticks">
              <h4>체결 내역 (최근 {quote.recent.length}건)</h4>
              <table className="grid compact">
                <tbody>
                  {quote.recent.map((k, j) => (
                    <tr key={j} className="nohover">
                      <td className="mono">{fmtKstTime(k.t)}</td>
                      <td className={`num ${ok(prevC) ? cls(k.p - prevC) : ''}`}>{fmtInt(k.p)}</td>
                      <td className={`num ${k.s === 1 ? 'gain' : k.s === 5 ? 'loss' : ''}`}>{fmtInt(k.v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}
        </aside>
      </div>
    </div>
  );
}

/** 10-level order book: asks (highest at the top) above bids, quantity bars, the current price row marked */
function OrderBook({ ob, price, prevClose }: { ob: NonNullable<MarketQuote['orderbook']>; price: number; prevClose?: number }) {
  const asks = [...ob.asks].slice(0, 10).reverse();
  const bids = ob.bids.slice(0, 10);
  const max = Math.max(1, ...ob.asks.map((a) => a[1]), ...ob.bids.map((b) => b[1]));
  const pc = (p: number) => (prevClose ? ((p - prevClose) / prevClose) * 100 : undefined);
  const row = (p: number, qty: number, side: 'ask' | 'bid', key: string) => {
    const d = pc(p);
    return (
      <tr key={key} className={`nohover ${p === price ? 'ob-cur' : ''}`}>
        <td className="ob-qty ob-ask">{side === 'ask' ? <><i style={{ width: `${(qty / max) * 100}%` }} /><span>{fmtInt(qty)}</span></> : null}</td>
        <td className={`ob-price ${d === undefined ? '' : d > 0 ? 'gain' : d < 0 ? 'loss' : ''}`}>
          {fmtInt(p)}
          {d !== undefined && <small> {d > 0 ? '+' : ''}{d.toFixed(2)}%</small>}
        </td>
        <td className="ob-qty ob-bid">{side === 'bid' ? <><i style={{ width: `${(qty / max) * 100}%` }} /><span>{fmtInt(qty)}</span></> : null}</td>
      </tr>
    );
  };
  return (
    <table className="ob">
      <tbody>
        {asks.map(([p, qty], i) => row(p, qty, 'ask', `a${i}`))}
        {bids.map(([p, qty], i) => row(p, qty, 'bid', `b${i}`))}
        {(ob.total_ask !== undefined || ob.total_bid !== undefined) && (
          <tr className="nohover ob-total">
            <td className="num loss">{fmtInt(ob.total_ask)}</td>
            <td className="center muted">총잔량</td>
            <td className="gain">{fmtInt(ob.total_bid)}</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
