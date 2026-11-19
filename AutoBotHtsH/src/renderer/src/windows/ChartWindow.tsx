// Update log
// - [2026-11-09] type=fix (Period settings: "RSI / ADX 계산" selects (MTS / Wilder); saved and typed parameters go through normalizeParams() -
//   periods are whole numbers (a typed 14.5 made RSI and its marks vanish), unknown values fall back to the defaults)
// - [2026-11-06] type=feat (Chart window: tick (realtime) / minute (1-60, 1-20 days) / day / week / month / year bars from KIS, candle /
//   line / OHLC style, overlays (MA 5/20/60/120, Bollinger, Ichimoku, VWAP) and panes (RSI, MACD, ADX, stochastic, Williams %R, MFI,
//   OBV) with editable periods, drawing tools (horizontal line, trend line, clear) kept per symbol, wheel zoom / drag pan, crosshair with
//   a data window, realtime trades rolled into the last bar, REST refresh every 15 s for minute charts. All choices are saved with the
//   window (layout))
// - [2026-11-06] type=feat (The data window moves to the right while the cursor is in the left half of the chart; buy / sell signal marks
//   on the indicator panes with an on / off check (saved); periods of the new indicators; ☆ button adds the symbol to watchlist groups.
//   The data window shows OHLCV + change only - indicator values are in the pane legends, a long list covered the lower panes)
// - [2026-11-06] type=fix (Tick charts stayed empty when the realtime connection was not on (the quote window showed "주기 조회"): the tick
//   chart now seeds its trades from the REST trade list (newest 30, KRX + NXT for NXT symbols) when it opens and keeps merging it every
//   3 s while realtime is down (every 10 s while it is on but this symbol had no realtime trade for 15 s); the toolbar note says which
//   source is used)
// - [2026-11-06] type=feat (Tick charts back-fill today's earlier trades like a broker HTS: on open they page backwards from now
//   (inquire_time_itemconclusion via 'ticks', 30 trades per call, 1,500 - 6,000 trades depending on the tick size) and draw as the pages
//   arrive; "◀ 이전 틱" loads more until the first trade of the day (cache cap 30,000). The REST catch-up uses the same API and fetches back to
//   the newest trade held, so trades are no longer lost when more than 30 arrive between polls)
// - [2026-11-06] type=fix (Catch-up keeps following `next` until the newest trade held when it started (it dropped the rest after one
//   call); the back-fill restarts on a new KST day)
// - [2026-11-06] type=feat (Toolbar warns when REST and realtime trades of the same shape carry different accumulated volumes (acmlMismatches))
// - [2026-11-12] type=feat (Earlier days on intraday charts: minute charts load 3 trading days by default (was today only); tick charts
//   show the previous trading day's 1-minute bars before today's tick bars - KIS serves individual trades of today only, so the earlier
//   day cannot be drawn as ticks (the toolbar says so). NXT mark next to the symbol search)
// - [2026-11-19] type=feat (The separate data window is gone: the hovered bar's date / OHLC / volume / change are the first line of the
//   chart's price legend (top left, like the AutoBotH chart))
// - [2026-11-19] type=perf (Charts are drawn from the cache at once (main process chartcache); an answer marked stale is refreshed right
//   away instead of after 15 - 60 s)
// - [2026-11-20] type=feat (Pane 투자자 누적 (user 2026-11-20): daily investor net buying ('investorDaily', about one year, cached in the
//   main process) summed over 1개월 / 3개월 / 6개월 / 1년 - buttons next to the indicator menu while the pane is on (saved); day / week /
//   month / year charts only)
// - [2026-11-20] type=feat (Pane heights by drag (user 2026-11-20): the price / volume split and the indicator pane height are saved with
//   the window (volR, oscH; a double click on the handle = automatic again). The 투자자 누적 pane says when it is loading or KIS refused)

// - [2026-11-20] type=fix (지표 menu / 기간 설정 close on a click outside them or Esc - before, only a click on the chart closed them)
// - [2026-11-20] type=fix (Switching 일 -> 분 (any bar kind / size / days / symbol) drew the old bars with the new settings until the new
//   data came: daily bars as minute bars got a dotted day line between every bar. The loaded data keeps the request it answers (key);
//   bars of another request are not drawn - "불러오는 중" instead)
import { useEffect, useMemo, useRef, useState } from 'react';
import { kstDay, useStore, type Win } from '../store';
import { PriceChart } from '../components/PriceChart';
import { IndicatorToggles } from '../components/IndicatorToggles';
import { SymbolSearch } from '../components/SymbolSearch';
import { NxtTag } from '../components/NxtTag';
import { useOutsideClose, useRealtime } from '../hooks';
import { MIN_SIZES, TICK_SIZES, type Bar, type ChartKind } from '../../../shared/price';
import { FRACTION_PARAMS, HTS_PARAMS, INV_MONTHS, INV_MONTHS_LABEL, normalizeParams, OSCILLATORS, type CalcMethod, type IndicatorKey, type IndicatorParams, type InvMonths } from '../../../shared/indicators';
import { WatchStar } from './ListWindows';
import { acmlMismatches, applyTrades, tickBarsFromTrades, TRADE_CAP } from '../../../shared/live';
import type { ChartRes, InvestorDay, TicksCursor, TicksRes } from '../../../shared/market';
import type { TradeTick } from '../../../shared/kis';

const KINDS: [ChartKind, string][] = [
  ['tick', '틱'],
  ['min', '분'],
  ['day', '일'],
  ['week', '주'],
  ['month', '월'],
  ['year', '년'],
];
const DAYS = [1, 2, 3, 5, 10, 20];
/** minute charts: trading days loaded when the window has no saved choice (today + the two days before) */
const DEFAULT_DAYS = 3;
const NO_BARS: Bar[] = [];
const NO_TRADES: TradeTick[] = [];
type Draw = { h: number[]; t: { t1: number; p1: number; t2: number; p2: number }[] };

type NumParam = { [K in keyof IndicatorParams]: IndicatorParams[K] extends number ? K : never }[keyof IndicatorParams];
const PARAM_FIELDS: [NumParam, string][] = [
  ['bbPeriod', '볼린저 기간'],
  ['bbMult', '볼린저 배수(σ)'],
  ['rsiPeriod', 'RSI 기간'],
  ['rsiSignal', 'RSI 시그널'],
  ['macdFast', 'MACD 단기'],
  ['macdSlow', 'MACD 장기'],
  ['macdSignal', 'MACD 시그널'],
  ['adxPeriod', 'ADX 기간'],
  ['stochK', '스토캐스틱 %K'],
  ['stochD', '스토캐스틱 %D'],
  ['williamsR', 'Williams %R'],
  ['williamsD', 'Williams %D'],
  ['mfiPeriod', 'MFI 기간'],
  ['obvMa', 'OBV 이동평균'],
  ['ichiTenkan', '일목 전환선'],
  ['ichiKijun', '일목 기준선'],
  ['ichiSenkouB', '일목 선행스팬B'],
  ['ichiShift', '일목 선행 이동'],
  ['envPeriod', '엔벨로프 기간'],
  ['envPct', '엔벨로프 폭(%)'],
  ['donchian', '가격채널 기간'],
  ['sarStep', 'SAR 가속'],
  ['sarMax', 'SAR 최대'],
  ['cciPeriod', 'CCI 기간'],
  ['atrPeriod', 'ATR 기간'],
  ['rocPeriod', 'ROC 기간'],
  ['rocSignal', 'ROC 시그널'],
  ['trixPeriod', 'TRIX 기간'],
  ['trixSignal', 'TRIX 시그널'],
  ['disparity', '이격도 기간'],
  ['psyPeriod', '심리도 기간'],
];
/** fractional parameters (step / minimum) */
const FRACTION: Partial<Record<NumParam, number>> = FRACTION_PARAMS;
const METHOD_FIELDS: ['rsiMethod' | 'adxMethod', string][] = [
  ['rsiMethod', 'RSI 계산'],
  ['adxMethod', 'ADX 계산'],
];

function ParamsDialog({ value, onSave, onClose }: { value: IndicatorParams; onSave: (p: IndicatorParams) => void; onClose: () => void }) {
  const [p, setP] = useState<IndicatorParams>(value);
  return (
    <div className="popover params" onMouseDown={(e) => e.stopPropagation()}>
      <h4>지표 기간 설정</h4>
      <div className="params-grid">
        {PARAM_FIELDS.map(([k, label]) => (
          <label key={k}>
            {label}
            <input type="number" min={FRACTION[k] ?? 1} step={FRACTION[k] ?? 1} value={p[k]} onChange={(e) => setP({ ...p, [k]: Math.max(FRACTION[k] ?? 1, Number(e.target.value) || (FRACTION[k] ?? 1)) })} />
          </label>
        ))}
        {METHOD_FIELDS.map(([k, label]) => (
          <label key={k}>
            {label}
            <select value={p[k]} onChange={(e) => setP({ ...p, [k]: e.target.value as CalcMethod })}>
              <option value="mts">MTS 방식</option>
              <option value="wilder">Wilder (봇)</option>
            </select>
          </label>
        ))}
      </div>
      <div className="actions">
        <button className="primary" onClick={() => (onSave(normalizeParams(p)), onClose())}>
          적용
        </button>
        <button onClick={() => setP(HTS_PARAMS)}>기본값</button>
        <button onClick={onClose}>닫기</button>
      </div>
    </div>
  );
}

export function ChartWindow({ win }: { win: Win }) {
  const setOpts = useStore((s) => s.setOpts);
  const setCode = useStore((s) => s.setCode);
  const rtOn = useStore((s) => s.auth?.rt.status === 'on');
  const rtMsg = useStore((s) => s.auth?.rt.msg);
  const mergeTrades = useStore((s) => s.mergeTrades);
  const trades = useStore((s) => s.rt.trades[win.code]) || NO_TRADES;
  const o = win.opts as {
    kind?: ChartKind;
    tickN?: number;
    minN?: number;
    days?: number;
    ind?: IndicatorKey[];
    style?: 'candle' | 'line' | 'ohlc';
    params?: IndicatorParams;
    draw?: Record<string, Draw>;
    sig?: boolean;
    invM?: InvMonths;
    volR?: number;
    oscH?: number;
  };
  const kind = o.kind || 'day';
  const n = kind === 'tick' ? o.tickN || 30 : o.minN || 1;
  const days = o.days || DEFAULT_DAYS;
  const ind = o.ind || ['ma', 'bb', 'macd'];
  const params = useMemo(() => normalizeParams(o.params), [o.params]);
  const draw = o.draw?.[win.code] || { h: [], t: [] };
  const intraday = kind === 'tick' || kind === 'min';
  const shown = useMemo(() => ind.filter((k) => (k !== 'vwap' || intraday) && (k !== 'investor' || !intraday)), [ind.join(), intraday]); // eslint-disable-line react-hooks/exhaustive-deps
  const invOn = shown.includes('investor');
  const invM: InvMonths = INV_MONTHS.includes(o.invM as InvMonths) ? (o.invM as InvMonths) : 3;
  useRealtime(win.code, { trades: true });

  const [res, setRes] = useState<ChartRes & { at: number; key?: string }>({ bars: [], at: 0 });
  const reqKey = `${win.code}|${kind}|${n}|${days}`;
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [chartKey, setChartKey] = useState(0);
  const [panel, setPanel] = useState<'' | 'ind' | 'params'>('');
  const [tool, setTool] = useState<'none' | 'hline' | 'tline'>('none');
  const panelRef = useOutsideClose<HTMLSpanElement>(panel !== '', () => setPanel(''));
  const sig = o.sig ?? true;
  const bodyRef = useRef<HTMLDivElement>(null);
  const [h, setH] = useState(300);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setH(el.clientHeight));
    ro.observe(el);
    setH(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // load (and refresh minute charts every 15 s; day+ charts every 60 s)
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const load = async (refresh: boolean) => {
      if (!refresh) (setLoading(true), setErr(''));
      const r = await window.hts.call<ChartRes>('chart', { code: win.code, kind, n, days, refresh });
      if (!alive) return;
      setLoading(false);
      if (r.ok) setRes({ ...r.data, at: Date.now(), key: `${win.code}|${kind}|${n}|${days}` });
      else if (!refresh) (setErr(r.error), setRes({ bars: [], at: 0 }));
      // [2026-11-19] a cached chart is drawn at once; when it is old the missing part is fetched right away
      const soon = r.ok && r.data.stale;
      if (kind !== 'tick') timer = setTimeout(() => void load(true), soon ? 0 : kind === 'min' ? (rtOn ? 30000 : 15000) : 60000);
    };
    void load(false);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [win.code, kind, n, days, rtOn]);

  // [2026-11-20] investor net buying for the 투자자 누적 pane (main process caches it; asked again every 5 minutes)
  const [inv, setInv] = useState<{ code: string; rows: InvestorDay[]; err: string }>({ code: '', rows: [], err: '' });
  useEffect(() => {
    if (!invOn) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const load = async () => {
      const r = await window.hts.call<InvestorDay[]>('investorDaily', win.code);
      if (!alive) return;
      if (r.ok) setInv({ code: win.code, rows: r.data, err: '' });
      else setInv((v) => (v.code === win.code ? { ...v, err: r.error } : { code: win.code, rows: [], err: r.error }));
      timer = setTimeout(() => void load(), 300000);
    };
    void load();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [win.code, invOn]);
  const invRows = inv.code === win.code ? inv.rows : undefined;

  // tick chart back-fill: today's earlier trades paged backwards (30 per call) from now; "이전 틱" loads more
  const [fill, setFill] = useState<{ busy: boolean; loaded: number; done: boolean; skipped: number }>({ busy: false, loaded: 0, done: false, skipped: 0 });
  const cursor = useRef<TicksCursor | null>(null);
  const fillGen = useRef(0);
  const loadOlder = async (target: number, gen: number) => {
    setFill((f) => ({ ...f, busy: true }));
    let got = 0;
    while (got < target && fillGen.current === gen) {
      const r = await window.hts.call<TicksRes>('ticks', { code: win.code, cursor: cursor.current || undefined, pages: 10 });
      if (fillGen.current !== gen) return;
      if (!r.ok) {
        setFill((f) => ({ ...f, busy: false }));
        return;
      }
      mergeTrades(win.code, r.data.trades);
      got += r.data.trades.length;
      cursor.current = r.data.next;
      const done = !r.data.next || (useStore.getState().rt.trades[win.code]?.length || 0) >= TRADE_CAP;
      setFill((f) => ({ busy: !done && got < target, loaded: f.loaded + r.data.trades.length, done, skipped: f.skipped + r.data.skipped }));
      if (done) return;
    }
  };
  const fillTarget = Math.max(1500, Math.min(6000, n * 100));
  // a new KST day restarts the back-fill (the trade cache is dropped by the store; the old cursor / done state belong to yesterday)
  const [day, setDay] = useState(kstDay);
  useEffect(() => {
    if (kind !== 'tick') return;
    const t = setInterval(() => setDay(kstDay()), 60000);
    return () => clearInterval(t);
  }, [kind]);
  useEffect(() => {
    if (kind !== 'tick') return;
    const gen = ++fillGen.current;
    cursor.current = null;
    setFill({ busy: true, loaded: 0, done: false, skipped: 0 });
    void loadOlder(fillTarget, gen);
    return () => void fillGen.current++;
  }, [kind, win.code, day]); // eslint-disable-line react-hooks/exhaustive-deps

  // catch up with REST while realtime is not delivering (every 3 s while it is down; every 10 s while it is on but this symbol had no
  // realtime trade for 15 s - a refused subscription), from now back to the newest trade already held
  useEffect(() => {
    if (kind !== 'tick') return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    /** an unfinished catch-up (gap longer than one poll's calls): the next poll continues it before looking at new trades */
    let pending: { cursor: TicksCursor; until: number } | null = null;
    const poll = async () => {
      const st = useStore.getState().rt;
      const rtAt = st.quotes[win.code]?.at || 0;
      if (!rtOn || Date.now() - rtAt > 15000) {
        // fetch from now back to the newest trade held when this catch-up started (fixed), following `next` past one call's pages
        const until = pending?.until ?? (st.trades[win.code] || []).reduce((m, t) => Math.max(m, t.acml || 0), 0);
        let cur: TicksCursor | undefined = pending?.cursor;
        pending = null;
        for (let i = 0; i < 6; i++) {
          const r = await window.hts.call<TicksRes>('ticks', { code: win.code, cursor: cur, pages: until ? 10 : 1, until: until || undefined });
          if (!alive) return;
          if (!r.ok) {
            // keep the unfinished range for the next poll (only when something was already fetched past `until`)
            if (cur && until) pending = { cursor: cur, until };
            break;
          }
          mergeTrades(win.code, r.data.trades);
          const reached = !until || r.data.trades.some((t) => (t.acml || 0) <= until);
          if (reached || !r.data.next) break;
          cur = r.data.next;
          if (i === 5) pending = { cursor: cur, until };
        }
      }
      timer = setTimeout(() => void poll(), rtOn ? 10000 : 3000);
    };
    timer = setTimeout(() => void poll(), rtOn ? 10000 : 3000);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [kind, win.code, rtOn, mergeTrades]);

  // tick chart: the previous trading day's 1-minute bars in front of today's tick bars (KIS has no trade list of earlier days)
  const [prevMin, setPrevMin] = useState<{ key: string; bars: Bar[] }>({ key: '', bars: NO_BARS });
  const prevKey = `${win.code}|${day}`;
  useEffect(() => {
    if (kind !== 'tick') return;
    let alive = true;
    void window.hts.call<ChartRes>('chart', { code: win.code, kind: 'min', n: 1, days: 2 }).then((r) => {
      if (!alive || !r.ok) return;
      // KST midnight of today: only bars of earlier days are kept
      const day0 = Math.floor((Date.now() + 9 * 3600000) / 86400000) * 86400000 - 9 * 3600000;
      setPrevMin({ key: prevKey, bars: r.data.bars.filter((b) => b.t < day0) });
    });
    return () => {
      alive = false;
    };
  }, [kind, prevKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const prevBars = kind === 'tick' && prevMin.key === prevKey ? prevMin.bars : NO_BARS;

  const bars: Bar[] = useMemo(() => {
    if (kind === 'tick') {
      const tb = tickBarsFromTrades(trades, n);
      return prevBars.length ? [...prevBars, ...tb] : tb;
    }
    // [2026-11-20] only the data of the current request (another kind's bars must not be drawn with these settings)
    if (res.key !== reqKey) return NO_BARS;
    // trades after the REST fetch time roll into the bars (REST data is newest up to about its fetch time)
    return applyTrades(res.bars, trades, kind, n, res.at - 1000);
  }, [res, trades, kind, n, prevBars, reqKey]);

  // REST and realtime trades must share one accumulated-volume basis, or the same trade is counted twice - shown when it happens
  const mismatch = useMemo(() => (kind === 'tick' ? acmlMismatches(trades) : 0), [kind, trades]);

  const saveDraw = (d: Draw) => setOpts(win.id, { draw: { ...(o.draw || {}), [win.code]: d } });
  const chartH = Math.max(200, h);
  const nOsc = shown.filter((k) => OSCILLATORS.includes(k)).length;
  const autoOsc = Math.max(56, Math.min(100, Math.floor((chartH * 0.42) / Math.max(1, nOsc)) - 8));
  // a height set by dragging, kept inside what the window can show (price + volume keep at least 140 px)
  const oscH = o.oscH ? Math.max(36, Math.min(o.oscH, Math.floor((chartH - 172) / Math.max(1, nOsc)) - 8)) : autoOsc;
  const invNote = !invOn ? '' : inv.err && inv.code === win.code ? `조회 오류: ${inv.err}` : !invRows ? '불러오는 중…' : invRows.length ? '' : '투자자 데이터 없음';

  return (
    <div className="chart-win">
      <div className="win-tb">
        <SymbolSearch compact value={win.code} onPick={(c) => setCode(win.id, c)} />
        <NxtTag code={win.code} />
        <WatchStar code={win.code} />
        <span className="seg">
          {KINDS.map(([k, l]) => (
            <button key={k} className={kind === k ? 'on' : ''} onClick={() => setOpts(win.id, { kind: k })}>
              {l}
            </button>
          ))}
        </span>
        {kind === 'tick' && (
          <select value={n} onChange={(e) => setOpts(win.id, { tickN: Number(e.target.value) })}>
            {TICK_SIZES.map((v) => (
              <option key={v} value={v}>
                {v}틱
              </option>
            ))}
          </select>
        )}
        {kind === 'min' && (
          <>
            <select value={n} onChange={(e) => setOpts(win.id, { minN: Number(e.target.value) })}>
              {MIN_SIZES.map((v) => (
                <option key={v} value={v}>
                  {v}분
                </option>
              ))}
            </select>
            <select value={days} onChange={(e) => setOpts(win.id, { days: Number(e.target.value) })} title="불러올 거래일 수 (지난 날짜 분봉은 실전 서버에서만)">
              {DAYS.map((v) => (
                <option key={v} value={v}>
                  {v}일
                </option>
              ))}
            </select>
          </>
        )}
        <select value={o.style || 'candle'} onChange={(e) => setOpts(win.id, { style: e.target.value })} title="차트 형태">
          <option value="candle">캔들</option>
          <option value="ohlc">OHLC 바</option>
          <option value="line">라인</option>
        </select>
        <span className="dropdown" ref={panelRef}>
          <button className={panel === 'ind' ? 'on' : ''} onClick={() => setPanel(panel === 'ind' ? '' : 'ind')}>
            지표 ({ind.length}) ▾
          </button>
          {panel === 'ind' && (
            <span className="popover ind-pop" onMouseDown={(e) => e.stopPropagation()}>
              <IndicatorToggles value={ind} onChange={(v) => setOpts(win.id, { ind: v })} kind={kind} params={params} />
              <span className="ind-pop-foot">
                <label className="check" title="보조지표 창에 매수(빨강 ▲) / 매도(파랑 ▼) 신호 표시">
                  <input type="checkbox" checked={sig} onChange={(e) => setOpts(win.id, { sig: e.target.checked })} /> 매수/매도 신호
                </label>
                <button className="mini" onClick={() => setPanel('params')}>
                  기간 설정…
                </button>
              </span>
            </span>
          )}
          {panel === 'params' && <ParamsDialog value={params} onSave={(p) => setOpts(win.id, { params: p })} onClose={() => setPanel('')} />}
        </span>
        {invOn && (
          <span className="seg" title={`투자자 누적 순매수 기간${inv.err && inv.code === win.code ? ` - 조회 오류: ${inv.err}` : ''}`}>
            {INV_MONTHS.map((m) => (
              <button key={m} className={invM === m ? 'on' : ''} onClick={() => setOpts(win.id, { invM: m })}>
                {INV_MONTHS_LABEL[m]}
              </button>
            ))}
          </span>
        )}
        <span className="seg tools" title="그리기 도구 (가격 영역 클릭)">
          <button className={tool === 'none' ? 'on' : ''} onClick={() => setTool('none')}>
            ↖
          </button>
          <button className={tool === 'hline' ? 'on' : ''} onClick={() => setTool('hline')} title="수평선: 가격 영역을 클릭">
            ―
          </button>
          <button className={tool === 'tline' ? 'on' : ''} onClick={() => setTool('tline')} title="추세선: 두 점을 차례로 클릭">
            ╱
          </button>
          <button onClick={() => saveDraw({ h: [], t: [] })} title="이 종목의 그린 선 모두 지우기" disabled={!draw.h.length && !draw.t.length}>
            지우기
          </button>
        </span>
        {kind === 'tick' && (
          <button className="mini" disabled={fill.busy || fill.done} onClick={() => void loadOlder(fillTarget, fillGen.current)} title={`오늘 이전 체결을 ${fillTarget.toLocaleString()}건 더 불러옵니다 (30건씩 조회)`}>
            ◀ 이전 틱
          </button>
        )}
        <button className="mini" onClick={() => setChartKey((k) => k + 1)} title="확대·이동 초기화 (차트 더블클릭)">
          ⟲
        </button>
        <span className="muted small">
          {loading ? '불러오는 중…' : err ? '' : `${bars.length.toLocaleString()}봉`}
          {kind === 'tick'
            ? ` · ${fill.busy ? `이전 체결 불러오는 중… ${fill.loaded.toLocaleString()}건` : fill.done ? '오늘 첫 체결까지 불러옴' : `이전 체결 ${fill.loaded.toLocaleString()}건 불러옴`}` +
              (fill.skipped ? ` (초당 30건 넘는 ${fill.skipped}개 구간은 일부 누락)` : '') +
              (prevBars.length ? ' · 이전 거래일은 1분봉 (지난 날짜 틱은 KIS 미제공)' : '') +
              (mismatch >= 3 ? ` · ⚠ 실시간·조회 체결 대조 불일치 ${mismatch}건 (중복 가능)` : '') +
              (rtOn ? ' · 실시간' : ` · 실시간 미연결${rtMsg ? ` (${rtMsg})` : ''} - 3초마다 조회`)
            : res.note
              ? ` · ${res.note}`
              : ''}
        </span>
      </div>
      <div className="chart-body" ref={bodyRef} onMouseDown={() => panel && setPanel('')}>
        {err && <div className="empty">{err}</div>}
        {!err && !loading && !bars.length && <div className="empty">{kind === 'tick' ? '체결을 기다리는 중입니다 (거래가 있는 시간에만 쌓입니다).' : '봉이 없습니다.'}</div>}
        {bars.length > 0 && (
          <PriceChart
            key={`${chartKey}/${win.code}/${kind}/${n}/${days}`}
            bars={bars}
            kind={kind}
            n={n}
            indicators={shown}
            investor={invRows}
            invMonths={invM}
            invNote={invNote}
            volRatio={o.volR}
            onVolRatio={(v) => setOpts(win.id, { volR: v ?? undefined })}
            onOscHeight={(v) => setOpts(win.id, { oscH: v ?? undefined })}
            trades={[]}
            date=""
            mode="index"
            axis="right"
            left={6}
            right={70}
            height={chartH}
            oscHeight={oscH}
            interactive
            tooltip={false}
            signals={sig}
            params={params}
            style={o.style || 'candle'}
            tool={tool}
            hlines={draw.h}
            tlines={draw.t}
            onAddHline={(p) => saveDraw({ ...draw, h: [...draw.h, p] })}
            onAddTline={(l) => saveDraw({ ...draw, t: [...draw.t, l] })}
          />
        )}
      </div>
    </div>
  );
}
