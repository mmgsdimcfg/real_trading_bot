/*
 * Update log
 * - [2026-11-16] type=feat - New: a [REJECT] reason of the buy pipeline split into the checks that failed, each with its measured value, the
 *   limit it was compared with and the r001 keys that set that limit (the pipeline side panel marks those rows red). The server writes
 *   several failed checks of the triple trigger into one reason (TRIPLE_CROSS_MISSING_A|B|C); every part becomes one item. Limits come
 *   from the reason text when the server wrote them (the value the running bot used), otherwise from the current setting.
 * - [2026-11-16] type=feat - MA checks with numbers: the server ends a triple-trigger reject with "| MAVAL 1m(5=..,20=..) t60(5=..,20=..,
 *   bars=n) ..." (r005 _ma_values_text); "MA5 below MA20" items show both values and the gap. The full reason is read from the raw
 *   line (fullReason) - the app's event reason keeps only the first part of "A|B|C"
 * - [2026-11-17] type=fix - Reasons without a rule fell back to "a key of the condition whose value equals the limit", which picked the
 *   wrong key: HYBRID_3MIN_CTX_LOW_ENTRY_ATR_0.38%_LT_0.40% marked GC_VOLATILITY_MIN_ATR_1MIN_PCT (then 0.4) although the 3-minute
 *   floor is RELAXED_ENTRY_ATR_MIN_PCT (0.004) from 12:00 / on the triple path, else STAGED_TP1_PCT x MIN_ENTRY_ATR_TO_TP1_RATIO.
 *   Rules added for the 3-minute ATR floor, the liquidity floors (LOW_VOL_MA_ABS / LOW_ABS_VOLUME + turnover alternative), the
 *   Edge AI probability and the aux sell score
 */

/** one failed check of a reject reason */
export interface BlockItem {
  /** the reason part this item was read from */
  code: string;
  /** what failed, in Korean */
  label: string;
  /** measured value ("1428초") - absent for checks without a number (e.g. MA5 below MA20) */
  actual?: string;
  /** what was required ("≤ 900초") */
  limit?: string;
  /** r001 keys that decide this check; the first one is the threshold to change */
  keys: string[];
}

/** current value of an r001 key (effective setting) - undefined when unknown */
export type ValueOf = (key: string) => unknown;

const UNIT: Record<string, string> = { s: '초', '%': '%', '/min': '회/분', '': '' };

const num = (v: unknown): number | null => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};

function fmtNum(s: string | number): string {
  const n = Number(s);
  if (!Number.isFinite(n)) return String(s);
  return Number.isInteger(n) ? n.toLocaleString('ko-KR') : String(+n.toFixed(3));
}

function seconds(s: string | number): string {
  const n = Number(s);
  return n >= 120 ? `${fmtNum(n)}초 (${(n / 60).toFixed(1)}분)` : `${fmtNum(n)}초`;
}

/** "≤ 900초" from a key's current value */
function limitOf(valueOf: ValueOf, key: string, op: '≤' | '≥', unit = ''): string | undefined {
  const v = num(valueOf(key));
  if (v === null) return undefined;
  return `${op} ${unit === '초' ? seconds(v) : fmtNum(v) + unit}`;
}

/** last MA values of one series from the MAVAL segment: 1m / t60 / t120 */
export interface MaValues { fast: number; slow: number; f: number; s: number; bars?: number }

/** "... | MAVAL 1m(5=86412.50,20=86500.20) t60(5=..,20=..,bars=45)" -> { '1m': {...}, t60: {...} } */
export function parseMaValues(raw: string): Record<string, MaValues> {
  const out: Record<string, MaValues> = {};
  const seg = /\|\s*MAVAL\s+([^|]*)/.exec(raw || '');
  if (!seg) return out;
  const re = /(\w+)\(([^)]*)\)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(seg[1]))) {
    const kv = m[2].split(',').map((p) => p.split('='));
    const ma = kv.filter(([k]) => /^\d+$/.test(k)).map(([k, v]) => [Number(k), Number(v)] as const);
    if (ma.length < 2) continue;
    const bars = kv.find(([k]) => k === 'bars');
    out[m[1]] = { fast: ma[0][0], f: ma[0][1], slow: ma[1][0], s: ma[1][1], ...(bars ? { bars: Number(bars[1]) } : {}) };
  }
  return out;
}

/** The whole reason of a raw [REJECT] line ("A|B|C" up to " | GATES"); `reason` when the raw line does not hold it */
export function fullReason(reason: string, raw: string): string {
  const i = reason ? (raw || '').indexOf(reason) : -1;
  if (i < 0) return reason;
  return raw.slice(i).split(/\s+\|\s+/)[0].trim();
}

function maActual(v: MaValues | undefined): string | undefined {
  if (!v || !Number.isFinite(v.f) || !Number.isFinite(v.s)) return undefined;
  const gap = v.s ? ((v.f - v.s) / v.s) * 100 : NaN;
  const n = (x: number) => x.toLocaleString('ko-KR', { maximumFractionDigits: 2 });
  return `MA${v.fast} ${n(v.f)} / MA${v.slow} ${n(v.s)}${Number.isFinite(gap) ? ` (${gap >= 0 ? '+' : ''}${gap.toFixed(2)}%)` : ''}${v.bars !== undefined ? ` · ${v.bars}봉` : ''}`;
}

type Builder = (m: RegExpMatchArray, valueOf: ValueOf, ma: Record<string, MaValues>) => Omit<BlockItem, 'code'>;

// specific checks of the entry trigger (r002 evaluate_triple_trigger, r013 tick crosses, r005 _009)
const SPECIFIC: [RegExp, Builder][] = [
  [/^1MIN_MA(\d+)_MA(\d+)_ABOVE_TOO_OLD_(\d+)s_GT_(\d+)s$/, (m) => ({
    label: `1분봉 MA${m[1]} > MA${m[2]} 상태가 너무 오래됨 (마지막 골든크로스 이후 경과 시간)`,
    actual: seconds(m[3]), limit: `≤ ${seconds(m[4])}`,
    keys: ['TRIPLE_1MIN_ABOVE_MAX_AGE_SECONDS', 'TRIPLE_1MIN_AGE_FROM_REGULAR_OPEN', 'TRIPLE_1MIN_CONDITION'],
  })],
  [/^1MIN_MA(\d+)_MA(\d+)_CROSS_EXPIRED_(\d+)s/, (m, v) => ({
    label: `1분봉 MA${m[1]}/MA${m[2]} 골든크로스 유효시간 지남`,
    actual: seconds(m[3]), limit: limitOf(v, 'TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS', '≤', '초'),
    keys: ['TRIPLE_1MIN_CROSS_MAX_AGE_SECONDS', 'TRIPLE_1MIN_CONDITION'],
  })],
  [/^1MIN_MA(\d+)_MA(\d+)_NOT_ABOVE$/, (m, _v, ma) => ({
    label: `1분봉 MA${m[1]}가 MA${m[2]} 아래`, limit: `MA${m[1]} > MA${m[2]}`, actual: maActual(ma['1m']),
    keys: ['TRIPLE_1MIN_MA_FAST', 'TRIPLE_1MIN_MA_SLOW', 'TRIPLE_1MIN_CONDITION'],
  })],
  [/^1MIN_MA(\d+)_MA(\d+)_(NO_CROSS\w*|CROSS_TIME_INVALID)$/, (m, _v, ma) => ({
    label: `1분봉 MA${m[1]}/MA${m[2]} 골든크로스 없음`, limit: '골든크로스 필요', actual: maActual(ma['1m']),
    keys: ['TRIPLE_1MIN_CONDITION', 'TRIPLE_1MIN_MA_FAST', 'TRIPLE_1MIN_MA_SLOW'],
  })],
  [/^TICK(\d+)_MA(\d+)_NOT_ABOVE_MA(\d+)$/, (m, _v, ma) => ({
    label: `${m[1]}틱봉 MA${m[2]}가 MA${m[3]} 아래`, limit: `MA${m[2]} > MA${m[3]}`, actual: maActual(ma[`t${m[1]}`]),
    keys: ['TRIPLE_TICK_CONDITION', 'TICK_TRIGGER_MA_FAST', 'TICK_TRIGGER_MA_SLOW', 'TICK_TRIGGER_BAR_SIZES'],
  })],
  [/^TICK(\d+)_CROSS_EXPIRED(?:_(\d+)s)?/, (m, v) => ({
    label: `${m[1]}틱봉 골든크로스 유효시간 지남`,
    actual: m[2] ? seconds(m[2]) : '유효시간 밖', limit: limitOf(v, 'TICK_TRIGGER_CROSS_MAX_AGE_SECONDS', '≤', '초'),
    keys: ['TICK_TRIGGER_CROSS_MAX_AGE_SECONDS', 'TRIPLE_TICK_CONDITION'],
  })],
  [/^TICK(\d+)_NO_CROSS_IN_SESSION$/, (m, _v, ma) => ({
    label: `${m[1]}틱봉 장중 골든크로스 없음 (오늘 첫 봉부터 계속 위)`, limit: '골든크로스 필요', actual: maActual(ma[`t${m[1]}`]),
    keys: ['TRIPLE_TICK_CONDITION', 'TICK_TRIGGER_MA_FAST', 'TICK_TRIGGER_MA_SLOW'],
  })],
  [/^TICK(\d+)_ABSTAIN_FEW_BARS_(\d+)$/, (m, v) => {
    const slow = num(v('TICK_TRIGGER_MA_SLOW'));
    return {
      label: `${m[1]}틱봉 개수 부족 (판정 보류)`, actual: `${m[2]}봉`, limit: slow !== null ? `≥ ${slow + 1}봉` : undefined,
      keys: ['TICK_TRIGGER_MA_SLOW', 'TICK_TRIGGER_PRIOR_DAY_WARMUP'],
    };
  }],
  [/^TICK(\d+)_(NOT_EVALUATED|ABSTAIN_\w+)$/, (m) => ({
    label: `${m[1]}틱봉 판정 못 함 (${m[2]})`, keys: ['TICK_TRIGGER_MODE', 'TICK_TRIGGER_FEED_STALE_SECONDS'],
  })],
  [/^TICK_LOW_DENSITY_([\d.]+)\/min_LT_([\d.]+)$/, (m) => ({
    label: '틱 체결 밀도 부족', actual: `${fmtNum(m[1])}회/분`, limit: `≥ ${fmtNum(m[2])}회/분`,
    keys: ['TICK_MIN_PRINTS_PER_MIN', 'TICK_MIN_PRINTS_WINDOW_MINUTES'],
  })],
  [/^(TICK_FEED_UNAVAILABLE|TICK_NO_RESULT|TICK_ABSTAIN_\w+)$/, (m) => ({
    label: `틱 데이터로 판정 못 함 (${m[1]})`, keys: ['TICK_TRIGGER_MODE', 'TICK_TRIGGER_FEED_STALE_SECONDS'],
  })],
  // r002 _gate_min_entry_atr_volatility: min(STAGED_TP1_PCT x MIN_ENTRY_ATR_TO_TP1_RATIO, RELAXED_ENTRY_ATR_MIN_PCT when relaxed)
  [/^HYBRID_(\d+)MIN_CTX_LOW_ENTRY_ATR_([\d.]+)%_LT_([\d.]+)%$/, (m, v) => {
    const relaxed = num(v('RELAXED_ENTRY_ATR_MIN_PCT'));
    const isRelaxed = relaxed !== null && Math.abs(relaxed * 100 - Number(m[3])) < 0.006;
    const base = ['MIN_ENTRY_ATR_TO_TP1_RATIO', 'STAGED_TP1_PCT'];
    const rel = ['RELAXED_ENTRY_ATR_MIN_PCT', 'RELAXED_ENTRY_ATR_FROM', 'ENABLE_RELAXED_ENTRY_ATR'];
    return {
      label: `${m[1]}분봉 변동성(ATR%) 부족 - ${isRelaxed ? '완화 하한 RELAXED_ENTRY_ATR_MIN_PCT (비율, 0.004 = 0.4%)' : '하한 = STAGED_TP1_PCT x MIN_ENTRY_ATR_TO_TP1_RATIO'}`,
      actual: `${fmtNum(m[2])}%`, limit: `≥ ${fmtNum(m[3])}%`,
      keys: [...(isRelaxed ? rel : base), ...(isRelaxed ? base : rel), 'ENABLE_MIN_ENTRY_ATR_FILTER'],
    };
  }],
  // [2026-11-17] r005 _010 daily market trend gate
  [/^MARKET_REGIME_DOWN_DAILY_MA(\d+)$/, (m) => ({
    label: `지수 ETF 일봉 ${m[1]}일 추세 하락 (종가 < MA${m[1]} 또는 MA${m[1]} 하락)`, limit: `종가 > MA${m[1]}, MA${m[1]} 상승`,
    keys: ['ENABLE_MARKET_DAILY_TREND_GATE', 'MARKET_DAILY_TREND_MA', 'MARKET_DAILY_TREND_SLOPE_DAYS', 'MARKET_DAILY_TREND_SCOPE'],
  })],
  // r002 aux reversal sell: the minimum pnl of that score
  [/^AUX_BLOCKED_SCORE_(\d+)_PNL_(-?[\d.]+)%_LT_(-?[\d.]+)%$/, (m) => ({
    label: `반전 점수 ${m[1]} 매도 최소 수익 미달`, actual: `${fmtNum(m[2])}%`, limit: `≥ ${fmtNum(m[3])}%`,
    keys: [`AUX_SELL_MIN_PNL_SCORE${Math.min(4, Math.max(2, Number(m[1])))}`, 'AUX_SELL_MIN_REALIZED_TARGET_PCT', 'AUX_SELL_TRIGGER_SLIPPAGE_BUFFER_PCT'],
  })],
  [/^(\d+)MIN_LIVE_BELOW_BB_MID$/, (m) => ({
    label: `현재가가 ${m[1]}분봉 볼린저 중심선 아래`, limit: '중심선 위', keys: ['TRIPLE_REQUIRE_1MIN_ABOVE_MID'],
  })],
];

// checks written as "<name>_<actual><unit>_GT|LT_<limit><unit>" - name prefix -> label and keys
const COMPARE: [string, string, string[]][] = [
  ['1MIN_CANDLE_NOT_BULLISH', '1분봉 캔들 상승률 부족', ['HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MIN_PCT']],
  ['1MIN_CHASE_BUY_INTRABAR', '1분봉 안에서 급등 (추격 매수 금지)', ['HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT']],
  ['1MIN_CHASE_BUY_BB_GAP', '1분봉 볼린저 중심선보다 너무 위 (추격 매수 금지)', [
    'HYBRID_1MIN_TRIGGER_BB_GAP_MAX_PCT', 'HYBRID_1MIN_TRIGGER_BB_GAP_DECAY_PCT_PER_BAR', 'HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_PCT',
    'HYBRID_1MIN_TRIGGER_BB_GAP_CEILING_UPTREND_PCT']],
  ['1MIN_LIVE_DROP', '확정 1분봉 종가보다 현재가 하락', ['HYBRID_1MIN_TRIGGER_LIVE_DROP_MAX_PCT']],
  ['CHASE_FROM_CROSS', '크로스 시점 가격보다 너무 오름 (추격 매수 금지)', ['TRIPLE_CHASE_FROM_CROSS_MAX_PCT', 'TRIPLE_CHASE_FROM_CROSS_REF']],
  ['GC_VOL_LOW_1MIN_ATR', '1분봉 변동성(ATR%) 부족', ['GC_VOLATILITY_MIN_ATR_1MIN_PCT']],
  ['HYBRID_1MIN_TRIGGER_EXPIRED', '1분봉 트리거 유효시간 지남', ['HYBRID_1MIN_TRIGGER_MAX_AGE_SECONDS']],
  ['HYBRID_1MIN_TRIGGER_CROSS_EXPIRED', '1분봉 크로스 유효시간 지남', ['HYBRID_1MIN_TRIGGER_CROSS_MAX_AGE_SECONDS']],
  ['LOW_VOL_MA_ABS', '3분봉 거래량 평균(MA20) 부족', ['MIN_ENTRY_VOL_MA', 'ENABLE_MIN_ENTRY_TURNOVER_ALT', 'MIN_ENTRY_VOL_MA_TURNOVER_ALT_KRW']],
  ['LOW_ABS_VOLUME', '3분봉 거래량 부족', ['MIN_ENTRY_VOLUME', 'ENABLE_MIN_ENTRY_TURNOVER_ALT', 'MIN_ENTRY_TURNOVER_ALT_KRW']],
  ['EDGE_AI_LOW_PROB', 'Edge AI 예측 수익 확률 부족', ['EDGE_AI_MIN_PROB_OVERRIDE', 'EDGE_AI_MODE']],
  ['CANDLE_NOT_BULLISH', '캔들 상승률 부족', []],
  ['CHASE_BUY_INTRABAR', '봉 안에서 급등 (추격 매수 금지)', []],
  ['CHASE_BUY_BB_GAP', '볼린저 중심선보다 너무 위 (추격 매수 금지)', []],
];

const COMPARE_RE = /^(.*?)_(?:p=)?(-?[\d.]+)(%|s|\/min)?_(GT|LT|GE|LE)_(-?[\d.]+)(%|s|\/min)?$/;

/** the reason without its family prefix, split into the failed parts */
function parts(reason: string): string[] {
  let r = reason.trim().split(/\s+\|\s+/)[0]; // "[REJECT] code | reason | GATES ..." - only the reason part
  for (const p of ['TRIPLE_CROSS_MISSING_', 'TRIPLE_CROSS_']) {
    if (r.startsWith(p)) {
      r = r.slice(p.length);
      break;
    }
  }
  return r.split('|').map((s) => s.trim()).filter(Boolean);
}

/**
 * Failed checks of one reject reason. `valueOf` gives the current setting of a key (for limits the reason does not carry);
 * `candidates` (the keys of the condition shown) is the fallback for a comparison whose name is not in the table: the key whose
 * current value equals the written limit.
 */
export function explainBlock(reason: string, valueOf: ValueOf, candidates: string[] = [], raw = ''): BlockItem[] {
  const out: BlockItem[] = [];
  const ma = parseMaValues(raw);
  for (const code of parts(fullReason(reason || '', raw))) {
    let item: Omit<BlockItem, 'code'> | null = null;
    for (const [re, build] of SPECIFIC) {
      const m = re.exec(code);
      if (m) {
        item = build(m, valueOf, ma);
        break;
      }
    }
    if (!item) {
      const m = COMPARE_RE.exec(code);
      if (m) {
        const [, name, a, ua = '', op, b, ub = ''] = m;
        const hit = COMPARE.find(([p]) => name === p || name.endsWith(`_${p}`) || name.startsWith(p));
        let keys = hit ? hit[2] : [];
        if (!keys.length) {
          const lim = Number(b);
          keys = candidates.filter((k) => num(valueOf(k)) === lim).slice(0, 1);
        }
        const cmp = op === 'GT' ? '≤' : op === 'GE' ? '<' : op === 'LT' ? '≥' : '>';
        item = {
          label: hit ? hit[1] : name.replace(/_/g, ' '),
          actual: `${fmtNum(a)}${UNIT[ua]}`,
          limit: `${cmp} ${fmtNum(b)}${UNIT[ub || ua]}`,
          keys,
        };
      }
    }
    out.push({ code, ...(item || { label: code, keys: [] }) });
  }
  return out;
}

/** key -> short note for the settings table ("현재 1,428초 / 기준 ≤ 900초"); the first key of an item carries the comparison */
export function blockedKeyNotes(items: BlockItem[]): Record<string, string> {
  const notes: Record<string, string> = {};
  for (const it of items) {
    it.keys.forEach((k, i) => {
      const note = i === 0 && (it.actual || it.limit)
        ? `막힘: ${it.label}${it.actual ? ` · 현재 ${it.actual}` : ''}${it.limit ? ` · 기준 ${it.limit}` : ''}`
        : `관련: ${it.label}`;
      notes[k] = notes[k] ? `${notes[k]}\n${note}` : note;
    });
  }
  return notes;
}
