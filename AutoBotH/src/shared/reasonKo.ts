/*
 * Update log
 * - [2026-11-11] type=feat - Korean wording for the bot's evaluation reason codes (board 사유 column, timeline tooltip). A reason
 *   is split at '|' / '+' and every token is matched against ordered regex rules (numbers kept); unknown tokens stay as they
 *   are, so a new code is still readable. The raw code is kept in the tooltip by the callers
 */

type Rule = [RegExp, (m: RegExpMatchArray) => string];

const TICK = (m: RegExpMatchArray, i = 1) => `${m[i]}틱`;

/** ordered: the first matching rule wins */
const RULES: Rule[] = [
  // --- triple cross (r005 _008 / r002 evaluate_triple_cross) ---
  [/^TRIPLE_CROSS_MISSING_1MIN_MA(\d+)_MA(\d+)_NOT_ABOVE$/, (m) => `1분봉 MA${m[1]}가 MA${m[2]} 아래`],
  [/^1MIN_MA(\d+)_MA(\d+)_NOT_ABOVE$/, (m) => `1분봉 MA${m[1]}가 MA${m[2]} 아래`],
  [/^1MIN_MA(\d+)_MA(\d+)_NO_CROSS_IN_FRAME$/, (m) => `1분봉 MA${m[1]}/MA${m[2]} 골든크로스 없음`],
  [/^(?:TRIPLE_CROSS_)?1MIN_MA(\d+)_ABOVE_MA(\d+)$/, (m) => `1분봉 MA${m[1]} > MA${m[2]}`],
  [/^TRIPLE_CROSS_MISSING_TICK(\d+)_MA(\d+)_NOT_ABOVE_MA(\d+)$/, (m) => `${TICK(m)} MA${m[2]}가 MA${m[3]} 아래`],
  [/^TICK(\d+)_MA(\d+)_NOT_ABOVE_MA(\d+)$/, (m) => `${TICK(m)} MA${m[2]}가 MA${m[3]} 아래`],
  [/^TRIPLE_CROSS_MISSING_TICK_ABSTAIN_FEED_STALE_(\d+)s$/, (m) => `틱 수신 끊김 ${m[1]}초 (판정 보류)`],
  [/^TICK_ABSTAIN_FEED_STALE_(\d+)s$/, (m) => `틱 수신 끊김 ${m[1]}초 (판정 보류)`],
  [/^TRIPLE_CROSS_MISSING_TICK(\d+)_CROSS_EXPIRED(?:_(\d+)s)?(?:_cross=(\S+))?$/, (m) => `${TICK(m)} 골든크로스 유효시간 지남${m[2] ? ` (${m[2]}초 전${m[3] ? ` · ${m[3]}` : ''})` : ''}`],
  [/^TICK(\d+)_CROSS_EXPIRED(?:_(\d+)s)?(?:_cross=(\S+))?$/, (m) => `${TICK(m)} 골든크로스 유효시간 지남${m[2] ? ` (${m[2]}초 전${m[3] ? ` · ${m[3]}` : ''})` : ''}`],
  [/^TRIPLE_CROSS_MISSING_TICK_LOW_DENSITY_([\d.]+)\/min_LT_([\d.]+)$/, (m) => `틱 밀도 부족 ${m[1]}/분 < ${m[2]}`],
  [/^TICK_LOW_DENSITY_([\d.]+)\/min_LT_([\d.]+)$/, (m) => `틱 밀도 부족 ${m[1]}/분 < ${m[2]}`],
  [/^TRIPLE_CROSS_MISSING_TICK(\d+)_ABSTAIN_FEW_BARS_(\d+)$/, (m) => `${TICK(m)}봉 부족 ${m[2]}개 (판정 보류)`],
  [/^TICK(\d+)_ABSTAIN_FEW_BARS_(\d+)$/, (m) => `${TICK(m)}봉 부족 ${m[2]}개 (판정 보류)`],
  [/^TRIPLE_CROSS_MISSING_TICK(\d+)_ABSTAIN_INVALID_BAR$/, (m) => `${TICK(m)}봉 비정상 (판정 보류)`],
  [/^TICK(\d+)_ABSTAIN_INVALID_BAR$/, (m) => `${TICK(m)}봉 비정상 (판정 보류)`],
  [/^TRIPLE_CROSS_MISSING_TICK_ABSTAIN_NO_FILE$/, () => '틱 파일 없음 (수집기가 이 종목을 구독하지 않음)'],
  [/^TICK_ABSTAIN_NO_FILE$/, () => '틱 파일 없음 (수집기가 이 종목을 구독하지 않음)'],
  [/^TRIPLE_CROSS_MISSING_TICK_ABSTAIN_NO_TICKS$/, () => '틱 없음'],
  [/^TICK_ABSTAIN_NO_TICKS$/, () => '틱 없음'],
  [/^TRIPLE_CROSS_MISSING_TICK(\d+)_NO_CROSS_IN_SESSION$/, (m) => `${TICK(m)} 장중 골든크로스 없음`],
  [/^TICK(\d+)_NO_CROSS_IN_SESSION$/, (m) => `${TICK(m)} 장중 골든크로스 없음`],
  [/^TICK(\d+)_NOT_EVALUATED$/, (m) => `${TICK(m)} 미평가`],
  [/^TICK(\d+)_CROSS_TIME_INVALID$/, (m) => `${TICK(m)} 크로스 시각 오류`],
  [/^TICK(\d+)_GOLDEN_CROSS$/, (m) => `${TICK(m)} 골든크로스`],
  [/^TRIPLE_CROSS_CHASE_FROM_CROSS_([\d.]+)%_GT_([\d.]+)%$/, (m) => `크로스 시점보다 ${m[1]}% 올라 추격 매수 금지 (한도 ${m[2]}%)`],
  [/^TRIPLE_CROSS_(\d+)MIN_LIVE_BELOW_BB_MID$/, (m) => `현재가가 ${m[1]}분봉 볼린저 중심선 아래`],
  [/^TRIPLE_CROSS_(\d+)MIN_CHASE_BUY_BB_GAP_([\d.]+)%_GT_([\d.]+)%$/, (m) => `${m[1]}분봉 볼린저 중심선 대비 ${m[2]}% 위 (한도 ${m[3]}%, 추격 매수 금지)`],
  [/^TRIPLE_CROSS_(\d+)MIN_CHASE_BUY_INTRABAR_([\d.]+)%_GT_([\d.]+)%$/, (m) => `${m[1]}분봉 안에서 ${m[2]}% 급등 (한도 ${m[3]}%, 추격 매수 금지)`],
  [/^TRIPLE_CROSS_(\d+)MIN_CANDLE_NOT_BULLISH_(-?[\d.]+)%_LT_(-?[\d.]+)%$/, (m) => `${m[1]}분봉이 음봉 ${m[2]}% (기준 ${m[3]}%)`],
  [/^TRIPLE_CROSS_(\d+)MIN_LIVE_DROP_([\d.]+)%_GT_([\d.]+)%$/, (m) => `${m[1]}분봉 종가보다 ${m[2]}% 하락 (한도 ${m[3]}%)`],
  [/^TRIPLE_CROSS_MISSING_(.+)$/, (m) => `트리플 크로스 미충족: ${m[1]}`],
  // --- hybrid 3-minute context ---
  [/^HYBRID_(\d+)MIN_CTX_LOW_ENTRY_ATR_([\d.]+)%_LT_([\d.]+)%$/, (m) => `${m[1]}분봉 변동성(ATR) ${m[2]}% < ${m[3]}% (너무 낮음)`],
  [/^HYBRID_(\d+)MIN_CTX_LOW_ABS_VOLUME_([\d,.]+)_LT_([\d,.]+)$/, (m) => `${m[1]}분봉 거래량 ${m[2]} < ${m[3]} (부족)`],
  [/^HYBRID_(\d+)MIN_CTX_LOW_VOL_MA_ABS_([\d,.]+)_LT_([\d,.]+)$/, (m) => `${m[1]}분봉 평균 거래량 ${m[2]} < ${m[3]} (부족)`],
  [/^HYBRID_(\d+)MIN_CTX_SCORE_(\d+)$/, (m) => `${m[1]}분 맥락 점수 ${m[2]}`],
  [/^HYBRID_(\d+)MIN_CTX_SCORE$/, (m) => `${m[1]}분 맥락 점수`],
  [/^HYBRID_(\d+)MIN_FRAME_UNAVAILABLE$/, (m) => `${m[1]}분봉 데이터 없음`],
  [/^HYBRID_(\d+)MIN_FRAME_STALE_last=(\S+?)_expected=(\S+)$/, (m) => `${m[1]}분봉 데이터 오래됨 (마지막 ${m[2]}, 기대 ${m[3]})`],
  // --- market / session / account ---
  [/^MARKET_REGIME_DOWN_(KOSPI|KOSDAQ)_(\d{6})_1MIN_MA(\d+)_MA(\d+)_NOT_ABOVE$/, (m) => `시장 하락 국면: ${m[1]}(${m[2]}) 1분봉 MA${m[3]}가 MA${m[4]} 아래`],
  [/^SESSION_OPEN_WARMUP$/, () => '장 시작 워밍업 (지표 준비 중)'],
  [/^INSUFFICIENT_BUYING_POWER_OR_BUDGET$/, () => '매수 가능 금액 또는 예산 부족'],
  [/^ORDERBOOK_ASK_THIN$/, () => '매도 호가 잔량 부족'],
  [/^can_trade_code_now=False$/, () => '지금은 이 종목 매매 불가 (시간대·종목 상태)'],
  [/^WAIT_NEXT_POLL_CONFIRM$/, () => '다음 폴링에서 재확인 대기'],
  [/^count=(\d+)\/(\d+)$/, (m) => `확인 ${m[1]}/${m[2]}`],
  [/^INTRABAR_RSI_([\d.]+)_OUT_OF_(\d+)-(\d+)$/, (m) => `봉 내 RSI ${m[1]} (${m[2]}~${m[3]} 범위 밖)`],
  [/^STALE_LIVE_PRICE$/, () => '현재가 오래됨'],
  [/^BOX_RANGE_HOLD_(.+)$/, (m) => `박스권 보류 (${m[1]})`],
  [/^INSUFFICIENT_BARS$/, () => '봉 부족'],
];

/** one token -> Korean (unknown tokens unchanged) */
export function reasonTokenKo(token: string): string {
  const t = token.trim();
  if (!t) return '';
  for (const [re, f] of RULES) {
    const m = re.exec(t);
    if (m) return f(m);
  }
  return t;
}

/** a reason code (tokens joined by '|' or '+') -> Korean, tokens joined by ' · ' */
export function reasonKo(reason: string | undefined | null): string {
  if (!reason) return '';
  const s = String(reason).trim();
  // "reason=X | count=1/2 | live=..." style: keep the first two fields only
  const fields = s.split(/\s*\|\s*/);
  const out: string[] = [];
  for (const f of fields) {
    const body = f.replace(/^reason=/, '');
    if (/^(live|bb_mid|bar|bars|signal)=/.test(body)) continue;
    for (const tok of body.split('+')) {
      const k = reasonTokenKo(tok);
      if (k) out.push(k);
    }
  }
  return out.join(' · ');
}
