/*
 * Update log
 * - [2026-11-09] type=fix - RSI / ADX tooltips name the calculation method (MTS / Wilder) and the ADX guide line (MTS 20, Wilder 25)
 * - [2026-11-06] type=feat - Indicator check boxes shared by the timeline price chart and the chart tab: overlays (MA, Bollinger,
 *   Ichimoku, VWAP - intraday only) and panes (RSI, MACD, ADX, stochastic, Williams %R, MFI, OBV); tooltips name the bot constants
 *   whose periods are used
 * - [2026-11-06] type=feat - (HtsH copy) tooltips from the chart's current indicator periods (no bot wording)
 * - [2026-11-06] type=feat - (HtsH) tooltips of the new indicators (Envelope, price channel, Parabolic SAR, CCI, ATR, ROC, TRIX,
 *   disparity, psychological line)
 * - [2026-11-20] type=feat - 투자자 누적 (개인 / 외국인 / 기관 cumulative net buying; day / week / month / year charts only)
 */

import { HTS_PARAMS, INDICATOR_LABEL, MA_PERIODS, OSCILLATORS, OVERLAYS, type IndicatorKey, type IndicatorParams } from '../../../shared/indicators';
import type { ChartKind } from '../../../shared/price';

export function indicatorTitle(k: IndicatorKey, P: IndicatorParams = HTS_PARAMS): string {
  switch (k) {
    case 'ma': return `이동평균선 ${MA_PERIODS.join('/')} (단순)`;
    case 'bb': return `볼린저 밴드 ${P.bbPeriod}, ${P.bbMult}σ`;
    case 'ichimoku': return `일목균형표 전환 ${P.ichiTenkan} / 기준 ${P.ichiKijun} / 선행B ${P.ichiSenkouB}, 선행 ${P.ichiShift}봉`;
    case 'vwap': return 'VWAP - 당일 누적 (종가×거래량) / 거래량 (분봉·틱봉만)';
    case 'rsi': return P.rsiMethod === 'wilder' ? `RSI ${P.rsiPeriod} (Wilder), 시그널 ${P.rsiSignal} (단순평균)` : `RSI ${P.rsiPeriod} (MTS 방식 - 단순평균), 시그널 ${P.rsiSignal} (지수평균), 시그널선 교차 = 매수/매도`;
    case 'macd': return `MACD ${P.macdFast}/${P.macdSlow}/${P.macdSignal}`;
    case 'adx': return `ADX ${P.adxPeriod} + DI+/DI- (${P.adxMethod === 'wilder' ? 'Wilder' : 'MTS 방식 - 지수평균'}), 기준선 ${P.adxMethod === 'wilder' ? 25 : 20}`;
    case 'stoch': return `스토캐스틱 %K ${P.stochK} / %D ${P.stochD}`;
    case 'williams': return `Williams %R ${P.williamsR} / %D ${P.williamsD}`;
    case 'mfi': return `MFI ${P.mfiPeriod}`;
    case 'obv': return `OBV - 당일(세션) 누적, MA ${P.obvMa}`;
    case 'env': return `엔벨로프 ${P.envPeriod}일 이동평균 ±${P.envPct}%`;
    case 'donchian': return `가격채널 - 직전 ${P.donchian}봉 최고가 / 최저가`;
    case 'sar': return `Parabolic SAR 가속 ${P.sarStep} / 최대 ${P.sarMax} (상승 추세 빨강 점, 하락 추세 파랑 점)`;
    case 'cci': return `CCI ${P.cciPeriod}, 기준선 ±100`;
    case 'atr': return `ATR ${P.atrPeriod} (Wilder) - 변동성`;
    case 'roc': return `ROC ${P.rocPeriod}봉 변화율(%), 시그널 ${P.rocSignal}`;
    case 'trix': return `TRIX ${P.trixPeriod} (3중 지수이동평균 변화율), 시그널 ${P.trixSignal}`;
    case 'disparity': return `이격도 - 종가 / ${P.disparity}일 이동평균 × 100`;
    case 'psy': return `심리도 - 최근 ${P.psyPeriod}봉 중 상승 봉 비율(%), 기준 25 / 75`;
    case 'investor': return '투자자 누적 순매수(주) - 개인 / 외국인 / 기관, 1개월·3개월·6개월·1년 중 선택 (일·주·월·년봉만)';
  }
}

export function IndicatorToggles({ value, onChange, kind, keys, params }: { value: IndicatorKey[]; onChange: (v: IndicatorKey[]) => void; kind: ChartKind; keys?: IndicatorKey[]; params?: IndicatorParams }) {
  const order = [...OVERLAYS, ...OSCILLATORS].filter((k) => !keys || keys.includes(k));
  const intraday = kind === 'tick' || kind === 'min';
  return (
    <span className="ind-toggles">
      {order.map((k, j) => (
        <label key={k} className={`check ${j > 0 && OSCILLATORS.includes(k) && OVERLAYS.includes(order[j - 1]) ? 'sep' : ''}`} title={indicatorTitle(k, params)}>
          <input
            type="checkbox"
            disabled={(k === 'vwap' && !intraday) || (k === 'investor' && intraday)}
            checked={value.includes(k)}
            onChange={(e) => onChange(e.target.checked ? order.filter((x) => x === k || value.includes(x)) : value.filter((x) => x !== k))}
          />{' '}
          {INDICATOR_LABEL[k]}
        </label>
      ))}
    </span>
  );
}
