/*
 * Update log
 * - [2026-11-06] type=feat - Indicator check boxes shared by the timeline price chart and the chart tab: overlays (MA, Bollinger,
 *   Ichimoku, VWAP - intraday only) and panes (RSI, MACD, ADX, stochastic, Williams %R, MFI, OBV); tooltips name the bot constants
 *   whose periods are used
 * - [2026-11-07] type=feat - 거래량 (volume) check box between the overlays and the panes
 */

import { BOT_PARAMS as P, INDICATOR_LABEL, MA_PERIODS, OSCILLATORS, OVERLAYS, VOLUME, type IndicatorKey } from '../../../shared/indicators';
import type { ChartKind } from '../../../shared/price';

export const INDICATOR_TITLE: Record<IndicatorKey, string> = {
  ma: `이동평균선 ${MA_PERIODS.join('/')} (단순, 차트 표준)`,
  bb: `볼린저 밴드 ${P.bbPeriod}, ${P.bbMult}σ (봇 BB_PERIOD / BB_STD_MULTIPLIER)`,
  ichimoku: `일목균형표 ${P.ichiTenkan}/${P.ichiKijun}/${P.ichiSenkouB}, 선행 ${P.ichiShift}봉 (봇은 사용하지 않음 - 표준값)`,
  vwap: 'VWAP - 당일 누적 (종가×거래량) / 거래량, 봇과 같은 식 (분봉·틱봉만)',
  vol: '거래량 막대 (가격 칸 아래)',
  rsi: `RSI ${P.rsiPeriod} (Wilder), 시그널 ${P.rsiSignal} (봇 RSI_PERIOD / RSI_SIGNAL_PERIOD)`,
  macd: `MACD ${P.macdFast}/${P.macdSlow}/${P.macdSignal} (봇 MACD_FAST / MACD_SLOW / MACD_SIGNAL_PERIOD)`,
  adx: `ADX ${P.adxPeriod} + DI+/DI- (봇 ADX_PERIOD), 기준선 25 = ADX_BUY_MIN`,
  stoch: `스토캐스틱 %K ${P.stochK} / %D ${P.stochD} (봇 STOCH_K_PERIOD / STOCH_D_PERIOD)`,
  williams: `Williams %R ${P.williamsR} / %D ${P.williamsD} (봇 WILLIAMS_R_PERIOD / WILLIAMS_D_PERIOD)`,
  mfi: `MFI ${P.mfiPeriod} (봇 MFI_PERIOD)`,
  obv: `OBV - 당일 누적, MA ${P.obvMa} (봇 OBV_MA_PERIOD)`,
};

export function IndicatorToggles({ value, onChange, kind, keys }: { value: IndicatorKey[]; onChange: (v: IndicatorKey[]) => void; kind: ChartKind; keys?: IndicatorKey[] }) {
  const order = [...OVERLAYS, VOLUME, ...OSCILLATORS].filter((k) => !keys || keys.includes(k));
  const intraday = kind === 'tick' || kind === 'min';
  return (
    <span className="ind-toggles">
      {order.map((k, j) => (
        <label key={k} className={`check ${j > 0 && !OVERLAYS.includes(k) && OVERLAYS.includes(order[j - 1]) ? 'sep' : ''}`} title={INDICATOR_TITLE[k]}>
          <input
            type="checkbox"
            disabled={k === 'vwap' && !intraday}
            checked={value.includes(k)}
            onChange={(e) => onChange(e.target.checked ? order.filter((x) => x === k || value.includes(x)) : value.filter((x) => x !== k))}
          />{' '}
          {INDICATOR_LABEL[k]}
        </label>
      ))}
    </span>
  );
}
