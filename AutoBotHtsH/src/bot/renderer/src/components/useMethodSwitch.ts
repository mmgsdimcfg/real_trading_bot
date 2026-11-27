/*
 * Update log
 * - [2026-11-27] type=refactor - The trading-method on / off handlers of the control tab (규칙 매매, Edge AI 매매 + its real orders,
 *   VWAP 매매 + its real orders), moved here unchanged so the method board (MethodView, user 2026-11-27: switch a method on / off
 *   on that page) uses the very same checks, warnings and confirmations. Every change goes through r016 (window.api.ctlSet) and
 *   is applied by an r003 restart - nothing is ordered or judged here.
 * - [2026-11-28] type=feat - every handler resolves true when the change was stored (the method board chains "method on" and
 *   "real orders on" for its paper / real cards and must stop when the first step was cancelled)
 */

import { useStore } from '../store';
import {
  AUTO_SELL_KO,
  autoSellOf,
  comboChanges,
  EDGE_LIVE_KEY,
  EDGE_LIVE_ON_WARNING,
  TRADE_KIND_KO,
  tradeModeOf,
  VWAP_LIVE_KEY,
  VWAP_LIVE_ON_WARNING,
  VWAP_MODE_KEY,
  VWAP_STEPS,
  type EdgeModelInfo,
} from '../../../shared/paper';

export type ServerKind = 'rules' | 'paper';

export interface MethodSwitch {
  /** stored settings (what the switches show; the running bot may differ until its restart) */
  rulesOn: boolean;
  edgeOn: boolean;
  edgeLiveOn: boolean;
  vwapOn: boolean;
  vwapLiveOn: boolean;
  /** the server knows the switch (false: older server / state not read yet - the control is locked) */
  has: { rules: boolean; edge: boolean; edgeLive: boolean; vwap: boolean; vwapLive: boolean };
  /** TRADING_STRATEGY_MODE is the rule-based one (normal): the rule / Edge AI switches apply */
  normalMode: boolean;
  /** each resolves true when the change was stored (false: cancelled, refused or not needed) */
  setServerKind: (kind: ServerKind, on: boolean) => Promise<boolean>;
  setEdgeLive: (on: boolean) => Promise<boolean>;
  setVwap: (on: boolean) => Promise<boolean>;
  setVwapLive: (on: boolean) => Promise<boolean>;
}

/** `edgeModel`: undefined = not loaded (the server decides whether real Edge AI orders may be switched on) */
export function useMethodSwitch(opts: { setBusy: (b: boolean) => void; edgeModel: EdgeModelInfo | null | undefined }): MethodSwitch {
  const control = useStore((s) => s.control);
  const bumpHistory = useStore((s) => s.bumpHistory);
  const setModeBusy = opts.setBusy;
  const edgeModel = opts.edgeModel;
  const paper = control.paper;
  const strategy = control.strategy;
  const tradeMode = tradeModeOf(strategy?.effective, paper?.effective);
  const ruleBuy = control.ruleBuy;
  const normalMode = !strategy || tradeMode === 'rules' || tradeMode === 'rules+paper';
  const rulesOn = ruleBuy ? ruleBuy.effective === 'true' : true;
  const paperOn = tradeMode === 'rules+paper';
  const vwapMode = control.vwapMode;
  const vwapOn = vwapMode?.effective === 'paper';
  const vwapLive = control.vwapLive;
  const vwapLiveOn = vwapOn && vwapLive?.effective === 'true';
  const vwapLimits = control.vwapLiveLimits;
  const autoSellMode = autoSellOf(control.autoSell?.effective);
  const edgeLive = control.edgeLive;
  const liveOn = edgeLive ? edgeLive.effective === 'true' : false;
  const liveLimits = control.edgeLiveLimits;
  const modelNotPromoted = edgeModel !== undefined && !edgeModel?.promoted;

  const setVwapLive = async (on: boolean): Promise<boolean> => {
    if (!vwapLive) return false;
    if (on && autoSellMode !== 'all') {
      window.alert('VWAP 매매 실제 매매는 자동 매도가 "전체 자동"일 때만 켤 수 있습니다 (신규 VWAP 매수의 조건).');
      return false;
    }
    const lim = vwapLimits;
    const limText = lim
      ? `1회 ${lim.order.effective.toLocaleString('ko-KR')}원 · 동시 ${lim.positions.effective}종목 · 하루 ${lim.entries.effective}회 · 하루 손실 ${lim.loss.effective.toLocaleString('ko-KR')}원`
      : '서버 기본 한도';
    const text = on
      ? `${VWAP_LIVE_ON_WARNING}\n\n한도: ${limText}\n` +
        '주문: 정규장 14:50 까지, 매도1호가 지정가 1회 (재주문 없음). 청산: 위 청산 기준 / 15:20.\n' +
        'VWAP 매매가 산 수량은 VWAP 매매 규칙으로만 팔고, 같은 종목의 직접 매수분은 팔지 않습니다.\n' +
        '켜져 있는 동안 VWAP 가상 매매는 새로 사지 않습니다.'
      : 'VWAP 매매 실제 매매를 끕니다 (가상 매매로 돌아갑니다).\n이미 VWAP 매매가 산 수량이 있으면 그 청산(손절 / 익절 / 시간 / 15:20)은 계속됩니다.';
    const restart = 'r003 재시작 후부터 적용됩니다. 미체결 주문이 있으면 정리된 뒤(또는 장 마감 후) 재시작하세요.';
    if (!window.confirm(`VWAP 매매 실제 매매: ${on ? '켬' : '끔'}\n\n${text}\n\n${restart}\n\n바꿀까요?`)) return false;
    if (on && !window.confirm('한 번 더 확인합니다.\n\n실제 계좌로 VWAP 매매 매수 주문이 나갑니다. 켤까요?')) return false;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet({ [VWAP_LIVE_KEY]: on });
      const rej = r.data?.rejected?.[VWAP_LIVE_KEY];
      if (!r.ok || rej) window.alert(`VWAP 매매 실제 주문 설정을 바꾸지 못했습니다: ${rej || r.error || '알 수 없는 오류'}`);
      else {
        bumpHistory();
        return true;
      }
    } finally {
      setModeBusy(false);
    }
    return false;
  };
  const setVwap = async (on: boolean): Promise<boolean> => {
    if (!vwapMode || on === vwapOn) return false;
    const text = on
      ? 'VWAP 매매를 켭니다 - 가상 매매로 시작합니다 (실제 주문 없음, 가상 매매 탭의 "VWAP 매매" 장부). 실제 주문은 "실제 매매"를 따로 켜야 나갑니다.\n\n' +
        VWAP_STEPS.map(([a, b], i) => `${i + 1}. ${a}: ${b}`).join('\n') +
        '\n\n앞선 검증에서 VWAP 위치의 예측력은 확인되지 않았습니다. r003 재시작 후 적용됩니다. 켤까요?'
      : 'VWAP 매매를 끕니다. 신규 매수를 멈추고, 오늘 열린 가상 보유는 마감 청산까지 그대로 청산됩니다.' +
        (vwapLive?.effective === 'true' ? '\nVWAP 실제 매매도 함께 끕니다 (이미 산 수량의 청산은 계속됩니다).' : '') +
        ' r003 재시작 후 적용됩니다. 끌까요?';
    if (!window.confirm(text)) return false;
    setModeBusy(true);
    try {
      // the live orders go off with the trade itself (and it starts with them off): nothing stays on out of sight
      const r = await window.api.ctlSet({ ...(vwapLive?.effective === 'true' ? { [VWAP_LIVE_KEY]: false } : {}), [VWAP_MODE_KEY]: on ? 'paper' : 'off' });
      const rej = r.data?.rejected?.[VWAP_MODE_KEY];
      if (!r.ok || rej) window.alert(`VWAP 매매 설정을 바꾸지 못했습니다: ${rej || r.error || '알 수 없는 오류'}`);
      else {
        bumpHistory();
        return true;
      }
    } finally {
      setModeBusy(false);
    }
    return false;
  };
  const setServerKind = async (kind: ServerKind, on: boolean): Promise<boolean> => {
    const next = { rules: kind === 'rules' ? on : rulesOn, paper: kind === 'paper' ? on : paperOn };
    const changes = comboChanges(next, { rules: rulesOn, paper: paperOn }, !!ruleBuy);
    if (!changes) return false;
    // [2026-11-25] Edge AI 매매 off: its live orders go off with it (stored too - nothing stays on out of sight)
    // ... and on: it starts with the live orders off too - they are switched on by their own checkbox and its warning only
    if (kind === 'paper' && control.edgeLive?.effective === 'true') changes[EDGE_LIVE_KEY] = false;
    const sellMode = autoSellOf(control.autoSell?.effective);
    const text =
      kind === 'rules'
        ? on
          ? '규칙 매매를 켭니다. 규칙 매수 조건을 통과하면 실제 매수 주문을 냅니다.'
          : '규칙 매매를 끕니다. 규칙 매수 조건을 통과해도 신규 매수 주문을 내지 않습니다(로그만).\n' +
            `이미 보유한 종목의 매도는 자동 매도 설정(${AUTO_SELL_KO[sellMode]})대로 계속됩니다.` +
            (sellMode === 'all' ? '' : '\n자동 매도가 전체가 아니므로 보유 종목이 남을 수 있습니다 - 직접 매도해야 합니다.')
        : on
          ? 'Edge AI 신호로 가상 매매를 합니다 (실제 주문 없음, 가상 매매 탭).' +
            (control.edgeLive?.effective === 'true' ? '\n저장돼 있던 Edge AI 실제 매매 설정은 끈 상태로 시작합니다 (켜려면 "실제 매매" 체크).' : '')
          : '신규 가상 매수를 멈춥니다. 오늘 열린 가상 보유는 마감 청산까지 그대로 청산됩니다.' +
            (control.edgeLive?.effective === 'true'
              ? '\nEdge AI 실제 매매도 함께 끕니다 (이미 산 Edge 수량의 청산은 계속됩니다).'
              : '');
    const restart = 'r003 재시작 후부터 적용됩니다. 미체결 주문이 있으면 정리된 뒤(또는 장 마감 후) 재시작하세요 - 재시작하면 미체결 주문 추적이 끊깁니다.';
    if (!window.confirm(`${TRADE_KIND_KO[kind]}: ${on ? '켬' : '끔'}\n\n${text}\n\n${restart}\n\n바꿀까요?`)) return false;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet(changes);
      if (!r.ok) window.alert(`매매 방식을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
      else return true;
    } finally {
      setModeBusy(false);
    }
    return false;
  };
  const setEdgeLive = async (on: boolean): Promise<boolean> => {
    if (on && modelNotPromoted) {
      window.alert(
        'Edge AI 실제 매매를 켤 수 없습니다.\n\n' +
          (edgeModel ? `현재 모델(${edgeModel.modelId})이 승격 심사를 통과하지 못했습니다 (검증 구간에서 통과 진입의 순손익이 0보다 크지 않음).` : '배포된 Edge AI 모델이 없습니다.') +
          '\n실제 주문은 승격된 모델에서만 나갑니다. 모델은 매일 장 마감 후 다시 심사됩니다.\n가상 매매는 그대로 쓸 수 있습니다.',
      );
      return false;
    }
    if (on && autoSellMode !== 'all') {
      window.alert('Edge AI 실제 매매는 자동 매도가 "전체 자동"일 때만 켤 수 있습니다 (신규 Edge 매수의 조건).');
      return false;
    }
    const lim = liveLimits;
    const limText = lim
      ? `1회 ${lim.order.effective.toLocaleString('ko-KR')}원 · 동시 ${lim.positions.effective}종목 · 하루 ${lim.entries.effective}회 · 하루 손실 ${lim.loss.effective.toLocaleString('ko-KR')}원에서 중단`
      : '서버 기본 한도';
    const text = on
      ? `${EDGE_LIVE_ON_WARNING.replace(/\n\n켤까요\?$/, '')}\n\n` +
        `한도: ${limText}\n` +
        '주문: 정규장 14:50 까지, 매도1호가 지정가 1회 (재주문 없음). 청산: 아래 청산 기준 / 15:20.\n' +
        'Edge 가 산 수량은 Edge 규칙으로만 팔고, 같은 종목의 직접 매수분은 팔지 않습니다.\n' +
        '켜져 있는 동안 가상 매매는 후보만 기록합니다.'
      : 'Edge AI 실제 매매를 끕니다 (가상 매매는 아래 체크대로).\n이미 Edge 가 산 수량이 있으면 그 청산(손절 / 익절 / 시간 / 15:20)은 계속됩니다.';
    const restart = 'r003 재시작 후부터 적용됩니다. 미체결 주문이 있으면 정리된 뒤(또는 장 마감 후) 재시작하세요.';
    if (!window.confirm(`Edge AI 실제 매매: ${on ? '켬' : '끔'}\n\n${text}\n\n${restart}\n\n바꿀까요?`)) return false;
    if (on && !window.confirm('한 번 더 확인합니다.\n\n실제 계좌로 Edge AI 매수 주문이 나갑니다. 켤까요?')) return false;
    setModeBusy(true);
    try {
      const r = await window.api.ctlSet({ [EDGE_LIVE_KEY]: on });
      if (!r.ok) window.alert(`Edge AI 실제 주문 설정을 바꾸지 못했습니다: ${r.error || JSON.stringify(r.data?.rejected || {})}`);
      else return true;
    } finally {
      setModeBusy(false);
    }
    return false;
  };
  return {
    rulesOn,
    edgeOn: paperOn,
    edgeLiveOn: liveOn,
    vwapOn,
    vwapLiveOn,
    has: { rules: !!ruleBuy, edge: !!paper && !!strategy, edgeLive: !!edgeLive, vwap: !!vwapMode, vwapLive: !!vwapLive },
    normalMode,
    setServerKind,
    setEdgeLive,
    setVwap,
    setVwapLive,
  };
}
