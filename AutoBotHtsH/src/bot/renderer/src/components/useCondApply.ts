/*
 * Update log
 * - [2026-11-27] type=refactor - The whole-condition switch of a buy condition (server r001 BUY_CONDITION_OFF) taken out of
 *   RuleSettings so the pipeline node toggle uses the very same confirm text / request / result message
 */

import { useState } from 'react';
import { useStore } from '../store';
import { BUY_CONDITION_SWITCH_KEY, buyConditionSwitch } from '../../../shared/control';
import type { Condition } from '../../../shared/types';

export function useCondApply(): { busy: boolean; msg: string; setCondApply: (c: Condition, apply: boolean) => Promise<void> } {
  const bumpHistory = useStore((s) => s.bumpHistory);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  // switch a whole buy condition on / off (the server refuses a required one)
  const setCondApply = async (c: Condition, apply: boolean) => {
    const head = `${String(c.no).padStart(2, '0')} ${c.title}`;
    const text = apply
      ? `${head}\n\n이 조건을 다시 사용합니다 (적용 해제 목록에서 뺌).\n조건 안의 설정값이 꺼져 있으면 켜도 아무것도 막지 않을 수 있습니다 - 저장 후 "적용 여부" 표시를 확인하세요.`
      : `${head}\n\n이 조건을 적용하지 않습니다 - 규칙 매매가 이 조건 때문에 매수를 막는 일이 없어집니다 (실제 매수가 늘 수 있습니다).` +
        (c.prep ? '\n뒤 조건이 쓰는 값은 계속 계산하고 차단만 하지 않습니다.' : '');
    if (!window.confirm(`${text}\n\nr003 재시작 후 적용됩니다. 바꿀까요?`)) return;
    setBusy(true);
    setMsg(`${head}: 저장 중…`);
    try {
      // [Codex round 3] the server switches this ONE number inside its write lock, from the list stored at that moment
      const r = await window.api.ctlSet(buyConditionSwitch(c.no, apply));
      const rej = r.data?.rejected?.[BUY_CONDITION_SWITCH_KEY];
      if (!r.ok || rej) setMsg(`${head}: 바꾸지 못했습니다 - ${rej || r.error || '알 수 없는 오류'}`);
      else {
        bumpHistory();
        const fresh = await window.api.refreshConditions();
        // [Codex round 4 / 5] a re-read that fell back to the list without the server's live state is not a confirmed result:
        // say so instead of "saved and shown"
        setMsg(
          fresh.ok && fresh.live === true
            ? `${head}: ${apply ? '사용' : '적용 해제'}으로 저장했습니다 - r003 재시작 후 적용됩니다.`
            : `${head}: 저장했지만 서버에서 목록을 다시 읽지 못했습니다${fresh.error ? ` (${fresh.error})` : ''} - 표시는 최신이 아닐 수 있습니다. 창을 다시 열어 확인하세요.`,
        );
      }
    } finally {
      setBusy(false);
    }
  };
  return { busy, msg, setCondApply };
}
