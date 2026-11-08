/*
 * Update log
 * - [2026-11-05] type=feat - editItemsFor(): value-editor rows of a condition from values + values_meta (ENABLE_* first; keys without
 *   meta - r015/bundled copies or editable:false - are read-only), nodeEnableKey() for the node-header toggle
 * - [2026-11-04] type=feat - Renderer helpers: condition lookup by "_NNN" key, labels of the pseudo nodes, number formats
 */

import type { Condition, ConditionsDoc, Side } from '../../shared/types';
import type { ValueType } from '../../shared/control';
import type { EditItem } from './components/ValuesEditor';
import { condKeyOf } from '../../shared/mapping';

export const PSEUDO_LABEL: Record<string, string> = {
  PREP: '프레임 준비 (지표 워밍업)',
  ORDER: '주문',
  SESSION_CLOSE: '세션 종료 강제 청산',
  UNKNOWN: '알 수 없음',
};

export const STAGE_LABEL: Record<string, string> = {
  PREP: '준비',
  STATE: 'STATE · 상태',
  TRIGGER: 'TRIGGER · 진입 신호',
  CONTEXT: 'CONTEXT · 3분 컨텍스트',
  PRE_ORDER: 'PRE_ORDER · 주문 직전',
  GUARD: 'GUARD · 가드',
  EXIT: 'EXIT · 청산 조건',
  ORDER: 'ORDER · 주문',
};

export function sortedConds(doc: ConditionsDoc | null, side: Side): Condition[] {
  if (!doc) return [];
  return [...(side === 'buy' ? doc.buy : doc.sell)].sort((a, b) => a.order - b.order);
}

export function condIndex(doc: ConditionsDoc | null, side: Side): Map<string, Condition> {
  return new Map(sortedConds(doc, side).map((c) => [condKeyOf(c.no), c]));
}

export function condTitle(doc: ConditionsDoc | null, side: Side, key: string | undefined): string {
  if (!key) return '-';
  if (PSEUDO_LABEL[key]) return PSEUDO_LABEL[key];
  const c = condIndex(doc, side).get(key);
  return c ? `${key} ${c.title}` : `${key} (조건 목록에 없음)`;
}

export const fmtInt = (n: number | undefined) => (n === undefined || !Number.isFinite(n) ? '-' : Math.round(n).toLocaleString('ko-KR'));
export const fmtKrw = (n: number | undefined) => (n === undefined || !Number.isFinite(n) ? '-' : `${n > 0 ? '+' : ''}${Math.round(n).toLocaleString('ko-KR')}원`);
export const fmtPct = (n: number | undefined) => (n === undefined || !Number.isFinite(n) ? '-' : `${n > 0 ? '+' : ''}${n.toFixed(2)}%`);
export const fmtBytes = (n: number) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`);

/** Korean market convention: gain = red, loss = blue */
export const pnlClass = (n: number | undefined) => (!n || !Number.isFinite(n) ? '' : n > 0 ? 'gain' : 'loss');

function guessType(v: unknown): ValueType {
  if (typeof v === 'boolean') return 'bool';
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'float';
  if (typeof v === 'string') return 'str';
  if (Array.isArray(v)) return 'list';
  if (v === null || v === undefined) return 'null';
  return 'dict';
}

export const isEnableItem = (it: Pick<EditItem, 'key' | 'type'>) => it.key.startsWith('ENABLE_') && it.type === 'bool';

/** Editor rows of one condition: ENABLE_* toggles first, then the other values in export order. */
export function editItemsFor(cond: Condition | undefined): EditItem[] {
  if (!cond) return [];
  const values = cond.values || {};
  const meta = cond.values_meta || {};
  const keys = [...new Set([...Object.keys(values), ...Object.keys(meta)])];
  const items: EditItem[] = keys.map((key) => {
    const m = meta[key];
    const v = values[key];
    if (!m) return { key, type: guessType(v), base: v, override: null, has_override: false, pending: false, editable: false, current: v };
    return { key, type: m.type, base: m.base, override: m.override, has_override: m.has_override, pending: m.pending, editable: m.editable !== false, current: v };
  });
  return [...items.filter(isEnableItem), ...items.filter((i) => !isEnableItem(i))];
}

/** The ENABLE_* key mirrored on the node header (first editable one). */
export function nodeEnableKey(cond: Condition | undefined): EditItem | undefined {
  return editItemsFor(cond).find((i) => isEnableItem(i) && i.editable);
}
