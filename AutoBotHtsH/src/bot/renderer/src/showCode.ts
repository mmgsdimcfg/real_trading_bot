/*
 * Update log
 * - [2026-11-20] type=feat - "show this symbol elsewhere" for a double click on a paper-trading row (user 2026-11-20). Same file in
 *   AutoBotH and AutoBotHtsH: AutoBotHtsH sets a handler (its chart / current price / stock info / news windows show the code);
 *   without one (AutoBotH) the 차트 tab opens with the code
 */

import { useStore } from './store';

let handler: ((code: string) => void) | null = null;

export function setShowCodeHandler(h: ((code: string) => void) | null): void {
  handler = h;
}

export function showCode(code: string): void {
  if (!/^[0-9A-Z]{6}$/.test(code)) return;
  if (handler) handler(code);
  else useStore.getState().openChart(code);
}
