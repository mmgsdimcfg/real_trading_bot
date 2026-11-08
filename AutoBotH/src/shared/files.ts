/*
 * Update log
 * - [2026-11-04] type=feat - File naming of runtime/log/YYYYMMDD/ (guide 2.1)
 * - [2026-11-05] type=fix - Per-symbol files are now named "{ymd}_{code}_{name}_buy_sell.txt" on the server; the "_buy_sell" suffix
 *   was taken as part of the name ("SK_buy_sell"). A trailing _buy_sell / _buy / _sell is dropped from the name
 */

export type LogFileKind = 'main' | 'run_buy_sell' | 'daily_buy_sell' | 'symbol' | 'other';

export interface LogFileInfo {
  kind: LogFileKind;
  code?: string;
  name?: string;
}

export function classifyLogFile(file: string, ymd: string): LogFileInfo {
  if (/^r003_trade_live_execute_\d{8}_\d{6}\.log$/.test(file)) return { kind: 'main' };
  if (/^r003_trade_live_execute_\d{8}_\d{6}_buy_sell\.log$/.test(file)) return { kind: 'run_buy_sell' };
  if (file === `${ymd}_buy_sell.log`) return { kind: 'daily_buy_sell' };
  const m = /^\d{8}_(\d{6})_(.+)\.txt$/.exec(file);
  if (m) return { kind: 'symbol', code: m[1], name: m[2].replace(/_(?:buy_sell|buy|sell)$/i, '') || undefined };
  return { kind: 'other' };
}
