/*
 * Update log
 * - [2026-11-04] type=feat - Worker thread that owns the DayEngine: the main process only moves bytes (SFTP reads) and the line
 *   splitting/regex/aggregation of 5-10 MB a day runs here, so the window never stalls during the first full parse
 */

import { parentPort } from 'node:worker_threads';
import { DayEngine } from '../shared/engine';
import { LineBuffer } from '../shared/lineBuffer';
import type { ConditionsDoc } from '../shared/types';

export type WorkerMsg =
  | { t: 'init'; date: string; conditions: ConditionsDoc; sessionOpen: string }
  | { t: 'conditions'; conditions: ConditionsDoc }
  | { t: 'chunk'; file: string; kind: 'main' | 'bs'; bytes: Uint8Array }
  | { t: 'name'; code: string; name: string }
  | { t: 'snapshot'; id: number }
  | { t: 'timeline'; id: number; code: string };

let engine: DayEngine | null = null;
let buffers = new Map<string, LineBuffer>();

parentPort!.on('message', (m: WorkerMsg) => {
  try {
    switch (m.t) {
      case 'init':
        engine = new DayEngine(m.date, m.conditions, { sessionOpen: m.sessionOpen });
        buffers = new Map();
        break;
      case 'conditions':
        engine?.setConditions(m.conditions);
        break;
      case 'name':
        engine?.setName(m.code, m.name);
        break;
      case 'chunk': {
        if (!engine) return;
        let lb = buffers.get(m.file);
        if (!lb) buffers.set(m.file, (lb = new LineBuffer()));
        const lines = lb.push(m.bytes);
        if (m.kind === 'main') engine.feedMainLines(lines);
        else engine.feedBuySellLines(lines);
        break;
      }
      case 'snapshot':
        parentPort!.postMessage({ id: m.id, data: engine ? engine.snapshot() : null });
        break;
      case 'timeline':
        parentPort!.postMessage({ id: m.id, data: engine ? engine.timeline(m.code) : { events: [], trades: [] } });
        break;
    }
  } catch (e) {
    if ('id' in m) parentPort!.postMessage({ id: m.id, error: String((e as Error)?.stack || e) });
    else parentPort!.postMessage({ id: -1, error: String((e as Error)?.stack || e) });
  }
});
