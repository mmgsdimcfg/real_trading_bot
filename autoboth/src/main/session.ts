/*
 * Update log
 * - [2026-11-04] type=feat - DaySession: one trading day = one worker + byte offsets per file. pollOnce() lists the day folder,
 *   reads only the bytes appended since the last poll (main logs in name order, then the daily buy_sell log), and picks up new
 *   main logs created by an r003 restart. A file that got smaller restarts the whole day from offset 0
 * - [2026-11-07] type=feat - `replay` (past days): the worker keeps the lines so setCursor(ts) can show the day as it was at ts.
 *   setSessionOpen() re-reads the day with a new funnel start time (the cursor is kept)
 */

import { Worker } from 'node:worker_threads';
import createEngineWorker from './engineWorker?nodeWorker';
import type { ConditionsDoc, EvalEvent, FileProgress, Snapshot, Trade } from '../shared/types';
import { classifyLogFile } from '../shared/files';
import type { LogSource } from './sources';
import type { WorkerMsg } from './engineWorker';

const MAX_READ = 4 * 1024 * 1024;

export class DaySession {
  private worker: Worker;
  private offsets = new Map<string, number>();
  private sizes = new Map<string, number>();
  private names = new Set<string>();
  private reqId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  private busy: Promise<void> | null = null;
  stopped = false;
  lastReceivedAt?: number;
  workerError?: string;

  constructor(
    readonly date: string,
    readonly source: LogSource,
    private conditions: ConditionsDoc,
    private sessionOpen: string,
    readonly replay = false,
  ) {
    this.worker = createEngineWorker({});
    this.worker.on('message', (m: { id: number; data?: unknown; error?: string }) => {
      if (m.id === -1) {
        this.workerError = m.error;
        return;
      }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      if (m.error) p.reject(new Error(m.error));
      else p.resolve(m.data);
    });
    this.worker.on('error', (e) => (this.workerError = String(e)));
    this.post({ t: 'init', date, conditions, sessionOpen, replay });
  }

  private post(m: WorkerMsg, transfer?: ArrayBuffer[]): void {
    this.worker.postMessage(m, transfer as never);
  }

  private call<T>(m: { t: 'snapshot' } | { t: 'timeline'; code: string } | { t: 'cursor'; ts: number | null }): Promise<T> {
    const id = this.reqId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.post({ ...(m as object), id } as WorkerMsg);
    });
  }

  setConditions(doc: ConditionsDoc): void {
    this.conditions = doc;
    this.post({ t: 'conditions', conditions: doc });
  }

  /** One incremental pass. Concurrent calls share the running pass. */
  poll(): Promise<void> {
    if (!this.busy) this.busy = this.pollInner().finally(() => (this.busy = null));
    return this.busy;
  }

  private async pollInner(): Promise<void> {
    const list = await this.source.list();
    if (this.stopped) return;
    const ymd = this.date;
    const mains: string[] = [];
    let daily: string | undefined;
    for (const f of list) {
      const info = classifyLogFile(f.name, ymd);
      this.sizes.set(f.name, f.size);
      if (info.kind === 'main') mains.push(f.name);
      else if (info.kind === 'daily_buy_sell') daily = f.name;
      else if (info.kind === 'symbol' && info.code && !this.names.has(f.name)) {
        this.names.add(f.name);
        this.post({ t: 'name', code: info.code, name: info.name || '' });
      }
    }
    mains.sort();
    const order: [string, 'main' | 'bs'][] = mains.map((n) => [n, 'main']);
    if (daily) order.push([daily, 'bs']);

    // rotation / rewrite: any file smaller than what we already consumed -> start the day over
    for (const [name] of order) {
      if ((this.sizes.get(name) || 0) < (this.offsets.get(name) || 0)) {
        this.source.invalidate(name);
        this.restart();
        break;
      }
    }

    for (const [name, kind] of order) {
      const size = this.sizes.get(name) || 0;
      let off = this.offsets.get(name) || 0;
      while (off < size && !this.stopped) {
        const end = Math.min(size, off + MAX_READ);
        const buf = await this.source.read(name, off, end);
        if (!buf.length) break;
        const bytes = new Uint8Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
        this.post({ t: 'chunk', file: name, kind, bytes }, [bytes.buffer as ArrayBuffer]);
        off += buf.length;
        this.offsets.set(name, off);
        this.lastReceivedAt = Date.now();
      }
    }
  }

  private restart(): void {
    this.offsets.clear();
    this.names.clear();
    this.post({ t: 'init', date: this.date, conditions: this.conditions, sessionOpen: this.sessionOpen, replay: this.replay });
    this.cursorTs = null;
  }

  /** replay cursor (epoch ms) of the last setCursor - re-applied after a re-read */
  cursorTs: number | null = null;

  /** Show the day as it was at `ts` (null = whole day). Only for replay sessions. */
  async setCursor(ts: number | null): Promise<void> {
    if (!this.replay) return;
    this.cursorTs = ts;
    await this.call<boolean>({ t: 'cursor', ts });
  }

  get funnelOpen(): string {
    return this.sessionOpen;
  }

  /** New funnel start time: the worker starts over and the next poll() reads the day again (cached bytes for remote days). */
  async setSessionOpen(open: string): Promise<void> {
    if (open === this.sessionOpen) return;
    if (this.busy) await this.busy.catch(() => undefined);
    const keep = this.cursorTs;
    this.sessionOpen = open;
    this.restart();
    await this.poll();
    if (keep !== null) await this.setCursor(keep);
  }

  files(): FileProgress[] {
    return [...this.offsets.keys()].sort().map((name) => ({ name, size: this.sizes.get(name) || 0, offset: this.offsets.get(name) || 0 }));
  }

  async snapshot(): Promise<Snapshot | null> {
    const s = await this.call<Snapshot | null>({ t: 'snapshot' });
    if (s) s.files = this.files();
    return s;
  }

  timeline(code: string): Promise<{ events: EvalEvent[]; trades: Trade[] }> {
    return this.call({ t: 'timeline', code });
  }

  stop(): void {
    this.stopped = true;
    for (const p of this.pending.values()) p.reject(new Error('session stopped'));
    this.pending.clear();
    void this.worker.terminate();
  }
}
