/*
 * Update log
 * - [2026-11-07] type=feat - Time replay of a past day. DayFeed keeps every complete line of the day (main logs in file-name order,
 *   then the daily buy_sell log) with the KST time of its prefix, next to the full DayEngine. setCursor(ts) builds a second
 *   DayEngine fed only with the lines logged at or before ts: moving forward feeds just the new lines, moving back rebuilds from
 *   the first line. Lines without a time prefix (continuations) take the time of the line before them. Pure TypeScript (worker +
 *   vitest); the server is not involved - it works on the bytes the session already read
 */

import { DayEngine, type EngineOptions } from './engine';
import { kstToEpoch } from './time';
import type { ConditionsDoc, EvalEvent, ReplayInfo, Snapshot, Trade } from './types';
import type { Bar } from './price';

/** Replay: intraday (tick / minute) bars that started after the cursor are not shown yet. Daily bars stay as they are. */
export function barsUpTo<T extends Pick<Bar, 't'>>(bars: T[], cursor: number | null | undefined, intraday: boolean): T[] {
  if (cursor === null || cursor === undefined || !intraday) return bars;
  let n = bars.length;
  while (n > 0 && bars[n - 1].t > cursor) n--;
  return n === bars.length ? bars : bars.slice(0, n);
}

// "2026-11-04 09:08:12,593 [INFO] ..." (main) / "2026-11-04 09:09:26 [INFO] ..." (buy_sell)
const PREFIX_RE = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(?:,(\d{3}))?/;

class LineStore {
  readonly lines: string[] = [];
  readonly ts: number[] = [];
  private last = 0;

  push(lines: string[]): void {
    for (const raw of lines) {
      const m = PREFIX_RE.exec(raw);
      if (m) this.last = kstToEpoch(m[1], m[2], m[3] ? +m[3] : 0);
      this.lines.push(raw);
      this.ts.push(this.last);
    }
  }

  /** index of the first line after `cursor`, scanning from `from` (lines are in log order; a rare earlier stamp is fed late) */
  upto(from: number, cursor: number): number {
    let i = from;
    while (i < this.ts.length && this.ts[i] <= cursor) i++;
    return i;
  }

  first(): number | undefined {
    return this.ts.find((t) => t > 0);
  }

  /** times of the lines whose text after the prefix matches `re` */
  marks(re: RegExp): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.lines.length; i++) if (this.ts[i] > 0 && re.test(this.lines[i])) out.push(this.ts[i]);
    return out;
  }
}

// buy_sell block line "... [INFO] >>> 159010_아스플로 Buy - reason=..."
const BS_BLOCK_LINE_RE = /\]\s*>>> \d{6}_\S*\s+(?:Buy|Sell) - /;

export class DayFeed {
  private full: DayEngine;
  private view: DayEngine | null = null;
  private cursor: number | null = null;
  private mainPos = 0;
  private bsPos = 0;
  private readonly main?: LineStore;
  private readonly bs?: LineStore;
  private readonly names: [string, string][] = [];
  private tradeMarks: number[] = [];
  private marksAt = -1;

  constructor(
    readonly date: string,
    private conditions: ConditionsDoc,
    private opts: EngineOptions,
    /** keep the lines for setCursor (past days); off for the live day */
    readonly replay: boolean,
  ) {
    this.full = new DayEngine(date, conditions, opts);
    if (replay) {
      this.main = new LineStore();
      this.bs = new LineStore();
    }
  }

  setConditions(doc: ConditionsDoc): void {
    this.conditions = doc;
    this.full.setConditions(doc);
    this.view?.setConditions(doc);
  }

  setName(code: string, name: string): void {
    this.names.push([code, name]);
    this.full.setName(code, name);
    this.view?.setName(code, name);
  }

  feedMain(lines: string[]): void {
    this.full.feedMainLines(lines);
    this.main?.push(lines);
  }

  feedBuySell(lines: string[]): void {
    this.full.feedBuySellLines(lines);
    this.bs?.push(lines);
  }

  /** null = the whole day (no replay view). Ignored when the feed keeps no lines. */
  setCursor(ts: number | null): void {
    if (!this.main || !this.bs) return;
    if (ts === null || !Number.isFinite(ts)) {
      this.view = null;
      this.cursor = null;
      return;
    }
    if (!this.view || this.cursor === null || ts < this.cursor) {
      this.view = new DayEngine(this.date, this.conditions, this.opts);
      for (const [c, n] of this.names) this.view.setName(c, n);
      this.mainPos = 0;
      this.bsPos = 0;
    }
    this.cursor = ts;
    const m = this.main.upto(this.mainPos, ts);
    if (m > this.mainPos) this.view.feedMainLines(this.main.lines.slice(this.mainPos, m));
    this.mainPos = m;
    const b = this.bs.upto(this.bsPos, ts);
    if (b > this.bsPos) this.view.feedBuySellLines(this.bs.lines.slice(this.bsPos, b));
    this.bsPos = b;
  }

  replayInfo(): ReplayInfo | undefined {
    if (!this.main || !this.bs) return undefined;
    const firsts = [this.main.first(), this.bs.first()].filter((t): t is number => t !== undefined);
    const lasts = [this.main.ts[this.main.ts.length - 1], this.bs.ts[this.bs.ts.length - 1]].filter((t): t is number => !!t);
    if (!firsts.length || !lasts.length) return undefined;
    if (this.marksAt !== this.bs.lines.length) {
      this.tradeMarks = this.bs.marks(BS_BLOCK_LINE_RE);
      this.marksAt = this.bs.lines.length;
    }
    return { first: Math.min(...firsts), last: Math.max(...lasts), cursor: this.cursor, trades: this.tradeMarks };
  }

  private engine(): DayEngine {
    return this.view || this.full;
  }

  snapshot(): Snapshot {
    const s = this.engine().snapshot();
    const r = this.replayInfo();
    if (r) s.replay = r;
    return s;
  }

  timeline(code: string): { events: EvalEvent[]; trades: Trade[] } {
    return this.engine().timeline(code);
  }
}
