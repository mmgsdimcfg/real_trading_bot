// Update log
// - [2026-11-22] type=feat (Rebound alert history (user 2026-11-22): every alert that was shown is kept in
//   %APPDATA%\<app>\rebound\history.json (newest first, at most HISTORY_MAX) with everything the popup showed, whether the user has
//   seen it (read) and what happened to its KakaoTalk message. An alert is "read" when its popup was closed by hand or opened, or
//   when it was looked at / marked in the settings dialog; a popup that closed by itself leaves it unread - the title bar counts those)

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'fs';
import { join } from 'path';
import { reboundDir, type AlertInfo } from './reboundAlert';

export const HISTORY_MAX = 500;

export interface HistoryItem extends AlertInfo {
  id: string;
  read: boolean;
  /** KakaoTalk message of this alert: '' not sent (off), 'sent', 'failed' (+ kakaoError) */
  kakao: '' | 'sent' | 'failed';
  kakaoError?: string;
}

export class ReboundHistory {
  private items: HistoryItem[] | null = null;
  private seq = 0;

  constructor(private onChange: () => void = () => undefined) {}

  private file(): string {
    return join(reboundDir(), 'history.json');
  }

  private load(): HistoryItem[] {
    if (this.items) return this.items;
    try {
      const v = JSON.parse(readFileSync(this.file(), 'utf8').replace(/^﻿/, '')) as unknown;
      this.items = Array.isArray(v) ? (v as HistoryItem[]).filter((x) => x && typeof x.id === 'string' && x.res && x.res.snap) : [];
    } catch {
      this.items = [];
    }
    return this.items;
  }

  /** written to a temp file and renamed, so a crash never leaves half a file */
  private save(): void {
    try {
      if (!existsSync(reboundDir())) mkdirSync(reboundDir(), { recursive: true });
      const tmp = `${this.file()}.tmp`;
      writeFileSync(tmp, JSON.stringify(this.items || []), 'utf8');
      renameSync(tmp, this.file());
    } catch {
      /* kept in memory for this run */
    }
    this.onChange();
  }

  list(): HistoryItem[] {
    return this.load();
  }

  get(id: string): HistoryItem | undefined {
    return this.load().find((x) => x.id === id);
  }

  unread(): number {
    return this.load().filter((x) => !x.read).length;
  }

  add(a: AlertInfo): HistoryItem {
    const item: HistoryItem = { ...a, id: `${a.at.toString(36)}-${a.code}-${(this.seq++).toString(36)}`, read: false, kakao: '' };
    const items = this.load();
    items.unshift(item);
    if (items.length > HISTORY_MAX) items.length = HISTORY_MAX;
    this.save();
    return item;
  }

  /** ids = null: every item */
  markRead(ids: string[] | null): void {
    let changed = false;
    for (const x of this.load()) if (!x.read && (!ids || ids.includes(x.id))) (x.read = true), (changed = true);
    if (changed) this.save();
  }

  setKakao(id: string, state: 'sent' | 'failed', error?: string): void {
    const x = this.get(id);
    if (!x) return;
    x.kakao = state;
    x.kakaoError = error;
    this.save();
  }

  /** ids = null: everything */
  remove(ids: string[] | null): void {
    const before = this.load().length;
    this.items = ids ? this.load().filter((x) => !ids.includes(x.id)) : [];
    if (this.items.length !== before) this.save();
  }
}
