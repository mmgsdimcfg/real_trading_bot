// Update log
// - [2026-11-06] type=feat (Stock master: downloads kospi_code.mst.zip / kosdaq_code.mst.zip from KIS's public master server (the same files
//   stocks_info/kis_*_code_mst.py of the official repo uses - no login needed), unzips them with a small zip reader (deflate via zlib),
//   decodes cp949, parses the fixed-width rows (shared/kis parseMasterRow) and caches the list as JSON in %APPDATA%/HtsH/master.json,
//   refreshed once per day. Search by code prefix, name, or Korean initial consonants (ㅅㅅㅈㅈ -> 삼성전자))
// - [2026-11-12] type=feat (NXT flag: nxt_kospi_code.mst.zip / nxt_kosdaq_code.mst.zip from the same public server mark the symbols
//   that also trade on NXT (MasterItem.nxt). A failed NXT download keeps the flags of the cached list. Cache version 2 - an older
//   cache (no flags) is refreshed at the next login)

import { app } from 'electron';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { inflateRawSync } from 'node:zlib';
import { parseMasterRow, parseNxtCodes, type Market, type MasterItem } from '../shared/kis';
import { matchScore } from '../shared/search';

const URLS: Record<Market, string> = {
  KOSPI: 'https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip',
  KOSDAQ: 'https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip',
};
const NXT_URLS = [
  'https://new.real.download.dws.co.kr/common/master/nxt_kospi_code.mst.zip',
  'https://new.real.download.dws.co.kr/common/master/nxt_kosdaq_code.mst.zip',
];
const CACHE_V = 2;

/** entries of a zip file (stored or deflated) */
export function unzip(buf: Buffer): { name: string; data: Buffer }[] {
  // end of central directory: last occurrence of PK\x05\x06
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('zip: end of central directory not found');
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  const out: { name: string; data: Buffer }[] = [];
  for (let k = 0; k < count; k++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('zip: bad central directory');
    const method = buf.readUInt16LE(p + 10);
    const csize = buf.readUInt32LE(p + 20);
    const nlen = buf.readUInt16LE(p + 28);
    const elen = buf.readUInt16LE(p + 30);
    const clen = buf.readUInt16LE(p + 32);
    const local = buf.readUInt32LE(p + 42);
    const name = buf.toString('latin1', p + 46, p + 46 + nlen);
    const lnl = buf.readUInt16LE(local + 26);
    const lel = buf.readUInt16LE(local + 28);
    const start = local + 30 + lnl + lel;
    const raw = buf.subarray(start, start + csize);
    out.push({ name, data: method === 0 ? Buffer.from(raw) : method === 8 ? inflateRawSync(raw) : Buffer.alloc(0) });
    p += 46 + nlen + elen + clen;
  }
  return out;
}

export function parseMasterText(text: string, market: Market): MasterItem[] {
  const out: MasterItem[] = [];
  for (const row of text.split('\n')) {
    const it = parseMasterRow(row, market);
    if (it) out.push(it);
  }
  return out;
}

interface Cache {
  v?: number;
  date: string;
  items: MasterItem[];
}

const cacheFile = () => join(app.getPath('userData'), 'master.json');
const todayKst = () => new Date(Date.now() + 9 * 3600000).toISOString().slice(0, 10);

export class Master {
  private items: MasterItem[] = [];
  private byCode = new Map<string, MasterItem>();
  private hasNxt = false;
  date = '';
  loading: Promise<void> | null = null;
  error = '';

  constructor(private fallback: MasterItem[] = []) {
    try {
      if (existsSync(cacheFile())) {
        const c = JSON.parse(readFileSync(cacheFile(), 'utf8')) as Cache;
        // an older cache has no NXT flags: usable now, refreshed by the next ensure()
        this.set(c.items, c.v === CACHE_V ? c.date : '');
      }
    } catch {
      /* cache is optional */
    }
    if (!this.items.length) this.set(fallback, '');
  }

  private set(items: MasterItem[], date: string): void {
    this.items = items;
    this.byCode = new Map(items.map((i) => [i.code, i]));
    this.hasNxt = items.some((i) => i.nxt);
    this.date = date;
  }

  get size(): number {
    return this.items.length;
  }

  /** the list carries NXT flags (false until the NXT master was downloaded once) */
  get nxtKnown(): boolean {
    return this.hasNxt;
  }

  /** download when the cache is older than today (or forced) */
  ensure(force = false): Promise<void> {
    if (!force && this.date === todayKst()) return Promise.resolve();
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        const all: MasterItem[] = [];
        const decoder = new TextDecoder('euc-kr');
        for (const market of ['KOSPI', 'KOSDAQ'] as Market[]) {
          const res = await fetch(URLS[market]);
          if (!res.ok) throw new Error(`${market} 마스터 다운로드 실패 (HTTP ${res.status})`);
          const files = unzip(Buffer.from(await res.arrayBuffer()));
          const mst = files.find((f) => f.name.endsWith('.mst'));
          if (!mst) throw new Error(`${market} 마스터 zip에 .mst 파일이 없습니다`);
          all.push(...parseMasterText(decoder.decode(mst.data), market));
        }
        if (all.length < 1000) throw new Error(`마스터 종목 수가 너무 적습니다 (${all.length})`);
        let nxtError = '';
        try {
          const nxt = new Set<string>();
          for (const url of NXT_URLS) {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const mst = unzip(Buffer.from(await res.arrayBuffer())).find((f) => f.name.endsWith('.mst'));
            if (!mst) throw new Error('zip에 .mst 파일이 없습니다');
            for (const c of parseNxtCodes(decoder.decode(mst.data))) nxt.add(c);
          }
          if (nxt.size < 100) throw new Error(`종목 수가 너무 적습니다 (${nxt.size})`);
          for (const it of all) if (nxt.has(it.code)) it.nxt = true;
        } catch (e) {
          // keep what the previous list knew - the price list itself is still good
          nxtError = `NXT 종목 목록 갱신 실패: ${(e as Error).message}`;
          for (const it of all) if (this.byCode.get(it.code)?.nxt) it.nxt = true;
        }
        this.set(all, todayKst());
        this.error = nxtError;
        writeFileSync(cacheFile(), JSON.stringify({ v: CACHE_V, date: this.date, items: all }), 'utf8');
      } catch (e) {
        this.error = (e as Error).message;
        if (!this.items.length) this.set(this.fallback, '');
      }
    })().finally(() => (this.loading = null));
    return this.loading;
  }

  get(code: string): MasterItem | undefined {
    return this.byCode.get(code);
  }

  search(q: string, limit = 30): MasterItem[] {
    const s = q.trim();
    if (!s) return [];
    const scored: [number, MasterItem][] = [];
    for (const it of this.items) {
      const sc = matchScore(it.code, it.name, s);
      if (sc > 0) scored.push([sc, it]);
    }
    scored.sort((a, b) => b[0] - a[0] || (a[1].group === 'ST' ? -1 : 1) - (b[1].group === 'ST' ? -1 : 1) || a[1].name.length - b[1].name.length);
    return scored.slice(0, limit).map((x) => x[1]);
  }
}
