/*
 * Update log
 * - [2026-11-04] type=feat - Byte-level line splitter for incremental reads: a line still being written on the server
 *   (no trailing \n yet) is held back until the next chunk; UTF-8 is decoded only for complete lines, so a multi-byte
 *   Korean character split across two SFTP reads is never garbled
 */

const NL = 0x0a;

export class LineBuffer {
  private rest: Uint8Array = new Uint8Array(0);
  private decoder = new TextDecoder('utf-8');

  /** Feed raw bytes, get back the complete lines (without \r\n). */
  push(chunk: Uint8Array): string[] {
    let buf: Uint8Array;
    if (this.rest.length) {
      buf = new Uint8Array(this.rest.length + chunk.length);
      buf.set(this.rest, 0);
      buf.set(chunk, this.rest.length);
    } else {
      buf = chunk;
    }
    const last = buf.lastIndexOf(NL);
    if (last < 0) {
      this.rest = buf.slice();
      return [];
    }
    const text = this.decoder.decode(buf.subarray(0, last));
    this.rest = buf.slice(last + 1);
    return text.split('\n').map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
  }

  /** Bytes held back (incomplete last line). */
  get pending(): number {
    return this.rest.length;
  }

  reset(): void {
    this.rest = new Uint8Array(0);
  }
}
