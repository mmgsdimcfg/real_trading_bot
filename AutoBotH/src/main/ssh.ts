/*
 * Update log
 * - [2026-11-04] type=feat - Read-only SSH access for the viewer: SFTP readdir / ranged read (byte offsets) and exec of fixed
 *   commands (date, r015_export_conditions.py). There is deliberately no write/delete/kill method in this class
 * - [2026-11-05] type=feat - SSH proxy option (same as TermH): when SshParams.proxy is set ('command' ProxyCommand / 'socks5' / 'http'),
 *   the tunnel from openProxySocket() (proxy.ts) is passed to ssh2 as cfg.sock instead of a direct TCP connection. Proxy steps and
 *   ProxyCommand stderr are emitted as 'log' events; the tunnel is destroyed on close() and on a failed connect
 * - [2026-11-05] type=feat - exec() takes an optional stdin string (written and closed right after the channel opens) for
 *   `r016_control.py watchlist set --stdin`; shellQuote/shellPath moved to shared/shell.ts (re-exported here). The class itself
 *   still has no file-write method: all server changes go through r016_control.py (main/control.ts)
 * - [2026-11-10] type=feat - implements FileHost (host.ts) next to the local host: listPrefix (the `ls <code>_*` the price reader
 *   used), readSmallFile + home() for the one-time KIS config import into the local mode. Still no write method
 */

import { Client, type SFTPWrapper, type ConnectConfig } from 'ssh2';
import { readFileSync } from 'node:fs';
import { EventEmitter } from 'node:events';
import type { Duplex } from 'node:stream';
import { openProxySocket, type ProxyConfig } from './proxy';
import type { FileHost, RemoteEntry } from './host';
import { shellQuote } from '../shared/shell';

export type { RemoteEntry } from './host';
export { shellQuote, shellPath } from '../shared/shell';

export interface SshParams {
  host: string;
  port: number;
  username: string;
  authMethod: 'key' | 'password';
  keyPath: string;
  secret?: string;
  /** undefined = direct connection */
  proxy?: ProxyConfig;
}

export class SshConnection extends EventEmitter implements FileHost {
  readonly kind = 'remote' as const;
  private client: Client | null = null;
  private sftp: SFTPWrapper | null = null;
  private proxySock: Duplex | null = null;
  connected = false;

  private destroyProxySock(): void {
    const s = this.proxySock;
    this.proxySock = null;
    if (s) {
      try {
        s.destroy();
      } catch {
        /* ignore */
      }
    }
  }

  async connect(p: SshParams): Promise<void> {
    this.close();
    if (!p.host) throw new Error('호스트가 설정되지 않았습니다.');
    const cfg: ConnectConfig = {
      host: p.host,
      port: p.port || 22,
      username: p.username,
      readyTimeout: 15000,
      keepaliveInterval: 15000,
      keepaliveCountMax: 3,
    };
    if (p.authMethod === 'password') cfg.password = p.secret || '';
    else {
      if (!p.keyPath) throw new Error('개인키 파일이 설정되지 않았습니다.');
      cfg.privateKey = readFileSync(p.keyPath);
      if (p.secret) cfg.passphrase = p.secret;
    }
    if (p.proxy) {
      // Via proxy: the proxy (or the ProxyCommand program) resolves and reaches the target; ssh2 runs over the tunnel unchanged
      const sock = await openProxySocket(p.proxy, { host: p.host, port: cfg.port!, username: p.username }, (_code, message) =>
        this.emit('log', message ? `[proxy] ${message}` : '[proxy] 터널 연결됨'),
      );
      this.proxySock = sock;
      cfg.sock = sock;
    }
    const client = new Client();
    const proxySock = this.proxySock;
    try {
      await new Promise<void>((resolve, reject) => {
        let settled = false;
        client
          .on('ready', () => {
            settled = true;
            resolve();
          })
          .on('error', (err) => {
            if (!settled) {
              settled = true;
              reject(err);
            } else this.emit('lost', err);
          })
          .on('close', () => {
            const was = this.connected;
            this.connected = false;
            this.sftp = null;
            if (this.client === client) this.client = null;
            if (proxySock && this.proxySock === proxySock) this.destroyProxySock();
            if (was) this.emit('lost', new Error('연결이 끊어졌습니다.'));
          })
          .connect(cfg);
      });
    } catch (e) {
      if (proxySock && this.proxySock === proxySock) this.destroyProxySock();
      throw e;
    }
    this.client = client;
    this.sftp = await new Promise<SFTPWrapper>((resolve, reject) => client.sftp((e, s) => (e ? reject(e) : resolve(s))));
    this.connected = true;
  }

  close(): void {
    const c = this.client;
    this.connected = false;
    this.client = null;
    this.sftp = null;
    if (c) {
      try {
        c.end();
      } catch {
        /* ignore */
      }
    }
    this.destroyProxySock();
  }

  private needSftp(): SFTPWrapper {
    if (!this.sftp || !this.connected) throw new Error('SSH가 연결되어 있지 않습니다.');
    return this.sftp;
  }

  readdir(dir: string): Promise<RemoteEntry[]> {
    const sftp = this.needSftp();
    return new Promise((resolve, reject) =>
      sftp.readdir(dir, (err, list) => {
        if (err) return reject(err);
        resolve(
          list.map((e) => ({
            name: e.filename,
            size: e.attrs.size,
            mtime: e.attrs.mtime * 1000,
            isDir: (e.attrs.mode & 0o170000) === 0o040000,
          })),
        );
      }),
    );
  }

  /** size / mtime of one remote file (null when it does not exist) */
  stat(path: string): Promise<{ size: number; mtime: number } | null> {
    const sftp = this.needSftp();
    return new Promise((resolve) =>
      sftp.stat(path, (err, st) => resolve(err ? null : { size: st.size, mtime: st.mtime * 1000 })),
    );
  }

  /** Read bytes [start, end) of a remote file. */
  readRange(path: string, start: number, end: number): Promise<Buffer> {
    const sftp = this.needSftp();
    if (end <= start) return Promise.resolve(Buffer.alloc(0));
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      const rs = sftp.createReadStream(path, { start, end: end - 1 });
      rs.on('data', (c: Buffer) => chunks.push(c));
      rs.on('error', reject);
      rs.on('end', () => resolve(Buffer.concat(chunks)));
    });
  }

  /** names in `dir` starting with `prefix` (one `ls` exec - cheaper than a readdir of a 3000-file data folder over SFTP) */
  async listPrefix(dir: string, prefix: string): Promise<string[]> {
    if (!/^[0-9A-Za-z_.-]*$/.test(prefix)) throw new Error(`prefix 형식 오류: ${prefix}`);
    const { out } = await this.exec(`cd ${shellQuote(dir)} 2>/dev/null && ls -1 -- ${prefix}* 2>/dev/null`, 20000);
    return out
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);
  }

  /** whole small file (null = missing) - KIS config import */
  async readSmallFile(path: string, maxBytes = 1024 * 1024): Promise<Buffer | null> {
    const st = await this.stat(path);
    if (!st) return null;
    if (st.size > maxBytes) throw new Error(`파일이 너무 큽니다 (${st.size} bytes): ${path}`);
    return this.readRange(path, 0, st.size);
  }

  /** remote $HOME (for ~/KIS/config) */
  async home(): Promise<string> {
    const { out } = await this.exec('printf %s "$HOME"', 15000);
    const h = out.trim();
    if (!h.startsWith('/')) throw new Error(`서버 HOME 을 읽지 못했습니다: ${h}`);
    return h;
  }

  exec(cmd: string, timeoutMs = 60000, stdin?: string): Promise<{ out: string; err: string; code: number | null }> {
    const client = this.client;
    if (!client || !this.connected) return Promise.reject(new Error('SSH가 연결되어 있지 않습니다.'));
    return new Promise((resolve, reject) => {
      client.exec(cmd, (err, stream) => {
        if (err) return reject(err);
        let out = '';
        let errOut = '';
        const timer = setTimeout(() => {
          try {
            stream.close();
          } catch {
            /* ignore */
          }
          reject(new Error(`원격 명령 시간 초과 (${timeoutMs / 1000}s)`));
        }, timeoutMs);
        stream.on('data', (d: Buffer) => (out += d.toString('utf8')));
        stream.stderr.on('data', (d: Buffer) => (errOut += d.toString('utf8')));
        stream.on('close', (code: number | null) => {
          clearTimeout(timer);
          resolve({ out, err: errOut, code });
        });
        if (stdin !== undefined) stream.end(stdin, 'utf8');
      });
    });
  }
}
