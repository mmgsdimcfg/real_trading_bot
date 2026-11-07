/*
 * Update log
 * - [2026-11-05] type=feat - Proxy tunnels for the SSH connection, ported from TermH/proxy.js. The tunnel is handed to ssh2 as the
 *   `sock` option (ssh2 treats any non-net.Socket Duplex as already connected and runs the SSH protocol over it).
 *   openProxySocket(proxy, target, onLog) supports
 *   (1) 'command' - OpenSSH-style ProxyCommand: the command line is run through the shell (%h=host, %p=port, %r=username, %%=%)
 *       and its stdin/stdout become the SSH byte stream (e.g. `cloudflared access ssh --hostname %h`, `ssh -W %h:%p jump`).
 *       stderr lines are forwarded to onLog; if the process exits before sending a byte, the last stderr line is reported as the
 *       error. On Windows the whole process tree is killed with taskkill /T (killing only cmd.exe would orphan the real program)
 *   (2) 'socks5' - SOCKS5 CONNECT with the target as a domain name (resolved by the proxy), optional username/password (RFC 1929)
 *   (3) 'http' - HTTP CONNECT with optional Basic Proxy-Authorization
 */

import net from 'node:net';
import { spawn, execFile, type ChildProcess } from 'node:child_process';
import { Duplex } from 'node:stream';

export type ProxyType = '' | 'command' | 'socks5' | 'http';

export interface ProxyConfig {
  type: Exclude<ProxyType, ''>;
  command?: string;
  host?: string;
  port?: number;
  username?: string;
  password?: string;
}

export interface ProxyTarget {
  host: string;
  port: number;
  username: string;
}

export type ProxyLog = (code: 'proxyCommand' | 'proxyServer' | 'proxyReady' | 'proxyStderr', message: string) => void;

const HANDSHAKE_TIMEOUT_MS = 20000;
const proxyChildren = new Set<ChildProcess>(); // live ProxyCommand processes - killed on app quit as a last resort

export function expandProxyCommand(cmd: string, target: ProxyTarget): string {
  return String(cmd || '').replace(/%([%hpr])/g, (_m, c: string) => {
    if (c === '%') return '%';
    if (c === 'h') return target.host || '';
    if (c === 'p') return String(target.port || 22);
    return target.username || '';
  });
}

function killTree(child: ChildProcess): void {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform === 'win32' && child.pid) {
    execFile('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true }, () => {});
  } else {
    try {
      child.kill();
    } catch {
      /* ignore */
    }
  }
}

export function killAllProxyChildren(): void {
  for (const child of proxyChildren) killTree(child);
  proxyChildren.clear();
}

/* ProxyCommand: child stdin/stdout wrapped as one Duplex */
function openCommandSocket(command: string, onLog: ProxyLog): Duplex {
  const child = spawn(command, { shell: true, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  proxyChildren.add(child);

  let lastStderr = '';
  let stderrBuf = '';
  let gotData = false;

  const sock = new Duplex({
    read() {
      child.stdout.resume();
    },
    write(chunk, _enc, cb) {
      if (!child.stdin.writable) return cb();
      child.stdin.write(chunk, () => cb());
    },
    final(cb) {
      try {
        child.stdin.end();
      } catch {
        /* ignore */
      }
      cb();
    },
    destroy(err, cb) {
      try {
        child.stdin.destroy();
      } catch {
        /* ignore */
      }
      killTree(child);
      cb(err);
    },
  });

  child.stdout.on('data', (d: Buffer) => {
    gotData = true;
    if (!sock.push(d)) child.stdout.pause();
  });
  child.stderr.on('data', (d: Buffer) => {
    stderrBuf += d.toString('utf8');
    const lines = stderrBuf.split(/\r?\n/);
    stderrBuf = lines.pop() || '';
    for (const line of lines) {
      const text = line.trim();
      if (!text) continue;
      lastStderr = text;
      onLog('proxyStderr', text);
    }
  });
  child.stdin.on('error', () => {}); // EPIPE after the process exited - reported via 'close' below
  child.on('error', (err) => {
    proxyChildren.delete(child);
    if (!sock.destroyed) sock.destroy(new Error('ProxyCommand: ' + err.message));
  });
  child.on('close', (code) => {
    proxyChildren.delete(child);
    if (stderrBuf.trim()) lastStderr = stderrBuf.trim();
    if (sock.destroyed) return;
    if (!gotData) {
      // Exited without producing a byte - the tunnel never came up; report why instead of a vague "connection closed"
      sock.destroy(new Error(`ProxyCommand 종료 (code ${code})` + (lastStderr ? ': ' + lastStderr : '')));
      return;
    }
    sock.push(null);
    sock.end();
    setTimeout(() => {
      if (!sock.destroyed) sock.destroy();
    }, 1000);
  });

  return sock;
}

type DoneCheck<T> = (b: Buffer) => { result: T; used: number } | null;

/* Read from a socket until `isDone(buffer)` returns a result, with a timeout */
function readUntil<T>(socket: net.Socket, isDone: DoneCheck<T>, cb: (err: Error | null, result?: T) => void): void {
  let buf = Buffer.alloc(0);
  let finished = false;
  const finish = (err: Error | null, result?: T, rest?: Buffer) => {
    if (finished) return;
    finished = true;
    clearTimeout(timer);
    socket.removeListener('data', onData);
    socket.removeListener('error', onError);
    socket.removeListener('close', onClose);
    socket.pause();
    if (!err && rest && rest.length) socket.unshift(rest);
    cb(err, result);
  };
  const onData = (d: Buffer) => {
    buf = Buffer.concat([buf, d]);
    let r: ReturnType<DoneCheck<T>>;
    try {
      r = isDone(buf);
    } catch (e) {
      return finish(e as Error);
    }
    if (r) finish(null, r.result, buf.subarray(r.used));
  };
  const onError = (err: Error) => finish(err);
  const onClose = () => finish(new Error('프록시가 연결을 닫았습니다.'));
  const timer = setTimeout(() => finish(new Error('프록시 핸드셰이크 시간 초과')), HANDSHAKE_TIMEOUT_MS);
  socket.on('data', onData);
  socket.on('error', onError);
  socket.on('close', onClose);
  socket.resume();
}

const SOCKS5_REPLY: Record<number, string> = {
  1: 'general SOCKS server failure',
  2: 'connection not allowed by ruleset',
  3: 'network unreachable',
  4: 'host unreachable',
  5: 'connection refused',
  6: 'TTL expired',
  7: 'command not supported',
  8: 'address type not supported',
};

function socks5Handshake(socket: net.Socket, proxy: ProxyConfig, target: ProxyTarget, cb: (err: Error | null) => void): void {
  const useAuth = !!proxy.username;
  socket.write(Buffer.from(useAuth ? [5, 2, 0, 2] : [5, 1, 0]));
  readUntil<number>(socket, (b) => (b.length >= 2 ? { result: b[1], used: 2 } : null), (err, method) => {
    if (err) return cb(err);
    if (method === 0xff) return cb(new Error('SOCKS5 proxy: no acceptable auth method'));
    const sendConnect = () => {
      const host = Buffer.from(target.host, 'utf8');
      const req = Buffer.alloc(7 + host.length);
      req[0] = 5;
      req[1] = 1;
      req[2] = 0;
      req[3] = 3;
      req[4] = host.length;
      host.copy(req, 5);
      req.writeUInt16BE(Number(target.port) || 22, 5 + host.length);
      socket.write(req);
      readUntil<boolean>(
        socket,
        (b) => {
          if (b.length < 5) return null;
          if (b[1] !== 0) throw new Error('SOCKS5 proxy: ' + (SOCKS5_REPLY[b[1]] || 'error ' + b[1]));
          const len = b[3] === 1 ? 4 : b[3] === 4 ? 16 : b[3] === 3 ? 1 + b[4] : 0;
          const total = 4 + len + 2;
          return b.length >= total ? { result: true, used: total } : null;
        },
        (e) => cb(e),
      );
    };
    if (method === 2) {
      const u = Buffer.from(proxy.username || '', 'utf8');
      const p = Buffer.from(proxy.password || '', 'utf8');
      socket.write(Buffer.concat([Buffer.from([1, u.length]), u, Buffer.from([p.length]), p]));
      readUntil<number>(socket, (b) => (b.length >= 2 ? { result: b[1], used: 2 } : null), (err2, status) => {
        if (err2) return cb(err2);
        if (status !== 0) return cb(new Error('SOCKS5 proxy: authentication failed'));
        sendConnect();
      });
    } else {
      sendConnect();
    }
  });
}

function httpConnectHandshake(socket: net.Socket, proxy: ProxyConfig, target: ProxyTarget, cb: (err: Error | null) => void): void {
  const hostPort = `${target.host}:${Number(target.port) || 22}`;
  let req = `CONNECT ${hostPort} HTTP/1.1\r\nHost: ${hostPort}\r\n`;
  if (proxy.username) {
    req += 'Proxy-Authorization: Basic ' + Buffer.from(`${proxy.username}:${proxy.password || ''}`).toString('base64') + '\r\n';
  }
  socket.write(req + '\r\n');
  readUntil<boolean>(
    socket,
    (b) => {
      const end = b.indexOf('\r\n\r\n');
      if (end < 0) {
        if (b.length > 16384) throw new Error('HTTP proxy: response header too large');
        return null;
      }
      const statusLine = b.subarray(0, b.indexOf('\r\n')).toString('latin1');
      const m = /^HTTP\/\d(?:\.\d)?\s+(\d{3})/.exec(statusLine);
      if (!m || m[1] !== '200') throw new Error('HTTP proxy: ' + (statusLine || 'invalid response'));
      return { result: true, used: end + 4 };
    },
    (e) => cb(e),
  );
}

/** Open the tunnel to `target` through `proxy`; resolves with the stream to pass to ssh2 as `sock`. */
export function openProxySocket(proxy: ProxyConfig, target: ProxyTarget, onLog: ProxyLog): Promise<Duplex> {
  return new Promise((resolve, reject) => {
    if (proxy.type === 'command') {
      const command = expandProxyCommand(proxy.command || '', target);
      onLog('proxyCommand', command);
      try {
        resolve(openCommandSocket(command, onLog));
      } catch (err) {
        reject(err);
      }
      return;
    }

    const proxyPort = Number(proxy.port) || (proxy.type === 'socks5' ? 1080 : 8080);
    onLog('proxyServer', `${proxy.type === 'socks5' ? 'SOCKS5' : 'HTTP'} ${proxy.host || ''}:${proxyPort}`);
    const socket = net.connect({ host: proxy.host, port: proxyPort });
    let done = false;
    const fail = (err: Error) => {
      if (done) return;
      done = true;
      socket.destroy();
      reject(err);
    };
    socket.once('error', fail);
    socket.setTimeout(HANDSHAKE_TIMEOUT_MS, () => fail(new Error('프록시 연결 시간 초과')));
    socket.once('connect', () => {
      const handshake = proxy.type === 'socks5' ? socks5Handshake : httpConnectHandshake;
      handshake(socket, proxy, target, (err) => {
        if (err) return fail(err);
        if (done) return;
        done = true;
        socket.removeListener('error', fail);
        socket.setTimeout(0);
        onLog('proxyReady', '');
        resolve(socket);
      });
    });
  });
}
