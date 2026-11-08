/*
 * Update log
 * - [2026-11-05] type=refactor - shellQuote/shellPath moved here from main/ssh.ts so the r016 command builder (shared/control.ts)
 *   can be unit-tested without ssh2; main/ssh.ts re-exports them
 */

/** POSIX single-quote a string for the remote shell. */
export function shellQuote(s: string): string {
  return "'" + String(s).replace(/'/g, "'\\''") + "'";
}

/** Quote a path but keep a leading "~/" so the remote shell expands it. */
export function shellPath(p: string): string {
  if (p.startsWith('~/')) return '~/' + shellQuote(p.slice(2));
  return shellQuote(p);
}
