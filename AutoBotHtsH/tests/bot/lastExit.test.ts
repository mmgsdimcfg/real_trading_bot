/*
 * Update log
 * - [2026-11-28] type=test - r016 status.last_exit (a start that ended before trading): lastExitText per reason, exitAfterStart
 *   (only an exit written after the status read before the start, and only while r003 is not running)
 */

import { describe, expect, it } from 'vitest';
import { exitAfterStart, lastExitText, type LastExit, type R016Status } from '../../src/bot/shared/control';

const exit = (reason: string, detail = '', at = '2026-11-28T22:26:16'): LastExit => ({ at, reason, detail, code: 0, pid: 1 });
const status = (generated_at: string, running: boolean, last_exit?: LastExit | null) => ({ generated_at, running, last_exit }) as unknown as R016Status;

describe('lastExitText', () => {
  it('closed day: the date and weekend / holiday', () => {
    const t = lastExitText(exit('market_closed', '2026-11-28 | CLOSED | source=weekday | reason=weekend'));
    expect(t).toContain('2026-11-28');
    expect(t).toContain('휴장일(주말)');
    expect(t).toContain('개장일에 다시 실행');
    expect(lastExitText(exit('market_closed', '2026-12-25 | CLOSED | source=kis | reason=holiday'))).toContain('휴장일(공휴일)');
    expect(lastExitText(exit('market_closed'))).toContain('오늘은 휴장일이라');
  });

  it('the other reasons and an unknown one', () => {
    expect(lastExitText(exit('standby', 'host role is standby'))).toContain('standby');
    expect(lastExitText(exit('already_running', 'pid=12'))).toContain('(pid=12)');
    expect(lastExitText(exit('auth_failed', 'token expired'))).toContain('KIS 인증에 실패');
    expect(lastExitText(exit('no_watchlist', '/x/r004.txt'))).toContain('감시목록에 종목이 없어');
    expect(lastExitText(exit('something_new', 'x'))).toContain('something_new: x');
  });
});

describe('exitAfterStart', () => {
  const before = status('2026-11-28T22:26:10', false);

  it('an exit written after the start, r003 not running', () => {
    expect(exitAfterStart(before, status('2026-11-28T22:26:21', false, exit('market_closed')))?.reason).toBe('market_closed');
  });

  it('none while r003 runs, without the field, or for an older exit', () => {
    expect(exitAfterStart(before, status('2026-11-28T22:26:21', true, exit('market_closed')))).toBeNull();
    expect(exitAfterStart(before, status('2026-11-28T22:26:21', false))).toBeNull();
    expect(exitAfterStart(before, status('2026-11-28T22:26:21', false, null))).toBeNull();
    expect(exitAfterStart(before, status('2026-11-28T22:26:21', false, exit('market_closed', '', '2026-11-27T08:00:00')))).toBeNull();
    expect(exitAfterStart(before, null)).toBeNull();
  });
});
