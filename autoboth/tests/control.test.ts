/*
 * Update log
 * - [2026-11-05] type=test - Remote control (guide 20261105 section 5): r016 response parsing for every example (1.1-1.6) plus
 *   rc 2/3 and noise around the JSON, shell quoting (Korean, spaces, JSON string values, rejected ; ` '), service button rules,
 *   market-hours check, value-editor parsing per type (int rejects fractions, dict JSON errors block the save, only changed keys
 *   are sent), watchlist row checks / stdin body, and the main-process exec wrapper with a mocked SSH exec (command string, stdin,
 *   serial queue)
 */

import { describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({ dialog: {}, ipcMain: { handle: vi.fn() } }));

import {
  buildR016Command,
  serverCodeDir,
  statusPollMs,
  checkWatchRows,
  collectChanges,
  extractJson,
  historyText,
  initialDraft,
  isMarketHours,
  parseDraft,
  parseR016Output,
  pctHint,
  pendingCount,
  serviceButtons,
  setArgs,
  watchlistStdin,
  writerName,
  type R016Status,
  type SchemaKey,
} from '../src/shared/control';
import { R016Client } from '../src/main/control';
import { shellQuote } from '../src/shared/shell';

// ---- fixtures (guide 1.1 - 1.6) ----

const STATUS: R016Status = {
  generated_at: '2026-11-05T22:29:15',
  service: { available: true, unit: 'autobot-r003.service', loaded: true, active_state: 'inactive', sub_state: 'dead', main_pid: null, started_at: null, restarts: 0, result: 'success' },
  running: false,
  external_r003: [],
  env: { R003_DRY_RUN: '0', AUTO_TRADING_RISK_PROFILE: 'neutral' },
  dry_run: false,
  risk_profile: 'neutral',
  override_count: 0,
  override_errors: [],
  pending: {
    overrides: { path: '/r/runtime/config/live_overrides.json', mtime: null, changed_since_start: false },
    env: { path: '/r/runtime/config/r003.env', mtime: null, changed_since_start: false },
    watchlist: { path: '/r/live_trading/r004_trade_watchlist_today.txt', mtime: '2026-11-05T17:46:11', changed_since_start: false },
  },
  pending_restart: false,
  latest_main_log: '/data/srv/autobot/runtime/log/20261105/r003_trade_live_execute_20261105_083236.log',
  watchlist_codes: 7,
};

const running = (patch: Partial<R016Status> = {}): R016Status => ({
  ...STATUS,
  running: true,
  service: { ...STATUS.service, active_state: 'active', sub_state: 'running', main_pid: 4242, started_at: '2026-11-05T08:32:36' },
  ...patch,
});

const j = (o: unknown) => JSON.stringify(o, null, 2);

describe('r016 response parser', () => {
  it('1.1 status', () => {
    const r = parseR016Output<R016Status>(j(STATUS), '', 0);
    expect(r.ok).toBe(true);
    expect(r.data?.service.active_state).toBe('inactive');
    expect(r.data?.pending.watchlist?.mtime).toBe('2026-11-05T17:46:11');
  });

  it('1.2 start ok / external r003 refusal (rc 2) / systemctl failure (rc 3)', () => {
    const ok = parseR016Output(j({ ok: true, verb: 'start', stderr: '', service: { active_state: 'active', main_pid: 4242 } }), '', 0);
    expect(ok.ok).toBe(true);
    const ext = parseR016Output(
      j({ ok: false, error: 'r003 already running outside the service (screen/terminal?) - stop it first', external_r003: [{ pid: 777, cmdline: 'python r003' }] }),
      '',
      2,
    );
    expect(ext.ok).toBe(false);
    expect(ext.rc).toBe(2);
    expect(ext.error).toContain('already running outside the service');
    const sys = parseR016Output(j({ ok: false, verb: 'stop', stderr: 'Failed to stop autobot-r003.service: Access denied' }), '', 3);
    expect(sys.ok).toBe(false);
    expect(sys.error).toBe('Failed to stop autobot-r003.service: Access denied');
  });

  it('1.3 schema', () => {
    const r = parseR016Output<{ keys: SchemaKey[] }>(
      j({ generated_at: 'x', overrides_file: 'f', errors: [], keys: [{ key: 'ENABLE_PROFIT_LOCK', type: 'bool', base: true, override: null, has_override: false, effective: true, pending: false }] }),
      '',
      0,
    );
    expect(r.ok).toBe(true);
    expect(r.data?.keys[0].type).toBe('bool');
  });

  it('1.4 conditions with values_meta', () => {
    const doc = {
      generated_at: 'x',
      buy: [{ order: 0, no: 29, func: '_029', stage: 'CONTEXT', title: 't', description: '', values: { ENABLE_MARKET_REGIME_GATE: true }, values_meta: { ENABLE_MARKET_REGIME_GATE: { editable: true, type: 'bool', base: true, override: null, has_override: false, pending: false } } }],
      sell: [],
      overrides: {},
      override_errors: [],
    };
    const r = parseR016Output<typeof doc>(j(doc), '', 0);
    expect(r.ok).toBe(true);
    expect(r.data?.buy[0].values_meta.ENABLE_MARKET_REGIME_GATE.editable).toBe(true);
  });

  it('1.5 set accepted / partly rejected (rc 2) / history', () => {
    const ok = parseR016Output(j({ ok: true, accepted: { ENABLE_PROFIT_LOCK: false }, rejected: {}, restart_required: true }), '', 0);
    expect(ok.ok).toBe(true);
    const part = parseR016Output<{ accepted: Record<string, unknown>; rejected: Record<string, string> }>(
      j({ ok: false, accepted: { MAX_OPEN_POSITIONS: 2 }, rejected: { ENABLE_PROFIT_LOCK: 'type mismatch (expected bool, got int)' } }),
      '',
      2,
    );
    expect(part.ok).toBe(false);
    expect(part.data?.accepted).toEqual({ MAX_OPEN_POSITIONS: 2 });
    expect(part.error).toContain('ENABLE_PROFIT_LOCK: type mismatch');
    const h = parseR016Output<{ rows: unknown[] }>(j({ rows: [{ ts: '2026-11-05T22:30:00', by: 'autoboth@hjcho', action: 'set', changes: { ENABLE_PROFIT_LOCK: { from: true, to: false } } }] }), '', 0);
    expect(h.data?.rows).toHaveLength(1);
  });

  it('1.6 watchlist get / set', () => {
    const g = parseR016Output<{ codes: unknown[] }>(j({ path: 'p', mtime: '2026-11-05T17:46:11', header: ['# x'], codes: [{ code: '161390', name: '한국타이어앤테크놀로지' }] }), '', 0);
    expect(g.ok).toBe(true);
    expect(g.data?.codes).toHaveLength(1);
    const s = parseR016Output<{ backup: string }>(j({ ok: true, count: 7, backup: '/r/runtime/config/watchlist_backup/r004_20261105_223000.txt', path: 'p' }), '', 0);
    expect(s.data?.backup).toContain('watchlist_backup');
  });

  it('noise before/after the JSON is cut off (first { .. last })', () => {
    const out = `WARNING: pandas deprecated\n${j(STATUS)}\nbye\n`;
    expect(parseR016Output(out, 'warn', 0).ok).toBe(true);
    expect(extractJson('no json here')).toBeUndefined();
  });

  it('no JSON at all -> error with the stderr tail', () => {
    const r = parseR016Output('', 'Traceback\nModuleNotFoundError: No module named x', 1);
    expect(r.ok).toBe(false);
    expect(r.error).toContain('ModuleNotFoundError');
  });

  it('exit 0 but {"error": ...} is a failure', () => {
    expect(parseR016Output(j({ error: 'unknown key FOO' }), '', 0).ok).toBe(false);
  });
});

describe('shell quoting / command building', () => {
  it('quotes Korean, spaces and JSON string values', () => {
    const cmd = buildR016Command('/data/srv/autobot', '~/venv/bin/python', 'autoboth@hjcho', ['set', 'EDGE_AI_MODE="shadow"', 'MARKET_REGIME_ETF={"KOSPI":"069500"}', '삼성 전자']);
    expect(cmd).toBe(
      `cd '/data/srv/autobot/real_trading_bot/autobot' && R016_BY='autoboth@hjcho' ~/'venv/bin/python' live_trading/r016_control.py 'set' 'EDGE_AI_MODE="shadow"' 'MARKET_REGIME_ETF={"KOSPI":"069500"}' '삼성 전자'`,
    );
  });
  it('shellQuote escapes single quotes', () => {
    expect(shellQuote("a'b")).toBe(`'a'\\''b'`);
  });
  it('rejects ; ` and \' in arguments', () => {
    expect(() => buildR016Command('/r', 'py', 'me', ['set', 'X="a;b"'])).toThrow();
    expect(() => buildR016Command('/r', 'py', 'me', ['set', 'X="`id`"'])).toThrow();
    expect(() => buildR016Command('/r', 'py', 'me', ['set', "X='a'"])).toThrow();
    expect(() => buildR016Command('/r', 'py', 'me', ['set', 'X="a\nb"'])).toThrow();
  });
  it('writer name keeps only safe characters', () => {
    expect(writerName('hjcho')).toBe('autoboth@hjcho');
    expect(writerName("Jo Hyun'; rm")).toBe('autoboth@JoHyunrm');
    expect(writerName('조현')).toBe('autoboth@user');
  });
});

describe('service buttons', () => {
  it('stopped, no external -> only start', () => {
    expect(serviceButtons(STATUS, false)).toMatchObject({ start: true, stop: false, restart: false });
  });
  it('running -> stop/restart, no start', () => {
    expect(serviceButtons(running(), false)).toMatchObject({ start: false, stop: true, restart: true });
  });
  it('external r003 -> start disabled with a reason', () => {
    const b = serviceButtons({ ...STATUS, running: true, external_r003: [{ pid: 777, cmdline: 'python r003' }] }, false);
    expect(b.start).toBe(false);
    expect(b.startReason).toContain('서비스 밖');
    expect(b.stop).toBe(false);
  });
  it('running with pending changes still allows restart', () => {
    const s = running({ pending_restart: true, pending: { ...STATUS.pending, overrides: { path: 'p', mtime: 'm', changed_since_start: true } } });
    expect(serviceButtons(s, false)).toMatchObject({ start: false, restart: true });
    expect(pendingCount(s)).toBe(1);
    expect(pendingCount(STATUS)).toBe(0);
  });
  it('busy or no status -> all disabled', () => {
    expect(serviceButtons(running(), true)).toMatchObject({ start: false, stop: false, restart: false });
    expect(serviceButtons(null, false)).toMatchObject({ start: false, stop: false, restart: false });
  });
  it('crash loop (activating / auto-restart) can be stopped', () => {
    const s = { ...STATUS, service: { ...STATUS.service, active_state: 'activating', sub_state: 'auto-restart' } };
    expect(serviceButtons(s, false)).toMatchObject({ stop: true, restart: false });
  });
});

describe('market hours (server KST string, no conversion)', () => {
  it('weekday 09:00-15:30', () => {
    expect(isMarketHours('2026-11-05T08:59:59')).toBe(false);
    expect(isMarketHours('2026-11-05T09:00:00')).toBe(true);
    expect(isMarketHours('2026-11-05T15:29:00')).toBe(true);
    expect(isMarketHours('2026-11-05T15:30:00')).toBe(false);
  });
  it('weekend is closed', () => {
    expect(isMarketHours('2026-11-07T10:00:00')).toBe(false); // Saturday
    expect(isMarketHours(undefined)).toBe(false);
  });
});

describe('value editor', () => {
  const k = (key: string, type: SchemaKey['type'], base: unknown, extra: Partial<SchemaKey> = {}): SchemaKey => ({
    key,
    type,
    base,
    override: null,
    has_override: false,
    pending: false,
    ...extra,
  });

  it('widget draft per type', () => {
    expect(initialDraft(k('ENABLE_X', 'bool', true))).toBe(true);
    expect(initialDraft(k('N', 'int', 3))).toBe('3');
    expect(initialDraft(k('D', 'dict', { a: 'b' }))).toBe('{"a":"b"}');
    expect(initialDraft(k('S', 'str', 'gate', { has_override: true, override: 'shadow' }))).toBe('shadow');
  });

  it('int rejects fractions and non-numbers, passes other JSON to the server', () => {
    expect(parseDraft('int', '2')).toEqual({ ok: true, value: 2 });
    expect(parseDraft('int', '1.5').ok).toBe(false);
    expect(parseDraft('int', '2.0').ok).toBe(false);
    expect(parseDraft('int', '"a"')).toEqual({ ok: true, value: 'a' });
    expect(parseDraft('int', 'a').ok).toBe(false);
    // `true` in an int field is not blocked by the app - the server rejects it (guide 6.3)
    expect(parseDraft('int', 'true')).toEqual({ ok: true, value: true });
  });

  it('float accepts decimals', () => {
    expect(parseDraft('float', '0.006')).toEqual({ ok: true, value: 0.006 });
    expect(parseDraft('float', 'x').ok).toBe(false);
  });

  it('dict / list JSON errors block', () => {
    expect(parseDraft('dict', '{"KOSPI":"069500"}')).toEqual({ ok: true, value: { KOSPI: '069500' } });
    expect(parseDraft('dict', '{KOSPI:1}').ok).toBe(false);
    expect(parseDraft('dict', '[1]').ok).toBe(false);
    expect(parseDraft('list', '[60,120]')).toEqual({ ok: true, value: [60, 120] });
    expect(parseDraft('list', '{"a":1}').ok).toBe(false);
  });

  it('str rejects quotes / semicolons / backticks', () => {
    expect(parseDraft('str', 'shadow')).toEqual({ ok: true, value: 'shadow' });
    expect(parseDraft('str', 'a;b').ok).toBe(false);
    expect(parseDraft('str', 'a"b').ok).toBe(false);
    expect(parseDraft('str', '`x`').ok).toBe(false);
  });

  it('only changed keys go into set; input errors are reported separately', () => {
    const items = [
      k('ENABLE_PROFIT_LOCK', 'bool', true),
      k('PROFIT_LOCK_GIVEBACK_PCT', 'float', 0.004),
      k('MARKET_REGIME_MA_FAST', 'int', 5),
      k('MARKET_REGIME_ETF', 'dict', { KOSPI: '069500' }),
      k('EDGE_AI_MODE', 'str', 'gate'),
      { ...k('EDGE_AI_DIR', 'str', '/x'), editable: false },
    ];
    const drafts = {
      ENABLE_PROFIT_LOCK: false,
      PROFIT_LOCK_GIVEBACK_PCT: '0.004', // unchanged
      MARKET_REGIME_MA_FAST: 'a', // blocked
      MARKET_REGIME_ETF: '{"KOSPI": "069500"}', // same value, different spacing
      EDGE_AI_MODE: 'shadow',
      EDGE_AI_DIR: '/y', // read-only, ignored
    };
    const c = collectChanges(items, drafts);
    expect(c.changes).toEqual({ ENABLE_PROFIT_LOCK: false, EDGE_AI_MODE: 'shadow' });
    expect(Object.keys(c.errors)).toEqual(['MARKET_REGIME_MA_FAST']);
    expect(setArgs(c.changes)).toEqual(['ENABLE_PROFIT_LOCK=false', 'EDGE_AI_MODE="shadow"']);
  });

  it('going back to the saved value is not a change; override counts as the saved value', () => {
    const it = k('X', 'int', 5, { has_override: true, override: 7 });
    expect(collectChanges([it], { X: '7' }).changes).toEqual({});
    expect(collectChanges([it], { X: '5' }).changes).toEqual({ X: 5 });
  });

  it('percent hint', () => {
    expect(pctHint('PROFIT_LOCK_GIVEBACK_PCT', 0.004)).toBe('= 0.40%');
    expect(pctHint('MAX_OPEN_POSITIONS', 2)).toBe('');
  });

  it('history text', () => {
    expect(historyText({ ts: '', by: '', action: 'set', changes: { ENABLE_PROFIT_LOCK: { from: true, to: false } } })).toBe('ENABLE_PROFIT_LOCK true->false');
    expect(historyText({ ts: '', by: '', action: 'watchlist', changes: { codes: ['1', '2'], backup: null } })).toBe('2종목, backup=null');
  });
});

describe('watchlist rows', () => {
  it('code must be 6 digits, name without , ; quotes, no duplicates', () => {
    const c = checkWatchRows([
      { code: '005930', name: '삼성전자' },
      { code: '5930', name: '' },
      { code: '000660', name: 'SK,하이닉스' },
      { code: '005930', name: 'dup' },
    ]);
    expect(Object.keys(c.rows).map(Number)).toEqual([1, 2, 3]);
    expect(c.list).toBeUndefined();
  });
  it('more than 100 rows is a list error', () => {
    const rows = Array.from({ length: 101 }, (_, i) => ({ code: String(100000 + i), name: '' }));
    expect(checkWatchRows(rows).list).toContain('100');
  });
  it('stdin body: one code,name per line, name optional', () => {
    expect(watchlistStdin([{ code: '005930', name: '삼성전자' }, { code: '000660', name: '' }])).toBe('005930,삼성전자\n000660\n');
  });
});

describe('R016Client (mocked SSH exec)', () => {
  const cfg = () => ({ remoteRoot: '/data/srv/autobot', pythonPath: '~/venv/bin/python', by: 'autoboth@hjcho' });

  it('sends the quoted command, timeout and stdin', async () => {
    const exec = vi.fn(async () => ({ out: j({ ok: true, count: 2, backup: 'b', path: 'p' }), err: '', code: 0 }));
    const c = new R016Client(exec, cfg);
    const r = await c.run(['watchlist', 'set', '--stdin'], { stdin: '005930,삼성전자\n000660\n' });
    expect(r.ok).toBe(true);
    expect(exec).toHaveBeenCalledWith(
      `cd '/data/srv/autobot/real_trading_bot/autobot' && R016_BY='autoboth@hjcho' ~/'venv/bin/python' live_trading/r016_control.py 'watchlist' 'set' '--stdin'`,
      30000,
      '005930,삼성전자\n000660\n',
    );
  });

  it('runs calls one at a time (serial queue)', async () => {
    let active = 0;
    let maxActive = 0;
    const order: string[] = [];
    const exec = vi.fn(async (cmd: string) => {
      active++;
      maxActive = Math.max(maxActive, active);
      await new Promise((r) => setTimeout(r, 10));
      order.push(cmd.split(' ').slice(-1)[0]);
      active--;
      return { out: '{"ok": true}', err: '', code: 0 };
    });
    const c = new R016Client(exec, cfg);
    await Promise.all([c.run(['status']), c.run(['history']), c.run(['schema'])]);
    expect(maxActive).toBe(1);
    expect(order).toEqual(["'status'", "'history'", "'schema'"]);
  });

  it('a failed call does not block the queue; unsafe args never reach exec', async () => {
    const exec = vi
      .fn()
      .mockRejectedValueOnce(new Error('원격 명령 시간 초과 (30s)'))
      .mockResolvedValueOnce({ out: '{"ok": true}', err: '', code: 0 });
    const c = new R016Client(exec, cfg);
    await expect(c.run(['status'])).rejects.toThrow('시간 초과');
    expect((await c.run(['status'])).ok).toBe(true);
    const bad = await c.run(['set', 'X="a;b"']);
    expect(bad.ok).toBe(false);
    expect(exec).toHaveBeenCalledTimes(2);
  });

  it('rc 2 rejection is returned, not retried', async () => {
    const exec = vi.fn(async () => ({ out: j({ ok: false, accepted: {}, rejected: { X: 'unknown key' } }), err: '', code: 2 }));
    const c = new R016Client(exec, cfg);
    const r = await c.run(['set', 'X=1']);
    expect(r.ok).toBe(false);
    expect(r.rc).toBe(2);
    expect(exec).toHaveBeenCalledTimes(1);
  });
});

describe('server code folder and status poll interval (2026-11-07)', () => {
  it('builds the monorepo server folder from the remote root', () => {
    expect(serverCodeDir('/data/srv/autobot')).toBe('/data/srv/autobot/real_trading_bot/autobot');
    expect(serverCodeDir('/data/srv/autobot//')).toBe('/data/srv/autobot/real_trading_bot/autobot');
    expect(buildR016Command('/data/srv/autobot/', 'py', 'me', ['status'])).toContain("cd '/data/srv/autobot/real_trading_bot/autobot' && ");
  });

  it('polls status every 30 s by default, never below 15 s, 120 s when hidden', () => {
    expect(statusPollMs(undefined, true)).toBe(30000);
    expect(statusPollMs(0, true)).toBe(30000);
    expect(statusPollMs(Number.NaN, true)).toBe(30000);
    expect(statusPollMs(5, true)).toBe(15000);
    expect(statusPollMs(45, true)).toBe(45000);
    expect(statusPollMs(30, false)).toBe(120000);
    expect(statusPollMs(300, false)).toBe(300000);
  });
});
