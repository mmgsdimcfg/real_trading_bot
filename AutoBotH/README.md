<!--
Update log
- [2026-11-06] type=docs - v2.2.0_008: chart tab for any code (server `bars` / `quote`, KIS on the server)
- [2026-11-06] type=docs - v2.2.0_007: title bar + ☰ menu, chart tab
- [2026-11-06] type=docs - v2.1.0_006: chart indicators (Bollinger, Ichimoku, RSI, MACD, ADX) and the server data request document
- [2026-11-05] type=docs - v2.1.0_005: timeline price chart (tick / minute / day / week / month bars)
- [2026-11-05] type=docs - v2.1.0: remote control (control tab, value editor, ENABLE toggles, watchlist editor) through r016_control.py
- [2026-11-05] type=docs - SSH proxy option (ProxyCommand / SOCKS5 / HTTP CONNECT, same as TermH)
- [2026-11-04] type=docs - v2: rewritten for the r003 condition pipeline viewer (read-only, electron-vite + React + TypeScript)
- [2026-10-19] type=docs owner=copilot - Rewrite README in English: app description, features, usage, build/release, project structure
-->
# AutoBotH

AutoBotH is a Windows desktop app (Electron) that shows how the KRX trading bot **r003** evaluates its buy and sell conditions. The bot runs on a Linux server (Raspberry Pi). AutoBotH reads the bot's logs over SSH and draws each symbol's progress through the condition pipeline. For example: "A passed conditions 1-8 and was blocked by _014 (ATR floor)", "which condition blocked the most polls today", or "which sell condition fired for B".

Log reading is **read-only**: SFTP reads and `date +%Y%m%d`. It does not run grep/awk or any other preprocessing on the server, because the Pi is running the trading loop. Raw log lines never leave the PC.

Since v2.1.0 the app can also **control** r003 (see "Remote control" below). Every change goes through one server script, `live_trading/r016_control.py`, run over SSH exec. The app never writes server files itself (no SFTP writes), never calls the order API, and never reads `~/KIS/config`, tokens or account data.

v1 (r006 run/simulate modes, remote strategy config editor) was removed in v2.0.0.

## Views

1. **Pipeline** - the 25 buy conditions in evaluation order (`order` from the condition JSON, not the number), on stage bands (STATE / TRIGGER / CONTEXT / PRE_ORDER / ORDER). `_008` is expanded into its sub-steps (cross trigger, candle/gap guard, live guard, chase-from-cross, legacy).
   - No symbol selected: a funnel. Node width and the red bar follow today's blocked polls. Badges show blocked / passed / waiting counts, with pre-open counts listed separately.
   - Symbol selected: passed nodes are green, the stopping node is red (amber for a wait), unreached nodes are gray. The `STEPS` diagnostics of the last rejected poll appear as dashed outlines. The side panel shows the condition description, its values and the raw log line.
   - The sell tab maps `[SELL HOLD]` / `[SELL TRIGGER]` onto the 15 sell conditions.
2. **Symbol board** - every watched symbol: last evaluation, stopping condition and reason, depth reached today, most-blocked condition, position, realized P&L. It is sortable and can be filtered by stopping condition. Clicking a row opens the pipeline for that symbol.
3. **Timeline** - one symbol's day. Between the buy and sell evaluation charts there is a **price chart**: 틱봉 (tick bars, 1-300 trades per bar), 분봉 (minute bars, 1-60 min), 일봉 / 주봉 / 월봉 (daily / weekly / monthly). Tick and minute candles use the buy chart's time axis, so they line up with the evaluation dots. Daily, weekly and monthly bars end at the viewed day, which is highlighted. Data, read over SFTP: tick bars and minute bars come from the bot's tick files `runtime/data/ticks/YYYYMMDD/<code>.csv` (watched symbols only, kept about 5 days). Without a tick file, minute bars come from the real 1-minute rows (`raw_bar=1`) of `runtime/data/YYYYMMDD/<code>_<name>_10s.txt`. D/W/M bars come from `<code>_<name>_daily.csv` (about 200 trading days); the viewed day's bar is built from 09:00-15:30 intraday data when that file has no row for it yet. Indicators can be switched on above the chart: Bollinger and Ichimoku on the price pane, and RSI, MACD and ADX/DI each in a pane below. The formulas and periods match the bot (`r002_strategy_core_shared.py` `calculate_indicators()` with the r001 values BB 20/2, RSI 14 Wilder + signal 6, MACD 5/12/4, ADX 7). `tests/indicators.test.ts` checks them against pandas output of the bot's own code. Ichimoku is not used by the bot, so it uses the standard 9/26/52. Tick and minute charts compute the indicators over the previous trading day's last 120 bars plus the day's bars, so they are valid from the first bar. What the server could add for more accurate charts is listed in `docs/server_data_request_prompt_20261106.md`. The rest of the timeline: The x axis is KST time and the y axis is the depth reached (condition order). Dot shape and color show the result. Buy and sell fills appear as markers. Hovering a dot shows the raw line. A separate strip shows sell evaluations.
4. **Chart (차트)** - an HTS-style chart for any symbol: pick a watched symbol or type any 6-digit code. Bars come from the server: the bot's files where they exist (tick files, 10-second files, `daily_history`), otherwise the server's `r016_control.py bars`, which queries the KIS OpenAPI on the server once that is added (`docs/server_kis_market_data_request_20261106.md`). The side panel shows the KIS current price, 10-level order book and latest trades through `r016_control.py quote` when the server has it. The app itself never talks to KIS and never sees a token or key. Tick, minute, daily, weekly and monthly bars; tick and minute charts show the previous trading day before the viewed day (can be turned off). Scroll the mouse wheel to zoom, drag to pan, double-click to return to the newest bars. Price axis on the right with a last-price tag and a price tag at the cursor. Overlays: moving averages 5/20/60/120, Bollinger, Ichimoku, VWAP. Panes: RSI, MACD, ADX/DI, stochastic, Williams %R, MFI, OBV, each with a legend. The side panel shows the current price and change vs. the previous close, the day's open/high/low/volume, best ask/bid and 체결강도 (buy/sell trade strength, from the tick file), the 200-day high/low, a data window for the bar under the cursor, a plain-language signal summary per indicator, the bot's state for the symbol and the latest trades. The bot's own indicators use the bot's formulas and periods. With the previous day off, they also use its warm-up (the previous day's last 180 one-minute bars; VWAP and OBV per day), so values match r003.
5. **Trade results** - built from `YYYYMMDD_buy_sell.log`: headline numbers, cumulative realized P&L, P&L by sell condition, and the trade table. Gains are red and losses blue (Korean convention).
6. **Control (제어)** - r003 service state and start/stop/restart, saved overrides, change history (see "Remote control").
7. **Settings / connection** - SSH settings (including an optional proxy, see below), remote root, Python path, poll interval, funnel start time, cache retention. 매매 시간 (서버 r003) (trading hours): the NXT session switch (`ENABLE_NXT_SESSION`, see "Remote control"). Also date selection, opening a local folder, refreshing the condition JSON, and the list of unknown tags/reasons.

**Past-day replay (재현).** Open a past date (설정/연결 > 날짜 / 데이터 소스) or a local folder, and a replay bar appears under the tabs (not on 제어 and 설정/연결). Move the time slider, press ▶ 재생 (play) at 1x-600x, step -1분 / +1분, jump to the previous / next trade (yellow ticks on the slider are the buy_sell log entries), or type HH:MM and press Enter. Pipeline, board, timeline, chart trade markers and trade results then show the day as the logs stood at that time; tick and minute price bars after that time are hidden (daily / weekly / monthly bars are not cut). 전체 (whole day) ends the replay. It runs in the app's worker thread on the bytes already loaded, so the server does no extra work. Today's live day has no replay bar.

The window has no OS title bar or menu bar (same style as TermH). The top row shows the app icon and name, then the status items, then ☰ (설정 = settings, 업데이트 확인 = check for updates, AutoBotH 정보 = about) and the minimize / maximize / close buttons. Drag empty space to move the window; double-click it to maximize. The status bar shows the connection state, the day (LIVE or loaded once), the last receive time, files/bytes/lines, the parse-warning count, `[MKT REGIME]`, the watch-list split and where the condition list came from.

## SSH proxy

The same proxy options as TermH, for servers that are reachable only through a tunnel (Cloudflare Access, a bastion host) or a proxy server. Pick one under 프록시 (proxy) on the settings view:

- **ProxyCommand** - an OpenSSH-style command line run through the shell; its stdin/stdout carry the SSH stream. `%h` = host, `%p` = port, `%r` = username, `%%` = `%`. Examples: `cloudflared access ssh --hostname %h`, `ssh -W %h:%p user@jumphost`. The program's stderr is shown in the status bar while connecting, and if it exits before the tunnel is up, its last stderr line is the error. On Windows the whole process tree is killed when the connection closes or the app quits.
- **SOCKS5** - SOCKS5 CONNECT (the proxy resolves the host), optional username/password. Default port 1080.
- **HTTP CONNECT** - optional Basic auth. Default port 8080.

The proxy password is stored encrypted like the SSH secret (see Security).

## Remote control

All calls run `cd <remote root>/real_trading_bot/AutoBot && R016_BY=autoboth@<Windows user> <python> live_trading/r016_control.py <command>`, with every argument single-quoted. The script prints one JSON object (exit code 0 = ok, 2 = validation failed or refused, 3 = systemctl failed). Calls go through **one serial queue**, so two calls never run at once. The timeout is 30 s, except start/stop/restart (120 s, because systemctl waits up to 90 s for r003 to save its state) and `conditions` (90 s). The app does not copy the server's validation rules. When the server refuses something, the app shows the server's message and does not retry. If the result is unknown (timeout or SSH drop), the app reads `status` again.

- **Control tab (제어)** - running / PID / start time / restart count / dry-run / risk profile, polled every `pollSec` while connected. The start button is disabled while r003 runs outside the service (screen or terminal); the PIDs and command lines are shown. Stop and restart are enabled while the unit is active (stop also while systemd is activating or auto-restarting). After an action, the app re-reads the state every 2 s until the unit settles. The badge "미적용 변경 N건" (N unapplied changes) means that the overrides, the env file or the watchlist changed after the service started. The tab also lists saved overrides with a reset-to-default button for each, a full reset, and the last 20 history rows (`history -n 20`).
- **Value editor** - clicking a pipeline node opens its 설정값 (values) table as an editor. bool values get a toggle, int/float/str an input (`*_PCT` values are ratios and show `= 0.40%`), list/dict a JSON text area. Nothing is sent while you type. 저장 (save) sends only the changed keys in one `set`, and each refused key shows the server's message on its row. Values with `editable:false` are read-only. Each node header mirrors the node's first `ENABLE_*` constant as an ON/OFF toggle; the toggle edits the panel draft and is saved with 저장. A yellow dot marks a saved override that the running bot has not loaded yet. All r001 constants (`schema --all`, about 380) can be edited under 설정/연결 > "고급: 전체 설정값" (advanced: all values).
- **NXT session** - 설정/연결 > 매매 시간 (서버 r003) > NXT 세션 운용 sets r001 `ENABLE_NXT_SESSION` with `set` (or `unset` when it goes back to the r001 value, so no override is left behind). The app reads it with `get ENABLE_NXT_SESSION` after connecting, after every override change and when the service start time changes, and shows the stored value, the running r003's value and "재시작 필요" (restart needed). Off: regular session only (15:20 entry cut-off and liquidation, exit at 15:30). On: also 08:00-08:50 and 15:30-20:00; same-day NXT-tradable positions are liquidated at 19:59:58 instead of 15:20, and r003 exits at 20:00.
- **Watchlist (r004)** - 종목 보드 (symbol board) > 감시목록 편집 (edit watchlist): add, delete or reorder rows (row order = r003's activation priority). Codes must be 6 digits, at most 100 rows. Saving sends `watchlist set --stdin`; the server backs up the old file first, and the app shows the backup path. The g002 scanner export rewrites this file every morning.
- **Nothing is hot-reloaded.** r003 reads the r001 constants and the r004 file only at start, so every change needs a restart. The service does not start at boot; after a server reboot, press 실행 (start) in the app.
- **Confirmations** come from the main process, so the window cannot skip them: start or restart in live-order mode (dry-run unchecked), stop or restart during market hours (Mon-Fri 09:00-15:30 by the server's KST clock), reset, and watchlist save.

## How it reads the server

- Log folder: `<remote root>/runtime/log/YYYYMMDD/` (default remote root: `/data/srv/autobot`).
- Main logs `r003_trade_live_execute_YYYYMMDD_HHMMSS.log` (a new file on every r003 restart) are read in name order, together with the daily `YYYYMMDD_buy_sell.log`. Symbol names come from the `YYYYMMDD_<code>_<name>.txt` file names and from `WATCH | ...` lines.
- Reads are **incremental by byte offset**: SFTP `readdir` gives the sizes, and only the new range is read. An incomplete last line is held back until the next poll. UTF-8 is decoded per complete line, so a Korean character split across two reads stays intact. If a file gets smaller, the whole day is reloaded from offset 0.
- **Past days' price files** are copied once into `%APPDATA%\autoboth\pricecache\YYYYMMDD\` (a past day's server files no longer change). After a past day's logs load, the app downloads in the background the tick file, the 10-second (1-minute bar) file and the daily file of every symbol in that day's log, plus the previous trading day's warm-up files. The status bar shows "주가 받는 중 3/22", then "주가 로컬 보관". It downloads only those symbols (about 70 MB a day), not the scanner's whole-market folder (about 1.4 GB). Charts of that day then come from the local copy, also without a connection. A download cut by a dropped connection resumes after the reconnect. Whole files are written to a temp file and renamed, so a broken download never leaves half a file. The server's daily answer (`bars --kind day`) of a past day is kept there too. Codes without server files (typed on the chart tab) still use the server's `bars`. Today's files are always read live.
- Downloaded bytes are cached in `%APPDATA%\autoboth\logcache\YYYYMMDD\`. A restart or a past date resumes from the cache instead of downloading again, and cached days can be opened offline. Folders older than the retention setting (default 30 days) are deleted at start.
- Parsing and aggregation run in a worker thread. The window only receives aggregated snapshots plus per-symbol timelines (the latest 2,000 events per symbol).
- "Today" is the server date. The app polls every `pollSec` (default 10 s), checks the server date every minute and switches to the new folder at midnight KST. Past dates load once, with no polling, and keep their lines in the worker for the replay bar. If the connection drops, it reconnects with back-off (5 s up to 60 s) and keeps the offsets.
- Condition list: fetched with `r016_control.py conditions` (falls back to `r015_export_conditions.py` on a server without r016, read-only display) once after connecting, after every override change, when the service start time changes, and with "조건 JSON 새로고침" (refresh condition JSON). If the fetch fails, the app uses the last good copy (`%APPDATA%\autoboth\conditions_cache.json`), then the bundled `resources/conditions_snapshot.json`. Log lines are matched to conditions **by number** (`_014`), so a renamed function does not break the mapping.
- Lines that match no rule (unknown tags or reasons) are kept and counted, never guessed. They are listed on the settings view, with digits collapsed to `#`.
- Funnel counts include only polls from the funnel start time. By default (자동 (서버 NXT 설정)) it follows the server's NXT setting: `08:00` when `ENABLE_NXT_SESSION` is on (the running r003's value, else the stored one), `09:00` when it is off. Untick it to type a time. Earlier rejections, such as pre-market `TICK_ABSTAIN_NO_FILE` waits, are shown separately as "개장 전" (pre-open). When the time changes, the open day is counted again. The setting affects only these counts, not r003's trading hours.

The mapping tables (guide sections 3.1 and 3.2) are in `src/shared/mapping.ts`.

## Security

- `contextIsolation` on, `nodeIntegration` off, sandboxed renderer. The preload exposes IPC only.
- The SSH key passphrase or password, and the proxy password, are stored encrypted with Electron `safeStorage` (DPAPI, tied to the Windows user) in `%APPDATA%\autoboth\autoboth-settings.json`. It is never stored in plain text and never sent to the renderer.

## Development

```
npm install
npm run dev         # electron-vite dev (hot reload)
npm start           # build, then run the built app
npm run test        # vitest
npm run typecheck   # tsc --noEmit
npm run build:win   # = npm run dist -> dist/AutoBotH_Setup_v<ver>_<build>.exe (+ _Portable.exe, latest.yml)
```

`npm run dev` / `npm start` go through `scripts/start.cjs`, which clears an inherited `ELECTRON_RUN_AS_NODE` before electron-vite starts Electron.

To open a copied log folder without SSH, use **로컬 폴더 열기…** (open local folder) on the settings view, or start the app with `--open-local <dir>`.

### Test fixtures

Copy one day of logs from the server. The `fixtures/` folder is git-ignored because it holds trading data.

```
scp -r user@host:/data/srv/autobot/runtime/log/20261104 fixtures/
```

`tests/parser.test.ts` covers the line grammar, every row of the mapping tables, buy_sell block parsing, incremental reads in pieces (including a split inside a multi-byte character) and the hold-back of an incomplete last line. `tests/fixture.test.ts` runs only when `fixtures/20261104/` exists. It checks that every line parses, that there are 12 sells totalling -20,265 KRW, that 159010 was sold by `_004`, and that `_014` is the most-blocked node with 257 blocked polls.

## Release & auto update

Unchanged from v1. `npm run release` builds and uploads to the target in `../release.config.json`. `updater.js` (shared by the make_utils apps, loaded at runtime from the app folder) checks `https://update.iccid.dev/AutoBotH/`. The version comes from `version.txt` (`<version>_<buildNumber>`, currently `2.0.0_001`).

## Project structure

```
src/shared/       types, KST time helpers, line grammar (logLine), log->condition mapping, DayEngine (aggregation), LineBuffer
src/shared/       control (r016 types, command builder, response parser, editor/watchlist checks), shell (quoting)
src/main/         index (app controller + IPC), ssh (SFTP reads + exec), proxy (ProxyCommand/SOCKS5/HTTP tunnel), sources (remote + byte cache, local folder),
                  session (offsets per file, worker), engineWorker, conditions (r016/r015 export + cache), settings (electron-store + safeStorage),
                  control (r016 serial queue, status polling, confirmations, ctl:* IPC)
src/preload/      contextBridge API
src/renderer/     React UI (views: Control, Pipeline, Board, Timeline, Trades, Settings; ValuesEditor, WatchlistEditor), zustand store; public/update-ui.js
resources/        conditions_snapshot.json (bundled fallback condition list)
tests/            vitest
updater.js        shared auto-update module
```

## License

Proprietary (UNLICENSED). Copyright (c) 2026 xgraph. All rights reserved - see [LICENSE](LICENSE). Third-party libraries remain under their own licenses.
