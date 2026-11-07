// Update log
// - [2026-11-06] type=fix (npm start failed with ERR_PACKAGE_PATH_NOT_EXPORTED: the electron-vite CLI is now found through its exported
//   package.json "bin" field instead of require.resolve('electron-vite/bin/electron-vite.js'))
// - [2026-11-04] type=feat (v2: runs electron-vite - `node scripts/start.cjs dev` (hot reload) or `preview` (built app) - with the same
//   ELECTRON_RUN_AS_NODE cleanup, since electron-vite starts Electron as a child of this process and the variable would be inherited)
// - [2026-10-07] type=fix (Fix "Cannot read properties of undefined (reading 'setApplicationMenu')" on npm start - ELECTRON_RUN_AS_NODE=1 inherited from VS Code/Claude Code etc. made Electron run as plain Node, so the electron module (app/Menu etc.) was empty. Clear this variable and then launch electron)

'use strict';

const { spawn } = require('child_process');
const path = require('path');

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const mode = process.argv[2] === 'dev' ? 'dev' : 'preview';
// electron-vite's "exports" does not expose ./bin/* (Node rejects the subpath with ERR_PACKAGE_PATH_NOT_EXPORTED) - resolve the
// exported package.json and take the CLI path from its "bin" field instead
const pkgJson = require.resolve('electron-vite/package.json');
const binRel = require(pkgJson).bin;
const cli = path.join(path.dirname(pkgJson), typeof binRel === 'string' ? binRel : binRel['electron-vite']);
const extra = mode === 'preview' ? ['--skipBuild'] : [];

const child = spawn(process.execPath, [cli, mode, ...extra, ...process.argv.slice(3)], {
  cwd: path.resolve(__dirname, '..'),
  env,
  stdio: 'inherit',
  windowsHide: false,
});
child.on('close', (code, signal) => process.exit(signal ? 1 : code ?? 0));
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => { if (!child.killed) child.kill(sig); });
}
