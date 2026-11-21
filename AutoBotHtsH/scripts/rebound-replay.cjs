// Update log
// - [2026-11-21] type=feat (`npm run replay -- <file> [options]`: bundles src/shared/reboundReplay.ts with esbuild (already installed
//   with electron-vite) into node_modules/.cache and runs its main() in plain Node - the replay uses the same detector as the app)

'use strict';

const path = require('path');
const fs = require('fs');
const esbuild = require('esbuild');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'node_modules', '.cache', 'rebound');
const outfile = path.join(outDir, 'replay.cjs');
fs.mkdirSync(outDir, { recursive: true });
esbuild.buildSync({
  entryPoints: [path.join(root, 'src', 'shared', 'reboundReplay.ts')],
  outfile,
  bundle: true,
  platform: 'node',
  format: 'cjs',
  logLevel: 'error',
});
process.exit(require(outfile).main(process.argv.slice(2)));
