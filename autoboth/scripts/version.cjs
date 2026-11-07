// Update log
// - [2026-10-07] type=feat (Introduce version input file: applies the single line in version.txt (e.g. 1.0.0_001) to package.json version (1.0.0) / build.buildNumber (001). Runs automatically before npm start/dist - prestart/predist)
//
// Usage: edit the version in version.txt, then run npm run dist (or npm start)
//   Format: <major>.<minor>.<patch>_<3-digit build number>   e.g. 1.0.0_001, 1.2.0_015  (a leading v is allowed)
//   Output: dist/<name>_Setup_v1.0.0_001.exe, dist/<name>_Setup_v1.0.0_001_Portable.exe
// Run directly: node scripts/version.cjs

'use strict';

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const versionFile = path.join(root, 'version.txt');
const pkgFile = path.join(root, 'package.json');

function fail(msg) {
  console.error(`[version] ${msg}`);
  process.exit(1);
}

if (!fs.existsSync(versionFile)) fail(`version.txt가 없습니다: ${versionFile}`);

// Ignore comments (#) and blank lines; use only the first value line
const line = fs.readFileSync(versionFile, 'utf8')
  .split(/\r?\n/)
  .map((s) => s.trim())
  .find((s) => s && !s.startsWith('#'));
const m = /^v?(\d+\.\d+\.\d+)_(\d{3})$/.exec(line || '');
if (!m) fail(`version.txt 형식이 잘못되었습니다: "${line || ''}" (예: 1.0.0_001)`);
const [, version, buildNumber] = m;

// Rewriting with JSON.stringify would change existing line breaks/layout, so only the values are replaced as strings
let text = fs.readFileSync(pkgFile, 'utf8');
const before = text;
const versionRe = /^(\s*"version"\s*:\s*")[^"]*(")/m;        // top-level version (first key that appears in the file)
const buildNumberRe = /("buildNumber"\s*:\s*")[^"]*(")/;
if (!buildNumberRe.test(text)) fail('package.json에 build.buildNumber가 없습니다');
text = text.replace(versionRe, `$1${version}$2`).replace(buildNumberRe, `$1${buildNumber}$2`);

const pkg = JSON.parse(text);
if (pkg.version !== version || !pkg.build || pkg.build.buildNumber !== buildNumber) {
  fail('package.json 반영에 실패했습니다(version / build.buildNumber 위치 확인 필요)');
}
if (text !== before) fs.writeFileSync(pkgFile, text);
console.log(`[version] v${version}_${buildNumber}${text !== before ? ' (package.json 갱신)' : ''}`);
