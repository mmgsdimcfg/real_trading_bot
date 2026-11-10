// Update log
// - [2026-11-10] type=fix (release.config.json is searched in the parent folders - the repository moved to make_utils/Trading/real_trading_bot, and the fixed '..' never reached make_utils)
// - [2026-10-26] type=feat (--yml-only: only completes dist/latest.yml (buildNumber/portableFile/releaseNotes) without uploading and without needing release.config.json - run by postdist, so a plain npm run dist already gives the latest.yml to copy to the update site by hand)
// - [2026-10-15] type=feat (Shared by every make_utils program: the .blockmap is uploaded only when present - BBoxH (PyInstaller/Inno Setup) has none)
// - [2026-10-15] type=feat (Release upload for the self-hosted update feed https://update.iccid.dev/<Folder>/: copies the installer, its .blockmap,
//   the portable exe and latest.yml from dist/ to <target>/<Folder>/, latest.yml last so clients never see a version whose installer is not there yet.
//   latest.yml gets buildNumber/portableFile (and releaseNotes from release-notes.txt if present) appended - read by updater.js.
//   The upload target comes from make_utils/release.config.json (shared by all programs, outside every repo):
//     { "target": "D:\\updates" }                      local folder / network share served as the update site
//     { "target": "user@n9x2k.iccid.dev:/srv/updates" }  scp (uses your ssh config - e.g. ProxyCommand cloudflared)
//   Can also be overridden with the UPDATE_TARGET environment variable)
//
// Usage: npm run release   (= npm run dist, then this script)
//        node scripts/release.cjs   (upload what is already in dist/)
//        node scripts/release.cjs --yml-only   (complete dist/latest.yml only - npm postdist)

'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const folderName = path.basename(root);   // program name = folder name (TermH) = sub-folder on the update site

function fail(msg) {
  console.error(`[release] ${msg}`);
  process.exit(1);
}

const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const version = pkg.version;
const buildNumber = (pkg.build && pkg.build.buildNumber) || '';
const outDir = path.join(root, (pkg.build && pkg.build.directories && pkg.build.directories.output) || 'dist');
const ymlOnly = process.argv.includes('--yml-only');

// ---- Upload target ----
let target = process.env.UPDATE_TARGET || '';
// make_utils/release.config.json is looked up in the parent folders - AutoBotH sits deeper than the other apps
// (make_utils/Trading/real_trading_bot/AutoBotH), so a fixed '..' pointed at a file that does not exist
const configFile = (() => {
  let dir = path.dirname(root);
  for (let i = 0; i < 4; i++) {
    const f = path.join(dir, 'release.config.json');
    if (fs.existsSync(f)) return f;
    dir = path.dirname(dir);
  }
  return path.resolve(root, '..', 'release.config.json');
})();
if (!target && fs.existsSync(configFile)) {
  try { target = JSON.parse(fs.readFileSync(configFile, 'utf8')).target || ''; } catch (e) { fail(`release.config.json 읽기 실패: ${e.message}`); }
}
if (!target && !ymlOnly) {
  fail(`업로드 대상이 없습니다. ${configFile} 파일을 만들어 주세요.\n` +
    '  예) { "target": "D:\\\\updates" }  또는  { "target": "user@n9x2k.iccid.dev:/srv/updates" }');
}

// ---- Files to publish ----
const ymlPath = path.join(outDir, 'latest.yml');
if (!fs.existsSync(ymlPath)) fail(`${ymlPath}가 없습니다. 먼저 npm run dist를 실행하세요`);
let yml = fs.readFileSync(ymlPath, 'utf8');
const ymlVersion = (/^version:\s*(\S+)/m.exec(yml) || [])[1];
if (ymlVersion !== version) fail(`dist/latest.yml(${ymlVersion})이 현재 버전(${version})과 다릅니다. npm run dist를 다시 실행하세요`);

const installer = ((/^path:\s*(.+)$/m.exec(yml) || [])[1] || '').trim();
if (!installer) fail('latest.yml에 path 항목이 없습니다');
const portable = `${folderName}_Setup_v${version}_${buildNumber}_Portable.exe`;
const files = [installer];
if (fs.existsSync(path.join(outDir, installer + '.blockmap'))) files.push(installer + '.blockmap');   // electron-builder only (differential download)
if (fs.existsSync(path.join(outDir, portable))) files.push(portable);
for (const f of files) {
  if (!fs.existsSync(path.join(outDir, f))) fail(`${f}가 dist에 없습니다`);
}

// ---- latest.yml extras (idempotent - previous extras are removed first) ----
yml = yml
  .replace(/^buildNumber:.*\r?\n/m, '')
  .replace(/^portableFile:.*\r?\n/m, '')
  .replace(/^releaseNotes:.*\r?\n(?:[ \t]+.*\r?\n?)*/m, '');
if (!yml.endsWith('\n')) yml += '\n';
yml += `buildNumber: '${buildNumber}'\n`;
if (files.includes(portable)) yml += `portableFile: ${portable}\n`;
const notesFile = path.join(root, 'release-notes.txt');
if (fs.existsSync(notesFile)) {
  const notes = fs.readFileSync(notesFile, 'utf8').replace(/\r/g, '').trim();
  if (notes) yml += 'releaseNotes: |-\n' + notes.split('\n').map(l => '  ' + l).join('\n') + '\n';
}
fs.writeFileSync(ymlPath, yml);
if (ymlOnly) {
  console.log(`[release] dist/latest.yml 갱신 - buildNumber ${buildNumber}${files.includes(portable) ? ", portableFile " + portable : ""} (업로드 안 함)`);
  process.exit(0);
}

// ---- Upload ----
const isScp = /^[^\\/:]+@[^:]+:/.test(target) || /^[a-z0-9.-]{2,}:\//i.test(target);
console.log(`[release] ${folderName} v${version}_${buildNumber} → ${target}${isScp ? ' (scp)' : ''}`);

function run(cmd, args) {
  const r = spawnSync(cmd, args, { stdio: 'inherit' });
  if (r.error) fail(`${cmd} 실행 실패: ${r.error.message}`);
  if (r.status !== 0) fail(`${cmd} 종료 코드 ${r.status}`);
}

if (isScp) {
  const sep = target.indexOf(':');
  const host = target.slice(0, sep);
  const remoteDir = target.slice(sep + 1).replace(/\/+$/, '') + '/' + folderName;
  run('ssh', [host, `mkdir -p '${remoteDir}'`]);
  run('scp', [...files.map(f => path.join(outDir, f)), `${host}:${remoteDir}/`]);
  run('scp', [ymlPath, `${host}:${remoteDir}/latest.yml`]);
} else {
  const destDir = path.join(target, folderName);
  fs.mkdirSync(destDir, { recursive: true });
  for (const f of files) {
    console.log(`  - ${f}`);
    fs.copyFileSync(path.join(outDir, f), path.join(destDir, f));
  }
  // latest.yml last, through a temp name, so a client never reads a half-written file
  const tmp = path.join(destDir, 'latest.yml.tmp');
  fs.copyFileSync(ymlPath, tmp);
  fs.renameSync(tmp, path.join(destDir, 'latest.yml'));
  console.log('  - latest.yml');
}
console.log(`[release] 완료 - https://update.iccid.dev/${folderName}/latest.yml`);
