/*
 * Update log
 * - [2026-10-23] type=feat - New module (shared by the make_utils Electron programs, same file in each - pairs with app-menu.js in the
 *   renderer). initAppAbout({ productName }) registers the IPC used by the About dialog:
 *   'app-about:info'          -> { name, version (v<version>_<build> from version.txt), electron, chrome, node, os, year, websiteLabel }
 *   'app-about:open-website'  -> opens https://www.iccid.dev in the default browser
 *   'app-about:contact'       -> opens mailto:<developer> with the subject/body built by the page
 *   'app-about:copy-email'    -> copies the developer address to the clipboard
 *   The developer address lives only here (main process) - it is never sent to the page.
 *   contextIsolation apps expose these through preload as window.appAbout:
 *     contextBridge.exposeInMainWorld('appAbout', {
 *       info: () => ipcRenderer.invoke('app-about:info'),
 *       openWebsite: () => ipcRenderer.invoke('app-about:open-website'),
 *       contact: (msg) => ipcRenderer.invoke('app-about:contact', msg),
 *       copyEmail: () => ipcRenderer.invoke('app-about:copy-email'),
 *     });
 */
'use strict';

const { app, ipcMain, shell, clipboard } = require('electron');
const fs = require('fs');
const os = require('os');
const path = require('path');

const WEBSITE_URL = 'https://www.iccid.dev';
const WEBSITE_LABEL = 'www.iccid.dev';
const DEVELOPER_EMAIL = 'xgraph.kr@gmail.com';

// "v1.0.0_001" from version.txt next to this file (falls back to the semver of package.json)
function versionLabel() {
  try {
    const line = fs.readFileSync(path.join(__dirname, 'version.txt'), 'utf8')
      .split(/\r?\n/).map((s) => s.trim()).find((s) => s && !s.startsWith('#'));
    const m = /^v?(\d+\.\d+\.\d+_\d{3})$/.exec(line || '');
    if (m) return `v${m[1]}`;
  } catch (e) { /* ignore */ }
  return `v${app.getVersion()}`;
}

let registered = false;
function initAppAbout({ productName } = {}) {
  if (registered) return;
  registered = true;
  ipcMain.handle('app-about:info', () => ({
    name: productName || app.getName(),
    version: versionLabel(),
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
    os: `Windows ${os.release()} (${os.arch()})`,
    year: new Date().getFullYear(),
    websiteLabel: WEBSITE_LABEL,
  }));
  ipcMain.handle('app-about:open-website', () => shell.openExternal(WEBSITE_URL));
  ipcMain.handle('app-about:contact', (_e, { subject, body } = {}) => {
    const q = `subject=${encodeURIComponent(subject || productName || 'Feedback')}&body=${encodeURIComponent(body || '')}`;
    return shell.openExternal(`mailto:${DEVELOPER_EMAIL}?${q}`);
  });
  ipcMain.handle('app-about:copy-email', () => { clipboard.writeText(DEVELOPER_EMAIL); return true; });
}

module.exports = { initAppAbout, versionLabel };
