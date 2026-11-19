// Update log
// - [2026-11-05] type=fix (electron-builder wrapper for npm run dist: on Windows, NsisTarget.computeScriptAndSignUninstaller runs the
//   freshly built, unsigned uninstaller stub (execWine -> spawn) to make it write the uninstaller. Windows Smart App Control blocks that
//   exe at random (it judges each new unsigned file by hash), and the build then dies with "spawn UNKNOWN". electron-builder already has
//   a path that does not run the stub: on macOS Catalina+ it extracts the uninstaller from the stub with UninstallerReader (parses the
//   NSIS data, no execution). isMacOsCatalina() is used only at that spot, so this wrapper makes it return true on Windows and then
//   starts the normal electron-builder CLI with the same arguments. If UninstallerReader ever fails, electron-builder falls back to
//   running the stub as before)
//
// Usage: node scripts/electron-builder.cjs <electron-builder args>   (package.json "dist")

'use strict';

if (process.platform === 'win32') {
  const macosVersion = require('app-builder-lib/out/util/macosVersion');
  macosVersion.isMacOsCatalina = () => true;
}

require('electron-builder/cli');
