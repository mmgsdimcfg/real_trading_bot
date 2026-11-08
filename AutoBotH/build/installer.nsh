; Custom NSIS script - electron-builder includes build/installer.nsh automatically (nsis.include default)
;
; Update log
; - [2026-10-25] type=fix - Fix blurry installer/uninstaller text on high-DPI displays (e.g. 2560x1600 at 150%+ scaling) -
;   the NSIS exe had no DPI-aware manifest, so Windows rendered it at 96 DPI and bitmap-stretched it. Mark it DPI-aware

ManifestDPIAware true
