/*
 * Update log
 * - [2026-10-26] type=fix - 'Skip this version' remembers the release label (v1.0.3_003 - info.label, falls back to info.version) instead of
 *   X.Y.Z only, since updater.js now also offers a higher build of the same X.Y.Z: skipping _003 no longer hides _004
 * - [2026-10-15] type=feat - New self-contained update dialog (shared by the make_utils Electron programs, same file in each - pairs with
 *   updater.js in main). Builds its own DOM/CSS (prefixed .uh-*), so it only needs a <script src="update-ui.js"> tag:
 *   - IPC through window.updater (exposed by preload when contextIsolation is on) or require('electron').ipcRenderer (nodeIntegration apps)
 *   - 'update-available' from the startup check opens the dialog unless the user chose "Skip this version" (localStorage 'skippedUpdateVersion')
 *   - adds a "Check for updates" link after the version text (#appVersion / .app-version) if the page has one
 *   - stages: available → Download (progress bar) → downloaded → Restart now / Install on quit; portable opens the download in the browser
 *   - Korean when <html lang="ko">, English otherwise; light/dark follows the page background; z-index above every app modal
 */
(function () {
  'use strict';

  const bridge = (() => {
    if (window.updater) return window.updater;
    try {
      const { ipcRenderer } = require('electron');
      return {
        check: () => ipcRenderer.invoke('update-check'),
        download: () => ipcRenderer.invoke('update-download'),
        install: () => ipcRenderer.send('update-install'),
        onAvailable: (cb) => ipcRenderer.on('update-available', (_e, info) => cb(info)),
        onProgress: (cb) => ipcRenderer.on('update-progress', (_e, p) => cb(p))
      };
    } catch (e) {
      return null;
    }
  })();
  if (!bridge) return;

  const KO = {
    title: '업데이트 있음',
    readyTitle: '업데이트 준비 완료',
    desc: '새 버전 {latest}이(가) 있습니다. (현재 {current})',
    readyDesc: '{latest} 다운로드가 끝났습니다. 프로그램을 재시작하면 설치됩니다.',
    download: '다운로드',
    openDownload: '다운로드 페이지 열기',
    portableHint: 'Portable 버전은 자동 설치가 되지 않습니다. 새 파일을 받아 기존 파일과 교체하세요.',
    downloading: '다운로드 중...',
    restart: '지금 재시작',
    later: '나중에',
    laterOnQuit: '종료할 때 설치',
    retry: '다시 시도',
    skip: '이 버전 건너뛰기',
    failed: '다운로드 실패: {message}',
    check: '업데이트 확인',
    checking: '확인 중...',
    upToDate: '최신 버전입니다',
    devMode: '개발 모드에서는 확인하지 않음',
    checkFailed: '업데이트 확인 실패',
    close: '닫기'
  };
  const EN = {
    title: 'Update available',
    readyTitle: 'Update ready',
    desc: 'A new version {latest} is available. (current {current})',
    readyDesc: '{latest} has been downloaded. Restart the program to install it.',
    download: 'Download',
    openDownload: 'Open download page',
    portableHint: 'The portable version cannot update itself. Download the new file and replace the old one.',
    downloading: 'Downloading...',
    restart: 'Restart now',
    later: 'Later',
    laterOnQuit: 'Install when I quit',
    retry: 'Retry',
    skip: 'Skip this version',
    failed: 'Download failed: {message}',
    check: 'Check for updates',
    checking: 'Checking...',
    upToDate: 'You are up to date',
    devMode: 'Not checked in development mode',
    checkFailed: 'Update check failed',
    close: 'Close'
  };
  const t = (key, vars) => {
    const dict = /^ko/i.test(document.documentElement.lang || '') ? KO : EN;
    return String(dict[key] || key).replace(/\{(\w+)\}/g, (_m, k) => (vars && vars[k] != null ? vars[k] : ''));
  };

  const CSS = `
.uh-overlay { position:fixed; inset:0; background:rgba(0,0,0,.55); display:none; align-items:center; justify-content:center; z-index:2147483000; font-family:'Segoe UI','Malgun Gothic',sans-serif; }
.uh-overlay.uh-open { display:flex; }
.uh-box { --uh-bg:#fff; --uh-bg2:#f1f3f6; --uh-text:#1f2328; --uh-muted:#6b7280; --uh-accent:#4c9aff;
  position:relative; width:400px; max-width:90vw; background:var(--uh-bg); color:var(--uh-text); border-radius:10px; overflow:hidden; box-shadow:0 12px 40px rgba(0,0,0,.35); }
.uh-box.uh-dark { --uh-bg:#262a31; --uh-bg2:#1d2026; --uh-text:#e6e8eb; --uh-muted:#9aa1ab; }
.uh-head { position:relative; padding:16px 44px 16px 18px; background:#1c2333; color:#fff; font-size:15px; font-weight:600; }
.uh-x { position:absolute; top:8px; right:8px; width:32px; height:32px; border:none; border-radius:6px; background:none; color:#fff; font-size:22px; line-height:1; cursor:pointer; }
.uh-x:hover { background:rgba(255,255,255,.15); }
.uh-body { padding:18px; }
.uh-desc { font-size:13px; color:var(--uh-muted); margin-bottom:12px; line-height:1.5; }
.uh-notes { font-size:12px; background:var(--uh-bg2); border-radius:8px; padding:10px 12px; margin-bottom:12px; max-height:160px; overflow:auto; white-space:pre-wrap; line-height:1.5; }
.uh-progress { height:6px; border-radius:3px; background:var(--uh-bg2); overflow:hidden; margin-bottom:8px; }
.uh-progress > div { height:100%; width:0; background:var(--uh-accent); transition:width .2s; }
.uh-status { font-size:12px; color:var(--uh-muted); min-height:16px; margin-bottom:12px; }
.uh-status.uh-error { color:#ef4444; }
.uh-actions { display:flex; justify-content:flex-end; align-items:center; gap:6px; flex-wrap:wrap; }
.uh-actions button { margin:0; border:none; border-radius:8px; padding:9px 16px; font-size:13px; cursor:pointer; background:var(--uh-bg2); color:var(--uh-text); font-family:inherit; }
.uh-actions button.uh-primary { background:var(--uh-accent); color:#fff; font-weight:600; }
.uh-actions button:disabled { opacity:.6; cursor:default; }
.uh-actions button.uh-skip { margin-right:auto; background:none; color:var(--uh-muted); padding-left:4px; padding-right:4px; }
.uh-actions button.uh-skip:hover { color:var(--uh-text); }
.uh-hidden { display:none !important; }
.uh-check { margin:0 0 0 8px; padding:0; border:none; background:none; font:inherit; font-size:inherit; color:#4c9aff; cursor:pointer; }
.uh-check:hover { text-decoration:underline; }
.uh-check:disabled { color:inherit; opacity:.7; text-decoration:none; cursor:default; }
`;

  let els = null;
  let info = null;
  let stage = 'idle';   // available | downloading | downloaded | failed

  function build() {
    if (els) return els;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    const overlay = document.createElement('div');
    overlay.className = 'uh-overlay';
    overlay.innerHTML = `
      <div class="uh-box" role="dialog" aria-modal="true">
        <div class="uh-head"><span class="uh-title"></span><button type="button" class="uh-x">×</button></div>
        <div class="uh-body">
          <div class="uh-desc"></div>
          <div class="uh-notes uh-hidden"></div>
          <div class="uh-progress uh-hidden"><div></div></div>
          <div class="uh-status"></div>
          <div class="uh-actions">
            <button type="button" class="uh-skip"></button>
            <button type="button" class="uh-later"></button>
            <button type="button" class="uh-primary"></button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const q = (s) => overlay.querySelector(s);
    els = {
      overlay, box: q('.uh-box'), title: q('.uh-title'), x: q('.uh-x'), desc: q('.uh-desc'), notes: q('.uh-notes'),
      progress: q('.uh-progress'), bar: q('.uh-progress > div'), status: q('.uh-status'),
      skip: q('.uh-skip'), later: q('.uh-later'), primary: q('.uh-primary')
    };
    els.x.addEventListener('click', close);
    els.later.addEventListener('click', close);
    els.skip.addEventListener('click', () => {
      if (info) { try { localStorage.setItem('skippedUpdateVersion', info.label || info.version); } catch (e) {} }
      close();
    });
    els.primary.addEventListener('click', onPrimary);
    // ESC closes this dialog first (capture phase, before the app's own ESC handling)
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && overlay.classList.contains('uh-open')) {
        e.preventDefault();
        e.stopPropagation();
        close();
      }
    }, true);
    return els;
  }

  // Light/dark from the page background, so the dialog matches whatever theme the app is using
  function isDarkPage() {
    const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?/.exec(getComputedStyle(document.body).backgroundColor || '');
    if (!m || m[4] === '0') return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return (0.299 * m[1] + 0.587 * m[2] + 0.114 * m[3]) < 128;
  }

  function render() {
    if (!info) return;
    const e = build();
    e.box.classList.toggle('uh-dark', isDarkPage());
    e.x.title = t('close');
    e.title.textContent = t(stage === 'downloaded' ? 'readyTitle' : 'title');
    e.desc.textContent = stage === 'downloaded'
      ? t('readyDesc', { latest: info.label })
      : t('desc', { latest: info.label, current: info.current });
    e.notes.textContent = info.releaseNotes || '';
    e.notes.classList.toggle('uh-hidden', !info.releaseNotes || stage === 'downloaded');
    e.progress.classList.toggle('uh-hidden', stage !== 'downloading');
    e.status.classList.toggle('uh-error', stage === 'failed');
    if (stage !== 'downloading' && stage !== 'failed') e.status.textContent = info.portable && stage === 'available' ? t('portableHint') : '';
    e.skip.textContent = t('skip');
    e.skip.classList.toggle('uh-hidden', stage !== 'available');
    e.later.textContent = t(stage === 'downloaded' ? 'laterOnQuit' : 'later');
    e.primary.disabled = stage === 'downloading';
    e.primary.textContent = t(
      stage === 'downloaded' ? 'restart'
        : stage === 'downloading' ? 'downloading'
        : stage === 'failed' ? 'retry'
        : info.portable ? 'openDownload' : 'download'
    );
  }

  function open(newInfo) {
    // A new check result replaces the info, but never resets a download that is running or finished
    if (stage !== 'downloading' && stage !== 'downloaded') {
      info = newInfo;
      stage = 'available';
    }
    render();
    els.overlay.classList.add('uh-open');
  }

  function close() {
    if (els) els.overlay.classList.remove('uh-open');
  }

  async function onPrimary() {
    if (!info) return;
    if (stage === 'downloaded') { bridge.install(); return; }
    if (stage === 'downloading') return;
    stage = 'downloading';
    els.bar.style.width = '0%';
    els.status.textContent = t('downloading');
    render();
    let r;
    try { r = await bridge.download(); } catch (err) { r = { ok: false, message: err.message }; }
    if (r && r.opened) {          // portable - the browser takes over the download
      stage = 'available';
      close();
      return;
    }
    if (r && r.ok) {
      stage = 'downloaded';
      render();
      els.overlay.classList.add('uh-open');   // reopen if it was closed while downloading
    } else {
      stage = 'failed';
      els.status.textContent = t('failed', { message: (r && r.message) || '' });
      render();
    }
  }

  bridge.onProgress((p) => {
    if (!els) return;
    const pct = Math.max(0, Math.min(100, p.percent || 0));
    els.bar.style.width = pct.toFixed(1) + '%';
    const mb = (n) => (n / 1048576).toFixed(1);
    els.status.textContent = p.total
      ? `${t('downloading')} ${pct.toFixed(0)}% (${mb(p.transferred || 0)} / ${mb(p.total)} MB)`
      : t('downloading');
  });

  bridge.onAvailable((newInfo) => {
    let skipped = '';
    try { skipped = localStorage.getItem('skippedUpdateVersion') || ''; } catch (e) {}
    if (newInfo && (newInfo.label || newInfo.version) === skipped) return;
    open(newInfo);
  });

  async function checkManual(btn) {
    if (btn.disabled) return;
    btn.disabled = true;
    btn.textContent = t('checking');
    let r;
    try { r = await bridge.check(); } catch (e) { r = { status: 'error', message: e.message }; }
    if (r.status === 'available') {
      btn.disabled = false;
      btn.textContent = t('check');
      open(r.info);
      return;
    }
    btn.textContent = r.status === 'none' ? t('upToDate') : r.status === 'dev' ? t('devMode') : t('checkFailed');
    btn.title = r.status === 'error' ? (r.message || '') : '';
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = t('check');
      btn.title = '';
    }, 4000);
  }

  // "Check for updates" link right after the version text (Settings footer) - only in apps that show a version
  function addCheckLinks() {
    document.querySelectorAll('#appVersion, .app-version').forEach((el) => {
      if (el.dataset.uhLinked || el.querySelector('.uh-check')) return;
      el.dataset.uhLinked = '1';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'uh-check';
      btn.textContent = t('check');
      btn.addEventListener('click', (e) => { e.stopPropagation(); checkManual(btn); });
      el.insertAdjacentElement('afterend', btn);
    });
  }

  window.UpdateUI = { open, close, check: () => bridge.check() };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => { build(); addCheckLinks(); });
  else { build(); addCheckLinks(); }
})();
