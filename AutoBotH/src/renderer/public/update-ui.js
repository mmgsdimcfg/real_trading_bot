/*
 * Update log
 * - [2026-11-23] type=feat - Redesigned dialog (design A, same as TermH / BBoxH): a dark card with its own header (icon, short title, old ->
 *   new version chips) and one screen per stage - available (what is new with NEW / FIX / UPD tags and "show all", installer size,
 *   Later / Update now, skip link) -> downloading (big percent, size, speed and time left, gradient bar, "restart automatically when
 *   done", Cancel / Continue in the background) -> downloaded (green box, On next start / Restart now); failed shows the reason with
 *   Retry. One filled primary and one grey secondary button per stage. Cancel needs updater.js 'update-cancel' (bridge.cancel) - the
 *   button is hidden with an older preload. Closing the card while downloading keeps the download running; the card comes back when
 *   it is done (or the app restarts at once when the box is ticked). Release notes lines "NEW: ..." / "FIX: ..." come from
 *   scripts/release.cjs (update log of package.json); other lines are shown as UPD
 * - [2026-11-18] type=fix - Progress text of a differential download says so: "(9.8 / 12.6 MB) - changed parts only, full 112.2 MB"
 *   (p.full from updater.js = installer size; shown when total is clearly smaller than full)
 * - [2026-10-26] type=fix - 'Skip this version' remembers the release label (v1.0.3_003 - info.label, falls back to info.version) instead of
 *   X.Y.Z only, since updater.js now also offers a higher build of the same X.Y.Z: skipping _003 no longer hides _004
 * - [2026-10-15] type=feat - New self-contained update dialog (shared by the make_utils Electron programs, same file in each - pairs with
 *   updater.js in main). Builds its own DOM/CSS (prefixed .uh-*), so it only needs a <script src="update-ui.js"> tag:
 *   - IPC through window.updater (exposed by preload when contextIsolation is on) or require('electron').ipcRenderer (nodeIntegration apps)
 *   - 'update-available' from the startup check opens the dialog unless the user chose "Skip this version" (localStorage 'skippedUpdateVersion')
 *   - adds a "Check for updates" link after the version text (#appVersion / .app-version) if the page has one
 *   - Korean when <html lang="ko">, English otherwise; z-index above every app modal
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
        cancel: () => ipcRenderer.send('update-cancel'),
        onAvailable: (cb) => ipcRenderer.on('update-available', (_e, info) => cb(info)),
        onProgress: (cb) => ipcRenderer.on('update-progress', (_e, p) => cb(p))
      };
    } catch (e) {
      return null;
    }
  })();
  if (!bridge) return;
  const canCancel = typeof bridge.cancel === 'function';

  const KO = {
    newTitle: '새 버전이 준비됐어요',
    downloadingTitle: '업데이트 다운로드 중',
    readyTitle: '업데이트 준비 완료',
    failedTitle: '업데이트를 받지 못했어요',
    thisRelease: '이번 업데이트',
    showAll: '전체 보기 ›',
    collapse: '접기',
    sizeLine: '설치 파일 {size} MB · 가능하면 바뀐 부분만 받습니다',
    portableHint: 'Portable 버전은 자동 설치가 되지 않습니다. 새 파일을 받아 기존 파일과 교체하세요.',
    now: '지금 업데이트',
    openDownload: '다운로드 페이지 열기',
    later: '나중에',
    skip: '이 버전 건너뛰기',
    autoRestart: '완료되면 자동으로 재시작',
    cancel: '취소',
    background: '백그라운드에서 계속',
    connecting: '연결 중…',
    diffOnly: '바뀐 부분만, 전체 {full} MB',
    minLeft: '약 {n}분 남음',
    secLeft: '{n}초 남음',
    readyBody: '재시작하면 새 버전이 설치됩니다.\n저장하지 않은 작업이 있다면 먼저 저장하세요.',
    laterOnQuit: '다음 실행 시',
    restart: '지금 재시작',
    failed: '다운로드 실패: {message}',
    close: '닫기',
    retry: '다시 시도',
    check: '업데이트 확인',
    checking: '확인 중...',
    upToDate: '최신 버전입니다',
    devMode: '개발 모드에서는 확인하지 않음',
    checkFailed: '업데이트 확인 실패'
  };
  const EN = {
    newTitle: 'A new version is ready',
    downloadingTitle: 'Downloading the update',
    readyTitle: 'Update ready',
    failedTitle: 'The update could not be downloaded',
    thisRelease: 'This update',
    showAll: 'Show all ›',
    collapse: 'Collapse',
    sizeLine: 'Installer {size} MB · only the changed parts are downloaded when possible',
    portableHint: 'The portable version cannot update itself. Download the new file and replace the old one.',
    now: 'Update now',
    openDownload: 'Open download page',
    later: 'Later',
    skip: 'Skip this version',
    autoRestart: 'Restart automatically when done',
    cancel: 'Cancel',
    background: 'Continue in the background',
    connecting: 'Connecting…',
    diffOnly: 'changed parts only, full {full} MB',
    minLeft: 'about {n} min left',
    secLeft: '{n} s left',
    readyBody: 'Restarting installs the new version.\nSave your work first if anything is unsaved.',
    laterOnQuit: 'On next start',
    restart: 'Restart now',
    failed: 'Download failed: {message}',
    close: 'Close',
    retry: 'Retry',
    check: 'Check for updates',
    checking: 'Checking...',
    upToDate: 'You are up to date',
    devMode: 'Not checked in development mode',
    checkFailed: 'Update check failed'
  };
  const t = (key, vars) => {
    const dict = /^ko/i.test(document.documentElement.lang || '') ? KO : EN;
    return String(dict[key] || key).replace(/\{(\w+)\}/g, (_m, k) => (vars && vars[k] != null ? vars[k] : ''));
  };
  const mb = (n) => (n / 1048576).toFixed(1);

  const ICON_DOWN = '<svg viewBox="0 0 24 24"><path d="M12 4v11"/><path d="M7 11l5 5 5-5"/><path d="M5 20h14"/></svg>';
  const ICON_OK = '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  const ICON_ERR = '<svg viewBox="0 0 24 24"><path d="M12 7v6"/><path d="M12 17v.5"/></svg>';

  const CSS = `
.uh-overlay { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:center; justify-content:center; z-index:2147483000; font-family:'Segoe UI','Malgun Gothic',sans-serif; }
.uh-overlay.uh-open { display:flex; }
.uh-card { position:relative; box-sizing:border-box; width:400px; max-width:92vw; background:#141926; color:#e8ecf4; border:1px solid #262d3d; border-radius:18px; padding:22px; box-shadow:0 24px 60px rgba(0,0,0,.55); text-align:left; line-height:1.4; }
.uh-card *, .uh-card *::before, .uh-card *::after { box-sizing:border-box; }
.uh-card button { font-family:inherit; margin:0; }
.uh-x { position:absolute; top:14px; right:14px; width:28px; height:28px; border-radius:8px; border:1px solid #2a3142; background:#1b2130; color:#8b94a7; font-size:16px; line-height:1; cursor:pointer; padding:0; }
.uh-x:hover { color:#fff; background:#252c3d; }
.uh-head { display:flex; align-items:center; gap:14px; margin-bottom:16px; padding-right:34px; }
.uh-icon { width:46px; height:46px; border-radius:13px; flex:none; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#4f8cff,#7c5cff); }
.uh-card[data-stage="downloaded"] .uh-icon { background:linear-gradient(135deg,#22c55e,#16a34a); }
.uh-card[data-stage="failed"] .uh-icon { background:linear-gradient(135deg,#f87171,#dc2626); }
.uh-icon svg { width:22px; height:22px; stroke:#fff; fill:none; stroke-width:2.4; stroke-linecap:round; stroke-linejoin:round; }
.uh-title { font-size:15px; font-weight:700; margin-bottom:6px; color:#e8ecf4; }
.uh-vers { display:flex; align-items:center; gap:7px; font-family:Consolas,monospace; font-size:11.5px; }
.uh-old { color:#6b7487; text-decoration:line-through; }
.uh-old:empty { display:none; }
.uh-arrow { color:#6b7487; }
.uh-chip { padding:1px 8px; border-radius:9px; border:1px solid #33508f; background:#1a2747; color:#8fb4ff; font-weight:700; }
.uh-card[data-stage="downloaded"] .uh-chip { border-color:#1f6b3d; background:#11301f; color:#6ee7a0; }
.uh-stage { display:none; }
.uh-card[data-stage="available"] .uh-stage[data-for="available"],
.uh-card[data-stage="downloading"] .uh-stage[data-for="downloading"],
.uh-card[data-stage="downloaded"] .uh-stage[data-for="downloaded"],
.uh-card[data-stage="failed"] .uh-stage[data-for="failed"] { display:block; }
.uh-notes { background:#0f131d; border:1px solid #232a3a; border-radius:12px; padding:12px 14px; margin-bottom:12px; }
.uh-notes-head { display:flex; justify-content:space-between; align-items:center; font-size:11.5px; color:#8b94a7; margin-bottom:8px; }
.uh-notes-head button { background:none; border:none; color:#6ea0ff; font-size:11.5px; font-weight:600; cursor:pointer; padding:0; }
.uh-notes-head button:hover { color:#9dbfff; }
.uh-notes-list { display:flex; flex-direction:column; gap:6px; max-height:74px; overflow:hidden; }
.uh-notes.uh-wide .uh-notes-list { max-height:240px; overflow-y:auto; }
.uh-note { display:flex; gap:8px; align-items:baseline; font-size:12.5px; line-height:1.35; }
.uh-note span:last-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.uh-notes.uh-wide .uh-note span:last-child { white-space:normal; }
.uh-tag { flex:none; font-family:Consolas,monospace; font-size:9.5px; font-weight:700; padding:1px 6px; border-radius:6px; }
.uh-tag.NEW { background:#1a2747; color:#8fb4ff; }
.uh-tag.FIX { background:#3a2a10; color:#fbbf24; }
.uh-tag.UPD { background:#252c3d; color:#aab3c5; }
.uh-meta { font-size:11.5px; color:#8b94a7; margin:0 2px 6px; }
.uh-meta:empty { display:none; }
.uh-hint { font-size:11.5px; color:#fbbf24; margin:0 2px 10px; }
.uh-hint:empty { display:none; }
.uh-actions { display:flex; gap:8px; margin-top:12px; }
.uh-btn { flex:1; height:42px; border-radius:11px; border:1px solid #2a3142; background:#1b2130; color:#e8ecf4; font-size:13px; font-weight:700; cursor:pointer; padding:0 10px; }
.uh-btn:hover { background:#252c3d; }
.uh-btn.uh-primary { border:none; background:linear-gradient(180deg,#4f8cff,#3b76f6); color:#fff; box-shadow:0 6px 18px rgba(59,118,246,.35); }
.uh-btn.uh-primary:hover, .uh-btn.uh-green:hover { filter:brightness(1.08); }
.uh-btn.uh-green { border:none; background:linear-gradient(180deg,#22c55e,#16a34a); color:#fff; box-shadow:0 6px 18px rgba(34,197,94,.3); }
.uh-btn.uh-ghost { flex:0 0 76px; border-color:transparent; background:none; color:#aab3c5; }
.uh-btn.uh-ghost:hover { background:#1b2130; color:#fff; }
.uh-skip { display:block; margin:10px auto 0; background:none; border:none; color:#6b7487; font-size:11.5px; cursor:pointer; padding:2px 6px; }
.uh-skip:hover { color:#aab3c5; }
.uh-prog-row { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:10px; }
.uh-pct b { font-size:30px; font-weight:800; letter-spacing:-.5px; color:#e8ecf4; }
.uh-pct span { font-size:12px; color:#8b94a7; margin-left:3px; }
.uh-prog-info { text-align:right; font-size:11.5px; color:#8b94a7; line-height:1.6; }
.uh-bar { height:8px; border-radius:4px; background:#232a3a; overflow:hidden; margin-bottom:14px; }
.uh-bar > div { height:100%; width:0; border-radius:4px; background:linear-gradient(90deg,#4f8cff,#8b6cff); transition:width .25s; }
.uh-check-row { display:flex; align-items:center; gap:8px; font-size:12.5px; color:#c3cad8; cursor:pointer; }
.uh-check-row input { width:15px; height:15px; margin:0; accent-color:#4f8cff; }
.uh-ready { background:#0f2418; border:1px solid #1f6b3d; border-radius:12px; padding:14px 16px; color:#86efac; font-size:12.5px; line-height:1.6; white-space:pre-line; margin-bottom:6px; }
.uh-error { background:#2a1416; border:1px solid #7f1d1d; border-radius:12px; padding:14px 16px; color:#fca5a5; font-size:12.5px; line-height:1.5; word-break:break-word; }
.uh-hidden { display:none !important; }
.uh-check { margin:0 0 0 8px; padding:0; border:none; background:none; font:inherit; font-size:inherit; color:#4c9aff; cursor:pointer; }
.uh-check:hover { text-decoration:underline; }
.uh-check:disabled { color:inherit; opacity:.7; text-decoration:none; cursor:default; }
`;

  let els = null;
  let info = null;
  let stage = 'idle';   // available | downloading | downloaded | failed
  let notesOpen = false;

  function build() {
    if (els) return els;
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    const overlay = document.createElement('div');
    overlay.className = 'uh-overlay';
    overlay.innerHTML = `
      <div class="uh-card" role="dialog" aria-modal="true" data-stage="available">
        <button type="button" class="uh-x">×</button>
        <div class="uh-head">
          <div class="uh-icon"></div>
          <div>
            <div class="uh-title"></div>
            <div class="uh-vers"><span class="uh-old"></span><span class="uh-arrow">→</span><span class="uh-chip"></span></div>
          </div>
        </div>
        <div class="uh-stage" data-for="available">
          <div class="uh-notes"><div class="uh-notes-head"><span class="uh-notes-title"></span><button type="button" class="uh-all"></button></div><div class="uh-notes-list"></div></div>
          <div class="uh-meta"></div>
          <div class="uh-hint"></div>
          <div class="uh-actions"><button type="button" class="uh-btn uh-later"></button><button type="button" class="uh-btn uh-primary uh-go"></button></div>
          <button type="button" class="uh-skip"></button>
        </div>
        <div class="uh-stage" data-for="downloading">
          <div class="uh-prog-row"><div class="uh-pct"><b>0</b><span>%</span></div><div class="uh-prog-info"><div class="uh-size"></div><div class="uh-speed"></div></div></div>
          <div class="uh-bar"><div></div></div>
          <label class="uh-check-row"><input type="checkbox" class="uh-auto"><span class="uh-auto-label"></span></label>
          <div class="uh-actions"><button type="button" class="uh-btn uh-ghost uh-cancel"></button><button type="button" class="uh-btn uh-bg"></button></div>
        </div>
        <div class="uh-stage" data-for="downloaded">
          <div class="uh-ready"></div>
          <div class="uh-actions"><button type="button" class="uh-btn uh-onquit"></button><button type="button" class="uh-btn uh-green uh-restart"></button></div>
        </div>
        <div class="uh-stage" data-for="failed">
          <div class="uh-error"></div>
          <div class="uh-actions"><button type="button" class="uh-btn uh-close"></button><button type="button" class="uh-btn uh-primary uh-retry"></button></div>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const q = (s) => overlay.querySelector(s);
    els = {
      overlay, card: q('.uh-card'), x: q('.uh-x'), icon: q('.uh-icon'), title: q('.uh-title'), old: q('.uh-old'), arrow: q('.uh-arrow'),
      chip: q('.uh-chip'), notes: q('.uh-notes'), notesTitle: q('.uh-notes-title'), all: q('.uh-all'), list: q('.uh-notes-list'),
      meta: q('.uh-meta'), hint: q('.uh-hint'), later: q('.uh-later'), go: q('.uh-go'), skip: q('.uh-skip'),
      pct: q('.uh-pct b'), size: q('.uh-size'), speed: q('.uh-speed'), bar: q('.uh-bar > div'), auto: q('.uh-auto'),
      autoLabel: q('.uh-auto-label'), cancel: q('.uh-cancel'), bg: q('.uh-bg'), ready: q('.uh-ready'), onquit: q('.uh-onquit'),
      restart: q('.uh-restart'), error: q('.uh-error'), close: q('.uh-close'), retry: q('.uh-retry')
    };
    els.x.addEventListener('click', close);
    els.later.addEventListener('click', close);
    els.bg.addEventListener('click', close);
    els.onquit.addEventListener('click', close);
    els.close.addEventListener('click', close);
    els.go.addEventListener('click', onPrimary);
    els.retry.addEventListener('click', onPrimary);
    els.restart.addEventListener('click', onPrimary);
    els.all.addEventListener('click', () => { notesOpen = !notesOpen; render(); });
    els.cancel.addEventListener('click', () => { if (stage === 'downloading' && canCancel) bridge.cancel(); });
    els.skip.addEventListener('click', () => {
      if (info) { try { localStorage.setItem('skippedUpdateVersion', info.label || info.version); } catch (e) {} }
      close();
    });
    // ESC closes this dialog first (capture phase, before the app's own ESC handling); a download keeps running
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && overlay.classList.contains('uh-open')) {
        e.preventDefault();
        e.stopPropagation();
        close();
      }
    }, true);
    return els;
  }

  /* release notes text -> [{ tag: 'NEW' | 'FIX' | 'UPD', text }] */
  function parseNotes(text) {
    return String(text || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean).map((l) => {
      const m = /^(NEW|FIX|UPD)\s*[:\-]\s*(.*)$/i.exec(l);
      return m ? { tag: m[1].toUpperCase(), text: m[2] } : { tag: 'UPD', text: l.replace(/^[-*•]\s*/, '') };
    });
  }

  function render() {
    if (!info) return;
    const e = build();
    e.card.dataset.stage = stage;
    e.x.title = t('close');
    e.icon.innerHTML = stage === 'downloaded' ? ICON_OK : stage === 'failed' ? ICON_ERR : ICON_DOWN;
    e.title.textContent = t(stage === 'downloaded' ? 'readyTitle' : stage === 'downloading' ? 'downloadingTitle' : stage === 'failed' ? 'failedTitle' : 'newTitle');
    const first = stage === 'available' || stage === 'failed';
    e.old.textContent = first ? (info.current || '') : '';
    e.arrow.classList.toggle('uh-hidden', !(first && info.current));
    e.chip.textContent = info.label || `v${info.version}`;

    // available
    const notes = parseNotes(info.releaseNotes);
    e.notes.classList.toggle('uh-hidden', !notes.length);
    e.notes.classList.toggle('uh-wide', notesOpen);
    e.notesTitle.textContent = t('thisRelease');
    e.all.textContent = t(notesOpen ? 'collapse' : 'showAll');
    e.list.innerHTML = '';
    for (const n of notes) {
      const row = document.createElement('div');
      row.className = 'uh-note';
      const tag = document.createElement('span');
      tag.className = `uh-tag ${n.tag}`;
      tag.textContent = n.tag;
      const txt = document.createElement('span');
      txt.textContent = n.text;
      txt.title = n.text;
      row.append(tag, txt);
      e.list.appendChild(row);
    }
    e.meta.textContent = !info.portable && info.size ? t('sizeLine', { size: mb(info.size) }) : '';
    e.hint.textContent = info.portable ? t('portableHint') : '';
    e.later.textContent = t('later');
    e.go.textContent = t(info.portable ? 'openDownload' : 'now');
    e.skip.textContent = t('skip');

    // downloading
    e.autoLabel.textContent = t('autoRestart');
    e.cancel.textContent = t('cancel');
    e.cancel.classList.toggle('uh-hidden', !canCancel);
    e.bg.textContent = t('background');

    // downloaded / failed
    e.ready.textContent = t('readyBody');
    e.onquit.textContent = t('laterOnQuit');
    e.restart.textContent = t('restart');
    e.close.textContent = t('close');
    e.retry.textContent = t(info.portable ? 'openDownload' : 'retry');
  }

  function show() {
    build();
    els.overlay.classList.add('uh-open');
  }

  function open(newInfo) {
    // A new check result replaces the info, but never resets a download that is running or finished
    if (stage !== 'downloading' && stage !== 'downloaded') {
      info = newInfo;
      stage = 'available';
      notesOpen = false;
    }
    render();
    show();
  }

  function close() {
    if (els) els.overlay.classList.remove('uh-open');
  }

  async function onPrimary() {
    if (!info) return;
    if (stage === 'downloaded') { bridge.install(); return; }
    if (stage === 'downloading') return;
    if (!info.portable) {
      stage = 'downloading';
      els.bar.style.width = '0%';
      els.pct.textContent = '0';
      els.size.textContent = info.size ? `0.0 / ${mb(info.size)} MB` : '';
      els.speed.textContent = t('connecting');
      render();
    }
    let r;
    try { r = await bridge.download(); } catch (err) { r = { ok: false, message: err.message }; }
    if (r && r.opened) {          // portable - the browser takes over the download
      stage = 'available';
      close();
      return;
    }
    if (r && r.ok) {
      stage = 'downloaded';
      if (els.auto.checked) { bridge.install(); return; }
      render();
      show();                     // back on screen if it was sent to the background
    } else if (r && r.cancelled) {
      stage = 'available';
      render();
    } else {
      stage = 'failed';
      render();
      els.error.textContent = t('failed', { message: (r && r.message) || '' });
      show();
    }
  }

  bridge.onProgress((p) => {
    if (!els || stage !== 'downloading') return;
    const pct = Math.max(0, Math.min(100, p.percent || 0));
    els.bar.style.width = pct.toFixed(1) + '%';
    els.pct.textContent = pct.toFixed(0);
    // a differential download (blockmap) fetches the changed blocks only: total is then smaller than the installer (p.full)
    const diff = p.total && p.full && p.total < p.full * 0.95 ? ` · ${t('diffOnly', { full: mb(p.full) })}` : '';
    els.size.textContent = p.total ? `${mb(p.transferred || 0)} / ${mb(p.total)} MB${diff}` : '';
    const left = p.speed > 0 && p.total ? Math.max(0, (p.total - (p.transferred || 0)) / p.speed) : null;
    const eta = left === null ? '' : left >= 60 ? t('minLeft', { n: Math.ceil(left / 60) }) : t('secLeft', { n: Math.max(1, Math.round(left)) });
    els.speed.textContent = p.speed > 0 ? `${mb(p.speed)} MB/s${eta ? ' · ' + eta : ''}` : t('connecting');
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
