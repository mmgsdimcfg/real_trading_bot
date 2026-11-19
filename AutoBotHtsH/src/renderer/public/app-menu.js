/*
 * Update log
 * - [2026-10-23] type=feat - New self-contained app menu + About dialog (shared by the make_utils Electron programs, same file in each -
 *   pairs with app-about.js in main; modelled on WiFiScannerH). Builds its own DOM/CSS (prefixed .amx-), so the page only needs
 *   <script src="app-menu.js"></script> after update-ui.js and one call:
 *     AppMenu.init({
 *       button: '#appMenuBtn',                  // the ☰ button the app placed in its title bar (required)
 *       productName: 'TermH',
 *       icon: 'assets/icon.png',                // packaged path (build/ is NOT packaged by electron-builder)
 *       tagline: { en: '...', ko: '...' },
 *       description: { en: '...', ko: '...' },
 *       onSettings: () => openSettings(),       // optional - adds the "Settings" item (the old ⚙ button moves here)
 *       items: [{ id, label: { en, ko }, icon: '…', onClick }],   // optional extra items above Check for updates
 *       onCheckUpdates: async () => result,     // optional - when the app has no update-ui.js (else UpdateUI is used)
 *       lang: () => 'en' | 'ko',                // optional - default: <html lang>
 *     })
 *   Menu: [Settings] [extra items] / Check for updates / About <name>. ESC, outside click and arrow keys work.
 *   About: icon, name, tagline, version (+ check for updates), description, website (www.iccid.dev), "Contact the developer"
 *   (mailto via main process - the address is never in the page) + "Copy address", license, third-party notice,
 *   "Copyright © <year> xgraph. All rights reserved.". Light/dark follows the page background.
 *   IPC through window.appAbout (preload, contextIsolation apps) or require('electron').ipcRenderer (nodeIntegration apps):
 *   'app-about:info' / 'app-about:open-website' / 'app-about:contact' / 'app-about:copy-email'.
 */
(function () {
  'use strict';

  const bridge = (() => {
    if (window.appAbout) return window.appAbout;
    try {
      const { ipcRenderer } = require('electron');
      return {
        info: () => ipcRenderer.invoke('app-about:info'),
        openWebsite: () => ipcRenderer.invoke('app-about:open-website'),
        contact: (msg) => ipcRenderer.invoke('app-about:contact', msg),
        copyEmail: () => ipcRenderer.invoke('app-about:copy-email'),
      };
    } catch (e) {
      return null;
    }
  })();

  const STR = {
    en: {
      menu: 'Menu', settings: 'Settings', checkUpdate: 'Check for updates', about: 'About {name}',
      website: 'Website', contact: 'Contact', contactBtn: 'Contact the developer', copyEmail: 'Copy address',
      contactHint: 'Opens your mail app with a pre-filled message that includes the version and system information.',
      license: 'License', licenseText: 'Proprietary software. Copying, modification or distribution without permission is prohibited.',
      thirdParty: 'Third-party components remain under their own licenses.',
      copyright: 'Copyright © {year} xgraph. All rights reserved.', close: 'Close',
      emailCopied: 'Email address copied to the clipboard.',
      checking: 'Checking for updates...', upToDate: 'You are using the latest version ({version}).',
      devMode: 'Update check is not available in development mode.', checkFailed: 'Update check failed: {message}',
      mailSubject: '[{name} {version}] Feedback',
      mailBody: 'Hello,\n\n(Write your message here)\n\n---\nApp: {name} {version}\nOS: {os}\nRuntime: Electron {electron}\n',
    },
    ko: {
      menu: '메뉴', settings: '설정', checkUpdate: '업데이트 확인', about: '{name} 정보',
      website: '웹사이트', contact: '문의', contactBtn: '개발자에게 문의하기', copyEmail: '주소 복사',
      contactHint: '버전과 시스템 정보가 미리 채워진 메일을 메일 앱에서 엽니다.',
      license: '라이선스', licenseText: '독점 소프트웨어. 허가 없이 복제·수정·배포할 수 없습니다.',
      thirdParty: '서드파티 구성 요소는 각자의 라이선스를 따릅니다.',
      copyright: 'Copyright © {year} xgraph. All rights reserved.', close: '닫기',
      emailCopied: '이메일 주소를 클립보드에 복사했습니다.',
      checking: '업데이트를 확인하는 중...', upToDate: '최신 버전({version})을 사용 중입니다.',
      devMode: '개발 모드에서는 업데이트를 확인하지 않습니다.', checkFailed: '업데이트 확인 실패: {message}',
      mailSubject: '[{name} {version}] 문의',
      mailBody: '안녕하세요,\n\n(문의 내용을 적어 주세요)\n\n---\n앱: {name} {version}\nOS: {os}\n런타임: Electron {electron}\n',
    },
  };

  const CSS = `
.amx-menu { position:fixed; min-width:220px; background:var(--amx-bg); color:var(--amx-text); border:1px solid var(--amx-border); border-radius:8px;
  box-shadow:0 10px 30px rgba(0,0,0,.28); padding:5px; z-index:2147482000; font:13px 'Segoe UI','Malgun Gothic',system-ui,sans-serif; -webkit-app-region:no-drag; }
.amx-hidden { display:none !important; }
.amx-item { display:flex; align-items:center; gap:9px; width:100%; margin:0; border:none; background:none; padding:7px 10px; border-radius:5px; text-align:left;
  color:inherit; font:inherit; cursor:pointer; }
.amx-item:hover, .amx-item:focus-visible { background:var(--amx-hover); outline:none; }
.amx-ic { width:16px; text-align:center; color:var(--amx-muted); flex-shrink:0; }
.amx-sep { height:1px; background:var(--amx-border); margin:4px 2px; }
.amx-light { --amx-bg:#fff; --amx-bg2:#f1f5f9; --amx-text:#1e293b; --amx-muted:#64748b; --amx-border:#e2e8f0; --amx-hover:#e0ecff; --amx-accent:#2563eb; }
.amx-dark { --amx-bg:#262a31; --amx-bg2:#1d2026; --amx-text:#e6e8eb; --amx-muted:#9aa1ab; --amx-border:#3a3f48; --amx-hover:#33404f; --amx-accent:#60a5fa; }
.amx-toast { position:fixed; top:52px; right:16px; max-width:420px; background:#1e293b; color:#fff; padding:9px 14px; border-radius:7px;
  box-shadow:0 8px 24px rgba(0,0,0,.3); font:12px 'Segoe UI','Malgun Gothic',system-ui,sans-serif; z-index:2147482001; }
.amx-toast.amx-err { background:#7f1d1d; }
.amx-overlay { position:fixed; inset:0; background:rgba(0,0,0,.5); display:flex; align-items:center; justify-content:center; z-index:2147482500;
  font:13px 'Segoe UI','Malgun Gothic',system-ui,sans-serif; -webkit-app-region:no-drag; }
.amx-about { position:relative; width:480px; max-width:calc(100vw - 32px); max-height:calc(100vh - 32px); overflow:auto; background:var(--amx-bg); color:var(--amx-text);
  border-radius:12px; box-shadow:0 20px 50px rgba(0,0,0,.45); }
.amx-x { position:absolute; top:8px; right:8px; width:32px; height:32px; margin:0; border:none; border-radius:6px; background:none; color:rgba(255,255,255,.8);
  font-size:22px; line-height:1; cursor:pointer; }
.amx-x:hover { background:rgba(255,255,255,.15); color:#fff; }
.amx-hero { display:flex; align-items:center; gap:16px; padding:22px 22px 18px; background:linear-gradient(135deg,#111a33,#1e2a55 60%,#3b2f7a); color:#fff; }
.amx-icon { width:72px; height:72px; flex-shrink:0; border-radius:16px; filter:drop-shadow(0 4px 10px rgba(0,0,0,.35)); }
.amx-name { font-size:20px; font-weight:700; letter-spacing:.01em; }
.amx-tag { font-size:12px; color:rgba(255,255,255,.75); margin-top:2px; }
.amx-ver { margin-top:8px; font:12px Consolas,'Cascadia Mono',monospace; color:#c7d2fe; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.amx-link { margin:0; padding:0; border:none; background:none; font:12px 'Segoe UI','Malgun Gothic',system-ui,sans-serif; color:#93c5fd; cursor:pointer; }
.amx-link:hover { text-decoration:underline; }
.amx-link:disabled { opacity:.7; text-decoration:none; cursor:default; }
.amx-body { padding:16px 22px 6px; }
.amx-desc { color:var(--amx-muted); line-height:1.55; margin:0 0 12px; }
.amx-kv { display:grid; grid-template-columns:92px 1fr; gap:8px 12px; align-items:start; margin:0; }
.amx-kv dt { color:var(--amx-muted); padding-top:3px; }
.amx-kv dd { margin:0; min-width:0; }
.amx-kv a { color:var(--amx-accent); text-decoration:none; }
.amx-kv a:hover { text-decoration:underline; }
.amx-row { display:flex; gap:6px; flex-wrap:wrap; }
.amx-btn { margin:0; border:1px solid var(--amx-border); background:var(--amx-bg); color:var(--amx-text); padding:5px 12px; border-radius:6px; font:inherit; cursor:pointer; }
.amx-btn:hover { border-color:var(--amx-accent); color:var(--amx-accent); }
.amx-btn.amx-primary { background:var(--amx-accent); border-color:var(--amx-accent); color:#fff; }
.amx-btn.amx-primary:hover { filter:brightness(1.08); color:#fff; }
.amx-hint { font-size:11px; color:var(--amx-muted); margin-top:4px; }
.amx-third { font-size:11px; color:var(--amx-muted); line-height:1.5; margin:12px 0 4px; }
.amx-foot { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 22px 16px; color:var(--amx-muted); font-size:11px;
  border-top:1px solid var(--amx-border); margin-top:8px; }
`;

  let opts = null;
  let menu = null;
  let toastEl = null;
  let toastTimer = null;
  let overlay = null;
  let info = null;

  const lang = () => {
    const l = opts && opts.lang ? opts.lang() : (document.documentElement.lang || 'en');
    return /^ko/i.test(l) ? 'ko' : 'en';
  };
  const t = (key, vars) => String(STR[lang()][key] ?? STR.en[key] ?? key).replace(/\{(\w+)\}/g, (_m, k) => (vars && vars[k] != null ? vars[k] : ''));
  const pick = (v) => (v && typeof v === 'object' ? (v[lang()] || v.en || '') : (v || ''));
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // Light/dark from the page background (same rule as update-ui.js)
  function themeClass() {
    const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?/.exec(getComputedStyle(document.body).backgroundColor || '');
    let dark;
    if (!m || m[4] === '0') dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    else dark = (0.299 * m[1] + 0.587 * m[2] + 0.114 * m[3]) < 128;
    return dark ? 'amx-dark' : 'amx-light';
  }

  function button() { return typeof opts.button === 'string' ? document.querySelector(opts.button) : opts.button; }

  async function getInfo() {
    if (info) return info;
    try { info = (bridge && await bridge.info()) || {}; } catch (e) { info = {}; }
    return info;
  }

  /* ─── menu ─── */
  function renderMenu() {
    const items = [];
    if (opts.onSettings) items.push({ id: 'settings', icon: '⚙', label: t('settings'), run: opts.onSettings });
    (opts.items || []).forEach((it) => items.push({ id: it.id, icon: it.icon || '', label: pick(it.label), run: it.onClick }));
    const html = [];
    items.forEach((it) => html.push(`<button type="button" class="amx-item" role="menuitem" data-id="${esc(it.id)}"><span class="amx-ic">${esc(it.icon)}</span><span>${esc(it.label)}</span></button>`));
    if (items.length) html.push('<div class="amx-sep"></div>');
    html.push(`<button type="button" class="amx-item" role="menuitem" data-id="update"><span class="amx-ic">⟳</span><span>${esc(t('checkUpdate'))}</span></button>`);
    html.push('<div class="amx-sep"></div>');
    html.push(`<button type="button" class="amx-item" role="menuitem" data-id="about"><span class="amx-ic">ⓘ</span><span>${esc(t('about', { name: opts.productName }))}</span></button>`);
    menu.innerHTML = html.join('');
    menu.className = `amx-menu ${themeClass()}${menu.classList.contains('amx-hidden') ? ' amx-hidden' : ''}`;
    menu._items = items;
  }

  function openMenu() {
    renderMenu();
    const b = button();
    const r = b.getBoundingClientRect();
    menu.classList.remove('amx-hidden');
    menu.style.top = `${Math.round(r.bottom + 4)}px`;
    menu.style.right = `${Math.max(8, Math.round(window.innerWidth - r.right))}px`;
    b.setAttribute('aria-expanded', 'true');
    const first = menu.querySelector('.amx-item');
    if (first) first.focus();
  }
  function closeMenu() {
    if (!menu || menu.classList.contains('amx-hidden')) return;
    menu.classList.add('amx-hidden');
    const b = button();
    if (b) b.setAttribute('aria-expanded', 'false');
  }

  function toast(text, { error = false, ms = 4000 } = {}) {
    toastEl.textContent = text;
    toastEl.classList.toggle('amx-err', error);
    toastEl.classList.remove('amx-hidden');
    clearTimeout(toastTimer);
    if (ms) toastTimer = setTimeout(() => toastEl.classList.add('amx-hidden'), ms);
  }

  /* ─── update check: update-ui.js (UpdateUI) or the app's own hook ─── */
  async function checkUpdates() {
    if (opts.onCheckUpdates) { await opts.onCheckUpdates(); return; }
    if (!window.UpdateUI) { toast(t('checkFailed', { message: 'update-ui.js' }), { error: true }); return; }
    toast(t('checking'), { ms: 0 });
    let r;
    try { r = await window.UpdateUI.check(); } catch (e) { r = { status: 'error', message: e.message }; }
    if (r.status === 'available') { toastEl.classList.add('amx-hidden'); window.UpdateUI.open(r.info); return; }
    const i = await getInfo();
    if (r.status === 'none') toast(t('upToDate', { version: r.current || i.version || '' }));
    else if (r.status === 'dev') toast(t('devMode'));
    else toast(t('checkFailed', { message: r.message || '' }), { error: true, ms: 7000 });
  }

  /* ─── About ─── */
  async function openAbout() {
    const i = await getInfo();
    closeAbout();
    overlay = document.createElement('div');
    overlay.className = `amx-overlay ${themeClass()}`;
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.innerHTML = `
      <div class="amx-about">
        <button type="button" class="amx-x" data-act="close" title="${esc(t('close'))}">×</button>
        <div class="amx-hero">
          ${opts.icon ? `<img class="amx-icon" src="${esc(opts.icon)}" alt="" draggable="false">` : ''}
          <div>
            <div class="amx-name">${esc(opts.productName)}</div>
            <div class="amx-tag">${esc(pick(opts.tagline))}</div>
            <div class="amx-ver"><span>${esc(i.version || '')}</span><button type="button" class="amx-link" data-act="update">${esc(t('checkUpdate'))}</button></div>
          </div>
        </div>
        <div class="amx-body">
          ${pick(opts.description) ? `<p class="amx-desc">${esc(pick(opts.description))}</p>` : ''}
          <dl class="amx-kv">
            <dt>${esc(t('website'))}</dt><dd><a href="#" data-act="website">${esc(i.websiteLabel || 'www.iccid.dev')}</a></dd>
            <dt>${esc(t('contact'))}</dt>
            <dd>
              <div class="amx-row">
                <button type="button" class="amx-btn amx-primary" data-act="contact">✉ ${esc(t('contactBtn'))}</button>
                <button type="button" class="amx-btn" data-act="copy">${esc(t('copyEmail'))}</button>
              </div>
              <div class="amx-hint">${esc(t('contactHint'))}</div>
            </dd>
            <dt>${esc(t('license'))}</dt><dd>${esc(t('licenseText'))}</dd>
          </dl>
          <p class="amx-third">${esc(t('thirdParty'))}</p>
        </div>
        <div class="amx-foot"><span>${esc(t('copyright', { year: i.year || new Date().getFullYear() }))}</span><button type="button" class="amx-btn" data-act="close">${esc(t('close'))}</button></div>
      </div>`;
    overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) closeAbout(); });
    overlay.addEventListener('click', async (e) => {
      const el = e.target.closest('[data-act]');
      if (!el) return;
      const act = el.dataset.act;
      if (act === 'close') closeAbout();
      else if (act === 'website') { e.preventDefault(); if (bridge) bridge.openWebsite(); }
      else if (act === 'update') checkUpdates();
      else if (act === 'contact') {
        if (bridge) bridge.contact({
          subject: t('mailSubject', { name: opts.productName, version: i.version || '' }),
          body: t('mailBody', { name: opts.productName, version: i.version || '', os: i.os || '', electron: i.electron || '' }),
        });
      } else if (act === 'copy') {
        try { if (bridge) await bridge.copyEmail(); toast(t('emailCopied')); } catch (err) { toast(err.message, { error: true }); }
      }
    });
    document.body.appendChild(overlay);
    const closeBtn = overlay.querySelector('.amx-foot [data-act="close"]');
    if (closeBtn) closeBtn.focus();
  }
  function closeAbout() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  function init(options) {
    opts = options || {};
    if (!opts.productName) opts.productName = document.title || 'App';
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    menu = document.createElement('div');
    menu.className = 'amx-menu amx-hidden';
    menu.setAttribute('role', 'menu');
    document.body.appendChild(menu);
    toastEl = document.createElement('div');
    toastEl.className = 'amx-toast amx-hidden';
    toastEl.setAttribute('role', 'status');
    document.body.appendChild(toastEl);

    const b = button();
    if (!b) { console.warn('[app-menu] menu button not found:', opts.button); return; }
    b.setAttribute('aria-haspopup', 'menu');
    b.setAttribute('aria-expanded', 'false');
    if (!b.title) b.title = t('menu');
    b.addEventListener('click', (e) => { e.stopPropagation(); if (menu.classList.contains('amx-hidden')) openMenu(); else closeMenu(); });
    menu.addEventListener('click', (e) => {
      const it = e.target.closest('.amx-item');
      if (!it) return;
      closeMenu();
      const id = it.dataset.id;
      if (id === 'update') checkUpdates();
      else if (id === 'about') openAbout();
      else { const def = (menu._items || []).find((x) => x.id === id); if (def && def.run) def.run(); }
    });
    menu.addEventListener('keydown', (e) => {
      const items = Array.from(menu.querySelectorAll('.amx-item'));
      const i = items.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); items[(i + 1) % items.length].focus(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
    });
    document.addEventListener('mousedown', (e) => { if (!e.target.closest('.amx-menu') && e.target !== b && !b.contains(e.target)) closeMenu(); }, true);
    window.addEventListener('blur', closeMenu);
    window.addEventListener('resize', closeMenu);
    // ESC: menu first, then About (capture phase, before the app's own ESC handling)
    window.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      if (!menu.classList.contains('amx-hidden')) { e.preventDefault(); e.stopPropagation(); closeMenu(); b.focus(); }
      else if (overlay) { e.preventDefault(); e.stopPropagation(); closeAbout(); }
    }, true);
  }

  window.AppMenu = { init, openAbout, closeAbout, openMenu, closeMenu, checkUpdates, toast };
})();
