/**
 * Custom replacements for browser alert / confirm / prompt.
 * Usage: await dlg.alert(msg)              → undefined
 *        await dlg.confirm(msg)            → true | false
 *        await dlg.prompt(msg, default)    → string | null
 */
const dlg = (() => {
  let overlay, iconBadge, titleEl, msgEl, inputWrap, inputLabel, inputEl, cancelBtn, okBtn;
  let _resolve = null;

  const TYPES = {
    alert:   { icon: 'ℹ',  iconClass: 'info',   title: 'お知らせ',         ok: 'OK',      okClass: 'btn-primary' },
    confirm: { icon: '⚠',  iconClass: 'danger', title: '削除の確認',       ok: '削除する', okClass: 'btn-danger'  },
    prompt:  { icon: '✏',  iconClass: 'prompt', title: '名前の変更',       ok: '変更する', okClass: 'btn-primary' },
    run:     { icon: '▶',  iconClass: 'info',   title: '実行確認',         ok: '実行する', okClass: 'btn-primary' },
    trigger: { icon: '⚡', iconClass: 'info',   title: 'ワークフロー確認', ok: 'トリガー', okClass: 'btn-primary' },
  };

  function init() {
    overlay = document.createElement('div');
    overlay.className = 'dlg-overlay';
    overlay.innerHTML = `
      <div class="dlg-box" role="dialog" aria-modal="true">
        <div class="dlg-header">
          <span class="dlg-icon-badge"></span>
          <span class="dlg-title"></span>
        </div>
        <div class="dlg-body">
          <p class="dlg-msg"></p>
          <div class="dlg-input-wrap">
            <label class="dlg-input-label">新しい名前</label>
            <input class="dlg-input" type="text" autocomplete="off">
          </div>
          <div class="dlg-actions">
            <button class="btn btn-sm btn-ghost dlg-cancel">キャンセル</button>
            <button class="btn btn-sm dlg-ok">OK</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    iconBadge  = overlay.querySelector('.dlg-icon-badge');
    titleEl    = overlay.querySelector('.dlg-title');
    msgEl      = overlay.querySelector('.dlg-msg');
    inputWrap  = overlay.querySelector('.dlg-input-wrap');
    inputLabel = overlay.querySelector('.dlg-input-label');
    inputEl    = overlay.querySelector('.dlg-input');
    cancelBtn  = overlay.querySelector('.dlg-cancel');
    okBtn      = overlay.querySelector('.dlg-ok');

    okBtn.addEventListener('click',     () => settle(true));
    cancelBtn.addEventListener('click', () => settle(false));
    overlay.addEventListener('click', e => { if (e.target === overlay) settle(false); });
    document.addEventListener('keydown', e => {
      if (!overlay.classList.contains('open')) return;
      if (e.key === 'Escape') { e.preventDefault(); settle(false); }
      if (e.key === 'Enter' && document.activeElement !== inputEl) { e.preventDefault(); settle(true); }
    });
  }

  function settle(confirmed) {
    if (!overlay.classList.contains('open') || !_resolve) return;
    overlay.classList.remove('open');
    const cb = _resolve;
    _resolve = null;
    cb(confirmed);
  }

  function open(type, message, defaultValue) {
    return new Promise(resolve => {
      const cfg = TYPES[type];
      msgEl.textContent = message;

      iconBadge.textContent = cfg.icon;
      iconBadge.className   = `dlg-icon-badge ${cfg.iconClass}`;
      titleEl.textContent   = cfg.title;

      const isPrompt  = type === 'prompt';
      const isAlert   = type === 'alert';

      inputWrap.style.display  = isPrompt ? '' : 'none';
      cancelBtn.style.display  = isAlert  ? 'none' : '';
      okBtn.className = `btn btn-sm dlg-ok ${cfg.okClass}`;
      okBtn.textContent = cfg.ok;

      if (isPrompt) {
        inputEl.value = defaultValue ?? '';
        _resolve = confirmed => resolve(confirmed ? inputEl.value : null);
        setTimeout(() => { inputEl.focus(); inputEl.select(); }, 40);
      } else {
        _resolve = confirmed => resolve(isAlert ? undefined : confirmed);
        setTimeout(() => okBtn.focus(), 40);
      }

      overlay.classList.add('open');
    });
  }

  document.addEventListener('DOMContentLoaded', init);

  return {
    alert:   (msg)               => open('alert',   msg),
    confirm: (msg)               => open('confirm', msg),
    prompt:  (msg, defaultValue) => open('prompt',  msg, defaultValue),
    run:     (msg)               => open('run',     msg),
    trigger: (msg)               => open('trigger', msg),
  };
})();
