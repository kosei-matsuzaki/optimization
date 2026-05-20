const _pd = JSON.parse(document.getElementById('page-data').textContent);
const ALL_RESULTS  = _pd.results;
let   RESULTS_META = _pd.meta || {};
let   runningSet   = new Set(_pd.running || []);

const STATUS_LABEL = { done: '完了', running: '実行中', stopped: '中断', failed: '失敗' };

// ── Table sort ──
let _sortCol = 'date', _sortDir = -1;
const STATUS_ORDER = { running: 0, done: 1, failed: 2, stopped: 3, '': 4 };

function _sortKey(name, col) {
  const m = RESULTS_META[name] || {};
  switch (col) {
    case 'name':   return name;
    case 'type':   return m.type || '';
    case 'date':   return m.created_at || name;
    case 'params': return (m.max_evals || 0) * 1000 + (m.n_runs || 0);
    case 'commit': return m.commit || '';
    case 'status': {
      const s = runningSet.has(name) ? 'running' : (m.status || (m.type ? 'done' : ''));
      return STATUS_ORDER[s] ?? 4;
    }
  }
  return '';
}

function sortResults(col) {
  if (_sortCol === col) _sortDir *= -1;
  else { _sortCol = col; _sortDir = 1; }

  const tbody = document.getElementById('results-tbody');
  if (!tbody) return;

  [...tbody.querySelectorAll('.rt-row[data-name]')]
    .sort((a, b) => {
      const va = _sortKey(a.dataset.name, col);
      const vb = _sortKey(b.dataset.name, col);
      if (typeof va === 'number') return (va - vb) * _sortDir;
      return String(va).localeCompare(String(vb)) * _sortDir;
    })
    .forEach(r => tbody.appendChild(r));

  document.querySelectorAll('.results-table thead th[data-sort]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.sort === col) th.classList.add(_sortDir === 1 ? 'sort-asc' : 'sort-desc');
  });
}

function getCardInfo(name) {
  const m = RESULTS_META[name] || {};
  let date = '', time = '', commit = m.commit || '', type = m.type || '';
  const rx = name.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_([^_]+)/);
  if (rx) {
    date = `${rx[1]}-${rx[2]}-${rx[3]}`;
    time = `${rx[4]}:${rx[5]}`;
    if (!commit) commit = rx[7];
    if (!type) type = name.includes('quick') ? 'quick' : 'workflow';
  }
  if (m.created_at) {
    const [d, t] = m.created_at.split('T');
    date = d || date;
    time = t ? t.slice(0, 5) : time;
  }
  const title = name;
  const sub   = rx ? time : `${date} ${time}`.trim();
  // Status: running overrides saved status; result.json present without status → done
  const status = runningSet.has(name) ? 'running' : (m.status || (m.type ? 'done' : ''));
  // n_runs / max_evals / set summary
  // For workflow runs and legacy quick runs the `set` field is absent; only
  // show it when explicitly provided (currently only quick runs since the
  // --all flag was introduced).
  const setStr = m.set === "all-26" ? "BBOB-26" : m.set === "quick-12" ? "quick-12" : null;
  const evalsStr = [
    m.n_runs    != null ? `${m.n_runs} runs` : null,
    m.max_evals != null ? `${Number(m.max_evals).toLocaleString()} evals` : null,
    setStr,
  ].filter(Boolean).join(' · ');
  const metaStr = [sub, evalsStr].filter(Boolean).join('  ·  ');
  return { title, sub, date, time, commit, type, status, evalsStr, metaStr };
}

function applyCardInfo(row, name) {
  const { title, date, time, evalsStr, commit, type, status } = getCardInfo(name);
  const linkEl = row.querySelector('.rt-link');
  if (linkEl) linkEl.textContent = title;
  const typeChip = row.querySelector('.rt-type-chip');
  if (typeChip) { typeChip.textContent = type || ''; typeChip.className = `rt-type-chip ${type || 'unknown'}`; }
  const dateEl = row.querySelector('.rt-date');
  if (dateEl) dateEl.textContent = date;
  const timeEl = row.querySelector('.rt-time');
  if (timeEl) timeEl.textContent = time;
  const paramsEl = row.querySelector('.rt-params');
  if (paramsEl) paramsEl.textContent = evalsStr;
  const commitEl = row.querySelector('.rt-commit');
  if (commitEl) commitEl.textContent = commit;
  const statusDiv = row.querySelector('.rc-status');
  const dot       = row.querySelector('.rc-dot');
  const label     = row.querySelector('.rc-label');
  if (status && statusDiv && dot && label) {
    statusDiv.className = `rc-status visible ${status}`;
    dot.className = `rc-dot ${status}`;
    label.textContent = STATUS_LABEL[status] || status;
  }
  if (type === 'quick') row.classList.add('is-quick');
  else if (type === 'workflow') row.classList.add('is-workflow');
  const stopBtn = row.querySelector('.stop-job');
  if (stopBtn) stopBtn.style.display = status === 'running' ? '' : 'none';
}

document.querySelectorAll('.rt-row[data-name]').forEach(row => {
  applyCardInfo(row, row.dataset.name);
});

// Apply initial sort indicator (date ↓)
(function() {
  const th = document.querySelector(`.results-table thead th[data-sort="${_sortCol}"]`);
  if (th) th.classList.add(_sortDir === 1 ? 'sort-asc' : 'sort-desc');
})();

document.getElementById('quick-count').textContent =
  ALL_RESULTS.filter(r => (RESULTS_META[r]?.type || '') === 'quick').length;

if (ALL_RESULTS.length) {
  const { date, time, commit } = getCardInfo(ALL_RESULTS[0]);
  document.getElementById('latest-date').textContent = date ? `${date} ${time}`.trim() : ALL_RESULTS[0];
  document.getElementById('latest-commit').textContent = commit ? `@ ${commit}` : '';
}

// ── HTML escape ──
function _esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Create a result card DOM element ──
function createResultCard(name) {
  const { title, date, time, evalsStr, commit, type, status } = getCardInfo(name);
  const typeClass    = type === 'quick' ? ' is-quick' : type === 'workflow' ? ' is-workflow' : '';
  const typeChipCls  = type || 'unknown';
  const statusHtml   = status
    ? `<div class="rc-status visible ${status}"><span class="rc-dot ${status}"></span><span class="rc-label">${_esc(STATUS_LABEL[status] || status)}</span></div>`
    : `<div class="rc-status"><span class="rc-dot"></span><span class="rc-label"></span></div>`;
  const row = document.createElement('tr');
  row.className = `rt-row${typeClass}`;
  row.id = `card-${name}`;
  row.dataset.name = name;
  row.innerHTML = `
    <td><a href="/results/${encodeURIComponent(name)}" class="rt-link">${_esc(title)}</a></td>
    <td><span class="rt-type-chip ${typeChipCls}">${_esc(type || '')}</span></td>
    <td><span class="rt-date">${_esc(date)}</span><span class="rt-time">${_esc(time)}</span></td>
    <td class="rt-params">${_esc(evalsStr)}</td>
    <td class="rt-commit">${_esc(commit)}</td>
    <td>${statusHtml}</td>
    <td class="rt-actions-cell">
      <button class="card-menu-btn" onclick="toggleCardMenu(${JSON.stringify(name)},event)" title="操作">⋮</button>
      <div class="card-menu-dropdown" id="menu-${_esc(name)}">
        <button class="stop-job" style="display:${status === 'running' ? '' : 'none'};" onclick="stopCurrentJob()">■ 実行停止</button>
        <button onclick="renameResult(${JSON.stringify(name)},event)">✏ 名前変更</button>
        <button class="danger" onclick="deleteResult(${JSON.stringify(name)},event)">🗑 削除</button>
      </div>
    </td>`;
  return row;
}

// ── Auto-refresh results panel ──
const _knownResults = new Set(ALL_RESULTS);

async function refreshResults() {
  try {
    const res = await fetch('/api/results');
    if (!res.ok) return;
    const { results, meta, running } = await res.json();

    // Always refresh running set (status may change without new results)
    runningSet = new Set(running || []);

    // Refresh status dots and stop buttons on existing rows
    document.querySelectorAll('.rt-row[data-name]').forEach(row => {
      const name = row.dataset.name;
      const savedMeta = meta[name] || {};
      const status = runningSet.has(name) ? 'running' : (savedMeta.status || (savedMeta.type ? 'done' : ''));
      const statusDiv = row.querySelector('.rc-status');
      const dot       = row.querySelector('.rc-dot');
      const label     = row.querySelector('.rc-label');
      if (status && statusDiv && dot && label) {
        statusDiv.className = `rc-status visible ${status}`;
        dot.className = `rc-dot ${status}`;
        label.textContent = STATUS_LABEL[status] || status;
      }
      const stopBtn = row.querySelector('.stop-job');
      if (stopBtn) stopBtn.style.display = status === 'running' ? '' : 'none';
    });

    const newResults = results.filter(r => !_knownResults.has(r));
    if (!newResults.length) return;

    // Update tracking state
    newResults.forEach(r => _knownResults.add(r));
    Object.assign(RESULTS_META, meta);

    // Stat cards
    const totalEl = document.querySelector('.stat-card.blue .stat-value');
    if (totalEl) totalEl.textContent = results.length;
    document.getElementById('quick-count').textContent =
      results.filter(r => (meta[r]?.type || '') === 'quick').length;
    if (results.length) {
      const { date, time, commit } = getCardInfo(results[0]);
      const ldEl = document.getElementById('latest-date');
      const lcEl = document.getElementById('latest-commit');
      if (ldEl) ldEl.textContent = date ? `${date} ${time}`.trim() : results[0];
      if (lcEl) lcEl.textContent = commit ? `@ ${commit}` : '';
    }

    // Get or create results-tbody inside results-panel
    const panel = document.getElementById('results-panel');
    let tbody = document.getElementById('results-tbody');
    if (!tbody) {
      const placeholder = panel.querySelector('[style*="text-align"]');
      if (placeholder) placeholder.remove();
      const wrap = document.createElement('div');
      wrap.className = 'results-table-wrap';
      wrap.innerHTML = `<table class="results-table">
        <thead><tr>
          <th class="sortable" data-sort="name"   onclick="sortResults('name')">Name</th>
          <th class="sortable" data-sort="type"   onclick="sortResults('type')">Type</th>
          <th class="sortable" data-sort="date"   onclick="sortResults('date')">Date</th>
          <th class="sortable" data-sort="params" onclick="sortResults('params')">Params</th>
          <th class="sortable" data-sort="commit" onclick="sortResults('commit')">Commit</th>
          <th class="sortable" data-sort="status" onclick="sortResults('status')">Status</th>
          <th></th>
        </tr></thead>
        <tbody id="results-tbody"></tbody>
      </table>`;
      panel.appendChild(wrap);
      tbody = document.getElementById('results-tbody');
    }

    // Prepend new rows (newest first)
    newResults.forEach(r => {
      if (!document.getElementById(`card-${r}`)) {
        const row = createResultCard(r);
        row.classList.add('card-new');
        tbody.prepend(row);
      }
    });

    // Update count tag
    const countTag = document.getElementById('results-count-tag');
    if (countTag) countTag.textContent = results.length;

  } catch (_) {}
}

// Poll every 15 seconds
setInterval(refreshResults, 15000);

// Load remote workflow runs once on page load (no periodic refresh)
loadGhRuns();

// ── Toast ──
function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), 3500);
}

// ── Quick Run ──
const terminal = document.getElementById('terminal');
const jobBadge = document.getElementById('job-badge');
const jobResultLink = document.getElementById('job-result-link');
const jobResultAnchor = document.getElementById('job-result-anchor');
const runBtn = document.getElementById('run-btn');
let pollingInterval = null;
let lastLineCount = 0;
let currentJobId = null;

const QUICK_JOB_KEY = 'quick_job_id';


function colorLine(line) {
  const el = document.createElement('div');
  if (/error|exception|failed|traceback/i.test(line)) el.className = 'err';
  else if (/warning|warn/i.test(line)) el.className = 'warn';
  else if (/saved|done|complete/i.test(line)) el.className = 'ok';
  el.textContent = line;
  return el;
}

// 指数バックオフ付きポーリング: 1s → 1.5s → 2.25s … 上限 5s
function pollStatus(jobId) {
  let stopped = false;
  let interval = 1000;
  pollingInterval = { stop: () => { stopped = true; } };

  const tick = async () => {
    if (stopped) return;
    try {
      const res = await fetch(`/api/status/${jobId}`);
      if (!res.ok) { scheduleNext(); return; }
      const data = await res.json();
      data.output.slice(lastLineCount).forEach(line => terminal.appendChild(colorLine(line)));
      lastLineCount = data.output.length;
      terminal.scrollTop = terminal.scrollHeight;
      if (data.status !== 'running') {
        runBtn.disabled = false;
        localStorage.removeItem(QUICK_JOB_KEY);
        if (data.status === 'done') {
          jobBadge.className = 'badge badge-done';
          jobBadge.textContent = '✓ Done';
          if (data.result_dir) {
            jobResultAnchor.href = `/results/${data.result_dir}`;
            jobResultLink.style.display = 'inline';
          }
          showToast('Run completed!', 'ok');
          setTimeout(refreshResults, 600);
        } else if (data.status === 'stopped') {
          jobBadge.className = 'badge badge-failed';
          jobBadge.textContent = '■ Stopped';
          showToast('Run stopped.', 'warn');
        } else {
          jobBadge.className = 'badge badge-failed';
          jobBadge.textContent = '✗ Failed';
          showToast('Run failed.', 'err');
        }
        return;
      }
    } catch (_) {}
    scheduleNext();
  };

  const scheduleNext = () => {
    if (stopped) return;
    setTimeout(tick, interval);
    interval = Math.min(interval * 1.5, 600000);
  };

  setTimeout(tick, interval);
}

async function stopCurrentJob() {
  try {
    if (currentJobId) {
      await fetch(`/api/stop/${currentJobId}`, { method: 'POST' });
    } else {
      await fetch('/api/shell-stop', { method: 'POST' });
    }
  } catch (_) {}
}

// ── Shell job polling (run.sh quick) ──
function pollShellJob() {
  const tick = async () => {
    try {
      const res = await fetch('/api/shell-job');
      if (!res.ok) return;
      const { running } = await res.json();
      if (!running) {
        runBtn.disabled = false;
        jobBadge.className = 'badge badge-done';
        jobBadge.textContent = 'Shell job ended';
        return;
      }
    } catch (_) {}
    setTimeout(tick, 3000);
  };
  setTimeout(tick, 3000);
}

// ── Restore running job after page navigation ──
(async () => {
  // 1) Web-started job (localStorage)
  const savedJobId = localStorage.getItem(QUICK_JOB_KEY);
  if (savedJobId) {
    try {
      const res = await fetch(`/api/status/${savedJobId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'running') {
          currentJobId = savedJobId;
          lastLineCount = data.output.length;
          document.getElementById('job-status').style.display = 'block';
          jobBadge.className = 'badge badge-running';
          jobBadge.innerHTML = '<span class="spinner"></span>&nbsp;Running';
          runBtn.disabled = true;
          data.output.forEach(line => terminal.appendChild(colorLine(line)));
          terminal.scrollTop = terminal.scrollHeight;
          pollStatus(savedJobId);
          return;
        }
      }
    } catch (_) {}
    localStorage.removeItem(QUICK_JOB_KEY);
  }

  // 2) Shell-started job (.quick.pid)
  try {
    const res = await fetch('/api/shell-job');
    if (res.ok) {
      const { running } = await res.json();
      if (running) {
        document.getElementById('job-status').style.display = 'block';
        jobBadge.className = 'badge badge-running';
        jobBadge.innerHTML = '<span class="spinner"></span>&nbsp;Running <span style="font-size:10px;opacity:.7;">(run.sh)</span>';
        terminal.style.display = 'none';
        runBtn.disabled = true;
        pollShellJob();
      }
    }
  } catch (_) {}
})();

// ── Config modal (Quick Run / Workflow settings) ──
const CM_KEY = 'config_modal_last';
const CM_DEFAULTS = {
  quick:    { n_runs: 10, max_evals: 2000,  label: '', dim: '2',
              methods: null, funcset: 'quick', funcs: null },
  workflow: { n_runs: 30, max_evals: 15000, use_all: false, label: '' },
};

// Caches populated from the API on first modal open
let CM_AVAILABLE_METHODS = null;
let CM_FUNCTIONS         = null;  // {categories: {...}, quick_12: [...]}

async function cmEnsureMethods() {
  if (CM_AVAILABLE_METHODS) return CM_AVAILABLE_METHODS;
  try {
    const res = await fetch('/api/methods');
    if (res.ok) {
      const data = await res.json();
      CM_AVAILABLE_METHODS = Array.isArray(data.methods) ? data.methods : [];
    }
  } catch (_) { CM_AVAILABLE_METHODS = []; }
  return CM_AVAILABLE_METHODS;
}
async function cmEnsureFunctions() {
  if (CM_FUNCTIONS) return CM_FUNCTIONS;
  try {
    const res = await fetch('/api/functions');
    if (res.ok) CM_FUNCTIONS = await res.json();
  } catch (_) { CM_FUNCTIONS = { categories: {}, quick_12: [] }; }
  return CM_FUNCTIONS;
}

function cmEscape(s) { return String(s).replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function cmRenderMethods(selected /* Set<string> | null = all */) {
  const box = document.getElementById('cm-methods-list-box');
  if (!box) return;
  if (!CM_AVAILABLE_METHODS || !CM_AVAILABLE_METHODS.length) {
    box.textContent = '(no methods)'; return;
  }
  box.innerHTML = CM_AVAILABLE_METHODS.map(m => {
    const checked = (!selected || selected.has(m)) ? 'checked' : '';
    return `<label class="cm-chip"><input type="checkbox" value="${cmEscape(m)}" ${checked}><span>${cmEscape(m)}</span></label>`;
  }).join('');
}

function cmRenderFunctions(selected /* Set<string> | null */) {
  const box = document.getElementById('cm-funcs-box');
  if (!box) return;
  if (!CM_FUNCTIONS || !CM_FUNCTIONS.categories) {
    box.textContent = '(no functions)'; return;
  }
  const cats = CM_FUNCTIONS.categories;
  box.innerHTML = Object.keys(cats).map(cat => {
    const fns = cats[cat];
    const items = fns.map(fn => {
      const checked = (selected && selected.has(fn)) ? 'checked' : '';
      // Show short prefix (F01, C03) as chip text; full name in tooltip
      const short = fn.split('-')[0];
      return `<label class="cm-chip" title="${cmEscape(fn)}"><input type="checkbox" value="${cmEscape(fn)}" ${checked}><span>${cmEscape(short)}</span></label>`;
    }).join('');
    return `<div class="cm-cat-row">
      <div class="cm-cat-head">
        <span class="cm-cat-name">${cmEscape(cat)}</span>
        <span class="cm-cat-count">${fns.length}</span>
        <button type="button" class="cm-mini" onclick="cmToggleCat(this, true)">全選択</button>
        <button type="button" class="cm-mini" onclick="cmToggleCat(this, false)">解除</button>
      </div>
      <div class="cm-cat-funcs">${items}</div>
    </div>`;
  }).join('');
}

function cmToggleAll(containerId, checked) {
  document.querySelectorAll(`#${containerId} input[type=checkbox]`)
    .forEach(cb => { cb.checked = checked; });
}
function cmToggleCat(btn, checked) {
  const row = btn.closest('.cm-cat-row');
  if (!row) return;
  row.querySelectorAll('input[type=checkbox]').forEach(cb => { cb.checked = checked; });
}

// Wire up funcset radio to show/hide the custom box and seed the checkboxes
function cmOnFuncsetChange() {
  const sel = document.querySelector('input[name=cm-funcset]:checked');
  if (!sel) return;
  const customBox = document.getElementById('cm-funcs-box');
  if (sel.value === 'custom') {
    customBox.style.display = '';
    // If nothing checked yet, default to quick-12 selection
    const anyChecked = customBox.querySelector('input[type=checkbox]:checked');
    if (!anyChecked && CM_FUNCTIONS) {
      cmRenderFunctions(new Set(CM_FUNCTIONS.quick_12));
    }
  } else {
    customBox.style.display = 'none';
  }
}
let cmMode = 'quick';

function openConfigModal(mode) {
  cmMode = mode;
  const overlay = document.getElementById('config-modal');
  const isQuick = mode === 'quick';

  // Load last-used values per mode, or fall back to defaults
  let saved = {};
  try { saved = (JSON.parse(localStorage.getItem(CM_KEY)) || {})[mode] || {}; } catch (_) {}
  const def = { ...CM_DEFAULTS[mode], ...saved };

  document.getElementById('cm-icon').textContent  = isQuick ? '▶' : '⚡';
  document.getElementById('cm-icon').className    = `dlg-icon-badge ${isQuick ? 'info' : 'success'}`;
  document.getElementById('cm-title').textContent = isQuick ? 'Quick Run 設定' : 'GitHub Actions 設定';
  document.getElementById('cm-msg').textContent   = isQuick
    ? 'ローカル環境で軽量動作確認を実行します。'
    : 'GitHub Actions 上で本実験ワークフローを起動します。';

  const nRunsEl    = document.getElementById('cm-n-runs');
  const maxEvalsEl = document.getElementById('cm-max-evals');
  nRunsEl.value    = def.n_runs;
  nRunsEl.max      = isQuick ? 20 : 100;
  maxEvalsEl.value = def.max_evals;
  maxEvalsEl.max   = isQuick ? 20000 : 50000;
  document.getElementById('cm-label').value     = def.label || '';
  if (isQuick) {
    document.getElementById('cm-dim').value = def.dim || '2';
    // Populate methods chips
    cmEnsureMethods().then(() => {
      const sel = (def.methods && def.methods.length)
        ? new Set(def.methods) : null;  // null = all checked
      cmRenderMethods(sel);
    });
    // Populate functions area and apply funcset selection
    cmEnsureFunctions().then(() => {
      const fsRadios = document.querySelectorAll('input[name=cm-funcset]');
      const fsVal = def.funcset || 'quick';
      fsRadios.forEach(r => { r.checked = (r.value === fsVal); });
      fsRadios.forEach(r => r.onchange = cmOnFuncsetChange);
      const customSel = (fsVal === 'custom' && def.funcs && def.funcs.length)
        ? new Set(def.funcs) : null;
      cmRenderFunctions(customSel);
      cmOnFuncsetChange();
    });
  }

  overlay.querySelectorAll('.cm-quick-only').forEach(el => {
    el.style.display = isQuick ? '' : 'none';
  });
  const okBtn = document.getElementById('cm-ok');
  okBtn.textContent = isQuick ? '▶ 実行する' : '⚡ トリガー';
  okBtn.className   = `btn btn-sm dlg-ok ${isQuick ? 'btn-primary' : 'btn-primary'}`;

  overlay.classList.add('open');
  setTimeout(() => nRunsEl.focus(), 40);
}

function closeConfigModal() {
  document.getElementById('config-modal').classList.remove('open');
}

function _saveConfigModal(mode, vals) {
  try {
    const all = JSON.parse(localStorage.getItem(CM_KEY)) || {};
    all[mode] = vals;
    localStorage.setItem(CM_KEY, JSON.stringify(all));
  } catch (_) {}
}

async function _runQuickFromModal() {
  // Collect selected methods from chips
  const methodsChecked = Array.from(
    document.querySelectorAll('#cm-methods-list-box input[type=checkbox]:checked')
  ).map(cb => cb.value);
  const allMethodsChecked = (
    CM_AVAILABLE_METHODS && methodsChecked.length === CM_AVAILABLE_METHODS.length
  );

  const funcset = (document.querySelector('input[name=cm-funcset]:checked') || {}).value || 'quick';
  let funcsList = [];
  if (funcset === 'custom') {
    funcsList = Array.from(
      document.querySelectorAll('#cm-funcs-box input[type=checkbox]:checked')
    ).map(cb => cb.value);
  }

  const vals = {
    n_runs:    document.getElementById('cm-n-runs').value,
    max_evals: document.getElementById('cm-max-evals').value,
    label:     document.getElementById('cm-label').value.trim(),
    dim:       document.getElementById('cm-dim').value,
    methods:   methodsChecked,
    funcset:   funcset,
    funcs:     funcsList,
  };
  _saveConfigModal('quick', vals);

  // Sanity: require at least 1 method and (when custom) at least 1 func
  if (!methodsChecked.length) {
    alert('比較手法を1つ以上選択してください'); return;
  }
  if (funcset === 'custom' && !funcsList.length) {
    alert('関数を1つ以上選択してください'); return;
  }

  closeConfigModal();

  const form = new FormData();
  form.append('n_runs',    vals.n_runs);
  form.append('max_evals', vals.max_evals);
  form.append('label',     vals.label);
  form.append('dim',       vals.dim);
  if (funcset === 'all')          form.append('use_all', 'true');
  else if (funcset === 'custom')  form.append('funcs',   funcsList.join(','));
  // methods: only send when not all-selected (server treats absent = all)
  if (!allMethodsChecked) form.append('methods', methodsChecked.join(','));

  if (pollingInterval) pollingInterval.stop();
  terminal.innerHTML = '';
  lastLineCount = 0;
  runBtn.disabled = true;
  document.getElementById('job-status').style.display = 'block';
  jobBadge.className = 'badge badge-running';
  jobBadge.innerHTML = '<span class="spinner"></span>&nbsp;Running';
  jobResultLink.style.display = 'none';
  const res = await fetch('/api/run', { method: 'POST', body: form });
  const { job_id } = await res.json();
  currentJobId = job_id;
  localStorage.setItem(QUICK_JOB_KEY, job_id);
  pollStatus(job_id);
}

async function _runWorkflowFromModal() {
  const vals = {
    n_runs:    document.getElementById('cm-n-runs').value,
    max_evals: document.getElementById('cm-max-evals').value,
  };
  _saveConfigModal('workflow', vals);
  closeConfigModal();

  const form = new FormData();
  form.append('n_runs',    vals.n_runs);
  form.append('max_evals', vals.max_evals);

  const btn = document.getElementById('gh-btn');
  btn.disabled = true; btn.textContent = 'Triggering…';
  const res = await fetch('/api/gh-trigger', { method: 'POST', body: form });
  const data = await res.json();
  btn.disabled = false; btn.textContent = '⚡ Trigger Workflow';
  showToast(data.message, data.ok ? 'ok' : 'err');
  if (data.ok) setTimeout(loadGhRuns, 2000);
}

// Modal init: bind buttons + close-on-overlay-click + Esc/Enter shortcuts
document.addEventListener('DOMContentLoaded', () => {
  const overlay = document.getElementById('config-modal');
  if (!overlay) return;
  document.getElementById('cm-cancel').addEventListener('click', closeConfigModal);
  document.getElementById('cm-ok').addEventListener('click', () => {
    cmMode === 'quick' ? _runQuickFromModal() : _runWorkflowFromModal();
  });
  overlay.addEventListener('click', e => { if (e.target === overlay) closeConfigModal(); });
  document.addEventListener('keydown', e => {
    if (!overlay.classList.contains('open')) return;
    if (e.key === 'Escape') { e.preventDefault(); closeConfigModal(); return; }
    if (e.key === 'Enter') {
      // Don't auto-submit while the user is typing in any input/textarea.
      // Submit only when focus is on the OK button (default <button> behavior).
      const tag = (e.target.tagName || '').toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      e.preventDefault();
      cmMode === 'quick' ? _runQuickFromModal() : _runWorkflowFromModal();
    }
  });
});

// ── Remote Runs ──
async function loadGhRuns() {
  const list = document.getElementById('gh-runs-list');
  list.innerHTML = '<p class="empty-state"><span class="spinner"></span> Loading…</p>';
  try {
    const res = await fetch('/api/gh-runs');
    if (!res.ok) throw new Error(await res.text());
    const runs = await res.json();
    if (!runs.length) { list.innerHTML = '<p class="empty-state">No runs found.</p>'; return; }
    list.innerHTML = '';
    runs.forEach(run => {
      const row = document.createElement('div');
      row.className = 'gh-run-row';
      row.id = `gh-run-${run.databaseId}`;
      const status = run.conclusion || run.status;
      const badgeClass = status === 'success' ? 'badge-done'
        : status === 'failure' ? 'badge-failed' : 'badge-running';
      const canDownload = run.conclusion === 'success';
      const date = new Date(run.createdAt).toLocaleString('ja-JP',
        {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
      row.innerHTML = `
        <div class="gh-run-meta">
          <span class="badge ${badgeClass}" style="font-size:10px;">${status}</span>
          <span style="font-size:10.5px;color:var(--muted);">${date}</span>
        </div>
        <div class="gh-run-title">${run.name} <span style="color:var(--muted);font-size:10.5px;">${(run.headSha || '').slice(0,7)}</span></div>
        ${canDownload ? `<button class="btn btn-outline btn-sm" id="dl-btn-${run.databaseId}"
          onclick="startDownload('${run.databaseId}')">⬇ Download</button>` : ''}
        <div id="dl-status-${run.databaseId}" style="font-size:11px;margin-top:4px;"></div>
      `;
      list.appendChild(row);
    });
    resumeDownloadPolling();
  } catch(e) {
    list.innerHTML = `<p class="empty-state" style="color:var(--danger);">Error: ${e.message}</p>`;
  }
}

// ── Download ──
const DL_JOB_KEY = 'dl_job';

async function startDownload(ghRunId) {
  const label = await dlg.prompt('この結果の実験名を入力してください（省略可）:', '');
  if (label === null) return;
  const btn = document.getElementById(`dl-btn-${ghRunId}`);
  if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
  const form = new FormData();
  form.append('run_id', ghRunId);
  form.append('label', label);
  const res = await fetch('/api/download', { method: 'POST', body: form });
  const { job_id } = await res.json();
  localStorage.setItem(DL_JOB_KEY, JSON.stringify({ job_id, gh_run_id: ghRunId }));
  pollDownload(job_id, ghRunId);
}

// ダウンロードは通信量が多いので初回 2s、以降 3s → 6s まで緩やかに伸ばす
function pollDownload(jobId, ghRunId) {
  const statusEl = document.getElementById(`dl-status-${ghRunId}`);
  const btn = document.getElementById(`dl-btn-${ghRunId}`);
  let interval = 2000;

  const tick = async () => {
    try {
      const res = await fetch(`/api/dl-status/${jobId}`);
      if (res.status === 404) {
        // Job state was lost (likely Flask reload during dev). Reset UI.
        localStorage.removeItem(DL_JOB_KEY);
        if (statusEl) {
          statusEl.innerHTML = `<span style="color:var(--danger);">ジョブ状態が失われました（サーバ再起動）。再ダウンロードしてください。</span>`;
        }
        if (btn) { btn.disabled = false; btn.textContent = '⬇ Download'; }
        return;
      }
      if (!res.ok) { scheduleNext(); return; }
      const data = await res.json();
      if (statusEl) {
        if (data.status === 'running' && typeof data.progress === 'number') {
          statusEl.innerHTML = `
            <div style="font-size:11px;color:var(--text-2);margin-bottom:3px;">${data.message}</div>
            <div style="height:5px;background:var(--surface-2);border-radius:3px;overflow:hidden;border:1px solid var(--border-light);">
              <div style="height:100%;width:${data.progress}%;background:linear-gradient(90deg,var(--accent-muted) 0%,var(--accent) 100%);transition:width .4s ease;"></div>
            </div>`;
        } else {
          const color = data.status === 'done' ? 'var(--success)'
            : data.status === 'failed' ? 'var(--danger)' : 'var(--warn)';
          statusEl.innerHTML = `<span style="color:${color};">${data.message}</span>`;
        }
      }
      if (data.status !== 'running') {
        localStorage.removeItem(DL_JOB_KEY);
        if (btn) { btn.disabled = false; btn.textContent = '⬇ Download'; }
        if (data.status === 'done') {
          showToast(data.message, 'ok');
          setTimeout(refreshResults, 600);
        } else {
          showToast(data.message, 'err');
        }
        return;
      }
    } catch (_) {}
    scheduleNext();
  };

  const scheduleNext = () => {
    setTimeout(tick, interval);
    interval = Math.min(interval * 1.5, 600000);
  };

  setTimeout(tick, interval);
}

// ── Card kebab menu ──
let _openMenu = null;

function toggleCardMenu(runId, e) {
  e.preventDefault(); e.stopPropagation();
  const btn  = e.currentTarget;
  const menu = document.getElementById(`menu-${runId}`);
  const isOpen = menu.classList.contains('open');
  closeAllMenus();
  if (!isOpen) {
    menu.classList.add('open');
    btn.classList.add('open');
    const r = btn.getBoundingClientRect();
    menu.style.top  = (r.bottom + 4) + 'px';
    menu.style.left = Math.max(4, r.right - 132) + 'px';
    _openMenu = runId;
  }
}

function closeAllMenus() {
  document.querySelectorAll('.card-menu-dropdown.open').forEach(m => m.classList.remove('open'));
  document.querySelectorAll('.card-menu-btn.open').forEach(b => b.classList.remove('open'));
  _openMenu = null;
}

document.addEventListener('click', closeAllMenus);

// ── Rename result ──
async function renameResult(runId, e) {
  e.preventDefault(); e.stopPropagation();
  closeAllMenus();
  const newName = await dlg.prompt('新しい名前を入力してください:', runId);
  if (!newName || newName === runId) return;
  const form = new FormData();
  form.append('new_name', newName);
  const res = await fetch(`/api/results/${encodeURIComponent(runId)}/rename`, { method: 'POST', body: form });
  const data = await res.json();
  if (data.ok) {
    showToast(`名前を変更しました: ${data.new_name}`, 'ok');
    setTimeout(() => location.reload(), 800);
  } else {
    showToast(data.message || '名前変更に失敗しました', 'err');
  }
}

// ── Delete result (dashboard) ──
async function deleteResult(runId, e) {
  e.preventDefault(); e.stopPropagation();
  closeAllMenus();
  if (!await dlg.confirm(`「${runId}」を削除しますか？\nこの操作は元に戻せません。`)) return;
  const res = await fetch(`/api/results/${encodeURIComponent(runId)}`, { method: 'DELETE' });
  const data = await res.json();
  if (data.ok) {
    const card = document.getElementById(`card-${runId}`);
    if (card) card.remove();
    showToast('削除しました', 'ok');
  } else {
    showToast(data.message || '削除に失敗しました', 'err');
  }
}

function resumeDownloadPolling() {
  const stored = localStorage.getItem(DL_JOB_KEY);
  if (!stored) return;
  try {
    const { job_id, gh_run_id } = JSON.parse(stored);
    const btn = document.getElementById(`dl-btn-${gh_run_id}`);
    if (btn) { btn.disabled = true; btn.textContent = 'Downloading…'; }
    pollDownload(job_id, gh_run_id);
  } catch(e) {
    localStorage.removeItem(DL_JOB_KEY);
  }
}
