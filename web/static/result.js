const _pd    = JSON.parse(document.getElementById('page-data').textContent);
const RUN_ID = _pd.run_id;
const DIMS   = _pd.dims;
let   DIMS_DATA = _pd.dims_data;
const ALL_RESULTS_META = _pd.all_results_meta || {};

let viewMode      = 'function';
let currentDim    = DIMS[0] || null;
let currentFunc   = null;
let currentMethod = null;
let currentType   = 'evals';
let currentMediaSrc = null;
let mediaIndex    = null;  // loaded from /api/media-index

// Compare mode selections
let cmpSelectedFuncs   = new Set();
let cmpSelectedMethods = new Set();

// ── ALL_TYPE_LABELS ──────────────────────────────────────────────────────────
const TYPE_LABELS = {
  'evals':              '評価点の蓄積',
  'evals_failed':       '評価点 (失敗)',
  'runs':               '探索軌跡',
  'population':         '集団の動き',
  'population_failed':  '集団 (失敗)',
  'landscape':          '関数地形',
  'convergence':        '収束曲線',
  'outbreak_dyn':         'アウトブレイク動態',
  'outbreak_dyn_failed':  'アウトブレイク動態 (失敗)',
  '3devals':            '3D 評価点',
  '3devals_failed':     '3D 評価点 (失敗)',
  '3dpopulation':       '3D 集団',
  '3dpopulation_failed':'3D 集団 (失敗)',
};

// Type display order
const TYPE_ORDER = [
  'evals', 'evals_failed', 'runs', 'population', 'population_failed',
  'landscape', 'convergence', 'outbreak_dyn', 'outbreak_dyn_failed',
  '3devals', '3devals_failed', '3dpopulation', '3dpopulation_failed',
];

// ── Overall evaluation constants ─────────────────────────────────────────────
const CAT_LABELS = {
  'separable':      'Separable',
  'moderate-cond':  'Moderate',
  'ill-cond':       'High Cond.',
  'multimodal':     'Multi-modal',
  'weak-structure': 'Weak Structure',
  'multi-optima':   'Custom',
  'deceptive-2d':   'Deceptive 2D',
};
let overallData = null;
let overallSortKey = 'bf';   // 'bf' | 'evals' | 'ecdf' — leaderboard sort key

function _sortedLeaderboard() {
  if (!overallData?.leaderboard) return [];
  const key = overallSortKey;
  const others = ['bf', 'evals', 'ecdf'].filter(k => k !== key);
  return [...overallData.leaderboard].sort((a, b) =>
    (a[`mean_rank_${key}`]      - b[`mean_rank_${key}`])
    || (a[`mean_rank_${others[0]}`] - b[`mean_rank_${others[0]}`])
    || (a[`mean_rank_${others[1]}`] - b[`mean_rank_${others[1]}`])
  );
}

// ── URL hash persistence: #dim2/F01-Sphere ──
function _parseHash() {
  const h = location.hash.slice(1);
  if (!h) return { dim: null, func: null };
  const parts = h.split('/');
  return parts.length >= 2
    ? { dim: parts[0], func: decodeURIComponent(parts.slice(1).join('/')) }
    : { dim: null,     func: decodeURIComponent(parts[0]) };
}
function _updateHash() {
  if (!currentDim || !currentFunc) return;
  history.replaceState(null, '', `#${currentDim}/${encodeURIComponent(currentFunc)}`);
}

// ── Sidebar run header init ──
function initSidebarRunHeader() {
  const meta = ALL_RESULTS_META[RUN_ID] || {};
  const chip = document.getElementById('srh-type-chip');
  const hdr  = document.getElementById('sidebar-run-header');
  if (chip) {
    const type = meta.type || '';
    chip.textContent = type || '—';
    chip.className = `srh-type-chip ${type || 'unknown'}`;
  }
  if (hdr) {
    const color = meta.type === 'quick' ? '#7c3aed' : meta.type === 'workflow' ? '#0284c7' : 'var(--accent)';
    hdr.style.borderTopColor = color;
  }
  const idEl = document.getElementById('sidebar-run-id');
  if (idEl) idEl.textContent = RUN_ID;
  const metaEl = document.getElementById('sidebar-run-meta');
  if (metaEl) {
    const lines = [];
    if (meta.created_at) {
      const [d, t] = meta.created_at.split('T');
      lines.push({ cls: 'srm-primary', text: d + (t ? '  ' + t.slice(0, 5) : '') });
    }
    const runInfo = [];
    if (meta.n_runs    != null) runInfo.push(`${meta.n_runs} runs`);
    if (meta.max_evals != null) runInfo.push(`${Number(meta.max_evals).toLocaleString()} evals`);
    if (runInfo.length) lines.push({ cls: '', text: runInfo.join(' · ') });
    if (meta.commit)    lines.push({ cls: 'srm-small', text: meta.commit });
    metaEl.innerHTML = lines.map(l =>
      `<span class="srm-line${l.cls ? ' ' + l.cls : ''}">${l.text}</span>`
    ).join('');
  }
  document.querySelectorAll('.sidebar-runs-section .srun-raw').forEach(el => {
    el.className = 'srun-primary';
  });
}

// ── Function groups ──
const FUNC_GROUPS = [
  { label: 'Separable',      color: '#4e9af1', ids: ['f01','f02','f03','f04','f05'] },
  { label: 'Moderate',       color: '#7dc87d', ids: ['f06','f07','f08','f09'] },
  { label: 'High Cond.',     color: '#e5a94c', ids: ['f10','f11','f12','f13','f14'] },
  { label: 'Multi-modal',    color: '#c07fe0', ids: ['f15','f16','f17','f18','f19'] },
  { label: 'Weak Structure', color: '#e06060', ids: ['f20','f21','f22','f23','f24'] },
];
function getFuncGroup(name) {
  const lc = name.toLowerCase();
  return FUNC_GROUPS.find(g => g.ids.some(id => lc.startsWith(id))) || null;
}

// ── Media URL resolver ──────────────────────────────────────────────────────
function mediaUrl(func, method, type) {
  if (type === 'landscape')   return `/media/${RUN_ID}/${currentDim}/${func}_landscape.svg`;
  if (type === 'convergence') return `/media/${RUN_ID}/${currentDim}/${func}_convergence.svg`;
  const entry = mediaIndex?.files.find(
    f => f.func === func && f.method === method && f.type === type
  );
  const ext = entry?.ext || 'webp';
  return `/media/${RUN_ID}/${currentDim}/${func}_${method}_${type}.${ext}`;
}

// Types that are always shown when funcs exist (function-level, not method-level)
const FUNC_LEVEL_TYPES = new Set(['landscape', 'convergence']);

// ── Build view type selector from index ────────────────────────────────────
function buildTypeSelector() {
  const sel = document.getElementById('media-selector');
  sel.innerHTML = '';

  const hasFuncs = mediaIndex && mediaIndex.funcs && mediaIndex.funcs.length > 0;
  const types = hasFuncs
    ? TYPE_ORDER.filter(t => FUNC_LEVEL_TYPES.has(t) || mediaIndex.types.includes(t))
    : [];

  if (!types.includes(currentType)) currentType = types[0] || 'evals';

  types.forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'vs-btn' + (t === currentType ? ' active' : '');
    btn.dataset.type = t;
    btn.textContent = TYPE_LABELS[t] || t;
    sel.appendChild(btn);
  });
}

// ── Switch view mode ────────────────────────────────────────────────────────
function switchViewMode(mode) {
  viewMode = mode;
  ['function','method','compare'].forEach(m => {
    document.getElementById(`vm-${m}`).classList.toggle('active', m === mode);
  });

  // Leaving overall mode? Restore a per-function selection so the panes have something to show.
  if (currentFunc === '__overall__') {
    setOverallMode(false);
    document.getElementById('overall-entry')?.classList.remove('active');
    const fns = (DIMS_DATA[currentDim] || {}).functions || [];
    if (fns.length) {
      currentFunc = fns[0];
      document.querySelectorAll('#func-list a')
        .forEach(a => a.classList.toggle('active', a.dataset.func === currentFunc));
      document.querySelectorAll('.card-func-badge').forEach(el => el.textContent = currentFunc);
      _updateHash();
      buildUnifiedTable(currentFunc);
    }
  }

  // Show/hide sidebar panels
  document.getElementById('func-panel').style.display    = mode === 'function' ? '' : 'none';
  document.getElementById('method-panel').style.display  = mode === 'method'   ? '' : 'none';
  document.getElementById('compare-panel').style.display = mode === 'compare'  ? 'flex' : 'none';

  // Show/hide main panels
  document.getElementById('media-grid-wrap').style.display    = mode === 'function' ? '' : 'none';
  document.getElementById('method-grid-wrap').style.display   = mode === 'method'   ? '' : 'none';
  document.getElementById('compare-matrix-wrap').style.display = mode === 'compare'  ? '' : 'none';
  document.getElementById('results-card').style.display       = mode === 'function'  ? '' : 'none';

  if (mode === 'function') {
    renderFunctionGrid();
  } else if (mode === 'method') {
    buildMethodList();
    renderMethodGrid();
  } else if (mode === 'compare') {
    buildComparePanel();
    renderCompareMatrix();
  }
}

// ── Dimension switch ─────────────────────────────────────────────────────────
function switchDim(dim) {
  overallData = null;
  currentDim = dim;
  document.querySelectorAll('#dim-tabs button')
    .forEach(b => b.classList.toggle('active', b.dataset.dim === dim));
  loadMediaIndex().then(() => {
    buildTypeSelector();
    buildFuncList();
    if (viewMode === 'method') {
      buildMethodList();
      renderMethodGrid();
    } else if (viewMode === 'compare') {
      buildComparePanel();
      renderCompareMatrix();
    }
  });
}

// ── Load media index from API ────────────────────────────────────────────────
async function loadMediaIndex() {
  try {
    const res = await fetch(`/api/media-index/${encodeURIComponent(RUN_ID)}/${currentDim}`);
    if (res.ok) {
      mediaIndex = await res.json();
    }
  } catch (_) {
    mediaIndex = null;
  }
}

// ── Function list ────────────────────────────────────────────────────────────
function buildFuncList() {
  const list    = document.getElementById('func-list');
  const countEl = document.getElementById('func-count');
  list.innerHTML = '';
  const rawFunctions = (DIMS_DATA[currentDim] || {}).functions || [];
  // Order: known groups in FUNC_GROUPS order, then Custom (ungrouped) at the bottom
  const _grpIdx = f => {
    const g = getFuncGroup(f);
    if (!g) return FUNC_GROUPS.length;
    return FUNC_GROUPS.findIndex(x => x.label === g.label);
  };
  const functions = [...rawFunctions].sort((a, b) => {
    const ia = _grpIdx(a), ib = _grpIdx(b);
    if (ia !== ib) return ia - ib;
    return a.localeCompare(b);
  });
  if (countEl) countEl.textContent = functions.length;

  let lastGroupLabel = null;
  let customHeaderAdded = false;
  functions.forEach((f, i) => {
    const grp    = getFuncGroup(f);
    const grpKey = grp ? grp.label : '__custom__';
    if (grpKey !== lastGroupLabel) {
      lastGroupLabel = grpKey;
      const hdr = document.createElement('li');
      hdr.className = 'func-group-header';
      if (grp) {
        hdr.innerHTML = `<span class="func-group-dot" style="background:${grp.color}"></span>${grp.label}`;
      } else if (!customHeaderAdded) {
        customHeaderAdded = true;
        hdr.innerHTML = `<span class="func-group-dot" style="background:var(--muted)"></span>Custom`;
      }
      list.appendChild(hdr);
    }
    const li = document.createElement('li');
    const a  = document.createElement('a');
    a.href = '#';
    a.textContent = f;
    a.dataset.func = f;
    a.style.setProperty('border-left-color', grp ? grp.color : 'var(--muted)', 'important');
    a.addEventListener('click', e => { e.preventDefault(); selectFunc(f); });
    li.appendChild(a);
    list.appendChild(li);
  });

  const { func: hashFunc } = _parseHash();
  if (hashFunc && functions.includes(hashFunc)) {
    selectFunc(hashFunc);
  } else {
    // Default landing: 全体評価
    selectOverall();
  }
}

function selectFunc(func) {
  setOverallMode(false);
  currentFunc = func;
  document.getElementById('overall-entry')?.classList.remove('active');
  document.querySelectorAll('#func-list a')
    .forEach(a => a.classList.toggle('active', a.dataset.func === func));
  document.querySelectorAll('.card-func-badge').forEach(el => el.textContent = func);
  _updateHash();
  renderFunctionGrid();
  buildUnifiedTable(func);
}

// ── Method list (Method mode) ────────────────────────────────────────────────
function buildMethodList() {
  const list    = document.getElementById('method-list');
  const countEl = document.getElementById('method-count');
  list.innerHTML = '';
  const methods = (mediaIndex && mediaIndex.methods) ? mediaIndex.methods : [];
  if (countEl) countEl.textContent = methods.length;

  if (!methods.length) {
    list.innerHTML = '<li style="padding:10px;color:var(--muted);font-size:11px;">手法データがありません。</li>';
    return;
  }

  methods.forEach((m, i) => {
    const li = document.createElement('li');
    const a  = document.createElement('a');
    a.href = '#';
    a.textContent = m;
    a.dataset.method = m;
    if (i === 0 && !currentMethod) { currentMethod = m; a.classList.add('active'); }
    if (m === currentMethod) a.classList.add('active');
    a.addEventListener('click', e => { e.preventDefault(); selectMethod(m); });
    li.appendChild(a);
    list.appendChild(li);
  });

  if (!currentMethod && methods.length) currentMethod = methods[0];
}

function selectMethod(method) {
  currentMethod = method;
  document.querySelectorAll('#method-list a')
    .forEach(a => a.classList.toggle('active', a.dataset.method === method));
  document.getElementById('media-func-badge').textContent = method;
  renderMethodGrid();
}

// ── Render: Function mode grid (all methods for current function) ─────────────
function renderFunctionGrid() {
  const grid = document.getElementById('media-grid');
  grid.innerHTML = '';
  grid.classList.remove('solo');
  if (!currentFunc || !mediaIndex) return;

  // Function-level types (landscape/convergence): single full-width cell
  if (FUNC_LEVEL_TYPES.has(currentType)) {
    grid.classList.add('solo');
    const url = mediaUrl(currentFunc, null, currentType);
    grid.appendChild(makeGridCell(currentFunc, null, currentType, url, currentFunc));
    return;
  }

  const methods = mediaIndex.methods || [];
  if (!methods.length) {
    grid.innerHTML = '<p class="empty-state">手法別データがありません。</p>';
    return;
  }

  methods.forEach(method => {
    const url = mediaUrl(currentFunc, method, currentType);
    grid.appendChild(makeGridCell(currentFunc, method, currentType, url, method));
  });
}

// ── Render: Method mode grid (all functions for current method) ──────────────
function renderMethodGrid() {
  const grid = document.getElementById('method-grid');
  grid.innerHTML = '';
  if (!mediaIndex) return;

  const funcs = mediaIndex.funcs || [];
  if (!funcs.length) {
    grid.innerHTML = '<p class="empty-state">関数データがありません。</p>';
    return;
  }

  // Function-level types (landscape/convergence): ignore selected method
  if (FUNC_LEVEL_TYPES.has(currentType)) {
    funcs.forEach(func => {
      const url = mediaUrl(func, null, currentType);
      grid.appendChild(makeGridCell(func, null, currentType, url, func));
    });
    return;
  }

  if (!currentMethod) return;
  funcs.forEach(func => {
    const url = mediaUrl(func, currentMethod, currentType);
    grid.appendChild(makeGridCell(func, currentMethod, currentType, url, func));
  });
}

// ── Render: Compare mode matrix ──────────────────────────────────────────────
function renderCompareMatrix() {
  const table = document.getElementById('compare-matrix');
  table.innerHTML = '';
  if (!mediaIndex) return;

  const funcs   = [...cmpSelectedFuncs];
  const methods = [...cmpSelectedMethods];

  // Function-level types: render a simple per-function grid (no method axis)
  if (FUNC_LEVEL_TYPES.has(currentType)) {
    if (!funcs.length) {
      table.innerHTML = '<tr><td style="padding:20px;color:var(--muted);font-size:12px;">関数を選択してください。</td></tr>';
      return;
    }
    const thead = document.createElement('thead');
    const hrow  = document.createElement('tr');
    hrow.appendChild(Object.assign(document.createElement('th'), { textContent: '' }));
    const th = document.createElement('th');
    th.textContent = TYPE_LABELS[currentType] || currentType;
    hrow.appendChild(th);
    thead.appendChild(hrow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    funcs.forEach(func => {
      const tr = document.createElement('tr');
      const rowHdr = document.createElement('td');
      rowHdr.className = 'row-header';
      rowHdr.textContent = func;
      tr.appendChild(rowHdr);
      const td = document.createElement('td');
      const url = mediaUrl(func, null, currentType);
      const div = document.createElement('div');
      div.className = 'compare-cell';
      const img = document.createElement('img');
      img.src = url; img.alt = func; img.style.cursor = 'zoom-in';
      img.onclick = () => openOverlay(url);
      img.onerror = () => { div.className = 'compare-cell not-available'; div.innerHTML = '<span style="color:var(--muted);font-size:10px;">N/A</span>'; };
      div.appendChild(img); td.appendChild(div); tr.appendChild(td);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return;
  }

  if (!funcs.length || !methods.length) {
    table.innerHTML = '<tr><td style="padding:20px;color:var(--muted);font-size:12px;">関数と手法を選択してください。</td></tr>';
    return;
  }

  // Header row
  const thead = document.createElement('thead');
  const hrow  = document.createElement('tr');
  hrow.appendChild(Object.assign(document.createElement('th'), { textContent: '' }));
  methods.forEach(m => {
    const th = document.createElement('th');
    th.textContent = m;
    hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  table.appendChild(thead);

  // Data rows
  const tbody = document.createElement('tbody');
  funcs.forEach(func => {
    const tr = document.createElement('tr');
    const th = document.createElement('td');
    th.className = 'row-header';
    th.textContent = func;
    tr.appendChild(th);
    methods.forEach(method => {
      const td = document.createElement('td');
      const url = mediaUrl(func, method, currentType);
      const div = document.createElement('div');
      div.className = 'compare-cell';
      const img = document.createElement('img');
      img.src = url;
      img.alt = `${func} / ${method}`;
      img.style.cursor = 'zoom-in';
      img.onclick = () => openOverlay(url);
      img.onerror = () => { div.className = 'compare-cell not-available'; div.innerHTML = '<span style="color:var(--muted);font-size:10px;">N/A</span>'; };
      div.appendChild(img);
      td.appendChild(div);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

// ── Grid cell factory ────────────────────────────────────────────────────────
function makeGridCell(func, method, type, url, label) {
  const cell = document.createElement('div');
  cell.className = 'grid-cell';

  const lbl = document.createElement('div');
  lbl.className = 'grid-cell-label';
  lbl.textContent = label;
  cell.appendChild(lbl);

  const wrap = document.createElement('div');
  wrap.className = 'grid-cell-img-wrap';
  const img = document.createElement('img');
  img.src = url;
  img.alt = label;
  img.loading = 'lazy';
  img.onerror = () => cell.classList.add('not-available');
  img.onclick = () => openOverlay(url);
  wrap.appendChild(img);
  cell.appendChild(wrap);

  return cell;
}

// ── Compare panel checkboxes ─────────────────────────────────────────────────
function buildComparePanel() {
  if (!mediaIndex) return;
  const funcs   = mediaIndex.funcs   || [];
  const methods = mediaIndex.methods || [];

  // Init selections if empty
  if (!cmpSelectedFuncs.size)   funcs.slice(0, 4).forEach(f => cmpSelectedFuncs.add(f));
  if (!cmpSelectedMethods.size) methods.forEach(m => cmpSelectedMethods.add(m));

  buildCheckList('cmp-func-list', funcs, cmpSelectedFuncs, (f, checked) => {
    if (checked) cmpSelectedFuncs.add(f); else cmpSelectedFuncs.delete(f);
    renderCompareMatrix();
  });
  buildCheckList('cmp-method-list', methods, cmpSelectedMethods, (m, checked) => {
    if (checked) cmpSelectedMethods.add(m); else cmpSelectedMethods.delete(m);
    renderCompareMatrix();
  });
}

function buildCheckList(containerId, items, selected, onChange) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'cmp-check-item';
    const id = `chk-${containerId}-${item}`;
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.id = id;
    chk.checked = selected.has(item);
    chk.onchange = () => onChange(item, chk.checked);
    const lbl = document.createElement('label');
    lbl.htmlFor = id;
    lbl.textContent = item;
    div.append(chk, lbl);
    container.appendChild(div);
  });
}

function cmpSelectAll(type) {
  if (!mediaIndex) return;
  const items = type === 'func' ? (mediaIndex.funcs || []) : (mediaIndex.methods || []);
  const set   = type === 'func' ? cmpSelectedFuncs : cmpSelectedMethods;
  const containerId = type === 'func' ? 'cmp-func-list' : 'cmp-method-list';
  items.forEach(i => set.add(i));
  document.querySelectorAll(`#${containerId} input[type=checkbox]`).forEach(c => { c.checked = true; });
  renderCompareMatrix();
}

// ── Overall evaluation ────────────────────────────────────────────────────────
function setOverallMode(on) {
  document.getElementById('media-card').style.display   = on ? 'none' : '';
  document.getElementById('results-card').style.display = on ? 'none' : '';
  document.getElementById('overall-wrap').style.display = on ? '' : 'none';
}

async function selectOverall() {
  currentFunc = '__overall__';
  document.querySelectorAll('#func-list a').forEach(a => a.classList.remove('active'));
  document.getElementById('overall-entry')?.classList.add('active');
  document.querySelectorAll('.card-func-badge').forEach(el => el.textContent = '');
  setOverallMode(true);
  if (!overallData) await fetchOverallData();
  renderOverall();
}

async function fetchOverallData() {
  try {
    const res = await fetch(`/api/overall/${encodeURIComponent(RUN_ID)}/${currentDim}`);
    if (res.ok) overallData = await res.json();
  } catch (_) { overallData = null; }
}

function renderOverall() {
  if (!overallData || !overallData.leaderboard || !overallData.leaderboard.length) {
    document.getElementById('overall-leaderboard-container').innerHTML =
      '<p class="empty-state">データがありません。</p>';
    return;
  }
  const lb = _sortedLeaderboard();
  _renderLeaderboard(lb);
  _renderHeatmap(lb);
  _renderRankProfile(lb);
  renderOverallWilcoxon();
}

function _renderLeaderboard(lb) {
  const nFuncs = (overallData.funcs || []).length;
  const fried = overallData.friedman || {};
  // Normalize mean_rank to a 0..1 fill where rank=1 -> full, rank=n -> empty.
  const nMethods = lb.length;
  const fillFor = mr => {
    if (nMethods <= 1) return 1;
    return Math.max(0, Math.min(1, (nMethods - mr) / (nMethods - 1)));
  };

  // Friedman test stats line (per indicator).
  const statBlock = (ind, label) => {
    const s = fried[ind];
    if (!s) return '';
    const chi2 = s.chi2 == null ? '—' : s.chi2.toFixed(2);
    const pStr = s.p == null ? '—' : (s.p < 1e-4 ? s.p.toExponential(1) : s.p.toFixed(4));
    const pCls = (s.p != null && s.p < 0.05) ? 'sig' : 'nonsig';
    const cd   = s.cd == null ? '—' : s.cd.toFixed(2);
    return `
      <div class="stat-block">
        <span class="ind">${label}</span>
        <span class="label">χ²_F</span><span>${chi2}</span>
        <span class="label">p</span><span class="pval ${pCls}">${pStr}</span>
        <span class="label">CD₀.₀₅</span><span>${cd}</span>
      </div>`;
  };
  let statsHtml = `
    <div class="ov-friedman-stats">
      ${statBlock('bf', 'bf')}
      ${statBlock('evals', 'Evals')}
      ${statBlock('ecdf', 'ECDF')}
      <div class="stat-block">
        <span class="label">N (funcs)</span><span>${nFuncs}</span>
        <span class="label">k (methods)</span><span>${nMethods}</span>
      </div>
    </div>`;

  const sortHdr = (key, label) => {
    const isActive = overallSortKey === key;
    const arrow = isActive ? '▼' : '↕';
    return `<div><span class="ov-rank-sort ${isActive ? 'active' : ''}" data-sort-key="${key}" title="${label}でソート">Mean Rank (${label})<span class="sort-arrow">${arrow}</span></span></div>`;
  };
  let html = statsHtml + `
    <div class="ov-rank-header">
      <div>#</div><div>Method</div>
      ${sortHdr('bf', 'bf')}${sortHdr('evals', 'Evals')}${sortHdr('ecdf', 'ECDF')}
      <div>SR@1e-4</div><div>#Best bf/Evals/ECDF</div><div>#Worst bf/Evals/ECDF</div>
    </div>
    <div class="ov-ranking">`;
  lb.forEach((row, i) => {
    const medalCls = i < 3 ? `is-${i + 1}` : '';
    const fillBf    = (fillFor(row.mean_rank_bf)    * 100).toFixed(1);
    const fillEvals = (fillFor(row.mean_rank_evals) * 100).toFixed(1);
    const fillEcdf  = (fillFor(row.mean_rank_ecdf)  * 100).toFixed(1);
    const sr = (row.mean_sr * 100).toFixed(1) + '%';
    const bestStr  = `${row.n_best_bf}/${row.n_best_evals}/${row.n_best_ecdf}`;
    const worstStr = `${row.n_worst_bf}/${row.n_worst_evals}/${row.n_worst_ecdf}`;
    const anyBest  = row.n_best_bf > 0 || row.n_best_evals > 0 || row.n_best_ecdf > 0;
    const anyWorst = row.n_worst_bf > 0 || row.n_worst_evals > 0 || row.n_worst_ecdf > 0;
    const bar = (fill, val, std) => `
      <div class="ov-rank-barwrap">
        <div class="ov-rank-bar"><div class="ov-rank-bar-fill ${medalCls}" style="width:${fill}%"></div></div>
        <div class="ov-rank-barval">${val.toFixed(2)}<span class="std">±${std.toFixed(2)}</span></div>
      </div>`;
    html += `
      <div class="ov-ranking-row ${medalCls}">
        <div class="ov-rank-num">${i + 1}</div>
        <div class="ov-rank-method" title="${htmlesc(row.method)}">${htmlesc(row.method)}</div>
        ${bar(fillBf,    row.mean_rank_bf,    row.rank_std_bf)}
        ${bar(fillEvals, row.mean_rank_evals, row.rank_std_evals)}
        ${bar(fillEcdf,  row.mean_rank_ecdf,  row.rank_std_ecdf)}
        <div class="ov-rank-metric">${sr}</div>
        <div class="ov-rank-metric ${anyBest ? '' : 'muted'}">${bestStr}</div>
        <div class="ov-rank-metric ${anyWorst ? 'danger' : 'muted'}">${worstStr}</div>
      </div>`;
  });
  html += '</div>';
  const container = document.getElementById('overall-leaderboard-container');
  container.innerHTML = html;
  if (!container.dataset.sortBound) {
    container.addEventListener('click', e => {
      const btn = e.target.closest('.ov-rank-sort');
      if (!btn) return;
      const key = btn.dataset.sortKey;
      if (!key || key === overallSortKey) return;
      overallSortKey = key;
      renderOverall();
    });
    container.dataset.sortBound = '1';
  }
}

function _heatmapBg(sr) {
  // 0=red(hsl 0), 1=green(hsl 120), pastel for readable text overlay
  const h = Math.round(sr * 120);
  return `hsl(${h},58%,80%)`;
}

function _medalCls(i) {
  return i === 0 ? 'is-1' : i === 1 ? 'is-2' : i === 2 ? 'is-3' : '';
}

function _renderHeatmap(lb) {
  const cats = overallData.categories || [];

  let html = '<div class="ov-tbl-wrap"><table class="ov-tbl"><thead><tr>';
  html += '<th>Method</th>';
  cats.forEach(c => { html += `<th>${htmlesc(CAT_LABELS[c] || c)}</th>`; });
  html += '</tr></thead><tbody>';

  lb.forEach((row, i) => {
    html += `<tr class="${_medalCls(i)}">`;
    html += `<td>${htmlesc(row.method)}</td>`;
    cats.forEach(c => {
      const v = row.category_sr?.[c];
      if (v == null) {
        html += `<td><span class="ov-heatmap-pill null">—</span></td>`;
      } else {
        const bg = _heatmapBg(v);
        html += `<td><span class="ov-heatmap-pill" style="background:${bg};">${(v * 100).toFixed(0)}%</span></td>`;
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  document.getElementById('overall-heatmap-container').innerHTML = html;
}

function _renderRankProfile(lb) {
  const funcs = overallData.funcs || [];
  const fr    = overallData.func_ranks || {};   // {bf:{f:{m:r}}, evals:{f:{m:r}}}
  const nMeth = lb.length;

  const shortLabel = f => f.replace(/^[FC]\d+-/, '');

  const buildTbl = (indKey, indLabel, meanKey) => {
    const frInd = fr[indKey] || {};
    // Sort by this indicator's mean rank (independent of leaderboard's sort key)
    // so each table's top-3 reflects ITS indicator. Tiebreakers: other two indicators.
    const others = ['bf', 'evals', 'ecdf'].filter(k => k !== indKey);
    const tblLb = [...lb].sort((a, b) =>
      (a[meanKey] - b[meanKey])
      || (a[`mean_rank_${others[0]}`] - b[`mean_rank_${others[0]}`])
      || (a[`mean_rank_${others[1]}`] - b[`mean_rank_${others[1]}`])
    );
    let h = `<div class="rank-profile-label" style="font-size:11px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:var(--muted);margin:8px 0 4px;">${indLabel}</div>`;
    h += '<div class="ov-tbl-wrap"><table class="ov-tbl sticky-col"><thead><tr>';
    h += '<th>Method</th>';
    funcs.forEach(f => { h += `<th title="${htmlesc(f)}">${htmlesc(shortLabel(f))}</th>`; });
    h += '<th class="ov-mean-col">Mean</th>';
    h += '</tr></thead><tbody>';
    tblLb.forEach((row, ri) => {
      h += `<tr class="${_medalCls(ri)}">`;
      h += `<td>${htmlesc(row.method)}</td>`;
      funcs.forEach(f => {
        const rank = frInd[f]?.[row.method];
        let chipCls = 'null', txt = '—';
        if (rank != null) {
          txt = rank % 1 === 0 ? String(rank) : rank.toFixed(1);
          if (rank === 1)         chipCls = 'best';
          else if (rank >= nMeth) chipCls = 'worst';
          else                    chipCls = '';
        }
        h += `<td><span class="ov-rank-chip ${chipCls}">${txt}</span></td>`;
      });
      h += `<td class="ov-mean-col">${row[meanKey].toFixed(2)}</td>`;
      h += '</tr>';
    });
    h += '</tbody></table></div>';
    return h;
  };

  document.getElementById('overall-rank-table-container').innerHTML =
    buildTbl('bf',    'bf (median_best_f)',         'mean_rank_bf') +
    buildTbl('evals', 'Evals (succ-only median)',   'mean_rank_evals') +
    buildTbl('ecdf',  'ECDF AUC',                   'mean_rank_ecdf');
}

// ── Fullscreen overlay ────────────────────────────────────────────────────────
function openOverlay(src) {
  if (!src) return;
  currentMediaSrc = src;
  document.getElementById('overlay-img').src = src;
  document.getElementById('media-overlay').classList.add('open');
}
function closeOverlay() {
  document.getElementById('media-overlay').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeOverlay(); });

// ── Type selector events ─────────────────────────────────────────────────────
document.getElementById('media-selector').addEventListener('click', e => {
  const btn = e.target.closest('button[data-type]');
  if (!btn) return;
  document.querySelectorAll('#media-selector button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentType = btn.dataset.type;

  if (viewMode === 'function') {
    renderFunctionGrid();
  } else if (viewMode === 'method') {
    renderMethodGrid();
  } else if (viewMode === 'compare') {
    renderCompareMatrix();
  }
});

// ── Number formatting ─────────────────────────────────────────────────────────
function fmtNum(val) {
  const f = parseFloat(val);
  if (isNaN(f) || val === '' || val == null) return val ?? '—';
  if (Math.abs(f) === 0) return '0';
  return Math.abs(f) >= 0.001 && Math.abs(f) < 10000
    ? f.toPrecision(4) : f.toExponential(2);
}

function rankColors(values, higherBetter) {
  const nums = values.map(v => parseFloat(v));
  const valid = nums.filter(v => !isNaN(v));
  if (valid.length < 2) return nums.map(() => '');
  const sorted = [...valid].sort((a, b) => a - b);
  return nums.map(v => {
    if (isNaN(v)) return '';
    const pos  = sorted.indexOf(v);
    const rank = higherBetter ? sorted.length - 1 - pos : pos;
    const n    = sorted.length;
    if (rank === 0)                      return 'cell-best';
    if (rank <= Math.floor(n * 0.25))    return 'cell-good';
    if (rank <= Math.floor(n * 0.5))     return 'cell-mid';
    if (rank < n - 1)                    return 'cell-bad';
    return 'cell-worst';
  });
}

// ── Unified table ─────────────────────────────────────────────────────────────
async function buildUnifiedTable(func) {
  const container = document.getElementById('unified-table-container');
  container.innerHTML = '<p class="empty-state">Loading…</p>';

  const summaryRows = (DIMS_DATA[currentDim]?.summary || [])
    .filter(r => r.function === func);
  if (!summaryRows.length) {
    container.innerHTML = '<p class="empty-state">No data.</p>';
    return;
  }

  const res = await fetch(`/api/stats/${RUN_ID}/${currentDim}/${func}`);
  const { rows: statRows } = await res.json();

  const byMethod = {};
  (statRows || []).forEach(r => {
    (byMethod[r.method] = byMethod[r.method] || []).push(r);
  });

  const bfColors  = rankColors(summaryRows.map(r => r.mean_best_f), false);
  const sr2Colors = rankColors(summaryRows.map(r => parseFloat(r['sr_1e-2'])), true);
  const sr4Colors = rankColors(summaryRows.map(r => parseFloat(r['sr_1e-4'] ?? r.success_rate)), true);
  const evalsRaw  = summaryRows.map(r => r.evals_succ_med ?? r.ert);
  const evalsColors = rankColors(evalsRaw.map(v => v === 'inf' || v == null ? 'Infinity' : v), false);
  const orColors  = rankColors(summaryRows.map(r => r.mean_optima_rate), true);

  function fmtEvals(val) {
    if (val == null || val === '' || val === 'inf') return '—';
    const n = parseFloat(val);
    return isNaN(n) || !isFinite(n) ? '—' : Math.round(n).toLocaleString();
  }
  function fmtSR(val) {
    if (val == null || val === '') return '—';
    if (typeof val === 'string' && val.endsWith('%')) return val;
    const n = parseFloat(val);
    return isNaN(n) ? '—' : (n * 100).toFixed(0) + '%';
  }

  // Multi-threshold SR profile rendered as a 7-bar mini ECDF chart.
  const SR_TARGETS = ['sr_1e-1', 'sr_1e-2', 'sr_1e-3', 'sr_1e-4', 'sr_1e-5', 'sr_1e-7', 'sr_1e-10'];
  const SR_TARGET_LABELS = ['1e⁻¹', '1e⁻²', '1e⁻³', '1e⁻⁴', '1e⁻⁵', '1e⁻⁷', '1e⁻¹⁰'];

  // Parse "85%" / "0.85" / 0.85 → 0.85 (fraction in [0,1]); null on missing.
  function parseSRFraction(raw) {
    if (raw == null || raw === '') return null;
    if (typeof raw === 'string') {
      const trimmed = raw.trim();
      if (trimmed.endsWith('%')) {
        const n = parseFloat(trimmed.slice(0, -1));
        return isNaN(n) ? null : n / 100;
      }
      const n = parseFloat(trimmed);
      return isNaN(n) ? null : n;
    }
    return typeof raw === 'number' ? raw : null;
  }

  function fmtECDFProfile(sr) {
    // 7 vertical bars; height of filled portion ∝ SR. Width 12px, height 22px.
    const BAR_W = 12, GAP = 2, H = 22, PAD_X = 2;
    const totalW = PAD_X * 2 + 7 * BAR_W + 6 * GAP;
    const bars = SR_TARGETS.map((k, i) => {
      const v = parseSRFraction(sr[k]);
      const valid = v != null;
      const filled = valid ? Math.max(0, Math.min(1, v)) : 0;
      const fillH = filled * H;
      const fillY = H - fillH;
      const x = PAD_X + i * (BAR_W + GAP);
      // Color intensity by SR — same hue, darker when higher
      const fillCol = valid
        ? `rgba(79, 70, 229, ${(0.35 + 0.55 * filled).toFixed(3)})`
        : 'rgba(0,0,0,0.0)';
      const tip = valid
        ? `SR@${SR_TARGET_LABELS[i]} = ${(filled * 100).toFixed(0)}%`
        : `SR@${SR_TARGET_LABELS[i]} not available`;
      return `
        <g><title>${tip}</title>
          <rect x="${x}" y="0" width="${BAR_W}" height="${H}" rx="1.5" fill="#e5e7eb"/>
          ${valid && filled > 0
            ? `<rect x="${x}" y="${fillY.toFixed(2)}" width="${BAR_W}" height="${fillH.toFixed(2)}" rx="1.5" fill="${fillCol}"/>`
            : `<line x1="${x + BAR_W / 2}" y1="${H / 2}" x2="${x + BAR_W / 2}" y2="${H / 2}" stroke="#9ca3af"/>`}
        </g>`;
    }).join('');
    return `<svg class="ecdf-bars" viewBox="0 0 ${totalW} ${H}" width="${totalW}" height="${H}" aria-label="ECDF profile">${bars}</svg>`;
  }


  const COLS = [
    { label: 'Method / Seed', desc: 'Click ▶ to expand per-run details.' },
    { label: 'best_f',        desc: 'Mean of final best f(x) across all runs. Lower is better. BBOB functions: global minimum = 0.' },
    { label: 'ECDF profile',  desc: 'BBOB-style success rate at multiple precision targets, displayed as 7 stacked tiles (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-10). Hover for exact values. Loose targets on the left, tight on the right.' },
    { label: 'Evals (succ med)', desc: 'Median number of evaluations to reach the 1e-4 target across successful runs only. Failed runs are excluded (no penalty extrapolation). Read together with SR — a small median with low SR means only a few lucky runs hit the target. — means no successful run.' },
    { label: 'time (s)',      desc: 'Mean wall-clock time per run (seconds).' },
    { label: 'optima rate',   desc: 'Fraction of distinct global optima found per run (capture radius ε = 0.1 × span). N/A for single-optimum functions.' },
    { label: 'evals',         desc: 'Total function evaluations used in this run.' },
  ];

  const TH = (label, extra='') => `<th${extra ? ' '+extra : ''}>${label}</th>`;
  const legendItems = COLS.map(c =>
    `<span class="col-legend-key">${c.label}</span><span class="col-legend-desc">${c.desc}</span>`
  ).join('');

  let html = `
  <details class="col-legend">
    <summary>凡例<span class="col-legend-toggle">▶</span></summary>
    <div class="col-legend-body">${legendItems}</div>
  </details>
  <div class="table-wrap">
  <table class="unified-table">
    <thead><tr>
      ${TH('Method / Seed', 'style="text-align:left;min-width:150px;"')}
      ${TH('best_f')} ${TH('ECDF profile', 'style="min-width:158px;"')} ${TH('Evals (succ med)')} ${TH('time (s)')} ${TH('optima rate')} ${TH('evals')}
    </tr></thead>
    <tbody id="unified-tbody">`;

  summaryRows.forEach((sr, si) => {
    const method = sr.method;
    const runs   = byMethod[method] || [];
    html += `
      <tr class="method-row" data-method="${htmlesc(method)}" data-expanded="false">
        <td><div class="method-cell">
          <span class="expand-toggle">${runs.length ? '▶' : '·'}</span>
          ${method}
        </div></td>
        <td class="${bfColors[si]}">${fmtNum(sr.mean_best_f)}</td>
        <td class="ecdf-cell">${fmtECDFProfile(sr)}</td>
        <td class="${evalsColors[si]}">${fmtEvals(sr.evals_succ_med ?? sr.ert)}</td>
        <td>${fmtNum(sr.mean_time_s)}</td>
        <td class="${orColors[si]}">${fmtNum(sr.mean_optima_rate)}</td>
        <td style="color:var(--muted);">—</td>
      </tr>`;
    runs.forEach(run => {
      const ok  = run.success === 'True';
      const ors = parseFloat(run.optima_rate);
      html += `
        <tr class="run-row" data-method="${htmlesc(method)}" style="display:none;">
          <td class="run-cell">seed ${run.seed}</td>
          <td>${fmtNum(run.best_f)}</td>
          <td class="${ok ? 'cell-success' : 'cell-failure'}" style="text-align:center;">${ok ? '✓' : '✗'}</td>
          <td style="color:var(--muted);">—</td>
          <td>${fmtNum(run.time_s)}</td>
          <td>${isNaN(ors) ? '—' : ors.toFixed(2)}</td>
          <td>${run.n_evals ?? '—'}</td>
        </tr>`;
    });
  });

  html += '</tbody></table></div>';
  container.innerHTML = html;

  container.querySelector('#unified-tbody').addEventListener('click', e => {
    const row = e.target.closest('.method-row');
    if (!row) return;
    const method   = row.dataset.method;
    const expanded = row.dataset.expanded === 'true';
    container.querySelectorAll('.run-row').forEach(r => {
      if (r.dataset.method === method) r.style.display = expanded ? 'none' : '';
    });
    row.dataset.expanded = expanded ? 'false' : 'true';
    const toggle = row.querySelector('.expand-toggle');
    if (toggle) toggle.textContent = expanded ? '▶' : '▼';
  });
}

function htmlesc(s) {
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
}

// ── Overall Wilcoxon section (shown in 全体評価 view) ──────────────────────
// Renders an aggregate summary tile per baseline + per-function matrix.
function renderOverallWilcoxon() {
  const card = document.getElementById('overall-wilcoxon-card');
  const container = document.getElementById('overall-wilcoxon-container');
  if (!container || !card) return;
  const rows = DIMS_DATA[currentDim]?.wilcoxon || [];
  if (!rows.length) { card.style.display = 'none'; container.innerHTML = ''; return; }
  card.style.display = '';

  const ref = rows[0].reference;
  const funcs   = Array.from(new Set(rows.map(r => r.function)));
  const methods = Array.from(new Set(rows.map(r => r.method)));

  // Categorize a row: 'win' (ref better), 'loss' (other better), 'tie' (no sig).
  const classify = (r) => {
    const pLess = parseFloat(r.p_value_ref_better);
    const n     = parseInt(r.n) || 0;
    const tie_c = parseInt(r.tie_count) || 0;
    if (tie_c === n) return 'tie';
    if (!isNaN(pLess) && pLess < 0.05) return 'win';
    if (!isNaN(pLess) && pLess > 0.95) return 'loss';
    return 'tie';
  };

  // Per-baseline overall counts (#functions where ref wins / ties / loses)
  const overallTiles = methods.map(m => {
    let wins = 0, ties = 0, losses = 0;
    rows.filter(r => r.method === m).forEach(r => {
      const c = classify(r);
      if (c === 'win') wins++; else if (c === 'loss') losses++; else ties++;
    });
    return `
      <div class="vs-tile">
        <div class="vs-tile-label">vs ${htmlesc(m)}</div>
        <div class="vs-tile-counts">
          <span class="win">勝 ${wins}</span>
          <span class="tie">引 ${ties}</span>
          <span class="loss">負 ${losses}</span>
        </div>
      </div>`;
  }).join('');

  // Per-function matrix
  const headerCells = methods.map(m => `<th>vs ${htmlesc(m)}</th>`).join('');
  const bodyRows = funcs.map(f => {
    const cells = methods.map(m => {
      const row = rows.find(r => r.function === f && r.method === m);
      if (!row) return `<td><span class="ov-wc-chip is-tie"><span class="mark">—</span></span></td>`;
      const pLess = parseFloat(row.p_value_ref_better);
      const wc = parseInt(row.win_count) || 0;
      const tc = parseInt(row.tie_count) || 0;
      const n  = parseInt(row.n) || 0;
      const lc = n - wc - tc;
      const a12 = parseFloat(row.a12);
      const a12mag = row.a12_magnitude || '';
      let cls = 'is-tie', mark = '=';
      if (tc === n) { mark = '='; }
      else if (!isNaN(pLess) && pLess < 0.01) { cls = 'is-strong'; mark = '★'; }
      else if (!isNaN(pLess) && pLess < 0.05) { cls = 'is-mild';   mark = '✓'; }
      else if (!isNaN(pLess) && pLess > 0.95) { cls = 'is-loss';   mark = '✗'; }
      const pStr   = isNaN(pLess) ? '—' : pLess.toExponential(2);
      const a12Str = isNaN(a12)   ? '—' : a12.toFixed(2);
      const tip  = `${ref} vs ${m} on ${f}: ${wc} 勝 / ${tc} 引 / ${lc} 負, p(${ref} better) = ${pStr}, A12 = ${a12Str} (${a12mag})`;
      // a12 badge: emphasised when magnitude is medium/large
      const a12Cls = (a12mag === 'large' || a12mag === 'medium') ? 'a12-strong' : 'a12-weak';
      return `<td title="${tip}"><span class="ov-wc-chip ${cls}">
        <span class="mark">${mark}</span>
        <span class="count">${wc}-${tc}-${lc}</span>
        <span class="a12 ${a12Cls}">A=${a12Str}</span>
      </span></td>`;
    }).join('');
    return `<tr><td>${htmlesc(f)}</td>${cells}</tr>`;
  }).join('');

  container.innerHTML = `
    <div style="font-size:11.5px;color:var(--muted);margin:0 0 10px;line-height:1.55;">
      ★ p &lt; 0.01（高度に有意 ${htmlesc(ref)} が優位）／ ✓ p &lt; 0.05（有意）／ = 引分け or 非有意 ／ ✗ p &gt; 0.95（相手が優位）。
      セル 2 段目は <span style="font-family:var(--mono);">勝-引-負</span> の seed 数 ／ 3 段目は Vargha–Delaney A₁₂（&gt;0.5 で ${htmlesc(ref)} 優位、|A−0.5| ≤ .06 negligible / ≤ .14 small / ≤ .21 medium / &gt; .21 large）。
    </div>
    <div class="wilcoxon-summary-row">${overallTiles}</div>
    <details open style="margin-top:14px;">
      <summary class="overall-details-summary">関数 × 既存手法のマトリクス</summary>
      <div class="ov-tbl-wrap" style="margin-top:8px;">
        <table class="ov-tbl sticky-col">
          <thead><tr><th>関数</th>${headerCells}</tr></thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
    </details>`;
}

// ── Sidebar run kebab menu ────────────────────────────────────────────────────
function toggleSidebarMenu(runId, e) {
  e.preventDefault(); e.stopPropagation();
  const btn  = e.currentTarget;
  const menu = document.getElementById(`srmenu-${runId}`);
  const isOpen = menu.classList.contains('open');
  closeAllSidebarMenus();
  if (!isOpen) {
    menu.classList.add('open'); btn.classList.add('open');
    const r = btn.getBoundingClientRect();
    menu.style.top  = (r.bottom + 4) + 'px';
    menu.style.left = Math.max(4, r.right - menu.offsetWidth) + 'px';
  }
}
function closeAllSidebarMenus() {
  document.querySelectorAll('.sidebar-run-dropdown.open').forEach(m => m.classList.remove('open'));
  document.querySelectorAll('.sidebar-run-menu-btn.open').forEach(b => b.classList.remove('open'));
}
document.addEventListener('click', closeAllSidebarMenus);

// ── Rename result ─────────────────────────────────────────────────────────────
async function renameResult(runId) {
  closeAllSidebarMenus();
  const newName = await dlg.prompt('新しい名前を入力してください:', runId);
  if (!newName || newName === runId) return;
  const form = new FormData();
  form.append('new_name', newName);
  const res  = await fetch(`/api/results/${encodeURIComponent(runId)}/rename`, { method: 'POST', body: form });
  const data = await res.json();
  if (data.ok) {
    location.href = `/results/${encodeURIComponent(data.new_name)}`;
  } else {
    await dlg.alert(data.message || '名前変更に失敗しました');
  }
}

// ── Delete result ─────────────────────────────────────────────────────────────
async function deleteResult(runId) {
  closeAllSidebarMenus();
  if (!await dlg.confirm(`「${runId}」を削除しますか？\nこの操作は元に戻せません。`)) return;
  const res  = await fetch(`/api/results/${encodeURIComponent(runId)}`, { method: 'DELETE' });
  const data = await res.json();
  if (data.ok) {
    if (runId === RUN_ID) { location.href = '/'; return; }
    const li = [...document.querySelectorAll('.results-list li')]
      .find(el => el.querySelector(`a[href="/results/${runId}"]`));
    if (li) li.remove();
  } else {
    await dlg.alert(data.message || '削除に失敗しました');
  }
}

// ── Dim tabs ──────────────────────────────────────────────────────────────────
document.getElementById('dim-tabs').addEventListener('click', e => {
  const btn = e.target.closest('button[data-dim]');
  if (btn) switchDim(btn.dataset.dim);
});

// ── Init ──────────────────────────────────────────────────────────────────────
initSidebarRunHeader();
(function() {
  const { dim: hashDim } = _parseHash();
  if (hashDim && DIMS.includes(hashDim) && hashDim !== currentDim) {
    currentDim = hashDim;
    document.querySelectorAll('#dim-tabs button')
      .forEach(b => b.classList.toggle('active', b.dataset.dim === currentDim));
  }
})();

if (currentDim) {
  loadMediaIndex().then(() => {
    buildTypeSelector();
    buildFuncList();
    document.getElementById('media-grid-wrap').style.display = '';
  });
}

// ── Auto-sync polling (every 15 s) ────────────────────────────────────────────
let _lastFuncKey = JSON.stringify(DIMS_DATA[DIMS[0]]?.functions ?? []);
let _lastSummKey = '';

async function pollResultData() {
  try {
    const res = await fetch(`/api/result-data/${encodeURIComponent(RUN_ID)}`);
    if (!res.ok) return;
    const data = await res.json();

    const newFuncs = data.dims_data[currentDim]?.functions ?? [];
    const newSumm  = data.dims_data[currentDim]?.summary   ?? [];
    const funcKey  = JSON.stringify(newFuncs);
    const summKey  = JSON.stringify(newSumm.filter(r => r.function === currentFunc));

    if (funcKey !== _lastFuncKey) {
      _lastFuncKey = funcKey;
      DIMS_DATA = data.dims_data;
      await loadMediaIndex();
      buildTypeSelector();
      buildFuncList();
    }
    if (currentFunc && summKey !== _lastSummKey) {
      _lastSummKey = summKey;
      DIMS_DATA = data.dims_data;
      buildUnifiedTable(currentFunc);
    }
  } catch (_) {}
}
setInterval(pollResultData, 15000);
