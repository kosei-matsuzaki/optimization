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

// Shape-tag axes (mirrors TAG_AXES in core/benchmarks.py) — used to order and
// group the tag columns of the 形状タグ別 unified matrix under axis headers.
const TAG_AXES = [
  { axis: 'modality',     tags: ['unimodal', 'multimodal', 'multi-global'] },
  { axis: 'separability', tags: ['separable', 'non-separable'] },
  { axis: 'conditioning', tags: ['well-conditioned', 'moderate-cond', 'ill-conditioned'] },
  { axis: 'structure',    tags: ['global-structure', 'weak-structure'] },
  { axis: 'landscape',    tags: ['smooth', 'linear', 'asymmetric', 'plateau', 'bent-valley',
                                 'sharp-ridge', 'rugged', 'deceptive', 'boundary-optimum', 'needle'] },
  { axis: 'suite-shape',  tags: ['hybrid', 'composition'] },
];
const TAG_AXIS_OF = {};
TAG_AXES.forEach(a => a.tags.forEach(t => { TAG_AXIS_OF[t] = a.axis; }));
// Restrict TAG_AXES to a given (already scope-limited) tag set, preserving axis order.
function _tagColumns(tagList) {
  const set = new Set(tagList);
  return TAG_AXES
    .map(a => ({ axis: a.axis, tags: a.tags.filter(t => set.has(t)) }))
    .filter(a => a.tags.length);
}

let overallRaw = null;              // full /api/overall payload (all suites)
let overallData = null;             // payload for the currently selected suite scope
let overallScope = null;            // 'bbob' | 'custom' | 'cec' | 'all'
let overallView = 'ranking';        // active overall sub-view: ranking | detail | wilcoxon
let overallDetailKey = 'sr_deep';   // selected indicator for the unified detail view
let overallSortKey = 'sr_deep';     // leaderboard sort column (default: SR@1e-10)

// Benchmark suite of a function, by name prefix (F*=BBOB, C*=Custom, G*=CEC2022).
function suiteOf(name) {
  return ({ F: 'bbob', C: 'custom', G: 'cec' })[(name || '')[0]] || 'other';
}
const SCOPE_LABELS = {
  bbob:   'BBOB',
  custom: 'Custom',
  cec:    'CEC2022',
  other:  'その他',
  all:    '全体（混在）',
};

// Sortable leaderboard columns. dir = natural "best first" direction:
//   rank columns ascending (rank 1 = best), score columns descending (higher = best).
const OV_SORT_KEYS = {
  bf:      { field: 'mean_rank_bf',    dir: 'asc'  },
  evals:   { field: 'mean_rank_evals', dir: 'asc'  },
  sr_deep: { field: 'mean_sr_deep',    dir: 'desc' },
  sr:      { field: 'mean_sr',         dir: 'desc' },
  pr:      { field: 'mean_pr',         dir: 'desc' },
};

// Indicators for the unified detail view (カテゴリ別 + 関数別).
//   type 'score' → 0..1 value, higher better, shown as % heatmap.
//   type 'rank'  → Friedman rank (1..k), lower better, shown as rank chips.
const OV_DETAIL_INDICATORS = [
  { key: 'sr_deep', label: 'SR@1e-10', type: 'score', desc: '各関数の最高精度の成功率（主指標）· 赤=低・緑=高' },
  { key: 'sr',      label: 'SR@1e-4',  type: 'score', desc: '各関数の成功率（補助）· 赤=低・緑=高' },
  { key: 'pr',      label: 'PR@1e-4',  type: 'score', desc: '各関数の多解の最適点発見率 · 赤=低・緑=高' },
  { key: 'bf',      label: 'best_f',   type: 'rank',  desc: 'best_f（全 run 平均 mean_best_f）の Friedman ランク · 緑=1位・赤=最下位 · 同着は平均ランク' },
  { key: 'evals',   label: 'Evals',    type: 'rank',  desc: '成功 run のみの平均評価数の Friedman ランク · 緑=1位・赤=最下位' },
];

function _sortedLeaderboard() {
  if (!overallData?.leaderboard) return [];
  const spec = OV_SORT_KEYS[overallSortKey] || OV_SORT_KEYS.evals;
  const sign = spec.dir === 'asc' ? 1 : -1;
  // Sort by the chosen column (best first); tiebreak by Evals rank then SR@1e-10.
  return [...overallData.leaderboard].sort((a, b) => {
    const pa = a[spec.field] ?? 0, pb = b[spec.field] ?? 0;
    if (pa !== pb) return sign * (pa - pb);
    if (a.mean_rank_evals !== b.mean_rank_evals) return a.mean_rank_evals - b.mean_rank_evals;
    return b.mean_sr_deep - a.mean_sr_deep;
  });
}

// ── URL hash persistence: #dim2/F01-Sphere or #dim2/__overall__/detail ──
function _parseHash() {
  const h = location.hash.slice(1);
  if (!h) return { dim: null, func: null, view: null };
  const parts = h.split('/');
  if (parts.length < 2) return { dim: null, func: decodeURIComponent(parts[0]), view: null };
  const dim = parts[0];
  const rest = parts.slice(1).map(decodeURIComponent);
  if (rest[0] === '__overall__') return { dim, func: '__overall__', view: rest[1] || null };
  return { dim, func: rest.join('/'), view: null };
}
function _updateHash() {
  if (!currentDim || !currentFunc) return;
  const tail = currentFunc === '__overall__'
    ? `__overall__/${overallView || 'ranking'}`
    : encodeURIComponent(currentFunc);
  history.replaceState(null, '', `#${currentDim}/${tail}`);
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
    _clearOverallNav();
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
  overallRaw = null;
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

  const { func: hashFunc, view: hashView } = _parseHash();
  if (hashFunc === '__overall__') {
    selectOverall(hashView || overallView);
  } else if (hashFunc && functions.includes(hashFunc)) {
    selectFunc(hashFunc);
  } else {
    // Default landing: 全体評価（ランキング）
    selectOverall(overallView);
  }
}

function selectFunc(func) {
  setOverallMode(false);
  currentFunc = func;
  _clearOverallNav();
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

// The three overall sub-views map to one card each; the scope bar is shared.
const OVERALL_VIEWS = ['ranking', 'detail', 'wilcoxon'];
function _setOverallNav(view) {
  document.querySelectorAll('#overall-nav .ov-nav-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.ovview === view));
}
function _clearOverallNav() {
  document.querySelectorAll('#overall-nav .ov-nav-btn').forEach(b => b.classList.remove('active'));
}
// Show only the card for the active sub-view (scope bar stays visible for all).
function _applyOverallView() {
  const map = {
    ranking:  'overall-ranking-card',
    detail:   'overall-detail-card',
    wilcoxon: 'overall-wilcoxon-card',
  };
  OVERALL_VIEWS.forEach(v => {
    const el = document.getElementById(map[v]);
    if (el) el.style.display = (v === overallView) ? '' : 'none';
  });
}

async function selectOverall(view) {
  if (OVERALL_VIEWS.includes(view)) overallView = view;
  currentFunc = '__overall__';
  _updateHash();   // persist #dim/__overall__/<view> so a reload stays put
  document.querySelectorAll('#func-list a').forEach(a => a.classList.remove('active'));
  _setOverallNav(overallView);
  document.querySelectorAll('.card-func-badge').forEach(el => el.textContent = '');
  setOverallMode(true);
  if (!overallRaw) await fetchOverallData();
  renderOverall();
}

async function fetchOverallData() {
  try {
    const res = await fetch(`/api/overall/${encodeURIComponent(RUN_ID)}/${currentDim}`);
    overallRaw = res.ok ? await res.json() : null;
  } catch (_) { overallRaw = null; }
  _applyScope();
}

// Point overallData at the payload for the selected suite scope. Keeps the
// prior scope across dim/data refreshes when it still exists; otherwise falls
// back to the first available scope (BBOB when present).
function _applyScope() {
  const scopes = overallRaw?.scopes || [];
  if (!scopes.length) { overallData = overallRaw; return; }
  if (!scopes.includes(overallScope)) overallScope = scopes[0];
  overallData = overallRaw.by_suite?.[overallScope] || overallRaw;
}

function _renderScopeBar() {
  const el = document.getElementById('overall-scope-bar');
  if (!el) return;
  const scopes = overallRaw?.scopes || [];
  // Nothing to switch between (single suite) → hide the bar entirely.
  if (scopes.length <= 1) { el.innerHTML = ''; el.style.display = 'none'; return; }
  el.style.display = '';
  const nFuncOf = s => (overallRaw.by_suite?.[s]?.funcs || []).length;
  el.innerHTML = '<span class="ov-scope-label">評価範囲</span>' + scopes.map(s =>
    `<button class="ov-scope-btn ${s === overallScope ? 'active' : ''}${s === 'all' ? ' is-all' : ''}" `
    + `data-scope="${s}">${htmlesc(SCOPE_LABELS[s] || s)}`
    + `<span class="ov-scope-n">${nFuncOf(s)}</span></button>`
  ).join('');
  if (!el.dataset.bound) {
    el.addEventListener('click', e => {
      const btn = e.target.closest('button[data-scope]');
      if (!btn || btn.dataset.scope === overallScope) return;
      overallScope = btn.dataset.scope;
      _applyScope();
      renderOverall();
    });
    el.dataset.bound = '1';
  }
}

function renderOverall() {
  _renderScopeBar();
  _applyOverallView();
  if (!overallData || !overallData.leaderboard || !overallData.leaderboard.length) {
    document.getElementById('overall-leaderboard-container').innerHTML =
      '<p class="empty-state">データがありません。</p>';
    return;
  }
  const lb = _sortedLeaderboard();
  _renderLeaderboard(lb);
  _renderDetail(lb);
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

  // Friedman test stats — rendered as a small table (指標 × χ²_F / p / 判定 / CD)
  // rather than an inline run, which was ambiguous once it wrapped.
  const frRow = (ind, label) => {
    const s = fried[ind];
    if (!s) return '';
    const chi2 = s.chi2 == null ? '—' : s.chi2.toFixed(2);
    const pStr = s.p == null ? '—' : (s.p < 1e-4 ? s.p.toExponential(1) : s.p.toFixed(4));
    const cd   = s.cd == null ? '—' : s.cd.toFixed(2);
    const sig  = (s.p != null && s.p < 0.05);
    const verdict = s.p == null ? ''
      : `<span class="fr-chip ${sig ? 'sig' : 'nonsig'}">${sig ? '有意' : 'n.s.'}</span>`;
    return `<tr>
        <td class="fr-ind">${label}</td>
        <td>${chi2}</td>
        <td class="fr-p ${sig ? 'sig' : 'nonsig'}">${pStr}</td>
        <td class="fr-verdict">${verdict}</td>
        <td>${cd}</td>
      </tr>`;
  };
  // Friedman section — its own heading (matches 総合ランキング), placed below
  // the leaderboard in a separate container.
  const statsHtml = `
    <div class="card-header-row ov-subhead">
      <span class="card-title-label">Friedman 検定</span>
      <span class="card-subtitle">N = ${nFuncs} 関数 · k = ${nMethods} 手法 · 手法間の順位差の有意性</span>
    </div>
    <table class="fr-tbl">
      <thead><tr><th>指標</th><th>χ²_F</th><th>p 値</th><th>判定</th><th>CD₀.₀₅</th></tr></thead>
      <tbody>${frRow('bf', 'best_f')}${frRow('evals', 'Evals')}</tbody>
    </table>
    <div class="fr-note">
      χ²_F・p 値 = 手法間の順位差が全体として有意か（<strong>p &lt; 0.05 で有意差あり</strong>）。
      CD₀.₀₅ = Nemenyi 臨界差 — 2 手法の平均ランク差がこの値以上なら、その 2 手法は有意に異なる。
    </div>`;

  const sortHdr = (key, label, title) => {
    const isActive = overallSortKey === key;
    const arrow = isActive ? (OV_SORT_KEYS[key].dir === 'asc' ? '▲' : '▼') : '↕';
    return `<div><span class="ov-rank-sort ${isActive ? 'active' : ''}" data-sort-key="${key}" title="${title} · クリックでソート">${label}<span class="sort-arrow">${arrow}</span></span></div>`;
  };
  let html = `
    <div class="ov-rank-header">
      <div>#</div><div>Method</div>
      ${sortHdr('bf', 'Mean Rank (best_f)', 'best_f（全 run 平均 mean_best_f）の Friedman 平均ランク（低い＝優）')}
      ${sortHdr('evals', 'Mean Rank (Evals)', '成功 run のみの平均評価数の Friedman 平均ランク（低い＝優）')}
      ${sortHdr('sr_deep', 'SR@1e-10', 'SR@1e-10（最高精度・主指標）')}
      ${sortHdr('sr', 'SR@1e-4', 'SR@1e-4（補助）')}
      ${sortHdr('pr', 'PR@1e-4', 'PR@1e-4（多解の最適点発見率）')}
      <div>#Best (bf/Evals)</div><div>#Worst (bf/Evals)</div>
    </div>
    <div class="ov-ranking">`;
  lb.forEach((row, i) => {
    const medalCls = i < 3 ? `is-${i + 1}` : '';
    const fillBf    = (fillFor(row.mean_rank_bf)    * 100).toFixed(1);
    const fillEvals = (fillFor(row.mean_rank_evals) * 100).toFixed(1);
    const srDeep = (row.mean_sr_deep * 100).toFixed(1) + '%';
    const sr     = (row.mean_sr * 100).toFixed(1) + '%';
    const pr     = ((row.mean_pr ?? 0) * 100).toFixed(1) + '%';
    const bestStr  = `${row.n_best_bf}/${row.n_best_evals}`;
    const worstStr = `${row.n_worst_bf}/${row.n_worst_evals}`;
    const anyBest  = row.n_best_bf > 0 || row.n_best_evals > 0;
    const anyWorst = row.n_worst_bf > 0 || row.n_worst_evals > 0;
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
        <div class="ov-rank-metric primary" title="SR@1e-10">${srDeep}</div>
        <div class="ov-rank-metric" title="SR@1e-4">${sr}</div>
        <div class="ov-rank-metric ${row.mean_pr ? '' : 'muted'}" title="PR@1e-4">${pr}</div>
        <div class="ov-rank-metric ${anyBest ? '' : 'muted'}">${bestStr}</div>
        <div class="ov-rank-metric ${anyWorst ? 'danger' : 'muted'}">${worstStr}</div>
      </div>`;
  });
  html += '</div>';
  const container = document.getElementById('overall-leaderboard-container');
  container.innerHTML = html;
  const friedmanEl = document.getElementById('overall-friedman-container');
  if (friedmanEl) friedmanEl.innerHTML = statsHtml;
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

// ── Unified detail view (category + per-function, switchable by indicator) ────
function _renderDetailSelector() {
  const el = document.getElementById('overall-detail-selector');
  el.innerHTML = OV_DETAIL_INDICATORS.map(ind =>
    `<button class="ov-detail-btn ${ind.key === overallDetailKey ? 'active' : ''}" `
    + `data-detail-key="${ind.key}" data-type="${ind.type}" title="${htmlesc(ind.desc)}">`
    + `${ind.label}</button>`
  ).join('');
  if (!el.dataset.bound) {
    el.addEventListener('click', e => {
      const btn = e.target.closest('button[data-detail-key]');
      if (!btn) return;
      const key = btn.dataset.detailKey;
      if (!key || key === overallDetailKey) return;
      overallDetailKey = key;
      renderOverall();
    });
    el.dataset.bound = '1';
  }
}

function _renderDetail(lb) {
  const ind = OV_DETAIL_INDICATORS.find(x => x.key === overallDetailKey) || OV_DETAIL_INDICATORS[0];
  _renderDetailSelector();
  document.getElementById('overall-detail-subtitle').textContent =
    `${ind.label} · ${ind.desc} · 手法は${ind.type === 'rank' ? '平均ランク（低い＝優）' : '平均値（高い＝優）'}順`;

  const funcs = overallData.funcs || [];
  const cats  = overallData.categories || [];
  const fcat  = overallData.func_categories || {};
  const nMeth = lb.length;
  const shortLabel = f => f.replace(/^[FC]\d+-/, '');

  // Per-(func, method) value for the selected indicator (rank vs raw score).
  const valOf = ind.type === 'rank'
    ? (f, m) => overallData.func_ranks?.[ind.key]?.[f]?.[m]
    : (f, m) => overallData.func_scores?.[ind.key]?.[f]?.[m];

  const meanOver = (m, fs) => {
    const vals = fs.map(f => valOf(f, m)).filter(v => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const meanAll = m => {
    const v = meanOver(m, funcs);
    return v == null ? (ind.type === 'rank' ? Infinity : -Infinity) : v;
  };

  // Method order: rank → ascending mean rank; score → descending mean score.
  const ordered = [...lb].sort((a, b) =>
    ind.type === 'rank' ? meanAll(a.method) - meanAll(b.method)
                        : meanAll(b.method) - meanAll(a.method));

  const scoreCell = v => v == null
    ? '<span class="ov-heatmap-pill null">—</span>'
    : `<span class="ov-heatmap-pill" style="background:${_heatmapBg(v)};">${(v * 100).toFixed(0)}%</span>`;
  const rankCell = v => {
    let chipCls = 'null', txt = '—';
    if (v != null) {
      txt = v % 1 === 0 ? String(v) : v.toFixed(1);
      chipCls = v === 1 ? 'best' : (v >= nMeth ? 'worst' : '');
    }
    return `<span class="ov-rank-chip ${chipCls}">${txt}</span>`;
  };
  const cell    = ind.type === 'rank' ? rankCell : scoreCell;
  const fmtMean = v => (v == null || !isFinite(v)) ? '—'
    : (ind.type === 'rank' ? v.toFixed(2) : (v * 100).toFixed(0) + '%');

  // ── カテゴリ別 ──
  let ch = '<div class="ov-tbl-wrap"><table class="ov-tbl"><thead><tr><th>Method</th>';
  cats.forEach(c => { ch += `<th>${htmlesc(CAT_LABELS[c] || c)}</th>`; });
  ch += '<th class="ov-mean-col">All</th></tr></thead><tbody>';
  ordered.forEach((row, i) => {
    ch += `<tr class="${_medalCls(i)}"><td>${htmlesc(row.method)}</td>`;
    cats.forEach(c => {
      const fs = funcs.filter(f => fcat[f] === c);
      ch += `<td>${cell(meanOver(row.method, fs))}</td>`;
    });
    ch += `<td class="ov-mean-col">${fmtMean(meanOver(row.method, funcs))}</td></tr>`;
  });
  ch += '</tbody></table></div>';
  document.getElementById('overall-detail-category').innerHTML = ch;

  // ── 形状タグ別 統合ビュー ──
  // One matrix whose columns are shape tags grouped by axis. Two stacked
  // sections share the columns: (1) each method's aggregate of the selected
  // indicator over the functions carrying the tag — "how does each method do on
  // this shape?"; (2) the function → tag correspondence dots — "which functions
  // define this tag?". A function contributes to every tag it carries (columns
  // overlap by design; orthogonal shape axes, not a partition).
  const ftags = overallData.func_tags || {};
  const axisCols = _tagColumns(overallData.all_tags || []);
  const flatTags = axisCols.flatMap(a => a.tags.map(t => ({ t, axis: a.axis })));
  const funcsWith = t => funcs.filter(f => (ftags[f] || []).includes(t));
  const nCols = flatTags.length + 2;   // row-header + tags + All

  let th = '<div class="ov-tbl-wrap"><table class="ov-tbl sticky-col ovm-matrix">';
  th += '<thead><tr><th class="ovm-corner" rowspan="2">手法 / 関数</th>';
  axisCols.forEach(a => {
    th += `<th class="ovm-axis tax-${a.axis}" colspan="${a.tags.length}">${htmlesc(a.axis)}</th>`;
  });
  th += '<th class="ov-mean-col" rowspan="2">All</th></tr><tr>';
  flatTags.forEach(({ t, axis }) => {
    const n = funcsWith(t).length;
    th += `<th class="ovm-tag tax-${axis}" title="${htmlesc(t)}（${n} 関数）">`
        + `<span class="ovm-tag-label">${htmlesc(t)}</span><span class="ovm-tag-n">${n}</span></th>`;
  });
  th += '</tr></thead><tbody>';

  // Section 1: per-method aggregate of the selected indicator, per tag.
  th += `<tr class="ovm-sect"><td colspan="${nCols}">手法別集計 — ${htmlesc(ind.label)}（各タグを持つ関数のみで${ind.type === 'rank' ? '平均ランク' : '平均'}）</td></tr>`;
  ordered.forEach((row, i) => {
    th += `<tr class="${_medalCls(i)}"><td class="ovm-rowh">${htmlesc(row.method)}</td>`;
    flatTags.forEach(({ t }) => { th += `<td>${cell(meanOver(row.method, funcsWith(t)))}</td>`; });
    th += `<td class="ov-mean-col">${fmtMean(meanOver(row.method, funcs))}</td></tr>`;
  });

  // Section 2: function → tag correspondence (which functions carry each tag).
  th += `<tr class="ovm-sect"><td colspan="${nCols}">関数のタグ対応（● = そのタグを持つ）</td></tr>`;
  funcs.forEach(f => {
    const set = new Set(ftags[f] || []);
    const m = f.match(/^([A-Za-z]+\d+)-(.*)$/);
    const num = m ? m[1] : '';
    const name = m ? m[2] : f;
    th += `<tr class="ovm-frow"><td class="ovm-rowh ovm-fn" title="${htmlesc(f)}">`
        + `<a class="ovm-fn-link" href="#" onclick="selectFunc('${htmlesc(f)}');return false;" title="${htmlesc(f)} を開く">`
        + (num ? `<span class="ovm-fn-num">${htmlesc(num)}</span>` : '')
        + `<span class="ovm-fn-name">${htmlesc(name)}</span></a>`
        + `<span class="ovm-fn-cat">${htmlesc(CAT_LABELS[fcat[f]] || fcat[f] || '')}</span></td>`;
    flatTags.forEach(({ t, axis }) => {
      th += set.has(t)
        ? `<td class="ovm-cell tax-${axis} on" title="${htmlesc(shortLabel(f) + ' · ' + t)}"><span class="ovm-dot"></span></td>`
        : '<td class="ovm-cell off"></td>';
    });
    th += `<td class="ov-mean-col ovm-tagcount">${(ftags[f] || []).length}</td></tr>`;
  });
  th += '</tbody></table></div>';
  document.getElementById('overall-detail-tags').innerHTML = th;

  // ── 関数別 ──
  let fh = '<div class="ov-tbl-wrap"><table class="ov-tbl sticky-col"><thead><tr><th>Method</th>';
  funcs.forEach(f => {
    const tt = (ftags[f] || []).join(', ');
    fh += `<th title="${htmlesc(f + (tt ? '  ·  ' + tt : ''))}">${htmlesc(shortLabel(f))}</th>`;
  });
  fh += '<th class="ov-mean-col">Mean</th></tr></thead><tbody>';
  ordered.forEach((row, i) => {
    fh += `<tr class="${_medalCls(i)}"><td>${htmlesc(row.method)}</td>`;
    funcs.forEach(f => { fh += `<td>${cell(valOf(f, row.method))}</td>`; });
    fh += `<td class="ov-mean-col">${fmtMean(meanOver(row.method, funcs))}</td></tr>`;
  });
  fh += '</tbody></table></div>';
  document.getElementById('overall-detail-function').innerHTML = fh;
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
  const evalsRaw  = summaryRows.map(r => r.evals_succ_mean ?? r.evals_succ_med ?? r.ert);
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

  // Per-precision success-rate heatmap: one colored cell per target precision.
  // Loose targets on the left (1e⁻¹) → tight on the right (1e⁻¹⁰, the primary metric).
  const SR_TARGETS = ['sr_1e-4', 'sr_1e-7', 'sr_1e-10'];
  const SR_TARGET_LABELS = ['1e⁻⁴', '1e⁻⁷', '1e⁻¹⁰'];
  const SR_TARGET_VALS = [1e-4, 1e-7, 1e-10];     // f(x) 閾値（f_opt=0 前提）

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

  // Sequential green heatmap: 0% → pale, 100% → deep green. Number is always
  // printed so the exact value is readable regardless of color.
  function srHeatColor(frac) {
    const f = Math.max(0, Math.min(1, frac));
    const L = 97 - 63 * f;              // lightness 97 → 34
    const S = 32 + 36 * f;              // saturation 32 → 68
    const bg = `hsl(150, ${S.toFixed(0)}%, ${L.toFixed(0)}%)`;
    const fg = L < 62 ? '#ffffff' : '#0f3d24';
    return { bg, fg };
  }

  // Emit the 7 <td> heatmap cells for one method's SR profile.
  function fmtSRHeatCells(sr) {
    return SR_TARGETS.map((k, i) => {
      const v = parseSRFraction(sr[k]);
      const cls = 'sr-heat';
      if (v == null) {
        return `<td class="${cls} sr-heat-na" title="SR@${SR_TARGET_LABELS[i]} 未計測">—</td>`;
      }
      const frac = Math.max(0, Math.min(1, v));
      const { bg, fg } = srHeatColor(frac);
      const pct = (frac * 100).toFixed(0);
      return `<td class="${cls}" style="background:${bg};color:${fg};"`
        + ` title="SR@${SR_TARGET_LABELS[i]} = ${pct}%">${pct}<span class="sr-heat-pct">%</span></td>`;
    }).join('');
  }

  // Per-seed ✓/✗ cells: did this single run reach each target precision?
  // f_opt = 0 for BBOB/Custom, so "reached" ⟺ final best_f ≤ target.
  function fmtRunReachCells(run) {
    const bf = parseFloat(run.best_f);
    return SR_TARGET_VALS.map((t, i) => {
      const cls = 'sr-reach';
      if (isNaN(bf)) return `<td class="${cls} sr-heat-na">—</td>`;
      const hit = bf <= t;
      return `<td class="${cls} ${hit ? 'sr-reach-hit' : 'sr-reach-miss'}"`
        + ` title="1 run が SR@${SR_TARGET_LABELS[i]} ${hit ? '到達' : '未到達'} (best_f=${run.best_f})">`
        + `${hit ? '✓' : '✗'}</td>`;
    }).join('');
  }


  const COLS = [
    { label: 'Method / Seed', desc: 'Click ▶ to expand per-run details.' },
    { label: 'best_f',        desc: 'Mean of final best f(x) across all runs. Lower is better. BBOB functions: global minimum = 0.' },
    { label: 'SR@target', desc: '各精度目標(1e⁻⁴ / 1e⁻⁷ / 1e⁻¹⁰)に到達した run の割合(%)。左=緩い→右=厳しい。セルの色は成功率(濃い緑=高)、数値は正確な%。1e⁻¹⁰が主指標。行を展開すると各 seed が目標ごとに ✓(到達)/✗(未到達) で表示される。' },
    { label: 'Evals (succ mean)', desc: 'Mean number of evaluations to reach the 1e-4 target across successful runs only. Failed runs are excluded (no penalty extrapolation). Taken over successful runs (small spread, outliers unlikely), so the mean is used rather than the median. Read together with SR. — means no successful run.' },
    { label: 'time (s)',      desc: 'Mean wall-clock time per run (seconds).' },
    { label: 'optima rate',   desc: 'Fraction of distinct global optima found per run (capture radius ε = 0.1 × span). N/A for single-optimum functions.' },
    { label: 'evals',         desc: 'Total function evaluations used in this run.' },
  ];

  const TH = (label, extra='') => `<th${extra ? ' '+extra : ''}>${label}</th>`;
  const legendItems = COLS.map(c =>
    `<span class="col-legend-key">${c.label}</span><span class="col-legend-desc">${c.desc}</span>`
  ).join('');

  // Sub-header row: one column per precision target (1e⁻¹ … 1e⁻¹⁰).
  const srSubHead = SR_TARGET_LABELS.map(lbl =>
    `<th class="sr-head">${lbl}</th>`
  ).join('');

  let html = `
  <details class="col-legend">
    <summary>凡例<span class="col-legend-toggle">▶</span></summary>
    <div class="col-legend-body">${legendItems}</div>
  </details>
  <div class="table-wrap">
  <table class="unified-table">
    <thead>
      <tr>
        <th rowspan="2" style="text-align:left;min-width:150px;">Method / Seed</th>
        <th rowspan="2">best_f</th>
        <th colspan="${SR_TARGETS.length}" class="sr-group-head">SR@target</th>
        <th rowspan="2">Evals (succ mean)</th>
        <th rowspan="2">time (s)</th>
        <th rowspan="2">optima rate</th>
        <th rowspan="2">evals</th>
      </tr>
      <tr>${srSubHead}</tr>
    </thead>
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
        ${fmtSRHeatCells(sr)}
        <td class="${evalsColors[si]}">${fmtEvals(sr.evals_succ_mean ?? sr.evals_succ_med ?? sr.ert)}</td>
        <td>${fmtNum(sr.mean_time_s)}</td>
        <td class="${orColors[si]}">${fmtNum(sr.mean_optima_rate)}</td>
        <td style="color:var(--muted);">—</td>
      </tr>`;
    runs.forEach(run => {
      const ors = parseFloat(run.optima_rate);
      html += `
        <tr class="run-row" data-method="${htmlesc(method)}" style="display:none;">
          <td class="run-cell">seed ${run.seed}</td>
          <td>${fmtNum(run.best_f)}</td>
          ${fmtRunReachCells(run)}
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
  let rows = DIMS_DATA[currentDim]?.wilcoxon || [];
  // Restrict the pairwise matrix to the selected suite scope so BBOB / Custom
  // win-tie-loss counts are not mixed (the "all" scope keeps every function).
  if (overallScope && overallScope !== 'all') {
    rows = rows.filter(r => suiteOf(r.function) === overallScope);
  }
  // Card visibility is owned by _applyOverallView (the 統計的優位差 sub-tab);
  // here we only fill content, showing an empty state when the scope has no data.
  if (!rows.length) {
    container.innerHTML = '<p class="empty-state">この評価範囲には Wilcoxon データがありません。</p>';
    return;
  }

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
      overallRaw = null;    // invalidate so the overall view refreshes in place
      overallData = null;
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
