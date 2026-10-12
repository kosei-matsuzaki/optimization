const _pd    = JSON.parse(document.getElementById('page-data').textContent);
const RUN_ID = _pd.run_id;
const DIMS   = _pd.dims;
let   DIMS_DATA = _pd.dims_data;
const ALL_RESULTS_META = _pd.all_results_meta || {};

let currentDim    = DIMS[0] || null;
let currentFunc   = null;
// ?view=viz_search (or any figure type) opens that tab; default: convergence
let currentType   = new URLSearchParams(location.search).get('view') || 'viz_curves';
let currentMediaSrc = null;
let mediaIndex    = null;  // loaded from /api/media-index

// Compare mode selections

// ── ALL_TYPE_LABELS ──────────────────────────────────────────────────────────
// Interactive views drawn by static/viz.js from data (function view only);
// the image types below are matplotlib files from runs made with --viz.
const VIZ_TYPES = ['viz_curves', 'viz_search', 'viz_internals'];
const TYPE_LABELS = {
  'viz_curves':         '収束',
  'viz_search':         '探索の様子',
  'viz_internals':      'MC-ESO の内部',
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
  cec:     { field: 'mean_rank_cec',   dir: 'asc'  },
  sr_deep: { field: 'mean_sr_deep',    dir: 'desc' },
  sr:      { field: 'mean_sr',         dir: 'desc' },
  pr:      { field: 'mean_pr',         dir: 'desc' },
};

// Indicators for the unified detail view (カテゴリ別 + 関数別).
//   type 'score' → 0..1 value, higher better, shown as % heatmap.
//   type 'rank'  → Friedman rank (1..k), lower better, shown as rank chips.
const OV_DETAIL_INDICATORS = [
  { key: 'sr_deep', label: 'SR@1e-10', type: 'score', desc: '各関数の最高精度の成功率（主指標）。赤に寄るほど低く、緑に寄るほど高い' },
  { key: 'sr',      label: 'SR@1e-4',  type: 'score', desc: '各関数の成功率（補助）。赤に寄るほど低く、緑に寄るほど高い' },
  { key: 'pr',      label: 'PR@1e-4（履歴）',  type: 'score', desc: 'PR@1e-4（履歴ベース）: 既知の大域最適 K 点のうち、評価履歴に半径内かつ f≤1e-4 の点がある割合（summary.csv の pr_1e-4）。BBOB は K=1 なので SR@1e-4 とほぼ同じ値になる。CEC2013 報告集合の cec_pr_* とは別物。赤に寄るほど低く、緑に寄るほど高い' },
  { key: 'bf',      label: 'best_f',   type: 'rank',  desc: 'best_f（全 run 平均 mean_best_f）の順位。緑=1位、赤=最下位、同着は上の順位をそろって取る（1, 1, 3, …）' },
  { key: 'evals',   label: 'Evals',    type: 'rank',  desc: '成功 run のみの平均評価数の Friedman ランク。緑=1位、赤=最下位' },
  { key: 'cec',     label: 'CEC 順位', type: 'rank',  desc: 'CEC の競技の慣例: run ごとの誤差 f − f*（1e-8 未満は 0）の平均で順位をつける。緑=1位、赤=最下位、同着は上の順位をそろって取る（1, 1, 3, …）' },
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
    if (meta.n_runs    != null) runInfo.push(`n=${meta.n_runs}`);
    if (meta.max_evals != null) runInfo.push(`${Number(meta.max_evals).toLocaleString()} evals`);
    if (runInfo.length) lines.push({ cls: '', text: runInfo.join(' / ') });
    if (meta.commit)    lines.push({ cls: 'srm-small', text: meta.commit });
    // Where it was computed: results only reproduce exactly on the same environment
    if (meta.env) lines.push({ cls: 'srm-small srm-env', text: `${meta.env.os} / ${meta.env.cpu} / ${meta.env.blas}` });
    else lines.push({ cls: 'srm-small srm-env', text: '計算環境の記録なし（2026-10-09 より前）' });
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
  { label: 'Separable',      color: '#3d6f99', ids: ['f01','f02','f03','f04','f05'] },
  { label: 'Moderate',       color: '#4f8a5b', ids: ['f06','f07','f08','f09'] },
  { label: 'High Cond.',     color: '#b07e24', ids: ['f10','f11','f12','f13','f14'] },
  { label: 'Multi-modal',    color: '#7a5c99', ids: ['f15','f16','f17','f18','f19'] },
  { label: 'Weak Structure', color: '#a3473a', ids: ['f20','f21','f22','f23','f24'] },
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

  // Image tabs only for files this dimension has (landscapes are 2D only, and
  // runs without --viz have none); the interactive views need no files.
  const imgTypes = TYPE_ORDER.filter(t => (mediaIndex?.types || []).includes(t));
  const methodsHere = _summaryMethods();
  const vizTypes = VIZ_TYPES.filter(t => t !== 'viz_internals' || methodsHere.some(m => m.startsWith('MC-ESO')));
  const types = [...vizTypes, ...imgTypes];
  if (!types.length) {
    sel.innerHTML = '<p class="empty-state">この run には画像がありません。収束・探索の様子・MC-ESO の内部は「関数」表示で見られます。</p>';
    return;
  }

  if (!types.includes(currentType)) currentType = types[0];

  types.forEach(t => {
    if (vizTypes.length && t === imgTypes[0]) {
      const sep = document.createElement('span');
      sep.className = 'vs-sep';
      sep.textContent = '画像';
      sel.appendChild(sep);
    }
    const btn = document.createElement('button');
    btn.className = 'vs-btn' + (t === currentType ? ' active' : '');
    btn.dataset.type = t;
    btn.textContent = TYPE_LABELS[t] || t;
    sel.appendChild(btn);
  });
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
  });
}

// ── Load media index from API ────────────────────────────────────────────────
async function loadMediaIndex() {
  try {
    const res = await fetch(`/api/media-index/${encodeURIComponent(RUN_ID)}/${currentDim}`);
    if (res.ok) {
      mediaIndex = await res.json();
      mediaIndex.has = new Set((mediaIndex.files || [])
        .map(f => `${f.func}|${f.method ?? ''}|${f.type}`));
      // Proposed method first (then its old default), the rest alphabetically
      const rank = m => m === 'MC-ESO' ? 0 : m.startsWith('MC-ESO') ? 1 : 2;
      mediaIndex.methods = (mediaIndex.methods || [])
        .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    }
  } catch (_) {
    mediaIndex = null;
  }
}
// Is there a file for this (function, method, type)? method = null for landscape / convergence.
function hasMedia(func, method, type) {
  return !!mediaIndex?.has?.has(`${func}|${method ?? ''}|${type}`);
}
const NO_FIGURE = '<p class="empty-state">この組み合わせの図はありません。</p>';

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



// ── Render: Function mode grid (all methods for current function) ─────────────
// Methods of the current dimension, in summary order.
function _summaryMethods() {
  const rows = DIMS_DATA[currentDim]?.summary || [];
  return [...new Set(rows.map(r => r.method))];
}

// One container per interactive view, so each keeps its own selections.
function renderViz() {
  const root = document.getElementById('viz-root');
  root.style.display = '';
  document.getElementById('media-grid-wrap').style.display = 'none';
  VIZ_TYPES.forEach(t => {
    let box = root.querySelector(`[data-viz="${t}"]`);
    if (!box) { box = document.createElement('div'); box.dataset.viz = t; root.appendChild(box); }
    box.style.display = t === currentType ? '' : 'none';
  });
  const box = root.querySelector(`[data-viz="${currentType}"]`);
  const key = `${currentDim}|${currentFunc}`;
  if (box.dataset.key === key) return;        // already drawn for this function
  box.dataset.key = key;
  const ctx = { runId: RUN_ID, dim: currentDim, func: currentFunc, methods: _summaryMethods() };
  const fn = { viz_curves: Viz.curves, viz_search: Viz.search, viz_internals: Viz.internals }[currentType];
  fn(box, ctx);
}

function renderFunctionGrid() {
  const grid = document.getElementById('media-grid');
  grid.innerHTML = '';
  grid.classList.remove('solo');
  if (currentFunc && currentFunc !== '__overall__' && currentType.startsWith('viz_')) {
    renderViz();
    return;
  }
  document.getElementById('viz-root').style.display = 'none';
  document.getElementById('media-grid-wrap').style.display = '';
  if (!currentFunc || !mediaIndex || !(mediaIndex.types || []).length) return;

  // Function-level types (landscape/convergence): single full-width cell
  if (FUNC_LEVEL_TYPES.has(currentType)) {
    if (!hasMedia(currentFunc, null, currentType)) { grid.innerHTML = NO_FIGURE; return; }
    grid.classList.add('solo');
    const url = mediaUrl(currentFunc, null, currentType);
    grid.appendChild(makeGridCell(currentFunc, null, currentType, url, currentFunc));
    return;
  }

  // Only methods that have this figure (e.g. outbreak dynamics exist for MC-ESO only)
  const methods = (mediaIndex.methods || []).filter(m => hasMedia(currentFunc, m, currentType));
  if (!methods.length) { grid.innerHTML = NO_FIGURE; return; }

  methods.forEach(method => {
    const url = mediaUrl(currentFunc, method, currentType);
    grid.appendChild(makeGridCell(currentFunc, method, currentType, url, method));
  });
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

  // Friedman / Nemenyi, read out per indicator in plain words: (1) are the
  // rank differences among all methods more than chance, (2) which methods
  // differ from the proposed method by at least the critical difference.
  const ref0 = refMethod();
  const IND_LABELS = { bf: '平均順位 best_f', evals: '平均順位 Evals', cec: 'CEC 順位' };
  const frBlock = ind => {
    const s = fried[ind];
    if (!s) return '';
    // Nemenyi's critical difference is defined on average-tie ranks.
    const key = lb[0] && lb[0][`mean_avgrank_${ind}`] != null ? `mean_avgrank_${ind}` : `mean_rank_${ind}`;
    const refRow = lb.find(r => r.method === ref0);
    const best = [...lb].sort((x, y) => x[key] - y[key])[0];
    const pStr = s.p == null ? '—' : (s.p < 1e-4 ? s.p.toExponential(1) : s.p.toFixed(4));
    const sig = s.p != null && s.p < 0.05;
    let overall = s.p == null ? '検定できない（関数か手法が少なすぎる）'
      : sig ? `手法による順位の違いは偶然では説明できない（p = ${pStr}）`
            : `手法による順位の違いは偶然の範囲（p = ${pStr}）。以下の差も確かではない`;
    let refLine = '';
    if (refRow && s.cd != null && sig) {
      const cd = s.cd;
      const better = lb.filter(r => r.method !== ref0 && refRow[key] - r[key] >= cd).map(r => r.method);
      const worse  = lb.filter(r => r.method !== ref0 && r[key] - refRow[key] >= cd).length;
      const gap = refRow[key] - best[key];
      const lead = best.method === ref0
        ? `${htmlesc(ref0)} がこの指標で 1 位（${refRow[key].toFixed(2)}）`
        : `${htmlesc(ref0)} ${refRow[key].toFixed(2)} と 1 位 ${htmlesc(best.method)} ${best[key].toFixed(2)} の差 ${gap.toFixed(2)} は`
          + (gap >= cd ? `臨界差 ${cd.toFixed(2)} 以上 ＝ <strong>有意に負けている</strong>` : `臨界差 ${cd.toFixed(2)} 未満 ＝ <strong>有意差なし</strong>`);
      refLine = `<div class="fr-line">${lead}。${htmlesc(ref0)} より有意に上: ${better.length ? better.map(htmlesc).join('、') : 'なし'}／有意に下: ${worse} 手法</div>`;
    }
    return `<div class="fr-item"><div class="fr-ind">${IND_LABELS[ind]}<span class="fr-chip ${sig ? 'sig' : 'nonsig'}">${sig ? '差あり' : '差なし'}</span></div>
      <div class="fr-line">${overall}</div>${refLine}</div>`;
  };
  const statsHtml = `
    <div class="card-header-row ov-subhead">
      <span class="card-title-label">順位の差は確かか（Friedman 検定）</span>
      <span class="card-subtitle">${nFuncs} 関数 × ${nMethods} 手法</span>
    </div>
    <div class="fr-note">
      上の平均順位は関数ごとの順位の平均なので、たまたまの差も含む。Friedman 検定で「手法によって順位が違う」と言えるかを確かめ、
      言える場合は Nemenyi の<strong>臨界差</strong>（2 手法の平均順位がこれ以上離れていれば、その 2 手法は有意に違う。手法と関数の数だけで決まる）で、
      ${htmlesc(ref0)} と有意に差がある手法を数える。手法が多いほど臨界差は大きくなり、差は出にくい。
    </div>
    ${frBlock('cec')}${frBlock('bf')}${frBlock('evals')}`;

  const sortHdr = (key, label, title) => {
    const isActive = overallSortKey === key;
    const arrow = isActive ? (OV_SORT_KEYS[key].dir === 'asc' ? '▲' : '▼') : '↕';
    return `<div><span class="ov-rank-sort ${isActive ? 'active' : ''}" data-sort-key="${key}" title="${title} · クリックでソート">${label}<span class="sort-arrow">${arrow}</span></span></div>`;
  };
  let html = `
    <div class="ov-rank-header">
      <div></div><div>手法</div>
      ${sortHdr('bf', '平均順位 best_f', 'best_f（全 run 平均 mean_best_f）の Friedman 平均ランク（低い＝優）')}
      ${sortHdr('evals', '平均順位 Evals', '成功 run のみの平均評価数の Friedman 平均ランク（低い＝優）')}
      ${sortHdr('cec', 'CEC 順位', 'CEC の競技の慣例: run ごとの誤差（1e-8 未満は 0）の平均による関数ごとの順位の平均（低い＝優）。精度を問わず誤差の大きさで比べる')}
      ${sortHdr('sr_deep', 'SR@1e-10', 'SR@1e-10（最高精度・主指標）')}
      ${sortHdr('sr', 'SR@1e-4', 'SR@1e-4（補助）')}
      ${sortHdr('pr', 'PR@1e-4（履歴）', 'PR@1e-4（履歴ベース）: 既知の大域最適 K 点のうち、評価履歴に半径内かつ f≤1e-4 の点がある割合（summary.csv の pr_1e-4）。BBOB は K=1 なので SR@1e-4 とほぼ同じ値になる。CEC2013 報告集合の cec_pr_* とは別物')}
      <div title="1 位になった関数の数（best_f / Evals）">1 位の数</div><div title="最下位になった関数の数（best_f / Evals）">最下位の数</div>
    </div>
    <div class="ov-ranking">`;
  const ref = refMethod();
  lb.forEach((row, i) => {
    const medalCls = row.method === ref ? 'is-ref' : '';
    const fillBf    = (fillFor(row.mean_rank_bf)    * 100).toFixed(1);
    const fillEvals = (fillFor(row.mean_rank_evals) * 100).toFixed(1);
    const fillCec   = (fillFor(row.mean_rank_cec ?? nMethods) * 100).toFixed(1);
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
        ${bar(fillCec,   row.mean_rank_cec ?? nMethods, row.rank_std_cec ?? 0)}
        <div class="ov-rank-metric primary" title="SR@1e-10">${srDeep}</div>
        <div class="ov-rank-metric" title="SR@1e-4">${sr}</div>
        <div class="ov-rank-metric ${row.mean_pr ? '' : 'muted'}" title="PR@1e-4（履歴ベース、pr_1e-4）">${pr}</div>
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

// Diverging success-rate ramp: brick (0%) -> warm neutral (50%) -> teal (100%).
// Low values carry the colour because failures are what a reader scans for;
// brick vs teal stays distinguishable under the common colour-vision deficiencies.
const HEAT_STOPS = [[0, [238, 190, 175]], [0.5, [246, 241, 233]], [1, [200, 227, 221]]];
function heatColor(frac) {
  const f = Math.max(0, Math.min(1, frac));
  const [i0, i1] = f <= 0.5 ? [0, 1] : [1, 2];
  const [a, ca] = HEAT_STOPS[i0], [b, cb] = HEAT_STOPS[i1];
  const t = (f - a) / (b - a);
  const c = ca.map((v, k) => Math.round(v + (cb[k] - v) * t));
  return { bg: `rgb(${c.join(',')})`, fg: f < 0.35 ? '#7a2716' : f >= 0.995 ? '#0a514b' : '#17202b' };
}
function _heatmapBg(sr) { return heatColor(sr).bg; }

// The method every other method is compared against (Wilcoxon reference).
function refMethod() {
  const w = DIMS_DATA[currentDim]?.wilcoxon;
  return (w && w.length && w[0].reference) || 'MC-ESO';
}
function _medalCls(_i, method) {
  return method && method === refMethod() ? 'is-ref' : '';
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
    `${ind.desc}。手法は${ind.type === 'rank' ? '平均ランクの小さい' : '平均値の高い'}順に並ぶ。`;

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
    : (({ bg, fg }) => `<span class="ov-heatmap-pill" style="background:${bg};color:${fg};">${(v * 100).toFixed(0)}%</span>`)(heatColor(v));
  const rankCell = v => {
    let chipCls = 'null', txt = '—';
    if (v != null) {
      txt = v % 1 === 0 ? String(v) : v.toFixed(1);
      chipCls = '';
      // Same scale as the success-rate cells: 1st → green, last → brick.
      const { bg, fg } = heatColor(nMeth > 1 ? (nMeth - v) / (nMeth - 1) : 1);
      return `<span class="ov-rank-chip" style="background:${bg};color:${fg};">${txt}</span>`;
    }
    return `<span class="ov-rank-chip ${chipCls}">${txt}</span>`;
  };
  const cell    = ind.type === 'rank' ? rankCell : scoreCell;
  const fmtMean = v => (v == null || !isFinite(v)) ? '—'
    : (ind.type === 'rank' ? v.toFixed(2) : (v * 100).toFixed(0) + '%');

  // ── カテゴリ別 ──
  let ch = '<div class="ov-tbl-wrap"><table class="ov-tbl"><thead><tr><th>手法</th>';
  cats.forEach(c => { ch += `<th>${htmlesc(CAT_LABELS[c] || c)}</th>`; });
  ch += '<th class="ov-mean-col">全体</th></tr></thead><tbody>';
  ordered.forEach((row, i) => {
    ch += `<tr class="${_medalCls(i, row.method)}"><td>${htmlesc(row.method)}</td>`;
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
  th += '<th class="ov-mean-col" rowspan="2">全体</th></tr><tr>';
  flatTags.forEach(({ t, axis }) => {
    const n = funcsWith(t).length;
    th += `<th class="ovm-tag tax-${axis}" title="${htmlesc(t)}（${n} 関数）">`
        + `<span class="ovm-tag-label">${htmlesc(t)}</span><span class="ovm-tag-n">${n}</span></th>`;
  });
  th += '</tr></thead><tbody>';

  // Section 1: per-method aggregate of the selected indicator, per tag.
  th += `<tr class="ovm-sect"><td colspan="${nCols}">手法ごとの集計: ${htmlesc(ind.label)}（そのタグを持つ関数だけで${ind.type === 'rank' ? '平均ランク' : '平均'}）</td></tr>`;
  ordered.forEach((row, i) => {
    th += `<tr class="${_medalCls(i, row.method)}"><td class="ovm-rowh">${htmlesc(row.method)}</td>`;
    flatTags.forEach(({ t }) => { th += `<td>${cell(meanOver(row.method, funcsWith(t)))}</td>`; });
    th += `<td class="ov-mean-col">${fmtMean(meanOver(row.method, funcs))}</td></tr>`;
  });

  // Section 2: function → tag correspondence (which functions carry each tag).
  th += `<tr class="ovm-sect"><td colspan="${nCols}">どの関数がどのタグを持つか（● = 持つ）</td></tr>`;
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
  let fh = '<div class="ov-tbl-wrap"><table class="ov-tbl sticky-col"><thead><tr><th>手法</th>';
  funcs.forEach(f => {
    const tt = (ftags[f] || []).join(', ');
    fh += `<th title="${htmlesc(f + (tt ? '  ·  ' + tt : ''))}">${htmlesc(shortLabel(f))}</th>`;
  });
  fh += '<th class="ov-mean-col">平均</th></tr></thead><tbody>';
  ordered.forEach((row, i) => {
    fh += `<tr class="${_medalCls(i, row.method)}"><td>${htmlesc(row.method)}</td>`;
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

  renderFunctionGrid();
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
  // Competition rank: a value's rank is the number of values strictly better
  // than it, so methods tied for first are all 'cell-best' (the old version
  // took the first sorted position, which put ties for the top of a
  // higher-is-better column at the bottom).
  // Non-finite values (Evals of a method with no successful run) get no
  // colour, so an all-failed column is not painted 'best'.
  const nums = values.map(v => parseFloat(v));
  const valid = nums.filter(v => isFinite(v));
  if (valid.length < 2) return nums.map(() => '');
  const n = valid.length;
  return nums.map(v => {
    if (!isFinite(v)) return '';
    const rank = valid.filter(w => higherBetter ? w > v : w < v).length;
    const worse = valid.filter(w => higherBetter ? w < v : w > v).length;
    if (rank === 0)                      return 'cell-best';
    if (worse === 0)                     return 'cell-worst';
    if (rank <= Math.floor(n * 0.25))    return 'cell-good';
    if (rank <= Math.floor(n * 0.5))     return 'cell-mid';
    return 'cell-bad';
  });
}

// ── Unified table ─────────────────────────────────────────────────────────────
// Per-function table sort state (kept across functions so a chosen column stays).
let unifiedSort = { key: null, dir: 'asc' };

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
  // CEC competitions' convention: per-run error f − f* below 1e-8 counts as 0,
  // averaged over the runs; methods ranked on that mean (ties share the
  // average rank). Same definition as core.run_data.cec_mean_errors.
  const CEC_ZERO = 1e-8;
  const cecErr = summaryRows.map(r => {
    const errs = (byMethod[r.method] || []).map(x => parseFloat(x.best_f)).filter(v => !isNaN(v));
    return errs.length ? errs.reduce((a, v) => a + (v < CEC_ZERO ? 0 : v), 0) / errs.length : Infinity;
  });
  const cecColors = rankColors(cecErr.map(v => isFinite(v) ? v : 'Infinity'), false);
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
  const srHeatColor = heatColor;

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
    { label: '誤差 f − f*（平均）', desc: '全 run の最終誤差 f − f* の平均（BBOB は最小値 0）。低いほど良い。' },
    { label: '誤差 f − f*（CEC）', desc: 'CEC の競技の慣例: run ごとの誤差のうち 1e-8 未満を 0 とみなして平均したもの。「平均」との違いは 1e-8 未満の端数だけで、全 run が 1e-8 未満なら 0 になり同着になる。全関数での順位の平均がランキング画面の「CEC 順位」。' },
    { label: 'SR@target', desc: '各精度目標（1e⁻⁴ / 1e⁻⁷ / 1e⁻¹⁰）に到達した run の割合。左ほど緩く右ほど厳しい。色は成功率（赤=低、緑=高）、数値は正確な %。1e⁻¹⁰ が主指標。行を展開すると各 seed が目標ごとに ✓（到達）/ ✗（未到達）で出る。' },
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
  // Sortable columns: key → [direction, value of a summary row]. Rows sort as
  // method groups (a method row keeps its expanded seed rows under it).
  const SORT_SPEC = {
    name:  ['asc',  (r, i) => r.method],
    bf:    ['asc',  (r, i) => parseFloat(r.mean_best_f)],
    cec:   ['asc',  (r, i) => cecErr[i]],
    sr4:   ['desc', (r, i) => parseSRFraction(r['sr_1e-4'])],
    sr7:   ['desc', (r, i) => parseSRFraction(r['sr_1e-7'])],
    sr10:  ['desc', (r, i) => parseSRFraction(r['sr_1e-10'])],
    evals: ['asc',  (r, i) => parseFloat(r.evals_succ_mean ?? r.evals_succ_med ?? r.ert)],
    time:  ['asc',  (r, i) => parseFloat(r.mean_time_s)],
    or:    ['desc', (r, i) => parseFloat(r.mean_optima_rate)],
  };
  const sortAttr = k => `data-sort="${k}" class="sortable${unifiedSort.key === k ? ' sorted' : ''}" title="クリックで並べ替え"`;
  const sortMark = k => unifiedSort.key === k ? (unifiedSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
  const SR_SORT_KEYS = ['sr4', 'sr7', 'sr10'];
  const srSubHead = SR_TARGET_LABELS.map((lbl, i) =>
    `<th ${sortAttr(SR_SORT_KEYS[i]).replace('class="', 'class="sr-head ')}>${lbl}${sortMark(SR_SORT_KEYS[i])}</th>`
  ).join('');

  let html = `
  <details class="col-legend">
    <summary>列の意味<span class="col-legend-toggle">▶</span></summary>
    <div class="col-legend-body">${legendItems}</div>
  </details>
  <div class="table-wrap">
  <table class="unified-table">
    <thead>
      <tr>
        <th rowspan="2" style="text-align:left;min-width:150px;" ${sortAttr('name')}>手法 / seed${sortMark('name')}</th>
        <th colspan="2" class="sr-group-head">誤差 f − f*</th>
        <th colspan="${SR_TARGETS.length}" class="sr-group-head">SR@target</th>
        <th rowspan="2" ${sortAttr('evals')}>Evals (succ mean)${sortMark('evals')}</th>
        <th rowspan="2" ${sortAttr('time')}>time (s)${sortMark('time')}</th>
        <th rowspan="2" ${sortAttr('or')}>optima rate${sortMark('or')}</th>
        <th rowspan="2">evals</th>
      </tr>
      <tr><th class="sr-head" ${sortAttr('bf')} title="全 run の平均">平均${sortMark('bf')}</th><th class="sr-head" ${sortAttr('cec')} title="CEC の慣例: 1e-8 未満を 0 として平均">CEC${sortMark('cec')}</th>${srSubHead}</tr>
    </thead>
    <tbody id="unified-tbody">`;

  let order = summaryRows.map((_, i) => i);
  if (unifiedSort.key && SORT_SPEC[unifiedSort.key]) {
    const get = SORT_SPEC[unifiedSort.key][1];
    const sign = unifiedSort.dir === 'asc' ? 1 : -1;
    const norm = v => (v == null || (typeof v === 'number' && isNaN(v))) ? null : v;
    order.sort((a, b) => {
      const va = norm(get(summaryRows[a], a)), vb = norm(get(summaryRows[b], b));
      if (va === null && vb === null) return 0;
      if (va === null) return 1;          // missing values always last
      if (vb === null) return -1;
      if (typeof va === 'string') return sign * va.localeCompare(vb);
      if (va !== vb) return sign * (va < vb ? -1 : 1);
      // Ties (e.g. many methods at 0%) fall back to the smaller mean error.
      const ea = parseFloat(summaryRows[a].mean_best_f), eb = parseFloat(summaryRows[b].mean_best_f);
      return (isNaN(ea) ? Infinity : ea) - (isNaN(eb) ? Infinity : eb) || 0;
    });
  }
  order.forEach(si => {
    const sr = summaryRows[si];
    const method = sr.method;
    const runs   = byMethod[method] || [];
    html += `
      <tr class="method-row${method === refMethod() ? ' is-ref' : ''}" data-method="${htmlesc(method)}" data-expanded="false">
        <td><div class="method-cell">
          <span class="expand-toggle">${runs.length ? '▶' : '·'}</span>
          ${method}
        </div></td>
        <td class="${bfColors[si]}">${fmtNum(sr.mean_best_f)}</td>
        <td class="${cecColors[si]}">${isFinite(cecErr[si]) ? fmtNum(cecErr[si]) : '—'}</td>
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
          <td>${(() => { const v = parseFloat(run.best_f); return isNaN(v) ? '—' : fmtNum(v < CEC_ZERO ? 0 : v); })()}</td>
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

  container.querySelectorAll('th[data-sort]').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.sort;
    unifiedSort = unifiedSort.key === k
      ? { key: k, dir: unifiedSort.dir === 'asc' ? 'desc' : 'asc' }
      : { key: k, dir: SORT_SPEC[k][0] };
    buildUnifiedTable(func);
  }));

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
      // Every seed tied: nothing to read beyond "=", so keep the cell quiet.
      if (tc === n) return `<td title="${tip}"><span class="ov-wc-chip is-tie"><span class="mark">=</span></span></td>`;
      return `<td title="${tip}"><span class="ov-wc-chip ${cls}">
        <span class="mark">${mark}</span>
        <span class="count">${wc}-${tc}-${lc}</span>
        <span class="a12 ${a12Cls}">A=${a12Str}</span>
      </span></td>`;
    }).join('');
    return `<tr><td>${htmlesc(f)}</td>${cells}</tr>`;
  }).join('');

  container.innerHTML = `
    <div class="wc-legend">
      ★ p &lt; 0.01（高度に有意 ${htmlesc(ref)} が優位）／ ✓ p &lt; 0.05（有意）／ = 引分け or 非有意 ／ ✗ p &gt; 0.95（相手が優位）。
      セル 2 段目は 勝-引-負 の seed 数 ／ 3 段目は Vargha–Delaney A₁₂（&gt;0.5 で ${htmlesc(ref)} 優位、|A−0.5| ≤ .06 negligible / ≤ .14 small / ≤ .21 medium / &gt; .21 large）。
    </div>
    <div class="wilcoxon-summary-row">${overallTiles}</div>
    <details open>
      <summary class="overall-details-summary">関数ごとの内訳</summary>
      <div class="ov-tbl-wrap" style="margin-top:10px;">
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
