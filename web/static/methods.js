// Methods page: formulas (KaTeX), results from methods_data.json (built by
// scripts/web/methods_data.py from the quick CSVs), the population-size figure,
// the lineage diagram, and the table-of-contents highlight.

const SVGNS = 'http://www.w3.org/2000/svg';
const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const pct = (v, d = 1) => v == null ? '—' : `${(v * 100).toFixed(d)}%`;
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

// Same diverging ramp as the results page: brick (0%) → neutral (50%) → teal (100%).
const HEAT_STOPS = [[0, [238, 190, 175]], [0.5, [246, 241, 233]], [1, [200, 227, 221]]];
function heatColor(frac) {
  const f = Math.max(0, Math.min(1, frac));
  const [i0, i1] = f <= 0.5 ? [0, 1] : [1, 2];
  const [a, ca] = HEAT_STOPS[i0], [b, cb] = HEAT_STOPS[i1];
  const t = (f - a) / (b - a);
  const c = ca.map((v, k) => Math.round(v + (cb[k] - v) * t));
  return { bg: `rgb(${c.join(',')})`, fg: f < 0.35 ? '#7a2716' : f >= 0.995 ? '#0a514b' : '#17202b' };
}

// ── KaTeX ────────────────────────────────────────────────────────────────────
function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.querySelector('.mt-main'), {
    delimiters: [
      { left: '\\[', right: '\\]', display: true },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false, strict: 'ignore',
  });
}

// ── Table of contents: highlight the section in view ────────────────────────
function initToc() {
  const links = [...document.querySelectorAll('.mt-toc a')];
  const byId = Object.fromEntries(links.map(a => [a.getAttribute('href').slice(1), a]));
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      links.forEach(a => a.classList.remove('active'));
      byId[e.target.id]?.classList.add('active');
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  Object.keys(byId).forEach(id => { const s = document.getElementById(id); if (s) obs.observe(s); });
}

// ── Population size N(t) ─────────────────────────────────────────────────────
function drawPopFigure() {
  const svg = document.getElementById('pop-fig');
  if (!svg) return;
  const W = 360, H = 210, L = 44, R = 46, T = 16, B = 34;
  const series = [
    { D: 2,  color: '#9aa5b1' },
    { D: 5,  color: '#3d6f99' },
    { D: 10, color: '#0e6b63' },
  ];
  const nOf = (D, t) => {
    const ni = Math.max(20, 16 * D), nf = Math.max(10, 4 * D);
    return nf + (ni - nf) * (1 - t) ** 2;
  };
  const yMax = 160;
  const X = t => L + t * (W - L - R);
  const Y = n => T + (1 - n / yMax) * (H - T - B);
  [0, 40, 80, 120, 160].forEach(n => {
    el('line', { x1: L, x2: W - R, y1: Y(n), y2: Y(n), class: 'pop-grid' }, svg);
    el('text', { x: L - 8, y: Y(n) + 4, 'text-anchor': 'end', class: 'pop-tick' }, svg).textContent = n;
  });
  [0, .25, .5, .75, 1].forEach(t => {
    el('text', { x: X(t), y: H - B + 16, 'text-anchor': 'middle', class: 'pop-tick' }, svg).textContent = `${Math.round(t * 100)}%`;
  });
  el('text', { x: (L + W - R) / 2, y: H - 4, 'text-anchor': 'middle', class: 'pop-tick' }, svg).textContent = '使った予算';
  el('line', { x1: L, x2: W - R, y1: Y(0), y2: Y(0), class: 'pop-axis' }, svg);
  // series run 2D → 10D, i.e. bottom → top; keep end labels at least 12px apart
  let lastLabelY = Infinity;
  series.forEach(s => {
    const pts = [];
    for (let i = 0; i <= 60; i++) { const t = i / 60; pts.push(`${X(t).toFixed(1)},${Y(nOf(s.D, t)).toFixed(1)}`); }
    el('polyline', { points: pts.join(' '), class: 'pop-line', stroke: s.color }, svg);
    const ly = Math.min(Y(nOf(s.D, 1)) + 4, lastLabelY - 12);
    lastLabelY = ly;
    const lab = el('text', { x: X(1) + 6, y: ly, class: 'pop-label', fill: s.color }, svg);
    lab.textContent = `${s.D}D`;
  });
}

// ── Results ──────────────────────────────────────────────────────────────────
let DATA = null;

function renderResults(dim) {
  const run = DATA.runs.find(r => r.dim === dim);
  if (!run) return;
  const ref = DATA.reference;
  document.querySelectorAll('#res-tabs button').forEach(b =>
    b.setAttribute('aria-selected', String(Number(b.dataset.dim) === dim)));

  document.getElementById('res-meta').innerHTML =
    `${run.dim} 次元、${run.n_runs} run、予算 ${Number(run.max_evals).toLocaleString()} 評価。`
    + `結果 <code>${esc(run.run)}</code>（commit <code>${esc(run.commit)}</code>）`;

  const order = [...run.methods].sort((a, b) =>
    (run.overall[b]['sr_1e-10'] ?? -1) - (run.overall[a]['sr_1e-10'] ?? -1));
  const max10 = Math.max(...order.map(m => run.overall[m]['sr_1e-10'] ?? 0));
  let h = `<thead><tr><th>手法</th><th>SR@1e-10</th><th>SR@1e-7</th><th>SR@1e-4</th>`
    + `<th title="成功した run だけの平均評価回数を、関数について平均したもの">平均評価回数</th>`
    + `<th title="MC-ESO が有意に良い / 悪い関数の数（Wilcoxon, p &lt; 0.05）">勝 / 負</th></tr></thead><tbody>`;
  order.forEach(m => {
    const o = run.overall[m];
    const isRef = m === ref;
    const w = (o['sr_1e-10'] ?? 0) / (max10 || 1) * 100;
    const wl = isRef ? '<span class="dim-muted">基準</span>'
      : `<span class="wl-win">${o.wins}</span><span class="wl-sep">/</span><span class="wl-loss">${o.losses}</span>`;
    const card = document.getElementById(`m-${m}`);
    const name = card ? `<a href="#m-${esc(m)}">${esc(m)}</a>` : esc(m);
    h += `<tr class="${isRef ? 'is-ref' : ''}"><td class="m-name">${name}</td>`
      + `<td><div class="sr-cell"><span class="sr-bar"><i style="width:${w.toFixed(1)}%"></i></span><span class="sr-val">${pct(o['sr_1e-10'])}</span></div></td>`
      + `<td>${pct(o['sr_1e-7'])}</td><td>${pct(o['sr_1e-4'])}</td>`
      + `<td>${o.evals == null ? '—' : Math.round(o.evals).toLocaleString()}</td><td>${wl}</td></tr>`;
  });
  document.getElementById('res-table').innerHTML = h + '</tbody>';

  document.getElementById('res-notes').innerHTML = notesFor(run, order, ref);
  renderHeat(run, order, ref);
}

// Reading notes, computed from the same data so they cannot drift from the table.
function notesFor(run, order, ref) {
  const o = run.overall;
  const others = order.filter(m => m !== ref && m !== `${ref}-v0`);
  const refSr = o[ref]['sr_1e-10'];
  const better = others.filter(m => o[m]['sr_1e-10'] > refSr + 1e-9);
  const tied = others.filter(m => Math.abs(o[m]['sr_1e-10'] - refSr) < 1e-9);
  let s = '<h4>既存手法との比較</h4><p>';
  if (!better.length && !tied.length) {
    const second = others[0];
    s += `SR@1e-10 は ${ref} が ${pct(refSr)} で最も高い。次は ${esc(second)} の ${pct(o[second]['sr_1e-10'])}。`;
  } else {
    s += `SR@1e-10 は ${ref} が ${pct(refSr)}。`;
    if (better.length) s += `上回る手法: ${better.map(m => `${esc(m)}（${pct(o[m]['sr_1e-10'])}）`).join('、')}。`;
    if (tied.length) s += `同率: ${tied.map(esc).join('、')}。`;
  }
  s += '</p>';

  const v0 = `${ref}-v0`;
  if (o[v0]) {
    const diff = (refSr - o[v0]['sr_1e-10']) * 100;
    const up = [], down = [];
    run.funcs.forEach(f => {
      const a = run.sr10_by_func[f][v0], b = run.sr10_by_func[f][ref];
      if (a == null || b == null) return;
      const d = Math.round((b - a) * 100);
      if (d > 0) up.push([f, d, a, b]); else if (d < 0) down.push([f, d, a, b]);
    });
    up.sort((x, y) => y[1] - x[1]);
    const fmt = ([f, d, a, b]) => `<span class="fn">${esc(f.split('-')[0])}</span> ${Math.round(a * 100)}→${Math.round(b * 100)}%`;
    s += `<h4>旧既定（${esc(v0)}）からの変化</h4><p>平均で <b class="${diff >= 0 ? 'up' : 'down'}">${diff >= 0 ? '+' : ''}${diff.toFixed(1)}pt</b>。`
      + (up.length ? `上がった関数 ${up.length}（${up.slice(0, 4).map(fmt).join('、')}${up.length > 4 ? ' ほか' : ''}）。` : '')
      + (down.length ? `<span class="down">下がった関数 ${down.length}</span>（${down.map(fmt).join('、')}）。` : '下がった関数は無い。')
      + '</p>';
  }
  return s;
}

function renderHeat(run, order, ref) {
  const funcs = run.funcs;
  let h = '<thead><tr><th></th>';
  funcs.forEach(f => {
    const [num, ...rest] = f.split('-');
    h += `<th class="fn-h" title="${esc(f)}"><span>${esc(num)}</span> ${esc(rest.join('-'))}</th>`;
  });
  h += '<th>平均</th></tr></thead><tbody>';
  order.forEach(m => {
    h += `<tr class="${m === ref ? 'is-ref' : ''}"><td class="h-name">${esc(m)}</td>`;
    funcs.forEach(f => {
      const v = run.sr10_by_func[f][m];
      if (v == null || Number.isNaN(v)) { h += '<td>—</td>'; return; }
      const { bg, fg } = heatColor(v);
      h += `<td style="background:${bg};color:${fg}" title="${esc(m)} / ${esc(f)}: ${Math.round(v * 100)}%">${Math.round(v * 100)}</td>`;
    });
    h += `<td class="h-mean">${pct(run.overall[m]['sr_1e-10'], 0)}</td></tr>`;
  });
  document.getElementById('res-heat').innerHTML = h + '</tbody>';
}

function decorateCards() {
  const runs = DATA.runs;
  document.querySelectorAll('.mt-card[data-method]').forEach(card => {
    const m = card.dataset.method;
    if (!runs.some(r => r.overall[m])) return;
    const div = document.createElement('div');
    div.className = 'mt-card-sr';
    div.innerHTML = '<span class="mt-card-sr-label">今回の quick の SR@1e-10</span>'
      + runs.map(r => `<span>${r.dim}D<b>${pct(r.overall[m]?.['sr_1e-10'])}</b></span>`).join('');
    card.appendChild(div);
  });
}

async function initResults() {
  try {
    const res = await fetch('/static/methods_data.json', { cache: 'no-cache' });
    DATA = await res.json();
  } catch (e) {
    document.getElementById('res-meta').textContent = '成績データ（methods_data.json）を読み込めませんでした。';
    return;
  }
  const tabs = document.getElementById('res-tabs');
  tabs.innerHTML = DATA.runs.map(r =>
    `<button role="tab" data-dim="${r.dim}" aria-selected="false">${r.dim} 次元</button>`).join('');
  tabs.addEventListener('click', e => {
    const b = e.target.closest('button[data-dim]');
    if (b) renderResults(Number(b.dataset.dim));
  });
  renderResults(DATA.runs[0].dim);
  decorateCards();
  const measured = new Set(DATA.runs.flatMap(r => r.methods));
  drawLineage(measured);
  let lastW = 0, timer = null;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const w = document.querySelector('.mt-lineage-wrap')?.clientWidth || 0;
      if (Math.abs(w - lastW) > 8) { lastW = w; drawLineage(measured); }
    }, 150);
  });
}

// ── Lineage diagram ──────────────────────────────────────────────────────────
const LANES = [
  { id: 'es',     label: '進化戦略',       sub: 'CMA-ES 系',       color: '#0e6b63' },
  { id: 'de',     label: '差分進化',       sub: 'DE 系',           color: '#a8741c' },
  { id: 'swarm',  label: '群知能・生物模倣', sub: '',               color: '#3d6f99' },
  { id: 'hybrid', label: '組み合わせ・選択', sub: '',               color: '#7a5c99' },
  { id: 'local',  label: '局所探索・多解',  sub: '',               color: '#6b7480' },
];
// id = card id suffix (m-<id>) and the registered method name.
const NODES = [
  ['NM-Restart', 1965, 'local'], ['PSO', 1995, 'swarm'], ['DE', 1997, 'de'],
  ['CMA-ES', 2001, 'es'], ['DEPSO', 2003, 'swarm'], ['Crowding-DE', 2004, 'local'],
  ['IPOP-CMA-ES', 2005, 'es'], ['BIPOP-CMA-ES', 2009, 'es'], ['PS-CMA-ES', 2009, 'es'],
  ['AMALGAM-SO', 2009, 'hybrid'], ['MOS', 2010, 'hybrid'], ['HMHH', 2010, 'hybrid'],
  ['r3pso', 2010, 'local'], ['NCDE', 2012, 'local'], ['ICMAES-ILS', 2013, 'es', 'iCMAES-ILS'],
  ['L-SHADE', 2014, 'de'], ['CoBiDE', 2014, 'de'], ['NMMSO', 2014, 'local'],
  ['SPS-L-SHADE-EIG', 2015, 'de'], ['MAP-Elites', 2015, 'local'], ['UMOEA-II', 2016, 'hybrid'],
  ['jSO', 2017, 'de'], ['LSHADE-cnEpSin', 2017, 'de'], ['LSHADE-SPACMA', 2017, 'de'],
  ['EBOwithCMAR', 2017, 'swarm'], ['HSES', 2018, 'es'], ['ELSHADE-SPACMA', 2018, 'de'],
  ['IMODE', 2020, 'de'], ['SaVOA', 2020, 'swarm'], ['NGOpt', 2020, 'hybrid', 'NGOpt / Portfolio'],
  ['APGSK-IMODE', 2021, 'de'], ['EA4eig', 2022, 'hybrid'], ['Repel-CMA-ES', 2024, 'es'],
  ['L-SRTDE', 2024, 'de'], ['MC-ESO', 2026, 'swarm', 'MC-ESO（本研究）'],
].map(([id, year, lane, label]) => ({ id, year, lane, label: label || id }));
const EDGES = [
  ['CMA-ES', 'IPOP-CMA-ES'], ['IPOP-CMA-ES', 'BIPOP-CMA-ES'], ['IPOP-CMA-ES', 'ICMAES-ILS'],
  ['IPOP-CMA-ES', 'Repel-CMA-ES'], ['CMA-ES', 'PS-CMA-ES'], ['PSO', 'PS-CMA-ES'], ['CMA-ES', 'HSES'],
  ['DE', 'Crowding-DE'], ['Crowding-DE', 'NCDE'], ['PSO', 'r3pso'], ['PSO', 'NMMSO'],
  ['DE', 'L-SHADE'], ['L-SHADE', 'jSO'], ['L-SHADE', 'LSHADE-cnEpSin'], ['L-SHADE', 'SPS-L-SHADE-EIG'],
  ['L-SHADE', 'LSHADE-SPACMA'], ['CMA-ES', 'LSHADE-SPACMA'], ['LSHADE-SPACMA', 'ELSHADE-SPACMA'],
  ['L-SHADE', 'L-SRTDE'], ['DE', 'CoBiDE'], ['DE', 'IMODE'], ['IMODE', 'APGSK-IMODE'],
  ['PSO', 'DEPSO'], ['DE', 'DEPSO'], ['CMA-ES', 'EBOwithCMAR'],
  ['CMA-ES', 'AMALGAM-SO'], ['PSO', 'AMALGAM-SO'], ['DE', 'MOS'], ['IPOP-CMA-ES', 'MOS'],
  ['PSO', 'HMHH'], ['CMA-ES', 'HMHH'], ['DE', 'UMOEA-II'], ['CMA-ES', 'UMOEA-II'],
  ['CoBiDE', 'EA4eig'], ['jSO', 'EA4eig'], ['CMA-ES', 'EA4eig'],
].map(([a, b]) => ({ a, b }));
const BORROWS = ['CMA-ES', 'DE', 'SaVOA', 'IPOP-CMA-ES', 'BIPOP-CMA-ES', 'L-SHADE']
  .map(a => ({ a, b: 'MC-ESO', borrow: true }));

function drawLineage(measured) {
  const svg = document.getElementById('lineage-svg');
  if (!svg) return;
  // Draw at the container's real width so text stays at its CSS size.
  const W = Math.max(900, Math.round(svg.parentElement.clientWidth)), LABEL_W = 132, PAD_R = 16, TOP = 34, ROW = 25, LANE_PAD = 14;
  const XS = LABEL_W + 70, X0 = LABEL_W + 24;           // 1995.. starts at XS, 1965 sits at X0
  const Y0 = 1993, Y1 = 2026.6;
  const xOf = y => y < 1990 ? X0 : XS + (y - Y0) / (Y1 - Y0) * (W - PAD_R - 196 - XS);
  const ctx = document.createElement('canvas').getContext('2d');
  const textW = (t, bold) => { ctx.font = `${bold ? 700 : 400} ${bold ? 13.5 : 12}px "IBM Plex Sans JP", sans-serif`; return ctx.measureText(t).width; };

  // Pack labels into sub-rows inside each lane so none overlap.
  const lanes = LANES.map(l => ({ ...l, rows: [] }));
  const laneOf = Object.fromEntries(lanes.map(l => [l.id, l]));
  const pos = {};
  [...NODES].sort((a, b) => a.year - b.year).forEach(n => {
    const lane = laneOf[n.lane];
    const x = xOf(n.year);
    const isRef = n.id === 'MC-ESO';
    const tw = textW(n.label, isRef);
    const left = x - 6;
    const right = x + 10 + tw + 10;
    let r = lane.rows.findIndex(end => end < left);
    if (r < 0) { r = lane.rows.length; lane.rows.push(right); } else lane.rows[r] = right;
    pos[n.id] = { x, r, lane: n.lane, isRef, node: n, labelEnd: x + 10 + tw + 4 };
  });
  let y = TOP;
  lanes.forEach(l => { l.y = y; l.h = Math.max(1, l.rows.length) * ROW + LANE_PAD * 2; y += l.h + 6; });
  const H = y + 4;
  Object.values(pos).forEach(p => { const l = laneOf[p.lane]; p.y = l.y + LANE_PAD + ROW / 2 + p.r * ROW; });

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  // lanes
  lanes.forEach(l => {
    el('rect', { x: 0, y: l.y, width: W, height: l.h, class: 'ln-lane-bg', opacity: lanes.indexOf(l) % 2 ? 0 : 1 }, svg);
    const t = el('text', { x: 16, y: l.y + 22, class: 'ln-lane-label', fill: l.color }, svg);
    t.textContent = l.label;
    if (l.sub) { const s = el('text', { x: 16, y: l.y + 39, class: 'ln-year' }, svg); s.textContent = l.sub; }
  });
  // year axis
  [[1965, '1965'], [1995], [2000], [2005], [2010], [2015], [2020], [2025]].forEach(([yr, lab]) => {
    const x = xOf(yr);
    el('line', { x1: x, x2: x, y1: TOP - 6, y2: H - 4, class: 'ln-yearline' }, svg);
    const t = el('text', { x, y: TOP - 12, 'text-anchor': 'middle', class: 'ln-year' }, svg);
    t.textContent = lab || String(yr);
  });
  const bx = (X0 + XS) / 2;
  el('path', { d: `M${bx - 5} ${TOP - 20} l4 -8 M${bx + 1} ${TOP - 20} l4 -8`, class: 'ln-break' }, svg);

  // edges
  const edgeEls = [];
  [...EDGES, ...BORROWS].forEach(e => {
    const a = pos[e.a], b = pos[e.b];
    if (!a || !b) return;
    // Leave from the end of the parent's label when the child is far enough to
    // the right; otherwise drop out of the dot vertically so the line never
    // runs through a label.
    let d;
    if (b.x - a.labelEnd > 36) {
      const dx = Math.max(24, (b.x - a.labelEnd) * 0.5);
      d = `M${a.labelEnd} ${a.y} C${a.labelEnd + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x - 6} ${b.y}`;
    } else {
      const my = (a.y + b.y) / 2;
      d = `M${a.x} ${a.y + (b.y > a.y ? 6 : -6)} C${a.x} ${my}, ${b.x - 24} ${b.y}, ${b.x - 6} ${b.y}`;
    }
    const p = el('path', {
      d,
      class: `ln-edge${e.borrow ? ' borrow' : ''}`,
    }, svg);
    p.dataset.a = e.a; p.dataset.b = e.b;
    edgeEls.push(p);
  });

  // nodes
  const nodeEls = {};
  Object.values(pos).forEach(p => {
    const n = p.node;
    const color = laneOf[n.lane].color;
    const g = el('g', {
      class: `ln-node${measured.has(n.id) ? ' measured' : ''}${p.isRef ? ' is-ref' : ''}`,
      tabindex: 0, role: 'link', color, 'aria-label': `${n.label}（${n.year}）`,
    }, svg);
    el('circle', { cx: p.x, cy: p.y, r: p.isRef ? 7 : 5, stroke: color }, g);
    const t = el('text', { x: p.x + (p.isRef ? 12 : 10), y: p.y + (p.isRef ? 4.5 : 4) }, g);
    t.textContent = n.label;
    const tt = el('title', {}, g);
    tt.textContent = `${n.label}（${n.year}）${measured.has(n.id) ? ' — 今回の quick で測定' : ''}`;
    nodeEls[n.id] = g;
    const go = () => {
      const card = document.getElementById(`m-${n.id}`);
      if (!card) return;
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.add('flash');
      setTimeout(() => card.classList.remove('flash'), 1600);
    };
    g.addEventListener('click', go);
    g.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); });
    const on = () => {
      svg.classList.add('has-hover');
      g.classList.add('hl');
      edgeEls.forEach(p => {
        if (p.dataset.a === n.id || p.dataset.b === n.id) {
          p.classList.add('hl');
          nodeEls[p.dataset.a]?.classList.add('hl');
          nodeEls[p.dataset.b]?.classList.add('hl');
        }
      });
    };
    const off = () => {
      svg.classList.remove('has-hover');
      svg.querySelectorAll('.hl').forEach(x => x.classList.remove('hl'));
    };
    g.addEventListener('mouseenter', on); g.addEventListener('mouseleave', off);
    g.addEventListener('focus', on); g.addEventListener('blur', off);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initToc();
  drawPopFigure();
  initResults();
  // KaTeX loads with defer; wait for it if needed.
  if (typeof renderMathInElement === 'function') renderMath();
  else window.addEventListener('load', renderMath);
});
