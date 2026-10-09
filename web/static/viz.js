// Interactive figures for one function of a run (results page, "図" card).
//
//   Viz.curves(el, ctx)     convergence: median per method from curves/{Func}.json.gz
//   Viz.search(el, ctx)     one run re-executed on the server (/api/replay): every
//                           evaluated point, the population, the best-so-far path,
//                           over the landscape (2D) or a slice through the optimum (≥3D)
//   Viz.internals(el, ctx)  MC-ESO internals for any run: σ, population size, strains,
//                           router signals and route, channel shares, restarts
//
// ctx = { runId, dim: 'dim2', func, methods: [...] } — methods in summary order.
(function () {
  'use strict';

  // ── look (mirrors core/visualize.py and style.css) ──────────────────────────
  const INK = '#17202b', INK2 = '#4a5563', MUTED = '#7d8794', RULE = '#dce1e3', GRID = '#e9edee';
  const OTHER = '#b3bac1', GOOD = '#00897b', BAD = '#c2412d';
  const METHOD_COLOR = {
    'MC-ESO': '#00897b', 'CMA-ES': '#4a3aa7', 'IPOP-CMA-ES': '#eb6834', 'BIPOP-CMA-ES': '#2a78d6',
    'DE': '#e87ba4', 'L-SHADE': '#eda100', 'PSO': '#e34948', 'SaVOA': '#008300',
  };
  const colorOf = m => METHOD_COLOR[m] || (m === 'MC-ESO-v0' ? '#5fb3a8' : null);
  const CHANNEL = {               // trace channel codes (core/optimizers/mceso.py TRACE_CHANNELS)
    0: ['接触感染', '#0e6b63'], 1: ['飛沫感染', '#c08a1e'], 2: ['空気感染', '#3d6f99'],
    3: ['慣性', '#7a5c99'], 4: ['移動', '#6b7480'], 5: ['スピルオーバーの撒き直し', '#c2412d'],
    6: ['初期集団', '#9aa5b1'], 7: ['局所探索', '#17202b'],
  };
  const ROUTE = { pending: ['未確定', '#d5dadd'], keepair: ['KEEP-AIR', '#3d6f99'],
                  droplet: ['DROPLET', '#c08a1e'], close: ['CLOSE', '#0e6b63'] };
  // Router thresholds (MultiChannelEpidemicOptimizer defaults)
  const TH = { cond_early: 4.0, cond: 3.0, align: 0.965, mgap: 0.36, commit_gen: 120 };
  const LAND = [[47, 95, 102], [134, 170, 169], [216, 227, 224], [246, 245, 240]];

  // ── tiny helpers ────────────────────────────────────────────────────────────
  const SVGNS = 'http://www.w3.org/2000/svg';
  const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  function svg(tag, attrs = {}, parent) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function bytes(b64) {
    const bin = atob(b64); const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  const f32 = b64 => b64 ? new Float32Array(bytes(b64).buffer) : null;
  const fmtE = v => (v == null || !isFinite(v)) ? '—' : (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4 ? (+v).toPrecision(3) : (+v).toExponential(2));
  const pow10 = l => fmtE(Math.pow(10, l));
  const fmtInt = v => Math.round(v).toLocaleString();
  function ramp(stops, t) {
    t = Math.max(0, Math.min(1, t)); const k = t * (stops.length - 1); const i = Math.min(stops.length - 2, Math.floor(k));
    const u = k - i; return stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * u));
  }
  async function getJSON(url) {
    const r = await fetch(url);
    let d = null; try { d = await r.json(); } catch (_) {}
    if (!r.ok) throw new Error((d && d.error) || `HTTP ${r.status}`);
    return d;
  }
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function note(el, html, cls = '') { el.innerHTML = `<p class="viz-note ${cls}">${html}</p>`; }

  // Line chart frame: x linear, y given scale. Returns {g, X, Y, w, h, svgEl}.
  function frame(parent, { width, height, x0, x1, y0, y1, yTicks, yFmt, xLabel, yLabel, m = { l: 56, r: 12, t: 10, b: 30 }, showX = true }) {
    const s = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'viz-svg', width, height }, parent);
    const w = width - m.l - m.r, h = height - m.t - m.b;
    const X = v => m.l + (v - x0) / ((x1 - x0) || 1) * w;
    const Y = v => m.t + (1 - (v - y0) / ((y1 - y0) || 1)) * h;
    const g = svg('g', {}, s);
    (yTicks || niceTicks(y0, y1, 4)).forEach(v => {
      svg('line', { x1: m.l, x2: m.l + w, y1: Y(v), y2: Y(v), class: 'viz-grid' }, g);
      svg('text', { x: m.l - 6, y: Y(v) + 3.5, class: 'viz-tick', 'text-anchor': 'end' }, g).textContent = yFmt ? yFmt(v) : v;
    });
    if (showX) {
      niceTicks(x0, x1, 6).forEach(v => {
        svg('text', { x: X(v), y: m.t + h + 15, class: 'viz-tick', 'text-anchor': 'middle' }, g).textContent = fmtInt(v);
      });
      if (xLabel) svg('text', { x: m.l + w / 2, y: height - 2, class: 'viz-axis', 'text-anchor': 'middle' }, g).textContent = xLabel;
    }
    svg('line', { x1: m.l, x2: m.l + w, y1: m.t + h, y2: m.t + h, class: 'viz-base' }, g);
    if (yLabel) svg('text', { x: m.l, y: m.t - 1, class: 'viz-axis' }, g).textContent = yLabel;
    return { s, g, X, Y, w, h, m };
  }
  const path = (xs, ys, X, Y) => xs.map((x, i) => (ys[i] == null || !isFinite(ys[i])) ? null
    : `${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).reduce((acc, p) => {
      if (p == null) { acc.pen = false; return acc; }
      acc.d += (acc.pen ? 'L' : 'M') + p; acc.pen = true; return acc;
    }, { d: '', pen: false }).d;

  // ── 1. convergence ──────────────────────────────────────────────────────────
  async function curves(el, ctx) {
    note(el, '読み込み中…');
    let data;
    try { data = await getJSON(`/api/curves/${encodeURIComponent(ctx.runId)}/${ctx.dim}/${encodeURIComponent(ctx.func)}`); }
    catch (_) {
      note(el, 'この run には収束データがありません（2026-10-09 より前の run）。同じ条件で回し直すと表示されます。');
      return;
    }
    const ev = data.evals, methods = Object.keys(data.methods);
    const lo = Math.log10(data.floor), all = methods.flatMap(m => data.methods[m].q3);
    const hi = Math.ceil(Math.max(...all, 0) + 0.2);
    el.innerHTML = '';
    const wrap = document.createElement('div'); wrap.className = 'viz-curves'; el.appendChild(wrap);
    const plot = document.createElement('div'); plot.className = 'viz-curves-plot'; wrap.appendChild(plot);
    const legend = document.createElement('div'); legend.className = 'viz-legend'; wrap.appendChild(legend);
    const W = Math.max(420, plot.clientWidth || 700), H = 380;
    const yt = []; for (let v = Math.floor(lo); v <= hi; v += (hi - lo > 10 ? 2 : 1)) yt.push(v);
    const F = frame(plot, { width: W, height: H, x0: 1, x1: ev[ev.length - 1], y0: lo, y1: hi, yTicks: yt,
      yFmt: v => `1e${v}`, xLabel: '評価回数', yLabel: 'f − f*（20 run の中央値）' });
    svg('line', { x1: F.X(1), x2: F.X(ev[ev.length - 1]), y1: F.Y(-10), y2: F.Y(-10), class: 'viz-target' }, F.g);
    svg('text', { x: F.X(ev[ev.length - 1]) - 4, y: F.Y(-10) - 4, class: 'viz-tick', 'text-anchor': 'end' }, F.g).textContent = '1e-10（SR の主指標）';
    const ref = 'MC-ESO';
    if (data.methods[ref]) {
      const q = data.methods[ref];
      const top = ev.map((x, i) => `${F.X(x).toFixed(1)},${F.Y(q.q3[i]).toFixed(1)}`);
      const bot = ev.map((x, i) => `${F.X(x).toFixed(1)},${F.Y(q.q1[i]).toFixed(1)}`).reverse();
      svg('path', { d: `M${top.join('L')}L${bot.join('L')}Z`, fill: colorOf(ref), opacity: 0.16 }, F.g);
    }
    const order = [...methods].sort((a, b) => (a === ref) - (b === ref) || (!!colorOf(a)) - (!!colorOf(b)));
    const lines = {};
    order.forEach(m => {
      const c = colorOf(m) || OTHER;
      lines[m] = svg('path', { d: path(ev, data.methods[m].median, F.X, F.Y), fill: 'none', stroke: c,
        'stroke-width': m === ref ? 2.4 : colorOf(m) ? 1.5 : 1, 'stroke-dasharray': m === 'MC-ESO-v0' ? '5 3' : '',
        class: 'viz-line', 'data-m': m }, F.g);
    });
    // legend ordered by final median
    const fin = m => data.methods[m].median[ev.length - 1];
    const ranked = [...methods].sort((a, b) => (b === ref) - (a === ref) || fin(a) - fin(b));
    legend.innerHTML = '<div class="viz-legend-head">最終値の小さい順</div>' + ranked.map(m =>
      `<button class="viz-legend-item" data-m="${esc(m)}"><i style="background:${colorOf(m) || OTHER}"></i>`
      + `<span>${esc(m)}</span><b>${pow10(fin(m))}</b></button>`).join('');
    const focus = m => {
      Object.entries(lines).forEach(([k, p]) => p.classList.toggle('viz-dim', !!m && k !== m));
      if (m && lines[m]) { lines[m].parentNode.appendChild(lines[m]); lines[m].classList.add('viz-hl'); }
      Object.values(lines).forEach(p => { if (p.dataset.m !== m) p.classList.remove('viz-hl'); });
    };
    legend.addEventListener('mouseover', e => { const b = e.target.closest('[data-m]'); if (b) focus(b.dataset.m); });
    legend.addEventListener('mouseleave', () => focus(null));
    // crosshair + tooltip
    const cross = svg('line', { y1: F.m.t, y2: F.m.t + F.h, class: 'viz-cross', visibility: 'hidden' }, F.g);
    const tip = document.createElement('div'); tip.className = 'viz-tip'; plot.appendChild(tip);
    F.s.addEventListener('mousemove', e => {
      const r = F.s.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
      if (px < F.m.l || px > F.m.l + F.w) { cross.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; return; }
      const xv = 1 + (px - F.m.l) / F.w * (ev[ev.length - 1] - 1);
      let k = 0; while (k < ev.length - 1 && ev[k + 1] <= xv) k++;
      cross.setAttribute('x1', F.X(ev[k])); cross.setAttribute('x2', F.X(ev[k])); cross.setAttribute('visibility', 'visible');
      const rows = methods.map(m => [m, data.methods[m].median[k]]).sort((a, b) => a[1] - b[1]);
      const show = rows.filter(([m], i) => i < 8 || m === ref);
      tip.innerHTML = `<div class="viz-tip-h">${fmtInt(ev[k])} 評価</div>` + show.map(([m, v]) =>
        `<div><i style="background:${colorOf(m) || OTHER}"></i>${esc(m)}<b>${pow10(v)}</b></div>`).join('')
        + (rows.length > show.length ? `<div class="viz-tip-more">ほか ${rows.length - show.length} 手法</div>` : '');
      tip.style.display = 'block';
      const left = (F.X(ev[k]) / W) * r.width;
      tip.style.left = `${left > r.width * 0.6 ? left - tip.offsetWidth - 12 : left + 12}px`;
      tip.style.top = '8px';
    });
    F.s.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; });
    const cap = document.createElement('p'); cap.className = 'viz-caption';
    cap.textContent = `線は ${data.n_runs} run の中央値、帯は MC-ESO の四分位。f − f* = 0 は 1e${lo} に描く。凡例にカーソルを合わせるとその手法だけ強調する。`;
    el.appendChild(cap);
  }

  // ── shared: per-function stats (seed list + recorded best) ──────────────────
  const statsCache = {};
  async function stats(ctx) {
    const key = `${ctx.runId}|${ctx.dim}|${ctx.func}`;
    if (!statsCache[key]) statsCache[key] = getJSON(`/api/stats/${encodeURIComponent(ctx.runId)}/${ctx.dim}/${encodeURIComponent(ctx.func)}`);
    return statsCache[key];
  }
  const replayCache = {};
  function replay(ctx, method, seed) {
    const key = `${ctx.runId}|${ctx.dim}|${ctx.func}|${method}|${seed}`;
    if (!replayCache[key]) {
      replayCache[key] = getJSON(`/api/replay/${encodeURIComponent(ctx.runId)}/${ctx.dim}/${encodeURIComponent(ctx.func)}/${encodeURIComponent(method)}/${seed}`)
        .catch(e => { delete replayCache[key]; throw e; });
    }
    return replayCache[key];
  }
  function seedOptions(rows, method, selected) {
    return rows.filter(r => r.method === method).map(r => {
      const ok = parseFloat(r.best_f) <= 1e-10;
      return `<option value="${r.seed}" ${String(r.seed) === String(selected) ? 'selected' : ''}>seed ${r.seed}  ${ok ? '✓ 到達' : '✕ 未到達'}  f=${fmtE(parseFloat(r.best_f))}</option>`;
    }).join('');
  }
  function verifiedBadge(d) {
    return d.verified
      ? `<span class="viz-badge ok" title="この PC で同じ seed を再実行し、記録された最終値と一致した">再実行・記録と一致</span>`
      : `<span class="viz-badge warn" title="記録 ${fmtE(d.recorded_best_f)} / 再実行 ${fmtE(d.best_f)}。run の後にコードが変わったか、seed で再現しない手法">再実行の結果が記録と違う</span>`;
  }

  // ── 2. search view ──────────────────────────────────────────────────────────
  async function search(el, ctx) {
    const st = await stats(ctx).catch(() => ({ rows: [] }));
    const rows = st.rows || [];
    const methods = ctx.methods.length ? ctx.methods : [...new Set(rows.map(r => r.method))];
    const state = el._state || (el._state = {
      method: methods.includes('MC-ESO') ? 'MC-ESO' : methods[0], seed: null,
      t: 1, a: 0, b: 1, layers: { evals: true, pop: true, path: true }, color: 'channel', playing: false,
    });
    const dimN = parseInt(ctx.dim.replace('dim', ''), 10);
    if (state.seed == null) {
      const mine = rows.filter(r => r.method === state.method);
      state.seed = mine.length ? mine[0].seed : 0;
    }
    el.innerHTML = `
      <div class="viz-controls">
        <label>手法 <select data-k="method">${methods.map(m => `<option ${m === state.method ? 'selected' : ''}>${esc(m)}</option>`).join('')}</select></label>
        <label>run <select data-k="seed">${seedOptions(rows, state.method, state.seed)}</select></label>
        ${dimN > 2 ? `<label>横軸 <select data-k="a">${Array.from({ length: dimN }, (_, i) => `<option value="${i}" ${i === state.a ? 'selected' : ''}>x${i + 1}</option>`).join('')}</select></label>
        <label>縦軸 <select data-k="b">${Array.from({ length: dimN }, (_, i) => `<option value="${i}" ${i === state.b ? 'selected' : ''}>x${i + 1}</option>`).join('')}</select></label>` : ''}
        <span class="viz-layers">
          <label><input type="checkbox" data-l="evals" ${state.layers.evals ? 'checked' : ''}> 評価点</label>
          <label><input type="checkbox" data-l="pop" ${state.layers.pop ? 'checked' : ''}> 集団</label>
          <label><input type="checkbox" data-l="path" ${state.layers.path ? 'checked' : ''}> 最良点の軌跡</label>
        </span>
        <span class="viz-status"></span>
      </div>
      <div class="viz-search"><p class="viz-note">${esc(state.method)} の seed ${state.seed} を再実行しています…</p></div>`;
    const status = el.querySelector('.viz-status');
    const body = el.querySelector('.viz-search');
    el.querySelector('.viz-controls').addEventListener('change', e => {
      const k = e.target.dataset.k, l = e.target.dataset.l;
      if (k === 'method') { state.method = e.target.value; state.seed = null; state.t = 1; }
      else if (k === 'seed') { state.seed = e.target.value; state.t = 1; }
      else if (k === 'a' || k === 'b') { state[k] = parseInt(e.target.value, 10); }
      if (l) { state.layers[l] = e.target.checked; el._redraw && el._redraw(); return; }
      if (state.a === state.b) state.b = (state.a + 1) % dimN;
      search(el, ctx);
    });
    let d, land;
    try {
      [d, land] = await Promise.all([
        replay(ctx, state.method, state.seed),
        getJSON(`/api/landscape/${dimN}/${encodeURIComponent(ctx.func)}?a=${state.a}&b=${state.b}`).catch(() => null),
      ]);
    } catch (e) { note(body, `再実行できませんでした: ${esc(e.message)}`, 'is-error'); return; }
    status.innerHTML = verifiedBadge(d);
    drawSearch(body, el, d, land, state, dimN);
  }

  function drawSearch(body, host, d, land, state, dimN) {
    const X = f32(d.x), lg = f32(d.log_gap), lb = f32(d.log_best_gap), dist = f32(d.dist);
    const ch = d.channel ? bytes(d.channel) : null;
    const n = d.n_evals, D = d.dim, lo = d.bounds[0], hi = d.bounds[1];
    const pop = d.pop, popX = f32(pop.x), popS = f32(pop.sigma), spread = f32(pop.spread);
    const offs = [0]; pop.sizes.forEach(s => offs.push(offs[offs.length - 1] + s));
    const colorBy = ch ? 'channel' : 'age';
    const mcol = colorOf(d.method) || '#3c4a57';
    // best-so-far path (indices where best improves)
    const bestIdx = []; { let b = Infinity; for (let i = 0; i < n; i++) if (lg[i] < b - 1e-12) { b = lg[i]; bestIdx.push(i); } }

    body.innerHTML = `
      <div class="viz-search-grid">
        <div class="viz-map"><canvas></canvas>
          <div class="viz-map-note">${dimN > 2 ? `背景は x${state.a + 1}–x${state.b + 1} 平面の断面（他の座標は最適解の値に固定）。点はその 2 座標への投影。` : '背景は関数の地形（濃いほど f が低い）。☆ = 大域最適。'}</div>
        </div>
        <div class="viz-side">
          <div class="viz-chart" data-c="gap"></div>
          <div class="viz-chart" data-c="dist"></div>
          <div class="viz-chart" data-c="spread"></div>
          <div class="viz-chan-legend"></div>
        </div>
      </div>
      <div class="viz-scrub">
        <button class="btn btn-sm btn-outline viz-play">再生</button>
        <input type="range" min="1" max="${n}" value="${Math.min(state.t > 1 ? state.t : n, n)}" step="1">
        <span class="viz-tlabel"></span>
      </div>`;
    if (state.t <= 1) state.t = n;
    const canvas = body.querySelector('canvas');
    const side = Math.min(560, Math.max(320, (host.clientWidth || body.clientWidth) * 0.5));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = side * dpr; canvas.height = side * dpr; canvas.style.width = canvas.style.height = `${side}px`;
    const cx = canvas.getContext('2d'); cx.scale(dpr, dpr);
    const PAD = 6, S = side - 2 * PAD;
    const px = v => PAD + (v - lo) / (hi - lo) * S, py = v => PAD + (1 - (v - lo) / (hi - lo)) * S;
    const A = dimN > 2 ? state.a : 0, B = dimN > 2 ? state.b : 1;
    // landscape background (cached offscreen)
    let bg = null;
    if (land) {
      // Shade by rank, not value: f spans many decades and a linear (even log)
      // scale washes most of the box out to one colour.
      const L = f32(land.log_f), N = land.n;
      const order = Array.from(L.keys()).sort((p, q) => L[p] - L[q]);
      const rank = new Float32Array(L.length); order.forEach((k, r) => { rank[k] = r / (L.length - 1); });
      bg = document.createElement('canvas'); bg.width = bg.height = N;
      const bc = bg.getContext('2d'), img = bc.createImageData(N, N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const c = ramp(LAND, rank[i * N + j]);
        const o = ((N - 1 - i) * N + j) * 4; img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
      bc.putImageData(img, 0, 0);
    }
    const chanColor = c => (CHANNEL[c] || ['', '#3c4a57'])[1];

    // side charts (built once; a cursor line moves with t)
    const sideW = Math.max(300, body.querySelector('.viz-side').clientWidth || 360);
    const charts = {};
    const mkScatter = (key, values, label, yFmt, yTicks) => {
      const host = body.querySelector(`[data-c="${key}"]`);
      let mn = Infinity, mx = -Infinity; for (const v of values) { if (isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }
      if (!isFinite(mn)) return;
      const F = frame(host, { width: sideW, height: 150, x0: 1, x1: n, y0: mn, y1: mx === mn ? mn + 1 : mx,
        yLabel: label, yFmt, yTicks, m: { l: 46, r: 8, t: 14, b: 22 } });
      // raster the points into an <image> for speed
      const c = document.createElement('canvas'), W2 = F.w, H2 = F.h; c.width = W2 * 2; c.height = H2 * 2;
      const g2 = c.getContext('2d'); g2.scale(2, 2);
      const step = Math.max(1, Math.floor(n / 8000));
      for (let i = 0; i < n; i += step) {
        const v = values[i]; if (!isFinite(v)) continue;
        g2.fillStyle = ch && colorBy === 'channel' ? chanColor(ch[i]) : mcol; g2.globalAlpha = 0.5;
        g2.fillRect((i / (n - 1)) * W2 - 0.8, (1 - (v - mn) / ((mx - mn) || 1)) * H2 - 0.8, 1.6, 1.6);
      }
      svg('image', { href: c.toDataURL(), x: F.m.l, y: F.m.t, width: W2, height: H2, preserveAspectRatio: 'none' }, F.g);
      const cur = svg('line', { y1: F.m.t, y2: F.m.t + F.h, class: 'viz-cursor' }, F.g);
      charts[key] = { F, cur };
    };
    const ticksLog = (a, b) => { const out = []; for (let v = Math.ceil(a); v <= b; v += Math.max(1, Math.round((b - a) / 4))) out.push(v); return out; };
    let gmn = Infinity, gmx = -Infinity; for (const v of lg) { if (v < gmn) gmn = v; if (v > gmx) gmx = v; }
    mkScatter('gap', lg, '評価ごとの f − f*（log）', v => `1e${v}`, ticksLog(gmn, gmx));
    if (dist) {
      const ld = Float32Array.from(dist, v => Math.log10(Math.max(v, 1e-12)));
      let a = Infinity, b = -Infinity; for (const v of ld) { if (v < a) a = v; if (v > b) b = v; }
      mkScatter('dist', ld, '最適解までの距離（log）', v => `1e${v}`, ticksLog(a, b));
    }
    // spread heatmap: coordinates × population frames
    if (spread && pop.sizes.length) {
      const host = body.querySelector('[data-c="spread"]'), nf = pop.sizes.length;
      const F = frame(host, { width: sideW, height: Math.min(150, 40 + D * 12), x0: 1, x1: n, y0: 0, y1: D,
        yTicks: [], yLabel: '集団の広がり（座標ごとの標準偏差、濃いほど大きい）', m: { l: 46, r: 8, t: 14, b: 22 } });
      const c = document.createElement('canvas'); c.width = nf; c.height = D;
      const g2 = c.getContext('2d'), img = g2.createImageData(nf, D);
      let mx = 0; for (const v of spread) if (v > 0) mx = Math.max(mx, Math.log10(v));
      let mnv = 0; for (const v of spread) if (v > 0) mnv = Math.min(mnv, Math.log10(v));
      for (let f = 0; f < nf; f++) for (let j = 0; j < D; j++) {
        const v = spread[f * D + j], t = v > 0 ? (Math.log10(v) - mnv) / ((mx - mnv) || 1) : 0;
        const col = ramp([[246, 245, 240], [134, 170, 169], [14, 75, 70]], t);
        const o = (j * nf + f) * 4; img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
      }
      g2.putImageData(img, 0, 0);
      // columns are not evenly spaced in evals; good enough as an overview
      svg('image', { href: c.toDataURL(), x: F.m.l, y: F.m.t, width: F.w, height: F.h, preserveAspectRatio: 'none', style: 'image-rendering:pixelated' }, F.g);
      for (let j = 0; j < D; j += Math.max(1, Math.ceil(D / 5))) svg('text', { x: F.m.l - 4, y: F.m.t + (j + 0.7) * F.h / D, class: 'viz-tick', 'text-anchor': 'end' }, F.g).textContent = `x${j + 1}`;
      const cur = svg('line', { y1: F.m.t, y2: F.m.t + F.h, class: 'viz-cursor' }, F.g);
      charts.spread = { F, cur };
    }
    if (ch) {
      const used = [...new Set(ch)].filter(c => CHANNEL[c]).sort();
      body.querySelector('.viz-chan-legend').innerHTML = '色 = 子を作った経路: ' + used.map(c =>
        `<span><i style="background:${CHANNEL[c][1]}"></i>${CHANNEL[c][0]}</span>`).join('');
    }

    const slider = body.querySelector('input[type=range]'), tl = body.querySelector('.viz-tlabel');
    function draw() {
      const t = state.t;
      cx.clearRect(0, 0, side, side);
      if (bg) { cx.imageSmoothingEnabled = true; cx.drawImage(bg, PAD, PAD, S, S); }
      else { cx.fillStyle = '#f3f5f4'; cx.fillRect(PAD, PAD, S, S); }
      cx.strokeStyle = RULE; cx.strokeRect(PAD + 0.5, PAD + 0.5, S - 1, S - 1);
      // optimum
      (d.optima || []).forEach(o => star(cx, px(o[A]), py(o[B]), 7, BAD));
      if (state.layers.evals) {
        const recent = Math.max(1, Math.round(n / 50));
        for (let i = 0; i < t; i++) {
          const isNew = i >= t - recent;
          cx.globalAlpha = isNew ? 0.95 : 0.28;
          cx.fillStyle = ch ? chanColor(ch[i]) : mcol;
          const r = isNew ? 2.4 : 1.5;
          cx.fillRect(px(X[i * D + A]) - r / 2, py(X[i * D + B]) - r / 2, r, r);
        }
        cx.globalAlpha = 1;
      }
      if (state.layers.path) {
        cx.beginPath(); let first = true;
        for (const i of bestIdx) { if (i >= t) break; const x = px(X[i * D + A]), y = py(X[i * D + B]); first ? cx.moveTo(x, y) : cx.lineTo(x, y); first = false; }
        cx.strokeStyle = 'rgba(23,32,43,.35)'; cx.lineWidth = 0.8; cx.stroke();
      }
      // population frame nearest to t
      let fi = 0; while (fi < pop.evals.length - 1 && pop.evals[fi + 1] <= t) fi++;
      if (state.layers.pop && pop.sizes.length) {
        for (let k = offs[fi]; k < offs[fi + 1]; k++) {
          const x = px(popX[k * D + A]), y = py(popX[k * D + B]);
          if (popS && D === 2) {
            const r = popS[k] / (hi - lo) * S;
            if (r > 1) { cx.beginPath(); cx.arc(x, y, r, 0, 2 * Math.PI); cx.strokeStyle = 'rgba(0,137,123,.35)'; cx.lineWidth = 1; cx.stroke(); }
          }
          cx.beginPath(); cx.arc(x, y, 3.6, 0, 2 * Math.PI); cx.fillStyle = mcol; cx.fill();
          cx.strokeStyle = '#fff'; cx.lineWidth = 1.2; cx.stroke();
        }
      }
      // current best
      let bi = -1; for (const i of bestIdx) { if (i >= t) break; bi = i; }
      if (bi >= 0) { cx.beginPath(); cx.arc(px(X[bi * D + A]), py(X[bi * D + B]), 4.5, 0, 2 * Math.PI); cx.fillStyle = INK; cx.fill(); cx.strokeStyle = '#fff'; cx.lineWidth = 1.5; cx.stroke(); }
      Object.values(charts).forEach(({ F, cur }) => { const xx = F.X(t); cur.setAttribute('x1', xx); cur.setAttribute('x2', xx); });
      const popN = pop.sizes.length ? pop.sizes[fi] : 0;
      tl.textContent = `${fmtInt(t)} / ${fmtInt(n)} 評価　最良 f − f* = ${pow10(lb[t - 1])}${popN ? `　集団 ${popN}` : ''}`;
    }
    function star(c, x, y, r, col) {
      c.beginPath();
      for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.45 : r; c.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
      c.closePath(); c.strokeStyle = col; c.lineWidth = 1.4; c.stroke();
    }
    host._redraw = draw;
    slider.addEventListener('input', () => { state.t = parseInt(slider.value, 10); draw(); });
    const playBtn = body.querySelector('.viz-play');
    playBtn.addEventListener('click', () => {
      if (state.playing) { state.playing = false; playBtn.textContent = '再生'; return; }
      state.playing = true; playBtn.textContent = '停止';
      if (state.t >= n) state.t = 1;
      const t0 = performance.now(), startT = state.t, dur = 9000 * (1 - startT / n) + 500;
      const tick = now => {
        if (!state.playing || !document.body.contains(canvas)) return;
        state.t = Math.min(n, Math.round(startT + (n - startT) * Math.min(1, (now - t0) / dur)));
        slider.value = state.t; draw();
        if (state.t < n) requestAnimationFrame(tick); else { state.playing = false; playBtn.textContent = '再生'; }
      };
      requestAnimationFrame(tick);
    });
    draw();
  }

  // ── 3. MC-ESO internals ─────────────────────────────────────────────────────
  async function internals(el, ctx) {
    const traced = ctx.methods.filter(m => m.startsWith('MC-ESO'));
    if (!traced.length) { note(el, 'この run には MC-ESO が含まれていません。'); return; }
    const state = el._state || (el._state = { method: traced[0], seed: null });
    let runs = [];
    try { runs = (await getJSON(`/api/mceso-runs/${encodeURIComponent(ctx.runId)}/${ctx.dim}`)).rows || []; } catch (_) {}
    const st = await stats(ctx).catch(() => ({ rows: [] }));
    let mine = runs.filter(r => r.function === ctx.func && r.method === state.method);
    const fromStats = (st.rows || []).filter(r => r.method === state.method);
    if (state.seed == null) state.seed = (mine[0] || fromStats[0] || { seed: 0 }).seed;

    const routeChip = r => { const [lab, col] = ROUTE[r] || [r, MUTED]; return `<span class="viz-route" style="--c:${col}">${esc(lab)}</span>`; };
    const share = r => {
      const parts = [['frac_close', 0], ['frac_droplet', 1], ['frac_airborne', 2], ['frac_reseed', 5]];
      return `<span class="viz-share">${parts.map(([k, c]) => `<i style="width:${(parseFloat(r[k]) || 0) * 100}%;background:${CHANNEL[c][1]}" title="${CHANNEL[c][0]} ${Math.round((parseFloat(r[k]) || 0) * 100)}%"></i>`).join('')}</span>`;
    };
    let tableHtml;
    if (mine.length) {
      const counts = {}; mine.forEach(r => { counts[r.route] = (counts[r.route] || 0) + 1; });
      tableHtml = `
        <div class="viz-route-sum">全 ${mine.length} run のルート: ${Object.entries(counts).map(([r, c]) => `${routeChip(r)} ${c}`).join('　')}</div>
        <div class="viz-runs-wrap"><table class="viz-runs">
          <thead><tr><th>seed</th><th>到達</th><th>最終 f</th><th>ルート</th><th>確定</th><th>スピルオーバー</th><th>盆地乗換え</th><th>経路の割合</th><th>集団</th></tr></thead>
          <tbody>${mine.map(r => {
            const ok = parseFloat(r.best_f) <= 1e-10;
            return `<tr data-seed="${r.seed}" class="${String(r.seed) === String(state.seed) ? 'is-sel' : ''}">
              <td>${r.seed}</td><td class="${ok ? 'ok' : 'ng'}">${ok ? '✓' : '✕'}</td><td>${fmtE(parseFloat(r.best_f))}</td>
              <td>${routeChip(r.route)}</td><td>${r.route_commit_evals ? fmtInt(+r.route_commit_evals) : '—'}</td>
              <td>${r.n_spillover}</td><td>${r.n_basin_switch}</td><td>${share(r)}</td><td>${r.n_pop_first}→${r.n_pop_last}</td></tr>`;
          }).join('')}</tbody></table></div>`;
    } else {
      tableHtml = `<p class="viz-note">この run には全 run の要約（mceso_runs.csv）がありません。run を選ぶと、その run を再実行して内部状態を出します。</p>
        <label class="viz-inline">run <select data-k="seed">${seedOptions(fromStats, state.method, state.seed)}</select></label>`;
    }
    el.innerHTML = `
      <div class="viz-controls">
        ${traced.length > 1 ? `<label>手法 <select data-k="method">${traced.map(m => `<option ${m === state.method ? 'selected' : ''}>${esc(m)}</option>`).join('')}</select></label>` : ''}
        <span class="viz-status"></span>
      </div>
      ${tableHtml}
      <div class="viz-internals"><p class="viz-note">seed ${esc(state.seed)} を再実行しています…</p></div>`;
    el.querySelector('.viz-controls').addEventListener('change', e => {
      if (e.target.dataset.k === 'method') { state.method = e.target.value; state.seed = null; internals(el, ctx); }
    });
    el.querySelector('select[data-k="seed"]')?.addEventListener('change', e => { state.seed = e.target.value; internals(el, ctx); });
    el.querySelector('.viz-runs tbody')?.addEventListener('click', e => {
      const tr = e.target.closest('tr[data-seed]'); if (!tr) return;
      state.seed = tr.dataset.seed; internals(el, ctx);
    });
    const body = el.querySelector('.viz-internals');
    let d;
    try { d = await replay(ctx, state.method, state.seed); }
    catch (e) { note(body, `再実行できませんでした: ${esc(e.message)}`, 'is-error'); return; }
    el.querySelector('.viz-status').innerHTML = verifiedBadge(d);
    if (!d.trace) { note(body, 'この手法は内部状態を記録していません。'); return; }
    drawInternals(body, d, ctx);
  }

  function drawInternals(body, d, ctx) {
    const g = d.trace.gen, routes = d.trace.routes, ev = g.evals, n = d.n_evals;
    const W = Math.max(520, body.clientWidth || 800);
    const dimN = d.dim;
    body.innerHTML = '';
    const evMarks = d.trace.events || [];
    const panels = [];
    const panel = (title, opts, drawFn) => {
      const box = document.createElement('div'); box.className = 'viz-panel'; body.appendChild(box);
      const F = frame(box, { width: W, height: opts.h || 120, x0: 1, x1: n, y0: opts.y0, y1: opts.y1, yTicks: opts.yTicks,
        yFmt: opts.yFmt, yLabel: title, showX: !!opts.showX, xLabel: opts.showX ? '評価回数' : '', m: { l: 64, r: 14, t: 16, b: opts.showX ? 30 : 8 } });
      // events
      evMarks.forEach(([e, kind]) => {
        const cls = kind === 'basin_switch' ? 'viz-ev-switch' : kind === 'exhausted' ? 'viz-ev-exh' : 'viz-ev-spill';
        svg('line', { x1: F.X(e), x2: F.X(e), y1: F.m.t, y2: F.m.t + F.h, class: cls }, F.g);
      });
      drawFn(F);
      panels.push(F);
      return F;
    };
    const logs = a => a.map(v => v == null || v <= 0 ? null : Math.log10(v));
    const rng = a => { const v = a.filter(x => x != null && isFinite(x)); return v.length ? [Math.min(...v), Math.max(...v)] : [0, 1]; };
    const ticks = ([a, b]) => { const o = []; const s = Math.max(1, Math.round((b - a) / 4)); for (let v = Math.ceil(a); v <= b; v += s) o.push(v); return o; };

    // route band (which route, from when)
    const band = document.createElement('div'); band.className = 'viz-routeband'; body.appendChild(band);
    const segs = []; let cur = null, start = 0;
    g.route.forEach((r, i) => { if (r !== cur) { if (cur != null) segs.push([cur, start, ev[i]]); cur = r; start = ev[i]; } });
    if (cur != null) segs.push([cur, start, n]);
    band.innerHTML = `<span class="viz-routeband-h">ルーター</span><div class="viz-routeband-bar" style="margin-left:${64}px;margin-right:14px">${segs.map(([r, a, b]) => {
      const [lab, col] = ROUTE[routes[r]] || ['?', MUTED];
      return `<i style="left:${(a / n) * 100}%;width:${((b - a) / n) * 100}%;background:${col}" title="${lab}（${fmtInt(a)}〜${fmtInt(b)} 評価）"></i>`;
    }).join('')}</div>`;

    const lb = logs(g.best_f.map(v => Math.max(v - (d.f_opt || 0), 1e-12))), r1 = rng(lb);
    panel('最良値 f − f*（log）', { y0: Math.max(r1[0], -12), y1: r1[1], yTicks: ticks([Math.max(r1[0], -12), r1[1]]), yFmt: v => `1e${v}` }, F => {
      svg('path', { d: path(ev, lb.map(v => v == null ? -12 : Math.max(v, -12)), F.X, F.Y), class: 'viz-l', stroke: GOOD }, F.g);
      if (r1[0] <= -10) svg('line', { x1: F.m.l, x2: F.m.l + F.w, y1: F.Y(-10), y2: F.Y(-10), class: 'viz-target' }, F.g);
    });
    const ls = logs(g.sigma), r2 = rng(ls);
    panel('全体の歩幅 σ（log）　灰色 = drilling 中', { y0: r2[0], y1: r2[1], yTicks: ticks(r2), yFmt: v => `1e${v}` }, F => {
      // drilling shading
      let s0 = null;
      g.drilling.forEach((v, i) => {
        if (v && s0 == null) s0 = ev[i];
        if ((!v || i === g.drilling.length - 1) && s0 != null) { svg('rect', { x: F.X(s0), y: F.m.t, width: Math.max(1, F.X(ev[i]) - F.X(s0)), height: F.h, class: 'viz-drill' }, F.g); s0 = null; }
      });
      svg('path', { d: path(ev, ls, F.X, F.Y), class: 'viz-l', stroke: INK }, F.g);
    });
    const np = g.n_pop, r3 = [0, Math.max(...np)];
    panel('集団サイズ', { y0: 0, y1: r3[1], h: 90 }, F => svg('path', { d: path(ev, np, F.X, F.Y), class: 'viz-l', stroke: INK2 }, F.g));
    panel('系統の数', { y0: 0, y1: Math.max(...g.n_elite, 1) + 1, h: 80, yTicks: niceTicks(0, Math.max(...g.n_elite, 1) + 1, 3) }, F =>
      svg('path', { d: path(ev, g.n_elite, F.X, F.Y), class: 'viz-l', stroke: INK2 }, F.g));
    // channel shares (stacked)
    panel('子の作り方の割合（世代ごと）', { y0: 0, y1: 1, h: 110, yTicks: [0, 0.5, 1], yFmt: v => `${v * 100}%` }, F => {
      const keys = [['n_close', 0], ['n_droplet', 1], ['n_air', 2], ['n_mom', 3], ['n_mig', 4]];
      const tot = ev.map((_, i) => keys.reduce((s, [k]) => s + (g[k][i] || 0), 0) || 1);
      let base = ev.map(() => 0);
      const step = Math.max(1, Math.floor(ev.length / 600));
      keys.forEach(([k, c]) => {
        if (!g[k].some(v => v > 0)) return;
        const top = base.map((b, i) => b + (g[k][i] || 0) / tot[i]);
        const up = [], dn = [];
        for (let i = 0; i < ev.length; i += step) { up.push(`${F.X(ev[i]).toFixed(1)},${F.Y(top[i]).toFixed(1)}`); dn.push(`${F.X(ev[i]).toFixed(1)},${F.Y(base[i]).toFixed(1)}`); }
        svg('path', { d: `M${up.join('L')}L${dn.reverse().join('L')}Z`, fill: CHANNEL[c][1], opacity: 0.85 }, F.g);
        base = top;
      });
    });
    // router signals
    const sig = (key, title, lines, fmt) => {
      const v = g[key]; const r = rng(v);
      const y0 = Math.min(r[0], ...lines.map(l => l[0])), y1 = Math.max(r[1], ...lines.map(l => l[0]));
      panel(title, { y0, y1: y1 === y0 ? y0 + 1 : y1, h: 90, yFmt: fmt }, F => {
        lines.forEach(([y, lab], i) => {
          svg('line', { x1: F.m.l, x2: F.m.l + F.w, y1: F.Y(y), y2: F.Y(y), class: 'viz-thr' }, F.g);
          // under the line at the right edge: clear of the panel title above
          svg('text', { x: F.m.l + F.w - 4 - i * 130, y: F.Y(y) + 11, class: 'viz-tick viz-thr-label', 'text-anchor': 'end' }, F.g).textContent = lab;
        });
        const commit = ev[Math.min(TH.commit_gen, ev.length - 1)];
        svg('line', { x1: F.X(commit), x2: F.X(commit), y1: F.m.t, y2: F.m.t + F.h, class: 'viz-commit' }, F.g);
        svg('path', { d: path(ev, v, F.X, F.Y), class: 'viz-l', stroke: INK }, F.g);
      });
    };
    sig('cond', 'ルーター信号: 固有値比 log10(λmax/λmin)', [[TH.cond, `DROPLET の閾値 ${TH.cond}`], [TH.cond_early, `早期確定 ${TH.cond_early}`]], v => v.toFixed(1));
    sig('align', 'ルーター信号: 軸への揃い（algA）', [[TH.align, `CLOSE の条件 ${TH.align}`]], v => v.toFixed(2));
    sig('mgap', 'ルーター信号: 座標方向の隙間（mgap）', [[TH.mgap, `CLOSE の条件 ${TH.mgap}`]], v => v.toFixed(2));
    if (g.cc_cond.some(v => v != null)) {
      const r = rng(g.cc_cond);
      panel('学習共分散の条件数 log10（3 次元以上）', { y0: Math.min(0, r[0]), y1: Math.max(1, r[1]), h: 90, yFmt: v => v.toFixed(1) }, F =>
        svg('path', { d: path(ev, g.cc_cond, F.X, F.Y), class: 'viz-l', stroke: GOOD }, F.g));
    }
    const thr = 300 * (dimN / 2);
    panel('停滞カウンタ（改善の無い評価回数）', { y0: 0, y1: Math.max(...g.no_improve, thr) * 1.05, h: 100, showX: true }, F => {
      svg('line', { x1: F.m.l, x2: F.m.l + F.w, y1: F.Y(thr), y2: F.Y(thr), class: 'viz-thr' }, F.g);
      svg('text', { x: F.m.l + 4, y: F.Y(thr) - 3, class: 'viz-tick' }, F.g).textContent = `スピルオーバーの閾値 ${thr}`;
      svg('path', { d: path(ev, g.no_improve, F.X, F.Y), class: 'viz-l', stroke: INK2 }, F.g);
    });
    const leg = document.createElement('p'); leg.className = 'viz-caption';
    leg.innerHTML = '縦線: <span class="viz-k spill"></span>スピルオーバー　<span class="viz-k switch"></span>盆地の乗換え　<span class="viz-k exh"></span>最初の谷を掘り切った時点　<span class="viz-k commit"></span>ルートの確定点（' + TH.commit_gen + ' 世代）。帯の色: 子の作り方 ' +
      [0, 1, 2].map(c => `<span class="viz-k" style="background:${CHANNEL[c][1]}"></span>${CHANNEL[c][0]}`).join('　');
    body.appendChild(leg);

    // shared crosshair
    const tip = document.createElement('div'); tip.className = 'viz-tip'; body.appendChild(tip);
    const crosses = panels.map(F => svg('line', { y1: F.m.t, y2: F.m.t + F.h, class: 'viz-cross', visibility: 'hidden' }, F.g));
    body.addEventListener('mousemove', e => {
      const s = e.target.closest('svg'); if (!s) return;
      const F = panels.find(p => p.s === s); if (!F) return;
      const r = s.getBoundingClientRect(), xpx = (e.clientX - r.left) * (W / r.width);
      if (xpx < F.m.l || xpx > F.m.l + F.w) return;
      const xv = 1 + (xpx - F.m.l) / F.w * (n - 1);
      let k = 0; while (k < ev.length - 1 && ev[k + 1] <= xv) k++;
      crosses.forEach((c, i) => { const xx = panels[i].X(ev[k]); c.setAttribute('x1', xx); c.setAttribute('x2', xx); c.setAttribute('visibility', 'visible'); });
      const rt = (ROUTE[routes[g.route[k]]] || ['?'])[0];
      tip.innerHTML = `<div class="viz-tip-h">${fmtInt(ev[k])} 評価（${k + 1} 世代目）</div>
        <div>最良 f − f*<b>${fmtE(g.best_f[k])}</b></div><div>σ<b>${fmtE(g.sigma[k])}</b></div>
        <div>集団<b>${g.n_pop[k]}</b></div><div>系統<b>${g.n_elite[k]}</b></div><div>ルート<b>${rt}</b></div>
        <div>接触 / 飛沫 / 空気<b>${g.n_close[k]} / ${g.n_droplet[k]} / ${g.n_air[k]}</b></div>
        <div>固有値比 / algA / mgap<b>${fmtE(g.cond[k])} / ${fmtE(g.align[k])} / ${fmtE(g.mgap[k])}</b></div>`;
      tip.style.display = 'block';
      const br = body.getBoundingClientRect();
      const left = e.clientX - br.left;
      tip.style.left = `${left > br.width * 0.6 ? left - tip.offsetWidth - 14 : left + 14}px`;
      tip.style.top = `${e.clientY - br.top + 10}px`;
    });
    body.addEventListener('mouseleave', () => { crosses.forEach(c => c.setAttribute('visibility', 'hidden')); tip.style.display = 'none'; });
  }

  window.Viz = { curves, search, internals };
})();
