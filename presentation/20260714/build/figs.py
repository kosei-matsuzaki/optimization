"""Generate data figures for the 2026-07-07 progress-report deck.

All numbers are read straight from the two designated result CSVs so the
figures cannot drift from the tables in the outline. Palette matches the deck
(MC-ESO red, muted grays for context).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
ABL = ROOT / "results/20260707_進捗報告データ_変更点ablation/dim2/summary.csv"
CMP = ROOT / "results/20260707_進捗報告データ_既存手法比較_10手法/dim2/summary.csv"
PREV = ROOT / "results/20260518_進捗報告データ/dim2/summary.csv"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

RED = "#C0392B"
RED_DK = "#7C1C11"
DARK = "#1F2733"
GRAY = "#5B6673"    # match build_deck.py GRAY so chart labels = slide caption tone
LGRAY = "#C7CDD4"
BLUE = "#2E6DA4"
GREEN = "#4A8B5C"
TEAL = "#2E8B8B"
AMBER = "#CF8A2B"
PURPLE = "#6B4E9E"
GRN_DK = "#2F6B3E"
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 15,
    # one shared type scale for the data charts (axis labels / ticks / legends);
    # explicit per-call sizes in the chart builders match these values.
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "#B8BFC7",
    "axes.linewidth": 1.0,
    # Embed text as glyph outlines so the SVG→EMF step can't substitute the
    # font (LibreOffice otherwise swaps Arial for a serif fallback). The look
    # stays exactly Arial and everything remains vector.
    "svg.fonttype": "path",
})

# All figures are saved as SVG (vector) and converted to EMF at build time so
# the deck embeds vector art throughout.
#
# Output layout: one SUBFOLDER per slide page — figs/<pNN_slug>/<panel>.svg —
# so it's obvious which page each image belongs to and each panel is a separate
# file that build_deck.py can place & size independently. convert.py knows which
# panels are line charts (→ inkscape, which keeps the line width) vs everything
# else (→ soffice).


def _dst(page, panel, ext="svg"):
    d = OUT / page
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{panel}.{ext}"


def save(fig, page, panel, mode="padded", pad=0.045, transparent=False):
    """Save `fig` to figs/<page>/<panel>.svg. mode: 'padded' (inset for the EMF
    edge-trim), 'tight' (bbox_inches tight), or 'plain'."""
    dst = _dst(page, panel)
    if mode == "tight":
        fig.savefig(dst, bbox_inches="tight", pad_inches=0.03,
                    transparent=transparent)
    elif mode == "plain":
        fig.savefig(dst, transparent=transparent)
    else:  # padded — pull the drawing inward so the EMF edge-trim only eats whitespace
        fig.canvas.draw()
        sp = fig.subplotpars
        fig.subplots_adjust(left=sp.left + pad, right=sp.right - pad,
                            top=sp.top - pad, bottom=sp.bottom + pad)
        fig.savefig(dst, transparent=transparent)
    plt.close(fig)


def _tint(hex_color, amount=0.55):
    """Blend a hex colour toward white by `amount` (0=original, 1=white).
    Solid tints avoid the alpha-fill artifacts LibreOffice adds when it
    rasterizes semi-transparent shapes into EMF."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _save_padded(fig, name, pad=0.06):
    """Save an SVG after pulling the drawing inward. LibreOffice's SVG→EMF
    export trims a thin strip off every edge; the inset keeps titles, axis
    labels and legends clear of that trim so only whitespace is lost."""
    fig.canvas.draw()
    sp = fig.subplotpars
    fig.subplots_adjust(left=sp.left + pad, right=sp.right - pad,
                        top=sp.top - pad, bottom=sp.bottom + pad)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["method"], {})[r["function"]] = r
    return rows


def pct(s):
    return float(str(s).strip("%"))


abl = load(ABL)
cmp = load(CMP)
LADDER = ["abl0_base2018", "abl1_ir", "abl2_floornich", "abl3_router", "MC-ESO"]
# Evaluation policy: aggregate results on 2-D BBOB-24 (F01-F24) only.
# Custom functions (C01-C11) appear solely in the multimodal figures.
funcs = sorted(fn for fn in abl["MC-ESO"] if fn.startswith("F"))


def mean(rows, m, col):
    return sum(pct(rows[m][fn][col]) for fn in funcs) / len(funcs)


# ─────────────────────────────────────────────────────────────────────────
# 1. Waterfall: cumulative SR@1e-10 across the ablation ladder
# ─────────────────────────────────────────────────────────────────────────
def _waterfall_panel(ax, col, ref_method, ylim, title):
    vals = [mean(abl, m, col) for m in LADDER]
    ref = mean(cmp, ref_method, col)
    base = vals[0]
    n = len(vals)
    bw = 0.6
    # best-baseline reference band + line
    ax.axhspan(ylim[0], ref, color="#F4F6F8", zorder=0)
    ax.axhline(ref, color=BLUE, ls=(0, (6, 3)), lw=1.6, zorder=4)
    # label at the right end, just above the line — clear of the bars in both
    # panels (rightmost bars float well above or sit below this level)
    ax.text(n - 0.35, ref + 0.2, f"{ref_method}  {ref:.1f}", fontsize=10.5,
            color=BLUE, fontweight="bold", va="bottom", ha="right", zorder=5)
    # base bar
    ax.bar(0, base, width=bw, color=DARK, zorder=3)
    ax.text(0, base + 0.28, f"{base:.1f}", ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color=DARK)
    prev = base
    for i in range(1, n):
        delta = vals[i] - prev
        is_final = i == n - 1
        up = delta > 0.05
        color = (RED if up else LGRAY) if is_final else ("#E4A9A2" if up else LGRAY)
        bottom = min(prev, vals[i])
        height = abs(delta) if abs(delta) > 0.05 else 0.0
        ax.bar(i, height, bottom=bottom, width=bw, color=color, zorder=3)
        ax.plot([i - 1 + bw / 2, i - bw / 2], [prev, prev], color="#BFC5CC",
                lw=1.1, zorder=2)
        dlab = f"+{delta:.1f}" if up else (f"{delta:.1f}" if delta < -0.05 else "±0.0")
        ax.text(i, max(prev, vals[i]) + 0.28, dlab, ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=(RED if is_final and up else ("#8A5750" if up else GRAY)))
        prev = vals[i]
    # final cumulative value: inside a tall final bar, else above it
    fcol = RED if vals[-1] >= vals[-2] else DARK
    if abs(vals[-1] - vals[-2]) > 1.3:
        ax.text(n - 1, (vals[-2] + vals[-1]) / 2, f"{vals[-1]:.1f}", ha="center",
                va="center", fontsize=14, fontweight="bold", color="white",
                zorder=6)
    else:
        ax.text(n - 1, max(vals[-2], vals[-1]) + 0.95, f"{vals[-1]:.1f}",
                ha="center", va="bottom", fontsize=13.5, fontweight="bold",
                color=fcol, zorder=6)
    ax.set_xticks(range(n))
    ax.set_xticklabels(["base", "+restart", "+floor", "+router", "+best2"],
                       fontsize=11, color=DARK)
    ax.set_xlim(-0.75, n - 0.3)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EBEDF0", zorder=1)
    return vals, ref


def fig_waterfall():
    # two SEPARATE panels so build_deck can place & size each independently
    fig, ax = plt.subplots(figsize=(6.0, 4.7))
    v4, _ = _waterfall_panel(ax, "sr_1e-4", "L-SHADE", (88, 99), "SR @ 1e-4")
    ax.set_ylabel("Mean SR  (BBOB-24, dim 2)", fontsize=12)
    fig.tight_layout()
    save(fig, "p11_waterfall", "sr1e4")
    fig, ax = plt.subplots(figsize=(6.0, 4.7))
    v10, _ = _waterfall_panel(ax, "sr_1e-10", "DE", (82, 95), "SR @ 1e-10")
    ax.set_ylabel("Mean SR  (BBOB-24, dim 2)", fontsize=12)
    fig.tight_layout()
    save(fig, "p11_waterfall", "sr1e10")
    print("waterfall 1e-4:", [round(v, 1) for v in v4],
          "| 1e-10:", [round(v, 1) for v in v10])


# ─────────────────────────────────────────────────────────────────────────
# 2. Shared method-comparison chart (used by both the 5/18 and the now slides):
#    pale bar = SR@1e-4, solid bar = SR@1e-10; MC-ESO red, baselines gray.
# ─────────────────────────────────────────────────────────────────────────
def _bar_colors(m):
    """(pale = 1e-4, solid = 1e-10) per method."""
    if m == "MC-ESO":
        return ("#E8A9A0", RED)
    if m == "MC-ESO-Old":
        return ("#F1D3CE", "#D98880")
    return ("#C9D0D8", "#9AA4AF")


def _order_methods(methods, key10):
    """MC-ESO first, then the rest by SR@1e-10 descending."""
    rest = sorted([m for m in methods if m != "MC-ESO"], key=lambda m: -key10[m])
    return ["MC-ESO"] + rest


def _method_bars(labels, data, name, title, figsize=(9.8, 4.9), rot=25, fs=11,
                 ymin=0, ymax=105):
    """data[label] = (sr_1e4, sr_1e10). Draws the shared paired-bar chart.
    ymin can be raised above 0 to zoom in on the relevant range."""
    import numpy as np
    from matplotlib.patches import Patch
    x = np.arange(len(labels))
    w = 0.38
    off = (ymax - ymin) * 0.012  # label offset scaled to the axis range
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK, pad=10)
    for i, m in enumerate(labels):
        p4, p10 = data[m]
        cp, cs = _bar_colors(m)
        ax.bar(x[i] - w / 2, p4, w, color=cp, zorder=3)
        ax.bar(x[i] + w / 2, p10, w, color=cs, zorder=3)
        ax.text(x[i] + w / 2, p10 + off, f"{p10:.0f}", ha="center", va="bottom",
                fontsize=9.5, color=DARK)
        if m in ("MC-ESO", "MC-ESO-Old"):  # show the 1e-4 value on the MC-ESO bars
            ax.text(x[i] - w / 2, p4 + off, f"{p4:.0f}", ha="center", va="bottom",
                    fontsize=11, color=GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=fs, color=DARK)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Mean SR  (BBOB-24, dim 2)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EBEDF0", zorder=0)
    ax.legend(handles=[Patch(color=_tint(GRAY, 0.35), label="pale  =  SR @ 1e-4"),
                       Patch(color=GRAY, label="solid  =  SR @ 1e-10")],
              loc="upper right", frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, *name)   # name = (page, panel)


def fig_methods():
    prev = load(PREV)
    bb = sorted(f for f in prev["MC-ESO"] if f.startswith("F"))

    def old(col):
        return sum(pct(prev["MC-ESO"][f][col]) for f in bb) / len(bb)

    # curated to the leading contenders; the weaker baselines are covered by
    # the Wilcoxon panel on the slide.
    base = ["MC-ESO", "DE", "IPOP-CMA-ES", "BIPOP-CMA-ES", "L-SHADE"]
    data = {m: (mean(cmp, m, "sr_1e-4"), mean(cmp, m, "sr_1e-10")) for m in base}
    data["MC-ESO-Old"] = (old("sr_1e-4"), old("sr_1e-10"))  # 5/18 MC-ESO
    key10 = {m: v[1] for m, v in data.items()}
    labels = _order_methods(list(data), key10)
    _method_bars(labels, data, ("p29_methods", "methods"),
                 "Success rate — leading methods (now, n=20)",
                 figsize=(6.9, 5.0), rot=25, fs=11.0, ymin=65, ymax=101)
    print("methods 1e-10:", {m: round(data[m][1], 1) for m in labels})


# ── BBOB official 5-group category breakdown (mirrors the web 成績詳細 view) ──
# Category order = increasing difficulty for MC-ESO's story: the two robust ends
# (separable/ill-cond) stay green, the weak-structure/multimodal tail is where
# every method thins out.
CAT_ORDER = ["separable", "moderate-cond", "ill-cond", "multimodal",
             "weak-structure"]
CAT_LABEL = {"separable": "Separable\nF01–05", "moderate-cond": "Moderate\nF06–09",
             "ill-cond": "Ill-cond.\nF10–14", "multimodal": "Multimodal\nF15–19",
             "weak-structure": "Weak-struct.\nF20–24"}


def _cat_cells(method, col, reduce_pct=True):
    """Per-category mean of `col` for `method` over BBOB-24, plus an ALL column.
    Non-finite evals (categories/functions with no successful run) are dropped;
    a category with no finite value returns None (drawn as '—')."""
    import math
    rows = cmp[method]
    per_cat, allv = [], []
    for cat in CAT_ORDER:
        vals = []
        for fn, r in rows.items():
            if not fn.startswith("F") or r.get("category") != cat:
                continue
            v = pct(r[col]) if reduce_pct else None
            if not reduce_pct:
                try:
                    v = float(r[col])
                except ValueError:
                    v = None
                if v is None or not math.isfinite(v):
                    continue
            vals.append(v)
        per_cat.append(sum(vals) / len(vals) if vals else None)
        allv += vals
    per_cat.append(sum(allv) / len(allv) if allv else None)
    return per_cat


def fig_category_matrix():
    """Two aligned heatmaps — SR@1e-10 and evals-to-success, method × BBOB
    category — the deck analogue of the web '成績詳細（カテゴリ別）' view. Rows
    are the 10 methods ordered by overall SR@1e-10; the shared row labels sit
    once on the far left so the two panels read as one table."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    methods = ["MC-ESO", "DE", "IPOP-CMA-ES", "BIPOP-CMA-ES", "L-SHADE",
               "SaVOA", "NM-Restart", "CMA-ES", "PSO", "NCDE"]
    methods.sort(key=lambda m: -_cat_cells(m, "sr_1e-10")[-1])   # by ALL SR desc
    cols = [CAT_LABEL[c] for c in CAT_ORDER] + ["ALL"]

    sr = np.array([_cat_cells(m, "sr_1e-10") for m in methods], dtype=float)
    ev = np.array([[np.nan if v is None else v
                    for v in _cat_cells(m, "evals_succ_mean", reduce_pct=False)]
                   for m in methods], dtype=float)

    # Muted red→yellow→green performance ramp (green = better), matching the
    # web's 緑=良い / 赤=悪い convention.
    RYG = LinearSegmentedColormap.from_list(
        "ryg", ["#CF6A5A", "#E8C56A", "#EFE9CF", "#7FB07E", "#4E8B5C"])

    fig, (axS, axE) = plt.subplots(1, 2, figsize=(11.4, 4.5),
                                   gridspec_kw={"wspace": 0.06})

    def _draw(ax, M, title, cmap, norm_fn, fmt, unit):
        n_r, n_c = M.shape
        ax.set_xlim(0, n_c); ax.set_ylim(0, n_r); ax.invert_yaxis()
        ax.set_aspect("auto")
        for i in range(n_r):
            for j in range(n_c):
                v = M[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#F1F2F4",
                                               edgecolor="white", lw=1.5))
                    ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                            fontsize=10, color="#AEB6BE")
                    continue
                t = norm_fn(v)
                col = cmap(t)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col,
                                           edgecolor="white", lw=1.5))
                lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                ax.text(j + 0.5, i + 0.5, fmt(v), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color=("white" if lum < 0.5 else DARK))
        # separator before the ALL column
        ax.axvline(n_c - 1, color="#8A94A0", lw=1.6)
        ax.set_xticks(np.arange(n_c) + 0.5)
        ax.set_xticklabels(cols[:-1] + ["ALL"], fontsize=9.5, color=DARK)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=14, fontweight="bold", color=DARK, pad=12)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # SR: 0→100 mapped straight onto the ramp (green = high SR)
    _draw(axS, sr, "SR @ 1e-10   (higher = better)", RYG,
          lambda v: np.clip(v / 100, 0, 1), lambda v: f"{v:.0f}", "%")
    # evals: fewer = faster = green. Clip the scale so a couple of slow outliers
    # don't wash out the useful 400–2000 range.
    EV_LO, EV_HI = 400.0, 2400.0
    _draw(axE, ev, "Evals to success   (fewer = faster)", RYG,
          lambda v: np.clip(1 - (v - EV_LO) / (EV_HI - EV_LO), 0, 1),
          lambda v: f"{v:.0f}", "")

    # shared row labels on the far left of the SR panel
    for i, m in enumerate(methods):
        is_mc = (m == "MC-ESO")
        axS.text(-0.18, i + 0.5, m, ha="right", va="center",
                 fontsize=10, fontweight="bold",
                 color=(RED_DK if is_mc else DARK))
    axS.set_yticks([])
    axE.set_yticks([])
    # Keep a whitespace border on all four sides: the SVG→EMF step crops the
    # edges, so tight bbox would clip the far-left row labels, the ALL column,
    # and the two-line category labels (see docs / slide_vector_pipeline note).
    fig.subplots_adjust(left=0.14, right=0.95, top=0.82, bottom=0.17, wspace=0.06)
    save(fig, "p31_category", "matrix", mode="plain")
    print("category matrix:", {m: round(_cat_cells(m, "sr_1e-10")[-1], 1)
                               for m in methods[:3]})


def fig_category_split_matrix():
    """One combined table — rows = the 9 methods (NCDE dropped) ordered by overall
    SR@1e-10, columns = BBOB categories (5 official groups + ALL). Each cell is
    split left/right with no inner gap: left = SR@1e-10 (red→green), right =
    evals-to-success (pale→blue, darker = faster), both carrying their number.
    The legend and the Wilcoxon panel sit beside the grid on the slide."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # Unified comparison set (CMA-ES / BIPOP / L-SHADE / NCDE moved to appendix)
    methods = ["MC-ESO", "IPOP-CMA-ES", "PSO", "DE", "SaVOA", "NM-Restart"]
    methods.sort(key=lambda m: -_cat_cells(m, "sr_1e-10")[-1])   # overall SR desc
    SHORT = {"MC-ESO": "MC-ESO", "IPOP-CMA-ES": "IPOP-CMA-ES", "PSO": "PSO",
             "DE": "DE", "SaVOA": "SaVOA", "NM-Restart": "NM-Restart"}
    col_labels = [CAT_LABEL[c] for c in CAT_ORDER] + ["ALL\nF01–24"]

    sr = np.array([_cat_cells(m, "sr_1e-10") for m in methods], dtype=float)
    ev = np.array([[np.nan if v is None else v
                    for v in _cat_cells(m, "evals_succ_mean", reduce_pct=False)]
                   for m in methods], dtype=float)              # rows = methods

    SR = LinearSegmentedColormap.from_list(
        "sr", ["#CF6A5A", "#E8C56A", "#EFE9CF", "#7FB07E", "#4E8B5C"])
    # Eval on a blue ramp — a hue far from the SR green so the two halves never
    # blur together; darker blue = fewer evals = faster.
    EV = LinearSegmentedColormap.from_list(
        "ev", ["#EEF3F8", "#9DC0E0", "#3D77B0", "#243F63"])       # dark = fast
    EV_LO, EV_HI = 300.0, 2400.0

    def _lum(c):
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    # Per-category winner within each column, computed separately for SR
    # (higher = better) and evals (fewer = better). Ties all count as winners.
    # The ALL column is an overall summary — not ranked.
    WIN_EDGE = RED

    def _rank_map(vals, higher_better):
        vv = [v for v in vals if v is not None
              and not (isinstance(v, float) and np.isnan(v))]
        if not vv:
            return {}
        best = max(vv) if higher_better else min(vv)
        return {i: 1 for i, v in enumerate(vals)
                if v is not None and not (isinstance(v, float) and np.isnan(v))
                and v == best}

    n_r, n_c = sr.shape
    cat_cols = range(n_c - 1)                            # every column but ALL
    srrank = {j: _rank_map(list(sr[:, j]), True) for j in cat_cols}
    evrank = {j: _rank_map(list(ev[:, j]), False) for j in cat_cols}

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    # Header band is embedded inside the y-range (ylim < 0) so the SVG→EMF edge
    # crop can only eat whitespace, never the column headers or the last row.
    ax.set_xlim(0, n_c); ax.set_ylim(-1.05, n_r + 0.05); ax.invert_yaxis()
    ax.set_aspect("auto")
    pad = 0.03                       # cell inset (between cells); no inner SR/eval gap

    def _border(xl, top, hw, rank):
        """Red frame around the per-category winning half-cell (inset slightly so
        the SR and eval frames of one cell keep a hairline gap in the middle)."""
        ax.add_patch(plt.Rectangle((xl + 0.012, top + 0.015), hw - 0.024, 0.85,
                     fill=False, edgecolor=WIN_EDGE, lw=2.4, zorder=5))

    for i in range(n_r):
        for j in range(n_c):
            x0, w = j + pad, 1 - 2 * pad
            hw = w / 2
            top = i + 0.06
            # left half — SR
            s = sr[i, j]
            sc = SR(np.clip(s / 100, 0, 1))
            ax.add_patch(plt.Rectangle((x0, top), hw, 0.88, facecolor=sc,
                                       edgecolor="none"))
            ax.text(x0 + hw / 2, i + 0.5, f"{s:.0f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=("white" if _lum(sc) < 0.5 else DARK))
            if j in srrank and i in srrank[j]:
                _border(x0, top, hw, srrank[j][i])
            # right half — evals (grey hatch if never successful)
            xr = x0 + hw
            e = ev[i, j]
            if np.isnan(e):
                ax.add_patch(plt.Rectangle((xr, top), hw, 0.88,
                             facecolor="#EDEEF0", edgecolor="#D2D6DB",
                             hatch="////", lw=0))
                ax.text(xr + hw / 2, i + 0.5, "—", ha="center", va="center",
                        fontsize=9, color="#AEB6BE")
            else:
                t = np.clip(1 - (e - EV_LO) / (EV_HI - EV_LO), 0, 1)
                ec = EV(t)
                ax.add_patch(plt.Rectangle((xr, top), hw, 0.88, facecolor=ec,
                             edgecolor="none"))
                ax.text(xr + hw / 2, i + 0.5, f"{e:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=("white" if _lum(ec) < 0.5 else DARK))
            if j in evrank and i in evrank[j]:
                _border(xr, top, hw, evrank[j][i])
    ax.axvline(n_c - 1, color="#8A94A0", lw=1.6)     # separate the ALL column
    # column headers (category groups) in the embedded header band
    for j, lab in enumerate(col_labels):
        ax.text(j + 0.5, -0.5, lab, ha="center", va="center", fontsize=9,
                fontweight="bold", color=DARK, linespacing=1.1)
    # row labels (method names)
    for i, m in enumerate(methods):
        ax.text(-0.08, i + 0.5, SHORT[m], ha="right", va="center", fontsize=9.5,
                fontweight="bold", color=(RED_DK if m == "MC-ESO" else DARK))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.955, top=0.91, bottom=0.09)
    save(fig, "p31_category", "catsplit", mode="plain")
    print("cat split matrix: rows", n_r, "cols", n_c)


# ─────────────────────────────────────────────────────────────────────────
# 3. PR up while SR stays pinned at 100% (multimodal / niching)
# ─────────────────────────────────────────────────────────────────────────
def fig_niching_sequence():
    """The niching mechanism, shown on a real multimodal landscape: one run
    captures optima ONE AT A TIME — drill a basin, then (σ-exhausted) restart
    repelled away from what's already found, onto the next optimum. Himmelblau
    (4 optima) makes the sequence legible. For the p34 (multimodality) slide."""
    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    gx = np.linspace(-5, 5, 240)
    X, Y = np.meshgrid(gx, gx)
    F = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
    Z = np.log1p(F)
    # 4 optima, ordered as a plausible capture sequence
    opt = [(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
    fig, ax = plt.subplots(figsize=(4.15, 3.75))
    ax.contourf(X, Y, Z, levels=30, cmap="viridis", zorder=0)
    # repelled-restart hops between consecutive optima (dashed white arrows)
    for a, b in zip(opt[:-1], opt[1:]):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=15, lw=2.0, color="white",
            linestyle=(0, (4, 2)), connectionstyle="arc3,rad=0.18", zorder=5))
    for k, (mx, my) in enumerate(opt, start=1):
        ax.plot(mx, my, marker="*", ms=20, color="#F2C14E", mec=DARK, mew=1.2,
                zorder=6)
        ax.annotate(str(k), (mx, my), xytext=(9, 9),
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=DARK, zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                              ec=DARK, lw=1.0))
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#B8BFC7")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save(fig, "p34_niching", "sequence", mode="plain")
    print("niching sequence: Himmelblau 4-optima capture path")


def fig_pr_vs_sr():
    import numpy as np
    from matplotlib.patches import Patch
    # PR@1e-4 by function, 5/18 base (grey) vs MC-ESO (function colour) — same
    # off/on bar idiom as the per-function SR slides (grey base + Δ label).
    fns = [("C02-SixHumpCamel", "C02\nSix-hump", "K=2", BLUE),
           ("C01-Himmelblau", "C01\nHimmelblau", "K=4", "#E08A2B"),
           ("C03-Shubert", "C03\nShubert", "K=18", GREEN)]
    x = np.arange(len(fns)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (fn, _lab, _k, col) in enumerate(fns):
        a = pct(abl["abl0_base2018"][fn]["pr_1e-4"]) * 100
        b = pct(abl["MC-ESO"][fn]["pr_1e-4"]) * 100
        ax.bar(i - w / 2, a, w, color="#C9D0D8", zorder=3)
        ax.bar(i + w / 2, b, w, color=col, zorder=3)
        ax.text(i + w / 2, b + 1.5, f"{b-a:+.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=RED_DK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}  ({k})" for _, lab, k, _ in fns], fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Peak ratio @ 1e-4  (% of optima found)", fontsize=12)
    # no in-figure title — the slide title/subtitle already carries it
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEF0F3", zorder=0)
    handles = [Patch(color="#C9D0D8", label="5/18 base"),
               Patch(color=DARK, label="MC-ESO")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "p33_pr_vs_sr", "pr")
    print("PR base→MC-ESO:",
          [(fn[:3], round(pct(abl["abl0_base2018"][fn]["pr_1e-4"]) * 100),
            round(pct(abl["MC-ESO"][fn]["pr_1e-4"]) * 100)) for fn, *_ in fns])


# ─────────────────────────────────────────────────────────────────────────
# 4. evals_succ_mean: ill-conditioned valleys converge ~2x faster
# ─────────────────────────────────────────────────────────────────────────
def fig_evals():
    """Evals-to-success across the four changes (ladder), the speed counterpart
    of the SR ladder: five bars per ill-conditioned valley, lower = faster. The
    ×speed-up (base → MC-ESO) is annotated above each group."""
    import numpy as np
    stages = [("abl0_base2018", "base"), ("abl1_ir", "+restart"),
              ("abl2_floornich", "+floor"), ("abl3_router", "+router"),
              ("MC-ESO", "+best2  (MC-ESO)")]
    cols = ["#C9D0D8", "#9AA4AF", "#E0A08C", "#CD6E55", RED]
    fns = [("F02-EllipsoidalSep", "F02"), ("F10-EllipsoidalRot", "F10"),
           ("F11-Discus", "F11"), ("F12-BentCigar", "F12"),
           ("F13-SharpRidge", "F13")]
    x = np.arange(len(fns)); w = 0.17
    fig, ax = plt.subplots(figsize=(12.2, 4.7))
    for j, (m, lab) in enumerate(stages):
        vals = [float(abl[m][fn]["evals_succ_mean"]) for fn, _ in fns]
        ax.bar(x + (j - 2) * w, vals, w, color=cols[j], label=lab, zorder=3)
    # ×speed-up base → MC-ESO above each group
    for i, (fn, _) in enumerate(fns):
        v0 = float(abl["abl0_base2018"][fn]["evals_succ_mean"])
        v1 = float(abl["MC-ESO"][fn]["evals_succ_mean"])
        ax.text(x[i], max(v0, v1) + 60, f"{v0 / v1:.1f}×", ha="center",
                fontsize=12.5, fontweight="bold", color=RED_DK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in fns], fontsize=11)
    ax.set_ylim(0, 1650)
    ax.set_ylabel("Evals to success  (mean / lower = faster)", fontsize=12)
    # no in-figure title — the slide title/subtitle already carries it
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEF0F3", zorder=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "p30_evals", "evals")
    print("evals ladder:", [l for _, l in fns])


# ─────────────────────────────────────────────────────────────────────────
# 4b. Convergence vs existing methods — one function per family weakness
# ─────────────────────────────────────────────────────────────────────────
# Convergence in the style of results/*/dim2/<F>_convergence.svg, but with a
# trimmed method set — the CMA-ES family collapsed to a single representative
# (plain CMA-ES) — and MC-ESO recoloured to the deck's red accent so it reads as
# the highlighted method. Drawn wide so two functions fit the p21 row layout.
_CMP_ORDER = ["IPOP-CMA-ES", "PSO", "DE", "SaVOA", "NM-Restart", "MC-ESO"]
_CMP_COL = {
    "IPOP-CMA-ES": "tab:blue", "PSO": "tab:orange", "DE": "tab:purple",
    "SaVOA": "tab:green", "NM-Restart": "tab:brown", "MC-ESO": RED,
}
_FAM_PAGE = {"a": "p27_family_conv", "b": "p27_family_conv",
             "c": "p28_family_conv", "d": "p28_family_conv"}


def _family_conv_panel(ax, d, tag):
    """The trimmed comparison set, mean curve + semi-transparent ±1σ band, in
    the style of results/*/dim2/<F>_convergence.svg (linear mean±std, floor 1%)."""
    import numpy as np
    g = d["grid"]
    for m in _CMP_ORDER:
        traj = np.maximum(d[f"{tag}_{m}"], 1e-16)   # (nseed, ngrid) gap values
        mean, std = traj.mean(0), traj.std(0)
        lower = np.maximum(mean - std, mean * 0.01)
        upper = mean + std
        mce = (m == "MC-ESO")
        ax.fill_between(g, lower, upper, color=_CMP_COL[m],
                        alpha=(0.22 if mce else 0.18), lw=0, zorder=2)
        ax.semilogy(g, mean, color=_CMP_COL[m], lw=(3.0 if mce else 1.7),
                    label=m, zorder=(6 if mce else 4))
    ax.axhline(1e-10, color="gray", ls="--", lw=0.9, label="target 1e-10",
               zorder=1)
    ax.set_ylim(1e-15, 3e3)
    ax.set_xlim(0, g[-1] * 1.02)
    ax.set_xlabel("evaluations", fontsize=12)
    ax.set_ylabel(r"best $f - f^*$  (log)", fontsize=12)
    ax.grid(True, which="both", color="#E7EAEE", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # legend outside on the right → adds width, keeps the panel short enough
    # that two functions still stack within one slide
    ax.legend(fontsize=10, ncol=1, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False, handlelength=1.5,
              labelspacing=0.5)


def fig_family_conv():
    import numpy as np
    d = np.load(OUT / "family_conv.npz", allow_pickle=True)
    for tag in "abcd":
        page = _FAM_PAGE[tag]
        ext = tuple(float(v) for v in d[f"{tag}_ext"])
        opt, land = d[f"{tag}_opt"], d[f"{tag}_land"]
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        _map2d(ax, land, ext, opt)
        save(fig, page, f"{tag}_map", mode="tight")
        fig = plt.figure(figsize=(3.5, 3.1))
        ax = fig.add_subplot(111, projection="3d")
        _surf3d(ax, land, ext)
        fig.subplots_adjust(left=0.0, right=1.0, top=1.06, bottom=-0.06)
        save(fig, page, f"{tag}_surf", mode="plain")
        # convergence: exported as high-DPI PNG (not EMF) — EMF/vector cannot
        # render the semi-transparent ±1σ bands, it flattens them to opaque.
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
        _family_conv_panel(ax, d, tag)
        fig.savefig(_dst(page, f"{tag}_conv", "png"), dpi=200,
                    bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print("family_conv: 12 panels (map/surf EMF + conv PNG ×4)")


# ─────────────────────────────────────────────────────────────────────────
# 5. Diagnostic ablation (channels vs restart) — from docs/history.md
# ─────────────────────────────────────────────────────────────────────────
def fig_diag():
    labels = ["MC-ESO\n(full)", "No spillover\n(channels only)",
              "Random restart\n(channels → isotropic)"]
    vals = [83.7, 68.6, 48.9]
    colors = [RED, "#B7A7A4", GRAY]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(range(3), vals, color=colors, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=15,
                fontweight="bold", color=DARK)
    ax.annotate("", xy=(1.95, 55), xytext=(0.4, 86),
                arrowprops=dict(arrowstyle="->", color="#B9BEC5", lw=1.5,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(1.5, 88, "channels are the main driver — not restart luck",
            fontsize=11.5, color="#556", ha="center", style="italic")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 95)
    ax.set_ylabel("Mean SR @ 1e-10", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EBEDF0", zorder=0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.99, 0.015, "diagnostic run (6/05): BBOB-24 + Custom-11, n=10",
             ha="right", va="bottom", fontsize=10, color="#8A94A0", style="italic")
    save(fig, "p08_diagnosis", "diag")


# ─────────────────────────────────────────────────────────────────────────
# 5b. Informed restart — real re-seed points on a 2-D Rastrigin landscape,
#     blind uniform restart (before) vs informed restart (after). Data comes
#     from capture_restart.py (a real MC-ESO run).
# ─────────────────────────────────────────────────────────────────────────
from matplotlib.colors import ListedColormap
import numpy as _np
# Landscape colour = the web visualization's viridis, lightened toward white so
# the overlaid scatter points stay legible (and solid, EMF-safe — no alpha).
_vir = plt.get_cmap("viridis")(_np.linspace(0, 1, 256))
_vir[:, :3] = _vir[:, :3] * 0.55 + 0.45
_LAND_CMAP = ListedColormap(_vir)


def _landscape(ax, gxg, gyg, gz, step=1):
    """Light SOLID colour fill + vector contour lines for a 2-D landscape.
    All-vector (no raster, no alpha) so it aligns with the scatter and survives
    the SVG→EMF conversion cleanly."""
    gx, gy, gv = gxg[::step, ::step], gyg[::step, ::step], gz[::step, ::step]
    ax.contourf(gx, gy, gv, levels=10, cmap=_LAND_CMAP, zorder=0)
    ax.contour(gx, gy, gv, levels=10, colors="#AEBBCB", linewidths=0.5,
               zorder=1, alpha=1.0)


def _restart_axis(ax, gz, opt, lo, hi):
    import numpy as np
    n = gz.shape[0]
    xs = np.linspace(lo, hi, n)
    gxg, gyg = np.meshgrid(xs, xs)
    _landscape(ax, gxg, gyg, gz, step=2)
    ax.plot(opt[0], opt[1], marker="*", ms=20, color="#F2C14E", mec=DARK,
            mew=1.0, zorder=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#B8BFC7")


def fig_restart():
    import numpy as np
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    d = np.load(OUT / "ir_data.npz", allow_pickle=True)
    gz = d["gz"]
    old_rs, new_rs, new_tags = d["old_rs"], d["new_rs"], d["new_tags"]
    reservoir, basins = d["reservoir"], d["basins"]
    opt = d["opt"]
    repel_r = float(d["repel_r"]); lo, hi = float(d["lo"]), float(d["hi"])

    def dist_to_opt(pts):
        return float(np.mean(np.linalg.norm(pts - opt, axis=1)))

    # ── before: blind uniform re-seeds scattered everywhere (standalone) ──
    fig, axL = plt.subplots(figsize=(5.3, 5.6))
    _restart_axis(axL, gz, opt, lo, hi)
    axL.scatter(old_rs[:, 0], old_rs[:, 1], s=46, color="#5B6673",
                edgecolor="white", linewidths=0.6, zorder=6)
    d_old = dist_to_opt(old_rs)
    axL.set_title("Before — blind uniform restart", fontsize=14,
                  fontweight="bold", color=DARK, pad=10)
    axL.text(0.5, -0.05, f"re-seeds land anywhere / avg dist to optimum {d_old:.1f}",
             transform=axL.transAxes, ha="center", fontsize=11.5, color="#55606B")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.11)
    save(fig, "p12_restart", "before", mode="plain")

    # ── after: reservoir re-ignition (red) + basin-repelled (blue) (standalone) ──
    fig, axR = plt.subplots(figsize=(5.3, 6.1))
    _restart_axis(axR, gz, opt, lo, hi)
    ub = np.unique(np.round(basins, 1), axis=0)
    for bx, by in ub:
        axR.add_patch(Circle((bx, by), repel_r, fill=False, ls=(0, (3, 2)),
                             ec="#8A94A0", lw=1.1, zorder=3))
        axR.plot(bx, by, marker="x", ms=10, color="#4A5561", mew=2.4, zorder=5)
    rep = new_rs[new_tags == "repelled"]
    res = new_rs[new_tags == "reservoir"]
    if len(rep) > 55:
        rep = rep[:: max(1, len(rep) // 55)]
    if len(res) > 55:
        res = res[:: max(1, len(res) // 55)]
    axR.scatter(rep[:, 0], rep[:, 1], s=34, color=BLUE, edgecolor="white",
                linewidths=0.4, zorder=6)
    axR.scatter(res[:, 0], res[:, 1], s=34, color=RED, edgecolor="white",
                linewidths=0.4, zorder=7)
    axR.scatter(reservoir[:, 0], reservoir[:, 1], marker="D", s=130,
                color="none", edgecolor=RED, linewidths=2.2, zorder=9)
    d_new = dist_to_opt(new_rs[new_tags == "reservoir"])
    axR.set_title("After — informed restart", fontsize=14, fontweight="bold",
                  color=DARK, pad=10)
    axR.text(0.5, -0.05, f"re-ignite good spots, avoid explored / avg dist {d_new:.1f}",
             transform=axR.transAxes, ha="center", fontsize=11.5, color="#55606B")
    handles = [
        Line2D([], [], marker="*", ls="", ms=13, color="#F2C14E", mec=DARK,
               label="global optimum"),
        Line2D([], [], marker="D", ls="", ms=9, mfc="none", mec=RED, mew=2,
               label="reservoir (good spots)"),
        Line2D([], [], marker="x", ls="", ms=9, color="#4A5561", mew=2.4,
               label="abandoned basin"),
        Line2D([], [], marker="o", ls="", ms=8, color=RED, label="re-ignite re-seed"),
        Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="repelled re-seed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, handletextpad=0.3, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.19)
    save(fig, "p12_restart", "after", mode="plain")
    print(f"restart fig: blind avg {d_old:.1f}, informed avg {d_new:.1f}")


# ─────────────────────────────────────────────────────────────────────────
# 5c. Adaptive anisotropy floor (close-contact channel) — real population and
#     the effective Gaussian sampling shape on an ill-conditioned vs a rugged
#     BBOB function. Data from capture_floor.py.
# ─────────────────────────────────────────────────────────────────────────
def _floor_panel(ax, lgz, ext, center, eigvecs, ev, sigma, cond, decision,
                 callout_pos, title, sub, col):
    import numpy as np
    from matplotlib.patches import Ellipse
    gx0, gx1, gy0, gy1 = ext
    ny, nx = lgz.shape
    gxg, gyg = np.meshgrid(np.linspace(gx0, gx1, nx), np.linspace(gy0, gy1, ny))
    _landscape(ax, gxg, gyg, lgz)
    # sample the children the close-contact channel actually draws around the
    # parent: child = parent + σ · (N(0,I) @ transformᵀ), transform = eigvecs·√ev
    transform = eigvecs * np.sqrt(ev)[None, :]
    rng = np.random.default_rng(0)
    off = center + sigma * (rng.standard_normal((150, 2)) @ transform.T)
    ax.scatter(off[:, 0], off[:, 1], s=26, color=col, edgecolor="white",
               linewidths=0.35, zorder=5)
    ang = float(np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1])))
    w = 2 * 2 * sigma * float(np.sqrt(ev[-1]))   # 2σ ellipse
    h = 2 * 2 * sigma * float(np.sqrt(ev[0]))
    ax.add_patch(Ellipse(center, w, h, angle=ang, fill=False, edgecolor=col,
                         lw=2.2, zorder=6))
    ax.plot(center[0], center[1], marker="*", ms=17, color="#F2C14E", mec=DARK,
            mew=1.0, zorder=8)
    # callout: the detection signal (population-covariance eigenvalue ratio)
    exp = int(np.floor(np.log10(cond)))
    mant = cond / 10 ** exp
    cond_txt = (f"λmax/λmin ≈ {cond:.0f}" if cond < 1000
                else f"λmax/λmin ≈ {mant:.0f}×10$^{{{exp}}}$")
    ax.annotate(f"{cond_txt}\n{decision}", xy=center,
                xytext=callout_pos, textcoords="axes fraction", ha="center",
                va="center", fontsize=11.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, lw=1.4,
                          alpha=1.0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                connectionstyle="arc3,rad=0.15"), zorder=10)
    ratio = float(np.sqrt(ev[-1] / ev[0]))
    ax.set_xlim(gx0, gx1); ax.set_ylim(gy0, gy1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#B8BFC7")
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK, pad=10)
    ax.text(0.5, -0.06, f"{sub} / Gaussian axis ratio {ratio:.0f} : 1",
            transform=ax.transAxes, ha="center", fontsize=11.5, color="#55606B")


def _floor_one(d, prefix, cond_key, decision, callout, title, sub, col, panel):
    import numpy as np
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.4, 5.7))
    _floor_panel(ax, d[f"{prefix}_lgz"],
                 (float(d[f"{prefix}_gx0"]), float(d[f"{prefix}_gx1"]),
                  float(d[f"{prefix}_gy0"]), float(d[f"{prefix}_gy1"])),
                 d[f"{prefix}_best"], d[f"{prefix}_eigvecs"], d[f"{prefix}_ev"],
                 float(d[f"{prefix}_sigma"]), float(d[cond_key]),
                 decision, callout, title, sub, col)
    handles = [
        Line2D([], [], marker="*", ls="", ms=12, color="#F2C14E", mec=DARK,
               label="parent (host)"),
        Line2D([], [], marker="o", ls="", ms=7, color="#7A828C",
               label="offspring"),
        Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec="#7A828C",
               mew=2, label="Gaussian 2σ"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, handletextpad=0.3, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.13)
    save(fig, "p15_floor", panel, mode="plain")


def fig_floor():
    import numpy as np
    d = np.load(OUT / "floor_data.npz", allow_pickle=True)
    _floor_one(d, "ill", "ill_ratio", "huge  →  release floor", (0.5, 0.83),
               "Ill-conditioned valley (F10)",
               "release anisotropy → align to the valley", RED, "illcond")
    _floor_one(d, "rug", "rug_ratio", "small  →  clamp floor", (0.24, 0.86),
               "Rugged multimodal (F15)",
               "clamp anisotropy → stay round & cautious", BLUE, "rugged")
    print(f"floor fig: ill ratio {float(d['ill_ratio']):.1e}, "
          f"rug ratio {float(d['rug_ratio']):.1e}")



def fig_router():
    """The per-landscape channel router as a two-step case split, drawn from the
    real committed signals of all 24 BBOB functions (capture_router.py).
      Step 1 — conditioning:   cond > 3            → DROPLET (air → droplet)
      Step 2 — separability:   algA > .965 & mgap > .36 → CLOSE (air → close),
                               else KEEP-AIR (= base, multimodal escape kept).
    Two separate panels so build_deck places them independently."""
    import numpy as np
    d = np.load(OUT / "router_data.npz", allow_pickle=True)
    fid, route = d["fid"], d["route"].astype(str)
    cond, algA, mgap = d["cond"], d["algA"], d["mgap"]
    ct, at, mt = float(d["cond_thresh"]), float(d["align_thresh"]), float(d["mgap_thresh"])
    RCOL = {"droplet": RED, "close": BLUE, "keepair": GREEN}
    rng = np.random.default_rng(0)

    def label(ax, xs, ys, mask, dy):
        for f, x, y in zip(fid[mask], xs[mask], ys[mask]):
            ax.annotate(f"F{int(f):02d}", (x, y), fontsize=7.5, color=DARK,
                        ha="center", va="center", xytext=(0, dy),
                        textcoords="offset points")

    GREY = "#C4CBD2"
    jit = rng.uniform(0.16, 0.84, len(fid))
    isd = route == "droplet"
    nd = ~isd

    # ── DROPLET condition: the conditioning signal (all 24) ──
    fig, ax = plt.subplots(figsize=(2.5, 2.7))
    ax.axvspan(ct, 6.6, color=_tint(RED, 0.82), zorder=0)
    ax.axvline(ct, color=RED, ls=(0, (5, 3)), lw=1.4, zorder=2)
    ax.scatter(cond[~isd], jit[~isd], s=26, color=GREY, edgecolor="white",
               linewidths=0.5, zorder=4)
    ax.scatter(cond[isd], jit[isd], s=32, color=RED, edgecolor="white",
               linewidths=0.5, zorder=5)
    ax.set_title("cond > 3  →  DROPLET", fontsize=9.5, fontweight="bold",
                 color=RED_DK, pad=5)
    ax.set_xlim(-0.3, 6.6); ax.set_ylim(0, 1.02); ax.set_yticks([])
    ax.set_xticks([0, 3, 6])
    ax.set_xlabel("cond = log10(λmax/λmin)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "p20_router", "cond_droplet")

    # ── CLOSE / KEEP-AIR condition: separability of the non-droplet functions ──
    for tag, hl, hcol in [("close", "close", BLUE), ("keepair", "keepair", GREEN)]:
        fig, ax = plt.subplots(figsize=(2.5, 2.7))
        ax.add_patch(plt.Rectangle((at, mt), 1.05 - at, 0.9 - mt,
                     color=_tint(BLUE, 0.82), zorder=0))
        ax.axvline(at, color=BLUE, ls=(0, (5, 3)), lw=1.2, zorder=2)
        ax.axhline(mt, color=BLUE, ls=(0, (5, 3)), lw=1.2, zorder=2)
        m_hl = nd & (route == hl)
        m_ot = nd & (route != hl)
        ax.scatter(algA[m_ot], mgap[m_ot], s=26, color=GREY, edgecolor="white",
                   linewidths=0.5, zorder=4)
        ax.scatter(algA[m_hl], mgap[m_hl], s=32, color=hcol, edgecolor="white",
                   linewidths=0.5, zorder=5)
        tlab = ("in the box → CLOSE" if hl == "close" else "outside → KEEP-AIR")
        ax.set_title(tlab, fontsize=9.5, fontweight="bold", color=hcol, pad=5)
        ax.set_xlim(0.78, 1.012); ax.set_ylim(0.1, 0.9)
        ax.set_xticks([0.8, 0.9, 1.0]); ax.set_yticks([0.2, 0.5, 0.8])
        ax.set_xlabel("algA", fontsize=9); ax.set_ylabel("mgap", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, "p20_router", f"sep_{tag}")

    print("router:", {r: int((route == r).sum()) for r in RCOL})


def fig_router_shapes():
    """One real 2-D BBOB landscape per route, showing the SHAPE that each
    decision threshold selects. Same light landscape style as p12/p16."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from core.benchmarks import _make_bbob
    cases = [("droplet", 10, "F10-Ellipsoidal"),   # ill-conditioned valley
             ("close", 1, "F01-Sphere"),           # separable, axis-aligned
             ("keepair", 3, "F03-Rastrigin")]      # multimodal (base)
    for tag, fid, name in cases:
        bench = _make_bbob(fid, name, "x", 2)
        lo, hi = bench.bounds
        gx = np.linspace(lo, hi, 150)
        X, Y = np.meshgrid(gx, gx)
        Z = np.array([[bench.func(np.array([x, y])) for x in gx] for y in gx])
        Z = np.log1p(Z - Z.min())
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        _landscape(ax, X, Y, Z, step=1)
        oi = np.unravel_index(np.argmin(Z), Z.shape)
        ax.plot(gx[oi[1]], gx[oi[0]], marker="*", ms=15, color="#F2C14E",
                mec=DARK, mew=1.0, zorder=8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#B8BFC7")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        save(fig, "p20_router", f"shape_{tag}", mode="plain")
    print("router shapes: droplet · close · keepair")


def fig_future_cards():
    """Two small icon-figures for the 'next steps' cards (p37): the hard-coded
    constants to generalise, and the dimension ladder. The multimodality card
    reuses the Shubert landscape thumbnail, so it is not drawn here."""
    from matplotlib.patches import FancyBboxPatch
    AMBER_DK, GRN_DK_ = "#8A5A16", "#2F6B3E"

    # ── card 1: hard-coded thresholds → derive from problem scale ────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.98, "hard-coded thresholds — one set per improvement",
            fontsize=8.5, color="#55606B", ha="center", va="top")
    # each row = the constants introduced by one improvement (①/②/③)
    rows = [("1  restart", "300 evals  /  1e-8"),
            ("2  floor", "φ  1e-2 … 1e-3"),
            ("3  router", "cond > 3  /  algA ≥ .965  /  mgap ≥ .36")]
    for (tag, params), y in zip(rows, [0.80, 0.65, 0.50]):
        ax.add_patch(FancyBboxPatch((0.03, y - 0.052), 0.26, 0.104,
                     boxstyle="round,pad=0.004", fc=_tint(AMBER, 0.78),
                     ec=AMBER, lw=1.0))
        ax.text(0.16, y, tag, fontsize=8, color=AMBER_DK, ha="center",
                va="center", fontweight="bold")
        ax.text(0.335, y, params, fontsize=8, color=DARK, ha="left",
                va="center", fontweight="bold")
    ax.annotate("", xy=(0.5, 0.30), xytext=(0.5, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6))
    ax.add_patch(FancyBboxPatch((0.11, 0.05), 0.78, 0.18,
                 boxstyle="round,pad=0.008", fc=_tint(GREEN, 0.82), ec=GREEN,
                 lw=1.4))
    ax.text(0.5, 0.14, "derive from problem scale", fontsize=10.5,
            color=GRN_DK_, ha="center", va="center", fontweight="bold")
    fig.tight_layout(pad=0.1)
    fig.savefig(_dst("p37_future", "params", "png"), dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # ── card 3: at dim3 MC-ESO loses its deep-precision lead ─────────────────
    #   SR@1e-10, BBOB-24 dim3, quick n=20 (results/20260713_165857_dim3_cmp).
    #   IPOP/BIPOP-CMA-ES overtake MC-ESO; the CMA restart family (blue) leads.
    dim3 = [("IPOP-CMA-ES", 65.0), ("BIPOP-CMA-ES", 60.0), ("MC-ESO", 57.3),
            ("DE", 56.0), ("NM-Restart", 54.4), ("CMA-ES", 53.1),
            ("SaVOA", 30.2), ("PSO", 20.6), ("NCDE", 17.9), ("L-SHADE", 17.1)]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    n = len(dim3)
    for i, (m, v) in enumerate(dim3):
        y = n - 1 - i                          # highest at the top
        if m == "MC-ESO":
            col, tc = RED, RED_DK
        elif m in ("IPOP-CMA-ES", "BIPOP-CMA-ES"):
            col, tc = BLUE, BLUE                # CMA restart family = the leaders
        else:
            col, tc = "#C9D0D8", GRAY
        ax.barh(y, v, height=0.72, color=col, zorder=3)
        ax.text(v + 1.5, y, f"{v:.0f}", va="center", ha="left", fontsize=8.5,
                fontweight="bold", color=tc)
    ax.set_yticks(range(n))
    ax.set_yticklabels([m for m, _ in dim3][::-1], fontsize=8.5,
                       color=DARK)
    for t, (m, _) in zip(ax.get_yticklabels()[::-1], dim3):
        if m == "MC-ESO":
            t.set_color(RED_DK); t.set_fontweight("bold")
    ax.set_xlim(0, 78); ax.set_xticks([])
    ax.set_xlabel("SR @ 1e-10  (%)  /  dim 3", fontsize=9, color="#55606B")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.2)
    fig.savefig(_dst("p37_future", "dim", "png"), dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("future cards: params · dim (dim3 method comparison)")


def fig_future_multimodal():
    """Clear icon for the multimodality 'next step' (p37): among several optima,
    only ONE is found (solid gold + green ring); the rest are still missed
    (hollow grey). Reads as 'find every optimum, not just one'."""
    from matplotlib.patches import Circle
    GRN_DK_ = "#2F6B3E"
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    xs, ys = [0.22, 0.5, 0.78], [0.64, 0.30]
    found = (0.5, 0.64)
    for yy in ys:
        for xx in xs:
            if abs(xx - found[0]) < 1e-6 and abs(yy - found[1]) < 1e-6:
                ax.add_patch(Circle((xx, yy), 0.135, fill=False, ec=GREEN,
                                    lw=2.4, zorder=4))
                ax.scatter([xx], [yy], marker="*", s=520, color="#F2C14E",
                           edgecolor=DARK, linewidths=1.1, zorder=6)
            else:
                ax.scatter([xx], [yy], marker="*", s=380, facecolors="#E7EAEE",
                           edgecolors="#9AA4AF", linewidths=1.3, zorder=5)
    ax.text(found[0], found[1] + 0.20, "found", fontsize=11, color=GRN_DK_,
            ha="center", fontweight="bold")
    ax.text(0.5, 0.05, "others still missed", fontsize=11, color="#8A93A0",
            ha="center", fontweight="bold")
    save(fig, "p37_future", "multimodal", mode="plain")
    print("future multimodal: 1 found / rest missed")


def fig_future_robustness():
    """Combined 'Robustness' icon for the p37 middle card: (top) multi-solution
    is still partial — one optimum found, the rest missed; (bottom) noise
    robustness — SR@1e-10 holds through mild noise, drops at severe (exploratory
    2026-07-07 diagnostic: none 92.9 / mild 89.8 / severe 70.2)."""
    from matplotlib.patches import Circle  # noqa: F401 (kept for parity)
    fig = plt.figure(figsize=(2.9, 2.85))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.08], hspace=0.72,
                          left=0.06, right=0.96, top=0.86, bottom=0.11)
    # ── multi-solution: 1 optimum found, the rest missed ──
    axT = fig.add_subplot(gs[0])
    axT.set_xlim(0, 1); axT.set_ylim(0, 1); axT.set_axis_off()
    axT.set_title("MULTI-SOLUTION", fontsize=8, color=GRAY,
                  fontweight="bold", pad=3)
    xs = [0.12, 0.31, 0.5, 0.69, 0.88]
    for i, x in enumerate(xs):
        if i == 2:
            axT.scatter([x], [0.5], marker="o", s=760, facecolors="none",
                        edgecolors=GREEN, linewidths=2.0, zorder=4)
            axT.scatter([x], [0.5], marker="*", s=300, color="#F2C14E",
                        edgecolor=DARK, linewidths=1.0, zorder=6)
        else:
            axT.scatter([x], [0.5], marker="*", s=210, facecolors="#E7EAEE",
                        edgecolors="#9AA4AF", linewidths=1.1, zorder=5)
    axT.text(0.5, -0.14, "finds 1 of many", fontsize=8.5, color="#8A93A0",
             ha="center", va="top", fontweight="bold")
    # ── under noise: SR@1e-10 holds through mild, drops at severe ──
    axB = fig.add_subplot(gs[1])
    lv, sr = ["none", "mild", "severe"], [92.9, 89.8, 70.2]
    cols = [DARK, GREEN, AMBER]
    axB.bar(range(3), sr, width=0.62, color=cols, zorder=3)
    for i, v in enumerate(sr):
        axB.text(i, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color=cols[i])
    axB.set_title("UNDER NOISE — SR@1e-10", fontsize=8, color=GRAY,
                  fontweight="bold", pad=4)
    axB.set_xticks(range(3)); axB.set_xticklabels(lv, fontsize=8.5)
    axB.set_ylim(0, 116); axB.set_yticks([])
    axB.spines[["top", "right", "left"]].set_visible(False)
    axB.tick_params(length=0)
    save(fig, "p37_future", "robustness", mode="plain")
    print("future robustness: multi-solution + noise")


def fig_multimodal_shapes():
    """Landscape thumbnails of the three multi-global test functions with EVERY
    global optimum starred — the visual facade (many optima, few actually
    found). Same light landscape style as the router/floor thumbnails."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from core.benchmarks import make_benchmark_by_name
    cases = [("himmelblau", "C01-Himmelblau"),
             ("sixhump", "C02-SixHumpCamel"),
             ("shubert", "C03-Shubert")]
    for tag, name in cases:
        b = make_benchmark_by_name(name, 2)
        lo, hi = b.bounds
        gx = np.linspace(lo, hi, 200)
        X, Y = np.meshgrid(gx, gx)
        Z = np.array([[b.func(np.array([x, y])) for x in gx] for y in gx])
        Z = np.log1p(Z - Z.min())
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        _landscape(ax, X, Y, Z, step=1)
        opt = np.array(b.optima_pos)
        ax.scatter(opt[:, 0], opt[:, 1], marker="*", s=95, color="#F2C14E",
                   edgecolor=DARK, linewidths=0.8, zorder=8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#B8BFC7")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        save(fig, "p32_multimodal", f"shape_{tag}", mode="plain")
    print("multimodal shapes: himmelblau · sixhump · shubert")


def fig_router_eqs():
    """The router decision as maths, for the p19 slide: each route's committing
    condition with the signal spelled out inline (cond = log10(λmax/λmin) > 3).
    CLOSE tests two signals, so it is stacked over two lines. Each is a tight
    transparent PNG placed at native size, same pipeline as the floor / best2
    equations (CM mathtext, alpha-cropped, DPI re-stamped)."""
    import matplotlib as mpl
    FS = 14                                       # was 12 — bigger for legibility
    with mpl.rc_context({"mathtext.fontset": "cm"}):
        def render(name, lines):
            # keep the inter-line gap tight so a two-line chip stays compact
            fig = plt.figure(figsize=(7.5, 0.46 + 0.42 * len(lines)))
            fig.patch.set_alpha(0)
            n = len(lines)
            for k, eq in enumerate(lines):
                y = 0.5 if n == 1 else (0.72 if k == 0 else 0.28)
                fig.text(0.5, y, eq, ha="center", va="center", fontsize=FS,
                         color=DARK)
            dst = _dst("p20_router", name, "png")
            fig.savefig(dst, dpi=EQ_DPI, transparent=True)
            plt.close(fig)
            _crop_alpha(dst, EQ_DPI)

        # full definitions, each signal spelled out; the ∧ on close is dropped —
        # the two signals just stack (full-width in the card → readable).
        render("dec_droplet",
               [r"$\mathrm{cond}=\log_{10}\dfrac{\lambda_{\max}}{\lambda_{\min}}\ >\ 3$"])
        render("dec_close",
               [r"$\mathrm{algA}=\langle\,\max_j|V_{ij}|\,\rangle_i\ \geq\ .965$",
                r"$\mathrm{mgap}=\max_i\,\mathrm{gap}_i/\mathrm{range}_i\ \geq\ .36$"])
        # KEEP-AIR carries no equation on the slide (= base), so none rendered.
    print("router equations: 3 inline decisions")


def _padlock(ax, x, y, col, s=0.16):
    """Tiny padlock glyph (body + shackle) for the 'locked' state."""
    from matplotlib.patches import Rectangle, Arc
    ax.add_patch(Rectangle((x - 0.7 * s, y - 0.55 * s), 1.4 * s, 1.05 * s,
                 fc=col, ec="none", zorder=8))
    ax.add_patch(Arc((x, y + 0.5 * s), 1.0 * s, 1.1 * s, angle=0, theta1=0,
                 theta2=180, color=col, lw=2.0, zorder=8))


def fig_router_apply():
    """How & when the route is applied (schematic, minimal text). Two panels
    shown side by side:
    (1) WHEN — the route is committed once (early latch or checkpoint), then
        locked (no per-generation flip-flop);
    (2) WHAT — the airborne child-count budget is re-allocated to the routed
        channel (not a probability). The airborne share is NOT a constant: it
        rides a σ-ramp from air_ratio down to 0, and the freed budget lands on
        the routed channel. Drawn as stacked shares against σ so the taper (and
        the snap-back of the droplet route once drilling starts, where
        _channel_ratios returns the base h2h_ratio) is visible.

    Parameters are pulled from the live MCESO signature so the figure cannot
    drift from the implementation."""
    import sys
    import inspect
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from core.optimizers.mceso import MultiChannelEpidemicOptimizer as MCESO
    dflt = {k: v.default for k, v in inspect.signature(MCESO.__init__).parameters.items()}
    AIR_R, H2H_R = dflt["air_ratio"], dflt["h2h_ratio"]
    SIG0, PSR = dflt["sigma"], dflt["precision_sigma_ratio"]

    AIR, DROP, CLOSE = GREEN, RED, BLUE
    PAGE = "p20_router"

    s_hi, s_lo = np.log10(SIG0), np.log10(PSR)

    def shares(rel_sigma, route):
        """(droplet, close, airborne) — mirrors MCESO._channel_ratios."""
        drilling = rel_sigma < PSR
        if drilling or route == "keepair":
            air = 0.0 if drilling else AIR_R
            return H2H_R, 1 - air - H2H_R, air
        t = np.clip((s_hi - np.log10(rel_sigma)) / (s_hi - s_lo), 0.0, 1.0)
        air = AIR_R * (1 - t)
        drop = H2H_R + (AIR_R - air) if route == "droplet" else H2H_R
        return drop, 1 - air - drop, air

    X_END = PSR / 30                         # a little past the drilling threshold
    xs_ramp = np.logspace(np.log10(SIG0), np.log10(PSR), 400)
    xs_drill = np.logspace(np.log10(PSR), np.log10(X_END), 60)
    # the drilling half is tinted (solid, not alpha — LibreOffice mangles alpha)
    TINT = [_tint(c, 0.62) for c in (DROP, CLOSE, AIR)]
    # one compact per-route panel (merged into the p21 router slide, one per
    # column beside that route's condition graph)
    routes = [("droplet", "droplet", "airborne → droplet"),
              ("close", "close", "airborne → close"),
              ("keepair", "keepair", "base split (fixed)")]
    for tag, route, note in routes:
        # the main figure of the p19 column — taller than wide to fill the space
        # below the (fixed-width) card
        fig, ax = plt.subplots(figsize=(3.0, 3.45))
        ramp = np.array([shares(x, route) for x in xs_ramp]).T * 100
        drill = np.array([shares(X_END, route)] * len(xs_drill)).T * 100
        ax.stackplot(xs_ramp, *ramp, colors=[DROP, CLOSE, AIR], zorder=3)
        ax.stackplot(xs_drill, *drill, colors=TINT, zorder=3)
        ax.axvline(PSR, color=DARK, ls=(0, (3, 3)), lw=1.1, zorder=5)
        ax.text(np.sqrt(PSR * X_END), 50, "drilling", rotation=90, ha="center",
                va="center", fontsize=8, color="#55606B", zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(SIG0, X_END); ax.set_ylim(0, 100)
        ax.set_yticks([0, 30, 70, 100])
        ax.set_xlabel("σ / span  (shrinks →)", fontsize=9, color="#55606B")
        ax.set_ylabel("child %", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.set_title(note, fontsize=9.5, fontweight="bold", color=DARK, pad=5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        save(fig, PAGE, f"budget_{tag}")
    print("router_apply: 3 per-route budget panels")


def _sr_change_bars(off_m, on_m, off_lab, title, page, route_npz=None,
                    on_color=RED, on_lab="on"):
    """Per-function SR@1e-10, mechanism off (off_m) vs on (on_m) — ALL functions
    whose SR changed (up or down). Shared by the floor & router result slides."""
    import numpy as np
    from matplotlib.patches import Patch
    rmap = None
    if route_npz:
        dr = np.load(OUT / route_npz, allow_pickle=True)
        rmap = {int(f): r for f, r in zip(dr["fid"], dr["route"].astype(str))}
    RCOL = {"droplet": RED, "close": BLUE, "keepair": GREEN}
    items = [(int(fn[1:3]), fn[:3], pct(abl[off_m][fn]["sr_1e-10"]),
              pct(abl[on_m][fn]["sr_1e-10"]))
             for fn in sorted(f for f in abl[on_m] if f.startswith("F"))
             if pct(abl[off_m][fn]["sr_1e-10"]) != pct(abl[on_m][fn]["sr_1e-10"])]
    x = np.arange(len(items)); w = 0.4
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (fid, short, a, b) in enumerate(items):
        col = RCOL[rmap.get(fid, "keepair")] if rmap else on_color
        ax.bar(i - w / 2, a, w, color="#C9D0D8", zorder=3)
        ax.bar(i + w / 2, b, w, color=col, zorder=3)
        ax.text(i + w / 2, b + 1, f"{b-a:+.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=(RED_DK if b > a else GRAY))
    ax.set_xticks(x); ax.set_xticklabels([s for _, s, _, _ in items], fontsize=11)
    ax.set_ylim(30, 108)
    ax.set_ylabel("SR @ 1e-10  (BBOB, dim 2, n=20)", fontsize=12)
    # no in-figure title — the slide title/subtitle already carries it
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEF0F3", zorder=0)
    if rmap:
        handles = [Patch(color="#C9D0D8", label=off_lab),
                   Patch(color=RED, label="→ droplet"),
                   Patch(color=BLUE, label="→ close"),
                   Patch(color=GREEN, label="→ keep-air")]
    else:
        handles = [Patch(color="#C9D0D8", label=off_lab),
                   Patch(color=on_color, label=on_lab)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False, fontsize=11, ncol=len(handles))
    fig.tight_layout()
    save(fig, page, "sr")
    print(f"{page}:", [(s, f"{a:.0f}->{b:.0f}") for _, s, a, b in items])


def fig_router_result():
    _sr_change_bars("abl2_floornich", "abl3_router", "router off  (abl2)",
                    "Router moves only the functions it routes — net +0.8 pt overall",
                    "p22_router_result", route_npz="router_data.npz")


def fig_floor_result_bar():
    _sr_change_bars("abl1_ir", "abl2_floornich", "floor off  (abl1)",
                    "Adaptive floor — SR@1e-10 by function  (net +3.1 pt overall)",
                    "p18_floor_result_bar", on_color=RED,
                    on_lab="adaptive floor  (abl2)")


def fig_restart_result_bar():
    _sr_change_bars("abl0_base2018", "abl1_ir", "blind restart  (abl0)",
                    "Informed restart — SR@1e-10 by function  (BBOB-24 net ±0; payoff is multimodal)",
                    "p14_restart_bar", on_color=RED,
                    on_lab="informed restart  (abl1)")


def _map2d(ax, land, ext, opt, row_label=None, header=None):
    """2-D top-down function map (full viridis, like the web visualization)."""
    import numpy as np
    gx0, gx1, gy0, gy1 = ext
    ny, nx = land.shape
    gxg, gyg = np.meshgrid(np.linspace(gx0, gx1, nx), np.linspace(gy0, gy1, ny))
    ax.contourf(gxg, gyg, land, levels=30, cmap="viridis", zorder=0)
    ax.plot(opt[0], opt[1], marker="*", ms=13, color="#F2C14E", mec=DARK,
            mew=0.9, zorder=8)
    ax.set_xlim(gx0, gx1); ax.set_ylim(gy0, gy1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#B8BFC7")
    if row_label:
        # horizontal (rotation=0) — EMF mangles rotated outlined text
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center",
                      fontsize=11.5, fontweight="bold", color=DARK, labelpad=8)
    if header:
        ax.set_title(header, fontsize=12.5, color="#55606B", pad=6)


def _surf3d(ax, land, ext, header=None):
    """3-D surface of the function landscape (viridis, like the web view)."""
    import numpy as np
    gx0, gx1, gy0, gy1 = ext
    s = 3
    L = land[::s, ::s]
    ny, nx = L.shape
    X, Y = np.meshgrid(np.linspace(gx0, gx1, nx), np.linspace(gy0, gy1, ny))
    ax.plot_surface(X, Y, L, cmap="viridis", linewidth=0, antialiased=True,
                    rcount=ny, ccount=nx)
    ax.set_axis_off()
    ax.view_init(elev=38, azim=-58)
    try:                       # enlarge the surface within its panel
        ax.set_box_aspect((1, 1, 0.55), zoom=1.5)
    except TypeError:
        ax.set_box_aspect((1, 1, 0.55))
    if header:
        ax.set_title(header, fontsize=12.5, color="#55606B", pad=0)


def _conv_panel(ax, g, off, on, header=None,
                labels=("without floor  (abl1)", "adaptive floor  (abl2)")):
    import numpy as np
    ax.plot(g, off, color="#9AA4AF", lw=2.8, label=labels[0],
            solid_capstyle="round", zorder=3)
    ax.plot(g, on, color=RED, lw=3.0, label=labels[1],
            solid_capstyle="round", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 1e2)
    ax.set_xlim(0, g[-1] * 1.06)   # keep the 5000 tick/marker clear of the trimmed edge
    ax.axhline(1e-10, color="#C6CCD2", ls=(0, (5, 3)), lw=1.3, zorder=1)
    ax.text(g[-1], 1.6e-10, "1e-10 target", ha="right", va="bottom",
            fontsize=10, color=GRAY)
    # the fixed floor STALLS above the target — mark its final value as a miss
    ax.plot(g[-1], off[-1], marker="X", ms=12, color="#7A828C", zorder=6,
            mec="white", mew=1.1)
    ax.annotate("stalls above target", xy=(g[-1], off[-1]),
                xytext=(g[-1] * 0.58, off[-1] * 22), fontsize=10.5,
                color="#5B6673", ha="center",
                arrowprops=dict(arrowstyle="-", color="#9AA4AF", lw=1.1))
    # the adaptive floor drills THROUGH the target — mark where it crosses
    hit = np.where(on <= 1e-10)[0]
    if len(hit):
        ax.plot(g[hit[0]], on[hit[0]], "o", ms=9, color=RED, zorder=7,
                mec="white", mew=1.1)
    ax.set_xlabel("evaluations", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.grid(True, color="#EEF0F3", zorder=0)
    if header:
        ax.set_title(header, fontsize=12.5, color="#55606B", pad=6)


def _conv_panels(npz_name, tags, page,
                 labels=("without floor  (abl1)", "adaptive floor  (abl2)"),
                 legend_loc="upper right"):
    """Per-function 2-D map / 3-D landscape / convergence, one image each, into
    figs/<page>/. Shared by the floor and router 'improved-seed' result slides.
    Conversion: map & surf via soffice; conv via inkscape (keeps the line width).
    """
    import numpy as np
    d = np.load(OUT / npz_name)
    g = d["grid"]
    for tag in tags:
        ext = tuple(float(v) for v in d[f"{tag}_ext"])
        opt, land = d[f"{tag}_opt"], d[f"{tag}_land"]
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        _map2d(ax, land, ext, opt)
        save(fig, page, f"{tag}_map", mode="tight")
        fig = plt.figure(figsize=(3.5, 3.1))
        ax = fig.add_subplot(111, projection="3d")
        _surf3d(ax, land, ext)
        fig.subplots_adjust(left=0.0, right=1.0, top=1.06, bottom=-0.06)
        save(fig, page, f"{tag}_surf", mode="plain")
        fig, ax = plt.subplots(figsize=(8.5, 3.3))
        _conv_panel(ax, g, d[f"{tag}_off"], d[f"{tag}_on"], labels=labels)
        ax.set_ylabel("best  f − f*", fontsize=12)
        ax.legend(frameon=False, fontsize=11, loc=legend_loc)
        save(fig, page, f"{tag}_conv", mode="tight")   # inkscape keeps edges
    print(f"{page}: 6 panels (map/surf/conv ×2)")


def fig_floor_panels():
    _conv_panels("floor_conv.npz", ("f10", "f19"), "p17_floor_result")


def fig_router_conv_panels():
    _conv_panels("router_conv.npz", ("a", "b"), "p21_router_conv",
                 labels=("router off  (abl2)", "router on  (abl3)"))


def fig_restart_conv_panels():
    # blind restart stalls at the top (wrong basin), so lower-left is the clear
    # spot for the legend
    _conv_panels("restart_conv.npz", ("a", "b"), "p13_restart_result",
                 labels=("blind restart  (abl0)", "informed restart  (abl1)"),
                 legend_loc="lower left")


def fig_best2_conv_panels():
    _conv_panels("best2_conv.npz", ("a", "b"), "p24_best2_result",
                 labels=("single difference  (abl3)", "route-gated best2  (MC-ESO)"))


def fig_best2_result_bar():
    _sr_change_bars("abl3_router", "MC-ESO", "single difference  (abl3)",
                    "Route-gated best2 — SR@1e-10 by function  (net +2.1 pt overall)",
                    "p25_best2_bar", on_color=RED,
                    on_lab="route-gated best2  (MC-ESO)")


def fig_ladder_bars():
    """Cumulative per-function SR@1e-10 across the four changes: five bars per
    function (base → +restart → +floor → +router → +best2 = MC-ESO), for every
    function whose SR moved over the whole ladder."""
    import numpy as np
    stages = [("abl0_base2018", "base"), ("abl1_ir", "+restart"),
              ("abl2_floornich", "+floor"), ("abl3_router", "+router"),
              ("MC-ESO", "+best2  (MC-ESO)")]
    cols = ["#C9D0D8", "#9AA4AF", "#E0A08C", "#CD6E55", RED]
    fns = [fn for fn in sorted(f for f in abl["MC-ESO"] if f.startswith("F"))
           if pct(abl["abl0_base2018"][fn]["sr_1e-10"])
           != pct(abl["MC-ESO"][fn]["sr_1e-10"])]
    x = np.arange(len(fns)); w = 0.17
    fig, ax = plt.subplots(figsize=(12.2, 4.7))
    for j, (m, lab) in enumerate(stages):
        vals = [pct(abl[m][fn]["sr_1e-10"]) for fn in fns]
        ax.bar(x + (j - 2) * w, vals, w, color=cols[j], label=lab, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([fn[:3] for fn in fns], fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("SR @ 1e-10  (BBOB, dim 2, n=20)", fontsize=12)
    # no in-figure title — the slide title/subtitle already carries it
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEF0F3", zorder=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False, fontsize=11)
    fig.tight_layout()
    save(fig, "p26_ladder", "sr")
    print("ladder:", [fn[:3] for fn in fns])


# Every individual named in the best2 equation, as bold-italic mathtext (\bm),
# so the diagram labels and the child= equation under it use one convention:
# bold italic = vector, plain italic = scalar (same rule as the floor maths).
EQ_SYM = {k: rf"$\mathbfit{{x}}_{{{s}}}$" for k, s in
          [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("p", "p"),
           ("strain", r"\mathrm{strain}")]}
DONOR_GRAY = "#5B6673"          # donor pair 1 — matches the 1st-difference arrow
EQ_DPI = 220                    # equation PNGs are placed at native size


def _best2_panel(ax, second):
    """DE-mutation over a real ill-conditioned valley (viridis landscape, the
    p12/p16 visual language). Every equation term is an actual population member:
    parent x_p (yellow star), the current elite x_strain sitting deep in the
    valley floor, and the uniform-random donors x_a,x_b (and x_c,x_d). Each
    difference is an arrow *between two individuals*; the child is the vector sum
    (dashed net move). With `second`=True the extra donor pair adds an across-
    valley component (donor diversity) that lifts the child out of the basin."""
    import numpy as np
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    GRAYD = "#3B434D"
    HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():          # same frame as the p12/p16 landscapes
        sp.set_edgecolor("#B8BFC7")
    # ill-conditioned valley (rotated anisotropic bowl), drawn with the SAME
    # light colourmap + grey contour lines + frame as the p12/p16 landscapes.
    gx = np.linspace(-0.05, 1.05, 240)
    X, Y = np.meshgrid(gx, gx)
    u = (X + Y - 1.0) / np.sqrt(2)        # along the valley floor (0 at centre)
    v = (Y - X) / np.sqrt(2)             # across the valley (steep)
    Z = np.log1p(1.0 * u ** 2 + 42.0 * v ** 2)
    ax.contourf(X, Y, Z, levels=10, cmap=_LAND_CMAP, zorder=0,
                vmin=Z.min() - 0.35 * Z.ptp(), vmax=Z.max())   # sit in the teal band
    ax.contour(X, Y, Z, levels=10, colors="#AEBBCB", linewidths=0.5, zorder=1)
    # population members named in the equation (elite deepest in the valley)
    P  = np.array([0.15, 0.12])                   # x_p : parent host (up the floor)
    S  = np.array([0.54, 0.56])                   # x_strain : current elite (valley floor)
    A  = np.array([0.42, 0.37]); Bd = np.array([0.24, 0.33])   # x_a, x_b : donor pair 1
    C  = np.array([0.63, 0.46]); Dd = np.array([0.40, 0.63])   # x_c, x_d : donor pair 2
    bg = np.array([[0.33, 0.29], [0.48, 0.47], [0.61, 0.58], [0.29, 0.41]])
    ax.scatter(bg[:, 0], bg[:, 1], s=22, color="#7B8794", alpha=0.55,
               edgecolors="white", linewidths=0.5, zorder=2)
    with mpl.rc_context({"mathtext.fontset": "cm"}):
        def vec(a, b, col, lw, dashed=False):
            an = ax.annotate("", xy=b, xytext=a, zorder=6,
                             arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                             ls=((0, (5, 3)) if dashed else "-"),
                                             shrinkA=0, shrinkB=0))
            an.arrow_patch.set_path_effects(HALO)
        def member(pt, lab, col, mk="o", ms=9, dx=0.0, dy=0.055,
                   ha="center", va="bottom"):
            ax.plot(*pt, marker=mk, ms=ms, color=col, mec="white", mew=1.3, zorder=7)
            ax.text(pt[0] + dx, pt[1] + dy, lab, fontsize=11, color="#12181F",
                    ha=ha, va=va, zorder=9, path_effects=HALO)
        # best pull toward the elite, then the difference(s) between donors
        vec(P, S, GRAYD, 1.4, dashed=True)
        ax.text(0.245, 0.40, "best pull", fontsize=9.5, color="#12181F",
                fontweight="bold", rotation=42, ha="center", va="bottom",
                zorder=9, path_effects=HALO)
        vec(Bd, A, DONOR_GRAY, 2.2)
        member(A, EQ_SYM["a"], DONOR_GRAY, dx=0.028, ha="left", dy=0.0, va="center")
        member(Bd, EQ_SYM["b"], DONOR_GRAY, dx=-0.028, dy=-0.055, ha="right", va="top")
        if second:
            vec(Dd, C, RED, 2.4)
            member(C, EQ_SYM["c"], RED, dx=0.028, ha="left", dy=0.0, va="center")
            member(Dd, EQ_SYM["d"], RED, dx=-0.028, dy=0.05, ha="right")
        member(S, EQ_SYM["strain"], DARK, ms=11, dx=0.03, ha="left", dy=0.02)
        member(P, EQ_SYM["p"], "#F2C14E", mk="*", ms=19, dx=-0.03, dy=0.0,
               ha="right", va="center")
        # child = vector sum (dashed net move from the parent)
        child = np.array([0.48, 0.50]) if not second else np.array([0.80, 0.30])
        vec(P, child, RED, 1.3, dashed=True)
        ax.plot(*child, "o", ms=13, color=RED, mec="white", mew=1.4, zorder=8)
        ax.text(child[0] + 0.03, child[1] + 0.02, "child", fontsize=11,
                color="#12181F", fontweight="bold", ha="left", va="bottom",
                zorder=9, path_effects=HALO)
        # p16-style call-out on the outcome, curved leader to the child
        note, ncol, nxy = (("2nd donor pair\nescapes the basin", RED, (0.42, 0.87))
                           if second else
                           ("single difference\nstays in the basin", GRAYD, (0.26, 0.88)))
        ax.annotate(note, xy=child, xytext=nxy, fontsize=9.5, color=ncol,
                    fontweight="bold", ha="center", va="center", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ncol, lw=1.3),
                    arrowprops=dict(arrowstyle="-", color=ncol, lw=1.3,
                                    connectionstyle="arc3,rad=-0.25"))


def fig_best2_mechanism():
    """Two panels for the branch: single-difference droplet (off route → base)
    vs the route-gated best2 (2nd difference → escapes). Titles are placed
    natively in the slide; the child= equations are fig_best2_eqs()."""
    for name, second in [("single", False), ("best2", True)]:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        _best2_panel(ax, second)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.03)
        save(fig, "p23_best2", name, mode="plain")
    print("best2 mechanism: single · best2")


def _crop_alpha(path, dpi, pad=4):
    """Crop a transparent PNG down to its non-transparent pixels (+pad px).

    `dpi` must be re-stamped on save: python-pptx reads it to place the picture
    at native size, and Pillow does not carry the pHYs chunk across a re-save.
    """
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    l, t, r, b = im.getchannel("A").getbbox()
    im.crop((max(l - pad, 0), max(t - pad, 0),
             min(r + pad, im.width), min(b + pad, im.height))
            ).save(path, dpi=(dpi, dpi))


def fig_best2_eqs():
    """The two child= equations, as transparent PNGs placed under their panel.

    They cannot be a single mathtext string: each term is colour-coded to the
    arrow that draws it (best pull dark · 1st difference grey · 2nd difference
    red) and mathtext has no \\color. So the terms are packed left-to-right on a
    shared baseline, one colour each, and the row is cropped tight.
    """
    import matplotlib as mpl
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    head = [(r"$\mathrm{child}=\mathbfit{x}_p+F\cdot[$", DARK),
            (r"$(\mathbfit{x}_{\mathrm{strain}}-\mathbfit{x}_p)$", DARK)]
    d1 = [(r"$+$", DARK), (r"$(\mathbfit{x}_a-\mathbfit{x}_b)$", DONOR_GRAY)]
    d2 = [(r"$+$", DARK), (r"$(\mathbfit{x}_c-\mathbfit{x}_d)$", RED)]
    tail = [(r"$]$", DARK)]
    rows = {"eq_single": head + d1 + tail, "eq_best2": head + d1 + d2 + tail}
    with mpl.rc_context({"mathtext.fontset": "cm"}):
        for name, pieces in rows.items():
            fig = plt.figure(figsize=(9, 0.7))
            fig.patch.set_alpha(0)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.patch.set_alpha(0)
            row = HPacker(pad=0, sep=3, align="baseline", children=[
                TextArea(t, textprops=dict(color=c, fontsize=17))
                for t, c in pieces])
            ax.add_artist(AnnotationBbox(row, (0.5, 0.5), frameon=False,
                                         xycoords="axes fraction",
                                         box_alignment=(0.5, 0.5)))
            dst = _dst("p23_best2", name, "png")
            # bbox_inches="tight" would keep the whole (invisible) axes, so the
            # row is cropped afterwards to the ink itself via the alpha channel
            fig.savefig(dst, dpi=EQ_DPI, transparent=True)
            plt.close(fig)
            _crop_alpha(dst, EQ_DPI)
    print("best2 equations: single · best2")


# The six equations of the adaptive anisotropy floor, each rendered as its own
# transparent image so build_deck.py can place it next to NATIVE pptx text (the
# explanation lives on the slide, not baked into the picture). Computer-Modern
# mathtext; bold italic (\mathbfit, i.e. LaTeX \bm) = vector/matrix, plain
# italic = scalar. \boldsymbol is avoided because in mathtext it leaves upper-
# case Greek (Λ) unbolded, so it cannot render the whole equation uniformly.
# Conceptual (not fully rigorous) equations — each shows only what the step
# *does*. eq1/eq4/eq5 are the unchanged close-contact pipeline; eq2/eq3 are the
# derivation of the now-adaptive floor φ. Each rendered tightly cropped so it
# can be dropped into the diagram; the exact coefficients live in the captions.
FLOOR_EQS = [
    r"$\mathbfit{C}=\mathbfit{V}\,\mathbfit{\Lambda}\,\mathbfit{V}^{\top}$",
    r"$r=\lambda_{\max}/\lambda_{\min}\ \ \longrightarrow\ \ "
    r"\rho=\mathrm{EMA}(\log_{10} r)$",
    r"$\varphi(\rho)=\varphi_{hi}\,(=10^{-2})\ \longrightarrow\ "
    r"\varphi_{lo}\,(=10^{-3})$",
    r"$\hat\lambda_i=\max(\tilde\lambda_i,\ \varphi)$",
    r"$\mathbfit{x}'=\mathbfit{x}_p+\sigma_i\,\mathbfit{V}"
    r"\sqrt{\hat{\mathbfit{\Lambda}}}\,\mathbfit{z}$",
]


def fig_floor_math_eqs():
    """Each conceptual equation, tightly cropped on a transparent canvas so it
    can be placed inside the pipeline / detail diagram at a fixed height."""
    import matplotlib as mpl
    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
    with mpl.rc_context({"mathtext.fontset": "cm"}):
        for i, eq in enumerate(FLOOR_EQS, 1):
            if i == 3:
                continue      # rendered below with the (= value) parts shrunk
            fig = plt.figure(figsize=(6, 0.7))
            fig.patch.set_alpha(0)
            fig.text(0.5, 0.5, eq, ha="center", va="center", fontsize=19,
                     color=DARK)
            # PNG placed at NATIVE size on the slide, so every equation keeps
            # the same font regardless of how tall its accents/hats make the crop
            fig.savefig(_dst("p16_floor_math", f"eq{i}", "png"), dpi=200,
                        bbox_inches="tight", pad_inches=0.02, transparent=True)
            plt.close(fig)
        # eq3: φ transition — the parameter values (= 10^-2 / 10^-3) a touch
        # smaller than the symbols, since they are just annotations
        parts = [(r"$\varphi(\rho)=\varphi_{hi}$", 19), (r"$(=10^{-2})$", 14),
                 (r"$\ \longrightarrow\ \varphi_{lo}$", 19), (r"$(=10^{-3})$", 14)]
        pack = HPacker(pad=0, sep=3, align="baseline", children=[
            TextArea(s, textprops=dict(fontsize=fs, color=DARK)) for s, fs in parts])
        fig = plt.figure(figsize=(9, 0.8))
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ab = AnnotationBbox(pack, (0.5, 0.5), xycoords="axes fraction",
                            frameon=False, box_alignment=(0.5, 0.5))
        ab.set_clip_on(False)
        ax.add_artist(ab)
        fig.savefig(_dst("p16_floor_math", "eq3", "png"), dpi=200,
                    bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
    print("floor_math: 5 equation panels")


def fig_floor_illustration():
    """Schematic of the adaptive floor: the sampling ellipse never collapses —
    its short axis is lifted to √φ — and φ ADAPTS to the conditioning (clamped
    high when rugged → rounder/explore, released low when ill-conditioned →
    thin/follow-the-valley). Not to scale; φ exaggerated for visibility. PNG so
    the translucent cloud survives."""
    import matplotlib as mpl
    from matplotlib.patches import Ellipse
    cases = [
        # title, sub, raw half-height (collapsed), floored short half-height, colour
        ("rugged  (multimodal)", r"clamp $\varphi$ high", 0.55, 1.35, GREEN),
        ("ill-conditioned", r"release $\varphi$ low", 0.14, 0.50, RED),
    ]
    RW = 2.9                                   # shared long half-axis
    with mpl.rc_context({"mathtext.fontset": "cm"}):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
        for ax, (title, sub, rh, fh, col) in zip(axes, cases):
            # floored sampling ellipse (solid, tinted) — what we sample from
            ax.add_patch(Ellipse((0, 0), 2 * RW, 2 * fh,
                         facecolor=_tint(col, 0.72), ec=col, lw=2.8, zorder=1))
            # collapsed population spread (dashed) — narrower than the floor
            ax.add_patch(Ellipse((0, 0), 2 * RW, 2 * rh, fill=False,
                         ls=(0, (4, 3)), ec="#6B7480", lw=1.7, zorder=3))
            # minor-axis floor arrow + label, kept clear to the right of centre
            ax.annotate("", xy=(0, fh), xytext=(0, -fh),
                        arrowprops=dict(arrowstyle="<|-|>", color=col, lw=2.4,
                                        mutation_scale=15), zorder=5)
            ax.text(0.18, fh + 0.16, r"$\sqrt{\varphi}$", fontsize=15, color=col,
                    va="bottom", ha="left", zorder=6)
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-1.75, 1.95)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.patch.set_alpha(0)              # transparent → sits on the pink zone
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, fontsize=12, color=DARK, fontweight="bold",
                         pad=6)
        fig.patch.set_alpha(0)
        fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 1))
        fig.savefig(_dst("p16_floor_math", "illus", "png"), dpi=200,
                    bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
    print("floor_math: illustration")


# ─────────────────────────────────────────────────────────────────────────
# 5b. Informed restart (spillover) — explanation page (p16-style)
# ─────────────────────────────────────────────────────────────────────────
def fig_restart_math():
    """Conceptual equations for the informed restart, each a transparent PNG
    placed at native size (like the floor page). s = search-space span."""
    import matplotlib as mpl
    eqs = {
        "reig":  r"$x' \sim \mathcal{N}\,\left(x_{\mathrm{archive}},\ (0.05\,s)^2\right)$",
        "repel": r"$x' \sim \mathrm{Unif}\ \ \mathrm{s.t.}\ \ \|x'-c\| > 0.1\,s$",
    }
    with mpl.rc_context({"mathtext.fontset": "cm"}):
        for name, eq in eqs.items():
            fig = plt.figure(figsize=(6, 0.7))
            fig.patch.set_alpha(0)
            fig.text(0.5, 0.5, eq, ha="center", va="center", fontsize=18,
                     color=DARK)
            fig.savefig(_dst("p12b_restart_math", name, "png"), dpi=200,
                        bbox_inches="tight", pad_inches=0.02, transparent=True)
            plt.close(fig)
        params = (r"$s=\mathrm{span}\quad \mathrm{archive\ fraction}=0.5\quad "
                  r"2\ \mathrm{fails}\to\mathrm{basin\ switch\ (reset}\ \sigma)$")
        fig = plt.figure(figsize=(6, 0.5))
        fig.patch.set_alpha(0)
        fig.text(0.5, 0.5, params, ha="center", va="center", fontsize=13,
                 color="#55606B")
        fig.savefig(_dst("p12b_restart_math", "eqp", "png"), dpi=200,
                    bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
    print("restart_math: 2 equations + params")


def fig_restart_illustration():
    """Schematic of the informed re-seed: half the dead slots re-ignite around
    archived elites (reservoir = good spots), the rest re-seed uniformly but are
    repelled from remembered abandoned basins. The current best is kept. PNG so
    the translucent repel/re-ignite disks survive; sits on the pink zone."""
    import numpy as np
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    rng = np.random.default_rng(3)
    # WIDE frame (aspect equal keeps disks round but the picture stays short)
    SX = 1.75
    ax.set_xlim(0, SX); ax.set_ylim(0, 1.0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    reser = [np.array([0.42, 0.66]), np.array([0.95, 0.78])]
    ab = np.array([0.58, 0.34])
    best = np.array([1.42, 0.42])
    # kept best (protected drilling basin)
    ax.plot(*best, marker="*", ms=20, color="#F2C14E", mec=DARK, mew=1.0, zorder=7)
    ax.text(best[0], best[1] - 0.09, "kept best", fontsize=8.5, ha="center",
            va="top", color=DARK, fontweight="bold")
    # reservoir elites → re-ignite Gaussian clouds (red)
    for c in reser:
        ax.add_patch(Circle(c, 0.11, facecolor=_tint(RED, 0.82), ec=RED,
                            ls=(0, (2, 2)), lw=1.2, zorder=2))
        ax.plot(*c, marker="D", ms=8, color="white", mec=RED, mew=1.7, zorder=5)
        pts = c + rng.normal(0, 0.04, (7, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=13, color=RED, zorder=4, linewidths=0)
    # abandoned basin → repel disk (herd immunity); uniform re-seeds avoid it
    ax.add_patch(Circle(ab, 0.17, facecolor=_tint(GRAY, 0.86), ec="#8A94A0",
                        ls=(0, (3, 3)), lw=1.4, zorder=1))
    ax.plot(*ab, marker="x", ms=11, color="#5B6673", mew=2.4, zorder=5)
    blue = []
    while len(blue) < 10:
        p = rng.uniform([0.08, 0.12], [SX - 0.08, 0.92])
        if (np.hypot(*(p - ab)) > 0.20 and np.hypot(*(p - best)) > 0.13
                and all(np.hypot(*(p - c)) > 0.14 for c in reser)):
            blue.append(p)
    blue = np.array(blue)
    ax.scatter(blue[:, 0], blue[:, 1], s=15, color=BLUE, zorder=4, linewidths=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_alpha(0); fig.patch.set_alpha(0)
    fig.tight_layout(pad=0.2)
    fig.savefig(_dst("p12b_restart_math", "illus", "png"), dpi=200,
                bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    print("restart_math: illustration")


def fig_restart_panels():
    """Two schematic panels for the informed re-seed — one per mode — so the
    p12 slide can use the same two-column composition as the best2 slide
    (condition → formula box → illustration → caption). Both share the kept
    best (protected drilling basin); the left re-ignites around archived
    reservoirs, the right re-seeds uniformly but repelled from a dead basin."""
    import numpy as np
    from matplotlib.patches import Circle, Rectangle
    import matplotlib.patheffects as pe
    HALO = [pe.withStroke(linewidth=2.4, foreground="white")]

    def _frame():
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((0, 0), 1, 1, fc="#F7F8FA", ec="none", zorder=0))
        for sp in ax.spines.values():
            sp.set_edgecolor("#C6CDD4")
        # kept best — protected, shared by both panels for continuity
        best = np.array([0.74, 0.16])
        ax.plot(*best, marker="*", ms=22, color="#F2C14E", mec=DARK, mew=1.0,
                zorder=7)
        ax.text(best[0], best[1] + 0.10, "kept best", fontsize=10, ha="center",
                va="bottom", color=DARK, fontweight="bold", zorder=8,
                path_effects=HALO)
        return fig, ax, best

    # ── Panel A: re-ignite near the reservoir ───────────────────────────────
    fig, ax, best = _frame()
    rng = np.random.default_rng(3)
    reser = [np.array([0.30, 0.60]), np.array([0.66, 0.74])]
    for c in reser:
        ax.add_patch(Circle(c, 0.15, facecolor=_tint(RED, 0.80), ec=RED,
                            ls=(0, (2, 2)), lw=1.4, zorder=2))
        ax.plot(*c, marker="D", ms=10, color="white", mec=RED, mew=1.9, zorder=5)
        pts = c + rng.normal(0, 0.055, (8, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=20, color=RED, zorder=4, linewidths=0)
    ax.annotate("reservoir\n(good spots)", xy=(reser[0][0] - 0.06, reser[0][1]),
                xytext=(0.13, 0.90), fontsize=10, color=RED, fontweight="bold",
                ha="center", va="center", zorder=8, path_effects=HALO,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    fig.tight_layout(pad=0.2)
    fig.savefig(_dst("p12b_restart_math", "panel_reig", "png"), dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # ── Panel B: repelled uniform (herd immunity) ───────────────────────────
    fig, ax, best = _frame()
    rng = np.random.default_rng(5)
    ab = np.array([0.40, 0.56])
    ax.add_patch(Circle(ab, 0.21, facecolor=_tint(GRAY, 0.84), ec="#8A94A0",
                        ls=(0, (3, 3)), lw=1.6, zorder=1))
    ax.plot(*ab, marker="x", ms=14, color="#5B6673", mew=2.8, zorder=5)
    ax.annotate("abandoned\nbasin", xy=(ab[0] + 0.02, ab[1] + 0.19),
                xytext=(0.30, 0.92), fontsize=10, color="#5B6673",
                fontweight="bold", ha="center", va="center", zorder=8,
                path_effects=HALO,
                arrowprops=dict(arrowstyle="->", color="#8A94A0", lw=1.3))
    blue = []
    while len(blue) < 12:
        p = rng.uniform([0.08, 0.10], [0.92, 0.92])
        if (np.hypot(*(p - ab)) > 0.25 and np.hypot(*(p - best)) > 0.14):
            blue.append(p)
    blue = np.array(blue)
    ax.scatter(blue[:, 0], blue[:, 1], s=22, color=BLUE, zorder=4, linewidths=0)
    fig.tight_layout(pad=0.2)
    fig.savefig(_dst("p12b_restart_math", "panel_repel", "png"), dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("restart_panels: re-ignite · repelled")


# ─────────────────────────────────────────────────────────────────────────
# 6. Three transmission channels — one schematic per channel (no text labels;
#    titles/captions live in build_deck.py). Reprise of the 5/18 figure.
# ─────────────────────────────────────────────────────────────────────────
STAR_KW = dict(marker="*", s=360, color=DARK, zorder=6, edgecolor="white",
               linewidths=0.8)


def _channel_axes():
    """One schematic cell of the 3x2 method grid.

    The drawings are all authored in the unit box, but the cell they sit in is
    landscape (a 3x2 grid on a 16:9 slide), so the x-limits are widened past
    the unit box and the content just centres itself in the resulting frame.
    Figure size is kept close to the size it is placed at, so the 10-11 pt
    labels inside stay ~10 pt on the slide.
    """
    import numpy as np  # noqa
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.set_xlim(-0.36, 1.36); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#D6DBE0")
    # top/bottom headroom: the equal-aspect box is height-limited and would
    # touch the canvas edges, so its horizontal borders get trimmed by the
    # SVG→EMF export without this margin.
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)
    return fig, ax


def fig_channel_contact():
    import numpy as np
    from matplotlib.patches import Ellipse
    fig, ax = _channel_axes()
    px, py = 0.5, 0.48
    ang = 30.0  # population covariance is anisotropic → tilted ellipse, not a circle
    ax.add_patch(Ellipse((px, py), 0.70, 0.34, angle=ang, fill=False,
                         ls=(0, (5, 4)), ec=BLUE, lw=1.7))
    loc = np.array([(-0.22, -0.03), (-0.11, 0.03), (-0.02, -0.05), (0.05, 0.02),
                    (0.15, -0.03), (0.26, 0.05), (-0.30, 0.02)])
    th = np.radians(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = loc @ R.T
    ax.scatter(px + pts[:, 0], py + pts[:, 1], s=130, color=BLUE, zorder=5,
               edgecolor="white", linewidths=0.8)
    ax.scatter([px], [py], **STAR_KW)
    ex, ey = px + 0.33 * np.cos(th), py + 0.33 * np.sin(th)
    ax.annotate("shaped by\npopulation", xy=(ex, ey), xytext=(px + 0.10, py + 0.36),
                fontsize=10.5, color=BLUE, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.annotate("parent", xy=(px, py), xytext=(px + 0.02, py - 0.34), fontsize=11,
                color=DARK, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=DARK, lw=1.0))
    save(fig, "p04_channels", "contact", mode="plain")
    plt.close(fig)


def fig_channel_droplet():
    fig, ax = _channel_axes()
    ax.scatter([0.2], [0.2], **STAR_KW)
    ax.scatter([0.82], [0.82], marker="*", s=360, color="#E38A7E", zorder=6,
               edgecolor="white", linewidths=0.8)
    ax.annotate("", xy=(0.74, 0.74), xytext=(0.24, 0.24),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=3.4))
    ax.scatter([0.52], [0.52], s=155, color=RED, zorder=7,
               edgecolor="white", linewidths=0.9)
    ax.text(0.82, 0.9, "elite strain", fontsize=11, color="#C85A4C",
            ha="center", fontweight="bold")
    ax.text(0.60, 0.5, "child", fontsize=10.5, color=DARK, ha="left")
    ax.text(0.2, 0.1, "parent", fontsize=11, color=DARK, ha="center",
            fontweight="bold")
    save(fig, "p04_channels", "droplet", mode="plain")
    plt.close(fig)


def fig_channel_airborne():
    import numpy as np
    from matplotlib.patches import Circle
    fig, ax = _channel_axes()
    hx, hy = 0.5, 0.5
    ax.add_patch(Circle((hx, hy), 0.4, fill=False, ls=(0, (5, 4)),
                        ec=GREEN, lw=1.7))
    ang = np.array([0.35, 0.9, 1.7, 2.3, 3.3, 4.3, 5.0, 5.7])
    rad = np.array([0.30, 0.37, 0.22, 0.34, 0.39, 0.30, 0.36, 0.31])
    ax.scatter(hx + rad * np.cos(ang), hy + rad * np.sin(ang), s=130,
               color=GREEN, zorder=5, edgecolor="white", linewidths=0.8)
    ax.scatter([hx], [hy], **STAR_KW)
    ax.annotate("broadcast\nradius", xy=(hx - 0.28, hy + 0.28),
                xytext=(hx - 0.22, hy + 0.40), fontsize=10.5, color=GREEN,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.annotate("random host", xy=(hx, hy), xytext=(1.12, 0.16),
                fontsize=11, color=DARK, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=DARK, lw=1.0))
    save(fig, "p04_channels", "airborne", mode="plain")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 7. Three population mechanisms — one schematic per mechanism (labels drawn
#    in build_deck.py, same style as the channel schematics).
# ─────────────────────────────────────────────────────────────────────────
def fig_mech_strain():
    from matplotlib.patches import Circle
    fig, ax = _channel_axes()
    elites = [(0.26, 0.36), (0.75, 0.32), (0.52, 0.76)]
    clusters = [[(-0.05, 0.04), (0.05, 0.02), (0.01, -0.06)],
                [(-0.05, -0.03), (0.05, 0.03), (0.0, 0.06)],
                [(-0.06, 0.0), (0.05, 0.04), (0.02, -0.05)]]
    for (ex, ey), cl in zip(elites, clusters):
        ax.add_patch(Circle((ex, ey), 0.14, fill=False, ls=(0, (4, 3)),
                            ec=TEAL, lw=1.4))
        for dx, dy in cl:
            ax.scatter([ex + dx], [ey + dy], s=70, color=TEAL, zorder=4,
                       edgecolor="white", linewidths=0.6)
        ax.scatter([ex], [ey], marker="*", s=300, color=TEAL, zorder=6,
                   edgecolor="white", linewidths=0.8)
    # min-separation double arrow between two elites
    ax.annotate("", xy=(0.63, 0.34), xytext=(0.38, 0.35),
                arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.3))
    ax.text(0.5, 0.15, "kept apart", fontsize=10.5, color=DARK, ha="center",
            va="center", fontweight="bold")
    save(fig, "p05_mechanisms", "strain", mode="plain")
    plt.close(fig)


def fig_mech_restart():
    """Spillover = the restart. As of 5/18 the re-seed was a blind uniform draw
    (§3 replaces it with the informed one), so that is what the recap shows:
    the stalled basin is abandoned, the best is kept, the rest land anywhere."""
    import numpy as np
    from matplotlib.patches import Circle
    fig, ax = _channel_axes()
    # the stalled basin, greyed out, with the incumbent best kept
    ax.add_patch(Circle((0.20, 0.50), 0.15, fill=False, ls=(0, (4, 3)),
                        ec=GRAY, lw=1.4))
    for dx, dy in [(-0.04, 0.03), (0.05, -0.02), (0.0, 0.06), (0.03, -0.06)]:
        ax.scatter([0.20 + dx], [0.50 + dy], s=62, color=LGRAY, zorder=4,
                   edgecolor="white", linewidths=0.6)
    ax.scatter([0.20], [0.50], marker="*", s=280, color=DARK, zorder=6,
               edgecolor="white", linewidths=0.8)
    ax.text(0.20, 0.22, "stalled / keep best", fontsize=10, color=GRAY,
            ha="center", va="center", fontweight="bold")
    # the re-seed: blind uniform, anywhere in the box
    ax.annotate("", xy=(0.52, 0.52), xytext=(0.38, 0.51),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=3.0))
    ax.text(0.45, 0.63, "re-seed", fontsize=10.5, color=PURPLE, ha="center",
            fontweight="bold")
    rng = np.random.default_rng(7)
    pts = rng.uniform([0.62, 0.26], [1.30, 0.92], size=(11, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=62, color=PURPLE, zorder=5,
               edgecolor="white", linewidths=0.6)
    ax.text(0.96, 0.13, "blind uniform", fontsize=10.5, color=PURPLE,
            ha="center", va="center", fontweight="bold")
    save(fig, "p05_mechanisms", "restart", mode="plain")
    plt.close(fig)


def fig_mech_drilling():
    """σ contracts hard once the outbreak is inside one basin, and the airborne
    channel is silenced so its wide noise cannot spoil the precision."""
    from matplotlib.patches import Circle
    fig, ax = _channel_axes()
    cx, cy = 0.36, 0.58
    for r, a in [(0.26, 0.28), (0.17, 0.50), (0.10, 0.72), (0.05, 1.0)]:
        ax.add_patch(Circle((cx, cy), r, fill=False, ls=(0, (3, 2)),
                            ec=AMBER, lw=1.5, alpha=a))
    ax.scatter([cx], [cy], marker="*", s=300, color=AMBER, zorder=6,
               edgecolor="white", linewidths=0.8)
    ax.annotate("", xy=(cx + 0.06, cy), xytext=(cx + 0.25, cy),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.4))
    ax.text(cx, 0.16, "σ × 0.85 per failed generation", fontsize=10.5,
            color=AMBER, ha="center", va="center", fontweight="bold")
    # the airborne channel is switched off while drilling
    for px, py in [(0.98, 0.72), (1.12, 0.60)]:
        ax.scatter([px], [py], marker="o", s=70, color=GREEN, alpha=0.30,
                   zorder=4, edgecolor="white", linewidths=0.6)
    ax.plot([0.99, 1.11], [0.72, 0.60], color=RED, lw=2.0, zorder=6)
    ax.plot([0.99, 1.11], [0.60, 0.72], color=RED, lw=2.0, zorder=6)
    ax.text(1.05, 0.44, "airborne off", fontsize=10.5, color=GREEN,
            ha="center", va="center", fontweight="bold")
    save(fig, "p05_mechanisms", "drilling", mode="plain")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 8. 5/18 standing — SR across precision thresholds, MC-ESO vs baselines
# ─────────────────────────────────────────────────────────────────────────
def fig_prev518():
    prev = load(PREV)
    bb = sorted(f for f in prev["MC-ESO"] if f.startswith("F"))

    def mm(meth, col):
        return sum(pct(prev[meth][f][col]) for f in bb) / len(bb)

    methods = ["MC-ESO", "DE", "SaVOA", "PSO", "CMA-ES"]
    data = {m: (mm(m, "sr_1e-4"), mm(m, "sr_1e-10")) for m in methods}
    key10 = {m: v[1] for m, v in data.items()}
    labels = _order_methods(methods, key10)
    # same shared style as the "now" slide, so the two read as one chart
    _method_bars(labels, data, ("p06_prev_result", "prev518"),
                 "Success rate by method — 5/18 (n=30)",
                 figsize=(8.6, 5.0), rot=25, fs=12, ymin=40, ymax=102)
    print("prev518:", {m: round(data[m][1], 1) for m in labels})


# ─────────────────────────────────────────────────────────────────────────
# 9. Four-change timeline icons — one compact schematic per refinement
# ─────────────────────────────────────────────────────────────────────────
def _icon_axes():
    fig, ax = plt.subplots(figsize=(2.9, 1.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig, ax


def fig_chg_restart():
    from matplotlib.patches import Ellipse
    fig, ax = _icon_axes()
    ax.add_patch(Ellipse((0.22, 0.52), 0.26, 0.62, fill=False, ls=(0, (3, 3)),
                        ec=GRAY, lw=1.6))
    ax.text(0.22, 0.52, "×", fontsize=27, color=GRAY, ha="center", va="center",
            fontweight="bold")
    ax.annotate("", xy=(0.58, 0.52), xytext=(0.42, 0.52),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.6))
    ax.scatter([0.78], [0.52], marker="*", s=320, color=RED, zorder=5,
               edgecolor="white", linewidths=0.8)
    for dx, dy in [(-0.07, 0.16), (0.09, 0.12), (0.11, -0.13), (-0.09, -0.16)]:
        ax.scatter([0.78 + dx], [0.52 + dy], s=60, color=RED, zorder=4,
                   edgecolor="white", linewidths=0.5)
    ax.text(0.22, 0.06, "abandon", fontsize=10.5, color=GRAY, ha="center")
    ax.text(0.80, 0.06, "re-ignite", fontsize=10.5, color=RED, ha="center",
            fontweight="bold")
    save(fig, "p10_timeline", "restart", mode="plain")
    plt.close(fig)


def fig_chg_floor():
    from matplotlib.patches import Ellipse
    fig, ax = _icon_axes()
    ax.add_patch(Ellipse((0.3, 0.55), 0.16, 0.66, angle=22, fill=False, ec=RED,
                        lw=2.4))
    ax.text(0.3, 0.05, "ill-cond", fontsize=11, color=RED, ha="center",
            fontweight="bold")
    ax.add_patch(Ellipse((0.72, 0.55), 0.34, 0.46, fill=False, ec=BLUE, lw=2.4))
    ax.text(0.72, 0.05, "rugged", fontsize=11, color=BLUE, ha="center",
            fontweight="bold")
    save(fig, "p10_timeline", "floor", mode="plain")
    plt.close(fig)


def fig_chg_router():
    fig, ax = _icon_axes()
    ax.scatter([0.14], [0.5], s=180, color=DARK, zorder=5)
    for ex, ey, c in [(0.86, 0.82, RED), (0.86, 0.5, BLUE), (0.86, 0.18, GREEN)]:
        ax.annotate("", xy=(ex - 0.02, ey), xytext=(0.2, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        ax.scatter([ex], [ey], s=150, color=c, zorder=5, edgecolor="white",
                   linewidths=0.6)
    save(fig, "p10_timeline", "router", mode="plain")
    plt.close(fig)


def fig_chg_best2():
    fig, ax = _icon_axes()
    ax.scatter([0.16], [0.42], marker="*", s=280, color=DARK, zorder=5,
               edgecolor="white", linewidths=0.7)
    ax.annotate("", xy=(0.58, 0.42), xytext=(0.23, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.4))
    ax.annotate("", xy=(0.82, 0.82), xytext=(0.58, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.8))
    ax.text(0.42, 0.26, "1st", fontsize=11, color=GRAY, ha="center",
            fontweight="bold")
    ax.text(0.82, 0.66, "2nd", fontsize=11, color=RED, ha="center",
            fontweight="bold")
    save(fig, "p10_timeline", "best2", mode="plain")
    plt.close(fig)


if __name__ == "__main__":
    fig_restart()
    fig_restart_conv_panels()
    fig_restart_result_bar()
    fig_best2_mechanism()
    fig_best2_eqs()
    fig_best2_conv_panels()
    fig_best2_result_bar()
    fig_ladder_bars()
    fig_restart_math()
    fig_restart_illustration()
    fig_restart_panels()
    fig_floor()
    fig_floor_math_eqs()
    fig_floor_illustration()
    fig_floor_panels()
    fig_floor_result_bar()
    fig_router()
    fig_router_shapes()
    fig_multimodal_shapes()
    fig_future_cards()
    fig_router_eqs()
    fig_router_apply()
    fig_router_conv_panels()
    fig_router_result()
    fig_prev518()
    fig_chg_restart()
    fig_chg_floor()
    fig_chg_router()
    fig_chg_best2()
    fig_channel_contact()
    fig_channel_droplet()
    fig_channel_airborne()
    fig_mech_strain()
    fig_mech_restart()
    fig_mech_drilling()
    fig_waterfall()
    fig_methods()
    fig_category_split_matrix()
    fig_pr_vs_sr()
    fig_evals()
    fig_family_conv()
    fig_diag()
    print("figures →", OUT)
