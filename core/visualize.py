from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .benchmarks import BenchmarkFunction
from .optimizers import OptimizeResult

# ── Look ──────────────────────────────────────────────────────────────────────
# Matches the results UI (web/static/style.css): ink text, recessive grid, one
# teal for the proposed method. Japanese labels need a CJK face; the list falls
# through to whatever the machine has (matplotlib >= 3.6 falls back per glyph).
INK, INK_2, MUTED, RULE, GRID, SURFACE = (
    "#17202b", "#4a5563", "#7d8794", "#dce1e3", "#e9edee", "#ffffff")
GOOD, BAD = "#00897b", "#c2412d"          # reached the target / did not

# First installed face wins; listing only installed ones keeps matplotlib from
# warning once per missing family. The variable "Noto Sans JP" is left out: on
# Windows matplotlib registers only its thin master.
_FONT_PREFS = ["IBM Plex Sans JP", "BIZ UDPGothic", "Yu Gothic UI", "Meiryo",
               "Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans"]
try:
    from matplotlib import font_manager as _fm
    _installed = {f.name for f in _fm.fontManager.ttflist}
    _FONTS = [f for f in _FONT_PREFS if f in _installed] + ["DejaVu Sans"]
except Exception:          # pragma: no cover
    _FONTS = ["DejaVu Sans"]

matplotlib.rcParams.update({
    "font.family": _FONTS,
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "path",
})

# Fixed colour per entity — a method keeps its colour in every figure, whatever
# else is plotted (validated with the dataviz palette checker: lightness, chroma,
# normal-vision and CVD separation all pass in this order; red/green sits in the
# CVD warning band, so line style and the legend carry identity as well).
_METHOD_COLOR: dict[str, str] = {
    "MC-ESO":       "#00897b",
    "CMA-ES":       "#4a3aa7",
    "IPOP-CMA-ES":  "#eb6834",
    "BIPOP-CMA-ES": "#2a78d6",
    "DE":           "#e87ba4",
    "L-SHADE":      "#eda100",
    "PSO":          "#e34948",
    "SaVOA":        "#008300",
}
_METHOD_DASH: dict[str, tuple] = {"MC-ESO-v0": (0, (4, 2))}
OTHER = "#b3bac1"          # every method without a fixed colour
_SOLO = "#3c4a57"          # per-method figure of a method without a fixed colour


def _method_color(name: str, fallback_idx: int = 0) -> str:
    if name == "MC-ESO-v0":
        return "#5fb3a8"
    return _METHOD_COLOR.get(name, _SOLO)


# Landscape: one hue (slate teal), dark = low f. Search points sit on top in the
# method's colour, so the background stays light over most of the box.
_LAND = LinearSegmentedColormap.from_list(
    "land", ["#2f5f66", "#86aaa9", "#d8e3e0", "#f6f5f0"])
_LAND_LIGHT = LinearSegmentedColormap.from_list(
    "land_light", ["#7d9e9d", "#b9cecb", "#e2ebe8", "#faf9f5"])
# f along a 3-D point cloud: one hue (blue), dark = near the optimum.
_FCMAP = LinearSegmentedColormap.from_list(
    "fval", ["#0d366b", "#2a78d6", "#86b6ef", "#dce9f8"])

_TARGET = 1e-10            # SR@1e-10, the primary metric
_FLOOR = 1e-12             # log-axis floor for exact zeros


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contour_data(
    benchmark: BenchmarkFunction,
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = benchmark.bounds
    xs = np.linspace(lo, hi, resolution)
    ys = np.linspace(lo, hi, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda x, y: benchmark.func(np.array([x, y])))(X, Y)
    return X, Y, Z


def _count_optima_found(
    result: OptimizeResult,
    benchmark: BenchmarkFunction,
    success_threshold: float = 1e-4,
) -> int:
    """How many known global optima this run discovered (nearest-attribution).

    Thin wrapper over the canonical ``runner.optima_found_mask`` so the count
    here matches the ``pr_*`` / ``mmo_sr_*`` columns written below.
    """
    if not benchmark.optima_pos:
        return 0
    from .runner import optima_found_mask
    span = benchmark.bounds[1] - benchmark.bounds[0]
    return int(optima_found_mask(
        result, benchmark.optima_pos, span, success_threshold).sum())


def _out_dir(output_dir: Path, subdir: str) -> Path:
    p = output_dir / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gap(values, benchmark: BenchmarkFunction) -> np.ndarray:
    """f − f* clipped to the log floor."""
    return np.maximum(np.asarray(values, dtype=float) - benchmark.optimum, _FLOOR)


def _thin(n: int, k: int = 700) -> np.ndarray:
    """Indices that keep a long curve's shape at a fraction of the SVG size."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.concatenate([np.linspace(0, n - 1, k).astype(int), [n - 1]]))


def _draw_convergence(
    ax: plt.Axes,
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    title: str | None = None,
) -> None:
    """Median best-so-far f − f* per method, quartile band for the coloured ones.

    Methods with a fixed colour (and MC-ESO-v0) are drawn in colour; all others
    are grey context lines under a single legend entry, so a 35-method run stays
    readable and a method never changes colour between figures.
    """
    common_max = max(max(len(r.history_best) for r in res)
                     for res in results_per_method.values())
    idx = _thin(common_max)
    evals = idx + 1
    curves = {}
    for name, results in results_per_method.items():
        padded = np.array([r.history_best + [r.history_best[-1]] * (common_max - len(r.history_best))
                           for r in results], dtype=float)
        g = _gap(padded, benchmark)[:, idx]
        curves[name] = (np.median(g, axis=0), np.percentile(g, 25, axis=0),
                        np.percentile(g, 75, axis=0))

    def colored(n):
        return n in _METHOD_COLOR or n in _METHOD_DASH

    n_other = 0
    for name, (med, _, _) in curves.items():
        if not colored(name):
            ax.plot(evals, med, color=OTHER, linewidth=0.8, alpha=0.8, zorder=1)
            n_other += 1
    order = sorted((n for n in curves if colored(n)),
                   key=lambda n: (n == "MC-ESO", -curves[n][0][-1]))
    for name in order:
        med, q1, q3 = curves[name]
        c = _method_color(name)
        is_ref = name == "MC-ESO"
        if is_ref or name in _METHOD_COLOR:
            ax.fill_between(evals, q1, q3, color=c, alpha=0.18 if is_ref else 0.06,
                            linewidth=0, zorder=2)
        ax.plot(evals, med, color=c, linewidth=2.2 if is_ref else 1.4,
                linestyle=_METHOD_DASH.get(name, "-"), zorder=4 if is_ref else 3,
                label=name)
    ax.set_yscale("log")
    ax.axhline(_TARGET, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.text(evals[-1], _TARGET, " 1e-10", color=MUTED, fontsize=7, va="center", ha="left")
    ax.set_ylim(bottom=_FLOOR / 2)
    ax.set_xlim(1, evals[-1])
    ax.set_xlabel("評価回数")
    ax.set_ylabel("f − f*（中央値、帯は四分位）")
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    if title:
        ax.set_title(title)
    # Legend: best final median first, MC-ESO always on top of the list.
    handles = [Line2D([], [], color=_method_color(n), linewidth=2.2 if n == "MC-ESO" else 1.4,
                      linestyle=_METHOD_DASH.get(n, "-"), label=n)
               for n in sorted(order, key=lambda n: (n != "MC-ESO", curves[n][0][-1]))]
    if n_other:
        handles.append(Line2D([], [], color=OTHER, linewidth=0.8, label=f"その他 {n_other} 手法"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, handlelength=2.2)


def _land_bg(ax: plt.Axes, X, Y, Z_plot, lo, hi, light: bool = True) -> None:
    ax.contourf(X, Y, Z_plot, levels=24, cmap=_LAND_LIGHT if light else _LAND, zorder=0)
    ax.contour(X, Y, Z_plot, levels=12, colors="#ffffff", linewidths=0.4, alpha=0.8, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(RULE)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$", rotation=0, labelpad=8)


def _draw_surface3d(
    ax: plt.Axes,
    benchmark: BenchmarkFunction,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
) -> None:
    Zp = np.log1p(Z - Z.min())
    ax.plot_surface(X, Y, Zp, cmap=_LAND, linewidth=0, antialiased=True, rcount=80, ccount=80)
    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            oz = np.log1p(benchmark.func(np.array(opt)) - Z.min())
            ax.scatter([opt[0]], [opt[1]], [oz], marker="*", color=BAD,
                       edgecolors="white", linewidths=0.6, s=110, zorder=5)
    _style_3d(ax)
    ax.set_zlabel("log(1 + f − f_min)", labelpad=2)


def _style_3d(ax) -> None:
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0))
        a._axinfo["grid"].update(color=GRID, linewidth=0.5)
        a.line.set_color(RULE)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.set_xlabel(r"$x_1$", labelpad=0); ax.set_ylabel(r"$x_2$", labelpad=0)


def _draw_optima(ax: plt.Axes, benchmark: BenchmarkFunction) -> None:
    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            # hollow and under the search marks: it shows where the optimum is
            # without hiding a population that has converged onto it
            ax.plot(opt[0], opt[1], marker="*", markerfacecolor="none",
                    markeredgecolor=BAD, markersize=13, markeredgewidth=1.3,
                    zorder=1.5, linestyle="none")


def _frame_title(ax, left: str, right: str = "") -> None:
    ax.set_title(left, loc="left", fontsize=9)
    if right:
        ax.set_title(right, loc="right", fontsize=8, fontweight="normal", color=INK_2)


def _save_anim(ani: animation.FuncAnimation, out_dir: Path, stem: str, fps: int) -> str:
    """Save animation as WebP (smaller), fallback to GIF. Returns extension used."""
    webp_path = out_dir / f"{stem}.webp"
    try:
        ani.save(str(webp_path), writer=animation.PillowWriter(fps=fps))
        return "webp"
    except Exception:
        webp_path.unlink(missing_ok=True)
    gif_path = out_dir / f"{stem}.gif"
    ani.save(str(gif_path), writer=animation.PillowWriter(fps=fps))
    return "gif"


# Animations: square frames at a size the UI's grid cells show sharply.
_ANIM_SIZE, _ANIM_DPI = (4.4, 4.4), 90


# ---------------------------------------------------------------------------
# Public: landscape SVG (2D contour + 3D surface, no method dependency)
# ---------------------------------------------------------------------------

def save_landscape_svg(
    benchmark: BenchmarkFunction,
    output_dir: str | Path = "results",
) -> None:
    """2D contour map + 3D surface. Only generated for 2D benchmarks."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, Z = _contour_data(benchmark, resolution=160)
    Z_plot = np.log1p(Z - Z.min())
    lo, hi = benchmark.bounds

    fig = plt.figure(figsize=(9.6, 4.4))
    ax_land = fig.add_subplot(1, 2, 1)
    ax_land.contourf(X, Y, Z_plot, levels=36, cmap=_LAND)
    ax_land.contour(X, Y, Z_plot, levels=14, colors="#ffffff", linewidths=0.4, alpha=0.7)
    _draw_optima(ax_land, benchmark)
    ax_land.set_xlim(lo, hi); ax_land.set_ylim(lo, hi); ax_land.set_aspect("equal")
    ax_land.grid(False)
    ax_land.set_xlabel(r"$x_1$"); ax_land.set_ylabel(r"$x_2$", rotation=0, labelpad=8)
    ax_land.set_title("等高線（濃いほど f が低い、★ = 大域最適）")

    ax_surf = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_surface3d(ax_surf, benchmark, X, Y, Z)
    ax_surf.set_title("曲面（log スケール）")
    ax_surf.view_init(elev=32, azim=-58)

    fig.suptitle(f"{benchmark.name}（{benchmark.category}）", x=0.06, ha="left",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    fig.savefig(output_dir / f"{benchmark.name}_landscape.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: convergence SVG (all methods, combined)
# ---------------------------------------------------------------------------

def save_convergence_svg(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
) -> None:
    """Convergence curves for all methods in one comparison plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.6))
    _draw_convergence(ax, benchmark, results_per_method)
    n_runs = max(len(r) for r in results_per_method.values())
    fig.suptitle(f"{benchmark.name}　収束の推移", x=0.07, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.text(0.07, 0.905, f"{benchmark.dim} 次元、{n_runs} run の最良値の推移。点線は SR の主指標の閾値 1e-10。",
             color=MUTED, fontsize=8, ha="left")
    fig.savefig(output_dir / f"{benchmark.name}_convergence.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method runs animation (2D only)
# ---------------------------------------------------------------------------

def save_method_runs_anim(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    fps: int = 3,
) -> None:
    """One frame per run: eval scatter + best trajectory. 2D only."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(results)
    lo, hi = benchmark.bounds
    color = _method_color(method_name)
    X, Y, Z = _contour_data(benchmark, resolution=100)
    Z_plot = np.log1p(Z - Z.min())

    fig, ax = plt.subplots(1, 1, figsize=_ANIM_SIZE, dpi=_ANIM_DPI)

    def draw_frame(run_idx: int) -> list:
        ax.clear()
        _land_bg(ax, X, Y, Z_plot, lo, hi)
        ok = None
        if run_idx < len(results):
            r = results[run_idx]
            if len(r.history_x):
                pts = np.asarray(r.history_x)
                s = max(1, len(pts) // 1500)
                ax.scatter(pts[::s, 0], pts[::s, 1], s=5, c=color, alpha=0.35,
                           linewidths=0, zorder=2, rasterized=True)
                best_f, traj = float("inf"), []
                for x, f in zip(r.history_x, r.history_best):
                    if f < best_f:
                        best_f = f
                        traj.append(x)
                if len(traj) > 1:
                    t = np.array(traj)
                    ax.plot(t[:, 0], t[:, 1], "-", color=INK, linewidth=0.7, alpha=0.45, zorder=3)
                bx = r.history_x[int(np.argmin(r.history_best))]
                ok = r.best_f - benchmark.optimum <= 1e-4
                ax.plot(bx[0], bx[1], marker="o" if ok else "X", color=GOOD if ok else BAD,
                        markersize=8, markeredgecolor="white", markeredgewidth=1.0,
                        zorder=9, linestyle="none")
        _draw_optima(ax, benchmark)
        state = "" if ok is None else ("　到達 ●" if ok else "　未到達 ✕")
        _frame_title(ax, method_name, f"run {run_idx + 1}/{n_runs}{state}")
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=n_runs,
                                  interval=1000 // fps, blit=False)
    fig.tight_layout()
    _save_anim(ani, output_dir, f"{benchmark.name}_{method_name}_runs", fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method eval accumulation animation (2D only)
# ---------------------------------------------------------------------------

def save_method_evals_anim(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    step: int = 100,
    fps: int = 6,
    best: bool = True,
) -> None:
    """Animate eval-point accumulation for best or worst run. 2D only."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    lo, hi = benchmark.bounds
    color = _method_color(method_name)
    X, Y, Z = _contour_data(benchmark, resolution=100)
    Z_plot = np.log1p(Z - Z.min())

    hx = np.asarray(run.history_x)
    total_evals = len(hx)
    n_frames = max(1, (total_evals + step - 1) // step)
    fig, ax = plt.subplots(1, 1, figsize=_ANIM_SIZE, dpi=_ANIM_DPI)
    which = "最良の run" if best else "最悪の run"

    def draw_frame(frame_idx: int) -> list:
        n_shown = min((frame_idx + 1) * step, total_evals)
        ax.clear()
        _land_bg(ax, X, Y, Z_plot, lo, hi)
        if n_shown:
            arr = hx[:n_shown]
            ax.scatter(arr[:, 0], arr[:, 1], s=5, c=color, alpha=0.4, linewidths=0, zorder=2)
            # the latest batch stands out from the accumulated cloud
            new = hx[max(0, n_shown - step):n_shown]
            ax.scatter(new[:, 0], new[:, 1], s=9, c=color, alpha=0.95, linewidths=0, zorder=3)
            bidx = int(np.argmin(run.history_best[:n_shown]))
            ax.plot(hx[bidx][0], hx[bidx][1], marker="o", color=INK, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=9, linestyle="none")
        _draw_optima(ax, benchmark)
        bv = run.history_best[n_shown - 1] - benchmark.optimum if n_shown else float("inf")
        _frame_title(ax, f"{method_name}（{which}）", f"{n_shown:,} 評価　f−f* = {bv:.1e}")
        return []

    suffix = "" if best else "_failed"
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    fig.tight_layout()
    _save_anim(ani, output_dir, f"{benchmark.name}_{method_name}_evals{suffix}", fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method population animation (2D only)
# ---------------------------------------------------------------------------

def save_method_population_anim(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    pop_frames: int = 20,
    fps: int = 6,
    best: bool = True,
) -> None:
    """Animate population over generations for best or worst run. 2D only."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    lo, hi = benchmark.bounds
    color = _method_color(method_name)
    X, Y, Z = _contour_data(benchmark, resolution=100)
    Z_plot = np.log1p(Z - Z.min())

    pops = run.history_pop
    if not pops:
        return
    s = max(1, len(pops) // pop_frames)
    indices = [min(i * s, len(pops) - 1) for i in range(pop_frames)]
    frames = [pops[idx] for idx in indices]
    eval_counts = [round((idx + 1) / len(pops) * run.n_evals) for idx in indices]
    n_frames = len(frames)
    has_sigma = bool(run.history_pop_sigma)
    which = "最良の run" if best else "最悪の run"

    fig, ax = plt.subplots(1, 1, figsize=_ANIM_SIZE, dpi=_ANIM_DPI)

    def draw_frame(frame_idx: int) -> list:
        ax.clear()
        _land_bg(ax, X, Y, Z_plot, lo, hi)
        fi = min(frame_idx, n_frames - 1)
        pop = frames[fi]
        if len(pop) > 0:
            if has_sigma:
                sig = run.history_pop_sigma[min(fi, len(run.history_pop_sigma) - 1)]
                for pos, sg in zip(pop, sig):
                    ax.add_patch(mpatches.Circle((float(pos[0]), float(pos[1])), float(sg),
                                                 fill=False, edgecolor=color, linewidth=0.8,
                                                 alpha=0.45, zorder=3))
            ax.scatter(pop[:, 0], pop[:, 1], s=22, c=color, edgecolors="white",
                       linewidths=0.6, zorder=4)
        _draw_optima(ax, benchmark)
        # A converged population collapses onto one point; say how small it is.
        spread = float(np.ptp(pop, axis=0).max()) if len(pop) > 1 else 0.0
        note = "　○ = 宿主ごとの σ" if has_sigma else ""
        _frame_title(ax, f"{method_name}（{which}）",
                     f"{eval_counts[fi]:,} 評価　広がり {spread:.0e}{note}")
        return []

    suffix = "" if best else "_failed"
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    fig.tight_layout()
    _save_anim(ani, output_dir, f"{benchmark.name}_{method_name}_population{suffix}", fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method 3D eval accumulation animation
# ---------------------------------------------------------------------------

def save_method_3devals_anim(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    fps: int = 8,
    n_frames: int = 30,
    best: bool = True,
) -> None:
    """3D eval accumulation colored by log(1+f). 3D benchmarks only."""
    if benchmark.dim != 3:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.colors import LogNorm

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    lo, hi = benchmark.bounds
    hx = np.asarray(run.history_x)
    gap = _gap(run.history_f, benchmark)
    vmax = float(np.percentile(gap, 98)) if len(gap) else 1.0
    norm = LogNorm(vmin=max(float(gap.min()), _FLOOR), vmax=max(vmax, 1e-8))
    total_evals = len(hx)
    step = max(1, total_evals // n_frames)
    which = "最良の run" if best else "最悪の run"

    fig = plt.figure(figsize=(5.0, 4.4), dpi=_ANIM_DPI)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=_FCMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label("f − f*（濃いほど最適に近い）", color=INK_2, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5, colors=MUTED)
    cbar.outline.set_visible(False)

    def draw_frame(frame_idx: int) -> list:
        ax.clear()
        n_shown = min((frame_idx + 1) * step, total_evals)
        if n_shown:
            ax.scatter(hx[:n_shown, 0], hx[:n_shown, 1], hx[:n_shown, 2],
                       c=gap[:n_shown], cmap=_FCMAP, norm=norm,
                       s=6, alpha=0.55, edgecolors="none", depthshade=False)
        if benchmark.optima_pos:
            for opt in benchmark.optima_pos:
                ax.scatter([opt[0]], [opt[1]], [opt[2]], marker="*", color=BAD,
                           s=140, edgecolors="white", linewidths=0.6, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        _style_3d(ax)
        ax.set_zlabel(r"$x_3$", labelpad=0)
        ax.set_title(f"{method_name}（{which}）  {n_shown:,} 評価", loc="left", fontsize=9)
        return []

    suffix = "" if best else "_failed"
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    _save_anim(ani, output_dir, f"{benchmark.name}_{method_name}_3devals{suffix}", fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method 3D population animation (camera rotates)
# ---------------------------------------------------------------------------

def save_method_3dpopulation_anim(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    pop_frames: int = 20,
    fps: int = 6,
    best: bool = True,
) -> None:
    """3D population colored by distance to optimum; camera rotates 180°."""
    if benchmark.dim != 3:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.colors import Normalize

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    lo, hi = benchmark.bounds
    opt_pos = np.array(benchmark.optima_pos[0]) if benchmark.optima_pos else None

    pops = run.history_pop
    if not pops:
        return
    s = max(1, len(pops) // pop_frames)
    indices = [min(i * s, len(pops) - 1) for i in range(pop_frames)]
    frames_pop = [pops[idx] for idx in indices]
    eval_counts = [round((idx + 1) / len(pops) * run.n_evals) for idx in indices]
    n_frames = len(frames_pop)
    which = "最良の run" if best else "最悪の run"

    if opt_pos is not None:
        norm = Normalize(vmin=0.0, vmax=float(np.sqrt(3) * (hi - lo)) / 2)
        cbar_label = "最適解までの距離（濃いほど近い）"
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
        cbar_label = "世代の進み"

    fig = plt.figure(figsize=(5.0, 4.4), dpi=_ANIM_DPI)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=_FCMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label(cbar_label, color=INK_2, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5, colors=MUTED)
    cbar.outline.set_visible(False)

    def draw_frame(frame_idx: int) -> list:
        ax.clear()
        azim = 30 + 180 * frame_idx / max(n_frames - 1, 1)
        fi = min(frame_idx, n_frames - 1)
        pop = frames_pop[fi]
        if len(pop) > 0:
            c_vals = (np.linalg.norm(pop - opt_pos, axis=1) if opt_pos is not None
                      else np.full(len(pop), frame_idx / max(n_frames - 1, 1)))
            ax.scatter(pop[:, 0], pop[:, 1], pop[:, 2], c=c_vals, cmap=_FCMAP, norm=norm,
                       s=30, edgecolors="white", linewidths=0.5, alpha=0.95, depthshade=False)
        if benchmark.optima_pos:
            for opt in benchmark.optima_pos:
                ax.scatter([opt[0]], [opt[1]], [opt[2]], marker="*", color=BAD,
                           s=150, edgecolors="white", linewidths=0.6, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        _style_3d(ax)
        ax.set_zlabel(r"$x_3$", labelpad=0)
        ax.view_init(elev=25, azim=azim)
        ax.set_title(f"{method_name}（{which}）  {eval_counts[fi]:,} 評価", loc="left", fontsize=9)
        return []

    suffix = "" if best else "_failed"
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    _save_anim(ani, output_dir, f"{benchmark.name}_{method_name}_3dpopulation{suffix}", fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method outbreak dynamics SVG (MC-ESO internals, one shared x-axis)
# ---------------------------------------------------------------------------

def save_method_vso_svg(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    best: bool = True,
) -> None:
    """Outbreak dynamics for MC-ESO: σ, best f, strain count, stagnation.

    Four stacked panels on one evaluation axis (no twin y-axes): spillovers
    show up in every panel at the same x as σ jumping back up and the
    stagnation counter resetting.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not any(r.history_sigma_global for r in results):
        return

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    color = _method_color(method_name)
    evals = np.array(run.history_eval_count) if run.history_eval_count else None

    def gen_x(n):
        return evals if evals is not None and len(evals) == n else np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.1, 0.6, 0.8], "hspace": 0.32})

    # σ: global step, per-host median with the quartile band, σ per offspring
    ax = axes[0]
    sg, ps = run.history_sigma_global, run.history_pop_sigma
    if sg:
        xs = gen_x(len(sg))
        if ps:
            n_g = min(len(sg), len(ps))
            ps_gen = ps[-n_g:] if len(ps) > len(sg) else ps[:n_g]
            g = xs[:n_g]
            ax.fill_between(g, [np.percentile(s, 25) for s in ps_gen],
                            [np.percentile(s, 75) for s in ps_gen],
                            color=color, alpha=0.18, linewidth=0, label="宿主の σ（四分位）")
            ax.plot(g, [np.median(s) for s in ps_gen], color=color, linewidth=1.4,
                    label="宿主の σ（中央値）")
        ax.plot(xs, sg, color=INK, linewidth=1.0, linestyle=(0, (4, 2)), label="全体の σ")
    se = run.history_sigma_eval
    if se:
        se_arr = np.array(se, dtype=float)
        n_init = run.n_evals - len(se_arr)
        ok = np.isfinite(se_arr)
        ax.scatter(np.arange(n_init, n_init + len(se_arr))[ok], se_arr[ok], s=2,
                   color=color, alpha=0.15, linewidths=0, label="子ごとの σ", rasterized=True)
    ax.set_yscale("log")
    ax.set_title("歩幅 σ", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=7,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)

    # best f − f*
    ax = axes[1]
    hb = run.history_best
    if hb:
        idx = _thin(len(hb), 1500)
        ax.plot(idx + 1, _gap(np.asarray(hb)[idx], benchmark), color=color, linewidth=1.6)
        ax.axhline(_TARGET, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yscale("log")
    ax.set_title("これまでの最良値 f − f*（点線 = 1e-10）", loc="left")

    # strains (niched elites) — its own panel, not a twin axis
    ax = axes[2]
    n_el = run.history_n_elite
    if n_el:
        ax.step(gen_x(len(n_el)), n_el, where="post", color=INK_2, linewidth=1.0)
        ax.set_ylim(0, max(n_el) + 1)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("系統の数", loc="left")

    # stagnation counter that triggers a spillover
    ax = axes[3]
    no_imp = run.history_no_improve
    if no_imp:
        ax.plot(gen_x(len(no_imp)), no_imp, color=color, linewidth=1.1)
        thr = 300 * (benchmark.dim / 2)
        ax.axhline(thr, color=BAD, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(0.005, thr, f" スピルオーバーの閾値 {thr:.0f}", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7, color=INK_2)
        ax.set_ylim(bottom=0)
    ax.set_title("停滞カウンタ（改善の無い評価回数）", loc="left")
    ax.set_xlabel("評価回数")

    which = "最良の run" if best else "最悪の run"
    fig.suptitle(f"{benchmark.name}　{method_name} の内部状態（{which}）", x=0.08, ha="left",
                 fontsize=11, fontweight="bold", color=INK, y=0.995)
    fig.savefig(output_dir / f"{benchmark.name}_{method_name}_outbreak_dyn{'' if best else '_failed'}.svg",
                format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: stats CSV + summary CSV  (unchanged)
# ---------------------------------------------------------------------------

def save_stats(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    times_per_method: dict[str, list[float]],
    output_dir: str | Path = "results",
    success_threshold: float = 1e-4,
) -> None:
    output_dir = Path(output_dir)
    stats_dir = _out_dir(output_dir, "stats")
    n_optima_total = len(benchmark.optima_pos) if benchmark.optima_pos else 0

    rows = []
    for method, results in results_per_method.items():
        times = times_per_method.get(method, [0.0] * len(results))
        for i, (r, t) in enumerate(zip(results, times)):
            optima_found = _count_optima_found(r, benchmark, success_threshold)
            rows.append({
                "method": method,
                "seed": i * 100,
                "time_s": f"{t:.3f}",
                "best_f": f"{r.best_f:.6e}",
                "n_evals": r.n_evals,
                "success": r.best_f <= success_threshold,
                "optima_found": optima_found,
                "optima_total": n_optima_total,
                "optima_rate": f"{optima_found / n_optima_total:.2f}" if n_optima_total else "N/A",
            })

    fieldnames = ["method", "seed", "time_s", "best_f", "n_evals",
                  "success", "optima_found", "optima_total", "optima_rate"]
    with open(stats_dir / f"{benchmark.name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.csv"
    summary_exists = summary_path.exists()
    from .runner import (_evals_to_target, ecdf_auc, SR_THRESHOLDS,
                         PEAK_THRESHOLDS, peak_metrics,
                         NICHE_ACCURACIES, niching_peak_metrics)
    sr_keys = [f"sr_{thr:.0e}".replace("e-0", "e-") for thr in SR_THRESHOLDS]
    span = benchmark.bounds[1] - benchmark.bounds[0]
    # Multi-modal columns: peak ratio (pr_*) and MMO success rate (mmo_sr_*) per
    # tolerance — the share of the K global optima found / runs finding all K.
    pr_keys = [f"pr_{thr:.0e}".replace("e-0", "e-") for thr in PEAK_THRESHOLDS]
    mmo_keys = [f"mmo_sr_{thr:.0e}".replace("e-0", "e-") for thr in PEAK_THRESHOLDS]
    # CEC2013-niching columns: peak ratio / success rate over the reported
    # solution set at the competition's accuracy levels. "N/A" outside the
    # niching suite (every other benchmark has a single global optimum).
    cec_pr_keys = [f"cec_pr_{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES]
    cec_sr_keys = [f"cec_sr_{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES]
    # Precision and F1 over the same reported set. PR is recall only, so it
    # cannot see a method that pads its answer up to the cap; these can.
    cec_pre_keys = [f"cec_pre_{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES]
    cec_f1_keys = [f"cec_f1_{a:.0e}".replace("e-0", "e-") for a in NICHE_ACCURACIES]
    fieldnames_s = ["function", "category", "tags", "method", "mean_time_s",
                    "mean_best_f", "median_best_f", *sr_keys,
                    "evals_succ_mean", "evals_succ_med", "ert", "ecdf_auc",
                    "mean_optima_found", "mean_optima_rate", "n_optima",
                    *pr_keys, *mmo_keys,
                    "cec_k", *cec_pr_keys, *cec_sr_keys,
                    "cec_pr_mean", "cec_sr_mean", "n_reported",
                    *cec_pre_keys, *cec_f1_keys, "cec_f1_mean"]
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_s)
        if not summary_exists:
            writer.writeheader()
        for method, results in results_per_method.items():
            times = times_per_method.get(method, [0.0] * len(results))
            optima_counts = [_count_optima_found(r, benchmark, success_threshold)
                             for r in results]
            best_fs = np.array([r.best_f for r in results])
            mean_optima = float(np.mean(optima_counts))
            success_mask = best_fs <= success_threshold
            n_success = int(np.sum(success_mask))
            evals_list = [_evals_to_target(r, success_threshold) for r in results]
            ert = f"{sum(evals_list) / n_success:.0f}" if n_success > 0 else "inf"
            succ_evals = [e for e, ok in zip(evals_list, success_mask) if ok]
            evals_succ_mean = f"{np.mean(succ_evals):.0f}"   if succ_evals else "inf"
            evals_succ_med  = f"{np.median(succ_evals):.0f}" if succ_evals else "inf"
            max_budget = max((len(r.history_f) for r in results), default=0)
            auc = ecdf_auc(results, SR_THRESHOLDS, max_budget) if max_budget else 0.0
            row = {
                "function": benchmark.name,
                "category": benchmark.category,
                "tags": "|".join(benchmark.tags),
                "method": method,
                "mean_time_s": f"{np.mean(times):.3f}",
                "mean_best_f":   f"{np.mean(best_fs):.4e}",
                "median_best_f": f"{np.median(best_fs):.4e}",
                "evals_succ_mean": evals_succ_mean,
                "evals_succ_med": evals_succ_med,
                "ert":           ert,
                "ecdf_auc":      f"{auc:.4f}",
                "mean_optima_found": f"{mean_optima:.2f}",
                "mean_optima_rate": f"{mean_optima / n_optima_total:.2f}" if n_optima_total else "N/A",
                "n_optima": n_optima_total,
            }
            for thr, key in zip(SR_THRESHOLDS, sr_keys):
                row[key] = f"{float(np.mean(best_fs <= thr)):.0%}"
            pm = peak_metrics(results, benchmark.optima_pos, span, PEAK_THRESHOLDS)
            for thr, pk, mk in zip(PEAK_THRESHOLDS, pr_keys, mmo_keys):
                k = f"{thr:.0e}".replace("e-0", "e-")
                row[pk] = f"{pm.get(f'pr_{k}', 0.0):.2f}" if n_optima_total else "N/A"
                row[mk] = f"{pm.get(f'mmo_sr_{k}', 0.0):.0%}" if n_optima_total else "N/A"
            npm = niching_peak_metrics(results, benchmark)
            has_cec = npm["n_optima"] > 0
            row["cec_k"] = npm["n_optima"] if has_cec else "N/A"
            for a, pk, sk in zip(NICHE_ACCURACIES, cec_pr_keys, cec_sr_keys):
                row[pk] = f"{npm[pk]:.2f}" if has_cec else "N/A"
                row[sk] = f"{npm[sk]:.0%}" if has_cec else "N/A"
            row["cec_pr_mean"] = f"{npm['cec_pr_mean']:.2f}" if has_cec else "N/A"
            row["cec_sr_mean"] = f"{npm['cec_sr_mean']:.0%}" if has_cec else "N/A"
            row["n_reported"] = f"{npm['n_reported']:.0f}" if has_cec else "N/A"
            for a, ek, fk in zip(NICHE_ACCURACIES, cec_pre_keys, cec_f1_keys):
                row[ek] = f"{npm[ek]:.2f}" if has_cec else "N/A"
                row[fk] = f"{npm[fk]:.2f}" if has_cec else "N/A"
            row["cec_f1_mean"] = f"{npm['cec_f1_mean']:.2f}" if has_cec else "N/A"
            writer.writerow(row)
