from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .benchmarks import BenchmarkFunction
from .optimizers import OptimizeResult

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "deepskyblue", "tab:brown"]

_METHOD_COLOR: dict[str, str] = {
    "CMA-ES": "tab:blue",
    "PSO":    "tab:orange",
    "GA":     "tab:green",
    "SaVOA":  "tab:red",
    "VSO":    "deepskyblue",
}


def _method_color(name: str, fallback_idx: int) -> str:
    return _METHOD_COLOR.get(name, _COLORS[fallback_idx % len(_COLORS)])


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
    if not benchmark.optima_pos:
        return 0
    span = benchmark.bounds[1] - benchmark.bounds[0]
    radius = max(0.5, 0.02 * span)
    found = 0
    for opt_coords in benchmark.optima_pos:
        opt = np.array(opt_coords)
        for x, f in zip(result.history_x, result.history_f):
            if f <= success_threshold and np.linalg.norm(x - opt) <= radius:
                found += 1
                break
    return found


def _out_dir(output_dir: Path, subdir: str) -> Path:
    p = output_dir / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def _draw_convergence(
    ax: plt.Axes,
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    title: str | None = None,
) -> None:
    common_max = max(
        max(len(r.history_best) for r in results)
        for results in results_per_method.values()
    )
    for idx, (method_name, results) in enumerate(results_per_method.items()):
        color = _method_color(method_name, idx)
        padded = np.array([
            r.history_best + [r.history_best[-1]] * (common_max - len(r.history_best))
            for r in results
        ])
        evals = np.arange(1, common_max + 1)
        mean = np.mean(padded, axis=0)
        std = np.std(padded, axis=0)
        lower = np.maximum(mean - std, mean * 0.01)
        upper = mean + std

        ax.semilogy(evals, mean, color=color, linewidth=1.6, label=method_name)
        ax.fill_between(evals, lower, upper, color=color, alpha=0.18)

    ax.axhline(benchmark.optimum + 1e-10, color="gray", linestyle="--",
               linewidth=0.8, label="optimum")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel(r"Best $f$ (log)")
    ax.set_title(title or "Convergence")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)


def _draw_surface3d(
    ax: plt.Axes,
    benchmark: BenchmarkFunction,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
) -> None:
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
    ax.contour(X, Y, Z, zdir="z", offset=float(Z.min()), levels=15, cmap="viridis", alpha=0.4)
    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            oz = benchmark.func(np.array(opt))
            ax.scatter([opt[0]], [opt[1]], [oz], marker="*", color="yellow",
                       edgecolors="black", linewidths=0.5, s=150, zorder=5)
    ax.set_xlabel(r"$x_1$", labelpad=2)
    ax.set_ylabel(r"$x_2$", labelpad=2)
    ax.set_zlabel(r"$f$", labelpad=2)
    ax.set_title("Landscape")
    ax.tick_params(axis="both", labelsize=7)


def _draw_optima(ax: plt.Axes, benchmark: BenchmarkFunction) -> None:
    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            ax.plot(opt[0], opt[1], "+", color="gold", markersize=9,
                    markeredgewidth=1.8, zorder=7)


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

    X, Y, Z = _contour_data(benchmark, resolution=200)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)
    lo, hi = benchmark.bounds

    fig = plt.figure(figsize=(10, 4.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.30)

    ax_land = fig.add_subplot(gs[0])
    ax_land.contourf(X, Y, Z_plot, levels=40, cmap="viridis", alpha=0.85)
    ax_land.contour(X, Y, Z_plot, levels=20, colors="white", linewidths=0.3, alpha=0.4)
    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            ax_land.plot(opt[0], opt[1], "+", color="gold", markersize=10,
                         markeredgewidth=2.0, zorder=7)
    ax_land.set_xlim(lo, hi); ax_land.set_ylim(lo, hi)
    ax_land.set_xlabel(r"$x_1$"); ax_land.set_ylabel(r"$x_2$")
    ax_land.set_title("2D Landscape")

    ax_surf = fig.add_subplot(gs[1], projection="3d")
    _draw_surface3d(ax_surf, benchmark, X, Y, Z)

    fig.suptitle(f"{benchmark.name}  [{benchmark.category}]", fontsize=12)
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

    if benchmark.dim == 3:
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.35)
        ax_conv = fig.add_subplot(gs[0])
        _draw_convergence(ax_conv, benchmark, results_per_method, title="Convergence")

        ax_3d = fig.add_subplot(gs[1], projection="3d")
        best_method = min(
            results_per_method,
            key=lambda m: min(r.best_f for r in results_per_method[m]),
        )
        best_run = min(results_per_method[best_method], key=lambda r: r.best_f)
        if best_run.history_x:
            arr = np.array(best_run.history_x)
            f_log = np.log1p(np.array(best_run.history_f))
            sc = ax_3d.scatter(
                arr[:, 0], arr[:, 1], arr[:, 2],
                c=f_log, cmap="viridis_r", s=6, alpha=0.4,
                edgecolors="none", depthshade=True,
            )
            fig.colorbar(sc, ax=ax_3d, shrink=0.55, pad=0.08, label="log(1+f)")
        if benchmark.optima_pos:
            for opt in benchmark.optima_pos:
                ax_3d.scatter([opt[0]], [opt[1]], [opt[2]], marker="*", color="red",
                               s=180, edgecolors="white", linewidths=0.5, zorder=5)
        lo3, hi3 = benchmark.bounds
        ax_3d.set_xlim(lo3, hi3); ax_3d.set_ylim(lo3, hi3); ax_3d.set_zlim(lo3, hi3)
        ax_3d.set_xlabel(r"$x_1$", labelpad=1, fontsize=7)
        ax_3d.set_ylabel(r"$x_2$", labelpad=1, fontsize=7)
        ax_3d.set_zlabel(r"$x_3$", labelpad=1, fontsize=7)
        ax_3d.tick_params(labelsize=6)
        ax_3d.set_title(
            f"{best_method} — eval distribution\nbright=near optimum  *=optimum",
            fontsize=7,
        )
    else:
        fig, ax_conv = plt.subplots(1, 1, figsize=(8, 5))
        _draw_convergence(ax_conv, benchmark, results_per_method, title="Convergence")

    fig.suptitle(f"{benchmark.name}  [{benchmark.category}]  — Convergence", fontsize=12)
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
    color = _method_color(method_name, 0)

    X, Y, Z = _contour_data(benchmark, resolution=100)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5), dpi=60)

    def draw_frame(run_idx: int) -> list:
        ax.clear()
        ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
        ax.contour(X, Y, Z_plot, levels=10, colors="white", linewidths=0.2, alpha=0.3)
        if run_idx < len(results):
            r = results[run_idx]
            if r.history_x:
                pts = np.array(r.history_x)
                s = max(1, len(pts) // 1000)
                ax.scatter(pts[::s, 0], pts[::s, 1], s=8, c=color,
                           alpha=0.35, zorder=2, rasterized=True)
                best_f, traj = float("inf"), []
                for x, f in zip(r.history_x, r.history_best):
                    if f < best_f:
                        best_f = f
                        traj.append(x)
                if len(traj) > 1:
                    t = np.array(traj)
                    ax.plot(t[:, 0], t[:, 1], "-", color=color, linewidth=1.2,
                            zorder=3, alpha=0.8)
                bidx = int(np.argmin(r.history_best))
                bx = r.history_x[bidx]
                dot_c = "lime" if r.best_f <= 1e-4 else "red"
                ax.plot(bx[0], bx[1], "o", color=dot_c, markersize=7,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        _draw_optima(ax, benchmark)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title(f"run {run_idx+1}/{n_runs}  lime=✓  red=✗", fontsize=8)
        return []

    fig.suptitle(f"{benchmark.name}  — {method_name}", fontsize=9)
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_runs,
                                  interval=1000 // fps, blit=False)
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
    color = _method_color(method_name, 0)

    X, Y, Z = _contour_data(benchmark, resolution=100)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)

    total_evals = len(run.history_x)
    n_frames = max(1, (total_evals + step - 1) // step)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5), dpi=60)

    def draw_frame(frame_idx: int) -> list:
        comp_limit = (frame_idx + 1) * step
        ax.clear()
        ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
        ax.contour(X, Y, Z_plot, levels=10, colors="white", linewidths=0.2, alpha=0.3)
        pts = run.history_x[:comp_limit]
        if pts:
            arr = np.array(pts)
            ax.scatter(arr[:, 0], arr[:, 1], s=8, c=color, alpha=0.4, zorder=2)
        if run.history_best[:comp_limit]:
            bidx = int(np.argmin(run.history_best[:comp_limit]))
            bx = run.history_x[bidx]
            ax.plot(bx[0], bx[1], "o", color="red", markersize=8,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        _draw_optima(ax, benchmark)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        n_shown = min(comp_limit, len(run.history_x))
        bv = run.history_best[n_shown - 1] if n_shown > 0 else float("inf")
        ax.set_title(f"e={n_shown}  f={bv:.2e}", fontsize=8)
        return []

    suffix = "" if best else "_failed"
    note = "eval accum." if best else "eval accum. (failed run)"
    fig.suptitle(f"{benchmark.name}  — {method_name}  {note}", fontsize=9)
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
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
    color = _method_color(method_name, 0)

    X, Y, Z = _contour_data(benchmark, resolution=100)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)

    pops = run.history_pop
    if not pops:
        return
    s = max(1, len(pops) // pop_frames)
    indices = [min(i * s, len(pops) - 1) for i in range(pop_frames)]
    frames = [pops[idx] for idx in indices]
    n_total = run.n_evals
    eval_counts = [round((idx + 1) / len(pops) * n_total) for idx in indices]
    n_frames = len(frames)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5), dpi=60)
    has_sigma = bool(run.history_pop_sigma)

    def draw_frame(frame_idx: int) -> list:
        ax.clear()
        ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
        ax.contour(X, Y, Z_plot, levels=10, colors="white", linewidths=0.2, alpha=0.3)
        fi = min(frame_idx, n_frames - 1)
        if len(frames[fi]) > 0:
            pop = frames[fi]
            ax.scatter(pop[:, 0], pop[:, 1], s=35, c=color,
                       edgecolors="white", linewidths=0.3, zorder=4, alpha=0.9)
            if has_sigma:
                sigma_fi = min(fi, len(run.history_pop_sigma) - 1)
                pop_sig = run.history_pop_sigma[sigma_fi]
                for pos, sig in zip(pop, pop_sig):
                    circ = mpatches.Circle(
                        (float(pos[0]), float(pos[1])), float(sig),
                        fill=True, facecolor=color, edgecolor=color,
                        linewidth=1.2, alpha=0.18, zorder=3,
                    )
                    ax.add_patch(circ)
        _draw_optima(ax, benchmark)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        sigma_note = "  [○=σ]" if has_sigma else ""
        ax.set_title(f"eval={eval_counts[fi]}{sigma_note}", fontsize=8)
        return []

    suffix = "" if best else "_failed"
    note = "population" if best else "population (failed run)"
    fig.suptitle(f"{benchmark.name}  — {method_name}  {note}", fontsize=9)
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
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

    from matplotlib.colors import Normalize

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    lo, hi = benchmark.bounds

    all_f_log = np.log1p(run.history_f)
    vmax = float(np.percentile(all_f_log, 98)) if len(all_f_log) else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))
    cmap = plt.get_cmap("viridis_r")

    total_evals = len(run.history_x)
    step = max(1, total_evals // n_frames)

    fig = plt.figure(figsize=(5.5, 4.8), dpi=60)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08,
                        label="log(1+f)  [bright=near opt.]")
    cbar.ax.tick_params(labelsize=7)

    def draw_frame(frame_idx: int) -> list:
        ax.clear()
        comp_limit = (frame_idx + 1) * step
        pts = run.history_x[:comp_limit]
        if pts:
            arr = np.array(pts)
            f_log = np.log1p(np.array(run.history_f[:comp_limit]))
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                       c=f_log, cmap=cmap, norm=norm,
                       s=8, alpha=0.45, edgecolors="none", depthshade=True)
        if benchmark.optima_pos:
            for opt in benchmark.optima_pos:
                ax.scatter([opt[0]], [opt[1]], [opt[2]], marker="*", color="red",
                            s=180, edgecolors="white", linewidths=0.5, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        ax.set_xlabel(r"$x_1$", labelpad=0, fontsize=7)
        ax.set_ylabel(r"$x_2$", labelpad=0, fontsize=7)
        ax.set_zlabel(r"$x_3$", labelpad=0, fontsize=7)
        ax.tick_params(labelsize=6)
        n_shown = min(comp_limit, len(run.history_x))
        ax.set_title(f"eval={n_shown}", fontsize=8)
        return []

    suffix = "" if best else "_failed"
    note = "3D eval accum." if best else "3D eval accum. (failed run)"
    fig.suptitle(f"{benchmark.name}  — {method_name}  {note}  * = optimum", fontsize=9)
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
    opt_pos = (np.array(benchmark.optima_pos[0])
               if benchmark.optima_pos else None)

    pops = run.history_pop
    if not pops:
        return
    s = max(1, len(pops) // pop_frames)
    indices = [min(i * s, len(pops) - 1) for i in range(pop_frames)]
    frames_pop = [pops[idx] for idx in indices]
    eval_counts = [round((idx + 1) / len(pops) * run.n_evals) for idx in indices]
    n_frames = len(frames_pop)

    if opt_pos is not None:
        max_dist = float(np.sqrt(3) * (hi - lo))
        norm = Normalize(vmin=0.0, vmax=max_dist)
        cmap = plt.get_cmap("viridis_r")
        cbar_label = "dist to optimum  [bright=near]"
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.get_cmap("viridis_r")
        cbar_label = "relative generation"

    az_start, az_range = 30, 180

    fig = plt.figure(figsize=(5.5, 4.8), dpi=60)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, label=cbar_label)
    cbar.ax.tick_params(labelsize=7)

    def draw_frame(frame_idx: int) -> list:
        ax.clear()
        azim = az_start + az_range * frame_idx / max(n_frames - 1, 1)
        fi = min(frame_idx, n_frames - 1)
        if len(frames_pop[fi]) > 0:
            pop = frames_pop[fi]
            if opt_pos is not None:
                dists = np.linalg.norm(pop - opt_pos, axis=1)
                c_vals = dists
            else:
                c_vals = np.full(len(pop), frame_idx / max(n_frames - 1, 1))
            ax.scatter(pop[:, 0], pop[:, 1], pop[:, 2],
                       c=c_vals, cmap=cmap, norm=norm,
                       s=40, edgecolors="white", linewidths=0.3, alpha=0.9, depthshade=True)
        if benchmark.optima_pos:
            for opt in benchmark.optima_pos:
                ax.scatter([opt[0]], [opt[1]], [opt[2]], marker="*", color="red",
                            s=200, edgecolors="white", linewidths=0.5, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        ax.set_xlabel(r"$x_1$", labelpad=0, fontsize=7)
        ax.set_ylabel(r"$x_2$", labelpad=0, fontsize=7)
        ax.set_zlabel(r"$x_3$", labelpad=0, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=azim)
        ax.set_title(f"eval={eval_counts[fi]}", fontsize=8)
        return []

    suffix = "" if best else "_failed"
    note = "3D population" if best else "3D population (failed run)"
    fig.suptitle(f"{benchmark.name}  — {method_name}  {note}  * = optimum", fontsize=9)
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    _save_anim(ani, output_dir, f"{benchmark.name}_{method_name}_3dpopulation{suffix}", fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: per-method VSO dynamics SVG (3-row, 1 column)
# ---------------------------------------------------------------------------

def save_method_vso_svg(
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    output_dir: str | Path = "results",
    best: bool = True,
) -> None:
    """σ dynamics / elite water level / virtual breathing for one method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not any(r.history_sigma_global for r in results):
        return

    run = (min(results, key=lambda r: r.best_f) if best
           else max(results, key=lambda r: r.best_f))
    color = _method_color(method_name, 0)

    fig, axes = plt.subplots(3, 1, figsize=(6, 9.5), squeeze=False)
    row_titles = ["σ per individual", "Elite water level", "Virtual breathing"]
    for row, rtitle in enumerate(row_titles):
        axes[row][0].set_ylabel(rtitle, fontsize=9, fontweight="bold", labelpad=8)

    evals = np.array(run.history_eval_count) if run.history_eval_count else None

    # ── Row 0: σ dynamics ─────────────────────────────────────────────────────
    ax = axes[0][0]
    sg = run.history_sigma_global
    ps = run.history_pop_sigma
    if sg:
        xs = evals if evals is not None and len(evals) == len(sg) else np.arange(len(sg))
        ax.semilogy(xs, sg, color="gray", linewidth=1.1, linestyle="--",
                    label="σ_global", zorder=3)
        if ps:
            n_g = min(len(sg), len(ps))
            ps_gen = ps[-n_g:] if len(ps) > len(sg) else ps[:n_g]
            med = np.array([np.median(s)          for s in ps_gen])
            q25 = np.array([np.percentile(s, 25)  for s in ps_gen])
            q75 = np.array([np.percentile(s, 75)  for s in ps_gen])
            mn  = np.array([np.min(s)              for s in ps_gen])
            mx  = np.array([np.max(s)              for s in ps_gen])
            g = xs[:n_g]
            ax.semilogy(g, med, color=color, linewidth=1.4, label="median σᵢ", zorder=4)
            ax.fill_between(g, q25, q75, color=color, alpha=0.28, zorder=2, label="Q25–Q75")
            ax.fill_between(g, mn,  mx,  color=color, alpha=0.10, zorder=1)
    ax.set_title(method_name, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, which="both", alpha=0.18)

    # ── Row 1: Elite water level ───────────────────────────────────────────────
    ax = axes[1][0]
    cutoffs = run.history_elite_cutoff
    n_el    = run.history_n_elite
    if cutoffs:
        xs = evals if evals is not None and len(evals) == len(cutoffs) else np.arange(len(cutoffs))
        ax.semilogy(xs, cutoffs, color=color, linewidth=1.4, label="elite cutoff")
        if n_el:
            ax2 = ax.twinx()
            g2 = xs[:len(n_el)]
            ax2.step(g2, n_el[:len(g2)], color="goldenrod", linewidth=0.9,
                     linestyle=":", where="post", label="n_elite")
            ax2.set_ylabel("n_elite", fontsize=7, color="goldenrod")
            ax2.tick_params(labelsize=6, colors="goldenrod")
            ax2.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, which="both", alpha=0.18)

    # ── Row 2: Virtual breathing ───────────────────────────────────────────────
    ax = axes[2][0]
    n_act  = run.history_n_active
    no_imp = run.history_no_improve
    if n_act:
        xs = evals if evals is not None and len(evals) == len(n_act) else np.arange(len(n_act))
        ax.step(xs, n_act, color=color, linewidth=1.4, where="post", label="n_active")
        if n_el:
            g2 = xs[:len(n_el)]
            ax.step(g2, n_el[:len(g2)], color="goldenrod", linewidth=1.0,
                    linestyle="--", where="post", label="n_elite")
        if no_imp:
            ax2 = ax.twinx()
            g3 = xs[:len(no_imp)]
            ax2.plot(g3, no_imp[:len(g3)], color="gray", linewidth=0.7,
                     linestyle=":", alpha=0.6)
            ax2.set_ylabel("no_improve", fontsize=7, color="gray")
            ax2.tick_params(labelsize=6, colors="gray")
            ax2.set_ylim(bottom=0)
    ax.set_xlabel("Evaluations", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.18)

    suffix = "" if best else "_failed"
    note   = "best run" if best else "failed (worst) run"
    fig.suptitle(
        f"{benchmark.name}  — {method_name}  VSO dynamics  ({note})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{benchmark.name}_{method_name}_vso_dyn{suffix}.svg",
        format="svg", bbox_inches="tight",
    )
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
    from .runner import _evals_to_target
    fieldnames_s = ["function", "category", "method", "mean_time_s",
                    "mean_best_f", "sr_1e-2", "sr_1e-4", "ert",
                    "mean_optima_found", "mean_optima_rate"]
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
            n_success = int(np.sum(best_fs <= success_threshold))
            evals_list = [_evals_to_target(r, success_threshold) for r in results]
            ert = f"{sum(evals_list) / n_success:.0f}" if n_success > 0 else "inf"
            writer.writerow({
                "function": benchmark.name,
                "category": benchmark.category,
                "method": method,
                "mean_time_s": f"{np.mean(times):.3f}",
                "mean_best_f": f"{np.mean(best_fs):.4e}",
                "sr_1e-2":     f"{float(np.mean(best_fs <= 1e-2)):.0%}",
                "sr_1e-4":     f"{float(np.mean(best_fs <= success_threshold)):.0%}",
                "ert":         ert,
                "mean_optima_found": f"{mean_optima:.2f}",
                "mean_optima_rate": f"{mean_optima / n_optima_total:.2f}" if n_optima_total else "N/A",
            })
