from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from benchmarks import BenchmarkFunction
from optimizers import OptimizeResult

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

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contour_data(
    benchmark: BenchmarkFunction,
    resolution: int = 200,
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


# ---------------------------------------------------------------------------
# Sub-plot builders (return axes, no file I/O)
# ---------------------------------------------------------------------------

def _draw_trajectory(
    ax: plt.Axes,
    benchmark: BenchmarkFunction,
    results: list[OptimizeResult],
    method_name: str,
    color: str,
    X: np.ndarray, Y: np.ndarray, Z_plot: np.ndarray,
) -> None:
    lo, hi = benchmark.bounds
    ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
    ax.contour(X, Y, Z_plot, levels=15, colors="white", linewidths=0.3, alpha=0.4)

    result = min(results, key=lambda r: r.best_f)
    if result.history_x:
        pts = np.array(result.history_x)
        step = max(1, len(pts) // 2000)
        pts = pts[::step]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c=color, alpha=0.2, zorder=2,
                   rasterized=True)

    best_f = float("inf")
    traj_x: list[np.ndarray] = []
    for x, f_val in zip(result.history_x, result.history_best):
        if f_val < best_f:
            best_f = f_val
            traj_x.append(x.copy())
    if len(traj_x) > 1:
        traj = np.array(traj_x)
        ax.plot(traj[:, 0], traj[:, 1], "-o", color=color, markersize=2,
                linewidth=1.0, zorder=3)

    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            ax.plot(opt[0], opt[1], "*", color="yellow", markersize=8,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(method_name)


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
        color = _COLORS[idx % len(_COLORS)]
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


# ---------------------------------------------------------------------------
# Public: save SVG per function (landscape + convergence)
# ---------------------------------------------------------------------------

def save_function_figure(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
) -> None:
    """Function landscape (2D contour + 3D surface) and convergence curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if benchmark.dim == 2:
        X, Y, Z = _contour_data(benchmark)
        Z_plot = np.log1p(Z - Z.min() + 1e-10)
        lo, hi = benchmark.bounds

        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.5, 1], wspace=0.35)

        # 2D landscape
        ax_land = fig.add_subplot(gs[0])
        ax_land.contourf(X, Y, Z_plot, levels=40, cmap="viridis", alpha=0.85)
        ax_land.contour(X, Y, Z_plot, levels=20, colors="white", linewidths=0.3, alpha=0.4)
        if benchmark.optima_pos:
            for opt in benchmark.optima_pos:
                ax_land.plot(opt[0], opt[1], "o", color="yellow", markersize=9,
                             markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        ax_land.set_xlim(lo, hi); ax_land.set_ylim(lo, hi)
        ax_land.set_xlabel(r"$x_1$"); ax_land.set_ylabel(r"$x_2$")
        ax_land.set_title("Landscape")

        # Convergence
        ax_conv = fig.add_subplot(gs[1])
        _draw_convergence(ax_conv, benchmark, results_per_method,
                          title="Convergence")

        # 3D surface
        ax_surf = fig.add_subplot(gs[2], projection="3d")
        _draw_surface3d(ax_surf, benchmark, X, Y, Z)

    else:
        fig, ax_conv = plt.subplots(figsize=(8, 5))
        _draw_convergence(ax_conv, benchmark, results_per_method,
                          title=f"{benchmark.name}  [{benchmark.category}]  — Convergence")

    fig.suptitle(f"{benchmark.name}  [{benchmark.category}]", fontsize=12)
    fig.savefig(output_dir / f"{benchmark.name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helper: build method-grid figure for GIFs
# ---------------------------------------------------------------------------

def _make_grid_fig(n_methods: int) -> tuple:
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False, dpi=80,
    )
    axes = [axes_grid[i // n_cols][i % n_cols] for i in range(n_methods)]
    for idx in range(n_methods, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes_grid[r][c].set_visible(False)
    return fig, axes


def _draw_optima(ax: plt.Axes, benchmark: BenchmarkFunction) -> None:
    if benchmark.optima_pos:
        for opt in benchmark.optima_pos:
            ax.plot(opt[0], opt[1], "o", color="yellow", markersize=7,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=6)


# ---------------------------------------------------------------------------
# Public: GIF — one frame per run (trajectory)
# ---------------------------------------------------------------------------

def save_runs_gif(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
    fps: int = 3,
) -> None:
    """One frame per run; shows eval scatter + best trajectory for each method."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results_per_method.keys())
    n_runs = min(len(v) for v in results_per_method.values())
    lo, hi = benchmark.bounds

    X, Y, Z = _contour_data(benchmark, resolution=150)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)

    fig, axes = _make_grid_fig(len(methods))

    def draw_frame(run_idx: int) -> list:
        for ax, method, color in zip(axes, methods, _COLORS):
            ax.clear()
            ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
            ax.contour(X, Y, Z_plot, levels=10, colors="white", linewidths=0.2, alpha=0.3)
            results = results_per_method[method]
            if run_idx < len(results):
                r = results[run_idx]
                if r.history_x:
                    pts = np.array(r.history_x)
                    s = max(1, len(pts) // 1000)
                    ax.scatter(pts[::s, 0], pts[::s, 1], s=2, c=color,
                               alpha=0.2, zorder=2, rasterized=True)
                    best_f, traj = float("inf"), []
                    for x, f in zip(r.history_x, r.history_best):
                        if f < best_f:
                            best_f = f
                            traj.append(x)
                    if len(traj) > 1:
                        t = np.array(traj)
                        ax.plot(t[:, 0], t[:, 1], "-", color=color,
                                linewidth=1.2, zorder=3, alpha=0.8)
                    bidx = int(np.argmin(r.history_best))
                    bx = r.history_x[bidx]
                    dot_c = "lime" if r.best_f <= 1e-4 else "red"
                    ax.plot(bx[0], bx[1], "o", color=dot_c, markersize=7,
                            markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            _draw_optima(ax, benchmark)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
            ax.set_title(f"{method}", fontsize=8)
        fig.suptitle(
            f"{benchmark.name}  run {run_idx + 1}/{n_runs}"
            f"  (lime=success, red=failure)",
            fontsize=10,
        )
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=n_runs,
                                  interval=1000 // fps, blit=False)
    ani.save(str(output_dir / f"{benchmark.name}_runs.gif"),
             writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: GIF — evaluation point accumulation
# ---------------------------------------------------------------------------

def save_evals_gif(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
    step: int = 100,
    fps: int = 6,
) -> None:
    """Animate eval-point accumulation (best run per method)."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results_per_method.keys())
    runs = [min(results_per_method[m], key=lambda r: r.best_f) for m in methods]
    lo, hi = benchmark.bounds

    X, Y, Z = _contour_data(benchmark, resolution=150)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)

    total_evals = max(len(r.history_x) for r in runs)
    n_frames = max(1, (total_evals + step - 1) // step)

    fig, axes = _make_grid_fig(len(methods))

    def draw_frame(frame_idx: int) -> list:
        comp_limit = (frame_idx + 1) * step
        for ax, method, run, color in zip(axes, methods, runs, _COLORS):
            ax.clear()
            ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
            ax.contour(X, Y, Z_plot, levels=10, colors="white", linewidths=0.2, alpha=0.3)
            pts = run.history_x[:comp_limit]
            if pts:
                arr = np.array(pts)
                ax.scatter(arr[:, 0], arr[:, 1], s=3, c=color, alpha=0.3, zorder=2)
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
            ax.set_title(f"{method}  e={n_shown}  f={bv:.2e}", fontsize=8)
        fig.suptitle(f"{benchmark.name}  [{benchmark.category}]  — eval accumulation",
                     fontsize=10)
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    ani.save(str(output_dir / f"{benchmark.name}_evals.gif"),
             writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public: GIF — population placement over generations
# ---------------------------------------------------------------------------

def save_population_gif(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
    pop_frames: int = 30,
    fps: int = 6,
) -> None:
    """Animate population distribution over generations (best run per method)."""
    if benchmark.dim != 2:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results_per_method.keys())
    runs = [min(results_per_method[m], key=lambda r: r.best_f) for m in methods]
    lo, hi = benchmark.bounds

    X, Y, Z = _contour_data(benchmark, resolution=150)
    Z_plot = np.log1p(Z - Z.min() + 1e-10)

    def get_frames(run: OptimizeResult) -> list[np.ndarray]:
        pops = run.history_pop
        if not pops:
            return []
        s = max(1, len(pops) // pop_frames)
        return [pops[min(i * s, len(pops) - 1)] for i in range(pop_frames)]

    pop_seqs = [get_frames(r) for r in runs]
    n_frames = max((len(f) for f in pop_seqs), default=1)

    fig, axes = _make_grid_fig(len(methods))

    def draw_frame(frame_idx: int) -> list:
        for ax, method, frames, color in zip(axes, methods, pop_seqs, _COLORS):
            ax.clear()
            ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.7)
            ax.contour(X, Y, Z_plot, levels=10, colors="white", linewidths=0.2, alpha=0.3)
            fi = min(frame_idx, len(frames) - 1) if frames else -1
            if fi >= 0 and len(frames[fi]) > 0:
                pop = frames[fi]
                ax.scatter(pop[:, 0], pop[:, 1], s=35, c=color,
                           edgecolors="white", linewidths=0.3, zorder=4, alpha=0.9)
            _draw_optima(ax, benchmark)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
            ax.set_title(f"{method}  gen={frame_idx + 1}", fontsize=8)
        fig.suptitle(f"{benchmark.name}  [{benchmark.category}]  — population",
                     fontsize=10)
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    ani.save(str(output_dir / f"{benchmark.name}_population.gif"),
             writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# keep for backward compatibility
def save_combined_gif(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
    **kwargs: object,
) -> None:
    save_evals_gif(benchmark, results_per_method, output_dir)
    save_population_gif(benchmark, results_per_method, output_dir)


# ---------------------------------------------------------------------------
# Public: stats CSV + summary CSV (unchanged API)
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
    fieldnames_s = ["function", "category", "method", "mean_time_s",
                    "mean_best_f", "success_rate", "mean_optima_found", "mean_optima_rate"]
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_s)
        if not summary_exists:
            writer.writeheader()
        for method, results in results_per_method.items():
            times = times_per_method.get(method, [0.0] * len(results))
            optima_counts = [_count_optima_found(r, benchmark, success_threshold) for r in results]
            best_fs = [r.best_f for r in results]
            mean_optima = float(np.mean(optima_counts))
            writer.writerow({
                "function": benchmark.name,
                "category": benchmark.category,
                "method": method,
                "mean_time_s": f"{np.mean(times):.3f}",
                "mean_best_f": f"{np.mean(best_fs):.4e}",
                "success_rate": f"{np.mean(np.array(best_fs) <= success_threshold):.0%}",
                "mean_optima_found": f"{mean_optima:.2f}",
                "mean_optima_rate": f"{mean_optima / n_optima_total:.2f}" if n_optima_total else "N/A",
            })


# ---------------------------------------------------------------------------
# Legacy wrappers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def save_results(
    benchmark: BenchmarkFunction,
    results_per_method: dict[str, list[OptimizeResult]],
    output_dir: str | Path = "results",
) -> None:
    save_function_figure(benchmark, results_per_method, output_dir)


def plot_surface_3d(benchmark: BenchmarkFunction, output_dir: str | Path = "results",
                    **_: object) -> None:
    pass  # now embedded in save_function_figure


def animate_comparison(benchmark: BenchmarkFunction,
                       results_per_method: dict[str, list[OptimizeResult]],
                       output_dir: str | Path = "results", **_: object) -> None:
    pass  # now embedded in save_combined_gif


def animate_population(benchmark: BenchmarkFunction,
                       results_per_method: dict[str, list[OptimizeResult]],
                       output_dir: str | Path = "results", **_: object) -> None:
    pass  # now embedded in save_combined_gif
