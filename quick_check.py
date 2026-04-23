"""Lightweight local sanity check.

Runs a small subset of BBOB functions with reduced settings so results
are visible in under a minute. Full experiments go through GitHub Actions.

Usage:
    python quick_check.py
    python quick_check.py --n-runs 5 --max-evals 3000
"""
from __future__ import annotations
import argparse
from pathlib import Path

from core.benchmarks import BENCHMARKS_BY_NAME, BENCHMARKS_3D_BY_NAME
from core.optimizers import (
    CMAESOptimizer, VirusOptimizer, PSOOptimizer,
    GAOptimizer, VOAOptimizer, SaVOAOptimizer, GeneticVirusOptimizer,
)
from core.runner import run_experiment, summarize
from core.visualize import (
    save_function_figure, save_runs_gif, save_evals_gif, save_population_gif,
    save_3d_evals_gif, save_3d_population_gif, save_stats,
)

# (function_name, dimension) — one representative per BBOB group, for each dim
_QUICK_FUNCTIONS: list[tuple[str, int]] = [
    # 2D
    ("F01-Sphere",       2),   # separable
    ("F08-Rosenbrock",   2),   # moderate-cond
    ("F15-RastriginRot", 2),   # multimodal
    ("F20-Schwefel",     2),   # weak-structure
    ("C01-Himmelblau",   2),   # 4 global optima
    # 3D
    ("F01-Sphere",       3),   # separable
    ("F08-Rosenbrock",   3),   # moderate-cond
    ("F15-RastriginRot", 3),   # multimodal
    ("F20-Schwefel",     3),   # weak-structure
]

_DIM_LOOKUP: dict[int, dict[str, object]] = {
    2: BENCHMARKS_BY_NAME,
    3: BENCHMARKS_3D_BY_NAME,
}

_OPTIMIZERS = {
    "CMA-ES": (CMAESOptimizer,          {}),
    "PSO":    (PSOOptimizer,            {}),
    "GA":     (GAOptimizer,             {}),
    "VOA":    (VOAOptimizer,            {}),
    "SaVOA":  (SaVOAOptimizer,          {}),
    "VSO":    (VirusOptimizer,          {}),
    "GVO":    (GeneticVirusOptimizer,   {}),
}


def _run_dim(benchmarks: list, dim_dir: Path, n_runs: int, max_evals: int) -> None:
    """Run all functions in a dimension group and save results to dim_dir."""
    dim_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'Function':<22} {'Method':<10} {'Mean':>12} {'Std':>12} {'Success':>8}")
    print("-" * 68)
    for bench in benchmarks:
        sigma0 = 0.2 * (bench.bounds[1] - bench.bounds[0])
        results_per_method: dict = {}
        times_per_method: dict = {}
        for method, (cls, kwargs) in _OPTIMIZERS.items():
            kw = {**kwargs, **({"sigma0": sigma0} if cls is CMAESOptimizer else {})}
            results, times = run_experiment(
                cls, bench, n_runs=n_runs, max_evals=max_evals, **kw
            )
            results_per_method[method] = results
            times_per_method[method] = times
            s = summarize(results)
            print(
                f"{bench.name:<22} {method:<10} "
                f"{s['mean']:>12.4e} {s['std']:>12.4e} {s['success_rate']:>7.0%}"
            )
        save_function_figure(bench, results_per_method, output_dir=dim_dir)
        if bench.dim == 2:
            save_runs_gif(bench, results_per_method, output_dir=dim_dir)
            save_evals_gif(bench, results_per_method, output_dir=dim_dir)
            save_population_gif(bench, results_per_method, output_dir=dim_dir)
        elif bench.dim == 3:
            save_3d_evals_gif(bench, results_per_method, output_dir=dim_dir)
            save_3d_population_gif(bench, results_per_method, output_dir=dim_dir)
        save_stats(bench, results_per_method, times_per_method, output_dir=dim_dir)
    print(f"Saved → {dim_dir.resolve()}/")


def main(n_runs: int = 3, max_evals: int = 2000, output_dir: Path = Path("results/quick")) -> None:
    output_dir = Path(output_dir)
    print(f"quick_check  n_runs={n_runs}  max_evals={max_evals}")

    # Group BenchmarkFunction objects by dimension
    benchmarks_by_dim: dict[int, list] = {}
    for fname, dim in _QUICK_FUNCTIONS:
        bench = _DIM_LOOKUP[dim][fname]
        benchmarks_by_dim.setdefault(dim, []).append(bench)

    for dim in sorted(benchmarks_by_dim):
        print(f"\n=== dim{dim} ===")
        _run_dim(benchmarks_by_dim[dim], output_dir / f"dim{dim}", n_runs, max_evals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",     type=int, default=3,                    help="Number of runs per method")
    parser.add_argument("--max-evals",  type=int, default=2000,                 help="Max function evaluations per run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/quick"), help="Output directory")
    args = parser.parse_args()
    main(n_runs=args.n_runs, max_evals=args.max_evals, output_dir=args.output_dir)
