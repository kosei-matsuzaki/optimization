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

from core.benchmarks import BENCHMARKS_BY_NAME
from core.optimizers import (
    CMAESOptimizer, VirusOptimizer, PSOOptimizer,
    GAOptimizer, SaVOAOptimizer,
)
from core.runner import run_experiment, summarize
from core.visualize import (
    save_landscape_svg, save_convergence_svg,
    save_method_runs_anim, save_method_evals_anim, save_method_population_anim,
    save_method_vso_svg, save_stats,
)

# (function_name, dimension) — two representatives per BBOB group, for each dim
_QUICK_FUNCTIONS: list[tuple[str, int]] = [
    # 2D — two per group + custom
    ("F01-Sphere",           2),   # separable        — unimodal baseline
    ("F03-RastriginSep",     2),   # separable        — separable multimodal
    ("F08-Rosenbrock",       2),   # moderate-cond    — banana valley
    ("F09-RosenbrockRot",    2),   # moderate-cond    — rotated, harder
    ("F10-EllipsoidalRot",   2),   # ill-cond         — cond ≈ 10^6
    ("F12-BentCigar",        2),   # ill-cond         — extreme cond ≈ 10^6
    ("F15-RastriginRot",     2),   # multimodal       — structured landscape
    ("F17-SchafferF7",       2),   # multimodal       — irregular rough landscape
    ("F20-Schwefel",         2),   # weak-structure   — deceptive optima
    ("F21-Gallagher101",     2),   # weak-structure   — 101 Gaussian peaks
    ("C01-Himmelblau",       2),   # custom           — 4 global optima
    ("C02-SixHumpCamel",     2),   # custom           — 2 global optima
]

_DIM_LOOKUP: dict[int, dict[str, object]] = {
    2: BENCHMARKS_BY_NAME,
}

_OPTIMIZERS = {
    "sd99-mr50":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.50}),
    "sd99-mr40":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.40}),
    "sd99-mr30":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.30}),
    "sd99-mr20":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.20}),
    "sd99-mr15":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.15}),
    "sd99-mr10":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.10}),
    "sd99-mr07":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.07}),
    "sd99-mr05":  (VirusOptimizer, {"sigma_decay": 0.99,  "sigma_min_ratio": 0.05}),
    "sd999-mr50": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.50}),
    "sd999-mr40": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.40}),
    "sd999-mr30": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.30}),
    "sd999-mr20": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.20}),
    "sd999-mr15": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.15}),
    "sd999-mr10": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.10}),
    "sd999-mr07": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.07}),
    "sd999-mr05": (VirusOptimizer, {"sigma_decay": 0.999, "sigma_min_ratio": 0.05}),
}


def _run_dim(benchmarks: list, dim_dir: Path, n_runs: int, max_evals: int) -> None:
    """Run all functions in a dimension group and save results to dim_dir."""
    dim_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'Function':<22} {'Method':<10} {'Mean':>12} {'Std':>12} "
          f"{'SR@1e-2':>8} {'SR@1e-4':>8} {'ERT':>9}")
    print("-" * 83)
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
            ert_str = f"{s['ert']:>9.0f}" if s['ert'] < float('inf') else "     ---"
            print(
                f"{bench.name:<22} {method:<10} "
                f"{s['mean']:>12.4e} {s['std']:>12.4e} "
                f"{s['sr_1e-2']:>7.0%} {s['success_rate']:>8.0%}{ert_str}"
            )

        # Per-method visualizations
        for method_name, results in results_per_method.items():
            if bench.dim == 2:
                save_method_runs_anim(bench, results, method_name, output_dir=dim_dir)
                save_method_evals_anim(bench, results, method_name, output_dir=dim_dir, best=True)
                save_method_evals_anim(bench, results, method_name, output_dir=dim_dir, best=False)
                save_method_population_anim(bench, results, method_name, output_dir=dim_dir, best=True)
                save_method_population_anim(bench, results, method_name, output_dir=dim_dir, best=False)
            save_method_vso_svg(bench, results, method_name, output_dir=dim_dir, best=True)
            save_method_vso_svg(bench, results, method_name, output_dir=dim_dir, best=False)

        # Function-level outputs
        save_landscape_svg(bench, output_dir=dim_dir)
        save_convergence_svg(bench, results_per_method, output_dir=dim_dir)
        save_stats(bench, results_per_method, times_per_method, output_dir=dim_dir)

    print(f"Saved → {dim_dir.resolve()}/")


def main(n_runs: int = 10, max_evals: int = 2000, output_dir: Path = Path("results/quick")) -> None:
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
    parser.add_argument("--n-runs",     type=int, default=10,                   help="Number of runs per method")
    parser.add_argument("--max-evals",  type=int, default=2000,                 help="Max function evaluations per run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/quick"), help="Output directory")
    args = parser.parse_args()
    main(n_runs=args.n_runs, max_evals=args.max_evals, output_dir=args.output_dir)
