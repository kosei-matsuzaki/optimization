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

from benchmarks import BENCHMARKS_BY_NAME
from optimizers import (
    CMAESOptimizer, VirusOptimizer, PSOOptimizer,
    GAOptimizer, VOAOptimizer, SaVOAOptimizer,
)
from runner import run_experiment, summarize
from visualize import save_function_figure, save_stats

# One representative function per BBOB group
_QUICK_FUNCTIONS = [
    "F01-Sphere",           # separable
    "F08-Rosenbrock",       # moderate-cond
    "F15-RastriginRot",     # multimodal
    "F20-Schwefel",         # weak-structure
]

_OPTIMIZERS = {
    "CMA-ES": (CMAESOptimizer, {}),
    "PSO":    (PSOOptimizer,   {}),
    "GA":     (GAOptimizer,    {}),
    "VOA":    (VOAOptimizer,   {}),
    "SaVOA":  (SaVOAOptimizer, {}),
    "VSO":    (VirusOptimizer, {}),
}


def main(n_runs: int = 3, max_evals: int = 2000, output_dir: Path = Path("results/quick")) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"quick_check  n_runs={n_runs}  max_evals={max_evals}")
    print(f"{'Function':<22} {'Method':<10} {'Mean':>12} {'Std':>12} {'Success':>8}")
    print("-" * 68)

    for fname in _QUICK_FUNCTIONS:
        bench = BENCHMARKS_BY_NAME[fname]
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

        save_function_figure(bench, results_per_method, output_dir=output_dir)
        save_stats(bench, results_per_method, times_per_method, output_dir=output_dir)

    print(f"\nFigures saved to: {output_dir.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",     type=int,          default=3,              help="Number of runs per method")
    parser.add_argument("--max-evals",  type=int,          default=2000,           help="Max function evaluations per run")
    parser.add_argument("--output-dir", type=Path,         default=Path("results/quick"), help="Output directory")
    args = parser.parse_args()
    main(n_runs=args.n_runs, max_evals=args.max_evals, output_dir=args.output_dir)
