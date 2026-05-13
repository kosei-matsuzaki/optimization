"""Lightweight local sanity check.

Runs a small subset of BBOB functions with reduced settings so results
are visible in under a minute. Full experiments go through GitHub Actions.

Usage:
    python quick_check.py
    python quick_check.py --n-runs 5 --max-evals 3000
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
from pathlib import Path

from core.benchmarks import BENCHMARKS_BY_NAME
from core.optimizers import (
    CMAESOptimizer, MultiChannelEpidemicOptimizer, PSOOptimizer,
    GAOptimizer, SaVOAOptimizer,
)
from core.runner import run_experiment, summarize, wilcoxon_vs_reference
from core.visualize import (
    save_landscape_svg, save_convergence_svg,
    save_method_runs_anim, save_method_evals_anim, save_method_population_anim,
    save_method_vso_svg, save_stats,
)


def _append_wilcoxon(dim_dir: Path, bench_name: str,
                     results_per_method: dict, reference: str = "MC-ESO") -> None:
    """Append paired Wilcoxon signed-rank rows comparing ``reference`` vs each
    other method, for this benchmark's run results."""
    if reference not in results_per_method:
        return
    ref_bests = np.array([r.best_f for r in results_per_method[reference]])
    path = dim_dir / "wilcoxon.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "function", "reference", "method", "n", "win_count", "tie_count",
            "p_value_two_sided", "p_value_ref_better",
        ])
        if write_header:
            writer.writeheader()
        for method, results in results_per_method.items():
            if method == reference:
                continue
            method_bests = np.array([r.best_f for r in results])
            # We test "is reference better than method?" → reference (cand) < method (ref)
            stat = wilcoxon_vs_reference(ref_bests, method_bests)
            writer.writerow({
                "function": bench_name,
                "reference": reference,
                "method": method,
                "n": stat["n"],
                "win_count": stat["win_count"],
                "tie_count": stat["tie_count"],
                "p_value_two_sided": f"{stat['p_value']:.4g}",
                "p_value_ref_better": f"{stat['p_less']:.4g}",
            })

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

# Full BBOB-26 set (F01-F24 + C01-C02), used when --all is passed.
# Built programmatically from the registry so adding new benchmarks doesn't
# require touching this list.
_ALL_FUNCTIONS: list[tuple[str, int]] = (
    [(f"F{i:02d}-{name.split('-', 1)[1]}", 2)
     for i in range(1, 25)
     for name in BENCHMARKS_BY_NAME
     if name.startswith(f"F{i:02d}-")]
    + [("C01-Himmelblau", 2), ("C02-SixHumpCamel", 2)]
)

_DIM_LOOKUP: dict[int, dict[str, object]] = {
    2: BENCHMARKS_BY_NAME,
}

# MC-ESO (Multi-Channel Epidemic Spread Optimizer): all core mechanisms
# (3-channel transmission + host competition + diversified spillover +
# gated σ adapt) are baked into the base implementation.
_MCESO_VARIANTS: dict[str, dict] = {
    "MC-ESO":  {},
}

_OPTIMIZERS = {
    "CMA-ES": (CMAESOptimizer, {}),
    "PSO":    (PSOOptimizer,   {}),
    "GA":     (GAOptimizer,    {}),
    "SaVOA":  (SaVOAOptimizer, {}),
    **{name: (MultiChannelEpidemicOptimizer, kw) for name, kw in _MCESO_VARIANTS.items()},
}


def _run_dim(benchmarks: list, dim_dir: Path, n_runs: int, max_evals: int) -> None:
    """Run all functions in a dimension group and save results to dim_dir."""
    dim_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'Function':<22} {'Method':<10} {'Mean':>12} "
          f"{'SR@1e-1':>7} {'SR@1e-2':>7} {'SR@1e-4':>7} {'SR@1e-7':>7} {'SR@1e-10':>8} {'ERT':>9}")
    print("-" * 100)
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
            ert_str = f"{s['ert']:>9.0f}" if s['ert'] < float('inf') else "      ---"
            print(
                f"{bench.name:<22} {method:<10} "
                f"{s['mean']:>12.4e} "
                f"{s['sr_1e-1']:>6.0%} {s['sr_1e-2']:>6.0%} {s['sr_1e-4']:>6.0%} "
                f"{s['sr_1e-7']:>6.0%} {s['sr_1e-10']:>7.0%}{ert_str}"
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
        _append_wilcoxon(dim_dir, bench.name, results_per_method, reference="MC-ESO")

    print(f"Saved → {dim_dir.resolve()}/")


def main(
    n_runs: int = 10,
    max_evals: int = 2000,
    output_dir: Path = Path("results/quick"),
    funcs: list[str] | None = None,
    use_all: bool = False,
) -> None:
    output_dir = Path(output_dir)
    func_set = _ALL_FUNCTIONS if use_all else _QUICK_FUNCTIONS
    print(f"quick_check  n_runs={n_runs}  max_evals={max_evals}  "
          f"set={'all-26' if use_all else 'quick-12'}  funcs={funcs or 'all'}")

    func_filter = set(funcs) if funcs else None

    # Group BenchmarkFunction objects by dimension
    benchmarks_by_dim: dict[int, list] = {}
    for fname, dim in func_set:
        if func_filter is not None and fname not in func_filter:
            continue
        bench = _DIM_LOOKUP[dim][fname]
        benchmarks_by_dim.setdefault(dim, []).append(bench)

    if not benchmarks_by_dim:
        raise SystemExit(f"No matching functions for filter {funcs}")

    for dim in sorted(benchmarks_by_dim):
        print(f"\n=== dim{dim} ===")
        _run_dim(benchmarks_by_dim[dim], output_dir / f"dim{dim}", n_runs, max_evals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",     type=int, default=10,                   help="Number of runs per method")
    parser.add_argument("--max-evals",  type=int, default=2000,                 help="Max function evaluations per run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/quick"), help="Output directory")
    parser.add_argument("--funcs",      type=str, default=None,
                        help="Comma-separated function names to run (default: all in selected set)")
    parser.add_argument("--all",        action="store_true",
                        help="Use the full BBOB-26 set (F01-F24 + C01-C02) instead of the quick-12 subset")
    args = parser.parse_args()
    funcs_list = [s.strip() for s in args.funcs.split(",")] if args.funcs else None
    main(n_runs=args.n_runs, max_evals=args.max_evals, output_dir=args.output_dir,
         funcs=funcs_list, use_all=args.all)
