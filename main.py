import warnings
warnings.filterwarnings("ignore")
import os
from multiprocessing import Pool
from pathlib import Path
from core.benchmarks import BENCHMARKS, BENCHMARKS_3D, BENCHMARKS_4D, CUSTOM_BENCHMARKS, BenchmarkFunction
from core.optimizers import (
    CMAESOptimizer, MultiChannelEpidemicOptimizer, PSOOptimizer,
    DEOptimizer, SaVOAOptimizer,
    LSHADEOptimizer, IPOPCMAESOptimizer, BIPOPCMAESOptimizer,
)
from core.runner import run_experiment, summarize
from core.visualize import (
    save_landscape_svg, save_convergence_svg,
    save_method_runs_anim, save_method_evals_anim, save_method_population_anim,
    save_method_3devals_anim, save_method_3dpopulation_anim,
    save_method_vso_svg, save_stats,
)


N_RUNS = 100
MAX_EVALS = 5000
OUTPUT_DIR = Path("results")

_BASE_OPTIMIZERS = {
    "PSO":     (PSOOptimizer,                  {}),
    "DE":      (DEOptimizer,                   {}),
    "SaVOA":   (SaVOAOptimizer,                {}),
    "L-SHADE": (LSHADEOptimizer,               {}),
    # MC-ESO now natively does sequential niching (multi-solution) + adaptive
    # anisotropy floor — both built into the base class, no separate variant.
    "MC-ESO":  (MultiChannelEpidemicOptimizer, {}),
}


def _make_optimizers(sigma0: float) -> dict:
    return {
        "CMA-ES":      (CMAESOptimizer,     {"sigma0": sigma0}),
        "IPOP-CMA-ES": (IPOPCMAESOptimizer, {"sigma0": sigma0}),
        "BIPOP-CMA-ES": (BIPOPCMAESOptimizer, {"sigma0": sigma0}),
        **_BASE_OPTIMIZERS,
    }


def _process_bench(args: tuple) -> list[tuple]:
    """Worker: run all optimizers on one benchmark and return result rows."""
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")

    bench_name, bench_dim, n_runs, max_evals, output_dir_str = args
    output_dir = Path(output_dir_str)

    from core.benchmarks import make_benchmark_by_name
    bench = make_benchmark_by_name(bench_name, bench_dim)

    sigma0 = 0.2 * (bench.bounds[1] - bench.bounds[0])
    optimizers = _make_optimizers(sigma0)

    rows: list[tuple] = []
    results_per_method: dict = {}
    times_per_method: dict = {}

    for method_name, (cls, kwargs) in optimizers.items():
        results, times = run_experiment(
            cls, bench, n_runs=n_runs, max_evals=max_evals, **kwargs
        )
        results_per_method[method_name] = results
        times_per_method[method_name] = times
        s = summarize(results)
        rows.append((bench.name, bench.category, method_name, s, sum(times) / len(times)))

        # Per-method visualizations
        if bench.dim == 2:
            save_method_runs_anim(bench, results, method_name, output_dir=output_dir)
            save_method_evals_anim(bench, results, method_name, output_dir=output_dir, best=True)
            save_method_evals_anim(bench, results, method_name, output_dir=output_dir, best=False)
            save_method_population_anim(bench, results, method_name, output_dir=output_dir, best=True)
            save_method_population_anim(bench, results, method_name, output_dir=output_dir, best=False)
        elif bench.dim == 3:
            save_method_3devals_anim(bench, results, method_name, output_dir=output_dir, best=True)
            save_method_3devals_anim(bench, results, method_name, output_dir=output_dir, best=False)
            save_method_3dpopulation_anim(bench, results, method_name, output_dir=output_dir, best=True)
            save_method_3dpopulation_anim(bench, results, method_name, output_dir=output_dir, best=False)
        save_method_vso_svg(bench, results, method_name, output_dir=output_dir, best=True)
        save_method_vso_svg(bench, results, method_name, output_dir=output_dir, best=False)

    # Function-level outputs (all methods combined)
    save_landscape_svg(bench, output_dir=output_dir)
    save_convergence_svg(bench, results_per_method, output_dir=output_dir)
    save_stats(bench, results_per_method, times_per_method, output_dir=output_dir)

    return rows


def run_dimension(bench_list: list[BenchmarkFunction], dim_label: str) -> None:
    import shutil
    output_dir = OUTPUT_DIR / dim_label
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*110}")
    print(f"  Dimension: {dim_label}")
    print(f"{'='*110}")
    print(f"{'Function':<18} {'Category':<14} {'Method':<12} "
          f"{'Mean':>12} {'Std':>12} {'SR@1e-2':>8} {'SR@1e-4':>8} {'EvalsSucc':>10} {'Time(s)':>9}")
    print("-" * 111)

    n_workers = min(os.cpu_count() or 2, len(bench_list))
    args = [(bench.name, bench.dim, N_RUNS, MAX_EVALS, str(output_dir)) for bench in bench_list]

    with Pool(n_workers) as pool:
        all_rows = pool.map(_process_bench, args)

    for bench, rows in zip(bench_list, all_rows):
        for name, category, method_name, s, avg_time in rows:
            ev = s['evals_succ_mean']
            ev_str = f"{ev:>10.0f}" if ev < float('inf') else "       ---"
            print(
                f"{name:<18} {category:<14} {method_name:<12} "
                f"{s['mean']:>12.4e} {s['std']:>12.4e} "
                f"{s['sr_1e-2']:>7.0%} {s['success_rate']:>8.0%}"
                f"{ev_str} {avg_time:>9.2f}"
            )

    print(f"\nResults saved to: {output_dir.resolve()}/")


def main() -> None:
    run_dimension(BENCHMARKS + CUSTOM_BENCHMARKS, "dim2")
    run_dimension(BENCHMARKS_3D,                  "dim3")


if __name__ == "__main__":
    main()
