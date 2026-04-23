import warnings
warnings.filterwarnings("ignore")
import os
from multiprocessing import Pool
from pathlib import Path
from core.benchmarks import BENCHMARKS, BENCHMARKS_3D, BENCHMARKS_4D, CUSTOM_BENCHMARKS, BenchmarkFunction
from core.optimizers import (
    CMAESOptimizer, VirusOptimizer, PSOOptimizer, GAOptimizer,
    VOAOptimizer, SaVOAOptimizer, GeneticVirusOptimizer,
)
from core.runner import run_experiment, summarize
from core.visualize import (
    save_function_figure, save_runs_gif, save_evals_gif, save_population_gif,
    save_3d_evals_gif, save_3d_population_gif, save_stats,
)


N_RUNS = 30
MAX_EVALS = 5000
OUTPUT_DIR = Path("results")

_BASE_OPTIMIZERS = {
    "PSO":   (PSOOptimizer,          {}),
    "GA":    (GAOptimizer,           {}),
    "VOA":   (VOAOptimizer,          {}),
    "SaVOA": (SaVOAOptimizer,        {}),
    "VSO":   (VirusOptimizer,        {}),
    "GVO":   (GeneticVirusOptimizer, {}),
}


def _make_optimizers(sigma0: float) -> dict:
    return {
        "CMA-ES": (CMAESOptimizer, {"sigma0": sigma0}),
        **_BASE_OPTIMIZERS,
    }


def _process_bench(args: tuple) -> list[tuple]:
    """Worker: run all optimizers on one benchmark and return result rows.

    Receives (bench_name, bench_dim, ...) instead of a BenchmarkFunction object
    to avoid pickling ioh closures, which are not serialisable.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")

    bench_name, bench_dim, n_runs, max_evals, output_dir_str = args
    output_dir = Path(output_dir_str)

    # Reconstruct benchmark locally so ioh closures stay within this process.
    from core.benchmarks import _make_bbob, _himmelblau, _six_hump_camel, _BBOB_SPECS
    if bench_name == "C01-Himmelblau":
        bench = _himmelblau()
    elif bench_name == "C02-SixHumpCamel":
        bench = _six_hump_camel()
    else:
        spec = next((s for s in _BBOB_SPECS if s[1] == bench_name), None)
        if spec is None:
            raise ValueError(f"Unknown benchmark: {bench_name}")
        bench = _make_bbob(spec[0], spec[1], spec[2], bench_dim)

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

    save_function_figure(bench, results_per_method, output_dir=output_dir)
    if bench.dim == 2:
        save_runs_gif(bench, results_per_method, output_dir=output_dir)
        save_evals_gif(bench, results_per_method, output_dir=output_dir)
        save_population_gif(bench, results_per_method, output_dir=output_dir)
    elif bench.dim == 3:
        save_3d_evals_gif(bench, results_per_method, output_dir=output_dir)
        save_3d_population_gif(bench, results_per_method, output_dir=output_dir)
    save_stats(bench, results_per_method, times_per_method, output_dir=output_dir)

    return rows


def run_dimension(bench_list: list[BenchmarkFunction], dim_label: str) -> None:
    import shutil
    output_dir = OUTPUT_DIR / dim_label
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*96}")
    print(f"  Dimension: {dim_label}")
    print(f"{'='*96}")
    print(f"{'Function':<18} {'Category':<14} {'Method':<12} "
          f"{'Mean':>12} {'Std':>12} {'Median':>12} {'Success':>8} {'Time(s)':>9}")
    print("-" * 96)

    n_workers = min(os.cpu_count() or 2, len(bench_list))
    args = [(bench.name, bench.dim, N_RUNS, MAX_EVALS, str(output_dir)) for bench in bench_list]

    with Pool(n_workers) as pool:
        all_rows = pool.map(_process_bench, args)

    for bench, rows in zip(bench_list, all_rows):
        for name, category, method_name, s, avg_time in rows:
            print(
                f"{name:<18} {category:<14} {method_name:<12} "
                f"{s['mean']:>12.4e} {s['std']:>12.4e} "
                f"{s['median']:>12.4e} {s['success_rate']:>7.0%} "
                f"{avg_time:>9.2f}"
            )

    print(f"\nResults saved to: {output_dir.resolve()}/")


def main() -> None:
    run_dimension(BENCHMARKS + CUSTOM_BENCHMARKS, "dim2")
    run_dimension(BENCHMARKS_3D,                  "dim3")


if __name__ == "__main__":
    main()
