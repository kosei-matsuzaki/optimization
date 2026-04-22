import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from benchmarks import BENCHMARKS, BENCHMARKS_3D, BENCHMARKS_4D
from optimizers import (
    CMAESOptimizer, VirusOptimizer, PSOOptimizer, GAOptimizer,
    VOAOptimizer, SaVOAOptimizer,
)
from runner import run_experiment, summarize
from visualize import save_function_figure, save_combined_gif, save_stats


N_RUNS = 30
MAX_EVALS = 5000
OUTPUT_DIR = Path("results")

_BASE_OPTIMIZERS = {
    "PSO":   (PSOOptimizer,   {}),
    "GA":    (GAOptimizer,    {}),
    "VOA":   (VOAOptimizer,   {}),
    "SaVOA": (SaVOAOptimizer, {}),
    "VSO":   (VirusOptimizer, {}),
}


def _make_optimizers(sigma0: float) -> dict:
    return {
        "CMA-ES": (CMAESOptimizer, {"sigma0": sigma0}),
        **_BASE_OPTIMIZERS,
    }


def run_dimension(bench_list, dim_label: str) -> None:
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

    for bench in bench_list:
        sigma0 = 0.2 * (bench.bounds[1] - bench.bounds[0])
        optimizers = _make_optimizers(sigma0)
        results_per_method: dict = {}
        times_per_method: dict = {}

        for method_name, (cls, kwargs) in optimizers.items():
            results, times = run_experiment(
                cls, bench,
                n_runs=N_RUNS,
                max_evals=MAX_EVALS,
                **kwargs,
            )
            results_per_method[method_name] = results
            times_per_method[method_name] = times
            s = summarize(results)
            print(
                f"{bench.name:<18} {bench.category:<14} {method_name:<12} "
                f"{s['mean']:>12.4e} {s['std']:>12.4e} "
                f"{s['median']:>12.4e} {s['success_rate']:>7.0%} "
                f"{sum(times)/len(times):>9.2f}"
            )

        save_function_figure(bench, results_per_method, output_dir=output_dir)
        if bench.dim == 2:
            save_combined_gif(bench, results_per_method, output_dir=output_dir)
        save_stats(bench, results_per_method, times_per_method, output_dir=output_dir)

    print(f"\nResults saved to: {output_dir.resolve()}/")


def main() -> None:
    run_dimension(BENCHMARKS,    "dim2")
    # run_dimension(BENCHMARKS_3D, "dim3")
    # run_dimension(BENCHMARKS_4D, "dim4")


if __name__ == "__main__":
    main()
