import time
import numpy as np
from .benchmarks import BenchmarkFunction
from .optimizers import BaseOptimizer, OptimizeResult


def run_experiment(
    optimizer_cls: type[BaseOptimizer],
    benchmark: BenchmarkFunction,
    n_runs: int = 10,
    max_evals: int = 5000,
    **optimizer_kwargs,
) -> tuple[list[OptimizeResult], list[float]]:
    results: list[OptimizeResult] = []
    times: list[float] = []
    for i in range(n_runs):
        opt = optimizer_cls(benchmark, seed=i * 100, **optimizer_kwargs)
        t0 = time.perf_counter()
        results.append(opt.optimize(max_evals=max_evals))
        times.append(time.perf_counter() - t0)
    return results, times


def summarize(
    results: list[OptimizeResult],
    success_threshold: float = 1e-4,
) -> dict:
    best_fs = np.array([r.best_f for r in results])
    return {
        "mean":         float(np.mean(best_fs)),
        "std":          float(np.std(best_fs)),
        "median":       float(np.median(best_fs)),
        "min":          float(np.min(best_fs)),
        "max":          float(np.max(best_fs)),
        "success_rate": float(np.mean(best_fs <= success_threshold)),
        "n_runs":       len(results),
    }
