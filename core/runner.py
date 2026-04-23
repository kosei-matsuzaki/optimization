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


def _evals_to_target(r: OptimizeResult, threshold: float) -> int:
    """First eval (1-based) where running min ≤ threshold; else len(history_f)."""
    best = float("inf")
    for i, f in enumerate(r.history_f):
        best = min(best, f)
        if best <= threshold:
            return i + 1
    return len(r.history_f)


def summarize(
    results: list[OptimizeResult],
    success_threshold: float = 1e-4,
) -> dict:
    """Return statistics including ERT (Expected Running Time, BBOB standard).

    ERT = total evals across all runs (penalizing failures with max budget)
          / number of successful runs.
    Inf when no run succeeds.
    """
    best_fs = np.array([r.best_f for r in results])
    n_success = int(np.sum(best_fs <= success_threshold))
    evals_list = [_evals_to_target(r, success_threshold) for r in results]
    ert = float(sum(evals_list) / n_success) if n_success > 0 else float("inf")
    return {
        "mean":         float(np.mean(best_fs)),
        "std":          float(np.std(best_fs)),
        "median":       float(np.median(best_fs)),
        "min":          float(np.min(best_fs)),
        "max":          float(np.max(best_fs)),
        "success_rate": float(np.mean(best_fs <= success_threshold)),
        "sr_1e-2":      float(np.mean(best_fs <= 1e-2)),
        "ert":          ert,
        "n_runs":       len(results),
    }
