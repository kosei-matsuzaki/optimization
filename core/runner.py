import time
import numpy as np
from .benchmarks import BenchmarkFunction
from .optimizers import BaseOptimizer, OptimizeResult

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAS_SCIPY = True
except ImportError:
    _wilcoxon = None
    _HAS_SCIPY = False


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


# BBOB-style ECDF target thresholds (log-spaced)
SR_THRESHOLDS: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-10)


def summarize(
    results: list[OptimizeResult],
    success_threshold: float = 1e-4,
) -> dict:
    """Return statistics including ERT (BBOB standard) and SR over multiple targets.

    ERT = total evals across all runs (failures counted at max budget) / # successes.
    Inf when no run succeeds.

    SR keys ``sr_{threshold}`` (sr_1e-1 .. sr_1e-10) give the BBOB-style ECDF
    profile — what fraction of runs hit each target precision.
    """
    best_fs = np.array([r.best_f for r in results])
    n_success = int(np.sum(best_fs <= success_threshold))
    evals_list = [_evals_to_target(r, success_threshold) for r in results]
    ert = float(sum(evals_list) / n_success) if n_success > 0 else float("inf")
    out: dict = {
        "mean":         float(np.mean(best_fs)),
        "std":          float(np.std(best_fs)),
        "median":       float(np.median(best_fs)),
        "min":          float(np.min(best_fs)),
        "max":          float(np.max(best_fs)),
        "success_rate": float(np.mean(best_fs <= success_threshold)),
        "ert":          ert,
        "n_runs":       len(results),
    }
    for thr in SR_THRESHOLDS:
        out[f"sr_{thr:.0e}".replace("e-0", "e-")] = float(np.mean(best_fs <= thr))
    return out


def wilcoxon_vs_reference(
    candidate_best_fs: np.ndarray,
    reference_best_fs: np.ndarray,
) -> dict:
    """Paired Wilcoxon signed-rank test comparing two methods over matched seeds.

    Returns dict with keys ``p_value`` (two-sided), ``p_less``
    (candidate < reference, i.e. candidate is better), and ``win_count``
    (# seeds where candidate strictly beat reference). When all paired
    differences are zero or scipy is unavailable, p_value/p_less are NaN.
    """
    cand = np.asarray(candidate_best_fs, dtype=float)
    ref = np.asarray(reference_best_fs, dtype=float)
    diff = cand - ref
    win_count = int(np.sum(diff < 0))
    tie_count = int(np.sum(diff == 0))
    if not _HAS_SCIPY or len(diff) < 2 or np.all(diff == 0):
        return {"p_value": float("nan"), "p_less": float("nan"),
                "win_count": win_count, "tie_count": tie_count, "n": int(len(diff))}
    try:
        # Two-sided: any direction of difference
        p_two = float(_wilcoxon(cand, ref, zero_method="wilcox").pvalue)
        # One-sided "candidate less" (i.e., candidate is better since minimization)
        p_less = float(_wilcoxon(cand, ref, alternative="less",
                                 zero_method="wilcox").pvalue)
    except (ValueError, RuntimeError):
        p_two = float("nan")
        p_less = float("nan")
    return {"p_value": p_two, "p_less": p_less,
            "win_count": win_count, "tie_count": tie_count, "n": int(len(diff))}
