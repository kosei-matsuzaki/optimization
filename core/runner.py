import time
import numpy as np
from .benchmarks import BenchmarkFunction
from .optimizers import BaseOptimizer, OptimizeResult

try:
    from scipy.stats import wilcoxon as _wilcoxon, mannwhitneyu as _mannwhitneyu
    _HAS_SCIPY = True
except ImportError:
    _wilcoxon = None
    _mannwhitneyu = None
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


def ecdf_auc(
    results: list[OptimizeResult],
    targets: tuple[float, ...] = SR_THRESHOLDS,
    max_evals: int | None = None,
) -> float:
    """Aggregate ECDF area-under-curve over (run × target) pairs, log-budget.

    For each (run, target) pair, evals-to-reach e is the first eval at which
    best_f ≤ target; +∞ if never. The aggregate ECDF at budget b is the
    fraction of pairs whose e ≤ b. AUC is computed on a log-budget axis
    from 1 to ``max_evals`` and normalised to [0, 1] — closer to 1 means
    targets are reached earlier and more often.

    Robust against SR=0 (still differentiates by how close runs got to a
    weaker target) and removes ERT's "inf when no success" failure mode.
    """
    n_runs = len(results)
    T = len(targets)
    if n_runs == 0 or T == 0:
        return 0.0
    if max_evals is None:
        max_evals = max((len(r.history_f) for r in results), default=0)
    if max_evals < 2:
        return 0.0
    log_max = np.log(max_evals)
    total_pairs = n_runs * T
    auc_sum = 0.0
    for r in results:
        # Running min lets us compute evals-to-target for all thresholds in one pass.
        running = float("inf")
        # For each target, find the first eval where running ≤ target. We do
        # all targets jointly by checking remaining unmet targets at each step.
        unmet = list(targets)
        first_hit: dict[float, int] = {}
        for i, f in enumerate(r.history_f):
            if f < running:
                running = f
            if not unmet:
                break
            still = []
            for thr in unmet:
                if running <= thr:
                    first_hit[thr] = i + 1
                else:
                    still.append(thr)
            unmet = still
        for thr in targets:
            e = first_hit.get(thr)
            if e is None or e > max_evals:
                continue
            # Pair contributes (log_max - log(e)) to the integral; / (log_max × total_pairs) at end.
            auc_sum += log_max - np.log(e)
    return float(auc_sum / (log_max * total_pairs))


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


def vargha_delaney_a12(
    candidate_best_fs: np.ndarray,
    reference_best_fs: np.ndarray,
) -> tuple[float, str]:
    """Vargha–Delaney A₁₂ effect size for minimisation.

    A₁₂ = P(candidate < reference) + 0.5 · P(candidate == reference).
    A₁₂ > 0.5 → candidate is *more likely* to be lower (better);
    < 0.5 → worse. Magnitude bins follow Vargha & Delaney (2000):
    |A₁₂ − 0.5| ≤ 0.06 negligible, ≤ 0.14 small, ≤ 0.21 medium, else large.
    """
    cand = np.asarray(candidate_best_fs, dtype=float)
    ref  = np.asarray(reference_best_fs,  dtype=float)
    n1, n2 = len(cand), len(ref)
    if n1 == 0 or n2 == 0:
        return float("nan"), "n/a"
    # Count wins (cand < ref) and ties pairwise.
    wins  = float(np.sum(cand[:, None] <  ref[None, :]))
    ties  = float(np.sum(cand[:, None] == ref[None, :]))
    a12 = (wins + 0.5 * ties) / (n1 * n2)
    d = abs(a12 - 0.5)
    if d <= 0.06:   mag = "negligible"
    elif d <= 0.14: mag = "small"
    elif d <= 0.21: mag = "medium"
    else:           mag = "large"
    return float(a12), mag


def wilcoxon_vs_reference(
    candidate_best_fs: np.ndarray,
    reference_best_fs: np.ndarray,
) -> dict:
    """Paired Wilcoxon signed-rank test plus Vargha–Delaney A₁₂ effect size.

    Returns dict with: ``p_value`` (two-sided Wilcoxon), ``p_less``
    (candidate is better), ``win_count``/``tie_count``, ``a12`` and
    ``a12_magnitude`` (negligible/small/medium/large). p values are NaN
    when all paired differences are zero or scipy is unavailable.
    """
    cand = np.asarray(candidate_best_fs, dtype=float)
    ref = np.asarray(reference_best_fs, dtype=float)
    diff = cand - ref
    win_count = int(np.sum(diff < 0))
    tie_count = int(np.sum(diff == 0))
    a12, a12_mag = vargha_delaney_a12(cand, ref)
    base = {"win_count": win_count, "tie_count": tie_count,
            "n": int(len(diff)), "a12": a12, "a12_magnitude": a12_mag}
    if not _HAS_SCIPY or len(diff) < 2 or np.all(diff == 0):
        return {"p_value": float("nan"), "p_less": float("nan"), **base}
    try:
        # Two-sided: any direction of difference
        p_two = float(_wilcoxon(cand, ref, zero_method="wilcox").pvalue)
        # One-sided "candidate less" (i.e., candidate is better since minimization)
        p_less = float(_wilcoxon(cand, ref, alternative="less",
                                 zero_method="wilcox").pvalue)
    except (ValueError, RuntimeError):
        p_two = float("nan")
        p_less = float("nan")
    return {"p_value": p_two, "p_less": p_less, **base}
