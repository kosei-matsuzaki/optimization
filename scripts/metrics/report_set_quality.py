"""Two questions the F1 table raises:
 (a) is MC-ESO's low precision real redundancy, or just an unfiltered report?
 (b) how much does the score depend on the niche radius rho?
Both are scored off the SAME runs, so neither costs extra evaluations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME
from core.optimizers import MultiChannelEpidemicOptimizer, NMMSOOptimizer
from core.runner import count_goptima, _seed_indices

FUNCS = ["N04-Himmelblau", "N05-SixHumpCamel", "N06-Shubert2D", "N07-Vincent2D",
         "N10-ModRastrigin2D"]
SEEDS = 10
EVALS = 5000
EPS = 1e-3


def dedup(X, F, rho):
    """The rho-greedy filter, best-f first: drop points sharing a basin with a
    better reported point. Post-processing only, zero evaluations."""
    order = np.argsort(F)
    keep = _seed_indices(X[order], rho)
    return X[order][keep], F[order][keep]


def f1(c, n_rep, k):
    p = c / n_rep if n_rep else 0.0
    r = c / k
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


print(f"eps={EPS:g} seeds={SEEDS} evals={EVALS}\n")
print(f"{'function':<20}{'K':>5}{'method':<10}"
      f"{'|rep|':>7}{'PR':>7}{'pre':>7}{'F1':>7}   "
      f"{'|ded|':>7}{'PRd':>7}{'pred':>7}{'F1d':>7}   "
      f"{'PR@.5r':>8}{'PR@2r':>8}")
print("-" * 108)
for name in FUNCS:
    b = NICHING_BENCHMARKS_BY_NAME[name]
    k, rho = b.n_global_optima, b.niche_rho
    cap = max(100, 2 * k)
    for label, cls in (("MC-ESO", MultiChannelEpidemicOptimizer), ("NMMSO", NMMSOOptimizer)):
        acc = []
        for s in range(SEEDS):
            r = cls(b, seed=s * 100).optimize(EVALS)
            X = np.asarray(r.final_solutions or [r.best_x], dtype=float)
            F = np.array([float(b.func(x)) for x in X])
            if len(F) > cap:
                keep = np.argsort(F)[:cap]
                X, F = X[keep], F[keep]
            c = count_goptima(X, F, k, rho, EPS)
            Xd, Fd = dedup(X, F, rho)
            cd = count_goptima(Xd, Fd, k, rho, EPS)
            acc.append((len(F), c / k, c / len(F), f1(c, len(F), k),
                        len(Fd), cd / k, cd / len(Fd), f1(cd, len(Fd), k),
                        count_goptima(X, F, k, rho * 0.5, EPS) / k,
                        count_goptima(X, F, k, rho * 2.0, EPS) / k))
        m = np.mean(np.array(acc), axis=0)
        print(f"{name:<20}{k:>5}{label:<10}"
              f"{m[0]:>7.1f}{m[1]:>7.2f}{m[2]:>7.2f}{m[3]:>7.2f}   "
              f"{m[4]:>7.1f}{m[5]:>7.2f}{m[6]:>7.2f}{m[7]:>7.2f}   "
              f"{m[8]:>8.2f}{m[9]:>8.2f}")
