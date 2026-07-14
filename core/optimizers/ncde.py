"""NCDE baseline (neighborhood-mutation crowding DE) — niching reference."""
from __future__ import annotations
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class NCDEOptimizer(BaseOptimizer):
    """Neighborhood-based Crowding Differential Evolution (Qu, Suganthan &
    Liang, 2012), on top of crowding DE (Thomsen, 2004).

    Two changes relative to the plain DE baseline (same n_pop/F/CR):

    * **Neighborhood mutation** — the three donors a, b, c are drawn from the
      ``m`` nearest population members of the target (Euclidean), not from the
      whole population. Difference vectors then stay within one basin, so each
      niche converges locally instead of being torn apart by cross-basin jumps.
    * **Crowding replacement** — the trial competes against the *nearest*
      population member instead of its own parent, applied immediately
      (steady-state). Subpopulations survive in separate basins because a
      trial can only displace its own neighbors.

    Serves as the dedicated multi-solution (peak ratio / MMO success)
    comparison for MC-ESO's sequential niching; sharing the DE lineage keeps
    the mechanism difference — parallel crowding niches vs σ-exhaustion
    sequential niching — cleanly isolated. Setting ``m >= n_pop - 1`` recovers
    plain crowding DE (global donors), which was measurably worse at deep
    precision (Himmelblau PR@1e-4 0% vs 80%+ with m=6 at 5000 evals).
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 30,
        F: float = 0.5,        # mutation scale (matches DE baseline)
        CR: float = 0.9,       # binomial crossover rate (matches DE baseline)
        m: int = 6,            # neighborhood size for donor selection
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.F = F
        self.CR = CR
        self.m = m

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        m = min(self.m, self.n_pop - 1)  # donors need m ≥ 3 neighbors

        pop = rng.uniform(lo, hi, (self.n_pop, self.dim))
        fit = np.array([self.func(x) for x in pop])

        history_x, history_f, history_pop = self._init_population_history(pop, fit)

        while len(history_f) < max_evals:
            for i in range(self.n_pop):
                if len(history_f) >= max_evals:
                    break
                # Neighborhood mutation: donors from the m nearest members.
                d = np.linalg.norm(pop - pop[i], axis=1)
                neigh = np.argsort(d)[1:m + 1]  # excludes the target itself
                a, b, c = rng.choice(neigh, size=3, replace=False)
                v = pop[a] + self.F * (pop[b] - pop[c])
                v = np.clip(v, lo, hi)
                mask = rng.random(self.dim) < self.CR
                mask[rng.integers(0, self.dim)] = True
                u = np.where(mask, v, pop[i])
                f_u = float(self.func(u))
                history_x.append(u.copy())
                history_f.append(f_u)
                # Crowding replacement: compete with the nearest member,
                # applied immediately so niches update within a generation.
                j = int(np.argmin(np.linalg.norm(pop - u, axis=1)))
                if f_u <= fit[j]:
                    pop[j] = u
                    fit[j] = f_u
            history_pop.append(pop.copy())

        return self._make_result(history_x, history_f, history_pop)
