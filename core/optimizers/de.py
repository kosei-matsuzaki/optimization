"""Differential Evolution baseline (rand/1/bin)."""
from __future__ import annotations
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class DEOptimizer(BaseOptimizer):
    """Differential Evolution / rand/1/bin (Storn & Price, 1997).

    For each target x_i in the current population, three distinct donors
    a, b, c are picked uniformly at random (all ≠ i) and a mutant is formed
    as ``v = x_a + F·(x_b - x_c)``. Binomial crossover with x_i (rate CR,
    plus one guaranteed inherited dimension) yields the trial u, which
    replaces x_i only if it improves on it. Updates are synchronous: all
    donors are drawn from the current generation, all replacements applied
    at gen end.

    DE/rand/1/bin is the canonical single-channel differential baseline
    against which MC-ESO's droplet channel (DE/current-to-best/1 with
    niched-elite pull) is contrasted.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 30,
        F: float = 0.5,        # mutation scale
        CR: float = 0.9,       # binomial crossover rate
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.F = F
        self.CR = CR

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds

        pop = rng.uniform(lo, hi, (self.n_pop, self.dim))
        fit = np.array([self.func(x) for x in pop])

        history_x, history_f, history_pop = self._init_population_history(pop, fit)

        while len(history_f) < max_evals:
            new_pop = pop.copy()
            new_fit = fit.copy()
            for i in range(self.n_pop):
                if len(history_f) >= max_evals:
                    break
                # Three distinct donors ≠ i
                others = np.array([k for k in range(self.n_pop) if k != i])
                a, b, c = rng.choice(others, size=3, replace=False)
                v = pop[a] + self.F * (pop[b] - pop[c])
                v = np.clip(v, lo, hi)
                # Binomial crossover; force ≥1 dim inherited from v
                mask = rng.random(self.dim) < self.CR
                mask[rng.integers(0, self.dim)] = True
                u = np.where(mask, v, pop[i])
                f_u = float(self.func(u))
                history_x.append(u.copy())
                history_f.append(f_u)
                if f_u <= fit[i]:
                    new_pop[i] = u
                    new_fit[i] = f_u
            pop = new_pop
            fit = new_fit
            history_pop.append(pop.copy())

        return self._make_result(history_x, history_f, history_pop)
