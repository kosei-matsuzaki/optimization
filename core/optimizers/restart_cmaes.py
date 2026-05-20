"""Restart-CMA-ES baselines: IPOP (Auger & Hansen 2005) and BIPOP (Hansen 2009).

Both wrap pycma under the BaseOptimizer interface so they slot directly into
the existing runner without changes.
"""
from __future__ import annotations

import numpy as np
import cma

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _RestartCMAESBase(BaseOptimizer):
    """Shared restart-CMA-ES driver. Subclasses set the regime."""

    bipop: bool = False
    incpopsize: int = 2

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        sigma0: float = 1.0,
    ):
        super().__init__(benchmark, seed)
        self.sigma0 = sigma0

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        # default popsize for CMA-ES; doubles each IPOP restart
        lambda0 = 4 + int(3 * np.log(self.dim))

        # Track # FEs consumed in "large" vs "small" regime (BIPOP)
        fe_large = 0
        fe_small = 0
        restart_idx = 0
        # Initial sigma & x0
        x0 = rng.uniform(lo, hi, self.dim)
        sigma = self.sigma0

        while len(history_f) < max_evals:
            remaining = max_evals - len(history_f)

            # Choose popsize for this restart
            if restart_idx == 0:
                popsize = lambda0
                regime = "large"
            elif self.bipop:
                # BIPOP: alternate to balance large vs small budgets
                if fe_small < fe_large:
                    # Small regime: λ uniformly in [λ0, λ_large/2]
                    lambda_large = lambda0 * (self.incpopsize ** restart_idx)
                    upper = max(lambda0, lambda_large // 2)
                    popsize = int(rng.integers(lambda0, upper + 1))
                    sigma = self.sigma0 * 10 ** (-2 * float(rng.random()))
                    regime = "small"
                else:
                    popsize = lambda0 * (self.incpopsize ** restart_idx)
                    sigma = self.sigma0
                    regime = "large"
            else:
                # IPOP: monotonically increasing
                popsize = lambda0 * (self.incpopsize ** restart_idx)
                sigma = self.sigma0
                regime = "large"

            opts = cma.CMAOptions()
            opts["seed"] = int(self.seed) + 1000 * restart_idx
            opts["bounds"] = [[lo] * self.dim, [hi] * self.dim]
            opts["maxfevals"] = remaining
            opts["popsize"] = int(popsize)
            opts["verbose"] = -9

            # x0 for this restart: random uniform (standard IPOP/BIPOP convention)
            x0_this = rng.uniform(lo, hi, self.dim)
            es = cma.CMAEvolutionStrategy(x0_this, sigma, opts)

            fe_before = len(history_f)
            while len(history_f) < max_evals and not es.stop():
                solutions = es.ask()
                fitnesses = [float(self.func(np.asarray(s))) for s in solutions]
                es.tell(solutions, fitnesses)
                history_pop.append(np.array(solutions))
                for s, f in zip(solutions, fitnesses):
                    history_x.append(np.asarray(s))
                    history_f.append(f)
            fe_used = len(history_f) - fe_before
            if regime == "large":
                fe_large += fe_used
            else:
                fe_small += fe_used
            restart_idx += 1

        return self._make_result(history_x, history_f, history_pop)


class IPOPCMAESOptimizer(_RestartCMAESBase):
    """IPOP-CMA-ES (Auger & Hansen, 2005). λ doubles each restart."""
    bipop = False


class BIPOPCMAESOptimizer(_RestartCMAESBase):
    """BIPOP-CMA-ES (Hansen, 2009). Alternates between large and small λ regimes."""
    bipop = True
