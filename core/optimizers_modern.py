"""Modern DE / CMA-ES baselines used as stronger comparison methods.

- LSHADEOptimizer  — wraps mealpy.L_SHADE (Tanabe & Fukunaga, CEC 2014)
- IPOPCMAESOptimizer  — increasing-population restart CMA-ES (Auger & Hansen 2005)
- BIPOPCMAESOptimizer — two-regime restart CMA-ES (Hansen 2009)

Each wraps an external library (mealpy / pycma) under the BaseOptimizer
interface so they slot directly into the existing runner without changes.
"""
from __future__ import annotations

import logging
import os
import numpy as np
import cma

from .benchmarks import BenchmarkFunction
from .optimizers import BaseOptimizer, OptimizeResult


def _eval_recording_wrapper(func, history_x, history_f, max_evals):
    """Wrap an objective to record (x, f) into history and to stop early.

    Returns a callable that raises `_BudgetExceeded` once max_evals is hit, so
    the optimizer's loop can be terminated cleanly.
    """
    class _BudgetExceeded(Exception):
        pass

    def wrapped(x):
        if len(history_f) >= max_evals:
            raise _BudgetExceeded()
        x = np.asarray(x, dtype=float)
        f = float(func(x))
        history_x.append(x.copy())
        history_f.append(f)
        return f

    wrapped._budget_exc = _BudgetExceeded
    return wrapped


# ---------------------------------------------------------------------------
# L-SHADE
# ---------------------------------------------------------------------------

class LSHADEOptimizer(BaseOptimizer):
    """L-SHADE — SHADE with Linear Population Size Reduction (CEC 2014 winner).

    Wraps mealpy.evolutionary_based.SHADE.L_SHADE. Uses the library's
    Termination(max_fe=...) so total function evaluations match max_evals.
    Per-eval history is captured by intercepting the objective function.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size: int | None = None,  # None → use paper recommendation 18 * d
        miu_f: float = 0.5,
        miu_cr: float = 0.5,
    ):
        super().__init__(benchmark, seed)
        # Tanabe & Fukunaga (CEC 2014) recommend N_init = 18 * d
        self.pop_size = 18 * benchmark.dim if pop_size is None else pop_size
        self.miu_f = miu_f
        self.miu_cr = miu_cr

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        from mealpy import FloatVar, Termination
        from mealpy.evolutionary_based.SHADE import L_SHADE

        lo, hi = self.bounds
        history_x: list[np.ndarray] = []
        history_f: list[float] = []

        wrapped = _eval_recording_wrapper(self.func, history_x, history_f, max_evals)

        # Generous epoch — termination will trigger on max_fe first
        epoch = int(np.ceil(max_evals / max(self.pop_size, 1))) + 20

        problem = {
            "bounds": FloatVar(lb=[float(lo)] * self.dim,
                               ub=[float(hi)] * self.dim, name="x"),
            "obj_func": wrapped,
            "minmax": "min",
            "log_to": None,
            "verbose": False,
        }
        term = Termination(max_fe=max_evals)
        model = L_SHADE(epoch=epoch, pop_size=self.pop_size,
                        miu_f=self.miu_f, miu_cr=self.miu_cr)

        # Suppress mealpy's chatty INFO logs at root
        logging.getLogger("mealpy").setLevel(logging.WARNING)
        # mealpy seeds its own RNG via the `seed` kwarg of solve()
        try:
            model.solve(problem, mode="single", termination=term, seed=self.seed)
        except wrapped._budget_exc:
            pass  # we hit max_fe inside the wrapped function

        return self._make_result(history_x, history_f, history_pop=None)


# ---------------------------------------------------------------------------
# IPOP / BIPOP CMA-ES
# ---------------------------------------------------------------------------

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
