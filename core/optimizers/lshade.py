"""L-SHADE — SHADE with Linear Population Size Reduction (CEC 2014 winner).

Wraps mealpy.evolutionary_based.SHADE.L_SHADE under the BaseOptimizer
interface so it slots directly into the existing runner without changes.
"""
from __future__ import annotations

import logging
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


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
