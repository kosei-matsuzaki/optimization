"""Multistart local search baseline (restarted Nelder-Mead).

Low-dimensional BBOB sanity baseline: COCO archive data shows that in 2-3D
a plain multistart local search is highly competitive, so any proposed
metaheuristic must demonstrably beat it to justify its machinery.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _BudgetExhausted(Exception):
    """Raised inside the objective wrapper to hard-stop scipy at max_evals."""


class MultistartNelderMeadOptimizer(BaseOptimizer):
    """Restarted Nelder-Mead simplex (Nelder & Mead, 1965).

    Repeatedly runs scipy's bounded Nelder-Mead from uniform-random start
    points until the evaluation budget is spent. Each restart runs to tight
    convergence (xatol/fatol well below the 1e-10 success threshold) so a
    start that lands in the global basin can polish to full depth; the next
    restart then samples a fresh random point. No information is carried
    between restarts — this is the deliberately-dumb "multistart local
    search" reference that dominates low-dimensional BBOB.
    """

    # Tighter than the 1e-10 success threshold so polishing doesn't stop short.
    _XATOL = 1e-12
    _FATOL = 1e-14

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds

        history_x: list[np.ndarray] = []
        history_f: list[float] = []

        def wrapped(x: np.ndarray) -> float:
            if len(history_f) >= max_evals:
                raise _BudgetExhausted
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = float(self.func(x))
            history_x.append(x.copy())
            history_f.append(f)
            return f

        while len(history_f) < max_evals:
            x0 = rng.uniform(lo, hi, self.dim)
            try:
                minimize(
                    wrapped, x0, method="Nelder-Mead",
                    bounds=[(lo, hi)] * self.dim,
                    options={
                        "maxfev": max_evals - len(history_f),
                        "xatol": self._XATOL,
                        "fatol": self._FATOL,
                    },
                )
            except _BudgetExhausted:
                break

        return self._make_result(history_x, history_f)
