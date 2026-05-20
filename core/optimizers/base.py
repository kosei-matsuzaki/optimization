"""Shared optimizer scaffolding: the result container and the abstract base."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..benchmarks import BenchmarkFunction


@dataclass
class OptimizeResult:
    best_x: np.ndarray
    best_f: float
    history_x: list[np.ndarray]  # all evaluated points (for trajectory)
    history_best: list[float]    # best_f at each evaluation (for convergence)
    history_f: list[float]       # raw f value at each evaluation
    history_pop: list[np.ndarray]  # population snapshot per generation (n, dim)
    n_evals: int
    # per-individual sigma per generation (n,) array; empty = not recorded
    history_pop_sigma: list[np.ndarray] = field(default_factory=list)
    # MC-ESO-specific dynamics (one entry per generation; empty for non-MC-ESO)
    history_sigma_global: list[float] = field(default_factory=list)
    history_n_elite: list[int] = field(default_factory=list)
    history_no_improve: list[int] = field(default_factory=list)
    history_eval_count: list[int] = field(default_factory=list)
    # sigma actually used to generate each offspring (one per eval after init pop;
    # nan = random reactivation with no parent sigma)
    history_sigma_eval: list[float] = field(default_factory=list)


class BaseOptimizer(ABC):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
    ):
        self.benchmark = benchmark
        self.func = benchmark.func
        self.bounds = benchmark.bounds
        self.dim = benchmark.dim
        self.seed = seed

    @abstractmethod
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        ...

    @staticmethod
    def _init_population_history(
        pop: np.ndarray, fit: np.ndarray
    ) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
        """Seed the running history with an already-evaluated initial population.

        Returns the ``(history_x, history_f, history_pop)`` buffers that the
        generation loop appends to — the initial population counts as the first
        ``len(pop)`` evaluations. Shared by the population-based baselines
        (PSO / DE / SaVOA); MC-ESO seeds its own history in ``_init_state``.
        """
        return list(pop), list(fit), [pop.copy()]

    def _make_result(
        self,
        history_x: list[np.ndarray],
        history_f: list[float],
        history_pop: Optional[list[np.ndarray]] = None,
    ) -> OptimizeResult:
        best_idx = int(np.argmin(history_f))
        history_best: list[float] = []
        current_best = float("inf")
        for f in history_f:
            if f < current_best:
                current_best = f
            history_best.append(current_best)
        return OptimizeResult(
            best_x=history_x[best_idx],
            best_f=history_f[best_idx],
            history_x=history_x,
            history_best=history_best,
            history_f=history_f,
            history_pop=history_pop or [],
            n_evals=len(history_f),
        )
