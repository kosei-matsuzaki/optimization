"""NMMSO baseline — niching migratory multi-swarm optimiser (external library)."""
from __future__ import annotations
import numpy as np

from pynmmso import Nmmso

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class NMMSOOptimizer(BaseOptimizer):
    """NMMSO (Fieldsend, 2014), via the ``pynmmso`` package.

    A multi-swarm method that grows, splits and merges swarms on its own: each
    swarm tracks one mode, swarms that drift into the same basin are merged,
    and new swarms are spawned from the most promising unexplored regions.
    Number of niches is discovered, never configured — like r3pso, and unlike
    every radius-based method.

    Competition-grade reference: NMMSO placed at the top of the GECCO/CEC
    niching competitions of its era and is the strongest niching baseline here
    that runs from a published implementation rather than a reimplementation,
    which keeps "your baseline was coded badly" off the table.

    Two adaptations are needed to fit this project's harness:

    * **Direction** — pynmmso maximises; the benchmark is negated on the way in
      and the recorded history is flipped back.
    * **Budget** — ``Nmmso.run`` only checks its budget between iterations, so
      it can overshoot. Once ``max_evals`` real evaluations are spent the
      wrapper stops calling the benchmark and answers ``-inf`` instead: the
      evaluation count stays exact, the run finishes its iteration, and no
      fake point can ever become a reported mode.

    Reported solutions are the modes NMMSO itself returns (one per swarm),
    which is exactly the "reported solution set" the CEC2013 rules ask for.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        swarm_size: int = 10,
    ):
        super().__init__(benchmark, seed)
        self.swarm_size = swarm_size

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        lo, hi = self.bounds
        dim = self.dim
        history_x: list[np.ndarray] = []
        history_f: list[float] = []

        func = self.func

        class _Problem:
            @staticmethod
            def fitness(params) -> float:
                if len(history_f) >= max_evals:
                    return float("-inf")          # budget spent: do not evaluate
                x = np.clip(np.asarray(params, dtype=float), lo, hi)
                f = float(func(x))
                history_x.append(x.copy())
                history_f.append(f)
                return -f                          # pynmmso maximises

            @staticmethod
            def get_bounds():
                return [lo] * dim, [hi] * dim

        # pynmmso draws from numpy's global RNG, so seeding is global by design.
        np.random.seed(int(self.seed) % (2 ** 32))
        modes = Nmmso(_Problem(), swarm_size=self.swarm_size).run(max_evals)

        solutions = [np.clip(np.asarray(m.location, dtype=float), lo, hi)
                     for m in modes if np.isfinite(m.value)]
        if not history_f:                          # degenerate: nothing evaluated
            x0 = np.full(dim, (lo + hi) / 2.0)
            history_x, history_f = [x0], [float(func(x0))]
        return self._make_result(history_x, history_f,
                                 solutions=solutions or None)
