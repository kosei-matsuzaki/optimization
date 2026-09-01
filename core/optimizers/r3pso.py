"""Ring-topology lbest PSO (r3pso) — niching without niching parameters."""
from __future__ import annotations
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class RingPSOOptimizer(BaseOptimizer):
    """r3pso / r2pso (Li, 2010, IEEE TEC 14(1):150-169).

    The only change from the PSO baseline is *who each particle follows*: the
    global best is replaced by the best personal best inside a ring
    neighbourhood over population indices (r3pso: left, self, right; r2pso:
    self, right). Indices are fixed for the whole run and unrelated to
    position, so neighbouring particles start in unrelated regions and the ring
    slowly resolves into overlapping sub-swarms, one per basin.

    The point of the method is that this needs **no niche radius, no species
    count, no sharing parameter** — the topology alone maintains the niches.
    That is why it appears in nearly every multimodal comparison, and why it is
    the right control for MC-ESO's strain coexistence, which does depend on a
    radius (``niche_radius_ratio``).

    Inertia and acceleration constants match ``PSOOptimizer`` exactly, so
    PSO vs r3pso isolates the topology and nothing else.

    Reported solutions are the particles' personal bests: in this algorithm the
    pbest network *is* the niche memory, not the current positions.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_particles: int = 30,   # matches PSOOptimizer; Li used larger swarms
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        ring: int = 3,           # 3 = r3pso (left/self/right), 2 = r2pso (self/right)
    ):
        super().__init__(benchmark, seed)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.ring = ring

    def _neighbourhood_best(self, pbest_pos: np.ndarray,
                            pbest_fit: np.ndarray) -> np.ndarray:
        """Best pbest within each particle's ring neighbourhood (n, dim)."""
        n = self.n_particles
        idx = np.arange(n)
        members = [idx, (idx + 1) % n] if self.ring == 2 else \
                  [(idx - 1) % n, idx, (idx + 1) % n]
        stacked = np.stack([pbest_fit[m] for m in members])       # (ring, n)
        winner = np.argmin(stacked, axis=0)                        # (n,)
        pick = np.stack(members)[winner, idx]                      # (n,)
        return pbest_pos[pick]

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        v_max = 0.2 * (hi - lo)

        pos = rng.uniform(lo, hi, (self.n_particles, self.dim))
        vel = rng.uniform(-v_max, v_max, (self.n_particles, self.dim))
        fit = np.array([self.func(x) for x in pos])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()

        history_x, history_f, history_pop = self._init_population_history(pos, fit)

        while len(history_f) < max_evals:
            nbest_pos = self._neighbourhood_best(pbest_pos, pbest_fit)
            r1 = rng.random((self.n_particles, self.dim))
            r2 = rng.random((self.n_particles, self.dim))
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (nbest_pos - pos))
            vel = np.clip(vel, -v_max, v_max)
            pos = np.clip(pos + vel, lo, hi)

            for i, x in enumerate(pos):
                if len(history_f) >= max_evals:
                    break
                f = self.func(x)
                history_x.append(x.copy())
                history_f.append(f)
                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_pos[i] = x.copy()

            history_pop.append(pos.copy())

        return self._make_result(history_x, history_f, history_pop,
                                 solutions=[p.copy() for p in pbest_pos])
