"""DEPSO — particle swarm with a differential-evolution operator (Zhang & Xie 2003).

    W.-J. Zhang, X.-F. Xie, "DEPSO: hybrid particle swarm with differential
    evolution operator", IEEE Int. Conf. on Systems, Man and Cybernetics
    (SMC 2003), vol. 4, pp. 3816–3821.
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class DEPSOOptimizer(BaseOptimizer):
    """DEPSO (Zhang & Xie, SMC 2003).

    PSO and a DE operator alternate generation by generation (PSO at odd
    generations, DE at even ones). Each generation costs one evaluation per
    particle.

    * PSO step (inertia-weight PSO): v ← w·v + c1·r1·(p_i − x) + c2·r2·(g − x),
      |v| ≤ v_max, x ← x + v (clipped to the box); pbest / gbest updated.
      Paper: w = 0.4, c1 = c2 = 2.
    * DE step, applied to the personal bests (not to the positions, so the
      swarm dynamics are not disrupted): for particle i a trial T = p_i is
      built with, for each dimension d,
      ``if rand() < CR or d == k: T_d = g_d + δ2_d``,
      δ2 = (Δ1 + Δ2)/2 with each Δ = p_A − p_B a difference of two randomly
      chosen pbests (the "bell-shaped", N = 2 variant the paper selects).
      T replaces p_i if it is better. Paper: CR = 0.1.

    Guesses (the paper could not be retrieved in full; the operator form and
    w / c1 / c2 / CR above come from its abstract and secondary descriptions):
    swarm size ``n_particles`` = 30 (project PSO default), v_max = 0.5·range,
    the four pbests used for δ2 are distinct and ≠ i, T is clipped to the box,
    replacement is strict (<).
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_particles: int = 30,     # guess (project PSO default)
        w: float = 0.4,            # paper
        c1: float = 2.0,           # paper
        c2: float = 2.0,           # paper
        CR: float = 0.1,           # paper
        vmax_frac: float = 0.5,    # guess: v_max = vmax_frac · (hi − lo)
    ):
        super().__init__(benchmark, seed)
        self.n_particles = max(int(n_particles), 5)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.CR = CR
        self.vmax_frac = vmax_frac

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = map(float, self.bounds)
        D = self.dim
        N = self.n_particles
        v_max = self.vmax_frac * (hi - lo)

        pos = rng.uniform(lo, hi, (N, D))
        vel = rng.uniform(-v_max, v_max, (N, D))
        pos = pos[:max_evals]
        fit = np.array([float(self.func(x)) for x in pos])
        history_x, history_f, history_pop = self._init_population_history(pos, fit)
        history_x = [x.copy() for x in history_x]
        history_f = [float(f) for f in history_f]
        if len(pos) < N:
            return self._make_result(history_x, history_f, history_pop)

        pbest = pos.copy()
        pbest_f = fit.copy()
        g = int(np.argmin(pbest_f))

        gen = 1
        while len(history_f) < max_evals:
            if gen % 2 == 1:  # PSO generation
                r1 = rng.random((N, D))
                r2 = rng.random((N, D))
                vel = (self.w * vel + self.c1 * r1 * (pbest - pos)
                       + self.c2 * r2 * (pbest[g] - pos))
                vel = np.clip(vel, -v_max, v_max)
                pos = np.clip(pos + vel, lo, hi)
                for i in range(N):
                    if len(history_f) >= max_evals:
                        break
                    f = float(self.func(pos[i]))
                    history_x.append(pos[i].copy())
                    history_f.append(f)
                    if f < pbest_f[i]:
                        pbest[i] = pos[i].copy()
                        pbest_f[i] = f
                        if f < pbest_f[g]:
                            g = i
                history_pop.append(pos.copy())
            else:  # DE generation on the personal bests
                for i in range(N):
                    if len(history_f) >= max_evals:
                        break
                    others = np.delete(np.arange(N), i)
                    a, b, c, d = rng.choice(others, size=4, replace=False)
                    delta2 = 0.5 * ((pbest[a] - pbest[b]) + (pbest[c] - pbest[d]))
                    mask = rng.random(D) < self.CR
                    mask[int(rng.integers(D))] = True
                    T = np.where(mask, pbest[g] + delta2, pbest[i])
                    T = np.clip(T, lo, hi)
                    f = float(self.func(T))
                    history_x.append(T.copy())
                    history_f.append(f)
                    if f < pbest_f[i]:
                        pbest[i] = T
                        pbest_f[i] = f
                        if f < pbest_f[g]:
                            g = i
                history_pop.append(pbest.copy())
            gen += 1

        return self._make_result(history_x, history_f, history_pop)
