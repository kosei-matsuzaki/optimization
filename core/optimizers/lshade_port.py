"""L-SHADE — faithful numpy port (Tanabe & Fukunaga, CEC 2014 winner).

Replaces the mealpy wrapper (``lshade.py``) as the comparison baseline. That
wrapper draws F / CR from NumPy's global RNG, which the runner never seeds, and
on BBOB 10D F10-EllipsoidalRot it stopped at an error of ~2e3 while faithful
jSO / L-SRTDE ports reach 1e-10. Everything here follows the paper:

    R. Tanabe and A. Fukunaga, "Improving the Search Performance of SHADE Using
    Linear Population Size Reduction", IEEE CEC 2014, pp. 1658-1665.

- current-to-pbest/1/bin, p = 0.11; archive |A| = round(2.6·NP) (r_arc = 2.6)
- success-history memory of size H = 6, initialised to 0.5
- CR_i ~ N(M_CR,r, 0.1) clipped to [0, 1]; CR = 0 when M_CR,r is the terminal
  value ⊥ (stored here as -1)
- F_i ~ Cauchy(M_F,r, 0.1); regenerated while F ≤ 0, truncated to 1 if F > 1
- M_F,k and M_CR,k are weighted Lehmer means of the successful values, weights
  ∝ |Δf|; M_CR,k becomes ⊥ once it is ⊥ or max(S_CR) = 0
- bound repair: a violating coordinate goes half-way between the bound and the
  parent's coordinate
- linear population size reduction from N_init = round(18·D) to N_min = 4 by
  the number of evaluations used; the worst individuals are removed and the
  archive is trimmed at random to its new size
- replacement when the trial is not worse (u ≤ x); only strict improvements
  enter S_F / S_CR and send the parent to the archive

Deviation: the budget is checked per evaluation, so the last generation is
evaluated only up to ``max_evals``.
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult

_TERMINAL = -1.0


class LSHADEPortOptimizer(BaseOptimizer):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_init_factor: float = 18.0,
        n_min: int = 4,
        memory_size: int = 6,
        p_best: float = 0.11,
        arc_rate: float = 2.6,
    ):
        super().__init__(benchmark, seed)
        self.n_init = max(n_min, int(round(n_init_factor * self.dim)))
        self.n_min = n_min
        self.memory_size = memory_size
        self.p_best = p_best
        self.arc_rate = arc_rate

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        d = self.dim
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        def evaluate(x: np.ndarray) -> float:
            f = float(self.func(x))
            history_x.append(x.copy())
            history_f.append(f)
            return f

        np_cur = min(self.n_init, max_evals)
        pop = rng.uniform(lo, hi, (np_cur, d))
        fit = np.array([evaluate(x) for x in pop])
        history_pop.append(pop.copy())

        H = self.memory_size
        m_f = np.full(H, 0.5)
        m_cr = np.full(H, 0.5)
        k = 0
        archive = np.empty((0, d))

        while len(history_f) < max_evals and np_cur >= 2:
            # Parameter generation
            r = rng.integers(0, H, np_cur)
            cr = np.where(m_cr[r] == _TERMINAL, 0.0,
                          np.clip(rng.normal(m_cr[r], 0.1), 0.0, 1.0))
            f_par = np.empty(np_cur)
            for i in range(np_cur):
                fi = m_f[r[i]] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                while fi <= 0.0:
                    fi = m_f[r[i]] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                f_par[i] = min(fi, 1.0)

            # current-to-pbest/1 with archive
            n_pbest = max(2, int(round(self.p_best * np_cur)))
            order = np.argsort(fit)
            union = np.vstack([pop, archive]) if len(archive) else pop
            trials = np.empty_like(pop)
            for i in range(np_cur):
                pb = pop[order[rng.integers(0, n_pbest)]]
                r1 = rng.integers(0, np_cur)
                while r1 == i:
                    r1 = rng.integers(0, np_cur)
                r2 = rng.integers(0, len(union))
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, len(union))
                v = pop[i] + f_par[i] * (pb - pop[i]) + f_par[i] * (pop[r1] - union[r2])
                v = np.where(v < lo, (lo + pop[i]) / 2.0, v)
                v = np.where(v > hi, (hi + pop[i]) / 2.0, v)
                mask = rng.random(d) < cr[i]
                mask[rng.integers(0, d)] = True
                trials[i] = np.where(mask, v, pop[i])

            # Selection
            s_f, s_cr, s_df, new_arc = [], [], [], []
            n_eval = min(np_cur, max_evals - len(history_f))
            for i in range(n_eval):
                fu = evaluate(trials[i])
                if fu <= fit[i]:
                    if fu < fit[i]:
                        s_f.append(f_par[i])
                        s_cr.append(cr[i])
                        s_df.append(fit[i] - fu)
                        new_arc.append(pop[i].copy())
                    pop[i] = trials[i]
                    fit[i] = fu
            history_pop.append(pop.copy())
            if n_eval < np_cur:
                break

            if new_arc:
                archive = np.vstack([archive, np.array(new_arc)]) if len(archive) else np.array(new_arc)

            # Memory update
            if s_f:
                w = np.array(s_df) / np.sum(s_df)
                sf = np.array(s_f)
                scr = np.array(s_cr)
                m_f[k] = float(np.sum(w * sf ** 2) / np.sum(w * sf))
                if m_cr[k] == _TERMINAL or scr.max() == 0.0:
                    m_cr[k] = _TERMINAL
                else:
                    m_cr[k] = float(np.sum(w * scr ** 2) / np.sum(w * scr))
                k = (k + 1) % H

            # Linear population size reduction
            nfe = len(history_f)
            np_next = int(round((self.n_min - self.n_init) / max_evals * nfe + self.n_init))
            np_next = max(self.n_min, np_next)
            if np_next < np_cur:
                keep = np.argsort(fit)[:np_next]
                pop, fit = pop[keep], fit[keep]
                np_cur = np_next
            arc_max = int(round(self.arc_rate * np_cur))
            if len(archive) > arc_max:
                archive = archive[rng.choice(len(archive), arc_max, replace=False)]

        return self._make_result(history_x, history_f, history_pop)
