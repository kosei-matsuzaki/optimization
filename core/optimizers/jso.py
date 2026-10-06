"""jSO — iL-SHADE successor, 2nd place at the IEEE CEC 2017 competition.

Pure-numpy implementation of

    J. Brest, M. S. Maučec, B. Bošković, "Single objective real-parameter
    optimization: algorithm jSO", IEEE CEC 2017, pp. 1311–1318.

built on the L-SHADE reference code structure (Tanabe & Fukunaga 2014), which
jSO extends. Written so the project does not depend on third-party DE packages
(pyade's ``jso`` deviates from the paper: memory size = population size,
M_F initialised to 0.5, phase switches by generation count).
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class JSOPortOptimizer(BaseOptimizer):
    """jSO (Brest, Maučec & Bošković, CEC 2017), pure-numpy port.

    Parameters and rules (paper, Sec. III / Algorithm 1):

    * NP_init = round(25·√D·ln D), NP_min = 4, linear population size
      reduction NP_{g+1} = round((NP_min − NP_init)/max_nfes · nfes + NP_init),
      worst individuals removed.
    * Historical memory H = 5, M_F = 0.3, M_CR = 0.8 initially; the last slot is
      fixed at M_F = M_CR = 0.9 (never overwritten).
    * External archive, |A| = round(1.0 · NP) (shrinks with NP; random deletion).
    * p for current-to-pBest grows linearly (iL-SHADE rule kept in jSO):
      p = (p_max − p_min)/max_nfes · nfes + p_min, p_max = 0.25,
      p_min = p_max / 2; the pbest pool has max(2, round(p·NP)) members.
    * CR_i ~ N(M_CR, 0.1) clipped to [0, 1]; CR_i = 0 if M_CR is terminal (⊥);
      CR_i ≥ 0.7 while nfes < 0.25·max_nfes, CR_i ≥ 0.6 while nfes < 0.5·max_nfes.
    * F_i ~ Cauchy(M_F, 0.1), redrawn while ≤ 0, truncated to 1;
      F_i ≤ 0.7 while nfes < 0.6·max_nfes.
    * Mutation DE/current-to-pBest-w/1:
      v = x_i + F_w (x_pbest − x_i) + F (x_r1 − x_r2), r1 ∈ P, r2 ∈ P ∪ A,
      F_w = 0.7F (nfes < 0.2·max), 0.8F (< 0.4·max), 1.2F otherwise.
    * Binomial crossover; bound violation repaired as (bound + x_i)/2
      (L-SHADE code convention).
    * Selection u ≤ x replaces; strict improvement records (F, CR, |Δf|) and
      archives the parent.
    * Memory update (k cycles over the first H−1 slots):
      M_F,k = (mean_WL(S_F) + M_F,k)/2, M_CR,k = (mean_WL(S_CR) + M_CR,k)/2 with
      the |Δf|-weighted Lehmer mean; M_CR,k = ⊥ if it was ⊥ or max(S_CR) = 0.

    Implementation guesses: memory index cycles over H−1 slots (equivalent to
    the reference code, which overrides the last slot on read); trials are
    generated for the whole population, then evaluated (synchronous); the
    budget is checked per evaluation.
    """

    TERMINAL = -1.0

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size: int | None = None,   # None → round(25·√D·ln D)  (paper)
        np_min: int = 4,               # paper
        memory_size: int = 5,          # paper: H = 5
        m_f_init: float = 0.3,         # paper (iL-SHADE used 0.5)
        m_cr_init: float = 0.8,        # paper
        arc_rate: float = 1.0,         # paper
        p_max: float = 0.25,           # paper
        p_min: float | None = None,    # None → p_max / 2  (paper)
    ):
        super().__init__(benchmark, seed)
        d = benchmark.dim
        default = int(round(25.0 * np.sqrt(d) * np.log(d))) if d > 1 else 25
        self.pop_size = max(default if pop_size is None else int(pop_size), np_min)
        self.np_min = np_min
        self.memory_size = memory_size
        self.m_f_init = m_f_init
        self.m_cr_init = m_cr_init
        self.arc_rate = arc_rate
        self.p_max = p_max
        self.p_min = p_max / 2 if p_min is None else p_min

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = map(float, self.bounds)
        D = self.dim
        H = self.memory_size
        N0 = self.pop_size

        history_x: list[np.ndarray] = []
        history_f: list[float] = []

        pop = rng.uniform(lo, hi, (N0, D))
        n0 = min(N0, max_evals)
        pop = pop[:n0]
        fit = np.array([float(self.func(x)) for x in pop])
        history_x, history_f, history_pop = self._init_population_history(pop, fit)
        history_x = [x.copy() for x in history_x]
        history_f = [float(f) for f in history_f]

        m_f = np.full(H, self.m_f_init)
        m_cr = np.full(H, self.m_cr_init)
        m_f[H - 1] = 0.9
        m_cr[H - 1] = 0.9
        k = 0
        archive = np.empty((0, D))
        np_cur = len(pop)

        while len(history_f) < max_evals and np_cur >= 2:
            nfes = len(history_f)
            frac = nfes / max_evals

            # --- parameter generation ---
            r = rng.integers(H, size=np_cur)
            mu_cr = m_cr[r]
            cr = np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0)
            cr[mu_cr == self.TERMINAL] = 0.0
            if frac < 0.25:
                cr = np.maximum(cr, 0.7)
            elif frac < 0.5:
                cr = np.maximum(cr, 0.6)

            mu_f = m_f[r]
            F = mu_f + 0.1 * np.tan(np.pi * (rng.random(np_cur) - 0.5))
            bad = F <= 0
            while bad.any():
                F[bad] = mu_f[bad] + 0.1 * np.tan(np.pi * (rng.random(int(bad.sum())) - 0.5))
                bad = F <= 0
            F = np.minimum(F, 1.0)
            if frac < 0.6:
                F = np.minimum(F, 0.7)
            if frac < 0.2:
                Fw = 0.7 * F
            elif frac < 0.4:
                Fw = 0.8 * F
            else:
                Fw = 1.2 * F

            p = (self.p_max - self.p_min) * frac + self.p_min
            n_pbest = max(2, int(round(p * np_cur)))
            order = np.argsort(fit, kind="stable")

            # --- mutation + crossover ---
            union = np.vstack([pop, archive]) if len(archive) else pop
            n_union = len(union)
            trials = np.empty_like(pop)
            for i in range(np_cur):
                pb = int(order[rng.integers(n_pbest)])
                r1 = int(rng.integers(np_cur))
                while r1 == i:
                    r1 = int(rng.integers(np_cur))
                r2 = int(rng.integers(n_union))
                while r2 == i or r2 == r1:
                    r2 = int(rng.integers(n_union))
                x = pop[i]
                v = x + Fw[i] * (pop[pb] - x) + F[i] * (pop[r1] - union[r2])
                v = np.where(v < lo, (lo + x) / 2.0, v)
                v = np.where(v > hi, (hi + x) / 2.0, v)
                mask = rng.random(D) < cr[i]
                mask[int(rng.integers(D))] = True
                trials[i] = np.where(mask, v, x)

            # --- evaluation + selection ---
            s_f, s_cr, s_df = [], [], []
            new_pop = pop.copy()
            new_fit = fit.copy()
            to_archive = []
            for i in range(np_cur):
                if len(history_f) >= max_evals:
                    break
                u = trials[i]
                fu = float(self.func(u))
                history_x.append(u.copy())
                history_f.append(fu)
                if fu < fit[i]:
                    s_f.append(F[i]); s_cr.append(cr[i]); s_df.append(fit[i] - fu)
                    to_archive.append(pop[i].copy())
                if fu <= fit[i]:
                    new_pop[i] = u
                    new_fit[i] = fu
            pop, fit = new_pop, new_fit

            # --- archive ---
            if to_archive:
                archive = np.vstack([archive, np.array(to_archive)])

            # --- memory update ---
            if s_f:
                w = np.asarray(s_df)
                w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
                sf = np.asarray(s_f)
                sc = np.asarray(s_cr)
                mean_f = float(np.sum(w * sf * sf) / np.sum(w * sf))
                m_f[k] = 0.5 * (mean_f + m_f[k])
                if m_cr[k] == self.TERMINAL or sc.max() == 0:
                    m_cr[k] = self.TERMINAL
                else:
                    mean_cr = float(np.sum(w * sc * sc) / np.sum(w * sc))
                    m_cr[k] = 0.5 * (mean_cr + m_cr[k])
                k = (k + 1) % max(H - 1, 1)

            # --- linear population size reduction ---
            plan = int(round((self.np_min - N0) / max_evals * len(history_f) + N0))
            plan = max(plan, self.np_min)
            if plan < np_cur:
                keep = np.argsort(fit, kind="stable")[:plan]
                pop, fit = pop[keep], fit[keep]
                np_cur = plan
            arc_max = int(round(self.arc_rate * np_cur))
            if len(archive) > arc_max:
                keep = rng.choice(len(archive), arc_max, replace=False)
                archive = archive[keep]

            history_pop.append(pop.copy())

        return self._make_result(history_x, history_f, history_pop)
