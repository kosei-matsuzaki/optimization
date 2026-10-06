"""MOS — Multiple Offspring Sampling hybrid of DE and IPOP-CMA-ES (BBOB-2010).

Source: A. LaTorre, S. Muelas, J. M. Peña, "Benchmarking a MOS-based algorithm
on the BBOB-2010 noiseless function testbed", GECCO 2010 BBOB workshop
(https://oa.upm.es/7689/), Algorithm 1 and Eqs. (1)-(3), Table 2.

What the paper fixes (implemented as stated):
  * HRH (High-level Relay Hybrid): one shared population; in every step the
    techniques run *in sequence*, each evolving the output population of the
    previous one, each for its share of that step's evaluations.
  * The run is split into a fixed number of steps (85); step i has FEs_i =
    max_evals / 85, and technique j gets Pi_j * FEs_i of them (Alg. 1, l.7).
  * Participation starts uniform (Pi_j = 1/n) and is updated after each step by
    the Dynamic Participation Function (Eq. 2/3) with reduction factor xi = 0.05
    and a 5 % minimum participation ratio.
  * Quality function (Eq. 1): the Average Fitness Increment Sigma_j of the step's
    offspring is used only when it agrees with the number of fitness
    improvements Gamma_j; otherwise Gamma_j is used.
  * DE: classic model, F = 0.5, CR = 0.5, exponential crossover, tournament-2
    selection. IPOP-CMA-ES: the restart criteria of CMA-ES trigger a restart of
    the *whole* population, whose size doubles (initial 15, max 6400).

Guesses (the paper does not specify; kept simple and listed in the report):
  * Eq. (1) consensus is read globally: Sigma is used for all techniques when the
    pairwise orderings by Sigma and by Gamma agree for every pair, else Gamma is
    used for all (mixing scales per technique would make Eq. 3 meaningless).
  * "Fitness increment" of an offspring = max(0, f_ref - f_child); for DE f_ref
    is the target vector, for CMA-ES the k-th best offspring is compared with
    the k-th best population member (rank pairing).
  * DE is steady-state (trial replaces target immediately, one target at a
    time cycling through the population) so that a step can end mid-generation;
    "tournament 2" selects the base vector by binary tournament; bound repair
    = midpoint between parent and violated bound.
  * CMA-ES (pycma) keeps its state across steps, lambda = population size,
    sigma0 = 0.2 * (hi - lo), x0 uniform; before each CMA-ES turn the shared
    population's best is injected (pycma ``inject``) so DE progress reaches it;
    after each CMA-ES generation the population keeps the best N of
    population + offspring (elitist merge).
  * Technique order inside a step: DE then CMA-ES. Participation is not reset
    at a restart.
"""
from __future__ import annotations

import warnings

import numpy as np
import cma

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class _BudgetExhausted(Exception):
    pass


# pycma termination keys that are budget/target bookkeeping, not convergence
_NON_RESTART_KEYS = {"maxfevals", "maxiter", "ftarget"}


class MOSOptimizer(BaseOptimizer):
    """HRH MOS hybrid of DE and IPOP-CMA-ES (LaTorre et al., BBOB-2010)."""

    TECHNIQUES = ("DE", "CMA-ES")

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size: int = 15,          # Table 2: initial population size
        max_pop_size: int = 6400,    # Table 2: max pop size after restarts
        de_F: float = 0.5,           # Table 2
        de_CR: float = 0.5,          # Table 2 (exponential crossover)
        min_participation: float = 0.05,  # Table 2: minimum participation 5 %
        n_steps: int = 85,           # Table 2: number of steps
        xi: float = 0.05,            # Eq. 3 reduction factor ("usually 0.05")
        sigma0_frac: float = 0.2,    # guess: CMA-ES sigma0 as share of range
    ):
        super().__init__(benchmark, seed)
        self.pop_size = pop_size
        self.max_pop_size = max_pop_size
        self.de_F = de_F
        self.de_CR = de_CR
        self.min_participation = min_participation
        self.n_steps = n_steps
        self.xi = xi
        self.sigma0_frac = sigma0_frac
        # (eval_count, [Pi_DE, Pi_CMA]) after every step; for diagnostics
        self.participation_history: list[tuple[int, list[float]]] = []
        self.restart_evals: list[int] = []

    # ── evaluation bookkeeping ────────────────────────────────────────────
    def _eval(self, x: np.ndarray) -> float:
        if len(self._hf) >= self._max_evals:
            raise _BudgetExhausted()
        x = np.asarray(x, dtype=float).copy()
        f = float(self.func(x))
        self._hx.append(x)
        self._hf.append(f)
        return f

    # ── participation function (Eqs. 1-3) ─────────────────────────────────
    def _quality(self, sig: np.ndarray, gam: np.ndarray) -> np.ndarray:
        n = len(sig)
        consensus = all(
            np.sign(sig[j] - sig[k]) == np.sign(gam[j] - gam[k])
            for j in range(n) for k in range(j + 1, n))
        return sig.copy() if consensus else gam.astype(float)

    def _update_participation(self, pi: np.ndarray, q: np.ndarray) -> np.ndarray:
        q_best = float(np.max(q))
        if q_best <= 0.0:
            return pi  # no technique produced any improvement: keep ratios
        best = q >= q_best
        delta = np.where(best, 0.0, self.xi * (q_best - q) / q_best * pi)
        eta = delta.sum() / best.sum()
        new = np.where(best, pi + eta, pi - delta)
        # minimum participation ratio: lift the starved, take it from the best
        low = new < self.min_participation
        if low.any():
            deficit = (self.min_participation - new[low]).sum()
            new[low] = self.min_participation
            new[best] -= deficit / best.sum()
        return new / new.sum()

    # ── CMA-ES helpers ────────────────────────────────────────────────────
    def _new_cma(self, x0: np.ndarray, popsize: int, restart_idx: int):
        lo, hi = self.bounds
        opts = cma.CMAOptions()
        opts["seed"] = int(self.seed) * 7919 + 1000 * restart_idx + 1
        opts["bounds"] = [[lo] * self.dim, [hi] * self.dim]
        opts["popsize"] = int(popsize)
        opts["maxfevals"] = np.inf
        opts["verbose"] = -9
        return cma.CMAEvolutionStrategy(
            x0, self.sigma0_frac * (hi - lo), opts)

    def _cma_converged(self, es) -> bool:
        return bool(set(es.stop().keys()) - _NON_RESTART_KEYS)

    # ── main loop ─────────────────────────────────────────────────────────
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        D = self.dim
        self._max_evals = max_evals
        self._hx: list[np.ndarray] = []
        self._hf: list[float] = []
        history_pop: list[np.ndarray] = []
        self.participation_history = []
        self.restart_evals = []

        n_tech = len(self.TECHNIQUES)
        pi = np.full(n_tech, 1.0 / n_tech)
        fes_step = max_evals / self.n_steps

        N = self.pop_size
        restart_idx = 0
        state = {"de_ptr": 0}

        def fresh_population(n):
            P = rng.uniform(lo, hi, (n, D))
            fP = np.empty(n)
            for i in range(n):
                fP[i] = self._eval(P[i])
            return P, fP

        try:
            pop, fit = fresh_population(N)
        except _BudgetExhausted:
            return self._make_result(self._hx, self._hf, None)
        es = self._new_cma(rng.uniform(lo, hi, D), N, restart_idx)
        history_pop.append(pop.copy())

        # ── techniques: each returns (increments list, n_improvements) ─────
        def run_de(alloc: int):
            nonlocal pop, fit
            incs: list[float] = []
            n_imp = 0
            used = 0
            n = len(pop)
            while used < alloc:
                i = state["de_ptr"] % n
                state["de_ptr"] = (i + 1) % n
                others = np.array([k for k in range(n) if k != i])
                # base vector: binary tournament (Table 2 "Tournament 2")
                c1, c2 = rng.choice(others, 2, replace=False)
                r1 = c1 if fit[c1] <= fit[c2] else c2
                rest = others[others != r1]
                r2, r3 = rng.choice(rest, 2, replace=False)
                v = pop[r1] + self.de_F * (pop[r2] - pop[r3])
                # exponential crossover
                u = pop[i].copy()
                j = int(rng.integers(D))
                L = 0
                while True:
                    u[j] = v[j]
                    j = (j + 1) % D
                    L += 1
                    if L >= D or rng.random() >= self.de_CR:
                        break
                # bound repair: midpoint between parent and the violated bound
                below, above = u < lo, u > hi
                u[below] = (pop[i][below] + lo) / 2.0
                u[above] = (pop[i][above] + hi) / 2.0
                fu = self._eval(u)
                used += 1
                incs.append(max(0.0, fit[i] - fu))
                if fu < fit[i]:
                    n_imp += 1
                if fu <= fit[i]:
                    pop[i], fit[i] = u, fu
            return incs, n_imp

        def run_cma(alloc: int):
            nonlocal pop, fit, es, N, restart_idx
            incs: list[float] = []
            n_imp = 0
            used = 0
            # hand the shared population's best (possibly found by DE) to CMA-ES
            b = int(np.argmin(fit))
            if es.best.f is None or fit[b] < es.best.f:
                es.inject([es.gp.geno(pop[b].copy(),
                                      from_bounds=es.boundary_handler.inverse)],
                          force=True)
            while used < alloc:
                if self._cma_converged(es):
                    # IPOP restart of the *overall* population (paper, Sec. 2)
                    restart_idx += 1
                    N = min(2 * N, self.max_pop_size)
                    self.restart_evals.append(len(self._hf))
                    pop, fit = fresh_population(N)
                    used += N
                    state["de_ptr"] = 0
                    es = self._new_cma(rng.uniform(lo, hi, D), N, restart_idx)
                    continue
                X = es.ask()
                fX = [self._eval(x) for x in X]  # may raise at the budget cap
                used += len(X)
                es.tell(X, fX)
                X = np.asarray(X, dtype=float)
                fX = np.asarray(fX)
                # quality: rank-paired increment against the population
                off_order = np.argsort(fX)
                pop_sorted = np.sort(fit)
                m = min(len(fX), len(pop_sorted))
                for r in range(len(fX)):
                    ref = pop_sorted[min(r, m - 1)]
                    inc = max(0.0, ref - fX[off_order[r]])
                    incs.append(inc)
                    if fX[off_order[r]] < ref:
                        n_imp += 1
                # elitist merge: best N of population + offspring
                allX = np.vstack([pop, X])
                allf = np.concatenate([fit, fX])
                keep = np.argsort(allf, kind="stable")[:N]
                pop, fit = allX[keep].copy(), allf[keep].copy()
            return incs, n_imp

        runners = (run_de, run_cma)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # pycma sigma-overflow chatter
                while len(self._hf) < max_evals:
                    sig = np.zeros(n_tech)
                    gam = np.zeros(n_tech)
                    for j, run in enumerate(runners):
                        alloc = max(1, int(round(pi[j] * fes_step)))
                        incs, n_imp = run(alloc)
                        sig[j] = float(np.mean(incs)) if incs else 0.0
                        gam[j] = n_imp
                    history_pop.append(pop.copy())
                    pi = self._update_participation(pi, self._quality(sig, gam))
                    self.participation_history.append(
                        (len(self._hf), pi.tolist()))
        except _BudgetExhausted:
            pass

        history_pop.append(pop.copy())
        return self._make_result(self._hx, self._hf, history_pop)
