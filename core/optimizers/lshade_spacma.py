"""LSHADE-SPACMA — L-SHADE with semi-parameter adaptation hybridised with CMA-ES.

Mohamed, Hadi, Fattouh & Jambi, "LSHADE with semi-parameter adaptation hybrid
with CMA-ES for solving CEC 2017 benchmark problems", IEEE CEC 2017 (3rd place).

Implemented from the paper and the authors' MATLAB code. The original file
(``LSHADE_SPACMA.m``) is not hosted publicly any more; the reference used here
is the copy embedded in Fu et al.'s mLSHADE-SPACMA repository
(github.com/ShengweiFu/mLSHADE-SPACMA-final-code, ``mLSHADE_SPACMA.m``), whose
modifications are left there as commented-out original lines. Those
modifications (rank-based r1, worst-1% perturbation, best-first archive,
F = 0.5 + 0.1·rand) are *not* adopted; this module follows the original lines.
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class LSHADESPACMAOptimizer(BaseOptimizer):
    """LSHADE-SPACMA (Mohamed et al., CEC 2017).

    One population, one generation loop. Each individual i is assigned to
    class 1 (L-SHADE, current-to-pbest/1 with archive) with probability
    FCP_i drawn from a memory of size H, otherwise to class 2 (a CMA-ES sample
    ``m + sigma * B D N(0, I)``). Both kinds of mutant then undergo binomial
    crossover with the parent (rate CR_i) and one-to-one greedy selection.
    The FCP memory slot is updated as
    ``FCP <- c*FCP + (1-c) * dif1 / (dif1 + dif2)`` clipped to [0.2, 0.8],
    where dif_k is the summed fitness improvement produced by class k.
    The CMA-ES mean / paths / covariance are updated every generation from
    the best mu survivors of the (shared) population.

    Semi-parameter adaptation (SPA): during the first half of the budget F is
    drawn uniformly in [0.45, 0.55] while CR is adapted (SHADE memory);
    in the second half F is drawn from Cauchy(M_F, 0.1) as in L-SHADE.
    Linear population size reduction from 18*D to 4.

    Parameters (default — source):
      pop_size_factor = 18     N_init = 18·D        — paper / MATLAB code
      min_pop_size    = 4                           — paper / MATLAB code
      memory_size     = 5      H                    — MATLAB code
      p_best_rate     = 0.11                        — MATLAB code
      arc_rate        = 1.4    |A| = round(1.4·N)   — MATLAB code
      fcp_init        = 0.5    initial FCP memory   — paper / MATLAB code
      fcp_lr          = 0.8    c (L_Rate)           — paper / MATLAB code
      fcp_min/fcp_max = 0.2/0.8 FCP clipping        — MATLAB code
      f_spa_lo/width  = 0.45/0.1 first-half F       — paper / MATLAB code
      sigma0          = 0.5    initial CMA step     — MATLAB code
    The CMA-ES strategy constants (cc, cs, c1, cmu, damps) are Hansen's
    purecmaes defaults computed once from the initial mu; only the weights and
    mueff are recomputed when the population shrinks (as in the MATLAB code).
    The initial CMA mean is ``rand(D, 1)`` in [0, 1]^D as in the MATLAB code
    (it is overwritten by the population's weighted mean after generation 1).
    Bound handling: midpoint between parent and violated bound (JADE rule).
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_size_factor: int = 18,
        min_pop_size: int = 4,
        memory_size: int = 5,
        p_best_rate: float = 0.11,
        arc_rate: float = 1.4,
        fcp_init: float = 0.5,
        fcp_lr: float = 0.8,
        fcp_min: float = 0.2,
        fcp_max: float = 0.8,
        f_spa_lo: float = 0.45,
        f_spa_width: float = 0.1,
        sigma0: float = 0.5,
    ):
        super().__init__(benchmark, seed)
        self.pop_size_factor = pop_size_factor
        self.min_pop_size = min_pop_size
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
        self.arc_rate = arc_rate
        self.fcp_init = fcp_init
        self.fcp_lr = fcp_lr
        self.fcp_min = fcp_min
        self.fcp_max = fcp_max
        self.f_spa_lo = f_spa_lo
        self.f_spa_width = f_spa_width
        self.sigma0 = sigma0
        # Diagnostics: one row per generation
        # (nfes, mean FCP memory, share of class-1 (L-SHADE) individuals, hybrid flag)
        self.fcp_log: list[tuple[int, float, float, int]] = []

    @staticmethod
    def _cma_weights(pop_size: int):
        mu_f = pop_size / 2.0
        mu = int(np.floor(mu_f))
        w = np.log(mu_f + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = w.sum() ** 2 / np.sum(w ** 2)
        return mu, w, mueff

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        lo, hi = float(lo), float(hi)
        D = self.dim
        self.fcp_log = []

        pop_size = int(self.pop_size_factor * D)
        max_pop_size = pop_size
        min_pop_size = self.min_pop_size
        H = self.memory_size

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        def evaluate(X: np.ndarray) -> np.ndarray:
            """Evaluate rows of X in order, stopping at the budget."""
            out = []
            for x in X:
                if len(history_f) >= max_evals:
                    break
                f = float(self.func(x))
                history_x.append(x.copy())
                history_f.append(f)
                out.append(f)
            return np.asarray(out, dtype=float)

        pop = lo + rng.random((pop_size, D)) * (hi - lo)
        fitness = evaluate(pop)
        if len(fitness) < pop_size:  # budget smaller than the initial population
            return self._make_result(history_x, history_f, [pop[:len(fitness)].copy()])
        history_pop.append(pop.copy())

        memory_sf = np.full(H, 0.5)
        memory_cr = np.full(H, 0.5)
        memory_fcp = np.full(H, self.fcp_init)
        memory_pos = 0

        archive = np.zeros((0, D))
        archive_np = int(round(self.arc_rate * pop_size))

        # --- CMA-ES state (Hansen's purecmaes, as in the MATLAB code) ---
        sigma = self.sigma0
        xmean = rng.random(D)
        mu, weights, mueff = self._cma_weights(pop_size)
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        pc = np.zeros(D)
        ps = np.zeros(D)
        B = np.eye(D)
        Dd = np.ones(D)
        C = np.eye(D)
        invsqrtC = np.eye(D)
        eigeneval = 0
        chiN = D ** 0.5 * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        hybrid = True

        # Degenerate CMA states (sigma -> 0 / inf) are caught by the
        # finiteness checks below, as the MATLAB code does; silence numpy.
        with np.errstate(all="ignore"):
            while len(history_f) < max_evals:
                nfes = len(history_f)
                sorted_index = np.argsort(fitness, kind="stable")

                mem_idx = rng.integers(0, H, pop_size)
                mu_sf = memory_sf[mem_idx]
                mu_cr = memory_cr[mem_idx]
                mem_ratio = rng.random(pop_size)

                # CR ~ N(M_CR, 0.1), terminal value -1 -> 0
                cr = rng.normal(mu_cr, 0.1)
                cr[mu_cr == -1] = 0.0
                cr = np.clip(cr, 0.0, 1.0)

                # F: semi-parameter adaptation
                if nfes <= 0.5 * max_evals:
                    sf = self.f_spa_lo + self.f_spa_width * rng.random(pop_size)
                else:
                    sf = mu_sf + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
                    bad = sf <= 0
                    while bad.any():
                        sf[bad] = mu_sf[bad] + 0.1 * np.tan(
                            np.pi * (rng.random(int(bad.sum())) - 0.5))
                        bad = sf <= 0
                sf = np.minimum(sf, 1.0)

                # Class assignment: True = class 1 (L-SHADE), False = class 2 (CMA-ES)
                cls1 = memory_fcp[mem_idx] >= mem_ratio
                if not hybrid:
                    cls1[:] = True

                # current-to-pbest/1 with archive
                pop_all = np.vstack([pop, archive]) if len(archive) else pop
                n_all = len(pop_all)
                r0 = np.arange(pop_size)
                r1 = rng.integers(0, pop_size, pop_size)
                while (pos := r1 == r0).any():
                    r1[pos] = rng.integers(0, pop_size, int(pos.sum()))
                r2 = rng.integers(0, n_all, pop_size)
                while (pos := (r2 == r1) | (r2 == r0)).any():
                    r2[pos] = rng.integers(0, n_all, int(pos.sum()))
                pNP = max(int(round(self.p_best_rate * pop_size)), 2)
                pbest = pop[sorted_index[rng.integers(0, pNP, pop_size)]]

                vi = np.empty((pop_size, D))
                if cls1.any():
                    i1 = cls1
                    vi[i1] = pop[i1] + sf[i1, None] * (
                        pbest[i1] - pop[i1] + pop[r1[i1]] - pop_all[r2[i1]])
                n2 = int((~cls1).sum())
                if n2:
                    z = rng.standard_normal((n2, D))
                    vi[~cls1] = xmean + sigma * (z * Dd) @ B.T

                if not np.all(np.isfinite(vi)):
                    # MATLAB: complex mutant -> switch the hybridisation off, redo gen
                    hybrid = False
                    continue

                # Bound handling: midpoint between parent and the violated bound
                low = vi < lo
                vi[low] = (pop[low] + lo) / 2
                high = vi > hi
                vi[high] = (pop[high] + hi) / 2

                # Binomial crossover with the parent (applies to both classes)
                mask = rng.random((pop_size, D)) > cr[:, None]
                mask[r0, rng.integers(0, D, pop_size)] = False
                ui = np.where(mask, pop, vi)

                child_fit = evaluate(ui)
                n_done = len(child_fit)
                self.fcp_log.append((nfes, float(memory_fcp.mean()),
                                     float(cls1.mean()), int(hybrid)))
                if n_done < pop_size:
                    # Budget exhausted mid-generation: apply selection to the
                    # evaluated part only, then stop.
                    better = child_fit < fitness[:n_done]
                    idx = np.nonzero(better)[0]
                    pop[idx] = ui[idx]
                    fitness[idx] = child_fit[idx]
                    history_pop.append(pop.copy())
                    break

                dif = np.abs(fitness - child_fit)
                better = fitness > child_fit
                good_cr = cr[better]
                good_f = sf[better]
                dif_val = dif[better]
                dif1 = dif[better & cls1].sum()
                dif2 = dif[better & ~cls1].sum()

                # Archive: add defeated parents, drop duplicates, random trim
                if archive_np > 0 and better.any():
                    archive = np.vstack([archive, pop[better]])
                    archive = np.unique(archive, axis=0)
                    if len(archive) > archive_np:
                        archive = archive[rng.permutation(len(archive))[:archive_np]]

                pop = pop.copy()
                pop[better] = ui[better]
                fitness = np.where(better, child_fit, fitness)

                if len(good_cr) > 0:
                    w = dif_val / dif_val.sum() if dif_val.sum() > 0 else \
                        np.full(len(dif_val), 1.0 / len(dif_val))
                    denom_f = np.dot(w, good_f)
                    if denom_f > 0:
                        memory_sf[memory_pos] = np.dot(w, good_f ** 2) / denom_f
                    if good_cr.max() == 0 or memory_cr[memory_pos] == -1:
                        memory_cr[memory_pos] = -1
                    else:
                        memory_cr[memory_pos] = np.dot(w, good_cr ** 2) / np.dot(w, good_cr)
                    if hybrid and dif1 + dif2 > 0:
                        v = (memory_fcp[memory_pos] * self.fcp_lr
                             + (1 - self.fcp_lr) * dif1 / (dif1 + dif2))
                        memory_fcp[memory_pos] = min(max(v, self.fcp_min), self.fcp_max)
                    memory_pos = (memory_pos + 1) % H

                # Linear population size reduction
                nfes = len(history_f)
                plan = int(round((min_pop_size - max_pop_size) / max_evals * nfes
                                 + max_pop_size))
                if pop_size > plan:
                    n_red = pop_size - plan
                    if pop_size - n_red < min_pop_size:
                        n_red = pop_size - min_pop_size
                    if n_red > 0:
                        keep = np.sort(np.argsort(fitness, kind="stable")[:pop_size - n_red])
                        pop = pop[keep]
                        fitness = fitness[keep]
                        pop_size = len(pop)
                        archive_np = int(round(self.arc_rate * pop_size))
                        if len(archive) > archive_np:
                            archive = archive[rng.permutation(len(archive))[:archive_np]]
                        mu, weights, mueff = self._cma_weights(pop_size)

                history_pop.append(pop.copy())

                # CMA-ES adaptation from the shared population
                if hybrid:
                    order = np.argsort(fitness, kind="stable")[:mu]
                    xold = xmean
                    xmean = weights @ pop[order]
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
                        invsqrtC @ (xmean - xold)) / sigma
                    hsig = (np.sum(ps ** 2) / (1 - (1 - cs) ** (2 * nfes / pop_size)) / D
                            < 2 + 4 / (D + 1))
                    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (
                        xmean - xold) / sigma
                    artmp = (pop[order] - xold) / sigma  # (mu, D)
                    C = ((1 - c1 - cmu) * C
                         + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                         + cmu * (artmp.T * weights) @ artmp)
                    sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
                    if nfes - eigeneval > pop_size / (c1 + cmu) / D / 10:
                        eigeneval = nfes
                        C = np.triu(C) + np.triu(C, 1).T
                        if not np.all(np.isfinite(C)) or not np.isfinite(sigma):
                            hybrid = False
                            continue
                        evals, B = np.linalg.eigh(C)
                        if np.any(evals <= 0):
                            # MATLAB: sqrt of a negative eigenvalue gives a complex
                            # D, so the next mutant is complex -> hybrid off.
                            hybrid = False
                            continue
                        Dd = np.sqrt(evals)
                        invsqrtC = (B / Dd) @ B.T

        return self._make_result(history_x, history_f, history_pop)
