"""L-SRTDE — Success Rate-based adaptive DE with LPSR (CEC 2024 winner).

Pure-numpy port of the author's C++ reference implementation
(V. Stanovov, https://github.com/VladimirStanovov/L-SRTDE_CEC-2024,
file ``L-SRTDE.cpp``), accompanying

    V. Stanovov, E. Semenkin, "Success Rate-based Adaptive Differential
    Evolution L-SRTDE for CEC 2024 Competition", IEEE CEC 2024.

The port follows the C++ code line by line, including its quirks (listed in
the class docstring), so that it behaves like the program that won the
competition rather than like a cleaned-up reading of the paper.
"""
from __future__ import annotations

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


def _sorted_indices(fit: np.ndarray) -> np.ndarray:
    """Ascending order; identity when all values are equal (as the C++ does)."""
    if fit.size == 0 or fit.min() == fit.max():
        return np.arange(fit.size)
    return np.argsort(fit, kind="stable")


class LSRTDEOptimizer(BaseOptimizer):
    """L-SRTDE (Stanovov & Semenkin, CEC 2024 winner), pure-numpy port.

    Algorithm (all values from ``L-SRTDE.cpp``):

    * Two populations: ``Popul`` (the archive-like "current" set: last front
      plus this generation's successful trials, sorted and truncated) and
      ``PopulFront`` (the working front of size NF).
    * NP_init = ``pop_per_dim`` · D = 20·D; linear reduction of the front to
      ``np_min`` = 4 over the evaluation budget (LPSR).
    * Success rate SR = (# successful trials) / NF of the previous generation.
    * F ~ N(meanF, 0.02) redrawn until in [0, 1], meanF = 0.4 + 0.25·tanh(5·SR).
    * Cr ~ N(M_Cr[r], 0.05) clipped to [0, 1]; memory size 5 initialised to 1.0;
      memory update M ← ½(meanWL(actual Cr of successes, |Δf|) + M).
    * Mutation r-current-to-pbest/1 with
      ``v = x_t + F(Popul[pbest] − x_t) + F(Front[r1] − Popul[r2])``, where the
      target t is drawn uniformly (not iterated), pbest is one of the top
      ``max(2, int(0.7·NF·exp(−7·SR)))`` of Popul, r1 is drawn from the front with
      rank weights exp(−3·i/NF), r2 uniform from Popul.
    * Binomial crossover; out-of-bounds coordinates are re-drawn uniformly.
    * A trial that is no worse than its target replaces front slot ``PFIndex``
      (cyclic pointer — *not* the target slot) and is appended to Popul.

    Faithfully reproduced quirks of the C++ code:
    * the successful trial overwrites the cyclic slot PFIndex, and the |Δf|
      weight is computed after that overwrite (0 when PFIndex == target);
    * the Popul end-of-generation sort covers ``NF_new + S`` entries although
      successes were stored from index ``NF_old`` on;
    * ``RemoveWorst`` keeps scanning the old front length after each removal;
    * ``pbest != target`` compares a Popul index with a front index.

    Deviations: the budget is checked per evaluation (the C++ only checks it
    between generations); bounds come from the benchmark instead of ±100;
    RNG is numpy's PCG64 seeded by ``seed`` instead of four mt19937 streams.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        pop_per_dim: int = 20,      # C++: PopSize = 20, NInds = PopSize·GNVars
        np_min: int = 4,            # C++: LPSR target 4
        memory_size: int = 5,       # C++: MemorySize = 5
        sigma_f: float = 0.02,      # C++: sigmaF
        sigma_cr: float = 0.05,     # C++: NormRand(MemoryCr, 0.05)
    ):
        super().__init__(benchmark, seed)
        self.pop_size = max(int(pop_per_dim * benchmark.dim), np_min)
        self.np_min = np_min
        self.memory_size = memory_size
        self.sigma_f = sigma_f
        self.sigma_cr = sigma_cr

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = map(float, self.bounds)
        D = self.dim
        N0 = self.pop_size
        H = self.memory_size

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        def evaluate(x: np.ndarray) -> float:
            f = float(self.func(x))
            history_x.append(x.copy())
            history_f.append(f)
            return f

        popul = rng.uniform(lo, hi, (2 * N0, D))
        fit_arr = np.full(2 * N0, np.inf)
        front = np.empty((N0, D))
        fit_front = np.full(N0, np.inf)
        mem_cr = np.ones(H)
        mem_iter = 0
        success_rate = 0.5
        nf = N0
        n_cur = N0
        pf_index = 0

        # Initial evaluation of the first NF rows of Popul
        for i in range(nf):
            if len(history_f) >= max_evals:
                break
            fit_arr[i] = evaluate(popul[i])
        if len(history_f) < nf:  # budget smaller than the initial population
            return self._make_result(history_x, history_f, None)
        order = _sorted_indices(fit_arr[:nf])
        front[:nf] = popul[order]
        fit_front[:nf] = fit_arr[order]
        history_pop.append(front[:nf].copy())

        while len(history_f) < max_evals:
            mean_f = 0.4 + np.tanh(success_rate * 5.0) * 0.25
            indices = _sorted_indices(fit_arr[:nf])          # into Popul
            indices2 = _sorted_indices(fit_front[:nf])       # into front
            rank_cdf = np.cumsum(np.exp(-np.arange(nf) / nf * 3.0))
            psize = max(2, int(nf * 0.7 * np.exp(-success_rate * 7.0)))

            s_cr: list[float] = []
            s_df: list[float] = []
            for _ in range(nf):
                if len(history_f) >= max_evals:
                    break
                t = int(rng.integers(nf))
                r_mem = int(rng.integers(H))
                while True:
                    prand = int(indices[rng.integers(psize)])
                    if prand != t:
                        break
                while True:
                    u = rng.random() * rank_cdf[-1]
                    pick = min(int(np.searchsorted(rank_cdf, u, side="right")), nf - 1)
                    rand1 = int(indices2[pick])
                    if rand1 != prand:
                        break
                while True:
                    rand2 = int(indices[rng.integers(nf)])
                    if rand2 != prand and rand2 != rand1:
                        break
                while True:
                    F = rng.normal(mean_f, self.sigma_f)
                    if 0.0 <= F <= 1.0:
                        break
                cr = min(max(rng.normal(mem_cr[r_mem], self.sigma_cr), 0.0), 1.0)

                xt = front[t]
                mask = rng.random(D) < cr
                mask[int(rng.integers(D))] = True
                v = xt + F * (popul[prand] - xt) + F * (front[rand1] - popul[rand2])
                out = mask & ((v < lo) | (v > hi))
                if out.any():
                    v[out] = rng.uniform(lo, hi, int(out.sum()))
                trial = np.where(mask, v, xt)
                actual_cr = float(mask.sum()) / D

                f_trial = evaluate(trial)
                if f_trial <= fit_front[t]:
                    k = n_cur + len(s_cr)
                    popul[k] = trial
                    fit_arr[k] = f_trial
                    front[pf_index] = trial
                    fit_front[pf_index] = f_trial
                    s_cr.append(actual_cr)
                    s_df.append(abs(fit_front[t] - f_trial))
                    pf_index = (pf_index + 1) % nf

            n_succ = len(s_cr)
            success_rate = n_succ / nf
            new_nf = int((self.np_min - N0) / max_evals * len(history_f) + N0)
            new_nf = max(new_nf, self.np_min)
            # RemoveWorst (C++ quirk: keeps scanning the old length nf)
            for _ in range(nf - new_nf):
                worst = int(np.argmax(fit_front[:nf]))
                front[worst:nf - 1] = front[worst + 1:nf].copy()
                fit_front[worst:nf - 1] = fit_front[worst + 1:nf].copy()
            nf = new_nf
            # Cr memory update (weighted Lehmer mean, weights |Δf|)
            if n_succ:
                sc = np.asarray(s_cr)
                wd = np.asarray(s_df)
                sw = wd.sum()
                if sw > 0:
                    ww = wd / sw
                    num, den = float(np.sum(ww * sc * sc)), float(np.sum(ww * sc))
                else:  # C++ divides 0/0 → NaN weights → falls back to 1.0
                    num, den = 0.0, 0.0
                mean_wl = num / den if abs(den) > 1e-8 else 1.0
                mem_cr[mem_iter] = 0.5 * (mean_wl + mem_cr[mem_iter])
                mem_iter = (mem_iter + 1) % H
            n_cur = nf + n_succ
            if n_cur > nf:
                order = _sorted_indices(fit_arr[:n_cur])
                fit_arr[:n_cur] = fit_arr[order]
                popul[:n_cur] = popul[order]
                n_cur = nf
            history_pop.append(front[:nf].copy())

        return self._make_result(history_x, history_f, history_pop)
